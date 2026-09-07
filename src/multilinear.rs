// Adapted from cc4s CTF interface/multilinear.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Dense TTTP: mode-aligned factors and balanced auxiliary-index blocking.
use crate::{
    algebra::{Semiring, Wire},
    mapping::{Distribution, Mapping},
    tensor::Tensor,
};

#[path = "multilinear_factor.rs"]
pub(crate) mod factor_alignment;
use factor_alignment::{aligned_factor, physical_mapping};

#[path = "multilinear_kernel.rs"]
pub(crate) mod kernel;

#[path = "tttp_blocking.rs"]
pub(crate) mod tttp_blocking;
pub use tttp_blocking::TttpBlocking;

#[path = "reshape.rs"]
mod reshape;
#[cfg(feature = "native-scalapack")]
#[path = "tensor_svd.rs"]
pub mod tensor_svd;

#[cfg(feature = "native-linalg")]
#[path = "solve_factor.rs"]
mod solve_factor;

impl<'c, 'r, A: Semiring + Clone> Tensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Matricized tensor times Khatri-Rao product, replacing the output factor.
    /// Supply all factors except `output_mode`, in ascending tensor-mode order.
    /// Factors are vectors or auxiliary-first matrices [k, mode_length], as in
    /// the pinned native MTTKRP kernel. Output shape is [mode_length] or [k, mode_length].
    pub fn mttkrp(&self, output_mode: usize, factors: &[&Self],
        output_distribution: Distribution) -> Self {
        let dist = self.distribution();
        let order = dist.shape.len();
        assert!(order >= 2 && output_mode < order);
        assert_eq!(factors.len(), order - 1);
        let vector = factors[0].distribution().shape.len() == 1;
        let width = if vector { 1 } else { factors[0].distribution().shape[0] };
        let shape_for = |mode: usize| if vector { vec![dist.shape[mode]] }
            else { vec![width, dist.shape[mode]] };
        assert_eq!(output_distribution.shape, shape_for(output_mode));
        assert_eq!(output_distribution.topology.size(), self.context().size());
        let algebra = self.algebra().clone();
        let rank = self.context().rank();
        let coordinates = dist.topology.coordinates(rank);
        let phases: Vec<_> = dist.mappings.iter().map(Mapping::physical_phase).collect();
        let mut arrays = Vec::with_capacity(order);
        let mut output_mapped = None;
        let mut factor_index = 0;
        for mode in 0..order {
            let mapping = physical_mapping(&dist.mappings[mode]);
            let color = mapping.physical_rank(&coordinates);
            let mappings = if vector { vec![mapping] } else { vec![Mapping::Unmapped, mapping] };
            let mapped = Distribution::new(shape_for(mode), dist.topology.clone(), mappings);
            let mut values = vec![algebra.zero(); mapped.local_len()];
            if mode == output_mode {
                output_mapped = Some(mapped);
            } else {
                let factor = factors[factor_index];
                factor_index += 1;
                assert!(std::ptr::eq(self.context(), factor.context()));
                assert_eq!(factor.distribution().shape, shape_for(mode));
                // Redistribute only to the canonical rank of each physical
                // mode shard, then broadcast along its complementary fiber.
                let fiber = self.context().split(Some(color as i32), rank as i32).unwrap();
                let keys: Vec<_> = if fiber.rank() == 0 {
                    (0..values.len()).filter_map(|offset| mapped.global_key(rank, offset)).collect()
                } else { Vec::new() };
                for (key, value) in keys.iter().zip(factor.read(&keys)) {
                    values[mapped.local_offset(rank, *key)] = value;
                }
                fiber.broadcast(0, &mut values);
                fiber.close();
            }
            arrays.push(values);
        }
        // Virtual blocks need key ordering before the source fiber grouping;
        // replicated tensor layers must not contribute duplicate reductions.
        let mut pairs = self.local_pairs();
        pairs.retain(|(key, _)| dist.owner(*key) == rank);
        pairs.sort_by_key(|&(key, _)| key);
        let mut values = std::mem::take(&mut arrays[output_mode]);
        let factors: Vec<_> = arrays.iter().map(Vec::as_slice).collect();
        kernel::mttkrp(
            &algebra,
            &dist.shape,
            &phases,
            width,
            output_mode,
            &pairs,
            &factors,
            &mut values,
        );
        let mapped = output_mapped.unwrap();
        let color = dist.mappings[output_mode].physical_rank(&coordinates);
        let fiber = self.context().split(Some(color as i32), rank as i32).unwrap();
        fiber.reduce_monoid(&algebra, &mut values, false, 0);
        let output_pairs: Vec<_> = if fiber.rank() == 0 {
            values.into_iter().enumerate().filter_map(|(offset, value)|
                mapped.global_key(rank, offset).map(|key| (key, value))).collect()
        } else { Vec::new() };
        fiber.close();
        let mut output = Self::new(self.context(), output_distribution, algebra);
        output.write_add(&output_pairs);
        output
    }

    /// Multiply entries by a product of mode vectors. Factors are ordered by
    /// strictly increasing mode; omitted modes contribute no factor.
    pub fn tttp_vectors(&mut self, factors: &[(usize, &Self)]) {
        assert!(!factors.is_empty());
        let algebra = self.algebra().clone();
        let distribution = self.distribution().clone();
        let rank = self.context().rank();
        let mut mapped = Vec::with_capacity(factors.len());
        for (index, &(mode, factor)) in factors.iter().enumerate() {
            assert!(std::ptr::eq(self.context(), factor.context()));
            assert!(mode < distribution.shape.len());
            assert!(index == 0 || factors[index - 1].0 < mode);
            assert_eq!(factor.distribution().shape, vec![distribution.shape[mode]]);
            let (factor_distribution, values) =
                aligned_factor(&distribution, mode, factor, 0, None, false);
            mapped.push((mode, factor_distribution, values));
        }
        self.transform(|key, value| {
            let coordinates = distribution.decode_key(key);
            for (mode, factor_distribution, vector) in &mapped {
                let factor_key = coordinates[*mode];
                *value = algebra.multiply(
                    value,
                    &vector[factor_distribution.local_offset(rank, factor_key)],
                );
            }
        });
    }

    /// Multiply entries by sum_k product_mode M_mode[coordinate,k].
    /// `aux_mode_first` selects [k,coordinate] factor storage. Blocking either
    /// supplies the source's balanced k-block count or a per-rank available-byte
    /// fact; the latter doubles locally and collectively selects the maximum.
    /// No operating-system memory probe is performed.
    pub fn tttp_matrices(&mut self, factors: &[(usize, &Self)],
        aux_mode_first: bool, blocking: TttpBlocking) {
        assert!(!factors.is_empty());
        let algebra = self.algebra().clone();
        let distribution = self.distribution().clone();
        let mode_axis = usize::from(aux_mode_first);
        let auxiliary_axis = 1 - mode_axis;
        assert_eq!(factors[0].1.distribution().shape.len(), 2);
        let k = factors[0].1.distribution().shape[auxiliary_axis];
        for (index, &(mode, factor)) in factors.iter().enumerate() {
            assert!(std::ptr::eq(self.context(), factor.context()));
            assert!(mode < distribution.shape.len());
            assert!(index == 0 || factors[index - 1].0 < mode);
            assert_eq!(factor.distribution().shape.len(), 2);
            assert_eq!(factor.distribution().shape[mode_axis], distribution.shape[mode]);
            assert_eq!(factor.distribution().shape[auxiliary_axis], k);
        }
        let rank = self.context().rank();
        let modes: Vec<_> = factors.iter().map(|(mode, _)| *mode).collect();
        let local_pairs = (0..self.local_storage().len())
            .filter(|&offset| distribution.global_key(rank, offset).is_some())
            .count();
        let divisions = tttp_blocking::resolve(
            self.context(),
            &distribution,
            &modes,
            k,
            local_pairs,
            std::mem::size_of::<A::Element>(),
            blocking,
        );
        let mut accumulated = if divisions > 1 {
            vec![algebra.zero(); local_pairs]
        } else { Vec::new() };
        let mut start = 0;
        for block in 0..divisions {
            let width = k / divisions + usize::from(block < k % divisions);
            let mut mapped = Vec::with_capacity(factors.len());
            for &(mode, factor) in factors {
                let (factor_distribution, values) = aligned_factor(
                    &distribution, mode, factor, start, Some(width), aux_mode_first);
                mapped.push((mode, factor_distribution, values));
            }
            let mut entry = 0;
            self.transform(|key, value| {
                let coordinates = distribution.decode_key(key);
                let mut sum = algebra.zero();
                for auxiliary in 0..width {
                    let mut product = algebra.one();
                    for (mode, factor_distribution, matrix) in &mapped {
                        let factor_coordinates = if aux_mode_first {
                            [auxiliary, coordinates[*mode]]
                        } else {
                            [coordinates[*mode], auxiliary]
                        };
                        let factor_key = factor_distribution.encode_key(&factor_coordinates);
                        product = algebra.multiply(
                            &product,
                            &matrix[factor_distribution.local_offset(rank, factor_key)],
                        );
                    }
                    sum = algebra.add(&sum, &product);
                }
                if divisions == 1 {
                    *value = algebra.multiply(value, &sum);
                } else {
                    accumulated[entry] = algebra.add(&accumulated[entry], &sum);
                }
                entry += 1;
            });
            start += width;
        }
        if divisions > 1 {
            let mut entry = 0;
            self.transform(|_, value| {
                *value = algebra.multiply(value, &accumulated[entry]);
                entry += 1;
            });
        }
    }
}
