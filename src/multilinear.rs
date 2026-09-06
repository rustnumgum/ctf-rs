// Adapted from cc4s CTF interface/multilinear.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Dense TTTP: mode-aligned factors and balanced auxiliary-index blocking.
//! Factor redistribution currently uses the tensor all-to-all implementation;
//! the source's specialized redistribution-plus-fiber-broadcast is pending.
use crate::{algebra::Arithmetic, mapping::{Distribution, Mapping}, tensor::Tensor};

#[path = "multilinear_kernel.rs"]
mod kernel;

#[path = "reshape.rs"]
mod reshape;
#[cfg(feature = "native-scalapack")]
#[path = "tensor_svd.rs"]
pub mod tensor_svd;

// TTTP factors follow the tensor's physical mode, not its virtual blocks.
fn physical_mapping(mapping: &Mapping) -> Mapping {
    match mapping {
        Mapping::Unmapped | Mapping::Virtual { .. } => Mapping::Unmapped,
        Mapping::Physical { axis, processes, child } => {
            assert_eq!(child.physical_phase(), 1, "TTTP requires one physical axis per mode");
            Mapping::Physical { axis: *axis, processes: *processes,
                child: Box::new(Mapping::Unmapped) }
        }
    }
}

impl<'c, 'r> Tensor<'c, 'r, Arithmetic<f64>> {
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
            let mut values = vec![0.; mapped.local_len()];
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
        kernel::mttkrp(&dist.shape, &phases, width, output_mode, &pairs, &factors, &mut values);
        let mapped = output_mapped.unwrap();
        let color = dist.mappings[output_mode].physical_rank(&coordinates);
        let fiber = self.context().split(Some(color as i32), rank as i32).unwrap();
        fiber.reduce_f64(0, &mut values);
        let output_pairs: Vec<_> = if fiber.rank() == 0 {
            values.into_iter().enumerate().filter_map(|(offset, value)|
                mapped.global_key(rank, offset).map(|key| (key, value))).collect()
        } else { Vec::new() };
        fiber.close();
        let mut output = Self::new(self.context(), output_distribution, Arithmetic::new());
        output.write_add(&output_pairs);
        output
    }

    /// Multiply entries by a product of mode vectors. Factors are ordered by
    /// strictly increasing mode; omitted modes contribute no factor.
    pub fn tttp_vectors(&mut self, factors: &[(usize, &Self)]) {
        assert!(!factors.is_empty());
        let distribution = self.distribution().clone();
        let rank = self.context().rank();
        let mut mapped = Vec::with_capacity(factors.len());
        for (index, &(mode, factor)) in factors.iter().enumerate() {
            assert!(std::ptr::eq(self.context(), factor.context()));
            assert!(mode < distribution.shape.len());
            assert!(index == 0 || factors[index - 1].0 < mode);
            assert_eq!(factor.distribution().shape, vec![distribution.shape[mode]]);
            let stride: usize = distribution.shape[..mode].iter().product();
            let mut vector = factor.clone();
            vector.redistribute(Distribution::new(factor.distribution().shape.clone(),
                distribution.topology.clone(), vec![physical_mapping(&distribution.mappings[mode])]));
            mapped.push((mode, stride, vector));
        }
        self.transform(|key, value| {
            for (mode, stride, vector) in &mapped {
                let coordinate = key / stride % distribution.shape[*mode];
                let offset = vector.distribution().local_offset(rank, coordinate);
                *value *= vector.local_storage()[offset];
            }
        });
    }

    /// Multiply entries by sum_k product_mode M_mode[coordinate,k].
    /// `aux_mode_first` selects [k,coordinate] factor storage. `divisions`
    /// explicitly selects the source's balanced k-blocks (1 through k), bounding
    /// resident redistributed factor storage without gathering the tensor.
    /// Automatic available-memory selection is not implemented here.
    pub fn tttp_matrices(&mut self, factors: &[(usize, &Self)],
        aux_mode_first: bool, divisions: usize) {
        assert!(!factors.is_empty());
        let distribution = self.distribution().clone();
        let mode_axis = usize::from(aux_mode_first);
        let auxiliary_axis = 1 - mode_axis;
        assert_eq!(factors[0].1.distribution().shape.len(), 2);
        let k = factors[0].1.distribution().shape[auxiliary_axis];
        assert!(divisions > 0 && divisions <= k);
        for (index, &(mode, factor)) in factors.iter().enumerate() {
            assert!(std::ptr::eq(self.context(), factor.context()));
            assert!(mode < distribution.shape.len());
            assert!(index == 0 || factors[index - 1].0 < mode);
            assert_eq!(factor.distribution().shape.len(), 2);
            assert_eq!(factor.distribution().shape[mode_axis], distribution.shape[mode]);
            assert_eq!(factor.distribution().shape[auxiliary_axis], k);
        }
        let rank = self.context().rank();
        let mut accumulated = if divisions > 1 {
            vec![0.; self.local_storage().len()]
        } else { Vec::new() };
        let mut start = 0;
        for block in 0..divisions {
            let width = k / divisions + usize::from(block < k % divisions);
            let mut mapped = Vec::with_capacity(factors.len());
            for &(mode, factor) in factors {
                let mut ranges: Vec<_> = factor.distribution().shape.iter().map(|&n| 0..n).collect();
                ranges[auxiliary_axis] = start..start + width;
                let mut matrix = if divisions == 1 { factor.clone() } else { factor.slice(&ranges) };
                let mut mappings = vec![Mapping::Unmapped; 2];
                mappings[mode_axis] = physical_mapping(&distribution.mappings[mode]);
                matrix.redistribute(Distribution::new(matrix.distribution().shape.clone(),
                    distribution.topology.clone(), mappings));
                let stride: usize = distribution.shape[..mode].iter().product();
                mapped.push((mode, stride, matrix));
            }
            self.transform(|key, value| {
                let mut sum = 0.;
                for auxiliary in 0..width {
                    let mut product = 1.;
                    for (mode, stride, matrix) in &mapped {
                        let coordinate = key / stride % distribution.shape[*mode];
                        let matrix_key = if aux_mode_first { auxiliary + width * coordinate }
                            else { coordinate + distribution.shape[*mode] * auxiliary };
                        product *= matrix.local_storage()[matrix.distribution().local_offset(rank, matrix_key)];
                    }
                    sum += product;
                }
                if divisions == 1 { *value *= sum; }
                else { accumulated[distribution.local_offset(rank, key)] += sum; }
            });
            start += width;
        }
        if divisions > 1 {
            self.transform(|key, value| *value *= accumulated[distribution.local_offset(rank, key)]);
        }
    }
}
