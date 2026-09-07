// Adapted from cc4s CTF interface/multilinear.cxx and interface/semiring.h.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Sparse TTTP and MTTKRP over stored COO keys.
use crate::{
    algebra::Arithmetic,
    mapping::{Distribution, Mapping},
    sparse::SparseTensor,
    tensor::Tensor,
};

use crate::multilinear::factor_alignment::{aligned_factor, physical_mapping};

impl<'c, 'r> SparseTensor<'c, 'r, Arithmetic<f64>> {
    /// Multiply stored entries by a product of mode vectors. Factors are
    /// ordered by strictly increasing mode; omitted modes contribute no factor.
    pub fn tttp_vectors(
        &mut self,
        factors: &[(usize, &Tensor<'_, '_, Arithmetic<f64>>)],
    ) {
        assert!(!factors.is_empty());
        let distribution = self.distribution().clone();
        let rank = self.context().rank();
        let mut aligned = Vec::with_capacity(factors.len());
        for (index, &(mode, factor)) in factors.iter().enumerate() {
            assert!(std::ptr::eq(self.context(), factor.context()));
            assert!(mode < distribution.shape.len());
            assert!(index == 0 || factors[index - 1].0 < mode);
            assert_eq!(factor.distribution().shape, vec![distribution.shape[mode]]);
            let (mapped, values) = aligned_factor(&distribution, mode, factor, 0, None, false);
            aligned.push((mode, mapped, values));
        }
        self.transform_stored(|key, value| {
            let coordinates = distribution.decode_key(key);
            for (mode, mapped, vector) in &aligned {
                let factor_key = coordinates[*mode];
                *value *= vector[mapped.local_offset(rank, factor_key)];
            }
        });
    }

    /// Multiply stored entries by `sum_k product_mode M_mode[coordinate,k]`.
    /// `aux_mode_first` selects `[k, coordinate]` factor storage. `divisions`
    /// selects balanced auxiliary-index blocks without densifying the tensor.
    pub fn tttp_matrices(
        &mut self,
        factors: &[(usize, &Tensor<'_, '_, Arithmetic<f64>>)],
        aux_mode_first: bool,
        divisions: usize,
    ) {
        assert!(!factors.is_empty());
        let distribution = self.distribution().clone();
        let mode_axis = usize::from(aux_mode_first);
        let auxiliary_axis = 1 - mode_axis;
        assert_eq!(factors[0].1.distribution().shape.len(), 2);
        let auxiliary_length = factors[0].1.distribution().shape[auxiliary_axis];
        assert!(divisions > 0 && divisions <= auxiliary_length);
        for (index, &(mode, factor)) in factors.iter().enumerate() {
            assert!(std::ptr::eq(self.context(), factor.context()));
            assert!(mode < distribution.shape.len());
            assert!(index == 0 || factors[index - 1].0 < mode);
            assert_eq!(factor.distribution().shape.len(), 2);
            assert_eq!(factor.distribution().shape[mode_axis], distribution.shape[mode]);
            assert_eq!(factor.distribution().shape[auxiliary_axis], auxiliary_length);
        }

        let rank = self.context().rank();
        let mut accumulated = if divisions > 1 {
            vec![0.; self.local_nnz()]
        } else {
            Vec::new()
        };
        let mut auxiliary_start = 0;
        for block in 0..divisions {
            let width = auxiliary_length / divisions
                + usize::from(block < auxiliary_length % divisions);
            let mut aligned = Vec::with_capacity(factors.len());
            for &(mode, factor) in factors {
                let (mapped, values) = aligned_factor(
                    &distribution,
                    mode,
                    factor,
                    auxiliary_start,
                    Some(width),
                    aux_mode_first,
                );
                aligned.push((mode, mapped, values));
            }

            let mut entry = 0;
            self.transform_stored(|key, value| {
                let coordinates = distribution.decode_key(key);
                let mut sum = 0.;
                for auxiliary in 0..width {
                    let mut product = 1.;
                    for (mode, mapped, matrix) in &aligned {
                        let factor_coordinates = if aux_mode_first {
                            [auxiliary, coordinates[*mode]]
                        } else {
                            [coordinates[*mode], auxiliary]
                        };
                        let factor_key = mapped.encode_key(&factor_coordinates);
                        product *= matrix[mapped.local_offset(rank, factor_key)];
                    }
                    sum += product;
                }
                if divisions == 1 {
                    *value *= sum;
                } else {
                    accumulated[entry] += sum;
                }
                entry += 1;
            });
            auxiliary_start += width;
        }
        if divisions > 1 {
            let mut entry = 0;
            self.transform_stored(|_, value| {
                *value *= accumulated[entry];
                entry += 1;
            });
        }
    }

    /// Matricized sparse tensor times a Khatri-Rao product. Supply all factors
    /// except `output_mode`, in ascending tensor-mode order. Factors are vectors
    /// or auxiliary-first matrices `[k, mode_length]`; output has the matching
    /// vector or matrix shape.
    pub fn mttkrp(
        &self,
        output_mode: usize,
        factors: &[&Tensor<'_, '_, Arithmetic<f64>>],
        output_distribution: Distribution,
    ) -> Tensor<'c, 'r, Arithmetic<f64>> {
        let distribution = self.distribution();
        let order = distribution.shape.len();
        assert!(order >= 2 && output_mode < order);
        assert_eq!(factors.len(), order - 1);
        let vector = factors[0].distribution().shape.len() == 1;
        let width = if vector {
            1
        } else {
            factors[0].distribution().shape[0]
        };
        let shape_for = |mode: usize| {
            if vector {
                vec![distribution.shape[mode]]
            } else {
                vec![width, distribution.shape[mode]]
            }
        };
        assert_eq!(output_distribution.shape, shape_for(output_mode));
        assert_eq!(output_distribution.topology.size(), self.context().size());

        let rank = self.context().rank();
        let mut aligned: Vec<Option<(Distribution, Vec<f64>)>> =
            (0..order).map(|_| None).collect();
        let mut factor_index = 0;
        for mode in 0..order {
            if mode == output_mode {
                continue;
            }
            let factor = factors[factor_index];
            factor_index += 1;
            assert!(std::ptr::eq(self.context(), factor.context()));
            assert_eq!(factor.distribution().shape, shape_for(mode));
            let values = if vector {
                aligned_factor(distribution, mode, factor, 0, None, false)
            } else {
                aligned_factor(distribution, mode, factor, 0, Some(width), true)
            };
            aligned[mode] = Some(values);
        }

        let output_mapping = physical_mapping(&distribution.mappings[output_mode]);
        let output_mappings = if vector {
            vec![output_mapping]
        } else {
            vec![Mapping::Unmapped, output_mapping]
        };
        let mapped_output = Distribution::new(
            shape_for(output_mode),
            distribution.topology.clone(),
            output_mappings,
        );
        let mut output_values = vec![0.; mapped_output.local_len()];
        // Source MTTKRP groups contiguous mode-zero fibers, reusing products
        // of the remaining factor rows. Virtual blocks require global-key order.
        let mut pairs = self.local_pairs();
        pairs.retain(|(key, _)| distribution.owner(*key) == rank);
        pairs.sort_by_key(|&(key, _)| key);
        let phases: Vec<_> = distribution.mappings.iter().map(Mapping::physical_phase).collect();
        let arrays: Vec<&[f64]> = aligned.iter().map(|factor| match factor {
            Some((_, values)) => values.as_slice(),
            None => &[],
        }).collect();
        crate::multilinear::kernel::mttkrp(&distribution.shape, &phases, width,
            output_mode, &pairs, &arrays, &mut output_values);

        let coordinates = distribution.topology.coordinates(rank);
        let color = distribution.mappings[output_mode].physical_rank(&coordinates);
        let fiber = self
            .context()
            .split(Some(color as i32), rank as i32)
            .unwrap();
        fiber.reduce_f64(0, &mut output_values);
        let pairs: Vec<_> = if fiber.rank() == 0 {
            output_values
                .into_iter()
                .enumerate()
                .filter_map(|(offset, value)| {
                    mapped_output.global_key(rank, offset).map(|key| (key, value))
                })
                .collect()
        } else {
            Vec::new()
        };
        fiber.close();

        let mut output = Tensor::new(self.context(), output_distribution, Arithmetic::new());
        output.write_add(&pairs);
        output
    }
}
