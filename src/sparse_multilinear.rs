// Adapted from cc4s CTF interface/multilinear.cxx and interface/semiring.h.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Sparse TTTP and MTTKRP over stored COO keys.
use crate::{
    algebra::Arithmetic,
    mapping::{Distribution, Mapping},
    sparse::SparseTensor,
    tensor::Tensor,
};

// Multilinear factors follow the tensor's physical mode, not its virtual blocks.
fn physical_mapping(mapping: &Mapping) -> Mapping {
    match mapping {
        Mapping::Unmapped | Mapping::Virtual { .. } => Mapping::Unmapped,
        Mapping::Physical {
            axis,
            processes,
            child,
        } => {
            assert_eq!(
                child.physical_phase(),
                1,
                "multilinear operations require one physical axis per mode"
            );
            Mapping::Physical {
                axis: *axis,
                processes: *processes,
                child: Box::new(Mapping::Unmapped),
            }
        }
    }
}

/// Read one vector or auxiliary submatrix onto the tensor mode's physical
/// mapping, then broadcast it along the complementary process fiber.
fn aligned_factor(
    tensor_distribution: &Distribution,
    mode: usize,
    factor: &Tensor<'_, '_, Arithmetic<f64>>,
    auxiliary_start: usize,
    auxiliary_width: Option<usize>,
    aux_mode_first: bool,
) -> (Distribution, Vec<f64>) {
    let mapping = physical_mapping(&tensor_distribution.mappings[mode]);
    let (shape, mappings) = if let Some(width) = auxiliary_width {
        if aux_mode_first {
            (
                vec![width, tensor_distribution.shape[mode]],
                vec![Mapping::Unmapped, mapping],
            )
        } else {
            (
                vec![tensor_distribution.shape[mode], width],
                vec![mapping, Mapping::Unmapped],
            )
        }
    } else {
        (
            vec![tensor_distribution.shape[mode]],
            vec![mapping],
        )
    };
    let mapped = Distribution::new(shape, tensor_distribution.topology.clone(), mappings);
    let rank = factor.context().rank();
    let coordinates = tensor_distribution.topology.coordinates(rank);
    let color = tensor_distribution.mappings[mode].physical_rank(&coordinates);
    let fiber = factor
        .context()
        .split(Some(color as i32), rank as i32)
        .unwrap();
    let requests: Vec<_> = if fiber.rank() == 0 {
        (0..mapped.local_len())
            .filter_map(|offset| {
                mapped.global_key(rank, offset).map(|key| {
                    let mut coordinates = mapped.decode_key(key);
                    if auxiliary_width.is_some() {
                        let auxiliary_axis = usize::from(!aux_mode_first);
                        coordinates[auxiliary_axis] += auxiliary_start;
                    }
                    (offset, factor.distribution().encode_key(&coordinates))
                })
            })
            .collect()
    } else {
        Vec::new()
    };
    let keys: Vec<_> = requests.iter().map(|&(_, key)| key).collect();
    let read = factor.read(&keys);
    let mut values = vec![0.; mapped.local_len()];
    for ((offset, _), value) in requests.into_iter().zip(read) {
        values[offset] = value;
    }
    fiber.broadcast(0, &mut values);
    fiber.close();
    (mapped, values)
}

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
        for (key, value) in self.local_pairs() {
            if distribution.owner(key) != rank {
                continue;
            }
            let coordinates = distribution.decode_key(key);
            for auxiliary in 0..width {
                let mut contribution = value;
                for mode in 0..order {
                    if mode == output_mode {
                        continue;
                    }
                    let (mapped, factor) = aligned[mode].as_ref().unwrap();
                    let factor_key = if vector {
                        coordinates[mode]
                    } else {
                        auxiliary + width * coordinates[mode]
                    };
                    contribution *= factor[mapped.local_offset(rank, factor_key)];
                }
                let output_key = if vector {
                    coordinates[output_mode]
                } else {
                    auxiliary + width * coordinates[output_mode]
                };
                output_values[mapped_output.local_offset(rank, output_key)] += contribution;
            }
        }

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
