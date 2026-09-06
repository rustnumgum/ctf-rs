// Adapted from cc4s CTF interface/multilinear.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Dense TTTP: mode-aligned factors and balanced auxiliary-index blocking.
//! Factor redistribution currently uses the tensor all-to-all implementation;
//! the source's specialized redistribution-plus-fiber-broadcast is pending.
use crate::{algebra::Arithmetic, mapping::{Distribution, Mapping}, tensor::Tensor};

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
