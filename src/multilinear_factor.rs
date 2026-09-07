// Adapted from cc4s CTF interface/multilinear.cxx factor remapping.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Physical-mode factor redistribution and complementary-fiber broadcast.
use crate::{
    algebra::{Monoid, Wire},
    mapping::{Distribution, Mapping},
    tensor::Tensor,
};

// Multilinear factors follow the tensor's physical mode, not its virtual blocks.
pub(crate) fn physical_mapping(mapping: &Mapping) -> Mapping {
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
pub(crate) fn aligned_factor<A: Monoid + Clone>(
    tensor_distribution: &Distribution,
    mode: usize,
    factor: &Tensor<'_, '_, A>,
    auxiliary_start: usize,
    auxiliary_width: Option<usize>,
    aux_mode_first: bool,
) -> (Distribution, Vec<A::Element>)
where
    A::Element: Wire,
{
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
    let mut values = vec![factor.algebra().zero(); mapped.local_len()];
    for ((offset, _), value) in requests.into_iter().zip(read) {
        values[offset] = value;
    }
    fiber.broadcast(0, &mut values);
    fiber.close();
    (mapped, values)
}
