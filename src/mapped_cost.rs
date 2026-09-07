// Adapted from cc4s contraction/{contraction,ctr_comm,ctr_tsr,
// ctr_2d_general}.cxx. Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Dense unfolded cost-tree construction for preflight-valid raw mappings.

use crate::{
    mapping::{Distribution, Mapping},
    plan_cost::{Collective, Tree},
};

#[derive(Clone)]
struct OperandState {
    block_size: usize,
    block_lengths: Vec<usize>,
    virtual_block_lengths: Vec<usize>,
}

struct Panel {
    steps: usize,
    panel_bytes: [usize; 3],
    movement: [Option<Collective>; 3],
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

fn head_physical(mapping: &Mapping) -> Option<(usize, usize)> {
    if let Mapping::Physical {
        axis, processes, ..
    } = mapping
    {
        Some((*axis, *processes))
    } else {
        None
    }
}

fn terminal_virtual(mapping: &Mapping) -> usize {
    match mapping {
        Mapping::Physical { child, .. } | Mapping::Virtual { child, .. } => {
            if matches!(**child, Mapping::Unmapped) {
                if let Mapping::Virtual { copies, .. } = mapping {
                    *copies
                } else {
                    1
                }
            } else {
                terminal_virtual(child)
            }
        }
        Mapping::Unmapped => 1,
    }
}

// contraction's comp_dim_map, including its directional NOT_MAPPED/VIRTUAL(1)
// convention. Valid generated maps have only terminal virtual children.
fn source_equal(left: &Mapping, right: &Mapping) -> bool {
    match (left, right) {
        (Mapping::Unmapped, Mapping::Unmapped) => true,
        (Mapping::Unmapped, Mapping::Virtual { copies: 1, .. }) => true,
        (Mapping::Unmapped, _) | (_, Mapping::Unmapped) => false,
        (
            Mapping::Physical {
                axis: left_axis,
                processes: left_processes,
                child: left_child,
            },
            Mapping::Physical {
                axis: right_axis,
                processes: right_processes,
                child: right_child,
            },
        ) => {
            left_axis == right_axis
                && left_processes == right_processes
                && match (
                    matches!(**left_child, Mapping::Unmapped),
                    matches!(**right_child, Mapping::Unmapped),
                ) {
                    (true, true) => true,
                    (false, false) => source_equal(left_child, right_child),
                    _ => false,
                }
        }
        (
            Mapping::Virtual {
                copies: left_copies,
                ..
            },
            Mapping::Virtual {
                copies: right_copies,
                ..
            },
        ) => left_copies == right_copies,
        _ => false,
    }
}

fn mark_physical(mapping: &Mapping, used: &mut [bool]) {
    match mapping {
        Mapping::Unmapped => {}
        Mapping::Physical { axis, child, .. } => {
            used[*axis] = true;
            mark_physical(child, used);
        }
        Mapping::Virtual { child, .. } => mark_physical(child, used),
    }
}

fn panel_size(
    state: &OperandState,
    dimension: usize,
    mapping: &Mapping,
    edge_length: usize,
) -> usize {
    let mut sub = if let Some((_, processes)) = head_physical(mapping) {
        state.block_size * processes / edge_length
    } else {
        state.block_size / edge_length
    };
    let mut leading = 1;
    for axis in dimension + 1..state.block_lengths.len() {
        sub = sub * state.virtual_block_lengths[axis] / state.block_lengths[axis];
        leading = leading * state.block_lengths[axis] / state.virtual_block_lengths[axis];
    }
    sub * leading
}

fn update_panel_state(
    state: &mut OperandState,
    dimension: usize,
    mapping: &Mapping,
    steps: usize,
) {
    if let Some((_, processes)) = head_physical(mapping) {
        state.block_size = state.block_size * processes / steps;
        state.block_lengths[dimension] =
            state.block_lengths[dimension] * processes / steps;
    } else {
        state.block_size /= steps;
        state.block_lengths[dimension] /= steps;
    }
}

fn physical_child_virtual(mapping: &Mapping, steps: usize) -> Option<usize> {
    if let Mapping::Physical {
        processes, child, ..
    } = mapping
    {
        match &**child {
            Mapping::Unmapped => None,
            Mapping::Virtual { copies, child } => {
                assert!(matches!(**child, Mapping::Unmapped));
                Some(processes * copies / steps)
            }
            Mapping::Physical { .. } => {
                panic!("preflight-valid mismatched maps cannot have a folded physical child")
            }
        }
    } else {
        None
    }
}

fn panel_virtual(first: &Mapping, second: &Mapping, steps: usize) -> usize {
    let mut phase = 1;
    if let Some(value) = physical_child_virtual(first, steps) {
        phase = value;
    }
    if let Some(value) = physical_child_virtual(second, steps) {
        phase = value;
    }
    if let Mapping::Virtual { copies, .. } = second {
        phase = copies / steps;
    }
    if let Mapping::Virtual { copies, .. } = first {
        phase = copies / steps;
    }
    phase
}

/// Construct the dense, unfolded source execution tree. The mappings must have
/// passed the NS unique-label mapping preflight. Redistribution, folding, and
/// resident tensor storage are not part of the returned tree.
pub fn dense_unfolded(
    distributions: [&Distribution; 3],
    indices: [&str; 3],
    element_bytes: usize,
    nodes_per_axis: &[f64],
    local_custom: bool,
    custom_reduce: bool,
) -> Tree {
    let topology = &distributions[0].topology;
    assert!(distributions
        .iter()
        .all(|distribution| &distribution.topology == topology));
    assert_eq!(nodes_per_axis.len(), topology.dimensions.len());

    let mut labels = Vec::new();
    let mut lengths = Vec::new();
    let mut normalized: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::new());
    for operand in 0..3 {
        assert!(indices[operand].is_ascii());
        assert_eq!(indices[operand].len(), distributions[operand].shape.len());
        for (axis, label) in indices[operand].bytes().enumerate() {
            assert!(!indices[operand].as_bytes()[..axis].contains(&label));
            let id = if let Some(id) = labels.iter().position(|&old| old == label) {
                assert_eq!(lengths[id], distributions[operand].shape[axis]);
                id
            } else {
                labels.push(label);
                lengths.push(distributions[operand].shape[axis]);
                labels.len() - 1
            };
            normalized[operand].push(id);
        }
    }
    let mut inverse = vec![[None; 3]; labels.len()];
    for operand in 0..3 {
        for (axis, &label) in normalized[operand].iter().enumerate() {
            inverse[label][operand] = Some(axis);
        }
    }

    let virtual_blocks: [Vec<usize>; 3] = distributions.each_ref().map(|distribution| {
        distribution
            .mappings
            .iter()
            .map(|mapping| mapping.phase() / mapping.physical_phase())
            .collect()
    });
    let block_shapes = distributions.each_ref().map(|distribution| distribution.block_shape());
    let mut states: [OperandState; 3] = std::array::from_fn(|operand| OperandState {
        block_size: distributions[operand].local_len(),
        block_lengths: block_shapes[operand]
            .iter()
            .zip(&virtual_blocks[operand])
            .map(|(&length, &copies)| length * copies)
            .collect(),
        virtual_block_lengths: block_shapes[operand].clone(),
    });

    let mut physical: [Vec<bool>; 3] =
        std::array::from_fn(|_| vec![false; topology.dimensions.len()]);
    for operand in 0..3 {
        for mapping in &distributions[operand].mappings {
            mark_physical(mapping, &mut physical[operand]);
        }
    }
    let initial_local_bytes = distributions
        .each_ref()
        .map(|distribution| distribution.local_len() * element_bytes);
    let mut replicate_inputs: [Vec<Collective>; 2] = std::array::from_fn(|_| Vec::new());
    let mut replicate_output = Vec::new();
    for axis in 0..topology.dimensions.len() {
        if !(physical[0][axis] || physical[1][axis] || physical[2][axis]) {
            continue;
        }
        for operand in 0..2 {
            if !physical[operand][axis] {
                replicate_inputs[operand].push(Collective {
                    ranks: topology.dimensions[axis],
                    nodes: nodes_per_axis[axis],
                    bytes: initial_local_bytes[operand],
                });
            }
        }
        if !physical[2][axis] {
            replicate_output.push(Collective {
                ranks: topology.dimensions[axis],
                nodes: nodes_per_axis[axis],
                bytes: initial_local_bytes[2],
            });
        }
    }

    let mut phases = vec![1usize; labels.len()];
    let mut panels = Vec::new();
    for label in 0..labels.len() {
        let present: Vec<_> = (0..3)
            .filter_map(|operand| inverse[label][operand].map(|axis| (operand, axis)))
            .collect();
        if present.len() != 2 {
            let (operand, axis) = present[0];
            phases[label] = terminal_virtual(&distributions[operand].mappings[axis]);
            continue;
        }

        let order: [usize; 2] = match (present[0].0, present[1].0) {
            (1, 2) => [1, 2],
            (0, 2) => [2, 0],
            (0, 1) => [0, 1],
            _ => unreachable!(),
        };
        let dimensions = order.map(|operand| inverse[label][operand].unwrap());
        let first_map = &distributions[order[0]].mappings[dimensions[0]];
        let second_map = &distributions[order[1]].mappings[dimensions[1]];
        if source_equal(second_map, first_map) {
            phases[label] = terminal_virtual(first_map);
            continue;
        }
        if matches!(first_map, Mapping::Virtual { .. })
            && matches!(second_map, Mapping::Virtual { .. })
        {
            phases[label] = if let Mapping::Virtual { copies, .. } = first_map {
                *copies
            } else {
                unreachable!()
            };
            continue;
        }

        let mut edge_length = 1;
        let mut steps = 1;
        for (&operand, &dimension) in order.iter().zip(&dimensions) {
            let mapping = &distributions[operand].mappings[dimension];
            if head_physical(mapping).is_some() {
                edge_length = lcm(edge_length, mapping.phase());
                steps = steps.max(mapping.phase());
            }
        }
        let mut sizes = [0usize; 3];
        let mut movement: [Option<Collective>; 3] = std::array::from_fn(|_| None);
        for (&operand, &dimension) in order.iter().zip(&dimensions) {
            let mapping = &distributions[operand].mappings[dimension];
            sizes[operand] = panel_size(&states[operand], dimension, mapping, edge_length);
            if let Some((axis, _)) = head_physical(mapping) {
                movement[operand] = Some(Collective {
                    ranks: topology.dimensions[axis],
                    nodes: nodes_per_axis[axis],
                    bytes: sizes[operand] * element_bytes,
                });
            }
        }
        for (&operand, &dimension) in order.iter().zip(&dimensions) {
            update_panel_state(
                &mut states[operand],
                dimension,
                &distributions[operand].mappings[dimension],
                steps,
            );
        }
        phases[label] = panel_virtual(first_map, second_map, steps);
        panels.push(Panel {
            steps: edge_length,
            panel_bytes: sizes.map(|size| size * element_bytes),
            movement,
        });
    }

    for operand in 0..3 {
        assert!(states[operand].block_size >= block_shapes[operand].iter().product());
    }
    let operand_bytes = block_shapes
        .each_ref()
        .map(|shape| shape.iter().product::<usize>() * element_bytes);
    let mut extents = vec![None; labels.len()];
    for operand in 0..3 {
        for (axis, &label) in normalized[operand].iter().enumerate() {
            if let Some(previous) = extents[label] {
                assert_eq!(previous, block_shapes[operand][axis]);
            } else {
                extents[label] = Some(block_shapes[operand][axis]);
            }
        }
    }
    let mut tree = Tree::Local {
        custom: local_custom,
        folded: false,
        operand_bytes,
        flops: 2.0
            * extents
                .into_iter()
                .map(Option::unwrap)
                .product::<usize>() as f64,
    };
    if phases.iter().product::<usize>() > 1 {
        tree = Tree::Virtual {
            phases,
            orders: std::array::from_fn(|operand| normalized[operand].len()),
            child: Box::new(tree),
        };
    }
    for panel in panels.into_iter().rev() {
        tree = Tree::Panels {
            steps: panel.steps,
            panel_bytes: panel.panel_bytes,
            movement: panel.movement,
            custom_reduce,
            child: Box::new(tree),
        };
    }
    if !replicate_inputs[0].is_empty()
        || !replicate_inputs[1].is_empty()
        || !replicate_output.is_empty()
    {
        tree = Tree::Replicated {
            inputs: replicate_inputs,
            output: replicate_output,
            custom_reduce,
            child: Box::new(tree),
        };
    }
    tree
}

/// Source `detail_estimate_mem_and_time` for a dense unfolded raw mapping.
/// Output redistribution time is counted twice, while only changed inputs add
/// resident mapped storage to the memory estimate.
pub fn estimate_dense_unfolded(
    old: [&Distribution; 3],
    mapped: [&Distribution; 3],
    indices: [&str; 3],
    models: &crate::cost::Models,
    element_bytes: usize,
    nodes_per_axis: &[f64],
    local_custom: bool,
    custom_reduce: bool,
) -> crate::grid_plan_cost::UnfoldedEstimate {
    let inner = dense_unfolded(
        mapped,
        indices,
        element_bytes,
        nodes_per_axis,
        local_custom,
        custom_reduce,
    )
    .estimate(models, 1);
    let mut redistribution_seconds = [0.; 3];
    let mut redistributed_input_bytes = 0;
    let mut redistribution_temporary_bytes = 0;
    for operand in 0..3 {
        if crate::redist_cost::same_mapping(old[operand], mapped[operand]) {
            continue;
        }
        let cost = crate::redist_cost::dense(
            old[operand],
            mapped[operand],
            element_bytes,
            models,
        );
        redistribution_seconds[operand] = cost.seconds * if operand == 2 { 2. } else { 1. };
        redistribution_temporary_bytes += cost.temporary_bytes;
        if operand < 2 {
            redistributed_input_bytes += mapped[operand].local_len() * element_bytes;
        }
    }
    let seconds = inner.seconds + redistribution_seconds.iter().sum::<f64>();
    let memory_bytes = redistributed_input_bytes
        + redistribution_temporary_bytes.max(inner.working_bytes);
    crate::grid_plan_cost::UnfoldedEstimate {
        inner,
        redistribution_seconds,
        redistributed_input_bytes,
        redistribution_temporary_bytes,
        seconds,
        memory_bytes,
    }
}
