// Adapted from cc4s contraction/{contraction,ctr_comm,ctr_2d_general}.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Dense f64 execution for a preflight-valid raw NS mapping. This is the
//! unfolded source path: outer replication, general 2D panels, virtualization,
//! and the sequential reference kernel.

use crate::{
    algebra::Arithmetic,
    context::Context,
    ctr_2d::{Layers, Panel},
    mapping::{Distribution, Mapping},
    tensor::Tensor,
};

#[derive(Clone)]
struct OperandState {
    block_size: usize,
    block_lengths: Vec<usize>,
    virtual_block_lengths: Vec<usize>,
}

#[derive(Clone, Copy)]
struct PanelSpec {
    axis: Option<usize>,
    outer: usize,
    inner: usize,
}

struct Level<'context> {
    edge: usize,
    specs: [PanelSpec; 3],
    comms: [Option<Context<'context>>; 3],
}

struct Execution<'context> {
    levels: Vec<Level<'context>>,
    block_shapes: [Vec<usize>; 3],
    virtual_phases: [Vec<usize>; 3],
    replicate: [Vec<Context<'context>>; 3],
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

// Literal comp_dim_map direction used by ctr_2d_gen_build. Generated raw
// candidates have only terminal virtual children at this execution stage.
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

fn panel_parts(
    state: &OperandState,
    dimension: usize,
    mapping: &Mapping,
    edge_length: usize,
) -> (usize, usize) {
    let mut inner = if let Some((_, processes)) = head_physical(mapping) {
        state.block_size * processes / edge_length
    } else {
        state.block_size / edge_length
    };
    let mut outer = 1;
    for axis in dimension + 1..state.block_lengths.len() {
        inner = inner * state.virtual_block_lengths[axis] / state.block_lengths[axis];
        outer = outer * state.block_lengths[axis] / state.virtual_block_lengths[axis];
    }
    (outer, inner)
}

fn update_panel_state(
    state: &mut OperandState,
    dimension: usize,
    mapping: &Mapping,
    steps: usize,
) {
    if let Some((_, processes)) = head_physical(mapping) {
        state.block_size = state.block_size * processes / steps;
        state.block_lengths[dimension] = state.block_lengths[dimension] * processes / steps;
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

fn build_execution<'context>(
    context: &'context Context<'_>,
    distributions: [&Distribution; 3],
    indices: [&str; 3],
) -> Execution<'context> {
    let topology = &distributions[0].topology;
    let mut labels = Vec::new();
    let mut lengths = Vec::new();
    let mut normalized: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::new());
    for operand in 0..3 {
        for (axis, label) in indices[operand].bytes().enumerate() {
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

    let initial_virtual: [Vec<usize>; 3] = distributions.each_ref().map(|distribution| {
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
            .zip(&initial_virtual[operand])
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
    let mut replicate: [Vec<Context<'context>>; 3] = std::array::from_fn(|_| Vec::new());
    for axis in 0..topology.dimensions.len() {
        if !(physical[0][axis] || physical[1][axis] || physical[2][axis]) {
            continue;
        }
        for operand in 0..3 {
            if !physical[operand][axis] {
                replicate[operand].push(topology.fiber(context, axis));
            }
        }
    }

    let mut phases = vec![1usize; labels.len()];
    let mut raw_levels = Vec::new();
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

        let mut edge = 1;
        let mut steps = 1;
        for (&operand, &dimension) in order.iter().zip(&dimensions) {
            let mapping = &distributions[operand].mappings[dimension];
            if head_physical(mapping).is_some() {
                edge = lcm(edge, mapping.phase());
                steps = steps.max(mapping.phase());
            }
        }
        let mut specs = [PanelSpec {
            axis: None,
            outer: 1,
            inner: 0,
        }; 3];
        for (&operand, &dimension) in order.iter().zip(&dimensions) {
            let mapping = &distributions[operand].mappings[dimension];
            let (outer, inner) = panel_parts(&states[operand], dimension, mapping, edge);
            specs[operand] = PanelSpec {
                axis: head_physical(mapping).map(|(axis, _)| axis),
                outer,
                inner,
            };
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
        raw_levels.push((edge, specs));
    }
    let levels = raw_levels
        .into_iter()
        .map(|(edge, specs)| Level {
            edge,
            comms: std::array::from_fn(|operand| {
                specs[operand]
                    .axis
                    .map(|axis| topology.fiber(context, axis))
            }),
            specs,
        })
        .collect();
    let virtual_phases =
        std::array::from_fn(|operand| normalized[operand].iter().map(|&label| phases[label]).collect());
    Execution {
        levels,
        block_shapes,
        virtual_phases,
        replicate,
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_levels(
    levels: &[Level<'_>],
    level: usize,
    layers: Layers,
    block_shapes: &[Vec<usize>; 3],
    virtual_phases: &[Vec<usize>; 3],
    indices: [&str; 3],
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    alpha: f64,
    beta: f64,
) {
    if level == levels.len() {
        crate::contraction::virtualized(
            &Arithmetic::<f64>::new(),
            &block_shapes[0],
            &virtual_phases[0],
            indices[0],
            a,
            &block_shapes[1],
            &virtual_phases[1],
            indices[1],
            b,
            &block_shapes[2],
            &virtual_phases[2],
            indices[2],
            c,
            &alpha,
            &beta,
        );
        return;
    }
    let current = &levels[level];
    let panels: [Panel<'_, '_>; 3] = std::array::from_fn(|operand| Panel {
        comm: current.comms[operand].as_ref(),
        outer: current.specs[operand].outer,
        inner: current.specs[operand].inner,
    });
    crate::ctr_2d::execute(
        current.edge,
        layers,
        panels[0],
        panels[1],
        panels[2],
        a,
        b,
        c,
        beta,
        |a, b, c, beta, layers| {
            execute_levels(
                levels,
                level + 1,
                layers,
                block_shapes,
                virtual_phases,
                indices,
                a,
                b,
                c,
                alpha,
                beta,
            )
        },
    );
}

impl Tensor<'_, '_, Arithmetic<f64>> {
    /// Execute `C = alpha*A*B + beta*C` using exact raw selected mappings.
    /// Inputs and output are restored to their original distributions; no
    /// aligned remapping, folding, or global tensor gather is performed.
    pub fn contract_from_mapped(
        &mut self,
        indices_c: &str,
        a: &Self,
        indices_a: &str,
        b: &Self,
        indices_b: &str,
        mapped: [Distribution; 3],
        alpha: f64,
        beta: f64,
    ) {
        assert!(std::ptr::eq(self.context(), a.context()));
        assert!(std::ptr::eq(self.context(), b.context()));
        assert_eq!(mapped[0].shape, a.distribution().shape);
        assert_eq!(mapped[1].shape, b.distribution().shape);
        assert_eq!(mapped[2].shape, self.distribution().shape);
        assert_eq!(mapped[0].topology.size(), self.context().size());
        assert!(crate::mapping_preflight::check(
            mapped.each_ref(),
            [indices_a, indices_b, indices_c]
        ));

        let execution = build_execution(
            self.context(),
            mapped.each_ref(),
            [indices_a, indices_b, indices_c],
        );
        let mut aa = (*a).clone();
        let mut bb = (*b).clone();
        let mut cc = self.clone();
        aa.redistribute(mapped[0].clone());
        bb.redistribute(mapped[1].clone());
        cc.redistribute(mapped[2].clone());

        for comm in &execution.replicate[0] {
            comm.broadcast(0, &mut aa.data);
        }
        for comm in &execution.replicate[1] {
            comm.broadcast(0, &mut bb.data);
        }
        let output_root = execution.replicate[2]
            .iter()
            .all(|comm| comm.rank() == 0);
        if output_root && beta != 1. {
            if beta == 0. {
                cc.data.fill(0.);
            } else {
                for value in &mut cc.data {
                    *value *= beta;
                }
            }
        }
        execute_levels(
            &execution.levels,
            0,
            Layers { count: 1, index: 0 },
            &execution.block_shapes,
            &execution.virtual_phases,
            [indices_a, indices_b, indices_c],
            &aa.data,
            &bb.data,
            &mut cc.data,
            alpha,
            if output_root { 1. } else { 0. },
        );
        for comm in &execution.replicate[2] {
            comm.reduce_f64(0, &mut cc.data);
        }
        for group in execution.replicate {
            for comm in group {
                comm.close();
            }
        }
        for level in execution.levels {
            for comm in level.comms.into_iter().flatten() {
                comm.close();
            }
        }
        cc.redistribute(self.distribution().clone());
        *self = cc;
    }
}
