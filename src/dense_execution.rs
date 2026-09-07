// Adapted from cc4s contraction/{contraction,ctr_comm,ctr_2d_general}.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Dense semiring execution for a preflight-valid raw NS mapping. This is the
//! Source paths share outer replication, general 2D panels and virtualization,
//! with sequential semiring or packed folded BLAS leaves.

use crate::{
    algebra::{Arithmetic, Semiring, Wire},
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
    has_replication: bool,
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
    let has_replication = (0..topology.dimensions.len())
        .any(|axis| (0..3).any(|operand| !physical[operand][axis]));
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
        has_replication,
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_levels<A: Semiring, F>(
    algebra: &A,
    levels: &[Level<'_>],
    level: usize,
    layers: Layers,
    a: &[A::Element],
    b: &[A::Element],
    c: &mut [A::Element],
    beta: A::Element,
    leaf: &mut F,
) where
    A::Element: Wire,
    F: FnMut(&[A::Element], &[A::Element], &mut [A::Element], A::Element),
{
    if level == levels.len() {
        leaf(a, b, c, beta);
        return;
    }
    let current = &levels[level];
    let panels: [Panel<'_, '_>; 3] = std::array::from_fn(|operand| Panel {
        comm: current.comms[operand].as_ref(),
        outer: current.specs[operand].outer,
        inner: current.specs[operand].inner,
    });
    crate::ctr_2d::execute(
        algebra,
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
                algebra,
                levels,
                level + 1,
                layers,
                a,
                b,
                c,
                beta,
                leaf,
            )
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn folded_virtualized<K: crate::linalg::LocalKernels>(
    descriptor: &crate::partial_fold::Descriptor,
    block_shapes: &[Vec<usize>; 3],
    virtual_phases: &[Vec<usize>; 3],
    indices: [&str; 3],
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    alpha: f64,
    beta: f64,
) {
    let block_sizes = descriptor
        .layouts
        .each_ref()
        .map(|layout| layout.group_lengths.iter().product::<usize>());
    let counts = virtual_phases.each_ref().map(|phases| phases.iter().product::<usize>());
    assert_eq!(a.len(), block_sizes[0] * counts[0]);
    assert_eq!(b.len(), block_sizes[1] * counts[1]);
    assert_eq!(c.len(), block_sizes[2] * counts[2]);
    let links: [Vec<crate::symmetry::Symmetry>; 3] = std::array::from_fn(|operand| {
        vec![crate::symmetry::Symmetry::NS; block_shapes[operand].len()]
    });
    let space = crate::summation::Indices::new(&[
        (&virtual_phases[0], indices[0]),
        (&virtual_phases[1], indices[1]),
        (&virtual_phases[2], indices[2]),
    ]);
    let mut visited = vec![false; counts[2]];
    space.for_each(|offsets| {
        let (ia, ib, ic) = (offsets[0], offsets[1], offsets[2]);
        crate::partial_fold_kernel::execute_packed::<K>(
            descriptor,
            block_shapes.each_ref().map(Vec::as_slice),
            links.each_ref().map(Vec::as_slice),
            indices,
            &a[ia * block_sizes[0]..(ia + 1) * block_sizes[0]],
            &b[ib * block_sizes[1]..(ib + 1) * block_sizes[1]],
            &mut c[ic * block_sizes[2]..(ic + 1) * block_sizes[2]],
            alpha,
            if visited[ic] { 1. } else { beta },
        );
        visited[ic] = true;
    });
}

impl<A: Semiring + Clone> Tensor<'_, '_, A>
where
    A::Element: Wire,
{
    /// Execute the source contraction using raw selected mappings. Products are
    /// right-scaled by alpha; nonscalar C is left-scaled by beta. A scalar leaf
    /// without a replication layer retains the source right-beta special case.
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
        alpha: A::Element,
        beta: A::Element,
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
        let algebra = self.algebra().clone();
        aa.redistribute(mapped[0].clone());
        bb.redistribute(mapped[1].clone());
        cc.redistribute(mapped[2].clone());

        for comm in &execution.replicate[0] {
            comm.broadcast(0, &mut aa.data);
        }
        for comm in &execution.replicate[1] {
            comm.broadcast(0, &mut bb.data);
        }
        // construct_dense_ctr installs ctr_replicate when any topology axis is
        // missing from any operand, even when an entirely unused axis gives
        // that layer no actual communicator.
        let has_replication = execution.has_replication;
        let output_root = execution.replicate[2]
            .iter()
            .all(|comm| comm.rank() == 0);
        if has_replication && output_root && beta != algebra.one() {
            if beta == algebra.zero() {
                cc.data.fill(algebra.zero());
            } else {
                for value in &mut cc.data {
                    *value = algebra.multiply(&beta, value);
                }
            }
        }
        let mut leaf = |a: &[A::Element],
                        b: &[A::Element],
                        c: &mut [A::Element],
                        beta: A::Element| {
            crate::contraction::virtualized(
                &algebra,
                &execution.block_shapes[0],
                &execution.virtual_phases[0],
                indices_a,
                a,
                &execution.block_shapes[1],
                &execution.virtual_phases[1],
                indices_b,
                b,
                &execution.block_shapes[2],
                &execution.virtual_phases[2],
                indices_c,
                c,
                &alpha,
                &beta,
            );
        };
        execute_levels(
            &algebra,
            &execution.levels,
            0,
            Layers { count: 1, index: 0 },
            &aa.data,
            &bb.data,
            &mut cc.data,
            if has_replication {
                if output_root {
                    algebra.one()
                } else {
                    algebra.zero()
                }
            } else {
                beta
            },
            &mut leaf,
        );
        for comm in &execution.replicate[2] {
            comm.reduce_monoid(&algebra, &mut cc.data, false, 0);
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

impl Tensor<'_, '_, Arithmetic<f64>> {
    #[allow(clippy::too_many_arguments)]
    fn execute_folded_mapped<K: crate::linalg::LocalKernels>(
        c: &mut Self,
        indices_c: &str,
        a: &mut Self,
        indices_a: &str,
        b: &mut Self,
        indices_b: &str,
        mapped: [Distribution; 3],
        descriptor: &crate::partial_fold::Descriptor,
        intra_node_lens: Option<&[usize]>,
        alpha: f64,
        beta: f64,
        restore_inputs: bool,
    ) {
        assert!(std::ptr::eq(c.context(), a.context()));
        assert!(std::ptr::eq(c.context(), b.context()));
        assert_eq!(mapped[0].shape, a.distribution().shape);
        assert_eq!(mapped[1].shape, b.distribution().shape);
        assert_eq!(mapped[2].shape, c.distribution().shape);
        assert_eq!(mapped[0].topology.size(), c.context().size());
        assert!(crate::mapping_preflight::check(
            mapped.each_ref(),
            [indices_a, indices_b, indices_c]
        ));

        let original = [
            a.distribution().clone(),
            b.distribution().clone(),
            c.distribution().clone(),
        ];
        let context = c.context();
        let algebra = c.algebra().clone();
        let reordered = intra_node_lens.map(|intra| {
            let rank = crate::node_reordering::reorder_rank(
                &mapped[0].topology.dimensions,
                intra,
                context.rank(),
            );
            context.split(Some(0), rank.try_into().unwrap()).unwrap()
        });
        let execution = build_execution(
            reordered.as_ref().unwrap_or(context),
            mapped.each_ref(),
            [indices_a, indices_b, indices_c],
        );
        for operand in 0..3 {
            assert_eq!(
                descriptor.layouts[operand]
                    .group_lengths
                    .iter()
                    .product::<usize>(),
                execution.block_shapes[operand].iter().product::<usize>()
            );
        }
        let virtual_blocks = mapped.each_ref().map(|distribution| {
            distribution
                .mappings
                .iter()
                .map(|mapping| mapping.phase() / mapping.physical_phase())
                .product::<usize>()
        });
        if a.distribution() != &mapped[0] {
            a.redistribute(mapped[0].clone());
        }
        if b.distribution() != &mapped[1] {
            b.redistribute(mapped[1].clone());
        }
        if c.distribution() != &mapped[2] {
            c.redistribute(mapped[2].clone());
        }
        a.data = descriptor.layouts[0].transpose(
            &a.data,
            virtual_blocks[0],
            crate::fold_layout::Direction::Forward,
        );
        b.data = descriptor.layouts[1].transpose(
            &b.data,
            virtual_blocks[1],
            crate::fold_layout::Direction::Forward,
        );
        c.data = descriptor.layouts[2].transpose(
            &c.data,
            virtual_blocks[2],
            crate::fold_layout::Direction::Forward,
        );

        let exchange = intra_node_lens.map(|intra| {
            let lens = &mapped[0].topology.dimensions;
            (
                crate::node_reordering::inverse_rank(lens, intra, context.rank()),
                crate::node_reordering::reorder_rank(lens, intra, context.rank()),
            )
        });
        if let Some((send, recv)) = exchange {
            if send != context.rank() {
                context.inner.replace_f64(&mut a.data, send, recv, 1322);
                context.inner.replace_f64(&mut b.data, send, recv, 1323);
                context.inner.replace_f64(&mut c.data, send, recv, 1324);
            }
        }

        for comm in &execution.replicate[0] {
            comm.broadcast(0, &mut a.data);
        }
        for comm in &execution.replicate[1] {
            comm.broadcast(0, &mut b.data);
        }
        let has_replication = execution.has_replication;
        let output_root = execution.replicate[2]
            .iter()
            .all(|comm| comm.rank() == 0);
        if has_replication && output_root && beta != 1. {
            if beta == 0. {
                c.data.fill(0.);
            } else {
                for value in &mut c.data {
                    *value *= beta;
                }
            }
        }
        let mut leaf = |a: &[f64], b: &[f64], c: &mut [f64], beta: f64| {
            folded_virtualized::<K>(
                descriptor,
                &execution.block_shapes,
                &execution.virtual_phases,
                [indices_a, indices_b, indices_c],
                a,
                b,
                c,
                alpha,
                beta,
            );
        };
        execute_levels(
            &algebra,
            &execution.levels,
            0,
            Layers { count: 1, index: 0 },
            &a.data,
            &b.data,
            &mut c.data,
            if has_replication {
                if output_root { 1. } else { 0. }
            } else {
                beta
            },
            &mut leaf,
        );
        for comm in &execution.replicate[2] {
            comm.reduce_f64(0, &mut c.data);
        }
        if let Some((send, recv)) = exchange {
            if send != context.rank() {
                context.inner.replace_f64(&mut c.data, recv, send, 1327);
                if restore_inputs {
                    context.inner.replace_f64(&mut a.data, recv, send, 1325);
                    context.inner.replace_f64(&mut b.data, recv, send, 1326);
                }
            }
        }
        c.data = descriptor.layouts[2].transpose(
            &c.data,
            virtual_blocks[2],
            crate::fold_layout::Direction::Backward,
        );
        if restore_inputs {
            a.data = descriptor.layouts[0].transpose(
                &a.data,
                virtual_blocks[0],
                crate::fold_layout::Direction::Backward,
            );
            b.data = descriptor.layouts[1].transpose(
                &b.data,
                virtual_blocks[1],
                crate::fold_layout::Direction::Backward,
            );
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
        if let Some(context) = reordered {
            context.close();
        }
        if c.distribution() != &original[2] {
            c.redistribute(original[2].clone());
        }
        if restore_inputs {
            if a.distribution() != &original[0] {
                a.redistribute(original[0].clone());
            }
            if b.distribution() != &original[1] {
                b.redistribute(original[1].clone());
            }
        }
    }

    /// Execute a selected dense folded raw mapping. Every local virtual block is
    /// transposed once before replication/panels and restored once after output
    /// reduction, matching map_fold rather than repacking individual panels.
    /// Optional intra-node dimensions reorder the communicator and local blocks
    /// using the source node-aware permutation. Pass select_dense's chosen lens
    /// or an explicitly requested node grid; None leaves rank order unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn contract_folded_from_mapped<K: crate::linalg::LocalKernels>(
        &mut self,
        indices_c: &str,
        a: &Self,
        indices_a: &str,
        b: &Self,
        indices_b: &str,
        mapped: [Distribution; 3],
        descriptor: &crate::partial_fold::Descriptor,
        intra_node_lens: Option<&[usize]>,
        alpha: f64,
        beta: f64,
    ) {
        let mut aa = (*a).clone();
        let mut bb = (*b).clone();
        let mut cc = self.clone();
        Self::execute_folded_mapped::<K>(
            &mut cc,
            indices_c,
            &mut aa,
            indices_a,
            &mut bb,
            indices_b,
            mapped,
            descriptor,
            intra_node_lens,
            alpha,
            beta,
            false,
        );
        *self = cc;
    }

    /// Execute the source low-memory folded path in-place. Input and output
    /// buffers are remapped for contraction and restored to their original
    /// distributions before returning; zero-length tensor dimensions are not
    /// supported by the source low-memory path.
    #[allow(clippy::too_many_arguments)]
    pub fn contract_folded_low_memory<K: crate::linalg::LocalKernels>(
        &mut self,
        indices_c: &str,
        a: &mut Self,
        indices_a: &str,
        b: &mut Self,
        indices_b: &str,
        mapped: [Distribution; 3],
        descriptor: &crate::partial_fold::Descriptor,
        intra_node_lens: Option<&[usize]>,
        alpha: f64,
        beta: f64,
    ) {
        assert!(
            self.distribution().shape.iter().all(|&length| length != 0)
                && a.distribution().shape.iter().all(|&length| length != 0)
                && b.distribution().shape.iter().all(|&length| length != 0),
            "source low-memory contraction does not support zero-length dimensions"
        );
        Self::execute_folded_mapped::<K>(
            self,
            indices_c,
            a,
            indices_a,
            b,
            indices_b,
            mapped,
            descriptor,
            intra_node_lens,
            alpha,
            beta,
            true,
        );
    }
}
