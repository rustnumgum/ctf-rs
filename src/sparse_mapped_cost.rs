// Adapted from cc4s CTF contraction/contraction.cxx::{construct_sparse_ctr,
// detail_estimate_mem_and_time} and ctr_2d_general.cxx::ctr_2d_gen_build at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Source sparse execution-tree assembly for an unfolded raw mapping.

use crate::{
    cost::Models,
    folding::Operand,
    mapping::{calc_dim, Distribution, Mapping},
    sparse_cost::{self, Fractions, Operand2d, ReplicaOperand, Storage},
};

#[derive(Clone, Copy, Debug)]
pub struct Inputs {
    pub storage: [Storage; 3],
    pub fractions: Fractions,
    pub custom: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unsupported {
    Storage,
    NonUniqueLabels,
    AOnlyLabel,
    Shape,
    Mapping,
    DenseVirtualSize { operand: usize, expected: usize },
}

#[derive(Clone, Debug)]
pub enum Tree {
    Pin(sparse_cost::KeyPinning, Box<Tree>),
    Replicate(sparse_cost::Replication, Box<Tree>),
    Panel(sparse_cost::TwoDimensional, Box<Tree>),
    Virtual(sparse_cost::Virtual, Box<Tree>),
    Local(sparse_cost::local::Local),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub seconds: f64,
    pub working_bytes: i64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TotalEstimate {
    pub seconds: f64,
    pub memory_bytes: usize,
    pub tree: Estimate,
    pub redistribution: [crate::redist_cost::Estimate; 3],
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub tree: Tree,
    fractions: Fractions,
    tree_fractions: Fractions,
    storage: [Storage; 3],
    fold: Option<FoldCost>,
    pub(crate) execution: Execution,
}

#[derive(Clone, Debug)]
struct FoldCost {
    transpose_seconds: [f64; 3],
    resident_bytes: usize,
    temporary_bytes: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct Execution {
    pub replication_axes: [Vec<usize>; 3],
    pub panels: Vec<ExecutionPanel>,
    pub virtual_dimensions: Vec<usize>,
    pub indices: [Vec<usize>; 3],
    pub block_shapes: [Vec<usize>; 3],
}

#[derive(Clone, Debug)]
pub(crate) struct ExecutionPanel {
    pub edge: usize,
    pub operands: [ExecutionOperand; 3],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExecutionOperand {
    pub topology_axis: Option<usize>,
    pub outer: usize,
    pub inner: usize,
}

impl Tree {
    fn estimate(&self, models: &Models, fractions: Fractions, layers: usize) -> Estimate {
        assert!(layers > 0);
        match self {
            Self::Local(local) => Estimate {
                seconds: local.estimate(models).seconds,
                working_bytes: 0,
            },
            Self::Virtual(level, child) => {
                let child = child.estimate(models, fractions, layers);
                Estimate {
                    seconds: level.est_time(child.seconds),
                    working_bytes: level.memory(child.working_bytes),
                }
            }
            Self::Panel(level, child) => {
                let child = child.estimate(models, fractions, 1);
                Estimate {
                    seconds: level.est_time(models, fractions, layers, child.seconds),
                    working_bytes: level.memory(fractions, child.working_bytes),
                }
            }
            Self::Replicate(level, child) => {
                let child = child.estimate(models, fractions, layers);
                Estimate {
                    seconds: level.est_time(models, fractions, child.seconds),
                    working_bytes: level.memory(fractions, child.working_bytes),
                }
            }
            Self::Pin(level, child) => {
                let child = child.estimate(models, fractions, layers);
                Estimate {
                    seconds: level.fixed_time(models, fractions) + child.seconds,
                    working_bytes: level.footprint(fractions) + child.working_bytes,
                }
            }
        }
    }
}

impl Plan {
    pub fn estimate(&self, models: &Models, layers: usize) -> Estimate {
        let mut estimate = self.tree.estimate(models, self.tree_fractions, layers);
        if let Some(fold) = &self.fold {
            estimate.seconds += fold.transpose_seconds.iter().sum::<f64>();
        }
        estimate
    }

    /// Unfolded `detail_estimate_mem_and_time`: changed input layouts remain
    /// resident, redistribution temporaries overlap the contraction tree, and
    /// output redistribution time is charged twice.
    pub fn estimate_with_redistribution(
        &self,
        old: [&Distribution; 3],
        mapped: [&Distribution; 3],
        models: &Models,
        layers: usize,
    ) -> TotalEstimate {
        let fractions = [self.fractions.a, self.fractions.b, self.fractions.c];
        let redistribution = std::array::from_fn(|operand| {
            if self.storage[operand].sparse {
                crate::redist_cost::sparse(
                    old[operand],
                    mapped[operand],
                    self.storage[operand].element_size,
                    self.storage[operand].pair_size,
                    fractions[operand],
                    models,
                )
            } else {
                crate::redist_cost::dense(
                    old[operand],
                    mapped[operand],
                    self.storage[operand].element_size,
                    models,
                )
            }
        });
        let tree_only = self.tree.estimate(models, self.tree_fractions, layers);
        let mut seconds = tree_only.seconds;
        if let Some(fold) = &self.fold {
            seconds += fold.transpose_seconds.iter().sum::<f64>();
        }
        let changed = std::array::from_fn::<_, 3, _>(|operand| {
            !crate::redist_cost::same_mapping(old[operand], mapped[operand])
        });
        let resident = (0..2).filter(|&operand| changed[operand]).map(|operand| {
            if self.storage[operand].sparse {
                (mapped[operand].local_len() as f64
                    * self.storage[operand].pair_size as f64
                    * fractions[operand]) as usize
            } else {
                mapped[operand].local_len() * self.storage[operand].element_size
            }
        }).sum::<usize>();
        let temporary = (0..3).filter(|&operand| changed[operand])
            .map(|operand| redistribution[operand].temporary_bytes).sum::<usize>();
        let (fold_resident, fold_temporary) = self.fold.as_ref()
            .map_or((0, 0), |fold| (fold.resident_bytes, fold.temporary_bytes));
        TotalEstimate {
            seconds: seconds + redistribution[0].seconds
                + redistribution[1].seconds + 2. * redistribution[2].seconds,
            memory_bytes: resident + temporary.max(
                fold_temporary.max(fold_resident + usize::try_from(tree_only.working_bytes).unwrap())
            ),
            tree: tree_only,
            redistribution,
        }
    }
}

#[derive(Clone)]
struct OperandState {
    block_size: usize,
    block_lengths: Vec<usize>,
    virtual_block_lengths: Vec<usize>,
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 { (a, b) = (b, a % b); }
    a
}

fn lcm(a: usize, b: usize) -> usize { a / gcd(a, b) * b }

fn head_physical(mapping: &Mapping) -> Option<(usize, usize)> {
    match mapping {
        Mapping::Physical { axis, processes, .. } => Some((*axis, *processes)),
        _ => None,
    }
}

fn terminal_virtual(mapping: &Mapping) -> usize {
    match mapping {
        Mapping::Unmapped => 1,
        Mapping::Physical { child, .. } | Mapping::Virtual { child, .. } => {
            if matches!(**child, Mapping::Unmapped) {
                if let Mapping::Virtual { copies, .. } = mapping { *copies } else { 1 }
            } else {
                terminal_virtual(child)
            }
        }
    }
}

// Source comp_dim_map includes the directional Unmapped/Virtual(1) case.
fn source_equal(left: &Mapping, right: &Mapping) -> bool {
    match (left, right) {
        (Mapping::Unmapped, Mapping::Unmapped) => true,
        (Mapping::Unmapped, Mapping::Virtual { copies: 1, .. }) => true,
        (Mapping::Unmapped, _) | (_, Mapping::Unmapped) => false,
        (Mapping::Physical { axis: la, processes: lp, child: lc },
         Mapping::Physical { axis: ra, processes: rp, child: rc }) => {
            la == ra && lp == rp && match (
                matches!(**lc, Mapping::Unmapped), matches!(**rc, Mapping::Unmapped)
            ) {
                (true, true) => true,
                (false, false) => source_equal(lc, rc),
                _ => false,
            }
        }
        (Mapping::Virtual { copies: left, .. }, Mapping::Virtual { copies: right, .. }) => left == right,
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

fn panel_strides(
    state: &OperandState,
    dimension: usize,
    mapping: &Mapping,
    edge: usize,
) -> (usize, usize) {
    let mut inner = if let Some((_, processes)) = head_physical(mapping) {
        state.block_size * processes / edge
    } else {
        state.block_size / edge
    };
    let mut outer = 1;
    for axis in dimension + 1..state.block_lengths.len() {
        inner = inner * state.virtual_block_lengths[axis] / state.block_lengths[axis];
        outer = outer * state.block_lengths[axis] / state.virtual_block_lengths[axis];
    }
    (outer, inner)
}

fn update_panel(state: &mut OperandState, dimension: usize, mapping: &Mapping, steps: usize) {
    if let Some((_, processes)) = head_physical(mapping) {
        state.block_size = state.block_size * processes / steps;
        state.block_lengths[dimension] = state.block_lengths[dimension] * processes / steps;
    } else {
        state.block_size /= steps;
        state.block_lengths[dimension] /= steps;
    }
}

fn physical_child_virtual(mapping: &Mapping, steps: usize) -> Option<usize> {
    let Mapping::Physical { processes, child, .. } = mapping else { return None };
    match &**child {
        Mapping::Unmapped => None,
        Mapping::Virtual { copies, child } => {
            assert!(matches!(**child, Mapping::Unmapped));
            Some(processes * copies / steps)
        }
        Mapping::Physical { .. } => unreachable!("mapping preflight rejects this panel"),
    }
}

fn panel_virtual(first: &Mapping, second: &Mapping, steps: usize) -> usize {
    let mut phase = 1;
    if let Some(value) = physical_child_virtual(first, steps) { phase = value; }
    if let Some(value) = physical_child_virtual(second, steps) { phase = value; }
    if let Mapping::Virtual { copies, .. } = second { phase = copies / steps; }
    if let Mapping::Virtual { copies, .. } = first { phase = copies / steps; }
    phase
}

fn folded_kind(storage: [bool; 3], coo_kernel: bool, custom: bool) -> Option<sparse_cost::local::Folded> {
    match storage {
        [true, false, false] if coo_kernel && !custom => Some(sparse_cost::local::Folded::CooDense),
        [true, false, false] => Some(sparse_cost::local::Folded::CsrDense),
        [true, true, false] => Some(sparse_cost::local::Folded::CsrSparseDense),
        [true, true, true] => Some(sparse_cost::local::Folded::CsrSparse),
        [true, false, true] => Some(sparse_cost::local::Folded::CcsrDense),
        _ => None,
    }
}

fn fold_cost(
    mapped: [&Distribution; 3],
    inputs: Inputs,
    descriptor: &crate::partial_fold::Descriptor,
    coo_kernel: bool,
) -> (FoldCost, Fractions) {
    let sparse = inputs.storage.map(|storage| storage.sparse);
    let csr_or_coo = sparse[1] || sparse[2] || inputs.custom || !coo_kernel;
    let use_ccsr = csr_or_coo && sparse == [true, false, true];
    let original = [inputs.fractions.a, inputs.fractions.b, inputs.fractions.c];
    let mut adjusted = original;
    let mut resident = 0usize;
    let mut temporary = 0usize;
    let dimensions = [descriptor.m, descriptor.k, descriptor.m];
    for operand in 0..2 {
        let size = mapped[operand].local_len();
        let storage = inputs.storage[operand];
        if !storage.sparse {
            resident += size * storage.element_size;
            continue;
        }
        let bytes = if !csr_or_coo {
            (original[operand] * size as f64 * (storage.element_size + 8) as f64) as usize
        } else if use_ccsr {
            let bytes = (original[operand] * size as f64 * (storage.element_size + 16) as f64) as usize;
            adjusted[operand] *= (storage.element_size + 16) as f64 / storage.pair_size as f64;
            bytes
        } else {
            let virtual_blocks: usize = mapped[operand].mappings.iter()
                .map(|mapping| mapping.phase() / mapping.physical_phase()).product();
            let bytes = (original[operand] * size as f64 * (storage.element_size + 4) as f64) as usize
                + virtual_blocks * dimensions[operand] * 4;
            adjusted[operand] = original[operand]
                * (storage.element_size + 4) as f64 / storage.pair_size as f64
                + (virtual_blocks * dimensions[operand] * 4) as f64
                    / (storage.pair_size * size) as f64;
            bytes
        };
        resident += bytes;
        temporary = temporary.max(if !csr_or_coo {
            resident
        } else if use_ccsr {
            resident + bytes
        } else {
            resident + (original[operand] * size as f64 * (storage.element_size + 8) as f64) as usize
        });
    }
    let size = mapped[2].local_len();
    let storage = inputs.storage[2];
    if storage.sparse {
        let (bytes, conversion) = if !csr_or_coo {
            ((original[2] * size as f64 * (storage.element_size + 8) as f64) as usize, 0)
        } else if use_ccsr {
            let bytes = (original[2] * size as f64 * (storage.element_size + 16) as f64) as usize;
            adjusted[2] *= (storage.element_size + 16) as f64 / storage.pair_size as f64;
            (bytes, bytes)
        } else {
            let virtual_blocks: usize = mapped[2].mappings.iter()
                .map(|mapping| mapping.phase() / mapping.physical_phase()).product();
            let bytes = (original[2] * size as f64 * (storage.element_size + 4) as f64) as usize
                + virtual_blocks * descriptor.m * 4;
            adjusted[2] = original[2]
                * (storage.element_size + 4) as f64 / storage.pair_size as f64
                + (virtual_blocks * descriptor.m * 4) as f64
                    / (storage.pair_size * size) as f64;
            let conversion = (original[2] * size as f64 * (storage.element_size + 8) as f64) as usize;
            (bytes, conversion)
        };
        resident += bytes;
        temporary = temporary.max(resident);
        temporary = temporary.max(
            bytes + conversion + (original[2] * size as f64 * storage.pair_size as f64) as usize
        );
    } else {
        resident += size * storage.element_size;
    }
    (
        FoldCost {
            transpose_seconds: descriptor.transpose_seconds,
            resident_bytes: resident,
            temporary_bytes: temporary,
        },
        Fractions { a: adjusted[0], b: adjusted[1], c: adjusted[2] },
    )
}

/// Build source's non-folded sparse tree. This is the outer/general k0 path;
/// folded k1--k5 selection needs fold metadata and is deliberately separate.
pub fn build_unfolded(
    mapped: [&Distribution; 3],
    indices: [&str; 3],
    inputs: Inputs,
) -> Result<Plan, Unsupported> {
    build(mapped, indices, inputs, None, false)
}

pub fn build(
    mapped: [&Distribution; 3],
    indices: [&str; 3],
    inputs: Inputs,
    fold: Option<&crate::partial_fold::Descriptor>,
    coo_kernel: bool,
) -> Result<Plan, Unsupported> {
    let sparse = inputs.storage.map(|storage| storage.sparse);
    if fold.is_none() && sparse != [true, false, false] {
        return Err(Unsupported::Storage);
    }
    if fold.is_some() && folded_kind(sparse, coo_kernel, inputs.custom).is_none() {
        return Err(Unsupported::Storage);
    }
    let topology = &mapped[0].topology;
    if mapped.iter().any(|distribution| &distribution.topology != topology) {
        return Err(Unsupported::Mapping);
    }
    let mut labels = Vec::new();
    let mut lengths = Vec::new();
    let mut normalized: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::new());
    for operand in 0..3 {
        if !indices[operand].is_ascii() || indices[operand].len() != mapped[operand].shape.len() {
            return Err(Unsupported::Shape);
        }
        for (axis, label) in indices[operand].bytes().enumerate() {
            if indices[operand].as_bytes()[..axis].contains(&label) {
                return Err(Unsupported::NonUniqueLabels);
            }
            let id = if let Some(id) = labels.iter().position(|&old| old == label) {
                if lengths[id] != mapped[operand].shape[axis] { return Err(Unsupported::Shape); }
                id
            } else {
                labels.push(label);
                lengths.push(mapped[operand].shape[axis]);
                labels.len() - 1
            };
            normalized[operand].push(id);
        }
    }
    if normalized[0].iter().any(|label| {
        !normalized[1].contains(label) && !normalized[2].contains(label)
    }) {
        return Err(Unsupported::AOnlyLabel);
    }
    if !crate::mapping_preflight::check(mapped, indices) {
        return Err(Unsupported::Mapping);
    }
    let mut inverse = vec![[None; 3]; labels.len()];
    for operand in 0..3 {
        for (axis, &label) in normalized[operand].iter().enumerate() {
            inverse[label][operand] = Some(axis);
        }
    }

    let dimensions = mapped.each_ref().map(|distribution| {
        let padded: Vec<_> = distribution.shape.iter().zip(&distribution.mappings)
            .map(|(&length, mapping)| length.div_ceil(mapping.phase()) * mapping.phase())
            .collect();
        calc_dim(distribution.local_len(), &padded, &distribution.mappings)
    });
    for operand in 0..3 {
        if inputs.storage[operand].sparse
            && inputs.storage[operand].dense_virtual_size != dimensions[operand].virtual_size
        {
            return Err(Unsupported::DenseVirtualSize {
                operand,
                expected: dimensions[operand].virtual_size,
            });
        }
    }
    let virtual_copies: [Vec<usize>; 3] = mapped.each_ref().map(|distribution| {
        distribution.mappings.iter()
            .map(|mapping| mapping.phase() / mapping.physical_phase()).collect()
    });
    let mut states: [OperandState; 3] = std::array::from_fn(|operand| {
        if inputs.storage[operand].sparse {
            OperandState {
                block_size: virtual_copies[operand].iter().product(),
                block_lengths: virtual_copies[operand].clone(),
                virtual_block_lengths: vec![1; mapped[operand].shape.len()],
            }
        } else {
            OperandState {
                block_size: mapped[operand].local_len(),
                block_lengths: dimensions[operand].block_edges.clone(),
                virtual_block_lengths: dimensions[operand].virtual_edges.clone(),
            }
        }
    });

    let mut physical: [Vec<bool>; 3] = std::array::from_fn(|_| vec![false; topology.dimensions.len()]);
    for operand in 0..3 {
        for mapping in &mapped[operand].mappings { mark_physical(mapping, &mut physical[operand]); }
    }
    let mut communicator_ranks: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::new());
    let mut replication_axes: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::new());
    let mut need_replication = false;
    for axis in 0..topology.dimensions.len() {
        need_replication |= physical.iter().any(|operand| !operand[axis]);
        if physical.iter().all(|operand| !operand[axis]) { continue; }
        for operand in 0..3 {
            if !physical[operand][axis] {
                communicator_ranks[operand].push(topology.dimensions[axis]);
                replication_axes[operand].push(axis);
            }
        }
    }

    let mut phases = vec![1; labels.len()];
    let mut panels = Vec::new();
    let mut execution_panels = Vec::new();
    for label in 0..labels.len() {
        let present: Vec<_> = (0..3).filter_map(|operand| {
            inverse[label][operand].map(|axis| (operand, axis))
        }).collect();
        if present.len() != 2 {
            let (operand, axis) = present[0];
            phases[label] = terminal_virtual(&mapped[operand].mappings[axis]);
            continue;
        }
        let order = match (present[0].0, present[1].0) {
            (1, 2) => [1, 2],
            (0, 2) => [2, 0],
            (0, 1) => [0, 1],
            _ => unreachable!(),
        };
        let axes = order.map(|operand| inverse[label][operand].unwrap());
        let first = &mapped[order[0]].mappings[axes[0]];
        let second = &mapped[order[1]].mappings[axes[1]];
        if source_equal(second, first) {
            phases[label] = terminal_virtual(first);
            continue;
        }
        if matches!(first, Mapping::Virtual { .. }) && matches!(second, Mapping::Virtual { .. }) {
            phases[label] = if let Mapping::Virtual { copies, .. } = first { *copies } else { unreachable!() };
            continue;
        }
        let mut edge = 1;
        let mut steps = 1;
        for (&operand, &axis) in order.iter().zip(&axes) {
            let mapping = &mapped[operand].mappings[axis];
            if head_physical(mapping).is_some() {
                edge = lcm(edge, mapping.phase());
                steps = steps.max(mapping.phase());
            }
        }
        let mut operands: [Operand2d; 3] = std::array::from_fn(|operand| Operand2d {
            storage: inputs.storage[operand], moving: false, ranks: 1, outer: 1, inner: 0,
        });
        let mut execution_operands = [ExecutionOperand {
            topology_axis: None, outer: 1, inner: 0,
        }; 3];
        for (&operand, &axis) in order.iter().zip(&axes) {
            let mapping = &mapped[operand].mappings[axis];
            let (outer, inner) = panel_strides(&states[operand], axis, mapping, edge);
            let movement = head_physical(mapping);
            operands[operand] = Operand2d {
                storage: inputs.storage[operand],
                moving: movement.is_some(),
                ranks: movement.map_or(1, |(topology_axis, _)| topology.dimensions[topology_axis]),
                outer,
                inner,
            };
            let execution_inner = if inputs.storage[operand].sparse || inner == 0 {
                inner
            } else {
                assert_eq!(inner % dimensions[operand].virtual_size, 0);
                inner / dimensions[operand].virtual_size
            };
            execution_operands[operand] = ExecutionOperand {
                topology_axis: movement.map(|(topology_axis, _)| topology_axis),
                outer,
                inner: execution_inner,
            };
        }
        for (&operand, &axis) in order.iter().zip(&axes) {
            update_panel(&mut states[operand], axis, &mapped[operand].mappings[axis], steps);
        }
        phases[label] = panel_virtual(first, second, steps);
        panels.push(sparse_cost::TwoDimensional {
            edge, a: operands[0], b: operands[1], c: operands[2],
        });
        execution_panels.push(ExecutionPanel { edge, operands: execution_operands });
    }

    let block_shapes = mapped.each_ref().map(|distribution| distribution.block_shape());
    let mut extents = vec![None; labels.len()];
    for operand in 0..3 {
        for (axis, &label) in normalized[operand].iter().enumerate() {
            if let Some(old) = extents[label] {
                if old != block_shapes[operand][axis] { return Err(Unsupported::Shape); }
            } else {
                extents[label] = Some(block_shapes[operand][axis]);
            }
        }
    }
    let (fold_cost, tree_fractions) = if let Some(descriptor) = fold {
        let (cost, adjusted) = fold_cost(mapped, inputs, descriptor, coo_kernel);
        (Some(cost), adjusted)
    } else {
        (None, inputs.fractions)
    };
    let fractions = [tree_fractions.a, tree_fractions.b, tree_fractions.c];
    let kernel = if let Some(descriptor) = fold {
        sparse_cost::local::Kernel::Folded {
            kind: folded_kind(sparse, coo_kernel, inputs.custom).unwrap(),
            m: descriptor.m,
            n: descriptor.n,
            k: descriptor.k,
        }
    } else {
        sparse_cost::local::Kernel::General {
            extents: extents.into_iter().map(Option::unwrap).collect(),
        }
    };
    let packed_elements = if let Some(descriptor) = fold {
        std::array::from_fn(|operand| {
            let layout = &descriptor.layouts[operand];
            layout.inner_ordering[layout.folded_shape.len()..].iter()
                .map(|&group| layout.group_lengths[group]).product()
        })
    } else {
        block_shapes.each_ref().map(|shape| shape.iter().product())
    };
    let local = sparse_cost::local::Local {
        kernel,
        custom: inputs.custom,
        packed_elements,
        element_bytes: inputs.storage.map(|storage| storage.element_size),
        sparse: inputs.storage.map(|storage| storage.sparse),
        nnz_fraction: fractions,
    };
    let mut tree = Tree::Local(local);
    if phases.iter().product::<usize>() > 1 {
        tree = Tree::Virtual(sparse_cost::Virtual {
            dimensions: phases.clone(),
            orders: normalized.each_ref().map(Vec::len),
        }, Box::new(tree));
    }
    for panel in panels.into_iter().rev() { tree = Tree::Panel(panel, Box::new(tree)); }
    if need_replication {
        let sizes = mapped.map(Distribution::local_len);
        tree = Tree::Replicate(sparse_cost::Replication {
            a: ReplicaOperand { storage: inputs.storage[0], size: sizes[0], communicator_ranks: communicator_ranks[0].clone() },
            b: ReplicaOperand { storage: inputs.storage[1], size: sizes[1], communicator_ranks: communicator_ranks[1].clone() },
            c: ReplicaOperand { storage: inputs.storage[2], size: sizes[2], communicator_ranks: communicator_ranks[2].clone() },
        }, Box::new(tree));
    }
    if topology.size() > 1 {
        for operand in (0..3).rev() {
            if inputs.storage[operand].sparse {
                tree = Tree::Pin(sparse_cost::KeyPinning {
                    operand: [Operand::A, Operand::B, Operand::C][operand],
                    dense_block_size: mapped[operand].local_len(),
                    pair_sizes: inputs.storage.map(|storage| storage.pair_size),
                }, Box::new(tree));
            }
        }
    }
    Ok(Plan {
        tree,
        fractions: inputs.fractions,
        tree_fractions,
        storage: inputs.storage,
        fold: fold_cost,
        execution: Execution {
            replication_axes,
            panels: execution_panels,
            virtual_dimensions: phases,
            indices: normalized,
            block_shapes,
        },
    })
}
