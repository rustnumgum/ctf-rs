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
    storage: [Storage; 3],
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
        self.tree.estimate(models, self.fractions, layers)
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
        let tree = self.estimate(models, layers);
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
        TotalEstimate {
            seconds: tree.seconds + redistribution[0].seconds
                + redistribution[1].seconds + 2. * redistribution[2].seconds,
            memory_bytes: resident + temporary.max(usize::try_from(tree.working_bytes).unwrap()),
            tree,
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

/// Build source's non-folded sparse tree. This is the outer/general k0 path;
/// folded k1--k5 selection needs fold metadata and is deliberately separate.
pub fn build_unfolded(
    mapped: [&Distribution; 3],
    indices: [&str; 3],
    inputs: Inputs,
) -> Result<Plan, Unsupported> {
    if inputs.storage.map(|storage| storage.sparse) != [true, false, false] {
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
    let mut need_replication = false;
    for axis in 0..topology.dimensions.len() {
        need_replication |= physical.iter().any(|operand| !operand[axis]);
        if physical.iter().all(|operand| !operand[axis]) { continue; }
        for operand in 0..3 {
            if !physical[operand][axis] {
                communicator_ranks[operand].push(topology.dimensions[axis]);
            }
        }
    }

    let mut phases = vec![1; labels.len()];
    let mut panels = Vec::new();
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
        }
        for (&operand, &axis) in order.iter().zip(&axes) {
            update_panel(&mut states[operand], axis, &mapped[operand].mappings[axis], steps);
        }
        phases[label] = panel_virtual(first, second, steps);
        panels.push(sparse_cost::TwoDimensional {
            edge, a: operands[0], b: operands[1], c: operands[2],
        });
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
    let fractions = [inputs.fractions.a, inputs.fractions.b, inputs.fractions.c];
    let local = sparse_cost::local::Local {
        kernel: sparse_cost::local::Kernel::General {
            extents: extents.into_iter().map(Option::unwrap).collect(),
        },
        custom: inputs.custom,
        packed_elements: block_shapes.each_ref().map(|shape| shape.iter().product()),
        element_bytes: inputs.storage.map(|storage| storage.element_size),
        sparse: inputs.storage.map(|storage| storage.sparse),
        nnz_fraction: fractions,
    };
    let mut tree = Tree::Local(local);
    if phases.iter().product::<usize>() > 1 {
        tree = Tree::Virtual(sparse_cost::Virtual {
            dimensions: phases,
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
        tree = Tree::Pin(sparse_cost::KeyPinning {
            operand: Operand::A,
            dense_block_size: mapped[0].local_len(),
            pair_sizes: inputs.storage.map(|storage| storage.pair_size),
        }, Box::new(tree));
    }
    Ok(Plan { tree, fractions: inputs.fractions, storage: inputs.storage })
}
