// Adapted from cc4s CTF summation/{summation,spsum_tsr}.cxx at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f. See LICENSE.
//! Automatic raw mapping selection for sparse summation.

use crate::{
    context::Context,
    cost::{Communication, Models},
    dense_search::Kind,
    map_tensor,
    mapping::{Distribution, Mapping, Topology},
    redist_cost,
    sparse_search::StorageSize,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pattern {
    SparseSparse,
    SparseDense,
}

impl Pattern {
    fn sparse(self) -> [bool; 2] {
        match self {
            Self::SparseSparse => [true, true],
            Self::SparseDense => [true, false],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub memory_limit: u64,
}

#[derive(Clone, Debug)]
pub struct Execution {
    pub pin_keys: [bool; 2],
    pub input_permutation: Vec<Option<usize>>,
    pub output_permutation: Vec<usize>,
    pub output_prefix: usize,
    pub replication_axes: [Vec<usize>; 2],
    pub virtual_dimensions: Vec<usize>,
    pub indices: [Vec<usize>; 2],
    pub block_shapes: [Vec<usize>; 2],
}

#[derive(Clone, Debug)]
pub struct Selected {
    pub kind: Kind,
    pub source_id: usize,
    pub seconds: f64,
    pub memory_bytes: u64,
    pub distributions: [Distribution; 2],
    pub pattern: Pattern,
    pub execution: Execution,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    NonAscii { operand: usize },
    RankMismatch { operand: usize },
    RepeatedLabel { operand: usize, label: u8 },
    LengthMismatch { label: u8, expected: usize, actual: usize },
}

#[derive(Clone)]
struct Problem {
    shapes: [Vec<usize>; 2],
    indices: [Vec<usize>; 2],
    dimensions: Vec<usize>,
}

impl Problem {
    fn new(old: [&Distribution; 2], indices: [&str; 2]) -> Result<Self, Error> {
        let mut labels = Vec::new();
        let mut dimensions = Vec::new();
        let mut normalized: [Vec<usize>; 2] = std::array::from_fn(|_| Vec::new());
        for operand in 0..2 {
            if !indices[operand].is_ascii() {
                return Err(Error::NonAscii { operand });
            }
            if indices[operand].len() != old[operand].shape.len() {
                return Err(Error::RankMismatch { operand });
            }
            for (axis, label) in indices[operand].bytes().enumerate() {
                if indices[operand].as_bytes()[..axis].contains(&label) {
                    return Err(Error::RepeatedLabel { operand, label });
                }
                let id = if let Some(id) = labels.iter().position(|&old| old == label) {
                    if dimensions[id] != old[operand].shape[axis] {
                        return Err(Error::LengthMismatch {
                            label,
                            expected: dimensions[id],
                            actual: old[operand].shape[axis],
                        });
                    }
                    id
                } else {
                    labels.push(label);
                    dimensions.push(old[operand].shape[axis]);
                    labels.len() - 1
                };
                normalized[operand].push(id);
            }
        }
        Ok(Self {
            shapes: old.map(|distribution| distribution.shape.clone()),
            indices: normalized,
            dimensions,
        })
    }

    fn common(&self) -> Vec<usize> {
        (0..self.dimensions.len()).filter(|label| {
            self.indices[0].contains(label) && self.indices[1].contains(label)
        }).collect()
    }

    fn map_candidate(
        &self,
        topology: &Topology,
        seed: usize,
        sparse_output: bool,
    ) -> Result<[Distribution; 2], map_tensor::Rejected> {
        let common = self.common();
        let mut common_maps = vec![Mapping::Unmapped; common.len()];
        map_tensor::assign(
            &common.iter().map(|&label| self.dimensions[label]).collect::<Vec<_>>(),
            topology,
            &(0..topology.dimensions.len()).collect::<Vec<_>>(),
            &vec![false; common.len() * common.len()],
            &mut vec![false; common.len()],
            &mut common_maps,
            false,
        )?;
        let mut maps: [Vec<Mapping>; 2] = std::array::from_fn(|operand| {
            vec![Mapping::Unmapped; self.indices[operand].len()]
        });
        for (position, &label) in common.iter().enumerate() {
            for operand in 0..2 {
                let axis = self.indices[operand].iter().position(|&old| old == label).unwrap();
                maps[operand][axis] = common_maps[position].clone();
            }
        }
        assign_remaining(&self.shapes[seed], topology, &mut maps[seed], true)?;
        for &label in &common {
            let source = self.indices[seed].iter().position(|&old| old == label).unwrap();
            let other = 1 - seed;
            let destination = self.indices[other].iter().position(|&old| old == label).unwrap();
            maps[other][destination] = maps[seed][source].clone();
        }
        assign_remaining(&self.shapes[1 - seed], topology, &mut maps[1 - seed], false)?;
        if one_sided_conflict(&maps, &self.indices, self.dimensions.len(), topology.dimensions.len()) {
            return Err(map_tensor::Rejected::NoAssignableDimension);
        }
        let distributions = std::array::from_fn(|operand| Distribution::new(
            self.shapes[operand].clone(), topology.clone(), maps[operand].clone()));
        if sparse_output && used_axes(&distributions[1]).iter().any(|&used| !used) {
            return Err(map_tensor::Rejected::NoAssignableDimension);
        }
        Ok(distributions)
    }

    fn execution(&self, distributions: &[Distribution; 2], pattern: Pattern) -> Execution {
        let common = self.common();
        let input_permutation = self.indices[0].iter().map(|label| {
            let output = self.indices[1].iter().position(|candidate| candidate == label)?;
            Some(self.indices[1][..output].iter().filter(|candidate| common.contains(candidate)).count())
        }).collect();
        let output_only = self.indices[1].iter()
            .filter(|label| !self.indices[0].contains(label)).count();
        let mut mapped = 0;
        let mut common_position = 0;
        let output_permutation = self.indices[1].iter().map(|label| {
            if self.indices[0].contains(label) {
                let position = output_only + common_position;
                common_position += 1;
                position
            } else {
                let position = mapped;
                mapped += 1;
                position
            }
        }).collect();
        let block_shapes = distributions.each_ref().map(|distribution| distribution.block_shape());
        let output_prefix = self.indices[1].iter().enumerate()
            .filter(|(_, label)| !self.indices[0].contains(label))
            .map(|(axis, _)| block_shapes[1][axis]).product();
        let used = distributions.each_ref().map(|distribution| used_axes(distribution));
        let replication_axes = std::array::from_fn(|operand| {
            (0..distributions[0].topology.dimensions.len()).filter(|&axis| {
                !used[operand][axis] && used[1 - operand][axis]
            }).collect()
        });
        let virtual_dimensions = (0..self.dimensions.len()).map(|label| {
            for operand in 0..2 {
                if let Some(axis) = self.indices[operand].iter().position(|&old| old == label) {
                    return terminal_virtual(&distributions[operand].mappings[axis]);
                }
            }
            unreachable!()
        }).collect();
        Execution {
            pin_keys: [true, matches!(pattern, Pattern::SparseSparse)]
                .map(|sparse| sparse && distributions[0].topology.size() > 1),
            input_permutation,
            output_permutation,
            output_prefix,
            replication_axes,
            virtual_dimensions,
            indices: self.indices.clone(),
            block_shapes,
        }
    }
}

fn mark_axes(mapping: &Mapping, used: &mut [bool]) {
    match mapping {
        Mapping::Unmapped => {}
        Mapping::Physical { axis, child, .. } => {
            used[*axis] = true;
            mark_axes(child, used);
        }
        Mapping::Virtual { child, .. } => mark_axes(child, used),
    }
}

fn used_axes(distribution: &Distribution) -> Vec<bool> {
    let mut used = vec![false; distribution.topology.dimensions.len()];
    for mapping in &distribution.mappings { mark_axes(mapping, &mut used); }
    used
}

fn physical_npe(distribution: &Distribution) -> usize {
    used_axes(distribution).iter().zip(&distribution.topology.dimensions)
        .filter_map(|(&used, &processes)| used.then_some(processes)).product()
}

fn terminal_virtual(mapping: &Mapping) -> usize {
    match mapping {
        Mapping::Unmapped => 1,
        Mapping::Physical { child, .. } => terminal_virtual(child),
        Mapping::Virtual { copies, child } => {
            if matches!(**child, Mapping::Unmapped) { *copies } else { terminal_virtual(child) }
        }
    }
}

fn assign_remaining(
    shape: &[usize],
    topology: &Topology,
    mappings: &mut [Mapping],
    fill: bool,
) -> Result<(), map_tensor::Rejected> {
    let mut used = vec![false; topology.dimensions.len()];
    for mapping in mappings.iter() { mark_axes(mapping, &mut used); }
    let axes: Vec<_> = used.iter().enumerate()
        .filter_map(|(axis, &used)| (!used).then_some(axis)).collect();
    let mut restricted: Vec<_> = mappings.iter()
        .map(|mapping| !matches!(mapping, Mapping::Unmapped)).collect();
    map_tensor::assign(
        shape, topology, &axes, &vec![false; shape.len() * shape.len()],
        &mut restricted, mappings, fill)
}

fn one_sided_conflict(
    maps: &[Vec<Mapping>; 2],
    indices: &[Vec<usize>; 2],
    labels: usize,
    topology_order: usize,
) -> bool {
    let mut counts = vec![0; topology_order];
    for label in 0..labels {
        let present = [indices[0].contains(&label), indices[1].contains(&label)];
        if present == [true, true] { continue; }
        let operand = usize::from(!present[0]);
        let axis = indices[operand].iter().position(|&old| old == label).unwrap();
        let mut used = vec![false; topology_order];
        mark_axes(&maps[operand][axis], &mut used);
        for (count, used) in counts.iter_mut().zip(used) {
            *count += usize::from(used);
            if *count > 1 { return true; }
        }
    }
    false
}

fn source_fraction(distribution: &Distribution, nonzeros: u64) -> f64 {
    let npe = physical_npe(distribution);
    npe.min(2) as f64 * nonzeros as f64 / distribution.local_len() as f64 / npe as f64
}

fn stored_fraction(distribution: &Distribution, nonzeros: u64) -> f64 {
    let npe = physical_npe(distribution);
    1.0f64.min(nonzeros as f64 / distribution.local_len() as f64 / npe as f64)
}

#[derive(Clone)]
struct Candidate {
    source_id: usize,
    score: u64,
    seconds: f64,
    memory_bytes: u64,
}

fn add_score(score: &mut u64, term: f64) {
    *score = (*score as f64 + term) as u64;
}

#[allow(clippy::too_many_arguments)]
fn candidate(
    source_id: usize,
    old: [&Distribution; 2],
    mapped: [&Distribution; 2],
    models: &Models,
    nonzeros: [Option<u64>; 2],
    sizes: [StorageSize; 2],
    custom_reduce: bool,
    pattern: Pattern,
    execution: &Execution,
    memory_limit: u64,
) -> Option<Candidate> {
    let sparse = pattern.sparse();
    let counts = [nonzeros[0].unwrap(), nonzeros[1].unwrap_or(0)];
    let fractions = [
        stored_fraction(old[0], counts[0]),
        if sparse[1] { stored_fraction(old[1], counts[1]) } else { 1. },
    ];
    let mapped_fractions = [
        source_fraction(mapped[0], counts[0]),
        if sparse[1] { source_fraction(mapped[1], counts[1]) } else { 1. },
    ];
    let log_processes = (mapped[0].topology.size() as f64).log2().max(1.);
    let mut score = (mapped[0].local_len() + mapped[1].local_len()) as u64;
    if !redist_cost::same_mapping(old[0], mapped[0]) {
        add_score(&mut score, 25. * mapped_fractions[0] * mapped[0].local_len() as f64
            * log_processes);
    }
    if !redist_cost::same_mapping(old[1], mapped[1]) {
        if sparse[1] {
            add_score(&mut score, 50. * mapped_fractions[0].max(mapped_fractions[1])
                * mapped[1].local_len() as f64 * log_processes);
        } else if redist_cost::can_block_reshuffle(old[1], mapped[1]) {
            add_score(&mut score, mapped[1].local_len() as f64 * log_processes);
        } else {
            add_score(&mut score, 10. * mapped[1].local_len() as f64 * log_processes);
        }
    }

    let redistributions = std::array::from_fn::<_, 2, _>(|operand| {
        if sparse[operand] {
            redist_cost::sparse(old[operand], mapped[operand], sizes[operand].element_bytes,
                sizes[operand].pair_bytes, fractions[operand], models)
        } else {
            redist_cost::dense(old[operand], mapped[operand], sizes[operand].element_bytes, models)
        }
    });
    let resident: usize = (0..2).map(|operand| {
        let unit = if sparse[operand] { sizes[operand].pair_bytes }
            else { sizes[operand].element_bytes };
        (mapped[operand].local_len() as f64 * fractions[operand] * unit as f64) as usize
    }).sum();
    let sparse_buffers: usize = (0..2).filter(|&operand| sparse[operand])
        .map(|operand| (mapped[operand].local_len() as f64 * fractions[operand]
            * sizes[operand].pair_bytes as f64) as usize).sum();
    let reduction_temporary = if sparse[1] { 0 } else {
        execution.replication_axes[1].iter().next()
            .map_or(0, |_| mapped[1].local_len() * sizes[1].element_bytes)
    };
    let memory_bytes = resident + sparse_buffers
        + redistributions.iter().map(|estimate| estimate.temporary_bytes).max().unwrap()
            .max(reduction_temporary);
    if memory_bytes as u64 >= memory_limit { return None; }

    let mut seconds = redistributions.iter().map(|estimate| estimate.seconds).sum();
    for operand in 0..2 {
        if sparse[operand] {
            let passes = if operand == 1 { 2. } else { 1. };
            seconds += passes * models.get("pin_keys_mdl").estimate(&[
                1., mapped[operand].local_len() as f64 * fractions[operand]]);
        }
        let unit = if sparse[operand] { sizes[operand].pair_bytes }
            else { sizes[operand].element_bytes };
        let bytes = (mapped[operand].local_len() as f64 * fractions[operand] * unit as f64) as usize;
        for &axis in &execution.replication_axes[operand] {
            let ranks = mapped[operand].topology.dimensions[axis];
            seconds += if operand == 0 {
                models.communication(Communication::Broadcast, ranks, bytes)
            } else {
                models.communication(Communication::AllReduce { custom: custom_reduce }, ranks, bytes)
            };
        }
    }
    Some(Candidate { source_id, score, seconds, memory_bytes: memory_bytes as u64 })
}

fn select_global(context: &Context<'_>, local: Option<Candidate>) -> Option<Candidate> {
    let score = local.as_ref().map_or(f64::MAX, |candidate| candidate.score as f64);
    let source = local.as_ref().map_or(-1, |candidate| candidate.source_id as i64);
    let (scores, sources) = context.inner.gather_plan_cost(score, source);
    let mut winner = [-1i32];
    if context.rank() == 0 {
        let mut best_score = f64::MAX;
        let mut best_source = i64::MAX;
        for rank in 0..context.size() {
            if sources[rank] >= 0 && (scores[rank] < best_score
                || (scores[rank] == best_score && sources[rank] < best_source))
            {
                best_score = scores[rank];
                best_source = sources[rank];
                winner[0] = rank as i32;
            }
        }
    }
    context.broadcast(0, &mut winner);
    if winner[0] < 0 { return None; }
    let root = winner[0] as usize;
    let mut words = if context.rank() == root {
        let candidate = local.as_ref().unwrap();
        [candidate.source_id as u64, candidate.score,
            candidate.seconds.to_bits(), candidate.memory_bytes]
    } else { [0; 4] };
    context.broadcast(root, &mut words);
    Some(Candidate { source_id: words[0] as usize, score: words[1],
        seconds: f64::from_bits(words[2]), memory_bytes: words[3] })
}

#[allow(clippy::too_many_arguments)]
pub fn search(
    context: &Context<'_>,
    old: [&Distribution; 2],
    indices: [&str; 2],
    catalog: &[Topology],
    models: &Models,
    nonzeros: [Option<u64>; 2],
    sizes: [StorageSize; 2],
    custom_reduce: bool,
    pattern: Pattern,
    options: Options,
) -> Result<Option<Selected>, Error> {
    assert!(old.iter().all(|distribution| distribution.topology.size() == context.size()));
    assert!(catalog.iter().all(|topology| topology.size() == context.size()));
    let sparse = pattern.sparse();
    for operand in 0..2 { assert_eq!(nonzeros[operand].is_some(), sparse[operand]); }
    let problem = Problem::new(old, indices)?;
    let mut local = None;
    for (template, topology) in catalog.iter().enumerate() {
        for seed in 0..2 {
            let source_id = 2 * template + seed;
            if source_id % context.size() != context.rank() { continue; }
            let Ok(distributions) = problem.map_candidate(
                topology, seed, matches!(pattern, Pattern::SparseSparse)) else { continue };
            let execution = problem.execution(&distributions, pattern);
            if matches!(pattern, Pattern::SparseSparse)
                && !execution.replication_axes[1].is_empty() { continue; }
            let Some(next) = candidate(source_id, old, distributions.each_ref(), models,
                nonzeros, sizes, custom_reduce, pattern, &execution, options.memory_limit)
            else { continue };
            if local.as_ref().is_none_or(|best: &Candidate| next.score < best.score
                || (next.score == best.score && next.source_id < best.source_id)) {
                local = Some(next);
            }
        }
    }
    let Some(chosen) = select_global(context, local) else { return Ok(None) };
    let distributions = problem.map_candidate(
        &catalog[chosen.source_id / 2], chosen.source_id % 2,
        matches!(pattern, Pattern::SparseSparse)).unwrap();
    let execution = problem.execution(&distributions, pattern);
    Ok(Some(Selected { kind: Kind::Normal, source_id: chosen.source_id,
        seconds: chosen.seconds, memory_bytes: chosen.memory_bytes,
        distributions, pattern, execution }))
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Signature {
    distributions: [Distribution; 2],
    indices: [Vec<usize>; 2],
}

pub struct SearchCache<'context, 'runtime> {
    context: &'context Context<'runtime>,
    catalog: &'context [Topology],
    models: &'context Models,
    sizes: [StorageSize; 2],
    custom_reduce: bool,
    pattern: Pattern,
    options: Options,
    plans: std::collections::HashMap<Signature, Selected>,
    stats: crate::planning::CacheStats,
}

impl<'context, 'runtime> SearchCache<'context, 'runtime> {
    pub fn new(
        context: &'context Context<'runtime>,
        catalog: &'context [Topology],
        models: &'context Models,
        sizes: [StorageSize; 2],
        custom_reduce: bool,
        pattern: Pattern,
        options: Options,
    ) -> Self {
        assert!(catalog.iter().all(|topology| topology.size() == context.size()));
        Self { context, catalog, models, sizes, custom_reduce, pattern, options,
            plans: std::collections::HashMap::new(), stats: crate::planning::CacheStats::default() }
    }

    pub fn stats(&self) -> crate::planning::CacheStats { self.stats }
    pub fn len(&self) -> usize { self.plans.len() }
    pub fn is_empty(&self) -> bool { self.plans.is_empty() }
    pub fn clear(&mut self) { self.plans.clear(); }

    pub fn prepare(
        &mut self,
        old: [&Distribution; 2],
        indices: [&str; 2],
        nonzeros: [Option<u64>; 2],
    ) -> Result<Option<&Selected>, Error> {
        let problem = Problem::new(old, indices)?;
        let signature = Signature {
            distributions: old.map(Clone::clone),
            indices: problem.indices,
        };
        let context = self.context;
        let catalog = self.catalog;
        let models = self.models;
        let sizes = self.sizes;
        let custom_reduce = self.custom_reduce;
        let pattern = self.pattern;
        let options = self.options;
        match self.plans.entry(signature) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                self.stats.hits += 1;
                Ok(Some(entry.into_mut()))
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                self.stats.misses += 1;
                let Some(selected) = search(context, old, indices, catalog, models,
                    nonzeros, sizes, custom_reduce, pattern, options)? else { return Ok(None) };
                Ok(Some(entry.insert(selected)))
            }
        }
    }
}
