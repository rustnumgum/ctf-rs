// Adapted from cc4s contraction/contraction.cxx, normal/exhaustive mapping
// selection and refinement (lines 2834-3190 and 3280-3342).
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Collective dense mapping search. This module deliberately excludes the
//! aligned `GridPlan` search space. SearchCache retains selected
//! raw layouts for a fixed context and immutable search configuration.

use crate::{
    context::Context,
    cost::Models,
    mapped_cost,
    mapping::{Distribution, Topology},
    mapping_variants,
    normal_mapping::ProblemError,
    normal_search,
};

#[derive(Clone, Debug)]
pub struct TopologyFacts {
    pub topology: Topology,
    pub nodes_per_axis: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub memory_limit: u64,
    pub weight: f64,
    pub allow_exhaustive: bool,
    pub enable_folding: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Normal,
    Exhaustive,
}

#[derive(Clone, Debug)]
pub struct Selected {
    pub kind: Kind,
    pub source_id: usize,
    pub seconds: f64,
    pub memory_bytes: u64,
    pub distributions: [Distribution; 3],
    pub fold: Option<crate::partial_fold::Descriptor>,
}

/// Context-scoped cache for this search configuration. Models and topology facts
/// are borrowed immutably so a stored estimate cannot silently acquire different
/// coefficients or node counts. Alpha, beta and tensor values are not cache keys.
///
/// A miss performs collective search; a hit and clear are local. As with source
/// collective planning, ranks must call prepare/clear in the same sequence.
pub struct SearchCache<'context,'runtime> {
    context: &'context Context<'runtime>,
    catalog: &'context [TopologyFacts],
    models: &'context Models,
    element_bytes: usize,
    local_custom: bool,
    custom_reduce: bool,
    options: Options,
    plans: std::collections::HashMap<(crate::planning::Signature,[Vec<u64>;3]),Selected>,
    stats: crate::planning::CacheStats,
}
impl<'context,'runtime> SearchCache<'context,'runtime> {
    pub fn new(context:&'context Context<'runtime>,catalog:&'context [TopologyFacts],
        models:&'context Models,element_bytes:usize,local_custom:bool,custom_reduce:bool,options:Options)->Self{
        Self{context,catalog,models,element_bytes,local_custom,custom_reduce,options,
            plans:std::collections::HashMap::new(),stats:crate::planning::CacheStats::default()}
    }
    pub fn stats(&self)->crate::planning::CacheStats{self.stats}
    pub fn len(&self)->usize{self.plans.len()}
    pub fn is_empty(&self)->bool{self.plans.is_empty()}
    pub fn clear(&mut self){self.plans.clear();}
    pub fn prepare(&mut self,old:[&Distribution;3],old_nodes:[&[f64];3],indices:[&str;3])
        ->Result<Option<&Selected>,Error>{
        let signature=crate::planning::Signature::new(old,indices,old[0].topology.clone());
        let key=(signature,old_nodes.map(|nodes|nodes.iter().map(|value|value.to_bits()).collect()));
        match self.plans.entry(key){
            std::collections::hash_map::Entry::Occupied(entry)=>{
                self.stats.hits+=1;Ok(Some(entry.into_mut()))
            },
            std::collections::hash_map::Entry::Vacant(entry)=>{
                self.stats.misses+=1;
                let Some(selected)=search_dense(self.context,old,old_nodes,indices,self.catalog,self.models,
                    self.element_bytes,self.local_custom,self.custom_reduce,self.options)?else{return Ok(None)};
                Ok(Some(entry.insert(selected)))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Normal(ProblemError),
    Exhaustive(mapping_variants::Rejected),
    NodeFactRankMismatch,
    ConflictingNodeFacts,
    MissingNodeFacts,
    InvalidNodeFacts,
    Fold(crate::partial_fold::Error),
}

impl From<ProblemError> for Error {
    fn from(value: ProblemError) -> Self {
        Self::Normal(value)
    }
}

impl From<mapping_variants::Rejected> for Error {
    fn from(value: mapping_variants::Rejected) -> Self {
        Self::Exhaustive(value)
    }
}

impl From<crate::partial_fold::Error> for Error {
    fn from(value: crate::partial_fold::Error) -> Self {
        Self::Fold(value)
    }
}

#[derive(Clone, Copy)]
struct Objective {
    memory_limit: u64,
    weight: f64,
    baseline_seconds: f64,
    baseline_memory: u64,
}

impl Objective {
    fn time_only(memory_limit: u64) -> Self {
        Self {
            memory_limit,
            weight: 0.,
            baseline_seconds: 0.,
            baseline_memory: 0,
        }
    }

    fn score(self, seconds: f64, memory: u64) -> f64 {
        if self.weight.abs() > 1e-8 {
            assert!(self.baseline_seconds > 0. && self.baseline_memory > 0);
            (seconds - self.baseline_seconds) / self.baseline_seconds
                + self.weight * (memory as f64 - self.baseline_memory as f64)
                    / self.baseline_memory as f64
        } else {
            seconds
        }
    }
}

#[derive(Clone, Copy)]
struct CandidateCost {
    source_id: usize,
    seconds: f64,
    memory_bytes: u64,
}

struct NodeFacts<'a> {
    old: [(&'a Topology, &'a [f64]); 3],
    catalog: &'a [TopologyFacts],
}

impl<'a> NodeFacts<'a> {
    fn new(
        old: [&'a Distribution; 3],
        old_nodes: [&'a [f64]; 3],
        catalog: &'a [TopologyFacts],
    ) -> Result<Self, Error> {
        for operand in 0..3 {
            if old_nodes[operand].len() != old[operand].topology.dimensions.len() {
                return Err(Error::NodeFactRankMismatch);
            }
            if old_nodes[operand]
                .iter()
                .any(|value| !value.is_finite() || *value < 0.)
            {
                return Err(Error::InvalidNodeFacts);
            }
        }
        for facts in catalog {
            if facts.nodes_per_axis.len() != facts.topology.dimensions.len() {
                return Err(Error::NodeFactRankMismatch);
            }
            if facts
                .nodes_per_axis
                .iter()
                .any(|value| !value.is_finite() || *value < 0.)
            {
                return Err(Error::InvalidNodeFacts);
            }
        }
        let this = Self {
            old: std::array::from_fn(|operand| {
                (&old[operand].topology, old_nodes[operand])
            }),
            catalog,
        };
        for operand in 0..3 {
            this.get(this.old[operand].0)?;
        }
        for facts in catalog {
            this.get(&facts.topology)?;
        }
        Ok(this)
    }

    fn get(&self, topology: &Topology) -> Result<&'a [f64], Error> {
        let mut found: Option<&[f64]> = None;
        for (candidate, nodes) in self.old.iter().copied().chain(
            self.catalog
                .iter()
                .map(|facts| (&facts.topology, facts.nodes_per_axis.as_slice())),
        ) {
            if candidate == topology {
                if found.is_some_and(|old| old != nodes) {
                    return Err(Error::ConflictingNodeFacts);
                }
                found = Some(nodes);
            }
        }
        found.ok_or(Error::MissingNodeFacts)
    }
}

fn consider(
    old: [&Distribution; 3],
    mapped: [&Distribution; 3],
    indices: [&str; 3],
    nodes_per_axis: &[f64],
    models: &Models,
    element_bytes: usize,
    local_custom: bool,
    custom_reduce: bool,
    enable_folding: bool,
    objective: Objective,
) -> Option<(f64, u64)> {
    // Source deliberately uses double here before detailed integer memory work.
    let mapped_resident_bytes: f64 = mapped
        .iter()
        .map(|distribution| distribution.local_len() as f64 * element_bytes as f64)
        .sum();
    if mapped_resident_bytes >= objective.memory_limit as f64 {
        return None;
    }
    let (seconds, memory_bytes) = if enable_folding && !local_custom {
        match crate::folded_cost::estimate_dense_folded(
            old,
            mapped,
            indices,
            models,
            element_bytes,
            nodes_per_axis,
            custom_reduce,
        )
        .expect("folded candidate estimation failed after mapping preflight")
        {
            Some(estimate) => (estimate.seconds, estimate.memory_bytes),
            None => {
                let estimate = mapped_cost::estimate_dense_unfolded(
                    old,
                    mapped,
                    indices,
                    models,
                    element_bytes,
                    nodes_per_axis,
                    local_custom,
                    custom_reduce,
                );
                (estimate.seconds, estimate.memory_bytes)
            }
        }
    } else {
        let estimate = mapped_cost::estimate_dense_unfolded(
            old,
            mapped,
            indices,
            models,
            element_bytes,
            nodes_per_axis,
            local_custom,
            custom_reduce,
        );
        (estimate.seconds, estimate.memory_bytes)
    };
    if memory_bytes as u64 >= objective.memory_limit {
        return None;
    }
    if mapped
        .iter()
        .any(|distribution| distribution.local_len() > i32::MAX as usize)
    {
        return None;
    }
    assert!(seconds >= 0.);
    Some((seconds, memory_bytes as u64))
}

fn select_global(
    context: &Context<'_>,
    local: Option<CandidateCost>,
    objective: Objective,
) -> Option<CandidateCost> {
    let seconds = local.map_or(f64::MAX, |candidate| candidate.seconds);
    let memory = local.map_or(-1, |candidate| candidate.memory_bytes.try_into().unwrap());
    let (times, memories) = context.inner.gather_plan_cost(seconds, memory);
    let mut winner = [-1i32];
    if context.rank() == 0 {
        let mut best = f64::MAX;
        for rank in 0..context.size() {
            if memories[rank] < 0 {
                continue;
            }
            let score = objective.score(times[rank], memories[rank] as u64);
            if score < best {
                best = score;
                winner[0] = rank as i32;
            }
        }
    }
    context.broadcast(0, &mut winner);
    if winner[0] < 0 {
        return None;
    }
    let root = winner[0] as usize;
    let mut words = if context.rank() == root {
        let candidate = local.unwrap();
        [
            candidate.source_id as u64,
            candidate.seconds.to_bits(),
            candidate.memory_bytes,
        ]
    } else {
        [0; 3]
    };
    context.broadcast(root, &mut words);
    Some(CandidateCost {
        source_id: words[0] as usize,
        seconds: f64::from_bits(words[1]),
        memory_bytes: words[2],
    })
}

#[allow(clippy::too_many_arguments)]
fn normal_pass(
    context: &Context<'_>,
    old: [&Distribution; 3],
    indices: [&str; 3],
    catalog: &[Topology],
    facts: &NodeFacts<'_>,
    models: &Models,
    element_bytes: usize,
    local_custom: bool,
    custom_reduce: bool,
    enable_folding: bool,
    objective: Objective,
) -> Result<Option<CandidateCost>, Error> {
    let shapes = old.map(|distribution| distribution.shape.as_slice());
    let mut local = None;
    let mut best = f64::MAX;
    normal_search::visit_local(
        context,
        shapes,
        indices,
        old.map(Some),
        catalog,
        |candidate| {
            let nodes = facts.get(&candidate.distributions[0].topology).unwrap();
            if let Some((seconds, memory_bytes)) = consider(
                old,
                candidate.distributions.each_ref(),
                indices,
                nodes,
                models,
                element_bytes,
                local_custom,
                custom_reduce,
                enable_folding,
                objective,
            ) {
                let score = objective.score(seconds, memory_bytes);
                if score < best {
                    best = score;
                    local = Some(CandidateCost {
                        source_id: candidate.source_id,
                        seconds,
                        memory_bytes,
                    });
                }
            }
        },
    )?;
    Ok(select_global(context, local, objective))
}

#[allow(clippy::too_many_arguments)]
fn exhaustive_pass(
    context: &Context<'_>,
    old: [&Distribution; 3],
    indices: [&str; 3],
    catalog: &[Topology],
    facts: &NodeFacts<'_>,
    models: &Models,
    element_bytes: usize,
    local_custom: bool,
    custom_reduce: bool,
    enable_folding: bool,
    objective: Objective,
    baseline: CandidateCost,
) -> Result<Option<CandidateCost>, Error> {
    let shapes = old.map(|distribution| distribution.shape.as_slice());
    let mut best = if objective.weight.abs() > 1e-8 {
        0.
    } else {
        baseline.seconds
    };
    let mut local = None;
    mapping_variants::visit_local_exhaustive(
        context,
        shapes,
        indices,
        catalog,
        |candidate| {
            let nodes = facts
                .get(&candidate.variant.distributions[0].topology)
                .unwrap();
            if let Some((seconds, memory_bytes)) = consider(
                old,
                candidate.variant.distributions.each_ref(),
                indices,
                nodes,
                models,
                element_bytes,
                local_custom,
                custom_reduce,
                enable_folding,
                objective,
            ) {
                let score = objective.score(seconds, memory_bytes);
                if score < best {
                    best = score;
                    local = Some(CandidateCost {
                        source_id: candidate.global_id,
                        seconds,
                        memory_bytes,
                    });
                }
            }
        },
    )?;
    Ok(select_global(context, local, objective))
}

/// Perform the source normal search, its optional weighted second pass, and
/// optional exhaustive refinement. `old_nodes` supplies facts for retained old
/// topologies, which need not occur in `catalog`; facts for equal topologies
/// must agree. An empty result means no normal candidate survived mapping,
/// preflight, strict memory bounds and the dense-local `INT_MAX` limit.
#[allow(clippy::too_many_arguments)]
pub fn search_dense(
    context: &Context<'_>,
    old: [&Distribution; 3],
    old_nodes: [&[f64]; 3],
    indices: [&str; 3],
    catalog: &[TopologyFacts],
    models: &Models,
    element_bytes: usize,
    local_custom: bool,
    custom_reduce: bool,
    options: Options,
) -> Result<Option<Selected>, Error> {
    let facts = NodeFacts::new(old, old_nodes, catalog)?;
    let topologies: Vec<_> = catalog.iter().map(|facts| facts.topology.clone()).collect();
    let time_objective = Objective::time_only(options.memory_limit);
    let Some(initial) = normal_pass(
        context,
        old,
        indices,
        &topologies,
        &facts,
        models,
        element_bytes,
        local_custom,
        custom_reduce,
        options.enable_folding,
        time_objective,
    )? else {
        return Ok(None);
    };
    let mut normal = initial;
    if options.weight.abs() > 1e-8 {
        let normal_objective = Objective {
            memory_limit: options.memory_limit,
            weight: options.weight,
            baseline_seconds: initial.seconds,
            baseline_memory: initial.memory_bytes,
        };
        normal = normal_pass(
            context,
            old,
            indices,
            &topologies,
            &facts,
            models,
            element_bytes,
            local_custom,
            custom_reduce,
            options.enable_folding,
            normal_objective,
        )?
        .expect("time-pass winner must remain a weighted-pass candidate");
    }

    let mut kind = Kind::Normal;
    let mut chosen = normal;
    if options.allow_exhaustive && normal.seconds >= 0.01 {
        let refinement_objective = Objective {
            memory_limit: options.memory_limit,
            weight: options.weight,
            baseline_seconds: normal.seconds,
            baseline_memory: normal.memory_bytes,
        };
        if let Some(exhaustive) = exhaustive_pass(
            context,
            old,
            indices,
            &topologies,
            &facts,
            models,
            element_bytes,
            local_custom,
            custom_reduce,
            options.enable_folding,
            refinement_objective,
            normal,
        )? {
            // The pinned source makes its final refinement choice by time even
            // when the searches themselves used the weighted objective.
            if exhaustive.seconds < normal.seconds {
                kind = Kind::Exhaustive;
                chosen = exhaustive;
            }
        }
    }

    let shapes = old.map(|distribution| distribution.shape.as_slice());
    let distributions = match kind {
        Kind::Normal => normal_search::reconstruct(
            chosen.source_id,
            shapes,
            indices,
            old.map(Some),
            &topologies,
        )?
        .distributions,
        Kind::Exhaustive => mapping_variants::reconstruct_exhaustive(
            chosen.source_id,
            shapes,
            indices,
            &topologies,
        )?
        .variant
        .distributions,
    };
    let fold = if options.enable_folding && !local_custom {
        let block_shapes = distributions.each_ref().map(|distribution| distribution.block_shape());
        let links: [Vec<crate::symmetry::Symmetry>; 3] = std::array::from_fn(|operand| {
            vec![crate::symmetry::Symmetry::NS; block_shapes[operand].len()]
        });
        let virtual_copies = distributions.each_ref().map(|distribution| {
            distribution
                .mappings
                .iter()
                .map(|mapping| mapping.phase() / mapping.physical_phase())
                .product()
        });
        match crate::partial_fold::select(
            block_shapes.each_ref().map(Vec::as_slice),
            links.each_ref().map(Vec::as_slice),
            indices,
            models,
            virtual_copies,
        )? {
            crate::partial_fold::Outcome::Selected(descriptor) => Some(descriptor),
            crate::partial_fold::Outcome::Ineligible(_) => None,
        }
    } else {
        None
    };
    Ok(Some(Selected {
        kind,
        source_id: chosen.source_id,
        seconds: chosen.seconds,
        memory_bytes: chosen.memory_bytes,
        distributions,
        fold,
    }))
}
