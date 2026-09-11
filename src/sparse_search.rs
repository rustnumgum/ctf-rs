// Adapted from cc4s CTF contraction.cxx normal/exhaustive mapping selection
// at f69cbb46e23bc2f39cda5722ce096f56301dab4f. See LICENSE.
//! Automatic selection for the unfolded sparse-A/dense-B/dense-C path.
use crate::{
    context::Context, cost::Models,
    dense_search::{CandidateCost, Kind, Objective, select_global},
    mapping::{Distribution, Topology}, mapping_variants, normal_search,
    sparse_cost::{Fractions, Storage}, sparse_mapped_cost::{self, Inputs},
};

pub use crate::dense_search::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pattern {
    SparseDenseDense { coo_kernel: bool },
    SparseSparseDense,
    SparseSparseSparse,
}

impl Pattern {
    pub(crate) fn sparse(self) -> [bool; 3] {
        match self {
            Self::SparseDenseDense { .. } => [true, false, false],
            Self::SparseSparseDense => [true, true, false],
            Self::SparseSparseSparse => [true, true, true],
        }
    }

    pub(crate) fn coo_kernel(self) -> bool {
        matches!(self, Self::SparseDenseDense { coo_kernel: true })
    }
}

#[derive(Clone, Debug)]
pub struct Selected {
    pub kind: Kind,
    pub source_id: usize,
    pub seconds: f64,
    pub memory_bytes: u64,
    pub distributions: [Distribution; 3],
    pub fold: Option<crate::partial_fold::Descriptor>,
    pub pattern: Pattern,
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub memory_limit: u64,
    pub weight: f64,
    pub allow_exhaustive: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageSize {
    pub element_bytes: usize,
    pub pair_bytes: usize,
}

fn fold_descriptor(
    mapped: [&Distribution; 3],
    indices: [&str; 3],
    models: &Models,
    fractions: Fractions,
) -> Result<Option<crate::partial_fold::Descriptor>, Error> {
    let block_shapes = mapped.map(Distribution::block_shape);
    let links = block_shapes.each_ref()
        .map(|shape| vec![crate::symmetry::Symmetry::NS; shape.len()]);
    let virtual_copies = mapped.map(|distribution| distribution.mappings.iter()
        .map(|mapping| mapping.phase() / mapping.physical_phase()).product());
    match crate::partial_fold::select_sparse(
        block_shapes.each_ref().map(Vec::as_slice),
        links.each_ref().map(Vec::as_slice),
        indices,
        models,
        virtual_copies,
        [fractions.a, fractions.b, fractions.c],
    )? {
        crate::partial_fold::Outcome::Selected(descriptor) => Ok(Some(descriptor)),
        crate::partial_fold::Outcome::Ineligible(_) => Ok(None),
    }
}

fn consider(old: [&Distribution; 3], mapped: [&Distribution; 3], indices: [&str; 3],
    models: &Models, mut inputs: Inputs, pattern: Pattern, folded: bool,
    objective: Objective) -> Result<Option<(f64, u64)>, Error> {
    let fractions = [inputs.fractions.a, inputs.fractions.b, inputs.fractions.c];
    // Source preliminary memory filter uses element widths even for sparse A.
    let resident: f64 = (0..3).map(|operand| mapped[operand].local_len() as f64
        * fractions[operand] * inputs.storage[operand].element_size as f64).sum();
    if resident >= objective.memory_limit as f64 { return Ok(None); }
    for operand in 0..3 {
        if inputs.storage[operand].sparse {
            inputs.storage[operand].dense_virtual_size = mapped[operand].block_shape().iter().product();
        }
    }
    let fold = if folded {
        fold_descriptor(mapped, indices, models, inputs.fractions)?
    } else {
        None
    };
    if folded && fold.is_none() && pattern.sparse() != [true, false, false] {
        return Ok(None);
    }
    let tree = sparse_mapped_cost::build(
        mapped,
        indices,
        inputs,
        fold.as_ref(),
        pattern.coo_kernel(),
    ).expect("sparse cost assembly failed after raw mapping preflight");
    let estimate = tree.estimate_with_redistribution(old, mapped, models, 1);
    if estimate.memory_bytes as u64 >= objective.memory_limit { return Ok(None); }
    // Only dense operands have the source INT_MAX dense-local count limit.
    if mapped.iter().enumerate().any(|(operand, distribution)|
        !inputs.storage[operand].sparse && distribution.local_len() > i32::MAX as usize)
    { return Ok(None); }
    assert!(estimate.seconds >= 0.);
    Ok(Some((estimate.seconds, estimate.memory_bytes as u64)))
}

fn pass(context: &Context<'_>, old: [&Distribution; 3], indices: [&str; 3],
    catalog: &[Topology], models: &Models, inputs: Inputs, pattern: Pattern, folded: bool,
    objective: Objective,
    exhaustive_baseline: Option<CandidateCost>) -> Result<Option<CandidateCost>, Error> {
    let shapes = old.map(|d| d.shape.as_slice());
    let mut best = exhaustive_baseline.map_or(f64::MAX, |baseline|
        if objective.weight.abs() > 1e-8 { 0. } else { baseline.seconds });
    let mut local = None;
    let mut failure = None;
    let mut visit = |source_id, mapped: [Distribution; 3]| {
        if failure.is_none() {
            match consider(old, mapped.each_ref(), indices, models, inputs, pattern, folded, objective) {
                Ok(Some((seconds, memory_bytes))) => {
                    let score = objective.score(seconds, memory_bytes);
                    if score < best {
                        best = score;
                        local = Some(CandidateCost { source_id, seconds, memory_bytes });
                    }
                }
                Ok(None) => {}
                Err(error) => failure = Some(error),
            }
        }
    };
    if exhaustive_baseline.is_some() {
        mapping_variants::visit_local_exhaustive(context, shapes, indices, catalog,
            |candidate| visit(candidate.global_id, candidate.variant.distributions))?;
    } else {
        normal_search::visit_local(context, shapes, indices, old.map(Some), catalog,
            |candidate| visit(candidate.source_id, candidate.distributions))?;
    }
    if let Some(error) = failure { return Err(error); }
    Ok(select_global(context, local, objective))
}

fn inputs(
    old: [&Distribution; 3],
    indices: [&str; 3],
    nonzeros: [Option<u64>; 3],
    output_fraction: Option<f64>,
    sizes: [StorageSize; 3],
    custom_reduce: bool,
    pattern: Pattern,
) -> Inputs {
    let sparse = pattern.sparse();
    for operand in 0..3 {
        assert_eq!(nonzeros[operand].is_some(), sparse[operand]);
    }
    let fractions = Fractions::from_layouts(old, indices, nonzeros, output_fraction);
    let inputs = Inputs {
        storage: std::array::from_fn(|operand| Storage { sparse: sparse[operand],
            element_size: sizes[operand].element_bytes,
            pair_size: sizes[operand].pair_bytes,
            dense_virtual_size: 0,
            custom_addition: custom_reduce }),
        fractions, custom: false,
    };
    inputs
}

#[allow(clippy::too_many_arguments)]
fn search_with(
    context: &Context<'_>,
    old: [&Distribution; 3],
    indices: [&str; 3],
    catalog: &[Topology],
    models: &Models,
    inputs: Inputs,
    pattern: Pattern,
    folded: bool,
    options: Options,
) -> Result<Option<Selected>, Error> {
    let Some(initial) = pass(context, old, indices, catalog, models, inputs, pattern, folded,
        Objective::time_only(options.memory_limit), None)? else { return Ok(None) };
    let normal = if options.weight.abs() > 1e-8 {
        pass(context, old, indices, catalog, models, inputs, pattern, folded, Objective {
            memory_limit: options.memory_limit, weight: options.weight,
            baseline_seconds: initial.seconds, baseline_memory: initial.memory_bytes,
        }, None)?.expect("time-pass winner must remain a weighted-pass candidate")
    } else { initial };
    let mut chosen = normal;
    let mut kind = Kind::Normal;
    if options.allow_exhaustive && normal.seconds >= 0.01 {
        if let Some(candidate) = pass(context, old, indices, catalog, models, inputs, pattern, folded, Objective {
            memory_limit: options.memory_limit, weight: options.weight,
            baseline_seconds: normal.seconds, baseline_memory: normal.memory_bytes,
        }, Some(normal))? {
            // Pinned final refinement uses time, even after weighted searches.
            if candidate.seconds < normal.seconds { chosen = candidate; kind = Kind::Exhaustive; }
        }
    }
    let shapes = old.map(|d| d.shape.as_slice());
    let distributions = match kind {
        Kind::Normal => normal_search::reconstruct(chosen.source_id, shapes, indices,
            old.map(Some), catalog)?.distributions,
        Kind::Exhaustive => mapping_variants::reconstruct_exhaustive(chosen.source_id,
            shapes, indices, catalog)?.variant.distributions,
    };
    let fold = if folded {
        fold_descriptor(distributions.each_ref(), indices, models, inputs.fractions)?
    } else {
        None
    };
    Ok(Some(Selected { kind, source_id: chosen.source_id, seconds: chosen.seconds,
        memory_bytes: chosen.memory_bytes, distributions, fold, pattern }))
}

/// Search source raw sparse candidates. Sparse counts are global canonical
/// stored-entry counts in the original layouts; dense operands use `None`.
#[allow(clippy::too_many_arguments)]
pub fn search(
    context: &Context<'_>,
    old: [&Distribution; 3],
    indices: [&str; 3],
    catalog: &[Topology],
    models: &Models,
    nonzeros: [Option<u64>; 3],
    output_fraction: Option<f64>,
    sizes: [StorageSize; 3],
    custom_reduce: bool,
    pattern: Pattern,
    options: Options,
) -> Result<Option<Selected>, Error> {
    search_with(
        context,
        old,
        indices,
        catalog,
        models,
        inputs(old, indices, nonzeros, output_fraction, sizes,
            custom_reduce, pattern),
        pattern,
        true,
        options,
    )
}

/// Fixed C1 unfolded sparse-A/dense-B/dense-C entry point.
pub fn search_unfolded(context: &Context<'_>, old: [&Distribution; 3],
    indices: [&str; 3], catalog: &[Topology], models: &Models,
    nonzeros_a: u64, sizes: [StorageSize; 3],
    custom_reduce: bool, options: Options) -> Result<Option<Selected>, Error> {
    let pattern = Pattern::SparseDenseDense { coo_kernel: false };
    search_with(
        context,
        old,
        indices,
        catalog,
        models,
        inputs(old, indices, [Some(nonzeros_a), None, None], None, sizes,
            custom_reduce, pattern),
        pattern,
        false,
        options,
    )
}

/// Context-scoped raw sparse search cache. Storage kind, leaf capability and
/// immutable model bank are fixed at construction. Counts and values are not
/// cache keys, matching the source contraction-signature cache.
pub struct SearchCache<'context, 'runtime> {
    context: &'context Context<'runtime>,
    catalog: &'context [Topology],
    models: &'context Models,
    sizes: [StorageSize; 3],
    custom_reduce: bool,
    pattern: Pattern,
    options: Options,
    plans: std::collections::HashMap<crate::planning::Signature, Selected>,
    stats: crate::planning::CacheStats,
}

impl<'context, 'runtime> SearchCache<'context, 'runtime> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &'context Context<'runtime>,
        catalog: &'context [Topology],
        models: &'context Models,
        sizes: [StorageSize; 3],
        custom_reduce: bool,
        pattern: Pattern,
        options: Options,
    ) -> Self {
        assert!(catalog.iter().all(|topology| topology.size() == context.size()));
        Self {
            context,
            catalog,
            models,
            sizes,
            custom_reduce,
            pattern,
            options,
            plans: std::collections::HashMap::new(),
            stats: crate::planning::CacheStats::default(),
        }
    }

    pub fn stats(&self) -> crate::planning::CacheStats { self.stats }
    pub fn len(&self) -> usize { self.plans.len() }
    pub fn is_empty(&self) -> bool { self.plans.is_empty() }
    pub fn clear(&mut self) { self.plans.clear(); }

    /// A miss is collective; a hit is local. All ranks must prepare and clear
    /// caches in the same sequence.
    pub fn prepare(
        &mut self,
        old: [&Distribution; 3],
        indices: [&str; 3],
        nonzeros: [Option<u64>; 3],
        output_fraction: Option<f64>,
    ) -> Result<Option<&Selected>, Error> {
        assert!(old.iter().all(|distribution| distribution.topology.size() == self.context.size()));
        let signature = crate::planning::Signature::new(old, indices, old[0].topology.clone());
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
                let Some(selected) = search(
                    context,
                    old,
                    indices,
                    catalog,
                    models,
                    nonzeros,
                    output_fraction,
                    sizes,
                    custom_reduce,
                    pattern,
                    options,
                )? else { return Ok(None) };
                Ok(Some(entry.insert(selected)))
            }
        }
    }
}
