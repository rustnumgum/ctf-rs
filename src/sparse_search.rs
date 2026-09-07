// Adapted from cc4s CTF contraction.cxx normal/exhaustive mapping selection
// at f69cbb46e23bc2f39cda5722ce096f56301dab4f. See LICENSE.
//! Automatic selection for the unfolded sparse-A/dense-B/dense-C path.
use crate::{
    context::Context, cost::Models,
    dense_search::{CandidateCost, Error, Kind, Objective, Selected, select_global},
    mapping::{Distribution, Topology}, mapping_variants, normal_search,
    sparse_cost::{Fractions, Storage}, sparse_mapped_cost::{self, Inputs},
};

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub memory_limit: u64,
    pub weight: f64,
    pub allow_exhaustive: bool,
}

fn consider(old: [&Distribution; 3], mapped: [&Distribution; 3], indices: [&str; 3],
    models: &Models, mut inputs: Inputs, objective: Objective) -> Option<(f64, u64)> {
    let fractions = [inputs.fractions.a, inputs.fractions.b, inputs.fractions.c];
    // Source preliminary memory filter uses element widths even for sparse A.
    let resident: f64 = (0..3).map(|operand| mapped[operand].local_len() as f64
        * fractions[operand] * inputs.storage[operand].element_size as f64).sum();
    if resident >= objective.memory_limit as f64 { return None; }
    inputs.storage[0].dense_virtual_size = mapped[0].block_shape().iter().product();
    let tree = sparse_mapped_cost::build_unfolded(mapped, indices, inputs)
        .expect("sparse cost assembly failed after raw mapping preflight");
    let estimate = tree.estimate_with_redistribution(old, mapped, models, 1);
    if estimate.memory_bytes as u64 >= objective.memory_limit { return None; }
    // Only dense operands have the source INT_MAX dense-local count limit.
    if mapped[1..].iter().any(|d| d.local_len() > i32::MAX as usize) { return None; }
    assert!(estimate.seconds >= 0.);
    Some((estimate.seconds, estimate.memory_bytes as u64))
}

fn pass(context: &Context<'_>, old: [&Distribution; 3], indices: [&str; 3],
    catalog: &[Topology], models: &Models, inputs: Inputs, objective: Objective,
    exhaustive_baseline: Option<CandidateCost>) -> Result<Option<CandidateCost>, Error> {
    let shapes = old.map(|d| d.shape.as_slice());
    let mut best = exhaustive_baseline.map_or(f64::MAX, |baseline|
        if objective.weight.abs() > 1e-8 { 0. } else { baseline.seconds });
    let mut local = None;
    let mut visit = |source_id, mapped: [Distribution; 3]| {
        if let Some((seconds, memory_bytes)) = consider(old, mapped.each_ref(), indices,
            models, inputs, objective) {
            let score = objective.score(seconds, memory_bytes);
            if score < best {
                best = score;
                local = Some(CandidateCost { source_id, seconds, memory_bytes });
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
    Ok(select_global(context, local, objective))
}

/// Collective source normal search, optional weighted pass and exhaustive
/// refinement. Counts refer to canonical stored entries in the original layout.
/// Values/coefficients are not retained; the selected raw distributions can be
/// passed directly to `Tensor::contract_sparse_from_mapped`.
pub fn search_unfolded(context: &Context<'_>, old: [&Distribution; 3],
    indices: [&str; 3], catalog: &[Topology], models: &Models,
    nonzeros_a: u64, element_bytes: usize, pair_bytes: usize,
    custom_reduce: bool, options: Options) -> Result<Option<Selected>, Error> {
    let fractions = Fractions::from_layouts(old, indices, [Some(nonzeros_a), None, None], None);
    let inputs = Inputs {
        storage: std::array::from_fn(|operand| Storage { sparse: operand == 0,
            element_size: element_bytes, pair_size: pair_bytes, dense_virtual_size: 0,
            custom_addition: custom_reduce }),
        fractions, custom: false,
    };
    let Some(initial) = pass(context, old, indices, catalog, models, inputs,
        Objective::time_only(options.memory_limit), None)? else { return Ok(None) };
    let normal = if options.weight.abs() > 1e-8 {
        pass(context, old, indices, catalog, models, inputs, Objective {
            memory_limit: options.memory_limit, weight: options.weight,
            baseline_seconds: initial.seconds, baseline_memory: initial.memory_bytes,
        }, None)?.expect("time-pass winner must remain a weighted-pass candidate")
    } else { initial };
    let mut chosen = normal;
    let mut kind = Kind::Normal;
    if options.allow_exhaustive && normal.seconds >= 0.01 {
        if let Some(candidate) = pass(context, old, indices, catalog, models, inputs, Objective {
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
    Ok(Some(Selected { kind, source_id: chosen.source_id, seconds: chosen.seconds,
        memory_bytes: chosen.memory_bytes, distributions, fold: None }))
}
