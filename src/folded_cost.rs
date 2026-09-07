// Adapted from cc4s contraction/contraction.cxx::detail_estimate_mem_and_time
// and contraction/ctr_tsr.cxx::est_{membw,fp}.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Dense folded cost for a preflight-valid raw NS mapping. Ineligible folds
//! return `None`; this module does not substitute an unfolded estimate.

use crate::{
    cost::Models,
    mapping::Distribution,
    partial_fold::{self, Descriptor, Outcome},
    plan_cost::{Estimate, Tree},
    symmetry::Symmetry,
};

#[derive(Clone, Debug)]
pub struct FoldedEstimate {
    pub descriptor: Descriptor,
    pub tree: Tree,
    pub inner: Estimate,
    pub transpose_seconds: [f64; 3],
    pub redistribution_seconds: [f64; 3],
    pub redistributed_input_bytes: usize,
    pub redistribution_temporary_bytes: usize,
    pub fold_resident_bytes: usize,
    pub fold_temporary_bytes: usize,
    pub seconds: f64,
    pub memory_bytes: usize,
}

fn replace_local(tree: Tree, operand_bytes: [usize; 3], flops: f64) -> Tree {
    match tree {
        Tree::Local { .. } => Tree::Local {
            custom: false,
            folded: true,
            operand_bytes,
            flops,
        },
        Tree::Virtual {
            phases,
            orders,
            child,
        } => Tree::Virtual {
            phases,
            orders,
            child: Box::new(replace_local(*child, operand_bytes, flops)),
        },
        Tree::Replicated {
            inputs,
            output,
            custom_reduce,
            child,
        } => Tree::Replicated {
            inputs,
            output,
            custom_reduce,
            child: Box::new(replace_local(*child, operand_bytes, flops)),
        },
        Tree::Panels {
            steps,
            panel_bytes,
            movement,
            custom_reduce,
            child,
        } => Tree::Panels {
            steps,
            panel_bytes,
            movement,
            custom_reduce,
            child: Box::new(replace_local(*child, operand_bytes, flops)),
        },
    }
}

fn residual_shapes(
    block_shapes: &[Vec<usize>; 3],
    descriptor: &Descriptor,
) -> [Vec<usize>; 3] {
    std::array::from_fn(|operand| {
        let mut shape = block_shapes[operand].clone();
        let selected = descriptor.layouts[operand].folded_shape.len();
        for &axis in &descriptor.layouts[operand].inner_ordering[..selected] {
            // All links are NS, so each source symmetry group is one axis.
            shape[axis] = 1;
        }
        shape
    })
}

fn normalized_extents(residual: &[Vec<usize>; 3], indices: [&str; 3]) -> Vec<usize> {
    let mut labels = Vec::new();
    let mut extents = Vec::new();
    for operand in 0..3 {
        for (axis, label) in indices[operand].bytes().enumerate() {
            if let Some(id) = labels.iter().position(|&old| old == label) {
                assert_eq!(extents[id], residual[operand][axis]);
            } else {
                labels.push(label);
                extents.push(residual[operand][axis]);
            }
        }
    }
    extents
}

/// Estimate the source dense folded branch for one raw mapped candidate.
/// The communication tree is constructed before the sequential folded leaf is
/// collapsed, just as construct_dense_ctr builds replication/panels/virtuality
/// around seq_tsr_ctr. Source intentionally omits `l` from leaf traffic/flops.
#[allow(clippy::too_many_arguments)]
pub fn estimate_dense_folded(
    old: [&Distribution; 3],
    mapped: [&Distribution; 3],
    indices: [&str; 3],
    models: &Models,
    element_bytes: usize,
    nodes_per_axis: &[usize],
    custom_reduce: bool,
) -> Result<Option<FoldedEstimate>, partial_fold::Error> {
    let block_shapes = mapped.map(Distribution::block_shape);
    let links: [Vec<Symmetry>; 3] =
        std::array::from_fn(|operand| vec![Symmetry::NS; block_shapes[operand].len()]);
    let virtual_copies = mapped.map(|distribution| {
        distribution
            .mappings
            .iter()
            .map(|mapping| mapping.phase() / mapping.physical_phase())
            .product()
    });
    let descriptor = match partial_fold::select(
        block_shapes.each_ref().map(Vec::as_slice),
        links.each_ref().map(Vec::as_slice),
        indices,
        models,
        virtual_copies,
    )? {
        Outcome::Selected(descriptor) => descriptor,
        Outcome::Ineligible(_) => return Ok(None),
    };

    let residual = residual_shapes(&block_shapes, &descriptor);
    let residual_sizes = residual
        .each_ref()
        .map(|shape| shape.iter().product::<usize>());
    // ctr_tsr.cxx:407-418: residual sy_packed_size times matrix dimensions,
    // deliberately without inner_params.l.
    let operand_bytes = [
        residual_sizes[0] * descriptor.m * descriptor.k * element_bytes,
        residual_sizes[1] * descriptor.n * descriptor.k * element_bytes,
        residual_sizes[2] * descriptor.m * descriptor.n * element_bytes,
    ];
    // ctr_tsr.cxx:421-441 likewise omits l, then multiplies every residual
    // normalized index extent selected from A, otherwise B, otherwise C.
    let mut flops = 2. * descriptor.m as f64 * descriptor.n as f64 * descriptor.k as f64;
    for extent in normalized_extents(&residual,indices){flops*=extent as f64;}
    let tree = replace_local(
        crate::mapped_cost::dense_unfolded(
            mapped,
            indices,
            element_bytes,
            nodes_per_axis,
            false,
            custom_reduce,
        ),
        operand_bytes,
        flops,
    );
    let inner = tree.estimate(models, 1);
    let transpose_seconds = descriptor.transpose_seconds;

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
    let fold_resident_bytes = mapped
        .iter()
        .map(|distribution| distribution.local_len() * element_bytes)
        .sum();
    let fold_temporary_bytes = 0;
    let mut seconds = transpose_seconds[0]+transpose_seconds[1]+transpose_seconds[2];
    seconds+=inner.seconds;
    for time in redistribution_seconds{seconds+=time;}
    let memory_bytes = redistributed_input_bytes
        + redistribution_temporary_bytes
            .max(fold_temporary_bytes.max(fold_resident_bytes + inner.working_bytes));
    Ok(Some(FoldedEstimate {
        descriptor,
        tree,
        inner,
        transpose_seconds,
        redistribution_seconds,
        redistributed_input_bytes,
        redistribution_temporary_bytes,
        fold_resident_bytes,
        fold_temporary_bytes,
        seconds,
        memory_bytes,
    }))
}
