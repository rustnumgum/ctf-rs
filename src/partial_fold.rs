// Adapted from cc4s contraction/contraction.cxx::{get_fold_ctr,
// select_ctr_perm,map_fold}. Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Dense partial/symmetric fold metadata. No folded storage conversion or
//! partial-fold execution is implied by this descriptor.

use crate::{
    cost::Models,
    fold_indices,
    fold_layout::FoldLayout,
    folding::Operand,
    linalg::Transpose,
    symmetry::Symmetry,
};

const PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [2, 0, 1],
    [1, 2, 0],
    [1, 0, 2],
    [2, 1, 0],
];
const TRANSPOSES: [(Transpose, Transpose, bool); 6] = [
    (Transpose::Yes, Transpose::No, false),
    (Transpose::No, Transpose::No, false),
    (Transpose::No, Transpose::Yes, false),
    (Transpose::No, Transpose::No, true),
    (Transpose::No, Transpose::Yes, true),
    (Transpose::Yes, Transpose::No, true),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    FoldIndices(fold_indices::Error),
    ShapeRankMismatch {
        operand: Operand,
        shape: usize,
        indices: usize,
    },
    FoldedLengthMismatch { label: usize },
    SizeOverflow,
}

impl From<fold_indices::Error> for Error {
    fn from(value: fold_indices::Error) -> Self {
        Self::FoldIndices(value)
    }
}

#[derive(Clone, Debug)]
pub enum Outcome {
    Selected(Descriptor),
    Ineligible(fold_indices::Eligibility),
}

#[derive(Clone, Debug)]
pub struct Descriptor {
    pub fold_labels: Vec<usize>,
    pub layouts: [FoldLayout; 3],
    pub permutation: usize,
    /// Source transpose estimates mapped back to original A/B/C. The operand
    /// occupying the selected third role already includes its factor of two.
    pub transpose_seconds: [f64; 3],
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batches: usize,
    pub trans_a: Transpose,
    pub trans_b: Transpose,
    pub transposed_output: bool,
}

fn normalized(indices: [&str; 3]) -> [Vec<usize>; 3] {
    let mut labels = Vec::new();
    std::array::from_fn(|operand| {
        indices[operand]
            .bytes()
            .map(|label| {
                if let Some(id) = labels.iter().position(|&old| old == label) {
                    id
                } else {
                    labels.push(label);
                    labels.len() - 1
                }
            })
            .collect()
    })
}

fn folded_inverse(layouts: &[FoldLayout; 3], count: usize) -> Vec<[Option<usize>; 3]> {
    let mut inverse = vec![[None; 3]; count];
    for operand in 0..3 {
        for (axis, &label) in layouts[operand].folded_indices.iter().enumerate() {
            inverse[label][operand] = Some(axis);
        }
    }
    inverse
}

fn labels_for(inverse: &[[Option<usize>; 3]], operands: &[usize]) -> Vec<usize> {
    inverse
        .iter()
        .enumerate()
        .filter_map(|(label, entry)| {
            entry.iter().enumerate()
                .all(|(operand, axis)| axis.is_some() == operands.contains(&operand))
                .then_some(label)
        })
        .collect()
}

fn axis_order(layout: &FoldLayout, labels: &[usize]) -> Vec<usize> {
    labels
        .iter()
        .map(|label| {
            layout
                .folded_indices
                .iter()
                .position(|candidate| candidate == label)
                .unwrap()
        })
        .collect()
}

fn permutation_orders(
    layouts: &[FoldLayout; 3],
    inverse: &[[Option<usize>; 3]],
    permutation: usize,
) -> [Vec<usize>; 3] {
    let roles = PERMUTATIONS[permutation];
    let k = labels_for(inverse, &[roles[0], roles[1]]);
    let m = labels_for(inverse, &[roles[0], roles[2]]);
    let n = labels_for(inverse, &[roles[1], roles[2]]);
    let l = labels_for(inverse, &[0, 1, 2]);
    let mut first = Vec::new();
    first.extend_from_slice(&k);
    first.extend_from_slice(&m);
    first.extend_from_slice(&l);
    let mut second = Vec::new();
    second.extend_from_slice(&k);
    second.extend_from_slice(&n);
    second.extend_from_slice(&l);
    let mut third = Vec::new();
    third.extend_from_slice(&m);
    third.extend_from_slice(&n);
    third.extend_from_slice(&l);
    let mut orders: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::new());
    orders[roles[0]] = axis_order(&layouts[roles[0]], &first);
    orders[roles[1]] = axis_order(&layouts[roles[1]], &second);
    orders[roles[2]] = axis_order(&layouts[roles[2]], &third);
    orders
}

fn product_for(
    inverse: &[[Option<usize>; 3]],
    dimensions: &[Option<usize>],
    operands: &[usize],
) -> Result<usize, Error> {
    inverse
        .iter()
        .zip(dimensions)
        .filter(|(entry, _)| {
            entry.iter().enumerate()
                .all(|(operand, axis)| axis.is_some() == operands.contains(&operand))
        })
        .try_fold(1usize, |value, (_, &length)| {
            value.checked_mul(length.unwrap()).ok_or(Error::SizeOverflow)
        })
}

/// Compose the source fold-index, tensor-fold, and dense permutation metadata.
/// Transpose models see every packed symmetry group and the complete selected-
/// prefix/residual ordering, exactly as map_fold does before local execution.
pub fn select(
    local_shapes: [&[usize]; 3],
    links: [&[Symmetry]; 3],
    indices: [&str; 3],
    models: &Models,
    virtual_copies: [usize; 3],
) -> Result<Outcome, Error> {
    select_permutations(
        local_shapes,
        links,
        indices,
        models,
        virtual_copies,
        [1.; 3],
        &[0, 1, 2, 3, 4, 5],
        false,
    )
}

/// Sparse `select_ctr_perm` considers only source permutation 1 (A,C,B).
/// Sparse transpose estimates are scaled by the stored fractions before the
/// selected third role receives the source's second output transpose charge.
pub fn select_sparse(
    local_shapes: [&[usize]; 3],
    links: [&[Symmetry]; 3],
    indices: [&str; 3],
    models: &Models,
    virtual_copies: [usize; 3],
    fractions: [f64; 3],
) -> Result<Outcome, Error> {
    select_permutations(
        local_shapes,
        links,
        indices,
        models,
        virtual_copies,
        fractions,
        &[1],
        true,
    )
}

fn select_permutations(
    local_shapes: [&[usize]; 3],
    links: [&[Symmetry]; 3],
    indices: [&str; 3],
    models: &Models,
    virtual_copies: [usize; 3],
    transpose_factors: [f64; 3],
    permutations: &[usize],
    sparse: bool,
) -> Result<Outcome, Error> {
    let operands = [Operand::A, Operand::B, Operand::C];
    for operand in 0..3 {
        if local_shapes[operand].len() != indices[operand].len() {
            return Err(Error::ShapeRankMismatch {
                operand: operands[operand],
                shape: local_shapes[operand].len(),
                indices: indices[operand].len(),
            });
        }
    }
    let fold = fold_indices::select(indices, links, sparse, false)?;
    if fold.eligibility != fold_indices::Eligibility::Eligible {
        return Ok(Outcome::Ineligible(fold.eligibility));
    }
    let normalized = normalized(indices);
    let base_layouts: [FoldLayout; 3] = std::array::from_fn(|operand| {
        FoldLayout::new(
            local_shapes[operand],
            links[operand],
            &normalized[operand],
            &fold.labels,
        )
    });
    let inverse = folded_inverse(&base_layouts, fold.labels.len());
    let mut dimensions = vec![None; fold.labels.len()];
    for operand in 0..3 {
        for (axis, &label) in base_layouts[operand].folded_indices.iter().enumerate() {
            let length = base_layouts[operand].folded_shape[axis];
            if dimensions[label].is_some_and(|old| old != length) {
                return Err(Error::FoldedLengthMismatch {
                    label: fold.labels[label],
                });
            }
            dimensions[label] = Some(length);
        }
    }
    // fold_idx retains every selected original label, while tensor::fold emits
    // one rec_tsr dimension per symmetry group (its terminal NS label). Thus
    // nonterminal SY/AS/SH labels intentionally remain holes here.
    let k = product_for(&inverse, &dimensions, &[0, 1])?;
    let m = product_for(&inverse, &dimensions, &[0, 2])?;
    let n = product_for(&inverse, &dimensions, &[1, 2])?;
    let batches = product_for(&inverse, &dimensions, &[0, 1, 2])?;

    let mut selected = None;
    let mut selected_time = f64::MAX;
    for &permutation in permutations {
        let orders = permutation_orders(&base_layouts, &inverse, permutation);
        let mut layouts = base_layouts.clone();
        for operand in 0..3 {
            layouts[operand].permute_folded(&orders[operand]);
        }
        let mut transpose_seconds = std::array::from_fn(|operand| {
            transpose_factors[operand]
                * virtual_copies[operand] as f64
                * models.transpose(
                    &layouts[operand].group_lengths,
                    &layouts[operand].inner_ordering,
                )
        });
        transpose_seconds[PERMUTATIONS[permutation][2]] *= 2.;
        let roles=PERMUTATIONS[permutation];
        let total = transpose_seconds[roles[0]]+transpose_seconds[roles[1]]+transpose_seconds[roles[2]];
        if total <= selected_time {
            selected_time = total;
            let (trans_a, trans_b, transposed_output) = TRANSPOSES[permutation];
            selected = Some(Descriptor {
                fold_labels: fold.labels.clone(),
                layouts,
                permutation,
                transpose_seconds,
                m,
                n,
                k,
                batches,
                trans_a,
                trans_b,
                transposed_output,
            });
        }
    }
    Ok(Outcome::Selected(selected.unwrap()))
}
