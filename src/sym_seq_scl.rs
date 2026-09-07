// Adapted from cc4s CTF scaling/sym_seq_scl.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Sequential indexed scaling over one packed symmetry block.

use crate::{algebra::Semiring, symmetry::Layout};

/// Source `inv_idx`: the last tensor axis carrying a label wins.
pub fn inverse_indices(indices: &[usize]) -> Vec<Option<usize>> {
    let count = indices.iter().copied().max().map_or(0, |index| index + 1);
    let mut inverse = vec![None; count];
    for (axis, &index) in indices.iter().enumerate() {
        inverse[index] = Some(axis);
    }
    inverse
}

pub(crate) fn validate(shape: &[usize], indices: &[usize]) {
    assert_eq!(shape.len(), indices.len());
    for (axis, &index) in indices.iter().enumerate() {
        for previous in 0..axis {
            if indices[previous] == index {
                assert_eq!(shape[previous], shape[axis]);
            }
        }
    }
}

pub(crate) fn selected(indices: &[usize], coordinates: &[usize]) -> bool {
    (0..indices.len()).all(|axis| {
        (0..axis).all(|previous| {
            indices[axis] != indices[previous]
                || coordinates[axis] == coordinates[previous]
        })
    })
}

/// Apply source `sym_seq_scl_cust` to represented packed coordinates.
/// If present, `alpha` is multiplied on the right before the endomorphism.
pub fn transform<A: Semiring>(
    algebra: &A,
    layout: &Layout,
    indices: &[usize],
    values: &mut [A::Element],
    alpha: Option<&A::Element>,
    mut function: impl FnMut(&mut A::Element),
) {
    validate(layout.shape(), indices);
    assert_eq!(values.len(), layout.len());
    for (coordinates, value) in layout.coordinates().zip(values) {
        if selected(indices, &coordinates) {
            if let Some(alpha) = alpha {
                *value = algebra.multiply(value, alpha);
            }
            function(value);
        }
    }
}

/// Apply source `sym_seq_scl_ref`: `value = value * alpha`.
pub fn scale<A: Semiring>(
    algebra: &A,
    layout: &Layout,
    indices: &[usize],
    values: &mut [A::Element],
    alpha: &A::Element,
) {
    transform(algebra, layout, indices, values, Some(alpha), |_| {});
}
