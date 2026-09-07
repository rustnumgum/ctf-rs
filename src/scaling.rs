// Adapted from cc4s CTF scaling/scaling.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Dense local scaling selection. Distributed tensors stay in their current
//! mapping; each rank visits only its owned storage, with no gathered tensor.

use crate::{
    algebra::Semiring,
    mapping::Distribution,
    sym_seq_scl::{selected, validate},
    symmetric_distribution::SymmetricDistribution,
};

/// Normalize byte labels in first-appearance order, as source `conv_idx` does.
pub fn normalize_indices(indices: &str) -> Vec<usize> {
    assert!(indices.is_ascii());
    let mut labels = Vec::new();
    indices
        .bytes()
        .map(|label| {
            if let Some(index) = labels.iter().position(|&old| old == label) {
                index
            } else {
                labels.push(label);
                labels.len() - 1
            }
        })
        .collect()
}

pub(crate) fn transform_dense<T>(
    distribution: &Distribution,
    rank: usize,
    indices: &str,
    data: &mut [T],
    mut function: impl FnMut(&mut T),
) {
    let indices = normalize_indices(indices);
    validate(&distribution.shape, &indices);
    for (offset, value) in data.iter_mut().enumerate() {
        if let Some(key) = distribution.global_key(rank, offset) {
            let coordinates = distribution.decode_key(key);
            if selected(&indices, &coordinates) {
                function(value);
            }
        }
    }
}

pub(crate) fn scale_dense<A: Semiring>(
    algebra: &A,
    distribution: &Distribution,
    rank: usize,
    indices: &str,
    data: &mut [A::Element],
    alpha: &A::Element,
) {
    transform_dense(distribution, rank, indices, data, |value| {
        *value = algebra.multiply(value, alpha);
    });
}

pub(crate) fn scale_all_dense<A: Semiring>(
    algebra: &A,
    distribution: &Distribution,
    rank: usize,
    data: &mut [A::Element],
    alpha: &A::Element,
) {
    for (offset, value) in data.iter_mut().enumerate() {
        if distribution.global_key(rank, offset).is_some() {
            *value = algebra.multiply(value, alpha);
        }
    }
}

pub(crate) fn transform_packed<T>(
    distribution: &SymmetricDistribution,
    rank: usize,
    indices: &str,
    data: &mut [T],
    mut function: impl FnMut(&mut T),
) {
    let indices = normalize_indices(indices);
    validate(&distribution.distribution().shape, &indices);
    for (offset, key) in distribution.local_pairs(rank) {
        let coordinates = distribution.distribution().decode_key(key);
        if selected(&indices, &coordinates) {
            function(&mut data[offset]);
        }
    }
}

pub(crate) fn scale_packed<A: Semiring>(
    algebra: &A,
    distribution: &SymmetricDistribution,
    rank: usize,
    indices: &str,
    data: &mut [A::Element],
    alpha: &A::Element,
) {
    transform_packed(distribution, rank, indices, data, |value| {
        *value = algebra.multiply(value, alpha);
    });
}

pub(crate) fn scale_all_packed<A: Semiring>(
    algebra: &A,
    distribution: &SymmetricDistribution,
    rank: usize,
    data: &mut [A::Element],
    alpha: &A::Element,
) {
    for (offset, _) in distribution.local_pairs(rank) {
        data[offset] = algebra.multiply(&data[offset], alpha);
    }
}
