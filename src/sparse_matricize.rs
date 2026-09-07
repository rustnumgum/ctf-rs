// Adapted from cc4s CTF sparse_formats/coo.cxx::{set_data,get_data} at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Raw tensor-pair/COO matricization. Forward conversion includes the source's
//! folded-symmetry rank. Reverse conversion is intentionally only the unfolded
//! algorithm implemented by the source.

use crate::{sparse_formats::Coo, symmetry::Symmetry};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matricization {
    pub shape: Vec<usize>,
    pub padded_shape: Vec<usize>,
    /// Adjacent symmetry links; every group ends in `NS`.
    pub links: Vec<Symmetry>,
    /// `all_flen` in original folded-group order.
    pub folded_shape: Vec<usize>,
    /// Source `rev_ordering`, a permutation of folded dimensions.
    pub reverse_ordering: Vec<usize>,
    /// Leading dimensions in reordered space assigned to the COO row.
    pub row_dimensions: usize,
    pub phases: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dematricization {
    pub shape: Vec<usize>,
    /// Source `rev_ordering`, a permutation of original dimensions.
    pub reverse_ordering: Vec<usize>,
    pub row_dimensions: usize,
    pub phases: Vec<usize>,
    pub phase_ranks: Vec<usize>,
}

fn inverse_ordering(reverse: &[usize]) -> Vec<usize> {
    let mut ordering = vec![0; reverse.len()];
    for (position, &dimension) in reverse.iter().enumerate() {
        ordering[dimension] = position;
    }
    ordering
}

fn reordered_lengths(lengths: &[usize], ordering: &[usize]) -> Vec<usize> {
    let mut reordered = vec![0; lengths.len()];
    for (dimension, &length) in lengths.iter().enumerate() {
        reordered[ordering[dimension]] = length;
    }
    reordered
}

fn matrix_strides(lengths: &[usize], row_dimensions: usize) -> (Vec<usize>, Vec<usize>) {
    let mut row = vec![0; row_dimensions];
    let mut column = vec![0; lengths.len() - row_dimensions];
    let mut stride = 1;
    for position in 0..row_dimensions {
        row[position] = stride;
        stride *= lengths[position];
    }
    stride = 1;
    for position in row_dimensions..lengths.len() {
        column[position - row_dimensions] = stride;
        stride *= lengths[position];
    }
    (row, column)
}

fn validate_permutation(permutation: &[usize]) {
    let mut sorted = permutation.to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..permutation.len()).collect::<Vec<_>>());
}

impl Matricization {
    fn validate(&self) {
        let order = self.shape.len();
        assert_eq!(self.padded_shape.len(), order);
        assert_eq!(self.links.len(), order);
        assert_eq!(self.phases.len(), order);
        assert_eq!(self.folded_shape.len(), self.reverse_ordering.len());
        assert!(self.row_dimensions <= self.folded_shape.len());
        validate_permutation(&self.reverse_ordering);
    }

    fn folded_key(&self, mut key: usize) -> usize {
        let mut folded = 0usize;
        let mut super_stride = 1usize;
        let mut last_index = -1isize;
        let mut folded_dimension = 0usize;
        for dimension in 0..self.shape.len() {
            let coordinate = (key % self.shape[dimension]) / self.phases[dimension];
            key /= self.shape[dimension];
            if dimension == 0 {
                folded += coordinate;
                if self.links[0] == Symmetry::NS {
                    last_index = 0;
                    super_stride = self.padded_shape[0] / self.phases[0];
                    folded_dimension = 1;
                }
            } else {
                let degree = (dimension as isize - last_index) as usize;
                let mut rank = 1usize;
                for offset in 0..degree {
                    rank *= coordinate + offset;
                    rank /= offset + 1;
                }
                folded += rank * super_stride;
                if self.links[dimension] == Symmetry::NS {
                    super_stride *= self.folded_shape[folded_dimension];
                    folded_dimension += 1;
                    last_index = dimension as isize;
                }
            }
        }
        folded
    }
}

/// Convert global column-major tensor keys to one-based COO coordinates.
/// Non-NS links all use the same source rising-binomial rank, including AS and
/// SH; this does not substitute strict antisymmetric combinadics.
pub fn matricize_pairs<E: Clone>(
    metadata: &Matricization,
    pairs: &[(usize, E)],
) -> Coo<E> {
    metadata.validate();
    let ordering = inverse_ordering(&metadata.reverse_ordering);
    let matrix_lengths = reordered_lengths(&metadata.folded_shape, &ordering);
    let (row_strides, column_strides) =
        matrix_strides(&matrix_lengths, metadata.row_dimensions);
    let rows = matrix_lengths[..metadata.row_dimensions].iter().product();
    let columns = matrix_lengths[metadata.row_dimensions..].iter().product();
    let folded = metadata.folded_shape.len() != metadata.shape.len();

    let entries = pairs
        .iter()
        .map(|(key, value)| {
            let mut local_key = if folded {
                metadata.folded_key(*key)
            } else {
                *key
            };
            let mut row = 1usize;
            let mut column = 1usize;
            let dimensions = if folded {
                metadata.folded_shape.len()
            } else {
                metadata.shape.len()
            };
            for dimension in 0..dimensions {
                let coordinate = if folded {
                    let coordinate = local_key % metadata.folded_shape[dimension];
                    local_key /= metadata.folded_shape[dimension];
                    coordinate
                } else {
                    let coordinate = (local_key % metadata.shape[dimension])
                        / metadata.phases[dimension];
                    local_key /= metadata.shape[dimension];
                    coordinate
                };
                let position = ordering[dimension];
                if position < metadata.row_dimensions {
                    row += coordinate * row_strides[position];
                } else {
                    column += coordinate * column_strides[position - metadata.row_dimensions];
                }
            }
            (row, column, value.clone())
        })
        .collect();
    Coo::new(rows, columns, entries)
}

impl Dematricization {
    fn validate(&self) {
        let order = self.shape.len();
        assert_eq!(self.reverse_ordering.len(), order);
        assert_eq!(self.phases.len(), order);
        assert_eq!(self.phase_ranks.len(), order);
        assert!(self.row_dimensions <= order);
        validate_permutation(&self.reverse_ordering);
    }
}

/// Convert unfolded COO coordinates back to global tensor keys and sort by key.
/// Like source `get_data`, this does not filter phase padding. Folded symmetry
/// reverse conversion is not exposed because the source marks it unimplemented.
pub fn dematricize_pairs<E: Clone>(
    metadata: &Dematricization,
    matrix: &Coo<E>,
) -> Vec<(usize, E)> {
    metadata.validate();
    let ordering = inverse_ordering(&metadata.reverse_ordering);
    let divided_lengths: Vec<_> = metadata
        .shape
        .iter()
        .zip(&metadata.phases)
        .map(|(&length, &phase)| length / phase + usize::from(length % phase != 0))
        .collect();
    let matrix_lengths = reordered_lengths(&divided_lengths, &ordering);
    let (row_strides, column_strides) =
        matrix_strides(&matrix_lengths, metadata.row_dimensions);
    assert_eq!(
        matrix.shape(),
        (
            matrix_lengths[..metadata.row_dimensions].iter().product(),
            matrix_lengths[metadata.row_dimensions..].iter().product(),
        )
    );

    let mut pairs: Vec<_> = matrix
        .entries()
        .iter()
        .map(|(row, column, value)| {
            let mut key = 0usize;
            let mut stride = 1usize;
            for dimension in 0..metadata.shape.len() {
                let position = ordering[dimension];
                let coordinate = if position < metadata.row_dimensions {
                    ((row - 1) / row_strides[position]) % matrix_lengths[position]
                } else {
                    ((column - 1) / column_strides[position - metadata.row_dimensions])
                        % matrix_lengths[position]
                };
                key += (coordinate * metadata.phases[dimension]
                    + metadata.phase_ranks[dimension])
                    * stride;
                stride *= metadata.shape[dimension];
            }
            (key, value.clone())
        })
        .collect();
    pairs.sort_unstable_by_key(|pair| pair.0);
    pairs
}
