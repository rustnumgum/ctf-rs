// Adapted from cc4s CTF interface/multilinear.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Tensor SVD via distributed matrix SVD, with no global tensor gather.

use crate::{
    algebra::Arithmetic,
    mapping::Distribution,
    tensor::Tensor,
};

/// Distributed matrix SVD strategy used by [`Tensor::tensor_svd`].
#[derive(Clone, Copy, Debug)]
pub enum TensorSvd {
    /// Compute a native SVD, then retain the requested rank and/or threshold.
    Truncated {
        rank: Option<usize>,
        threshold: f64,
    },
    /// Compute the source randomized SVD path with an explicit random seed.
    Randomized {
        rank: usize,
        iterations: usize,
        oversampling: usize,
        seed: u64,
    },
}

impl<'c, 'r> Tensor<'c, 'r, Arithmetic<f64>> {
    /// Factor a tensor as `U * S * VT` over the supplied index partition.
    ///
    /// `left` and `right` each contain `auxiliary` exactly once; all other
    /// input labels occur exactly once across the two output index strings.
    /// The input is matricized in the order of the non-auxiliary labels in
    /// `left` followed by those in `right`. The native distributed matrix SVD
    /// then operates on that cyclic matrix, and the factors are reshaped and
    /// permuted into the requested output orders.
    pub fn tensor_svd(
        &self,
        indices: &str,
        left: &str,
        auxiliary: char,
        right: &str,
        grid: [usize; 2],
        method: TensorSvd,
    ) -> Result<(Self, Self, Self), i32> {
        assert!(indices.is_ascii() && left.is_ascii() && right.is_ascii() && auxiliary.is_ascii());
        let input = indices.as_bytes();
        let left = left.as_bytes();
        let right = right.as_bytes();
        let auxiliary = auxiliary as u8;

        assert!(input.iter().enumerate().all(|(i, &label)| {
            !input[..i].contains(&label)
        }));
        assert!(left.iter().enumerate().all(|(i, &label)| {
            !left[..i].contains(&label)
        }));
        assert!(right.iter().enumerate().all(|(i, &label)| {
            !right[..i].contains(&label)
        }));
        assert!(!input.contains(&auxiliary));
        assert_eq!(left.iter().filter(|&&label| label == auxiliary).count(), 1);
        assert_eq!(right.iter().filter(|&&label| label == auxiliary).count(), 1);
        assert_eq!(left.len() + right.len() - 2, input.len());

        let left_nonaux: Vec<_> = left
            .iter()
            .copied()
            .filter(|&label| label != auxiliary)
            .collect();
        let right_nonaux: Vec<_> = right
            .iter()
            .copied()
            .filter(|&label| label != auxiliary)
            .collect();
        let output_nonaux = left_nonaux
            .iter()
            .chain(right_nonaux.iter())
            .copied()
            .collect::<Vec<_>>();
        assert!(input.iter().all(|label| {
            output_nonaux.iter().filter(|candidate| *candidate == label).count() == 1
        }));
        assert!(output_nonaux.iter().all(|label| input.contains(label)));

        let input_axes = |label: &u8| {
            input
                .iter()
                .position(|candidate| candidate == label)
                .unwrap()
        };
        let axes: Vec<_> = left_nonaux
            .iter()
            .chain(right_nonaux.iter())
            .map(input_axes)
            .collect();
        let reordered = self.permute_axes(&axes);

        let dims_l: Vec<_> = left_nonaux
            .iter()
            .map(|label| self.distribution().shape[input_axes(label)])
            .collect();
        let dims_r: Vec<_> = right_nonaux
            .iter()
            .map(|label| self.distribution().shape[input_axes(label)])
            .collect();
        let rows = dims_l.iter().product();
        let columns = dims_r.iter().product();
        let matrix = reordered.reshape(Distribution::cyclic(
            vec![rows, columns],
            self.context().size(),
        ));

        let (u_matrix, singular, vt_matrix) = match method {
            TensorSvd::Truncated { rank, threshold } => {
                matrix.svd_truncated(grid, rank, threshold)?
            }
            TensorSvd::Randomized {
                rank,
                iterations,
                oversampling,
                seed,
            } => matrix.svd_randomized(grid, rank, iterations, oversampling, seed, None)?,
        };
        let rank = singular.distribution().shape[0];

        let mut u_shape = dims_l;
        u_shape.push(rank);
        // The source's `U += U` after zero construction is a no-op artifact;
        // this direct reshape stores each factor entry exactly once.
        let u_canonical = u_matrix.reshape(Distribution::cyclic(
            u_shape,
            self.context().size(),
        ));
        let mut vt_shape = vec![rank];
        vt_shape.extend(dims_r);
        let vt_canonical = vt_matrix.reshape(Distribution::cyclic(
            vt_shape,
            self.context().size(),
        ));

        let u_axes: Vec<_> = left
            .iter()
            .map(|&label| {
                if label == auxiliary {
                    left_nonaux.len()
                } else {
                    left_nonaux.iter().position(|&candidate| candidate == label).unwrap()
                }
            })
            .collect();
        let vt_axes: Vec<_> = right
            .iter()
            .map(|&label| {
                if label == auxiliary {
                    0
                } else {
                    1 + right_nonaux
                        .iter()
                        .position(|&candidate| candidate == label)
                        .unwrap()
                }
            })
            .collect();
        Ok((
            u_canonical.permute_axes(&u_axes),
            singular,
            vt_canonical.permute_axes(&vt_axes),
        ))
    }
}
