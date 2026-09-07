// Adapted from cc4s CTF interface/multilinear.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Tensor SVD via distributed matrix SVD, with no global tensor gather.

use crate::{
    algebra::{Arithmetic, Complex, Monoid, Wire},
    mapping::Distribution,
    sparse::SparseTensor,
    tensor::Tensor,
};

struct TensorSvdLayout {
    input_axes: Vec<usize>,
    left_dimensions: Vec<usize>,
    right_dimensions: Vec<usize>,
    left_axes: Vec<usize>,
    right_axes: Vec<usize>,
}

impl TensorSvdLayout {
    fn new(distribution: &Distribution, indices: &str, left: &str,
        auxiliary: char, right: &str) -> Self {
        assert!(indices.is_ascii() && left.is_ascii() && right.is_ascii() && auxiliary.is_ascii());
        let input = indices.as_bytes();
        let left = left.as_bytes();
        let right = right.as_bytes();
        let auxiliary = auxiliary as u8;

        assert!(input.iter().enumerate().all(|(i, &label)| !input[..i].contains(&label)));
        assert!(left.iter().enumerate().all(|(i, &label)| !left[..i].contains(&label)));
        assert!(right.iter().enumerate().all(|(i, &label)| !right[..i].contains(&label)));
        assert_eq!(input.len(), distribution.shape.len());
        assert!(!input.contains(&auxiliary));
        assert_eq!(left.iter().filter(|&&label| label == auxiliary).count(), 1);
        assert_eq!(right.iter().filter(|&&label| label == auxiliary).count(), 1);
        assert_eq!(left.len() + right.len() - 2, input.len());

        let left_nonaux: Vec<_> = left.iter().copied()
            .filter(|&label| label != auxiliary).collect();
        let right_nonaux: Vec<_> = right.iter().copied()
            .filter(|&label| label != auxiliary).collect();
        let output_nonaux: Vec<_> = left_nonaux.iter().chain(&right_nonaux).copied().collect();
        assert!(input.iter().all(|label| {
            output_nonaux.iter().filter(|candidate| *candidate == label).count() == 1
        }));
        assert!(output_nonaux.iter().all(|label| input.contains(label)));
        let input_axis = |label: &u8| input.iter()
            .position(|candidate| candidate == label).unwrap();
        let input_axes = left_nonaux.iter().chain(&right_nonaux).map(input_axis).collect();
        let left_dimensions = left_nonaux.iter()
            .map(|label| distribution.shape[input_axis(label)]).collect();
        let right_dimensions = right_nonaux.iter()
            .map(|label| distribution.shape[input_axis(label)]).collect();
        let left_axes = left.iter().map(|&label| {
            if label == auxiliary { left_nonaux.len() }
            else { left_nonaux.iter().position(|&candidate| candidate == label).unwrap() }
        }).collect();
        let right_axes = right.iter().map(|&label| {
            if label == auxiliary { 0 }
            else { 1 + right_nonaux.iter().position(|&candidate| candidate == label).unwrap() }
        }).collect();
        Self { input_axes, left_dimensions, right_dimensions, left_axes, right_axes }
    }

    fn matrix_distribution(&self, processes: usize) -> Distribution {
        Distribution::cyclic(vec![self.left_dimensions.iter().product(),
            self.right_dimensions.iter().product()], processes)
    }
}

fn finish_svd<'c, 'r, A: Monoid + Clone>(layout: TensorSvdLayout,
    u_matrix: Tensor<'c, 'r, A>, singular: Tensor<'c, 'r, A>,
    vt_matrix: Tensor<'c, 'r, A>) -> (Tensor<'c, 'r, A>, Tensor<'c, 'r, A>, Tensor<'c, 'r, A>)
where A::Element: Wire {
    let rank = singular.distribution().shape[0];
    let processes = singular.context().size();
    let mut u_shape = layout.left_dimensions;
    u_shape.push(rank);
    // The source's `U += U` after zero construction is a no-op artifact.
    let u_canonical = u_matrix.reshape(Distribution::cyclic(u_shape, processes));
    let mut vt_shape = vec![rank];
    vt_shape.extend(layout.right_dimensions);
    let vt_canonical = vt_matrix.reshape(Distribution::cyclic(vt_shape, processes));
    (u_canonical.permute_axes(&layout.left_axes), singular,
        vt_canonical.permute_axes(&layout.right_axes))
}

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

macro_rules! tensor_svd_methods {
($scalar:ty) => {
impl<'c, 'r> Tensor<'c, 'r, Arithmetic<$scalar>> {
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
        let layout = TensorSvdLayout::new(self.distribution(), indices, left, auxiliary, right);
        let reordered = self.permute_axes(&layout.input_axes);
        let matrix = reordered.reshape(layout.matrix_distribution(self.context().size()));

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
        Ok(finish_svd(layout, u_matrix, singular, vt_matrix))
    }
}

impl<'c, 'r> SparseTensor<'c, 'r, Arithmetic<$scalar>> {
    /// Truncated tensor SVD with sparse permutation and reshape up to the native
    /// distributed matrix boundary. Densification is local to that distributed
    /// matrix layout; no global tensor is gathered.
    pub fn tensor_svd_truncated(
        &self,
        indices: &str,
        left: &str,
        auxiliary: char,
        right: &str,
        grid: [usize; 2],
        rank: Option<usize>,
        threshold: f64,
    ) -> Result<(Tensor<'c, 'r, Arithmetic<$scalar>>,
        Tensor<'c, 'r, Arithmetic<$scalar>>,
        Tensor<'c, 'r, Arithmetic<$scalar>>), i32> {
        let layout = TensorSvdLayout::new(self.distribution(), indices, left, auxiliary, right);
        let reordered = self.permute_axes(&layout.input_axes);
        let sparse_matrix = reordered.reshape(layout.matrix_distribution(self.context().size()));
        let matrix = sparse_matrix.into_dense();
        let (u_matrix, singular, vt_matrix) = matrix.svd_truncated(grid, rank, threshold)?;
        Ok(finish_svd(layout, u_matrix, singular, vt_matrix))
    }

    /// Randomized tensor SVD using sparse matrix products throughout the source
    /// subspace algorithm. The tensor remains sparse through matricization.
    pub fn tensor_svd_randomized(
        &self,
        indices: &str,
        left: &str,
        auxiliary: char,
        right: &str,
        grid: [usize; 2],
        rank: usize,
        iterations: usize,
        oversampling: usize,
        seed: u64,
    ) -> Result<(Tensor<'c, 'r, Arithmetic<$scalar>>,
        Tensor<'c, 'r, Arithmetic<$scalar>>,
        Tensor<'c, 'r, Arithmetic<$scalar>>), i32> {
        let layout = TensorSvdLayout::new(self.distribution(), indices, left, auxiliary, right);
        let reordered = self.permute_axes(&layout.input_axes);
        let sparse_matrix = reordered.reshape(layout.matrix_distribution(self.context().size()));
        let (u_matrix, singular, vt_matrix) = sparse_matrix.svd_randomized(
            grid, rank, iterations, oversampling, seed, None,
        )?;
        Ok(finish_svd(layout, u_matrix, singular, vt_matrix))
    }
}
};
}
tensor_svd_methods!(f32);
tensor_svd_methods!(f64);
tensor_svd_methods!(Complex<f32>);
tensor_svd_methods!(Complex<f64>);
