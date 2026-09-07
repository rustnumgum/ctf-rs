// Adapted from cc4s CTF interface/matrix.cxx::Matrix::svd_rand at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Randomized SVD retaining the sparse input through both sparse matrix products.

use crate::{
    algebra::{Arithmetic, Complex, Group, Monoid, Semiring},
    mapping::{Distribution, Mapping, Topology},
    sparse::SparseTensor,
    tensor::Tensor,
};

fn distribution(shape: &[usize], grid: [usize; 2]) -> Distribution {
    let topology = Topology::new(grid.to_vec());
    let mut row = Mapping::Unmapped;
    row.augment_physical(&topology, 0);
    let mut column = Mapping::Unmapped;
    column.augment_physical(&topology, 1);
    Distribution::new(shape.to_vec(), topology, vec![row, column])
}

macro_rules! sparse_randomized_svd {
($scalar:ty) => {
impl<'c, 'r> SparseTensor<'c, 'r, Arithmetic<$scalar>> {
    /// Source randomized matrix SVD. The sparse matrix and its plain transpose
    /// remain sparse; all subspace, Gram, projected, and factor matrices are
    /// distributed dense tensors.
    pub fn svd_randomized(
        &self,
        grid: [usize; 2],
        rank: usize,
        iterations: usize,
        oversampling: usize,
        seed: u64,
        guess: Option<&mut Tensor<'c, 'r, Arithmetic<$scalar>>>,
    ) -> Result<(Tensor<'c, 'r, Arithmetic<$scalar>>,
        Tensor<'c, 'r, Arithmetic<$scalar>>,
        Tensor<'c, 'r, Arithmetic<$scalar>>), i32> {
        assert_eq!(self.distribution().shape.len(), 2);
        let (m, n) = (self.distribution().shape[0], self.distribution().shape[1]);
        assert!(rank > 0 && rank <= m.min(n));
        let width = (rank + oversampling).min(m.min(n));
        let algebra = Arithmetic::<$scalar>::new();
        let zero = algebra.zero();
        let one = algebra.one();
        let mut guess = guess;
        let mut subspace = if let Some(guess) = guess.as_deref() {
            assert!(std::ptr::eq(self.context(), guess.context()));
            assert!(rank + oversampling <= m.min(n));
            assert_eq!(guess.distribution().shape, vec![m, width]);
            guess.clone()
        } else {
            let mut values = Tensor::<Arithmetic<$scalar>>::new(
                self.context(),
                distribution(&[m, width], grid),
                Arithmetic::new(),
            );
            let mut generator = crate::random::Generator::new(
                seed.wrapping_add(self.context().rank() as u64),
            );
            values.fill_random(algebra.negate(&one), one.clone(), &mut generator);
            values.qr(grid)?.0
        };
        for _ in 0..iterations {
            // Source uses transpose rather than conjugate transpose here,
            // including for complex element types.
            let sparse_transpose = self.permute_axes(&[1, 0]);
            let mut gram = Tensor::<Arithmetic<$scalar>>::new(
                self.context(),
                distribution(&[m, m], grid),
                Arithmetic::new(),
            );
            gram.gemm_sparse(self, &sparse_transpose, grid, one.clone(), zero.clone());
            let mut next = Tensor::<Arithmetic<$scalar>>::new(
                self.context(),
                distribution(&[m, width], grid),
                Arithmetic::new(),
            );
            next.gemm_2d::<crate::linalg::Native>(
                &gram,
                &subspace,
                grid,
                one.clone(),
                zero.clone(),
            );
            subspace = next.qr(grid)?.0;
        }
        if iterations > 0 {
            if let Some(guess) = guess.as_deref_mut() {
                *guess = subspace.clone();
            }
        }
        let u = if width > rank {
            subspace.slice(&[0..m, 0..rank])
        } else {
            subspace
        };
        let transpose = u.permute_axes(&[1, 0]);
        let mut projected = Tensor::<Arithmetic<$scalar>>::new(
            self.context(),
            distribution(&[rank, n], grid),
            Arithmetic::new(),
        );
        projected.gemm_dense_sparse(&transpose, self, grid, one.clone(), zero.clone());
        let (rotation, singular, vt) = projected.svd(grid)?;
        let mut left = Tensor::<Arithmetic<$scalar>>::new(
            self.context(),
            distribution(&[m, rank], grid),
            Arithmetic::new(),
        );
        left.gemm_2d::<crate::linalg::Native>(&u, &rotation, grid, one, zero);
        Ok((left, singular, vt))
    }
}
};
}

sparse_randomized_svd!(f32);
sparse_randomized_svd!(f64);
sparse_randomized_svd!(Complex<f32>);
sparse_randomized_svd!(Complex<f64>);
