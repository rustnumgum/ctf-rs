// Adapted from cc4s CTF sparse_formats/coo.cxx::coomm,
// interface/semiring.h::{default_coomm,Semiring::coomm}, and
// interface/kernel.h::Kernel::coomm at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Native COO/dense-to-dense CPU kernels. COO coordinates are one-based and
//! dense matrices are contiguous column-major buffers.

use std::ops::{AddAssign, Mul};

use crate::{algebra::Semiring, sparse_formats::Coo};

impl<T: Clone> Coo<T> {
    /// Source `default_coomm`: `C = C*beta`, followed in stored COO order by
    /// `C += alpha*(A*B)`. The right-side beta multiplication and old-first
    /// `AddAssign` are intentionally distinct from the generic semiring path.
    pub fn default_coomm(
        &self,
        columns: usize,
        b: &[T],
        alpha: &T,
        beta: &T,
        c: &mut [T],
    ) where
        T: Mul<Output = T> + AddAssign,
    {
        let (rows, inner) = self.shape();
        assert_eq!(b.len(), inner * columns);
        assert_eq!(c.len(), rows * columns);
        for output in c.iter_mut() {
            *output = output.clone() * beta.clone();
        }
        for (row, column, value) in self.entries() {
            for output_column in 0..columns {
                let contribution = alpha.clone()
                    * (value.clone() * b[output_column * inner + column - 1].clone());
                c[output_column * rows + row - 1] += contribution;
            }
        }
    }

    /// Source generic `Semiring::coomm` fallback. That branch is defined only
    /// for multiplicative-identity beta and accumulates contribution-first:
    /// `C = add(alpha*(A*B), C)`. Duplicate coordinates and stored zeros are
    /// visited in their existing COO order.
    pub fn coomm<A: Semiring<Element = T>>(
        &self,
        algebra: &A,
        columns: usize,
        b: &[T],
        alpha: &T,
        beta: &T,
        c: &mut [T],
    ) where T: PartialEq {
        assert!(beta == &algebra.one());
        let (rows, inner) = self.shape();
        assert_eq!(b.len(), inner * columns);
        assert_eq!(c.len(), rows * columns);
        for (row, column, value) in self.entries() {
            for output_column in 0..columns {
                let product = algebra.multiply(value, &b[output_column * inner + column - 1]);
                let contribution = algebra.multiply(alpha, &product);
                let at = output_column * rows + row - 1;
                c[at] = algebra.add(&contribution, &c[at]);
            }
        }
    }

    /// Source `Kernel<f,g>::coomm`, including mixed A/B/C element types.
    /// Custom COO execution requires identity beta and absent-or-identity alpha;
    /// neither coefficient is otherwise applied. `accumulate(value, output)` is
    /// called once for every stored COO entry and dense output column.
    ///
    /// The ordinary `functions.h::Bivar_Function` does not override `ccoomm`
    /// and reaches the source base-class assertion, so this method deliberately
    /// represents only the defined `Kernel<f,g>` route.
    pub fn coomm_kernel<B, C: PartialEq, A: Semiring<Element = C>>(
        &self,
        algebra: &A,
        columns: usize,
        b: &[B],
        alpha: Option<&C>,
        beta: &C,
        c: &mut [C],
        function: impl Fn(&T, &B) -> C,
        accumulate: impl Fn(C, &mut C),
    ) {
        let one = algebra.one();
        assert!(beta == &one);
        assert!(alpha.is_none() || alpha == Some(&one));
        let (rows, inner) = self.shape();
        assert_eq!(b.len(), inner * columns);
        assert_eq!(c.len(), rows * columns);
        for (row, column, value) in self.entries() {
            for output_column in 0..columns {
                let result = function(value, &b[output_column * inner + column - 1]);
                accumulate(result, &mut c[output_column * rows + row - 1]);
            }
        }
    }
}
