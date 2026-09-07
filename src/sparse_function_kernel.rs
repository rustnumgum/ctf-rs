// Adapted from cc4s CTF interface/functions.h Bivar_Function::{csrmm,
// csrmultd} and contraction/spctr_tsr.cxx at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Local folded sparse kernels for an arbitrary bivariate value function.
//!
//! These kernels deliberately do not model the function as semiring
//! multiplication: absent sparse structure is never evaluated, while explicit
//! stored zeros and dense zeros are passed to the function.

use crate::{algebra::Semiring, sparse_formats::{Coo, Csr}};

fn prescale<A: Semiring>(algebra: &A, c: &mut [A::Element], beta: &A::Element) {
    let one = algebra.one();
    if beta != &one {
        let zero = algebra.zero();
        if beta == &zero {
            c.fill(zero);
        } else {
            for value in c {
                *value = algebra.multiply(beta, value);
            }
        }
    }
}

/// Source `Bivar_Function::csrmm`: sparse A times column-major dense B into
/// column-major dense C. The surrounding source layer applies beta once before
/// entering the custom kernel; custom CSR dispatch requires identity alpha.
/// Each stored A entry is evaluated against every corresponding dense B value,
/// and accumulation is `old_C + function(A, B)`.
pub fn csr_dense<
    A: Semiring,
    F: Fn(&A::Element, &A::Element) -> A::Element,
>(
    algebra: &A,
    a: &Csr<A::Element>,
    n: usize,
    b: &[A::Element],
    c: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
    function: F,
) {
    let (m, k) = a.shape();
    assert_eq!(b.len(), k.checked_mul(n).expect("matrix size overflow"));
    assert_eq!(c.len(), m.checked_mul(n).expect("matrix size overflow"));
    prescale(algebra, c, beta);
    assert!(alpha == &algebra.one(),
        "source custom CSR kernel requires identity alpha");

    for row_a in 0..m {
        for col_b in 0..n {
            let output = col_b * m + row_a;
            let start = a.row_offsets()[row_a] - 1;
            let end = a.row_offsets()[row_a + 1] - 1;
            for entry_a in start..end {
                let col_a = a.columns()[entry_a] - 1;
                let value = function(&a.values()[entry_a], &b[col_b * k + col_a]);
                c[output] = algebra.add(&c[output], &value);
            }
        }
    }
}

/// Source `Bivar_Function::csrmultd`: sparse A and sparse B into column-major
/// dense C. Only stored A entries and stored entries in the matching B row are
/// evaluated; explicit stored zeros remain function arguments. Beta is applied
/// once before traversal, alpha must be the multiplicative identity, and each
/// update is `old_C + function(A, B)`.
pub fn csr_sparse<
    A: Semiring,
    F: Fn(&A::Element, &A::Element) -> A::Element,
>(
    algebra: &A,
    a: &Csr<A::Element>,
    b: &Csr<A::Element>,
    c: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
    function: F,
) {
    let (m, k) = a.shape();
    let (rows_b, n) = b.shape();
    assert_eq!(k, rows_b);
    assert_eq!(c.len(), m.checked_mul(n).expect("matrix size overflow"));
    prescale(algebra, c, beta);
    assert!(alpha == &algebra.one(),
        "source custom CSR kernel requires identity alpha");

    for row_a in 0..m {
        let start_a = a.row_offsets()[row_a] - 1;
        let end_a = a.row_offsets()[row_a + 1] - 1;
        for entry_a in start_a..end_a {
            let row_b = a.columns()[entry_a] - 1;
            let start_b = b.row_offsets()[row_b] - 1;
            let end_b = b.row_offsets()[row_b + 1] - 1;
            for entry_b in start_b..end_b {
                let col_b = b.columns()[entry_b] - 1;
                let output = col_b * m + row_a;
                let value = function(&a.values()[entry_a], &b.values()[entry_b]);
                c[output] = algebra.add(&c[output], &value);
            }
        }
    }
}

/// Source `Bivar_Function::csrmultcsr`: sparse A and sparse B into sparse C.
/// Product structure is the symbolic row-wise union of reachable B columns;
/// the first numeric path writes `function(A, B)` directly and later paths add
/// on the right. Existing C is the left operand of the final source `csr_add`.
/// Alpha and beta are deliberately absent:
/// custom CSR dispatch requires identity alpha and passes identity beta, while
/// the pinned high-level sparse-output path handles beta in a separate sparse
/// summation.
pub fn csr_sparse_output<
    A: Semiring,
    F: Fn(&A::Element, &A::Element) -> A::Element,
>(
    algebra: &A,
    a: &Csr<A::Element>,
    b: &Csr<A::Element>,
    c: &Csr<A::Element>,
    function: F,
) -> Csr<A::Element> {
    let (m, k) = a.shape();
    let (rows_b, n) = b.shape();
    assert_eq!(k, rows_b);
    assert_eq!(c.shape(), (m, n));

    let mut entries = Vec::new();
    let mut present = vec![false; n];
    let mut values: Vec<Option<A::Element>> = vec![None; n];
    for row_a in 0..m {
        present.fill(false);
        let start_a = a.row_offsets()[row_a] - 1;
        let end_a = a.row_offsets()[row_a + 1] - 1;
        for entry_a in start_a..end_a {
            let row_b = a.columns()[entry_a] - 1;
            let start_b = b.row_offsets()[row_b] - 1;
            let end_b = b.row_offsets()[row_b + 1] - 1;
            for entry_b in start_b..end_b {
                present[b.columns()[entry_b] - 1] = true;
            }
        }

        for value in &mut values {
            *value = None;
        }
        for entry_a in start_a..end_a {
            let row_b = a.columns()[entry_a] - 1;
            let start_b = b.row_offsets()[row_b] - 1;
            let end_b = b.row_offsets()[row_b + 1] - 1;
            for entry_b in start_b..end_b {
                let col_b = b.columns()[entry_b] - 1;
                let value = function(&a.values()[entry_a], &b.values()[entry_b]);
                values[col_b] = Some(match values[col_b].take() {
                    Some(previous) => algebra.add(&previous, &value),
                    None => value,
                });
            }
        }
        for col_b in 0..n {
            if present[col_b] {
                entries.push((row_a + 1, col_b + 1, values[col_b].take().unwrap()));
            }
        }
    }

    let product = Coo::new(m, n, entries).to_csr();
    c.add(&product, algebra)
}
