// Adapted from cc4s CTF interface/semiring.h::gen_csrmultd.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.

use crate::{
    algebra::Semiring,
    sparse_formats::Csr,
};

pub(crate) fn csr_sparse_dense<A: Semiring>(
    a: &Csr<A::Element>,
    b: &Csr<A::Element>,
    c: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
    algebra: &A,
) {
    let (m, k) = a.shape();
    let (b_rows, n) = b.shape();
    assert_eq!(k, b_rows);
    assert_eq!(c.len(), m * n);

    let one = algebra.one();
    if beta != &one {
        for value in c.iter_mut() {
            *value = algebra.multiply(beta, value);
        }
    }

    for row_a in 0..m {
        let row_start = a.row_offsets()[row_a] - 1;
        let row_end = a.row_offsets()[row_a + 1] - 1;
        for i_a in row_start..row_end {
            let row_b = a.columns()[i_a] - 1;
            let b_row_start = b.row_offsets()[row_b] - 1;
            let b_row_end = b.row_offsets()[row_b + 1] - 1;
            for i_b in b_row_start..b_row_end {
                let col_b = b.columns()[i_b] - 1;
                let product = algebra.multiply(&a.values()[i_a], &b.values()[i_b]);
                let product = if alpha != &one {
                    algebra.multiply(alpha, &product)
                } else {
                    product
                };
                let c_index = col_b * m + row_a;
                // The source fadd call discards its return value; Rust's add returns
                // the updated element, so assign it to preserve the intended update.
                c[c_index] = algebra.add(&c[c_index], &product);
            }
        }
    }
}
