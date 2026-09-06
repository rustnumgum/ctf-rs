// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
// Dense NS reference branch adapted from contraction/sym_seq_ctr.cxx.
use crate::{algebra::Semiring,summation::Indices};

/// Sequential local contraction. This reference kernel, like upstream, scales
/// the whole supplied C block. A Tensor-level diagonal operation must extract
/// its diagonal before invoking this kernel if nonselected entries are to stay.
pub fn sequential<A: Semiring>(algebra: &A,
    shape_a: &[usize], indices_a: &str, a: &[A::Element],
    shape_b: &[usize], indices_b: &str, b: &[A::Element],
    shape_c: &[usize], indices_c: &str, c: &mut [A::Element],
    alpha: &A::Element,beta: &A::Element) {
    sequential_function(algebra,shape_a,indices_a,a,shape_b,indices_b,b,shape_c,indices_c,c,
        alpha,beta,|a,b|algebra.multiply(a,b));
}

/// Custom bivariate contraction: apply f(A,B), right-scale by alpha, accumulate.
pub fn sequential_function<A: Semiring>(algebra: &A,
    shape_a: &[usize], indices_a: &str, a: &[A::Element],
    shape_b: &[usize], indices_b: &str, b: &[A::Element],
    shape_c: &[usize], indices_c: &str, c: &mut [A::Element],
    alpha: &A::Element,beta: &A::Element,
    function: impl Fn(&A::Element,&A::Element)->A::Element) {
    assert_eq!(a.len(),shape_a.iter().product());
    assert_eq!(b.len(),shape_b.iter().product());
    assert_eq!(c.len(),shape_c.iter().product());
    let space = Indices::new(&[(shape_a,indices_a),(shape_b,indices_b),(shape_c,indices_c)]);
    for value in c.iter_mut() {
        *value = if *beta == algebra.zero() {algebra.zero()}
            else if indices_a.is_empty() && indices_b.is_empty() && indices_c.is_empty() {algebra.multiply(value,beta)}
            else {algebra.multiply(beta,value)};
    }
    space.for_each(|offsets| {
        let value = function(&a[offsets[0]],&b[offsets[1]]);
        let scaled = algebra.multiply(&value,alpha);
        c[offsets[2]] = algebra.add(&scaled,&c[offsets[2]]);
    });
}
