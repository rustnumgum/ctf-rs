// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
// Dense NS reference branch adapted from contraction/sym_seq_ctr.cxx.
use crate::{algebra::Semiring,summation::Indices};

/// ctr_virt traversal, applying beta once to each visited local C virtual block.
pub fn virtualized<A: Semiring>(algebra: &A,
    shape_a: &[usize], virtual_a: &[usize], indices_a: &str, a: &[A::Element],
    shape_b: &[usize], virtual_b: &[usize], indices_b: &str, b: &[A::Element],
    shape_c: &[usize], virtual_c: &[usize], indices_c: &str, c: &mut [A::Element],
    alpha: &A::Element,beta: &A::Element) {
    for (shape,phases) in [(shape_a,virtual_a),(shape_b,virtual_b),(shape_c,virtual_c)] {
        assert_eq!(shape.len(),phases.len()); assert!(phases.iter().all(|&p|p > 0));
    }
    let sa: usize = shape_a.iter().product(); let sb: usize = shape_b.iter().product(); let sc: usize = shape_c.iter().product();
    let nc: usize = virtual_c.iter().product();
    assert_eq!(a.len(),sa*virtual_a.iter().product::<usize>());
    assert_eq!(b.len(),sb*virtual_b.iter().product::<usize>());
    assert_eq!(c.len(),sc*nc);
    let space = Indices::new(&[(virtual_a,indices_a),(virtual_b,indices_b),(virtual_c,indices_c)]);
    let mut visited = vec![false;nc]; let one = algebra.one();
    space.for_each(|offsets| {
        let (ia,ib,ic) = (offsets[0],offsets[1],offsets[2]);
        sequential(algebra,shape_a,indices_a,&a[ia*sa..(ia+1)*sa],shape_b,indices_b,&b[ib*sb..(ib+1)*sb],
            shape_c,indices_c,&mut c[ic*sc..(ic+1)*sc],alpha,if visited[ic] {&one} else {beta});
        visited[ic] = true;
    });
}

/// Native-double ctr_replicate: input broadcast, root-only beta, output Reduce
/// (not Allreduce), and clearing broadcast replicas. Buffers are local blocks.
pub fn replicated_f64(a_comms: &[&crate::context::Context<'_>],b_comms: &[&crate::context::Context<'_>],
    c_comms: &[&crate::context::Context<'_>],
    shape_a: &[usize],virtual_a: &[usize],indices_a: &str,a: &mut [f64],
    shape_b: &[usize],virtual_b: &[usize],indices_b: &str,b: &mut [f64],
    shape_c: &[usize],virtual_c: &[usize],indices_c: &str,c: &mut [f64],alpha: f64,beta: f64) {
    for comm in a_comms {comm.broadcast(0,a);}
    for comm in b_comms {comm.broadcast(0,b);}
    let root = c_comms.iter().all(|comm|comm.rank() == 0);
    if root && beta != 1. {
        if beta == 0. {c.fill(0.);} else {for value in c.iter_mut() {*value *= beta;}}
    }
    virtualized(&crate::algebra::Arithmetic::<f64>::new(),shape_a,virtual_a,indices_a,a,
        shape_b,virtual_b,indices_b,b,shape_c,virtual_c,indices_c,c,&alpha,&if root {1.} else {0.});
    for comm in c_comms {comm.reduce_f64(0,c);}
    if a_comms.iter().any(|comm|comm.rank() != 0) {a.fill(0.);}
    if b_comms.iter().any(|comm|comm.rank() != 0) {b.fill(0.);}
}

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
