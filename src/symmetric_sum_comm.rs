// Adapted from cc4s CTF tsum_virt and tsum_replicate.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{algebra::{Group,Semiring,Wire},context::Context,
    summation::Indices,symmetry::Layout,symmetric_sum::sequential};

/// Execute explicit packed virtual blocks. The mapper must already have aligned
/// shared index phases and block lengths; global padding is a caller concern.
pub fn virtualized<A:Group+Semiring>(algebra:&A,
    layout_a:&Layout, virtual_a:&[usize], indices_a:&str, a:&[A::Element],
    layout_b:&Layout, virtual_b:&[usize], indices_b:&str, b:&mut[A::Element],
    alpha:&A::Element,beta:&A::Element) {
    assert_eq!(layout_a.shape().len(),virtual_a.len());
    assert_eq!(layout_b.shape().len(),virtual_b.len());
    assert!(virtual_a.iter().chain(virtual_b).all(|&n| n>0));
    let size_a=layout_a.symmetric_len();let size_b=layout_b.symmetric_len();
    let count_a:usize=virtual_a.iter().product();let count_b:usize=virtual_b.iter().product();
    assert_eq!(a.len(),size_a*count_a);assert_eq!(b.len(),size_b*count_b);
    let space=Indices::new(&[(virtual_a,indices_a),(virtual_b,indices_b)]);
    let mut visited=vec![false;count_b];let one=algebra.one();
    space.for_each(|offsets| {
        let ia=offsets[0];let ib=offsets[1];
        sequential(algebra,layout_a,indices_a,&a[ia*size_a..(ia+1)*size_a],
            layout_b,indices_b,&mut b[ib*size_b..(ib+1)*size_b],alpha,
            if visited[ib] {&one} else {beta});
        visited[ib]=true;
    });
}

/// Source replication schedule: input-fiber broadcasts, old output on roots
/// only, virtual execution, then output-fiber reductions in supplied order.
/// This low-level executor does not perform automatic symmetry mapping or
/// broken-symmetry permutation expansion.
pub fn replicated<A:Group+Semiring>(algebra:&A,
    input_comms:&[&Context<'_>],output_comms:&[&Context<'_>],
    layout_a:&Layout,virtual_a:&[usize],indices_a:&str,a:&mut[A::Element],
    layout_b:&Layout,virtual_b:&[usize],indices_b:&str,b:&mut[A::Element],
    alpha:&A::Element,beta:&A::Element,commutative:bool) where A::Element:Wire {
    for comm in input_comms {comm.broadcast(0,a);}
    let root=output_comms.iter().all(|comm|comm.rank()==0);
    if !root {b.fill(algebra.zero());}
    let one=algebra.one();
    virtualized(algebra,layout_a,virtual_a,indices_a,a,layout_b,virtual_b,indices_b,b,
        alpha,if root {beta} else {&one});
    for comm in output_comms {comm.all_reduce_monoid(algebra,b,commutative);}
}
