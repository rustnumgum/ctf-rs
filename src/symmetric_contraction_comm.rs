// Adapted from cc4s CTF ctr_virt and ctr_replicate.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{algebra::{Group,Semiring,Wire},context::Context,summation::Indices,
    symmetry::Layout,symmetric_contraction::sequential};

/// Raw packed virtual blocks, with caller-aligned label phases and local
/// extents. Beta applies once per visited output block.
pub fn virtualized<A:Group+Semiring>(algebra:&A,
    la:&Layout,va:&[usize],ia:&str,a:&[A::Element],
    lb:&Layout,vb:&[usize],ib:&str,b:&[A::Element],
    lc:&Layout,vc:&[usize],ic:&str,c:&mut[A::Element],
    alpha:&A::Element,beta:&A::Element) {
    for (layout,phases) in [(la,va),(lb,vb),(lc,vc)] {
        assert_eq!(layout.shape().len(),phases.len());
        assert!(phases.iter().all(|&p|p>0));
    }
    let sa=la.symmetric_len();let sb=lb.symmetric_len();let sc=lc.symmetric_len();
    let nc:usize=vc.iter().product();
    assert_eq!(a.len(),sa*va.iter().product::<usize>());
    assert_eq!(b.len(),sb*vb.iter().product::<usize>());assert_eq!(c.len(),sc*nc);
    let space=Indices::new(&[(va,ia),(vb,ib),(vc,ic)]);
    let mut visited=vec![false;nc];let one=algebra.one();
    space.for_each(|offsets| {
        let (xa,xb,xc)=(offsets[0],offsets[1],offsets[2]);
        sequential(algebra,la,ia,&a[xa*sa..(xa+1)*sa],lb,ib,&b[xb*sb..(xb+1)*sb],
            lc,ic,&mut c[xc*sc..(xc+1)*sc],alpha,if visited[xc]{&one}else{beta});
        visited[xc]=true;
    });
}

/// Explicit ctr_replicate: broadcast both operands, retain beta-scaled old C
/// on roots, execute children, Reduce C (not Allreduce), clear input replicas.
/// Diagonal extraction, global padding cleanup and operation multiplicities
/// belong to the caller's upper-layer plan.
pub fn replicated<A:Group+Semiring>(algebra:&A,
    a_comms:&[&Context<'_>],b_comms:&[&Context<'_>],c_comms:&[&Context<'_>],
    la:&Layout,va:&[usize],ia:&str,a:&mut[A::Element],
    lb:&Layout,vb:&[usize],ib:&str,b:&mut[A::Element],
    lc:&Layout,vc:&[usize],ic:&str,c:&mut[A::Element],
    alpha:&A::Element,beta:&A::Element,commutative:bool) where A::Element:Wire {
    for comm in a_comms {comm.broadcast(0,a);}
    for comm in b_comms {comm.broadcast(0,b);}
    let root=c_comms.iter().all(|comm|comm.rank()==0);
    if root && *beta!=algebra.one() {
        for value in c.iter_mut() {
            *value=if *beta==algebra.zero(){algebra.zero()}else{algebra.multiply(beta,value)};
        }
    }
    let child_beta=if root{algebra.one()}else{algebra.zero()};
    virtualized(algebra,la,va,ia,a,lb,vb,ib,b,lc,vc,ic,c,alpha,&child_beta);
    for comm in c_comms {comm.reduce_monoid(algebra,c,commutative,0);}
    if a_comms.iter().any(|comm|comm.rank()!=0) {a.fill(algebra.zero());}
    if b_comms.iter().any(|comm|comm.rank()!=0) {b.fill(algebra.zero());}
}
