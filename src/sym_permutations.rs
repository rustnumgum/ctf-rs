// Adapted from cc4s symmetry/symmetrization.cxx order_perm/add_sym_perm/get_sym_perms.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Broken-symmetry operation permutations in upstream discovery order. Inputs
//! refer to fixed operands; deduplication compares their index maps, not signs.
use crate::{symmetry::Symmetry,sym_indices::{align_pair,align_triple}};
use Symmetry::{NS,AS};

#[derive(Clone,Debug,PartialEq,Eq)]
pub struct Permutation<const N:usize> {pub indices:[Vec<u8>;N],pub sign:i32}

fn normalize<const N:usize>(mut permutation:Permutation<N>,sym:[&[Symmetry];N])->Permutation<N> {
    loop {
        let mut changed=false;
        for operand in 0..N {
            for i in 0..sym[operand].len() {
                if sym[operand][i]==NS {continue;}
                // Source captures partner positions once before iterating j.
                let positions:[Option<usize>;N]=std::array::from_fn(|other|
                    permutation.indices[other].iter().rposition(|&l|l==permutation.indices[operand][i]));
                let mut j=i;
                while sym[operand][j]!=NS {
                    j+=1;
                    let preserved=(0..N).filter(|&other|other!=operand).all(|other| {
                        let right=permutation.indices[other].iter().rposition(|&l|l==permutation.indices[operand][j]);
                        match (positions[other],right) {
                            (None,None)=>true,(Some(left),Some(right))=>sym[other][left.min(right)..left.max(right)].iter().all(|&s|s!=NS),
                            _=>false,
                        }
                    });
                    if preserved && permutation.indices[operand][i]>permutation.indices[operand][j] {
                        permutation.indices[operand].swap(i,j);
                        if sym[operand][i]==AS {permutation.sign=-permutation.sign;}
                        changed=true;
                    }
                }
            }
        }
        if !changed {break;}
    }
    let [a,rest @ ..]=&mut permutation.indices[..] else {unreachable!()};
    if N==2 {
        permutation.sign*=align_pair(a,sym[0],&mut rest[0],sym[1]);
    } else {
        let (b,c)=rest.split_at_mut(1);
        permutation.sign*=align_triple(a,sym[0],&mut b[0],sym[1],&mut c[0],sym[2]);
    }
    permutation
}
fn add<const N:usize>(permutations:&mut Vec<Permutation<N>>,new:Permutation<N>,sym:[&[Symmetry];N]) {
    let normalized=normalize(new,sym);
    if !permutations.iter().any(|p|p.indices==normalized.indices) {permutations.push(normalized);}
}
/// Both get_sym_perms overloads use the same expanding-list transposition loop.
pub fn enumerate<const N:usize>(indices:[&[u8];N],sym:[&[Symmetry];N])->Vec<Permutation<N>> {
    assert!(N==2||N==3);
    for operand in 0..N {
        assert_eq!(indices[operand].len(),sym[operand].len());
        if let Some(last)=sym[operand].last() {assert_eq!(*last,NS);}
    }
    let mut permutations=Vec::new();
    add(&mut permutations,Permutation{indices:indices.map(<[u8]>::to_vec),sign:1},sym);
    for operand in 0..N {
        for i in 0..indices[operand].len() {
            let mut j=i;
            while sym[operand][j]!=NS {
                j+=1;let mut k=0;
                while k<permutations.len() {
                    let mut new=permutations[k].clone();
                    if sym[operand][j-1]==AS {new.sign=-new.sign;}
                    new.indices[operand].swap(i,j);add(&mut permutations,new,sym);k+=1;
                }
            }
        }
    }
    permutations
}

/// cmp_sym_perms: first symmetry group's circular generator and its order/sign.
pub fn circular_generator(sym:&[Symmetry])->(Vec<usize>,usize,i32) {
    assert!(!sym.is_empty() && sym[0]!=NS);
    let order=sym.iter().position(|&s|s==NS).unwrap()+1;
    let sign=if order%2==0 && sym[..order].contains(&AS) {-1} else {1};
    let permutation=(0..sym.len()).map(|i|if i<order {(i+1)%order} else {i}).collect();
    (permutation,order,sign)
}
