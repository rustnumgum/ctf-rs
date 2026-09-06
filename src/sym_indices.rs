// Adapted from cc4s symmetry/sym_indices.{h,cxx}, written by Devin Matthews.
// See LICENSE for the upstream CTF license.
//! Symmetry index bookkeeping; these factors assume the corresponding upstream
//! aligned index ordering and are not a replacement for full symmetrization.
use crate::symmetry::Symmetry;
use Symmetry::{NS,AS,SY};

pub fn relative_sign<T:PartialEq>(first:&[T],second:&[T])->i32 {
    assert_eq!(first.len(),second.len());let mut seen=vec![false;first.len()];let mut sign=1;
    for i in 0..first.len() {
        if seen[i] {continue;}
        let mut j=i;
        loop {
            let k=(0..first.len()).find(|&k|first[k]==second[j]&&!seen[k]).expect("not a permutation");
            j=k;seen[j]=true;if j==i {break;}sign=-sign;
        }
    }
    sign
}
fn validate(indices:&[u8],sym:&[Symmetry]) {
    assert_eq!(indices.len(),sym.len());if let Some(last)=sym.last() {assert_eq!(*last,NS);}
}
fn same_group(sym:&[Symmetry],a:usize,b:usize)->bool {
    sym[a.min(b)..a.max(b)].iter().all(|&s|s!=NS)
}
/// Three-operand align_symmetric_indices. Groups must have the same operand
/// incidence and lie in a shared symmetry group in every participating operand.
pub fn align_triple(a:&[u8],sym_a:&[Symmetry],b:&mut [u8],sym_b:&[Symmetry],c:&mut [u8],sym_c:&[Symmetry])->i32 {
    validate(a,sym_a);validate(b,sym_b);validate(c,sym_c);
    #[derive(Clone)]
    struct Locator {order:usize,label:u8,positions:[Option<usize>;3]}
    let mut indices=Vec::new();
    for (i,&label) in a.iter().enumerate() {
        let ib=b.iter().position(|&x|x==label);let ic=c.iter().position(|&x|x==label);
        if ib.is_some()||ic.is_some() {indices.push(Locator{order:0,label,positions:[Some(i),ib,ic]});}
    }
    for (i,&label) in b.iter().enumerate() {
        if !a.contains(&label) {
            if let Some(ic)=c.iter().position(|&x|x==label) {indices.push(Locator{order:0,label,positions:[None,Some(i),Some(ic)]});}
        }
    }
    let symbols=[sym_a,sym_b,sym_c];let mut factor=1;
    while !indices.is_empty() {
        let mut group=vec![indices.remove(0)];let mut i=0;
        while i<indices.len() {
            let compatible=(0..3).all(|operand|match (group[0].positions[operand],indices[i].positions[operand]) {
                (None,None)=>true,(Some(x),Some(y))=>same_group(symbols[operand],x,y),_=>false,
            });
            if compatible {let mut entry=indices.remove(i);entry.order=group.len();group.push(entry);} else {i+=1;}
        }
        if group.len()<2 {continue;}
        let reference:Vec<_>=group.iter().map(|entry|entry.order).collect();
        let has_a=group[0].positions[0].is_some();
        for operand in if has_a {1..3} else {2..3} {
            if group[0].positions[operand].is_none() {continue;}
            group.sort_by_key(|entry|entry.positions[operand]);
            let order:Vec<_>=group.iter().map(|entry|entry.order).collect();
            for i in 0..group.len() {
                let destination=group[group[i].order].positions[operand].unwrap();
                if operand==1 {b[destination]=group[i].label;} else {c[destination]=group[i].label;}
            }
            if symbols[operand][group[0].positions[operand].unwrap()]==AS {factor*=relative_sign(&reference,&order);}
        }
    }
    factor
}
/// Two-operand align_symmetric_indices. A stays fixed; reorder common symmetry
/// groups in B and return the associated antisymmetric sign.
pub fn align_pair(a:&[u8],sym_a:&[Symmetry],b:&mut [u8],sym_b:&[Symmetry])->i32 {
    validate(a,sym_a);validate(b,sym_b);
    // Tuple: original group ordinal, label, position A, position B.
    let mut indices:Vec<_>=a.iter().enumerate().filter_map(|(i,&label)|b.iter().position(|&x|x==label).map(|j|(0,label,i,j))).collect();
    let mut factor=1;
    while !indices.is_empty() {
        let mut group=vec![indices.remove(0)];let mut i=0;
        while i<indices.len() {
            if same_group(sym_a,group[0].2,indices[i].2)&&same_group(sym_b,group[0].3,indices[i].3) {
                let mut item=indices.remove(i);item.0=group.len();group.push(item);
            } else {i+=1;}
        }
        if group.len()<2 {continue;}
        let order_a:Vec<_>=group.iter().map(|item|item.0).collect();
        group.sort_by_key(|item|item.3);
        let order_b:Vec<_>=group.iter().map(|item|item.0).collect();
        for i in 0..group.len() {b[group[group[i].0].3]=group[i].1;}
        if sym_b[group[0].3]==AS {factor*=relative_sign(&order_a,&order_b);}
    }
    factor
}
/// Three-operand overcounting_factor, for already aligned contraction indices.
pub fn contraction_factor(a:&[u8],sym_a:&[Symmetry],b:&[u8],sym_b:&[Symmetry],c:&[u8])->usize {
    validate(a,sym_a);validate(b,sym_b);let mut factor=1;let mut i=0;
    while i<a.len() {
        if let Some(mut j)=b.iter().position(|&x|x==a[i]) {
            if !c.contains(&a[i]) {
                let mut run=1;
                while i<a.len()&&j<b.len()&&sym_a[i]!=NS&&sym_b[j]!=NS&&a[i]==b[j] {
                    run+=1;i+=1;j+=1;
                }
                if i<a.len()&&j<b.len()&&a[i]!=b[j] {run-=1;}
                for n in 2..=run {factor*=n;}
            }
        }
        i+=1;
    }
    factor
}
/// Two-operand overcounting_factor for summation. Fully reduced AS groups cancel;
/// SH groups use factorial multiplicity, while SY is handled by its separate path.
pub fn summation_factor(a:&[u8],sym_a:&[Symmetry],b:&[u8])->usize {
    validate(a,sym_a);let mut factor=1;let mut i=0;
    while i<a.len() {
        let mut run=0;
        if !b.contains(&a[i]) {
            run=1;
            while sym_a[i]!=NS {i+=1;if !b.contains(&a[i]) {run+=1;}}
        }
        if run>=2 {
            if sym_a[i+1-run]==AS {return 0;}
            if sym_a[i+1-run]!=SY {for n in 2..=run {factor*=n;}}
        }
        i+=1;
    }
    factor
}
