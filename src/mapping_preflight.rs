// Adapted from cc4s contraction::check_mapping and check_self_mapping.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Mapping preflight for nonsymmetric, unique-label contraction candidates.
use crate::mapping::{Distribution,Mapping};
fn physical(map:&Mapping)->Vec<usize>{
    let mut result=Vec::new();let mut current=map;
    while let Mapping::Physical{axis,child,..}=current{result.push(*axis);current=child;}
    result
}
fn mark(axes:impl IntoIterator<Item=usize>,used:&mut[bool])->bool{
    for axis in axes{if used[axis]{return false;}used[axis]=true;}true
}
/// Preserve source shared-label phase checks and 2D mismatch bookkeeping.
/// Diagonals must be projected before this unique-label candidate check.
pub fn check(distributions:[&Distribution;3],indices:[&str;3])->bool{
    let topology=&distributions[0].topology;
    if distributions.iter().any(|d|&d.topology!=topology){return false;}
    let mut labels=Vec::new();let mut dimensions=Vec::new();
    for operand in 0..3{
        assert!(indices[operand].is_ascii());assert_eq!(indices[operand].len(),distributions[operand].shape.len());
        let mut used=vec![false;topology.dimensions.len()];
        for(axis,label)in indices[operand].bytes().enumerate(){
            assert!(!indices[operand].as_bytes()[..axis].contains(&label));
            if let Some(at)=labels.iter().position(|&old|old==label){
                assert_eq!(dimensions[at],distributions[operand].shape[axis]);
            }else{labels.push(label);dimensions.push(distributions[operand].shape[axis]);}
            let chain=physical(&distributions[operand].mappings[axis]);
            // Literal nested source check: each subsequent physical child must
            // be exactly one topology axis after the current physical node.
            for i in 0..chain.len(){if chain[i+1..].iter().any(|&j|j!=chain[i]+1){return false;}}
            if !mark(chain,&mut used){return false;}
        }
    }
    let maps:Vec<[Option<&Mapping>;3]>=labels.iter().map(|&label|std::array::from_fn(|operand|
        indices[operand].bytes().position(|x|x==label).map(|axis|&distributions[operand].mappings[axis]))).collect();
    let mut mapped=vec![false;topology.dimensions.len()];let mut mismatched=mapped.clone();
    for entry in &maps{
        if let [Some(a),Some(b),Some(c)]=entry{
            if a!=b||b!=c{return false;}
            let axes=physical(a);
            if !mark(axes.iter().copied(),&mut mapped){return false;}
            for axis in axes{mismatched[axis]=true;}
        }
    }
    for entry in &maps{
        if entry.iter().all(Option::is_some){continue;}
        for(left,right,other)in[(0,1,2),(0,2,1),(2,1,0)]{
            if entry[other].is_some(){continue;}
            match(entry[left],entry[right]){
                (None,None)=>{},
                (Some(a),None)|(None,Some(a))=>{if !mark(physical(a),&mut mapped){return false;}},
                (Some(a),Some(b))=>{
                    if a.phase()!=b.phase(){return false;}
                    if a==b{
                        // Source marks only the head of an equal pair mapping.
                        if let Mapping::Physical{axis,..}=a{if !mark([*axis],&mut mapped){return false;}}
                    }else{
                        let aa=physical(a);if aa.len()>1{return false;}
                        if !mark(aa,&mut mismatched)||!mark(physical(b),&mut mismatched){return false;}
                    }
                }
            }
        }
    }
    mismatched.iter().zip(mapped).all(|(&m,p)|!m||p)
}
