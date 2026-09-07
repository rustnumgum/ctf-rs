// Adapted from cc4s CTF mapping/mapping.cxx::{map_self_indices,check_self_mapping}
// at f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{map_tensor::{coordinate_symmetry,Rejected},mapping::Mapping};

/// Mark earlier repeated-index dimensions virtual. If this creates any new
/// virtual map, source deliberately skips symmetry coordination on this call.
pub fn map_self_indices(maps:&mut[Mapping],indices:&[usize],symmetry_table:&[bool])->Result<(),Rejected>{
    let n=maps.len();assert_eq!(indices.len(),n);assert_eq!(symmetry_table.len(),n*n);
    let mut previous=vec![None;indices.iter().max().map_or(0,|i|i+1)];
    let mut table=symmetry_table.to_vec();let mut newly_virtual=false;
    for (axis,&label) in indices.iter().enumerate(){
        if let Some(other)=previous[label]{
            table[other*n+axis]=true;table[axis*n+other]=true;
            if matches!(maps[other],Mapping::Unmapped){
                maps[other]=Mapping::Virtual{copies:1,child:Box::new(Mapping::Unmapped)};
                newly_virtual=true;
            }
        }
        previous[label]=Some(axis);
    }
    if !newly_virtual{coordinate_symmetry(maps,&table)?;}
    Ok(())
}

/// Literal source self-map checks, including the reverse repeated-index walk
/// and the stricter-than-adjacency physical descendant test.
pub fn check_self_mapping(maps:&[Mapping],indices:&[usize])->bool{
    assert_eq!(maps.len(),indices.len());
    for (dimension,map) in maps.iter().enumerate(){
        let mut current=map;
        while let Mapping::Physical{axis,child,..}=current{
            let mut descendant=child.as_ref();
            loop{
                match descendant{
                    Mapping::Physical{axis:next,child,..}=>{
                        if *next!=axis+1{return false;}descendant=child;
                    },
                    Mapping::Virtual{child,..}=>descendant=child,
                    Mapping::Unmapped=>break,
                }
            }
            for other in &maps[dimension+1..]{
                let mut other=other;
                while let Mapping::Physical{axis:other_axis,child,..}=other{
                    if axis==other_axis{return false;}other=child;
                }
            }
            current=child;
        }
    }
    let mut last:Vec<Option<usize>>=vec![None;indices.iter().max().map_or(0,|i|i+1)];
    for (axis,&label) in indices.iter().enumerate().rev(){
        if let Some(other)=last[label]{
            for map in [&maps[axis],&maps[other]]{
                match map{
                    Mapping::Physical{child,..}|Mapping::Virtual{child,..}=>{
                        if !matches!(**child,Mapping::Unmapped){return false;}
                    },
                    Mapping::Unmapped=>{},
                }
            }
            if matches!(maps[axis],Mapping::Physical{..})||maps[axis].phase()!=maps[other].phase(){return false;}
        }else{last[label]=Some(axis);}
    }
    true
}
