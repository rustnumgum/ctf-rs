// Adapted from cc4s tensor/untyped_tensor.cxx:2922-3006 and permute_target.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Local tensor::fold metadata. This does not transpose or alias tensor storage.
use crate::symmetry::{Layout,Symmetry};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Direction { Forward, Backward }

#[derive(Clone,Debug,PartialEq,Eq)]
pub struct FoldLayout {
    /// Packed lengths of every original adjacent symmetry group.
    pub group_lengths: Vec<usize>,
    /// Source rec_tsr shape, before select_ctr_perm reorders its dimensions.
    pub folded_shape: Vec<usize>,
    /// Indices into the contraction-wide fold label list (source fidx).
    pub folded_indices: Vec<usize>,
    /// Original group IDs: selected prefix followed by the unselected remainder.
    pub inner_ordering: Vec<usize>,
}
impl FoldLayout {
    /// `local_shape` is source calc_dim's virtual-block shape, not global lengths.
    /// `indices` and `fold_labels` use the same contraction-wide normalized IDs.
    pub fn new(local_shape:&[usize],links:&[Symmetry],indices:&[usize],fold_labels:&[usize])->Self{
        assert_eq!(local_shape.len(),links.len());assert_eq!(indices.len(),links.len());
        if let Some(last)=links.last(){assert_eq!(*last,Symmetry::NS);}
        let mut group_lengths=Vec::new();let mut folded_shape=Vec::new();let mut folded_indices=Vec::new();
        let mut selected=Vec::new();let mut remaining=Vec::new();let mut start=0;
        for end in 0..links.len(){if links[end]==Symmetry::NS{
            let length=Layout::new(local_shape[start..=end].to_vec(),links[start..=end].to_vec()).len();
            let group=group_lengths.len();group_lengths.push(length);
            if let Some(label)=fold_labels.iter().position(|&label|label==indices[end]){
                selected.push(group);folded_shape.push(length);folded_indices.push(label);
            }else{remaining.push(group);}
            start=end+1;
        }}
        selected.extend(remaining);
        Self{group_lengths,folded_shape,folded_indices,inner_ordering:selected}
    }

    /// Source permute_target changes only the selected prefix; residual groups
    /// keep their original relative order. Shape and indices retain source order.
    pub fn permute_folded(&mut self,order:&[usize]){
        assert_eq!(order.len(),self.folded_shape.len());
        let mut sorted=order.to_vec();sorted.sort_unstable();assert_eq!(sorted,(0..order.len()).collect::<Vec<_>>());
        let old=self.inner_ordering[..order.len()].to_vec();
        for(i,&axis)in order.iter().enumerate(){self.inner_ordering[i]=old[axis];}
    }

    /// Source nosym_transpose over each local virtual block. Compressed groups
    /// remain packed dimensions; neither direction expands their symmetry.
    /// Data is owned independently rather than sharing CTF's raw alias pointers.
    pub fn transpose<T:Clone>(&self,data:&[T],virtual_blocks:usize,direction:Direction)->Vec<T>{
        let block:usize=self.group_lengths.iter().product();
        assert_eq!(data.len(),block*virtual_blocks);
        let mut strides=vec![0;self.group_lengths.len()];let mut stride=1;
        let identity:Vec<_>=(0..self.group_lengths.len()).collect();
        let (source_order,destination_order)=match direction{
            Direction::Forward=>(identity.as_slice(),self.inner_ordering.as_slice()),
            Direction::Backward=>(self.inner_ordering.as_slice(),identity.as_slice()),
        };
        for &axis in source_order{strides[axis]=stride;stride*=self.group_lengths[axis];}
        (0..data.len()).map(|offset|{
            let base=offset/block*block;let mut remainder=offset%block;let mut source=base;
            for &axis in destination_order{
                source+=(remainder%self.group_lengths[axis])*strides[axis];
                remainder/=self.group_lengths[axis];
            }data[source].clone()
        }).collect()
    }
}
