// Packed-size recurrence adapted from cc4s shared/util.cxx; offset summation
// follows shared/iter_tsr.h. Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Local compressed symmetry layout. Distributed symmetry mapping and contraction
//! multiplicities are separate from this packed storage and are not implied.
use crate::algebra::{Group,Monoid};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Symmetry { NS,SY,AS,SH }

/// Adjacent-axis links; each group ends with NS, as in the source sym array.
#[derive(Clone,Debug)]
pub struct Layout {shape:Vec<usize>,links:Vec<Symmetry>}
impl Layout {
    pub fn new(shape:Vec<usize>,links:Vec<Symmetry>)->Self {
        assert_eq!(shape.len(),links.len());
        if let Some(last)=links.last() {assert_eq!(*last,Symmetry::NS);}
        for i in 0..links.len().saturating_sub(1) {
            if links[i]!=Symmetry::NS {
                assert_eq!(shape[i],shape[i+1]);
                if links[i+1]!=Symmetry::NS {assert_eq!(links[i],links[i+1]);}
            }
        }
        Self{shape,links}
    }
    pub fn shape(&self)->&[usize] {&self.shape}
    pub fn links(&self)->&[Symmetry] {&self.links}
    fn groups(&self)->impl Iterator<Item=(usize,usize,Symmetry)>+'_ {
        let mut start=0;
        self.links.iter().enumerate().filter_map(move |(end,&link)| {
            if link!=Symmetry::NS {return None;}
            let group=(start,end+1,self.links[start]);start=end+1;Some(group)
        })
    }
    /// packed_size (AS/SH exclude repeated coordinates).
    pub fn len(&self)->usize {
        self.groups().map(|(start,end,kind)|group_size(self.shape[start],end-start,kind==Symmetry::SY)).product()
    }
    pub fn is_empty(&self)->bool {self.len()==0}
    /// sy_packed_size: used by upstream intermediate layouts even for AS/SH.
    pub fn symmetric_len(&self)->usize {
        self.groups().map(|(start,end,_)|group_size(self.shape[start],end-start,true)).product()
    }
    /// Canonical coordinate normalization; None denotes an AS/SH structural zero.
    /// Returns packed offset and antisymmetric parity (+1 or -1).
    pub fn locate(&self,coordinates:&[usize])->Option<(usize,i32)> {
        assert_eq!(coordinates.len(),self.shape.len());
        for (&c,&n) in coordinates.iter().zip(&self.shape) {assert!(c<n);}
        let mut offset=0;let mut stride=1;let mut sign=1;
        for (start,end,kind) in self.groups() {
            let mut group=coordinates[start..end].to_vec();
            // Adjacent swaps preserve the parity without relying on sort internals.
            for i in 1..group.len() {
                let mut j=i;
                while j>0 && group[j]<group[j-1] {
                    group.swap(j,j-1);if kind==Symmetry::AS {sign=-sign;}j-=1;
                }
            }
            if matches!(kind,Symmetry::AS|Symmetry::SH) && group.windows(2).any(|c|c[0]==c[1]) {return None;}
            let mut rank=0;
            for (i,&c) in group.iter().enumerate() {
                rank+=group_size(c,i+1,kind==Symmetry::SY);
            }
            offset+=stride*rank;stride*=group_size(self.shape[start],end-start,kind==Symmetry::SY);
        }
        Some((offset,sign))
    }
}
fn group_size(n:usize,order:usize,symmetric:bool)->usize {
    if !symmetric && n<order {return 0;}
    let mut product=1u128;
    for i in 0..order {
        let factor=if symmetric {n+i} else {n-i};
        product=product*factor as u128/(i+1) as u128;
    }
    product.try_into().unwrap()
}

/// Owned packed local data. No hidden unpack-to-dense storage or communications.
pub struct Packed<A:Monoid> {layout:Layout,algebra:A,values:Vec<A::Element>}
impl<A:Group> Packed<A> {
    pub fn new(layout:Layout,algebra:A)->Self {
        let values=vec![algebra.zero();layout.len()];Self{layout,algebra,values}
    }
    pub fn layout(&self)->&Layout {&self.layout}
    pub fn values(&self)->&[A::Element] {&self.values}
    pub fn values_mut(&mut self)->&mut [A::Element] {&mut self.values}
    pub fn read(&self,coordinates:&[usize])->A::Element {
        match self.layout.locate(coordinates) {
            None=>self.algebra.zero(),Some((offset,1))=>self.values[offset].clone(),
            Some((offset,_))=>self.algebra.negate(&self.values[offset]),
        }
    }
    pub fn write_add(&mut self,coordinates:&[usize],value:&A::Element) {
        if let Some((offset,sign))=self.layout.locate(coordinates) {
            let value=if sign==1 {value.clone()} else {self.algebra.negate(value)};
            self.values[offset]=self.algebra.add(&self.values[offset],&value);
        }
    }
}
