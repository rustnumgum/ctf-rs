// Canonical sum_tensors routing adapted from cc4s CTF summation.cxx and sparse_rw.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{algebra::{Group,Semiring,Wire},diagonal::Projection};
use super::SymmetricTensor;

impl<'c,'r,A:Group+Semiring+Clone> SymmetricTensor<'c,'r,A> where A::Element:Wire {
    /// Indexed sum with symmetry processing disabled, as in home_sum_tsr(...,
    /// false). Only the source and destination canonical domains participate.
    /// This is NOT orbit-expanded symmetrization or a symmetry-aware sum_from.
    /// Distributions are explicit; contributions route from unique source owners
    /// to destination replicas without gathering either global tensor.
    pub fn sum_canonical_from(&mut self,indices_b:&str,a:&Self,indices_a:&str,
                              alpha:A::Element,beta:A::Element) {
        assert!(std::ptr::eq(self.context,a.context));
        let mut aligned_output=indices_b.as_bytes().to_vec();
        let sign=crate::sym_indices::align_pair(indices_a.as_bytes(),a.distribution.links(),
            &mut aligned_output,self.distribution.links());
        assert_eq!(sign,1,"raw sum requires upper-layer antisymmetric sign handling");
        let indices_b=std::str::from_utf8(&aligned_output).unwrap();
        let da=a.distribution.distribution();let db=self.distribution.distribution();
        let input=Projection::new(&da.shape,indices_a);
        let output=Projection::new(&db.shape,indices_b);
        let mut broadcasts=1;
        let sources:Vec<_>=output.labels.bytes().enumerate().map(|(axis,label)| {
            let source=input.labels.bytes().position(|candidate|candidate==label);
            if let Some(source)=source {assert_eq!(input.shape[source],output.shape[axis]);}
            else {broadcasts*=output.shape[axis];}
            source
        }).collect();
        let mut contributions=Vec::new();
        for (key,value) in a.local_pairs() {
            if da.owner(key)!=self.context.rank(){continue;}
            let Some(coordinates)=input.project(&da.decode_key(key)) else {continue;};
            let value=self.algebra.multiply(&value,&alpha);
            for mut broadcast in 0..broadcasts {
                let target:Vec<_>=sources.iter().enumerate().map(|(axis,source)| {
                    if let Some(source)=source {coordinates[*source]}
                    else {let coordinate=broadcast%output.shape[axis];broadcast/=output.shape[axis];coordinate}
                }).collect();
                let key=db.encode_key(&output.expand(&target));
                if self.distribution.canonicalize(key)==Some((key,1)) {
                    contributions.push((key,value.clone()));
                }
            }
        }
        let algebra=self.algebra.clone();
        self.transform_indexed(indices_b,|value| *value=algebra.multiply(&beta,value));
        self.write_add(&contributions);
    }
}
