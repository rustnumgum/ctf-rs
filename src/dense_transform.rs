// Adapted from cc4s CTF bivariate accumulator transforms and indexed I/O.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{algebra::{Monoid,Wire},diagonal::Projection,tensor::Tensor};

impl<A:Monoid> Tensor<'_, '_, A>{
    /// Apply a bivariate accumulator to each selected output entry. Input
    /// indices must be subsets of the output indices; no reduction or additive
    /// interpretation of the mutable-output function is assumed. Repeated
    /// output indices restrict updates to the selected diagonal.
    pub fn transform_from<I:Monoid,J:Monoid>(&mut self,indices_c:&str,
        a:&Tensor<'_, '_, I>,indices_a:&str,b:&Tensor<'_, '_, J>,indices_b:&str,
        mut function:impl FnMut(&I::Element,&J::Element,&mut A::Element))
        where I::Element:Wire,J::Element:Wire {
        assert!(std::ptr::eq(self.context(),a.context())&&std::ptr::eq(self.context(),b.context()));
        let output=Projection::new(&self.distribution().shape,indices_c);
        let inputs=[Projection::new(&a.distribution().shape,indices_a),
            Projection::new(&b.distribution().shape,indices_b)];
        let axes:Vec<Vec<usize>>=inputs.iter().map(|input|input.labels.bytes().enumerate().map(|(axis,label)|{
            let at=output.labels.bytes().position(|candidate|candidate==label)
                .expect("accumulator input indices must occur in output");
            assert_eq!(input.shape[axis],output.shape[at]);at
        }).collect()).collect();
        let mut keys=[Vec::new(),Vec::new()];
        let distributions=[a.distribution(),b.distribution()];
        for offset in 0..self.distribution().local_len(){
            let Some(key)=self.distribution().global_key(self.context().rank(),offset)else{continue};
            let Some(coordinates)=output.project(&self.distribution().decode_key(key))else{continue};
            for input in 0..2{
                let indices:Vec<_>=axes[input].iter().map(|&axis|coordinates[axis]).collect();
                keys[input].push(distributions[input].encode_key(&inputs[input].expand(&indices)));
            }
        }
        let aa=a.read(&keys[0]);let bb=b.read(&keys[1]);
        let mut values=aa.iter().zip(&bb);
        self.transform_indexed(indices_c,|output|{
            let(a,b)=values.next().unwrap();function(a,b,output);
        });
    }
}
