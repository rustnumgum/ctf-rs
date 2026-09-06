// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Distributed dense storage and key-based all-to-all read/write.
use crate::{algebra::{Monoid, Semiring, Wire}, context::Context, mapping::Distribution};

#[derive(Debug)]
pub enum BlasContractionError { Mapping(crate::map_tensor::Rejected), Folding(crate::folding::Rejected) }

impl Tensor<'_, '_, crate::algebra::Arithmetic<f64>> {
    /// Fully foldable unique-label dense contraction on an explicit grid. Uses
    /// upstream aligned replication/virtual traversal with a folded BLAS child.
    /// Unsupported folds return an error before any tensor redistribution; no
    /// reference-kernel fallback or global gather is hidden here.
    pub fn contract_blas_on_grid<K:crate::linalg::LocalKernels>(&mut self,indices_c:&str,
        a:&Self,indices_a:&str,b:&Self,indices_b:&str,topology:crate::mapping::Topology,
        alpha:f64,beta:f64)->Result<(),BlasContractionError> {
        assert!(std::ptr::eq(self.context,a.context)&&std::ptr::eq(self.context,b.context));
        assert_eq!(topology.size(),self.context.size());
        let indices=[indices_a,indices_b,indices_c];
        // Check semantic foldability before asking the mapping layer to assign it.
        crate::folding::Plan::new([&a.distribution.shape,&b.distribution.shape,&self.distribution.shape],indices)
            .map_err(BlasContractionError::Folding)?;
        let plan=crate::planning::GridPlan::prepare([&a.distribution,&b.distribution,&self.distribution],indices,topology)
            .map_err(BlasContractionError::Mapping)?;
        let mapped=plan.mapped_distributions();
        let shapes:Vec<_>=mapped.iter().map(|d|d.block_shape()).collect();
        let local=crate::folding::Plan::new([&shapes[0],&shapes[1],&shapes[2]],indices)
            .map_err(BlasContractionError::Folding)?;
        let phases:Vec<Vec<_>>=mapped.iter().map(|d|d.mappings.iter().map(|m|m.phase()/m.physical_phase()).collect()).collect();
        let block_sizes:Vec<usize>=shapes.iter().map(|shape|shape.iter().product()).collect();
        let mut aa=Self{context:a.context,algebra:a.algebra,distribution:a.distribution.clone(),data:a.data.clone()};
        let mut bb=Self{context:b.context,algebra:b.algebra,distribution:b.distribution.clone(),data:b.data.clone()};
        let mut cc=Self{context:self.context,algebra:self.algebra,distribution:self.distribution.clone(),data:self.data.clone()};
        aa.redistribute(mapped[0].clone());bb.redistribute(mapped[1].clone());cc.redistribute(mapped[2].clone());
        fn mark(map:&crate::mapping::Mapping,used:&mut [bool]) {
            use crate::mapping::Mapping;
            match map {Mapping::Unmapped=>{},Mapping::Virtual{child,..}=>mark(child,used),
                Mapping::Physical{axis,child,..}=>{used[*axis]=true;mark(child,used);}}
        }
        let topology=&mapped[0].topology;
        let mut used=vec![vec![false;topology.dimensions.len()];3];
        for operand in 0..3 {for map in &mapped[operand].mappings {mark(map,&mut used[operand]);}}
        let mut comms:[Vec<Context<'_>>;3]=std::array::from_fn(|_|Vec::new());
        for axis in 0..topology.dimensions.len() {
            if !(0..3).any(|operand|used[operand][axis]) {continue;}
            for operand in 0..3 {if !used[operand][axis] {comms[operand].push(topology.fiber(self.context,axis));}}
        }
        for comm in &comms[0] {comm.broadcast(0,&mut aa.data);}
        for comm in &comms[1] {comm.broadcast(0,&mut bb.data);}
        let root=comms[2].iter().all(|comm|comm.rank()==0);
        if root {
            if beta==0. {cc.data.fill(0.);} else if beta!=1. {for value in &mut cc.data {*value*=beta;}}
        }
        let space=crate::summation::Indices::new(&[(&phases[0],indices_a),(&phases[1],indices_b),(&phases[2],indices_c)]);
        let mut visited=vec![false;phases[2].iter().product()];
        space.for_each(|offsets| {
            let [ia,ib,ic]=[offsets[0],offsets[1],offsets[2]];
            local.execute::<K>(&aa.data[ia*block_sizes[0]..(ia+1)*block_sizes[0]],
                &bb.data[ib*block_sizes[1]..(ib+1)*block_sizes[1]],
                &mut cc.data[ic*block_sizes[2]..(ic+1)*block_sizes[2]],alpha,
                if root||visited[ic] {1.} else {0.});
            visited[ic]=true;
        });
        for comm in &comms[2] {comm.reduce_f64(0,&mut cc.data);}
        for group in comms {for comm in group {comm.close();}}
        cc.redistribute(self.distribution.clone());self.data=cc.data;Ok(())
    }
    /// Explicit-grid distributed matrix multiplication C = alpha*A*B + beta*C.
    /// Cyclic k phases are equalized to lcm(grid rows, grid columns), then the
    /// upstream 2D panel executor drives local BLAS. Original distributions stay.
    pub fn gemm_2d<K: crate::linalg::LocalKernels>(&mut self, a: &Self,b: &Self,grid: [usize;2],alpha: f64,beta: f64) {
        use crate::{mapping::{Mapping,Topology},ctr_2d::{Panel,Layers},contraction::Folded,linalg::Transpose};
        assert!(std::ptr::eq(self.context,a.context) && std::ptr::eq(self.context,b.context));
        assert_eq!(a.distribution.shape.len(),2); assert_eq!(b.distribution.shape.len(),2); assert_eq!(self.distribution.shape.len(),2);
        let (m,k,n) = (a.distribution.shape[0],a.distribution.shape[1],b.distribution.shape[1]);
        assert_eq!(b.distribution.shape[0],k); assert_eq!(self.distribution.shape,vec![m,n]);
        let topology = Topology::new(grid.to_vec()); assert_eq!(topology.size(),self.context.size());
        let (mut x,mut y) = (grid[0],grid[1]); while y != 0 {(x,y) = (y,x%y);}
        let steps = grid[0]/x*grid[1];
        let mut row = Mapping::Unmapped; row.augment_physical(&topology,0);
        let mut column = Mapping::Unmapped; column.augment_physical(&topology,1);
        let mut krow = row.clone(); krow.augment_virtual(steps);
        let mut kcolumn = column.clone(); kcolumn.augment_virtual(steps);
        let mut aa = Self {context:a.context,algebra:a.algebra,distribution:a.distribution.clone(),data:a.data.clone()};
        let mut bb = Self {context:b.context,algebra:b.algebra,distribution:b.distribution.clone(),data:b.data.clone()};
        let mut cc = Self {context:self.context,algebra:self.algebra,distribution:self.distribution.clone(),data:self.data.clone()};
        aa.redistribute(Distribution::new(vec![m,k],topology.clone(),vec![row.clone(),kcolumn]));
        bb.redistribute(Distribution::new(vec![k,n],topology.clone(),vec![krow,column.clone()]));
        cc.redistribute(Distribution::new(vec![m,n],topology.clone(),vec![row,column]));
        let ma = m.div_ceil(grid[0]); let nb = n.div_ceil(grid[1]); let kb = k.div_ceil(steps);
        let across_columns = topology.fiber(self.context,1); let across_rows = topology.fiber(self.context,0);
        crate::ctr_2d::execute(steps,Layers {count:1,index:0},
            Panel {comm:Some(&across_columns),outer:1,inner:ma*kb},
            Panel {comm:Some(&across_rows),outer:1,inner:kb*nb},Panel {comm:None,outer:1,inner:0},
            &aa.data,&bb.data,&mut cc.data,beta,|a,b,c,beta,_| {
                crate::contraction::folded_f64::<K>(Folded {m:ma,n:nb,k:kb,batches:1,
                    trans_a:Transpose::No,trans_b:Transpose::No,transposed_output:false},a,b,c,alpha,beta);
            });
        across_columns.close(); across_rows.close();
        cc.redistribute(self.distribution.clone()); self.data = cc.data;
    }
}

impl<A: Semiring> Tensor<'_, '_, A> where A::Element: Wire {
    /// Distributed unary sum; f is applied to each alpha-scaled input before
    /// any reduction, preserving custom-function ordering for nonlinear f.
    pub fn sum_function_from(&mut self,indices_b:&str,input:&Self,indices_a:&str,topology:crate::mapping::Topology,
        alpha:A::Element,beta:A::Element,function:impl Fn(&A::Element)->A::Element)->Result<(),crate::map_tensor::Rejected> where A:Clone {
        let mut transformed=Self {context:input.context,algebra:input.algebra.clone(),distribution:input.distribution.clone(),data:input.data.clone()};
        for (offset,value) in transformed.data.iter_mut().enumerate() {
            if transformed.distribution.global_key(transformed.context.rank(),offset).is_some() {*value=function(&input.algebra.multiply(value,&alpha));}
        }
        self.sum_from(indices_b,&transformed,indices_a,topology,self.algebra.one(),beta)
    }
    /// Dense slice insertion following extract/remap/rank-shift/scatter. Beta is
    /// applied only inside the destination slice, including empty local shards.
    pub fn assign_slice(&mut self,ranges:&[std::ops::Range<usize>],source:&Self,source_ranges:&[std::ops::Range<usize>],
        alpha:&A::Element,beta:&A::Element) where A:Clone {
        assert!(std::ptr::eq(self.context,source.context));assert_eq!(ranges.len(),self.distribution.shape.len());
        for (r,&n) in ranges.iter().zip(&self.distribution.shape) {assert!(r.start<=r.end&&r.end<=n);}
        let shape:Vec<_>=ranges.iter().map(|r|r.end-r.start).collect();
        let offsets:Vec<_>=ranges.iter().map(|r|r.start).collect();
        let mut part=source.slice(source_ranges);assert_eq!(part.distribution.shape,shape);
        let mut distribution=self.distribution.clone();distribution.shape=shape;part.redistribute(distribution);
        let send=self.distribution.shifted_rank(self.context.rank(),&offsets,true);
        let recv=self.distribution.shifted_rank(self.context.rank(),&offsets,false);
        if send!=self.context.rank()&&!part.data.is_empty() {
            let mut bytes=Vec::with_capacity(part.data.len()*A::Element::WIDTH);for value in &part.data {value.encode(&mut bytes);}
            let mut received=vec![0u8;bytes.len()];self.context.inner.send_receive(&bytes,send,recv,&mut received);
            for (value,bytes) in part.data.iter_mut().zip(received.chunks_exact(A::Element::WIDTH)) {*value=A::Element::decode(bytes);}
        }
        for (offset,value) in part.data.iter().enumerate() {
            if let Some(key)=part.distribution.global_key(recv,offset) {
                let coordinates:Vec<_>=part.distribution.decode_key(key).iter().zip(&offsets).map(|(&c,&o)|c+o).collect();
                let key=self.distribution.encode_key(&coordinates);let index=self.distribution.local_offset(self.context.rank(),key);
                let previous=if *beta==self.algebra.zero() {self.algebra.zero()} else {self.algebra.multiply(&self.data[index],beta)};
                self.data[index]=self.algebra.add(&self.algebra.multiply(value,alpha),&previous);
            }
        }
    }
    /// Distributed indexed sum, including repeated input/output labels.
    pub fn sum_from(&mut self,indices_b:&str,input:&Self,indices_a:&str,topology:crate::mapping::Topology,
        alpha:A::Element,beta:A::Element)->Result<(),crate::map_tensor::Rejected> where A:Clone {
        let pa=crate::diagonal::Projection::new(&input.distribution.shape,indices_a);
        let pb=crate::diagonal::Projection::new(&self.distribution.shape,indices_b);
        if !pa.repeated()&&!pb.repeated() {return self.sum_from_on_grid(indices_b,input,indices_a,topology,alpha,beta);}
        let (a,ia)=input.extract_diagonal(indices_a);let (mut b,ib)=self.extract_diagonal(indices_b);
        b.sum_from_on_grid(&ib,&a,&ia,topology,alpha,beta)?;
        self.replace_diagonal(indices_b,&b);Ok(())
    }
    /// Distributed reference contraction including repeated labels. The explicit
    /// grid execution remains distinct from cost-based/BLAS-folded planning.
    pub fn contract_from(&mut self,indices_c:&str,a:&Self,indices_a:&str,b:&Self,indices_b:&str,
        topology:crate::mapping::Topology,alpha:A::Element,beta:A::Element)->Result<(),crate::map_tensor::Rejected> where A:Clone {
        let pa=crate::diagonal::Projection::new(&a.distribution.shape,indices_a);
        let pb=crate::diagonal::Projection::new(&b.distribution.shape,indices_b);
        let pc=crate::diagonal::Projection::new(&self.distribution.shape,indices_c);
        if !pa.repeated()&&!pb.repeated()&&!pc.repeated() {return self.contract_from_on_grid(indices_c,a,indices_a,b,indices_b,topology,alpha,beta);}
        let (aa,ia)=a.extract_diagonal(indices_a);let (bb,ib)=b.extract_diagonal(indices_b);let (mut cc,ic)=self.extract_diagonal(indices_c);
        cc.contract_from_on_grid(&ic,&aa,&ia,&bb,&ib,topology,alpha,beta)?;
        self.replace_diagonal(indices_c,&cc);Ok(())
    }
    /// Collective affine key write. Duplicate incoming keys are summed before
    /// alpha/beta are applied once; keys absent from the request are unchanged.
    pub fn write_scaled(&mut self,pairs:&[(usize,A::Element)],alpha:&A::Element,beta:&A::Element) {
        let mut buckets=vec![Vec::new();self.context.size()];
        for (key,value) in pairs {for (rank,bucket) in buckets.iter_mut().enumerate() {
            if self.distribution.owns(rank,*key) {(*key as u64).encode(bucket);value.encode(bucket);}
        }}
        let mut incoming=std::collections::BTreeMap::new();
        for bytes in self.context.inner.exchange(&buckets) {for pair in bytes.chunks_exact(8+A::Element::WIDTH) {
            let key=u64::decode(&pair[..8]) as usize;let value=A::Element::decode(&pair[8..]);
            incoming.entry(key).and_modify(|old|*old=self.algebra.add(old,&value)).or_insert(value);
        }}
        for (key,value) in incoming {
            let offset=self.distribution.local_offset(self.context.rank(),key);
            let previous=if *beta==self.algebra.zero() {self.algebra.zero()} else {self.algebra.multiply(&self.data[offset],beta)};
            self.data[offset]=self.algebra.add(&self.algebra.multiply(&value,alpha),&previous);
        }
    }
    /// Map unique-label operands to a supplied topology, execute the generic
    /// aligned contraction and restore C's distribution. This is explicit-grid
    /// execution; candidate cost selection and BLAS folding are not implied.
    pub fn contract_from_on_grid(&mut self,indices_c:&str,a:&Self,indices_a:&str,b:&Self,indices_b:&str,
        topology:crate::mapping::Topology,alpha:A::Element,beta:A::Element)->Result<(),crate::map_tensor::Rejected>
    where A:Clone {
        assert!(std::ptr::eq(self.context,a.context)&&std::ptr::eq(self.context,b.context));assert_eq!(topology.size(),self.context.size());
        let plan=crate::planning::GridPlan::prepare([&a.distribution,&b.distribution,&self.distribution],
            [indices_a,indices_b,indices_c],topology)?;
        self.contract_with_plan(indices_c,a,indices_a,b,indices_b,&plan,alpha,beta);
        Ok(())
    }
    /// Reuse a context-local explicit-grid mapping plan. Collective when executed;
    /// cache hits avoid remapping search, not the necessary tensor redistribution.
    pub fn contract_cached(&mut self,indices_c:&str,a:&Self,indices_a:&str,b:&Self,indices_b:&str,
        topology:crate::mapping::Topology,cache:&mut crate::planning::PlanCache<'_,'_>,
        alpha:A::Element,beta:A::Element)->Result<(),crate::map_tensor::Rejected> where A:Clone {
        assert!(std::ptr::eq(self.context,cache.context()));
        let plan=cache.prepare([&a.distribution,&b.distribution,&self.distribution],[indices_a,indices_b,indices_c],topology)?;
        self.contract_with_plan(indices_c,a,indices_a,b,indices_b,plan,alpha,beta);
        Ok(())
    }
    /// Execute a prepared aligned plan against matching shapes/index maps/current
    /// distributions. Plan reuse never reuses old tensor values or scalar factors.
    pub fn contract_with_plan(&mut self,indices_c:&str,a:&Self,indices_a:&str,b:&Self,indices_b:&str,
        plan:&crate::planning::GridPlan,alpha:A::Element,beta:A::Element) where A:Clone {
        assert!(std::ptr::eq(self.context,a.context)&&std::ptr::eq(self.context,b.context));
        assert_eq!(plan.signature().topology().size(),self.context.size());
        assert!(plan.matches([&a.distribution,&b.distribution,&self.distribution],[indices_a,indices_b,indices_c]));
        let mut aa=Self {context:a.context,algebra:a.algebra.clone(),distribution:a.distribution.clone(),data:a.data.clone()};
        let mut bb=Self {context:b.context,algebra:b.algebra.clone(),distribution:b.distribution.clone(),data:b.data.clone()};
        let mut cc=Self {context:self.context,algebra:self.algebra.clone(),distribution:self.distribution.clone(),data:self.data.clone()};
        let mapped=plan.mapped_distributions();
        aa.redistribute(mapped[0].clone());bb.redistribute(mapped[1].clone());cc.redistribute(mapped[2].clone());
        cc.contract_from_aligned(indices_c,&aa,indices_a,&bb,indices_b,alpha,beta);
        cc.redistribute(self.distribution.clone());self.data=cc.data;
    }
    /// Collective unique-label contraction on already aligned distributions.
    /// Shared labels must have identical maps; mismatched physical labels need
    /// the 2D/planning layer instead. Native user reductions support custom rings.
    pub fn contract_from_aligned(&mut self,indices_c:&str,a:&Self,indices_a:&str,b:&Self,indices_b:&str,
        alpha:A::Element,beta:A::Element) {
        use crate::mapping::Mapping;
        assert!(std::ptr::eq(self.context,a.context)&&std::ptr::eq(self.context,b.context));
        let operands=[(indices_a,&a.distribution),(indices_b,&b.distribution),(indices_c,&self.distribution)];
        for &(indices,d) in &operands {
            assert_eq!(d.topology,self.distribution.topology);assert!(indices.is_ascii());assert_eq!(indices.len(),d.shape.len());
            for (i,label) in indices.bytes().enumerate() {assert!(!indices.as_bytes()[..i].contains(&label),"aligned contraction requires unique labels");}
        }
        for i in 0..3 {for j in 0..i {
            for (di,label) in operands[i].0.bytes().enumerate() {
                if let Some(dj)=operands[j].0.bytes().position(|l|l==label) {
                    assert_eq!(operands[i].1.shape[di],operands[j].1.shape[dj]);
                    assert_eq!(operands[i].1.mappings[di],operands[j].1.mappings[dj]);
                }
            }
        }}
        fn mark(map:&Mapping,label:u8,axes:&mut [Option<u8>]) {
            match map {
                Mapping::Unmapped=>{},
                Mapping::Physical {axis,child,..}=>{axes[*axis]=Some(label);mark(child,label,axes);},
                Mapping::Virtual {child,..}=>mark(child,label,axes),
            }
        }
        let topology=&self.distribution.topology;
        let mut axes=vec![vec![None;topology.dimensions.len()];3];
        for i in 0..3 {for (map,label) in operands[i].1.mappings.iter().zip(operands[i].0.bytes()) {mark(map,label,&mut axes[i]);}}
        let mut comms:[Vec<Context<'_>>;3]=std::array::from_fn(|_|Vec::new());
        for axis in 0..topology.dimensions.len() {
            let labels:Vec<_>=(0..3).filter_map(|i|axes[i][axis]).collect();
            if labels.is_empty() {continue;}
            assert!(labels.iter().all(|&label|label==labels[0]),"mismatched physical labels require 2D communication");
            for i in 0..3 {if axes[i][axis].is_none() {comms[i].push(topology.fiber(self.context,axis));}}
        }
        let phases:Vec<Vec<_>>=operands.iter().map(|(_,d)|d.mappings.iter().map(|m|m.phase()/m.physical_phase()).collect()).collect();
        let shapes:Vec<_>=operands.iter().map(|(_,d)|d.block_shape()).collect();
        let mut adata=a.data.clone();let mut bdata=b.data.clone();
        crate::contraction::replicated(&self.algebra,&comms[0].iter().collect::<Vec<_>>(),&comms[1].iter().collect::<Vec<_>>(),&comms[2].iter().collect::<Vec<_>>(),
            &shapes[0],&phases[0],indices_a,&mut adata,&shapes[1],&phases[1],indices_b,&mut bdata,
            &shapes[2],&phases[2],indices_c,&mut self.data,&alpha,&beta,false);
        for group in comms {for comm in group {comm.close();}}
        // ctr_replicate leaves valid output on roots only. Key redistribution
        // reads canonical owners and restores the Tensor's replica invariant.
        self.redistribute(self.distribution.clone());
    }
    /// Align unique-label operands on a requested topology using the upstream
    /// physical-axis assignment primitive, execute, and restore output layout.
    /// Returns a mapping candidate rejection without changing either tensor.
    pub fn sum_from_on_grid(&mut self,indices_b: &str,input: &Self,indices_a: &str,
        topology: crate::mapping::Topology,alpha: A::Element,beta: A::Element) -> Result<(),crate::map_tensor::Rejected>
    where A: Clone {
        use crate::mapping::Mapping;
        assert!(std::ptr::eq(self.context,input.context));assert_eq!(topology.size(),self.context.size());
        let mut labels=Vec::new();let mut shape=Vec::new();
        for (indices,distribution) in [(indices_a,&input.distribution),(indices_b,&self.distribution)] {
            assert!(indices.is_ascii());assert_eq!(indices.len(),distribution.shape.len());
            for (i,label) in indices.bytes().enumerate() {
                assert!(!indices.as_bytes()[..i].contains(&label),"unique labels required by this entry point");
                if let Some(j)=labels.iter().position(|&l|l==label) {assert_eq!(shape[j],distribution.shape[i]);}
                else {labels.push(label);shape.push(distribution.shape[i]);}
            }
        }
        let mut maps=vec![Mapping::Unmapped;labels.len()];
        // A scalar has no mapped dimensions; all physical axes remain replicas.
        if !labels.is_empty() {
            crate::map_tensor::assign(&shape,&topology,&(0..topology.dimensions.len()).collect::<Vec<_>>(),
                &vec![false;labels.len()*labels.len()],&mut vec![false;labels.len()],&mut maps,true)?;
        }
        let mapped=|indices:&str,shape:&[usize]|Distribution::new(shape.to_vec(),topology.clone(),
            indices.bytes().map(|label|maps[labels.iter().position(|&l|l==label).unwrap()].clone()).collect());
        let mut a=Self {context:input.context,algebra:input.algebra.clone(),distribution:input.distribution.clone(),data:input.data.clone()};
        let mut b=Self {context:self.context,algebra:self.algebra.clone(),distribution:self.distribution.clone(),data:self.data.clone()};
        a.redistribute(mapped(indices_a,&a.distribution.shape));
        b.redistribute(mapped(indices_b,&b.distribution.shape));
        b.sum_from_aligned(indices_b,&a,indices_a,alpha,beta);
        b.redistribute(self.distribution.clone());self.data=b.data;
        Ok(())
    }
    /// Collective sum for unique labels with explicitly aligned distributions.
    /// Shared labels have identical maps; each topology axis maps to the same
    /// label in both operands, or to only one operand. Automatic remapping is
    /// deliberately not hidden in this entry point.
    pub fn sum_from_aligned(&mut self, indices_b: &str, input: &Self, indices_a: &str, alpha: A::Element, beta: A::Element) {
        use crate::mapping::Mapping;
        assert!(std::ptr::eq(self.context,input.context));
        assert_eq!(self.distribution.topology,input.distribution.topology);
        for (indices,distribution) in [(indices_a,&input.distribution),(indices_b,&self.distribution)] {
            assert!(indices.is_ascii());
            assert_eq!(indices.len(),distribution.shape.len());
            for (i,label) in indices.bytes().enumerate() { assert!(!indices.as_bytes()[..i].contains(&label),"aligned entry point requires unique labels"); }
        }
        for (ia,label) in indices_a.bytes().enumerate() {
            if let Some(ib) = indices_b.bytes().position(|x|x == label) {
                assert_eq!(input.distribution.shape[ia],self.distribution.shape[ib]);
                assert_eq!(input.distribution.mappings[ia],self.distribution.mappings[ib]);
            }
        }
        fn map_labels(map: &Mapping, label: u8, axes: &mut [Option<u8>]) {
            match map {
                Mapping::Unmapped => {},
                Mapping::Physical {axis,child,..} => { axes[*axis] = Some(label); map_labels(child,label,axes); }
                Mapping::Virtual {child,..} => map_labels(child,label,axes),
            }
        }
        let topology = &self.distribution.topology;
        let mut a_axes = vec![None;topology.dimensions.len()];
        let mut b_axes = a_axes.clone();
        for (map,label) in input.distribution.mappings.iter().zip(indices_a.bytes()) { map_labels(map,label,&mut a_axes); }
        for (map,label) in self.distribution.mappings.iter().zip(indices_b.bytes()) { map_labels(map,label,&mut b_axes); }
        for (a,b) in a_axes.iter().zip(&b_axes) { if a.is_some() && b.is_some() { assert_eq!(a,b); } }
        let mut input_comms = Vec::new();
        let mut output_comms = Vec::new();
        for axis in 0..a_axes.len() {
            match (a_axes[axis],b_axes[axis]) {
                (None,Some(_)) => input_comms.push(topology.fiber(self.context,axis)),
                (Some(_),None) => output_comms.push(topology.fiber(self.context,axis)),
                _ => {},
            }
        }
        let virtual_a: Vec<_> = input.distribution.mappings.iter().map(|m|m.phase()/m.physical_phase()).collect();
        let virtual_b: Vec<_> = self.distribution.mappings.iter().map(|m|m.phase()/m.physical_phase()).collect();
        let mut a = input.data.clone();
        crate::summation::replicated(&self.algebra,&input_comms.iter().collect::<Vec<_>>(),&output_comms.iter().collect::<Vec<_>>(),
            &input.distribution.block_shape(),&virtual_a,indices_a,&mut a,
            &self.distribution.block_shape(),&virtual_b,indices_b,&mut self.data,&alpha,&beta,false);
        for (offset,value) in self.data.iter_mut().enumerate() {
            if self.distribution.global_key(self.context.rank(),offset).is_none() { *value = self.algebra.zero(); }
        }
        for comm in input_comms { comm.close(); }
        for comm in output_comms { comm.close(); }
    }
}

pub struct Tensor<'context, 'runtime, A: Monoid> {
    context: &'context Context<'runtime>,
    algebra: A,
    distribution: Distribution,
    data: Vec<A::Element>,
}

impl<'c, 'r, A: Monoid> Tensor<'c, 'r, A> {
    pub fn new(context: &'c Context<'r>, distribution: Distribution, algebra: A) -> Self {
        assert_eq!(context.size(), distribution.topology.size());
        let data = vec![algebra.zero(); distribution.local_len()];
        Self { context, algebra, distribution, data }
    }
    pub fn context(&self) -> &'c Context<'r> { self.context }
    pub fn algebra(&self) -> &A { &self.algebra }
    pub fn distribution(&self) -> &Distribution { &self.distribution }
    pub fn local_storage(&self) -> &[A::Element] { &self.data }
    pub fn local_pairs(&self) -> Vec<(usize, A::Element)> {
        self.data.iter().enumerate().filter_map(|(offset, value)| {
            self.distribution.global_key(self.context.rank(), offset).map(|key| (key, value.clone()))
        }).collect()
    }
    pub fn transform(&mut self, mut function: impl FnMut(usize, &mut A::Element)) {
        for (offset, value) in self.data.iter_mut().enumerate() {
            if let Some(key) = self.distribution.global_key(self.context.rank(), offset) { function(key, value); }
        }
    }
    /// Apply only where equal index labels have equal coordinates. This is the
    /// diagonal selection used by upstream sequential scaling/endomorphisms.
    pub fn transform_indexed(&mut self, labels: &str, mut function: impl FnMut(&mut A::Element)) {
        assert!(labels.is_ascii());
        let labels = labels.as_bytes();
        assert_eq!(labels.len(), self.distribution.shape.len());
        for i in 0..labels.len() {
            for j in 0..i {
                if labels[i] == labels[j] { assert_eq!(self.distribution.shape[i], self.distribution.shape[j]); }
            }
        }
        for (offset, value) in self.data.iter_mut().enumerate() {
            if let Some(key) = self.distribution.global_key(self.context.rank(), offset) {
                let coordinates = self.distribution.decode_key(key);
                if (0..labels.len()).all(|i| (0..i).all(|j| labels[i] != labels[j] || coordinates[i] == coordinates[j])) {
                    function(value);
                }
            }
        }
    }
}

impl<'c, 'r, A: Monoid + Clone> Tensor<'c, 'r, A> where A::Element: Wire {
    /// Extract one coordinate per repeated label. Only canonical local entries
    /// are sent; no global tensor or diagonal is gathered on a single process.
    pub fn extract_diagonal(&self,labels:&str)->(Self,String) {
        let projection=crate::diagonal::Projection::new(&self.distribution.shape,labels);
        if !projection.repeated() {return (Self {context:self.context,algebra:self.algebra.clone(),distribution:self.distribution.clone(),data:self.data.clone()},projection.labels);}
        let mut result=Self::new(self.context,Distribution::cyclic(projection.shape.clone(),self.context.size()),self.algebra.clone());
        let mut pairs=Vec::new();
        for (key,value) in self.local_pairs() {
            if self.distribution.owner(key)!=self.context.rank() {continue;}
            if let Some(coordinates)=projection.project(&self.distribution.decode_key(key)) {pairs.push((result.distribution.encode_key(&coordinates),value));}
        }
        result.write_add(&pairs);(result,projection.labels)
    }
    /// Replace selected diagonal entries, preserving all off-diagonal data.
    pub fn replace_diagonal(&mut self,labels:&str,input:&Self) {
        assert!(std::ptr::eq(self.context,input.context));
        let projection=crate::diagonal::Projection::new(&self.distribution.shape,labels);
        assert_eq!(projection.shape,input.distribution.shape);
        let mut pairs=Vec::new();
        for (key,value) in input.local_pairs() {
            if input.distribution.owner(key)!=self.context.rank() {continue;}
            let coordinates=projection.expand(&input.distribution.decode_key(key));pairs.push((self.distribution.encode_key(&coordinates),value));
        }
        let zero=self.algebra.zero();self.transform_indexed(labels,|value|*value=zero.clone());self.write_add(&pairs);
    }
    /// Collective dense slice, retaining physical/virtual mappings. Extract the
    /// local sub-block, then shift its owner by offsets modulo physical phases,
    /// as in upstream redistribution/slice.cxx. No global tensor is gathered.
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Self {
        assert_eq!(ranges.len(), self.distribution.shape.len());
        for (range, &n) in ranges.iter().zip(&self.distribution.shape) {
            assert!(range.start <= range.end && range.end <= n);
        }
        let offsets: Vec<_> = ranges.iter().map(|r|r.start).collect();
        let mut distribution = self.distribution.clone();
        distribution.shape = ranges.iter().map(|r|r.end-r.start).collect();
        let destination = self.distribution.shifted_rank(self.context.rank(), &offsets, false);
        let source = self.distribution.shifted_rank(self.context.rank(), &offsets, true);
        let mut result = Self::new(self.context, distribution, self.algebra.clone());
        for (key, value) in self.local_pairs() {
            let coordinates = self.distribution.decode_key(key);
            if coordinates.iter().zip(ranges).all(|(&c,r)|r.contains(&c)) {
                let sliced: Vec<_> = coordinates.iter().zip(&offsets).map(|(&c,&o)|c-o).collect();
                let new_key = result.distribution.encode_key(&sliced);
                let offset = result.distribution.local_offset(destination, new_key);
                result.data[offset] = value;
            }
        }
        if destination != self.context.rank() {
            let mut bytes = Vec::with_capacity(result.data.len()*A::Element::WIDTH);
            for value in &result.data { value.encode(&mut bytes); }
            let mut received = vec![0u8; bytes.len()];
            // For a globally empty slice no rank needs a data exchange.
            if !bytes.is_empty() { self.context.inner.send_receive(&bytes,destination,source,&mut received); }
            for (value, bytes) in result.data.iter_mut().zip(received.chunks_exact(A::Element::WIDTH)) {
                *value = A::Element::decode(bytes);
            }
        }
        result
    }
    /// Reorder tensor axes and their mappings together. Physical ownership is
    /// unchanged, so this transpose is local and has no implicit MPI collective.
    pub fn permute_axes(&self, axes: &[usize]) -> Self {
        let order = self.distribution.shape.len();
        assert_eq!(axes.len(),order);
        let mut seen = vec![false;order];
        for &axis in axes { assert!(axis < order && !seen[axis]); seen[axis] = true; }
        let distribution = Distribution::new(axes.iter().map(|&i|self.distribution.shape[i]).collect(),
            self.distribution.topology.clone(), axes.iter().map(|&i|self.distribution.mappings[i].clone()).collect());
        let mut result = Self::new(self.context,distribution,self.algebra.clone());
        for (key,value) in self.local_pairs() {
            let coordinates = self.distribution.decode_key(key);
            let permuted: Vec<_> = axes.iter().map(|&i|coordinates[i]).collect();
            let key = result.distribution.encode_key(&permuted);
            let offset = result.distribution.local_offset(self.context.rank(),key);
            result.data[offset] = value;
        }
        result
    }
}

impl<A: Semiring> Tensor<'_, '_, A> {
    pub fn scale(&mut self, alpha: &A::Element) {
        for (offset, value) in self.data.iter_mut().enumerate() {
            if self.distribution.global_key(self.context.rank(), offset).is_some() {
                *value = self.algebra.multiply(alpha, value);
            }
        }
    }
}

impl<A: Monoid> Tensor<'_, '_, A> where A::Element: Wire {
    /// Collective random access. Every rank participates, even with no requests.
    pub fn read(&self, keys: &[usize]) -> Vec<A::Element> {
        let mut requests = vec![Vec::new(); self.context.size()];
        for (position, &key) in keys.iter().enumerate() {
            let bucket = &mut requests[self.distribution.owner(key)];
            (position as u64).encode(bucket); (key as u64).encode(bucket);
        }
        let received = self.context.inner.exchange(&requests);
        let mut replies = vec![Vec::new(); self.context.size()];
        for (rank, bytes) in received.iter().enumerate() {
            for request in bytes.chunks_exact(16) {
                let position = u64::decode(&request[..8]);
                let key = u64::decode(&request[8..]) as usize;
                let offset = self.distribution.local_offset(self.context.rank(), key);
                position.encode(&mut replies[rank]); self.data[offset].encode(&mut replies[rank]);
            }
        }
        let received = self.context.inner.exchange(&replies);
        let mut result = vec![self.algebra.zero(); keys.len()];
        for bytes in received {
            for reply in bytes.chunks_exact(8 + A::Element::WIDTH) {
                let position = u64::decode(&reply[..8]) as usize;
                result[position] = A::Element::decode(&reply[8..]);
            }
        }
        result
    }
    /// Collective additive writes. Duplicate keys are reduced in rank order.
    pub fn write_add(&mut self, pairs: &[(usize, A::Element)]) {
        let mut buckets = vec![Vec::new(); self.context.size()];
        for (key, value) in pairs {
            for (rank, bucket) in buckets.iter_mut().enumerate() {
                if self.distribution.owns(rank, *key) {
                    (*key as u64).encode(bucket); value.encode(bucket);
                }
            }
        }
        for bytes in self.context.inner.exchange(&buckets) {
            for pair in bytes.chunks_exact(8 + A::Element::WIDTH) {
                let key = u64::decode(&pair[..8]) as usize;
                let offset = self.distribution.local_offset(self.context.rank(), key);
                self.data[offset] = self.algebra.add(&self.data[offset], &A::Element::decode(&pair[8..]));
            }
        }
    }
    /// Collective distribution switch, sending each unique entry once per new
    /// replica. Only local data and communication buckets are allocated.
    pub fn redistribute(&mut self, distribution: Distribution) {
        assert_eq!(distribution.shape, self.distribution.shape);
        assert_eq!(distribution.topology.size(), self.context.size());
        let mut buckets = vec![Vec::new(); self.context.size()];
        for (key, value) in self.local_pairs() {
            if self.distribution.owner(key) != self.context.rank() { continue; }
            for (rank, bucket) in buckets.iter_mut().enumerate() {
                if distribution.owns(rank, key) {
                    (key as u64).encode(bucket); value.encode(bucket);
                }
            }
        }
        let received = self.context.inner.exchange(&buckets);
        let mut data = vec![self.algebra.zero(); distribution.local_len()];
        for bytes in received {
            for pair in bytes.chunks_exact(8 + A::Element::WIDTH) {
                let key = u64::decode(&pair[..8]) as usize;
                data[distribution.local_offset(self.context.rank(), key)] = A::Element::decode(&pair[8..]);
            }
        }
        self.distribution = distribution; self.data = data;
    }
    pub fn reduce(&self) -> A::Element {
        let mut value = self.algebra.zero();
        for (key, entry) in self.local_pairs() {
            if self.distribution.owner(key) == self.context.rank() { value = self.algebra.add(&value, &entry); }
        }
        self.context.all_reduce(&self.algebra, &value)
    }
}
