// Adapted responsibilities from cc4s contraction_signature, contraction_plan,
// and World::ctr_sig_map. Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Context-owned explicit-grid plans. This is the reusable mapping/execution
//! boundary, not the unfinished automatic cost-based candidate search. Cached
//! plans contain no tensor values, MPI handles or borrowed tensor storage.
use std::collections::HashMap;
use crate::{context::Context, mapping::{Distribution,Mapping,Topology}, map_tensor::Rejected};

/// Dense signature: lengths, index maps and original physical/virtual mappings.
/// Alpha, beta and tensor values deliberately do not participate. Full equality
/// is checked rather than the upstream hash-only equality (collisions must not
/// select an incompatible plan). Requested topology is also part of this key.
#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub struct Signature {
    distributions: [Distribution;3],
    indices: [Vec<usize>;3],
    topology: Topology,
}
impl Signature {
    pub fn new(distributions: [&Distribution;3], indices: [&str;3], topology: Topology) -> Self {
        let mut labels = Vec::new();
        let mut lengths = Vec::new();
        let normalized = std::array::from_fn(|operand| {
            assert!(indices[operand].is_ascii());
            assert_eq!(indices[operand].len(),distributions[operand].shape.len());
            indices[operand].bytes().enumerate().map(|(axis,label)| {
                if let Some(i) = labels.iter().position(|&l| l == label) {
                    assert_eq!(lengths[i],distributions[operand].shape[axis]);
                    i
                } else {
                    labels.push(label);
                    lengths.push(distributions[operand].shape[axis]);
                    labels.len()-1
                }
            }).collect()
        });
        Self { distributions:distributions.map(Clone::clone),indices:normalized,topology }
    }
    pub fn distributions(&self) -> &[Distribution;3] { &self.distributions }
    pub fn indices(&self) -> &[Vec<usize>;3] { &self.indices }
    pub fn topology(&self) -> &Topology { &self.topology }
}

/// Fully executable unique-label aligned plan, including the original output
/// distribution to restore. No estimated time/memory is invented for this path.
#[derive(Clone,Debug)]
pub struct GridPlan {
    signature: Signature,
    mapped: [Distribution;3],
}
impl GridPlan {
    pub(crate) fn pack(&self)->Vec<u64> {
        fn map(m:&Mapping,out:&mut Vec<u64>) {
            match m {
                Mapping::Unmapped=>out.push(0),
                Mapping::Physical{axis,processes,child}=>{out.extend([1,*axis as u64,*processes as u64]);map(child,out);},
                Mapping::Virtual{copies,child}=>{out.extend([2,*copies as u64]);map(child,out);},
            }
        }
        fn vector(v:&[usize],out:&mut Vec<u64>) {out.push(v.len() as u64);out.extend(v.iter().map(|&x|x as u64));}
        fn distribution(d:&Distribution,out:&mut Vec<u64>) {
            vector(&d.shape,out);vector(&d.topology.dimensions,out);
            for m in &d.mappings {map(m,out);}
        }
        let mut out=Vec::new();
        for d in &self.signature.distributions {distribution(d,&mut out);}
        for indices in &self.signature.indices {vector(indices,&mut out);}
        vector(&self.signature.topology.dimensions,&mut out);
        for d in &self.mapped {distribution(d,&mut out);}
        out
    }
    pub(crate) fn unpack(words:&[u64])->Self {
        fn value(words:&mut std::slice::Iter<'_,u64>)->usize {(*words.next().unwrap()).try_into().unwrap()}
        fn vector(words:&mut std::slice::Iter<'_,u64>)->Vec<usize> {(0..value(words)).map(|_|value(words)).collect()}
        fn map(words:&mut std::slice::Iter<'_,u64>)->Mapping {
            match value(words) {
                0=>Mapping::Unmapped,
                1=>Mapping::Physical{axis:value(words),processes:value(words),child:Box::new(map(words))},
                2=>Mapping::Virtual{copies:value(words),child:Box::new(map(words))},
                _=>panic!("invalid mapping tag"),
            }
        }
        fn distribution(words:&mut std::slice::Iter<'_,u64>)->Distribution {
            let shape=vector(words);let topology=Topology::new(vector(words));
            let maps=(0..shape.len()).map(|_|map(words)).collect();Distribution::new(shape,topology,maps)
        }
        let mut words=words.iter();
        let distributions=std::array::from_fn(|_|distribution(&mut words));
        let indices=std::array::from_fn(|_|vector(&mut words));
        let topology=Topology::new(vector(&mut words));
        let mapped=std::array::from_fn(|_|distribution(&mut words));assert!(words.next().is_none());
        Self {signature:Signature{distributions,indices,topology},mapped}
    }
    pub fn prepare(distributions: [&Distribution;3], indices: [&str;3], topology: Topology) -> Result<Self,Rejected> {
        let signature = Signature::new(distributions,indices,topology);
        Self::from_signature(signature)
    }
    fn from_signature(signature: Signature) -> Result<Self,Rejected> {
        let mut shape = Vec::new();
        for operand in 0..3 {
            for (axis,&label) in signature.indices[operand].iter().enumerate() {
                assert!(!signature.indices[operand][..axis].contains(&label),"grid plans require unique labels");
                if label == shape.len() { shape.push(signature.distributions[operand].shape[axis]); }
            }
        }
        let mut maps = vec![Mapping::Unmapped;shape.len()];
        if !shape.is_empty() {
            crate::map_tensor::assign(&shape,&signature.topology,
                &(0..signature.topology.dimensions.len()).collect::<Vec<_>>(),
                &vec![false;shape.len()*shape.len()],&mut vec![false;shape.len()],&mut maps,true)?;
        }
        let mapped = std::array::from_fn(|operand| Distribution::new(
            signature.distributions[operand].shape.clone(),signature.topology.clone(),
            signature.indices[operand].iter().map(|&label| maps[label].clone()).collect()));
        Ok(Self { signature,mapped })
    }
    pub fn signature(&self) -> &Signature { &self.signature }
    pub fn mapped_distributions(&self) -> &[Distribution;3] { &self.mapped }
    pub(crate) fn matches(&self, distributions: [&Distribution;3], indices: [&str;3]) -> bool {
        self.signature == Signature::new(distributions,indices,self.signature.topology.clone())
    }
}

#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct CacheStats { pub hits: usize, pub misses: usize }

/// Explicitly supplied cache scoped to exactly one communication context. Cache
/// lookup/clear/destruction is local; execution remains an explicit collective.
pub struct PlanCache<'context,'runtime> {
    context: &'context Context<'runtime>,
    plans: HashMap<Signature,GridPlan>,
    stats: CacheStats,
}
impl<'context,'runtime> PlanCache<'context,'runtime> {
    pub fn new(context: &'context Context<'runtime>) -> Self {
        Self { context,plans:HashMap::new(),stats:CacheStats::default() }
    }
    pub fn context(&self) -> &'context Context<'runtime> { self.context }
    pub fn stats(&self) -> CacheStats { self.stats }
    pub fn len(&self) -> usize { self.plans.len() }
    pub fn is_empty(&self) -> bool { self.plans.is_empty() }
    pub fn clear(&mut self) { self.plans.clear(); }
    pub fn prepare(&mut self, distributions: [&Distribution;3], indices: [&str;3], topology: Topology) -> Result<&GridPlan,Rejected> {
        assert_eq!(topology.size(),self.context.size());
        let signature = Signature::new(distributions,indices,topology);
        match self.plans.entry(signature) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                self.stats.hits += 1;
                Ok(entry.into_mut())
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                self.stats.misses += 1;
                let plan = GridPlan::from_signature(entry.key().clone())?;
                Ok(entry.insert(plan))
            }
        }
    }
}
