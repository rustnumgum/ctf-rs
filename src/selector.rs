// Adapted from cc4s contraction/contraction_selector.h and contraction_plan.h.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Explicit candidate filtering and collective selection. Candidate generation
//! and full-tree cost calculation remain separate responsibilities.
use crate::{context::Context,planning::{GridPlan,Signature},mapping::Distribution};

#[derive(Clone,Debug)]
pub struct Candidate {
    pub plan:GridPlan,
    pub topology_id:i64,
    pub exhaustive:bool,
    pub seconds:f64,
    pub memory_bytes:u64,
}
impl Candidate {
    fn pack(&self)->Vec<u64> {
        let mut words=vec![self.topology_id as u64,u64::from(self.exhaustive),self.seconds.to_bits(),self.memory_bytes];
        words.extend(self.plan.pack());words
    }
    fn unpack(words:&[u64])->Self {
        Self {topology_id:words[0] as i64,exhaustive:words[1]!=0,seconds:f64::from_bits(words[2]),
            memory_bytes:words[3],plan:GridPlan::unpack(&words[4..])}
    }
}

/// Source get_replicates includes virtual copies as well as physical replicas.
pub fn replication_factor(distribution:&Distribution)->usize {
    let physical:usize=distribution.mappings.iter().map(|m|m.physical_phase()).product();
    let virtuals:usize=distribution.mappings.iter().map(|m|m.phase()/m.physical_phase()).product();
    distribution.topology.size()/physical*virtuals
}
#[derive(Clone,Copy)]
pub enum Filter { MaxMemory(u64),MaxTime(f64),NoReplication(usize) }
/// evaluate_mappings time/memory objective. Baseline is required only for a
/// non-negligible weight (source cutoff |weight|>1e-8).
#[derive(Clone,Copy)]
pub struct Objective {pub memory_limit:u64,pub weight:f64,pub baseline_seconds:f64,pub baseline_memory:u64}
impl Objective {
    fn score(self,seconds:f64,memory:u64)->f64 {
        if self.weight.abs()>1e-8 {
            assert!(self.baseline_seconds>0. && self.baseline_memory>0);
            (seconds-self.baseline_seconds)/self.baseline_seconds+
                self.weight*(memory as f64-self.baseline_memory as f64)/self.baseline_memory as f64
        } else {seconds}
    }
}
impl Filter {
    pub fn accepts(self,candidate:&Candidate)->bool {
        match self {
            Self::MaxMemory(limit)=>candidate.memory_bytes<=limit,
            Self::MaxTime(limit)=>candidate.seconds<=limit,
            Self::NoReplication(operand)=>replication_factor(&candidate.plan.mapped_distributions()[operand])==1,
        }
    }
    pub fn upstream(name:&str)->Self {
        match name {
            "max_memory"=>Self::MaxMemory(10*1024*1024),"max_time"=>Self::MaxTime(0.1),
            "no_replicate_A"=>Self::NoReplication(0),"no_replicate_B"=>Self::NoReplication(1),
            "no_replicate_C"=>Self::NoReplication(2),_=>panic!("unknown candidate filter"),
        }
    }
}
pub struct Selector<'context,'runtime> {
    context:&'context Context<'runtime>,
    signature:Option<Signature>,
    candidates:Vec<Candidate>,
    selected:Option<Candidate>,
    scan:bool,
}
impl<'context,'runtime> Selector<'context,'runtime> {
    pub fn new(context:&'context Context<'runtime>)->Self {
        Self {context,signature:None,candidates:Vec::new(),selected:None,scan:false}
    }
    pub fn set_scan(&mut self,scan:bool) {self.scan=scan;}
    pub fn scan(&self)->bool {self.scan}
    pub fn candidates(&self)->&[Candidate] {&self.candidates}
    pub fn selected(&self)->Option<&Candidate> {self.selected.as_ref()}
    /// Signature refers to the original contraction, independently of candidate
    /// target topology. Full structural equality replaces upstream hash-only check.
    pub fn store(&mut self,signature:&Signature,candidate:Candidate) {
        assert_eq!(candidate.plan.signature().topology().size(),self.context.size());
        if self.signature.as_ref()!=Some(signature) {
            self.signature=Some(signature.clone());self.candidates.clear();self.selected=None;
        }
        self.candidates.push(candidate);
    }
    pub fn filter(&mut self,filters:&[Filter]) {
        self.candidates.retain(|candidate|filters.iter().all(|&filter|filter.accepts(candidate)));
    }
    pub fn clear(&mut self) {self.candidates.clear();self.selected=None;}
    pub fn reset(&mut self) {self.clear();self.scan=false;}
    /// Collective normal-search winner selection for already estimated dense
    /// candidates. Source strict memory bound and INT_MAX dense-local count limit
    /// are enforced before ranking. Local and cross-rank ties retain first order.
    /// Exhaustive-search generation/refinement is not performed by this method.
    pub fn select_best(&mut self,objective:Objective)->bool {
        self.scan=false;self.selected=None;
        let mut local=None;let mut score=f64::MAX;
        for candidate in &self.candidates {
            if candidate.exhaustive {continue;}
            if candidate.memory_bytes>=objective.memory_limit {continue;}
            if candidate.plan.mapped_distributions().iter().any(|d|d.local_len()>i32::MAX as usize) {continue;}
            assert!(candidate.seconds>=0.);
            let current=objective.score(candidate.seconds,candidate.memory_bytes);
            if current<score {score=current;local=Some(candidate);}
        }
        let seconds=local.map_or(f64::MAX,|c|c.seconds);
        let memory=local.map_or(-1,|c|c.memory_bytes.try_into().unwrap());
        let (times,memories)=self.context.inner.gather_plan_cost(seconds,memory);
        let mut winner=[-1i32];
        if self.context.rank()==0 {
            let mut best=f64::MAX;
            for rank in 0..self.context.size() {
                if memories[rank]<0 {continue;}
                let value=objective.score(times[rank],memories[rank] as u64);
                if value<best {best=value;winner[0]=rank as i32;}
            }
        }
        self.context.broadcast(0,&mut winner);
        if winner[0]<0 {return false;}
        let root=winner[0] as usize;
        let mut words=if self.context.rank()==root {local.unwrap().pack()} else {Vec::new()};
        let mut size=[words.len() as u64];self.context.broadcast(root,&mut size);
        words.resize(size[0].try_into().unwrap(),0);self.context.broadcast(root,&mut words);
        self.selected=Some(Candidate::unpack(&words));true
    }
    /// Collective: first matching local candidate; allgather availability; lowest
    /// matching rank broadcasts payload size then payload, as in selectCandidate.
    /// Returns false on every rank if the requested candidate is absent globally.
    pub fn select(&mut self,topology_id:i64,exhaustive:bool)->bool {
        self.scan=false;self.selected=None;
        let local=self.candidates.iter().find(|c|c.topology_id==topology_id&&c.exhaustive==exhaustive);
        let found=self.context.inner.all_gather_i32(i32::from(local.is_some()));
        if let Some(root)=found.iter().position(|&flag|flag==1) {
            let mut words=if self.context.rank()==root {local.unwrap().pack()} else {Vec::new()};
            let mut size=[words.len() as u64];self.context.broadcast(root,&mut size);
            words.resize(size[0].try_into().unwrap(),0);self.context.broadcast(root,&mut words);
            self.selected=Some(Candidate::unpack(&words));true
        } else {false}
    }
}
