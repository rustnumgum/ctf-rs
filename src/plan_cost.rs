// Adapted from cc4s contraction/{ctr_tsr,ctr_comm,ctr_2d_general}.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Recursive dense contraction cost formulas. Working bytes follow the source
//! mem_rec model, not measured Rust allocator usage or total tensor residency.
use crate::cost::{Models,Communication};

#[derive(Clone,Debug)]
pub struct Collective {
    pub ranks:usize,
    pub nodes:usize,
    pub bytes:usize,
}
#[derive(Clone,Debug)]
pub enum Tree {
    Local { custom:bool,folded:bool,operand_bytes:[usize;3],flops:f64 },
    Virtual { phases:Vec<usize>,orders:[usize;3],child:Box<Tree> },
    Replicated { inputs:[Vec<Collective>;2],output:Vec<Collective>,custom_reduce:bool,child:Box<Tree> },
    Panels { steps:usize,panel_bytes:[usize;3],movement:[Option<Collective>;3],custom_reduce:bool,child:Box<Tree> },
}
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct Estimate {pub seconds:f64,pub working_bytes:usize,pub internode_volume:f64}
impl Tree {
    pub fn estimate(&self,models:&Models,layers:usize)->Estimate {
        assert!(layers>0);
        match self {
            Self::Local{custom,folded,operand_bytes,flops}=>Estimate {
                seconds:models.local_contraction(*custom,*folded,operand_bytes.iter().sum::<usize>() as f64,*flops),
                working_bytes:0,internode_volume:0.,
            },
            Self::Virtual{phases,orders,child}=>{
                assert!(phases.iter().all(|&p|p>0));let count=phases.iter().product::<usize>() as f64;
                let mut estimate=child.estimate(models,layers);estimate.seconds*=count;
                // ctr_virt does not override the base volume estimator, which
                // returns zero in this revision (unlike its time recursion).
                estimate.internode_volume=0.;
                // Upstream VIRT_NTD=1 and sizeof(int)=4. This models source
                // bookkeeping; it is not a Rust Vec capacity accounting claim.
                estimate.working_bytes+=(orders.iter().sum::<usize>()+4*phases.len())*4;estimate
            },
            Self::Replicated{inputs,output,custom_reduce,child}=>{
                let mut estimate=child.estimate(models,layers);
                for comm in inputs.iter().flatten() {
                    estimate.seconds+=models.communication(Communication::Broadcast,comm.ranks,comm.bytes);
                    estimate.internode_volume+=(comm.bytes*comm.nodes) as f64;
                }
                for comm in output {
                    estimate.seconds+=models.communication(Communication::Reduce{custom:*custom_reduce},comm.ranks,comm.bytes);
                    estimate.internode_volume+=(comm.bytes*comm.nodes) as f64;
                }
                estimate
            },
            Self::Panels{steps,panel_bytes,movement,custom_reduce,child}=>{
                assert!(*steps>0);let calls=*steps as f64/layers.min(*steps) as f64;
                let mut estimate=child.estimate(models,1);let mut auxiliary=0;
                for (operand,comm) in movement.iter().enumerate() {
                    if let Some(comm)=comm {
                        assert_eq!(comm.bytes,panel_bytes[operand]);
                        let op=if operand==2 {Communication::Reduce{custom:*custom_reduce}} else {Communication::Broadcast};
                        estimate.seconds+=models.communication(op,comm.ranks,comm.bytes);
                        estimate.internode_volume+=(comm.bytes*comm.nodes) as f64;
                        auxiliary=auxiliary.max(comm.bytes);
                    }
                }
                estimate.seconds*=calls;estimate.internode_volume*=calls;
                estimate.working_bytes+=panel_bytes.iter().sum::<usize>()+auxiliary;estimate
            },
        }
    }
}
