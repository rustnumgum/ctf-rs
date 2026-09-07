use ctf::{context::{Context,Runtime},cost::Models,mapping::Distribution,mapped_cost,
    normal_search,mapping_variants,topology_candidates};

#[derive(Clone,Copy,Debug,PartialEq)]
struct Record{id:usize,exhaustive:bool,seconds:f64,memory:usize}
fn models(scale:f64)->Models{
    let mut m=Models::upstream(1);
    for name in ["seq_tsr_ctr_mdl_ref","bcast_mdl","red_mdl","dgtog_res_mdl"]{
        m.get_mut(name).set_coefficients(&[0.,0.,scale]);
    }
    m.get_mut("blres_mdl").set_coefficients(&[0.,scale]);m
}
// Serial reference over the rank-partitioned candidate stream. Broadcast every
// record only in this test; production selection must exchange just local winners.
fn records(c:&Context<'_>,old:[&Distribution;3],m:&Models,limit:usize)->Vec<Record>{
    let catalog=topology_candidates::all_shapes(c.size());let shapes=old.map(|d|d.shape.as_slice());
    let indices=["ik","kj","ij"];let mut local=Vec::new();
    let mut add=|id,exhaustive,mapped:[Distribution;3]|{
        if mapped.iter().map(|d|d.local_len()*8).sum::<usize>()>=limit{return;}
        if mapped.iter().any(|d|d.local_len()>i32::MAX as usize){return;}
        let nodes=vec![1;mapped[0].topology.dimensions.len()];
        let e=mapped_cost::estimate_dense_unfolded(old,mapped.each_ref(),indices,m,8,&nodes,false,false);
        if e.memory_bytes<limit{local.push(Record{id,exhaustive,seconds:e.seconds,memory:e.memory_bytes});}
    };
    normal_search::visit_local(c,shapes,indices,old.map(Some),&catalog,|x|add(x.source_id,false,x.distributions)).unwrap();
    mapping_variants::visit_local_exhaustive(c,shapes,indices,&catalog,|x|add(x.global_id,true,x.variant.distributions)).unwrap();
    let mut all=Vec::new();
    for root in 0..c.size(){
        let mut count=[local.len()as u64];c.broadcast(root,&mut count);
        for i in 0..count[0]as usize{
            let mut words=if c.rank()==root{let r=local[i];[r.id as u64,r.exhaustive as u64,r.seconds.to_bits(),r.memory as u64]}else{[0;4]};
            c.broadcast(root,&mut words);
            all.push(Record{id:words[0]as usize,exhaustive:words[1]!=0,seconds:f64::from_bits(words[2]),memory:words[3]as usize});
        }
    }all
}
fn best(records:&[Record],exhaustive:bool,weight:f64,baseline:Option<Record>)->Option<Record>{
    let mut chosen=None;let mut score=if exhaustive{
        if weight.abs()>1e-8{0.}else{baseline.unwrap().seconds}
    }else{f64::MAX};
    for &r in records.iter().filter(|r|r.exhaustive==exhaustive){
        let value=if weight.abs()>1e-8{let b=baseline.unwrap();
            (r.seconds-b.seconds)/b.seconds+weight*(r.memory as f64-b.memory as f64)/b.memory as f64
        }else{r.seconds};
        if value<score{score=value;chosen=Some(r);}
    }chosen
}
fn expected(records:&[Record],weight:f64,refine:bool)->Option<Record>{
    let initial=best(records,false,0.,None)?;
    let normal=if weight.abs()>1e-8{best(records,false,weight,Some(initial)).unwrap()}else{initial};
    if refine&&normal.seconds>=0.01{
        if let Some(exhaustive)=best(records,true,weight,Some(normal)){
            if exhaustive.seconds<normal.seconds{return Some(exhaustive);}
        }
}Some(normal)
}

fn run(c:&Context<'_>){
    use ctf::dense_search::{self,Kind,Options,TopologyFacts};
    let old=[Distribution::cyclic(vec![3,4],c.size()),Distribution::cyclic(vec![4,5],c.size()),
        Distribution::cyclic(vec![3,5],c.size())];
    let catalog:Vec<_>=topology_candidates::all_shapes(c.size()).into_iter()
        .map(|topology|TopologyFacts{nodes_per_axis:vec![1;topology.dimensions.len()],topology}).collect();
    let limit=1_000_000;
    for scale in [1.,0.000001]{
        let m=models(scale);let reference=records(c,old.each_ref(),&m,limit);
        for(weight,refine)in[(0.,false),(0.,true),(2.,true)]{
            let result=dense_search::search_dense_unfolded(c,old.each_ref(),[&[1];3],["ik","kj","ij"],
                &catalog,&m,8,false,false,Options{memory_limit:limit as u64,weight,allow_exhaustive:refine}).unwrap().unwrap();
            let actual=Record{id:result.source_id,exhaustive:matches!(result.kind,Kind::Exhaustive),
                seconds:result.seconds,memory:result.memory_bytes as usize};
            assert_eq!(Some(actual),expected(&reference,weight,refine));
            if scale<1.{assert!(result.seconds<0.01);assert!(matches!(result.kind,Kind::Normal));}
            assert!(ctf::mapping_preflight::check(result.distributions.each_ref(),["ik","kj","ij"]));
        }
    }
    let no_candidate=dense_search::search_dense_unfolded(c,old.each_ref(),[&[1];3],["ik","kj","ij"],
        &catalog,&models(1.),8,false,false,Options{memory_limit:1,weight:0.,allow_exhaustive:true}).unwrap();
    assert!(no_candidate.is_none());
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS dense_search: source rank-order winners, weighted two-pass, exhaustive refinement, .01 cutoff, strict memory; world+parity; exact");}
    world.close();runtime.finalize();}
