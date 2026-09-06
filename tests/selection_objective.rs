use ctf::{algebra::Arithmetic,context::{Context,Runtime},mapping::{Distribution,Topology},planning::GridPlan,
    selector::{Candidate,Objective,Selector},tensor::Tensor};
fn exercise(context:&Context<'_>) {
    let np=context.size();let make=|shape|Tensor::new(context,Distribution::cyclic(shape,np),Arithmetic::<i64>::new());
    let mut a=make(vec![3,2]);let mut b=make(vec![2,2]);let mut c=make(vec![3,2]);
    a.transform(|_,v|*v=3);b.transform(|_,v|*v=4);
    let plan=GridPlan::prepare([a.distribution(),b.distribution(),c.distribution()],["ik","kj","ij"],Topology::new(vec![np])).unwrap();
    let signature=plan.signature().clone();let mut selector=Selector::new(context);
    // Rank 0 owns the fast large candidate; the last rank owns a slower small one.
    if context.rank()==0 {selector.store(&signature,Candidate{plan:plan.clone(),topology_id:10,exhaustive:false,seconds:1.,memory_bytes:80});}
    if context.rank()==np-1 {selector.store(&signature,Candidate{plan:plan.clone(),topology_id:20,exhaustive:false,seconds:2.,memory_bytes:20});}
    let objective=Objective{memory_limit:100,weight:0.,baseline_seconds:1.,baseline_memory:80};
    assert!(selector.select_best(objective));assert_eq!(selector.selected().unwrap().topology_id,10);
    assert!(selector.select_best(Objective{weight:2.,..objective}));assert_eq!(selector.selected().unwrap().topology_id,20);
    assert!(selector.select_best(Objective{memory_limit:80,..objective}));assert_eq!(selector.selected().unwrap().topology_id,20);
    assert!(!selector.select_best(Objective{memory_limit:20,..objective}));assert!(selector.selected().is_none());
    assert!(selector.select_best(Objective{weight:1e-8,baseline_seconds:0.,baseline_memory:0,..objective}));
    assert_eq!(selector.selected().unwrap().topology_id,10);
    c.contract_with_plan("ij",&a,"ik",&b,"kj",&selector.selected().unwrap().plan,1,0);
    for (_,value) in c.local_pairs() {assert_eq!(value,24);}
    selector.clear();
    for offset in [0,1] {selector.store(&signature,Candidate{plan:plan.clone(),topology_id:100+context.rank() as i64*2+offset,
        exhaustive:false,seconds:1.,memory_bytes:10});}
    assert!(selector.select_best(objective));assert_eq!(selector.selected().unwrap().topology_id,100);
    selector.clear();selector.store(&signature,Candidate{plan,topology_id:0,exhaustive:true,seconds:0.,memory_bytes:0});
    assert!(!selector.select_best(objective));
}
fn main() {
    let runtime=Runtime::initialize();let world=runtime.world();exercise(&world);
    let child=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();exercise(&child);child.close();
    if world.rank()==0 {println!("DIGIT / PASS selection_objective: time/memory tradeoff, strict budget, weight cutoff, rank/local ties, selected execution and subcontexts; ranks={}",world.size());}
    world.close();runtime.finalize();
}
