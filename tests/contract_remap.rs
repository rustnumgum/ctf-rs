use ctf::{algebra::Arithmetic,context::Runtime,mapping::{Distribution,Topology},tensor::Tensor};
fn main() {
    let runtime=Runtime::initialize();let world=runtime.world();let np=world.size();
    let topo=Topology::new(if np==4 {vec![2,2]} else {vec![np]});
    let mut a=Tensor::new(&world,Distribution::cyclic(vec![3,2,5],np),Arithmetic::<i64>::new());
    let mut b=Tensor::new(&world,Distribution::cyclic(vec![5,2,2],np),Arithmetic::<i64>::new());
    let mut c=Tensor::new(&world,Distribution::cyclic(vec![3,2],np),Arithmetic::<i64>::new());
    a.transform(|key,v|*v=(key%3+1) as i64);b.transform(|key,v|*v=(key/10+2) as i64);c.transform(|_,v|*v=10);
    let distribution=c.distribution().clone();
    c.contract_from_on_grid("ij",&a,"iab",&b,"baj",topo,2,3).unwrap();
    assert_eq!(*c.distribution(),distribution);
    for (key,v) in c.local_pairs() {assert_eq!(v,30+20*(key%3+1) as i64*(key/3+2) as i64);}
    drop(c);drop(b);drop(a);
    if world.rank()==0 {println!("DIGIT / PASS contract_remap: high-order two-index reduction, mismatched input layouts, restored output; ranks={np}");}
    world.close();runtime.finalize();
}
