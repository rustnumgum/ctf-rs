use ctf::{algebra::Arithmetic,context::Runtime,mapping::{Distribution,Mapping,Topology},tensor::Tensor};
fn main() {
    let runtime=Runtime::initialize();let world=runtime.world();let np=world.size();
    let topology=Topology::new(vec![np]);let mut kmap=Mapping::Unmapped;kmap.augment_physical(&topology,0);kmap.augment_virtual(2*np);
    let mut a=Tensor::new(&world,Distribution::new(vec![3,5],topology.clone(),vec![Mapping::Unmapped,kmap.clone()]),Arithmetic::<i64>::new());
    let mut b=Tensor::new(&world,Distribution::new(vec![5,2],topology.clone(),vec![kmap,Mapping::Unmapped]),Arithmetic::<i64>::new());
    let mut c=Tensor::new(&world,Distribution::new(vec![3,2],topology,vec![Mapping::Unmapped;2]),Arithmetic::<i64>::new());
    a.transform(|key,v|*v=(key%3+1) as i64);b.transform(|key,v|*v=(key/5+2) as i64);c.transform(|_,v|*v=10);
    c.contract_from_aligned("ij",&a,"ik",&b,"kj",2,3);
    for (key,v) in c.local_pairs() {assert_eq!(v,30+10*(key%3+1) as i64*(key/3+2) as i64);}
    // Inputs must remain unchanged despite internal broadcast-replica cleanup.
    for (key,v) in a.local_pairs() {assert_eq!(v,(key%3+1) as i64);}
    drop(c);drop(b);drop(a);
    if world.rank()==0 {println!("DIGIT / PASS tensor_contract: generic aligned matrix contraction, physical+virtual reduction, restored replicas; ranks={np}");}
    world.close();runtime.finalize();
}
