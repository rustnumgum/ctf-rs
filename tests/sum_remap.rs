use ctf::{algebra::Arithmetic,context::Runtime,mapping::{Distribution,Topology},tensor::Tensor};
fn main() {
    let runtime=Runtime::initialize();let world=runtime.world();let np=world.size();
    let topo=Topology::new(if np==4 {vec![2,2]} else {vec![np]});
    let mut a=Tensor::new(&world,Distribution::cyclic(vec![5,7],np),Arithmetic::<i64>::new());
    a.transform(|key,v|*v=(key+1) as i64);
    let mut b=Tensor::new(&world,Distribution::cyclic(vec![5,3],np),Arithmetic::<i64>::new());
    b.transform(|_,v|*v=10);let original=b.distribution().clone();
    b.sum_from_on_grid("ij",&a,"ik",topo.clone(),2,3).unwrap();
    assert_eq!(*b.distribution(),original);
    for (key,v) in b.local_pairs() {assert_eq!(v,30+2*(112+7*(key%5)) as i64);}
    let mut transposed=Tensor::new(&world,Distribution::cyclic(vec![7,5],np),Arithmetic::<i64>::new());
    transposed.sum_from_on_grid("ki",&a,"ik",topo,1,0).unwrap();
    for (key,v) in transposed.local_pairs() {assert_eq!(v,(key/7+5*(key%7)+1) as i64);}
    drop(transposed);drop(b);drop(a);
    if world.rank()==0 {println!("DIGIT / PASS sum_remap: mapped union indices, reduction+broadcast, transpose, restored distribution; ranks={np}");}
    world.close();runtime.finalize();
}
