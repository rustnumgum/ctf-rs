use ctf::{algebra::Arithmetic,context::Runtime,mapping::{Distribution,Topology},tensor::Tensor,linalg::Native};
fn main() {
    let runtime=Runtime::initialize();let world=runtime.world();
    let child=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();
    let np=child.size();let make=|shape|Tensor::new(&child,Distribution::cyclic(shape,np),Arithmetic::<f64>::new());
    let mut a=make(vec![3,3]);a.transform(|key,v|*v=if key%3==key/3 {1.} else {0.});
    let mut b=make(vec![3,2]);b.transform(|_,v|*v=2.);
    let mut c=make(vec![3,2]);c.contract_from("ij",&a,"ik",&b,"kj",Topology::new(vec![np]),1.,0.).unwrap();
    for (_,v) in c.local_pairs() {assert_eq!(v,2.);}
    c.gemm_2d::<Native>(&a,&b,[np,1],1.,1.);for (_,v) in c.local_pairs() {assert_eq!(v,4.);}
    let mut trace=make(vec![]);trace.sum_from("",&a,"ii",Topology::new(vec![np]),1.,0.).unwrap();assert_eq!(trace.reduce(),3.);
    drop(trace);drop(c);drop(b);drop(a);child.close();world.barrier();
    if world.rank()==0 {println!("DIGIT / PASS subcomm_dense: split-context repeated sum, generic contraction and MPI BLAS; world ranks={}",world.size());}
    world.close();runtime.finalize();
}
