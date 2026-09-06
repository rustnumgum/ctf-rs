use ctf::{algebra::Arithmetic,context::Runtime,mapping::Distribution,tensor::Tensor,linalg::Native};
fn main() {
    let runtime = Runtime::initialize();let world = runtime.world();let np = world.size();
    let grid = if np == 4 {[2,2]} else {[np,1]};
    let (m,k,n) = (5,7,3);
    let mut a = Tensor::new(&world,Distribution::cyclic(vec![m,k],np),Arithmetic::<f64>::new());
    let mut b = Tensor::new(&world,Distribution::cyclic(vec![k,n],np),Arithmetic::<f64>::new());
    let mut c = Tensor::new(&world,Distribution::cyclic(vec![m,n],np),Arithmetic::<f64>::new());
    a.transform(|key,v|*v=(key%m+1) as f64);
    b.transform(|key,v|*v=(key/k+2) as f64);
    c.transform(|_,v|*v=10.);
    let original = c.distribution().clone();
    c.gemm_2d::<Native>(&a,&b,grid,2.,3.);
    assert_eq!(*c.distribution(),original);
    for (key,v) in c.local_pairs() {assert_eq!(v,30.+(2*k*(key%m+1)*(key/m+2)) as f64);}
    // Empty local row ownership on multi-rank grids, and a zero reduction length.
    let a0 = Tensor::new(&world,Distribution::cyclic(vec![1,0],np),Arithmetic::<f64>::new());
    let b0 = Tensor::new(&world,Distribution::cyclic(vec![0,1],np),Arithmetic::<f64>::new());
    let mut c0 = Tensor::new(&world,Distribution::cyclic(vec![1,1],np),Arithmetic::<f64>::new());
    c0.transform(|_,v|*v=5.);c0.gemm_2d::<Native>(&a0,&b0,grid,1.,2.);
    assert_eq!(c0.read(&[0]),vec![10.]);
    drop(c0);drop(b0);drop(a0);drop(c);drop(b);drop(a);
    if world.rank()==0 {println!("DIGIT / PASS tensor_gemm: MPI panels + BLAS, uneven dimensions, restored distribution, zero reduction; ranks={np}");}
    world.close();runtime.finalize();
}
