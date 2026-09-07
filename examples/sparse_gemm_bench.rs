//! One fixed representative sparse measurement; no warmup/sweep/speedup claim.
use ctf::{algebra::Arithmetic,context::Runtime,mapping::Distribution,sparse::SparseTensor};
fn main(){
    let runtime=Runtime::initialize();let world=runtime.world();let np=world.size();
    assert!([1,2,4].contains(&np));
    let grid=if np==4{[2,2]}else{[np,1]};let(m,k,n)=(127,83,97);
    let make=|shape:Vec<usize>,modulus:usize|{
        let distribution=Distribution::cyclic(shape.clone(),np);
        let mut tensor=SparseTensor::new(&world,distribution.clone(),Arithmetic::<f64>::new());
        let pairs:Vec<_>=(0..shape.iter().product()).filter(|&key|key%modulus==0&&distribution.owner(key)==world.rank())
            .map(|key|(key,((key%17)+1)as f64/17.0)).collect();
        tensor.write_add(&pairs);tensor
    };
    let a=make(vec![m,k],5);let b=make(vec![k,n],7);
    let mut c=SparseTensor::new(&world,Distribution::cyclic(vec![m,n],np),Arithmetic::<f64>::new());
    world.barrier();let start=std::time::Instant::now();
    c.gemm_sparse(&a,&b,grid,1.0,0.0);
    world.barrier();let seconds=start.elapsed().as_secs_f64();
    let nnz=world.all_reduce(&Arithmetic::<i64>::new(),&(c.local_nnz()as i64));
    if world.rank()==0{println!("sparse_gemm m={m} k={k} n={n} ranks={np} grid={}x{} elapsed_s={seconds:.6} output_nnz={nnz}",grid[0],grid[1]);}
    drop(c);drop(b);drop(a);world.close();runtime.finalize();
}
