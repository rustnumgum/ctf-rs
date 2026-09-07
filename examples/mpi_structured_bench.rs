//! One-shot representative structured contractions; no speedup claim.
use ctf::{algebra::Arithmetic,context::Runtime,mapping::{Distribution,Mapping,Topology},
    sparse::SparseTensor,symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,symmetry::Symmetry::{NS,SY}};
fn main(){
    let case=std::env::args().nth(1).expect("case must be sparse or symmetric");
    let runtime=Runtime::initialize();let world=runtime.world();let np=world.size();let n=96;
    match case.as_str(){
        "sparse"=>{
            let d=Distribution::cyclic(vec![n,n],np);
            let mut a=SparseTensor::new(&world,d.clone(),Arithmetic::<f64>::new());
            let mut b=SparseTensor::new(&world,d.clone(),Arithmetic::<f64>::new());
            let mut output=SparseTensor::new(&world,d.clone(),Arithmetic::<f64>::new());
            a.write_add(&(0..n*n).filter(|&key|d.owner(key)==world.rank()&&(key%n+3*(key/n))%11==0)
                .map(|key|(key,1.)).collect::<Vec<_>>());
            b.write_add(&(0..n*n).filter(|&key|d.owner(key)==world.rank()&&(2*(key%n)+key/n)%13==0)
                .map(|key|(key,1.)).collect::<Vec<_>>());
            world.barrier();let start=std::time::Instant::now();
            output.gemm_sparse(&a,&b,if np==4{[2,2]}else{[np,1]},1.,0.);
            world.barrier();let elapsed=start.elapsed().as_secs_f64();
            let keys=[0,n+1,n*n-1];let expected:Vec<_>=keys.iter().map(|&key|{
                let(i,j)=(key%n,key/n);(0..n).filter(|&k|(i+3*k)%11==0&&(2*k+j)%13==0).count() as f64
            }).collect();assert_eq!(output.read(&keys),expected);
            if world.rank()==0{println!("sparse_gemm n={n} ranks={np} elapsed_s={elapsed:.6}");}
        },
        "symmetric"=>{
            let topology=Topology::new(vec![np]);let mut maps=vec![Mapping::Unmapped;2];
            maps[0].augment_physical(&topology,0);maps[1].augment_virtual(np);
            let d=SymmetricDistribution::new(Distribution::new(vec![n,n],topology.clone(),maps),vec![SY,NS]);
            let mut a=SymmetricTensor::new(&world,d.clone(),Arithmetic::<f64>::new());a.transform(|_,v|*v=1.);
            let mut b=SymmetricTensor::new(&world,d,Arithmetic::<f64>::new());b.transform(|_,v|*v=1.);
            let mut output=SymmetricTensor::new(&world,SymmetricDistribution::new(
                Distribution::cyclic(vec![n,n],np),vec![NS,NS]),Arithmetic::<f64>::new());
            world.barrier();let start=std::time::Instant::now();
            output.contract_from_on("ij",&a,"ik",&b,"kj",topology,"k",1.,0.,true).unwrap();
            world.barrier();let elapsed=start.elapsed().as_secs_f64();
            assert_eq!(output.read(&[0,n+1,n*n-1]),vec![n as f64;3]);
            if world.rank()==0{println!("symmetric_gemm n={n} ranks={np} elapsed_s={elapsed:.6}");}
        },
        _=>panic!("case must be sparse or symmetric"),
    }
    world.close();runtime.finalize();
}
