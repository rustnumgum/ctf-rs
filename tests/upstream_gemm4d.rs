// Adapted from pinned CTF test/gemm_4D.cxx, NS associativity branch.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use ctf::{algebra::Arithmetic,context::Runtime,linalg::Native,mapping::{Distribution,Topology},tensor::Tensor};
fn main() {
    let runtime=Runtime::initialize();let world=runtime.world();let np=world.size();let n=7;
    let topology=Topology::new(if np==4 {vec![2,2]} else {vec![np]});
    let make=||Tensor::new(&world,Distribution::cyclic(vec![n;4],np),Arithmetic::<f64>::new());
    let mut a=make();let mut b=make();let mut c=make();let mut state=((13*world.rank() as u64)<<16)|0x330e;
    for tensor in [&mut a,&mut b,&mut c] {tensor.transform(|_,v| {
        state=(state.wrapping_mul(0x5deece66d).wrapping_add(11))&((1<<48)-1);*v=state as f64/(1u64<<48) as f64-0.5;
    });}
    let mut ab=make();ab.contract_blas_on_grid::<Native>("ijkl",&a,"ijmn",&b,"mnkl",topology.clone(),1.,0.).unwrap();
    let mut left=make();left.contract_blas_on_grid::<Native>("ijkl",&ab,"ijmn",&c,"mnkl",topology.clone(),1.,0.).unwrap();
    let mut bc=make();bc.contract_blas_on_grid::<Native>("ijkl",&b,"ijmn",&c,"mnkl",topology.clone(),1.,0.).unwrap();
    let mut right=make();right.contract_blas_on_grid::<Native>("ijkl",&a,"ijmn",&bc,"mnkl",topology,1.,0.).unwrap();
    let mut maximum=0f64;
    for ((key,x),(other,y)) in left.local_pairs().into_iter().zip(right.local_pairs()) {
        assert_eq!(key,other);let error=(x-y).abs();assert!(error<1e-6);maximum=maximum.max(error);
    }
    println!("upstream_gemm4d NS rank={}: maximum absolute associativity error={maximum}, bound<1e-6",world.rank());
    drop(right);drop(bc);drop(left);drop(ab);drop(c);drop(b);drop(a);world.close();runtime.finalize();
}
