use ctf::{algebra::Arithmetic,context::{Context,Runtime},mapping::{Distribution,Mapping,Topology},
    symmetry::Symmetry::{self,*},symmetric_distribution::SymmetricDistribution,symmetric_tensor::SymmetricTensor};
fn tensor<'c,'r>(c:&'c Context<'r>,n:usize,links:Vec<Symmetry>,replicas:bool)
    ->SymmetricTensor<'c,'r,Arithmetic<f64>> {
    let topology=Topology::new(vec![c.size()]);let mut maps=vec![Mapping::Unmapped;links.len()];
    if !replicas && !maps.is_empty(){maps[0].augment_physical(&topology,0);}
    for map in &mut maps {map.augment_virtual(c.size()*2);}
    SymmetricTensor::new(c,SymmetricDistribution::new(Distribution::new(vec![n;links.len()],topology,maps),links),Arithmetic::new())
}
fn close(actual:Vec<f64>,expected:Vec<f64>) {
    assert_eq!(actual.len(),expected.len());
    for (a,b) in actual.into_iter().zip(expected){assert!(a.is_finite()&&(a-b).abs()<=1e-6,"actual={a}, expected={b}");}
}
fn run(c:&Context<'_>) {
    let mut sy=tensor(c,3,vec![SY,NS],false);sy.transform(|key,v|*v=key as f64+1.);
    let keys:Vec<_>=(0..9).collect();
    let mut ns=tensor(c,3,vec![NS,NS],true);ns.transform(|_,v|*v=10.);
    ns.sum_from("ij",&sy,"ij",2.,3.);
    close(ns.read(&keys),vec![32.,38.,44.,38.,40.,46.,44.,46.,48.]);
    let mut reduced=tensor(c,0,vec![],true);reduced.transform(|_,v|*v=10.);
    reduced.sum_from("",&sy,"ij",2.,3.);close(reduced.read(&[0]),vec![136.]);
    let mut a=tensor(c,3,vec![NS,NS],true);a.transform(|key,v|*v=key as f64+1.);
    let mut projected=tensor(c,3,vec![SY,NS],false);projected.transform(|_,v|*v=10.);
    projected.sum_from("ij",&a,"ij",2.,3.);
    // Symmetry-aware sum is an orbit sum, not the canonical-only repack.
    close(projected.read(&[0,3,4,6,7,8]),vec![34.,42.,50.,50.,58.,66.]);
    let mut transposed=tensor(c,3,vec![SY,NS],true);
    transposed.sum_from("ij",&sy,"ji",1.,0.);
    close(transposed.read(&[0,3,4,6,7,8]),vec![1.,4.,5.,7.,8.,9.]);
    let mut anti=tensor(c,3,vec![AS,NS],true);anti.transform(|_,v|*v=10.);
    anti.sum_from("ij",&sy,"ij",2.,3.);close(anti.read(&[3,6,7]),vec![30.;3]);
    let mut diagonal=tensor(c,3,vec![NS,NS],false);diagonal.transform(|_,v|*v=10.);
    diagonal.sum_from("ii",&sy,"ii",2.,3.);
    close(diagonal.read(&keys),vec![32.,10.,10.,10.,40.,10.,10.,10.,48.]);
    let mut triple=tensor(c,3,vec![SY,SY,NS],false);
    triple.transform(|key,v|*v=if key==21{7.}else{0.});
    let mut expanded=tensor(c,3,vec![NS,NS,NS],true);
    expanded.sum_from("ijk",&triple,"ijk",1.,0.);
    close(expanded.read(&[21,15,19,7,11,5,0]),vec![7.,7.,7.,7.,7.,7.,0.]);
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let parity=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&parity);parity.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_sy_sum: SY unfolding/projection, diagonals, mixed AS, three axes; atol=1e-6; world+parity");}
    world.close();runtime.finalize();}
