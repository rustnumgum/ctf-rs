use ctf::{algebra::Arithmetic,context::{Context,Runtime},mapping::{Distribution,Mapping,Topology},
    symmetry::Symmetry::{self,*},symmetric_distribution::SymmetricDistribution,symmetric_tensor::SymmetricTensor};
fn tensor<'c,'r>(c:&'c Context<'r>,links:Vec<Symmetry>)->SymmetricTensor<'c,'r,Arithmetic<f64>>{
    let topology=Topology::new(vec![c.size()]);let mut maps=vec![Mapping::Unmapped;links.len()];
    maps[2].augment_physical(&topology,0);for map in &mut maps{map.augment_virtual(2*c.size());}
    SymmetricTensor::new(c,SymmetricDistribution::new(Distribution::new(vec![3;links.len()],topology,maps),links),Arithmetic::new())
}
fn close(actual:Vec<f64>,expected:Vec<f64>){assert_eq!(actual.len(),expected.len());
    for(a,b)in actual.into_iter().zip(expected){assert!(a.is_finite()&&(a-b).abs()<=1e-6,"actual={a} expected={b}");}}
fn run(c:&Context<'_>){
    let mut a=tensor(c,vec![NS,AS,NS]);a.transform(|key,v|*v=key as f64+1.);
    let before=a.read(&(0..27).collect::<Vec<_>>());
    let(mut d,labels)=a.extract_diagonal("iji");
    assert_eq!(labels,"ij");assert_eq!(d.distribution().links(),&[NS,NS]);
    close(d.read(&(0..9).collect::<Vec<_>>()),vec![0.,11.,21.,-10.,0.,24.,-19.,-23.,0.]);
    d.transform(|_,v|*v*=2.);a.replace_diagonal("iji",&d);
    // Source rw=0 uses beta=0 on only the first output permutation. In the
    // second canonical chamber, old output survives and receives the new term.
    let expected:Vec<_>=(0..27).map(|key|{let i=key%3;let j=key/3%3;let k=key/9;
        let factor=if i==j.max(k){2.}else if i==j.min(k){3.}else{1.};
        factor*before[key]}).collect();
    close(a.read(&(0..27).collect::<Vec<_>>()),expected);
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let parity=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&parity);parity.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_cross_diagonal: cross-AS source run_diag extraction/reinsertion; atol=1e-6; world+parity");}
    world.close();runtime.finalize();}
