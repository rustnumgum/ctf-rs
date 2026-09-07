//! Port of test/diag_sym.cxx: preserve its norm < 1e-10 criterion.
use ctf::{algebra::Arithmetic,context::{Context,Runtime},mapping::{Distribution,Mapping,Topology},
    symmetry::Symmetry::{self,*},symmetric_distribution::SymmetricDistribution,symmetric_tensor::SymmetricTensor};
fn tensor<'c,'r>(c:&'c Context<'r>,links:Vec<Symmetry>)->SymmetricTensor<'c,'r,Arithmetic<f64>> {
    let topology=Topology::new(vec![c.size()]);let mut maps=vec![Mapping::Unmapped;links.len()];
    maps[0].augment_physical(&topology,0);
    for map in &mut maps {map.augment_virtual(2*c.size());}
    SymmetricTensor::new(c,SymmetricDistribution::new(Distribution::new(vec![3;links.len()],topology,maps),links),Arithmetic::new())
}
fn run(c:&Context<'_>){
    let mut ma=tensor(c,vec![NS,NS]);ma.transform(|key,v|*v=(key as f64-4.)/16.);
    let mut mb=tensor(c,vec![NS,NS]);mb.transform(|key,v|*v=(2.*key as f64-7.)/32.);
    let mut a=tensor(c,vec![SY,NS,SY,NS]);let mut b=tensor(c,vec![SY,NS,SY,NS]);
    let mut difference=tensor(c,vec![SY,NS,SY,NS]);
    a.sum_from("abij",&ma,"ii",1.,0.);
    b.sum_from("abij",&ma,"jj",1.,0.);
    a.sum_from("abij",&mb,"aa",-1.,1.);
    b.sum_from("abij",&mb,"bb",-1.,1.);
    difference.sum_from("abij",&a,"abij",1.,0.);
    difference.sum_from("abij",&b,"abij",-1.,1.);
    let values=difference.read(&(0..81).collect::<Vec<_>>());
    let norm=values.iter().map(|v|v*v).sum::<f64>().sqrt();
    assert!(norm.is_finite()&&norm<1e-10,"source diag_sym norm={norm}");
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let parity=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&parity);parity.close();
    if world.rank()==0{println!("DIGIT / PASS upstream_diag_sym: source paired-symmetry diagonal summation identity; norm<1e-10; world+parity");}
    world.close();runtime.finalize();}
