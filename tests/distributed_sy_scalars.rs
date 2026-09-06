use ctf::{algebra::{Arithmetic,Complex,Group,Monoid,Semiring,Wire},context::{Context,Runtime},
    mapping::{Distribution,Mapping,Topology},scalar_conversion::CastFromF64,
    symmetry::Symmetry::{self,*},symmetric_distribution::SymmetricDistribution,symmetric_tensor::SymmetricTensor};
fn tensor<'c,'r,A:Group>(c:&'c Context<'r>,links:Vec<Symmetry>,algebra:A)
    ->SymmetricTensor<'c,'r,A> {
    let topology=Topology::new(vec![c.size()]);let mut maps=vec![Mapping::Unmapped;links.len()];
    if !maps.is_empty(){maps[0].augment_physical(&topology,0);}
    for map in &mut maps {map.augment_virtual(2*c.size());}
    SymmetricTensor::new(c,SymmetricDistribution::new(Distribution::new(vec![3;links.len()],topology,maps),links),algebra)
}
fn real<A:Group+Semiring+Clone+CastFromF64>(c:&Context<'_>,algebra:A) where A::Element:Wire+std::fmt::Debug {
    let mut a=tensor(c,vec![SY,NS],algebra.clone());
    a.transform(|key,v|*v=algebra.cast_f64(key as f64+1.));
    let mut b=tensor(c,vec![NS,NS],algebra.clone());
    b.sum_from("ij",&a,"ij",algebra.cast_f64(2.),algebra.zero());
    assert_eq!(b.read(&(0..9).collect::<Vec<_>>()),[2.,8.,14.,8.,10.,16.,14.,16.,18.].into_iter().map(|v|algebra.cast_f64(v)).collect::<Vec<_>>());
    let mut reduced=tensor(c,vec![],algebra.clone());
    reduced.sum_from("",&a,"ij",algebra.one(),algebra.zero());
    assert_eq!(reduced.read(&[0]),vec![algebra.cast_f64(53.)]);
}
// A non-Copy scalar exercises owned coefficient propagation, not f64 aliases.
#[derive(Clone,Debug,PartialEq)]struct Boxed(Box<f64>);
impl Wire for Boxed {
    const WIDTH:usize=8;
    fn encode(&self,out:&mut Vec<u8>){(*self.0).encode(out);}
    fn decode(bytes:&[u8])->Self{Self(Box::new(f64::decode(bytes)))}
}
#[derive(Clone)]struct BoxedRing;
impl Monoid for BoxedRing {type Element=Boxed;
    fn zero(&self)->Boxed{Boxed(Box::new(0.))}
    fn add(&self,a:&Boxed,b:&Boxed)->Boxed{Boxed(Box::new(*a.0+*b.0))}}
impl Group for BoxedRing {fn negate(&self,a:&Boxed)->Boxed{Boxed(Box::new(-*a.0))}}
impl Semiring for BoxedRing {
    fn one(&self)->Boxed{Boxed(Box::new(1.))}
    fn multiply(&self,a:&Boxed,b:&Boxed)->Boxed{Boxed(Box::new(*a.0 * *b.0))}}
impl CastFromF64 for BoxedRing {fn cast_f64(&self,v:f64)->Boxed{Boxed(Box::new(v))}}
fn complex(c:&Context<'_>){
    let mut a=tensor(c,vec![SY,NS],Arithmetic::<Complex<f64>>::new());
    a.transform(|key,v|*v=Complex{re:key as f64+1.,im:2.*(key as f64+1.)});
    let mut b=tensor(c,vec![NS,NS],Arithmetic::<Complex<f64>>::new());
    b.sum_from("ij",&a,"ij",Complex{re:0.,im:1.},Complex{re:0.,im:0.});
    let expected=[1.,4.,7.,4.,5.,8.,7.,8.,9.].map(|v|Complex{re:-2.*v,im:v});
    assert_eq!(b.read(&(0..9).collect::<Vec<_>>()),expected);
}
fn run(c:&Context<'_>){real(c,Arithmetic::<f32>::new());real(c,Arithmetic::<i32>::new());
    real(c,Arithmetic::<i64>::new());real(c,Arithmetic::<Complex<f32>>::new());real(c,BoxedRing);complex(c);}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let parity=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&parity);parity.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_sy_scalars: f32, integers, complex, non-Copy custom ring; exactly representable fixtures; world+parity");}
    world.close();runtime.finalize();}
