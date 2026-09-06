use ctf::{algebra::{CustomMonoid,Wire},context::Runtime};
#[derive(Clone,Debug,PartialEq)]
struct Affine(i64,i64);
impl Wire for Affine {
    const WIDTH:usize=16;
    fn encode(&self,out:&mut Vec<u8>) {self.0.encode(out);self.1.encode(out);}
    fn decode(bytes:&[u8])->Self {Self(i64::decode(&bytes[..8]),i64::decode(&bytes[8..]))}
}
fn main() {
    let runtime=Runtime::initialize();let world=runtime.world();
    let compose=CustomMonoid {identity:Affine(1,0),addition:|a:&Affine,b:&Affine|Affine(a.0*b.0,a.0*b.1+a.1)};
    let mut values=[Affine(2,world.rank() as i64),Affine(1,1)];
    world.all_reduce_monoid(&compose,&mut values,false);
    let mut expected=Affine(1,0);
    for rank in 0..world.size() {expected=Affine(expected.0*2,expected.0*rank as i64+expected.1);}
    assert_eq!(values,[expected,Affine(1,world.size() as i64)]);
    world.all_reduce_monoid(&compose,&mut [],false);
    let modulus=7i64;
    let sum=CustomMonoid {identity:0i64,addition:|a:&i64,b:&i64|(a+b)%modulus};
    let mut value=[world.rank() as i64+1];world.all_reduce_monoid(&sum,&mut value,true);
    assert_eq!(value,[(world.size()*(world.size()+1)/2) as i64%modulus]);
    if world.rank()==0 {println!("DIGIT / PASS custom_reduce: MPI user op, ordered affine monoid, captured algebra and zero count; ranks={}",world.size());}
    world.close();runtime.finalize();
}
