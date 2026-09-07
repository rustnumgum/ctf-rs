use ctf::{algebra::{Arithmetic,Complex},context::{Context,Runtime},mapping::{Distribution,Mapping,Topology},
    symmetric_distribution::SymmetricDistribution,symmetric_tensor::SymmetricTensor,symmetry::Symmetry::{self,NS,SY,AS,SH}};

fn layout(n:usize,kind:Symmetry,np:usize)->SymmetricDistribution {
    let topology=Topology::new(vec![np]);let mut first=Mapping::Unmapped;
    first.augment_physical(&topology,0);first.augment_virtual(2*np);
    let mut second=Mapping::Unmapped;second.augment_virtual(2*np);
    SymmetricDistribution::new(Distribution::new(vec![n,n],topology,vec![first,second]),vec![kind,NS])
}
macro_rules! real_case {
    ($name:ident,$t:ty)=>{
        fn $name(context:&Context<'_>){for kind in [SY,AS,SH,NS]{for n in [3,1]{
            let distribution=layout(n,kind,context.size());
            let mut tensor=SymmetricTensor::new(context,distribution.clone(),Arithmetic::<$t>::new());
            tensor.transform(|key,x|*x=(key%2+1)as $t);
            let before=tensor.local_storage().to_vec();
            let input=|key|distribution.canonicalize(key).map(|(canonical,sign)|((canonical%2+1)as i32*sign)as $t).unwrap_or(0 as $t);
            let unpacked=tensor.unpack(Distribution::cyclic(vec![n,n],context.size()));
            for(key,x)in unpacked.local_pairs(){assert_eq!(x,input(key));}
            let expected1=(0..n*n).map(|key|(input(key)as f64).abs()).sum::<f64>();
            let expected2=(0..n*n).map(|key|(input(key)as f64).powi(2)).sum::<f64>().sqrt();
            let expected_max=(0..n*n).map(|key|(input(key)as f64).abs()).fold(0.,f64::max);
            for(actual,expected)in[(tensor.norm1(),expected1),(tensor.norm2(),expected2),(tensor.norm_infty(),expected_max)]{
                assert!(actual.is_finite()&&(actual-expected).abs()<1e-6,"symmetric norm {kind:?} n={n} actual={actual} expected={expected}");
            }assert_eq!(tensor.local_storage(),before);
        }}}
    }
}
real_case!(int8,i8);real_case!(int16,i16);real_case!(int32,i32);real_case!(int64,i64);real_case!(real32,f32);real_case!(real64,f64);
macro_rules! complex_case {
    ($name:ident,$t:ty)=>{
        fn $name(context:&Context<'_>){for kind in [SY,AS,SH,NS]{
            let distribution=layout(3,kind,context.size());
            let mut tensor=SymmetricTensor::new(context,distribution.clone(),Arithmetic::<Complex<$t>>::new());
            tensor.transform(|_,x|*x=Complex::new(3. as $t,4. as $t));
            let expected=(0..9).filter(|&key|distribution.canonicalize(key).is_some()).count()as f64*25.;
            let actual=tensor.norm2();assert!(actual.is_finite()&&(actual-expected.sqrt()).abs()<1e-6);
            let unpacked=tensor.unpack(Distribution::cyclic(vec![3,3],context.size()));
            for(key,x)in unpacked.local_pairs(){let sign=distribution.canonicalize(key).map(|(_,sign)|sign).unwrap_or(0)as $t;
                assert_eq!(x,Complex::new(3. as $t*sign,4. as $t*sign));}
        }}
    }
}
complex_case!(complex32,f32);complex_case!(complex64,f64);
fn run(c:&Context<'_>){int8(c);int16(c);int32(c);int64(c);real32(c);real64(c);complex32(c);complex64(c);}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS symmetric_norms: SY/AS/SH/NS, source typed-square vs manual norm2, exact distributed unpack, virtual/padded/empty shards; finite abs<1e-6, world+parity");}
    world.close();runtime.finalize();}
