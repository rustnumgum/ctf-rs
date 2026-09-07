use ctf::{algebra::{Arithmetic,Complex},context::{Context,Runtime},mapping::{Distribution,Mapping,Topology},
    sparse::SparseTensor,tensor::Tensor};

macro_rules! real_case {
    ($name:ident,$t:ty)=>{
        fn $name(context:&Context<'_>){
            for replicated in [false,true]{
                let distribution=if replicated{Distribution::new(vec![3],Topology::new(vec![context.size()]),vec![Mapping::Unmapped])}
                    else{Distribution::cyclic(vec![3],context.size())};
                let input=[-3 as $t,0 as $t,4 as $t];
                let mut dense=Tensor::new(context,distribution.clone(),Arithmetic::<$t>::new());
                dense.transform(|key,x|*x=input[key]);
                let mut sparse=SparseTensor::new(context,distribution.clone(),Arithmetic::<$t>::new());
                let pairs:Vec<_>=input.iter().enumerate().filter(|(key,_)|distribution.owner(*key)==context.rank()).map(|(key,&x)|(key,x)).collect();
                sparse.write_add(&pairs);
                let before=dense.local_storage().to_vec();let sparse_before=sparse.local_pairs();
                let norm2=5.*if replicated{(context.size()as f64).sqrt()}else{1.};
                for actual in [dense.norm1(),sparse.norm1()]{assert!(actual.is_finite()&&(actual-7.).abs()<1e-6);}
                for actual in [dense.norm_infty(),sparse.norm_infty()]{assert_eq!(actual,4.);}
                for actual in [dense.norm2(),sparse.norm2()]{assert!(actual.is_finite()&&(actual-norm2).abs()<1e-6,"source manual norm2 {actual} expected{norm2}");}
                assert_eq!(dense.local_storage(),before);assert_eq!(sparse.local_pairs(),sparse_before);
            }
        }
    }
}
real_case!(int8,i8);real_case!(int16,i16);real_case!(int32,i32);real_case!(int64,i64);
real_case!(real32,f32);real_case!(real64,f64);
macro_rules! complex_case {
    ($name:ident,$t:ty)=>{
        fn $name(context:&Context<'_>){
            let distribution=Distribution::cyclic(vec![3],context.size());
            let input=[Complex::new(3. as $t,4. as $t),Complex::new(0. as $t,0. as $t),Complex::new(0. as $t,-12. as $t)];
            let mut dense=Tensor::new(context,distribution.clone(),Arithmetic::<Complex<$t>>::new());dense.transform(|key,x|*x=input[key]);
            let mut sparse=SparseTensor::new(context,distribution.clone(),Arithmetic::<Complex<$t>>::new());
            sparse.write_add(&input.iter().enumerate().filter(|(key,_)|distribution.owner(*key)==context.rank()).map(|(key,&x)|(key,x)).collect::<Vec<_>>());
            for actual in [dense.norm2(),sparse.norm2()]{assert!(actual.is_finite()&&(actual-13.).abs()<1e-6);}
        }
    }
}
complex_case!(complex32,f32);complex_case!(complex64,f64);
fn boolean(context:&Context<'_>){
    let distribution=Distribution::cyclic(vec![3],context.size());
    let mut dense=Tensor::new(context,distribution.clone(),Arithmetic::<bool>::new());dense.transform(|key,x|*x=key!=1);
    let mut sparse=SparseTensor::new(context,distribution.clone(),Arithmetic::<bool>::new());
    sparse.write_add(&(0..3).filter(|&key|distribution.owner(key)==context.rank()).map(|key|(key,key!=1)).collect::<Vec<_>>());
    for actual in [dense.norm2(),sparse.norm2()]{assert!((actual-2f64.sqrt()).abs()<1e-6);}
}
fn run(c:&Context<'_>){int8(c);int16(c);int32(c);int64(c);real32(c);real64(c);complex32(c);complex64(c);boolean(c);}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS tensor_norms: dense/sparse real norms, complex/bool norm2, source NS storage reduction on replicas, empty local shards, unchanged inputs; finite abs<1e-6; world+parity");}
    world.close();runtime.finalize();}
