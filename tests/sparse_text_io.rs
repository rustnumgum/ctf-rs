use ctf::{algebra::Arithmetic,context::{Context,Runtime},mapping::{Distribution,Mapping,Topology},sparse::SparseTensor,tensor::Tensor};
use std::fmt::Write;

macro_rules! scalar_case {
    ($name:ident,$t:ty,$convert:expr)=>{
        fn $name(context:&Context<'_>,token:u64,scope:usize){
            let value=$convert;let entries=[(0usize,2),(5,3),(0,-1),(11,4),(7,-2)];
            let topology=Topology::new(vec![context.size()]);let mut mode=Mapping::Unmapped;
            mode.augment_physical(&topology,0);mode.augment_virtual(2*context.size());
            let distribution=Distribution::new(vec![3,4],topology,vec![mode,Mapping::Unmapped]);
            for with_values in [false,true]{for reverse in [false,true]{
                let base=std::env::temp_dir().join(format!("ctf-rs-io-{token}-{scope}-{}-{with_values}-{reverse}",stringify!($name)));
                let input=base.with_extension("input.txt");let output=base.with_extension("output.txt");
                if context.rank()==0{
                    let mut text=String::new();for &(key,x)in &entries{
                        let(i,j)=(key%3,key/3);let(a,b)=if reverse{(j,i)}else{(i,j)};
                        write!(&mut text,"{a} {b}").unwrap();if with_values{write!(&mut text," {}",value(x)).unwrap();}text.push('\n');
                    }std::fs::write(&input,text).unwrap();
                }context.barrier();
                let expected=|key:usize|{
                    let mut x=if key==2{1 as $t}else{0 as $t};for &(k,n)in &entries{if k==key{x+=if with_values{value(n)}else{1 as $t};}}x
                };
                for sparse_case in [false,true]{
                    if context.rank()==0{std::fs::write(&output,"stale content\n".repeat(100)).unwrap();}context.barrier();
                    if sparse_case{
                        let mut tensor=SparseTensor::new(context,distribution.clone(),Arithmetic::<$t>::new());
                        tensor.write_add(&if context.rank()==0{vec![(2,1 as $t)]}else{Vec::new()});
                        tensor.read_sparse_from_file(&input,with_values,reverse);
                        for(key,x)in tensor.local_pairs(){assert_eq!(x,expected(key));}
                        tensor.write_sparse_to_file(&output,with_values,reverse);
                    }else{
                        let mut tensor=Tensor::new(context,distribution.clone(),Arithmetic::<$t>::new());tensor.transform(|key,x|*x=if key==2{1 as $t}else{0 as $t});
                        tensor.read_sparse_from_file(&input,with_values,reverse);
                        for(key,x)in tensor.local_pairs(){assert_eq!(x,expected(key));}
                        tensor.write_sparse_to_file(&output,with_values,reverse);
                    }
                    // Read the MPI-written coordinate file into the other storage
                    // representation; no matrix gather is used by either API.
                    let mut restored=Tensor::new(context,distribution.clone(),Arithmetic::<$t>::new());
                    restored.read_sparse_from_file(&output,with_values,reverse);
                    for(key,x)in restored.local_pairs(){assert_eq!(x,if with_values{expected(key)}else{(expected(key)!=0 as $t)as u8 as $t});}
                    context.barrier();if context.rank()==0{std::fs::remove_file(&output).unwrap();}context.barrier();
                }
                if context.rank()==0{std::fs::remove_file(&input).unwrap();}context.barrier();
            }}
        }
    }
}
scalar_case!(int32,i32,|x:i32|x);
scalar_case!(int64,i64,|x:i32|x as i64);
scalar_case!(real32,f32,|x:i32|x as f32/2.);
scalar_case!(real64,f64,|x:i32|x as f64/2.);
fn empty(context:&Context<'_>,token:u64,scope:usize){
    let path=std::env::temp_dir().join(format!("ctf-rs-io-{token}-{scope}-empty.txt"));
    let distribution=Distribution::cyclic(vec![1,1],context.size());
    let tensor=SparseTensor::new(context,distribution.clone(),Arithmetic::<i64>::new());
    tensor.write_sparse_to_file(&path,true,false);
    let mut read=SparseTensor::new(context,distribution,Arithmetic::<i64>::new());read.read_sparse_from_file(&path,true,false);
    assert_eq!(read.local_nnz(),0);context.barrier();if context.rank()==0{assert_eq!(std::fs::metadata(&path).unwrap().len(),0);std::fs::remove_file(&path).unwrap();}context.barrier();
}
fn run(c:&Context<'_>,token:u64,scope:usize){int32(c,token,scope);int64(c,token,scope);real32(c,token,scope);real64(c,token,scope);empty(c,token,scope);}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();let mut token=[std::process::id()as u64];world.broadcast(0,&mut token);run(&world,token[0],0);
    let color=world.rank()%2;let child=world.split(Some(color as i32),world.rank()as i32).unwrap();run(&child,token[0],color+1);child.close();
    if world.rank()==0{println!("DIGIT / PASS sparse_text_io: four scalar families, distributed dense/sparse additive reads and writes, duplicate keys, reverse/no-value records, tiny/empty files; exact fixtures, world+parity");}
    world.close();runtime.finalize();}
