use ctf::{algebra::Arithmetic,context::{Context,Runtime},mapping::{Distribution,Mapping,Topology},tensor::Tensor};
fn run(c:&Context<'_>){
    for size in [0,1,3] { for replicated in [false,true] {
        let distribution=if replicated {Distribution::new(vec![size],Topology::new(vec![c.size()]),vec![Mapping::Unmapped])}
            else {Distribution::cyclic(vec![size],c.size())};
        for active in [false,true] {
            let mut t=Tensor::new(c,distribution.clone(),Arithmetic::<bool>::new());
            t.transform(|key,x|*x=active&&key+1==size);
            let expected=if active&&size>0{1.0}else{0.0};
            assert_eq!(t.norm_infty(),expected);
            let sparse=t.into_sparse(|_|true);assert_eq!(sparse.norm_infty(),expected);
        }
    }}
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS bool_norm: exact boolean MAXABS, dense/sparse false records, replicas and empty shards/domains; world+parity");}
    world.close();runtime.finalize();}
