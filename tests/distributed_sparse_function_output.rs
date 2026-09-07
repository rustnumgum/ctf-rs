use ctf::{algebra::Arithmetic,context::{Context,Runtime},mapping::Distribution,sparse::SparseTensor};
fn run(c:&Context<'_>){
    let d=Distribution::cyclic(vec![2,2],c.size());
    let mut a=SparseTensor::new(c,d.clone(),Arithmetic::<i64>::new());
    let mut b=SparseTensor::new(c,d.clone(),Arithmetic::<i64>::new());
    a.write_add(&[(0,0),(3,2)].into_iter().filter(|(key,_)|d.owner(*key)==c.rank()).collect::<Vec<_>>());
    b.write_add(&[(0,0),(3,0)].into_iter().filter(|(key,_)|d.owner(*key)==c.rank()).collect::<Vec<_>>());
    for beta in [0,3]{
        let mut output=SparseTensor::new(c,d.clone(),Arithmetic::<i64>::new());
        output.write_add(&[(1,3),(3,5)].into_iter().filter(|(key,_)|d.owner(*key)==c.rank()).collect::<Vec<_>>());
        output.gemm_sparse_function(&a,&b,if c.size()==4{[2,2]}else{[c.size(),1]},1,beta,|a,b|a+b);
        let expected=if beta==0{vec![(0,0),(1,0),(3,2)]}else{vec![(0,0),(1,9),(3,17)]};
        let local:Vec<_>=expected.into_iter().filter(|(key,_)|d.owner(*key)==c.rank()).collect();
        assert_eq!(output.local_pairs(),local,"beta={beta}");
        assert_eq!(output.distribution(),&d);
    }
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_sparse_function_output: source symbolic/numeric CSR custom product, structural zeros, beta clear/merge, world+parity; exact keys and i64");}
    world.close();runtime.finalize();}
