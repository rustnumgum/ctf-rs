use ctf::{algebra::Arithmetic,context::{Context,Runtime},mapping::{Distribution,Topology},
    sparse::SparseTensor,tensor::Tensor};
fn run(c:&Context<'_>){
    let d=Distribution::cyclic(vec![],c.size());
    let mut b=Tensor::new(c,Distribution::cyclic(vec![4],c.size()),Arithmetic::<i64>::new());
    b.transform(|key,value|*value=[0,1,2,0][key]);
    for stored in [false,true]{for factors in [&[][..],&[(b'i',3),(b'x',3)][..]]{
        let mut a=SparseTensor::new(c,d.clone(),Arithmetic::<i64>::new());
        a.write_add(&if stored && d.owner(0)==c.rank(){vec![(0,0)]}else{vec![]});
        let mut output=Tensor::new(c,Distribution::cyclic(vec![4,2],c.size()),Arithmetic::<i64>::new());
        output.transform(|_,value|*value=3);
        output.contract_from_sparse_dense_function_on("ix",&a,"",&b,"i",
            Topology::new(vec![c.size()]),"i",factors,1,3,true,|a,b|2*a+b+1);
        let keys:Vec<_>=(0..8).collect();
        let expected:Vec<_>=keys.iter().map(|&key|9+if stored{[0,1,2,0][key%4]+1}else{0}).collect();
        assert_eq!(output.read(&keys),expected,"stored={stored}");
    }}
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_sparse_function: non-distributive function, stored and dense zeros, missing keys, C-only output, broadcasts/reductions, world+parity; exact i64");}
    world.close();runtime.finalize();}
