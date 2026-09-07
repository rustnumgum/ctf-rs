//! Source krnl_type=5: CCSR sparse output from sparse A and dense B.
use ctf::{algebra::Arithmetic, context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology}, sparse::SparseTensor, tensor::Tensor};

fn sparse<'c, 'r>(context: &'c Context<'r>, shape: Vec<usize>, pairs: &[(usize,i64)])
    -> SparseTensor<'c,'r,Arithmetic<i64>> {
    let topology=Topology::new(vec![context.size()]);
    let mut maps=vec![Mapping::Unmapped;shape.len()];
    maps[0].augment_physical(&topology,0);
    maps[0].augment_virtual(2*context.size());
    let distribution=Distribution::new(shape,topology,maps);
    let mut tensor=SparseTensor::new(context,distribution.clone(),Arithmetic::new());
    tensor.write_add(&pairs.iter().copied().filter(|(key,_)|distribution.owner(*key)==context.rank()).collect::<Vec<_>>());
    tensor
}

fn matrix(context:&Context<'_>,grid:[usize;2]) {
    let pairs=vec![(0,2),(10,-1),(7,3),(4,0)];
    let a=sparse(context,vec![5,3],&pairs);
    let mut b=Tensor::new(context,Distribution::cyclic(vec![3,3],context.size()),Arithmetic::<i64>::new());
    b.transform(|key,value|*value=key as i64+1);
    for beta in [0,3] {
        let old=vec![(1,5),(8,0)];
        let mut c=sparse(context,vec![5,3],&old);
        let original=c.distribution().clone();
        c.gemm_sparse_dense(&a,&b,grid,2,beta);
        assert_eq!(c.distribution(),&original);
        let mut expected=Vec::new();
        for j in 0..3 { for i in 0..5 {
            let key=i+5*j;
            if [0,2,4].contains(&i) || (beta!=0 && old.iter().any(|(k,_)|*k==key)) {
                let product:i64=pairs.iter().filter(|(key,_)|key%5==i)
                    .map(|(key,value)|value*((key/5+3*j) as i64+1)).sum();
                let old=old.iter().find(|(k,_)|*k==key).map_or(0,|(_,v)|*v);
                expected.push((key,2*product+beta*old));
            }
        }}
        let expected:Vec<_>=expected.into_iter().filter(|(key,_)|original.owner(*key)==context.rank()).collect();
        let mut actual=c.local_pairs(); actual.sort_by_key(|pair|pair.0);
        assert_eq!(actual,expected,"grid={grid:?}, beta={beta}");
    }
    let empty=sparse(context,vec![1,0],&[]);
    let b=Tensor::new(context,Distribution::cyclic(vec![0,1],context.size()),Arithmetic::<i64>::new());
    let mut c=sparse(context,vec![1,1],&[(0,7)]);
    c.gemm_sparse_dense(&empty,&b,grid,2,3);
    assert_eq!(c.read(&[0]),vec![21]);
}

fn folded(context:&Context<'_>,grid:[usize;2]) {
    let pairs=vec![(0,2),(16,3),(18,4),(34,-1),(2,99)];
    let a=sparse(context,vec![2,3,3,2],&pairs);
    let mut b=Tensor::new(context,Distribution::cyclic(vec![3,2,2],context.size()),Arithmetic::<i64>::new());
    b.transform(|key,value|*value=key as i64+1);
    let mut c=sparse(context,vec![2,2,2],&(0..8).map(|key|(key,7)).collect::<Vec<_>>());
    let original=c.distribution().clone();
    c.contract_from_sparse_dense("iji",&a,"ikkx",&b,"kjy",grid,2,3).unwrap();
    assert_eq!(c.distribution(),&original);
    let mut expected_pairs=Vec::new();
    for key in 0..8 {
        let i=key%2; let j=key/2%2;
        let expected=if key/4!=i {7} else {
            let mut product=0;
            for k in 0..3 { for x in 0..2 { for y in 0..2 {
                let ak=i+8*k+18*x;
                let av=pairs.iter().find(|(key,_)|*key==ak).map_or(0,|(_,value)|*value);
                product+=av*((k+3*j+6*y) as i64+1);
            }}}
            21+2*product
        };
        if original.owner(key)==context.rank() {expected_pairs.push((key,expected));}
    }
    let mut actual=c.local_pairs(); actual.sort_by_key(|pair|pair.0);
    assert_eq!(actual,expected_pairs,"grid={grid:?}");
}

fn run(context:&Context<'_>) {
    let grids=if context.size()==4 {vec![[2,2],[1,4]]} else {vec![[context.size(),1]]};
    for grid in grids {matrix(context,grid);folded(context,grid);}
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_sparse_dense_output: CCSR represented rows/zeros, beta once, rectangular grids, empty k, repeated labels, input-only sums, restored sparse layout; exact i64; world+parity");}
    world.close();runtime.finalize();}
