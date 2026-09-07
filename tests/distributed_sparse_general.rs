use ctf::{algebra::Arithmetic, context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology}, sparse::SparseTensor, tensor::Tensor};
fn run(c: &Context<'_>) {
    let original = Topology::new(vec![c.size()]);
    let mut map = Mapping::Unmapped; map.augment_physical(&original,0); map.augment_virtual(2*c.size());
    let distribution = Distribution::new(vec![2,3],original,vec![map,Mapping::Unmapped]);
    let entries = [(0,2_i64),(3,-1),(4,3)];
    let mut a = SparseTensor::new(c, distribution.clone(),Arithmetic::new());
    a.write_add(&entries.iter().copied().filter(|(key,_)|distribution.owner(*key)==c.rank()).collect::<Vec<_>>());
    let mut b = Tensor::new(c,Distribution::cyclic(vec![3,2],c.size()),Arithmetic::<i64>::new());
    b.transform(|key,value|*value=key as i64-2);
    let mut plans = vec![(Topology::new(vec![c.size()]),"i"),
        (Topology::new(vec![c.size()]),"j"),(Topology::new(vec![c.size()]),"k")];
    if c.size()==4 { plans.push((Topology::new(vec![2,2]),"ij")); }
    for (topology,physical) in plans { for empty in [false,true] { for factors in [&[][..],&[(b'i',2),(b'j',2),(b'k',3),(b'x',2)][..]] {
        let aa = if empty { SparseTensor::new(c,distribution.clone(),Arithmetic::new()) } else {a.clone()};
        let mut output = Tensor::new(c,Distribution::cyclic(vec![2,2,3],c.size()),Arithmetic::new());
        output.transform(|key,value|*value=key as i64+7);
        let saved = output.distribution().clone();
        output.contract_from_sparse_dense_on("ijx",&aa,"ik",&b,"kj",topology.clone(),physical,factors,2,3,true);
        assert_eq!(output.distribution(),&saved);
        let keys:Vec<_>=(0..12).collect();
        let expected:Vec<_>=keys.iter().map(|&key| {
            let i=key%2;let j=key/2%2;
            3*(key as i64+7)+if empty{0}else{entries.iter().filter(|(ak,_)|ak%2==i)
                .map(|&(ak,av)|2*av*((ak/2+3*j) as i64-2)).sum::<i64>()}
        }).collect();
        assert_eq!(output.read(&keys),expected,"physical={physical} empty={empty}");
    } } }
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_sparse_general: C-only indices, sparse/dense fiber broadcasts, output reduction, original distribution restore, empty sparse shards, world+parity; exact i64");}
    world.close();runtime.finalize();}
