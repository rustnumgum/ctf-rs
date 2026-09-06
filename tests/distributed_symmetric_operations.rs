use ctf::{algebra::Arithmetic, context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology}, symmetry::Symmetry,
    symmetric_distribution::SymmetricDistribution, symmetric_tensor::SymmetricTensor};

fn run(context: &Context<'_>) {
    let topology = Topology::new(vec![context.size()]);
    let mut first = Mapping::Unmapped;
    first.augment_physical(&topology, 0);
    first.augment_virtual(2*context.size());
    let mut second = Mapping::Unmapped;
    second.augment_virtual(2*context.size());
    let distribution = Distribution::new(vec![5,5], topology, vec![first,second]);
    let mut tensor = SymmetricTensor::new(context,
        SymmetricDistribution::new(distribution, vec![Symmetry::SY,Symmetry::NS]),
        Arithmetic::<i64>::new());
    tensor.transform(|key, value| *value = key as i64 + 1);
    tensor.scale_indexed("ii", &3);
    tensor.transform_indexed("ii", |value| *value += 7);
    tensor.scale(&2);
    let keys: Vec<_> = (0..25).collect();
    let expected: Vec<_> = keys.iter().map(|&key| {
        let i = key % 5; let j = key / 5;
        let value = (i.min(j)+5*i.max(j)+1) as i64;
        if i == j { 2*(3*value+7) } else { 2*value }
    }).collect();
    assert_eq!(tensor.read(&keys), expected);
    let mut anti = tensor.repack_groups(vec![Symmetry::AS,Symmetry::NS]);
    let anti_expected: Vec<_> = keys.iter().map(|&key| {
        let i=key%5; let j=key/5;
        if i==j {0} else if i>j {-expected[key]} else {expected[key]}
    }).collect();
    assert_eq!(anti.read(&keys), anti_expected);
    let mut hollow = anti.repack_groups(vec![Symmetry::SH,Symmetry::NS]);
    hollow.scale_indexed("ii", &1000);
    hollow.transform_indexed("ii", |value| *value = 5000);
    assert_eq!(hollow.read(&keys), anti_expected.iter().map(|v| v.abs()).collect::<Vec<_>>());
    let symmetric = hollow.repack_groups(vec![Symmetry::SY,Symmetry::NS]);
    assert_eq!(symmetric.read(&[0,6,12,18,24]), vec![0;5]);
    // Rank contributions and symmetry-equivalent duplicate keys: beta once.
    anti.write_scaled(&[(5,3),(1,-4),(0,999)], &2, &3);
    assert_eq!(anti.read(&[5,1,0,10]),
        vec![36+14*context.size() as i64, -36-14*context.size() as i64, 0, 22]);
    let valid = anti.distribution().local_pairs(context.rank());
    for (offset, value) in anti.local_storage().iter().enumerate() {
        if !valid.iter().any(|&(slot,_)| slot==offset) { assert_eq!(*value,0); }
    }
}
fn main() {
    let runtime=Runtime::initialize(); let world=runtime.world();
    run(&world);
    let parity=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();
    run(&parity); parity.close();
    if world.rank()==0 {println!("DIGIT / PASS distributed_symmetric_operations: indexed scaling, group repack, scaled writes; exact i64; world+parity");}
    world.close(); runtime.finalize();
}
