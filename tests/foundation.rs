//! Exact foundation acceptance. These cases do not stand in for the full upstream suite.
use ctf::{algebra::{Arithmetic, CustomMonoid, CustomSemiring, Monoid, Semiring},
    context::{Context, Runtime}, mapping::{Distribution, Mapping, Topology}, tensor::Tensor};

fn layouts() {
    let topology = Topology::new(vec![6, 4]);
    let expected = [0,1,2,6,7,8,3,4,5,9,10,11,12,13,14,18,19,20,15,16,17,21,22,23];
    for (rank, &mapped) in expected.iter().enumerate() {
        assert_eq!(topology.reorder_rank(&[3,2], rank), mapped);
        assert_eq!(topology.inverse_reorder_rank(&[3,2], mapped), rank);
    }
    let topology = Topology::new(vec![2]);
    let mut first = Mapping::Unmapped;
    first.augment_physical(&topology, 0);
    first.augment_virtual(6);
    assert_eq!(first.phase(), 6);
    assert_eq!(first.physical_phase(), 2);
    let d = Distribution::new(vec![7,3], topology, vec![first, Mapping::Unmapped]);
    assert_eq!(d.block_shape(), vec![2,3]);
    assert_eq!(d.local_len(), 18);
    assert_eq!(d.local_offset(0, 0), 0);
    assert_eq!(d.local_offset(0, 6), 1);
    assert_eq!(d.local_offset(0, 7), 2);
    assert_eq!(d.local_offset(0, 2), 6);
    assert_eq!(d.local_offset(0, 4), 12);
    for rank in 0..2 {
        let mut seen = Vec::new();
        for offset in 0..d.local_len() {
            if let Some(key) = d.global_key(rank, offset) {
                assert_eq!(d.owner(key), rank);
                assert_eq!(d.local_offset(rank, key), offset);
                seen.push(key);
            }
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..21).filter(|key| (key%7)%2 == rank).collect::<Vec<_>>());
    }
    assert_eq!(Distribution::cyclic(vec![7,0,7], 4).local_len(), 0);
    let scalar = Distribution::cyclic(vec![], 4);
    assert_eq!(scalar.global_len(), 1);
    assert_eq!(scalar.owner(0), 0);
    assert_eq!(scalar.global_key(0,0), Some(0));
}

fn distributed(world: &Context<'_>) {
    let rank = world.rank(); let np = world.size();
    let mut tensor = Tensor::new(world, Distribution::cyclic(vec![7,3], np), Arithmetic::<i64>::new());
    let pairs: Vec<_> = (0..21).filter(|key| key%np == rank).map(|key| (key, key as i64+1)).collect();
    tensor.write_add(&pairs);
    let requests: Vec<_> = (0..21).rev().filter(|key| (key+1)%np == rank).collect();
    assert_eq!(tensor.read(&requests), requests.iter().map(|&k| k as i64+1).collect::<Vec<_>>());
    assert_eq!(tensor.reduce(), 231);
    tensor.write_add(&[(0,1), (0,2)]);
    assert_eq!(tensor.read(&[0]), vec![1+3*np as i64]);
    let topology = Topology::new(vec![np]);
    let mut columns = Mapping::Unmapped;
    columns.augment_physical(&topology, 0);
    columns.augment_virtual(np*2);
    tensor.redistribute(Distribution::new(vec![7,3], topology.clone(), vec![Mapping::Unmapped, columns]));
    for (key, value) in tensor.local_pairs() {
        assert_eq!(value, key as i64+1+if key == 0 {3*np as i64} else {0});
    }
    tensor.redistribute(Distribution::new(vec![7,3], topology, vec![Mapping::Unmapped;2]));
    assert_eq!(tensor.local_pairs().len(),21);
    assert_eq!(tensor.reduce(), 231+3*np as i64);
    tensor.redistribute(Distribution::cyclic(vec![7,3], np));
    assert_eq!(tensor.reduce(), 231+3*np as i64);
    let mut empty = Tensor::new(world, Distribution::cyclic(vec![7,0,7], np), Arithmetic::<i64>::new());
    empty.write_add(&[]);
    assert!(empty.read(&[]).is_empty());
    assert_eq!(empty.reduce(),0);
    let mut small = Tensor::new(world, Distribution::cyclic(vec![1], np), Arithmetic::<i64>::new());
    small.write_add(if rank == 0 { &[(0,17)] } else { &[] });
    if rank != 0 { assert!(small.local_pairs().is_empty()); }
    assert_eq!(small.read(&[0]),vec![17]);
    let mut scalar = Tensor::new(world, Distribution::cyclic(vec![], np), Arithmetic::<i64>::new());
    scalar.write_add(if rank == 0 { &[(0,42)] } else { &[] });
    assert_eq!(scalar.reduce(),42);
    let algebra = CustomSemiring {
        monoid: CustomMonoid { identity: false, addition: |a: &bool, b: &bool| *a || *b },
        identity: true, multiplication: |a: &bool, b: &bool| *a && *b,
    };
    assert!(algebra.multiply(&algebra.one(), &true));
    assert!(!algebra.zero());
    assert!(world.all_reduce(&algebra, &(rank == np-1)));
    let child = world.split(Some((rank%2) as i32), -(rank as i32)).unwrap();
    let expected_size = (np + 1 - rank%2)/2;
    assert_eq!(child.size(),expected_size);
    assert_eq!(child.all_reduce(&Arithmetic::<i64>::new(), &1),expected_size as i64);
    child.close();
    let excluded = world.split((rank == 0).then_some(0),rank as i32);
    if let Some(child) = excluded { assert_eq!(child.size(),1); child.close(); }
    let shared = world.split_shared();
    assert!(shared.size() <= np);
    shared.close();
    let fiber = Topology::new(vec![np]).fiber(world,0);
    assert_eq!(fiber.size(),np); fiber.close();
}

fn main() {
    layouts();
    let runtime = Runtime::initialize();
    let world = runtime.world();
    distributed(&world);
    if world.rank() == 0 { println!("DIGIT / PASS foundation: exact layouts, algebra, read/write, redistribution, empty shards, subcommunicators; ranks={}",world.size()); }
    world.close(); runtime.finalize();
}
