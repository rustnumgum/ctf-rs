use ctf::{
    algebra::{Arithmetic, CustomMonoid},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    sparse::SparseTensor,
};

fn arithmetic_io(context: &Context<'_>) {
    let size = context.size() as i64;
    let rank = context.rank() as i64;
    let rank_sum = size * (size - 1) / 2;
    let mut tensor = SparseTensor::new(
        context,
        Distribution::cyclic(vec![5, 7], context.size()),
        Arithmetic::<i64>::new(),
    );
    tensor.write_add(&[(0, rank + 2), (0, 5), (6, rank + 10)]);
    tensor.transform_stored(|key, value| *value += key as i64);

    let key_zero = rank_sum + 7 * size;
    let key_six = rank_sum + 10 * size + 6;
    assert_eq!(
        tensor.read(&[6, 4, 0, 6, 0, 34]),
        vec![key_six, 0, key_zero, key_six, key_zero, 0]
    );
    assert_eq!(tensor.reduce(), key_zero + key_six);

    let topology = Topology::new(vec![context.size(), 1]);
    let mut mapped_axis = Mapping::Unmapped;
    mapped_axis.augment_physical(&topology, 0);
    mapped_axis.augment_virtual(context.size() * 2);
    let mapped = Distribution::new(
        vec![5, 7],
        topology.clone(),
        vec![Mapping::Unmapped, mapped_axis],
    );
    tensor.redistribute(mapped.clone());
    assert_eq!(tensor.distribution(), &mapped);
    assert_eq!(tensor.read(&[34, 0, 6]), vec![0, key_zero, key_six]);
    assert_eq!(tensor.reduce(), key_zero + key_six);

    let replicated = Distribution::new(vec![5, 7], topology, vec![Mapping::Unmapped; 2]);
    tensor.redistribute(replicated.clone());
    assert_eq!(tensor.distribution(), &replicated);
    assert_eq!(tensor.read(&[0, 6, 4]), vec![key_zero, key_six, 0]);
    assert_eq!(tensor.reduce(), key_zero + key_six);

    let cyclic = Distribution::cyclic(vec![5, 7], context.size());
    tensor.redistribute(cyclic.clone());
    assert_eq!(tensor.distribution(), &cyclic);
    assert_eq!(tensor.reduce(), key_zero + key_six);

    let mut explicit_zero = SparseTensor::new(
        context,
        Distribution::cyclic(vec![5, 7], context.size()),
        Arithmetic::<i64>::new(),
    );
    let zero_pairs: &[(usize, i64)] = if context.rank() == 0 {
        &[(0, 1), (0, -1)]
    } else {
        &[]
    };
    explicit_zero.write_add(zero_pairs);
    if context.rank() == 0 {
        assert_eq!(explicit_zero.local_pairs(), vec![(0, 0)]);
    } else {
        assert!(explicit_zero.local_pairs().is_empty());
    }
    explicit_zero.sparsify(|value| *value != 0);
    assert!(explicit_zero.local_pairs().is_empty());

    let mut scaled = SparseTensor::new(
        context,
        Distribution::cyclic(vec![5, 7], context.size()),
        Arithmetic::<i64>::new(),
    );
    let old_pairs: &[(usize, i64)] = if context.rank() == 0 {
        &[(5, 10), (7, 11)]
    } else {
        &[]
    };
    scaled.write_add(old_pairs);
    let updates = vec![(5, rank + 1), (5, 2), (6, 4)];
    scaled.write_scaled(&updates, &3, &2);
    let scaled_old = 20 + 3 * (rank_sum + 3 * size);
    let scaled_new = 12 * size;
    assert_eq!(
        scaled.read(&[7, 5, 5, 6, 4]),
        vec![11, scaled_old, scaled_old, scaled_new, 0]
    );
    scaled.scale(&2);
    assert_eq!(
        scaled.read(&[7, 5, 6, 4]),
        vec![22, 2 * scaled_old, 2 * scaled_new, 0]
    );

    let far_key = 1_000_000usize * 999_999 + 999_999;
    let mut huge = SparseTensor::new(
        context,
        Distribution::cyclic(vec![1_000_000, 1_000_000], context.size()),
        Arithmetic::<i64>::new(),
    );
    huge.write_add(&[(0, rank + 1), (far_key, 7)]);
    assert!(huge.local_nnz() <= 2);
    let first = rank_sum + size;
    let far = 7 * size;
    assert_eq!(
        huge.read(&[far_key, 12345, 0, far_key]),
        vec![far, 0, first, far]
    );
    assert_eq!(huge.reduce(), first + far);
}

fn analytic_views(context: &Context<'_>) {
    let distribution = Distribution::cyclic(vec![5, 7], context.size());
    let mut source = SparseTensor::new(context, distribution.clone(), Arithmetic::<i64>::new());
    let pairs: Vec<_> = (0..distribution.global_len())
        .filter(|&key| distribution.owner(key) == context.rank())
        .map(|key| {
            let coordinates = distribution.decode_key(key);
            (key, (100 * coordinates[1] + coordinates[0] + 1) as i64)
        })
        .collect();
    source.write_add(&pairs);
    assert_eq!(source.reduce(), 10605);

    let sliced = source.slice(&[1..4, 2..7]);
    assert_eq!(sliced.distribution().shape, vec![3, 5]);
    for (key, value) in sliced.local_pairs() {
        let coordinates = sliced.distribution().decode_key(key);
        let expected = 100 * (coordinates[1] + 2) + (coordinates[0] + 1) + 1;
        assert_eq!(value, expected as i64);
    }
    assert_eq!(sliced.read(&[0, 3, 0]), vec![202, 302, 202]);

    let transposed = source.permute_axes(&[1, 0]);
    assert_eq!(transposed.distribution().shape, vec![7, 5]);
    for (key, value) in transposed.local_pairs() {
        let coordinates = transposed.distribution().decode_key(key);
        assert_eq!(value, (100 * coordinates[0] + coordinates[1] + 1) as i64);
    }
    assert_eq!(transposed.read(&[7, 0, 7]), vec![2, 1, 2]);
    assert_eq!(transposed.reduce(), 10605);
}

fn max_monoid_io(context: &Context<'_>) {
    let mut tensor = SparseTensor::new(
        context,
        Distribution::cyclic(vec![5, 7], context.size()),
        CustomMonoid {
            identity: i64::MIN,
            addition: |a: &i64, b: &i64| (*a).max(*b),
        },
    );
    let rank = context.rank() as i64;
    tensor.write_add(&[(3, rank), (3, 2 * rank + 1), (8, 100 - rank)]);
    let expected_duplicate = 2 * (context.size() as i64 - 1) + 1;
    assert_eq!(
        tensor.read(&[4, 3, 3, 8, 34]),
        vec![
            i64::MIN,
            expected_duplicate,
            expected_duplicate,
            100,
            i64::MIN
        ]
    );
    assert_eq!(tensor.reduce(), 100);
}

fn run(context: &Context<'_>) {
    arithmetic_io(context);
    analytic_views(context);
    max_monoid_io(context);
    // Source sp_write combines the first new value before an existing value.
    let mut first = SparseTensor::new(
        context,
        Distribution::cyclic(vec![1], context.size()),
        CustomMonoid {
            identity: 0i64,
            addition: |a: &i64, b: &i64| if *a != 0 { *a } else { *b },
        },
    );
    first.write_add(if context.rank() == 0 { &[(0, 9)] } else { &[] });
    first.write_add(if context.rank() == 0 { &[(0, 3)] } else { &[] });
    assert_eq!(first.read(&[0]), vec![3]);
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let world_rank = world.rank();
    run(&world);

    let parity = world
        .split(Some((world_rank % 2) as i32), world_rank as i32)
        .unwrap();
    run(&parity);
    parity.close();

    if world_rank == 0 {
        println!(
            "DIGIT / PASS distributed_sparse_io: sparse indexed I/O, redistribution, views, custom identity; world+parity"
        );
    }
    world.close();
    drop(universe);
}
