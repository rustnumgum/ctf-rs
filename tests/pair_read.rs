use ctf::{
    algebra::{Arithmetic, Complex},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    sparse::SparseTensor,
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, AS, NS, SH, SY},
    tensor::Tensor,
};

fn virtual_distribution(context: &Context<'_>, shape: &[usize]) -> Distribution {
    let topology = Topology::new(vec![context.size()]);
    let mut first = Mapping::Unmapped;
    first.augment_physical(&topology, 0);
    first.augment_virtual(2 * context.size());
    let mut second = Mapping::Unmapped;
    second.augment_virtual(2 * context.size());
    Distribution::new(shape.to_vec(), topology, vec![first, second])
}

fn replicated_distribution(context: &Context<'_>, shape: &[usize]) -> Distribution {
    let topology = Topology::new(vec![context.size()]);
    let mut first = Mapping::Unmapped;
    first.augment_virtual(context.size());
    Distribution::new(shape.to_vec(), topology, vec![first, Mapping::Unmapped])
}

fn dense_i64(context: &Context<'_>, distribution: Distribution) {
    let mut tensor = Tensor::new(context, distribution.clone(), Arithmetic::<i64>::new());
    tensor.transform(|key, value| *value = key as i64 - 2);
    let expected: Vec<_> = (0..distribution.global_len())
        .map(|key| (key, key as i64 - 2))
        .collect();
    let expected_nonzero: Vec<_> = expected
        .iter()
        .copied()
        .filter(|(_, value)| *value != 0)
        .collect();
    assert_eq!(tensor.all_pairs(false), expected);
    assert_eq!(tensor.all_pairs(true), expected_nonzero);
    assert_eq!(
        tensor.all_data(),
        expected.iter().map(|(_, value)| *value).collect::<Vec<_>>()
    );
}

fn dense_complex(context: &Context<'_>, distribution: Distribution) {
    let mut tensor = Tensor::new(
        context,
        distribution.clone(),
        Arithmetic::<Complex<f64>>::new(),
    );
    tensor.transform(|key, value| {
        *value = Complex::new(key as f64 + 0.5, -(key as f64) - 1.25);
    });
    let expected: Vec<_> = (0..distribution.global_len())
        .map(|key| (key, Complex::new(key as f64 + 0.5, -(key as f64) - 1.25)))
        .collect();
    assert_eq!(tensor.all_pairs(false), expected);
    assert_eq!(
        tensor.all_data(),
        expected.iter().map(|(_, value)| *value).collect::<Vec<_>>()
    );
}

fn dense_empty_and_shard(context: &Context<'_>) {
    let tiny_distribution = Distribution::cyclic(vec![1], context.size());
    let mut tiny = Tensor::new(context, tiny_distribution.clone(), Arithmetic::<i64>::new());
    tiny.transform(|_, value| *value = 7);
    assert_eq!(tiny.all_pairs(false), vec![(0, 7)]);
    assert_eq!(tiny.all_data(), vec![7]);
    if context.rank() != tiny_distribution.owner(0) {
        assert!(tiny.local_pairs().is_empty());
    }

    let empty = Tensor::new(
        context,
        Distribution::cyclic(vec![0, 2], context.size()),
        Arithmetic::<i64>::new(),
    );
    assert!(empty.all_pairs(false).is_empty());
    assert!(empty.all_pairs(true).is_empty());
    assert!(empty.all_data().is_empty());
    let zeros = Tensor::new(
        context,
        Distribution::cyclic(vec![2], context.size()),
        Arithmetic::<i64>::new(),
    );
    assert!(zeros.all_pairs(true).is_empty());
    let sparse_empty = SparseTensor::new(
        context,
        Distribution::cyclic(vec![2], context.size()),
        Arithmetic::<i64>::new(),
    );
    assert!(sparse_empty.all_pairs(true).is_empty());
    assert_eq!(sparse_empty.all_data(), vec![0, 0]);
}

fn sparse_case(context: &Context<'_>, token: u64, scope: usize) {
    let distribution = Distribution::cyclic(vec![3, 2], context.size());
    let entries = [(0, 0_i64), (2, 5), (5, -3)];
    let mut sparse = SparseTensor::new(context, distribution.clone(), Arithmetic::<i64>::new());
    let owned: Vec<_> = entries
        .iter()
        .copied()
        .filter(|(key, _)| distribution.owner(*key) == context.rank())
        .collect();
    sparse.write_add(&owned);
    assert_eq!(sparse.all_pairs(true), entries);
    let expected: Vec<_> = (0..6)
        .map(|key| {
            let value = entries
                .iter()
                .find_map(|(entry, value)| (*entry == key).then_some(*value))
                .unwrap_or(0);
            (key, value)
        })
        .collect();
    assert_eq!(sparse.all_pairs(false), expected);
    assert_eq!(
        sparse.all_data(),
        expected.iter().map(|(_, value)| *value).collect::<Vec<_>>()
    );

    let mut dense = Tensor::new(context, distribution.clone(), Arithmetic::<i64>::new());
    dense.transform(|key, value| *value = if key % 2 == 0 { 0 } else { key as i64 });
    let sparse_with_zeros = dense.into_sparse(|_| true);
    let path = std::env::temp_dir().join(format!("ctf-pair-read-{token}-{scope}.txt"));
    sparse_with_zeros.write_sparse_to_file(&path, true, false);
    context.barrier();
    let mut restored = SparseTensor::new(context, distribution, Arithmetic::<i64>::new());
    restored.read_sparse_from_file(&path, true, false);
    let expected_file: Vec<_> = (0..6)
        .map(|key| (key, if key % 2 == 0 { 0 } else { key as i64 }))
        .collect();
    assert_eq!(restored.all_pairs(true), expected_file);
    context.barrier();
    if context.rank() == 0 {
        std::fs::remove_file(path).unwrap();
    }
    context.barrier();
}

fn symmetric_distribution(context: &Context<'_>, kind: Symmetry) -> SymmetricDistribution {
    let topology = Topology::new(vec![context.size()]);
    let mut first = Mapping::Unmapped;
    first.augment_physical(&topology, 0);
    first.augment_virtual(2 * context.size());
    let mut second = Mapping::Unmapped;
    second.augment_virtual(2 * context.size());
    SymmetricDistribution::new(
        Distribution::new(vec![3, 3], topology, vec![first, second]),
        vec![kind, NS],
    )
}

fn symmetric_case(context: &Context<'_>) {
    for kind in [SY, AS, SH] {
        let distribution = symmetric_distribution(context, kind);
        let mut tensor =
            SymmetricTensor::new(context, distribution.clone(), Arithmetic::<i64>::new());
        tensor.transform(|key, value| *value = if key == 3 { 0 } else { key as i64 + 1 });

        let packed: Vec<_> = (0..9)
            .filter_map(|key| {
                distribution
                    .canonicalize(key)
                    .filter(|&(canonical, sign)| canonical == key && sign == 1)
                    .map(|_| (key, if key == 3 { 0 } else { key as i64 + 1 }))
            })
            .collect();
        let nonzero: Vec<_> = packed
            .iter()
            .copied()
            .filter(|(_, value)| *value != 0)
            .collect();
        assert_eq!(tensor.all_pairs(false, false), packed);
        assert_eq!(
            tensor.all_data(false),
            packed.iter().map(|(_, value)| *value).collect::<Vec<_>>()
        );
        assert_eq!(tensor.all_pairs(true, false), nonzero);
        assert_eq!(tensor.all_pairs(true, true), nonzero);

        let full: Vec<_> = (0..9)
            .map(|key| {
                let value = distribution
                    .canonicalize(key)
                    .map(|(canonical, sign)| {
                        (if canonical == 3 {
                            0
                        } else {
                            canonical as i64 + 1
                        }) * sign as i64
                    })
                    .unwrap_or(0);
                (key, value)
            })
            .collect();
        assert_eq!(tensor.all_pairs(false, true), full);
        assert_eq!(
            tensor.all_data(true),
            full.iter().map(|(_, value)| *value).collect::<Vec<_>>()
        );
    }
}

fn run(context: &Context<'_>, token: u64, scope: usize) {
    dense_i64(context, virtual_distribution(context, &[3, 2]));
    dense_i64(context, replicated_distribution(context, &[3, 2]));
    dense_complex(context, replicated_distribution(context, &[3, 2]));
    dense_empty_and_shard(context);
    sparse_case(context, token, scope);
    symmetric_case(context);
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let mut token = [std::process::id() as u64];
    world.broadcast(0, &mut token);
    run(&world, token[0], 0);

    let color = world.rank() % 2;
    let parity = world
        .split(Some(color as i32), world.rank() as i32)
        .unwrap();
    run(&parity, token[0], color + 1);
    parity.close();

    if world.rank() == 0 {
        println!(
            "DIGIT / PASS pair_read: dense/sparse/symmetric all-pairs/data, explicit zeros, virtual/replicated layouts, packed/full symmetry, empty shards/dimensions, i64/complex; world+parity"
        );
    }
    world.close();
    drop(universe);
}
