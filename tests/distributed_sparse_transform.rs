//! Sparse stored-value mapping and dense-to-sparse accumulation.
//!
//! The fixture keeps the output sparse: only keys already stored in `B` are
//! visited, including an explicitly stored zero, while absent keys remain
//! implicit.
use ctf::{
    algebra::{Arithmetic, CustomMonoid, Wire},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    sparse::SparseTensor,
    tensor::Tensor,
};

#[derive(Clone, Debug, PartialEq)]
struct Pair {
    a: i64,
    b: i64,
}

impl Wire for Pair {
    const WIDTH: usize = 16;

    fn encode(&self, output: &mut Vec<u8>) {
        self.a.encode(output);
        self.b.encode(output);
    }

    fn decode(input: &[u8]) -> Self {
        Self {
            a: i64::decode(&input[..8]),
            b: i64::decode(&input[8..16]),
        }
    }
}

type PairMonoid = CustomMonoid<Pair, fn(&Pair, &Pair) -> Pair>;

fn pair_add(a: &Pair, b: &Pair) -> Pair {
    Pair {
        a: a.a + b.a,
        b: a.b + b.b,
    }
}

fn pair_monoid() -> PairMonoid {
    CustomMonoid {
        identity: Pair { a: 0, b: 0 },
        addition: pair_add,
    }
}

fn virtual2(context: &Context<'_>, shape: Vec<usize>) -> Distribution {
    let topology = Topology::new(vec![context.size()]);
    let mut first = Mapping::Unmapped;
    first.augment_physical(&topology, 0);
    first.augment_virtual(context.size() * 2);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    mappings[0] = first;
    Distribution::new(shape, topology, mappings)
}

fn switchdist(context: &Context<'_>, shape: Vec<usize>) -> Distribution {
    let topology = Topology::new(vec![context.size(), 1]);
    let mut mapped = Mapping::Unmapped;
    mapped.augment_physical(&topology, 0);
    mapped.augment_virtual(context.size() * 2);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    mappings[1] = mapped;
    Distribution::new(shape, topology, mappings)
}

fn sparse_i64<'c, 'r>(
    context: &'c Context<'r>,
    distribution: Distribution,
    entries: &[(usize, i64)],
) -> SparseTensor<'c, 'r, Arithmetic<i64>> {
    let mut tensor = SparseTensor::new(context, distribution.clone(), Arithmetic::new());
    let owned: Vec<_> = entries
        .iter()
        .copied()
        .filter(|(key, _)| distribution.owner(*key) == context.rank())
        .collect();
    tensor.write_add(&owned);
    tensor
}

fn sparse_pair<'c, 'r>(
    context: &'c Context<'r>,
    distribution: Distribution,
    entries: &[(usize, Pair)],
) -> SparseTensor<'c, 'r, PairMonoid> {
    let mut tensor = SparseTensor::new(context, distribution.clone(), pair_monoid());
    let owned: Vec<_> = entries
        .iter()
        .cloned()
        .filter(|(key, _)| distribution.owner(*key) == context.rank())
        .collect();
    tensor.write_add(&owned);
    tensor
}

fn dense_i64<'c, 'r>(context: &'c Context<'r>, values: &[i64]) -> Tensor<'c, 'r, Arithmetic<i64>> {
    let distribution = Distribution::cyclic(vec![values.len()], context.size());
    let entries: Vec<_> = values.iter().copied().enumerate().collect();
    let owned: Vec<_> = entries
        .into_iter()
        .filter(|(key, _)| distribution.owner(*key) == context.rank())
        .collect();
    let mut tensor = Tensor::new(context, distribution, Arithmetic::new());
    tensor.write_add(&owned);
    tensor
}

fn source_value(key: usize) -> i64 {
    match key {
        0 => 5,
        4 => 0,
        7 => -3,
        10 => 8,
        _ => 0,
    }
}

fn all_keys(len: usize) -> Vec<usize> {
    (0..len).collect()
}

fn map_and_accumulate(context: &Context<'_>) {
    let shape = vec![3, 2, 2];
    let distribution = virtual2(context, shape.clone());
    let source_entries = [(0, 5), (4, 2), (4, -2), (7, -3), (10, 8)];
    let source = sparse_i64(context, distribution.clone(), &source_entries);
    let keys = all_keys(distribution.global_len());

    let mut explicit_zero_copies = [source
        .local_pairs()
        .iter()
        .filter(|(key, value)| *key == 4 && *value == 0)
        .count() as f64];
    context.sum_f64(&mut explicit_zero_copies);
    assert!(explicit_zero_copies[0] > 0.);

    let mapped = source.map_stored(pair_monoid(), |value| Pair {
        a: *value,
        b: 2 * *value,
    });
    assert_eq!(mapped.distribution(), &distribution);
    assert_eq!(
        source
            .local_pairs()
            .into_iter()
            .map(|(key, _)| key)
            .collect::<Vec<_>>(),
        mapped
            .local_pairs()
            .into_iter()
            .map(|(key, _)| key)
            .collect::<Vec<_>>()
    );
    let expected_mapped: Vec<_> = keys
        .iter()
        .map(|&key| {
            let value = source_value(key);
            Pair {
                a: value,
                b: 2 * value,
            }
        })
        .collect();
    assert_eq!(mapped.read(&keys), expected_mapped);

    let mapped_back = mapped.map_stored(Arithmetic::<i64>::new(), |value| value.a + value.b);
    assert_eq!(
        mapped_back.read(&keys),
        keys.iter()
            .map(|&key| 3 * source_value(key))
            .collect::<Vec<_>>()
    );
    assert_eq!(mapped_back.local_nnz(), source.local_nnz());
    let mut mapped_zero_copies = [mapped_back
        .local_pairs()
        .iter()
        .filter(|(key, value)| *key == 4 && *value == 0)
        .count() as f64];
    context.sum_f64(&mut mapped_zero_copies);
    assert!(mapped_zero_copies[0] > 0.);

    let switch = switchdist(context, shape.clone());
    let mut accumulated = mapped;
    accumulated.redistribute(switch.clone());
    assert_eq!(accumulated.distribution(), &switch);

    let values_i = [10, -4, 7];
    let values_k = [3, -6];
    let dense_i = dense_i64(context, &values_i);
    let dense_k = dense_i64(context, &values_k);
    accumulated.accumulate_from_dense("ijk", &dense_i, "i", |value, output| {
        output.b += *value;
    });
    accumulated.accumulate_from_dense("ijk", &dense_k, "k", |value, output| {
        output.b += *value;
    });

    let expected: Vec<_> = keys
        .iter()
        .map(|&key| {
            let coordinates = switch.decode_key(key);
            let value = source_value(key);
            if matches!(key, 0 | 4 | 7 | 10) {
                Pair {
                    a: value,
                    b: 2 * value + values_i[coordinates[0]] + values_k[coordinates[2]],
                }
            } else {
                Pair { a: 0, b: 0 }
            }
        })
        .collect();
    assert_eq!(accumulated.read(&keys), expected);
}

fn repeated_output(context: &Context<'_>) {
    let shape = vec![3, 3, 2];
    let distribution = switchdist(context, shape.clone());
    let entries = [
        (0, Pair { a: 1, b: 10 }),
        (1, Pair { a: 2, b: 20 }),
        (3, Pair { a: 3, b: 30 }),
        (4, Pair { a: 4, b: 40 }),
        (8, Pair { a: 5, b: 50 }),
        (9, Pair { a: 6, b: 60 }),
        (13, Pair { a: 7, b: 70 }),
    ];
    let mut output = sparse_pair(context, distribution.clone(), &entries);
    let before = output.read(&(0..distribution.global_len()).collect::<Vec<_>>());
    let dense = dense_i64(context, &[7, 11, 13]);
    output.accumulate_from_dense("iij", &dense, "i", |value, pair| {
        pair.b += *value;
    });

    let expected: Vec<_> = before
        .into_iter()
        .enumerate()
        .map(|(key, mut pair)| {
            let coordinates = distribution.decode_key(key);
            if entries.iter().any(|(entry, _)| *entry == key) && coordinates[0] == coordinates[1] {
                pair.b += [7, 11, 13][coordinates[0]];
            }
            pair
        })
        .collect();
    assert_eq!(
        output.read(&(0..distribution.global_len()).collect::<Vec<_>>()),
        expected
    );
}

fn input_zero_semantics(context: &Context<'_>) {
    let shape = vec![3, 2, 2];
    let distribution = switchdist(context, shape.clone());
    let entries = [(0, Pair { a: 9, b: 2 }), (4, Pair { a: 0, b: 0 })];
    let mut output = sparse_pair(context, distribution.clone(), &entries);
    let stored = output.local_nnz();
    let dense = dense_i64(context, &[0, 0, 0]);
    let mut calls = 0usize;
    output.accumulate_from_dense("ijk", &dense, "i", |value, pair| {
        calls += 1;
        pair.b += *value + 1;
    });
    // The source sparsifies dense A before its sparse accumulator kernel.
    assert_eq!(calls, 0);
    assert_eq!(output.local_nnz(), stored);
    assert_eq!(
        output.read(&[0, 4, 1]),
        vec![
            Pair { a: 9, b: 2 },
            Pair { a: 0, b: 0 },
            Pair { a: 0, b: 0 },
        ]
    );
    let input = sparse_i64(
        context,
        Distribution::cyclic(vec![3], context.size()),
        &[(0, 0), (2, 7)],
    );
    output.accumulate_from_sparse("ijk", &input, "i", |value, pair| {
        calls += 1;
        pair.b += *value + 1;
    });
    let mut count = [calls as f64];
    context.sum_f64(&mut count);
    assert_eq!(count[0], 1.);
    assert_eq!(output.local_nnz(), stored);
    assert_eq!(
        output.read(&[0, 4, 2]),
        vec![
            Pair { a: 9, b: 3 },
            Pair { a: 0, b: 0 },
            Pair { a: 0, b: 0 },
        ]
    );
}

fn run(context: &Context<'_>) {
    map_and_accumulate(context);
    repeated_output(context);
    input_zero_semantics(context);
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
            "DIGIT / PASS distributed_sparse_transform: stored map, dense accumulation, diagonals, explicit zeros; world+parity"
        );
    }
    world.close();
    drop(universe);
}
