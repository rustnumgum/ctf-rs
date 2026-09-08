//! Exact distributed sparse contractions and diagonal projection coverage.
//!
//! Repeated labels are represented by diagonal projections before the existing
//! sparse fold path runs.  The fixtures deliberately store large off-diagonal
//! values so an accidental use of those entries cannot pass the small integer
//! oracles.
use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    sparse::SparseTensor,
    tensor::Tensor,
};

type Algebra = Arithmetic<i64>;

fn grid(context: &Context<'_>) -> [usize; 2] {
    if context.size() == 4 {
        [2, 2]
    } else {
        [context.size(), 1]
    }
}

fn cyclic(context: &Context<'_>, shape: Vec<usize>) -> Distribution {
    Distribution::cyclic(shape, context.size())
}

fn virtual2(context: &Context<'_>, shape: Vec<usize>) -> Distribution {
    let topology = Topology::new(vec![context.size()]);
    let mut first = Mapping::Unmapped;
    first.augment_physical(&topology, 0);
    first.augment_virtual(context.size() * 2);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    if !mappings.is_empty() {
        mappings[0] = first;
    }
    Distribution::new(shape, topology, mappings)
}

fn entries_for(distribution: &Distribution, value: impl Fn(&[usize]) -> i64) -> Vec<(usize, i64)> {
    (0..distribution.global_len())
        .filter_map(|key| {
            let value = value(&distribution.decode_key(key));
            (value != 0).then_some((key, value))
        })
        .collect()
}

fn sparse<'c, 'r>(
    context: &'c Context<'r>,
    distribution: Distribution,
    entries: &[(usize, i64)],
) -> SparseTensor<'c, 'r, Algebra> {
    let owned: Vec<_> = entries
        .iter()
        .copied()
        .filter(|(key, _)| distribution.owner(*key) == context.rank())
        .collect();
    let mut tensor = SparseTensor::new(context, distribution, Algebra::new());
    tensor.write_add(&owned);
    tensor
}

fn sparse_fn<'c, 'r>(
    context: &'c Context<'r>,
    distribution: Distribution,
    value: impl Fn(&[usize]) -> i64,
) -> SparseTensor<'c, 'r, Algebra> {
    let entries = entries_for(&distribution, value);
    sparse(context, distribution, &entries)
}

fn dense<'c, 'r>(
    context: &'c Context<'r>,
    distribution: Distribution,
    entries: &[(usize, i64)],
) -> Tensor<'c, 'r, Algebra> {
    let mut tensor = Tensor::new(context, distribution, Algebra::new());
    tensor.transform(|key, value| {
        *value = entries
            .iter()
            .find_map(|&(entry, value)| (entry == key).then_some(value))
            .unwrap_or(0);
    });
    tensor
}

fn dense_fn<'c, 'r>(
    context: &'c Context<'r>,
    distribution: Distribution,
    value: impl Fn(&[usize]) -> i64,
) -> Tensor<'c, 'r, Algebra> {
    let entries = entries_for(&distribution, value);
    dense(context, distribution, &entries)
}

fn all_keys(distribution: &Distribution) -> Vec<usize> {
    (0..distribution.global_len()).collect()
}

fn a_repeated_value(coordinates: &[usize]) -> i64 {
    let i = coordinates[0];
    let j = coordinates[1];
    let k = coordinates[2];
    if i != j {
        10_000 + 100 * i as i64 + 10 * j as i64 + k as i64
    } else {
        match (i, k) {
            (0, 0) => 2,
            (1, 1) => -3,
            (2, 0) => 5,
            _ => 0,
        }
    }
}

fn b_unique_value(coordinates: &[usize]) -> i64 {
    match (coordinates[0], coordinates[1]) {
        (0, 0) => 3,
        (0, 2) => -4,
        (1, 1) => 5,
        (1, 3) => 7,
        _ => 0,
    }
}

fn product_from_repeated_a(i: usize, j: usize) -> i64 {
    (0..2)
        .map(|k| {
            let a = a_repeated_value(&[i, i, k]);
            a * b_unique_value(&[k, j])
        })
        .sum()
}

fn old_unique(coordinates: &[usize]) -> i64 {
    200 + 10 * coordinates[0] as i64 + coordinates[1] as i64
}

fn expected_unique() -> Vec<i64> {
    (0..12)
        .map(|key| {
            let i = key % 3;
            let j = key / 3;
            2 * product_from_repeated_a(i, j) + 3 * old_unique(&[i, j])
        })
        .collect()
}

fn repeated_a_unique_output(context: &Context<'_>) {
    let g = grid(context);
    let a_distribution = cyclic(context, vec![3, 3, 2]);
    let b_distribution = cyclic(context, vec![2, 4]);
    let output_distribution = virtual2(context, vec![3, 4]);
    let a_sparse = sparse_fn(context, a_distribution.clone(), a_repeated_value);
    let a_dense = dense_fn(context, a_distribution.clone(), a_repeated_value);
    let b_sparse = sparse_fn(context, b_distribution.clone(), b_unique_value);
    let b_dense = dense_fn(context, b_distribution.clone(), b_unique_value);
    let expected = expected_unique();
    let output_keys = all_keys(&output_distribution);

    let mut sparse_output = sparse_fn(context, output_distribution.clone(), old_unique);
    sparse_output
        .contract_from("ij", &a_sparse, "iik", &b_sparse, "kj", g, 2, 3)
        .unwrap();
    assert_eq!(sparse_output.distribution(), &output_distribution);
    assert_eq!(sparse_output.read(&output_keys), expected);

    let mut dense_output = dense_fn(context, output_distribution.clone(), old_unique);
    dense_output
        .contract_from_sparse("ij", &a_sparse, "iik", &b_sparse, "kj", g, 2, 3)
        .unwrap();
    assert_eq!(dense_output.distribution(), &output_distribution);
    assert_eq!(dense_output.read(&output_keys), expected);

    let mut sparse_a_dense_b = dense_fn(context, output_distribution.clone(), old_unique);
    sparse_a_dense_b
        .contract_from_sparse_dense("ij", &a_sparse, "iik", &b_dense, "kj", g, 2, 3)
        .unwrap();
    assert_eq!(sparse_a_dense_b.read(&output_keys), expected);

    let mut dense_a_sparse_b = dense_fn(context, output_distribution.clone(), old_unique);
    dense_a_sparse_b
        .contract_from_dense_sparse("ij", &a_dense, "iik", &b_sparse, "kj", g, 2, 3)
        .unwrap();
    assert_eq!(dense_a_sparse_b.read(&output_keys), expected);

    assert_eq!(a_sparse.distribution(), &a_distribution);
    assert_eq!(a_dense.distribution(), &a_distribution);
    assert_eq!(b_sparse.distribution(), &b_distribution);
    assert_eq!(b_dense.distribution(), &b_distribution);
}

fn old_repeated_output(coordinates: &[usize]) -> i64 {
    1_000 + 100 * coordinates[0] as i64 + 10 * coordinates[1] as i64 + coordinates[2] as i64
}

fn expected_repeated_output() -> Vec<i64> {
    (0..36)
        .map(|key| {
            let i0 = key % 3;
            let i1 = (key / 3) % 3;
            let j = key / 9;
            let old = old_repeated_output(&[i0, i1, j]);
            if i0 == i1 {
                2 * product_from_repeated_a(i0, j) + 3 * old
            } else {
                old
            }
        })
        .collect()
}

fn repeated_a_repeated_output(context: &Context<'_>) {
    let g = grid(context);
    let a_distribution = cyclic(context, vec![3, 3, 2]);
    let b_distribution = cyclic(context, vec![2, 4]);
    let output_distribution = virtual2(context, vec![3, 3, 4]);
    let a_sparse = sparse_fn(context, a_distribution.clone(), a_repeated_value);
    let a_dense = dense_fn(context, a_distribution.clone(), a_repeated_value);
    let b_sparse = sparse_fn(context, b_distribution.clone(), b_unique_value);
    let b_dense = dense_fn(context, b_distribution.clone(), b_unique_value);
    let expected = expected_repeated_output();
    let output_keys = all_keys(&output_distribution);

    let mut sparse_output = sparse_fn(context, output_distribution.clone(), old_repeated_output);
    sparse_output
        .contract_from("iij", &a_sparse, "iik", &b_sparse, "kj", g, 2, 3)
        .unwrap();
    assert_eq!(sparse_output.read(&output_keys), expected);

    let mut dense_output = dense_fn(context, output_distribution.clone(), old_repeated_output);
    dense_output
        .contract_from_sparse("iij", &a_sparse, "iik", &b_sparse, "kj", g, 2, 3)
        .unwrap();
    assert_eq!(dense_output.read(&output_keys), expected);

    let mut sparse_a_dense_b = dense_fn(context, output_distribution.clone(), old_repeated_output);
    sparse_a_dense_b
        .contract_from_sparse_dense("iij", &a_sparse, "iik", &b_dense, "kj", g, 2, 3)
        .unwrap();
    assert_eq!(sparse_a_dense_b.read(&output_keys), expected);

    let mut dense_a_sparse_b = dense_fn(context, output_distribution.clone(), old_repeated_output);
    dense_a_sparse_b
        .contract_from_dense_sparse("iij", &a_dense, "iik", &b_sparse, "kj", g, 2, 3)
        .unwrap();
    assert_eq!(dense_a_sparse_b.read(&output_keys), expected);
    assert_eq!(sparse_output.distribution(), &output_distribution);
    assert_eq!(dense_output.distribution(), &output_distribution);
}

fn a_unique_for_repeated_b(coordinates: &[usize]) -> i64 {
    2 + 3 * coordinates[0] as i64 - 2 * coordinates[1] as i64
}

fn b_repeated_value(coordinates: &[usize]) -> i64 {
    let k = coordinates[0];
    let j0 = coordinates[1];
    let j1 = coordinates[2];
    if j0 != j1 {
        20_000 + 100 * k as i64 + 10 * j0 as i64 + j1 as i64
    } else {
        match (k, j0) {
            (0, 0) => 3,
            (0, 2) => -4,
            (1, 1) => 5,
            (1, 3) => 7,
            _ => 0,
        }
    }
}

fn product_from_repeated_b(i: usize, j: usize) -> i64 {
    (0..2)
        .map(|k| a_unique_for_repeated_b(&[i, k]) * b_repeated_value(&[k, j, j]))
        .sum()
}

fn expected_repeated_b() -> Vec<i64> {
    (0..12)
        .map(|key| {
            let i = key % 3;
            let j = key / 3;
            2 * product_from_repeated_b(i, j) + 3 * old_unique(&[i, j])
        })
        .collect()
}

fn repeated_b_unique_output(context: &Context<'_>) {
    let g = grid(context);
    let a_distribution = cyclic(context, vec![3, 2]);
    let b_distribution = cyclic(context, vec![2, 4, 4]);
    let output_distribution = virtual2(context, vec![3, 4]);
    let a_sparse = sparse_fn(context, a_distribution.clone(), a_unique_for_repeated_b);
    let a_dense = dense_fn(context, a_distribution.clone(), a_unique_for_repeated_b);
    let b_sparse = sparse_fn(context, b_distribution.clone(), b_repeated_value);
    let b_dense = dense_fn(context, b_distribution.clone(), b_repeated_value);
    let expected = expected_repeated_b();
    let output_keys = all_keys(&output_distribution);

    let mut sparse_output = sparse_fn(context, output_distribution.clone(), old_unique);
    sparse_output
        .contract_from("ij", &a_sparse, "ik", &b_sparse, "kjj", g, 2, 3)
        .unwrap();
    assert_eq!(sparse_output.read(&output_keys), expected);

    let mut dense_output = dense_fn(context, output_distribution.clone(), old_unique);
    dense_output
        .contract_from_sparse("ij", &a_sparse, "ik", &b_sparse, "kjj", g, 2, 3)
        .unwrap();
    assert_eq!(dense_output.read(&output_keys), expected);

    let mut sparse_a_dense_b = dense_fn(context, output_distribution.clone(), old_unique);
    sparse_a_dense_b
        .contract_from_sparse_dense("ij", &a_sparse, "ik", &b_dense, "kjj", g, 2, 3)
        .unwrap();
    assert_eq!(sparse_a_dense_b.read(&output_keys), expected);

    let mut dense_a_sparse_b = dense_fn(context, output_distribution.clone(), old_unique);
    dense_a_sparse_b
        .contract_from_dense_sparse("ij", &a_dense, "ik", &b_sparse, "kjj", g, 2, 3)
        .unwrap();
    assert_eq!(dense_a_sparse_b.read(&output_keys), expected);
}

fn expected_repeated_a_diagonal() -> Vec<i64> {
    (0..6)
        .map(|key| {
            let i = key % 3;
            let k = key / 3;
            a_repeated_value(&[i, i, k])
        })
        .collect()
}

fn diagonal_source_value(coordinates: &[usize]) -> i64 {
    a_repeated_value(coordinates)
}

fn expected_without_iik_diagonal(diagonal: impl Fn(usize, usize) -> i64) -> Vec<i64> {
    (0..18)
        .map(|key| {
            let i = key % 3;
            let j = (key / 3) % 3;
            let k = key / 9;
            if i == j {
                diagonal(i, k)
            } else {
                diagonal_source_value(&[i, j, k])
            }
        })
        .collect()
}

fn sparse_iik_extract_replace(context: &Context<'_>) {
    let distribution = virtual2(context, vec![3, 3, 2]);
    let source = sparse_fn(context, distribution.clone(), diagonal_source_value);
    let (diagonal, labels) = source.extract_diagonal("iik");
    assert_eq!(labels, "ik");
    assert_eq!(diagonal.distribution().shape, vec![3, 2]);
    assert_eq!(
        diagonal.read(&all_keys(diagonal.distribution())),
        expected_repeated_a_diagonal()
    );
    assert_eq!(source.distribution(), &distribution);

    let zero_distribution = cyclic(context, vec![3, 2]);
    let zero = sparse(context, zero_distribution.clone(), &[(0, 0)]);
    let mut zeroed = source.clone();
    zeroed.replace_diagonal("iik", &zero);
    assert_eq!(zeroed.distribution(), &distribution);
    assert_eq!(
        zeroed.read(&all_keys(&distribution)),
        expected_without_iik_diagonal(|_, _| 0),
    );

    let explicit = sparse_fn(context, zero_distribution, |coordinates| {
        match (coordinates[0], coordinates[1]) {
            (0, 0) => 17,
            (1, 0) => -19,
            (2, 1) => 23,
            _ => 0,
        }
    });
    zeroed.replace_diagonal("iik", &explicit);
    assert_eq!(
        zeroed.read(&all_keys(&distribution)),
        expected_without_iik_diagonal(|i, k| match (i, k) {
            (0, 0) => 17,
            (1, 0) => -19,
            (2, 1) => 23,
            _ => 0,
        }),
    );
}

fn triple_diagonal_source(coordinates: &[usize]) -> i64 {
    if coordinates[0] == coordinates[1] && coordinates[1] == coordinates[2] {
        [7, -5, 11][coordinates[0]]
    } else {
        30_000 + 100 * coordinates[0] as i64 + 10 * coordinates[1] as i64 + coordinates[2] as i64
    }
}

fn triple_expected(diagonal: impl Fn(usize) -> i64) -> Vec<i64> {
    (0..27)
        .map(|key| {
            let i0 = key % 3;
            let i1 = (key / 3) % 3;
            let i2 = key / 9;
            if i0 == i1 && i1 == i2 {
                diagonal(i0)
            } else {
                triple_diagonal_source(&[i0, i1, i2])
            }
        })
        .collect()
}

fn triple_diagonal_extract_replace(context: &Context<'_>) {
    let distribution = virtual2(context, vec![3, 3, 3]);
    let source = sparse_fn(context, distribution.clone(), triple_diagonal_source);
    let (vector, labels) = source.extract_diagonal("iii");
    assert_eq!(labels, "i");
    assert_eq!(vector.distribution().shape, vec![3]);
    assert_eq!(vector.read(&[0, 1, 2]), vec![7, -5, 11]);
    if context.size() > 3 && context.rank() >= 3 {
        assert!(vector.local_pairs().is_empty());
    }

    let vector_distribution = cyclic(context, vec![3]);
    let zero = sparse(context, vector_distribution.clone(), &[(0, 0)]);
    let mut zeroed = source.clone();
    zeroed.replace_diagonal("iii", &zero);
    assert_eq!(
        zeroed.read(&all_keys(&distribution)),
        triple_expected(|_| 0)
    );

    let explicit = sparse_fn(context, vector_distribution, |coordinates| {
        [13, -17, 29][coordinates[0]]
    });
    zeroed.replace_diagonal("iii", &explicit);
    assert_eq!(
        zeroed.read(&all_keys(&distribution)),
        triple_expected(|i| [13, -17, 29][i]),
    );
}

fn run(context: &Context<'_>) {
    repeated_a_unique_output(context);
    repeated_a_repeated_output(context);
    repeated_b_unique_output(context);
    sparse_iik_extract_replace(context);
    triple_diagonal_extract_replace(context);
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
            "DIGIT / PASS distributed_sparse_diagonal: repeated sparse contractions, repeated A/B/output labels, exact diagonal extract/replace; world+parity"
        );
    }
    world.close();
    drop(universe);
}
