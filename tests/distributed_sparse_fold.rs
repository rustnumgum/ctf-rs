//! Distributed sparse indexed contractions through the bounded fold path.
//!
//! The main fixture has two contracted labels, one batch label, and all three
//! operands in a different index order.  Sparse inputs intentionally omit
//! entries so the exact integer oracle also checks algebraic zeros.
use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    sparse::SparseTensor,
    tensor::Tensor,
};

type Algebra = Arithmetic<i64>;

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

fn sparse<'c, 'r>(
    context: &'c Context<'r>,
    distribution: Distribution,
    entries: &[(usize, i64)],
) -> SparseTensor<'c, 'r, Algebra> {
    let mut tensor = SparseTensor::new(context, distribution.clone(), Algebra::new());
    let owned: Vec<_> = entries
        .iter()
        .copied()
        .filter(|(key, _)| distribution.owner(*key) == context.rank())
        .collect();
    tensor.write_add(&owned);
    tensor
}

fn dense<'c, 'r>(
    context: &'c Context<'r>,
    distribution: Distribution,
    entries: &[(usize, i64)],
) -> Tensor<'c, 'r, Algebra> {
    let mut tensor = Tensor::new(context, distribution, Algebra::new());
    tensor.transform(|key, value| *value = lookup(entries, key));
    tensor
}

fn lookup(entries: &[(usize, i64)], key: usize) -> i64 {
    entries
        .iter()
        .find_map(|&(entry, value)| (entry == key).then_some(value))
        .unwrap_or(0)
}

fn a_value(key: usize) -> i64 {
    (key as i64 % 9) - 4
}

fn b_value(key: usize) -> i64 {
    ((key as i64 * 7 + 2) % 11) - 5
}

fn a_entries() -> Vec<(usize, i64)> {
    (0..24)
        .filter(|key| key % 3 != 0)
        .map(|key| (key, a_value(key)))
        .collect()
}

fn b_entries() -> Vec<(usize, i64)> {
    (0..36)
        .filter(|key| key % 4 != 1)
        .map(|key| (key, b_value(key)))
        .collect()
}

fn folded_expected(a: &[(usize, i64)], b: &[(usize, i64)], old: impl Fn(usize) -> i64) -> Vec<i64> {
    let mut expected = vec![0; 12];
    for batch in 0..2 {
        for i in 0..2 {
            for j in 0..3 {
                let mut product = 0;
                for l in 0..2 {
                    for k in 0..3 {
                        let a_key = i + 2 * k + 6 * l + 12 * batch;
                        let b_key = l + 2 * j + 6 * k + 18 * batch;
                        product += lookup(a, a_key) * lookup(b, b_key);
                    }
                }
                let c_key = j + 3 * i + 6 * batch;
                expected[c_key] = 2 * product + 3 * old(c_key);
            }
        }
    }
    expected
}

fn main_fixture(context: &Context<'_>, grid: [usize; 2]) {
    let a_distribution = cyclic(context, vec![2, 3, 2, 2]);
    let b_distribution = cyclic(context, vec![2, 3, 3, 2]);
    let c_distribution = virtual2(context, vec![3, 2, 2]);
    let a_entries = a_entries();
    let b_entries = b_entries();
    let sparse_old = [(0, 7), (5, -11), (11, 13)];
    let dense_old = |key| -17 + 4 * key as i64;
    let expected_sparse = folded_expected(&a_entries, &b_entries, |key| lookup(&sparse_old, key));
    let expected_dense = folded_expected(&a_entries, &b_entries, dense_old);

    let a = sparse(context, a_distribution.clone(), &a_entries);
    let b = sparse(context, b_distribution.clone(), &b_entries);
    let a_before = a.local_pairs();
    let b_before = b.local_pairs();

    let mut sparse_c = sparse(context, c_distribution.clone(), &sparse_old);
    sparse_c
        .contract_from("jib", &a, "iklb", &b, "ljkb", grid, 2, 3)
        .unwrap();
    assert_eq!(sparse_c.distribution(), &c_distribution);
    assert_eq!(sparse_c.read(&(0..12).collect::<Vec<_>>()), expected_sparse);

    let mut dense_c = dense(
        context,
        c_distribution.clone(),
        &(0..12).map(|key| (key, dense_old(key))).collect::<Vec<_>>(),
    );
    dense_c
        .contract_from_sparse("jib", &a, "iklb", &b, "ljkb", grid, 2, 3)
        .unwrap();
    assert_eq!(dense_c.distribution(), &c_distribution);
    assert_eq!(dense_c.read(&(0..12).collect::<Vec<_>>()), expected_dense);

    let dense_b = dense(context, b_distribution.clone(), &b_entries);
    let dense_b_before = dense_b.local_pairs();
    let mut dense_c_from_dense = dense(
        context,
        c_distribution.clone(),
        &(0..12).map(|key| (key, dense_old(key))).collect::<Vec<_>>(),
    );
    dense_c_from_dense
        .contract_from_sparse_dense("jib", &a, "iklb", &dense_b, "ljkb", grid, 2, 3)
        .unwrap();
    assert_eq!(dense_c_from_dense.distribution(), &c_distribution);
    assert_eq!(
        dense_c_from_dense.read(&(0..12).collect::<Vec<_>>()),
        expected_dense
    );

    assert_eq!(a.distribution(), &a_distribution);
    assert_eq!(b.distribution(), &b_distribution);
    assert_eq!(a.local_pairs(), a_before);
    assert_eq!(b.local_pairs(), b_before);
    assert_eq!(dense_b.distribution(), &b_distribution);
    assert_eq!(dense_b.local_pairs(), dense_b_before);
}

fn outer_expected(a: &[i64], b: &[i64], old: impl Fn(usize) -> i64) -> Vec<i64> {
    (0..6)
        .map(|key| {
            let j = key % 3;
            let i = key / 3;
            2 * a[i] * b[j] + 3 * old(key)
        })
        .collect()
}

fn outer_product(context: &Context<'_>, grid: [usize; 2]) {
    let a_values = [3, 0];
    let b_values = [-2, 0, 4];
    let a_entries = [(0, a_values[0])];
    let b_entries = [(0, b_values[0]), (2, b_values[2])];
    let a_distribution = cyclic(context, vec![2]);
    let b_distribution = cyclic(context, vec![3]);
    let c_distribution = virtual2(context, vec![3, 2]);
    let sparse_old = [(1, 9), (5, -4)];
    let dense_old = |key| 11 - key as i64;
    let expected_sparse = outer_expected(&a_values, &b_values, |key| lookup(&sparse_old, key));
    let expected_dense = outer_expected(&a_values, &b_values, dense_old);

    let a = sparse(context, a_distribution.clone(), &a_entries);
    let b = sparse(context, b_distribution.clone(), &b_entries);
    let mut sparse_c = sparse(context, c_distribution.clone(), &sparse_old);
    sparse_c
        .contract_from("ji", &a, "i", &b, "j", grid, 2, 3)
        .unwrap();
    assert_eq!(sparse_c.read(&(0..6).collect::<Vec<_>>()), expected_sparse);

    let mut dense_c = dense(
        context,
        c_distribution.clone(),
        &(0..6).map(|key| (key, dense_old(key))).collect::<Vec<_>>(),
    );
    dense_c
        .contract_from_sparse("ji", &a, "i", &b, "j", grid, 2, 3)
        .unwrap();
    assert_eq!(dense_c.read(&(0..6).collect::<Vec<_>>()), expected_dense);

    let dense_b = dense(context, b_distribution, &b_entries);
    let mut dense_c_from_dense = dense(
        context,
        c_distribution,
        &(0..6).map(|key| (key, dense_old(key))).collect::<Vec<_>>(),
    );
    dense_c_from_dense
        .contract_from_sparse_dense("ji", &a, "i", &dense_b, "j", grid, 2, 3)
        .unwrap();
    assert_eq!(
        dense_c_from_dense.read(&(0..6).collect::<Vec<_>>()),
        expected_dense
    );
}

fn dot_expected(a: &[i64], b: &[i64], old: i64) -> i64 {
    2 * a.iter().zip(b).map(|(a, b)| a * b).sum::<i64>() + 3 * old
}

fn scalar_dot(context: &Context<'_>, grid: [usize; 2]) {
    let a_values = [4, -3];
    let b_values = [2, 0];
    let a_entries = [(0, a_values[0]), (1, a_values[1])];
    let b_entries = [(0, b_values[0])];
    let a_distribution = cyclic(context, vec![2]);
    let b_distribution = cyclic(context, vec![2]);
    let c_distribution = cyclic(context, vec![]);
    let old = 7;
    let expected = dot_expected(&a_values, &b_values, old);

    let a = sparse(context, a_distribution, &a_entries);
    let b = sparse(context, b_distribution.clone(), &b_entries);
    let mut sparse_c = sparse(context, c_distribution.clone(), &[(0, old)]);
    sparse_c
        .contract_from("", &a, "i", &b, "i", grid, 2, 3)
        .unwrap();
    assert_eq!(sparse_c.read(&[0]), vec![expected]);

    let mut dense_c = dense(context, c_distribution.clone(), &[(0, old)]);
    dense_c
        .contract_from_sparse("", &a, "i", &b, "i", grid, 2, 3)
        .unwrap();
    assert_eq!(dense_c.read(&[0]), vec![expected]);

    let dense_b = dense(context, b_distribution, &b_entries);
    let mut dense_c_from_dense = dense(context, c_distribution, &[(0, old)]);
    dense_c_from_dense
        .contract_from_sparse_dense("", &a, "i", &dense_b, "i", grid, 2, 3)
        .unwrap();
    assert_eq!(dense_c_from_dense.read(&[0]), vec![expected]);
}

fn huge_sparse_reshape(context: &Context<'_>) {
    let source_distribution = cyclic(context, vec![1_000_000, 1_000_000]);
    let target_distribution = cyclic(context, vec![1_000_000_000_000]);
    let far_key = 1_000_000 * 999_999 + 999_999;
    let source = sparse(
        context,
        source_distribution.clone(),
        &[(0, 7), (far_key, -13)],
    );
    assert!(source.local_nnz() <= 2);
    let reshaped = source.reshape(target_distribution.clone());
    assert_eq!(reshaped.distribution(), &target_distribution);
    assert!(reshaped.local_nnz() <= 2);
    assert_eq!(reshaped.read(&[0, far_key]), vec![7, -13]);
}

fn run(context: &Context<'_>) {
    let grid = if context.size() == 4 {
        [2, 2]
    } else {
        [context.size(), 1]
    };
    main_fixture(context, grid);
    outer_product(context, grid);
    scalar_dot(context, grid);
    huge_sparse_reshape(context);
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
            "DIGIT / PASS distributed_sparse_fold: folded sparse contractions, outer product, dot product, reshape; world+parity"
        );
    }
    world.close();
    drop(universe);
}
