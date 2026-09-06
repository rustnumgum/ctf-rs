//! Exact distributed sparse GEMM coverage for arithmetic semirings.
//!
//! The fixtures deliberately use different source and destination layouts:
//! both inputs start cyclic, while every output uses a virtual second copy of
//! its first axis.  The result is checked by a small serial column-major
//! oracle, so this test also covers missing sparse entries as algebraic zero.
use ctf::{
    algebra::{Arithmetic, CustomMonoid, CustomSemiring},
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    sparse::SparseTensor,
    tensor::Tensor,
};

type Algebra = Arithmetic<i64>;

fn cyclic(context: &Context<'_>, shape: Vec<usize>) -> Distribution {
    Distribution::cyclic(shape, context.size())
}

fn virtual_first_axis(context: &Context<'_>, shape: Vec<usize>) -> Distribution {
    let topology = Topology::new(vec![context.size()]);
    let mut first = Mapping::Unmapped;
    first.augment_physical(&topology, 0);
    first.augment_virtual(context.size() * 2);
    Distribution::new(shape, topology, vec![first, Mapping::Unmapped])
}

fn sparse_tensor<'c, 'r>(
    context: &'c Context<'r>,
    distribution: Distribution,
    pairs: &[(usize, i64)],
) -> SparseTensor<'c, 'r, Algebra> {
    let mut tensor = SparseTensor::new(context, distribution.clone(), Algebra::new());
    let owned: Vec<_> = pairs
        .iter()
        .copied()
        .filter(|(key, _)| distribution.owner(*key) == context.rank())
        .collect();
    tensor.write_add(&owned);
    tensor
}

fn dense_tensor<'c, 'r>(
    context: &'c Context<'r>,
    distribution: Distribution,
    pairs: &[(usize, i64)],
) -> Tensor<'c, 'r, Algebra> {
    let mut tensor = Tensor::new(context, distribution.clone(), Algebra::new());
    let owned: Vec<_> = pairs
        .iter()
        .copied()
        .filter(|(key, _)| distribution.owner(*key) == context.rank())
        .collect();
    tensor.write_add(&owned);
    tensor
}

fn a_value(i: usize, k: usize) -> i64 {
    17 + 7 * i as i64 - 5 * k as i64
}

fn b_value(k: usize, j: usize) -> i64 {
    -11 + 3 * k as i64 + 13 * j as i64
}

fn keep_a(i: usize, k: usize, m: usize, kdim: usize) -> bool {
    // Keep the final K column absent in the main rectangular fixture.  That
    // leaves a source-side missing K column in addition to ordinary holes.
    if kdim > 1 && k + 1 == kdim {
        return false;
    }
    m * kdim == 1 || (17 * i + 29 * k + 1) % 4 < 2
}

fn keep_b(k: usize, j: usize, kdim: usize, n: usize) -> bool {
    kdim * n == 1 || (31 * k + 13 * j + 2) % 4 < 2
}

fn sparse_a_pairs(m: usize, kdim: usize) -> Vec<(usize, i64)> {
    (0..m * kdim)
        .filter_map(|key| {
            let i = key % m;
            let k = key / m;
            keep_a(i, k, m, kdim).then_some((key, a_value(i, k)))
        })
        .collect()
}

fn sparse_b_pairs(kdim: usize, n: usize) -> Vec<(usize, i64)> {
    (0..kdim * n)
        .filter_map(|key| {
            let k = key % kdim;
            let j = key / kdim;
            keep_b(k, j, kdim, n).then_some((key, b_value(k, j)))
        })
        .collect()
}

fn dense_b_pairs(kdim: usize, n: usize) -> Vec<(usize, i64)> {
    (0..kdim * n)
        .map(|key| {
            let k = key % kdim;
            let j = key / kdim;
            (key, b_value(k, j))
        })
        .collect()
}

fn sparse_c_pairs(m: usize, n: usize) -> Vec<(usize, i64)> {
    let len = m * n;
    let mut pairs = vec![(0, 23)];
    if len > 1 {
        pairs.push((len / 2, -19));
    }
    if len > 2 {
        pairs.push((len - 1, 31));
    }
    pairs
}

fn sparse_c_value(key: usize, m: usize, n: usize) -> i64 {
    sparse_c_pairs(m, n)
        .into_iter()
        .find_map(|(entry, value)| (entry == key).then_some(value))
        .unwrap_or(0)
}

fn dense_c_value(key: usize) -> i64 {
    400 - 3 * key as i64
}

fn sparse_product(m: usize, kdim: usize, n: usize, a: &[(usize, i64)], b: &[(usize, i64)]) -> Vec<i64> {
    let mut result = vec![0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut value = 0;
            for k in 0..kdim {
                let a_key = i + m * k;
                let b_key = k + kdim * j;
                let av = a
                    .iter()
                    .find_map(|&(key, value)| (key == a_key).then_some(value))
                    .unwrap_or(0);
                let bv = b
                    .iter()
                    .find_map(|&(key, value)| (key == b_key).then_some(value))
                    .unwrap_or(0);
                value += av * bv;
            }
            result[i + m * j] = value;
        }
    }
    result
}

fn dense_right_product(m: usize, kdim: usize, n: usize, a: &[(usize, i64)]) -> Vec<i64> {
    let b = dense_b_pairs(kdim, n);
    sparse_product(m, kdim, n, a, &b)
}

fn expected(product: &[i64], old: impl Fn(usize) -> i64) -> Vec<i64> {
    product
        .iter()
        .enumerate()
        .map(|(key, value)| 2 * value + 3 * old(key))
        .collect()
}

fn keys(len: usize) -> Vec<usize> {
    (0..len).collect()
}

fn check_sparse_sparse(
    context: &Context<'_>,
    m: usize,
    kdim: usize,
    n: usize,
    grid: [usize; 2],
) {
    let a_pairs = sparse_a_pairs(m, kdim);
    let b_pairs = sparse_b_pairs(kdim, n);
    let product = sparse_product(m, kdim, n, &a_pairs, &b_pairs);
    let expected_sparse = expected(&product, |key| sparse_c_value(key, m, n));
    let expected_dense = expected(&product, dense_c_value);
    let a_distribution = cyclic(context, vec![m, kdim]);
    let b_distribution = cyclic(context, vec![kdim, n]);
    let c_distribution = virtual_first_axis(context, vec![m, n]);
    let a = sparse_tensor(context, a_distribution.clone(), &a_pairs);
    let b = sparse_tensor(context, b_distribution.clone(), &b_pairs);

    let mut sparse_c = sparse_tensor(context, c_distribution.clone(), &sparse_c_pairs(m, n));
    sparse_c.gemm_sparse(&a, &b, grid, 2, 3);
    assert_eq!(sparse_c.distribution(), &c_distribution);
    assert_eq!(a.distribution(), &a_distribution);
    assert_eq!(b.distribution(), &b_distribution);
    assert_eq!(sparse_c.read(&keys(m * n)), expected_sparse);

    let mut dense_c = dense_tensor(
        context,
        c_distribution.clone(),
        &(0..m * n)
            .map(|key| (key, dense_c_value(key)))
            .collect::<Vec<_>>(),
    );
    dense_c.gemm_sparse(&a, &b, grid, 2, 3);
    assert_eq!(dense_c.distribution(), &c_distribution);
    assert_eq!(dense_c.read(&keys(m * n)), expected_dense);
}

fn check_sparse_dense(
    context: &Context<'_>,
    m: usize,
    kdim: usize,
    n: usize,
    grid: [usize; 2],
) {
    let a_pairs = sparse_a_pairs(m, kdim);
    let b_pairs = dense_b_pairs(kdim, n);
    let product = dense_right_product(m, kdim, n, &a_pairs);
    let expected_values = expected(&product, dense_c_value);
    let a_distribution = cyclic(context, vec![m, kdim]);
    let b_distribution = cyclic(context, vec![kdim, n]);
    let c_distribution = virtual_first_axis(context, vec![m, n]);
    let a = sparse_tensor(context, a_distribution.clone(), &a_pairs);
    let b = dense_tensor(context, b_distribution.clone(), &b_pairs);
    let mut c = dense_tensor(
        context,
        c_distribution.clone(),
        &(0..m * n)
            .map(|key| (key, dense_c_value(key)))
            .collect::<Vec<_>>(),
    );
    c.gemm_sparse_dense(&a, &b, grid, 2, 3);
    assert_eq!(c.distribution(), &c_distribution);
    assert_eq!(a.distribution(), &a_distribution);
    assert_eq!(b.distribution(), &b_distribution);
    assert_eq!(c.read(&keys(m * n)), expected_values);
}

fn check_empty_a_panel(context: &Context<'_>, grid: [usize; 2]) {
    // A has no stored values at all.  This is the strongest form of an empty
    // A panel and exercises the collective path with no source K updates.
    let (m, kdim, n) = (3, 1, 5);
    let a_pairs = Vec::new();
    let b_pairs = sparse_b_pairs(kdim, n);
    let product = vec![0; m * n];
    let expected_sparse = expected(&product, |key| sparse_c_value(key, m, n));
    let expected_dense = expected(&product, dense_c_value);
    let a_distribution = cyclic(context, vec![m, kdim]);
    let b_distribution = cyclic(context, vec![kdim, n]);
    let c_distribution = virtual_first_axis(context, vec![m, n]);
    let a = sparse_tensor(context, a_distribution.clone(), &a_pairs);
    let b = sparse_tensor(context, b_distribution.clone(), &b_pairs);

    let mut sparse_c = sparse_tensor(context, c_distribution.clone(), &sparse_c_pairs(m, n));
    sparse_c.gemm_sparse(&a, &b, grid, 2, 3);
    assert_eq!(sparse_c.distribution(), &c_distribution);
    assert_eq!(sparse_c.read(&keys(m * n)), expected_sparse);

    let mut dense_c = dense_tensor(
        context,
        c_distribution.clone(),
        &(0..m * n)
            .map(|key| (key, dense_c_value(key)))
            .collect::<Vec<_>>(),
    );
    dense_c.gemm_sparse(&a, &b, grid, 2, 3);
    assert_eq!(dense_c.distribution(), &c_distribution);
    assert_eq!(dense_c.read(&keys(m * n)), expected_dense);
}

fn run(context: &Context<'_>) {
    let np = context.size();
    let primary_grid = if np == 4 { [2, 2] } else { [np, 1] };
    for &(m, kdim, n) in &[(5, 7, 3), (1, 1, 1), (3, 1, 5)] {
        check_sparse_sparse(context, m, kdim, n, primary_grid);
        check_sparse_dense(context, m, kdim, n, primary_grid);
    }
    check_empty_a_panel(context, primary_grid);
    min_plus(context, primary_grid);

    // A second rectangular grid makes the K phase nonuniform for k=7 on a
    // four-rank communicator and exercises the opposite panel orientation.
    if np == 4 {
        check_sparse_sparse(context, 5, 7, 3, [1, np]);
        check_sparse_dense(context, 5, 7, 3, [1, np]);
        check_empty_a_panel(context, [1, np]);
    }
}

fn min_plus(context: &Context<'_>, grid: [usize; 2]) {
    const INF: i64 = 1_000_000;
    let algebra = CustomSemiring {
        monoid: CustomMonoid { identity: INF, addition: |a: &i64, b: &i64| (*a).min(*b) },
        identity: 0,
        multiplication: |a: &i64, b: &i64| if *a == INF || *b == INF { INF } else { a + b },
    };
    let mut a = SparseTensor::new(context, cyclic(context, vec![2, 3]), algebra.clone());
    let mut b = SparseTensor::new(context, cyclic(context, vec![3, 2]), algebra.clone());
    a.write_add(if context.rank() == 0 { &[(0, 1), (3, 2)] } else { &[] });
    b.write_add(if context.rank() == 0 { &[(0, 4), (4, 3)] } else { &[] });
    let mut c = SparseTensor::new(context, cyclic(context, vec![2, 2]), algebra.clone());
    c.gemm_sparse(&a, &b, grid, 0, 0);
    assert_eq!(c.read(&[0, 1, 2, 3]), vec![5, INF, INF, 5]);
    let mut dense = Tensor::new(context, cyclic(context, vec![2, 2]), algebra);
    dense.gemm_sparse(&a, &b, grid, 0, 0);
    assert_eq!(dense.read(&[0, 1, 2, 3]), vec![5, INF, INF, 5]);
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let world_rank = world.rank();
    run(&world);

    let parity = world
        .split(Some((world_rank % 2) as i32), world_rank as i32)
        .unwrap();
    run(&parity);
    parity.close();

    if world_rank == 0 {
        println!(
            "DIGIT / PASS distributed_sparse_gemm: sparse*sparse and sparse*dense GEMM, exact i64 oracle, empty A panels, restored virtual output; world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
