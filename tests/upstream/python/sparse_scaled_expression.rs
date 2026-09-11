// Adapted from pinned test/python/test_sparse.py::test_scaled_expression.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.

use ctf::{
    algebra::Arithmetic,
    context::Context,
    cost::Models,
    mapping::{Distribution, Topology},
    random::Generator,
    sparse::SparseTensor,
    sparse_search::{Options, Pattern, SearchCache},
    tensor::Tensor,
    topology_candidates,
};

type F64 = Arithmetic<f64>;

const DENSITY: f64 = 0.1;

fn options() -> Options {
    Options {
        memory_limit: 1u64 << 60,
        weight: 0.0,
        allow_exhaustive: true,
    }
}

fn canonical_nnz(tensor: &SparseTensor<'_, '_, F64>) -> u64 {
    let local = tensor
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| tensor.distribution().owner(*key) == tensor.context().rank())
        .count() as u64;
    tensor.context().all_reduce(&Arithmetic::<u64>::new(), &local)
}

fn prepare(
    context: &Context<'_>,
    old: [&Distribution; 3],
    indices: [&str; 3],
    nonzeros: [Option<u64>; 3],
    pattern: Pattern,
) -> ctf::sparse_search::Selected {
    let catalog = topology_candidates::all_shapes(context.size());
    let models = Models::upstream(1);
    let mut cache = SearchCache::new(
        context,
        &catalog,
        &models,
        8,
        16,
        false,
        pattern,
        options(),
    );
    cache
        .prepare(old, indices, nonzeros, None)
        .unwrap()
        .unwrap()
        .clone()
}

fn allclose(label: &str, reference: &Tensor<'_, '_, F64>, actual: &Tensor<'_, '_, F64>) {
    assert_eq!(reference.distribution().shape, actual.distribution().shape);
    let keys: Vec<_> = (0..reference.distribution().global_len()).collect();
    let reference_values = reference.read(&keys);
    let actual_values = actual.read(&keys);
    let difference: f64 = reference_values
        .into_iter()
        .zip(actual_values)
        .map(|(expected, value)| (expected - value).abs())
        .sum();
    if reference.context().rank() == 0 {
        println!(
            "sparse_scaled_expression {label}: sum(abs(diff)) = {difference:e}, bound=1e-14"
        );
    }
    assert!(difference < 1e-14, "sum(abs(diff)) = {difference:e}");
}

fn dense_expression<'c, 'r>(
    context: &'c Context<'r>,
    a: &Tensor<'c, 'r, F64>,
    b: &Tensor<'c, 'r, F64>,
    old_c: &Tensor<'c, 'r, F64>,
) -> (Tensor<'c, 'r, F64>, Tensor<'c, 'r, F64>) {
    let topology = Topology::new(vec![context.size()]);
    let mut c_dn = old_c.clone();
    c_dn
        .contract_from("ijk", a, "ijl", b, "kjl", topology.clone(), 2.3, 1.0)
        .unwrap();
    c_dn
        .sum_from("ijk", old_c, "ijk", topology.clone(), 7.0, 1.0)
        .unwrap();
    c_dn
        .sum_from("ijk", a, "ijk", topology.clone(), -1.0, 1.0)
        .unwrap();
    c_dn
        .sum_from("ijk", a, "ijk", topology.clone(), -1.0, 1.0)
        .unwrap();
    c_dn.sum_from("ijk", b, "ijk", topology.clone(), -2.0, 1.0)
        .unwrap();

    // c_np += 2.3*einsum(...) + 7*c_np - a_np - a_np - 2*b_np,
    // with c_np retaining its pre-assignment values while the RHS is built.
    let mut grouped_rhs = Tensor::new(context, old_c.distribution().clone(), F64::new());
    grouped_rhs
        .contract_from("ijk", a, "ijl", b, "kjl", topology.clone(), 2.3, 0.0)
        .unwrap();
    grouped_rhs
        .sum_from("ijk", old_c, "ijk", topology.clone(), 7.0, 1.0)
        .unwrap();
    grouped_rhs
        .sum_from("ijk", a, "ijk", topology.clone(), -1.0, 1.0)
        .unwrap();
    grouped_rhs
        .sum_from("ijk", a, "ijk", topology.clone(), -1.0, 1.0)
        .unwrap();
    grouped_rhs
        .sum_from("ijk", b, "ijk", topology, -2.0, 1.0)
        .unwrap();
    let mut c_np = old_c.clone();
    c_np
        .sum_from("ijk", &grouped_rhs, "ijk", Topology::new(vec![context.size()]), 1.0, 1.0)
        .unwrap();

    (c_np, c_dn)
}

fn sparse_expression<'c, 'r>(
    context: &'c Context<'r>,
    a: &SparseTensor<'c, 'r, F64>,
    b: &SparseTensor<'c, 'r, F64>,
    c: SparseTensor<'c, 'r, F64>,
) -> Tensor<'c, 'r, F64> {
    let old_c = c.clone();
    let selected = prepare(
        context,
        [a.distribution(), b.distribution(), c.distribution()],
        ["ijl", "kjl", "ijk"],
        [Some(canonical_nnz(a)), Some(canonical_nnz(b)), Some(canonical_nnz(&c))],
        Pattern::SparseSparseSparse,
    );
    let mut result = c;
    result.contract_sparse_from_selected(
        "ijk",
        a,
        "ijl",
        b,
        "kjl",
        &selected,
        2.3,
        1.0,
        true,
    );
    result.sum_from("ijk", &old_c, "ijk", 7.0, 1.0);
    result.sum_from("ijk", a, "ijk", -1.0, 1.0);
    result.sum_from("ijk", a, "ijk", -1.0, 1.0);
    result.sum_from("ijk", b, "ijk", -2.0, 1.0);
    result.into_dense()
}

fn run(context: &Context<'_>) {
    let n = 5;
    let shape = vec![n, n, n];
    let distribution = Distribution::cyclic(shape, context.size());
    let mut generator = Generator::new(context.rank() as u64);

    let mut a_sp = SparseTensor::new(context, distribution.clone(), F64::new());
    let mut b_sp = SparseTensor::new(context, distribution.clone(), F64::new());
    let mut c_sp = SparseTensor::new(context, distribution, F64::new());
    a_sp.fill_random_sparse(0.0, 1.0, DENSITY, &mut generator);
    b_sp.fill_random_sparse(0.0, 1.0, DENSITY, &mut generator);
    c_sp.fill_random_sparse(0.0, 1.0, DENSITY, &mut generator);

    let a_dn = a_sp.clone().into_dense();
    let b_dn = b_sp.clone().into_dense();
    let c_dn = c_sp.clone().into_dense();
    let (reference, dense_actual) = dense_expression(context, &a_dn, &b_dn, &c_dn);
    let actual = sparse_expression(context, &a_sp, &b_sp, c_sp);
    allclose("c_np vs c_dn", &reference, &dense_actual);
    allclose("c_np vs c_sp", &reference, &actual);
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    run(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS sparse_scaled_expression: ijl,kjl->ijk with sparse A/B/C and dense twin; n=5 sp=0.1; sum(abs(diff))<1e-14; world+parity"
        );
    }
    world.close();
    drop(universe);
}
