// Adapted from pinned test/python/test_sparse.py::test_einsum_hadamard.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.

use ctf::{
    algebra::Arithmetic,
    context::Context,
    cost::Models,
    mapping::{Distribution, Topology},
    random::Generator,
    sparse::SparseTensor,
    sparse_search::{Options, Pattern, SearchCache, StorageSize},
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

fn dense_einsum<'c, 'r>(
    context: &'c Context<'r>,
    a: &Tensor<'c, 'r, F64>,
    indices_a: &str,
    b: &Tensor<'c, 'r, F64>,
    indices_b: &str,
    shape_c: Vec<usize>,
    indices_c: &str,
) -> Tensor<'c, 'r, F64> {
    let mut output = Tensor::new(
        context,
        Distribution::cyclic(shape_c, context.size()),
        F64::new(),
    );
    output
        .contract_from(
            indices_c,
            a,
            indices_a,
            b,
            indices_b,
            Topology::new(vec![context.size()]),
            1.0,
            0.0,
        )
        .unwrap();
    output
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
            "sparse_einsum_hadamard {label}: sum(abs(diff)) = {difference:e}, bound=1e-14"
        );
    }
    assert!(difference < 1e-14, "sum(abs(diff)) = {difference:e}");
}

fn run(context: &Context<'_>) {
    let n = 11;
    let input_shape = vec![n, n, n];
    let output_shape = vec![n, n, n, n];
    let mut generator = Generator::new(context.rank() as u64);

    let mut a = SparseTensor::new(
        context,
        Distribution::cyclic(input_shape.clone(), context.size()),
        F64::new(),
    );
    let mut b = SparseTensor::new(
        context,
        Distribution::cyclic(input_shape.clone(), context.size()),
        F64::new(),
    );
    let mut c = Tensor::new(
        context,
        Distribution::cyclic(input_shape, context.size()),
        F64::new(),
    );
    a.fill_random_sparse(0.0, 1.0, DENSITY, &mut generator);
    b.fill_random_sparse(0.0, 1.0, DENSITY, &mut generator);
    c.fill_random_sparse(0.0, 1.0, DENSITY, &mut generator);

    let dense_a = a.clone().into_dense();
    let dense_b = b.clone().into_dense();
    let reference_sparse = dense_einsum(
        context,
        &dense_a,
        "ijk",
        &dense_b,
        "jkl",
        output_shape.clone(),
        "ijkl",
    );
    let mut actual_sparse = SparseTensor::new(
        context,
        Distribution::cyclic(output_shape.clone(), context.size()),
        F64::new(),
    );
    let catalog = topology_candidates::all_shapes(context.size());
    let models = Models::upstream(1);
    let mut cache = SearchCache::new(
        context,
        &catalog,
        &models,
        [StorageSize { element_bytes: 8, pair_bytes: 16 }; 3],
        false,
        Pattern::SparseSparseSparse,
        options(),
    );
    actual_sparse.contract_sparse(
        "ijkl",
        &a,
        "ijk",
        &b,
        "jkl",
        &mut cache,
        1.0,
        0.0,
        None,
        true,
    ).unwrap();
    let actual_sparse_dense = actual_sparse.into_dense();
    allclose("d1 vs e1", &reference_sparse, &actual_sparse_dense);

    let dense_c = c;
    let reference_dense = dense_einsum(
        context,
        &dense_a,
        "ijk",
        &dense_c,
        "jkl",
        output_shape.clone(),
        "ijkl",
    );
    let mut actual_dense = Tensor::new(
        context,
        Distribution::cyclic(output_shape, context.size()),
        F64::new(),
    );
    let mut cache = SearchCache::new(
        context,
        &catalog,
        &models,
        [StorageSize { element_bytes: 8, pair_bytes: 16 }; 3],
        false,
        Pattern::SparseDenseDense { coo_kernel: false },
        options(),
    );
    actual_dense.contract_sparse(
        "ijkl",
        &a,
        "ijk",
        &dense_c,
        "jkl",
        &mut cache,
        1.0,
        0.0,
        None,
        true,
    ).unwrap();
    allclose("d2 vs e2", &reference_dense, &actual_dense);
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
            "DIGIT / PASS sparse_einsum_hadamard: ijk,jkl->ijkl sparse-sparse and sparse-dense; n=11 sp=0.1; sum(abs(diff))<1e-14; world+parity"
        );
    }
    world.close();
    drop(universe);
}
