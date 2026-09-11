// Adapted from pinned test/python/test_sparse.py::test_complex.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//
// The source test is named "complex", but its arrays and CTF tensors contain
// real values.  The indexed-view scale factors are expression coefficients:
// they do not mutate either source tensor before the assignment.

use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Topology},
    tensor::Tensor,
};

type F64 = Arithmetic<f64>;

fn c_order_arange(tensor: &mut Tensor<'_, '_, F64>) {
    let distribution = tensor.distribution().clone();
    tensor.transform(|key, value| {
        let coordinates = distribution.decode_key(key);
        *value = (coordinates[0] * 9 + coordinates[1] * 3 + coordinates[2]) as f64;
    });
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
            "sparse_complex {label}: sum(abs(diff)) = {difference:e}, bound=1e-14"
        );
    }
    assert!(difference < 1e-14, "sum(abs(diff)) = {difference:e}");
}

fn run(context: &Context<'_>) {
    let distribution = Distribution::cyclic(vec![3, 3, 3], context.size());
    let topology = Topology::new(vec![context.size()]);
    let mut a1 = Tensor::new(context, distribution.clone(), F64::new());
    let mut b1 = Tensor::new(context, distribution.clone(), F64::new());
    c_order_arange(&mut a1);
    c_order_arange(&mut b1);

    // This intentionally follows the source: a1i is b1.i("kij"), not
    // a1.i("kij").  The two initial arrays are equal, but the aliasing is
    // part of the indexed-view expression being ported.
    let b1_before = b1.clone();
    let mut actual = b1;
    actual
        .sum_from("ijk", &b1_before, "kij", topology.clone(), 0.7, 0.2)
        .unwrap();

    // Dense Rust evaluation of the source's expected expression:
    // b0 = .2*b0 + .7*a0.transpose([1,2,0]).
    let reference_distribution = b1_before.distribution().clone();
    let mut reference = b1_before.clone();
    reference.transform(|key, value| {
        let coordinates = reference_distribution.decode_key(key);
        let i = coordinates[0];
        let j = coordinates[1];
        let k = coordinates[2];
        let b0 = (i * 9 + j * 3 + k) as f64;
        let a0_transposed = (k * 9 + i * 3 + j) as f64;
        *value = 0.2 * b0 + 0.7 * a0_transposed;
    });
    allclose("b0 vs b1", &reference, &actual);
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
            "DIGIT / PASS sparse_complex: real arange(27) indexed kij coefficient assignment; sum(abs(diff))<1e-14; world+parity"
        );
    }
    world.close();
    drop(universe);
}
