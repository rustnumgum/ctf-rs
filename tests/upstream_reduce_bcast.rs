//! Source test/reduce_bcast.cxx equations and Frobenius bound 1e-6.
use ctf::{
    algebra::Arithmetic, context::Context, mapping::Distribution,
    symmetric_distribution::SymmetricDistribution, symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::NS,
};
fn tensor<'c, 'r>(
    c: &'c Context<'r>,
    shape: Vec<usize>,
) -> SymmetricTensor<'c, 'r, Arithmetic<f64>> {
    let order = shape.len();
    SymmetricTensor::new(
        c,
        SymmetricDistribution::new(Distribution::cyclic(shape, c.size()), vec![NS; order]),
        Arithmetic::new(),
    )
}
fn check(t: &SymmetricTensor<'_, '_, Arithmetic<f64>>) {
    let values = t.read(&(0..9).collect::<Vec<_>>());
    let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        norm.is_finite() && norm <= 1e-6,
        "source reduce_bcast norm={norm}"
    );
}
fn run(context: &Context<'_>) {
    let mut b = tensor(context, vec![3, 1]);
    b.transform(|key, v| *v = (key as f64 + 1.) / 8.);
    let mut c = tensor(context, vec![3, 3]);
    c.transform(|key, v| *v = (key as f64 + 2.) / 32.);
    let mut c2 = tensor(context, vec![3, 3]);
    c2.sum_from("ij", &c, "ij", 1., 0.);
    let mut d = tensor(context, vec![3]);
    c.sum_from("ij", &b, "ik", 1., 1.);
    d.sum_from("i", &b, "ij", 1., 0.);
    c2.sum_from("ij", &d, "i", 1., 1.);
    c.sum_from("ij", &c2, "ij", -1., 1.);
    check(&c);
    c.sum_from("ij", &c2, "ij", 1., 0.);
    c.sum_from("ij", &b, "ik", 1., 1.);
    d.sum_from("i", &b, "ik", 1., 0.);
    c2.sum_from("ij", &d, "i", 1., 1.);
    c.sum_from("ij", &c2, "ij", -1., 1.);
    check(&c);
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
            "DIGIT / PASS upstream_reduce_bcast: source reduction/broadcast identities; norm<=1e-6; world+parity"
        );
    }
    world.close();
    drop(universe);
}
