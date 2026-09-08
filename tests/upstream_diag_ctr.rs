//! test/diag_ctr.cxx with its nonzero-trace and absolute 1e-10 checks.
use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::NS,
};
fn tensor<'c, 'r>(
    c: &'c Context<'r>,
    shape: Vec<usize>,
) -> SymmetricTensor<'c, 'r, Arithmetic<f64>> {
    let topology = Topology::new(vec![c.size()]);
    let mut maps = vec![Mapping::Unmapped; shape.len()];
    if !maps.is_empty() {
        maps[0].augment_physical(&topology, 0);
    }
    let order = shape.len();
    SymmetricTensor::new(
        c,
        SymmetricDistribution::new(Distribution::new(shape, topology, maps), vec![NS; order]),
        Arithmetic::new(),
    )
}
fn run(c: &Context<'_>) {
    let mut a = tensor(c, vec![3, 2, 3, 2]);
    a.transform(|key, v| *v = (key as f64 + 1.) / 64.);
    let mut trace = tensor(c, vec![]);
    trace.sum_from("", &a, "aiai", 1., 0.);
    let first = trace.read(&[0])[0];
    assert!(first.is_finite() && first.abs() >= 1e-10);
    let mut ma = tensor(c, vec![3, 2]);
    ma.sum_from("ai", &a, "aiai", 1., 0.);
    trace.sum_from("", &ma, "ai", -1., 1.);
    let residual = trace.read(&[0])[0];
    assert!(residual.is_finite() && residual.abs() <= 1e-10);
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
            "DIGIT / PASS upstream_diag_ctr: source double-diagonal trace identity; absolute 1e-10; world+parity"
        );
    }
    world.close();
    drop(universe);
}
