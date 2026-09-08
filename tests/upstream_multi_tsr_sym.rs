//! Port of pinned CTF test/multi_tsr_sym.cxx.
use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, *},
};

fn tensor<'c, 'r>(
    context: &'c Context<'r>,
    shape: Vec<usize>,
    links: Vec<Symmetry>,
) -> SymmetricTensor<'c, 'r, Arithmetic<f64>> {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    if !mappings.is_empty() {
        mappings[0].augment_physical(&topology, 0);
    }
    for mapping in &mut mappings {
        mapping.augment_virtual(2 * context.size());
    }
    SymmetricTensor::new(
        context,
        SymmetricDistribution::new(Distribution::new(shape, topology, mappings), links),
        Arithmetic::new(),
    )
}

fn fixture(global_key: usize, seed: u64) -> f64 {
    let mixed = (global_key as u64)
        .wrapping_add(seed)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .rotate_left(29);
    (mixed % 2003) as f64 / 2003.0 - 0.5
}

fn contraction_grid(context: &Context<'_>) -> (Topology, &'static str) {
    if context.size() == 4 {
        (Topology::new(vec![2, 2]), "ik")
    } else {
        (Topology::new(vec![context.size()]), "k")
    }
}

fn run(context: &Context<'_>) -> f64 {
    let n = 3;
    let m = 3;
    let mut a = tensor(context, vec![n, m], vec![NS, NS]);
    let mut c_ns = tensor(context, vec![n, n], vec![NS, NS]);
    let mut c_sy = tensor(context, vec![n, n], vec![SY, NS]);
    let mut difference = tensor(context, vec![n, n], vec![NS, NS]);

    a.transform(|key, value| *value = fixture(key, 13));
    let (topology, physical_labels) = contraction_grid(context);
    c_ns.contract_from_on(
        "ij",
        &a,
        "ik",
        &a,
        "jk",
        topology.clone(),
        physical_labels,
        1.0,
        0.0,
        true,
    )
    .unwrap();
    c_sy.contract_from_on(
        "ij",
        &a,
        "ik",
        &a,
        "jk",
        topology,
        physical_labels,
        1.0,
        0.0,
        true,
    )
    .unwrap();

    difference.sum_from("ij", &c_sy, "ij", 1.0, 0.0);
    difference.sum_from("ij", &c_ns, "ij", -1.0, 1.0);
    let keys: Vec<_> = (0..n * n).collect();
    let error = difference
        .read(&keys)
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!(
        error.is_finite() && error < 1.0e-6,
        "source multi_tsr_sym error={error}"
    );
    error
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let world_error = run(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    let parity_error = run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS upstream_multi_tsr_sym: source NS/SY Gram equality; world_error={world_error:e}; parity_error={parity_error:e}; norm<1e-6"
        );
    }
    world.close();
    drop(universe);
}
