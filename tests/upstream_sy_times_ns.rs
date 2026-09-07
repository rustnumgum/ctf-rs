//! Port of pinned CTF test/sy_times_ns.cxx.
use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
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
        (Topology::new(vec![2, 2]), "kl")
    } else {
        (Topology::new(vec![context.size()]), "k")
    }
}

fn run_case(context: &Context<'_>, nonzero_operands: bool, case: &str) -> f64 {
    let n = 3;
    let mut b = tensor(context, vec![n; 4], vec![NS, NS, NS, NS]);
    let mut a = tensor(context, vec![n, n], vec![SY, NS]);
    let mut an = tensor(context, vec![n, n], vec![NS, NS]);
    let mut c = tensor(context, vec![n, n], vec![SY, NS]);
    let mut cn = tensor(context, vec![n, n], vec![NS, NS]);

    // The pinned source leaves A and B zero because both write calls are
    // disabled.  Keep that literal case, then separately exercise the intended
    // nonzero operation using rank-independent global-key fixtures.
    if nonzero_operands {
        a.transform(|key, value| *value = fixture(key, 13));
        b.transform(|key, value| *value = fixture(key, 29));
    }
    c.transform(|key, value| *value = fixture(key, 47));

    an.sum_from("ij", &a, "ij", 1.0, 0.0);
    cn.sum_from("ij", &c, "ij", 1.0, 0.0);

    let (topology, physical_labels) = contraction_grid(context);
    c.contract_from_on(
        "ij",
        &a,
        "ij",
        &b,
        "ijkl",
        topology.clone(),
        physical_labels,
        1.0,
        1.0,
        true,
    )
    .unwrap();
    cn.contract_from_on(
        "ij",
        &an,
        "ij",
        &b,
        "ijkl",
        topology.clone(),
        physical_labels,
        1.0,
        1.0,
        true,
    )
    .unwrap();
    cn.contract_from_on(
        "ji",
        &an,
        "ij",
        &b,
        "ijkl",
        topology,
        physical_labels,
        1.0,
        1.0,
        true,
    )
    .unwrap();
    cn.sum_from("ij", &c, "ij", -1.0, 1.0);

    let keys: Vec<_> = (0..n * n).collect();
    let norm = cn
        .read(&keys)
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!(
        norm.is_finite() && norm < 1.0e-10,
        "sy_times_ns {case} norm={norm}"
    );
    norm
}

fn run(context: &Context<'_>) -> (f64, f64) {
    let literal = run_case(context, false, "literal-source");
    let nonzero = run_case(context, true, "adapted-nonzero");
    (literal, nonzero)
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let (world_literal, world_nonzero) = run(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    let (parity_literal, parity_nonzero) = run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS upstream_sy_times_ns: literal-source world={world_literal:e} parity={parity_literal:e}; adapted-nonzero world={world_nonzero:e} parity={parity_nonzero:e}; norm<1e-10"
        );
    }
    world.close();
    runtime.finalize();
}
