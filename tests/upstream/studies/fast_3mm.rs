//! Bounded port of pinned `studies/fast_3mm.cxx`.

use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, *},
};

const N: usize = 5;

struct Drand48(u64);

impl Drand48 {
    fn seeded(seed: u64) -> Self {
        Self((seed << 16) | 0x330e)
    }

    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(0x5deece66d).wrapping_add(11) & ((1 << 48) - 1);
        self.0 as f64 / (1u64 << 48) as f64
    }
}

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

fn matrix<'c, 'r>(
    context: &'c Context<'r>,
    link: Symmetry,
) -> SymmetricTensor<'c, 'r, Arithmetic<f64>> {
    tensor(context, vec![N, N], vec![link, NS])
}

fn run(context: &Context<'_>) -> f64 {
    let mut t = matrix(context, NS);
    let mut v = matrix(context, NS);
    let mut random = Drand48::seeded(173 * context.rank() as u64);
    t.transform(|_, value| *value = random.next());
    v.transform(|_, value| *value = random.next());

    let topology = Topology::new(vec![context.size()]);
    let mut z_ns = matrix(context, NS);
    z_ns.contract_from_on(
        "af",
        &t,
        "ae",
        &v,
        "ef",
        topology.clone(),
        "a",
        1.0,
        0.0,
        true,
    )
    .unwrap();
    let mut w_answer = matrix(context, SH);
    w_answer
        .contract_from_on(
            "ab",
            &z_ns,
            "af",
            &t,
            "fb",
            topology.clone(),
            "a",
            1.0,
            0.0,
            true,
        )
        .unwrap();

    let mut z_as = matrix(context, AS);
    z_as.contract_from_on(
        "af",
        &t,
        "ae",
        &v,
        "ef",
        topology.clone(),
        "a",
        1.0,
        0.0,
        true,
    )
    .unwrap();
    let mut z_sy = matrix(context, SY);
    z_sy.contract_from_on(
        "af",
        &t,
        "ae",
        &v,
        "ef",
        topology.clone(),
        "a",
        1.0,
        0.0,
        true,
    )
    .unwrap();
    // The Rust NS->SY orbit sum visits a diagonal orbit twice.  Pinned CTF's
    // packed contraction keeps that coincidence surface once; the source's
    // explicit 0.5*Z_SY["aa"] correction relies on that convention.
    z_sy.scale_indexed("aa", &0.5);

    let mut w = matrix(context, SH);
    w.contract_from_on(
        "ab",
        &z_sy,
        "af",
        &t,
        "fb",
        topology.clone(),
        "a",
        0.5,
        0.0,
        true,
    )
    .unwrap();
    w.contract_from_on(
        "ab",
        &z_sy,
        "aa",
        &t,
        "ab",
        topology.clone(),
        "a",
        0.5,
        1.0,
        true,
    )
    .unwrap();
    w.contract_from_on("ab", &z_as, "af", &t, "fb", topology, "a", 0.5, 1.0, true)
        .unwrap();
    w.sum_from("ab", &w_answer, "ab", -1.0, 1.0);

    let norm = w.norm2();
    assert!(norm.is_finite() && norm <= 1.0e-10, "fast_3mm norm={norm}");
    norm
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let world_norm = run(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    let parity_norm = run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS upstream_fast_3mm: source SY/AS decomposition; n=5; world_norm={world_norm:e}; parity_norm={parity_norm:e}; norm<=1e-10"
        );
    }
    world.close();
    runtime.finalize();
}
