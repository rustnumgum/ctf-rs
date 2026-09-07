//! Bounded port of pinned `studies/fast_sym_4D.cxx`.

use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, *},
};

const N: usize = 6;

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
    order: usize,
    links: Vec<Symmetry>,
) -> SymmetricTensor<'c, 'r, Arithmetic<f64>> {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; order];
    mappings[0].augment_physical(&topology, 0);
    for mapping in &mut mappings {
        mapping.augment_virtual(2 * context.size());
    }
    SymmetricTensor::new(
        context,
        SymmetricDistribution::new(Distribution::new(vec![N; order], topology, mappings), links),
        Arithmetic::new(),
    )
}

fn run(context: &Context<'_>) -> f64 {
    let sh4 = vec![SH, NS, NS, NS];
    let sy5 = vec![SY, SY, NS, NS, NS];
    let mut a = tensor(context, 4, sh4.clone());
    let mut b = tensor(context, 4, sh4.clone());
    let mut random = Drand48::seeded(context.rank() as u64 * 347 + 23);
    a.transform(|_, value| *value = 2.0 * random.next() - 1.0);
    b.transform(|_, value| *value = 2.0 * random.next() - 1.0);

    let topology = Topology::new(vec![context.size()]);
    let mut answer = tensor(context, 4, sh4.clone());
    answer
        .contract_from_on(
            "ijab",
            &a,
            "ikal",
            &b,
            "kjlb",
            topology.clone(),
            "i",
            1.0,
            0.0,
            true,
        )
        .unwrap();

    let mut a_replicated = tensor(context, 5, sy5.clone());
    a_replicated.sum_from("ijkal", &a, "ijal", 1.0, 1.0);
    let mut b_replicated = tensor(context, 5, sy5.clone());
    b_replicated.sum_from("ijklb", &b, "ijlb", 1.0, 1.0);
    let mut z = tensor(context, 5, sy5);
    z.contract_from_on(
        "ijkab",
        &a_replicated,
        "ijkal",
        &b_replicated,
        "ijklb",
        topology.clone(),
        "i",
        1.0,
        1.0,
        true,
    )
    .unwrap();

    let mut c = tensor(context, 4, sh4.clone());
    c.sum_from("ijab", &z, "ijkab", 1.0, 1.0);
    c.contract_from_on(
        "ijab",
        &a,
        "ijal",
        &b,
        "ijlb",
        topology.clone(),
        "i",
        -(N as f64),
        1.0,
        true,
    )
    .unwrap();
    c.contract_from_on(
        "ijab",
        &a,
        "ikal",
        &b,
        "iklb",
        topology.clone(),
        "i",
        -1.0,
        1.0,
        true,
    )
    .unwrap();
    c.contract_from_on(
        "ijab",
        &a,
        "ikal",
        &b,
        "ijlb",
        topology.clone(),
        "i",
        -1.0,
        1.0,
        true,
    )
    .unwrap();
    c.contract_from_on(
        "ijab", &a, "ijal", &b, "iklb", topology, "i", -1.0, 1.0, true,
    )
    .unwrap();

    let mut difference = tensor(context, 4, sh4);
    difference.sum_from("ijab", &c, "ijab", 1.0, 1.0);
    difference.sum_from("ijab", &answer, "ijab", -1.0, 1.0);
    let norm = difference.norm2();
    assert!(
        norm.is_finite() && norm <= 1.0e-10,
        "fast_sym_4D norm={norm}"
    );
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
            "DIGIT / PASS upstream_fast_sym_4D: source SH fast symmetric 4D contraction; n=6; world_norm={world_norm:e}; parity_norm={parity_norm:e}; norm<=1e-10"
        );
    }
    world.close();
    runtime.finalize();
}
