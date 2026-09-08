//! Bounded port of pinned `studies/fast_diagram.cxx`.

use ctf::{
    algebra::Arithmetic,
    context::Context,
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
    let mut t = tensor(context, 4, sh4.clone());
    let mut v = tensor(context, 4, sh4.clone());
    let mut random = Drand48::seeded(173 * context.rank() as u64);
    t.transform(|_, value| *value = random.next());
    v.transform(|_, value| *value = random.next());

    let topology = Topology::new(vec![context.size()]);
    let mut z_ns = tensor(context, 4, vec![NS; 4]);
    z_ns.contract_from_on(
        "afin",
        &t,
        "aeim",
        &v,
        "efmn",
        topology.clone(),
        "a",
        1.0,
        0.0,
        true,
    )
    .unwrap();
    let mut w_answer = tensor(context, 4, sh4.clone());
    w_answer
        .contract_from_on(
            "abij",
            &z_ns,
            "afin",
            &t,
            "fbnj",
            topology.clone(),
            "a",
            1.0,
            0.0,
            true,
        )
        .unwrap();

    let mut z_as = tensor(context, 4, vec![AS, NS, NS, NS]);
    z_as.contract_from_on(
        "afin",
        &t,
        "aeim",
        &v,
        "efmn",
        topology.clone(),
        "a",
        1.0,
        0.0,
        true,
    )
    .unwrap();
    // Rust's NS->AS orbit projection returns the unnormalized difference.
    // Pinned CTF's packed intermediate carries the half-projection used by
    // this source identity, whose later AS term has no explicit 0.5 factor.
    z_as.scale(&0.5);
    let mut z_sh = tensor(context, 4, sh4.clone());
    z_sh.contract_from_on(
        "afin",
        &t,
        "aeim",
        &v,
        "efmn",
        topology.clone(),
        "a",
        1.0,
        0.0,
        true,
    )
    .unwrap();
    // As above, normalize the off-diagonal SH orbit sum to the packed
    // half-projection expected by the source's coefficient-free recombination.
    z_sh.scale(&0.5);
    let mut z_diagonal = tensor(context, 3, vec![NS; 3]);
    z_diagonal
        .contract_from_on(
            "ain",
            &t,
            "aeim",
            &v,
            "eamn",
            topology.clone(),
            "a",
            1.0,
            0.0,
            true,
        )
        .unwrap();

    let mut w = tensor(context, 4, sh4);
    w.contract_from_on(
        "abij",
        &z_as,
        "afin",
        &t,
        "fbnj",
        topology.clone(),
        "a",
        1.0,
        0.0,
        true,
    )
    .unwrap();
    w.contract_from_on(
        "abij",
        &z_sh,
        "afin",
        &t,
        "fbnj",
        topology.clone(),
        "a",
        1.0,
        1.0,
        true,
    )
    .unwrap();
    w.contract_from_on(
        "abij",
        &z_diagonal,
        "ain",
        &t,
        "abnj",
        topology,
        "a",
        1.0,
        1.0,
        true,
    )
    .unwrap();
    w.sum_from("abij", &w_answer, "abij", -1.0, 1.0);

    let norm = w.norm2();
    assert!(
        norm.is_finite() && norm <= 1.0e-10,
        "fast_diagram norm={norm}"
    );
    norm
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let world_norm = run(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    let parity_norm = run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS upstream_fast_diagram: source AS/SH/diagonal decomposition; n=5; world_norm={world_norm:e}; parity_norm={parity_norm:e}; norm<=1e-10"
        );
    }
    world.close();
    drop(universe);
}
