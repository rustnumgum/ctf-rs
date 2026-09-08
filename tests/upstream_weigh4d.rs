//! Bounded port of the active NS/SY/AS cases in pinned `test/weigh_4D.cxx`.

use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, *},
};

const N: usize = 3;
const RELATIVE_TOLERANCE: f64 = 1.0e-10;
const NONZERO_CUTOFF: f64 = 1.0e-10;

struct Drand48(u64);

impl Drand48 {
    fn seeded(seed: u64) -> Self {
        Self((seed << 16) | 0x330e)
    }

    fn next(&mut self) -> f64 {
        self.0 = (self.0.wrapping_mul(0x5deece66d).wrapping_add(0xb)) & ((1 << 48) - 1);
        self.0 as f64 / (1u64 << 48) as f64
    }
}

fn tensor<'c, 'r>(
    context: &'c Context<'r>,
    links: Vec<Symmetry>,
) -> SymmetricTensor<'c, 'r, Arithmetic<f64>> {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; 4];
    mappings[0].augment_physical(&topology, 0);
    for mapping in &mut mappings {
        mapping.augment_virtual(2 * context.size());
    }
    SymmetricTensor::new(
        context,
        SymmetricDistribution::new(Distribution::new(vec![N; 4], topology, mappings), links),
        Arithmetic::new(),
    )
}

fn run_case(context: &Context<'_>, kind: Symmetry) {
    let mut a = tensor(context, vec![kind, NS, kind, NS]);
    let mut b = tensor(context, vec![kind, NS, kind, NS]);
    let mut c = tensor(context, vec![kind, NS, kind, NS]);

    // Match the source's single srand48(13*rank) stream across A, B, and C.
    let mut random = Drand48::seeded(13 * context.rank() as u64);
    a.transform(|_, value| *value = random.next() - 0.5);
    b.transform(|_, value| *value = random.next() - 0.5);
    c.transform(|_, value| *value = random.next() - 0.5);
    let original_a = a.local_pairs();

    let topology = Topology::new(vec![context.size()]);
    c.contract_from_on(
        "ijkl",
        &a,
        "ijkl",
        &b,
        "klij",
        topology.clone(),
        "i",
        1.0,
        0.0,
        true,
    )
    .unwrap();

    // The source updates C in place while reading its old values.  Keep an
    // ownership-safe same-distribution copy for the function contraction.
    let source_c = c.redistribute(c.distribution().clone());
    c.contract_function_from_on(
        "ijkl",
        &source_c,
        "ijkl",
        &b,
        "klij",
        topology,
        "i",
        1.0,
        0.0,
        true,
        |left: &f64, right: &f64| *left / *right,
    )
    .unwrap();

    let keys: Vec<_> = original_a.iter().map(|&(key, _)| key).collect();
    let values = c.read(&keys);
    for ((key, expected), actual) in original_a.into_iter().zip(values) {
        assert!(
            expected.is_finite() && actual.is_finite(),
            "non-finite key {key}"
        );
        // Preserve the source criterion's signed denominator expression.
        if expected.abs() > NONZERO_CUTOFF
            && (actual - expected).abs() / expected > RELATIVE_TOLERANCE
        {
            panic!("weigh_4D {kind:?} key {key}: expected {expected}, actual {actual}");
        }
    }
}

fn run(context: &Context<'_>) {
    for kind in [NS, SY, AS] {
        run_case(context, kind);
    }
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    run(&world);

    let rank = world.rank();
    let parity = world.split(Some((rank % 2) as i32), rank as i32).unwrap();
    run(&parity);
    parity.close();

    if rank == 0 {
        println!(
            "DIGIT / PASS upstream_weigh4d: source NS/SY/AS C=A*B and in-place divide; signed relative criterion; world+parity"
        );
    }
    world.close();
    drop(universe);
}
