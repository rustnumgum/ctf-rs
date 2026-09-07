//! Bounded port of every active symmetry branch in pinned `test/gemm_4D.cxx`.

use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, *},
};

const N: usize = 7;

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
    kind: Symmetry,
) -> SymmetricTensor<'c, 'r, Arithmetic<f64>> {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; 4];
    mappings[0].augment_physical(&topology, 0);
    for mapping in &mut mappings {
        mapping.augment_virtual(2 * context.size());
    }
    SymmetricTensor::new(
        context,
        SymmetricDistribution::new(
            Distribution::new(vec![N; 4], topology, mappings),
            vec![kind, NS, kind, NS],
        ),
        Arithmetic::new(),
    )
}

fn run_case(context: &Context<'_>, kind: Symmetry) {
    let mut a = tensor(context, kind);
    let mut b = tensor(context, kind);
    let mut c = tensor(context, kind);
    let mut random = Drand48::seeded(13 * context.rank() as u64);
    for operand in [&mut a, &mut b, &mut c] {
        operand.transform(|_, value| *value = random.next() - 0.5);
    }

    let mut ab = tensor(context, kind);
    ab.contract_from("ijkl", &a, "ijmn", &b, "mnkl", 1.0, 0.0, true)
        .unwrap();
    let mut left = tensor(context, kind);
    left.contract_from("ijkl", &ab, "ijmn", &c, "mnkl", 1.0, 0.0, true)
        .unwrap();

    let mut bc = tensor(context, kind);
    bc.contract_from("ijkl", &b, "ijmn", &c, "mnkl", 1.0, 0.0, true)
        .unwrap();
    let mut right = tensor(context, kind);
    right
        .contract_from("ijkl", &a, "ijmn", &bc, "mnkl", 1.0, 0.0, true)
        .unwrap();

    for ((key, actual), (other, expected)) in
        left.local_pairs().into_iter().zip(right.local_pairs())
    {
        assert_eq!(key, other);
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "gemm_4D {kind:?} key {key}: left={actual}, right={expected}"
        );
    }
}

fn run(context: &Context<'_>) {
    for kind in [NS, SY, AS, SH] {
        run_case(context, kind);
    }
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world);

    let rank = world.rank();
    let parity = world.split(Some((rank % 2) as i32), rank as i32).unwrap();
    run(&parity);
    parity.close();

    if rank == 0 {
        println!(
            "DIGIT / PASS upstream_gemm4d: source NS/SY/AS/SH associativity; n=7; abs(error)<1e-6; world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
