//! Source test/repack.cxx canonical-domain copy, extended to strict and partial groups.
use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, *},
};
fn distribution(
    context: &Context<'_>,
    links: Vec<Symmetry>,
    replicas: bool,
) -> SymmetricDistribution {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; links.len()];
    if !replicas {
        mappings[0].augment_physical(&topology, 0);
    }
    for mapping in &mut mappings {
        mapping.augment_virtual(context.size() * 2);
    }
    SymmetricDistribution::new(
        Distribution::new(vec![3; links.len()], topology, mappings),
        links,
    )
}
fn run(context: &Context<'_>) {
    let mut ns = SymmetricTensor::new(
        context,
        distribution(context, vec![NS, NS], false),
        Arithmetic::<i64>::new(),
    );
    ns.transform(|key, value| *value = key as i64 + 1);
    let keys: Vec<_> = (0..9).collect();
    for kind in [SY, AS, SH] {
        let packed = ns.repack_to(distribution(context, vec![kind, NS], true));
        let expected: Vec<_> = keys
            .iter()
            .map(|&key| {
                let i = key % 3;
                let j = key / 3;
                let value = (i.min(j) + 3 * i.max(j) + 1) as i64;
                if i == j && kind != SY {
                    0
                } else if i > j && kind == AS {
                    -value
                } else {
                    value
                }
            })
            .collect();
        assert_eq!(packed.read(&keys), expected);
        let unpacked = packed.repack_to(distribution(context, vec![NS, NS], false));
        let chamber: Vec<_> = keys
            .iter()
            .map(|&key| {
                let i = key % 3;
                let j = key / 3;
                if i < j || (i == j && kind == SY) {
                    key as i64 + 1
                } else {
                    0
                }
            })
            .collect();
        assert_eq!(unpacked.read(&keys), chamber);
    }
    let mut triple = SymmetricTensor::new(
        context,
        distribution(context, vec![NS, NS, NS], false),
        Arithmetic::<i64>::new(),
    );
    triple.transform(|key, value| *value = key as i64 + 1);
    let keys: Vec<_> = (0..27).collect();
    for kind in [SY, AS] {
        let full = triple.repack_to(distribution(context, vec![kind, kind, NS], true));
        let partial = full.repack_to(distribution(context, vec![kind, NS, NS], false));
        let back = partial.repack_to(distribution(context, vec![NS, NS, NS], true));
        let expected: Vec<_> = keys
            .iter()
            .map(|&key| {
                let i = key % 3;
                let j = key / 3 % 3;
                let k = key / 9;
                if (kind == SY && i <= j && j <= k) || (kind == AS && i < j && j < k) {
                    key as i64 + 1
                } else {
                    0
                }
            })
            .collect();
        assert_eq!(back.read(&keys), expected);
    }
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
            "DIGIT / PASS distributed_symmetric_repack: canonical intersection, SY/AS/SH, partial groups, replicas; exact i64; world+parity"
        );
    }
    world.close();
    drop(universe);
}
