use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, *},
};
fn tensor<'c, 'r>(
    c: &'c Context<'r>,
    links: Vec<Symmetry>,
    replicas: bool,
) -> SymmetricTensor<'c, 'r, Arithmetic<i64>> {
    let topology = Topology::new(vec![c.size()]);
    let mut maps = vec![Mapping::Unmapped; links.len()];
    if !replicas && !maps.is_empty() {
        maps[0].augment_physical(&topology, 0);
    }
    for map in &mut maps {
        map.augment_virtual(2 * c.size());
    }
    SymmetricTensor::new(
        c,
        SymmetricDistribution::new(
            Distribution::new(vec![3; links.len()], topology, maps),
            links,
        ),
        Arithmetic::new(),
    )
}
fn run(c: &Context<'_>) {
    let mut a = tensor(c, vec![SY, NS], false);
    a.transform(|key, v| *v = key as i64 + 1);
    let mut b = tensor(c, vec![SY, NS], true);
    b.transform(|key, v| *v = key as i64 + 2);
    let keys: Vec<_> = (0..9).collect();
    let expected: Vec<_> = keys
        .iter()
        .map(|&key| {
            let i = key % 3;
            let j = key / 3;
            let sum: i64 = (0..3)
                .filter(|&k| i <= k && k <= j)
                .map(|k| (i + 3 * k + 1) as i64 * (k + 3 * j + 2) as i64)
                .sum();
            30 + 2 * sum
        })
        .collect();
    for reduced_axis in [true, false] {
        let (topology, physical) = if reduced_axis {
            (Topology::new(vec![c.size()]), "k")
        } else if c.size() == 4 {
            (Topology::new(vec![2, 2]), "ij")
        } else {
            (Topology::new(vec![c.size()]), "i")
        };
        let mut out = tensor(c, vec![NS, NS], false);
        out.transform(|_, v| *v = 10);
        let old_distribution = out.distribution().distribution().clone();
        out.contract_canonical_on("ij", &a, "ik", &b, "kj", topology, physical, 2, 3, true)
            .unwrap();
        assert_eq!(out.read(&keys), expected);
        assert_eq!(out.distribution().distribution(), &old_distribution);
    }
    let mut hadamard = tensor(c, vec![SY, NS], true);
    hadamard.transform(|_, v| *v = 10);
    hadamard
        .contract_canonical_on(
            "ij",
            &a,
            "ij",
            &b,
            "ij",
            Topology::new(vec![c.size()]),
            "i",
            2,
            3,
            true,
        )
        .unwrap();
    assert_eq!(
        hadamard.read(&[0, 3, 4, 6, 7, 8]),
        vec![34, 70, 90, 142, 174, 210]
    );
    // Canonical dot has no symmetry overcounting factor; upper orchestration
    // must supply that separately for a semantic full-domain contraction.
    let mut dot = tensor(c, vec![], true);
    dot.transform(|_, v| *v = 10);
    dot.contract_canonical_on(
        "",
        &a,
        "ij",
        &b,
        "ij",
        Topology::new(vec![c.size()]),
        "i",
        2,
        3,
        true,
    )
    .unwrap();
    assert_eq!(dot.read(&[0]), vec![570]);
    let mut anti = tensor(c, vec![AS, NS], false);
    anti.transform(|key, v| *v = key as i64 + 1);
    let mut out = tensor(c, vec![AS, NS], true);
    out.contract_canonical_on(
        "ij",
        &anti,
        "ij",
        &b,
        "ij",
        Topology::new(vec![c.size()]),
        "j",
        1,
        0,
        true,
    )
    .unwrap();
    assert_eq!(
        out.read(&[0, 3, 4, 6, 7, 8, 1]),
        vec![0, 20, 0, 56, 72, 0, -20]
    );
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
            "DIGIT / PASS distributed_canonical_contraction: explicit physical mapping, packed kernel, broadcasts/reduction, distribution restore; exact i64; world+parity"
        );
    }
    world.close();
    drop(universe);
}
