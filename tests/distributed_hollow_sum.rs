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
    n: usize,
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
            Distribution::new(vec![n; links.len()], topology, maps),
            links,
        ),
        Arithmetic::new(),
    )
}
fn matrices(c: &Context<'_>) {
    let keys: Vec<_> = (0..9).collect();
    for kind in [AS, SH] {
        let mut a = tensor(c, 3, vec![kind, NS], false);
        a.transform(|key, value| *value = key as i64 + 1);
        let mut ns = tensor(c, 3, vec![NS, NS], true);
        ns.transform(|_, value| *value = 10);
        ns.sum_hollow_from("ij", &a, "ij", 2, 3);
        let expected: Vec<_> = keys
            .iter()
            .map(|&key| {
                let i = key % 3;
                let j = key / 3;
                if i == j {
                    30
                } else {
                    let sign = if i > j && kind == AS { -1 } else { 1 };
                    30 + 2 * sign * (i.min(j) + 3 * i.max(j) + 1) as i64
                }
            })
            .collect();
        assert_eq!(ns.read(&keys), expected);
        let mut transposed = tensor(c, 3, vec![kind, NS], true);
        transposed.sum_hollow_from("ij", &a, "ji", 1, 0);
        let sign = if kind == AS { -1 } else { 1 };
        assert_eq!(
            transposed.read(&[3, 6, 7]),
            vec![4 * sign, 7 * sign, 8 * sign]
        );
        let mut reduced = tensor(c, 0, vec![], true);
        reduced.transform(|_, value| *value = 10);
        reduced.sum_hollow_from("", &a, "ij", 2, 3);
        assert_eq!(reduced.read(&[0]), vec![if kind == AS { 30 } else { 106 }]);
        let other = if kind == AS { SH } else { AS };
        let mut mismatch = tensor(c, 3, vec![other, NS], true);
        mismatch.transform(|_, value| *value = 10);
        mismatch.sum_hollow_from("ij", &a, "ij", 2, 3);
        assert_eq!(mismatch.read(&[3, 6, 7]), vec![30; 3]);
    }
    let mut a = tensor(c, 3, vec![NS, NS], true);
    a.transform(|key, value| *value = key as i64 + 1);
    for kind in [AS, SH] {
        let mut out = tensor(c, 3, vec![kind, NS], false);
        out.transform(|_, value| *value = 10);
        out.sum_hollow_from("ij", &a, "ij", 2, 3);
        assert_eq!(
            out.read(&[3, 6, 7]),
            if kind == AS {
                vec![34, 38, 34]
            } else {
                vec![42, 50, 58]
            }
        );
    }
}
fn triples(c: &Context<'_>) {
    let mut ns = tensor(c, 3, vec![NS, NS, NS], false);
    ns.transform(|key, value| *value = if key == 21 { 7 } else { 0 });
    for kind in [AS, SH] {
        let mut packed = tensor(c, 3, vec![kind, kind, NS], true);
        packed.sum_hollow_from("ijk", &ns, "ijk", 1, 0);
        assert_eq!(packed.read(&[21]), vec![7]);
        let mut expanded = tensor(c, 3, vec![NS, NS, NS], false);
        expanded.sum_hollow_from("ijk", &packed, "ijk", 1, 0);
        assert_eq!(
            expanded.read(&[21, 15, 19, 7, 11, 5, 0]),
            if kind == AS {
                vec![7, -7, -7, 7, 7, -7, 0]
            } else {
                vec![7, 7, 7, 7, 7, 7, 0]
            }
        );
    }
}
fn run(c: &Context<'_>) {
    matrices(c);
    triples(c);
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
            "DIGIT / PASS distributed_hollow_sum: unfolding, projection, AS cancellation, SH factors; exact i64; world+parity"
        );
    }
    world.close();
    drop(universe);
}
