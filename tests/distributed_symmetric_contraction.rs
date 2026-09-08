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
) -> SymmetricTensor<'c, 'r, Arithmetic<f64>> {
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
fn close(actual: Vec<f64>, expected: Vec<f64>) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.into_iter().zip(expected) {
        assert!(
            a.is_finite() && (a - b).abs() <= 1e-6,
            "actual={a}, expected={b}"
        );
    }
}
fn value(i: usize, j: usize, kind: Symmetry, shift: f64) -> f64 {
    if i == j && kind != SY {
        return 0.;
    }
    let sign = if i > j && kind == AS { -1. } else { 1. };
    sign * ((i.min(j) + 3 * i.max(j) + 1) as f64 + shift)
}
fn run(c: &Context<'_>) {
    for kind in [SY, AS, SH] {
        let mut a = tensor(c, vec![kind, NS], false);
        a.transform(|key, v| *v = key as f64 + 1.);
        let mut b = tensor(c, vec![kind, NS], true);
        b.transform(|key, v| *v = key as f64 + 2.);
        let mut out = tensor(c, vec![NS, NS], false);
        out.transform(|_, v| *v = 10.);
        out.contract_from_on(
            "ij",
            &a,
            "ik",
            &b,
            "kj",
            Topology::new(vec![c.size()]),
            "k",
            2.,
            3.,
            true,
        )
        .unwrap();
        let expected: Vec<_> = (0..9)
            .map(|key| {
                let i = key % 3;
                let j = key / 3;
                30. + 2.
                    * (0..3)
                        .map(|k| value(i, k, kind, 0.) * value(k, j, kind, 1.))
                        .sum::<f64>()
            })
            .collect();
        close(out.read(&(0..9).collect::<Vec<_>>()), expected);
        let mut dot = tensor(c, vec![], true);
        dot.transform(|_, v| *v = 10.);
        dot.contract_from_on(
            "",
            &a,
            "ij",
            &b,
            "ij",
            Topology::new(vec![c.size()]),
            "i",
            2.,
            3.,
            true,
        )
        .unwrap();
        let expected = 30.
            + 2. * (0..9)
                .map(|key| value(key % 3, key / 3, kind, 0.) * value(key % 3, key / 3, kind, 1.))
                .sum::<f64>();
        close(dot.read(&[0]), vec![expected]);
    }
    let mut sy = tensor(c, vec![SY, NS], false);
    sy.transform(|key, v| *v = key as f64 + 1.);
    let mut as_ = tensor(c, vec![AS, NS], true);
    as_.transform(|key, v| *v = key as f64 + 2.);
    let mut scalar = tensor(c, vec![], true);
    scalar.transform(|_, v| *v = 10.);
    scalar
        .contract_from_on(
            "",
            &sy,
            "ij",
            &as_,
            "ij",
            Topology::new(vec![c.size()]),
            "i",
            2.,
            3.,
            true,
        )
        .unwrap();
    close(scalar.read(&[0]), vec![30.]);
    let mut hadamard = tensor(c, vec![SY, NS], false);
    hadamard
        .contract_from_on(
            "ij",
            &sy,
            "ij",
            &sy,
            "ij",
            Topology::new(vec![c.size()]),
            "i",
            1.,
            0.,
            true,
        )
        .unwrap();
    close(
        hadamard.read(&[0, 3, 4, 6, 7, 8]),
        vec![1., 16., 25., 49., 64., 81.],
    );
    let mut diagonal = tensor(c, vec![NS, NS], true);
    diagonal.transform(|_, v| *v = 10.);
    diagonal
        .contract_from_on(
            "ii",
            &sy,
            "ii",
            &sy,
            "ii",
            Topology::new(vec![c.size()]),
            "i",
            2.,
            3.,
            true,
        )
        .unwrap();
    close(
        diagonal.read(&(0..9).collect::<Vec<_>>()),
        vec![32., 10., 10., 10., 80., 10., 10., 10., 192.],
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
            "DIGIT / PASS distributed_symmetric_contraction: full-domain SY/AS/SH products, overcount, mixed cancellation, diagonals; atol=1e-6; world+parity"
        );
    }
    world.close();
    drop(universe);
}
