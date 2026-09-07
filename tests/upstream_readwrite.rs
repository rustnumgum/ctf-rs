// Port of the pinned `test/readwrite_test.cxx` diagonal-write fixture.
//
// Indexed CTF expressions are represented by the Rust key-based write/read
// APIs. The source's NS/SY/SH/AS diagonal equations and `1.E-10` checks are
// kept unchanged; the source literal named `shape_AS4` is also SH, so that
// literal is preserved below.

use ctf::{
    algebra::{Arithmetic, Monoid, Semiring},
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    symmetry::Symmetry::{self, NS, SH, SY},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    tensor::Tensor,
};

fn diagonal_pairs(n: usize) -> Vec<(usize, f64)> {
    (0..n)
        .map(|i| {
            let key = i + i * n + i * n * n + i * n * n * n;
            (key, (i + 1) as f64)
        })
        .collect()
}

fn symmetric_distribution(context: &Context<'_>, links: Vec<Symmetry>) -> SymmetricDistribution {
    let topology = Topology::new(vec![context.size()]);
    let mut first = Mapping::Unmapped;
    first.augment_physical(&topology, 0);
    first.augment_virtual(2 * context.size());
    let mut second = Mapping::Unmapped;
    second.augment_virtual(2 * context.size());
    let third = second.clone();
    let fourth = second.clone();
    let mappings = vec![first, second, third, fourth];
    SymmetricDistribution::new(Distribution::new(vec![3, 3, 3, 3], topology, mappings), links)
}

fn logical_sum(tensor: &SymmetricTensor<'_, '_, Arithmetic<f64>>) -> f64 {
    tensor.unpack(Distribution::cyclic(tensor.distribution().distribution().shape.clone(),tensor.context().size())).reduce()
}

fn run(context: &Context<'_>, n: usize) {
    let shape = vec![n, n, n, n];
    let expected = n as f64 * (n as f64 + 1.0) / 2.0;
    let values = if context.rank() == 0 {
        diagonal_pairs(n)
    } else {
        Vec::new()
    };

    let algebra = Arithmetic::<f64>::new();
    let mut a_ns = Tensor::new(
        context,
        Distribution::cyclic(shape.clone(), context.size()),
        algebra,
    );
    a_ns.write_add(&values);
    assert!((a_ns.reduce() - expected).abs() <= 1e-10);

    let mut a_sy = SymmetricTensor::new(
        context,
        symmetric_distribution(context, vec![SY, NS, SY, NS]),
        Arithmetic::<f64>::new(),
    );
    let mut a_sh = SymmetricTensor::new(
        context,
        symmetric_distribution(context, vec![SH, NS, SH, NS]),
        Arithmetic::<f64>::new(),
    );
    // Preserve the pinned source's shape_AS4 literal, which is SH rather than AS.
    let mut a_as = SymmetricTensor::new(
        context,
        symmetric_distribution(context, vec![SH, NS, SH, NS]),
        Arithmetic::<f64>::new(),
    );
    a_sy.write_add(&values);
    a_sh.write_add(&values);
    a_as.write_add(&values);
    assert!((logical_sum(&a_sy) - expected).abs() <= 1e-10);
    assert_eq!(logical_sum(&a_as), 0.0);
    assert_eq!(logical_sum(&a_sh), 0.0);

    let square_roots: Vec<_> = values
        .iter()
        .map(|&(key, value)| (key, value.sqrt()))
        .collect();
    a_sy.write_scaled(
        &square_roots,
        &Arithmetic::<f64>::new().one(),
        &Arithmetic::<f64>::new().zero(),
    );

    a_ns = a_sy.unpack(Distribution::cyclic(shape,context.size()));
    let mut dot = Tensor::new(context,Distribution::cyclic(vec![],context.size()),algebra);
    dot.contract_from("",&a_ns,"ijkl",&a_ns,"ijkl",Topology::new(vec![context.size()]),1.0,0.0).unwrap();
    let sum = dot.read(&[0])[0];
    assert!((sum - expected).abs() <= 1e-10);

    let mut dot = SymmetricTensor::new(context,SymmetricDistribution::new(Distribution::cyclic(vec![],context.size()),vec![]),algebra);
    dot.contract_from_on("",&a_sy,"ijkl",&a_sy,"ijkl",Topology::new(vec![context.size()]),"i",1.0,0.0,true).unwrap();
    let sum = dot.read(&[0])[0];
    assert!((sum - expected).abs() <= 1e-10);
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world, 3);

    let rank = world.rank();
    let parity = world
        .split(Some((rank % 2) as i32), rank as i32)
        .unwrap();
    run(&parity, 3);
    parity.close();

    world.barrier();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS upstream_readwrite: NS/SY/SH/AS diagonal writes and self contractions, world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
