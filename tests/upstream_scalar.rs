// Port of the active cases in the pinned `test/scalar.cxx` fixture.
//
// Rust represents the source `Scalar<>` through a zero-order dense tensor;
// canonical all-pairs reads expose the single source scalar entry. The
// source's `#if 0` antisymmetric-matrix block is disabled and is therefore
// not an executed test case here.

use ctf::{
    algebra::{Arithmetic, Monoid, Semiring},
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    symmetry::Symmetry::{self, NS, SY},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    tensor::Tensor,
};

fn root_pair(context: &Context<'_>, value: f64) -> Vec<(usize, f64)> {
    if context.rank() == 0 {
        vec![(0, value)]
    } else {
        Vec::new()
    }
}

fn run(context: &Context<'_>) {
    let scalar_distribution = Distribution::cyclic(vec![], context.size());
    let algebra = Arithmetic::<f64>::new();
    let mut a = Tensor::new(context, scalar_distribution.clone(), algebra);

    assert_eq!(a.all_pairs(false), vec![(0, 0.0)]);
    a.write_add(&root_pair(context, 4.2));
    assert_eq!(a.all_pairs(false), vec![(0, 4.2)]);
    assert!(a.all_data()[0] - 4.2 < 1e-9);

    let mut b = Tensor::new(
        context,
        scalar_distribution.clone(),
        Arithmetic::<f64>::new(),
    );
    b.write_add(&root_pair(context, 4.3));
    assert!(b.all_data()[0] - 4.3 < 1e-9);
    b.write_scaled(
        &root_pair(context, 4.2),
        &Arithmetic::<f64>::new().one(),
        &Arithmetic::<f64>::new().zero(),
    );
    assert!(b.all_data()[0] - 4.2 < 1e-9);

    let n = 3usize;
    let topology = Topology::new(vec![context.size()]);
    let mut e = symmetric_tensor(context, vec![n, n], vec![NS, NS]);
    let mut e2 = symmetric_tensor(context, vec![n, n], vec![NS, NS]);
    e.transform(|_, value| *value = 13.1);
    e2.transform(|_, value| *value = 13.1);
    e.sum_from("ij", &e2, "ij", -1.0, 1.0);
    assert!(e.norm2() < 1e-6);

    let d = symmetric_tensor(context, vec![n, 0, n, n], vec![NS, NS, SY, NS]);
    e.transform(|_, value| *value = 13.1);
    let e_operand = e.redistribute(e.distribution().clone());
    e.contract_from_on(
        "ii",
        &d,
        "klij",
        &e_operand,
        "ki",
        topology.clone(),
        "k",
        1.0,
        0.0,
        true,
    )
    .unwrap();
    assert!(e.norm1() > 1e-10);

    let e_operand = e.redistribute(e.distribution().clone());
    e.contract_from_on(
        "ij",
        &d,
        "klij",
        &e_operand,
        "ki",
        topology,
        "k",
        1.0,
        0.0,
        true,
    )
    .unwrap();
    assert!(e.norm1().abs() < 1e-10);
}

fn symmetric_tensor<'c, 'r>(
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

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world);

    let rank = world.rank();
    let parity = world
        .split(Some((rank % 2) as i32), rank as i32)
        .unwrap();
    run(&parity);
    parity.close();

    world.barrier();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS upstream_scalar: scalar tensor, zero-edge contractions, norm tolerances, world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
