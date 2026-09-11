// Adapted from pinned test/python/test_sparse.py::test_sparse_SY.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    random::Generator,
    sparse_symmetric::SparseSymmetricTensor,
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{NS, SY},
};

fn check(label: &str, delta: f64, context: &Context<'_>) {
    if context.rank() == 0 {
        println!("sparse_sy {label}: sum(abs(diff))={delta:e}; bound=1e-14");
    }
    assert!(delta < 1e-14, "{label}: sum(abs(diff))={delta:e}");
}

fn run(context: &Context<'_>) {
    let fixtures = [
        (vec![4, 4], vec![SY, NS]),
        (vec![3, 3, 3], vec![NS, SY, NS]),
        (vec![4; 4], vec![NS, NS, SY, NS]),
        (vec![4; 4], vec![SY, NS, NS, NS]),
        (vec![4; 4], vec![SY, NS, SY, NS]),
        (vec![4; 4], vec![SY, SY, SY, NS]),
    ];
    let mut generator = Generator::new(5330 + context.rank() as u64);
    for (shape, links) in fixtures {
        let topology = Topology::new(vec![context.size()]);
        let mappings = (0..shape.len()).map(|axis| {
            let mut mapping = Mapping::Unmapped;
            if axis == 0 { mapping.augment_physical(&topology, 0); }
            else { mapping.augment_virtual(context.size()); }
            mapping
        }).collect();
        let distribution = SymmetricDistribution::new(
            Distribution::new(shape, topology, mappings), links);
        let mut x = SymmetricTensor::new(context, distribution, Arithmetic::<f64>::new());
        x.fill_random(1., 1., &mut generator);
        let y = SparseSymmetricTensor::from_dense(&x, |value| value.abs() > 0.);
        let keys: Vec<_> = (0..x.distribution().distribution().global_len()).collect();
        let dense_values = x.read(&keys);
        let sparse_values = y.read(&keys);
        check("X versus Y", dense_values.iter().zip(sparse_values)
            .map(|(a,b)| (a-b).abs()).sum(), context);
        let dense_norm = x.norm2();
        x.add_sparse(&y, -1.);
        check("X-Y versus zero", x.read(&keys).iter().map(|v| v.abs()).sum(), context);
        check("vecnorm(X) versus vecnorm(Y)", (dense_norm - y.norm2()).abs(), context);
    }
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = Context::world(&universe);
    run(&world);
    let parity = world.split(Some((world.rank() % 2) as i32), world.rank() as i32).unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!("DIGIT / PASS sparse_sy: six source shapes/symmetries; three strict allclose comparisons; world+parity");
    }
    world.close();
    drop(universe);
}
