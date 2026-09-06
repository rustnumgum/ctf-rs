//! Exact source-layout and symmetric I/O acceptance; no floating tolerances.
use ctf::{algebra::Arithmetic, context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology}, symmetry::Symmetry,
    symmetric_distribution::SymmetricDistribution, symmetric_tensor::SymmetricTensor};

fn layout(context: &Context<'_>, n: usize, order: usize, kind: Symmetry,
          replicated: bool) -> SymmetricDistribution {
    let topology = Topology::new(if replicated { vec![context.size()] }
        else if context.size() == 4 { vec![2, 2] } else { vec![context.size(), 1] });
    let mut mappings = vec![Mapping::Unmapped; order];
    let phase = if replicated { 2 } else { topology.dimensions[0] };
    if !replicated {
        mappings[0].augment_physical(&topology, 0);
        mappings[1].augment_physical(&topology, 1);
    }
    for mapping in &mut mappings { mapping.augment_virtual(phase); }
    let mut links = vec![kind; order];
    links[order - 1] = Symmetry::NS;
    SymmetricDistribution::new(Distribution::new(vec![n; order], topology, mappings), links)
}

fn expected(i: usize, j: usize, kind: Symmetry) -> i64 {
    if i == j && kind != Symmetry::SY { return 0; }
    let value = 1 + i.min(j) as i64 + 10 * i.max(j) as i64;
    if kind == Symmetry::AS && i > j { -value } else { value }
}

fn matrix(context: &Context<'_>, n: usize, kind: Symmetry) {
    let source = layout(context, n, 2, kind, false);
    if context.size() == 4 && n == 5 {
        assert_eq!(source.local_len(), 6);
        if context.rank() == 1 {
            assert_eq!(source.local_pairs(1), vec![(1, 11), (3, 21), (4, 23)]);
        }
    }
    let mut tensor = SymmetricTensor::new(context, source, Arithmetic::<i64>::new());
    let pairs: Vec<_> = (0..n*n).filter_map(|key| {
        let i = key % n;
        let j = key / n;
        // Write reversed coordinates: AS input must have reversed sign.
        (i >= j && key % context.size() == context.rank())
            .then_some((key, expected(i, j, kind)))
    }).collect();
    tensor.write_add(&pairs);
    let keys: Vec<_> = (0..n*n).rev().chain([0, 0]).collect();
    let reference: Vec<_> = keys.iter().map(|&key| expected(key % n, key / n, kind)).collect();
    assert_eq!(tensor.read(&keys), reference);
    let valid = tensor.distribution().local_pairs(context.rank());
    for (offset, value) in tensor.local_storage().iter().enumerate() {
        if !valid.iter().any(|&(slot, _)| slot == offset) { assert_eq!(*value, 0); }
    }
    assert!(tensor.local_pairs().iter().all(|&(key, _)| {
        let i = key % n; let j = key / n;
        if kind == Symmetry::SY { i <= j } else { i < j }
    }));
    let replica = tensor.redistribute(layout(context, n, 2, kind, true));
    assert_eq!(replica.read(&keys), reference);
    let mut restored = replica.redistribute(layout(context, n, 2, kind, false));
    restored.transform(|_, value| *value *= 2);
    assert_eq!(restored.read(&keys), reference.iter().map(|v| 2*v).collect::<Vec<_>>());
    // Each rank contributes; duplicate and symmetry-equivalent keys coalesce.
    if n > 1 {
        restored.write_add(&[(n, 3), (n, 4), (1, if kind == Symmetry::AS { -5 } else { 5 })]);
        assert_eq!(restored.read(&[n]), vec![2*expected(0, 1, kind) + 12*context.size() as i64]);
    }
}

fn triple(context: &Context<'_>) {
    let mut tensor = SymmetricTensor::new(context,
        layout(context, 3, 3, Symmetry::AS, false), Arithmetic::<i64>::new());
    let pairs = if context.rank() == 0 { vec![(21, 7), (0, 1000)] } else { vec![] };
    tensor.write_add(&pairs); // canonical (0,1,2), structural-zero (0,0,0).
    assert_eq!(tensor.read(&[21, 15, 19, 7, 11, 5, 0]), vec![7, -7, -7, 7, 7, -7, 0]);
    let redistributed = tensor.redistribute(layout(context, 3, 3, Symmetry::AS, true));
    assert_eq!(redistributed.read(&[21, 5]), vec![7, -7]);
}

fn run(context: &Context<'_>) {
    for kind in [Symmetry::SY, Symmetry::AS, Symmetry::SH] {
        matrix(context, 5, kind);
        matrix(context, 1, kind);
    }
    triple(context);
}
fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world);
    let parity = world.split(Some((world.rank() % 2) as i32), world.rank() as i32).unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 { println!("DIGIT / PASS distributed_symmetric_io: SY/AS/SH packed holes, parity, replicas, virtual redistribution; exact i64; world+parity"); }
    world.close();
    runtime.finalize();
}
