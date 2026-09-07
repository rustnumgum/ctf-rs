//! One bounded dense redistribution measurement with an exact key probe.
use ctf::{
    algebra::Arithmetic,
    context::Runtime,
    mapping::{Distribution, Mapping, Topology},
    tensor::Tensor,
};

fn axis_distribution(shape: &[usize], axis: usize, processes: usize) -> Distribution {
    let topology = Topology::new(vec![processes]);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    mappings[axis].augment_physical(&topology, 0);
    Distribution::new(shape.to_vec(), topology, mappings)
}

fn value(key: usize) -> i64 {
    key as i64 * 17 - 5
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let processes = world.size();
    let shape = vec![48, 40];
    let source_distribution = axis_distribution(&shape, 0, processes);
    let destination_distribution = axis_distribution(&shape, 1, processes);

    let mut source = Tensor::new(
        &world,
        source_distribution,
        Arithmetic::<i64>::new(),
    );
    source.transform(|key, entry| *entry = value(key));

    let probe_keys = [0, 1, shape[0] + 2, shape.iter().product::<usize>() - 1];
    let expected: Vec<_> = probe_keys.iter().map(|&key| value(key)).collect();
    let mut checked = source.clone();
    checked.redistribute(destination_distribution.clone());
    assert_eq!(checked.read(&probe_keys), expected);

    world.barrier();
    let start = std::time::Instant::now();
    source.redistribute(destination_distribution);
    world.barrier();
    let elapsed = start.elapsed().as_secs_f64();

    if world.rank() == 0 {
        println!(
            "redistribution shape={}x{} ranks={} elapsed_s={elapsed:.6}",
            shape[0], shape[1], processes
        );
    }

    drop(source);
    drop(checked);
    world.close();
    runtime.finalize();
}
