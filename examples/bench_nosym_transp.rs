//! One bounded local nonsymmetric transpose measurement with an exact probe.
use ctf::{
    algebra::Arithmetic,
    context::Runtime,
    mapping::Distribution,
    tensor::Tensor,
};

fn value(key: usize) -> f64 {
    key as f64 + 0.25
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let processes = world.size();

    let probe_shape = [3, 4];
    let mut probe = Tensor::new(
        &world,
        Distribution::cyclic(probe_shape.to_vec(), processes),
        Arithmetic::<f64>::new(),
    );
    probe.transform(|key, entry| *entry = value(key));
    let transposed_probe = probe.permute_axes(&[1, 0]);
    let probe_keys: Vec<_> = (0..probe_shape.iter().product()).collect();
    let expected: Vec<_> = probe_keys
        .iter()
        .map(|&key| {
            let source_axis_1 = key % probe_shape[1];
            let source_axis_0 = key / probe_shape[1];
            value(source_axis_0 + probe_shape[0] * source_axis_1)
        })
        .collect();
    assert_eq!(transposed_probe.read(&probe_keys), expected);
    assert_eq!(
        transposed_probe
            .permute_axes(&[1, 0])
            .read(&(0..probe_shape.iter().product()).collect::<Vec<_>>()),
        (0..probe_shape.iter().product())
            .map(value)
            .collect::<Vec<_>>()
    );

    let shape = vec![192, 160];
    let mut source = Tensor::new(
        &world,
        Distribution::cyclic(shape.clone(), processes),
        Arithmetic::<f64>::new(),
    );
    source.transform(|key, entry| *entry = value(key));

    world.barrier();
    let start = std::time::Instant::now();
    let transposed = source.permute_axes(&[1, 0]);
    world.barrier();
    let elapsed = start.elapsed().as_secs_f64();

    if world.rank() == 0 {
        println!(
            "nosym_transpose shape={}x{} ranks={} elapsed_s={elapsed:.6}",
            shape[0], shape[1], processes
        );
    }

    drop(transposed);
    drop(source);
    drop(transposed_probe);
    drop(probe);
    world.close();
    runtime.finalize();
}
