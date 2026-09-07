//! Deterministic port of pinned CTF test/univar_function.cxx.
use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Topology},
    tensor::Tensor,
};

const N: usize = 5;

fn initial(key: usize) -> f64 {
    ((key * 37 + 11) % 101) as f64 / 100.0 - 0.5
}

fn run(context: &Context<'_>) {
    let shape = vec![N + 1, N, N + 2, N + 3];
    let length: usize = shape.iter().product();
    let mut a = Tensor::new(
        context,
        Distribution::cyclic(shape, context.size()),
        Arithmetic::<f64>::new(),
    );
    a.transform(|key, value| *value = initial(key));
    let start = a.clone();
    a.sum_function_from(
        "ijkl",
        &start,
        "ijkl",
        Topology::new(vec![context.size()]),
        1.0,
        0.25,
        |value| value * value * value * value,
    ).unwrap();

    let keys: Vec<_> = (0..length).collect();
    let values = a.read(&keys);
    for (key, value) in values.into_iter().enumerate() {
        let old = initial(key);
        let expected = 0.25 * old + old * old * old * old;
        assert!((expected - value).abs() < 1.0e-6,
            "source univar_function mismatch at key {key}: expected {expected}, got {value}");
    }
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world);
    let parity = world.split(Some((world.rank() % 2) as i32), world.rank() as i32).unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!("DIGIT / PASS upstream_univar_function: A=0.25*A+A^4; abs<1e-6; world+parity");
    }
    world.close();
    runtime.finalize();
}
