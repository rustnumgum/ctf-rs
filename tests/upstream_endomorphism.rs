// Deterministic port of pinned CTF test/endomorphism.cxx.
use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::Distribution,
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
    a.transform_indexed("ijkl", |value| *value = *value * *value * *value);

    let keys: Vec<_> = (0..length).collect();
    let values = a.read(&keys);
    for (key, value) in values.into_iter().enumerate() {
        let old = initial(key);
        let expected = old * old * old;
        assert!((expected - value).abs() < 1.0e-6,
            "source endomorphism mismatch at key {key}: expected {expected}, got {value}");
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
        println!("DIGIT / PASS upstream_endomorphism: A=A^3; abs<1e-6; world+parity");
    }
    world.close();
    runtime.finalize();
}
