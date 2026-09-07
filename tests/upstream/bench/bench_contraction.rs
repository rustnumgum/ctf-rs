//! Informational native benchmark corresponding to `bench/bench_contraction.cxx`.
//!
//! The source defaults are retained (`n=4`, three `ik`/`kj` -> `ij`
//! iterations).  Native Rust uses explicit Tensor contraction and reports one
//! world timing; parity executes the same fixed workload for the acceptance
//! driver but is not printed as a second sample.

use std::time::Instant;

use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Topology},
    tensor::Tensor,
};

const N: usize = 4;
const ITERATIONS: usize = 3;

fn distribution(context: &Context<'_>, indices: &str) -> Distribution {
    assert!(indices.is_ascii());
    Distribution::cyclic(vec![N; indices.len()], context.size())
}

fn run(context: &Context<'_>) -> f64 {
    let mut a = Tensor::new(
        context,
        distribution(context, "ik"),
        Arithmetic::<f64>::new(),
    );
    let mut b = Tensor::new(
        context,
        distribution(context, "kj"),
        Arithmetic::<f64>::new(),
    );
    let mut c = Tensor::new(
        context,
        distribution(context, "ij"),
        Arithmetic::<f64>::new(),
    );
    a.transform(|key, value| *value = (1 + key % 5) as f64 / 5.0);
    b.transform(|key, value| *value = (2 + key % 7) as f64 / 7.0);
    c.transform(|_, value| *value = 0.0);
    let topology = Topology::new(vec![context.size()]);

    context.barrier();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        c.contract_from("ij", &a, "ik", &b, "kj", topology.clone(), 1.0, 1.0)
            .unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();
    context.barrier();
    elapsed / ITERATIONS as f64
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let world_seconds = run(&world);
    if world.rank() == 0 {
        println!(
            "INFO bench_contraction: C[ij] += A[ik]*B[kj], n={N}, iterations={ITERATIONS}, world sec/iter={world_seconds}"
        );
    }
    world.close();
    runtime.finalize();
}
