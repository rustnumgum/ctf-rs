//! Pinned `examples/checkpoint.cxx`: dense MPI-IO checkpoint round trip.

use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Topology},
    random::Generator,
    tensor::Tensor,
};
use std::path::PathBuf;

type Dense<'c, 'r> = Tensor<'c, 'r, Arithmetic<f64>>;

fn checkpoint(context: &Context<'_>, path: &PathBuf) -> (f64, f64) {
    let n = 7usize;
    let distribution = Distribution::cyclic(vec![n, n], context.size());
    let mut a = Dense::new(context, distribution.clone(), Arithmetic::new());
    let mut a2 = Dense::new(context, distribution.clone(), Arithmetic::new());
    let mut a3 = Dense::new(context, distribution.clone(), Arithmetic::new());
    let mut a4 = Dense::new(context, distribution.clone(), Arithmetic::new());
    let mut a5 = Dense::new(context, distribution, Arithmetic::new());

    let mut generator = Generator::new(13 * context.rank() as u64);
    a.fill_random(0.0, 1.0, &mut generator);
    a.transform_indexed("ii", |value| *value = 0.0);
    a2.sum_from(
        "ij",
        &a,
        "ij",
        Topology::new(vec![context.size()]),
        1.0,
        0.0,
    )
    .unwrap();
    a3.sum_from(
        "ij",
        &a,
        "ij",
        Topology::new(vec![context.size()]),
        2.0,
        0.0,
    )
    .unwrap();

    a2.write_dense_to_file(path, 0);
    a3.write_dense_to_file(path, (n * n * std::mem::size_of::<f64>()) as u64);
    a4.read_dense_from_file(path, 0);
    a.sum_from(
        "ij",
        &a4,
        "ij",
        Topology::new(vec![context.size()]),
        -1.0,
        1.0,
    )
    .unwrap();
    let first = a.norm2();

    a5.read_dense_from_file(path, (n * n * std::mem::size_of::<f64>()) as u64);
    a5.sum_from(
        "ij",
        &a4,
        "ij",
        Topology::new(vec![context.size()]),
        -2.0,
        1.0,
    )
    .unwrap();
    let second = a5.norm2();
    (first, second)
}

fn run(context: &Context<'_>, token: u64, scope: usize) {
    let path = std::env::temp_dir().join(format!("ctf-checkpoint-{token}-{scope}.bin"));
    let (first, second) = checkpoint(context, &path);
    let bound = 1.0e-9 * 7.0;
    assert!(first <= bound, "checkpoint first norm {first:e}");
    assert!(second <= bound, "checkpoint second norm {second:e}");
    context.barrier();
    if context.rank() == 0 {
        std::fs::remove_file(&path).unwrap();
    }
    context.barrier();
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let mut token = [std::process::id() as u64];
    world.broadcast(0, &mut token);
    run(&world, token[0], 0);
    let color = world.rank() % 2;
    let parity = world
        .split(Some(color as i32), world.rank() as i32)
        .unwrap();
    run(&parity, token[0], color + 1);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS checkpoint: dense two-offset MPI-IO round trip; n=7; both norms<=1e-9*n; world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
