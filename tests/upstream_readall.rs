//! Port of the pinned `test/readall_test.cxx` fixture.
//!
//! The source fills a rank-zero MPI_SELF tensor with `drand48` values before
//! `add_from_subworld` and checks `read_all`. The POSIX 48-bit recurrence is
//! evaluated directly in Rust, retaining the source's seed-zero sequence.

use ctf::{
    algebra::{Arithmetic, Monoid, Semiring},
    context::{Context, Runtime},
    mapping::Distribution,
    tensor::Tensor,
};

fn run(context: &Context<'_>, n: usize, m: usize) {
    let shape = vec![n, m, n, m];
    let total = n * m * n * m;
    let algebra = Arithmetic::<f64>::new();
    let mut state = 0x330e_u64;
    let reference: Vec<_> = (0..total).map(|_| {
        state = state.wrapping_mul(0x5deece66d).wrapping_add(0xb) & ((1_u64<<48)-1);
        state as f64 / (1_u64<<48) as f64
    }).collect();

    let mut tensor = Tensor::new(context, Distribution::cyclic(shape.clone(), context.size()), algebra);

    // Preserve the source's rank-zero MPI_SELF path without implicit contexts.
    let child = context.split((context.rank() == 0).then_some(0), context.rank() as i32);
    let source_distribution = Distribution::cyclic(shape, 1);
    let source = child.as_ref().map(|child| {
        let mut source = Tensor::new(child, source_distribution.clone(), Arithmetic::<f64>::new());
        source.transform(|key, value| *value = reference[key]);
        source
    });
    tensor.add_from_subworld(
        source.as_ref(),
        &source_distribution,
        algebra.one(),
        algebra.zero(),
    );
    drop(source);
    if let Some(child) = child {
        child.close();
    }

    let actual = tensor.all_data();
    assert_eq!(actual.len(), total);
    for (key, value) in actual.iter().enumerate() {
        let expected = reference[key];
        assert!((value - expected).abs() <= 1e-10, "key {key}: {value} != {expected}");
    }

    let pairs = tensor.all_pairs(false);
    assert_eq!(pairs.len(), total);
    assert!(pairs
        .iter()
        .enumerate()
        .all(|(key, &(actual_key, _))| key == actual_key));
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world, 2, 3);

    let rank = world.rank();
    let parity = world
        .split(Some((rank % 2) as i32), rank as i32)
        .unwrap();
    run(&parity, 2, 3);
    parity.close();

    world.barrier();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS upstream_readall: root subworld fill, distributed read_all equivalent, world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
