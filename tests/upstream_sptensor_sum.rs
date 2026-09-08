//! Bounded deterministic port of pinned CTF test/sptensor_sum.cxx.

use ctf::{algebra::Arithmetic, context::Context, mapping::Distribution, sparse::SparseTensor};

const N: usize = 2;
const TOLERANCE: f64 = 1.0e-9;

fn expected(key: usize) -> Option<f64> {
    match key {
        1 => Some(3.2),
        2 => Some(66.0),
        3 => Some(7.2),
        4 => Some(1.4),
        8 => Some(-0.8),
        _ => None,
    }
}

fn run(context: &Context<'_>) {
    let shape = vec![N, N, N, N];
    let distribution = Distribution::cyclic(shape, context.size());
    let algebra = Arithmetic::<f64>::new();
    let mut a = SparseTensor::new(context, distribution.clone(), algebra);
    let mut b = SparseTensor::new(context, distribution, algebra);

    if context.rank() == context.size() / 2 {
        a.write_add(&[(1, 3.2), (2, 42.0), (4, 1.4), (8, -0.8)]);
        b.write_add(&[(2, 24.0), (3, 7.2)]);
    } else {
        a.write_add(&[]);
        b.write_add(&[]);
    }

    // Upstream equation: B["abij"] += A["abij"].
    b.sum_from("abij", &a, "abij", 1.0, 1.0);

    for (key, value) in b.local_pairs() {
        let expected = expected(key).unwrap_or_else(|| panic!("unexpected stored key {key}"));
        assert!(
            (value - expected).abs() <= TOLERANCE,
            "sptensor_sum mismatch at key {key}: expected {expected}, got {value}"
        );
    }
    let keys = [1, 2, 3, 4, 8];
    for ((&key, value), expected) in keys
        .iter()
        .zip(b.read(&keys))
        .zip([3.2, 66.0, 7.2, 1.4, -0.8])
    {
        assert!(
            (value - expected).abs() <= TOLERANCE,
            "sptensor_sum missing/wrong key {key}: expected {expected}, got {value}"
        );
    }
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let rank = world.rank();
    run(&world);

    let parity = world.split(Some((rank % 2) as i32), rank as i32).unwrap();
    run(&parity);
    parity.close();

    if rank == 0 {
        println!(
            "DIGIT / PASS upstream_sptensor_sum: B[abij]+=A[abij], sparse union and overlap; abs<=1e-9; world+parity"
        );
    }
    world.close();
    drop(universe);
}
