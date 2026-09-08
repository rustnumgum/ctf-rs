// Acceptance metrics from pinned test/python/test_la.py cholesky/solve_tri.
use ctf::{
    algebra::Arithmetic, context::Context, linalg::Native, mapping::Distribution, tensor::Tensor,
};
fn close(reference: &Tensor<'_, '_, Arithmetic<f64>>, actual: &Tensor<'_, '_, Arithmetic<f64>>) {
    let mut residual = [0., 0.];
    for ((key, a), (other, b)) in reference
        .local_pairs()
        .into_iter()
        .zip(actual.local_pairs())
    {
        assert_eq!(key, other);
        residual[0] += (a - b).abs();
        residual[1] += a.abs();
    }
    reference.context().sum_f64(&mut residual);
    assert!(
        residual[0] <= 1e-3 || residual[0] / residual[1] <= 1e-3,
        "L1 residual={}, reference norm={}",
        residual[0],
        residual[1]
    );
}
fn exercise(context: &Context<'_>) {
    let np = context.size();
    let grid = if np == 4 { [2, 2] } else { [np, 1] };
    let make = |m, n| {
        Tensor::new(
            context,
            Distribution::cyclic(vec![m, n], np),
            Arithmetic::<f64>::new(),
        )
    };
    // Include nonuniform cyclic blocks and ranks with no mathematical local rows.
    for n in [1, 4, 5] {
        let mut a = make(n, n);
        a.transform(|key, value| {
            let i = key % n;
            let j = key / n;
            *value = (0..=i.min(j))
                .map(|k| {
                    let left = if i == k { (i + 2) as f64 } else { 1. };
                    let right = if j == k { (j + 2) as f64 } else { 1. };
                    left * right
                })
                .sum();
        });
        for lower in [true, false] {
            let factor = a.cholesky(grid, lower).unwrap();
            assert_eq!(factor.distribution(), a.distribution());
            let transposed = factor.permute_axes(&[1, 0]);
            let mut reconstructed = make(n, n);
            if lower {
                reconstructed.gemm_2d::<Native>(&factor, &transposed, grid, 1., 0.);
            } else {
                reconstructed.gemm_2d::<Native>(&transposed, &factor, grid, 1., 0.);
            }
            close(&a, &reconstructed);
            let mut forbidden = [0.];
            for (key, value) in factor.local_pairs() {
                if (lower && key % n < key / n) || (!lower && key % n > key / n) {
                    forbidden[0] += value * value;
                }
            }
            context.sum_f64(&mut forbidden);
            assert!(forbidden[0].sqrt() <= 1e-6);
        }
    }
    for lower in [true, false] {
        for from_left in [true, false] {
            for transpose in [false, true] {
                let (m, n) = (4, 7);
                let order = if from_left { m } else { n };
                let mut factor = make(order, order);
                factor.transform(|key, value| {
                    let i = key % order;
                    let j = key / order;
                    *value = if i == j {
                        (i + 2) as f64
                    } else if (lower && i > j) || (!lower && i < j) {
                        0.25
                    } else {
                        0.
                    };
                });
                let mut rhs = make(m, n);
                rhs.transform(|key, value| *value = (key + 1) as f64 / 8.);
                let solution = rhs
                    .solve_tri(&factor, grid, lower, from_left, transpose)
                    .unwrap();
                assert_eq!(solution.distribution(), rhs.distribution());
                let op = if transpose {
                    factor.permute_axes(&[1, 0])
                } else {
                    factor.clone()
                };
                let mut reconstructed = make(m, n);
                if from_left {
                    reconstructed.gemm_2d::<Native>(&op, &solution, grid, 1., 0.);
                } else {
                    reconstructed.gemm_2d::<Native>(&solution, &op, grid, 1., 0.);
                }
                close(&rhs, &reconstructed);
            }
        }
    }
}
fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    exercise(&world);
    let child = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    exercise(&child);
    child.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS distributed_matrix: ScaLAPACK Cholesky and triangular solves, upstream L1/triangle bounds, subcommunicators, ranks={}",
            world.size()
        );
    }
    world.close();
    drop(universe);
}
