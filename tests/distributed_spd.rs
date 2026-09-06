// Pinned test/python/test_la.py::test_solve acceptance, expressed as A X = B.
use ctf::{algebra::Arithmetic, context::{Context, Runtime}, linalg::Native,
    mapping::Distribution, tensor::Tensor};

fn exercise(context: &Context<'_>) {
    let np = context.size();
    let grid = if np == 4 { [2, 2] } else { [np, 1] };
    for (n, nrhs) in [(1, 1), (5, 3), (11, 1), (11, 4), (11, 12), (11, 15), (11, 31)] {
        let make = |rows, cols| Tensor::new(context,
            Distribution::cyclic(vec![rows, cols], np), Arithmetic::<f64>::new());
        let mut a = make(n, n);
        a.transform(|key, value| {
            let (i, j) = (key % n, key / n);
            *value = if i == j { (n + i + 1) as f64 } else { 0.25 };
        });
        let mut rhs = make(n, nrhs);
        rhs.transform(|key, value| *value = (key % 13 + 1) as f64 / 7.);
        let solution = rhs.solve_spd(&a).unwrap();
        assert_eq!(solution.distribution(), rhs.distribution());
        let mut reconstructed = make(n, nrhs);
        reconstructed.gemm_2d::<Native>(&a, &solution, grid, 1., 0.);
        let mut norms = [0., 0.];
        for ((key, expected), (other, actual)) in rhs.local_pairs().into_iter()
            .zip(reconstructed.local_pairs()) {
            assert_eq!(key, other);
            assert!(actual.is_finite());
            norms[0] += (actual - expected).abs();
            norms[1] += actual.abs();
        }
        context.sum_f64(&mut norms);
        assert!(norms[0] <= 1e-3 || norms[0] / norms[1] <= 1e-3,
            "SPD residual {} / {}", norms[0], norms[1]);
    }
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    exercise(&world);
    let child = world.split(Some((world.rank() % 2) as i32), world.rank() as i32).unwrap();
    exercise(&child);
    child.close();
    if world.rank() == 0 {
        println!("DIGIT / PASS distributed_spd: identity padding, virtual columns, subcommunicators, ranks={}", world.size());
    }
    world.close();
    runtime.finalize();
}
