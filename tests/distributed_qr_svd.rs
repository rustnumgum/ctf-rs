// Residual/orthogonality criteria adapted from pinned scalapack_tests/{qr,svd}.cxx.
use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    linalg::Native,
    mapping::Distribution,
    tensor::Tensor,
};
type Matrix<'c, 'r> = Tensor<'c, 'r, Arithmetic<f64>>;
fn residual(reference: &Matrix<'_, '_>, actual: &mut Matrix<'_, '_>, bound: f64) -> f64 {
    actual.redistribute(reference.distribution().clone());
    let mut error = [0.];
    for ((key, a), (other, b)) in reference
        .local_pairs()
        .into_iter()
        .zip(actual.local_pairs())
    {
        assert_eq!(key, other);
        error[0] += (a - b).powi(2);
    }
    reference.context().sum_f64(&mut error);
    let norm = error[0].sqrt();
    assert!(norm <= bound, "norm={norm}, bound={bound}");
    norm
}
fn orthogonal(matrix: &Matrix<'_, '_>, columns: bool, grid: [usize; 2], bound: f64) {
    let k = matrix.distribution().shape[usize::from(columns)];
    let transpose = matrix.permute_axes(&[1, 0]);
    let mut gram = Matrix::new(
        matrix.context(),
        Distribution::cyclic(vec![k, k], matrix.context().size()),
        Arithmetic::new(),
    );
    if columns {
        gram.gemm_2d::<Native>(&transpose, matrix, grid, 1., 0.);
    } else {
        gram.gemm_2d::<Native>(matrix, &transpose, grid, 1., 0.);
    }
    let mut error = [0.];
    for (key, value) in gram.local_pairs() {
        error[0] += (value - if key % k == key / k { 1. } else { 0. }).powi(2);
    }
    matrix.context().sum_f64(&mut error);
    assert!(
        error[0].sqrt() <= bound,
        "orthogonality={}",
        error[0].sqrt()
    );
}
fn exercise(context: &Context<'_>) {
    let np = context.size();
    let grid = if np == 4 { [2, 2] } else { [np, 1] };
    for (m, n) in [(13, 7), (5, 8), (1, 1)] {
        let mut a = Matrix::new(
            context,
            Distribution::cyclic(vec![m, n], np),
            Arithmetic::new(),
        );
        a.transform(|key, v| {
            *v = ((key * 17 + 11) % 31) as f64 / 31. + if key % m == key / m { 1. } else { 0. }
        });
        let (q, r) = a.qr(grid).unwrap();
        assert_eq!(q.distribution().shape, vec![m, m.min(n)]);
        assert_eq!(r.distribution().shape, vec![m.min(n), n]);
        orthogonal(&q, true, grid, (m * n) as f64 * 1e-6);
        let mut reconstructed = a.clone();
        reconstructed.gemm_2d::<Native>(&q, &r, grid, 1., 0.);
        let qr_error = residual(&a, &mut reconstructed, (m * n * n) as f64 * 1e-6);
        let (mut u, s, vt) = a.svd(grid).unwrap();
        orthogonal(&u, true, grid, (m * n) as f64 * 1e-6);
        orthogonal(&vt, false, grid, (m * n) as f64 * 1e-6);
        // Vector values alone may be read collectively; no matrix factor gather.
        let values = s.read(&(0..m.min(n)).collect::<Vec<_>>());
        assert!(values.windows(2).all(|v| v[0] >= v[1]));
        u.transform(|key, v| *v *= values[key / m]);
        reconstructed.gemm_2d::<Native>(&u, &vt, grid, 1., 0.);
        let svd_error = residual(&a, &mut reconstructed, (m * n * n) as f64 * 1e-6);
        if context.rank() == 0 {
            println!(
                "distributed QR/SVD {m}x{n}: QR residual={qr_error}, SVD residual={svd_error}"
            );
        }
    }
}
fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    exercise(&world);
    let child = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    exercise(&child);
    child.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS distributed_qr_svd: original Frobenius bounds; ranks={}",
            world.size()
        );
    }
    world.close();
    runtime.finalize();
}
