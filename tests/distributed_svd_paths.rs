// Distributed truncated/randomized SVD paths; bounds follow scalapack_tests/svd.cxx.
use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    linalg::Native,
    mapping::Distribution,
    tensor::Tensor,
};

type Matrix<'c, 'r> = Tensor<'c, 'r, Arithmetic<f64>>;

fn make_matrix<'c,'r>(context: &'c Context<'r>, m: usize, n: usize) -> Matrix<'c,'r> {
    Tensor::new(
        context,
        Distribution::cyclic(vec![m, n], context.size()),
        Arithmetic::new(),
    )
}

fn diagonal<'c,'r>(context: &'c Context<'r>, values: &[f64]) -> Matrix<'c,'r> {
    let n = values.len();
    let mut matrix = make_matrix(context, n, n);
    matrix.transform(|key, value| {
        let row = key % n;
        let column = key / n;
        *value = if row == column { values[row] } else { 0. };
    });
    matrix
}

fn rank_two_fixture<'c,'r>(context: &'c Context<'r>) -> Matrix<'c,'r> {
    let (m, n) = (5, 4);
    let mut matrix = make_matrix(context, m, n);
    matrix.transform(|key, value| {
        let row = key % m;
        let column = key / m;
        *value = match row {
            0 => [3., 1., 2., -1.][column],
            1 => [0., 4., -1., 2.][column],
            _ => 0.,
        };
    });
    matrix
}

fn rank_two_guess<'c,'r>(context: &'c Context<'r>) -> Matrix<'c,'r> {
    let (m, rank) = (5, 2);
    let mut guess = make_matrix(context, m, rank);
    guess.transform(|key, value| {
        let row = key % m;
        let column = key / m;
        *value = if row == column { 1. } else { 0. };
    });
    guess
}

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
    let norm = error[0].sqrt();
    assert!(norm <= bound, "orthogonality={norm}, bound={bound}");
}

fn scale_left(
    u: &mut Matrix<'_, '_>,
    singular: &Matrix<'_, '_>,
    rows: usize,
    rank: usize,
) {
    let values = singular.read(&(0..rank).collect::<Vec<_>>());
    u.transform(|key, value| *value *= values[key / rows]);
}

fn assert_shapes(u: &Matrix<'_, '_>, singular: &Matrix<'_, '_>, vt: &Matrix<'_, '_>, m: usize, rank: usize, n: usize) {
    assert_eq!(u.distribution().shape, vec![m, rank]);
    assert_eq!(singular.distribution().shape, vec![rank]);
    assert_eq!(vt.distribution().shape, vec![rank, n]);
}

fn exercise_truncated(context: &Context<'_>, grid: [usize; 2]) {
    let source = diagonal(context, &[9., 4., 1.]);

    let (mut u, singular, vt) = source.svd_truncated(grid, Some(2), 4.).unwrap();
    assert_shapes(&u, &singular, &vt, 3, 2, 3);
    scale_left(&mut u, &singular, 3, 2);
    let mut reconstructed = make_matrix(context, 3, 3);
    reconstructed.gemm_2d::<Native>(&u, &vt, grid, 1., 0.);
    let expected = diagonal(context, &[9., 4., 0.]);
    residual(&expected, &mut reconstructed, 27e-6);

    // In the pinned source, a positive requested rank plus a threshold above
    // the largest singular value computes retained=0 and therefore returns
    // the full factors through the retained>0 truncation guard.
    let (mut u, singular, vt) = source.svd_truncated(grid, Some(2), 10.).unwrap();
    assert_shapes(&u, &singular, &vt, 3, 3, 3);
    scale_left(&mut u, &singular, 3, 3);
    let mut reconstructed = make_matrix(context, 3, 3);
    reconstructed.gemm_2d::<Native>(&u, &vt, grid, 1., 0.);
    residual(&source, &mut reconstructed, 27e-6);
}

fn exercise_randomized(context: &Context<'_>, grid: [usize; 2]) {
    let source = rank_two_fixture(context);
    let guess = rank_two_guess(context);
    let (m, n, rank) = (5, 4, 2);
    let bound_orthogonality = (m * n) as f64 * 1e-6;
    let bound_residual = (m * n * n) as f64 * 1e-6;

    let (mut u, singular, vt) = source
        .svd_randomized(grid, rank, 1, 0, 0x5eed, Some(&guess))
        .unwrap();
    assert_shapes(&u, &singular, &vt, m, rank, n);
    orthogonal(&u, true, grid, bound_orthogonality);
    orthogonal(&vt, false, grid, bound_orthogonality);
    scale_left(&mut u, &singular, m, rank);
    let mut reconstructed = make_matrix(context, m, n);
    reconstructed.gemm_2d::<Native>(&u, &vt, grid, 1., 0.);
    residual(&source, &mut reconstructed, bound_residual);

    // With no supplied guess, the fixed seed makes the random starting space
    // reproducible.  The source truncates Q to rank before projected SVD.
    let (mut u, singular, vt) = source
        .svd_randomized(grid, rank, 1, 1, 0x5eed, None)
        .unwrap();
    assert_shapes(&u, &singular, &vt, m, rank, n);
    orthogonal(&u, true, grid, bound_orthogonality);
    orthogonal(&vt, false, grid, bound_orthogonality);
    scale_left(&mut u, &singular, m, rank);
    let mut reconstructed = make_matrix(context, m, n);
    reconstructed.gemm_2d::<Native>(&u, &vt, grid, 1., 0.);
    residual(&source, &mut reconstructed, bound_residual);
}

fn exercise(context: &Context<'_>) {
    let np = context.size();
    let grid = if np == 4 { [2, 2] } else { [np, 1] };
    exercise_truncated(context, grid);
    exercise_randomized(context, grid);
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
            "DIGIT / PASS distributed_svd_paths: truncated and randomized distributed SVD paths; ranks={}",
            world.size()
        );
    }
    world.close();
    runtime.finalize();
}
