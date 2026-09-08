// Reconstruction and orthogonality from pinned scalapack_tests/eigh.cxx.
use ctf::{
    algebra::Arithmetic, context::Context, linalg::Native, mapping::Distribution, tensor::Tensor,
};
fn exercise(context: &Context<'_>) {
    let np = context.size();
    let grid = if np == 4 { [2, 2] } else { [np, 1] };
    for (n, degenerate) in [(5, false), (5, true), (1, false)] {
        let mut a = Tensor::new(
            context,
            Distribution::cyclic(vec![n, n], np),
            Arithmetic::<f64>::new(),
        );
        a.transform(|key, value| {
            let i = key % n;
            let j = key / n;
            *value = if degenerate {
                if i == j { (i / 2 + 1) as f64 } else { 0. }
            } else {
                (0..=i.min(j))
                    .map(|k| {
                        let x = if i == k { (i + 2) as f64 } else { 0.5 };
                        let y = if j == k { (j + 2) as f64 } else { 0.5 };
                        x * y
                    })
                    .sum()
            };
        });
        let (mut vectors, values) = a.eigh().unwrap();
        assert_eq!(vectors.distribution(), a.distribution());
        let eigenvalues = values.read(&(0..n).collect::<Vec<_>>());
        assert!(eigenvalues.windows(2).all(|w| w[0] <= w[1]));
        let transpose = vectors.permute_axes(&[1, 0]);
        let mut gram = a.clone();
        gram.gemm_2d::<Native>(&transpose, &vectors, grid, 1., 0.);
        let mut norms = [0., 0.];
        for (key, value) in gram.local_pairs() {
            norms[0] += (value - if key % n == key / n { 1. } else { 0. }).powi(2);
        }
        vectors.transform(|key, value| *value *= eigenvalues[key / n]);
        let mut reconstructed = a.clone();
        reconstructed.gemm_2d::<Native>(&vectors, &transpose, grid, 1., 0.);
        for ((key, x), (other, y)) in a.local_pairs().into_iter().zip(reconstructed.local_pairs()) {
            assert_eq!(key, other);
            norms[1] += (x - y).powi(2);
        }
        context.sum_f64(&mut norms);
        let bound = (n * n) as f64 * 1e-6;
        assert!(
            norms.iter().all(|x| x.sqrt() <= bound),
            "norms={norms:?}, bound={bound}"
        );
        if context.rank() == 0 {
            println!(
                "eigh n={n}, degenerate={degenerate}: orthogonality={}, reconstruction={}",
                norms[0].sqrt(),
                norms[1].sqrt()
            );
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
            "DIGIT / PASS distributed_eigh: square-subworld eigensolver and parent redistribution; ranks={}",
            world.size()
        );
    }
    world.close();
    drop(universe);
}
