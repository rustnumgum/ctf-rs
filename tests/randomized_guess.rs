use ctf::{
    algebra::Arithmetic, context::Context, linalg::Native, mapping::Distribution, tensor::Tensor,
};
fn run(context: &Context<'_>) {
    let np = context.size();
    let grid = if np == 4 { [2, 2] } else { [np, 1] };
    let make = |m, n| {
        Tensor::new(
            context,
            Distribution::cyclic(vec![m, n], np),
            Arithmetic::<f64>::new(),
        )
    };
    let mut a = make(5, 4);
    a.transform(|key, x| {
        *x = ((key * 17 + 11) % 31) as f64 / 31. + if key % 5 == key / 5 { 1. } else { 0. }
    });
    for iterations in [0, 1] {
        let mut guess = make(5, 3);
        guess.transform(|key, x| *x = ((key * 7 + 3) % 17 + 1) as f64 / 4.);
        let before = guess.local_pairs();
        let original = guess.distribution().clone();
        let (u, s, vt) = a
            .svd_randomized(grid, 2, iterations, 1, 19, Some(&mut guess))
            .unwrap();
        assert_eq!(u.distribution().shape, vec![5, 2]);
        assert_eq!(s.distribution().shape, vec![2]);
        assert_eq!(vt.distribution().shape, vec![2, 4]);
        assert_eq!(guess.distribution().shape, vec![5, 3]);
        if iterations == 0 {
            assert_eq!(guess.distribution(), &original);
            assert_eq!(guess.local_pairs(), before);
        } else {
            let transpose = guess.permute_axes(&[1, 0]);
            let mut gram = make(3, 3);
            gram.gemm_2d::<Native>(&transpose, &guess, grid, 1., 0.);
            let mut error = [0.];
            for (key, x) in gram.local_pairs() {
                assert!(x.is_finite());
                error[0] += (x - if key % 3 == key / 3 { 1. } else { 0. }).powi(2);
            }
            context.sum_f64(&mut error);
            assert!(error[0].sqrt() <= 5. * 3. * 1e-6);
        }
    }
}
fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    run(&world);
    let child = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&child);
    child.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS randomized_guess: zero-iteration in/out data exact; final oversampled guess retained and QR-orthogonal; world+parity"
        );
    }
    world.close();
    drop(universe);
}
