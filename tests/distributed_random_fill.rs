use ctf::{
    algebra::{Arithmetic, Complex},
    context::Context,
    linalg::Native,
    mapping::Distribution,
    random::Generator,
    tensor::Tensor,
};
macro_rules! scalar_case {
    ($name:ident,$t:ty,$min:expr,$max:expr,$expected:expr,$zero:expr) => {
        fn $name(c: &Context<'_>) {
            let mut tensor = Tensor::new(
                c,
                Distribution::cyclic(vec![5, 3], c.size()),
                Arithmetic::<$t>::new(),
            );
            let mut actual = Generator::new(c.rank() as u64);
            let mut reference = Generator::new(c.rank() as u64);
            tensor.fill_random($min, $max, &mut actual);
            for (offset, &value) in tensor.local_storage().iter().enumerate() {
                let draw = reference.unit_interval();
                let expected = if tensor.distribution().global_key(c.rank(), offset).is_some() {
                    ($expected)(draw)
                } else {
                    $zero
                };
                assert_eq!(
                    value,
                    expected,
                    "typed random value/padding at rank {} offset {offset}",
                    c.rank()
                );
            }
            assert_eq!(
                actual.next_u64(),
                reference.next_u64(),
                "padding consumes RNG draws"
            );
        }
    };
}
scalar_case!(
    real32,
    f32,
    -1f32,
    1f32,
    |x: f64| (x as f32) * 2f32 - 1f32,
    0f32
);
scalar_case!(real64, f64, -1f64, 1f64, |x: f64| x * 2. - 1., 0f64);
scalar_case!(
    complex32,
    Complex<f32>,
    Complex::new(-1f32, 0f32),
    Complex::new(1f32, 0f32),
    |x: f64| Complex::new((x as f32) * 2f32 - 1f32, 0f32),
    Complex::new(0f32, 0f32)
);
scalar_case!(
    complex64,
    Complex<f64>,
    Complex::new(-1f64, 0f64),
    Complex::new(1f64, 0f64),
    |x: f64| Complex::new(x * 2. - 1., 0.),
    Complex::new(0f64, 0f64)
);
fn run(c: &Context<'_>) {
    real32(c);
    real64(c);
    complex32(c);
    complex64(c);
    let grid = if c.size() == 4 { [2, 2] } else { [c.size(), 1] };
    let mut a = Tensor::new(
        c,
        Distribution::cyclic(vec![5, 4], c.size()),
        Arithmetic::<f64>::new(),
    );
    a.transform(|key, x| {
        let i = key % 5;
        let j = key / 5;
        *x = ((i + 1) * (j + 1) + (i % 2 + 1) * (j % 2)) as f64;
    });
    let (mut u, s, vt) = a.svd_randomized(grid, 2, 1, 1, 19, None).unwrap();
    let values = s.read(&[0, 1]);
    u.transform(|key, x| *x *= values[key / 5]);
    let mut reconstructed = a.clone();
    reconstructed.gemm_2d::<Native>(&u, &vt, grid, 1., 0.);
    let mut error = [0.];
    for ((key, expected), (other, actual)) in
        a.local_pairs().into_iter().zip(reconstructed.local_pairs())
    {
        assert_eq!(key, other);
        assert!(actual.is_finite());
        error[0] += (actual - expected).powi(2);
    }
    c.sum_f64(&mut error);
    assert!(error[0].sqrt() <= 5. * 4. * 4. * 1e-6);
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
            "PASS distributed_random_fill: four types, pre-scaling precision and padding draw consumption exact; world+parity"
        );
    }
    world.close();
    drop(universe);
}
