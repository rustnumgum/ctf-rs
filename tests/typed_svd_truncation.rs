// Source rank/threshold branches with known singular values, no vector-phase comparison.
use ctf::{
    algebra::{Arithmetic, Complex, Group, Monoid, Semiring},
    context::Context,
    mapping::{Distribution, Topology},
    tensor::Tensor,
};
macro_rules! scalar_case {
    ($name:ident,$t:ty,$value:expr,$norm2:expr,$real:expr,$rounded_keep:expr) => {
        fn $name(context: &Context<'_>) {
            let value = $value;
            let norm2 = $norm2;
            let real = $real;
            let algebra = Arithmetic::<$t>::new();
            let np = context.size();
            let grid = if np == 4 { [2, 2] } else { [np, 1] };
            let topology = Topology::new(grid.to_vec());
            let mut a = Tensor::new(
                context,
                Distribution::cyclic(vec![5, 3], np),
                algebra.clone(),
            );
            a.transform(|key, x| {
                *x = if key % 5 == key / 5 {
                    value([5., 3., 1.][key / 5])
                } else {
                    algebra.zero()
                }
            });
            for (rank, threshold, keep) in [
                (None, 0., 3),
                (Some(2), 0., 2),
                (None, 3., 2),
                (Some(1), 3., 1),
                (Some(0), 4., 1),
                (Some(0), 0., 3),
                (None, 10., 3),
                (Some(9), 0., 3),
                (None, 3. + 1e-8, $rounded_keep),
            ] {
                let (mut u, s, vt) = a.svd_truncated(grid, rank, threshold).unwrap();
                assert_eq!(u.distribution().shape, vec![5, keep]);
                assert_eq!(s.distribution().shape, vec![keep]);
                assert_eq!(vt.distribution().shape, vec![keep, 3]);
                let values = s.read(&(0..keep).collect::<Vec<_>>());
                for (i, &x) in values.iter().enumerate() {
                    assert!(real(x).is_finite() && (real(x) - [5., 3., 1.][i]).abs() < 1e-6);
                }
                u.transform(|key, x| *x = algebra.multiply(x, &values[key / 5]));
                let mut reconstructed =
                    Tensor::new(context, a.distribution().clone(), algebra.clone());
                reconstructed
                    .contract_from_on_grid(
                        "ij",
                        &u,
                        "ik",
                        &vt,
                        "kj",
                        topology.clone(),
                        algebra.one(),
                        algebra.zero(),
                    )
                    .unwrap();
                let mut error = [0.];
                for (key, x) in reconstructed.local_pairs() {
                    assert!(norm2(x).is_finite());
                    let expected = if key % 5 == key / 5 && key / 5 < keep {
                        value([5., 3., 1.][key / 5])
                    } else {
                        algebra.zero()
                    };
                    error[0] += norm2(algebra.add(&x, &algebra.negate(&expected)));
                }
                context.sum_f64(&mut error);
                assert!(error[0].sqrt() <= 5. * 3. * 3. * 1e-6);
            }
        }
    };
}
scalar_case!(
    real32,
    f32,
    |r: f64| r as f32,
    |x: f32| (x as f64).powi(2),
    |x: f32| x as f64,
    2
);
scalar_case!(real64, f64, |r: f64| r, |x: f64| x * x, |x: f64| x, 1);
scalar_case!(
    complex32,
    Complex<f32>,
    |r: f64| Complex::new(0., r as f32),
    |x: Complex<f32>| (x.re as f64).powi(2) + (x.im as f64).powi(2),
    |x: Complex<f32>| x.re as f64,
    2
);
scalar_case!(
    complex64,
    Complex<f64>,
    |r: f64| Complex::new(0., r),
    |x: Complex<f64>| x.norm_squared(),
    |x: Complex<f64>| x.re,
    1
);
fn run(c: &Context<'_>) {
    real32(c);
    real64(c);
    complex32(c);
    complex64(c);
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
            "DIGIT / PASS typed_svd_truncation: four native scalar types, exact rank/threshold branches including zero-rank full-factor quirk; distributed sliced reconstruction; world+parity"
        );
    }
    world.close();
    drop(universe);
}
