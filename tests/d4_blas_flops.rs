//! Exact D4 probes for the four CPU BLAS SYR scalar families and flop totals.
#![cfg(feature = "native-linalg")]

use ctf::{
    algebra::{Arithmetic, Complex, Monoid, Semiring},
    flop_counter::FlopCounter,
    linalg::{Gemm, GemmKernel, Native, Syr, SyrKernel, Transpose, Uplo},
};

fn expected_syr<T>(
    uplo: Uplo,
    n: usize,
    alpha: &T,
    x: &[T],
    input: &[T],
    algebra: &Arithmetic<T>,
) -> Vec<T>
where
    T: Clone,
    Arithmetic<T>: Semiring<Element = T>,
{
    let mut expected = input.to_vec();
    for column in 0..n {
        for row in 0..n {
            let selected = match uplo {
                Uplo::Lower => row >= column,
                Uplo::Upper => row <= column,
            };
            if selected {
                let product = algebra.multiply(alpha, &algebra.multiply(&x[row], &x[column]));
                expected[row + n * column] = algebra.add(&expected[row + n * column], &product);
            }
        }
    }
    expected
}

fn syr_case<T>(uplo: Uplo, alpha: T, x: Vec<T>, input: Vec<T>)
where
    T: Clone + PartialEq + std::fmt::Debug,
    Arithmetic<T>: Semiring<Element = T>,
    Native: SyrKernel<T>,
{
    let algebra = Arithmetic::<T>::new();
    let expected = expected_syr(uplo, 3, &alpha, &x, &input, &algebra);
    let mut actual = input;
    Native::syr(Syr {
        uplo,
        n: 3,
        alpha,
        x: &x,
        incx: 1,
        a: &mut actual,
        lda: 3,
    });
    assert_eq!(actual, expected);
}

fn exact_syr_cases() {
    syr_case(
        Uplo::Lower,
        2.0f32,
        vec![1.0, -2.0, 3.0],
        (1..=9).map(|v| v as f32).collect(),
    );
    syr_case(
        Uplo::Upper,
        -3.0f64,
        vec![2.0, 1.0, -1.0],
        (1..=9).map(|v| v as f64).collect(),
    );
    syr_case(
        Uplo::Lower,
        Complex::new(1.0f32, 1.0),
        vec![
            Complex::new(1.0, 2.0),
            Complex::new(-2.0, 1.0),
            Complex::new(0.0, -1.0),
        ],
        (1..=9)
            .map(|v| Complex::new(v as f32, -(v as f32)))
            .collect(),
    );
    syr_case(
        Uplo::Upper,
        Complex::new(-1.0f64, 2.0),
        vec![
            Complex::new(2.0, -1.0),
            Complex::new(1.0, 1.0),
            Complex::new(-1.0, 0.0),
        ],
        (1..=9)
            .map(|v| Complex::new(v as f64, (v as f64) / 2.0))
            .collect(),
    );
}

fn exact_flop_aggregation(context: &ctf::context::Context<'_>) {
    let counter = FlopCounter::new();
    let mut c = [0.0f64];
    Native::gemm(Gemm {
        trans_a: Transpose::No,
        trans_b: Transpose::No,
        m: 1,
        n: 1,
        k: 1,
        alpha: 1.0,
        a: &[2.0],
        lda: 1,
        b: &[3.0],
        ldb: 1,
        beta: 0.0,
        c: &mut c,
        ldc: 1,
    });
    assert_eq!(c, [6.0]);
    assert_eq!(counter.local(), 2);
    let expected = (2 * context.size()) as i64;
    assert_eq!(counter.count(context), expected);
    let mut counter = counter;
    counter.zero();
    assert_eq!(counter.local(), 0);
}

fn main() {
    exact_syr_cases();
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    exact_flop_aggregation(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    exact_flop_aggregation(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS d4_blas_flops: four-type plain-transpose SYR and production GEMM flop snapshots exact; world+parity"
        );
    }
    world.close();
    drop(universe);
}
