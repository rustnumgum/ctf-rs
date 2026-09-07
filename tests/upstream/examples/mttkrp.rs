//! Bounded native port of `examples/mttkrp.cxx`.
//!
//! The source compares a direct MTTKRP with a per-column TTTP construction.
//! This keeps that independent equation, the sparse order-four fixture,
//! `norm2 / T.get_tot_size(false) < 1e-5` criterion, and all four source scalar
//! families.  Matrix factors use the native MTTKRP convention `[R, mode]`;
//! their values still represent the source's logical `U[mode, R]` entries.

use ctf::{
    algebra::{Arithmetic, Complex, Monoid, Ring, Semiring, Wire},
    context::{Context, Runtime},
    mapping::{Distribution, Topology},
    sparse::SparseTensor,
    tensor::Tensor,
};

const N: usize = 7;
const RANK: usize = 7;
const SPARSE_FRACTION_TENTHS: usize = 8;

trait Scalar: Clone + PartialEq + Wire {
    fn from_ratio(real: i64, imaginary: i64, denominator: i64) -> Self;
    fn distance(left: &Self, right: &Self) -> f64;
}

impl Scalar for f32 {
    fn from_ratio(real: i64, _imaginary: i64, denominator: i64) -> Self {
        real as f32 / denominator as f32
    }

    fn distance(left: &Self, right: &Self) -> f64 {
        (*left - *right).abs() as f64
    }
}

impl Scalar for f64 {
    fn from_ratio(real: i64, _imaginary: i64, denominator: i64) -> Self {
        real as f64 / denominator as f64
    }

    fn distance(left: &Self, right: &Self) -> f64 {
        (*left - *right).abs()
    }
}

impl Scalar for Complex<f32> {
    fn from_ratio(real: i64, imaginary: i64, denominator: i64) -> Self {
        Self::new(
            real as f32 / denominator as f32,
            imaginary as f32 / denominator as f32,
        )
    }

    fn distance(left: &Self, right: &Self) -> f64 {
        (left.re - right.re).hypot(left.im - right.im) as f64
    }
}

impl Scalar for Complex<f64> {
    fn from_ratio(real: i64, imaginary: i64, denominator: i64) -> Self {
        Self::new(
            real as f64 / denominator as f64,
            imaginary as f64 / denominator as f64,
        )
    }

    fn distance(left: &Self, right: &Self) -> f64 {
        (left.re - right.re).hypot(left.im - right.im)
    }
}

fn input_value<T: Scalar>(key: usize) -> T {
    let real = (key.wrapping_mul(17).wrapping_add(3) % 15) as i64 - 7;
    let imaginary = (key.wrapping_mul(11).wrapping_add(1) % 7) as i64 - 3;
    T::from_ratio(real, imaginary, 7)
}

fn factor_value<T: Scalar>(mode: usize, row: usize, auxiliary: usize) -> T {
    let real = (1 + (mode * 13 + row * 5 + auxiliary * 3) % 7) as i64;
    T::from_ratio(real, 0, 7)
}

fn make_input<'c, 'r, T>(
    context: &'c Context<'r>,
    shape: &[usize],
) -> SparseTensor<'c, 'r, Arithmetic<T>>
where
    T: Scalar,
    Arithmetic<T>: Ring<Element = T> + Clone,
{
    let distribution = Distribution::cyclic(shape.to_vec(), context.size());
    let mut input = SparseTensor::new(context, distribution.clone(), Arithmetic::new());
    let pairs: Vec<_> = (0..distribution.global_len())
        .filter(|key| {
            (key.wrapping_mul(19).wrapping_add(7) % 10) < SPARSE_FRACTION_TENTHS
                && distribution.owner(*key) == context.rank()
        })
        .map(|key| (key, input_value(key)))
        .collect();
    input.write_add(&pairs);
    input
}

fn make_factor<'c, 'r, T>(
    context: &'c Context<'r>,
    mode: usize,
    rows: usize,
) -> Tensor<'c, 'r, Arithmetic<T>>
where
    T: Scalar,
    Arithmetic<T>: Ring<Element = T> + Clone,
{
    let mut factor = Tensor::new(
        context,
        Distribution::cyclic(vec![RANK, rows], context.size()),
        Arithmetic::<T>::new(),
    );
    factor.transform(|key, value| {
        let auxiliary = key % RANK;
        let row = key / RANK;
        *value = factor_value(mode, row, auxiliary);
    });
    factor
}

fn factor_column<'c, 'r, T>(
    context: &'c Context<'r>,
    factor: &Tensor<'c, 'r, Arithmetic<T>>,
    component: usize,
    rows: usize,
) -> Tensor<'c, 'r, Arithmetic<T>>
where
    T: Scalar,
    Arithmetic<T>: Ring<Element = T> + Clone,
{
    let mut column = Tensor::new(
        context,
        Distribution::cyclic(vec![rows], context.size()),
        Arithmetic::<T>::new(),
    );
    let pairs: Vec<_> = factor
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| factor.distribution().owner(*key) == context.rank())
        .filter_map(|(key, value)| (key % RANK == component).then_some((key / RANK, value)))
        .collect();
    column.write_add(&pairs);
    column
}

fn residual<'c, 'r, T>(
    actual: &Tensor<'c, 'r, Arithmetic<T>>,
    expected: &Tensor<'c, 'r, Arithmetic<T>>,
) -> f64
where
    T: Scalar,
    Arithmetic<T>: Ring<Element = T> + Clone,
{
    assert_eq!(actual.distribution(), expected.distribution());
    let expected_pairs = expected.local_pairs();
    let keys: Vec<_> = expected_pairs.iter().map(|(key, _)| *key).collect();
    let actual_values = actual.read(&keys);
    let mut squared = 0.0;
    for ((_, expected_value), actual_value) in expected_pairs.iter().zip(actual_values.iter()) {
        let difference = T::distance(actual_value, expected_value);
        squared += difference * difference;
    }
    actual.context().sum_f64(std::slice::from_mut(&mut squared));
    squared.sqrt()
}

fn case<'c, 'r, T>(context: &'c Context<'r>)
where
    T: Scalar,
    Arithmetic<T>: Ring<Element = T> + Clone,
{
    let shape = [N, N + 1, N + 2, N + 3];
    let input = make_input(context, &shape);
    let factors: Vec<_> = (0..shape.len())
        .map(|mode| make_factor(context, mode, shape[mode]))
        .collect();
    let topology = Topology::new(vec![context.size()]);
    let labels = ["i", "j", "k", "l"];
    let size = shape.iter().product::<usize>() as f64;
    let algebra = Arithmetic::<T>::new();

    for output_mode in 0..shape.len() {
        let references: Vec<_> = (0..shape.len())
            .filter(|&mode| mode != output_mode)
            .map(|mode| &factors[mode])
            .collect();
        let output_distribution =
            Distribution::cyclic(vec![RANK, shape[output_mode]], context.size());
        let actual = input.mttkrp(output_mode, &references, output_distribution.clone());

        let mut expected = Tensor::new(context, output_distribution.clone(), algebra.clone());
        for component in 0..RANK {
            let vectors: Vec<_> = (0..shape.len())
                .filter(|&mode| mode != output_mode)
                .map(|mode| factor_column(context, &factors[mode], component, shape[mode]))
                .collect();
            let vector_factors: Vec<_> = (0..shape.len())
                .filter(|&mode| mode != output_mode)
                .zip(vectors.iter())
                .map(|(mode, vector)| (mode, vector))
                .collect();
            let mut weighted = input.clone();
            weighted.tttp_vectors(&vector_factors);
            let weighted = weighted.into_dense();
            let mut reduced = Tensor::new(
                context,
                Distribution::cyclic(vec![shape[output_mode]], context.size()),
                algebra.clone(),
            );
            reduced
                .sum_from(
                    labels[output_mode],
                    &weighted,
                    "ijkl",
                    topology.clone(),
                    algebra.one(),
                    algebra.zero(),
                )
                .unwrap();
            let pairs: Vec<_> = reduced
                .local_pairs()
                .into_iter()
                .filter(|(key, _)| reduced.distribution().owner(*key) == context.rank())
                .map(|(row, value)| (output_distribution.encode_key(&[component, row]), value))
                .collect();
            expected.write_add(&pairs);
        }

        let norm = residual(&actual, &expected);
        assert!(
            norm / size < 1e-5,
            "MTTKRP mode={output_mode}: norm={norm}, size={size}"
        );
    }
}

fn run(context: &Context<'_>) {
    case::<f32>(context);
    case::<f64>(context);
    case::<Complex<f32>>(context);
    case::<Complex<f64>>(context);
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS mttkrp: sparse order-4 TTTP/MTTKRP, f32/f64/complex32/complex64, norm/size<1e-5; world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
