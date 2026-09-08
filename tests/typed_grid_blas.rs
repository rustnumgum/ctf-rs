use ctf::{
    algebra::{Arithmetic, Complex, Monoid, Semiring, Wire},
    context::Context,
    linalg::{GemmKernel, Native},
    mapping::{Distribution, Topology},
    tensor::Tensor,
};

fn matrix_a<T>(value: &impl Fn(i32, i32) -> T, i: usize, k: usize) -> T {
    value(1 + i as i32 + 2 * k as i32, k as i32 - i as i32)
}

fn matrix_b<T>(value: &impl Fn(i32, i32) -> T, k: usize, j: usize) -> T {
    value(2 + k as i32 - j as i32, 1 + j as i32 + k as i32)
}

fn exercise_gemm<T>(
    context: &Context<'_>,
    grid: [usize; 2],
    value: &impl Fn(i32, i32) -> T,
    close: &impl Fn(&T, &T) -> bool,
) where
    T: Clone + PartialEq + Wire,
    Arithmetic<T>: Semiring<Element = T> + Clone,
    Native: GemmKernel<T>,
{
    let algebra = Arithmetic::<T>::new();
    let make = |shape| {
        Tensor::new(
            context,
            Distribution::cyclic(shape, context.size()),
            algebra.clone(),
        )
    };
    let mut a = make(vec![5, 3]);
    let mut b = make(vec![3, 2]);
    let mut c = make(vec![5, 2]);
    a.transform(|key, entry| *entry = matrix_a(value, key % 5, key / 5));
    b.transform(|key, entry| *entry = matrix_b(value, key % 3, key / 3));
    let old_c = value(3, -2);
    c.transform(|_, entry| *entry = old_c.clone());
    let alpha = value(2, 1);
    let beta = value(-1, 2);
    let old_distributions = [
        a.distribution().clone(),
        b.distribution().clone(),
        c.distribution().clone(),
    ];
    let old_a = a.local_storage().to_vec();
    let old_b = b.local_storage().to_vec();
    c.gemm_2d::<Native>(&a, &b, grid, alpha.clone(), beta.clone());
    assert_eq!(
        [a.distribution(), b.distribution(), c.distribution()],
        old_distributions.each_ref()
    );
    assert!(a.local_storage() == old_a.as_slice());
    assert!(b.local_storage() == old_b.as_slice());
    for (key, actual) in c.local_pairs() {
        let (i, j) = (key % 5, key / 5);
        let mut sum = algebra.zero();
        for k in 0..3 {
            sum = algebra.add(
                &sum,
                &algebra.multiply(&matrix_a(value, i, k), &matrix_b(value, k, j)),
            );
        }
        let expected = algebra.add(
            &algebra.multiply(&alpha, &sum),
            &algebra.multiply(&beta, &old_c),
        );
        assert!(
            close(&actual, &expected),
            "typed gemm rank {} grid {grid:?} key {key}",
            context.rank()
        );
    }

    let mut tiny_a = make(vec![1, 1]);
    let mut tiny_b = make(vec![1, 1]);
    let mut tiny_c = make(vec![1, 1]);
    tiny_a.transform(|_, entry| *entry = value(2, 1));
    tiny_b.transform(|_, entry| *entry = value(-1, 2));
    tiny_c.transform(|_, entry| *entry = value(4, -1));
    let tiny_a_data = tiny_a.local_storage().to_vec();
    let tiny_b_data = tiny_b.local_storage().to_vec();
    let tiny_distribution = tiny_c.distribution().clone();
    tiny_c.gemm_2d::<Native>(&tiny_a, &tiny_b, grid, alpha.clone(), beta.clone());
    assert_eq!(tiny_c.distribution(), &tiny_distribution);
    assert!(tiny_a.local_storage() == tiny_a_data.as_slice());
    assert!(tiny_b.local_storage() == tiny_b_data.as_slice());
    let tiny_product = algebra.multiply(&value(2, 1), &value(-1, 2));
    let tiny_expected = algebra.add(
        &algebra.multiply(&alpha, &tiny_product),
        &algebra.multiply(&beta, &value(4, -1)),
    );
    for (_, actual) in tiny_c.local_pairs() {
        assert!(close(&actual, &tiny_expected));
    }
}

fn batch_a<T>(value: &impl Fn(i32, i32) -> T, l: usize, i: usize, k: usize) -> T {
    value(
        1 + l as i32 + i as i32 + 2 * k as i32,
        l as i32 + k as i32 - i as i32,
    )
}

fn batch_b<T>(value: &impl Fn(i32, i32) -> T, l: usize, j: usize, k: usize) -> T {
    value(
        2 + 2 * l as i32 + k as i32 - j as i32,
        1 + j as i32 + k as i32 - l as i32,
    )
}

fn exercise_folded<T>(
    context: &Context<'_>,
    grid: [usize; 2],
    value: &impl Fn(i32, i32) -> T,
    close: &impl Fn(&T, &T) -> bool,
) where
    T: Clone + PartialEq + Wire,
    Arithmetic<T>: Semiring<Element = T> + Clone,
    Native: GemmKernel<T>,
{
    let algebra = Arithmetic::<T>::new();
    let make = |shape| {
        Tensor::new(
            context,
            Distribution::cyclic(shape, context.size()),
            algebra.clone(),
        )
    };
    // Batch l is deliberately first and the matrix axes use noncanonical orders.
    let mut a = make(vec![2, 5, 3]); // lik
    let mut b = make(vec![2, 2, 3]); // ljk
    let mut c = make(vec![2, 2, 5]); // lji
    a.transform(|key, entry| {
        let l = key % 2;
        let i = key / 2 % 5;
        let k = key / 10;
        *entry = batch_a(value, l, i, k);
    });
    b.transform(|key, entry| {
        let l = key % 2;
        let j = key / 2 % 2;
        let k = key / 4;
        *entry = batch_b(value, l, j, k);
    });
    c.transform(|key, entry| {
        let l = key % 2;
        let j = key / 2 % 2;
        let i = key / 4;
        *entry = value(3 + i as i32, -2 + l as i32 + j as i32);
    });
    let alpha = value(2, 1);
    let beta = value(-1, 2);
    let old_distributions = [
        a.distribution().clone(),
        b.distribution().clone(),
        c.distribution().clone(),
    ];
    let old_a = a.local_storage().to_vec();
    let old_b = b.local_storage().to_vec();
    c.contract_blas_on_grid::<Native>(
        "lji",
        &a,
        "lik",
        &b,
        "ljk",
        Topology::new(grid.to_vec()),
        alpha.clone(),
        beta.clone(),
    )
    .unwrap();
    assert_eq!(
        [a.distribution(), b.distribution(), c.distribution()],
        old_distributions.each_ref()
    );
    assert!(a.local_storage() == old_a.as_slice());
    assert!(b.local_storage() == old_b.as_slice());
    for (key, actual) in c.local_pairs() {
        let l = key % 2;
        let j = key / 2 % 2;
        let i = key / 4;
        let mut sum = algebra.zero();
        for k in 0..3 {
            sum = algebra.add(
                &sum,
                &algebra.multiply(&batch_a(value, l, i, k), &batch_b(value, l, j, k)),
            );
        }
        let old = value(3 + i as i32, -2 + l as i32 + j as i32);
        let expected = algebra.add(
            &algebra.multiply(&alpha, &sum),
            &algebra.multiply(&beta, &old),
        );
        assert!(
            close(&actual, &expected),
            "typed fold rank {} grid {grid:?} key {key}",
            context.rank()
        );
    }
}

fn exercise_type<T>(
    context: &Context<'_>,
    value: impl Fn(i32, i32) -> T,
    close: impl Fn(&T, &T) -> bool,
) where
    T: Clone + PartialEq + Wire,
    Arithmetic<T>: Semiring<Element = T> + Clone,
    Native: GemmKernel<T>,
{
    let mut grids = vec![[context.size(), 1], [1, context.size()]];
    if context.size() == 4 {
        grids.push([2, 2]);
    }
    grids.dedup();
    for grid in grids {
        exercise_gemm(context, grid, &value, &close);
        exercise_folded(context, grid, &value, &close);
    }
}

fn exercise(context: &Context<'_>) {
    exercise_type(
        context,
        |real, _| real as f32,
        |actual, expected| actual.is_finite() && (actual - expected).abs() < 1e-6,
    );
    exercise_type(
        context,
        |real, _| real as f64,
        |actual, expected| actual.is_finite() && (actual - expected).abs() < 1e-6,
    );
    exercise_type(
        context,
        |real, imaginary| Complex::new(real as f32, imaginary as f32),
        |actual, expected| {
            actual.re.is_finite()
                && actual.im.is_finite()
                && (actual.re - expected.re).abs() < 1e-6
                && (actual.im - expected.im).abs() < 1e-6
        },
    );
    exercise_type(
        context,
        |real, imaginary| Complex::new(real as f64, imaginary as f64),
        |actual, expected| {
            actual.re.is_finite()
                && actual.im.is_finite()
                && (actual.re - expected.re).abs() < 1e-6
                && (actual.im - expected.im).abs() < 1e-6
        },
    );
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    exercise(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    exercise(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS typed_grid_blas: four scalar types, rectangular/square grids, uneven and empty shards, reordered batches; exact inputs/layouts and finite abs<1e-6; world+parity"
        );
    }
    world.close();
    drop(universe);
}
