//! Bounded port of pinned `examples/algebraic_multigrid.cxx`.
//! The source CLI values are fixed at n=4, nlvl=2, ndiv=2, nsmooth=3;
//! `nlvl--` therefore leaves one coarse transition in the V-cycle.

use ctf::{
    algebra::Arithmetic,
    context::Context,
    cost::Models,
    mapping::{Distribution, Topology},
    random::Generator,
    sparse::SparseTensor,
    sparse_search::{Options, Pattern, SearchCache},
    tensor::Tensor,
    topology_candidates,
};

type Algebra = Arithmetic<f32>;
type Matrix<'c, 'r> = SparseTensor<'c, 'r, Algebra>;
type Vector<'c, 'r> = Tensor<'c, 'r, Algebra>;

const N: usize = 4;
const ORIGINAL_LEVELS: usize = 2;
const LEVELS: usize = ORIGINAL_LEVELS - 1;
const NDIV: usize = 2;
const NSMOOTH: usize = 3;
const OMEGA: f32 = 0.333;

fn sparse<'c, 'r>(context: &'c Context<'r>, shape: Vec<usize>) -> Matrix<'c, 'r> {
    Matrix::new(
        context,
        Distribution::cyclic(shape, context.size()),
        Algebra::new(),
    )
}

fn vector<'c, 'r>(context: &'c Context<'r>, length: usize) -> Vector<'c, 'r> {
    Vector::new(
        context,
        Distribution::cyclic(vec![length], context.size()),
        Algebra::new(),
    )
}

fn write_primary(matrix: &mut Matrix<'_, '_>, pairs: Vec<(usize, f32)>) {
    let rank = matrix.context().rank();
    let pairs: Vec<_> = pairs
        .into_iter()
        .filter(|(key, _)| matrix.distribution().owner(*key) == rank)
        .collect();
    matrix.write_add(&pairs);
}

fn nonzeros(matrix: &Matrix<'_, '_>) -> u64 {
    let rank = matrix.context().rank();
    let local = matrix
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| matrix.distribution().owner(*key) == rank)
        .count() as u64;
    matrix
        .context()
        .all_reduce(&Arithmetic::<u64>::new(), &local)
}

fn sparse_product(
    output: &mut Matrix<'_, '_>,
    indices_c: &str,
    a: &Matrix<'_, '_>,
    indices_a: &str,
    b: &Matrix<'_, '_>,
    indices_b: &str,
    alpha: f32,
    beta: f32,
    cache: &mut SearchCache<'_, '_>,
) {
    let selected = cache
        .prepare(
            [
                a.distribution(),
                b.distribution(),
                output.distribution(),
            ],
            [indices_a, indices_b, indices_c],
            [Some(nonzeros(a)), Some(nonzeros(b)), Some(nonzeros(output))],
            None,
        )
        .unwrap()
        .expect("automatic sparse-sparse-sparse search found no valid mapping");
    output.contract_sparse_from_selected(
        indices_c,
        a,
        indices_a,
        b,
        indices_b,
        selected,
        alpha,
        beta,
        true,
    );
}

fn sparse_matvec(
    output: &mut Vector<'_, '_>,
    indices_c: &str,
    a: &Matrix<'_, '_>,
    indices_a: &str,
    x: &Vector<'_, '_>,
    indices_x: &str,
    alpha: f32,
    beta: f32,
    cache: &mut SearchCache<'_, '_>,
) {
    let selected = cache
        .prepare(
            [a.distribution(), x.distribution(), output.distribution()],
            [indices_a, indices_x, indices_c],
            [Some(nonzeros(a)), None, None],
            None,
        )
        .unwrap()
        .expect("automatic sparse-dense-dense search found no valid mapping");
    output.contract_sparse_from_selected(
        indices_c,
        a,
        indices_a,
        x,
        indices_x,
        selected,
        alpha,
        beta,
        true,
    );
}

fn multiply_pointwise(output: &mut Vector<'_, '_>, right: &Vector<'_, '_>) {
    let left = output.clone();
    output.transform_from("i", &left, "i", right, "i", |left, right, value| {
        *value = *left * *right;
    });
}

fn add_scaled(output: &mut Vector<'_, '_>, input: &Vector<'_, '_>, alpha: f32) {
    output
        .sum_from(
            "i",
            input,
            "i",
            Topology::new(vec![output.context().size()]),
            alpha,
            1.0,
        )
        .unwrap();
}

fn diagonal_reciprocal<'c, 'r>(a: &Matrix<'c, 'r>, length: usize) -> Vector<'c, 'r> {
    let mut d = vector(a.context(), length);
    d.sum_from_sparse("i", a, "ii", 1.0, 0.0);
    d.transform(|_, value| {
        *value = if value.abs() > 0.0 {
            1.0 / *value
        } else {
            0.0
        };
    });
    d
}

fn smooth_jacobi(
    a: &Matrix<'_, '_>,
    x: &mut Vector<'_, '_>,
    b: &Vector<'_, '_>,
    nsm: usize,
    cache: &mut SearchCache<'_, '_>,
) {
    let length = x.distribution().shape[0];
    let d = diagonal_reciprocal(a, length);
    let mut remainder = a.clone();
    remainder.transform_stored(|key, value| {
        if key % length == key / length {
            *value = 0.0;
        }
    });

    let mut x1 = vector(x.context(), length);
    for _ in 0..nsm {
        sparse_matvec(
            &mut x1,
            "i",
            &remainder,
            "ij",
            x,
            "j",
            -1.0,
            0.0,
            cache,
        );
        add_scaled(&mut x1, b, 1.0);
        multiply_pointwise(&mut x1, &d);
        x.scale(&(1.0 - OMEGA));
        add_scaled(x, &x1, OMEGA);
    }
}

fn residual<'c, 'r>(
    a: &Matrix<'c, 'r>,
    x: &Vector<'c, 'r>,
    b: &Vector<'c, 'r>,
    cache: &mut SearchCache<'_, '_>,
) -> Vector<'c, 'r> {
    let mut r = b.clone();
    sparse_matvec(&mut r, "i", a, "ij", x, "j", -1.0, 1.0, cache);
    r
}

fn vcycle(
    a: &Matrix<'_, '_>,
    x: &mut Vector<'_, '_>,
    b: &Vector<'_, '_>,
    interpolation: &[Matrix<'_, '_>],
    coarse: &[Matrix<'_, '_>],
    level: usize,
    smoothing: &[usize],
    cache: &mut SearchCache<'_, '_>,
) {
    let length = x.distribution().shape[0];
    if length == 1 {
        let denominator = a.read(&[0])[0];
        let numerator = b.read(&[0])[0];
        x.transform(|_, value| *value = numerator / denominator);
    } else {
        smooth_jacobi(a, x, b, smoothing[0], cache);
    }
    let r = residual(a, x, b, cache);
    if level == 0 {
        return;
    }

    let coarse_length = interpolation[0].distribution().shape[1];
    let mut restricted = vector(x.context(), coarse_length);
    sparse_matvec(
        &mut restricted,
        "i",
        &interpolation[0],
        "ji",
        &r,
        "j",
        1.0,
        1.0,
        cache,
    );
    let mut correction = vector(x.context(), coarse_length);
    vcycle(
        &coarse[0],
        &mut correction,
        &restricted,
        &interpolation[1..],
        &coarse[1..],
        level - 1,
        &smoothing[1..],
        cache,
    );
    sparse_matvec(
        x,
        "i",
        &interpolation[0],
        "ij",
        &correction,
        "j",
        1.0,
        1.0,
        cache,
    );
    smooth_jacobi(a, x, b, smoothing[0], cache);
}

fn poisson<'c, 'r>(context: &'c Context<'r>) -> Matrix<'c, 'r> {
    let length = N * N * N;
    let mut a = sparse(context, vec![length, length]);
    let mut pairs = Vec::new();

    // Preserve `A["ii"]=3; A["ij"] += A["ji"]`: the transpose-add
    // doubles the diagonal as well as mirroring the forward stencil entries.
    for diagonal in 0..length {
        pairs.push((diagonal + diagonal * length, 3.0));
        pairs.push((diagonal + diagonal * length, 3.0));
    }
    for column in 0..length {
        for delta in [1, N, N * N] {
            let inside = match delta {
                1 => (column + 1) % N != 0,
                N => column + N < length && (column / N + 1) % N != 0,
                _ => column + N * N < length && (column / (N * N) + 1) % N != 0,
            };
            if inside {
                let row = column + delta;
                pairs.push((row + column * length, -1.0));
                pairs.push((column + row * length, -1.0));
            }
        }
    }
    write_primary(&mut a, pairs);
    a
}

fn transfers<'c, 'r>(context: &'c Context<'r>) -> Vec<Matrix<'c, 'r>> {
    let mut result = Vec::with_capacity(LEVELS);
    let mut fine = N * N * N;
    let mut nn = N;
    let divisor = NDIV * NDIV * NDIV;
    for _ in 0..LEVELS {
        let coarse = fine / divisor;
        let mut transfer = sparse(context, vec![fine, coarse]);
        let mut pairs = Vec::with_capacity(coarse * divisor);
        for column in 0..coarse {
            let j1 = column / (nn * nn);
            let j2 = (column / nn) % nn;
            let j3 = column % nn;
            for k1 in 0..NDIV {
                for k2 in 0..NDIV {
                    for k3 in 0..NDIV {
                        let row = (j1 * NDIV + k1) * nn * nn
                            + (j2 * NDIV + k2) * nn
                            + j3 * NDIV
                            + k3;
                        pairs.push((row + column * fine, 1.0 / divisor as f32));
                    }
                }
            }
        }
        write_primary(&mut transfer, pairs);
        result.push(transfer);
        fine = coarse;
        // Preserve the source update (`nn=n/ndiv`), rather than repeated division.
        nn = N / NDIV;
    }
    result
}

fn setup<'c, 'r>(
    a: &Matrix<'c, 'r>,
    tentative: &[Matrix<'c, 'r>],
    cache: &mut SearchCache<'_, '_>,
) -> (Vec<Matrix<'c, 'r>>, Vec<Matrix<'c, 'r>>) {
    let mut interpolation = Vec::with_capacity(tentative.len());
    let mut coarse = Vec::with_capacity(tentative.len());
    let mut current = a.clone();

    for t in tentative {
        let rows = current.distribution().shape[0];
        let columns = t.distribution().shape[1];
        let mut d = sparse(a.context(), vec![rows, rows]);
        d.sum_from("ii", &current, "ii", 1.0, 0.0);
        d.transform_stored(|_, value| *value = OMEGA / *value);

        let mut f = sparse(a.context(), vec![rows, columns]);
        sparse_product(
            &mut f, "ik", &current, "ij", t, "jk", 1.0, 0.0, cache,
        );
        let mut p = t.clone();
        sparse_product(&mut p, "ik", &d, "il", &f, "lk", -1.0, 1.0, cache);

        let mut ap = sparse(a.context(), vec![rows, columns]);
        sparse_product(
            &mut ap, "lj", &current, "lk", &p, "kj", 1.0, 0.0, cache,
        );
        let mut ptap = sparse(a.context(), vec![columns, columns]);
        sparse_product(
            &mut ptap, "ij", &p, "li", &ap, "lj", 1.0, 0.0, cache,
        );

        interpolation.push(p);
        current = ptap.clone();
        coarse.push(ptap);
    }
    (interpolation, coarse)
}

fn run(context: &Context<'_>) -> (f64, f64, f64) {
    let length = N * N * N;
    let catalog = topology_candidates::all_shapes(context.size());
    let models = Models::upstream(1);
    let options = Options {
        memory_limit: u64::MAX,
        weight: 0.0,
        allow_exhaustive: true,
    };
    let mut sparse_cache = SearchCache::new(
        context,
        &catalog,
        &models,
        size_of::<f32>(),
        size_of::<u64>() + size_of::<f32>(),
        false,
        Pattern::SparseSparseSparse,
        options,
    );
    let mut matvec_cache = SearchCache::new(
        context,
        &catalog,
        &models,
        size_of::<f32>(),
        size_of::<u64>() + size_of::<f32>(),
        false,
        Pattern::SparseDenseDense { coo_kernel: false },
        options,
    );
    let a = poisson(context);
    let tentative = transfers(context);
    let (interpolation, coarse) = setup(&a, &tentative, &mut sparse_cache);

    let h = 1.0f32 / (N + 1) as f32;
    let mut b = vector(context, length);
    b.transform(|key, value| {
        let x = key / (N * N);
        let y = (key / N) % N;
        let z = key % N;
        *value = ((1.0 / (N + 1) as f64)
            * (1.0 / (N + 1) as f64)
            * (h as f64 * std::f64::consts::PI * (1 + x) as f64).sin()
            * (h as f64 * std::f64::consts::PI * (1 + y) as f64).sin()
            * (h as f64 * std::f64::consts::PI * (1 + z) as f64).sin())
            as f32;
    });
    let mut exact = vector(context, length);
    exact.transform(|key, value| {
        let x = key / (N * N);
        let y = (key / N) % N;
        let z = key % N;
        *value = (1.0 / (3.0 * std::f64::consts::PI * std::f64::consts::PI)
            * (h as f64 * std::f64::consts::PI * (1 + x) as f64).sin()
            * (h as f64 * std::f64::consts::PI * (1 + y) as f64).sin()
            * (h as f64 * std::f64::consts::PI * (1 + z) as f64).sin())
            as f32;
    });
    let _truncation_norm = residual(&a, &exact, &b, &mut matvec_cache).norm2();

    let total = exact.reduce() / length as f32;
    let mut random = vector(context, length);
    let mut generator = Generator::new(context.rank() as u64);
    random.fill_random(-0.1 * total, 0.1 * total, &mut generator);
    let mut x = exact;
    add_scaled(&mut x, &random, 1.0);
    let mut fine_only = x.clone();

    let smoothing = [NSMOOTH; ORIGINAL_LEVELS];
    let started = std::time::Instant::now();
    vcycle(
        &a,
        &mut x,
        &b,
        &interpolation,
        &coarse,
        LEVELS,
        &smoothing,
        &mut matvec_cache,
    );
    let vcycle_seconds = started.elapsed().as_secs_f64();
    smooth_jacobi(
        &a,
        &mut fine_only,
        &b,
        2 * smoothing[0],
        &mut matvec_cache,
    );
    let rnorm_alt = residual(&a, &fine_only, &b, &mut matvec_cache).norm2();
    let rnorm = residual(&a, &x, &b, &mut matvec_cache).norm2();
    assert!(
        rnorm < rnorm_alt,
        "source multigrid residual {rnorm:e} is not below fine-grid Jacobi {rnorm_alt:e}"
    );
    (rnorm, rnorm_alt, vcycle_seconds)
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = Context::world(&universe);
    let (rnorm, rnorm_alt, vcycle_seconds) = run(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS algebraic_multigrid: n=4 original_nlvl=2 ndiv=2 nsmooth=3; multigrid_rnorm={rnorm:e} < jacobi_rnorm={rnorm_alt:e}; vcycle_seconds={vcycle_seconds:.6}; world+parity"
        );
    }
    world.close();
    drop(universe);
}
