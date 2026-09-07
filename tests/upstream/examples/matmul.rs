//! Pinned `examples/matmul.cxx`: sparse matrix multiplication with a dense
//! reference and the source's default dimensions and sparsity fractions.

use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Topology},
    random::Generator,
    sparse::SparseTensor,
};

fn grid(context: &Context<'_>) -> [usize; 2] {
    [context.size(), 1]
}

fn matmul(context: &Context<'_>) -> f64 {
    let (m, n, k) = (17usize, 32usize, 9usize);
    let fraction = 0.2;
    let algebra = Arithmetic::<f64>::new();
    let mut generator = Generator::new(context.rank() as u64);

    let mut a = SparseTensor::new(
        context,
        Distribution::cyclic(vec![m, k], context.size()),
        algebra.clone(),
    );
    let mut b = SparseTensor::new(
        context,
        Distribution::cyclic(vec![k, n], context.size()),
        algebra.clone(),
    );
    let mut c = SparseTensor::new(
        context,
        Distribution::cyclic(vec![m, n], context.size()),
        algebra.clone(),
    );
    a.fill_random_sparse(0.0, 1.0, fraction, &mut generator);
    b.fill_random_sparse(0.0, 1.0, fraction, &mut generator);
    c.fill_random_sparse(0.0, 1.0, fraction, &mut generator);

    let mut reference = c.clone().into_dense();
    let dense_a = a.clone().into_dense();
    let dense_b = b.clone().into_dense();
    reference
        .contract_from(
            "ij",
            &dense_a,
            "ik",
            &dense_b,
            "kj",
            Topology::new(vec![context.size()]),
            0.5,
            1.0,
        )
        .unwrap();
    reference
        .contract_from(
            "ij",
            &dense_a,
            "ik",
            &dense_b,
            "kj",
            Topology::new(vec![context.size()]),
            0.5,
            1.0,
        )
        .unwrap();

    c.gemm_sparse(&a, &b, grid(context), 0.5, 1.0);
    c.gemm_sparse(&a, &b, grid(context), 0.5, 1.0);
    let actual = c.into_dense();
    reference
        .sum_from(
            "ij",
            &actual,
            "ij",
            Topology::new(vec![context.size()]),
            -1.0,
            1.0,
        )
        .unwrap();
    reference.norm2()
}

fn run(context: &Context<'_>) {
    let error = matmul(context);
    assert!(error <= 1.0e-6, "matmul residual {error:e}");
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
            "DIGIT / PASS matmul: sparse A/B/C GEMM equals dense reference; m=17 n=32 k=9 sp=0.2; residual<=1e-6; world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
