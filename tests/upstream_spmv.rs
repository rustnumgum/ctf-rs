//! Bounded port of the active numerical path in `examples/spmv.cxx`.
//!
//! The acceptance below preserves its sparse A, dense/sparse output flags,
//! operand-order (dense*bivariate sparse) term, and norm criteria.

use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Topology},
    random::Generator,
    sparse::SparseTensor,
    tensor::Tensor,
};

const N: usize = 5;
const DENSE_A_FRACTION: f64 = 0.5 / N as f64;
const C1_FRACTION: f64 = 0.5;
const INITIAL_CUTOFF: f64 = 1.0e-6;
const RESIDUAL_TOLERANCE: f64 = 1.0e-6;

type Algebra = Arithmetic<f64>;

fn grid(context: &Context<'_>) -> [usize; 2] {
    if context.size() == 4 {
        [2, 2]
    } else {
        [context.size(), 1]
    }
}

fn inputs<'c, 'r>(
    context: &'c Context<'r>,
) -> (
    Tensor<'c, 'r, Algebra>,
    Tensor<'c, 'r, Algebra>,
    Tensor<'c, 'r, Algebra>,
    SparseTensor<'c, 'r, Algebra>,
) {
    let mut generator = Generator::new(17 + context.rank() as u64);
    let matrix_distribution = Distribution::cyclic(vec![N, N], context.size());
    let vector_distribution = Distribution::cyclic(vec![N], context.size());

    let mut b = Tensor::new(context, vector_distribution.clone(), Algebra::new());
    b.fill_random(0.0, 1.0, &mut generator);

    // The source calls fill_sp_random on dense c1 and dnA, producing dense
    // storage with sparse-valued fixtures; fill_random_sparse has the same
    // storage/algebra contract. This bounded fixture explicitly seeds CTF's
    // MT generator; it does not treat the source srand48 call as its MT seed.
    let mut c1 = Tensor::new(context, vector_distribution, Algebra::new());
    c1.fill_random_sparse(0.0, 1.0, C1_FRACTION, &mut generator);
    let mut dn_a = Tensor::new(context, matrix_distribution.clone(), Algebra::new());
    dn_a.fill_random_sparse(0.0, 1.0, DENSE_A_FRACTION, &mut generator);

    let mut sp_a = SparseTensor::new(context, matrix_distribution, Arithmetic::new());
    // spA["ij"] += dnA["ij"]: no dense staging in the sparse destination.
    sp_a.sum_from_dense("ij", &dn_a, "ij", 1.0, 0.0);
    (b, dn_a, c1, sp_a)
}

fn check_dense_output(
    context: &Context<'_>,
    b: &Tensor<'_, '_, Algebra>,
    dn_a: &Tensor<'_, '_, Algebra>,
    c1_initial: &Tensor<'_, '_, Algebra>,
    sp_a: &SparseTensor<'_, '_, Algebra>,
) {
    let topology = Topology::new(grid(context).to_vec());
    let mut c1 = c1_initial.clone();
    c1.contract_from("i", dn_a, "ij", b, "j", topology.clone(), 1.0, 1.0)
        .unwrap();

    let vector_distribution = Distribution::cyclic(vec![N], context.size());
    let mut c2 = Tensor::new(context, vector_distribution, Arithmetic::new());
    c2.sum_from("i", c1_initial, "i", topology.clone(), 1.0, 0.0)
        .unwrap();
    c2.contract_from_sparse_dense("i", sp_a, "ij", b, "j", grid(context), 0.5, 1.0)
        .unwrap();
    // This is the source's b["j"]*spA["ij"] term.  The dense/sparse
    // entry point preserves that operand order instead of replacing it with
    // a dense product.
    c2.contract_from_dense_sparse("i", b, "j", sp_a, "ij", grid(context), 0.5, 1.0)
        .unwrap();

    let initial_norm = c2.norm2();
    assert!(initial_norm.is_finite() && initial_norm >= INITIAL_CUTOFF);
    c2.sum_from("i", &c1, "i", topology, -1.0, 1.0).unwrap();
    let residual = c2.norm2();
    assert!(residual.is_finite() && residual <= RESIDUAL_TOLERANCE);
}

fn check_sparse_output(
    context: &Context<'_>,
    b: &Tensor<'_, '_, Algebra>,
    c1_initial: &Tensor<'_, '_, Algebra>,
    dn_a: &Tensor<'_, '_, Algebra>,
    sp_a: &SparseTensor<'_, '_, Algebra>,
) {
    let grid = grid(context);
    let mut c1 = c1_initial.clone();
    c1.contract_from(
        "i",
        dn_a,
        "ij",
        b,
        "j",
        Topology::new(grid.clone().to_vec()),
        1.0,
        1.0,
    )
    .unwrap();
    let vector_distribution = Distribution::cyclic(vec![N], context.size());
    let mut c2 = SparseTensor::new(context, vector_distribution, Arithmetic::new());
    // c2["i"] = c1["i"] retains sparse output storage and filters dense
    // zeros during the dense-to-sparse indexed sum.
    c2.sum_from_dense("i", c1_initial, "i", 1.0, 0.0);
    c2.contract_from_sparse_dense("i", sp_a, "ij", b, "j", grid, 0.5, 1.0)
        .unwrap();
    c2.contract_from_dense_sparse("i", b, "j", sp_a, "ij", grid, 0.5, 1.0)
        .unwrap();

    let initial_norm = c2.norm2();
    assert!(initial_norm.is_finite() && initial_norm >= INITIAL_CUTOFF);
    // c2["i"] -= c1["i"] without densifying the sparse output.
    c2.sum_from_dense("i", &c1, "i", -1.0, 1.0);
    let residual = c2.norm2();
    assert!(residual.is_finite() && residual <= RESIDUAL_TOLERANCE);
}

fn run(context: &Context<'_>) {
    let (b, dn_a, c1_initial, sp_a) = inputs(context);
    check_dense_output(context, &b, &dn_a, &c1_initial, &sp_a);
    check_sparse_output(context, &b, &c1_initial, &dn_a, &sp_a);
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    run(&world);

    let rank = world.rank();
    let parity = world.split(Some((rank % 2) as i32), rank as i32).unwrap();
    run(&parity);
    parity.close();

    if rank == 0 {
        println!(
            "DIGIT / PASS upstream_spmv: sparse A*dense b, dense/sparse outputs, source operand-order second term; initial norm>=1e-6 and residual<=1e-6; world+parity"
        );
    }
    world.close();
    drop(universe);
}
