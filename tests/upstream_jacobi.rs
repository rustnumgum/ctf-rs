// Adapted from cc4s CTF examples/jacobi.cxx at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f. See LICENSE.
//! Bounded port of the active Jacobi driver in pinned `examples/jacobi.cxx`.
//!
//! The dense and sparse iterations use the distributed contraction APIs rather
//! than gathering a matrix or vector.  The source's sparse threshold, diagonal
//! shift, two residual stopping loops, and final comparison are retained.

use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Topology},
    random::Generator,
    sparse::SparseTensor,
    tensor::Tensor,
};

const N: usize = 3;
type Algebra = Arithmetic<f64>;
type Dense<'c, 'r> = Tensor<'c, 'r, Algebra>;
type Sparse<'c, 'r> = SparseTensor<'c, 'r, Algebra>;

fn dense<'c, 'r>(context: &'c Context<'r>, shape: Vec<usize>) -> Dense<'c, 'r> {
    Tensor::new(
        context,
        Distribution::cyclic(shape, context.size()),
        Algebra::new(),
    )
}

fn jacobi_dense<'c, 'r>(
    remainder: &Dense<'c, 'r>,
    b: &Dense<'c, 'r>,
    d: &Dense<'c, 'r>,
    x: &mut Dense<'c, 'r>,
    topology: &Topology,
) {
    let old = x.clone();
    x.contract_from("i", remainder, "ij", &old, "j", topology.clone(), -1.0, 0.0)
        .unwrap();
    x.sum_from("i", b, "i", topology.clone(), 1.0, 1.0).unwrap();
    let old = x.clone();
    x.contract_from("i", d, "i", &old, "i", topology.clone(), 1.0, 0.0)
        .unwrap();
}

fn jacobi_sparse<'c, 'r>(
    remainder: &Sparse<'c, 'r>,
    b: &Dense<'c, 'r>,
    d: &Dense<'c, 'r>,
    x: &mut Dense<'c, 'r>,
    topology: &Topology,
    grid: [usize; 2],
) {
    let old = x.clone();
    x.contract_from_sparse_dense("i", remainder, "ij", &old, "j", grid, -1.0, 0.0)
        .unwrap();
    x.sum_from("i", b, "i", topology.clone(), 1.0, 1.0).unwrap();
    let old = x.clone();
    x.contract_from("i", d, "i", &old, "i", topology.clone(), 1.0, 0.0)
        .unwrap();
}

fn dense_residual<'c, 'r>(
    matrix: &Dense<'c, 'r>,
    b: &Dense<'c, 'r>,
    x: &Dense<'c, 'r>,
    topology: &Topology,
) -> Dense<'c, 'r> {
    let mut result = b.clone();
    result
        .contract_from("i", matrix, "ij", x, "j", topology.clone(), -1.0, 1.0)
        .unwrap();
    result
}

fn sparse_residual<'c, 'r>(
    matrix: &Sparse<'c, 'r>,
    b: &Dense<'c, 'r>,
    x: &Dense<'c, 'r>,
    grid: [usize; 2],
) -> Dense<'c, 'r> {
    let mut result = b.clone();
    result
        .contract_from_sparse_dense("i", matrix, "ij", x, "j", grid, -1.0, 1.0)
        .unwrap();
    result
}

fn run(context: &Context<'_>) {
    let grid = [context.size(), 1];
    let topology = Topology::new(grid.to_vec());
    let mut generator = Generator::new(context.rank() as u64 * 27);

    // Source order is b, c1, c2=c1, then dense dnA.  Generator is the
    // reproducible bounded fixture adapter for source srand48(rank).
    let mut b = dense(context, vec![N]);
    b.fill_random(0.0, 1.0, &mut generator);
    let mut c1 = dense(context, vec![N]);
    c1.fill_random(0.0, 1.0, &mut generator);
    let mut c2 = c1.clone();

    let mut source_a = dense(context, vec![N, N]);
    source_a.fill_random(0.0, 1.0, &mut generator);
    // spA.sparsify(.5) uses the source's default absolute threshold behavior;
    // positive random values therefore retain precisely values greater than .5.
    let mut sp_a = source_a.into_sparse(|value| value.abs() > 0.5);
    let diagonal: Vec<_> = (0..N).map(|index| index + N * index).collect();
    let diagonal_shift: Vec<_> = diagonal
        .iter()
        .copied()
        .filter(|&key| sp_a.distribution().owner(key) == context.rank())
        .map(|key| (key, 2.0 * N as f64))
        .collect();
    sp_a.write_add(&diagonal_shift);

    // dnA = spA after the shift, while dnR/spR are the corresponding
    // off-diagonal remainder matrices with source diagonal-zero behavior.
    let dn_a = sp_a.clone().into_dense();
    let mut dn_r = dn_a.clone();
    dn_r.transform_indexed("ii", |value| *value = 0.0);
    let mut sp_r = sp_a.clone();
    let diagonal_zero: Vec<_> = diagonal
        .iter()
        .copied()
        .filter(|&key| sp_r.distribution().owner(key) == context.rank())
        .map(|key| (key, 0.0))
        .collect();
    sp_r.write_scaled(&diagonal_zero, &1.0, &0.0);

    let owned_diagonal: Vec<_> = diagonal
        .iter()
        .copied()
        .filter(|&key| sp_a.distribution().owner(key) == context.rank())
        .collect();
    let diagonal_values = sp_a.read(&owned_diagonal);
    let mut d = dense(context, vec![N]);
    let inverse_diagonal: Vec<_> = owned_diagonal
        .iter()
        .zip(diagonal_values)
        .map(|(&key, value)| (key % N, 1.0 / value))
        .collect();
    d.write_add(&inverse_diagonal);

    for _ in 0..100 {
        jacobi_dense(&dn_r, &b, &d, &mut c1, &topology);
        let residual = dense_residual(&dn_a, &b, &c1, &topology);
        let residual_norm = residual.norm2();
        assert!(residual_norm.is_finite());
        if residual_norm < 1.0e-4 {
            break;
        }
    }
    for _ in 0..100 {
        jacobi_sparse(&sp_r, &b, &d, &mut c2, &topology, grid);
        let residual = sparse_residual(&sp_a, &b, &c2, grid);
        let residual_norm = residual.norm2();
        assert!(residual_norm.is_finite());
        if residual_norm < 1.0e-4 {
            break;
        }
    }

    c2.sum_from("i", &c1, "i", topology, -1.0, 1.0).unwrap();
    // Preserve the source's final acceptance criterion exactly.
    assert!(c2.norm2() <= 1.0e-6);
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
            "DIGIT / PASS upstream_jacobi: dense/sparse Jacobi iterations, source residual stopping threshold 1e-4 and final difference<=1e-6; bounded n=3; world+parity"
        );
    }
    world.close();
    drop(universe);
}
