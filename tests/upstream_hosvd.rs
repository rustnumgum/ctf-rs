// Adapted from cc4s CTF examples/hosvd.cxx at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f. See LICENSE.
//! Bounded port of the active sequential HOSVD driver in the pinned
//! `examples/hosvd.cxx`.
//!
//! The four source SVDs, the singular-value scaling of the current core, the
//! four-factor reconstruction, and the source residual bound are retained.
//! The fixture uses the existing typed random sparse filler for both a dense
//! allocation (`frac_sp = 1`) and a sparse allocation (`frac_sp = .8`);
//! dimensions are the source mode lengths at bounded `n = 2`.

use ctf::{
    algebra::{Arithmetic, Complex, Group, Monoid, Semiring},
    context::{Context, Runtime},
    mapping::{Distribution, Topology},
    multilinear::tensor_svd::TensorSvd,
    random::Generator,
    sparse::SparseTensor,
    tensor::Tensor,
};

const N: usize = 2;

fn grid(context: &Context<'_>) -> [usize; 2] {
    if context.size() == 4 {
        [2, 2]
    } else {
        [context.size(), 1]
    }
}

fn assert_source_bound(tnorm: f64, residual: f64, rank: usize) {
    assert!(tnorm.is_finite());
    assert!(residual.is_finite());
    let ratio = rank as f64 / N as f64;
    let bound = tnorm * (1.0 - ratio * ratio * ratio * ratio) + 1.0e-4;
    assert!(
        residual <= bound,
        "HOSVD source residual bound: residual={residual}, bound={bound}"
    );
}

macro_rules! hosvd_sequence {
    (
        $context:expr,
        $grid:expr,
        $topology:expr,
        $n:expr,
        $rank:expr,
        $first:expr,
        $algebra:expr
    ) => {{
        let grid = $grid;
        let topology = $topology;
        let n = $n;
        let rank = $rank;
        let algebra = $algebra;

        let (mut u, s1, v1) = $first;
        let old = u.clone();
        u.contract_from(
            "aijk",
            &s1,
            "a",
            &old,
            "aijk",
            topology.clone(),
            algebra.one(),
            algebra.zero(),
        )
        .unwrap();

        let (mut u, s2, v2) = u
            .tensor_svd(
                "aijk",
                "abij",
                'b',
                "bk",
                grid,
                TensorSvd::Truncated {
                    rank: Some(rank + 2),
                    threshold: 0.0,
                },
            )
            .unwrap();
        let old = u.clone();
        u.contract_from(
            "abij",
            &s2,
            "b",
            &old,
            "abij",
            topology.clone(),
            algebra.one(),
            algebra.zero(),
        )
        .unwrap();

        let (mut u, s3, v3) = u
            .tensor_svd(
                "abij",
                "abci",
                'c',
                "cj",
                grid,
                TensorSvd::Truncated {
                    rank: Some(rank + 1),
                    threshold: 0.0,
                },
            )
            .unwrap();
        let old = u.clone();
        u.contract_from(
            "abci",
            &s3,
            "c",
            &old,
            "abci",
            topology.clone(),
            algebra.one(),
            algebra.zero(),
        )
        .unwrap();

        let (u, s4, v4) = u
            .tensor_svd(
                "abci",
                "abcd",
                'd',
                "di",
                grid,
                TensorSvd::Truncated {
                    rank: Some(rank),
                    threshold: 0.0,
                },
            )
            .unwrap();
        let old = u.clone();
        let mut core = old;
        core.contract_from(
            "abcd",
            &s4,
            "d",
            &u,
            "abcd",
            topology.clone(),
            algebra.one(),
            algebra.zero(),
        )
        .unwrap();

        let mut q = Tensor::new(
            $context,
            Distribution::cyclic(vec![rank + 2, rank + 1, rank, n + 3], $context.size()),
            algebra.clone(),
        );
        q.contract_from(
            "bcdl",
            &core,
            "abcd",
            &v1,
            "al",
            topology.clone(),
            algebra.one(),
            algebra.zero(),
        )
        .unwrap();

        let mut q2 = Tensor::new(
            $context,
            Distribution::cyclic(vec![rank + 1, rank, n + 3, n + 2], $context.size()),
            algebra.clone(),
        );
        q2.contract_from(
            "cdlk",
            &q,
            "bcdl",
            &v2,
            "bk",
            topology.clone(),
            algebra.one(),
            algebra.zero(),
        )
        .unwrap();

        let mut q3 = Tensor::new(
            $context,
            Distribution::cyclic(vec![rank, n + 3, n + 2, n + 1], $context.size()),
            algebra.clone(),
        );
        q3.contract_from(
            "dlkj",
            &q2,
            "cdlk",
            &v3,
            "cj",
            topology.clone(),
            algebra.one(),
            algebra.zero(),
        )
        .unwrap();

        let mut reconstructed = Tensor::new(
            $context,
            Distribution::cyclic(vec![n, n + 1, n + 2, n + 3], $context.size()),
            algebra.clone(),
        );
        reconstructed
            .contract_from(
                "ijkl",
                &q3,
                "dlkj",
                &v4,
                "di",
                topology.clone(),
                algebra.one(),
                algebra.zero(),
            )
            .unwrap();
        reconstructed
    }};
}

macro_rules! scalar_case {
    ($name:ident, $scalar:ty, $minimum:expr, $maximum:expr) => {
        fn $name(context: &Context<'_>) {
            let grid = grid(context);
            let topology = Topology::new(grid.to_vec());
            let algebra = Arithmetic::<$scalar>::new();
            let minimum = $minimum;
            let maximum = $maximum;

            for sparse in [false, true] {
                for rank in [1usize, 2] {
                    let shape = vec![N, N + 1, N + 2, N + 3];
                    let seed = context.rank() as u64 * 27
                        + rank as u64 * 101
                        + if sparse { 1009 } else { 0 };
                    let mut generator = Generator::new(seed);
                    let fraction = if sparse { 0.8 } else { 1.0 };

                    if sparse {
                        let mut tensor = SparseTensor::new(
                            context,
                            Distribution::cyclic(shape, context.size()),
                            algebra.clone(),
                        );
                        // This is the Rust adaptation of source fill_sp_random:
                        // sparse storage is generated first, then only present
                        // values receive the typed random sample.
                        tensor.fill_random_sparse(
                            minimum,
                            maximum,
                            fraction,
                            &mut generator,
                        );
                        let tnorm = tensor.norm2();
                        let reconstructed = hosvd_sequence!(
                            context,
                            grid,
                            topology.clone(),
                            N,
                            rank,
                            tensor
                                .tensor_svd_truncated(
                                    "ijkl",
                                    "aijk",
                                    'a',
                                    "al",
                                    grid,
                                    Some(rank + 3),
                                    0.0,
                                )
                                .unwrap(),
                            algebra.clone()
                        );
                        let one = algebra.one();
                        let minus_one = algebra.negate(&one);
                        tensor.sum_from_dense(
                            "ijkl",
                            &reconstructed,
                            "ijkl",
                            minus_one,
                            one,
                        );
                        assert_source_bound(tnorm, tensor.norm2(), rank);
                    } else {
                        let mut tensor = Tensor::new(
                            context,
                            Distribution::cyclic(shape, context.size()),
                            algebra.clone(),
                        );
                        // fraction=1 keeps the dense allocation while using
                        // the same source-compatible candidate/sample order.
                        tensor.fill_random_sparse(
                            minimum,
                            maximum,
                            fraction,
                            &mut generator,
                        );
                        let tnorm = tensor.norm2();
                        let reconstructed = hosvd_sequence!(
                            context,
                            grid,
                            topology.clone(),
                            N,
                            rank,
                            tensor
                                .tensor_svd(
                                    "ijkl",
                                    "aijk",
                                    'a',
                                    "al",
                                    grid,
                                    TensorSvd::Truncated {
                                        rank: Some(rank + 3),
                                        threshold: 0.0,
                                    },
                                )
                                .unwrap(),
                            algebra.clone()
                        );
                        let one = algebra.one();
                        let minus_one = algebra.negate(&one);
                        tensor
                            .sum_from(
                                "ijkl",
                                &reconstructed,
                                "ijkl",
                                topology.clone(),
                                minus_one,
                                one,
                            )
                            .unwrap();
                        assert_source_bound(tnorm, tensor.norm2(), rank);
                    }
                }
            }
        }
    };
}

scalar_case!(real32, f32, -1.0f32, 1.0f32);
scalar_case!(real64, f64, -1.0f64, 1.0f64);
scalar_case!(complex32, Complex<f32>, Complex::new(-1.0f32, 0.0f32), Complex::new(1.0f32, 0.0f32));
scalar_case!(complex64, Complex<f64>, Complex::new(-1.0f64, 0.0f64), Complex::new(1.0f64, 0.0f64));

fn run(context: &Context<'_>) {
    real32(context);
    real64(context);
    complex32(context);
    complex64(context);
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world);

    let rank = world.rank();
    let parity = world
        .split(Some((rank % 2) as i32), rank as i32)
        .unwrap();
    run(&parity);
    parity.close();

    if rank == 0 {
        println!(
            "DIGIT / PASS upstream_hosvd: source sequential four-mode SVD/scaling/reconstruction, n=2 R=1/2, dense+sparse fixtures, f32/f64/Complex32/64, finite source residual bound; world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
