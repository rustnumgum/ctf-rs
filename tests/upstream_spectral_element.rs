//! Bounded port of the active kernel in pinned `examples/spectral_element.cxx`.
//!
//! The source's G tensors are rank-three random tensors (despite the comment
//! calling them diagonal).  This keeps the three D/u contractions, the nine
//! elementwise G*w accumulations, and the three final contractions exactly;
//! only the benchmark timing and the source default size are omitted.

use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Topology},
    random::Generator,
    tensor::Tensor,
};

const N: usize = 3;
type Algebra = Arithmetic<f64>;
type Dense<'c, 'r> = Tensor<'c, 'r, Algebra>;

fn dense<'c, 'r>(context: &'c Context<'r>, shape: Vec<usize>) -> Dense<'c, 'r> {
    Tensor::new(
        context,
        Distribution::cyclic(shape, context.size()),
        Algebra::new(),
    )
}

fn run(context: &Context<'_>) {
    let topology = Topology::new(vec![context.size()]);
    // The source uses its random-fill helper without a numerical oracle.  The
    // existing typed Generator gives this bounded port a reproducible rank-local
    // stream while preserving source allocation order: u, D, then G[0..3][0..3].
    let mut generator = Generator::new(context.rank() as u64 * 27);
    let mut u = dense(context, vec![N, N, N]);
    u.fill_random(0.0, 1.0, &mut generator);
    let mut d = dense(context, vec![N, N]);
    d.fill_random(0.0, 1.0, &mut generator);
    let g: Vec<Vec<Dense<'_, '_>>> = (0..3)
        .map(|_| {
            (0..3)
                .map(|_| {
                    let mut tensor = dense(context, vec![N, N, N]);
                    tensor.fill_random(0.0, 1.0, &mut generator);
                    tensor
                })
                .collect()
        })
        .collect();
    let mut w: Vec<_> = (0..3)
        .map(|_| dense(context, vec![N, N, N]))
        .collect();
    let mut z: Vec<_> = (0..3)
        .map(|_| dense(context, vec![N, N, N]))
        .collect();

    // Preserve the source's initial u for all three first-stage products;
    // source assignment overwrites u only after z has been formed.
    let initial_u = u.clone();
    w[0]
        .contract_from(
            "ijk",
            &d,
            "kl",
            &initial_u,
            "ijl",
            topology.clone(),
            1.0,
            0.0,
        )
        .unwrap();
    w[1]
        .contract_from(
            "ijk",
            &d,
            "jl",
            &initial_u,
            "ilk",
            topology.clone(),
            1.0,
            0.0,
        )
        .unwrap();
    w[2]
        .contract_from(
            "ijk",
            &d,
            "il",
            &initial_u,
            "ljk",
            topology.clone(),
            1.0,
            0.0,
        )
        .unwrap();

    for a in 0..3 {
        for b in 0..3 {
            // All labels occur in all three tensors, so this is the source's
            // elementwise product rather than a reduction over an index.
            z[a]
                .contract_from(
                    "ijk",
                    &g[a][b],
                    "ijk",
                    &w[b],
                    "ijk",
                    topology.clone(),
                    1.0,
                    1.0,
                )
                .unwrap();
        }
    }

    // Source `u[ijk] = ...` applies beta=0 on the first term and beta=1 on
    // the two subsequent terms.
    u.contract_from(
        "ijk",
        &d,
        "lk",
        &z[0],
        "ijl",
        topology.clone(),
        1.0,
        0.0,
    )
    .unwrap();
    u.contract_from(
        "ijk",
        &d,
        "lj",
        &z[1],
        "ilk",
        topology.clone(),
        1.0,
        1.0,
    )
    .unwrap();
    u.contract_from(
        "ijk",
        &d,
        "li",
        &z[2],
        "ljk",
        topology,
        1.0,
        1.0,
    )
    .unwrap();

    // Preserve the source-only acceptance quantity and bound.
    let norm = u.norm2();
    assert!(norm.is_finite() && norm >= 1.0e-6);
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
            "DIGIT / PASS upstream_spectral_element: source D/u/G spectral kernel, bounded n=3, norm2>=1e-6; world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
