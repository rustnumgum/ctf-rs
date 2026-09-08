//! Reimplementation of pinned test/ccsdt_map_test.cxx.
//! The source is a zero-initialized mapping/execution smoke test, not a
//! nonzero CCSDT numerical reference. Its niter option is unused.
use ctf::{
    algebra::Arithmetic,
    context::Context,
    linalg::Native,
    mapping::{Distribution, Topology},
    tensor::Tensor,
};

fn run(context: &Context<'_>) {
    let n = 4;
    let make = |order| {
        Tensor::new(
            context,
            Distribution::cyclic(vec![n; order], context.size()),
            Arithmetic::<f64>::new(),
        )
    };
    let w = make(4);
    let t = make(4);
    let mut z = make(6);
    let topology = Topology::new(if context.size() == 4 {
        vec![2, 2]
    } else {
        vec![context.size()]
    });
    z.contract_blas_on_grid::<Native>("hijmno", &w, "hijk", &t, "kmno", topology, 1.0, 1.0)
        .unwrap();
    // Explicitly record the zero-input invariant; this is not evidence for
    // nonzero numerical accuracy (the original driver has no such assertion).
    assert!(z.local_pairs().iter().all(|(_, value)| *value == 0.0));
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
            "DIGIT / PASS upstream_ccsdt_map: source n=4 zero-input six-order mapping smoke; world+parity"
        );
    }
    world.close();
    drop(universe);
}
