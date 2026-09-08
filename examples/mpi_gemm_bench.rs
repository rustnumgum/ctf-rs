//! Single explicitly requested representative measurement, not a speedup claim.
use ctf::{algebra::Arithmetic, linalg::Native, mapping::Distribution, tensor::Tensor};
fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let np = world.size();
    assert!([1, 2, 4].contains(&np));
    let grid = if np == 4 { [2, 2] } else { [np, 1] };
    let (m, k, n) = (128, 192, 160);
    let make = |shape| {
        Tensor::new(
            &world,
            Distribution::cyclic(shape, np),
            Arithmetic::<f64>::new(),
        )
    };
    let mut a = make(vec![m, k]);
    let mut b = make(vec![k, n]);
    let mut c = make(vec![m, n]);
    a.transform(|key, v| *v = ((key % 17) + 1) as f64 / 17.);
    b.transform(|key, v| *v = ((key % 13) + 1) as f64 / 13.);
    world.barrier();
    let start = std::time::Instant::now();
    c.gemm_2d::<Native>(&a, &b, grid, 1., 0.);
    world.barrier();
    let elapsed = start.elapsed().as_secs_f64();
    if world.rank() == 0 {
        println!(
            "gemm m={m} k={k} n={n} ranks={np} grid={}x{} elapsed_s={elapsed:.6}",
            grid[0], grid[1]
        );
    }
    drop(c);
    drop(b);
    drop(a);
    world.close();
    drop(universe);
}
