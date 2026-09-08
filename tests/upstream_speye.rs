//! Bounded port of pinned CTF test/speye.cxx.

use ctf::{algebra::Arithmetic, context::Context, mapping::Distribution, sparse::SparseTensor};

const N: usize = 4;
const ORDER: usize = 3;
const TOLERANCE: f64 = 1.0e-9;

fn run(context: &Context<'_>) {
    let distribution = Distribution::cyclic(vec![N; ORDER], context.size());
    let mut identity = SparseTensor::new(context, distribution, Arithmetic::<f64>::new());
    let repeated = "i".repeat(ORDER);
    let mut one = SparseTensor::new(
        context,
        Distribution::cyclic(vec![], context.size()),
        Arithmetic::<f64>::new(),
    );
    one.write_add(&if context.rank() == 0 {
        vec![(0, 1.0)]
    } else {
        vec![]
    });
    // Exercise indexed scalar broadcast itself, not precomputed diagonal keys.
    identity.sum_from(&repeated, &one, "", 1.0, 0.0);
    let changed: String = (0..ORDER)
        .map(|axis| char::from(b'i' + axis as u8))
        .collect();
    let scalar_distribution = Distribution::cyclic(Vec::new(), context.size());
    let mut sum1 = SparseTensor::new(
        context,
        scalar_distribution.clone(),
        Arithmetic::<f64>::new(),
    );
    let mut sum2 = SparseTensor::new(context, scalar_distribution, Arithmetic::<f64>::new());
    // Upstream reductions: sum1 = A["ijk..."]; sum2 = A["iii..."].
    sum1.sum_from("", &identity, &changed, 1.0, 0.0);
    sum2.sum_from("", &identity, &repeated, 1.0, 0.0);

    let sum1 = sum1.read(&[0])[0];
    let sum2 = sum2.read(&[0])[0];
    assert!((sum1 - N as f64).abs() < TOLERANCE);
    assert!((sum2 - N as f64).abs() < TOLERANCE);
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let rank = world.rank();
    run(&world);

    let parity = world.split(Some((rank % 2) as i32), rank as i32).unwrap();
    run(&parity);
    parity.close();

    if rank == 0 {
        println!(
            "DIGIT / PASS upstream_speye: sparse A[iii]=1, distinct/repeated reductions equal n; abs<1e-9; world+parity"
        );
    }
    world.close();
    drop(universe);
}
