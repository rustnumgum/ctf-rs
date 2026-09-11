use std::path::Path;

use ctf::{
    algebra::Arithmetic,
    context::Context,
    cost::Models,
    mapping::Distribution,
    random::Generator,
    sparse::SparseTensor,
    sparse_search::StorageSize,
    sparse_sum_search::{Options, Pattern, SearchCache},
    topology_candidates,
};

const N: usize = 3;
const SPARSE_FRACTION: f64 = 0.1;
const NORM_BOUND: f64 = 1.0e-7 * N as f64 * N as f64 * 0.1 * N as f64;

type Sparse<'c, 'r> = SparseTensor<'c, 'r, Arithmetic<f64>>;

fn checkpoint<'c, 'r>(context: &'c Context<'r>, path: &Path) -> f64 {
    let distribution = Distribution::cyclic(vec![N, N, N], context.size());
    let mut generator = Generator::new(context.rank() as u64);
    let mut u: Sparse<'c, 'r> =
        SparseTensor::new(context, distribution.clone(), Arithmetic::<f64>::new());
    u.fill_random_sparse(0., 1., SPARSE_FRACTION, &mut generator);
    u.write_sparse_to_file(path, true, false);

    let mut v: Sparse<'c, 'r> =
        SparseTensor::new(context, distribution, Arithmetic::<f64>::new());
    v.read_sparse_from_file(path, true, false);
    let catalog = topology_candidates::all_shapes(context.size());
    let models = Models::upstream(1);
    let mut cache = SearchCache::new(context, &catalog, &models,
        [StorageSize { element_bytes: 8, pair_bytes: 16 }; 2],
        false, Pattern::SparseSparse, Options { memory_limit: u64::MAX });
    let count = |tensor: &Sparse<'_, '_>| {
        let local = tensor.local_pairs().into_iter()
            .filter(|(key, _)| tensor.distribution().owner(*key) == context.rank())
            .count() as u64;
        context.all_reduce(&Arithmetic::<u64>::new(), &local)
    };
    let selected = cache.prepare([u.distribution(), v.distribution()], ["ijk", "ijk"],
        [Some(count(&u)), Some(count(&v))]).unwrap().unwrap();
    v.sum_sparse_from_selected("ijk", &u, "ijk", selected, -1., 1.);
    v.norm2()
}

fn run(context: &Context<'_>, token: u64, scope: usize) -> f64 {
    let path = std::env::temp_dir().join(format!("ctf-checkpoint-sparse-{token}-{scope}.txt"));
    let q = checkpoint(context, &path);
    context.barrier();
    if context.rank() == 0 {
        std::fs::remove_file(&path).unwrap();
    }
    context.barrier();
    q
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = Context::world(&universe);
    let mut token = [std::process::id() as u64];
    world.broadcast(0, &mut token);
    let q = run(&world, token[0], 0);
    assert!(q < NORM_BOUND, "checkpoint sparse norm {q:e} exceeds {NORM_BOUND:e}");

    let color = world.rank() % 2;
    let parity = world
        .split(Some(color as i32), world.rank() as i32)
        .unwrap();
    let parity_q = run(&parity, token[0], 1 + color);
    assert!(
        parity_q < NORM_BOUND,
        "checkpoint sparse parity norm {parity_q:e} exceeds {NORM_BOUND:e}"
    );
    parity.close();

    if world.rank() == 0 {
        println!(
            "DIGIT / PASS checkpoint_sparse: rank-seeded MT sparse 3D .1 text round trip; n=3; Q={q:e}; ref=source six-decimal sparse file round trip; bound=1e-8*n^3=2.7e-7; world+parity"
        );
    }
    world.close();
    drop(universe);
}
