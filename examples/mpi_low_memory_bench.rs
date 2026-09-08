//! One representative low-memory contraction; no speedup or precision study.
use ctf::{
    algebra::Arithmetic,
    cost::Models,
    dense_search::{Options, SearchCache, TopologyFacts},
    linalg::Native,
    mapping::Distribution,
    memcontrol::{MemoryFraction, ProcessMemoryBudget, ProcessesPerMachine},
    tensor::Tensor,
    topology_candidates,
};
fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let np = world.size();
    let shared = world.split_shared();
    let processes = ProcessesPerMachine::new(shared.size()).unwrap();
    shared.close();
    let budget = ProcessMemoryBudget::discover(processes, MemoryFraction::new(3, 4).unwrap())
        .unwrap()
        .collective_min(&world);
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
    a.transform(|key, v| *v = (key % 17 + 1) as f64 / 17.);
    b.transform(|key, v| *v = (key % 13 + 1) as f64 / 13.);
    let catalog: Vec<_> = topology_candidates::all_shapes(np)
        .into_iter()
        .map(|topology| TopologyFacts {
            nodes_per_axis: vec![0.; topology.dimensions.len()],
            topology,
        })
        .collect();
    let models = Models::upstream(1);
    let mut cache = SearchCache::new(
        &world,
        &catalog,
        &models,
        8,
        false,
        true,
        Options {
            memory_limit: budget.bytes(),
            weight: 2.,
            allow_exhaustive: true,
            enable_folding: true,
        },
    );
    let plan = cache
        .prepare(
            [a.distribution(), b.distribution(), c.distribution()],
            [&[0.]; 3],
            ["ik", "kj", "ij"],
        )
        .unwrap()
        .unwrap();
    world.barrier();
    let start = std::time::Instant::now();
    c.contract_folded_low_memory::<Native>(
        "ij",
        &mut a,
        "ik",
        &mut b,
        "kj",
        plan.distributions.clone(),
        plan.fold.as_ref().unwrap(),
        None,
        1.,
        0.,
    );
    world.barrier();
    let elapsed = start.elapsed().as_secs_f64();
    if world.rank() == 0 {
        println!(
            "lowmem m={m} k={k} n={n} ranks={np} elapsed_s={elapsed:.6} source_memory_bytes={} process_budget_bytes={}",
            plan.memory_bytes,
            budget.bytes()
        );
    }
    drop(cache);
    drop(c);
    drop(b);
    drop(a);
    world.close();
    drop(universe);
}
