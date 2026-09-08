use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    planning::GridPlan,
    selector::{Candidate, Filter, Selector, replication_factor},
    tensor::Tensor,
};
fn exercise(context: &Context<'_>) {
    let np = context.size();
    let make = |shape| {
        Tensor::new(
            context,
            Distribution::cyclic(shape, np),
            Arithmetic::<i64>::new(),
        )
    };
    let mut a = make(vec![3, 5]);
    let mut b = make(vec![5, 2]);
    let mut c = make(vec![3, 2]);
    a.transform(|_, v| *v = 2);
    b.transform(|_, v| *v = 3);
    let plan = GridPlan::prepare(
        [a.distribution(), b.distribution(), c.distribution()],
        ["ik", "kj", "ij"],
        Topology::new(vec![np]),
    )
    .unwrap();
    let signature = plan.signature().clone();
    let candidate = Candidate {
        plan,
        topology_id: 13,
        exhaustive: true,
        seconds: 0.125,
        memory_bytes: 4096,
    };
    let mut selector = Selector::new(context);
    selector.set_scan(true);
    // Only the last rank initially owns the requested plan.
    if context.rank() == np - 1 {
        selector.store(&signature, candidate.clone());
    }
    assert!(selector.select(13, true));
    assert!(!selector.scan());
    let selected = selector.selected().unwrap();
    assert_eq!(selected.seconds, 0.125);
    assert_eq!(selected.memory_bytes, 4096);
    assert_eq!(selected.plan.signature(), &signature);
    assert_eq!(
        selected.plan.mapped_distributions(),
        candidate.plan.mapped_distributions()
    );
    c.contract_with_plan("ij", &a, "ik", &b, "kj", &selected.plan, 1, 0);
    for (_, v) in c.local_pairs() {
        assert_eq!(v, 30);
    }
    assert!(!selector.select(13, false));
    assert!(selector.selected().is_none());
    // Lowest rank wins when all ranks advertise the same candidate ID.
    selector.clear();
    let mut local = candidate.clone();
    local.memory_bytes = context.rank() as u64;
    selector.store(&signature, local);
    assert!(selector.select(13, true));
    assert_eq!(selector.selected().unwrap().memory_bytes, 0);
    selector.filter(&[Filter::upstream("max_time")]);
    assert!(selector.candidates().is_empty());
    assert!(!selector.select(13, true));
    selector.store(&signature, candidate.clone());
    selector.filter(&[Filter::MaxTime(0.125), Filter::MaxMemory(4096)]);
    assert_eq!(selector.candidates().len(), 1);
    let mut other = signature.distributions().clone();
    other[0].shape[0] = 4;
    other[2].shape[0] = 4;
    let other = GridPlan::prepare(
        [&other[0], &other[1], &other[2]],
        ["ik", "kj", "ij"],
        Topology::new(vec![np]),
    )
    .unwrap();
    let changed = other.signature().clone();
    let mut next = candidate.clone();
    next.plan = other;
    selector.store(&changed, next);
    assert_eq!(selector.candidates().len(), 1);
    assert!(selector.selected().is_none());
    selector.reset();
    assert!(selector.candidates().is_empty());
    let mut virtual_map = Mapping::Unmapped;
    virtual_map.augment_virtual(3);
    let distribution = Distribution::new(vec![6], Topology::new(vec![np]), vec![virtual_map]);
    assert_eq!(replication_factor(&distribution), np * 3);
    assert!(Filter::upstream("max_memory").accepts(&candidate));
}
fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    exercise(&world);
    let child = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    exercise(&child);
    child.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS selector: remote plan broadcast and execution, lowest-rank winner, missing candidates, filters, reset, subcontexts; ranks={}",
            world.size()
        );
    }
    world.close();
    drop(universe);
}
