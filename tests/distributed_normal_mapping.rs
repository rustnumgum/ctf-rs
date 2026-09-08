use ctf::{
    algebra::Arithmetic, context::Context, mapping::Distribution, normal_search,
    topology_candidates,
};
fn run(c: &Context<'_>) {
    let catalog = topology_candidates::all_shapes(c.size());
    let a = Distribution::cyclic(vec![3, 4], c.size());
    let b = Distribution::cyclic(vec![4, 5], c.size());
    let output = Distribution::cyclic(vec![3, 5], c.size());
    let mut seen = vec![0.; 6 * (catalog.len() + 8)];
    let mut order = Vec::new();
    normal_search::visit_local(
        c,
        [&[3, 4], &[4, 5], &[3, 5]],
        ["ik", "kj", "ij"],
        [Some(&a), Some(&b), Some(&output)],
        &catalog,
        |candidate| {
            assert_eq!((candidate.template - 1) % c.size(), c.rank());
            assert_eq!(
                candidate.source_id,
                6 * candidate.template + candidate.permutation
            );
            assert_eq!(seen[candidate.source_id], 0.);
            seen[candidate.source_id] = 1.;
            order.push((candidate.permutation, candidate.template));
            for distribution in candidate.distributions {
                assert_eq!(distribution.topology.size(), c.size());
            }
        },
    )
    .unwrap();
    assert!(order.windows(2).all(|pair| pair[0] < pair[1]));
    c.sum_f64(&mut seen);
    assert!(seen.iter().all(|&count| count <= 1.));
    assert!(seen.iter().any(|&count| count == 1.));
    let count = c.all_reduce(&Arithmetic::<i64>::new(), &(order.len() as i64));
    assert_eq!(count, seen.iter().sum::<f64>() as i64);
    let mut templates = Vec::new();
    normal_search::visit_local(
        c,
        [&[3, 4], &[4, 5], &[3, 5]],
        ["ik", "kj", "ij"],
        [None, None, None],
        &catalog,
        |candidate| templates.push(candidate.template),
    )
    .unwrap();
    assert!(templates.iter().all(|&template| template >= 8));
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
            "DIGIT / PASS distributed_normal_mapping: source six-permutation search, old-layout subsets, template partitioning and IDs; world+parity; exact"
        );
    }
    world.close();
    drop(universe);
}
