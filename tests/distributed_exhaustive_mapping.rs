use ctf::{context::Context, mapping_variants::visit_local_exhaustive, topology_candidates};
fn run(c: &Context<'_>) {
    let catalog = topology_candidates::all_shapes(c.size());
    let mut ids = Vec::new();
    let raw = visit_local_exhaustive(
        c,
        [&[3, 4], &[4, 5], &[3, 5]],
        ["ik", "kj", "ij"],
        &catalog,
        |candidate| {
            assert_eq!(candidate.global_id % c.size(), c.rank());
            ids.push(candidate.global_id);
        },
    )
    .unwrap();
    let expected_raw = match c.size() {
        1 => 1,
        2 => 3,
        4 => 9,
        _ => unreachable!(),
    };
    assert_eq!(raw, expected_raw);
    let expected: Vec<_> = (0..raw).filter(|id| id % c.size() == c.rank()).collect();
    assert_eq!(ids, expected);
    // Singleton physical assignments are rejected without renumbering raw IDs.
    let mut survivors = Vec::new();
    let raw = visit_local_exhaustive(
        c,
        [&[3, 2], &[3], &[3]],
        ["ix", "i", "i"],
        &catalog,
        |candidate| survivors.push(candidate.global_id),
    )
    .unwrap();
    let (expected_raw, accepted) = match c.size() {
        1 => (1, vec![0]),
        2 => (2, vec![0]),
        4 => (3, vec![0, 1]),
        _ => unreachable!(),
    };
    assert_eq!(raw, expected_raw);
    assert_eq!(
        survivors,
        accepted
            .into_iter()
            .filter(|id| id % c.size() == c.rank())
            .collect::<Vec<_>>()
    );
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
            "DIGIT / PASS distributed_exhaustive_mapping: catalog order, raw global IDs, rank partitioning and rejection holes; world+parity; exact"
        );
    }
    world.close();
    drop(universe);
}
