use ctf::{
    context::Context,
    mapping::{Distribution, Mapping},
    mapping_variants, normal_search, topology_candidates,
};

fn encode(distributions: &[Distribution; 3]) -> Vec<u64> {
    fn mapping(m: &Mapping, v: &mut Vec<u64>) {
        match m {
            Mapping::Unmapped => v.push(0),
            Mapping::Virtual { copies, child } => {
                v.extend([1, *copies as u64]);
                mapping(child, v);
            }
            Mapping::Physical {
                axis,
                processes,
                child,
            } => {
                v.extend([2, *axis as u64, *processes as u64]);
                mapping(child, v);
            }
        }
    }
    let mut v = Vec::new();
    for d in distributions {
        v.push(d.topology.dimensions.len() as u64);
        v.extend(d.topology.dimensions.iter().map(|&x| x as u64));
        v.push(d.shape.len() as u64);
        v.extend(d.shape.iter().map(|&x| x as u64));
        for m in &d.mappings {
            mapping(m, &mut v);
        }
    }
    v
}
fn run(c: &Context<'_>) {
    let catalog = topology_candidates::all_shapes(c.size());
    let shapes: [&[usize]; 3] = [&[3, 4], &[4, 5], &[3, 5]];
    let indices = ["ik", "kj", "ij"];
    let old = shapes.map(|shape| Distribution::cyclic(shape.to_vec(), c.size()));
    let mut normal = Vec::new();
    let mut exhaustive = Vec::new();
    normal_search::visit_local(
        c,
        shapes,
        indices,
        old.each_ref().map(Some),
        &catalog,
        |candidate| normal.push(candidate),
    )
    .unwrap();
    mapping_variants::visit_local_exhaustive(c, shapes, indices, &catalog, |candidate| {
        exhaustive.push(candidate)
    })
    .unwrap();
    // Each owning rank announces accepted IDs and its exact raw map. Every rank
    // rebuilds independently, including ranks with no candidate of that template.
    for root in 0..c.size() {
        for kind in 0..2 {
            let local: Vec<(usize, Vec<u64>)> = if kind == 0 {
                normal
                    .iter()
                    .map(|x| (x.source_id, encode(&x.distributions)))
                    .collect()
            } else {
                exhaustive
                    .iter()
                    .map(|x| (x.global_id, encode(&x.variant.distributions)))
                    .collect()
            };
            let mut count = [local.len() as u64];
            c.broadcast(root, &mut count);
            for position in 0..count[0] as usize {
                let mut id = [if c.rank() == root {
                    local[position].0 as u64
                } else {
                    0
                }];
                c.broadcast(root, &mut id);
                let mut expected = if c.rank() == root {
                    local[position].1.clone()
                } else {
                    Vec::new()
                };
                let mut length = [expected.len() as u64];
                c.broadcast(root, &mut length);
                expected.resize(length[0] as usize, 0);
                c.broadcast(root, &mut expected);
                let result = if kind == 0 {
                    normal_search::reconstruct(
                        id[0] as usize,
                        shapes,
                        indices,
                        old.each_ref().map(Some),
                        &catalog,
                    )
                    .unwrap()
                    .distributions
                } else {
                    mapping_variants::reconstruct_exhaustive(
                        id[0] as usize,
                        shapes,
                        indices,
                        &catalog,
                    )
                    .unwrap()
                    .variant
                    .distributions
                };
                assert_eq!(encode(&result), expected);
            }
        }
    }
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
            "DIGIT / PASS selected_mapping: normal and exhaustive IDs rebuilt on all ranks; exact layouts; world+parity"
        );
    }
    world.close();
    drop(universe);
}
