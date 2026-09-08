use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    tensor::Tensor,
};
fn run(context: &Context<'_>) {
    for replication in 0..if context.size() == 4 { 3 } else { 2 } {
        let topology = Topology::new(if replication == 2 {
            vec![2, 2]
        } else {
            vec![context.size()]
        });
        let mut mode = Mapping::Unmapped;
        if replication != 1 {
            mode.augment_physical(&topology, 0);
            mode.augment_virtual(2 * topology.dimensions[0]);
        }
        let primary = match replication {
            1 => context.rank() == 0,
            2 => context.rank() < 2,
            _ => true,
        };
        let distribution = Distribution::new(vec![3, 2], topology, vec![mode, Mapping::Unmapped]);
        let input = |key: usize| (key % 5) as i64 - 2;
        for filter in 0..4 {
            let keep = |x: i64| match filter {
                0 => x != 0,
                1 => x > 1,
                2 => x.abs() > 1,
                _ => true,
            };
            let mut dense = Tensor::new(context, distribution.clone(), Arithmetic::<i64>::new());
            dense.transform(|key, x| *x = input(key));
            let before = dense.local_storage().to_vec();
            let mut seen = Vec::new();
            let sparse = dense.into_sparse(|x| {
                seen.push(*x);
                keep(*x)
            });
            assert_eq!(
                seen,
                if primary { before } else { Vec::new() },
                "source predicate includes padding only on primary layer"
            );
            assert_eq!(sparse.distribution(), &distribution);
            let mut expected: Vec<_> = (0..6)
                .filter(|&key| distribution.owner(key) == context.rank() && keep(input(key)))
                .map(|key| (key, input(key)))
                .collect();
            let block_size = distribution.block_shape().iter().product::<usize>();
            expected.sort_by_key(|&(key, _)| {
                (
                    distribution.local_offset(context.rank(), key) / block_size,
                    key,
                )
            });
            assert_eq!(sparse.local_pairs(), expected);
            let dense = sparse.into_dense();
            assert_eq!(dense.distribution(), &distribution);
            for (key, x) in dense.local_pairs() {
                assert_eq!(x, if keep(input(key)) { input(key) } else { 0 });
            }
            for (offset, &x) in dense.local_storage().iter().enumerate() {
                if distribution.global_key(context.rank(), offset).is_none() {
                    assert_eq!(x, 0);
                }
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
            "DIGIT / PASS storage_conversion: exact nonzero/signed/absolute/custom predicates, padding call order, virtual blocks, primary sparse layers and collective dense replicas; world+parity"
        );
    }
    world.close();
    drop(universe);
}
