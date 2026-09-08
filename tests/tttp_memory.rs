use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    multilinear::TttpBlocking,
    sparse::SparseTensor,
    tensor::Tensor,
};

fn run(context: &Context<'_>) {
    let topology = Topology::new(vec![context.size()]);
    let mut mode = Mapping::Unmapped;
    mode.augment_physical(&topology, 0);
    mode.augment_virtual(2 * context.size());
    let distribution = Distribution::new(
        vec![3, 2, 4],
        topology,
        vec![mode, Mapping::Unmapped, Mapping::Unmapped],
    );
    let make_factor = |mode: usize, first: bool| {
        let n = distribution.shape[mode];
        let shape = if first { vec![5, n] } else { vec![n, 5] };
        let mut f = Tensor::new(
            context,
            Distribution::cyclic(shape, context.size()),
            Arithmetic::<i64>::new(),
        );
        f.transform(|key, x| {
            let (i, r) = if first {
                (key / 5, key % 5)
            } else {
                (key % n, key / n)
            };
            *x = (i + r + mode + 1) as i64;
        });
        f
    };
    for first in [false, true] {
        let f0 = make_factor(0, first);
        let f2 = make_factor(2, first);
        for sparse_case in [false, true] {
            let mut dense = Tensor::new(context, distribution.clone(), Arithmetic::<i64>::new());
            dense.transform(|key, x| *x = (key % 4 + 1) as i64);
            let mut sparse =
                SparseTensor::new(context, distribution.clone(), Arithmetic::<i64>::new());
            let pairs: Vec<_> = [(0, 2), (11, -3), (17, 1), (23, 0)]
                .into_iter()
                .filter(|(key, _)| distribution.owner(*key) == context.rank())
                .collect();
            sparse.write_add(&pairs);
            let npair = if sparse_case {
                sparse.local_nnz()
            } else {
                dense.local_pairs().len()
            };
            // On rank zero the equality boundary admits div=4 (kd=2), but not
            // div=2 (kd=3). Other ranks admit div=1. MPI_MAX must choose div=4.
            let available = if context.rank() == 0 {
                (2 * 3 * 2 / context.size() + 2 * 4 * 2 + npair) as u64 * 8
            } else {
                10000
            };
            let old_keys: Vec<_> = sparse.local_pairs().iter().map(|p| p.0).collect();
            let actual = if sparse_case {
                sparse.tttp_matrices(
                    &[(0, &f0), (2, &f2)],
                    first,
                    TttpBlocking::AvailableBytes(available),
                );
                assert_eq!(sparse.distribution(), &distribution);
                assert_eq!(
                    old_keys,
                    sparse.local_pairs().iter().map(|p| p.0).collect::<Vec<_>>()
                );
                sparse.local_pairs()
            } else {
                dense.tttp_matrices(
                    &[(0, &f0), (2, &f2)],
                    first,
                    TttpBlocking::AvailableBytes(available),
                );
                assert_eq!(dense.distribution(), &distribution);
                dense.local_pairs()
            };
            for (key, x) in actual {
                let base = if sparse_case {
                    match key {
                        0 => 2,
                        11 => -3,
                        17 => 1,
                        23 => 0,
                        _ => unreachable!(),
                    }
                } else {
                    (key % 4 + 1) as i64
                };
                let expected = base
                    * (0..5)
                        .map(|r| (key % 3 + r + 1) as i64 * (key / 6 + r + 3) as i64)
                        .sum::<i64>();
                assert_eq!(x, expected, "budgeted TTTP key={key}");
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
            "DIGIT / PASS tttp_memory: exact dense/sparse blocked TTTP, uneven rank-local budgets and MPI_MAX, both factor orientations, virtual mappings and stored zeros; world+parity"
        );
    }
    world.close();
    drop(universe);
}
