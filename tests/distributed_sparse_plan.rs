use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Topology},
    planning::PlanCache,
    sparse::SparseTensor,
    tensor::Tensor,
};
fn run(context: &Context<'_>) {
    let da = Distribution::cyclic(vec![5, 3], context.size());
    let db = Distribution::cyclic(vec![3, 4], context.size());
    let dc = Distribution::cyclic(vec![5, 4], context.size());
    let mut b = Tensor::new(context, db, Arithmetic::<i64>::new());
    b.transform(|key, v| *v = key as i64 - 2);
    let mut cache = PlanCache::new(context);
    let topologies = [
        Topology::new(vec![context.size()]),
        Topology::new(if context.size() == 4 {
            vec![2, 2]
        } else {
            vec![1, context.size()]
        }),
    ];
    for topology in topologies {
        let mut a = SparseTensor::new(context, da.clone(), Arithmetic::<i64>::new());
        let entries: Vec<_> = (0..15)
            .filter(|&key| key % 3 != 0 && da.owner(key) == context.rank())
            .map(|key| (key, key as i64 % 7 - 3))
            .collect();
        a.write_add(&entries);
        for empty in [false, true] {
            if empty {
                a.sparsify(|_| false);
            }
            let mut c = Tensor::new(context, dc.clone(), Arithmetic::<i64>::new());
            c.transform(|_, v| *v = if empty { 5 } else { 3 });
            let plan = cache
                .prepare(
                    [a.distribution(), b.distribution(), c.distribution()],
                    ["ik", "kj", "ij"],
                    topology.clone(),
                )
                .unwrap();
            c.contract_sparse_with_plan("ij", &a, "ik", &b, "kj", plan, 2, 3, true);
            assert_eq!(c.distribution(), &dc);
            let keys: Vec<_> = (0..20).collect();
            let expected: Vec<_> = keys
                .iter()
                .map(|&key| {
                    if empty {
                        15
                    } else {
                        let i = key % 5;
                        let j = key / 5;
                        9 + (0..3)
                            .filter(|&k| (i + 5 * k) % 3 != 0)
                            .map(|k| 2 * ((i + 5 * k) as i64 % 7 - 3) * ((k + 3 * j) as i64 - 2))
                            .sum::<i64>()
                    }
                })
                .collect();
            assert_eq!(c.read(&keys), expected);
        }
    }
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.stats().misses, 2);
    assert_eq!(cache.stats().hits, 2);
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
            "DIGIT / PASS distributed_sparse_plan: existing GridPlan/PlanCache reuse, changing nnz/values, topology switch, restored distribution; exact i64; world+parity"
        );
    }
    world.close();
    drop(universe);
}
