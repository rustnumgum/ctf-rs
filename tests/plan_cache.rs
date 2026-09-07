use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Topology},
    planning::{CacheStats, GridPlan, PlanCache, Signature},
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
    a.transform(|key, v| *v = (key % 3 + 1) as i64);
    b.transform(|key, v| *v = (key / 5 + 2) as i64);
    c.transform(|_, v| *v = 7);
    let original = c.distribution().clone();
    let topology = Topology::new(vec![np]);
    let mut cache = PlanCache::new(context);
    c.contract_cached("ij", &a, "ik", &b, "kj", topology.clone(), &mut cache, 2, 3)
        .unwrap();
    for (key, v) in c.local_pairs() {
        assert_eq!(v, 21 + 10 * (key % 3 + 1) as i64 * (key / 3 + 2) as i64);
    }
    assert_eq!(cache.stats(), CacheStats { hits: 0, misses: 1 });
    a.transform(|key, v| *v = 2 * (key % 3 + 1) as i64);
    c.contract_cached("xy", &a, "xz", &b, "zy", topology.clone(), &mut cache, 3, 0)
        .unwrap();
    for (key, v) in c.local_pairs() {
        assert_eq!(v, 30 * (key % 3 + 1) as i64 * (key / 3 + 2) as i64);
    }
    assert_eq!(cache.stats(), CacheStats { hits: 1, misses: 1 });
    assert_eq!(cache.len(), 1);
    assert_eq!(c.distribution(), &original);

    // A requested topology switch must not retrieve the old grid's plan.
    let other = Topology::new(if np == 4 { vec![2, 2] } else { vec![1, np] });
    c.contract_cached("ij", &a, "ik", &b, "kj", other, &mut cache, 1, 0)
        .unwrap();
    assert_eq!(cache.stats(), CacheStats { hits: 1, misses: 2 });
    // A changed input mapping is independently keyed even with unchanged shape.
    a.redistribute(Distribution::new(
        vec![3, 5],
        topology.clone(),
        vec![ctf::mapping::Mapping::Unmapped; 2],
    ));
    c.contract_cached("ij", &a, "ik", &b, "kj", topology.clone(), &mut cache, 1, 0)
        .unwrap();
    assert_eq!(cache.stats(), CacheStats { hits: 1, misses: 3 });
    for (key, v) in c.local_pairs() {
        assert_eq!(v, 10 * (key % 3 + 1) as i64 * (key / 3 + 2) as i64);
    }
    let plan = GridPlan::prepare(
        [a.distribution(), b.distribution(), c.distribution()],
        ["ik", "kj", "ij"],
        topology.clone(),
    )
    .unwrap();
    c.contract_with_plan("ij", &a, "ik", &b, "kj", &plan, 2, 1);
    for (key, v) in c.local_pairs() {
        assert_eq!(v, 30 * (key % 3 + 1) as i64 * (key / 3 + 2) as i64);
    }

    let left = Signature::new(
        [a.distribution(), b.distribution(), c.distribution()],
        ["ik", "kj", "ij"],
        topology.clone(),
    );
    let right = Signature::new(
        [a.distribution(), b.distribution(), c.distribution()],
        ["xz", "zy", "xy"],
        topology.clone(),
    );
    assert_eq!(left, right);
    cache.clear();
    assert!(cache.is_empty());
    let mut scalar_a = make(vec![]);
    let mut scalar_b = make(vec![]);
    let mut scalar_c = make(vec![]);
    scalar_a.transform(|_, v| *v = 3);
    scalar_b.transform(|_, v| *v = 4);
    scalar_c
        .contract_cached(
            "",
            &scalar_a,
            "",
            &scalar_b,
            "",
            topology.clone(),
            &mut cache,
            2,
            0,
        )
        .unwrap();
    assert_eq!(scalar_c.read(&[0]), vec![24]);
    let zero_a = make(vec![1, 0]);
    let zero_b = make(vec![0, 1]);
    let mut zero_c = make(vec![1, 1]);
    zero_c.transform(|_, v| *v = 5);
    zero_c
        .contract_cached(
            "ij", &zero_a, "ik", &zero_b, "kj", topology, &mut cache, 1, 3,
        )
        .unwrap();
    assert_eq!(zero_c.read(&[0]), vec![15]);
}
fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    exercise(&world);
    let child = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    exercise(&child);
    child.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS plan_cache: reuse, value/scalar changes, grid/distribution switches, scalars/empty contraction and subcontexts; ranks={}",
            world.size()
        );
    }
    world.close();
    runtime.finalize();
}
