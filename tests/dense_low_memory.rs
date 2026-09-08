use ctf::{
    algebra::Arithmetic,
    context::Context,
    cost::Models,
    dense_search::{Options, SearchCache, TopologyFacts},
    linalg::Native,
    mapping::{Distribution, Topology},
    normal_mapping::Problem,
    partial_fold::{self, Descriptor, Outcome},
    symmetry::Symmetry::NS,
    tensor::Tensor,
    topology_candidates,
};

fn av(key: usize) -> f64 {
    (key % 7 + 1) as f64
}
fn bv(key: usize) -> f64 {
    (key % 5) as f64 - 2.
}
fn exercise(
    context: &Context<'_>,
    mapped: [Distribution; 3],
    fold: &Descriptor,
    intra: Option<&[usize]>,
    factor: f64,
) {
    let mut a = Tensor::new(
        context,
        Distribution::cyclic(vec![2, 3, 5], context.size()),
        Arithmetic::<f64>::new(),
    );
    let mut b = Tensor::new(
        context,
        Distribution::cyclic(vec![5, 2], context.size()),
        Arithmetic::<f64>::new(),
    );
    let mut c = Tensor::new(
        context,
        Distribution::cyclic(vec![3, 2], context.size()),
        Arithmetic::<f64>::new(),
    );
    a.transform(|key, v| *v = factor * av(key));
    b.transform(|key, v| *v = bv(key));
    c.transform(|_, v| *v = 3.);
    let old = [
        a.distribution().clone(),
        b.distribution().clone(),
        c.distribution().clone(),
    ];
    c.contract_folded_low_memory::<Native>(
        "ij", &mut a, "xik", &mut b, "kj", mapped, fold, intra, 2., 3.,
    );
    assert_eq!(
        [a.distribution(), b.distribution(), c.distribution()],
        old.each_ref()
    );
    for (key, value) in a.local_pairs() {
        assert_eq!(value, factor * av(key));
    }
    for (key, value) in b.local_pairs() {
        assert_eq!(value, bv(key));
    }
    for (key, value) in c.local_pairs() {
        let i = key % 3;
        let j = key / 3;
        let mut sum = 0.;
        for k in 0..5 {
            for x in 0..2 {
                sum += av(x + 2 * i + 6 * k) * bv(k + 5 * j);
            }
        }
        let expected = 2. * factor * sum + 9.;
        assert!(
            value.is_finite() && (value - expected).abs() < 1e-6,
            "rank {}, key {key}: {value} != {expected}",
            context.rank()
        );
    }
}

fn run(context: &Context<'_>) {
    let shapes: [&[usize]; 3] = [&[2, 3, 5], &[5, 2], &[3, 2]];
    let indices = ["xik", "kj", "ij"];
    let topology = Topology::new(if context.size() == 4 {
        vec![2, 2]
    } else {
        vec![context.size()]
    });
    let models = Models::upstream(1);
    let problem = Problem::new(shapes, indices).unwrap();
    for permutation in 0..6 {
        let mapped = problem
            .map_to_topology(&topology, permutation, [None; 3])
            .unwrap();
        let blocks = mapped.each_ref().map(Distribution::block_shape);
        let links = blocks.each_ref().map(|s| vec![NS; s.len()]);
        let copies = mapped.each_ref().map(|d| {
            d.mappings
                .iter()
                .map(|m| m.phase() / m.physical_phase())
                .product()
        });
        let Outcome::Selected(fold) = partial_fold::select(
            blocks.each_ref().map(Vec::as_slice),
            links.each_ref().map(Vec::as_slice),
            indices,
            &models,
            copies,
        )
        .unwrap() else {
            panic!("fold")
        };
        exercise(
            context,
            mapped,
            &fold,
            if context.size() == 4 {
                Some(&[1, 2])
            } else {
                None
            },
            1.,
        );
    }
    let old = shapes.map(|shape| Distribution::cyclic(shape.to_vec(), context.size()));
    let catalog: Vec<_> = topology_candidates::all_shapes(context.size())
        .into_iter()
        .map(|topology| TopologyFacts {
            nodes_per_axis: vec![1.; topology.dimensions.len()],
            topology,
        })
        .collect();
    let mut cache = SearchCache::new(
        context,
        &catalog,
        &models,
        8,
        false,
        true,
        Options {
            memory_limit: 1_000_000,
            weight: 2.,
            allow_exhaustive: true,
            enable_folding: true,
        },
    );
    for factor in [1., 2.] {
        let selected = cache
            .prepare(old.each_ref(), [&[1.]; 3], indices)
            .unwrap()
            .unwrap();
        exercise(
            context,
            selected.distributions.clone(),
            selected.fold.as_ref().unwrap(),
            None,
            factor,
        );
    }
    assert_eq!(
        cache.stats(),
        ctf::planning::CacheStats { hits: 1, misses: 1 }
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
            "DIGIT / PASS dense_low_memory: mutable input restoration exact, weighted cache reuse, partial folded raw panels and node backmapping; world+parity; abs<1e-6"
        );
    }
    world.close();
    drop(universe);
}
