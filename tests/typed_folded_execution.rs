use ctf::{
    algebra::{Arithmetic, Complex, Monoid, Semiring, Wire},
    context::Context,
    cost::Models,
    dense_search::{Options, SearchCache, TopologyFacts},
    linalg::{GemmKernel, Native},
    mapping::{Distribution, Topology},
    normal_mapping::Problem,
    partial_fold::{self, Descriptor, Outcome},
    symmetry::Symmetry::NS,
    tensor::Tensor,
    topology_candidates,
};

fn exercise<T: Clone + PartialEq + Wire>(
    context: &Context<'_>,
    mapped: [Distribution; 3],
    fold: &Descriptor,
    value: &impl Fn(i32, i32) -> T,
    close: &impl Fn(&T, &T) -> bool,
    low_memory: bool,
    factor: i32,
    intra: Option<&[usize]>,
) where
    Arithmetic<T>: Semiring<Element = T> + Clone,
    Native: GemmKernel<T>,
{
    let algebra = Arithmetic::<T>::new();
    let make = |shape| {
        Tensor::new(
            context,
            Distribution::cyclic(shape, context.size()),
            algebra.clone(),
        )
    };
    let mut a = make(vec![2, 3, 3]);
    let mut b = make(vec![3, 2]);
    let mut c = make(vec![3, 2]);
    a.transform(|key, v| *v = value(factor * (key % 5 + 1) as i32, factor));
    b.transform(|key, v| *v = value((key % 3 + 1) as i32, -1));
    c.transform(|_, v| *v = value(3, 2));
    let old = [
        a.distribution().clone(),
        b.distribution().clone(),
        c.distribution().clone(),
    ];
    if low_memory {
        c.contract_folded_low_memory::<Native>(
            "ij",
            &mut a,
            "xik",
            &mut b,
            "kj",
            mapped,
            fold,
            intra,
            value(2, 1),
            value(1, -1),
        );
    } else {
        c.contract_folded_from_mapped::<Native>(
            "ij",
            &a,
            "xik",
            &b,
            "kj",
            mapped,
            fold,
            intra,
            value(2, 1),
            value(1, -1),
        );
    }
    assert_eq!(
        [a.distribution(), b.distribution(), c.distribution()],
        old.each_ref()
    );
    for (key, actual) in a.local_pairs() {
        assert!(actual == value(factor * (key % 5 + 1) as i32, factor));
    }
    for (key, actual) in b.local_pairs() {
        assert!(actual == value((key % 3 + 1) as i32, -1));
    }
    for (key, actual) in c.local_pairs() {
        let i = key % 3;
        let j = key / 3;
        let mut sum = algebra.zero();
        for k in 0..3 {
            for x in 0..2 {
                let aa = value(factor * ((x + 2 * i + 6 * k) % 5 + 1) as i32, factor);
                let bb = value(((k + 3 * j) % 3 + 1) as i32, -1);
                sum = algebra.add(&sum, &algebra.multiply(&aa, &bb));
            }
        }
        let expected = algebra.add(
            &algebra.multiply(&value(2, 1), &sum),
            &algebra.multiply(&value(1, -1), &value(3, 2)),
        );
        assert!(
            close(&actual, &expected),
            "typed folded rank {} key {key}",
            context.rank()
        );
    }
}
fn run_type<T: Clone + PartialEq + Wire>(
    context: &Context<'_>,
    value: impl Fn(i32, i32) -> T,
    close: impl Fn(&T, &T) -> bool,
) where
    Arithmetic<T>: Semiring<Element = T> + Clone,
    Native: GemmKernel<T>,
{
    let shapes: [&[usize]; 3] = [&[2, 3, 3], &[3, 2], &[3, 2]];
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
        for low_memory in [false, true] {
            exercise(
                context,
                mapped.clone(),
                &fold,
                &value,
                &close,
                low_memory,
                1,
                if context.size() == 4 {
                    Some(&[1, 2])
                } else {
                    None
                },
            );
        }
    }
    let old = shapes.map(|shape| Distribution::cyclic(shape.to_vec(), context.size()));
    let catalog: Vec<_> = topology_candidates::all_shapes(context.size())
        .into_iter()
        .map(|topology| TopologyFacts {
            nodes_per_axis: vec![0.; topology.dimensions.len()],
            topology,
        })
        .collect();
    let mut cache = SearchCache::new(
        context,
        &catalog,
        &models,
        T::WIDTH,
        false,
        true,
        Options {
            memory_limit: 1_000_000,
            weight: 2.,
            allow_exhaustive: true,
            enable_folding: true,
        },
    );
    for factor in [1, 2] {
        let plan = cache
            .prepare(old.each_ref(), [&[0.]; 3], indices)
            .unwrap()
            .unwrap();
        exercise(
            context,
            plan.distributions.clone(),
            plan.fold.as_ref().unwrap(),
            &value,
            &close,
            true,
            factor,
            None,
        );
    }
    assert_eq!(
        cache.stats(),
        ctf::planning::CacheStats { hits: 1, misses: 1 }
    );
}
fn run(context: &Context<'_>) {
    run_type(
        context,
        |r, _| r as f32,
        |a, b| a.is_finite() && (a - b).abs() < 1e-6,
    );
    run_type(
        context,
        |r, _| r as f64,
        |a, b| a.is_finite() && (a - b).abs() < 1e-6,
    );
    run_type(
        context,
        |r, i| Complex::new(r as f32, i as f32),
        |a, b| {
            a.re.is_finite()
                && a.im.is_finite()
                && (a.re - b.re).abs() < 1e-6
                && (a.im - b.im).abs() < 1e-6
        },
    );
    run_type(
        context,
        |r, i| Complex::new(r as f64, i as f64),
        |a, b| {
            a.re.is_finite()
                && a.im.is_finite()
                && (a.re - b.re).abs() < 1e-6
                && (a.im - b.im).abs() < 1e-6
        },
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
            "DIGIT / PASS typed_folded_execution: four BLAS scalar types, raw panels, partial folds, node ordering, home/lowmem, weighted cache; exact inputs/layouts and finite abs<1e-6; world+parity"
        );
    }
    world.close();
    drop(universe);
}
