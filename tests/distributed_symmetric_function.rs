//! Core packed custom-function orchestration: non-distributive function,
//! repeated input/output labels, physical reduction and preserved off-diagonals.
use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::NS,
};

fn make<'c, 'r>(
    context: &'c Context<'r>,
    shape: Vec<usize>,
) -> SymmetricTensor<'c, 'r, Arithmetic<i64>> {
    let topology = Topology::new(vec![context.size()]);
    let mut maps = vec![Mapping::Unmapped; shape.len()];
    maps[0].augment_physical(&topology, 0);
    let links = vec![NS; shape.len()];
    SymmetricTensor::new(
        context,
        SymmetricDistribution::new(Distribution::new(shape, topology, maps), links),
        Arithmetic::new(),
    )
}

fn run(context: &Context<'_>) {
    let mut a = make(context, vec![2, 3, 3]);
    let mut b = make(context, vec![3, 2]);
    a.transform(|key, value| *value = key as i64 + 1);
    b.transform(|key, value| *value = key as i64 - 2);
    for physical in ["i", "j", "k"] {
        let mut output = make(context, vec![2, 2, 2]);
        output.transform(|_, value| *value = 7);
        let original = output.distribution().clone();
        output
            .contract_function_from_on(
                "iji",
                &a,
                "ikk",
                &b,
                "kj",
                Topology::new(vec![context.size()]),
                physical,
                2,
                3,
                true,
                |a, b| a * a * b * b,
            )
            .unwrap();
        assert_eq!(
            output.distribution().distribution(),
            original.distribution()
        );
        assert_eq!(output.distribution().links(), original.links());
        for (key, value) in output.local_pairs() {
            let i = key % 2;
            let j = key / 2 % 2;
            let expected = if key / 4 != i {
                7
            } else {
                21 + 2
                    * (0..3)
                        .map(|k| {
                            let a = (i + 8 * k + 1) as i64;
                            let b = (k + 3 * j) as i64 - 2;
                            a * a * b * b
                        })
                        .sum::<i64>()
            };
            assert_eq!(value, expected, "physical={physical}, key={key}");
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
            "DIGIT / PASS distributed_symmetric_function: exact non-distributive polynomial, repeated labels, physical i/j/k, preserved output layout/off-diagonals; world+parity"
        );
    }
    world.close();
    drop(universe);
}
