//! Integer fill_random_base: multiply in double, truncate, add integer minimum.
use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    random::Generator,
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{AS, NS, SH, SY},
    tensor::Tensor,
};

macro_rules! cases {
    ($name:ident, $scalar:ty) => {
        fn $name(context: &Context<'_>) {
            for (minimum, maximum) in [(-7 as $scalar, 9 as $scalar), (9, -7), (4, 4)] {
                for shape in [vec![3, 3], vec![1], vec![0], vec![]] {
                    let distribution = Distribution::cyclic(shape, context.size());
                    let mut tensor =
                        Tensor::new(context, distribution.clone(), Arithmetic::<$scalar>::new());
                    let mut random = Generator::new(13 * context.rank() as u64);
                    let mut reference = Generator::new(13 * context.rank() as u64);
                    tensor.fill_random(minimum, maximum, &mut random);
                    for (offset, &actual) in tensor.local_storage().iter().enumerate() {
                        let expected = (reference.unit_interval() * (maximum - minimum) as f64)
                            as $scalar
                            + minimum;
                        assert_eq!(
                            actual,
                            if distribution.global_key(context.rank(), offset).is_some() {
                                expected
                            } else {
                                0
                            }
                        );
                    }
                    assert_eq!(random.next_u64(), reference.next_u64());
                }
                for kind in [SY, AS, SH] {
                    let topology = Topology::new(vec![context.size()]);
                    let mut mappings = vec![Mapping::Unmapped; 2];
                    mappings[0].augment_physical(&topology, 0);
                    for mapping in &mut mappings {
                        mapping.augment_virtual(2 * context.size());
                    }
                    let distribution = SymmetricDistribution::new(
                        Distribution::new(vec![3, 3], topology, mappings),
                        vec![kind, NS],
                    );
                    let mut valid = vec![false; distribution.local_len()];
                    for (offset, _) in distribution.local_pairs(context.rank()) {
                        valid[offset] = true;
                    }
                    let mut tensor =
                        SymmetricTensor::new(context, distribution, Arithmetic::<$scalar>::new());
                    let mut random = Generator::new(13 * context.rank() as u64);
                    let mut reference = Generator::new(13 * context.rank() as u64);
                    tensor.fill_random(minimum, maximum, &mut random);
                    for (offset, &actual) in tensor.local_storage().iter().enumerate() {
                        let expected = (reference.unit_interval() * (maximum - minimum) as f64)
                            as $scalar
                            + minimum;
                        assert_eq!(actual, if valid[offset] { expected } else { 0 });
                    }
                    assert_eq!(random.next_u64(), reference.next_u64());
                }
            }
        }
    };
}
cases!(i32_cases, i32);
cases!(i64_cases, i64);
fn run(context: &Context<'_>) {
    i32_cases(context);
    i64_cases(context);
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
            "DIGIT / PASS integer_random: exact i32/i64 cast order, draw state, dense/SY/AS/SH, reversed/constant intervals, empty/scalar; world+parity"
        );
    }
    world.close();
    drop(universe);
}
