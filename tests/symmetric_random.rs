//! Exact packed-allocation random-fill coverage for symmetric tensors.

use ctf::{
    algebra::{Arithmetic, Complex},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    random::Generator,
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, *},
};

fn tensor<'c, 'r, A>(
    context: &'c Context<'r>,
    shape: Vec<usize>,
    links: Vec<Symmetry>,
    algebra: A,
) -> SymmetricTensor<'c, 'r, A>
where
    A: ctf::algebra::Group,
{
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    if !mappings.is_empty() {
        mappings[0].augment_physical(&topology, 0);
    }
    for mapping in &mut mappings {
        mapping.augment_virtual(2 * context.size());
    }
    SymmetricTensor::new(
        context,
        SymmetricDistribution::new(Distribution::new(shape, topology, mappings), links),
        algebra,
    )
}

fn valid_offsets<A: ctf::algebra::Group>(tensor: &SymmetricTensor<'_, '_, A>) -> Vec<bool> {
    let mut valid = vec![false; tensor.local_storage().len()];
    for (offset, _) in tensor.distribution().local_pairs(tensor.context().rank()) {
        valid[offset] = true;
    }
    valid
}

macro_rules! check_case {
    ($name:ident, $scalar:ty, $minimum:expr, $maximum:expr, $expected:expr, $zero:expr) => {
        fn $name(context: &Context<'_>, links: Vec<Symmetry>) {
            let mut actual = tensor(context, vec![3, 3], links, Arithmetic::<$scalar>::new());
            let valid = valid_offsets(&actual);
            let mut generator = Generator::new(context.rank() as u64);
            let mut reference = Generator::new(context.rank() as u64);
            actual.fill_random($minimum, $maximum, &mut generator);
            for (offset, &value) in actual.local_storage().iter().enumerate() {
                let draw = reference.unit_interval();
                let expected = if valid[offset] {
                    ($expected)(draw)
                } else {
                    $zero
                };
                assert_eq!(
                    value,
                    expected,
                    "{} packed offset {offset}",
                    stringify!($scalar)
                );
            }
            assert_eq!(generator.next_u64(), reference.next_u64());
        }
    };
}

check_case!(
    check_f32,
    f32,
    -1f32,
    1f32,
    |x: f64| (x as f32) * 2f32 - 1f32,
    0f32
);
check_case!(check_f64, f64, -1f64, 1f64, |x: f64| x * 2. - 1., 0f64);
check_case!(
    check_complex32,
    Complex<f32>,
    Complex::new(-1f32, 0f32),
    Complex::new(1f32, 0f32),
    |x: f64| Complex::new((x as f32) * 2f32 - 1f32, 0f32),
    Complex::new(0f32, 0f32)
);
check_case!(
    check_complex64,
    Complex<f64>,
    Complex::new(-1f64, 0f64),
    Complex::new(1f64, 0f64),
    |x: f64| Complex::new(x * 2. - 1., 0.),
    Complex::new(0f64, 0f64)
);

fn check_scalar_and_zero_extent(context: &Context<'_>) {
    let mut scalar = tensor(context, vec![], vec![], Arithmetic::<f64>::new());
    let mut scalar_generator = Generator::new(context.rank() as u64);
    let mut scalar_reference = Generator::new(context.rank() as u64);
    scalar.fill_random(-2.0, 3.0, &mut scalar_generator);
    let draw = scalar_reference.unit_interval();
    assert_eq!(scalar.local_storage(), &[(draw * 5.0 - 2.0)]);
    assert_eq!(scalar_generator.next_u64(), scalar_reference.next_u64());

    let mut empty = tensor(context, vec![0, 0], vec![SY, NS], Arithmetic::<f64>::new());
    let mut empty_generator = Generator::new(context.rank() as u64);
    let mut empty_reference = Generator::new(context.rank() as u64);
    empty.fill_random(-1.0, 1.0, &mut empty_generator);
    assert!(empty.local_storage().is_empty());
    assert_eq!(empty_generator.next_u64(), empty_reference.next_u64());
}

fn run(context: &Context<'_>) {
    for links in [vec![SY, NS], vec![AS, NS], vec![SH, NS]] {
        check_f32(context, links.clone());
        check_f64(context, links.clone());
        check_complex32(context, links.clone());
        check_complex64(context, links);
    }
    check_scalar_and_zero_extent(context);
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    run(&world);
    let rank = world.rank();
    let parity = world.split(Some((rank % 2) as i32), rank as i32).unwrap();
    run(&parity);
    parity.close();
    if rank == 0 {
        println!(
            "DIGIT / PASS symmetric_random: exact typed packed draws, padding/virtual holes, SY/AS/SH, scalar/zero extent; world+parity"
        );
    }
    world.close();
    drop(universe);
}
