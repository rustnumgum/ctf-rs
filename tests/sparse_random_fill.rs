use ctf::{
    algebra::{Arithmetic, Complex, CustomMonoid},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    random::Generator,
    sparse::SparseTensor,
    tensor::Tensor,
};
macro_rules! scalar_case {
    ($name:ident,$t:ty,$min:expr,$max:expr,$sample:expr,$zero:expr,$one:expr) => {
        fn $name(context: &Context<'_>) {
            let topology = Topology::new(vec![context.size()]);
            let mut mode = Mapping::Unmapped;
            mode.augment_physical(&topology, 0);
            mode.augment_virtual(2 * context.size());
            let distribution =
                Distribution::new(vec![3, 4], topology, vec![mode, Mapping::Unmapped]);
            for sparse_case in [true, false] {
                for fraction in [0.5, 0.] {
                    let seed = 37 + context.rank() as u64;
                    let mut oracle = Generator::new(seed);
                    // Literal pinned exponential-series count for N=12, frac=0.5 is 8.
                    let count = if fraction == 0.5 { 8 } else { 0 };
                    let local = count / context.size()
                        + usize::from(context.rank() < count % context.size());
                    let mut mask = [0u64];
                    for _ in 0..local {
                        mask[0] |= 1 << ((oracle.unit_interval() * 12.) as usize);
                    }
                    context.all_reduce_monoid(
                        &CustomMonoid {
                            identity: 0u64,
                            addition: |a: &u64, b: &u64| *a | *b,
                        },
                        &mut mask,
                        true,
                    );
                    let block_size = distribution.block_shape().iter().product::<usize>();
                    let mut keys: Vec<_> = (0..12)
                        .filter(|&key| {
                            distribution.owns(context.rank(), key)
                                && (!sparse_case || mask[0] & (1 << key) != 0)
                        })
                        .collect();
                    keys.sort_by_key(|&key| {
                        (
                            distribution.local_offset(context.rank(), key) / block_size,
                            key,
                        )
                    });
                    let expected: Vec<_> = keys
                        .into_iter()
                        .map(|key| {
                            let draw = oracle.unit_interval();
                            (
                                key,
                                if mask[0] & (1 << key) != 0 {
                                    ($sample)(draw)
                                } else {
                                    $zero
                                },
                            )
                        })
                        .collect();
                    let mut generator = Generator::new(seed);
                    let actual = if sparse_case {
                        let mut tensor = SparseTensor::new(
                            context,
                            distribution.clone(),
                            Arithmetic::<$t>::new(),
                        );
                        tensor.write_add(&if context.rank() == 0 {
                            vec![(0, $one)]
                        } else {
                            Vec::new()
                        });
                        tensor.fill_random_sparse($min, $max, fraction, &mut generator);
                        assert_eq!(tensor.distribution(), &distribution);
                        tensor.local_pairs()
                    } else {
                        let mut tensor =
                            Tensor::new(context, distribution.clone(), Arithmetic::<$t>::new());
                        tensor.transform(|_, x| *x = $one);
                        tensor.fill_random_sparse($min, $max, fraction, &mut generator);
                        assert_eq!(tensor.distribution(), &distribution);
                        tensor.local_pairs()
                    };
                    assert_eq!(
                        actual, expected,
                        "source sparse random case sparse={sparse_case} fraction={fraction}"
                    );
                    assert_eq!(
                        generator.next_u64(),
                        oracle.next_u64(),
                        "source post-dedup random consumption"
                    );
                }
            }
        }
    };
}
scalar_case!(
    real32,
    f32,
    -1f32,
    1f32,
    |x: f64| (x as f32) * 2. - 1.,
    0f32,
    1f32
);
scalar_case!(real64, f64, -1f64, 1f64, |x: f64| x * 2. - 1., 0f64, 1f64);
scalar_case!(
    complex32,
    Complex<f32>,
    Complex::new(-1f32, 0.),
    Complex::new(1f32, 0.),
    |x: f64| Complex::new((x as f32) * 2. - 1., 0.),
    Complex::new(0f32, 0.),
    Complex::new(1f32, 0.)
);
scalar_case!(
    complex64,
    Complex<f64>,
    Complex::new(-1f64, 0.),
    Complex::new(1f64, 0.),
    |x: f64| Complex::new(x * 2. - 1., 0.),
    Complex::new(0f64, 0.),
    Complex::new(1f64, 0.)
);
scalar_case!(
    int32,
    i32,
    2i32,
    7i32,
    |x: f64| (x as i32) * 5 + 2,
    0i32,
    1i32
);
scalar_case!(
    int64,
    i64,
    2i64,
    7i64,
    |x: f64| (x as i64) * 5 + 2,
    0i64,
    1i64
);
scalar_case!(int32_zero, i32, 0i32, 1i32, |x: f64| x as i32, 0i32, 1i32);
scalar_case!(boolean, bool, false, true, |x: f64| x != 0., false, true);
scalar_case!(
    boolean_reverse,
    bool,
    true,
    false,
    |x: f64| x == 0.,
    false,
    true
);
fn run(c: &Context<'_>) {
    real32(c);
    real64(c);
    complex32(c);
    complex64(c);
    int32(c);
    int64(c);
    int32_zero(c);
    boolean(c);
    boolean_reverse(c);
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
            "DIGIT / PASS sparse_random_fill: exact source candidate count/keys/dedup/sample order, seven scalar families, dense and sparse branches, zero density resets; world+parity"
        );
    }
    world.close();
    drop(universe);
}
