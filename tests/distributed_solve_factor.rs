use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    tensor::Tensor,
};

type Dense<'c, 'r> = Tensor<'c, 'r, Arithmetic<f64>>;

const AUXILIARY: usize = 2;

#[derive(Clone, Copy)]
enum SourceLayout {
    Cyclic,
    ModeTwoPhysical,
    ModeTwoVirtual,
}

fn global_len(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn decode_key(mut key: usize, shape: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .map(|&extent| {
            let coordinate = key % extent;
            key /= extent;
            coordinate
        })
        .collect()
}

fn weight(key: usize) -> f64 {
    1. + (key % 3) as f64 / 4.
}

fn factor_value(coordinate: usize, auxiliary: usize) -> f64 {
    match auxiliary {
        0 => 1.,
        1 => (coordinate + 1) as f64,
        _ => unreachable!(),
    }
}

fn exact_x(output_mode: usize, output_coordinate: usize, auxiliary: usize) -> f64 {
    0.35 + 0.07 * output_mode as f64 + 0.11 * output_coordinate as f64 + 0.19 * auxiliary as f64
}

fn source_distribution(
    context: &Context<'_>,
    shape: &[usize],
    layout: SourceLayout,
) -> Distribution {
    match layout {
        SourceLayout::Cyclic => Distribution::cyclic(shape.to_vec(), context.size()),
        SourceLayout::ModeTwoPhysical | SourceLayout::ModeTwoVirtual => {
            let topology = Topology::new(vec![context.size(), 1]);
            let mut mode_two = Mapping::Unmapped;
            mode_two.augment_physical(&topology, 0);
            if matches!(layout, SourceLayout::ModeTwoVirtual) {
                mode_two.augment_virtual(context.size() * 2);
            }
            let mut mappings = vec![Mapping::Unmapped; shape.len()];
            mappings[2] = mode_two;
            Distribution::new(shape.to_vec(), topology, mappings)
        }
    }
}

fn make_source<'c, 'r>(
    context: &'c Context<'r>,
    shape: &[usize],
    layout: SourceLayout,
) -> Dense<'c, 'r> {
    let mut source = Dense::new(
        context,
        source_distribution(context, shape, layout),
        Arithmetic::new(),
    );
    source.transform(|key, value| *value = weight(key));
    source
}

fn factor_modes(order: usize, output_mode: usize) -> Vec<usize> {
    (0..order).filter(|&mode| mode != output_mode).collect()
}

fn make_factors<'c, 'r>(
    context: &'c Context<'r>,
    shape: &[usize],
    output_mode: usize,
) -> Vec<Dense<'c, 'r>> {
    factor_modes(shape.len(), output_mode)
        .into_iter()
        .map(|mode| {
            let mut factor = Dense::new(
                context,
                Distribution::cyclic(vec![AUXILIARY, shape[mode]], context.size()),
                Arithmetic::new(),
            );
            factor.transform(|key, value| {
                let auxiliary = key % AUXILIARY;
                let coordinate = key / AUXILIARY;
                *value = factor_value(coordinate, auxiliary);
            });
            factor
        })
        .collect()
}

fn normal_matrix(shape: &[usize], output_mode: usize, output_coordinate: usize) -> [[f64; 2]; 2] {
    let mut normal = [[0.; AUXILIARY]; AUXILIARY];
    for key in 0..global_len(shape) {
        let coordinates = decode_key(key, shape);
        if coordinates[output_mode] != output_coordinate {
            continue;
        }
        let mut vector = [1.; AUXILIARY];
        for mode in factor_modes(shape.len(), output_mode) {
            vector[1] *= factor_value(coordinates[mode], 1);
        }
        let weighted = weight(key);
        for row in 0..AUXILIARY {
            for column in 0..AUXILIARY {
                normal[row][column] += weighted * vector[row] * vector[column];
            }
        }
    }
    normal
}

fn make_rhs<'c, 'r>(
    context: &'c Context<'r>,
    shape: &[usize],
    output_mode: usize,
) -> (Dense<'c, 'r>, Dense<'c, 'r>) {
    let distribution = Distribution::cyclic(vec![AUXILIARY, shape[output_mode]], context.size());
    let mut rhs = Dense::new(context, distribution.clone(), Arithmetic::new());
    let mut expected = Dense::new(context, distribution.clone(), Arithmetic::new());
    rhs.transform(|key, value| {
        let coordinates = distribution.decode_key(key);
        let output_coordinate = coordinates[1];
        let normal = normal_matrix(shape, output_mode, output_coordinate);
        *value = (0..AUXILIARY)
            .map(|column| {
                normal[coordinates[0]][column] * exact_x(output_mode, output_coordinate, column)
            })
            .sum();
    });
    expected.transform(|key, value| {
        let coordinates = distribution.decode_key(key);
        *value = exact_x(output_mode, coordinates[1], coordinates[0]);
    });
    (rhs, expected)
}

fn assert_allclose(reference: &Dense<'_, '_>, actual: &Dense<'_, '_>) {
    assert_eq!(actual.distribution(), reference.distribution());
    for ((key, expected), (other, value)) in reference
        .local_pairs()
        .into_iter()
        .zip(actual.local_pairs())
    {
        assert_eq!(key, other);
        assert!(expected.is_finite(), "non-finite reference at key={key}");
        assert!(
            value.is_finite(),
            "non-finite solve_factor value at key={key}"
        );
        let difference = (value - expected).abs();
        let bound = 1e-8 + 1e-5 * expected.abs();
        assert!(
            difference <= bound,
            "solve_factor mismatch at key={key}: actual={value}, expected={expected}, difference={difference}, bound={bound}"
        );
    }
}

fn exercise_case(context: &Context<'_>, shape: &[usize], layout: SourceLayout, output_mode: usize) {
    let source = make_source(context, shape, layout);
    let factors = make_factors(context, shape, output_mode);
    let factor_references: Vec<_> = factors.iter().collect();
    let (rhs, expected) = make_rhs(context, shape, output_mode);
    let actual = source
        .solve_factor(output_mode, &factor_references, &rhs)
        .unwrap();
    assert_allclose(&expected, &actual);
}

fn exercise(context: &Context<'_>) {
    for shape in [&[3, 4, 5][..], &[1, 3, 2][..]] {
        for layout in [
            SourceLayout::Cyclic,
            SourceLayout::ModeTwoPhysical,
            SourceLayout::ModeTwoVirtual,
        ] {
            for output_mode in 0..shape.len() {
                exercise_case(context, shape, layout, output_mode);
            }
        }
    }
}

fn exercise_singular(context: &Context<'_>) {
    let shape = [3, 4, 5];
    let output_mode = 0;
    let mut source = make_source(context, &shape, SourceLayout::Cyclic);
    source.transform(|_, value| *value = 0.);
    let factors = make_factors(context, &shape, output_mode);
    let factor_references: Vec<_> = factors.iter().collect();
    let rhs = Dense::new(
        context,
        Distribution::cyclic(vec![AUXILIARY, shape[output_mode]], context.size()),
        Arithmetic::new(),
    );
    match source.solve_factor(output_mode, &factor_references, &rhs) {
        Err(info) => assert_eq!(info, 1),
        Ok(_) => panic!("solve_factor accepted a singular zero normal matrix"),
    }
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    exercise(&world);
    exercise_singular(&world);
    let child = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    exercise(&child);
    exercise_singular(&child);
    child.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS distributed_solve_factor: weighted factor normal-equation solves, all output modes, cyclic/mode-2/virtual-2 layouts, empty shards, parity subcommunicators, ranks={}",
            world.size()
        );
    }
    world.close();
    drop(universe);
}
