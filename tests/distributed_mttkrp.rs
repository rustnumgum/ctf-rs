use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    tensor::Tensor,
};

type Dense<'c, 'r> = Tensor<'c, 'r, Arithmetic<f64>>;

#[derive(Clone, Copy)]
enum SourceLayout {
    Cyclic,
    ModeTwoPhysical,
    ModeTwoVirtual,
}

fn tensor_value(key: usize) -> f64 {
    (1 + (key * 17 + 11) % 23) as f64
}

fn vector_value(mode: usize, coordinate: usize) -> f64 {
    (2 + (mode * 7 + coordinate * 3) % 5) as f64
}

fn matrix_value(mode: usize, coordinate: usize, auxiliary: usize) -> f64 {
    (1 + (mode * 11 + coordinate * 5 + auxiliary * 3) % 7) as f64 / 7.
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

fn make_tensor<'c, 'r>(
    context: &'c Context<'r>,
    shape: &[usize],
    layout: SourceLayout,
) -> Dense<'c, 'r> {
    let mut tensor = Tensor::new(
        context,
        source_distribution(context, shape, layout),
        Arithmetic::new(),
    );
    tensor.transform(|key, value| *value = tensor_value(key));
    tensor
}

fn factor_modes(order: usize, output_mode: usize) -> Vec<usize> {
    (0..order).filter(|&mode| mode != output_mode).collect()
}

fn make_vector_factors<'c, 'r>(
    context: &'c Context<'r>,
    shape: &[usize],
    output_mode: usize,
) -> Vec<Dense<'c, 'r>> {
    factor_modes(shape.len(), output_mode)
        .into_iter()
        .map(|mode| {
            let mut factor = Tensor::new(
                context,
                Distribution::cyclic(vec![shape[mode]], context.size()),
                Arithmetic::new(),
            );
            factor.transform(|key, value| *value = vector_value(mode, key));
            factor
        })
        .collect()
}

fn make_matrix_factors<'c, 'r>(
    context: &'c Context<'r>,
    shape: &[usize],
    output_mode: usize,
    auxiliary: usize,
) -> Vec<Dense<'c, 'r>> {
    factor_modes(shape.len(), output_mode)
        .into_iter()
        .map(|mode| {
            let mut factor = Tensor::new(
                context,
                Distribution::cyclic(vec![auxiliary, shape[mode]], context.size()),
                Arithmetic::new(),
            );
            factor.transform(|key, value| {
                let auxiliary_coordinate = key % auxiliary;
                let coordinate = key / auxiliary;
                *value = matrix_value(mode, coordinate, auxiliary_coordinate);
            });
            factor
        })
        .collect()
}

fn vector_expected(
    output_coordinate: usize,
    tensor_shape: &[usize],
    output_mode: usize,
) -> f64 {
    let mut expected = 0.;
    for tensor_key in 0..global_len(tensor_shape) {
        let coordinates = decode_key(tensor_key, tensor_shape);
        if coordinates[output_mode] != output_coordinate {
            continue;
        }
        let mut value = tensor_value(tensor_key);
        for mode in factor_modes(tensor_shape.len(), output_mode) {
            value *= vector_value(mode, coordinates[mode]);
        }
        expected += value;
    }
    expected
}

fn matrix_expected(
    auxiliary: usize,
    output_coordinate: usize,
    tensor_shape: &[usize],
    output_mode: usize,
) -> f64 {
    let mut expected = 0.;
    for tensor_key in 0..global_len(tensor_shape) {
        let coordinates = decode_key(tensor_key, tensor_shape);
        if coordinates[output_mode] != output_coordinate {
            continue;
        }
        let mut value = tensor_value(tensor_key);
        for mode in factor_modes(tensor_shape.len(), output_mode) {
            value *= matrix_value(mode, coordinates[mode], auxiliary);
        }
        expected += value;
    }
    expected
}

fn assert_output_count(tensor: &Dense<'_, '_>, expected: usize) {
    let mut count = [tensor.local_pairs().len() as f64];
    tensor.context().sum_f64(&mut count);
    assert_eq!(count[0], expected as f64, "global MTTKRP output count");
}

fn assert_vectors(tensor: &Dense<'_, '_>, shape: &[usize], output_mode: usize) {
    for (key, actual) in tensor.local_pairs() {
        let output_coordinate = decode_key(key, &tensor.distribution().shape)[0];
        let expected = vector_expected(output_coordinate, shape, output_mode);
        assert_eq!(actual, expected, "MTTKRP vector mismatch at key={key}");
    }
    assert_output_count(tensor, shape[output_mode]);
}

fn assert_matrices(
    tensor: &Dense<'_, '_>,
    shape: &[usize],
    output_mode: usize,
    auxiliary: usize,
) {
    let mut error = [0.];
    for (key, actual) in tensor.local_pairs() {
        let coordinates = decode_key(key, &tensor.distribution().shape);
        let expected = matrix_expected(coordinates[0], coordinates[1], shape, output_mode);
        assert!(actual.is_finite(), "non-finite MTTKRP value at key={key}");
        error[0] += (actual - expected).abs();
    }
    tensor.context().sum_f64(&mut error);
    assert!(error[0] <= 1e-5, "global MTTKRP L1 error={}", error[0]);
    assert_output_count(tensor, auxiliary * shape[output_mode]);
}

fn exercise_vectors(context: &Context<'_>, shape: &[usize], layout: SourceLayout) {
    for output_mode in 0..shape.len() {
        let tensor = make_tensor(context, shape, layout);
        let factors = make_vector_factors(context, shape, output_mode);
        let references: Vec<_> = factors.iter().collect();
        let output_distribution =
            Distribution::cyclic(vec![shape[output_mode]], context.size());
        let actual = tensor.mttkrp(output_mode, &references, output_distribution.clone());
        assert_eq!(actual.distribution(), &output_distribution);
        assert_vectors(&actual, shape, output_mode);
    }
}

fn exercise_matrices(context: &Context<'_>, shape: &[usize], layout: SourceLayout) {
    let auxiliary = 3;
    for output_mode in 0..shape.len() {
        let tensor = make_tensor(context, shape, layout);
        let factors = make_matrix_factors(context, shape, output_mode, auxiliary);
        let references: Vec<_> = factors.iter().collect();
        let output_distribution =
            Distribution::cyclic(vec![auxiliary, shape[output_mode]], context.size());
        let actual = tensor.mttkrp(output_mode, &references, output_distribution.clone());
        assert_eq!(actual.distribution(), &output_distribution);
        assert_matrices(&actual, shape, output_mode, auxiliary);
    }
}

fn exercise(context: &Context<'_>) {
    for shape in [&[3, 2, 5][..], &[1, 2, 1][..]] {
        for layout in [
            SourceLayout::Cyclic,
            SourceLayout::ModeTwoPhysical,
            SourceLayout::ModeTwoVirtual,
        ] {
            exercise_vectors(context, shape, layout);
            exercise_matrices(context, shape, layout);
        }
    }
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
            "DIGIT / PASS distributed_mttkrp: vector and aux-first matrix MTTKRP, all output modes, cyclic/mode-2/virtual-2 layouts, empty shards, parity subcommunicators, ranks={}",
            world.size()
        );
    }
    world.close();
    runtime.finalize();
}
