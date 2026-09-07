use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    tensor::Tensor,
};

type Dense<'c, 'r> = Tensor<'c, 'r, Arithmetic<f64>>;

fn base_value(key: usize) -> f64 {
    (1 + (key * 17 + 11) % 23) as f64
}

fn vector_value(mode: usize, coordinate: usize) -> f64 {
    (2 + (mode * 7 + coordinate * 3) % 5) as f64
}

fn matrix_value(mode: usize, coordinate: usize, auxiliary: usize) -> f64 {
    (1 + (mode * 11 + coordinate * 5 + auxiliary * 3) % 7) as f64 / 7.
}

fn tensor_distribution(context: &Context<'_>, shape: &[usize], mode_two: bool) -> Distribution {
    if !mode_two {
        return Distribution::cyclic(shape.to_vec(), context.size());
    }
    let topology = Topology::new(vec![context.size(), 1]);
    let mut mode = Mapping::Unmapped;
    mode.augment_physical(&topology, 0);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    mappings[2] = mode;
    Distribution::new(shape.to_vec(), topology, mappings)
}

fn make_tensor<'c, 'r>(
    context: &'c Context<'r>,
    shape: &[usize],
    mode_two: bool,
) -> Dense<'c, 'r> {
    let mut tensor = Tensor::new(
        context,
        tensor_distribution(context, shape, mode_two),
        Arithmetic::new(),
    );
    tensor.transform(|key, value| *value = base_value(key));
    tensor
}

fn make_vector_factors<'c, 'r>(
    context: &'c Context<'r>,
    shape: &[usize],
    modes: &[usize],
) -> Vec<(usize, Dense<'c, 'r>)> {
    modes
        .iter()
        .map(|&mode| {
            let mut factor = Tensor::new(
                context,
                Distribution::cyclic(vec![shape[mode]], context.size()),
                Arithmetic::new(),
            );
            factor.transform(|key, value| *value = vector_value(mode, key));
            (mode, factor)
        })
        .collect()
}

fn make_matrix_factors<'c, 'r>(
    context: &'c Context<'r>,
    shape: &[usize],
    modes: &[usize],
    k: usize,
    aux_mode_first: bool,
) -> Vec<(usize, Dense<'c, 'r>)> {
    modes
        .iter()
        .map(|&mode| {
            let factor_shape = if aux_mode_first {
                vec![k, shape[mode]]
            } else {
                vec![shape[mode], k]
            };
            let mut factor = Tensor::new(
                context,
                Distribution::cyclic(factor_shape, context.size()),
                Arithmetic::new(),
            );
            factor.transform(|key, value| {
                let (coordinate, auxiliary) = if aux_mode_first {
                    (key / k, key % k)
                } else {
                    (key % shape[mode], key / shape[mode])
                };
                *value = matrix_value(mode, coordinate, auxiliary);
            });
            (mode, factor)
        })
        .collect()
}

fn vector_expected(key: usize, shape: &[usize], modes: &[usize]) -> f64 {
    let coordinates = Distribution::cyclic(shape.to_vec(), 1).decode_key(key);
    modes.iter().fold(base_value(key), |value, &mode| {
        value * vector_value(mode, coordinates[mode])
    })
}

fn matrix_expected(
    key: usize,
    shape: &[usize],
    modes: &[usize],
    k: usize,
) -> f64 {
    let coordinates = Distribution::cyclic(shape.to_vec(), 1).decode_key(key);
    let multiplier = (0..k)
        .map(|auxiliary| {
            modes.iter().fold(1., |value, &mode| {
                value * matrix_value(mode, coordinates[mode], auxiliary)
            })
        })
        .sum::<f64>();
    base_value(key) * multiplier
}

fn assert_vectors(tensor: &Dense<'_, '_>, shape: &[usize], modes: &[usize]) {
    for (key, actual) in tensor.local_pairs() {
        let expected = vector_expected(key, shape, modes);
        assert_eq!(
            actual,
            expected,
            "TTTP vector mismatch at key={key}, rank={}",
            tensor.context().rank()
        );
    }
}

fn assert_matrices(tensor: &Dense<'_, '_>, shape: &[usize], modes: &[usize], k: usize) {
    let mut error = [0.];
    for (key, actual) in tensor.local_pairs() {
        assert!(actual.is_finite(), "non-finite TTTP value at key={key}");
        let expected = matrix_expected(key, shape, modes, k);
        let difference = (actual - expected).abs();
        error[0] += difference;
    }
    tensor.context().sum_f64(&mut error);
    assert!(error[0] <= 1e-5, "global TTTP L1 error={}", error[0]);
}

fn exercise_vectors(context: &Context<'_>, mode_two: bool, modes: &[usize]) {
    let shape = [3, 2, 5];
    let mut tensor = make_tensor(context, &shape, mode_two);
    let factors = make_vector_factors(context, &shape, modes);
    let references: Vec<_> = factors
        .iter()
        .map(|&(mode, ref factor)| (mode, factor))
        .collect();
    tensor.tttp_vectors(&references);
    assert_vectors(&tensor, &shape, modes);
}

fn exercise_matrices(
    context: &Context<'_>,
    mode_two: bool,
    modes: &[usize],
    aux_mode_first: bool,
    divisions: usize,
) {
    let shape = [3, 2, 5];
    let k = 5;
    let mut tensor = make_tensor(context, &shape, mode_two);
    let factors = make_matrix_factors(context, &shape, modes, k, aux_mode_first);
    let references: Vec<_> = factors
        .iter()
        .map(|&(mode, ref factor)| (mode, factor))
        .collect();
    tensor.tttp_matrices(&references, aux_mode_first, ctf::multilinear::TttpBlocking::Divisions(divisions));
    assert_matrices(&tensor, &shape, modes, k);
}

fn exercise_empty_shards(context: &Context<'_>) {
    let shape = [1, 2, 1];
    let modes = [0, 2];
    let mut tensor = make_tensor(context, &shape, false);
    let factors = make_vector_factors(context, &shape, &modes);
    let references: Vec<_> = factors
        .iter()
        .map(|&(mode, ref factor)| (mode, factor))
        .collect();
    tensor.tttp_vectors(&references);
    assert_vectors(&tensor, &shape, &modes);

    for aux_mode_first in [false, true] {
        let mut tensor = make_tensor(context, &shape, false);
        let factors = make_matrix_factors(context, &shape, &modes, 5, aux_mode_first);
        let references: Vec<_> = factors
            .iter()
            .map(|&(mode, ref factor)| (mode, factor))
            .collect();
        tensor.tttp_matrices(&references, aux_mode_first, ctf::multilinear::TttpBlocking::Divisions(3));
        assert_matrices(&tensor, &shape, &modes, 5);
    }
}

fn exercise(context: &Context<'_>) {
    for mode_two in [false, true] {
        for modes in [&[0, 2][..], &[0, 1, 2][..]] {
            exercise_vectors(context, mode_two, modes);
            for aux_mode_first in [false, true] {
                for divisions in [1, 3] {
                    exercise_matrices(context, mode_two, modes, aux_mode_first, divisions);
                }
            }
        }
    }
    exercise_empty_shards(context);
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
            "DIGIT / PASS distributed_tttp: vector and matrix TTTP, mode remapping, balanced divisions, empty shards, parity subcommunicators, ranks={}",
            world.size()
        );
    }
    world.close();
    runtime.finalize();
}
