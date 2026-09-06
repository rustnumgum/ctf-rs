use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    multilinear::tensor_svd::TensorSvd,
    tensor::Tensor,
};

type Dense<'c, 'r> = Tensor<'c, 'r, Arithmetic<f64>>;

fn deterministic<'c, 'r>(context: &'c Context<'r>, shape: Vec<usize>) -> Dense<'c, 'r> {
    let mut tensor = Dense::new(
        context,
        Distribution::cyclic(shape, context.size()),
        Arithmetic::new(),
    );
    tensor.transform(|key, value| {
        *value = ((key * 37 + 11) % 101) as f64 / 17. + (key % 7) as f64 / 13.;
    });
    tensor
}

fn rank_one<'c, 'r>(context: &'c Context<'r>) -> Dense<'c, 'r> {
    let mut tensor = Dense::new(
        context,
        Distribution::cyclic(vec![3, 2, 2], context.size()),
        Arithmetic::new(),
    );
    let distribution = tensor.distribution().clone();
    tensor.transform(|key, value| {
        let coordinates = distribution.decode_key(key);
        *value = (coordinates[0] + 1) as f64
            * (coordinates[1] + 2) as f64
            * (coordinates[2] + 3) as f64;
    });
    tensor
}

fn rename_auxiliary(labels: &str, auxiliary: char, replacement: char) -> String {
    labels
        .chars()
        .map(|label| if label == auxiliary { replacement } else { label })
        .collect()
}

fn factor_shape(
    source: &Dense<'_, '_>,
    indices: &str,
    factor_indices: &str,
    auxiliary: char,
    rank: usize,
) -> Vec<usize> {
    factor_indices
        .chars()
        .map(|label| {
            if label == auxiliary {
                rank
            } else {
                source.distribution().shape[indices.find(label).unwrap()]
            }
        })
        .collect()
}

fn scale_left(
    left: &mut Dense<'_, '_>,
    singular: &Dense<'_, '_>,
    indices: &str,
    auxiliary: char,
) {
    let rank = singular.distribution().shape[0];
    let values = singular.read(&(0..rank).collect::<Vec<_>>());
    let auxiliary_axis = indices.find(auxiliary).unwrap();
    let distribution = left.distribution().clone();
    left.transform(|key, value| {
        let auxiliary_coordinate = distribution.decode_key(key)[auxiliary_axis];
        *value *= values[auxiliary_coordinate];
    });
}

fn normalized_residual(reference: &Dense<'_, '_>, actual: &Dense<'_, '_>) {
    let mut error = [0.];
    for ((key, reference_value), (other, actual_value)) in reference
        .local_pairs()
        .into_iter()
        .zip(actual.local_pairs())
    {
        assert_eq!(key, other);
        error[0] += (reference_value - actual_value).powi(2);
    }
    reference.context().sum_f64(&mut error);
    let normalized = error[0].sqrt() / reference.distribution().global_len() as f64;
    assert!(normalized < 1e-6, "normalized reconstruction error={normalized}");
}

fn assert_orthogonal(
    factor: &Dense<'_, '_>,
    factor_indices: &str,
    auxiliary: char,
    topology: Topology,
) {
    let rank = factor.distribution().shape[factor_indices.find(auxiliary).unwrap()];
    let replacement = if auxiliary == 'r' { 's' } else { 'r' };
    let renamed = rename_auxiliary(factor_indices, auxiliary, replacement);
    let output_indices = format!("{auxiliary}{replacement}");
    let mut gram = Dense::new(
        factor.context(),
        Distribution::cyclic(vec![rank, rank], factor.context().size()),
        Arithmetic::new(),
    );
    gram
        .contract_from(
            &output_indices,
            factor,
            factor_indices,
            factor,
            renamed.as_str(),
            topology,
            1.,
            0.,
        )
        .unwrap();

    let mut residual = [0., 0.];
    for (key, value) in gram.local_pairs() {
        let coordinates = gram.distribution().decode_key(key);
        let expected = if coordinates[0] == coordinates[1] { 1. } else { 0. };
        residual[0] += (value - expected).abs();
        residual[1] += expected.abs();
    }
    factor.context().sum_f64(&mut residual);
    assert!(
        residual[0] <= 1e-3 || residual[0] / residual[1] <= 1e-3,
        "orthogonality residual={}, reference L1={}",
        residual[0],
        residual[1]
    );
}

fn assert_factor_shapes(
    source: &Dense<'_, '_>,
    indices: &str,
    left_indices: &str,
    right_indices: &str,
    auxiliary: char,
    left: &Dense<'_, '_>,
    singular: &Dense<'_, '_>,
    right: &Dense<'_, '_>,
    rank: usize,
) {
    assert_eq!(
        left.distribution().shape,
        factor_shape(source, indices, left_indices, auxiliary, rank)
    );
    assert_eq!(singular.distribution().shape, vec![rank]);
    assert_eq!(
        right.distribution().shape,
        factor_shape(source, indices, right_indices, auxiliary, rank)
    );
}

fn reconstruct(
    source: &Dense<'_, '_>,
    indices: &str,
    left_indices: &str,
    right_indices: &str,
    auxiliary: char,
    grid: [usize; 2],
    mut left: Dense<'_, '_>,
    singular: &Dense<'_, '_>,
    right: &Dense<'_, '_>,
) {
    let topology = Topology::new(grid.to_vec());
    assert_orthogonal(&left, left_indices, auxiliary, topology.clone());
    assert_orthogonal(right, right_indices, auxiliary, topology.clone());
    scale_left(&mut left, singular, left_indices, auxiliary);

    let mut reconstructed = Dense::new(
        source.context(),
        source.distribution().clone(),
        Arithmetic::new(),
    );
    reconstructed
        .contract_from(
            indices,
            &left,
            left_indices,
            right,
            right_indices,
            topology,
            1.,
            0.,
        )
        .unwrap();
    normalized_residual(source, &reconstructed);
}

fn exercise_tensor_svd(context: &Context<'_>, grid: [usize; 2]) {
    let source = deterministic(context, vec![3, 2, 2]);
    for (left_indices, right_indices) in [("caq", "qb"), ("qca", "bq")] {
        let (left_rows, right_columns) = (
            left_indices
                .chars()
                .filter(|&label| label != 'q')
                .map(|label| source.distribution().shape["abc".find(label).unwrap()])
                .product::<usize>(),
            right_indices
                .chars()
                .filter(|&label| label != 'q')
                .map(|label| source.distribution().shape["abc".find(label).unwrap()])
                .product::<usize>(),
        );
        let (left, singular, right) = source
            .tensor_svd(
                "abc",
                left_indices,
                'q',
                right_indices,
                grid,
                TensorSvd::Truncated {
                    rank: None,
                    threshold: 0.,
                },
            )
            .unwrap();
        let rank = left_rows.min(right_columns);
        assert_factor_shapes(
            &source,
            "abc",
            left_indices,
            right_indices,
            'q',
            &left,
            &singular,
            &right,
            rank,
        );
        reconstruct(
            &source,
            "abc",
            left_indices,
            right_indices,
            'q',
            grid,
            left,
            &singular,
            &right,
        );
    }

    let low_rank = rank_one(context);
    let (left, singular, right) = low_rank
        .tensor_svd(
            "abc",
            "caq",
            'q',
            "qb",
            grid,
            TensorSvd::Truncated {
                rank: Some(1),
                threshold: 0.,
            },
        )
        .unwrap();
    assert_factor_shapes(
        &low_rank,
        "abc",
        "caq",
        "qb",
        'q',
        &left,
        &singular,
        &right,
        1,
    );
    reconstruct(
        &low_rank,
        "abc",
        "caq",
        "qb",
        'q',
        grid,
        left,
        &singular,
        &right,
    );

    let (left, singular, right) = low_rank
        .tensor_svd(
            "abc",
            "caq",
            'q',
            "qb",
            grid,
            TensorSvd::Randomized {
                rank: 1,
                iterations: 1,
                oversampling: 1,
                seed: 123,
            },
        )
        .unwrap();
    assert_factor_shapes(
        &low_rank,
        "abc",
        "caq",
        "qb",
        'q',
        &left,
        &singular,
        &right,
        1,
    );
    reconstruct(
        &low_rank,
        "abc",
        "caq",
        "qb",
        'q',
        grid,
        left,
        &singular,
        &right,
    );

    let source = deterministic(context, vec![4, 5, 6, 3]);
    for (left_indices, right_indices) in [
        ("ija", "akl"),
        ("ika", "ajl"),
        ("iakj", "la"),
        ("alk", "jai"),
    ] {
        let (left_rows, right_columns) = (
            left_indices
                .chars()
                .filter(|&label| label != 'a')
                .map(|label| source.distribution().shape["ijkl".find(label).unwrap()])
                .product::<usize>(),
            right_indices
                .chars()
                .filter(|&label| label != 'a')
                .map(|label| source.distribution().shape["ijkl".find(label).unwrap()])
                .product::<usize>(),
        );
        let (left, singular, right) = source
            .tensor_svd(
                "ijkl",
                left_indices,
                'a',
                right_indices,
                grid,
                TensorSvd::Truncated {
                    rank: None,
                    threshold: 0.,
                },
            )
            .unwrap();
        let rank = left_rows.min(right_columns);
        assert_factor_shapes(
            &source,
            "ijkl",
            left_indices,
            right_indices,
            'a',
            &left,
            &singular,
            &right,
            rank,
        );
        reconstruct(
            &source,
            "ijkl",
            left_indices,
            right_indices,
            'a',
            grid,
            left,
            &singular,
            &right,
        );
    }
}

fn changed_distribution(shape: Vec<usize>, topology: Topology, physical_axis: usize) -> Distribution {
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    mappings[physical_axis].augment_physical(&topology, 0);
    Distribution::new(shape, topology, mappings)
}

fn exercise_reshape(context: &Context<'_>) {
    let np = context.size();
    let mut source = Dense::new(
        context,
        Distribution::cyclic(vec![3, 2, 2], np),
        Arithmetic::new(),
    );
    source.transform(|key, value| *value = 10. + key as f64);
    let target = changed_distribution(vec![4, 3], Topology::new(vec![np]), 1);
    let reshaped = source.reshape(target.clone());
    assert_eq!(reshaped.distribution(), &target);
    assert_eq!(
        reshaped.read(&(0..12).collect::<Vec<_>>()),
        (0..12).map(|key| 10. + key as f64).collect::<Vec<_>>()
    );

    let mut singleton = Dense::new(
        context,
        Distribution::cyclic(vec![1], np),
        Arithmetic::new(),
    );
    singleton.transform(|_, value| *value = 37.5);
    let singleton_target = changed_distribution(vec![1, 1], Topology::new(vec![np]), 1);
    let reshaped_singleton = singleton.reshape(singleton_target.clone());
    assert_eq!(reshaped_singleton.distribution(), &singleton_target);
    assert_eq!(reshaped_singleton.read(&[0]), vec![37.5]);
}

fn exercise(context: &Context<'_>) {
    let np = context.size();
    let grid = if np == 4 { [2, 2] } else { [np, 1] };
    exercise_tensor_svd(context, grid);
    exercise_reshape(context);
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
            "DIGIT / PASS distributed_tensor_svd: tensor SVD layouts, truncation/randomization, reshape; ranks={}",
            world.size()
        );
    }
    world.close();
    runtime.finalize();
}
