use ctf::{
    algebra::{Arithmetic, Complex, Monoid, Semiring, Wire},
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    tensor::Tensor,
};

#[derive(Clone, Copy, PartialEq)]
enum ChildMode {
    Cyclic,
    Virtual,
    Replicated,
}

fn replicated_distribution(shape: &[usize], processes: usize) -> Distribution {
    let topology = Topology::new(vec![processes]);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    mappings[0] = Mapping::Virtual {
        copies: processes,
        child: Box::new(Mapping::Unmapped),
    };
    Distribution::new(shape.to_vec(), topology, mappings)
}

fn child_distribution(shape: &[usize], processes: usize, mode: ChildMode) -> Distribution {
    match mode {
        ChildMode::Cyclic => Distribution::cyclic(shape.to_vec(), processes),
        ChildMode::Replicated => replicated_distribution(shape, processes),
        ChildMode::Virtual => {
            let topology = Topology::new(vec![processes]);
            let mut first = Mapping::Unmapped;
            first.augment_physical(&topology, 0);
            first.augment_virtual(processes * 2);
            let mut mappings = vec![Mapping::Unmapped; shape.len()];
            mappings[0] = first;
            Distribution::new(shape.to_vec(), topology, mappings)
        }
    }
}

fn affine<A: Semiring>(
    algebra: &A,
    input: &A::Element,
    old: &A::Element,
    alpha: &A::Element,
    beta: &A::Element,
) -> A::Element {
    algebra.add(
        &algebra.multiply(input, alpha),
        &algebra.multiply(old, beta),
    )
}

fn exercise_group<A, F, G, C>(
    context: &Context<'_>,
    group: usize,
    parent_replicated: bool,
    child_mode: ChildMode,
    algebra: A,
    source_value: F,
    old_value: G,
    alpha: A::Element,
    beta: A::Element,
    close: C,
) where
    A: Semiring + Clone,
    A::Element: Wire + std::fmt::Debug,
    F: Fn(usize) -> A::Element,
    G: Fn(usize) -> A::Element,
    C: Fn(&A::Element, &A::Element) -> bool,
{
    let world_size = context.size();
    let rank = context.rank();
    let active = rank % 2 == group;
    let child_size = (0..world_size).filter(|candidate| candidate % 2 == group).count();
    let child = context.split(
        active.then_some(0),
        (world_size - 1 - rank) as i32,
    );
    if active {
        let child = child.as_ref().unwrap();
        assert_eq!(child.size(), child_size);
        let last_active = group + 2 * (child_size - 1);
        assert_eq!(child.rank(), (last_active - rank) / 2);
    } else {
        assert!(child.is_none());
    }

    for shape in [vec![3, 2], vec![1]] {
        let parent_distribution = if parent_replicated {
            replicated_distribution(&shape, world_size)
        } else {
            Distribution::cyclic(shape.clone(), world_size)
        };
        let target_distribution = child_distribution(&shape, child_size, child_mode);
        let mut source = Tensor::new(
            context,
            parent_distribution.clone(),
            algebra.clone(),
        );
        source.transform(|key, value| *value = source_value(key));
        let source_before = source.local_pairs();
        let source_distribution_before = source.distribution().clone();

        let mut destination = child.as_ref().map(|child| {
            let mut tensor = Tensor::new(child, target_distribution.clone(), algebra.clone());
            tensor.transform(|key, value| *value = old_value(key));
            tensor
        });
        source.add_to_subworld(
            destination.as_mut(),
            &target_distribution,
            alpha.clone(),
            beta.clone(),
        );
        assert_eq!(source.local_pairs(), source_before);
        assert_eq!(source.distribution(), &source_distribution_before);
        if let Some(destination) = destination.as_ref() {
            assert_eq!(destination.distribution(), &target_distribution);
            for (key, actual) in destination.local_pairs() {
                let expected = affine(
                    &algebra,
                    &source_value(key),
                    &old_value(key),
                    &alpha,
                    &beta,
                );
                assert!(close(&actual, &expected), "add_to_subworld key {key}");
            }
            if shape.as_slice() == [1] && child_mode == ChildMode::Cyclic && child_size > 1 {
                let expected = if destination.context().rank() == target_distribution.owner(0) {
                    1
                } else {
                    0
                };
                assert_eq!(destination.local_pairs().len(), expected);
            }
        }

        let child_source = child.as_ref().map(|child| {
            let mut tensor = Tensor::new(child, target_distribution.clone(), algebra.clone());
            tensor.transform(|key, value| *value = source_value(key));
            tensor
        });
        let child_source_before = child_source.as_ref().map(|tensor| tensor.local_pairs());
        let mut parent_destination = Tensor::new(
            context,
            parent_distribution.clone(),
            algebra.clone(),
        );
        parent_destination.transform(|key, value| *value = old_value(key));
        let parent_distribution_before = parent_destination.distribution().clone();
        parent_destination.add_from_subworld(
            child_source.as_ref(),
            &target_distribution,
            alpha.clone(),
            beta.clone(),
        );
        assert_eq!(parent_destination.distribution(), &parent_distribution_before);
        if let (Some(child_source), Some(before)) = (child_source.as_ref(), child_source_before) {
            assert_eq!(child_source.local_pairs(), before);
            assert_eq!(child_source.distribution(), &target_distribution);
        }
        for (key, actual) in parent_destination.local_pairs() {
            let expected = affine(
                &algebra,
                &source_value(key),
                &old_value(key),
                &alpha,
                &beta,
            );
            assert!(close(&actual, &expected), "add_from_subworld key {key}");
        }
    }
    if let Some(child) = child { child.close(); }
}

#[derive(Clone, Debug, PartialEq)]
struct Matrix([i64; 4]);

impl Wire for Matrix {
    const WIDTH: usize = 32;
    fn encode(&self, output: &mut Vec<u8>) {
        for value in self.0 {
            value.encode(output);
        }
    }
    fn decode(input: &[u8]) -> Self {
        Self(std::array::from_fn(|i| i64::decode(&input[8 * i..8 * (i + 1)])))
    }
}

#[derive(Clone)]
struct MatrixAlgebra;

impl Monoid for MatrixAlgebra {
    type Element = Matrix;
    fn zero(&self) -> Matrix { Matrix([0; 4]) }
    fn add(&self, a: &Matrix, b: &Matrix) -> Matrix {
        Matrix(std::array::from_fn(|i| a.0[i] + b.0[i]))
    }
}

impl Semiring for MatrixAlgebra {
    fn one(&self) -> Matrix { Matrix([1, 0, 0, 1]) }
    fn multiply(&self, a: &Matrix, b: &Matrix) -> Matrix {
        Matrix([
            a.0[0] * b.0[0] + a.0[1] * b.0[2],
            a.0[0] * b.0[1] + a.0[1] * b.0[3],
            a.0[2] * b.0[0] + a.0[3] * b.0[2],
            a.0[2] * b.0[1] + a.0[3] * b.0[3],
        ])
    }
}

fn run(context: &Context<'_>) {
    for parent_replicated in [false, true] {
        for child_mode in [ChildMode::Cyclic, ChildMode::Virtual, ChildMode::Replicated] {
            for group in 0..2 {
                if group < context.size() {
                    exercise_group(
                        context,
                        group,
                        parent_replicated,
                        child_mode,
                        Arithmetic::<i64>::new(),
                        |key| key as i64 + 1,
                        |key| key as i64 * 3 + 7,
                        2,
                        3,
                        |actual, expected| actual == expected,
                    );
                    exercise_group(
                        context,
                        group,
                        parent_replicated,
                        child_mode,
                        Arithmetic::<Complex<f64>>::new(),
                        |key| Complex::new(key as f64 + 1.25, -(key as f64) - 0.5),
                        |key| Complex::new(key as f64 * 0.75 + 2., key as f64 + 1.5),
                        Complex::new(2., -1.),
                        Complex::new(-1., 0.5),
                        |actual, expected| {
                            actual.re.is_finite()
                                && actual.im.is_finite()
                                && (actual.re - expected.re).abs() < 1e-6
                                && (actual.im - expected.im).abs() < 1e-6
                        },
                    );
                }
            }
        }
    }

    let alpha = Matrix([1, 2, 3, 4]);
    let beta = Matrix([2, -1, 1, 3]);
    assert_ne!(MatrixAlgebra.multiply(&alpha, &beta), MatrixAlgebra.multiply(&beta, &alpha));
    for group in 0..2 {
        if group < context.size() {
            exercise_group(
                context,
                group,
                true,
                ChildMode::Virtual,
                MatrixAlgebra,
                |key| Matrix([key as i64 + 1, 2, -3, key as i64 + 4]),
                |key| Matrix([key as i64 + 5, -2, 1, key as i64 + 7]),
                alpha.clone(),
                beta.clone(),
                |actual, expected| actual == expected,
            );
        }
    }
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let rank = world.rank();
    run(&world);
    let parity = world
        .split(Some((rank % 2) as i32), rank as i32)
        .unwrap();
    run(&parity);
    parity.close();
    world.barrier();
    if rank == 0 {
        println!(
            "DIGIT / PASS subworld_transfer: alpha/beta parent-child transfers, cyclic/virtual/replicated layouts, reversed odd/even child ranks, i64/complex/noncommutative matrix"
        );
    }
    world.close();
    runtime.finalize();
}
