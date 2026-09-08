use ctf::{
    algebra::{Arithmetic, Complex, Monoid, Semiring, Wire},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    sparse::SparseTensor,
    tensor::Tensor,
};

#[derive(Clone, PartialEq)]
struct Matrix([i64; 4]);
impl Wire for Matrix {
    const WIDTH: usize = 32;
    fn encode(&self, out: &mut Vec<u8>) {
        for x in self.0 {
            x.encode(out);
        }
    }
    fn decode(bytes: &[u8]) -> Self {
        Self(std::array::from_fn(|i| {
            i64::decode(&bytes[i * 8..i * 8 + 8])
        }))
    }
}
#[derive(Clone)]
struct MatrixSemiring;
impl Monoid for MatrixSemiring {
    type Element = Matrix;
    fn zero(&self) -> Matrix {
        Matrix([0; 4])
    }
    fn add(&self, a: &Matrix, b: &Matrix) -> Matrix {
        Matrix(std::array::from_fn(|i| a.0[i] + b.0[i]))
    }
}
impl Semiring for MatrixSemiring {
    fn one(&self) -> Matrix {
        Matrix([1, 0, 0, 1])
    }
    fn multiply(&self, a: &Matrix, b: &Matrix) -> Matrix {
        Matrix(std::array::from_fn(|i| {
            let (r, c) = (i % 2, i / 2);
            a.0[r] * b.0[2 * c] + a.0[r + 2] * b.0[2 * c + 1]
        }))
    }
}

fn exercise<A: Semiring + Clone>(
    context: &Context<'_>,
    algebra: A,
    value: impl Fn(i32, i32) -> A::Element,
    close: impl Fn(&A::Element, &A::Element) -> bool,
) where
    A::Element: Wire,
{
    let shape = [3, 2, 4];
    let topology = Topology::new(if context.size() == 4 {
        vec![2, 2]
    } else {
        vec![context.size()]
    });
    let mut mode = Mapping::Unmapped;
    mode.augment_physical(&topology, 0);
    mode.augment_virtual(2 * topology.dimensions[0]);
    // A virtual mode and, at four ranks, a replicated tensor layer.
    let distribution = Distribution::new(
        shape.to_vec(),
        topology,
        vec![mode, Mapping::Unmapped, Mapping::Unmapped],
    );
    let entries = [
        (0, value(2, 1)),
        (11, value(-3, 2)),
        (17, value(1, -1)),
        (23, algebra.zero()),
    ];
    let input = |key| {
        entries
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, x)| x.clone())
            .unwrap_or_else(|| algebra.zero())
    };
    let make_source = || {
        let mut dense = Tensor::new(context, distribution.clone(), algebra.clone());
        dense.transform(|key, x| *x = input(key));
        let mut sparse = SparseTensor::new(context, distribution.clone(), algebra.clone());
        let owned: Vec<_> = entries
            .iter()
            .filter(|(key, _)| distribution.owner(*key) == context.rank())
            .cloned()
            .collect();
        sparse.write_add(&owned);
        (dense, sparse)
    };
    let coordinate = |key: usize, mode: usize| [key % 3, key / 3 % 2, key / 6][mode];
    let factor_value = |mode: usize, row: usize, r: usize| {
        value((1 + (mode + row + r) % 4) as i32, mode as i32 - r as i32)
    };
    let make_factor = |mode: usize, width: Option<usize>, first: bool| {
        let fs = match width {
            None => vec![shape[mode]],
            Some(k) => {
                if first {
                    vec![k, shape[mode]]
                } else {
                    vec![shape[mode], k]
                }
            }
        };
        let mut factor = Tensor::new(
            context,
            Distribution::cyclic(fs, context.size()),
            algebra.clone(),
        );
        factor.transform(|key, x| {
            let (row, r) = match width {
                None => (key, 0),
                Some(k) => {
                    if first {
                        (key / k, key % k)
                    } else {
                        (key % shape[mode], key / shape[mode])
                    }
                }
            };
            *x = factor_value(mode, row, r);
        });
        factor
    };
    for width in [None, Some(5)] {
        for first in [true, false] {
            if width.is_none() && !first {
                continue;
            }
            let f0 = make_factor(0, width, first);
            let f2 = make_factor(2, width, first);
            let (d0, d2) = (f0.local_storage().to_vec(), f2.local_storage().to_vec());
            let (mut dense, mut sparse) = make_source();
            let keys: Vec<_> = sparse.local_pairs().iter().map(|p| p.0).collect();
            if width.is_none() {
                dense.tttp_vectors(&[(0, &f0), (2, &f2)]);
                sparse.tttp_vectors(&[(0, &f0), (2, &f2)]);
            } else {
                dense.tttp_matrices(
                    &[(0, &f0), (2, &f2)],
                    first,
                    ctf::multilinear::TttpBlocking::Divisions(2),
                );
                sparse.tttp_matrices(
                    &[(0, &f0), (2, &f2)],
                    first,
                    ctf::multilinear::TttpBlocking::Divisions(2),
                );
            }
            let expected = |key| {
                let mut weight = algebra.zero();
                for r in 0..width.unwrap_or(1) {
                    weight = algebra.add(
                        &weight,
                        &algebra.multiply(
                            &factor_value(0, coordinate(key, 0), r),
                            &factor_value(2, coordinate(key, 2), r),
                        ),
                    );
                }
                algebra.multiply(&input(key), &weight)
            };
            for (key, x) in dense.local_pairs() {
                assert!(close(&x, &expected(key)), "dense TTTP key={key}");
            }
            for (key, x) in sparse.local_pairs() {
                assert!(close(&x, &expected(key)), "sparse TTTP key={key}");
            }
            assert_eq!(
                keys,
                sparse.local_pairs().iter().map(|p| p.0).collect::<Vec<_>>()
            );
            assert_eq!(dense.distribution(), &distribution);
            assert_eq!(sparse.distribution(), &distribution);
            assert!(f0.local_storage() == d0.as_slice() && f2.local_storage() == d2.as_slice());
        }
    }
    let (dense, sparse) = make_source();
    for output_mode in 0..3 {
        for width in [None, Some(5)] {
            let modes: Vec<_> = (0..3).filter(|&m| m != output_mode).collect();
            let factors: Vec<_> = modes.iter().map(|&m| make_factor(m, width, true)).collect();
            let refs: Vec<_> = factors.iter().collect();
            let out_shape = match width {
                None => vec![shape[output_mode]],
                Some(k) => vec![k, shape[output_mode]],
            };
            let out_dist = Distribution::cyclic(out_shape, context.size());
            let outputs = [
                dense.mttkrp(output_mode, &refs, out_dist.clone()),
                sparse.mttkrp(output_mode, &refs, out_dist.clone()),
            ];
            for output in outputs {
                assert_eq!(output.distribution(), &out_dist);
                for (key, x) in output.local_pairs() {
                    let (row, r) = match width {
                        None => (key, 0),
                        Some(k) => (key / k, key % k),
                    };
                    let mut expected = algebra.zero();
                    for (input_key, input_value) in &entries {
                        if coordinate(*input_key, output_mode) == row {
                            let mut product = input_value.clone();
                            for &m in &modes {
                                product = algebra.multiply(
                                    &product,
                                    &factor_value(m, coordinate(*input_key, m), r),
                                );
                            }
                            expected = algebra.add(&expected, &product);
                        }
                    }
                    assert!(close(&x, &expected), "MTTKRP mode={output_mode} key={key}");
                }
            }
        }
    }
}
fn run(c: &Context<'_>) {
    exercise(
        c,
        Arithmetic::<f32>::new(),
        |r, _| r as f32,
        |a, b| a.is_finite() && (a - b).abs() < 1e-6,
    );
    exercise(
        c,
        Arithmetic::<f64>::new(),
        |r, _| r as f64,
        |a, b| a.is_finite() && (a - b).abs() < 1e-6,
    );
    exercise(
        c,
        Arithmetic::<Complex<f32>>::new(),
        |r, i| Complex::new(r as f32, i as f32),
        |a, b| {
            a.re.is_finite()
                && a.im.is_finite()
                && (a.re - b.re).abs() < 1e-6
                && (a.im - b.im).abs() < 1e-6
        },
    );
    exercise(
        c,
        Arithmetic::<Complex<f64>>::new(),
        |r, i| Complex::new(r as f64, i as f64),
        |a, b| {
            a.re.is_finite()
                && a.im.is_finite()
                && (a.re - b.re).abs() < 1e-6
                && (a.im - b.im).abs() < 1e-6
        },
    );
    exercise(c, Arithmetic::<i64>::new(), |r, _| r as i64, |a, b| a == b);
    exercise(
        c,
        MatrixSemiring,
        |r, i| Matrix([r as i64, i as i64, 1, (r + 1) as i64]),
        |a, b| a == b,
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
            "DIGIT / PASS typed_multilinear: dense/sparse TTTP and MTTKRP, four float types, exact integers/noncommutative matrix semiring, virtual/replicated layouts and stored zeros; abs<1e-6; world+parity"
        );
    }
    world.close();
    drop(universe);
}
