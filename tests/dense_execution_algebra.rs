use ctf::{
    algebra::{Arithmetic, Complex, Monoid, Semiring, Wire},
    context::Context,
    mapping::{Distribution, Topology},
    normal_mapping::Problem,
    tensor::Tensor,
};

#[derive(Clone, Debug, PartialEq)]
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
            i64::decode(&bytes[8 * i..8 * i + 8])
        }))
    }
}
#[derive(Clone)]
struct MatrixRing;
impl Monoid for MatrixRing {
    type Element = Matrix;
    fn zero(&self) -> Matrix {
        Matrix([0; 4])
    }
    fn add(&self, a: &Matrix, b: &Matrix) -> Matrix {
        Matrix(std::array::from_fn(|i| a.0[i] + b.0[i]))
    }
}
impl Semiring for MatrixRing {
    fn one(&self) -> Matrix {
        Matrix([1, 0, 0, 1])
    }
    fn multiply(&self, a: &Matrix, b: &Matrix) -> Matrix {
        Matrix(std::array::from_fn(|i| {
            let row = i % 2;
            let col = i / 2;
            a.0[row] * b.0[2 * col] + a.0[row + 2] * b.0[2 * col + 1]
        }))
    }
}
fn exercise<A: Semiring + Clone>(
    c: &Context<'_>,
    low_memory: bool,
    algebra: A,
    value: impl Fn(usize) -> A::Element,
    close: impl Fn(&A::Element, &A::Element) -> bool,
) where
    A::Element: Wire,
{
    let make = |shape| Tensor::new(c, Distribution::cyclic(shape, c.size()), algebra.clone());
    let mut a = make(vec![3, 3]);
    let mut b = make(vec![3, 2]);
    let mut output = make(vec![3, 2]);
    a.transform(|key, v| *v = value(key % 5 + 1));
    b.transform(|key, v| *v = value(key % 7 + 1));
    let topology = Topology::new(if c.size() == 4 {
        vec![2, 2]
    } else {
        vec![c.size()]
    });
    let problem = Problem::new([&[3, 3], &[3, 2], &[3, 2]], ["ik", "kj", "ij"]).unwrap();
    for permutation in 0..6 {
        output.transform(|_, v| *v = value(4));
        let mapped = problem
            .map_to_topology(&topology, permutation, [None; 3])
            .unwrap();
        let old = [
            a.distribution().clone(),
            b.distribution().clone(),
            output.distribution().clone(),
        ];
        let intra = if c.size() == 4 {
            Some(&[1, 2][..])
        } else {
            None
        };
        if low_memory {
            output.contract_low_memory_from_mapped(
                "ij",
                &mut a,
                "ik",
                &mut b,
                "kj",
                mapped,
                intra,
                value(2),
                value(3),
            );
        } else {
            output.contract_from_mapped(
                "ij",
                &a,
                "ik",
                &b,
                "kj",
                mapped,
                intra,
                value(2),
                value(3),
            );
        }
        assert_eq!(
            [a.distribution(), b.distribution(), output.distribution()],
            old.each_ref()
        );
        for (key, actual) in a.local_pairs() {
            assert!(actual == value(key % 5 + 1), "A restored exactly");
        }
        for (key, actual) in b.local_pairs() {
            assert!(actual == value(key % 7 + 1), "B restored exactly");
        }
        for (key, actual) in output.local_pairs() {
            let i = key % 3;
            let j = key / 3;
            let mut expected = algebra.multiply(&value(3), &value(4));
            for k in 0..3 {
                let product =
                    algebra.multiply(&value((i + 3 * k) % 5 + 1), &value((k + 3 * j) % 7 + 1));
                expected = algebra.add(&algebra.multiply(&product, &value(2)), &expected);
            }
            assert!(close(&actual, &expected), "non-scalar source algebra order");
        }
    }
    let mut a = make(vec![]);
    let mut b = make(vec![]);
    let mut output = make(vec![]);
    a.transform(|_, v| *v = value(1));
    b.transform(|_, v| *v = value(2));
    output.transform(|_, v| *v = value(4));
    let mut topologies = vec![Topology::new(vec![c.size()])];
    if c.size() == 1 {
        topologies.push(Topology::new(vec![]));
    }
    for topology in topologies {
        output.transform(|_, v| *v = value(4));
        let mapped = std::array::from_fn(|_| Distribution::new(vec![], topology.clone(), vec![]));
        if low_memory {
            output.contract_low_memory_from_mapped(
                "",
                &mut a,
                "",
                &mut b,
                "",
                mapped,
                None,
                value(2),
                value(3),
            );
        } else {
            output.contract_from_mapped("", &a, "", &b, "", mapped, None, value(2), value(3));
        }
        for (_, actual) in a.local_pairs() {
            assert!(actual == value(1));
        }
        for (_, actual) in b.local_pairs() {
            assert!(actual == value(2));
        }
        let prior = if topology.dimensions.is_empty() {
            algebra.multiply(&value(4), &value(3))
        } else {
            algebra.multiply(&value(3), &value(4))
        };
        let expected = algebra.add(
            &algebra.multiply(&algebra.multiply(&value(1), &value(2)), &value(2)),
            &prior,
        );
        for (_, actual) in output.local_pairs() {
            assert!(close(&actual, &expected), "scalar source beta side");
        }
    }
}
fn run(c: &Context<'_>) {
    let matrix = |n: usize| Matrix([n as i64, 1, 2, n as i64 + 1]);
    // These alpha/beta operands must distinguish left from right multiplication.
    let beta = Matrix([1, 2, 3, 4]);
    let old = Matrix([2, 0, 1, 3]);
    assert_ne!(
        MatrixRing.multiply(&beta, &old),
        MatrixRing.multiply(&old, &beta)
    );
    for low_memory in [false, true] {
        exercise(
            c,
            low_memory,
            MatrixRing,
            |n| if n == 3 { beta.clone() } else { matrix(n) },
            |a, b| a == b,
        );
        exercise(
            c,
            low_memory,
            Arithmetic::<i64>::new(),
            |n| n as i64,
            |a, b| a == b,
        );
        exercise(
            c,
            low_memory,
            Arithmetic::<f32>::new(),
            |n| n as f32,
            |a, b| a.is_finite() && (a - b).abs() < 1e-6,
        );
        exercise(
            c,
            low_memory,
            Arithmetic::<Complex<f64>>::new(),
            |n| Complex::new(n as f64, 1.),
            |a, b| (a.re - b.re).abs() < 1e-6 && (a.im - b.im).abs() < 1e-6,
        );
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
            "DIGIT / PASS dense_execution_algebra: home/lowmem with Wire node backmapping; input restoration/matrix ring/integer exact; f32/complex abs<1e-6; raw permutations/scalar coefficient order; world+parity"
        );
    }
    world.close();
    drop(universe);
}
