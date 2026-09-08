use ctf::{
    algebra::{Group, Monoid, Semiring, Wire},
    context::Context,
    mapping::Distribution,
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{AS, NS},
    tensor::Tensor,
};
#[derive(Clone, Debug, PartialEq)]
struct Matrix([i64; 4]);
impl Wire for Matrix {
    const WIDTH: usize = 32;
    fn encode(&self, b: &mut Vec<u8>) {
        for x in self.0 {
            x.encode(b);
        }
    }
    fn decode(b: &[u8]) -> Self {
        Self(std::array::from_fn(|i| i64::decode(&b[8 * i..8 * i + 8])))
    }
}
#[derive(Clone)]
struct Ring;
impl Monoid for Ring {
    type Element = Matrix;
    fn zero(&self) -> Matrix {
        Matrix([0; 4])
    }
    fn add(&self, a: &Matrix, b: &Matrix) -> Matrix {
        Matrix(std::array::from_fn(|i| a.0[i] + b.0[i]))
    }
}
impl Group for Ring {
    fn negate(&self, a: &Matrix) -> Matrix {
        Matrix(a.0.map(|x| -x))
    }
}
impl Semiring for Ring {
    fn one(&self) -> Matrix {
        Matrix([1, 0, 0, 1])
    }
    fn multiply(&self, a: &Matrix, b: &Matrix) -> Matrix {
        Matrix(std::array::from_fn(|i| {
            let r = i / 2;
            let c = i % 2;
            a.0[2 * r] * b.0[c] + a.0[2 * r + 1] * b.0[2 + c]
        }))
    }
}
fn run(c: &Context<'_>) {
    let old = Matrix([1, 2, 3, 4]);
    let alpha = Matrix([0, 1, 1, 0]);
    let beta = Matrix([2, 0, 0, 3]);
    let updates = if c.rank() == 0 {
        vec![(0, Matrix([5, 6, 7, 8])), (0, Matrix([1, 0, 2, 1]))]
    } else {
        vec![]
    };
    let mut dense = Tensor::new(c, Distribution::cyclic(vec![2], c.size()), Ring);
    dense.transform(|_, x| *x = old.clone());
    dense.write_scaled(&updates, &alpha, &beta);
    assert_eq!(
        dense.read(&[0, 1]),
        vec![Matrix([11, 13, 15, 18]), old.clone()]
    );
    let d = SymmetricDistribution::new(Distribution::cyclic(vec![2, 2], 1), vec![AS, NS]);
    let child = c.split((c.rank() == 0).then_some(0), 0);
    if let Some(child) = &child {
        let mut sym = SymmetricTensor::new(child, d, Ring);
        sym.transform(|_, x| *x = old.clone());
        sym.write_scaled(
            &[(2, Matrix([5, 6, 7, 8])), (1, Matrix([1, 0, 2, 1]))],
            &alpha,
            &beta,
        );
        assert_eq!(sym.read(&[2]), vec![Matrix([7, 11, 13, 18])]);
    }
    if let Some(child) = child {
        child.close();
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
            "DIGIT / PASS indexed_write_order: exact source left coefficients, duplicate beta-once and antisymmetric signs with noncommutative matrix ring"
        );
    }
    world.close();
    drop(universe);
}
