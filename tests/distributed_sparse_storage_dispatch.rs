//! Literal home_contract storage dispatch, including source pointer-predicate
//! zero retention and ordinary dense/sparse operand reversal.
use ctf::{
    algebra::{Arithmetic, Monoid, Semiring, Wire},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    sparse::SparseTensor,
    tensor::Tensor,
};

fn distribution(context: &Context<'_>, shape: Vec<usize>) -> Distribution {
    let topology = Topology::new(vec![context.size()]);
    let mut maps = vec![Mapping::Unmapped; shape.len()];
    maps[0].augment_physical(&topology, 0);
    maps[0].augment_virtual(2 * context.size());
    Distribution::new(shape, topology, maps)
}

fn dense_output_conversion(context: &Context<'_>, grid: [usize; 2]) {
    let mut a = Tensor::new(
        context,
        Distribution::cyclic(vec![2, 3], context.size()),
        Arithmetic::<i64>::new(),
    );
    let mut b = Tensor::new(
        context,
        Distribution::cyclic(vec![3, 2], context.size()),
        Arithmetic::<i64>::new(),
    );
    a.transform(|key, value| *value = key as i64 - 2);
    b.transform(|key, value| *value = if key < 3 { 0 } else { key as i64 - 2 });
    let matrix_distribution = distribution(context, vec![2, 2]);
    let mut matrix = SparseTensor::new(
        context,
        matrix_distribution.clone(),
        Arithmetic::<i64>::new(),
    );
    matrix.gemm_dense(&a, &b, grid, 2, 0);
    assert_eq!(matrix.distribution(), &matrix_distribution);
    let matrix_expected: Vec<_> = (0..4)
        .filter(|key| matrix_distribution.owner(*key) == context.rank())
        .map(|key| {
            let i = key % 2;
            let j = key / 2;
            (
                key,
                2 * (0..3)
                    .map(|k| {
                        ((i + 2 * k) as i64 - 2) * if j == 0 { 0 } else { (k + 3 * j) as i64 - 2 }
                    })
                    .sum::<i64>(),
            )
        })
        .collect();
    let mut matrix_actual = matrix.local_pairs();
    matrix_actual.sort_by_key(|pair| pair.0);
    assert_eq!(matrix_actual, matrix_expected);
    let original = distribution(context, vec![2, 2, 2]);
    let mut c = SparseTensor::new(context, original.clone(), Arithmetic::<i64>::new());
    c.write_add(&if context.rank() == original.owner(4) {
        vec![(4, 7)]
    } else {
        vec![]
    });
    c.contract_from_dense("iji", &a, "ik", &b, "kj", grid, 2, 3)
        .unwrap();
    assert_eq!(c.distribution(), &original);
    let expected: Vec<_> = (0..8)
        .filter(|key| original.owner(*key) == context.rank())
        .map(|key| {
            let i = key % 2;
            let j = key / 2 % 2;
            let value = if key / 4 != i {
                if key == 4 { 7 } else { 0 }
            } else {
                2 * (0..3)
                    .map(|k| {
                        ((i + 2 * k) as i64 - 2) * if j == 0 { 0 } else { (k + 3 * j) as i64 - 2 }
                    })
                    .sum::<i64>()
            };
            (key, value)
        })
        .collect();
    let mut actual = c.local_pairs();
    actual.sort_by_key(|pair| pair.0);
    // Even untouched off-diagonal zero values are stored: source uses v!=caddid
    // as a pointer comparison, not a numerical nonzero predicate.
    assert_eq!(actual, expected);
}

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
struct Matrices;
impl Monoid for Matrices {
    type Element = Matrix;
    fn zero(&self) -> Matrix {
        Matrix([0; 4])
    }
    fn add(&self, a: &Matrix, b: &Matrix) -> Matrix {
        Matrix(std::array::from_fn(|i| a.0[i] + b.0[i]))
    }
}
impl Semiring for Matrices {
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

fn swapped_product(context: &Context<'_>, grid: [usize; 2]) {
    let a_value = Matrix([1, 2, 0, 1]);
    let b_value = Matrix([2, 0, 0, 3]);
    let mut a = Tensor::new(
        context,
        Distribution::cyclic(vec![1, 1], context.size()),
        Matrices,
    );
    a.transform(|_, value| *value = a_value.clone());
    let mut b = SparseTensor::new(
        context,
        Distribution::cyclic(vec![1, 1], context.size()),
        Matrices,
    );
    b.write_add(&if context.rank() == 0 {
        vec![(0, b_value)]
    } else {
        vec![]
    });
    let original = distribution(context, vec![1, 1]);
    let mut c = SparseTensor::new(context, original.clone(), Matrices);
    c.gemm_dense_sparse(&a, &b, grid, Matrices.one(), Matrices.zero());
    assert_eq!(c.distribution(), &original);
    assert_eq!(c.read(&[0]), vec![Matrix([2, 4, 0, 3])]);
    c.contract_from_dense_sparse(
        "ij",
        &a,
        "ik",
        &b,
        "kj",
        grid,
        Matrices.one(),
        Matrices.zero(),
    )
    .unwrap();
    assert_eq!(c.distribution(), &original);
    assert_eq!(c.read(&[0]), vec![Matrix([2, 4, 0, 3])]); // B*A, not A*B=[2,6,0,3].
}

fn run(context: &Context<'_>) {
    for grid in if context.size() == 4 {
        vec![[2, 2], [1, 4]]
    } else {
        vec![[context.size(), 1]]
    } {
        dense_output_conversion(context, grid);
        swapped_product(context, grid);
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
            "DIGIT / PASS distributed_sparse_storage_dispatch: source dense-result zero retention, repeated output preservation, sparse-B operand reversal with noncommutative algebra; exact values/keys; world+parity"
        );
    }
    world.close();
    drop(universe);
}
