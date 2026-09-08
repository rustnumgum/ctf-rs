//! Dense-by-sparse GEMM and folded contraction, including a noncommutative scalar.
use ctf::{
    algebra::{Arithmetic, CustomMonoid, CustomSemiring, Monoid, Wire},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    sparse::SparseTensor,
    tensor::Tensor,
};

fn cyclic(c: &Context<'_>, shape: Vec<usize>) -> Distribution {
    Distribution::cyclic(shape, c.size())
}
fn virtual_first(c: &Context<'_>, shape: Vec<usize>) -> Distribution {
    let topology = Topology::new(vec![c.size()]);
    let mut first = Mapping::Unmapped;
    first.augment_physical(&topology, 0);
    first.augment_virtual(c.size() * 2);
    let mut maps = vec![Mapping::Unmapped; shape.len()];
    maps[0] = first;
    Distribution::new(shape, topology, maps)
}
fn dense<'c, 'r, A: Monoid>(
    c: &'c Context<'r>,
    d: Distribution,
    algebra: A,
    entries: &[(usize, A::Element)],
) -> Tensor<'c, 'r, A>
where
    A::Element: Wire,
{
    let mut t = Tensor::new(c, d, algebra);
    t.transform(|key, value| {
        if let Some((_, entry)) = entries.iter().find(|(k, _)| *k == key) {
            *value = entry.clone();
        }
    });
    t
}
fn sparse<'c, 'r, A: Monoid>(
    c: &'c Context<'r>,
    d: Distribution,
    algebra: A,
    entries: &[(usize, A::Element)],
) -> SparseTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    let mut t = SparseTensor::new(c, d.clone(), algebra);
    let owned: Vec<_> = entries
        .iter()
        .filter(|(key, _)| d.owner(*key) == c.rank())
        .cloned()
        .collect();
    t.write_add(&owned);
    t
}
fn entries<T: Clone>(values: &[T]) -> Vec<(usize, T)> {
    values.iter().cloned().enumerate().collect()
}
fn lookup(values: &[(usize, i64)], key: usize) -> i64 {
    values
        .iter()
        .find_map(|&(k, v)| (k == key).then_some(v))
        .unwrap_or(0)
}
fn product(a: &[i64], b: &[(usize, i64)]) -> Vec<i64> {
    (0..8)
        .map(|key| {
            let i = key % 2;
            let j = key / 2;
            (0..3).map(|k| a[i + 2 * k] * lookup(b, k + 3 * j)).sum()
        })
        .collect()
}
fn scaled(product: &[i64], old: &[i64]) -> Vec<i64> {
    product
        .iter()
        .zip(old)
        .map(|(value, old)| 2 * value + 3 * old)
        .collect()
}
fn grid(c: &Context<'_>) -> [usize; 2] {
    if c.size() == 4 { [2, 2] } else { [c.size(), 1] }
}

fn arithmetic(c: &Context<'_>, grid: [usize; 2]) {
    let a_values = [2_i64, -3, 5, 7, -11, 13];
    let b_values = [
        (0, 2),
        (1, -3),
        (3, 5),
        (6, -11),
        (7, 13),
        (8, 17),
        (10, 19),
        (11, -23),
    ];
    let product = product(&a_values, &b_values);
    let gemm_old = [7_i64, -11, 13, -17, 19, -23, 29, -31];
    let high_old = [37_i64, -41, 43, -47, 53, -59, 61, -67];
    let expected_gemm = scaled(&product, &gemm_old);
    let expected_high: Vec<_> = (0..8)
        .map(|key| {
            let j = key % 4;
            let i = key / 4;
            2 * product[i + 2 * j] + 3 * high_old[key]
        })
        .collect();

    let ad = cyclic(c, vec![2, 3]);
    let bd = cyclic(c, vec![3, 4]);
    let cd = virtual_first(c, vec![2, 4]);
    let a = dense(c, ad.clone(), Arithmetic::<i64>::new(), &entries(&a_values));
    let b = sparse(c, bd.clone(), Arithmetic::<i64>::new(), &b_values);
    let mut output = dense(c, cd.clone(), Arithmetic::<i64>::new(), &entries(&gemm_old));
    output.gemm_dense_sparse(&a, &b, grid, 2, 3);
    assert_eq!(output.read(&(0..8).collect::<Vec<_>>()), expected_gemm);
    assert_eq!(output.distribution(), &cd);

    let high_d = virtual_first(c, vec![4, 2]);
    let mut high = dense(
        c,
        high_d.clone(),
        Arithmetic::<i64>::new(),
        &entries(&high_old),
    );
    high.contract_from_dense_sparse("ji", &a, "ik", &b, "kj", grid, 2, 3)
        .unwrap();
    assert_eq!(high.read(&(0..8).collect::<Vec<_>>()), expected_high);
    assert_eq!(high.distribution(), &high_d);
    assert_eq!(a.distribution(), &ad);
    assert_eq!(b.distribution(), &bd);
}

#[derive(Clone, Debug, PartialEq)]
struct Mat([i64; 4]);
impl Mat {
    fn zero() -> Self {
        Self([0; 4])
    }
    fn one() -> Self {
        Self([1, 0, 0, 1])
    }
}
impl Wire for Mat {
    const WIDTH: usize = 32;
    fn encode(&self, output: &mut Vec<u8>) {
        for value in self.0 {
            value.encode(output);
        }
    }
    fn decode(input: &[u8]) -> Self {
        Self(std::array::from_fn(|i| {
            i64::decode(&input[8 * i..8 * (i + 1)])
        }))
    }
}
fn mat_add(a: &Mat, b: &Mat) -> Mat {
    Mat(std::array::from_fn(|i| a.0[i] + b.0[i]))
}
fn mat_mul(a: &Mat, b: &Mat) -> Mat {
    Mat([
        a.0[0] * b.0[0] + a.0[1] * b.0[2],
        a.0[0] * b.0[1] + a.0[1] * b.0[3],
        a.0[2] * b.0[0] + a.0[3] * b.0[2],
        a.0[2] * b.0[1] + a.0[3] * b.0[3],
    ])
}
type MatrixAlgebra =
    CustomSemiring<CustomMonoid<Mat, fn(&Mat, &Mat) -> Mat>, fn(&Mat, &Mat) -> Mat>;
fn matrix_algebra() -> MatrixAlgebra {
    CustomSemiring {
        monoid: CustomMonoid {
            identity: Mat::zero(),
            addition: mat_add,
        },
        identity: Mat::one(),
        multiplication: mat_mul,
    }
}
fn matrix_order(c: &Context<'_>, grid: [usize; 2]) {
    let cd = virtual_first(c, vec![1, 1]);
    let a = dense(
        c,
        cyclic(c, vec![1, 1]),
        matrix_algebra(),
        &[(0, Mat([0, 1, 0, 0]))],
    );
    let b = sparse(
        c,
        cyclic(c, vec![1, 1]),
        matrix_algebra(),
        &[(0, Mat([0, 0, 1, 0]))],
    );
    let mut output = dense(c, cd.clone(), matrix_algebra(), &[(0, Mat::zero())]);
    output.gemm_dense_sparse(&a, &b, grid, Mat::one(), Mat::zero());
    assert_eq!(output.read(&[0]), vec![Mat([1, 0, 0, 0])]);
    assert_eq!(output.distribution(), &cd);
    let mut high = dense(c, cd.clone(), matrix_algebra(), &[(0, Mat::zero())]);
    high.contract_from_dense_sparse("ij", &a, "ik", &b, "kj", grid, Mat::one(), Mat::zero())
        .unwrap();
    assert_eq!(high.read(&[0]), vec![Mat([1, 0, 0, 0])]);
    assert_eq!(high.distribution(), &cd);
}
fn run(c: &Context<'_>) {
    let grid = grid(c);
    arithmetic(c, grid);
    matrix_order(c, grid);
}
fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let rank = world.rank();
    run(&world);
    let parity = world.split(Some((rank % 2) as i32), rank as i32).unwrap();
    run(&parity);
    parity.close();
    if rank == 0 {
        println!(
            "DIGIT / PASS distributed_dense_sparse: dense*sparse GEMM and contraction, i64 oracle, noncommutative matrix order, world+parity"
        );
    }
    world.close();
    drop(universe);
}
