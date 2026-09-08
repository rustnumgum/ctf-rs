//! Distributed indexed sums where at least one operand is sparse.
//!
//! The first case is the small deterministic fixture from CTF's
//! `test/sptensor_sum.cxx`; the remaining cases use integer values so every
//! affine update can be checked exactly.
use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    sparse::SparseTensor,
    tensor::Tensor,
};

type Algebra = Arithmetic<i64>;

fn cyclic(context: &Context<'_>, shape: Vec<usize>) -> Distribution {
    Distribution::cyclic(shape, context.size())
}

fn virtual2(context: &Context<'_>, shape: Vec<usize>) -> Distribution {
    let topology = Topology::new(vec![context.size()]);
    let mut first = Mapping::Unmapped;
    first.augment_physical(&topology, 0);
    first.augment_virtual(context.size() * 2);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    if !mappings.is_empty() {
        mappings[0] = first;
    }
    Distribution::new(shape, topology, mappings)
}

fn sparse<'c, 'r>(
    context: &'c Context<'r>,
    distribution: Distribution,
    entries: &[(usize, i64)],
) -> SparseTensor<'c, 'r, Algebra> {
    let mut tensor = SparseTensor::new(context, distribution.clone(), Algebra::new());
    let owned: Vec<_> = entries
        .iter()
        .copied()
        .filter(|(key, _)| distribution.owner(*key) == context.rank())
        .collect();
    tensor.write_add(&owned);
    tensor
}

fn sparse_from_rank<'c, 'r>(
    context: &'c Context<'r>,
    distribution: Distribution,
    entries: &[(usize, i64)],
    writer: usize,
) -> SparseTensor<'c, 'r, Algebra> {
    let mut tensor = SparseTensor::new(context, distribution, Algebra::new());
    let owned = if context.rank() == writer {
        entries.to_vec()
    } else {
        Vec::new()
    };
    tensor.write_add(&owned);
    tensor
}

fn dense<'c, 'r>(
    context: &'c Context<'r>,
    distribution: Distribution,
    entries: &[(usize, i64)],
) -> Tensor<'c, 'r, Algebra> {
    let mut tensor = Tensor::new(context, distribution.clone(), Algebra::new());
    let owned: Vec<_> = entries
        .iter()
        .copied()
        .filter(|(key, _)| distribution.owner(*key) == context.rank())
        .collect();
    tensor.write_add(&owned);
    tensor
}

fn matrix_entries() -> [(usize, i64); 5] {
    // Shape [2, 3], column-major keys i + 2*j.  The final matrix entry is
    // deliberately absent so reductions also exercise sparse missing keys.
    [(0, 2), (1, 3), (2, 5), (3, 7), (4, 11)]
}

fn matrix_value(key: usize) -> i64 {
    matrix_entries()
        .iter()
        .find_map(|&(entry, value)| (entry == key).then_some(value))
        .unwrap_or(0)
}

fn full_pairs(len: usize, base: i64) -> Vec<(usize, i64)> {
    (0..len).map(|key| (key, base + key as i64)).collect()
}

fn transpose_expected(base: i64) -> Vec<i64> {
    let mut expected = vec![0; 6];
    for i in 0..2 {
        for j in 0..3 {
            let key = j + 3 * i;
            expected[key] = 3 * (base + key as i64) + 2 * matrix_value(i + 2 * j);
        }
    }
    expected
}

fn fixture(context: &Context<'_>) {
    let distribution = cyclic(context, vec![3, 3, 3, 3]);
    let a_entries = [(1, 3), (2, 42), (4, 1), (8, -1)];
    let b_entries = [(2, 24), (3, 7)];
    let a = sparse_from_rank(
        context,
        distribution.clone(),
        &a_entries,
        context.size() / 2,
    );
    let mut b = sparse_from_rank(
        context,
        distribution.clone(),
        &b_entries,
        context.size() / 2,
    );
    b.sum_from("abij", &a, "abij", 1, 1);
    let keys = [1, 2, 3, 4, 8];
    assert_eq!(b.read(&keys), vec![3, 66, 7, 1, -1]);
    assert_eq!(b.reduce(), 76);

    let mut dense_b = dense(context, distribution, &b_entries);
    dense_b.sum_from_sparse("abij", &a, "abij", 1, 1);
    assert_eq!(dense_b.read(&keys), vec![3, 66, 7, 1, -1]);
    assert_eq!(dense_b.reduce(), 76);
}

fn sparse_to_sparse_core(context: &Context<'_>) {
    let source_distribution = cyclic(context, vec![2, 3]);
    let a_entries = matrix_entries();
    let a = sparse(context, source_distribution, &a_entries);

    // Aij -> Cji, with a different (virtual-2) destination layout.
    let transpose_base = 50;
    let mut transpose = sparse(
        context,
        virtual2(context, vec![3, 2]),
        &full_pairs(6, transpose_base),
    );
    transpose.sum_from("ji", &a, "ij", 2, 3);
    let expected = transpose_expected(transpose_base);
    assert_eq!(transpose.read(&(0..6).collect::<Vec<_>>()), expected);
    assert_eq!(transpose.reduce(), expected.iter().sum());

    // Aij -> Ci reduces the j index.
    let mut reduction = sparse(context, cyclic(context, vec![2]), &[(0, 10), (1, 20)]);
    reduction.sum_from("i", &a, "ij", 2, 3);
    assert_eq!(reduction.read(&[0, 1]), vec![66, 80]);
    assert_eq!(reduction.reduce(), 146);

    // Ai -> Cij broadcasts over j.
    let vector = sparse(context, cyclic(context, vec![2]), &[(0, 2), (1, 3)]);
    let broadcast_base = 40;
    let mut broadcast = sparse(
        context,
        cyclic(context, vec![2, 3]),
        &full_pairs(6, broadcast_base),
    );
    broadcast.sum_from("ij", &vector, "i", 2, 3);
    let expected: Vec<_> = (0..6)
        .map(|key| {
            let i = key % 2;
            3 * (broadcast_base + key as i64) + 2 * [2, 3][i]
        })
        .collect();
    assert_eq!(broadcast.read(&(0..6).collect::<Vec<_>>()), expected);
    assert_eq!(broadcast.reduce(), expected.iter().sum());

    // Aii -> C is a trace; off-diagonal source entries must be ignored.
    let trace = sparse(
        context,
        cyclic(context, vec![3, 3]),
        &[(0, 2), (1, 100), (3, 101), (4, 5), (8, 7)],
    );
    let mut scalar = sparse(context, cyclic(context, vec![]), &[(0, 10)]);
    scalar.sum_from("", &trace, "ii", 2, 3);
    assert_eq!(scalar.read(&[0]), vec![58]);
    assert_eq!(scalar.reduce(), 58);

    // Cii <- Ai updates only the diagonal; off-diagonal C entries survive.
    let diagonal_source = sparse(context, cyclic(context, vec![3]), &[(0, 4), (1, 6), (2, 8)]);
    let diagonal_base = 100;
    let mut diagonal = sparse(
        context,
        cyclic(context, vec![3, 3]),
        &full_pairs(9, diagonal_base),
    );
    diagonal.sum_from("ii", &diagonal_source, "i", 2, 3);
    let expected: Vec<_> = (0..9)
        .map(|key| {
            let i = key % 3;
            let j = key / 3;
            if i == j {
                3 * (diagonal_base + key as i64) + 2 * [4, 6, 8][i]
            } else {
                diagonal_base + key as i64
            }
        })
        .collect();
    assert_eq!(diagonal.read(&(0..9).collect::<Vec<_>>()), expected);
    assert_eq!(diagonal.reduce(), expected.iter().sum());
}

fn sparse_to_dense_core(context: &Context<'_>) {
    let source_distribution = cyclic(context, vec![2, 3]);
    let a_entries = matrix_entries();
    let a = sparse(context, source_distribution, &a_entries);

    // Aij -> Cji, with a different (virtual-2) destination layout.
    let transpose_base = 50;
    let mut transpose = dense(
        context,
        virtual2(context, vec![3, 2]),
        &full_pairs(6, transpose_base),
    );
    transpose.sum_from_sparse("ji", &a, "ij", 2, 3);
    let expected = transpose_expected(transpose_base);
    assert_eq!(transpose.read(&(0..6).collect::<Vec<_>>()), expected);
    assert_eq!(transpose.reduce(), expected.iter().sum());

    // Aij -> Ci reduces the j index.
    let mut reduction = dense(context, cyclic(context, vec![2]), &[(0, 10), (1, 20)]);
    reduction.sum_from_sparse("i", &a, "ij", 2, 3);
    assert_eq!(reduction.read(&[0, 1]), vec![66, 80]);
    assert_eq!(reduction.reduce(), 146);

    // Ai -> Cij broadcasts over j.
    let vector = sparse(context, cyclic(context, vec![2]), &[(0, 2), (1, 3)]);
    let broadcast_base = 40;
    let mut broadcast = dense(
        context,
        cyclic(context, vec![2, 3]),
        &full_pairs(6, broadcast_base),
    );
    broadcast.sum_from_sparse("ij", &vector, "i", 2, 3);
    let expected: Vec<_> = (0..6)
        .map(|key| {
            let i = key % 2;
            3 * (broadcast_base + key as i64) + 2 * [2, 3][i]
        })
        .collect();
    assert_eq!(broadcast.read(&(0..6).collect::<Vec<_>>()), expected);
    assert_eq!(broadcast.reduce(), expected.iter().sum());

    // Aii -> C is a trace; off-diagonal source entries must be ignored.
    let trace = sparse(
        context,
        cyclic(context, vec![3, 3]),
        &[(0, 2), (1, 100), (3, 101), (4, 5), (8, 7)],
    );
    let mut scalar = dense(context, cyclic(context, vec![]), &[(0, 10)]);
    scalar.sum_from_sparse("", &trace, "ii", 2, 3);
    assert_eq!(scalar.read(&[0]), vec![58]);
    assert_eq!(scalar.reduce(), 58);

    // Cii <- Ai updates only the diagonal; off-diagonal C entries survive.
    let diagonal_source = sparse(context, cyclic(context, vec![3]), &[(0, 4), (1, 6), (2, 8)]);
    let diagonal_base = 100;
    let mut diagonal = dense(
        context,
        cyclic(context, vec![3, 3]),
        &full_pairs(9, diagonal_base),
    );
    diagonal.sum_from_sparse("ii", &diagonal_source, "i", 2, 3);
    let expected: Vec<_> = (0..9)
        .map(|key| {
            let i = key % 3;
            let j = key / 3;
            if i == j {
                3 * (diagonal_base + key as i64) + 2 * [4, 6, 8][i]
            } else {
                diagonal_base + key as i64
            }
        })
        .collect();
    assert_eq!(diagonal.read(&(0..9).collect::<Vec<_>>()), expected);
    assert_eq!(diagonal.reduce(), expected.iter().sum());
}

fn dense_to_sparse_permutation(context: &Context<'_>) {
    let source = dense(
        context,
        cyclic(context, vec![2, 3]),
        &[(0, 2), (1, 3), (2, 5), (3, 7), (4, 11), (5, 13)],
    );
    let base = 70;
    let mut output = sparse(context, virtual2(context, vec![3, 2]), &full_pairs(6, base));
    output.sum_from_dense("ji", &source, "ij", 2, 3);
    let mut expected = vec![0; 6];
    for i in 0..2 {
        for j in 0..3 {
            let key = j + 3 * i;
            expected[key] = 3 * (base + key as i64) + 2 * matrix_value(i + 2 * j);
        }
    }
    // The dense source has a value at the previously absent key (i=1,j=2).
    expected[5] = 3 * (base + 5) + 2 * 13;
    assert_eq!(output.read(&(0..6).collect::<Vec<_>>()), expected);
    assert_eq!(output.reduce(), expected.iter().sum());
}

fn run(context: &Context<'_>) {
    fixture(context);
    sparse_to_sparse_core(context);
    sparse_to_dense_core(context);
    dense_to_sparse_permutation(context);
    // The high-level source sparsifies dense input before applying alpha;
    // explicit zeros resulting from alpha=0 are retained by sparse merging.
    let a = dense(context, cyclic(context, vec![2, 3]), &matrix_entries());
    let mut b = sparse(context, cyclic(context, vec![2, 3]), &[(5, 9)]);
    b.sum_from_dense("ij", &a, "ij", 0, 0);
    assert_eq!(b.read(&(0..6).collect::<Vec<_>>()), vec![0; 6]);
    for (key, value) in b.local_pairs() {
        assert!(key < 6);
        assert_eq!(value, 0);
    }
    let mut count = [b.local_nnz() as f64];
    context.sum_f64(&mut count);
    assert_eq!(count[0], 6.);
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let world_rank = world.rank();
    run(&world);

    let parity = world
        .split(Some((world_rank % 2) as i32), world_rank as i32)
        .unwrap();
    run(&parity);
    parity.close();

    if world_rank == 0 {
        println!(
            "DIGIT / PASS distributed_sparse_sum: sparse/dense indexed sums, reductions, broadcasts, diagonals and layouts; world+parity"
        );
    }
    world.close();
    drop(universe);
}
