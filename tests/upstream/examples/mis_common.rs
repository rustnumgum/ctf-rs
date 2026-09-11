//! Shared source fixtures and sparse graph operations for MIS and 2-MIS.

use std::mem::size_of;

use ctf::{
    algebra::{Arithmetic, Monoid, Semiring, Wire},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    random::Generator,
    sparse::SparseTensor,
    sparse_search::{SearchCache, StorageSize},
    sparse_sum_search::SearchCache as SumSearchCache,
    sparse_symmetric::SparseSymmetricTensor,
    symmetric_distribution::SymmetricDistribution,
    symmetry::Symmetry::{NS, SH},
    tensor::Tensor,
};

pub const N: usize = 16;
pub const SPARSE_FRACTION: f64 = 0.1;

pub type Graph<'c, 'r> = SparseSymmetricTensor<'c, 'r, Arithmetic<f32>>;
pub type SparseVector<'c, 'r> = SparseTensor<'c, 'r, Arithmetic<f32>>;

fn graph_distribution(context: &Context<'_>) -> SymmetricDistribution {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; 2];
    mappings[0].augment_physical(&topology, 0);
    for mapping in &mut mappings {
        mapping.augment_virtual(2 * context.size());
    }
    SymmetricDistribution::new(
        Distribution::new(vec![N, N], topology, mappings),
        vec![SH, NS],
    )
}

pub fn random_graph<'c, 'r>(context: &'c Context<'r>) -> Graph<'c, 'r> {
    let mut graph = Graph::new(
        context,
        graph_distribution(context),
        Arithmetic::<f32>::new(),
    );
    // The source's srand48 call does not reseed CTF's rank-seeded MT stream.
    let mut generator = Generator::new(context.rank() as u64);
    graph.fill_random_sparse(1.0, 1.0, SPARSE_FRACTION, &mut generator);
    graph
}

/// Source `Tensor(undir_A, nosym)`: retain only the canonical SH triangle.
pub fn directed_graph<'c, 'r>(graph: &Graph<'c, 'r>) -> SparseVector<'c, 'r> {
    graph
        .clone()
        .into_canonical_nonsymmetric(Distribution::cyclic(
            vec![N, N],
            graph.context().size(),
        ))
}

/// Logical undirected adjacency in ordinary sparse storage for raw planning.
pub fn undirected_graph<'c, 'r>(graph: &Graph<'c, 'r>) -> SparseVector<'c, 'r> {
    graph.unpack_orbits(Distribution::cyclic(
        vec![N, N],
        graph.context().size(),
    ))
}

pub fn sparse_vector<'c, 'r>(context: &'c Context<'r>) -> SparseVector<'c, 'r> {
    SparseVector::new(
        context,
        Distribution::cyclic(vec![N], context.size()),
        Arithmetic::<f32>::new(),
    )
}

pub fn storage_sizes() -> [StorageSize; 3] {
    [StorageSize {
        element_bytes: size_of::<f32>(),
        pair_bytes: size_of::<u64>() + size_of::<f32>(),
    }; 3]
}

pub fn nonzeros<A: ctf::algebra::Monoid>(tensor: &SparseTensor<'_, '_, A>) -> u64
where
    A::Element: Wire,
{
    let rank = tensor.context().rank();
    let local = tensor
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| tensor.distribution().owner(*key) == rank)
        .count() as u64;
    tensor
        .context()
        .all_reduce(&Arithmetic::<u64>::new(), &local)
}

pub fn rebind_sparse<'c, 'r, A>(
    source: &SparseTensor<'c, 'r, Arithmetic<f32>>,
    algebra: A,
) -> SparseTensor<'c, 'r, A>
where
    A: Monoid<Element = f32> + Clone,
    A::Element: Wire,
{
    let mut result = SparseTensor::new(
        source.context(),
        source.distribution().clone(),
        algebra,
    );
    let rank = source.context().rank();
    let pairs: Vec<_> = source
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| source.distribution().owner(*key) == rank)
        .collect();
    result.write_add(&pairs);
    result
}

pub fn sparse_matvec<A>(
    output: &mut SparseTensor<'_, '_, A>,
    matrix: &SparseTensor<'_, '_, A>,
    input: &SparseTensor<'_, '_, A>,
    alpha: A::Element,
    beta: A::Element,
    cache: &mut SearchCache<'_, '_>,
) where
    A: Semiring + Clone,
    A::Element: Wire,
{
    let selected = cache
        .prepare(
            [
                matrix.distribution(),
                input.distribution(),
                output.distribution(),
            ],
            ["ij", "j", "i"],
            [
                Some(nonzeros(matrix)),
                Some(nonzeros(input)),
                Some(nonzeros(output)),
            ],
            None,
        )
        .unwrap()
        .expect("automatic sparse matrix-vector search found no valid mapping");
    output.contract_sparse_from_selected(
        "i", matrix, "ij", input, "j", selected, alpha, beta, true,
    );
}

pub fn sparse_dense_matvec<A>(
    output: &mut Tensor<'_, '_, A>,
    matrix: &SparseTensor<'_, '_, A>,
    input: &SparseTensor<'_, '_, A>,
    alpha: A::Element,
    beta: A::Element,
    cache: &mut SearchCache<'_, '_>,
) where
    A: Semiring + Clone,
    A::Element: Wire,
{
    let selected = cache
        .prepare(
            [
                matrix.distribution(),
                input.distribution(),
                output.distribution(),
            ],
            ["ij", "j", "i"],
            [Some(nonzeros(matrix)), Some(nonzeros(input)), None],
            None,
        )
        .unwrap()
        .expect("automatic sparse-sparse-dense matrix-vector search found no valid mapping");
    output.contract_sparse_sparse_from_selected(
        "i", matrix, "ij", input, "j", selected, alpha, beta, true,
    );
}

pub fn sparse_add<'c, 'r, A>(
    output: &mut SparseTensor<'c, 'r, A>,
    input: &SparseTensor<'c, 'r, A>,
    alpha: A::Element,
    beta: A::Element,
    cache: &mut SumSearchCache<'_, '_>,
) where
    A: Semiring + Clone,
    A::Element: Wire,
{
    let selected = cache
        .prepare(
            [input.distribution(), output.distribution()],
            ["i", "i"],
            [Some(nonzeros(input)), Some(nonzeros(output))],
        )
        .unwrap()
        .expect("automatic sparse-sparse summation search found no valid mapping");
    output.sum_sparse_from_selected("i", input, "i", selected, alpha, beta);
}

pub fn dense_add_sparse<A>(
    output: &mut Tensor<'_, '_, A>,
    input: &SparseTensor<'_, '_, A>,
    alpha: A::Element,
    beta: A::Element,
    cache: &mut SumSearchCache<'_, '_>,
) where
    A: Semiring + Clone,
    A::Element: Wire,
{
    let selected = cache
        .prepare(
            [input.distribution(), output.distribution()],
            ["i", "i"],
            [Some(nonzeros(input)), None],
        )
        .unwrap()
        .expect("automatic sparse-dense summation search found no valid mapping");
    output.sum_sparse_from_selected("i", input, "i", selected, alpha, beta, true);
}
