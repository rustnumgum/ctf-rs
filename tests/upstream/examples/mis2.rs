//! Bounded Rust port of pinned `examples/mis2.cxx`.
//!
//! The directed lower-triangle view finds roots with ordinary addition.  The
//! two-hop root-label propagation retains the source's max semiring and
//! sparse stored-entry filtering, while every matrix-vector contraction goes
//! through the automatic raw sparse planner.

mod mis_common;

use ctf::{
    algebra::{Arithmetic, Monoid, Semiring, Wire},
    context::Context,
    cost::Models,
    mapping::{Distribution, Topology},
    sparse::SparseTensor,
    sparse_search::{Options, Pattern, SearchCache},
    sparse_sum_search::{
        Options as SumOptions, Pattern as SumPattern, SearchCache as SumSearchCache,
    },
    topology_candidates,
};

use mis_common::{
    dense_add_sparse, directed_graph, nonzeros, random_graph, sparse_add, sparse_matvec,
    storage_sizes, undirected_graph, SparseVector, N, SPARSE_FRACTION,
};

#[derive(Clone, Copy)]
struct MaxSemiring;

impl Monoid for MaxSemiring {
    type Element = f32;

    fn zero(&self) -> Self::Element {
        0.0
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        (*left).max(*right)
    }
}

impl Semiring for MaxSemiring {
    fn one(&self) -> Self::Element {
        1.0
    }

    fn multiply(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        *left * *right
    }
}

fn options() -> Options {
    Options {
        memory_limit: u64::MAX,
        weight: 0.0,
        allow_exhaustive: true,
    }
}

fn max_matvec<B>(
    output: &mut SparseTensor<'_, '_, MaxSemiring>,
    matrix: &SparseTensor<'_, '_, Arithmetic<f32>>,
    input: &SparseTensor<'_, '_, B>,
    matrix_indices: &str,
    cache: &mut SearchCache<'_, '_>,
) where
    B: Monoid<Element = f32>,
    B::Element: Wire,
{
    let selected = cache
        .prepare(
            [matrix.distribution(), input.distribution(), output.distribution()],
            [matrix_indices, "j", "i"],
            [Some(nonzeros(matrix)), Some(nonzeros(input)), Some(nonzeros(output))],
            None,
        )
        .unwrap()
        .expect("automatic sparse max-semiring matrix-vector search found no valid mapping");
    output.contract_sparse_function_from_selected(
        "i",
        matrix,
        matrix_indices,
        input,
        "j",
        selected,
        |left, right| *left * *right,
        |value, destination| *destination = (*destination).max(value),
    );
}

fn ones<'c, 'r>(context: &'c Context<'r>) -> ctf::tensor::Tensor<'c, 'r, Arithmetic<f32>> {
    let mut result = ctf::tensor::Tensor::new(
        context,
        Distribution::cyclic(vec![N], context.size()),
        Arithmetic::<f32>::new(),
    );
    result.transform(|_, value| *value = 1.0);
    result
}

fn mis2<'c, 'r>(
    context: &'c Context<'r>,
    directed: &SparseVector<'c, 'r>,
    undirected: &SparseVector<'c, 'r>,
    ordinary_cache: &mut SearchCache<'_, '_>,
    max_cache: &mut SearchCache<'_, '_>,
    sparse_sum_cache: &mut SumSearchCache<'_, '_>,
    dense_sum_cache: &mut SumSearchCache<'_, '_>,
) -> SparseVector<'c, 'r> {
    let mut removed = ctf::tensor::Tensor::new(
        context,
        Distribution::cyclic(vec![N], context.size()),
        Arithmetic::<f32>::new(),
    );
    let mut selected = SparseTensor::new(
        context,
        removed.distribution().clone(),
        Arithmetic::<f32>::new(),
    );
    let mut recursive_steps = 0;

    loop {
        let mut candidates = ones(context);
        candidates
            .sum_from(
                "i",
                &removed,
                "i",
                Topology::new(vec![context.size()]),
                -1.0,
                1.0,
            )
            .unwrap();
        let mut candidates = candidates.into_sparse(|value| *value == 1.0);
        recursive_steps += 1;
        if nonzeros(&candidates) == 0 {
            break;
        }

        let old_candidates = candidates.clone();
        sparse_matvec(
            &mut candidates,
            directed,
            &old_candidates,
            1.0,
            1.0,
            ordinary_cache,
        );
        let removed_sparse = removed.clone().into_sparse(|value| *value != 0.0);
        sparse_add(
            &mut candidates,
            &removed_sparse,
            (N + 1) as f32,
            1.0,
            sparse_sum_cache,
        );
        candidates.sparsify(|value| *value == 1.0);

        let candidate_distribution = candidates.distribution().clone();
        candidates.transform_stored(|key, value| {
            let coordinate = candidate_distribution.decode_key(key)[0];
            *value = coordinate as f32 + 1.0;
        });
        let mut roots = SparseTensor::new(
            context,
            candidates.distribution().clone(),
            MaxSemiring,
        );
        let mut max_two_hop = SparseTensor::new(
            context,
            candidates.distribution().clone(),
            MaxSemiring,
        );
        max_matvec(&mut roots, undirected, &candidates, "ij", max_cache);
        max_matvec(
            &mut max_two_hop,
            undirected,
            &roots,
            "ji",
            max_cache,
        );

        candidates.transform_stored(|_, value| *value += 1.0);
        let max_two_hop = max_two_hop.map_stored(Arithmetic::<f32>::new(), |value| *value);
        sparse_add(
            &mut candidates,
            &max_two_hop,
            -1.0,
            1.0,
            sparse_sum_cache,
        );
        candidates.sparsify(|value| *value > 0.9);

        sparse_add(
            &mut selected,
            &candidates,
            1.0,
            1.0,
            sparse_sum_cache,
        );

        let old_candidates = candidates.clone();
        sparse_matvec(
            &mut candidates,
            undirected,
            &old_candidates,
            1.0,
            1.0,
            ordinary_cache,
        );
        let old_candidates = candidates.clone();
        sparse_matvec(
            &mut candidates,
            undirected,
            &old_candidates,
            1.0,
            1.0,
            ordinary_cache,
        );
        dense_add_sparse(
            &mut removed,
            &candidates,
            1.0,
            1.0,
            dense_sum_cache,
        );
    }

    if context.rank() == 0 {
        println!("took {recursive_steps} recursive steps");
    }
    selected.transform_stored(|_, value| *value = 1.0);
    selected
}

fn stored_count(
    tensor: &SparseVector<'_, '_>,
    predicate: impl Fn(f32) -> bool,
) -> i64 {
    let rank = tensor.context().rank();
    tensor
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| tensor.distribution().owner(*key) == rank)
        .filter(|(_, value)| predicate(*value))
        .count() as i64
}

fn run(context: &Context<'_>) {
    let graph = random_graph(context);
    let directed = directed_graph(&graph);
    let undirected = undirected_graph(&graph);
    let catalog = topology_candidates::all_shapes(context.size());
    let models = Models::upstream(1);
    let sizes = storage_sizes();
    let mut ordinary_cache = SearchCache::new(
        context,
        &catalog,
        &models,
        sizes,
        false,
        Pattern::SparseSparseSparse,
        options(),
    );
    let mut max_cache = SearchCache::new(
        context,
        &catalog,
        &models,
        sizes,
        true,
        Pattern::SparseSparseSparse,
        options(),
    );
    let mut sparse_sum_cache = SumSearchCache::new(
        context,
        &catalog,
        &models,
        [sizes[0], sizes[1]],
        true,
        SumPattern::SparseSparse,
        SumOptions {
            memory_limit: u64::MAX,
        },
    );
    let mut dense_sum_cache = SumSearchCache::new(
        context,
        &catalog,
        &models,
        [sizes[0], sizes[1]],
        true,
        SumPattern::SparseDense,
        SumOptions {
            memory_limit: u64::MAX,
        },
    );

    let selected = mis2(
        context,
        &directed,
        &undirected,
        &mut ordinary_cache,
        &mut max_cache,
        &mut sparse_sum_cache,
        &mut dense_sum_cache,
    );
    let mut checked = selected.clone();
    let old_selected = selected.clone();
    sparse_matvec(
        &mut checked,
        &undirected,
        &old_selected,
        1.0,
        1.0,
        &mut ordinary_cache,
    );

    let too_many_local = stored_count(&checked, |value| value > 1.1);
    let too_few_local = stored_count(&checked, |value| value < 0.9);
    let arithmetic = Arithmetic::<i64>::new();
    let too_many = context.all_reduce(&arithmetic, &too_many_local);
    let too_few = context.all_reduce(&arithmetic, &too_few_local);
    let selected_nnz = nonzeros(&selected);

    if context.rank() == 0 {
        println!(
            "mis2: n={N}, sp_frac={SPARSE_FRACTION}, selected_nnz={selected_nnz}, >1.1={too_many}, <.9={too_few}"
        );
    }
    assert_eq!(too_many, 0, "2-MIS is not 2-independent");
    assert_eq!(too_few, 0, "2-MIS is not maximal on stored entries");
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = Context::world(&universe);
    run(&world);

    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS mis2: source sparse 2-independent-set expressions and stored-entry checks; n=16 sp=.1; world+parity"
        );
    }
    world.close();
    drop(universe);
}
