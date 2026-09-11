//! Pinned `examples/mis.cxx` on the source n=16, sparse-fraction=0.1 fixture.

mod mis_common;

use ctf::{
    algebra::{Arithmetic, Monoid, Semiring},
    context::Context,
    cost::Models,
    mapping::{Distribution, Topology},
    sparse_search::{Options as ContractionOptions, Pattern as ContractionPattern, SearchCache},
    sparse_sum_search::{
        Options as SumOptions, Pattern as SumPattern, SearchCache as SumSearchCache,
    },
    tensor::Tensor,
    topology_candidates,
};
use mis_common::{
    N, dense_add_sparse, directed_graph, nonzeros, random_graph, sparse_add,
    sparse_dense_matvec, sparse_matvec, sparse_vector, storage_sizes, undirected_graph,
};

type Algebra = Arithmetic<f32>;
type DenseVector<'c, 'r> = Tensor<'c, 'r, Algebra>;

fn dense_vector<'c, 'r>(context: &'c Context<'r>) -> DenseVector<'c, 'r> {
    DenseVector::new(
        context,
        Distribution::cyclic(vec![N], context.size()),
        Algebra::new(),
    )
}

fn run(context: &Context<'_>) {
    let graph = random_graph(context);
    let directed = directed_graph(&graph);
    let undirected = undirected_graph(&graph);
    let catalog = topology_candidates::all_shapes(context.size());
    let models = Models::upstream(1);
    let contraction_options = ContractionOptions {
        memory_limit: u64::MAX,
        weight: 0.0,
        allow_exhaustive: true,
    };
    let sizes = storage_sizes();
    let mut sparse_matvec_cache = SearchCache::new(
        context,
        &catalog,
        &models,
        sizes,
        false,
        ContractionPattern::SparseSparseSparse,
        contraction_options,
    );
    let mut dense_matvec_cache = SearchCache::new(
        context,
        &catalog,
        &models,
        sizes,
        false,
        ContractionPattern::SparseSparseDense,
        contraction_options,
    );
    let sum_sizes = [sizes[0]; 2];
    let mut sparse_sum_cache = SumSearchCache::new(
        context,
        &catalog,
        &models,
        sum_sizes,
        false,
        SumPattern::SparseSparse,
        SumOptions {
            memory_limit: u64::MAX,
        },
    );
    let mut dense_sum_cache = SumSearchCache::new(
        context,
        &catalog,
        &models,
        sum_sizes,
        false,
        SumPattern::SparseDense,
        SumOptions {
            memory_limit: u64::MAX,
        },
    );

    let algebra = Algebra::new();
    let mut removed = dense_vector(context);
    let mut independent = sparse_vector(context);
    loop {
        let mut candidates = dense_vector(context);
        candidates.transform(|_, value| *value = algebra.one());
        candidates
            .sum_from(
                "i",
                &removed,
                "i",
                Topology::new(vec![context.size()]),
                -1.0,
                algebra.one(),
            )
            .unwrap();
        let mut roots = candidates.into_sparse(|value| *value == 1.0);
        if nonzeros(&roots) == 0 {
            break;
        }

        let old_roots = roots.clone();
        sparse_matvec(
            &mut roots,
            &directed,
            &old_roots,
            algebra.one(),
            algebra.one(),
            &mut sparse_matvec_cache,
        );
        let removed_sparse = removed.clone().into_sparse(|value| *value != algebra.zero());
        sparse_add(
            &mut roots,
            &removed_sparse,
            1.0 + N as f32,
            algebra.one(),
            &mut sparse_sum_cache,
        );
        roots.sparsify(|value| *value == 1.0);

        sparse_add(
            &mut independent,
            &roots,
            algebra.one(),
            algebra.one(),
            &mut sparse_sum_cache,
        );
        let old_roots = roots.clone();
        sparse_matvec(
            &mut roots,
            &directed,
            &old_roots,
            algebra.one(),
            algebra.one(),
            &mut sparse_matvec_cache,
        );
        dense_add_sparse(
            &mut removed,
            &roots,
            algebra.one(),
            algebra.one(),
            &mut dense_sum_cache,
        );
    }

    let independent_size = nonzeros(&independent);
    if context.rank() == 0 {
        println!("Found MIS of size {independent_size}");
    }
    let mut neighbors = dense_vector(context);
    sparse_dense_matvec(
        &mut neighbors,
        &undirected,
        &independent,
        algebra.one(),
        algebra.one(),
        &mut dense_matvec_cache,
    );

    let rank = context.rank();
    let selected: Vec<_> = independent
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| independent.distribution().owner(*key) == rank)
        .collect();
    let keys: Vec<_> = selected.iter().map(|(key, _)| *key).collect();
    let neighbor_values = neighbors.read(&keys);
    let overlap = selected
        .iter()
        .zip(neighbor_values)
        .map(|((_, value), neighbor)| *value * neighbor)
        .sum::<f32>();
    let overlap = context.all_reduce(&algebra, &overlap);
    assert_eq!(overlap, 0.0, "MIS is not independent");

    dense_add_sparse(
        &mut neighbors,
        &independent,
        algebra.one(),
        algebra.one(),
        &mut dense_sum_cache,
    );
    let local_missing = neighbors
        .local_pairs()
        .into_iter()
        .filter(|(key, value)| neighbors.distribution().owner(*key) == rank && *value == 0.0)
        .count() as i32;
    let missing = context.all_reduce(&Arithmetic::<i32>::new(), &local_missing);
    assert_eq!(missing, 0, "MIS is not maximal");
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
            "DIGIT / PASS mis: source SH random graph n=16 sp=0.1; independent overlap=0 and every vertex covered; world+parity"
        );
    }
    world.close();
    drop(universe);
}
