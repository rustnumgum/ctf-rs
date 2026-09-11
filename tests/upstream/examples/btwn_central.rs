// Adapted from pinned examples/btwn_central.cxx and
// examples/btwn_central_kernels.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.

mod btwn_central_kernels;

use std::collections::HashMap;

use ctf::{
    algebra::{Arithmetic, Monoid, Semiring, Wire},
    context::Context,
    cost::Models,
    mapping::{Distribution, Topology},
    sparse::SparseTensor,
    sparse_search::{Options, Pattern, SearchCache, StorageSize},
    tensor::Tensor,
    topology_candidates,
};

use btwn_central_kernels::{
    bellman_accumulate, bellman_function, brandes_accumulate, brandes_function,
    cpath_from_mpath_initial, cpath_from_mpath_naive, cpath_from_mpath_zero_score, CPath,
    CPathMonoid, GlibcRand, MPath, MPathSemiring, Tropical, INFINITY,
};

type AdjacencyAlgebra = Tropical;
type F64 = Arithmetic<f64>;

const N: usize = 6;
const SPARSITY: f64 = 0.20;
const BATCH_SIZE: usize = 2;

#[derive(Clone, Copy)]
struct Minimum;

impl Monoid for Minimum {
    type Element = i32;

    fn zero(&self) -> Self::Element {
        2
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        (*left).min(*right)
    }
}

fn options() -> Options {
    Options {
        memory_limit: 1u64 << 60,
        weight: 0.0,
        allow_exhaustive: true,
    }
}

fn canonical_nnz<A: Monoid>(tensor: &SparseTensor<'_, '_, A>) -> u64 {
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

fn adjacency<'c, 'r>(context: &'c Context<'r>) -> SparseTensor<'c, 'r, AdjacencyAlgebra> {
    let distribution = Distribution::cyclic(vec![N, N], context.size());
    let mut result = SparseTensor::new(context, distribution.clone(), AdjacencyAlgebra);
    let edges_per_row = ((N as f64 * SPARSITY) as usize).max(1);
    let weight_bound = (N * N).min(20);
    let mut generator = GlibcRand::new((context.rank() + 1) as u32);
    let row_start = context.rank() * N / context.size();
    let row_end = (context.rank() + 1) * N / context.size();
    let mut pairs = Vec::new();

    for row in row_start..row_end {
        let mut columns = Vec::with_capacity(edges_per_row);
        while columns.len() < edges_per_row {
            let column = generator.index(N);
            if columns.contains(&column) {
                continue;
            }
            columns.push(column);
            let weight = generator.index(weight_bound) as i32 + 1;
            pairs.push((distribution.encode_key(&[row, column]), weight));
        }
    }
    result.write_add(&pairs);

    // A["ii"] = 0 in the source is an overwrite, not an additive zero.
    let diagonal: Vec<_> = (0..N)
        .map(|index| (distribution.encode_key(&[index, index]), 0))
        .filter(|(key, _)| distribution.owner(*key) == context.rank())
        .collect();
    let one = AdjacencyAlgebra.one();
    let zero = AdjacencyAlgebra.zero();
    result.write_scaled(&diagonal, &one, &zero);
    result
}

fn apply_bellman_post(
    all_b: &Tensor<'_, '_, MPathSemiring>,
    b: &mut SparseTensor<'_, '_, MPathSemiring>,
) {
    let pairs = b.local_pairs();
    let keys: Vec<_> = pairs.iter().map(|(key, _)| *key).collect();
    let previous = all_b.read(&keys);
    let previous: HashMap<_, _> = keys.into_iter().zip(previous).collect();
    b.transform_stored(|key, value| {
        let old = previous[&key];
        if old.w < value.w || (old.w == value.w && value.m == 0) {
            value.w = INFINITY;
        }
    });
}

fn apply_brandes_post(
    all_b: &Tensor<'_, '_, MPathSemiring>,
    c_b: &mut SparseTensor<'_, '_, CPathMonoid>,
    multiply_score_by_path_count: bool,
) {
    let pairs = c_b.local_pairs();
    let keys: Vec<_> = pairs.iter().map(|(key, _)| *key).collect();
    let shortest_paths = all_b.read(&keys);
    let shortest_paths: HashMap<_, _> = keys.into_iter().zip(shortest_paths).collect();
    c_b.transform_stored(|key, value| {
        let path = shortest_paths[&key];
        let score = if path.w == value.w {
            if multiply_score_by_path_count {
                value.c * path.m as f64
            } else {
                value.c
            }
        } else {
            0.0
        };
        *value = CPath::new(path.w, 1.0 / path.m as f32, score);
    });
}

fn accumulate_scores(
    context: &Context<'_>,
    scores: &Tensor<'_, '_, CPathMonoid>,
    output: &mut Tensor<'_, '_, F64>,
) {
    let rank = context.rank();
    let mut local = vec![0.0; N];
    for (key, value) in scores.local_pairs() {
        if scores.distribution().owner(key) == rank {
            let coordinates = scores.distribution().decode_key(key);
            local[coordinates[0]] += value.c;
        }
    }
    let pairs: Vec<_> = (0..N)
        .filter(|&key| output.distribution().owner(key) == rank)
        .map(|key| (key, local[key]))
        .collect();
    output.write_add(&pairs);
}

fn fast<'c, 'r>(
    context: &'c Context<'r>,
    adjacency: &SparseTensor<'c, 'r, AdjacencyAlgebra>,
) -> Tensor<'c, 'r, F64> {
    let mut source = adjacency.clone();
    let source_distribution = source.distribution().clone();
    source.transform_stored(|key, value| {
        let coordinates = source_distribution.decode_key(key);
        if coordinates[0] == coordinates[1] {
            *value = INFINITY;
        }
    });

    let catalog = topology_candidates::all_shapes(context.size());
    let models = Models::upstream(1);
    let mut bellman_cache = SearchCache::new(
        context,
        &catalog,
        &models,
        [
            StorageSize {
                element_bytes: <i32 as Wire>::WIDTH,
                pair_bytes: 8 + <i32 as Wire>::WIDTH,
            },
            StorageSize {
                element_bytes: MPath::WIDTH,
                pair_bytes: 8 + MPath::WIDTH,
            },
            StorageSize {
                element_bytes: MPath::WIDTH,
                pair_bytes: 8 + MPath::WIDTH,
            },
        ],
        true,
        Pattern::SparseSparseSparse,
        options(),
    );
    let mut brandes_cache = SearchCache::new(
        context,
        &catalog,
        &models,
        [
            StorageSize {
                element_bytes: <i32 as Wire>::WIDTH,
                pair_bytes: 8 + <i32 as Wire>::WIDTH,
            },
            StorageSize {
                element_bytes: CPath::WIDTH,
                pair_bytes: 8 + CPath::WIDTH,
            },
            StorageSize {
                element_bytes: CPath::WIDTH,
                pair_bytes: 8 + CPath::WIDTH,
            },
        ],
        true,
        Pattern::SparseSparseSparse,
        options(),
    );

    let vector_distribution = Distribution::cyclic(vec![N], context.size());
    let mut result = Tensor::new(context, vector_distribution, F64::new());
    let rank = context.rank();
    let source_nnz = canonical_nnz(&source);

    for batch_start in (0..N).step_by(BATCH_SIZE) {
        let batch_size = BATCH_SIZE.min(N - batch_start);
        let batch_distribution = Distribution::cyclic(vec![N, batch_size], context.size());
        let input = source.slice(&[0..N, batch_start..batch_start + batch_size]);
        let mut b = SparseTensor::new(context, batch_distribution.clone(), MPathSemiring);
        let initial: Vec<_> = input
            .local_pairs()
            .into_iter()
            .filter(|(key, _)| input.distribution().owner(*key) == rank)
            .map(|(key, weight)| (key, MPath::new(weight, 1)))
            .collect();
        b.write_add(&initial);
        let mut all_b = Tensor::new(context, batch_distribution.clone(), MPathSemiring);
        all_b.write_add(&initial);

        for _ in 0..N {
            let mut c = b.clone();
            c.sparsify(|path| path.w < INFINITY);
            if canonical_nnz(&c) == 0 {
                break;
            }
            b.sparsify(|_| false);
            let selected = bellman_cache
                .prepare(
                    [source.distribution(), c.distribution(), b.distribution()],
                    ["ik", "kj", "ij"],
                    [
                        Some(source_nnz),
                        Some(canonical_nnz(&c)),
                        Some(canonical_nnz(&b)),
                    ],
                    None,
                )
                .unwrap()
                .unwrap()
                .clone();
            b.contract_sparse_function_from_selected(
                "ij",
                &source,
                "ik",
                &c,
                "kj",
                &selected,
                bellman_function,
                bellman_accumulate,
            );
            apply_bellman_post(&all_b, &mut b);
            let updates: Vec<_> = b
                .local_pairs()
                .into_iter()
                .filter(|(key, _)| b.distribution().owner(*key) == rank)
                .collect();
            all_b.write_add(&updates);
        }

        let identity: Vec<_> = (0..batch_size)
            .map(|column| {
                (
                    all_b
                        .distribution()
                        .encode_key(&[batch_start + column, column]),
                    MPath::new(0, 1),
                )
            })
            .filter(|(key, _)| all_b.distribution().owner(*key) == rank)
            .collect();
        all_b.write_add(&identity);

        let mut c_b = SparseTensor::new(context, batch_distribution.clone(), CPathMonoid);
        let initial: Vec<_> = all_b
            .local_pairs()
            .into_iter()
            .filter(|(key, _)| all_b.distribution().owner(*key) == rank)
            .map(|(key, path)| (key, cpath_from_mpath_initial(path)))
            .collect();
        c_b.write_add(&initial);
        let mut all_c_b = Tensor::new(context, batch_distribution, CPathMonoid);
        let initial_scores: Vec<_> = all_b
            .local_pairs()
            .into_iter()
            .filter(|(key, _)| all_b.distribution().owner(*key) == rank)
            .map(|(key, path)| (key, cpath_from_mpath_zero_score(path)))
            .collect();
        all_c_b.write_add(&initial_scores);

        let source_nnz = canonical_nnz(&source);
        for _ in 0..N {
            let mut c = c_b.clone();
            c.sparsify(|path| path.w >= 0 && path.c != 0.0);
            if canonical_nnz(&c) == 0 {
                break;
            }
            c_b.sparsify(|_| false);
            let selected = brandes_cache
                .prepare(
                    [source.distribution(), c.distribution(), c_b.distribution()],
                    ["ki", "kj", "ij"],
                    [
                        Some(source_nnz),
                        Some(canonical_nnz(&c)),
                        Some(canonical_nnz(&c_b)),
                    ],
                    None,
                )
                .unwrap()
                .unwrap()
                .clone();
            c_b.contract_sparse_function_from_selected(
                "ij",
                &source,
                "ki",
                &c,
                "kj",
                &selected,
                brandes_function,
                brandes_accumulate,
            );
            apply_brandes_post(&all_b, &mut c_b, true);
            let updates: Vec<_> = c_b
                .local_pairs()
                .into_iter()
                .filter(|(key, _)| c_b.distribution().owner(*key) == rank)
                .collect();
            all_c_b.write_add(&updates);
        }
        apply_brandes_post(&all_b, &mut c_b, false);
        all_c_b.transform(|_, path| {
            if path.w == 0 {
                path.c = 0.0;
            }
        });
        accumulate_scores(context, &all_c_b, &mut result);
    }
    result
}

fn naive<'c, 'r>(
    context: &'c Context<'r>,
    adjacency: &SparseTensor<'c, 'r, AdjacencyAlgebra>,
) -> Tensor<'c, 'r, F64> {
    let distribution = Distribution::cyclic(vec![N, N], context.size());
    let mut p = Tensor::new(context, distribution.clone(), MPathSemiring);
    let rank = context.rank();
    let initial: Vec<_> = adjacency
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| adjacency.distribution().owner(*key) == rank)
        .map(|(key, weight)| (key, MPath::new(weight, 1)))
        .collect();
    p.write_add(&initial);
    p.transform_indexed("ii", |path| *path = MPath::new(INFINITY, 1));
    let pi = p.clone();
    let algebra = MPathSemiring;
    let topology = Topology::new(vec![context.size()]);
    for _ in 0..N {
        p.transform_indexed("ii", |path| *path = MPath::new(0, 1));
        let old_p = p.clone();
        p.contract_from(
            "ij",
            &pi,
            "ik",
            &old_p,
            "kj",
            topology.clone(),
            algebra.one(),
            algebra.zero(),
        )
        .unwrap();
    }
    p.transform_indexed("ii", |path| *path = MPath::new(INFINITY, 1));

    let mut postv = Tensor::new(
        context,
        Distribution::cyclic(vec![N, N, N], context.size()),
        CPathMonoid,
    );
    let p_distribution = p.distribution().clone();
    let postv_distribution = postv.distribution().clone();
    let mut needed = vec![false; N * N];
    for (key, _) in postv.local_pairs() {
        if postv_distribution.owner(key) != rank {
            continue;
        }
        let coordinates = postv_distribution.decode_key(key);
        needed[p_distribution.encode_key(&[coordinates[0], coordinates[1]])] = true;
        needed[p_distribution.encode_key(&[coordinates[1], coordinates[2]])] = true;
        needed[p_distribution.encode_key(&[coordinates[0], coordinates[2]])] = true;
    }
    let requested_keys: Vec<_> = needed
        .into_iter()
        .enumerate()
        .filter_map(|(key, needed)| needed.then_some(key))
        .collect();
    let requested_values = p.read(&requested_keys);
    let path_values: HashMap<_, _> = requested_keys.into_iter().zip(requested_values).collect();
    postv.transform(|key, value| {
        let coordinates = postv_distribution.decode_key(key);
        let path_key = p_distribution.encode_key(&[coordinates[0], coordinates[2]]);
        *value = cpath_from_mpath_naive(path_values[&path_key]);
    });
    postv.transform(|key, value| {
        let coordinates = postv_distribution.decode_key(key);
        let a_key = p_distribution.encode_key(&[coordinates[0], coordinates[1]]);
        let b_key = p_distribution.encode_key(&[coordinates[1], coordinates[2]]);
        let c_key = p_distribution.encode_key(&[coordinates[0], coordinates[2]]);
        let a = path_values[&a_key];
        let b = path_values[&b_key];
        let c = path_values[&c_key];
        value.c = if c.w < INFINITY && a.w + b.w == c.w {
            (a.m as f64 * b.m as f64) / value.m as f64
        } else {
            0.0
        };
    });

    let mut result = Tensor::new(
        context,
        Distribution::cyclic(vec![N], context.size()),
        F64::new(),
    );
    let mut local = vec![0.0; N];
    for (key, value) in postv.local_pairs() {
        if postv_distribution.owner(key) == rank {
            let coordinates = postv.distribution().decode_key(key);
            local[coordinates[1]] += value.c;
        }
    }
    let pairs: Vec<_> = (0..N).map(|key| (key, local[key])).collect();
    result.write_add(&pairs);
    result
}

fn difference_norm<'c, 'r>(
    reference: &Tensor<'c, 'r, F64>,
    actual: &Tensor<'c, 'r, F64>,
) -> f64 {
    assert_eq!(reference.distribution().shape, actual.distribution().shape);
    let mut difference = reference.clone();
    difference
        .sum_from(
            "i",
            actual,
            "i",
            Topology::new(vec![reference.context().size()]),
            -1.0,
            1.0,
        )
        .unwrap();
    difference.norm2()
}

fn run(context: &Context<'_>) {
    let graph = adjacency(context);
    let actual = fast(context, &graph);
    let reference = naive(context, &graph);
    let difference = difference_norm(&reference, &actual);
    let local_pass = i32::from(difference <= N as f64 * 1e-6);
    let pass = context.all_reduce(&Minimum, &local_pass);
    if context.rank() == 0 {
        println!(
            "btwn_central: norm2 = {difference:e}, bound=6e-6"
        );
        if pass == 1 {
            println!("{{ betweenness centrality }} passed");
        } else {
            println!("{{ betweenness centrality }} failed");
        }
    }
    assert_eq!(pass, 1);
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    run(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS btwn_central: Bellman/Brandes custom sparse SSS versus dense naive; n=6 sp=0.2 bsize=2 test=1 sp_B=1 sp_C=1; norm2<=6e-6; world+parity"
        );
    }
    world.close();
    drop(universe);
}
