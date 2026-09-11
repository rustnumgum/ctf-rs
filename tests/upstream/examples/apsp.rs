//! Bounded Rust port of pinned `examples/apsp.cxx`.
//!
//! The dense path uses tropical path doubling.  The sparse path retains the
//! source's augmented paths: `Pi` contains exactly the paths with `i` hops and
//! is selected/executed through the automatic sparse planner for
//! `Pi * P -> P`.

use ctf::{
    algebra::{Arithmetic, Monoid, Semiring, Wire},
    context::Context,
    cost::Models,
    mapping::{Distribution, Topology},
    random::Generator,
    sparse::SparseTensor,
    sparse_search::{Options, Pattern, SearchCache, StorageSize},
    tensor::Tensor,
    topology_candidates,
};

const N: usize = 9;
const INFINITY: i32 = i32::MAX / 2;

#[derive(Clone, Copy)]
struct Tropical;

impl Monoid for Tropical {
    type Element = i32;

    fn zero(&self) -> Self::Element {
        INFINITY
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        (*left).min(*right)
    }
}

impl Semiring for Tropical {
    fn one(&self) -> Self::Element {
        0
    }

    fn multiply(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        *left + *right
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Path {
    weight: i32,
    hops: i32,
}

impl Wire for Path {
    const WIDTH: usize = 8;

    fn encode(&self, output: &mut Vec<u8>) {
        self.weight.encode(output);
        self.hops.encode(output);
    }

    fn decode(input: &[u8]) -> Self {
        Self {
            weight: i32::decode(&input[..4]),
            hops: i32::decode(&input[4..]),
        }
    }
}

#[derive(Clone, Copy)]
struct PathAlgebra;

impl Monoid for PathAlgebra {
    type Element = Path;

    fn zero(&self) -> Self::Element {
        Path {
            weight: INFINITY,
            hops: 0,
        }
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        if left.weight < right.weight
            || (left.weight == right.weight && left.hops < right.hops)
        {
            *left
        } else {
            *right
        }
    }
}

impl Semiring for PathAlgebra {
    fn one(&self) -> Self::Element {
        Path {
            weight: 0,
            hops: 0,
        }
    }

    fn multiply(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        Path {
            weight: left.weight + right.weight,
            hops: left.hops + right.hops,
        }
    }
}

fn adjacency<'c, 'r>(context: &'c Context<'r>) -> Tensor<'c, 'r, Tropical> {
    let distribution = Distribution::cyclic(vec![N, N], context.size());
    let mut source = Tensor::new(
        context,
        distribution.clone(),
        Arithmetic::<i32>::new(),
    );
    let mut generator = Generator::new(context.rank() as u64);
    source.fill_random(0, (N * N) as i32, &mut generator);

    // Match the source's rank-seeded random adjacency fixture and no-loop
    // replacement.  The integer random fill follows upstream_sssp.rs.
    source.transform(|key, value| {
        let coordinates = distribution.decode_key(key);
        if coordinates[0] == coordinates[1] {
            *value = 0;
        }
    });

    let mut result = Tensor::new(context, distribution, Tropical);
    result.write_add(&source.local_pairs());
    result
}

fn dense_path_doubling<'c, 'r>(
    context: &'c Context<'r>,
    adjacency: &Tensor<'c, 'r, Tropical>,
) -> Tensor<'c, 'r, Tropical> {
    let algebra = Tropical;
    let mut distances = adjacency.clone();
    let topology = Topology::new(vec![context.size()]);
    let mut hops = 1;
    while hops < N {
        let old_distances = distances.clone();
        distances
            .contract_from(
                "ij",
                &old_distances,
                "ik",
                &old_distances,
                "kj",
                topology.clone(),
                algebra.one(),
                algebra.one(),
            )
            .unwrap();
        hops <<= 1;
    }
    distances
}

fn path_matrix<'c, 'r>(
    context: &'c Context<'r>,
    adjacency: &Tensor<'c, 'r, Tropical>,
) -> Tensor<'c, 'r, PathAlgebra> {
    let mut paths = Tensor::new(context, adjacency.distribution().clone(), PathAlgebra);
    let pairs: Vec<_> = adjacency
        .local_pairs()
        .into_iter()
        .map(|(key, weight)| (
            key,
            Path {
                weight,
                hops: 1,
            },
        ))
        .collect();
    paths.write_add(&pairs);
    paths
}

fn canonical_nnz(matrix: &SparseTensor<'_, '_, PathAlgebra>) -> u64 {
    let rank = matrix.context().rank();
    let local = matrix
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| matrix.distribution().owner(*key) == rank)
        .count() as u64;
    matrix
        .context()
        .all_reduce(&Arithmetic::<u64>::new(), &local)
}

fn sparse_path_doubling<'c, 'r>(
    context: &'c Context<'r>,
    adjacency: &Tensor<'c, 'r, Tropical>,
) -> Tensor<'c, 'r, PathAlgebra> {
    let algebra = PathAlgebra;
    let mut paths = path_matrix(context, adjacency);
    let candidate_catalog = topology_candidates::all_shapes(context.size());
    let models = Models::upstream(1);
    let mut cache = SearchCache::new(
        context,
        &candidate_catalog,
        &models,
        [StorageSize {
            element_bytes: Path::WIDTH,
            pair_bytes: 8 + Path::WIDTH,
        }; 3],
        true,
        Pattern::SparseDenseDense { coo_kernel: false },
        Options {
            memory_limit: u64::MAX,
            weight: 0.0,
            allow_exhaustive: true,
        },
    );

    let mut hops = 1;
    while hops < N {
        let old_paths = paths.clone();
        let paths_i = old_paths
            .clone()
            .into_sparse(|path| path.hops == hops as i32);
        let global_nnz = canonical_nnz(&paths_i);
        let selected = cache
            .prepare(
                [
                    paths_i.distribution(),
                    old_paths.distribution(),
                    paths.distribution(),
                ],
                ["ik", "kj", "ij"],
                [Some(global_nnz), None, None],
                None,
            )
            .unwrap()
            .expect("automatic sparse APSP selection rejected the fixture");
        paths.contract_sparse_from_selected(
            "ij",
            &paths_i,
            "ik",
            &old_paths,
            "kj",
            selected,
            algebra.one(),
            algebra.one(),
            true,
        );
        hops <<= 1;
    }
    paths
}

fn no_difference(
    context: &Context<'_>,
    distances: &Tensor<'_, '_, Tropical>,
    paths: &Tensor<'_, '_, PathAlgebra>,
) -> u64 {
    let keys: Vec<_> = (0..N * N).collect();
    let distance_values = distances.read(&keys);
    let path_values = paths.read(&keys);
    let local = keys
        .iter()
        .enumerate()
        .filter(|(position, key)| {
            distances.distribution().owner(**key) == context.rank()
                && path_values[*position].weight != distance_values[*position]
        })
        .count() as u64;
    context.all_reduce(&Arithmetic::<u64>::new(), &local)
}

fn run(context: &Context<'_>) -> u64 {
    let adjacency = adjacency(context);
    let distances = dense_path_doubling(context, &adjacency);
    let paths = sparse_path_doubling(context, &adjacency);
    let q = no_difference(context, &distances, &paths);
    assert_eq!(q, 0, "APSP path-weight difference count {q}");
    q
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = Context::world(&universe);
    let q = run(&world);

    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&parity);
    parity.close();

    if world.rank() == 0 {
        println!(
            "DIGIT / PASS apsp: dense tropical path doubling equals sparse augmented path doubling; n={N}; Q={q}; ref=0; bound=0; world+parity"
        );
    }
    world.close();
    drop(universe);
}
