//! Bounded port of the active Bellman--Ford path in `examples/sssp.cxx`.
//!
//! The source uses a tropical integer semiring for the adjacency matrix and
//! vector distances.  Its `P["ij"]` conversion is intentionally retained as
//! an ordinary integer SUM: a rank-one Vector copies only its first index
//! label, and `Term::operator int()` reduces through the default int ring.

use ctf::{
    algebra::{Arithmetic, Monoid, Semiring},
    context::Context,
    mapping::{Distribution, Topology},
    random::Generator,
    sparse::SparseTensor,
    tensor::Tensor,
};

const N: usize = 7;
const INF: i32 = (N * N) as i32;
const SPARSE_CUTOFF: i32 = (5 * N) as i32;

#[derive(Clone, Copy)]
struct Tropical {
    infinity: i32,
}

impl Tropical {
    fn new() -> Self {
        Self { infinity: INF }
    }
}

impl Monoid for Tropical {
    type Element = i32;

    fn zero(&self) -> Self::Element {
        self.infinity
    }

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (*a).min(*b)
    }
}

impl Semiring for Tropical {
    fn one(&self) -> Self::Element {
        0
    }

    fn multiply(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        *a + *b
    }
}

fn adjacency<'c, 'r>(context: &'c Context<'r>) -> SparseTensor<'c, 'r, Tropical> {
    let distribution = Distribution::cyclic(vec![N, N], context.size());

    // The pinned example uses drand48. The current Rust source
    // generator is MT19937-64, so this preserves the source bounds and
    // allocation-local integer cast while making the bounded fixture explicit.
    let mut source = Tensor::new(context, distribution.clone(), Arithmetic::<i32>::new());
    let mut generator = Generator::new(context.rank() as u64);
    source.fill_random(0, INF, &mut generator);
    source.transform(|key, value| {
        let coordinates = distribution.decode_key(key);
        if coordinates[0] == coordinates[1] {
            *value = INF;
        }
    });

    // Materialize the random integer matrix in the tropical algebra, then
    // retain exactly the source's `A.sparsify(a < 5*n)` branch.
    let mut dense = Tensor::new(context, distribution, Tropical::new());
    dense.write_add(&source.local_pairs());
    dense.into_sparse(|value| *value < SPARSE_CUTOFF)
}

fn seed_vector<'c, 'r>(context: &'c Context<'r>) -> Tensor<'c, 'r, Tropical> {
    let distribution = Distribution::cyclic(vec![N], context.size());
    let mut vector = Tensor::new(context, distribution, Tropical::new());
    let seed = if context.rank() == 0 {
        vec![(0, 0)]
    } else {
        Vec::new()
    };
    vector.write_add(&seed);
    vector
}

fn total_weight(context: &Context<'_>, vector: &Tensor<'_, '_, Tropical>) -> i32 {
    let local = vector
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| vector.distribution().owner(*key) == context.rank())
        .map(|(_, value)| value)
        .sum::<i32>();
    context.all_reduce(&Arithmetic::<i32>::new(), &local)
}

fn bellman_ford(
    context: &Context<'_>,
    adjacency: &SparseTensor<'_, '_, Tropical>,
    distances: &mut Tensor<'_, '_, Tropical>,
    topology: &Topology,
) -> bool {
    let algebra = Tropical::new();
    let mut round = 0;
    let mut new_total_weight = total_weight(context, distances);

    loop {
        if round == N + 1 {
            return false;
        }
        round += 1;

        // This is the source Q(P) snapshot and also makes the read-before-
        // write operand explicit for the Rust contraction API.
        let old_distances = distances.clone();
        distances.contract_from_sparse_dense_on(
            "i",
            adjacency,
            "ij",
            &old_distances,
            "j",
            topology.clone(),
            "i",
            &[],
            algebra.one(),
            algebra.one(),
            true,
        );

        let previous_total_weight = new_total_weight;
        new_total_weight = total_weight(context, distances);
        assert!(new_total_weight <= previous_total_weight);
        if new_total_weight >= previous_total_weight {
            return true;
        }
    }
}

fn run(context: &Context<'_>) {
    let topology = Topology::new(vec![context.size()]);
    let adjacency = adjacency(context);
    let mut distances = seed_vector(context);

    assert!(bellman_ford(context, &adjacency, &mut distances, &topology));

    // Match the source reset `v["i"] = n*n` and source-at-zero write.
    let infinity = Tropical::new().zero();
    distances.transform(|_, value| *value = infinity);
    let seed = if context.rank() == 0 {
        vec![(0, 0)]
    } else {
        Vec::new()
    };
    distances.write_add(&seed);

    let mut negative = adjacency.clone();
    let cycle = if context.rank() == 0 {
        vec![(1, 1), (N + 2, -1), (2 * N, -1)]
    } else {
        Vec::new()
    };
    negative.write_scaled(&cycle, &0, &INF);
    assert!(!bellman_ford(context, &negative, &mut distances, &topology));
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    run(&world);

    let rank = world.rank();
    let parity = world.split(Some((rank % 2) as i32), rank as i32).unwrap();
    run(&parity);
    parity.close();

    if rank == 0 {
        println!(
            "DIGIT / PASS upstream_sssp: bounded tropical sparse Bellman-Ford, source SUM convergence check, injected negative cycle; world+parity"
        );
    }
    world.close();
    drop(universe);
}
