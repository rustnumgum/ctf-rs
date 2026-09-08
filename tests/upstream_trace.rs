//! Bounded port of the active matrix-trace path in `examples/trace.cxx`.
//!
//! The source uses four NS matrices, forms the four cyclic orderings of an
//! ABCD product, extracts each diagonal, and reduces it.  This driver keeps
//! those equations and the source relative trace criterion.  The bounded
//! `N` is bounded; the source rank-seeded drand48 recurrence is retained.

use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Topology},
    tensor::Tensor,
};

const N: usize = 3;
type Algebra = Arithmetic<f64>;
type Dense<'c, 'r> = Tensor<'c, 'r, Algebra>;

struct Drand48(u64);
impl Drand48 {
    fn next(&mut self) -> f64 {
        self.0 = (self.0.wrapping_mul(0x5deece66d).wrapping_add(11)) & ((1u64 << 48) - 1);
        self.0 as f64 / (1u64 << 48) as f64
    }
}

fn matrix<'c, 'r>(context: &'c Context<'r>) -> Dense<'c, 'r> {
    Tensor::new(
        context,
        Distribution::cyclic(vec![N, N], context.size()),
        Arithmetic::<f64>::new(),
    )
}

fn random_matrix<'c, 'r>(context: &'c Context<'r>, generator: &mut Drand48) -> Dense<'c, 'r> {
    let mut result = matrix(context);
    result.transform(|_, value| *value = generator.next());
    result
}

fn multiply<'c, 'r>(
    context: &'c Context<'r>,
    left: &Dense<'c, 'r>,
    left_indices: &str,
    right: &Dense<'c, 'r>,
    right_indices: &str,
    output_indices: &str,
    topology: &Topology,
) -> Dense<'c, 'r> {
    let mut result = matrix(context);
    result
        .contract_from(
            output_indices,
            left,
            left_indices,
            right,
            right_indices,
            topology.clone(),
            1.0,
            0.0,
        )
        .unwrap();
    result
}

fn trace_of<'c, 'r>(
    matrix: &Dense<'c, 'r>,
    diagonal: &mut Dense<'c, 'r>,
    topology: &Topology,
) -> f64 {
    // DIAG["i"] = matrix["ii"]; DIAG.reduce(OP_SUM), matching trace.cxx.
    diagonal
        .sum_from("i", matrix, "ii", topology.clone(), 1.0, 0.0)
        .unwrap();
    diagonal.reduce()
}

fn run(context: &Context<'_>) {
    let topology = Topology::new(vec![context.size()]);
    let mut generator = Drand48(((13 * context.rank() as u64) << 16) | 0x330e);
    // One source stream per rank across the valid local entries of A/B/C/D.
    let a = random_matrix(context, &mut generator);
    let b = random_matrix(context, &mut generator);
    let c = random_matrix(context, &mut generator);
    let d = random_matrix(context, &mut generator);

    // C1[ij] = A[ia]*B[ab]*C[bc]*D[cj].
    let ab = multiply(context, &a, "ia", &b, "ab", "ib", &topology);
    let abc = multiply(context, &ab, "ib", &c, "bc", "ic", &topology);
    let c1 = multiply(context, &abc, "ic", &d, "cj", "ij", &topology);

    // C2[ij] = D[ia]*A[ab]*B[bc]*C[cj].
    let da = multiply(context, &d, "ia", &a, "ab", "ib", &topology);
    let dab = multiply(context, &da, "ib", &b, "bc", "ic", &topology);
    let c2 = multiply(context, &dab, "ic", &c, "cj", "ij", &topology);

    // C3[ij] = C[ia]*D[ab]*A[bc]*B[cj].
    let cd = multiply(context, &c, "ia", &d, "ab", "ib", &topology);
    let cda = multiply(context, &cd, "ib", &a, "bc", "ic", &topology);
    let c3 = multiply(context, &cda, "ic", &b, "cj", "ij", &topology);

    // C4[ij] = B[ia]*C[ab]*D[bc]*A[cj].
    let bc = multiply(context, &b, "ia", &c, "ab", "ib", &topology);
    let bcd = multiply(context, &bc, "ib", &d, "bc", "ic", &topology);
    let c4 = multiply(context, &bcd, "ic", &a, "cj", "ij", &topology);

    let mut diagonal = Tensor::new(
        context,
        Distribution::cyclic(vec![N], context.size()),
        Arithmetic::<f64>::new(),
    );
    let tr1 = trace_of(&c1, &mut diagonal, &topology);
    let tr2 = trace_of(&c2, &mut diagonal, &topology);
    let tr3 = trace_of(&c3, &mut diagonal, &topology);
    let tr4 = trace_of(&c4, &mut diagonal, &topology);

    assert!(tr1.is_finite() && tr2.is_finite() && tr3.is_finite() && tr4.is_finite());
    // Preserve trace.cxx's strict source criterion and its denominator choice.
    let pass = (tr1 - tr2).abs() / tr1 <= 1.0e-10
        && (tr2 - tr3).abs() / tr2 <= 1.0e-10
        && (tr3 - tr4).abs() / tr3 <= 1.0e-10;
    assert!(pass);
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
            "DIGIT / PASS upstream_trace: NS ABCD cyclic trace identities, seeded f64 fixture, source relative error<=1e-10; world+parity"
        );
    }
    world.close();
    drop(universe);
}
