//! Bounded port of the active recursive tensor-contraction path in
//! `examples/scan.cxx`.
//!
//! The source's Matrix(2, 2, 2) is AS (the third argument is the matrix
//! symmetry), then it repacks AS -> SH -> NS.  `repack_to` preserves that
//! canonical-only repack, so the resulting W is the strict upper-triangular
//! shift used by the recursive scan.  The source read_all check is represented
//! by explicit all-rank exports; no independent prefix oracle is used.

use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    random::Generator,
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, AS, NS, SH},
    tensor::Tensor as DenseTensor,
};

const LOG_N: usize = 3;
type Algebra = Arithmetic<f64>;
type Dense<'c, 'r> = DenseTensor<'c, 'r, Algebra>;
type Tensor<'c, 'r> = SymmetricTensor<'c, 'r, Algebra>;

fn distribution(
    context: &Context<'_>,
    shape: Vec<usize>,
    links: Vec<Symmetry>,
) -> SymmetricDistribution {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    if let Some(first) = mappings.first_mut() {
        first.augment_physical(&topology, 0);
    }
    for mapping in &mut mappings {
        mapping.augment_virtual(2 * context.size());
    }
    SymmetricDistribution::new(Distribution::new(shape, topology, mappings), links)
}

fn labels(order: usize) -> String {
    (0..order).map(|axis| (b'a' + axis as u8) as char).collect()
}

fn shift_matrix<'c, 'r>(context: &'c Context<'r>) -> Tensor<'c, 'r> {
    // Matrix<dtype> W(2, V.lens[0], V.lens[0], ...) has atr=2, i.e. AS.
    let mut w_as = Tensor::new(
        context,
        distribution(context, vec![2, 2], vec![AS, NS]),
        Algebra::new(),
    );
    w_as.transform(|_, value| *value = 1.0);
    let w_sh = w_as.repack_to(distribution(context, vec![2, 2], vec![SH, NS]));
    w_sh.repack_to(distribution(context, vec![2, 2], vec![NS, NS]))
}

fn clone_tensor<'c, 'r>(tensor: &Tensor<'c, 'r>) -> Tensor<'c, 'r> {
    tensor.repack_to(tensor.distribution().clone())
}

fn rec_scan<'c, 'r>(tensor: &mut Tensor<'c, 'r>) {
    let order = tensor.distribution().links().len();
    let w = shift_matrix(tensor.context());
    let topology = Topology::new(vec![tensor.context().size()]);

    if order == 1 {
        let old = clone_tensor(tensor);
        tensor
            .contract_from_on("a", &w, "ba", &old, "b", topology, "a", 1.0, 0.0, true)
            .unwrap();
        return;
    }

    let output_indices = labels(order);
    let reduced_indices = &output_indices[1..];
    let mut reduced = Tensor::new(
        tensor.context(),
        distribution(tensor.context(), vec![2; order - 1], vec![NS; order - 1]),
        Algebra::new(),
    );
    reduced.sum_from(reduced_indices, tensor, &output_indices, 1.0, 0.0);
    rec_scan(&mut reduced);

    let new_label = (b'a' + order as u8) as char;
    let old_indices = format!("{new_label}{reduced_indices}");
    let matrix_indices = format!("{new_label}a");
    let old = clone_tensor(tensor);
    tensor
        .contract_from_on(
            &output_indices,
            &w,
            &matrix_indices,
            &old,
            &old_indices,
            topology,
            "a",
            1.0,
            0.0,
            true,
        )
        .unwrap();
    tensor.sum_from(&output_indices, &reduced, reduced_indices, 1.0, 1.0);
}

fn scan<'c, 'r>(context: &'c Context<'r>, vector: &mut Dense<'c, 'r>) {
    let mut tensor = Tensor::new(
        context,
        distribution(context, vec![2; LOG_N], vec![NS; LOG_N]),
        Algebra::new(),
    );

    // The source obtains local vector indices/data, writes those indexed
    // entries into V, then obtains local V indices/data and writes them back
    // into the vector.  These collective additive writes target zeroed
    // tensors, so they preserve the source overwrite semantics without a
    // dense gather.
    let vector_pairs = vector.local_pairs();
    tensor.write_add(&vector_pairs);
    rec_scan(&mut tensor);

    let output_pairs = tensor.local_pairs();
    vector.transform(|_, value| *value = 0.0);
    vector.write_add(&output_pairs);
}

fn run(context: &Context<'_>) {
    let length = 1usize << LOG_N;
    let vector_distribution = Distribution::cyclic(vec![length], context.size());
    let mut vector = Dense::new(context, vector_distribution, Algebra::new());
    let mut generator = Generator::new(context.rank() as u64 * 27);
    vector.fill_random(0.0, 1.0, &mut generator);

    // Source get_all_data/read_all are acceptance exports, not computation.
    let start_data = vector.all_data();
    scan(context, &mut vector);
    let data = vector.all_data();
    for i in 1..length {
        let error = data[i] - start_data[i - 1] - data[i - 1];
        // The source uses abs(error) < 1.E-9*(1<<logn); retain that strict
        // bound and reject non-finite numerical results explicitly.
        assert!(error.is_finite() && error.abs() < 1.0e-9 * length as f64);
    }
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
            "DIGIT / PASS upstream_scan: recursive AS->SH->NS shift contractions, seeded f64 vector adapter, source abs(error)<1e-9*N; world+parity"
        );
    }
    world.close();
    drop(universe);
}
