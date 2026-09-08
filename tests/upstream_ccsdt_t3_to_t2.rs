// Bounded port of the active path in pinned `test/ccsdt_t3_to_t2.cxx`.
//
// The source compares a symmetry-aware CCSDT T3-to-T2 contraction with the
// explicitly expanded nonsymmetric form.  The Rust path preserves the
// source's partial-AS reference storage and uses the symmetry-aware indexed
// sum/contraction APIs for both execution paths.

use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, *},
};

const N: usize = 3;
const M: usize = 4;

fn symmetric_tensor<'c, 'r>(
    context: &'c Context<'r>,
    shape: Vec<usize>,
    links: Vec<Symmetry>,
) -> SymmetricTensor<'c, 'r, Arithmetic<f64>> {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    if !mappings.is_empty() {
        mappings[0].augment_physical(&topology, 0);
    }
    for mapping in &mut mappings {
        mapping.augment_virtual(2 * context.size());
    }
    SymmetricTensor::new(
        context,
        SymmetricDistribution::new(Distribution::new(shape, topology, mappings), links),
        Arithmetic::new(),
    )
}

fn fixture(global_key: usize, seed: u64) -> f64 {
    let mixed = (global_key as u64)
        .wrapping_add(seed)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .rotate_left(29);
    // The source uses fill_random(0., 1.); this bounded port uses a
    // rank-independent canonical-key fixture in the same interval. Random
    // allocation-order filling is covered separately by symmetric_random.
    ((mixed % 2003) as f64 + 1.0) / 2004.0
}

fn symmetric_square_norm<'c, 'r>(
    context: &'c Context<'r>,
    tensor: &SymmetricTensor<'c, 'r, Arithmetic<f64>>,
    topology: Topology,
) -> f64 {
    let mut scalar = symmetric_tensor(context, vec![], vec![]);
    scalar
        .contract_from_on(
            "", tensor, "ijkl", tensor, "ijkl", topology, "i", 1.0, 0.0, true,
        )
        .unwrap();
    scalar.read(&[0])[0].sqrt()
}

fn run(context: &Context<'_>) {
    let as_a = {
        let mut tensor = symmetric_tensor(context, vec![N, N, N, M], vec![AS, NS, NS, NS]);
        tensor.transform(|key, value| *value = fixture(key, 13));
        tensor
    };
    let as_b = {
        let mut tensor = symmetric_tensor(
            context,
            vec![M, M, M, N, N, N],
            vec![AS, AS, NS, AS, AS, NS],
        );
        tensor.transform(|key, value| *value = fixture(key, 29));
        tensor
    };
    let mut as_c = {
        let mut tensor = symmetric_tensor(context, vec![M, M, N, N], vec![AS, NS, AS, NS]);
        tensor.transform(|key, value| *value = fixture(key, 47));
        tensor
    };

    // Preserve the source's partial-AS reference storage.  NS_A is fully
    // nonsymmetric, while NS_B and NS_C retain only the first AS pair.
    let mut ns_a = symmetric_tensor(context, vec![N, N, N, M], vec![NS, NS, NS, NS]);
    ns_a.sum_from("mnje", &as_a, "mnje", 1.0, 0.0);
    let mut ns_b = symmetric_tensor(
        context,
        vec![M, M, M, N, N, N],
        vec![AS, NS, NS, NS, NS, NS],
    );
    ns_b.sum_from("abeimn", &as_b, "abeimn", 1.0, 0.0);
    let mut ns_c = symmetric_tensor(context, vec![M, M, N, N], vec![AS, NS, NS, NS]);
    ns_c.sum_from("abij", &as_c, "abij", 1.0, 0.0);

    let topology = Topology::new(vec![context.size()]);
    as_c.contract_from_on(
        "abij",
        &as_a,
        "mnje",
        &as_b,
        "abeimn",
        topology.clone(),
        "a",
        0.5,
        1.0,
        true,
    )
    .unwrap();

    ns_c.contract_from_on(
        "abij",
        &ns_a,
        "mnje",
        &ns_b,
        "abeimn",
        topology.clone(),
        "a",
        0.5,
        1.0,
        true,
    )
    .unwrap();
    ns_c.contract_from_on(
        "abji",
        &ns_a,
        "mnje",
        &ns_b,
        "abeimn",
        topology.clone(),
        "a",
        -0.5,
        1.0,
        true,
    )
    .unwrap();

    let nrm_as = symmetric_square_norm(context, &as_c, topology.clone());
    let nrm_ns = symmetric_square_norm(context, &ns_c, topology.clone());
    assert!(nrm_as.is_finite() && nrm_ns.is_finite());
    assert!((nrm_as - as_c.norm2()).abs() < 1.0e-6);
    assert!((nrm_ns - ns_c.norm2()).abs() < 1.0e-6);

    ns_c.sum_from("abij", &as_c, "abij", -1.0, 1.0);
    let residual = ns_c.norm2();
    assert!(residual.is_finite() && residual <= 1.0e-6);
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
            "DIGIT / PASS upstream_ccsdt_t3_to_t2: AS packed and NS expanded CCSDT contraction equality; norm checks <1e-6; residual <=1e-6; world+parity"
        );
    }
    world.close();
    drop(universe);
}
