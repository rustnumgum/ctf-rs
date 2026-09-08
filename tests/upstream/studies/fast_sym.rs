// Port of pinned CTF `studies/fast_sym.cxx`.

use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, *},
};

const N: usize = 13;

fn tensor<'c, 'r>(
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
    ((mixed % 2003) as f64 + 1.0) / 2004.0
}

fn contract<'c, 'r>(
    output: &mut SymmetricTensor<'c, 'r, Arithmetic<f64>>,
    output_indices: &str,
    left: &SymmetricTensor<'c, 'r, Arithmetic<f64>>,
    left_indices: &str,
    right: &SymmetricTensor<'c, 'r, Arithmetic<f64>>,
    right_indices: &str,
    alpha: f64,
    beta: f64,
) {
    output
        .contract_from(
            output_indices,
            left,
            left_indices,
            right,
            right_indices,
            alpha,
            beta,
            true,
        )
        .unwrap();
}

fn run(context: &Context<'_>) -> f64 {
    let mut a = tensor(context, vec![N, N], vec![SH, NS]);
    let mut b = tensor(context, vec![N, N], vec![SH, NS]);
    let mut c = tensor(context, vec![N, N], vec![SH, NS]);
    let mut c_answer = tensor(context, vec![N, N], vec![SH, NS]);
    let mut a_rep = tensor(context, vec![N; 3], vec![SY, SY, NS]);
    let mut b_rep = tensor(context, vec![N; 3], vec![SY, SY, NS]);
    let mut z = tensor(context, vec![N; 3], vec![SY, SY, NS]);
    let mut a_sum = tensor(context, vec![N], vec![NS]);
    let mut b_sum = tensor(context, vec![N], vec![NS]);
    let mut c_sum = tensor(context, vec![N], vec![NS]);

    a.transform(|key, value| *value = fixture(key, 173));
    b.transform(|key, value| *value = fixture(key, 349));

    contract(&mut c_answer, "ij", &a, "ik", &b, "kj", 1.0, 0.0);
    a_rep.sum_from("ijk", &a, "ij", 1.0, 1.0);
    b_rep.sum_from("ijk", &b, "ij", 1.0, 1.0);
    contract(&mut z, "ijk", &a_rep, "ijk", &b_rep, "ijk", 1.0, 1.0);
    c.sum_from("ij", &z, "ijk", 1.0, 1.0);
    contract(&mut c_sum, "i", &a, "ik", &b, "ik", 1.0, 1.0);
    a_sum.sum_from("i", &a, "ik", 1.0, 1.0);
    b_sum.sum_from("i", &b, "ik", 1.0, 1.0);
    contract(&mut c, "ij", &a, "ij", &b, "ij", -(N as f64), 1.0);
    c.sum_from("ij", &c_sum, "i", -1.0, 1.0);
    contract(&mut c, "ij", &a_sum, "i", &b, "ij", -1.0, 1.0);
    contract(&mut c, "ij", &a, "ij", &b_sum, "j", -1.0, 1.0);

    let mut difference = tensor(context, vec![N, N], vec![SY, NS]);
    difference.sum_from("ij", &c, "ij", 1.0, 1.0);
    difference.sum_from("ij", &c_answer, "ij", -1.0, 1.0);
    let norm = difference.norm2();
    assert!(
        norm.is_finite() && norm <= 1.0e-10,
        "source fast_sym norm={norm}"
    );
    norm
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let world_norm = run(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    let parity_norm = run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS upstream_fast_sym: n=13; source fast symmetric matrix product identity; world={world_norm:e}; parity={parity_norm:e}; norm<=1e-10"
        );
    }
    world.close();
    drop(universe);
}
