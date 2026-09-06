// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
// Adapted from pinned CTF examples/sparse_mp3.cxx, dense-T path.
//
// The upstream fixture seeds drand48 from the rank.  This port uses keyed
// global fixtures instead, so the world and parity runs have identical
// tensors while retaining the source's Ea/Ei and integral value intervals.
use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Topology},
    sparse::SparseTensor,
    tensor::Tensor,
};

type F64 = Arithmetic<f64>;

const NV: usize = 3;
const NO: usize = 2;
const SPARSITY: f64 = 0.8;

fn fixture_unit(key: usize, stream: u64) -> f64 {
    let mut state = (key as u64)
        .wrapping_add(0x9e37_79b9_7f4a_7c15u64.wrapping_mul(stream + 1));
    state ^= state >> 30;
    state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    state ^= state >> 27;
    state = state.wrapping_mul(0x94d0_49bb_1331_11ebu64);
    state ^= state >> 31;
    (state >> 11) as f64 / (1u64 << 53) as f64
}

fn fixture_value(key: usize, stream: u64, low: f64, high: f64) -> f64 {
    low + (high - low) * fixture_unit(key, stream)
}

fn dense_fixture<'c, 'r>(
    context: &'c Context<'r>,
    shape: Vec<usize>,
    stream: u64,
    low: f64,
    high: f64,
) -> Tensor<'c, 'r, F64> {
    let mut tensor = Tensor::new(
        context,
        Distribution::cyclic(shape, context.size()),
        F64::new(),
    );
    tensor.transform(|key, value| *value = fixture_value(key, stream, low, high));
    tensor
}

fn integral_fixture<'c, 'r>(
    context: &'c Context<'r>,
    shape: Vec<usize>,
    stream: u64,
) -> Tensor<'c, 'r, F64> {
    let mut tensor = dense_fixture(context, shape, stream, -1.0, 1.0);
    tensor.transform(|_, value| {
        if value.abs() < SPARSITY {
            *value = 0.0;
        }
    });
    tensor
}

fn sparse_copy<'c, 'r>(dense: &Tensor<'c, 'r, F64>) -> SparseTensor<'c, 'r, F64> {
    let mut sparse = SparseTensor::new(dense.context(), dense.distribution().clone(), F64::new());
    let pairs: Vec<_> = dense
        .local_pairs()
        .into_iter()
        .filter(|(_, value)| *value != 0.0)
        .collect();
    sparse.write_add(&pairs);
    sparse
}

fn divide_ea_ei<'c, 'r>(
    ea: &Tensor<'c, 'r, F64>,
    ei: &Tensor<'c, 'r, F64>,
    tensor: &mut Tensor<'c, 'r, F64>,
    topology: &Topology,
) {
    let distribution = tensor.distribution().clone();
    let mut denominator = Tensor::new(tensor.context(), distribution.clone(), F64::new());
    denominator
        .sum_from("abij", ei, "i", topology.clone(), 1.0, 0.0)
        .unwrap();
    denominator
        .sum_from("abij", ei, "j", topology.clone(), 1.0, 1.0)
        .unwrap();
    denominator
        .sum_from("abij", ea, "a", topology.clone(), -1.0, 1.0)
        .unwrap();
    denominator
        .sum_from("abij", ea, "b", topology.clone(), -1.0, 1.0)
        .unwrap();
    denominator.transform(|_, value| *value = 1.0 / *value);

    // T and D have the same distribution.  Read only T's local keys and
    // rebuild T through its collective write path; no full tensor gather.
    let local = tensor.local_pairs();
    let keys: Vec<_> = local.iter().map(|(key, _)| *key).collect();
    let local_denominator = denominator.read(&keys);
    let scaled: Vec<_> = local
        .into_iter()
        .zip(local_denominator)
        .map(|((key, value), denominator)| (key, value * denominator))
        .collect();
    let mut result = Tensor::new(tensor.context(), distribution, F64::new());
    result.write_add(&scaled);
    *tensor = result;
}

fn mp3_dense<'c, 'r>(
    ea: &Tensor<'c, 'r, F64>,
    ei: &Tensor<'c, 'r, F64>,
    fab: &Tensor<'c, 'r, F64>,
    fij: &Tensor<'c, 'r, F64>,
    vabij: &Tensor<'c, 'r, F64>,
    vijab: &Tensor<'c, 'r, F64>,
    vabcd: &Tensor<'c, 'r, F64>,
    vijkl: &Tensor<'c, 'r, F64>,
    vaibj: &Tensor<'c, 'r, F64>,
    topology: &Topology,
) -> f64 {
    let mut t = Tensor::new(ea.context(), vabij.distribution().clone(), F64::new());
    t.sum_from("abij", vabij, "abij", topology.clone(), 1.0, 0.0)
        .unwrap();
    divide_ea_ei(ea, ei, &mut t, topology);

    let mut z = Tensor::new(ea.context(), vabij.distribution().clone(), F64::new());
    z.sum_from("abij", vijab, "ijab", topology.clone(), 1.0, 0.0)
        .unwrap();
    z.contract_from("abij", fab, "af", &t, "fbij", topology.clone(), 1.0, 1.0)
        .unwrap();
    z.contract_from("abij", fij, "ni", &t, "abnj", topology.clone(), -1.0, 1.0)
        .unwrap();
    z.contract_from("abij", vabcd, "abef", &t, "efij", topology.clone(), 0.5, 1.0)
        .unwrap();
    z.contract_from("abij", vijkl, "mnij", &t, "abmn", topology.clone(), 0.5, 1.0)
        .unwrap();
    z.contract_from("abij", vaibj, "amei", &t, "ebmj", topology.clone(), 1.0, 1.0)
        .unwrap();
    divide_ea_ei(ea, ei, &mut z, topology);

    let mut energy = Tensor::new(ea.context(), Distribution::cyclic(vec![], ea.context().size()), F64::new());
    energy
        .contract_from("", vabij, "abij", &z, "abij", topology.clone(), 1.0, 0.0)
        .unwrap();
    let value = energy.reduce();
    assert!(value.is_finite());
    value
}

fn mp3_sparse_dense<'c, 'r>(
    ea: &Tensor<'c, 'r, F64>,
    ei: &Tensor<'c, 'r, F64>,
    fab: &Tensor<'c, 'r, F64>,
    fij: &Tensor<'c, 'r, F64>,
    vabij: &SparseTensor<'c, 'r, F64>,
    vijab: &SparseTensor<'c, 'r, F64>,
    vabcd: &SparseTensor<'c, 'r, F64>,
    vijkl: &SparseTensor<'c, 'r, F64>,
    vaibj: &SparseTensor<'c, 'r, F64>,
    grid: [usize; 2],
    topology: &Topology,
) -> f64 {
    let mut t = Tensor::new(ea.context(), Distribution::cyclic(vec![NV, NV, NO, NO], ea.context().size()), F64::new());
    t.sum_from_sparse("abij", vabij, "abij", 1.0, 0.0);
    divide_ea_ei(ea, ei, &mut t, topology);

    let mut z = Tensor::new(ea.context(), Distribution::cyclic(vec![NV, NV, NO, NO], ea.context().size()), F64::new());
    z.sum_from_sparse("abij", vijab, "ijab", 1.0, 0.0);
    z.contract_from("abij", fab, "af", &t, "fbij", topology.clone(), 1.0, 1.0)
        .unwrap();
    z.contract_from("abij", fij, "ni", &t, "abnj", topology.clone(), -1.0, 1.0)
        .unwrap();
    z.contract_from_sparse_dense("abij", vabcd, "abef", &t, "efij", grid, 0.5, 1.0)
        .unwrap();
    z.contract_from_sparse_dense("abij", vijkl, "mnij", &t, "abmn", grid, 0.5, 1.0)
        .unwrap();
    z.contract_from_sparse_dense("abij", vaibj, "amei", &t, "ebmj", grid, 1.0, 1.0)
        .unwrap();
    divide_ea_ei(ea, ei, &mut z, topology);

    let mut energy = Tensor::new(ea.context(), Distribution::cyclic(vec![], ea.context().size()), F64::new());
    energy
        .contract_from_sparse_dense("", vabij, "abij", &z, "abij", grid, 1.0, 0.0)
        .unwrap();
    let value = energy.reduce();
    assert!(value.is_finite());
    value
}

fn run(context: &Context<'_>) -> (f64, f64, f64) {
    let topology = Topology::new(if context.size() == 4 {
        vec![2, 2]
    } else {
        vec![context.size(), 1]
    });
    let grid = if context.size() == 4 {
        [2, 2]
    } else {
        [context.size(), 1]
    };

    let ea = dense_fixture(context, vec![NV], 0, 9.0, 18.0);
    let ei = dense_fixture(context, vec![NO], 1, -8.0, -4.0);
    let fab = dense_fixture(context, vec![NV, NV], 2, -1.0, 1.0);
    let fij = dense_fixture(context, vec![NO, NO], 3, -1.0, 1.0);
    let vabij = integral_fixture(context, vec![NV, NV, NO, NO], 4);
    let vijab = integral_fixture(context, vec![NO, NO, NV, NV], 5);
    let vabcd = integral_fixture(context, vec![NV, NV, NV, NV], 6);
    let vijkl = integral_fixture(context, vec![NO, NO, NO, NO], 7);
    let vaibj = integral_fixture(context, vec![NV, NO, NV, NO], 8);

    let dense_energy = mp3_dense(
        &ea, &ei, &fab, &fij, &vabij, &vijab, &vabcd, &vijkl, &vaibj, &topology,
    );

    // Source-style sparse integral copies: all five integral tensors are
    // thresholded at sp=.8 and retain only canonical nonzero local pairs.
    let sparse_vabij = sparse_copy(&vabij);
    let sparse_vijab = sparse_copy(&vijab);
    let sparse_vabcd = sparse_copy(&vabcd);
    let sparse_vijkl = sparse_copy(&vijkl);
    let sparse_vaibj = sparse_copy(&vaibj);

    let sparse_energy = mp3_sparse_dense(
        &ea,
        &ei,
        &fab,
        &fij,
        &sparse_vabij,
        &sparse_vijab,
        &sparse_vabcd,
        &sparse_vijkl,
        &sparse_vaibj,
        grid,
        &topology,
    );

    assert!(dense_energy.is_finite());
    assert_ne!(dense_energy, 0.0);
    assert!(sparse_energy.is_finite());
    let relative = ((dense_energy - sparse_energy) / dense_energy).abs();
    assert!(
        relative < 1.0e-6,
        "dense/sparse MP3 mismatch: dense={dense_energy:e} sparse={sparse_energy:e} relative={relative:e}"
    );
    (dense_energy, sparse_energy, relative)
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let world_rank = world.rank();
    let world_result = run(&world);

    let parity = world
        .split(Some((world_rank % 2) as i32), world_rank as i32)
        .unwrap();
    let parity_result = run(&parity);
    let parity_size = parity.size();
    parity.close();

    if world_rank == 0 {
        println!(
            "DIGIT / PASS upstream sparse_mp3 dense-T sparse*dense: world_ranks={} dense_energy={:e} sparse_energy={:e} relative={:e}",
            world.size(), world_result.0, world_result.1, world_result.2
        );
        println!(
            "DIGIT / PASS upstream sparse_mp3 dense-T sparse*dense: parity_ranks={} dense_energy={:e} sparse_energy={:e} relative={:e}",
            parity_size, parity_result.0, parity_result.1, parity_result.2
        );
    }
    world.close();
    runtime.finalize();
}
