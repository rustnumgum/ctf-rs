//! Native Rust port of the active iteration in pinned `examples/ccsd.cxx`.

use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, AS, NS, SH},
};

type Symmetric<'c, 'r> = SymmetricTensor<'c, 'r, Arithmetic<f64>>;

fn tensor<'c, 'r>(
    context: &'c Context<'r>,
    shape: Vec<usize>,
    links: Vec<Symmetry>,
) -> Symmetric<'c, 'r> {
    let topology = Topology::new(vec![context.size()]);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    if !mappings.is_empty() {
        mappings[0].augment_physical(&topology, 0);
        for mapping in &mut mappings {
            mapping.augment_virtual(context.size());
        }
    }
    SymmetricTensor::new(
        context,
        SymmetricDistribution::new(Distribution::new(shape, topology, mappings), links),
        Arithmetic::new(),
    )
}

fn zero_like<'c, 'r>(source: &Symmetric<'c, 'r>) -> Symmetric<'c, 'r> {
    SymmetricTensor::new(
        source.context(),
        source.distribution().clone(),
        Arithmetic::new(),
    )
}

struct Integrals<'c, 'r> {
    aa: Symmetric<'c, 'r>,
    ii: Symmetric<'c, 'r>,
    ab: Symmetric<'c, 'r>,
    ai: Symmetric<'c, 'r>,
    ia: Symmetric<'c, 'r>,
    ij: Symmetric<'c, 'r>,
    abcd: Symmetric<'c, 'r>,
    abci: Symmetric<'c, 'r>,
    aibc: Symmetric<'c, 'r>,
    aibj: Symmetric<'c, 'r>,
    abij: Symmetric<'c, 'r>,
    ijab: Symmetric<'c, 'r>,
    aijk: Symmetric<'c, 'r>,
    ijak: Symmetric<'c, 'r>,
    ijkl: Symmetric<'c, 'r>,
}

impl<'c, 'r> Integrals<'c, 'r> {
    fn new(context: &'c Context<'r>, no: usize, nv: usize) -> Self {
        let mut values = Self {
            aa: tensor(context, vec![nv], vec![NS]),
            ii: tensor(context, vec![no], vec![NS]),
            ab: tensor(context, vec![nv, nv], vec![AS, NS]),
            ai: tensor(context, vec![nv, no], vec![NS, NS]),
            ia: tensor(context, vec![no, nv], vec![NS, NS]),
            ij: tensor(context, vec![no, no], vec![AS, NS]),
            abcd: tensor(context, vec![nv, nv, nv, nv], vec![AS, NS, AS, NS]),
            abci: tensor(context, vec![nv, nv, nv, no], vec![AS, NS, NS, NS]),
            aibc: tensor(context, vec![nv, no, nv, nv], vec![NS, NS, AS, NS]),
            aibj: tensor(context, vec![nv, no, nv, no], vec![NS, NS, NS, NS]),
            abij: tensor(context, vec![nv, nv, no, no], vec![AS, NS, AS, NS]),
            ijab: tensor(context, vec![no, no, nv, nv], vec![AS, NS, AS, NS]),
            aijk: tensor(context, vec![nv, no, no, no], vec![NS, NS, AS, NS]),
            ijak: tensor(context, vec![no, no, nv, no], vec![AS, NS, NS, NS]),
            ijkl: tensor(context, vec![no, no, no, no], vec![AS, NS, AS, NS]),
        };
        for (ordinal, value) in [
            &mut values.aa,
            &mut values.ii,
            &mut values.ab,
            &mut values.ai,
            &mut values.ia,
            &mut values.ij,
            &mut values.abcd,
            &mut values.abci,
            &mut values.aibc,
            &mut values.aibj,
            &mut values.abij,
            &mut values.ijab,
            &mut values.aijk,
            &mut values.ijak,
            &mut values.ijkl,
        ]
        .into_iter()
        .enumerate()
        {
            value.transform(|key, element| {
                *element = ((key * 16 + ordinal) % 13077) as f64 / 13077.0 - 0.5;
            });
        }
        values
    }
}

struct Amplitudes<'c, 'r> {
    ai: Symmetric<'c, 'r>,
    abij: Symmetric<'c, 'r>,
}

impl<'c, 'r> Amplitudes<'c, 'r> {
    fn new(context: &'c Context<'r>, no: usize, nv: usize) -> Self {
        let mut values = Self {
            ai: tensor(context, vec![nv, no], vec![NS, NS]),
            abij: tensor(context, vec![nv, nv, no, no], vec![AS, NS, AS, NS]),
        };
        for (ordinal, value) in [&mut values.ai, &mut values.abij].into_iter().enumerate() {
            value.transform(|key, element| {
                *element = ((key * 13 + ordinal) % 13077) as f64 / 13077.0 - 0.5;
            });
        }
        values
    }
}

fn ccsd<'c, 'r>(v: &Integrals<'c, 'r>, t: &mut Amplitudes<'c, 'r>) {
    let mut t21 = zero_like(&t.abij);
    t21.set_local_canonical_storage(&t.abij.local_canonical_storage());
    t21.contract_from("abij", &t.ai, "ai", &t.ai, "bj", 0.5, 1.0, true)
        .unwrap();

    let mut fme = zero_like(&v.ia);
    fme.sum_from("me", &v.ia, "me", 1.0, 0.0);
    fme.contract_from("me", &v.ijab, "mnef", &t.ai, "fn", 1.0, 1.0, true)
        .unwrap();

    let mut fae = zero_like(&v.ab);
    fae.sum_from("ae", &v.ab, "ae", 1.0, 0.0);
    fae.contract_from("ae", &fme, "me", &t.ai, "am", -1.0, 1.0, true)
        .unwrap();
    fae.contract_from("ae", &v.ijab, "mnef", &t.abij, "afmn", -0.5, 1.0, true)
        .unwrap();
    fae.contract_from("ae", &v.aibc, "anef", &t.ai, "fn", 1.0, 1.0, true)
        .unwrap();

    let mut fmi = zero_like(&v.ij);
    fmi.sum_from("mi", &v.ij, "mi", 1.0, 0.0);
    fmi.contract_from("mi", &fme, "me", &t.ai, "ei", 1.0, 1.0, true)
        .unwrap();
    fmi.contract_from("mi", &v.ijab, "mnef", &t.abij, "efin", 0.5, 1.0, true)
        .unwrap();
    fmi.contract_from("mi", &v.ijak, "mnfi", &t.ai, "fn", 1.0, 1.0, true)
        .unwrap();

    let mut wmnei = zero_like(&v.ijak);
    wmnei.sum_from("mnei", &v.ijak, "mnei", 1.0, 0.0);
    wmnei.sum_from("mnei", &v.ijak, "mnei", 1.0, 1.0);
    wmnei
        .contract_from("mnei", &v.ijab, "mnef", &t.ai, "fi", 1.0, 1.0, true)
        .unwrap();

    let mut wmnij = zero_like(&v.ijkl);
    wmnij.sum_from("mnij", &v.ijkl, "mnij", 1.0, 0.0);
    wmnij
        .contract_from("mnij", &v.ijak, "mnei", &t.ai, "ej", -1.0, 1.0, true)
        .unwrap();
    wmnij
        .contract_from("mnij", &v.ijab, "mnef", &t21, "efij", 1.0, 1.0, true)
        .unwrap();

    let mut wamei = zero_like(&v.aibj);
    wamei.sum_from("amei", &v.aibj, "amei", 1.0, 0.0);
    wamei
        .contract_from("amei", &wmnei, "mnei", &t.ai, "an", -1.0, 1.0, true)
        .unwrap();
    wamei
        .contract_from("amei", &v.aibc, "amef", &t.ai, "fi", 1.0, 1.0, true)
        .unwrap();
    wamei
        .contract_from("amei", &v.ijab, "mnef", &t.abij, "afin", 0.5, 1.0, true)
        .unwrap();

    let mut wamij = zero_like(&v.aijk);
    wamij.sum_from("amij", &v.aijk, "amij", 1.0, 0.0);
    wamij
        .contract_from("amij", &v.aibj, "amei", &t.ai, "ej", 1.0, 1.0, true)
        .unwrap();
    wamij
        .contract_from("amij", &v.aibc, "amef", &t.abij, "efij", 1.0, 1.0, true)
        .unwrap();

    let mut zai = zero_like(&v.ai);
    zai.sum_from("ai", &v.ai, "ai", 1.0, 0.0);
    zai.contract_from("ai", &fmi, "mi", &t.ai, "am", -1.0, 1.0, true)
        .unwrap();
    zai.contract_from("ai", &v.ab, "ae", &t.ai, "ei", 1.0, 1.0, true)
        .unwrap();
    zai.contract_from("ai", &v.aibj, "amei", &t.ai, "em", 1.0, 1.0, true)
        .unwrap();
    zai.contract_from("ai", &v.abij, "aeim", &fme, "me", 1.0, 1.0, true)
        .unwrap();
    zai.contract_from("ai", &v.aibc, "amef", &t21, "efim", 0.5, 1.0, true)
        .unwrap();
    zai.contract_from("ai", &wmnei, "mnei", &t21, "eamn", -0.5, 1.0, true)
        .unwrap();

    let mut zabij = zero_like(&v.abij);
    zabij.sum_from("abij", &v.abij, "abij", 1.0, 0.0);
    zabij
        .contract_from("abij", &v.abci, "abei", &t.ai, "ej", 1.0, 1.0, true)
        .unwrap();
    zabij
        .contract_from("abij", &wamei, "amei", &t.abij, "ebmj", 1.0, 1.0, true)
        .unwrap();
    zabij
        .contract_from("abij", &wamij, "amij", &t.ai, "bm", -1.0, 1.0, true)
        .unwrap();
    zabij
        .contract_from("abij", &fae, "ae", &t.abij, "ebij", 1.0, 1.0, true)
        .unwrap();
    zabij
        .contract_from("abij", &fmi, "mi", &t.abij, "abmj", -1.0, 1.0, true)
        .unwrap();
    zabij
        .contract_from("abij", &v.abcd, "abef", &t21, "efij", 0.5, 1.0, true)
        .unwrap();
    zabij
        .contract_from("abij", &wmnij, "mnij", &t21, "abmn", 0.5, 1.0, true)
        .unwrap();

    let context = v.ai.context();
    let no = v.ii.distribution().distribution().shape[0];
    let nv = v.aa.distribution().distribution().shape[0];
    let mut dai = tensor(context, vec![nv, no], vec![NS, NS]);
    dai.sum_from("ai", &v.ii, "i", 1.0, 0.0);
    dai.sum_from("ai", &v.aa, "a", -1.0, 1.0);
    let mut dabij = tensor(context, vec![nv, nv, no, no], vec![SH, NS, SH, NS]);
    dabij.sum_from("abij", &v.ii, "i", 1.0, 0.0);
    dabij.sum_from("abij", &v.ii, "j", 1.0, 1.0);
    dabij.sum_from("abij", &v.aa, "a", -1.0, 1.0);
    dabij.sum_from("abij", &v.aa, "b", -1.0, 1.0);

    t.ai.contract_function_from(
        "ai",
        &zai,
        "ai",
        &dai,
        "ai",
        1.0,
        0.0,
        true,
        |numerator, denominator| numerator / denominator,
    )
    .unwrap();
    t.abij
        .contract_function_from(
            "abij",
            &zabij,
            "abij",
            &dabij,
            "abij",
            1.0,
            0.0,
            true,
            |numerator, denominator| numerator / denominator,
        )
        .unwrap();
}

fn run(context: &Context<'_>) {
    const NO: usize = 4;
    const NV: usize = 6;
    let integrals = Integrals::new(context, NO, NV);
    let mut amplitudes = Amplitudes::new(context, NO, NV);
    ccsd(&integrals, &mut amplitudes);
    let singles_norm = amplitudes.ai.norm2();
    let doubles_norm = amplitudes.abij.norm2();
    amplitudes.ai.scale(&(1.0 / singles_norm));
    amplitudes.abij.scale(&(1.0 / doubles_norm));
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world);
    let rank = world.rank();
    let parity = world.split(Some((rank % 2) as i32), rank as i32).unwrap();
    run(&parity);
    parity.close();
    if rank == 0 {
        println!(
            "DIGIT / PASS upstream_ccsd: pinned no=4 nv=6 deterministic integral/amplitude fixture and full iteration completed; source has no numerical gate; world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
