//! Native Rust port of pinned `examples/ao_mo_transf.cxx`.

use ctf::{
    algebra::{Arithmetic, Group, Semiring, Wire},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    random::Generator,
    scalar_conversion::CastFromF64,
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, AS, NS},
};

fn symmetric<'c, 'r, A>(
    context: &'c Context<'r>,
    shape: Vec<usize>,
    links: Vec<Symmetry>,
    algebra: A,
) -> SymmetricTensor<'c, 'r, A>
where
    A: Group,
{
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
        algebra,
    )
}

fn ao_mo<'c, 'r, A>(
    u: &SymmetricTensor<'c, 'r, A>,
    c: &SymmetricTensor<'c, 'r, A>,
    n: usize,
    m: usize,
) -> SymmetricTensor<'c, 'r, A>
where
    A: Group + Semiring + CastFromF64 + Clone,
    A::Element: Wire,
{
    let context = u.context();
    let mut u1 = symmetric(
        context,
        vec![m, n, n, n],
        vec![NS, NS, AS, NS],
        c.algebra().clone(),
    );
    u1.contract_from(
        "ajkl",
        u,
        "ijkl",
        c,
        "ia",
        c.algebra().one(),
        c.algebra().zero(),
        true,
    )
    .unwrap();

    let mut u2_ns = symmetric(
        context,
        vec![m, m, n, n],
        vec![NS, NS, AS, NS],
        c.algebra().clone(),
    );
    u2_ns
        .contract_from(
            "abkl",
            &u1,
            "ajkl",
            c,
            "jb",
            c.algebra().one(),
            c.algebra().zero(),
            true,
        )
        .unwrap();
    let mut u2 = symmetric(
        context,
        vec![m, m, n, n],
        vec![AS, NS, AS, NS],
        c.algebra().clone(),
    );
    u2.sum_from(
        "abkl",
        &u2_ns,
        "abkl",
        c.algebra().one(),
        c.algebra().zero(),
    );

    let mut u3 = symmetric(
        context,
        vec![m, m, m, n],
        vec![AS, NS, NS, NS],
        c.algebra().clone(),
    );
    u3.contract_from(
        "abcl",
        &u2,
        "abkl",
        c,
        "kc",
        c.algebra().one(),
        c.algebra().zero(),
        true,
    )
    .unwrap();

    let mut v_ns = symmetric(
        context,
        vec![m, m, m, m],
        vec![AS, NS, NS, NS],
        c.algebra().clone(),
    );
    v_ns.contract_from(
        "abcd",
        &u3,
        "abcl",
        c,
        "ld",
        c.algebra().one(),
        c.algebra().zero(),
        true,
    )
    .unwrap();
    let mut v = symmetric(
        context,
        vec![m, m, m, m],
        vec![AS, NS, AS, NS],
        c.algebra().clone(),
    );
    v.sum_from("abcd", &v_ns, "abcd", c.algebra().one(), c.algebra().zero());
    v
}

fn run(context: &Context<'_>) {
    const N: usize = 6;
    const M: usize = 9;
    let mut generator = Generator::new(context.rank() as u64);
    let mut u = symmetric(
        context,
        vec![N, N, N, N],
        vec![AS, NS, AS, NS],
        Arithmetic::<f64>::new(),
    );
    u.fill_random(-1.0, 1.0, &mut generator);
    let mut c = symmetric(context, vec![N, M], vec![NS, NS], Arithmetic::<f64>::new());
    c.fill_random(-1.0, 1.0, &mut generator);
    let v = ao_mo(&u, &c, N, M);

    let mut u_f32 = symmetric(
        context,
        vec![N, N, N, N],
        vec![AS, NS, AS, NS],
        Arithmetic::<f32>::new(),
    );
    u_f32.set_local_canonical_storage(
        &u.local_canonical_storage()
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>(),
    );
    let mut c_f32 = symmetric(context, vec![N, M], vec![NS, NS], Arithmetic::<f32>::new());
    c_f32.set_local_canonical_storage(
        &c.local_canonical_storage()
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>(),
    );
    let v_f32 = ao_mo(&u_f32, &c_f32, N, M);
    let mut v_f32_as_f64 = symmetric(
        context,
        vec![M, M, M, M],
        vec![AS, NS, AS, NS],
        Arithmetic::<f64>::new(),
    );
    v_f32_as_f64.set_local_canonical_storage(
        &v_f32
            .local_canonical_storage()
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>(),
    );
    let mut float_error = symmetric(
        context,
        vec![M, M, M, M],
        vec![AS, NS, AS, NS],
        Arithmetic::<f64>::new(),
    );
    float_error.sum_from("abcd", &v, "abcd", 1.0, 0.0);
    float_error.sum_from("abcd", &v_f32_as_f64, "abcd", -1.0, 1.0);
    let _float_vs_double_norm = float_error.norm2();

    let u_ns = u.unpack(Distribution::cyclic(vec![N, N, N, N], context.size()));
    let c_ns = c.unpack(Distribution::cyclic(vec![N, M], context.size()));
    let topology = Topology::new(vec![context.size()]);
    let mut u1 = ctf::tensor::Tensor::new(
        context,
        Distribution::cyclic(vec![M, N, N, N], context.size()),
        Arithmetic::<f64>::new(),
    );
    u1.contract_from(
        "ajkl",
        &u_ns,
        "ijkl",
        &c_ns,
        "ia",
        topology.clone(),
        1.0,
        0.0,
    )
    .unwrap();
    let mut u2 = ctf::tensor::Tensor::new(
        context,
        Distribution::cyclic(vec![M, M, N, N], context.size()),
        Arithmetic::<f64>::new(),
    );
    u2.contract_from("abkl", &u1, "ajkl", &c_ns, "jb", topology.clone(), 1.0, 0.0)
        .unwrap();
    let mut u3 = ctf::tensor::Tensor::new(
        context,
        Distribution::cyclic(vec![M, M, M, N], context.size()),
        Arithmetic::<f64>::new(),
    );
    u3.contract_from("abcl", &u2, "abkl", &c_ns, "kc", topology.clone(), 1.0, 0.0)
        .unwrap();
    let mut v_ns = ctf::tensor::Tensor::new(
        context,
        Distribution::cyclic(vec![M, M, M, M], context.size()),
        Arithmetic::<f64>::new(),
    );
    v_ns.contract_from("abcd", &u3, "abcl", &c_ns, "ld", topology.clone(), 1.0, 0.0)
        .unwrap();
    let v_dense = v.unpack(Distribution::cyclic(vec![M, M, M, M], context.size()));
    v_ns.sum_from("abcd", &v_dense, "abcd", topology, -1.0, 1.0)
        .unwrap();
    let _nonsymmetric_vs_antisymmetric_norm = v_ns.norm2();
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
            "DIGIT / PASS upstream_ao_mo_transf: pinned n=6 m=9 double, float and nonsymmetric AO-MO paths completed; source has no numerical gate; world+parity"
        );
    }
    world.close();
    drop(universe);
}
