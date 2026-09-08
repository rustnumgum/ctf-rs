//! Native Rust port of pinned `examples/neural_network.cxx`.

use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Topology},
    random::Generator,
    tensor::Tensor,
};

type Dense<'c, 'r> = Tensor<'c, 'r, Arithmetic<f64>>;

fn dense<'c, 'r>(context: &'c Context<'r>, shape: Vec<usize>) -> Dense<'c, 'r> {
    Tensor::new(
        context,
        Distribution::cyclic(shape, context.size()),
        Arithmetic::new(),
    )
}

fn rotation<'c, 'r>(context: &'c Context<'r>, n: usize, shift: usize) -> Dense<'c, 'r> {
    let mut matrix = dense(context, vec![n, n]);
    let pairs = if context.rank() == 0 {
        (0..n)
            .map(|i| (((i + shift) % n) + i * n, 1.0))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    matrix.write_add(&pairs);
    matrix
}

fn run(context: &Context<'_>) {
    const N: usize = 16;
    const M: usize = 9;
    const D: usize = 4;
    const SPARSITY: f64 = 0.2;
    let topology = Topology::new(vec![context.size()]);
    let mut generator = Generator::new(context.rank() as u64);

    let mut x = dense(context, vec![N, N, M]);
    x.fill_random_sparse(0.0, 1.0, SPARSITY, &mut generator);
    let mut w = dense(context, vec![D, D, M]);
    w.fill_random(0.0, 1.0, &mut generator);

    let x5 = x.reshape(Distribution::cyclic(
        vec![N / D, D, N / D, D, M],
        context.size(),
    ));
    let y_shape = vec![N / D, D, N / D, D];
    let mut y4 = dense(context, y_shape.clone());
    let rd = rotation(context, D, 1);
    let rn4 =
        rotation(context, N, 1).reshape(Distribution::cyclic(y_shape.clone(), context.size()));
    let rnd4 =
        rotation(context, N, D).reshape(Distribution::cyclic(y_shape.clone(), context.size()));

    for _a in 0..D {
        for _b in 0..D {
            y4.contract_from("iajb", &x5, "iajbk", &w, "abk", topology.clone(), 1.0, 1.0)
                .unwrap();

            let old_w = w.clone();
            w.contract_from("abk", &rd, "bc", &old_w, "ack", topology.clone(), 1.0, 0.0)
                .unwrap();
            let old_y = y4.clone();
            y4.contract_from(
                "iajb",
                &rn4,
                "jbkc",
                &old_y,
                "iakc",
                topology.clone(),
                1.0,
                0.0,
            )
            .unwrap();
        }

        let old_y = y4.clone();
        y4.contract_from(
            "iajb",
            &rnd4,
            "kcjb",
            &old_y,
            "iakc",
            topology.clone(),
            1.0,
            0.0,
        )
        .unwrap();
        let old_w = w.clone();
        w.contract_from("abk", &rd, "ac", &old_w, "cbk", topology.clone(), 1.0, 0.0)
            .unwrap();
        let old_y = y4.clone();
        y4.contract_from(
            "iajb",
            &rn4,
            "iakc",
            &old_y,
            "kcjb",
            topology.clone(),
            1.0,
            0.0,
        )
        .unwrap();
    }
    let old_y = y4.clone();
    y4.contract_from("iajb", &rnd4, "kcia", &old_y, "kcjb", topology, 1.0, 0.0)
        .unwrap();

    let y = y4.reshape(Distribution::cyclic(vec![N, N], context.size()));
    assert!(y.norm2() >= 1.0e-6);
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
            "DIGIT / PASS upstream_neural_network: pinned n=16 m=9 d=4 sparsity=0.2 rotation/convolution chain; output norm2>=1e-6; world+parity"
        );
    }
    world.close();
    drop(universe);
}
