//! Native Rust port of pinned `examples/qinformatics.cxx`.

use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Topology},
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

// One drand48 draw immediately after the source's srand48(seed).
fn seeded_value(seed: usize) -> f64 {
    const MASK: u64 = (1_u64 << 48) - 1;
    let state = (((seed as u64) & 0xffff_ffff) << 16) | 0x330e;
    let next = (0x5deece66d_u64.wrapping_mul(state).wrapping_add(0xb)) & MASK;
    next as f64 / (1_u64 << 48) as f64
}

fn run(context: &Context<'_>) {
    const D: usize = 2;
    const L: usize = 15;
    let topology = Topology::new(vec![context.size()]);

    let mut folded_h = dense(context, vec![D * D, D * D]);
    folded_h.transform(|key, value| *value = seeded_value(key * 23));
    let mut folded_in = dense(context, {
        let mut shape = vec![D * D; L / 2];
        shape.push(D);
        shape
    });
    folded_in.transform(|key, value| *value = seeded_value(key));
    let mut folded_out = dense(context, folded_in.distribution().shape.clone());

    for (position, input_labels) in [
        "xcegikmo", "axegikmo", "acxgikmo", "acexikmo", "acegxkmo", "acegixmo", "acegikxo",
    ]
    .into_iter()
    .enumerate()
    {
        let h_labels = ["ax", "cx", "ex", "gx", "ix", "kx", "mx"][position];
        folded_out
            .contract_from(
                "acegikmo",
                &folded_h,
                h_labels,
                &folded_in,
                input_labels,
                topology.clone(),
                1.0,
                if position == 0 { 0.0 } else { 1.0 },
            )
            .unwrap();
    }

    let second_shape = {
        let mut shape = vec![D];
        shape.extend(std::iter::repeat_n(D * D, L / 2));
        shape
    };
    let second_in = folded_in.reshape(Distribution::cyclic(second_shape.clone(), context.size()));
    let mut second_out = folded_out.reshape(Distribution::cyclic(second_shape, context.size()));
    for (h_labels, input_labels) in [
        ("bx", "axdfhjln"),
        ("dx", "abxfhjln"),
        ("fx", "abdxhjln"),
        ("hx", "abdfxjln"),
        ("jx", "abdfhxln"),
        ("lx", "abdfhjxn"),
        ("nx", "abdfhjlx"),
    ] {
        second_out
            .contract_from(
                "abdfhjln",
                &folded_h,
                h_labels,
                &second_in,
                input_labels,
                topology.clone(),
                1.0,
                1.0,
            )
            .unwrap();
    }
    let _folded_norm = second_out.norm2();

    let mut input = dense(context, vec![D; L]);
    input.transform(|key, value| *value = seeded_value(key));
    let mut h = dense(context, vec![D; 4]);
    h.transform(|key, value| *value = seeded_value(key * 23));
    let mut output = dense(context, vec![D; L]);
    for (position, (h_labels, input_labels)) in [
        ("abxy", "xycdefghijklmno"),
        ("bcxy", "axydefghijklmno"),
        ("cdxy", "abxyefghijklmno"),
        ("dexy", "abcxyfghijklmno"),
        ("efxy", "abcdxyghijklmno"),
        ("fgxy", "abcdexyhijklmno"),
        ("ghxy", "abcdefxyijklmno"),
        ("hixy", "abcdefgxyjklmno"),
        ("ijxy", "abcdefghxyklmno"),
        ("jkxy", "abcdefghixylmno"),
        ("klxy", "abcdefghijxymno"),
        ("lmxy", "abcdefghijkxyno"),
        ("mnxy", "abcdefghijklxyo"),
        ("noxy", "abcdefghijklmxy"),
    ]
    .into_iter()
    .enumerate()
    {
        output
            .contract_from(
                "abcdefghijklmno",
                &h,
                h_labels,
                &input,
                input_labels,
                topology.clone(),
                1.0,
                if position == 0 { 0.0 } else { 1.0 },
            )
            .unwrap();
    }
    let _unfolded_norm = output.norm2();
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
            "DIGIT / PASS upstream_qinformatics: pinned d=2 L=15 folded and unfolded contraction chains completed; source has no numerical gate; world+parity"
        );
    }
    world.close();
    drop(universe);
}
