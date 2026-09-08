//! D5 exact checks for the value-facing interface replacements.
//!
//! The pinned Vector/Scalar constructors are represented by one- and
//! zero-order dense `Tensor`s.  This test exercises the only new algorithms
//! (`vector::arange` and scalar root-set/root-broadcast) and checks the
//! host `Universe`/`Context` world responsibilities on both communicators.

use ctf::{
    algebra::Arithmetic, context::Context, mapping::Distribution, scalar, tensor::Tensor, vector,
};

fn run(context: &Context<'_>) {
    let len = 7;
    let vector = vector::arange(context, 3_i64, len, 4_i64);
    assert_eq!(vector.distribution().shape, vec![len]);
    assert_eq!(
        vector.read(&(0..len).collect::<Vec<_>>()),
        (0..len)
            .map(|index| 3 + 4 * index as i64)
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        vector.reduce(),
        (0..len).map(|index| 3 + 4 * index as i64).sum()
    );

    // Direct Tensor construction is the native replacement for Vector's
    // inherited constructors and storage; its rank-one shape is explicit.
    let mut direct = Tensor::new(
        context,
        Distribution::cyclic(vec![len], context.size()),
        Arithmetic::<i64>::new(),
    );
    direct.transform(|index, value| *value = 3 + 4 * index as i64);
    assert_eq!(
        direct.read(&(0..len).collect::<Vec<_>>()),
        vector.read(&(0..len).collect::<Vec<_>>())
    );

    // Scalar uses a zero-order Tensor.  set_value updates only the canonical
    // root; value performs the source-equivalent broadcast to all replicas.
    let mut scalar_tensor = Tensor::new(
        context,
        Distribution::cyclic(Vec::new(), context.size()),
        Arithmetic::<i64>::new(),
    );
    scalar::set_value(&mut scalar_tensor, 17);
    assert_eq!(scalar::value(&scalar_tensor), 17);
    scalar::set_value(
        &mut scalar_tensor,
        if context.rank() == 0 { -9 } else { 12345 },
    );
    assert_eq!(scalar::value(&scalar_tensor), -9);

    // Universe/Context is the native replacement for World; this assertion is
    // intentionally kept in the D5 value test instead of adding a wrapper.
    assert!(context.rank() < context.size());
    context.barrier();
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    run(&world);

    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&parity);
    parity.close();

    world.barrier();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS d5_value_interfaces: arange and scalar root/broadcast; Tensor/Context replacements; world+parity"
        );
    }
    world.close();
    drop(universe);
}
