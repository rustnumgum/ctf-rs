use ctf::{common, context::Context};

fn labels_and_column_major_indices_are_exact() {
    assert_eq!(common::default_indices(0, 0), b"");
    assert_eq!(common::default_indices(4, 0), b"abcd");
    assert_eq!(common::default_indices(4, 2), b"cdef");

    let lengths = [2, 3, 4];
    assert_eq!(common::decode_index(&lengths, 0), [0, 0, 0]);
    assert_eq!(common::decode_index(&lengths, 17), [1, 2, 2]);
    assert_eq!(common::decode_index(&lengths, 23), [1, 2, 3]);
    for index in 0..24 {
        let coordinates = common::decode_index(&lengths, index);
        assert_eq!(common::encode_index(&lengths, &coordinates), index);
    }
}

fn source_scan_recurrences_are_exact() {
    let mut inclusive = [1_i64, 9, 2, 9, 3, 9, 4, 9];
    common::parallel_postfix(7, 2, &mut inclusive);
    assert_eq!(inclusive, [1, 9, 3, 9, 6, 9, 10, 9]);

    let mut exclusive = [1_i64, 9, 2, 9, 3, 9, 4, 9];
    common::parallel_prefix(7, 2, &mut exclusive);
    assert_eq!(exclusive, [0, 9, 1, 9, 3, 9, 6, 9]);

    let mut one = [7_i32];
    common::parallel_postfix(1, 1, &mut one);
    assert_eq!(one, [7]);
    common::parallel_prefix(1, 1, &mut one);
    assert_eq!(one, [0]);

    assert_eq!(common::prefix(&[3_i32, 1, 4, 1, 5]), [0, 3, 4, 8, 9]);
}

fn run(_context: &Context<'_>) {
    labels_and_column_major_indices_are_exact();
    source_scan_recurrences_are_exact();
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
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS d5_common: default labels, column-major index conversion and scan recurrences exact; world+parity"
        );
    }
    world.close();
    drop(universe);
}
