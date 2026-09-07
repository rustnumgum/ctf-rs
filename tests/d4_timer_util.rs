use ctf::{
    context::{Context, Runtime},
    int_timer::{FunctionTimer, Timer, TimerRegistry},
    symmetry::Symmetry::{AS, NS, SY},
    util,
};

fn timer_accumulates_monotonic_wall_snapshots() {
    let mut timer = Timer::new("wall");
    assert_eq!(timer.snapshot().calls, 0);
    timer.start();
    let first = timer.stop();
    let first_snapshot = timer.snapshot();
    assert!(first >= 0.0);
    assert_eq!(first_snapshot.calls, 1);
    timer.start();
    let second = timer.stop();
    let second_snapshot = timer.snapshot();
    assert!(second >= 0.0);
    assert_eq!(second_snapshot.calls, 2);
    assert!(second_snapshot.acc_time >= first_snapshot.acc_time);
}

fn nested_registry_keeps_exclusive_accounting_and_sorted_snapshot() {
    let mut registry = TimerRegistry::new();
    registry.start("outer");
    registry.start("inner");
    let inner = registry.stop("inner");
    let outer = registry.stop("outer");
    assert!(inner >= 0.0 && outer >= inner);
    assert!(registry.exclusive_time() >= 0.0);
    let snapshots = registry.snapshot();
    assert_eq!(
        snapshots
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>(),
        vec!["outer", "inner"]
    );
    assert_eq!(
        snapshots.iter().map(|s| s.calls).collect::<Vec<_>>(),
        vec![1, 1]
    );
    assert!(snapshots[0].acc_excl_time <= snapshots[0].acc_time);
    assert!(snapshots[1].acc_excl_time <= snapshots[1].acc_time);
}

fn function_timer_totals_use_context(context: &Context<'_>) {
    let mut timer = FunctionTimer::new("reduce");
    timer.record(3.5, 1.25);
    timer.compute_totals(context);
    assert_eq!(timer.total_time, 3.5 * context.size() as f64);
    assert_eq!(timer.total_excl_time, 2.25 * context.size() as f64);
    assert_eq!(timer.total_calls, context.size() as i64);
}

fn packed_sizes_and_indices_match_source_recurrences() {
    assert_eq!(util::sy_packed_size(0, &[], &[]), 1);
    assert_eq!(util::packed_size(0, &[], &[]), 1);
    assert_eq!(util::sy_packed_size(1, &[4], &[NS]), 4);
    assert_eq!(util::sy_packed_size(3, &[4, 4, 4], &[SY, SY, NS]), 20);
    assert_eq!(util::packed_size(3, &[4, 4, 4], &[AS, AS, NS]), 4);
    assert_eq!(util::sy_calc_idx_arr(2, &[3, 3], &[SY, NS], 4), vec![1, 2]);
    assert_eq!(util::calc_idx_arr(2, &[3, 3], &[AS, NS], 2), vec![1, 2]);
    assert_eq!(util::sy_calc_idx_arr(2, &[3, 3], &[AS, NS], 2), vec![1, 1]);
}

fn integer_permutation_and_choice_helpers_are_exact() {
    assert_eq!(util::factorize(60), vec![2, 2, 3, 5]);
    assert_eq!(util::gcd(84, 30), 6);
    assert_eq!(util::lcm(84, 30), 420);
    assert_eq!(util::fact(6), 720);
    assert_eq!(util::choose(6, 2), 15);
    assert_eq!(util::chchoose(4, 3), 20);
    assert_eq!(util::get_choice(4, 2, 0), vec![0, 0]);
    assert_eq!(util::get_choice(4, 2, 3), vec![0, 3]);

    let mut ints = [10, 20, 30, 40];
    util::permute(&[2, 0, 3, 1], &mut ints);
    assert_eq!(ints, [30, 10, 40, 20]);
    util::permute(&[1, 3, 0, 2], &mut ints);
    assert_eq!(ints, [10, 20, 30, 40]);
}

fn column_major_and_block_copy_helpers_preserve_bytes() {
    let mut copied = vec![0u8; 8];
    util::lda_cpy(1, 2, 2, 3, 4, &[1, 2, 9, 3, 4, 9], &mut copied);
    assert_eq!(&copied[..8], &[1, 2, 0, 0, 3, 4, 0, 0]);

    let sizes_a = [1, 2, 99, 3, 4, 99];
    let offsets_a = [0, 1, 3, 6, 9, 13];
    let (sizes_b, offsets_b) = util::socopy(2, 2, 3, 2, &sizes_a);
    assert_eq!(sizes_b, [1, 2, 3, 4]);
    assert_eq!(offsets_b, [0, 1, 3, 6]);
    let source = b"abcdefghijklmno";
    let mut target = vec![0u8; 10];
    util::spcopy(
        2,
        2,
        3,
        2,
        &sizes_a,
        &offsets_a,
        source,
        &sizes_b,
        &offsets_b,
        &mut target,
    );
    assert_eq!(&target[..10], b"abcghijklm");
}

fn run(context: &Context<'_>) {
    timer_accumulates_monotonic_wall_snapshots();
    nested_registry_keeps_exclusive_accounting_and_sorted_snapshot();
    function_timer_totals_use_context(context);
    packed_sizes_and_indices_match_source_recurrences();
    integer_permutation_and_choice_helpers_are_exact();
    column_major_and_block_copy_helpers_preserve_bytes();
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world);
    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS d4_timer_util: timer invariants and util integer/index/layout checks exact; world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
