use ctf::{
    context::{Context, Runtime},
    memcontrol::{MemoryFraction, MemorySnapshot, ProcessMemoryBudget, ProcessesPerMachine},
};

fn run(context: &Context<'_>) {
    let parsed = MemorySnapshot::from_linux(
        "MemTotal: 16000 kB\nMemAvailable: 6000 kB\n",
        "VmRSS: 1000 kB\n",
        Some("12288000\n"),
    )
    .unwrap();
    assert_eq!(
        parsed,
        MemorySnapshot {
            total_bytes: 12_288_000,
            used_bytes: 1_024_000
        }
    );
    let budget = ProcessMemoryBudget::from_snapshot(
        parsed,
        ProcessesPerMachine::new(2).unwrap(),
        MemoryFraction::new(3, 4).unwrap(),
    );
    assert_eq!(budget.bytes(), 3_584_000);

    let rank_snapshot = MemorySnapshot {
        total_bytes: 10_000,
        used_bytes: context.rank() as u64 * 100,
    };
    let local = ProcessMemoryBudget::from_snapshot(
        rank_snapshot,
        ProcessesPerMachine::new(1).unwrap(),
        MemoryFraction::new(1, 2).unwrap(),
    );
    assert_eq!(
        local.collective_min(context).bytes(),
        5_000 - (context.size() as u64 - 1) * 100
    );

    let live = MemorySnapshot::discover().unwrap();
    assert!(live.total_bytes > 0);
    let live_budget = ProcessMemoryBudget::from_snapshot(
        live,
        ProcessesPerMachine::new(1).unwrap(),
        MemoryFraction::new(1, 2).unwrap(),
    );
    assert!(live_budget.bytes() <= live.total_bytes / 2);
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
            "DIGIT / PASS d4_memcontrol: source cap/process arithmetic, OS discovery and collective budget exact; world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
