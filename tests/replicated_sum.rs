use ctf::{context::Runtime, summation::replicated_f64};
fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let rank = world.rank();
    let np = world.size();
    let mut a = [rank as f64 + 1., rank as f64 + 2.];
    let mut b = [10., 20.];
    replicated_f64(
        &[],
        &[&world],
        &[2],
        &[1],
        "i",
        &mut a,
        &[2],
        &[1],
        "i",
        &mut b,
        2.,
        3.,
    );
    assert_eq!(
        b,
        [30. + (np * (np + 1)) as f64, 60. + (np * (np + 3)) as f64]
    );
    let mut a = if rank == 0 { [2., 7.] } else { [99., 99.] };
    let mut b = [0.; 2];
    replicated_f64(
        &[&world],
        &[],
        &[2],
        &[1],
        "i",
        &mut a,
        &[2],
        &[1],
        "i",
        &mut b,
        1.,
        0.,
    );
    assert_eq!(b, [2., 7.]);
    let mut a = [1., 2., 3., 4.];
    let mut b = [10.];
    replicated_f64(
        &[],
        &[&world],
        &[2],
        &[2],
        "i",
        &mut a,
        &[],
        &[],
        "",
        &mut b,
        1.,
        2.,
    );
    assert_eq!(b, [20. + 10. * np as f64]);
    replicated_f64(
        &[],
        &[&world],
        &[0],
        &[1],
        "i",
        &mut [],
        &[0],
        &[1],
        "i",
        &mut [],
        1.,
        0.,
    );
    if rank == 0 {
        println!(
            "DIGIT / PASS replicated_sum: broadcast, native block Allreduce, root beta, virtual reduction, zero counts; ranks={np}"
        );
    }
    world.close();
    runtime.finalize();
}
