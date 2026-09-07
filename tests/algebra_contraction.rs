use ctf::{
    algebra::{Arithmetic, CustomMonoid, CustomSemiring},
    context::Runtime,
    contraction::replicated,
};
fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let np = world.size();
    let mut a = [world.rank() as i64 + 1];
    let mut b = [2];
    let mut c = [10];
    replicated(
        &Arithmetic::<i64>::new(),
        &[],
        &[],
        &[&world],
        &[],
        &[],
        "",
        &mut a,
        &[],
        &[],
        "",
        &mut b,
        &[],
        &[],
        "",
        &mut c,
        &2,
        &3,
        true,
    );
    if world.rank() == 0 {
        assert_eq!(c, [30 + 2 * (np * (np + 1)) as i64]);
    }
    let algebra = CustomSemiring {
        monoid: CustomMonoid {
            identity: false,
            addition: |a: &bool, b: &bool| *a || *b,
        },
        identity: true,
        multiplication: |a: &bool, b: &bool| *a && *b,
    };
    let mut a = [world.rank() == np - 1];
    let mut b = [true];
    let mut c = [false];
    replicated(
        &algebra,
        &[],
        &[],
        &[&world],
        &[],
        &[],
        "",
        &mut a,
        &[],
        &[],
        "",
        &mut b,
        &[],
        &[],
        "",
        &mut c,
        &true,
        &false,
        true,
    );
    if world.rank() == 0 {
        assert_eq!(c, [true]);
    }
    let mut values = [1i64];
    world.reduce_monoid(&Arithmetic::<i64>::new(), &mut values, true, np - 1);
    assert_eq!(values, [if world.rank() == np - 1 { np as i64 } else { 1 }]);
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS algebra_contraction: native generic Reduce, root beta, Boolean semiring, nonzero root; ranks={np}"
        );
    }
    world.close();
    runtime.finalize();
}
