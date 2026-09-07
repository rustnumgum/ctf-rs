use ctf::{
    algebra::{Arithmetic, CustomMonoid, CustomSemiring},
    context::Runtime,
    mapping::{Distribution, Mapping, Topology},
    tensor::Tensor,
};
fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let np = world.size();
    let mut a = Tensor::new(
        &world,
        Distribution::cyclic(vec![5, 2], np),
        Arithmetic::<i64>::new(),
    );
    a.transform(|key, v| *v = (key + 1) as i64);
    let d = Distribution::new(vec![2], Topology::new(vec![np]), vec![Mapping::Unmapped]);
    let mut b = Tensor::new(&world, d.clone(), Arithmetic::<i64>::new());
    b.sum_from_aligned("j", &a, "ij", 2, 0);
    assert_eq!(b.read(&[0, 1]), vec![30, 80]);
    fn algebra()
    -> CustomSemiring<CustomMonoid<bool, fn(&bool, &bool) -> bool>, fn(&bool, &bool) -> bool> {
        CustomSemiring {
            monoid: CustomMonoid {
                identity: false,
                addition: |a, b| *a || *b,
            },
            identity: true,
            multiplication: |a, b| *a && *b,
        }
    }
    let mut flags = Tensor::new(&world, Distribution::cyclic(vec![5, 2], np), algebra());
    flags.transform(|key, v| *v = key == 3);
    let mut any = Tensor::new(&world, d, algebra());
    any.sum_from_aligned("j", &flags, "ij", true, false);
    assert_eq!(any.read(&[0, 1]), vec![true, false]);
    drop(any);
    drop(flags);
    drop(b);
    drop(a);
    if world.rank() == 0 {
        println!("DIGIT / PASS algebra_sum: integer and Boolean Tensor reductions; ranks={np}");
    }
    world.close();
    runtime.finalize();
}
