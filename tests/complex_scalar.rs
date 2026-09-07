use ctf::{
    algebra::{Arithmetic, Complex, Group, Semiring},
    context::Runtime,
    contraction::sequential,
    mapping::Distribution,
    tensor::Tensor,
};
fn main() {
    let a = Arithmetic::<Complex<f64>>::new();
    let z = Complex::<f64>::new(2., 3.);
    assert_eq!(a.multiply(&z, &z), Complex::new(-5., 12.));
    assert_eq!(a.negate(&z), Complex::new(-2., -3.));
    assert_eq!(z.conjugate(), Complex::new(2., -3.));
    assert_eq!(z.norm_squared(), 13.);
    let f = Arithmetic::<Complex<f32>>::new();
    assert_eq!(
        f.multiply(&Complex::new(0., 1.), &Complex::new(0., 1.)),
        Complex::new(-1., 0.)
    );
    let mut product = [Complex::new(0., 0.)];
    sequential(
        &a,
        &[2],
        "i",
        &[z, z],
        &[2],
        "i",
        &[z, z],
        &[],
        "",
        &mut product,
        &Complex::new(1., 0.),
        &Complex::new(0., 0.),
    );
    assert_eq!(product, [Complex::new(-10., 24.)]);
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let np = world.size();
    let mut value = [Complex::new(world.rank() as f64 + 1., 1.)];
    world.all_reduce_monoid(&a, &mut value, true);
    assert_eq!(value, [Complex::new((np * (np + 1) / 2) as f64, np as f64)]);
    let mut tensor = Tensor::new(&world, Distribution::cyclic(vec![3], np), a);
    let pairs = [(1, z)];
    tensor.write_add(if world.rank() == 0 { &pairs } else { &[] });
    assert_eq!(tensor.read(&[1]), vec![z]);
    tensor.scale(&Complex::new(0., 1.));
    assert_eq!(tensor.read(&[1]), vec![Complex::new(-3., 2.)]);
    drop(tensor);
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS complex_scalar: f32/f64 complex algebra, contraction, Wire and MPI; ranks={np}"
        );
    }
    world.close();
    runtime.finalize();
}
