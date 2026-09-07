use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    linalg::Native,
    mapping::{Distribution, Topology},
    tensor::Tensor,
};
fn exercise(context: &Context<'_>) {
    let np = context.size();
    let topology = Topology::new(if np == 4 { vec![2, 2] } else { vec![np] });
    let make = |shape| {
        Tensor::new(
            context,
            Distribution::cyclic(shape, np),
            Arithmetic::<f64>::new(),
        )
    };
    // Nonadjacent pair of contracted dimensions and a shared batch dimension.
    let mut a = make(vec![3, 2, 5, 2]);
    let mut b = make(vec![2, 5, 2, 4]);
    let mut c = make(vec![4, 2, 3]);
    a.transform(|key, value| *value = (key % 3 + 1) as f64);
    b.transform(|key, value| *value = (key / 20 + 2) as f64);
    c.transform(|_, value| *value = 7.);
    let original = c.distribution().clone();
    c.contract_blas_on_grid::<Native>("jwi", &a, "ixyw", &b, "wyxj", topology.clone(), 2., 3.)
        .unwrap();
    assert_eq!(c.distribution(), &original);
    for (key, value) in c.local_pairs() {
        assert_eq!(
            value,
            21. + 20. * (key / 8 + 1) as f64 * (key % 4 + 2) as f64
        );
    }
    // Outer product and scalar operands are fully foldable with k=1.
    let mut x = make(vec![2]);
    let mut y = make(vec![3]);
    let mut z = make(vec![3, 2]);
    x.transform(|key, value| *value = (key + 1) as f64);
    y.transform(|key, value| *value = (key + 3) as f64);
    z.contract_blas_on_grid::<Native>("ji", &x, "i", &y, "j", topology.clone(), 1., 0.)
        .unwrap();
    for (key, value) in z.local_pairs() {
        assert_eq!(value, (key / 3 + 1) as f64 * (key % 3 + 3) as f64);
    }
    let empty_a = make(vec![1, 0]);
    let empty_b = make(vec![0, 1]);
    let mut empty_c = make(vec![1, 1]);
    empty_c.transform(|_, v| *v = 5.);
    empty_c
        .contract_blas_on_grid::<Native>(
            "ij",
            &empty_a,
            "ik",
            &empty_b,
            "kj",
            topology.clone(),
            1.,
            3.,
        )
        .unwrap();
    assert_eq!(empty_c.read(&[0]), vec![15.]);
    let mut scalar_a = make(vec![]);
    let mut scalar_b = make(vec![]);
    let mut scalar_c = make(vec![]);
    scalar_a.transform(|_, v| *v = 2.);
    scalar_b.transform(|_, v| *v = 3.);
    scalar_c
        .contract_blas_on_grid::<Native>("", &scalar_a, "", &scalar_b, "", topology, 2., 0.)
        .unwrap();
    assert_eq!(scalar_c.read(&[0]), vec![12.]);
}
fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    exercise(&world);
    let child = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    exercise(&child);
    child.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS tensor_blas_fold: high-order weighted multi-index reduction, outer product, empty contraction and scalar; ranks={}",
            world.size()
        );
    }
    world.close();
    runtime.finalize();
}
