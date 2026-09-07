use ctf::{
    algebra::Arithmetic,
    context::Runtime,
    mapping::{Distribution, Mapping, Topology},
    tensor::Tensor,
};
fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let topology = Topology::new(vec![world.size()]);
    let mut physical = Mapping::Unmapped;
    physical.augment_physical(&topology, 0);
    physical.augment_virtual(world.size() * 2);
    let mut a = Tensor::new(
        &world,
        Distribution::new(
            vec![5, 3],
            topology.clone(),
            vec![physical.clone(), Mapping::Unmapped],
        ),
        Arithmetic::<f64>::new(),
    );
    a.transform(|key, value| *value = (key + 1) as f64);
    let mut row = Tensor::new(
        &world,
        Distribution::new(vec![5], topology.clone(), vec![physical.clone()]),
        Arithmetic::<f64>::new(),
    );
    row.sum_from_aligned("i", &a, "ij", 1., 0.);
    for (key, value) in row.local_pairs() {
        assert_eq!(value, 18. + 3. * key as f64);
    }
    let mut column = Tensor::new(
        &world,
        Distribution::new(vec![3], topology.clone(), vec![Mapping::Unmapped]),
        Arithmetic::<f64>::new(),
    );
    column.transform(|_, v| *v = 10.);
    column.sum_from_aligned("j", &a, "ij", 2., 3.);
    assert_eq!(column.read(&[0, 1, 2]), vec![60., 110., 160.]);
    let mut expanded = Tensor::new(&world, a.distribution().clone(), Arithmetic::<f64>::new());
    expanded.sum_from_aligned("ij", &column, "j", 1., 0.);
    for (key, value) in expanded.local_pairs() {
        assert_eq!(value, [60., 110., 160.][key / 5]);
    }
    let mut transposed = Tensor::new(
        &world,
        Distribution::new(vec![3, 5], topology, vec![Mapping::Unmapped, physical]),
        Arithmetic::<f64>::new(),
    );
    transposed.sum_from_aligned("ji", &a, "ij", 1., 0.);
    for (key, value) in transposed.local_pairs() {
        assert_eq!(value, (key / 3 + (key % 3) * 5 + 1) as f64);
    }
    drop(transposed);
    drop(expanded);
    drop(column);
    drop(row);
    drop(a);
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS tensor_sum: aligned Tensor API, local/physical reductions, broadcast and transpose; ranks={}",
            world.size()
        );
    }
    world.close();
    runtime.finalize();
}
