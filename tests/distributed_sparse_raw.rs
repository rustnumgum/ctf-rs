use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    normal_mapping::Problem,
    sparse::SparseTensor,
    tensor::Tensor,
};

fn run_case(context: &Context<'_>, mapped: [Distribution; 3], empty: bool) {
    let da = Distribution::cyclic(vec![5, 3], context.size());
    let db = Distribution::cyclic(vec![3, 4], context.size());
    let dc = Distribution::cyclic(vec![5, 4], context.size());
    let mut a = SparseTensor::new(context, da.clone(), Arithmetic::<i64>::new());
    if !empty {
        let pairs: Vec<_> = (0..15).filter(|&key| key % 4 != 0 && da.owner(key) == context.rank())
            .map(|key| (key, key as i64 % 7 - 3)).collect();
        a.write_add(&pairs);
    }
    let mut b = Tensor::new(context, db, Arithmetic::<i64>::new());
    b.transform(|key, value| *value = key as i64 - 2);
    let mut c = Tensor::new(context, dc.clone(), Arithmetic::<i64>::new());
    c.transform(|_, value| *value = 5);
    c.contract_sparse_from_mapped("ij", &a, "ik", &b, "kj", mapped, 2, 3, true);
    assert_eq!(c.distribution(), &dc);
    let keys: Vec<_> = (0..20).collect();
    let expected: Vec<_> = keys.iter().map(|&key| {
        if empty { return 15; }
        let i = key % 5;
        let j = key / 5;
        15 + (0..3).filter(|&k| (i + 5 * k) % 4 != 0).map(|k| {
            2 * ((i + 5 * k) as i64 % 7 - 3) * ((k + 3 * j) as i64 - 2)
        }).sum::<i64>()
    }).collect();
    assert_eq!(c.read(&keys), expected);
}

fn run(context: &Context<'_>) {
    let shapes: [&[usize]; 3] = [&[5, 3], &[3, 4], &[5, 4]];
    let topology = if context.size() == 4 {
        Topology::new(vec![2, 2])
    } else {
        Topology::new(vec![context.size(), 1])
    };
    let mismatched = Problem::new(shapes, ["ik", "kj", "ij"]).unwrap()
        .map_to_topology(&topology, 0, [None; 3]).unwrap();
    run_case(context, mismatched, false);
    run_case(context, Problem::new(shapes, ["ik", "kj", "ij"]).unwrap()
        .map_to_topology(&topology, 0, [None; 3]).unwrap(), true);

    let topology = Topology::new(vec![context.size()]);
    let physical = || Mapping::Physical { axis: 0, processes: context.size(),
        child: Box::new(Mapping::Unmapped) };
    let virtual_two = || Mapping::Virtual { copies: 2,
        child: Box::new(Mapping::Unmapped) };
    let virtual_mapped = [
        Distribution::new(vec![5, 3], topology.clone(), vec![physical(), virtual_two()]),
        Distribution::new(vec![3, 4], topology.clone(), vec![virtual_two(), Mapping::Unmapped]),
        Distribution::new(vec![5, 4], topology, vec![physical(), Mapping::Unmapped]),
    ];
    run_case(context, virtual_mapped, false);
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world);
    let parity = world.split(Some((world.rank() % 2) as i32), world.rank() as i32).unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!("DIGIT / PASS distributed_sparse_raw: source raw mapped panels, beta, uneven shapes, empty shards, virtual factors; exact i64; world+parity");
    }
    world.close();
    runtime.finalize();
}
