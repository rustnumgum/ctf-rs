use ctf::{
    algebra::Arithmetic,
    context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology},
    tensor::Tensor,
};

fn old_and_new(context: &Context<'_>, shape: Vec<usize>) -> (Distribution, Distribution) {
    let np = context.size();
    if np % 2 == 0 {
        let topology = Topology::new(vec![2, np / 2]);
        let mut old_map = Mapping::Unmapped;
        old_map.augment_physical(&topology, 0);
        old_map.augment_virtual(4);
        let mut new_map = Mapping::Unmapped;
        new_map.augment_physical(&topology, 0);
        if np / 2 > 1 {
            new_map.augment_physical(&topology, 1);
        }
        new_map.augment_virtual(2 * np);
        (
            Distribution::new(
                shape.clone(),
                topology.clone(),
                vec![old_map, Mapping::Unmapped],
            ),
            Distribution::new(shape, topology, vec![new_map, Mapping::Unmapped]),
        )
    } else {
        let topology = Topology::new(vec![np]);
        let mut new_map = Mapping::Unmapped;
        new_map.augment_physical(&topology, 0);
        new_map.augment_virtual(2 * np);
        (
            Distribution::new(
                shape.clone(),
                topology.clone(),
                vec![Mapping::Unmapped; 2],
            ),
            Distribution::new(shape, topology, vec![new_map, Mapping::Unmapped]),
        )
    }
}

fn assert_root_only(tensor: &Tensor<'_, '_, Arithmetic<i64>>, context: &Context<'_>) {
    let distribution = tensor.distribution();
    for (offset, &value) in tensor.local_storage().iter().enumerate() {
        let expected = distribution
            .global_key(context.rank(), offset)
            .filter(|&key| distribution.owner(key) == context.rank())
            .map_or(0, |key| key as i64 + 11);
        assert_eq!(value, expected, "rank {}, offset {offset}", context.rank());
    }
}

fn run(context: &Context<'_>) {
    let (old, new) = old_and_new(context, vec![7, 3]);
    let mut tensor = Tensor::new(context, old, Arithmetic::<i64>::new());
    tensor.transform(|key, value| *value = key as i64 + 11);
    tensor.redistribute(new);
    assert_root_only(&tensor, context);

    let empty = Distribution::cyclic(vec![0, 3], context.size());
    let mut empty_tensor = Tensor::new(context, empty.clone(), Arithmetic::<i64>::new());
    empty_tensor.redistribute(empty);
    assert!(empty_tensor.local_storage().is_empty());

    let mut scalar = Tensor::new(
        context,
        Distribution::new(vec![], Topology::new(vec![context.size()]), vec![]),
        Arithmetic::<i64>::new(),
    );
    scalar.transform(|_, value| *value = 29);
    scalar.redistribute(Distribution::new(
        vec![],
        Topology::new(vec![context.size()]),
        vec![],
    ));
    assert_eq!(scalar.local_storage(), if context.rank() == 0 { &[29] } else { &[0] });

    let shape = vec![1; 13];
    let mut high_order = Tensor::new(
        context,
        Distribution::cyclic(shape.clone(), context.size()),
        Arithmetic::<i64>::new(),
    );
    high_order.transform(|_, value| *value = 41);
    high_order.redistribute(Distribution::new(
        shape.clone(),
        Topology::new(vec![context.size()]),
        vec![Mapping::Unmapped; shape.len()],
    ));
    assert_eq!(high_order.local_storage(), &[41]);
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
        println!("DIGIT / PASS dgtog_redistribution: exact LCM counts/ROR roots/padding/virtual/scalar/high-order legacy; world+parity");
    }
    world.close();
    runtime.finalize();
}
