use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Topology},
    tensor::Tensor,
};
fn run(c: &Context<'_>) {
    let mut a = Tensor::new(
        c,
        Distribution::cyclic(vec![2, 3], c.size()),
        Arithmetic::<i64>::new(),
    );
    let mut b = Tensor::new(
        c,
        Distribution::cyclic(vec![3, 2], c.size()),
        Arithmetic::<i64>::new(),
    );
    a.transform(|key, v| *v = key as i64 - 2);
    b.transform(|key, v| *v = key as i64 % 3);
    for physical in ["i", "j", "k"] {
        let mut output = Tensor::new(
            c,
            Distribution::cyclic(vec![2, 2, 2], c.size()),
            Arithmetic::<i64>::new(),
        );
        output.transform(|_, v| *v = 7);
        let original = output.distribution().clone();
        output.contract_function_on(
            "ijx",
            &a,
            "ik",
            &b,
            "kj",
            Topology::new(vec![c.size()]),
            physical,
            2,
            3,
            |a, b| a + b + 1,
        );
        let keys: Vec<_> = (0..8).collect();
        let expected: Vec<_> = keys
            .iter()
            .map(|&key| {
                let i = key % 2;
                let j = key / 2 % 2;
                21 + 2
                    * (0..3)
                        .map(|k| (i + 2 * k) as i64 - 2 + (k + 3 * j) as i64 % 3 + 1)
                        .sum::<i64>()
            })
            .collect();
        assert_eq!(output.read(&keys), expected, "physical={physical}");
        assert_eq!(output.distribution(), &original);
    }
    let mut diagonal = Tensor::new(
        c,
        Distribution::cyclic(vec![2, 2, 2], c.size()),
        Arithmetic::<i64>::new(),
    );
    diagonal.transform(|_, v| *v = 7);
    diagonal.contract_function_on(
        "iji",
        &a,
        "ik",
        &b,
        "kj",
        Topology::new(vec![c.size()]),
        "k",
        2,
        3,
        |a, b| a + b + 1,
    );
    let keys: Vec<_> = (0..8).collect();
    let expected: Vec<_> = keys
        .iter()
        .map(|&key| {
            let i = key % 2;
            let j = key / 2 % 2;
            if key / 4 != i {
                return 7;
            }
            21 + 2
                * (0..3)
                    .map(|k| (i + 2 * k) as i64 - 2 + (k + 3 * j) as i64 % 3 + 1)
                    .sum::<i64>()
        })
        .collect();
    assert_eq!(diagonal.read(&keys), expected);
}
fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    run(&world);
    let child = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&child);
    child.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS distributed_dense_function: non-distributive function, no padding contributions, output-only indices, broadcasts/reductions, world+parity; exact i64"
        );
    }
    world.close();
    drop(universe);
}
