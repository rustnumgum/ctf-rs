use ctf::{
    algebra::Arithmetic,
    context::Context,
    cost::Models,
    mapping::{Distribution, Topology},
    sparse::SparseTensor,
    sparse_search::{self, Options},
    tensor::Tensor,
};

fn run(context: &Context<'_>) {
    let mut catalog = vec![Topology::new(vec![context.size()])];
    for dimensions in [vec![context.size(), 1], vec![1, context.size()]] {
        let topology = Topology::new(dimensions);
        if !catalog.contains(&topology) {
            catalog.push(topology);
        }
    }
    if context.size() == 4 {
        catalog.push(Topology::new(vec![2, 2]));
    }
    let mut models = Models::upstream(1);
    // Keep source exhaustive-refinement threshold active without measuring time.
    models
        .get_mut("seq_tsr_spctr_k0")
        .set_coefficients(&[1., 0., 1.]);
    let da = Distribution::cyclic(vec![5, 3], context.size());
    let db = Distribution::cyclic(vec![3, 4], context.size());
    let dc = Distribution::cyclic(vec![5, 4], context.size());
    let mut b = Tensor::new(context, db, Arithmetic::<i64>::new());
    b.transform(|key, x| *x = key as i64 - 2);
    for (weight, empty) in [(0., false), (0.5, true)] {
        let mut a = SparseTensor::new(context, da.clone(), Arithmetic::<i64>::new());
        let global: Vec<_> = (0..15)
            .filter(|&key| !empty && key % 3 != 0)
            .map(|key| (key, key as i64 % 7 - 3))
            .collect();
        a.write_add(
            &global
                .iter()
                .filter(|&&(key, _)| da.owner(key) == context.rank())
                .copied()
                .collect::<Vec<_>>(),
        );
        let mut c = Tensor::new(context, dc.clone(), Arithmetic::<i64>::new());
        c.transform(|_, x| *x = 3);
        let selected = sparse_search::search_unfolded(
            context,
            [a.distribution(), b.distribution(), c.distribution()],
            ["ik", "kj", "ij"],
            &catalog,
            &models,
            global.len() as u64,
            8,
            16,
            true,
            Options {
                memory_limit: 1 << 30,
                weight,
                allow_exhaustive: true,
            },
        )
        .unwrap()
        .unwrap();
        assert!(selected.seconds >= 0.01 && selected.memory_bytes < (1 << 30));
        assert!(selected.fold.is_none());
        let mut winner = [
            selected.source_id as u64,
            selected.seconds.to_bits(),
            selected.memory_bytes,
        ];
        let local = winner;
        context.broadcast(0, &mut winner);
        assert_eq!(winner, local);
        c.contract_sparse_from_mapped("ij", &a, "ik", &b, "kj", selected.distributions, 2, 3, true);
        assert_eq!(c.distribution(), &dc);
        let keys: Vec<_> = (0..20).collect();
        let expected: Vec<_> = keys
            .iter()
            .map(|&key| {
                let i = key % 5;
                let j = key / 5;
                9 + global
                    .iter()
                    .filter(|&&(a_key, _)| a_key % 5 == i)
                    .map(|&(a_key, value)| 2 * value * ((a_key / 5 + 3 * j) as i64 - 2))
                    .sum::<i64>()
            })
            .collect();
        assert_eq!(c.read(&keys), expected);
    }
    assert!(
        sparse_search::search_unfolded(
            context,
            [&da, b.distribution(), &dc],
            ["ik", "kj", "ij"],
            &catalog,
            &models,
            10,
            8,
            16,
            true,
            Options {
                memory_limit: 0,
                weight: 0.,
                allow_exhaustive: true
            }
        )
        .unwrap()
        .is_none()
    );
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
            "DIGIT / PASS distributed_sparse_search: normal/weighted/exhaustive selection, raw execution, zero nnz, strict memory rejection, exact i64; world+parity"
        );
    }
    world.close();
    drop(universe);
}
