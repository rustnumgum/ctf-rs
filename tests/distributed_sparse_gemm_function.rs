use ctf::{
    algebra::Arithmetic, context::Context, mapping::Distribution, sparse::SparseTensor,
    tensor::Tensor,
};
fn run(c: &Context<'_>) {
    let d = Distribution::cyclic(vec![2, 2], c.size());
    let mut b = Tensor::new(c, d.clone(), Arithmetic::<i64>::new());
    b.transform(|key, value| *value = [0, 1, 2, 0][key]);
    let mut bb = SparseTensor::new(c, d.clone(), Arithmetic::<i64>::new());
    bb.write_add(
        &[(0, 0), (3, 0)]
            .into_iter()
            .filter(|(key, _)| d.owner(*key) == c.rank())
            .collect::<Vec<_>>(),
    );
    for empty in [false, true] {
        let mut a = SparseTensor::new(c, d.clone(), Arithmetic::<i64>::new());
        if !empty {
            a.write_add(
                &[(0, 0), (3, 2)]
                    .into_iter()
                    .filter(|(key, _)| d.owner(*key) == c.rank())
                    .collect::<Vec<_>>(),
            );
        }
        let mut dense = Tensor::new(c, d.clone(), Arithmetic::<i64>::new());
        dense.transform(|_, value| *value = 3);
        let mut sparse = dense.clone();
        let grid = if c.size() == 4 { [2, 2] } else { [c.size(), 1] };
        dense.gemm_sparse_dense_function(&a, &b, grid, 1, 3, |a, b| a + b + 1);
        sparse.gemm_sparse_function(&a, &bb, grid, 1, 3, |a, b| a + b + 1);
        assert_eq!(
            dense.read(&[0, 1, 2, 3]),
            if empty {
                vec![9; 4]
            } else {
                vec![10, 13, 12, 12]
            }
        );
        assert_eq!(
            sparse.read(&[0, 1, 2, 3]),
            if empty {
                vec![9; 4]
            } else {
                vec![10, 9, 9, 12]
            }
        );
        assert_eq!(dense.distribution(), &d);
        assert_eq!(sparse.distribution(), &d);
    }
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
            "DIGIT / PASS distributed_sparse_gemm_function: source CSR custom kernels and distributed panels, stored zeros, missing entries, empty sparse shards, world+parity; exact i64"
        );
    }
    world.close();
    drop(universe);
}
