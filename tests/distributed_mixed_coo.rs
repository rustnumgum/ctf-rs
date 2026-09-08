use ctf::{
    algebra::Arithmetic,
    context::Context,
    sparse_2d::{Layers, Panel, execute_coo_dense},
    sparse_formats::Coo,
};
fn whole() -> Panel<'static, 'static> {
    Panel {
        comm: None,
        outer: 1,
        inner: 0,
    }
}
fn run(context: &Context<'_>) {
    let p = context.size();
    let rank = context.rank();
    let edge = 2 * p;
    let local: Vec<_> = [rank, rank + p]
        .into_iter()
        .map(|s| {
            Coo::new(
                1,
                1,
                if s % 3 == 0 {
                    vec![]
                } else {
                    vec![(1, 1, s as i32), (1, 1, 0)]
                },
            )
        })
        .collect();
    let c = execute_coo_dense(
        &Arithmetic::<i64>::new(),
        edge,
        Layers { count: 1, index: 0 },
        Panel {
            comm: Some(context),
            outer: 1,
            inner: 1,
        },
        whole(),
        whole(),
        &local,
        &[vec![true, false]],
        vec![vec![10_i64; 2]],
        3,
        |a, b, mut c, beta, _| {
            for value in &mut c[0] {
                *value *= beta;
            }
            a[0].coomm_kernel(
                &Arithmetic::<i64>::new(),
                2,
                &b[0],
                None,
                &1,
                &mut c[0],
                |a, b| *a as i64 + i64::from(*b),
                |value, c| *c += value,
            );
            c
        },
    );
    let mut expected = vec![30_i64; 2];
    for s in 0..edge {
        if s % 3 != 0 {
            expected[0] += s as i64 + 2;
            expected[1] += s as i64;
        }
    }
    assert_eq!(c, vec![expected]);
    let a: Vec<_> = (0..edge)
        .map(|s| Coo::new(1, 1, vec![(1, 1, s as i32)]))
        .collect();
    let b: Vec<_> = [rank, rank + p]
        .into_iter()
        .map(|s| vec![s % 2 == 0, false])
        .collect();
    let c = execute_coo_dense(
        &Arithmetic::<i64>::new(),
        edge,
        Layers { count: 1, index: 0 },
        Panel {
            comm: None,
            outer: 1,
            inner: 1,
        },
        Panel {
            comm: Some(context),
            outer: 1,
            inner: 1,
        },
        Panel {
            comm: Some(context),
            outer: 1,
            inner: 1,
        },
        &a,
        &b,
        vec![vec![10_i64; 2]; 2],
        2,
        |a, b, mut c, beta, _| {
            for value in &mut c[0] {
                *value *= beta;
            }
            a[0].coomm_kernel(
                &Arithmetic::<i64>::new(),
                2,
                &b[0],
                Some(&1),
                &1,
                &mut c[0],
                |a, b| *a as i64 + i64::from(*b),
                |value, c| *c += value,
            );
            c
        },
    );
    let expected: Vec<_> = [rank, rank + p]
        .into_iter()
        .map(|s| {
            vec![
                20 + p as i64 * (s as i64 + i64::from(s % 2 == 0)),
                20 + p as i64 * s as i64,
            ]
        })
        .collect();
    assert_eq!(c, expected);
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
            "DIGIT / PASS distributed_mixed_coo: i32/bool/i64 panels, custom leaf, simultaneous B broadcast/C reduction, explicit zeros and empty COO; exact; world+parity"
        );
    }
    world.close();
    drop(universe);
}
