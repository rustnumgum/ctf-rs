use ctf::{
    algebra::Arithmetic,
    context::Context,
    sparse_2d::{Layers, Panel, execute_pairs_dense},
    sparse_sequential::sequential,
};

fn whole() -> Panel<'static, 'static> {
    Panel {
        comm: None,
        outer: 1,
        inner: 0,
    }
}
// Three-dimensional local keys, including an explicitly stored zero and
// genuinely empty panels. These are not flattened CSR matrix coordinates.
fn pairs(step: usize) -> Vec<(usize, i64)> {
    if step % 3 == 0 {
        vec![]
    } else {
        vec![
            (0, step as i64 + 1),
            (3, if step % 2 == 0 { 0 } else { -2 }),
            (6, 3),
        ]
    }
}
fn leaf(
    a: &[Vec<(usize, i64)>],
    b: &[Vec<i64>],
    mut c: Vec<Vec<i64>>,
    beta: i64,
    layers: Layers,
) -> Vec<Vec<i64>> {
    assert_eq!(layers, Layers { count: 1, index: 0 });
    sequential(
        &Arithmetic::<i64>::new(),
        &[2, 2, 2],
        "ikl",
        &a[0],
        &[2, 2],
        "kj",
        &b[0],
        &[2, 2, 2],
        "ijl",
        &mut c[0],
        &1,
        &beta,
    );
    c
}
fn add_expected(c: &mut [i64], step: usize, b: &[i64], factor: i64) {
    if step % 3 == 0 {
        return;
    }
    for j in 0..2 {
        c[2 * j] += factor * (step as i64 + 1) * b[2 * j];
        c[1 + 2 * j] += factor * if step % 2 == 0 { 0 } else { -2 } * b[1 + 2 * j];
        c[4 + 2 * j] += factor * 3 * b[1 + 2 * j];
    }
}
fn inputs(context: &Context<'_>) {
    let edge = 2 * context.size();
    let rank = context.rank();
    let moving = Panel {
        comm: Some(context),
        outer: 1,
        inner: 1,
    };
    let a: Vec<_> = [rank, rank + context.size()]
        .into_iter()
        .map(pairs)
        .collect();
    let b = vec![2, 3, -1, 4];
    for layers in [Layers { count: 1, index: 0 }, Layers { count: 2, index: 1 }] {
        let c = execute_pairs_dense(
            &Arithmetic::<i64>::new(),
            edge,
            layers,
            moving,
            whole(),
            whole(),
            &a,
            &[b.clone()],
            vec![vec![10; 8]],
            3,
            leaf,
        );
        let mut expected = vec![30; 8];
        for step in (layers.index..edge).step_by(layers.count) {
            add_expected(&mut expected, step, &b, 1);
        }
        assert_eq!(c, vec![expected]);
    }
    let b: Vec<_> = [rank, rank + context.size()]
        .into_iter()
        .map(|step| vec![step as i64 + 1, 2, 3, -1])
        .collect();
    let c = execute_pairs_dense(
        &Arithmetic::<i64>::new(),
        edge,
        Layers { count: 1, index: 0 },
        whole(),
        moving,
        whole(),
        &[pairs(1)],
        &b,
        vec![vec![10; 8]],
        3,
        leaf,
    );
    let mut expected = vec![30; 8];
    for step in 0..edge {
        add_expected(&mut expected, 1, &[step as i64 + 1, 2, 3, -1], 1);
    }
    assert_eq!(c, vec![expected]);
}
fn outputs(context: &Context<'_>) {
    let p = context.size();
    let edge = 2 * p;
    let rank = context.rank();
    let a: Vec<_> = (0..edge).map(pairs).collect();
    let b = [vec![2, 3, -1, 4], vec![5, -2, 1, 3]];
    for moving in [false, true] {
        for beta in [0, 2] {
            let local_steps = if moving { 2 } else { edge };
            let c = execute_pairs_dense(
                &Arithmetic::<i64>::new(),
                edge,
                Layers { count: 1, index: 0 },
                Panel {
                    comm: None,
                    outer: 1,
                    inner: 1,
                },
                whole(),
                Panel {
                    comm: if moving { Some(context) } else { None },
                    outer: 2,
                    inner: 1,
                },
                &a,
                &b,
                vec![vec![10; 8]; 2 * local_steps],
                beta,
                |a, b, mut c, beta, l| {
                    assert_eq!(l, Layers { count: 1, index: 0 });
                    for strip in 0..2 {
                        sequential(
                            &Arithmetic::<i64>::new(),
                            &[2, 2, 2],
                            "ikl",
                            &a[0],
                            &[2, 2],
                            "kj",
                            &b[strip],
                            &[2, 2, 2],
                            "ijl",
                            &mut c[strip],
                            &1,
                            &beta,
                        );
                    }
                    c
                },
            );
            let mut expected = Vec::new();
            for strip in 0..2 {
                for local in 0..local_steps {
                    let step = if moving { rank + local * p } else { local };
                    let mut block = vec![10 * beta; 8];
                    add_expected(
                        &mut block,
                        step,
                        &b[strip],
                        if moving { p as i64 } else { 1 },
                    );
                    expected.push(block);
                }
            }
            assert_eq!(c, expected);
        }
    }
}
fn nested(context: &Context<'_>) {
    let edge = 2 * context.size();
    let a: Vec<_> = [context.rank(), context.rank() + context.size()]
        .into_iter()
        .map(pairs)
        .collect();
    let b = [vec![2, 3, -1, 4], vec![5, -2, 1, 3]];
    let c = execute_pairs_dense(
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
        &a,
        &b,
        vec![vec![10; 8]],
        3,
        |a, b, c, beta, l| {
            execute_pairs_dense(
                &Arithmetic::<i64>::new(),
                2,
                l,
                whole(),
                Panel {
                    comm: None,
                    outer: 1,
                    inner: 1,
                },
                whole(),
                a,
                b,
                c,
                beta,
                leaf,
            )
        },
    );
    let mut expected = vec![30; 8];
    for step in 0..edge {
        for b in &b {
            add_expected(&mut expected, step, b, 1);
        }
    }
    assert_eq!(c, vec![expected]);
}
fn run(context: &Context<'_>) {
    inputs(context);
    outputs(context);
    nested(context);
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
            "DIGIT / PASS distributed_sparse_2d_pairs: raw 3D keys, empty/stored-zero panels, A/B broadcast, dense cyclic reduction, strided output, layers and recursion; exact i64; world+parity"
        );
    }
    world.close();
    drop(universe);
}
