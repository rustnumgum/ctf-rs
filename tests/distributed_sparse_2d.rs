use ctf::{
    algebra::Arithmetic,
    context::Context,
    sparse_2d::{Layers, Panel, execute_ccsr_dense, execute_csr},
    sparse_formats::{Coo, Csr},
};
fn matrix(value: i64) -> Csr<i64> {
    Coo::new(
        1,
        1,
        if value == 0 {
            vec![]
        } else {
            vec![(1, 1, value)]
        },
    )
    .to_csr()
}
fn values(blocks: &[Csr<i64>]) -> Vec<i64> {
    blocks.iter().map(|b| b.values().iter().sum()).collect()
}
fn whole() -> Panel<'static, 'static> {
    Panel {
        comm: None,
        outer: 1,
        inner: 0,
    }
}
fn product_sum(
    a: &[Csr<i64>],
    b: &[Csr<i64>],
    mut c: Vec<Csr<i64>>,
    beta: i64,
    layers: Layers,
) -> Vec<Csr<i64>> {
    assert_eq!(layers, Layers { count: 1, index: 0 });
    for index in 0..a.len() {
        c[0] = a[index].multiply_sparse(
            &b[index],
            &1,
            if index == 0 { &beta } else { &1 },
            Some(&c[0]),
            &Arithmetic::<i64>::new(),
        );
    }
    c
}
fn input_panels(context: &Context<'_>) {
    let p = context.size();
    let rank = context.rank();
    let edge = 2 * p;
    let moving = Panel {
        comm: Some(context),
        outer: 2,
        inner: 1,
    };
    let mut data = Vec::new();
    for strip in 0..2 {
        for step in [rank, rank + p] {
            data.push(matrix(if strip == 1 && step % 3 == 0 {
                0
            } else {
                (step + strip + 1) as i64
            }));
        }
    }
    let weights = vec![matrix(1), matrix(2)];
    for layers in [
        Layers { count: 1, index: 0 },
        Layers { count: 2, index: 0 },
        Layers { count: 2, index: 1 },
    ] {
        let expected = 30
            + (0..edge)
                .filter(|step| step % layers.count == layers.index)
                .map(|step| {
                    (step + 1) as i64
                        + if step % 3 == 0 {
                            0
                        } else {
                            2 * (step + 2) as i64
                        }
                })
                .sum::<i64>();
        let c = execute_csr(
            &Arithmetic::<i64>::new(),
            edge,
            layers,
            moving,
            whole(),
            whole(),
            &data,
            &weights,
            vec![matrix(10)],
            3,
            product_sum,
        );
        assert_eq!(values(&c), vec![expected]);
        let c = execute_csr(
            &Arithmetic::<i64>::new(),
            edge,
            layers,
            whole(),
            moving,
            whole(),
            &weights,
            &data,
            vec![matrix(10)],
            3,
            product_sum,
        );
        assert_eq!(values(&c), vec![expected]);
    }
    let c = execute_csr(
        &Arithmetic::<i64>::new(),
        edge,
        Layers {
            count: 2 * edge,
            index: edge + 1,
        },
        moving,
        whole(),
        whole(),
        &data,
        &weights,
        vec![matrix(10)],
        3,
        |a, b, mut c, beta, layer| {
            assert_eq!(layer, Layers { count: 2, index: 1 });
            for index in 0..a.len() {
                c[0] = a[index].multiply_sparse(
                    &b[index],
                    &1,
                    if index == 0 { &beta } else { &1 },
                    Some(&c[0]),
                    &Arithmetic::<i64>::new(),
                );
            }
            c
        },
    );
    assert_eq!(values(&c), vec![38]);
    let c = execute_csr(
        &Arithmetic::<i64>::new(),
        edge,
        Layers {
            count: edge + 1,
            index: 1,
        },
        moving,
        whole(),
        whole(),
        &data,
        &weights,
        vec![matrix(10)],
        3,
        |a, b, mut c, beta, layer| {
            assert_eq!(
                layer,
                Layers {
                    count: edge + 1,
                    index: 1
                }
            );
            for index in 0..a.len() {
                c[0] = a[index].multiply_sparse(
                    &b[index],
                    &1,
                    if index == 0 { &beta } else { &1 },
                    Some(&c[0]),
                    &Arithmetic::<i64>::new(),
                );
            }
            c
        },
    );
    assert_eq!(
        values(&c),
        vec![
            30 + (0..edge)
                .map(|step| (step + 1) as i64
                    + if step % 3 == 0 {
                        0
                    } else {
                        2 * (step + 2) as i64
                    })
                .sum::<i64>()
        ]
    );
}
fn output_panels(context: &Context<'_>) {
    let p = context.size();
    let rank = context.rank();
    let edge = 2 * p;
    let input = Panel {
        comm: None,
        outer: 1,
        inner: 1,
    };
    let a: Vec<_> = (1..=edge).map(|x| matrix(x as i64)).collect();
    for outer in [1, 2] {
        for moving in [false, true] {
            let output = Panel {
                comm: if moving { Some(context) } else { None },
                outer,
                inner: 1,
            };
            let local_steps = if moving { 2 } else { edge };
            let c = execute_csr(
                &Arithmetic::<i64>::new(),
                edge,
                Layers { count: 1, index: 0 },
                input,
                whole(),
                output,
                &a,
                &[matrix(1)],
                vec![matrix(10); outer * local_steps],
                2,
                |a, _, mut c, beta, layer| {
                    assert_eq!(layer, Layers { count: 1, index: 0 });
                    for strip in 0..outer {
                        c[strip] = a[0].multiply_sparse(
                            &matrix(strip as i64 + 1),
                            &1,
                            &beta,
                            Some(&c[strip]),
                            &Arithmetic::<i64>::new(),
                        );
                    }
                    c
                },
            );
            let mut expected = Vec::new();
            for strip in 0..outer {
                for local in 0..local_steps {
                    let step = if moving { rank + local * p } else { local };
                    // Source sparse moving C adds old C unscaled; a stationary
                    // strided fresh panel replaces old C rather than dense-style beta.
                    expected.push(
                        (if moving {
                            10
                        } else if outer == 1 {
                            20
                        } else {
                            0
                        }) + (strip + 1) as i64
                            * (step + 1) as i64
                            * if moving { p as i64 } else { 1 },
                    );
                }
            }
            assert_eq!(values(&c), expected);
        }
    }
}
fn nested(context: &Context<'_>) {
    let p = context.size();
    let rank = context.rank();
    let edge = 2 * p;
    let a: Vec<_> = [rank, rank + p]
        .into_iter()
        .map(|i| matrix(i as i64 + 1))
        .collect();
    let c = execute_csr(
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
        &[matrix(1), matrix(2)],
        vec![matrix(10)],
        3,
        |a, b, c, beta, layer| {
            execute_csr(
                &Arithmetic::<i64>::new(),
                2,
                layer,
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
                product_sum,
            )
        },
    );
    assert_eq!(values(&c), vec![30 + 3 * (edge * (edge + 1) / 2) as i64]);
}
fn ccsr(context: &Context<'_>) {
    let p = context.size();
    let rank = context.rank();
    let edge = 2 * p;
    let a: Vec<_> = (0..edge)
        .map(|step| {
            matrix(if step % 2 == 0 { step as i64 + 1 } else { 0 })
                .to_coo()
                .to_ccsr()
        })
        .collect();
    let c = execute_ccsr_dense(
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
            comm: Some(context),
            outer: 1,
            inner: 1,
        },
        &a,
        &[vec![1]],
        vec![matrix(10).to_coo().to_ccsr(); 2],
        2,
        |a, b, mut c, beta, layer| {
            assert_eq!(layer, Layers { count: 1, index: 0 });
            c[0] = a[0].multiply_dense(1, &b[0], &1, &beta, Some(&c[0]), &Arithmetic::<i64>::new());
            c
        },
    );
    let actual: Vec<i64> = c.iter().map(|b| b.values().iter().sum()).collect();
    let expected: Vec<_> = [rank, rank + p]
        .into_iter()
        .map(|step| {
            10 + if step % 2 == 0 {
                p as i64 * (step as i64 + 1)
            } else {
                0
            }
        })
        .collect();
    assert_eq!(actual, expected);
    let a = vec![matrix(2).to_coo().to_ccsr()];
    let b: Vec<_> = [rank, rank + p]
        .into_iter()
        .map(|step| vec![step as i64 + 1])
        .collect();
    let c = execute_ccsr_dense(
        &Arithmetic::<i64>::new(),
        edge,
        Layers { count: 1, index: 0 },
        whole(),
        Panel {
            comm: Some(context),
            outer: 1,
            inner: 1,
        },
        whole(),
        &a,
        &b,
        vec![matrix(10).to_coo().to_ccsr()],
        3,
        |a, b, mut c, beta, layer| {
            assert_eq!(layer, Layers { count: 1, index: 0 });
            c[0] = a[0].multiply_dense(1, &b[0], &1, &beta, Some(&c[0]), &Arithmetic::<i64>::new());
            c
        },
    );
    assert_eq!(
        c[0].values().iter().sum::<i64>(),
        30 + (edge * (edge + 1)) as i64
    );
    let a: Vec<_> = [rank, rank + p]
        .into_iter()
        .map(|step| matrix(step as i64 + 1).to_coo().to_ccsr())
        .collect();
    let c = execute_ccsr_dense(
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
        &[vec![2]],
        vec![matrix(10).to_coo().to_ccsr()],
        3,
        |a, b, mut c, beta, _| {
            c[0] = a[0].multiply_dense(1, &b[0], &1, &beta, Some(&c[0]), &Arithmetic::<i64>::new());
            c
        },
    );
    assert_eq!(
        c[0].values().iter().sum::<i64>(),
        30 + (edge * (edge + 1)) as i64
    );
}
fn run(context: &Context<'_>) {
    input_panels(context);
    output_panels(context);
    nested(context);
    ccsr(context);
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
            "DIGIT / PASS distributed_sparse_2d: moving A/B/C, strided sparse output, variable payloads, CSR/CCSR reduction, layer subsets and recursive child; exact i64; world+parity"
        );
    }
    world.close();
    drop(universe);
}
