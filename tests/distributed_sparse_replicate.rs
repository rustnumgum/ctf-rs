//! Sparse replicate/virtual CPU kernels with native CSR/CCSR output reduction.
use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::Topology,
    sparse_contraction_comm::{replicated_ccsr_dense, replicated_csr},
    sparse_formats::Coo,
};

fn csr(context: &Context<'_>) {
    for beta in [0, 3] {
        let mut a: Vec<_> = [vec![(1, 1, 2), (2, 1, 0)], vec![(1, 1, 3)]]
            .into_iter()
            .map(|entries| {
                Coo::new(2, 1, if context.rank() == 0 { entries } else { vec![] }).to_csr()
            })
            .collect();
        let r = context.rank() as i64;
        let mut b = vec![
            Coo::new(1, 2, vec![(1, 1, r + 1), (1, 2, 1)]).to_csr(),
            Coo::new(1, 2, vec![(1, 2, r + 2)]).to_csr(),
        ];
        let old_b = b.clone();
        let c = vec![Coo::new(2, 2, vec![(2, 2, 5)]).to_csr()];
        let result = replicated_csr(
            &Arithmetic::<i64>::new(),
            &[context],
            &[],
            &[context],
            &[2],
            [&[0], &[0], &[]],
            &mut a,
            &mut b,
            c,
            &2,
            &beta,
        );
        assert_eq!(result.is_some(), context.rank() == 0);
        if let Some(result) = result {
            let p = context.size() as i64;
            assert_eq!(
                result[0].to_coo().entries(),
                &[
                    (1, 1, 2 * p * (p + 1)),
                    (1, 2, 4 * p + 3 * p * (p + 3)),
                    (2, 1, 0),
                    (2, 2, 5 * beta)
                ]
            );
        }
        if context.rank() != 0 {
            assert!(a.iter().all(|matrix| matrix.values().is_empty()));
        }
        assert_eq!(b, old_b);
    }
}

fn ccsr(context: &Context<'_>) {
    let mut a = vec![
        Coo::new(2, 1, vec![(1, 1, context.rank() as i64 + 1), (2, 1, 0)]).to_ccsr(),
        Coo::new(2, 1, vec![(1, 1, 2)]).to_ccsr(),
    ];
    let old_a = a.clone();
    let mut b = if context.rank() == 0 {
        vec![vec![2, 3], vec![4, 5]]
    } else {
        vec![vec![0, 0], vec![0, 0]]
    };
    let c = vec![Coo::new(2, 2, vec![(2, 2, 5)]).to_ccsr()];
    let result = replicated_ccsr_dense(
        &Arithmetic::<i64>::new(),
        &[],
        &[context],
        &[context],
        &[2],
        [&[0], &[0], &[]],
        &mut a,
        &mut b,
        c,
        &2,
        &3,
    );
    assert_eq!(result.is_some(), context.rank() == 0);
    if let Some(result) = result {
        let p = context.size() as i64;
        assert_eq!(
            result[0].to_coo().entries(),
            &[
                (1, 1, 2 * p * (p + 1) + 16 * p),
                (1, 2, 3 * p * (p + 1) + 20 * p),
                (2, 1, 0),
                (2, 2, 15)
            ]
        );
    }
    assert_eq!(a, old_a);
    if context.rank() != 0 {
        assert!(b.iter().flatten().all(|&value| value == 0));
    }
}

fn multiple_fibers(context: &Context<'_>) {
    let topology = Topology::new(if context.size() == 4 {
        vec![2, 2]
    } else {
        vec![context.size(), 1]
    });
    let row = topology.fiber(context, 0);
    let column = topology.fiber(context, 1);
    let mut a = vec![Coo::new(1, 1, vec![(1, 1, context.rank() as i64 + 1)]).to_csr()];
    let mut b = vec![Coo::new(1, 1, vec![(1, 1, 2)]).to_csr()];
    let c = vec![Coo::new(1, 1, vec![(1, 1, 5)]).to_csr()];
    let result = replicated_csr(
        &Arithmetic::<i64>::new(),
        &[],
        &[],
        &[&row, &column],
        &[],
        [&[], &[], &[]],
        &mut a,
        &mut b,
        c,
        &2,
        &3,
    );
    assert_eq!(result.is_some(), context.rank() == 0);
    if let Some(result) = result {
        let p = context.size() as i64;
        assert_eq!(
            result[0].to_coo().entries(),
            &[(1, 1, 15 + 2 * p * (p + 1))]
        );
    }
    row.close();
    column.close();
}

fn ccsr_broadcast(context: &Context<'_>) {
    let mut a = vec![
        Coo::new(
            2,
            1,
            if context.rank() == 0 {
                vec![(1, 1, 2), (2, 1, 0)]
            } else {
                vec![]
            },
        )
        .to_ccsr(),
    ];
    let mut b = vec![vec![3, 4]];
    let c = vec![Coo::new(2, 2, vec![]).to_ccsr()];
    let result = replicated_ccsr_dense(
        &Arithmetic::<i64>::new(),
        &[context],
        &[],
        &[],
        &[],
        [&[], &[], &[]],
        &mut a,
        &mut b,
        c,
        &2,
        &0,
    )
    .unwrap();
    assert_eq!(
        result[0].to_coo().entries(),
        &[(1, 1, 12), (1, 2, 16), (2, 1, 0), (2, 2, 0)]
    );
    if context.rank() != 0 {
        assert!(a.is_empty());
    }
}

fn local() {
    let mut a = vec![Coo::new(1, 1, vec![(1, 1, 2)]).to_csr()];
    let mut b = vec![Coo::new(1, 1, vec![(1, 1, 3)]).to_csr()];
    let c = vec![Coo::new(1, 1, vec![(1, 1, 5)]).to_csr()];
    let result = replicated_csr(
        &Arithmetic::<i64>::new(),
        &[],
        &[],
        &[],
        &[],
        [&[], &[], &[]],
        &mut a,
        &mut b,
        c,
        &2,
        &3,
    )
    .unwrap();
    assert_eq!(result[0].to_coo().entries(), &[(1, 1, 27)]);
}
fn virtual_outputs(context: &Context<'_>) {
    let mut a: Vec<_> = [2, 3]
        .into_iter()
        .map(|v| Coo::new(1, 1, vec![(1, 1, v)]).to_csr())
        .collect();
    let mut b: Vec<_> = [4, 5]
        .into_iter()
        .map(|v| {
            Coo::new(
                1,
                1,
                if context.rank() == 0 {
                    vec![(1, 1, v)]
                } else {
                    vec![]
                },
            )
            .to_csr()
        })
        .collect();
    let c = vec![Coo::new(1, 1, vec![(1, 1, 2)]).to_csr(); 4];
    let result = replicated_csr(
        &Arithmetic::<i64>::new(),
        &[],
        &[context],
        &[context],
        &[2, 2],
        [&[0], &[1], &[0, 1]],
        &mut a,
        &mut b,
        c,
        &2,
        &3,
    );
    assert_eq!(result.is_some(), context.rank() == 0);
    if let Some(blocks) = result {
        for (offset, block) in blocks.iter().enumerate() {
            assert_eq!(
                block.to_coo().entries(),
                &[(
                    1,
                    1,
                    6 + 2
                        * (2 + offset as i64 % 2)
                        * (4 + offset as i64 / 2)
                        * context.size() as i64
                )]
            );
        }
    }
    if context.rank() != 0 {
        assert!(b.is_empty());
    }
}
fn run(context: &Context<'_>) {
    csr(context);
    ccsr(context);
    ccsr_broadcast(context);
    multiple_fibers(context);
    local();
    virtual_outputs(context);
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
            "DIGIT / PASS distributed_sparse_replicate: CSR/CCSR broadcasts, virtual beta-once, output-fiber reductions, replica cleanup, stored zeros; exact i64; world+parity"
        );
    }
    world.close();
    drop(universe);
}
