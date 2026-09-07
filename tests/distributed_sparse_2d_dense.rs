use ctf::{algebra::Arithmetic, context::{Context, Runtime},
    sparse_2d::{execute_csr_dense, execute_csr_sparse_dense, Layers, Panel},
    sparse_formats::{Coo, Csr}, sparse_function_kernel};

fn matrix(value: i64) -> Csr<i64> {
    Coo::new(1, 1, if value == 0 { vec![] } else { vec![(1, 1, value)] }).to_csr()
}
fn whole() -> Panel<'static, 'static> { Panel { comm: None, outer: 1, inner: 0 } }
fn dense_leaf(a: &[Csr<i64>], b: &[Vec<i64>], mut c: Vec<Vec<i64>>,
    beta: i64, _: Layers) -> Vec<Vec<i64>> {
    for i in 0..a.len() {
        a[i].multiply_dense(b[i].len(), &b[i], &1, if i == 0 { &beta } else { &1 },
            &mut c[0], &Arithmetic::<i64>::new());
    }
    c
}
fn sparse_leaf(a: &[Csr<i64>], b: &[Csr<i64>], mut c: Vec<Vec<i64>>,
    beta: i64, _: Layers) -> Vec<Vec<i64>> {
    for i in 0..a.len() {
        sparse_function_kernel::csr_sparse(&Arithmetic::<i64>::new(), &a[i], &b[i],
            &mut c[0], &1, if i == 0 { &beta } else { &1 }, |x, y| x * y);
    }
    c
}
fn inputs(context: &Context<'_>) {
    let p = context.size(); let rank = context.rank(); let edge = 2 * p;
    let moving = Panel { comm: Some(context), outer: 2, inner: 1 };
    let value = |strip: usize, step: usize| if strip == 1 && step % 3 == 0 { 0 }
        else { (strip + step + 1) as i64 };
    let mut sparse = Vec::new(); let mut dense = Vec::new();
    for strip in 0..2 { for step in [rank, rank + p] {
        sparse.push(matrix(value(strip, step))); dense.push(vec![value(strip, step)]);
    }}
    let weights = [matrix(1), matrix(2)];
    let dense_weights = [vec![1], vec![2]];
    for layers in [Layers { count: 1, index: 0 }, Layers { count: 2, index: 0 },
        Layers { count: 2, index: 1 }, Layers { count: 2 * edge, index: edge + 1 },
        Layers { count: edge + 1, index: 1 }] {
        let steps: Vec<_> = if layers.count <= edge {
            (layers.index..edge).step_by(layers.count).collect()
        } else if layers.count % edge == 0 { vec![layers.index % edge] }
        else { (0..edge).collect() };
        let expected = vec![vec![30 + steps.iter().map(|&s| value(0, s) + 2 * value(1, s)).sum::<i64>()]];
        let next = if layers.count <= edge { Layers { count: 1, index: 0 } }
            else if layers.count % edge == 0 { Layers { count: layers.count / edge, index: layers.index / edge } }
            else { layers };
        let c = execute_csr_dense(&Arithmetic::<i64>::new(), edge, layers,
            moving, whole(), whole(), &sparse, &dense_weights, vec![vec![10]], 3,
            |a,b,c,beta,l| { assert_eq!(l, next); dense_leaf(a,b,c,beta,l) });
        assert_eq!(c, expected);
        let c = execute_csr_dense(&Arithmetic::<i64>::new(), edge, layers,
            whole(), moving, whole(), &weights, &dense, vec![vec![10]], 3, dense_leaf);
        assert_eq!(c, expected);
        for move_a in [false, true] {
            let (ap,bp,a,b) = if move_a { (moving,whole(),&sparse[..],&weights[..]) }
                else { (whole(),moving,&weights[..],&sparse[..]) };
            let c = execute_csr_sparse_dense(&Arithmetic::<i64>::new(), edge, layers,
                ap,bp,whole(),a,b,vec![vec![10]],3,
                |a,b,c,beta,l| { assert_eq!(l, next); sparse_leaf(a,b,c,beta,l) });
            assert_eq!(c, expected);
        }
    }
}
fn outputs(context: &Context<'_>) {
    let p = context.size(); let rank = context.rank(); let edge = 2 * p;
    let a: Vec<_> = (0..edge).map(|s| matrix(if s % 3 == 0 { 0 } else { s as i64 + 1 })).collect();
    for outer in [1, 2] { for inner in [1, 2] { for moving in [false, true] {
        let local_steps = if moving { 2 } else { edge };
        let output = Panel { comm: if moving { Some(context) } else { None }, outer, inner };
        let input = Panel { comm: None, outer: 1, inner: 1 };
        let b: Vec<_> = (0..outer*inner).map(|s| vec![s as i64 + 1, -(s as i64 + 2)]).collect();
        let bs: Vec<_> = b.iter().map(|v| Coo::new(1,2,vec![(1,1,v[0]),(1,2,v[1])]).to_csr()).collect();
        for beta in [0, 2] {
            let old = vec![vec![10, -10]; outer * inner * local_steps];
            let c = execute_csr_dense(&Arithmetic::<i64>::new(), edge, Layers { count: 1, index: 0 },
                input,whole(),output,&a,&b,old.clone(),beta,|a,b,mut c,beta,_| {
                    for s in 0..c.len() { a[0].multiply_dense(2,&b[s],&1,&beta,&mut c[s],&Arithmetic::<i64>::new()); } c
                });
            let cs = execute_csr_sparse_dense(&Arithmetic::<i64>::new(), edge, Layers { count: 1, index: 0 },
                input,whole(),output,&a,&bs,old,beta,|a,b,mut c,beta,_| {
                    for s in 0..c.len() { sparse_function_kernel::csr_sparse(&Arithmetic::<i64>::new(),
                        &a[0],&b[s],&mut c[s],&1,&beta,|x,y| x*y); } c
                });
            let mut expected = Vec::new();
            for strip in 0..outer { for local in 0..local_steps { for block in 0..inner {
                let step = if moving { rank + local*p } else { local };
                let v = if step%3 == 0 { 0 } else { step as i64+1 } * if moving { p as i64 } else { 1 };
                let weight = (strip*inner+block) as i64;
                expected.push(vec![10*beta+v*(weight+1), -10*beta-v*(weight+2)]);
            }}}
            assert_eq!(c, expected); assert_eq!(cs, expected);
        }
    }}}
}
fn nested(context: &Context<'_>) {
    let edge = 2*context.size();
    let a: Vec<_> = [context.rank(), context.rank()+context.size()].into_iter().map(|s| matrix(s as i64+1)).collect();
    let c = execute_csr_dense(&Arithmetic::<i64>::new(),edge,Layers { count:1,index:0 },
        Panel { comm:Some(context),outer:1,inner:1 },whole(),whole(),&a,&[vec![1],vec![2]],vec![vec![10]],3,
        |a,b,c,beta,l| execute_csr_dense(&Arithmetic::<i64>::new(),2,l,whole(),
            Panel { comm:None,outer:1,inner:1 },whole(),a,b,c,beta,dense_leaf));
    assert_eq!(c,vec![vec![30+3*(edge*(edge+1)/2) as i64]]);
}
fn run(context: &Context<'_>) { inputs(context); outputs(context); nested(context); }
fn main() {
    let runtime=Runtime::initialize(); let world=runtime.world(); run(&world);
    let child=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();
    run(&child); child.close();
    if world.rank()==0 { println!("DIGIT / PASS distributed_sparse_2d_dense: CSR/dense and CSR/CSR inputs, dense output, cyclic reduction, strips, layer subsets, recursive child; exact i64; world+parity"); }
    world.close(); runtime.finalize();
}
