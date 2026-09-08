// Port of test/permute_multiworld.cxx: dense NS blocked read/write equations.
use ctf::{algebra::Arithmetic, context::Context, mapping::Distribution, tensor::Tensor};
fn block(n: usize, rank: usize, np: usize) -> (usize, usize) {
    let base = n / np;
    (
        base * rank + (n % np).min(rank),
        base + usize::from(rank < n % np),
    )
}
fn case(c: &Context<'_>, n: usize) {
    let rows = (1..=c.size())
        .filter(|&x| c.size() % x == 0 && x <= c.size() / x)
        .max()
        .unwrap();
    let cols = c.size() / rows;
    let (r0, nr) = block(n, c.rank() % rows, rows);
    let (c0, nc) = block(n, c.rank() / rows, cols);
    let maps = vec![
        (r0..r0 + nr).map(Some).collect(),
        (c0..c0 + nc).map(Some).collect(),
    ];
    let child = c.split(Some(c.rank() as i32), 0).unwrap();
    let mut parent = Tensor::new(
        c,
        Distribution::cyclic(vec![n, n], c.size()),
        Arithmetic::<f64>::new(),
    );
    parent.transform(|key, x| *x = key as f64);
    {
        let mut local = Tensor::new(
            &child,
            Distribution::cyclic(vec![nr, nc], 1),
            Arithmetic::<f64>::new(),
        );
        parent.gather_permuted_into(Some(&mut local), &maps, 1.0, 1.0);
        for (key, x) in local.local_pairs() {
            assert_eq!(x, ((key / nr + c0) * n + key % nr + r0) as f64);
        }
        local.transform(|key, x| *x = (n * n - ((key / nr + c0) * n + key % nr + r0)) as f64);
        parent.transform(|_, x| *x = 0.0);
        parent.scatter_permuted_from(Some(&local), &maps, 1.0, 1.0);
    }
    child.close();
    for (key, x) in parent.local_pairs() {
        assert!(x.is_finite() && (x - (n * n - key) as f64).abs() < 1e-9);
    }
}
fn masks(c: &Context<'_>) {
    let mut parent = Tensor::new(
        c,
        Distribution::cyclic(vec![4], c.size()),
        Arithmetic::<i64>::new(),
    );
    parent.transform(|key, x| *x = key as i64 + 1);
    let child = c.split((c.rank() == 0).then_some(0), 0);
    let maps = vec![vec![Some(3), None, Some(1)]];
    let mut local = child.as_ref().map(|child| {
        let mut t = Tensor::new(
            child,
            Distribution::cyclic(vec![3], 1),
            Arithmetic::<i64>::new(),
        );
        t.transform(|_, x| *x = 10);
        t
    });
    parent.gather_permuted_into(local.as_mut(), &maps, 2, 3);
    if let Some(t) = &local {
        assert_eq!(t.all_data(), vec![38, 10, 34]);
    }
    if let Some(t) = &mut local {
        t.transform(|key, x| *x = if key == 0 { 0 } else { 5 });
    }
    parent.transform(|_, x| *x = 7);
    parent.scatter_permuted_from(local.as_ref(), &maps, 2, 3);
    for (key, x) in parent.local_pairs() {
        assert_eq!(x, if key == 1 { 31 } else { 7 });
    }
    drop(local);
    if let Some(child) = child {
        child.close();
    }
}
fn run(c: &Context<'_>) {
    case(c, 5);
    case(c, 1);
    masks(c);
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
            "DIGIT / PASS upstream_permute_multiworld: NS blocked cross-world gather/scatter, nonuniform/empty regions, skip maps and sparse-of-dense scatter semantics; exact/abs<1e-9, world+parity"
        );
    }
    world.close();
    drop(universe);
}
