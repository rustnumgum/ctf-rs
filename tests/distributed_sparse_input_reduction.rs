use ctf::{algebra::Arithmetic, context::{Context, Runtime},
    mapping::{Distribution, Mapping, Topology}, sparse::SparseTensor, tensor::Tensor};
type Dense<'c, 'r> = Tensor<'c, 'r, Arithmetic<i64>>;
type Sparse<'c, 'r> = SparseTensor<'c, 'r, Arithmetic<i64>>;

fn source_distribution(c: &Context<'_>, shape: Vec<usize>, replicated: bool) -> Distribution {
    let topology = Topology::new(vec![c.size()]);
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    if !replicated {
        mappings[0].augment_physical(&topology, 0);
        mappings[0].augment_virtual(2 * c.size());
    }
    Distribution::new(shape, topology, mappings)
}
fn sparse<'c, 'r>(dense: &Dense<'c, 'r>) -> Sparse<'c, 'r> {
    let mut result = Sparse::new(dense.context(), dense.distribution().clone(), Arithmetic::new());
    let pairs: Vec<_> = dense.local_pairs().into_iter().filter(|(key, value)|
        *value != 0 && dense.distribution().owner(*key) == dense.context().rank()).collect();
    result.write_add(&pairs); result
}
fn run(c: &Context<'_>) {
    for repeated in [false, true] { for replicated in [false, true] { for x in [0, 2] {
        let (ia, ib, ic, sa, sb, sc) = if repeated {
            ("ixpkk", "kqqj", "ijj", vec![2,x,3,3,3], vec![3,2,2,2], vec![2,2,2])
        } else { ("ixpk", "kqj", "ij", vec![2,x,3,3], vec![3,2,2], vec![2,2]) };
        let mut a = Dense::new(c, source_distribution(c, sa, replicated), Arithmetic::new());
        let mut b = Dense::new(c, source_distribution(c, sb, replicated), Arithmetic::new());
        a.transform(|key, value| *value = key as i64 % 5 - 2);
        b.transform(|key, value| *value = key as i64 % 7 - 3);
        let aa = sparse(&a); let bb = sparse(&b);
        let mut initial = Dense::new(c, Distribution::cyclic(sc.clone(), c.size()), Arithmetic::new());
        initial.transform(|_, value| *value = 7);
        let mut ss = sparse(&initial); let mut sd = initial.clone();
        let mut ds = initial.clone(); let mut dd = initial.clone();
        let grid = if c.size() == 4 { [2,2] } else { [c.size(),1] };
        ss.contract_from(ic, &aa, ia, &bb, ib, grid, 3, 2).unwrap();
        dd.contract_from_sparse(ic, &aa, ia, &bb, ib, grid, 3, 2).unwrap();
        sd.contract_from_sparse_dense(ic, &aa, ia, &b, ib, grid, 3, 2).unwrap();
        ds.contract_from_dense_sparse(ic, &a, ia, &bb, ib, grid, 3, 2).unwrap();
        let keys: Vec<_> = (0..sc.iter().product()).collect();
        let expected: Vec<_> = keys.iter().map(|&key| {
            let i = key % 2; let j = key / 2 % 2;
            if repeated && j != key / 4 { return 7; }
            let mut total = 0;
            for k in 0..3 { for p in 0..3 { for q in 0..2 { for xi in 0..x {
                let ak = i + 2 * (xi + x * (p + 3 * (k + if repeated { 3*k } else { 0 })));
                let bk = if repeated { k + 3 * (q + 2 * (q + 2*j)) } else { k + 3 * (q + 2*j) };
                total += (ak as i64 % 5 - 2) * (bk as i64 % 7 - 3);
            } } } }
            14 + 3 * total
        }).collect();
        for actual in [ss.read(&keys), dd.read(&keys), sd.read(&keys), ds.read(&keys)] {
            assert_eq!(actual, expected, "repeated={repeated} replicated={replicated} x={x}");
        }
    } } }
}
fn main() {
    let runtime = Runtime::initialize(); let world = runtime.world(); run(&world);
    let child = world.split(Some((world.rank()%2) as i32), world.rank() as i32).unwrap();
    run(&child); child.close();
    if world.rank()==0 { println!("DIGIT / PASS distributed_sparse_input_reduction: A/B-only labels, four sparse/mixed paths, repeated indices, empty reduction, replicas, virtual blocks, world+parity; exact i64"); }
    world.close(); runtime.finalize();
}
