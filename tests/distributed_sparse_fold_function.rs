use ctf::{
    algebra::Arithmetic, context::Context, mapping::Distribution, sparse::SparseTensor,
    tensor::Tensor,
};
fn run(c: &Context<'_>) {
    let ae = [(0, 0_i64), (1, 2), (4, -1), (7, 3)];
    let be = [(0, 0_i64), (1, 2), (6, 3), (11, -1)];
    for repeated in [false, true] {
        let (sa, ia, sc, ic) = if repeated {
            (vec![2, 2, 2, 2], "ikkl", vec![3, 2, 2, 2], "jiil")
        } else {
            (vec![2, 2, 2], "ikl", vec![3, 2, 2], "jil")
        };
        let da = Distribution::cyclic(sa, c.size());
        let db = Distribution::cyclic(vec![2, 3, 2], c.size());
        let dc = Distribution::cyclic(sc, c.size());
        let mut a = SparseTensor::new(c, da.clone(), Arithmetic::<i64>::new());
        let mut pairs: Vec<_> = ae
            .iter()
            .map(|&(key, v)| {
                let (i, k, l) = (key % 2, key / 2 % 2, key / 4);
                (
                    if repeated {
                        i + 2 * (k + 2 * (k + 2 * l))
                    } else {
                        key
                    },
                    v,
                )
            })
            .collect();
        if repeated {
            pairs.push((2, 99));
        }
        a.write_add(
            &pairs
                .into_iter()
                .filter(|(key, _)| da.owner(*key) == c.rank())
                .collect::<Vec<_>>(),
        );
        let mut b = SparseTensor::new(c, db.clone(), Arithmetic::<i64>::new());
        b.write_add(
            &be.iter()
                .copied()
                .filter(|(key, _)| db.owner(*key) == c.rank())
                .collect::<Vec<_>>(),
        );
        let mut bd = Tensor::new(c, db, Arithmetic::<i64>::new());
        bd.transform(|key, value| {
            *value = be.iter().find(|(k, _)| *k == key).map_or(0, |(_, v)| *v)
        });
        let count = dc.shape.iter().product();
        let keys: Vec<_> = (0..count).collect();
        let mut ss = SparseTensor::new(c, dc.clone(), Arithmetic::<i64>::new());
        ss.write_add(
            &keys
                .iter()
                .filter(|&&key| dc.owner(key) == c.rank())
                .map(|&key| (key, 7))
                .collect::<Vec<_>>(),
        );
        let mut dd = Tensor::new(c, dc.clone(), Arithmetic::<i64>::new());
        dd.transform(|_, value| *value = 7);
        let mut sd = dd.clone();
        let grid = if c.size() == 4 { [2, 2] } else { [c.size(), 1] };
        ss.contract_from_function(ic, &a, ia, &b, "kjl", grid, 1, 2, |a, b| a + b + 1)
            .unwrap();
        dd.contract_from_sparse_function(ic, &a, ia, &b, "kjl", grid, 1, 2, |a, b| a + b + 1)
            .unwrap();
        sd.contract_from_sparse_dense_function(ic, &a, ia, &bd, "kjl", grid, 1, 2, |a, b| {
            a + b + 1
        })
        .unwrap();
        let expected = |dense_b: bool| {
            keys.iter()
                .map(|&key| {
                    let j = key % 3;
                    let i = key / 3 % 2;
                    let l = if repeated { key / 12 } else { key / 6 };
                    if repeated && i != key / 6 % 2 {
                        return 7;
                    }
                    let mut total = 14;
                    for k in 0..2 {
                        if let Some((_, av)) =
                            ae.iter().find(|(key, _)| *key == i + 2 * (k + 2 * l))
                        {
                            let bv = be.iter().find(|(key, _)| *key == k + 2 * (j + 3 * l));
                            if dense_b || bv.is_some() {
                                total += av + bv.map_or(0, |(_, v)| *v) + 1;
                            }
                        }
                    }
                    total
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(ss.read(&keys), expected(false));
        assert_eq!(dd.read(&keys), expected(false));
        assert_eq!(sd.read(&keys), expected(true));
        assert_eq!(ss.distribution(), &dc);
        assert_eq!(dd.distribution(), &dc);
        assert_eq!(sd.distribution(), &dc);
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
            "DIGIT / PASS distributed_sparse_fold_function: high-order custom sparse/dense contractions, batches, axis permutation, repeated input/output indices, world+parity; exact i64"
        );
    }
    world.close();
    drop(universe);
}
