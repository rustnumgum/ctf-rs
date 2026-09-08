use ctf::{
    algebra::{Arithmetic, Complex, Ring, Wire},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    sparse::SparseTensor,
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{AS, NS, SH, SY},
    tensor::Tensor,
};
use std::{fmt::Debug, path::Path};

fn reset(c: &Context<'_>, path: &Path) {
    if c.rank() == 0 {
        std::fs::write(path, vec![0xabu8; 512]).unwrap();
    }
    c.barrier();
}
fn disk<E: Wire>(c: &Context<'_>, path: &Path, values: &[E]) {
    c.barrier();
    if c.rank() == 0 {
        let actual = std::fs::read(path).unwrap();
        let mut expected = vec![0xab; 13];
        for value in values {
            value.encode(&mut expected);
        }
        expected.resize(512, 0xab);
        assert_eq!(
            actual, expected,
            "offset, ordered values, preserved prefix and suffix"
        );
    }
    c.barrier();
}
fn dense_sparse<A: Ring + Clone>(
    c: &Context<'_>,
    path: &Path,
    algebra: A,
    value: impl Fn(usize) -> A::Element,
) where
    A::Element: Wire + Debug,
{
    for replicated in [false, true] {
        let top = Topology::new(vec![c.size()]);
        let mut mode = Mapping::Unmapped;
        if !replicated {
            mode.augment_physical(&top, 0);
        }
        mode.augment_virtual(2 * c.size());
        let distribution = Distribution::new(vec![3, 2], top, vec![mode, Mapping::Unmapped]);
        let expected: Vec<_> = (0..6).map(&value).collect();
        let mut input = Tensor::new(c, distribution.clone(), algebra.clone());
        input.transform(|key, x| *x = value(key));
        reset(c, path);
        input.write_dense_to_file(path, 13);
        disk(c, path, &expected);
        let mut restored = Tensor::new(
            c,
            Distribution::cyclic(vec![3, 2], c.size()),
            algebra.clone(),
        );
        restored.transform(|_, x| *x = algebra.one());
        restored.read_dense_from_file(path, 13);
        assert_eq!(restored.all_data(), expected);
        let mut sparse = SparseTensor::new(c, distribution, algebra.clone());
        sparse.read_dense_from_file(path, 13);
        let nonzeros: Vec<_> = expected
            .iter()
            .cloned()
            .enumerate()
            .filter(|(_, x)| x != &algebra.zero())
            .collect();
        assert_eq!(sparse.all_pairs(true), nonzeros);
        reset(c, path);
        sparse.write_dense_to_file(path, 13);
        disk(c, path, &expected);
    }
}
fn symmetry(c: &Context<'_>, path: &Path) {
    for kind in [SY, AS, SH] {
        let top = Topology::new(vec![c.size()]);
        let mut first = Mapping::Unmapped;
        first.augment_physical(&top, 0);
        first.augment_virtual(2 * c.size());
        let mut second = Mapping::Unmapped;
        second.augment_virtual(2 * c.size());
        let distribution = SymmetricDistribution::new(
            Distribution::new(vec![3, 3], top, vec![first, second]),
            vec![kind, NS],
        );
        let mut tensor = SymmetricTensor::new(c, distribution.clone(), Arithmetic::<i64>::new());
        tensor.transform(|k, x| *x = if k == 3 { 0 } else { k as i64 + 1 });
        let expected: Vec<_> = (0..9)
            .map(|k| {
                distribution.canonicalize(k).map_or(0, |(key, sign)| {
                    (if key == 3 { 0 } else { key as i64 + 1 }) * sign as i64
                })
            })
            .collect();
        reset(c, path);
        tensor.write_dense_to_file(path, 13);
        disk(c, path, &expected);
        tensor.transform(|_, x| *x = 99);
        tensor.read_dense_from_file(path, 13);
        assert_eq!(tensor.all_data(true), expected);
        // Noncanonical file entries must not be accumulated into packed orbits.
        let mut ns = Tensor::new(
            c,
            Distribution::cyclic(vec![3, 3], c.size()),
            Arithmetic::<i64>::new(),
        );
        ns.transform(|key, x| *x = key as i64 + 10);
        ns.write_dense_to_file(path, 13);
        tensor.read_dense_from_file(path, 13);
        for (key, x) in tensor.local_pairs() {
            assert_eq!(x, key as i64 + 10);
        }
    }
}
fn run(c: &Context<'_>, token: u64, scope: usize) {
    let path = std::env::temp_dir().join(format!("ctf-binary-{token}-{scope}.bin"));
    macro_rules! real {($($t:ty),*)=>{$(dense_sparse(c,&path,Arithmetic::<$t>::new(),|key|key as $t-2 as $t);)*};}
    real!(i8, i16, i32, i64, f32, f64);
    dense_sparse(c, &path, Arithmetic::<Complex<f32>>::new(), |key| {
        Complex::new(key as f32 - 2.0, 0.5 * (key as f32 - 2.0))
    });
    dense_sparse(c, &path, Arithmetic::<Complex<f64>>::new(), |key| {
        Complex::new(key as f64 - 2.0, 0.5 * (key as f64 - 2.0))
    });
    symmetry(c, &path);
    for shape in [vec![1], vec![0, 2], vec![]] {
        let mut tiny = Tensor::new(
            c,
            Distribution::cyclic(shape, c.size()),
            Arithmetic::<i64>::new(),
        );
        tiny.transform(|_, x| *x = 7);
        reset(c, &path);
        tiny.write_dense_to_file(&path, 13);
        tiny.transform(|_, x| *x = 9);
        tiny.read_dense_from_file(&path, 13);
        for (_, x) in tiny.local_pairs() {
            assert_eq!(x, 7);
        }
    }
    c.barrier();
    if c.rank() == 0 {
        std::fs::remove_file(path).unwrap();
    }
    c.barrier();
}
fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = ctf::context::Context::world(&universe);
    let mut token = [std::process::id() as u64];
    world.broadcast(0, &mut token);
    run(&world, token[0], 0);
    let color = world.rank() % 2;
    let child = world
        .split(Some(color as i32), world.rank() as i32)
        .unwrap();
    run(&child, token[0], color + 1);
    child.close();
    if world.rank() == 0 {
        println!(
            "DIGIT / PASS binary_io: exact typed bytes, offsets, dense/sparse/SY/AS/SH overwrite, virtual/replicated layouts, empty/scalar domains; world+parity"
        );
    }
    world.close();
    drop(universe);
}
