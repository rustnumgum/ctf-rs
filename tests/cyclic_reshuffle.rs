use ctf::{
    algebra::{Arithmetic, Complex, Monoid, Wire},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    tensor::Tensor,
};
use std::fmt::Debug;
#[derive(Clone, Debug, PartialEq)]
struct Word([u8; 3]);
impl Wire for Word {
    const WIDTH: usize = 3;
    fn encode(&self, bytes: &mut Vec<u8>) {
        bytes.extend(self.0);
    }
    fn decode(bytes: &[u8]) -> Self {
        Self(bytes.try_into().unwrap())
    }
}
#[derive(Clone)]
struct Xor;
impl Monoid for Xor {
    type Element = Word;
    fn zero(&self) -> Word {
        Word([0; 3])
    }
    fn add(&self, a: &Word, b: &Word) -> Word {
        Word(std::array::from_fn(|i| a.0[i] ^ b.0[i]))
    }
}
fn layouts(c: &Context<'_>, shape: &[usize]) -> Vec<Distribution> {
    let top = Topology::new(vec![c.size()]);
    let mut distributions = vec![
        Distribution::cyclic(shape.to_vec(), c.size()),
        Distribution::new(
            shape.to_vec(),
            top.clone(),
            vec![Mapping::Unmapped; shape.len()],
        ),
    ];
    for axis in 0..shape.len() {
        let mut maps = vec![Mapping::Unmapped; shape.len()];
        maps[axis].augment_physical(&top, 0);
        maps[axis].augment_virtual(2 * c.size());
        for (other, map) in maps.iter_mut().enumerate() {
            if other != axis {
                map.augment_virtual(2);
            }
        }
        distributions.push(Distribution::new(shape.to_vec(), top.clone(), maps));
    }
    if c.size() == 4 && !shape.is_empty() {
        let top = Topology::new(vec![2, 2]);
        let mut maps = vec![Mapping::Unmapped; shape.len()];
        maps[0].augment_physical(&top, 1);
        maps[0].augment_virtual(4);
        distributions.push(Distribution::new(shape.to_vec(), top, maps));
    }
    distributions
}
fn check<A: Monoid + Clone>(c: &Context<'_>, algebra: A, value: impl Fn(usize) -> A::Element)
where
    A::Element: Wire + Debug,
{
    for shape in [vec![3, 5], vec![1, 2], vec![0, 3], vec![]] {
        let distributions = layouts(c, &shape);
        for old in &distributions {
            let mut tensor = Tensor::new(c, old.clone(), algebra.clone());
            tensor.transform(|key, x| *x = value(key));
            for target in &distributions {
                tensor.redistribute(target.clone());
                assert_eq!(tensor.distribution(), target);
                for (offset, x) in tensor.local_storage().iter().enumerate() {
                    let expected = target
                        .global_key(c.rank(), offset)
                        .map_or_else(|| algebra.zero(), &value);
                    assert_eq!(*x, expected, "rank {}, local offset {offset}", c.rank());
                }
            }
        }
    }
}
fn run(c: &Context<'_>) {
    check(c, Arithmetic::<i8>::new(), |key| key as i8 - 5);
    check(c, Arithmetic::<bool>::new(), |key| key % 3 == 1);
    check(c, Arithmetic::<Complex<f64>>::new(), |key| {
        Complex::new(key as f64 + 0.5, -(key as f64))
    });
    check(c, Xor, |key| {
        Word([key as u8, key as u8 + 1, 2 * key as u8])
    });
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
            "DIGIT / PASS cyclic_reshuffle: exact values/padding/replicas, virtual phase changes, empty/scalar, i8/bool/complex/non-Copy Wire; world+parity"
        );
    }
    world.close();
    drop(universe);
}
