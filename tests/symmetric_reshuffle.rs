use ctf::{
    algebra::{Arithmetic, Complex, Group, Wire},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, AS, NS, SH, SY},
};
use std::{collections::BTreeMap, fmt::Debug};
fn layouts(c: &Context<'_>, shape: &[usize], links: &[Symmetry]) -> Vec<SymmetricDistribution> {
    let top = Topology::new(vec![c.size()]);
    let mut result = vec![SymmetricDistribution::new(
        Distribution::new(
            shape.to_vec(),
            top.clone(),
            vec![Mapping::Unmapped; shape.len()],
        ),
        links.to_vec(),
    )];
    for axis in 0..shape.len() {
        let mut maps = vec![Mapping::Unmapped; shape.len()];
        maps[axis].augment_physical(&top, 0);
        for map in &mut maps {
            map.augment_virtual(2 * c.size());
        }
        result.push(SymmetricDistribution::new(
            Distribution::new(shape.to_vec(), top.clone(), maps),
            links.to_vec(),
        ));
    }
    if c.size() == 4 && !shape.is_empty() {
        let top = Topology::new(vec![2, 2]);
        let mut maps = vec![Mapping::Unmapped; shape.len()];
        maps[0].augment_physical(&top, 1);
        for map in &mut maps {
            map.augment_virtual(4);
        }
        result.push(SymmetricDistribution::new(
            Distribution::new(shape.to_vec(), top, maps),
            links.to_vec(),
        ));
    }
    result
}
fn check<A: Group + Clone>(c: &Context<'_>, algebra: A, value: impl Fn(usize) -> A::Element)
where
    A::Element: Wire + Debug,
{
    let mut cases = Vec::new();
    for kind in [SY, AS, SH] {
        for n in [0, 1, 3] {
            cases.push((vec![n, n], vec![kind, NS]));
        }
        cases.push((vec![4, 4, 4], vec![kind, kind, NS]));
    }
    cases.push((vec![3, 3, 2, 2], vec![SY, NS, AS, NS]));
    cases.push((vec![], vec![]));
    for (shape, links) in cases {
        let distributions = layouts(c, &shape, &links);
        for old in &distributions {
            let mut input = SymmetricTensor::new(c, old.clone(), algebra.clone());
            input.transform(|key, x| *x = value(key));
            let before = input.local_storage().to_vec();
            for target in &distributions {
                let result = input.redistribute(target.clone());
                let expected: BTreeMap<_, _> = target
                    .local_pairs(c.rank())
                    .into_iter()
                    .map(|(offset, key)| (offset, value(key)))
                    .collect();
                for (offset, x) in result.local_storage().iter().enumerate() {
                    assert_eq!(
                        *x,
                        expected
                            .get(&offset)
                            .cloned()
                            .unwrap_or_else(|| algebra.zero())
                    );
                }
                assert_eq!(input.local_storage(), before);
            }
        }
    }
}
fn run(c: &Context<'_>) {
    check(c, Arithmetic::<i64>::new(), |key| key as i64 - 5);
    check(c, Arithmetic::<Complex<f64>>::new(), |key| {
        Complex::new(key as f64 - 5.0, 0.5 * key as f64)
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
            "DIGIT / PASS symmetric_reshuffle: exact SY/AS/SH/mixed groups, physical/virtual/replica changes, holes/padding, empty/scalar, integer/complex; world+parity"
        );
    }
    world.close();
    drop(universe);
}
