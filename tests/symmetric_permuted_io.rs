use ctf::{
    algebra::Arithmetic,
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetric_tensor::SymmetricTensor,
    symmetry::Symmetry::{self, AS, NS, SH, SY},
    tensor::Tensor,
};
use std::collections::BTreeMap;
fn layout(np: usize, kind: Symmetry, axis: usize) -> SymmetricDistribution {
    let top = Topology::new(vec![np]);
    let mut maps = vec![Mapping::Unmapped; 2];
    maps[axis].augment_physical(&top, 0);
    for map in &mut maps {
        map.augment_virtual(2 * np);
    }
    SymmetricDistribution::new(Distribution::new(vec![3, 3], top, maps), vec![kind, NS])
}
fn mapped(key: usize, maps: &[Vec<Option<usize>>]) -> Option<usize> {
    Some(maps[0][key % 3]? + 3 * maps[1][key / 3]?)
}
fn values(key: usize) -> i64 {
    if key == 3 { 0 } else { key as i64 + 1 }
}
fn expected_gather(
    source: Option<&SymmetricDistribution>,
    dest: Option<&SymmetricDistribution>,
    maps: &[Vec<Option<usize>>],
) -> BTreeMap<usize, i64> {
    let mut result = BTreeMap::new();
    for key in 0..9 {
        if let Some(parent_key) = mapped(key, maps) {
            let incoming = match source {
                Some(d) => d
                    .canonicalize(parent_key)
                    .map_or(0, |(key, sign)| sign as i64 * (key as i64 + 1)),
                None => parent_key as i64 + 1,
            };
            let canonical = match dest {
                Some(d) => d.canonicalize(key),
                None => Some((key, 1)),
            };
            if let Some((key, sign)) = canonical {
                *result.entry(key).or_insert(15) += 2 * sign as i64 * incoming;
            }
        }
    }
    result
}
fn expected_scatter(
    source: Option<&SymmetricDistribution>,
    dest: Option<&SymmetricDistribution>,
    maps: &[Vec<Option<usize>>],
) -> BTreeMap<usize, i64> {
    let mut result = BTreeMap::new();
    for key in 0..9 {
        if source.is_some_and(|d| d.canonicalize(key) != Some((key, 1))) || values(key) == 0 {
            continue;
        }
        if let Some(parent_key) = mapped(key, maps) {
            let canonical = match dest {
                Some(d) => d.canonicalize(parent_key),
                None => Some((parent_key, 1)),
            };
            if let Some((parent_key, sign)) = canonical {
                *result.entry(parent_key).or_insert(21) += 2 * sign as i64 * values(key);
            }
        }
    }
    result
}
fn run(c: &Context<'_>) {
    for root_only in [false, true] {
        let child = c.split(
            (!root_only || c.rank() == 0).then_some(0),
            -(c.rank() as i32),
        );
        let np = if root_only { 1 } else { c.size() };
        for kind in [SY, AS, SH] {
            for masked in [false, true] {
                let maps = vec![vec![Some(2), if masked { None } else { Some(1) }, Some(0)]; 2];
                let parent_dist = layout(c.size(), kind, 0);
                let child_dist = layout(np, kind, 1);
                let mut parent =
                    SymmetricTensor::new(c, parent_dist.clone(), Arithmetic::<i64>::new());
                parent.transform(|key, x| *x = key as i64 + 1);
                let mut dense = Tensor::new(
                    c,
                    Distribution::cyclic(vec![3, 3], c.size()),
                    Arithmetic::<i64>::new(),
                );
                dense.transform(|key, x| *x = key as i64 + 1);
                let mut sym_child = child.as_ref().map(|child| {
                    let mut t =
                        SymmetricTensor::new(child, child_dist.clone(), Arithmetic::<i64>::new());
                    t.transform(|_, x| *x = 5);
                    t
                });
                let mut dense_child = child.as_ref().map(|child| {
                    let mut t = Tensor::new(
                        child,
                        Distribution::cyclic(vec![3, 3], np),
                        Arithmetic::<i64>::new(),
                    );
                    t.transform(|_, x| *x = 5);
                    t
                });
                parent.gather_permuted_into(sym_child.as_mut(), &maps, 2, 3);
                if let Some(t) = &sym_child {
                    let expected = expected_gather(Some(&parent_dist), Some(&child_dist), &maps);
                    for (key, x) in t.local_pairs() {
                        assert_eq!(x, *expected.get(&key).unwrap_or(&5));
                    }
                }
                parent.gather_permuted_into_dense(dense_child.as_mut(), &maps, 2, 3);
                if let Some(t) = &dense_child {
                    let expected = expected_gather(Some(&parent_dist), None, &maps);
                    for (key, x) in t.local_pairs() {
                        assert_eq!(x, *expected.get(&key).unwrap_or(&5));
                    }
                }
                if let Some(t) = &mut sym_child {
                    t.transform(|_, x| *x = 5);
                }
                dense.gather_permuted_into_symmetric(sym_child.as_mut(), &maps, 2, 3);
                if let Some(t) = &sym_child {
                    let expected = expected_gather(None, Some(&child_dist), &maps);
                    for (key, x) in t.local_pairs() {
                        assert_eq!(x, *expected.get(&key).unwrap_or(&5));
                    }
                }
                if let Some(t) = &mut sym_child {
                    t.transform(|key, x| *x = values(key));
                }
                if let Some(t) = &mut dense_child {
                    t.transform(|key, x| *x = values(key));
                }
                parent.transform(|_, x| *x = 7);
                parent.scatter_permuted_from(sym_child.as_ref(), &maps, 2, 3);
                let expected = expected_scatter(Some(&child_dist), Some(&parent_dist), &maps);
                for (key, x) in parent.local_pairs() {
                    assert_eq!(x, *expected.get(&key).unwrap_or(&7));
                }
                parent.transform(|_, x| *x = 7);
                parent.scatter_permuted_from_dense(dense_child.as_ref(), &maps, 2, 3);
                let expected = expected_scatter(None, Some(&parent_dist), &maps);
                for (key, x) in parent.local_pairs() {
                    assert_eq!(x, *expected.get(&key).unwrap_or(&7));
                }
                dense.transform(|_, x| *x = 7);
                dense.scatter_permuted_from_symmetric(sym_child.as_ref(), &maps, 2, 3);
                let expected = expected_scatter(Some(&child_dist), None, &maps);
                for (key, x) in dense.local_pairs() {
                    assert_eq!(x, *expected.get(&key).unwrap_or(&7));
                }
            }
        }
        if let Some(child) = child {
            child.close();
        }
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
            "DIGIT / PASS symmetric_permuted_io: six mixed/packed paths, signed full-orbit gather, canonical scatter, skipped coords, beta-once, reversed/root-only subworlds; exact world+parity"
        );
    }
    world.close();
    drop(universe);
}
