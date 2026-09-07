//! Offset-only reshuffle plans for compressed symmetric storage.
//!
//! Canonical coordinates are generated directly from symmetry bounds; the
//! rectangular tensor domain is never expanded or filtered after the fact.

use crate::{
    cyclic_reshuffle::Plan,
    mapping::Distribution,
    symmetry::Symmetry,
    symmetric_distribution::SymmetricDistribution,
};

pub(crate) fn plan(
    old: &SymmetricDistribution,
    new: &SymmetricDistribution,
    rank: usize,
) -> Plan {
    assert_eq!(old.distribution().shape, new.distribution().shape);
    assert_eq!(old.links(), new.links());
    assert_eq!(
        old.distribution().topology.size(),
        new.distribution().topology.size()
    );
    let size = old.distribution().topology.size();
    assert!(rank < size);

    let mut send = vec![Vec::new(); size];
    visit_canonical_keys(old, rank, |key| {
        if old.distribution().owner(key) != rank {
            return;
        }
        let offset = old.local_offset(rank, key);
        for (destination, bucket) in send.iter_mut().enumerate() {
            if new.distribution().owns(destination, key) {
                bucket.push(offset);
            }
        }
    });

    let mut receive = vec![Vec::new(); size];
    visit_canonical_keys(new, rank, |key| {
        let source = old.distribution().owner(key);
        receive[source].push(new.local_offset(rank, key));
    });

    Plan { send, receive }
}

/// Visit canonical packed keys in ascending global-key order. Axis zero is
/// the fastest coordinate, so recursion starts at the highest axis and each
/// lower axis is bounded by its next coordinate in the same symmetry group.
pub(crate) fn visit_canonical_keys(
    distribution: &SymmetricDistribution,
    rank: usize,
    mut visit: impl FnMut(usize),
) {
    let rectangular = distribution.distribution();
    if rectangular.shape.iter().any(|&extent| extent == 0) {
        return;
    }
    if rectangular.shape.is_empty() {
        visit(0);
        return;
    }

    let rank_coordinates = rectangular.topology.coordinates(rank);
    let residues: Vec<_> = rectangular
        .mappings
        .iter()
        .map(|mapping| mapping.physical_rank(&rank_coordinates))
        .collect();
    if residues
        .iter()
        .zip(&rectangular.shape)
        .any(|(&residue, &extent)| residue >= extent)
    {
        return;
    }

    let phases: Vec<_> = rectangular
        .mappings
        .iter()
        .map(|mapping| mapping.physical_phase())
        .collect();
    let links = distribution.links();
    let mut coordinates = residues.clone();

    fn visit_axis(
        axis: usize,
        rectangular: &Distribution,
        links: &[Symmetry],
        residues: &[usize],
        phases: &[usize],
        coordinates: &mut [usize],
        visit: &mut impl FnMut(usize),
    ) {
        let upper = if links[axis] == Symmetry::NS {
            rectangular.shape[axis]
        } else {
            let next = coordinates[axis + 1];
            next + usize::from(links[axis] == Symmetry::SY)
        };
        let mut coordinate = residues[axis];
        while coordinate < upper {
            coordinates[axis] = coordinate;
            if axis == 0 {
                visit(rectangular.encode_key(coordinates));
            } else {
                visit_axis(
                    axis - 1,
                    rectangular,
                    links,
                    residues,
                    phases,
                    coordinates,
                    visit,
                );
            }
            coordinate += phases[axis];
        }
        coordinates[axis] = residues[axis];
    }

    visit_axis(
        rectangular.shape.len() - 1,
        rectangular,
        links,
        &residues,
        &phases,
        &mut coordinates,
        &mut visit,
    );
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::plan;
    use crate::{
        mapping::{Distribution, Mapping, Topology},
        symmetry::Symmetry::{self, AS, NS, SH, SY},
        symmetric_distribution::SymmetricDistribution,
    };

    fn physical(topology: &Topology, axis: usize, total_phase: usize) -> Mapping {
        let mut mapping = Mapping::Unmapped;
        mapping.augment_physical(topology, axis);
        mapping.augment_virtual(total_phase);
        mapping
    }

    fn virtual_map(total_phase: usize) -> Mapping {
        let mut mapping = Mapping::Unmapped;
        mapping.augment_virtual(total_phase);
        mapping
    }

    fn assert_plan(old: &SymmetricDistribution, new: &SymmetricDistribution) {
        let size = old.distribution().topology.size();
        for rank in 0..size {
            let actual = plan(old, new, rank);
            let old_pairs = old.local_pairs(rank);
            let new_pairs = new.local_pairs(rank);
            let old_by_offset: BTreeMap<_, _> = old_pairs.iter().map(|&(offset, key)| (offset, key)).collect();

            let mut expected_send = vec![Vec::new(); size];
            let mut old_canonical = old_pairs;
            old_canonical.sort_by_key(|&(_, key)| key);
            for &(offset, key) in &old_canonical {
                if old.distribution().owner(key) != rank {
                    continue;
                }
                for destination in 0..size {
                    if new.distribution().owns(destination, key) {
                        expected_send[destination].push(offset);
                    }
                }
            }

            let mut expected_receive = vec![Vec::new(); size];
            let mut new_canonical = new_pairs;
            new_canonical.sort_by_key(|&(_, key)| key);
            for &(offset, key) in &new_canonical {
                expected_receive[old.distribution().owner(key)].push(offset);
            }

            assert_eq!(actual.send, expected_send);
            assert_eq!(actual.receive, expected_receive);
            for destination in 0..size {
                let send_keys: Vec<_> = actual.send[destination]
                    .iter()
                    .map(|offset| old_by_offset[offset])
                    .collect();
                let destination_plan = plan(old, new, destination);
                let destination_pairs = new.local_pairs(destination);
                let destination_by_offset: BTreeMap<_, _> = destination_pairs
                    .iter()
                    .map(|&(offset, key)| (offset, key))
                    .collect();
                let receive_keys: Vec<_> = destination_plan.receive[rank]
                    .iter()
                    .map(|offset| destination_by_offset[offset])
                    .collect();
                assert_eq!(send_keys, receive_keys);
            }
        }
    }

    #[test]
    fn switched_physical_virtual_axes_cover_all_symmetries() {
        for &size in &[1, 2, 4] {
            let topology = Topology::new(vec![size]);
            for &kind in &[SY, AS, SH] {
                let old = SymmetricDistribution::new(
                    Distribution::new(
                        vec![5, 5],
                        topology.clone(),
                        vec![physical(&topology, 0, 2 * size), virtual_map(2 * size)],
                    ),
                    vec![kind, NS],
                );
                let new = SymmetricDistribution::new(
                    Distribution::new(
                        vec![5, 5],
                        topology.clone(),
                        vec![virtual_map(2 * size), physical(&topology, 0, 2 * size)],
                    ),
                    vec![kind, NS],
                );
                assert_plan(&old, &new);
            }
        }
    }

    #[test]
    fn replicated_higher_order_and_empty_regions_are_exact() {
        for &size in &[1, 2, 4] {
            let topology = Topology::new(vec![size]);
            for &kind in &[SY, AS, SH] {
                let old = SymmetricDistribution::new(
                    Distribution::new(
                        vec![4, 4, 4],
                        topology.clone(),
                        vec![virtual_map(2 * size), virtual_map(2 * size), virtual_map(2 * size)],
                    ),
                    vec![kind, kind, NS],
                );
                let new = SymmetricDistribution::new(
                    Distribution::new(
                        vec![4, 4, 4],
                        topology.clone(),
                        vec![virtual_map(2 * size), virtual_map(2 * size), virtual_map(2 * size)],
                    ),
                    vec![kind, kind, NS],
                );
                assert_plan(&old, &new);
            }

            let empty = SymmetricDistribution::new(
                Distribution::new(vec![0, 0], topology.clone(), vec![virtual_map(2 * size), virtual_map(2 * size)]),
                vec![SY, NS],
            );
            assert_plan(&empty, &empty);

            let scalar = SymmetricDistribution::new(
                Distribution::new(vec![], topology.clone(), vec![]),
                vec![],
            );
            assert_plan(&scalar, &scalar);
        }
    }

    #[test]
    fn strict_symmetry_skips_zero_diagonal_without_rectangular_scan() {
        let topology = Topology::new(vec![2]);
        let old = SymmetricDistribution::new(
            Distribution::new(
                vec![3, 3],
                topology.clone(),
                vec![physical(&topology, 0, 2), virtual_map(2)],
            ),
            vec![AS, NS],
        );
        let new = SymmetricDistribution::new(
            Distribution::new(
                vec![3, 3],
                topology,
                vec![virtual_map(2), physical(&Topology::new(vec![2]), 0, 2)],
            ),
            vec![AS, NS],
        );
        assert_plan(&old, &new);
    }
}
