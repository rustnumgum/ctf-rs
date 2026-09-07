// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Closed-form dense DGTOG replica-bucket counts and peer locations.

use crate::mapping::{Distribution, Mapping};

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub common_phase: Vec<usize>,
    pub replica_phase: Vec<usize>,
    pub residues: Vec<Vec<usize>>,
    pub counts: Vec<usize>,
    pub displacements: Vec<usize>,
    pub peers: Vec<usize>,
    pub is_root: bool,
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

fn physical_residue(mapping: &Mapping, coordinates: &[usize]) -> usize {
    mapping.physical_rank(coordinates)
}

fn assign_physical(mapping: &Mapping, mut residue: usize, coordinates: &mut [usize]) {
    match mapping {
        Mapping::Unmapped => assert_eq!(residue, 0),
        Mapping::Virtual { child, .. } => assign_physical(child, residue, coordinates),
        Mapping::Physical {
            axis,
            processes,
            child,
        } => {
            coordinates[*axis] = residue % processes;
            residue /= processes;
            assign_physical(child, residue, coordinates);
        }
    }
}

fn rank_from_residues(distribution: &Distribution, residues: &[usize]) -> usize {
    let mut coordinates = vec![0; distribution.topology.dimensions.len()];
    for (mapping, &residue) in distribution.mappings.iter().zip(residues) {
        assign_physical(mapping, residue, &mut coordinates);
    }
    distribution.topology.rank(&coordinates)
}

fn residue_count(extent: usize, phase: usize, residue: usize) -> usize {
    if residue < extent {
        (extent - residue).div_ceil(phase)
    } else {
        0
    }
}

/// Source `calc_drv_displs` for ordinary dense storage. Bucket zero is the
/// source rank's physical residue; dimension zero is the fastest replica digit.
pub(crate) fn layout(
    source: &Distribution,
    target: &Distribution,
    rank: usize,
) -> Layout {
    assert_eq!(source.shape, target.shape);
    assert_eq!(source.topology.size(), target.topology.size());
    assert!(rank < source.topology.size());

    let coordinates = source.topology.coordinates(rank);
    let source_residue: Vec<_> = source
        .mappings
        .iter()
        .map(|mapping| physical_residue(mapping, &coordinates))
        .collect();
    let common_phase: Vec<_> = source
        .mappings
        .iter()
        .zip(&target.mappings)
        .map(|(old, new)| lcm(old.physical_phase(), new.physical_phase()))
        .collect();
    let replica_phase: Vec<_> = common_phase
        .iter()
        .zip(&source.mappings)
        .map(|(&common, mapping)| common / mapping.physical_phase())
        .collect();
    let bucket_count = replica_phase.iter().product();
    let is_root = rank_from_residues(source, &source_residue) == rank;

    let mut residues = Vec::with_capacity(bucket_count);
    let mut counts = Vec::with_capacity(bucket_count);
    let mut peers = Vec::with_capacity(bucket_count);
    for bucket in 0..bucket_count {
        let mut quotient = bucket;
        let mut bucket_residue = Vec::with_capacity(source.shape.len());
        let mut peer_residue = Vec::with_capacity(source.shape.len());
        for axis in 0..source.shape.len() {
            let digit = quotient % replica_phase[axis];
            quotient /= replica_phase[axis];
            let residue = source_residue[axis]
                + digit * source.mappings[axis].physical_phase();
            bucket_residue.push(residue);
            peer_residue.push(residue % target.mappings[axis].physical_phase());
        }
        counts.push(
            source
                .shape
                .iter()
                .zip(&common_phase)
                .zip(&bucket_residue)
                .map(|((&extent, &phase), &residue)| residue_count(extent, phase, residue))
                .product(),
        );
        peers.push(rank_from_residues(target, &peer_residue));
        residues.push(bucket_residue);
    }

    let mut next = 0usize;
    let displacements = counts
        .iter()
        .map(|&count| {
            let displacement = next;
            next = next.checked_add(count).unwrap();
            displacement
        })
        .collect();
    Layout {
        common_phase,
        replica_phase,
        residues,
        counts,
        displacements,
        peers,
        is_root,
    }
}

#[cfg(test)]
mod tests {
    use super::layout;
    use crate::mapping::{Distribution, Mapping, Topology};

    fn physical(topology: &Topology, axis: usize) -> Mapping {
        let mut mapping = Mapping::Unmapped;
        mapping.augment_physical(topology, axis);
        mapping
    }

    #[test]
    fn lcm_replica_counts_displacements_and_roots_are_exact() {
        let topology = Topology::new(vec![2, 3]);
        let old = Distribution::new(
            vec![7, 5],
            topology.clone(),
            vec![physical(&topology, 0), Mapping::Unmapped],
        );
        let new = Distribution::new(
            vec![7, 5],
            topology,
            vec![physical(&Topology::new(vec![2, 3]), 1), Mapping::Unmapped],
        );

        let root = layout(&old, &new, 1);
        assert!(root.is_root);
        assert_eq!(root.common_phase, vec![6, 1]);
        assert_eq!(root.replica_phase, vec![3, 1]);
        assert_eq!(root.residues, vec![vec![1, 0], vec![3, 0], vec![5, 0]]);
        assert_eq!(root.counts, vec![5, 5, 5]);
        assert_eq!(root.displacements, vec![0, 5, 10]);
        assert_eq!(root.peers, vec![2, 0, 4]);

        let replica = layout(&old, &new, 3);
        assert!(!replica.is_root);
        assert_eq!(replica.counts, root.counts);
        assert_eq!(replica.peers, root.peers);
    }

    #[test]
    fn padding_and_empty_axes_have_zero_closed_form_counts() {
        let topology = Topology::new(vec![4]);
        let old = Distribution::cyclic(vec![2, 0, 3], 4);
        let new = Distribution::new(
            vec![2, 0, 3],
            topology,
            vec![Mapping::Unmapped; 3],
        );
        assert_eq!(layout(&old, &new, 3).counts, vec![0]);
    }
}
