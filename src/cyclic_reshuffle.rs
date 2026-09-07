//! Offset-only dense cyclic redistribution plans.
//!
//! The plan follows the source `pad_cyclic_pup_virt_buff` ordering: valid
//! local keys are visited in ascending global-key order, so each sender and
//! receiver bucket has matching offsets without carrying keys in the payload.

use crate::mapping::Distribution;

pub(crate) struct Plan {
    pub send: Vec<Vec<usize>>,
    pub receive: Vec<Vec<usize>>,
}

impl Plan {
    pub(crate) fn new(old: &Distribution, new: &Distribution, rank: usize) -> Self {
        assert_eq!(old.shape, new.shape);
        assert_eq!(old.topology.size(), new.topology.size());
        let size = old.topology.size();
        assert!(rank < size);

        let mut send = vec![Vec::new(); size];
        visit_local_keys(old, rank, |key| {
            if old.owner(key) != rank {
                return;
            }
            let offset = old.local_offset(rank, key);
            for (destination, bucket) in send.iter_mut().enumerate() {
                if new.owns(destination, key) {
                    bucket.push(offset);
                }
            }
        });

        let mut receive = vec![Vec::new(); size];
        visit_local_keys(new, rank, |key| {
            let source = old.owner(key);
            receive[source].push(new.local_offset(rank, key));
        });

        Self { send, receive }
    }
}

/// Visit a rank's valid local keys in global column-major order. Physical
/// residues define the first coordinate on every axis; virtual copies remain
/// represented by all later values spaced by that axis' physical phase.
fn visit_local_keys(distribution: &Distribution, rank: usize, mut visit: impl FnMut(usize)) {
    let shape = &distribution.shape;
    if shape.iter().any(|&extent| extent == 0) {
        return;
    }
    if shape.is_empty() {
        visit(0);
        return;
    }

    let rank_coordinates = distribution.topology.coordinates(rank);
    let residues: Vec<_> = distribution
        .mappings
        .iter()
        .map(|mapping| mapping.physical_rank(&rank_coordinates))
        .collect();
    if residues
        .iter()
        .zip(shape)
        .any(|(&residue, &extent)| residue >= extent)
    {
        return;
    }

    let phases: Vec<_> = distribution
        .mappings
        .iter()
        .map(|mapping| mapping.physical_phase())
        .collect();
    let mut coordinates = residues.clone();
    loop {
        visit(distribution.encode_key(&coordinates));

        // Dimension zero is the fastest-varying global coordinate.
        let mut axis = 0;
        while axis < shape.len() {
            let next = coordinates[axis] + phases[axis];
            if next < shape[axis] {
                coordinates[axis] = next;
                break;
            }
            coordinates[axis] = residues[axis];
            axis += 1;
        }
        if axis == shape.len() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Plan;
    use crate::mapping::{Distribution, Mapping, Topology};

    fn assert_matching_offsets(old: &Distribution, new: &Distribution) {
        let size = old.topology.size();
        for rank in 0..size {
            let plan = Plan::new(old, new, rank);
            for destination in 0..size {
                let send_keys: Vec<_> = plan.send[destination]
                    .iter()
                    .map(|&offset| old.global_key(rank, offset).unwrap())
                    .collect();
                let destination_plan = Plan::new(old, new, destination);
                let receive_keys: Vec<_> = destination_plan.receive[rank]
                    .iter()
                    .map(|&offset| new.global_key(destination, offset).unwrap())
                    .collect();
                assert_eq!(send_keys, receive_keys);
            }
        }
    }

    fn physical(topology: &Topology, axis: usize, virtual_copies: usize) -> Mapping {
        let mut mapping = Mapping::Unmapped;
        mapping.augment_physical(topology, axis);
        if virtual_copies > 1 {
            mapping.augment_virtual(virtual_copies);
        }
        mapping
    }

    #[test]
    fn physical_virtual_reshuffles_keep_global_order() {
        for &size in &[1, 2, 4] {
            let topology = Topology::new(vec![size]);
            let old = Distribution::new(
                vec![5, 4],
                topology.clone(),
                vec![physical(&topology, 0, 2 * size), Mapping::Unmapped],
            );
            let new = Distribution::new(
                vec![5, 4],
                topology.clone(),
                vec![Mapping::Unmapped, physical(&topology, 0, 2 * size)],
            );
            assert_matching_offsets(&old, &new);
        }
    }

    #[test]
    fn replicated_and_mixed_layouts_keep_global_order() {
        for &size in &[1, 2, 4] {
            let topology = Topology::new(vec![size]);
            let old = Distribution::new(
                vec![4, 3],
                topology.clone(),
                vec![Mapping::Virtual { copies: 2 * size, child: Box::new(Mapping::Unmapped) }, Mapping::Unmapped],
            );
            let new = Distribution::new(
                vec![4, 3],
                topology.clone(),
                vec![physical(&topology, 0, 2 * size), Mapping::Unmapped],
            );
            assert_matching_offsets(&old, &new);
        }
    }

    #[test]
    fn scalar_and_zero_axes_have_exact_empty_or_scalar_plans() {
        let topology = Topology::new(vec![4]);
        let scalar = Distribution::new(vec![], topology.clone(), vec![]);
        for rank in 0..4 {
            let plan = Plan::new(&scalar, &scalar, rank);
            if rank == 0 {
                assert_eq!(plan.send[0], vec![0]);
            } else {
                assert!(plan.send.iter().all(Vec::is_empty));
            }
            assert_eq!(plan.receive[0], vec![0]);
        }

        let empty = Distribution::new(vec![0, 3], topology.clone(), vec![
            physical(&topology, 0, 1),
            Mapping::Unmapped,
        ]);
        for rank in 0..4 {
            let plan = Plan::new(&empty, &empty, rank);
            assert!(plan.send.iter().all(Vec::is_empty));
            assert!(plan.receive.iter().all(Vec::is_empty));
        }
    }
}
