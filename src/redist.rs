//! Equal-phase dense block reshuffle.
//!
//! When every logical axis keeps the same total cyclic phase, complete virtual
//! blocks can move without inspecting their elements.  Only each old primary
//! layer sends and only each new primary layer receives; replica layers retain
//! additive identity, matching the source block-reshuffle primitive.

use crate::{
    algebra::{Monoid, Wire},
    context::Context,
    mapping::{Distribution, Mapping},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockTransfer {
    pub source_block: usize,
    pub destination_block: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockReshufflePlan {
    pub rank: usize,
    pub block_len: usize,
    pub send: Vec<Vec<BlockTransfer>>,
    pub receive: Vec<Vec<BlockTransfer>>,
    pub send_counts: Vec<usize>,
    pub receive_counts: Vec<usize>,
    pub send_displacements: Vec<usize>,
    pub receive_displacements: Vec<usize>,
    source_len: usize,
    destination_len: usize,
}

impl BlockReshufflePlan {
    pub fn new(old: &Distribution, new: &Distribution, rank: usize) -> Self {
        assert_eq!(old.shape, new.shape);
        assert_eq!(old.topology.size(), new.topology.size());
        assert!(rank < old.topology.size());
        assert!(old
            .mappings
            .iter()
            .zip(&new.mappings)
            .all(|(old, new)| old.phase() == new.phase()));

        let size = old.topology.size();
        let block_len: usize = old.block_shape().iter().product();
        assert_eq!(old.block_shape(), new.block_shape());
        let mut send = vec![Vec::new(); size];
        let mut receive = vec![Vec::new(); size];

        for source in 0..size {
            if !is_primary_layer(old, source) {
                continue;
            }
            let rank_coordinates = old.topology.coordinates(source);
            let physical_residues: Vec<_> = old
                .mappings
                .iter()
                .map(|mapping| mapping.physical_rank(&rank_coordinates))
                .collect();
            let old_virtual_phases: Vec<_> = old
                .mappings
                .iter()
                .map(|mapping| mapping.phase() / mapping.physical_phase())
                .collect();
            let number_of_blocks: usize = old_virtual_phases.iter().product();

            for source_block in 0..number_of_blocks {
                let mut remainder = source_block;
                let mut destination_block = 0;
                let mut destination_stride = 1;
                let mut residues = Vec::with_capacity(old.mappings.len());
                for ((mapping, &physical_residue), &virtual_phase) in old
                    .mappings
                    .iter()
                    .zip(&physical_residues)
                    .zip(&old_virtual_phases)
                {
                    let virtual_rank = remainder % virtual_phase;
                    remainder /= virtual_phase;
                    residues.push(
                        physical_residue + virtual_rank * mapping.physical_phase(),
                    );
                }
                for (&residue, mapping) in residues.iter().zip(&new.mappings) {
                    destination_block +=
                        (residue / mapping.physical_phase()) * destination_stride;
                    destination_stride *= mapping.phase() / mapping.physical_phase();
                }
                let destination = primary_owner(new, &residues);
                let transfer = BlockTransfer {
                    source_block,
                    destination_block,
                };
                if source == rank {
                    send[destination].push(transfer);
                }
                if destination == rank {
                    receive[source].push(transfer);
                }
            }
        }

        let send_counts = send
            .iter()
            .map(|transfers| transfers.len() * block_len)
            .collect::<Vec<_>>();
        let receive_counts = receive
            .iter()
            .map(|transfers| transfers.len() * block_len)
            .collect::<Vec<_>>();
        let send_displacements = displacements(&send_counts);
        let receive_displacements = displacements(&receive_counts);

        Self {
            rank,
            block_len,
            send,
            receive,
            send_counts,
            receive_counts,
            send_displacements,
            receive_displacements,
            source_len: old.local_len(),
            destination_len: new.local_len(),
        }
    }

    pub fn pack<T: Wire>(&self, source: &[T]) -> Vec<Vec<u8>> {
        assert_eq!(source.len(), self.source_len);
        let mut buckets = vec![Vec::new(); self.send.len()];
        for (transfers, bucket) in self.send.iter().zip(&mut buckets) {
            bucket.reserve(transfers.len() * self.block_len * T::WIDTH);
            for transfer in transfers {
                let start = transfer.source_block * self.block_len;
                for value in &source[start..start + self.block_len] {
                    value.encode(bucket);
                }
            }
        }
        buckets
    }

    pub fn unpack<A: Monoid>(&self, algebra: &A, received: &[Vec<u8>]) -> Vec<A::Element>
    where
        A::Element: Wire,
    {
        assert_eq!(received.len(), self.receive.len());
        let mut destination = vec![algebra.zero(); self.destination_len];
        for ((bytes, transfers), &count) in received
            .iter()
            .zip(&self.receive)
            .zip(&self.receive_counts)
        {
            assert_eq!(bytes.len(), count * A::Element::WIDTH);
            let mut chunks = bytes.chunks_exact(A::Element::WIDTH);
            for transfer in transfers {
                let start = transfer.destination_block * self.block_len;
                for output in &mut destination[start..start + self.block_len] {
                    *output = A::Element::decode(chunks.next().unwrap());
                }
            }
            assert!(chunks.next().is_none());
        }
        destination
    }

    pub fn execute<A: Monoid>(
        &self,
        context: &Context<'_>,
        algebra: &A,
        source: &[A::Element],
    ) -> Vec<A::Element>
    where
        A::Element: Wire,
    {
        assert_eq!(context.rank(), self.rank);
        assert_eq!(context.size(), self.send.len());
        let received = context.inner.exchange(&self.pack(source));
        self.unpack(algebra, &received)
    }
}

fn is_primary_layer(distribution: &Distribution, rank: usize) -> bool {
    let coordinates = distribution.topology.coordinates(rank);
    let residues: Vec<_> = distribution
        .mappings
        .iter()
        .map(|mapping| mapping.physical_rank(&coordinates))
        .collect();
    primary_owner(distribution, &residues) == rank
}

fn primary_owner(distribution: &Distribution, residues: &[usize]) -> usize {
    let mut coordinates = vec![0; distribution.topology.dimensions.len()];
    for (mapping, &residue) in distribution.mappings.iter().zip(residues) {
        assign_physical(mapping, residue, &mut coordinates);
    }
    distribution.topology.rank(&coordinates)
}

fn assign_physical(mapping: &Mapping, remainder: usize, coordinates: &mut [usize]) {
    match mapping {
        Mapping::Unmapped => {}
        Mapping::Physical {
            axis,
            processes,
            child,
        } => {
            coordinates[*axis] = remainder % processes;
            assign_physical(child, remainder / processes, coordinates);
        }
        Mapping::Virtual { child, .. } => assign_physical(child, remainder, coordinates),
    }
}

fn displacements(counts: &[usize]) -> Vec<usize> {
    let mut total = 0;
    counts
        .iter()
        .map(|&count| {
            let displacement = total;
            total += count;
            displacement
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::BlockReshufflePlan;
    use crate::{
        algebra::Arithmetic,
        mapping::{Distribution, Mapping, Topology},
    };

    #[test]
    fn full_virtual_blocks_move_with_padding_and_primary_layer_semantics() {
        let topology = Topology::new(vec![2]);
        let mut old_mapping = Mapping::Unmapped;
        old_mapping.augment_physical(&topology, 0);
        old_mapping.augment_virtual(4);
        let mut new_mapping = Mapping::Unmapped;
        new_mapping.augment_virtual(4);
        let old = Distribution::new(vec![5], topology.clone(), vec![old_mapping]);
        let new = Distribution::new(vec![5], topology, vec![new_mapping]);
        let plans: Vec<_> = (0..2)
            .map(|rank| BlockReshufflePlan::new(&old, &new, rank))
            .collect();
        assert_eq!(plans[0].block_len, 2);

        let sources = [vec![10i64, 11, 12, 13], vec![20, 21, 22, 23]];
        let packed: Vec<_> = plans
            .iter()
            .zip(&sources)
            .map(|(plan, source)| plan.pack(source))
            .collect();
        let received = |destination: usize| {
            (0..2)
                .map(|source| packed[source][destination].clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            plans[0].unpack(&Arithmetic::<i64>::new(), &received(0)),
            [10, 11, 20, 21, 12, 13, 22, 23]
        );
        assert_eq!(
            plans[1].unpack(&Arithmetic::<i64>::new(), &received(1)),
            [0; 8]
        );
        assert_eq!(plans[0].send_displacements, [0, 4]);
        assert_eq!(plans[0].receive_counts, [4, 4]);
    }

    #[test]
    fn scalar_moves_only_between_primary_layers() {
        let topology = Topology::new(vec![3]);
        let distribution = Distribution::new(vec![], topology, vec![]);
        let root = BlockReshufflePlan::new(&distribution, &distribution, 0);
        assert_eq!(root.send_counts, [1, 0, 0]);
        assert_eq!(root.receive_counts, [1, 0, 0]);
        for rank in 1..3 {
            let replica = BlockReshufflePlan::new(&distribution, &distribution, rank);
            assert!(replica.send_counts.iter().all(|&count| count == 0));
            assert!(replica.receive_counts.iter().all(|&count| count == 0));
        }
    }
}
