//! Direct serialized subworld accumulation for compressed symmetric storage.

use super::SymmetricTensor;
use crate::{
    algebra::{Ring, Wire},
    cyclic_reshuffle::Plan,
    subworld::orient_subworld,
    symmetric_distribution::SymmetricDistribution,
    symmetric_reshuffle::visit_canonical_keys,
};

fn plan(
    old: &SymmetricDistribution,
    new: &SymmetricDistribution,
    old_rank: Option<usize>,
    new_rank: Option<usize>,
    source_parents: &[usize],
    destination_parents: &[usize],
    parent_size: usize,
) -> Plan {
    let mut send = vec![Vec::new(); parent_size];
    if let Some(rank) = old_rank {
        visit_canonical_keys(old, rank, |key| {
            if old.distribution().owner(key) != rank {
                return;
            }
            let offset = old.local_offset(rank, key);
            for (destination, &parent) in destination_parents.iter().enumerate() {
                if new.distribution().owns(destination, key) {
                    send[parent].push(offset);
                }
            }
        });
    }

    let mut receive = vec![Vec::new(); parent_size];
    if let Some(rank) = new_rank {
        visit_canonical_keys(new, rank, |key| {
            let source = source_parents[old.distribution().owner(key)];
            receive[source].push(new.local_offset(rank, key));
        });
    }
    Plan { send, receive }
}

fn encode_send<E: Wire>(data: &[E], send: &[Vec<usize>]) -> Vec<Vec<u8>> {
    send.iter()
        .map(|offsets| {
            let mut bytes = Vec::with_capacity(offsets.len() * E::WIDTH);
            for &offset in offsets {
                data[offset].encode(&mut bytes);
            }
            bytes
        })
        .collect()
}

fn apply_received<A: Ring + Clone>(
    tensor: &mut SymmetricTensor<'_, '_, A>,
    received: Vec<Vec<u8>>,
    receive: &[Vec<usize>],
    algebra: &A,
    alpha: &A::Element,
    beta: &A::Element,
) where
    A::Element: Wire,
{
    for (message, offsets) in received.into_iter().zip(receive) {
        assert_eq!(message.len(), offsets.len() * A::Element::WIDTH);
        for (bytes, &offset) in message.chunks_exact(A::Element::WIDTH).zip(offsets) {
            let incoming = A::Element::decode(bytes);
            let incoming = algebra.multiply(&incoming, alpha);
            let old = algebra.multiply(&tensor.data[offset], beta);
            tensor.data[offset] = algebra.add(&incoming, &old);
        }
    }
}

impl<A: Ring + Clone> SymmetricTensor<'_, '_, A>
where
    A::Element: Wire,
{
    /// Accumulate `self*alpha + destination*beta` into every compressed
    /// destination replica. All parent ranks participate; inactive child ranks
    /// pass `None`.
    pub fn add_to_subworld(
        &self,
        destination: Option<&mut SymmetricTensor<'_, '_, A>>,
        target_distribution: &SymmetricDistribution,
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert_eq!(
            self.distribution.distribution().shape,
            target_distribution.distribution().shape
        );
        assert_eq!(self.distribution.links(), target_distribution.links());
        let child_size = target_distribution.distribution().topology.size();
        let child_rank = destination.as_ref().map(|destination| {
            assert_eq!(destination.context.size(), child_size);
            assert_eq!(
                destination.distribution.distribution(),
                target_distribution.distribution()
            );
            assert_eq!(destination.distribution.links(), target_distribution.links());
            destination.context.rank()
        });
        let parent_for_child = orient_subworld(self.context, child_rank, child_size);

        let parent_rank = self.context.rank();
        let identity: Vec<_> = (0..self.context.size()).collect();
        let plan = plan(
            &self.distribution,
            target_distribution,
            Some(parent_rank),
            child_rank,
            &identity,
            &parent_for_child,
            self.context.size(),
        );
        let buckets = encode_send(&self.data, &plan.send);
        let received = self.context.inner.exchange(&buckets);
        if let Some(destination) = destination {
            apply_received(
                destination,
                received,
                &plan.receive,
                &self.algebra,
                &alpha,
                &beta,
            );
        }
    }

    /// Accumulate `source*alpha + self*beta` from canonical compressed child
    /// owners. All parent ranks participate; inactive child ranks pass `None`.
    pub fn add_from_subworld(
        &mut self,
        source: Option<&SymmetricTensor<'_, '_, A>>,
        source_distribution: &SymmetricDistribution,
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert_eq!(
            self.distribution.distribution().shape,
            source_distribution.distribution().shape
        );
        assert_eq!(self.distribution.links(), source_distribution.links());
        let child_size = source_distribution.distribution().topology.size();
        let child_rank = source.map(|source| {
            assert_eq!(source.context.size(), child_size);
            assert_eq!(
                source.distribution.distribution(),
                source_distribution.distribution()
            );
            assert_eq!(source.distribution.links(), source_distribution.links());
            source.context.rank()
        });
        let parent_for_child = orient_subworld(self.context, child_rank, child_size);

        let identity: Vec<_> = (0..self.context.size()).collect();
        let plan = plan(
            source_distribution,
            &self.distribution,
            child_rank,
            Some(self.context.rank()),
            &parent_for_child,
            &identity,
            self.context.size(),
        );
        let buckets = source.map_or_else(
            || vec![Vec::new(); self.context.size()],
            |source| encode_send(&source.data, &plan.send),
        );
        let received = self.context.inner.exchange(&buckets);
        let algebra = self.algebra.clone();
        apply_received(self, received, &plan.receive, &algebra, &alpha, &beta);
    }
}
