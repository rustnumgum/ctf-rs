//! Direct serialized subworld accumulation for compressed symmetric storage.

use super::SymmetricTensor;
use crate::{
    algebra::{Ring, Wire},
    subworld::orient_subworld,
    symmetric_distribution::SymmetricDistribution,
};

fn apply_received<A: Ring + Clone>(
    tensor: &mut SymmetricTensor<'_, '_, A>,
    received: Vec<Vec<u8>>,
    algebra: &A,
    alpha: &A::Element,
    beta: &A::Element,
) where
    A::Element: Wire,
{
    let rank = tensor.context.rank();
    for message in received {
        for pair in message.chunks_exact(8 + A::Element::WIDTH) {
            let key = u64::decode(&pair[..8]) as usize;
            let incoming = A::Element::decode(&pair[8..]);
            let offset = tensor.distribution.local_offset(rank, key);
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
        let mut buckets = vec![Vec::new(); self.context.size()];
        for (key, value) in self.local_pairs() {
            if self.distribution.distribution().owner(key) != parent_rank {
                continue;
            }
            for child_rank in 0..child_size {
                if target_distribution.distribution().owns(child_rank, key) {
                    let bucket = &mut buckets[parent_for_child[child_rank]];
                    (key as u64).encode(bucket);
                    value.encode(bucket);
                }
            }
        }
        let received = self.context.inner.exchange(&buckets);
        if let Some(destination) = destination {
            apply_received(destination, received, &self.algebra, &alpha, &beta);
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
        orient_subworld(self.context, child_rank, child_size);

        let mut buckets = vec![Vec::new(); self.context.size()];
        if let Some(source) = source {
            let child_rank = source.context.rank();
            for (key, value) in source.local_pairs() {
                if source_distribution.distribution().owner(key) != child_rank {
                    continue;
                }
                for parent_rank in 0..self.context.size() {
                    if self.distribution.distribution().owns(parent_rank, key) {
                        let bucket = &mut buckets[parent_rank];
                        (key as u64).encode(bucket);
                        value.encode(bucket);
                    }
                }
            }
        }
        let received = self.context.inner.exchange(&buckets);
        let algebra = self.algebra.clone();
        apply_received(self, received, &algebra, &alpha, &beta);
    }
}
