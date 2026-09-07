// Adapted from cc4s CTF tensor/untyped_tensor.cxx add_{to,from}_subworld.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Dense accumulation between a parent context and an explicitly described subworld.

use crate::{
    algebra::{Semiring, Wire},
    context::Context,
    mapping::Distribution,
    tensor::Tensor,
};

pub(crate) fn orient_subworld(
    parent: &Context<'_>,
    local_child_rank: Option<usize>,
    child_size: usize,
) -> Vec<usize> {
    let local = local_child_rank.map_or(-1, |rank| i32::try_from(rank).unwrap());
    let ranks = parent.inner.all_gather_i32(local);
    let mut parent_for_child = vec![usize::MAX; child_size];
    for (parent_rank, child_rank) in ranks.into_iter().enumerate() {
        if child_rank < 0 {
            continue;
        }
        let child_rank = usize::try_from(child_rank).unwrap();
        assert!(child_rank < child_size);
        assert_eq!(parent_for_child[child_rank], usize::MAX);
        parent_for_child[child_rank] = parent_rank;
    }
    assert!(parent_for_child.iter().all(|&rank| rank != usize::MAX));
    parent_for_child
}

fn apply_received<A: Semiring + Clone>(
    tensor: &mut Tensor<'_, '_, A>,
    received: Vec<Vec<u8>>,
    algebra: &A,
    alpha: &A::Element,
    beta: &A::Element,
) where
    A::Element: Wire,
{
    let rank = tensor.context().rank();
    for message in received {
        for pair in message.chunks_exact(8 + A::Element::WIDTH) {
            let key = u64::decode(&pair[..8]) as usize;
            let incoming = A::Element::decode(&pair[8..]);
            let offset = tensor.distribution().local_offset(rank, key);
            let incoming = algebra.multiply(&incoming, alpha);
            let old = algebra.multiply(&tensor.data[offset], beta);
            tensor.data[offset] = algebra.add(&incoming, &old);
        }
    }
}

impl<A: Semiring + Clone> Tensor<'_, '_, A>
where
    A::Element: Wire,
{
    /// Accumulate `self*alpha + destination*beta` into every destination
    /// replica. All parent ranks participate; ranks outside the child pass None.
    pub fn add_to_subworld(
        &self,
        destination: Option<&mut Tensor<'_, '_, A>>,
        target_distribution: &Distribution,
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert_eq!(self.distribution().shape, target_distribution.shape);
        let child_size = target_distribution.topology.size();
        let child_rank = destination.as_ref().map(|destination| {
            assert_eq!(destination.context().size(), child_size);
            assert_eq!(destination.distribution(), target_distribution);
            destination.context().rank()
        });
        let parent_for_child = orient_subworld(self.context(), child_rank, child_size);

        let parent_rank = self.context().rank();
        let mut buckets = vec![Vec::new(); self.context().size()];
        for (key, value) in self.local_pairs() {
            if self.distribution().owner(key) != parent_rank {
                continue;
            }
            for child_rank in 0..child_size {
                if target_distribution.owns(child_rank, key) {
                    let bucket = &mut buckets[parent_for_child[child_rank]];
                    (key as u64).encode(bucket);
                    value.encode(bucket);
                }
            }
        }
        let received = self.context().inner.exchange(&buckets);
        if let Some(destination) = destination {
            apply_received(destination, received, self.algebra(), &alpha, &beta);
        }
    }

    /// Accumulate `source*alpha + self*beta` from every canonical child source
    /// owner. All parent ranks participate; ranks outside the child pass None.
    pub fn add_from_subworld(
        &mut self,
        source: Option<&Tensor<'_, '_, A>>,
        source_distribution: &Distribution,
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert_eq!(self.distribution().shape, source_distribution.shape);
        let child_size = source_distribution.topology.size();
        let child_rank = source.map(|source| {
            assert_eq!(source.context().size(), child_size);
            assert_eq!(source.distribution(), source_distribution);
            source.context().rank()
        });
        orient_subworld(self.context(), child_rank, child_size);

        let mut buckets = vec![Vec::new(); self.context().size()];
        if let Some(source) = source {
            let child_rank = source.context().rank();
            for (key, value) in source.local_pairs() {
                if source_distribution.owner(key) != child_rank {
                    continue;
                }
                for parent_rank in 0..self.context().size() {
                    if self.distribution().owns(parent_rank, key) {
                        let bucket = &mut buckets[parent_rank];
                        (key as u64).encode(bucket);
                        value.encode(bucket);
                    }
                }
            }
        }
        let received = self.context().inner.exchange(&buckets);
        let algebra = self.algebra().clone();
        apply_received(self, received, &algebra, &alpha, &beta);
    }
}
