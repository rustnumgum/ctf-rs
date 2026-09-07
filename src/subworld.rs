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
    expected_offsets: &[Vec<usize>],
    algebra: &A,
    alpha: &A::Element,
    beta: &A::Element,
) where
    A::Element: Wire,
{
    for (message, offsets) in received.into_iter().zip(expected_offsets) {
        assert_eq!(message.len(), offsets.len() * A::Element::WIDTH);
        for (offset, bytes) in offsets.iter().zip(message.chunks_exact(A::Element::WIDTH)) {
            let incoming = A::Element::decode(bytes);
            let incoming = algebra.multiply(&incoming, alpha);
            let old = algebra.multiply(&tensor.data[*offset], beta);
            tensor.data[*offset] = algebra.add(&incoming, &old);
        }
    }
}

fn plan(
    old: &Distribution,
    new: &Distribution,
    old_rank: Option<usize>,
    new_rank: Option<usize>,
    source_parents: &[usize],
    destination_parents: &[usize],
    parent_size: usize,
) -> crate::cyclic_reshuffle::Plan {
    let mut send = vec![Vec::new(); parent_size];
    if let Some(rank) = old_rank {
        crate::cyclic_reshuffle::visit_local_keys(old, rank, |key| {
            if old.owner(key) != rank {
                return;
            }
            let offset = old.local_offset(rank, key);
            for (destination, &parent) in destination_parents.iter().enumerate() {
                if new.owns(destination, key) {
                    send[parent].push(offset);
                }
            }
        });
    }

    let mut receive = vec![Vec::new(); parent_size];
    if let Some(rank) = new_rank {
        crate::cyclic_reshuffle::visit_local_keys(new, rank, |key| {
            let source = old.owner(key);
            receive[source_parents[source]].push(new.local_offset(rank, key));
        });
    }

    crate::cyclic_reshuffle::Plan { send, receive }
}

fn encode_offsets<A: Semiring + Clone>(
    data: &[A::Element],
    send: &[Vec<usize>],
) -> Vec<Vec<u8>>
where
    A::Element: Wire,
{
    send.iter()
        .map(|offsets| {
            let mut bytes = Vec::with_capacity(offsets.len() * A::Element::WIDTH);
            for &offset in offsets {
                data[offset].encode(&mut bytes);
            }
            assert_eq!(bytes.len(), offsets.len() * A::Element::WIDTH);
            bytes
        })
        .collect()
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

        let parent_size = self.context().size();
        let plan = plan(
            self.distribution(),
            target_distribution,
            Some(self.context().rank()),
            child_rank,
            &(0..parent_size).collect::<Vec<_>>(),
            &parent_for_child,
            parent_size,
        );
        let buckets = encode_offsets::<A>(&self.data, &plan.send);
        let received = self.context().inner.exchange(&buckets);
        if let Some(destination) = destination {
            apply_received(
                destination,
                received,
                &plan.receive,
                self.algebra(),
                &alpha,
                &beta,
            );
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
        let parent_for_child = orient_subworld(self.context(), child_rank, child_size);
        let parent_size = self.context().size();
        let plan = plan(
            source_distribution,
            self.distribution(),
            child_rank,
            Some(self.context().rank()),
            &parent_for_child,
            &(0..parent_size).collect::<Vec<_>>(),
            parent_size,
        );
        let buckets = source
            .map(|source| encode_offsets::<A>(&source.data, &plan.send))
            .unwrap_or_else(|| vec![Vec::new(); parent_size]);
        let received = self.context().inner.exchange(&buckets);
        let algebra = self.algebra().clone();
        apply_received(self, received, &plan.receive, &algebra, &alpha, &beta);
    }
}
