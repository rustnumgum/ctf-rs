//! Distributed compressed symmetry storage and collective indexed access.
// Canonical indexed access adapted from cc4s CTF sparse_rw.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{
    algebra::{Group, Wire},
    context::Context,
    symmetric_distribution::SymmetricDistribution,
};

#[path = "symmetric_operations.rs"]
mod operations;
#[path = "symmetric_sum_tensor.rs"]
mod summation;
#[path = "symmetric_hollow_sum.rs"]
mod hollow_summation;
#[path = "symmetric_diagonal.rs"]
mod diagonal;
#[path = "symmetric_sy_sum.rs"]
mod sy_summation;
#[path = "symmetric_contract_tensor.rs"]
mod contraction;
#[path = "symmetric_contract.rs"]
mod symmetric_contraction;

impl<'c, 'r, A: Group + crate::algebra::Semiring> SymmetricTensor<'c, 'r, A>
where A::Element: Wire {
    /// Collective indexed write: incoming*alpha + old*beta. Beta applies once
    /// per touched canonical key; untouched entries are unchanged.
    pub fn write_scaled(&mut self, pairs: &[(usize, A::Element)],
                        alpha: &A::Element, beta: &A::Element) {
        let mut buckets = vec![Vec::new(); self.context.size()];
        for (key, value) in pairs {
            let Some((canonical, sign)) = self.distribution.canonicalize(*key) else { continue; };
            let value = if sign == 1 { value.clone() } else { self.algebra.negate(value) };
            for (rank, bucket) in buckets.iter_mut().enumerate() {
                if self.distribution.distribution().owns(rank, canonical) {
                    (canonical as u64).encode(bucket);
                    value.encode(bucket);
                }
            }
        }
        let mut incoming = Vec::new();
        for bytes in self.context.inner.exchange(&buckets) {
            for pair in bytes.chunks_exact(8 + A::Element::WIDTH) {
                incoming.push((u64::decode(&pair[..8]) as usize, A::Element::decode(&pair[8..])));
            }
        }
        incoming.sort_by_key(|pair| pair.0);
        let mut position = 0;
        while position < incoming.len() {
            let key = incoming[position].0;
            let offset = self.distribution.local_offset(self.context.rank(), key);
            let mut value = self.algebra.add(
                &self.algebra.multiply(&incoming[position].1, alpha),
                &self.algebra.multiply(&self.data[offset], beta));
            position += 1;
            while position < incoming.len() && incoming[position].0 == key {
                value = self.algebra.add(&value, &self.algebra.multiply(&incoming[position].1, alpha));
                position += 1;
            }
            self.data[offset] = value;
        }
    }
}

/// Dense packed storage for the canonical entries of a symmetric tensor.
/// Allocated padding and noncanonical holes remain present in `local_storage`,
/// but tensor operations visit only canonical valid slots.
pub struct SymmetricTensor<'context, 'runtime, A: Group> {
    context: &'context Context<'runtime>,
    distribution: SymmetricDistribution,
    algebra: A,
    data: Vec<A::Element>,
}

impl<'c, 'r, A: Group> SymmetricTensor<'c, 'r, A> {
    pub fn new(
        context: &'c Context<'r>,
        distribution: SymmetricDistribution,
        algebra: A,
    ) -> Self {
        assert_eq!(
            context.size(),
            distribution.distribution().topology.size()
        );
        let data = vec![algebra.zero(); distribution.local_len()];
        Self {
            context,
            distribution,
            algebra,
            data,
        }
    }

    pub fn context(&self) -> &'c Context<'r> {
        self.context
    }

    pub fn distribution(&self) -> &SymmetricDistribution {
        &self.distribution
    }

    pub fn algebra(&self) -> &A {
        &self.algebra
    }

    /// Raw packed allocation, including padding and noncanonical holes.
    pub fn local_storage(&self) -> &[A::Element] {
        &self.data
    }

    /// Canonical local entries in physical packed-offset order.
    pub fn local_pairs(&self) -> Vec<(usize, A::Element)> {
        self.distribution
            .local_pairs(self.context.rank())
            .into_iter()
            .map(|(offset, key)| (key, self.data[offset].clone()))
            .collect()
    }

    /// Apply a local endomorphism once to every canonical valid slot.
    pub fn transform(&mut self, mut function: impl FnMut(usize, &mut A::Element)) {
        for (offset, key) in self.distribution.local_pairs(self.context.rank()) {
            function(key, &mut self.data[offset]);
        }
    }
}

impl<'c, 'r, A: Group + Clone> SymmetricTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Collective random access through canonical owners. Input ordering and
    /// duplicates are preserved; AS/SH structural diagonals read as zero.
    pub fn read(&self, keys: &[usize]) -> Vec<A::Element> {
        let mut requests = vec![Vec::new(); self.context.size()];
        let mut signs = vec![0; keys.len()];
        for (position, &key) in keys.iter().enumerate() {
            if let Some((canonical, sign)) = self.distribution.canonicalize(key) {
                signs[position] = sign;
                let owner = self.distribution.distribution().owner(canonical);
                (position as u64).encode(&mut requests[owner]);
                (canonical as u64).encode(&mut requests[owner]);
            }
        }

        let mut replies = vec![Vec::new(); self.context.size()];
        for (rank, bytes) in self.context.inner.exchange(&requests).iter().enumerate() {
            for request in bytes.chunks_exact(16) {
                let position = u64::decode(&request[..8]);
                let key = u64::decode(&request[8..]) as usize;
                let offset = self
                    .distribution
                    .local_offset(self.context.rank(), key);
                position.encode(&mut replies[rank]);
                self.data[offset].encode(&mut replies[rank]);
            }
        }

        let mut output = vec![self.algebra.zero(); keys.len()];
        for bytes in self.context.inner.exchange(&replies) {
            for reply in bytes.chunks_exact(8 + A::Element::WIDTH) {
                let position = u64::decode(&reply[..8]) as usize;
                output[position] = A::Element::decode(&reply[8..]);
            }
        }
        for (value, sign) in output.iter_mut().zip(signs) {
            if sign == -1 {
                *value = self.algebra.negate(value);
            }
        }
        output
    }

    /// Collective additive writes. Equivalent permutations are reduced in
    /// source-rank/request order, with the first incoming value added before
    /// the old stored value, matching the sparse indexed-write convention.
    pub fn write_add(&mut self, pairs: &[(usize, A::Element)]) {
        let mut buckets = vec![Vec::new(); self.context.size()];
        for (key, value) in pairs {
            let Some((canonical, sign)) = self.distribution.canonicalize(*key) else {
                continue;
            };
            let value = if sign == 1 {
                value.clone()
            } else {
                self.algebra.negate(value)
            };
            for (rank, bucket) in buckets.iter_mut().enumerate() {
                if self.distribution.distribution().owns(rank, canonical) {
                    (canonical as u64).encode(bucket);
                    value.encode(bucket);
                }
            }
        }

        let mut incoming = Vec::new();
        for bytes in self.context.inner.exchange(&buckets) {
            for pair in bytes.chunks_exact(8 + A::Element::WIDTH) {
                incoming.push((
                    u64::decode(&pair[..8]) as usize,
                    A::Element::decode(&pair[8..]),
                ));
            }
        }
        // Stable sorting groups keys without changing source-rank/request order.
        incoming.sort_by_key(|pair| pair.0);
        let mut start = 0;
        while start < incoming.len() {
            let key = incoming[start].0;
            let offset = self
                .distribution
                .local_offset(self.context.rank(), key);
            let mut value = self.algebra.add(&incoming[start].1, &self.data[offset]);
            start += 1;
            while start < incoming.len() && incoming[start].0 == key {
                value = self.algebra.add(&value, &incoming[start].1);
                start += 1;
            }
            self.data[offset] = value;
        }
    }

    /// Redistribute canonical entries only, sending each source value from its
    /// canonical owner to every destination replica.
    pub fn redistribute(&self, target: SymmetricDistribution) -> Self {
        assert_eq!(
            target.distribution().shape,
            self.distribution.distribution().shape
        );
        assert_eq!(target.links(), self.distribution.links());
        assert_eq!(target.distribution().topology.size(), self.context.size());

        let pairs: Vec<_> = self
            .local_pairs()
            .into_iter()
            .filter(|(key, _)| {
                self.distribution.distribution().owner(*key) == self.context.rank()
            })
            .collect();
        let mut result = Self::new(self.context, target, self.algebra.clone());
        result.write_add(&pairs);
        result
    }
}
