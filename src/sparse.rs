// Adapted from cc4s CTF sparse key storage and redistribution/sparse_rw.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Distributed sparse tensor storage: sorted keys inside each virtual block.
use crate::{algebra::{Monoid, Semiring, Wire}, context::Context, mapping::Distribution};

#[path = "sparse_sum.rs"]
mod summation;

#[path = "sparse_gemm.rs"]
mod gemm;

#[derive(Clone)]
pub struct SparseTensor<'c, 'r, A: Monoid> {
    context: &'c Context<'r>,
    distribution: Distribution,
    algebra: A,
    blocks: Vec<Vec<(usize, A::Element)>>,
}

impl<'c, 'r, A: Monoid> SparseTensor<'c, 'r, A> {
    pub fn new(context: &'c Context<'r>, distribution: Distribution, algebra: A) -> Self {
        assert_eq!(context.size(), distribution.topology.size());
        let virtual_blocks = distribution.mappings.iter()
            .map(|mapping| mapping.phase() / mapping.physical_phase()).product();
        Self { context, distribution, algebra, blocks: vec![Vec::new(); virtual_blocks] }
    }
    pub fn context(&self) -> &'c Context<'r> { self.context }
    pub fn distribution(&self) -> &Distribution { &self.distribution }
    pub fn algebra(&self) -> &A { &self.algebra }
    /// Stored pairs in virtual-block order; keys are sorted within each block.
    pub fn local_pairs(&self) -> Vec<(usize, A::Element)> {
        self.blocks.iter().flatten().cloned().collect()
    }
    pub fn local_nnz(&self) -> usize { self.blocks.iter().map(Vec::len).sum() }
    /// Apply a local endomorphism only to stored values, preserving structure.
    /// Absent entries stay absent even if the function maps zero to nonzero.
    pub fn transform_stored(&mut self, mut function: impl FnMut(usize, &mut A::Element)) {
        for (key, value) in self.blocks.iter_mut().flatten() { function(*key, value); }
    }
    /// Remove stored entries rejected by the predicate. Writes and transforms
    /// deliberately retain explicit zeros, matching upstream's separate sparsify.
    pub fn sparsify(&mut self, mut keep: impl FnMut(&A::Element) -> bool) {
        for block in &mut self.blocks { block.retain(|(_, value)| keep(value)); }
    }
    fn block(&self, key: usize) -> usize {
        self.distribution.local_offset(self.context.rank(), key)
            / self.distribution.block_shape().iter().product::<usize>()
    }
}

impl<A: Monoid> SparseTensor<'_, '_, A> where A::Element: Wire {
    /// Collective indexed reads; absent entries return the algebra's identity.
    pub fn read(&self, keys: &[usize]) -> Vec<A::Element> {
        let mut requests = vec![Vec::new(); self.context.size()];
        for (position, &key) in keys.iter().enumerate() {
            assert!(key < self.distribution.global_len());
            let bucket = &mut requests[self.distribution.owner(key)];
            (position as u64).encode(bucket);
            (key as u64).encode(bucket);
        }
        let mut replies = vec![Vec::new(); self.context.size()];
        for (rank, bytes) in self.context.inner.exchange(&requests).iter().enumerate() {
            for request in bytes.chunks_exact(16) {
                let position = u64::decode(&request[..8]);
                let key = u64::decode(&request[8..]) as usize;
                let block = &self.blocks[self.block(key)];
                let value = match block.binary_search_by_key(&key, |pair| pair.0) {
                    Ok(index) => block[index].1.clone(),
                    Err(_) => self.algebra.zero(),
                };
                position.encode(&mut replies[rank]);
                value.encode(&mut replies[rank]);
            }
        }
        let mut output = vec![self.algebra.zero(); keys.len()];
        for bytes in self.context.inner.exchange(&replies) {
            for reply in bytes.chunks_exact(8 + A::Element::WIDTH) {
                let position = u64::decode(&reply[..8]) as usize;
                output[position] = A::Element::decode(&reply[8..]);
            }
        }
        output
    }

    fn receive_updates(&self, pairs: &[(usize, A::Element)]) -> Vec<Vec<(usize, A::Element)>> {
        let mut buckets = vec![Vec::new(); self.context.size()];
        for (key, value) in pairs {
            assert!(*key < self.distribution.global_len());
            for (rank, bucket) in buckets.iter_mut().enumerate() {
                if self.distribution.owns(rank, *key) {
                    (*key as u64).encode(bucket);
                    value.encode(bucket);
                }
            }
        }
        let mut updates = vec![Vec::new(); self.blocks.len()];
        for bytes in self.context.inner.exchange(&buckets) {
            for pair in bytes.chunks_exact(8 + A::Element::WIDTH) {
                let key = u64::decode(&pair[..8]) as usize;
                updates[self.block(key)].push((key, A::Element::decode(&pair[8..])));
            }
        }
        updates
    }
    /// Additive writes reduce duplicate keys in source-rank/request order.
    pub fn write_add(&mut self, pairs: &[(usize, A::Element)]) {
        let updates = self.receive_updates(pairs);
        for (block, mut incoming) in self.blocks.iter_mut().zip(updates) {
            // Stable sorting retains addition order for a custom monoid.
            incoming.sort_by_key(|pair| pair.0);
            let mut old = std::mem::take(block).into_iter().peekable();
            let mut previous = None;
            for (key, value) in incoming {
                while old.peek().is_some_and(|pair| pair.0 < key) {
                    block.push(old.next().unwrap());
                }
                if previous == Some(key) {
                    let last = block.last_mut().unwrap();
                    last.1 = self.algebra.add(&last.1, &value);
                } else {
                    let value = if old.peek().is_some_and(|pair| pair.0 == key) {
                        self.algebra.add(&value, &old.next().unwrap().1)
                    } else { value };
                    block.push((key, value));
                }
                previous = Some(key);
            }
            block.extend(old);
        }
    }

    /// Redistribute stored pairs only, from canonical source owners to replicas.
    pub fn redistribute(&mut self, target: Distribution) {
        assert_eq!(target.shape, self.distribution.shape);
        assert_eq!(target.topology.size(), self.context.size());
        let mut buckets = vec![Vec::new(); self.context.size()];
        for (key, value) in self.blocks.iter().flatten() {
            if self.distribution.owner(*key) != self.context.rank() { continue; }
            for (rank, bucket) in buckets.iter_mut().enumerate() {
                if target.owns(rank, *key) {
                    (*key as u64).encode(bucket);
                    value.encode(bucket);
                }
            }
        }
        let virtual_blocks = target.mappings.iter()
            .map(|mapping| mapping.phase() / mapping.physical_phase()).product();
        self.distribution = target;
        self.blocks = vec![Vec::new(); virtual_blocks];
        for bytes in self.context.inner.exchange(&buckets) {
            for pair in bytes.chunks_exact(8 + A::Element::WIDTH) {
                let key = u64::decode(&pair[..8]) as usize;
                let block = self.block(key);
                self.blocks[block].push((key, A::Element::decode(&pair[8..])));
            }
        }
        for block in &mut self.blocks { block.sort_by_key(|pair| pair.0); }
    }
    pub fn reduce(&self) -> A::Element {
        let mut value = self.algebra.zero();
        for (key, entry) in self.blocks.iter().flatten() {
            if self.distribution.owner(*key) == self.context.rank() {
                value = self.algebra.add(&value, entry);
            }
        }
        self.context.all_reduce(&self.algebra, &value)
    }
}

impl<A: Semiring> SparseTensor<'_, '_, A> {
    /// Scale stored values; structural zeros remain until explicitly sparsified.
    pub fn scale(&mut self, alpha: &A::Element) {
        for (_, value) in self.blocks.iter_mut().flatten() {
            *value = self.algebra.multiply(alpha, value);
        }
    }
    /// Right-scale stored entries selected by repeated-index constraints,
    /// matching the sparse summation kernel's value*beta convention.
    pub fn scale_indexed(&mut self, labels: &str, alpha: &A::Element) {
        let projection = crate::diagonal::Projection::new(&self.distribution.shape, labels);
        for (key, value) in self.blocks.iter_mut().flatten() {
            if projection.project(&self.distribution.decode_key(*key)).is_some() {
                *value = self.algebra.multiply(value, alpha);
            }
        }
    }
}

impl<A: Semiring> SparseTensor<'_, '_, A> where A::Element: Wire {
    /// For requested keys only, combine first_request*alpha with old*beta,
    /// then append further requests, preserving sparse_rw's algebra order.
    /// Old-only keys are unchanged; duplicate requests apply beta exactly once.
    pub fn write_scaled(&mut self, pairs: &[(usize, A::Element)],
        alpha: &A::Element, beta: &A::Element) {
        let updates = self.receive_updates(pairs);
        for (block, mut incoming) in self.blocks.iter_mut().zip(updates) {
            incoming.sort_by_key(|pair| pair.0);
            let mut old = std::mem::take(block).into_iter().peekable();
            let mut previous = None;
            for (key, value) in incoming {
                while old.peek().is_some_and(|pair| pair.0 < key) {
                    block.push(old.next().unwrap());
                }
                let value = self.algebra.multiply(&value, alpha);
                if previous == Some(key) {
                    let last = block.last_mut().unwrap();
                    last.1 = self.algebra.add(&last.1, &value);
                } else {
                    let value = if old.peek().is_some_and(|pair| pair.0 == key) {
                        let old = old.next().unwrap().1;
                        self.algebra.add(&value, &self.algebra.multiply(&old, beta))
                    } else { value };
                    block.push((key, value));
                }
                previous = Some(key);
            }
            block.extend(old);
        }
    }
}

impl<'c, 'r, A: Monoid + Clone> SparseTensor<'c, 'r, A> where A::Element: Wire {
    /// Reindex stored entries without changing physical ownership.
    pub fn permute_axes(&self, axes: &[usize]) -> Self {
        assert_eq!(axes.len(), self.distribution.shape.len());
        let mut seen = vec![false; axes.len()];
        for &axis in axes { assert!(axis < axes.len() && !seen[axis]); seen[axis] = true; }
        let target = Distribution::new(axes.iter().map(|&axis| self.distribution.shape[axis]).collect(),
            self.distribution.topology.clone(),
            axes.iter().map(|&axis| self.distribution.mappings[axis].clone()).collect());
        let mut result = Self::new(self.context, target, self.algebra.clone());
        for (key, value) in self.blocks.iter().flatten() {
            let coordinates = self.distribution.decode_key(*key);
            let key = result.distribution.encode_key(&axes.iter().map(|&axis| coordinates[axis]).collect::<Vec<_>>());
            let block = result.block(key);
            result.blocks[block].push((key, value.clone()));
        }
        for block in &mut result.blocks { block.sort_by_key(|pair| pair.0); }
        result
    }
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Self {
        assert_eq!(ranges.len(), self.distribution.shape.len());
        for (range, &length) in ranges.iter().zip(&self.distribution.shape) {
            assert!(range.start <= range.end && range.end <= length);
        }
        let mut target = self.distribution.clone();
        target.shape = ranges.iter().map(|range| range.end - range.start).collect();
        let mut result = Self::new(self.context, target, self.algebra.clone());
        let mut pairs = Vec::new();
        for (key, value) in self.blocks.iter().flatten() {
            if self.distribution.owner(*key) != self.context.rank() { continue; }
            let coordinates = self.distribution.decode_key(*key);
            if coordinates.iter().zip(ranges).all(|(&coordinate, range)| range.contains(&coordinate)) {
                let coordinates: Vec<_> = coordinates.iter().zip(ranges)
                    .map(|(&coordinate, range)| coordinate - range.start).collect();
                pairs.push((result.distribution.encode_key(&coordinates), value.clone()));
            }
        }
        result.write_add(&pairs);
        result
    }
}
