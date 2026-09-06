// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Distributed dense storage and key-based all-to-all read/write.
use crate::{algebra::{Monoid, Semiring, Wire}, context::Context, mapping::Distribution};

pub struct Tensor<'context, 'runtime, A: Monoid> {
    context: &'context Context<'runtime>,
    algebra: A,
    distribution: Distribution,
    data: Vec<A::Element>,
}

impl<'c, 'r, A: Monoid> Tensor<'c, 'r, A> {
    pub fn new(context: &'c Context<'r>, distribution: Distribution, algebra: A) -> Self {
        assert_eq!(context.size(), distribution.topology.size());
        let data = vec![algebra.zero(); distribution.local_len()];
        Self { context, algebra, distribution, data }
    }
    pub fn context(&self) -> &'c Context<'r> { self.context }
    pub fn algebra(&self) -> &A { &self.algebra }
    pub fn distribution(&self) -> &Distribution { &self.distribution }
    pub fn local_storage(&self) -> &[A::Element] { &self.data }
    pub fn local_pairs(&self) -> Vec<(usize, A::Element)> {
        self.data.iter().enumerate().filter_map(|(offset, value)| {
            self.distribution.global_key(self.context.rank(), offset).map(|key| (key, value.clone()))
        }).collect()
    }
    pub fn transform(&mut self, mut function: impl FnMut(usize, &mut A::Element)) {
        for (offset, value) in self.data.iter_mut().enumerate() {
            if let Some(key) = self.distribution.global_key(self.context.rank(), offset) { function(key, value); }
        }
    }
    /// Apply only where equal index labels have equal coordinates. This is the
    /// diagonal selection used by upstream sequential scaling/endomorphisms.
    pub fn transform_indexed(&mut self, labels: &str, mut function: impl FnMut(&mut A::Element)) {
        assert!(labels.is_ascii());
        let labels = labels.as_bytes();
        assert_eq!(labels.len(), self.distribution.shape.len());
        for i in 0..labels.len() {
            for j in 0..i {
                if labels[i] == labels[j] { assert_eq!(self.distribution.shape[i], self.distribution.shape[j]); }
            }
        }
        for (offset, value) in self.data.iter_mut().enumerate() {
            if let Some(key) = self.distribution.global_key(self.context.rank(), offset) {
                let coordinates = self.distribution.decode_key(key);
                if (0..labels.len()).all(|i| (0..i).all(|j| labels[i] != labels[j] || coordinates[i] == coordinates[j])) {
                    function(value);
                }
            }
        }
    }
}

impl<'c, 'r, A: Monoid + Clone> Tensor<'c, 'r, A> where A::Element: Wire {
    /// Collective dense slice, retaining physical/virtual mappings. Extract the
    /// local sub-block, then shift its owner by offsets modulo physical phases,
    /// as in upstream redistribution/slice.cxx. No global tensor is gathered.
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Self {
        assert_eq!(ranges.len(), self.distribution.shape.len());
        for (range, &n) in ranges.iter().zip(&self.distribution.shape) {
            assert!(range.start <= range.end && range.end <= n);
        }
        let offsets: Vec<_> = ranges.iter().map(|r|r.start).collect();
        let mut distribution = self.distribution.clone();
        distribution.shape = ranges.iter().map(|r|r.end-r.start).collect();
        let destination = self.distribution.shifted_rank(self.context.rank(), &offsets, false);
        let source = self.distribution.shifted_rank(self.context.rank(), &offsets, true);
        let mut result = Self::new(self.context, distribution, self.algebra.clone());
        for (key, value) in self.local_pairs() {
            let coordinates = self.distribution.decode_key(key);
            if coordinates.iter().zip(ranges).all(|(&c,r)|r.contains(&c)) {
                let sliced: Vec<_> = coordinates.iter().zip(&offsets).map(|(&c,&o)|c-o).collect();
                let new_key = result.distribution.encode_key(&sliced);
                let offset = result.distribution.local_offset(destination, new_key);
                result.data[offset] = value;
            }
        }
        if destination != self.context.rank() {
            let mut bytes = Vec::with_capacity(result.data.len()*A::Element::WIDTH);
            for value in &result.data { value.encode(&mut bytes); }
            let mut received = vec![0u8; bytes.len()];
            // For a globally empty slice no rank needs a data exchange.
            if !bytes.is_empty() { self.context.inner.send_receive(&bytes,destination,source,&mut received); }
            for (value, bytes) in result.data.iter_mut().zip(received.chunks_exact(A::Element::WIDTH)) {
                *value = A::Element::decode(bytes);
            }
        }
        result
    }
    /// Reorder tensor axes and their mappings together. Physical ownership is
    /// unchanged, so this transpose is local and has no implicit MPI collective.
    pub fn permute_axes(&self, axes: &[usize]) -> Self {
        let order = self.distribution.shape.len();
        assert_eq!(axes.len(),order);
        let mut seen = vec![false;order];
        for &axis in axes { assert!(axis < order && !seen[axis]); seen[axis] = true; }
        let distribution = Distribution::new(axes.iter().map(|&i|self.distribution.shape[i]).collect(),
            self.distribution.topology.clone(), axes.iter().map(|&i|self.distribution.mappings[i].clone()).collect());
        let mut result = Self::new(self.context,distribution,self.algebra.clone());
        for (key,value) in self.local_pairs() {
            let coordinates = self.distribution.decode_key(key);
            let permuted: Vec<_> = axes.iter().map(|&i|coordinates[i]).collect();
            let key = result.distribution.encode_key(&permuted);
            let offset = result.distribution.local_offset(self.context.rank(),key);
            result.data[offset] = value;
        }
        result
    }
}

impl<A: Semiring> Tensor<'_, '_, A> {
    pub fn scale(&mut self, alpha: &A::Element) {
        for (offset, value) in self.data.iter_mut().enumerate() {
            if self.distribution.global_key(self.context.rank(), offset).is_some() {
                *value = self.algebra.multiply(alpha, value);
            }
        }
    }
}

impl<A: Monoid> Tensor<'_, '_, A> where A::Element: Wire {
    /// Collective random access. Every rank participates, even with no requests.
    pub fn read(&self, keys: &[usize]) -> Vec<A::Element> {
        let mut requests = vec![Vec::new(); self.context.size()];
        for (position, &key) in keys.iter().enumerate() {
            let bucket = &mut requests[self.distribution.owner(key)];
            (position as u64).encode(bucket); (key as u64).encode(bucket);
        }
        let received = self.context.inner.exchange(&requests);
        let mut replies = vec![Vec::new(); self.context.size()];
        for (rank, bytes) in received.iter().enumerate() {
            for request in bytes.chunks_exact(16) {
                let position = u64::decode(&request[..8]);
                let key = u64::decode(&request[8..]) as usize;
                let offset = self.distribution.local_offset(self.context.rank(), key);
                position.encode(&mut replies[rank]); self.data[offset].encode(&mut replies[rank]);
            }
        }
        let received = self.context.inner.exchange(&replies);
        let mut result = vec![self.algebra.zero(); keys.len()];
        for bytes in received {
            for reply in bytes.chunks_exact(8 + A::Element::WIDTH) {
                let position = u64::decode(&reply[..8]) as usize;
                result[position] = A::Element::decode(&reply[8..]);
            }
        }
        result
    }
    /// Collective additive writes. Duplicate keys are reduced in rank order.
    pub fn write_add(&mut self, pairs: &[(usize, A::Element)]) {
        let mut buckets = vec![Vec::new(); self.context.size()];
        for (key, value) in pairs {
            for (rank, bucket) in buckets.iter_mut().enumerate() {
                if self.distribution.owns(rank, *key) {
                    (*key as u64).encode(bucket); value.encode(bucket);
                }
            }
        }
        for bytes in self.context.inner.exchange(&buckets) {
            for pair in bytes.chunks_exact(8 + A::Element::WIDTH) {
                let key = u64::decode(&pair[..8]) as usize;
                let offset = self.distribution.local_offset(self.context.rank(), key);
                self.data[offset] = self.algebra.add(&self.data[offset], &A::Element::decode(&pair[8..]));
            }
        }
    }
    /// Collective distribution switch, sending each unique entry once per new
    /// replica. Only local data and communication buckets are allocated.
    pub fn redistribute(&mut self, distribution: Distribution) {
        assert_eq!(distribution.shape, self.distribution.shape);
        assert_eq!(distribution.topology.size(), self.context.size());
        let mut buckets = vec![Vec::new(); self.context.size()];
        for (key, value) in self.local_pairs() {
            if self.distribution.owner(key) != self.context.rank() { continue; }
            for (rank, bucket) in buckets.iter_mut().enumerate() {
                if distribution.owns(rank, key) {
                    (key as u64).encode(bucket); value.encode(bucket);
                }
            }
        }
        let received = self.context.inner.exchange(&buckets);
        let mut data = vec![self.algebra.zero(); distribution.local_len()];
        for bytes in received {
            for pair in bytes.chunks_exact(8 + A::Element::WIDTH) {
                let key = u64::decode(&pair[..8]) as usize;
                data[distribution.local_offset(self.context.rank(), key)] = A::Element::decode(&pair[8..]);
            }
        }
        self.distribution = distribution; self.data = data;
    }
    pub fn reduce(&self) -> A::Element {
        let mut value = self.algebra.zero();
        for (key, entry) in self.local_pairs() {
            if self.distribution.owner(key) == self.context.rank() { value = self.algebra.add(&value, &entry); }
        }
        self.context.all_reduce(&self.algebra, &value)
    }
}
