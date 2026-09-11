// Adapted from cc4s CTF sparse unary functions and accumulator transforms.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{algebra::{Monoid, Wire}, diagonal::Projection, mapping::Distribution,
    sparse::SparseTensor, tensor::Tensor};

impl<'c, 'r, A: Monoid> SparseTensor<'c, 'r, A> {
    /// Apply a typed unary function to stored entries only. The distribution
    /// and explicit sparse structure are preserved, even when f(value) is zero.
    /// This operation is local and does not require an MPI wire format for B.
    pub fn map_stored<B: Monoid>(&self, algebra: B,
        mut function: impl FnMut(&A::Element) -> B::Element) -> SparseTensor<'c, 'r, B> {
        let mut result = SparseTensor::new(self.context(), self.distribution().clone(), algebra);
        for (target, source) in result.blocks.iter_mut().zip(&self.blocks) {
            target.extend(source.iter().map(|(key, value)| (*key, function(value))));
        }
        result
    }

    /// Accumulate a dense operand into existing sparse entries, without adding
    /// any new sparse keys. Input labels must occur in the output labels;
    /// repeated output labels restrict the transform to its indexed diagonal.
    /// Dense additive-identity values are skipped: upstream sparsifies dense A
    /// before executing its sparse accumulator kernel.
    /// Collective indexed reads communicate only input entries needed locally.
    pub fn accumulate_from_dense<I: Monoid>(&mut self, indices_b: &str,
        a: &Tensor<'_, '_, I>, indices_a: &str,
        mut function: impl FnMut(&I::Element, &mut A::Element)) where I::Element: Wire {
        assert!(std::ptr::eq(self.context(), a.context()));
        let (keys, positions) = self.accumulator_keys(indices_b, a.distribution(), indices_a);
        let values = a.read(&keys);
        for ((block, offset), value) in positions.into_iter().zip(values) {
            if value != a.algebra().zero() {
                function(&value, &mut self.blocks[block][offset].1);
            }
        }
    }

    /// Sparse-input accumulator: absent input entries do not invoke the
    /// function, while explicit stored zeros do. Output structure is unchanged.
    pub fn accumulate_from_sparse<I: Monoid>(&mut self, indices_b: &str,
        a: &SparseTensor<'_, '_, I>, indices_a: &str,
        mut function: impl FnMut(&I::Element, &mut A::Element)) where I::Element: Wire {
        assert!(std::ptr::eq(self.context(), a.context()));
        let (keys, positions) = self.accumulator_keys(indices_b, a.distribution(), indices_a);
        let mut requests = vec![Vec::new(); self.context().size()];
        for (position, &key) in keys.iter().enumerate() {
            let bucket = &mut requests[a.distribution().owner(key)];
            (position as u64).encode(bucket);
            (key as u64).encode(bucket);
        }
        let mut replies = vec![Vec::new(); self.context().size()];
        for (rank, bytes) in self.context().inner.exchange(&requests).iter().enumerate() {
            for request in bytes.chunks_exact(16) {
                let position = u64::decode(&request[..8]);
                let key = u64::decode(&request[8..]) as usize;
                let block = &a.blocks[a.block(key)];
                if let Ok(index) = block.binary_search_by_key(&key, |pair| pair.0) {
                    position.encode(&mut replies[rank]);
                    block[index].1.encode(&mut replies[rank]);
                }
            }
        }
        let mut values = vec![None; positions.len()];
        for bytes in self.context().inner.exchange(&replies) {
            for reply in bytes.chunks_exact(8 + I::Element::WIDTH) {
                let position = u64::decode(&reply[..8]) as usize;
                values[position] = Some(I::Element::decode(&reply[8..]));
            }
        }
        for ((block, offset), value) in positions.into_iter().zip(values) {
            if let Some(value) = value { function(&value, &mut self.blocks[block][offset].1); }
        }
    }

    fn accumulator_keys(&self, indices_b: &str, a: &Distribution, indices_a: &str)
        -> (Vec<usize>, Vec<(usize, usize)>) {
        let output = Projection::new(&self.distribution().shape, indices_b);
        let input = Projection::new(&a.shape, indices_a);
        let axes: Vec<_> = input.labels.bytes().enumerate().map(|(axis, label)| {
            let source = output.labels.bytes().position(|candidate| candidate == label)
                .expect("accumulator input labels must occur in output");
            assert_eq!(input.shape[axis], output.shape[source]);
            source
        }).collect();
        let mut keys = Vec::new();
        let mut positions = Vec::new();
        for (block, pairs) in self.blocks.iter().enumerate() {
            for (offset, (key, _)) in pairs.iter().enumerate() {
                let Some(coordinates) = output.project(&self.distribution().decode_key(*key))
                    else { continue; };
                let input_coordinates: Vec<_> = axes.iter().map(|&axis| coordinates[axis]).collect();
                keys.push(a.encode_key(&input.expand(&input_coordinates)));
                positions.push((block, offset));
            }
        }
        (keys, positions)
    }
}
