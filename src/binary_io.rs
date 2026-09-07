//! Dense binary tensor I/O adapted from `tensor/untyped_tensor.cxx`.
//!
//! The Rust sparse and symmetric paths keep the requested byte offset when
//! delegating through the dense logical domain; the upstream recursive helper
//! drops it for those storage kinds.

use std::path::Path;

use crate::{
    algebra::{Ring, Semiring, Wire},
    sparse::SparseTensor,
    symmetric_tensor::SymmetricTensor,
    tensor::Tensor,
};

fn chunk_bounds(total: usize, rank: usize, size: usize) -> (usize, usize) {
    let base = total / size;
    let remainder = total % size;
    let start = base * rank + rank.min(remainder);
    let length = base + usize::from(rank < remainder);
    (start, length)
}

fn byte_offset(offset: u64, start: usize, width: usize) -> u64 {
    offset + u64::try_from(start).unwrap() * u64::try_from(width).unwrap()
}

fn encode_values<T: Wire>(values: &[T]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * T::WIDTH);
    for value in values {
        value.encode(&mut bytes);
    }
    assert_eq!(bytes.len(), values.len() * T::WIDTH);
    bytes
}

impl<'c, 'r, A: Semiring + Clone> Tensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Write the logical dense tensor in balanced contiguous global-key chunks.
    pub fn write_dense_to_file(&self, path: &Path, offset: u64) {
        let (start, length) = chunk_bounds(
            self.distribution().global_len(),
            self.context().rank(),
            self.context().size(),
        );
        let keys: Vec<_> = (start..start + length).collect();
        let values = self.read(&keys);
        let bytes = encode_values(&values);
        self.context().inner.write_binary_at(
            path,
            byte_offset(offset, start, A::Element::WIDTH),
            &bytes,
        );
    }

    /// Replace every logical tensor entry from balanced contiguous file chunks.
    pub fn read_dense_from_file(&mut self, path: &Path, offset: u64) {
        let (start, length) = chunk_bounds(
            self.distribution().global_len(),
            self.context().rank(),
            self.context().size(),
        );
        let bytes = self.context().inner.read_binary_at(
            path,
            byte_offset(offset, start, A::Element::WIDTH),
            length * A::Element::WIDTH,
        );
        assert_eq!(bytes.len(), length * A::Element::WIDTH);
        let pairs: Vec<_> = (start..start + length)
            .zip(bytes.chunks_exact(A::Element::WIDTH))
            .map(|(key, bytes)| (key, A::Element::decode(bytes)))
            .collect();
        let one = self.algebra().one();
        let zero = self.algebra().zero();
        self.write_scaled(&pairs, &one, &zero);
    }
}

impl<'c, 'r, A: Semiring + Clone> SparseTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Write sparse values over the logical dense key domain without
    /// allocating a separate distributed dense tensor.
    pub fn write_dense_to_file(&self, path: &Path, offset: u64) {
        let (start, length) = chunk_bounds(
            self.distribution().global_len(),
            self.context().rank(),
            self.context().size(),
        );
        let keys: Vec<_> = (start..start + length).collect();
        let values = self.read(&keys);
        let bytes = encode_values(&values);
        self.context().inner.write_binary_at(
            path,
            byte_offset(offset, start, A::Element::WIDTH),
            &bytes,
        );
    }

    /// Read a logical dense file, then sparsify the resulting values in place.
    pub fn read_dense_from_file(&mut self, path: &Path, offset: u64) {
        let context = self.context();
        let distribution = self.distribution().clone();
        let algebra = self.algebra().clone();
        let zero = self.algebra().zero();
        let mut dense = Tensor::new(context, distribution, algebra);
        dense.read_dense_from_file(path, offset);
        *self = dense.into_sparse(|value| value != &zero);
    }
}

impl<'c, 'r, A: Ring + Clone> SymmetricTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Write the logical rectangular tensor, including structural zeros.
    pub fn write_dense_to_file(&self, path: &Path, offset: u64) {
        let (start, length) = chunk_bounds(
            self.distribution().distribution().global_len(),
            self.context().rank(),
            self.context().size(),
        );
        let keys: Vec<_> = (start..start + length).collect();
        let values = self.read(&keys);
        let bytes = encode_values(&values);
        self.context().inner.write_binary_at(
            path,
            byte_offset(offset, start, A::Element::WIDTH),
            &bytes,
        );
    }

    /// Read canonical entries from each rank's logical file chunk. Noncanonical
    /// orbit values and structural zeros are intentionally ignored.
    pub fn read_dense_from_file(&mut self, path: &Path, offset: u64) {
        let (start, length) = chunk_bounds(
            self.distribution().distribution().global_len(),
            self.context().rank(),
            self.context().size(),
        );
        let bytes = self.context().inner.read_binary_at(
            path,
            byte_offset(offset, start, A::Element::WIDTH),
            length * A::Element::WIDTH,
        );
        assert_eq!(bytes.len(), length * A::Element::WIDTH);
        let mut pairs = Vec::new();
        for (key, bytes) in (start..start + length).zip(bytes.chunks_exact(A::Element::WIDTH)) {
            if let Some((canonical, sign)) = self.distribution().canonicalize(key) {
                if canonical == key && sign == 1 {
                    pairs.push((canonical, A::Element::decode(bytes)));
                }
            }
        }
        let one = self.algebra().one();
        let zero = self.algebra().zero();
        self.write_scaled(&pairs, &one, &zero);
    }
}
