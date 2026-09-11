// Adapted from pinned CTF sparse symmetric summation and vecnorm semantics.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{
    algebra::{Arithmetic, Group, Semiring, Wire},
    sparse_symmetric::SparseSymmetricTensor,
    symmetric_tensor::SymmetricTensor,
};

impl SparseSymmetricTensor<'_, '_, Arithmetic<f64>> {
    /// Logical full-orbit norm from primary canonical sparse entries.
    pub fn norm2(&self) -> f64 {
        let mut squared = self.local_pairs().into_iter()
            .map(|(key, value)| value * value * self.orbit_multiplicity(key) as f64)
            .sum::<f64>();
        self.context().sum_f64(std::slice::from_mut(&mut squared));
        squared.sqrt()
    }
}

impl<'c, 'r, A: Group + Semiring + Clone> SymmetricTensor<'c, 'r, A>
where A::Element: Wire {
    /// Add a same-index sparse symmetric tensor to packed dense storage.
    /// Only local canonical destination keys are requested; absent sparse
    /// values contribute the additive identity and padding remains unchanged.
    pub fn add_sparse(&mut self, source: &SparseSymmetricTensor<'c, 'r, A>, alpha: A::Element) {
        assert!(std::ptr::eq(self.context(), source.context()));
        assert_eq!(self.distribution().distribution().shape,
                   source.distribution().distribution().shape);
        assert_eq!(self.distribution().links(), source.distribution().links());
        let pairs = self.local_pairs();
        let keys: Vec<_> = pairs.iter().map(|(key, _)| *key).collect();
        let incoming = source.read(&keys);
        let values: Vec<_> = pairs.into_iter().zip(incoming)
            .map(|((_, old), value)| self.algebra().add(&old, &self.algebra().multiply(&value, &alpha)))
            .collect();
        self.set_local_canonical_storage(&values);
    }
}
