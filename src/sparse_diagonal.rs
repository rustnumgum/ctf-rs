// Adapted from cc4s CTF tensor::extract_diag and diagonal reinsertion.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{algebra::{Monoid, Wire}, diagonal::Projection,
    mapping::Distribution, sparse::SparseTensor};

impl<'c, 'r, A: Monoid + Clone> SparseTensor<'c, 'r, A> where A::Element: Wire {
    /// Select equal coordinates for repeated labels, retaining stored entries
    /// only. The resulting labels follow first occurrence in the input string.
    pub fn extract_diagonal(&self, labels: &str) -> (Self, String) {
        let projection = Projection::new(&self.distribution().shape, labels);
        if !projection.repeated() { return (self.clone(), projection.labels); }
        let mut result = Self::new(self.context(),
            Distribution::cyclic(projection.shape.clone(), self.context().size()), self.algebra().clone());
        let mut pairs = Vec::new();
        for (key, value) in self.local_pairs() {
            if self.distribution().owner(key) != self.context().rank() { continue; }
            if let Some(coordinates) = projection.project(&self.distribution().decode_key(key)) {
                pairs.push((result.distribution().encode_key(&coordinates), value));
            }
        }
        result.write_add(&pairs);
        (result, projection.labels)
    }

    /// Replace the selected sparse diagonal, including its structure, while
    /// preserving all off-diagonal keys and the original distribution.
    pub fn replace_diagonal(&mut self, labels: &str, input: &Self) {
        assert!(std::ptr::eq(self.context(), input.context()));
        let projection = Projection::new(&self.distribution().shape, labels);
        assert_eq!(projection.shape, input.distribution().shape);
        let mut pairs = Vec::new();
        for (key, value) in input.local_pairs() {
            if input.distribution().owner(key) != self.context().rank() { continue; }
            let coordinates = projection.expand(&input.distribution().decode_key(key));
            pairs.push((self.distribution().encode_key(&coordinates), value));
        }
        let distribution = &self.distribution;
        for block in &mut self.blocks {
            block.retain(|(key, _)| projection.project(&distribution.decode_key(*key)).is_none());
        }
        self.write_add(&pairs);
    }
}
