// Adapted from tensor/untyped_tensor.cxx reshape_tensor's key-based path.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Collective column-major reshape using canonical key redistribution.
use crate::{algebra::{Monoid, Wire}, mapping::Distribution, tensor::Tensor};

impl<'c, 'r, A: Monoid + Clone> Tensor<'c, 'r, A>
where A::Element: Wire {
    /// Preserve flattened column-major keys while changing shape and layout.
    /// Only canonical owners send; destination replicas are populated by the
    /// distributed write. No full tensor gather or communicating destructor.
    pub fn reshape(&self, target: Distribution) -> Self {
        assert_eq!(self.distribution().shape.iter().product::<usize>(),
            target.shape.iter().product::<usize>());
        assert_eq!(target.topology.size(), self.context().size());
        let pairs: Vec<_> = self.local_pairs().into_iter()
            .filter(|(key, _)| self.distribution().owner(*key) == self.context().rank()).collect();
        let mut result = Self::new(self.context(), target, self.algebra().clone());
        result.write_add(&pairs);
        result
    }
}
