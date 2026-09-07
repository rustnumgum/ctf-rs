//! Collective distributed expansion of canonical symmetric storage.

use crate::{
    algebra::{Group, Wire},
    mapping::Distribution,
    tensor::Tensor,
};
use super::SymmetricTensor;

impl<'c, 'r, A: Group + Clone> SymmetricTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Collectively expand canonical symmetric values into a rectangular tensor.
    /// Indexed reads perform the canonical permutation/sign handling; only the
    /// requested target-local keys are materialized on this rank.
    pub fn unpack(&self, target: Distribution) -> Tensor<'c, 'r, A> {
        assert_eq!(target.shape, self.distribution().distribution().shape);
        assert_eq!(target.topology.size(), self.context().size());

        let rank = self.context().rank();
        let keys: Vec<_> = (0..target.local_len())
            .filter_map(|offset| target.global_key(rank, offset))
            .collect();
        let values = self.read(&keys);
        let mut result = Tensor::new(self.context(), target, self.algebra().clone());
        for (key, value) in keys.into_iter().zip(values) {
            let offset = result.distribution().local_offset(rank, key);
            result.data[offset] = value;
        }
        result
    }
}
