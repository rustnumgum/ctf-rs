//! Explicit replicated pair/data reads adapted from `untyped_tensor.cxx`.

use crate::{
    algebra::{Group, Monoid, Wire},
    context::Context,
    mapping::Distribution,
    sparse::SparseTensor,
    symmetric_tensor::SymmetricTensor,
    tensor::Tensor,
};

fn gather_pairs<E: Wire>(context: &Context<'_>, pairs: Vec<(usize, E)>) -> Vec<(usize, E)> {
    let width = 8 + E::WIDTH;
    let mut bytes = Vec::with_capacity(pairs.len() * width);
    for (key, value) in pairs {
        (key as u64).encode(&mut bytes);
        value.encode(&mut bytes);
    }
    let gathered = context.inner.all_gather_bytes(&bytes);
    assert_eq!(gathered.len() % width, 0);
    let mut result = Vec::with_capacity(gathered.len() / width);
    for pair in gathered.chunks_exact(width) {
        result.push((u64::decode(&pair[..8]) as usize, E::decode(&pair[8..])));
    }
    result.sort_by_key(|(key, _)| *key);
    result
}

impl<'c, 'r, A: Monoid + Clone> Tensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Replicate sorted canonical key/value pairs on every rank.
    pub fn all_pairs(&self, nonzeros_only: bool) -> Vec<(usize, A::Element)> {
        if self.distribution().global_len() == 0 {
            return Vec::new();
        }
        let rank = self.context().rank();
        let zero = self.algebra().zero();
        let pairs = self
            .local_pairs()
            .into_iter()
            .filter(|(key, value)| {
                self.distribution().owner(*key) == rank
                    && (!nonzeros_only || value != &zero)
            })
            .collect();
        gather_pairs(self.context(), pairs)
    }

    /// Replicate sorted canonical values on every rank.
    pub fn all_data(&self) -> Vec<A::Element> {
        self.all_pairs(false)
            .into_iter()
            .map(|(_, value)| value)
            .collect()
    }
}

impl<'c, 'r, A: Monoid + Clone> SparseTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Replicate sparse pairs on every rank. Stored sparse zeros are retained
    /// when `nonzeros_only` is true; the dense path is used for false.
    pub fn all_pairs(&self, nonzeros_only: bool) -> Vec<(usize, A::Element)> {
        if self.distribution().global_len() == 0 {
            return Vec::new();
        }
        if !nonzeros_only {
            return self.clone().into_dense().all_pairs(false);
        }
        let rank = self.context().rank();
        let pairs = self
            .local_pairs()
            .into_iter()
            .filter(|(key, _)| self.distribution().owner(*key) == rank)
            .collect();
        gather_pairs(self.context(), pairs)
    }

    /// Replicate the full dense values, including implicit zeros, on every rank.
    pub fn all_data(&self) -> Vec<A::Element> {
        self.all_pairs(false)
            .into_iter()
            .map(|(_, value)| value)
            .collect()
    }
}

impl<'c, 'r, A: Group + Clone> SymmetricTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Replicate symmetric pairs on every rank. Nonzero reads remain packed
    /// canonical pairs regardless of `unpack_sym`.
    pub fn all_pairs(
        &self,
        nonzeros_only: bool,
        unpack_sym: bool,
    ) -> Vec<(usize, A::Element)> {
        let distribution = self.distribution().distribution();
        if distribution.global_len() == 0 {
            return Vec::new();
        }
        if !nonzeros_only && unpack_sym {
            return self
                .unpack(Distribution::cyclic(
                    distribution.shape.clone(),
                    self.context().size(),
                ))
                .all_pairs(false);
        }
        let rank = self.context().rank();
        let zero = self.algebra().zero();
        let pairs = self
            .local_pairs()
            .into_iter()
            .filter(|(key, value)| {
                distribution.owner(*key) == rank
                    && (!nonzeros_only || value != &zero)
            })
            .collect();
        gather_pairs(self.context(), pairs)
    }

    /// Replicate symmetric pair values on every rank.
    pub fn all_data(&self, unpack_sym: bool) -> Vec<A::Element> {
        self.all_pairs(false, unpack_sym)
            .into_iter()
            .map(|(_, value)| value)
            .collect()
    }
}
