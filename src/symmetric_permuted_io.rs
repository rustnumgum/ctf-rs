//! Dense/symmetric permutation transfers adapted from
//! `tensor/untyped_tensor.cxx::permute`.

use crate::{
    algebra::{Ring, Wire},
    cyclic_reshuffle::visit_local_keys,
    mapping::Distribution,
    permuted_io::{mapped_key, validate_maps, validate_parent_maps},
    symmetric_tensor::SymmetricTensor,
    tensor::Tensor,
};

fn assert_injective_maps(maps: &[Vec<Option<usize>>]) {
    for coordinates in maps {
        let retained: Vec<_> = coordinates.iter().flatten().copied().collect();
        let mut unique = retained.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), retained.len(), "gather coordinate maps must be injective");
    }
}

fn full_child_requests(
    parent: &Distribution,
    child: &Distribution,
    maps: &[Vec<Option<usize>>],
    rank: usize,
) -> (Vec<usize>, Vec<usize>) {
    let full = Distribution::cyclic(child.shape.clone(), child.topology.size());
    let mut parent_keys = Vec::new();
    let mut child_keys = Vec::new();
    visit_local_keys(&full, rank, |child_key| {
        if full.owner(child_key) != rank {
            return;
        }
        if let Some(parent_key) = mapped_key(parent, &full, maps, child_key) {
            parent_keys.push(parent_key);
            child_keys.push(child_key);
        }
    });
    (parent_keys, child_keys)
}

fn local_dense_child_requests(
    parent: &Distribution,
    child: &Distribution,
    maps: &[Vec<Option<usize>>],
    rank: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut parent_keys = Vec::new();
    let mut child_keys = Vec::new();
    for offset in 0..child.local_len() {
        let Some(child_key) = child.global_key(rank, offset) else {
            continue;
        };
        if child.owner(child_key) != rank {
            continue;
        }
        if let Some(parent_key) = mapped_key(parent, child, maps, child_key) {
            parent_keys.push(parent_key);
            child_keys.push(child_key);
        }
    }
    (parent_keys, child_keys)
}

impl<'c, 'r, A: Ring + Clone> SymmetricTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Gather a symmetric parent through the full logical child domain. The
    /// target's packed writer performs the child symmetry canonicalization.
    pub fn gather_permuted_into(
        &self,
        destination: Option<&mut SymmetricTensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        let parent = self.distribution().distribution();
        assert_injective_maps(maps);
        validate_parent_maps(parent, maps);
        assert!(destination
            .as_ref()
            .is_none_or(|destination| destination.context().size() <= self.context().size()));

        let (parent_keys, child_keys) = if let Some(destination) = destination.as_ref() {
            let child = destination.distribution().distribution();
            validate_maps(parent, child, maps);
            full_child_requests(parent, child, maps, destination.context().rank())
        } else {
            (Vec::new(), Vec::new())
        };
        let values = self.read(&parent_keys);
        if let Some(destination) = destination {
            let pairs: Vec<_> = child_keys.into_iter().zip(values).collect();
            destination.write_scaled(&pairs, &alpha, &beta);
        }
    }

    /// Gather a symmetric parent into a dense child over the child's own
    /// canonical local entries, without creating mirrored child values.
    pub fn gather_permuted_into_dense(
        &self,
        destination: Option<&mut Tensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        let parent = self.distribution().distribution();
        assert_injective_maps(maps);
        validate_parent_maps(parent, maps);
        assert!(destination
            .as_ref()
            .is_none_or(|destination| destination.context().size() <= self.context().size()));

        let (parent_keys, child_keys) = if let Some(destination) = destination.as_ref() {
            let child = destination.distribution();
            validate_maps(parent, child, maps);
            local_dense_child_requests(parent, child, maps, destination.context().rank())
        } else {
            (Vec::new(), Vec::new())
        };
        let values = self.read(&parent_keys);
        if let Some(destination) = destination {
            let pairs: Vec<_> = child_keys.into_iter().zip(values).collect();
            destination.write_scaled(&pairs, &alpha, &beta);
        }
    }

    /// Scatter only canonical nonzero symmetric child entries into a
    /// symmetric parent; the target writer handles mapped orbit signs.
    pub fn scatter_permuted_from(
        &mut self,
        source: Option<&SymmetricTensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        let parent = self.distribution().distribution();
        validate_parent_maps(parent, maps);
        assert!(source
            .as_ref()
            .is_none_or(|source| source.context().size() <= self.context().size()));
        let mut pairs = Vec::new();
        if let Some(source) = source {
            let child = source.distribution().distribution();
            validate_maps(parent, child, maps);
            let rank = source.context().rank();
            let zero = source.algebra().zero();
            for (child_key, value) in source.local_pairs() {
                if child.owner(child_key) != rank || value == zero {
                    continue;
                }
                if let Some(parent_key) = mapped_key(parent, child, maps, child_key) {
                    pairs.push((parent_key, value));
                }
            }
        }
        self.write_scaled(&pairs, &alpha, &beta);
    }

    /// Scatter canonical nonzero dense child entries into a symmetric parent.
    pub fn scatter_permuted_from_dense(
        &mut self,
        source: Option<&Tensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        let parent = self.distribution().distribution();
        validate_parent_maps(parent, maps);
        assert!(source
            .as_ref()
            .is_none_or(|source| source.context().size() <= self.context().size()));
        let mut pairs = Vec::new();
        if let Some(source) = source {
            let child = source.distribution();
            validate_maps(parent, child, maps);
            let rank = source.context().rank();
            let zero = source.algebra().zero();
            for (child_key, value) in source.local_pairs() {
                if child.owner(child_key) != rank || value == zero {
                    continue;
                }
                if let Some(parent_key) = mapped_key(parent, child, maps, child_key) {
                    pairs.push((parent_key, value));
                }
            }
        }
        self.write_scaled(&pairs, &alpha, &beta);
    }
}

impl<'c, 'r, A: Ring + Clone> Tensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Gather a dense parent through the full logical child domain into a
    /// symmetric child; packed canonicalization happens in the target write.
    pub fn gather_permuted_into_symmetric(
        &self,
        destination: Option<&mut SymmetricTensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert_injective_maps(maps);
        validate_parent_maps(self.distribution(), maps);
        assert!(destination
            .as_ref()
            .is_none_or(|destination| destination.context().size() <= self.context().size()));

        let (parent_keys, child_keys) = if let Some(destination) = destination.as_ref() {
            let child = destination.distribution().distribution();
            validate_maps(self.distribution(), child, maps);
            full_child_requests(self.distribution(), child, maps, destination.context().rank())
        } else {
            (Vec::new(), Vec::new())
        };
        let values = self.read(&parent_keys);
        if let Some(destination) = destination {
            let pairs: Vec<_> = child_keys.into_iter().zip(values).collect();
            destination.write_scaled(&pairs, &alpha, &beta);
        }
    }

    /// Scatter canonical nonzero symmetric child entries into a dense parent.
    pub fn scatter_permuted_from_symmetric(
        &mut self,
        source: Option<&SymmetricTensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        validate_parent_maps(self.distribution(), maps);
        assert!(source
            .as_ref()
            .is_none_or(|source| source.context().size() <= self.context().size()));
        let mut pairs = Vec::new();
        if let Some(source) = source {
            let child = source.distribution().distribution();
            validate_maps(self.distribution(), child, maps);
            let rank = source.context().rank();
            let zero = source.algebra().zero();
            for (child_key, value) in source.local_pairs() {
                if child.owner(child_key) != rank || value == zero {
                    continue;
                }
                if let Some(parent_key) = mapped_key(self.distribution(), child, maps, child_key) {
                    pairs.push((parent_key, value));
                }
            }
        }
        self.write_scaled(&pairs, &alpha, &beta);
    }
}
