//! Sparse-containing parent/child coordinate permutation transfers following
//! `tensor/untyped_tensor.cxx::permute` and sparse_rw's canonical write rules.

use std::collections::BTreeSet;

use crate::{
    algebra::{Ring, Semiring, Wire},
    mapping::Distribution,
    permuted_io::{mapped_key, validate_maps, validate_parent_maps},
    sparse::SparseTensor,
    symmetric_tensor::SymmetricTensor,
    tensor::Tensor,
};

fn validate_gather_maps(maps: &[Vec<Option<usize>>]) {
    for coordinates in maps {
        let retained: Vec<_> = coordinates.iter().flatten().copied().collect();
        assert_eq!(
            retained.iter().copied().collect::<BTreeSet<_>>().len(),
            retained.len(),
            "gather coordinate maps must be injective"
        );
    }
}

impl<'c, 'r, A: Semiring + Clone> SparseTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Gather sparse parent values, including implicit zeros, into an optional
    /// dense child through injective child-to-parent coordinate maps.
    pub fn gather_permuted_into_dense(
        &self,
        destination: Option<&mut Tensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert!(destination
            .as_ref()
            .is_none_or(|destination| destination.context().size() <= self.context().size()));
        validate_parent_maps(self.distribution(), maps);
        validate_gather_maps(maps);

        let (parent_keys, child_keys) = if let Some(destination) = destination.as_ref() {
            validate_maps(self.distribution(), destination.distribution(), maps);
            let rank = destination.context().rank();
            let mut keys = Vec::new();
            for offset in 0..destination.local_storage().len() {
                let Some(child_key) = destination.distribution().global_key(rank, offset) else {
                    continue;
                };
                if destination.distribution().owner(child_key) != rank {
                    continue;
                }
                if let Some(parent_key) = mapped_key(
                    self.distribution(),
                    destination.distribution(),
                    maps,
                    child_key,
                ) {
                    keys.push((parent_key, child_key));
                }
            }
            (
                keys.iter().map(|&(parent, _)| parent).collect(),
                keys.iter().map(|&(_, child)| child).collect(),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        let values = self.read(&parent_keys);
        if let Some(destination) = destination {
            destination.write_scaled(
                &child_keys.into_iter().zip(values).collect::<Vec<_>>(),
                &alpha,
                &beta,
            );
        }
    }

    /// Scatter every canonically owned stored sparse child pair, including
    /// explicit zeros, into this sparse parent.
    pub fn scatter_permuted_from(
        &mut self,
        source: Option<&SparseTensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert!(source
            .as_ref()
            .is_none_or(|source| source.context().size() <= self.context().size()));
        validate_parent_maps(self.distribution(), maps);
        let mut pairs = Vec::new();
        if let Some(source) = source {
            validate_maps(self.distribution(), source.distribution(), maps);
            let rank = source.context().rank();
            for (child_key, value) in source.local_pairs() {
                if source.distribution().owner(child_key) != rank {
                    continue;
                }
                if let Some(parent_key) = mapped_key(
                    self.distribution(),
                    source.distribution(),
                    maps,
                    child_key,
                ) {
                    pairs.push((parent_key, value));
                }
            }
        }
        self.write_scaled(&pairs, &alpha, &beta);
    }

    /// Scatter canonically owned nonzero dense child values into this sparse
    /// parent; padding, replicas, and numerical zeros do not contribute.
    pub fn scatter_permuted_from_dense(
        &mut self,
        source: Option<&Tensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert!(source
            .as_ref()
            .is_none_or(|source| source.context().size() <= self.context().size()));
        validate_parent_maps(self.distribution(), maps);
        let mut pairs = Vec::new();
        if let Some(source) = source {
            validate_maps(self.distribution(), source.distribution(), maps);
            let rank = source.context().rank();
            let zero = source.algebra().zero();
            for (child_key, value) in source.local_pairs() {
                if source.distribution().owner(child_key) != rank || value == zero {
                    continue;
                }
                if let Some(parent_key) = mapped_key(
                    self.distribution(),
                    source.distribution(),
                    maps,
                    child_key,
                ) {
                    pairs.push((parent_key, value));
                }
            }
        }
        self.write_scaled(&pairs, &alpha, &beta);
    }
}

impl<'c, 'r, A: Ring + Clone> SparseTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Gather sparse parent values over the child's full rectangular domain;
    /// the compressed destination write canonicalizes and signed-accumulates
    /// symmetry-orbit duplicates.
    pub fn gather_permuted_into_symmetric(
        &self,
        destination: Option<&mut SymmetricTensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert!(destination
            .as_ref()
            .is_none_or(|destination| destination.context().size() <= self.context().size()));
        validate_parent_maps(self.distribution(), maps);
        validate_gather_maps(maps);

        let (parent_keys, child_keys) = if let Some(destination) = destination.as_ref() {
            let child = destination.distribution().distribution();
            validate_maps(self.distribution(), child, maps);
            let enumeration = Distribution::cyclic(child.shape.clone(), destination.context().size());
            let rank = destination.context().rank();
            let mut keys = Vec::new();
            for offset in 0..enumeration.local_len() {
                let Some(child_key) = enumeration.global_key(rank, offset) else {
                    continue;
                };
                if enumeration.owner(child_key) != rank {
                    continue;
                }
                if let Some(parent_key) =
                    mapped_key(self.distribution(), &enumeration, maps, child_key)
                {
                    keys.push((parent_key, child_key));
                }
            }
            (
                keys.iter().map(|&(parent, _)| parent).collect(),
                keys.iter().map(|&(_, child)| child).collect(),
            )
        } else {
            (Vec::new(), Vec::new())
        };

        let values = self.read(&parent_keys);
        if let Some(destination) = destination {
            destination.write_scaled(
                &child_keys.into_iter().zip(values).collect::<Vec<_>>(),
                &alpha,
                &beta,
            );
        }
    }

    /// Scatter canonically owned nonzero compressed child values without
    /// expanding their symmetry orbit.
    pub fn scatter_permuted_from_symmetric(
        &mut self,
        source: Option<&SymmetricTensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert!(source
            .as_ref()
            .is_none_or(|source| source.context().size() <= self.context().size()));
        validate_parent_maps(self.distribution(), maps);
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
                if let Some(parent_key) =
                    mapped_key(self.distribution(), child, maps, child_key)
                {
                    pairs.push((parent_key, value));
                }
            }
        }
        self.write_scaled(&pairs, &alpha, &beta);
    }
}

impl<'c, 'r, A: Semiring + Clone> Tensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Scatter every canonically owned stored sparse child pair, including
    /// explicit zeros, into this dense parent.
    pub fn scatter_permuted_from_sparse(
        &mut self,
        source: Option<&SparseTensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert!(source
            .as_ref()
            .is_none_or(|source| source.context().size() <= self.context().size()));
        validate_parent_maps(self.distribution(), maps);
        let mut pairs = Vec::new();
        if let Some(source) = source {
            validate_maps(self.distribution(), source.distribution(), maps);
            let rank = source.context().rank();
            for (child_key, value) in source.local_pairs() {
                if source.distribution().owner(child_key) != rank {
                    continue;
                }
                if let Some(parent_key) = mapped_key(
                    self.distribution(),
                    source.distribution(),
                    maps,
                    child_key,
                ) {
                    pairs.push((parent_key, value));
                }
            }
        }
        self.write_scaled(&pairs, &alpha, &beta);
    }
}

impl<'c, 'r, A: Ring + Clone> SymmetricTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Scatter every canonically owned stored sparse child pair, including
    /// explicit zeros, into this compressed parent.
    pub fn scatter_permuted_from_sparse(
        &mut self,
        source: Option<&SparseTensor<'_, '_, A>>,
        maps: &[Vec<Option<usize>>],
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert!(source
            .as_ref()
            .is_none_or(|source| source.context().size() <= self.context().size()));
        let parent = self.distribution().distribution();
        validate_parent_maps(parent, maps);
        let mut pairs = Vec::new();
        if let Some(source) = source {
            validate_maps(parent, source.distribution(), maps);
            let rank = source.context().rank();
            for (child_key, value) in source.local_pairs() {
                if source.distribution().owner(child_key) != rank {
                    continue;
                }
                if let Some(parent_key) =
                    mapped_key(parent, source.distribution(), maps, child_key)
                {
                    pairs.push((parent_key, value));
                }
            }
        }
        self.write_scaled(&pairs, &alpha, &beta);
    }
}
