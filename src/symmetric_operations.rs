// Adapted from cc4s CTF sym_seq_scl and group-preserving tensor repack.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{
    algebra::{Group, Semiring, Wire},
    symmetry::Symmetry,
    symmetric_distribution::SymmetricDistribution,
};

use super::SymmetricTensor;

impl<'c, 'r, A: Group + Clone> SymmetricTensor<'c, 'r, A>
where A::Element: Wire {
    /// Source tensor repack (home_sum_tsr with symmetry handling disabled).
    /// Copy the intersection of source and target canonical domains, without
    /// orbit expansion or symmetrization. Target mapping is explicit.
    pub fn repack_to(&self, target: SymmetricDistribution) -> Self {
        assert_eq!(self.distribution.distribution().shape, target.distribution().shape);
        let mut added = false;
        let mut removed = false;
        for (&old, &new) in self.distribution.links().iter().zip(target.links()) {
            added |= old == Symmetry::NS && new != Symmetry::NS;
            removed |= old != Symmetry::NS && new == Symmetry::NS;
        }
        assert!(!(added && removed), "repack cannot both add and remove symmetry boundaries");
        let pairs: Vec<_> = self.local_pairs().into_iter().filter(|(key, _)| {
            self.distribution.distribution().owner(*key) == self.context.rank()
                && target.canonicalize(*key) == Some((*key, 1))
        }).collect();
        let mut result = Self::new(self.context, target, self.algebra.clone());
        result.write_add(&pairs);
        result
    }
}

impl<'c, 'r, A: Group> SymmetricTensor<'c, 'r, A> {
    /// Apply an endomorphism to canonical local entries selected by repeated
    /// labels. AS/SH structural zeros and packed holes are never visited.
    pub fn transform_indexed(&mut self, indices: &str, mut function: impl FnMut(&mut A::Element)) {
        crate::scaling::transform_packed(
            &self.distribution,
            self.context.rank(),
            indices,
            &mut self.data,
            |value| function(value),
        );
    }
}

impl<'c, 'r, A: Group + Semiring> SymmetricTensor<'c, 'r, A> {
    /// Right-scale canonical local entries selected by repeated labels, as in
    /// `scaling/sym_seq_scl.cxx`. The multiplication order is value * alpha.
    pub fn scale_indexed(&mut self, indices: &str, alpha: &A::Element) {
        crate::scaling::scale_packed(
            &self.algebra,
            &self.distribution,
            self.context.rank(),
            indices,
            &mut self.data,
            alpha,
        );
    }

    /// Right-scale canonical local entries only, following sym_seq_scl.
    pub fn scale(&mut self, alpha: &A::Element) {
        crate::scaling::scale_all_packed(
            &self.algebra,
            &self.distribution,
            self.context.rank(),
            &mut self.data,
            alpha,
        );
    }
}

impl<'c, 'r, A> SymmetricTensor<'c, 'r, A>
where
    A: Group + Clone,
{
    /// Repack symmetry kinds within the existing NS-delimited groups.
    ///
    /// This is the direct SY/AS/SH relabeling path from
    /// `tensor/untyped_tensor.cxx:223-270`: changing an NS boundary is rejected
    /// because it requires a summation rather than a packed-data copy.
    pub fn repack_groups(&self, new_links: Vec<Symmetry>) -> Self {
        let old_links = self.distribution.links();
        assert_eq!(old_links.len(), new_links.len());
        for (&old, &new) in old_links.iter().zip(&new_links) {
            assert_eq!(old == Symmetry::NS, new == Symmetry::NS);
        }

        let distribution = SymmetricDistribution::new(
            self.distribution.distribution().clone(),
            new_links,
        );
        let mut result = Self::new(self.context, distribution, self.algebra.clone());
        assert_eq!(result.data.len(), self.data.len());

        // All three symmetry kinds use the same SY-sized packed allocation.
        // Start with zero storage and copy only target-valid slots, which
        // clears both padding and structural holes in the target layout.
        for (offset, _) in result.distribution.local_pairs(self.context.rank()) {
            result.data[offset] = self.data[offset].clone();
        }
        result
    }
}
