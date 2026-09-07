//! Symmetry-aware repeated-index extraction and reinsertion.
//!
//! Axis deletion and projected symmetry links follow
//! `tensor::extract_diag` in the pinned CTF source.  Each deleted axis is
//! transferred through the symmetry-aware summation path with `run_diag=1`.
// Axis/link deletion adapted from cc4s CTF tensor::extract_diag.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.

use crate::{
    algebra::{Group, Semiring, Wire},
    diagonal::Projection,
    mapping::{Distribution, Mapping},
    scalar_conversion::CastFromF64,
    symmetric_distribution::SymmetricDistribution,
    symmetry::Symmetry,
};

use super::SymmetricTensor;

struct ProjectedDistribution {
    shape: Vec<usize>,
    mappings: Vec<Mapping>,
    links: Vec<Symmetry>,
}

fn first_duplicate(labels: &[u8]) -> Option<(usize, usize)> {
    (0..labels.len()).find_map(|i| {
        (i + 1..labels.len())
            .find(|&j| labels[i] == labels[j])
            .map(|j| (i, j))
    })
}

/// CTF's `ex_idx_map` and `diag_idx_map` for one `(i,j)` deletion.  These are
/// deliberately independent of the user's index map: only this pair is a
/// diagonal in the internal `run_diag=1` summation.
fn diagonal_maps(order: usize, i: usize, j: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(order <= 128, "symmetric index maps require at most 128 axes");
    let output: Vec<_> = (0..order - 1).map(|axis| axis as u8).collect();
    let input = (0..order)
        .map(|axis| {
            if axis < j {
                axis as u8
            } else if axis == j {
                i as u8
            } else {
                (axis - 1) as u8
            }
        })
        .collect();
    (input, output)
}

impl ProjectedDistribution {
    /// Remove one axis using CTF's projected-link rule.
    fn one(distribution: &SymmetricDistribution, j: usize) -> Self {
        let source = distribution.distribution();
        let mut shape = source.shape.clone();
        let mut mappings = source.mappings.clone();
        let mut links = distribution.links().to_vec();

        // Removing j joins its two neighboring links only when both were the
        // same source symmetry kind; otherwise it creates an NS cut.
        let bridge = if links[j - 1] == links[j] {
            links[j]
        } else {
            Symmetry::NS
        };
        shape.remove(j);
        mappings.remove(j);
        links.remove(j);
        links[j - 1] = bridge;

        Self {
            shape,
            mappings,
            links,
        }
    }

    fn finish(self, topology: crate::mapping::Topology) -> SymmetricDistribution {
        SymmetricDistribution::new(
            Distribution::new(self.shape, topology, self.mappings),
            self.links,
        )
    }
}

impl<'c, 'r, A> SymmetricTensor<'c, 'r, A>
where
    A: Group + Semiring + Clone + CastFromF64,
    A::Element: Wire,
{
    /// Extract one axis per repeated label in CTF's first-pair order.  Every
    /// step executes the full symmetry summation with automatic diagonal
    /// preprocessing disabled, exactly as `tensor::extract_diag(..., rw=1)`.
    pub fn extract_diagonal(&self, labels: &str) -> (Self, String) {
        let source = self.distribution.distribution();
        let projection = Projection::new(&source.shape, labels);
        if !projection.repeated() {
            return (
                Self {
                    context: self.context,
                    distribution: self.distribution.clone(),
                    algebra: self.algebra.clone(),
                    data: self.data.clone(),
                },
                projection.labels,
            );
        }

        let mut current = Self {
            context: self.context,
            distribution: self.distribution.clone(),
            algebra: self.algebra.clone(),
            data: self.data.clone(),
        };
        let mut current_labels = labels.as_bytes().to_vec();
        while let Some((i, j)) = first_duplicate(&current_labels) {
            let source = current.distribution.distribution();
            assert_eq!(source.shape[i], source.shape[j]);
            let topology = source.topology.clone();
            let projected = ProjectedDistribution::one(&current.distribution, j);
            let mut result = Self::new(
                self.context,
                projected.finish(topology),
                self.algebra.clone(),
            );
            let (input_map, output_map) = diagonal_maps(current_labels.len(), i, j);
            result.sum_from_run_diag(
                &output_map,
                &current,
                &input_map,
                self.algebra.one(),
                self.algebra.zero(),
            );
            current_labels.remove(j);
            current = result;
        }
        assert_eq!(current.distribution.distribution().shape, projection.shape);
        (current, String::from_utf8(current_labels).unwrap())
    }

    /// Replace the selected diagonal by recursively rebuilding CTF's output
    /// diagonal stack, then applying `extract_diag(..., rw=0)` in reverse.
    pub fn replace_diagonal(&mut self, labels: &str, input: &Self) {
        assert!(std::ptr::eq(self.context, input.context));
        let projection = Projection::new(&self.distribution.distribution().shape, labels);
        assert_eq!(projection.shape, input.distribution.distribution().shape);
        let Some((i, j)) = first_duplicate(labels.as_bytes()) else {
            assert_eq!(self.distribution.links(), input.distribution.links());
            self.sum_from_run_diag(
                labels.as_bytes(),
                input,
                labels.as_bytes(),
                self.algebra.one(),
                self.algebra.zero(),
            );
            return;
        };

        let source = self.distribution.distribution();
        assert_eq!(source.shape[i], source.shape[j]);
        let projected = ProjectedDistribution::one(&self.distribution, j);
        let mut reduced = Self::new(
            self.context,
            projected.finish(source.topology.clone()),
            self.algebra.clone(),
        );
        let (output_map, reduced_map) = diagonal_maps(labels.len(), i, j);
        reduced.sum_from_run_diag(
            &reduced_map,
            self,
            &output_map,
            self.algebra.one(),
            self.algebra.zero(),
        );

        let mut reduced_labels = labels.as_bytes().to_vec();
        reduced_labels.remove(j);
        reduced.replace_diagonal(
            std::str::from_utf8(&reduced_labels).unwrap(),
            input,
        );
        self.sum_from_run_diag(
            &output_map,
            &reduced,
            &reduced_map,
            self.algebra.one(),
            self.algebra.zero(),
        );
    }
}
