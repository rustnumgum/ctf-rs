// Canonical tensor contraction routing adapted from cc4s CTF contraction.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::{
    algebra::{Group, Semiring, Wire},
    context::Context,
    mapping::{Distribution, Mapping, Topology},
    symmetric_distribution::SymmetricDistribution,
    symmetry::{Layout, Symmetry},
};

use super::SymmetricTensor;

impl<'c, 'r, A: Group + Semiring + Clone> SymmetricTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Contract the raw canonical domains on an explicit label-to-grid mapping.
    ///
    /// `physical_labels[axis]` names the union label mapped to that topology
    /// axis. Repeating a label chains multiple physical axes onto its mapping.
    /// Labels must be unique within each operand; diagonal projection belongs to
    /// the caller. This operation applies neither orbit expansion nor symmetry
    /// overcount factors.
    pub fn contract_canonical_on(
        &mut self,
        indices_c: &str,
        a: &Self,
        indices_a: &str,
        b: &Self,
        indices_b: &str,
        topology: Topology,
        physical_labels: &str,
        alpha: A::Element,
        beta: A::Element,
        commutative: bool,
    ) -> Result<(), crate::map_tensor::Rejected> {
        assert!(std::ptr::eq(self.context, a.context));
        assert!(std::ptr::eq(self.context, b.context));
        assert_eq!(topology.size(), self.context.size());
        assert!(physical_labels.is_ascii());
        assert_eq!(physical_labels.len(), topology.dimensions.len());

        fn validate(indices: &str, shape: &[usize]) {
            assert!(indices.is_ascii());
            assert_eq!(indices.len(), shape.len());
            for (axis, label) in indices.bytes().enumerate() {
                assert!(
                    !indices.as_bytes()[..axis].contains(&label),
                    "canonical contraction requires unique operand labels"
                );
            }
        }

        validate(indices_a, &a.distribution.distribution().shape);
        validate(indices_b, &b.distribution.distribution().shape);
        validate(indices_c, &self.distribution.distribution().shape);

        let mut aligned_b = indices_b.as_bytes().to_vec();
        let mut aligned_c = indices_c.as_bytes().to_vec();
        let sign = crate::sym_indices::align_triple(
            indices_a.as_bytes(),
            a.distribution.links(),
            &mut aligned_b,
            b.distribution.links(),
            &mut aligned_c,
            self.distribution.links(),
        );
        assert_eq!(
            sign, 1,
            "raw contraction requires upper-layer antisymmetric sign handling"
        );
        let indices_b = std::str::from_utf8(&aligned_b).unwrap();
        let indices_c = std::str::from_utf8(&aligned_c).unwrap();

        let operands = [
            (indices_a, &a.distribution),
            (indices_b, &b.distribution),
            (indices_c, &self.distribution),
        ];
        let mut labels = Vec::new();
        let mut dimensions = Vec::new();
        for &(indices, distribution) in &operands {
            for (axis, label) in indices.bytes().enumerate() {
                let dimension = distribution.distribution().shape[axis];
                if let Some(union_axis) = labels.iter().position(|&old| old == label) {
                    assert_eq!(dimensions[union_axis], dimension);
                } else {
                    labels.push(label);
                    dimensions.push(dimension);
                }
            }
        }

        let mut maps = vec![Mapping::Unmapped; labels.len()];
        for (topology_axis, label) in physical_labels.bytes().enumerate() {
            let union_axis = labels
                .iter()
                .position(|&candidate| candidate == label)
                .expect("each physical topology axis must name a union label");
            maps[union_axis].augment_physical(&topology, topology_axis);
        }

        let mut symmetry_table = vec![false; labels.len() * labels.len()];
        for &(indices, distribution) in &operands {
            for axis in 0..distribution.links().len().saturating_sub(1) {
                if distribution.links()[axis] == Symmetry::NS {
                    continue;
                }
                let left = labels
                    .iter()
                    .position(|&label| label == indices.as_bytes()[axis])
                    .unwrap();
                let right = labels
                    .iter()
                    .position(|&label| label == indices.as_bytes()[axis + 1])
                    .unwrap();
                symmetry_table[left * labels.len() + right] = true;
                symmetry_table[right * labels.len() + left] = true;
            }
        }
        crate::map_tensor::coordinate_symmetry(&mut maps, &symmetry_table)?;

        let mapped: [SymmetricDistribution; 3] = std::array::from_fn(|operand| {
            let (indices, original) = operands[operand];
            let mappings = indices
                .bytes()
                .map(|label| {
                    maps[labels.iter().position(|&candidate| candidate == label).unwrap()].clone()
                })
                .collect();
            SymmetricDistribution::new(
                Distribution::new(
                    original.distribution().shape.clone(),
                    topology.clone(),
                    mappings,
                ),
                original.links().to_vec(),
            )
        });

        // All mapper failures and layout validation precede the first collective.
        let layouts: [Layout; 3] = std::array::from_fn(|operand| {
            Layout::new(
                mapped[operand].distribution().block_shape(),
                mapped[operand].links().to_vec(),
            )
        });
        let phases: [Vec<usize>; 3] = std::array::from_fn(|operand| {
            mapped[operand]
                .distribution()
                .mappings
                .iter()
                .map(|mapping| mapping.phase() / mapping.physical_phase())
                .collect()
        });

        let mut aa = a.redistribute(mapped[0].clone());
        let mut bb = b.redistribute(mapped[1].clone());
        let mut cc = self.redistribute(mapped[2].clone());

        let aligned = [indices_a, indices_b, indices_c];
        let mut communicators: [Vec<Context<'_>>; 3] = std::array::from_fn(|_| Vec::new());
        for (axis, label) in physical_labels.bytes().enumerate() {
            for operand in 0..3 {
                if !aligned[operand].as_bytes().contains(&label) {
                    communicators[operand].push(topology.fiber(self.context, axis));
                }
            }
        }
        let communicator_refs: [Vec<&Context<'_>>; 3] =
            std::array::from_fn(|operand| communicators[operand].iter().collect());

        crate::symmetric_contraction_comm::replicated(
            &self.algebra,
            &communicator_refs[0],
            &communicator_refs[1],
            &communicator_refs[2],
            &layouts[0],
            &phases[0],
            indices_a,
            &mut aa.data,
            &layouts[1],
            &phases[1],
            indices_b,
            &mut bb.data,
            &layouts[2],
            &phases[2],
            indices_c,
            &mut cc.data,
            &alpha,
            &beta,
            commutative,
        );

        drop(communicator_refs);
        for group in communicators {
            for communicator in group {
                communicator.close();
            }
        }

        // Reduce leaves authoritative C values only on roots. Distribution
        // owners choose coordinate zero on every replica dimension, exactly
        // those roots, while local_pairs excludes padding and packed holes.
        let rank = self.context.rank();
        let contributions: Vec<_> = cc
            .distribution
            .local_pairs(rank)
            .into_iter()
            .filter(|&(_, key)| cc.distribution.distribution().owner(key) == rank)
            .map(|(offset, key)| (key, cc.data[offset].clone()))
            .collect();

        for (offset, _) in self.distribution.local_pairs(rank) {
            self.data[offset] = self.algebra.zero();
        }
        self.write_add(&contributions);
        Ok(())
    }
}
