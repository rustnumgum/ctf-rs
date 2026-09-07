// Canonical sum_tensors routing adapted from cc4s CTF summation.cxx and sparse_rw.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use super::SymmetricTensor;
use crate::{
    algebra::{Group, Semiring, Wire},
    context::Context,
    diagonal::Projection,
    mapping::Topology,
    symmetry::Layout,
};

impl<'c, 'r, A: Group + Semiring + Clone> SymmetricTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Indexed sum with symmetry processing disabled, as in home_sum_tsr(...,
    /// false). Only the source and destination canonical domains participate.
    /// This is NOT orbit-expanded symmetrization or a symmetry-aware sum_from.
    /// Unique-label tasks select an aligned compressed grid automatically;
    /// diagonal `run_diag` tasks retain direct canonical-key routing. Neither
    /// path gathers a global tensor.
    pub fn sum_canonical_from(
        &mut self,
        indices_b: &str,
        a: &Self,
        indices_a: &str,
        alpha: A::Element,
        beta: A::Element,
    ) {
        assert!(std::ptr::eq(self.context, a.context));
        let repeated = [indices_a, indices_b].iter().any(|indices| {
            indices
                .bytes()
                .enumerate()
                .any(|(axis, label)| indices.as_bytes()[..axis].contains(&label))
        });
        if repeated || (indices_a.is_empty() && indices_b.is_empty()) {
            self.sum_canonical_keyed_from(indices_b, a, indices_a, alpha, beta);
            return;
        }

        let mut planned_indices_b = indices_b.as_bytes().to_vec();
        crate::sym_indices::align_pair(
            indices_a.as_bytes(),
            a.distribution.links(),
            &mut planned_indices_b,
            self.distribution.links(),
        );
        let planned_indices_b = std::str::from_utf8(&planned_indices_b).unwrap();
        let plan = super::contraction::automatic_plan(
            self.context.size(),
            &[
                (indices_a, &a.distribution),
                (planned_indices_b, &self.distribution),
            ],
        )
        .expect("automatic symmetric sum mapping failed");
        self.sum_canonical_from_on(
            indices_b,
            a,
            indices_a,
            plan.topology,
            &plan.physical_labels,
            alpha,
            beta,
            false,
        )
        .expect("selected symmetric sum mapping failed");
    }

    /// Execute a canonical compressed sum on an explicit aligned grid plan.
    /// Input-only fibers broadcast, output-only fibers reduce, and the mapped
    /// result is restored from unique roots to the original output layout.
    #[allow(clippy::too_many_arguments)]
    pub fn sum_canonical_from_on(
        &mut self,
        indices_b: &str,
        a: &Self,
        indices_a: &str,
        topology: Topology,
        physical_labels: &str,
        alpha: A::Element,
        beta: A::Element,
        commutative: bool,
    ) -> Result<(), crate::map_tensor::Rejected> {
        assert!(std::ptr::eq(self.context, a.context));
        assert_eq!(topology.size(), self.context.size());
        assert!(indices_a.is_ascii() && indices_b.is_ascii());
        assert_eq!(indices_a.len(), a.distribution.links().len());
        assert_eq!(indices_b.len(), self.distribution.links().len());
        for indices in [indices_a, indices_b] {
            for (axis, label) in indices.bytes().enumerate() {
                assert!(
                    !indices.as_bytes()[..axis].contains(&label),
                    "mapped canonical sum requires unique operand labels"
                );
            }
        }
        if a.distribution.distribution().shape.contains(&0)
            || self.distribution.distribution().shape.contains(&0)
        {
            let algebra = self.algebra.clone();
            self.transform_indexed(indices_b, |value| *value = algebra.multiply(&beta, value));
            return Ok(());
        }

        let mut aligned_b = indices_b.as_bytes().to_vec();
        let sign = crate::sym_indices::align_pair(
            indices_a.as_bytes(),
            a.distribution.links(),
            &mut aligned_b,
            self.distribution.links(),
        );
        assert_eq!(
            sign, 1,
            "raw sum requires upper-layer antisymmetric sign handling"
        );
        let indices_b = std::str::from_utf8(&aligned_b).unwrap();
        let mapped = super::contraction::mapped_distributions(
            [
                (indices_a, &a.distribution),
                (indices_b, &self.distribution),
            ],
            &topology,
            physical_labels,
        )?;
        let layouts = [
            Layout::new(
                mapped[0].distribution().block_shape(),
                mapped[0].links().to_vec(),
            ),
            Layout::new(
                mapped[1].distribution().block_shape(),
                mapped[1].links().to_vec(),
            ),
        ];
        let phases: [Vec<usize>; 2] = std::array::from_fn(|operand| {
            mapped[operand]
                .distribution()
                .mappings
                .iter()
                .map(|mapping| mapping.phase() / mapping.physical_phase())
                .collect()
        });

        let mut aa = a.redistribute(mapped[0].clone());
        let mut bb = self.redistribute(mapped[1].clone());
        let aligned = [indices_a, indices_b];
        let mut communicators: [Vec<Context<'_>>; 2] = std::array::from_fn(|_| Vec::new());
        for (axis, label) in physical_labels.bytes().enumerate() {
            for operand in 0..2 {
                if !aligned[operand].as_bytes().contains(&label) {
                    communicators[operand].push(topology.fiber(self.context, axis));
                }
            }
        }
        let communicator_refs: [Vec<&Context<'_>>; 2] =
            std::array::from_fn(|operand| communicators[operand].iter().collect());

        crate::symmetric_sum_comm::replicated(
            &self.algebra,
            &communicator_refs[0],
            &communicator_refs[1],
            &layouts[0],
            &phases[0],
            indices_a,
            &mut aa.data,
            &layouts[1],
            &phases[1],
            indices_b,
            &mut bb.data,
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

        let rank = self.context.rank();
        let contributions: Vec<_> = bb
            .distribution
            .local_pairs(rank)
            .into_iter()
            .filter(|&(_, key)| bb.distribution.distribution().owner(key) == rank)
            .map(|(offset, key)| (key, bb.data[offset].clone()))
            .collect();
        for (offset, _) in self.distribution.local_pairs(rank) {
            self.data[offset] = self.algebra.zero();
        }
        self.write_add(&contributions);
        Ok(())
    }

    fn sum_canonical_keyed_from(
        &mut self,
        indices_b: &str,
        a: &Self,
        indices_a: &str,
        alpha: A::Element,
        beta: A::Element,
    ) {
        let mut aligned_output = indices_b.as_bytes().to_vec();
        let sign = crate::sym_indices::align_pair(
            indices_a.as_bytes(),
            a.distribution.links(),
            &mut aligned_output,
            self.distribution.links(),
        );
        assert_eq!(
            sign, 1,
            "raw sum requires upper-layer antisymmetric sign handling"
        );
        let indices_b = std::str::from_utf8(&aligned_output).unwrap();
        let da = a.distribution.distribution();
        let db = self.distribution.distribution();
        let input = Projection::new(&da.shape, indices_a);
        let output = Projection::new(&db.shape, indices_b);
        let mut broadcasts = 1;
        let sources: Vec<_> = output
            .labels
            .bytes()
            .enumerate()
            .map(|(axis, label)| {
                let source = input
                    .labels
                    .bytes()
                    .position(|candidate| candidate == label);
                if let Some(source) = source {
                    assert_eq!(input.shape[source], output.shape[axis]);
                } else {
                    broadcasts *= output.shape[axis];
                }
                source
            })
            .collect();
        let mut contributions = Vec::new();
        for (key, value) in a.local_pairs() {
            if da.owner(key) != self.context.rank() {
                continue;
            }
            let Some(coordinates) = input.project(&da.decode_key(key)) else {
                continue;
            };
            let value = self.algebra.multiply(&value, &alpha);
            for mut broadcast in 0..broadcasts {
                let target: Vec<_> = sources
                    .iter()
                    .enumerate()
                    .map(|(axis, source)| {
                        if let Some(source) = source {
                            coordinates[*source]
                        } else {
                            let coordinate = broadcast % output.shape[axis];
                            broadcast /= output.shape[axis];
                            coordinate
                        }
                    })
                    .collect();
                let key = db.encode_key(&output.expand(&target));
                if self.distribution.canonicalize(key) == Some((key, 1)) {
                    contributions.push((key, value.clone()));
                }
            }
        }
        let algebra = self.algebra.clone();
        self.transform_indexed(indices_b, |value| *value = algebra.multiply(&beta, value));
        self.write_add(&contributions);
    }
}
