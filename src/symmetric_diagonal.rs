//! Symmetry-aware repeated-index extraction and reinsertion.
//!
//! Axis deletion and projected symmetry links follow
//! `tensor::extract_diag` in the pinned CTF source. Values are obtained through
//! indexed reads for the explicitly supported diagonal patterns; broken
//! nontrivial symmetry groups require CTF's full symmetry summation machinery.
// Axis/link deletion adapted from cc4s CTF tensor::extract_diag.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.

use crate::{
    algebra::{Group, Wire},
    diagonal::Projection,
    mapping::{Distribution, Mapping},
    symmetric_distribution::SymmetricDistribution,
    symmetry::Symmetry,
};

use super::SymmetricTensor;

struct ProjectedDistribution {
    shape: Vec<usize>,
    mappings: Vec<Mapping>,
    links: Vec<Symmetry>,
}

fn assert_supported_diagonal(links: &[Symmetry], labels: &str) {
    let labels = labels.as_bytes();
    let mut group = vec![0; links.len()];
    let mut groups = Vec::new();
    let mut start = 0;
    for end in 0..links.len() {
        if links[end] == Symmetry::NS {
            let index = groups.len();
            group[start..=end].fill(index);
            groups.push((start, end + 1, links[start]));
            start = end + 1;
        }
    }

    // Equality inside an AS/SH group makes the complete selected diagonal a
    // structural zero, independently of any other repeated label.
    for i in 0..labels.len() {
        for j in i + 1..labels.len() {
            if labels[i] == labels[j]
                && group[i] == group[j]
                && matches!(groups[group[i]].2, Symmetry::AS | Symmetry::SH)
            {
                return;
            }
        }
    }

    for i in 0..labels.len() {
        if labels[..i].contains(&labels[i]) {
            continue;
        }
        let occurrences: Vec<_> = (i..labels.len())
            .filter(|&axis| labels[axis] == labels[i])
            .collect();
        if occurrences.len() == 1 {
            continue;
        }

        let singleton_ns = occurrences.iter().all(|&axis| {
            let (begin, end, kind) = groups[group[axis]];
            kind == Symmetry::NS && end - begin == 1
        });
        let isolated_sy_pair = occurrences.len() == 2
            && group[occurrences[0]] == group[occurrences[1]]
            && {
                let (begin, end, kind) = groups[group[occurrences[0]]];
                kind == Symmetry::SY
                    && end - begin == 2
                    && occurrences[0] == begin
                    && occurrences[1] == begin + 1
            };
        assert!(
            singleton_ns || isolated_sy_pair,
            "diagonal extraction across a nontrivial symmetry group requires full symmetry summation"
        );
    }
}

impl ProjectedDistribution {
    /// Repeated axes are removed one at a time in CTF's first-pair order.
    fn new(distribution: &SymmetricDistribution, labels: &str) -> Self {
        let source = distribution.distribution();
        let mut shape = source.shape.clone();
        let mut mappings = source.mappings.clone();
        let mut links = distribution.links().to_vec();
        let mut labels = labels.as_bytes().to_vec();

        loop {
            let duplicate = (0..labels.len()).find_map(|i| {
                (i + 1..labels.len())
                    .find(|&j| labels[i] == labels[j])
                    .map(|j| (i, j))
            });
            let Some((i, j)) = duplicate else { break };
            assert_eq!(shape[i], shape[j]);

            // Removing j joins its two neighboring links only when both were
            // the same source symmetry kind; otherwise it creates an NS cut.
            let bridge = if links[j - 1] == links[j] {
                links[j]
            } else {
                Symmetry::NS
            };
            shape.remove(j);
            mappings.remove(j);
            links.remove(j);
            links[j - 1] = bridge;
            labels.remove(j);
        }

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

impl<'c, 'r, A: Group + Clone> SymmetricTensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Extract one axis per repeated label, preserving the symmetry links that
    /// survive CTF's recursive axis-deletion rule.  Only projected canonical
    /// keys are requested; source symmetry normalization supplies their values
    /// without gathering or unpacking either tensor globally.
    ///
    /// Supported repeated labels are pure NS axes, an isolated two-axis SY
    /// group indexed `ii`, and structural-zero AS/SH diagonals.  Other repeated
    /// labels that touch a nontrivial symmetry group require the unrestricted
    /// source symmetry-summation path and are rejected.
    pub fn extract_diagonal(&self, labels: &str) -> (Self, String) {
        let source = self.distribution.distribution();
        let projection = Projection::new(&source.shape, labels);
        assert_supported_diagonal(self.distribution.links(), labels);
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

        let projected = ProjectedDistribution::new(&self.distribution, labels);
        assert_eq!(projected.shape, projection.shape);
        let topology = source.topology.clone();
        let mut result = Self::new(
            self.context,
            projected.finish(topology),
            self.algebra.clone(),
        );

        let destination = result.distribution.distribution();
        let destination_keys: Vec<_> = result
            .distribution
            .local_pairs(self.context.rank())
            .into_iter()
            .map(|(_, key)| key)
            .filter(|&key| destination.owner(key) == self.context.rank())
            .collect();
        let source_keys: Vec<_> = destination_keys
            .iter()
            .map(|&key| {
                let coordinates = destination.decode_key(key);
                source.encode_key(&projection.expand(&coordinates))
            })
            .collect();
        let pairs: Vec<_> = destination_keys
            .into_iter()
            .zip(self.read(&source_keys))
            .collect();
        result.write_add(&pairs);
        (result, projection.labels)
    }

    /// Replace the selected semantic diagonal while preserving every source
    /// canonical entry whose orbit does not intersect that diagonal.
    pub fn replace_diagonal(&mut self, labels: &str, input: &Self) {
        assert!(std::ptr::eq(self.context, input.context));
        let projection = Projection::new(&self.distribution.distribution().shape, labels);
        assert_supported_diagonal(self.distribution.links(), labels);
        let projected = ProjectedDistribution::new(&self.distribution, labels);
        assert_eq!(projection.shape, input.distribution.distribution().shape);
        assert_eq!(projected.links, input.distribution.links());

        let input_distribution = input.distribution.distribution();
        let output_distribution = self.distribution.distribution();
        let mut pairs = Vec::new();
        let mut clear_requests = vec![Vec::new(); self.context.size()];

        for (key, value) in input.local_pairs() {
            if input_distribution.owner(key) != self.context.rank() {
                continue;
            }
            let coordinates = input_distribution.decode_key(key);
            let output_key = output_distribution.encode_key(&projection.expand(&coordinates));
            let Some((canonical, _)) = self.distribution.canonicalize(output_key) else {
                continue;
            };
            for (rank, requests) in clear_requests.iter_mut().enumerate() {
                if output_distribution.owns(rank, canonical) {
                    (canonical as u64).encode(requests);
                }
            }
            pairs.push((output_key, value));
        }

        let zero = self.algebra.zero();
        for requests in self.context.inner.exchange(&clear_requests) {
            for key in requests.chunks_exact(8) {
                let key = u64::decode(key) as usize;
                let offset = self
                    .distribution
                    .local_offset(self.context.rank(), key);
                self.data[offset] = zero.clone();
            }
        }
        self.write_add(&pairs);
    }
}
