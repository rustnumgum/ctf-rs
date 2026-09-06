//! Distributed compressed symmetry layout over a rectangular distribution.
// Layout rules adapted from cc4s CTF set_padding, assign_keys and depad_tsr.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.

use crate::{
    mapping::Distribution,
    symmetry::{Layout, Symmetry},
};

/// A distributed tensor layout whose local physical blocks use SY packing for
/// every compressed symmetry group. AS and SH affect global validity, but not
/// the allocated local block size.
#[derive(Clone, Debug)]
pub struct SymmetricDistribution {
    distribution: Distribution,
    links: Vec<Symmetry>,
    block_layout: Layout,
}

impl SymmetricDistribution {
    pub fn new(distribution: Distribution, links: Vec<Symmetry>) -> Self {
        // Validate the global symmetry groups and their equal axis lengths.
        let _ = Layout::new(distribution.shape.clone(), links.clone());
        for i in 0..links.len().saturating_sub(1) {
            if links[i] != Symmetry::NS {
                assert_eq!(
                    distribution.mappings[i].phase(),
                    distribution.mappings[i + 1].phase(),
                    "symmetry-group axes must have equal total phases",
                );
            }
        }

        let packed_links = links
            .iter()
            .map(|&link| match link {
                Symmetry::AS | Symmetry::SH => Symmetry::SY,
                other => other,
            })
            .collect();
        let block_layout = Layout::new(distribution.block_shape(), packed_links);

        Self {
            distribution,
            links,
            block_layout,
        }
    }

    pub fn distribution(&self) -> &Distribution {
        &self.distribution
    }

    pub fn links(&self) -> &[Symmetry] {
        &self.links
    }

    pub fn local_len(&self) -> usize {
        self.virtual_blocks() * self.block_layout.len()
    }

    /// Normalize a rectangular global key into symmetry-canonical order.
    /// Repeated AS/SH coordinates are structural zeros.
    pub fn canonicalize(&self, key: usize) -> Option<(usize, i32)> {
        let mut coordinates = self.distribution.decode_key(key);
        let mut sign = 1;
        let mut start = 0;

        while start < self.links.len() {
            let mut end = start;
            while self.links[end] != Symmetry::NS {
                end += 1;
            }
            let kind = self.links[start];

            for i in start + 1..=end {
                let mut j = i;
                while j > start && coordinates[j] < coordinates[j - 1] {
                    coordinates.swap(j, j - 1);
                    if kind == Symmetry::AS {
                        sign = -sign;
                    }
                    j -= 1;
                }
            }
            if matches!(kind, Symmetry::AS | Symmetry::SH)
                && coordinates[start..=end]
                    .windows(2)
                    .any(|pair| pair[0] == pair[1])
            {
                return None;
            }

            start = end + 1;
        }

        Some((self.distribution.encode_key(&coordinates), sign))
    }

    /// Return the allocated local offset for an owned canonical global key.
    pub fn local_offset(&self, rank: usize, key: usize) -> usize {
        assert_eq!(self.canonicalize(key), Some((key, 1)));
        assert!(self.distribution.owns(rank, key));

        let coordinates = self.distribution.decode_key(key);
        let mut virtual_block = 0;
        let mut virtual_stride = 1;
        let mut quotients = Vec::with_capacity(coordinates.len());

        for (&coordinate, mapping) in coordinates.iter().zip(&self.distribution.mappings) {
            let physical_phase = mapping.physical_phase();
            let total_phase = mapping.phase();
            let virtual_phase = total_phase / physical_phase;
            virtual_block += ((coordinate % total_phase) / physical_phase) * virtual_stride;
            virtual_stride *= virtual_phase;
            quotients.push(coordinate / total_phase);
        }

        let packed_rank = self.block_layout.locate(&quotients).unwrap().0;
        virtual_block * self.block_layout.len() + packed_rank
    }

    /// Enumerate semantic local entries as `(allocated offset, canonical key)`,
    /// filtering padding, noncanonical packed holes, and AS/SH structural zeros.
    pub fn local_pairs(&self, rank: usize) -> Vec<(usize, usize)> {
        assert!(rank < self.distribution.topology.size());
        let packed_block_size = self.block_layout.len();
        let rectangular_block_size: usize = self.block_layout.shape().iter().product();
        let mut pairs = Vec::new();

        for virtual_block in 0..self.virtual_blocks() {
            for (packed_rank, quotients) in self.block_layout.coordinates().enumerate() {
                let rectangular_rank = column_major_rank(&quotients, self.block_layout.shape());
                let rectangular_offset =
                    virtual_block * rectangular_block_size + rectangular_rank;
                if let Some(raw_key) = self.distribution.global_key(rank, rectangular_offset) {
                    if matches!(self.canonicalize(raw_key), Some((key, _)) if key == raw_key) {
                        pairs.push((virtual_block * packed_block_size + packed_rank, raw_key));
                    }
                }
            }
        }

        pairs
    }

    fn virtual_blocks(&self) -> usize {
        self.distribution
            .mappings
            .iter()
            .map(|mapping| mapping.phase() / mapping.physical_phase())
            .product()
    }
}

fn column_major_rank(coordinates: &[usize], shape: &[usize]) -> usize {
    let mut rank = 0;
    let mut stride = 1;
    for (&coordinate, &extent) in coordinates.iter().zip(shape) {
        rank += coordinate * stride;
        stride *= extent;
    }
    rank
}
