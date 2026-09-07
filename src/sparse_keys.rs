// Adapted from cc4s CTF contraction/spctr_tsr.cxx::spctr_pin_keys and
// tensor/algstrct.cxx::{ConstPairIterator::pin,depin} at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Raw sparse-key pinning around virtual-block contraction. Pair vectors are in
//! virtual-block order with dimension zero varying fastest and are sorted by
//! key on entry. Transformations preserve their order and stored zero values.
//! The source contraction wrapper leaves its operand-C pin destination null;
//! this module ports only the defined key transforms and exposes output depin
//! directly rather than reproducing that null-pointer orchestration.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyMetadata {
    pub shape: Vec<usize>,
    /// Full physical-plus-virtual phase (`divisor` in the source).
    pub phases: Vec<usize>,
    pub virtual_dimensions: Vec<usize>,
    pub physical_ranks: Vec<usize>,
}

impl KeyMetadata {
    fn validate(&self, block_count: usize) {
        let order = self.shape.len();
        assert_eq!(self.phases.len(), order);
        assert_eq!(self.virtual_dimensions.len(), order);
        assert_eq!(self.physical_ranks.len(), order);
        for dimension in 0..order {
            assert!(self.phases[dimension] > 0);
            assert!(self.virtual_dimensions[dimension] > 0);
            assert_eq!(
                self.phases[dimension] % self.virtual_dimensions[dimension],
                0
            );
            assert!(
                self.physical_ranks[dimension]
                    < self.phases[dimension] / self.virtual_dimensions[dimension]
            );
        }
        assert_eq!(
            block_count,
            self.virtual_dimensions.iter().product::<usize>()
        );
    }

    fn divided_lengths(&self) -> Vec<usize> {
        self.shape
            .iter()
            .zip(&self.phases)
            .map(|(&length, &phase)| length / phase + usize::from(length % phase != 0))
            .collect()
    }

    fn virtual_offsets(&self, mut block: usize) -> Vec<usize> {
        self.virtual_dimensions
            .iter()
            .zip(&self.phases)
            .zip(&self.physical_ranks)
            .map(|((&virtual_dimension, &phase), &physical_rank)| {
                let offset = (block % virtual_dimension) * (phase / virtual_dimension)
                    + physical_rank;
                block /= virtual_dimension;
                offset
            })
            .collect()
    }
}

/// Replace full-shape keys by keys in the phase-divided local block. As in the
/// source, the physical/virtual residue is not encoded into the pinned key: its
/// virtual-block position already carries that information.
pub fn pin_blocks<E: Clone>(
    metadata: &KeyMetadata,
    blocks: &[Vec<(usize, E)>],
) -> Vec<Vec<(usize, E)>> {
    metadata.validate(blocks.len());
    let divided_lengths = metadata.divided_lengths();
    blocks
        .iter()
        .map(|block| {
            block
                .iter()
                .map(|(key, value)| {
                    let mut key = *key;
                    let mut pinned = 0usize;
                    let mut stride = 1usize;
                    for dimension in 0..metadata.shape.len() {
                        pinned += ((key % metadata.shape[dimension])
                            / metadata.phases[dimension])
                            * stride;
                        stride *= divided_lengths[dimension];
                        key /= metadata.shape[dimension];
                    }
                    (pinned, value.clone())
                })
                .collect()
        })
        .collect()
}

/// Restore pinned output keys to the full tensor key space. This is the
/// contraction source's `check_padding=true` path: if any virtual offset can
/// address a padded part of a partial final phase, individually reconstructed
/// pairs outside the logical shape are omitted and per-block counts shrink.
pub fn depin_output<E: Clone>(
    metadata: &KeyMetadata,
    blocks: &[Vec<(usize, E)>],
) -> Vec<Vec<(usize, E)>> {
    metadata.validate(blocks.len());
    let divided_lengths = metadata.divided_lengths();
    let check_padding = (0..blocks.len()).any(|block| {
        metadata
            .virtual_offsets(block)
            .into_iter()
            .enumerate()
            .any(|(dimension, offset)| {
                metadata.shape[dimension] % metadata.phases[dimension] != 0
                    && offset >= metadata.shape[dimension] % metadata.phases[dimension]
            })
    });

    blocks
        .iter()
        .enumerate()
        .map(|(block_index, block)| {
            let offsets = metadata.virtual_offsets(block_index);
            block
                .iter()
                .filter_map(|(key, value)| {
                    let mut key = *key;
                    let mut global = 0usize;
                    let mut stride = 1usize;
                    let mut outside = false;
                    for dimension in 0..metadata.shape.len() {
                        let coordinate = (key % divided_lengths[dimension])
                            * metadata.phases[dimension]
                            + offsets[dimension];
                        if check_padding && coordinate >= metadata.shape[dimension] {
                            outside = true;
                        }
                        global += coordinate * stride;
                        stride *= metadata.shape[dimension];
                        key /= divided_lengths[dimension];
                    }
                    (!outside).then(|| (global, value.clone()))
                })
                .collect()
        })
        .collect()
}
