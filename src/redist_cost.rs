// Adapted from cc4s tensor/untyped_tensor.cxx and redistribution/{redist,
// dgtog_redist}.cxx. Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Dense redistribution estimates using the pinned source model branches.

use crate::{cost::Models, mapping::Distribution};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub seconds: f64,
    pub temporary_bytes: usize,
}

/// Source's no-redistribution preflight: the topology and every dimension map
/// are unchanged. Structural topology equality replaces CTF's pointer identity.
pub fn same_mapping(old: &Distribution, new: &Distribution) -> bool {
    assert_eq!(old.shape, new.shape);
    old.topology == new.topology && old.mappings == new.mappings
}

/// Dense `can_block_reshuffle`: only the total phase of each tensor dimension
/// participates. In particular, a changed physical-axis assignment with the
/// same phases remains eligible; the source adds no separate transpose cost.
pub fn can_block_reshuffle(old: &Distribution, new: &Distribution) -> bool {
    assert_eq!(old.shape, new.shape);
    old.mappings
        .iter()
        .zip(&new.mappings)
        .all(|(old, new)| old.phase() == new.phase())
}

fn virtual_blocks(distribution: &Distribution) -> usize {
    distribution
        .mappings
        .iter()
        .map(|mapping| mapping.phase() / mapping.physical_phase())
        .product()
}

/// Estimate a dense redistribution between layouts of the same tensor.
/// `temporary_bytes` is source `get_redist_mem`; resident tensor storage is not
/// included. Sparse redistribution and offset/permutation reshuffles are outside
/// this API.
pub fn dense(
    old: &Distribution,
    new: &Distribution,
    element_bytes: usize,
    models: &Models,
) -> Estimate {
    assert_eq!(old.shape, new.shape);
    assert_eq!(old.topology.size(), new.topology.size());
    if same_mapping(old, new) {
        return Estimate {
            seconds: 0.,
            temporary_bytes: 0,
        };
    }

    let bytes = old.local_len().max(new.local_len()) * element_bytes;
    if can_block_reshuffle(old, new) {
        Estimate {
            seconds: models.get("blres_mdl").estimate(&[
                (virtual_blocks(old) + virtual_blocks(new)) as f64,
                bytes as f64,
            ]),
            temporary_bytes: bytes,
        }
    } else {
        let log_processes = (new.topology.size() as f64).log2();
        Estimate {
            seconds: models.get("dgtog_res_mdl").estimate(&[
                1.,
                log_processes,
                bytes as f64 * log_processes,
            ]),
            temporary_bytes: bytes + bytes / 2,
        }
    }
}
