// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Default DGTOG ROR dense redistribution (source `DGTOG_SWITCH == 1`).

use crate::{
    algebra::{Monoid, Wire},
    ffi::mpi::{Comm, Transfer},
    mapping::Distribution,
};

use super::{dgtog_bucket, dgtog_calc_cnt};

fn bytes_transfers(
    peers: &[usize],
    counts: &[usize],
    displacements: &[usize],
    width: usize,
) -> Vec<Transfer> {
    peers
        .iter()
        .zip(counts)
        .zip(displacements)
        .map(|((&peer, &count), &displacement)| Transfer {
            peer,
            displacement: displacement.checked_mul(width).unwrap(),
            count: count.checked_mul(width).unwrap(),
        })
        .collect()
}

/// Old physical-layer roots pack LCM replica buckets, exchange them directly
/// with new physical-layer roots, and new roots unpack. All other new replicas
/// are initialized to the additive identity.
pub(crate) fn reshuffle<A: Monoid>(
    communicator: &Comm,
    algebra: &A,
    old: &Distribution,
    new: &Distribution,
    old_data: &[A::Element],
) -> Vec<A::Element>
where
    A::Element: Wire,
{
    assert_eq!(old.shape, new.shape);
    assert_eq!(old.topology.size(), communicator.size());
    assert_eq!(new.topology.size(), communicator.size());
    assert!(old.shape.len() <= 12);
    assert_eq!(old_data.len(), old.local_len());

    let rank = communicator.rank();
    let send_layout = dgtog_calc_cnt::layout(old, new, rank);
    let receive_layout = dgtog_calc_cnt::layout(new, old, rank);

    let send_buffer = if send_layout.is_root {
        let send_offsets = dgtog_bucket::offsets(
            old,
            rank,
            &send_layout.common_phase,
            &send_layout.residues,
        );
        dgtog_bucket::pack(
            old_data,
            &send_offsets,
            &send_layout.counts,
            &send_layout.displacements,
        )
    } else {
        Vec::new()
    };
    let receive_elements = if receive_layout.is_root {
        receive_layout
            .counts
            .last()
            .zip(receive_layout.displacements.last())
            .map_or(0, |(&count, &displacement)| count + displacement)
    } else {
        0
    };
    let mut receive_buffer = vec![0; receive_elements * A::Element::WIDTH];

    let sends = if send_layout.is_root {
        bytes_transfers(
            &send_layout.peers,
            &send_layout.counts,
            &send_layout.displacements,
            A::Element::WIDTH,
        )
    } else {
        Vec::new()
    };
    let receives = if receive_layout.is_root {
        bytes_transfers(
            &receive_layout.peers,
            &receive_layout.counts,
            &receive_layout.displacements,
            A::Element::WIDTH,
        )
    } else {
        Vec::new()
    };
    communicator.redistribute_ror(&send_buffer, &sends, &mut receive_buffer, &receives);

    let mut new_data = vec![algebra.zero(); new.local_len()];
    if receive_layout.is_root {
        let receive_offsets = dgtog_bucket::offsets(
            new,
            rank,
            &receive_layout.common_phase,
            &receive_layout.residues,
        );
        dgtog_bucket::unpack(
            &receive_buffer,
            &receive_offsets,
            &receive_layout.counts,
            &receive_layout.displacements,
            &mut new_data,
        );
    }
    new_data
}
