// Adapted from cc4s mapping/topology.cxx and contraction/contraction.cxx
// node-aware remapping selection (lines 4685-4730).
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Pure node-aware rank permutation and dense communication-volume selection.
//! Communicator morphing and tensor data exchange are separate execution work.

use crate::{cost::Models, mapping::Distribution};

/// Original (not reordered) topology.cxx:138-154 average remote-node counts.
/// Node-major rank placement and the explicit ranks-per-node value follow the
/// source topology constructor. Fractional averages must not be rounded.
pub fn original_peer_counts(lens: &[usize], ranks_per_node: usize) -> Vec<f64> {
    assert!(ranks_per_node > 0 && lens.iter().all(|&length| length > 0));
    let ranks: usize = lens.iter().product();
    let mut stride = 1;
    lens.iter().map(|&length| {
        let groups = ranks / (stride * length);
        let peers = if stride >= ranks_per_node {
            (length - 1) as f64
        } else {
            let mut count = 0.;
            for node in 0..ranks / ranks_per_node {
                let offset = (node * ranks_per_node) % (stride * length);
                let distance = offset.min(stride * length - offset);
                count += stride.min(distance) as f64 / (groups * stride) as f64;
            }
            count
        };
        stride *= length;
        peers
    }).collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub intra_node_lens: Vec<usize>,
    pub inter_node_lens: Vec<usize>,
    pub original_volume: f64,
    pub selected_volume: f64,
}

/// topology.cxx::get_topo_reorder_rank. `rank` is node-major and the returned
/// rank uses the topology's column-major dimension order.
pub fn reorder_rank(lens: &[usize], intra_node_lens: &[usize], rank: usize) -> usize {
    assert_eq!(lens.len(), intra_node_lens.len());
    assert!(lens
        .iter()
        .zip(intra_node_lens)
        .all(|(&length, &intra)| intra > 0 && length % intra == 0));
    let size: usize = lens.iter().product();
    assert!(rank < size);
    let num_intra_node: usize = intra_node_lens.iter().product();
    let mut intra_node_rank = rank % num_intra_node;
    let mut node_rank = rank / num_intra_node;
    let mut lda = 1;
    let mut new_rank = 0;
    for axis in 0..lens.len() {
        let inter = lens[axis] / intra_node_lens[axis];
        let axis_node_rank = node_rank % inter;
        node_rank /= inter;
        let axis_intra_node_rank = intra_node_rank % intra_node_lens[axis];
        intra_node_rank /= intra_node_lens[axis];
        new_rank +=
            (axis_node_rank * intra_node_lens[axis] + axis_intra_node_rank) * lda;
        lda *= lens[axis];
    }
    new_rank
}

/// topology.cxx::get_inv_topo_reorder_rank. `rank` uses topology dimension
/// order and the returned rank is node-major.
pub fn inverse_rank(lens: &[usize], intra_node_lens: &[usize], rank: usize) -> usize {
    assert_eq!(lens.len(), intra_node_lens.len());
    assert!(lens
        .iter()
        .zip(intra_node_lens)
        .all(|(&length, &intra)| intra > 0 && length % intra == 0));
    let size: usize = lens.iter().product();
    assert!(rank < size);
    let mut remaining = rank;
    let mut intra_node_rank = 0;
    let mut node_rank = 0;
    let mut lda_node_rank = 1;
    let mut lda_intra_node_rank = 1;
    for axis in 0..lens.len() {
        intra_node_rank +=
            (remaining % intra_node_lens[axis]) * lda_intra_node_rank;
        node_rank += ((remaining % lens[axis]) / intra_node_lens[axis]) * lda_node_rank;
        remaining /= lens[axis];
        lda_node_rank *= lens[axis] / intra_node_lens[axis];
        lda_intra_node_rank *= intra_node_lens[axis];
    }
    intra_node_rank + lda_intra_node_rank * node_rank
}

/// Select the first strictly best node-aware grid, then return it only when it
/// strictly improves on the original topology volume. The chosen candidate's
/// `comm_nodes` value on axis i is `inter_node_lens[i]-1`, matching the reordered
/// topology constructor rather than treating the number of nodes as peers.
#[allow(clippy::too_many_arguments)]
pub fn select_dense(
    distributions: [&Distribution; 3],
    indices: [&str; 3],
    original_nodes: &[f64],
    ranks_per_node: usize,
    element_bytes: usize,
    custom_reduce: bool,
    models: &Models,
) -> Option<Choice> {
    let topology = &distributions[0].topology;
    assert!(distributions
        .iter()
        .all(|distribution| &distribution.topology == topology));
    assert_eq!(original_nodes.len(), topology.dimensions.len());
    assert!(ranks_per_node > 0);
    if ranks_per_node == 1 {
        return None;
    }
    let ranks = topology.size();
    let nodes = (ranks / ranks_per_node).max(1);
    let original_volume = crate::mapped_cost::dense_unfolded(
        distributions,
        indices,
        element_bytes,
        original_nodes,
        false,
        custom_reduce,
    )
    .estimate(models, 1)
    .internode_volume;

    let mut selected = None;
    let mut selected_volume = f64::MAX;
    for inter_node_lens in
        crate::node_aware::inter_node_grids(&topology.dimensions, nodes)
    {
        let intra_node_lens: Vec<_> = topology
            .dimensions
            .iter()
            .zip(&inter_node_lens)
            .map(|(&length, &inter)| length / inter)
            .collect();
        let communication_nodes: Vec<_> =
            inter_node_lens.iter().map(|&count| (count - 1) as f64).collect();
        let volume = crate::mapped_cost::dense_unfolded(
            distributions,
            indices,
            element_bytes,
            &communication_nodes,
            false,
            custom_reduce,
        )
        .estimate(models, 1)
        .internode_volume;
        if volume < selected_volume {
            selected_volume = volume;
            selected = Some((intra_node_lens, inter_node_lens));
        }
    }
    if selected_volume < original_volume {
        let (intra_node_lens, inter_node_lens) = selected.unwrap();
        Some(Choice {
            intra_node_lens,
            inter_node_lens,
            original_volume,
            selected_volume,
        })
    } else {
        None
    }
}
