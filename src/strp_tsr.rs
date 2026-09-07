// Adapted from cc4s CTF scaling/strp_tsr.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Dense virtual-grid strip and restore used for physical/virtual diagonals.

use crate::mapping::{Mapping, Topology};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StripPlan {
    edge_lengths: Vec<usize>,
    strip_dimensions: Vec<usize>,
    strip_indices: Vec<usize>,
    block_size: usize,
}

impl StripPlan {
    pub fn new(
        edge_lengths: Vec<usize>,
        strip_dimensions: Vec<usize>,
        strip_indices: Vec<usize>,
        block_size: usize,
    ) -> Self {
        assert!(!edge_lengths.is_empty());
        assert_eq!(edge_lengths.len(), strip_dimensions.len());
        assert_eq!(edge_lengths.len(), strip_indices.len());
        for ((&edge, &dimension), &index) in edge_lengths
            .iter()
            .zip(&strip_dimensions)
            .zip(&strip_indices)
        {
            assert!(dimension > 0 && edge % dimension == 0);
            assert!(index < dimension);
        }
        Self {
            edge_lengths,
            strip_dimensions,
            strip_indices,
            block_size,
        }
    }

    pub fn edge_lengths(&self) -> &[usize] { &self.edge_lengths }
    pub fn strip_dimensions(&self) -> &[usize] { &self.strip_dimensions }
    pub fn strip_indices(&self) -> &[usize] { &self.strip_indices }

    pub fn stripped_len(&self) -> usize {
        self.block_size
            * self.edge_lengths
                .iter()
                .zip(&self.strip_dimensions)
                .map(|(&edge, &dimension)| edge / dimension)
                .product::<usize>()
    }

    fn source_offsets(&self) -> Vec<usize> {
        let mut strides = Vec::with_capacity(self.edge_lengths.len());
        let mut stride = 1usize;
        let mut coordinates = Vec::with_capacity(self.edge_lengths.len());
        let mut offset = 0usize;
        for ((&edge, &dimension), &index) in self.edge_lengths
            .iter()
            .zip(&self.strip_dimensions)
            .zip(&self.strip_indices)
        {
            strides.push(stride);
            let coordinate = index * (edge / dimension);
            coordinates.push(coordinate);
            offset += coordinate * stride;
            stride *= edge;
        }

        let runs: usize = self.edge_lengths[1..]
            .iter()
            .zip(&self.strip_dimensions[1..])
            .map(|(&edge, &dimension)| edge / dimension)
            .product();
        let mut offsets = Vec::with_capacity(runs);
        loop {
            offsets.push(offset);
            let mut axis = 1;
            while axis < self.edge_lengths.len() {
                let first = self.strip_indices[axis]
                    * (self.edge_lengths[axis] / self.strip_dimensions[axis]);
                let end = (self.strip_indices[axis] + 1)
                    * (self.edge_lengths[axis] / self.strip_dimensions[axis]);
                offset -= coordinates[axis] * strides[axis];
                coordinates[axis] += 1;
                if coordinates[axis] == end {
                    coordinates[axis] = first;
                }
                offset += coordinates[axis] * strides[axis];
                if coordinates[axis] != first {
                    break;
                }
                axis += 1;
            }
            if axis == self.edge_lengths.len() {
                break;
            }
        }
        offsets
    }

    /// Source direction zero: copy the selected hyper-rectangle to a compact buffer.
    pub fn strip<T: Clone>(&self, data: &[T]) -> Vec<T> {
        assert_eq!(
            data.len(),
            self.block_size * self.edge_lengths.iter().product::<usize>()
        );
        let run = (self.edge_lengths[0] / self.strip_dimensions[0]) * self.block_size;
        let mut buffer = Vec::with_capacity(self.stripped_len());
        for offset in self.source_offsets() {
            let start = offset * self.block_size;
            buffer.extend_from_slice(&data[start..start + run]);
        }
        buffer
    }

    /// Source direction one: restore only the selected region.
    pub fn restore<T: Clone>(&self, buffer: &[T], data: &mut [T]) {
        assert_eq!(buffer.len(), self.stripped_len());
        assert_eq!(
            data.len(),
            self.block_size * self.edge_lengths.iter().product::<usize>()
        );
        let run = (self.edge_lengths[0] / self.strip_dimensions[0]) * self.block_size;
        for (source, offset) in buffer.chunks_exact(run).zip(self.source_offsets()) {
            let start = offset * self.block_size;
            data[start..start + run].clone_from_slice(source);
        }
    }
}

fn head_physical(mapping: &Mapping) -> Option<usize> {
    match mapping {
        Mapping::Physical { processes, .. } => Some(*processes),
        _ => None,
    }
}

/// Build source `strip_diag` metadata. `block_edges` and `local_size` are
/// reduced in place exactly as the C++ builder reduces `blk_edge_len/blk_sz`.
pub fn strip_diagonal(
    indices: &[usize],
    virtual_block_size: usize,
    mappings: &[Mapping],
    topology: &Topology,
    rank: usize,
    block_edges: &mut [usize],
    local_size: &mut usize,
) -> Option<StripPlan> {
    assert_eq!(indices.len(), mappings.len());
    assert_eq!(indices.len(), block_edges.len());
    let label_count = indices.iter().copied().max().map_or(0, |index| index + 1);
    let mut physical_axes = vec![None; label_count];
    for (axis, (&index, mapping)) in indices.iter().zip(mappings).enumerate() {
        if head_physical(mapping).is_some() {
            assert!(physical_axes[index].is_none());
            physical_axes[index] = Some(axis);
        }
    }
    let needs_strip = indices.iter().zip(mappings).any(|(&index, mapping)| {
        matches!(mapping, Mapping::Virtual { .. }) && physical_axes[index].is_some()
    });
    if !needs_strip {
        return None;
    }

    let rank_coordinates = topology.coordinates(rank);
    let edge_lengths: Vec<_> = mappings
        .iter()
        .map(|mapping| mapping.phase() / mapping.physical_phase())
        .collect();
    let mut strip_dimensions = vec![1usize; mappings.len()];
    let mut strip_indices = vec![0usize; mappings.len()];
    for (axis, (&index, mapping)) in indices.iter().zip(mappings).enumerate() {
        if let Mapping::Virtual { copies, .. } = mapping {
            if let Some(physical_axis) = physical_axes[index] {
                let physical_mapping = &mappings[physical_axis];
                assert_eq!(Some(*copies), head_physical(physical_mapping));
                strip_dimensions[axis] = edge_lengths[axis];
                strip_indices[axis] = physical_mapping.physical_rank(&rank_coordinates);
            }
        }
        block_edges[axis] /= strip_dimensions[axis];
        *local_size /= strip_dimensions[axis];
    }
    Some(StripPlan::new(
        edge_lengths,
        strip_dimensions,
        strip_indices,
        virtual_block_size,
    ))
}
