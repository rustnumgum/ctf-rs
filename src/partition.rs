// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
// Adapted from src/interface/partition.{h,cxx} in the pinned CTF source.
//! User-specified process-grid partitions and their tensor-index binding.

use crate::mapping::{Distribution, Mapping, Topology};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Partition {
    lens: Vec<usize>,
}

impl Partition {
    pub fn new(lens: impl Into<Vec<usize>>) -> Self {
        let lens = lens.into();
        assert!(lens.iter().all(|&length| length > 0));
        Self { lens }
    }

    pub fn order(&self) -> usize {
        self.lens.len()
    }

    pub fn lens(&self) -> &[usize] {
        &self.lens
    }

    pub fn indexed(&self, indices: &str) -> IdxPartition {
        IdxPartition::new(self.clone(), indices)
    }

    pub fn topology(&self) -> Topology {
        Topology::new(self.lens.clone())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdxPartition {
    partition: Partition,
    indices: String,
}

impl IdxPartition {
    pub fn new(partition: Partition, indices: &str) -> Self {
        assert!(indices.is_ascii());
        assert_eq!(indices.len(), partition.order());
        Self {
            partition,
            indices: indices.to_owned(),
        }
    }

    pub fn partition(&self) -> &Partition {
        &self.partition
    }

    pub fn indices(&self) -> &str {
        &self.indices
    }

    /// Omit unit process-grid dimensions while preserving source index order.
    pub fn reduce_order(&self) -> Self {
        let mut lens = Vec::new();
        let mut indices = Vec::new();
        for (&length, index) in self.partition.lens.iter().zip(self.indices.bytes()) {
            if length != 1 {
                lens.push(length);
                indices.push(index);
            }
        }
        Self::new(
            Partition::new(lens),
            std::str::from_utf8(&indices).unwrap(),
        )
    }

    /// Bind the partition axes to matching tensor labels.
    pub fn distribution(&self, shape: Vec<usize>, tensor_indices: &str) -> Distribution {
        assert!(tensor_indices.is_ascii());
        assert_eq!(tensor_indices.len(), shape.len());
        let topology = self.partition.topology();
        let mut mappings = vec![Mapping::Unmapped; shape.len()];
        for (topology_axis, label) in self.indices.bytes().enumerate() {
            let tensor_axis = tensor_indices
                .bytes()
                .position(|candidate| candidate == label)
                .expect("partition index must name a tensor dimension");
            mappings[tensor_axis].augment_physical(&topology, topology_axis);
        }
        Distribution::new(shape, topology, mappings)
    }
}
