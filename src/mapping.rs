// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
// Adapted map-chain and topology-reordering routines from the pinned CTF source.
//! Physical/virtual map chains and cyclic layout, from src/mapping.
use crate::context::Context;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Topology { pub dimensions: Vec<usize> }
impl Topology {
    pub fn new(dimensions: Vec<usize>) -> Self {
        assert!(dimensions.iter().all(|&n| n > 0));
        Self { dimensions }
    }
    pub fn size(&self) -> usize { self.dimensions.iter().product() }
    pub fn coordinates(&self, mut rank: usize) -> Vec<usize> {
        assert!(rank < self.size());
        self.dimensions.iter().map(|&n| { let c = rank % n; rank /= n; c }).collect()
    }
    pub fn rank(&self, coordinates: &[usize]) -> usize {
        assert_eq!(coordinates.len(), self.dimensions.len());
        let mut stride = 1;
        let mut rank = 0;
        for (&c, &n) in coordinates.iter().zip(&self.dimensions) {
            assert!(c < n); rank += c * stride; stride *= n;
        }
        rank
    }
    /// topology.cxx get_topo_reorder_rank: node-major to intra-node grid order.
    pub fn reorder_rank(&self, intra_node: &[usize], rank: usize) -> usize {
        assert_eq!(intra_node.len(), self.dimensions.len());
        assert!(rank < self.size());
        let ppn: usize = intra_node.iter().product();
        let mut ir = rank % ppn;
        let mut nr = rank / ppn;
        let mut coordinates = Vec::new();
        for (&n, &i) in self.dimensions.iter().zip(intra_node) {
            assert!(i > 0 && n % i == 0);
            coordinates.push((nr % (n / i)) * i + ir % i);
            nr /= n / i; ir /= i;
        }
        self.rank(&coordinates)
    }
    pub fn inverse_reorder_rank(&self, intra_node: &[usize], rank: usize) -> usize {
        assert_eq!(intra_node.len(), self.dimensions.len());
        let mut ir = 0; let mut nr = 0; let mut si = 1; let mut sn = 1;
        for ((c, &n), &i) in self.coordinates(rank).iter().zip(&self.dimensions).zip(intra_node) {
            assert!(i > 0 && n % i == 0);
            ir += (c % i) * si; nr += (c / i) * sn;
            si *= i; sn *= n / i;
        }
        ir + si * nr
    }
    /// Split the communicator along a physical grid dimension.
    pub fn fiber<'a>(&self, context: &'a Context<'_>, axis: usize) -> Context<'a> {
        assert_eq!(self.size(), context.size());
        let mut coordinates = self.coordinates(context.rank());
        let key = coordinates[axis]; coordinates[axis] = 0;
        context.split(Some(self.rank(&coordinates).try_into().unwrap()), key.try_into().unwrap()).unwrap()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Mapping {
    Unmapped,
    Physical { axis: usize, processes: usize, child: Box<Mapping> },
    Virtual { copies: usize, child: Box<Mapping> },
}
impl Mapping {
    pub fn phase(&self) -> usize {
        match self {
            Self::Unmapped => 1,
            Self::Physical { processes, child, .. } => processes * child.phase(),
            Self::Virtual { copies, child } => copies * child.phase(),
        }
    }
    pub fn physical_phase(&self) -> usize {
        match self {
            Self::Unmapped => 1,
            Self::Physical { processes, child, .. } => processes * child.physical_phase(),
            Self::Virtual { child, .. } => child.physical_phase(),
        }
    }
    pub fn physical_rank(&self, coordinates: &[usize]) -> usize {
        match self {
            Self::Unmapped => 0,
            Self::Physical { axis, processes, child } => coordinates[*axis] + processes * child.physical_rank(coordinates),
            Self::Virtual { child, .. } => child.physical_rank(coordinates),
        }
    }
    pub fn augment_physical(&mut self, topology: &Topology, axis: usize) {
        let child = Box::new(std::mem::replace(self, Self::Unmapped));
        *self = Self::Physical { axis, processes: topology.dimensions[axis], child };
    }
    pub fn augment_virtual(&mut self, total_phase: usize) {
        assert!(total_phase > 0);
        match self {
            Self::Unmapped => *self = Self::Virtual { copies: total_phase, child: Box::new(Self::Unmapped) },
            Self::Physical { processes, child, .. } => {
                assert_eq!(total_phase % *processes, 0);
                child.augment_virtual(total_phase / *processes);
            }
            Self::Virtual { copies, child } => {
                assert!(matches!(**child, Self::Unmapped));
                *copies = total_phase;
            }
        }
    }
    fn assign_owner(&self, mut remainder: usize, coordinates: &mut [usize]) {
        match self {
            Self::Unmapped => {},
            Self::Physical { axis, processes, child } => {
                coordinates[*axis] = remainder % processes; remainder /= processes;
                child.assign_owner(remainder, coordinates);
            }
            Self::Virtual { child, .. } => child.assign_owner(remainder, coordinates),
        }
    }
    fn validate(&self, topology: &Topology, used: &mut [bool]) {
        match self {
            Self::Unmapped => {},
            Self::Physical { axis, processes, child } => {
                assert_eq!(*processes, topology.dimensions[*axis]);
                assert!(!used[*axis], "physical axis assigned twice");
                used[*axis] = true; child.validate(topology, used);
            }
            Self::Virtual { copies, child } => { assert!(*copies > 0); child.validate(topology, used); }
        }
    }
}

/// Padded local blocks are column-major within each virtual block; virtual
/// blocks themselves are column-major. Physical ownership is cyclic.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Distribution {
    pub shape: Vec<usize>,
    pub topology: Topology,
    pub mappings: Vec<Mapping>,
}
impl Distribution {
    pub fn new(shape: Vec<usize>, topology: Topology, mappings: Vec<Mapping>) -> Self {
        assert_eq!(shape.len(), mappings.len());
        let mut used = vec![false; topology.dimensions.len()];
        for m in &mappings { m.validate(&topology, &mut used); }
        Self { shape, topology, mappings }
    }
    pub fn cyclic(shape: Vec<usize>, processes: usize) -> Self {
        let topology = Topology::new(vec![processes]);
        let mut mappings = vec![Mapping::Unmapped; shape.len()];
        if !shape.is_empty() { mappings[0].augment_physical(&topology, 0); }
        Self::new(shape, topology, mappings)
    }
    pub fn global_len(&self) -> usize { self.shape.iter().product() }
    pub fn block_shape(&self) -> Vec<usize> {
        self.shape.iter().zip(&self.mappings).map(|(&n, m)| n.div_ceil(m.phase())).collect()
    }
    pub fn local_len(&self) -> usize {
        self.block_shape().iter().product::<usize>() * self.mappings.iter().map(|m| m.phase()/m.physical_phase()).product::<usize>()
    }
    pub fn decode_key(&self, mut key: usize) -> Vec<usize> {
        assert!(key < self.global_len());
        self.shape.iter().map(|&n| { let c = key%n; key /= n; c }).collect()
    }
    pub fn encode_key(&self, coordinates: &[usize]) -> usize {
        assert_eq!(coordinates.len(), self.shape.len());
        let mut key = 0; let mut stride = 1;
        for (&c, &n) in coordinates.iter().zip(&self.shape) {
            assert!(c < n); key += c * stride; stride *= n;
        }
        key
    }
    pub fn owner(&self, key: usize) -> usize {
        let coordinates = self.decode_key(key);
        let mut owner = vec![0; self.topology.dimensions.len()];
        for (&c, m) in coordinates.iter().zip(&self.mappings) { m.assign_owner(c, &mut owner); }
        self.topology.rank(&owner)
    }
    /// Physical cyclic neighbor used by slice.cxx's rank-shift step. Unused
    /// topology coordinates retain their replica coordinate.
    pub fn shifted_rank(&self, rank: usize, offsets: &[usize], forward: bool) -> usize {
        assert_eq!(offsets.len(), self.shape.len());
        let mut coordinates = self.topology.coordinates(rank);
        let original = coordinates.clone();
        for (m, &offset) in self.mappings.iter().zip(offsets) {
            let p = m.physical_phase();
            let r = m.physical_rank(&original);
            let shifted = if forward { (r + offset%p)%p } else { (r + p - offset%p)%p };
            m.assign_owner(shifted, &mut coordinates);
        }
        self.topology.rank(&coordinates)
    }
    /// Includes replicas along unused topology dimensions.
    pub fn owns(&self, rank: usize, key: usize) -> bool {
        let rank_coordinates = self.topology.coordinates(rank);
        self.decode_key(key).iter().zip(&self.mappings).all(|(&c, m)| c % m.physical_phase() == m.physical_rank(&rank_coordinates))
    }
    pub fn local_offset(&self, rank: usize, key: usize) -> usize {
        assert!(self.owns(rank, key));
        let coordinates = self.decode_key(key);
        let block = self.block_shape();
        let mut virtual_offset = 0; let mut local_offset = 0;
        let mut sv = 1; let mut sl = 1;
        for ((&c, m), &b) in coordinates.iter().zip(&self.mappings).zip(&block) {
            let vp = m.phase()/m.physical_phase();
            virtual_offset += ((c % m.phase())/m.physical_phase()) * sv;
            local_offset += (c/m.phase()) * sl;
            sv *= vp; sl *= b;
        }
        virtual_offset * sl + local_offset
    }
    /// Returns None for padding, rather than manufacturing an out-of-range key.
    pub fn global_key(&self, rank: usize, offset: usize) -> Option<usize> {
        assert!(offset < self.local_len());
        let block = self.block_shape();
        let block_size: usize = block.iter().product();
        let mut v = offset/block_size; let mut l = offset%block_size;
        let rank_coordinates = self.topology.coordinates(rank);
        let mut coordinates = Vec::new();
        for ((m, &b), &n) in self.mappings.iter().zip(&block).zip(&self.shape) {
            let vp = m.phase()/m.physical_phase();
            let c = (l%b)*m.phase() + (v%vp)*m.physical_phase() + m.physical_rank(&rank_coordinates);
            if c >= n { return None; }
            coordinates.push(c); l /= b; v /= vp;
        }
        Some(self.encode_key(&coordinates))
    }
}
