// Adapted from cc4s CTF contraction/{spctr_2d_general,spctr_comm,spctr_tsr}.cxx
// at f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Metadata-only cost formulas for sparse contraction communication layers.
//! Sizes are bytes only after applying the source formula; sparse 2D panel
//! `outer * inner` counts virtual blocks, while dense panels count elements.

use crate::cost::{Communication, Models};

#[path = "sparse_cost_local.rs"]
pub mod local;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fractions {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Storage {
    pub sparse: bool,
    pub element_size: usize,
    pub pair_size: usize,
    /// Dense entries represented by one sparse virtual block.
    pub dense_virtual_size: usize,
    /// Whether a dense output reduction uses a custom MPI addition operation.
    pub custom_addition: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operand2d {
    pub storage: Storage,
    pub moving: bool,
    pub ranks: usize,
    pub outer: usize,
    pub inner: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoDimensional {
    pub edge: usize,
    pub a: Operand2d,
    pub b: Operand2d,
    pub c: Operand2d,
}

fn sparse_bytes(storage: Storage, blocks: usize, fraction: f64) -> usize {
    ((storage.pair_size * blocks) as f64
        * fraction
        * storage.dense_virtual_size as f64) as usize
}

fn dense_bytes(storage: Storage, elements: usize, fraction: f64) -> usize {
    ((storage.element_size * elements) as f64 * fraction) as usize
}

fn csr_reduction(models: &Models, ranks: usize, bytes: usize) -> f64 {
    assert!(ranks > 0);
    models
        .get("csrred_mdl")
        .estimate(&[1., (ranks as f64).log2(), bytes as f64])
}

fn add_truncated(total: &mut i64, term: f64) {
    *total = (*total as f64 + term) as i64;
}

impl TwoDimensional {
    /// Communication performed directly by this 2D layer. Every message-size
    /// expression is converted to an integer before entering its model, as in
    /// the source's `int64_t` model arguments.
    pub fn fixed_time(&self, models: &Models, fractions: Fractions, layers: usize) -> f64 {
        assert!(self.edge > 0 && layers > 0);
        let mut time = 0.;
        for (operand, fraction) in [(self.a, fractions.a), (self.b, fractions.b)] {
            if operand.moving {
                let blocks = operand.outer * operand.inner;
                let bytes = if operand.storage.sparse {
                    sparse_bytes(operand.storage, blocks, fraction)
                } else {
                    dense_bytes(operand.storage, blocks, fraction)
                };
                time += models.communication(Communication::Broadcast, operand.ranks, bytes);
            }
        }
        if self.c.moving {
            let blocks = self.c.outer * self.c.inner;
            if self.c.storage.sparse {
                let bytes = sparse_bytes(self.c.storage, blocks, fractions.c);
                time += csr_reduction(models, self.c.ranks, bytes);
            } else {
                let bytes = dense_bytes(self.c.storage, blocks, fractions.c);
                time += models.communication(
                    Communication::Reduce {
                        custom: self.c.storage.custom_addition,
                    },
                    self.c.ranks,
                    bytes,
                );
            }
        }
        time * self.edge as f64 / layers.min(self.edge) as f64
    }

    /// Recursive estimate. `child_time` is the child's estimate at one layer,
    /// matching the source's explicit `est_time_rec(1, ...)` call.
    pub fn est_time(
        &self,
        models: &Models,
        fractions: Fractions,
        layers: usize,
        child_time: f64,
    ) -> f64 {
        assert!(self.edge > 0 && layers > 0);
        child_time * self.edge as f64 / layers.min(self.edge) as f64
            + self.fixed_time(models, fractions, layers)
    }

    /// Storage owned directly by this level. Compound `int64_t += double`
    /// conversions are retained by truncating each sparse term separately.
    pub fn footprint(&self, fractions: Fractions) -> i64 {
        let mut memory = 0i64;
        for (operand, fraction) in [(self.a, fractions.a), (self.b, fractions.b)] {
            let size = operand.outer * operand.inner;
            if operand.storage.sparse {
                if operand.moving || (operand.inner != 0 && operand.outer != 1) {
                    add_truncated(
                        &mut memory,
                        (operand.storage.pair_size * size) as f64
                            * fraction
                            * operand.storage.dense_virtual_size as f64,
                    );
                }
            } else {
                memory += (operand.storage.element_size * size) as i64;
            }
        }

        let size = self.c.outer * self.c.inner;
        if self.c.storage.sparse {
            if self.c.moving || (self.c.inner != 0 && self.c.outer != 1) {
                add_truncated(
                    &mut memory,
                    3.0 * self.c.storage.pair_size as f64
                        * size as f64
                        * fractions.c
                        * self.c.storage.dense_virtual_size as f64,
                );
            }
        } else {
            memory += (self.c.storage.element_size * size) as i64;
        }
        memory
    }

    pub fn temp(&self, fractions: Fractions) -> i64 {
        let mut memory = 0i64;
        if self.c.moving && self.c.storage.sparse {
            add_truncated(
                &mut memory,
                (self.c.storage.pair_size * self.c.outer * self.c.inner) as f64
                    * fractions.c
                    * self.c.storage.dense_virtual_size as f64,
            );
        }
        memory
    }

    pub fn memory(&self, fractions: Fractions, child_memory: i64) -> i64 {
        self.temp(fractions).max(child_memory) + self.footprint(fractions)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicaOperand {
    pub storage: Storage,
    /// Dense element count used by the replication layer's source estimate.
    pub size: usize,
    pub communicator_ranks: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Replication {
    pub a: ReplicaOperand,
    pub b: ReplicaOperand,
    pub c: ReplicaOperand,
}

fn replica_bytes(operand: &ReplicaOperand, fraction: f64) -> usize {
    let unit = if operand.storage.sparse {
        operand.storage.pair_size
    } else {
        operand.storage.element_size
    };
    (fraction * operand.size as f64 * unit as f64) as usize
}

impl Replication {
    pub fn fixed_time(&self, models: &Models, fractions: Fractions) -> f64 {
        let mut time = 0.;
        for (operand, fraction) in [(&self.a, fractions.a), (&self.b, fractions.b)] {
            let bytes = replica_bytes(operand, fraction);
            for &ranks in &operand.communicator_ranks {
                time += models.communication(Communication::Broadcast, ranks, bytes);
            }
        }
        let bytes = replica_bytes(&self.c, fractions.c);
        for &ranks in &self.c.communicator_ranks {
            if self.c.storage.sparse {
                time += csr_reduction(models, ranks, bytes);
            } else {
                time += models.communication(
                    Communication::Reduce {
                        custom: self.c.storage.custom_addition,
                    },
                    ranks,
                    bytes,
                );
            }
        }
        time
    }

    /// Replication retains the parent's layer count, so the supplied child
    /// estimate is added without rescaling.
    pub fn est_time(&self, models: &Models, fractions: Fractions, child_time: f64) -> f64 {
        child_time + self.fixed_time(models, fractions)
    }

    pub fn footprint(&self, fractions: Fractions) -> i64 {
        let mut memory = 0i64;
        if self.a.communicator_ranks.len() > 1 && self.a.storage.sparse {
            add_truncated(
                &mut memory,
                fractions.a * (self.a.size * self.a.storage.pair_size) as f64,
            );
        }
        if self.b.communicator_ranks.len() > 1 && self.b.storage.sparse {
            add_truncated(
                &mut memory,
                fractions.b * (self.b.size * self.b.storage.pair_size) as f64,
            );
        }
        if !self.c.communicator_ranks.is_empty() && self.c.storage.sparse {
            add_truncated(
                &mut memory,
                3.0 * fractions.c * (self.c.size * self.c.storage.pair_size) as f64,
            );
        }
        memory
    }

    pub fn temp(&self) -> i64 {
        if !self.c.communicator_ranks.is_empty() && !self.c.storage.sparse {
            (self.c.storage.element_size * self.c.size) as i64
        } else {
            0
        }
    }

    pub fn memory(&self, fractions: Fractions, child_memory: i64) -> i64 {
        self.temp().max(child_memory) + self.footprint(fractions)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Virtual {
    pub dimensions: Vec<usize>,
    pub orders: [usize; 3],
}

impl Virtual {
    /// The source currently treats virtual-block work as communication-like and
    /// multiplies the supplied child estimate by the full virtual block count.
    pub fn est_time(&self, child_time: f64) -> f64 {
        self.dimensions.iter().product::<usize>() as f64 * child_time
    }

    pub fn footprint(&self) -> i64 {
        let integers = self.orders.iter().sum::<usize>() + 4 * self.dimensions.len();
        (integers * size_of::<i32>()) as i64
    }

    pub fn memory(&self, child_memory: i64) -> i64 {
        child_memory + self.footprint()
    }
}
