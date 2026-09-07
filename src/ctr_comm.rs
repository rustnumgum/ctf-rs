// Adapted from cc4s CTF contraction/ctr_comm.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Dense outer-replication communication for raw mapped contractions.

use crate::{
    algebra::{Semiring, Wire},
    context::Context,
    mapping::{Distribution, Mapping},
};

fn mark_physical(mapping: &Mapping, used: &mut [bool]) {
    match mapping {
        Mapping::Unmapped => {}
        Mapping::Physical { axis, child, .. } => {
            used[*axis] = true;
            mark_physical(child, used);
        }
        Mapping::Virtual { child, .. } => mark_physical(child, used),
    }
}

/// The physical topology fibers missing from each operand.
///
/// A wholly unused topology axis still makes the source install an empty
/// replication layer, but does not create any communicator or split work into
/// speculative 2.5D layers. Every created fiber is owned here and must be
/// released with [`Replication::close`].
pub struct Replication<'context> {
    comms: [Vec<Context<'context>>; 3],
    active: bool,
}

impl<'context> Replication<'context> {
    /// Derive `ctr_replicate` fibers directly from the three physical mappings.
    pub fn new(context: &'context Context<'_>, distributions: [&Distribution; 3]) -> Self {
        let topology = &distributions[0].topology;
        let mut physical: [Vec<bool>; 3] =
            std::array::from_fn(|_| vec![false; topology.dimensions.len()]);
        for operand in 0..3 {
            for mapping in &distributions[operand].mappings {
                mark_physical(mapping, &mut physical[operand]);
            }
        }

        let active = (0..topology.dimensions.len())
            .any(|axis| (0..3).any(|operand| !physical[operand][axis]));
        let mut comms: [Vec<Context<'context>>; 3] = std::array::from_fn(|_| Vec::new());
        for axis in 0..topology.dimensions.len() {
            if !(physical[0][axis] || physical[1][axis] || physical[2][axis]) {
                continue;
            }
            for operand in 0..3 {
                if !physical[operand][axis] {
                    comms[operand].push(topology.fiber(context, axis));
                }
            }
        }
        Self { comms, active }
    }

    /// Broadcast inputs, apply beta only on the output-root replica, execute
    /// the recursive child once, reduce C to root, and clear nonroot input
    /// replicas. The child beta is one on the root and zero elsewhere whenever
    /// the source replication wrapper is present.
    pub fn execute<A, F>(
        &self,
        algebra: &A,
        a: &mut [A::Element],
        b: &mut [A::Element],
        c: &mut [A::Element],
        beta: A::Element,
        child: F,
    ) where
        A: Semiring,
        A::Element: Wire,
        F: FnOnce(&[A::Element], &[A::Element], &mut [A::Element], A::Element),
    {
        for comm in &self.comms[0] {
            comm.broadcast(0, a);
        }
        for comm in &self.comms[1] {
            comm.broadcast(0, b);
        }

        let output_root = self.comms[2].iter().all(|comm| comm.rank() == 0);
        if self.active && output_root && beta != algebra.one() {
            if beta == algebra.zero() {
                c.fill(algebra.zero());
            } else {
                for value in c.iter_mut() {
                    *value = algebra.multiply(&beta, value);
                }
            }
        }
        let child_beta = if self.active {
            if output_root {
                algebra.one()
            } else {
                algebra.zero()
            }
        } else {
            beta
        };
        child(a, b, c, child_beta);

        for comm in &self.comms[2] {
            comm.reduce_monoid(algebra, c, false, 0);
        }
        if self.comms[0].iter().any(|comm| comm.rank() != 0) {
            a.fill(algebra.zero());
        }
        if self.comms[1].iter().any(|comm| comm.rank() != 0) {
            b.fill(algebra.zero());
        }
    }

    /// Collectively release every topology fiber created by [`Replication::new`].
    pub fn close(self) {
        for group in self.comms {
            for comm in group {
                comm.close();
            }
        }
    }
}
