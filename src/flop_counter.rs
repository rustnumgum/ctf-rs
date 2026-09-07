//! Explicit local floating-point operation accounting.
//!
//! The pinned CTF implementation stores a running process-local total and
//! lets `Flop_counter::count` all-reduce the interval. Rust keeps that total
//! in a safe atomic and each counter value stores only its baseline snapshot.

use crate::{algebra::Arithmetic, context::Context};
use std::sync::atomic::{AtomicI64, Ordering};

static COMPUTED_FLOP_COUNT: AtomicI64 = AtomicI64::new(0);

/// Add operations executed by this process to the running total.
pub fn add_computed_flops(flops: i64) {
    COMPUTED_FLOP_COUNT.fetch_add(flops, Ordering::Relaxed);
}

/// Read this process's running total.
pub fn get_computed_flops() -> i64 {
    COMPUTED_FLOP_COUNT.load(Ordering::Relaxed)
}

/// A local flop interval counter with MPI aggregation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlopCounter {
    /// Local total at the beginning of the current interval.
    pub start_count: i64,
}

impl FlopCounter {
    /// Start a counter at the process's current total.
    pub fn new() -> Self {
        Self {
            start_count: get_computed_flops(),
        }
    }

    /// Restart the measured interval at the current local total.
    pub fn zero(&mut self) {
        self.start_count = get_computed_flops();
    }

    /// Return this rank's count since construction or the last [`zero`].
    pub fn local(&self) -> i64 {
        get_computed_flops() - self.start_count
    }

    /// Return the interval total over every rank in `context`.
    pub fn count(&self, context: &Context<'_>) -> i64 {
        context.all_reduce(&Arithmetic::<i64>::new(), &self.local())
    }
}

impl Default for FlopCounter {
    fn default() -> Self {
        Self::new()
    }
}
