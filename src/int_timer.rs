//! CPU/MPI timer state adapted from `shared/int_timer.cxx`.
//!
//! The source keeps one accumulated record per symbol and accounts for nested
//! timers by subtracting time already charged to an inner symbol.  This module
//! keeps that accounting explicit and leaves the communicator operation to a
//! small all-reduce trait so that the MPI context remains the owner of raw
//! communicators.

use crate::{algebra::Arithmetic, context::Context};
use std::{collections::BTreeMap, time::Instant};

/// The size of the source timer name buffer.
pub const MAX_NAME_LENGTH: usize = 53;

/// A local timer record.  `acc_*` are rank-local accumulators; `total_*` are
/// filled by `compute_totals`, matching the source's two-stage accounting.
#[derive(Clone, Debug)]
pub struct FunctionTimer {
    pub name: String,
    pub start_time: Option<Instant>,
    pub start_excl_time: f64,
    pub acc_time: f64,
    pub acc_excl_time: f64,
    pub calls: i64,
    pub total_time: f64,
    pub total_excl_time: f64,
    pub total_calls: i64,
}

impl FunctionTimer {
    pub fn new(name: impl Into<String>) -> Self {
        Self::new_at(name, Instant::now(), 0.0)
    }

    pub fn new_at(name: impl Into<String>, start_time: Instant, start_excl_time: f64) -> Self {
        let name = name.into();
        assert!(name.len() + 1 < MAX_NAME_LENGTH);
        Self {
            name,
            start_time: Some(start_time),
            start_excl_time,
            acc_time: 0.0,
            acc_excl_time: 0.0,
            calls: 0,
            total_time: 0.0,
            total_excl_time: 0.0,
            total_calls: 0,
        }
    }

    /// Set the source start snapshot before a new invocation.
    pub fn start_at(&mut self, start_time: Instant, start_excl_time: f64) {
        self.start_time = Some(start_time);
        self.start_excl_time = start_excl_time;
    }

    /// Add one invocation with a known wall duration and already-accounted
    /// child duration.  This is the arithmetic used by `stop_at` and is also
    /// useful when an execution layer has its own wall-clock sample.
    pub fn record(&mut self, elapsed: f64, child_time: f64) {
        self.acc_time += elapsed;
        self.acc_excl_time += elapsed - child_time;
        self.calls += 1;
    }

    /// Stop this record at a monotonic timestamp.  `excl_time` is the current
    /// registry-wide exclusive clock, as in the source implementation.
    pub fn stop_at(&mut self, stop_time: Instant, excl_time: f64) -> f64 {
        let start_time = self.start_time.expect("timer must be started before stop");
        let elapsed = stop_time.duration_since(start_time).as_secs_f64();
        self.record(elapsed, excl_time - self.start_excl_time);
        elapsed
    }

    /// Local snapshot used before an MPI reduction.
    pub fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            name: self.name.clone(),
            acc_time: self.acc_time,
            acc_excl_time: self.acc_excl_time,
            calls: self.calls,
        }
    }

    /// Sum this rank's local counters over the communicator.
    pub fn compute_totals(&mut self, context: &Context<'_>) {
        self.total_time = context.all_reduce(&Arithmetic::<f64>::new(), &self.acc_time);
        self.total_excl_time = context.all_reduce(&Arithmetic::<f64>::new(), &self.acc_excl_time);
        self.total_calls = context.all_reduce(&Arithmetic::<i64>::new(), &self.calls);
    }
}

/// A copyable rank-local timer snapshot, suitable for explicit MPI exchange.
#[derive(Clone, Debug, PartialEq)]
pub struct TimerSnapshot {
    pub name: String,
    pub acc_time: f64,
    pub acc_excl_time: f64,
    pub calls: i64,
}

/// One non-nested wall timer.  `TimerRegistry` is the form to use when the
/// source's nested exclusive accounting is needed.
#[derive(Debug)]
pub struct Timer {
    pub timer_name: String,
    function: FunctionTimer,
    running: bool,
}

impl Timer {
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let function = FunctionTimer::new(name.clone());
        Self {
            timer_name: name,
            function,
            running: false,
        }
    }

    pub fn start(&mut self) {
        self.function.start_at(Instant::now(), 0.0);
        self.running = true;
    }

    /// Stop once; repeated stops are no-ops like the source's `exited` guard.
    pub fn stop(&mut self) -> f64 {
        if !self.running {
            return 0.0;
        }
        let elapsed = self.function.stop_at(Instant::now(), 0.0);
        self.running = false;
        elapsed
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn elapsed(&self) -> f64 {
        self.function.acc_time
            + if self.running {
                self.function
                    .start_time
                    .expect("running timer has a start")
                    .elapsed()
                    .as_secs_f64()
            } else {
                0.0
            }
    }

    pub fn snapshot(&self) -> TimerSnapshot {
        self.function.snapshot()
    }

    pub fn function_timer(&self) -> &FunctionTimer {
        &self.function
    }

    /// The source method performs profile finalization.  Explicit Rust
    /// registries own their lifetime, so this only marks the timer stopped.
    pub fn exit(&mut self) {
        self.running = false;
    }
}

/// Explicit registry for source-style same-name accumulation and nested
/// exclusive timing.
#[derive(Debug, Default)]
pub struct TimerRegistry {
    timers: BTreeMap<String, FunctionTimer>,
    excl_time: f64,
}

impl TimerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start(&mut self, name: impl Into<String>) {
        self.start_at(name, Instant::now());
    }

    pub fn start_at(&mut self, name: impl Into<String>, start_time: Instant) {
        let name = name.into();
        let excl_time = self.excl_time;
        let timer = self
            .timers
            .entry(name.clone())
            .or_insert_with(|| FunctionTimer::new_at(name, start_time, excl_time));
        timer.start_at(start_time, excl_time);
    }

    pub fn stop(&mut self, name: &str) -> f64 {
        self.stop_at(name, Instant::now())
    }

    pub fn stop_at(&mut self, name: &str, stop_time: Instant) -> f64 {
        let timer = self
            .timers
            .get_mut(name)
            .expect("timer must be started before stop");
        let start_excl_time = timer.start_excl_time;
        let elapsed = timer.stop_at(stop_time, self.excl_time);
        self.excl_time = start_excl_time + elapsed;
        elapsed
    }

    pub fn exclusive_time(&self) -> f64 {
        self.excl_time
    }

    pub fn get(&self, name: &str) -> Option<&FunctionTimer> {
        self.timers.get(name)
    }

    pub fn snapshot(&self) -> Vec<TimerSnapshot> {
        let mut snapshot: Vec<_> = self.timers.values().map(FunctionTimer::snapshot).collect();
        snapshot.sort_by(|left, right| right.acc_time.total_cmp(&left.acc_time));
        snapshot
    }

    pub fn compute_totals(&mut self, context: &Context<'_>) {
        for timer in self.timers.values_mut() {
            timer.compute_totals(context);
        }
    }

    pub fn clear(&mut self) {
        self.timers.clear();
        self.excl_time = 0.0;
    }
}
