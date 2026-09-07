// Adapted from cc4s shared/memcontrol.{h,cxx}.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Per-process memory capacity for low-memory planning.
//!
//! The source computes `memcap * physical_memory / processes_per_machine -
//! process_bytes_used`. Rust allocations are not routed through a central CTF
//! allocator, so `used_bytes` is the operating-system resident set. Linux also
//! caps physical memory by an active cgroup limit instead of assuming bare host
//! RAM; this is the platform-corrected form of the same source formula.

use crate::{algebra::CustomMonoid, context::Context};
use std::{fmt, io, num::NonZeroUsize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub total_bytes: u64,
    pub used_bytes: u64,
}

impl MemorySnapshot {
    pub fn from_linux(
        meminfo: &str,
        status: &str,
        cgroup_limit: Option<&str>,
    ) -> Result<Self, Error> {
        let host = kib_field(meminfo, "MemTotal")?;
        let used = kib_field(status, "VmRSS")?;
        let limit = cgroup_limit.map(parse_limit).transpose()?.flatten();
        let total = limit.map_or(host, |limit| host.min(limit));
        if total == 0 {
            return Err(Error::InvalidData("memory total"));
        }
        Ok(Self {
            total_bytes: total,
            used_bytes: used,
        })
    }

    pub fn discover() -> Result<Self, Error> {
        platform::discover()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryFraction {
    numerator: u64,
    denominator: u64,
}

impl MemoryFraction {
    pub fn new(numerator: u64, denominator: u64) -> Result<Self, Error> {
        if denominator == 0 || numerator == 0 || numerator > denominator {
            return Err(Error::InvalidFraction);
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }

    fn of(self, bytes: u64) -> u64 {
        ((bytes as u128 * self.numerator as u128) / self.denominator as u128) as u64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessesPerMachine(NonZeroUsize);

impl ProcessesPerMachine {
    pub fn new(value: usize) -> Result<Self, Error> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(Error::ZeroProcessesPerMachine)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessMemoryBudget(u64);

impl ProcessMemoryBudget {
    pub fn from_snapshot(
        snapshot: MemorySnapshot,
        processes: ProcessesPerMachine,
        fraction: MemoryFraction,
    ) -> Self {
        let capacity = fraction.of(snapshot.total_bytes) / processes.0.get() as u64;
        Self(capacity.saturating_sub(snapshot.used_bytes))
    }

    pub fn discover(
        processes: ProcessesPerMachine,
        fraction: MemoryFraction,
    ) -> Result<Self, Error> {
        Ok(Self::from_snapshot(
            MemorySnapshot::discover()?,
            processes,
            fraction,
        ))
    }

    pub fn collective_min(self, context: &Context<'_>) -> Self {
        let minimum = CustomMonoid {
            identity: u64::MAX,
            addition: |left: &u64, right: &u64| (*left).min(*right),
        };
        Self(context.all_reduce(&minimum, &self.0))
    }

    pub fn bytes(self) -> u64 {
        self.0
    }
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    InvalidData(&'static str),
    InvalidFraction,
    ZeroProcessesPerMachine,
    UnsupportedPlatform,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "memory discovery failed: {error}"),
            Self::InvalidData(field) => write!(f, "invalid memory data: {field}"),
            Self::InvalidFraction => f.write_str("memory fraction must be in (0, 1]"),
            Self::ZeroProcessesPerMachine => f.write_str("processes per machine must be nonzero"),
            Self::UnsupportedPlatform => f.write_str("memory discovery is unsupported"),
        }
    }
}

impl std::error::Error for Error {}

fn kib_field(contents: &str, name: &'static str) -> Result<u64, Error> {
    let prefix = format!("{name}:");
    let line = contents
        .lines()
        .find(|line| line.starts_with(&prefix))
        .ok_or(Error::InvalidData(name))?;
    let mut words = line[prefix.len()..].split_whitespace();
    let value: u64 = words
        .next()
        .ok_or(Error::InvalidData(name))?
        .parse()
        .map_err(|_| Error::InvalidData(name))?;
    if words.next() != Some("kB") || words.next().is_some() {
        return Err(Error::InvalidData(name));
    }
    value.checked_mul(1024).ok_or(Error::InvalidData(name))
}

fn parse_limit(value: &str) -> Result<Option<u64>, Error> {
    let value = value.trim();
    if value == "max" {
        Ok(None)
    } else {
        value
            .parse()
            .map(Some)
            .map_err(|_| Error::InvalidData("cgroup memory limit"))
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::*;
    use std::path::Path;

    pub(super) fn discover() -> Result<MemorySnapshot, Error> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").map_err(Error::Io)?;
        let status = std::fs::read_to_string("/proc/self/status").map_err(Error::Io)?;
        let v2 = Path::new("/sys/fs/cgroup/memory.max");
        let v1 = Path::new("/sys/fs/cgroup/memory/memory.limit_in_bytes");
        let limit = if v2.exists() {
            Some(std::fs::read_to_string(v2).map_err(Error::Io)?)
        } else if v1.exists() {
            Some(std::fs::read_to_string(v1).map_err(Error::Io)?)
        } else {
            None
        };
        MemorySnapshot::from_linux(&meminfo, &status, limit.as_deref())
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use super::*;
    use std::{ffi::c_void, mem::size_of};

    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        load: u32,
        total: u64,
        available: u64,
        total_page: u64,
        available_page: u64,
        total_virtual: u64,
        available_virtual: u64,
        extended: u64,
    }
    #[repr(C)]
    struct ProcessMemoryCounters {
        size: u32,
        faults: u32,
        peak: usize,
        working_set: usize,
        quota_peak_paged: usize,
        quota_paged: usize,
        quota_peak_nonpaged: usize,
        quota_nonpaged: usize,
        pagefile: usize,
        peak_pagefile: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }

    pub(super) fn discover() -> Result<MemorySnapshot, Error> {
        let mut host: MemoryStatusEx = unsafe { std::mem::zeroed() };
        host.length = size_of::<MemoryStatusEx>() as u32;
        if unsafe { GlobalMemoryStatusEx(&mut host) } == 0 {
            return Err(Error::Io(io::Error::last_os_error()));
        }
        let mut process: ProcessMemoryCounters = unsafe { std::mem::zeroed() };
        process.size = size_of::<ProcessMemoryCounters>() as u32;
        if unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut process, process.size) } == 0 {
            return Err(Error::Io(io::Error::last_os_error()));
        }
        if host.total == 0 {
            return Err(Error::InvalidData("GlobalMemoryStatusEx"));
        }
        Ok(MemorySnapshot {
            total_bytes: host.total,
            used_bytes: process.working_set as u64,
        })
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
mod platform {
    use super::*;
    pub(super) fn discover() -> Result<MemorySnapshot, Error> {
        Err(Error::UnsupportedPlatform)
    }
}
