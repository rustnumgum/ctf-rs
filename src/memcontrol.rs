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
    // SAFETY: signatures mirror the documented Win32 ABI for
    // kernel32!GlobalMemoryStatusEx and kernel32!GetCurrentProcess (fixed
    // argument count and layout, no varargs); soundness of each call is
    // argued at its call site below.
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
    }
    // SAFETY: signature mirrors the documented Win32 ABI for
    // psapi!GetProcessMemoryInfo; soundness of the call is argued at its
    // call site below.
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }

    pub(super) fn discover() -> Result<MemorySnapshot, Error> {
        // SAFETY: MemoryStatusEx is #[repr(C)] with only u32/u64 fields, so
        // the all-zero bit pattern is a valid value; `length` is set to the
        // struct size (as the API requires) before it crosses the FFI
        // boundary below.
        let mut host: MemoryStatusEx = unsafe { std::mem::zeroed() };
        host.length = size_of::<MemoryStatusEx>() as u32;
        // SAFETY: `host` is a uniquely-owned, live MemoryStatusEx with
        // `length` already set as GlobalMemoryStatusEx requires; the
        // pointer stays valid for the duration of this synchronous call.
        if unsafe { GlobalMemoryStatusEx(&mut host) } == 0 {
            return Err(Error::Io(io::Error::last_os_error()));
        }
        // SAFETY: same zero-is-valid argument as above; ProcessMemoryCounters
        // is #[repr(C)] with only u32/usize fields.
        let mut process: ProcessMemoryCounters = unsafe { std::mem::zeroed() };
        process.size = size_of::<ProcessMemoryCounters>() as u32;
        // SAFETY: GetCurrentProcess() returns the pseudo-handle -1, valid
        // for the process lifetime and needing no close; `process` is a
        // uniquely-owned, live buffer of exactly `process.size` bytes, the
        // size GetProcessMemoryInfo is told to write into.
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

// Pinned upstream (src/shared/memcontrol.cxx proc_bytes_total, __MACH__
// branch, lines 370-374) reads total physical memory through
// `sysctl(CTL_HW, HW_MEMSIZE)`; this uses the equivalent named
// `sysctlbyname("hw.memsize")` form. Upstream's used-memory figure
// (proc_bytes_used) is mallinfo()-based allocator bookkeeping, which the
// Linux and Windows branches above already replace with the OS-reported
// current resident set size (VmRSS, WorkingSetSize); this branch keeps
// that same "current RSS" semantics via `task_info`'s
// `MACH_TASK_BASIC_INFO.resident_size`, rather than reproducing
// mallinfo() or getrusage's peak-based `ru_maxrss`.
#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use std::ffi::{CString, c_char, c_void};

    const MACH_TASK_BASIC_INFO: i32 = 20;
    const KERN_SUCCESS: i32 = 0;

    // time_value_t: two 32-bit fields, seconds and microseconds.
    #[repr(C)]
    struct TimeValue {
        seconds: i32,
        microseconds: i32,
    }
    // mach_task_basic_info_data_t (<mach/task_info.h>); 12
    // natural_t-sized (4-byte) words, matched field-for-field so the
    // struct's size and layout equal what the kernel writes.
    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: TimeValue,
        system_time: TimeValue,
        policy: i32,
        suspend_count: i32,
    }

    // SAFETY: signatures mirror the documented Darwin libSystem ABI
    // (`<sys/sysctl.h>`, `<mach/mach_init.h>`, `<mach/task.h>`); no
    // `#[link(...)]` is needed because every macOS binary links against
    // libSystem, which provides all three. `mach_task_self_` is a plain
    // extern global (not a callable function); the real `mach_task_self()`
    // is a header-only macro that just reads it, so it is declared and
    // read here rather than called.
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> i32;
        static mach_task_self_: u32;
        fn task_info(
            target_task: u32,
            flavor: i32,
            task_info_out: *mut i32,
            task_info_out_count: *mut u32,
        ) -> i32;
    }

    pub(super) fn discover() -> Result<MemorySnapshot, Error> {
        let name = CString::new("hw.memsize").unwrap();
        let mut total: u64 = 0;
        let mut total_len = std::mem::size_of::<u64>();
        // SAFETY: `name` is a valid, NUL-terminated, live CString for the
        // call; `oldp` points to a live, uniquely-owned `u64` and
        // `oldlenp` to its exact byte length, both of which
        // sysctlbyname is told and required to respect; `newp`/`newlen`
        // are null/0, the documented "read only" form.
        let code = unsafe {
            sysctlbyname(
                name.as_ptr(),
                (&mut total as *mut u64).cast(),
                &mut total_len,
                std::ptr::null_mut(),
                0,
            )
        };
        if code != 0 {
            return Err(Error::Io(io::Error::last_os_error()));
        }
        if total == 0 || total_len != std::mem::size_of::<u64>() {
            return Err(Error::InvalidData("sysctlbyname hw.memsize"));
        }

        // SAFETY: MachTaskBasicInfo is #[repr(C)] with only u64/i32
        // fields, so the all-zero bit pattern is a valid value.
        let mut info: MachTaskBasicInfo = unsafe { std::mem::zeroed() };
        let mut count =
            (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<i32>()) as u32;
        // SAFETY: `mach_task_self_` is the current task's own port,
        // valid for the process's whole lifetime and needing no
        // deallocation; `info` is a live, uniquely-owned buffer whose
        // size in 4-byte words was just placed in `count`, matching what
        // `task_info_out`/`task_info_out_count` require; `task_info`
        // only reads the current task's accounting and writes into
        // `info` for the duration of this synchronous call.
        let result = unsafe {
            task_info(
                mach_task_self_,
                MACH_TASK_BASIC_INFO,
                (&mut info as *mut MachTaskBasicInfo).cast(),
                &mut count,
            )
        };
        if result != KERN_SUCCESS {
            return Err(Error::InvalidData("mach task_info"));
        }

        Ok(MemorySnapshot {
            total_bytes: total,
            used_bytes: info.resident_size,
        })
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
mod platform {
    use super::*;
    pub(super) fn discover() -> Result<MemorySnapshot, Error> {
        Err(Error::UnsupportedPlatform)
    }
}
