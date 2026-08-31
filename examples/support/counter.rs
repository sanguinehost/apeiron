//! Instructions and cycles for a region of this process, read without an external `perf`.
//!
//! Opens two counters on the calling thread through `perf_event_open`, enables them around a
//! closure and reads the deltas. Takes an event region to measure; returns retired instructions
//! and core cycles for it, or `None` when the kernel refuses the counters.
//!
//! # Why in-process and not `perf stat`
//!
//! `perf stat` counts a whole process, so an arm has to be the only thing the process does and a
//! row costs a fork. That is why this crate's counter-backed harnesses ran one arm per process and
//! its wall-clock harnesses ran all arms in one -- and the two disagreed by 3x on
//! `benches/against_the_field.rs` without either being able to check the other. Counted here, a
//! row carries its own nanoseconds and its own instruction count from the same loop, and a
//! disagreement between them is a benchmark bug rather than a mystery.
//!
//! # The counters are raw, because this CPU is hybrid
//!
//! On a P-core/E-core part `PERF_TYPE_HARDWARE` names a generic event that resolves against
//! whichever PMU the thread lands on, and the two PMUs are not comparable. This opens the
//! `cpu_core` PMU by its `/sys` type and asks for raw `0xc0` (instructions retired) and `0x3c`
//! (core cycles), so a run pinned off the P-cores fails to open rather than reporting an E-core
//! number as if it were the same measurement.

use std::fs;
use std::os::fd::{FromRawFd, OwnedFd};

#[repr(C)]
#[derive(Default)]
struct Attr {
    kind: u32,
    size: u32,
    config: u64,
    sample: u64,
    sample_type: u64,
    read_format: u64,
    flags: u64,
    wakeup: u32,
    bp_type: u32,
    config1: u64,
    config2: u64,
    branch_sample_type: u64,
    sample_regs_user: u64,
    sample_stack_user: u32,
    clockid: i32,
    sample_regs_intr: u64,
    aux_watermark: u32,
    sample_max_stack: u16,
    reserved_2: u16,
    aux_sample_size: u32,
    reserved_3: u32,
    sig_data: u64,
}

const DISABLED: u64 = 1 << 0;
const EXCLUDE_KERNEL: u64 = 1 << 5;
const EXCLUDE_HV: u64 = 1 << 6;

const IOC_ENABLE: u64 = 0x2400;
const IOC_DISABLE: u64 = 0x2401;
const IOC_RESET: u64 = 0x2403;

/// Two open counters on this thread: retired instructions and core cycles.
pub struct Counter {
    instructions: OwnedFd,
    cycles: OwnedFd,
}

/// What one measured region cost.
#[derive(Clone, Copy, Default)]
pub struct Counts {
    pub instructions: u64,
    pub cycles: u64,
}

fn open(pmu: u32, config: u64) -> Option<OwnedFd> {
    let mut attr = Attr {
        kind: pmu,
        size: u32::try_from(std::mem::size_of::<Attr>()).ok()?,
        config,
        flags: DISABLED | EXCLUDE_KERNEL | EXCLUDE_HV,
        ..Attr::default()
    };
    // pid 0, cpu -1: this thread, wherever it runs. group -1: its own group.
    let fd = unsafe {
        libc::syscall(
            libc::SYS_perf_event_open,
            std::ptr::from_mut(&mut attr),
            0,
            -1,
            -1,
            0u64,
        )
    };
    let fd = i32::try_from(fd).ok()?;
    if fd < 0 {
        return None;
    }
    Some(unsafe { OwnedFd::from_raw_fd(fd) })
}

impl Counter {
    /// Opens the two counters, or returns `None` if the kernel refuses.
    ///
    /// Refusal is expected and not an error: `perf_event_paranoid` above 2, a container without
    /// `CAP_PERFMON`, or a machine with no `cpu_core` PMU. A caller reports nanoseconds alone.
    pub fn open() -> Option<Self> {
        let pmu: u32 = fs::read_to_string("/sys/bus/event_source/devices/cpu_core/type")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(4);
        Some(Self { instructions: open(pmu, 0x00c0)?, cycles: open(pmu, 0x003c)? })
    }

    fn ioctl(&self, request: u64) {
        for fd in [&self.instructions, &self.cycles] {
            unsafe { libc::ioctl(std::os::fd::AsRawFd::as_raw_fd(fd), request, 0) };
        }
    }

    fn read(fd: &OwnedFd) -> u64 {
        let mut buf = [0u8; 8];
        let n = unsafe {
            libc::read(
                std::os::fd::AsRawFd::as_raw_fd(fd),
                buf.as_mut_ptr().cast(),
                8,
            )
        };
        if n == 8 { u64::from_ne_bytes(buf) } else { 0 }
    }

    /// Counts one closure. Returns its value and what it cost.
    pub fn count<T>(&self, f: impl FnOnce() -> T) -> (T, Counts) {
        self.ioctl(IOC_RESET);
        self.ioctl(IOC_ENABLE);
        let out = f();
        self.ioctl(IOC_DISABLE);
        (
            out,
            Counts {
                instructions: Self::read(&self.instructions),
                cycles: Self::read(&self.cycles),
            },
        )
    }
}
