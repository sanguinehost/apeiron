//! Opt-in wall-clock attribution of the executor's phases.
//!
//! Off by default; `APEIRON_TRACE=1` (read once) or [`set_enabled`] turns it on. Off, a span is
//! one relaxed atomic load and no clock read. On, each span reads the monotonic clock twice and
//! adds its nanoseconds, its column count and its model-derived weight bytes and multiply-adds to
//! process-wide counters, so a harness can divide bytes by time (achieved bandwidth) and
//! multiply-adds by time (achieved throughput) per phase.
//!
//! The leaf phases do not nest. Inside [`Phase::Pass`] and [`Phase::Mtp`] the untraced remainder is
//! the enclosing time minus the leaves, so a report accounts for every nanosecond of a pass.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::time::Instant;

pub use crate::engine::error::Phase;

/// Every phase, in report order.
pub const PHASES: [Phase; 16] = [
    Phase::Embed,
    Phase::Norm,
    Phase::Residual,
    Phase::AttnProjIn,
    Phase::AttnCore,
    Phase::AttnProjOut,
    Phase::AffineProjIn,
    Phase::AffineConv,
    Phase::AffineHeads,
    Phase::AffineProjOut,
    Phase::Ffn,
    Phase::Head,
    Phase::Mtp,
    Phase::Restore,
    Phase::Pass,
    Phase::PoolWait,
];

impl Phase {
    /// The stable report name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::Norm => "norm",
            Self::Residual => "residual",
            Self::AttnProjIn => "attn.proj_in",
            Self::AttnCore => "attn.core",
            Self::AttnProjOut => "attn.proj_out",
            Self::AffineProjIn => "affine.proj_in",
            Self::AffineConv => "affine.conv",
            Self::AffineHeads => "affine.heads",
            Self::AffineProjOut => "affine.proj_out",
            Self::Ffn => "ffn",
            Self::Head => "head",
            Self::Mtp => "mtp",
            Self::Restore => "restore",
            Self::Pass => "pass",
            Self::PoolWait => "pool.wait",
        }
    }

    /// Whether the phase encloses leaf phases (excluded from the leaf partition).
    #[must_use]
    pub const fn encloses(self) -> bool {
        matches!(self, Self::Pass | Self::Mtp)
    }

    /// Whether the phase overlaps others (excluded from every partition sum).
    #[must_use]
    pub const fn overlay(self) -> bool {
        matches!(self, Self::PoolWait)
    }

    const fn index(self) -> usize {
        self as usize
    }
}

const N: usize = PHASES.len();

/// Per-phase counters: nanoseconds, spans, columns, weight bytes, multiply-adds, plus the
/// outside-pass KV allocation and prefix-cache insert timers.
struct Counters {
    ns: [AtomicU64; N],
    calls: [AtomicU64; N],
    cols: [AtomicU64; N],
    bytes: [AtomicU64; N],
    macs: [AtomicU64; N],
    alloc_ns: AtomicU64,
    alloc_calls: AtomicU64,
    insert_ns: AtomicU64,
    insert_calls: AtomicU64,
}

#[allow(
    clippy::declare_interior_mutable_const,
    reason = "array-repeat initializer for the static counter table"
)]
const ZERO: AtomicU64 = AtomicU64::new(0);

static COUNTERS: Counters = Counters {
    ns: [ZERO; N],
    calls: [ZERO; N],
    cols: [ZERO; N],
    bytes: [ZERO; N],
    macs: [ZERO; N],
    alloc_ns: ZERO,
    alloc_calls: ZERO,
    insert_ns: ZERO,
    insert_calls: ZERO,
};

/// Wait-histogram buckets: bucket `i` counts `Phase::PoolWait` spans in
/// `[2^i, 2^(i+1))` nanoseconds. Forty-eight buckets cover 1 ns to about 6.5 days.
const WAIT_BUCKETS: usize = 48;

/// The dispatch-size classes the wait is split by: `1`, `2`, `3-4`, `5-8`, `9-16`,
/// `17-32`, `33-64`, `65+` tasks. The `PoolWait` span carries the dispatch's task count in
/// its `cols`, so the class is the pool width the wait was paid on.
const WAIT_CLASSES: usize = 8;

/// The labels of [`WAIT_CLASSES`], in class order.
pub const WAIT_CLASS_LABELS: [&str; WAIT_CLASSES] =
    ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"];

#[allow(
    clippy::declare_interior_mutable_const,
    reason = "array-repeat initializer for the wait-histogram table"
)]
const ZERO_WAIT: AtomicU64 = AtomicU64::new(0);

/// One class's zeroed bucket row; a const item, so the outer repeat below may copy a non-`Copy`
/// array.
#[allow(
    clippy::declare_interior_mutable_const,
    reason = "array-repeat initializer for the wait-histogram table"
)]
const ZERO_ROW: [AtomicU64; WAIT_BUCKETS] = [ZERO_WAIT; WAIT_BUCKETS];

static WAIT_HIST: [[AtomicU64; WAIT_BUCKETS]; WAIT_CLASSES] = [ZERO_ROW; WAIT_CLASSES];
static WAIT_MAX_NS: [AtomicU64; WAIT_CLASSES] = [ZERO_WAIT; WAIT_CLASSES];

/// The per-dispatch `Phase::PoolWait` distribution, as power-of-two buckets by dispatch size.
///
/// `buckets[c][i]` counts class-`c` waits in `[2^i, 2^(i+1))` nanoseconds; `max_ns[c]` is the
/// longest class-`c` wait. The total over a class is the same count as the `pool.wait` phase's
/// `calls` for the dispatches of that size, so a distribution and a total are read from one
/// reset epoch.
#[derive(Clone, Copy, Debug)]
pub struct WaitHistogram {
    /// Counts per dispatch-size class and power-of-two nanosecond bucket.
    pub buckets: [[u64; WAIT_BUCKETS]; WAIT_CLASSES],
    /// The longest observed wait per class.
    pub max_ns: [u64; WAIT_CLASSES],
}

/// The `PoolWait` histogram since the last [`reset`].
#[must_use]
pub fn pool_wait_histogram() -> WaitHistogram {
    let mut buckets = [[0u64; WAIT_BUCKETS]; WAIT_CLASSES];
    let mut max_ns = [0u64; WAIT_CLASSES];
    for c in 0..WAIT_CLASSES {
        for (dst, src) in buckets[c].iter_mut().zip(WAIT_HIST[c].iter()) {
            *dst = src.load(Ordering::Relaxed);
        }
        max_ns[c] = WAIT_MAX_NS[c].load(Ordering::Relaxed);
    }
    WaitHistogram { buckets, max_ns }
}

/// The bucket a wait of `ns` nanoseconds falls in. A zero-length wait is bucket 0.
fn wait_bucket(ns: u64) -> usize {
    let e = usize::try_from(ns.max(1).ilog2()).unwrap_or(WAIT_BUCKETS - 1);
    if e < WAIT_BUCKETS {
        e
    } else {
        WAIT_BUCKETS - 1
    }
}

/// The class a dispatch of `n` tasks falls in; `0` and `1` share the single-task class.
fn wait_class(n: u64) -> usize {
    match n {
        0 | 1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=8 => 3,
        9..=16 => 4,
        17..=32 => 5,
        33..=64 => 6,
        _ => 7,
    }
}

fn record_wait(ns: u64, n: u64) {
    let c = wait_class(n);
    WAIT_HIST[c][wait_bucket(ns)].fetch_add(1, Ordering::Relaxed);
    WAIT_MAX_NS[c].fetch_max(ns, Ordering::Relaxed);
}

/// 0 unresolved, 1 off, 2 on.
static STATE: AtomicU8 = AtomicU8::new(0);

/// Whether tracing is on; resolves `APEIRON_TRACE` on first use.
#[inline]
#[must_use]
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let on = std::env::var("APEIRON_TRACE").is_ok_and(|v| v != "0");
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Turns tracing on or off for the process.
pub fn set_enabled(on: bool) {
    STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
}

/// Zeroes every counter.
pub fn reset() {
    for i in 0..N {
        COUNTERS.ns[i].store(0, Ordering::Relaxed);
        COUNTERS.calls[i].store(0, Ordering::Relaxed);
        COUNTERS.cols[i].store(0, Ordering::Relaxed);
        COUNTERS.bytes[i].store(0, Ordering::Relaxed);
        COUNTERS.macs[i].store(0, Ordering::Relaxed);
    }
    COUNTERS.alloc_ns.store(0, Ordering::Relaxed);
    COUNTERS.alloc_calls.store(0, Ordering::Relaxed);
    COUNTERS.insert_ns.store(0, Ordering::Relaxed);
    COUNTERS.insert_calls.store(0, Ordering::Relaxed);
    clear_wait_table(&WAIT_HIST, &WAIT_MAX_NS);
}

/// Zero a wait-histogram table; [`reset`] calls this loop.
fn clear_wait_table(hist: &[[AtomicU64; WAIT_BUCKETS]], max: &[AtomicU64]) {
    for class in hist {
        for bucket in class {
            bucket.store(0, Ordering::Relaxed);
        }
    }
    for m in max {
        m.store(0, Ordering::Relaxed);
    }
}

/// Records one outside-pass KV allocation duration in nanoseconds.
#[inline]
pub fn record_alloc(ns: u64) {
    COUNTERS.alloc_ns.fetch_add(ns, Ordering::Relaxed);
    COUNTERS.alloc_calls.fetch_add(1, Ordering::Relaxed);
}

/// Records one outside-pass prefix-cache insert duration in nanoseconds.
#[inline]
pub fn record_insert(ns: u64) {
    COUNTERS.insert_ns.fetch_add(ns, Ordering::Relaxed);
    COUNTERS.insert_calls.fetch_add(1, Ordering::Relaxed);
}

/// Returns `(alloc_ns, alloc_calls, insert_ns, insert_calls)` accumulated since [`reset`].
#[must_use]
pub fn outside_split() -> (u64, u64, u64, u64) {
    (
        COUNTERS.alloc_ns.load(Ordering::Relaxed),
        COUNTERS.alloc_calls.load(Ordering::Relaxed),
        COUNTERS.insert_ns.load(Ordering::Relaxed),
        COUNTERS.insert_calls.load(Ordering::Relaxed),
    )
}

/// A running span; its drop adds the elapsed time and its work to the phase.
#[must_use = "a span records on drop"]
pub struct Span {
    phase: Phase,
    start: Instant,
    cols: u64,
    bytes: u64,
    macs: u64,
}

/// Opens a span over `cols` columns reading `bytes` weight bytes for `macs` multiply-adds, or
/// `None` when tracing is off.
#[inline]
#[must_use]
pub fn span(phase: Phase, cols: usize, bytes: u64, macs: u64) -> Option<Span> {
    if enabled() {
        Some(Span {
            phase,
            start: Instant::now(),
            cols: cols as u64,
            bytes,
            macs,
        })
    } else {
        None
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let i = self.phase.index();
        let ns = u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        COUNTERS.ns[i].fetch_add(ns, Ordering::Relaxed);
        COUNTERS.calls[i].fetch_add(1, Ordering::Relaxed);
        COUNTERS.cols[i].fetch_add(self.cols, Ordering::Relaxed);
        COUNTERS.bytes[i].fetch_add(self.bytes, Ordering::Relaxed);
        COUNTERS.macs[i].fetch_add(self.macs, Ordering::Relaxed);
        if self.phase == Phase::PoolWait {
            record_wait(ns, self.cols);
        }
    }
}

/// `Q8_0` bytes of an `n_out × n_in` weight: 34 bytes per 32 weights.
#[must_use]
pub const fn q8_bytes(n_out: usize, n_in: usize) -> u64 {
    (n_out as u64) * (n_in as u64).div_ceil(32) * 34
}

/// Multiply-adds of `cols` columns through an `n_out × n_in` projection.
#[must_use]
pub const fn macs(n_out: usize, n_in: usize, cols: usize) -> u64 {
    (n_out as u64) * (n_in as u64) * (cols as u64)
}

/// One phase's accumulated counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseStat {
    /// Report name.
    pub name: &'static str,
    /// Whether the phase encloses the leaf phases.
    pub encloses: bool,
    /// Whether the phase overlaps the others (not part of any partition).
    pub overlay: bool,
    /// Nanoseconds inside the phase.
    pub ns: u64,
    /// Spans recorded.
    pub calls: u64,
    /// Columns carried.
    pub cols: u64,
    /// Model-derived weight bytes read (`Q8_0` geometry).
    pub bytes: u64,
    /// Model-derived multiply-adds.
    pub macs: u64,
}

/// The counters of every phase, in [`PHASES`] order.
#[must_use]
pub fn snapshot() -> Vec<PhaseStat> {
    let mut out: Vec<PhaseStat> = Vec::with_capacity(PHASES.len());
    let mut k = 0usize;
    while k < PHASES.len() {
        let p = PHASES[k];
        let i = p.index();
        out.push(PhaseStat {
            name: p.name(),
            encloses: p.encloses(),
            overlay: p.overlay(),
            ns: COUNTERS.ns[i].load(Ordering::Relaxed),
            calls: COUNTERS.calls[i].load(Ordering::Relaxed),
            cols: COUNTERS.cols[i].load(Ordering::Relaxed),
            bytes: COUNTERS.bytes[i].load(Ordering::Relaxed),
            macs: COUNTERS.macs[i].load(Ordering::Relaxed),
        });
        k += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every phase has a distinct index equal to its place in [`PHASES`].
    #[test]
    fn phases_index_their_table() {
        for (i, p) in PHASES.iter().enumerate() {
            assert_eq!(p.index(), i, "{}", p.name());
        }
    }

    /// A wait falls in the bucket whose range contains it; the bucket is its floor log2.
    #[test]
    fn waits_bucket_by_floor_log2() {
        assert_eq!(wait_bucket(0), 0);
        assert_eq!(wait_bucket(1), 0);
        assert_eq!(wait_bucket(2), 1);
        assert_eq!(wait_bucket(1023), 9);
        assert_eq!(wait_bucket(1024), 10);
        assert_eq!(wait_bucket(235_000), 17);
        assert_eq!(wait_bucket(u64::MAX), WAIT_BUCKETS - 1);
    }

    /// Dispatch sizes map to the eight class labels without a gap.
    #[test]
    fn dispatch_sizes_class_by_width() {
        assert_eq!(wait_class(0), 0);
        assert_eq!(wait_class(1), 0);
        assert_eq!(wait_class(2), 1);
        assert_eq!(wait_class(3), 2);
        assert_eq!(wait_class(4), 2);
        assert_eq!(wait_class(5), 3);
        assert_eq!(wait_class(8), 3);
        assert_eq!(wait_class(9), 4);
        assert_eq!(wait_class(16), 4);
        assert_eq!(wait_class(17), 5);
        assert_eq!(wait_class(32), 5);
        assert_eq!(wait_class(33), 6);
        assert_eq!(wait_class(64), 6);
        assert_eq!(wait_class(65), 7);
        assert_eq!(wait_class(1_295), 7);
    }

    /// A dropped `PoolWait` span reaches the class table through the real recording path.
    #[test]
    fn a_pool_wait_span_reaches_the_histogram() {
        set_enabled(true);
        reset();
        for _ in 0..3 {
            let s = span(Phase::PoolWait, 8, 0, 0);
            std::thread::sleep(std::time::Duration::from_micros(200));
            drop(s);
        }
        let h = pool_wait_histogram();
        let counted: u64 = h.buckets[3].iter().sum();
        assert!(counted >= 3, "class 5-8 counted {counted}");
        assert!(h.max_ns[3] >= 1, "class 5-8 max {}", h.max_ns[3]);
        let stat = snapshot()[Phase::PoolWait.index()];
        assert_eq!(stat.name, "pool.wait");
        assert!(stat.calls >= 3, "phase counted {}", stat.calls);
        reset();
        set_enabled(false);
    }
}
