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

/// A traced phase. The leaf phases partition a pass; [`Phase::Pass`] and [`Phase::Mtp`] enclose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Token embedding lookup and dequantization.
    Embed,
    /// RMS norms (pre-mixer, pre-FFN, output).
    Norm,
    /// Residual additions.
    Residual,
    /// Softmax-layer Q/K/V (and gate) projections.
    AttnProjIn,
    /// Bias, gate split, q/k norm, `RoPE`, cache append and the attention heads.
    AttnCore,
    /// Softmax-layer output projection.
    AttnProjOut,
    /// Affine-layer input projections (qkv, gate, beta, alpha).
    AffineProjIn,
    /// Affine-layer causal convolution.
    AffineConv,
    /// Affine-layer per-head state updates and reads.
    AffineHeads,
    /// Affine-layer output projection.
    AffineProjOut,
    /// The `SwiGLU` FFN: quantization, gate, up, hidden, down.
    Ffn,
    /// The output head (dense or bit-plane) and its argmax.
    Head,
    /// The MTP draft block (encloses its attention and FFN leaf phases).
    Mtp,
    /// Speculative rollback: cache truncation and affine restore.
    Restore,
    /// One `compose_cols` pass (encloses the leaf phases).
    Pass,
    /// Overlay: the dispatching thread's wait for the pool's other participants after finishing
    /// its own share, one span per dispatch (`calls` counts dispatches, `cols` their tasks). It
    /// overlaps the leaf phases and is excluded from the partition.
    PoolWait,
}

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

/// Per-phase counters: nanoseconds, spans, columns, weight bytes, multiply-adds.
struct Counters {
    ns: [AtomicU64; N],
    calls: [AtomicU64; N],
    cols: [AtomicU64; N],
    bytes: [AtomicU64; N],
    macs: [AtomicU64; N],
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
};

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
    PHASES
        .iter()
        .map(|&p| {
            let i = p.index();
            PhaseStat {
                name: p.name(),
                encloses: p.encloses(),
                overlay: p.overlay(),
                ns: COUNTERS.ns[i].load(Ordering::Relaxed),
                calls: COUNTERS.calls[i].load(Ordering::Relaxed),
                cols: COUNTERS.cols[i].load(Ordering::Relaxed),
                bytes: COUNTERS.bytes[i].load(Ordering::Relaxed),
                macs: COUNTERS.macs[i].load(Ordering::Relaxed),
            }
        })
        .collect()
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
}
