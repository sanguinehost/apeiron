//! Sub-phase timing of the resident ADMM step, under `APEIRON_NQ_ADMM_TRACE=1`.
//!
//! The outer iteration is `cb1` (plane build and the three limb products), the inverse of `hw`
//! (`inv`: the device recursion plus its host leaves), `cb2` (`Rᵀ·H⁻¹` and the read map), the
//! `rank1` power steps, and `commit` (`U += X − Z`, `Z ← Z'`, the peer's `Z_x`). The counters are
//! nanoseconds summed **over both sides**: `cb1`, `inv`, `cb2` and `rank1` run once per side per
//! iteration on two threads, `commit` once per pair. [`take`] drains them, so one factorisation
//! prints its own split.
//!
//! The phases are read with `Instant::elapsed` around the submits that already wait, so the trace
//! adds no synchronisation; the residual between the loop's wall and the sums is the host work
//! outside the frames (draws, thread spawn, flag reads) and the readback.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// The five phases, in print order.
pub(crate) const NAMES: [&str; 5] = ["cb1", "inv", "cb2", "rank1", "commit"];
/// Indices into [`NAMES`] and the counters.
pub(crate) const CB1: usize = 0;
pub(crate) const INV: usize = 1;
pub(crate) const CB2: usize = 2;
pub(crate) const RANK1: usize = 3;
pub(crate) const COMMIT: usize = 4;

static NANOS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Committed outer iterations since the last [`take`].
static ITERS: AtomicU64 = AtomicU64::new(0);

/// Whether the trace is on (`APEIRON_NQ_ADMM_TRACE` set and not `0`), read once.
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("APEIRON_NQ_ADMM_TRACE").is_ok_and(|v| v.trim() != "0"))
}

/// Add the time `since` its start to `phase`; a no-op unless the trace is on.
///
/// # Complexity
///
/// `O(1)`.
pub(crate) fn add(phase: usize, since: Instant) {
    if !on() {
        return;
    }
    let ns = u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX);
    NANOS[phase].fetch_add(ns, Ordering::Relaxed);
}

/// Count one committed outer iteration; a no-op unless the trace is on.
///
/// # Complexity
///
/// `O(1)`.
pub(crate) fn iter() {
    if on() {
        ITERS.fetch_add(1, Ordering::Relaxed);
    }
}

/// The phase sums in milliseconds and the committed iteration count, drained. `None` when the
/// trace is off.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub(crate) fn take() -> Option<([f64; 5], u64)> {
    if !on() {
        return None;
    }
    let mut ms = [0f64; 5];
    for (m, cell) in ms.iter_mut().zip(&NANOS) {
        *m = cell.swap(0, Ordering::Relaxed) as f64 / 1e6;
    }
    Some((ms, ITERS.swap(0, Ordering::Relaxed)))
}
