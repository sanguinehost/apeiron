//! Exact-integer first-order sigma-delta (`APEIRON-ONEBIT-W8`; `docs/epics/ternary-hadamard.md`
//! Part II).
//!
//! [`sd_binary_block`] walks a block of fixed-point weights `w` and emits the sign stream
//! `s_t = if w_t + e_t ≥ 0 then +1 else −1`, updating the error state `e_{t+1} = w_t + e_t − a·s_t`
//! from `e_0 = 0`. It is the executed body the refinement `Numeric.SigmaDeltaRefinement` links to
//! the canonical `Numeric.SigmaDelta.sdState`/`sdSign`. Every add, multiply and subtraction is
//! checked `i64`: an overflow (or a shape mismatch) returns `None` and nothing wraps.
//!
//! [`sd_binary_block`] returns the final state and the maximum `|e_t|` over the block (`t ≥ 1`).
//! The peak is the `stab = max_t |e_t|/a` diagnostic the W7 harness reports; it is carried in the
//! same loop so the build path never re-runs the recurrence.
//!
//! This module is dependency-free (only `core`) so `proof/extract/lib.rs` can pull it in by
//! `#[path]` and the extracted body is this source.

#![allow(
    clippy::many_single_char_names,
    reason = "the sigma-delta block carries the error state `e`, the scale `a`, the step index `t` \
              and the block length `n` of the recurrence restated in `Numeric/SigmaDelta.lean`; the \
              pedantic lints stay on for the rest of the crate."
)]

/// One sigma-delta step: for the incoming weight `wt`, the current state `e` and the block scale
/// `a`, return `(sign, e', |e'|)` where `sign = if wt + e ≥ 0 then 1 else −1` (as an `i8` with
/// value `±1`) and `e' = wt + e − a·sign`, or `None` if a checked operation (including the
/// magnitude `|e'|`, unrepresentable only at `i64::MIN`) leaves `i64`.
///
/// All three outputs are exact: `sign` is `1`/`-1`, `e'` is the checked difference `(wt + e) − a·sign`,
/// and the third is `|e'|`.
fn sd_step(wt: i64, e: i64, a: i64) -> Option<(i8, i64, i64)> {
    match wt.checked_add(e) {
        None => None,
        Some(u) => {
            let sign: i8 = if u >= 0 { 1 } else { -1 };
            match a.checked_mul(i64::from(sign)) {
                None => None,
                Some(q) => match u.checked_sub(q) {
                    None => None,
                    Some(e2) => match 0_i64.checked_sub(e2) {
                        None => None,
                        Some(neg) => {
                            let mag: i64 = if e2 < 0 { neg } else { e2 };
                            Some((sign, e2, mag))
                        }
                    },
                },
            }
        }
    }
}

/// First-order sigma-delta over the block `w`, writing the emitted sign (`+1`/`-1` as `i8`) into
/// `s` and returning `(final state, max |e_t| over t ≥ 1)`, or `None` on a shape mismatch or a
/// checked overflow.
///
/// The recurrence is `e_0 = 0`, `s_t = if w_t + e_t ≥ 0 then +1 else −1`,
/// `e_{t+1} = w_t + e_t − a·s_t`; the returned peak is `max_{1 ≤ t ≤ len} |e_t|`.
///
/// # Errors
/// Returns `None` if `s.len() != w.len()`, or if a checked add/multiply/subtraction in the
/// recurrence leaves `i64` (the state or the scale is out of range for the fixed-point block).
///
/// # Complexity
/// `O(w.len())` checked `i64` operations and `O(1)` extra space; one `i8` write per weight.
pub fn sd_binary_block(w: &[i64], a: i64, s: &mut [i8]) -> Option<(i64, i64)> {
    let n = w.len();
    if s.len() != n {
        return None;
    }
    let mut e: i64 = 0;
    let mut peak: i64 = 0;
    let mut t: usize = 0;
    let mut ok: bool = true;
    while t < n && ok {
        let r = sd_step(w[t], e, a);
        match r {
            None => {
                ok = false;
            }
            Some((sign, e2, mag)) => {
                s[t] = sign;
                e = e2;
                if mag > peak {
                    peak = mag;
                }
            }
        }
        t += 1;
    }
    if ok {
        Some((e, peak))
    } else {
        None
    }
}
