//! Bit-identical fast forms of two attention-stage loops of the composed step.
//!
//! - [`sigmoid_gate`]: `att[i] ← mul_q32(att[i], fixed_sigmoid(gate[i]))`, the output gate.
//!   [`fixed_sigmoid`](crate::numeric::fixed_ln::fixed_sigmoid) is the oracle: `sigmoid_lanes`
//!   returns its word for every `i64` input tested (edges, random magnitudes, and a dense sweep of
//!   `[-18, 18] * ONE` at stride 97 in the tests), with the
//!   `i128` division replaced by an exact `u128` quotient and a remainder check in `u64`
//!   remainder and the sign branches of the `i128` rounding shifts made branch-free.
//! - [`rotate_pair_prefixes`]: `RoPE` on the rotary prefix of every head of a query and a key at
//!   one position. The rotation angle depends on `(pos, j)` only, so the `fixed_sin_cos` of each
//!   rotary pair is evaluated once and applied to every head of both vectors, where
//!   [`RopeTable::rotate_one`] re-evaluates it per head.
//!
//! Both are Q32.32 integer arithmetic on the path every carrier takes; the f32 carrier's outputs
//! and the Q32 default's outputs are unchanged word for word.

#![allow(
    clippy::inline_always,
    reason = "the one-line arithmetic helpers sit in the gate's hot loop and must inline into the lane loop"
)]

use super::{RopeTable, StepError};
use crate::numeric::fixed_trig::{fixed_sin_cos, turn_to_radians};

const ONE: i64 = 1 << 32;
const LOWER: i64 = -16 * ONE;
const LN2_Q32: i64 = 2_977_044_472;
const LN2_RECIP_Q32: i64 = 6_196_328_019;

/// `round(a · r / 2^32)`, ties away from zero (`fixed_exp`'s `round_div_2p32` of the product),
/// without a sign branch: `(v + 2^31 + (v >> 127)) >> 32` is `floor((v + 2^31) / 2^32)` for
/// `v >= 0` and `-floor((2^31 - v) / 2^32)` for `v < 0`. One `i64 x i64 -> i128` multiply: `|a| <
/// 2^34` and `|r| < 2^32` on the window, so the operands fit `i64` and the result fits `i64`.
#[inline(always)]
#[allow(
    clippy::cast_possible_truncation,
    reason = "|a * r / 2^32| < 2^34 on the exp window"
)]
fn mul_round_2p32(a: i64, r: i64) -> i64 {
    let v = i128::from(a) * i128::from(r);
    ((v + (1_i128 << 31) + (v >> 127)) >> 32) as i64
}

/// Test-only one-lane form of [`sigmoid_lanes`]: the word `fixed_sigmoid(x).unwrap()`, so the
/// oracle comparisons cover the code `sigmoid_gate` runs.
#[cfg(test)]
pub(crate) fn sigmoid_q32(x: i64) -> i64 {
    sigmoid_lanes::<1>([x])[0]
}

#[cfg(test)]
pub(crate) static REM_NEG_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
pub(crate) static REM_POS_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `floor(n_wide / den)` saturated to `u64`: the quotient estimate of `sigmoid_lanes`.
///
/// # Complexity
///
/// `O(1)`: one `u128 / u64` division (the quotient is below `2^33`, `den >= 2^32`).
#[inline(always)]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the quotient is checked against u64::MAX on the line above the cast"
)]
fn recip_est(n_wide: u128, den: u64) -> u64 {
    if den == 0 {
        u64::MAX
    } else {
        let qq = n_wide / u128::from(den);
        if qq > u128::from(u64::MAX) {
            u64::MAX
        } else {
            qq as u64
        }
    }
}

/// The Q32.32 sigmoid words of `L` inputs, each equal to `fixed_sigmoid(x).unwrap()`.
///
/// With `e = exp(-|x|)` (`0` below `-16 * ONE`) and `den = ONE + e`, the oracle's word is
/// `round(N / den)` for `N = 2^64` (`x >= 0`) or `e * 2^32` (`x < 0`), ties up, clamped to
/// `[0, ONE]`. The quotient is `floor(N / den)` computed exactly in integers (`N / den < 2^32.1`), then
/// corrected by the signed remainder, computed modulo `2^64`
/// where the true remainder lies in `[-den, 2 * den)` with `den <= 2^33`. The stages run lane by
/// lane so the `L` independent dependency chains (the Horner recurrence is 7 serial multiplies)
/// sit adjacent in the instruction stream.
///
/// # Complexity
///
/// `O(L)`: per lane one range reduction, 7 Horner steps and one `u128` division; `O(L)` stack.
#[inline(always)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    reason = "values are range-bounded as in `sigmoid_q32`; the lane loops index parallel arrays"
)]
fn sigmoid_lanes<const L: usize>(x: [i64; L]) -> [i64; L] {
    let mut t = [0i64; L];
    let mut neg = [false; L];
    let mut r = [0i64; L];
    let mut k = [0i64; L];
    let mut live = [false; L];
    for l in 0..L {
        neg[l] = x[l] < 0;
        t[l] = if neg[l] { x[l] } else { -x[l] };
        live[l] = t[l] >= LOWER;
        // A dead lane (below the window) reduces `0`, whose exp is discarded.
        let tt = if live[l] { t[l] } else { 0 };
        k[l] = ((i128::from(tt) * i128::from(LN2_RECIP_Q32) + (1_i128 << 63)) >> 64) as i64;
        r[l] = tt - k[l] * LN2_Q32;
    }
    let mut acc = [852_176i64; L];
    for c in [
        5_965_232i64,
        35_791_394,
        178_956_971,
        715_827_883,
        2_147_483_648,
        4_294_967_296,
        4_294_967_296,
    ] {
        for l in 0..L {
            acc[l] = mul_round_2p32(acc[l], r[l]) + c;
        }
    }
    let mut out = [0i64; L];
    for l in 0..L {
        let n = (-k[l]) as u32;
        let half = (1_i64 << n) >> 1;
        let e = if live[l] { (acc[l] + half) >> n } else { 0 };
        let den = (ONE + e) as u64;
        let half = den >> 1;
        let hi: u64 = if neg[l] { e as u64 } else { 1 << 32 };
        let n_wide = (u128::from(hi) << 32) + u128::from(half);
        let n_low = (hi << 32).wrapping_add(half);
        let est = recip_est(n_wide, den);
        let rem = n_low.wrapping_sub(est.wrapping_mul(den)) as i64;
        let q = if rem < 0 {
            #[cfg(test)]
            REM_NEG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            est - 1
        } else if rem >= den as i64 {
            #[cfg(test)]
            REM_POS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            est + 1
        } else {
            est
        };
        out[l] = q.min(ONE as u64) as i64;
    }
    out
}

/// `att[i] ← att[i] · sigmoid(gate[i])` through `sigmoid_lanes` and the checked Q32.32 product of
/// `mul_q32`; refusals are the oracle loop's: a width mismatch, then the first product outside
/// `i64` (none is reachable for a sigmoid in `[0, ONE]`).
///
/// # Complexity
///
/// `O(att.len())` time, no allocation.
#[allow(
    clippy::many_single_char_names,
    reason = "the lane, tail and error bindings of the chunked loop are the gate's own names"
)]
pub(crate) fn sigmoid_gate(att: &mut [i64], gate: &[i64]) -> Result<(), StepError> {
    if att.len() != gate.len() {
        return Err(StepError::BadConfig(String::from("gate width")));
    }
    let n = att.len();
    let full = n / LANES * LANES;
    let mut err: Option<StepError> = None;
    let mut base = 0usize;
    while base < full && err.is_none() {
        let mut x = [0i64; LANES];
        x.copy_from_slice(&gate[base..base + LANES]);
        let s = sigmoid_lanes::<LANES>(x);
        let mut l = 0usize;
        while l < LANES && err.is_none() {
            match gate_product(att[base + l], s[l]) {
                Ok(v) => att[base + l] = v,
                Err(e) => err = Some(e),
            }
            l += 1;
        }
        base += LANES;
    }
    let mut t = full;
    while t < n && err.is_none() {
        match gate_product(att[t], sigmoid_lanes::<1>([gate[t]])[0]) {
            Ok(v) => att[t] = v,
            Err(e) => err = Some(e),
        }
        t += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// Words evaluated together by [`sigmoid_lanes`] in [`sigmoid_gate`].
const LANES: usize = 8;

/// `mul_q32(a, s)` for a sigmoid word `s`.
#[inline(always)]
fn gate_product(a: i64, s: i64) -> Result<i64, StepError> {
    crate::engine::value::mul_q32(a, s).ok_or(StepError::Overflow)
}

/// Widest rotary half (`rot / 2`) [`rotate_pair_prefixes`] holds on the stack.
const MAX_PAIRS: usize = 128;

/// `RoPE` at position `pos` on the first `rope.head_dim()` components of every `hd`-wide head of
/// `query` and then of `key`, equal to two [`super::rotate_head_prefixes`] calls in that order.
///
/// Each rotary pair's `(sin, cos)` is one `fixed_sin_cos` of `(turns[j] · pos) mod 2^64`, shared by
/// all heads of both vectors. Refusals and their order are the two calls': a geometry mismatch of
/// `query` ([`RopeError::VectorLength`](super::RopeError::VectorLength)), then the first angle or
/// component outside its window while rotating `query`, then the same for `key`. A table wider than
/// `MAX_PAIRS` pairs takes the per-head path.
///
/// # Complexity
///
/// `O((query.len() + key.len()) / hd · rot)` multiplications and `rot / 2` `fixed_sin_cos`
/// evaluations; `O(MAX_PAIRS)` stack.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the phase is the exact product modulo 2^64, as in `RopeTable::rotate_head`"
)]
pub(crate) fn rotate_pair_prefixes(
    rt: &RopeTable,
    query: &mut [i64],
    key: &mut [i64],
    hd: usize,
    pos: u32,
) -> Result<(), StepError> {
    let rot = rt.head_dim();
    let d2 = rot / 2;
    if d2 > MAX_PAIRS {
        super::rotate_head_prefixes(rt, query, hd, pos)?;
        return super::rotate_head_prefixes(rt, key, hd, pos);
    }
    let mut table: Option<[(i64, i64); MAX_PAIRS]> = None;
    let mut err: Option<StepError> = None;
    let mut which = 0usize;
    while which < 2 && err.is_none() {
        let x: &mut [i64] = if which == 0 { &mut *query } else { &mut *key };
        if hd == 0 || rot > hd || !x.len().is_multiple_of(hd) {
            err = Some(StepError::Rope(super::RopeError::VectorLength));
        }
        let mut base = 0usize;
        while base < x.len() && err.is_none() {
            if table.is_none() {
                let mut t = [(0i64, 0i64); MAX_PAIRS];
                let mut j = 0usize;
                while j < d2 && err.is_none() {
                    let phase = (u128::from(rt.turns()[j]) * u128::from(pos)) as u64;
                    match fixed_sin_cos(turn_to_radians(phase)) {
                        Ok(sc) => t[j] = (sc.sin, sc.cos),
                        Err(_) => err = Some(StepError::Rope(super::RopeError::OutOfRange)),
                    }
                    j += 1;
                }
                if err.is_none() {
                    table = Some(t);
                }
            }
            if err.is_none() {
                match table.as_ref() {
                    None => err = Some(StepError::Rope(super::RopeError::OutOfRange)),
                    Some(t) => {
                        let head = &mut x[base..base + rot];
                        let mut j = 0usize;
                        while j < d2 && err.is_none() {
                            let (sin, cos) = t[j];
                            let (x0, x1) = (i128::from(head[j]), i128::from(head[j + d2]));
                            let re = x0 * i128::from(cos) - x1 * i128::from(sin);
                            let im = x1 * i128::from(cos) + x0 * i128::from(sin);
                            let (o0, o1) = (super::round_shift_32(re), super::round_shift_32(im));
                            match (i64::try_from(o0), i64::try_from(o1)) {
                                (Ok(o0), Ok(o1)) => {
                                    head[j] = o0;
                                    head[j + d2] = o1;
                                }
                                _ => err = Some(StepError::Rope(super::RopeError::OutOfRange)),
                            }
                            j += 1;
                        }
                    }
                }
            }
            base += hd;
        }
        which += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::unusual_byte_groupings,
    clippy::map_unwrap_or,
    reason = "test generators and timing arithmetic"
)]
mod tests {
    use super::*;
    use crate::engine::fp::F64;
    use crate::numeric::fixed_ln::fixed_sigmoid;
    use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

    fn lcg(s: &mut u64) -> u64 {
        *s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *s
    }

    fn check(x: i64) {
        assert_eq!(
            sigmoid_q32(x),
            fixed_sigmoid(x).expect("total"),
            "sigmoid word differs at x={x}"
        );
    }

    #[test]
    fn sigmoid_q32_equals_the_oracle_on_edges_and_samples() {
        for x in [
            0,
            1,
            -1,
            i64::MAX,
            i64::MIN,
            i64::MIN + 1,
            LOWER,
            LOWER - 1,
            LOWER + 1,
            -LOWER,
            -LOWER + 1,
            -LOWER - 1,
            ONE,
            -ONE,
        ] {
            check(x);
        }
        let mut s = 0x5151_5eed_u64;
        for _ in 0..400_000 {
            let r = lcg(&mut s);
            // Magnitudes from 2^-32 up to 2^40 (past both saturation bounds), both signs.
            let bits = (r >> 58) as u32 % 41 + 1;
            let m = (lcg(&mut s) >> (64 - bits)) as i64;
            check(if r & 1 == 0 { m } else { -m });
        }
        let mut x = -20 * ONE;
        while x <= 20 * ONE {
            check(x);
            x += ONE / 4093;
        }
    }

    /// Dense sweep of the whole non-saturated window plus margins against the oracle, 16 threads.
    #[test]
    #[ignore = "dense sweep (about 1.7e9 oracle calls): run in release"]
    fn sigmoid_q32_equals_the_oracle_on_a_dense_sweep() {
        let lo = -18 * ONE;
        let hi = 18 * ONE;
        let step = 97i64;
        let threads = 16i64;
        std::thread::scope(|sc| {
            for t in 0..threads {
                sc.spawn(move || {
                    let mut x = lo + t * step;
                    while x <= hi {
                        check(x);
                        x += threads * step;
                    }
                });
            }
        });
    }

    /// Single-thread ns per word of the gate forms over gate inputs spread as in a real layer.
    #[test]
    #[ignore = "timing harness: run under the exclusive host lock"]
    fn sigmoid_gate_micro() {
        let n = 1 << 16;
        let mut s = 0x1234_u64;
        let gate: Vec<i64> = (0..n)
            .map(|_| (lcg(&mut s) >> 27) as i64 - (1 << 36))
            .collect();
        let att0: Vec<i64> = (0..n)
            .map(|_| (lcg(&mut s) >> 22) as i64 - (1 << 41))
            .collect();
        let time = |name: &str, f: &dyn Fn(&mut [i64])| {
            let mut best = f64::MAX;
            for _ in 0..30 {
                let mut a = att0.clone();
                let t0 = std::time::Instant::now();
                f(&mut a);
                std::hint::black_box(&a);
                best = best.min(t0.elapsed().as_secs_f64());
            }
            println!("{name:12} {:.2} ns/word", best * 1e9 / n as f64);
        };
        time("oracle", &|a| {
            for i in 0..n {
                let g = fixed_sigmoid(gate[i]).unwrap();
                a[i] = crate::engine::value::mul_q32(a[i], g).unwrap();
            }
        });
        time("twin", &|a| sigmoid_gate(a, &gate).unwrap());
    }

    #[test]
    fn sigmoid_gate_equals_the_oracle_loop_and_keeps_its_width_refusal() {
        let mut s = 0xfeed_u64;
        let n = 1024;
        let att0: Vec<i64> = (0..n)
            .map(|_| (lcg(&mut s) >> 20) as i64 - (1 << 43))
            .collect();
        let gate: Vec<i64> = (0..n)
            .map(|_| (lcg(&mut s) >> 26) as i64 - (1 << 37))
            .collect();
        let mut want = att0.clone();
        for i in 0..n {
            let g = fixed_sigmoid(gate[i]).unwrap();
            want[i] = crate::engine::value::mul_q32(want[i], g).unwrap();
        }
        let mut got = att0.clone();
        sigmoid_gate(&mut got, &gate).unwrap();
        assert_eq!(got, want);
        assert!(matches!(
            sigmoid_gate(&mut got, &gate[1..]),
            Err(StepError::BadConfig(_))
        ));
        let mut ext = vec![i64::MAX, i64::MIN, 1, -1];
        let g = vec![i64::MAX, i64::MAX, i64::MIN, 0];
        let mut w = ext.clone();
        for i in 0..4 {
            w[i] = crate::engine::value::mul_q32(w[i], fixed_sigmoid(g[i]).unwrap()).unwrap();
        }
        sigmoid_gate(&mut ext, &g).unwrap();
        assert_eq!(ext, w);
    }

    #[test]
    fn rotate_pair_prefixes_equals_two_rotate_head_prefixes() {
        let hd = 256usize;
        for rot in [64usize, 256, 8] {
            let rope = RopeTable::new(F64::from_f64(1.0e7), rot).unwrap();
            let mut s = 0xc0de_u64 + rot as u64;
            for (nq, nk, pos) in [
                (8usize, 2usize, 0u32),
                (8, 2, 1),
                (8, 2, 199),
                (8, 2, 131_071),
                (1, 1, 7),
                (0, 2, 5),
                (2, 0, 5),
            ] {
                let mk = |n: usize, s: &mut u64| -> Vec<i64> {
                    (0..n * hd)
                        .map(|_| (lcg(s) >> 22) as i64 - (1 << 41))
                        .collect()
                };
                let (q0, k0) = (mk(nq, &mut s), mk(nk, &mut s));
                let (mut q1, mut k1) = (q0.clone(), k0.clone());
                super::super::rotate_head_prefixes(&rope, &mut q1, hd, pos).unwrap();
                super::super::rotate_head_prefixes(&rope, &mut k1, hd, pos).unwrap();
                let (mut q2, mut k2) = (q0.clone(), k0.clone());
                rotate_pair_prefixes(&rope, &mut q2, &mut k2, hd, pos).unwrap();
                assert_eq!(q1, q2, "query rot {rot} pos {pos}");
                assert_eq!(k1, k2, "key rot {rot} pos {pos}");
            }
        }
    }

    #[test]
    fn rotate_pair_prefixes_keeps_the_geometry_and_range_refusals() {
        let rope = RopeTable::new(F64::from_f64(1.0e7), 64).unwrap();
        let mut q = vec![0i64; 300];
        let mut k = vec![0i64; 256];
        assert!(matches!(
            rotate_pair_prefixes(&rope, &mut q, &mut k, 256, 3),
            Err(StepError::Rope(super::super::RopeError::VectorLength))
        ));
        let mut q = vec![0i64; 256];
        let mut k = vec![0i64; 300];
        assert!(matches!(
            rotate_pair_prefixes(&rope, &mut q, &mut k, 256, 3),
            Err(StepError::Rope(super::super::RopeError::VectorLength))
        ));
        let mut q = vec![i64::MAX; 256];
        let mut k = vec![0i64; 256];
        let want = super::super::rotate_head_prefixes(&rope, &mut q.clone(), 256, 0);
        let got = rotate_pair_prefixes(&rope, &mut q, &mut k, 256, 0);
        assert_eq!(want.is_err(), got.is_err());
        let mut q = vec![0i64; 256];
        let mut k = vec![i64::MAX; 256];
        let mut q_w = q.clone();
        let want = super::super::rotate_head_prefixes(&rope, &mut q_w, 256, 1)
            .and_then(|()| super::super::rotate_head_prefixes(&rope, &mut k.clone(), 256, 1));
        let got = rotate_pair_prefixes(&rope, &mut q, &mut k, 256, 1);
        assert_eq!(format!("{want:?}"), format!("{got:?}"));
    }

    /// The first mismatch of one sigmoid slice task, as atomics the pool task stores into.
    struct FirstMis {
        x: AtomicI64,
        got: AtomicI64,
        want: AtomicI64,
    }

    /// The shared borrows of the exhaustive sigmoid dispatch.
    struct SigmoidSliceCtx<'a> {
        slice_start: i64,
        slice_end: i64,
        slice_len: u64,
        tasks: usize,
        chunk_mismatches: &'a [AtomicU64],
        first_mis: &'a [FirstMis],
    }

    impl crate::engine::chart::task::IndexTask for SigmoidSliceCtx<'_> {
        fn run(&self, t: usize) {
            sigmoid_slice_task(self, t);
        }
    }

    /// Compares `sigmoid_q32` with `fixed_sigmoid` over task `t`'s share of the slice and records
    /// the mismatch count and the lowest mismatching word in the task's own slots.
    fn sigmoid_slice_task(c: &SigmoidSliceCtx<'_>, t: usize) {
        let (tasks, slice_len) = (c.tasks, c.slice_len);
        let t_start: i64 = c.slice_start + (t as u128 * slice_len as u128 / tasks as u128) as i64;
        let t_end: i64 = if t + 1 == tasks {
            c.slice_end
        } else {
            c.slice_start + ((t + 1) as u128 * slice_len as u128 / tasks as u128) as i64
        };

        let mut local_mismatches = 0u64;
        let mut local_first = None;

        for x in t_start..t_end {
            let got = sigmoid_q32(x);
            let want = fixed_sigmoid(x).expect("total on domain");
            if got != want {
                local_mismatches += 1;
                if local_first.is_none() {
                    local_first = Some((x, got, want));
                }
            }
        }

        if local_mismatches > 0 {
            c.chunk_mismatches[t].store(local_mismatches, Ordering::Relaxed);
            if let Some((x, got, want)) = local_first {
                c.first_mis[t].x.store(x, Ordering::Relaxed);
                c.first_mis[t].got.store(got, Ordering::Relaxed);
                c.first_mis[t].want.store(want, Ordering::Relaxed);
            }
        }
    }

    /// Exhaustive equivalence check of `sigmoid_q32(x) == fixed_sigmoid(x)` for every `x` in the
    /// non-saturating set `[-16 * ONE, 16 * ONE]`, split over `Pool::global()`.
    ///
    /// The input domain is `[-68_719_476_736, 68_719_476_736]` (137,438,953,473 points).
    /// Outside this range, `fixed_sigmoid` and `sigmoid_q32` are proven identical constant saturations:
    /// `0` for `x < -16 * ONE` and `ONE` for `x > 16 * ONE`.
    ///
    /// Slice selection:
    /// - `APEIRON_SIGMOID_SLICE="1%"` or `"measure"`: runs a 1% slice (1,374,389,535 words) to measure throughput.
    /// - `APEIRON_SIGMOID_SLICE="i/n"`: runs slice `i` of `n` (0-indexed or 1-indexed).
    /// - Not set: runs the full domain (`0/1`).
    #[test]
    #[ignore = "exhaustive equivalence over non-saturating domain [-16*ONE, 16*ONE]: run in release"]
    fn sigmoid_q32_equals_fixed_sigmoid_exhaustive() {
        // Reset branch counters for this run
        REM_NEG_COUNT.store(0, Ordering::Relaxed);
        REM_POS_COUNT.store(0, Ordering::Relaxed);

        let full_lo = LOWER; // -16 * ONE = -68_719_476_736
        let full_hi = -LOWER; // 16 * ONE = 68_719_476_736
                              // Total points = 32 * ONE + 1 = 137_438_953_473
        let full_len: u128 = (full_hi as i128 - full_lo as i128 + 1) as u128;

        let slice_env = std::env::var("APEIRON_SIGMOID_SLICE").unwrap_or_default();
        let (slice_idx, slice_count, is_measure_1pct) =
            if slice_env == "1%" || slice_env == "measure" {
                (0usize, 100usize, true)
            } else if let Some((idx_str, n_str)) = slice_env.split_once('/') {
                let mut i: usize = idx_str.trim().parse().expect("valid slice idx");
                let n: usize = n_str.trim().parse().expect("valid slice count");
                assert!(n > 0, "slice count must be > 0");
                if i == n {
                    // If 1-based indexing used for the final slice (e.g. 4/4)
                    i = n - 1;
                } else {
                    assert!(i < n, "slice idx must be < n (0-indexed: 0..n)");
                }
                (i, n, false)
            } else {
                (0, 1, false)
            };

        let slice_start: i64 =
            full_lo + (slice_idx as u128 * full_len / slice_count as u128) as i64;
        let slice_end: i64 = if slice_idx + 1 == slice_count {
            full_hi + 1
        } else {
            full_lo + ((slice_idx + 1) as u128 * full_len / slice_count as u128) as i64
        };
        let slice_len: u64 = (slice_end as i128 - slice_start as i128) as u64;

        println!(
            "Running slice {slice_idx}/{slice_count} on [{slice_start}, {slice_end}) ({slice_len} words)"
        );

        let tasks = 256usize;

        let chunk_mismatches: Vec<AtomicU64> = (0..tasks).map(|_| AtomicU64::new(0)).collect();
        let first_mis: Vec<FirstMis> = (0..tasks)
            .map(|_| FirstMis {
                x: AtomicI64::new(i64::MAX),
                got: AtomicI64::new(0),
                want: AtomicI64::new(0),
            })
            .collect();

        let t0 = std::time::Instant::now();

        let ctx = SigmoidSliceCtx {
            slice_start,
            slice_end,
            slice_len,
            tasks,
            chunk_mismatches: &chunk_mismatches,
            first_mis: &first_mis,
        };
        crate::engine::chart::pool::run_global(tasks, &ctx);

        let elapsed = t0.elapsed();
        let secs = elapsed.as_secs_f64();
        let rate = slice_len as f64 / secs;

        let mut total_mismatches = 0u64;
        let mut overall_first: Option<(i64, i64, i64)> = None;

        for t in 0..tasks {
            let m = chunk_mismatches[t].load(Ordering::Relaxed);
            total_mismatches += m;
            if m > 0 {
                let x = first_mis[t].x.load(Ordering::Relaxed);
                let got = first_mis[t].got.load(Ordering::Relaxed);
                let want = first_mis[t].want.load(Ordering::Relaxed);
                match overall_first {
                    None => overall_first = Some((x, got, want)),
                    Some((prev_x, _, _)) if x < prev_x => overall_first = Some((x, got, want)),
                    _ => {}
                }
            }
        }

        let rem_neg = REM_NEG_COUNT.load(Ordering::Relaxed);
        let rem_pos = REM_POS_COUNT.load(Ordering::Relaxed);
        let rem_zero = slice_len - rem_neg - rem_pos;

        println!(
            "Slice {slice_idx}/{slice_count}: elapsed={secs:.2}s ({rate:.2e} words/s), mismatches={total_mismatches}"
        );
        println!(
            "Remainder branches: rem < 0 (neg) = {rem_neg}, rem >= den (pos) = {rem_pos}, 0 <= rem < den = {rem_zero}"
        );

        if is_measure_1pct {
            let projected_total_s = secs * 100.0;
            println!(
                "1% measure complete. Projected total wall time for 100% (137.4B words): {projected_total_s:.1}s"
            );
        }

        assert_eq!(
            total_mismatches,
            0,
            "mismatch found! total={total_mismatches}, first at x={:?}: got={}, want={}",
            overall_first.map(|f| f.0),
            overall_first.map(|f| f.1).unwrap_or(0),
            overall_first.map(|f| f.2).unwrap_or(0)
        );

        // gate_product equivalence check:
        // Status: SAMPLED (att x gate domain is 2^128, sampled over dense random inputs plus i64 extremes).
        // Checks that sigmoid_gate(att, gate) matches fixed_sigmoid then mul_q32 on all cases.
        println!("Checking gate_product against oracle loop on dense sample including extremes...");
        let mut s = 0x5160_1d_u64;
        let mut lcg = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            s
        };

        let mut sample_att = vec![
            0i64,
            1,
            -1,
            2,
            -2,
            i64::MAX,
            i64::MAX - 1,
            i64::MIN,
            i64::MIN + 1,
            ONE,
            -ONE,
            2 * ONE,
            -2 * ONE,
            LOWER,
            LOWER - 1,
            LOWER + 1,
            -LOWER,
            -LOWER - 1,
            -LOWER + 1,
        ];
        // 100_000 random att values
        for _ in 0..100_000 {
            let r = lcg();
            let bits = (r >> 58) as u32 % 64 + 1;
            let m = (lcg() >> (64 - bits)) as i64;
            sample_att.push(if r & 1 == 0 { m } else { -m });
        }

        let mut sample_gate = Vec::with_capacity(sample_att.len());
        for i in 0..sample_att.len() {
            if i < 19 {
                sample_gate.push(sample_att[18 - i]);
            } else {
                let r = lcg();
                let bits = (r >> 58) as u32 % 42 + 1;
                let m = (lcg() >> (64 - bits)) as i64;
                sample_gate.push(if r & 1 == 0 { m } else { -m });
            }
        }

        let mut want_att = sample_att.clone();
        for i in 0..sample_att.len() {
            let g = fixed_sigmoid(sample_gate[i]).unwrap();
            want_att[i] = crate::engine::value::mul_q32(want_att[i], g).unwrap();
        }

        let mut got_att = sample_att.clone();
        sigmoid_gate(&mut got_att, &sample_gate).unwrap();
        assert_eq!(got_att, want_att, "gate_product dense sample mismatch");
        println!(
            "gate_product dense sample passed ({} samples)",
            sample_att.len()
        );
    }

    /// Control for the exact quotient: the `f64` estimate that `recip_est` replaced, copied from
    /// the pre-change body.
    fn old_recip(n_wide: u128, den: u64) -> u64 {
        (n_wide as f64 / den as f64) as u64
    }

    /// The lane tail of `sigmoid_lanes` after the estimate, with the estimator as a parameter.
    /// Returns the output word and whether the negative / positive correction fired.
    fn lane_tail(
        k: i64,
        acc: i64,
        live: bool,
        neg: bool,
        est_of: fn(u128, u64) -> u64,
    ) -> (i64, bool, bool) {
        let n = (-k) as u32;
        let half = (1_i64 << n) >> 1;
        let e = if live { (acc + half) >> n } else { 0 };
        let den = (ONE + e) as u64;
        let half = den >> 1;
        let hi: u64 = if neg { e as u64 } else { 1 << 32 };
        let n_wide = (u128::from(hi) << 32) + u128::from(half);
        let n_low = (hi << 32).wrapping_add(half);
        let est = est_of(n_wide, den);
        let rem = n_low.wrapping_sub(est.wrapping_mul(den)) as i64;
        let (q, under, over) = if rem < 0 {
            (est - 1, true, false)
        } else if rem >= den as i64 {
            (est + 1, false, true)
        } else {
            (est, false, false)
        };
        (q.min(ONE as u64) as i64, under, over)
    }

    /// `recip_est` against `old_recip` over 1M lane inputs in the accepted domain (`k` in
    /// `[-23, 0]`, `e <= ONE`) plus edge inputs. The outputs must be identical, and the exact
    /// estimate must never trigger the correction.
    #[test]
    fn recip_est_exact_matches_f64_estimate_over_lanes() {
        let mut s = 0xdec0_de_u64;
        let (mut mismatches, mut old_fix, mut new_neg, mut new_pos, mut cases) =
            (0u64, 0u64, 0u64, 0u64, 0u64);
        let mut check = |k: i64, acc: i64, live: bool, neg: bool| {
            let (o, ou, oo) = lane_tail(k, acc, live, neg, old_recip);
            let (w, nu, no) = lane_tail(k, acc, live, neg, recip_est);
            cases += 1;
            if o != w {
                mismatches += 1;
                assert!(
                    mismatches < 4,
                    "k={k} acc={acc} live={live} neg={neg}: old {o} new {w}"
                );
            }
            old_fix += u64::from(ou || oo);
            new_neg += u64::from(nu);
            new_pos += u64::from(no);
        };
        for _ in 0..1_000_000 {
            let r = lcg(&mut s);
            let n = (r >> 59) % 24;
            let live = (r >> 58) & 1 == 1;
            let neg = (r >> 57) & 1 == 1;
            let acc = ((lcg(&mut s) >> 1) % (1_u64 << (32 + n))) as i64;
            check(-(n as i64), acc, live, neg);
        }
        for n in 0..24_u32 {
            let top = (1_i64 << 32) << n;
            for acc in [0, 1, top - 1, top, top + 1] {
                for live in [false, true] {
                    for neg in [false, true] {
                        check(-i64::from(n), acc, live, neg);
                    }
                }
            }
        }
        assert_eq!(mismatches, 0, "estimate outputs differ");
        assert_eq!(
            new_neg + new_pos,
            0,
            "exact estimate triggered a correction"
        );
        eprintln!(
            "recip_est control: {cases} lane cases, mismatches {mismatches}, \
             old-estimate corrections {old_fix}, new REM_NEG {new_neg}, new REM_POS {new_pos}"
        );
    }
}
