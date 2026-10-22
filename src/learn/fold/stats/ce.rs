//! Softmax, sampled-label cross entropy and the inverse-CDF target draw of the held-out capture,
//! split out of [`super::hocapture`] to hold the file under the tree's line limit. Every public item
//! is re-exported from `hocapture`, so its paths do not move.

#![allow(
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    reason = "moved unchanged from `hocapture`: the single-letter names are the softmax and sampler \
              symbols, and the `u64 as i64` narrows a Q32.32 sampler word clamped below `ONE`"
)]

use super::hocapture::{expect, shape, HocaptureError, FLUSH, SEED_BIAS};
use crate::engine::ONE;
use crate::learn::grad::head;
use crate::learn::solve::q32::{rshift_rne, Fx};
use crate::numeric::fixed_exp::fixed_exp;

/// `n / d` rounded to nearest, ties away from zero; `d > 0`. `q32` has no general divide-and-round,
/// so this is the file's one such rule (it is `grad::head`'s private one).
pub(super) fn round_div(n: i128, d: i128) -> i128 {
    let (q, r) = (n / d, n % d);
    let mag = if r < 0 { 0 - r } else { r };
    if 2 * mag >= d {
        if n >= 0 {
            q + 1
        } else {
            q - 1
        }
    } else {
        q
    }
}

/// `x` narrowed to `i64`, or [`HocaptureError::Overflow`].
pub(super) fn to64(x: i128) -> Result<i64, HocaptureError> {
    i64::try_from(x).map_err(|_| HocaptureError::Overflow)
}

/// `a + b` in the [`Fx`] carrier: the operand with the larger exponent is kept, the other is
/// right-shifted by the exponent gap (round to nearest, ties to even) and the significands are added
/// in `i128`, renormalised once by [`Fx::from_i128`]. A gap of 63 bits or more drops the smaller
/// term, whose magnitude is then under the larger significand's last place.
///
/// # Errors
/// [`HocaptureError::Overflow`] when a checked exponent difference, shift or add leaves its range
/// (unreachable for accumulators built from Q32.32 products; kept for totality).
pub(super) fn fx_add(a: &Fx, b: &Fx) -> Result<Fx, HocaptureError> {
    if a.m == 0 || b.m == 0 {
        return Ok(if a.m == 0 { b.clone() } else { a.clone() });
    }
    let (hi, lo) = if a.e >= b.e { (a, b) } else { (b, a) };
    let gap = hi.e.checked_sub(lo.e).ok_or(HocaptureError::Overflow)?;
    if gap > 62 {
        return Ok(hi.clone());
    }
    let shift = u32::try_from(gap).map_err(|_| HocaptureError::Overflow)?;
    let shifted = rshift_rne(i128::from(lo.m), shift).ok_or(HocaptureError::Overflow)?;
    let sum = i128::from(hi.m)
        .checked_add(shifted)
        .ok_or(HocaptureError::Overflow)?;
    Fx::from_i128(sum, hi.e).map_err(|_| HocaptureError::Overflow)
}

/// Softmax of one Q32.32 logit row, `pᵢ = round(eᵢ·ONE/S)`.
///
/// The rule is [`head`]'s private `softmax_row` (`src/learn/grad/head.rs:279-304`): shift by the row
/// maximum, `eᵢ = fixed_exp(uᵢ)` with `uᵢ < −16·ONE` flushed to `0`, `S = Σeᵢ`, one rounding per
/// `pᵢ`. It is mirrored here over the same public [`fixed_exp`] because `head`'s function is private
/// and this task does not own `head.rs`; the gate pins the two through `kl_row`
/// (`softmax_matches_head`).
///
/// # Errors
/// [`HocaptureError::Shape`] for an empty row, or when [`fixed_exp`] refuses the shifted value
/// (unreachable: every shifted value is `≤ 0`); [`HocaptureError::Overflow`] when a sum or a
/// probability leaves `i64`.
///
/// # Complexity
/// `O(v)`: `v` comparisons, `v` [`fixed_exp`] calls and `v` `i128` divisions; `O(v)` words out.
pub fn softmax_row(z: &[i64]) -> Result<Vec<i64>, HocaptureError> {
    let v = z.len();
    if v == 0 {
        return Err(shape("empty softmax row"));
    }
    let mut m = z[0];
    for &zi in z {
        if zi > m {
            m = zi;
        }
    }
    let mut e = vec![0i64; v];
    let mut sum: i128 = 0;
    // One flat pass: the fragment carries no `continue` and no `?` inside a loop body, so
    // `softmax_term` holds the flush edge and the two refusals and each loop carries its exit.
    let mut err: Option<HocaptureError> = None;
    for i in 0..v {
        if err.is_none() {
            let u = i128::from(z[i]) - i128::from(m);
            match softmax_term(u) {
                Ok(ev) => {
                    e[i] = ev;
                    sum += i128::from(ev);
                }
                Err(e) => err = Some(e),
            }
        }
    }
    if let Some(e) = err {
        return Err(e);
    }
    let mut p = vec![0i64; v];
    let one = i128::from(ONE);
    let mut err = None;
    for i in 0..v {
        if err.is_none() {
            match to64(round_div(i128::from(e[i]) * one, sum)) {
                Ok(pi) => p[i] = pi,
                Err(e) => err = Some(e),
            }
        }
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(p)
}

/// `exp(z_i − m)` for one shifted Q32.32 logit, zero below the [`FLUSH`] edge. `O(1)`;
/// [`HocaptureError::Overflow`] when the shifted word leaves `i64`, the shape refusal when
/// [`fixed_exp`] refuses.
fn softmax_term(u: i128) -> Result<i64, HocaptureError> {
    if u < FLUSH {
        return Ok(0);
    }
    let zi = to64(u)?;
    fixed_exp(zi).map_err(|_| shape("fixed_exp refused"))
}

/// The chunked head's `Σ_t log p_t` read at the sampled targets, negated: the sampled-label
/// cross-entropy the taps differentiate, in Q32.32.
///
/// This is `head::log_softmax_row` per row, summed in exact `i128` and rounded once, so the gate's
/// central difference does not go through [`softmax_row`]'s own rounding.
///
/// # Errors
/// [`HocaptureError::Shape`] when `logits.len() != t·vocab`, `targets.len() != t`, or a target is
/// past the vocabulary; [`HocaptureError::Overflow`] on a checked sum or a narrowing.
///
/// # Complexity
/// `O(t·vocab)`; one softmax per row.
pub fn cross_entropy_sum(
    logits: &[i64],
    t: usize,
    vocab: usize,
    targets: &[u32],
) -> Result<i64, HocaptureError> {
    expect(t, vocab, logits.len(), "cross_entropy logits")?;
    if targets.len() != t {
        return Err(shape("cross_entropy targets"));
    }
    let mut acc: i128 = 0;
    for r in 0..t {
        let id = usize::try_from(targets[r]).map_err(|_| HocaptureError::Overflow)?;
        if id >= vocab {
            return Err(shape("cross_entropy target >= vocab"));
        }
        let lp = head::log_softmax_row(&logits[r * vocab..(r + 1) * vocab])
            .map_err(|_| shape("log_softmax_row refused"))?;
        acc -= i128::from(lp[id]);
    }
    to64(round_div(acc, 1))
}

/// `p_i − [i = s_r]` for row `r`.
///
/// # Errors
/// `Shape` for a target past the vocabulary, `Overflow` from the narrowing.
///
/// # Complexity
/// `O(vocab)`: one [`softmax_row`] and one write per entry.
fn grad_row(
    logits: &[i64],
    targets: &[u32],
    vocab: usize,
    r: usize,
) -> Result<Vec<i64>, HocaptureError> {
    let target = *targets
        .get(r)
        .ok_or_else(|| shape("logits_grad_ce target"))?;
    let id = usize::try_from(target).map_err(|_| HocaptureError::Overflow)?;
    if id >= vocab {
        return Err(shape("logits_grad_ce target >= vocab"));
    }
    let p = softmax_row(&logits[r * vocab..(r + 1) * vocab])?;
    let mut out = p;
    out[id] = out[id].checked_sub(ONE).ok_or(HocaptureError::Overflow)?;
    Ok(out)
}

/// The logit gradient of the sampled-label loss `Σ_t CE(z_t, s_t)`: `out[t·vocab + i] = p_ti −
/// [i = s_t]`, in Q32.32.
///
/// # Errors
/// [`HocaptureError::Shape`] when `logits.len() != t·vocab`, `out.len() != t·vocab`, `targets.len()
/// != t`, or a target is past the vocabulary.
///
/// # Complexity
/// Time `O(t·vocab)`, once per draw (`K` per capture): a per-token vector loop, not a matrix
/// pass, so it stays serial and the Gram below takes the file's pool budget.
pub fn logits_grad_ce(
    logits: &[i64],
    targets: &[u32],
    vocab: usize,
    out: &mut [i64],
) -> Result<(), HocaptureError> {
    let t = targets.len();
    expect(t, vocab, logits.len(), "logits_grad_ce logits")?;
    expect(t, vocab, out.len(), "logits_grad_ce out")?;
    if vocab == 0 || t == 0 {
        return Err(shape("logits_grad_ce dimensions"));
    }
    for r in 0..t {
        let row = grad_row(logits, targets, vocab, r)?;
        out[r * vocab..(r + 1) * vocab].copy_from_slice(&row);
    }
    Ok(())
}

/// One xorshift64\* step, returning the uniform `u ∈ [0, ONE)` of its 53 high bits.
///
/// `u = (r mod 2⁵³)/2⁵³` in Q32.32 is `u·2³² = (r >> 32)` truncated; the +`2²⁰` before the shift
/// rounds to nearest instead, and the clamp keeps the word strictly below `ONE` so the inverse-CDF
/// walk can still pass a row whose probabilities sum to `ONE`.
fn xorshift64s(state: &mut u64) -> i64 {
    let mut s = if *state == 0 { SEED_BIAS } else { *state };
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    *state = s;
    let r = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
    let word = ((r >> 11).wrapping_add(1 << 20)) >> 21;
    let cap = (ONE as u64) - 1;
    let w = if word > cap { cap } else { word };
    w as i64
}

/// `K` sampled targets per token from the teacher softmax of `logits`, draw-major then token-major:
/// `out[j·t + i]` is draw `j`'s target at token `i`.
///
/// The distribution is the reference's (`torch.multinomial(probs, 1)` per draw over the `N·T`
/// rows, `fold_transfer.py:1260`); the stream is this file's inline xorshift64\* from `seed`. The
/// inverse-CDF walk reads the same Q32.32 softmax row the loss gradient uses, in exact `i128`, and
/// falls back to the last index with non-zero probability when the uniform lands past the row's
/// rounding residue.
///
/// # Errors
/// [`HocaptureError::Shape`] when `K == 0`, `vocab == 0`, `logits.len() != t·vocab`, or `t == 0`.
///
/// # Complexity
/// Time `O(K·t·vocab)`: one [`softmax_row`] per token (a serial per-token loop) plus one
/// inverse-CDF walk per draw; space `O(t·vocab)` softmax words plus `O(K·t)` output.
pub fn sample_targets(
    logits: &[i64],
    t: usize,
    vocab: usize,
    k: usize,
    seed: u64,
) -> Result<Vec<u32>, HocaptureError> {
    if k == 0 || vocab == 0 || t == 0 {
        return Err(shape("sample_targets dimensions"));
    }
    expect(t, vocab, logits.len(), "sample_targets logits")?;
    let mut rows = vec![0i64; t * vocab];
    for r in 0..t {
        let p = softmax_row(&logits[r * vocab..(r + 1) * vocab])?;
        rows[r * vocab..(r + 1) * vocab].copy_from_slice(&p);
    }
    let mut state = if seed == 0 { SEED_BIAS } else { seed };
    let mut out = vec![0u32; k * t];
    // Flag form: the inverse-CDF walk is its own function (no `break` in a nested loop) and the
    // narrowing refusal is carried out of the draw loop instead of returned inside it.
    let mut failed = false;
    for j in 0..k {
        for r in 0..t {
            if !failed {
                let p = &rows[r * vocab..(r + 1) * vocab];
                let draw = i128::from(xorshift64s(&mut state));
                let pick = pick_from_row(p, draw);
                match u32::try_from(pick) {
                    Ok(v) => out[j * t + r] = v,
                    Err(_) => failed = true,
                }
            }
        }
    }
    if failed {
        return Err(HocaptureError::Overflow);
    }
    Ok(out)
}

/// The inverse-CDF draw on one Q32.32 row: the first index whose cumulative probability passes
/// `draw`, else the last index of positive probability. `O(v)` comparisons; space `O(1)`.
fn pick_from_row(p: &[i64], draw: i128) -> usize {
    let mut cum: i128 = 0;
    let mut pick = p.len() - 1;
    let mut found = false;
    let mut i = 0;
    while i < p.len() && !found {
        cum += i128::from(p[i]);
        if draw < cum {
            pick = i;
            found = true;
        }
        i += 1;
    }
    if !found {
        for i in 0..p.len() {
            if p[i] > 0 {
                pick = i;
            }
        }
    }
    pick
}
