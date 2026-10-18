//! Damped Newton-Fisher descent on the final RMSNorm gain `g`, minimising readout KL.
//!
//! Port of `fold_transfer.py` lines 980–1027 (`fnorm_fit`) onto the Q32.32 integer carrier: every
//! value is a `round(x·2³²)` word, products are `i128` rounded once by
//! [`crate::learn::solve::q32::round_q32_even`], the binary16 step round is
//! [`crate::learn::solve::q32::round_f16`], and the softmax is
//! [`crate::learn::fold::stats::hocapture::softmax_row`] (the crate's `fixed_exp` per shifted
//! logit, one rounding per probability). The function takes and returns words: the caller converts
//! its `f64` inputs at its own boundary, once.
//!
//! `ht`/`hs` are the teacher/student final hidden states (`n·t·hid`, row-major), `g0` the fixed
//! teacher gain (`hid`), `e` the tied embedding (`vocab·hid`), `eps` the RMSNorm epsilon; `steps`,
//! `ridge`, `lr` are the schedule and `coh_prior` scores positions `1..T` from the coh-prior ratio.
//! Every scored row is RMS-normalized, the teacher and student logit rows softmaxed, and the step
//! takes `g = f16(g − lr·(grad + reg·(g − g_target)) / (fisher + reg))` with
//! `grad_j = Σ_tokens u_s,j·((ps − pt)·E)_j` and
//! `fisher_j = Σ_tokens u_s,j²·clamp((ps·E²)_j − (ps·E)_j², ≥ 1e-6)`, both divided by
//! `m = n·(T − [coh_prior])`.
//!
//! Every loop here is one row, one token or one `hid`/`vocab` vector: at most ~10⁴ word operations
//! per call at the production shapes (`hid = 896`, `vocab = 10²–10⁵`), below a pool dispatch, so
//! all three stay serial. Time `O(steps·n·T·vocab·hid)`, space `O(hid + vocab)`.
//!
//! # Partiality
//!
//! Refuses a shape disagreement or a zero dimension ([`FnormError::ShapeMismatch`]), an `eps` word
//! that is not positive ([`FnormError::BadEps`]) and a checked quantity out of range
//! ([`FnormError::Overflow`]); the Python propagates `NaN` and wraps. A `0/0` coh-prior ratio is
//! `0`: the integer path has no `NaN`. A non-finite input is refused by the caller's
//! `to_grid` boundary, not here.
//!
//! # Grade
//!
//! `tested` (`tests/learn_fold_fnorm.rs`); no extraction or refinement links this body to a Lean
//! theorem, so nothing here is `proved`.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::doc_markdown,
    reason = "the body indexes the row-major `[n, t, hid]`, `[vocab, hid]` and `[hid]` buffers by \
              `s/p/j/k` and names the carriers `ut/us`, `pt/ps`, `mu`, `sec` as the reference does; \
              the `usize`/`i64` casts are the shape counts, already fixed by the shape check"
)]

use core::cmp::Ordering;

use crate::learn::fold::stats::hocapture::softmax_row;
use crate::learn::solve::q32::{round_f16, round_q32_even, rshift_rne, Fx};
use crate::numeric::fixed_rsqrt::isqrt;

/// The Q32.32 unit `1.0`.
const ONE: i64 = 1 << 32;

/// `0.5` as a Q32.32 word: the `alpha` floor of the coh-prior target.
const HALF_W: i64 = ONE / 2;

/// `1e-6` as a Q32.32 word, `round(1e-6 · 2³²)`: the Fisher and variance floors.
const MIN_W: i64 = 4_295;

/// A refusal from the RMSNorm-gain fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FnormError {
    /// A slice length that is not the declared shape, or a zero dimension (`n`, `t`, `hid`,
    /// `vocab`, or the `coh_prior` scored-position count).
    ShapeMismatch,
    /// `eps` is not a positive Q32.32 word, so `rsqrt(mean(h²) + eps)` is undefined.
    BadEps,
    /// A finite input outside the Q32.32 grid range, or a checked integer quantity that left its
    /// range.
    Overflow,
}

/// `round(num / den)` to nearest, ties to even; `None` when `den == 0` or the quotient leaves
/// `i64`. `O(1)`.
fn div_int(num: i128, den: i128) -> Option<i64> {
    if den == 0 {
        return None;
    }
    let mut q = num / den;
    let r = num % den;
    let twice = r.unsigned_abs().checked_mul(2)?;
    let up = match twice.cmp(&den.unsigned_abs()) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => q % 2 != 0,
    };
    if up {
        q = if r > 0 {
            q.checked_add(1)?
        } else {
            q.checked_sub(1)?
        };
    }
    i64::try_from(q).ok()
}

/// `a·b` as a Q32.32 word, one rounding to nearest, ties to even. `O(1)`.
fn mul_q(a: i64, b: i64) -> Result<i64, FnormError> {
    round_q32_even(i128::from(a) * i128::from(b)).map_err(|_| FnormError::Overflow)
}

/// `a/b` as a Q32.32 word, one rounding to nearest, ties to even. `O(1)`.
fn div_q(a: i64, b: i64) -> Result<i64, FnormError> {
    div_int(i128::from(a) << 32, i128::from(b)).ok_or(FnormError::Overflow)
}

/// `√z` as a Q32.32 word for a word `z ≥ 0`, rounded to nearest: the exact integer root of
/// `z·2³²` ([`isqrt`]), incremented when the remainder exceeds it (the midpoint `r² + r + ½` is
/// not an integer). `O(1)`, at most 64 two-bit steps.
fn sqrt_q(z: i64) -> Result<i64, FnormError> {
    if z < 0 {
        return Err(FnormError::Overflow);
    }
    let n = i128::from(z) << 32;
    let r = isqrt(n);
    let r = if n - r * r > r { r + 1 } else { r };
    i64::try_from(r).map_err(|_| FnormError::Overflow)
}

/// The binary16 round of the word `v` as a word: [`round_f16`] with one rounding from the exact
/// grid value, then one shift back. `O(1)`.
fn f16_word(v: i64) -> Result<i64, FnormError> {
    let r = round_f16(Fx { m: v, e: -32 }).map_err(|_| FnormError::Overflow)?;
    let shift = i64::from(r.e) + 32;
    let s = u32::try_from(shift.unsigned_abs()).map_err(|_| FnormError::Overflow)?;
    let w = if shift >= 0 {
        i128::from(r.m).checked_shl(s).ok_or(FnormError::Overflow)?
    } else {
        rshift_rne(i128::from(r.m), s).ok_or(FnormError::Overflow)?
    };
    i64::try_from(w).map_err(|_| FnormError::Overflow)
}

/// RMS-normalize row `h[base..base+hid]`: `out[j] = h[base+j] · rsqrt(mean(h²) + eps)`, the mean
/// by one rounded quotient of the exact `i128` sum of squares, the root by [`sqrt_q`] and the
/// reciprocal root by one rounded quotient. Time `O(hid)`, space the caller's `out`.
fn rms_row(
    h: &[i64],
    base: usize,
    hid: usize,
    eps: i64,
    out: &mut [i64],
) -> Result<(), FnormError> {
    let mut sq: i128 = 0;
    let mut ok = true;
    for j in 0..hid {
        if ok {
            let x = i128::from(h[base + j]);
            match x.checked_mul(x) {
                Some(xx) => match sq.checked_add(xx) {
                    Some(v) => sq = v,
                    None => ok = false,
                },
                None => ok = false,
            }
        }
    }
    if !ok {
        return Err(FnormError::Overflow);
    }
    let mean = div_int(sq, (hid as i128) << 32).ok_or(FnormError::Overflow)?;
    let m = mean.checked_add(eps).ok_or(FnormError::Overflow)?;
    let inv = div_int(1_i128 << 64, i128::from(sqrt_q(m)?)).ok_or(FnormError::Overflow)?;
    let mut ok = true;
    for j in 0..hid {
        if ok {
            match mul_q(h[base + j], inv) {
                Ok(v) => out[j] = v,
                Err(_) => ok = false,
            }
        }
    }
    if ok {
        Ok(())
    } else {
        Err(FnormError::Overflow)
    }
}

/// Fit the final RMSNorm gain `g` by damped Newton-Fisher descent on the readout KL.
///
/// Every argument and the returned `Vec<i64>` are Q32.32 words: the caller converts its `f64`
/// inputs at its own boundary, once. `eps`, `ridge` and `lr` are words too.
///
/// # Errors
///
/// As the module's partiality note: [`FnormError::ShapeMismatch`], [`FnormError::BadEps`] or
/// [`FnormError::Overflow`].
///
/// # Complexity
///
/// Time `O(steps·n·T·vocab·hid)` (two `O(vocab·hid)` logit passes, two `O(vocab)` softmaxes and
/// one `O(vocab·hid)` gradient pass per scored token); space `O(hid + vocab)`.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the teacher/student hidden, their dims, the gain, the tied embedding, the schedule \
              and the coh-prior flag are the Python signature, and the body is that one descent"
)]
pub fn fnorm_fit(
    htq: &[i64],
    hsq: &[i64],
    n: usize,
    t: usize,
    hid: usize,
    g0q: &[i64],
    eq: &[i64],
    vocab: usize,
    eps_q: i64,
    steps: usize,
    ridge_q: i64,
    lr_q: i64,
    coh_prior: bool,
) -> Result<Vec<i64>, FnormError> {
    if n == 0 || t == 0 || hid == 0 || vocab == 0 {
        return Err(FnormError::ShapeMismatch);
    }
    let nt = n
        .checked_mul(t)
        .and_then(|x| x.checked_mul(hid))
        .ok_or(FnormError::ShapeMismatch)?;
    let vh = vocab.checked_mul(hid).ok_or(FnormError::ShapeMismatch)?;
    if htq.len() != nt || hsq.len() != nt || g0q.len() != hid || eq.len() != vh {
        return Err(FnormError::ShapeMismatch);
    }
    if eps_q <= 0 {
        return Err(FnormError::BadEps);
    }
    let pos_start = usize::from(coh_prior);
    let t_eff = t - pos_start;
    if t_eff == 0 {
        return Err(FnormError::ShapeMismatch);
    }

    let n_tok = i128::from(
        i64::try_from(n.checked_mul(t_eff).ok_or(FnormError::Overflow)?)
            .map_err(|_| FnormError::Overflow)?,
    );

    // coh-prior target (Python lines 989–1003).
    let mut g_target = g0q.to_vec();
    if coh_prior {
        let (mut num, mut den_t, mut den_s) =
            (vec![0_i128; hid], vec![0_i128; hid], vec![0_i128; hid]);
        let (mut ut, mut us) = (vec![0_i64; hid], vec![0_i64; hid]);
        for s in 0..n {
            for p in pos_start..t {
                let base = (s * t + p) * hid;
                rms_row(htq, base, hid, eps_q, &mut ut)?;
                rms_row(hsq, base, hid, eps_q, &mut us)?;
                for j in 0..hid {
                    let (a, b) = (i128::from(ut[j]), i128::from(us[j]));
                    num[j] = num[j].checked_add(a * b).ok_or(FnormError::Overflow)?;
                    den_t[j] = den_t[j].checked_add(a * a).ok_or(FnormError::Overflow)?;
                    den_s[j] = den_s[j].checked_add(b * b).ok_or(FnormError::Overflow)?;
                }
            }
        }
        for j in 0..hid {
            let num = round_q32_even(num[j]).map_err(|_| FnormError::Overflow)?;
            let den_t = round_q32_even(den_t[j]).map_err(|_| FnormError::Overflow)?;
            let den_s = round_q32_even(den_s[j]).map_err(|_| FnormError::Overflow)?;
            let root = sqrt_q(mul_q(den_t, den_s)?)?;
            let rho = if root == 0 {
                0
            } else {
                div_q(num, root)?.clamp(0, ONE)
            };
            let alpha = if den_t == 0 {
                0
            } else {
                sqrt_q(div_q(den_s, den_t)?)?
            };
            g_target[j] = mul_q(g0q[j], div_q(rho, alpha.max(HALF_W))?)?;
        }
    }

    let mut g = g0q.to_vec();
    let (mut ut, mut us) = (vec![0_i64; hid], vec![0_i64; hid]);
    let (mut pt, mut ps) = (vec![0_i64; vocab], vec![0_i64; vocab]);
    let (mut utg, mut usg) = (vec![0_i64; hid], vec![0_i64; hid]);
    for _ in 0..steps {
        let (mut sg, mut sf) = (vec![0_i128; hid], vec![0_i128; hid]);
        for s in 0..n {
            for p in pos_start..t {
                let base = (s * t + p) * hid;
                rms_row(htq, base, hid, eps_q, &mut ut)?;
                rms_row(hsq, base, hid, eps_q, &mut us)?;
                for j in 0..hid {
                    utg[j] = mul_q(ut[j], g0q[j])?;
                    usg[j] = mul_q(us[j], g[j])?;
                }
                for k in 0..vocab {
                    let (mut zt, mut zs) = (0_i128, 0_i128);
                    for j in 0..hid {
                        let ek = i128::from(eq[k * hid + j]);
                        zt = zt
                            .checked_add(i128::from(utg[j]) * ek)
                            .ok_or(FnormError::Overflow)?;
                        zs = zs
                            .checked_add(i128::from(usg[j]) * ek)
                            .ok_or(FnormError::Overflow)?;
                    }
                    pt[k] = round_q32_even(zt).map_err(|_| FnormError::Overflow)?;
                    ps[k] = round_q32_even(zs).map_err(|_| FnormError::Overflow)?;
                }
                pt = softmax_row(&pt).map_err(|_| FnormError::Overflow)?;
                ps = softmax_row(&ps).map_err(|_| FnormError::Overflow)?;
                for j in 0..hid {
                    let (mut diff, mut mu, mut sec) = (0_i128, 0_i128, 0_i128);
                    for k in 0..vocab {
                        let ek = i128::from(eq[k * hid + j]);
                        let d = i128::from(ps[k]) - i128::from(pt[k]);
                        diff = diff.checked_add(d * ek).ok_or(FnormError::Overflow)?;
                        let w = mul_q(ps[k], eq[k * hid + j])?;
                        mu = mu.checked_add(i128::from(w)).ok_or(FnormError::Overflow)?;
                        sec = sec
                            .checked_add(i128::from(w) * ek)
                            .ok_or(FnormError::Overflow)?;
                    }
                    let diff = round_q32_even(diff).map_err(|_| FnormError::Overflow)?;
                    let mu = i64::try_from(mu).map_err(|_| FnormError::Overflow)?;
                    let sec = round_q32_even(sec).map_err(|_| FnormError::Overflow)?;
                    let var = sec
                        .checked_sub(mul_q(mu, mu)?)
                        .ok_or(FnormError::Overflow)?
                        .max(MIN_W);
                    let g1 = mul_q(us[j], diff)?;
                    let f1 = mul_q(mul_q(us[j], us[j])?, var)?;
                    sg[j] = sg[j]
                        .checked_add(i128::from(g1))
                        .ok_or(FnormError::Overflow)?;
                    sf[j] = sf[j]
                        .checked_add(i128::from(f1))
                        .ok_or(FnormError::Overflow)?;
                }
            }
        }
        let mut fsum: i128 = 0;
        let (mut grad, mut fisher) = (vec![0_i64; hid], vec![0_i64; hid]);
        for j in 0..hid {
            grad[j] = div_int(sg[j], n_tok).ok_or(FnormError::Overflow)?;
            fisher[j] = div_int(sf[j], n_tok).ok_or(FnormError::Overflow)?;
            fsum = fsum
                .checked_add(i128::from(fisher[j]))
                .ok_or(FnormError::Overflow)?;
        }
        let fmean = div_int(fsum, hid as i128).ok_or(FnormError::Overflow)?;
        let reg = mul_q(ridge_q, fmean.max(MIN_W))?;
        for j in 0..hid {
            let pull = mul_q(
                reg,
                g[j].checked_sub(g_target[j]).ok_or(FnormError::Overflow)?,
            )?;
            let num = grad[j].checked_add(pull).ok_or(FnormError::Overflow)?;
            let den = fisher[j].checked_add(reg).ok_or(FnormError::Overflow)?;
            let step = div_q(mul_q(lr_q, num)?, den)?;
            g[j] = f16_word(g[j].checked_sub(step).ok_or(FnormError::Overflow)?)?;
        }
    }
    Ok(g)
}
