//! Damped Newton-Fisher descent on the final RMSNorm gain `g`, minimising readout KL.
//!
//! Port of `fold_transfer.py` lines 980–1027 (`fnorm_fit`). The Python fixes *what* the fit
//! computes; the body here reproduces its arithmetic in `f64` on the CPU, on the crate's build
//! path.
//!
//! # Inputs and shapes
//!
//! - `ht`, `hs`: teacher and student final hidden states, row-major `N × T × hid` flat.
//! - `n`, `t`, `hid`: sequence, position and hidden counts; `ht.len() == hs.len() == n·t·hid`.
//! - `g0`: the fixed teacher RMSNorm gain, `hid`.
//! - `e` (the tied embedding `E`): `vocab × hid`, row-major.
//! - `eps`: the RMSNorm epsilon; `steps`, `ridge`, `lr`: the descent schedule.
//! - `coh_prior`: score positions `1..T` and start the target from the coh-prior ratio.
//!
//! The fitted gain `g` (`hid`) leaves, each step rounded by [`f16r`] exactly where the Python
//! rounds: `g = f16r(g − lr · (grad + reg·(g − g_target)) / (fisher + reg))`.
//!
//! # Semantics
//!
//! Every scored row `h` is RMS-normalized with `u_j = h_j · rsqrt(mean(h²) + eps)`. The teacher
//! logits use `(u_t · g0)`, the student logits `(u_s · g)`; the `vocab`-way softmax of each gives
//! `pt`/`ps`. With `m = n·(T − [coh_prior])` scored tokens, the step accumulates
//! `grad_j = Σ_tokens u_s,j · ((ps − pt)·E)_j` and the Fisher diagonal
//! `fisher_j = Σ_tokens u_s,j² · clamp((ps·E²)_j − (ps·E)_j², ≥ 1e-6)`, divides both by `m`,
//! sets `reg = ridge · clamp(mean_j fisher_j, ≥ 1e-6)`, and takes the damped Newton step above.
//! For `coh_prior` the target is `g_target_j = g0_j · clamp(rho_j, 0, 1) / clamp(alpha_j, ≥ 0.5)`
//! with `rho_j = clamp(num_j/√(den_t·den_s), 0, 1)`, `alpha_j = √(den_s/den_t)` from the
//! position `1..T` sums `num = Σ u_t·u_s`, `den_t = Σ u_t²`, `den_s = Σ u_s²`; otherwise
//! `g_target = g0`.
//!
//! # Batching
//!
//! The Python accumulates `grad`/`fisher` over batches of 2 sequences (5 for the coh-prior sums)
//! and divides once by `m`; this port sums per sequence and divides once, so the result does not
//! depend on the batch width.
//!
//! # Deviations from the Python
//!
//! - Refuses a non-positive or non-finite `eps` (the `rsqrt` argument must be positive) and any
//!   non-finite entry of `ht`, `hs`, `g0` or `e`; the Python propagates `NaN` silently.
//! - `steps == 0` returns `g0` unrounded, matching the Python (the round is inside the loop).
//! - The per-sequence summation order differs from the Python's batch grouping; the value is
//!   identical up to floating-point associativity, as both divide a single total once.
//!
//! # Grade
//!
//! `tested` (`tests/learn_fold_fnorm.rs`). No extraction or refinement links this body to a Lean
//! theorem, so nothing here is `proved`.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    reason = "the kernels index row-major `[n, t, hid]`, `[vocab, hid]` and `[hid]` buffers by \
              `s/p/j/k` and name the carriers `ut/us`, `pt/ps`, `mu`, `sec` as the reference does; \
              `hid`, `n_tok` are widened to `f64` after their ranges are fixed by the shape check"
)]

use crate::learn::fold::fit::quant::f16r;

/// A refusal from the RMSNorm-gain fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FnormError {
    /// A slice length that is not the declared shape, or a zero dimension (`n`, `t`, `hid`,
    /// `vocab`, or the `coh_prior` scored-position count).
    ShapeMismatch,
    /// A non-finite entry of `ht`, `hs`, `g0` or `e`.
    NonFinite,
    /// `eps` is not finite and positive, so `rsqrt(mean(h²) + eps)` is undefined.
    BadEps,
}

/// RMS-normalize row `h[base..base+hid]`: `out[j] = h[base+j] · rsqrt(mean(h²) + eps)`.
///
/// # Complexity
///
/// Time `O(hid)` (one sum-of-squares pass and one scale pass); space is the caller's `out`.
fn rms_row(h: &[f64], base: usize, hid: usize, eps: f64, out: &mut [f64]) {
    let mut sq = 0.0_f64;
    for j in 0..hid {
        let x = h[base + j];
        sq += x * x;
    }
    let inv = 1.0 / (sq / (hid as f64) + eps).sqrt();
    for j in 0..hid {
        out[j] = h[base + j] * inv;
    }
}

/// In-place softmax of a row (`z[k] = exp(z[k] − max) / Σ exp`).
///
/// # Complexity
///
/// Time `O(len)` (one max scan, one `exp` pass, one scale pass); space `O(1)`.
fn softmax_row(z: &mut [f64]) {
    let mut m = f64::NEG_INFINITY;
    for &x in z.iter() {
        if x > m {
            m = x;
        }
    }
    let mut sum = 0.0_f64;
    for x in z.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }
    for x in z.iter_mut() {
        *x /= sum;
    }
}

/// Fit the final RMSNorm gain `g` by damped Newton-Fisher descent on the readout KL.
///
/// See the module documentation for the inputs, shapes, semantics and batching rule.
///
/// # Errors
///
/// [`FnormError::ShapeMismatch`] for a zero dimension or a slice length that is not the declared
/// shape (including a `coh_prior` call with `t == 1`, which has no scored position);
/// [`FnormError::NonFinite`] for a non-finite entry of `ht`, `hs`, `g0` or `e`; [`FnormError::BadEps`]
/// for an `eps` that is not finite and positive.
///
/// # Complexity
///
/// Time `O(steps · n · T · vocab · hid)`: per scored token one `O(hid)` normalization each for the
/// teacher and student, two `O(vocab·hid)` logit passes, one `O(vocab)` softmax each, and one
/// `O(vocab·hid)` pass for the gradient and the Fisher diagonal. Space `O(hid + vocab)` for the
/// per-token scratch and the accumulators, plus the `hid` output.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the teacher/student hidden, their dims, the gain, the tied embedding, the schedule \
              and the coh-prior flag are the Python signature, and the body is that one straight \
              descent: the coh-prior target, then the per-step gradient/Fisher accumulation and \
              the damped Newton update"
)]
pub fn fnorm_fit(
    ht: &[f64],
    hs: &[f64],
    n: usize,
    t: usize,
    hid: usize,
    g0: &[f64],
    e: &[f64],
    vocab: usize,
    eps: f64,
    steps: usize,
    ridge: f64,
    lr: f64,
    coh_prior: bool,
) -> Result<Vec<f64>, FnormError> {
    if n == 0 || t == 0 || hid == 0 || vocab == 0 {
        return Err(FnormError::ShapeMismatch);
    }
    let nt = n
        .checked_mul(t)
        .and_then(|x| x.checked_mul(hid))
        .ok_or(FnormError::ShapeMismatch)?;
    let vh = vocab.checked_mul(hid).ok_or(FnormError::ShapeMismatch)?;
    if ht.len() != nt || hs.len() != nt || g0.len() != hid || e.len() != vh {
        return Err(FnormError::ShapeMismatch);
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(FnormError::BadEps);
    }
    if ht.iter().any(|x| !x.is_finite())
        || hs.iter().any(|x| !x.is_finite())
        || g0.iter().any(|x| !x.is_finite())
        || e.iter().any(|x| !x.is_finite())
    {
        return Err(FnormError::NonFinite);
    }
    let pos_start = usize::from(coh_prior);
    let t_eff = t - pos_start;
    if t_eff == 0 {
        return Err(FnormError::ShapeMismatch);
    }
    let n_tok = (n as f64) * (t_eff as f64);

    // coh-prior target (Python lines 989–1003).
    let mut g_target = g0.to_vec();
    if coh_prior {
        let mut num = vec![0.0_f64; hid];
        let mut den_t = vec![0.0_f64; hid];
        let mut den_s = vec![0.0_f64; hid];
        let mut ut = vec![0.0_f64; hid];
        let mut us = vec![0.0_f64; hid];
        for s in 0..n {
            for p in pos_start..t {
                let base = (s * t + p) * hid;
                rms_row(ht, base, hid, eps, &mut ut);
                rms_row(hs, base, hid, eps, &mut us);
                for j in 0..hid {
                    num[j] += ut[j] * us[j];
                    den_t[j] += ut[j] * ut[j];
                    den_s[j] += us[j] * us[j];
                }
            }
        }
        for j in 0..hid {
            let prod = (den_t[j] * den_s[j]).max(1e-12).sqrt();
            let rho = (num[j] / prod).clamp(0.0, 1.0);
            let alpha = (den_s[j] / den_t[j].max(1e-12)).sqrt();
            g_target[j] = g0[j] * (rho / alpha.max(0.5));
        }
    }

    let mut g = g0.to_vec();
    let mut ut = vec![0.0_f64; hid];
    let mut us = vec![0.0_f64; hid];
    let mut pt = vec![0.0_f64; vocab];
    let mut ps = vec![0.0_f64; vocab];
    let mut grad = vec![0.0_f64; hid];
    let mut fisher = vec![0.0_f64; hid];
    let mut dg = vec![0.0_f64; hid];
    let mut df = vec![0.0_f64; hid];
    for _ in 0..steps {
        grad.fill(0.0);
        fisher.fill(0.0);
        for s in 0..n {
            dg.fill(0.0);
            df.fill(0.0);
            for p in pos_start..t {
                let base = (s * t + p) * hid;
                rms_row(ht, base, hid, eps, &mut ut);
                rms_row(hs, base, hid, eps, &mut us);
                for k in 0..vocab {
                    let mut zt = 0.0_f64;
                    let mut zs = 0.0_f64;
                    for j in 0..hid {
                        let ek = e[k * hid + j];
                        zt += ut[j] * g0[j] * ek;
                        zs += us[j] * g[j] * ek;
                    }
                    pt[k] = zt;
                    ps[k] = zs;
                }
                softmax_row(&mut pt);
                softmax_row(&mut ps);
                for j in 0..hid {
                    let mut diff_e = 0.0_f64;
                    let mut mu = 0.0_f64;
                    let mut sec = 0.0_f64;
                    for k in 0..vocab {
                        let ek = e[k * hid + j];
                        diff_e += (ps[k] - pt[k]) * ek;
                        mu += ps[k] * ek;
                        sec += ps[k] * ek * ek;
                    }
                    dg[j] += us[j] * diff_e;
                    let var = (sec - mu * mu).max(1e-6);
                    df[j] += us[j] * us[j] * var;
                }
            }
            for j in 0..hid {
                grad[j] += dg[j];
                fisher[j] += df[j];
            }
        }
        for j in 0..hid {
            grad[j] /= n_tok;
            fisher[j] /= n_tok;
        }
        let mut fsum = 0.0_f64;
        for &x in &fisher {
            fsum += x;
        }
        let reg = ridge * (fsum / (hid as f64)).max(1e-6);
        for j in 0..hid {
            g[j] = f16r(g[j] - lr * (grad[j] + reg * (g[j] - g_target[j])) / (fisher[j] + reg));
        }
    }
    Ok(g)
}
