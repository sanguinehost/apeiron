//! Dense backward maps in Q32.32 `i64` with `i128` accumulation (W19 task T1).
//!
//! Ports the backward passes of the frozen `tools/onebit/fold_transfer.py`: the linear map
//! `y = x Wᵀ` (`x` is `n × i`, `W` is `o × i`, `y` is `n × o`), `rms(x, w, eps) = w · x · rsqrt(mean(x²) + eps)`
//! (`fold_transfer.py` line 70), and `SwiGLU` `m = silu(g) · u` (line 90). The forward bodies are the
//! engine's extracted kernels (`engine::rms_norm_fixed`, `engine::silu_fixed`,
//! `numeric::fixed_ln::fixed_sigmoid`); the backward bodies are the exact Jacobian-transpose products
//! of those forwards at the stated rounding. Prior art: `../fold` rev `d497161f` `modelfold/src/arch.rs`
//! `rmsnorm_bwd` (line 3610) and `silu_bwd` (line 3659), `f32`; `RoPE` and `GQA` backward are not here.
//!
//! Rounding is `round_shift_32` (`(v + 2³¹) >> 32`, the engine's value-path epilogue) after every
//! Q32.32 × Q32.32 product, and `(v + d/2) / d` for the division by `n` in `RMSNorm`, as in
//! `engine::value::div_round_nearest`. Every add, multiply, narrowing and dimension product is
//! checked: an overflow, empty input, shape mismatch or out-of-window `RMSNorm` argument returns
//! [`DenseError`]; nothing wraps and nothing panics.
//!
//! Extraction shape: `while` loops with a carried `ok` flag, no `?` or early `return` inside a loop,
//! no closures and no iterator adaptors; `core` and `alloc::vec::Vec` only.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::manual_let_else,
    clippy::manual_map,
    reason = "the narrowing `as i64` is range-checked against i64::MAX/MIN immediately before it, \
              the widening `i128::from` forms are total, and `usize` lengths are converted to i64 \
              only after `i64::try_from` succeeds; the single-letter names are the recurrence \
              symbols of the Python source (`x`, `w`, `g`, `u`, `r`, `s`, `n`, `i`, `o`, `k`); \
              let-else and a closure-free `match` are the forms the pinned extraction fragment \
              models, as in `engine::norm`"
)]

use crate::engine::{rms_norm_fixed, silu_fixed, ONE};
use crate::numeric::fixed_ln::fixed_sigmoid;
use crate::numeric::fixed_rsqrt::fixed_rsqrt;

/// A refused dense backward or forward evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseError {
    /// A dimension or the input row is zero-length, so the map is undefined.
    Empty,
    /// A slice length disagrees with the declared dimensions.
    Shape,
    /// A checked `i128`/`i64` operation, a dimension product or an `i64` narrowing left its range.
    Overflow,
    /// An `RMSNorm` argument `mean(x²) + eps` left the `fixed_rsqrt` window or `i64`, or a `SiLU` or
    /// sigmoid refusal was reported by its callee.
    Range,
}

/// Round-to-nearest shift `(v + 2³¹) >> 32` of an `i128` Q64.64 product to Q32.32, or `None` on
/// overflow of the rounding add.
fn round32(v: i128) -> Option<i128> {
    match v.checked_add(1_i128 << 31) {
        Some(s) => Some(s >> 32),
        None => None,
    }
}

/// `v` as `i64`, or `None` if it is outside `i64`.
fn narrow(v: i128) -> Option<i64> {
    if v > i64::MAX as i128 || v < i64::MIN as i128 {
        None
    } else {
        Some(v as i64)
    }
}

/// Q32.32 product `round(a · b / 2³²)`, or `None` on overflow.
fn mul_round(a: i64, b: i64) -> Option<i64> {
    match i128::from(a).checked_mul(i128::from(b)) {
        Some(p) => match round32(p) {
            Some(r) => narrow(r),
            None => None,
        },
        None => None,
    }
}

/// Lift a checked `Option` to `DenseError::Overflow`.
fn ov(o: Option<i64>) -> Result<i64, DenseError> {
    match o {
        Some(v) => Ok(v),
        None => Err(DenseError::Overflow),
    }
}

/// Signed `round(v / d)`, nearest with ties away from zero (the rule of the engine's
/// `round_div_signed`, `src/engine/attn.rs`), `d ≥ 1`, or `None` if the result leaves `i64`.
fn div_round(v: i64, d: i64) -> Option<i64> {
    let num = i128::from(v);
    let den = i128::from(d);
    let q = if num >= 0 {
        (num + den / 2) / den
    } else {
        (num - den / 2) / den
    };
    narrow(q)
}

/// Check that `len` is a `rows × cols` row-major buffer with both dimensions non-zero.
fn expect_len(rows: usize, cols: usize, len: usize) -> Result<(), DenseError> {
    if rows == 0 || cols == 0 {
        return Err(DenseError::Empty);
    }
    match rows.checked_mul(cols) {
        None => Err(DenseError::Overflow),
        Some(total) => {
            if total == len {
                Ok(())
            } else {
                Err(DenseError::Shape)
            }
        }
    }
}

/// Unrounded `y[r,c] = Σ_k x[r,k] · w[c,k]`, row-major `n × o`, as `i128` accumulators.
///
/// # Complexity
///
/// `n·o·i` checked `i128` multiply-adds, `O(n·o)` words of output. Derivation: three nested
/// `while` loops over `r < n`, `c < o`, `k < i`, each iteration one product and one add.
fn linear_fwd_acc(
    x: &[i64],
    w: &[i64],
    n: usize,
    i: usize,
    o: usize,
) -> Result<Vec<i128>, DenseError> {
    let mut out: Vec<i128> = Vec::with_capacity(n * o);
    let mut r = 0usize;
    let mut ok = true;
    while r < n && ok {
        let mut c = 0usize;
        while c < o && ok {
            let mut k = 0usize;
            let mut acc: i128 = 0;
            while k < i && ok {
                match i128::from(x[r * i + k]).checked_mul(i128::from(w[c * i + k])) {
                    Some(p) => match acc.checked_add(p) {
                        Some(s) => acc = s,
                        None => ok = false,
                    },
                    None => ok = false,
                }
                k += 1;
            }
            if ok {
                out.push(acc);
            }
            c += 1;
        }
        r += 1;
    }
    if ok {
        Ok(out)
    } else {
        Err(DenseError::Overflow)
    }
}

/// Unrounded `dx[r,k] = Σ_c dy[r,c] · w[c,k]`, row-major `n × i`, as `i128` accumulators.
///
/// # Complexity
///
/// `n·i·o` checked `i128` multiply-adds, `O(n·i)` words of output. Derivation: nested `while`
/// loops over `r < n`, `k < i`, `c < o`; one product and one add per iteration.
fn linear_bwd_input_acc(
    dy: &[i64],
    w: &[i64],
    n: usize,
    i: usize,
    o: usize,
) -> Result<Vec<i128>, DenseError> {
    let mut out: Vec<i128> = Vec::with_capacity(n * i);
    let mut r = 0usize;
    let mut ok = true;
    while r < n && ok {
        let mut k = 0usize;
        while k < i && ok {
            let mut c = 0usize;
            let mut acc: i128 = 0;
            while c < o && ok {
                match i128::from(dy[r * o + c]).checked_mul(i128::from(w[c * i + k])) {
                    Some(p) => match acc.checked_add(p) {
                        Some(s) => acc = s,
                        None => ok = false,
                    },
                    None => ok = false,
                }
                c += 1;
            }
            if ok {
                out.push(acc);
            }
            k += 1;
        }
        r += 1;
    }
    if ok {
        Ok(out)
    } else {
        Err(DenseError::Overflow)
    }
}

/// Unrounded `dw[c,k] = Σ_r dy[r,c] · x[r,k]`, row-major `o × i`, as `i128` accumulators.
///
/// # Complexity
///
/// `o·i·n` checked `i128` multiply-adds, `O(o·i)` words of output. Derivation: nested `while`
/// loops over `c < o`, `k < i`, `r < n`; one product and one add per iteration.
fn linear_bwd_weight_acc(
    dy: &[i64],
    x: &[i64],
    n: usize,
    i: usize,
    o: usize,
) -> Result<Vec<i128>, DenseError> {
    let mut out: Vec<i128> = Vec::with_capacity(o * i);
    let mut c = 0usize;
    let mut ok = true;
    while c < o && ok {
        let mut k = 0usize;
        while k < i && ok {
            let mut r = 0usize;
            let mut acc: i128 = 0;
            while r < n && ok {
                match i128::from(dy[r * o + c]).checked_mul(i128::from(x[r * i + k])) {
                    Some(p) => match acc.checked_add(p) {
                        Some(s) => acc = s,
                        None => ok = false,
                    },
                    None => ok = false,
                }
                r += 1;
            }
            if ok {
                out.push(acc);
            }
            k += 1;
        }
        c += 1;
    }
    if ok {
        Ok(out)
    } else {
        Err(DenseError::Overflow)
    }
}

/// Round every accumulator to Q32.32 with [`round32`] and narrow it.
///
/// # Complexity
///
/// `O(len)` checked shifts and narrowings; `O(len)` words of output.
fn round_all(acc: &[i128]) -> Result<Vec<i64>, DenseError> {
    let n = acc.len();
    let mut out: Vec<i64> = Vec::with_capacity(n);
    let mut k = 0usize;
    let mut ok = true;
    while k < n && ok {
        match round32(acc[k]) {
            Some(v) => match narrow(v) {
                Some(q) => out.push(q),
                None => ok = false,
            },
            None => ok = false,
        }
        k += 1;
    }
    if ok {
        Ok(out)
    } else {
        Err(DenseError::Overflow)
    }
}

/// Linear forward `y = x Wᵀ` in Q32.32: `x` is `n × i`, `w` is `o × i`, result `n × o`, each entry
/// `round32(Σ_k x[r,k]·w[c,k])`.
///
/// # Errors
///
/// [`DenseError::Empty`] for a zero dimension; [`DenseError::Shape`] when `x` or `w` has the wrong
/// length; [`DenseError::Overflow`] when `n·i`, `o·i` or `n·o` leaves `usize`, an `i128` accumulation
/// leaves its range, or an output leaves `i64`.
///
/// # Complexity
///
/// `O(n·o·i)` checked multiply-adds; derivation as [`linear_fwd_acc`].
pub fn linear_fwd(
    x: &[i64],
    w: &[i64],
    n: usize,
    i: usize,
    o: usize,
) -> Result<Vec<i64>, DenseError> {
    expect_len(n, i, x.len())?;
    expect_len(o, i, w.len())?;
    let acc = linear_fwd_acc(x, w, n, i, o)?;
    round_all(&acc)
}

/// Linear input adjoint `dx = dy · W` in Q32.32: `dy` is `n × o`, `w` is `o × i`, result `n × i`.
///
/// # Errors
///
/// As [`linear_fwd`], with `dy` in place of `x` and `dx` of length `n·i`.
///
/// # Complexity
///
/// `O(n·i·o)` checked multiply-adds; derivation as [`linear_bwd_input_acc`].
pub fn linear_bwd_input(
    dy: &[i64],
    w: &[i64],
    n: usize,
    i: usize,
    o: usize,
) -> Result<Vec<i64>, DenseError> {
    expect_len(n, o, dy.len())?;
    expect_len(o, i, w.len())?;
    let acc = linear_bwd_input_acc(dy, w, n, i, o)?;
    round_all(&acc)
}

/// Linear weight adjoint `dW = dyᵀ · x` in Q32.32: `dy` is `n × o`, `x` is `n × i`, result `o × i`.
///
/// # Errors
///
/// As [`linear_fwd`], with `dy` of shape `n × o`, `x` of shape `n × i`, and `dw` of length `o·i`.
///
/// # Complexity
///
/// `O(o·i·n)` checked multiply-adds; derivation as [`linear_bwd_weight_acc`].
pub fn linear_bwd_weight(
    dy: &[i64],
    x: &[i64],
    n: usize,
    i: usize,
    o: usize,
) -> Result<Vec<i64>, DenseError> {
    expect_len(n, o, dy.len())?;
    expect_len(n, i, x.len())?;
    let acc = linear_bwd_weight_acc(dy, x, n, i, o)?;
    round_all(&acc)
}

/// The Q32.32 inverse root `r = fixed_rsqrt(mean(x²) + eps)` with the mean computed exactly as
/// [`rms_norm_fixed`] computes it: `Σ x²` in `i128`, divided by `n·2³²` with one round-to-nearest.
///
/// # Errors
///
/// [`DenseError::Empty`] for an empty row; [`DenseError::Range`] when `mean + eps` leaves `i64` or
/// `fixed_rsqrt` refuses (argument outside `[2⁻²⁰, 2²⁰]`), and [`DenseError::Overflow`] on a checked
/// `i128` operation.
///
/// # Complexity
///
/// `O(n)` checked `i128` multiply-adds; derivation: one loop over `x`, one division, one `fixed_rsqrt`.
fn inv_rms(x: &[i64], eps: i64) -> Result<i64, DenseError> {
    let n = x.len();
    if n == 0 {
        return Err(DenseError::Empty);
    }
    let nd = match i64::try_from(n) {
        Ok(v) => v,
        Err(_) => return Err(DenseError::Overflow),
    };
    let mut sum_sq: i128 = 0;
    let mut k = 0usize;
    let mut ok = true;
    while k < n && ok {
        match i128::from(x[k]).checked_mul(i128::from(x[k])) {
            Some(sq) => match sum_sq.checked_add(sq) {
                Some(s) => sum_sq = s,
                None => ok = false,
            },
            None => ok = false,
        }
        k += 1;
    }
    if !ok {
        return Err(DenseError::Overflow);
    }
    let den = match i128::from(nd).checked_mul(i128::from(ONE)) {
        Some(d) => d,
        None => return Err(DenseError::Overflow),
    };
    let half = match sum_sq.checked_add(den / 2) {
        Some(h) => h,
        None => return Err(DenseError::Overflow),
    };
    let mean = half / den;
    let arg = match mean.checked_add(i128::from(eps)) {
        Some(a) => a,
        None => return Err(DenseError::Overflow),
    };
    let arg64 = match narrow(arg) {
        Some(a) => a,
        None => return Err(DenseError::Range),
    };
    match fixed_rsqrt(arg64) {
        Ok(r) => Ok(r),
        Err(_) => Err(DenseError::Range),
    }
}

/// `RMSNorm` forward with gain, `y_i = round(g_i · round(x_i · r) / 2³²)`, where the normalised
/// value `round(x_i · r)` is [`rms_norm_fixed`]'s output and `r = fixed_rsqrt(mean(x²) + eps)`.
///
/// # Errors
///
/// [`DenseError::Empty`] for an empty row; [`DenseError::Shape`] when `gamma` differs in length from
/// `x`; [`DenseError::Range`] when `rms_norm_fixed` refuses (`mean + eps` outside the `fixed_rsqrt`
/// window or `i64`); [`DenseError::Overflow`] when the gain product leaves `i64`.
///
/// # Complexity
///
/// `O(n)` checked operations: one [`rms_norm_fixed`] call (`O(n)`) and one `n`-long gain loop.
pub fn rmsnorm_fwd(x: &[i64], gamma: &[i64], eps: i64) -> Result<Vec<i64>, DenseError> {
    let n = x.len();
    if n == 0 {
        return Err(DenseError::Empty);
    }
    if gamma.len() != n {
        return Err(DenseError::Shape);
    }
    let normed = match rms_norm_fixed(x, eps) {
        Ok(v) => v,
        Err(crate::engine::NormError::EmptyRow) => return Err(DenseError::Empty),
        Err(crate::engine::NormError::OutOfRange) => return Err(DenseError::Range),
    };
    let mut out: Vec<i64> = Vec::with_capacity(n);
    let mut k = 0usize;
    let mut ok = true;
    while k < n && ok {
        match mul_round(gamma[k], normed[k]) {
            Some(v) => out.push(v),
            None => ok = false,
        }
        k += 1;
    }
    if ok {
        Ok(out)
    } else {
        Err(DenseError::Overflow)
    }
}

/// `RMSNorm` backward: returns `(dx, dgamma)` for `y = g · x · r`, `r = (mean(x²) + eps)^(-1/2)`.
///
/// With `a_j = g_j·dy_j` and `S = Σ_j a_j·x_j`:
/// `dx_i = r·a_i − x_i·r³·S/n` and `dgamma_i = dy_i·x_i·r`. The `r³` term is the derivative of `r`
/// through `mean(x²)`: `∂r/∂x_i = −r³·x_i/n`.
///
/// # Errors
///
/// [`DenseError::Empty`] for an empty row; [`DenseError::Shape`] when `gamma` or `dy` differs in
/// length from `x`; [`DenseError::Range`] from [`inv_rms`]; [`DenseError::Overflow`] on any checked
/// product, sum, narrowing or `n` conversion.
///
/// # Complexity
///
/// `O(n)` checked operations: one [`inv_rms`] (`O(n)`), one `a`/`S` loop and one output loop, each
/// `n` iterations. Space `O(n)` for `a` and the two outputs.
pub fn rmsnorm_bwd(
    x: &[i64],
    gamma: &[i64],
    eps: i64,
    dy: &[i64],
) -> Result<(Vec<i64>, Vec<i64>), DenseError> {
    let n = x.len();
    if n == 0 {
        return Err(DenseError::Empty);
    }
    if gamma.len() != n || dy.len() != n {
        return Err(DenseError::Shape);
    }
    let nd = match i64::try_from(n) {
        Ok(v) => v,
        Err(_) => return Err(DenseError::Overflow),
    };
    let r = inv_rms(x, eps)?;
    // r² is rounded once to Q32.32 and r³ is never rounded: t = round(x·r²·r / 2⁶⁴) is one product
    // chain with a single final rounding. Rounding r³ itself to Q32.32 costs 2⁻⁵ relative when
    // r ≈ 2⁻¹⁰ real (the big-row fixture). With r²_raw ≥ 2⁻¹²·2³² the Q32 r² is within 1.2·10⁻⁴
    // relative, and every intermediate fits i128 inside the fixed_rsqrt window (x·r² ≤ √n·2⁷⁴).
    let r2 = ov(mul_round(r, r))?;
    // a_j = g_j·dy_j (Q32.32) and S = Σ_j a_j·x_j, rounded once.
    let mut a: Vec<i64> = Vec::with_capacity(n);
    let mut acc: i128 = 0;
    let mut k = 0usize;
    let mut ok = true;
    while k < n && ok {
        match mul_round(gamma[k], dy[k]) {
            Some(av) => {
                a.push(av);
                match i128::from(av).checked_mul(i128::from(x[k])) {
                    Some(p) => match acc.checked_add(p) {
                        Some(s) => acc = s,
                        None => ok = false,
                    },
                    None => ok = false,
                }
            }
            None => ok = false,
        }
        k += 1;
    }
    if !ok {
        return Err(DenseError::Overflow);
    }
    let s = ov(match round32(acc) {
        Some(v) => narrow(v),
        None => None,
    })?;
    let mut dx: Vec<i64> = Vec::with_capacity(n);
    let mut dg: Vec<i64> = Vec::with_capacity(n);
    k = 0;
    ok = true;
    while k < n && ok {
        // p = r·a_i ; q = (x_i·r³)·S / n ; dx_i = p − q.
        let p = mul_round(r, a[k]);
        // t = round(x·r³) in Q32.32 = round(x_raw·r2_raw·r_raw / 2⁶⁴).
        let t = match i128::from(x[k]).checked_mul(i128::from(r2)) {
            Some(xp) => match xp.checked_mul(i128::from(r)) {
                Some(xpr) => match xpr.checked_add(1_i128 << 63) {
                    Some(sum) => narrow(sum >> 64),
                    None => None,
                },
                None => None,
            },
            None => None,
        };
        let q = match t {
            Some(tv) => match mul_round(tv, s) {
                Some(ts) => div_round(ts, nd),
                None => None,
            },
            None => None,
        };
        let d = match (p, q) {
            (Some(pv), Some(qv)) => pv.checked_sub(qv),
            _ => None,
        };
        let g = match mul_round(dy[k], x[k]) {
            Some(v) => mul_round(v, r),
            None => None,
        };
        match (d, g) {
            (Some(dv), Some(gv)) => {
                dx.push(dv);
                dg.push(gv);
            }
            _ => ok = false,
        }
        k += 1;
    }
    if ok {
        Ok((dx, dg))
    } else {
        Err(DenseError::Overflow)
    }
}

/// `SiLU` forward `silu(g)` via [`silu_fixed`]; the callee's error is reported as [`DenseError::Range`].
///
/// # Errors
///
/// [`DenseError::Range`] if [`silu_fixed`] refuses. The callee's refusal arms are unreachable for
/// any `i64` input: they require a `fixed_exp` argument outside its window, which the saturation
/// branches of `silu_fixed` exclude.
fn silu_one(g: i64) -> Result<i64, DenseError> {
    match silu_fixed(g) {
        Ok(v) => Ok(v),
        Err(_) => Err(DenseError::Range),
    }
}

/// Logistic `σ(g)` via [`fixed_sigmoid`]; the callee's error is reported as [`DenseError::Range`].
///
/// # Errors
///
/// [`DenseError::Range`] if [`fixed_sigmoid`] refuses. Its documented contract is `Ok` for every
/// `i64`, so this arm is unreachable.
fn sigmoid_one(g: i64) -> Result<i64, DenseError> {
    match fixed_sigmoid(g) {
        Ok(v) => Ok(v),
        Err(_) => Err(DenseError::Range),
    }
}

/// `SiLU` derivative `silu'(g) = σ(g)·(1 + g·(1 − σ(g)))`, given `s = σ(g)` in Q32.32.
///
/// # Errors
///
/// [`DenseError::Overflow`] on a checked product, or a result outside `i64`.
fn silu_deriv(g: i64, s: i64) -> Result<i64, DenseError> {
    let om = ov(ONE.checked_sub(s))?;
    let gm = ov(mul_round(g, om))?;
    let t = i128::from(ONE) + i128::from(gm);
    let st = match i128::from(s).checked_mul(t) {
        Some(p) => p,
        None => return Err(DenseError::Overflow),
    };
    let r = match round32(st) {
        Some(v) => v,
        None => return Err(DenseError::Overflow),
    };
    match narrow(r) {
        Some(v) => Ok(v),
        None => Err(DenseError::Overflow),
    }
}

/// `SwiGLU` forward `m_i = round(silu(g_i) · u_i / 2³²)`.
///
/// # Errors
///
/// [`DenseError::Empty`] for an empty input; [`DenseError::Shape`] when `up` differs in length from
/// `gate`; [`DenseError::Range`] from [`silu_one`]; [`DenseError::Overflow`] when a product leaves `i64`.
///
/// # Complexity
///
/// `O(n)` checked operations: one [`silu_fixed`] (`O(1)`) and one product per element.
pub fn swiglu_fwd(gate: &[i64], up: &[i64]) -> Result<Vec<i64>, DenseError> {
    let n = gate.len();
    if n == 0 {
        return Err(DenseError::Empty);
    }
    if up.len() != n {
        return Err(DenseError::Shape);
    }
    let mut out: Vec<i64> = Vec::with_capacity(n);
    let mut k = 0usize;
    let mut err = DenseError::Overflow;
    let mut ok = true;
    while k < n && ok {
        match silu_one(gate[k]) {
            Ok(sv) => match mul_round(sv, up[k]) {
                Some(v) => out.push(v),
                None => ok = false,
            },
            Err(e) => {
                err = e;
                ok = false;
            }
        }
        k += 1;
    }
    if ok {
        Ok(out)
    } else {
        Err(err)
    }
}

/// One `SwiGLU` backward element: `(dgate, dup)` with `dup = dy·silu(g)` and
/// `dgate = dy·u·silu'(g)`, each product rounded to Q32.32.
///
/// # Errors
///
/// As [`silu_one`], [`sigmoid_one`], [`silu_deriv`] and [`DenseError::Overflow`] for the products.
fn swiglu_one_bwd(g: i64, u: i64, dy: i64) -> Result<(i64, i64), DenseError> {
    let s = sigmoid_one(g)?;
    let dsilu = silu_deriv(g, s)?;
    let sv = silu_one(g)?;
    let dgu = ov(mul_round(dy, u))?;
    let dgate = ov(mul_round(dgu, dsilu))?;
    let dup = ov(mul_round(dy, sv))?;
    Ok((dgate, dup))
}

/// `SwiGLU` backward: `(dgate, dup)` for `m = silu(gate) · up`.
///
/// # Errors
///
/// [`DenseError::Empty`] for an empty input; [`DenseError::Shape`] when `up` or `dy` differs in
/// length from `gate`; any error of [`swiglu_one_bwd`] for the first failing element.
///
/// # Complexity
///
/// `O(n)` checked operations, one [`swiglu_one_bwd`] per element; `O(n)` words of output.
pub fn swiglu_bwd(
    gate: &[i64],
    up: &[i64],
    dy: &[i64],
) -> Result<(Vec<i64>, Vec<i64>), DenseError> {
    let n = gate.len();
    if n == 0 {
        return Err(DenseError::Empty);
    }
    if up.len() != n || dy.len() != n {
        return Err(DenseError::Shape);
    }
    let mut dgate: Vec<i64> = Vec::with_capacity(n);
    let mut dup: Vec<i64> = Vec::with_capacity(n);
    let mut k = 0usize;
    let mut err = DenseError::Overflow;
    let mut ok = true;
    while k < n && ok {
        match swiglu_one_bwd(gate[k], up[k], dy[k]) {
            Ok((dg, du)) => {
                dgate.push(dg);
                dup.push(du);
            }
            Err(e) => {
                err = e;
                ok = false;
            }
        }
        k += 1;
    }
    if ok {
        Ok((dgate, dup))
    } else {
        Err(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Xorshift64 draw in `[-scale, scale)`; tests only. The residue is below `2·scale ≤ 2⁶²`, so the
    /// `u64 → i64` cast cannot wrap.
    #[allow(
        clippy::cast_possible_wrap,
        reason = "residue below 2·scale, scale ≤ 2^62 in every fixture"
    )]
    fn draw(s: &mut u64, scale: i64) -> i64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        let m = (*s % (2 * scale as u64)) as i64;
        m - scale
    }

    /// Central-difference step `h = 2⁻¹²` in Q32.32 (`2²⁰` raw).
    const H: i64 = 1 << 20;

    /// Sum of the `i128` products `dy · f(x)` over the output, `Q64.64` units: `L` for the FD check.
    fn weighted_sum(dy: &[i64], y: &[i64]) -> i128 {
        let mut acc: i128 = 0;
        for (d, v) in dy.iter().zip(y.iter()) {
            acc += i128::from(*d) * i128::from(*v);
        }
        acc
    }

    /// Derivative in Q32.32 units from the `Q64.64` difference of `L` over the step `2h`:
    /// `deriv_raw = ΔL · 2¹¹ / 2⁶⁴ · 2³² = ΔL / 2²¹`, rounded to nearest.
    fn fd_raw(lp: i128, lm: i128) -> i128 {
        let d = lp - lm;
        (d + (1 << 20)) >> 21
    }

    fn abs_diff(a: i128, b: i128) -> i128 {
        if a > b {
            a - b
        } else {
            b - a
        }
    }

    #[test]
    fn linear_adjoint_identity_exact_in_i128() {
        let shapes: [(usize, usize, usize); 3] = [(2, 3, 4), (1, 1, 1), (5, 7, 3)];
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut s = 0usize;
        while s < 3 {
            let (n, i, o) = shapes[s];
            let x: Vec<i64> = (0..n * i).map(|_| draw(&mut seed, 1 << 30)).collect();
            let w: Vec<i64> = (0..o * i).map(|_| draw(&mut seed, 1 << 30)).collect();
            let dy: Vec<i64> = (0..n * o).map(|_| draw(&mut seed, 1 << 30)).collect();
            let y = linear_fwd_acc(&x, &w, n, i, o).expect("fwd acc");
            let dx = linear_bwd_input_acc(&dy, &w, n, i, o).expect("bwd input acc");
            let dw = linear_bwd_weight_acc(&dy, &x, n, i, o).expect("bwd weight acc");
            // <dY, f(X)> = <f*(dY), X> = <dW, W>, all exact in i128.
            let lhs: i128 = y
                .iter()
                .zip(dy.iter())
                .map(|(a, d)| a * i128::from(*d))
                .sum();
            let mid: i128 = dx
                .iter()
                .zip(x.iter())
                .map(|(a, v)| a * i128::from(*v))
                .sum();
            let rhs: i128 = dw
                .iter()
                .zip(w.iter())
                .map(|(a, v)| a * i128::from(*v))
                .sum();
            assert_eq!(lhs, mid, "fixture {s}: input adjoint");
            assert_eq!(lhs, rhs, "fixture {s}: weight adjoint");
            s += 1;
        }
    }

    #[test]
    fn linear_public_outputs_are_rounded_raw() {
        let x = [ONE, 2 * ONE, -ONE];
        let w = [3 * ONE / 2, -ONE, ONE / 4, ONE, 0, 5 * ONE / 3];
        let y = linear_fwd(&x, &w, 1, 3, 2).expect("fwd");
        let raw = linear_fwd_acc(&x, &w, 1, 3, 2).expect("acc");
        assert_eq!(y.len(), 2);
        assert_eq!(i128::from(y[0]), round32(raw[0]).expect("r"));
        assert_eq!(i128::from(y[1]), round32(raw[1]).expect("r"));
        let dy = [ONE, -2 * ONE];
        let dx = linear_bwd_input(&dy, &w, 1, 3, 2).expect("bwd input");
        let dw = linear_bwd_weight(&dy, &x, 1, 3, 2).expect("bwd weight");
        assert_eq!(dx.len(), 3);
        assert_eq!(dw.len(), 6);
    }

    #[test]
    fn linear_refusals_reach_every_arm() {
        // Empty: a zero dimension.
        assert_eq!(linear_fwd(&[], &[ONE], 0, 1, 1), Err(DenseError::Empty));
        assert_eq!(
            linear_bwd_input(&[ONE], &[], 1, 1, 0),
            Err(DenseError::Empty)
        );
        // Shape: lengths disagree with dimensions.
        assert_eq!(
            linear_fwd(&[ONE, ONE], &[ONE], 1, 3, 1),
            Err(DenseError::Shape)
        );
        assert_eq!(
            linear_bwd_weight(&[ONE], &[ONE, ONE], 1, 2, 2),
            Err(DenseError::Shape)
        );
        // Overflow: a dimension product leaves usize.
        assert_eq!(
            linear_fwd(&[], &[], usize::MAX, 2, 1),
            Err(DenseError::Overflow)
        );
        // Overflow: three i64::MAX products exceed i128 in the accumulation.
        let m = [i64::MAX; 3];
        assert_eq!(linear_fwd(&m, &m, 1, 3, 1), Err(DenseError::Overflow));
        assert_eq!(
            linear_bwd_input(&m[..1], &m, 1, 3, 1),
            Err(DenseError::Overflow)
        );
        // Overflow: a single product 2⁹⁶ rounds to 2⁶⁴, which is outside i64.
        assert_eq!(
            linear_fwd(&[1 << 48], &[1 << 48], 1, 1, 1),
            Err(DenseError::Overflow)
        );
    }

    /// `RMSNorm` FD fixture check. `L = Σ dy·rmsnorm(x,g)`; compares `dx` and `dgamma`, each element
    /// with tolerance `tol` in Q32.32 ULPs.
    fn fd_rmsnorm(x: &[i64], gamma: &[i64], eps: i64, dy: &[i64], tol: i128) {
        let (dx, dg) = rmsnorm_bwd(x, gamma, eps, dy).expect("bwd");
        let n = x.len();
        // The forward uses `fixed_rsqrt`, whose relative error is ≤ 1.2·10⁻⁷ ≈ 2⁻²³ (its measured
        // maximum). A central difference differentiates that error: each evaluation carries
        // |y_j|·2⁻²³ raw, and the quotient amplifies the difference by 2¹¹ over 2³² units of dy, so
        // the derivative error is Σ_j |dy_j|·2·|y_j|·2⁻²³ / 2²¹ raw = Σ_j |dy_j|·|y_j| / 2⁴³ (this is
        // the bound; the measured miss on the big-row fixture is about a fifth of it).
        let y = rmsnorm_fwd(x, gamma, eps).expect("y");
        let mut rsqrt_tol: i128 = 0;
        for j in 0..n {
            rsqrt_tol += i128::from(dy[j].abs()) * i128::from(y[j].abs());
        }
        let tol = tol + rsqrt_tol / (1_i128 << 43) + 1;
        for k in 0..n {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[k] += H;
            xm[k] -= H;
            let lp = weighted_sum(dy, &rmsnorm_fwd(&xp, gamma, eps).expect("fwd+"));
            let lm = weighted_sum(dy, &rmsnorm_fwd(&xm, gamma, eps).expect("fwd-"));
            let fd = fd_raw(lp, lm);
            assert!(
                abs_diff(fd, i128::from(dx[k])) <= tol,
                "dx[{k}]: fd {fd} vs {}",
                dx[k]
            );
            let mut gp = gamma.to_vec();
            let mut gm = gamma.to_vec();
            gp[k] += H;
            gm[k] -= H;
            let lp = weighted_sum(dy, &rmsnorm_fwd(x, &gp, eps).expect("gamma+"));
            let lm = weighted_sum(dy, &rmsnorm_fwd(x, &gm, eps).expect("gamma-"));
            let fd = fd_raw(lp, lm);
            assert!(
                abs_diff(fd, i128::from(dg[k])) <= tol,
                "dgamma[{k}]: fd {fd} vs {}",
                dg[k]
            );
        }
    }

    /// Tolerance (ULP) for the `RMSNorm` FD check, derived from the rounding steps.
    ///
    /// Each `rmsnorm_fwd` output carries at most `E = 2` ULP of rounding: one round in `x·r`, one in
    /// `g·(x·r)`. The central difference divides by `2h` and multiplies by `2¹¹`: an error of `E` ULP in
    /// each of `f(x ± h)` gives `2·E·2¹¹` derivative ULP, weighted by `|dy_j|` (at most one unit per
    /// output here, so `Σ|dy|/2³² ≤ n`). The `O(h²)` truncation is `h²·|f'''|/6`, below `2¹⁰` ULP for
    /// the fixtures (`|x| ≤ 2⁴¹`, `eps ≥ 2⁻¹⁰`). Total `2¹² · n + 2¹⁰`.
    fn rms_tol(n: usize) -> i128 {
        (1_i128 << 12) * n as i128 + (1_i128 << 10)
    }

    #[test]
    fn rmsnorm_bwd_central_difference_fixtures() {
        let eps = ONE / 1024;
        let mut seed: u64 = 0xdead_beef_cafe_f00d;
        let mut t = 0usize;
        while t < 3 {
            let n = 3 + t;
            let x: Vec<i64> = (0..n).map(|_| draw(&mut seed, 2 * ONE)).collect();
            let g: Vec<i64> = (0..n).map(|_| ONE + draw(&mut seed, ONE / 2)).collect();
            let dy: Vec<i64> = (0..n).map(|_| draw(&mut seed, ONE)).collect();
            let sum_abs: i128 = dy.iter().map(|v| i128::from(v.abs())).sum();
            let tol = rms_tol(n) + sum_abs / (1 << 20);
            fd_rmsnorm(&x, &g, eps, &dy, tol);
            t += 1;
        }
    }

    #[test]
    fn rmsnorm_bwd_boundary_rows() {
        let eps = ONE / 1024;
        // Zero row: r = eps^(-1/2) = 32, dx = r·g·dy exactly.
        let zero = [0_i64; 4];
        let g = [ONE; 4];
        let dy = [ONE, -ONE, ONE / 2, 2 * ONE];
        // At x = 0 the central difference of f(x) = x·r(x) carries the O(h²) truncation term
        // r0·h²/(2·n·eps). With r(x) ≈ r0·(1 − x²/(2·n·eps)) near 0 and h² = 2⁻²⁴ real, in raw ULP this
        // is r0_raw·2⁸/(2·n·eps_raw); at eps = 2²² raw, n = 4 it is 2²⁰, matching the observed miss.
        let r0 = inv_rms(&zero, eps).expect("r0");
        // Each dx_k carries this truncation times dy_k, so the bound uses max |dy_k| (here 2 real).
        let trunc = i128::from(r0) * 256 / (2 * 4 * i128::from(eps));
        let dymax = i128::from(dy.iter().map(|v| v.abs()).max().expect("nonempty"));
        fd_rmsnorm(
            &zero,
            &g,
            eps,
            &dy,
            rms_tol(4) + trunc * dymax / i128::from(ONE),
        );
        let (dx, _) = rmsnorm_bwd(&zero, &g, eps, &dy).expect("zero bwd");
        for k in 0..4 {
            let want =
                i128::from(mul_round(mul_round(ONE, dy[k]).expect("a"), 32 * ONE).expect("r"));
            assert!(
                abs_diff(i128::from(dx[k]), want) <= 4,
                "zero dx[{k}] {} vs {want}",
                dx[k]
            );
        }
        // One-element row.
        let x1 = [3 * ONE / 2];
        let g1 = [2 * ONE];
        let dy1 = [ONE];
        fd_rmsnorm(&x1, &g1, eps, &dy1, rms_tol(1));
        // Max-magnitude admissible row: |x| = 2⁴¹ (512 real), mean(x²) = 2¹⁸ real, inside the window.
        let big = [1_i64 << 41, -(1_i64 << 41), 1_i64 << 40];
        let gb = [ONE; 3];
        let dyb = [ONE, ONE, -ONE];
        fd_rmsnorm(&big, &gb, eps, &dyb, rms_tol(3));
    }

    #[test]
    fn rmsnorm_refusals_reach_every_arm() {
        let eps = ONE / 1024;
        assert_eq!(rmsnorm_fwd(&[], &[], eps), Err(DenseError::Empty));
        assert_eq!(rmsnorm_bwd(&[], &[], eps, &[]), Err(DenseError::Empty));
        assert_eq!(
            rmsnorm_fwd(&[ONE, ONE], &[ONE], eps),
            Err(DenseError::Shape)
        );
        assert_eq!(
            rmsnorm_bwd(&[ONE], &[ONE], eps, &[ONE, ONE]),
            Err(DenseError::Shape)
        );
        // Range: mean(x²) = 2⁶⁴ real-scaled leaves i64 (rms_norm_fixed OutOfRange, inv_rms narrowing).
        assert_eq!(rmsnorm_fwd(&[1 << 48], &[ONE], eps), Err(DenseError::Range));
        assert_eq!(
            rmsnorm_bwd(&[1 << 48], &[ONE], eps, &[ONE]),
            Err(DenseError::Range)
        );
        // Range: mean(x²) = 2⁵⁴ is in i64 but above the fixed_rsqrt window 2⁵².
        assert_eq!(rmsnorm_fwd(&[1 << 43], &[ONE], eps), Err(DenseError::Range));
        assert_eq!(
            rmsnorm_bwd(&[1 << 43], &[ONE], eps, &[ONE]),
            Err(DenseError::Range)
        );
        // Overflow in the gain product: normalised value 1.414 times i64::MAX.
        assert_eq!(
            rmsnorm_fwd(&[2 * ONE, 0], &[i64::MAX, ONE], eps),
            Err(DenseError::Overflow)
        );
        // Overflow in the backward: a = g·dy = 2⁹⁴ leaves i64.
        assert_eq!(
            rmsnorm_bwd(&[ONE], &[i64::MAX], eps, &[i64::MAX]),
            Err(DenseError::Overflow)
        );
    }

    /// `SwiGLU` FD check: `L = Σ dy·swiglu(g,u)`; compares `dgate` and `dup` with tolerance `tol` ULP.
    ///
    /// Tolerance derivation: `silu_fixed` is within `E = 2` ULP of the real `silu`, its rounding
    /// included; the central difference amplifies an error of `E` ULP per evaluation by `2¹¹` per
    /// unit of `|dy|`. `silu'` carries the `fixed_sigmoid` error and the `fixed_exp` error, `≤ 2⁴`
    /// ULP for the fixtures, multiplied by `|u|` and `|dy|`: the `dgate` tolerance is `2¹² · n + 2¹⁴`.
    fn fd_swiglu(g: &[i64], u: &[i64], dy: &[i64], tol: i128) {
        let (dgate, dup) = swiglu_bwd(g, u, dy).expect("bwd");
        let n = g.len();
        for k in 0..n {
            let mut gp = g.to_vec();
            let mut gm = g.to_vec();
            gp[k] += H;
            gm[k] -= H;
            let lp = weighted_sum(dy, &swiglu_fwd(&gp, u).expect("fwd g+"));
            let lm = weighted_sum(dy, &swiglu_fwd(&gm, u).expect("fwd g-"));
            let fd = fd_raw(lp, lm);
            assert!(
                abs_diff(fd, i128::from(dgate[k])) <= tol,
                "dgate[{k}]: fd {fd} vs {}",
                dgate[k]
            );
            let mut up = u.to_vec();
            let mut um = u.to_vec();
            up[k] += H;
            um[k] -= H;
            let lp = weighted_sum(dy, &swiglu_fwd(g, &up).expect("fwd u+"));
            let lm = weighted_sum(dy, &swiglu_fwd(g, &um).expect("fwd u-"));
            let fd = fd_raw(lp, lm);
            assert!(
                abs_diff(fd, i128::from(dup[k])) <= tol,
                "dup[{k}]: fd {fd} vs {}",
                dup[k]
            );
        }
    }

    #[test]
    fn swiglu_bwd_central_difference_fixtures() {
        let mut seed: u64 = 0x0123_4567_89ab_cdef;
        let mut t = 0usize;
        while t < 3 {
            let n = 3 + t;
            let g: Vec<i64> = (0..n).map(|_| draw(&mut seed, 3 * ONE)).collect();
            let u: Vec<i64> = (0..n).map(|_| draw(&mut seed, 2 * ONE)).collect();
            let dy: Vec<i64> = (0..n).map(|_| draw(&mut seed, ONE)).collect();
            let sum_abs: i128 = dy.iter().map(|v| i128::from(v.abs())).sum();
            let tol = (1_i128 << 12) * n as i128 + (1_i128 << 14) + sum_abs / (1 << 16);
            fd_swiglu(&g, &u, &dy, tol);
            t += 1;
        }
    }

    #[test]
    fn swiglu_bwd_boundary_values() {
        // Zero row: silu'(0) = 1/2, silu(0) = 0, so dgate = dy·u/2 and dup = 0.
        let g = [0_i64; 3];
        let u = [ONE, -2 * ONE, 3 * ONE];
        let dy = [ONE; 3];
        let (dg, du) = swiglu_bwd(&g, &u, &dy).expect("zero");
        for k in 0..3 {
            assert!(
                abs_diff(i128::from(dg[k]), i128::from(u[k]) / 2) <= 2,
                "zero dgate[{k}] {}",
                dg[k]
            );
            assert_eq!(du[k], 0, "zero dup[{k}]");
        }
        fd_swiglu(&g, &u, &dy, (1 << 12) * 3 + (1 << 14));
        // One-element row and saturated gates (|g| beyond the exact window).
        fd_swiglu(&[ONE], &[ONE], &[ONE], (1 << 12) + (1 << 14));
        let gs = [20 * ONE, -20 * ONE, 16 * ONE];
        let us = [ONE, ONE, ONE];
        let dys = [ONE; 3];
        let (dgs, dus) = swiglu_bwd(&gs, &us, &dys).expect("sat");
        assert!(
            abs_diff(i128::from(dgs[0]), i128::from(ONE)) <= 2,
            "saturated high dgate"
        );
        assert!(abs_diff(i128::from(dgs[1]), 0) <= 2, "saturated low dgate");
        // dup = dy·silu(g) with dy = u = 1: high gate saturates to silu(g) = g; low gate is the
        // in-window silu(-20) = -20/(1 + e^20) ≈ -177 ULP, not zero.
        assert!(
            abs_diff(i128::from(dus[0]), i128::from(20 * ONE)) <= 2,
            "saturated high dup"
        );
        let low = silu_fixed(-20 * ONE).expect("silu(-20)");
        assert_eq!(dus[1], low, "low dup is silu(-20)");
        assert!(low < -100 && low > -250, "silu(-20) magnitude {low}");
    }

    #[test]
    fn swiglu_refusals_reach_every_arm() {
        assert_eq!(swiglu_fwd(&[], &[]), Err(DenseError::Empty));
        assert_eq!(swiglu_bwd(&[], &[], &[]), Err(DenseError::Empty));
        assert_eq!(swiglu_fwd(&[ONE, ONE], &[ONE]), Err(DenseError::Shape));
        assert_eq!(
            swiglu_bwd(&[ONE], &[ONE], &[ONE, ONE]),
            Err(DenseError::Shape)
        );
        // Overflow in the forward product: silu(15) ≈ 15 times i64::MAX.
        assert_eq!(
            swiglu_fwd(&[15 * ONE], &[i64::MAX]),
            Err(DenseError::Overflow)
        );
        // Overflow in the backward product dy·u = 2⁹⁴ before the silu factor.
        assert_eq!(
            swiglu_bwd(&[0], &[i64::MAX], &[i64::MAX]),
            Err(DenseError::Overflow)
        );
    }

    #[test]
    fn div_round_is_nearest_ties_away_from_zero_for_both_signs() {
        // (v, d, round(v/d)): exact quotients, below and above the half, and the ties.
        let cases: [(i64, i64, i64); 10] = [
            (8, 4, 2),
            (-8, 4, -2),
            (1, 4, 0),
            (-1, 4, 0),
            (3, 4, 1),
            (-3, 4, -1),
            (2, 4, 1),
            (-2, 4, -1),
            (-2, 3, -1),
            (-5, 1, -5),
        ];
        for (v, d, want) in cases {
            assert_eq!(div_round(v, d), Some(want), "div_round({v}, {d})");
        }
    }
}
