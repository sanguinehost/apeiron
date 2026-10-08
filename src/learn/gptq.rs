//! GPTQ-ternary: error-feedback quantization to balanced trits and affine carriers.
//!
//! Subsumed source: `../fold/modelfold/src/quant.rs` at fold HEAD `d497161f`
//! (sha256 `4042468783b41ccf6ac6dbf0d68966dbfa02c91c54ab8bd30f1e7e75a612cf62`),
//! module `pub mod gptq` (lines 7–488).
//!
//! # Algorithmic foundation
//!
//! Quantizing weights by minimizing context-free weight difference `‖W − Ŵ‖²` fails to minimize the
//! output distortion `‖(W − Ŵ)Xᵀ‖²` experienced by the model forward pass. GPTQ compensates by
//! quantizing column-by-column and pushing each column's residual onto subsequent unquantized
//! columns through the inverse Hessian `H⁻¹` (`H = XᵀX + 0.01·mean(diag)·I`).
//!
//! # Two consumptions of the code set
//!
//! - **Path A — the affine code chart:** [`AffineChart`] with codes `q ∈ 0..=levels`, per-(row, group)
//!   `(d, lo)`, and dequantization `w ≈ d·q + lo`.
//! - **Path B — the balanced-trit planes:** [`Quant::trit_planes`] decomposes each code
//!   `c = Σ_k tₖ·3ᵏ`, `tₖ ∈ {−1, 0, +1}`, into $L$ Q2 planes ([`Q2Plane`]). Each plane carries
//!   `{0, 1, 2}` codes with per-(row, group) `(d, m) = (place·s, −place·s)`, enabling multiply-free
//!   ternary dots `Σ₊ − Σ₋` that recombine by radix-3 place value.
//!
//! # Partiality and errors
//!
//! Fallible entry points return `Result<_, GptqError>`. Slice length and matrix dimension mismatches
//! return [`GptqError::ShapeMismatch`]. Non-positive block sizes or level counts return
//! [`GptqError::ZeroGroup`] or [`GptqError::ZeroLevels`]. A Cholesky factorization encountering a
//! non-positive or NaN diagonal pivot returns [`GptqError::NotPositiveDefinite`]. No function in
//! this module panics on caller input.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    reason = "the flat-slice numeric kernels index row-major [rows, n] buffers by r/i/j/k/o, \
              casts between f64/f32 and integer codes reflect the quantized representations, \
              and doc formulas use mathematical indices matching modelfold quant.rs."
)]

use crate::format::quant::{f16_to_f32, q4_k_scale_min, Q4_K_BLOCK_BYTES, Q4_K_SCALE_SIZE, QK_K};
use crate::learn::seal::quantize_row_q4_k;

/// A refusal from a GPTQ operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GptqError {
    /// A buffer length or dimension parameter violates the expected matrix shape.
    ShapeMismatch,
    /// `group` block size is 0.
    ZeroGroup,
    /// `levels` quantization level count is 0.
    ZeroLevels,
    /// Hessian or inverse Hessian factor is not positive-definite at a Cholesky pivot.
    NotPositiveDefinite,
}

impl core::fmt::Display for GptqError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShapeMismatch => write!(f, "matrix shape or buffer length mismatch"),
            Self::ZeroGroup => write!(f, "group size must be non-zero"),
            Self::ZeroLevels => write!(f, "levels must be non-zero"),
            Self::NotPositiveDefinite => {
                write!(f, "matrix is not positive-definite at a Cholesky pivot")
            }
        }
    }
}

impl std::error::Error for GptqError {}

/// A GPTQ result: integer trit codes, per-row/per-group scales, and the dequantized golden.
/// `codes`/`scale` are Path A (the affine chart); [`Quant::trit_planes`] is Path B.
///
/// Ported from `modelfold/src/quant.rs:44-73`.
#[derive(Clone, Debug, PartialEq)]
pub struct Quant {
    /// Balanced integer codes `c ∈ [−maxlev, maxlev]`, row-major `[out, n]`.
    pub codes: Vec<i8>,
    /// Per-row, per-group scale `s` (`w ≈ s·c`), row-major `[out, groups]`.
    pub scale: Vec<f32>,
    /// The dequantized weights `Ŵ = s·c`, row-major `[out, n]` — the CPU golden the kernel matches.
    pub wq: Vec<f32>,
    /// Output rows.
    pub out: usize,
    /// Contraction width (columns).
    pub n: usize,
    /// Group (block) length along `n` — the scale granularity, and the Q2 block for Path B.
    pub group: usize,
    /// Trits per weight (`L`).
    pub levels: u32,
}

impl Quant {
    /// The largest code magnitude, `(3^L − 1)/2`.
    ///
    /// Ported from `modelfold/src/quant.rs:64-66`.
    #[must_use]
    pub fn maxlev(&self) -> i64 {
        (3i64.pow(self.levels) - 1) / 2
    }

    /// Number of groups (blocks) along `n`.
    ///
    /// Ported from `modelfold/src/quant.rs:69-72`.
    #[must_use]
    pub fn groups(&self) -> usize {
        self.n.div_ceil(self.group)
    }

    /// Path B: split each code `c = Σ_k tₖ·3ᵏ` into `levels` balanced-ternary planes, each a Q2
    /// chart. Recombining the plane dots by place value reproduces the affine dot exactly
    /// (`TernaryDP4A.dot_plane_sum`, `TernaryDP4A.balanced_ternary_lossless`).
    ///
    /// Ported from `modelfold/src/quant.rs:244-275`.
    ///
    /// # Complexity
    /// - Temporal: $O(L \cdot \text{out} \cdot (\text{groups} + n))$ operations.
    ///   Measure: loop iterations and arithmetic assignments. Derivation: allocates $L$ planes;
    ///   assigns $L \times \text{out} \times \text{groups}$ scale and offset values in loop
    ///   `0..self.out` over `0..groups`; extracts $L$ balanced trits for each of $\text{out} \times n$
    ///   entries in loop `0..self.out` over `0..self.n`.
    /// - Spatial: $O(L \cdot \text{out} \cdot (n + 2 \cdot \text{groups}))$ bytes/words.
    ///   Measure: heap bytes allocated across $L$ [`Q2Plane`] structs ($L \times \text{out} \times n$
    ///   bytes for `code`, $2 \times L \times \text{out} \times \text{groups} \times 4$ bytes for `d` and `m`).
    #[must_use]
    pub fn trit_planes(&self) -> Vec<Q2Plane> {
        let groups = self.groups();
        let mut planes: Vec<Q2Plane> = (0..self.levels)
            .map(|k| Q2Plane {
                place: 3i64.pow(k),
                d: vec![0f32; self.out * groups],
                m: vec![0f32; self.out * groups],
                code: vec![0u8; self.out * self.n],
            })
            .collect();
        for plane in &mut planes {
            let place = plane.place as f32;
            for o in 0..self.out {
                for gi in 0..groups {
                    let s = self.scale[o * groups + gi] * place;
                    plane.d[o * groups + gi] = s; // d = place·s
                    plane.m[o * groups + gi] = -s; // m = −place·s  ⇒ d·q+m = place·s·(q−1)
                }
            }
        }
        for o in 0..self.out {
            for j in 0..self.n {
                let trits =
                    balanced_trits(i32::from(self.codes[o * self.n + j]), self.levels as usize);
                for (k, plane) in planes.iter_mut().enumerate() {
                    plane.code[o * self.n + j] = (trits[k] + 1) as u8; // t ∈ {−1,0,1} → q ∈ {0,1,2}
                }
            }
        }
        planes
    }

    /// Dequantize Path B: sum every plane's `place·s·(q−1)` — bit-parity target is [`Quant::wq`]
    /// (Path A), the two consumptions agree by construction.
    ///
    /// Ported from `modelfold/src/quant.rs:280-294`.
    ///
    /// # Complexity
    /// - Temporal: $O(P \cdot \text{out} \cdot n)$ multiply-accumulate operations where $P = \text{planes.len()}$.
    ///   Measure: arithmetic operations in nested loops iterating through $P$ planes, $\text{out}$ rows,
    ///   and $n$ columns.
    /// - Spatial: $O(\text{out} \cdot n)$ floats.
    ///   Measure: allocates a single `Vec<f32>` buffer of length $\text{out} \times n$ (size in bytes:
    ///   $\text{out} \cdot n \cdot 4$).
    ///
    /// # Errors
    /// Returns [`GptqError::ShapeMismatch`] if any plane's `code` buffer length is not $\text{out} \cdot n$
    /// or if its `d` or `m` buffer length is not $\text{out} \cdot \text{groups}$.
    pub fn dequant_planes(&self, planes: &[Q2Plane]) -> Result<Vec<f32>, GptqError> {
        let groups = self.groups();
        for plane in planes {
            if plane.code.len() != self.out * self.n
                || plane.d.len() != self.out * groups
                || plane.m.len() != self.out * groups
            {
                return Err(GptqError::ShapeMismatch);
            }
        }
        let mut w = vec![0f32; self.out * self.n];
        for plane in planes {
            for o in 0..self.out {
                for j in 0..self.n {
                    let gi = (j / self.group).min(groups - 1);
                    let d = plane.d[o * groups + gi];
                    let m = plane.m[o * groups + gi];
                    w[o * self.n + j] += d * f32::from(plane.code[o * self.n + j]) + m;
                }
            }
        }
        Ok(w)
    }
}

/// One balanced-ternary plane in the Q2 chart: `{0,1,2}` codes with
/// per-(row, group) `(d, m) = (place·s, −place·s)`, so `d·q + m = place·s·(q−1) = place·s·t`.
///
/// Ported from `modelfold/src/quant.rs:77-86`.
#[derive(Clone, Debug, PartialEq)]
pub struct Q2Plane {
    /// Place value `3ᵏ` for plane `k` — how this plane's ternary dot recombines with the others.
    pub place: i64,
    /// Per-(row, group) scale `d = place·s`, row-major `[out, groups]`.
    pub d: Vec<f32>,
    /// Per-(row, group) offset `m = −place·s`, row-major `[out, groups]`.
    pub m: Vec<f32>,
    /// The `{0,1,2}` trit codes (`q = t + 1`), row-major `[out, n]`.
    pub code: Vec<u8>,
}

/// The carrier's asymmetric affine chart — codes `q ∈ 0..=levels`, per-(row, group) `(d, lo)`,
/// dequant `w ≈ d·q + lo`.
///
/// Ported from `modelfold/src/quant.rs:402-416`.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineChart {
    /// Codes `q ∈ 0..=levels`, row-major `[out, n]`.
    pub codes: Vec<u8>,
    /// Per-(row, group) step `d`, row-major `[out, groups]`.
    pub d: Vec<f32>,
    /// Per-(row, group) offset `lo`, row-major `[out, groups]`.
    pub lo: Vec<f32>,
    /// Dequantized `Ŵ = d·q + lo`, row-major `[out, n]` — the CPU golden the GPU chart matches.
    pub wq: Vec<f32>,
    /// Output rows.
    pub out: usize,
    /// Contraction width (columns).
    pub n: usize,
    /// Group (block) length along `n`.
    pub group: usize,
    /// Quantization level ceiling.
    pub levels: u32,
}

/// Quantize `w` (`[out, n]`, row-major) to `levels` balanced trits with GPTQ error-feedback over
/// the calibration Gram `gram` (`[n, n]`, `= XᵀX`). `group` is the scale/block granularity along `n`.
///
/// Ported from `modelfold/src/quant.rs:93-147`.
///
/// # Complexity
/// - Temporal: $O(n^3 + \text{out} \cdot n^2)$ arithmetic operations (FLOPs).
///   Measure: floating-point operations. Derivation: $O(n^3)$ operations in `feedback_hi` for the
///   damped `UL` Cholesky factor $V$ (`H = V Vᵀ`) and its inverse $U = V^{-1}$; then for each of
///   $\text{out}$ rows, error residual push onto the remaining $(n - j)$ columns performs
///   $\sum_{j=0}^{n-1}(n-j) = n(n+1)/2$ multiply-subtracts per row, closing in
///   $O(\text{out} \cdot n^2)$ operations.
/// - Spatial: $O(n^2 + \text{out} \cdot n + \text{out} \cdot \text{groups})$ words allocated.
///   Measure: heap bytes across Hessian buffers ($n \times n \times 8$ bytes), working float buffer
///   `wk` ($\text{out} \cdot n \times 8$ bytes), `wq` ($\text{out} \cdot n \times 4$ bytes), `codes`
///   ($\text{out} \cdot n$ bytes), and `scale` ($\text{out} \cdot \text{groups} \times 4$ bytes).
///
/// # Errors
/// Returns [`GptqError::ShapeMismatch`] if `out == 0`, `n == 0`, `w.len() != out * n`, or `gram.len() != n * n`.
/// Returns [`GptqError::ZeroGroup`] if `group == 0`.
/// Returns [`GptqError::ZeroLevels`] if `levels == 0`.
/// Returns [`GptqError::NotPositiveDefinite`] if Cholesky factorization encounters a non-positive pivot.
pub fn gptq_ternary(
    w: &[f32],
    out: usize,
    n: usize,
    gram: &[f64],
    levels: u32,
    group: usize,
) -> Result<Quant, GptqError> {
    if out == 0 || n == 0 || w.len() != out * n || gram.len() != n * n {
        return Err(GptqError::ShapeMismatch);
    }
    if group == 0 {
        return Err(GptqError::ZeroGroup);
    }
    if levels == 0 {
        return Err(GptqError::ZeroLevels);
    }
    let maxlev = ((3i64.pow(levels) - 1) / 2) as f64;
    let groups = n.div_ceil(group);
    let hi = feedback_hi(gram, n)?;

    let mut wk: Vec<f64> = w.iter().map(|&x| f64::from(x)).collect();
    let mut codes = vec![0i8; out * n];
    let mut scale = vec![0f32; out * groups];
    let mut wq = vec![0f32; out * n];
    let mut g0 = 0;
    let mut gi = 0;
    while g0 < n {
        let g1 = (g0 + group).min(n);
        // per-row scale over this group of the CURRENT (error-fed) W — max-abs / maxlev.
        for o in 0..out {
            let mx = (g0..g1).map(|j| wk[o * n + j].abs()).fold(0.0f64, f64::max);
            let s = (mx / maxlev).max(1e-12);
            scale[o * groups + gi] = s as f32;
            for j in g0..g1 {
                let hijj = hi[j * n + j];
                let wv = wk[o * n + j];
                let c = (wv / s).round().clamp(-maxlev, maxlev);
                let qv = c * s;
                codes[o * n + j] = c as i8;
                wq[o * n + j] = qv as f32;
                // push the residual onto columns j.. through row j of U.
                let err = (wv - qv) / hijj;
                for k in j..n {
                    wk[o * n + k] -= err * hi[j * n + k];
                }
            }
        }
        g0 = g1;
        gi += 1;
    }
    Ok(Quant {
        codes,
        scale,
        wq,
        out,
        n,
        group,
        levels,
    })
}

/// [`gptq_ternary`] at `levels = 1` with per-group scale = **absmean** instead of `absmax/maxlev`.
/// With one trit, absmax zeroes every weight under half the group's largest (heavy-tailed rows
/// collapse to ~all-zero); absmean preserves the mass.
///
/// Ported from `modelfold/src/quant.rs:154-197`.
///
/// # Complexity
/// - Temporal: $O(n^3 + \text{out} \cdot n^2)$ arithmetic operations (FLOPs).
///   Measure: floating-point operations. Derivation: $O(n^3)$ operations in `feedback_hi`; group
///   absmean takes $\sum (g_1 - g_0) = n$ additions per row ($O(\text{out} \cdot n)$); column error-push
///   takes $n(n+1)/2$ multiply-subtracts per row, closing in $O(\text{out} \cdot n^2)$.
/// - Spatial: $O(n^2 + \text{out} \cdot n + \text{out} \cdot \text{groups})$ words allocated.
///   Measure: heap allocations identical in footprint to [`gptq_ternary`].
///
/// # Errors
/// Returns [`GptqError::ShapeMismatch`] if `out == 0`, `n == 0`, `w.len() != out * n`, or `gram.len() != n * n`.
/// Returns [`GptqError::ZeroGroup`] if `group == 0`.
/// Returns [`GptqError::NotPositiveDefinite`] if Cholesky factorization encounters a non-positive pivot.
pub fn gptq_ternary_absmean(
    w: &[f32],
    out: usize,
    n: usize,
    gram: &[f64],
    group: usize,
) -> Result<Quant, GptqError> {
    if out == 0 || n == 0 || w.len() != out * n || gram.len() != n * n {
        return Err(GptqError::ShapeMismatch);
    }
    if group == 0 {
        return Err(GptqError::ZeroGroup);
    }
    let groups = n.div_ceil(group);
    let hi = feedback_hi(gram, n)?;
    let mut wk: Vec<f64> = w.iter().map(|&x| f64::from(x)).collect();
    let mut codes = vec![0i8; out * n];
    let mut scale = vec![0f32; out * groups];
    let mut wq = vec![0f32; out * n];
    let mut g0 = 0;
    let mut gi = 0;
    while g0 < n {
        let g1 = (g0 + group).min(n);
        for o in 0..out {
            let mean = (g0..g1).map(|j| wk[o * n + j].abs()).sum::<f64>() / (g1 - g0) as f64;
            let s = mean.max(1e-12);
            scale[o * groups + gi] = s as f32;
            for j in g0..g1 {
                let hijj = hi[j * n + j];
                let wv = wk[o * n + j];
                let c = (wv / s).round().clamp(-1.0, 1.0);
                let qv = c * s;
                codes[o * n + j] = c as i8;
                wq[o * n + j] = qv as f32;
                let err = (wv - qv) / hijj;
                for k in j..n {
                    wk[o * n + k] -= err * hi[j * n + k];
                }
            }
        }
        g0 = g1;
        gi += 1;
    }
    Ok(Quant {
        codes,
        scale,
        wq,
        out,
        n,
        group,
        levels: 1,
    })
}

/// Naive round-to-trit — the same per-group scale, NO error-feedback. The baseline GPTQ beats;
/// the gap between this and [`gptq_ternary`] is exactly the error-feedback's worth.
///
/// Ported from `modelfold/src/quant.rs:202-237`.
///
/// # Complexity
/// - Temporal: $O(\text{out} \cdot n)$ comparisons and arithmetic operations.
///   Measure: comparisons and float operations. Derivation: iterates over $\text{out}$ rows and group
///   blocks; finding max-abs takes $\sum (g_1 - g_0) = n$ steps and rounding takes $n$ steps per row,
///   closing in $2 \cdot \text{out} \cdot n = O(\text{out} \cdot n)$ operations.
/// - Spatial: $O(\text{out} \cdot n + \text{out} \cdot \text{groups})$ words allocated.
///   Measure: heap allocations for `codes` ($\text{out} \cdot n$ bytes), `scale`
///   ($\text{out} \cdot \text{groups} \times 4$ bytes), and `wq` ($\text{out} \cdot n \times 4$ bytes).
///
/// # Errors
/// Returns [`GptqError::ShapeMismatch`] if `out == 0`, `n == 0`, or `w.len() != out * n`.
/// Returns [`GptqError::ZeroGroup`] if `group == 0`.
/// Returns [`GptqError::ZeroLevels`] if `levels == 0`.
pub fn round_ternary(
    w: &[f32],
    out: usize,
    n: usize,
    levels: u32,
    group: usize,
) -> Result<Quant, GptqError> {
    if out == 0 || n == 0 || w.len() != out * n {
        return Err(GptqError::ShapeMismatch);
    }
    if group == 0 {
        return Err(GptqError::ZeroGroup);
    }
    if levels == 0 {
        return Err(GptqError::ZeroLevels);
    }
    let maxlev = ((3i64.pow(levels) - 1) / 2) as f64;
    let groups = n.div_ceil(group);
    let mut codes = vec![0i8; out * n];
    let mut scale = vec![0f32; out * groups];
    let mut wq = vec![0f32; out * n];
    for o in 0..out {
        let mut g0 = 0;
        let mut gi = 0;
        while g0 < n {
            let g1 = (g0 + group).min(n);
            let mx = (g0..g1)
                .map(|j| f64::from(w[o * n + j]).abs())
                .fold(0.0f64, f64::max);
            let s = (mx / maxlev).max(1e-12);
            scale[o * groups + gi] = s as f32;
            for j in g0..g1 {
                let c = (f64::from(w[o * n + j]) / s).round().clamp(-maxlev, maxlev);
                codes[o * n + j] = c as i8;
                wq[o * n + j] = (c * s) as f32;
            }
            g0 = g1;
            gi += 1;
        }
    }
    Ok(Quant {
        codes,
        scale,
        wq,
        out,
        n,
        group,
        levels,
    })
}

/// GPTQ error-feedback into the carrier's asymmetric affine chart (`w ≈ d·q + lo`, `q ∈ 0..=levels`),
/// over the calibration Gram `gram` (`[n, n]`). The per-group `(lo, d)` is a min/max fit of the
/// CURRENT (error-fed) block — the SAME chart the naive pack builds, but each column's residual is
/// pushed onto the remaining columns through `H⁻¹`, so the codes minimize the OUTPUT distortion.
///
/// Ported from `modelfold/src/quant.rs:426-487`.
///
/// # Complexity
/// - Temporal: $O(n^3 + \text{out} \cdot n^2)$ arithmetic operations (FLOPs).
///   Measure: floating-point operations. Derivation: $O(n^3)$ operations in `feedback_hi`; for each of
///   $\text{out}$ rows and group blocks, min/max fit takes $O(n)$ operations, and error-feedback column
///   update pushes error onto $(n - j)$ remaining columns taking $n(n+1)/2$ multiply-subtracts per row,
///   closing in $O(\text{out} \cdot n^2)$.
/// - Spatial: $O(n^2 + \text{out} \cdot n + \text{out} \cdot \text{groups})$ words allocated.
///   Measure: heap allocations across Hessian buffers ($n \times n \times 8$ bytes), working float buffer
///   `wk` ($\text{out} \cdot n \times 8$ bytes), `wq` ($\text{out} \cdot n \times 4$ bytes), `codes`
///   ($\text{out} \cdot n$ bytes), and `d`/`lo` scale vectors ($2 \times \text{out} \cdot \text{groups} \times 4$ bytes).
///
/// # Errors
/// Returns [`GptqError::ShapeMismatch`] if `out == 0`, `n == 0`, `w.len() != out * n`, or `gram.len() != n * n`.
/// Returns [`GptqError::ZeroGroup`] if `group == 0`.
/// Returns [`GptqError::ZeroLevels`] if `levels == 0`.
/// Returns [`GptqError::NotPositiveDefinite`] if Cholesky factorization encounters a non-positive pivot.
pub fn gptq_affine(
    w: &[f32],
    out: usize,
    n: usize,
    gram: &[f64],
    levels: u32,
    group: usize,
) -> Result<AffineChart, GptqError> {
    if out == 0 || n == 0 || w.len() != out * n || gram.len() != n * n {
        return Err(GptqError::ShapeMismatch);
    }
    if group == 0 {
        return Err(GptqError::ZeroGroup);
    }
    if levels == 0 {
        return Err(GptqError::ZeroLevels);
    }
    let lvl = f64::from(levels);
    let groups = n.div_ceil(group);
    let hi = feedback_hi(gram, n)?;

    let mut wk: Vec<f64> = w.iter().map(|&x| f64::from(x)).collect();
    let mut codes = vec![0u8; out * n];
    let mut ds = vec![0f32; out * groups];
    let mut los = vec![0f32; out * groups];
    let mut wq = vec![0f32; out * n];
    let mut g0 = 0;
    let mut gi = 0;
    while g0 < n {
        let g1 = (g0 + group).min(n);
        for o in 0..out {
            // (lo, d) = asymmetric fit of THIS block of the current (error-fed) row.
            let lo = (g0..g1)
                .map(|j| wk[o * n + j])
                .fold(f64::INFINITY, f64::min);
            let hg = (g0..g1)
                .map(|j| wk[o * n + j])
                .fold(f64::NEG_INFINITY, f64::max);
            let d = if hg > lo { (hg - lo) / lvl } else { 1.0 };
            ds[o * groups + gi] = d as f32;
            los[o * groups + gi] = lo as f32;
            for j in g0..g1 {
                let hijj = hi[j * n + j];
                let wv = wk[o * n + j];
                let q = ((wv - lo) / d).round().clamp(0.0, lvl);
                let qv = d * q + lo;
                codes[o * n + j] = q as u8;
                wq[o * n + j] = qv as f32;
                let err = (wv - qv) / hijj;
                for k in j..n {
                    wk[o * n + k] -= err * hi[j * n + k];
                }
            }
        }
        g0 = g1;
        gi += 1;
    }
    Ok(AffineChart {
        codes,
        d: ds,
        lo: los,
        wq,
        out,
        n,
        group,
        levels,
    })
}

/// Quantize weight matrix `w` (`[out, n]`, row-major) to `ggml` `Q4_K` super-blocks with GPTQ
/// error-feedback over the calibration Gram `gram` (`[n, n]`, `= XᵀX`), with the `τ`-weighted
/// block fit from [`quantize_row_q4_k`] as the rounding operator.
///
/// Follows the error-feedback structure of [`gptq_affine`]: for each 256-column super-block, when its
/// first column is reached, the Q4_K super-block parameters (`d`, `dmin`, 6-bit sub-scales and mins)
/// are fit from the CURRENT error-updated weights using [`quantize_row_q4_k`]. Then, column-by-column,
/// each weight is quantized to `q = clamp(round((w + m_sub) / d_sub), 0, 15)`, producing dequantized
/// `x̂ = d_sub·q − m_sub`, residual error `err = (w − x̂) / U[j,j]`, and the error is pushed onto all
/// subsequent columns `k > j` through row `j` of the upper Cholesky factor `U`.
///
/// Output is packed into `ggml` `Q4_K` super-block layout (144 bytes per 256 weights,
/// `out * (n / 256) * 144` bytes total), readable by [`crate::format::quant::dequant_q4_k_block`].
///
/// # Complexity
/// - Temporal: $O(n^3 + \text{out} \cdot n^2)$ arithmetic operations (FLOPs).
///   Measure: floating-point operations. Derivation: $O(n^3)$ operations in `feedback_hi` for the
///   damped `UL` Cholesky factor and its inverse $U$; for each of $\text{out}$ rows, $n / 256$ super-blocks
///   are fit via `quantize_row_q4_k` taking $O(n)$ operations per row; then column-by-column residual
///   push performs $\sum_{j=0}^{n-1} (n - 1 - j) = n(n-1)/2$ multiply-subtracts per row, closing in
///   $O(\text{out} \cdot n^2)$ arithmetic operations.
/// - Spatial: $O(n^2 + \text{out} \cdot n)$ words allocated.
///   Measure: heap allocations across Hessian buffers ($n \times n \times 8$ bytes), working float buffer
///   `wk` ($\text{out} \cdot n \times 8$ bytes), and the packed output buffer ($144 \times \text{out} \cdot n / 256$ bytes).
///
/// # Errors
/// Returns [`GptqError::ShapeMismatch`] if `out == 0`, `n == 0`, `w.len() != out * n`, `gram.len() != n * n`,
/// `n % 256 != 0`, or if `tau` is supplied with `tau.len() != n`.
/// Returns [`GptqError::NotPositiveDefinite`] if Cholesky factorization encounters a non-positive pivot.
pub fn gptq_q4k(
    w: &[f32],
    out: usize,
    n: usize,
    gram: &[f64],
    tau: Option<&[f32]>,
) -> Result<Vec<u8>, GptqError> {
    gptq_q4k_damped(w, out, n, gram, tau, 0.01)
}

/// [`gptq_q4k`] with an explicit Hessian damping fraction: the feedback factor is built from
/// `H = G + damping·mean(diag(G))·I` instead of the production `0.01·mean(diag(G))`. `damping =
/// 0.01` reproduces [`gptq_q4k`] exactly. `seal_model --gram` defaults to `damping = 1.0`, the
/// lowest bits/token of {0.01, 0.1, 1.0, 10.0} on a held-out text, and the only one measured
/// beyond its paired error against the no-Gram seal.
///
/// # Errors
/// Returns [`GptqError::ShapeMismatch`] if `out == 0`, `n == 0`, `w.len() != out * n`,
/// `gram.len() != n * n`, `n % 256 != 0`, or if `tau` is supplied with `tau.len() != n`.
/// Returns [`GptqError::NotPositiveDefinite`] if the damped Cholesky factorization encounters a
/// non-positive pivot.
pub fn gptq_q4k_damped(
    w: &[f32],
    out: usize,
    n: usize,
    gram: &[f64],
    tau: Option<&[f32]>,
    damping: f64,
) -> Result<Vec<u8>, GptqError> {
    if out == 0 || n == 0 || w.len() != out * n || gram.len() != n * n || !n.is_multiple_of(QK_K) {
        return Err(GptqError::ShapeMismatch);
    }
    if let Some(t) = tau {
        if t.len() != n {
            return Err(GptqError::ShapeMismatch);
        }
    }
    let hi = feedback_hi_damped(gram, n, damping)?;
    let n_blocks = n / QK_K;
    let mut out_bytes = vec![0u8; out * n_blocks * Q4_K_BLOCK_BYTES];
    let mut wk: Vec<f64> = w.iter().map(|&x| f64::from(x)).collect();

    let mut cur_slice = [0f32; QK_K];
    let mut codes = [0u8; QK_K];
    let mut sc12 = [0u8; Q4_K_SCALE_SIZE];

    for o in 0..out {
        let row_base = o * n;
        let mut g0 = 0;
        let mut bi = 0;
        while g0 < n {
            let g1 = g0 + QK_K;
            for (idx, slot) in cur_slice.iter_mut().enumerate() {
                *slot = wk[row_base + g0 + idx] as f32;
            }
            let tau_slice = tau.map(|t| &t[g0..g1]);
            let fit_block =
                quantize_row_q4_k(&cur_slice, tau_slice).map_err(|_| GptqError::ShapeMismatch)?;

            let d_bits = u16::from_le_bytes([fit_block[0], fit_block[1]]);
            let m_bits = u16::from_le_bytes([fit_block[2], fit_block[3]]);
            let d = f16_to_f32(d_bits);
            let dmin = f16_to_f32(m_bits);
            sc12.copy_from_slice(&fit_block[4..4 + Q4_K_SCALE_SIZE]);

            for j in g0..g1 {
                let col_in_blk = j - g0;
                let group_idx = col_in_blk / 32;
                let (sc, m) = q4_k_scale_min(&sc12, group_idx);
                let d_sub = f64::from(d * f32::from(sc));
                let m_sub = f64::from(dmin * f32::from(m));
                let hijj = hi[j * n + j];
                let wv = wk[row_base + j];

                let q = if d_sub > 0.0 {
                    ((wv + m_sub) / d_sub).round().clamp(0.0, 15.0) as u8
                } else {
                    0u8
                };
                codes[col_in_blk] = q;
                let x_hat = d_sub * f64::from(q) - m_sub;
                let err = (wv - x_hat) / hijj;
                let hi_row = &hi[j * n + (j + 1)..j * n + n];
                let wk_row = &mut wk[row_base + (j + 1)..row_base + n];
                for (w_slot, &h_val) in wk_row.iter_mut().zip(hi_row) {
                    *w_slot -= err * h_val;
                }
            }

            let blk_out_idx = (o * n_blocks + bi) * Q4_K_BLOCK_BYTES;
            let blk_out = &mut out_bytes[blk_out_idx..blk_out_idx + Q4_K_BLOCK_BYTES];
            blk_out[0..2].copy_from_slice(&d_bits.to_le_bytes());
            blk_out[2..4].copy_from_slice(&m_bits.to_le_bytes());
            blk_out[4..4 + Q4_K_SCALE_SIZE].copy_from_slice(&sc12);

            let qs = &mut blk_out[4 + Q4_K_SCALE_SIZE..];
            for b in 0..4 {
                for l in 0..32 {
                    qs[32 * b + l] = codes[64 * b + l] | (codes[64 * b + 32 + l] << 4);
                }
            }

            g0 = g1;
            bi += 1;
        }
    }
    Ok(out_bytes)
}

/// Relative output distortion `tr[(W−Ŵ) G (W−Ŵ)ᵀ] / tr[W G Wᵀ]` — the τ-metric the forward feels.
///
/// Ported from `modelfold/src/quant.rs:299-320`.
///
/// # Complexity
/// - Temporal: $O(\text{out} \cdot n^2)$ multiply-accumulate operations.
///   Measure: floating-point operations. Derivation: evaluates quadratic form $\sum_{o=0}^{\text{out}-1}
///   \text{row}_o \cdot G \cdot \text{row}_o^T$ twice: once for residual $r = w - w_q$ and once for $w$.
///   Each row vector-matrix-vector product takes $n^2$ multiply-accumulate steps over $n$ columns,
///   evaluated across $\text{out}$ rows, totaling $2 \cdot \text{out} \cdot n^2$ operations.
/// - Spatial: $O(\text{out} \cdot n)$ floats.
///   Measure: allocates a residual buffer `r` of length $\text{out} \times n$ ($\text{out} \cdot n \times 4$ bytes).
///
/// # Errors
/// Returns [`GptqError::ShapeMismatch`] if `out == 0`, `n == 0`, `w.len() != out * n`, `wq.len() != out * n`,
/// or `gram.len() != n * n`.
pub fn output_error(
    w: &[f32],
    wq: &[f32],
    out: usize,
    n: usize,
    gram: &[f64],
) -> Result<f64, GptqError> {
    if out == 0 || n == 0 || w.len() != out * n || wq.len() != out * n || gram.len() != n * n {
        return Err(GptqError::ShapeMismatch);
    }
    let quad = |a: &[f32]| -> f64 {
        let mut tot = 0.0f64;
        for o in 0..out {
            let row = &a[o * n..(o + 1) * n];
            for i in 0..n {
                let ri = f64::from(row[i]);
                if ri == 0.0 {
                    continue;
                }
                let mut s = 0.0f64;
                for j in 0..n {
                    s += gram[i * n + j] * f64::from(row[j]);
                }
                tot += ri * s;
            }
        }
        tot
    };
    let r: Vec<f32> = w.iter().zip(wq).map(|(a, b)| a - b).collect();
    let q_w = quad(w).max(1e-30);
    Ok(quad(&r) / q_w)
}

/// The balanced-ternary digits of `c` (`c = Σ_k t[k]·3ᵏ`, each `t[k] ∈ {−1,0,+1}`), `l` digits.
/// Exact and unique for `|c| ≤ (3^l − 1)/2`.
///
/// Ported from `modelfold/src/quant.rs:324-339`.
///
/// # Complexity
/// - Temporal: $O(l)$ integer division and modulo operations.
///   Measure: integer operations across $l$ loop iterations. Derivation: iterates through $l$ trit
///   positions, computing $c \pmod 3$ and $c / 3$ per step, closing in exactly $l$ iterations.
/// - Spatial: $O(l)$ bytes allocated.
///   Measure: allocates a `Vec<i8>` of length $l$.
fn balanced_trits(mut c: i32, l: usize) -> Vec<i8> {
    let mut t = vec![0i8; l];
    for tk in &mut t {
        let mut r = c % 3;
        c /= 3;
        if r == 2 {
            r = -1;
            c += 1;
        } else if r == -2 {
            r = 1;
            c -= 1;
        }
        *tk = r as i8;
    }
    t
}

/// The error-feedback factor `U` (`Uᵀ U = H⁻¹`, upper-triangular `[n, n]`) from the calibration
/// Gram: `H = G + 0.01·mean(diag)·I` (damped, dead-column guarded). The damping and dead-column
/// guard are unchanged; only the factor path is. `H` is factored as `H = V Vᵀ` by [`ul_cholesky`]
/// (the `UL` factor, from the bottom-right corner), and `U := V⁻¹` by [`upper_tri_inverse`]. Since
/// `Uᵀ U = V⁻ᵀ V⁻¹ = (V Vᵀ)⁻¹ = H⁻¹` and `U` is upper-triangular with a positive diagonal,
/// `U = cholesky_upper(H⁻¹)` by the uniqueness of the positive-diagonal Cholesky factor. This
/// removes the `n`-right-hand-side identity solve the prior path paid.
///
/// Ported from `modelfold/src/quant.rs:358-374`.
///
/// # Complexity
/// - Temporal: $O(n^3)$ floating-point operations.
///   Measure: arithmetic operations. Derivation: regularizing $H$ takes $O(n^2)$ operations;
///   [`ul_cholesky`] takes $n^3/6$ operations (a dot product over the strict upper triangle per row
///   pair); [`upper_tri_inverse`] takes $n^3/6$; the total closes at $\approx n^3/3$, against the
///   prior $\approx 2.7\,n^3$ ($n^3/3$ Cholesky plus $n$ dense right-hand-side solves plus $n^3/3$
///   upper Cholesky of the inverse).
/// - Spatial: $O(n^2)$ f64 words allocated.
///   Measure: heap buffers for the regularized Hessian $H$ ($n^2 \times 8$ bytes), the upper factor
///   $V$ ($n^2 \times 8$ bytes), and the resulting inverse factor $U$ ($n^2 \times 8$ bytes).
fn feedback_hi(gram: &[f64], n: usize) -> Result<Vec<f64>, GptqError> {
    feedback_hi_damped(gram, n, 0.01)
}

/// [`feedback_hi`] with an explicit damping fraction of `mean(diag(G))`; `damping = 0.01`
/// reproduces [`feedback_hi`] exactly.
fn feedback_hi_damped(gram: &[f64], n: usize, damping: f64) -> Result<Vec<f64>, GptqError> {
    if gram.len() != n * n || n == 0 {
        return Err(GptqError::ShapeMismatch);
    }
    let mean_diag: f64 = (0..n).map(|i| gram[i * n + i]).sum::<f64>() / n as f64;
    if mean_diag.is_nan() {
        return Err(GptqError::NotPositiveDefinite);
    }
    let damp = damping * mean_diag;
    let mut h = gram.to_vec();
    for i in 0..n {
        h[i * n + i] += damp;
        if h[i * n + i] == 0.0 {
            h[i * n + i] = 1.0; // a dead input channel ⇒ pin its pivot (no column to feed)
        }
    }
    let v = ul_cholesky(&h, n)?;
    upper_tri_inverse(&v, n)
}

/// The `UL` Cholesky factor `V` (`H = V Vᵀ`, `V` upper-triangular row-major `[n, n]`) of the
/// symmetric positive-definite `h`, built from the bottom-right corner. Column `j` runs from
/// `n-1` down to `0`: the pivot is `V[j,j] = sqrt(H[j,j] − Σ_{k>j} V[j,k]²)`, then each subdiagonal
/// `V[i,j] = (H[i,j] − Σ_{k>j} V[i,k]·V[j,k]) / V[j,j]` for `i < j`. Both sums are dot products over
/// the contiguous row slices `k > j`.
///
/// # Complexity
/// - Temporal: $n^3/6$ floating-point operations.
///   Measure: arithmetic operations. Derivation: column `j` costs `(n-1-j)` multiply-accumulates
///   for the pivot plus `j` subdiagonal dot products of length `n-1-j`, i.e. `(j+1)(n-1-j)`; summing
///   over `j` gives `Σ_j (j+1)(n-1-j) = n(n+1)(n+2)/6`.
/// - Spatial: $O(n^2)$ f64 words allocated.
///   Measure: single `n × n` output buffer `v` of $n^2 \times 8$ bytes.
///
/// # Errors
/// Returns [`GptqError::ShapeMismatch`] if `h.len() != n * n`.
/// Returns [`GptqError::NotPositiveDefinite`] if a diagonal pivot is non-positive or NaN.
pub fn ul_cholesky(h: &[f64], n: usize) -> Result<Vec<f64>, GptqError> {
    if h.len() != n * n {
        return Err(GptqError::ShapeMismatch);
    }
    let mut v = vec![0.0f64; n * n];
    for j in (0..n).rev() {
        let mut s = h[j * n + j];
        for k in (j + 1)..n {
            let vjk = v[j * n + k];
            s -= vjk * vjk;
        }
        if s.is_nan() || s <= 0.0 {
            return Err(GptqError::NotPositiveDefinite);
        }
        let vjj = s.sqrt();
        v[j * n + j] = vjj;
        for i in 0..j {
            // H is symmetric, so H[i,j] = H[j,i] is read contiguously along row j.
            let mut t = h[j * n + i];
            for k in (j + 1)..n {
                t -= v[i * n + k] * v[j * n + k];
            }
            v[i * n + j] = t / vjj;
        }
    }
    Ok(v)
}

/// Inverse `U = V⁻¹` of an upper-triangular `v` (row-major `[n, n]`, positive diagonal), returned
/// row-major upper-triangular. Computed by block back-substitution over `64 × 64` diagonal blocks,
/// which is the GPTQ error-feedback's unit of work.
///
/// # Complexity
/// - Temporal: $n^3/6$ floating-point operations.
///   Measure: arithmetic operations. Derivation: the block back-substitution accumulates
///   `Σ_{i} Σ_{l>i} Σ_{j>=l} 64·64·64` multiply-accumulates of the dense block products `V_il·U_lj`
///   (the `n^3/6` of the triangular inverse), plus `O(n^2·64)` for the diagonal-block inverses and
///   their triangular applications. Blocking keeps the working set of a block product (two `64 × 64`
///   blocks, `64 KiB`) resident in L2 (2 MiB per P-core, i9-14900KF), instead of streaming a
///   `94 MiB` triangle (`n = 4864`) per row sweep from memory.
/// - Spatial: $O(n^2)$ f64 words allocated.
///   Measure: single `n × n` output buffer of $n^2 \times 8$ bytes.
///
/// # Errors
/// Returns [`GptqError::ShapeMismatch`] if `v.len() != n * n`.
/// Returns [`GptqError::NotPositiveDefinite`] if a diagonal entry is non-positive or NaN.
pub fn upper_tri_inverse(v: &[f64], n: usize) -> Result<Vec<f64>, GptqError> {
    const B: usize = 64;
    if v.len() != n * n {
        return Err(GptqError::ShapeMismatch);
    }
    let mut u = vec![0.0f64; n * n];
    let m = n.div_ceil(B);
    for i in (0..m).rev() {
        let i0 = i * B;
        let si = (i0 + B).min(n) - i0;
        // (1) Invert the diagonal block V[i0..i0+si, i0..i0+si] into U.
        for jj in (0..si).rev() {
            let diag = v[(i0 + jj) * n + i0 + jj];
            if diag.is_nan() || diag <= 0.0 {
                return Err(GptqError::NotPositiveDefinite);
            }
            u[(i0 + jj) * n + i0 + jj] = 1.0 / diag;
            for rr in (0..jj).rev() {
                let mut s = 0.0;
                for k in (rr + 1)..=jj {
                    s += v[(i0 + rr) * n + i0 + k] * u[(i0 + k) * n + i0 + jj];
                }
                let d = v[(i0 + rr) * n + i0 + rr];
                if d.is_nan() || d <= 0.0 {
                    return Err(GptqError::NotPositiveDefinite);
                }
                u[(i0 + rr) * n + i0 + jj] = -s / d;
            }
        }
        // (2) Accumulate W_ij = Σ_{l=i+1}^{j} V_il·U_lj into the row block i, for every j > i.
        for l in (i + 1)..m {
            let l0 = l * B;
            let sl = (l0 + B).min(n) - l0;
            for j in l..m {
                let j0 = j * B;
                let sj = (j0 + B).min(n) - j0;
                for a in 0..si {
                    let urow = (i0 + a) * n + j0;
                    let vrow = (i0 + a) * n + l0;
                    for cc in 0..sl {
                        let vv = v[vrow + cc];
                        if vv == 0.0 {
                            continue;
                        }
                        let xrow = (l0 + cc) * n + j0;
                        for b in 0..sj {
                            u[urow + b] += vv * u[xrow + b];
                        }
                    }
                }
            }
        }
        // (3) X_ij = −V_ii⁻¹·W_ij by one triangular back-substitution per column, for every j > i.
        for j in (i + 1)..m {
            let j0 = j * B;
            let sj = (j0 + B).min(n) - j0;
            for c in 0..sj {
                let col = j0 + c;
                let last = i0 + si - 1;
                u[last * n + col] = -u[last * n + col] / v[last * n + last];
                for rr in (0..si - 1).rev() {
                    let mut acc = -u[(i0 + rr) * n + col];
                    for k in (rr + 1)..si {
                        acc -= v[(i0 + rr) * n + i0 + k] * u[(i0 + k) * n + col];
                    }
                    u[(i0 + rr) * n + col] = acc / v[(i0 + rr) * n + i0 + rr];
                }
            }
        }
    }
    Ok(u)
}
