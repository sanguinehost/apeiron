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
///   Cholesky solve forming $H^{-1}$ and its upper Cholesky factor $U$; then for each of $\text{out}$
///   rows, error residual push onto the remaining $(n - j)$ columns performs $\sum_{j=0}^{n-1}(n-j) =
///   n(n+1)/2$ multiply-subtracts per row, closing in $O(\text{out} \cdot n^2)$ operations.
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
/// Gram: `H = G + 0.01·mean(diag)·I` (damped, dead-column guarded), `H⁻¹` via [`crate::learn::ridge::solve_spd`],
/// then its upper Cholesky factor via [`cholesky_upper`].
///
/// Ported from `modelfold/src/quant.rs:358-374`.
///
/// # Complexity
/// - Temporal: $O(n^3)$ floating-point operations.
///   Measure: arithmetic operations. Derivation: regularizing $H$ takes $O(n^2)$ operations;
///   solving $H \cdot X = I$ via `solve_spd` takes $O(n^3)$ ($n^3/3$ Cholesky factor plus $n$ solves
///   each taking $n^2$ substitution steps); upper Cholesky decomposition of $H^{-1}$ takes $O(n^3/3)$
///   operations via `cholesky_upper`. Total closes in $O(n^3)$.
/// - Spatial: $O(n^2)$ f64 words allocated.
///   Measure: heap buffers for regularized Hessian $H$ ($n^2 \times 8$ bytes), identity matrix ($n^2 \times 8$ bytes),
///   $H^{-1}$ solution ($n^2 \times 8$ bytes), and resulting upper factor $U$ ($n^2 \times 8$ bytes).
fn feedback_hi(gram: &[f64], n: usize) -> Result<Vec<f64>, GptqError> {
    if gram.len() != n * n || n == 0 {
        return Err(GptqError::ShapeMismatch);
    }
    let mean_diag: f64 = (0..n).map(|i| gram[i * n + i]).sum::<f64>() / n as f64;
    if mean_diag.is_nan() {
        return Err(GptqError::NotPositiveDefinite);
    }
    let damp = 0.01 * mean_diag;
    let mut h = gram.to_vec();
    for i in 0..n {
        h[i * n + i] += damp;
        if h[i * n + i] == 0.0 {
            h[i * n + i] = 1.0; // a dead input channel ⇒ pin its pivot (no column to feed)
        }
    }
    let mut ident = vec![0.0f64; n * n];
    for i in 0..n {
        ident[i * n + i] = 1.0;
    }
    let hinv = crate::learn::ridge::solve_spd(&h, &ident, n, n)
        .map_err(|_| GptqError::NotPositiveDefinite)?;
    cholesky_upper(&hinv, n)
}

/// Upper Cholesky `U` (`Uᵀ U = a`, `U` upper-triangular row-major `[d, d]`) of symmetric PD `a`.
/// GPTQ-specific: the error-feedback needs the UPPER factor of `H⁻¹` (column-elimination order).
///
/// Ported from `modelfold/src/quant.rs:379-397`.
///
/// # Complexity
/// - Temporal: $O(d^3 / 3)$ arithmetic operations.
///   Measure: floating-point operations. Derivation: outer loop $j \in [0, d)$ performs $j$
///   multiply-accumulates to compute diagonal pivot $s = a[j,j] - \sum_{k<j} u_{k,j}^2$; inner loop
///   over $i \in [j+1, d)$ performs $j$ multiply-accumulates per column. Summing over all $j$ yields
///   $\sum_{j=0}^{d-1} [j + j(d - 1 - j)] = d^3 / 3$ operations.
/// - Spatial: $O(d^2)$ f64 words allocated.
///   Measure: single $d \times d$ buffer `u` of $d^2 \times 8$ bytes.
///
/// # Errors
/// Returns [`GptqError::ShapeMismatch`] if `a.len() != d * d`.
/// Returns [`GptqError::NotPositiveDefinite`] if any diagonal pivot is non-positive or NaN.
fn cholesky_upper(a: &[f64], d: usize) -> Result<Vec<f64>, GptqError> {
    if a.len() != d * d {
        return Err(GptqError::ShapeMismatch);
    }
    let mut u = vec![0.0f64; d * d];
    for j in 0..d {
        let mut s = a[j * d + j];
        for k in 0..j {
            s -= u[k * d + j] * u[k * d + j];
        }
        if s.is_nan() || s <= 0.0 {
            return Err(GptqError::NotPositiveDefinite);
        }
        let ujj = s.sqrt();
        u[j * d + j] = ujj;
        for i in (j + 1)..d {
            let mut s = a[j * d + i];
            for k in 0..j {
                s -= u[k * d + j] * u[k * d + i];
            }
            u[j * d + i] = s / ujj;
        }
    }
    Ok(u)
}
