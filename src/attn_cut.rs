//! Certified attention cut: a softmax read over partial sums that carries its own error bound.
//!
//! Statements: `Apeiron.Numeric.AttentionCut` in the `sanguine` corpus (not in this tree;
//! source sha256 `3bbd73592d6ec51f…`). No refinement theorem links this module to them. Grade
//! `tested` only: the Lean statements are over `ℝ`; this module is `f64` arithmetic over `f32` data
//! and adds the rounding of that arithmetic, which `Accumulator.lean`'s layer owns and which nothing
//! here bounds.
//!
//! # Frames
//!
//! Lean states `N = Σ e^{sᵢ}·vᵢ`, `Z = Σ e^{sᵢ}` over unshifted exponentials. A [`Part`] stores
//! the same sums shifted by a frame `m`: `N_m = e^{-m}·N`, `Z_m = e^{-m}·Z`, with its error
//! budgets `εN_m = e^{-m}·εN`, `εZ_m = e^{-m}·εZ`. The three quantities the read uses are
//! homogeneous of degree 0 in that common factor: `N/Z`, and `(εN + V·εZ)/Z'`. A part rebases to
//! a frame `M ≥ m` by multiplying all four sums by `e^{m-M} ≤ 1`, which cannot overflow.
//!
//! # What each function refines
//!
//! | function | Lean theorem | statement used |
//! |---|---|---|
//! | [`Part::exact`] | `partition_error` (exact part) | the part's stand-in equals its sum, `ε = 0` |
//! | [`Part::dropped`] | `dropped_mass_bound`, `num_le` | a dropped set `D` of scores `≤ m−Δ` has `\|num D\| ≤ V·den D` and `den D ≤ \|D\|·e^{m−Δ}`; stand-in `(0,0)`, `εZ = \|D\|·e^{−Δ}`, `εN = V·εZ` in frame `m` |
//! | [`Part::merge`] | `partition_error` | over a disjoint union the errors of the parts add |
//! | [`certified_read`] | `certified_read`, `ratio_error` | `\|N/Z − N'/Z'\| ≤ (εN + V·εZ)/Z'` |
//!
//! # Obligations the caller carries
//!
//! `certified_read` is the theorem `certified_read` with these hypotheses; only the ones marked
//! *checked* are tested at run time.
//!
//! * `0 < Z'`: *checked*, [`CutError::NonPositiveDenominator`] otherwise.
//! * `0 ≤ V`: *checked*, [`CutError::BadValueBound`] otherwise.
//! * `0 < Z`: the true softmax denominator over at least one key, positive for finite scores.
//! * `|N| ≤ V·Z`: every value of every key, kept or dropped, satisfies `|v| ≤ V` componentwise
//!   (Lean `num_le`).
//! * `|N−N'| ≤ εN`, `|Z−Z'| ≤ εZ`: each part's budget covers its true sum (exact parts are
//!   constructed so; a dropped part's `delta`, `vmax` and `mk` must describe its keys).
//! * Disjointness (`partition_error` `hdisj`): each key is in at most one part. Parts carry no key
//!   indices, so [`Part::merge`] cannot check this.

/// Why a certified read is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutError {
    /// No parts were given.
    NoParts,
    /// `Z' ≤ 0` or not finite: Lean's `0 < Z'` fails, the bound is not defined.
    NonPositiveDenominator,
    /// `V < 0` or not finite: Lean's `0 ≤ V` fails.
    BadValueBound,
    /// The bound itself is not finite (budget overflow).
    NonFiniteBound,
}

/// A max-shifted partial softmax sum with its error budgets, all in frame `m`.
///
/// `n[j] = Σ e^{sᵢ−m}·vᵢ[j]`, `z = Σ e^{sᵢ−m}` over the keys the part stands for; the true sums
/// differ by at most `en` per component (max-norm) and `ez`.
#[derive(Debug, Clone, PartialEq)]
pub struct Part {
    /// Shifted numerator stand-in, one entry per value component.
    pub n: Vec<f64>,
    /// Shifted denominator stand-in.
    pub z: f64,
    /// Bound on `|N_j − n_j|` for every component `j`, in frame `m`.
    pub en: f64,
    /// Bound on `|Z − z|`, in frame `m`.
    pub ez: f64,
    /// The frame: all four sums are scaled by `e^{−m}`. `-∞` for the empty part.
    pub m: f64,
}

impl Part {
    /// The empty part of value dimension `dim`: no keys, frame `-∞`, all sums zero.
    #[must_use]
    pub fn empty(dim: usize) -> Self {
        Part {
            n: vec![0.0; dim],
            z: 0.0,
            en: 0.0,
            ez: 0.0,
            m: f64::NEG_INFINITY,
        }
    }

    /// Keys read exactly in frame `m`: `ε = 0` (`partition_error` with an exact part).
    ///
    /// `values` is row-major, `scores.len()` rows of `dim`. Every `sᵢ − m` should be `≤ 0` so no
    /// exponential overflows (not required for correctness).
    ///
    /// Complexity: time `O(len·dim)` multiply-adds plus `len` `exp` calls; space `O(dim)` words.
    /// One pass over the keys, each key touching `dim` accumulators.
    ///
    /// # Panics
    /// If `values.len() != scores.len() * dim`.
    #[must_use]
    pub fn exact(scores: &[f32], values: &[f32], dim: usize, m: f64) -> Self {
        assert_eq!(
            values.len(),
            scores.len() * dim,
            "values must be scores.len() rows of dim"
        );
        let mut p = Part::empty(dim);
        if scores.is_empty() {
            return p;
        }
        p.m = m;
        for (i, &s) in scores.iter().enumerate() {
            let w = (f64::from(s) - m).exp();
            p.z += w;
            for (acc, &v) in p.n.iter_mut().zip(&values[i * dim..(i + 1) * dim]) {
                *acc += w * f64::from(v);
            }
        }
        p
    }

    /// Summary of `count` dropped keys of value dimension `dim`, each with `|v| ≤ vmax` and
    /// score `≤ mk − delta`, in frame `m = mk` (`dropped_mass_bound`).
    ///
    /// Stand-in `(n, z) = (0, 0)`. True `den D ≤ count·e^{mk−Δ}` and `|num D| ≤ vmax·den D`
    /// (`num_le`), so in frame `mk`: `ez = count·e^{−Δ}`, `en = vmax·ez`. `mk` is the score the
    /// kept keys reach, so the merged `Z'` is at least `1` in this frame and the bound is at most
    /// `(vmax + V)·count·e^{−Δ}`.
    ///
    /// Complexity: `O(dim)` words, one `exp`.
    #[must_use]
    pub fn dropped(dim: usize, count: usize, delta: f64, vmax: f64, mk: f64) -> Self {
        if count == 0 {
            return Part::empty(dim);
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "count is a key count, far below 2^53"
        )]
        let ez = count as f64 * (-delta).exp();
        Part {
            n: vec![0.0; dim],
            z: 0.0,
            en: vmax * ez,
            ez,
            m: mk,
        }
    }

    /// Rebase to frame `big ≥ self.m`: scale all four sums by `e^{m−big}` (`≤ 1`).
    fn rebased(&self, big: f64) -> Part {
        if self.m == f64::NEG_INFINITY {
            // empty part: all sums are zero, any frame holds them
            return Part {
                m: big,
                ..self.clone()
            };
        }
        let f = (self.m - big).exp();
        Part {
            n: self.n.iter().map(|x| x * f).collect(),
            z: self.z * f,
            en: self.en * f,
            ez: self.ez * f,
            m: big,
        }
    }

    /// Merge two disjoint parts: rebase both to the larger frame and add numerators,
    /// denominators and budgets (`partition_error`: errors add).
    ///
    /// Disjointness of the key sets is the caller's obligation; parts hold no indices.
    ///
    /// Complexity: time `O(dim)` multiplies and adds, space `O(dim)` words (the result).
    ///
    /// # Panics
    /// If the value dimensions differ.
    #[must_use]
    pub fn merge(&self, other: &Part) -> Part {
        assert_eq!(
            self.n.len(),
            other.n.len(),
            "parts must share a value dimension"
        );
        let big = self.m.max(other.m);
        if big == f64::NEG_INFINITY {
            return self.clone();
        }
        let (a, b) = (self.rebased(big), other.rebased(big));
        Part {
            n: a.n.iter().zip(&b.n).map(|(x, y)| x + y).collect(),
            z: a.z + b.z,
            en: a.en + b.en,
            ez: a.ez + b.ez,
            m: big,
        }
    }
}

/// Read the softmax output `N/Z` from `parts` with its certified max-norm error bound
/// `(εN + V·εZ)/Z'` (Lean `certified_read`, applied per component with the shared `εN`).
///
/// `v` is the bound `V` with `|v| ≤ V` for every key's value, kept or dropped. Hypotheses as in
/// the module header: `Z' > 0` and `V ≥ 0` are checked; the rest are the caller's.
///
/// Complexity: `parts.len() − 1` merges, so time `O(parts.len()·dim)` and space `O(dim)` words
/// (the running merge). The merge is a left fold; each step is `O(dim)`.
///
/// # Errors
/// [`CutError`] when `parts` is empty, `Z' ≤ 0` or non-finite, `V` is negative or non-finite, or
/// the bound overflows. The failure branch returns no number, never a NaN.
pub fn certified_read(parts: &[Part], v: f64) -> Result<(Vec<f64>, f64), CutError> {
    let (first, rest) = parts.split_first().ok_or(CutError::NoParts)?;
    if !(v.is_finite() && v >= 0.0) {
        return Err(CutError::BadValueBound);
    }
    let total = rest.iter().fold(first.clone(), |acc, p| acc.merge(p));
    if !(total.z.is_finite() && total.z > 0.0) {
        return Err(CutError::NonPositiveDenominator);
    }
    let bound = (total.en + v * total.ez) / total.z;
    if !bound.is_finite() {
        return Err(CutError::NonFiniteBound);
    }
    Ok((total.n.iter().map(|x| x / total.z).collect(), bound))
}
