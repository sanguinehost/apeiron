//! Top-`r` eigenpairs of a symmetric PSD Q32.32 `i64` matrix by block subspace iteration with a
//! Rayleigh–Ritz acceleration — the primitive the low-rank corrector needs in place of the full
//! `O(n³)`-per-sweep Jacobi [`eigh`](crate::learn::fold::fit::eigh::eigh) (`31.1 s` at `n = 896`).
//! The corrector consumes only the top `rank` (`1..=8`) eigenvectors of `M = (ΔW·Hl)ΔWᵀ` (`o × o`,
//! `o = 896`) and `Σ_k max(λ_k, 0)` over them (`src/learn/fold/fit/corrector.rs:255-300`); the
//! full spectrum is never needed.
//!
//! # Method
//!
//! Block subspace iteration on an `n × s` orthonormal basis `Q`, `s = r + p` with the oversampling
//! `p = min(8, n − r)`:
//!
//! 1. `Y = M·Q` through [`gemm_nn`] (one blocked, threaded Q32.32 product per iteration).
//! 2. Rayleigh–Ritz on the current basis: `T = Qᵀ·Y` (`s × s`, exact `i128` accumulation, one
//!    round-to-nearest per entry), diagonalised by the integer Jacobi [`jacobi`] — rotations from
//!    exact integer arithmetic (`t = sign(θ)/(|θ| + √(1+θ²))` with [`isqrt`]), no float.
//! 3. `Q' = Q·V`, `Y' = Y·V` for the rotation basis `V` (so `Y' = M·Q'` stays exact), and the
//!    residual `‖M v_k − λ_k v_k‖∞` over the top `r` columns is the stop test.
//! 4. The power step `Q ← orthonormalise(Y')`, which also resets the orthonormality drift of the
//!    rotation product every iteration.
//!
//! Convergence is linear with the oversampled ratio `λ_{s+1}/λ_r` per iteration (`s = r + p`); the
//! extra `p` columns carry the gap that makes the kept `r` converge, and are dropped at the end.
//! The orthonormalisation is modified Gram–Schmidt with one re-orthogonalisation pass in `i128`
//! (each column normalised before it is used by the next, so the projection coefficient is exact
//! against a unit column). The measured worst case of `‖QᵀQ − I‖max` over the tested shapes
//! (`n ∈ {8, 64, 130}`, `r ∈ {1, 2, 4}`; the print in `tests/learn_solve_eig.rs`) is `1.1e-9`,
//! asserted at `2⁻²⁴`: below the fixed-point floor, so the Cholesky-QR route (whose factor `L` is
//! private to [`super::chol`], and whose `chol_solve` returns `G⁻¹` rather than `L`) buys nothing
//! here.
//!
//! # Stop test and partiality
//!
//! The stop test is the max column residual over the kept `r` columns against `tol = λ_max·2⁻²⁰`
//! ([`RESIDUAL_SHIFT`]) or [`MAX_ITERS`] iterations. The residual is re-checked on the *returned*
//! (orthonormalised, sign-fixed-to-be) columns through a fresh [`gemm_nn`] image before the result
//! is handed back, so the returned object is what was verified. Non-convergence is
//! [`EigError::NotConverged`] carrying the last residual, never a silently wrong answer.
//!
//! # Accepted domain
//!
//! `M` is `n × n` row-major Q32.32, symmetric to within one raw unit, PSD, with every entry below
//! `2³⁶` in magnitude (real `16`). The power step needs `rank(M) ≥ s`: the plane corrector's `M`
//! has rank `o`, and a lower-rank `M` is refused with [`EigError::DegenerateBasis`] rather than
//! answered with a null subspace.
//!
//! Grade: `tested` (`tests/learn_solve_eig.rs`); no body here is extracted or refined.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::must_use_candidate,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the narrowing `as i64` appears only inside `narrow`, which range-checks against \
              `i64::MAX`/`i64::MIN` immediately before the cast; the single-letter names are the \
              arithmetic's own index symbols (`n`, `r`, `s`, `p`, `i`, `j`, `k`, `l`, `q`, `v`, \
              `y`, `t`), which read closer to the recurrence than spelled-out names. The `Result` \
              returns are already `#[must_use]`."
)]

use super::gemm::{gemm_nn, GemmError};
use super::syrk::{narrow, round32, within_bound};
use crate::numeric::fixed_rsqrt::isqrt;

/// The loop-safe form of `?` on a checked `Option`: yield `$o`'s value, or record
/// [`EigError::Overflow`] in the `$err` flag and yield `0`. The enclosing loop tests `$err`, so no
/// early `return` or `?` sits inside a loop body — the shape the pinned Aeneas accepts.
macro_rules! ck {
    ($err:ident, $o:expr) => {
        match $o {
            Some(v) => v,
            None => {
                $err = Some(EigError::Overflow);
                0
            }
        }
    };
}

/// As [`ck`], for a `Result` carrying an [`EigError`].
macro_rules! ckr {
    ($err:ident, $r:expr) => {
        match $r {
            Ok(v) => v,
            Err(e) => {
                $err = Some(e);
                0
            }
        }
    };
}

/// Raw Q32.32 `1.0`.
const ONE: i64 = 1 << 32;
/// Oversampling columns beyond `r`: `p = min(OVERSAMPLE, n − r)` carries the gap
/// `λ_{r+p+1}/λ_r` that sets the per-iteration convergence rate of the kept `r`.
const OVERSAMPLE: usize = 8;
/// The stop-test exponent: the residual tolerance is `λ_max · 2⁻RESIDUAL_SHIFT` in raw Q32.32.
const RESIDUAL_SHIFT: u32 = 20;
/// The subspace-iteration ceiling; reaching it returns [`EigError::NotConverged`].
pub const MAX_ITERS: usize = 60;
/// The seed of the deterministic start basis (a fixed xorshift64, the module's only randomness).
const START_SEED: u64 = 0x2545_f491_4f6c_dd1d;
/// Sweep ceiling of the `s × s` integer Jacobi ([`jacobi`]).
const JACOBI_SWEEPS: usize = 40;
/// Off-diagonal convergence exponent of [`jacobi`], relative to the diagonal scale.
const OFF_DIAG_SHIFT: u32 = 26;
/// Off-diagonal convergence floor of [`jacobi`], in raw Q32.32 units (the rounding floor).
const OFF_DIAG_FLOOR: u32 = 4;

/// A refusal from the top-`r` eigenpair extraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EigError {
    /// A slice length disagrees with the declared `n × n` shape.
    ShapeMismatch,
    /// `n == 0`, so the matrix has no entries.
    Empty,
    /// `r == 0`: the corrector's `rank = 0` reading is the empty corrector, not an eigenproblem.
    RankZero,
    /// `r > n`: more eigenpairs were asked for than the matrix has.
    RankOutOfRange,
    /// An entry of `M` reaches the documented magnitude bound `2³⁶`.
    InputBound,
    /// `M` is not symmetric beyond one raw unit (`|M_ij − M_ji| > 1`).
    NotSymmetric,
    /// A checked `i128` operation left its range, an `i128 → i64` narrowing did not fit, or a
    /// dimension product left `usize`.
    Overflow,
    /// The power step mapped the start `s`-subspace to a rank-deficient set, so `rank(M) < s` and
    /// the oversampled block cannot carry an `s`-dimensional basis.
    DegenerateBasis,
    /// [`MAX_ITERS`] iterations ran without the residual falling to the tolerance; carries the last
    /// max column residual, in raw Q32.32.
    NotConverged {
        /// The last max column residual `‖M v_k − λ_k v_k‖∞` over the kept columns, raw Q32.32.
        residual: i64,
    },
}

/// The top `r` eigenpairs of a symmetric PSD Q32.32 matrix, ascending.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopEig {
    /// The `r` largest eigenvalues, ascending in raw Q32.32 (entry `r − 1` is the largest).
    pub evals: Vec<i64>,
    /// The `r` eigenvectors, row-major `n × r` in raw Q32.32, column `k` the eigenvector for
    /// `evals[k]` (unit 2-norm, the largest-magnitude entry of each column positive).
    pub evecs: Vec<i64>,
    /// The subspace iterations run.
    pub iters: usize,
}

/// Lift a checked `Option` to [`EigError::Overflow`].
fn ov<T>(o: Option<T>) -> Result<T, EigError> {
    match o {
        Some(v) => Ok(v),
        None => Err(EigError::Overflow),
    }
}

/// [`round32`] (round-to-nearest of an `i128` Q64.64 value to Q32.32) lifted to
/// [`EigError::Overflow`].
fn r32(v: i128) -> Result<i128, EigError> {
    ov(round32(v))
}

/// Signed `round(v / d)`, nearest with ties away from zero, or `None` if an intermediate or the
/// result leaves `i128`.
fn div_round(v: i128, d: i128) -> Option<i128> {
    if d == 0 {
        return None;
    }
    let half = d.unsigned_abs() / 2;
    if v >= 0 {
        match v.checked_add(i128::try_from(half).ok()?) {
            Some(n) => n.checked_div(d),
            None => None,
        }
    } else {
        match v.checked_sub(i128::try_from(half).ok()?) {
            Some(n) => n.checked_div(d),
            None => None,
        }
    }
}

/// `[M]i128` narrowing with [`EigError::Overflow`] on a value outside `i64`.
fn to_i64(v: i128) -> Result<i64, EigError> {
    ov(narrow(v))
}

/// The `n × n` row-major `M` is symmetric to within one raw unit.
///
/// # Complexity
/// `n(n−1)/2` comparisons in `i128`, `O(1)` space.
fn is_symmetric(m: &[i64], n: usize) -> bool {
    let mut ok = true;
    let mut i = 0;
    while i < n && ok {
        let mut j = i + 1;
        while j < n && ok {
            let d = i128::from(m[i * n + j]) - i128::from(m[j * n + i]);
            if d.abs() > 1 {
                ok = false;
            }
            j += 1;
        }
        i += 1;
    }
    ok
}

/// Deterministic 64-bit xorshift: the start basis' only randomness (no `rand`, no entropy).
struct Lcg(u64);

impl Lcg {
    /// A generator at `seed`.
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64-bit word.
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// An `i64` in `[lo, hi)` for `0 < hi − lo ≤ 2⁴⁴`; the high 44 bits are used so a power-of-two
    /// span does not read a short LCG cycle.
    fn next_range(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (hi - lo).unsigned_abs();
        lo + ((self.next_u64() >> 20) % span) as i64
    }
}

/// The deterministic `n × s` start basis: columns of a fixed LCG-seeded integer matrix in
/// `(−2³⁰, 2³⁰)` raw Q32.32, never the identity (the identity can be exactly invariant under a
/// structured `M`).
///
/// # Complexity
/// `n·s` xorshift steps and `O(n·s)` space.
fn start_basis(n: usize, s: usize) -> Vec<i64> {
    let mut lcg = Lcg::new(START_SEED);
    let mut q = vec![0i64; n * s];
    for e in &mut q {
        *e = lcg.next_range(-(1i64 << 30), 1i64 << 30);
    }
    q
}

/// Modified Gram–Schmidt with one re-orthogonalisation pass in `i128`: the `s` columns of the
/// row-major `n × s` `y` are replaced by an orthonormal basis (Q32.32, unit 2-norm) of the same
/// span, or [`EigError::DegenerateBasis`] if a column has no component outside its predecessors.
/// Each projection coefficient and each rotated entry is rounded to nearest once from an exact
/// `i128` accumulation; the column norm is the floored `isqrt` of the exact squared norm.
///
/// # Errors
/// [`EigError::DegenerateBasis`] if a column is exactly in the span of its predecessors;
/// [`EigError::Overflow`] on any checked operation or narrowing.
///
/// # Complexity
/// Time: `2·n·s(s−1)/2` checked `i128` multiply-adds for the two orthogonalisation passes plus
/// `n·s` for the normalisation — `O(n·s²)`. Space: `O(n)` scratch beyond the output.
fn orthonormalise(y: &[i64], n: usize, s: usize) -> Result<Vec<i64>, EigError> {
    let mut q = vec![0i64; n * s];
    let mut col = vec![0i64; n];
    let mut err: Option<EigError> = None;
    let mut pass = 0;
    while pass < 2 && err.is_none() {
        let mut j = 0;
        while j < s && err.is_none() {
            let mut i = 0;
            while i < n {
                col[i] = if pass == 0 {
                    y[i * s + j]
                } else {
                    q[i * s + j]
                };
                i += 1;
            }
            // Columns `k < j` are already unit, so `c = <q_k, col> / 2³²` is the projection
            // coefficient in Q32.32.
            let mut k = 0;
            while k < j && err.is_none() {
                let mut acc: i128 = 0;
                let mut i = 0;
                while i < n {
                    let pr = ck!(
                        err,
                        i128::from(q[i * s + k]).checked_mul(i128::from(col[i]))
                    );
                    acc = ck!(err, acc.checked_add(pr));
                    i += 1;
                }
                let c = ck!(err, div_round(acc, i128::from(ONE)));
                let mut i = 0;
                while i < n && err.is_none() {
                    let pr = ckr!(err, r32(ck!(err, c.checked_mul(i128::from(q[i * s + k])))));
                    col[i] = ckr!(err, to_i64(ck!(err, i128::from(col[i]).checked_sub(pr))));
                    i += 1;
                }
                k += 1;
            }
            let mut nrm2: i128 = 0;
            let mut i = 0;
            while i < n && err.is_none() {
                let v = i128::from(col[i]);
                nrm2 = ck!(err, nrm2.checked_add(ck!(err, v.checked_mul(v))));
                i += 1;
            }
            let nrm = isqrt(nrm2);
            if err.is_none() && nrm <= 0 {
                err = Some(EigError::DegenerateBasis);
            }
            if err.is_none() {
                let mut i = 0;
                while i < n && err.is_none() {
                    let num = ck!(err, i128::from(col[i]).checked_mul(i128::from(ONE)));
                    q[i * s + j] = ckr!(err, to_i64(ck!(err, div_round(num, nrm))));
                    i += 1;
                }
            }
            j += 1;
        }
        pass += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(q),
    }
}

/// The `s × s` Rayleigh quotient `T = Qᵀ·Y` of the orthonormal `n × s` `q` against the `n × s`
/// image `y = M·q`: `T[i,j] = round32(Σ_l q[l,i]·y[l,j])`, one rounding per entry, exactly
/// symmetric (the upper triangle is mirrored onto the lower, so the Jacobi's rotations preserve
/// symmetry bit-exactly).
///
/// # Errors
/// [`EigError::Overflow`] on any checked `i128` operation or narrowing.
///
/// # Complexity
/// Time: `n·s²` checked multiply-adds. Space: `O(s²)`.
fn rayleigh(q: &[i64], y: &[i64], n: usize, s: usize) -> Result<Vec<i64>, EigError> {
    let mut t = vec![0i64; s * s];
    let mut err: Option<EigError> = None;
    let mut i = 0;
    while i < s && err.is_none() {
        let mut j = i;
        while j < s && err.is_none() {
            let mut acc: i128 = 0;
            let mut l = 0;
            while l < n {
                let pr = ck!(
                    err,
                    i128::from(q[l * s + i]).checked_mul(i128::from(y[l * s + j]))
                );
                acc = ck!(err, acc.checked_add(pr));
                l += 1;
            }
            let v = ckr!(err, to_i64(ckr!(err, r32(acc))));
            t[i * s + j] = v;
            t[j * s + i] = v;
            j += 1;
        }
        i += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(t),
    }
}

/// The `n × s` product `Q·V` of the `n × s` `q` and the `s × s` `v`: `out[i,j] =
/// round32(Σ_k q[i,k]·v[k,j])`, one rounding per entry.
///
/// # Errors
/// [`EigError::Overflow`] on any checked `i128` operation or narrowing.
///
/// # Complexity
/// Time: `n·s²` checked multiply-adds. Space: `O(n·s)`.
fn mul_small(q: &[i64], v: &[i64], n: usize, s: usize) -> Result<Vec<i64>, EigError> {
    let mut out = vec![0i64; n * s];
    let mut err: Option<EigError> = None;
    let mut i = 0;
    while i < n && err.is_none() {
        let mut j = 0;
        while j < s && err.is_none() {
            let mut acc: i128 = 0;
            let mut k = 0;
            while k < s {
                let pr = ck!(
                    err,
                    i128::from(q[i * s + k]).checked_mul(i128::from(v[k * s + j]))
                );
                acc = ck!(err, acc.checked_add(pr));
                k += 1;
            }
            out[i * s + j] = ckr!(err, to_i64(ckr!(err, r32(acc))));
            j += 1;
        }
        i += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// One two-sided Jacobi rotation of the symmetric `s × s` `t` on the pair `(p, q)`, applying the
/// same rotation to the columns `p`, `q` of the `s × s` basis `v`. The rotation annihilates
/// `t[p,q]`: `θ = (t_qq − t_pp) / (2 t_pq)`, `t = sign(θ)/(|θ| + √(1+θ²))`, `c = 1/√(1+t²)`,
/// `s_ = t·c`, all in raw Q32.32 with one round-to-nearest per stored entry and the square roots
/// from [`isqrt`] — no float. A pair with `|θ| > 2²⁸` is left alone: its `|t| ≤ 2⁻²⁹` is below the
/// fixed-point resolution of the update anyway.
///
/// # Errors
/// [`EigError::Overflow`] on any checked `i128` operation or narrowing.
///
/// # Complexity
/// Time: `O(s)` checked multiply-adds (two rows/columns and two basis columns). Space: `O(1)`.
fn rotate(t: &mut [i64], v: &mut [i64], s: usize, p: usize, q: usize) -> Result<(), EigError> {
    let apq = i128::from(t[p * s + q]);
    if apq == 0 {
        return Ok(());
    }
    let app = i128::from(t[p * s + p]);
    let aqq = i128::from(t[q * s + q]);
    let num = ov((aqq - app).checked_mul(1i128 << 32))?;
    let den = apq * 2;
    // `|θ| > 2²⁸` in *real* units is `|θ_raw| = |num/den| > 2⁶⁰`, where the rotation's `|t| ≤ 2⁻²⁹`
    // is below the resolution of the update; the bound keeps `θ²` inside `i128`.
    if num.abs() > ov(den.abs().checked_mul(1i128 << 60))? {
        return Ok(());
    }
    let theta = ov(div_round(num, den))?;
    let th2 = r32(ov(theta.checked_mul(theta))?)?;
    let one_plus = ov(th2.checked_add(i128::from(ONE)))?;
    if one_plus <= 0 {
        return Ok(());
    }
    let root = isqrt(ov(one_plus.checked_mul(1i128 << 32))?);
    let den2 = ov(theta.abs().checked_add(root))?;
    if den2 <= 0 {
        return Ok(());
    }
    let sign = if theta < 0 { -1i128 } else { 1i128 };
    let tv = ov(div_round(ov(sign.checked_mul(1i128 << 64))?, den2))?;
    let t2 = r32(ov(tv.checked_mul(tv))?)?;
    let c2 = ov(t2.checked_add(i128::from(ONE)))?;
    let root_c = isqrt(ov(c2.checked_mul(1i128 << 32))?);
    if root_c <= 0 {
        return Ok(());
    }
    let cv = ov(div_round(1i128 << 64, root_c))?;
    let sv = r32(ov(tv.checked_mul(cv))?)?;
    let tapq = r32(ov(tv.checked_mul(apq))?)?;
    t[p * s + p] = to_i64(ov(app.checked_sub(tapq))?)?;
    t[q * s + q] = to_i64(ov(aqq.checked_add(tapq))?)?;
    t[p * s + q] = 0;
    t[q * s + p] = 0;
    // The zero-diagonal sweep and the basis rotation below are the two loops of the body; both
    // carry their checked-arithmetic failure in `err` rather than a `?`, so no early `return` sits
    // inside a loop.
    let mut err: Option<EigError> = None;
    let mut i = 0;
    while i < s && err.is_none() {
        if i != p && i != q {
            let tip = i128::from(t[i * s + p]);
            let tiq = i128::from(t[i * s + q]);
            let np = ckr!(
                err,
                r32(ck!(
                    err,
                    ck!(err, cv.checked_mul(tip)).checked_sub(ck!(err, sv.checked_mul(tiq)))
                ))
            );
            let nq = ckr!(
                err,
                r32(ck!(
                    err,
                    ck!(err, sv.checked_mul(tip)).checked_add(ck!(err, cv.checked_mul(tiq)))
                ))
            );
            let np = ckr!(err, to_i64(np));
            let nq = ckr!(err, to_i64(nq));
            t[i * s + p] = np;
            t[p * s + i] = np;
            t[i * s + q] = nq;
            t[q * s + i] = nq;
        }
        i += 1;
    }
    // The basis columns `p`, `q` rotate for every row, including `p` and `q` themselves.
    let mut i = 0;
    while i < s && err.is_none() {
        let vip = i128::from(v[i * s + p]);
        let viq = i128::from(v[i * s + q]);
        let vp = ckr!(
            err,
            r32(ck!(
                err,
                ck!(err, cv.checked_mul(vip)).checked_sub(ck!(err, sv.checked_mul(viq)))
            ))
        );
        let vq = ckr!(
            err,
            r32(ck!(
                err,
                ck!(err, sv.checked_mul(vip)).checked_add(ck!(err, cv.checked_mul(viq)))
            ))
        );
        v[i * s + p] = ckr!(err, to_i64(vp));
        v[i * s + q] = ckr!(err, to_i64(vq));
        i += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// The `s × s` integer Jacobi eigensolver: sweeps of [`rotate`] over the upper triangle until the
/// largest off-diagonal magnitude falls to `max(scale·2⁻OFF_DIAG_SHIFT, 2^OFF_DIAG_FLOOR)`
/// (`scale` the largest diagonal magnitude) or [`JACOBI_SWEEPS`] sweeps run. Returns the diagonal
/// eigenvalues (unsorted) and the basis `V` whose column `k` is the eigenvector for `vals[k]`.
///
/// # Errors
/// [`EigError::Overflow`] from any rotation.
///
/// # Complexity
/// Time: `JACOBI_SWEEPS·s(s−1)/2` rotations of `O(s)` each — `O(s³)` worst case, `s ≤ 16` here.
/// Space: `O(s²)`.
fn jacobi(t_in: &[i64], s: usize) -> Result<(Vec<i64>, Vec<i64>), EigError> {
    let mut t = t_in.to_vec();
    let mut v = vec![0i64; s * s];
    let mut i = 0;
    while i < s {
        v[i * s + i] = ONE;
        i += 1;
    }
    let mut err: Option<EigError> = None;
    let mut sweep = 0;
    while sweep < JACOBI_SWEEPS && err.is_none() {
        let mut off: i64 = 0;
        let mut scale: i64 = 0;
        let mut i = 0;
        while i < s {
            let d = t[i * s + i].abs();
            if d > scale {
                scale = d;
            }
            let mut j = i + 1;
            while j < s {
                let o = t[i * s + j].abs();
                if o > off {
                    off = o;
                }
                j += 1;
            }
            i += 1;
        }
        let tol = (scale >> OFF_DIAG_SHIFT).max(1i64 << OFF_DIAG_FLOOR);
        if off <= tol {
            // The convergence exit of the sweep loop, as a loop-counter clamp rather than a
            // `break` (which Aeneas rejects inside the nested rotation loops).
            sweep = JACOBI_SWEEPS;
        } else {
            let mut p = 0;
            while p < s && err.is_none() {
                let mut q = p + 1;
                while q < s && err.is_none() {
                    match rotate(&mut t, &mut v, s, p, q) {
                        Ok(()) => {}
                        Err(e) => err = Some(e),
                    }
                    q += 1;
                }
                p += 1;
            }
            sweep += 1;
        }
    }
    let mut vals = vec![0i64; s];
    let mut i = 0;
    while i < s {
        vals[i] = t[i * s + i];
        i += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok((vals, v)),
    }
}

/// Sort `vals` ascending, carrying the matching columns of the `s × s` row-major `v` through the
/// same swaps. Insertion sort: deterministic, `O(s²)` time outer and `O(1)` extra space; `s` is a
/// block width, not a batch size.
fn sort_ascending(vals: &mut [i64], v: &mut [i64], s: usize) {
    for i in 1..vals.len() {
        let mut j = i;
        while j > 0 && vals[j - 1] > vals[j] {
            vals.swap(j - 1, j);
            for r in 0..s {
                v.swap(r * s + (j - 1), r * s + j);
            }
            j -= 1;
        }
    }
}

/// The top `r` columns (`s − r .. s`) of the row-major `n × s` `q`, as row-major `n × r`.
///
/// # Complexity
/// `n·r` copies, `O(n·r)` space.
fn top_columns(q: &[i64], n: usize, s: usize, r: usize) -> Vec<i64> {
    let mut out = vec![0i64; n * r];
    for i in 0..n {
        for k in 0..r {
            out[i * r + k] = q[i * s + (s - r + k)];
        }
    }
    out
}

/// The max over the kept columns of `‖w_k − λ_k q_k‖∞`, where `w` is the `n × cols` image of the
/// `n × cols` basis `q` (both row-major, column block `[col0, col0 + r)` of the same span) and
/// `vals` the `r` matching ascending eigenvalues. Row `i` of kept column `k` is
/// `w[i·cols + col0 + k]` and `q[i·cols + col0 + k]`.
///
/// # Errors
/// [`EigError::Overflow`] on any checked `i128` operation or narrowing.
///
/// # Complexity
/// Time: `n·r` checked multiply-subtracts. Space: `O(1)`.
fn residual_max(
    w: &[i64],
    q: &[i64],
    vals: &[i64],
    n: usize,
    cols: usize,
    col0: usize,
) -> Result<i64, EigError> {
    let r = vals.len();
    let mut worst: i64 = 0;
    let mut err: Option<EigError> = None;
    let mut k = 0;
    while k < r && err.is_none() {
        let lam = i128::from(vals[k]);
        let mut i = 0;
        while i < n && err.is_none() {
            let idx = i * cols + (col0 + k);
            let lv = ckr!(err, r32(ck!(err, lam.checked_mul(i128::from(q[idx])))));
            let d = ck!(err, i128::from(w[idx]).checked_sub(lv)).abs();
            let d = ckr!(err, to_i64(d));
            if d > worst {
                worst = d;
            }
            i += 1;
        }
        k += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(worst),
    }
}

/// Negate each of the `r` columns of the row-major `n × r` `evecs` whose largest-magnitude entry is
/// negative, so the returned column has a positive largest-magnitude entry (ties broken by the
/// lowest index). Negation cannot leave `i64`: the entries are unit-2-norm Q32.32.
///
/// # Complexity
/// `O(n·r)` comparisons and negations, `O(1)` space.
fn fix_signs(evecs: &mut [i64], n: usize, r: usize) {
    let mut k = 0;
    while k < r {
        // The sign is carried as the value at the first largest-magnitude entry rather than that
        // entry's index: assigning the loop counter to a local is the one shape whose borrow
        // fixed point the pinned Aeneas cannot compute (`Could not compute a loop fixed point`).
        let mut best_abs: i128 = -1;
        let mut best_val: i64 = 0;
        let mut i = 0;
        while i < n {
            let x = evecs[i * r + k];
            let a = i128::from(x).abs();
            if a > best_abs {
                best_abs = a;
                best_val = x;
            }
            i += 1;
        }
        if best_val < 0 {
            let mut i = 0;
            while i < n {
                evecs[i * r + k] = -evecs[i * r + k];
                i += 1;
            }
        }
        k += 1;
    }
}

/// A [`GemmError`] from a product whose operands were already validated here (shape, non-zero
/// dimensions, the `2³⁶` entry bound and `n < `[`super::gemm::MAX_K`]) can only be
/// [`GemmError::Overflow`], so it maps to [`EigError::Overflow`].
fn gemm_err(_: GemmError) -> EigError {
    EigError::Overflow
}

/// The outcome of one subspace-iteration step ([`ritz_step`]).
enum Ritz {
    /// The stop test failed: the next basis `q ← orth(Y·V)` and the residual `est` it left.
    Continue {
        /// The next orthonormal `n × s` basis.
        q: Vec<i64>,
        /// This step's max column residual `‖M v_k − λ_k v_k‖∞` over the kept columns, raw Q32.32.
        est: i64,
    },
    /// The stop test passed and the returned columns were confirmed on a fresh exact image.
    Done(TopEig),
}

/// One block subspace-iteration step on the orthonormal `n × s` `q`: `Y = M·q` ([`gemm_nn`]),
/// `T = qᵀY` ([`rayleigh`]) diagonalised by [`jacobi`], the two rotation products, the residual
/// test, and — on convergence — the confirmation of the returned columns against a fresh exact
/// image `M·V`. This is the body of [`top_eigh`]'s sweep, factored into a helper so every `?` sits
/// outside a loop (the shape the pinned Aeneas accepts).
///
/// # Errors
/// [`EigError::Overflow`] from any checked step, [`EigError::DegenerateBasis`] from the power-step
/// orthonormalisation.
///
/// # Complexity
/// As one iteration of [`top_eigh`]: `n²·s` [`gemm_nn`] multiply-accumulates plus `O(n·s²)`, and on
/// the exit the extra `n²·r` image. Space `O(n·s + s²)`.
fn ritz_step(
    m: &[i64],
    q: &[i64],
    n: usize,
    r: usize,
    s: usize,
    iters: usize,
    threads: usize,
) -> Result<Ritz, EigError> {
    let y = gemm_nn(m, q, n, n, s, threads).map_err(gemm_err)?;
    let t = rayleigh(q, &y, n, s)?;
    let (mut vals, mut v) = jacobi(&t, s)?;
    sort_ascending(&mut vals, &mut v, s);
    let qv = mul_small(q, &v, n, s)?;
    let yv = mul_small(&y, &v, n, s)?;
    let tol = (vals[s - 1] >> RESIDUAL_SHIFT).max(1);
    let est = residual_max(&yv, &qv, &vals[s - r..], n, s, s - r)?;
    if est <= tol {
        // Confirm the object that is returned: the orthonormalised top columns against a fresh
        // exact image `M·V`, not the rotated candidate pair the loop carries.
        let vf = orthonormalise(&top_columns(&qv, n, s, r), n, r)?;
        let w = gemm_nn(m, &vf, n, n, r, threads).map_err(gemm_err)?;
        let exact = residual_max(&w, &vf, &vals[s - r..], n, r, 0)?;
        if exact <= tol {
            let mut evecs = vf;
            fix_signs(&mut evecs, n, r);
            return Ok(Ritz::Done(TopEig {
                evals: vals[s - r..].to_vec(),
                evecs,
                iters,
            }));
        }
    }
    // The power step: the next basis spans `M·Q'`, and the orthonormalisation resets the rounding
    // drift the rotation products left behind.
    Ok(Ritz::Continue {
        q: orthonormalise(&yv, n, s)?,
        est,
    })
}

/// The top `r` eigenpairs of the symmetric PSD `n × n` Q32.32 `m`, ascending, by block subspace
/// iteration with the oversampling `p = min(`[`OVERSAMPLE`]`, n − r)` and a Rayleigh–Ritz
/// acceleration: `Y = M·Q` ([`gemm_nn`]), `T = Qᵀ Y` diagonalised by [`jacobi`], the power step
/// `Q ← orth(Y·V)`, and the stop test `‖M v_k − λ_k v_k‖∞ ≤ λ_max·2⁻`[`RESIDUAL_SHIFT`].
/// [`MAX_ITERS`] iterations without reaching it is [`EigError::NotConverged`].
///
/// `threads` is handed to [`gemm_nn`]; every other step is serial and the accumulation order is
/// fixed, so the result is bit-identical for every `threads`.
///
/// # Errors
/// [`EigError::Empty`] if `n == 0`; [`EigError::RankZero`] if `r == 0`; [`EigError::RankOutOfRange`]
/// if `r > n`; [`EigError::ShapeMismatch`] if `m.len() != n·n`; [`EigError::InputBound`] if an entry
/// reaches `2³⁶`; [`EigError::NotSymmetric`] if `M` is non-symmetric beyond one raw unit;
/// [`EigError::Overflow`] on any checked `i128` operation or narrowing, or a dimension product that
/// leaves `usize`; [`EigError::DegenerateBasis`] if `rank(M) < r + p`;
/// [`EigError::NotConverged`] if the residual never reaches the tolerance.
///
/// # Complexity
/// Time: `MAX_ITERS·(n²·s + O(n·s²))` checked `i128` multiply-accumulates, `s = r + p` — that is
/// `n²(r+p)` per iteration for the [`gemm_nn`] product `Y = M·Q`, the dominant term, plus
/// `O(n·(r+p)²)` for the Rayleigh quotient, the two rotation products and the power-step
/// orthonormalisation; the exit iteration adds one `n²·r` [`gemm_nn`] image for the exact residual
/// check of the returned columns. Space: `O(n·(r+p) + (r+p)²)` for the basis, the images and the
/// `s × s` Ritz problem. The iteration count is the first `k` with `(λ_{s+1}/λ_r)^k ≤ 2⁻`
/// [`RESIDUAL_SHIFT`] (up to the initial coefficients and the rounding floor), the linear
/// convergence rate of subspace iteration with the gap the `p` oversampling columns carry.
pub fn top_eigh(m: &[i64], n: usize, r: usize, threads: usize) -> Result<TopEig, EigError> {
    if n == 0 {
        return Err(EigError::Empty);
    }
    if r == 0 {
        return Err(EigError::RankZero);
    }
    if r > n {
        return Err(EigError::RankOutOfRange);
    }
    let nn = ov(n.checked_mul(n))?;
    if m.len() != nn {
        return Err(EigError::ShapeMismatch);
    }
    if !within_bound(m, None) {
        return Err(EigError::InputBound);
    }
    if !is_symmetric(m, n) {
        return Err(EigError::NotSymmetric);
    }
    let s = r + OVERSAMPLE.min(n - r);
    let mut q = orthonormalise(&start_basis(n, s), n, s)?;
    let mut last: i64 = i64::MAX;
    let mut err: Option<EigError> = None;
    let mut done: Option<TopEig> = None;
    let mut iters = 1usize;
    while iters <= MAX_ITERS && err.is_none() && done.is_none() {
        match ritz_step(m, &q, n, r, s, iters, threads) {
            Ok(Ritz::Continue { q: next, est }) => {
                q = next;
                last = est;
            }
            Ok(Ritz::Done(eig)) => done = Some(eig),
            Err(e) => err = Some(e),
        }
        iters += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    if let Some(eig) = done {
        return Ok(eig);
    }
    Err(EigError::NotConverged { residual: last })
}
