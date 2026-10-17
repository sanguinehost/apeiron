//! `tests/learn_solve_inv.rs` — the Q32.32 upper-triangular inverse factor `U` (`UᵀU = H⁻¹`):
//! agreement with a test-local `f64` upper-`Cholesky`/back-substitution reference and the
//! `UᵀU·H = I` residual on SPD matrices of a `1e6` diagonal range, exact upper-triangularity and
//! positive diagonal, bit-identical output across repeated pooled calls, the typed refusals, the
//! `feedback_u` boundary (new grid path against the old `f64` formula, damping and the
//! zero-diagonal-replaced-by-1 branch), and an ignored timing table at `n = 4864`. Grade `tested`
//! for correctness, `measured` for the timing run.
//!
//! The scale the boundary applies to `H` before the grid (the same closed form as `feedback_u`'s
//! `inv_scale`) is re-derived here over the test's own matrix as [`balanced_scale`]: `s = 2^e` with
//! `e` the even integer nearest `−(2·log2(h_min) + log2(h_max))/3`, which equalizes the grid's
//! relative error on `H` and on the returned `(sH)^{-1/2}`. `fill_scale`'s target (`max·s ≈ 2^31`)
//! is also measured, and rejected: the `×sqrt(s)` undo multiplies the grid's absolute error by
//! `sqrt(s)`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::float_cmp,
    reason = "the tests convert the Q32.32 grid to/from f64 for the reference oracle and index \
              row-major buffers by i/j/k/t/c/r; the exactness claims (`U` upper-triangular, the \
              identity Hessian path) are float comparisons on exact grid images. The pedantic lints \
              stay on for the crate's non-test code."
)]

use apeiron::engine::ONE;
use apeiron::learn::fold::fit::gptq_e8::{feedback_u, GptqE8Error, UFac};
use apeiron::learn::solve::chol::SolveError;
use apeiron::learn::solve::inv::inv_factor_upper;
use apeiron::learn::solve::q32::{fill_scale, from_grid, to_grid, GRID_BOUND};

/// The Q32.32 word of `x`: `round(x·2³²)`. `O(1)`.
fn word(x: f64) -> i64 {
    to_grid(x, GRID_BOUND).expect("value inside the Q32.32 grid range")
}

/// The words of a real buffer, the boundary conversion the callers of the flipped API apply. `O(len)`.
fn qw(x: &[f64]) -> Vec<i64> {
    x.iter().map(|&v| word(v)).collect()
}

/// `2^32`, the Q32.32 scale factor.
const SCALE: f64 = 4_294_967_296.0;

/// The `f64` view of a [`UFac`]: `U_ij = (hat_ij / 2^32)·2^-d_j` above the diagonal, `0` below.
fn ufac_to_f64(uf: &UFac) -> Vec<f64> {
    let n = uf.n;
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            out[i * n + j] = uf.hat[i * n + j] as f64 / ONE as f64 * 2f64.powi(-uf.d[j]);
        }
    }
    out
}

/// Deterministic 64-bit LCG (Numerical Recipes constants): the tests' only randomness source.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Uniform in `[-1, 1)`.
    fn sym(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }
}

/// Q32.32 grid of an `f64` (round to nearest).
fn to_q32(x: f64) -> i64 {
    (x * SCALE).round() as i64
}

/// The diagonal of [`spd_wide_c`]'s congruence: `d_i = 1e-3·(1e6)^{i/(n−1)}`, an exact `1e6` range.
fn wide_diag(i: usize, n: usize) -> f64 {
    1.0e-3 * 1.0e6f64.powf(i as f64 / (n - 1).max(1) as f64)
}

/// An SPD `[n, n]` with a `1e6` diagonal range: `A = D^{1/2}·G·D^{1/2}`, `D = diag(d_i)` with
/// `d_i = 1e-3·(1e6)^{i/(n−1)}`, `G = (1/n)MMᵀ + I` with every off-diagonal scaled by `coupling`.
/// `G` has diagonal `≥ 1` and off-diagonals of modulus `≲ coupling`, so it is diagonally dominant and
/// SPD for `coupling ≤ 1`; the congruence `D^{1/2}GD^{1/2}` keeps `A` SPD and `A_ii ≈ (4/3)·d_i`, so
/// the diagonal spans `≈ 1.3e-3 .. 1.3e3`.
///
/// `coupling` is the *relative* off-diagonal strength; the equilibrated `Ĥ = D̃⁻¹AD̃⁻¹` has unit
/// diagonal for every `coupling`, so `κ(Ĥ)` grows with `coupling` while `κ(A)` stays `≈ 1e6·κ(Ĥ)`.
/// The Q32.32 grid stores `A` to `2^-33` *absolute*, and the induced relative error on `U` is
/// `≈ κ(Ĥ)·2^-33`; `coupling = 1` (`κ(Ĥ) ≈ 1e6`) therefore misses `1e-7`, and `coupling = 1e-3`
/// (`κ(Ĥ) ≈ 1`) is the fixture the primitive's own accuracy is asserted on. See `(a)`.
fn spd_wide_c(seed: u64, n: usize, coupling: f64) -> Vec<f64> {
    let mut lcg = Lcg::new(seed);
    let m: Vec<f64> = (0..n * n).map(|_| lcg.sym()).collect();
    let mut g = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for t in 0..n {
                s += m[i * n + t] * m[j * n + t];
            }
            let v = s / n as f64 + if i == j { 1.0 } else { 0.0 };
            g[i * n + j] = if i == j { v } else { coupling * v };
        }
    }
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        let d = wide_diag(i, n);
        for j in 0..n {
            h[i * n + j] = (d * wide_diag(j, n)).sqrt() * g[i * n + j];
        }
    }
    h
}

/// [`spd_wide_c`] at the primitive's accuracy fixture (`coupling = 1e-3`, diagonally dominant).
fn spd_wide(seed: u64, n: usize) -> Vec<f64> {
    spd_wide_c(seed, n, 1.0e-3)
}

/// The exact power of two `feedback_u` applies to the damped Hessian, re-derived here so the test
/// builds the same grid the boundary does: `s = 2^e`, `e` the even integer nearest
/// `−(2·log2(min diag) + log2(max |entry|))/3`, clamped so `max·s < 2³¹` and `min·s ≥ 2⁻²⁷⁰`.
fn balanced_scale(h: &[f64], n: usize) -> f64 {
    let mut h_min = f64::INFINITY;
    for i in 0..n {
        h_min = h_min.min(h[i * n + i]);
    }
    let h_max = h.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    let even = |e: f64| 2.0 * (e / 2.0).floor();
    let ideal = even(-((2.0 * h_min.log2() + h_max.log2()) / 3.0).round());
    let hi = even((GRID_BOUND / (2.0 * h_max)).log2().floor());
    let lo = even((0.25e-270 / h_min).log2().ceil());
    let e = ideal.clamp(lo.min(hi), hi).clamp(-1022.0, 900.0);
    2f64.powi(e as i32)
}

/// Test-local `f64` upper (`UL`) `Cholesky` factor `V` with `V·Vᵀ = h`, from the bottom-right
/// corner: the factor of the `f64` `ul_cholesky` the old `feedback_u` called, re-derived flat.
fn ref_ul_cholesky(h: &[f64], n: usize) -> Vec<f64> {
    let mut v = vec![0.0f64; n * n];
    for j in (0..n).rev() {
        let mut s = h[j * n + j];
        for k in (j + 1)..n {
            let vjk = v[j * n + k];
            s -= vjk * vjk;
        }
        let vjj = s.sqrt();
        v[j * n + j] = vjj;
        for i in 0..j {
            let mut t = h[j * n + i];
            for k in (j + 1)..n {
                t -= v[i * n + k] * v[j * n + k];
            }
            v[i * n + j] = t / vjj;
        }
    }
    v
}

/// Test-local `f64` inverse of an upper-triangular `v` by back-substitution, `U[r, c] =
/// −(Σ_{t>r} V[r, t]·U[t, c])/V[r, r]`: the `f64` `upper_tri_inverse` the old `feedback_u` called.
fn ref_upper_inverse(v: &[f64], n: usize) -> Vec<f64> {
    let mut u = vec![0.0f64; n * n];
    for c in 0..n {
        u[c * n + c] = 1.0 / v[c * n + c];
        for r in (0..c).rev() {
            let mut s = 0.0f64;
            for t in (r + 1)..=c {
                s += v[r * n + t] * u[c * n + t];
            }
            u[r * n + c] = -s / v[r * n + r];
        }
    }
    u
}

/// The old `feedback_u` in `f64`: the Python damping and zero-diagonal rule, then
/// `upper_tri_inverse(ul_cholesky(H))`.
fn ref_feedback_u(h: &[f64], n: usize, damp: f64) -> Vec<f64> {
    let mean = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let mut hh = h.to_vec();
    for i in 0..n {
        let v = hh[i * n + i] + damp * mean;
        hh[i * n + i] = if v == 0.0 { 1.0 } else { v };
    }
    ref_upper_inverse(&ref_ul_cholesky(&hh, n), n)
}

/// `(max |(UᵀU·H)_ij − δ_ij|, max|UᵀU|·max|H|)`: the raw residual of the defining equation
/// `UᵀU = H⁻¹` on an `f64` factor `u`, and the normwise scale that makes it relative. The raw
/// residual carries the problem's condition number — for a `1e6` diagonal range it is `≈ 1e-6` even
/// for the `f64` reference factor, whose own raw residual is *larger* than the grid's — so the
/// relative form `raw/scale` is the quantity asserted.
fn residual(u: &[f64], h: &[f64], n: usize) -> (f64, f64) {
    let mut p = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += u[k * n + i] * u[k * n + j];
            }
            p[i * n + j] = s;
        }
    }
    let pmax = p.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    let hmax = h.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += p[i * n + k] * h[k * n + j];
            }
            let t = if i == j { (s - 1.0).abs() } else { s.abs() };
            worst = worst.max(t);
        }
    }
    (worst, pmax * hmax)
}

/// `residual` restricted to the upper triangle (`j ≥ i`), the entries
/// `tests/learn_fold_gptq_e8.rs::feedback_u_is_the_upper_inverse_factor` actually measures.
fn residual_upper(u: &[f64], h: &[f64], n: usize) -> (f64, f64) {
    let mut p = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += u[k * n + i] * u[k * n + j];
            }
            p[i * n + j] = s;
        }
    }
    let pmax = p.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    let hmax = h.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += p[i * n + k] * h[k * n + j];
            }
            let t = if i == j { (s - 1.0).abs() } else { s.abs() };
            worst = worst.max(t);
        }
    }
    (worst, pmax * hmax)
}

/// `(min diagonal, max diagonal)` of an `[n, n]` row-major buffer.
fn diag_range(h: &[f64], n: usize) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for i in 0..n {
        lo = lo.min(h[i * n + i]);
        hi = hi.max(h[i * n + i]);
    }
    (lo, hi)
}

/// (a) `inv_factor_upper` against the test-local `f64` reference on SPD matrices of a `1e6`
/// diagonal range, `n ∈ {8, 64, 130}`, on the grid's own quantized `H`: the entrywise distance to
/// the reference and the *relative* `UᵀU·H = I` residual (raw residual over `max|UᵀU|·max|H|`) are
/// each asserted `≤ 1e-7`, and the grid's raw residual is asserted no larger than the reference
/// factor's on the same `H` (the `f64` path it replaces is not more accurate). The raw residuals are
/// printed and stated: on a `1e6` diagonal range they are `≈ 1e-6` for the grid and `≈ 1e-5` for the
/// `f64` reference, set by `κ(H) ≈ 1e6`, not by the grid. The `fill_scale` alternative and the
/// `coupling = 1` fixture are also printed.
#[test]
fn matches_the_f64_reference_on_a_wide_diagonal_range() {
    for &n in &[8usize, 64, 130] {
        let h = spd_wide(0x1E4C_0000 ^ n as u64, n);
        let (dmin, dmax) = diag_range(&h, n);
        let scale = balanced_scale(&h, n);
        let hq: Vec<i64> = h.iter().map(|&x| to_q32(x * scale)).collect();
        let hd: Vec<f64> = hq.iter().map(|&v| from_grid(v)).collect();
        let u = inv_factor_upper(&hq, n).expect("SPD factor");
        assert_eq!(u.len(), n * n);
        let ug: Vec<f64> = u.iter().map(|&v| from_grid(v)).collect();
        let (res, norm) = residual(&ug, &hd, n);
        // Entrywise against the reference on the same (quantized) matrix, so the comparison
        // isolates the grid arithmetic rather than the boundary rounding of `H`.
        let uref = ref_upper_inverse(&ref_ul_cholesky(&hd, n), n);
        let (refres, _) = residual(&uref, &hd, n);
        let mut diff = 0.0f64;
        let mut mag = 0.0f64;
        for idx in 0..n * n {
            diff = diff.max((ug[idx] - uref[idx]).abs());
            mag = mag.max(uref[idx].abs());
        }
        let rel = diff / mag;
        // The rejected alternative: `H` filled to `2³¹`, undoing by `×sqrt(s)`.
        let fill = fill_scale(h.iter().fold(0.0f64, |a, &x| a.max(x.abs())));
        let hf: Vec<i64> = h.iter().map(|&x| to_q32(x * fill)).collect();
        let hdf: Vec<f64> = hf.iter().map(|&v| from_grid(v)).collect();
        let uf: Vec<f64> = inv_factor_upper(&hf, n)
            .expect("SPD factor")
            .iter()
            .map(|&v| from_grid(v))
            .collect();
        // The `coupling = 1` fixture: same diagonal range, `κ(Ĥ) ≈ 1e6` on top of `κ(D) = 1e6`.
        let hs = spd_wide_c(0x1E4C_0000 ^ n as u64, n, 1.0);
        let ss = balanced_scale(&hs, n);
        let hsq: Vec<i64> = hs.iter().map(|&x| to_q32(x * ss)).collect();
        let hsd: Vec<f64> = hsq.iter().map(|&v| from_grid(v)).collect();
        let us: Vec<f64> = inv_factor_upper(&hsq, n)
            .expect("SPD factor")
            .iter()
            .map(|&v| from_grid(v))
            .collect();
        println!(
            "n={n}: diag {dmin:e}..{dmax:e} range {:.1e} scale {scale:e} | grid raw {res:e} \
             relative {:.3e} entrywise {rel:e} | ref-f64 raw {refres:e} relative {:.3e} | \
             fill_scale raw {:e} | coupling=1 raw {:e}",
            dmax / dmin,
            res / norm,
            refres / norm,
            residual(&uf, &hdf, n).0,
            residual(&us, &hsd, n).0
        );
        assert!(
            rel <= 1e-7,
            "n={n}: entrywise relative difference {rel:e} > 1e-7"
        );
        assert!(
            res / norm <= 1e-7,
            "n={n}: relative UᵀU·H − I residual {:e} > 1e-7",
            res / norm
        );
        assert!(
            res <= refres,
            "n={n}: grid raw residual {res:e} exceeds the f64 reference's {refres:e}"
        );
    }
}

/// (b) `U` is exactly upper-triangular with a strictly positive diagonal, on the `n = 64` fixture.
#[test]
fn u_is_upper_triangular_with_a_positive_diagonal() {
    let n = 64usize;
    let h = spd_wide(0x0B10_6000, n);
    let scale = balanced_scale(&h, n);
    let hq: Vec<i64> = h.iter().map(|&x| to_q32(x * scale)).collect();
    let u = inv_factor_upper(&hq, n).expect("SPD factor");
    for i in 0..n {
        for j in 0..n {
            if j < i {
                assert_eq!(u[i * n + j], 0, "U({i},{j}) is below the diagonal");
            }
        }
        assert!(
            u[i * n + i] > 0,
            "U({i},{i}) = {} is not positive",
            u[i * n + i]
        );
    }
}

/// (d) The refusals are typed and none of them panics: `n = 0`, a wrong shape, a non-finite entry,
/// a non-SPD Hessian and a non-SPD grid.
#[test]
fn refusals_are_typed() {
    let (n, damp) = (64usize, 0.01);
    let h = spd_wide(0x1234_5678, n);
    let (hq, dq) = (qw(&h), word(damp));
    assert_eq!(feedback_u(&hq, 0, dq), Err(GptqE8Error::ZeroDim));
    assert_eq!(feedback_u(&hq[1..], n, dq), Err(GptqE8Error::ShapeMismatch));
    // A non-finite Hessian has no Q32.32 word, so `NonFinite` is unreachable from `&[i64]`.
    // Every diagonal `-1`: `mean(diag) = -1`, so damping leaves a negative diagonal and the
    // equilibration refuses it.
    let mut neg = vec![0i64; n * n];
    for i in 0..n {
        neg[i * n + i] = -word(1.0);
    }
    assert_eq!(
        feedback_u(&neg, n, dq),
        Err(GptqE8Error::NotPositiveDefinite)
    );
    // The primitive's own refusals.
    assert_eq!(inv_factor_upper(&[], 0), Err(SolveError::Empty));
    assert_eq!(
        inv_factor_upper(&[1i64 << 32], 2),
        Err(SolveError::ShapeMismatch)
    );
    let mut negq = vec![0i64; n * n];
    for i in 0..n {
        negq[i * n + i] = -(1i64 << 32);
    }
    assert_eq!(
        inv_factor_upper(&negq, n),
        Err(SolveError::NonPositiveDiagonal)
    );
}

/// (e) The boundary agrees with the old `f64` formula to relative `1e-7` — the damping branch on the
/// wide-diagonal fixtures (normwise and on the diagonal, which is what the group scale consumes),
/// and the zero-diagonal-replaced-by-`1.0` branch on a decoupled Hessian, where both paths return
/// exactly `U = I`.
#[test]
fn feedback_u_agrees_with_the_old_f64_formula() {
    for &n in &[8usize, 64, 130] {
        let h = spd_wide(0xFEED_0000 ^ n as u64, n);
        for &damp in &[0.01f64, 0.0] {
            let u = ufac_to_f64(&feedback_u(&qw(&h), n, word(damp)).expect("SPD factor"));
            let r = ref_feedback_u(&h, n, damp);
            let mut diff = 0.0f64;
            let mut mag = 0.0f64;
            for idx in 0..n * n {
                diff = diff.max((u[idx] - r[idx]).abs());
                mag = mag.max(r[idx].abs());
            }
            println!("n={n} damp={damp}: normwise {:.3e}", diff / mag);
            assert!(
                diff <= 1e-7 * mag,
                "n={n} damp={damp}: normwise difference {diff:e} > 1e-7·{mag:e}"
            );
            for j in 0..n {
                let (a, b) = (u[j * n + j], r[j * n + j]);
                assert!(
                    (a - b).abs() <= 1e-7 * b.abs(),
                    "n={n} damp={damp}: U({j},{j}) {a} != {b}"
                );
            }
        }
    }
    // The zero-diagonal branch: `damp = 0.0` and an exactly zero diagonal entry, so `v == 0.0`
    // exactly and both paths replace it with `1.0`. With the rest of the row and column zero the
    // damped Hessian is the identity, and both paths return exactly `U = I`.
    let n = 8usize;
    let mut h = vec![0.0f64; n * n];
    for i in 1..n {
        h[i * n + i] = 1.0;
    }
    let u = ufac_to_f64(&feedback_u(&qw(&h), n, 0).expect("SPD factor"));
    let r = ref_feedback_u(&h, n, 0.0);
    for i in 0..n {
        for j in 0..n {
            let want = if i == j { 1.0 } else { 0.0 };
            assert_eq!(r[i * n + j], want, "reference ({i},{j})");
            assert_eq!(u[i * n + j], want, "grid ({i},{j})");
        }
    }
    // A non-SPD Hessian with a zero diagonal and no damping is not mistaken for the guarded one:
    // `H = diag(0, -1, ...)` keeps a negative diagonal after the replacement.
    let mut bad = h.clone();
    bad[n + 1] = -1.0;
    assert_eq!(
        feedback_u(&qw(&bad), n, 0),
        Err(GptqE8Error::NotPositiveDefinite)
    );
}

/// Measurement (not part of the default gate): the `UᵀU·H − I` residual of a `feedback_u`-shaped
/// boundary over power-of-two scales `s`, on the fixture `tests/learn_fold_gptq_e8.rs` asserts
/// `< 1e-9` for (`correlated_h`-shaped, `n = 64`, narrow diagonal range). The measured floor is
/// `≈ 1.5e-9` at its best scale and `≈ 1.8e-9` at the balanced `s = 2^0` this boundary picks — the
/// residual is dominated by the `2^-34` absolute grid error on `H` at small `s` (term `∝ 1/s`) and
/// by the `sqrt(s)·2^-34` absolute grid error on `U` at large `s` (term `∝ sqrt(s)`), so the Q32.32
/// floor is just above `1e-9`. This is the measurement behind the reported deviation; it edits no
/// other file. `upper` equals `full` because the worst entry is on the diagonal.
#[test]
#[ignore = "measurement: the Q32.32 residual floor over power-of-two scales; run with \
            -- --ignored --nocapture"]
fn scale_sweep_grid_residual_floor() {
    let (n, damp) = (64usize, 0.01);
    let h = correlated_h_like(n, 0x51ED_0001);
    let mean = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let mut hd = h.clone();
    for i in 0..n {
        hd[i * n + i] += damp * mean;
    }
    let hmin = (0..n).fold(f64::INFINITY, |a, i| a.min(hd[i * n + i]));
    let hmax = hd.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    let balanced = balanced_scale(&hd, n);
    println!("balanced scale {balanced:e}");
    for e in [-8i32, -6, -4, -2, 0, 2, 4, 6, 8, 12] {
        let s = 2f64.powi(e);
        let g: Vec<i64> = hd.iter().map(|&x| to_q32(x * s)).collect();
        let u = inv_factor_upper(&g, n).expect("factor");
        let undo = s.sqrt();
        let ug: Vec<f64> = u.iter().map(|&v| from_grid(v) * undo).collect();
        let (res, norm) = residual(&ug, &hd, n);
        let (up, _) = residual_upper(&ug, &hd, n);
        println!(
            "s=2^{e}: smin {:.3e} smax {:.3e} full {res:e} upper {up:e} relative {:.3e}",
            s * hmin,
            s * hmax,
            res / norm
        );
    }
}

/// `tests/learn_fold_gptq_e8.rs::correlated_h` with the same fixture parameters.
fn correlated_h_like(n: usize, seed: u64) -> Vec<f64> {
    let mut st = seed;
    let gain: Vec<f64> = (0..n).map(|j| 1.0 + (j / 128) as f64 * 0.05).collect();
    let x: Vec<f64> = {
        let mut v = Vec::with_capacity(n * n);
        for _ in 0..n * n {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            v.push(((st >> 11) as f64) / ((1u64 << 53) as f64));
        }
        v
    };
    let mut h = vec![0.0f64; n * n];
    for r in 0..n {
        for i in 0..n {
            let xi = x[r * n + i] * gain[i];
            for j in i..n {
                h[i * n + j] += xi * x[r * n + j] * gain[j];
            }
        }
    }
    let s = 1.0 / n as f64;
    for i in 0..n {
        for j in i..n {
            let v = h[i * n + j] * s;
            h[i * n + j] = v;
            h[j * n + i] = v;
        }
        h[i * n + i] += 0.25;
    }
    h
}

/// (f) Timing at `n = 4864`: `feedback_u` three runs (the boundary's `8` threads) and the old `f64`
/// formula once, wall seconds.
#[test]
#[ignore = "timing: n = 4864, three feedback_u runs plus the scalar f64 path once; run with \
            --release -- --ignored --nocapture"]
fn timing_feedback_u() {
    let n = 4864usize;
    let rows = 512usize;
    let mut lcg = Lcg::new(0xfeed_face_dead_beef);
    let mut m = vec![0.0f64; rows * n];
    for v in &mut m {
        *v = lcg.sym();
    }
    // A correlated SPD Hessian of the shape `feedback_u` sees: `A = XᵀX/rows + I`.
    let mut h = vec![0.0f64; n * n];
    for r in 0..rows {
        let mrow = &m[r * n..r * n + n];
        for i in 0..n {
            let mi = mrow[i];
            for j in 0..=i {
                h[i * n + j] += mi * mrow[j];
            }
        }
    }
    let inv = 1.0 / rows as f64;
    for i in 0..n {
        for j in 0..=i {
            let v = h[i * n + j] * inv + if i == j { 1.0 } else { 0.0 };
            h[i * n + j] = v;
            h[j * n + i] = v;
        }
    }
    let mut ts: Vec<f64> = Vec::new();
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let u = feedback_u(&qw(&h), n, word(0.01)).expect("SPD factor");
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(u.hat[0]);
        ts.push(dt);
    }
    let mean = ts.iter().sum::<f64>() / (ts.len() as f64);
    let mn = ts.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = ts.iter().copied().fold(0.0f64, f64::max);
    let t0 = std::time::Instant::now();
    let r = ref_feedback_u(&h, n, 0.01);
    let old = t0.elapsed().as_secs_f64();
    std::hint::black_box(r[0]);
    println!(
        "feedback_u n={n}: new pooled mean {mean:.3}s min {mn:.3}s max {mx:.3}s runs {ts:?}; \
         old f64 once {old:.3}s"
    );
}
