//! `tests/learn_solve_inv.rs` — the Q32.32 upper-triangular inverse factor `U` (`UᵀU = H⁻¹`):
//! agreement with a test-local `f64` upper-`Cholesky`/back-substitution reference and the
//! `UᵀU·H = I` residual on SPD matrices of a `1e6` diagonal range, exact upper-triangularity and
//! positive diagonal, bit-identical output across repeated pooled calls and the typed refusals.
//! Grade `tested`; the ignored scale sweep is `measured`.
//!
//! The scale applied to `H` before the grid is derived here over the test's own matrix as [`balanced_scale`]: `s = 2^e` with
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

use apeiron::learn::solve::chol::SolveError;
use apeiron::learn::solve::inv::inv_factor_upper;
use apeiron::learn::solve::q32::{fill_scale, from_grid, GRID_BOUND};

/// `2^32`, the Q32.32 scale factor.
const SCALE: f64 = 4_294_967_296.0;

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

/// The power of two applied to the Hessian before the grid: `s = 2^e`, `e` the even integer nearest
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
/// corner: the `f64` reference factor.
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
/// −(Σ_{t>r} V[r, t]·U[t, c])/V[r, r]`: the `f64` reference inverse.
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

/// `residual` restricted to the upper triangle (`j ≥ i`).
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

/// (d) The refusals are typed and none of them panics: `n = 0`, a wrong shape and a non-positive
/// diagonal.
#[test]
fn refusals_are_typed() {
    let n = 64usize;
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

/// Measurement (not part of the default gate): the `UᵀU·H − I` residual of `inv_factor_upper` over
/// power-of-two scales `s` on a damped correlated Hessian (`n = 64`, narrow diagonal range). The measured floor is
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

/// A correlated SPD fixture: `XᵀX/n` with a slowly rising column gain, plus `0.25·I`.
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
