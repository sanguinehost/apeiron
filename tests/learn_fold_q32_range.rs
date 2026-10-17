//! `tests/learn_fold_q32_range.rs` — the two `Q32.32` register defects the production
//! Qwen2.5-0.5B transfer hit in the fit steps, each as the smallest geometry that reproduces it.
//!
//! 1. `feedback_u`'s even global exponent had no lower bound: at the stage-O Gram scale
//!    (`max|H| ≈ 0.07`, measured) it returned `-32`, where `1.0` at the renormalised scale
//!    `2^(32−g)` is `2^64` and the factor refused with `Overflow`. The exponent is floored at `-30`.
//! 2. `c_step` and `d_step` multiplied the normal system's Gram (`2^46` words at `o = 4864`) by the
//!    metric (`2^53`) as a `Q32.32` word, needing 67 bits; the whole system is now rescaled by the
//!    exact `2^-s`, which leaves the solution and every smaller fit bit for bit unchanged.
//!
//! Grade `tested`; nothing here is `proved`. Both are mutation controls: removing `MIN_G` in
//! `gptq_e8_carrier.rs` or the `product_shift` rescale in `steps.rs` turns the corresponding test
//! red with `Err(Overflow)` / `Err(ShapeMismatch)`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "the fixtures mirror the row-major indices and the Q32.32 casts of the ported kernel; \
              the short `o/i/j/k` indices are the reference arithmetic's vocabulary"
)]

use apeiron::engine::ONE;
use apeiron::learn::fold::fit::gptq_e8::{feedback_u, GptqE8Error, BLOCK};
use apeiron::learn::fold::fit::steps::{c_step, compose, d_step, GROUP};
use apeiron::learn::solve::q32;

/// Out rows of the step fixtures: the production `o = 4864` Gram's accumulation order, at the
/// smallest row count that reaches its word register (one row adds `2^36`, so 1024 rows are `2^46`).
const O: usize = 1024;
/// In columns (one 128-block).
const N: usize = 128;
/// Group count along `n`.
const G: usize = N / GROUP;

/// xorshift64: deterministic, no crate.
fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Deterministic uniform in `[−1, 1)`.
fn uniform(s: &mut u64) -> f64 {
    (xorshift(s) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// The `Q32.32` words of a real buffer. `O(len)`.
fn q(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&v| q32::to_grid(v, q32::GRID_BOUND).expect("fixture inside the grid"))
        .collect()
}

/// A word slice read back to the reals. `O(len)`.
fn unq(x: &[i64]) -> Vec<f64> {
    x.iter().copied().map(q32::from_grid).collect()
}

fn max_abs(x: &[f64]) -> f64 {
    x.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

/// `max |a − b| / max(|b|, 1)`: the entrywise relative disagreement of two metric vectors.
fn max_rel_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs() / y.abs().max(1.0)))
}

/// A symmetric positive-definite `n × n` metric: `(1/n)·A·Aᵀ + I` with `A` uniform in `[-1, 1)`.
fn spd_metric(rng: &mut u64, n: usize) -> Vec<f64> {
    let a: Vec<f64> = (0..n * n).map(|_| uniform(rng)).collect();
    let mut hl = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += a[i * n + k] * a[j * n + k];
            }
            hl[i * n + j] = acc / (n as f64) + if i == j { 1.0 } else { 0.0 };
        }
    }
    hl
}

/// `H = (1/t)·XᵀX + 0.25·I` for a deterministic uniform `X` (`t = n`) with a per-8-block column
/// gain, so the diagonal traces are distinct. The stage-O Gram's shape at `max|H| ≈ 0.1` after the
/// `1/8` scale the caller applies.
fn correlated_h(n: usize, seed: u64) -> Vec<f64> {
    let t = n;
    let mut st = seed;
    let gain: Vec<f64> = (0..n).map(|j| 1.0 + (j / BLOCK) as f64 * 0.05).collect();
    let x: Vec<f64> = (0..t * n).map(|_| uniform(&mut st)).collect();
    let mut h = vec![0.0f64; n * n];
    for r in 0..t {
        for i in 0..n {
            let xi = x[r * n + i] * gain[i];
            for j in i..n {
                h[i * n + j] += xi * x[r * n + j] * gain[j];
            }
        }
    }
    let s = 1.0 / t as f64;
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

/// [`feedback_u`] on the word boundary: `Uᴴ U` is the damped inverse of a small-scale Hessian.
///
/// At `max|H| ≈ 0.07` the even global exponent is `-32`; without the `MIN_G` floor the sentinel
/// `1.0` at the renormalised scale is `2^64` and the factor returns `Err(Overflow)`.
#[test]
fn feedback_u_small_hessian_keeps_the_identity_representable() {
    let (n, damp) = (64usize, 0.01);
    // `correlated_h`'s entries are O(1); one eighth leaves `max|H|` near `0.1`, i.e.
    // `floor(log2 max|H|) <= -4`, so the unscaled exponent is `<= -32`.
    let h: Vec<f64> = correlated_h(n, 0x0F00_0004)
        .iter()
        .map(|x| x * 0.125)
        .collect();
    let uf = feedback_u(&q(&h), n, q(&[damp])[0]).expect("small-Hessian SPD factor");
    let u = {
        let one = ONE as f64;
        let mut out = vec![0.0f64; n * n];
        for i in 0..n {
            for (j, d) in uf.d.iter().enumerate().skip(i) {
                out[i * n + j] = uf.hat[i * n + j] as f64 / one * 2f64.powi(-d);
            }
        }
        out
    };
    let mean = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let mut hd = h.clone();
    for i in 0..n {
        hd[i * n + i] += damp * mean;
    }
    let mut p = vec![0.0f64; n * n];
    for (ij, pij) in p.iter_mut().enumerate() {
        let (i, j) = (ij / n, ij % n);
        *pij = (0..n).map(|k| u[k * n + i] * u[k * n + j]).sum();
    }
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in i..n {
            let m: f64 = (0..n).map(|k| p[i * n + k] * hd[k * n + j]).sum();
            worst = worst.max(if i == j { (m - 1.0).abs() } else { m.abs() });
        }
    }
    // The grid floor of this fixture's residual is one rounding (`2^-33` of a `2^30`-scale
    // inverse, ~3.9e-9 measured); the bound leaves it a factor of 1.3 of headroom.
    assert!(worst < 5e-9, "small-Hessian UᴴU − I residual {worst:e}");
    // The refusal it used to return is gone at this scale, and the shape refusals still fire.
    assert_eq!(feedback_u(&q(&h), 0, 0), Err(GptqE8Error::ZeroDim));
    assert_eq!(
        feedback_u(&q(&h)[1..], n, 0),
        Err(GptqE8Error::ShapeMismatch)
    );
}

/// The `c_step` and `d_step` normal systems at the production metric word scale.
///
/// The `wsyrk` Gram is `2^45..2^46` words (the panel shift lands `Sh` at `2^34`, `n = 256`
/// accumulate); `Hl` at `2^20..2^21` real is a 53-bit word, the measured production register. The
/// `Q32.32` product `A ⊙ H` therefore needs `45 + 53 − 32 = 66` bits, three above `i64`. The step
/// rescales the whole system by the exact `2^-s`, so the planted `c0` is still recovered and the
/// solution agrees with the same fit at the raw metric scale; before the rescale both steps return
/// `Err(ShapeMismatch)`.
///
/// Mutation control: removing `product_shift`'s `shift_in_place` from `c_step_wshl` (or
/// `d_step_wshl`) turns this test red at the step's `unwrap`.
#[test]
fn c_and_d_steps_hold_at_the_production_metric_scale() {
    let mut rng = 0x5EED_0007u64;
    let s: Vec<f64> = (0..O * N)
        .map(|_| if uniform(&mut rng) < 0.0 { -1.0 } else { 1.0 })
        .collect();
    // The row scale carries the production weight register: `|Ws| ≈ 2^-4` (real weights), so
    // `Ws·Hl` stays below `2^50` — the measured production `wshl` — while `Hl` is a 53-bit word.
    let r: Vec<f64> = (0..O * G)
        .map(|_| (0.2 + 0.8 * uniform(&mut rng).abs()) / 8.0)
        .collect();
    let c0: Vec<f64> = (0..N)
        .map(|_| 0.2 + 0.8 * uniform(&mut rng).abs())
        .collect();
    let hl = spd_metric(&mut rng, N);
    let s_rot: Vec<f64> = (0..O * N)
        .map(|_| if uniform(&mut rng) < 0.0 { -1.0 } else { 1.0 })
        .collect();
    let ws = compose(&q(&s), &q(&r), &q(&c0), O, N).expect("compose inside the grid");
    let hl_q = q(&hl);
    let hl_big: Vec<i64> = hl_q
        .iter()
        .map(|&v| v.checked_shl(20).expect("metric word times 2^20"))
        .collect();
    let bl = |x: &[i64]| {
        64 - x
            .iter()
            .fold(0u64, |m, &v| m.max(v.unsigned_abs()))
            .leading_zeros()
    };
    println!(
        "metric: raw {:e}, big {:e} ({} / {} word bits)",
        max_abs(&unq(&hl_q)),
        max_abs(&unq(&hl_big)),
        bl(&hl_q),
        bl(&hl_big)
    );
    // The planted `c0` is the exact un-ridged solution of both systems: the metric enters `A` and
    // `b` linearly, so a pure power-of-two scale of `Hl` leaves the minimiser fixed.
    let c_raw = c_step(&ws, &hl_q, &q(&s), &q(&r), O, N, None).expect("raw-scale c_step");
    let c_big = c_step(&ws, &hl_big, &q(&s), &q(&r), O, N, None).expect("big-scale c_step");
    let err = max_abs_diff(&unq(&c_big), &c0);
    println!(
        "c_step big metric: c0 err {:e}, vs raw {:e}",
        err,
        max_rel_diff(&unq(&c_big), &unq(&c_raw))
    );
    assert!(
        err <= 1e-6 * (1.0 + max_abs(&c0)),
        "c0 recovery err {err:e}"
    );
    assert!(
        max_rel_diff(&unq(&c_big), &unq(&c_raw)) <= 1e-6,
        "big vs raw metric c"
    );
    let d_raw = d_step(&ws, &hl_q, &q(&s_rot), &q(&r), &q(&c0), O, N, None).expect("raw d_step");
    let d_big = d_step(&ws, &hl_big, &q(&s_rot), &q(&r), &q(&c0), O, N, None).expect("big d_step");
    println!(
        "d_step big metric: vs raw {:e}",
        max_rel_diff(&unq(&d_big), &unq(&d_raw))
    );
    assert!(
        max_rel_diff(&unq(&d_big), &unq(&d_raw)) <= 1e-6,
        "big vs raw metric d"
    );
}
