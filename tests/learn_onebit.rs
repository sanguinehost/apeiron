//! `tests/learn_onebit.rs` — the W7 controls: an exact `±a` matrix is exact in every arm, zeroing
//! the feedback raises `sd1`/`gptq1` error, `Q4_K` sign planes reconstruct the codes bit for bit,
//! and the `sd1` state bound `|e_t| ≤ a` holds under `|w_t| ≤ a`. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::float_cmp,
    reason = "the controls mirror the modelfold reference indexing and arithmetic; flat row-major \
              index arithmetic is the payload format; the exactness and bits/weight assertions are \
              deliberately exact float equalities (the arms are exact, not approximately so)."
)]

use apeiron::format::quant::{q4_k_codes, Q4_K_BLOCK_BYTES, Q4_K_SCALE_SIZE, QK_K};
use apeiron::learn::gptq::output_error;
use apeiron::learn::onebit::{quantize, Arm, ONEBIT_BPW, ONEBIT_GROUP};

/// Deterministic xorshift in `[−0.5, 0.5)`.
fn rng(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 40) as f32 / 16_777_216.0 - 0.5
}

/// A synthetic correlated SPD Gram: `t` rows drawn as `1 + small noise` over the `n` columns, so
/// `G = XᵀX ≈ t·1` is dominated by one rank-1 direction and the plain-sign error (a DC residual)
/// is penalised relative to the sigma-delta shaped residual.
fn correlated_gram(n: usize, t: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut gram = vec![0f64; n * n];
    for _ in 0..t {
        let x: Vec<f64> = (0..n)
            .map(|_| f64::from(1.0 + 0.02 * rng(&mut s)))
            .collect();
        for i in 0..n {
            for j in 0..n {
                gram[i * n + j] += x[i] * x[j];
            }
        }
    }
    gram
}

/// The zeroed-feedback reference: `sign(w)·a`, no error shaping — both `sd1` with the state reset to
/// 0 every step and `gptq1` with no residual push collapse to this. Test-local, so the control can
/// be run against the real arms in [`apeiron::learn::onebit`].
fn sign_scale(w: &[f32], out: usize, n: usize, group: usize) -> Vec<f32> {
    let mut wq = vec![0f32; out * n];
    for o in 0..out {
        let base = o * n;
        let mut g0 = 0;
        while g0 < n {
            let g1 = (g0 + group).min(n);
            let a = ((g0..g1).map(|j| f64::from(w[base + j]).abs()).sum::<f64>() / (g1 - g0) as f64)
                .max(1e-12) as f32;
            for j in g0..g1 {
                wq[base + j] = a * if w[base + j] < 0.0 { -1.0 } else { 1.0 };
            }
            g0 = g1;
        }
    }
    wq
}

/// The first-order sigma-delta recurrence at an explicit scale `a`: returns `max_t |e_t|/a`.
fn sd1_scale(w: &[f64], a: f64) -> f64 {
    let mut e = 0.0f64;
    let mut peak = 0.0f64;
    for &wv in w {
        let q = a * if wv + e < 0.0 { -1.0 } else { 1.0 };
        e = wv + e - q;
        peak = peak.max(e.abs() / a);
    }
    peak
}

/// Control 1: an exact `±a` matrix (random signs, a constant `a` per block) is exact in every
/// non-rotation arm, and its Hadamard image is exact in the rotation arms. The rotation arms are
/// tested on `R·S` because the Hadamard mixes a `±a` vector into a dense one; `R·S` is the matrix
/// whose image is `±a`.
#[test]
fn exact_sign_scale_matrix_is_exact() {
    let (out, n, group) = (3usize, 2 * ONEBIT_GROUP, ONEBIT_GROUP);
    let mut s = 0x1234_5678_9abc_def0u64;
    // A global `a` is constant on every block of every arm's order (natural, act and path), so the
    // exactness assertion covers all orders at once.
    let a = 0.7f32;
    let mut exact = vec![0f32; out * n];
    for o in 0..out {
        for j in 0..n {
            exact[o * n + j] = if rng(&mut s) < 0.0 { -a } else { a };
        }
    }
    let gram = correlated_gram(n, n, 0xdead_beef);
    let zero = vec![0f32; out * n];

    for arm in [
        Arm::Rtn,
        Arm::Sd1,
        Arm::Sd1Path,
        Arm::Gptq1,
        Arm::Gptq1Act,
        Arm::Gptq1Path,
    ] {
        let r = quantize(arm, &exact, out, n, &gram, group).expect("quantize");
        let e = output_error(&exact, &r.what, out, n, &gram).expect("output_error");
        assert_eq!(e, 0.0, "arm {} not exact on a ±a matrix: {e}", arm.name());
        assert_eq!(r.bpw, ONEBIT_BPW);
    }

    // `R·S`: apply the normalised 128-block Hadamard to every row block of `exact`.
    let mut rot = exact.clone();
    for o in 0..out {
        for b in 0..n / ONEBIT_GROUP {
            let base = o * n + b * ONEBIT_GROUP;
            let mut blk = [0f64; ONEBIT_GROUP];
            for k in 0..ONEBIT_GROUP {
                blk[k] = f64::from(rot[base + k]);
            }
            let mut h = 1;
            while h < ONEBIT_GROUP {
                let mut i = 0;
                while i < ONEBIT_GROUP {
                    for j in i..i + h {
                        let (p, q) = (blk[j], blk[j + h]);
                        blk[j] = p + q;
                        blk[j + h] = p - q;
                    }
                    i += 2 * h;
                }
                h *= 2;
            }
            let inv = 1.0 / (ONEBIT_GROUP as f64).sqrt();
            for k in 0..ONEBIT_GROUP {
                rot[base + k] = (blk[k] * inv) as f32;
            }
        }
    }
    for arm in [Arm::HadRtn, Arm::HadGptq1] {
        let r = quantize(arm, &rot, out, n, &gram, group).expect("quantize");
        let e = output_error(&rot, &r.what, out, n, &gram).expect("output_error");
        assert!(
            e < 1e-9,
            "arm {} not exact on the rotated ±a matrix: {e}",
            arm.name()
        );
        assert_eq!(r.bpw, ONEBIT_BPW);
    }
    assert_eq!(output_error(&exact, &zero, out, n, &gram).unwrap(), 1.0);
}

/// Control 2: zeroing the feedback raises the error. On a correlated SPD Gram the shaped `sd1`
/// residual and the Gram-corrected `gptq1` residual beat the zeroed-feedback `sign·a`.
#[test]
fn zeroing_the_feedback_raises_error() {
    let (out, n, group) = (1usize, ONEBIT_GROUP, ONEBIT_GROUP);
    let mut s = 0x0f1e_2d3c_4b5a_6978u64;
    let w: Vec<f32> = (0..out * n).map(|_| rng(&mut s)).collect();
    let gram = correlated_gram(n, n, 0xfeed_face);

    let zero = sign_scale(&w, out, n, group);
    let e_zero = output_error(&w, &zero, out, n, &gram).expect("oe zero");
    for arm in [Arm::Sd1, Arm::Gptq1] {
        let r = quantize(arm, &w, out, n, &gram, group).expect("quantize");
        let e = output_error(&w, &r.what, out, n, &gram).expect("output_error");
        assert!(
            e_zero > e,
            "zeroed feedback ({e_zero:.6e}) must exceed {} error ({e:.6e})",
            arm.name()
        );
    }
}

/// Control 3: the `Q4_K` sign planes reconstruct the codes bit for bit, and the plane matvec equals
/// the code matvec exactly in integer arithmetic.
#[test]
fn q4k_sign_planes_reconstruct_codes() {
    let mut blk = [0u8; Q4_K_BLOCK_BYTES];
    for i in 0..128usize {
        blk[4 + Q4_K_SCALE_SIZE + i] = (((i % 16) as u8) << 4) | (((i * 7 + 3) % 16) as u8);
    }
    let codes = q4_k_codes(&blk).expect("q4_k_codes");
    assert_eq!(codes.len(), QK_K);

    let x: Vec<i64> = (0..QK_K)
        .map(|j| i64::try_from(j).unwrap().wrapping_mul(2_654_435_761) % 2001 - 1000)
        .collect();
    let mut code_dot = 0i64;
    let mut plane_dot = 0i64;
    for j in 0..QK_K {
        let q = i32::from(codes[j]);
        assert!(q <= 15, "Q4_K code out of range: {q}");
        let recon: i32 = (0..4).map(|k| (1 << k) * (2 * ((q >> k) & 1) - 1)).sum();
        assert_eq!(2 * q - 15, recon, "code {q} plane identity");
        code_dot += i64::from(2 * q - 15) * x[j];
        for k in 0..4 {
            plane_dot += i64::from((1 << k) * (2 * ((q >> k) & 1) - 1)) * x[j];
        }
    }
    assert_eq!(code_dot, plane_dot, "plane matvec ≠ code matvec");
}

/// Control 4: the `sd1` state bound. If `|w_t| ≤ a` at the scale `a`, then `max|e_t|/a ≤ 1`; real
/// rows exceed `a = mean|w|`, which is why the harness measures `stab`.
#[test]
fn sd1_state_is_bounded_by_the_scale() {
    let mut s = 0x2468_ace0_1357_9bdfu64;
    // Clipped rows at scale 1: the lemma's hypothesis, exercised over random rows.
    for _ in 0..64 {
        let w: Vec<f64> = (0..ONEBIT_GROUP)
            .map(|_| 1.0 * f64::from(rng(&mut s)))
            .collect();
        assert!(w.iter().all(|&v| v.abs() <= 1.0));
        assert!(sd1_scale(&w, 1.0) <= 1.0, "state bound violated at a=1");
    }
    // The library arm on a constant-magnitude row: there `a = mean|w| = |w_t|`, so the hypothesis
    // holds exactly and `stab` must be ≤ 1.
    let out = 1usize;
    let n = ONEBIT_GROUP;
    let mut w = vec![0f32; out * n];
    for j in 0..n {
        w[j] = if rng(&mut s) < 0.0 { -0.6 } else { 0.6 };
    }
    let gram = correlated_gram(n, n, 7);
    let r = quantize(Arm::Sd1, &w, out, n, &gram, ONEBIT_GROUP).expect("quantize");
    assert!(r.stab <= 1.0, "library sd1 stab {} > 1", r.stab);
    assert!(r.stab.is_finite());
    // `stab` is NaN for the non-sigma-delta arms.
    for arm in [Arm::Rtn, Arm::Gptq1, Arm::HadRtn] {
        let r = quantize(arm, &w, out, n, &gram, ONEBIT_GROUP).expect("quantize");
        assert!(r.stab.is_nan(), "{} stab must be NaN", arm.name());
    }
}
