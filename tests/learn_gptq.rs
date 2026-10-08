//! `tests/learn_gptq.rs` — GPTQ-ternary tests: error-feedback beats naive rounding,
//! affine chart beats naive pack, balanced-trit planes reconstruct affine dequantization,
//! and error paths refuse invalid input. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "the test setup mirrors modelfold tests/gptq.rs reference indexing and arithmetic."
)]

use apeiron::format::quant::{dequant_q4_k_block, Q4_K_BLOCK_BYTES, QK_K};
use apeiron::learn::gptq::{
    gptq_affine, gptq_q4k, gptq_q4k_damped, gptq_ternary, gptq_ternary_absmean, output_error,
    round_ternary, ul_cholesky, upper_tri_inverse, GptqError,
};
use apeiron::learn::ridge::solve_spd;
use apeiron::learn::seal::quantize_row_q4_k;

/// The naive per-block asymmetric affine fit — bit-for-bit what `decode_common::pack_proj` does
/// (min/max, no error-feedback). The baseline `gptq_affine` must beat.
fn naive_affine(w: &[f32], out: usize, n: usize, levels: u32, group: usize) -> Vec<f32> {
    let lvl = levels as f32;
    let mut wq = vec![0f32; out * n];
    for o in 0..out {
        let mut g0 = 0;
        while g0 < n {
            let g1 = (g0 + group).min(n);
            let lo = (g0..g1).map(|j| w[o * n + j]).fold(f32::INFINITY, f32::min);
            let hi = (g0..g1)
                .map(|j| w[o * n + j])
                .fold(f32::NEG_INFINITY, f32::max);
            let d = if hi > lo { (hi - lo) / lvl } else { 1.0 };
            for j in g0..g1 {
                let q = ((w[o * n + j] - lo) / d).round().clamp(0.0, lvl);
                wq[o * n + j] = d * q + lo;
            }
            g0 = g1;
        }
    }
    wq
}

/// Deterministic xorshift in [−0.5, 0.5).
fn rng(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 40) as f32 / 16_777_216.0 - 0.5
}

/// Build a synthetic `(W, X)` with unequal per-channel activation energy, and the calibration Gram
/// `G = XᵀX` computed directly.
fn setup(out: usize, n: usize, t: usize, seed: u64) -> (Vec<f32>, Vec<f64>) {
    let mut s = seed;
    let w: Vec<f32> = (0..out * n).map(|_| rng(&mut s)).collect();
    let gain: Vec<f32> = (0..n)
        .map(|j| if j % 3 == 0 { 6.0 } else { 0.25 })
        .collect();
    let shared: Vec<f32> = (0..t).map(|_| rng(&mut s)).collect();
    let x: Vec<f32> = (0..t * n)
        .map(|i| {
            let (ti, j) = (i / n, i % n);
            (rng(&mut s) + 0.9 * shared[ti]) * gain[j]
        })
        .collect();
    let xf: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let mut gram = vec![0f64; n * n];
    for r in 0..t {
        let row = &xf[r * n..(r + 1) * n];
        for i in 0..n {
            let xi = row[i];
            let g_row = &mut gram[i * n..(i + 1) * n];
            for j in 0..n {
                g_row[j] += xi * row[j];
            }
        }
    }
    (w, gram)
}

/// The whole thesis: at `L = 3` (3 trits, the 4-bit→ternary target), error-feedback lands a
/// strictly smaller OUTPUT distortion than naive rounding — the residual it pushes forward is
/// distortion it does not pay.
#[test]
fn gptq_beats_round_l3() {
    let (out, n, t) = (16usize, 96usize, 128usize);
    let (w, gram) = setup(out, n, t, 0x1234_5678);
    let group = 32;
    let fed = gptq_ternary(&w, out, n, &gram, 3, group).expect("gptq_ternary failed");
    let raw = round_ternary(&w, out, n, 3, group).expect("round_ternary failed");
    let e_fed = output_error(&w, &fed.wq, out, n, &gram).expect("output_error fed failed");
    let e_raw = output_error(&w, &raw.wq, out, n, &gram).expect("output_error raw failed");
    println!(
        "L=3: feedback={e_fed:.4e} naive={e_raw:.4e} ratio={:.2}x",
        e_raw / e_fed
    );
    assert!(
        e_fed < e_raw,
        "error-feedback {e_fed:.4e} must beat naive round {e_raw:.4e}"
    );
    // 3 trits over the 4-bit range should be a small forward distortion outright.
    assert!(
        e_fed < 0.05,
        "L=3 GPTQ distortion {e_fed:.4e} unexpectedly large"
    );
}

/// The carrier-chart integration: GPTQ error-feedback into the SAME asymmetric affine chart the
/// GPU decode reads (`w=d·q+lo`) beats the naive min/max pack on output distortion — at both the
/// 2-bit (levels=3) and 4-bit (levels=15) carrier widths.
#[test]
fn gptq_affine_beats_naive_pack() {
    let (out, n, t) = (16usize, 96usize, 128usize);
    let (w, gram) = setup(out, n, t, 0x0FF1_CE55);
    for levels in [3u32, 15] {
        let ch = gptq_affine(&w, out, n, &gram, levels, 32).expect("gptq_affine failed");
        let naive = naive_affine(&w, out, n, levels, 32);
        let e_gptq = output_error(&w, &ch.wq, out, n, &gram).expect("output_error gptq failed");
        let e_naive = output_error(&w, &naive, out, n, &gram).expect("output_error naive failed");
        println!(
            "levels={levels}: gptq={e_gptq:.4e} naive={e_naive:.4e} ratio={:.2}x",
            e_naive / e_gptq
        );
        assert!(
            e_gptq < e_naive,
            "levels={levels}: gptq {e_gptq:.4e} must beat naive {e_naive:.4e}"
        );
        assert!(
            ch.codes.iter().all(|&q| u32::from(q) <= levels),
            "codes exceed the carrier width"
        );
    }
}

/// Path B ≡ Path A: summing the balanced-trit Q2 planes reproduces the affine dequant bit-for-bit
/// (`balanced_ternary_lossless` / `dot_plane_sum`) — the two consumptions are one code set.
#[test]
fn planes_reconstruct_affine() {
    let (out, n, t) = (8usize, 48usize, 64usize);
    let (w, gram) = setup(out, n, t, 0xBEEF_CAFE);
    for levels in [1u32, 2, 3] {
        let q = gptq_ternary(&w, out, n, &gram, levels, 24).expect("gptq_ternary failed");
        let planes = q.trit_planes();
        assert_eq!(planes.len() as u32, levels);
        let back = q.dequant_planes(&planes).expect("dequant_planes failed");
        assert!(
            q.wq.iter()
                .zip(&back)
                .all(|(a, b)| (a - b).abs() <= 1e-4 * a.abs().max(1.0)),
            "L={levels}: plane sum must reconstruct the affine golden"
        );
        // every plane code is a genuine trit {0,1,2}.
        assert!(planes.iter().all(|p| p.code.iter().all(|&c| c <= 2)));
    }
}

/// Error paths and boundary validation: shape mismatches, zero group, zero levels,
/// and non-positive definite matrices must be refused with the appropriate [`GptqError`].
#[test]
fn refusals() {
    let (out, n) = (4usize, 8usize);
    let w = vec![1.0f32; out * n];
    let gram = vec![1.0f64; n * n];

    // Shape mismatch on w
    let bad_w = vec![1.0f32; out * n - 1];
    assert_eq!(
        gptq_ternary(&bad_w, out, n, &gram, 3, 4),
        Err(GptqError::ShapeMismatch)
    );
    assert_eq!(
        gptq_ternary_absmean(&bad_w, out, n, &gram, 4),
        Err(GptqError::ShapeMismatch)
    );
    assert_eq!(
        round_ternary(&bad_w, out, n, 3, 4),
        Err(GptqError::ShapeMismatch)
    );
    assert_eq!(
        gptq_affine(&bad_w, out, n, &gram, 3, 4),
        Err(GptqError::ShapeMismatch)
    );
    assert_eq!(
        output_error(&bad_w, &w, out, n, &gram),
        Err(GptqError::ShapeMismatch)
    );

    // Shape mismatch on gram
    let bad_gram = vec![1.0f64; n * n - 1];
    assert_eq!(
        gptq_ternary(&w, out, n, &bad_gram, 3, 4),
        Err(GptqError::ShapeMismatch)
    );
    assert_eq!(
        gptq_ternary_absmean(&w, out, n, &bad_gram, 4),
        Err(GptqError::ShapeMismatch)
    );
    assert_eq!(
        gptq_affine(&w, out, n, &bad_gram, 3, 4),
        Err(GptqError::ShapeMismatch)
    );
    assert_eq!(
        output_error(&w, &w, out, n, &bad_gram),
        Err(GptqError::ShapeMismatch)
    );

    // Shape mismatch on wq in output_error
    assert_eq!(
        output_error(&w, &bad_w, out, n, &gram),
        Err(GptqError::ShapeMismatch)
    );

    // group == 0
    assert_eq!(
        gptq_ternary(&w, out, n, &gram, 3, 0),
        Err(GptqError::ZeroGroup)
    );
    assert_eq!(
        gptq_ternary_absmean(&w, out, n, &gram, 0),
        Err(GptqError::ZeroGroup)
    );
    assert_eq!(round_ternary(&w, out, n, 3, 0), Err(GptqError::ZeroGroup));
    assert_eq!(
        gptq_affine(&w, out, n, &gram, 3, 0),
        Err(GptqError::ZeroGroup)
    );

    // levels == 0
    assert_eq!(
        gptq_ternary(&w, out, n, &gram, 0, 4),
        Err(GptqError::ZeroLevels)
    );
    assert_eq!(round_ternary(&w, out, n, 0, 4), Err(GptqError::ZeroLevels));
    assert_eq!(
        gptq_affine(&w, out, n, &gram, 0, 4),
        Err(GptqError::ZeroLevels)
    );

    // Non-positive definite Gram: e.g. -100 * I
    let mut neg_gram = vec![0.0f64; n * n];
    for i in 0..n {
        neg_gram[i * n + i] = -100.0;
    }
    assert_eq!(
        gptq_ternary(&w, out, n, &neg_gram, 3, 4),
        Err(GptqError::NotPositiveDefinite)
    );
    assert_eq!(
        gptq_ternary_absmean(&w, out, n, &neg_gram, 4),
        Err(GptqError::NotPositiveDefinite)
    );
    assert_eq!(
        gptq_affine(&w, out, n, &neg_gram, 3, 4),
        Err(GptqError::NotPositiveDefinite)
    );

    // dequant_planes shape mismatch
    let q = gptq_ternary(&w, out, n, &gram, 1, 4).expect("valid quant");
    let mut bad_planes = q.trit_planes();
    bad_planes[0].code.pop();
    assert_eq!(q.dequant_planes(&bad_planes), Err(GptqError::ShapeMismatch));
}

/// Refusal paths for `gptq_q4k`: shape mismatch and non-positive definite Hessian.
#[test]
fn gptq_q4k_refusals() {
    let (out, n) = (4usize, 8usize);
    let w = vec![1.0f32; out * n];
    let bad_w = vec![1.0f32; out * n - 1];
    let gram = vec![1.0f64; n * n];
    let bad_gram = vec![1.0f64; n * n - 1];

    assert_eq!(
        gptq_q4k(&bad_w, out, n, &gram, None),
        Err(GptqError::ShapeMismatch)
    );
    assert_eq!(
        gptq_q4k(&w, out, n, &bad_gram, None),
        Err(GptqError::ShapeMismatch)
    );
    // n = 8 is not a multiple of 256
    assert_eq!(
        gptq_q4k(&w, out, n, &gram, None),
        Err(GptqError::ShapeMismatch)
    );
    // n = 256 shape
    let (n256, out256) = (256usize, 1usize);
    let w256 = vec![1.0f32; out256 * n256];
    let bad_tau = vec![1.0f32; n256 - 1];
    let mut neg_gram256 = vec![0.0f64; n256 * n256];
    for i in 0..n256 {
        neg_gram256[i * n256 + i] = -100.0;
    }
    assert_eq!(
        gptq_q4k(&w256, out256, n256, &neg_gram256, None),
        Err(GptqError::NotPositiveDefinite)
    );
    let gram256 = vec![1.0f64; n256 * n256];
    assert_eq!(
        gptq_q4k(&w256, out256, n256, &gram256, Some(&bad_tau)),
        Err(GptqError::ShapeMismatch)
    );
}

/// The carrier-chart integration: GPTQ error-feedback into the `Q4_K` block format beats plain
/// round-to-block (`quantize_row_q4_k`) on output distortion over the calibration Gram.
#[test]
fn gptq_q4k_beats_q4k_round() {
    let (out, n, t) = (4usize, 512usize, 256usize);
    let (w, gram) = setup(out, n, t, 0x5EED_0001);
    let tau: Vec<f32> = (0..n)
        .map(|j| (gram[j * n + j] / t as f64) as f32)
        .collect();

    let gptq_bytes = gptq_q4k(&w, out, n, &gram, Some(&tau)).expect("gptq_q4k failed");
    let expected_len = out * (n / QK_K) * Q4_K_BLOCK_BYTES;
    assert_eq!(
        gptq_bytes.len(),
        expected_len,
        "byte length must equal out * (n / 256) * 144"
    );

    // Decode gptq output
    let mut wq_gptq = vec![0f32; out * n];
    for o in 0..out {
        for b in 0..(n / QK_K) {
            let blk_idx = (o * (n / QK_K) + b) * Q4_K_BLOCK_BYTES;
            let mut out_blk = [0f32; QK_K];
            dequant_q4_k_block(
                &gptq_bytes[blk_idx..blk_idx + Q4_K_BLOCK_BYTES],
                &mut out_blk,
            )
            .expect("dequant_q4_k_block");
            wq_gptq[o * n + b * QK_K..o * n + (b + 1) * QK_K].copy_from_slice(&out_blk);
        }
    }

    // Quantize with plain quantize_row_q4_k per row and decode
    let mut plain_bytes = Vec::with_capacity(expected_len);
    for o in 0..out {
        let row = &w[o * n..(o + 1) * n];
        let row_bytes = quantize_row_q4_k(row, Some(&tau)).expect("quantize_row_q4_k");
        plain_bytes.extend_from_slice(&row_bytes);
    }
    assert_eq!(plain_bytes.len(), expected_len);

    let mut wq_plain = vec![0f32; out * n];
    for o in 0..out {
        for b in 0..(n / QK_K) {
            let blk_idx = (o * (n / QK_K) + b) * Q4_K_BLOCK_BYTES;
            let mut out_blk = [0f32; QK_K];
            dequant_q4_k_block(
                &plain_bytes[blk_idx..blk_idx + Q4_K_BLOCK_BYTES],
                &mut out_blk,
            )
            .expect("dequant_q4_k_block");
            wq_plain[o * n + b * QK_K..o * n + (b + 1) * QK_K].copy_from_slice(&out_blk);
        }
    }

    let e_gptq = output_error(&w, &wq_gptq, out, n, &gram).expect("output_error gptq");
    let e_plain = output_error(&w, &wq_plain, out, n, &gram).expect("output_error plain");
    println!(
        "q4_k: gptq={e_gptq:.4e} plain={e_plain:.4e} ratio={:.2}x",
        e_plain / e_gptq
    );
    assert!(
        e_gptq < e_plain,
        "gptq output_error {e_gptq:.4e} must beat plain {e_plain:.4e}"
    );
}

/// `damping = 0.01` is the `gptq_q4k` path byte-for-byte; `damping = 1.0` (the `seal_model`
/// default) changes the feedback factor and so the packed bytes.
#[test]
fn gptq_q4k_damped_matches_and_moves() {
    let (out, n, t) = (2usize, 256usize, 128usize);
    let (w, gram) = setup(out, n, t, 0x5EED_0002);
    let base = gptq_q4k(&w, out, n, &gram, None).expect("gptq_q4k");
    let same = gptq_q4k_damped(&w, out, n, &gram, None, 0.01).expect("damped 0.01");
    let moved = gptq_q4k_damped(&w, out, n, &gram, None, 1.0).expect("damped 1.0");
    assert_eq!(base, same, "damping 0.01 must reproduce gptq_q4k");
    assert_ne!(base, moved, "damping 1.0 must change the packed bytes");
}

/// Deterministic f64 xorshift in `[0, 1)`.
fn rng_f64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// The `cholesky_upper` factor this change replaced: upper Cholesky `U` (`UᵀU = a`) of symmetric PD
/// `a`. Spelled inline so the test carries the prior `feedback_hi` path (`solve_spd` on the identity
/// right-hand sides, then this upper Cholesky of `H⁻¹`) as its own reference.
fn cholesky_upper_ref(a: &[f64], d: usize) -> Vec<f64> {
    let mut u = vec![0f64; d * d];
    for j in 0..d {
        let mut s = a[j * d + j];
        for k in 0..j {
            s -= u[k * d + j] * u[k * d + j];
        }
        assert!(s > 0.0, "reference Cholesky hit a non-positive pivot");
        let ujj = s.sqrt();
        u[j * d + j] = ujj;
        for i in (j + 1)..d {
            let mut s2 = a[j * d + i];
            for k in 0..j {
                s2 -= u[k * d + j] * u[k * d + i];
            }
            u[j * d + i] = s2 / ujj;
        }
    }
    u
}

/// The GPTQ feedback-factor identity: the new path `upper_tri_inverse(ul_cholesky(H))` computes
/// `U = cholesky_upper(H⁻¹)`, satisfies `max|UᵀU·H − I| ≤ 1e-8`, and equals the prior path
/// (`solve_spd(H, I)` then the old `cholesky_upper` of `H⁻¹`) to ≤ 1e-9 relative. `H = AᵀA + damping`
/// on a random `A`.
#[test]
fn feedback_factor_identity() {
    let (n, t) = (300usize, 1200usize);
    let mut s = 0x9E37_79B9_7F4A_7C15u64;

    // H = AᵀA + damping·I, A random [t, n].
    let mut a = vec![0f64; t * n];
    for x in &mut a {
        *x = rng_f64(&mut s) - 0.5;
    }
    let mut h = vec![0f64; n * n];
    for r in 0..t {
        let row = &a[r * n..(r + 1) * n];
        for i in 0..n {
            let xi = row[i];
            let hr = &mut h[i * n..(i + 1) * n];
            for j in 0..n {
                hr[j] += xi * row[j];
            }
        }
    }
    let mean_diag: f64 = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let damping = 0.01 * mean_diag;
    for i in 0..n {
        h[i * n + i] += damping;
    }

    // New path: U = V⁻¹ with H = V Vᵀ.
    let v = ul_cholesky(&h, n).expect("ul_cholesky");
    let u_new = upper_tri_inverse(&v, n).expect("upper_tri_inverse");

    // Identity Uᵀ U H = I via P = U·H, then M = Uᵀ·P.
    let mut p = vec![0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let uik = u_new[i * n + k];
            if uik == 0.0 {
                continue;
            }
            for j in 0..n {
                p[i * n + j] += uik * h[k * n + j];
            }
        }
    }
    let mut m = vec![0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let uki = u_new[k * n + i];
            if uki == 0.0 {
                continue;
            }
            for j in 0..n {
                m[i * n + j] += uki * p[k * n + j];
            }
        }
    }
    let mut max_id_err = 0f64;
    for i in 0..n {
        for j in 0..n {
            let target = if i == j { 1.0 } else { 0.0 };
            max_id_err = max_id_err.max((m[i * n + j] - target).abs());
        }
    }
    assert!(
        max_id_err <= 1e-8,
        "max|UᵀU·H − I| = {max_id_err:.3e} exceeds 1e-8"
    );

    // Prior path reference.
    let mut ident = vec![0f64; n * n];
    for i in 0..n {
        ident[i * n + i] = 1.0;
    }
    let hinv = solve_spd(&h, &ident, n, n).expect("solve_spd on identity");
    let u_old = cholesky_upper_ref(&hinv, n);

    let mut max_diff = 0f64;
    let mut scale = 0f64;
    for i in 0..n * n {
        max_diff = max_diff.max((u_new[i] - u_old[i]).abs());
        scale = scale.max(u_old[i].abs());
    }
    let rel = max_diff / scale;
    assert!(
        rel <= 1e-9,
        "new vs old relative difference {rel:.3e} exceeds 1e-9 (max|ΔU|={max_diff:.3e}, scale={scale:.3e})"
    );
    println!("feedback_factor_identity n={n}: max|UᵀU·H−I|={max_id_err:.3e} rel(ΔU)={rel:.3e}");
}
