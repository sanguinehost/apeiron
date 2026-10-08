//! `tests/learn_seal.rs` — the seal core: `Q4_K` layout round-trip, the calibrated-beats-
//! uncalibrated thesis, uniform/none equivalence, input refusals, and a real `Q4_K` tensor
//! re-quantization. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    reason = "the fixed-seed LCG maps a 24-bit integer field to a float in [0, 1); the narrowing is \
              the point of the pseudorandom stream."
)]

use apeiron::format::gguf::parse_header;
use apeiron::format::quant::{
    dequant_q4_k_block, dequant_q6_k_block, dequant_super_block, f16_to_f32, q5_0_codes, q5_0_d,
    required_bytes, GGML_TYPE_Q4_K, Q4_K_BLOCK_BYTES, Q5_0_BLOCK_BYTES, Q6_K_BLOCK_BYTES, QK_K,
};
use apeiron::learn::seal::{
    channel_energy, quantize_matrix_q4_k, quantize_row_q4_k, quantize_row_q5_0, quantize_row_q6_k,
    SealError,
};

/// A small fixed-seed 64-bit LCG (no new crates), returning a value in `[-0.5, 0.5)`.
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 40) as f32 / 16_777_216.0) - 0.5
}

/// `‖X(W−Wq)‖²` — the full forward error over `t` tokens of a `n_out × n_in` row-major matrix.
fn function_distortion(
    x: &[f32],
    w: &[f32],
    wq: &[f32],
    t: usize,
    n_out: usize,
    n_in: usize,
) -> f64 {
    let mut total = 0f64;
    for ti in 0..t {
        for r in 0..n_out {
            let mut acc = 0f64;
            for j in 0..n_in {
                let xv = f64::from(x[ti * n_in + j]);
                acc += xv * (f64::from(w[r * n_in + j]) - f64::from(wq[r * n_in + j]));
            }
            total += acc * acc;
        }
    }
    total
}

/// Dequantize every `Q4_K` block of a row-major matrix back to `f32`.
fn dequant_matrix(bytes: &[u8], n_out: usize, n_in: usize) -> Vec<f32> {
    let nb_row = n_in / QK_K;
    let mut out = Vec::with_capacity(n_out * n_in);
    let mut tmp = [0f32; QK_K];
    for r in 0..n_out {
        for b in 0..nb_row {
            let off = (r * nb_row + b) * Q4_K_BLOCK_BYTES;
            dequant_q4_k_block(&bytes[off..off + Q4_K_BLOCK_BYTES], &mut tmp).unwrap();
            out.extend_from_slice(&tmp);
        }
    }
    out
}

/// Every block dequantizes, the output is `144` bytes per `256` weights, and the per-group error
/// stays within one quantization step of that group's range.
#[test]
fn layout_round_trips() {
    let n = 768usize; // three super-blocks
    let mut seed = 0x0000_C0FF_EE00_1234u64;
    let row: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 4.0).collect();
    let bytes = quantize_row_q4_k(&row, None).unwrap();
    assert_eq!(bytes.len(), n / QK_K * Q4_K_BLOCK_BYTES);

    let mut out = [0f32; QK_K];
    for b in 0..n / QK_K {
        let blk = &bytes[b * Q4_K_BLOCK_BYTES..(b + 1) * Q4_K_BLOCK_BYTES];
        dequant_q4_k_block(blk, &mut out).unwrap();
        for g in 0..8 {
            let grp = &row[b * QK_K + g * 32..b * QK_K + g * 32 + 32];
            let hi = grp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lo = grp.iter().copied().fold(f32::INFINITY, f32::min);
            let bound = (hi - lo) / 15.0 + 1e-3;
            for (i, &orig) in grp.iter().enumerate() {
                let err = (orig - out[g * 32 + i]).abs();
                assert!(
                    err <= bound,
                    "block {b} group {g} element {i}: |{orig} - {}| = {err} > {bound}",
                    out[g * 32 + i]
                );
            }
        }
    }
}

/// Calibrating the importance with the input channel energy `τ` lowers the true forward error
/// `‖X(W−Wq)‖²` below the uniform-importance fit.
#[test]
fn calibrated_beats_uncalibrated() {
    let (t, n_out, n_in) = (64usize, 16usize, 512usize);
    let mut seed = 0x1234_5678_9abc_def0u64;
    // Some input channels carry 30x the energy of the rest.
    let gain: Vec<f32> = (0..n_in)
        .map(|j| if j % 8 == 0 { 30.0 } else { 1.0 })
        .collect();
    let x: Vec<f32> = (0..t * n_in)
        .map(|i| lcg(&mut seed) * gain[i % n_in])
        .collect();
    let w: Vec<f32> = (0..n_out * n_in).map(|_| lcg(&mut seed)).collect();

    let tau = channel_energy(&x, t, n_in).unwrap();
    let bytes_cal = quantize_matrix_q4_k(&w, n_out, n_in, Some(&tau)).unwrap();
    let bytes_unc = quantize_matrix_q4_k(&w, n_out, n_in, None).unwrap();
    let wq_cal = dequant_matrix(&bytes_cal, n_out, n_in);
    let wq_unc = dequant_matrix(&bytes_unc, n_out, n_in);

    let d_cal = function_distortion(&x, &w, &wq_cal, t, n_out, n_in);
    let d_unc = function_distortion(&x, &w, &wq_unc, t, n_out, n_in);
    eprintln!("calibrated={d_cal:.6e} uncalibrated={d_unc:.6e}");
    assert!(
        d_cal < d_unc,
        "calibrated {d_cal:.6e} must be < uncalibrated {d_unc:.6e}"
    );
}

/// Uniform importance (all ones) is the `None` path: the same weighting with `qw ≡ 1`, so the bytes
/// are identical.
#[test]
fn uniform_weights_match_none() {
    let n = 512usize;
    let mut seed = 0x0000_DEAD_BEEF_0000u64;
    let row: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 2.0).collect();
    let ones = vec![1f32; n];
    let a = quantize_row_q4_k(&row, None).unwrap();
    let b = quantize_row_q4_k(&row, Some(&ones)).unwrap();
    assert_eq!(
        a, b,
        "uniform importance must reproduce the None path byte-for-byte"
    );
}

/// Bad shapes, a NaN, an infinity and a weights-length mismatch all refuse instead of panicking.
#[test]
fn refuses_bad_input() {
    assert!(matches!(
        quantize_row_q4_k(&vec![0f32; 200], None),
        Err(SealError::NotMultiple { .. })
    ));

    let mut nan = vec![0f32; QK_K];
    nan[7] = f32::NAN;
    assert!(quantize_row_q4_k(&nan, None).is_err());

    let mut inf = vec![0f32; QK_K];
    inf[3] = f32::INFINITY;
    assert!(quantize_row_q4_k(&inf, None).is_err());

    let row = vec![0f32; QK_K];
    assert!(quantize_row_q4_k(&row, Some(&vec![1f32; QK_K - 1])).is_err());
    let bad_w = vec![f32::NAN; QK_K];
    assert!(quantize_row_q4_k(&row, Some(&bad_w)).is_err());

    // channel_energy shape and finiteness.
    assert!(channel_energy(&[0f32; 10], 2, 6).is_err());
    assert!(channel_energy(&[f32::NAN; 12], 2, 6).is_err());
}

/// On the real `Qwen2.5-0.5B Q4_K_M` fixture: dequantize one `Q4_K` tensor to `f32`, re-quantize
/// with uniform importance, and confirm the re-dequantized error is at the block's quantization
/// step. Skips when the fixture is absent.
#[test]
fn real_tensor_distortion() {
    let Some(path) = apeiron::engine::testdata::model_path("models/gguf/qwen05b-Q4_K_M.gguf")
    else {
        eprintln!("skip: fixture absent");
        return;
    };
    let bytes = std::fs::read(&path).expect("read fixture");
    let header = parse_header(&bytes, bytes.len() as u64).expect("parse fixture header");
    let data0 = usize::try_from(header.data0).expect("data0 fits usize");

    // Prefer the named tensor, else the first `Q4_K` tensor with a whole number of blocks.
    let td = header
        .tensors
        .iter()
        .find(|t| {
            t.ggml_type == GGML_TYPE_Q4_K
                && t.name.windows(8).any(|w| w == b"ffn_down")
                && t.dims.iter().product::<u64>() % QK_K as u64 == 0
        })
        .or_else(|| {
            header.tensors.iter().find(|t| {
                t.ggml_type == GGML_TYPE_Q4_K && t.dims.iter().product::<u64>() % QK_K as u64 == 0
            })
        })
        .expect("a Q4_K tensor in the fixture");
    let n = usize::try_from(td.dims.iter().product::<u64>()).expect("tensor size fits usize");
    let need = usize::try_from(required_bytes(GGML_TYPE_Q4_K, n as u64).unwrap()).unwrap();
    let off = data0 + usize::try_from(td.offset).expect("offset fits usize");
    let raw = &bytes[off..off + need];

    let mut orig = vec![0f32; n];
    dequant_super_block(GGML_TYPE_Q4_K, raw, n, &mut orig).unwrap();

    let requant = quantize_row_q4_k(&orig, None).unwrap();
    let mut back = vec![0f32; n];
    dequant_super_block(GGML_TYPE_Q4_K, &requant, n, &mut back).unwrap();

    let max_err = orig
        .iter()
        .zip(&back)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let absmax = orig.iter().map(|v| v.abs()).fold(0f32, f32::max);
    eprintln!(
        "real tensor {} n={n} absmax={absmax:.6e} max_err={max_err:.6e}",
        String::from_utf8_lossy(&td.name)
    );
    assert!(
        max_err <= absmax / 15.0 + 1e-2,
        "re-quant max error {max_err} exceeds one quantization step of absmax {absmax}"
    );
}

/// The `τ`-weighted distortion `Σ_j τ_j (x_j − x̂_j)²` of a row.
fn tau_distortion(x: &[f32], xq: &[f32], tau: &[f32]) -> f64 {
    x.iter()
        .zip(xq)
        .zip(tau)
        .map(|((&a, &b), &t)| f64::from(t) * (f64::from(a) - f64::from(b)).powi(2))
        .sum()
}

/// Dequantize a `Q6_K` row of 210-byte super-blocks back to `f32`.
fn dequant_row_q6_k(bytes: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut tmp = [0f32; QK_K];
    for b in 0..n / QK_K {
        let off = b * Q6_K_BLOCK_BYTES;
        dequant_q6_k_block(&bytes[off..off + Q6_K_BLOCK_BYTES], &mut tmp).unwrap();
        out.extend_from_slice(&tmp);
    }
    out
}

/// Dequantize a `Q5_0` row of 22-byte blocks back to `f32`: `w = d·code`.
fn dequant_row_q5_0(bytes: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for b in 0..n / 32 {
        let blk = &bytes[b * Q5_0_BLOCK_BYTES..(b + 1) * Q5_0_BLOCK_BYTES];
        let d = f16_to_f32(q5_0_d(blk).unwrap());
        out.extend(q5_0_codes(blk).unwrap().iter().map(|&c| d * f32::from(c)));
    }
    out
}

/// `Q6_K`: the `τ`-weighted fit beats the uniform fit, and each 256-weight super-block is 210 bytes.
#[test]
fn q6_k_weighted_fit_and_layout() {
    let n = 256usize;
    let mut seed = 0x0000_0000_0BAD_F00Du64;
    let row: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 3.0).collect();
    // Two hot channels per 16-weight group carry 200× the energy.
    let tau: Vec<f32> = (0..n)
        .map(|j| {
            if j % 16 == 3 || j % 16 == 11 {
                200.0
            } else {
                1.0
            }
        })
        .collect();

    let bytes_w = quantize_row_q6_k(&row, Some(&tau)).unwrap();
    let bytes_u = quantize_row_q6_k(&row, None).unwrap();
    assert_eq!(bytes_w.len(), n / QK_K * Q6_K_BLOCK_BYTES);
    assert_eq!(bytes_w.len(), 210, "Q6_K is 210 bytes per 256 weights");

    let d_w = tau_distortion(&row, &dequant_row_q6_k(&bytes_w, n), &tau);
    let d_u = tau_distortion(&row, &dequant_row_q6_k(&bytes_u, n), &tau);
    eprintln!("Q6_K tau-weighted={d_w:.6e} uniform={d_u:.6e}");
    assert!(
        d_w < d_u,
        "τ-weighted distortion {d_w:.6e} must be < uniform {d_u:.6e}"
    );
}

/// `Q5_0`: the `τ`-weighted fit beats the reference fit, and each 32-weight block is 22 bytes.
#[test]
fn q5_0_weighted_fit_and_layout() {
    let n = 64usize; // two Q5_0 blocks
    let mut seed = 0x0000_0000_FEED_5EEDu64;
    let row: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 2.0).collect();
    // One hot channel per 8 carries 300× the energy.
    let tau: Vec<f32> = (0..n)
        .map(|j| if j % 8 == 2 { 300.0 } else { 1.0 })
        .collect();

    let bytes_w = quantize_row_q5_0(&row, Some(&tau)).unwrap();
    let bytes_u = quantize_row_q5_0(&row, None).unwrap();
    assert_eq!(bytes_w.len(), n / 32 * Q5_0_BLOCK_BYTES);
    assert_eq!(bytes_w.len(), 2 * 22, "Q5_0 is 22 bytes per 32 weights");

    let d_w = tau_distortion(&row, &dequant_row_q5_0(&bytes_w, n), &tau);
    let d_u = tau_distortion(&row, &dequant_row_q5_0(&bytes_u, n), &tau);
    eprintln!("Q5_0 tau-weighted={d_w:.6e} reference={d_u:.6e}");
    assert!(
        d_w < d_u,
        "τ-weighted distortion {d_w:.6e} must be < reference {d_u:.6e}"
    );
}
