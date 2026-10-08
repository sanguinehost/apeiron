//! Diagnostic harness: on which side does `gptq_q4k`'s Q4_K error feedback lose KLD?
//!
//! For every `ffn_down` tensor the typemap types `Q4_K`, this compares the calibration-Gram and
//! held-out-Gram output distortion `output_error(W, Ŵ, G)` of the τ-weighted round-to-block
//! (`quantize_row_q4_k`) against `gptq_q4k` at Hessian damping 0.01, 0.1 and 1.0 × mean(diag).
//! Writes a `DIAG` line per tensor (τ and the 0.01 variant), a `DIAGV` line per damping variant,
//! and — for the first tensor — the `output_error` control.
//!
//! Usage:
//! ```text
//! gptq_diag --f16 <f16.gguf> --typemap <Q4_K_M.gguf> --imatrix <imatrix.gguf> \
//!          --gram <calib/> --gram-eval <eval/> [--index-range lo,hi]
//! ```
//!
//! `--index-range lo,hi` selects the half-open sub-range `[lo, hi)` of the typemap's `Q4_K`
//! `ffn_down` tensors, so a run fits the wall-clock budget.
#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    reason = "measurement harness: numeric casts and flat row-major index arithmetic are the \
              payload format, and the short reference names mirror examples/seal_model.rs"
)]

use apeiron::format::gguf::Gguf;
use apeiron::format::quant::{
    dequant_q4_k_block, f16_to_f32, GGML_TYPE_Q4_K, Q4_K_BLOCK_BYTES, Q4_K_SCALE_SIZE, QK_K,
};
use apeiron::learn::gptq::{gptq_q4k_damped, output_error};
use apeiron::learn::imatrix::read_imatrix;
use apeiron::learn::seal::quantize_row_q4_k;
use std::path::{Path, PathBuf};

/// The damping fractions of `mean(diag(G))` the harness sweeps; index 0 is production `gptq_q4k`.
const DAMPINGS: [f64; 3] = [0.01, 0.1, 1.0];

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// `n_out` for a tensor whose elements are stored `dims[0]` fastest: the product of the trailing
/// dimensions. `n_in` is `dims[0]`.
fn out_dims(dims: &[u64]) -> usize {
    let mut acc: u64 = 1;
    for &d in &dims[1..] {
        acc *= d;
    }
    usize::try_from(acc).expect("n_out fits usize")
}

/// Dequantize an `F16` tensor payload to `f32`.
fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Read a little-endian `f64` matrix (row-major) captured by `capture_gram.py` / `capture_eval.py`.
fn read_f64(path: &Path) -> Vec<f64> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(
        b.len() % 8,
        0,
        "{} is not a whole number of f64",
        path.display()
    );
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Dequantize a packed `out × n` `Q4_K` matrix (`out · n/256` super-blocks) to row-major `f32`.
fn dequant_q4k_matrix(bytes: &[u8], out: usize, n: usize) -> Vec<f32> {
    let nb = n / QK_K;
    assert_eq!(bytes.len(), out * nb * Q4_K_BLOCK_BYTES, "packed Q4_K size");
    let mut wq = vec![0f32; out * n];
    let mut blk = [0f32; QK_K];
    for o in 0..out {
        for b in 0..nb {
            let idx = (o * nb + b) * Q4_K_BLOCK_BYTES;
            dequant_q4_k_block(&bytes[idx..idx + Q4_K_BLOCK_BYTES], &mut blk).expect("dequant");
            wq[o * n + b * QK_K..o * n + (b + 1) * QK_K].copy_from_slice(&blk);
        }
    }
    wq
}

/// The τ-weighted round-to-block baseline: each row quantized by `quantize_row_q4_k(row, τ)`.
fn tau_weighted_bytes(w: &[f32], out: usize, n: usize, tau: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(out * (n / QK_K) * Q4_K_BLOCK_BYTES);
    for o in 0..out {
        let row = &w[o * n..(o + 1) * n];
        bytes.extend_from_slice(&quantize_row_q4_k(row, Some(tau)).expect("quantize_row_q4_k"));
    }
    bytes
}

struct Job {
    name: String,
    w: Vec<f32>,
    n_out: usize,
    n_in: usize,
    tau: Vec<f32>,
    calib_path: PathBuf,
    eval_path: PathBuf,
    /// Compute the `output_error` control on this tensor.
    control: bool,
}

struct Control {
    self_err: f64,
    tau_err: f64,
    flip_err: f64,
}

struct Diag {
    name: String,
    calib_tau: f64,
    eval_tau: f64,
    /// `[calib, eval]` per `DAMPINGS[k]`.
    gptq: [[f64; 2]; 3],
    control: Option<Control>,
}

fn compute_diag(job: &Job) -> Diag {
    let calib = read_f64(&job.calib_path);
    let eval = read_f64(&job.eval_path);
    assert_eq!(calib.len(), job.n_in * job.n_in, "calib gram size");
    assert_eq!(eval.len(), job.n_in * job.n_in, "eval gram size");

    let tau_bytes = tau_weighted_bytes(&job.w, job.n_out, job.n_in, &job.tau);
    let wq_tau = dequant_q4k_matrix(&tau_bytes, job.n_out, job.n_in);
    let calib_tau =
        output_error(&job.w, &wq_tau, job.n_out, job.n_in, &calib).expect("oe tau calib");
    let eval_tau = output_error(&job.w, &wq_tau, job.n_out, job.n_in, &eval).expect("oe tau eval");

    let mut gptq = [[0.0f64; 2]; 3];
    for (k, &d) in DAMPINGS.iter().enumerate() {
        let bytes = gptq_q4k_damped(&job.w, job.n_out, job.n_in, &calib, Some(&job.tau), d)
            .expect("gptq_q4k_damped");
        let wq = dequant_q4k_matrix(&bytes, job.n_out, job.n_in);
        gptq[k][0] = output_error(&job.w, &wq, job.n_out, job.n_in, &calib).expect("oe gptq calib");
        gptq[k][1] = output_error(&job.w, &wq, job.n_out, job.n_in, &eval).expect("oe gptq eval");
    }

    let control = if job.control {
        // Identity gives exactly 0; flipping one code in the τ baseline moves the output error off
        // it. The flip edits the lowest nibble of the first code byte of row 0, block 0 — the
        // dequantized value at column 0 changes by ±one code step.
        let self_err = output_error(&job.w, &job.w, job.n_out, job.n_in, &calib).expect("oe self");
        let mut flipped = tau_bytes.clone();
        flipped[4 + Q4_K_SCALE_SIZE] ^= 0x01;
        let wq_flip = dequant_q4k_matrix(&flipped, job.n_out, job.n_in);
        let flip_err =
            output_error(&job.w, &wq_flip, job.n_out, job.n_in, &calib).expect("oe flip");
        Some(Control {
            self_err,
            tau_err: calib_tau,
            flip_err,
        })
    } else {
        None
    };

    Diag {
        name: job.name.clone(),
        calib_tau,
        eval_tau,
        gptq,
        control,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(f16_path), Some(typemap_path), Some(imatrix_path), Some(calib_dir), Some(eval_dir)) = (
        arg(&args, "--f16"),
        arg(&args, "--typemap"),
        arg(&args, "--imatrix"),
        arg(&args, "--gram"),
        arg(&args, "--gram-eval"),
    ) else {
        eprintln!(
            "usage: gptq_diag --f16 <f16.gguf> --typemap <Q4_K_M.gguf> --imatrix <imatrix.gguf> \
             --gram <calib/> --gram-eval <eval/> [--index-range lo,hi]"
        );
        std::process::exit(2);
    };
    let (lo, hi) = match arg(&args, "--index-range") {
        None => (0usize, usize::MAX),
        Some(s) => {
            let (a, b) = s.split_once(',').expect("--index-range lo,hi");
            (a.parse().expect("lo"), b.parse().expect("hi"))
        }
    };

    let tm = Gguf::open(&typemap_path).expect("open typemap");
    let f16 = Gguf::open(&f16_path).expect("open f16");
    let imatrix = read_imatrix(&imatrix_path).expect("read imatrix");

    // Every `Q4_K` `ffn_down.weight` in typemap order.
    let names: Vec<String> = tm
        .header
        .tensors
        .iter()
        .filter(|t| t.ggml_type == GGML_TYPE_Q4_K)
        .map(|t| String::from_utf8_lossy(&t.name).into_owned())
        .filter(|n| n.ends_with(".ffn_down.weight"))
        .collect();
    println!("Q4_K ffn_down tensors: {} ({:?})", names.len(), names);

    let mut jobs: Vec<Job> = Vec::new();
    for (gi, name) in names.iter().enumerate() {
        if gi < lo || gi >= hi {
            continue;
        }
        let fd = f16
            .header
            .tensor(name)
            .unwrap_or_else(|| panic!("f16 missing {name}"));
        let n_in = usize::try_from(fd.dims[0]).expect("n_in");
        let n_out = out_dims(&fd.dims);
        let raw = f16.tensor_bytes(fd).expect("read f16 tensor");
        let w = f16_bytes_to_f32(&raw);
        assert_eq!(w.len(), n_out * n_in, "{name} weight count");
        let tau = imatrix
            .vector(name)
            .unwrap_or_else(|| panic!("no imatrix entry for {name}"))
            .to_vec();
        assert_eq!(tau.len(), n_in, "{name} τ length");
        let layer = name.split('.').nth(1).expect("blk.<i>");
        jobs.push(Job {
            name: name.clone(),
            w,
            n_out,
            n_in,
            tau,
            calib_path: Path::new(&calib_dir).join(format!("blk.{layer}.ffn_down_in.f64")),
            eval_path: Path::new(&eval_dir).join(format!("blk.{layer}.ffn_down_in.f64")),
            control: gi == 0,
        });
    }
    println!("RUN range=[{lo},{hi}) tensors={}", jobs.len());

    let results: Vec<Diag> = std::thread::scope(|s| {
        let handles: Vec<_> = jobs
            .iter()
            .map(|job| s.spawn(move || compute_diag(job)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut sum_calib_tau = 0.0;
    let mut sum_eval_tau = 0.0;
    let mut sum_calib_gptq = [0.0f64; 3];
    let mut sum_eval_gptq = [0.0f64; 3];
    for d in &results {
        println!(
            "DIAG {} {:.6e} {:.6e} {:.6e} {:.6e}",
            d.name, d.calib_tau, d.gptq[0][0], d.eval_tau, d.gptq[0][1]
        );
        for (k, &f) in DAMPINGS.iter().enumerate() {
            println!(
                "DIAGV {} damp={} {:.6e} {:.6e}",
                d.name, f, d.gptq[k][0], d.gptq[k][1]
            );
        }
        if let Some(c) = &d.control {
            println!(
                "CONTROL {} self={:.6e} tau={:.6e} flipped={:.6e}",
                d.name, c.self_err, c.tau_err, c.flip_err
            );
            assert_eq!(c.self_err, 0.0, "output_error(W, W, G) must be 0");
            assert!(
                c.flip_err > 0.0,
                "flipped τ code must be strictly larger than 0"
            );
        }
        sum_calib_tau += d.calib_tau;
        sum_eval_tau += d.eval_tau;
        for k in 0..3 {
            sum_calib_gptq[k] += d.gptq[k][0];
            sum_eval_gptq[k] += d.gptq[k][1];
        }
    }
    for (k, &f) in DAMPINGS.iter().enumerate() {
        println!(
            "RUNSUM n={} damp={} calib_ratio={:.6} eval_ratio={:.6} sum_calib_tau={:.6e} \
             sum_calib_gptq={:.6e} sum_eval_tau={:.6e} sum_eval_gptq={:.6e}",
            results.len(),
            f,
            sum_calib_gptq[k] / sum_calib_tau,
            sum_eval_gptq[k] / sum_eval_tau,
            sum_calib_tau,
            sum_calib_gptq[k],
            sum_eval_tau,
            sum_eval_gptq[k],
        );
    }
}
