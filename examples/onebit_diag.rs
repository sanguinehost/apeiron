//! Diagnostic harness: the W7 one-bit arms on held-out `ffn_down` Grams.
//!
//! For every `ffn_down.weight` in the f16 model within `--index-range`, this quantizes each row with
//! every one-bit arm of [`apeiron::learn::onebit`] (groups of 128, scale `a = mean|w|`) plus the
//! in-tree ternary reference [`gptq_ternary`] at `levels = 1`, `group = 128`, and reports the
//! held-out and calibration relative output error
//! `output_error(W, Ŵ, H) / output_error(W, 0, H)`, the measured `sd1` stability `stab = max|e|/a`
//! (`nan` for the feedback and rotation arms), and the bits/weight.
//!
//! Usage:
//! ```text
//! onebit_diag --f16 <f16.gguf> --gram <calib/> --gram-eval <eval/> [--index-range lo,hi]
//! ```
//!
//! `--index-range lo,hi` selects the half-open sub-range `[lo, hi)` of the model's `ffn_down`
//! tensors, so a run fits the wall-clock budget. `output_error(W, 0, H)` is `1` by construction
//! (the metric is already normalised by `tr[W H Wᵀ]`), so the printed ratio is `output_error` once.
#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    reason = "measurement harness: numeric casts and flat row-major index arithmetic are the \
              payload format, and the short reference names mirror examples/gptq_diag.rs"
)]

use apeiron::format::gguf::Gguf;
use apeiron::format::quant::f16_to_f32;
use apeiron::learn::gptq::{gptq_ternary, output_error};
use apeiron::learn::onebit::{quantize, Arm, ONEBIT_BPW, ONEBIT_GROUP};
use std::path::{Path, PathBuf};

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

struct Job {
    layer: usize,
    w: Vec<f32>,
    n_out: usize,
    n_in: usize,
    calib_path: PathBuf,
    eval_path: PathBuf,
}

struct Line {
    name: String,
    wq: Vec<f32>,
    rel_calib: f64,
    stab: f64,
}

/// Quantize every arm on one tensor and measure the calibration error; the held-out measurement is
/// taken by [`finish`] after the calibration Gram is dropped.
fn run_calib(job: &Job) -> (Vec<f32>, Vec<Line>) {
    let calib = read_f64(&job.calib_path);
    assert_eq!(calib.len(), job.n_in * job.n_in, "calib gram size");

    let mut lines = Vec::with_capacity(Arm::ALL.len() + 1);
    for arm in Arm::ALL {
        let r = quantize(arm, &job.w, job.n_out, job.n_in, &calib, ONEBIT_GROUP)
            .unwrap_or_else(|e| panic!("quantize {}: {e}", arm.name()));
        let rel_calib = output_error(&job.w, &r.what, job.n_out, job.n_in, &calib)
            .unwrap_or_else(|e| panic!("output_error {} calib: {e}", arm.name()));
        lines.push(Line {
            name: arm.name().to_string(),
            wq: r.what,
            rel_calib,
            stab: r.stab,
        });
    }

    let reference = gptq_ternary(&job.w, job.n_out, job.n_in, &calib, 1, ONEBIT_GROUP)
        .expect("gptq_ternary levels=1");
    let rel_calib = output_error(&job.w, &reference.wq, job.n_out, job.n_in, &calib)
        .expect("output_error reference calib");
    lines.push(Line {
        name: "ref-gptq3".to_string(),
        wq: reference.wq,
        rel_calib,
        stab: f64::NAN,
    });
    (job.w.clone(), lines)
}

/// Measure the held-out error and format one `ARM` line per arm.
fn finish(job: &Job, w: &[f32], lines: Vec<Line>) -> Vec<String> {
    let eval = read_f64(&job.eval_path);
    assert_eq!(eval.len(), job.n_in * job.n_in, "eval gram size");
    lines
        .into_iter()
        .map(|l| {
            let rel_eval = output_error(w, &l.wq, job.n_out, job.n_in, &eval)
                .unwrap_or_else(|e| panic!("output_error {} eval: {e}", l.name));
            let stab = if l.stab.is_nan() {
                "nan".to_string()
            } else {
                format!("{:.6}", l.stab)
            };
            format!(
                "ARM blk={} arm={} rel_err={:.6e} rel_err_calib={:.6e} stab={stab} bpw={ONEBIT_BPW:.6}",
                job.layer, l.name, rel_eval, l.rel_calib
            )
        })
        .collect()
}

fn run(job: &Job) -> Vec<String> {
    let (w, lines) = run_calib(job);
    finish(job, &w, lines)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(f16_path), Some(calib_dir), Some(eval_dir)) = (
        arg(&args, "--f16"),
        arg(&args, "--gram"),
        arg(&args, "--gram-eval"),
    ) else {
        eprintln!(
            "usage: onebit_diag --f16 <f16.gguf> --gram <calib/> --gram-eval <eval/> \
             [--index-range lo,hi]"
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

    let f16 = Gguf::open(&f16_path).expect("open f16");
    let layer_of = |n: &str| -> usize {
        n.rsplit('.')
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    };
    let mut names: Vec<String> = f16
        .header
        .tensors
        .iter()
        .map(|t| String::from_utf8_lossy(&t.name).into_owned())
        .filter(|n| n.ends_with(".ffn_down.weight"))
        .collect();
    // GGUF name order is lexicographic (`blk.10` before `blk.2`); order by block number so
    // `--index-range lo,hi` selects `[lo, hi)` by layer.
    names.sort_by_key(|n| layer_of(n));
    println!(
        "ffn_down tensors: {} layers={:?}",
        names.len(),
        names.iter().map(|n| layer_of(n)).collect::<Vec<_>>()
    );

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
        let layer: usize = name
            .rsplit('.')
            .nth(2)
            .expect("blk.<i>")
            .parse()
            .expect("layer");
        jobs.push(Job {
            layer,
            w,
            n_out,
            n_in,
            calib_path: Path::new(&calib_dir).join(format!("blk.{layer}.ffn_down_in.f64")),
            eval_path: Path::new(&eval_dir).join(format!("blk.{layer}.ffn_down_in.f64")),
        });
    }
    println!("RUN range=[{lo},{hi}) tensors={}", jobs.len());

    let results: Vec<Vec<String>> = std::thread::scope(|s| {
        let handles: Vec<_> = jobs.iter().map(|job| s.spawn(move || run(job))).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for lines in results {
        for line in lines {
            println!("{line}");
        }
    }
}
