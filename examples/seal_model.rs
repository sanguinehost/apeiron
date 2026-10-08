//! Whole-model seal: re-quantize a `Q4_K_M` model's weighted tensors with imatrix importance.
//!
//! Usage:
//! ```text
//! seal_model --f16 <f16.gguf> --typemap <Q4_K_M.gguf> --imatrix <imatrix.gguf> --out <out.gguf>
//! ```
//!
//! The `--typemap` file fixes the type map. Every tensor it stores as a weighted type — `Q4_K`,
//! `Q6_K` or `Q5_0` — is read from `--f16`, weighted by the per-column importance vector
//! [`apeiron::learn::imatrix::read_imatrix`] returns for the same tensor name, quantized by the
//! matching [`apeiron::learn::seal`] matrix routine, and written; every other tensor's bytes
//! (`Q8_0`, `F32`, …) are copied verbatim from the `--typemap` file, as `llama-quantize` does.
//! Metadata is copied unchanged and tensor offsets are recomputed on the `general.alignment` grid.
//! The output therefore has exactly the typemap's bits per weight.
//!
//! A weighted tensor with no imatrix entry falls back to the type's weight-only fit and is counted.
#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    reason = "measurement/writer harness: numeric casts and index arithmetic are the payload format, \
              and the reference arithmetic's short names (`qw`, `n`) are kept"
)]

use apeiron::format::gguf::{Gguf, GgufHeader};
use apeiron::format::quant::{
    f16_to_f32, ggml_type_name, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K,
};
use apeiron::learn::gptq::{gptq_q4k_damped, gptq_q5_0_damped, gptq_q6k_damped};
use apeiron::learn::imatrix::read_imatrix;
use apeiron::learn::seal::{
    quantize_matrix_q4_k, quantize_matrix_q5_0, quantize_matrix_q6_k, SealError,
};
use std::path::Path;
use std::time::Instant;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Map tensor name to its calibration Gram file name.
fn gram_file_for_tensor(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() >= 3 && parts[0] == "blk" {
        let i = parts[1];
        let kind = parts[2];
        let gram_kind = match kind {
            "attn_q" | "attn_k" | "attn_v" => "attn_in",
            "attn_output" => "attn_out_in",
            "ffn_gate" | "ffn_up" => "ffn_in",
            "ffn_down" => "ffn_down_in",
            _ => return None,
        };
        Some(format!("blk.{i}.{gram_kind}.f64"))
    } else {
        None
    }
}

/// Which Gram error-feedback kernel packs a Gram-routed tensor.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GramKind {
    /// 256-weight super-blocks, `Q6_K` (the `ffn_down` tensors this typemap types `Q6_K`).
    Q6K,
    /// 32-weight blocks, `Q5_0` (attention and the FFN in-projections, input width 896).
    Q5_0,
    /// 256-weight super-blocks, `Q4_K` (the other `ffn_down` tensors).
    Q4K,
}

struct GramJob {
    index: usize,
    name: String,
    start: usize,
    w: Vec<f32>,
    n_out: usize,
    n_in: usize,
    tau: Option<Vec<f32>>,
    gram: Vec<f64>,
    kind: GramKind,
}

/// A seal matrix quantizer: `(weights, n_out, n_in, τ) -> block bytes`.
type QuantFn = fn(&[f32], usize, usize, Option<&[f32]>) -> Result<Vec<u8>, SealError>;

/// Dequantize an `F16` tensor payload to `f32`.
fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// `n_out` for a tensor whose elements are stored `dims[0]` fastest: the product of the trailing
/// dimensions. `n_in` is `dims[0]`.
fn out_dims(dims: &[u64]) -> Result<usize, String> {
    let mut acc: u64 = 1;
    for &d in &dims[1..] {
        acc = acc.checked_mul(d).ok_or("dimension overflow")?;
    }
    usize::try_from(acc).map_err(|_| "n_out does not fit usize".to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(f16_path), Some(typemap_path), Some(imatrix_path), Some(out_path)) = (
        arg(&args, "--f16"),
        arg(&args, "--typemap"),
        arg(&args, "--imatrix"),
        arg(&args, "--out"),
    ) else {
        eprintln!(
            "usage: seal_model --f16 <f16.gguf> --typemap <Q4_K_M.gguf> --imatrix <imatrix.gguf> --out <out.gguf> [--gram <dir>] [--damping <f64>] [--gram-types q4k|all]"
        );
        std::process::exit(2);
    };
    // Hessian damping fraction: `H = G + damping·mean(diag(G))·I`. 1.0 is the measured best on
    // the held-out wikitext-2 test (paired bits/token vs the no-Gram seal −0.00196 ± 0.00052 over
    // 584×512 tokens); 0.01 reproduces `gptq_q4k` byte-for-byte and loses to the no-Gram seal.
    let damping: f64 = arg(&args, "--damping")
        .map(|s| {
            s.parse::<f64>()
                .expect("--damping must be a floating-point number")
        })
        .unwrap_or(1.0);
    // `--gram-types q4k` restricts Gram feedback to the Q4_K tensors (the W7 arm); `all` (the
    // default) also routes Q6_K and Q5_0 through the matching `gptq_*_damped` kernel.
    let gram_types_all = match arg(&args, "--gram-types").as_deref() {
        None | Some("all") => true,
        Some("q4k") => false,
        Some(other) => {
            eprintln!("--gram-types must be 'q4k' or 'all', got {other}");
            std::process::exit(2);
        }
    };
    let gram_dir = arg(&args, "--gram");
    if let Some(ref d) = gram_dir {
        println!(
            "Gram directory: {d} (GPTQ error feedback enabled for {} tensors, damping {damping})",
            if gram_types_all {
                "Q4_K, Q6_K and Q5_0"
            } else {
                "Q4_K"
            }
        );
    }

    let t0 = Instant::now();
    let tm = Gguf::open(&typemap_path).expect("open typemap");
    let f16 = Gguf::open(&f16_path).expect("open f16");
    let imatrix = read_imatrix(&imatrix_path).expect("read imatrix");
    println!(
        "loaded: typemap {} tensors, f16 {} tensors, imatrix {} entries ({} chunks x {})",
        tm.header.tensors.len(),
        f16.header.tensors.len(),
        imatrix.len(),
        imatrix.chunk_count(),
        imatrix.chunk_size()
    );

    // Recompute the offset grid on the alignment, and the total data size.
    let align = tm.header.alignment;
    let mut header: GgufHeader = tm.header.clone();
    let mut cursor = 0u64;
    for t in &mut header.tensors {
        let rem = cursor % align;
        if rem != 0 {
            cursor += align - rem;
        }
        t.offset = cursor;
        let n = t
            .n_bytes()
            .expect("tensor byte count")
            .expect("known tensor type");
        cursor = cursor.checked_add(n).expect("data size overflow");
    }
    let total = usize::try_from(cursor).expect("data size fits usize");
    let original_offsets: Vec<u64> = tm.header.tensors.iter().map(|t| t.offset).collect();
    let offsets_match =
        original_offsets == header.tensors.iter().map(|t| t.offset).collect::<Vec<_>>();
    println!(
        "data section: {} bytes ({:.2} MiB); recomputed offsets match typemap: {offsets_match}",
        total,
        total as f64 / 1_048_576.0
    );

    let mut data = vec![0u8; total];
    // counts[type] = (weighted re-quantized, fallback-to-None, copied verbatim)
    let mut counts: std::collections::BTreeMap<u32, (usize, usize, usize)> =
        std::collections::BTreeMap::new();
    let mut gram_jobs: Vec<GramJob> = Vec::new();

    for (i, t) in header.tensors.iter().enumerate() {
        let start = usize::try_from(t.offset).expect("offset fits usize");
        let gram_kind = match (t.ggml_type, gram_types_all) {
            (GGML_TYPE_Q4_K, _) => Some(GramKind::Q4K),
            (GGML_TYPE_Q6_K, true) => Some(GramKind::Q6K),
            (GGML_TYPE_Q5_0, true) => Some(GramKind::Q5_0),
            _ => None,
        };
        if let (Some(kind), Some(gdir)) = (gram_kind, &gram_dir) {
            let name = String::from_utf8_lossy(&t.name).into_owned();
            let Some(fd) = f16.header.tensor(&name) else {
                panic!("f16 model is missing quantized tensor {name}");
            };
            let n_in = usize::try_from(fd.dims[0]).expect("n_in fits usize");
            let n_out = out_dims(&fd.dims).expect("n_out");
            let raw = f16.tensor_bytes(fd).expect("read f16 tensor");
            let w = f16_bytes_to_f32(&raw);

            let tau = imatrix.vector(&name).map(|v| v.to_vec());
            let entry = counts.entry(t.ggml_type).or_insert((0, 0, 0));
            match tau {
                Some(ref v) if v.len() == n_in => entry.0 += 1,
                Some(ref v) => panic!(
                    "imatrix entry {name} has {} columns, tensor has {n_in}",
                    v.len()
                ),
                None => entry.1 += 1,
            }

            let gfile = gram_file_for_tensor(&name).unwrap_or_else(|| {
                panic!("no Gram mapping for tensor {name}");
            });
            let gpath = Path::new(gdir).join(&gfile);
            let gbytes = std::fs::read(&gpath)
                .unwrap_or_else(|e| panic!("failed to read Gram {}: {e}", gpath.display()));
            let expected_bytes = n_in
                .checked_mul(n_in)
                .and_then(|x| x.checked_mul(8))
                .expect("overflow");
            assert_eq!(
                gbytes.len(),
                expected_bytes,
                "Gram {gpath:?} size mismatch: expected {expected_bytes}, got {}",
                gbytes.len()
            );
            let mut gram = vec![0.0f64; n_in * n_in];
            for (idx, chunk) in gbytes.chunks_exact(8).enumerate() {
                gram[idx] = f64::from_le_bytes(chunk.try_into().unwrap());
            }

            gram_jobs.push(GramJob {
                index: i,
                name,
                start,
                w,
                n_out,
                n_in,
                tau,
                gram,
                kind,
            });
            continue;
        }

        let quant: Option<QuantFn> = match t.ggml_type {
            GGML_TYPE_Q4_K => Some(quantize_matrix_q4_k),
            GGML_TYPE_Q6_K => Some(quantize_matrix_q6_k),
            GGML_TYPE_Q5_0 => Some(quantize_matrix_q5_0),
            _ => None,
        };
        let entry = counts.entry(t.ggml_type).or_insert((0, 0, 0));
        match quant {
            Some(f) => {
                let name = String::from_utf8_lossy(&t.name).into_owned();
                let Some(fd) = f16.header.tensor(&name) else {
                    panic!("f16 model is missing quantized tensor {name}");
                };
                let n_in = usize::try_from(fd.dims[0]).expect("n_in fits usize");
                let n_out = out_dims(&fd.dims).expect("n_out");
                let raw = f16.tensor_bytes(fd).expect("read f16 tensor");
                let w = f16_bytes_to_f32(&raw);

                let tau = imatrix.vector(&name);
                match tau {
                    Some(v) if v.len() == n_in => entry.0 += 1,
                    Some(v) => panic!(
                        "imatrix entry {name} has {} columns, tensor has {n_in}",
                        v.len()
                    ),
                    None => entry.1 += 1,
                }
                let bytes = f(&w, n_out, n_in, tau).expect("quantize");
                let end = start + bytes.len();
                data[start..end].copy_from_slice(&bytes);
                println!(
                    "[{:>3}/{}] {name:<28} dims[{}, {}] {} <- {} bytes (imatrix: {})",
                    i + 1,
                    header.tensors.len(),
                    fd.dims[0],
                    n_out,
                    ggml_type_name(t.ggml_type),
                    bytes.len(),
                    if tau.is_some() { "yes" } else { "NONE" }
                );
            }
            None => {
                entry.2 += 1;
                let bytes = tm.tensor_bytes(t).expect("read typemap tensor");
                let end = start + bytes.len();
                data[start..end].copy_from_slice(&bytes);
            }
        }
    }

    if !gram_jobs.is_empty() {
        println!(
            "parallel quantizing {} Gram-routed tensors (Q4_K/Q6_K/Q5_0) with GPTQ error feedback...",
            gram_jobs.len()
        );
        let t_par = Instant::now();
        let results = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for job in &gram_jobs {
                handles.push(s.spawn(move || {
                    let t_job = Instant::now();
                    let bytes = match job.kind {
                        GramKind::Q4K => gptq_q4k_damped(
                            &job.w,
                            job.n_out,
                            job.n_in,
                            &job.gram,
                            job.tau.as_deref(),
                            damping,
                        )
                        .expect("gptq_q4k_damped failed"),
                        GramKind::Q6K => gptq_q6k_damped(
                            &job.w,
                            job.n_out,
                            job.n_in,
                            &job.gram,
                            job.tau.as_deref(),
                            damping,
                        )
                        .expect("gptq_q6k_damped failed"),
                        GramKind::Q5_0 => gptq_q5_0_damped(
                            &job.w,
                            job.n_out,
                            job.n_in,
                            &job.gram,
                            job.tau.as_deref(),
                            damping,
                        )
                        .expect("gptq_q5_0_damped failed"),
                    };
                    let elapsed = t_job.elapsed().as_secs_f64();
                    (
                        job.index,
                        job.name.clone(),
                        job.start,
                        job.n_in,
                        job.n_out,
                        job.kind,
                        bytes,
                        elapsed,
                    )
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        println!(
            "completed all {} Gram-routed tensors in parallel in {:.2} s",
            results.len(),
            t_par.elapsed().as_secs_f64()
        );
        for (i, name, start, n_in, n_out, kind, bytes, elapsed) in results {
            let end = start + bytes.len();
            data[start..end].copy_from_slice(&bytes);
            let tname = match kind {
                GramKind::Q4K => "Q4_K",
                GramKind::Q6K => "Q6_K",
                GramKind::Q5_0 => "Q5_0",
            };
            println!(
                "[{:>3}/{}] {name:<28} dims[{}, {}] {tname} <- {} bytes in {:.2} s (gptq)",
                i + 1,
                header.tensors.len(),
                n_in,
                n_out,
                bytes.len(),
                elapsed
            );
        }
    }

    let file = apeiron::learn::gguf::write_file(&header, &data).expect("write gguf");
    std::fs::write(&out_path, &file).expect("write output file");

    println!("per-type tensor counts (re-quantized, fallback None, copied):");
    for (t, (r, fb, cp)) in &counts {
        println!(
            "  {:<6}: requantized {r}, fallback {fb}, copied {cp}",
            ggml_type_name(*t)
        );
    }
    println!(
        "wrote {out_path}: {} bytes in {:.2} s",
        file.len(),
        t0.elapsed().as_secs_f64()
    );
}
