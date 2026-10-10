//! Whole-model one-bit writer (W9): quantize a selected set of linear tensors with a one-bit arm
//! and write a GGUF that `llama-perplexity` loads.
//!
//! Usage:
//! ```text
//! onebit_model --base <gguf> --f16 <f16.gguf> --sel <kinds>[@lo-hi] --out <out.gguf>
//!              (--arm <rtn|gptq1-act|had-gptq1|…> --gram <dir> --store q1_0|f16 | --take <gguf>)
//!              [--threads N]
//! ```
//!
//! Every tensor of `--base` is copied byte for byte, except the selected ones. `--sel` is a comma
//! list of tensor kinds (`attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`,
//! `ffn_down`, or `linear` for all seven), optionally restricted to blocks `lo..=hi`. A selected
//! tensor is either
//! - quantized from `--f16` by [`apeiron::learn::onebit::quantize`] with `--arm` over the
//!   calibration Gram in `--gram` (groups of 128), and stored as `Q1_0` (one sign plane and one
//!   `f16` scale per 128 weights, the llama.cpp `block_q1_0` layout) or as `F16` (the dequantized
//!   `Ŵ`); or
//! - copied from `--take` with its stored type.
//!
//! `Q1_0` storage holds `Ŵ` exactly when every block of `Ŵ` is `±a` for one `a`; the writer checks
//! this per block and exits non-zero otherwise. The Hadamard arms return `R·(±a)`, which `Q1_0`
//! cannot hold without an online `R₁₂₈` on the activations, so they take `--store f16`: the file
//! then carries the function of the 1.125 b/w code, not its storage.
//!
//! Offsets are recomputed on `general.alignment`. The writer first checks that its byte geometry
//! reproduces every offset of `--base` unchanged, so a copied type it cannot size is refused rather
//! than mis-laid.
#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    reason = "writer harness: numeric casts and flat row-major index arithmetic are the payload \
              format"
)]

use apeiron::format::gguf::{Gguf, GgufHeader, TensorDesc};
use apeiron::format::quant::{
    block_geom, f16_to_f32, GGML_TYPE_F16, GGML_TYPE_Q1_0, Q1_0_BLOCK_BYTES, Q1_0_QK,
};
use apeiron::learn::onebit::{quantize, Arm, ONEBIT_GROUP};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// llama.cpp `IQ1_S` (`block_iq1_s`: `d` f16, `qs[32]`, `qh[8]` u16).
const GGML_TYPE_IQ1_S: u32 = 19;
/// llama.cpp `IQ4_NL` (`block_iq4_nl`: `d` f16, `qs[16]`).
const GGML_TYPE_IQ4_NL: u32 = 20;
/// llama.cpp `IQ1_M` (`block_iq1_m`: `qs[32]`, `qh[16]`, `scales[8]`).
const GGML_TYPE_IQ1_M: u32 = 29;

/// Per-tensor output slot: `(ggml type, payload bytes)`, filled by the worker that owns the index.
type Slot = Option<(u32, Vec<u8>)>;

const KINDS: [&str; 7] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// `(weights per block, bytes per block)`: the crate table plus the three llama.cpp types this
/// writer only copies.
fn geom(t: u32) -> Option<(u64, u64)> {
    if let Some((qk, bb)) = block_geom(t) {
        return Some((qk as u64, bb as u64));
    }
    match t {
        GGML_TYPE_IQ1_S => Some((256, 50)),
        GGML_TYPE_IQ4_NL => Some((32, 18)),
        GGML_TYPE_IQ1_M => Some((256, 56)),
        _ => None,
    }
}

fn n_bytes(t: &TensorDesc) -> u64 {
    let n: u64 = t.dims.iter().product();
    let (qk, bb) =
        geom(t.ggml_type).unwrap_or_else(|| panic!("no geometry for ggml type {}", t.ggml_type));
    assert!(
        n.is_multiple_of(qk),
        "{} elements not a whole number of blocks",
        n
    );
    n / qk * bb
}

/// Lay the tensors out on the alignment grid; returns the data size.
fn layout(h: &mut GgufHeader) -> u64 {
    let align = h.alignment;
    let mut cursor = 0u64;
    for t in &mut h.tensors {
        cursor = cursor.div_ceil(align) * align;
        t.offset = cursor;
        cursor += n_bytes(t);
    }
    cursor
}

/// IEEE 754 binary16 bits of `x`, round to nearest even (the encoder of `src/gpu/decode/pack.rs`
/// `f32_to_f16`, which is behind the `gpu` feature).
fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 255 {
        return s | 0x7c00 | u16::from(m != 0);
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return s | 0x7c00;
    }
    let (mant, shift) = if e16 <= 0 {
        if e16 < -10 {
            return s;
        }
        (m | 0x80_0000, (13 + 1 - e16) as u32)
    } else {
        (m, 13u32)
    };
    let base = if e16 <= 0 { 0u32 } else { (e16 as u32) << 10 };
    let kept = mant >> shift;
    let rem = mant & ((1 << shift) - 1);
    let half = 1u32 << (shift - 1);
    let round_up = rem > half || (rem == half && kept & 1 == 1);
    s | ((base | (kept & 0x3ff)) + u32::from(round_up)) as u16
}

/// `Ŵ` (`[out, n]`, every 128-block `±a`) as `Q1_0` blocks: `d = f16(a)`, bit `j` set iff
/// `Ŵ_j ≥ 0` (`dequantize_row_q1_0`: `bit ? d : −d`). Returns `Err` naming the first block that is
/// not `±a`.
fn encode_q1_0(w: &[f32]) -> Result<Vec<u8>, String> {
    assert!(w.len().is_multiple_of(Q1_0_QK));
    let mut out = Vec::with_capacity(w.len() / Q1_0_QK * Q1_0_BLOCK_BYTES);
    for (b, blk) in w.chunks_exact(Q1_0_QK).enumerate() {
        let a = blk[0].abs();
        if let Some(j) = blk.iter().position(|v| v.abs() != a) {
            return Err(format!(
                "block {b} weight {j}: |{}| != a = {a}; Q1_0 cannot hold this block",
                blk[j]
            ));
        }
        out.extend_from_slice(&f32_to_f16(a).to_le_bytes());
        let mut qs = [0u8; Q1_0_QK / 8];
        for (j, v) in blk.iter().enumerate() {
            if *v >= 0.0 {
                qs[j / 8] |= 1 << (j % 8);
            }
        }
        out.extend_from_slice(&qs);
    }
    Ok(out)
}

fn arm_of(name: &str) -> Arm {
    *Arm::ALL
        .iter()
        .find(|a| a.name() == name)
        .unwrap_or_else(|| panic!("unknown arm {name}"))
}

/// `(kind, block)` of `blk.<i>.<kind>.weight`.
fn kind_block(name: &str) -> Option<(&str, usize)> {
    let p: Vec<&str> = name.split('.').collect();
    if p.len() == 4 && p[0] == "blk" && p[3] == "weight" {
        Some((p[2], p[1].parse().ok()?))
    } else {
        None
    }
}

fn gram_file(kind: &str, blk: usize) -> String {
    let g = match kind {
        "attn_q" | "attn_k" | "attn_v" => "attn_in",
        "attn_output" => "attn_out_in",
        "ffn_gate" | "ffn_up" => "ffn_in",
        "ffn_down" => "ffn_down_in",
        _ => unreachable!(),
    };
    format!("blk.{blk}.{g}.f64")
}

fn read_gram(path: &Path, n: usize) -> Vec<f64> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(b.len(), n * n * 8, "Gram {} size", path.display());
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(base_path), Some(sel), Some(out_path)) = (
        arg(&args, "--base"),
        arg(&args, "--sel"),
        arg(&args, "--out"),
    ) else {
        eprintln!(
            "usage: onebit_model --base <gguf> --f16 <gguf> --sel <kinds>[@lo-hi] --out <gguf> \
             (--arm <arm> --gram <dir> --store q1_0|f16 | --take <gguf>) [--threads N]"
        );
        std::process::exit(2);
    };
    let take = arg(&args, "--take");
    let threads: usize = arg(&args, "--threads").map_or(8, |s| s.parse().expect("--threads"));

    let (kinds_s, range) = match sel.split_once('@') {
        Some((k, r)) => {
            let (lo, hi) = r.split_once('-').expect("--sel range is lo-hi");
            (
                k,
                Some((lo.parse::<usize>().unwrap(), hi.parse::<usize>().unwrap())),
            )
        }
        None => (sel.as_str(), None),
    };
    let kinds: Vec<&str> = if kinds_s == "linear" {
        KINDS.to_vec()
    } else {
        kinds_s.split(',').collect()
    };
    for k in &kinds {
        assert!(KINDS.contains(k), "unknown kind {k}");
    }
    let selected = |name: &str| -> bool {
        kind_block(name).is_some_and(|(k, b)| {
            kinds.contains(&k) && range.is_none_or(|(lo, hi)| (lo..=hi).contains(&b))
        })
    };

    let t0 = Instant::now();
    let base = Gguf::open(&base_path).expect("open base");
    let mut header = base.header.clone();
    let orig: Vec<u64> = header.tensors.iter().map(|t| t.offset).collect();
    layout(&mut header);
    let relaid: Vec<u64> = header.tensors.iter().map(|t| t.offset).collect();
    assert_eq!(
        orig, relaid,
        "writer geometry does not reproduce the base layout"
    );

    let names: Vec<String> = header
        .tensors
        .iter()
        .map(|t| String::from_utf8_lossy(&t.name).into_owned())
        .collect();
    let sel_idx: Vec<usize> = (0..names.len()).filter(|&i| selected(&names[i])).collect();
    assert!(!sel_idx.is_empty(), "--sel matches no tensor");

    // Selected payloads, by tensor index.
    let payload: Mutex<Vec<Slot>> = Mutex::new(vec![None; names.len()]);
    if let Some(tp) = &take {
        let src = Gguf::open(tp).expect("open --take");
        for &i in &sel_idx {
            let td = src.header.tensor(&names[i]).expect("--take lacks tensor");
            assert_eq!(td.dims, header.tensors[i].dims, "{} dims", names[i]);
            let n = n_bytes(td);
            let bytes = src.tensor_bytes_prefix(td, n).expect("read --take");
            payload.lock().unwrap()[i] = Some((td.ggml_type, bytes));
        }
        println!("took {} tensors from {tp}", sel_idx.len());
    } else {
        let f16_path = arg(&args, "--f16").expect("--f16");
        let gram_dir = arg(&args, "--gram").expect("--gram");
        let arm = arm_of(&arg(&args, "--arm").expect("--arm"));
        let store = match arg(&args, "--store").as_deref() {
            Some("q1_0") => GGML_TYPE_Q1_0,
            Some("f16") => GGML_TYPE_F16,
            _ => panic!("--store q1_0|f16"),
        };
        let f16 = Gguf::open(&f16_path).expect("open f16");
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&i) = sel_idx.get(k) else { break };
                    let name = &names[i];
                    let tj = Instant::now();
                    let fd = f16.header.tensor(name).expect("f16 tensor");
                    assert_eq!(fd.ggml_type, GGML_TYPE_F16, "{name} is not F16 in --f16");
                    let n_in = fd.dims[0] as usize;
                    let n_out = (fd.dims.iter().product::<u64>() as usize) / n_in;
                    let w: Vec<f32> = f16
                        .tensor_bytes(fd)
                        .expect("read f16")
                        .chunks_exact(2)
                        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect();
                    let (kind, blk) = kind_block(name).unwrap();
                    let gram = read_gram(&Path::new(&gram_dir).join(gram_file(kind, blk)), n_in);
                    let q = quantize(arm, &w, n_out, n_in, &gram, ONEBIT_GROUP)
                        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
                    let bytes = if store == GGML_TYPE_Q1_0 {
                        encode_q1_0(&q.what).unwrap_or_else(|e| panic!("{name}: {e}"))
                    } else {
                        q.what
                            .iter()
                            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                            .collect()
                    };
                    println!(
                        "{name:<26} [{n_in}, {n_out}] {} -> {} bytes in {:.1} s",
                        arm.name(),
                        bytes.len(),
                        tj.elapsed().as_secs_f64()
                    );
                    payload.lock().unwrap()[i] = Some((store, bytes));
                });
            }
        });
    }

    let mut payload = payload.into_inner().unwrap();
    for &i in &sel_idx {
        header.tensors[i].ggml_type = payload[i].as_ref().expect("payload").0;
    }
    let total = layout(&mut header) as usize;
    let mut data = vec![0u8; total];
    let (mut sel_bytes, mut sel_elems) = (0u64, 0u64);
    for (i, t) in header.tensors.iter().enumerate() {
        let start = t.offset as usize;
        let bytes = match payload[i].take() {
            Some((_, b)) => {
                sel_bytes += b.len() as u64;
                sel_elems += t.dims.iter().product::<u64>();
                b
            }
            None => {
                let bt = &base.header.tensors[i];
                base.tensor_bytes_prefix(bt, n_bytes(bt))
                    .expect("read base")
            }
        };
        assert_eq!(bytes.len() as u64, n_bytes(t), "{} payload size", names[i]);
        data[start..start + bytes.len()].copy_from_slice(&bytes);
    }
    let file = apeiron::learn::gguf::write_file(&header, &data).expect("write gguf");
    std::fs::write(&out_path, &file).expect("write output");
    println!(
        "selected {} tensors, {} weights, {:.4} stored b/w; wrote {out_path}: {} bytes in {:.1} s",
        sel_idx.len(),
        sel_elems,
        sel_bytes as f64 * 8.0 / sel_elems as f64,
        file.len(),
        t0.elapsed().as_secs_f64()
    );
}
