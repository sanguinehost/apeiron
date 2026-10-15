//! Q2 — normal-mode freeze-out: rate allocation on the joint cross-layer Fisher.
//!
//! Rate model over the seven linears of the 24 blocks of Qwen2.5-0.5B (f16 teacher). Plane 1 is
//! `HadRtn(W)` everywhere (`Q1 = quantize(Arm::HadRtn, …)`, 1.125 b/w). Plane 2 is `HadRtn(W − Q1)` added
//! back on the selected row-blocks: the student on those rows is `Q1 + Q2(W − Q1)`, its error
//! `E2 = Q1 + Q2(W − Q1) − W`. The allocation unit is a row-block of 16 consecutive rows of one tensor.
//! The budget is `1.178 b/w` over the linears: selected parameters ≤ `f·P`, `f = (1.178 − 1.125)/1.125`.
//!
//! Fisher sketch: K calibration windows (disjoint from the held-out windows by byte offset). Per window
//! the teacher forward (`apeiron::learn::grad::model::forward`, Q32.32), a fixed-seed draw
//! `y_t ~ p_t` (splitmix64), `dlogits_t = onehot(y_t) − p_t` in Q32.32, and the backward. Per row-block
//! `c_{k,b} = ⟨g_k[b], E1[b]⟩ / 2^32` and `d_{k,b} = ⟨g_k[b], E2[b]⟩ / 2^32` (f64), stored per sample.
//!
//! Arms (same rate): `uniform` (plane 1 only), `separable` (rank by `mean_k(c² − d²)`, top to budget),
//! `joint` (greedy on `mean_k s_k²`, `s_k = Σ_b (sel ? d : c)_{k,b}`, one block per step).
//!
//! Modes (each measured run is one invocation under the shared lock, `< 600 s`):
//! - `--time`: one sample, one student build, one window: projects the wall of the other runs.
//! - `--fisher <k0> <k1>`: sketch samples `[k0, k1)`; writes `sk/k<k>.bin` and `plan.tsv`.
//! - `--sketch`: reads all samples; Gram eigenvalues, off-diagonal fraction, the three selections,
//!   controls (a) and (b). No model.
//! - `--arm <uniform|separable|joint>`: builds the student (reflink copy of the teacher, f16 rows
//!   overwritten), held-out KL on three windows; `uniform` also runs control (c).
//! - `--merge`: folds the JSON parts into `q2-receipt.json`.
//!
//! Executing language: everything here is Rust on the crate's build path (no Python).
//!
//! Trust boundary: the claim is about the crate's integer decode and backward paths and the in-tree
//! quantiser, under `APEIRON_F32_CARRIER=0`; it does not reach rustc/LLVM codegen or the `unsafe` container.

#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::format_push_string,
    reason = "measurement harness: numeric casts between f16/f32/f64 and the Q32.32 integer \
              logits, indexed loops over flat buffers, and long report builders are the \
              instrument, not a library boundary"
)]

use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::io::{Seek, SeekFrom, Write as IoWrite};
use std::process::Command;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::format::bpe::Bpe;
use apeiron::format::quant::{self, f16_to_f32};
use apeiron::learn::grad::model::{backward, forward, Config, Grads, Weights};
use apeiron::learn::onebit::{quantize, Arm};

const MODEL: &str = concat!(env!("HOME"), "/models/gguf/qwen05b-f16.gguf");
const TOKENIZER: &str = concat!(env!("HOME"), "/models/qwen2.5-0.5b/tokenizer.json");
const CORPUS: &str = concat!(env!("HOME"), "/models/gpu-ppl-corpus.txt");
const SCRATCH: &str = concat!(env!("HOME"), "/oc-tmp/lanes/qarch/s-q2");
const SOURCE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/qarch_q2.rs");
const ONEBIT_SRC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/learn/onebit.rs");

/// Tokens per window.
const T: usize = 64;
/// Held-out windows (the Q1 offsets); the byte spans `[off, off + SPAN)` are disjoint from `CAL`.
const HELD: [usize; 3] = [0, 20_000, 44_000];
/// Calibration window byte offsets: 4000..=16000 step 1000 and 24000, 28000, 32000. Every span
/// `[off, off + 4000)` lies in `[4000, 20000)` or `[24000, 36000)`, disjoint from the held-out spans.
/// The 65,536-byte corpus has no room for 16 disjoint spans, so calibration windows overlap each other.
const CAL: [usize; 16] = [
    4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000, 11_000, 12_000, 13_000, 14_000, 15_000,
    16_000, 24_000, 28_000, 32_000,
];
const SPAN: usize = 4_000;
/// Row-block height (rows per allocation unit).
const ROWS: usize = 16;
const PLANE1_BPW: f64 = 1.125;
const TARGET_BPW: f64 = 1.178;
const ONE_F: f64 = 4_294_967_296.0;

const LINEARS: [&str; 7] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];

// ── plan: the linears and their row-blocks ──────────────────────────────────────────────────────

/// One linear weight `[out, inp]` (row-major) of one block.
struct Lin {
    layer: usize,
    suffix: &'static str,
    name: String,
    out: usize,
    inp: usize,
    off: u64,
    blk0: usize,
    nblk: usize,
}

/// The whole allocation: linears and their row-blocks, global block index `b`.
struct Plan {
    lins: Vec<Lin>,
    blk_lin: Vec<usize>,
    bparams: Vec<usize>,
}

fn plan(eng: &Engine) -> Result<Plan, String> {
    let mut lins = Vec::new();
    let mut blk_lin = Vec::new();
    let mut bparams = Vec::new();
    for l in 0..eng.config.n_layer {
        for suffix in LINEARS {
            let name = format!("blk.{l}.{suffix}.weight");
            let t = eng
                .table()
                .get(&name)
                .ok_or_else(|| format!("missing tensor {name}"))?;
            if t.ggml_type != quant::GGML_TYPE_F16 || t.dims.len() != 2 {
                return Err(format!("{name}: not a rank-2 F16 tensor"));
            }
            let inp = usize::try_from(t.dims[0]).map_err(|_| format!("{name}: dims"))?;
            let out = usize::try_from(t.dims[1]).map_err(|_| format!("{name}: dims"))?;
            if inp % 128 != 0 || out % ROWS != 0 {
                return Err(format!("{name}: [{out}, {inp}] not tileable"));
            }
            if t.n_bytes != Some((out * inp * 2) as u64) {
                return Err(format!("{name}: stored bytes {:?}", t.n_bytes));
            }
            let li = lins.len();
            let nblk = out / ROWS;
            lins.push(Lin {
                layer: l,
                suffix,
                name,
                out,
                inp,
                off: t.byte_range.start,
                blk0: blk_lin.len(),
                nblk,
            });
            for _ in 0..nblk {
                blk_lin.push(li);
                bparams.push(ROWS * inp);
            }
        }
    }
    Ok(Plan {
        lins,
        blk_lin,
        bparams,
    })
}

fn total_params(p: &Plan) -> usize {
    p.bparams.iter().sum()
}

/// Round-to-nearest-even `f32` → `f16` bits (the inverse of [`f16_to_f32`] on the f16 grid).
fn f32_to_f16_bits(x: f32) -> u16 {
    let xi = x.to_bits();
    let sign = ((xi >> 16) & 0x8000) as u16;
    let exp = ((xi >> 23) & 0xff) as i32;
    let man = xi & 0x007f_ffff;
    if exp == 0xff {
        let m = if man == 0 { 0u16 } else { 0x0200u16 };
        return sign | 0x7c00 | m;
    }
    if exp == 0 {
        // f32 subnormal or zero is far below the f16 grid: rounds to a signed zero.
        return sign;
    }
    let mut e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let man24 = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut m = man24 >> shift;
        let rem = man24 & ((1u32 << shift) - 1);
        if rem > half || (rem == half && (m & 1) == 1) {
            m += 1;
        }
        return sign | m as u16;
    }
    let mut m = man >> 13;
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
        m += 1;
        if m == 0x400 {
            m = 0;
            e += 1;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | m as u16
}
/// The `n × n` identity Gram the `HadRtn` arm is handed (it uses no Gram feedback).
fn identity_gram(n: usize) -> Vec<f64> {
    let mut g = vec![0f64; n * n];
    for i in 0..n {
        g[i * n + i] = 1.0;
    }
    g
}
/// The Q32.32 logits of every position of `ids`, as `f64` (a `>> 32`).
fn forward_logits(eng: &Engine, ids: &[u32]) -> Result<Vec<Vec<f64>>, String> {
    let mut kv = eng
        .decode_state(ids.len())
        .map_err(|e| format!("decode_state: {e:?}"))?;
    let mut out = Vec::with_capacity(ids.len());
    for &id in ids {
        let z = eng
            .step_logits(&mut kv, id)
            .map_err(|e| format!("step_logits: {e:?}"))?;
        out.push(z.iter().map(|&v| v as f64 / ONE_F).collect());
    }
    Ok(out)
}
fn cmd_out(prog: &str, args: &[&str]) -> String {
    Command::new(prog)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "-".into())
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        let _ = write!(s, "{x:02x}");
    }
    s
}

// ── windows and per-tensor planes ───────────────────────────────────────────────────────────────

/// `T` token ids of the `SPAN`-byte window at each byte offset.
fn tokens_for(offs: &[usize]) -> Result<Vec<Vec<u32>>, String> {
    let bpe = Bpe::from_tokenizer_json(TOKENIZER).map_err(|e| format!("tokenizer: {e:?}"))?;
    let raw = std::fs::read(CORPUS).map_err(|e| format!("{CORPUS}: {e}"))?;
    let mut out = Vec::with_capacity(offs.len());
    for &off in offs {
        if off + SPAN > raw.len() {
            return Err(format!("window offset {off} past {CORPUS}"));
        }
        let text = String::from_utf8_lossy(&raw[off..off + SPAN]).into_owned();
        let ids = bpe
            .encode(&text)
            .map_err(|e| format!("encode@{off}: {e:?}"))?;
        if ids.len() < T {
            return Err(format!("window@{off}: {} tokens < {T}", ids.len()));
        }
        out.push(ids[..T].to_vec());
    }
    Ok(out)
}

/// One linear's `(W, Q1, Q2)` as f32, row-major `[out, inp]`: `W` from the teacher's f16 bytes,
/// `Q1 = HadRtn(W)`, `Q2 = HadRtn(W − Q1)`. Both planes are the in-tree `quantize`.
type Quant = (Vec<f32>, Vec<f32>, Vec<f32>);

fn quant_lin(eng: &Engine, lin: &Lin) -> Result<Quant, String> {
    let cnt = lin.out * lin.inp;
    let raw = eng
        .payload(lin.off, (cnt * 2) as u64)
        .map_err(|e| format!("{}: {e:?}", lin.name))?;
    let w: Vec<f32> = (0..cnt)
        .map(|i| f16_to_f32(u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]])))
        .collect();
    let gram = identity_gram(lin.inp);
    let q1 = quantize(Arm::HadRtn, &w, lin.out, lin.inp, &gram, 128)
        .map_err(|e| format!("{}: quantize {e:?}", lin.name))?
        .what;
    let d: Vec<f32> = (0..cnt)
        .map(|i| (f64::from(w[i]) - f64::from(q1[i])) as f32)
        .collect();
    let q2 = quantize(Arm::HadRtn, &d, lin.out, lin.inp, &gram, 128)
        .map_err(|e| format!("{}: quantize residual {e:?}", lin.name))?
        .what;
    Ok((w, q1, q2))
}

fn grad_of<'a>(g: &'a Grads, layer: usize, suffix: &str) -> &'a [i64] {
    let l = &g.layers[layer];
    match suffix {
        "attn_q" => &l.wq[..],
        "attn_k" => &l.wk[..],
        "attn_v" => &l.wv[..],
        "attn_output" => &l.wo[..],
        "ffn_gate" => &l.w_gate[..],
        "ffn_up" => &l.w_up[..],
        _ => &l.w_down[..],
    }
}

// ── Fisher sketch ───────────────────────────────────────────────────────────────────────────────

fn next_u01(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Per row-block `(c, d)` of calibration window `k`: one score sample `g = backward(dlogits)` with
/// `dlogits_t = onehot(y_t) − p_t` (Q32.32) for a fixed-seed draw `y_t ~ p_t`, and
/// `c_b = ⟨g[b], E1[b]⟩ / 2^32`, `d_b = ⟨g[b], E2[b]⟩ / 2^32`, `E1 = Q1 − W`, `E2 = Q1 + Q2 − W`.
///
/// # Complexity
/// Forward and backward `O(T·n_embd·(n_embd + n_ff) + vocab·T·n_embd)` plus the per-tensor
/// quantisation `O(out·inp·log 128)` for each of the 168 linears; the scoring loop is `O(Σ out·inp)`.
fn fisher_sample(
    eng: &Engine,
    w: &Weights,
    cfg: &Config,
    p: &Plan,
    k: usize,
    toks: &[u32],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let (logits, tape) = forward(w, cfg, toks).map_err(|e| format!("forward: {e:?}"))?;
    let v = cfg.vocab;
    let t = toks.len();
    let mut dl = vec![0i64; t * v];
    let mut state = 0x5132_0000_0000_0000u64 ^ (k as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
    let mut pr = vec![0f64; v];
    for ti in 0..t {
        let z = &logits[ti * v..(ti + 1) * v];
        let m = z
            .iter()
            .map(|&x| x as f64 / ONE_F)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut s = 0f64;
        for i in 0..v {
            pr[i] = (z[i] as f64 / ONE_F - m).exp();
            s += pr[i];
        }
        for x in pr.iter_mut() {
            *x /= s;
        }
        let u = next_u01(&mut state);
        let mut acc = 0f64;
        let mut y = v - 1;
        for i in 0..v {
            acc += pr[i];
            if u < acc {
                y = i;
                break;
            }
        }
        for i in 0..v {
            let onehot = if i == y { 1.0 } else { 0.0 };
            dl[ti * v + i] = ((onehot - pr[i]) * ONE_F).round() as i64;
        }
    }
    let grads = backward(w, cfg, &tape, &dl).map_err(|e| format!("backward: {e:?}"))?;
    let nb = p.bparams.len();
    let mut c = vec![0f64; nb];
    let mut d = vec![0f64; nb];
    for lin in &p.lins {
        let (wv, q1, q2) = quant_lin(eng, lin)?;
        let g = grad_of(&grads, lin.layer, lin.suffix);
        for bi in 0..lin.nblk {
            let (mut cs, mut ds) = (0f64, 0f64);
            for i in bi * ROWS * lin.inp..(bi + 1) * ROWS * lin.inp {
                let gi = g[i] as f64;
                let e1 = f64::from(q1[i]) - f64::from(wv[i]);
                let e2 = e1 + f64::from(q2[i]);
                cs += gi * e1;
                ds += gi * e2;
            }
            c[lin.blk0 + bi] = cs / ONE_F;
            d[lin.blk0 + bi] = ds / ONE_F;
        }
    }
    Ok((c, d))
}

fn sample_path(k: usize) -> String {
    format!("{SCRATCH}/sk/k{k}.bin")
}

fn write_sample(k: usize, c: &[f64], d: &[f64]) -> Result<(), Box<dyn Error>> {
    let mut bytes = Vec::with_capacity((c.len() + d.len()) * 8);
    for x in c.iter().chain(d) {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(sample_path(k), bytes)?;
    Ok(())
}

fn read_sample(k: usize, nb: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let bytes = std::fs::read(sample_path(k)).map_err(|e| format!("sample {k}: {e}"))?;
    if bytes.len() != 16 * nb {
        return Err(format!("sample {k}: {} bytes != {}", bytes.len(), 16 * nb));
    }
    let v: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect();
    Ok((v[..nb].to_vec(), v[nb..].to_vec()))
}

/// sha256 of the concatenated sample files `0..kk`.
fn sketch_sha(kk: usize) -> Result<String, String> {
    let mut all = Vec::new();
    for k in 0..kk {
        all.extend(std::fs::read(sample_path(k)).map_err(|e| format!("sample {k}: {e}"))?);
    }
    Ok(hex(&apeiron::sha256::sha256(&all)))
}

fn write_plan(p: &Plan) -> Result<(), Box<dyn Error>> {
    let mut s = String::from("b\tlayer\tsuffix\tr0\tparams\n");
    for (b, &li) in p.blk_lin.iter().enumerate() {
        let lin = &p.lins[li];
        let r0 = (b - lin.blk0) * ROWS;
        let _ = writeln!(
            s,
            "{b}\t{}\t{}\t{r0}\t{}",
            lin.layer, lin.suffix, p.bparams[b]
        );
    }
    std::fs::write(format!("{SCRATCH}/plan.tsv"), s)?;
    Ok(())
}

fn read_plan_params() -> Result<Vec<usize>, String> {
    let text = std::fs::read_to_string(format!("{SCRATCH}/plan.tsv"))
        .map_err(|e| format!("plan.tsv: {e}"))?;
    text.lines()
        .skip(1)
        .map(|l| {
            l.split('\t')
                .nth(4)
                .ok_or_else(|| format!("plan row: {l}"))?
                .parse::<usize>()
                .map_err(|e| format!("plan row {l}: {e}"))
        })
        .collect()
}

// ── selections ──────────────────────────────────────────────────────────────────────────────────

/// Separable: rank every block by `gain` (descending, index ascending on ties) and take each that fits:
/// the equal-rate protocol spends the whole budget.
fn select_separable(gain: &[f64], bp: &[usize], budget: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..gain.len()).collect();
    order.sort_by(|&a, &b| gain[b].partial_cmp(&gain[a]).unwrap().then(a.cmp(&b)));
    let mut used = 0usize;
    let mut out = Vec::new();
    for b in order {
        if used + bp[b] <= budget {
            used += bp[b];
            out.push(b);
        }
    }
    out.sort_unstable();
    out
}

/// Joint greedy, one block per step: the fitting block with the most negative change of
/// `mean_k s_k²`, `s_k` the sum of `c` over unselected and `d` over selected blocks. With
/// `diag = Some(gain)` the change is the diagonal `−gain_b` and `s` is not tracked (control (a)).
///
/// # Complexity
/// `O(S · nb · K)` for `S` selected blocks, `nb` blocks, `K` samples: each step scans every candidate
/// and its `K`-term change; the update adds `O(K)`.
fn select_joint(
    c: &[Vec<f64>],
    d: &[Vec<f64>],
    bp: &[usize],
    budget: usize,
    diag: Option<&[f64]>,
    stop: bool,
) -> Vec<usize> {
    let kk = c.len();
    let nb = bp.len();
    let mut s = vec![0f64; kk];
    if diag.is_none() {
        for k in 0..kk {
            s[k] = c[k].iter().sum();
        }
    }
    let mut chosen = vec![false; nb];
    let mut used = 0usize;
    let mut out = Vec::new();
    loop {
        let mut best: Option<usize> = None;
        let mut best_change = f64::INFINITY;
        for b in 0..nb {
            if chosen[b] || used + bp[b] > budget {
                continue;
            }
            let change = match diag {
                Some(g) => -g[b],
                None => {
                    let mut acc = 0f64;
                    for k in 0..kk {
                        let delta = d[k][b] - c[k][b];
                        acc += (s[k] + delta) * (s[k] + delta) - s[k] * s[k];
                    }
                    acc / kk as f64
                }
            };
            if change < best_change {
                best_change = change;
                best = Some(b);
            }
        }
        let Some(b) = best else { break };
        if stop && best_change >= 0.0 {
            break;
        }
        chosen[b] = true;
        used += bp[b];
        out.push(b);
        if diag.is_none() {
            for k in 0..kk {
                s[k] += d[k][b] - c[k][b];
            }
        }
    }
    out.sort_unstable();
    out
}

/// Cyclic Jacobi eigenvalues of a symmetric matrix.
fn jacobi_eigen(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    for _ in 0..100 {
        let mut off = 0f64;
        for p in 0..n {
            for q in 0..n {
                if p != q {
                    off += a[p][q] * a[p][q];
                }
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    (0..n).map(|i| a[i][i]).collect()
}

fn read_sel(name: &str) -> Result<Vec<usize>, String> {
    let text = std::fs::read_to_string(format!("{SCRATCH}/sel-{name}.txt"))
        .map_err(|e| format!("sel-{name}.txt: {e}"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<usize>().map_err(|e| format!("sel: {e}")))
        .collect()
}

// ── student and KL ──────────────────────────────────────────────────────────────────────────────

/// A reflink copy of the teacher with every linear overwritten: plane 1 on every row, plane 2 on the
/// selected row-blocks (`Q1 + Q2`), as f16 bytes at the teacher's file offsets.
fn build_student(eng: &Engine, p: &Plan, sel: &[bool], path: &str) -> Result<(), String> {
    let _ = std::fs::remove_file(path);
    let cp = Command::new("cp")
        .arg("--reflink=always")
        .arg(MODEL)
        .arg(path)
        .status();
    match cp {
        Ok(s) if s.success() => {}
        _ => {
            std::fs::copy(MODEL, path).map_err(|e| format!("copy {path}: {e}"))?;
        }
    }
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|e| format!("open {path}: {e}"))?;
    for lin in &p.lins {
        let (_w, q1, q2) = quant_lin(eng, lin)?;
        let mut bytes = Vec::with_capacity(q1.len() * 2);
        for r in 0..lin.out {
            let on = sel[lin.blk0 + r / ROWS];
            for i in r * lin.inp..(r + 1) * lin.inp {
                let v = if on { q1[i] + q2[i] } else { q1[i] };
                bytes.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
        }
        f.seek(SeekFrom::Start(lin.off))
            .map_err(|e| format!("seek {path}: {e}"))?;
        f.write_all(&bytes)
            .map_err(|e| format!("write {path}: {e}"))?;
    }
    Ok(())
}

/// `mean_t KL(p_t ‖ q_t)` from the two logit sets (f64, log-sum-exp per position).
fn kl_mean(zp: &[Vec<f64>], zq: &[Vec<f64>]) -> f64 {
    let mut total = 0f64;
    for (a, b) in zp.iter().zip(zq) {
        let ma = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mb = b.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sa: f64 = a.iter().map(|&x| (x - ma).exp()).sum();
        let sb: f64 = b.iter().map(|&x| (x - mb).exp()).sum();
        let (la, lb) = (ma + sa.ln(), mb + sb.ln());
        let mut acc = 0f64;
        for (&x, &y) in a.iter().zip(b) {
            acc += (x - la).exp() * ((x - la) - (y - lb));
        }
        total += acc;
    }
    total / zp.len() as f64
}

fn json_f(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.6e}")
    } else {
        "null".into()
    }
}

fn json_list(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|&x| json_f(x)).collect();
    format!("[{}]", parts.join(","))
}

/// The three steps of one invocation: wall, in seconds.
fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

// ── modes ───────────────────────────────────────────────────────────────────────────────────────

/// One Fisher pass (one sample), one student build, one held-out window: projects the other walls.
fn cmd_time() -> Result<(), Box<dyn Error>> {
    let eng = Engine::open(MODEL).map_err(|e| format!("open: {e:?}"))?;
    let p = plan(&eng)?;
    let nb = p.bparams.len();
    let t0 = Instant::now();
    let (w, cfg) = Weights::from_engine(&eng).map_err(|e| format!("weights: {e:?}"))?;
    let t_w = secs(t0);
    let toks = tokens_for(&[CAL[0]])?;
    let t1 = Instant::now();
    let (c, d) = fisher_sample(&eng, &w, &cfg, &p, 0, &toks[0])?;
    let t_s = secs(t1);
    println!(
        "time: weights {t_w:.1}s; one Fisher sample {t_s:.1}s (blocks {nb}, sum c {:.3e}, sum d {:.3e})",
        c.iter().sum::<f64>(),
        d.iter().sum::<f64>()
    );
    drop(w);
    let sel = vec![false; nb];
    let t2 = Instant::now();
    let spath = format!("{SCRATCH}/student-time.gguf");
    build_student(&eng, &p, &sel, &spath)?;
    let t_b = secs(t2);
    let student = Engine::open(&spath).map_err(|e| format!("open student: {e:?}"))?;
    let held = tokens_for(&HELD[..1])?;
    let t3 = Instant::now();
    let zq = forward_logits(&student, &held[0])?;
    let t_f = secs(t3);
    println!(
        "time: student build {t_b:.1}s; one window ({T} steps) {t_f:.1}s; logits {}x{}",
        zq.len(),
        zq[0].len()
    );
    let per_arm = t_b + 6.0 * t_f;
    println!(
        "projection: Fisher {} samples {:.0}s + weights {t_w:.0}s; one arm {per_arm:.0}s (build + 3 windows x teacher and student)",
        CAL.len(),
        CAL.len() as f64 * t_s,
    );
    Ok(())
}

fn cmd_fisher(k0: usize, k1: usize) -> Result<(), Box<dyn Error>> {
    if k1 > CAL.len() || k0 >= k1 {
        return Err(format!("range [{k0}, {k1}) outside 0..{}", CAL.len()).into());
    }
    let eng = Engine::open(MODEL).map_err(|e| format!("open: {e:?}"))?;
    let p = plan(&eng)?;
    write_plan(&p)?;
    std::fs::create_dir_all(format!("{SCRATCH}/sk"))?;
    let t0 = Instant::now();
    let (w, cfg) = Weights::from_engine(&eng).map_err(|e| format!("weights: {e:?}"))?;
    println!(
        "weights {:.1}s; blocks {}; linear params {}",
        secs(t0),
        p.bparams.len(),
        total_params(&p)
    );
    let toks = tokens_for(&CAL)?;
    for k in k0..k1 {
        let ts = Instant::now();
        let (c, d) = fisher_sample(&eng, &w, &cfg, &p, k, &toks[k])?;
        write_sample(k, &c, &d)?;
        println!("sample {k} (window @{}) {:.1}s", CAL[k], secs(ts));
    }
    Ok(())
}

fn cmd_sketch(kk: usize) -> Result<(), Box<dyn Error>> {
    let bp = read_plan_params()?;
    let nb = bp.len();
    let ptot: usize = bp.iter().sum();
    let f = (TARGET_BPW - PLANE1_BPW) / PLANE1_BPW;
    let budget = (f * ptot as f64).floor() as usize;
    let mut cs = Vec::with_capacity(kk);
    let mut ds = Vec::with_capacity(kk);
    for k in 0..kk {
        let (c, d) = read_sample(k, nb)?;
        cs.push(c);
        ds.push(d);
    }
    let sha = sketch_sha(kk)?;

    // Gram of the plane-1 scores, top eigenvalues, and the off-diagonal share of mean_k s_k².
    let mut gram = vec![vec![0f64; kk]; kk];
    for a in 0..kk {
        for b in 0..kk {
            gram[a][b] = cs[a].iter().zip(&cs[b]).map(|(x, y)| x * y).sum::<f64>() / nb as f64;
        }
    }
    let trace: f64 = (0..kk).map(|a| gram[a][a]).sum();
    let mut eig = jacobi_eigen(gram);
    eig.sort_by(|x, y| y.partial_cmp(x).unwrap());
    let eig_sum: f64 = eig.iter().sum();
    let eig_trace_err = (eig_sum - trace).abs() / trace.abs().max(1e-300);
    let s: Vec<f64> = cs.iter().map(|c| c.iter().sum()).collect();
    let total = s.iter().map(|x| x * x).sum::<f64>() / kk as f64;
    let diag: f64 = (0..nb)
        .map(|b| cs.iter().map(|c| c[b] * c[b]).sum::<f64>() / kk as f64)
        .sum();
    let off_frac = (total - diag) / total;

    // Gains and the three selections.
    let gain_of = |c: &[Vec<f64>], d: &[Vec<f64>]| -> Vec<f64> {
        (0..nb)
            .map(|b| {
                (0..kk)
                    .map(|k| c[k][b] * c[k][b] - d[k][b] * d[k][b])
                    .sum::<f64>()
                    / kk as f64
            })
            .collect()
    };
    let gain = gain_of(&cs, &ds);
    let sep = select_separable(&gain, &bp, budget);
    let jnt = select_joint(&cs, &ds, &bp, budget, None, false);
    let jdiag = select_joint(&cs, &ds, &bp, budget, Some(&gain), false);
    let jstop = select_joint(&cs, &ds, &bp, budget, None, true);
    println!(
        "joint with the no-gain stop rule (informational): blocks {} params {} b/w {:.6}",
        jstop.len(),
        jstop.iter().map(|&b| bp[b]).sum::<usize>(),
        PLANE1_BPW * (1.0 + jstop.iter().map(|&b| bp[b]).sum::<usize>() as f64 / ptot as f64)
    );

    // Control (a): the joint greedy on the diagonal objective selects exactly the separable set.
    let ctrl_a = jdiag == sep;
    // Control (b): E2 := E1 (d = c) gives every gain 0 and no selection.
    let gain0 = gain_of(&cs, &cs);
    let ctrl_b_gain = gain0.iter().all(|&g| g == 0.0);
    let ctrl_b_sel = select_joint(&cs, &cs, &bp, budget, None, true);
    let ctrl_b = ctrl_b_gain && ctrl_b_sel.is_empty();

    let pars = |sel: &[usize]| -> usize { sel.iter().map(|&b| bp[b]).sum() };
    let overlap = sep.iter().filter(|b| jnt.binary_search(b).is_ok()).count();
    for (name, sel) in [
        ("uniform", Vec::new()),
        ("separable", sep.clone()),
        ("joint", jnt.clone()),
    ] {
        let mut txt = String::new();
        for b in &sel {
            let _ = writeln!(txt, "{b}");
        }
        std::fs::write(format!("{SCRATCH}/sel-{name}.txt"), txt)?;
        println!(
            "arm {name:9} blocks {:6} params {:10} ({:.4} of linear) b/w {:.6}",
            sel.len(),
            pars(&sel),
            pars(&sel) as f64 / ptot as f64,
            PLANE1_BPW * (1.0 + pars(&sel) as f64 / ptot as f64)
        );
    }
    println!("budget params {budget} of {ptot} (f = {f:.6}); separable/joint overlap {overlap}");
    println!("gram eigen (top 8): {:?}", &eig[..8.min(eig.len())]);
    println!("gram trace {trace:.6e}, eig sum {eig_sum:.6e}, rel err {eig_trace_err:.2e}");
    println!("off-diagonal share of mean_k s_k^2 (plane 1): {off_frac:.6}");
    println!(
        "control (a) joint-diagonal == separable: {ctrl_a} ({} blocks)",
        sep.len()
    );
    println!(
        "control (b) E2:=E1 gains all zero: {ctrl_b_gain}; joint selects none: {}",
        ctrl_b_sel.is_empty()
    );
    let ok = ctrl_a && ctrl_b;
    let receipt = format!(
        "{{\"samples\":{kk},\"sketch_sha256\":\"{sha}\",\"blocks\":{nb},\"total_params\":{ptot},\"budget_params\":{budget},\"f\":{},\"gram_eig_top8\":{},\"gram_trace\":{},\"gram_eig_rel_err\":{},\"offdiag_share\":{},\"overlap_separable_joint\":{overlap},\"control_a\":{ctrl_a},\"control_b\":{ctrl_b},\"control_b_joint_sel\":{}}}",
        json_f(f),
        json_list(&eig[..8.min(eig.len())]),
        json_f(trace),
        json_f(eig_trace_err),
        json_f(off_frac),
        ctrl_b_sel.len()
    );
    std::fs::write(format!("{SCRATCH}/q2-sketch.json"), receipt)?;
    if !ok {
        return Err("control (a) or (b) failed".into());
    }
    Ok(())
}

fn cmd_arm(name: &str) -> Result<(), Box<dyn Error>> {
    let eng = Engine::open(MODEL).map_err(|e| format!("open: {e:?}"))?;
    let p = plan(&eng)?;
    let nb = p.bparams.len();
    let sel_list = read_sel(name)?;
    let mut sel = vec![false; nb];
    for &b in &sel_list {
        sel[b] = true;
    }
    let ptot = total_params(&p);
    let psel: usize = sel_list.iter().map(|&b| p.bparams[b]).sum();
    let bpw = PLANE1_BPW * (1.0 + psel as f64 / ptot as f64);
    let sketch = sketch_sha(CAL.len())?;

    let spath = format!("{SCRATCH}/student-{name}.gguf");
    let t0 = Instant::now();
    build_student(&eng, &p, &sel, &spath)?;
    let t_build = secs(t0);
    let student = Engine::open(&spath).map_err(|e| format!("open student: {e:?}"))?;
    let toks = tokens_for(&HELD)?;
    let mut kls = Vec::new();
    let mut walls = Vec::new();
    for (wi, tk) in toks.iter().enumerate() {
        let t1 = Instant::now();
        let zp = forward_logits(&eng, tk)?;
        let zq = forward_logits(&student, tk)?;
        let kl = kl_mean(&zp, &zq);
        walls.push(secs(t1));
        println!(
            "arm {name} window {wi} (@{}): KL {kl:.6e} ({:.1}s)",
            HELD[wi], walls[wi]
        );
        kls.push(kl);
    }
    drop(student);

    // Control (c), uniform arm only: the teacher as its own student must give KL exactly 0.
    let ctrl_c = if name == "uniform" {
        let twin = Engine::open(MODEL).map_err(|e| format!("open twin: {e:?}"))?;
        let zp = forward_logits(&eng, &toks[0])?;
        let zq = forward_logits(&twin, &toks[0])?;
        let kl = kl_mean(&zp, &zq);
        println!("control (c) teacher as student: KL {kl:.6e}");
        if kl != 0.0 {
            return Err(format!("control (c) failed: KL {kl:e} != 0").into());
        }
        Some(kl)
    } else {
        None
    };

    let (lo, hi) = kls
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    println!("arm {name}: b/w {bpw:.6}, KL mean {mean:.6e} (min {lo:.6e}, max {hi:.6e})");
    let json = format!(
        "{{\"arm\":\"{name}\",\"blocks\":{},\"params\":{psel},\"total_params\":{ptot},\"bpw\":{},\"kl\":{},\"kl_mean\":{},\"kl_min\":{},\"kl_max\":{},\"control_c\":{},\"walls_s\":{},\"build_s\":{},\"teacher_sha256\":\"{}\",\"student_sha256\":\"{}\",\"sketch_sha256\":\"{sketch}\"}}",
        sel_list.len(),
        json_f(bpw),
        json_list(&kls),
        json_f(mean),
        json_f(lo),
        json_f(hi),
        ctrl_c.map_or("null".to_string(), json_f),
        json_list(&walls),
        json_f(t_build),
        cmd_out("sha256sum", &[MODEL]).split_whitespace().next().unwrap_or("-"),
        cmd_out("sha256sum", &[spath.as_str()]).split_whitespace().next().unwrap_or("-"),
    );
    std::fs::write(format!("{SCRATCH}/q2-arm-{name}.json"), json)?;
    Ok(())
}

fn cmd_merge() -> Result<(), Box<dyn Error>> {
    let read = |f: &str| -> Result<String, Box<dyn Error>> {
        Ok(std::fs::read_to_string(format!("{SCRATCH}/{f}"))?)
    };
    let sketch = read("q2-sketch.json")?;
    let arms = ["uniform", "separable", "joint"]
        .iter()
        .map(|a| read(&format!("q2-arm-{a}.json")).map(|t| format!("\"{a}\":{t}")))
        .collect::<Result<Vec<_>, _>>()?;
    let receipt =
        format!(
        "{{\"source_sha256\":\"{}\",\"onebit_sha256\":\"{}\",\"sketch\":{sketch},\"arms\":{{{}}}}}",
        cmd_out("sha256sum", &[SOURCE]).split_whitespace().next().unwrap_or("-"),
        cmd_out("sha256sum", &[ONEBIT_SRC]).split_whitespace().next().unwrap_or("-"),
        arms.join(",")
    );
    std::fs::write(format!("{SCRATCH}/q2-receipt.json"), receipt)?;
    println!("wrote {SCRATCH}/q2-receipt.json");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a: Vec<&str> = args.iter().map(String::as_str).collect();
    match a.as_slice() {
        ["--time"] => cmd_time(),
        ["--fisher", k0, k1] => cmd_fisher(k0.parse()?, k1.parse()?),
        ["--sketch"] => cmd_sketch(CAL.len()),
        ["--arm", name @ ("uniform" | "separable" | "joint")] => cmd_arm(name),
        ["--merge"] => cmd_merge(),
        _ => Err("usage: qarch_q2 --time | --fisher <k0> <k1> | --sketch | --arm <uniform|separable|joint> | --merge".into()),
    }
}
