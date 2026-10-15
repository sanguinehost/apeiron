//! Q1 — the cross-layer interference Gram of per-layer quantisation error, on the crate's Rust path.
//!
//! The teacher is `/home/socol/models/gguf/qwen05b-f16.gguf` (Qwen2.5-0.5B, f16) run through the
//! production engine ([`Engine::open`] / [`Engine::step_logits`]: Q32.32 `i64` logits, `>> 32`).
//! A student of layer `l` replaces the seven `blk.l.*` linears by `W + α·(Q(W) − W)`, with `Q` the
//! in-tree one-bit quantiser `apeiron::learn::onebit::quantize(Arm::HadRtn, …)` on an identity Gram
//! and the f16 result written back over the teacher's byte ranges in a reflink copy.
//!
//! Per (α, window): the teacher distribution `p_t`, each single-layer student's
//! `q_{l,t}` and `δψ_{l,t} = √q_{l,t} − √p_t`, the Gram `Γ_lm = mean_t ⟨δψ_l, δψ_m⟩`, the isolated
//! `KL_l = KL(p_t ‖ q_{l,t})`, and for each group `G` the measured `KL_G` (all layers of `G`
//! substituted at once), the amplitude prediction `2‖Σ_{l∈G} δψ_l‖² = 2 Σ_{l,m∈G} Γ_lm`, the
//! isolated sum `Σ_{l∈G} KL_l`, and the two ratios.
//!
//! Run: `qarch_q1 --alpha <a> [--window <w>]` writes `q1-part-<a>-w<w>.tsv` under the scratch
//! directory; `qarch_q1 --merge` folds every part into `q1-receipt.json`; `qarch_q1 --time <n>`
//! projects the wall clock from `n` teacher decode steps.
//!
//! Executing language: everything here is Rust on the crate's build path (no Python).
//!
//! Trust boundary: the claim is about the crate's own integer decode path and the in-tree
//! quantiser; it does not reach rustc/LLVM codegen or the `unsafe` container.

#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::format_push_string,
    reason = "measurement harness: numeric casts between f16/f32/f64 and the Q32.32 integer \
              logits, indexed loops over the flat [T, vocab] activation buffer, and long report \
              builders are the instrument, not a library boundary"
)]

use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::io::{Seek, SeekFrom, Write as IoWrite};
use std::process::Command;
use std::time::Instant;

use apeiron::engine::loader::{Engine, TensorEntry};
use apeiron::format::bpe::Bpe;
use apeiron::format::quant::{self, f16_to_f32};
use apeiron::learn::onebit::{quantize, Arm};

/// The f16 teacher.
const MODEL: &str = concat!(env!("HOME"), "/models/gguf/qwen05b-f16.gguf");
/// The tokeniser `examples/attn_cut_probe.rs` uses.
const TOKENIZER: &str = concat!(env!("HOME"), "/models/qwen2.5-0.5b/tokenizer.json");
/// The perplexity corpus the windows are cut from.
const CORPUS: &str = concat!(env!("HOME"), "/models/gpu-ppl-corpus.txt");
/// Scratch: student files, part files and the receipt.
const SCRATCH: &str = concat!(env!("HOME"), "/oc-tmp/lanes/qarch/s-q1");
/// The receipt the merge writes.
const RECEIPT: &str = concat!(env!("HOME"), "/oc-tmp/lanes/qarch/s-q1/q1-receipt.json");

/// Decode positions per window.
const T: usize = 64;
/// The three disjoint windows, as byte offsets into [`CORPUS`].
const WINDOW_OFFSETS: [usize; 3] = [0, 20_000, 44_000];
/// Bytes read from each offset before tokenising (one prompt's worth of text).
const WINDOW_SPAN: usize = 4_000;
/// The interpolation strengths.
const ALPHAS: [f64; 4] = [0.125, 0.25, 0.5, 1.0];
/// Prefix/suffix cut points.
const GROUP_K: [usize; 5] = [4, 8, 12, 16, 20];
/// The seven substituted linears of a block.
const LINEARS: [&str; 7] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];
/// `2^32`, the Q32.32 unit.
const ONE_F: f64 = 4_294_967_296.0;

// ── f16 encode/decode ───────────────────────────────────────────────────────────────────────────

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

// ── the quantised student ───────────────────────────────────────────────────────────────────────

/// The `n × n` identity Gram the `HadRtn` arm is handed (it uses no Gram feedback).
fn identity_gram(n: usize) -> Vec<f64> {
    let mut g = vec![0f64; n * n];
    for i in 0..n {
        g[i * n + i] = 1.0;
    }
    g
}

/// One tensor's interpolated f16 bytes, `W + α·(Q(W) − W)`, at the teacher's absolute file offset.
fn interpolated_tensor(
    eng: &Engine,
    t: &TensorEntry,
    name: &str,
    alpha: f64,
) -> Result<(u64, Vec<u8>), String> {
    if t.ggml_type != quant::GGML_TYPE_F16 {
        return Err(format!("{name}: ggml type {} is not F16", t.ggml_type));
    }
    if t.dims.len() != 2 {
        return Err(format!("{name}: rank {} is not 2", t.dims.len()));
    }
    let n = usize::try_from(t.dims[0]).map_err(|_| format!("{name}: dims"))?;
    let n_out = usize::try_from(t.dims[1]).map_err(|_| format!("{name}: dims"))?;
    if n % 128 != 0 {
        return Err(format!(
            "{name}: input width {n} is not a multiple of the 128 one-bit block"
        ));
    }
    let cnt = n_out
        .checked_mul(n)
        .ok_or_else(|| format!("{name}: element count overflows"))?;
    let nb = u64::try_from(cnt)
        .map_err(|_| format!("{name}: element count"))?
        .checked_mul(2)
        .ok_or_else(|| format!("{name}: byte count overflows"))?;
    if t.n_bytes != Some(nb) {
        return Err(format!("{name}: stored {:?} bytes != {nb}", t.n_bytes));
    }
    let raw = eng
        .payload(t.byte_range.start, nb)
        .map_err(|e| format!("{name}: {e:?}"))?;
    let mut w = vec![0f32; cnt];
    for i in 0..cnt {
        w[i] = f16_to_f32(u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]));
    }
    let gram = identity_gram(n);
    let q = quantize(Arm::HadRtn, &w, n_out, n, &gram, 128)
        .map_err(|e| format!("{name}: quantize {e:?}"))?;
    let mut bytes = vec![0u8; cnt * 2];
    for i in 0..cnt {
        let wi = f64::from(w[i]);
        // α = 0 is exactly `W`: the literal `wi + 0·(Q−wi)` maps `−0.0` to `+0.0` on the f16 grid.
        let v = if alpha == 0.0 {
            w[i]
        } else {
            (wi + alpha * (f64::from(q.what[i]) - wi)) as f32
        };
        let b = f32_to_f16_bits(v).to_le_bytes();
        bytes[2 * i] = b[0];
        bytes[2 * i + 1] = b[1];
    }
    Ok((t.byte_range.start, bytes))
}

/// Every substituted tensor of block `l`, as `(absolute file offset, f16 bytes)`.
fn interpolated_layer(eng: &Engine, l: usize, alpha: f64) -> Result<Vec<(u64, Vec<u8>)>, String> {
    let mut out = Vec::with_capacity(LINEARS.len());
    for suffix in LINEARS {
        let name = format!("blk.{l}.{suffix}.weight");
        let t = eng
            .table()
            .get(&name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        out.push(interpolated_tensor(eng, t, &name, alpha)?);
    }
    Ok(out)
}

/// A reflink copy of the teacher with `layers`' tensors overwritten in place.
fn build_student(
    cache: &[Vec<(u64, Vec<u8>)>],
    layers: &[usize],
    path: &str,
) -> Result<(), String> {
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
    for &l in layers {
        for (off, bytes) in &cache[l] {
            f.seek(SeekFrom::Start(*off))
                .map_err(|e| format!("seek {path}: {e}"))?;
            f.write_all(bytes)
                .map_err(|e| format!("write {path}: {e}"))?;
        }
    }
    Ok(())
}

// ── the engine forward ──────────────────────────────────────────────────────────────────────────

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

/// `softmax(z)` in place into `p`, returning `(Σ p_v z_v, log-sum-exp(z))` — the two scalars that
/// turn the logits into `KL(p ‖ q)` without a second `ln` pass over the vocabulary.
fn softmax_into(z: &[f64], p: &mut [f32]) -> (f64, f64) {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut s = 0f64;
    for &v in z {
        s += (v - m).exp();
    }
    let inv = 1.0 / s;
    let mut acc = 0f64;
    for i in 0..z.len() {
        let pv = ((z[i] - m).exp() * inv) as f32;
        p[i] = pv;
        acc += f64::from(pv) * z[i];
    }
    (acc, m + s.ln())
}

/// One student's `(mean_t KL(p ‖ q), δψ flattened [T, vocab])` against the teacher's
/// `p`, `Σ p z_p` and `lse(z_p)`.
///
/// `p` and `q` are both rounded to `f32` before the square roots, so identical logits give
/// `q == p` bit for bit and `δψ = 0` exactly, and `Σ p z_p` is accumulated over the stored `f32`
/// `p`, so identical logits give `KL = 0` exactly. That is what makes control (a) an exact zero.
fn kl_and_psi(
    zq: &[Vec<f64>],
    p: &[f32],
    sum_p_zp: &[f64],
    lse_p: &[f64],
    v: usize,
    psi: &mut [f32],
    e: &mut [f64],
) -> f64 {
    let mut kl = 0f64;
    for t in 0..zq.len() {
        let z = &zq[t];
        let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut s = 0f64;
        let mut sum_p_zq = 0f64;
        for i in 0..v {
            let ev = (z[i] - m).exp();
            e[i] = ev;
            s += ev;
            sum_p_zq += f64::from(p[t * v + i]) * z[i];
        }
        let lse_q = m + s.ln();
        kl += sum_p_zp[t] - lse_p[t] - sum_p_zq + lse_q;
        let inv = 1.0 / s;
        let dst = &mut psi[t * v..(t + 1) * v];
        for i in 0..v {
            let qv = (e[i] * inv) as f32;
            let pv = p[t * v + i];
            dst[i] = (f64::from(qv).sqrt() - f64::from(pv).sqrt()) as f32;
        }
    }
    kl / zq.len() as f64
}

// ── groups ──────────────────────────────────────────────────────────────────────────────────────

/// One substitution group.
struct Group {
    name: String,
    layers: Vec<usize>,
}

/// Every single layer, the prefixes `{0..=k}`, the suffixes `{k..=23}` and the whole trunk.
fn groups(nl: usize) -> Vec<Group> {
    let mut g = Vec::new();
    for l in 0..nl {
        g.push(Group {
            name: format!("L{l}"),
            layers: vec![l],
        });
    }
    for k in GROUP_K {
        if k < nl {
            g.push(Group {
                name: format!("prefix_0_{k}"),
                layers: (0..=k).collect(),
            });
        }
        if k < nl {
            g.push(Group {
                name: format!("suffix_{k}_{}", nl - 1),
                layers: (k..nl).collect(),
            });
        }
    }
    g.push(Group {
        name: format!("all_{nl}"),
        layers: (0..nl).collect(),
    });
    g
}

/// The amplitude prediction `2 Σ_{l,m∈G} s_l s_m Γ_lm`; `flip` negates one member's sign.
fn predicted(gram: &[Vec<f64>], layers: &[usize], flip: Option<usize>) -> f64 {
    let mut acc = 0f64;
    for &l in layers {
        for &m in layers {
            let sl = if Some(l) == flip { -1.0 } else { 1.0 };
            let sm = if Some(m) == flip { -1.0 } else { 1.0 };
            acc += sl * sm * gram[l][m];
        }
    }
    2.0 * acc
}

// ── one (α, window) ─────────────────────────────────────────────────────────────────────────────

/// One group's measured/predicted/isolated numbers.
struct GroupResult {
    name: String,
    layers: Vec<usize>,
    measured: f64,
    pred: f64,
    isolated: f64,
}

/// The full per-(α, window) record.
struct WindowResult {
    alpha: f64,
    window: usize,
    kl_single: Vec<f64>,
    gram: Vec<Vec<f64>>,
    groups: Vec<GroupResult>,
    /// `2Γ_ll / KL_l` per layer — control (c)'s quadratic-regime diagnostic.
    ratios: Vec<f64>,
}

impl WindowResult {
    fn to_tsv(&self, model_sha: &str) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "model_sha256\t{model_sha}");
        let _ = writeln!(s, "alpha\t{}", self.alpha);
        let _ = writeln!(s, "window\t{}", self.window);
        let _ = writeln!(s, "vocab\t{}", self.kl_single.len());
        for (l, v) in self.kl_single.iter().enumerate() {
            let _ = writeln!(s, "kl_single\t{l}\t{v:.12e}");
        }
        for (l, row) in self.gram.iter().enumerate() {
            for (m, v) in row.iter().enumerate() {
                let _ = writeln!(s, "gram\t{l}\t{m}\t{v:.12e}");
            }
        }
        for g in &self.groups {
            let layers: Vec<String> = g.layers.iter().map(usize::to_string).collect();
            let _ = writeln!(
                s,
                "group\t{}\t{}\t{:.12e}\t{:.12e}\t{:.12e}",
                g.name,
                layers.join(","),
                g.measured,
                g.pred,
                g.isolated
            );
        }
        let _ = writeln!(s, "end");
        s
    }
}

/// Run one (α, window): the teacher pass, the 24 single-layer students, the Gram and the groups.
fn run_window(
    eng: &Engine,
    cache: &[Vec<(u64, Vec<u8>)>],
    ids: &[u32],
    alpha: f64,
    window: usize,
    ctrl_b: &mut Option<String>,
    ctrls: &mut Vec<String>,
) -> Result<WindowResult, String> {
    let cfg = &eng.config;
    let nl = cfg.n_layer;
    let v = cfg.vocab_size;
    let t0 = Instant::now();

    // Teacher.
    let zt = forward_logits(eng, ids)?;
    let mut p = vec![0f32; T * v];
    let mut sum_p_zp = vec![0f64; T];
    let mut lse_p = vec![0f64; T];
    for t in 0..T {
        let (a, l) = softmax_into(&zt[t], &mut p[t * v..(t + 1) * v]);
        sum_p_zp[t] = a;
        lse_p[t] = l;
    }
    drop(zt);
    println!(
        "  [α={alpha} w={window}] teacher {T} positions in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    // The single-layer students: δψ and KL_l.
    let mut psi = vec![0f32; nl * T * v];
    let mut kl_single = vec![0f64; nl];
    let mut e = vec![0f64; v];
    let path = format!("{SCRATCH}/q1-student-{alpha}-L.gguf");
    for l in 0..nl {
        let ts = Instant::now();
        build_student(cache, &[l], &path)?;
        let seng = Engine::open(&path).map_err(|e| format!("open student L{l}: {e:?}"))?;
        let zq = forward_logits(&seng, ids)?;
        drop(seng);
        let _ = std::fs::remove_file(&path);
        let dst = &mut psi[l * T * v..(l + 1) * T * v];
        kl_single[l] = kl_and_psi(&zq, &p, &sum_p_zp, &lse_p, v, dst, &mut e);
        print!(
            "\r  [α={alpha} w={window}] single {}/{} ({:.1}s)   ",
            l + 1,
            nl,
            ts.elapsed().as_secs_f64()
        );
        let _ = std::io::stdout().flush();
    }
    println!();
    for l in 0..nl {
        if kl_single[l] < 0.0 {
            return Err(format!("control (c): KL_{l} = {} < 0", kl_single[l]));
        }
    }

    // Γ_lm = mean_t ⟨δψ_l, δψ_m⟩.
    let mut gram = vec![vec![0f64; nl]; nl];
    for l in 0..nl {
        for m in l..nl {
            let a = &psi[l * T * v..(l + 1) * T * v];
            let b = &psi[m * T * v..(m + 1) * T * v];
            let mut acc = 0f64;
            for i in 0..T * v {
                acc += f64::from(a[i]) * f64::from(b[i]);
            }
            let g = acc / T as f64;
            gram[l][m] = g;
            gram[m][l] = g;
        }
    }

    // Control (b): the instrument can see the cross term — flipping one member's sign changes the
    // group prediction.
    if ctrl_b.is_none() {
        let g = groups(nl);
        let target = g
            .iter()
            .find(|g| g.name == "prefix_0_4")
            .ok_or("control (b): no prefix_0_4 group")?;
        let base = predicted(&gram, &target.layers, None);
        let flipped = predicted(&gram, &target.layers, Some(0));
        if !(base - flipped).abs().gt(&1e-30) {
            return Err(format!(
                "control (b): sign flip left the prediction unchanged ({base} vs {flipped})"
            ));
        }
        let msg = format!(
            "control (b) sign-flip on δψ_0 in {}: predicted {:.9e} → {:.9e}, |Δ| {:.3e}",
            target.name,
            base,
            flipped,
            (base - flipped).abs()
        );
        println!("  {msg}");
        *ctrl_b = Some(msg);
    }

    // Control (c), quadratic regime: 2‖δψ_l‖² = 2Γ_ll against KL_l. Reported here; the α = 0.125
    // factor-2 band is asserted by the caller once every window's part file is on disk.
    let mut ratios = Vec::with_capacity(nl);
    for l in 0..nl {
        ratios.push(2.0 * gram[l][l] / kl_single[l].max(1e-300));
    }
    let smin = ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let smax = ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let msg = format!(
        "control (c) α={alpha} w={window}: KL_l ≥ 0 for all {nl}; 2Γ_ll/KL_l min {smin:.4} max {smax:.4}"
    );
    println!("  {msg}");
    if (alpha - 0.125).abs() < 1e-12 {
        for l in 0..nl {
            println!(
                "    c-layer L{l}: KL {:.6e} 2Γ_ll {:.6e} ratio {:.4}",
                kl_single[l],
                ratios[l] * kl_single[l],
                ratios[l]
            );
        }
    }
    ctrls.push(msg);

    // The groups: singles reuse KL_l; every other group is substituted and measured.
    let mut out_groups = Vec::new();
    let gs = groups(nl);
    let gpath = format!("{SCRATCH}/q1-student-{alpha}-G.gguf");
    for g in &gs {
        let pred = predicted(&gram, &g.layers, None);
        let isolated: f64 = g.layers.iter().map(|&l| kl_single[l]).sum();
        let measured = if g.layers.len() == 1 {
            kl_single[g.layers[0]]
        } else {
            let ts = Instant::now();
            build_student(cache, &g.layers, &gpath)?;
            let seng = Engine::open(&gpath).map_err(|e| format!("open group {}: {e:?}", g.name))?;
            let zq = forward_logits(&seng, ids)?;
            drop(seng);
            let mut e2 = vec![0f64; v];
            let mut sink = vec![0f32; zq.len() * v];
            let kl = kl_and_psi(&zq, &p, &sum_p_zp, &lse_p, v, &mut sink, &mut e2);
            let _ = std::fs::remove_file(&gpath);
            println!(
                "  [α={alpha} w={window}] group {} ({} layers) KL {kl:.6e} in {:.1}s",
                g.name,
                g.layers.len(),
                ts.elapsed().as_secs_f64()
            );
            kl
        };
        out_groups.push(GroupResult {
            name: g.name.clone(),
            layers: g.layers.clone(),
            measured,
            pred,
            isolated,
        });
    }
    drop(psi);
    println!(
        "  [α={alpha} w={window}] window total {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    Ok(WindowResult {
        alpha,
        window,
        kl_single,
        gram,
        groups: out_groups,
        ratios,
    })
}

// ── the α = 0 control ───────────────────────────────────────────────────────────────────────────

/// Control (a): the α = 0 student is byte-identical to the teacher, so its logits are bit-identical
/// and `Γ = 0`, `KL = 0` exactly.
fn control_a(eng: &Engine, ids: &[u32]) -> Result<String, String> {
    let nl = eng.config.n_layer;
    let v = eng.config.vocab_size;
    let mut cache = Vec::with_capacity(nl);
    let mut checked: u64 = 0;
    for l in 0..nl {
        let blk = interpolated_layer(eng, l, 0.0)?;
        for (off, bytes) in &blk {
            let raw = eng
                .payload(*off, bytes.len() as u64)
                .map_err(|e| format!("control (a): {e:?}"))?;
            if raw != bytes.as_slice() {
                let i = raw
                    .iter()
                    .zip(bytes.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(0)
                    & !1;
                return Err(format!(
                    "control (a): α=0 bytes differ at offset {off}, f16 {:#06x} vs {:#06x}",
                    u16::from_le_bytes([raw[i], raw[i + 1]]),
                    u16::from_le_bytes([bytes[i], bytes[i + 1]])
                ));
            }
            checked += bytes.len() as u64;
        }
        cache.push(blk);
    }
    let path = format!("{SCRATCH}/q1-student-0-a.gguf");
    build_student(&cache, &(0..nl).collect::<Vec<_>>(), &path)?;
    drop(cache);
    let seng = Engine::open(&path).map_err(|e| format!("control (a): open {e:?}"))?;
    let zt = forward_logits(eng, ids)?;
    let zs = forward_logits(&seng, ids)?;
    drop(seng);
    let _ = std::fs::remove_file(&path);
    for t in 0..T {
        if zt[t] != zs[t] {
            return Err(format!(
                "control (a): α=0 logits differ from the teacher at position {t}"
            ));
        }
    }
    // The stored quantities, exactly: p over the α=0 student, δψ ≡ 0, KL ≡ 0.
    let mut p = vec![0f32; T * v];
    let mut sum_p_zp = vec![0f64; T];
    let mut lse_p = vec![0f64; T];
    for t in 0..T {
        let (a, l) = softmax_into(&zt[t], &mut p[t * v..(t + 1) * v]);
        sum_p_zp[t] = a;
        lse_p[t] = l;
    }
    let mut psi = vec![0f32; T * v];
    let mut e = vec![0f64; v];
    let kl = kl_and_psi(&zs, &p, &sum_p_zp, &lse_p, v, &mut psi, &mut e);
    if kl != 0.0 {
        return Err(format!("control (a): α=0 KL = {kl} is not exactly 0"));
    }
    if let Some((i, x)) = psi.iter().enumerate().find(|(_, x)| **x != 0.0) {
        return Err(format!("control (a): α=0 δψ[{i}] = {x} is not exactly 0"));
    }
    Ok(format!(
        "control (a) α=0: {checked} f16 bytes identical to the teacher over {nl} layers; \
         {T} logit vectors bit-identical; δψ ≡ 0 and Γ = 0, KL = 0 exactly"
    ))
}

// ── metadata ────────────────────────────────────────────────────────────────────────────────────

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

/// The three windows as token ids, with their recorded byte offsets.
fn windows() -> Result<[(usize, Vec<u32>); 3], String> {
    let bpe = Bpe::from_tokenizer_json(TOKENIZER).map_err(|e| format!("tokenizer: {e:?}"))?;
    let raw = std::fs::read(CORPUS).map_err(|e| format!("{CORPUS}: {e}"))?;
    let mut out = Vec::new();
    for off in WINDOW_OFFSETS {
        if off >= raw.len() {
            return Err(format!("window offset {off} past {CORPUS}"));
        }
        let b = (off + WINDOW_SPAN).min(raw.len());
        let text = String::from_utf8_lossy(&raw[off..b]).into_owned();
        let ids = bpe
            .encode(&text)
            .map_err(|e| format!("encode@{off}: {e:?}"))?;
        if ids.len() < T {
            return Err(format!("window@{off}: {} tokens < {T}", ids.len()));
        }
        out.push((off, ids[..T].to_vec()));
    }
    Ok([out.remove(0), out.remove(0), out.remove(0)])
}

fn alpha_tag(a: f64) -> String {
    format!("{a}")
}

fn model_sha256(eng: &Engine) -> Result<String, String> {
    let all = eng
        .payload(0, eng.file_len)
        .map_err(|e| format!("sha256 payload: {e:?}"))?;
    Ok(hex(&apeiron::sha256::sha256(all)))
}

// ── merge ───────────────────────────────────────────────────────────────────────────────────────

/// One part file, parsed back.
struct Part {
    alpha: f64,
    window: usize,
    kl_single: Vec<f64>,
    gram: Vec<Vec<f64>>,
    groups: Vec<(String, Vec<usize>, f64, f64, f64)>,
}

fn parse_part(text: &str) -> Result<Part, String> {
    let mut p = Part {
        alpha: f64::NAN,
        window: usize::MAX,
        kl_single: Vec::new(),
        gram: Vec::new(),
        groups: Vec::new(),
    };
    let mut rows: Vec<(usize, usize, f64)> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "alpha" => p.alpha = f[1].parse().map_err(|_| "alpha".to_string())?,
            "window" => p.window = f[1].parse().map_err(|_| "window".to_string())?,
            "kl_single" => p
                .kl_single
                .push(f[2].parse().map_err(|_| "kl_single".to_string())?),
            "gram" => rows.push((
                f[1].parse().map_err(|_| "gram l".to_string())?,
                f[2].parse().map_err(|_| "gram m".to_string())?,
                f[3].parse().map_err(|_| "gram v".to_string())?,
            )),
            "group" => {
                let layers: Vec<usize> = f[2]
                    .split(',')
                    .map(|x| x.parse().unwrap_or(usize::MAX))
                    .collect();
                p.groups.push((
                    f[1].to_string(),
                    layers,
                    f[3].parse().map_err(|_| "group measured".to_string())?,
                    f[4].parse().map_err(|_| "group predicted".to_string())?,
                    f[5].parse().map_err(|_| "group isolated".to_string())?,
                ));
            }
            _ => {}
        }
    }
    let n = p.kl_single.len();
    let mut g = vec![vec![0f64; n]; n];
    for (l, m, v) in rows {
        g[l][m] = v;
    }
    p.gram = g;
    Ok(p)
}

fn ms(v: &[f64]) -> (f64, f64, f64) {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mean, lo, hi)
}

fn json(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.6e}")
    } else {
        "null".into()
    }
}

fn merge() -> Result<(), Box<dyn Error>> {
    let mut parts: Vec<Part> = Vec::new();
    let mut names: Vec<String> = std::fs::read_dir(SCRATCH)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("q1-part-") && n.ends_with(".tsv"))
        .collect();
    names.sort();
    for n in &names {
        let text = std::fs::read_to_string(format!("{SCRATCH}/{n}"))?;
        parts.push(parse_part(&text).map_err(|e| format!("{n}: {e}"))?);
    }
    if parts.is_empty() {
        return Err("merge: no q1-part-*.tsv found".into());
    }
    parts.sort_by(|a, b| {
        a.alpha
            .partial_cmp(&b.alpha)
            .unwrap()
            .then(a.window.cmp(&b.window))
    });

    let eng = Engine::open(MODEL).map_err(|e| format!("{e:?}"))?;
    let sha = model_sha256(&eng)?;
    let wins = windows()?;
    let machine = cmd_out("uname", &["-n"]);
    let nproc = cmd_out("nproc", &[]);
    let rev = cmd_out("git", &["rev-parse", "HEAD"]);
    let nl = eng.config.n_layer;

    // Per-(α, window) records, grouped by α.
    let mut alphas: Vec<f64> = Vec::new();
    for p in &parts {
        if !alphas.iter().any(|a| (a - p.alpha).abs() < 1e-15) {
            alphas.push(p.alpha);
        }
    }
    alphas.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut body = String::new();
    let _ = writeln!(body, "{{");
    let _ = writeln!(body, "  \"task\": \"APEIRON-QARCH-Q1\",");
    let _ = writeln!(body, "  \"machine\": \"{machine}\",");
    let _ = writeln!(body, "  \"nproc\": \"{nproc}\",");
    let _ = writeln!(body, "  \"git_rev\": \"{rev}\",");
    let _ = writeln!(body, "  \"model\": \"{MODEL}\",");
    let _ = writeln!(body, "  \"model_sha256\": \"{sha}\",");
    let _ = writeln!(body, "  \"tokenizer\": \"{TOKENIZER}\",");
    let _ = writeln!(body, "  \"corpus\": \"{CORPUS}\",");
    let _ = writeln!(body, "  \"t\": {T},");
    let _ = writeln!(body, "  \"vocab\": {},", eng.config.vocab_size);
    let _ = writeln!(body, "  \"n_layer\": {nl},");
    let _ = writeln!(
        body,
        "  \"window_offsets\": [{}],",
        WINDOW_OFFSETS
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut wj = String::new();
    for (i, (off, ids)) in wins.iter().enumerate() {
        let bytes: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
        let _ = write!(
            wj,
            "{}{{\"window\": {i}, \"offset\": {off}, \"n_tokens\": {}, \"token_ids_sha256\": \"{}\"}}",
            if i == 0 { "" } else { ", " },
            ids.len(),
            hex(&apeiron::sha256::sha256(&bytes))
        );
    }
    let _ = writeln!(body, "  \"windows\": [{wj}],");
    let _ = writeln!(
        body,
        "  \"linears\": [{}],",
        LINEARS
            .iter()
            .map(|s| format!("\"{s}\""))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut per_alpha = String::new();
    let mut tables = String::new();
    for a in &alphas {
        let mine: Vec<&Part> = parts
            .iter()
            .filter(|p| (p.alpha - a).abs() < 1e-15)
            .collect();
        let _ = write!(
            per_alpha,
            "{}{{\n    \"alpha\": {},\n",
            if per_alpha.is_empty() { "" } else { ",\n" },
            a
        );
        let _ = write!(per_alpha, "    \"per_window\": [");
        for (i, p) in mine.iter().enumerate() {
            let _ = write!(
                per_alpha,
                "{}{{\"window\": {}, \"kl_single\": [{}], \"gram\": [{}], \"groups\": [",
                if i == 0 { "" } else { ", " },
                p.window,
                p.kl_single
                    .iter()
                    .map(|v| json(*v))
                    .collect::<Vec<_>>()
                    .join(", "),
                p.gram
                    .iter()
                    .map(|r| format!(
                        "[{}]",
                        r.iter().map(|v| json(*v)).collect::<Vec<_>>().join(", ")
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            for (k, (name, layers, m, pr, iso)) in p.groups.iter().enumerate() {
                let _ = write!(
                    per_alpha,
                    "{}{{\"name\": \"{name}\", \"layers\": [{}], \"kl_measured\": {}, \"kl_predicted\": {}, \"isolated_sum\": {}, \"measured_over_predicted\": {}, \"isolated_over_measured\": {}}}",
                    if k == 0 { "" } else { ", " },
                    layers.iter().map(usize::to_string).collect::<Vec<_>>().join(", "),
                    json(*m),
                    json(*pr),
                    json(*iso),
                    json(m / pr),
                    json(iso / m)
                );
            }
            let _ = write!(per_alpha, "]}}");
        }
        let _ = write!(per_alpha, "],\n    \"group_means\": [");
        let gnames: Vec<String> = mine[0].groups.iter().map(|g| g.0.clone()).collect();
        for (k, name) in gnames.iter().enumerate() {
            let mut mm = Vec::new();
            let mut pp = Vec::new();
            let mut mp = Vec::new();
            let mut im = Vec::new();
            for p in &mine {
                let g = &p.groups[k];
                mm.push(g.2);
                pp.push(g.3);
                mp.push(g.2 / g.3);
                im.push(g.4 / g.2);
            }
            let (mm_m, mm_lo, mm_hi) = ms(&mm);
            let (pp_m, pp_lo, pp_hi) = ms(&pp);
            let (mp_m, mp_lo, mp_hi) = ms(&mp);
            let (im_m, im_lo, im_hi) = ms(&im);
            let _ = write!(
                per_alpha,
                "{}{{\"name\": \"{name}\", \"kl_measured\": {{\"mean\": {}, \"min\": {}, \"max\": {}}}, \"kl_predicted\": {{\"mean\": {}, \"min\": {}, \"max\": {}}}, \"measured_over_predicted\": {{\"mean\": {}, \"min\": {}, \"max\": {}}}, \"isolated_over_measured\": {{\"mean\": {}, \"min\": {}, \"max\": {}}}}}",
                if k == 0 { "" } else { ", " },
                json(mm_m), json(mm_lo), json(mm_hi),
                json(pp_m), json(pp_lo), json(pp_hi),
                json(mp_m), json(mp_lo), json(mp_hi),
                json(im_m), json(im_lo), json(im_hi)
            );
        }
        let _ = write!(per_alpha, "],\n    \"gram_mean\"");
        // Γ aggregate over the windows of this α.
        let mut gram_mean = vec![vec![0f64; nl]; nl];
        for p in &mine {
            for l in 0..nl {
                for m in 0..nl {
                    gram_mean[l][m] += p.gram[l][m];
                }
            }
        }
        let c = mine.len() as f64;
        let mut sum_all = 0f64;
        let mut sum_diag = 0f64;
        let mut sum_off = 0f64;
        let mut adj = 0f64;
        let mut adj_n = 0;
        let mut far = 0f64;
        let mut far_n = 0;
        for l in 0..nl {
            for m in 0..nl {
                let g = gram_mean[l][m] / c;
                sum_all += g;
                if l == m {
                    sum_diag += g;
                } else {
                    sum_off += g;
                    let d = l.abs_diff(m);
                    if d <= 1 {
                        adj += g;
                        adj_n += 1;
                    } else if d >= 8 {
                        far += g;
                        far_n += 1;
                    }
                }
            }
        }
        let _ = write!(
            per_alpha,
            ": {{\"mean_over_entries\": {}, \"mean_diagonal\": {}, \"mean_offdiagonal\": {}, \"sum_diagonal\": {}, \"sum_abs_all\": {}, \"diagonal_share\": {}, \"offdiag_adjacent_mean\": {}, \"offdiag_far_mean\": {}}}",
            json(sum_all / (nl * nl) as f64),
            json(sum_diag / nl as f64),
            json(sum_off / (nl * (nl - 1)) as f64),
            json(sum_diag),
            json(sum_diag + sum_off.abs()),
            json(sum_diag / (sum_diag + sum_off.abs()).max(1e-300)),
            json(if adj_n > 0 { adj / adj_n as f64 } else { f64::NAN }),
            json(if far_n > 0 { far / far_n as f64 } else { f64::NAN })
        );
        let _ = write!(per_alpha, "}}");
        // stdout table.
        let _ = writeln!(
            tables,
            "alpha {a}: Γ diagonal share {:.4}, offdiag mean {:.3e}, adjacent(|l-m|≤1) mean {:.3e}, far(|l-m|≥8) mean {:.3e}",
            sum_diag / (sum_diag + sum_off.abs()).max(1e-300),
            sum_off / (nl * (nl - 1)) as f64,
            if adj_n > 0 { adj / adj_n as f64 } else { f64::NAN },
            if far_n > 0 { far / far_n as f64 } else { f64::NAN }
        );
        for name in [
            "all_24",
            "prefix_0_4",
            "prefix_0_8",
            "prefix_0_12",
            "prefix_0_16",
            "prefix_0_20",
            "suffix_4_23",
            "suffix_8_23",
            "suffix_12_23",
            "suffix_16_23",
            "suffix_20_23",
        ] {
            if let Some(k) = gnames.iter().position(|n| n == name) {
                let mut mm = Vec::new();
                let mut pp = Vec::new();
                let mut mp = Vec::new();
                let mut im = Vec::new();
                for p in &mine {
                    let g = &p.groups[k];
                    mm.push(g.2);
                    pp.push(g.3);
                    mp.push(g.2 / g.3);
                    im.push(g.4 / g.2);
                }
                let (mm_m, mm_lo, mm_hi) = ms(&mm);
                let (pp_m, ..) = ms(&pp);
                let (mp_m, mp_lo, mp_hi) = ms(&mp);
                let (im_m, ..) = ms(&im);
                let _ = writeln!(
                    tables,
                    "  {name:<12} KL_G {mm_m:.4e} [{mm_lo:.4e},{mm_hi:.4e}]  pred {pp_m:.4e}  meas/pred {mp_m:.3} [{mp_lo:.3},{mp_hi:.3}]  iso/meas {im_m:.3}"
                );
            }
        }
    }
    let _ = writeln!(body, "  \"per_alpha\": [{per_alpha}],");

    // Controls: every controls file the campaign wrote.
    let mut ctrls: Vec<String> = Vec::new();
    let mut cnames: Vec<String> = std::fs::read_dir(SCRATCH)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("q1-controls") && n.ends_with(".txt"))
        .collect();
    cnames.sort();
    for n in &cnames {
        if let Ok(t) = std::fs::read_to_string(format!("{SCRATCH}/{n}")) {
            for line in t.lines().filter(|l| !l.trim().is_empty()) {
                if !ctrls.iter().any(|c| c == line) {
                    ctrls.push(line.to_string());
                }
            }
        }
    }
    let mut cj = String::new();
    for line in &ctrls {
        let _ = write!(
            cj,
            "{}\"{}\"",
            if cj.is_empty() { "" } else { ", " },
            line.replace('\\', "/").replace('"', "'")
        );
    }
    let _ = writeln!(body, "  \"controls\": [{cj}]");
    let _ = writeln!(body, "}}");

    std::fs::write(RECEIPT, &body)?;
    println!("{tables}");
    println!("receipt: {RECEIPT} ({} bytes)", body.len());
    Ok(())
}

// ── main ────────────────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    if args.iter().any(|a| a == "--merge") {
        return merge();
    }

    if args.iter().any(|a| a == "--selftest") {
        let mut bad = 0u32;
        let mut first: Vec<(u16, u16, f32)> = Vec::new();
        for h in 0u32..65536 {
            let h = h as u16;
            let x = f16_to_f32(h);
            if x.is_nan() {
                continue;
            }
            let r = f32_to_f16_bits(x);
            if r != h {
                bad += 1;
                if first.len() < 12 {
                    first.push((h, r, x));
                }
            }
        }
        println!("f16 round-trip: {bad} mismatches of 65536");
        for (h, r, x) in first {
            println!("  {h:#06x} -> {x:e} -> {r:#06x}");
        }
        return Ok(());
    }

    let eng = Engine::open(MODEL).map_err(|e| format!("open {MODEL}: {e:?}"))?;
    let wins = windows()?;

    if let Some(n) = opt("--time") {
        let n: usize = n.parse().unwrap_or(8);
        let ids = &wins[0].1;
        let t0 = Instant::now();
        let _ = forward_logits(&eng, &ids[..n.min(ids.len())])?;
        let el = t0.elapsed().as_secs_f64();
        println!(
            "time: {} teacher steps in {el:.2}s = {:.2} tok/s; projected (α,w) cost {:.0}s",
            n.min(ids.len()),
            n as f64 / el,
            35.0 * T as f64 * el / n as f64 + 30.0
        );
        return Ok(());
    }

    let nl = eng.config.n_layer;
    let selected: Vec<f64> = match opt("--alpha") {
        Some(a) => vec![a.parse()?],
        None => ALPHAS.to_vec(),
    };
    let win_sel: Vec<usize> = match opt("--window") {
        Some(w) => vec![w.parse()?],
        None => vec![0, 1, 2],
    };

    // Control (a): α = 0 is the teacher, byte for byte.
    let ca = control_a(&eng, &wins[0].1)?;
    println!("{ca}");
    let sha = model_sha256(&eng)?;
    println!("model sha256 {sha}");

    let mut c_fail: Vec<String> = Vec::new();
    for alpha in &selected {
        let alpha = *alpha;
        let t0 = Instant::now();
        println!("α={alpha}: building the interpolated one-bit student cache");
        let mut cache = Vec::with_capacity(nl);
        for l in 0..nl {
            cache.push(interpolated_layer(&eng, l, alpha)?);
        }
        println!(
            "α={alpha}: student cache built in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        let mut ctrl_b: Option<String> = None;
        let mut ctrls: Vec<String> = vec![ca.clone()];
        for &w in &win_sel {
            let r = run_window(&eng, &cache, &wins[w].1, alpha, w, &mut ctrl_b, &mut ctrls)?;
            let tsv = r.to_tsv(&sha);
            let path = format!("{SCRATCH}/q1-part-{}-w{}.tsv", alpha_tag(alpha), w);
            std::fs::write(&path, &tsv)?;
            println!("wrote {path}");
            if (alpha - 0.125).abs() < 1e-12 {
                let lo = r.ratios.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = r.ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if lo < 0.5 || hi > 2.0 {
                    let m = format!(
                        "control (c) α=0.125 w={w}: 2‖δψ_l‖²/KL_l leaves [0.5, 2] (min {lo:.4}, max {hi:.4})"
                    );
                    println!("  {m}");
                    ctrls.push(m.clone());
                    c_fail.push(m);
                }
            }
        }
        if let Some(b) = &ctrl_b {
            ctrls.push(b.clone());
        }
        let mut text = String::new();
        for c in &ctrls {
            let _ = writeln!(text, "{c}");
        }
        std::fs::write(
            format!("{SCRATCH}/q1-controls-{}.txt", alpha_tag(alpha)),
            text,
        )?;
        println!("α={alpha} complete in {:.1}s", t0.elapsed().as_secs_f64());
    }
    drop(eng);
    if !c_fail.is_empty() {
        return Err(c_fail.join("; ").into());
    }
    Ok(())
}
