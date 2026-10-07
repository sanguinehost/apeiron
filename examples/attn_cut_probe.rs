//! Probe: how many keys must a certified attention cut still read on a real model?
//!
//! Refines (grade `tested`, never `proved`) `Apeiron.Numeric.AttentionCut.dropped_mass_bound`
//! (`proof/Apeiron/Numeric/AttentionCut.lean`, uncommitted, sha256 3bbd73592d6ec51f...): if every key
//! in `D` scores at most `m - delta` and some kept key scores at least `m`, then
//! `|out_full - out_K| <= 2 V |D| e^{-delta}` with `K = rest`, `|v_i| <= V`.
//!
//! Run: `cargo run --release --locked --features gpu --example attn_cut_probe [model.gguf]`
//!
//! Data. qwen05b `Q4_K_M`, 5 prompts of 512 tokens (3 windows of `prose8.txt`, 2 of the llama.cpp
//! docs corpus), tokenised by `apeiron::format::bpe`. The engine does not expose the query vector or
//! the per-layer K/V planes, so the example runs its own f32 forward over the engine's dequantised
//! weights (`dequant_row_fixed`, `decode_vectors`, `rope_table`) and validates the K/V planes it
//! produces against the engine's own `KvState` (`Exact` and `Q8Act` kernels) before measuring.
//!
//! Per (layer, head, query t, causal keys 0..t): `s_i = scale q.k_i`, `m = max s`, `D = {s_i <= m - delta}`,
//! `V = max |v|_inf` over the causal prefix of the kv head. All attention arithmetic is f64 over the
//! f32 q/k/v (the f32 values are taken as exact reals).

#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::format_push_string,
    reason = "measurement harness: numeric casts, indexed loops and exact float sentinels are required"
)]

use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use apeiron::engine::loader::Engine;
use apeiron::engine::{dequant_row_fixed, KernelMode};
use apeiron::format::bpe::Bpe;

const MODEL: &str = "~/models/gguf/qwen05b-Q4_K_M.gguf";
const TOKENIZER: &str = "~/models/qwen2.5-0.5b/tokenizer.json";
const PROSE: &str = "~/fold/hashtrinity/prose8.txt";
const DOCS: &str = "~/gpu-ppl-corpus.txt";
const T: usize = 512;
const HD: usize = 64;
const BLK: usize = 16;
const DELTAS: [f64; 6] = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
const BDELTAS: [f64; 3] = [4.0, 8.0, 12.0];
const EPS: [f64; 3] = [1e-3, 1e-4, 1e-5];
const BUCKET_NAMES: [&str; 5] = ["t<=64", "t<=128", "t<=256", "t<=512", "all"];
const ONE: f64 = 4_294_967_296.0;

fn bucket(n: usize) -> usize {
    match n {
        0..=64 => 0,
        65..=128 => 1,
        129..=256 => 2,
        _ => 3,
    }
}

// ---- accumulators ---------------------------------------------------------------------------

/// Fixed-grid cut: per (bucket, delta).
#[derive(Clone, Copy, Default)]
struct G {
    n: f64,
    frac: f64,
    bound: f64,
    real: f64,
    hold: f64,
}
/// Certified / oracle cut at a tolerance: per (bucket, eps).
#[derive(Clone, Copy, Default)]
struct E {
    n: f64,
    cert: f64,
    oracle: f64,
    real: f64,
    hold: f64,
    lt25: f64,
}
/// Block-box certificate: per (bucket, delta).
#[derive(Clone, Copy, Default)]
struct B {
    n: f64,
    vfrac: f64,
    kfrac: f64,
    bound: f64,
    real: f64,
    hold: f64,
}
/// Softmax mass split (queries with t > 64).
#[derive(Clone, Copy, Default)]
struct M {
    n: f64,
    top8: f64,
    sink: f64,
    win: f64,
    union: f64,
}

#[derive(Clone, Default)]
struct Acc {
    g: [[G; 6]; 4],
    e: [[E; 3]; 4],
    b: [[B; 3]; 4],
    m: M,
    checks: f64,
    holds: f64,
    worst: f64,
    first_violation: Option<String>,
}

impl Acc {
    fn check(&mut self, real: f64, bound: f64, what: impl FnOnce() -> String) -> bool {
        self.checks += 1.0;
        let ok = real <= bound;
        if ok {
            self.holds += 1.0;
        } else if self.first_violation.is_none() {
            self.first_violation = Some(what());
        }
        if bound > 0.0 {
            self.worst = self.worst.max(real / bound);
        }
        ok
    }
    fn merge(&mut self, o: &Acc) {
        for b in 0..4 {
            for j in 0..6 {
                let (x, y) = (&mut self.g[b][j], &o.g[b][j]);
                x.n += y.n;
                x.frac += y.frac;
                x.bound += y.bound;
                x.real += y.real;
                x.hold += y.hold;
            }
            for j in 0..3 {
                let (x, y) = (&mut self.e[b][j], &o.e[b][j]);
                x.n += y.n;
                x.cert += y.cert;
                x.oracle += y.oracle;
                x.real += y.real;
                x.hold += y.hold;
                x.lt25 += y.lt25;
                let (x, y) = (&mut self.b[b][j], &o.b[b][j]);
                x.n += y.n;
                x.vfrac += y.vfrac;
                x.kfrac += y.kfrac;
                x.bound += y.bound;
                x.real += y.real;
                x.hold += y.hold;
            }
        }
        self.m.n += o.m.n;
        self.m.top8 += o.m.top8;
        self.m.sink += o.m.sink;
        self.m.win += o.m.win;
        self.m.union += o.m.union;
        self.checks += o.checks;
        self.holds += o.holds;
        self.worst = self.worst.max(o.worst);
        if self.first_violation.is_none() {
            self.first_violation.clone_from(&o.first_violation);
        }
    }
}

// ---- dense f32 forward ----------------------------------------------------------------------

fn dot32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    for (ca, cb) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for i in 0..8 {
            acc[i] += ca[i] * cb[i];
        }
    }
    acc.iter().sum()
}

/// `y[t][o] = sum_i w[o][i] x[t][i] (+ bias[o])`; `w` is `n_out x n_in` row-major, `x` is `t x n_in`.
fn gemm(
    w: &[f32],
    n_out: usize,
    n_in: usize,
    x: &[f32],
    t: usize,
    bias: Option<&[f32]>,
    nth: usize,
) -> Vec<f32> {
    let mut yt = vec![0f32; n_out * t];
    let rows = n_out.div_ceil(nth);
    std::thread::scope(|s| {
        for (ci, ych) in yt.chunks_mut(rows * t).enumerate() {
            s.spawn(move || {
                let mut tb = 0;
                while tb < t {
                    let te = (tb + 128).min(t);
                    for (ri, yrow) in ych.chunks_mut(t).enumerate() {
                        let o = ci * rows + ri;
                        let wr = &w[o * n_in..(o + 1) * n_in];
                        for tt in tb..te {
                            yrow[tt] = dot32(wr, &x[tt * n_in..(tt + 1) * n_in]);
                        }
                    }
                    tb = te;
                }
            });
        }
    });
    let mut y = vec![0f32; n_out * t];
    for o in 0..n_out {
        let b = bias.map_or(0.0, |b| b[o]);
        for tt in 0..t {
            y[tt * n_out + o] = yt[o * t + tt] + b;
        }
    }
    y
}

fn rmsnorm(x: &[f32], gain: &[f32], eps: f64, d: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (xr, or) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let ms = xr.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / d as f64;
        let r = 1.0 / (ms + eps).sqrt();
        for i in 0..d {
            or[i] = (f64::from(xr[i]) * r * f64::from(gain[i])) as f32;
        }
    }
    out
}

fn q32_vec(v: &[i64]) -> Vec<f32> {
    v.iter().map(|&x| (x as f64 / ONE) as f32).collect()
}

fn weight(e: &Engine, name: &str) -> Result<(Vec<f32>, usize, usize), Box<dyn Error>> {
    let t = e
        .table()
        .get(name)
        .ok_or_else(|| format!("missing {name}"))?;
    let n_in = t.dims[0] as usize;
    let n_out = t.dims[1] as usize;
    let nb = t.n_bytes.ok_or("no byte count")?;
    let raw = e.payload(t.byte_range.start, nb)?;
    let mut out = Vec::with_capacity(n_in * n_out);
    dequant_row_fixed(t.ggml_type, raw, n_in * n_out, &mut out)?;
    if out.len() != n_in * n_out {
        return Err(format!("{name}: dequantised {} != {}", out.len(), n_in * n_out).into());
    }
    Ok((q32_vec(&out), n_in, n_out))
}

// ---- per-(prompt, layer, head) analysis -----------------------------------------------------

struct Prompt {
    /// f64 copies of this layer's roped K and V, position-major `(pos * nkv + kvh) * HD`.
    k: Vec<f64>,
    v: Vec<f64>,
    /// prefix max of `|v|_inf` per kv head, `[kvh][pos]`.
    vpm: Vec<Vec<f64>>,
    /// per kv head, per block: min and max of each key dimension, `[kvh][blk][HD]`.
    bmin: Vec<Vec<f64>>,
    bmax: Vec<Vec<f64>>,
}

fn prompt_layer(k32: &[f32], v32: &[f32], nkv: usize) -> Prompt {
    let k: Vec<f64> = k32.iter().map(|&x| f64::from(x)).collect();
    let v: Vec<f64> = v32.iter().map(|&x| f64::from(x)).collect();
    let mut vpm = vec![vec![0f64; T]; nkv];
    let nblk = T / BLK;
    let mut bmin = vec![vec![f64::INFINITY; nblk * HD]; nkv];
    let mut bmax = vec![vec![f64::NEG_INFINITY; nblk * HD]; nkv];
    for h in 0..nkv {
        let mut run = 0f64;
        for p in 0..T {
            let row = (p * nkv + h) * HD;
            for d in 0..HD {
                run = run.max(v[row + d].abs());
                let b = (p / BLK) * HD + d;
                bmin[h][b] = bmin[h][b].min(k[row + d]);
                bmax[h][b] = bmax[h][b].max(k[row + d]);
            }
            vpm[h][p] = run;
        }
    }
    Prompt {
        k,
        v,
        vpm,
        bmin,
        bmax,
    }
}

/// One head over all `T` queries of one prompt. Cost per query is `O(n (HD + log n))` for the `n`
/// causal keys (scores, one pass of `n` value adds per bucket set, one sort), counted in f64 flops.
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
fn analyze_head(q32: &[f32], p: &Prompt, h: usize, nh: usize, nkv: usize) -> (Vec<f32>, Acc) {
    let kvh = h / (nh / nkv);
    let scale = 1.0 / (HD as f64).sqrt();
    let mut acc = Acc::default();
    let mut att = vec![0f32; T * HD];
    let (mut s, mut g, mut w) = (vec![0f64; T], vec![0f64; T], vec![0f64; T]);
    let mut order: Vec<u32> = Vec::with_capacity(T);
    let mut err = vec![0f64; T + 1];
    for t0 in 0..T {
        let n = t0 + 1;
        let bk = bucket(n);
        let mut qv = [0f64; HD];
        for d in 0..HD {
            qv[d] = f64::from(q32[t0 * nh * HD + h * HD + d]);
        }
        let mut m = f64::NEG_INFINITY;
        for i in 0..n {
            let kr = &p.k[(i * nkv + kvh) * HD..(i * nkv + kvh) * HD + HD];
            let mut a = 0f64;
            for d in 0..HD {
                a += qv[d] * kr[d];
            }
            s[i] = scale * a;
            m = m.max(s[i]);
        }
        let vmax = p.vpm[kvh][t0];
        // bucket b = number of grid deltas <= gap
        let mut nbk = [[0f64; HD]; 7];
        let mut zb = [0f64; 7];
        let mut cnt = [0usize; 7];
        for i in 0..n {
            g[i] = m - s[i];
            w[i] = (-g[i]).exp();
            let b = DELTAS.iter().filter(|&&d| g[i] >= d).count();
            zb[b] += w[i];
            cnt[b] += 1;
            let vr = &p.v[(i * nkv + kvh) * HD..(i * nkv + kvh) * HD + HD];
            for d in 0..HD {
                nbk[b][d] += w[i] * vr[d];
            }
        }
        let mut tn = [0f64; HD];
        let mut tz = 0f64;
        for b in 0..7 {
            tz += zb[b];
            for d in 0..HD {
                tn[d] += nbk[b][d];
            }
        }
        let mut full = [0f64; HD];
        for d in 0..HD {
            full[d] = tn[d] / tz;
            att[t0 * HD + d] = full[d] as f32;
        }
        // fixed grid
        let mut cn = [0f64; HD];
        let mut cz = 0f64;
        let mut ck = 0usize;
        for j in 0..6 {
            cz += zb[j];
            ck += cnt[j];
            for d in 0..HD {
                cn[d] += nbk[j][d];
            }
            let dn = n - ck;
            let mut real = 0f64;
            for d in 0..HD {
                real = real.max((full[d] - cn[d] / cz).abs());
            }
            let bound = 2.0 * vmax * dn as f64 * (-DELTAS[j]).exp();
            let ok = acc.check(real, bound, || {
                format!(
                    "grid delta={} n={n} real={real:e} bound={bound:e} |D|={dn}",
                    DELTAS[j]
                )
            });
            let c = &mut acc.g[bk][j];
            c.n += 1.0;
            c.frac += ck as f64 / n as f64;
            if vmax > 0.0 {
                c.bound += bound / vmax;
                c.real += real / vmax;
            }
            c.hold += f64::from(u8::from(ok));
        }
        // sorted sweep: oracle prefix error, certified cut, mass split
        order.clear();
        order.extend(0..n as u32);
        order.sort_unstable_by(|&a, &b| g[a as usize].total_cmp(&g[b as usize]));
        let mut sn = [0f64; HD];
        let mut sz = 0f64;
        for c in 1..=n {
            let i = order[c - 1] as usize;
            sz += w[i];
            let vr = &p.v[(i * nkv + kvh) * HD..(i * nkv + kvh) * HD + HD];
            let mut e = 0f64;
            for d in 0..HD {
                sn[d] += w[i] * vr[d];
                e = e.max((sn[d] / sz - full[d]).abs());
            }
            err[c] = e;
        }
        err[n] = 0.0; // K = all keys: the same object, not a re-summation
        for (ei, &eps) in EPS.iter().enumerate() {
            let tol = eps * vmax;
            let oracle = (1..=n).find(|&c| err[c] <= tol).unwrap_or(n);
            let mut cert = n;
            let mut bound_c = 0f64;
            for c in 1..n {
                if g[order[c] as usize] > g[order[c - 1] as usize] {
                    let b = 2.0 * vmax * (n - c) as f64 * (-g[order[c] as usize]).exp();
                    if b <= tol {
                        cert = c;
                        bound_c = b;
                        break;
                    }
                }
            }
            let real = err[cert];
            let ok = acc.check(real, bound_c, || {
                format!("eps={eps:e} n={n} cut={cert} real={real:e} bound={bound_c:e}")
            });
            let c = &mut acc.e[bk][ei];
            c.n += 1.0;
            c.cert += cert as f64 / n as f64;
            c.oracle += oracle as f64 / n as f64;
            if vmax > 0.0 {
                c.real += real / vmax;
            }
            c.hold += f64::from(u8::from(ok));
            c.lt25 += f64::from(u8::from((cert as f64) < 0.25 * n as f64));
        }
        if n > 64 {
            let mut in_set = vec![false; n];
            let mut top = 0f64;
            for &i in order.iter().take(8) {
                top += w[i as usize];
                in_set[i as usize] = true;
            }
            let win: f64 = (n - 64..n).map(|i| w[i]).sum();
            for flag in in_set.iter_mut().skip(n - 64) {
                *flag = true;
            }
            in_set[0] = true;
            let uni: f64 = (0..n).filter(|&i| in_set[i]).map(|i| w[i]).sum();
            acc.m.n += 1.0;
            acc.m.top8 += top / tz;
            acc.m.sink += w[0] / tz;
            acc.m.win += win / tz;
            acc.m.union += uni / tz;
        }
        // block-box certificate: read key 0 and the last 64 keys, bound the rest by block boxes
        let r0max = (0..n)
            .filter(|&i| i == 0 || i >= n.saturating_sub(64))
            .map(|i| s[i])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut ubs: Vec<(usize, f64)> = Vec::new();
        let mut bl = 1;
        while (bl + 1) * BLK <= n.saturating_sub(64) {
            let (bmn, bmx) = (
                &p.bmin[kvh][bl * HD..bl * HD + HD],
                &p.bmax[kvh][bl * HD..bl * HD + HD],
            );
            let mut ub = 0f64;
            for d in 0..HD {
                ub += (qv[d] * bmn[d]).max(qv[d] * bmx[d]);
            }
            ubs.push((bl, scale * ub));
            bl += 1;
        }
        for (bi, &bd) in BDELTAS.iter().enumerate() {
            let mut pn = [0f64; HD];
            let mut pz = 0f64;
            let mut pk = 0usize;
            for &(blk, ub) in &ubs {
                if ub <= r0max - bd {
                    for i in blk * BLK..(blk + 1) * BLK {
                        let vr = &p.v[(i * nkv + kvh) * HD..(i * nkv + kvh) * HD + HD];
                        pz += w[i];
                        pk += 1;
                        for d in 0..HD {
                            pn[d] += w[i] * vr[d];
                        }
                    }
                }
            }
            let real = if pk == 0 {
                0.0
            } else {
                let mut r = 0f64;
                for d in 0..HD {
                    r = r.max((full[d] - (tn[d] - pn[d]) / (tz - pz)).abs());
                }
                r
            };
            let bound = 2.0 * vmax * pk as f64 * (-bd).exp();
            let ok = acc.check(real, bound, || {
                format!("box delta={bd} n={n} real={real:e} bound={bound:e} |D|={pk}")
            });
            let c = &mut acc.b[bk][bi];
            c.n += 1.0;
            c.vfrac += (n - pk) as f64 / n as f64;
            c.kfrac += ((n - pk) + 2 * ubs.len()) as f64 / n as f64;
            if vmax > 0.0 {
                c.bound += bound / vmax;
                c.real += real / vmax;
            }
            c.hold += f64::from(u8::from(ok));
        }
    }
    (att, acc)
}

// ---- driver ---------------------------------------------------------------------------------

fn kv_planes(dbg: &str, n_layer: usize) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), Box<dyn Error>> {
    let st = dbg.find("storage:").ok_or("no storage in KvState debug")?;
    let mut pos = st;
    let mut planes = Vec::new();
    for key in ["k: [", "v: ["] {
        let at = dbg[pos..].find(key).ok_or("plane key not found")? + pos + key.len() - 1;
        let bytes = dbg.as_bytes();
        let (mut depth, mut i) = (0i32, at);
        let (mut layers, mut cur, mut tok) = (Vec::new(), Vec::<f64>::new(), String::new());
        loop {
            let c = bytes[i] as char;
            match c {
                '[' => depth += 1,
                ']' | ',' | ' ' => {
                    if !tok.is_empty() {
                        cur.push(tok.parse::<f64>()?);
                        tok.clear();
                    }
                    if c == ']' {
                        depth -= 1;
                        if depth == 1 {
                            layers.push(std::mem::take(&mut cur));
                        }
                    }
                }
                _ => tok.push(c),
            }
            i += 1;
            if depth == 0 {
                break;
            }
        }
        pos = i;
        planes.push(layers);
    }
    let v = planes.pop().unwrap();
    let k = planes.pop().unwrap();
    if k.len() != n_layer || v.len() != n_layer {
        return Err(format!("parsed {} / {} layers, want {n_layer}", k.len(), v.len()).into());
    }
    Ok((k, v))
}

fn rel_l2(a: &[f32], b: &[f64]) -> (f64, f64) {
    let (mut num, mut den, mut mx) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let d = f64::from(*x) - y;
        num += d * d;
        den += y * y;
        mx = mx.max(d.abs());
    }
    ((num / den.max(1e-300)).sqrt(), mx)
}

fn mean_min_max(v: &[f64]) -> String {
    if v.is_empty() {
        return "-".into();
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let mn = v.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    format!("{mean:.4} [{mn:.4},{mx:.4}]")
}

fn sci(v: &[f64]) -> String {
    if v.is_empty() {
        return "-".into();
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let mn = v.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    format!("{mean:.2e} [{mn:.1e},{mx:.1e}]")
}

fn prompts(bpe: &Bpe) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
    let mut out = Vec::new();
    for (path, off) in [
        (PROSE, 0usize),
        (PROSE, 2_000_000),
        (PROSE, 5_000_000),
        (DOCS, 0),
        (DOCS, 30_000),
    ] {
        let raw = std::fs::read(path)?;
        let mut a = off;
        if a > 0 {
            while a < raw.len() && raw[a - 1] != b'\n' {
                a += 1;
            }
        }
        let b = (a + 8000).min(raw.len());
        let text = String::from_utf8_lossy(&raw[a..b]).into_owned();
        let ids = bpe.encode(&text)?;
        if ids.len() < T {
            return Err(format!("{path}@{off}: {} tokens < {T}", ids.len()).into());
        }
        out.push(ids[..T].to_vec());
    }
    Ok(out)
}

#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
fn run(path: &str) -> Result<bool, Box<dyn Error>> {
    let nth = 16usize;
    let eng = Engine::open(path)?;
    let cfg = &eng.config;
    let (nl, nh, nkv, d) = (cfg.n_layer, cfg.n_head, cfg.n_head_kv, cfg.n_embd);
    assert_eq!(cfg.head_dim, HD);
    let bpe = Bpe::from_tokenizer_json(TOKENIZER)?;
    let ps = prompts(&bpe)?;
    let np = ps.len();
    println!("model {path}\nlayers {nl} heads {nh} kv {nkv} hd {HD} n_embd {d}; prompts {np} x {T} tokens");
    for (i, ids) in ps.iter().enumerate() {
        let bytes: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
        let h = apeiron::sha256::sha256(&bytes);
        let hex: String = h.iter().take(8).map(|b| format!("{b:02x}")).collect();
        println!(
            "prompt {i}: ids sha256[..16] {hex}  first ids {:?}",
            &ids[..6]
        );
    }
    let rope = eng.rope_table().map_err(|e| format!("{e:?}"))?;
    // cos/sin tables from the engine's own rotation: rotate (1, 0) pairs once per position
    let half = HD / 2;
    let mut cs = vec![(0f64, 0f64); T * half];
    for p in 0..T {
        let mut x = vec![0i64; HD];
        for xj in x.iter_mut().take(half) {
            *xj = ONE as i64;
        }
        rope.rotate_one(&mut x, p as u32)
            .map_err(|e| format!("{e:?}"))?;
        for j in 0..half {
            cs[p * half + j] = (x[j] as f64 / ONE, x[j + half] as f64 / ONE);
        }
    }
    let rot = |x: &mut [f32], heads: usize, p: usize| {
        for hh in 0..heads {
            let b = hh * HD;
            for j in 0..half {
                let (c, s) = cs[p * half + j];
                let (x0, x1) = (f64::from(x[b + j]), f64::from(x[b + j + half]));
                x[b + j] = (x0 * c - x1 * s) as f32;
                x[b + j + half] = (x1 * c + x0 * s) as f32;
            }
        }
    };
    // embeddings
    let mut x: Vec<f32> = Vec::with_capacity(np * T * d);
    for ids in &ps {
        for &id in ids {
            x.extend(q32_vec(&eng.token_embedding(id)?));
        }
    }
    let vecs = eng.decode_vectors();
    let eps = cfg.rms_eps;
    let accs: Vec<Vec<Mutex<Acc>>> = (0..nl)
        .map(|_| (0..nh).map(|_| Mutex::new(Acc::default())).collect())
        .collect();
    let mut kcap: Vec<Vec<f32>> = Vec::new(); // prompt 0, roped K per layer (validation)
    let mut vcap: Vec<Vec<f32>> = Vec::new();
    let t0 = std::time::Instant::now();
    for l in 0..nl {
        let g1 = q32_vec(
            vecs.norms_of(l)
                .and_then(|n| n[0].as_deref())
                .ok_or("attn_norm")?,
        );
        let g2 = q32_vec(
            vecs.norms_of(l)
                .and_then(|n| n[1].as_deref())
                .ok_or("ffn_norm")?,
        );
        let bias = |i: usize| vecs.biases_of(l).and_then(|b| b[i].as_deref()).map(q32_vec);
        let (bq, bkk, bv) = (bias(0), bias(1), bias(2));
        let xn = rmsnorm(&x, &g1, eps, d);
        let (wq, ni, no) = weight(&eng, &format!("blk.{l}.attn_q.weight"))?;
        let mut q = gemm(&wq, no, ni, &xn, np * T, bq.as_deref(), nth);
        drop(wq);
        let (wk, ni, no) = weight(&eng, &format!("blk.{l}.attn_k.weight"))?;
        let mut k = gemm(&wk, no, ni, &xn, np * T, bkk.as_deref(), nth);
        drop(wk);
        let (wv, ni, no) = weight(&eng, &format!("blk.{l}.attn_v.weight"))?;
        let v = gemm(&wv, no, ni, &xn, np * T, bv.as_deref(), nth);
        drop(wv);
        for pp in 0..np * T {
            let pos = pp % T;
            rot(&mut q[pp * nh * HD..(pp + 1) * nh * HD], nh, pos);
            rot(&mut k[pp * nkv * HD..(pp + 1) * nkv * HD], nkv, pos);
        }
        kcap.push(k[..T * nkv * HD].to_vec());
        vcap.push(v[..T * nkv * HD].to_vec());
        let pl: Vec<Prompt> = (0..np)
            .map(|pi| {
                prompt_layer(
                    &k[pi * T * nkv * HD..(pi + 1) * T * nkv * HD],
                    &v[pi * T * nkv * HD..(pi + 1) * T * nkv * HD],
                    nkv,
                )
            })
            .collect();
        let mut att = vec![0f32; np * T * nh * HD];
        let next = AtomicUsize::new(0);
        let results: Mutex<Vec<(usize, usize, Vec<f32>)>> = Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for _ in 0..nth {
                s.spawn(|| loop {
                    let job = next.fetch_add(1, Ordering::Relaxed);
                    if job >= np * nh {
                        break;
                    }
                    let (pi, h) = (job / nh, job % nh);
                    let qp = &q[pi * T * nh * HD..(pi + 1) * T * nh * HD];
                    let (a, ac) = analyze_head(qp, &pl[pi], h, nh, nkv);
                    accs[l][h].lock().unwrap().merge(&ac);
                    results.lock().unwrap().push((pi, h, a));
                });
            }
        });
        for (pi, h, a) in results.into_inner().unwrap() {
            for t in 0..T {
                let dst = ((pi * T + t) * nh + h) * HD;
                att[dst..dst + HD].copy_from_slice(&a[t * HD..(t + 1) * HD]);
            }
        }
        let (wo, ni, no) = weight(&eng, &format!("blk.{l}.attn_output.weight"))?;
        let o = gemm(&wo, no, ni, &att, np * T, None, nth);
        drop(wo);
        for (xi, oi) in x.iter_mut().zip(&o) {
            *xi += oi;
        }
        let xn2 = rmsnorm(&x, &g2, eps, d);
        let (wg, ni, no) = weight(&eng, &format!("blk.{l}.ffn_gate.weight"))?;
        let mut gate = gemm(&wg, no, ni, &xn2, np * T, None, nth);
        drop(wg);
        let (wu, ni, no) = weight(&eng, &format!("blk.{l}.ffn_up.weight"))?;
        let up = gemm(&wu, no, ni, &xn2, np * T, None, nth);
        drop(wu);
        for (gi, ui) in gate.iter_mut().zip(&up) {
            let gv = f64::from(*gi);
            *gi = (gv / (1.0 + (-gv).exp()) * f64::from(*ui)) as f32;
        }
        let (wd, ni, no) = weight(&eng, &format!("blk.{l}.ffn_down.weight"))?;
        let dn = gemm(&wd, no, ni, &gate, np * T, None, nth);
        drop(wd);
        for (xi, di) in x.iter_mut().zip(&dn) {
            *xi += di;
        }
        eprintln!("layer {l} done, {:.0}s", t0.elapsed().as_secs_f64());
    }

    // ---- validation against the engine's own KvState (prompt 0) ----
    println!("\n== validation: my f32 K/V planes (prompt 0) vs engine KvState ==");
    for (mode, ntok) in [(KernelMode::Exact, 16usize), (KernelMode::Q8Act, T)] {
        let mut st = eng.state(ntok + 1)?;
        let tt = std::time::Instant::now();
        eng.prefill_state(&mut st, &ps[0][..ntok], mode)?;
        let (ek, ev) = kv_planes(&format!("{:?}", st.kv), nl)?;
        let (mut wk, mut wv, mut mk, mut mv) = (0f64, 0f64, 0f64, 0f64);
        for l in 0..nl {
            let width = nkv * HD;
            let a = &kcap[l][..ntok * width];
            let b = &ek[l][..ntok * width];
            let (r, m) = rel_l2(a, b);
            wk = wk.max(r);
            mk = mk.max(m);
            let (r, m) = rel_l2(&vcap[l][..ntok * width], &ev[l][..ntok * width]);
            wv = wv.max(r);
            mv = mv.max(m);
        }
        println!(
            "{mode:?} x{ntok} tokens ({:.1}s): K rel-L2 max over layers {wk:.3e} (max abs {mk:.2e}); V rel-L2 {wv:.3e} (max abs {mv:.2e})",
            tt.elapsed().as_secs_f64()
        );
    }

    // ---- aggregate ----
    let mut all: Vec<Acc> = Vec::new();
    for l in 0..nl {
        for h in 0..nh {
            all.push(accs[l][h].lock().unwrap().clone());
        }
    }
    let pairs = all.len();
    let col = |f: &dyn Fn(&Acc, usize) -> Option<f64>| -> Vec<Vec<f64>> {
        (0..5)
            .map(|b| all.iter().filter_map(|a| f(a, b)).collect())
            .collect()
    };
    let pooled = |a: &Acc, b: usize, j: usize, sel: &dyn Fn(&G) -> f64| -> Option<f64> {
        let cells: Vec<&G> = if b < 4 {
            vec![&a.g[b][j]]
        } else {
            (0..4).map(|x| &a.g[x][j]).collect()
        };
        let n: f64 = cells.iter().map(|c| c.n).sum();
        if n == 0.0 {
            None
        } else {
            Some(cells.iter().map(|c| sel(c)).sum::<f64>() / n)
        }
    };
    let header = || {
        print!("{:<8}", "delta");
        for n in BUCKET_NAMES {
            print!(" {n:<26}");
        }
        println!();
    };
    println!("\n== fixed-delta cut: {pairs} (layer,head) pairs, mean [min,max] over pairs ==");
    for (title, sel, sc) in [
        (
            "|K|/t (keys still read)",
            (|c: &G| c.frac) as fn(&G) -> f64,
            false,
        ),
        ("bound 2|D|e^-delta (units of V)", |c: &G| c.bound, true),
        ("realised max-norm error (units of V)", |c: &G| c.real, true),
    ] {
        println!("-- {title}");
        header();
        for (j, dl) in DELTAS.iter().enumerate() {
            print!("{dl:<8}");
            let cols = col(&|a, b| pooled(a, b, j, &sel));
            for c in &cols {
                print!(" {:<26}", if sc { sci(c) } else { mean_min_max(c) });
            }
            println!();
        }
    }
    println!("\n== certified cut at tolerance eps (absolute error <= eps*V), per query ==");
    println!("cert = smallest |K| with 2|D|e^-delta <= eps (delta = gap of the first dropped key); oracle = smallest prefix of the score order with realised error <= eps*V (not certifiable)");
    for (ei, eps) in EPS.iter().enumerate() {
        println!("-- eps {eps:e}   (mean |K|/t [min,max over pairs]; share of queries with cert |K|/t < 0.25)");
        print!("{:<8}", "");
        for n in BUCKET_NAMES {
            print!(" {n:<26}");
        }
        println!();
        for (name, which) in [("cert", 0), ("oracle", 1), ("lt25", 2)] {
            print!("{name:<8}");
            let cols: Vec<Vec<f64>> = (0..5)
                .map(|b| {
                    all.iter()
                        .filter_map(|a| {
                            let cells: Vec<&E> = if b < 4 {
                                vec![&a.e[b][ei]]
                            } else {
                                (0..4).map(|x| &a.e[x][ei]).collect()
                            };
                            let n: f64 = cells.iter().map(|c| c.n).sum();
                            if n == 0.0 {
                                return None;
                            }
                            let sum: f64 = cells
                                .iter()
                                .map(|c| [c.cert, c.oracle, c.lt25][which])
                                .sum();
                            Some(sum / n)
                        })
                        .collect()
                })
                .collect();
            for c in &cols {
                print!(" {:<26}", mean_min_max(c));
            }
            println!();
        }
    }
    println!("\n== block-box certificate (no key read for a pruned block): block {BLK}, read key 0 + last 64 keys, bound the rest by per-dimension min/max boxes ==");
    println!("V = value-side |K'|/t; K = key-side reads incl. 2 box vectors per checked block, in key-vector units / t");
    for (bi, bd) in BDELTAS.iter().enumerate() {
        println!("-- delta {bd}");
        print!("{:<8}", "");
        for n in BUCKET_NAMES {
            print!(" {n:<26}");
        }
        println!();
        for (name, which) in [("V frac", 0), ("K frac", 1), ("real/V", 2)] {
            print!("{name:<8}");
            let cols: Vec<Vec<f64>> = (0..5)
                .map(|b| {
                    all.iter()
                        .filter_map(|a| {
                            let cells: Vec<&B> = if b < 4 {
                                vec![&a.b[b][bi]]
                            } else {
                                (0..4).map(|x| &a.b[x][bi]).collect()
                            };
                            let n: f64 = cells.iter().map(|c| c.n).sum();
                            if n == 0.0 {
                                return None;
                            }
                            let sum: f64 = cells
                                .iter()
                                .map(|c| [c.vfrac, c.kfrac, c.real][which])
                                .sum();
                            Some(sum / n)
                        })
                        .collect()
                })
                .collect();
            for c in &cols {
                print!(" {:<26}", if which == 2 { sci(c) } else { mean_min_max(c) });
            }
            println!();
        }
    }
    println!(
        "\n== softmax mass split, queries with t > 64, per layer (mean over heads and queries) =="
    );
    println!(
        "{:<6} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "layer", "top8", "sink0", "win64", "union", "remainder"
    );
    for l in 0..nl {
        let mut m = M::default();
        for h in 0..nh {
            let a = &all[l * nh + h].m;
            m.n += a.n;
            m.top8 += a.top8;
            m.sink += a.sink;
            m.win += a.win;
            m.union += a.union;
        }
        let n = m.n.max(1.0);
        println!(
            "{l:<6} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>10.4}",
            m.top8 / n,
            m.sink / n,
            m.win / n,
            m.union / n,
            1.0 - m.union / n
        );
    }
    let mut tot = Acc::default();
    for a in &all {
        tot.merge(a);
    }
    println!(
        "\nchecks {} (grid {} deltas + {} eps cuts + {} box deltas per query)",
        tot.checks,
        DELTAS.len(),
        EPS.len(),
        BDELTAS.len()
    );
    println!(
        "max realised/bound over all non-zero bounds: {:.4}",
        tot.worst
    );
    let pct = 100.0 * tot.holds / tot.checks;
    if let Some(v) = &tot.first_violation {
        println!("COUNTEREXAMPLE (first): {v}");
    }
    println!("bound holds: {pct:.2}%");
    Ok(tot.holds == tot.checks)
}

fn main() -> std::process::ExitCode {
    let path = std::env::args().nth(1).unwrap_or_else(|| MODEL.to_string());
    match run(&path) {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::FAILURE,
        Err(e) => {
            eprintln!("attn_cut_probe: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
