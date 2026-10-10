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
//!
//! B4 arm (after the sections above, which are unchanged): one 16,384-token prompt through the same
//! forward; queries are the last 4 positions of the prefixes t in {512, 2048, 4096, 8192},
//! every layer and head. Non-kept keys are grouped (positional blocks, static PCA tree per
//! (layer, kv head), per-query score-sorted oracle runs, 8..64 keys) and a node is summarised by its
//! p-term Taylor value about the mean key's score when `B <= 1` and the certified bound
//! (`apeiron::attn_cut::certified_read`) stays within `eps V`. Reports `read_equiv/t` with nodes
//! costed at their stored moments, a spread-only reference (node = 1 key), and growth exponents.

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

use apeiron::attn_cut::{certified_read, Part};
use apeiron::engine::loader::Engine;
use apeiron::engine::{dequant_row_fixed, KernelMode};
use apeiron::format::bpe::Bpe;

const MODEL: &str = concat!(env!("HOME"), "/models/gguf/qwen05b-Q4_K_M.gguf");
const TOKENIZER: &str = concat!(env!("HOME"), "/models/qwen2.5-0.5b/tokenizer.json");
const PROSE: &str = concat!(env!("HOME"), "/Workspace/fold/hashtrinity/prose8.txt");
const DOCS: &str = concat!(env!("HOME"), "/models/gpu-ppl-corpus.txt");
const T: usize = 512;
/// Head dimension, fixed at build time: `PROBE_HD=128` builds the instrument for Qwen2.5-7B; the default 64 is
/// Qwen2.5-0.5B. `run_dump` and `run_tail` assert it against the model's `head_dim`.
const HD: usize = match option_env!("PROBE_HD") {
    Some(s) => parse_usize(s),
    None => 64,
};

/// Softmax scale `1/sqrt(HD)`.
const SCALE: f32 = match HD {
    64 => 0.125,
    128 => 0.088_388_35,
    _ => panic!("PROBE_HD must be 64 or 128"),
};

const fn parse_usize(s: &str) -> usize {
    let b = s.as_bytes();
    let mut i = 0;
    let mut n = 0;
    while i < b.len() {
        assert!(b[i].is_ascii_digit(), "PROBE_HD must be a decimal integer");
        n = n * 10 + (b[i] - b'0') as usize;
        i += 1;
    }
    n
}
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

/// Applies `f(first_item, chunk)` to `nth` contiguous chunks of `data` (`row` elements per item) on scoped threads.
fn par_rows<T: Send>(data: &mut [T], row: usize, nth: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let per = (data.len() / row).div_ceil(nth.max(1)).max(1);
    std::thread::scope(|sc| {
        for (c, ch) in data.chunks_mut(per * row).enumerate() {
            let f = &f;
            sc.spawn(move || f(c * per, ch));
        }
    });
}

/// `dot32` of four queries against one key row in AVX2 registers: four independent accumulator chains, each result
/// bit-identical to `dot32(q[t], b)` (separate multiply and add, the same lane order and final sum).
#[target_feature(enable = "avx2")]
unsafe fn dot32x4_avx(q: [&[f32]; 4], b: &[f32]) -> [f32; 4] {
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    assert!(q.iter().all(|x| x.len() == b.len()) && b.len().is_multiple_of(8));
    let mut acc: [__m256; 4] = [_mm256_setzero_ps(); 4];
    for c in 0..b.len() / 8 {
        // SAFETY: `c * 8 + 8 <= b.len() == q[t].len()` by the assert above.
        let kb = unsafe { _mm256_loadu_ps(b.as_ptr().add(c * 8)) };
        for t in 0..4 {
            let qa = unsafe { _mm256_loadu_ps(q[t].as_ptr().add(c * 8)) };
            acc[t] = _mm256_add_ps(acc[t], _mm256_mul_ps(qa, kb));
        }
    }
    let mut out = [0f32; 4];
    for t in 0..4 {
        let mut lanes = [0f32; 8];
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc[t]) };
        out[t] = lanes.iter().sum();
    }
    out
}

#[inline(always)]
fn gemm_tile(ych: &mut [f32], ld: usize, t: usize, o0: usize, w: &[f32], n_in: usize, x: &[f32]) {
    let mut tb = 0;
    while tb < t {
        let te = (tb + 128).min(t);
        for (ri, yrow) in ych.chunks_mut(ld).enumerate() {
            let o = o0 + ri;
            let wr = &w[o * n_in..(o + 1) * n_in];
            for tt in tb..te {
                yrow[tt] = dot32(wr, &x[tt * n_in..(tt + 1) * n_in]);
            }
        }
        tb = te;
    }
}

/// `gemm_tile` compiled for AVX2+FMA; the arithmetic is the same (no contraction of mul and add).
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_tile_avx(
    ych: &mut [f32],
    ld: usize,
    t: usize,
    o0: usize,
    w: &[f32],
    n_in: usize,
    x: &[f32],
) {
    gemm_tile(ych, ld, t, o0, w, n_in, x);
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
    // row stride `ld` is not a power of two: a stride of 2^k floats makes every row of a tile map to
    // the same cache sets
    let ld = t + 17;
    let mut yt = vec![0f32; n_out * ld];
    let rows = n_out.div_ceil(nth);
    let avx = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    std::thread::scope(|s| {
        for (ci, ych) in yt.chunks_mut(rows * ld).enumerate() {
            s.spawn(move || {
                if avx {
                    // SAFETY: avx2 and fma were detected above.
                    unsafe { gemm_tile_avx(ych, ld, t, rows * ci, w, n_in, x) }
                } else {
                    gemm_tile(ych, ld, t, rows * ci, w, n_in, x);
                }
            });
        }
    });
    let mut y = vec![0f32; n_out * t];
    let per = t.div_ceil(nth);
    std::thread::scope(|s| {
        for (ci, ych) in y.chunks_mut(per * n_out).enumerate() {
            let yt = &yt;
            s.spawn(move || {
                for (ri, yrow) in ych.chunks_mut(n_out).enumerate() {
                    let tt = ci * per + ri;
                    for o in 0..n_out {
                        yrow[o] = yt[o * ld + tt] + bias.map_or(0.0, |b| b[o]);
                    }
                }
            });
        }
    });
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
    let (lc, lb, lw) = lemma_selftest(100_000);
    println!(
        "lemma self-test (moment_remainder on f64): {lc} node-moment cases, {lb} violations, worst realised/bound {lw:.4}"
    );
    println!(
        "model {path}\nlayers {nl} heads {nh} kv {nkv} hd {HD} n_embd {d}; prompts {np} x {T} tokens"
    );
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
    let qk_energy = std::env::var_os("QK_ENERGY").is_some();
    let mut qke: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = Vec::new();
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
        // QK_DUMP=<dir>: prompt 0's unrotated q and k per layer, f32 LE, for the
        // faces-review slope decomposition (rope applied offline from cs).
        if let Some(dir) = std::env::var_os("QK_DUMP") {
            let dir = std::path::PathBuf::from(dir);
            let le = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
            std::fs::write(dir.join(format!("q{l}.bin")), le(&q[..T * nh * HD]))?;
            std::fs::write(dir.join(format!("k{l}.bin")), le(&k[..T * nkv * HD]))?;
            if l == 0 {
                let c: Vec<u8> = cs
                    .iter()
                    .flat_map(|&(c, s)| [c, s])
                    .flat_map(f64::to_le_bytes)
                    .collect();
                std::fs::write(dir.join("cs.bin"), c)?;
            }
        }
        for pp in 0..np * T {
            let pos = pp % T;
            rot(&mut q[pp * nh * HD..(pp + 1) * nh * HD], nh, pos);
            rot(&mut k[pp * nkv * HD..(pp + 1) * nkv * HD], nkv, pos);
        }
        if qk_energy {
            qk_energy_layer(l, &q[..T * nh * HD], &k[..T * nkv * HD], nh, nkv, &mut qke);
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

    if qk_energy {
        qk_energy_report(&qke, nh);
        return Ok(true);
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
    println!(
        "cert = smallest |K| with 2|D|e^-delta <= eps (delta = gap of the first dropped key); oracle = smallest prefix of the score order with realised error <= eps*V (not certifiable)"
    );
    for (ei, eps) in EPS.iter().enumerate() {
        println!(
            "-- eps {eps:e}   (mean |K|/t [min,max over pairs]; share of queries with cert |K|/t < 0.25)"
        );
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
    println!(
        "\n== block-box certificate (no key read for a pruned block): block {BLK}, read key 0 + last 64 keys, bound the rest by per-dimension min/max boxes =="
    );
    println!(
        "V = value-side |K'|/t; K = key-side reads incl. 2 box vectors per checked block, in key-vector units / t"
    );
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
    drop(x);
    let b4 = run_b4(&eng, &bpe, &ps[0], &kcap, &vcap)?;
    Ok(tot.holds == tot.checks && lb == 0 && b4)
}

// ---- B4: moment nodes carrying the diffuse mass, 16k context ------------------------------
//
// Lemma refined (grade `tested`, f64 over f32 data): `Apeiron.Numeric.AttentionCut.moment_remainder`
// (read-only source `wt-ringreturn/proof/Apeiron/Numeric/AttentionCut.lean` l.312): for a node S
// with `|s_i - c| <= B <= 1` and `p >= 1`,
// `|num S - e^c sum_{r<p} (sum_i (s_i-c)^r v_i)/r!| <= e^c (sum_i |v_i|) B^p (p+1)/(p! p)`.
// `moment_one_stored` / `moment_two_stored` are the identities that let a node store the first and
// second moments once; here the moments are summed directly, the stored-table cost is counted.

const T16: usize = 8_192;
const QT: [usize; 4] = [512, 2048, 4096, 8192];
const NQ: usize = 4;
const LS: [usize; 4] = [8, 16, 32, 64];
const NG: usize = 12;
const GKIND: [&str; 3] = ["pos", "pca", "orc"];
const PS: [usize; 3] = [1, 2, 3];
const KEYF: f64 = 128.0;
const NBIN: usize = 400;
const NTH4: usize = 24;
const DV: usize = HD;

/// Lemma constant `(p+1)/(p! p)`.
fn rp(p: usize) -> f64 {
    let f = [1.0, 1.0, 2.0, 6.0][p];
    (p as f64 + 1.0) / (f * p as f64)
}

/// Floats a node stores for order `p`: denominator moments `sum_{r<p} d^r`, numerator moments
/// `dv sum_{r<p} d^r`, and the centre key (`d`).
fn node_cost(p: usize) -> f64 {
    let (mut mom, mut dp) = (0.0, 1.0);
    for _ in 0..p {
        mom += dp;
        dp *= HD as f64;
    }
    (1.0 + DV as f64) * mom + HD as f64
}

/// `R_p^{-1}(eps)`: the owner's simple threshold on the projected spread.
fn bmax(eps: f64, p: usize) -> f64 {
    (eps / rp(p)).powf(1.0 / p as f64)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The lemma on `f64`: `p`-term node value plus remainder bound dominates the realised node error,
/// for `count` random nodes of 8..64 keys (three score/value families, including the adversarial
/// two-point family with the values on the high-score keys). Returns (cases, violations, worst
/// realised/bound).
fn lemma_selftest(count: usize) -> (usize, usize, f64) {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut cases, mut bad, mut worst) = (0usize, 0usize, 0f64);
    for _ in 0..count {
        let nsz = 8 + (rng.next() * 57.0) as usize;
        let b0 = 10f64.powf(-3.0 * rng.next());
        let kind = (rng.next() * 3.0) as usize;
        let c0 = 4.0 * rng.next() - 2.0;
        let mut s = vec![0f64; nsz];
        let mut v = vec![[0f64; 5]; nsz];
        let f = 0.1 + 0.8 * rng.next();
        for i in 0..nsz {
            let u = rng.next();
            s[i] = match kind {
                0 => c0 + b0 * (2.0 * u - 1.0),
                1 => c0 + if u < f { b0 } else { -b0 },
                _ => c0 + b0 * u * u,
            };
            for j in 0..4 {
                let r = rng.next();
                v[i][j] = match kind {
                    0 => 2.0 * r - 1.0,
                    1 => {
                        if s[i] > c0 {
                            1.0 + r
                        } else {
                            0.01 * r
                        }
                    }
                    _ => (2.0 * r - 1.0) * u,
                };
            }
            v[i][4] = 1.0;
        }
        let mut c = s.iter().sum::<f64>() / nsz as f64;
        let mut bb = s.iter().map(|x| (x - c).abs()).fold(0.0, f64::max);
        if bb > 0.999_999 {
            let k = 0.999_999 / bb;
            for x in &mut s {
                *x = c + (*x - c) * k;
            }
            c = s.iter().sum::<f64>() / nsz as f64;
            bb = s.iter().map(|x| (x - c).abs()).fold(0.0, f64::max);
        }
        let m = c + 6.0 * rng.next();
        let ec = (c - m).exp();
        for p in PS {
            for j in 0..5 {
                let num: f64 = (0..nsz).map(|i| (s[i] - m).exp() * v[i][j]).sum();
                let mut tay = 0f64;
                let mut fact = 1.0;
                for r in 0..p {
                    if r > 0 {
                        fact *= r as f64;
                    }
                    let mom: f64 = (0..nsz).map(|i| (s[i] - c).powi(r as i32) * v[i][j]).sum();
                    tay += mom / fact;
                }
                let real = (num - ec * tay).abs();
                let bound =
                    ec * (0..nsz).map(|i| v[i][j].abs()).sum::<f64>() * bb.powi(p as i32) * rp(p);
                // f64 summation rounding of `num`: a bound below it is not resolvable
                let round = 1e-14
                    * (0..nsz)
                        .map(|i| (s[i] - m).exp() * v[i][j].abs())
                        .sum::<f64>();
                let (real, bound) = (real, bound + round);
                cases += 1;
                if bound > 0.0 {
                    worst = worst.max(real / bound);
                }
                if real > bound {
                    bad += 1;
                }
            }
        }
    }
    (cases, bad, worst)
}

/// Disjoint key groups: node `k` is `idx[off[k]..off[k+1]]`.
#[derive(Clone, Default)]
struct Grp {
    off: Vec<usize>,
    idx: Vec<u32>,
}

impl Grp {
    fn nodes(&self) -> usize {
        self.off.len().saturating_sub(1)
    }
    fn from_ranges(arr: &[u32], ranges: &[(usize, usize)]) -> Grp {
        let mut g = Grp {
            off: vec![0],
            idx: Vec::new(),
        };
        for &(lo, hi) in ranges {
            g.idx.extend_from_slice(&arr[lo..hi]);
            g.off.push(g.idx.len());
        }
        g
    }
}

/// Positional blocks of `L` over `[0, region)`; a short tail is left out (read exactly).
fn pos_groups(region: usize) -> Vec<Grp> {
    let arr: Vec<u32> = (0..region as u32).collect();
    LS.iter()
        .map(|&l| {
            let r: Vec<(usize, usize)> = (0..region / l).map(|k| (k * l, (k + 1) * l)).collect();
            Grp::from_ranges(&arr, &r)
        })
        .collect()
}

/// Top principal direction of the keys `arr` by 6 power iterations on the centred covariance.
fn top_pc(arr: &[u32], keys: &[f64]) -> [f64; HD] {
    let n = arr.len() as f64;
    let mut mu = [0f64; HD];
    for &i in arr {
        let r = &keys[i as usize * HD..(i as usize + 1) * HD];
        for d in 0..HD {
            mu[d] += r[d] / n;
        }
    }
    let mut v = [0f64; HD];
    for d in 0..HD {
        v[d] = ((d + 1) as f64).sin();
    }
    for _ in 0..6 {
        let mut w = [0f64; HD];
        for &i in arr {
            let r = &keys[i as usize * HD..(i as usize + 1) * HD];
            let mut a = 0f64;
            for d in 0..HD {
                a += (r[d] - mu[d]) * v[d];
            }
            for d in 0..HD {
                w[d] += a * (r[d] - mu[d]);
            }
        }
        let nr = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nr == 0.0 {
            return [0.0; HD];
        }
        for d in 0..HD {
            v[d] = w[d] / nr;
        }
    }
    v
}

/// Static PCA tree over the keys `[0, region)` of one (layer, kv head): a cell of `n >= 2L` keys
/// splits at the median of its projection on the top principal direction, so leaves hold `L..2L-1`
/// keys. One recursion yields the four leaf sets (same splits, different stop size).
///
/// Complexity: time `O(region * HD * 14 * log2(region/8))` multiply-adds, space `O(region)`.
fn pca_groups(keys: &[f64], region: usize) -> Vec<Grp> {
    let mut arr: Vec<u32> = (0..region as u32).collect();
    let mut leaves: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 4];
    let mut stack = vec![(0usize, region, 0b1111u8)];
    while let Some((lo, hi, mask)) = stack.pop() {
        let n = hi - lo;
        let mut mk = mask;
        for li in 0..4 {
            if (mk >> li) & 1 == 1 && n < 2 * LS[li] {
                leaves[li].push((lo, hi));
                mk &= !(1 << li);
            }
        }
        if mk == 0 {
            continue;
        }
        let dir = top_pc(&arr[lo..hi], keys);
        let mut pr: Vec<(f64, u32)> = arr[lo..hi]
            .iter()
            .map(|&i| {
                let r = &keys[i as usize * HD..(i as usize + 1) * HD];
                (r.iter().zip(&dir).map(|(a, b)| a * b).sum::<f64>(), i)
            })
            .collect();
        let mid = n / 2;
        pr.select_nth_unstable_by(mid, |a, b| a.0.total_cmp(&b.0));
        for (t, (_, i)) in pr.iter().enumerate() {
            arr[lo + t] = *i;
        }
        stack.push((lo, lo + mid, mk));
        stack.push((lo + mid, hi, mk));
    }
    leaves
        .iter_mut()
        .map(|r| {
            r.sort_unstable();
            Grp::from_ranges(&arr, r)
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Cell {
    n: f64,
    rsum: f64,
    rmin: f64,
    rmax: f64,
    abs: f64,
    nk: f64,
    used: f64,
}
impl Default for Cell {
    fn default() -> Self {
        Cell {
            n: 0.0,
            rsum: 0.0,
            rmin: f64::INFINITY,
            rmax: f64::NEG_INFINITY,
            abs: 0.0,
            nk: 0.0,
            used: 0.0,
        }
    }
}
impl Cell {
    fn add(&mut self, r: f64, abs: f64, nk: f64, used: f64) {
        self.n += 1.0;
        self.rsum += r;
        self.rmin = self.rmin.min(r);
        self.rmax = self.rmax.max(r);
        self.abs += abs;
        self.nk += nk;
        self.used += used;
    }
    fn merge(&mut self, o: &Cell) {
        self.n += o.n;
        self.rsum += o.rsum;
        self.rmin = self.rmin.min(o.rmin);
        self.rmax = self.rmax.max(o.rmax);
        self.abs += o.abs;
        self.nk += o.nk;
        self.used += o.used;
    }
    fn fmt(&self) -> String {
        if self.n == 0.0 {
            return "-".into();
        }
        format!(
            "{:.3} [{:.3},{:.3}]",
            self.rsum / self.n,
            self.rmin,
            self.rmax
        )
    }
}

fn ci(ti: usize, g: usize, mode: usize, ei: usize, pi: usize) -> usize {
    (((ti * NG + g) * 2 + mode) * 3 + ei) * 3 + pi
}

#[derive(Clone)]
struct Acc4 {
    cell: Vec<Cell>,
    cut: Vec<Cell>,
    hist: Vec<f64>,
    bcnt: Vec<f64>,
    checks: f64,
    holds: f64,
    worst: f64,
    errs: f64,
    first_violation: Option<String>,
}

impl Acc4 {
    fn new() -> Self {
        Acc4 {
            cell: vec![Cell::default(); 5 * NG * 2 * 3 * 3],
            cut: vec![Cell::default(); 5 * 3 * 2],
            hist: vec![0.0; 5 * NG * NBIN],
            bcnt: vec![0.0; 5 * NG * 11],
            checks: 0.0,
            holds: 0.0,
            worst: 0.0,
            errs: 0.0,
            first_violation: None,
        }
    }
    fn check(&mut self, real: f64, bound: f64, what: impl FnOnce() -> String) {
        self.checks += 1.0;
        if real <= bound {
            self.holds += 1.0;
        } else if self.first_violation.is_none() {
            self.first_violation = Some(what());
        }
        if bound > 0.0 {
            self.worst = self.worst.max(real / bound);
        }
    }
    fn merge(&mut self, o: &Acc4) {
        for (a, b) in self.cell.iter_mut().zip(&o.cell) {
            a.merge(b);
        }
        for (a, b) in self.cut.iter_mut().zip(&o.cut) {
            a.merge(b);
        }
        for (a, b) in self.hist.iter_mut().zip(&o.hist) {
            *a += b;
        }
        for (a, b) in self.bcnt.iter_mut().zip(&o.bcnt) {
            *a += b;
        }
        self.checks += o.checks;
        self.holds += o.holds;
        self.worst = self.worst.max(o.worst);
        self.errs += o.errs;
        if self.first_violation.is_none() {
            self.first_violation.clone_from(&o.first_violation);
        }
    }
}

/// Per-node quantities of one query: centre `c`, spread `B`, moments in frame `m`.
struct NodeSt {
    size: Vec<usize>,
    b: Vec<f64>,
    ec: Vec<f64>,
    z1: Vec<f64>,
    z2: Vec<f64>,
    amax: Vec<f64>,
    a0: Vec<f64>,
    a1: Vec<f64>,
    a2: Vec<f64>,
}

/// Complexity: time `O(|keys in grp| * DV)` multiply-adds (four DV-wide accumulations per key),
/// space `O(nodes * DV)` words.
fn node_stats(grp: &Grp, s: &[f64], vv: &[f64], m: f64) -> NodeSt {
    let nn = grp.nodes();
    let mut st = NodeSt {
        size: Vec::with_capacity(nn),
        b: Vec::with_capacity(nn),
        ec: Vec::with_capacity(nn),
        z1: Vec::with_capacity(nn),
        z2: Vec::with_capacity(nn),
        amax: Vec::with_capacity(nn),
        a0: vec![0.0; nn * DV],
        a1: vec![0.0; nn * DV],
        a2: vec![0.0; nn * DV],
    };
    for k in 0..nn {
        let mem = &grp.idx[grp.off[k]..grp.off[k + 1]];
        let sz = mem.len();
        let c = mem.iter().map(|&i| s[i as usize]).sum::<f64>() / sz as f64;
        let (mut bb, mut z1, mut z2) = (0f64, 0f64, 0f64);
        let mut ab = [0f64; DV];
        let (a0, a1, a2) = (
            &mut st.a0[k * DV..(k + 1) * DV],
            &mut st.a1[k * DV..(k + 1) * DV],
            &mut st.a2[k * DV..(k + 1) * DV],
        );
        for &i in mem {
            let i = i as usize;
            let ds = s[i] - c;
            let d2 = ds * ds;
            bb = bb.max(ds.abs());
            z1 += ds;
            z2 += d2;
            let vr = &vv[i * DV..(i + 1) * DV];
            for j in 0..DV {
                a0[j] += vr[j];
                a1[j] += ds * vr[j];
                a2[j] += d2 * vr[j];
                ab[j] += vr[j].abs();
            }
        }
        st.size.push(sz);
        st.b.push(bb);
        st.ec.push((c - m).exp());
        st.z1.push(z1);
        st.z2.push(z2);
        st.amax.push(ab.iter().copied().fold(0.0, f64::max));
    }
    st
}

/// Order-`p` node errors and stand-in denominators, nodes with `B <= 1` sorted by `eps_Z`.
struct PNode {
    ez: Vec<f64>,
    en: Vec<f64>,
    zt: Vec<f64>,
    ord: Vec<u32>,
}

fn pnode(st: &NodeSt, p: usize) -> PNode {
    let nn = st.b.len();
    let (mut ez, mut en, mut zt) = (vec![0.0; nn], vec![0.0; nn], vec![0.0; nn]);
    let mut ord: Vec<u32> = Vec::new();
    for k in 0..nn {
        let bp = st.b[k].powi(p as i32);
        ez[k] = st.ec[k] * st.size[k] as f64 * rp(p) * bp;
        en[k] = st.ec[k] * st.amax[k] * rp(p) * bp;
        let mut z = st.size[k] as f64;
        if p >= 2 {
            z += st.z1[k];
        }
        if p >= 3 {
            z += st.z2[k] / 2.0;
        }
        zt[k] = st.ec[k] * z;
        if st.b[k] <= 1.0 && st.size[k] >= 8 {
            ord.push(k as u32);
        }
    }
    ord.sort_unstable_by(|&a, &b| ez[a as usize].total_cmp(&ez[b as usize]));
    PNode { ez, en, zt, ord }
}

struct Tot {
    z: f64,
    k: usize,
    dcnt: usize,
    nv: [f64; DV],
}

struct Cn {
    ze: Vec<f64>,
    ke: Vec<u32>,
    kd: Vec<u32>,
}

/// Greedy node selection, smallest `eps_Z` first: a node is summarised iff its nondropped keys
/// outnumber its cost in key-equivalents and the total certified bound stays `<= tol`.
///
/// Complexity: time `O(|eligible nodes|)` per call, space `O(1)` beyond the returned list.
fn greedy(
    pn: &PNode,
    cn: &Cn,
    tot: &Tot,
    delta: f64,
    v: f64,
    tol: f64,
    cost_keys: f64,
) -> Vec<u32> {
    let ed = (-delta).exp();
    let (mut zp, mut dc, mut sen, mut sez) = (tot.z, tot.dcnt as f64, 0f64, 0f64);
    let mut sel = Vec::new();
    for &k in &pn.ord {
        let k = k as usize;
        if f64::from(cn.ke[k]) <= cost_keys {
            continue;
        }
        let zp2 = zp - cn.ze[k] + pn.zt[k];
        if zp2 <= 0.0 {
            continue;
        }
        let dc2 = dc - f64::from(cn.kd[k]);
        let (sen2, sez2) = (sen + pn.en[k], sez + pn.ez[k]);
        let b = (sen2 + v * sez2 + 2.0 * v * dc2 * ed) / zp2;
        if b <= tol {
            zp = zp2;
            dc = dc2;
            sen = sen2;
            sez = sez2;
            sel.push(k as u32);
        }
    }
    sel
}

#[derive(Clone, Copy)]
struct Best {
    read: f64,
    nsel: usize,
    inn: usize,
    kexf: usize,
}

struct LayerCtx<'a> {
    kk: &'a [f64],
    vv: &'a [f64],
    vpm: &'a [f64],
    posg: &'a [Grp],
    pca: &'a [Grp],
}

/// One query (position `pos`, causal keys `0..=pos`) against every grouping, delta, tolerance,
/// order and cost mode.
///
/// Per (grouping, delta, eps, p, mode) the greedy selection of summarised nodes is built, its parts
/// are merged through `apeiron::attn_cut::certified_read`, and the realised f64 error over all keys
/// is checked against the bound. Dropped keys cost nothing to identify (as in the earlier
/// sections); a summarised node costs `node_cost(p)/128` key-equivalents (mode 0) or one
/// key-equivalent (mode 1, spread-only reference).
///
/// Complexity: time `O(n * HD)` for scores and the full read, `O(n * HD)` per grouping for the node
/// moments (12 groupings), `O(n)` per (grouping, delta) for the counts, and `O(|eligible nodes|)`
/// per greedy call (at most 12 * 9 * 3 * 3 * 2 calls); space `O(n + nodes * HD)` words.
#[allow(clippy::too_many_lines)]
fn analyze_query(cx: &LayerCtx, qv: &[f64], pos: usize, ti: usize, acc: &mut Acc4) {
    let n = pos + 1;
    let scale = 1.0 / (HD as f64).sqrt();
    let (kk, vv) = (cx.kk, cx.vv);
    let mut s = vec![0f64; n];
    let mut m = f64::NEG_INFINITY;
    for i in 0..n {
        let kr = &kk[i * HD..(i + 1) * HD];
        let mut a = 0f64;
        for d in 0..HD {
            a += qv[d] * kr[d];
        }
        s[i] = scale * a;
        m = m.max(s[i]);
    }
    let v = cx.vpm[pos];
    if v <= 0.0 {
        return;
    }
    let (mut gp, mut w) = (vec![0f64; n], vec![0f64; n]);
    let mut fnum = [0f64; DV];
    let mut zf = 0f64;
    for i in 0..n {
        gp[i] = m - s[i];
        w[i] = (-gp[i]).exp();
        zf += w[i];
        for j in 0..DV {
            fnum[j] += w[i] * vv[i * DV + j];
        }
    }
    let mut full = [0f64; DV];
    for j in 0..DV {
        full[j] = fnum[j] / zf;
    }
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by(|&a, &b| gp[a as usize].total_cmp(&gp[b as usize]));
    // delta candidates: the grid, plus per eps the gap rule of the earlier section
    let mut dl: Vec<f64> = DELTAS.to_vec();
    let mut cand: Vec<Vec<usize>> = vec![(0..DELTAS.len()).collect(); 3];
    for (ei, &eps) in EPS.iter().enumerate() {
        let tol = eps * v;
        for c in 1..n {
            let (gc, gq) = (gp[order[c] as usize], gp[order[c - 1] as usize]);
            if gc > gq && 2.0 * v * (n - c) as f64 * (-gc).exp() <= tol {
                dl.push(gc);
                cand[ei].push(dl.len() - 1);
                break;
            }
        }
    }
    let nd = dl.len();
    let ed: Vec<f64> = dl.iter().map(|d| (-d).exp()).collect();
    let tots: Vec<Tot> = dl
        .iter()
        .map(|&dd| {
            let mut t = Tot {
                z: 0.0,
                k: 0,
                dcnt: 0,
                nv: [0.0; DV],
            };
            for i in 0..n {
                if gp[i] < dd {
                    t.z += w[i];
                    t.k += 1;
                    for j in 0..DV {
                        t.nv[j] += w[i] * vv[i * DV + j];
                    }
                } else {
                    t.dcnt += 1;
                }
            }
            t
        })
        .collect();
    // cut only (no nodes): realised vs bound for every delta, then the per-eps best
    let cutb: Vec<f64> = (0..nd)
        .map(|di| {
            let t = &tots[di];
            if t.dcnt == 0 {
                0.0
            } else {
                2.0 * v * t.dcnt as f64 * ed[di] / t.z
            }
        })
        .collect();
    for di in 0..nd {
        let t = &tots[di];
        if t.dcnt > 0 {
            let mut real = 0f64;
            for j in 0..DV {
                real = real.max((t.nv[j] / t.z - full[j]).abs());
            }
            let (dd, bound) = (dl[di], cutb[di]);
            acc.check(real, bound, || {
                format!(
                    "B4 cut-only delta={dd} n={n} real={real:e} bound={bound:e} |D|={}",
                    t.dcnt
                )
            });
        }
    }
    let mut best0 = [n as f64; 3];
    let mut best0g = [n as f64; 3];
    for (ei, &eps) in EPS.iter().enumerate() {
        for &di in &cand[ei] {
            if cutb[di] <= eps * v {
                best0[ei] = best0[ei].min(tots[di].k as f64);
                if di < DELTAS.len() {
                    best0g[ei] = best0g[ei].min(tots[di].k as f64);
                }
            }
        }
        acc.cut[(ti * 3 + ei) * 2].add(best0[ei] / n as f64, best0[ei], 0.0, 0.0);
        acc.cut[(ti * 3 + ei) * 2 + 1].add(best0g[ei] / n as f64, best0g[ei], 0.0, 0.0);
    }
    let kexf0: Vec<usize> = best0.iter().map(|&x| x as usize).collect();
    let thr: Vec<f64> = (0..9).map(|x| bmax(EPS[x % 3], PS[x / 3])).collect();
    for g in 0..NG {
        let (kind, li) = (g / 4, g % 4);
        let owned;
        let grp: &Grp = match kind {
            0 => &cx.posg[li],
            1 => &cx.pca[li],
            _ => {
                let nb = n / LS[li];
                owned = Grp {
                    off: (0..=nb).map(|k| k * LS[li]).collect(),
                    idx: order[..nb * LS[li]].to_vec(),
                };
                &owned
            }
        };
        let nn = grp.nodes();
        let st = node_stats(grp, &s, vv, m);
        for k in 0..nn {
            let b = st.b[k];
            let bin = if b <= 0.0 {
                0
            } else {
                (((b.log10() + 4.0) / 0.02).floor().max(0.0) as usize).min(NBIN - 1)
            };
            acc.hist[(ti * NG + g) * NBIN + bin] += 1.0;
            let r = (ti * NG + g) * 11;
            acc.bcnt[r] += 1.0;
            if b <= 1.0 {
                acc.bcnt[r + 1] += 1.0;
            }
            for x in 0..9 {
                if b <= thr[x] {
                    acc.bcnt[r + 2 + x] += 1.0;
                }
            }
        }
        let pns: Vec<PNode> = PS.iter().map(|&p| pnode(&st, p)).collect();
        let mut best = vec![
            Best {
                read: 0.0,
                nsel: 0,
                inn: 0,
                kexf: 0
            };
            2 * 3 * 3
        ];
        for mode in 0..2 {
            for ei in 0..3 {
                for pi in 0..3 {
                    best[(mode * 3 + ei) * 3 + pi] = Best {
                        read: best0[ei],
                        nsel: 0,
                        inn: 0,
                        kexf: kexf0[ei],
                    };
                }
            }
        }
        let mut ne = vec![0f64; nn * DV];
        let mut done = vec![false; nn];
        for di in 0..nd {
            let users: Vec<usize> = (0..3).filter(|&ei| cand[ei].contains(&di)).collect();
            if users.is_empty() {
                continue;
            }
            let delta = dl[di];
            let tot = &tots[di];
            let mut cn = Cn {
                ze: vec![0.0; nn],
                ke: vec![0; nn],
                kd: vec![0; nn],
            };
            for k in 0..nn {
                for &i in &grp.idx[grp.off[k]..grp.off[k + 1]] {
                    let i = i as usize;
                    if gp[i] < delta {
                        cn.ze[k] += w[i];
                        cn.ke[k] += 1;
                    } else {
                        cn.kd[k] += 1;
                    }
                }
            }
            done.fill(false);
            for &ei in &users {
                let tol = EPS[ei] * v;
                for pi in 0..3 {
                    let p = PS[pi];
                    let pn = &pns[pi];
                    for mode in 0..2 {
                        let cost_keys = if mode == 0 { node_cost(p) / KEYF } else { 1.0 };
                        let sel = greedy(pn, &cn, tot, delta, v, tol, cost_keys);
                        if sel.is_empty() {
                            continue;
                        }
                        let (mut nsum, mut ne_sum) = ([0f64; DV], [0f64; DV]);
                        let (mut zts, mut sen, mut sez, mut ze_sum) = (0f64, 0f64, 0f64, 0f64);
                        let (mut ke_sum, mut kd_sum, mut inn) = (0usize, 0usize, 0usize);
                        for &k in &sel {
                            let k = k as usize;
                            if !done[k] {
                                let row = &mut ne[k * DV..(k + 1) * DV];
                                row.fill(0.0);
                                for &i in &grp.idx[grp.off[k]..grp.off[k + 1]] {
                                    let i = i as usize;
                                    if gp[i] < delta {
                                        for j in 0..DV {
                                            row[j] += w[i] * vv[i * DV + j];
                                        }
                                    }
                                }
                                done[k] = true;
                            }
                            for j in 0..DV {
                                ne_sum[j] += ne[k * DV + j];
                                let mut t = st.a0[k * DV + j];
                                if p >= 2 {
                                    t += st.a1[k * DV + j];
                                }
                                if p >= 3 {
                                    t += st.a2[k * DV + j] / 2.0;
                                }
                                nsum[j] += st.ec[k] * t;
                            }
                            zts += pn.zt[k];
                            sen += pn.en[k];
                            sez += pn.ez[k];
                            ze_sum += cn.ze[k];
                            ke_sum += cn.ke[k] as usize;
                            kd_sum += cn.kd[k] as usize;
                            inn += st.size[k];
                        }
                        let exact = Part {
                            n: (0..DV).map(|j| tot.nv[j] - ne_sum[j]).collect(),
                            z: tot.z - ze_sum,
                            en: 0.0,
                            ez: 0.0,
                            m,
                        };
                        let nodep = Part {
                            n: nsum.to_vec(),
                            z: zts,
                            en: sen,
                            ez: sez,
                            m,
                        };
                        let dropp = Part::dropped(DV, tot.dcnt - kd_sum, delta, v, m);
                        let Ok((out, bound)) = certified_read(&[exact, nodep, dropp], v) else {
                            acc.errs += 1.0;
                            continue;
                        };
                        let mut real = 0f64;
                        for j in 0..DV {
                            real = real.max((out[j] - full[j]).abs());
                        }
                        acc.check(real, bound, || {
                            format!(
                                "B4 {}{} p={p} mode={mode} eps={:e} delta={delta} t={} n={n} nsel={} real={real:e} bound={bound:e}",
                                GKIND[kind], LS[li], EPS[ei], QT[ti], sel.len()
                            )
                        });
                        if bound <= tol {
                            let kexf = tot.k - ke_sum;
                            let read = kexf as f64 + sel.len() as f64 * cost_keys;
                            let b = &mut best[(mode * 3 + ei) * 3 + pi];
                            if read < b.read {
                                *b = Best {
                                    read,
                                    nsel: sel.len(),
                                    inn,
                                    kexf,
                                };
                            }
                        }
                    }
                }
            }
        }
        for mode in 0..2 {
            for ei in 0..3 {
                for pi in 0..3 {
                    let b = best[(mode * 3 + ei) * 3 + pi];
                    let nonkept = (n - b.kexf) as f64;
                    let nk = if nonkept > 0.0 {
                        b.inn as f64 / nonkept
                    } else {
                        0.0
                    };
                    acc.cell[ci(ti, g, mode, ei, pi)].add(
                        b.read / n as f64,
                        b.read,
                        nk,
                        f64::from(u8::from(b.nsel > 0)),
                    );
                }
            }
        }
    }
}

const ACH: usize = 64;
const AKB: usize = 256;

/// One job of `attend_all`: queries `ch*64..ch*64+64` of head `h`.
#[inline(always)]
fn attend_job(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    (nh, nkv): (usize, usize),
    (ch, h): (usize, usize),
    scr: &mut [f32],
) -> Vec<f32> {
    let kvh = h / (nh / nkv);
    let q0 = ch * ACH;
    let nk = q0 + ACH;
    let ld = nk + 17;
    let krow = |i: usize| &k[(i * nkv + kvh) * HD..(i * nkv + kvh) * HD + HD];
    let vrow = |i: usize| &v[(i * nkv + kvh) * HD..(i * nkv + kvh) * HD + HD];
    for kb in (0..nk).step_by(AKB) {
        let kend = (kb + AKB).min(nk);
        for qi in 0..ACH {
            let lim = kend.min(q0 + qi + 1);
            if kb >= lim {
                continue;
            }
            let pos = q0 + qi;
            let qh = &q[(pos * nh + h) * HD..(pos * nh + h) * HD + HD];
            for i in kb..lim {
                scr[qi * ld + i] = dot32(qh, krow(i)) * SCALE;
            }
        }
    }
    let mut mx = [f32::NEG_INFINITY; ACH];
    for qi in 0..ACH {
        for i in 0..=q0 + qi {
            mx[qi] = mx[qi].max(scr[qi * ld + i]);
        }
    }
    let mut accv = vec![0f32; ACH * HD];
    let mut z = [0f64; ACH];
    for kb in (0..nk).step_by(AKB) {
        let kend = (kb + AKB).min(nk);
        for qi in 0..ACH {
            let lim = kend.min(q0 + qi + 1);
            let ac = &mut accv[qi * HD..(qi + 1) * HD];
            for i in kb..lim {
                let gap = scr[qi * ld + i] - mx[qi];
                if gap < -30.0 {
                    continue;
                }
                let wg = gap.exp();
                z[qi] += f64::from(wg);
                let vr = vrow(i);
                for d in 0..HD {
                    ac[d] += wg * vr[d];
                }
            }
        }
    }
    let mut out = vec![0f32; ACH * HD];
    for qi in 0..ACH {
        for d in 0..HD {
            out[qi * HD + d] = (f64::from(accv[qi * HD + d]) / z[qi]) as f32;
        }
    }
    out
}

/// `attend_job` compiled for AVX2+FMA (same arithmetic: no mul/add contraction).
#[target_feature(enable = "avx2,fma")]
unsafe fn attend_job_avx(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dims: (usize, usize),
    job: (usize, usize),
    scr: &mut [f32],
) -> Vec<f32> {
    attend_job(q, k, v, dims, job, scr)
}

/// Causal attention for all `t` positions of one layer (f32 scores and weights, f64 normaliser),
/// the forward pass of the 16k arm. A job is 64 consecutive queries of one head; keys are visited
/// in blocks of 256 so a K or V block is read once per 64 queries instead of once per query. Keys
/// whose score is more than 30 below the row maximum (weight < 1e-13) are skipped.
///
/// Only the query chunks `chs` (units of 64 rows) are computed; row `i` of the output is query
/// `chs.start * 64 + i`. Each job is the same computation whatever `chs` is, so a range split is exact.
///
/// Complexity: time `O(nh * t^2/2 * HD)` multiply-adds for scores and again for values over the full
/// range (a chunk ending at row `r` costs its rows times `r`), space `O(rows * nh * HD)` words for the
/// output plus `64 * t` words of scores per thread.
#[allow(clippy::too_many_arguments)]
fn attend_all(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    nh: usize,
    nkv: usize,
    nth: usize,
    chs: std::ops::Range<usize>,
) -> Vec<f32> {
    let nch = chs.len();
    let next = AtomicUsize::new(0);
    let avx = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    let results: Mutex<Vec<(usize, usize, Vec<f32>)>> = Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for _ in 0..nth {
            sc.spawn(|| {
                let mut scr = vec![0f32; ACH * (t + 17)];
                loop {
                    let job = next.fetch_add(1, Ordering::Relaxed);
                    if job >= nch * nh {
                        break;
                    }
                    let (ch, h) = (chs.end - 1 - job / nh, job % nh);
                    let out = if avx {
                        // SAFETY: avx2 and fma were detected above.
                        unsafe { attend_job_avx(q, k, v, (nh, nkv), (ch, h), &mut scr) }
                    } else {
                        attend_job(q, k, v, (nh, nkv), (ch, h), &mut scr)
                    };
                    results.lock().unwrap().push((h, ch, out));
                }
            });
        }
    });
    let mut att = vec![0f32; nch * ACH * nh * HD];
    for (h, ch, out) in results.into_inner().unwrap() {
        for qi in 0..ACH {
            let dst = (((ch - chs.start) * ACH + qi) * nh + h) * HD;
            att[dst..dst + HD].copy_from_slice(&out[qi * HD..(qi + 1) * HD]);
        }
    }
    att
}

/// B4 analysis of one layer: all heads, the last `NQ` positions of each prefix in `QT`.
fn analyze_layer(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    nh: usize,
    nkv: usize,
    posg: &[Vec<Grp>],
    total: &Mutex<Acc4>,
) {
    let mut kk = vec![vec![0f64; T16 * HD]; nkv];
    let mut vv = vec![vec![0f64; T16 * HD]; nkv];
    let mut vpm = vec![vec![0f64; T16]; nkv];
    for kvh in 0..nkv {
        let mut run = 0f64;
        for p in 0..T16 {
            let src = (p * nkv + kvh) * HD;
            for d in 0..HD {
                kk[kvh][p * HD + d] = f64::from(k[src + d]);
                let x = f64::from(v[src + d]);
                vv[kvh][p * HD + d] = x;
                run = run.max(x.abs());
            }
            vpm[kvh][p] = run;
        }
    }
    let tree_jobs: Vec<(usize, usize)> = (0..nkv)
        .flat_map(|a| (0..QT.len()).map(move |b| (a, b)))
        .collect();
    let trees: Mutex<Vec<((usize, usize), Vec<Grp>)>> = Mutex::new(Vec::new());
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..NTH4.min(tree_jobs.len()) {
            sc.spawn(|| loop {
                let j = next.fetch_add(1, Ordering::Relaxed);
                if j >= tree_jobs.len() {
                    break;
                }
                let (kvh, ti) = tree_jobs[j];
                let g = pca_groups(&kk[kvh], QT[ti] - NQ);
                trees.lock().unwrap().push(((kvh, ti), g));
            });
        }
    });
    let mut pca: Vec<Vec<Grp>> = vec![Vec::new(); nkv * QT.len()];
    for ((kvh, ti), g) in trees.into_inner().unwrap() {
        pca[kvh * QT.len() + ti] = g;
    }
    let mut jobs: Vec<(usize, usize, usize)> = Vec::new();
    for ti in (0..QT.len()).rev() {
        for h in 0..nh {
            for qi in 0..NQ {
                jobs.push((h, ti, qi));
            }
        }
    }
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..NTH4 {
            sc.spawn(|| {
                let mut local = Acc4::new();
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    if j >= jobs.len() {
                        break;
                    }
                    let (h, ti, qi) = jobs[j];
                    let kvh = h / (nh / nkv);
                    let pos = QT[ti] - NQ + qi;
                    let cx = LayerCtx {
                        kk: &kk[kvh],
                        vv: &vv[kvh],
                        vpm: &vpm[kvh],
                        posg: &posg[ti],
                        pca: &pca[kvh * QT.len() + ti],
                    };
                    let mut qv = [0f64; HD];
                    for d in 0..HD {
                        qv[d] = f64::from(q[(pos * nh + h) * HD + d]);
                    }
                    analyze_query(&cx, &qv, pos, ti, &mut local);
                }
                total.lock().unwrap().merge(&local);
            });
        }
    });
}

fn long_ids(bpe: &Bpe) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut ids: Vec<u32> = Vec::new();
    for (path, off, len) in [
        (PROSE, 0usize, 80_000usize),
        (PROSE, 2_000_000, 80_000),
        (PROSE, 5_000_000, 80_000),
    ] {
        if ids.len() >= T16 {
            break;
        }
        let raw = std::fs::read(path)?;
        let mut a = off;
        if a > 0 {
            while a < raw.len() && raw[a - 1] != b'\n' {
                a += 1;
            }
        }
        let b = (a + len).min(raw.len());
        let text = String::from_utf8_lossy(&raw[a..b]).into_owned();
        ids.extend(bpe.encode(&text)?);
    }
    if ids.len() < T16 {
        return Err(format!("long context: {} tokens < {T16}", ids.len()).into());
    }
    ids.truncate(T16);
    Ok(ids)
}

fn median_b(hist: &[f64]) -> f64 {
    let tot: f64 = hist.iter().sum();
    if tot == 0.0 {
        return f64::NAN;
    }
    let mut c = 0f64;
    for (i, h) in hist.iter().enumerate() {
        c += h;
        if c >= tot / 2.0 {
            return if i == 0 {
                0.0
            } else {
                10f64.powf(i as f64 * 0.02 - 4.0 + 0.01)
            };
        }
    }
    f64::NAN
}

fn grp_name(g: usize) -> String {
    format!("{}{}", GKIND[g / 4], LS[g % 4])
}

/// Least-squares slope of `ln y` on `ln t` over the positive points.
fn slope(ys: &[f64]) -> f64 {
    let pts: Vec<(f64, f64)> = QT
        .iter()
        .zip(ys)
        .filter(|(_, &y)| y > 0.0)
        .map(|(&t, &y)| ((t as f64).ln(), y.ln()))
        .collect();
    let n = pts.len() as f64;
    if pts.len() < 2 {
        return f64::NAN;
    }
    let (mx, my) = (
        pts.iter().map(|p| p.0).sum::<f64>() / n,
        pts.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    sxy / sxx
}

/// The 16k arm: one 16,384-token prompt through the f32 forward, the B4 analysis of every layer
/// at its prefixes in `QT`, then the tables. Returns whether every certified bound dominated its
/// realised error.
#[allow(clippy::too_many_lines)]
fn run_b4(
    eng: &Engine,
    bpe: &Bpe,
    ids0: &[u32],
    kcap: &[Vec<f32>],
    vcap: &[Vec<f32>],
) -> Result<bool, Box<dyn Error>> {
    let nth = 16usize;
    let cfg = &eng.config;
    let (nl, nh, nkv, d) = (cfg.n_layer, cfg.n_head, cfg.n_head_kv, cfg.n_embd);
    let tt0 = std::time::Instant::now();
    let ids = long_ids(bpe)?;
    let same = ids[..T] == *ids0;
    let bytes: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
    let hex: String = apeiron::sha256::sha256(&bytes)
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("\n== B4: 16k context, moment nodes ==");
    println!(
        "long prompt: {} tokens, ids sha256[..16] {hex}; first {T} ids equal prompt 0: {same}; queries: last {NQ} positions of prefixes {QT:?}, all {nl} layers x {nh} heads (no stride)",
        ids.len()
    );
    println!(
        "node cost (floats) = (1+dv) sum_(r<p) d^r + d, d=dv={HD}; key = {KEYF} floats; break-even = cost/128 key-equivalents"
    );
    for p in PS {
        println!(
            "  p={p}: cost {:.0} floats, break-even {:.1} keys, C_p={:.4}, B_max(eps=1e-3,1e-4,1e-5) = {:.4}, {:.4}, {:.4}",
            node_cost(p),
            node_cost(p) / KEYF,
            rp(p),
            bmax(EPS[0], p),
            bmax(EPS[1], p),
            bmax(EPS[2], p)
        );
    }
    let rope = eng.rope_table().map_err(|e| format!("{e:?}"))?;
    let half = HD / 2;
    let mut cs = vec![(0f64, 0f64); T16 * half];
    for p in 0..T16 {
        let mut x = vec![0i64; HD];
        for xj in x.iter_mut().take(half) {
            *xj = ONE as i64;
        }
        rope.rotate_one(&mut x, p as u32)
            .map_err(|e| format!("rope position {p}: {e:?}"))?;
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
    let mut x: Vec<f32> = Vec::with_capacity(T16 * d);
    for &id in &ids {
        x.extend(q32_vec(&eng.token_embedding(id)?));
    }
    let vecs = eng.decode_vectors();
    let eps = cfg.rms_eps;
    let posg: Vec<Vec<Grp>> = QT.iter().map(|&t| pos_groups(t - NQ)).collect();
    let total = Mutex::new(Acc4::new());
    let (mut wk, mut wv) = (0f64, 0f64);
    let (mut t_fwd, mut t_an) = (0f64, 0f64);
    for l in 0..nl {
        let tl = std::time::Instant::now();
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
        let (wq, ni, no) = weight(eng, &format!("blk.{l}.attn_q.weight"))?;
        let mut q = gemm(&wq, no, ni, &xn, T16, bq.as_deref(), nth);
        drop(wq);
        let (wkm, ni, no) = weight(eng, &format!("blk.{l}.attn_k.weight"))?;
        let mut k = gemm(&wkm, no, ni, &xn, T16, bkk.as_deref(), nth);
        drop(wkm);
        let (wvm, ni, no) = weight(eng, &format!("blk.{l}.attn_v.weight"))?;
        let v = gemm(&wvm, no, ni, &xn, T16, bv.as_deref(), nth);
        drop(wvm);
        for pp in 0..T16 {
            rot(&mut q[pp * nh * HD..(pp + 1) * nh * HD], nh, pp);
            rot(&mut k[pp * nkv * HD..(pp + 1) * nkv * HD], nkv, pp);
        }
        if same {
            let a: Vec<f64> = kcap[l].iter().map(|&y| f64::from(y)).collect();
            let b: Vec<f64> = vcap[l].iter().map(|&y| f64::from(y)).collect();
            wk = wk.max(rel_l2(&k[..T * nkv * HD], &a).0);
            wv = wv.max(rel_l2(&v[..T * nkv * HD], &b).0);
        }
        let ta = std::time::Instant::now();
        let tproj = tl.elapsed().as_secs_f64();
        analyze_layer(&q, &k, &v, nh, nkv, &posg, &total);
        t_an += ta.elapsed().as_secs_f64();
        let tan = tl.elapsed().as_secs_f64();
        let att = attend_all(&q, &k, &v, T16, nh, nkv, nth, 0..T16 / ACH);
        let tat = tl.elapsed().as_secs_f64();
        let (wo, ni, no) = weight(eng, &format!("blk.{l}.attn_output.weight"))?;
        let o = gemm(&wo, no, ni, &att, T16, None, nth);
        drop(wo);
        for (xi, oi) in x.iter_mut().zip(&o) {
            *xi += oi;
        }
        let xn2 = rmsnorm(&x, &g2, eps, d);
        let (wg, ni, no) = weight(eng, &format!("blk.{l}.ffn_gate.weight"))?;
        let mut gate = gemm(&wg, no, ni, &xn2, T16, None, nth);
        drop(wg);
        let (wu, ni, no) = weight(eng, &format!("blk.{l}.ffn_up.weight"))?;
        let up = gemm(&wu, no, ni, &xn2, T16, None, nth);
        drop(wu);
        let per = gate.len().div_ceil(nth);
        std::thread::scope(|sc| {
            for (gc, uc) in gate.chunks_mut(per).zip(up.chunks(per)) {
                sc.spawn(move || {
                    for (gi, ui) in gc.iter_mut().zip(uc) {
                        let gv = f64::from(*gi);
                        *gi = (gv / (1.0 + (-gv).exp()) * f64::from(*ui)) as f32;
                    }
                });
            }
        });
        let (wd, ni, no) = weight(eng, &format!("blk.{l}.ffn_down.weight"))?;
        let dn = gemm(&wd, no, ni, &gate, T16, None, nth);
        drop(wd);
        for (xi, di) in x.iter_mut().zip(&dn) {
            *xi += di;
        }
        t_fwd += tl.elapsed().as_secs_f64();
        eprintln!(
            "  layer {l}: qkv {tproj:.1}s, analysis {:.1}s, attention {:.1}s, rest {:.1}s",
            tan - tproj,
            tat - tan,
            tl.elapsed().as_secs_f64() - tat
        );
        eprintln!(
            "B4 layer {l} done, {:.0}s (analysis so far {t_an:.0}s)",
            tt0.elapsed().as_secs_f64()
        );
    }
    println!(
        "forward+analysis {:.0}s (analysis {t_an:.0}s, layers total {t_fwd:.0}s); first-{T}-token K/V of this run vs the pp512 arm (prompt 0): K rel-L2 max {wk:.2e}, V rel-L2 max {wv:.2e}",
        tt0.elapsed().as_secs_f64()
    );
    let a = total.into_inner().unwrap();
    let ratio = |c: &Cell| {
        if c.n == 0.0 {
            f64::NAN
        } else {
            c.rsum / c.n
        }
    };
    println!(
        "\ncut only (no nodes), read/t mean [min,max] over (layer,head,query); rows adaptive delta / grid delta"
    );
    for ei in 0..3 {
        for (nm, w) in [("adapt", 0), ("grid", 1)] {
            print!("eps {:e} {nm:<5}", EPS[ei]);
            for ti in 0..QT.len() {
                print!(
                    " t={:<5} {:<22}",
                    QT[ti],
                    a.cut[(ti * 3 + ei) * 2 + w].fmt()
                );
            }
            println!();
        }
    }
    let mut cells_le_half: Vec<String> = Vec::new();
    let mut min_c = (f64::INFINITY, String::new());
    let mut min_f = (f64::INFINITY, String::new());
    for (mode, mname) in [
        (0usize, "C (nodes cost node_cost(p)/128)"),
        (1, "F (node cost = 1 key-equivalent; spread-only)"),
    ] {
        println!(
            "\n== mode {mname}: read_equiv/t, mean [min,max]; exp = growth exponent of mean read_equiv in t =="
        );
        for ei in 0..3 {
            for pi in 0..3 {
                println!("-- eps {:e}, p={}", EPS[ei], PS[pi]);
                print!("{:<7}", "group");
                for t in QT {
                    print!(" {:<24}", format!("t={t}"));
                }
                println!(" exp");
                for g in 0..NG {
                    print!("{:<7}", grp_name(g));
                    let mut ys = Vec::new();
                    for ti in 0..QT.len() {
                        let c = &a.cell[ci(ti, g, mode, ei, pi)];
                        print!(" {:<24}", c.fmt());
                        ys.push(if c.n > 0.0 { c.abs / c.n } else { 0.0 });
                        let r = ratio(c);
                        if QT[ti] >= 4096 && r.is_finite() {
                            let tag = format!(
                                "mode{mode} {} eps={:e} p={} t={}: {r:.3}",
                                grp_name(g),
                                EPS[ei],
                                PS[pi],
                                QT[ti]
                            );
                            let mn = if mode == 0 { &mut min_c } else { &mut min_f };
                            if r < mn.0 {
                                *mn = (r, tag.clone());
                            }
                            if r <= 0.5 {
                                cells_le_half.push(tag);
                            }
                        }
                    }
                    println!(" {:.2}", slope(&ys));
                }
            }
        }
    }
    println!(
        "\n== summarised share (nodes' keys / non-kept keys) and share of queries using >= 1 node, mode C, eps 1e-3, p=2 and p=3, mean over (layer,head,query) =="
    );
    for pi in [1usize, 2] {
        println!("-- p={}", PS[pi]);
        for g in 0..NG {
            print!("{:<7}", grp_name(g));
            for ti in 0..QT.len() {
                let c = &a.cell[ci(ti, g, 0, 0, pi)];
                print!(
                    " t={:<5} {:.4}/{:.4}  ",
                    QT[ti],
                    c.nk / c.n.max(1.0),
                    c.used / c.n.max(1.0)
                );
            }
            println!();
        }
    }
    println!("\n== summarised share, mode F, eps 1e-3, p=3 (spread-only upper reference) ==");
    for g in 0..NG {
        print!("{:<7}", grp_name(g));
        for ti in 0..QT.len() {
            let c = &a.cell[ci(ti, g, 1, 0, 2)];
            print!(
                " t={:<5} {:.4}/{:.4}  ",
                QT[ti],
                c.nk / c.n.max(1.0),
                c.used / c.n.max(1.0)
            );
        }
        println!();
    }
    println!("\n== node spread over all nodes of the grouping: share with B<=1 / median B ==");
    print!("{:<7}", "group");
    for t in QT {
        print!(" {:<22}", format!("t={t}"));
    }
    println!();
    for g in 0..NG {
        print!("{:<7}", grp_name(g));
        for ti in 0..QT.len() {
            let r = (ti * NG + g) * 11;
            let nn = a.bcnt[r].max(1.0);
            let med = median_b(&a.hist[(ti * NG + g) * NBIN..(ti * NG + g + 1) * NBIN]);
            print!(" {:<22}", format!("{:.4} / {med:.3}", a.bcnt[r + 1] / nn));
        }
        println!();
    }
    for ti in [2usize, 3] {
        println!(
            "\n== share of nodes with B <= B_max(eps,p), t={} (columns: p=1 eps 1e-3,1e-4,1e-5 | p=2 ... | p=3 ...) ==",
            QT[ti]
        );
        for g in 0..NG {
            print!("{:<7}", grp_name(g));
            let r = (ti * NG + g) * 11;
            let nn = a.bcnt[r].max(1.0);
            for x in 0..9 {
                print!(" {:.4}", a.bcnt[r + 2 + x] / nn);
            }
            println!();
        }
    }
    println!(
        "\nmin read/t over (mode C, t>=4096): {:.3} at {}",
        min_c.0, min_c.1
    );
    println!(
        "min read/t over (mode F, t>=4096): {:.3} at {}",
        min_f.0, min_f.1
    );
    println!(
        "cells (t>=4096) with read/t <= 0.5: {}",
        cells_le_half.len()
    );
    for c in cells_le_half.iter().take(20) {
        println!("  {c}");
    }
    println!(
        "\nB4 checks {} (cut-only and node reads, every candidate delta); errors {}; max realised/bound {:.4}",
        a.checks, a.errs, a.worst
    );
    if let Some(v) = &a.first_violation {
        println!("COUNTEREXAMPLE (first): {v}");
    }
    let pct = 100.0 * a.holds / a.checks.max(1.0);
    println!("B4 bound holds: {pct:.2}%");
    Ok(a.holds == a.checks && a.errs == 0.0)
}

/// Per (layer, q head, RoPE pair j): prompt 0, all causal (t, s <= t). `.0[h*half+j]` is
/// sum |q_j(t)||k_j(s)| (bound product), `.1[h*half+j]` is sum (q_j(t) . k_j(s))^2 (score-contribution
/// second moment); the pair is the 2-vector (x[j], x[j+half]). O(T nh half) per layer via prefix sums.
fn qk_energy_layer(
    _l: usize,
    q: &[f32],
    k: &[f32],
    nh: usize,
    nkv: usize,
    out: &mut Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>,
) {
    let half = HD / 2;
    let mut a = vec![0f64; nh * half];
    let mut v = vec![0f64; nh * half];
    let mut c = vec![0f64; nh * half];
    for h in 0..nh {
        let kvh = h / (nh / nkv);
        for j in 0..half {
            let (mut sk, mut sxx, mut sxy, mut syy) = (0f64, 0f64, 0f64, 0f64);
            let (mut mx, mut my) = (0f64, 0f64);
            for t in 0..T {
                let kb = (t * nkv + kvh) * HD;
                let (kx, ky) = (f64::from(k[kb + j]), f64::from(k[kb + j + half]));
                sk += kx.hypot(ky);
                sxx += kx * kx;
                sxy += kx * ky;
                syy += ky * ky;
                mx += kx;
                my += ky;
                let qb = (t * nh + h) * HD;
                let (qx, qy) = (f64::from(q[qb + j]), f64::from(q[qb + j + half]));
                a[h * half + j] += qx.hypot(qy) * sk;
                let m2 = qx * qx * sxx + 2.0 * qx * qy * sxy + qy * qy * syy;
                v[h * half + j] += m2;
                let sum = qx * mx + qy * my;
                c[h * half + j] += m2 - sum * sum / (t + 1) as f64;
            }
        }
    }
    out.push((a, v, c));
}

fn qk_energy_report(qke: &[(Vec<f64>, Vec<f64>, Vec<f64>)], nh: usize) {
    let half = HD / 2;
    let j0 = 19usize; // parabolic pairs at L = 512
    let uniform = (half - j0) as f64 / half as f64;
    println!(
        "\n== QK_ENERGY: share of pairs j >= {j0} (uniform {uniform:.4}), prompt 0, T = {T} =="
    );
    println!("layer  bound  second-moment  centred(per-query)  heads(centred>uniform)/{nh}");
    let mut sh = [Vec::new(), Vec::new(), Vec::new()];
    let mut pair = [vec![0f64; half], vec![0f64; half], vec![0f64; half]];
    for (l, (a, v, c)) in qke.iter().enumerate() {
        let mut hi = [0f64; 3];
        let mut all = [0f64; 3];
        let mut nover = 0usize;
        for h in 0..nh {
            let (mut hc, mut htc) = (0f64, 0f64);
            for j in 0..half {
                let x = [a[h * half + j], v[h * half + j], c[h * half + j]];
                for m in 0..3 {
                    all[m] += x[m];
                    pair[m][j] += x[m];
                    if j >= j0 {
                        hi[m] += x[m];
                    }
                }
                htc += x[2];
                if j >= j0 {
                    hc += x[2];
                }
            }
            if hc / htc > uniform {
                nover += 1;
            }
        }
        for m in 0..3 {
            sh[m].push(hi[m] / all[m]);
        }
        println!(
            "{l:5}  {:.4}  {:.4}  {:.4}  {nover}",
            hi[0] / all[0],
            hi[1] / all[1],
            hi[2] / all[2]
        );
    }
    println!(
        "layer mean [min,max]: bound {}  second-moment {}  centred {}",
        mean_min_max(&sh[0]),
        mean_min_max(&sh[1]),
        mean_min_max(&sh[2])
    );
    let tot: Vec<f64> = pair.iter().map(|p| p.iter().sum()).collect();
    println!("per-pair share over all layers/heads (j: bound, second-moment, centred):");
    for j in 0..half {
        println!(
            "  j={j:2} {:.4} {:.4} {:.4}",
            pair[0][j] / tot[0],
            pair[1][j] / tot[1],
            pair[2][j] / tot[2]
        );
    }
}

/// `KV_DUMP=<dir> KV_T=<tokens>`: one contiguous `prose8.txt` prompt (offset 0, so its first `T`
/// ids are prompt 0's) through the f32 forward; writes per layer the unrotated q, k and v
/// (`q{l}.bin` `KV_T x n_head x 64`, `k{l}.bin`/`v{l}.bin` `KV_T x n_head_kv x 64`, f32 LE), the
/// rope table `cs.bin` (`KV_T x 32` (cos, sin) f64 LE) and `ids.bin` (u32 LE). Causal, so a
/// prefix of length t is the dump at t. Each layer runs in three phases over token chunks of
/// `KV_CHUNK` rows (default 8192): q, k, v; attention and the output projection; the FFN. Every
/// row's result is the same computation whatever the chunking, so the dump does not depend on
/// `KV_CHUNK`. The residual is checkpointed in `x.bin` / `next.txt` (`l`, or `l phase chunk`): an
/// invocation stops after the chunk that crosses `KV_BUDGET` seconds (default 420) and exits 3;
/// rerun to continue. Each layer's first `T` rows of q and k are compared with `QK_REF/q{l}.bin`,
/// `k{l}.bin` (the T=512 dumps) when `QK_REF` is set.
///
/// Cost per layer: projections `KV_T (n_embd (n_head + 2 n_head_kv) HD + n_embd^2 + 3 n_embd n_ff)`
/// multiply-adds, attention `n_head KV_T (KV_T + 1) HD` (causal, q.k and p.v); memory `KV_T` rows of
/// x, q, k and v plus `KV_CHUNK n_ff` f32 four times (gate, up and their transposes).
#[allow(clippy::too_many_lines)]
fn run_dump(path: &str, dir: &std::path::Path, tn: usize) -> Result<bool, Box<dyn Error>> {
    use std::io::Write;
    let nth = 16usize;
    let budget: f64 = std::env::var("KV_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(420.0);
    let t0 = std::time::Instant::now();
    let eng = Engine::open(path)?;
    let cfg = &eng.config;
    let (nl, nh, nkv, d) = (cfg.n_layer, cfg.n_head, cfg.n_head_kv, cfg.n_embd);
    assert_eq!(cfg.head_dim, HD);
    assert_eq!(tn % ACH, 0, "KV_T must be a multiple of {ACH}");
    let bpe = Bpe::from_tokenizer_json(TOKENIZER)?;
    let raw = std::fs::read(PROSE)?;
    let text = String::from_utf8_lossy(&raw[..(tn * 8).min(raw.len())]).into_owned();
    let mut ids = bpe.encode(&text)?;
    if ids.len() < tn {
        return Err(format!("dump prompt: {} tokens < {tn}", ids.len()).into());
    }
    ids.truncate(tn);
    let p0 = prompts(&bpe)?.swap_remove(0);
    let bytes: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
    let hex: String = apeiron::sha256::sha256(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!(
        "dump: {tn} tokens from {PROSE}[0..], ids sha256 {hex}; first {T} ids equal prompt 0: {}",
        ids[..T] == p0[..]
    );
    let le = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    let half = HD / 2;
    let rope = eng.rope_table().map_err(|e| format!("{e:?}"))?;
    let mut cs = vec![(0f64, 0f64); tn * half];
    for p in 0..tn {
        let mut x = vec![0i64; HD];
        for xj in x.iter_mut().take(half) {
            *xj = ONE as i64;
        }
        rope.rotate_one(&mut x, p as u32)
            .map_err(|e| format!("rope position {p}: {e:?}"))?;
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
    // Progress: `next.txt` is `l` (layer `l` from its start) or `l p c` (layer `l`, phase `p` = 1 attention
    // or 2 FFN, next token chunk `c`); `x.bin` is the residual at that point.
    let cn: usize = std::env::var("KV_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192)
        .min(tn);
    assert!(
        cn.is_multiple_of(ACH) && cn >= T,
        "KV_CHUNK must be a multiple of {ACH} and at least {T}"
    );
    let nc = tn.div_ceil(cn);
    let rows = |c: usize| c * cn..((c + 1) * cn).min(tn);
    let next_path = dir.join("next.txt");
    let (start, mut phase, mut chunk) = match std::fs::read_to_string(&next_path) {
        Ok(s) => {
            let f: Vec<usize> = s
                .split_whitespace()
                .map(str::parse)
                .collect::<Result<_, _>>()?;
            match f[..] {
                [l] => (l, 0, 0),
                [l, p, c] => (l, p, c),
                _ => return Err(format!("next.txt: {s:?}").into()),
            }
        }
        Err(_) => (0, 0, 0),
    };
    let mut x: Vec<f32> = if (start, phase, chunk) == (0, 0, 0) {
        std::fs::write(dir.join("ids.bin"), &bytes)?;
        let c: Vec<u8> = cs
            .iter()
            .flat_map(|&(c, s)| [c, s])
            .flat_map(f64::to_le_bytes)
            .collect();
        std::fs::write(dir.join("cs.bin"), c)?;
        let mut x = Vec::with_capacity(tn * d);
        for &id in &ids {
            x.extend(q32_vec(&eng.token_embedding(id)?));
        }
        x
    } else {
        let b = std::fs::read(dir.join("x.bin"))?;
        assert_eq!(b.len(), tn * d * 4, "x.bin size");
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    println!(
        "resume at layer {start} phase {phase} chunk {chunk} of {nl} layers, {nc} chunks of {cn}"
    );
    let save = |x: &[f32], l: usize, p: usize, c: usize| -> Result<(), Box<dyn Error>> {
        let tmp = dir.join("x.bin.tmp");
        std::fs::write(&tmp, le(x))?;
        std::fs::rename(&tmp, dir.join("x.bin"))?;
        let at = if p == 0 {
            format!("{l}\n")
        } else {
            format!("{l} {p} {c}\n")
        };
        std::fs::write(&next_path, at)?;
        Ok(())
    };
    let rd = |n: &std::path::Path| -> Result<Vec<f32>, Box<dyn Error>> {
        Ok(std::fs::read(n)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    let qref = std::env::var_os("QK_REF").map(std::path::PathBuf::from);
    let vecs = eng.decode_vectors();
    let eps = cfg.rms_eps;
    for l in start..nl {
        let tl = std::time::Instant::now();
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
        let budget_hit = |p: usize, c: usize, x: &[f32]| -> Result<bool, Box<dyn Error>> {
            if t0.elapsed().as_secs_f64() > budget {
                save(x, l, p, c)?;
                println!(
                    "budget {budget}s reached at layer {l} phase {p} chunk {c}; rerun to continue"
                );
                return Ok(true);
            }
            Ok(false)
        };
        if phase == 0 {
            // q, k, v by token chunk, streamed to disk unrotated.
            let (wq, ni, nq) = weight(&eng, &format!("blk.{l}.attn_q.weight"))?;
            let (wkm, _, nk) = weight(&eng, &format!("blk.{l}.attn_k.weight"))?;
            let (wvm, _, nv) = weight(&eng, &format!("blk.{l}.attn_v.weight"))?;
            let open = |n: &str| -> Result<std::io::BufWriter<std::fs::File>, Box<dyn Error>> {
                Ok(std::io::BufWriter::new(std::fs::File::create(dir.join(n))?))
            };
            let (mut fq, mut fk, mut fv) = (
                open(&format!("q{l}.bin"))?,
                open(&format!("k{l}.bin"))?,
                open(&format!("v{l}.bin"))?,
            );
            for c in 0..nc {
                let r = rows(c);
                let xn = rmsnorm(&x[r.start * d..r.end * d], &g1, eps, d);
                let q = gemm(&wq, nq, ni, &xn, r.len(), bq.as_deref(), nth);
                let k = gemm(&wkm, nk, ni, &xn, r.len(), bkk.as_deref(), nth);
                let v = gemm(&wvm, nv, ni, &xn, r.len(), bv.as_deref(), nth);
                if let (Some(rf), 0) = (&qref, c) {
                    let rf64 = |n: &str| -> Result<Vec<f64>, Box<dyn Error>> {
                        Ok(rd(&rf.join(n))?.into_iter().map(f64::from).collect())
                    };
                    let (eq, _) = rel_l2(&q[..T * nh * HD], &rf64(&format!("q{l}.bin"))?);
                    let (ek, _) = rel_l2(&k[..T * nkv * HD], &rf64(&format!("k{l}.bin"))?);
                    println!("layer {l}: first {T} rows vs QK_REF rel-L2 q {eq:.3e} k {ek:.3e}");
                }
                fq.write_all(&le(&q))?;
                fk.write_all(&le(&k))?;
                fv.write_all(&le(&v))?;
            }
            fq.flush()?;
            fk.flush()?;
            fv.flush()?;
            (phase, chunk) = (1, 0);
        }
        let tp = tl.elapsed().as_secs_f64();
        if phase == 1 {
            let mut q = rd(&dir.join(format!("q{l}.bin")))?;
            let mut k = rd(&dir.join(format!("k{l}.bin")))?;
            let v = rd(&dir.join(format!("v{l}.bin")))?;
            for pp in 0..tn {
                rot(&mut q[pp * nh * HD..(pp + 1) * nh * HD], nh, pp);
                rot(&mut k[pp * nkv * HD..(pp + 1) * nkv * HD], nkv, pp);
            }
            let (wo, ni, no) = weight(&eng, &format!("blk.{l}.attn_output.weight"))?;
            while chunk < nc {
                let r = rows(chunk);
                let att = attend_all(&q, &k, &v, tn, nh, nkv, nth, r.start / ACH..r.end / ACH);
                let o = gemm(&wo, no, ni, &att, r.len(), None, nth);
                for (xi, oi) in x[r.start * d..r.end * d].iter_mut().zip(&o) {
                    *xi += oi;
                }
                chunk += 1;
                if chunk < nc && budget_hit(1, chunk, &x)? {
                    std::process::exit(3);
                }
            }
            chunk = 0;
        }
        let ta = tl.elapsed().as_secs_f64();
        let (wg, ni, nf) = weight(&eng, &format!("blk.{l}.ffn_gate.weight"))?;
        let (wu, _, _) = weight(&eng, &format!("blk.{l}.ffn_up.weight"))?;
        let (wd, nfi, no) = weight(&eng, &format!("blk.{l}.ffn_down.weight"))?;
        while chunk < nc {
            let r = rows(chunk);
            let xn2 = rmsnorm(&x[r.start * d..r.end * d], &g2, eps, d);
            let mut gate = gemm(&wg, nf, ni, &xn2, r.len(), None, nth);
            let up = gemm(&wu, nf, ni, &xn2, r.len(), None, nth);
            drop(xn2);
            let per = gate.len().div_ceil(nth);
            std::thread::scope(|sc| {
                for (gc, uc) in gate.chunks_mut(per).zip(up.chunks(per)) {
                    sc.spawn(move || {
                        for (gi, ui) in gc.iter_mut().zip(uc) {
                            let gv = f64::from(*gi);
                            *gi = (gv / (1.0 + (-gv).exp()) * f64::from(*ui)) as f32;
                        }
                    });
                }
            });
            drop(up);
            let dn = gemm(&wd, no, nfi, &gate, r.len(), None, nth);
            for (xi, di) in x[r.start * d..r.end * d].iter_mut().zip(&dn) {
                *xi += di;
            }
            chunk += 1;
            if chunk < nc && budget_hit(2, chunk, &x)? {
                std::process::exit(3);
            }
        }
        (phase, chunk) = (0, 0);
        save(&x, l + 1, 0, 0)?;
        println!(
            "layer {l}: qkv {tp:.1}s attention {:.1}s rest {:.1}s; total {:.0}s",
            ta - tp,
            tl.elapsed().as_secs_f64() - ta,
            t0.elapsed().as_secs_f64()
        );
        if l + 1 < nl && t0.elapsed().as_secs_f64() > budget {
            println!("budget {budget}s reached after layer {l}; rerun to continue");
            std::process::exit(3);
        }
    }
    println!("dump complete: {nl} layers");
    Ok(true)
}

const TL_SINK: usize = 4;
const TL_W: usize = 256;
const TL_P: usize = 64;

/// A decode-tail selection arm. `f64` fields are the per-head mass tolerance `eps` (kept mass
/// `>= 1 - eps`, KL `-ln(1 - eps)`); `Fix` carries the far-token budget; `Grp(b, r)` is one page set
/// of `b` far tokens per KV head, ranked by the group's highest bound and recomputed every `r` positions.
#[derive(Clone, Copy)]
enum Arm {
    Dense,
    Oracle(f64),
    Near,
    Order(f64),
    Cert(f64),
    Fix(usize),
    Grp(usize, usize),
    Tree(usize, usize, usize),
    Clu(usize, usize, usize),
}

impl Arm {
    fn name(self) -> String {
        let kl = |e: f64| -(1.0 - e).ln();
        match self {
            Arm::Dense => "dense".into(),
            Arm::Oracle(e) => format!("oracle-{:.2}", kl(e)),
            Arm::Near => "near".into(),
            Arm::Order(e) => format!("a64-order-{:.2}", kl(e)),
            Arm::Cert(e) => format!("a64-cert-{:.2}", kl(e)),
            Arm::Fix(b) => format!("a64-fix{b}"),
            Arm::Grp(b, r) => match sum_mode() {
                Some(m) if sum_unit() != TL_P => format!("grp-fix{b}-r{r}-sum{m}-u{}", sum_unit()),
                Some(m) => format!("grp-fix{b}-r{r}-sum{m}"),
                None => format!("grp-fix{b}-r{r}"),
            },
            Arm::Tree(b, r, s) => format!("tree{b}-r{r}-s{s}"),
            Arm::Clu(b, r, s) => format!("clu{b}-r{r}-s{s}"),
        }
    }
}

fn rows_f32(
    p: &std::path::Path,
    width: usize,
    r0: usize,
    r1: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(p)?;
    f.seek(SeekFrom::Start((r0 * width * 4) as u64))?;
    let mut b = vec![0u8; (r1 - r0) * width * 4];
    f.read_exact(&mut b)?;
    let mut out = vec![0f32; b.len() / 4];
    par_rows(&mut out, 1, PAR, |i0, ch| {
        for (o, x) in ch.iter_mut().enumerate() {
            let c = &b[(i0 + o) * 4..(i0 + o) * 4 + 4];
            *x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
    });
    Ok(out)
}

/// One layer's keys for the tail queries: the dump's rotated prefix rows `[0, pre)`, the arm's own
/// rotated tail rows, and Quest's element-wise page (min, max) over 64-key pages from `TL_SINK`.
/// `KV_TAIL_SUM=m` makes `Grp` summarise every omitted far page instead of dropping it: mode 0 adds mass
/// `64 exp(mean score)`, mode 1 the exact page mass, mode 2 `64 exp(mean + variance / 2)`, each with the page's mean
/// value row.
fn sum_mode() -> Option<u8> {
    std::env::var("KV_TAIL_SUM")
        .ok()
        .and_then(|v| v.parse().ok())
}

/// `KV_TAIL_DENSE=l,..` runs the listed layers dense; `KV_TAIL_SPARSE=l,..` runs every other layer dense. Layers
/// outside both lists run the arm. Attribution of the tail KLD to layers.
fn layer_arm(l: usize, arm: Arm) -> Arm {
    let has = |k: &str| {
        std::env::var(k).ok().map(|v| {
            v.split(',')
                .filter_map(|x| x.parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
    };
    let dense = has("KV_TAIL_DENSE").is_some_and(|d| d.contains(&l));
    let off = has("KV_TAIL_SPARSE").is_some_and(|s| !s.contains(&l));
    if dense || off {
        return Arm::Dense;
    }
    match (arm, layer_budget(l)) {
        (Arm::Grp(_, r), Some(n)) => Arm::Grp(n, r),
        _ => arm,
    }
}

/// `KV_TAIL_LFARB=lo-hi:n,..` sets the far budget of layers `lo..=hi` to `n` tokens; other layers keep the arm's.
fn layer_budget(l: usize) -> Option<usize> {
    let v = std::env::var("KV_TAIL_LFARB").ok()?;
    v.split(',').find_map(|e| {
        let (r, n) = e.split_once(':')?;
        let (lo, hi) = r.split_once('-')?;
        let (lo, hi): (usize, usize) = (lo.parse().ok()?, hi.parse().ok()?);
        (lo..=hi).contains(&l).then(|| n.parse().ok()).flatten()
    })
}

/// `KV_TAIL_SUMU=u` sets the summary unit to `u` keys (default `TL_P`, a divisor of `TL_P`): a sub-page of `u` keys
/// carries one mass and one mean value row.
fn sum_unit() -> usize {
    std::env::var("KV_TAIL_SUMU")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&u| u >= 1 && TL_P.is_multiple_of(u))
        .unwrap_or(TL_P)
}

struct TailCtx<'a> {
    vbar: &'a [f32],
    pre: usize,
    nkv: usize,
    kp: &'a [f32],
    vp: &'a [f32],
    kt: &'a [f32],
    vt: &'a [f32],
    npg: usize,
    mn: &'a [f32],
    mx: &'a [f32],
    mutb: bool,
}

/// Quest's bound on `q.k / 8` over far page `b` of KV head `g`; the min-only `q.m` under `mutb`.
fn page_bound(cx: &TailCtx, qh: &[f32], g: usize, b: usize) -> f64 {
    let o = (g * cx.npg + b) * HD;
    let (mnr, mxr) = (&cx.mn[o..o + HD], &cx.mx[o..o + HD]);
    let mut acc = 0f64;
    for i in 0..HD {
        let (a, c) = (
            f64::from(qh[i]) * f64::from(mnr[i]),
            f64::from(qh[i]) * f64::from(mxr[i]),
        );
        acc += if cx.mutb { a } else { a.max(c) };
    }
    acc * f64::from(SCALE)
}

/// Far pages at position `p`: the 64-key pages between the sinks and the near window.
fn far_pages(p: usize) -> usize {
    (p + 1).saturating_sub(TL_W + TL_SINK) / TL_P
}

/// One tail query of one head: output, keys read by this head (T5/T6 accounting), bound vectors read
/// by this head, kept mass, certification broken, and the keys used (for the KV head's union).
struct TailOut {
    o: Vec<f32>,
    reads: f64,
    meta: f64,
    kept: f64,
    viol: bool,
    sel: Vec<bool>,
    /// `KV_TAIL_POS` proxies from the page bounds: `ln(1 + B/Z_read)` with `B` the bound mass of the unread far
    /// pages, and the best unread bound minus the best read score (both in score units); zero when unset.
    prox: [f64; 2],
}

/// One tail query of one head under arm `arm`. Keys read count a page bound as 2 vectors and a
/// selected page as 64; dense and oracle read `p+1` and the oracle's kept count respectively. `Grp`
/// takes the KV head's page set `given`, whose bounds are counted once per KV head by the caller. The
/// scores of all `p+1` keys are computed for every arm to report the kept mass; that scan is the
/// instrument, not the arm's cost.
///
/// Complexity: time `O((p+1) HD)` multiply-adds for the scores and values, plus `O(n64 HD)` for the
/// bounds and an `O((p+1) log(p+1))` sort (oracle) or `O(n64 log n64)` (pages); space `O(p+1)` words.
fn tail_head(cx: &TailCtx, arm: Arm, qh: &[f32], g: usize, p: usize, given: &[usize]) -> TailOut {
    let key = |j: usize| key_row(cx, g, j);
    let val = |j: usize| val_row(cx, g, j);
    let n = p + 1;
    let sc: Vec<f64> = (0..n)
        .map(|j| f64::from(dot32(qh, key(j))) * f64::from(SCALE))
        .collect();
    let m = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
    let z: f64 = w.iter().sum();
    let n64 = far_pages(p);
    let nb = TL_SINK + n64 * TL_P;
    let page = |b: usize| TL_SINK + b * TL_P..TL_SINK + (b + 1) * TL_P;
    let mut sel = vec![false; n];
    let mut meta = 0f64;
    let reads = if let Some((s, r)) = simple_sel(arm, p, given) {
        sel = s;
        r
    } else if let Arm::Oracle(eps) = arm {
        let mut ord: Vec<usize> = (0..n).collect();
        ord.sort_by(|&a, &b| w[b].total_cmp(&w[a]));
        let mut acc = 0.0;
        let mut c = 0;
        while c < n && acc < (1.0 - eps) * z {
            acc += w[ord[c]];
            sel[ord[c]] = true;
            c += 1;
        }
        c as f64
    } else {
        for s in sel.iter_mut().take(TL_SINK) {
            *s = true;
        }
        for s in &mut sel[nb..] {
            *s = true;
        }
        let nn = TL_SINK + (n - nb);
        {
            let u: Vec<f64> = (0..n64).map(|b| page_bound(cx, qh, g, b)).collect();
            let mut ord: Vec<usize> = (0..n64).collect();
            ord.sort_by(|&a, &b| u[b].total_cmp(&u[a]));
            meta = (2 * n64) as f64;
            let near_w: f64 = (0..TL_SINK).chain(nb..n).map(|j| w[j]).sum();
            let take = match arm {
                Arm::Order(eps) => {
                    let mut kept = near_w;
                    let mut j = 0;
                    while j < n64 && kept < (1.0 - eps) * z {
                        kept += page(ord[j]).map(|i| w[i]).sum::<f64>();
                        j += 1;
                    }
                    j
                }
                Arm::Cert(eps) => {
                    let c = (0..TL_SINK)
                        .chain(nb..n)
                        .map(|j| sc[j])
                        .chain(u.iter().copied())
                        .fold(f64::NEG_INFINITY, f64::max);
                    let mut zr: f64 = (0..TL_SINK).chain(nb..n).map(|j| (sc[j] - c).exp()).sum();
                    let mut suf = vec![0f64; n64 + 1];
                    for j in (0..n64).rev() {
                        suf[j] = suf[j + 1] + TL_P as f64 * (u[ord[j]] - c).exp();
                    }
                    let mut j = 0;
                    while j < n64 && suf[j] > zr * eps / (1.0 - eps) {
                        zr += page(ord[j]).map(|i| (sc[i] - c).exp()).sum::<f64>();
                        j += 1;
                    }
                    j
                }
                Arm::Fix(b) => (b / TL_P).min(n64),
                Arm::Dense
                | Arm::Oracle(_)
                | Arm::Near
                | Arm::Grp(..)
                | Arm::Tree(..)
                | Arm::Clu(..) => unreachable!(),
            };
            for &b in &ord[..take] {
                for i in page(b) {
                    sel[i] = true;
                }
            }
            (nn + 2 * n64 + TL_P * take) as f64
        }
    };
    let mut out = vec![0f64; HD];
    let mut zs = 0f64;
    for j in (0..n).filter(|&j| sel[j]) {
        zs += w[j];
        for (o, &vv) in out.iter_mut().zip(val(j)) {
            *o += w[j] * f64::from(vv);
        }
    }
    let kept = zs / z;
    TailOut {
        o: out.iter().map(|&o| (o / zs) as f32).collect(),
        reads,
        meta,
        kept,
        viol: matches!(arm, Arm::Cert(eps) if kept < 1.0 - eps - 1e-6),
        sel,
        prox: [0.0; 2],
    }
}

/// Row `j` of the key cache of KV head `g`: the dump prefix, then the recomputed tail.
fn key_row<'a>(cx: &TailCtx<'a>, g: usize, j: usize) -> &'a [f32] {
    let (src, r) = if j < cx.pre {
        (cx.kp, j)
    } else {
        (cx.kt, j - cx.pre)
    };
    &src[(r * cx.nkv + g) * HD..(r * cx.nkv + g + 1) * HD]
}

/// Row `j` of the value cache of KV head `g`.
fn val_row<'a>(cx: &TailCtx<'a>, g: usize, j: usize) -> &'a [f32] {
    let (src, r) = if j < cx.pre {
        (cx.vp, j)
    } else {
        (cx.vt, j - cx.pre)
    };
    &src[(r * cx.nkv + g) * HD..(r * cx.nkv + g + 1) * HD]
}

/// The key set and key reads of the arms whose selection does not depend on the softmax weights
/// (`Dense`, `Near`, `Grp`, `Tree`, `Clu`); `None` for the others.
fn simple_sel(arm: Arm, p: usize, given: &[usize]) -> Option<(Vec<bool>, f64)> {
    if !matches!(
        arm,
        Arm::Dense | Arm::Near | Arm::Grp(..) | Arm::Tree(..) | Arm::Clu(..)
    ) {
        return None;
    }
    let n = p + 1;
    let mut sel = vec![false; n];
    if let Arm::Dense = arm {
        sel.fill(true);
        return Some((sel, n as f64));
    }
    let nb = TL_SINK + far_pages(p) * TL_P;
    sel[..TL_SINK].fill(true);
    sel[nb..].fill(true);
    let nn = TL_SINK + (n - nb);
    if let Arm::Near = arm {
        return Some((sel, nn as f64));
    }
    for &b in given {
        sel[TL_SINK + b * TL_P..TL_SINK + (b + 1) * TL_P].fill(true);
    }
    Some((sel, (nn + TL_P * given.len()) as f64))
}

/// Keys per tile of `tail_group`: 256 K rows and 256 V rows are 256 KiB, resident in L2 across the block's queries.
const TAIL_KT: usize = 256;
/// Scoped threads for the instrument's own serial-in-principle work (rotation, boxes, bytes, k-means assignment).
const PAR: usize = 32;

/// `tail_head` for every query head in `hs` at every position in `ps` of KV head `g`, for the arms `simple_sel`
/// accepts. Keys are visited in tiles of `TAIL_KT` rows and each tile is applied to all `hs.len() * ps.len()`
/// queries before the next, so a K or V row is read from memory once per block, not once per query. Per query the
/// arithmetic and the accumulation order over `j` are those of `tail_head` (two passes: the scores and their maximum,
/// then `exp(s - m)` summed into `z`, `zs` and the output), so the results are bit-identical.
///
/// Complexity per block: `Q * n` dot products of length `HD`, `Q * n_sel * HD` multiply-adds for the output, with
/// `Q = hs.len() * ps.len()` and `n` the context; K and V traffic `2 * n * HD` words per block against
/// `2 * Q * n * HD` per block in `tail_head`; space `Q * n` f64 scores and `Q * n` bools for the selections.
/// `KV_TAIL_RERANK=c` adds an `O(pool)` selection per query (`select_nth_unstable`), pool = the arm's far set.
/// `KV_TAIL_ORACLE` adds an `O(n)` selection (`select_nth_unstable`) per query: `Q * n` comparisons in expectation.
fn tail_group(
    cx: &TailCtx,
    arm: Arm,
    q: &[f32],
    nh: usize,
    g: usize,
    ps: std::ops::Range<usize>,
    gsets: &[Vec<Vec<usize>>],
) -> Vec<(usize, usize, TailOut)> {
    struct Qry<'a> {
        h: usize,
        i: usize,
        qh: &'a [f32],
        n: usize,
        sel: Vec<bool>,
        sc: Vec<f64>,
        ps: Vec<f64>,
        ps2: Vec<f64>,
        pw: Vec<f64>,
        reads: f64,
        m: f64,
        z: f64,
        zs: f64,
        out: Vec<f64>,
    }
    let hs = g * (nh / cx.nkv)..(g + 1) * (nh / cx.nkv);
    let sum = if matches!(arm, Arm::Grp(..)) {
        sum_mode()
    } else {
        None
    };
    let unit = sum_unit();
    let nu = cx.npg * (TL_P / unit);
    let mut qs: Vec<Qry> = Vec::new();
    for i in ps {
        for h in hs.clone() {
            let p = cx.pre + i;
            let (sel, reads) = simple_sel(arm, p, &gsets[i][g]).expect("simple arm");
            qs.push(Qry {
                h,
                i,
                qh: &q[(i * nh + h) * HD..(i * nh + h + 1) * HD],
                n: p + 1,
                sel,
                sc: vec![0f64; p + 1],
                ps: vec![0f64; far_pages(p) * (TL_P / sum_unit())],
                ps2: vec![0f64; far_pages(p) * (TL_P / sum_unit())],
                pw: vec![0f64; far_pages(p) * (TL_P / sum_unit())],
                reads,
                m: f64::NEG_INFINITY,
                z: 0.0,
                zs: 0.0,
                out: vec![0f64; HD],
            });
        }
    }
    let nmax = qs.iter().map(|x| x.n).max().unwrap_or(0);
    let avx = std::is_x86_feature_detected!("avx2");
    for j0 in (0..nmax).step_by(TAIL_KT) {
        for grp in qs.chunks_mut(4) {
            // Four queries per key row: four independent accumulator chains per row instead of one.
            let nmin = grp.iter().map(|x| x.n).min().unwrap_or(0);
            let batch = avx && grp.len() == 4;
            let split = if batch {
                nmin.min(j0 + TAIL_KT).max(j0)
            } else {
                j0
            };
            if batch {
                let qq = [grp[0].qh, grp[1].qh, grp[2].qh, grp[3].qh];
                for j in j0..split {
                    // SAFETY: avx2 was detected above.
                    let d = unsafe { dot32x4_avx(qq, key_row(cx, g, j)) };
                    for (x, &dv) in grp.iter_mut().zip(&d) {
                        let s = f64::from(dv) * f64::from(SCALE);
                        x.sc[j] = s;
                        x.m = x.m.max(s);
                    }
                }
            }
            for x in grp.iter_mut() {
                for j in split..(j0 + TAIL_KT).min(x.n) {
                    let s = f64::from(dot32(x.qh, key_row(cx, g, j))) * f64::from(SCALE);
                    x.sc[j] = s;
                    x.m = x.m.max(s);
                }
            }
        }
    }
    if matches!(arm, Arm::Grp(..)) && std::env::var_os("KV_TAIL_ORACLE").is_some() {
        // Diagnostic: each query keeps its own exact top-`c` far tokens, `c` the count the page set selected.
        for x in &mut qs {
            let nb = TL_SINK + far_pages(x.n - 1) * TL_P;
            let c = x.sel[TL_SINK..nb].iter().filter(|&&b| b).count();
            x.sel[TL_SINK..nb].fill(false);
            if c > 0 {
                let mut ord: Vec<usize> = (TL_SINK..nb).collect();
                ord.select_nth_unstable_by(c - 1, |&a, &b| x.sc[b].total_cmp(&x.sc[a]));
                for &j in &ord[..c] {
                    x.sel[j] = true;
                }
            }
        }
    }
    if let Some(c) = std::env::var("KV_TAIL_RERANK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|_| matches!(arm, Arm::Grp(..)))
    {
        // Diagnostic: two-stage gather. The page set is the candidate pool; each query keeps its own exact top-`c`
        // far tokens inside the pool. Pool scoring is not charged to the byte count.
        for x in &mut qs {
            let nb = TL_SINK + far_pages(x.n - 1) * TL_P;
            let mut pool: Vec<usize> = (TL_SINK..nb).filter(|&j| x.sel[j]).collect();
            if pool.len() > c {
                pool.select_nth_unstable_by(c - 1, |&a, &b| x.sc[b].total_cmp(&x.sc[a]));
                x.sel[TL_SINK..nb].fill(false);
                for &j in &pool[..c] {
                    x.sel[j] = true;
                }
            }
        }
    }
    for j0 in (0..nmax).step_by(TAIL_KT) {
        for x in &mut qs {
            for j in j0..(j0 + TAIL_KT).min(x.n) {
                let w = (x.sc[j] - x.m).exp();
                x.z += w;
                if sum.is_some() && !x.sel[j] && j >= TL_SINK && j < TL_SINK + x.ps.len() * unit {
                    let b = (j - TL_SINK) / unit;
                    x.ps[b] += x.sc[j];
                    x.ps2[b] += x.sc[j] * x.sc[j];
                    x.pw[b] += w;
                }
                if x.sel[j] {
                    x.zs += w;
                    for (o, &vv) in x.out.iter_mut().zip(val_row(cx, g, j)) {
                        *o += w * f64::from(vv);
                    }
                }
            }
        }
    }
    if let Some(mode) = sum {
        let pf = unit as f64;
        for x in &mut qs {
            for b in 0..x.ps.len() {
                if x.sel[TL_SINK + b * unit] {
                    continue;
                }
                let mu = x.ps[b] / pf;
                let mass = match mode {
                    0 => pf * (mu - x.m).exp(),
                    1 => x.pw[b],
                    _ => {
                        let var = (x.ps2[b] / pf - mu * mu).max(0.0);
                        pf * (mu + 0.5 * var - x.m).exp()
                    }
                };
                x.zs += mass;
                let vb = &cx.vbar[(g * nu + b) * HD..(g * nu + b + 1) * HD];
                for (o, &vv) in x.out.iter_mut().zip(vb) {
                    *o += mass * f64::from(vv);
                }
            }
        }
    }
    let want_pos = std::env::var_os("KV_TAIL_POS").is_some();
    qs.into_iter()
        .map(|x| {
            let prox = if want_pos {
                unread_proxy(cx, &x.sel, x.qh, g, x.m, x.zs, &x.sc)
            } else {
                [0.0; 2]
            };
            let kept = x.zs / x.z;
            let o = x.out.iter().map(|&o| (o / x.zs) as f32).collect();
            (
                x.h,
                x.i,
                TailOut {
                    o,
                    reads: x.reads,
                    meta: 0.0,
                    kept,
                    viol: false,
                    prox,
                    sel: x.sel,
                },
            )
        })
        .collect()
}

/// The deployable page-bound proxies of one query: `ln(1 + B/Z_read)` with `B = sum 64 exp(u_b - m)` over the
/// far pages not read and `Z_read` the read softmax mass (`zs`, relative to the maximum score `m`), and the best
/// unread bound minus the best read score. Both use only page bounds and read scores.
///
/// Complexity: time `O(n64 HD)` multiply-adds (one bound per far page), space `O(1)`.
fn unread_proxy(
    cx: &TailCtx,
    sel: &[bool],
    qh: &[f32],
    g: usize,
    m: f64,
    zs: f64,
    sc: &[f64],
) -> [f64; 2] {
    let n = sc.len();
    let n64 = far_pages(n - 1);
    let (mut bmass, mut bmax) = (0f64, f64::NEG_INFINITY);
    for b in 0..n64 {
        if sel[TL_SINK + b * TL_P] {
            continue;
        }
        let u = page_bound(cx, qh, g, b);
        bmass += TL_P as f64 * (u - m).exp();
        bmax = bmax.max(u);
    }
    let best_read = (0..n)
        .filter(|&j| sel[j])
        .map(|j| sc[j])
        .fold(f64::NEG_INFINITY, f64::max);
    [(1.0 + bmass / zs).ln(), bmax - best_read]
}

/// `Grp(b, r)` page sets for every tail position `i` and KV head `g`: at `i % r == 0` the far pages
/// are ranked by the highest Quest bound over the group's query heads (lowest first under `rev`, the
/// mutation) and the top `b / 64` kept; between refreshes the set is reused, plus every page sealed since.
/// Returns the sets and the bound vectors read per (position, KV head): `2 n64` at a refresh, else 0.
///
/// Complexity: per refresh `O(gs n64 HD)` multiply-adds and an `O(n64 log n64)` sort per KV head;
/// space `O(nt nkv b/64)` page indices.
fn group_pages(
    cx: &TailCtx,
    q: &[f32],
    nh: usize,
    nt: usize,
    b: usize,
    r: usize,
    rev: bool,
) -> (Vec<Vec<Vec<usize>>>, Vec<Vec<f64>>) {
    let gs = nh / cx.nkv;
    let mut sets = vec![vec![Vec::new(); cx.nkv]; nt];
    let mut meta = vec![vec![0f64; cx.nkv]; nt];
    for g in 0..cx.nkv {
        let (mut cur, mut n64r) = (Vec::new(), 0usize);
        for i in 0..nt {
            let n64 = far_pages(cx.pre + i);
            if i % r == 0 {
                let u: Vec<f64> = (0..n64)
                    .map(|pg| {
                        (g * gs..(g + 1) * gs)
                            .map(|h| {
                                page_bound(cx, &q[(i * nh + h) * HD..(i * nh + h + 1) * HD], g, pg)
                            })
                            .fold(f64::NEG_INFINITY, f64::max)
                    })
                    .collect();
                let mut ord: Vec<usize> = (0..n64).collect();
                if rev {
                    ord.sort_by(|&a, &c| u[a].total_cmp(&u[c]));
                } else {
                    ord.sort_by(|&a, &c| u[c].total_cmp(&u[a]));
                }
                ord.truncate((b / TL_P).min(n64));
                cur = ord;
                n64r = n64;
                meta[i][g] = (2 * n64) as f64;
            } else {
                cur.extend(n64r..n64);
                n64r = n64;
            }
            sets[i][g].clone_from(&cur);
        }
    }
    (sets, meta)
}

/// Quest's bound on `q.k / 8` over the element-wise box `[lo, hi]`, maximised over the query heads `hs`.
fn group_box_bound(q: &[f32], hs: std::ops::Range<usize>, lo: &[f32], hi: &[f32]) -> f64 {
    hs.map(|h| {
        let qh = &q[h * HD..(h + 1) * HD];
        (0..HD)
            .map(|i| {
                let (a, c) = (f64::from(qh[i]), (f64::from(lo[i]), f64::from(hi[i])));
                (a * c.0).max(a * c.1)
            })
            .sum::<f64>()
            * f64::from(SCALE)
    })
    .fold(f64::NEG_INFINITY, f64::max)
}

/// The `k` entries of `cand` with the highest `score` (lowest under `rev`, the mutation).
fn top_by(mut cand: Vec<usize>, score: impl Fn(usize) -> f64, k: usize, rev: bool) -> Vec<usize> {
    let sc: std::collections::HashMap<usize, f64> = cand.iter().map(|&c| (c, score(c))).collect();
    if rev {
        cand.sort_by(|a, c| sc[a].total_cmp(&sc[c]));
    } else {
        cand.sort_by(|a, c| sc[c].total_cmp(&sc[a]));
    }
    cand.truncate(k);
    cand
}

/// Element-wise boxes of consecutive groups of `TR_F` boxes (`lo`, `hi`: `n x HD`).
fn merge_boxes(lo: &[f32], hi: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = lo.len() / HD;
    let m = n.div_ceil(TR_F);
    let (mut l2, mut h2) = (vec![f32::INFINITY; m * HD], vec![f32::NEG_INFINITY; m * HD]);
    for c in 0..n {
        for i in 0..HD {
            let o = (c / TR_F) * HD + i;
            l2[o] = l2[o].min(lo[c * HD + i]);
            h2[o] = h2[o].max(hi[c * HD + i]);
        }
    }
    (l2, h2)
}

/// Fanout of the page index: 16 children per tree node, `ceil(n64 / 16)` clusters.
const TR_F: usize = 16;

/// `Tree(b, r, s)` and `Clu(b, r, s)` page sets: `group_pages` with the flat scan of every page bound
/// replaced by an index. `Tree`: the boxes of 16, 256, ... consecutive pages; from the top level (at most
/// 16 nodes, all read) keep the `s` highest group bounds and read their children, down to the pages.
/// `Clu`: pages k-means-clustered (8 Lloyd steps, Euclidean, on the box centre) into `ceil(n64 / 16)`
/// clusters at the first refresh, each later-sealed page inserted into the nearest centroid's cluster
/// (centroids fixed after the first refresh); read every cluster box, keep `s`, read their pages. Either way the top
/// `b / 64` pages are kept and every page sealed since is added between refreshes. Returns the sets, the
/// bound vectors read per (position, KV head), and per refresh the share of the flat scan's top `b / 64`
/// pages recovered (the flat ranking is the instrument, not the arm's cost).
///
/// Complexity: tree reads `2 (n_top + 16 s L)` bound vectors per refresh for `L = ceil(log16 n64) - 1`
/// expanded levels; cluster reads `2 (K + |members of s clusters|)`; instrument time `O(gs n64 HD)` per
/// refresh for the recall reference plus `O(8 n64 K HD)` per KV head for the clustering; space
/// `O(n64 HD)` words for the boxes.
#[allow(clippy::too_many_lines)]
fn index_pages(
    cx: &TailCtx,
    q: &[f32],
    nh: usize,
    nt: usize,
    arm: Arm,
    rev: bool,
) -> (Vec<Vec<Vec<usize>>>, Vec<Vec<f64>>, Vec<f64>) {
    let (b, r, s, clu) = match arm {
        Arm::Tree(b, r, s) => (b, r, s, false),
        Arm::Clu(b, r, s) => (b, r, s, true),
        _ => unreachable!(),
    };
    let gs = nh / cx.nkv;
    let keep = b / TL_P;
    let per_g = |g: usize| -> (Vec<Vec<usize>>, Vec<f64>, Vec<f64>) {
        let mut sets_g = vec![Vec::new(); nt];
        let mut meta_g = vec![0f64; nt];
        let mut recall_g = Vec::new();
        let hs = g * gs..(g + 1) * gs;
        let pl = |pg: usize| (g * cx.npg + pg) * HD;
        let mut members: Vec<Vec<usize>> = Vec::new();
        let (mut clo, mut chi, mut cen) = (Vec::new(), Vec::new(), Vec::new());
        let mut ncl = 0usize;
        let (mut cur, mut n64r) = (Vec::new(), 0usize);
        for i in 0..nt {
            let n64 = far_pages(cx.pre + i);
            if i % r != 0 {
                cur.extend(n64r..n64);
                n64r = n64;
                sets_g[i].clone_from(&cur);
                continue;
            }
            let qi = &q[i * nh * HD..(i + 1) * nh * HD];
            let pbound = |pg: usize| {
                group_box_bound(
                    qi,
                    hs.clone(),
                    &cx.mn[pl(pg)..pl(pg) + HD],
                    &cx.mx[pl(pg)..pl(pg) + HD],
                )
            };
            let flat = top_by((0..n64).collect(), pbound, keep, false);
            let mut rd = 0usize;
            let chosen = if clu {
                if members.is_empty() {
                    let k = n64.div_ceil(TR_F);
                    let ctr: Vec<f32> = (0..n64)
                        .flat_map(|pg| (0..HD).map(move |e| (pg, e)))
                        .map(|(pg, e)| 0.5 * (cx.mn[pl(pg) + e] + cx.mx[pl(pg) + e]))
                        .collect();
                    cen = (0..k)
                        .flat_map(|c| ctr[(c * n64 / k) * HD..(c * n64 / k + 1) * HD].to_vec())
                        .collect();
                    let mut asg = vec![0usize; n64];
                    for _ in 0..8 {
                        par_rows(&mut asg, 1, 8, |p0, ch| {
                            for (o, a) in ch.iter_mut().enumerate() {
                                let pg = p0 + o;
                                let d2 = |c: usize| -> f32 {
                                    (0..HD)
                                        .map(|e| (ctr[pg * HD + e] - cen[c * HD + e]).powi(2))
                                        .sum()
                                };
                                *a = (0..k).min_by(|&a, &c| d2(a).total_cmp(&d2(c))).unwrap();
                            }
                        });
                        let mut sum = vec![0f32; k * HD];
                        let mut cnt = vec![0usize; k];
                        for pg in 0..n64 {
                            cnt[asg[pg]] += 1;
                            for e in 0..HD {
                                sum[asg[pg] * HD + e] += ctr[pg * HD + e];
                            }
                        }
                        for c in (0..k).filter(|&c| cnt[c] > 0) {
                            for e in 0..HD {
                                cen[c * HD + e] = sum[c * HD + e] / cnt[c] as f32;
                            }
                        }
                    }
                    members = vec![Vec::new(); k];
                    clo = vec![f32::INFINITY; k * HD];
                    chi = vec![f32::NEG_INFINITY; k * HD];
                    for pg in 0..n64 {
                        let c = asg[pg];
                        members[c].push(pg);
                        for e in 0..HD {
                            clo[c * HD + e] = clo[c * HD + e].min(cx.mn[pl(pg) + e]);
                            chi[c * HD + e] = chi[c * HD + e].max(cx.mx[pl(pg) + e]);
                        }
                    }
                    ncl = n64;
                }
                for pg in ncl..n64 {
                    let d2 = |c: usize| -> f32 {
                        (0..HD)
                            .map(|e| {
                                (0.5 * (cx.mn[pl(pg) + e] + cx.mx[pl(pg) + e]) - cen[c * HD + e])
                                    .powi(2)
                            })
                            .sum()
                    };
                    let c = (0..members.len())
                        .min_by(|&a, &c| d2(a).total_cmp(&d2(c)))
                        .unwrap();
                    members[c].push(pg);
                    for e in 0..HD {
                        clo[c * HD + e] = clo[c * HD + e].min(cx.mn[pl(pg) + e]);
                        chi[c * HD + e] = chi[c * HD + e].max(cx.mx[pl(pg) + e]);
                    }
                }
                ncl = ncl.max(n64);
                let live: Vec<usize> = (0..members.len())
                    .filter(|&c| !members[c].is_empty())
                    .collect();
                rd += live.len();
                let cb = |c: usize| {
                    group_box_bound(
                        qi,
                        hs.clone(),
                        &clo[c * HD..(c + 1) * HD],
                        &chi[c * HD..(c + 1) * HD],
                    )
                };
                let cand: Vec<usize> = top_by(live, cb, s, rev)
                    .into_iter()
                    .flat_map(|c| members[c].iter().copied().filter(|&pg| pg < n64))
                    .collect();
                rd += cand.len();
                top_by(cand, pbound, keep, rev)
            } else {
                let mut lv: Vec<(Vec<f32>, Vec<f32>)> = vec![(
                    (0..n64)
                        .flat_map(|pg| cx.mn[pl(pg)..pl(pg) + HD].to_vec())
                        .collect(),
                    (0..n64)
                        .flat_map(|pg| cx.mx[pl(pg)..pl(pg) + HD].to_vec())
                        .collect(),
                )];
                while lv[lv.len() - 1].0.len() / HD > TR_F {
                    let (lo, hi) = &lv[lv.len() - 1];
                    let up = merge_boxes(lo, hi);
                    lv.push(up);
                }
                let top = lv.len() - 1;
                let mut front: Vec<usize> = (0..lv[top].0.len() / HD).collect();
                rd += front.len();
                for k in (1..=top).rev() {
                    let (lo, hi) = &lv[k];
                    let nb = |c: usize| {
                        group_box_bound(
                            qi,
                            hs.clone(),
                            &lo[c * HD..(c + 1) * HD],
                            &hi[c * HD..(c + 1) * HD],
                        )
                    };
                    let n_below = lv[k - 1].0.len() / HD;
                    front = top_by(front, nb, s, rev)
                        .into_iter()
                        .flat_map(|c| c * TR_F..((c + 1) * TR_F).min(n_below))
                        .collect();
                    rd += front.len();
                }
                top_by(front, pbound, keep, rev)
            };
            recall_g.push(if flat.is_empty() {
                1.0
            } else {
                chosen.iter().filter(|c| flat.contains(c)).count() as f64 / flat.len() as f64
            });
            meta_g[i] = (2 * rd) as f64;
            cur = chosen;
            n64r = n64;
            sets_g[i].clone_from(&cur);
        }
        (sets_g, meta_g, recall_g)
    };
    let parts: Vec<_> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..cx.nkv)
            .map(|g| {
                sc.spawn({
                    let f = &per_g;
                    move || f(g)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut sets = vec![vec![Vec::new(); cx.nkv]; nt];
    let mut meta = vec![vec![0f64; cx.nkv]; nt];
    let mut recall = Vec::new();
    for (g, (sg, mg, rg)) in parts.into_iter().enumerate() {
        for (i, (sv, mv)) in sg.into_iter().zip(mg).enumerate() {
            sets[i][g] = sv;
            meta[i][g] = mv;
        }
        recall.extend(rg);
    }
    (sets, meta, recall)
}

/// `KV_TAIL=<dump dir>` (attention-cut T5): whole-model next-token KLD with page selection in the decode tail.
/// The `KV_DUMP` directory at `KV_T=32768` is the dense prefill cache. For `t` in {8192, 32768} the
/// last 8 positions are recomputed from token embeddings through all layers once per arm; each tail
/// query attends to the dump's prefix keys `[0, t-8)` and the arm's own tail keys. Control: the dense
/// arm's unrotated tail q, k, v must equal the dump's rows (max rel-L2 <= 1e-4), and the certified arm
/// must keep mass >= 1-eps in every head; either failure exits 1. `HATTN_MUT=1` replaces Quest's bound
/// by the min-only `q.m`, which is not a bound, and must break the certification.
///
/// Cost per `t`: `6 arms x 8 positions x` one layer stack of projections, plus attention
/// `6 x 24 x 8 x n_head x t x 64` multiply-adds twice (scores, values); memory: one layer's prefix k, v
/// (`t x n_head_kv x 64` f32 each) and the dequantised LM head (`n_embd x vocab` f32).
#[allow(clippy::too_many_lines)]
fn run_tail(path: &str, dir: &std::path::Path) -> Result<bool, Box<dyn Error>> {
    let nth = 16usize;
    let t0 = std::time::Instant::now();
    let mutb = std::env::var("HATTN_MUT").as_deref() == Ok("1");
    let rev = std::env::var("HATTN_MUT").as_deref() == Ok("2");
    let eps = 1.0 - (-0.05f64).exp();
    let ekl = |kl: f64| 1.0 - (-kl).exp();
    let (arms, tails, nt): (Vec<Arm>, Vec<usize>, usize) =
        if std::env::var_os("KV_TAIL_TREE").is_some() {
            (
                vec![
                    Arm::Dense,
                    Arm::Grp(2048, 8),
                    Arm::Tree(2048, 8, 4),
                    Arm::Tree(2048, 8, 8),
                    Arm::Clu(2048, 8, 4),
                    Arm::Clu(2048, 8, 8),
                ],
                vec![4096, 8192, 16384, 32768],
                64,
            )
        } else if std::env::var_os("KV_TAIL_GROUP").is_some() {
            (
                vec![
                    Arm::Dense,
                    Arm::Fix(1024),
                    Arm::Fix(2048),
                    Arm::Grp(1024, 1),
                    Arm::Grp(2048, 1),
                    Arm::Grp(1024, 8),
                    Arm::Grp(2048, 8),
                    Arm::Oracle(ekl(0.25)),
                ],
                vec![4096, 8192, 16384, 32768],
                64,
            )
        } else if std::env::var_os("KV_TAIL_SWEEP").is_some() {
            (
                vec![
                    Arm::Dense,
                    Arm::Near,
                    Arm::Fix(256),
                    Arm::Fix(512),
                    Arm::Fix(1024),
                    Arm::Fix(2048),
                    Arm::Oracle(ekl(0.25)),
                    Arm::Oracle(ekl(1.0)),
                    Arm::Order(ekl(1.0)),
                ],
                vec![4096, 8192, 16384, 32768],
                64,
            )
        } else {
            (
                vec![
                    Arm::Dense,
                    Arm::Oracle(eps),
                    Arm::Near,
                    Arm::Order(eps),
                    Arm::Cert(eps),
                    Arm::Fix(2048),
                ],
                vec![8192, 32768],
                8,
            )
        };
    // `KV_TAIL_FARB=n` replaces the 2048-token far budget of every indexed arm with `n`.
    let arms: Vec<Arm> = match std::env::var("KV_TAIL_FARB") {
        Ok(v) => {
            let n: usize = v.parse().map_err(|e| format!("KV_TAIL_FARB: {e}"))?;
            arms.into_iter()
                .map(|a| match a {
                    Arm::Grp(_, r) => Arm::Grp(n, r),
                    Arm::Tree(_, r, s) => Arm::Tree(n, r, s),
                    Arm::Clu(_, r, s) => Arm::Clu(n, r, s),
                    other => other,
                })
                .collect()
        }
        Err(_) => arms,
    };
    // `KV_TAIL_ARMS=2,4` keeps the dense reference (index 0) and the listed arms: each arm runs its own whole-model
    // forward, so a subset gives the same per-arm figures as the full list.
    let arms = match std::env::var("KV_TAIL_ARMS") {
        Ok(v) => {
            let keep: Vec<usize> = v
                .split(',')
                .map(str::parse)
                .collect::<Result<_, _>>()
                .map_err(|e| format!("KV_TAIL_ARMS: {e}"))?;
            arms.into_iter()
                .enumerate()
                .filter(|(i, _)| *i == 0 || keep.contains(i))
                .map(|(_, a)| a)
                .collect()
        }
        Err(_) => arms,
    };
    assert!(
        nt <= TL_W,
        "tail longer than the window: far pages would hold tail keys"
    );
    // `KV_TAIL_TS=8192,32768` runs exactly the listed contexts (each at most the dump's length), so a larger model's run splits into bounded calls.
    let tails: Vec<usize> = match std::env::var("KV_TAIL_TS") {
        Ok(v) => {
            let keep: Vec<usize> = v
                .split(',')
                .map(str::parse)
                .collect::<Result<_, _>>()
                .map_err(|e| format!("KV_TAIL_TS: {e}"))?;
            keep
        }
        Err(_) => tails,
    };
    // `KV_QTAIL=t1,t2,..` (the `gpu_dump` option): `q{l}.bin` holds only rows `[t-nt, t)` for each listed `t`, in
    // that order, so the block for `t` starts at row `index * nt`.
    let qtail: Option<Vec<usize>> = match std::env::var("KV_QTAIL") {
        Ok(v) => Some(
            v.split(',')
                .map(str::parse)
                .collect::<Result<_, _>>()
                .map_err(|e| format!("KV_QTAIL: {e}"))?,
        ),
        Err(_) => None,
    };
    let eng = Engine::open(path)?;
    let cfg = &eng.config;
    let (nl, nh, nkv, d) = (cfg.n_layer, cfg.n_head, cfg.n_head_kv, cfg.n_embd);
    assert_eq!(cfg.head_dim, HD);
    let half = HD / 2;
    let ids: Vec<u32> = std::fs::read(dir.join("ids.bin"))?
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let cs: Vec<(f64, f64)> = std::fs::read(dir.join("cs.bin"))?
        .chunks_exact(16)
        .map(|c| {
            let f = |o: usize| f64::from_le_bytes(c[o..o + 8].try_into().unwrap());
            (f(0), f(8))
        })
        .collect();
    let tn = ids.len();
    assert_eq!(cs.len(), tn * half);
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
    let rel = |a: &[f32], b: &[f32]| {
        let (mut num, mut den) = (0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (f64::from(x) - f64::from(y)).powi(2);
            den += f64::from(y).powi(2);
        }
        (num / den).sqrt()
    };
    let vecs = eng.decode_vectors();
    let reps = cfg.rms_eps;
    let gout = q32_vec(vecs.output_norm.as_deref().ok_or("output_norm")?);
    let lm_name = eng.lm_head().name.clone();
    let (wlm, lni, lno) = weight(&eng, &lm_name)?;
    println!(
        "tail: {tn}-token dump {}, LM head {lm_name} {lni}x{lno}; eps {eps:.5}; HATTN_MUT {mutb}",
        dir.display()
    );
    let na = arms.len();
    let mut ok = true;
    for &t in &tails {
        if t > tn {
            return Err(format!("dump has {tn} tokens < {t}").into());
        }
        let pre = t - nt;
        let mut emb = Vec::with_capacity(nt * d);
        for &id in &ids[pre..t] {
            emb.extend(q32_vec(&eng.token_embedding(id)?));
        }
        let mut xs = vec![emb; na];
        let mut reads: Vec<Vec<f64>> = vec![Vec::new(); na];
        let mut bytes: Vec<Vec<f64>> = vec![Vec::new(); na];
        let mut hkl: Vec<Vec<f64>> = vec![Vec::new(); na];
        let mut rec: Vec<Vec<f64>> = vec![Vec::new(); na];
        let mut rmeta: Vec<Vec<f64>> = vec![Vec::new(); na];
        // Per arm and tail position: sums of head KL, bound-mass proxy and gap proxy, their maxima, and the count.
        let mut pos: Vec<Vec<[f64; 7]>> = vec![vec![[0.0; 7]; nt]; na];
        let mut viol = 0usize;
        let mut ctl = 0f64;
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
            let w = |s: &str| weight(&eng, &format!("blk.{l}.{s}.weight"));
            let (wq, wk, wv, wo) = (w("attn_q")?, w("attn_k")?, w("attn_v")?, w("attn_output")?);
            let (wg, wu, wd) = (w("ffn_gate")?, w("ffn_up")?, w("ffn_down")?);
            let kvw = nkv * HD;
            let mut kp = rows_f32(&dir.join(format!("k{l}.bin")), kvw, 0, pre)?;
            par_rows(&mut kp, kvw, PAR, |j0, ch| {
                for (o, row) in ch.chunks_mut(kvw).enumerate() {
                    rot(row, nkv, j0 + o);
                }
            });
            let vp = rows_f32(&dir.join(format!("v{l}.bin")), kvw, 0, pre)?;
            let qpre = match &qtail {
                Some(l) => l.iter().position(|&x| x == t).ok_or("KV_QTAIL lacks t")? * nt,
                None => pre,
            };
            let dq = rows_f32(&dir.join(format!("q{l}.bin")), nh * HD, qpre, qpre + nt)?;
            let dk = rows_f32(&dir.join(format!("k{l}.bin")), kvw, pre, t)?;
            let dv = rows_f32(&dir.join(format!("v{l}.bin")), kvw, pre, t)?;
            let npg = (t - TL_W - TL_SINK) / TL_P;
            let mut mn = vec![f32::INFINITY; nkv * npg * HD];
            let mut mx = vec![f32::NEG_INFINITY; nkv * npg * HD];
            let boxes = |ext: fn(f32, f32) -> f32, dst: &mut [f32]| {
                par_rows(dst, HD, PAR, |i0, ch| {
                    for (o, bx) in ch.chunks_mut(HD).enumerate() {
                        let (g, b) = ((i0 + o) / npg, (i0 + o) % npg);
                        for j in TL_SINK + b * TL_P..TL_SINK + (b + 1) * TL_P {
                            let kr = &kp[(j * nkv + g) * HD..(j * nkv + g + 1) * HD];
                            for (e, &x) in bx.iter_mut().zip(kr) {
                                *e = ext(*e, x);
                            }
                        }
                    }
                });
            };
            boxes(f32::min, &mut mn);
            boxes(f32::max, &mut mx);
            let mut vbar: Vec<f32> = Vec::new();
            if sum_mode().is_some() {
                let unit = sum_unit();
                let nu = npg * (TL_P / unit);
                vbar = vec![0f32; nkv * nu * HD];
                par_rows(&mut vbar, HD, PAR, |i0, ch| {
                    for (o, row) in ch.chunks_mut(HD).enumerate() {
                        let (g, b) = ((i0 + o) / nu, (i0 + o) % nu);
                        for j in TL_SINK + b * unit..TL_SINK + (b + 1) * unit {
                            let vr = &vp[(j * nkv + g) * HD..(j * nkv + g + 1) * HD];
                            for (e, &x) in row.iter_mut().zip(vr) {
                                *e += x / unit as f32;
                            }
                        }
                    }
                });
            }
            for (a, x) in xs.iter_mut().enumerate() {
                let xn = rmsnorm(x, &g1, reps, d);
                let mut q = gemm(&wq.0, wq.2, wq.1, &xn, nt, bq.as_deref(), nth);
                let mut k = gemm(&wk.0, wk.2, wk.1, &xn, nt, bkk.as_deref(), nth);
                let v = gemm(&wv.0, wv.2, wv.1, &xn, nt, bv.as_deref(), nth);
                if a == 0 {
                    ctl = ctl.max(rel(&q, &dq)).max(rel(&k, &dk)).max(rel(&v, &dv));
                }
                for i in 0..nt {
                    rot(&mut q[i * nh * HD..(i + 1) * nh * HD], nh, pre + i);
                    rot(&mut k[i * kvw..(i + 1) * kvw], nkv, pre + i);
                }
                let cx = TailCtx {
                    vbar: &vbar,
                    pre,
                    nkv,
                    kp: &kp,
                    vp: &vp,
                    kt: &k,
                    vt: &v,
                    npg,
                    mn: &mn,
                    mx: &mx,
                    mutb,
                };
                let arm = layer_arm(l, arms[a]);
                let (gsets, gmeta) = match arm {
                    Arm::Grp(b, r) => group_pages(&cx, &q, nh, nt, b, r, rev),
                    Arm::Tree(..) | Arm::Clu(..) => {
                        let (st, mt, rc) = index_pages(&cx, &q, nh, nt, arm, rev);
                        rec[a].extend(rc);
                        (st, mt)
                    }
                    _ => (vec![vec![Vec::new(); nkv]; nt], vec![vec![0f64; nkv]; nt]),
                };
                rmeta[a].extend(gmeta.iter().flatten().copied().filter(|&m| m > 0.0));
                let res: Vec<Vec<TailOut>> = if simple_sel(arm, pre, &[]).is_some()
                    && std::env::var("KV_TAIL_SLOW").is_err()
                {
                    let pb = 2;
                    let units: Vec<(usize, usize)> = (0..nkv)
                        .flat_map(|g| (0..nt).step_by(pb).map(move |i0| (g, i0)))
                        .collect();
                    let next = std::sync::atomic::AtomicUsize::new(0);
                    let workers = std::thread::available_parallelism()
                        .map_or(8, usize::from)
                        .min(units.len());
                    let outs: Vec<Vec<(usize, usize, TailOut)>> = std::thread::scope(|sc| {
                        let jobs: Vec<_> = (0..workers)
                            .map(|_| {
                                let (cx, q, gsets, units, next) = (&cx, &q, &gsets, &units, &next);
                                sc.spawn(move || {
                                    let mut mine = Vec::new();
                                    loop {
                                        let u =
                                            next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        let Some(&(g, i0)) = units.get(u) else { break };
                                        mine.extend(tail_group(
                                            cx,
                                            arm,
                                            q,
                                            nh,
                                            g,
                                            i0..(i0 + pb).min(nt),
                                            gsets,
                                        ));
                                    }
                                    mine
                                })
                            })
                            .collect();
                        jobs.into_iter().map(|j| j.join().unwrap()).collect()
                    });
                    let mut slots: Vec<Vec<Option<TailOut>>> =
                        (0..nh).map(|_| (0..nt).map(|_| None).collect()).collect();
                    for (h, i, to) in outs.into_iter().flatten() {
                        slots[h][i] = Some(to);
                    }
                    slots
                        .into_iter()
                        .map(|r| r.into_iter().map(|t| t.unwrap()).collect())
                        .collect()
                } else {
                    std::thread::scope(|sc| {
                        let hs: Vec<_> = (0..nh)
                            .map(|h| {
                                let (cx, q, gsets) = (&cx, &q, &gsets);
                                sc.spawn(move || {
                                    (0..nt)
                                        .map(|i| {
                                            let qh = &q[(i * nh + h) * HD..(i * nh + h + 1) * HD];
                                            let g = h / (nh / nkv);
                                            tail_head(cx, arm, qh, g, pre + i, &gsets[i][g])
                                        })
                                        .collect()
                                })
                            })
                            .collect();
                        hs.into_iter().map(|j| j.join().unwrap()).collect()
                    })
                };
                let gs = nh / nkv;
                let mut bv = vec![0f64; nt * nkv];
                par_rows(&mut bv, 1, PAR, |u0, ch| {
                    for (o, slot) in ch.iter_mut().enumerate() {
                        let (i, g) = ((u0 + o) / nkv, (u0 + o) % nkv);
                        let n = pre + i + 1;
                        let used = (0..n)
                            .filter(|&j| (g * gs..(g + 1) * gs).any(|h| res[h][i].sel[j]))
                            .count();
                        let meta = res[g * gs][i].meta + gmeta[i][g];
                        *slot = (2 * used) as f64 / (2 * n) as f64 + meta / (2 * n) as f64;
                    }
                });
                bytes[a].extend(bv);
                let mut att = vec![0f32; nt * nh * HD];
                for (h, rh) in res.into_iter().enumerate() {
                    for (i, to) in rh.into_iter().enumerate() {
                        att[(i * nh + h) * HD..(i * nh + h + 1) * HD].copy_from_slice(&to.o);
                        reads[a].push(to.reads / (pre + i + 1) as f64);
                        hkl[a].push(-to.kept.ln());
                        let e = &mut pos[a][i];
                        e[0] += -to.kept.ln();
                        e[1] += to.prox[0];
                        e[2] = e[2].max(to.prox[0]);
                        e[3] += to.prox[1];
                        e[4] = e[4].max(to.prox[1]);
                        e[5] += 1.0;
                        e[6] = e[6].max(-to.kept.ln());
                        viol += usize::from(to.viol);
                    }
                }
                let o = gemm(&wo.0, wo.2, wo.1, &att, nt, None, nth);
                for (xi, oi) in x.iter_mut().zip(&o) {
                    *xi += oi;
                }
                let xn2 = rmsnorm(x, &g2, reps, d);
                let mut gate = gemm(&wg.0, wg.2, wg.1, &xn2, nt, None, nth);
                let up = gemm(&wu.0, wu.2, wu.1, &xn2, nt, None, nth);
                for (gi, ui) in gate.iter_mut().zip(&up) {
                    let gv = f64::from(*gi);
                    *gi = (gv / (1.0 + (-gv).exp()) * f64::from(*ui)) as f32;
                }
                let dn = gemm(&wd.0, wd.2, wd.1, &gate, nt, None, nth);
                for (xi, di) in x.iter_mut().zip(&dn) {
                    *xi += di;
                }
            }
        }
        let xa: Vec<f32> = xs.iter().flat_map(|x| rmsnorm(x, &gout, reps, d)).collect();
        let logits = gemm(&wlm, lno, lni, &xa, na * nt, None, nth);
        let lsm = |r: &[f32]| {
            let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lz = r.iter().map(|&v| f64::from(v - m).exp()).sum::<f64>().ln();
            r.iter()
                .map(|&v| f64::from(v - m) - lz)
                .collect::<Vec<f64>>()
        };
        let argmax = |r: &[f32]| (0..r.len()).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap();
        println!(
            "\n== t={t}: dense-tail control max rel-L2 (q,k,v, {nl} layers) {ctl:.3e}; certification violations {viol} =="
        );
        let dense_bytes = arms
            .iter()
            .zip(&bytes)
            .filter(|(arm, _)| matches!(arm, Arm::Dense))
            .all(|(_, b)| b.iter().all(|&x| x == 1.0));
        println!("dense bytes/dense = 1 at every (KV head, position): {dense_bytes}");
        ok &= ctl <= 1e-4 && viol == 0 && dense_bytes;
        for a in 0..na {
            let mut kl: Vec<f64> = Vec::new();
            let mut top = 0;
            for i in 0..nt {
                let rd = &logits[i * lno..(i + 1) * lno];
                let ra = &logits[(a * nt + i) * lno..(a * nt + i + 1) * lno];
                let (pd, pa) = (lsm(rd), lsm(ra));
                kl.push(
                    pd.iter()
                        .zip(&pa)
                        .map(|(&x, &y)| x.exp() * (x - y))
                        .sum::<f64>(),
                );
                top += usize::from(argmax(rd) == argmax(ra));
            }
            let mut r = reads[a].clone();
            r.sort_by(f64::total_cmp);
            let mut hk = hkl[a].clone();
            hk.sort_by(f64::total_cmp);
            let q = |v: &[f64], f: f64| v[((v.len() - 1) as f64 * f) as usize];
            let shown = if nt <= 8 {
                format!(
                    " [{}]",
                    kl.iter()
                        .map(|v| format!("{v:.3}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            } else {
                let mut ks = kl.clone();
                ks.sort_by(f64::total_cmp);
                format!(
                    " median {:.4} p90 {:.4}; > 0.05 at {}",
                    q(&ks, 0.5),
                    q(&ks, 0.9),
                    ks.iter().filter(|&&v| v > 0.05).count()
                )
            };
            println!(
                "{:<16} KLD mean {:.4} max {:.4}{shown}; top-1 {top}/{nt}; reads/t mean {:.4} median {:.4} p90 {:.4}; bytes/dense mean {:.4}; head KL mean {:.4} max {:.4}; bound vectors/refresh {:.1}; flat-set recall {:.3}",
                arms[a].name(),
                kl.iter().sum::<f64>() / kl.len() as f64,
                kl.iter().copied().fold(0.0, f64::max),
                r.iter().sum::<f64>() / r.len() as f64,
                q(&r, 0.5),
                q(&r, 0.9),
                bytes[a].iter().sum::<f64>() / bytes[a].len() as f64,
                hk.iter().sum::<f64>() / hk.len() as f64,
                hk[hk.len() - 1],
                rmeta[a].iter().sum::<f64>() / rmeta[a].len().max(1) as f64,
                rec[a].iter().sum::<f64>() / rec[a].len().max(1) as f64,
            );
        }
        if std::env::var_os("KV_TAIL_POS").is_some() {
            for a in 0..na {
                for i in 0..nt {
                    let rd = &logits[i * lno..(i + 1) * lno];
                    let ra = &logits[(a * nt + i) * lno..(a * nt + i + 1) * lno];
                    let (pd, pa) = (lsm(rd), lsm(ra));
                    let kld: f64 = pd.iter().zip(&pa).map(|(&x, &y)| x.exp() * (x - y)).sum();
                    let e = pos[a][i];
                    let ent = |lp: &[f64]| -lp.iter().map(|&x| x.exp() * x).sum::<f64>();
                    let margin = |lp: &[f64]| {
                        let (mut t1, mut t2) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
                        for &x in lp {
                            if x > t1 {
                                t2 = t1;
                                t1 = x;
                            } else if x > t2 {
                                t2 = x;
                            }
                        }
                        t1 - t2
                    };
                    println!(
                        "POS {} {i} kld {kld:.5} hkl {:.5} p_mean {:.5} p_max {:.4} d_mean {:.4} d_max {:.4} hkl_max {:.4} ent_arm {:.4} marg_arm {:.4} ent_dense {:.4} marg_dense {:.4}",
                        arms[a].name(),
                        e[0] / e[5],
                        e[1] / e[5],
                        e[2],
                        e[3] / e[5],
                        e[4],
                        e[6],
                        ent(&pa),
                        margin(&pa),
                        ent(&pd),
                        margin(&pd)
                    );
                }
            }
        }
        println!("elapsed {:.0}s", t0.elapsed().as_secs_f64());
    }
    println!("{}", if ok { "TAIL PASS" } else { "TAIL FAIL" });
    Ok(ok)
}

fn main() -> std::process::ExitCode {
    let path = std::env::args().nth(1).unwrap_or_else(|| MODEL.to_string());
    if let Some(dir) = std::env::var_os("KV_TAIL") {
        return match run_tail(&path, std::path::Path::new(&dir)) {
            Ok(true) => std::process::ExitCode::SUCCESS,
            Ok(false) => std::process::ExitCode::FAILURE,
            Err(e) => {
                eprintln!("attn_cut_probe: {e}");
                std::process::ExitCode::FAILURE
            }
        };
    }
    let res = match (std::env::var_os("KV_DUMP"), std::env::var("KV_T")) {
        (Some(dir), Ok(tn)) => match tn.parse() {
            Ok(tn) => run_dump(&path, std::path::Path::new(&dir), tn),
            Err(e) => Err(format!("KV_T: {e}").into()),
        },
        _ => run(&path),
    };
    match res {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::FAILURE,
        Err(e) => {
            eprintln!("attn_cut_probe: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
