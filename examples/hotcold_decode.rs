//! Hot/cold decode-step attention at depth `t`: dense attention against 4-bit-key-scan top-`B` selection.
//!
//! One decode step at depth `t` of Qwen2.5-7B-Instruct-1M (28 layers, 28 query heads, 4 KV heads, `head_dim` 128)
//! over the real KV cache of the `gpu_dump` prose dump (`g7-1m`): keys post-RoPE, values, and the last tail query of
//! the dump at position `t - 1`. Layers are processed one at a time (the f32 cache of one layer is 4 GiB at
//! `t = 2^20`; all 28 layers do not fit in RAM), the layer load is outside the timer, and the attention time of a
//! decode step is the sum of the 28 per-layer times.
//!
//! Arms, per layer and repetition:
//! * `dense`: the engine's attention core (`attn_core_f32_strided`) per query head over all `t` rows, one head per
//!   thread. It reads each KV head's rows once per query head of the group (7x).
//! * `floor`: one pass of AVX2 loads over the layer's K and V planes: the time to stream the bytes `dense` must
//!   read at least once.
//! * `hot`: a 4-bit code per key row (`scale = max|k|/7`, nibble `round(k/scale) + 8`, dim `i` in the low nibble and
//!   dim `i + 64` in the high nibble of byte `i`) is scanned against the int8-quantised query for all 7 query heads of
//!   a KV head per code row; per query head the top-`B` far rows by score (histogram threshold), the 4 sinks and the
//!   near window are gathered (sorted) and attended with the same engine core.
//!
//! Controls on one layer: selecting every far row equals dense (rel-L2), and selecting the lowest-`B` scores is far
//! from dense.
//!
//! Run: `hotcold_decode --t 524288 [--b 8192] [--reps 5] [--dir /home/socol/oc-tmp/hattn-long/g7-1m]`.
#![allow(
    clippy::pedantic,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::allow_attributes_without_reason,
    clippy::undocumented_unsafe_blocks,
    reason = "measurement harness: numeric casts, indexed loops and raw SIMD are the point"
)]

use std::error::Error;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use apeiron::engine::attn_core_f32_strided;

const HD: usize = 128;
const HALF: usize = HD / 2;
const NH: usize = 28;
const NKV: usize = 4;
const NL: usize = 28;
const GROUP: usize = NH / NKV;
const KVW: usize = NKV * HD;
const QW: usize = NH * HD;
const SINK: usize = 4;
const WIN: usize = 256;
const PAGE: usize = 64;
const NTH: usize = 32;
const QBLOCKS: [usize; 3] = [262_144, 524_288, 1_048_576];
const NT_TAIL: usize = 64;

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
impl<T> SendPtr<T> {
    fn p(&self) -> *mut T {
        self.0
    }
}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

#[derive(Clone, Copy)]
struct Job(*const (dyn Fn(usize) + Sync + 'static));
unsafe impl Send for Job {}

struct St {
    gen: u64,
    job: Option<Job>,
    pending: usize,
    quit: bool,
}
struct Shared {
    st: Mutex<St>,
    go: Condvar,
    fin: Condvar,
}

/// A fixed pool of `n - 1` parked threads plus the caller; `run` blocks until every participant has returned.
struct Pool {
    sh: Arc<Shared>,
    n: usize,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    fn new(n: usize) -> Pool {
        let sh = Arc::new(Shared {
            st: Mutex::new(St {
                gen: 0,
                job: None,
                pending: 0,
                quit: false,
            }),
            go: Condvar::new(),
            fin: Condvar::new(),
        });
        let handles = (1..n)
            .map(|w| {
                let sh = Arc::clone(&sh);
                std::thread::spawn(move || {
                    let mut seen = 0u64;
                    loop {
                        let job = {
                            let mut s = sh.st.lock().unwrap();
                            while s.gen == seen && !s.quit {
                                s = sh.go.wait(s).unwrap();
                            }
                            if s.quit {
                                return;
                            }
                            seen = s.gen;
                            s.job.unwrap()
                        };
                        unsafe { (*job.0)(w) };
                        let mut s = sh.st.lock().unwrap();
                        s.pending -= 1;
                        if s.pending == 0 {
                            sh.fin.notify_one();
                        }
                    }
                })
            })
            .collect();
        Pool { sh, n, handles }
    }

    fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        // SAFETY: `run` does not return until `pending == 0`, so no worker dereferences `job` after `f` is dead.
        let job = Job(unsafe {
            std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f)
        });
        {
            let mut s = self.sh.st.lock().unwrap();
            s.job = Some(job);
            s.gen += 1;
            s.pending = self.n - 1;
            self.sh.go.notify_all();
        }
        f(0);
        let mut s = self.sh.st.lock().unwrap();
        while s.pending > 0 {
            s = self.sh.fin.wait(s).unwrap();
        }
        s.job = None;
    }

    /// Runs `f(i)` for `i` in `0..n`, items claimed from a shared counter.
    fn each(&self, n: usize, f: impl Fn(usize) + Sync) {
        let ctr = AtomicUsize::new(0);
        self.run(&|_| loop {
            let i = ctr.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            f(i);
        });
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.sh.st.lock().unwrap().quit = true;
        self.sh.go.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1).cloned())
}

/// Reads `buf.len()` f32 values at byte offset `off` of `p`, in parallel 16 MiB pieces.
fn read_f32(pool: &Pool, p: &Path, off: u64, buf: &mut [f32]) -> Result<(), Box<dyn Error>> {
    let f = std::fs::File::open(p)?;
    let total = buf.len() * 4;
    let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), total) };
    let ptr = SendPtr(bytes.as_mut_ptr());
    let piece = 16 << 20;
    let np = total.div_ceil(piece);
    let err = AtomicUsize::new(0);
    pool.each(np, |i| {
        let (a, b) = (i * piece, ((i + 1) * piece).min(total));
        let s = unsafe { std::slice::from_raw_parts_mut(ptr.p().add(a), b - a) };
        if f.read_exact_at(s, off + a as u64).is_err() {
            err.store(1, Ordering::Relaxed);
        }
    });
    if err.load(Ordering::Relaxed) != 0 {
        return Err(format!("short read {}", p.display()).into());
    }
    Ok(())
}

/// Rotates `heads` heads of `x` in place at position `p` (NEOX pairing, the dump's `cs` table).
fn rot_row(x: &mut [f32], heads: usize, cs: &[f64], p: usize) {
    for hh in 0..heads {
        let b = hh * HD;
        for j in 0..HALF {
            let (c, s) = (cs[(p * HALF + j) * 2], cs[(p * HALF + j) * 2 + 1]);
            let (x0, x1) = (f64::from(x[b + j]), f64::from(x[b + j + HALF]));
            x[b + j] = (x0 * c - x1 * s) as f32;
            x[b + j + HALF] = (x1 * c + x0 * s) as f32;
        }
    }
}

/// Scores rows `j0..j1` of one KV head's codes against the 7 int8 queries of its group.
#[target_feature(enable = "avx2")]
unsafe fn scan(
    codes: *const u8,
    scales: *const f32,
    j0: usize,
    j1: usize,
    qi: &[[i8; HD]; GROUP],
    qs: &[f32; GROUP],
    sumq: &[i32; GROUP],
    out: [*mut f32; GROUP],
) {
    use std::arch::x86_64::*;
    let m = _mm256_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    let mut qv = [[_mm256_setzero_si256(); 4]; GROUP];
    for h in 0..GROUP {
        for c in 0..4 {
            qv[h][c] = _mm256_loadu_si256(qi[h].as_ptr().add(c * 32).cast());
        }
    }
    for j in j0..j1 {
        let p = codes.add(j * 64);
        let c0 = _mm256_loadu_si256(p.cast());
        let c1 = _mm256_loadu_si256(p.add(32).cast());
        let lo0 = _mm256_and_si256(c0, m);
        let hi0 = _mm256_and_si256(_mm256_srli_epi16(c0, 4), m);
        let lo1 = _mm256_and_si256(c1, m);
        let hi1 = _mm256_and_si256(_mm256_srli_epi16(c1, 4), m);
        let sc = *scales.add(j);
        for h in 0..GROUP {
            let a = _mm256_add_epi16(
                _mm256_add_epi16(
                    _mm256_maddubs_epi16(lo0, qv[h][0]),
                    _mm256_maddubs_epi16(lo1, qv[h][1]),
                ),
                _mm256_add_epi16(
                    _mm256_maddubs_epi16(hi0, qv[h][2]),
                    _mm256_maddubs_epi16(hi1, qv[h][3]),
                ),
            );
            let s = _mm256_madd_epi16(a, ones);
            let t = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
            let t = _mm_hadd_epi32(t, t);
            let t = _mm_hadd_epi32(t, t);
            let d = _mm_cvtsi128_si32(t) - 8 * sumq[h];
            *out[h].add(j) = d as f32 * (sc * qs[h]);
        }
    }
}

/// Sum of `x` by AVX2 loads; the streaming-read floor.
#[target_feature(enable = "avx2")]
unsafe fn sum_stream(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut a = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 32 <= x.len() {
        for c in 0..4 {
            a[c] = _mm256_add_ps(a[c], _mm256_loadu_ps(x.as_ptr().add(i + c * 8)));
        }
        i += 32;
    }
    let mut l = [0f32; 8];
    let s = _mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3]));
    _mm256_storeu_ps(l.as_mut_ptr(), s);
    l.iter().sum()
}

/// Indices (relative to `region`) of the `b` largest scores. Time `O(len)` over three passes plus a
/// `select_nth` over the boundary histogram bin; space `O(bins + b)`.
fn select_top(region: &[f32], b: usize) -> Vec<u32> {
    let len = region.len();
    if b >= len {
        return (0..len as u32).collect();
    }
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &s in region {
        lo = lo.min(s);
        hi = hi.max(s);
    }
    if hi <= lo {
        return (0..b as u32).collect();
    }
    const NB: usize = 4096;
    let inv = (NB as f32 - 0.01) / (hi - lo);
    let bin = |s: f32| (((s - lo) * inv) as usize).min(NB - 1);
    let mut hist = vec![0u32; NB];
    for &s in region {
        hist[bin(s)] += 1;
    }
    let (mut acc, mut bs) = (0usize, NB);
    while bs > 0 && acc < b {
        bs -= 1;
        acc += hist[bs] as usize;
    }
    let cnt_hi = acc - hist[bs] as usize;
    let mut out: Vec<u32> = Vec::with_capacity(b);
    let mut bd: Vec<(f32, u32)> = Vec::new();
    for (i, &s) in region.iter().enumerate() {
        let bi = bin(s);
        if bi > bs {
            out.push(i as u32);
        } else if bi == bs {
            bd.push((s, i as u32));
        }
    }
    let need = b - cnt_hi;
    if need < bd.len() {
        bd.select_nth_unstable_by(need - 1, |x, y| y.0.total_cmp(&x.0));
        bd.truncate(need);
    }
    out.extend(bd.iter().map(|x| x.1));
    out
}

struct HeadBuf {
    sel: Vec<u32>,
    gk: Vec<f32>,
    gv: Vec<f32>,
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        num += (f64::from(x) - f64::from(y)).powi(2);
        den += f64::from(y).powi(2);
    }
    (num / den).sqrt()
}

/// Gathers rows `sel` (sorted here) of KV head `g` into `hb` and runs the engine attention core over them.
fn gather_attn(
    hb: &mut HeadBuf,
    k: &[f32],
    v: &[f32],
    g: usize,
    qh: &[f32],
    out: &mut [f32],
) -> (u64, u64) {
    let t0 = Instant::now();
    hb.sel.sort_unstable();
    let n = hb.sel.len();
    if hb.gk.len() < n * HD {
        hb.gk.resize(n * HD, 0.0);
        hb.gv.resize(n * HD, 0.0);
    }
    for (m, &j) in hb.sel.iter().enumerate() {
        let o = (j as usize * NKV + g) * HD;
        hb.gk[m * HD..(m + 1) * HD].copy_from_slice(&k[o..o + HD]);
        hb.gv[m * HD..(m + 1) * HD].copy_from_slice(&v[o..o + HD]);
    }
    let t1 = Instant::now();
    attn_core_f32_strided(
        qh,
        &hb.gk,
        &hb.gv,
        HD,
        HD,
        n - 1,
        1.0 / (HD as f32).sqrt(),
        out,
    )
    .expect("attn core");
    let t2 = Instant::now();
    ((t1 - t0).as_nanos() as u64, (t2 - t1).as_nanos() as u64)
}

fn stats(v: &[f64]) -> (f64, f64, f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    (
        m,
        sd,
        v.iter().cloned().fold(f64::INFINITY, f64::min),
        v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let t: usize = arg("--t").ok_or("--t")?.parse()?;
    let b: usize = arg("--b").map_or(Ok(8192), |s| s.parse())?;
    let reps: usize = arg("--reps").map_or(Ok(5), |s| s.parse())?;
    let nl: usize = arg("--layers").map_or(Ok(NL), |s| s.parse())?;
    let dir =
        PathBuf::from(arg("--dir").unwrap_or_else(|| "/home/socol/oc-tmp/hattn-long/g7-1m".into()));
    let pool = Pool::new(NTH);
    let blk = QBLOCKS.iter().position(|&x| x == t);
    let blk_i = blk.unwrap_or(0);
    println!(
        "hotcold_decode t={t} B={b} reps={reps} layers={nl} threads={NTH} q-source=block{} ({})",
        QBLOCKS[blk_i],
        if blk.is_some() {
            "real tail query at t-1"
        } else {
            "pseudo-query: tail query of block 262144 rotated at t-1"
        }
    );
    let mut cs = vec![0f64; t * HALF * 2];
    {
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(cs.as_mut_ptr().cast::<u8>(), cs.len() * 8) };
        let f = std::fs::File::open(dir.join("cs.bin"))?;
        let ptr = SendPtr(bytes.as_mut_ptr());
        let total = bytes.len();
        let piece = 16 << 20;
        pool.each(total.div_ceil(piece), |i| {
            let (a, e) = (i * piece, ((i + 1) * piece).min(total));
            let s = unsafe { std::slice::from_raw_parts_mut(ptr.p().add(a), e - a) };
            f.read_exact_at(s, a as u64).expect("cs read");
        });
    }
    let nb = SINK + (t - (WIN + SINK)) / PAGE * PAGE;
    let far = nb - SINK;
    let mut k = vec![0f32; t * KVW];
    let mut v = vec![0f32; t * KVW];
    let mut codes = vec![0u8; NKV * t * 64];
    let mut scales = vec![0f32; NKV * t];
    let mut sc = vec![0f32; NH * t];
    let bufs: Vec<Mutex<HeadBuf>> = (0..NH)
        .map(|_| {
            Mutex::new(HeadBuf {
                sel: Vec::new(),
                gk: Vec::new(),
                gv: Vec::new(),
            })
        })
        .collect();
    let mut tm: [Vec<Vec<f64>>; 4] = std::array::from_fn(|_| vec![vec![0f64; nl]; reps]);
    // tm[0] dense, tm[1] floor, tm[2] hot scan, tm[3] hot select+gather+attn (seconds per layer)
    let task_ns: [AtomicU64; 3] = std::array::from_fn(|_| AtomicU64::new(0));
    let warm_ns: [AtomicU64; 3] = std::array::from_fn(|_| AtomicU64::new(0));
    let (mut rel_sum, mut rel_max) = (0f64, 0f64);
    let mut ctl: Vec<(usize, f64, f64, f64)> = Vec::new();
    let sink = AtomicU64::new(0);
    let t_all = Instant::now();
    for l in 0..nl {
        let tl = Instant::now();
        read_f32(&pool, &dir.join(format!("k{l}.bin")), 0, &mut k)?;
        read_f32(&pool, &dir.join(format!("v{l}.bin")), 0, &mut v)?;
        let mut q = vec![0f32; QW];
        read_f32(
            &pool,
            &dir.join(format!("q{l}.bin")),
            (((blk_i * NT_TAIL + NT_TAIL - 1) * QW) * 4) as u64,
            &mut q,
        )?;
        rot_row(&mut q, NH, &cs, t - 1);
        {
            let kp = SendPtr(k.as_mut_ptr());
            let cp = SendPtr(codes.as_mut_ptr());
            let sp = SendPtr(scales.as_mut_ptr());
            let per = 4096;
            pool.each(t.div_ceil(per), |c| {
                for j in c * per..((c + 1) * per).min(t) {
                    let row = unsafe { std::slice::from_raw_parts_mut(kp.p().add(j * KVW), KVW) };
                    rot_row(row, NKV, &cs, j);
                    for g in 0..NKV {
                        let kr = &row[g * HD..(g + 1) * HD];
                        let mx = kr.iter().fold(0f32, |a, x| a.max(x.abs()));
                        let s = if mx > 0.0 { mx / 7.0 } else { 1.0 };
                        let mut u = [0u8; HD];
                        for d in 0..HD {
                            u[d] = ((kr[d] / s).round().clamp(-7.0, 7.0) + 8.0) as u8;
                        }
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(cp.p().add((g * t + j) * 64), 64)
                        };
                        for i in 0..64 {
                            dst[i] = u[i] | (u[i + 64] << 4);
                        }
                        unsafe { *sp.p().add(g * t + j) = s };
                    }
                }
            });
        }
        let mut qi = vec![[[0i8; HD]; GROUP]; NKV];
        let mut qs = vec![[0f32; GROUP]; NKV];
        let mut sumq = vec![[0i32; GROUP]; NKV];
        for h in 0..NH {
            let (g, r) = (h / GROUP, h % GROUP);
            let qh = &q[h * HD..(h + 1) * HD];
            let mx = qh.iter().fold(0f32, |a, x| a.max(x.abs()));
            let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
            for d in 0..HD {
                qi[g][r][d] = (qh[d] / s).round().clamp(-127.0, 127.0) as i8;
            }
            qs[g][r] = s;
            sumq[g][r] = qi[g][r].iter().map(|&x| i32::from(x)).sum();
        }
        let load_s = tl.elapsed().as_secs_f64();
        let mut dense_out = vec![0f32; QW];
        let mut hot_out = vec![0f32; QW];
        let chunk = 8192usize;
        let nchunk = far.div_ceil(chunk);
        let run_dense = |out: &mut [f32]| {
            let op = SendPtr(out.as_mut_ptr());
            pool.each(NH, |h| {
                let g = h / GROUP;
                let o = unsafe { std::slice::from_raw_parts_mut(op.p().add(h * HD), HD) };
                attn_core_f32_strided(
                    &q[h * HD..(h + 1) * HD],
                    &k[g * HD..],
                    &v[g * HD..],
                    HD,
                    KVW,
                    t - 1,
                    1.0 / (HD as f32).sqrt(),
                    o,
                )
                .expect("dense core");
            });
        };
        let run_scan = |sc: &mut [f32]| {
            let sp = SendPtr(sc.as_mut_ptr());
            pool.each(NKV * nchunk, |u| {
                let (g, c) = (u / nchunk, u % nchunk);
                let (j0, j1) = (SINK + c * chunk, (SINK + (c + 1) * chunk).min(nb));
                let out: [*mut f32; GROUP] =
                    std::array::from_fn(|r| unsafe { sp.p().add((g * GROUP + r) * t) });
                unsafe {
                    scan(
                        codes.as_ptr().add(g * t * 64),
                        scales.as_ptr().add(g * t),
                        j0,
                        j1,
                        &qi[g],
                        &qs[g],
                        &sumq[g],
                        out,
                    );
                }
            });
        };
        // Hot arm in three pool phases: select (per head), gather (per head x row-chunk, prefetched),
        // attention (per head). Phase wall times accumulate into `tasks` (ns).
        let run_hot = |sc: &[f32], out: &mut [f32], bsel: usize, tasks: &[AtomicU64; 3]| {
            let op = SendPtr(out.as_mut_ptr());
            let t0 = Instant::now();
            pool.each(NH, |h| {
                let region = &sc[h * t + SINK..h * t + nb];
                let top = select_top(region, bsel);
                let mut hb = bufs[h].lock().unwrap();
                hb.sel.clear();
                hb.sel.extend(0..SINK as u32);
                hb.sel.extend(top.iter().map(|&i| i + SINK as u32));
                hb.sel.extend(nb as u32..t as u32);
                hb.sel.sort_unstable();
                let n = hb.sel.len();
                if hb.gk.len() < n * HD {
                    hb.gk.resize(n * HD, 0.0);
                    hb.gv.resize(n * HD, 0.0);
                }
            });
            let t1 = Instant::now();
            let ptrs: Vec<(SendPtr<f32>, SendPtr<f32>, SendPtr<u32>, usize)> = bufs
                .iter()
                .map(|m| {
                    let mut hb = m.lock().unwrap();
                    (
                        SendPtr(hb.gk.as_mut_ptr()),
                        SendPtr(hb.gv.as_mut_ptr()),
                        SendPtr(hb.sel.as_mut_ptr()),
                        hb.sel.len(),
                    )
                })
                .collect();
            let gch = 256usize;
            let nu = ptrs[0].3.div_ceil(gch);
            pool.each(NH * nu, |u| {
                let (h, c) = (u / nu, u % nu);
                let g = h / GROUP;
                let (gk, gv, sel, n) = (&ptrs[h].0, &ptrs[h].1, &ptrs[h].2, ptrs[h].3);
                let (m0, m1) = (c * gch, ((c + 1) * gch).min(n));
                for m in m0..m1 {
                    if m + 8 < n {
                        let jn = unsafe { *sel.p().add(m + 8) } as usize;
                        let on = (jn * NKV + g) * HD;
                        unsafe {
                            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                            for c64 in 0..HD / 16 {
                                _mm_prefetch(
                                    k.as_ptr().add(on + c64 * 16).cast::<i8>(),
                                    _MM_HINT_T0,
                                );
                                _mm_prefetch(
                                    v.as_ptr().add(on + c64 * 16).cast::<i8>(),
                                    _MM_HINT_T0,
                                );
                            }
                        }
                    }
                    let j = unsafe { *sel.p().add(m) } as usize;
                    let o = (j * NKV + g) * HD;
                    unsafe {
                        std::ptr::copy_nonoverlapping(k.as_ptr().add(o), gk.p().add(m * HD), HD);
                        std::ptr::copy_nonoverlapping(v.as_ptr().add(o), gv.p().add(m * HD), HD);
                    }
                }
            });
            let t2 = Instant::now();
            pool.each(NH, |h| {
                let hb = bufs[h].lock().unwrap();
                let o = unsafe { std::slice::from_raw_parts_mut(op.p().add(h * HD), HD) };
                attn_core_f32_strided(
                    &q[h * HD..(h + 1) * HD],
                    &hb.gk,
                    &hb.gv,
                    HD,
                    HD,
                    hb.sel.len() - 1,
                    1.0 / (HD as f32).sqrt(),
                    o,
                )
                .expect("attn core");
            });
            let t3 = Instant::now();
            tasks[0].fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
            tasks[1].fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
            tasks[2].fetch_add((t3 - t2).as_nanos() as u64, Ordering::Relaxed);
        };
        // Unrecorded warm-up of the hot arm.
        run_scan(&mut sc);
        run_hot(&sc, &mut hot_out, b, &warm_ns);
        for r in 0..reps {
            let t0 = Instant::now();
            run_dense(&mut dense_out);
            tm[0][r][l] = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            let np = (t * KVW).div_ceil(1 << 20);
            let (kk, vv) = (&k, &v);
            pool.each(2 * np, |i| {
                let (src, c) = if i < np { (kk, i) } else { (vv, i - np) };
                let s = &src[c << 20..((c + 1) << 20).min(src.len())];
                let x = unsafe { sum_stream(s) };
                sink.fetch_add(x.to_bits().into(), Ordering::Relaxed);
            });
            tm[1][r][l] = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            run_scan(&mut sc);
            tm[2][r][l] = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            run_hot(&sc, &mut hot_out, b, &task_ns);
            tm[3][r][l] = t0.elapsed().as_secs_f64();
            if r == 0 {
                let mut layer_rel = 0f64;
                for h in 0..NH {
                    layer_rel += rel(
                        &hot_out[h * HD..(h + 1) * HD],
                        &dense_out[h * HD..(h + 1) * HD],
                    );
                }
                layer_rel /= NH as f64;
                rel_sum += layer_rel;
                rel_max = rel_max.max(layer_rel);
            }
        }
        if l == 14.min(nl - 1) {
            // Controls on heads 0 and 27: select-all (B = far) equals dense; lowest-B scores is far from dense.
            for &h in &[0usize, NH - 1] {
                let g = h / GROUP;
                let qh = &q[h * HD..(h + 1) * HD];
                let mk = |top: Vec<u32>| {
                    let mut hb = HeadBuf {
                        sel: Vec::new(),
                        gk: Vec::new(),
                        gv: Vec::new(),
                    };
                    hb.sel.extend(0..SINK as u32);
                    hb.sel.extend(top.iter().map(|&i| i + SINK as u32));
                    hb.sel.extend(nb as u32..t as u32);
                    let mut o = vec![0f32; HD];
                    gather_attn(&mut hb, &k, &v, g, qh, &mut o);
                    o
                };
                let all = mk(select_top(&sc[h * t + SINK..h * t + nb], far));
                let real = mk(select_top(&sc[h * t + SINK..h * t + nb], b));
                let neg: Vec<f32> = sc[h * t + SINK..h * t + nb].iter().map(|x| -x).collect();
                let bad = mk(select_top(&neg, b));
                let d = &dense_out[h * HD..(h + 1) * HD];
                ctl.push((h, rel(&all, d), rel(&real, d), rel(&bad, d)));
            }
        }
        // Selection recall against the exact-f32 top-B on one head, layer 14 only.
        if l == 14.min(nl - 1) {
            let h = 0usize;
            let g = h / GROUP;
            let qh = &q[h * HD..(h + 1) * HD];
            let exact: Vec<f32> = (SINK..nb)
                .map(|j| {
                    let o = (j * NKV + g) * HD;
                    qh.iter()
                        .zip(&k[o..o + HD])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                })
                .collect();
            let te = select_top(&exact, b);
            let ta = select_top(&sc[h * t + SINK..h * t + nb], b);
            let set: std::collections::HashSet<u32> = te.iter().copied().collect();
            let hit = ta.iter().filter(|x| set.contains(x)).count();
            println!(
                "recall layer {l} head {h}: 4-bit top-{b} overlaps exact-f32 top-{b} on {hit}/{b}"
            );
        }
        println!(
            "layer {l}: load+index {load_s:.1}s; dense {:.1} ms, floor {:.1} ms, hot {:.1}+{:.1} ms (rep mean)",
            tm[0].iter().map(|r| r[l]).sum::<f64>() / reps as f64 * 1e3,
            tm[1].iter().map(|r| r[l]).sum::<f64>() / reps as f64 * 1e3,
            tm[2].iter().map(|r| r[l]).sum::<f64>() / reps as f64 * 1e3,
            tm[3].iter().map(|r| r[l]).sum::<f64>() / reps as f64 * 1e3,
        );
    }
    let tot = |a: usize| -> Vec<f64> {
        (0..reps)
            .map(|r| tm[a][r].iter().sum::<f64>() * 1e3)
            .collect()
    };
    let hot: Vec<f64> = (0..reps)
        .map(|r| (tm[2][r].iter().sum::<f64>() + tm[3][r].iter().sum::<f64>()) * 1e3)
        .collect();
    println!("RESULT t={t} B={b} layers={nl} reps={reps} (ms per decode step, attention only, sum over layers; mean sd min max)");
    for (name, v) in [
        ("dense(engine core)", tot(0)),
        ("read-floor(K+V once)", tot(1)),
        ("hot scan", tot(2)),
        ("hot select+gather+attn", tot(3)),
        ("hot total", hot.clone()),
    ] {
        let (m, sd, lo, hi) = stats(&v);
        println!("RESULT {name:<24} {m:10.2} {sd:8.2} {lo:10.2} {hi:10.2}");
    }
    let denom = (reps * nl) as f64 * 1e6;
    println!(
        "RESULT hot phase wall ms/layer (mean): select {:.3} gather {:.3} attn {:.3}",
        task_ns[0].load(Ordering::Relaxed) as f64 / denom,
        task_ns[1].load(Ordering::Relaxed) as f64 / denom,
        task_ns[2].load(Ordering::Relaxed) as f64 / denom,
    );
    println!(
        "RESULT quality: hot-vs-dense rel-L2 of attention output, layer mean {:.4}, worst layer {:.4}",
        rel_sum / nl as f64,
        rel_max
    );
    for (h, all, real, bad) in &ctl {
        println!("CONTROL layer 14 head {h}: select-all vs dense rel-L2 {all:.3e}; B={b} {real:.3e}; lowest-B (mutation) {bad:.3e}");
    }
    let code_bytes = (NL * NKV * t * (64 + 4)) as f64;
    let full_f32 = (NL * 2 * t * KVW * 4) as f64;
    println!(
        "RESULT memory t={t}: 4-bit codes+scales {:.3} GiB, full f32 K+V {:.2} GiB (f16 derived {:.2} GiB), gathered rows per token {:.3} GiB f32",
        code_bytes / (1u64 << 30) as f64,
        full_f32 / (1u64 << 30) as f64,
        full_f32 / 2.0 / (1u64 << 30) as f64,
        (NL * NH * (b + SINK + (t - nb)) * HD * 4 * 2) as f64 / (1u64 << 30) as f64,
    );
    println!(
        "elapsed {:.0}s sink {}",
        t_all.elapsed().as_secs_f64(),
        sink.load(Ordering::Relaxed) & 1
    );
    Ok(())
}
