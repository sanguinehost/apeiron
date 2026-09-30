//! Instrument for the prefill `Q8_0` GEMM (`gemm_q8_0`, `gemm_q8_0_group`, `gemm_q8_0_swiglu`).
//!
//! Inputs are the real ones: the weights are the model's own payloads and the activation columns
//! are logged from the engine's real prefill (`Engine::prefill_state`), then replayed through the
//! same entry points. The execution path of those kernels is selected by the data (row and
//! column scale spread against `spread_limit`, the fast-bound and pack refusals, the `finish`
//! refusals, the carrier), so every call carries its branch counters ([`hooks`]) and an oracle
//! (`oracle`) recomputes the selectors from the weight bytes and activation scales alone.
//!
//! Run: `APEIRON_F32_CARRIER=1 cargo test --release --lib --locked gemm_probe -- --ignored --nocapture`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::print_stdout,
    clippy::format_push_string,
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "measurement instrument: metrics on exact integer shapes, printed tables, test-only"
)]

/// Branch counters and the call log the kernels feed under `cfg(test)`.
pub(crate) mod hooks {
    use crate::engine::chart::QuantAct;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering::Relaxed};
    use std::sync::Mutex;

    /// Names of the branch counters, in index order; the rest are spread-histogram bins.
    pub(crate) const NAMES: [&str; 9] = [
        "x2_noshift",
        "x2_shift",
        "tail_group2",
        "fast_finish",
        "exact_spread",
        "fast_refused",
        "odd_row",
        "swiglu_f32",
        "swiglu_q32",
    ];
    pub(crate) const X2_NOSHIFT: usize = 0;
    pub(crate) const X2_SHIFT: usize = 1;
    pub(crate) const TAIL_G2: usize = 2;
    pub(crate) const FAST_OK: usize = 3;
    pub(crate) const EXACT_SPREAD: usize = 4;
    pub(crate) const FAST_REFUSED: usize = 5;
    pub(crate) const ODD_ROW: usize = 6;
    pub(crate) const SWIGLU_F32: usize = 7;
    pub(crate) const SWIGLU_Q32: usize = 8;
    pub(crate) const HIST0: usize = 9;
    pub(crate) const HIST_BINS: usize = 40;
    pub(crate) const N_CTR: usize = HIST0 + HIST_BINS;

    static GLOBAL: [AtomicU64; N_CTR] = [const { AtomicU64::new(0) }; N_CTR];
    /// Whether task-local counters flush into [`GLOBAL`] (off while timing).
    pub(crate) static COUNT_ON: AtomicBool = AtomicBool::new(false);

    #[must_use]
    pub(crate) fn snapshot() -> [u64; N_CTR] {
        std::array::from_fn(|i| GLOBAL[i].load(Relaxed))
    }

    /// Per-task counters, flushed on drop.
    pub(crate) struct Local([Cell<u64>; N_CTR]);

    impl Local {
        pub(crate) fn new() -> Self {
            Local([const { Cell::new(0) }; N_CTR])
        }
        pub(crate) fn add(&self, i: usize) {
            self.0[i].set(self.0[i].get() + 1);
        }
        /// Histogram of `row scale spread + group spread` against the limit.
        pub(crate) fn spreads(&self, a: u32, b: Option<u32>) {
            self.add(HIST0 + (a as usize).min(HIST_BINS - 1));
            if let Some(b) = b {
                self.add(HIST0 + (b as usize).min(HIST_BINS - 1));
            }
        }
    }

    impl Drop for Local {
        fn drop(&mut self) {
            if COUNT_ON.load(Relaxed) {
                for (g, l) in GLOBAL.iter().zip(&self.0) {
                    if l.get() != 0 {
                        g.fetch_add(l.get(), Relaxed);
                    }
                }
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(crate) enum Kind {
        Gemm,
        Group,
        Swiglu,
    }

    /// One weight payload of a call: address, byte length and rows.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Part {
        pub(crate) ptr: usize,
        pub(crate) len: usize,
        pub(crate) n_out: usize,
    }

    /// One logged call.
    pub(crate) struct Call {
        pub(crate) kind: Kind,
        pub(crate) n_in: usize,
        pub(crate) k: usize,
        pub(crate) parts: Vec<Part>,
        /// `(scales, codes)` per column; filled only in capture mode.
        pub(crate) cols: Vec<(Vec<i64>, Vec<i8>)>,
        /// Counter delta of the call (valid when [`COUNT_ON`]).
        pub(crate) ctr: [u64; N_CTR],
        pub(crate) ok: bool,
        pub(crate) ns: u64,
    }

    /// 0 off, 1 time only, 2 time and copy inputs.
    static MODE: AtomicU8 = AtomicU8::new(0);
    static LOG: Mutex<Vec<Call>> = Mutex::new(Vec::new());

    pub(crate) fn start(mode: u8) {
        LOG.lock().unwrap().clear();
        MODE.store(mode, Relaxed);
    }

    pub(crate) fn stop() -> Vec<Call> {
        MODE.store(0, Relaxed);
        std::mem::take(&mut *LOG.lock().unwrap())
    }

    pub(crate) struct Guard {
        call: Option<Call>,
        before: [u64; N_CTR],
        t0: Option<std::time::Instant>,
        ok: Cell<bool>,
    }

    impl Guard {
        pub(crate) fn ok(&self) {
            self.ok.set(true);
        }
    }

    pub(crate) fn enter(
        kind: Kind,
        n_in: usize,
        parts: &[(&[u8], usize)],
        qs: &[QuantAct],
    ) -> Guard {
        let mode = MODE.load(Relaxed);
        let mut g = Guard {
            call: None,
            before: [0; N_CTR],
            t0: None,
            ok: Cell::new(false),
        };
        if mode == 0 {
            return g;
        }
        let cols = if mode == 2 {
            qs.iter()
                .map(|q| (q.scales().to_vec(), q.codes().to_vec()))
                .collect()
        } else {
            Vec::new()
        };
        g.call = Some(Call {
            kind,
            n_in,
            k: qs.len(),
            parts: parts
                .iter()
                .map(|(r, n)| Part {
                    ptr: r.as_ptr() as usize,
                    len: r.len(),
                    n_out: *n,
                })
                .collect(),
            cols,
            ctr: [0; N_CTR],
            ok: false,
            ns: 0,
        });
        g.before = snapshot();
        g.t0 = Some(std::time::Instant::now());
        g
    }

    /// [`enter`] for [`Kind::Group`] parts `(type, payload, n_out)`.
    pub(crate) fn enter_group(
        n_in: usize,
        parts: &[(u32, &[u8], usize)],
        qs: &[QuantAct],
    ) -> Guard {
        let pr: Vec<(&[u8], usize)> = parts.iter().map(|(_, r, n)| (*r, *n)).collect();
        enter(Kind::Group, n_in, &pr, qs)
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            let (Some(mut c), Some(t0)) = (self.call.take(), self.t0) else {
                return;
            };
            c.ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let after = snapshot();
            for i in 0..N_CTR {
                c.ctr[i] = after[i] - self.before[i];
            }
            c.ok = self.ok.get();
            LOG.lock().unwrap().push(c);
        }
    }
}

use self::hooks::{Call, Kind, Part, N_CTR};
use super::gemm::{gemm_q8_0, gemm_q8_0_group, gemm_q8_0_swiglu, probe_limit, probe_pack_ok};
use super::pow2::f16_split;
use crate::engine::chart::ffn::quant_acts_pooled;
use crate::engine::chart::{quant_act, QuantAct};
use crate::engine::loader::Engine;
use crate::engine::KernelMode;
use crate::format::quant::{GGML_TYPE_Q8_0, Q8_0_BLOCK_BYTES};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

pub(super) const MODEL: &str = "/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf";
const IDS: &str = "/home/socol/oc-tmp/q35-t5/prompt2.ids";
const MODEL_SHA_PREFIX: &str = "37ae482d";

// ---- host -------------------------------------------------------------------------------------

fn proc_stat() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/stat").unwrap_or_default();
    let v: Vec<u64> = s
        .lines()
        .next()
        .unwrap_or("")
        .split_whitespace()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    if v.len() < 5 {
        return (0, 0);
    }
    let total: u64 = v.iter().take(8).sum();
    (total, v[3] + v[4])
}

/// Host-wide busy fraction over `ms` milliseconds (`/proc/stat`).
fn host_busy(ms: u64) -> f64 {
    let (t0, i0) = proc_stat();
    std::thread::sleep(std::time::Duration::from_millis(ms));
    let (t1, i1) = proc_stat();
    let dt = t1.saturating_sub(t0).max(1);
    1.0 - (i1.saturating_sub(i0)) as f64 / dt as f64
}

/// Seconds of quiet-gate waiting left in the whole run (within the 300 s cap of the quiet-host
/// rule; the gate command's budget is 540 s).
/// Start of the probe.
static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

static WAIT_LEFT: std::sync::Mutex<f64> = std::sync::Mutex::new(120.0);

/// Waits (within the run's remaining wait budget) for a host-wide busy fraction below 0.05 over
/// 2 s; returns (quiet, busy).
pub(super) fn wait_quiet() -> (bool, f64) {
    loop {
        let t = Instant::now();
        let b = host_busy(2000);
        let mut left = WAIT_LEFT.lock().unwrap();
        if b < 0.05 {
            return (true, b);
        }
        if *left <= 0.0 {
            return (false, b);
        }
        drop(left);
        std::thread::sleep(std::time::Duration::from_secs(3));
        left = WAIT_LEFT.lock().unwrap();
        *left -= t.elapsed().as_secs_f64();
    }
}

/// Clock ticks of this process (`utime + stime`, all threads).
fn self_ticks() -> u64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let after = s.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = after.split_whitespace().collect();
    let u: u64 = f.get(11).and_then(|x| x.parse().ok()).unwrap_or(0);
    let k: u64 = f.get(12).and_then(|x| x.parse().ok()).unwrap_or(0);
    u + k
}

/// A window over which the CPU time not spent by this process is measured.
pub(super) struct Load {
    host: (u64, u64),
    me: u64,
}

impl Load {
    pub(super) fn begin() -> Load {
        Load {
            host: proc_stat(),
            me: self_ticks(),
        }
    }
    /// Fraction of host CPU time spent by other processes since `begin`.
    pub(super) fn foreign(&self) -> f64 {
        let (t, i) = proc_stat();
        let dt = t.saturating_sub(self.host.0).max(1) as f64;
        let busy = (t - i).saturating_sub(self.host.0 - self.host.1) as f64;
        let me = self_ticks().saturating_sub(self.me) as f64;
        ((busy - me) / dt).max(0.0)
    }
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// Mean `scaling_cur_freq` (MHz) of cpus `lo..hi`.
fn mean_mhz(lo: usize, hi: usize) -> f64 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for c in lo..hi {
        let p = format!("/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq");
        if let Ok(s) = std::fs::read_to_string(p) {
            if let Ok(khz) = s.trim().parse::<f64>() {
                sum += khz / 1000.0;
                n += 1.0;
            }
        }
    }
    if n == 0.0 {
        0.0
    } else {
        sum / n
    }
}

fn freq_line() -> String {
    format!(
        "MHz P(cpu0-15) {:.0} E(cpu16-31) {:.0}",
        mean_mhz(0, 16),
        mean_mhz(16, 32)
    )
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .unwrap_or("?")
        .trim()
        .to_string()
}

fn governor() -> String {
    std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        .unwrap_or_default()
        .trim()
        .to_string()
}

// ---- statistics -------------------------------------------------------------------------------

fn best_med(v: &[f64]) -> (f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[0], s[s.len() / 2])
}

// ---- inputs -----------------------------------------------------------------------------------

/// A `QuantAct` column rebuilt through the engine's own quantiser from its logged
/// `(scales, codes)`: `x_j = S_b·code_j`, whose absolute maximum `S_b·127` or `S_b·c_max` selects
/// the same power-of-two scale and whose codes round back exactly. Returns the column and whether
/// scales and codes equal the logged ones.
fn rebuild_col(scales: &[i64], codes: &[i8]) -> (QuantAct, bool) {
    let x: Vec<i64> = codes
        .iter()
        .enumerate()
        .map(|(j, &c)| scales[j / 32] * i64::from(c))
        .collect();
    let q = quant_act(&x).expect("logged column requantizes");
    let same = q.scales() == scales && q.codes() == codes;
    (q, same)
}

pub(super) fn rebuild(cols: &[(Vec<i64>, Vec<i8>)]) -> (Vec<QuantAct>, usize) {
    let mut bad = 0;
    let qs = cols
        .iter()
        .map(|(s, c)| {
            let (q, same) = rebuild_col(s, c);
            bad += usize::from(!same);
            q
        })
        .collect();
    (qs, bad)
}

/// The payload of a logged part.
///
/// # Safety
///
/// The `Engine` the part was logged from is alive and unmoved: its payloads are slices of one
/// boxed file buffer.
pub(super) unsafe fn raw(p: &Part) -> &'static [u8] {
    // SAFETY: by the caller's contract `ptr..ptr + len` is inside the live file buffer.
    unsafe { std::slice::from_raw_parts(p.ptr as *const u8, p.len) }
}

/// Runs one call through the production entry point and drops its output.
fn run_call(c: &Call, qs: &[QuantAct]) -> bool {
    // SAFETY: the caller keeps the engine alive for the whole probe.
    unsafe {
        match c.kind {
            Kind::Gemm => {
                let r = gemm_q8_0(
                    GGML_TYPE_Q8_0,
                    raw(&c.parts[0]),
                    c.parts[0].n_out,
                    c.n_in,
                    qs,
                );
                let ok = r.is_some();
                black_box(r);
                ok
            }
            Kind::Group => {
                let parts: Vec<(u32, &[u8], usize)> = c
                    .parts
                    .iter()
                    .map(|p| (GGML_TYPE_Q8_0, raw(p), p.n_out))
                    .collect();
                let r = gemm_q8_0_group(&parts, c.n_in, qs);
                let ok = r.is_some();
                black_box(r);
                ok
            }
            Kind::Swiglu => {
                let r = gemm_q8_0_swiglu(
                    raw(&c.parts[0]),
                    raw(&c.parts[1]),
                    c.parts[0].n_out,
                    c.n_in,
                    qs,
                );
                let ok = r.is_some();
                black_box(r);
                ok
            }
        }
    }
}

fn weight_bytes(c: &Call) -> u64 {
    c.parts
        .iter()
        .map(|p| (p.n_out * c.n_in / 32 * Q8_0_BLOCK_BYTES) as u64)
        .sum()
}

fn macs(c: &Call, k: usize) -> f64 {
    c.parts
        .iter()
        .map(|p| p.n_out as f64 * c.n_in as f64 * k as f64)
        .sum()
}

fn tensor_names(e: &Engine) -> BTreeMap<usize, String> {
    let mut m = BTreeMap::new();
    for t in e.table().entries() {
        if let Some(nb) = t.n_bytes {
            if let Ok(sl) = e.payload(t.byte_range.start, nb) {
                m.insert(sl.as_ptr() as usize, t.name.clone());
            }
        }
    }
    m
}

fn strip_layer(n: &str) -> String {
    let mut it = n.splitn(3, '.');
    match (it.next(), it.next(), it.next()) {
        (Some("blk"), Some(l), Some(rest)) if l.parse::<usize>().is_ok() => format!("blk.*.{rest}"),
        _ => n.to_string(),
    }
}

fn layer_of(n: &str) -> usize {
    n.split('.')
        .nth(1)
        .and_then(|l| l.parse().ok())
        .unwrap_or(0)
}

struct Named<'a> {
    call: &'a Call,
    names: Vec<String>,
}

impl Named<'_> {
    fn class(&self) -> String {
        let n: Vec<String> = self.names.iter().map(|n| strip_layer(n)).collect();
        format!(
            "{:?} {} n_in={}",
            self.call.kind,
            n.join("+"),
            self.call.n_in
        )
    }
    fn layer(&self) -> usize {
        layer_of(&self.names[0])
    }
}

// ---- path oracle ------------------------------------------------------------------------------

/// `⌈log₂ blocks⌉`, derived here from the contract `30 + spread + ⌈log₂ blocks⌉ ≤ 63`.
fn oracle_limit(blocks: usize) -> u32 {
    let log = usize::BITS - blocks.saturating_sub(1).leading_zeros();
    63 - 30 - log
}

/// The exponent spread `k_max − k_min` of every row of a payload.
fn row_spreads(raw: &[u8], n_out: usize, blocks: usize) -> Vec<u32> {
    let rb = blocks * Q8_0_BLOCK_BYTES;
    (0..n_out)
        .map(|r| {
            let row = &raw[r * rb..(r + 1) * rb];
            let (mut lo, mut hi) = (u32::MAX, 0u32);
            for b in 0..blocks {
                let h =
                    u16::from_le_bytes([row[b * Q8_0_BLOCK_BYTES], row[b * Q8_0_BLOCK_BYTES + 1]]);
                let (_, k) = f16_split(h).expect("finite weight scale");
                lo = lo.min(k);
                hi = hi.max(k);
            }
            hi - lo
        })
        .collect()
}

/// The spread of each 8-column group of the activation scales (`pack`'s `spread`).
fn group_spreads(qs: &[QuantAct]) -> Vec<u32> {
    qs.chunks(8)
        .map(|g| {
            let mut sp = 0;
            for q in g {
                let tz: Vec<Option<u32>> = q
                    .scales()
                    .iter()
                    .map(|&s| {
                        if s == 0 {
                            None
                        } else {
                            Some(s.trailing_zeros())
                        }
                    })
                    .collect();
                let lo = tz.iter().flatten().copied().min().unwrap_or(0);
                for t in tz.into_iter().flatten() {
                    sp = sp.max(t - lo);
                }
            }
            sp
        })
        .collect()
}

/// The branch counters the kernel must report for `c` on `qs`, from the weight bytes and the
/// activation scales alone (no kernel code): pairing of rows, group pairs, tail group, exact
/// decisions and the spread histogram.
fn oracle(c: &Call, qs: &[QuantAct], carrier_f32: bool) -> [u64; N_CTR] {
    use hooks::{
        EXACT_SPREAD, HIST0, HIST_BINS, ODD_ROW, SWIGLU_F32, SWIGLU_Q32, TAIL_G2, X2_NOSHIFT,
        X2_SHIFT,
    };
    let blocks = c.n_in / 32;
    let limit = oracle_limit(blocks);
    let gs = group_spreads(qs);
    let ng = gs.len();
    let mut o = [0u64; N_CTR];
    let unit = |o: &mut [u64; N_CTR], spa: u32, spb: u32, pair: bool, swiglu: bool| {
        let hist = |o: &mut [u64; N_CTR], s: u32| o[HIST0 + (s as usize).min(HIST_BINS - 1)] += 1;
        let sub = |o: &mut [u64; N_CTR], g: usize| {
            hist(o, spa + gs[g]);
            if pair || swiglu {
                hist(o, spb + gs[g]);
            }
            if spa + gs[g] > limit || ((pair || swiglu) && spb + gs[g] > limit) {
                o[EXACT_SPREAD] += 1;
            }
            if swiglu {
                o[if carrier_f32 { SWIGLU_F32 } else { SWIGLU_Q32 }] += 1;
            }
        };
        let mut g = 0;
        while g + 1 < ng {
            if spa == 0 && spb == 0 && gs[g] == 0 && gs[g + 1] == 0 {
                o[X2_NOSHIFT] += 1;
            } else {
                o[X2_SHIFT] += 1;
            }
            sub(o, g);
            sub(o, g + 1);
            g += 2;
        }
        if g < ng {
            o[TAIL_G2] += 1;
            sub(o, g);
        }
    };
    // SAFETY: the engine is alive for the probe.
    unsafe {
        match c.kind {
            Kind::Gemm | Kind::Group => {
                for p in &c.parts {
                    let sp = row_spreads(raw(p), p.n_out, blocks);
                    let mut r = 0;
                    while r < p.n_out {
                        if r + 1 < p.n_out {
                            unit(&mut o, sp[r], sp[r + 1], true, false);
                            r += 2;
                        } else {
                            unit(&mut o, sp[r], sp[r], false, false);
                            o[ODD_ROW] += 1;
                            r += 1;
                        }
                    }
                }
            }
            Kind::Swiglu => {
                let n = c.parts[0].n_out;
                let a = row_spreads(raw(&c.parts[0]), n, blocks);
                let b = row_spreads(raw(&c.parts[1]), n, blocks);
                for i in 0..n {
                    unit(&mut o, a[i], b[i], true, true);
                }
            }
        }
    }
    o
}

/// Indices the oracle can derive (`fast_finish`/`fast_refused` depend on the data's `finish`
/// validity and are checked only for their sum).
fn oracle_indices() -> Vec<usize> {
    let mut v = vec![
        hooks::X2_NOSHIFT,
        hooks::X2_SHIFT,
        hooks::TAIL_G2,
        hooks::EXACT_SPREAD,
        hooks::ODD_ROW,
        hooks::SWIGLU_F32,
        hooks::SWIGLU_Q32,
    ];
    v.extend(hooks::HIST0..N_CTR);
    v
}

fn ctr_line(c: &[u64; N_CTR]) -> String {
    let mut s = String::new();
    for (i, n) in hooks::NAMES.iter().enumerate() {
        s += &format!("{n}={} ", c[i]);
    }
    let hist: Vec<String> = (0..hooks::HIST_BINS)
        .filter(|&b| c[hooks::HIST0 + b] != 0)
        .map(|b| format!("{b}:{}", c[hooks::HIST0 + b]))
        .collect();
    s + &format!("spread_hist[{}]", hist.join(" "))
}

fn add_ctr(a: &mut [u64; N_CTR], b: &[u64; N_CTR]) {
    for i in 0..N_CTR {
        a[i] += b[i];
    }
}

// ---- synthetic fixture of `bench_q8_0_gemm_shapes` (the instrument under test) ------------------

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *seed >> 33
}

fn fixture_payload(rows: usize, n_in: usize, scales: &[u16], seed: &mut u64) -> Vec<u8> {
    let mut out = Vec::new();
    for _ in 0..rows * (n_in / 32) {
        let s = scales[lcg(seed) as usize % scales.len()];
        out.extend_from_slice(&s.to_le_bytes());
        for _ in 0..32 {
            out.push((lcg(seed) % 256) as u8);
        }
    }
    out
}

fn fixture_columns(k: usize, n_in: usize, range: u32, seed: &mut u64) -> Vec<QuantAct> {
    (0..k)
        .map(|_| {
            let x: Vec<i64> = (0..n_in)
                .map(|j| {
                    if (j / 32) % 7 == 3 {
                        return 0;
                    }
                    let mag = 1i64 << (20 + (lcg(seed) as u32 % range.max(1)));
                    (lcg(seed).cast_signed() % (2 * mag + 1)) - mag
                })
                .collect();
            quant_act(&x).expect("activations quantize")
        })
        .collect()
}

// ---- timing -----------------------------------------------------------------------------------

/// Times `f` (ms): a 0.25 s spin warm-up (clock ramp), then up to `max_reps` calls or `budget_s`
/// seconds (at least 15 calls).
fn time_ms(budget_s: f64, max_reps: usize, mut f: impl FnMut()) -> Vec<f64> {
    let w = Instant::now();
    f();
    while w.elapsed().as_secs_f64() < (0.25f64).min(budget_s * 2.0) {
        f();
    }
    let mut v = Vec::new();
    let t = Instant::now();
    while v.len() < max_reps && (v.len() < 15 || t.elapsed().as_secs_f64() < budget_s) {
        let s = Instant::now();
        f();
        v.push(s.elapsed().as_secs_f64() * 1e3);
    }
    v
}

/// Runs `f` until a window with foreign CPU share below 0.05 is observed (at most 8 attempts,
/// waiting for a quiet host between attempts); returns the result and that share (the lowest seen
/// when no attempt was clean).
fn clean<T>(mut f: impl FnMut() -> T) -> (T, f64) {
    let mut best: Option<(T, f64)> = None;
    for attempt in 0..8 {
        if attempt > 0 {
            if *WAIT_LEFT.lock().unwrap() <= 0.0
                || START.get().is_some_and(|s| s.elapsed().as_secs() > 400)
            {
                break;
            }
            wait_quiet();
        }
        let l = Load::begin();
        let r = f();
        let fr = l.foreign();
        if fr < 0.05 {
            return (r, fr);
        }
        if best.as_ref().is_none_or(|b| fr < b.1) {
            best = Some((r, fr));
        }
    }
    best.expect("one attempt")
}

/// The VNNI peak (GMAC/s) of the 8 P-cores and 16 E-cores at the given mean MHz: `vpdpbusd` ymm
/// is 32 MAC, at 2 per cycle on ADL-P (TP 0.50, ports 0/1) and 1 per 2 cycles on ADL-E (TP 2.00)
/// (uops.info, `instructions.xml` dated 2026-03-29, sha256 e5e702ca…, `VPDPBUSD_VEX (YMM, YMM,
/// YMM)`); Raptor Cove and Gracemont in the i9-14900KF share those port layouts (anchor: uops.info
/// lists no RPL entry).
fn peak_gmacs(p_mhz: f64, e_mhz: f64) -> (f64, f64, f64) {
    let p_core = 64.0 * p_mhz / 1e3;
    let e_core = 16.0 * e_mhz / 1e3;
    (8.0 * p_core + 16.0 * e_core, p_core, e_core)
}

/// `k` columns from a base list (cyclic past its end).
pub(super) fn take_k(base: &[(Vec<i64>, Vec<i8>)], k: usize) -> Vec<(Vec<i64>, Vec<i8>)> {
    (0..k).map(|i| base[i % base.len()].clone()).collect()
}

pub(super) fn read_ids() -> Vec<u32> {
    std::fs::read_to_string(IDS)
        .expect("ids")
        .split_whitespace()
        .filter_map(|x| x.parse().ok())
        .collect()
}

fn sha256(path: &str) -> String {
    let out = std::process::Command::new("sha256sum")
        .arg(path)
        .output()
        .expect("sha256sum");
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_string()
}

fn ffn_trace_ns(f: impl FnOnce()) -> (u64, u64) {
    crate::engine::trace::set_enabled(true);
    crate::engine::trace::reset();
    f();
    let s = crate::engine::trace::snapshot();
    crate::engine::trace::set_enabled(false);
    let p = s.iter().find(|p| p.name == "ffn").expect("ffn phase");
    (p.ns, p.calls)
}

#[test]
#[ignore = "release-only instrument: APEIRON_F32_CARRIER=1 cargo test --release --lib --locked gemm_probe -- --ignored --nocapture"]
fn gemm_probe_real_inputs() {
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni")) {
        println!("gemm_probe: host lacks AVX-VNNI; nothing to measure");
        return;
    }
    let carrier_f32 =
        crate::engine::mixer::carrier_from_env() == crate::engine::mixer::Carrier::F32;
    let width = crate::engine::chart::pool::Pool::global().width();
    START.get_or_init(Instant::now);
    let quick = std::env::var("APEIRON_PROBE_QUICK").is_ok();
    println!("== gemm_probe ==");
    println!(
        "host: {} governor {} | pool width {width} | carrier {} | APEIRON_PREFILL_COLS block {}",
        cpu_model(),
        governor(),
        if carrier_f32 {
            "F32 (APEIRON_F32_CARRIER=1)"
        } else {
            "Q32.32"
        },
        crate::engine::step::prefill_block()
    );
    let sha = sha256(MODEL);
    println!("model {MODEL} sha256 {sha}");
    assert!(sha.starts_with(MODEL_SHA_PREFIX), "model sha256 {sha}");
    let ids = read_ids();
    println!("prompt ids {IDS}: {} ids", ids.len());
    let e = Engine::open(MODEL).expect("open model");
    let names = tensor_names(&e);
    let mode = KernelMode::Q8Act;
    let prefill = |e: &Engine| {
        let mut st = e.state(ids.len() + 1).expect("state");
        e.prefill_state(&mut st, &ids, mode).expect("prefill");
    };
    // Warm-up (page-in, lazy tables, pool start).
    prefill(&e);

    // ---- B. real prefill: capture, then timing-only pass, then the trace ---------------------
    hooks::COUNT_ON.store(true, std::sync::atomic::Ordering::Relaxed);
    hooks::start(2);
    prefill(&e);
    let cap = hooks::stop();
    hooks::COUNT_ON.store(false, std::sync::atomic::Ordering::Relaxed);
    let (quiet, busy0) = wait_quiet();
    println!(
        "quiet gate: busy {busy0:.3} over 2 s quiet={quiet} | uptime {} | {}",
        loadavg(),
        freq_line()
    );
    // Two unrecorded prefills to ramp the clocks after the idle gate, then three recorded ones.
    prefill(&e);
    prefill(&e);
    let mut prod_runs: Vec<Vec<Call>> = Vec::new();
    let mut prefill_ms = Vec::new();
    let mut max_foreign = 0.0f64;
    for _ in 0..3 {
        let ((ms, run), fr) = clean(|| {
            hooks::start(1);
            let t = Instant::now();
            prefill(&e);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            (ms, hooks::stop())
        });
        prefill_ms.push(ms);
        prod_runs.push(run);
        max_foreign = max_foreign.max(fr);
    }
    println!(
        "real prefill pp{} wall with hooks: {:.1} / {:.1} / {:.1} ms ({:.0} tok/s last); max foreign CPU share {max_foreign:.3} | {}",
        ids.len(),
        prefill_ms[0],
        prefill_ms[1],
        prefill_ms[2],
        ids.len() as f64 / prefill_ms[2] * 1e3,
        freq_line()
    );
    let named: Vec<Named> = cap
        .iter()
        .map(|c| Named {
            call: c,
            names: c
                .parts
                .iter()
                .map(|p| {
                    names
                        .get(&p.ptr)
                        .cloned()
                        .unwrap_or_else(|| format!("?@{:x}", p.ptr))
                })
                .collect(),
        })
        .collect();

    // ---- C. (a) every projection shape prefill routes through the GEMM ------------------------
    println!(
        "\n-- (a) GEMM calls of one real prefill (class: calls, k per call, n_out, weight MB) --"
    );
    let mut classes: BTreeMap<String, Vec<&Named>> = BTreeMap::new();
    for n in &named {
        classes.entry(n.class()).or_default().push(n);
    }
    for (cl, v) in &classes {
        let ks: Vec<usize> = v.iter().map(|n| n.call.k).collect();
        let mut kset: BTreeMap<usize, usize> = BTreeMap::new();
        for k in &ks {
            *kset.entry(*k).or_default() += 1;
        }
        let outs: Vec<usize> = v[0].call.parts.iter().map(|p| p.n_out).collect();
        println!(
            "{cl}: {} calls, k {:?}, n_out {:?}, weights {:.2} MB/call, refused {}",
            v.len(),
            kset,
            outs,
            weight_bytes(v[0].call) as f64 / 1e6,
            v.iter().filter(|n| !n.call.ok).count()
        );
    }
    let total_gemm_calls = cap.len();
    let refused = cap.iter().filter(|c| !c.ok).count();
    println!("total logged GEMM calls {total_gemm_calls}, refused (contract path) {refused}");

    // Rebuild the real columns once.
    let mut rebuilt: Vec<Vec<QuantAct>> = Vec::new();
    let mut requant_bad = 0usize;
    let mut cols_total = 0usize;
    for c in &cap {
        let (qs, bad) = rebuild(&c.cols);
        requant_bad += bad;
        cols_total += qs.len();
        rebuilt.push(qs);
    }
    println!(
        "activation columns logged from the real prefill: {cols_total}; requantized through quant_act with scales/codes differing: {requant_bad}"
    );
    assert_eq!(
        requant_bad, 0,
        "requantization must reproduce the logged columns"
    );

    // ---- D. (c) path identity ----------------------------------------------------------------
    println!("\n-- (c) path identity --");
    hooks::COUNT_ON.store(true, std::sync::atomic::Ordering::Relaxed);
    let mut per_class: BTreeMap<String, ([u64; N_CTR], [u64; N_CTR], usize, usize, usize)> =
        BTreeMap::new();
    let mut replay_eq_prod = 0usize;
    let mut oracle_eq = 0usize;
    let mut first_bad: Option<String> = None;
    let oidx = oracle_indices();
    for (i, n) in named.iter().enumerate() {
        let c = n.call;
        hooks::start(1);
        let ok = run_call(c, &rebuilt[i]);
        let mut rep = hooks::stop();
        let r = rep.pop().expect("replay logged");
        let same_as_prod = r.ctr == c.ctr && ok == c.ok;
        let orc = oracle(c, &rebuilt[i], carrier_f32);
        let units: u64 = match c.kind {
            Kind::Swiglu => c.parts[0].n_out as u64,
            _ => c.parts.iter().map(|p| p.n_out.div_ceil(2) as u64).sum(),
        } * rebuilt[i].len().div_ceil(8) as u64;
        let oracle_ok = c.ok
            && oidx.iter().all(|&j| orc[j] == c.ctr[j])
            && c.ctr[hooks::FAST_OK] + c.ctr[hooks::FAST_REFUSED] + c.ctr[hooks::EXACT_SPREAD]
                == units;
        replay_eq_prod += usize::from(same_as_prod);
        oracle_eq += usize::from(oracle_ok);
        if (!same_as_prod || !oracle_ok) && first_bad.is_none() {
            first_bad = Some(format!(
                "call {i} {}: replay==prod {same_as_prod} oracle==kernel {oracle_ok}\n  prod   {}\n  replay {}\n  oracle {}",
                n.class(),
                ctr_line(&c.ctr),
                ctr_line(&r.ctr),
                ctr_line(&orc)
            ));
        }
        let ent = per_class
            .entry(n.class())
            .or_insert(([0; N_CTR], [0; N_CTR], 0, 0, 0));
        add_ctr(&mut ent.0, &c.ctr);
        add_ctr(&mut ent.1, &r.ctr);
        ent.2 += 1;
        ent.3 += usize::from(same_as_prod);
        ent.4 += usize::from(oracle_ok);
    }
    hooks::COUNT_ON.store(false, std::sync::atomic::Ordering::Relaxed);
    for (cl, (prod, rep, n, same, orc)) in &per_class {
        println!("{cl}: {n} calls; replay==prod {same}/{n}; kernel==oracle {orc}/{n}");
        println!("  production {}", ctr_line(prod));
        println!("  replay     {}", ctr_line(rep));
    }
    let id_ok = replay_eq_prod == named.len() && oracle_eq == named.len() && refused == 0;
    println!(
        "PATH IDENTITY (replayed real inputs vs production, per call, all branch counters + ok flag): {replay_eq_prod}/{} equal",
        named.len()
    );
    println!(
        "PATH ORACLE (kernel counters vs selectors recomputed from weight bytes and activation scales): {oracle_eq}/{} equal",
        named.len()
    );
    if let Some(b) = &first_bad {
        println!("first divergence: {b}");
    }
    println!(
        "VERDICT real-input replay: {}",
        if id_ok {
            "PATH IDENTITY HOLDS"
        } else {
            "PATH IDENTITY FAILS"
        }
    );

    // The instrument under test: the synthetic fixture of bench_q8_0_gemm_shapes.
    println!("\n-- old fixture (bench_q8_0_gemm_shapes: k=64, scales 0x2000/0x2400/0x1c00/0xa000, columns range 8) --");
    hooks::COUNT_ON.store(true, std::sync::atomic::Ordering::Relaxed);
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut fixture_differs = false;
    for &(n_in, n_out) in &[(1024usize, 3584usize), (3584, 1024), (1024, 6144)] {
        let payload = fixture_payload(n_out, n_in, &[0x2000, 0x2400, 0x1c00, 0xa000], &mut seed);
        let qs = fixture_columns(64, n_in, 8, &mut seed);
        hooks::start(1);
        let _ = black_box(gemm_q8_0(GGML_TYPE_Q8_0, &payload, n_out, n_in, &qs));
        let mut l = hooks::stop();
        let r = l.pop().expect("logged");
        println!(
            "fixture {n_in}->{n_out} k=64 ok={}: {}",
            r.ok,
            ctr_line(&r.ctr)
        );
        let mix = |c: &[u64; N_CTR]| {
            let t =
                (c[hooks::FAST_OK] + c[hooks::FAST_REFUSED] + c[hooks::EXACT_SPREAD]).max(1) as f64;
            (
                c[hooks::X2_NOSHIFT] as f64
                    / (c[hooks::X2_NOSHIFT] + c[hooks::X2_SHIFT]).max(1) as f64,
                c[hooks::FAST_OK] as f64 / t,
                c[hooks::EXACT_SPREAD] as f64 / t,
                c[hooks::FAST_REFUSED] as f64 / t,
            )
        };
        let mean_spread = |c: &[u64; N_CTR]| {
            let n: u64 = c[hooks::HIST0..].iter().sum();
            let w: u64 = (0..hooks::HIST_BINS)
                .map(|b| b as u64 * c[hooks::HIST0 + b])
                .sum();
            w as f64 / n.max(1) as f64
        };
        let mut prod_gemm = [0u64; N_CTR];
        for (k, (p, ..)) in &per_class {
            if k.starts_with("Gemm") {
                add_ctr(&mut prod_gemm, p);
            }
        }
        let (fn_, ff, fe, fr) = mix(&r.ctr);
        let (pn, pf, pe, pr) = mix(&prod_gemm);
        println!(
            "  branch mix (noshift share of x2, fast_finish, exact_spread, fast_refused): fixture {fn_:.3}/{ff:.3}/{fe:.4}/{fr:.4} vs production Gemm {pn:.3}/{pf:.3}/{pe:.4}/{pr:.4}; mean total spread fixture {:.2} production {:.2}",
            mean_spread(&r.ctr),
            mean_spread(&prod_gemm)
        );
        fixture_differs |= (fn_ - pn).abs() > 0.02
            || (fe - pe).abs() > 0.02
            || (fr - pr).abs() > 0.02
            || (mean_spread(&r.ctr) - mean_spread(&prod_gemm)).abs() > 1.0;
    }
    hooks::COUNT_ON.store(false, std::sync::atomic::Ordering::Relaxed);
    println!(
        "VERDICT old fixture: {}",
        if fixture_differs {
            "BRANCH MIX OR SPREAD DISTRIBUTION DIFFERS from production inputs"
        } else {
            "branch mix and spread match production"
        }
    );

    // ---- E. (e) reproduction of the real FFN cost --------------------------------------------
    println!("\n-- (e) FFN reproduction --");
    let is_ffn = |n: &Named| {
        n.names
            .iter()
            .any(|s| s.contains("ffn_down") || s.contains("ffn_gate"))
    };
    let ffn: Vec<usize> = (0..named.len()).filter(|&i| is_ffn(&named[i])).collect();
    let prod_ns =
        |run: &Vec<Call>| -> f64 { ffn.iter().map(|&i| run[i].ns as f64).sum::<f64>() / 1e6 };
    for (r, run) in prod_runs.iter().enumerate() {
        assert_eq!(
            run.len(),
            cap.len(),
            "production call sequence is deterministic"
        );
        for (a, b) in run.iter().zip(&cap) {
            assert!(
                a.kind == b.kind && a.k == b.k && a.n_in == b.n_in,
                "same sequence"
            );
        }
        println!(
            "production FFN GEMM calls (hooked, in place) run {r}: {:.2} ms over {} calls",
            prod_ns(run),
            ffn.len()
        );
    }
    // Every class of the real prefill, in place: sum of call time and share of the wall time.
    {
        let run = &prod_runs[2];
        let wall = prefill_ms[2];
        let mut by: BTreeMap<String, (usize, f64)> = BTreeMap::new();
        for (i, n) in named.iter().enumerate() {
            let e = by
                .entry(format!("{} k={}", n.class(), n.call.k))
                .or_default();
            e.0 += 1;
            e.1 += run[i].ns as f64 / 1e6;
        }
        let total: f64 = by.values().map(|v| v.1).sum();
        println!("production GEMM time in place (run 2, wall {wall:.1} ms): total {total:.1} ms = {:.1}% of wall", 100.0 * total / wall);
        for (k, (n, ms)) in &by {
            println!(
                "  {k}: {n} calls, {ms:.2} ms ({:.3} ms/call)",
                ms / *n as f64
            );
        }
    }
    // Trace of the real prefill: phase ffn over the blocked prefill and over the final token.
    let (((ns_blocks, calls_blocks), (ns_last, calls_last)), f_tr) = clean(|| {
        let mut st = e.state(ids.len() + 1).expect("state");
        let head = &ids[..ids.len() - 1];
        let a = ffn_trace_ns(|| {
            e.prefill_cols(&mut st, head, mode).expect("prefill_cols");
        });
        let b = ffn_trace_ns(|| {
            e.next_token_state(&mut st, ids[ids.len() - 1], mode)
                .expect("last token");
        });
        (a, b)
    });
    max_foreign = max_foreign.max(f_tr);
    println!(
        "trace phase ffn (foreign {f_tr:.3}): blocked prefill {:.2} ms / {calls_blocks} calls ({:.3} ms/call); final token (k=1, non-GEMM path) {:.2} ms / {calls_last} calls ({:.3} ms/call)",
        ns_blocks as f64 / 1e6,
        ns_blocks as f64 / 1e6 / calls_blocks as f64,
        ns_last as f64 / 1e6,
        ns_last as f64 / 1e6 / calls_last as f64
    );
    println!(
        "  => the traced FFN calls of one pp{} run are {calls_blocks} blocked calls (k per block) + {calls_last} k=1 calls: {:.3} ms/call average over all {} calls, {:.2} ms total",
        ids.len(),
        (ns_blocks + ns_last) as f64 / 1e6 / (calls_blocks + calls_last) as f64,
        calls_blocks + calls_last,
        (ns_blocks + ns_last) as f64 / 1e6
    );
    // Quantization cost inside the FFN span (gate/up input and down input), on rebuilt x.
    let mut quant_ms = 0.0;
    let mut quant_seen: BTreeMap<(bool, usize), (f64, usize)> = BTreeMap::new();
    for &i in &ffn {
        let c = named[i].call;
        let is_sw = matches!(c.kind, Kind::Swiglu);
        if !is_sw && !named[i].names[0].contains("ffn_down") {
            continue;
        }
        if let Some(e) = quant_seen.get_mut(&(is_sw, c.k)) {
            e.1 += 1;
            continue;
        }
        let xs: Vec<Vec<i64>> = c
            .cols
            .iter()
            .map(|(s, cd)| {
                cd.iter()
                    .enumerate()
                    .map(|(j, &v)| s[j / 32] * i64::from(v))
                    .collect()
            })
            .collect();
        let (v, _) = clean(|| {
            time_ms(0.1, 40, || {
                black_box(quant_acts_pooled(&xs));
            })
        });
        quant_seen.insert((is_sw, c.k), (best_med(&v).1, 1));
    }
    for (med, n) in quant_seen.values() {
        quant_ms += med * *n as f64;
    }
    println!("pooled activation quantization of the same columns (median per call, sum over the swiglu+down calls): {quant_ms:.2} ms");
    // Replays of the FFN call sequence in isolation; per-call times kept.
    let seq = |reps: usize| -> Vec<Vec<f64>> {
        let mut v = Vec::new();
        for _ in 0..reps {
            let mut pass = Vec::new();
            for &i in &ffn {
                let t = Instant::now();
                run_call(named[i].call, &rebuilt[i]);
                pass.push(t.elapsed().as_secs_f64() * 1e3);
            }
            v.push(pass);
        }
        v
    };
    let _ = seq(3);
    let (quiet2, b2) = wait_quiet();
    println!(
        "quiet gate before replay: busy {b2:.3} quiet={quiet2} | uptime {}",
        loadavg()
    );
    let (passes, f_seq) = clean(|| seq(9));
    max_foreign = max_foreign.max(f_seq);
    let totals: Vec<f64> = passes.iter().map(|p| p.iter().sum()).collect();
    let (sb, sm) = best_med(&totals);
    println!("replay of the production FFN GEMM sequence (layer-cycling, 530 MB of FFN weights; foreign {f_seq:.3}): best {sb:.2} ms median {sm:.2} ms over {} calls; {}", ffn.len(), freq_line());
    let per_call_med: Vec<f64> = (0..ffn.len())
        .map(|j| best_med(&passes.iter().map(|p| p[j]).collect::<Vec<_>>()).1)
        .collect();
    let prod_med = {
        let mut v: Vec<f64> = prod_runs.iter().map(prod_ns).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    println!(
        "REPRODUCTION: isolated replay {sm:.2} ms vs in-place hooked production {prod_med:.2} ms (ratio {:.3}); replay + quantization {quant_ms:.2} ms = {:.2} ms vs traced blocked FFN {:.2} ms (ratio {:.3}); quiet={}",
        sm / prod_med,
        sm + quant_ms,
        ns_blocks as f64 / 1e6,
        (sm + quant_ms) / (ns_blocks as f64 / 1e6),
        quiet && quiet2 && max_foreign < 0.05
    );
    // Per-class cold (layer-cycling) and in-place numbers at the production k.
    let mut cold: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut inpl: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (j, &i) in ffn.iter().enumerate() {
        let key = format!("{} k={}", named[i].class(), named[i].call.k);
        cold.entry(key.clone()).or_default().push(per_call_med[j]);
        inpl.entry(key)
            .or_default()
            .push(prod_runs[2][i].ns as f64 / 1e6);
    }
    for (k, v) in &cold {
        println!(
            "  {k}: isolated replay median {:.3} ms/call, in place {:.3} ms/call",
            best_med(v).1,
            best_med(&inpl[k]).1
        );
    }

    // ---- composed FFN call: the brief's k=200 figure ---------------------------------------------
    println!("\n-- composed FFN call (fused gate/up+SwiGLU, then down) --");
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for l in 0..24usize {
        let sw = named
            .iter()
            .position(|n| matches!(n.call.kind, Kind::Swiglu) && n.layer() == l);
        let dn = named
            .iter()
            .position(|n| n.names[0].contains("ffn_down") && n.layer() == l);
        if let (Some(a), Some(b)) = (sw, dn) {
            pairs.push((a, b));
        }
    }
    println!("{} layers with both calls", pairs.len());
    let mut composed_rows: Vec<(usize, f64, f64)> = Vec::new();
    for &k in if quick {
        &[64usize][..]
    } else {
        &[64usize, 200][..]
    } {
        let mut sets: Vec<(Vec<QuantAct>, Vec<QuantAct>)> = Vec::new();
        for &(a, b) in &pairs {
            let l = named[a].layer();
            let base = |want: &dyn Fn(&Named) -> bool| -> Vec<(Vec<i64>, Vec<i8>)> {
                named
                    .iter()
                    .filter(|n| n.layer() == l && want(n))
                    .flat_map(|n| n.call.cols.iter().cloned())
                    .collect()
            };
            let sw_cols = base(&|n| matches!(n.call.kind, Kind::Swiglu));
            let dn_cols = base(&|n| n.names[0].contains("ffn_down"));
            let _ = b;
            sets.push((
                rebuild(&take_k(&sw_cols, k)).0,
                rebuild(&take_k(&dn_cols, k)).0,
            ));
        }
        let macs_ffn = 3.0 * 1024.0 * 3584.0 * k as f64;
        // Cold: layers in sequence, 6 cycles (weights come from beyond the caches).
        let (cyc, fr) = clean(|| {
            let mut per_layer: Vec<(Vec<f64>, Vec<f64>)> =
                vec![(Vec::new(), Vec::new()); pairs.len()];
            for _ in 0..3 {
                for (j, &(a, b)) in pairs.iter().enumerate() {
                    run_call(named[a].call, &sets[j].0);
                    run_call(named[b].call, &sets[j].1);
                }
            }
            for _ in 0..6 {
                for (j, &(a, b)) in pairs.iter().enumerate() {
                    let t = Instant::now();
                    run_call(named[a].call, &sets[j].0);
                    let t1 = t.elapsed().as_secs_f64() * 1e3;
                    let t = Instant::now();
                    run_call(named[b].call, &sets[j].1);
                    per_layer[j].0.push(t1);
                    per_layer[j].1.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            per_layer
        });
        max_foreign = max_foreign.max(fr);
        let mut sw_all: Vec<f64> = cyc.iter().map(|p| best_med(&p.0).1).collect();
        let mut dn_all: Vec<f64> = cyc.iter().map(|p| best_med(&p.1).1).collect();
        let tot: Vec<f64> = sw_all.iter().zip(&dn_all).map(|(a, b)| a + b).collect();
        let (tb, tm) = best_med(&tot);
        sw_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        dn_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean_tot = tot.iter().sum::<f64>() / tot.len() as f64;
        println!(
            "k={k} cold (layer-cycling, foreign {fr:.3}): per call swiglu median {:.3} ms + down median {:.3} ms; composed mean {mean_tot:.3} ms, median {tm:.3}, best layer {tb:.3} ms => {:.1} GMAC/s (mean), {:.1} (best layer); {}",
            sw_all[sw_all.len() / 2],
            dn_all[dn_all.len() / 2],
            macs_ffn / (mean_tot * 1e-3) / 1e9,
            macs_ffn / (tb * 1e-3) / 1e9,
            freq_line()
        );
        // Hot: layer 12 repeated (weights resident).
        let j = pairs.len() / 2;
        let (a, b) = pairs[j];
        let (hv, fr) = clean(|| {
            time_ms(0.4, 200, || {
                run_call(named[a].call, &sets[j].0);
                run_call(named[b].call, &sets[j].1);
            })
        });
        max_foreign = max_foreign.max(fr);
        let (hb, hm) = best_med(&hv);
        println!(
            "k={k} hot (layer {} repeated, foreign {fr:.3}): composed best {hb:.3} ms median {hm:.3} ms => {:.1} GMAC/s (best), {:.1} (median); cold/hot mean ratio {:.2}",
            named[a].layer(),
            macs_ffn / (hb * 1e-3) / 1e9,
            macs_ffn / (hm * 1e-3) / 1e9,
            mean_tot / hm
        );
        composed_rows.push((k, mean_tot, hm));
    }

    // ---- F. (d) per-shape table ---------------------------------------------------------------
    println!("\n-- (d) per shape, per k (pool width {width}; warm weights; GMAC/s = n_out*n_in*k/t; peak = VNNI ymm peak at the row's measured MHz) --");
    println!(
        "{:<44} {:>4} {:>8} {:>8} {:>8} {:>8} {:>7} {:>8} {:>6} {:>6} {:>10}",
        "class (layer)",
        "k",
        "best ms",
        "med ms",
        "GMAC/s b",
        "GMAC/s m",
        "GB/s b",
        "pack b",
        "%peak",
        "frgn",
        "MHz P/E"
    );
    let (quiet4, b4) = wait_quiet();
    println!(
        "quiet gate before the table: busy {b4:.3} quiet={quiet4} | uptime {}",
        loadavg()
    );
    let mut layer_pick: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (cl, v) in &classes {
        let mut layers: Vec<usize> = v.iter().map(|n| n.layer()).collect();
        layers.sort_unstable();
        layers.dedup();
        let picks = if layers.len() > 2 && cl.contains("ffn") {
            vec![
                layers[0],
                layers[layers.len() / 2],
                layers[layers.len() - 1],
            ]
        } else if layers.len() > 2 {
            vec![layers[0], layers[layers.len() - 1]]
        } else {
            layers
        };
        layer_pick.insert(cl.clone(), picks);
    }
    for (cl, v) in &classes {
        if quick && !cl.contains("ffn") {
            continue;
        }
        for &lay in &layer_pick[cl] {
            let calls: Vec<&&Named> = v.iter().filter(|n| n.layer() == lay).collect();
            // Base columns: every logged column of this (class, layer), in prefill order (199).
            let base: Vec<(Vec<i64>, Vec<i8>)> = calls
                .iter()
                .flat_map(|n| n.call.cols.iter().cloned())
                .collect();
            let c0 = calls[0].call;
            for &k in if quick {
                &[64usize, 128, 200][..]
            } else {
                &[1usize, 8, 64, 128, 200][..]
            } {
                let (qs, bad) = rebuild(&take_k(&base, k));
                assert_eq!(bad, 0);
                let blocks = c0.n_in / 32;
                let ((v, pv, mhz), fr) = clean(|| {
                    let v = time_ms(0.4, 200, || {
                        run_call(c0, &qs);
                    });
                    let mhz = (mean_mhz(0, 16), mean_mhz(16, 32));
                    let pv = time_ms(0.15, 100, || {
                        black_box(probe_pack_ok(&qs, blocks));
                    });
                    (v, pv, mhz)
                });
                max_foreign = max_foreign.max(fr);
                let (b, m) = best_med(&v);
                let (pb, _) = best_med(&pv);
                let gm = macs(c0, k);
                let (peak, _, _) = peak_gmacs(mhz.0, mhz.1);
                println!(
                    "{:<44} {:>4} {:>8.3} {:>8.3} {:>8.1} {:>8.1} {:>7.1} {:>8.3} {:>6.1} {:>6.3} {:>4.0}/{:<4.0}",
                    format!("{} L{lay}", short_class(cl)),
                    k,
                    b,
                    m,
                    gm / (b * 1e-3) / 1e9,
                    gm / (m * 1e-3) / 1e9,
                    weight_bytes(c0) as f64 / (b * 1e-3) / 1e9,
                    pb,
                    100.0 * gm / (b * 1e-3) / 1e9 / peak,
                    fr,
                    mhz.0,
                    mhz.1
                );
            }
        }
    }
    println!(
        "limit(blocks=32)={} limit(blocks=112)={}",
        probe_limit(32),
        probe_limit(112)
    );
    let (pk, pc, ec) = peak_gmacs(mean_mhz(0, 16).max(4500.0), mean_mhz(16, 32).max(3500.0));
    println!("peak anchor example at 4500/3500 MHz: machine {pk:.0} GMAC/s = 8 P x {pc:.0} + 16 E x {ec:.0}");
    println!("{}", freq_line());
    println!("uptime after: {}", loadavg());
    println!(
        "GRADE inputs: quiet_before_prefill_runs={quiet} quiet_before_replay={quiet2} quiet_before_table={quiet4} max_foreign_cpu_share={max_foreign:.3} path_identity={id_ok} fixture_spread_differs={fixture_differs} composed_k200_cold_mean_ms={:.3}",
        composed_rows.last().map_or(0.0, |r| r.1)
    );
}

/// A shorter class label for the table.
fn short_class(c: &str) -> String {
    c.replace("blk.*.", "")
        .replace(".weight", "")
        .replace("Group attn_q+attn_k+attn_v", "Group qkv(attn)")
        .replace(
            "Group attn_qkv+attn_gate+ssm_beta+ssm_alpha",
            "Group qkv+gate+ba(ssm)",
        )
        .replace("Swiglu ffn_gate+ffn_up", "Swiglu gate+up")
}
