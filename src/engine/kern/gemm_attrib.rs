//! Per-pool-task attribution of the prefill GEMM (`cfg(test)`): which CPU ran each task, when, and
//! how many weight and activation bytes it touched. The kernels record into [`SLOTS`] (one slot per
//! task index of the live dispatch); `gemm_attrib_real_inputs` replays real layer-12 shapes and
//! prints per-participant busy and idle time, the makespan-to-mean ratio and ns/MAC on P and E.
//!
//! Run: `APEIRON_F32_CARRIER=1 cargo test --release --lib --locked gemm_attrib -- --ignored --nocapture`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::borrow_as_ptr,
    reason = "measurement instrument: metrics on exact integer shapes, printed tables, test-only"
)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::Instant;

const N_SLOTS: usize = 4096;

struct Slot {
    t0: AtomicU64,
    t1: AtomicU64,
    /// `cpu + 1`; 0 is unwritten.
    cpu: AtomicU64,
    rows: AtomicU64,
    wbytes: AtomicU64,
    abytes: AtomicU64,
}

static SLOTS: [Slot; N_SLOTS] = [const {
    Slot {
        t0: AtomicU64::new(0),
        t1: AtomicU64::new(0),
        cpu: AtomicU64::new(0),
        rows: AtomicU64::new(0),
        wbytes: AtomicU64::new(0),
        abytes: AtomicU64::new(0),
    }
}; N_SLOTS];
static ON: AtomicBool = AtomicBool::new(false);
static D0: AtomicU64 = AtomicU64::new(0);
static D1: AtomicU64 = AtomicU64::new(0);
static PH: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
static ORIGIN: OnceLock<Instant> = OnceLock::new();

fn now() -> u64 {
    u64::try_from(ORIGIN.get_or_init(Instant::now).elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Stamps phase `i` of the live call.
pub(crate) fn mark(i: usize) {
    if ON.load(Relaxed) {
        PH[i].store(now(), Relaxed);
    }
}

/// Marks the start of a `Pool::run` dispatch (the caller's thread).
pub(crate) fn dispatch_begin() {
    if ON.load(Relaxed) {
        D0.store(now(), Relaxed);
    }
}

/// Marks the end of a `Pool::run` dispatch.
pub(crate) fn dispatch_end() {
    if ON.load(Relaxed) {
        D1.store(now(), Relaxed);
    }
}

/// Records one task's interval on drop.
pub(crate) struct TaskGuard {
    idx: usize,
    t0: u64,
    rows: u64,
    wbytes: u64,
    abytes: u64,
}

/// Starts recording task `idx` (`None` while recording is off).
pub(crate) fn task(idx: usize, rows: usize, wbytes: usize, abytes: usize) -> Option<TaskGuard> {
    if !ON.load(Relaxed) || idx >= N_SLOTS {
        return None;
    }
    Some(TaskGuard {
        idx,
        t0: now(),
        rows: rows as u64,
        wbytes: wbytes as u64,
        abytes: abytes as u64,
    })
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        let t1 = now();
        // SAFETY: `sched_getcpu` has no preconditions.
        let cpu = unsafe { libc::sched_getcpu() };
        let s = &SLOTS[self.idx];
        s.t0.store(self.t0, Relaxed);
        s.t1.store(t1, Relaxed);
        s.rows.store(self.rows, Relaxed);
        s.wbytes.store(self.wbytes, Relaxed);
        s.abytes.store(self.abytes, Relaxed);
        s.cpu.store(u64::try_from(cpu).unwrap_or(0) + 1, Relaxed);
    }
}

/// One recorded dispatch.
#[derive(Clone)]
struct Disp {
    d0: u64,
    d1: u64,
    ph: [u64; 6],
    /// Whole-call wall ns (entry to drop of the result).
    call: u64,
    /// `(cpu, t0, t1, rows, wbytes, abytes)` per task.
    tasks: Vec<(usize, u64, u64, u64, u64, u64)>,
}

fn collect(n_tasks: usize, call: u64) -> Disp {
    let tasks = (0..n_tasks.min(N_SLOTS))
        .filter_map(|i| {
            let s = &SLOTS[i];
            let c = s.cpu.load(Relaxed);
            (c != 0).then(|| {
                (
                    (c - 1) as usize,
                    s.t0.load(Relaxed),
                    s.t1.load(Relaxed),
                    s.rows.load(Relaxed),
                    s.wbytes.load(Relaxed),
                    s.abytes.load(Relaxed),
                )
            })
        })
        .collect();
    for s in SLOTS.iter().take(n_tasks.min(N_SLOTS)) {
        s.cpu.store(0, Relaxed);
    }
    let mut ph = [0u64; 6];
    for (i, p) in ph.iter_mut().enumerate() {
        *p = PH[i].load(Relaxed);
    }
    Disp {
        d0: D0.load(Relaxed),
        d1: D1.load(Relaxed),
        ph,
        call,
        tasks,
    }
}

fn is_p(cpu: usize) -> bool {
    cpu < 16
}

/// Per-participant (by CPU) aggregate over `reps` dispatches.
#[derive(Default, Clone)]
struct Part {
    busy: f64,
    tasks: f64,
    macs: f64,
    /// Start of first task after the dispatch began.
    lag: f64,
    /// Dispatch end minus last task end.
    tail: f64,
    reps_seen: f64,
}

struct Summary {
    parts: Vec<Part>,
    wall_ms: f64,
    makespan_over_mean: f64,
    slowest_cpu: usize,
    p_nsmac: f64,
    e_nsmac: f64,
    p_task_ms: f64,
    e_task_ms: f64,
    p_tasks: f64,
    e_tasks: f64,
    wbytes: f64,
    abytes: f64,
    /// Mean microseconds: checks, pack, alloc, dispatch window, `set_len`, drop of the result.
    phases: [f64; 6],
    call_ms: f64,
}

fn summarize(ds: &[Disp], macs_per_row: f64) -> Summary {
    let n = ds.len() as f64;
    let mut parts = vec![Part::default(); 32];
    let mut wall = 0.0;
    let mut ratio = 0.0;
    let mut slow = vec![0usize; 32];
    let (mut pn, mut pm, mut en, mut em) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut pt, mut et, mut pc, mut ec) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut wb, mut ab) = (0.0, 0.0);
    let mut phases = [0.0f64; 6];
    let mut callns = 0.0f64;
    for d in ds {
        let f = |a: u64, b: u64| b.saturating_sub(a) as f64;
        callns += d.call as f64;
        phases[0] += f(d.ph[0], d.ph[1]);
        phases[1] += f(d.ph[1], d.ph[2]);
        phases[2] += f(d.ph[2], d.ph[3]);
        phases[3] += f(d.d0, d.d1);
        phases[4] += f(d.d1, d.ph[4]);
        phases[5] += f(d.ph[4], d.ph[5]);
        let w = (d.d1 - d.d0) as f64;
        wall += w;
        let mut busy = vec![0.0f64; 32];
        let mut last = vec![0u64; 32];
        let mut first = vec![u64::MAX; 32];
        for &(cpu, t0, t1, rows, wbytes, abytes) in &d.tasks {
            let c = cpu.min(31);
            let dt = (t1 - t0) as f64;
            busy[c] += dt;
            last[c] = last[c].max(t1);
            first[c] = first[c].min(t0);
            parts[c].tasks += 1.0;
            parts[c].macs += rows as f64 * macs_per_row;
            wb += wbytes as f64;
            ab += abytes as f64;
            if is_p(c) {
                pn += dt;
                pm += rows as f64 * macs_per_row;
                pt += dt;
                pc += 1.0;
            } else {
                en += dt;
                em += rows as f64 * macs_per_row;
                et += dt;
                ec += 1.0;
            }
        }
        let mut max_end = 0.0f64;
        let mut max_c = 0;
        for c in 0..32 {
            parts[c].busy += busy[c];
            if last[c] != 0 {
                parts[c].reps_seen += 1.0;
                parts[c].lag += first[c].saturating_sub(d.d0) as f64;
                parts[c].tail += d.d1.saturating_sub(last[c]) as f64;
                let e = (last[c] - d.d0) as f64;
                if e > max_end {
                    max_end = e;
                    max_c = c;
                }
            }
        }
        slow[max_c] += 1;
        let total_busy: f64 = busy.iter().sum();
        let active = busy.iter().filter(|b| **b > 0.0).count().max(1) as f64;
        ratio += max_end / (total_busy / active);
    }
    let slowest_cpu = (0..32).max_by_key(|c| slow[*c]).unwrap_or(0);
    let _ = n;
    Summary {
        parts,
        wall_ms: wall / n / 1e6,
        makespan_over_mean: ratio / n,
        slowest_cpu,
        p_nsmac: pn / pm.max(1.0),
        e_nsmac: en / em.max(1.0),
        p_task_ms: pt / pc.max(1.0) / 1e6,
        e_task_ms: et / ec.max(1.0) / 1e6,
        p_tasks: pc / n,
        e_tasks: ec / n,
        wbytes: wb / n,
        abytes: ab / n,
        phases: phases.map(|p| p / n / 1e3),
        call_ms: callns / n / 1e6,
    }
}

fn print_summary(label: &str, s: &Summary, reps: usize) {
    println!(
        "ATTRIB {label}: call {:.4} ms; dispatch window {:.4} ms; tasks P {:.1} E {:.1}; task mean P {:.4} ms E {:.4} ms; ns/MAC P {:.5} E {:.5} (E/P {:.2}); makespan/mean-busy {:.3}; slowest cpu {}; weight bytes/call {:.2} MB, activation bytes touched/call {:.1} MB",
        s.call_ms,
        s.wall_ms,
        s.p_tasks,
        s.e_tasks,
        s.p_task_ms,
        s.e_task_ms,
        s.p_nsmac,
        s.e_nsmac,
        s.e_nsmac / s.p_nsmac.max(1e-12),
        s.makespan_over_mean,
        s.slowest_cpu,
        s.wbytes / 1e6,
        s.abytes / 1e6
    );
    if s.phases[0] + s.phases[1] > 0.0 && s.phases[5] < 1e6 {
        println!(
        "  phases us/call (gemm_q8_0 only): checks {:.1} pack {:.1} alloc {:.1} | dispatch window {:.1} | set_len {:.1} | drop result {:.1} | total {:.1}",
        s.phases[0], s.phases[1], s.phases[2], s.phases[3], s.phases[4], s.phases[5],
        s.phases.iter().sum::<f64>()
    );
    }
    if std::env::var("APEIRON_ATTRIB_BRIEF").is_ok() {
        return;
    }
    println!(
        "  cpu  class tasks/call busy_ms/call idle_ms/call start_lag_us tail_us  GMAC/s-while-busy"
    );
    let wall = s.wall_ms;
    for c in 0..32 {
        let p = &s.parts[c];
        if p.tasks == 0.0 {
            continue;
        }
        let busy = p.busy / reps as f64 / 1e6;
        println!(
            "  {c:>3}  {}     {:>5.2}   {:>8.4}   {:>8.4}   {:>8.1}   {:>8.1}   {:>8.1}",
            if is_p(c) { "P" } else { "E" },
            p.tasks / reps as f64,
            busy,
            (wall - busy).max(0.0),
            p.lag / p.reps_seen.max(1.0) / 1e3,
            p.tail / p.reps_seen.max(1.0) / 1e3,
            p.macs / p.busy.max(1.0)
        );
    }
}

/// Runs `call` `reps` times with recording on; returns the recorded dispatches.
fn record(n_tasks: usize, reps: usize, call: impl Fn()) -> Vec<Disp> {
    let mut out = Vec::with_capacity(reps);
    for _ in 0..reps {
        ON.store(true, Relaxed);
        let t0 = now();
        call();
        let t1 = now();
        ON.store(false, Relaxed);
        out.push(collect(n_tasks, t1 - t0));
    }
    out
}

#[test]
#[ignore = "release-only instrument: APEIRON_F32_CARRIER=1 cargo test --release --lib --locked gemm_attrib -- --ignored --nocapture"]
fn gemm_attrib_real_inputs() {
    use crate::engine::kern::gemm::{gemm_q8_0, gemm_q8_0_swiglu};
    use crate::engine::kern::gemm_probe::{hooks, rebuild, take_k, wait_quiet, Load};
    use crate::engine::loader::Engine;
    use crate::engine::KernelMode;
    use crate::format::quant::GGML_TYPE_Q8_0;
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni")) {
        println!("gemm_attrib: host lacks AVX-VNNI");
        return;
    }
    let e = Engine::open(crate::engine::kern::gemm_probe::MODEL).expect("open model");
    let ids = crate::engine::kern::gemm_probe::read_ids();
    let mut st = e.state(ids.len() + 1).expect("state");
    e.prefill_state(&mut st, &ids, KernelMode::Q8Act)
        .expect("warm");
    hooks::start(2);
    let mut st = e.state(ids.len() + 1).expect("state");
    e.prefill_state(&mut st, &ids, KernelMode::Q8Act)
        .expect("prefill");
    let cap = hooks::stop();
    let reps: usize = std::env::var("APEIRON_ATTRIB_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    let pick = |kind: hooks::Kind, n_in: usize, n_out: usize| {
        cap.iter()
            .filter(|c| c.kind == kind && c.n_in == n_in && c.parts[0].n_out == n_out && c.k == 64)
            .nth(12)
            .expect("call")
    };
    let down = pick(hooks::Kind::Gemm, 3584, 1024);
    let out = pick(hooks::Kind::Gemm, 2048, 1024);
    let swi = pick(hooks::Kind::Swiglu, 1024, 3584);
    println!(
        "== gemm_attrib == pool width {}",
        crate::engine::chart::pool::Pool::global().width()
    );
    let ks: Vec<usize> = std::env::var("APEIRON_ATTRIB_K").ok().map_or_else(
        || vec![64, 200],
        |v| v.split(',').filter_map(|x| x.parse().ok()).collect(),
    );
    let attempts: usize = std::env::var("APEIRON_ATTRIB_ATT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let rpts = [crate::engine::kern::gemm::ROWS_PER_TASK];
    for (name, c) in [("down", down), ("attn_out", out), ("swiglu", swi)] {
        if std::env::var("APEIRON_ATTRIB_SHAPES").is_ok_and(|v| !v.split(',').any(|x| x == name)) {
            continue;
        }
        for &k in &ks {
            let cols = take_k(&c.cols, k);
            let (qs, _) = rebuild(&cols);
            let n_out = c.parts[0].n_out;
            // SAFETY: the engine is alive for the whole test.
            let (wa, wb) = unsafe {
                (
                    crate::engine::kern::gemm_probe::raw(&c.parts[0]),
                    c.parts
                        .get(1)
                        .map(|p| crate::engine::kern::gemm_probe::raw(p)),
                )
            };
            let call = || {
                if let Some(wb) = wb {
                    let _ = std::hint::black_box(gemm_q8_0_swiglu(wa, wb, n_out, c.n_in, &qs));
                } else {
                    let _ = std::hint::black_box(gemm_q8_0(GGML_TYPE_Q8_0, wa, n_out, c.n_in, &qs));
                }
            };
            for _ in 0..30 {
                call();
            }
            for &rpt in &rpts {
                let n_tasks = n_out.div_ceil(rpt);
                let macs_per_row = c.n_in as f64 * k as f64 * if wb.is_some() { 2.0 } else { 1.0 };
                let mut best: Option<(Summary, f64, bool, f64)> = None;
                for _ in 0..attempts {
                    let (quiet, busy) = wait_quiet();
                    let l = Load::begin();
                    let ds = record(n_tasks, reps, call);
                    let foreign = l.foreign();
                    let s = summarize(&ds, macs_per_row);
                    let better = best
                        .as_ref()
                        .is_none_or(|b| (foreign < 0.06, -s.wall_ms) > (b.1 < 0.06, -b.0.wall_ms));
                    if better {
                        best = Some((s, foreign, quiet, busy));
                    }
                }
                let (s, foreign, quiet, busy) = best.expect("one attempt");
                let gmac_call = n_out as f64 * macs_per_row / (s.call_ms * 1e6);
                let gmac_win = n_out as f64 * macs_per_row / (s.wall_ms * 1e6);
                println!(
                    "\n-- {name} k={k} rpt={rpt} n_in={} n_out={n_out} ({n_tasks} tasks) quiet={quiet} busy={busy:.3} foreign={foreign:.3} call GMAC/s {gmac_call:.1} (dispatch-window {gmac_win:.1})",
                    c.n_in
                );
                print_summary(&format!("{name} k={k} rpt={rpt}"), &s, reps);
            }
        }
    }
}

// ---- in-engine versus isolated replay, at the production block width -------------------------

/// Samples mean cpufreq (MHz) of the P and E cores every 5 ms until finished; samples carry [`now`].
struct FreqSampler {
    stop: std::sync::Arc<AtomicBool>,
    h: Option<std::thread::JoinHandle<Vec<(u64, f64, f64)>>>,
}

impl FreqSampler {
    fn start() -> Self {
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let s2 = stop.clone();
        let h = std::thread::spawn(move || {
            let mut v = Vec::new();
            while !s2.load(Relaxed) {
                let t = now();
                let p = crate::engine::kern::gemm_probe::mean_mhz(0, 16);
                let e = crate::engine::kern::gemm_probe::mean_mhz(16, 32);
                v.push((t, p, e));
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            v
        });
        Self { stop, h: Some(h) }
    }
    fn finish(mut self) -> Vec<(u64, f64, f64)> {
        self.stop.store(true, Relaxed);
        self.h.take().map(|h| h.join().unwrap()).unwrap_or_default()
    }
}

/// `(minor faults, voluntary context switches, involuntary context switches)` of this process.
fn rusage() -> (i64, i64, i64) {
    // SAFETY: `getrusage` fills a caller-owned struct; all-zero is a valid initial value.
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: as above.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
    (u.ru_minflt, u.ru_nvcsw, u.ru_nivcsw)
}

fn mean_in(v: &[(u64, f64, f64)], a: u64, b: u64) -> (f64, f64, usize) {
    let s: Vec<_> = v.iter().filter(|x| x.0 >= a && x.0 < b).collect();
    let n = s.len().max(1) as f64;
    (
        s.iter().map(|x| x.1).sum::<f64>() / n,
        s.iter().map(|x| x.2).sum::<f64>() / n,
        s.len(),
    )
}

/// `(label, n_out sum, k, macs, ns)` of one recorded call.
type Row = (String, usize, usize, f64, u64);

fn rows_of(
    cap: &[crate::engine::kern::gemm_probe::hooks::Call],
    names: &std::collections::BTreeMap<usize, String>,
) -> Vec<Row> {
    use crate::engine::kern::gemm_probe::strip_layer;
    cap.iter()
        .map(|c| {
            let n: Vec<String> = c
                .parts
                .iter()
                .map(|p| strip_layer(names.get(&p.ptr).map_or("?", String::as_str)))
                .collect();
            let macs: f64 = c
                .parts
                .iter()
                .map(|p| p.n_out as f64 * c.n_in as f64 * c.k as f64)
                .sum();
            (
                format!("{:?} {} n_in={}", c.kind, n.join("+"), c.n_in),
                c.parts.iter().map(|p| p.n_out).sum(),
                c.k,
                macs,
                c.ns,
            )
        })
        .collect()
}

fn by_label(rows: &[Row]) -> std::collections::BTreeMap<String, (usize, f64, f64)> {
    let mut m = std::collections::BTreeMap::new();
    for r in rows {
        let e = m.entry(r.0.clone()).or_insert((0usize, 0.0f64, 0.0f64));
        e.0 += 1;
        e.1 += r.3;
        e.2 += r.4 as f64;
    }
    m
}

/// In-engine GEMM calls at the production block width against an isolated back-to-back replay of
/// the same calls (same weights, same columns, same order), plus the cold-start ramp.
///
/// Run: `APEIRON_F32_CARRIER=1 cargo test --release --lib --locked gemm_attrib_inengine -- --ignored --nocapture`.
#[test]
#[ignore = "release-only instrument: see the doc comment"]
fn gemm_attrib_inengine() {
    use crate::engine::kern::gemm_probe::{
        hooks, loadavg, rebuild, run_call, tensor_names, wait_quiet, Load,
    };
    use crate::engine::loader::Engine;
    use crate::engine::KernelMode;
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni")) {
        println!("gemm_attrib_inengine: host lacks AVX-VNNI");
        return;
    }
    let e = Engine::open(crate::engine::kern::gemm_probe::MODEL).expect("open model");
    let ids = crate::engine::kern::gemm_probe::read_ids();
    let names = tensor_names(&e);
    let prefill = |e: &Engine| {
        let mut st = e.state(ids.len() + 1).expect("state");
        e.prefill_state(&mut st, &ids, KernelMode::Q8Act)
            .expect("prefill");
    };
    println!(
        "== gemm_attrib_inengine == pp{} block {} pool width {}",
        ids.len(),
        crate::engine::step::prefill_block(),
        crate::engine::chart::pool::Pool::global().width()
    );
    prefill(&e);
    hooks::start(2);
    prefill(&e);
    let cap = hooks::stop();
    println!("captured {} GEMM calls per prefill", cap.len());
    let reps: usize = std::env::var("APEIRON_ATTRIB_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    // ---- A. in-engine: cold start after the idle gate, then hot repeats --------------------
    let sampler = FreqSampler::start();
    let (quiet, busy) = wait_quiet();
    println!("gate quiet={quiet} busy={busy:.3} load {}", loadavg());
    let mut runs: Vec<(Vec<Row>, u64, u64)> = Vec::new();
    let mut ru: Vec<(i64, i64, i64)> = Vec::new();
    let mut foreign = 0.0f64;
    for _ in 0..reps {
        let l = Load::begin();
        let r0 = rusage();
        hooks::start(1);
        let t0 = now();
        prefill(&e);
        let t1 = now();
        let c = hooks::stop();
        let r1 = rusage();
        ru.push((r1.0 - r0.0, r1.1 - r0.1, r1.2 - r0.2));
        foreign = foreign.max(l.foreign());
        runs.push((rows_of(&c, &names), t0, t1));
    }
    let samples = sampler.finish();
    println!(
        "-- A. in-engine prefill, consecutive after the idle gate (max foreign {foreign:.3}) --"
    );
    for (i, (rows, t0, t1)) in runs.iter().enumerate() {
        let gemm_ns: u64 = rows.iter().map(|r| r.4).sum();
        let macs: f64 = rows.iter().map(|r| r.3).sum();
        let (p, en, n) = mean_in(&samples, *t0, *t1);
        let wall = (t1 - t0) as f64 / 1e6;
        println!(
            "prefill {i}: wall {wall:.1} ms ({:.0} tok/s) gemm-call sum {:.1} ms = {:.1} GMAC/s; MHz P {p:.0} E {en:.0} ({n} samples); minor faults {} vol-cs {} invol-cs {}",
            ids.len() as f64 / wall * 1e3,
            gemm_ns as f64 / 1e6,
            macs / gemm_ns as f64,
            ru[i].0,
            ru[i].1,
            ru[i].2
        );
    }
    for (tag, idx) in [("cold(0)", 0usize), ("hot(last)", reps - 1)] {
        let rows = &runs[idx].0;
        println!("  {tag}: per class, GMAC/s over the first / middle / last third of that class's calls:");
        let mut cls: std::collections::BTreeMap<String, Vec<(usize, f64, u64)>> =
            std::collections::BTreeMap::new();
        for (i, r) in rows.iter().enumerate() {
            cls.entry(r.0.clone()).or_default().push((i, r.3, r.4));
        }
        for (l, v) in &cls {
            let n = v.len();
            let third = |lo: usize, hi: usize| {
                let (m, ns) = v[lo..hi]
                    .iter()
                    .fold((0.0, 0u64), |a, x| (a.0 + x.1, a.1 + x.2));
                m / ns.max(1) as f64
            };
            let (a, b) = (n / 3, 2 * n / 3);
            println!(
                "    {l}: {n} calls, {:.1} / {:.1} / {:.1} GMAC/s, total {:.2} ms",
                third(0, a.max(1)),
                third(a, b.max(a + 1)),
                third(b, n),
                v.iter().map(|x| x.2).sum::<u64>() as f64 / 1e6
            );
        }
    }
    // ---- B. isolated replay of the same calls in the same order -----------------------------
    let rebuilt: Vec<_> = cap.iter().map(|c| rebuild(&c.cols).0).collect();
    let mut rep_runs: Vec<Vec<Row>> = Vec::new();
    let mut rforeign = 0.0f64;
    let (quiet, busy) = wait_quiet();
    println!("-- B. isolated replay, same calls/order/weights, back to back (gate quiet={quiet} busy={busy:.3}) --");
    for _ in 0..reps {
        let l = Load::begin();
        hooks::start(1);
        for (c, qs) in cap.iter().zip(&rebuilt) {
            run_call(c, qs);
        }
        let c = hooks::stop();
        rforeign = rforeign.max(l.foreign());
        rep_runs.push(rows_of(&c, &names));
    }
    println!("max foreign {rforeign:.3}");
    let hot_in = by_label(&runs[reps - 1].0);
    let cold_in = by_label(&runs[0].0);
    let hot_rep = by_label(&rep_runs[reps - 1]);
    println!("class: calls | in-engine cold ms, GMAC/s | in-engine hot ms, GMAC/s | replay hot ms, GMAC/s | hot in-engine / replay");
    for (l, (n, m, ns)) in &hot_in {
        let (_, _, cns) = cold_in[l];
        let (_, rm, rns) = hot_rep[l];
        println!(
            "  {l}: {n} | {:.2} {:.1} | {:.2} {:.1} | {:.2} {:.1} | {:.3}",
            cns / 1e6,
            m / cns,
            ns / 1e6,
            m / ns,
            rns / 1e6,
            rm / rns,
            ns / rns
        );
    }
    let tot = |m: &std::collections::BTreeMap<String, (usize, f64, f64)>| {
        m.values().fold((0.0, 0.0), |a, x| (a.0 + x.1, a.1 + x.2))
    };
    let (im, ins) = tot(&hot_in);
    let (rm, rns) = tot(&hot_rep);
    println!(
        "TOTAL hot in-engine {:.2} ms {:.1} GMAC/s | replay {:.2} ms {:.1} GMAC/s | ratio {:.3}",
        ins / 1e6,
        im / ins,
        rns / 1e6,
        rm / rns,
        ins / rns
    );
    // ---- C. traced prefill: phase ms against the hooked GEMM-call ms of the same pass ---------
    let (quiet, busy) = wait_quiet();
    println!("-- C. traced prefill (gate quiet={quiet} busy={busy:.3}) --");
    for _ in 0..2 {
        prefill(&e);
    }
    for pass in 0..3 {
        let l = Load::begin();
        crate::engine::trace::set_enabled(true);
        crate::engine::trace::reset();
        hooks::start(1);
        let t0 = now();
        prefill(&e);
        let t1 = now();
        let c = hooks::stop();
        let stats = crate::engine::trace::snapshot();
        crate::engine::trace::set_enabled(false);
        let foreign = l.foreign();
        let wall = (t1 - t0) as f64 / 1e6;
        println!(
            "traced pass {pass}: wall {wall:.1} ms ({:.0} tok/s), foreign {foreign:.3}",
            ids.len() as f64 / wall * 1e3
        );
        if pass == 2 {
            for s in stats.iter().filter(|s| !s.encloses) {
                println!(
                    "  phase {:<16} {:>8.2} ms calls {:>4} {}",
                    s.name,
                    s.ns as f64 / 1e6,
                    s.calls,
                    if s.macs > 0 {
                        format!("{:.1} GMAC/s", s.macs as f64 / s.ns as f64)
                    } else {
                        String::new()
                    }
                );
            }
            for (l, (n, m, ns)) in by_label(&rows_of(&c, &names)) {
                println!(
                    "  hooked {l}: {n} calls {:.2} ms {:.1} GMAC/s",
                    ns / 1e6,
                    m / ns
                );
            }
        }
    }
}

/// Task-shape sweep at the production block width: for the six prefill shape classes, time the
/// real call (layer-12 weights, real columns, k = `APEIRON_ATTRIB_K`, default 200) under each
/// rows-per-task in `APEIRON_ATTRIB_RPT` (default `2,4,8,16,32`), rotating the order every round.
///
/// Run: `APEIRON_F32_CARRIER=1 cargo test --release --lib --locked gemm_attrib_shapes -- --ignored --nocapture`.
#[test]
#[ignore = "release-only instrument: see the doc comment"]
fn gemm_attrib_shapes() {
    use crate::engine::kern::gemm::TASK_ROWS_OVERRIDE;
    use crate::engine::kern::gemm_probe::{
        hooks, rebuild, run_call, take_k, tensor_names, wait_quiet, Load,
    };
    use crate::engine::loader::Engine;
    use crate::engine::KernelMode;
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni")) {
        println!("gemm_attrib_shapes: host lacks AVX-VNNI");
        return;
    }
    let env_list = |k: &str, d: &str| -> Vec<usize> {
        std::env::var(k)
            .unwrap_or_else(|_| d.to_string())
            .split(',')
            .filter_map(|x| x.parse().ok())
            .collect()
    };
    let ks = env_list("APEIRON_ATTRIB_K", "200");
    let rpts = env_list("APEIRON_ATTRIB_RPT", "2,4,8,16,32");
    let rounds: usize = std::env::var("APEIRON_ATTRIB_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9);
    let per: usize = std::env::var("APEIRON_ATTRIB_PER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15);
    if std::env::var("APEIRON_ATTRIB_NOTHP").is_ok() {
        // SAFETY: `prctl(PR_SET_THP_DISABLE)` only changes this process's THP policy.
        unsafe { libc::prctl(libc::PR_SET_THP_DISABLE, 1, 0, 0, 0) };
    }
    // The engine keeps each call's outputs alive while other allocations sit above them, so the
    // heap is never trimmed between GEMM calls; without this, run_call's immediate drop trims and
    // the next call takes ~1400 page faults inside its tasks (measured: 2.8 ms vs 1.44 ms swiglu).
    // SAFETY: `mallopt` only sets allocator parameters.
    unsafe {
        libc::mallopt(libc::M_TRIM_THRESHOLD, 1 << 30);
        libc::mallopt(libc::M_TOP_PAD, 64 << 20);
    }
    let e = Engine::open(crate::engine::kern::gemm_probe::MODEL).expect("open model");
    let ids = crate::engine::kern::gemm_probe::read_ids();
    let names = tensor_names(&e);
    let mut st = e.state(ids.len() + 1).expect("state");
    e.prefill_state(&mut st, &ids, KernelMode::Q8Act)
        .expect("warm");
    hooks::start(2);
    let mut st = e.state(ids.len() + 1).expect("state");
    e.prefill_state(&mut st, &ids, KernelMode::Q8Act)
        .expect("prefill");
    let cap = hooks::stop();
    let nth: usize = std::env::var("APEIRON_ATTRIB_NTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let mut seen: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut picks = Vec::new();
    for (i, r) in rows_of(&cap, &names).iter().enumerate() {
        let n = seen.entry(r.0.clone()).or_insert(0);
        *n += 1;
        if *n == nth {
            picks.push(i);
        }
    }
    let shapes: Vec<String> = std::env::var("APEIRON_ATTRIB_SHAPES")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    println!(
        "== gemm_attrib_shapes == pool width {} rounds {rounds} calls/round {per}",
        crate::engine::chart::pool::Pool::global().width()
    );
    for &ci in &picks {
        let c = &cap[ci];
        let label = rows_of(std::slice::from_ref(c), &names).remove(0).0;
        let short = label.split_whitespace().next().unwrap_or("").to_string()
            + &format!(
                " n_in={} n_out={:?}",
                c.n_in,
                c.parts.iter().map(|p| p.n_out).collect::<Vec<_>>()
            );
        if !shapes.is_empty() && !shapes.iter().any(|s| label.contains(s.as_str())) {
            continue;
        }
        for &k in &ks {
            let cols = take_k(&c.cols, k);
            let (qs, _) = rebuild(&cols);
            // `APEIRON_ATTRIB_COPY=huge|small`: run on a copy of the weights in a 2 MiB-aligned
            // mapping with MADV_HUGEPAGE (`huge`) or MADV_NOHUGEPAGE (`small`).
            let copied = std::env::var("APEIRON_ATTRIB_COPY").ok().map(|m| {
                let huge = m == "huge";
                let parts = c
                    .parts
                    .iter()
                    .map(|p| {
                        let al = 2usize << 20;
                        let len = p.len.div_ceil(al) * al;
                        // SAFETY: anonymous private mapping, leaked for the test's lifetime; the
                        // copy reads `p.len` bytes of the live engine buffer into it.
                        unsafe {
                            let m = libc::mmap(
                                std::ptr::null_mut(),
                                len + al,
                                libc::PROT_READ | libc::PROT_WRITE,
                                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                                -1,
                                0,
                            );
                            assert_ne!(m, libc::MAP_FAILED);
                            let base = (m as usize).div_ceil(al) * al;
                            libc::madvise(
                                base as *mut libc::c_void,
                                len,
                                if huge {
                                    libc::MADV_HUGEPAGE
                                } else {
                                    libc::MADV_NOHUGEPAGE
                                },
                            );
                            std::ptr::copy_nonoverlapping(
                                p.ptr as *const u8,
                                base as *mut u8,
                                p.len,
                            );
                            hooks::Part {
                                ptr: base,
                                len: p.len,
                                n_out: p.n_out,
                            }
                        }
                    })
                    .collect();
                hooks::Call {
                    kind: c.kind,
                    n_in: c.n_in,
                    k: c.k,
                    parts,
                    cols: Vec::new(),
                    ctr: [0; hooks::N_CTR],
                    ok: false,
                    ns: 0,
                }
            });
            let call = copied.as_ref().unwrap_or(c);
            let macs: f64 = c
                .parts
                .iter()
                .map(|p| p.n_out as f64 * c.n_in as f64 * k as f64)
                .sum();
            let mut samples: Vec<Vec<f64>> = vec![Vec::new(); rpts.len()];
            let mut clean_flags: Vec<Vec<bool>> = vec![Vec::new(); rpts.len()];
            let mut walls: Vec<Vec<(f64, f64)>> = vec![Vec::new(); rpts.len()];
            let mut foreign = 0.0f64;
            let (quiet, busy) = wait_quiet();
            for r in 0..rounds {
                for j in 0..rpts.len() {
                    let ix = (j + r) % rpts.len();
                    TASK_ROWS_OVERRIDE.store(rpts[ix], Relaxed);
                    for _ in 0..3 {
                        run_call(call, &qs);
                    }
                    let mut v = Vec::with_capacity(per);
                    let mut last_f = 1.0;
                    let mut last_wall = 0.0;
                    let mut last_faults = 0.0;
                    for attempt in 0..4 {
                        let l = Load::begin();
                        v.clear();
                        let mut wall_sum = 0.0;
                        let f0 = rusage().0;
                        for _ in 0..per {
                            hooks::start(1);
                            let t = Instant::now();
                            run_call(call, &qs);
                            wall_sum += t.elapsed().as_secs_f64() * 1e3;
                            let c = hooks::stop();
                            v.push(c[0].ns as f64 / 1e6);
                        }
                        last_wall = wall_sum / per as f64;
                        last_faults = (rusage().0 - f0) as f64 / per as f64;
                        let f = l.foreign();
                        last_f = f;
                        foreign = foreign.max(f);
                        if f < 0.05 || attempt == 3 {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(500));
                    }
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    samples[ix].push(v[v.len() / 2]);
                    clean_flags[ix].push(last_f < 0.05);
                    walls[ix].push((last_wall, last_faults));
                }
            }
            TASK_ROWS_OVERRIDE.store(0, Relaxed);
            println!(
                "-- {short} k={k} (gate quiet={quiet} busy={busy:.3}; max foreign {foreign:.3})"
            );
            let med = |v: &Vec<f64>| {
                let mut s = v.clone();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                s[s.len() / 2]
            };
            let bix = rpts.iter().position(|&x| x == 16).unwrap_or(0);
            let base = samples[bix].clone();
            for (ix, &rpt) in rpts.iter().enumerate() {
                let good: Vec<usize> = (0..rounds)
                    .filter(|&r| clean_flags[ix][r] && clean_flags[bix][r])
                    .collect();
                let m = med(&good.iter().map(|&r| samples[ix][r]).collect());
                let ratios: Vec<f64> = good.iter().map(|&r| base[r] / samples[ix][r]).collect();
                let ratios = if ratios.is_empty() {
                    vec![f64::NAN]
                } else {
                    ratios
                };
                let w = walls[ix].last().copied().unwrap_or((0.0, 0.0));
                let tasks: Vec<usize> = c.parts.iter().map(|p| p.n_out.div_ceil(rpt)).collect();
                println!(
                    "   rpt {rpt:>3} tasks {tasks:?}: median over {} clean rounds {m:.4} ms = {:.1} GMAC/s; speedup vs rpt16 per-round median {:.3} (min {:.3} max {:.3}); last-round wall incl. drop {:.3} ms, minor faults/call {:.0}",
                    good.len(),
                    macs / m / 1e6,
                    med(&ratios),
                    ratios.iter().copied().fold(f64::INFINITY, f64::min),
                    ratios.iter().copied().fold(0.0, f64::max),
                    w.0,
                    w.1
                );
            }
        }
    }
}
