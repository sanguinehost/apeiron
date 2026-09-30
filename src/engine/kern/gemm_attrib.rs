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
