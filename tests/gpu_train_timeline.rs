//! The 56-dispatch training-step tape with in-buffer device timestamps: per-dispatch spans, the
//! barrier gaps between them, and the host split of `Trainer::step`.
//!
//! The instrument is one command buffer. A timestamp is written before the first copy, before and
//! after every one of the 56 dispatches, and after the readback barriers, so the device figure is
//! the buffer's own execution and not two submissions queue-ordered around `step()`; that
//! cross-submission instrument measured host wall plus queue wait and is the defect this test
//! replaces. The step's host split is `[bias write, pool, record, submit+idle]`.
//!
//! The Adam pipelines are built in `Trainer::new` and replayed at every step, so the bias-write
//! phase is two f32 stores: the gate asserts it is below 100 µs, the step's whole host wall below
//! 5 ms, and the control (nine `adam_update_dispatch` builds per step, the pre-fix path) at or
//! above 1 ms.
//!
//! Two campaigns of 3 warm + 7 reps are reported; each row is the mean over its 7 reps with the
//! observed min/max.
//!
//! ```text
//! flock /home/socol/oc-tmp/gpu.lock -c 'cargo test --release --features gpu \
//!   --test gpu_train_timeline -- --nocapture --test-threads=1'
//! ```
//!
//! Board row: `apeiron-gpu-train-w1-t05-GVrduzS5mnRVc4oeQ1xRD7`.
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_train_step timeline test"
)]

use apeiron::gpu::decode::GpuCtx;
use apeiron::gpu::nn::adam_update_dispatch;
use apeiron::gpu::nn::train::{HostWeights, LayerCfg, StepProfile, Trainer, STEP_DISPATCHES};
use apeiron::gpu::vulkan::*;
use std::time::Instant;

/// Production layer shape (Qwen2.5-0.5B layer 0).
const T: usize = 64;
const I: usize = 896;
const H: usize = 4864;
/// KL stand-in: `kl_row` reads the first `v` words of `x2` as the row logits.
const V: usize = 64;
const NH: usize = 14;
const NKV: usize = 2;
const HD: usize = 64;
const QW: usize = NH * HD;
const KW: usize = NKV * HD;

/// Warmup steps per campaign, not reported.
const WARM: usize = 3;
/// Reported timed steps per campaign.
const REPS: usize = 7;
/// `round(1e-3·2^32)`.
const LR_Q32: i64 = 4_294_967;

/// Fixed-seed LCG (Knuth's MMIX constants), uniform in `[-0.5, 0.5)` — `HostWeights::lcg`'s stream.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        u - 0.5
    }
}

fn real_to_q32(v: f64) -> i64 {
    (v * 4_294_967_296.0).round() as i64
}

fn to_f32(v: i64) -> f32 {
    (v as f64 / 4_294_967_296.0) as f32
}

/// mean, min, max.
fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mean, min, max)
}

fn fmt_stats(v: &[f64]) -> String {
    let (mean, min, max) = stats(v);
    format!("{mean:.1} µs (min {min:.1}, max {max:.1}, n {})", v.len())
}

fn sum(v: &[f64]) -> f64 {
    v.iter().sum()
}

fn gpu_identity() -> String {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "device name/driver unavailable (nvidia-smi absent)".to_string(),
    }
}

/// One campaign: `WARM` uncounted steps then `REPS` traced steps.
struct Campaign {
    wall: Vec<f64>,
    total: Vec<f64>,
    dispatch_sum: Vec<f64>,
    gap_sum: Vec<f64>,
    prologue: Vec<f64>,
    tail: Vec<f64>,
    host: Vec<[f64; 4]>,
    dispatch: Vec<Vec<f64>>,
    labels: Vec<&'static str>,
}

fn campaign(trainer: &mut Trainer, g: &GpuCtx, pool: &QueryPool, label: &str) -> Campaign {
    let mut out = Campaign {
        wall: Vec::with_capacity(REPS),
        total: Vec::with_capacity(REPS),
        dispatch_sum: Vec::with_capacity(REPS),
        gap_sum: Vec::with_capacity(REPS),
        prologue: Vec::with_capacity(REPS),
        tail: Vec::with_capacity(REPS),
        host: Vec::with_capacity(REPS),
        dispatch: Vec::with_capacity(REPS),
        labels: Vec::new(),
    };
    for w in 0..WARM {
        let t0 = Instant::now();
        trainer.step(g);
        println!(
            "campaign {label} warm {w}: wall {:.1} µs",
            t0.elapsed().as_secs_f64() * 1e6
        );
    }
    for r in 0..REPS {
        let t0 = Instant::now();
        let p: StepProfile = trainer.step_traced(g, pool);
        let wall = t0.elapsed().as_secs_f64() * 1e6;
        assert_eq!(
            p.dispatch_us.len(),
            STEP_DISPATCHES,
            "one span per dispatch"
        );
        assert_eq!(p.gap_us.len(), STEP_DISPATCHES - 1, "one gap per barrier");
        assert_eq!(p.labels.len(), STEP_DISPATCHES, "one label per dispatch");
        assert!(
            p.dispatch_us
                .iter()
                .chain(&p.gap_us)
                .all(|v| v.is_finite() && *v >= 0.0),
            "campaign {label} rep {r}: non-finite or negative interval"
        );
        let loss = trainer.losses();
        assert!(
            loss.iter().all(|l| l.is_finite() && *l >= -1e-4),
            "campaign {label} rep {r}: loss not finite non-negative"
        );
        out.labels = p.labels.clone();
        println!(
            "campaign {label} rep {r}: wall {wall:.1} µs; device {:.1} µs; dispatch sum {:.1}; \
             gap sum {:.1}; host [bias {:.1}, pool {:.1}, record {:.1}, submit {:.1}]",
            p.total_us,
            sum(&p.dispatch_us),
            sum(&p.gap_us),
            p.host_us[0],
            p.host_us[1],
            p.host_us[2],
            p.host_us[3]
        );
        out.wall.push(wall);
        out.total.push(p.total_us);
        out.dispatch_sum.push(sum(&p.dispatch_us));
        out.gap_sum.push(sum(&p.gap_us));
        out.prologue.push(p.prologue_us);
        out.tail.push(p.tail_us);
        out.host.push(p.host_us);
        out.dispatch.push(p.dispatch_us);
    }
    out
}

fn report(label: &str, c: &Campaign) {
    println!("CAMPAIGN {label}");
    println!(" device span           {}", fmt_stats(&c.total));
    println!(" dispatch sum (56)     {}", fmt_stats(&c.dispatch_sum));
    println!(" barrier sum (55)      {}", fmt_stats(&c.gap_sum));
    println!(" prologue              {}", fmt_stats(&c.prologue));
    println!(" readback tail         {}", fmt_stats(&c.tail));
    let mean_of = |f: fn(&[f64; 4]) -> f64| -> Vec<f64> { c.host.iter().map(|h| f(h)).collect() };
    println!(" host bias write       {}", fmt_stats(&mean_of(|h| h[0])));
    println!(" host pool             {}", fmt_stats(&mean_of(|h| h[1])));
    println!(" host record           {}", fmt_stats(&mean_of(|h| h[2])));
    println!(" host submit+idle      {}", fmt_stats(&mean_of(|h| h[3])));
    let host_total: Vec<f64> = c.host.iter().map(|h| h.iter().sum::<f64>()).collect();
    println!(" host split sum        {}", fmt_stats(&host_total));
    println!(" host wall (external)  {}", fmt_stats(&c.wall));
    // Per-dispatch mean across the campaign's reps, top ten by device span.
    let n = c.dispatch.first().map_or(0, Vec::len);
    let mean: Vec<f64> = (0..n)
        .map(|k| c.dispatch.iter().map(|d| d[k]).sum::<f64>() / c.dispatch.len() as f64)
        .collect();
    let mut top: Vec<usize> = (0..n).collect();
    top.sort_by(|&a, &b| mean[b].total_cmp(&mean[a]));
    println!(" top ten dispatch spans:");
    for k in top.into_iter().take(10) {
        println!("   {:9.1} µs  {} [{k}]", mean[k], c.labels[k]);
    }
}

/// The removed per-step phase: one and nine `adam_update_dispatch` builds with every input ready.
/// Returns `(one-pipeline mean, nine-pipeline mean)` in µs over three runs each.
unsafe fn build_control(g: &GpuCtx, cfg: &LayerCfg) -> (f64, f64) {
    let (i, h) = (cfg.i as usize, cfg.h as usize);
    let widths = [QW * i, KW * i, KW * i, i * QW, h * i, h * i, i * h, i, i];
    let max = widths.iter().copied().max().unwrap_or(1);
    let b = g.dev(max * 4);
    let mut one = Vec::with_capacity(3);
    let mut nine = Vec::with_capacity(3);
    // Distinct, process-unique ordinals in the band where both bias factors are distinct f32
    // values, so neither the in-process pipeline cache nor the driver's on-disk shader cache can
    // serve a previous build: the pre-fix step built a new module at every step ordinal, which is
    // the same condition.
    let base = 100
        + std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos() % 500);
    let mut t = base;
    println!("control ordinal base {t}");
    // The first pipeline creation of the process pays driver lazy init; discard it.
    let s = adam_update_dispatch(g, b, b, b, b, widths[0] as u32, LR_Q32, t);
    std::hint::black_box(&s);
    t += 1;
    for _ in 0..3 {
        let t0 = Instant::now();
        let s = adam_update_dispatch(g, b, b, b, b, widths[0] as u32, LR_Q32, t);
        one.push(t0.elapsed().as_secs_f64() * 1e6);
        std::hint::black_box(&s);
        t += 1;
        let t0 = Instant::now();
        for n in 0..9 {
            let s = adam_update_dispatch(g, b, b, b, b, widths[n] as u32, LR_Q32, t);
            std::hint::black_box(&s);
        }
        nine.push(t0.elapsed().as_secs_f64() * 1e6);
        t += 1;
    }
    let (o, _, _) = stats(&one);
    let (n9, n9min, n9max) = stats(&nine);
    println!(
        "control runs: one build {:.1}, {:.1}, {:.1} µs; nine builds {:.1}, {:.1}, {:.1} µs",
        one[0], one[1], one[2], nine[0], nine[1], nine[2]
    );
    println!(
        "control means: one pipeline {o:.1} µs; nine {n9:.1} µs (min {n9min:.1}, max {n9max:.1}); \
         per pipeline from the nine {:.1} µs",
        n9 / 9.0
    );
    (o, n9)
}

#[test]
fn gpu_train_step_timeline() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: no Vulkan device: {e}");
            return;
        }
    };
    println!(
        "machine: [{}]; timestamp_period {} ns ({} bit); instrument: one command buffer, a \
         timestamp before and after every dispatch; gpu lock /home/socol/oc-tmp/gpu.lock",
        gpu_identity(),
        vk.timestamp_period,
        vk.timestamp_valid_bits
    );
    assert!(
        vk.timestamp_valid_bits >= 32 && vk.timestamp_period > 0.0,
        "device reports no usable timestamps"
    );

    let cfg = LayerCfg::new(
        T as u32, I as u32, H as u32, V as u32, NH as u32, NKV as u32, HD as u32,
    )
    .expect("layer cfg");
    let seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rng = Lcg(seed);
    let x: Vec<f32> = (0..T * I)
        .map(|_| to_f32(real_to_q32(rng.unit())))
        .collect();
    let zt: Vec<f32> = (0..T * V)
        .map(|_| to_f32(real_to_q32(rng.unit())))
        .collect();
    let hw = HostWeights::lcg(&cfg, seed);

    let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let mut trainer = Trainer::new(&g, cfg, &hw);
    let pool = unsafe { vk.timestamp_pool(StepProfile::QUERIES) };
    trainer.set_batch(&x, &zt);

    let a = campaign(&mut trainer, &g, &pool, "A");
    let (one_build, nine_build) = unsafe { build_control(&g, &cfg) };
    let b = campaign(&mut trainer, &g, &pool, "B");
    report("A", &a);
    report("B", &b);

    let post = stats(&a.wall).0;
    let post_b = stats(&b.wall).0;
    let bias_max = {
        let a_max = stats(&a.host.iter().map(|h| h[0]).collect::<Vec<_>>()).2;
        let b_max = stats(&b.host.iter().map(|h| h[0]).collect::<Vec<_>>()).2;
        a_max.max(b_max)
    };
    println!(
        "CONTROL pre-fix nine Adam builds: one pipeline {one_build:.1} µs, nine {nine_build:.1} µs \
         per step (3 runs); post-fix host wall A {post:.1} µs, B {post_b:.1} µs; pre-fix wall proxy \
         A {:.1} µs, B {:.1} µs",
        post + nine_build,
        post_b + nine_build
    );

    // Post-fix gate: the step builds no pipeline and its host wall is below 5 ms.
    assert!(
        bias_max < 100.0,
        "the bias-write phase is not a two-word store (max {bias_max:.1} µs)"
    );
    assert!(
        post < 5_000.0 && post_b < 5_000.0,
        "post-fix host wall {post:.1} / {post_b:.1} µs is not below 5 ms"
    );
    // Control: the pre-fix path rebuilds nine pipelines each step, at or above 1 ms.
    assert!(
        nine_build >= 1_000.0,
        "control nine Adam builds {nine_build:.1} µs is not at or above 1 ms"
    );
    // The dispatch tape is the 56 the composition names.
    assert_eq!(STEP_DISPATCHES, 56);
    assert_eq!(a.labels.len(), 56);
    assert_eq!(b.labels.len(), 56);
    // The device span is carried by the dispatches, not by the host gap: dispatch sums are a
    // non-trivial share of the buffer's own span.
    assert!(
        stats(&a.dispatch_sum).0 > 0.5 * stats(&a.total).0,
        "dispatch sum {:.1} is not more than half the device span {:.1}",
        stats(&a.dispatch_sum).0,
        stats(&a.total).0
    );
}
