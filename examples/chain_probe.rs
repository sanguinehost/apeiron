//! FACES-H5 — dispatch-chain overlap probe (lane FACES-H5).
//!
//! Implements the arms of `escalation/FACES-R1-H5-PREDICTION.md` exactly: `E` (empty graph), `S`
//! (one `add_inplace` dispatch), `D2`/`D16` (a 2/16-long dependent chain on one x — B reads A's
//! write) and `I2`/`I16` (2/16 independent dispatches on distinct x). The instrument is
//! `Gpu::time` (device timestamps around one recorded command buffer); `PREFILL_BAR` selects the
//! inter-dispatch barrier inside the runtime (0 = shipped `vkCmdPipelineBarrier`, 9 = no barrier).
//! Kernel: `simt_nn::add_inplace_module(n, 256)` (`x[i] += y[i]`; binding 0 = x rw, 1 = y), local
//! 256, groups `((n+255)/256, 1)`. Sizes `n = 2^14` and `2^22`; `x0[i] = (i % 7)`, `y = 1.0`.
//!
//! Cost (AGENTS §8). Per process, per n, the timed population is `REPS` reps × 6 arms plus `WARMUP`
//! discarded rounds, each round re-uploading the arm's x buffers (an O(n) host→device staging copy)
//! and recording one command buffer holding ≤ 16 dispatches; the total host-recorded dispatches are
//! ≤ `(REPS + WARMUP) × 6 × 16` per n. Host memory is O(n): the `x0`/`y` vectors plus the device x
//! buffers (1, 2 or 16 of them). `time` reports the device timestamp delta of the recorded graph,
//! so the reset copy is outside the measured interval.
//!
//! Env: `CHAIN_REPS` (default 300), `CHAIN_LOG2N` (comma-separated `log2 n`, default `14,22`),
//! `PREFILL_BAR` (read once, by the runtime).
//!
//! Run: `PREFILL_BAR=<b> cargo run --release --features gpu --example chain_probe`

#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    reason = "measurement harness: raw device-buffer images and numeric casts are the point"
)]

use apeiron::gpu::runtime::bar_mode;
use apeiron::gpu::{Buffer, Dispatch, Gpu, Pipeline};
use apeiron::simt_nn::add_inplace_module;

/// Workgroup size of `add_inplace_module`.
const LOCAL: u32 = 256;
/// Timed reps per arm (env `CHAIN_REPS`).
const DEFAULT_REPS: usize = 300;
/// Discarded warm-up rounds per arm before the timed reps.
const WARMUP: usize = 20;
/// Decode-class and prefill-class sizes.
const SIZES: [u32; 2] = [1 << 14, 1 << 22];

/// The six arms of the prediction.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    E,
    S,
    D2,
    I2,
    D16,
    I16,
}

impl Arm {
    const ALL: [Arm; 6] = [Arm::E, Arm::S, Arm::D2, Arm::I2, Arm::D16, Arm::I16];

    fn name(self) -> &'static str {
        match self {
            Arm::E => "E",
            Arm::S => "S",
            Arm::D2 => "D2",
            Arm::I2 => "I2",
            Arm::D16 => "D16",
            Arm::I16 => "I16",
        }
    }

    /// Dispatch count: E is the empty graph, S a single dispatch, D2/I2 two, D16/I16 sixteen.
    fn chains(self) -> usize {
        match self {
            Arm::E => 0,
            Arm::S => 1,
            Arm::D2 | Arm::I2 => 2,
            Arm::D16 | Arm::I16 => 16,
        }
    }

    /// Independent arms dispatch to distinct x buffers; dependent arms chain on one x.
    fn independent(self) -> bool {
        matches!(self, Arm::I2 | Arm::I16)
    }

    fn index(self) -> usize {
        Arm::ALL.iter().position(|&a| a == self).expect("in ALL")
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Workgroup counts for one `add_inplace` dispatch over `n` elements.
fn groups(n: u32) -> (u32, u32) {
    (n.div_ceil(LOCAL), 1)
}

/// SPIR-V words for `add_inplace_module(n, LOCAL)`, little-endian byte image.
fn spv(n: u32) -> Vec<u8> {
    add_inplace_module(n, LOCAL)
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect()
}

/// One arm's device state: the x buffers to reset and the recorded dispatch list.
struct ArmRun {
    arm: Arm,
    xs: Vec<Buffer>,
    dispatches: Vec<Dispatch>,
}

fn build_arm(gpu: &Gpu, pipe: &Pipeline, y: &Buffer, x0: &[f32], n: u32, arm: Arm) -> ArmRun {
    let g = groups(n);
    let nbuf = match arm {
        Arm::E => 0,
        Arm::S | Arm::D2 | Arm::D16 => 1,
        Arm::I2 | Arm::I16 => arm.chains(),
    };
    let xs: Vec<Buffer> = (0..nbuf).map(|_| gpu.upload(x0)).collect();
    let dispatches = if arm.chains() == 0 {
        Vec::new()
    } else if arm.independent() {
        xs.iter()
            .map(|x| Dispatch {
                pipe: *pipe,
                dset: gpu.bind(pipe, &[x, y]),
                groups: g,
            })
            .collect()
    } else {
        let d = Dispatch {
            pipe: *pipe,
            dset: gpu.bind(pipe, &[&xs[0], y]),
            groups: g,
        };
        vec![d; arm.chains()]
    };
    ArmRun {
        arm,
        xs,
        dispatches,
    }
}

/// Reset an arm's x buffers to the initial image (outside the timed interval).
fn reset(gpu: &Gpu, xs: &[Buffer], x0: &[f32]) {
    for x in xs {
        gpu.write_into(x, x0);
    }
}

/// The timestamp-pair of one arm. `E` is `time(&[])` when the device accepts an empty graph;
/// otherwise it is a timestamp pair wrapping one zero-group dispatch (stated in the output).
fn time_arm(gpu: &Gpu, r: &ArmRun, empty_ok: bool, zero: &Dispatch) -> f64 {
    if r.dispatches.is_empty() {
        if empty_ok {
            gpu.time(&[])
        } else {
            gpu.time(&[*zero])
        }
    } else {
        gpu.time(&r.dispatches)
    }
}

/// Linear-interpolated percentile of an ascending slice, `p` in `[0, 1]`.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = p * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

/// Median and inter-quartile range of one arm's samples; sorts in place.
fn median_iqr(samples: &mut [f64]) -> (f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite samples"));
    (
        percentile(samples, 0.5),
        percentile(samples, 0.75) - percentile(samples, 0.25),
    )
}

/// P5: run a 16-long dependent chain on a host-readable x and count entries not equal to `x0+16`.
fn d16_correctness(gpu: &Gpu, pipe: &Pipeline, n: u32, x0: &[f32]) -> usize {
    let (xb, xptr) = gpu.host_read_buffer(x0.len() * 4);
    gpu.write_into(&xb, x0);
    let y = gpu.upload(&vec![1.0f32; x0.len()]);
    let ds = gpu.bind(pipe, &[&xb, &y]);
    let d = Dispatch {
        pipe: *pipe,
        dset: ds,
        groups: groups(n),
    };
    gpu.run(&[d; 16]);
    let p = xptr.cast::<f32>();
    let mut mism = 0usize;
    for (i, &v) in x0.iter().enumerate() {
        let got = unsafe { p.add(i).read_volatile() };
        if got != v + 16.0 {
            mism += 1;
        }
    }
    mism
}

/// Time every arm at one `n`, print the per-arm lines, the derived overlap/slope lines and the
/// local prediction checks, then the P5 correctness count.
fn run_size(gpu: &Gpu, n: u32, bar: u32, reps: usize, empty_ok: bool) {
    let nu = n as usize;
    let x0: Vec<f32> = (0..nu).map(|i| (i % 7) as f32).collect();
    let y = gpu.upload(&vec![1.0f32; nu]);
    let pipe = gpu.pipeline(&spv(n), 2);
    let arms: Vec<ArmRun> = Arm::ALL
        .iter()
        .map(|&a| build_arm(gpu, &pipe, &y, &x0, n, a))
        .collect();

    // Zero-group fallback for E: the S arm's buffers dispatched with 0 workgroups (no writes).
    let s_arm = arms
        .iter()
        .find(|a| a.arm == Arm::S)
        .expect("S arm present");
    let zero = Dispatch {
        pipe,
        dset: gpu.bind(&pipe, &[&s_arm.xs[0], &y]),
        groups: (0, 1),
    };

    for _ in 0..WARMUP {
        for r in &arms {
            reset(gpu, &r.xs, &x0);
            let _ = time_arm(gpu, r, empty_ok, &zero);
        }
    }

    let mut samples: Vec<Vec<f64>> = arms.iter().map(|_| Vec::with_capacity(reps)).collect();
    for _ in 0..reps {
        for (i, r) in arms.iter().enumerate() {
            reset(gpu, &r.xs, &x0);
            samples[i].push(time_arm(gpu, r, empty_ok, &zero));
        }
    }

    if samples.iter().flatten().any(|t| !t.is_finite()) {
        println!("SKIP n={n} bar={bar}: non-finite device timestamps");
        return;
    }

    let mut med = vec![0f64; arms.len()];
    let mut iqr = vec![0f64; arms.len()];
    for (i, s) in samples.iter_mut().enumerate() {
        let (m, q) = median_iqr(s);
        med[i] = m;
        iqr[i] = q;
    }

    for (i, r) in arms.iter().enumerate() {
        println!(
            "ARM n={n} bar={bar} {} median_us={:.4} iqr_us={:.4}",
            r.arm.name(),
            med[i],
            iqr[i]
        );
    }

    let e = med[Arm::E.index()];
    let single = med[Arm::S.index()] - e;
    let d2 = med[Arm::D2.index()] - e;
    let i2 = med[Arm::I2.index()] - e;
    let o_d = 2.0 * single - d2;
    let o_i = 2.0 * single - i2;
    let slope = (med[Arm::D16.index()] - med[Arm::S.index()]) / 15.0;
    println!("DERIVED n={n} bar={bar} s={single:.4} o_D={o_d:.4} o_I={o_i:.4} slope={slope:.4}");

    // Local checks (the prediction's thresholds; the arm population is only BAR=0 or BAR=9 per run).
    let tol_d = (3.0 * iqr[Arm::D2.index()]).max(0.3);
    let tol_p4 = (3.0 * (iqr[Arm::I2.index()] + iqr[Arm::D2.index()])).max(0.3);
    println!(
        "P1 n={n} bar={bar} o_D<=tol {} ({o_d:.4} <= {tol_d:.4})",
        o_d <= tol_d
    );
    println!(
        "P2 n={n} bar={bar} slope>=0.9 {} ({slope:.4})",
        slope >= 0.9
    );
    println!("P3 n={n} bar={bar} o_I>=0.25 {} ({o_i:.4})", o_i >= 0.25);
    println!(
        "P4 n={n} bar={bar} |I2-D2|<=tol {} ({:.4} <= {tol_p4:.4})",
        (i2 - d2).abs() <= tol_p4,
        (i2 - d2).abs()
    );

    let mism = d16_correctness(gpu, &pipe, n, &x0);
    println!("D16 n={n} bar={bar} mismatches={mism}");
}

fn main() {
    let bar = bar_mode();
    let reps = env_usize("CHAIN_REPS", DEFAULT_REPS);
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(msg) => {
            println!("SKIP: {msg}");
            return;
        }
    };
    println!("device: {} bar={bar} reps={reps}", gpu.device_name());

    let empty_ok = gpu.time(&[]).is_finite();
    if !empty_ok {
        println!("NOTE: time(&[]) is non-finite; E is measured as one zero-group dispatch");
    }

    let sizes: Vec<u32> = match std::env::var("CHAIN_LOG2N") {
        Ok(v) => v
            .split(',')
            .map(|e| {
                1u32 << e
                    .trim()
                    .parse::<u32>()
                    .expect("CHAIN_LOG2N: log2 n in 0..=24")
            })
            .collect(),
        Err(_) => SIZES.to_vec(),
    };
    for &n in &sizes {
        run_size(&gpu, n, bar, reps, empty_ok);
    }
}
