//! The tiled linear GEMMs (`gpu::nn::fwd::lin`, `gpu::nn::bwd::lin`) against a bit-identical CPU
//! reference and against the kernel they replace.
//!
//! The contract of all three ops is one f32 accumulator per output, seeded `0.0` and updated by
//! `Fma(a, b, acc)` for `k` ascending over the whole reduction. Register tiling changes which lane
//! issues which product and how many products share a staged tile; it must not change the word.
//! Test `a` therefore compares **bit patterns** (`f32::to_bits`) against a CPU reference that runs
//! exactly `acc = a.mul_add(b, acc)` in ascending `k`, at the four small shapes and a 256-row slice
//! of the production shape. Test `b` times the nine training-tape cells with
//! `vkCmdWriteTimestamp2` around the dispatch and reports GFLOP/s = 2·n·i·o/t; the paired
//! replacement ratio over the one-output-per-lane body it replaced is in the lane receipt
//!, not here, because that body no longer exists in the tree.
//!
//! ```text
//! cargo test --release --features gpu --test gpu_linear_tiled -- --test-threads=1 --nocapture
//! ```
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_linear_fwd/gpu_linear_bwd tests"
)]

use apeiron::gpu::decode::{rec_step, GpuCtx, Step};
use apeiron::gpu::nn::fwd::linear_fwd_dispatch;
use apeiron::gpu::nn::{linear_bwd_input_dispatch, linear_bwd_weight_dispatch};
use apeiron::gpu::vulkan::*;

/// Fixed-seed LCG (Knuth's MMIX constants), uniform in `[-0.5, 0.5)`.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        (u - 0.5) as f32
    }
}

fn fill(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..len).map(|_| rng.unit()).collect()
}

/// `y[r,c] = Σ_k x[r·i+k]·w[c·i+k]`, `acc = 0.0`, `k` ascending, `f32::mul_add`.
fn cpu_fwd(x: &[f32], w: &[f32], n: usize, i: usize, o: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * o];
    for r in 0..n {
        let xr = &x[r * i..r * i + i];
        for c in 0..o {
            let wr = &w[c * i..c * i + i];
            let mut acc = 0.0f32;
            for k in 0..i {
                acc = xr[k].mul_add(wr[k], acc);
            }
            y[r * o + c] = acc;
        }
    }
    y
}

/// `dx[r,c] = Σ_k dy[r·o+k]·w[k·i+c]`, `acc = 0.0`, `k` ascending, `f32::mul_add`.
fn cpu_bwd_input(dy: &[f32], w: &[f32], n: usize, i: usize, o: usize) -> Vec<f32> {
    let mut dx = vec![0.0f32; n * i];
    for r in 0..n {
        let dyr = &dy[r * o..r * o + o];
        for c in 0..i {
            let mut acc = 0.0f32;
            for k in 0..o {
                acc = dyr[k].mul_add(w[k * i + c], acc);
            }
            dx[r * i + c] = acc;
        }
    }
    dx
}

/// `dw[r,c] = Σ_k dy[k·o+r]·x[k·i+c]`, `acc = 0.0`, `k` ascending, `f32::mul_add`.
fn cpu_bwd_weight(dy: &[f32], x: &[f32], n: usize, i: usize, o: usize) -> Vec<f32> {
    let mut dw = vec![0.0f32; o * i];
    for r in 0..o {
        for c in 0..i {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc = dy[k * o + r].mul_add(x[k * i + c], acc);
            }
            dw[r * i + c] = acc;
        }
    }
    dw
}

/// Record `step` and copy its `count` f32 words back.
///
/// # Safety
/// Vulkan submission over caller-owned handles; `out` must hold `count` f32.
unsafe fn run(vk: &Vk, step: &Step, out: Handle, count: usize) -> Vec<f32> {
    let (rb, m) = vk
        .buffer(
            count as u64 * 4,
            BUFFER_USAGE_TRANSFER_DST,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )
        .expect("readback buffer");
    let cb = vk.command_buffer();
    vk.begin(cb);
    rec_step(vk, cb, step);
    vk.barrier(
        cb,
        STAGE_COMPUTE,
        STAGE_TRANSFER,
        ACCESS_SHADER_WRITE,
        ACCESS_TRANSFER_READ,
    );
    let region = BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: (count * 4) as u64,
    };
    (vk.copy_buffer)(cb, out, rb, 1, &region);
    vk.barrier(
        cb,
        STAGE_TRANSFER,
        STAGE_HOST,
        ACCESS_TRANSFER_WRITE,
        ACCESS_HOST_READ,
    );
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
    let p = vk.map(m).cast::<f32>();
    std::slice::from_raw_parts(p, count).to_vec()
}

/// First differing index and both words, or `None` when the two vectors are bit-identical.
fn first_word_diff(got: &[f32], want: &[f32]) -> Option<(usize, u32, u32)> {
    for (k, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        if g.to_bits() != w.to_bits() {
            return Some((k, g.to_bits(), w.to_bits()));
        }
    }
    None
}

fn assert_words(label: &str, got: &[f32], want: &[f32]) {
    match first_word_diff(got, want) {
        None => println!("{label}: {} words identical", want.len()),
        Some((k, g, w)) => panic!(
            "{label}: first differing word at {k}: device {g:#010x} ({}) vs mul_add reference \
             {w:#010x} ({})",
            f32::from_bits(g),
            f32::from_bits(w)
        ),
    }
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

#[test]
fn linear_gemm_words_identical() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    println!("device: {}", gpu_identity());
    let shapes = [
        ("1x1x1", 1usize, 1usize, 1usize),
        ("17x33x65", 17, 33, 65),
        ("100x129x47", 100, 129, 47),
        ("64x64x64", 64, 64, 64),
        ("prod-slice-256x1024x3584", 256, 1024, 3584),
    ];
    for &(label, n, i, o) in &shapes {
        let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
        let x = fill(n * i, 0x9e37_79b9_7f4a_7c15);
        let w = fill(o * i, 0x0123_4567_89ab_cdef);
        let dy = fill(n * o, 0xfeed_face_dead_beef);

        // forward: y = x·Wᵀ
        let x_b = unsafe { g.up_f32(&x) };
        let w_b = unsafe { g.up_f32(&w) };
        let y_b = unsafe { g.dev(n * o * 4) };
        let step = unsafe { linear_fwd_dispatch(&g, x_b, w_b, y_b, n as u32, i as u32, o as u32) };
        let got = unsafe { run(&vk, &step, y_b, n * o) };
        assert_words(&format!("fwd {label}"), &got, &cpu_fwd(&x, &w, n, i, o));

        // backward input adjoint: dx = dy·W
        let dy_b = unsafe { g.up_f32(&dy) };
        let dx_b = unsafe { g.dev(n * i * 4) };
        let step =
            unsafe { linear_bwd_input_dispatch(&g, dy_b, w_b, dx_b, n as u32, i as u32, o as u32) };
        let got = unsafe { run(&vk, &step, dx_b, n * i) };
        assert_words(
            &format!("bwd-in {label}"),
            &got,
            &cpu_bwd_input(&dy, &w, n, i, o),
        );

        // backward weight adjoint: dW = dyᵀ·x
        let dw_b = unsafe { g.dev(o * i * 4) };
        let step = unsafe {
            linear_bwd_weight_dispatch(&g, dy_b, x_b, dw_b, n as u32, i as u32, o as u32)
        };
        let got = unsafe { run(&vk, &step, dw_b, o * i) };
        assert_words(
            &format!("bwd-w {label}"),
            &got,
            &cpu_bwd_weight(&dy, &x, n, i, o),
        );
    }
}

/// The nine production cells: the three training-tape GEMMs at the three tape shapes.
const CELLS: [(&str, usize, usize, usize); 9] = [
    ("fwd", 2048, 1024, 3584),
    ("fwd", 2048, 3584, 1024),
    ("fwd", 2048, 2048, 1024),
    ("bwd-in", 2048, 1024, 3584),
    ("bwd-in", 2048, 3584, 1024),
    ("bwd-in", 2048, 2048, 1024),
    ("bwd-w", 2048, 1024, 3584),
    ("bwd-w", 2048, 3584, 1024),
    ("bwd-w", 2048, 2048, 1024),
];

#[test]
fn linear_gemm_timings() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    println!(
        "machine: device+driver [{}]; 5 runs after 3 warmup; device time = two \
         vkCmdWriteTimestamp2 around the dispatch; timestamp_period {} ns",
        gpu_identity(),
        vk.timestamp_period
    );
    let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let pool = unsafe { vk.timestamp_pool(2) };
    // The largest cell needs every operand buffer of the three ops: x (2048·3584), w (3584²),
    // dy (2048·3584), y/dx (2048·3584), dw (3584²).
    let x = fill(2048 * 3584, 0x9e37_79b9_7f4a_7c15);
    let w = fill(3584 * 3584, 0x0123_4567_89ab_cdef);
    let dy = fill(2048 * 3584, 0xfeed_face_dead_beef);
    let x_b = unsafe { g.up_f32(&x) };
    let w_b = unsafe { g.up_f32(&w) };
    let dy_b = unsafe { g.up_f32(&dy) };
    let y_b = unsafe { g.dev(2048 * 3584 * 4) };
    let dw_b = unsafe { g.dev(3584 * 3584 * 4) };
    for &(op, n, i, o) in &CELLS {
        let (nu, iu, ou) = (n as u32, i as u32, o as u32);
        let step = unsafe {
            if op == "fwd" {
                linear_fwd_dispatch(&g, x_b, w_b, y_b, nu, iu, ou)
            } else if op == "bwd-in" {
                linear_bwd_input_dispatch(&g, dy_b, w_b, y_b, nu, iu, ou)
            } else {
                linear_bwd_weight_dispatch(&g, dy_b, x_b, dw_b, nu, iu, ou)
            }
        };
        for _ in 0..3 {
            unsafe { timed(&vk, &step, &pool) };
        }
        let mut t = Vec::with_capacity(5);
        for _ in 0..5 {
            t.push(unsafe { timed(&vk, &step, &pool) });
        }
        let (mean, sd) = mean_spread(&t);
        let flops = 2.0 * (n as f64) * (i as f64) * (o as f64);
        println!(
            "TIMING {op} ({n},{i},{o}): device mean {mean:.1} us sd {sd:.1} us (n {}); \
             {:.1} GFLOP/s",
            t.len(),
            flops / (mean * 1e-6) / 1e9
        );
        assert!(
            mean > 0.0 && flops / (mean * 1e-6) / 1e9 > 1000.0,
            "{op} ({n},{i},{o}): device time {mean} us gives under 1 TFLOP/s — the timed window \
             did not bracket the dispatch"
        );
    }
}

/// Device time in µs of one dispatch: two `vkCmdWriteTimestamp2` around the step.
///
/// # Safety
/// Vulkan submission over caller-owned handles.
unsafe fn timed(vk: &Vk, step: &Step, pool: &QueryPool) -> f64 {
    let cb = vk.command_buffer();
    vk.begin(cb);
    vk.cmd_reset_timestamps(cb, pool);
    vk.cmd_write_timestamp(cb, pool, STAGE_ALL_COMMANDS, 0);
    rec_step(vk, cb, step);
    vk.cmd_write_timestamp(cb, pool, STAGE_ALL_COMMANDS, 1);
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
    let ts = vk.read_timestamps(pool);
    let ticks = ts[1].wrapping_sub(ts[0]) as f64;
    ticks * f64::from(vk.timestamp_period) / 1000.0
}

fn mean_spread(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}
