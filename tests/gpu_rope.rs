//! GPU RoPE forward and adjoint (`gpu::nn::rope_fwd_module`, `gpu::nn::rope_bwd_module`) against
//! the Q32.32 CPU oracle `learn::grad::attn::{rope_fwd, rope_bwd}`.
//!
//! The phase tables are host inputs: with `f32` device trigonometry they cannot reproduce the
//! engine's exact integer round `(turns[j]·pos) mod 2^64` in `Z/2^64`. This test builds `cos`/`sin`
//! per `(pos, j)` with the same public engine functions the oracle reaches
//! (`RopeTable::turns`, `turn_to_radians`, `fixed_sin_cos`) and converts each Q32.32 word to f32
//! exactly once, `value / 2^32` rounded to nearest f32. The `RopeTable` itself is built through
//! `RopeTable::from_turns` with the same `round(theta^(-2j/head_dim) / (2π) · 2^64)` formula
//! `RopeTable::new` uses, because `engine::num::fp::F64` is crate-private and `new` is not
//! reachable from an integration test; the turn words are therefore identical to `new`'s.
//!
//! Shapes `(t, nh, hd)`: `(1, 2, 8)`, `(7, 2, 16)`, `(16, 4, 64)`; position = row index. Inputs
//! are a fixed-seed LCG in `[-0.5, 0.5)` real (Q32.32 units ±2³¹). Device f32 outputs convert back
//! to Q32.32 by round-to-nearest (saturating), and the per-entry difference is bounded per product
//! and shape at about twice the observed maximum.
//!
//! ```text
//! cargo test --release --features gpu --test gpu_rope -- --nocapture
//! ```
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_linear_bwd/gpu_head tests"
)]

use apeiron::engine::RopeTable;
use apeiron::gpu::decode::{rec_step, GpuCtx};
use apeiron::gpu::nn::fwd::{rope_bwd_dispatch, rope_fwd_dispatch};
use apeiron::gpu::vulkan::*;
use apeiron::learn::grad::attn::{rope_bwd, rope_fwd};
use apeiron::numeric::fixed_trig::{fixed_sin_cos, turn_to_radians};

/// Fixed-seed LCG (Knuth's MMIX constants), uniform in `[-0.5, 0.5)`.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        u - 0.5
    }
}

fn real_to_q32(v: f64) -> i64 {
    (v * 4294967296.0).round() as i64
}

fn to_f32(v: i64) -> f32 {
    (v as f64 / 4294967296.0) as f32
}

/// f32 → Q32.32, round-to-nearest and saturating.
fn to_q32(v: f32) -> i64 {
    let scaled = f64::from(v) * 4294967296.0;
    if scaled >= i64::MAX as f64 {
        i64::MAX
    } else if scaled <= i64::MIN as f64 {
        i64::MIN
    } else {
        scaled.round() as i64
    }
}

/// `RopeTable::new(F64::from_f64(10000.0), hd)` by its documented turn formula, since `F64` is
/// crate-private.
fn rope_table(hd: usize) -> RopeTable {
    let theta = 10_000.0_f64;
    let unit = 18_446_744_073_709_551_616.0_f64; // 2^64 phases per full turn
    let two_pi = 2.0 * std::f64::consts::PI;
    let turns: Vec<u64> = (0..hd / 2)
        .map(|j| {
            let q = (theta.powf(-2.0 * (j as f64) / (hd as f64)) * unit / two_pi).round();
            assert!(
                q.is_finite() && (0.0..unit).contains(&q),
                "turn out of range"
            );
            q as u64
        })
        .collect();
    RopeTable::from_turns(hd, turns).expect("turn table")
}

/// `cos`/`sin` f32 tables per `(pos, j)`, from the oracle's own integer phase and trig.
fn trig_tables(table: &RopeTable, t: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let d2 = hd / 2;
    let mut cos = vec![0f32; t * d2];
    let mut sin = vec![0f32; t * d2];
    for pos in 0..t {
        for j in 0..d2 {
            let phase = ((u128::from(table.turns()[j]) * u128::from(pos as u32))
                & u128::from(u64::MAX)) as u64;
            let sc = fixed_sin_cos(turn_to_radians(phase)).expect("fixed_sin_cos");
            let off = pos * d2 + j;
            cos[off] = (sc.cos as f64 / 4294967296.0) as f32;
            sin[off] = (sc.sin as f64 / 4294967296.0) as f32;
        }
    }
    (cos, sin)
}

/// One shape and its pinned per-entry bounds in Q32.32 units (about twice the observed maximum,
/// stated per product).
struct Fixture {
    label: &'static str,
    t: usize,
    nh: usize,
    hd: usize,
    bound_fwd: f64,
    bound_bwd: f64,
}

#[test]
fn rope_matches_cpu() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    println!("device: {}", gpu_identity());
    let fixtures = [
        Fixture {
            label: "t=1 nh=2 hd=8",
            t: 1,
            nh: 2,
            hd: 8,
            bound_fwd: 130.0,
            bound_bwd: 140.0,
        },
        Fixture {
            label: "t=7 nh=2 hd=16",
            t: 7,
            nh: 2,
            hd: 16,
            bound_fwd: 520.0,
            bound_bwd: 330.0,
        },
        Fixture {
            label: "t=16 nh=4 hd=64",
            t: 16,
            nh: 4,
            hd: 64,
            bound_fwd: 520.0,
            bound_bwd: 540.0,
        },
    ];
    for f in &fixtures {
        run_fixture(&vk, f);
    }
}

fn run_fixture(vk: &Vk, f: &Fixture) {
    let (t, nh, hd) = (f.t, f.nh, f.hd);
    let n = t * nh * hd;
    let table = rope_table(hd);
    let (cos, sin) = trig_tables(&table, t, hd);
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let x: Vec<i64> = (0..n).map(|_| real_to_q32(rng.unit())).collect();
    let dy: Vec<i64> = (0..n).map(|_| real_to_q32(rng.unit())).collect();

    // CPU oracle, one token vector at a time (position = row index).
    let mut cpu_fwd = Vec::with_capacity(n);
    let mut cpu_bwd = Vec::with_capacity(n);
    for pos in 0..t {
        let row = &x[pos * nh * hd..(pos + 1) * nh * hd];
        cpu_fwd.extend(rope_fwd(row, pos as u32, &table).expect("rope_fwd"));
        let drow = &dy[pos * nh * hd..(pos + 1) * nh * hd];
        cpu_bwd.extend(rope_bwd(drow, pos as u32, &table).expect("rope_bwd"));
    }
    assert!(cpu_fwd.iter().all(|&v| v != 0), "oracle produced all zeros");

    let xf: Vec<f32> = x.iter().map(|&v| to_f32(v)).collect();
    let dyf: Vec<f32> = dy.iter().map(|&v| to_f32(v)).collect();
    let (g, _stage) = GpuCtx::new(vk).expect("GpuCtx::new");
    let x_buf = unsafe { g.up_f32(&xf) };
    let dy_buf = unsafe { g.up_f32(&dyf) };
    let cos_buf = unsafe { g.up_f32(&cos) };
    let sin_buf = unsafe { g.up_f32(&sin) };
    let out_buf = unsafe { g.dev(n * 4) };
    let dx_buf = unsafe { g.dev(n * 4) };

    let (t32, nh32, hd32) = (t as u32, nh as u32, hd as u32);
    let step_fwd =
        unsafe { rope_fwd_dispatch(&g, x_buf, cos_buf, sin_buf, out_buf, t32, nh32, hd32) };
    let step_bwd =
        unsafe { rope_bwd_dispatch(&g, dy_buf, cos_buf, sin_buf, dx_buf, t32, nh32, hd32) };

    let host = |count: usize| unsafe {
        let (b, m) = vk
            .buffer(
                count as u64 * 4,
                BUFFER_USAGE_TRANSFER_DST,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
            .expect("readback buffer");
        (b, vk.map(m).cast::<f32>())
    };
    let (rb_out, rb_out_p) = host(n);
    let (rb_dx, rb_dx_p) = host(n);
    let cb = unsafe { vk.command_buffer() };
    unsafe {
        vk.begin(cb);
        rec_step(vk, cb, &step_fwd);
        vk.barrier(
            cb,
            STAGE_COMPUTE,
            STAGE_COMPUTE,
            ACCESS_SHADER_WRITE,
            ACCESS_SHADER_READ,
        );
        rec_step(vk, cb, &step_bwd);
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
            size: (n * 4) as u64,
        };
        (vk.copy_buffer)(cb, out_buf, rb_out, 1, &region);
        (vk.copy_buffer)(cb, dx_buf, rb_dx, 1, &region);
        vk.barrier(
            cb,
            STAGE_TRANSFER,
            STAGE_HOST,
            ACCESS_TRANSFER_WRITE,
            ACCESS_HOST_READ,
        );
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        vk.submit_wait(cb);
    }
    let gpu_out: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_out_p, n).to_vec() };
    let gpu_dx: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_dx_p, n).to_vec() };
    assert!(gpu_out.iter().all(|v| v.is_finite()), "non-finite out");
    assert!(gpu_dx.iter().all(|v| v.is_finite()), "non-finite dx");

    let (max_fwd, at_fwd) = max_err(&gpu_out, &cpu_fwd, hd);
    let (max_bwd, at_bwd) = max_err(&gpu_dx, &cpu_bwd, hd);
    println!(
        "{}: fwd max |gpu-cpu| = {max_fwd} units at {at_fwd:?}; bwd max = {max_bwd} units at \
         {at_bwd:?}; pinned bounds fwd {} / bwd {}",
        f.label, f.bound_fwd, f.bound_bwd
    );
    assert!(
        max_fwd as f64 <= f.bound_fwd,
        "{}: fwd error {max_fwd} units at {at_fwd:?} exceeds the pinned bound {}",
        f.label,
        f.bound_fwd
    );
    assert!(
        max_bwd as f64 <= f.bound_bwd,
        "{}: bwd error {max_bwd} units at {at_bwd:?} exceeds the pinned bound {}",
        f.label,
        f.bound_bwd
    );
}

fn max_err(gpu: &[f32], cpu: &[i64], cols: usize) -> (i128, (usize, usize)) {
    let mut max = 0i128;
    let mut at = (0usize, 0usize);
    for (idx, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let d = (i128::from(to_q32(g)) - i128::from(c)).abs();
        if d > max {
            max = d;
            at = (idx / cols, idx % cols);
        }
    }
    (max, at)
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
