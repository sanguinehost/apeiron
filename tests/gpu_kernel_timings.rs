//! Timed dispatch campaign for the GPU training backward rows: `P83`, `P85`, `P86`, `P87`, `P88`,
//! `P89`, `P90`, `P91` in the perf-predictions note.
//!
//! Every fixture is the one its parity test builds (`tests/gpu_norm_backward.rs`,
//! `tests/gpu_linear_bwd.rs`, `tests/gpu_attn_bwd.rs`, `tests/gpu_adam.rs`): the same fixed-seed
//! LCG, the same amplitudes, the same shapes, and for attention the same oracle forward
//! `attn_fwd_kern` so the uploaded `p` is the parity fixture's. Steps are built once, outside the
//! timed window (`make_step_named` compiles SPIR-V and creates a pipeline); each run then allocates
//! a fresh command buffer, records the one step, and times `vkBeginCommandBuffer` through
//! `vk.submit_wait` (`vkQueueSubmit` + `vkQueueWaitIdle`) with `std::time::Instant`. A run is
//! therefore the dispatch's device execution plus its host submission cost, not its SPIR-V build.
//!
//! Per cell: 3 warmup runs (not reported) then 7 repeats, every repeat printed, then mean, min and
//! max. Every output is read back and asserted finite. The machine line names the device, the
//! driver, the repeat count and the GPU lock the run holds.
//!
//! The wall clock around `vkQueueWaitIdle` carries the driver's submission cost, which is a fixed
//! charge per submission and not the kernel. Every run therefore also brackets the dispatch with
//! two `vkCmdWriteTimestamp2` queries in the same command buffer and reports the device-time
//! difference as `device`, and an empty command buffer is measured as the `submit_floor` control.
//! The registered bands in the perf-predictions note are device time (traffic / 960 GB/s),
//! so the band verdict uses `device`; `wall` is reported beside it, as the brief names that
//! instrument.
//!
//! Run with the GPU lock so no parity test of another lane overlaps a timed window:
//!
//! ```text
//! flock ~/gpu.lock -c \
//!   'cargo test --release --features gpu --test gpu_kernel_timings -- --nocapture --test-threads=1'
//! ```
//!
//! Board row: `apeiron-gpu-train-w1-t05-GVrduzS5mnRVc4oeQ1xRD7`.
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_norm_backward/gpu_linear_bwd tests"
)]

use apeiron::gpu::decode::{rec_step, GpuCtx, Step};
use apeiron::gpu::nn::{
    adam_update_dispatch, attn_bwd_dkdv_dispatch, attn_bwd_dq_dispatch, attn_bwd_ds_dispatch,
    attn_bwd_rowsum_dispatch, linear_bwd_input_dispatch, linear_bwd_weight_dispatch,
    rmsnorm_bwd_dispatch, rmsnorm_dgamma_dispatch,
};
use apeiron::gpu::vulkan::*;
use apeiron::learn::grad::attn::attn_fwd_kern;
use std::time::Instant;

/// Qwen2.5-0.5B hidden width: the norm-backward fixture width.
const W: usize = 896;
/// Reported repeats per cell.
const REPS: usize = 7;
/// Warmup runs per cell, not reported.
const WARM: usize = 3;
/// `round(1e-3·2^32)`: the oracle's `lr` word, as in `tests/gpu_adam.rs`.
const LR_Q32: i64 = 4_294_967;

/// Fixed-seed LCG (Knuth's MMIX constants), uniform in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        u * 2.0 - 1.0
    }
}

fn real_to_q32(v: f64) -> i64 {
    (v * 4294967296.0).round() as i64
}

fn to_f32(v: i64) -> f32 {
    (v as f64 / 4294967296.0) as f32
}

/// One cell: every reported run, in microseconds, plus its label and shape. `wall` brackets the
/// submission (`vkBeginCommandBuffer` → `vkQueueWaitIdle`); `dev` is the two-timestamp device
/// interval around the one dispatch, read from the same command buffer.
struct Series {
    cell: &'static str,
    shape: String,
    wall: Vec<f64>,
    dev: Vec<f64>,
}

impl Series {
    fn new(cell: &'static str, shape: String) -> Self {
        Self {
            cell,
            shape,
            wall: Vec::with_capacity(REPS),
            dev: Vec::with_capacity(REPS),
        }
    }

    fn record(&mut self, wall: f64, dev: f64) {
        self.wall.push(wall);
        self.dev.push(dev);
        println!(
            "cell {} {} run {}: wall {wall:.3} us device {dev:.3} us",
            self.cell,
            self.shape,
            self.wall.len()
        );
    }

    fn summary(&self) {
        let (wm, wmin, wmax) = stats(&self.wall);
        let (dm, dmin, dmax) = stats(&self.dev);
        println!(
            "SUMMARY {} {} wall mean {wm:.3} us min {wmin:.3} max {wmax:.3}; device mean {dm:.3} us min {dmin:.3} max {dmax:.3} n {}",
            self.cell,
            self.shape,
            self.dev.len()
        );
    }
}

fn stats(v: &[f64]) -> (f64, f64, f64) {
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n as f64;
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mean, min, max)
}

/// Record `steps` (in order, `rec_step`'s compute→compute barrier after each) into a fresh command
/// buffer between two GPU timestamps and time begin → `submit_wait` on the host wall clock.
/// Returns `(wall us, device us)`. An empty `steps` is the submission-floor control.
///
/// # Safety
/// Vulkan submission over caller-owned handles; the steps' buffers must outlive the call.
unsafe fn timed(vk: &Vk, steps: &[&Step], pool: &QueryPool) -> (f64, f64) {
    let cb = vk.command_buffer();
    let t0 = Instant::now();
    vk.begin(cb);
    vk.cmd_reset_timestamps(cb, pool);
    vk.cmd_write_timestamp(cb, pool, STAGE_ALL_COMMANDS, 0);
    for step in steps {
        rec_step(vk, cb, step);
    }
    vk.cmd_write_timestamp(cb, pool, STAGE_ALL_COMMANDS, 1);
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
    let wall = t0.elapsed().as_secs_f64() * 1e6;
    let ts = vk.read_timestamps(pool);
    let ticks = ts[1].wrapping_sub(ts[0]) as f64;
    (wall, ticks * f64::from(vk.timestamp_period) / 1000.0)
}

/// 3 warmup runs, then 7 reported repeats of one step (or one multi-dispatch chain).
fn run_series(
    vk: &Vk,
    pool: &QueryPool,
    steps: &[&Step],
    cell: &'static str,
    shape: String,
) -> Series {
    for _ in 0..WARM {
        unsafe { timed(vk, steps, pool) };
    }
    let mut s = Series::new(cell, shape);
    for _ in 0..REPS {
        let (wall, dev) = unsafe { timed(vk, steps, pool) };
        s.record(wall, dev);
    }
    s.summary();
    s
}

/// Copy a device buffer back and return its f32 words.
///
/// # Safety
/// `src` must hold at least `count` f32; the readback buffer is allocated here.
unsafe fn read_f32(vk: &Vk, src: Handle, count: usize) -> Vec<f32> {
    let (rb, m) = vk
        .buffer(
            count as u64 * 4,
            BUFFER_USAGE_TRANSFER_DST,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )
        .expect("readback buffer");
    let p = vk.map(m).cast::<f32>();
    let cb = vk.command_buffer();
    vk.begin(cb);
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
    (vk.copy_buffer)(cb, src, rb, 1, &region);
    vk.barrier(
        cb,
        STAGE_TRANSFER,
        STAGE_HOST,
        ACCESS_TRANSFER_WRITE,
        ACCESS_HOST_READ,
    );
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
    std::slice::from_raw_parts(p, count).to_vec()
}

fn assert_finite(cell: &str, shape: &str, what: &str, v: &[f32]) {
    assert!(
        v.iter().all(|x| x.is_finite()),
        "{cell} {shape}: non-finite {what} ({}/{} entries)",
        v.iter().filter(|x| !x.is_finite()).count(),
        v.len()
    );
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
fn gpu_training_dispatch_timings() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    // `nvidia-smi --query-gpu=name,driver_version` prints "name, driver" in that order.
    println!(
        "machine: device+driver [{}]; repeats {REPS} after {WARM} warmup; timestamp_period {} ns \
         ({}-bit); timed window = vkBeginCommandBuffer..vkQueueWaitIdle, device time = two \
         vkCmdWriteTimestamp2 around the dispatch; gpu lock ~/gpu.lock",
        gpu_identity(),
        vk.timestamp_period,
        vk.timestamp_valid_bits
    );
    assert!(
        vk.timestamp_valid_bits >= 32 && vk.timestamp_period > 0.0,
        "device reports no usable timestamps ({} bits, period {})",
        vk.timestamp_valid_bits,
        vk.timestamp_period
    );
    let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let pool = unsafe { vk.timestamp_pool(2) };

    // Control: an empty command buffer through the same window is the submission floor. A cell at
    // this number is instrument-limited, not kernel-limited.
    run_series(
        &vk,
        &pool,
        &[],
        "submit_floor",
        "empty command buffer".to_string(),
    );

    norm_bwd(&vk, &g, &pool);
    linear_bwd(&vk, &g, &pool);
    attn_bwd(&vk, &g, &pool);
    adam(&vk, &g, &pool);
    println!("done: 25 dispatch cells + 4 four-pass cells + 1 control, {REPS} reported runs each");
}

/// (a) `rmsnorm_bwd` main dispatch and the `dγ` reduce at `t = 512`, `w = 896` (P83, P85).
fn norm_bwd(vk: &Vk, g: &GpuCtx, pool: &QueryPool) {
    let t = 512;
    // `tests/gpu_norm_backward.rs` fixture "t=512": |x|, |dy| ≤ 2^20 Q32.32 units, γ in [0.5, 1.5).
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let xa = (1u64 << 20) as f64 / 4294967296.0;
    let x: Vec<i64> = (0..t * W).map(|_| real_to_q32(xa * rng.unit())).collect();
    let dy: Vec<i64> = (0..t * W).map(|_| real_to_q32(xa * rng.unit())).collect();
    let gamma: Vec<i64> = (0..W)
        .map(|_| real_to_q32(0.75 + 0.25 * rng.unit()))
        .collect();
    let xf: Vec<f32> = x.iter().map(|&v| to_f32(v)).collect();
    let yf: Vec<f32> = dy.iter().map(|&v| to_f32(v)).collect();
    let gf: Vec<f32> = gamma.iter().map(|&v| to_f32(v)).collect();

    let (x_buf, g_buf, y_buf, dx_buf, dgr_buf, dg_buf) = unsafe {
        (
            g.up_f32(&xf),
            g.up_f32(&gf),
            g.up_f32(&yf),
            g.dev(t * W * 4),
            g.dev(t * W * 4),
            g.dev(W * 4),
        )
    };
    let main_step = unsafe {
        rmsnorm_bwd_dispatch(
            g, x_buf, g_buf, y_buf, dx_buf, dgr_buf, t as u32, W as u32, 1e-6f32,
        )
    };
    let dg_step = unsafe { rmsnorm_dgamma_dispatch(g, dgr_buf, dg_buf, t as u32, W as u32) };

    let shape = format!("t={t} w={W}");
    run_series(vk, pool, &[&main_step], "norm_bwd_main", shape.clone());
    let dx = unsafe { read_f32(vk, dx_buf, t * W) };
    assert_finite("norm_bwd_main", &shape, "dx", &dx);

    run_series(vk, pool, &[&dg_step], "norm_dgamma_reduce", shape.clone());
    let dg = unsafe { read_f32(vk, dg_buf, W) };
    assert_finite("norm_dgamma_reduce", &shape, "dgamma", &dg);
}

/// (b) `linear_bwd_input` and the tiled (`P89`) `linear_bwd_weight` on the parity shapes
/// (P86, P87, P89).
fn linear_bwd(vk: &Vk, g: &GpuCtx, pool: &QueryPool) {
    let shapes = [(1usize, 32usize, 64usize), (7, 64, 128), (512, 256, 256)];
    for (n, i, o) in shapes {
        // `tests/gpu_linear_bwd.rs`: LCG in [-1, 1) real, Q32.32 ±2^32.
        let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
        let x: Vec<i64> = (0..n * i).map(|_| real_to_q32(rng.unit())).collect();
        let dy: Vec<i64> = (0..n * o).map(|_| real_to_q32(rng.unit())).collect();
        let w: Vec<i64> = (0..o * i).map(|_| real_to_q32(rng.unit())).collect();
        let xf: Vec<f32> = x.iter().map(|&v| to_f32(v)).collect();
        let yf: Vec<f32> = dy.iter().map(|&v| to_f32(v)).collect();
        let wf: Vec<f32> = w.iter().map(|&v| to_f32(v)).collect();

        let (y_buf, w_buf, x_buf, dx_buf, dw_buf) = unsafe {
            (
                g.up_f32(&yf),
                g.up_f32(&wf),
                g.up_f32(&xf),
                g.dev(n * i * 4),
                g.dev(o * i * 4),
            )
        };
        let dx_step = unsafe {
            linear_bwd_input_dispatch(g, y_buf, w_buf, dx_buf, n as u32, i as u32, o as u32)
        };
        let dw_step = unsafe {
            linear_bwd_weight_dispatch(g, y_buf, x_buf, dw_buf, n as u32, i as u32, o as u32)
        };

        let shape = format!("n={n} i={i} o={o}");
        run_series(vk, pool, &[&dx_step], "linear_bwd_input", shape.clone());
        let dx = unsafe { read_f32(vk, dx_buf, n * i) };
        assert_finite("linear_bwd_input", &shape, "dx", &dx);

        run_series(vk, pool, &[&dw_step], "linear_bwd_weight", shape.clone());
        let dw = unsafe { read_f32(vk, dw_buf, o * i) };
        assert_finite("linear_bwd_weight", &shape, "dw", &dw);
    }
}

/// (c) the four attention-backward passes on the parity shapes (P88, P90).
fn attn_bwd(vk: &Vk, g: &GpuCtx, pool: &QueryPool) {
    let shapes = [
        (1usize, 2usize, 1usize, 8usize),
        (7, 2, 1, 16),
        (16, 4, 2, 16),
        (64, 4, 2, 32),
        // Two 32-column d-tiles and one full tile plus a 16-column partial tile (P97).
        (8, 2, 1, 64),
        (5, 2, 1, 48),
    ];
    for (t, nh, nkv, hd) in shapes {
        // `tests/gpu_attn_bwd.rs`: LCG in [-0.5, 0.5) real for q, k, v, dout; `p` from the oracle
        // forward so the uploaded probabilities are the parity fixture's.
        let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
        let q: Vec<i64> = (0..t * nh * hd)
            .map(|_| real_to_q32(rng.unit() * 0.5))
            .collect();
        let k: Vec<i64> = (0..t * nkv * hd)
            .map(|_| real_to_q32(rng.unit() * 0.5))
            .collect();
        let v: Vec<i64> = (0..t * nkv * hd)
            .map(|_| real_to_q32(rng.unit() * 0.5))
            .collect();
        let dout: Vec<i64> = (0..t * nh * hd)
            .map(|_| real_to_q32(rng.unit() * 0.5))
            .collect();
        let (_out, p) = attn_fwd_kern(&q, &k, &v, nh, nkv, hd).expect("cpu attn_fwd_kern");
        let qf: Vec<f32> = q.iter().map(|&x| to_f32(x)).collect();
        let kf: Vec<f32> = k.iter().map(|&x| to_f32(x)).collect();
        let vf: Vec<f32> = v.iter().map(|&x| to_f32(x)).collect();
        let pf: Vec<f32> = p.iter().map(|&x| to_f32(x)).collect();
        let dof: Vec<f32> = dout.iter().map(|&x| to_f32(x)).collect();

        let (q_buf, k_buf, v_buf, p_buf, do_buf, rs_buf, ds_buf, dq_buf, dk_buf, dv_buf) = unsafe {
            (
                g.up_f32(&qf),
                g.up_f32(&kf),
                g.up_f32(&vf),
                g.up_f32(&pf),
                g.up_f32(&dof),
                g.dev(t * nh * 4),
                g.dev(nh * t * t * 4),
                g.dev(t * nh * hd * 4),
                g.dev(t * nkv * hd * 4),
                g.dev(t * nkv * hd * 4),
            )
        };
        let (tu, nhu, nkvu, hdu) = (t as u32, nh as u32, nkv as u32, hd as u32);
        let rs_step = unsafe {
            attn_bwd_rowsum_dispatch(g, v_buf, p_buf, do_buf, rs_buf, tu, nhu, nkvu, hdu)
        };
        let ds_step = unsafe {
            attn_bwd_ds_dispatch(g, v_buf, p_buf, do_buf, rs_buf, ds_buf, tu, nhu, nkvu, hdu)
        };
        let dq_step = unsafe { attn_bwd_dq_dispatch(g, k_buf, ds_buf, dq_buf, tu, nhu, nkvu, hdu) };
        let dkdv_step = unsafe {
            attn_bwd_dkdv_dispatch(
                g, q_buf, p_buf, ds_buf, do_buf, dk_buf, dv_buf, tu, nhu, nkvu, hdu,
            )
        };

        let shape = format!("t={t} nh={nh} nkv={nkv} hd={hd}");
        run_series(vk, pool, &[&rs_step], "attn_rowsum", shape.clone());
        let rs = unsafe { read_f32(vk, rs_buf, t * nh) };
        assert_finite("attn_rowsum", &shape, "rowsum", &rs);

        run_series(vk, pool, &[&ds_step], "attn_ds", shape.clone());
        let ds = unsafe { read_f32(vk, ds_buf, nh * t * t) };
        assert_finite("attn_ds", &shape, "ds", &ds);

        run_series(vk, pool, &[&dq_step], "attn_dq", shape.clone());
        let dq = unsafe { read_f32(vk, dq_buf, t * nh * hd) };
        assert_finite("attn_dq", &shape, "dq", &dq);

        run_series(vk, pool, &[&dkdv_step], "attn_dkdv", shape.clone());
        let dk = unsafe { read_f32(vk, dk_buf, t * nkv * hd) };
        assert_finite("attn_dkdv", &shape, "dk", &dk);
        let dv = unsafe { read_f32(vk, dv_buf, t * nkv * hd) };
        assert_finite("attn_dkdv", &shape, "dv", &dv);

        // P90's registered cell: the four passes in one command buffer, rowsum → ds → dq → dk/dv,
        // with `rec_step`'s compute→compute barrier after each, exactly as `tests/gpu_attn_bwd.rs`.
        run_series(
            vk,
            pool,
            &[&rs_step, &ds_step, &dq_step, &dkdv_step],
            "attn_four_pass",
            shape.clone(),
        );
        let dq2 = unsafe { read_f32(vk, dq_buf, t * nh * hd) };
        assert_finite("attn_four_pass", &shape, "dq", &dq2);
        let dv2 = unsafe { read_f32(vk, dv_buf, t * nkv * hd) };
        assert_finite("attn_four_pass", &shape, "dv", &dv2);
    }
}

/// (d) `adam_update` at `n = 896` (P91), the parity fixture's words and first step.
fn adam(vk: &Vk, g: &GpuCtx, pool: &QueryPool) {
    let n = 896;
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let p0: Vec<f32> = (0..n)
        .map(|_| to_f32(real_to_q32(0.5 * rng.unit())))
        .collect();
    let grad: Vec<f32> = (0..n)
        .map(|_| to_f32(real_to_q32(0.5 * rng.unit())))
        .collect();
    let zeros = vec![0.0f32; n];
    let (p_buf, g_buf, m_buf, v_buf) = unsafe {
        (
            g.up_f32(&p0),
            g.up_f32(&grad),
            g.up_f32(&zeros),
            g.up_f32(&zeros),
        )
    };
    let step = unsafe { adam_update_dispatch(g, p_buf, g_buf, m_buf, v_buf, n as u32, LR_Q32, 1) };
    let shape = format!("n={n}");
    run_series(vk, pool, &[&step], "adam_update", shape.clone());
    let p = unsafe { read_f32(vk, p_buf, n) };
    assert_finite("adam_update", &shape, "p", &p);
}
