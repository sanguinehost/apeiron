#![cfg(feature = "gpu")]
//! **DP4A correctness sweep, before any ratio claim.**
//!
//! Every shape below is built synthetically (no GGUF, no model file), packed into the chart the
//! served `gemv_actq` kernel reads, dispatched through `make_actq_gemv` (`ActQ::Copy`,
//! `ScaleFace::Dm32`), and compared lane by lane against an exact `i128` Q32.32 reference:
//!
//! ```text
//! w = d_b·q + lo_b,  x = c_j·2⁻⁷,  S = 2⁻⁷
//! y_q32 = round_shift_32( Σ_b (d_q32·Σ_j q·c + lo_q32·Σ_j c) << 25 )
//! ```
//!
//! with `d_q32 = d·2³²`, `lo_q32 = lo·2³²` (both exact: `d`, `lo` are f16). The reference folds
//! the whole row once; the GPU folds per 32-block in f32 and reduces split-K partials in f32, so
//! the reported max abs error is the device's f32 epilogue/accumulation error against the
//! single-rounded exact value. The activation grid is built so the device quantizer is lossless
//! (`sx = fl(fl(127·2⁻⁷)·fl(1/127)) = 2⁻⁷`, codes equal `c`), which the test re-derives on the
//! host and asserts before dispatch — otherwise the comparison would be measuring the quantizer,
//! not the DP4A fold.
//!
//! Coverage: Q8 (one `OpSUDot` per word) and Q4 (two nibble phases per word); `k` = 32, 64, 96,
//! 128, 160, 256, 512, 1024 (the 32-block boundaries); rows that are identically zero both by
//! centred code and by zero scale, ±127 codes, power-of-two and non-power-of-two f16 block
//! scales, and a zero activation block. `k % 32 != 0` is covered as a refusal of the registered
//! `GemvSpec` and of the served dispatcher, not as a result.
//!
//! No timing, no ratio, no claim beyond this comparison.
//!
//! Board row: `apeiron-kern-dp4a-w1-t01-DnMhEktartal5Q65p8dxKE`.

use apeiron::format::quant::f16_to_f32;
use apeiron::gpu::decode::layer::make_actq_gemv;
use apeiron::gpu::decode::pack::f32_to_f16;
use apeiron::gpu::decode::{rec_gemv, segs, GpuCtx};
use apeiron::gpu::nn::{ActQ, Fusion, GemvSpec, ScaleFace, SpecError};
use apeiron::gpu::simt::Carrier;
use apeiron::gpu::vulkan::*;

const QK: usize = 32;
/// `fl(1/127)` as the emitter's `cf_bits(0x3c01_0204)` writes it.
const INV127: u32 = 0x3c01_0204;
/// The activation scale this grid forces: `2⁻⁷`, exact in f32.
const SX: f32 = 1.0 / 128.0;
/// `2⁻⁷` in Q32.32.
const SX_Q32: i128 = 1 << 25;

/// f16 that round-trips exactly, or a failed premise.
fn f16_exact(x: f32) -> u16 {
    let h = f32_to_f16(x);
    assert_eq!(f16_to_f32(h), x, "test scale {x} is not f16-exact");
    h
}

fn next_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Power-of-two f16 scales, and the same width of non-power-of-two f16 scales (`3·2⁻ⁿ`).
const POW2: [f32; 4] = [0.125, 0.0625, 0.03125, 0.015625];
const MIXED: [f32; 8] = [
    0.125, 0.09375, // 3·2⁻⁵
    0.0625, 0.046875, // 3·2⁻⁶
    0.03125, 0.0234375, // 3·2⁻⁷
    0.015625, 0.01171875, // 3·2⁻⁸
];

fn scale_of(block: usize, power_only: bool) -> f32 {
    if power_only {
        POW2[block % POW2.len()]
    } else {
        MIXED[block % MIXED.len()]
    }
}

/// Chart centre (the affine offset) and code width per carrier.
fn chart_center(carrier: Carrier) -> u32 {
    match carrier {
        Carrier::Q8 => 128,
        Carrier::Q4 => 8,
        other => panic!("sweep covers Q8/Q4 only, got {other:?}"),
    }
}

fn codes_per_word(carrier: Carrier) -> u32 {
    carrier.pack()
}

/// One 32-code block for row `r` (row kind `r % 8`): 0 centred (weight zero), 1 zero scale,
/// 2 extremal, otherwise uniform random over the carrier's legal signed range.
fn block_codes(carrier: Carrier, r: usize, state: &mut u64) -> [u8; QK] {
    let center = chart_center(carrier) as i32;
    let lo_signed = match carrier {
        Carrier::Q8 => -127,
        Carrier::Q4 => -8,
        _ => unreachable!(),
    };
    let hi_signed = match carrier {
        Carrier::Q8 => 127,
        Carrier::Q4 => 7,
        _ => unreachable!(),
    };
    let mut c = [center as u8; QK];
    match r % 8 {
        0 => {}
        2 => {
            for (j, v) in c.iter_mut().enumerate() {
                *v = if j % 2 == 0 {
                    (center + hi_signed) as u8
                } else {
                    (center + lo_signed) as u8
                };
            }
        }
        _ => {
            let span = u32::from((hi_signed - lo_signed) as u16) + 1; // Q8: 255, Q4: 16
            for v in &mut c {
                let signed = lo_signed + (next_u32(state) % span) as i32;
                *v = (center + signed) as u8;
            }
        }
    }
    c
}

struct Chart {
    words: Vec<u32>,
    dm: Vec<u32>,
    /// f16-decoded block scale, `[block·m + row]`.
    d: Vec<f32>,
    lo: Vec<f32>,
    /// Chart code, `[row·k + j]`.
    codes: Vec<u8>,
}

fn build_chart(carrier: Carrier, m: usize, k: usize, power_only: bool, seed: u64) -> Chart {
    assert!(k % QK == 0, "chart builders require k % 32 == 0");
    let blocks = k / QK;
    let pack = codes_per_word(carrier) as usize;
    let width = (QK as u32 / pack as u32) as u32;
    let center = chart_center(carrier);
    let mut chart = Chart {
        words: vec![0u32; (k / pack) * m],
        dm: vec![0u32; blocks * m],
        d: vec![0.0; blocks * m],
        lo: vec![0.0; blocks * m],
        codes: vec![0u8; m * k],
    };
    let mut state = seed;
    for r in 0..m {
        let zero_scale = r % 8 == 1;
        for b in 0..blocks {
            let (d, lo) = if zero_scale {
                (0.0f32, 0.0f32)
            } else {
                let d = scale_of(b, power_only);
                (d, -(center as f32) * d)
            };
            let dh = f16_exact(d);
            let lh = f16_exact(lo);
            chart.dm[b * m + r] = u32::from(dh) | (u32::from(lh) << 16);
            chart.d[b * m + r] = f16_to_f32(dh);
            chart.lo[b * m + r] = f16_to_f32(lh);
            let c = block_codes(carrier, r, &mut state);
            for (j, &code) in c.iter().enumerate() {
                let kk = b * QK + j;
                chart.words[(kk / pack) * m + r] |= u32::from(code) << ((kk % pack) as u32 * width);
                chart.codes[r * k + kk] = code;
            }
        }
    }
    chart
}

/// The activation grid: codes uniform in `[-127, 127]` with `c[0] = 127` per block so the
/// device scale is exactly `2⁻⁷`, `c[1] = -127` on odd blocks, and `zero_block` (when in range)
/// all zero.
fn make_acts(k: usize, seed: u64, zero_block: Option<usize>) -> (Vec<f32>, Vec<i8>) {
    let blocks = k / QK;
    let mut state = seed;
    let mut codes = vec![0i8; k];
    for b in 0..blocks {
        if zero_block == Some(b) {
            continue;
        }
        for c in &mut codes[b * QK..(b + 1) * QK] {
            *c = ((next_u32(&mut state) % 255) as i32 - 127) as i8;
        }
        codes[b * QK] = 127;
        if b % 2 == 1 {
            codes[b * QK + 1] = -127;
        }
    }
    let x = codes.iter().map(|&c| f32::from(c) * SX).collect();
    (x, codes)
}

/// Host restatement of the served quantizer (`Kb::emit_quantize_from_shared`,
/// `src/gpu/nn/mod.rs:1101-1186`): f32 max/scale/sum in element order, then the same
/// `trunc(clamp(v/s + sign·0.5, -127, 127))` code.
fn host_quantize(x: &[f32]) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    let blocks = x.len() / QK;
    let mut codes = vec![0i8; x.len()];
    let mut sx = vec![0.0f32; blocks];
    let mut xs = vec![0.0f32; blocks];
    for b in 0..blocks {
        let mut mx = f32::from_bits(0x2b8c_bccc); // 1e-12
        let mut sum = 0.0f32;
        for &v in &x[b * QK..(b + 1) * QK] {
            mx = mx.max(v.abs());
            sum += v;
        }
        let sc = mx * f32::from_bits(INV127);
        sx[b] = sc;
        xs[b] = sum;
        for (j, &v) in x[b * QK..(b + 1) * QK].iter().enumerate() {
            let q = v / sc;
            let adj = if q > 0.0 { 0.5 } else { -0.5 };
            let qa = (q + adj).min(127.0).max(-127.0);
            codes[b * QK + j] = ((qa as i32) as u32 & 0xff) as u8 as i8;
        }
    }
    (codes, sx, xs)
}

struct Reference {
    y: Vec<f32>,
    sum_abs: Vec<f64>,
}

/// The exact Q32.32 contract value of every row of the chart against the code vector.
fn exact_reference(m: usize, k: usize, chart: &Chart, acts: &[i8]) -> Reference {
    let blocks = k / QK;
    let mut y = vec![0.0f32; m];
    let mut sum_abs = vec![0.0f64; m];
    for r in 0..m {
        let mut acc: i128 = 0;
        let mut abs = 0.0f64;
        for b in 0..blocks {
            let mut dot: i128 = 0;
            let mut sumc: i128 = 0;
            for j in 0..QK {
                let q = i128::from(chart.codes[r * k + b * QK + j]);
                let c = i128::from(acts[b * QK + j]);
                dot += q * c;
                sumc += c;
            }
            let d = f64::from(chart.d[b * m + r]);
            let lo = f64::from(chart.lo[b * m + r]);
            let d_q32 = (d * 4_294_967_296.0) as i128;
            let lo_q32 = (lo * 4_294_967_296.0) as i128;
            acc += (d_q32 * dot + lo_q32 * sumc) * SX_Q32;
            abs += d.abs() * f64::from(SX) * (dot.abs() as f64)
                + lo.abs() * f64::from(SX) * (sumc.abs() as f64);
        }
        let y_q32 = (acc + (1i128 << 31)) >> 32;
        y[r] = (y_q32 as f64 / 4_294_967_296.0) as f32;
        sum_abs[r] = abs;
    }
    Reference { y, sum_abs }
}

/// One dispatch of the served fused-actq DP4A GEMV, returning the split-K-reduced `m` vector.
///
/// # Safety
/// Vulkan pipeline/descriptor/buffer creation over caller-owned handles.
unsafe fn dispatch(
    vk: &Vk,
    g: &GpuCtx<'_>,
    carrier: Carrier,
    chart: &Chart,
    x: &[f32],
    m: usize,
    k: usize,
) -> Vec<f32> {
    let w = g.up_u32(&chart.words);
    let dm = g.up_u32(&chart.dm);
    let src = g.up_f32(x);
    let span = (k / QK) as u32;
    let s = segs(m as u32, span);
    assert!(s >= 1, "split-K segments must be at least one");
    let y = g.dev(m * s as usize * 4);
    let gemv = make_actq_gemv(
        g,
        carrier,
        w,
        dm,
        src,
        y,
        m as u32,
        k as u32,
        ActQ::Copy,
        None,
        ScaleFace::Dm32,
    );
    let (rb, rb_mem) = vk
        .buffer(
            (m * s as usize * 4) as u64,
            BUFFER_USAGE_TRANSFER_DST,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )
        .expect("readback buffer");
    let cb = vk.command_buffer();
    vk.begin(cb);
    rec_gemv(vk, cb, &gemv);
    let region = BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: (m * s as usize * 4) as u64,
    };
    (vk.copy_buffer)(cb, y, rb, 1, &region);
    vk.barrier(
        cb,
        STAGE_TRANSFER,
        STAGE_HOST,
        ACCESS_TRANSFER_WRITE,
        ACCESS_HOST_READ,
    );
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
    let partials: Vec<f32> = {
        let ptr = vk.map(rb_mem).cast::<f32>();
        std::slice::from_raw_parts(ptr, m * s as usize).to_vec()
    };
    (0..m)
        .map(|r| (0..s as usize).map(|seg| partials[seg * m + r]).sum())
        .collect()
}

struct Shape {
    carrier: Carrier,
    m: usize,
    k: usize,
    /// Power-of-two block scales only, where the whole sum is exactly representable in f32 and
    /// every lane must be bit-identical.
    exact: bool,
    zero_block: Option<usize>,
}

fn run_shape(vk: &Vk, g: &GpuCtx<'_>, shape: &Shape, seed: u64) -> (f64, usize, f64) {
    let (m, k) = (shape.m, shape.k);
    let chart = build_chart(shape.carrier, m, k, shape.exact, seed);
    let (x, want_codes) = make_acts(k, seed ^ 0x5eed_1234, shape.zero_block);

    // Premise check: the device quantizer is lossless on this grid. If this fails the comparison
    // is measuring activation quantisation, not the DP4A fold.
    let (got_codes, sx, xs) = host_quantize(&x);
    assert_eq!(got_codes, want_codes, "host quantizer codes drifted");
    for (b, &s) in sx.iter().enumerate() {
        if shape.zero_block == Some(b) {
            continue;
        }
        assert_eq!(s.to_bits(), SX.to_bits(), "block {b}: scale is not 2^-7");
    }
    for (b, &sum) in xs.iter().enumerate() {
        let want: f32 = want_codes[b * QK..(b + 1) * QK]
            .iter()
            .map(|&c| f32::from(c) * SX)
            .sum();
        assert_eq!(
            sum.to_bits(),
            want.to_bits(),
            "block {b}: block sum drifted"
        );
    }

    let reference = exact_reference(m, k, &chart, &want_codes);
    let gpu = unsafe { dispatch(vk, g, shape.carrier, &chart, &x, m, k) };

    let blocks = k / QK;
    let s = segs(m as u32, blocks as u32) as usize;
    let mut max_abs = 0.0f64;
    let mut identical = 0usize;
    let mut ref_scale = 0.0f64;
    for r in 0..m {
        let g = f64::from(gpu[r]);
        let e = f64::from(reference.y[r]);
        let err = (g - e).abs();
        max_abs = max_abs.max(err);
        ref_scale = ref_scale.max(e.abs());
        if gpu[r].to_bits() == reference.y[r].to_bits() {
            identical += 1;
        }
        // Forward bound: every per-block term is exactly representable for this scale set, so the
        // only error is the f32 summation of `blocks` partials plus the split-K reduction.
        let tol = 2.0 * f64::from(f32::EPSILON) * ((blocks + s + 8) as f64) * reference.sum_abs[r]
            + f64::MIN_POSITIVE;
        assert!(
            err <= tol,
            "{} m={m} k={k} row {r}: gpu {gpu:?} exact {:?} err {err:e} > tol {tol:e}",
            shape.carrier.name(),
            reference.y[r]
        );
        if shape.exact {
            assert_eq!(
                gpu[r].to_bits(),
                reference.y[r].to_bits(),
                "{} m={m} k={k} row {r}: exact-scale shape is not bit-identical: gpu {:?} exact {:?}",
                shape.carrier.name(),
                gpu[r],
                reference.y[r]
            );
        }
    }

    println!(
        "{:>2} m={m:<3} k={k:<4} blocks={blocks:<3} segs={s:<3} \
         max|err|={max_abs:.3e} bit-identical={identical}/{m} ref_scale={ref_scale:.3e}{}",
        shape.carrier.name(),
        if shape.exact { " [exact]" } else { "" }
    );
    (max_abs, identical, ref_scale)
}

#[test]
fn dp4a_sweep() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    let (g, _stage_mem) = GpuCtx::new(&vk).expect("GpuCtx::new");

    // ── K tails: the chart carrier's registered refusal, and the served dispatcher's ──────────
    for &k in &[33u32, 47, 63, 65, 100, 130, 255] {
        for carrier in [Carrier::Q8, Carrier::Q4] {
            let spec = GemvSpec::decode(carrier, true).fused(Fusion::ActQ(ActQ::Copy));
            assert_eq!(
                spec.emit(8, k, 256),
                Err(SpecError::KNotBlockAligned { k }),
                "{carrier:?} k={k} must be refused by the registered spec"
            );
        }
        println!("refused k={k} (k % 32 != 0): GemvSpec emits KNotBlockAligned");
    }
    // The dispatcher turns that Err into a panic; observe it firing, with its message suppressed.
    let old_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let served = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let w = g.up_u32(&[0u32; 8]);
        let dm = g.up_u32(&[0u32; 2]);
        let src = g.up_f32(&[0.0f32; 33]);
        let y = g.dev(8 * 4);
        make_actq_gemv(
            &g,
            Carrier::Q8,
            w,
            dm,
            src,
            y,
            8,
            33,
            ActQ::Copy,
            None,
            ScaleFace::Dm32,
        );
    }));
    std::panic::set_hook(old_hook);
    assert!(
        served.is_err(),
        "make_actq_gemv accepted k=33; the chart carrier must refuse a K tail"
    );
    println!("refused k=33 at the served dispatcher: make_actq_gemv panicked");

    // ── the sweep ─────────────────────────────────────────────────────────────────────────────
    println!("carrier m   k    blocks segs  max|gpu-exact|   bit-identical  ref_scale");
    let mut shapes: Vec<Shape> = Vec::new();
    for &carrier in &[Carrier::Q8, Carrier::Q4] {
        for &(m, k) in &[(8, 32), (8, 64), (16, 96), (16, 128), (16, 256)] {
            shapes.push(Shape {
                carrier,
                m,
                k,
                exact: true,
                zero_block: None,
            });
        }
        for &(m, k, zb) in &[
            (16, 96, Some(1usize)),
            (16, 160, None),
            (16, 256, None),
            (16, 512, None),
            (16, 1024, None),
        ] {
            shapes.push(Shape {
                carrier,
                m,
                k,
                exact: false,
                zero_block: zb,
            });
        }
    }

    let mut worst = 0.0f64;
    for (i, shape) in shapes.iter().enumerate() {
        let (max_abs, _, _) = run_shape(&vk, &g, shape, 0xd4a5_0000_u64 + i as u64);
        worst = worst.max(max_abs);
    }
    println!("sweep max |gpu - exact Q32.32| over every shape and row: {worst:.3e}");
}
