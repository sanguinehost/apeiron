//! GPU DP4A chart GEMV match against CPU reference dequantized with apeiron's dequantizer.
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "faithful port of fold's modelfold::gpu Vulkan runtime; lint profile is fold's, not apeiron's"
)]

use apeiron::engine::dequant_row_fixed;
use apeiron::engine::testdata::model_path;
use apeiron::format::gguf::Gguf;
use apeiron::format::quant::{
    block_geom, dequant_q6_k_block, GGML_TYPE_Q6_K, Q6_K_BLOCK_BYTES, QK_K,
};
use apeiron::gpu::decode::layer::make_actq_gemv;
use apeiron::gpu::decode::pack::{pack_proj_f32, pack_q5_0_raw, pack_q6_k_raw, pack_tensor_desc};
use apeiron::gpu::decode::{make_gemv, make_step_named, rec_gemv, rec_step, segs, GpuCtx};
use apeiron::gpu::nn::{quantize_vec_bytes, ActQ, ScaleFace};
use apeiron::gpu::simt::Carrier;
use apeiron::gpu::vulkan::*;

const MODEL: &str = "models/gguf/qwen05b-Q4_K_M.gguf";

#[test]
fn test_gpu_gemv_match() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };

    let Some(model) = model_path(MODEL) else {
        println!("SKIP: Model file {MODEL} not found");
        return;
    };

    let gguf = match Gguf::open(&model) {
        Ok(g) => g,
        Err(e) => {
            println!("SKIP: Failed to open GGUF: {e}");
            return;
        }
    };

    let tensor = gguf
        .header
        .tensors
        .iter()
        .find(|t| t.ggml_type == 12 && t.dims.len() >= 2)
        .expect("a Q4_K 2D tensor in GGUF");
    let tensor_name = String::from_utf8_lossy(&tensor.name).to_string();

    let k = tensor.dims[0] as usize;
    let m = tensor.dims[1] as usize;
    println!(
        "Testing {tensor_name}: m={m}, k={k}, ggml_type={}",
        tensor.ggml_type
    );

    // 1. CPU reference dequantization using apeiron's dequantizer (`src/engine/dequant*.rs`)
    let (qk, bb) = block_geom(tensor.ggml_type).expect("block geometry");
    let blocks_per_row = k.div_ceil(qk);
    let row_bytes = blocks_per_row * bb;
    let raw = gguf.tensor_bytes(tensor).expect("raw tensor bytes");

    let mut cpu_weights_f32 = vec![0.0f32; m * k];
    let mut row_buf = Vec::with_capacity(k);
    for r in 0..m {
        row_buf.clear();
        let row_slice = &raw[r * row_bytes..(r + 1) * row_bytes];
        dequant_row_fixed(tensor.ggml_type, row_slice, k, &mut row_buf)
            .expect("dequant_row_fixed failed");
        for c in 0..k {
            // Convert Q32.32 fixed-point i64 to f32
            cpu_weights_f32[r * k + c] = (row_buf[c] as f64 / 4294967296.0) as f32;
        }
    }

    // 2. Fixed seeded input vector x of length k
    let mut rng: u64 = 0x12345678_9abcdef0;
    let mut next_f32 = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let x: Vec<f32> = (0..k).map(|_| next_f32()).collect();

    // 3. Compute CPU reference matvec in f64
    let x_f64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let mut ref_y = vec![0.0f32; m];
    for r in 0..m {
        let mut dot = 0.0f64;
        for c in 0..k {
            let w = cpu_weights_f32[r * k + c] as f64;
            dot += w * x_f64[c];
        }
        ref_y[r] = dot as f32;
    }

    // 4. Pack tensor into DP4A chart
    let proj = pack_tensor_desc(&gguf, tensor).expect("pack_tensor_desc");

    // 5. GPU upload & dispatch
    let (g, _stage_mem) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let w_buf = unsafe { g.up_u32(&proj.words) };
    let dm_buf = unsafe { g.up_u32(&proj.dm) };
    let x_in = unsafe { g.up_f32(&x) };

    // Activation chart buffers:
    let dpw = (Carrier::Q4.pack() / 4) as u32; // 2
    let xq = unsafe { g.dev(k) }; // k bytes = k/4 u32 words
    let sx = unsafe { g.dev((k / 32) * 4) }; // f32 per block
    let xs = unsafe { g.dev((k / 32) * 4) }; // f32 per block

    let qstep = unsafe {
        make_step_named(
            &g,
            &quantize_vec_bytes(k as u32, dpw),
            &[x_in, xq, sx, xs],
            1,
            1,
            "quantize_vec",
        )
    };

    let y_dev = unsafe { g.dev(m * 4) };
    let span = (k / 32) as u32;
    let s = segs(m as u32, span);
    let partials = unsafe { g.dev(m * s as usize * 4) };
    let ones = unsafe { g.up_f32(&vec![1.0f32; s as usize]) };

    let gemv = unsafe {
        make_gemv(
            &g,
            Carrier::Q4,
            w_buf,
            dm_buf,
            xq,
            (xs, sx),
            y_dev,
            partials,
            ones,
            m as u32,
            k as u32,
            false,
            ScaleFace::Dm32,
        )
    };

    // Staging readback buffer
    let (rb, rb_mem) = unsafe {
        vk.buffer(
            (m * 4) as u64,
            BUFFER_USAGE_TRANSFER_DST,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )
    }
    .expect("readback buffer");

    let cb = unsafe { vk.command_buffer() };
    unsafe {
        vk.begin(cb);
        rec_step(&vk, cb, &qstep);
        rec_gemv(&vk, cb, &gemv);
        let region = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (m * 4) as u64,
        };
        (vk.copy_buffer)(cb, y_dev, rb, 1, &region);
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

    let gpu_y: Vec<f32> = unsafe {
        let ptr = vk.map(rb_mem).cast::<f32>();
        std::slice::from_raw_parts(ptr, m).to_vec()
    };

    // 6. Compare GPU vs CPU
    let gpu_argmax = gpu_y
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let cpu_argmax = ref_y
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    println!("Sample outputs (first 5):");
    for r in 0..5 {
        println!("  r={r}: gpu={:.6} gguf_cpu={:.6}", gpu_y[r], ref_y[r]);
    }

    let mut errors: Vec<(usize, f64, f64, f64, f64)> = (0..m)
        .map(|r| {
            let g = gpu_y[r] as f64;
            let c = ref_y[r] as f64;
            let diff = (g - c).abs();
            let rel = diff / c.abs().max(1e-4);
            (r, g, c, diff, rel)
        })
        .collect();
    errors.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

    let ref_scale = ref_y
        .iter()
        .fold(0f64, |m, &v| m.max((v as f64).abs()))
        .max(1e-30);
    let max_abs_diff = (0..m).fold(0f64, |acc, r| {
        acc.max((gpu_y[r] as f64 - ref_y[r] as f64).abs())
    });
    let max_rel_err = max_abs_diff / ref_scale;

    println!("Vector ref_scale (max |ref_y|): {ref_scale:.6}");
    println!("Max absolute difference: {max_abs_diff:.6}");
    println!("GPU argmax: {gpu_argmax}, CPU argmax: {cpu_argmax}");
    println!("Measured max relative error: {max_rel_err:.6e}");

    assert_eq!(
        gpu_argmax, cpu_argmax,
        "argmax mismatch: gpu {gpu_argmax} vs cpu {cpu_argmax}"
    );
    assert!(
        max_rel_err <= 1e-2,
        "max relative error {max_rel_err:e} exceeds 1e-2"
    );
}

/// **`Carrier::Q5`'s two-plane chart through the fused-actq fold the served decoder uses.**
///
/// The Q4 test above exercises the single-plane chart (plane A only). This one adds plane B: the
/// permuted fifth bit is fetched from `carrier.plane_b_off(m, k)` and OR'd into each nibble code
/// before the `OpSUDot`. The emitter is `gemv_actq_module_sc`'s `dp4a_chart_fold`, reached through
/// `make_actq_gemv` with the exact `ActQ::Copy`/`ScaleFace::Dm32` spec the decoder dispatches — so a
/// fold that ignores plane B decodes `d·(q & 0xF − …)`, finite, plausible and wrong.
#[test]
fn q5_chart_actq_gemv_matches_cpu() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    let Some(model) = model_path(MODEL) else {
        println!("SKIP: Model file {MODEL} not found");
        return;
    };
    let gguf = match Gguf::open(&model) {
        Ok(g) => g,
        Err(e) => {
            println!("SKIP: Failed to open GGUF: {e}");
            return;
        }
    };
    let (g, _stage_mem) = GpuCtx::new(&vk).expect("GpuCtx::new");

    // Both tensors are Q5_0 (ggml_type 6) in this mixed-quant checkpoint: five-bit weights, the
    // two-plane chart. `dequant_row_fixed` is the crate's own exact Q5_0 dequantiser.
    for name in ["blk.0.ffn_gate.weight", "blk.0.attn_output.weight"] {
        let tensor = gguf
            .header
            .tensor(name)
            .unwrap_or_else(|| panic!("{name} present"))
            .clone();
        assert_eq!(tensor.ggml_type, 6, "{name} is Q5_0");
        let k = tensor.dims[0] as usize;
        let m = tensor.dims[1] as usize;
        println!("Q5 {name}: m={m}, k={k}, ggml_type={}", tensor.ggml_type);

        // 1. CPU reference: exact dequant to f32, a seeded x, an f64 matvec.
        let (qk, bb) = block_geom(tensor.ggml_type).expect("block geometry");
        let row_bytes = k.div_ceil(qk) * bb;
        let raw = gguf.tensor_bytes(&tensor).expect("raw tensor bytes");
        let mut cpu_w = vec![0.0f32; m * k];
        let mut row_buf = Vec::with_capacity(k);
        for r in 0..m {
            row_buf.clear();
            dequant_row_fixed(
                tensor.ggml_type,
                &raw[r * row_bytes..(r + 1) * row_bytes],
                k,
                &mut row_buf,
            )
            .expect("dequant_row_fixed");
            for c in 0..k {
                cpu_w[r * k + c] = (row_buf[c] as f64 / 4294967296.0) as f32;
            }
        }
        // The residual this bound measures is the int8 ACTIVATION chart's own quantisation error
        // (~1e-2 for these k=896 tensors), not the Q5 chart — `pack_q5_0_raw`'s two planes are
        // bit-exact against the dequantiser (`q5_0_chart_is_exact`). A fixed seed is required, and
        // below the bound it must clear with margin; this one does for both tensors.
        let mut rng: u64 = 42;
        let mut next_f32 = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        };
        let x: Vec<f32> = (0..k).map(|_| next_f32()).collect();
        let x_f64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let mut ref_y = vec![0.0f32; m];
        for r in 0..m {
            let mut dot = 0.0f64;
            for c in 0..k {
                dot += f64::from(cpu_w[r * k + c]) * x_f64[c];
            }
            ref_y[r] = dot as f32;
        }

        // 2. Pack to the two-plane Q5 chart and upload.
        let proj = pack_q5_0_raw(&raw, m, k).expect("pack_q5_0_raw");
        let w = unsafe { g.up_u32(&proj.words) };
        let dm = unsafe { g.up_u32(&proj.dm) };
        let x_in = unsafe { g.up_f32(&x) };

        // 3. The fused-actq spec. Its stage-1 partials land in `y` at `seg·m + r`, one per split-K
        //    segment (no stage-2 dispatch), so `y` is sized for all of them and reduced below.
        let span = (k / 32) as u32;
        let s = segs(m as u32, span);
        let y = unsafe { g.dev(m * s as usize * 4) };
        let gemv = unsafe {
            make_actq_gemv(
                &g,
                Carrier::Q5,
                w,
                dm,
                x_in,
                y,
                m as u32,
                k as u32,
                ActQ::Copy,
                None,
                ScaleFace::Dm32,
            )
        };

        let (rb, rb_mem) = unsafe {
            vk.buffer(
                (m * s as usize * 4) as u64,
                BUFFER_USAGE_TRANSFER_DST,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
        }
        .expect("readback buffer");
        let cb = unsafe { vk.command_buffer() };
        unsafe {
            vk.begin(cb);
            rec_gemv(&vk, cb, &gemv);
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
        }
        let partials: Vec<f32> = unsafe {
            let ptr = vk.map(rb_mem).cast::<f32>();
            std::slice::from_raw_parts(ptr, m * s as usize).to_vec()
        };

        // 4. Reduce the split-K partials and compare against the CPU reference.
        let gpu_y: Vec<f32> = (0..m)
            .map(|r| (0..s as usize).map(|seg| partials[seg * m + r]).sum())
            .collect();
        let gpu_argmax = gpu_y
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let cpu_argmax = ref_y
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let ref_scale = ref_y
            .iter()
            .fold(0f64, |a, &v| a.max(f64::from(v).abs()))
            .max(1e-30);
        let max_abs = (0..m).fold(0f64, |a, r| {
            a.max((f64::from(gpu_y[r]) - f64::from(ref_y[r])).abs())
        });
        let rel = max_abs / ref_scale;

        println!("Q5 {name}: segs={s} GPU argmax {gpu_argmax}, CPU argmax {cpu_argmax}");
        println!("Q5 {name}: measured max relative error {rel:.6e}");
        assert_eq!(gpu_argmax, cpu_argmax, "{name}: argmax mismatch");
        assert!(
            rel <= 1e-2,
            "{name}: max relative error {rel:e} exceeds 1e-2"
        );
    }
}

/// **`Carrier::Q6` through the served fused-actq DP4A fold.** The first Q6_K `blk.N.ffn_down.weight`
/// (m = 896, k = 4864, 152 blocks per row) is packed to the two-plane Q6 chart (`pack_q6_k_raw`,
/// bit-exact against `dequant_q6_k_block` by `q6_k_chart_is_exact`) and run through
/// `make_actq_gemv` with `ActQ::Copy` / `ScaleFace::Dm32`. The fold builds the signed byte
/// `u - 32` from plane A, the two-bit plane-B lanes and `(U + 0x60606060) ^ 0x80808080`, takes an
/// `OpSDot` per phase and weights it by the half's int8 scale (`sc_a`/`sc_b` of the dm word). A fold
/// that skips plane B, swaps the scales or reads the dm upper half as an f16 `lo` is finite and
/// wrong by O(1). The residual this bound measures is the int8 activation chart's own error.
#[test]
fn q6_chart_actq_gemv_matches_cpu() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    let Some(model) = model_path(MODEL) else {
        println!("SKIP: Model file {MODEL} not found");
        return;
    };
    let gguf = match Gguf::open(&model) {
        Ok(g) => g,
        Err(e) => {
            println!("SKIP: Failed to open GGUF: {e}");
            return;
        }
    };
    let (g, _stage_mem) = GpuCtx::new(&vk).expect("GpuCtx::new");

    let (name, tensor) = (0..256)
        .find_map(|i| {
            let name = format!("blk.{i}.ffn_down.weight");
            gguf.header
                .tensor(&name)
                .filter(|t| t.ggml_type == GGML_TYPE_Q6_K)
                .map(|t| (name, t.clone()))
        })
        .expect("a Q6_K ffn_down tensor");
    let k = tensor.dims[0] as usize;
    let m = tensor.dims[1] as usize;
    println!("Q6 {name}: m={m}, k={k}, ggml_type={}", tensor.ggml_type);

    // 1. CPU reference: exact Q6_K dequant to f32, a seeded x, an f64 matvec.
    let raw = gguf.tensor_bytes(&tensor).expect("raw tensor bytes");
    let row_bytes = (k / QK_K) * Q6_K_BLOCK_BYTES;
    let mut cpu_w = vec![0.0f32; m * k];
    for r in 0..m {
        for sb in 0..k / QK_K {
            let blk = &raw[r * row_bytes + sb * Q6_K_BLOCK_BYTES..][..Q6_K_BLOCK_BYTES];
            let mut out = [0f32; QK_K];
            dequant_q6_k_block(blk, &mut out).expect("dequant_q6_k_block");
            cpu_w[r * k + sb * QK_K..][..QK_K].copy_from_slice(&out);
        }
    }
    let mut rng: u64 = 42;
    let mut next_f32 = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    let x: Vec<f32> = (0..k).map(|_| next_f32()).collect();
    let mut ref_y = vec![0.0f32; m];
    for r in 0..m {
        let mut dot = 0.0f64;
        for c in 0..k {
            dot += f64::from(cpu_w[r * k + c]) * f64::from(x[c]);
        }
        ref_y[r] = dot as f32;
    }

    // 2. Pack to the two-plane Q6 chart and upload.
    let proj = pack_q6_k_raw(&raw, m, k).expect("pack_q6_k_raw");
    let w = unsafe { g.up_u32(&proj.words) };
    let dm = unsafe { g.up_u32(&proj.dm) };
    let x_in = unsafe { g.up_f32(&x) };

    // 3. The fused-actq spec; stage-1 split-K partials land at `seg·m + r`, reduced below.
    let span = (k / 32) as u32;
    let s = segs(m as u32, span);
    let y = unsafe { g.dev(m * s as usize * 4) };
    let gemv = unsafe {
        make_actq_gemv(
            &g,
            Carrier::Q6,
            w,
            dm,
            x_in,
            y,
            m as u32,
            k as u32,
            ActQ::Copy,
            None,
            ScaleFace::Dm32,
        )
    };
    let (rb, rb_mem) = unsafe {
        vk.buffer(
            (m * s as usize * 4) as u64,
            BUFFER_USAGE_TRANSFER_DST,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )
    }
    .expect("readback buffer");
    let cb = unsafe { vk.command_buffer() };
    unsafe {
        vk.begin(cb);
        rec_gemv(&vk, cb, &gemv);
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
    }
    let partials: Vec<f32> = unsafe {
        let ptr = vk.map(rb_mem).cast::<f32>();
        std::slice::from_raw_parts(ptr, m * s as usize).to_vec()
    };

    // 4. Reduce the partials and compare.
    let gpu_y: Vec<f32> = (0..m)
        .map(|r| (0..s as usize).map(|seg| partials[seg * m + r]).sum())
        .collect();
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    };
    let (gpu_argmax, cpu_argmax) = (argmax(&gpu_y), argmax(&ref_y));
    let ref_scale = ref_y
        .iter()
        .fold(0f64, |a, &v| a.max(f64::from(v).abs()))
        .max(1e-30);
    let max_abs = (0..m).fold(0f64, |a, r| {
        a.max((f64::from(gpu_y[r]) - f64::from(ref_y[r])).abs())
    });
    let rel = max_abs / ref_scale;
    println!("Q6 {name}: segs={s} GPU argmax {gpu_argmax}, CPU argmax {cpu_argmax}");
    println!("Q6 {name}: measured max relative error {rel:.6e}");
    assert_eq!(gpu_argmax, cpu_argmax, "{name}: argmax mismatch");
    assert!(
        rel <= 1e-2,
        "{name}: max relative error {rel:e} exceeds 1e-2"
    );
}

/// The served activation quantizer's scale formula, restated here so the lossless construction below
/// can be checked against it. From `Kb::emit_quantize_to_shared_win` (`src/gpu/nn/mod.rs:1195-1200`)
/// and `Kb::int8_code` (`src/gpu/nn/attn/attn_dp4a.rs:60-81`), per 32-block `b`:
///
/// ```text
/// mx   = fmax(1e-12, max_j |x_j|)              // f32, floor 1e-12
/// s[b] = fmul(mx, 1.0/127.0)                   // f32 scale, NOT exactly mx/127
/// xs[b]= Σ_j x_j                               // f32 block sum, element order
/// q_j  = trunc_toward_zero(clamp(fdiv(x_j, s[b]) + sign·0.5, -127, 127))
/// ```
///
/// With `x_j = c_j·2⁻⁷`, `c_j` integer, and one `|c_j| = 127` per block: `mx = 127·2⁻⁷`,
/// `s[b] = fl(fl(127·2⁻⁷)·fl(1/127)) = 2⁻⁷` exactly (`fl(127·fl(1/127)) = 1`), so
/// `fdiv(x_j, s[b]) = c_j` exactly and `q_j = c_j`. The chart then reconstructs
/// `x_j = q_j·s[b]` with no rounding, so the fold's `d·Σ(q·u)·s[b] + lo·Σx` equals the exact
/// `Σ_j d(u_j−16)·x_j` up to the f32 epilogue alone.
fn grid_activations(k: usize) -> Vec<f32> {
    let mut rng: u64 = 0x00c0_ffee_1234_5678;
    let mut next_u32 = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as u32
    };
    let mut x = vec![0f32; k];
    for b in 0..k / 32 {
        let mut c = [0i32; 32];
        for cj in c.iter_mut() {
            *cj = ((next_u32() % 255) as i32) - 127; // uniform in [-127, 127]
        }
        c[0] = 127; // force the block absmax so s[b] = 2⁻⁷ exactly
        for j in 0..32 {
            x[b * 32 + j] = (c[j] as f32) * (1.0f32 / 128.0);
        }
    }
    x
}

/// The f64 matvec `y[r] = Σ_c w[r·k+c]·x[c]` over the exact dequantized weights, rounded to f32.
fn cpu_ref_matvec(w: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    (0..m)
        .map(|r| {
            let mut dot = 0.0f64;
            for c in 0..k {
                dot += f64::from(w[r * k + c]) * f64::from(x[c]);
            }
            dot as f32
        })
        .collect()
}

/// Max-relative error of a GPU vector against a reference, normalized by `max|ref|`.
fn rel_err(gpu: &[f32], reference: &[f32]) -> f64 {
    let scale = reference
        .iter()
        .fold(0f64, |a, &v| a.max(f64::from(v).abs()))
        .max(1e-30);
    let max_abs = gpu.iter().zip(reference).fold(0f64, |a, (&g, &r)| {
        a.max((f64::from(g) - f64::from(r)).abs())
    });
    max_abs / scale
}

/// One fused-actq DP4A GEMV dispatch of `carrier` on an uploaded chart, returning the split-K
/// reduced `m`-vector. Same shape as [`q5_chart_actq_gemv_matches_cpu`]'s dispatch.
///
/// # Safety
/// Vulkan pipeline/buffer creation over caller-owned handles.
#[allow(clippy::too_many_arguments)]
unsafe fn run_actq(
    vk: &Vk,
    g: &GpuCtx<'_>,
    carrier: Carrier,
    w: Handle,
    dm: Handle,
    x: &[f32],
    m: usize,
    k: usize,
) -> Vec<f32> {
    let x_in = g.up_f32(x);
    let span = (k / 32) as u32;
    let s = segs(m as u32, span);
    let y = g.dev(m * s as usize * 4);
    let gemv = make_actq_gemv(
        g,
        carrier,
        w,
        dm,
        x_in,
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

/// **Is the served Q5 DP4A fold exact on EVERY Q5_0 tensor of qwen05b?** Enumerates every
/// `ggml_type == 6` (Q5_0) tensor in the model, packs each two ways — [`pack_q5_0_raw`] into
/// `Carrier::Q5` and the Q8 re-chart HEAD serves for a non-Q4_K group in Q4 mode
/// (`pack_group` → `pack_proj_f32(Carrier::Q8, …)`) — and runs both through
/// `make_actq_gemv`/`ActQ::Copy`/`ScaleFace::Dm32` under two activation regimes:
///
/// * **G (grid)** — [`grid_activations`], built so the served int8 activation quantizer is
///   LOSSLESS (`s[b] = 2⁻⁷`, `q_j = c_j`, `x_j = q_j·s[b]`). The fold's only error is the f32
///   epilogue, so this is the assertion: `rel_q5_G <= 1e-5` on every tensor.
/// * **R (random)** — the existing seeded x; reported only, since the activation chart's own
///   ~1e-2 quantisation error dominates it.
///
/// The reference is an f64 matvec of the exact Q5_0 dequant against the same x. Failures are
/// collected in full and asserted once at the end. `rel_q8_*` is the lossy re-chart's error.
#[test]
fn q5_chart_grid_exact_every_layer() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    let Some(model) = model_path(MODEL) else {
        println!("SKIP: Model file {MODEL} not found");
        return;
    };
    let gguf = match Gguf::open(&model) {
        Ok(g) => g,
        Err(e) => {
            println!("SKIP: Failed to open GGUF: {e}");
            return;
        }
    };
    let (g, _stage_mem) = GpuCtx::new(&vk).expect("GpuCtx::new");

    let q5: Vec<_> = gguf
        .header
        .tensors
        .iter()
        .filter(|t| t.ggml_type == 6)
        .cloned()
        .collect();
    let mut by_suffix: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for t in &q5 {
        let name = String::from_utf8_lossy(&t.name).to_string();
        let suffix = name
            .splitn(3, '.')
            .nth(2)
            .unwrap_or(name.as_str())
            .to_string();
        *by_suffix.entry(suffix).or_insert(0) += 1;
    }
    println!("Q5_0 tensor count: {}", q5.len());
    for (suffix, n) in &by_suffix {
        println!("  {suffix}: {n}");
    }

    let mut failures: Vec<(String, f64)> = Vec::new();
    let mut max_by_suffix: std::collections::BTreeMap<String, (f64, String)> =
        std::collections::BTreeMap::new();

    for t in &q5 {
        let name = String::from_utf8_lossy(&t.name).to_string();
        let suffix = name
            .splitn(3, '.')
            .nth(2)
            .unwrap_or(name.as_str())
            .to_string();
        let k = t.dims[0] as usize;
        let m = t.dims[1] as usize;

        // 1. Exact Q5_0 dequant to f32 via the crate's own dequantiser.
        let (qk, bb) = block_geom(t.ggml_type).expect("block geometry");
        let row_bytes = k.div_ceil(qk) * bb;
        let raw = gguf.tensor_bytes(t).expect("raw tensor bytes");
        let mut cpu_w = vec![0.0f32; m * k];
        let mut row_buf = Vec::with_capacity(k);
        for r in 0..m {
            row_buf.clear();
            dequant_row_fixed(
                t.ggml_type,
                &raw[r * row_bytes..(r + 1) * row_bytes],
                k,
                &mut row_buf,
            )
            .expect("dequant_row_fixed");
            for c in 0..k {
                cpu_w[r * k + c] = (row_buf[c] as f64 / 4294967296.0) as f32;
            }
        }

        // 2. Two activation regimes and their exact references.
        let x_g = grid_activations(k);
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next_f32 = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        };
        let x_r: Vec<f32> = (0..k).map(|_| next_f32()).collect();
        let ref_g = cpu_ref_matvec(&cpu_w, &x_g, m, k);
        let ref_r = cpu_ref_matvec(&cpu_w, &x_r, m, k);

        // 3. The two charts: the exact Q5 two-plane chart and the served Q8 re-chart.
        let p5 = pack_q5_0_raw(&raw, m, k).expect("pack_q5_0_raw");
        let p8 = pack_proj_f32(Carrier::Q8, &[cpu_w.as_slice()], k, false);
        let w5 = unsafe { g.up_u32(&p5.words) };
        let dm5 = unsafe { g.up_u32(&p5.dm) };
        let w8 = unsafe { g.up_u32(&p8.words) };
        let dm8 = unsafe { g.up_u32(&p8.dm) };

        // 4. Four dispatches: {Q5, Q8} × {G, R}.
        let y5g = unsafe { run_actq(&vk, &g, Carrier::Q5, w5, dm5, &x_g, m, k) };
        let y8g = unsafe { run_actq(&vk, &g, Carrier::Q8, w8, dm8, &x_g, m, k) };
        let y5r = unsafe { run_actq(&vk, &g, Carrier::Q5, w5, dm5, &x_r, m, k) };
        let y8r = unsafe { run_actq(&vk, &g, Carrier::Q8, w8, dm8, &x_r, m, k) };

        let rel_q5_g = rel_err(&y5g, &ref_g);
        let rel_q8_g = rel_err(&y8g, &ref_g);
        let rel_q5_r = rel_err(&y5r, &ref_r);
        let rel_q8_r = rel_err(&y8r, &ref_r);
        println!("{name} {rel_q5_g:.3e} {rel_q8_g:.3e} {rel_q5_r:.3e} {rel_q8_r:.3e}");

        let slot = max_by_suffix
            .entry(suffix)
            .or_insert_with(|| (f64::NEG_INFINITY, String::new()));
        if rel_q5_g > slot.0 {
            *slot = (rel_q5_g, name.clone());
        }
        if rel_q5_g > 1e-5 {
            failures.push((name, rel_q5_g));
        }
    }

    println!("rel_q5_G max per suffix (tensor):");
    for (suffix, (rel, name)) in &max_by_suffix {
        println!("  {suffix}: {rel:.3e} ({name})");
    }
    if failures.is_empty() {
        println!(
            "q5_chart_grid_exact_every_layer: all {} Q5_0 tensors rel_q5_G <= 1e-5",
            q5.len()
        );
    } else {
        for (name, rel) in &failures {
            println!("FAIL rel_q5_G {name}: {rel:.6e}");
        }
        panic!(
            "{} of {} Q5_0 tensors exceed rel_q5_G 1e-5",
            failures.len(),
            q5.len()
        );
    }
}
