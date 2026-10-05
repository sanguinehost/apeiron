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
use apeiron::format::gguf::Gguf;
use apeiron::format::quant::block_geom;
use apeiron::gpu::decode::pack::pack_tensor_desc;
use apeiron::gpu::decode::{make_gemv, make_step_named, rec_gemv, rec_step, segs, GpuCtx};
use apeiron::gpu::vulkan::*;
use apeiron::simt::Carrier;
use apeiron::simt_nn::{quantize_vec_bytes, ScaleFace};
use std::path::Path;

#[test]
fn test_gpu_gemv_match() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };

    let model_path = "/home/socol/models/gguf/qwen05b-Q4_K_M.gguf";
    if !Path::new(model_path).exists() {
        println!("SKIP: Model file {model_path} not found");
        return;
    }

    let gguf = match Gguf::open(model_path) {
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
