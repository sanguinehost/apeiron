//! Whole-prompt GPU prefill (`GpuDecode::prefill`) against the token-by-token GPU path and the CPU
//! engine, Qwen2.5-0.5B-Q4_K_M.
//!
//! For each of the five prompts of `gpu_decode_step`: prefill the whole prompt in one call, take the
//! next-token argmax, and assert it equals (a) the token-by-token `GpuDecode::step` argmax and
//! (b) the CPU engine's greedy first token. Reports (does not assert) the max relative error of the
//! final-normed hidden at the last prompt position, GPU prefill vs GPU token-by-token, and how many
//! of 8 greedy continuation tokens (decoding on from the prefilled KV cache) agree with the
//! token-by-token path.
//!
//!   cargo test --release --locked --features gpu --test `gpu_prefill_match` -- --ignored --nocapture
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_decode_step test"
)]

use apeiron::engine::testdata::model_path;
use apeiron::engine::{Engine, KernelMode};
use apeiron::format::bpe::Bpe;
use apeiron::format::gguf::Gguf;
use apeiron::gpu::decode::GpuDecode;
use apeiron::gpu::decode::{make_step_named, rec_step, GpuCtx};
use apeiron::gpu::vulkan::*;
use apeiron::simt::Carrier;
use apeiron::simt_nn::{attn_prefill_coopmat_bytes, quantize_batch_bytes};

const MODEL: &str = "models/gguf/qwen05b-Q4_K_M.gguf";
const TOKENIZER: &str = "models/qwen2.5-0.5b/tokenizer.json";
const PROMPTS: [&str; 6] = [
    "The capital of France is",
    "The largest planet in the solar system is",
    "Water boils at a temperature of",
    "The author of Hamlet is",
    "Two plus two equals",
    // >= 3 key slabs of 64 and >= 4 row tiles of 32: the online-softmax rescale (slab maxima that
    // move between slabs) and the row-tile launch order are exercised only past one slab
    "In the early years of the nineteenth century the town grew slowly along the river, first as a \
     market for grain and wool, then as a place where merchants kept warehouses and boatmen found \
     work. The mill, built in 1823 by a family from the north, employed forty people at its \
     height, and the church bell rang the hours for the whole valley. When the railway arrived \
     forty years later it bypassed the old quay entirely, and within a generation the warehouses \
     stood empty. Visitors today walk the towpath past the ruined mill, read the plaques on the \
     bridge, and ask the same question the first settlers asked: the name of the river is",
];
const TMAX: u32 = 512;
const MAXB: usize = 256;
const N_CONT: usize = 8;

/// SEL-TIE-PF: identical constants and meaning to
/// `tests/gpu_decode_step.rs:47-55` (SEL-TIE-CONTRACT). A prompt whose whole-prompt GPU prefill
/// first token differs from the token-by-token GPU first token is admissible only when all hold —
/// prefill agrees exactly with the CPU reference, the prompt is not `long` (its `cpu_logits` is a
/// copy of the token-by-token row), the CPU-exact logit gap
/// `cpu_logits[cpu_first] - cpu_logits[tbt_first] <= TIE_GAP` (logit units, f64), and the relative
/// L2 error of the token-by-token logits against the CPU logits `<= TIE_REL_L2` — and at most
/// `TIE_MAX` prompt per arm run is admitted this way. A prompt that does not qualify keeps the exact
/// `assert_eq!`. CPU logits are `Engine::step_logits_state(..., KernelMode::Exact)` over the prompt
/// ids, Q32 scaled by `1/4294967296.0`.
const TIE_GAP: f64 = 0.5;
const TIE_REL_L2: f64 = 0.25;
const TIE_MAX: usize = 1;

fn continue_greedy(d: &mut GpuDecode, first: usize, mut pos: usize) -> Vec<usize> {
    let mut out = vec![first];
    let mut next = first;
    for _ in 1..N_CONT {
        d.step(pos, next);
        pos += 1;
        next = d.last_argmax();
        out.push(next);
    }
    out
}

#[test]
#[ignore = "whole-model GPU run with a CPU reference: release, `--ignored`"]
fn test_gpu_prefill_match_q8() {
    arms("test_gpu_prefill_match_q8", Carrier::Q8);
}

#[test]
#[ignore = "whole-model GPU run with a CPU reference: release, `--ignored`"]
fn test_gpu_prefill_match_q4() {
    arms("test_gpu_prefill_match_q4", Carrier::Q4);
}

/// Runs both coopmat arms of one carrier. `Vk` has no `Drop`, so every load leaves a live device
/// until process exit, and a process holding four whole-model devices crashes (SIGSEGV) or hangs in
/// the driver's exit teardown about half the time (7 of 15 runs; 0 of 8 with the old two devices,
/// 0 of 7 with one arm per process). The off arm therefore re-runs this test in a child process
/// (`PM_ARM=off`), so no process holds more than two devices. `PM_ARM=on|off` selects one arm.
fn arms(name: &str, carrier: Carrier) {
    match std::env::var("PM_ARM").as_deref() {
        Ok("on") => run(carrier, true),
        Ok("off") => run(carrier, false),
        _ => {
            run(carrier, true);
            let status = std::process::Command::new(std::env::current_exe().expect("current_exe"))
                .args([
                    name,
                    "--exact",
                    "--ignored",
                    "--test-threads=1",
                    "--nocapture",
                ])
                .env("PM_ARM", "off")
                .status()
                .expect("spawn the off-arm child");
            assert!(status.success(), "off arm child failed: {status}");
        }
    }
}

static PROBED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Restores `APEIRON_COOPMAT` when an arm unwinds, so one arm's override cannot leak into the next
/// test in the same process (the harness runs the two carrier tests serially under
/// `--test-threads=1`).
struct CoopmatEnv(Option<String>);

impl Drop for CoopmatEnv {
    fn drop(&mut self) {
        unsafe {
            match &self.0 {
                Some(v) => std::env::set_var("APEIRON_COOPMAT", v),
                None => std::env::remove_var("APEIRON_COOPMAT"),
            }
        }
    }
}

/// `coopmat` selects the arm: `Vk::load` reads `APEIRON_COOPMAT` to decide `has_coopmat`, so each
/// arm takes its own device load. The `false` arm (`APEIRON_COOPMAT=0`) forces the DP4A floor and
/// must reach the same first token; a device without the tensor-core extension runs both arms on
/// DP4A.
fn run(carrier: Carrier, coopmat: bool) {
    let _env = CoopmatEnv(std::env::var("APEIRON_COOPMAT").ok());
    unsafe {
        std::env::set_var("APEIRON_COOPMAT", if coopmat { "1" } else { "0" });
    }
    println!(
        "== carrier {carrier:?}, coopmat arm {}",
        if coopmat {
            "on"
        } else {
            "off (APEIRON_COOPMAT=0)"
        }
    );
    // `Vk` has no `Drop`, so every load leaves a live instance until process exit and the driver's
    // teardown of many instances crashes or hangs intermittently; the probe runs once per process.
    if !PROBED.swap(true, std::sync::atomic::Ordering::Relaxed) && unsafe { Vk::load() }.is_err() {
        println!("SKIP: No Vulkan device present");
        return;
    }
    let (Some(model), Some(tokenizer)) = (model_path(MODEL), model_path(TOKENIZER)) else {
        println!("SKIP: model or tokenizer not found");
        return;
    };
    let gguf = Gguf::open(&model).expect("gguf open");
    let engine = Engine::open(&model).expect("engine open");
    let bpe = Bpe::from_tokenizer_json(&tokenizer).expect("tokenizer");
    let mut gpu = GpuDecode::new_with_prefill(&gguf, &engine.config, TMAX, carrier, MAXB)
        .expect("GpuDecode::new_with_prefill");

    let mut worst_rel = 0f64;
    let mut results = Vec::new();
    let mut cont_total = 0usize;
    for prompt in PROMPTS {
        let ids = bpe.encode(prompt).expect("encode");
        let toks: Vec<usize> = ids.iter().map(|&t| t as usize).collect();
        let n = toks.len();
        if prompt.len() > 400 {
            assert!(n >= 130, "long prompt must span >= 3 key slabs, got {n}");
        }

        // token-by-token GPU path
        for (pos, &t) in toks.iter().enumerate() {
            gpu.step(pos, t);
        }
        let tbt_first = gpu.last_argmax();
        let tbt_logits: Vec<f32> = gpu.step_logits().to_vec();
        let tbt_hidden: Vec<f64> = gpu.last_hidden().to_vec();
        let tbt_cont = continue_greedy(&mut gpu, tbt_first, n);

        // whole-prompt GPU prefill, then decode on from its KV cache
        if let Ok(prof) = gpu.prefill_profile(&toks) {
            let attn_nodes: Vec<_> = prof
                .iter()
                .filter(|(name, _)| name.contains("attn_"))
                .collect();
            let quant_nodes: Vec<_> = prof
                .iter()
                .filter(|(name, _)| name.starts_with("quant_"))
                .collect();
            println!(
                "  [PLAN] prompt len {n}: attn nodes: {attn_nodes:?}, quant nodes: {quant_nodes:?}"
            );
        }
        let logits_head: Vec<f32> = gpu.prefill(&toks).expect("prefill").to_vec();
        let pf_first = gpu.last_argmax();
        let pf_hidden: Vec<f64> = gpu.last_hidden().to_vec();
        let lg_ref = tbt_logits
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt();
        let lg_d = tbt_logits
            .iter()
            .zip(&logits_head)
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        println!(
            "  logits rel L2 err (last pos, prefill vs tbt) {:.3e}",
            lg_d / lg_ref
        );
        let host_arg = logits_head
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                if v > b.1 {
                    (i, v)
                } else {
                    b
                }
            })
            .0;
        let pf_cont = continue_greedy(&mut gpu, pf_first, n);

        // causality / stale-state control: a longer unrelated prompt leaves its K/V at positions
        // `n..` of the prefill cache; re-running the same prompt must reproduce the first run
        // bit for bit (a mask that lets position i read key i+1 reads that stale data instead)
        let poison: Vec<usize> = (0..48).map(|i| 1000 + 17 * i).collect();
        gpu.prefill(&poison).expect("poison prefill");
        let logits_again: Vec<f32> = gpu.prefill(&toks).expect("prefill again").to_vec();
        let hidden_again: Vec<f64> = gpu.last_hidden().to_vec();
        let repeat_exact = logits_again == logits_head && hidden_again == pf_hidden;

        // the long prompt is checked against token-by-token and for stale state only: the
        // token-serial CPU reference over it costs more than the gate budget
        let long = prompt.len() > 400;
        let mut st = engine.state(n + 1).expect("cpu state");
        let mut st2 = engine.state(n + 1).expect("cpu state");
        let mut cl = Vec::new();
        if !long {
            // CPU logits at the last prompt position (token-serial exact engine) as the accuracy reference
            for &t in &ids {
                cl = engine
                    .step_logits_state(&mut st2, t, KernelMode::Exact)
                    .expect("cpu step");
            }
        }
        let cpu_logits: Vec<f64> = if long {
            tbt_logits.iter().map(|&v| v as f64).collect()
        } else {
            cl.iter().map(|&v| v as f64 / 4294967296.0).collect()
        };
        let l2f = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
                / b.iter().map(|y| y * y).sum::<f64>().sqrt()
        };
        let pf_l: Vec<f64> = logits_head.iter().map(|&v| v as f64).collect();
        let tb_l: Vec<f64> = tbt_logits.iter().map(|&v| v as f64).collect();
        println!(
            "  logits rel L2 vs CPU exact: prefill {:.3e}, token-by-token {:.3e}",
            l2f(&pf_l, &cpu_logits),
            l2f(&tb_l, &cpu_logits)
        );
        let cpu_first = if long {
            tbt_first
        } else {
            engine
                .prefill_state(&mut st, &ids, KernelMode::Exact)
                .expect("cpu prefill") as usize
        };
        // SEL-TIE-PF: the tie admission inputs, computed where `cpu_logits` and `tb_l` are live.
        let tie_gap = cpu_logits[cpu_first] - cpu_logits[tbt_first];
        let tie_rel_l2 = l2f(&tb_l, &cpu_logits);

        let refmax = tbt_hidden.iter().fold(0f64, |m, v| m.max(v.abs()));
        let dmax = tbt_hidden
            .iter()
            .zip(&pf_hidden)
            .fold(0f64, |m, (a, b)| m.max((a - b).abs()));
        let rel = dmax / refmax;
        let l2 = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let diff: Vec<f64> = tbt_hidden
            .iter()
            .zip(&pf_hidden)
            .map(|(a, b)| a - b)
            .collect();
        let rel_l2 = l2(&diff) / l2(&tbt_hidden);

        // single-token control: attention over one position is trivial, so this isolates the
        // projection / norm / FFN stack from the attention and KV re-layout
        gpu.step(0, toks[0]);
        let h1: Vec<f64> = gpu.last_hidden().to_vec();
        gpu.prefill(&toks[..1]).expect("prefill n=1");
        let d1: Vec<f64> = h1
            .iter()
            .zip(gpu.last_hidden())
            .map(|(a, b)| a - b)
            .collect();
        let rel_l2_n1 = l2(&d1) / l2(&h1);
        worst_rel = worst_rel.max(rel);
        let cont_agree = tbt_cont
            .iter()
            .zip(&pf_cont)
            .take_while(|(a, b)| a == b)
            .count();
        println!(
            "prompt {prompt:?} n={n}\n  next token: prefill={pf_first} token-by-token={tbt_first} cpu={cpu_first} host-scan={host_arg}\n  hidden(last pos) max|d|={dmax:.3e} max|ref|={refmax:.3e} max rel err={rel:.3e}, rel L2 err={rel_l2:.3e}; n=1 rel L2 err={rel_l2_n1:.3e}\n  continuation agreement {cont_agree}/{N_CONT}: prefill {pf_cont:?} vs tbt {tbt_cont:?}"
        );
        println!("  repeat after a longer unrelated prefill bit-identical: {repeat_exact}");
        cont_total += cont_agree;
        results.push((
            prompt,
            pf_first,
            tbt_first,
            cpu_first,
            host_arg,
            repeat_exact,
            tie_gap,
            tie_rel_l2,
            long,
        ));
    }
    // The continuation decodes from the decode-layout KV (`kcq`, `sk`, `vdec`) the rope node
    // writes; a wrong store there leaves the first token intact and collapses this sum. Floors are
    // the measured totals of the committed kernels (bit-identical across runs and across the
    // `kv_seed` fusion), so a drop is a regression in the decode-layout KV. The Q4 coopmat floor is
    // re-pinned 43 -> 29 for the down GEMM's split-K (`DEC_DOWN_SPLITK=3`, f32 sum order
    // (p0+p1)+p2; first tokens agree on every prompt, PF PPL moves -0.019); `DEC_DOWN_SPLITK=1`
    // restores the serial order, where the old floor of 43 held. With carrier selection on (Q5/Q4
    // per-layer carriers) the f32 sum order changes: first tokens and the 512-prompt population
    // hold (SEL-ACCEPT: paired rel L2 delta -8.53e-3), and the Q4 coopmat total re-pins 29 -> 26.
    println!("carrier {carrier:?} coopmat {coopmat}: continuation total {cont_total}/{N_CONT} per prompt");
    let floor = match (carrier, coopmat) {
        (Carrier::Q4, true) => 26,
        (Carrier::Q4, false) => 36,
        (Carrier::Q8, true) => 16,
        (Carrier::Q8, false) => 26,
        _ => 0,
    };
    assert!(
        cont_total >= floor,
        "continuation agreement {cont_total} below the pinned floor {floor} for {carrier:?} (coopmat {coopmat})"
    );
    println!("carrier {carrier:?}: worst hidden max rel err {worst_rel:.3e}");
    let mut n_admit = 0usize;
    for (prompt, pf, tbt, cpu, host, repeat_exact, gap, rel_l2, long) in results {
        assert!(
            repeat_exact,
            "prefill depends on stale cache state for {prompt:?} (causal mask)"
        );
        assert_eq!(
            pf, host,
            "device argmax != host scan of the prefill logits for {prompt:?}"
        );
        assert_eq!(pf, cpu, "prefill vs CPU first token for {prompt:?}");
        if pf != tbt {
            // SEL-TIE-PF: a prefill/token-by-token first-token mismatch is admitted only when the
            // prefill agrees exactly with the CPU reference, the prompt is not `long` (its
            // `cpu_logits` is a copy of the token-by-token row), the CPU-exact logit gap is within
            // `TIE_GAP` and the token-by-token-vs-CPU relative L2 is within `TIE_REL_L2`.
            let admit = pf == cpu && !long && gap <= TIE_GAP && rel_l2 <= TIE_REL_L2;
            assert!(
                admit,
                "prefill vs token-by-token first token for {prompt:?} (gap {gap:.4}, relL2 {rel_l2:.4e})"
            );
            println!("tie admitted: {prompt:?} pf {pf} tbt {tbt} gap {gap:.4} relL2 {rel_l2:.4e}");
            n_admit += 1;
        }
    }
    assert!(
        n_admit <= TIE_MAX,
        "{n_admit} prompts admitted by the tie contract, limit {TIE_MAX}"
    );
    println!("tie contract: {n_admit} prompt(s) admitted (limit {TIE_MAX})");
}

unsafe fn readback_u32(vk: &Vk, src: Handle, n: usize) -> Vec<u32> {
    let (rb, mem) = vk
        .buffer(
            n as u64 * 4,
            BUFFER_USAGE_TRANSFER_DST,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )
        .expect("readback buffer");
    let cb = vk.command_buffer();
    vk.begin(cb);
    let region = BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: n as u64 * 4,
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
    std::slice::from_raw_parts(vk.map(mem).cast::<u32>(), n).to_vec()
}

unsafe fn readback_f32(vk: &Vk, src: Handle, n: usize) -> Vec<f32> {
    let (rb, mem) = vk
        .buffer(
            n as u64 * 4,
            BUFFER_USAGE_TRANSFER_DST,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )
        .expect("readback buffer");
    let cb = vk.command_buffer();
    vk.begin(cb);
    let region = BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: n as u64 * 4,
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
    std::slice::from_raw_parts(vk.map(mem).cast::<f32>(), n).to_vec()
}

unsafe fn run_step(vk: &Vk, s: &apeiron::gpu::decode::Step) {
    let cb = vk.command_buffer();
    vk.begin(cb);
    rec_step(vk, cb, s);
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
}

/// Asserts that the coopmat epilogue with DPW=2 produces bit-identical codes, scales, and sums
/// to quantize_batch_module(n, 2) on the same attention output.
#[test]
#[ignore = "GPU test comparing coopmat DPW=2 chart against quantize_batch_module"]
fn test_coopmat_dpw2_chart_matches_quantize() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device: {e}");
            return;
        }
    };
    if !vk.has_coopmat {
        println!("SKIP: device lacks cooperative matrix");
        return;
    }
    let (g, _stage_mem) = GpuCtx::new(&vk).expect("GpuCtx");

    const NH: u32 = 14;
    const NKV: u32 = 2;
    const HD: u32 = 64;
    const QKVM: u32 = 1152;
    const TMAX: u32 = 64;
    const SCALE: f32 = 0.125;
    const QN: u32 = NH * HD; // 896

    // Setup input buffers for attention: 1 query token (pos0=0, nn=1)
    let params_data = vec![0u32, 1u32];
    let b_params = unsafe { g.up_u32(&params_data) };

    // Query buffer: QKVM floats for row 0.
    // Query 0 of each head set to 1.0, rest 0.0
    let mut qkv = vec![0.0f32; QKVM as usize];
    for h in 0..NH as usize {
        qkv[h * HD as usize] = 1.0;
    }
    let b_qkv = unsafe { g.up_f32(&qkv) };

    // Key code buffer: (NKV * HD * TMAX) floats.
    // Key 0 of dim 0 set to 1.0, rest 0.0
    let mut kc = vec![0.0f32; (NKV * HD * TMAX) as usize];
    for kvh in 0..NKV as usize {
        kc[kvh * HD as usize * TMAX as usize] = 1.0;
    }
    let b_kc = unsafe { g.up_f32(&kc) };

    // Key scale buffer: TMAX * NKV floats
    let mut ks = vec![0.0f32; (TMAX * NKV) as usize];
    for kvh in 0..NKV as usize {
        ks[kvh] = 1.0 / SCALE;
    }
    let b_ks = unsafe { g.up_f32(&ks) };

    // Value scale buffer: TMAX * NKV floats
    let mut vs = vec![0.0f32; (TMAX * NKV) as usize];
    for kvh in 0..NKV as usize {
        vs[kvh] = 1.0;
    }
    let b_vs = unsafe { g.up_f32(&vs) };

    // Value code buffer: (NKV * TMAX * HD) floats.
    let mut vc = vec![0.0f32; (NKV * TMAX * HD) as usize];
    let mut expected_src = vec![0.0f32; QN as usize];
    for h in 0..NH as usize {
        let kvh = h / (NH as usize / NKV as usize);
        for d in 0..HD as usize {
            let val = (d as f32 - 32.0) * 0.5;
            vc[(kvh * TMAX as usize + 0) * HD as usize + d] = val;
            expected_src[h * HD as usize + d] = val;
        }
    }
    let b_vc = unsafe { g.up_f32(&vc) };

    // Outputs for coopmat epilogue:
    let b_xq_coop = unsafe { g.dev_zeroed((QN as usize / 4) * 4) };
    let b_xsc_coop = unsafe { g.dev_zeroed((QN as usize / 32) * 4) };
    let b_xsm_coop = unsafe { g.dev_zeroed((QN as usize / 32) * 4) };

    let coop_spv = attn_prefill_coopmat_bytes(NH, NKV, QKVM, TMAX, SCALE, 2);
    let coop_step = unsafe {
        make_step_named(
            &g,
            &coop_spv,
            &[
                b_qkv, b_kc, b_ks, b_vc, b_vs, b_xq_coop, b_params, b_xsc_coop, b_xsm_coop,
            ],
            NH,
            1,
            "attn_coopmat_dpw2",
        )
    };
    unsafe {
        run_step(&vk, &coop_step);
    }

    // Run quantize_batch_module(QN, 2) on the same attention output expected_src
    let b_src = unsafe { g.up_f32(&expected_src) };
    let b_xq_quant = unsafe { g.dev_zeroed((QN as usize / 4) * 4) };
    let b_xsc_quant = unsafe { g.dev_zeroed((QN as usize / 32) * 4) };
    let b_xsm_quant = unsafe { g.dev_zeroed((QN as usize / 32) * 4) };

    let quant_spv = quantize_batch_bytes(QN, 2);
    let quant_step = unsafe {
        make_step_named(
            &g,
            &quant_spv,
            &[b_src, b_xq_quant, b_xsc_quant, b_xsm_quant],
            1,
            1,
            "quantize_batch_dpw2",
        )
    };
    unsafe {
        run_step(&vk, &quant_step);
    }

    // Readback and assert exact bit-identical match
    let xq_coop = unsafe { readback_u32(&vk, b_xq_coop, QN as usize / 4) };
    let xq_quant = unsafe { readback_u32(&vk, b_xq_quant, QN as usize / 4) };
    let xsc_coop = unsafe { readback_f32(&vk, b_xsc_coop, QN as usize / 32) };
    let xsc_quant = unsafe { readback_f32(&vk, b_xsc_quant, QN as usize / 32) };
    let xsm_coop = unsafe { readback_f32(&vk, b_xsm_coop, QN as usize / 32) };
    let xsm_quant = unsafe { readback_f32(&vk, b_xsm_quant, QN as usize / 32) };

    assert_eq!(
        xq_coop, xq_quant,
        "codes mismatch between coopmat dpw=2 and quantize_batch(n, 2)"
    );
    assert_eq!(
        xsc_coop, xsc_quant,
        "scales mismatch between coopmat dpw=2 and quantize_batch(n, 2)"
    );
    assert_eq!(
        xsm_coop, xsm_quant,
        "sums mismatch between coopmat dpw=2 and quantize_batch(n, 2)"
    );
    println!(
        "OK: test_coopmat_dpw2_chart_matches_quantize bit-identical on {} words, {} scales, {} sums",
        xq_coop.len(),
        xsc_coop.len(),
        xsm_coop.len()
    );
}
