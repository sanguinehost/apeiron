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
//!   cargo test --release --locked --features gpu --test gpu_prefill_match -- --ignored --nocapture
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
use apeiron::gpu::vulkan::Vk;
use apeiron::simt::Carrier;

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
        ));
    }
    // The continuation decodes from the decode-layout KV (`kcq`, `sk`, `vdec`) the rope node
    // writes; a wrong store there leaves the first token intact and collapses this sum. Floors are
    // the measured totals of the committed kernels (bit-identical across runs and across the
    // `kv_seed` fusion), so a drop is a regression in the decode-layout KV. The Q4 coopmat floor is
    // re-pinned 43 -> 29 for the down GEMM's split-K (`DEC_DOWN_SPLITK=3`, f32 sum order
    // (p0+p1)+p2; first tokens agree on every prompt, PF PPL moves -0.019); `DEC_DOWN_SPLITK=1`
    // restores the serial order, where the old floor of 43 held.
    let floor = match (carrier, coopmat) {
        (Carrier::Q4, true) => 29,
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
    for (prompt, pf, tbt, cpu, host, repeat_exact) in results {
        assert!(
            repeat_exact,
            "prefill depends on stale cache state for {prompt:?} (causal mask)"
        );
        assert_eq!(
            pf, host,
            "device argmax != host scan of the prefill logits for {prompt:?}"
        );
        assert_eq!(
            pf, tbt,
            "prefill vs token-by-token first token for {prompt:?}"
        );
        assert_eq!(pf, cpu, "prefill vs CPU first token for {prompt:?}");
    }
}
