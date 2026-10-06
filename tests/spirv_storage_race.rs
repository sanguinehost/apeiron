//! Detector for the cross-invocation storage-buffer race in the `simt_nn` SPIR-V emitter.
//!
//! The modules use the GLSL450 memory model. A workgroup barrier whose `MemorySemantics` carry
//! only `WorkgroupMemory` (0x100) does **not** order one invocation's store to a storage buffer
//! (SSBO) against another invocation's load. In GLSL450 a storage-buffer location is coherent
//! for another invocation only when the variable is decorated `Coherent` and the barrier's
//! semantics include `UniformMemory` (0x40). Absent both, a reader may observe the pre-barrier
//! value: the landed q4 prefill-perplexity blow-up.
//!
//! This walks the SPIR-V words of every module the `simt_nn` emitter builds and flags, per
//! function, a storage-class variable `V` with an `OpStore` (directly or through an
//! `OpAccessChain`) followed in program order by an `OpControlBarrier` and then an `OpLoad` from
//! `V`, unless `V` is `Coherent` and that barrier carries `UniformMemory`.
//!
//! Negative control: run this file against `git show HEAD:src/simt_nn/batch.rs` /
//! `mod.rs` in a throwaway copy — it must flag `quantize_at` and the `RoPE` exchange. The
//! `detector_flags_the_known_race` unit control below fires the same analyser on a hand-built
//! module, so the detector cannot pass by finding nothing.

use apeiron::simt::Carrier;
use apeiron::simt_nn::*;
use std::collections::{HashMap, HashSet};

const SC_UNIFORM: u32 = 2;
const SC_STORAGE_BUFFER: u32 = 12;
const DEC_BLOCK: u32 = 2;
const DEC_COHERENT: u32 = 23;
const DEC_BINDING: u32 = 33;
const OP_DECORATE: u32 = 71;
const OP_CONSTANT: u32 = 43;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONTROL_BARRIER: u32 = 224;
const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_END: u32 = 56;

const SEM_UNIFORM_MEMORY: u32 = 0x40;

#[derive(Debug, Clone)]
struct Violation {
    module: &'static str,
    var: u32,
    binding: Option<u32>,
    store_off: usize,
    barrier_off: usize,
    load_off: usize,
    coherent: bool,
    barrier_sem: u32,
    store_idx: Option<u32>,
    load_idx: Option<u32>,
}

/// The root variable an `OpAccessChain`/copy chain resolves to.
fn root_var(mut id: u32, chain: &HashMap<u32, u32>) -> u32 {
    for _ in 0..64 {
        match chain.get(&id) {
            Some(&b) => id = b,
            None => break,
        }
    }
    id
}

/// Every storage-buffer race in one module's instruction stream.
#[allow(clippy::too_many_lines)]
fn analyze(module: &'static str, w: &[u32]) -> Vec<Violation> {
    let mut binding: HashMap<u32, u32> = HashMap::new();
    let mut coherent: HashSet<u32> = HashSet::new();
    let mut bufblock: HashSet<u32> = HashSet::new();
    let mut var_sc: HashMap<u32, u32> = HashMap::new();
    let mut var_ty: HashMap<u32, u32> = HashMap::new();
    let mut ptr_pointee: HashMap<u32, u32> = HashMap::new();
    let mut chain: HashMap<u32, u32> = HashMap::new();
    let mut chain_idx: HashMap<u32, u32> = HashMap::new();
    let mut const_val: HashMap<u32, u32> = HashMap::new();

    let mut i = 5usize;
    while i < w.len() {
        let wc = (w[i] >> 16) as usize;
        assert!(
            wc >= 1 && i + wc <= w.len(),
            "{module}: stream misaligned at {i}"
        );
        let op = w[i] & 0xFFFF;
        let ops = &w[i + 1..i + wc];
        match op {
            OP_DECORATE => {
                let (t, d) = (ops[0], ops[1]);
                match d {
                    DEC_COHERENT => {
                        coherent.insert(t);
                    }
                    DEC_BINDING => {
                        binding.insert(t, ops[2]);
                    }
                    DEC_BLOCK => {
                        bufblock.insert(t);
                    }
                    _ => {}
                }
            }
            OP_VARIABLE => {
                var_ty.insert(ops[1], ops[0]);
                var_sc.insert(ops[1], ops[2]);
            }
            OP_TYPE_POINTER => {
                ptr_pointee.insert(ops[0], ops[2]);
            }
            OP_ACCESS_CHAIN => {
                chain.insert(ops[1], ops[2]);
                if ops.len() > 3 {
                    chain_idx.insert(ops[1], ops[ops.len() - 1]);
                }
            }
            OP_CONSTANT => {
                const_val.insert(ops[1], ops[2]);
            }
            _ => {}
        }
        i += wc;
    }

    let is_storage = |v: u32| -> bool {
        match var_sc.get(&v).copied() {
            Some(SC_STORAGE_BUFFER) => true,
            Some(SC_UNIFORM) => var_ty
                .get(&v)
                .and_then(|t| ptr_pointee.get(t))
                .is_some_and(|p| bufblock.contains(p)),
            _ => false,
        }
    };

    // Per-function event streams (a store in one function and a load in another is not this race).
    let mut violations = Vec::new();
    let mut fstore: Vec<(u32, usize, u32)> = Vec::new(); // (root var, word offset, ptr)
    let mut fload: Vec<(u32, usize, u32)> = Vec::new();
    let mut fbar: Vec<(usize, u32)> = Vec::new(); // (word offset, semantics)
    let mut in_fn = false;

    let flush = |name: &'static str,
                 fstore: &mut Vec<(u32, usize, u32)>,
                 fload: &mut Vec<(u32, usize, u32)>,
                 fbar: &mut Vec<(usize, u32)>,
                 out: &mut Vec<Violation>| {
        for &(v, l, lp) in fload.iter() {
            // nearest barrier before this load
            let bar = fbar
                .iter()
                .filter(|&&(b, _)| b < l)
                .max_by_key(|&&(b, _)| b);
            let Some(&(b, sem)) = bar else { continue };
            // latest store to v strictly before that barrier
            let st = fstore
                .iter()
                .filter(|&&(sv, s, _)| sv == v && s < b)
                .max_by_key(|&&(_, s, _)| s);
            if let Some(&(_, s, sp)) = st {
                out.push(Violation {
                    module: name,
                    var: v,
                    binding: binding.get(&v).copied(),
                    store_off: s,
                    barrier_off: b,
                    load_off: l,
                    coherent: coherent.contains(&v),
                    barrier_sem: sem,
                    store_idx: chain_idx.get(&sp).copied(),
                    load_idx: chain_idx.get(&lp).copied(),
                });
            }
        }
        fstore.clear();
        fload.clear();
        fbar.clear();
    };

    let mut i = 5usize;
    while i < w.len() {
        let wc = (w[i] >> 16) as usize;
        let op = w[i] & 0xFFFF;
        let ops = &w[i + 1..i + wc];
        match op {
            OP_FUNCTION => in_fn = true,
            OP_FUNCTION_END if in_fn => {
                flush(module, &mut fstore, &mut fload, &mut fbar, &mut violations);
                in_fn = false;
            }
            OP_STORE if in_fn => {
                let v = root_var(ops[0], &chain);
                if is_storage(v) {
                    fstore.push((v, i, ops[0]));
                }
            }
            OP_LOAD if in_fn => {
                let v = root_var(ops[2], &chain);
                if is_storage(v) {
                    fload.push((v, i, ops[2]));
                }
            }
            OP_CONTROL_BARRIER if in_fn => {
                // the semantics operand is a constant id; resolve it before testing 0x40
                let sem = const_val.get(&ops[2]).copied().unwrap_or(ops[2]);
                fbar.push((i, sem));
            }
            _ => {}
        }
        i += wc;
    }
    violations
}

/// A violation is a violation unless the variable is `Coherent` and the separating barrier
/// carries `UniformMemory` in its semantics.
fn is_violation(v: &Violation) -> bool {
    !(v.coherent && (v.barrier_sem & SEM_UNIFORM_MEMORY) != 0)
}

/// Every module the `simt_nn` emitter builds on the decode, prefill/batch and coopmat paths.
#[allow(clippy::too_many_lines)]
fn modules() -> Vec<(&'static str, Vec<u32>)> {
    vec![
        ("embed", embed_f16_module(896, 151_936, 256)),
        (
            "fusedropekv",
            fused_rope_kv_module(14, 2, 64, 2048, 1e6, 0, 64, false),
        ),
        (
            "flashpartial",
            flash_partial_module(14, 2, 64, 2048, 0.125, 8, 64, false, false, false, 0),
        ),
        (
            "thetascan",
            theta_scan_module(14, 2, 64, 2048, 0.125, 64, 8),
        ),
        (
            "thetaselect",
            theta_select_module(14, 2048, 64, 8, 1, 2, 16.635, true, 0, 0),
        ),
        ("flashcombine", flash_combine_module(14, 64, 8)),
        ("silu", silu_mul_module(4864, 256, 4864)),
        ("add", add_inplace_module(896, 256)),
        ("qknorm", qk_norm_module(14, 128, 1e-6, 0)),
        ("rmsnorm", rmsnorm_module(896, 1e-6, 2, None, true)),
        ("rmsaddqseg", rmsnorm_module(896, 1e-6, 2, Some(8), true)),
        ("rmsfused", rmsnorm_module(896, 1e-6, 0, None, false)),
        ("quantvec", quantize_vec_module(896, 2)),
        ("siluq", silu_mul_q_module(4864, 4864, 2)),
        (
            "embchart",
            embed_chart_module(896, 151_936, 256, Carrier::Q4),
        ),
        ("amaxp", argmax_partial_module(151_936, 256, 256)),
        ("amaxf", argmax_finish_module(256)),
        (
            "lastrowgather",
            last_row_gather_module(224, 28, 0xf_ffff, 0x1_ffff),
        ),
        ("rmsqsb", rmsnorm_batch_qsb_module(896, 1e-6, true)),
        ("rmsqsb0", rmsnorm_batch_qsb_module(896, 1e-6, false)),
        ("quantsb", quantize_batch_sb_module(4864)),
        (
            "lastrowgathersb",
            last_row_gather_sb_module(224, 4, 0xf_ffff, 0x1_ffff),
        ),
        // the batch family the prefill path runs
        ("quantize_batch", quantize_batch_module(896, 2)),
        (
            "rmsnorm_batch_q",
            rmsnorm_batch_q_module(896, 1e-6, 2, false),
        ),
        (
            "rmsnorm_batch_q_res",
            rmsnorm_batch_q_module(896, 1e-6, 2, true),
        ),
        ("swiglu_batch", swiglu_batch_module(4864)),
        (
            "qkv_rope_batch",
            qkv_rope_batch_module(14, 2, 64, 1152, 2048, 1e6, 1e-6, false, false),
        ),
        (
            "qkv_rope_batch_bias",
            qkv_rope_batch_module(14, 2, 64, 1152, 2048, 1e6, 1e-6, true, false),
        ),
        (
            "qkv_rope_batch_qkn",
            qkv_rope_batch_module(14, 2, 64, 1152, 2048, 1e6, 1e-6, false, true),
        ),
        // prefill attention + glue
        (
            "attn_prefill",
            attn_prefill_module(14, 2, 64, 1152, 2048, 0.125),
        ),
        (
            "attn_prefill_tiled",
            attn_prefill_tiled_module(14, 2, 64, 1152, 2048, 0.125, 128, 4, 4, false),
        ),
        (
            "embed_chart_batch",
            embed_chart_batch_module(896, 151_936, 256, Carrier::Q4),
        ),
        ("kv_seed", kv_seed_module(2, 64, 2048)),
        // norm/gemv fusion (decode) + coopmat
        (
            "gemv_norm",
            gemv_norm_module(256, 256, 64, 1, Carrier::Q8, 1e-6, None),
        ),
        (
            "gemv_actq",
            gemv_actq_module(256, 256, 1, Carrier::Q8, ActQ::Copy),
        ),
        (
            "gemv_actq_silu",
            gemv_actq_module(256, 256, 1, Carrier::Q8, ActQ::SiluMul(256)),
        ),
        (
            "gemv_dp4a_norm",
            gemv_dp4a_norm_module(4096, 14336, 256, 8, Carrier::Q8, 1e-6, None),
        ),
        ("gemv_coop", gemv_coop_module(256, 256, 32, 8, Carrier::Q8)),
        // MoE / pool / gridsync
        ("moe_topk", moe_topk_module(64, 4, true, true, 1.0)),
        ("moe_silu_slot", silu_mul_q_slot_module(4864, 2)),
        ("pool_fill", pool_fill_module(2, 64, 2048, 16, 4, 128)),
        ("gridsync", gridsync_module(4, 8, 256, 256)),
        ("nulldispatch", nulldispatch_module(256, 256)),
        // deltanet + gated attention
        ("dnrule", deltanet_rule_module(128, 48, 16, (0, 0, 0))),
        ("dnconv", deltanet_conv_silu_module(10240, 4)),
        ("dnl2", deltanet_l2_heads_module(16, 128, 1e-6, 0)),
        ("dngk", deltanet_gk_module(48)),
        ("dngate", deltanet_gate_norm_module(128, 48, 1e-6)),
        ("attnsplit", attn_qgate_split_module(8, 2, 256)),
        ("attngate", attn_gate_apply_module(2048)),
        ("copyvec", copy_vec_module(1024, 0)),
    ]
}

/// Per-invocation FALSE POSITIVES. The coarse rule (store to V … barrier … load from V) cannot see
/// addresses, so a kernel that writes a storage vector in one strided loop and re-reads the SAME
/// lane's element later is flagged even though no other invocation's value is observed. Each name
/// below is proved per-invocation in the comment and in the kernel source:
///
/// * `fusedropekv` (`src/simt_nn/glue.rs::fused_rope_kv_fold_module`) is no longer here: the
///   rope-leader hand-off now crosses invocations through workgroup memory (`red`/`sval`), as
///   `src/simt_nn/batch.rs::qkv_rope_batch_module` does.
/// * `rmsqsb`, `rmsnorm_batch_q_res` (`src/simt_nn/batch.rs`): the `residual` prologue folds
///   `x[idx] += res[idx]` in `loop_step(tid, n, WG)` at `idx = i·n + j`, and the later sumsq/scale
///   loops read `x` at the identical `idx` with the identical `tid`/`WG` stride. Same lane, same
///   element; the separating barriers order the shared reduction, not this read.
/// * `rmsaddqseg` (`src/simt_nn/norm.rs::rmsnorm_module` with `residual = Some(8)`): the same
///   in-place residual fold, `store_f(x, i, s)` then `rmsnorm_wg`'s strided `load_f(x, i)` with the
///   caller's single `WG`; `i` is the same lane's index in both.
const ALLOW: &[&str] = &["rmsqsb", "rmsnorm_batch_q_res", "rmsaddqseg"];

#[test]
fn no_unexpected_cross_invocation_storage_race() {
    let mut bad = Vec::new();
    let mut allow_hits = 0usize;
    let mut allow_matched: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for (name, w) in modules() {
        for v in analyze(name, &w) {
            if !is_violation(&v) {
                continue;
            }
            if ALLOW.contains(&name) {
                allow_hits += 1;
                allow_matched.insert(name);
            } else {
                bad.push(v);
            }
        }
    }
    for v in &bad {
        eprintln!(
            "RACE {module}: var %{var} binding {binding:?}: store@{store_off} barrier@{barrier_off} \
             (sem 0x{sem:x}) load@{load_off} coherent={coherent} store_idx={sidx:?} load_idx={lidx:?}",
            module = v.module,
            var = v.var,
            binding = v.binding,
            store_off = v.store_off,
            barrier_off = v.barrier_off,
            sem = v.barrier_sem,
            load_off = v.load_off,
            coherent = v.coherent,
            sidx = v.store_idx,
            lidx = v.load_idx,
        );
    }
    // The allow-list is load-bearing: a name that stops matching is a silent hole.
    for want in ALLOW {
        assert!(
            allow_matched.contains(want),
            "allow-list entry {want} matched no violation — it is stale and may hide a site"
        );
    }
    assert!(
        bad.is_empty(),
        "{} unexpected cross-invocation storage-buffer race(s) across {} modules",
        bad.len(),
        modules().len()
    );
    eprintln!("allow-list {allow_hits} per-invocation hits");
}

/// Unit control: the analyser fires on a hand-built store→barrier→load and is silenced only by
/// `Coherent` + `UniformMemory`.
#[test]
fn detector_flags_the_known_race() {
    const MAGIC: u32 = 0x0723_0203;
    // OpVariable: [type, id, storage class] -> word count 4.
    let var = [(4 << 16) | OP_VARIABLE, 1, 10, SC_STORAGE_BUFFER];
    let store = [(3 << 16) | OP_STORE, 10, 11];
    let load = [(4 << 16) | OP_LOAD, 1, 12, 10];
    let bar = |sem: u32| [(4 << 16) | OP_CONTROL_BARRIER, 2, 2, sem];
    let fn_begin = [(5 << 16) | OP_FUNCTION, 1, 50, 0, 2];
    let fn_end = [(1 << 16) | OP_FUNCTION_END];

    let mut race = vec![MAGIC, 0x0001_0300, 0, 100, 0];
    race.extend(fn_begin);
    race.extend(var);
    race.extend(store);
    race.extend(bar(0x108));
    race.extend(load);
    race.extend(fn_end);
    let v = analyze("synthetic", &race);
    assert_eq!(v.len(), 1, "the detector missed a plain store→barrier→load");
    assert!(is_violation(&v[0]));

    // Coherent + UniformMemory silences it. OpDecorate: [target, decoration] -> word count 3.
    let mut ok = vec![MAGIC, 0x0001_0300, 0, 100, 0];
    ok.extend(fn_begin);
    ok.extend([(3 << 16) | OP_DECORATE, 10, DEC_COHERENT]);
    ok.extend(var);
    ok.extend(store);
    ok.extend(bar(0x108 | SEM_UNIFORM_MEMORY));
    ok.extend(load);
    ok.extend(fn_end);
    let v = analyze("synthetic-ok", &ok);
    assert_eq!(v.len(), 1);
    assert!(
        !is_violation(&v[0]),
        "Coherent + UniformMemory must silence"
    );

    // No barrier between: same-invocation ordering, not the race.
    let mut local = vec![MAGIC, 0x0001_0300, 0, 100, 0];
    local.extend(fn_begin);
    local.extend(var);
    local.extend(store);
    local.extend(load);
    local.extend(fn_end);
    assert!(analyze("synthetic-local", &local).is_empty());
}
