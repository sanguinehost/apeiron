//! `tests/learn_fold_heldout.rs` — the held-out quality evaluator: identical models give zero KL and
//! TV and unit top-1 agreement, a perturbed `lm_head` row matches an independent `f64` softmax-KL
//! recomputation to `1e-6` relative, the mean over several sequences is the token-weighted mean of
//! the per-sequence values, and the named refusals hold. Grade `tested`.
//!
//! The fixture is a tiny one-layer `qwen2` [`Config`]/[`Weights`] built directly from the public
//! structs (the minimum of the `src/learn/grad/model/tests.rs` fixture pattern, which otherwise
//! serializes a GGUF and loads it through [`apeiron::engine::Engine`]).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "the test's fixed-seed LCG maps an integer word into a Q32.32 weight, and the f64 \
              softmax-KL reference is the independent recomputation the port is checked against; the \
              exact float comparisons are the point of the identical-model test (equal logits give 0 \
              and 1 bit-for-bit), which `assert_eq!` states directly; the pedantic lints stay on for \
              the crate"
)]

use apeiron::engine::{Engine, ONE};
use apeiron::learn::fold::run::npy::{chunks, read_npy_i4, split};
use apeiron::learn::fold::stats::heldout::{
    held_out, held_out_engines, held_out_engines_per_seq_timed, held_out_seq, EngineWalls, HeldOut,
};
use apeiron::learn::grad::model::{forward, Config, LayerWeights, Weights};

/// Embedding width, query heads, KV head width, feed-forward width, vocabulary, layers.
const E: usize = 4;
const NH: usize = 1;
const NKV: usize = 1;
const HD: usize = 2;
const QW: usize = NH * HD; // 2
const KVW: usize = NKV * HD; // 2
const FF: usize = 6;
const V: usize = 8;
const L: usize = 1;

/// A fixed-seed 64-bit LCG (Knuth's MMIX constants), the crate's test drift.
struct Lcg(u64);

impl Lcg {
    /// The next word.
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// A Q32.32 weight uniform on `[−0.5, 0.5)`, i.e. `[−ONE/2, ONE/2)`.
    fn weight(&mut self) -> i64 {
        let r = ((self.next() >> 33) & 0xffff) as i64; // 0..=65535
        (r - 32768) * (ONE >> 16)
    }

    /// `n` weights.
    fn vec(&mut self, n: usize) -> Vec<i64> {
        (0..n).map(|_| self.weight()).collect()
    }
}

/// The tiny model geometry; `eps` is `round(2^-12 · ONE)`.
fn config() -> Config {
    Config {
        n_layer: L,
        n_head: NH,
        n_head_kv: NKV,
        n_embd: E,
        n_ff: FF,
        vocab: V,
        head_dim: HD,
        eps: ONE >> 12,
        rope_theta: 10_000.0,
    }
}

/// A one-layer Q32.32 model with `seed`'s weights and unit norm gains.
fn weights(seed: u64) -> Weights {
    let mut g = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
    let one = vec![ONE; E];
    let layers = (0..L)
        .map(|_| LayerWeights {
            attn_norm: one.clone(),
            wq: g.vec(QW * E),
            bq: Some(g.vec(QW)),
            wk: g.vec(KVW * E),
            bk: Some(g.vec(KVW)),
            wv: g.vec(KVW * E),
            bv: Some(g.vec(KVW)),
            wo: g.vec(E * QW),
            ffn_norm: one.clone(),
            w_gate: g.vec(FF * E),
            w_up: g.vec(FF * E),
            w_down: g.vec(E * FF),
        })
        .collect();
    Weights {
        layers,
        output_norm: one,
        token_embd: g.vec(V * E),
        lm_head: g.vec(V * E),
    }
}

/// The `f64` softmax of one row of real logits.
fn softmax_f64(z: &[f64]) -> Vec<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|&x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|&x| x / s).collect()
}

/// The independent reference: the mean forward `KL(teacher ‖ student)` in nats from the Q32.32
/// logits, computed in `f64`, one row at a time (`Σ_x p_t(x)·(ln p_t(x) − ln p_s(x)) / T`).
fn f64_kl_mean(teacher: &Weights, student: &Weights, cfg: &Config, toks: &[u32]) -> f64 {
    let (lt, _) = forward(teacher, cfg, toks).expect("teacher forward");
    let (ls, _) = forward(student, cfg, toks).expect("student forward");
    let t = toks.len();
    let v = cfg.vocab;
    let one = ONE as f64;
    let mut total = 0.0;
    for r in 0..t {
        let zt: Vec<f64> = lt[r * v..(r + 1) * v]
            .iter()
            .map(|&z| z as f64 / one)
            .collect();
        let zs: Vec<f64> = ls[r * v..(r + 1) * v]
            .iter()
            .map(|&z| z as f64 / one)
            .collect();
        let pt = softmax_f64(&zt);
        let ps = softmax_f64(&zs);
        for x in 0..v {
            total += pt[x] * (pt[x].ln() - ps[x].ln());
        }
    }
    total / t as f64
}

/// A white-box check that the fixture's logits are non-degenerate: the two models differ, so the
/// KL tests below are not vacuous.
#[test]
fn fixture_is_non_degenerate() {
    let cfg = config();
    let teacher = weights(1);
    let student = weights(2);
    let toks: [u32; 4] = [1, 2, 3, 4];
    let h = held_out_seq(&teacher, &student, &cfg, &toks).expect("eval");
    assert!(h.kl > 0.0, "distinct models must give positive KL");
    assert!((0.0..=1.0).contains(&h.tv));
    assert!((0.0..=1.0).contains(&h.top1));
    assert!(h.nll_teacher > 0.0 && h.nll_student > 0.0);
}

/// Teacher equal to student: zero KL, zero TV, unit top-1, equal NLLs — exactly, not approximately.
#[test]
fn identical_teacher_student_is_zero() {
    let cfg = config();
    let w = weights(1);
    let s1: [u32; 3] = [1, 2, 3];
    let s2: [u32; 4] = [4, 5, 6, 7];
    let seqs: [&[u32]; 2] = [&s1, &s2];
    let h = held_out(&w, &w, &cfg, &seqs).expect("eval");
    assert_eq!(h.tokens, 7);
    assert_eq!(h.kl, 0.0, "equal logits give 0 exactly");
    assert_eq!(h.tv, 0.0);
    assert_eq!(h.top1, 1.0);
    assert_eq!(h.nll_student, h.nll_teacher);
}

/// A perturbed `lm_head` row gives a positive KL that equals the independent `f64` softmax-KL
/// recomputation to `1e-6` relative. The mutation control swaps `kl_row`'s arguments, which breaks
/// this equality: `KL(p_t ‖ p_s) ≠ KL(p_s ‖ p_t)`.
#[test]
fn perturbed_lm_head_matches_f64_softmax_kl() {
    let cfg = config();
    let teacher = weights(1);
    let mut student = teacher.clone();
    for k in 0..E {
        student.lm_head[k] += 3 * (ONE >> 2);
    }
    let toks: [u32; 5] = [1, 2, 3, 4, 5];
    let h = held_out_seq(&teacher, &student, &cfg, &toks).expect("eval");
    assert!(h.kl > 0.0, "the perturbed student must give positive KL");
    let want = f64_kl_mean(&teacher, &student, &cfg, &toks);
    let rel = (h.kl - want).abs() / want.abs().max(1e-12);
    assert!(
        rel <= 1e-6,
        "KL {:.12} vs f64 reference {:.12} (relative {rel:e})",
        h.kl,
        want
    );
}

/// The pooled mean over several sequences is the token-weighted mean of the per-sequence values:
/// KL/TV/top-1 weighted by `T_i`, the NLL by the `T_i − 1` next-token positions.
#[test]
fn multi_sequence_mean_is_token_weighted() {
    let cfg = config();
    let teacher = weights(1);
    let student = weights(2);
    let s1: [u32; 3] = [1, 2, 3];
    let s2: [u32; 5] = [7, 6, 5, 4, 3];
    let seqs: [&[u32]; 2] = [&s1, &s2];
    let h = held_out(&teacher, &student, &cfg, &seqs).expect("pooled");
    let a = held_out_seq(&teacher, &student, &cfg, &s1).expect("s1");
    let b = held_out_seq(&teacher, &student, &cfg, &s2).expect("s2");
    assert_eq!(h.tokens, a.tokens + b.tokens);

    let n = (a.tokens + b.tokens) as f64;
    let wkl = (a.kl * a.tokens as f64 + b.kl * b.tokens as f64) / n;
    let wtv = (a.tv * a.tokens as f64 + b.tv * b.tokens as f64) / n;
    let wt1 = (a.top1 * a.tokens as f64 + b.top1 * b.tokens as f64) / n;
    let m = ((a.tokens - 1) + (b.tokens - 1)) as f64;
    let wnll_s =
        (a.nll_student * (a.tokens - 1) as f64 + b.nll_student * (b.tokens - 1) as f64) / m;
    let wnll_t =
        (a.nll_teacher * (a.tokens - 1) as f64 + b.nll_teacher * (b.tokens - 1) as f64) / m;
    assert!((h.kl - wkl).abs() <= 1e-12 * wkl.abs().max(1.0), "kl {h:?}");
    assert!((h.tv - wtv).abs() <= 1e-12 * wtv.abs().max(1.0), "tv {h:?}");
    assert!((h.top1 - wt1).abs() <= 1e-12, "top1 {h:?}");
    assert!(
        (h.nll_student - wnll_s).abs() <= 1e-12 * wnll_s.abs().max(1.0),
        "nll_student {h:?}"
    );
    assert!(
        (h.nll_teacher - wnll_t).abs() <= 1e-12 * wnll_t.abs().max(1.0),
        "nll_teacher {h:?}"
    );
}

/// Refusals: an empty sequence, a one-token sequence (no next-token position), a token id at or past
/// the vocabulary, and an empty sequence list.
#[test]
fn refusals() {
    let cfg = config();
    let w = weights(1);
    assert!(held_out_seq(&w, &w, &cfg, &[]).is_err(), "empty sequence");
    assert!(
        held_out_seq(&w, &w, &cfg, &[1]).is_err(),
        "one-token sequence"
    );
    assert!(
        held_out_seq(&w, &w, &cfg, &[1, V as u32]).is_err(),
        "token id == vocab"
    );
    assert!(
        held_out_seq(&w, &w, &cfg, &[1, V as u32 + 3]).is_err(),
        "token id > vocab"
    );
    let empty: [&[u32]; 0] = [];
    assert!(held_out(&w, &w, &cfg, &empty).is_err(), "no sequences");
}

/// The C4 token file the measurement's chunk 50 is read from.
const C4_TOKENS: &str = "/home/socol/oc-tmp/lanes/w18/kd/c4data/train-c4.qwen.npy";
/// The f16 teacher.
const QWEN05B_F16: &str = "/home/socol/models/gguf/qwen05b-f16.gguf";
/// The `Q4_K_M` student.
const QWEN05B_Q4KM: &str = "/home/socol/models/gguf/qwen05b-Q4_K_M.gguf";

/// The C4 token stream cut into the protocol's `512`-token chunks.
fn c4_chunks() -> Vec<Vec<u32>> {
    let bytes = std::fs::read(C4_TOKENS).expect("read C4 token file");
    let ids = read_npy_i4(&bytes).expect("parse C4 token file");
    chunks(&ids, 512).expect("chunk C4 tokens")
}

/// The first `n` ids of the protocol's first held-out chunk (chunk `50`).
fn held_chunk_head(all: &[Vec<u32>], n: usize) -> Vec<u32> {
    all.get(50)
        .expect("C4 chunk 50")
        .iter()
        .take(n)
        .copied()
        .collect()
}

/// The token-weighted pooling of the per-sequence scores, the same aggregation
/// [`held_out`]/[`held_out_engines`] perform in exact integers (the per-sequence values are
/// already the rounded means, so this recomputation carries `f64` rounding the accumulator does
/// not; it is used only to print and to check the engine path's own pooling).
fn pool(scores: &[HeldOut]) -> HeldOut {
    let tokens: u64 = scores.iter().map(|h| h.tokens).sum();
    let pairs: u64 = scores.iter().map(|h| h.tokens.saturating_sub(1)).sum();
    let tn = tokens as f64;
    let pn = pairs as f64;
    HeldOut {
        tokens,
        kl: scores.iter().map(|h| h.kl * h.tokens as f64).sum::<f64>() / tn,
        tv: scores.iter().map(|h| h.tv * h.tokens as f64).sum::<f64>() / tn,
        top1: scores.iter().map(|h| h.top1 * h.tokens as f64).sum::<f64>() / tn,
        nll_student: scores
            .iter()
            .map(|h| h.nll_student * h.tokens.saturating_sub(1) as f64)
            .sum::<f64>()
            / pn,
        nll_teacher: scores
            .iter()
            .map(|h| h.nll_teacher * h.tokens.saturating_sub(1) as f64)
            .sum::<f64>()
            / pn,
    }
}

/// Peak resident set size in KiB, read from `/proc/self/status` (`VmHWM`).
fn rss_peak_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmHWM:")
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|kib| kib.parse().ok())
    })
}

/// Print one measured set: pooled figures, per-chunk KL list with its mean ± `sd/√n`, and the
/// per-chunk wall.
fn report_set(label: &str, scores: &[HeldOut], walls: &[EngineWalls], tot_s: f64) {
    let p = pool(scores);
    let n = scores.len() as f64;
    let mean = scores.iter().map(|h| h.kl).sum::<f64>() / n;
    let var = if scores.len() > 1 {
        scores.iter().map(|h| (h.kl - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let se = (var / n).sqrt();
    let list: Vec<String> = scores.iter().map(|h| format!("{:.6}", h.kl)).collect();
    println!(
        "{label}: n {} tokens {} KL {:.6} TV {:.6} top-1 {:.6} NLL_s {:.4} NLL_t {:.4}",
        scores.len(),
        p.tokens,
        p.kl,
        p.tv,
        p.top1,
        p.nll_student,
        p.nll_teacher
    );
    println!("  per-chunk KL: [{}]", list.join(", "));
    println!("  per-chunk KL mean {mean:.6} ± {se:.6} (sd/sqrt n)");
    println!("  wall {tot_s:.3} s total, {:.3} s/chunk", tot_s / n);
    for (i, (h, w)) in scores.iter().zip(walls).enumerate() {
        println!(
            "  chunk {i}: wall {:.3} s, KL {:.6}, top-1 {:.3}",
            w.wall_ns as f64 / 1e9,
            h.kl,
            h.top1
        );
    }
}

/// The measurement link between the two logit sources, not a parity control: on the real `qwen05b`
/// f16 / `Q4_K_M` pair and the first 16 ids of held-out chunk 50, print the production `Q8Act`
/// engine path's `|ΔKL|`, `|ΔTV|`, `|ΔNLL|` and top-1 against [`held_out`] (grad `forward`). The
/// two paths compute different arithmetic (int8 activations against exact Q32), so the test asserts
/// only that it ran: 16 scored positions and finite values. `APEIRON_F32_CARRIER=0` selects the Q32
/// `KvState` carrier the grad forward refines, isolating the projection arithmetic.
#[test]
#[ignore = "loads two GGUFs (~1.4 GB) and the C4 token file; run explicitly with --ignored"]
fn heldout_engine_matches_grad_qwen05b() {
    std::env::set_var("APEIRON_F32_CARRIER", "0");
    let all = c4_chunks();
    let tokens = held_chunk_head(&all, 16);
    let teacher = Engine::open(QWEN05B_F16).expect("open f16 teacher");
    let student = Engine::open(QWEN05B_Q4KM).expect("open Q4_K_M student");
    let t0 = std::time::Instant::now();
    let eng = held_out_engines(&teacher, &student, &[&tokens]).expect("engine path");
    let eng_s = t0.elapsed().as_secs_f64();
    let (wt, cfg) = Weights::from_engine(&teacher).expect("teacher weights");
    let (ws, _) = Weights::from_engine(&student).expect("student weights");
    let t1 = std::time::Instant::now();
    let grad = held_out(&wt, &ws, &cfg, &[&tokens]).expect("grad path");
    let grad_s = t1.elapsed().as_secs_f64();

    println!(
        "probe: tokens {} |dKL| {:.6} (Q8Act {:.6} vs grad {:.6}) |dTV| {:.6} ({:.6} vs {:.6}) \
         |dNLL_s| {:.6} top-1 {:.4} vs {:.4}; wall Q8Act {eng_s:.2} s, grad {grad_s:.2} s",
        eng.tokens,
        (eng.kl - grad.kl).abs(),
        eng.kl,
        grad.kl,
        (eng.tv - grad.tv).abs(),
        eng.tv,
        grad.tv,
        (eng.nll_student - grad.nll_student).abs(),
        eng.top1,
        grad.top1
    );
    assert_eq!(eng.tokens, 16, "scored positions");
    assert_eq!(grad.tokens, 16, "scored positions");
    for (name, h) in [("Q8Act", eng), ("grad", grad)] {
        assert!(
            h.kl.is_finite()
                && h.tv.is_finite()
                && h.nll_student.is_finite()
                && h.nll_teacher.is_finite(),
            "{name} produced a non-finite figure: {h:?}"
        );
    }
}

/// The engine path's identity: the same engine as teacher and student gives zero KL, zero TV, unit
/// top-1 and equal NLLs exactly, because the two logit buffers are the same bytes.
#[test]
#[ignore = "loads qwen05b-f16 (~1 GB); run explicitly with --ignored"]
fn heldout_engine_identity_qwen05b() {
    let all = c4_chunks();
    let tokens = held_chunk_head(&all, 16);
    let engine = Engine::open(QWEN05B_F16).expect("open f16");
    let t0 = std::time::Instant::now();
    let (per, walls) =
        held_out_engines_per_seq_timed(&engine, &engine, &[&tokens]).expect("per-seq eval");
    println!(
        "identity eval wall {:.3} s (engine wall {:.3} s)",
        t0.elapsed().as_secs_f64(),
        walls[0].wall_ns as f64 / 1e9
    );
    assert_eq!(per.len(), 1);
    let h = per[0];
    println!("identity: {h:?}");
    let t1 = std::time::Instant::now();
    let (per2, walls2) =
        held_out_engines_per_seq_timed(&engine, &engine, &[&tokens]).expect("second eval");
    println!(
        "second eval wall {:.3} s (engine wall {:.3} s)",
        t1.elapsed().as_secs_f64(),
        walls2[0].wall_ns as f64 / 1e9
    );
    assert_eq!(per2, per);
    assert_eq!(h.kl, 0.0, "equal logits give 0 exactly");
    assert_eq!(h.tv, 0.0);
    assert_eq!(h.top1, 1.0);
    assert_eq!(h.nll_student, h.nll_teacher);
}

/// The measurement on the engine's shipped `Q8Act` path: C4 chunks of 512 tokens, held-out
/// `[50, 90)` and in-sample `[0, 20)`, teacher f16 against student `Q4_K_M`. The first 16 positions
/// of the first held-out chunk time the anchor, and one full 512-token chunk is the pilot; the
/// campaign then runs the largest `n_h = n_i = n <= 20` whose pilot plus `2n` chunks fit a 470 s
/// wall, so the whole run stays inside the 560 s gate on a contended box. The table states `n`.
#[test]
#[ignore = "loads two GGUFs and scores many C4 chunks; run explicitly with --ignored"]
fn heldout_q4km_qwen05b() {
    let all = c4_chunks();
    let sp = split(all.len(), 50, 20, 40).expect("protocol split");
    let teacher = Engine::open(QWEN05B_F16).expect("open f16 teacher");
    let student = Engine::open(QWEN05B_Q4KM).expect("open Q4_K_M student");

    let pilot_idx = sp.heldout[0];
    let anchor: Vec<u32> = all[pilot_idx].iter().take(16).copied().collect();
    let t0 = std::time::Instant::now();
    let (anchor_scores, anchor_walls) =
        held_out_engines_per_seq_timed(&teacher, &student, &[&anchor])
            .expect("16-position anchor of the first held-out chunk");
    let anchor_s = t0.elapsed().as_secs_f64();
    let scale = 512.0 / anchor.len() as f64;
    let per_chunk = scale * anchor_s;
    println!(
        "anchor: {} positions of chunk {pilot_idx}: {anchor_s:.3} s (engine {:.3} s), KL {:.6}, \
         top-1 {:.6}",
        anchor.len(),
        anchor_walls[0].wall_ns as f64 / 1e9,
        anchor_scores[0].kl,
        anchor_scores[0].top1
    );
    println!("projection: one 512-token chunk = 32 x anchor = {per_chunk:.1} s");
    assert!(
        4.0 * per_chunk <= 470.0,
        "engine campaign cannot fit the gate: per-chunk projection {per_chunk:.1} s"
    );

    let t_pilot = std::time::Instant::now();
    let (pilot, pilot_walls) =
        held_out_engines_per_seq_timed(&teacher, &student, &[all[pilot_idx].as_slice()])
            .expect("pilot chunk");
    let pilot_s = t_pilot.elapsed().as_secs_f64();
    println!(
        "pilot: held-out chunk {pilot_idx}: total {pilot_s:.3} s (engine wall {:.3} s), KL {:.6}, \
         top-1 {:.6}",
        pilot_walls[0].wall_ns as f64 / 1e9,
        pilot[0].kl,
        pilot[0].top1
    );

    // The pilot is one chunk; `2n` more must fit the 470 s campaign budget with it.
    let mut n = ((470.0 / pilot_s - 1.0) / 2.0).floor() as usize;
    n = n.min(sp.heldout.len()).min(sp.in_sample.len());
    assert!(
        n >= 2,
        "engine campaign cannot fit the gate: {pilot_s:.3} s/chunk"
    );
    println!(
        "projection: n_h = n_i = {n} ({} chunks) -> {:.1} s from {:.3} s/chunk",
        2 * n,
        2.0 * n as f64 * pilot_s,
        pilot_s
    );

    let held: Vec<&[u32]> = sp.heldout[..n].iter().map(|&c| all[c].as_slice()).collect();
    let in_s: Vec<&[u32]> = sp.in_sample[..n]
        .iter()
        .map(|&c| all[c].as_slice())
        .collect();

    let t1 = std::time::Instant::now();
    let (held_scores, held_walls) =
        held_out_engines_per_seq_timed(&teacher, &student, &held).expect("held-out campaign");
    let held_s = t1.elapsed().as_secs_f64();
    let t2 = std::time::Instant::now();
    let (in_scores, in_walls) =
        held_out_engines_per_seq_timed(&teacher, &student, &in_s).expect("in-sample campaign");
    let in_s_wall = t2.elapsed().as_secs_f64();

    println!(
        "reference (torch, f32-dequantised Q4_K_M, n = 40/20; docs/evidence/onebit-w15.md:303): \
         in-sample 0.0174, held-out 0.0197"
    );
    report_set("held-out", &held_scores, &held_walls, held_s);
    report_set("in-sample", &in_scores, &in_walls, in_s_wall);
    if let Some(kib) = rss_peak_kib() {
        println!("RSS peak: {kib} KiB ({:.1} MiB)", kib as f64 / 1024.0);
    }
    println!(
        "campaign wall {:.3} s (pilot {:.3} s separate)",
        held_s + in_s_wall,
        pilot_s
    );

    assert_eq!(
        pool(&held_scores).tokens,
        held_scores.len() as u64 * 512,
        "held-out token count"
    );
    assert_eq!(
        pool(&in_scores).tokens,
        in_scores.len() as u64 * 512,
        "in-sample token count"
    );
    assert!(held_scores.iter().all(|h| h.kl > 0.0));
}
