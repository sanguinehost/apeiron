//! The held-out evaluator and the `--kl-divergence-base` port: identical models give zero KL and TV,
//! a perturbed `lm_head` row matches an independent `f64` softmax-KL, the pooled mean is
//! token-weighted, the reader round-trips and refuses, and `heldout_kldbase_qwen05b` checks the
//! `Q8Act` engine against llama.cpp's own KLD. Grade `tested`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "the LCG maps a word into a Q32.32 weight and the f64 softmax-KL reference is the \
              independent recomputation; the exact float comparisons are the identical-model test"
)]

use apeiron::engine::{Engine, ONE};
use apeiron::learn::fold::stats::heldout::{
    held_out, held_out_kldbase_chunk, held_out_seq, score_base_row, BaseScore,
};
use apeiron::learn::fold::stats::kldbase::KldBase;
use apeiron::learn::grad::model::{forward, Config, LayerWeights, Weights};
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

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

fn real(z: &[i64]) -> Vec<f64> {
    z.iter().map(|&q| q as f64 / ONE as f64).collect()
}

/// Agreement of a migrated aggregate with its `f64` oracle, within the carrier's derived bound: a
/// mean is one fixed-point division, at most half a grid quantum (`2⁻³³`) from the exact quotient,
/// and a token-weighted recomposition of per-sequence means adds two more such divisions; `4·2⁻³³`
/// covers the three roundings, with the `f64` products and sum far below their own quantum.
fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 4.0 * 2f64.powi(-33) * want.abs().max(1.0)
}

/// FNV-1a 64 fold of `bytes` into `h`.
fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The numerical-recipes LCG the recorded witness fixture was generated with: distinct from the
/// file's, so the recorded digests are reproduced exactly.
struct WitnessLcg(u64);

impl WitnessLcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn word(&mut self) -> i64 {
        (self.next() >> 33) as i64 % 2001 - 1000
    }
}

/// The recorded base-row digest: `v = 64`, `z` small words scaled by `ONE/256`, `lp` straddling the
/// `−16` flush threshold, target `3` and base argmax `2`.
fn witness_base_row(seed: u64) -> u64 {
    let mut rng = WitnessLcg::new(seed);
    let z: Vec<i64> = (0..64).map(|_| rng.word() * (ONE >> 8)).collect();
    let lp: Vec<i64> = (0..64)
        .map(|_| {
            let x = -0.05 - ((rng.next() >> 40) % 199) as f64 / 10.0;
            to_grid(x, GRID_BOUND).expect("base log p on the grid")
        })
        .collect();
    let s = score_base_row(&z, &lp, 3, Some(2)).expect("row");
    let mut h = fnv1a(0xcbf2_9ce4_8422_2325, b"row");
    for w in [s.kl, s.nll_student, s.nll_base, s.top1 as i64] {
        h = fnv1a(h, &w.to_le_bytes());
    }
    h
}

/// The recorded held-out aggregate digest: [`weights`]`(seed)`, a perturbed `lm_head`, and the five
/// mean words over the `[1, 2, 3, 4, 5]` sequence.
fn witness_heldout(seed: u64) -> u64 {
    let teacher = weights(seed);
    let mut student = teacher.clone();
    for k in 0..E {
        student.lm_head[k] += 3 * (ONE >> 2);
    }
    let toks: [u32; 5] = [1, 2, 3, 4, 5];
    let hh = held_out_seq(&teacher, &student, &config(), &toks).expect("eval");
    let tokens = i64::try_from(hh.tokens).expect("tokens");
    let mut h = fnv1a(0xcbf2_9ce4_8422_2325, b"heldout");
    for w in [
        tokens,
        hh.kl,
        hh.tv,
        hh.top1,
        hh.nll_student,
        hh.nll_teacher,
    ] {
        h = fnv1a(h, &w.to_le_bytes());
    }
    h
}

/// The recorded pre-migration digests of the held-out face, from the source at base `57d21789`.
/// The word accumulators are unchanged by the migration; only the report struct's field type moved
/// from a decimal to the exact word, so these digests must hold exactly.
#[test]
fn recorded_heldout_digests_hold() {
    for (seed, want) in [
        (0x0BA5_E001u64, 0xc408_a9ce_25fa_6911u64),
        (0x0BA5_E002, 0x2933_5644_f070_f9ff),
    ] {
        assert_eq!(witness_base_row(seed), want, "base row seed {seed:016x}");
    }
    for (seed, want) in [
        (0x5EED_0001u64, 0x04b1_ffdc_6145_1ee5u64),
        (0x5EED_0002, 0x411a_4aa3_4db2_9db4),
    ] {
        assert_eq!(witness_heldout(seed), want, "held-out seed {seed:016x}");
    }
}

/// The independent reference: the mean forward `KL(teacher ‖ student)` in nats from the Q32.32
/// logits, computed in `f64`, one row at a time (`Σ_x p_t(x)·(ln p_t(x) − ln p_s(x)) / T`).
fn f64_kl_mean(teacher: &Weights, student: &Weights, cfg: &Config, toks: &[u32]) -> f64 {
    let (lt, _) = forward(teacher, cfg, toks).expect("teacher forward");
    let (ls, _) = forward(student, cfg, toks).expect("student forward");
    let (t, v) = (toks.len(), cfg.vocab);
    let mut total = 0.0;
    for r in 0..t {
        let (pt, ps) = (
            softmax_f64(&real(&lt[r * v..][..v])),
            softmax_f64(&real(&ls[r * v..][..v])),
        );
        total += (0..v)
            .map(|x| pt[x] * (pt[x].ln() - ps[x].ln()))
            .sum::<f64>();
    }
    total / t as f64
}

/// The pre-migration base-row scorer, verbatim in `f64` (`src/learn/fold/stats/heldout.rs` at base
/// `9f440192`, the body of `heldout_kldbase_chunk`'s row loop): the row maximum `m`, the exponential
/// sum `Σ exp(q − m)`, `lse = m/2³² + ln Σ`, the KL over entries with `log p > −16`, and the two NLL
/// terms. Test-only `f64`; the migrated [`score_base_row`] is compared against it.
fn ref_score_base_row(z: &[i64], lp: &[f64], target: usize) -> (f64, f64, f64) {
    let inv = 1.0 / ONE as f64;
    let m = z
        .iter()
        .map(|&q| q as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = z.iter().map(|&q| ((q as f64 - m) * inv).exp()).sum();
    let lse = m * inv + sum.ln();
    let mut kl = 0.0;
    for (i, &q) in z.iter().enumerate() {
        if lp[i] > -16.0 {
            kl += lp[i].exp() * (lp[i] - (q as f64).mul_add(inv, -lse));
        }
    }
    let nll_s = -((z[target] as f64).mul_add(inv, -lse));
    (kl, nll_s, -lp[target])
}

/// The derived bound of [`score_base_row`] against [`ref_score_base_row`] on a `v`-wide row, in nats.
///
/// `fixed_exp`'s documented error is `1e-8·exp + 4·2⁻³²`, so `S = Σ exp(z − max)` carries a relative
/// error `≤ 1e-8 + 4v·2⁻³²` (`S ≥ 1`: the maximum's own term), and `lse = max + ln S` adds
/// `fixed_ln`'s `7·2⁻³²` and the addition's `2⁻³³`:
/// `δ = 1e-8 + (4v + 7)·2⁻³² + 2⁻³³`. A KL term carries `p·(1e-8·|lp| + 2δ + 2⁻³³)` with `p ≤ 1` and
/// `|lp| ≤ 20`, `v` terms add up, and the row's `round_q32_even` adds `2⁻³³`; the student NLL shares
/// the `lse` error alone.
fn row_tol(v: usize) -> f64 {
    let q = 2f64.powi(-32);
    let d_lse = 1e-8 + (4.0 * v as f64 + 7.0) * q + 0.5 * q;
    v as f64 * (1e-8 * 20.0 + 2.0 * d_lse + 0.5 * q) + 0.5 * q
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
    assert!(h.kl > 0, "distinct models must give positive KL");
    assert!((0..=ONE).contains(&h.tv), "tv is a fraction of ONE");
    assert!((0..=ONE).contains(&h.top1), "top1 is a fraction of ONE");
    assert!(h.nll_teacher > 0 && h.nll_student > 0);
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
    let want = (h.tokens, h.kl, h.tv, h.top1);
    assert_eq!(want, (7, 0, 0, ONE), "equal logits give 0 and 1");
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
    assert!(h.kl > 0, "the perturbed student must give positive KL");
    let want = f64_kl_mean(&teacher, &student, &cfg, &toks);
    let got = from_grid(h.kl);
    let rel = (got - want).abs() / want.abs().max(1e-12);
    assert!(
        rel <= 1e-6,
        "KL {got:.12} vs f64 reference {want:.12} (relative {rel:e})"
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
    let m = ((a.tokens - 1) + (b.tokens - 1)) as f64;
    let w = |x: i64, y: i64| (from_grid(x) * a.tokens as f64 + from_grid(y) * b.tokens as f64) / n;
    let wn = |x: i64, y: i64| {
        (from_grid(x) * (a.tokens - 1) as f64 + from_grid(y) * (b.tokens - 1) as f64) / m
    };
    assert!(close(from_grid(h.kl), w(a.kl, b.kl)), "kl {h:?}");
    assert!(close(from_grid(h.tv), w(a.tv, b.tv)), "tv {h:?}");
    assert!(
        (from_grid(h.top1) - w(a.top1, b.top1)).abs() <= 4.0 * 2f64.powi(-33),
        "top1 {h:?}"
    );
    let ns = wn(a.nll_student, b.nll_student);
    let nt = wn(a.nll_teacher, b.nll_teacher);
    assert!(close(from_grid(h.nll_student), ns), "nll_s");
    assert!(close(from_grid(h.nll_teacher), nt), "nll_t");
}

/// The fixed-point base row agrees with its `f64` oracle ([`ref_score_base_row`]) within the derived
/// [`row_tol`]; the base NLL is the stored word negated.
#[test]
fn base_row_matches_the_f64_oracle() {
    let mut g = Lcg(0x5EED_5EED);
    let v = V;
    let z: Vec<i64> = (0..v)
        .map(|_| ((g.next() >> 40) % 7) as i64 * (ONE >> 1) - 3 * ONE)
        .collect();
    let lp_word: Vec<i64> = (0..v)
        .map(|_| {
            let x = -0.05 - ((g.next() >> 40) % 199) as f64 / 10.0;
            to_grid(x, GRID_BOUND).expect("base log p on the grid")
        })
        .collect();
    let lp: Vec<f64> = lp_word.iter().map(|&w| from_grid(w)).collect();
    let (target, base_argmax) = (3usize, Some(2usize));
    let got = score_base_row(&z, &lp_word, target, base_argmax).expect("row");
    let (kl, nll_s, nll_b) = ref_score_base_row(&z, &lp, target);
    assert!(
        (from_grid(got.kl) - kl).abs() <= row_tol(v),
        "kl {} vs oracle {kl} (bound {})",
        from_grid(got.kl),
        row_tol(v)
    );
    let q = 2f64.powi(-32);
    let d_lse = 1e-8 + (4.0 * v as f64 + 7.0) * q + 0.5 * q;
    assert!(
        (from_grid(got.nll_student) - nll_s).abs() <= 2.0 * d_lse + q,
        "nll_student {} vs oracle {nll_s}",
        from_grid(got.nll_student)
    );
    // The base side has no arithmetic: its NLL is the stored word, negated.
    assert_eq!(from_grid(got.nll_base), nll_b, "nll_base");
    let mut best = 0usize;
    for i in 1..v {
        if z[i] > z[best] {
            best = i;
        }
    }
    assert_eq!(got.top1, u64::from(Some(best) == base_argmax), "top1");
    assert!(
        kl > 0.0 && nll_s > 0.0,
        "the fixture must be non-degenerate"
    );
}

/// Refusals: empty inputs, a one-token sequence, and a token id at or past the vocabulary.
#[test]
fn refusals() {
    let cfg = config();
    let w = weights(1);
    let bad = |t: &[u32]| held_out_seq(&w, &w, &cfg, t).is_err();
    assert!(bad(&[]), "empty sequence");
    assert!(bad(&[1]), "1 token");
    assert!(bad(&[1, V as u32]), "token == vocab");
    assert!(bad(&[1, V as u32 + 3]), "token > vocab");
    let empty: [&[u32]; 0] = [];
    assert!(held_out(&w, &w, &cfg, &empty).is_err(), "no sequences");
}

/// The llama.cpp base files, its own KLD logs and the `Q8_0` student.
const KLD_DIR: &str = "/home/socol/oc-tmp/kld";
const QWEN05B_Q8: &str = "/home/socol/oc-tmp/kld/qwen05b-Q8_0.gguf";

/// A synthetic base file: tokens, then rows with `scale = 0.5`, `min = −3`, codes `0, 2, 4, 1, …`.
fn base_bytes(n_ctx: u32, n_vocab: u32, n_chunk: u32, chunks: usize) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"_logits_");
    for v in [n_ctx, n_vocab, n_chunk] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for t in 0..n_ctx * n_chunk {
        b.extend_from_slice(&(t as i32).to_le_bytes());
    }
    let nv = 2 * (n_vocab as usize).div_ceil(2) + 4;
    for _ in 0..chunks {
        for _ in 0..(n_ctx - 1 - n_ctx / 2) {
            b.extend_from_slice(&0.5f32.to_le_bytes());
            b.extend_from_slice(&(-3.0f32).to_le_bytes());
            for v in 0..(nv - 4) {
                b.extend_from_slice(&[0u16, 2, 4, 1][v % 4].to_le_bytes());
            }
        }
    }
    b
}

/// The base reader's round trip on a synthetic file, its layout refusals, and its boundary
/// accessors (a row past `rows`, a code past the vocabulary).
#[test]
fn kldbase_round_trips_and_refuses() {
    let dir = std::env::temp_dir().join(format!("apeiron-kldbase-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("ok.kld");
    std::fs::write(&path, base_bytes(8, 4, 2, 2)).expect("write base");
    let mut base = KldBase::open(&path).expect("open base");
    assert_eq!(base.n_chunk(), 2);
    let first = base.next_chunk().expect("chunk 0");
    assert_eq!(first.tokens, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!((first.rows, first.n_vocab), (3, 4));
    assert_eq!(first.log_prob(0, 1), Some(-2.0), "0.5·2 − 3");
    assert_eq!(first.log_prob(0, 2), Some(-1.0), "0.5·4 − 3");
    assert_eq!(first.log_prob(0, 4), None, "past the vocabulary");
    assert_eq!(first.log_prob(3, 0), None, "row past the stored rows");
    assert_eq!(first.argmax(0), Some(2), "code 4 is the largest");
    assert_eq!(first.argmax(3), None);
    assert_eq!(base.next_chunk().expect("chunk 1").tokens[0], 8);
    assert!(base.next_chunk().is_err(), "all chunks read");

    let bad = dir.join("bad.kld");
    let mut b = base_bytes(8, 4, 1, 1);
    b[0] = b'X';
    std::fs::write(&bad, &b).expect("write bad signature");
    assert!(KldBase::open(&bad).is_err(), "signature");
    std::fs::write(&bad, base_bytes(8, 4, 2, 1)).expect("write short body");
    let mut short = KldBase::open(&bad).expect("header reads");
    assert!(short.next_chunk().is_ok());
    assert!(short.next_chunk().is_err(), "truncated row block");
    std::fs::write(&bad, base_bytes(1, 4, 1, 1)).expect("write n_ctx = 1");
    assert!(KldBase::open(&bad).is_err(), "n_ctx < 2");
    std::fs::remove_dir_all(&dir).expect("clean scratch");
}

/// llama.cpp's `kl_divergence` log: KLD and top-1 are running means (`perplexity.cpp:1878–1898`),
/// the base NLL is `ln(PPL) − ln(PPL(Q)/PPL(base))`; the last lines are the totals.
struct LlamaLog {
    kld: Vec<f64>,
    bnll: Vec<f64>,
    top1: Vec<f64>,
    mean_kld: f64,
    mean_se: f64,
    top: f64,
}

/// Parse one `--kl-divergence` log at the pinned llama.cpp revision.
fn llama_log(path: &str) -> LlamaLog {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let (mut kld, mut bnll, mut top1) = (Vec::new(), Vec::new(), Vec::new());
    let (mut mean_kld, mut mean_se, mut top) = (0.0, 0.0, 0.0);
    for line in text.lines() {
        let nums: Vec<f64> = line
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        if nums.len() == 11 {
            kld.push(nums[5]);
            bnll.push(nums[1].ln() - nums[3]);
            top1.push(nums[9] / 100.0);
        } else if line.contains("Mean") && line.contains("KLD") {
            mean_kld = nums.first().copied().unwrap_or(0.0);
            mean_se = nums.get(1).copied().unwrap_or(0.0);
        } else if line.starts_with("Same top p") {
            top = nums.first().copied().unwrap_or(0.0) / 100.0;
        }
    }
    LlamaLog {
        kld,
        bnll,
        top1,
        mean_kld,
        mean_se,
        top,
    }
}

/// The mean over chunks `skip .. skip+n` of a running-mean column, from its printed values.
fn slice_mean(cum: &[f64], skip: usize, n: usize) -> f64 {
    let end = cum[skip + n - 1] * (skip + n) as f64;
    let before = if skip == 0 {
        0.0
    } else {
        cum[skip - 1] * skip as f64
    };
    (end - before) / n as f64
}

/// Score one text's base file from chunk `skip` for at most `limit` chunks, with each chunk's wall.
fn score_base(
    student: &Engine,
    text: &str,
    skip: usize,
    limit: usize,
) -> (Vec<BaseScore>, Vec<f64>) {
    let path = format!("{KLD_DIR}/{text}.c40.kld");
    let mut base = KldBase::open(std::path::Path::new(&path)).expect("open base file");
    for _ in 0..skip.min(base.n_chunk()) {
        let _ = base.next_chunk().expect("skipped chunk");
    }
    let n = base.n_chunk().saturating_sub(skip).min(limit);
    let (mut scores, mut walls) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let chunk = base.next_chunk().expect("base chunk");
        let t = std::time::Instant::now();
        scores.push(held_out_kldbase_chunk(student, &chunk).expect("score chunk"));
        walls.push(t.elapsed().as_secs_f64());
    }
    (scores, walls)
}

/// Print and check one slice: our `Q8_0` against llama.cpp's own `Q8_0` KLD on the same base file,
/// with its `Q4_K_M` column as the comparator. The bounds are `10 %` of the `Q4_K_M` gap on the same
/// file (`0.0025` nats for KLD and base NLL) and `25 %` of its top-1 gap (`0.02`); the residual is
/// `Q8Act` activation arithmetic, not the protocol: the base NLL is a pure re-read.
fn report_kld(
    text: &str,
    skip: usize,
    scores: &[BaseScore],
    walls: &[f64],
    q8: &LlamaLog,
    q4: &LlamaLog,
) {
    let n = scores.len();
    let nf = n as f64;
    assert!(
        n > 0 && skip + n <= q8.kld.len() && skip + n <= q4.kld.len(),
        "{text}: {n} chunks from {skip} against {} / {} llama.cpp lines",
        q8.kld.len(),
        q4.kld.len()
    );
    let mean = |f: fn(&BaseScore) -> i64| scores.iter().map(|s| from_grid(f(s))).sum::<f64>() / nf;
    let (mean_kl, top, nll_s, nll_b) = (
        mean(|s| s.kl),
        mean(|s| s.top1),
        mean(|s| s.nll_student),
        mean(|s| s.nll_base),
    );
    let (l8, t8, b8) = (
        slice_mean(&q8.kld, skip, n),
        slice_mean(&q8.top1, skip, n),
        slice_mean(&q8.bnll, skip, n),
    );
    let l4 = slice_mean(&q4.kld, skip, n);
    let (mut d_chunk, mut rmin, mut rmax) = (0.0f64, f64::INFINITY, f64::NEG_INFINITY);
    let mut ratios = Vec::with_capacity(n);
    for (i, s) in scores.iter().enumerate() {
        d_chunk = d_chunk.max((from_grid(s.kl) - slice_mean(&q8.kld, skip + i, 1)).abs());
        let r = from_grid(s.kl) / slice_mean(&q4.kld, skip + i, 1);
        rmin = rmin.min(r);
        rmax = rmax.max(r);
        ratios.push(r);
    }
    let s_chunk = walls.iter().sum::<f64>() / nf;
    let s_max = walls.iter().fold(0.0f64, |a, &w| a.max(w));
    let var = scores
        .iter()
        .map(|s| (from_grid(s.kl) - mean_kl).powi(2))
        .sum::<f64>()
        / (nf - 1.0).max(1.0);
    println!(
        "{text}: chunks {skip}..{}, {} rows, {s_chunk:.3} s/chunk mean ({s_max:.3} max)",
        skip + n,
        scores[0].tokens
    );
    println!(
        "  ours Q8_0   : KL {mean_kl:.6} ± {:.6} (sd/√n)  top-1 {top:.6}  NLL_s {nll_s:.4}  base NLL {nll_b:.4}",
        (var / nf).sqrt()
    );
    println!(
        "  llama Q8_0  : KL {l8:.6}  top-1 {t8:.6}  base NLL {b8:.6}  (full file: KL {:.6} ± {:.6}, top-1 {:.6}); \
         Q4_K_M: KL {l4:.6} (full file: {:.4} ± {:.4}, top-1 {:.4})",
        q8.mean_kld, q8.mean_se, q8.top, q4.mean_kld, q4.mean_se, q4.top
    );
    let (dkl, dt, db) = ((mean_kl - l8).abs(), (top - t8).abs(), (nll_b - b8).abs());
    println!(
        "  |Δ| vs llama Q8_0: KL {dkl:.6}, top-1 {dt:.6}, base NLL {db:.6}; per-chunk KL max {d_chunk:.6}"
    );
    println!(
        "  paired per-chunk ours/Q4_K_M: mean {:.4}, range {rmin:.4} .. {rmax:.4}",
        ratios.iter().sum::<f64>() / nf
    );
    assert!(
        dkl <= 0.0025 && dt <= 0.02 && db <= 0.0025,
        "{text}: vs llama.cpp Q8_0 on the same base — KL |Δ| {dkl:.6} (≤ 0.0025), top-1 {dt:.4} \
         (≤ 0.02), base NLL {db:.6} (≤ 0.0025)"
    );
}

/// The like-for-like control and the table over each text's base file. `APEIRON_KLD_TEXT` selects one
/// text, `APEIRON_KLD_SKIP`/`APEIRON_KLD_CHUNKS` one slice, so the campaign splits; every comparison
/// is against llama.cpp's column at the same chunk index and so holds on any slice.
#[test]
#[ignore = "reads two ~3 GB llama.cpp base files and the Q8_0 GGUF; run explicitly with --ignored"]
fn heldout_kldbase_qwen05b() {
    let env = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let skip = env("APEIRON_KLD_SKIP", 0);
    let limit = env("APEIRON_KLD_CHUNKS", usize::MAX);
    let only = std::env::var("APEIRON_KLD_TEXT").unwrap_or_default();
    let student = Engine::open(QWEN05B_Q8).expect("open Q8_0");
    let x = vec![ONE / 4; student.config.n_embd];
    let t = std::time::Instant::now();
    let _ = student.lm_head_fixed(&x).expect("head column");
    let head = t.elapsed().as_secs_f64();
    println!("head {head:.4} s/col, {:.1} s/chunk", 255.0 * head);
    for text in ["train400k", "calib"] {
        if !only.is_empty() && only != text {
            continue;
        }
        let (scores, walls) = score_base(&student, text, skip, limit);
        let q8 = llama_log(&format!("{KLD_DIR}/{text}.c40.qwen05b-Q8_0.log"));
        let q4 = llama_log(&format!("{KLD_DIR}/{text}.c40.qwen05b-Q4_K_M.log"));
        report_kld(text, skip, &scores, &walls, &q8, &q4);
    }
}
