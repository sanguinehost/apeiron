//! Decode-dominance measurement instruments: static weight-block statistics and
//! teacher-forced draft acceptance.
//!
//! Run (release): `cargo run --release --locked --example decode_probe -- [flags]`
//!
//! Flags: `--tokens N` (default 2048), `--text PATH` (default `README.md`),
//! `--only qwen|bonsai`, `--skip-weights`, `--skip-accept`, `--prune` (opt-in instrument 3),
//! `--prune-tokens N` (default 64), `--threads N` (default `available_parallelism`).
//!
//! Three instruments, all grades `measured` (`AGENTS.md`, evidence grades):
//!
//! 1. **Weight-block census.** For each quantized projection class of each GGUF, decode the
//!    weight codes and group them into consecutive 32-weight blocks. Report the fraction of
//!    all-zero blocks, of stored zero codes, and of exactly-zero dequantized weights. This is the
//!    static side of the zero-skip lever (L3): the payload census, not the runtime activation
//!    census.
//! 2. **Draft acceptance.** Teacher-force a fixed English text through the real engine, record the
//!    greedy target at every position, and measure the per-token argmax agreement `alpha` of three
//!    draft sources: a prompt-lookup n-gram (`n = 1..4`), an online bigram table, and a uniform
//!    shuffled control that must land at `1/vocab`. It reports `E = (1-alpha^(k+1))/(1-alpha)` and
//!    the empirical mean accepted run length for `k = 2, 4, 8`.
//! 3. **Bound-pruned LM head (opt-in, `--prune`).** For each teacher-forced token, an exact weight
//!    row summary gives `Â_i` and a sound `E_i ≥ |L_i − Â_i|`; with `B_i = ⌊E_i/2^32⌋ + 1` the
//!    candidate set `C = {i : Ĉ_i + B_i ≥ max_j(Ĉ_j − B_j)}` contains every maximiser of the exact
//!    `Q8Act` logit `L`, so `argmax` over `C` is `argmax_fixed(L)`. The probe measures `|C|` for
//!    five `Q8_0` top-bit truncations of the Qwen head and the `PQ2_0` gap / zero-summary bound of
//!    Bonsai. The head input `x̂ = dequant(quant_act(x))` is reconstructed by mirroring the `Q8Act`
//!    forward with the public kernels, asserted per token to equal the engine's logits bit for bit.
//!
//! The control is enforced: the shuffled source's `alpha` must be `<= 0.01` or the probe exits 1.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::Write as _;
use std::process::ExitCode;
use std::thread;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::{
    argmax_fixed, f16_to_fixed, matvec_fixed, rms_norm_fixed, silu_fixed, KernelMode, KvState,
};
use apeiron::format::bpe::Bpe;
use apeiron::format::quant;

/// `~/models/gguf/qwen05b-Q4_K_M.gguf`.
const DEFAULT_QWEN: &str = concat!(env!("HOME"), "/models/gguf/qwen05b-Q4_K_M.gguf");
/// Qwen2.5-0.5B's `tokenizer.json`.
const DEFAULT_QWEN_TOKENIZER: &str = concat!(env!("HOME"), "/models/qwen2.5-0.5b/tokenizer.json");
/// `Ternary-Bonsai-1.7B-PQ2_0.gguf`.
const DEFAULT_BONSAI: &str = concat!(
    env!("HOME"),
    "/models/ternary-bonsai-1.7b/Ternary-Bonsai-1.7B-PQ2_0.gguf"
);
/// Bonsai's tokenization reference. The Prism checkpoint is Qwen-family; its sibling
/// `~/models/bonsai/tokenizer.json` is a `LlamaTokenizer` (vocab 32000, byte-fallback)
/// that does not match the model's 151936-token embedding, so the measured command's Qwen
/// tokenizer is reused (it tokenizes the probe prompt identically to the reference run).
const DEFAULT_BONSAI_TOKENIZER: &str = concat!(env!("HOME"), "/models/qwen2.5-0.5b/tokenizer.json");
/// The fixed English text, chosen by the task.
const DEFAULT_TEXT: &str = concat!(env!("HOME"), "/Workspace/apeiron/README.md");
/// Teacher-forced positions by default.
const DEFAULT_TOKENS: usize = 2048;
/// The draft lengths whose expected-token count is reported.
const K_VALUES: [usize; 3] = [2, 4, 8];
/// The longest draft any source is asked to produce.
const K_MAX: usize = 8;
/// The n-gram lengths of the prompt-lookup source.
const NGRAM_NS: [usize; 4] = [1, 2, 3, 4];
/// The shuffled control must not exceed this `alpha`.
const SHUFFLED_MAX_ALPHA: f64 = 0.01;

// ── command line ────────────────────────────────────────────────────────────────────────────────

/// Parsed flags with the task defaults.
struct Opts {
    tokens: usize,
    text: String,
    only: Option<String>,
    skip_weights: bool,
    skip_accept: bool,
    prune: bool,
    prune_tokens: usize,
    threads: usize,
}

/// Parse `argv`, refusing an unknown flag or a malformed value.
fn parse_args() -> Result<Opts, String> {
    let mut opts = Opts {
        tokens: DEFAULT_TOKENS,
        text: DEFAULT_TEXT.to_string(),
        only: None,
        skip_weights: false,
        skip_accept: false,
        prune: false,
        prune_tokens: PRUNE_TOKENS,
        threads: std::thread::available_parallelism().map_or(8, std::num::NonZeroUsize::get),
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0usize;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--tokens" => {
                i += 1;
                let v = args.get(i).ok_or("--tokens needs a value")?;
                opts.tokens = v
                    .parse::<usize>()
                    .map_err(|e| format!("bad --tokens {v:?}: {e}"))?;
            }
            "--text" => {
                i += 1;
                opts.text
                    .clone_from(args.get(i).ok_or("--text needs a value")?);
            }
            "--only" => {
                i += 1;
                let v = args.get(i).ok_or("--only needs a value")?;
                if v != "qwen" && v != "bonsai" {
                    return Err(format!("--only expects qwen or bonsai, got {v:?}"));
                }
                opts.only = Some(v.clone());
            }
            "--skip-weights" => opts.skip_weights = true,
            "--skip-accept" => opts.skip_accept = true,
            "--prune" => opts.prune = true,
            "--prune-tokens" => {
                i += 1;
                let v = args.get(i).ok_or("--prune-tokens needs a value")?;
                opts.prune_tokens = v
                    .parse::<usize>()
                    .map_err(|e| format!("bad --prune-tokens {v:?}: {e}"))?;
            }
            "--threads" => {
                i += 1;
                let v = args.get(i).ok_or("--threads needs a value")?;
                opts.threads = v
                    .parse::<usize>()
                    .map_err(|e| format!("bad --threads {v:?}: {e}"))?;
            }
            "--help" | "-h" => {
                return Err(
                    "usage: decode_probe [--tokens N] [--text PATH] [--only qwen|bonsai] \
                            [--skip-weights] [--skip-accept] [--prune] [--prune-tokens N] \
                            [--threads N]"
                        .to_string(),
                );
            }
            other => return Err(format!("unknown flag {other:?}")),
        }
        i += 1;
    }
    if opts.tokens < 8 {
        return Err("--tokens must be at least 8".to_string());
    }
    if opts.threads == 0 {
        return Err("--threads must be at least 1".to_string());
    }
    if opts.prune_tokens < 4 {
        return Err("--prune-tokens must be at least 4".to_string());
    }
    Ok(opts)
}

// ── part 1: weight-block census ─────────────────────────────────────────────────────────────────

/// Accumulated per-class weight census over every layer of one projection.
#[derive(Default, Clone)]
struct ClassStats {
    /// Tensors folded into this class.
    tensors: u64,
    /// Consecutive 32-weight blocks seen.
    blocks32: u64,
    /// Individual codes seen (always `32 * blocks32` for supported geometries).
    codes: u64,
    /// Stored code field equal to zero.
    code_zero: u64,
    /// Dequantized weight exactly `0.0`.
    w_zero: u64,
    /// 32-weight blocks whose stored codes are all zero.
    allzero_code: u64,
    /// 32-weight blocks whose dequantized weights are all exactly zero.
    allzero_w: u64,
    /// The ggml type names contributing to this class.
    types: Vec<&'static str>,
}

impl ClassStats {
    /// Fold `other` into `self` for the total row.
    fn merge(&mut self, other: &ClassStats) {
        self.tensors += other.tensors;
        self.blocks32 += other.blocks32;
        self.codes += other.codes;
        self.code_zero += other.code_zero;
        self.w_zero += other.w_zero;
        self.allzero_code += other.allzero_code;
        self.allzero_w += other.allzero_w;
        for t in &other.types {
            if !self.types.contains(t) {
                self.types.push(t);
            }
        }
    }
}

/// The projection class of a tensor name: the segment before `.weight`/`.bias`, so every layer's
/// `blk.{l}.attn_q.weight` folds into `attn_q`.
fn tensor_class(name: &str) -> &str {
    let mut it = name.rsplit('.');
    let last = it.next().unwrap_or(name);
    let prev = it.next();
    match last {
        "weight" | "bias" => prev.unwrap_or(last),
        _ => last,
    }
}

/// Whether the crate has a payload-level code reader for this ggml type.
fn has_reader(t: u32) -> bool {
    matches!(
        t,
        quant::GGML_TYPE_Q4_K
            | quant::GGML_TYPE_Q6_K
            | quant::GGML_TYPE_Q8_0
            | quant::GGML_TYPE_Q5_0
            | quant::GGML_TYPE_PQ2_0
    )
}

/// Whether the ggml type is a float vector rather than a quantized code stream.
fn is_float_type(t: u32) -> bool {
    t == quant::GGML_TYPE_F32 || t == quant::GGML_TYPE_F16
}

/// The slot of `class` in `stats`, inserting an empty entry when it is new.
fn class_slot(stats: &mut Vec<(String, ClassStats)>, class: &str) -> usize {
    if let Some(i) = stats.iter().position(|(c, _)| c == class) {
        return i;
    }
    stats.push((class.to_string(), ClassStats::default()));
    stats.len() - 1
}

/// Decode one block of type `t` into `codes` (the stored field value) and `weights` (the exact
/// `f32` dequantization). Returns `false` for a type [`has_reader`] rejects.
///
/// `Q4_K`/`Q6_K` use the crate's own block dequantizers; `Q5_0`/`Q8_0` are one scale times the
/// signed code; `PQ2_0` is `(code - 1) * d` (the fork layout the kernel reads).
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "stored code fields are 2..6 bits and are reinterpreted as their unsigned field \
              value; `Q8_0`'s byte is the i8 code bit pattern"
)]
fn decode_block(
    t: u32,
    blk: &[u8],
    codes: &mut Vec<u8>,
    weights: &mut Vec<f32>,
) -> Result<bool, Box<dyn Error>> {
    match t {
        quant::GGML_TYPE_Q4_K => {
            let c = quant::q4_k_codes(blk)?;
            let mut w = [0f32; quant::QK_K];
            quant::dequant_q4_k_block(blk, &mut w)?;
            codes.extend_from_slice(&c);
            weights.extend_from_slice(&w);
        }
        quant::GGML_TYPE_Q6_K => {
            let c = quant::q6_k_codes(blk)?;
            let mut w = [0f32; quant::QK_K];
            quant::dequant_q6_k_block(blk, &mut w)?;
            let mut i = 0usize;
            while i < quant::QK_K {
                codes.push((i32::from(c[i]) + 32) as u8);
                i += 1;
            }
            weights.extend_from_slice(&w);
        }
        quant::GGML_TYPE_Q8_0 => {
            let c = quant::q8_0_codes(blk, 0)?;
            let d = quant::q8_0_scale(blk, 0)?;
            let mut i = 0usize;
            while i < quant::Q8_0_QK {
                codes.push(c[i] as u8);
                weights.push(d * f32::from(c[i]));
                i += 1;
            }
        }
        quant::GGML_TYPE_Q5_0 => {
            let c = quant::q5_0_codes(blk)?;
            let d = quant::q5_0_scale(blk, 0)?;
            let mut i = 0usize;
            while i < quant::Q5_0_QK {
                codes.push((i32::from(c[i]) + 16) as u8);
                weights.push(d * f32::from(c[i]));
                i += 1;
            }
        }
        quant::GGML_TYPE_PQ2_0 => {
            let d = quant::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let mut k = 0usize;
            while k < 32 {
                let byte = blk[2 + k];
                let mut c = 0usize;
                while c < 4 {
                    let code = (byte >> (2 * c)) & 3;
                    codes.push(code);
                    weights.push((f32::from(code) - 1.0) * d);
                    c += 1;
                }
                k += 1;
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Fold one block's codes and weights into `stats`, 32 weights at a time.
#[allow(
    clippy::float_cmp,
    reason = "the property is an exactly-zero dequantized weight; `== 0.0` is the contract here, \
              including -0.0, not an approximate comparison"
)]
fn accumulate(stats: &mut ClassStats, codes: &[u8], weights: &[f32]) {
    let n = codes.len().min(weights.len());
    let mut g = 0usize;
    while (g + 1) * 32 <= n {
        let lo = g * 32;
        let mut all_code = true;
        let mut all_w = true;
        let mut i = lo;
        while i < lo + 32 {
            if codes[i] == 0 {
                stats.code_zero += 1;
            } else {
                all_code = false;
            }
            if weights[i] == 0.0 {
                stats.w_zero += 1;
            } else {
                all_w = false;
            }
            i += 1;
        }
        stats.codes += 32;
        stats.blocks32 += 1;
        if all_code {
            stats.allzero_code += 1;
        }
        if all_w {
            stats.allzero_w += 1;
        }
        g += 1;
    }
}

/// `num / den` as a display ratio; `NaN` for an empty population (never a silent zero).
#[allow(
    clippy::cast_precision_loss,
    reason = "counts are far below 2^53; the quotient is a display fraction, not a token count"
)]
fn ratio(num: u64, den: u64) -> f64 {
    if den == 0 {
        f64::NAN
    } else {
        num as f64 / den as f64
    }
}

/// `"{:.6}"` of `ratio(num, den)`, so an empty population prints `NaN`.
fn fmt_frac(num: u64, den: u64) -> String {
    format!("{:.6}", ratio(num, den))
}

/// Tokens per second for a display line.
#[allow(
    clippy::cast_precision_loss,
    reason = "the token count is below 2^53; this is the reported timing, not a token count"
)]
fn tps(tokens: usize, secs: f64) -> f64 {
    if secs <= 0.0 {
        f64::NAN
    } else {
        tokens as f64 / secs
    }
}

/// One census row: counts plus the four fractions the task names.
fn print_row(label: &str, s: &ClassStats) {
    println!(
        "{label:<16} {:>8} {:>12} {:>14} {:>11} {:>11} {:>10} {:>10}  {}",
        s.tensors,
        s.blocks32,
        s.codes,
        fmt_frac(s.code_zero, s.codes),
        fmt_frac(s.allzero_code, s.blocks32),
        fmt_frac(s.w_zero, s.codes),
        fmt_frac(s.allzero_w, s.blocks32),
        s.types.join("+")
    );
}

/// Read every quantized tensor payload of `path` and print the per-class 32-weight census.
///
/// # Complexity
///
/// `O(sum of tensor elements)` time, `O(1)` extra space beyond one block's codes and weights.
fn weight_census(path: &str) -> Result<(), Box<dyn Error>> {
    let engine = Engine::open(path)?;
    let mut stats: Vec<(String, ClassStats)> = Vec::new();
    let mut skipped: Vec<(String, &'static str)> = Vec::new();
    let mut codes: Vec<u8> = Vec::with_capacity(quant::QK_K);
    let mut weights: Vec<f32> = Vec::with_capacity(quant::QK_K);
    let mut e = 0usize;
    while e < engine.table().len() {
        let t = &engine.table().entries()[e];
        e += 1;
        if is_float_type(t.ggml_type) {
            continue;
        }
        let class = tensor_class(&t.name).to_string();
        if !has_reader(t.ggml_type) {
            let tn = quant::ggml_type_name(t.ggml_type);
            if !skipped.iter().any(|(c, n)| c == &class && n == &tn) {
                skipped.push((class, tn));
            }
            continue;
        }
        let Some(n_bytes) = t.n_bytes else { continue };
        let Some((qk, bb)) = quant::block_geom(t.ggml_type) else {
            continue;
        };
        let raw = engine.payload(t.byte_range.start, n_bytes)?;
        let n_elements =
            usize::try_from(t.n_elements).map_err(|_| "tensor element count overflows usize")?;
        let n_blocks = n_elements / qk;
        let ti = quant::ggml_type_name(t.ggml_type);
        let slot = class_slot(&mut stats, &class);
        stats[slot].1.tensors += 1;
        if !stats[slot].1.types.contains(&ti) {
            stats[slot].1.types.push(ti);
        }
        let mut b = 0usize;
        while b < n_blocks {
            let off = b * bb;
            if off + bb > raw.len() {
                return Err(format!("tensor {}: block {b} leaves the payload", t.name).into());
            }
            codes.clear();
            weights.clear();
            if !decode_block(t.ggml_type, &raw[off..off + bb], &mut codes, &mut weights)? {
                return Err(format!("tensor {}: no reader for its block type", t.name).into());
            }
            accumulate(&mut stats[slot].1, &codes, &weights);
            b += 1;
        }
    }

    let mut totals = ClassStats::default();
    println!();
    println!("== weight census: {path} ==");
    println!(
        "{:<16} {:>8} {:>12} {:>14} {:>11} {:>11} {:>10} {:>10}  types",
        "class", "tensors", "blocks32", "codes", "zero_code", "all0_code", "zero_w", "all0_w"
    );
    let mut i = 0usize;
    while i < stats.len() {
        let (class, s) = &stats[i];
        print_row(class, s);
        totals.merge(s);
        i += 1;
    }
    print_row("TOTAL", &totals);
    let mut i = 0usize;
    while i < skipped.len() {
        println!("skipped (no reader): {} [{}]", skipped[i].0, skipped[i].1);
        i += 1;
    }
    Ok(())
}

// ── part 2: teacher-forced draft acceptance ─────────────────────────────────────────────────────

/// Which draft strategy is being measured.
enum Source {
    /// Prompt-lookup n-gram with the given `n`.
    Ngram(usize),
    /// The online bigram table.
    Bigram,
    /// The uniform control.
    Shuffled,
}

/// `last token -> next token -> count`, built incrementally from the context so far.
type Bigram = HashMap<u32, HashMap<u32, u32>>;

/// A small deterministic xorshift so the control is reproducible across runs.
struct Rng(u64);

impl Rng {
    /// Seed the generator; the low bit is forced set so the state never sticks at zero.
    const fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }

    /// The next `u64` (xorshift64).
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Per-source acceptance counters over the teacher-forced positions.
#[derive(Default, Clone)]
struct AccStats {
    /// Positions where the source proposed at least one token.
    attempted: u64,
    /// Positions where the first proposal equalled the greedy target.
    matched: u64,
    /// Summed longest accepted drafted prefix, per [`K_VALUES`] entry.
    run_sum: [u64; 3],
}

/// Prompt-lookup: match the last `n` history tokens against earlier occurrences in `ids[..=t]` and
/// continue with the follower; repeat up to `k` tokens. The search never reads past `t`, so no
/// draft peeks at the text it predicts.
///
/// # Complexity
///
/// `O(k · n · t)` per call in the worst case (a full suffix scan); the caller runs it once per
/// position per `n`.
#[allow(
    clippy::many_single_char_names,
    reason = "`n` is the n-gram order, `k` the draft length, `s` the occurrence start, `j` the \
              suffix offset; each is the name the recurrence uses"
)]
fn ngram_draft(ids: &[u32], t: usize, n: usize, k: usize) -> Vec<u32> {
    let base = &ids[..=t];
    let mut drafted: Vec<u32> = Vec::with_capacity(k);
    while drafted.len() < k {
        let history_len = base.len() + drafted.len();
        let ns = n.min(history_len);
        if ns == 0 || base.len() <= ns {
            break;
        }
        let suffix_start = history_len - ns;
        let mut hit: Option<usize> = None;
        let mut s = base.len() - ns;
        while s > 0 {
            s -= 1;
            let mut j = 0usize;
            let mut ok = true;
            while j < ns {
                let off = suffix_start + j;
                let tok = if off < base.len() {
                    base[off]
                } else {
                    drafted[off - base.len()]
                };
                if ids[s + j] != tok {
                    ok = false;
                    break;
                }
                j += 1;
            }
            if ok {
                hit = Some(s);
                break;
            }
        }
        match hit {
            Some(s) => drafted.push(ids[s + ns]),
            None => break,
        }
    }
    drafted
}

/// Bigram continuation: repeatedly take the most frequent successor of the current token, breaking
/// count ties toward the lower token id so the draft is deterministic.
///
/// # Complexity
///
/// `O(k · D)` where `D` is the successor-set size of the current token.
#[allow(
    clippy::implicit_hasher,
    reason = "the caller's table is the crate-default std hasher; a generic hasher adds no value to \
              a one-off probe"
)]
fn bigram_draft(bigram: &Bigram, last: u32, k: usize) -> Vec<u32> {
    let mut drafted: Vec<u32> = Vec::with_capacity(k);
    let mut cur = last;
    while drafted.len() < k {
        let Some(m) = bigram.get(&cur) else { break };
        let mut best: Option<(u32, u32)> = None;
        for (&tok, &c) in m {
            best = match best {
                None => Some((c, tok)),
                Some((bc, bt)) => {
                    if c > bc || (c == bc && tok < bt) {
                        Some((c, tok))
                    } else {
                        Some((bc, bt))
                    }
                }
            };
        }
        let Some((_, tok)) = best else { break };
        drafted.push(tok);
        cur = tok;
    }
    drafted
}

/// Uniform shuffled control: `k` draws from `[0, vocab)`. Its agreement rate is `1/vocab`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "`rng % vocab < vocab <= 2^31`, so the narrowing to `u32` loses no bit"
)]
fn shuffled_draft(rng: &mut Rng, k: usize, vocab: usize) -> Vec<u32> {
    let m = vocab as u64;
    let mut drafted: Vec<u32> = Vec::with_capacity(k);
    while drafted.len() < k {
        drafted.push((rng.next_u64() % m) as u32);
    }
    drafted
}

/// Dispatch one source's draft at position `t`.
#[allow(
    clippy::too_many_arguments,
    reason = "the probe passes the read-only context and the mutable rng explicitly; bundling them \
              into a struct would only rename the same arguments"
)]
fn draft_for(
    source: &Source,
    ids: &[u32],
    t: usize,
    targets: &[u32],
    bigram: &Bigram,
    rng: &mut Rng,
    k: usize,
    vocab: usize,
) -> Vec<u32> {
    let _ = targets;
    match source {
        Source::Ngram(n) => ngram_draft(ids, t, *n, k),
        Source::Bigram => bigram_draft(bigram, ids[t], k),
        Source::Shuffled => shuffled_draft(rng, k, vocab),
    }
}

/// `E = (1 - alpha^(k+1)) / (1 - alpha)` under i.i.d. acceptance; `k + 1` at `alpha == 1`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "`k` is 2, 4 or 8, so the `i32` exponent is exact and `k + 1` is exact in `f64`"
)]
fn expected_tokens(alpha: f64, k: usize) -> f64 {
    if alpha >= 1.0 {
        return (k + 1) as f64;
    }
    (1.0 - alpha.powi((k + 1) as i32)) / (1.0 - alpha)
}

/// One acceptance row: coverage, `alpha`, and `E`/empirical mean run for each `k`.
#[allow(
    clippy::cast_precision_loss,
    reason = "counts are far below 2^53; these are display statistics"
)]
fn print_acc_row(name: &str, a: &AccStats, n: u64) {
    let alpha = ratio(a.matched, a.attempted);
    let cov = ratio(a.attempted, n);
    let mut line = format!("{name:<22} {cov:>8.4} {alpha:>10.6}");
    let mut ki = 0usize;
    while ki < K_VALUES.len() {
        let k = K_VALUES[ki];
        let e = expected_tokens(alpha, k);
        let run = ratio(a.run_sum[ki], a.attempted);
        write!(line, " {e:>9.3}/{run:>7.3}").expect("writing to a String cannot fail");
        ki += 1;
    }
    println!("{line}");
}

/// Teacher-force `n_tokens` of `text_path` through `path`, then measure every draft source.
///
/// Returns an error (exit 1) when the shuffled control's `alpha` exceeds [`SHUFFLED_MAX_ALPHA`].
#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    reason = "`vocab` and the token count are below 2^53 on this model; the quotient is the \
              reported `1/vocab` reference, not a token count. The body is one teacher-forced \
              pass followed by the per-position draft loop; `s`/`t`/`k`/`m`/`i` name the \
              source, position, draft length, accepted prefix and loop cursor they index"
)]
fn acceptance(
    path: &str,
    tokenizer: &str,
    text_path: &str,
    n_tokens: usize,
) -> Result<(), Box<dyn Error>> {
    let mode = match std::env::var("APEIRON_KERNEL").as_deref() {
        Ok("q8act") => KernelMode::Q8Act,
        _ => KernelMode::Exact,
    };
    let bpe = Bpe::from_tokenizer_json(tokenizer)?;
    let text = std::fs::read_to_string(text_path)?;
    let all = bpe.encode(&text)?;
    let l = n_tokens.min(all.len());
    if l < 16 {
        return Err(format!("{text_path} tokenizes to only {l} tokens").into());
    }
    let ids: Vec<u32> = all[..l].to_vec();
    let engine = Engine::open(path)?;
    let vocab = engine.config.vocab_size;
    let mut kv = engine.decode_state(l)?;

    let fwd_start = Instant::now();
    let mut targets: Vec<u32> = Vec::with_capacity(l - 1);
    let mut i = 0usize;
    while i < l {
        let logits = engine.step_logits_with_mode(&mut kv, ids[i], mode)?;
        if i + 1 < l {
            targets.push(argmax_fixed(&logits).map_err(|e| format!("{e:?}"))?);
        }
        i += 1;
    }
    let fwd = fwd_start.elapsed();

    let mut sources: Vec<(String, Source)> = Vec::new();
    let mut ns = 0usize;
    while ns < NGRAM_NS.len() {
        sources.push((
            format!("prompt-lookup n={}", NGRAM_NS[ns]),
            Source::Ngram(NGRAM_NS[ns]),
        ));
        ns += 1;
    }
    sources.push(("bigram (online)".to_string(), Source::Bigram));
    sources.push(("shuffled (control)".to_string(), Source::Shuffled));
    let mut accs: Vec<AccStats> = vec![AccStats::default(); sources.len()];

    let mut bigram: Bigram = HashMap::new();
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    let n_pos = targets.len() as u64;
    let mut t = 0usize;
    while t + 1 < l {
        if t >= 1 {
            *bigram
                .entry(ids[t - 1])
                .or_default()
                .entry(ids[t])
                .or_insert(0) += 1;
        }
        let mut s = 0usize;
        while s < sources.len() {
            let draft = draft_for(
                &sources[s].1,
                &ids,
                t,
                &targets,
                &bigram,
                &mut rng,
                K_MAX,
                vocab,
            );
            let a = &mut accs[s];
            if let Some(&first) = draft.first() {
                a.attempted += 1;
                if first == targets[t] {
                    a.matched += 1;
                }
            }
            let mut ki = 0usize;
            while ki < K_VALUES.len() {
                let k = K_VALUES[ki];
                let mut m = 0usize;
                while m < k
                    && m < draft.len()
                    && t + m < targets.len()
                    && draft[m] == targets[t + m]
                {
                    m += 1;
                }
                a.run_sum[ki] += m as u64;
                ki += 1;
            }
            s += 1;
        }
        t += 1;
    }

    println!();
    println!("== draft acceptance: {path} ==");
    println!("tokenizer  {tokenizer}");
    println!("text       {text_path}");
    println!("kernel     {mode:?}");
    println!("tokens     {l}   positions {n_pos}   vocab {vocab}");
    println!(
        "forward    {:.3} s ({:.1} tok/s) [shared host]",
        fwd.as_secs_f64(),
        tps(l, fwd.as_secs_f64())
    );
    println!(
        "{:<22} {:>8} {:>10} {:>18} {:>18} {:>18}",
        "source", "cov", "alpha", "k=2 E/run", "k=4 E/run", "k=8 E/run"
    );
    let mut s = 0usize;
    while s < sources.len() {
        print_acc_row(&sources[s].0, &accs[s], n_pos);
        s += 1;
    }
    println!(
        "alpha = matches / positions with a proposal; cov = proposal coverage; \
         E = (1-alpha^(k+1))/(1-alpha), run = empirical mean accepted drafted prefix over \
         proposing positions, so E ~= 1 + run. Per-position tokens advanced = 1 + cov*run."
    );

    let si = sources
        .iter()
        .position(|(_, src)| matches!(src, Source::Shuffled))
        .ok_or("internal: shuffled source missing")?;
    let alpha = ratio(accs[si].matched, accs[si].attempted);
    let expected = 1.0 / vocab as f64;
    println!(
        "control    shuffled alpha {alpha:.3e}, expected ~1/vocab {expected:.3e}, \
         matches {} of {} positions",
        accs[si].matched, accs[si].attempted
    );
    if alpha.is_nan() || alpha > SHUFFLED_MAX_ALPHA {
        return Err(
            format!("CONTROL FAILED: shuffled alpha {alpha:.6} > {SHUFFLED_MAX_ALPHA}").into(),
        );
    }
    println!("control    PASS (shuffled alpha <= {SHUFFLED_MAX_ALPHA})");
    Ok(())
}

// ── part 3: bound-pruned exact greedy LM head (candidate-set probe) ─────────────────────────────
//
// The `Q8Act` head is `matvec_fixed` over the quantized activation. A summary gives `Â_i` and a
// sound `E_i ≥ |L_i − Â_i|`; with `B_i = ⌊E_i/2^32⌋ + 1` every maximiser of the exact `L` lies in
// `C = {i : Ĉ_i + B_i ≥ max_j(Ĉ_j − B_j)}`, so `argmax` over `C` of `L` is `argmax_fixed(L)`.
// The probe measures `|C|` for five `Q8_0` truncations of Qwen and the `PQ2_0` gap / zero-summary
// bound of Bonsai, rebuilding `dequant(quant_act(x))` with the public kernels (checked per token).

/// One Q32.32 unit.
const ONE_Q32: i128 = 1 << 32;
/// Activation-quantization block width (`chart::ACT_QK`).
const ACT_QK: usize = 32;
/// Byte size of one `Q8_0` block: `f16` scale plus 32 codes.
const Q8_0_BYTES: usize = 34;
/// Byte size of one `PQ2_0` block: `f16` scale plus 128 two-bit codes.
const PQ2_0_BYTES: usize = 34;
/// Host read bandwidth named by the task, for the projected head times.
const HOST_BW: f64 = 82.0e9;
/// The `Q8_0` top-bit truncation widths measured.
const KS: [usize; 5] = [2, 3, 4, 5, 6];
/// Default teacher-forced length of the head-prune instrument.
const PRUNE_TOKENS: usize = 64;

/// `value.rs::round_shift_32`: round-to-nearest `v / 2^32`, ties upward.
#[rustfmt::skip]
const fn round_shift_32(v: i128) -> i128 {
    (v + (1_i128 << 31)) >> 32
}

/// `Engine::eps_fixed`: `round(rms_eps · 2^32)`, refusing a negative, non-finite or out-of-range
/// value.
#[rustfmt::skip]
#[allow(clippy::cast_possible_truncation, reason = "the range guard above the cast")]
fn eps_fixed(rms_eps: f64) -> Option<i64> {
    if !rms_eps.is_finite() || rms_eps < 0.0 { return None; }
    let q = (rms_eps * 4_294_967_296.0).round();
    if !(q.is_finite() && (0.0..9_223_372_036_854_775_808.0).contains(&q)) { return None; }
    Some(q as i64)
}

/// Checked `x[i] += y[i]` (`step.rs::add_assign_fixed`).
#[rustfmt::skip]
fn add_assign_fixed(x: &mut [i64], y: &[i64]) -> Result<(), String> {
    if x.len() != y.len() { return Err("residual row width mismatch".to_string()); }
    let mut i = 0usize;
    while i < x.len() {
        x[i] = x[i].checked_add(y[i]).ok_or("fixed-point add overflow")?;
        i += 1;
    }
    Ok(())
}

/// `rms_norm_fixed` followed by the per-channel Q32.32 gain (`step.rs::rms_with_gain_slice`).
#[rustfmt::skip]
fn rms_with_gain(gain: &[i64], x: &[i64], eps: i64) -> Result<Vec<i64>, String> {
    if gain.len() != x.len() { return Err("norm gain width mismatch".to_string()); }
    let base = rms_norm_fixed(x, eps).map_err(|e| format!("{e:?}"))?;
    let mut out: Vec<i64> = Vec::with_capacity(base.len());
    let mut i = 0usize;
    while i < base.len() {
        let prod = i128::from(base[i]) * i128::from(gain[i]);
        out.push(i64::try_from(round_shift_32(prod)).map_err(|_| "gain overflow".to_string())?);
        i += 1;
    }
    Ok(out)
}

/// Per-head QK `RMSNorm` (`loader.rs::qk_norm_heads`); `None` gain leaves `x` unchanged.
#[rustfmt::skip]
fn qk_norm_heads(gain: Option<&[i64]>, x: &mut [i64], hd: usize, eps: i64) -> Result<(), String> {
    let Some(gain) = gain else { return Ok(()) };
    if hd == 0 || gain.len() != hd || !x.len().is_multiple_of(hd) {
        return Err("qk norm geometry".to_string());
    }
    let mut h = 0usize;
    while h < x.len() / hd {
        let base = h * hd;
        let normed = rms_norm_fixed(&x[base..base + hd], eps).map_err(|e| format!("{e:?}"))?;
        let mut i = 0usize;
        while i < hd {
            let prod = i128::from(normed[i]) * i128::from(gain[i]);
            x[base + i] = i64::try_from(round_shift_32(prod)).map_err(|_| "qk norm ovf".to_string())?;
            i += 1;
        }
        h += 1;
    }
    Ok(())
}

/// `chart/mod.rs::quant_act`: one scale `S_b = ceil(absmax_b/127)` per 32-wide block and
/// round-to-nearest int8 codes, with the `dequantizes` guard that refuses a scale-code product
/// outside `i64`. Returns `(scales, codes)`, or `None` on refusal.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap,
        reason = "`scale_wide <= i64::MAX` is checked, and `|signed| <= 127` before the i8 cast")]
#[rustfmt::skip]
fn act_quantize(x: &[i64]) -> Option<(Vec<i64>, Vec<i8>)> {
    let mut scales: Vec<i64> = Vec::with_capacity(x.len().div_ceil(ACT_QK));
    let mut codes: Vec<i8> = Vec::with_capacity(x.len());
    let mut block = 0usize;
    while block < x.len().div_ceil(ACT_QK) {
        let base = block * ACT_QK;
        let take = ACT_QK.min(x.len() - base);
        let mut absmax: u64 = 0;
        let mut lane = 0usize;
        while lane < take { absmax = absmax.max(x[base + lane].unsigned_abs()); lane += 1; }
        let scale_wide = absmax.div_ceil(127);
        if scale_wide > i64::MAX.unsigned_abs() { return None; }
        scales.push(scale_wide as i64);
        let mut lane = 0usize;
        while lane < take {
            let value = x[base + lane];
            if scale_wide == 0 { codes.push(0); lane += 1; continue; }
            let m = value.unsigned_abs();
            let magnitude = m / scale_wide + u64::from(2 * (m % scale_wide) >= scale_wide);
            let signed = if value < 0 { -i128::from(magnitude) } else { i128::from(magnitude) };
            if !(-127..=127).contains(&signed) { return None; }
            codes.push(signed as i8);
            lane += 1;
        }
        block += 1;
    }
    let mut j = 0usize;
    while j < codes.len() {
        scales[j / ACT_QK].checked_mul(i64::from(codes[j]))?;
        j += 1;
    }
    Some((scales, codes))
}

/// `QuantAct::dequant`: `x̂_j = S_b · xq_j` in Q32.32.
#[rustfmt::skip]
fn act_dequant(scales: &[i64], codes: &[i8]) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::with_capacity(codes.len());
    let mut j = 0usize;
    while j < codes.len() { out.push(scales[j / ACT_QK] * i64::from(codes[j])); j += 1; }
    out
}

/// [`matvec_fixed`] over contiguous whole-row chunks on `threads` scoped threads; rows are
/// independent, so the concatenation is bit-identical to the unchunked call.
#[rustfmt::skip]
fn matvec_rows_par(ggml_type: u32, raw: &[u8], n_out: usize, n_in: usize, x: &[i64], threads: usize)
    -> Result<Vec<i64>, String> {
    let (qk, bb) = quant::block_geom(ggml_type).ok_or("no quant geometry")?;
    let row_bytes = n_in.div_ceil(qk).checked_mul(bb).ok_or("row stride overflow")?;
    let chunk = n_out.div_ceil(threads.max(1).min(n_out.max(1)));
    let mut out: Vec<i64> = vec![0i64; n_out];
    let failure: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
    thread::scope(|s| {
        for (ci, slot) in out.chunks_mut(chunk).enumerate() {
            let start = ci * chunk;
            let raw_slice = &raw[start * row_bytes..(start + slot.len()) * row_bytes];
            let failure = &failure;
            s.spawn(move || match matvec_fixed(ggml_type, raw_slice, slot.len(), n_in, x) {
                Ok(v) => slot.copy_from_slice(&v),
                Err(e) => {
                    let mut g = failure.lock().expect("probe mutex");
                    if g.is_none() { *g = Some(format!("{e:?}")); }
                }
            });
        }
    });
    match failure.into_inner().expect("probe mutex") {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// One `Q8Act` projection: quantize `x`, dequantize, and apply the public [`matvec_fixed`].
#[rustfmt::skip]
fn proj_q8(engine: &Engine, name: &str, n_out: usize, n_in: usize, x: &[i64], threads: usize)
    -> Result<Vec<i64>, String> {
    let t = engine.table().get(name).ok_or_else(|| format!("no tensor {name}"))?;
    let n = t.n_bytes.ok_or_else(|| format!("no extent {name}"))?;
    let raw = engine.payload(t.byte_range.start, n).map_err(|e| format!("{name}: {e:?}"))?;
    let (scales, codes) = act_quantize(x).ok_or_else(|| format!("{name}: activation refused"))?;
    matvec_rows_par(t.ggml_type, raw, n_out, n_in, &act_dequant(&scales, &codes), threads)
}

/// The `Q8Act` forward's final normed residual, composing the public kernels as
/// `step.rs::compose_logits` does under [`KernelMode::Q8Act`] (every projection input quantized).
/// `kv` advances by one position.
#[rustfmt::skip]
fn mirror_head_input(engine: &Engine, kv: &mut KvState, id: u32, threads: usize)
    -> Result<Vec<i64>, String> {
    let cfg = &engine.config;
    let eps = eps_fixed(cfg.rms_eps).ok_or("bad rms_eps")?;
    let vectors = engine.decode_vectors();
    let rope = engine.rope_table().map_err(|e| format!("{e:?}"))?;
    let pos = u32::try_from(kv.occupied(0).map_err(|e| format!("{e:?}"))?)
        .map_err(|_| "position overflow".to_string())?;
    let (hd, n_head) = (cfg.head_dim, cfg.n_head);
    let (q_out, kv_out) = (n_head * hd, cfg.n_head_kv * hd);
    let mut hidden = engine.token_embedding(id).map_err(|e| format!("{e:?}"))?;
    let mut layer = 0usize;
    while layer < cfg.n_layer {
        let p = |part: &str, n_out: usize, n_in: usize, x: &[i64]| {
            proj_q8(engine, &format!("blk.{layer}.{part}.weight"), n_out, n_in, x, threads)
        };
        let gain = vectors.norms_of(layer).and_then(|n| n[0].as_deref()).ok_or("attn_norm")?;
        let normed = rms_with_gain(gain, &hidden, eps)?;
        let mut query = p("attn_q", q_out, cfg.n_embd, &normed)?;
        let mut key = p("attn_k", kv_out, cfg.n_embd, &normed)?;
        let mut value = p("attn_v", kv_out, cfg.n_embd, &normed)?;
        for (vec, slot) in [(&mut query, 0usize), (&mut key, 1), (&mut value, 2)] {
            if let Some(b) = vectors.biases_of(layer).and_then(|b| b[slot].as_deref()) {
                add_assign_fixed(vec, b)?;
            }
        }
        let qk = vectors.qk_norms_of(layer);
        qk_norm_heads(qk.and_then(|n| n[0].as_deref()), &mut query, hd, eps)?;
        qk_norm_heads(qk.and_then(|n| n[1].as_deref()), &mut key, hd, eps)?;
        rope.rotate_one(&mut query, pos).map_err(|e| format!("{e:?}"))?;
        rope.rotate_one(&mut key, pos).map_err(|e| format!("{e:?}"))?;
        let appended = kv.append(layer, &key, &value).map_err(|e| format!("{e:?}"))?;
        let mut att: Vec<i64> = vec![0i64; q_out];
        let mut h = 0usize;
        while h < n_head {
            let (s, e) = (h * hd, h * hd + hd);
            kv.attn_head(layer, h, &query[s..e], appended, &mut att[s..e])
                .map_err(|er| format!("{er:?}"))?;
            h += 1;
        }
        let out = p("attn_output", cfg.n_embd, cfg.n_embd, &att)?;
        add_assign_fixed(&mut hidden, &out)?;
        let fgain = vectors.norms_of(layer).and_then(|n| n[1].as_deref()).ok_or("ffn_norm")?;
        let h2 = rms_with_gain(fgain, &hidden, eps)?;
        let gate = p("ffn_gate", cfg.n_ff, cfg.n_embd, &h2)?;
        let up = p("ffn_up", cfg.n_ff, cfg.n_embd, &h2)?;
        let mut act: Vec<i64> = vec![0i64; cfg.n_ff];
        let mut i = 0usize;
        while i < cfg.n_ff {
            let s = silu_fixed(gate[i]).map_err(|e| format!("{e:?}"))?;
            act[i] = i64::try_from(round_shift_32(i128::from(s) * i128::from(up[i])))
                .map_err(|_| "swiglu overflow".to_string())?;
            i += 1;
        }
        let down = p("ffn_down", cfg.n_embd, cfg.n_ff, &act)?;
        add_assign_fixed(&mut hidden, &down)?;
        layer += 1;
    }
    rms_with_gain(vectors.output_norm.as_deref().ok_or("output_norm")?, &hidden, eps)
}

/// The head payload, its row count and byte stride.
#[rustfmt::skip]
fn head_payload(engine: &Engine, row_bytes: usize) -> Result<(Vec<u8>, usize), String> {
    let t = engine.lm_head();
    let n = t.n_bytes.ok_or("head has no extent")?;
    let raw = engine.payload(t.byte_range.start, n).map_err(|e| format!("{e:?}"))?.to_vec();
    let vocab = engine.config.vocab_size;
    if raw.len() < vocab * row_bytes { return Err("short head payload".to_string()); }
    Ok((raw, vocab))
}

/// The `Q8_0` head's static summary metadata: per block the exact `f16` scale and, per truncation
/// width, the largest low-bit magnitude `m_ib = max_j |q_ij − trunc_k(q_ij)|`.
struct Q8Head {
    vocab: usize,
    blocks: usize,
    row_bytes: usize,
    scales: Vec<i64>,
    mres: Vec<[u8; 5]>,
    raw: Vec<u8>,
}

#[allow(clippy::too_many_lines, clippy::cast_possible_truncation, clippy::cast_possible_wrap,
        clippy::cast_lossless, clippy::cast_sign_loss, clippy::many_single_char_names,
        reason = "the extraction-shaped loops name their counters; the casts are code fields <= 8 bits")]
#[rustfmt::skip]
impl Q8Head {
    /// Read the `Q8_0` head payload and the per-block metadata.
    fn build(engine: &Engine) -> Result<Q8Head, String> {
        let blocks = engine.config.n_embd.div_ceil(ACT_QK);
        let row_bytes = blocks * Q8_0_BYTES;
        let (raw, vocab) = head_payload(engine, row_bytes)?;
        let mut scales: Vec<i64> = Vec::with_capacity(vocab * blocks);
        let mut mres: Vec<[u8; 5]> = Vec::with_capacity(vocab * blocks);
        let mut i = 0usize;
        while i < vocab {
            let row = &raw[i * row_bytes..(i + 1) * row_bytes];
            let mut b = 0usize;
            while b < blocks {
                let blk = &row[b * Q8_0_BYTES..(b + 1) * Q8_0_BYTES];
                let bits = u16::from_le_bytes([blk[0], blk[1]]);
                scales.push(f16_to_fixed(bits).map_err(|e| format!("head scale: {e:?}"))?);
                let codes = quant::q8_0_codes(blk, 0).map_err(|e| format!("{e:?}"))?;
                let mut m = [0u8; 5];
                let mut lane = 0usize;
                while lane < ACT_QK {
                    let q = i32::from(codes[lane]);
                    let mut ki = 0usize;
                    while ki < KS.len() {
                        let shift = 8 - KS[ki];
                        m[ki] = m[ki].max((q - ((q >> shift) << shift)).unsigned_abs() as u8);
                        ki += 1;
                    }
                    lane += 1;
                }
                mres.push(m);
                b += 1;
            }
            i += 1;
        }
        Ok(Q8Head { vocab, blocks, row_bytes, scales, mres, raw })
    }

    /// Per width `k = KS[ki]`: the summary logit `Ĉ_i` (Q32.32, rounded) and the two accumulator
    /// bounds, `E_i = Σ_b d_ib·r_k·A_b` (`r_k = 2^(8−k) − 1`) and `E_i = Σ_b d_ib·m_ib·A_b`.
    fn prune(&self, xhat: &[i64], ab: &[i128], ki: usize, threads: usize)
        -> Vec<(i128, i128, i128)> {
        let shift = 8 - KS[ki];
        let r_k = i128::from((1_i64 << shift) - 1);
        let mut trunc = [0i64; 256];
        let mut q = 0usize;
        while q < 256 { trunc[q] = i64::from(((q as u8) as i8 as i32 >> shift) << shift); q += 1; }
        let (blocks, row_bytes, vocab) = (self.blocks, self.row_bytes, self.vocab);
        let mut out: Vec<(i128, i128, i128)> = vec![(0, 0, 0); vocab];
        let chunk = vocab.div_ceil(threads.max(1));
        thread::scope(|s| {
            for (ci, slot) in out.chunks_mut(chunk).enumerate() {
                let row0 = ci * chunk;
                let trunc = &trunc;
                let (scales, mres, raw) = (&self.scales, &self.mres, &self.raw);
                s.spawn(move || {
                    for (ri, o) in slot.iter_mut().enumerate() {
                        let i = row0 + ri;
                        let row = &raw[i * row_bytes..(i + 1) * row_bytes];
                        let (mut acc, mut e_worst, mut e_byte) = (0i128, 0i128, 0i128);
                        let mut b = 0usize;
                        while b < blocks {
                            let blk = &row[b * Q8_0_BYTES..(b + 1) * Q8_0_BYTES];
                            let codes = quant::q8_0_codes(blk, 0).expect("prechecked payload");
                            let base = b * ACT_QK;
                            let mut p: i128 = 0;
                            let mut lane = 0usize;
                            while lane < ACT_QK {
                                p += i128::from(trunc[codes[lane] as u8 as usize])
                                    * i128::from(xhat[base + lane]);
                                lane += 1;
                            }
                            let d = i128::from(scales[i * blocks + b]);
                            acc += d * p;
                            e_worst += d * r_k * ab[b];
                            e_byte += d * i128::from(mres[i * blocks + b][ki]) * ab[b];
                            b += 1;
                        }
                        *o = (round_shift_32(acc), e_worst, e_byte);
                    }
                });
            }
        });
        out
    }
}

/// The `PQ2_0` head's static metadata: per 128-weight block the exact `f16` scale.
struct Pq2Head {
    blocks: usize,
    row_bytes: usize,
    scales: Vec<i64>,
}

#[rustfmt::skip]
impl Pq2Head {
    /// Read the `PQ2_0` head payload and its block scales.
    fn build(engine: &Engine) -> Result<Pq2Head, String> {
        let blocks = engine.config.n_embd.div_ceil(quant::PQ2_0_QK);
        let row_bytes = blocks * PQ2_0_BYTES;
        let (raw, vocab) = head_payload(engine, row_bytes)?;
        let mut scales: Vec<i64> = Vec::with_capacity(vocab * blocks);
        let mut i = 0usize;
        while i < vocab {
            let row = &raw[i * row_bytes..(i + 1) * row_bytes];
            let mut b = 0usize;
            while b < blocks {
                let blk = &row[b * PQ2_0_BYTES..(b + 1) * PQ2_0_BYTES];
                let bits = u16::from_le_bytes([blk[0], blk[1]]);
                scales.push(f16_to_fixed(bits).map_err(|e| format!("head scale: {e:?}"))?);
                b += 1;
            }
            i += 1;
        }
        Ok(Pq2Head { blocks, row_bytes, scales })
    }

    /// The zero-summary Hölder bound `Σ_b d_ib·A_b` per row, `A_b` summed over the four 32-wide
    /// activation blocks that tile one 128-weight block.
    fn zero_bounds(&self, ab: &[i128], threads: usize) -> Vec<i128> {
        let subs = quant::PQ2_0_QK / ACT_QK;
        let mut out: Vec<i128> = vec![0; self.scales.len() / self.blocks];
        let chunk = out.len().div_ceil(threads.max(1));
        let (blocks, scales) = (self.blocks, &self.scales);
        thread::scope(|s| {
            for (ci, slot) in out.chunks_mut(chunk).enumerate() {
                let row0 = ci * chunk;
                s.spawn(move || {
                    for (ri, o) in slot.iter_mut().enumerate() {
                        let i = row0 + ri;
                        let mut e: i128 = 0;
                        let mut b = 0usize;
                        while b < blocks {
                            let a: i128 = ab[b * subs..(b + 1) * subs].iter().sum();
                            e += i128::from(scales[i * blocks + b]) * a;
                            b += 1;
                        }
                        *o = e;
                    }
                });
            }
        });
        out
    }
}

/// `(median, p90, max)` of a sample; `NaN`s for an empty sample.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss,
        reason = "display quantiles over counts below 2^53")]
#[rustfmt::skip]
fn dist(v: &[f64]) -> (f64, f64, f64) {
    if v.is_empty() { return (f64::NAN, f64::NAN, f64::NAN); }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = |p: f64| ((p * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1;
    (s[idx(0.5)], s[idx(0.9)], s[s.len() - 1])
}

/// `argmax_fixed` over the ascending candidate indices `cand`, using the exact logits `l`.
#[rustfmt::skip]
fn argmax_in_candidates(l: &[i64], cand: &[usize]) -> usize {
    let mut best: Option<usize> = None;
    for &i in cand { if best.is_none_or(|b| l[i] > l[b]) { best = Some(i); } }
    best.expect("the maximiser is a candidate")
}

/// One head-prune measurement row.
struct PruneRow {
    label: String,
    cand: Vec<f64>,
    summary_bytes: usize,
    violations: u64,
    control_half: u64,
    control_slack: u64,
    max_frac: f64,
    argmax_mismatch: u64,
}

#[rustfmt::skip]
impl PruneRow {
    fn new(label: String, summary_bytes: usize) -> PruneRow {
        PruneRow { label, cand: Vec::new(), summary_bytes, violations: 0, control_half: 0,
                   control_slack: 0, max_frac: 0.0, argmax_mismatch: 0 }
    }
}

/// The opt-in bound-pruned LM-head instrument over the teacher-forced `README.md` run.
#[allow(clippy::too_many_lines, clippy::many_single_char_names, clippy::cast_precision_loss,
        clippy::cast_possible_truncation, clippy::cast_sign_loss,
        reason = "one teacher-forced pass plus the per-summary tables; counts are below 2^53")]
#[rustfmt::skip]
fn head_prune(path: &str, tokenizer: &str, text_path: &str, n_tokens: usize, threads: usize)
    -> Result<(), Box<dyn Error>> {
    let mode = KernelMode::Q8Act;
    let bpe = Bpe::from_tokenizer_json(tokenizer)?;
    let text = std::fs::read_to_string(text_path)?;
    let all = bpe.encode(&text)?;
    let l = n_tokens.min(all.len());
    if l < 4 { return Err(format!("{text_path} tokenizes to only {l} tokens").into()); }
    let ids: Vec<u32> = all[..l].to_vec();
    let engine = Engine::open(path)?;
    let (vocab, n_embd) = (engine.config.vocab_size, engine.config.n_embd);
    let head = engine.lm_head();
    let (htype, hname) = (head.ggml_type, head.name.clone());
    let q8 = if htype == quant::GGML_TYPE_Q8_0 { Some(Q8Head::build(&engine)?) } else { None };
    let pq2 = if htype == quant::GGML_TYPE_PQ2_0 { Some(Pq2Head::build(&engine)?) } else { None };
    if q8.is_none() && pq2.is_none() {
        return Err(format!("head {hname}: no prune summary for ggml type {htype}").into());
    }
    let mut rows: Vec<PruneRow> = Vec::new();
    if let Some(h) = &q8 {
        for &k in &KS {
            rows.push(PruneRow::new(format!("(a) k={k} worst r_k"), (2 + 4 * k) * h.blocks));
            rows.push(PruneRow::new(format!("(b) k={k} max byte"), (3 + 4 * k) * h.blocks));
        }
    }
    let (mut kv_ref, mut kv_mirror) = (engine.decode_state(l)?, engine.decode_state(l)?);
    let (mut gaps, mut zero_bounds, mut argmax_row) = (Vec::new(), Vec::new(), Vec::new());
    let (mut mirror_mismatch, mut zero_viol, mut zero_half, mut zero_slack) = (0u64, 0u64, 0u64, 0u64);

    let fwd_start = Instant::now();
    let mut t = 0usize;
    while t < l {
        let lref = engine.step_logits_with_mode(&mut kv_ref, ids[t], mode)?;
        let xf = mirror_head_input(&engine, &mut kv_mirror, ids[t], threads)?;
        let (scales, codes) = act_quantize(&xf).ok_or("mirror activation refused")?;
        let xhat = act_dequant(&scales, &codes);
        if engine.lm_head_fixed(&xhat).map_err(|e| format!("{e:?}"))? != lref { mirror_mismatch += 1; }
        let mut ab: Vec<i128> = vec![0; n_embd.div_ceil(ACT_QK)];
        let mut j = 0usize;
        while j < xhat.len() { ab[j / ACT_QK] += i128::from(xhat[j]).abs(); j += 1; }
        let mut sorted = lref.clone();
        sorted.sort_unstable();
        gaps.push((sorted[sorted.len() - 1] - sorted[sorted.len() - 2]) as f64);

        if let Some(h) = &q8 {
            for ki in 0..KS.len() {
                let res = h.prune(&xhat, &ab, ki, threads);
                for (mi, use_byte) in [(0usize, false), (1usize, true)] {
                    let e = |x: (i128, i128, i128)| if use_byte { x.2 } else { x.1 };
                    let max_low = res.iter().map(|&x| x.0 - (e(x) / ONE_Q32 + 1)).max().unwrap_or(0);
                    let (mut cand, mut viol, mut half, mut slack, mut max_frac) =
                        (Vec::<usize>::new(), 0u64, 0u64, 0u64, 0.0f64);
                    for (i, &x) in res.iter().enumerate() {
                        let b = e(x) / ONE_Q32 + 1;
                        let err = (i128::from(lref[i]) - x.0).abs();
                        viol += u64::from(err > b);
                        half += u64::from(err > e(x) / 2 / ONE_Q32 + 1);
                        slack += u64::from(err > 1);
                        let frac = err as f64 / b.max(1) as f64;
                        max_frac = max_frac.max(frac);
                        if x.0 + b >= max_low { cand.push(i); }
                    }
                    let r = &mut rows[ki * 2 + mi];
                    r.cand.push(cand.len() as f64);
                    (r.violations, r.control_half, r.control_slack) = (r.violations + viol, r.control_half + half, r.control_slack + slack);
                    r.max_frac = r.max_frac.max(max_frac);
                    if argmax_in_candidates(&lref, &cand) as u32
                        != argmax_fixed(&lref).map_err(|er| format!("{er:?}"))? {
                        r.argmax_mismatch += 1;
                    }
                }
            }
        }
        if let Some(p) = &pq2 {
            let e0 = p.zero_bounds(&ab, threads);
            for (i, &v) in e0.iter().enumerate() {
                zero_bounds.push((v / ONE_Q32) as f64);
                let err = i128::from(lref[i]).abs();
                zero_viol += u64::from(err > v / ONE_Q32 + 1);
                zero_half += u64::from(err > v / 2 / ONE_Q32 + 1);
                zero_slack += u64::from(err > 1);
            }
            let top = argmax_fixed(&lref).map_err(|er| format!("{er:?}"))? as usize;
            argmax_row.push((e0[top] / ONE_Q32) as f64);
        }
        t += 1;
    }
    let fwd = fwd_start.elapsed();

    println!();
    println!("== bound-pruned LM head: {path} ==");
    println!("text       {text_path}   tokens {l}   threads {threads}   kernel {mode:?} [shared host]");
    println!("head       {hname} {} rows {vocab} width {n_embd}", quant::ggml_type_name(htype));
    println!("mirror     Q8Act forward; logits differ from engine on {mirror_mismatch} of {l} tokens");
    let (g50, g90, gmax) = dist(&gaps);
    println!("gap        exact top-1 - top-2 logit (Q32.32): median {g50:.1}  p90 {g90:.1}  max {gmax:.1}");
    println!("rounding   B_i = floor(E_i/2^32) + 1; C = {{ i : Ĉ_i + B_i >= max_j (Ĉ_j - B_j) }}");
    if let Some(h) = &q8 {
        let (row_bytes, full_ms) = (h.row_bytes, (vocab * h.row_bytes) as f64 / HOST_BW * 1e3);
        println!();
        println!("{:<20} {:>7} {:>7} {:>7} {:>8} {:>8} {:>9} {:>9} {:>6} {:>6} {:>6} {:>6}",
                 "summary", "med|C|", "p90", "max", "med/vocab", "sumB/row", "proj ms",
                 "full ms", "viol", "ctl1/2", "ctl->1", "err/B");
        for r in &rows {
            let (m, p, x) = dist(&r.cand);
            let proj = (vocab as f64 * r.summary_bytes as f64 + m * row_bytes as f64) / HOST_BW * 1e3;
            println!("{:<20} {:>7.0} {:>7.0} {:>7.0} {:>8.4} {:>8} {:>9.3} {:>9.3} {:>6} {:>6} {:>6} {:>6.3}",
                     r.label, m, p, x, m / vocab as f64, r.summary_bytes, proj, full_ms,
                     r.violations, r.control_half, r.control_slack, r.max_frac);
        }
        println!("row bytes  full Q8_0 row {row_bytes}; proj = (vocab·sumB/row + |C|·row bytes)/82.0 GB/s");
        println!("mirror     {:.1} s for {l} tokens [shared host]", fwd.as_secs_f64());
        println!("viol = |L_i - Ĉ_i| > B_i (must be 0); ctl = the same count with E_i halved / dropped");
    }
    if let Some(p) = &pq2 {
        let (m, p90, x) = dist(&zero_bounds);
        let (am, _, _) = dist(&argmax_row);
        println!();
        println!("PQ2_0      row bytes {}   zero-summary Holder bound sum_b d_ib*A_b (Q32.32):", p.row_bytes);
        println!("           median {m:.1}  p90 {p90:.1}  max {x:.1}   (argmax-row median {am:.1})");
        println!("           soundness |L_i| <= floor(E0_i/2^32)+1: violations {zero_viol} of {}; ctl1/2 {zero_half}, ctl->1 {zero_slack}",
                 zero_bounds.len());
        println!("           a zero summary prunes nothing: the bound exceeds the measured gaps");
    }

    if mirror_mismatch > 0 {
        return Err(format!("SOUNDNESS FAILED: mirror != engine on {mirror_mismatch} tokens").into());
    }
    let bad: usize = rows.iter().map(|r| (r.violations + r.argmax_mismatch) as usize).sum();
    let short = rows.iter().filter(|r| r.cand.len() != l).count();
    if bad + short + zero_viol as usize > 0 {
        return Err(format!("SOUNDNESS FAILED: {} violations", bad + short + zero_viol as usize).into());
    }
    let ctl_half: u64 = zero_half + rows.iter().map(|r| r.control_half).sum::<u64>();
    let ctl_slack: u64 = zero_slack + rows.iter().map(|r| r.control_slack).sum::<u64>();
    if ctl_half == 0 && !rows.is_empty() {
        let max_frac = rows.iter().map(|r| r.max_frac).fold(0.0f64, f64::max);
        let shift = (1.0f64 / max_frac).log2().ceil() as u32;
        println!("control    E_i/2 gave 0 violations: E_i is loose, the error uses at most {max_frac:.4} of B_i, so the first E_i >> s that fires has s = {shift}");
    }
    if ctl_slack == 0 {
        return Err("CONTROL FAILED: neither the halved nor the E=0 bound produced a violation".into());
    }
    println!("control    PASS: E_i dropped -> {ctl_slack} violations (seeded bound defect fires); E_i/2 -> {ctl_half}");
    Ok(())
}

// ── entry point ─────────────────────────────────────────────────────────────────────────────────

/// Run both instruments over the selected model(s).
fn run() -> Result<(), Box<dyn Error>> {
    let opts = parse_args()?;
    let mut models: Vec<(&str, &str, &str)> = Vec::new();
    match opts.only.as_deref() {
        Some("qwen") => models.push(("qwen05b", DEFAULT_QWEN, DEFAULT_QWEN_TOKENIZER)),
        Some("bonsai") => models.push(("bonsai", DEFAULT_BONSAI, DEFAULT_BONSAI_TOKENIZER)),
        _ => {
            models.push(("qwen05b", DEFAULT_QWEN, DEFAULT_QWEN_TOKENIZER));
            models.push(("bonsai", DEFAULT_BONSAI, DEFAULT_BONSAI_TOKENIZER));
        }
    }
    let mut m = 0usize;
    while m < models.len() {
        let (name, model, tokenizer) = models[m];
        println!();
        println!("###### {name} ######");
        if !opts.skip_weights {
            weight_census(model)?;
        }
        if !opts.skip_accept {
            acceptance(model, tokenizer, &opts.text, opts.tokens)?;
        }
        if opts.prune {
            head_prune(
                model,
                tokenizer,
                &opts.text,
                opts.prune_tokens,
                opts.threads,
            )?;
        }
        m += 1;
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("decode_probe: {e}");
            ExitCode::FAILURE
        }
    }
}
