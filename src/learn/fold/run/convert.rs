//! The encoding boundary between the engine's `Q32.32` carrier
//! ([`crate::learn::grad::model::Weights`]) and the fold driver's word weights
//! ([`crate::learn::fold::layer::driver::LayerWeights`]), plus the `GGUF` scalar payloads the
//! transfer run writes: a weight is an `i64` scaled by `2^-32`. [`teacher_from_grad`] lifts a loaded
//! [`Weights`] into a [`Teacher`] the driver consumes with no numeric conversion, [`student_to_grad`]
//! lowers the fitted word layers back, and every elementwise pass over a whole tensor is the one
//! [`Pass`] map split across the engine pool — one contiguous chunk per participant, bit-identical
//! to the serial map.
//!
//! The loader decodes an `f16`/`bf16`/`f32` payload to the same words once, round half-even
//! ([`decode_f16`], [`decode_bf16`], [`decode_f32`]); the writer re-encodes words
//! ([`encode_f16`], [`encode_f32`]). No `f64` buffer carries a weight on this path.
//!
//! ```text
//! q32_to_f64(v) = v / 2^32                 # exact for |v| < 2^21
//! f64_to_q32(x) = RNE(x · 2^32) : i64      # exact decode of (sign, exponent, significand)
//! ```
//!
//! so `f64_to_q32(q32_to_f64(v)) == v` for every `Q32.32` word and `|f64_to_q32(x)/2^32 - x| <=
//! 2^-33` for any admissible `x`; a tie at the grid quantum rounds to even.
//!
//! # Evidence grade
//!
//! `tested`: the gate's `cargo test` on the build path (`tests/learn_fold_run_convert.rs`). No Lean
//! refinement links this file; it is not on the extracted fragment, so nothing here is `proved`.
//!
//! # Trust boundary
//!
//! Not covered: the `GGUF` load ([`Weights::from_engine`], the engine's contract), the `GGUF` header
//! write, the driver's fit arithmetic, and rustc/LLVM codegen. No `unsafe`, and no `f64` multiply
//! or `f64` rounding on the computing path.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "the `Q32.32 → f64` scalar boundary (`v as f64` is the documented exact scalar for \
              `|v| < 2^21`), the bit decode's `m`/`e`/`k` significand-exponent-shift symbols, and \
              narrowing casts bound-checked at the site (`e < 2^11`, a biased exponent in `1..=30`, \
              a rounded significand below `2048`)"
)]

use crate::engine::chart::pool::{chunks_mut_global, width_global, FirstErr};
use crate::engine::chart::task::ChunkTask;
use crate::engine::f16_to_fixed;
use crate::engine::fp::F32;
use crate::learn::fold::stats::parts::LayerCfg;
use crate::learn::grad::model::{Config, LayerWeights, Weights};
use crate::learn::solve::q32::{round_f16, rshift_rne, Fx};

/// The `Q32.32` scale `2^32`.
const SCALE: f64 = 4_294_967_296.0;

/// The biased `f64` exponent of `2^31` (`1023 + 31`); `0x7ff` (`NaN`/infinity) is checked first.
const EXP_2_31: u64 = 1054;

/// A refusal from the engine-weights/fold-driver conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConvertError {
    /// A tensor length disagrees with the geometry the other side declares.
    Shape(String),
    /// A `Q→f64` conversion needs a `Qwen2` attention bias the loaded model does not carry.
    MissingBias(String),
    /// A finite `f64` is out of the `Q32.32` range (`|x| >= 2^31`), or a `Q32.32` word has no
    /// finite binary16 encoding.
    Range(String),
    /// A `f64` is `NaN` or infinite and cannot be quantized.
    NonFinite(String),
}

impl core::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Shape(m) => write!(f, "shape: {m}"),
            Self::MissingBias(m) => write!(f, "missing bias: {m}"),
            Self::Range(m) => write!(f, "range: {m}"),
            Self::NonFinite(m) => write!(f, "non-finite: {m}"),
        }
    }
}

impl std::error::Error for ConvertError {}

/// The teacher the fold driver consumes: the per-layer weights in the driver's layout, the final
/// `RMSNorm` gain and the token embedding — all `Q32.32` words — plus the geometry.
#[derive(Clone, Debug)]
pub struct Teacher {
    /// The shared per-layer geometry (all layers have the same shape in `Qwen2`).
    pub layer_cfg: LayerCfg,
    /// One entry per trunk layer, in the driver's [`LayerWeights`] layout.
    pub layers: Vec<LayerWeights>,
    /// The final `RMSNorm` gain `output_norm.weight`, length `hidden`.
    pub fnorm: Vec<i64>,
    /// The token embedding `token_embd.weight`, row-major `[vocab, hidden]`.
    pub embed: Vec<i64>,
    /// Vocabulary size.
    pub vocab: usize,
    /// Trunk layer count.
    pub n_layer: usize,
}

/// `v / 2^32`, the `Q32.32` word as an `f64`. Exact for `|v| < 2^21`.
///
/// # Complexity
/// `O(1)`.
#[must_use]
pub fn q32_to_f64(v: i64) -> f64 {
    v as f64 / SCALE
}

/// `RNE(x · 2^32)` as an `i64` by an integer decode of the `f64` bits: the exponent field bounds
/// `|x| < 2^31`, `m` is the 53-bit significand shifted onto the `2^-32` grid (`x·2^32 = ±m·2^k`),
/// and [`rshift_rne`] rounds a tie to the even word. No `f64` multiply or rounding runs here.
///
/// # Errors
/// [`ConvertError::NonFinite`] for a `NaN`/infinite `x`; [`ConvertError::Range`] for `|x| >= 2^31`.
///
/// # Complexity
/// `O(1)`.
pub fn f64_to_q32(x: f64) -> Result<i64, ConvertError> {
    let bits = x.to_bits();
    let e = (bits >> 52) & 0x7ff;
    let refuse = || ConvertError::Range(format!("{x}"));
    match e {
        0x7ff => return Err(ConvertError::NonFinite(format!("{x}"))),
        0 => return Ok(0), // subnormal: below the half quantum of the `2^-32` grid
        EXP_2_31.. => return Err(refuse()),
        _ => {}
    }
    let m = i128::from((bits & 0x000f_ffff_ffff_ffff) | (1 << 52));
    let k = e as i32 - 1043;
    // `x·2^32 = ±m·2^k`; the shift clamp is exact because `m <= 2^53` leaves a value below the
    // half quantum whenever `k < -126`.
    let mag = if k >= 0 {
        m << k
    } else {
        rshift_rne(m, k.unsigned_abs().min(126)).ok_or_else(refuse)?
    };
    let y = i64::try_from(mag).map_err(|_| refuse())?;
    Ok(if bits >> 63 == 0 { y } else { -y })
}

/// The scalar encoding one pooled pass applies: `F16`/`F32` = `Q32.32`→2/4-byte little-endian
/// payload.
#[derive(Clone, Copy)]
enum Map {
    F16,
    F32,
}

/// One elementwise whole-tensor pass, split across the engine pool: `q` the `Q32.32` source, `first`
/// the lowest refused element's offset.
///
/// # Complexity
/// Time `O(n)` per call: one decode and one store per output element, ≈1–3 ns each against a few-µs
/// dispatch. A dense tensor is 10^5–10^8 words = 0.1–100 ms per call, 20×–10^4× its dispatch; the ≤5
/// norm/bias vectors per layer are ≤896 words, ≈3 µs lost each (≈0.4 ms per 24-layer run), so one
/// pooled body serves both. Space `O(n)`.
struct Pass<'a> {
    map: Map,
    q: &'a [i64],
    first: &'a FirstErr<usize, ConvertError>,
}

impl ChunkTask<u8> for Pass<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [u8]) {
        if matches!(self.map, Map::F16) {
            for (i, slot) in piece.chunks_exact_mut(2).enumerate() {
                match f16_pair(self.q[start / 2 + i]) {
                    Ok(b) => slot.copy_from_slice(&b),
                    Err(e) => return self.first.record(start / 2 + i, e),
                }
            }
        } else {
            for (i, slot) in piece.chunks_exact_mut(4).enumerate() {
                let bits = F32::from_q32(self.q[start / 4 + i]).to_bits();
                slot.copy_from_slice(&bits.to_le_bytes());
            }
        }
    }
}

/// Dispatch one elementwise pass over `out` (`w` bytes per element).
///
/// # Errors
/// The lowest refused element's error; `F32` cannot refuse.
fn pass(map: Map, q: &[i64], w: usize, out: &mut [u8]) -> Result<(), ConvertError> {
    if out.is_empty() {
        return Ok(());
    }
    let first = FirstErr::new();
    let per_task = w * (out.len() / w).div_ceil(width_global()).max(1);
    let task = Pass {
        map,
        q,
        first: &first,
    };
    chunks_mut_global(out, per_task, &task);
    match first.take() {
        Some((_, e)) => Err(e),
        None => Ok(()),
    }
}

/// The little-endian binary16 payload of a whole tensor, one RNE narrowing of the grid value per word.
///
/// # Errors
/// [`ConvertError::Range`] for a rounded magnitude with no finite binary16 encoding.
pub fn encode_f16(v: &[i64]) -> Result<Vec<u8>, ConvertError> {
    let mut out = vec![0u8; v.len() * 2];
    pass(Map::F16, v, 2, &mut out)?;
    Ok(out)
}

/// The little-endian binary32 payload of a whole tensor ([`F32::from_q32`]); no `f64` is formed.
#[must_use]
pub fn encode_f32(v: &[i64]) -> Vec<u8> {
    let mut out = vec![0u8; v.len() * 4];
    let _ = pass(Map::F32, v, 4, &mut out);
    out
}

/// Read whole little-endian `f16`s from `raw` into `Q32.32` words; an exact decode, since every
/// finite `f16` is a multiple of `2^-24`.
///
/// # Errors
/// [`ConvertError::Shape`] for a payload that is not a whole number of 2-byte elements;
/// [`ConvertError::NonFinite`] for an `Inf`/`NaN` element.
pub fn decode_f16(raw: &[u8]) -> Result<Vec<i64>, ConvertError> {
    if !raw.len().is_multiple_of(2) {
        return Err(ConvertError::Shape(format!(
            "{} bytes is not a whole number of f16 words",
            raw.len()
        )));
    }
    raw.chunks_exact(2)
        .map(|c| {
            let h = u16::from_le_bytes([c[0], c[1]]);
            f16_to_fixed(h).map_err(|_| ConvertError::NonFinite(format!("f16 {h:#06x}")))
        })
        .collect()
}

/// Read whole little-endian `bf16`s from `raw` into `Q32.32` words: the top 16 bits of the `f32`
/// bit pattern, widened exactly and rounded half-even at `2^-32`.
///
/// # Errors
/// [`ConvertError::Shape`] for a payload that is not a whole number of 2-byte elements;
/// [`ConvertError::NonFinite`] for an `Inf`/`NaN` element; [`ConvertError::Range`] for a magnitude
/// that leaves the `Q32.32` grid.
pub fn decode_bf16(raw: &[u8]) -> Result<Vec<i64>, ConvertError> {
    if !raw.len().is_multiple_of(2) {
        return Err(ConvertError::Shape(format!(
            "{} bytes is not a whole number of bf16 words",
            raw.len()
        )));
    }
    raw.chunks_exact(2)
        .map(|c| f32_bits_to_q32(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect()
}

/// Read whole little-endian `f32`s from `raw` into `Q32.32` words, rounded half-even at `2^-32`.
///
/// # Errors
/// [`ConvertError::Shape`] for a payload that is not a whole number of 4-byte elements;
/// [`ConvertError::NonFinite`] for an `Inf`/`NaN` element; [`ConvertError::Range`] for a magnitude
/// that leaves the `Q32.32` grid.
pub fn decode_f32(raw: &[u8]) -> Result<Vec<i64>, ConvertError> {
    if !raw.len().is_multiple_of(4) {
        return Err(ConvertError::Shape(format!(
            "{} bytes is not a whole number of f32 words",
            raw.len()
        )));
    }
    raw.chunks_exact(4)
        .map(|c| f32_bits_to_q32(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

/// One `f32` bit pattern as a `Q32.32` word: an exact widening into the `f64` exponent-and-significand
/// layout, then the integer half-even decode of [`f64_to_q32`]. Subnormals are below the half
/// quantum and decode to zero; `Inf`/`NaN` refuse.
fn f32_bits_to_q32(bits: u32) -> Result<i64, ConvertError> {
    let e = (bits >> 23) & 0xff;
    if e == 0xff {
        return Err(ConvertError::NonFinite(format!("f32 {bits:#010x}")));
    }
    if e == 0 {
        return Ok(0);
    }
    let wide = (u64::from(bits >> 31) << 63)
        | (u64::from(e + 896) << 52)
        | (u64::from(bits & 0x007f_ffff) << 29);
    f64_to_q32(f64::from_bits(wide))
}

/// The binary16 bit pattern of a rounded [`Fx`] `m · 2^e` ([`round_f16`]'s output): sign, biased
/// exponent and 10-bit significand. A significand that rounded up to `2^11` is the next binade
/// exactly and is renormalised, never refused.
///
/// # Errors
/// [`ConvertError::Range`] for a rounded magnitude with no finite binary16 encoding.
fn f16_bits(x: &Fx) -> Result<u16, ConvertError> {
    let (mut a, mut e) = (x.m.unsigned_abs(), x.e);
    if a == 2048 {
        (a, e) = (1024, e + 1);
    }
    let sign = if x.m < 0 { 0x8000 } else { 0 };
    // `round_f16` uses the subnormal quantum `q = -24` below `2^-14`, where `a < 1024`.
    if a == 0 {
        return Ok(sign);
    }
    if e == -24 && a < 1024 {
        return Ok(sign | (a as u16));
    }
    if !((1024..2048).contains(&a) && (1..=30).contains(&(e + 25))) {
        return Err(ConvertError::Range(format!("f16 {x:?}")));
    }
    Ok(sign | (((e + 25) as u16) << 10) | ((a - 1024) as u16))
}

/// The little-endian binary16 payload of one `Q32.32` word.
fn f16_pair(word: i64) -> Result<[u8; 2], ConvertError> {
    let r = Fx::from_i128(i128::from(word), -32).and_then(round_f16);
    let refuse = |e| ConvertError::Range(format!("{word}: {e:?}"));
    Ok(f16_bits(&r.map_err(refuse)?)?.to_le_bytes())
}

/// `a · b` or [`ConvertError::Shape`] when the product leaves `usize`.
fn area(a: usize, b: usize, what: &str) -> Result<usize, ConvertError> {
    a.checked_mul(b)
        .ok_or_else(|| ConvertError::Shape(format!("{what}: {a}·{b} overflow")))
}

/// `actual == expected` or [`ConvertError::Shape`].
fn expect_len(actual: usize, expected: usize, what: &str) -> Result<(), ConvertError> {
    (actual == expected)
        .then_some(())
        .ok_or_else(|| ConvertError::Shape(format!("{what}: {actual} != {expected}")))
}

/// Lift the engine-loaded `Q32.32` [`Weights`] into the word [`Teacher`] the fold driver consumes.
/// Nothing is converted: the engine already carries the words, so the lift is a shape check and a
/// clone. The geometry comes from `cfg` (`eps = cfg.eps / 2^32`, `theta = cfg.rope_theta`) and the
/// per-layer field map is one-to-one (`ln1 = attn_norm`, `ln2 = ffn_norm`, `q/k/v = wq/wk/wv` with
/// their biases, `o = wo`, `gate/up/down = w_gate/w_up/w_down`).
///
/// # Errors
/// [`ConvertError::MissingBias`] for a missing attention bias; [`ConvertError::Shape`] for a layer
/// count, tensor length or `usize` product that disagrees with `cfg`.
///
/// # Complexity
/// Time and space `O(n)` over the whole weight set (`n_layer·(n_embd² + n_embd·n_ff) + vocab·n_embd`
/// words), one shape check per tensor and one clone.
pub fn teacher_from_grad(w: &Weights, cfg: &Config) -> Result<Teacher, ConvertError> {
    let hidden = cfg.n_embd;
    let q_out = area(cfg.n_head, cfg.head_dim, "q_out")?;
    let kv_out = area(cfg.n_head_kv, cfg.head_dim, "kv_out")?;
    let inter = cfg.n_ff;
    let n_layer = w.layers.len();
    expect_len(cfg.n_layer, n_layer, "n_layer")?;
    expect_len(w.output_norm.len(), hidden, "output_norm")?;
    expect_len(
        w.token_embd.len(),
        area(cfg.vocab, hidden, "token_embd")?,
        "token_embd",
    )?;
    expect_len(w.lm_head.len(), w.token_embd.len(), "lm_head")?;

    let layer_cfg = LayerCfg {
        n_head: cfg.n_head,
        n_kv_head: cfg.n_head_kv,
        head_dim: cfg.head_dim,
        hidden,
        inter,
        eps: cfg.eps as f64 / SCALE,
        theta: cfg.rope_theta,
    };

    let q = area(q_out, hidden, "wq")?;
    let kv = area(kv_out, hidden, "wk")?;
    let wo = area(hidden, q_out, "wo")?;
    let gate = area(inter, hidden, "w_gate")?;
    let up = area(inter, hidden, "w_up")?;
    let down = area(hidden, inter, "w_down")?;
    let mut layers: Vec<LayerWeights> = Vec::with_capacity(n_layer);
    for (l, lw) in w.layers.iter().enumerate() {
        for (got, want, name) in [
            (lw.attn_norm.len(), hidden, "attn_norm"),
            (lw.wq.len(), q, "attn_q.weight"),
            (lw.wk.len(), kv, "attn_k.weight"),
            (lw.wv.len(), kv, "attn_v.weight"),
            (lw.wo.len(), wo, "attn_output.weight"),
            (lw.ffn_norm.len(), hidden, "ffn_norm"),
            (lw.w_gate.len(), gate, "ffn_gate.weight"),
            (lw.w_up.len(), up, "ffn_up.weight"),
            (lw.w_down.len(), down, "ffn_down.weight"),
        ] {
            expect_len(got, want, &format!("blk.{l}.{name}"))?;
        }
        let bq = lw
            .bq
            .as_deref()
            .ok_or_else(|| ConvertError::MissingBias(format!("blk.{l}.attn_q.bias")))?;
        let bk = lw
            .bk
            .as_deref()
            .ok_or_else(|| ConvertError::MissingBias(format!("blk.{l}.attn_k.bias")))?;
        let bv = lw
            .bv
            .as_deref()
            .ok_or_else(|| ConvertError::MissingBias(format!("blk.{l}.attn_v.bias")))?;
        expect_len(bq.len(), q_out, &format!("blk.{l}.attn_q.bias"))?;
        expect_len(bk.len(), kv_out, &format!("blk.{l}.attn_k.bias"))?;
        expect_len(bv.len(), kv_out, &format!("blk.{l}.attn_v.bias"))?;
        layers.push(lw.clone());
    }

    Ok(Teacher {
        layer_cfg,
        layers,
        fnorm: w.output_norm.clone(),
        embed: w.token_embd.clone(),
        vocab: cfg.vocab,
        n_layer,
    })
}

/// Lower the driver's fitted word layers and final-norm gain back into a `Q32.32` [`Weights`] whose
/// forward is the engine's: the first `layers.len()` layers from `layers` (with `Some` biases),
/// `output_norm` from `fnorm`, and `token_embd`/`lm_head` cloned from `teacher`. No value is
/// converted; the shape checks are the only guard.
///
/// # Errors
/// [`ConvertError::Shape`] for a layer count or tensor length that disagrees with `teacher`.
///
/// # Complexity
/// Time and space `O(n)` over the fitted layers and the final gain: one clone per tensor.
pub fn student_to_grad(
    teacher: &Weights,
    layers: &[LayerWeights],
    fnorm: &[i64],
) -> Result<Weights, ConvertError> {
    if layers.len() != teacher.layers.len() {
        return Err(ConvertError::Shape(format!(
            "layer count {} != {}",
            layers.len(),
            teacher.layers.len()
        )));
    }
    let mut out: Vec<LayerWeights> = Vec::with_capacity(layers.len());
    for (l, (lw, tl)) in layers.iter().zip(&teacher.layers).enumerate() {
        for (got, want, name) in [
            (lw.attn_norm.len(), tl.attn_norm.len(), "attn_norm"),
            (lw.ffn_norm.len(), tl.ffn_norm.len(), "ffn_norm"),
            (lw.wq.len(), tl.wq.len(), "attn_q.weight"),
            (lw.wk.len(), tl.wk.len(), "attn_k.weight"),
            (lw.wv.len(), tl.wv.len(), "attn_v.weight"),
            (lw.wo.len(), tl.wo.len(), "attn_output.weight"),
            (lw.w_gate.len(), tl.w_gate.len(), "ffn_gate.weight"),
            (lw.w_up.len(), tl.w_up.len(), "ffn_up.weight"),
            (lw.w_down.len(), tl.w_down.len(), "ffn_down.weight"),
        ] {
            expect_len(got, want, &format!("blk.{l}.{name}"))?;
        }
        let q_out = tl
            .bq
            .as_ref()
            .map_or(lw.bq.as_ref().map_or(0, Vec::len), Vec::len);
        let kv_out = tl
            .bk
            .as_ref()
            .map_or(lw.bk.as_ref().map_or(0, Vec::len), Vec::len);
        expect_len(
            lw.bq.as_ref().map_or(0, Vec::len),
            q_out,
            &format!("blk.{l}.attn_q.bias"),
        )?;
        expect_len(
            lw.bk.as_ref().map_or(0, Vec::len),
            kv_out,
            &format!("blk.{l}.attn_k.bias"),
        )?;
        expect_len(
            lw.bv.as_ref().map_or(0, Vec::len),
            kv_out,
            &format!("blk.{l}.attn_v.bias"),
        )?;
        out.push(lw.clone());
    }
    let output_norm = fnorm.to_vec();
    expect_len(output_norm.len(), teacher.output_norm.len(), "output_norm")?;
    Ok(Weights {
        layers: out,
        output_norm,
        token_embd: teacher.token_embd.clone(),
        lm_head: teacher.lm_head.clone(),
    })
}
