//! The glue between the engine-loaded `Q32.32` model ([`crate::learn::grad::model::Weights`],
//! [`crate::learn::grad::model::Config`]) and the `f64` fold driver
//! ([`crate::learn::fold::layer::driver::LayerWeights`]), in both directions.
//!
//! The driver runs the fold transfer in `f64` (`stats::parts`, `fold_transfer.py`); the forward and
//! the gradient live in the engine's `Q32.32` carrier (`grad::model`), where a weight is an `i64`
//! scaled by `2^-32`. [`teacher_from_grad`] lifts a loaded [`Weights`] into a [`Teacher`] the driver
//! consumes; [`student_to_grad`] lowers the driver's fitted `f64` layers back into a [`Weights`]
//! whose `Q32.32` forward is the engine's.
//!
//! The fixed-point scalar is
//!
//! ```text
//! q32_to_f64(v) = v / 2^32                 # exact for |v| < 2^21
//! f64_to_q32(x) = round(x · 2^32) : i64    # |x| < 2^31, finite, else refused
//! ```
//!
//! so `f64_to_q32(q32_to_f64(v)) == v` for every `Q32.32` word and `|f64_to_q32(x) as f64/2^32 - x|
//! <= 2^-33` for any admissible `x` (round-to-nearest of one `f64` multiply-add).
//!
//! # Evidence grade
//!
//! `tested`: the gate's `cargo test` on the build path (`tests/learn_fold_run_convert.rs`). No Lean
//! refinement links this file; it is not on the extracted fragment, so nothing here is `proved`.
//!
//! # Trust boundary
//!
//! Covers the elementwise scalar conversion and the field map between the two representations. Not
//! covered: the `GGUF` load ([`Weights::from_engine`], the engine's contract), the driver's fit
//! arithmetic, and rustc/LLVM codegen. The `f64 → i64` cast sits behind the range check in
//! [`f64_to_q32`]; no `unsafe` appears here.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    reason = "the `f64↔Q32.32` boundary: `q32_to_f64`'s `i64 as f64` is the documented scalar \
              (exact for the admitted range `|v| < 2^21`) and `f64_to_q32`'s `f64 as i64` runs only \
              after the finite and `|q| < 2^63` checks, so the truncation is unreachable"
)]

use crate::learn::fold::layer::driver::LayerWeights;
use crate::learn::fold::stats::parts::LayerCfg;
use crate::learn::grad::model::{Config, LayerWeights as GradLayerWeights, Weights};

/// The `Q32.32` scale `2^32`.
const SCALE: f64 = 4_294_967_296.0;

/// `2^31`: the largest magnitude an admissible `f64` may carry (`|x| < 2^31` keeps `x·2^32 < 2^63`).
const RANGE_BOUND: f64 = 2_147_483_648.0;

/// `2^63`: the exclusive bound on the rounded `Q32.32` word, so `q as i64` is exact.
const I64_BOUND: f64 = 9_223_372_036_854_775_808.0;

/// A refusal from the engine-weights/fold-driver conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConvertError {
    /// A tensor length disagrees with the geometry the other side declares.
    Shape(String),
    /// A `Q→f64` conversion needs a `Qwen2` attention bias the loaded model does not carry.
    MissingBias(String),
    /// A finite `f64` is out of the `Q32.32` representable range (`|x| >= 2^31`).
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

/// The `f64` teacher the fold driver consumes: the per-layer weights in the driver's layout, the
/// final `RMSNorm` gain and the token embedding, plus the geometry.
#[derive(Clone, Debug)]
pub struct Teacher {
    /// The shared per-layer geometry (all layers have the same shape in `Qwen2`).
    pub layer_cfg: LayerCfg,
    /// One entry per trunk layer, in the driver's [`LayerWeights`] layout.
    pub layers: Vec<LayerWeights>,
    /// The final `RMSNorm` gain `output_norm.weight`, length `hidden`.
    pub fnorm: Vec<f64>,
    /// The token embedding `token_embd.weight`, row-major `[vocab, hidden]`.
    pub embed: Vec<f64>,
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

/// `round(x · 2^32)` as an `i64`, or a typed refusal.
///
/// # Errors
///
/// [`ConvertError::NonFinite`] for a `NaN`/infinite `x`; [`ConvertError::Range`] for a finite `x`
/// with `|x| >= 2^31` (whose `Q32.32` word would leave `i64`).
///
/// # Complexity
/// `O(1)`.
pub fn f64_to_q32(x: f64) -> Result<i64, ConvertError> {
    if !x.is_finite() {
        return Err(ConvertError::NonFinite(format!("{x}")));
    }
    if x.abs() >= RANGE_BOUND {
        return Err(ConvertError::Range(format!("{x}")));
    }
    let q = (x * SCALE).round();
    if !q.is_finite() || q.abs() >= I64_BOUND {
        return Err(ConvertError::Range(format!("{x}")));
    }
    Ok(q as i64)
}

/// `q32_to_f64` over a whole tensor.
///
/// # Complexity
/// `O(n)` in the element count.
#[must_use]
fn q32_to_f64_slice(v: &[i64]) -> Vec<f64> {
    v.iter().copied().map(q32_to_f64).collect()
}

/// [`f64_to_q32`] over a whole tensor, failing on the first refused element.
///
/// # Errors
/// The first element's [`ConvertError::NonFinite`] or [`ConvertError::Range`].
///
/// # Complexity
/// `O(n)` in the element count.
fn f64_to_q32_slice(v: &[f64]) -> Result<Vec<i64>, ConvertError> {
    v.iter().copied().map(f64_to_q32).collect()
}

/// `a · b` or [`ConvertError::Shape`] when the product leaves `usize`.
fn area(a: usize, b: usize, what: &str) -> Result<usize, ConvertError> {
    a.checked_mul(b)
        .ok_or_else(|| ConvertError::Shape(format!("{what}: {a}·{b} overflow")))
}

/// `actual == expected` or [`ConvertError::Shape`].
fn expect_len(actual: usize, expected: usize, what: &str) -> Result<(), ConvertError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ConvertError::Shape(format!(
            "{what}: {actual} != {expected}"
        )))
    }
}

/// Lift the engine-loaded `Q32.32` [`Weights`] into the `f64` [`Teacher`] the fold driver consumes.
///
/// The `f64` layer geometry is built from `cfg` (`n_head_kv = cfg.n_head_kv`, `hidden = cfg.n_embd`,
/// `inter = cfg.n_ff`, `eps = cfg.eps / 2^32`, `theta = cfg.rope_theta`); the per-layer field map is
/// `ln1 = attn_norm`, `ln2 = ffn_norm`, `q = wq/qb`, `k = wk/kb`, `v = wv/vb`, `o = wo`,
/// `gate = w_gate`, `up = w_up`, `down = w_down`. The three attention biases must be present, as
/// `Qwen2` carries them and the driver requires them.
///
/// # Errors
///
/// [`ConvertError::MissingBias`] for a `bq`/`bk`/`bv` the model does not carry;
/// [`ConvertError::Shape`] for a layer count or a tensor length that disagrees with `cfg`, or a
/// `usize` product that leaves range.
///
/// # Complexity
/// Time `O(n_layer·(n_embd² + n_embd·n_ff) + vocab·n_embd)` conversions (the whole weight set), each
/// a divide by a power of two; space `O` of the same element count, one `f64` per `Q32.32` word.
/// Derivation: one pass over each tensor's elements.
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

    let mut layers: Vec<LayerWeights> = Vec::with_capacity(n_layer);
    for (l, lw) in w.layers.iter().enumerate() {
        expect_len(lw.attn_norm.len(), hidden, &format!("blk.{l}.attn_norm"))?;
        expect_len(
            lw.wq.len(),
            area(q_out, hidden, "wq")?,
            &format!("blk.{l}.attn_q.weight"),
        )?;
        expect_len(
            lw.wk.len(),
            area(kv_out, hidden, "wk")?,
            &format!("blk.{l}.attn_k.weight"),
        )?;
        expect_len(
            lw.wv.len(),
            area(kv_out, hidden, "wv")?,
            &format!("blk.{l}.attn_v.weight"),
        )?;
        expect_len(
            lw.wo.len(),
            area(hidden, q_out, "wo")?,
            &format!("blk.{l}.attn_output.weight"),
        )?;
        expect_len(lw.ffn_norm.len(), hidden, &format!("blk.{l}.ffn_norm"))?;
        expect_len(
            lw.w_gate.len(),
            area(inter, hidden, "w_gate")?,
            &format!("blk.{l}.ffn_gate.weight"),
        )?;
        expect_len(
            lw.w_up.len(),
            area(inter, hidden, "w_up")?,
            &format!("blk.{l}.ffn_up.weight"),
        )?;
        expect_len(
            lw.w_down.len(),
            area(hidden, inter, "w_down")?,
            &format!("blk.{l}.ffn_down.weight"),
        )?;
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
        layers.push(LayerWeights {
            ln1: q32_to_f64_slice(&lw.attn_norm),
            ln2: q32_to_f64_slice(&lw.ffn_norm),
            q: q32_to_f64_slice(&lw.wq),
            qb: q32_to_f64_slice(bq),
            k: q32_to_f64_slice(&lw.wk),
            kb: q32_to_f64_slice(bk),
            v: q32_to_f64_slice(&lw.wv),
            vb: q32_to_f64_slice(bv),
            o: q32_to_f64_slice(&lw.wo),
            gate: q32_to_f64_slice(&lw.w_gate),
            up: q32_to_f64_slice(&lw.w_up),
            down: q32_to_f64_slice(&lw.w_down),
        });
    }

    Ok(Teacher {
        layer_cfg,
        layers,
        fnorm: q32_to_f64_slice(&w.output_norm),
        embed: q32_to_f64_slice(&w.token_embd),
        vocab: cfg.vocab,
        n_layer,
    })
}

/// Lower the driver's fitted `f64` layers and final-norm gain back into a `Q32.32` [`Weights`] whose
/// forward is the engine's. The first `layers.len()` layers come from `layers` (with `Some` biases);
/// `output_norm` comes from `fnorm`; `token_embd` and `lm_head` are cloned from `teacher`.
///
/// # Errors
///
/// [`ConvertError::Shape`] when `layers.len()` differs from `teacher.layers.len()`, when a `f64`
/// tensor's length differs from the matching `teacher` tensor, or when `fnorm` does not match
/// `teacher.output_norm`; [`ConvertError::NonFinite`] and [`ConvertError::Range`] from any
/// out-of-domain `f64` element.
///
/// # Complexity
/// Time `O(n_layer·(n_embd² + n_embd·n_ff) + n_embd)` conversions (the fitted layers and the final
/// gain); space `O` of the same element count, one `Q32.32` word per `f64`. Derivation: one pass
/// over each tensor's elements; `token_embd`/`lm_head` are cloned, not converted.
pub fn student_to_grad(
    teacher: &Weights,
    layers: &[LayerWeights],
    fnorm: &[f64],
) -> Result<Weights, ConvertError> {
    if layers.len() != teacher.layers.len() {
        return Err(ConvertError::Shape(format!(
            "layer count {} != {}",
            layers.len(),
            teacher.layers.len()
        )));
    }
    let mut out: Vec<GradLayerWeights> = Vec::with_capacity(layers.len());
    for (l, (lw, tl)) in layers.iter().zip(&teacher.layers).enumerate() {
        expect_len(
            lw.ln1.len(),
            tl.attn_norm.len(),
            &format!("blk.{l}.attn_norm"),
        )?;
        expect_len(
            lw.ln2.len(),
            tl.ffn_norm.len(),
            &format!("blk.{l}.ffn_norm"),
        )?;
        expect_len(lw.q.len(), tl.wq.len(), &format!("blk.{l}.attn_q.weight"))?;
        expect_len(lw.k.len(), tl.wk.len(), &format!("blk.{l}.attn_k.weight"))?;
        expect_len(lw.v.len(), tl.wv.len(), &format!("blk.{l}.attn_v.weight"))?;
        expect_len(
            lw.o.len(),
            tl.wo.len(),
            &format!("blk.{l}.attn_output.weight"),
        )?;
        expect_len(
            lw.gate.len(),
            tl.w_gate.len(),
            &format!("blk.{l}.ffn_gate.weight"),
        )?;
        expect_len(
            lw.up.len(),
            tl.w_up.len(),
            &format!("blk.{l}.ffn_up.weight"),
        )?;
        expect_len(
            lw.down.len(),
            tl.w_down.len(),
            &format!("blk.{l}.ffn_down.weight"),
        )?;
        let q_out = tl.bq.as_ref().map_or(lw.qb.len(), Vec::len);
        let kv_out = tl.bk.as_ref().map_or(lw.kb.len(), Vec::len);
        expect_len(lw.qb.len(), q_out, &format!("blk.{l}.attn_q.bias"))?;
        expect_len(lw.kb.len(), kv_out, &format!("blk.{l}.attn_k.bias"))?;
        expect_len(lw.vb.len(), kv_out, &format!("blk.{l}.attn_v.bias"))?;
        out.push(GradLayerWeights {
            attn_norm: f64_to_q32_slice(&lw.ln1)?,
            wq: f64_to_q32_slice(&lw.q)?,
            bq: Some(f64_to_q32_slice(&lw.qb)?),
            wk: f64_to_q32_slice(&lw.k)?,
            bk: Some(f64_to_q32_slice(&lw.kb)?),
            wv: f64_to_q32_slice(&lw.v)?,
            bv: Some(f64_to_q32_slice(&lw.vb)?),
            wo: f64_to_q32_slice(&lw.o)?,
            ffn_norm: f64_to_q32_slice(&lw.ln2)?,
            w_gate: f64_to_q32_slice(&lw.gate)?,
            w_up: f64_to_q32_slice(&lw.up)?,
            w_down: f64_to_q32_slice(&lw.down)?,
        });
    }
    let output_norm = f64_to_q32_slice(fnorm)?;
    expect_len(output_norm.len(), teacher.output_norm.len(), "output_norm")?;
    Ok(Weights {
        layers: out,
        output_norm,
        token_embd: teacher.token_embd.clone(),
        lm_head: teacher.lm_head.clone(),
    })
}
