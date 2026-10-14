//! Whole-model Qwen2 forward with stage capture and whole-model backward in Q32.32 (W19 wave 2).
//!
//! Composes the landed per-op modules [`dense`], [`attn`] and [`super::head`] into the engine's
//! per-layer schedule: `attn_norm` → Q/K/V projection with the Qwen2 attention bias → `RoPE` →
//! causal GQA attention → O projection → residual; then `ffn_norm` → `SwiGLU` → down → residual;
//! then `output_norm` → head. The order, bias placement, GQA head mapping, `RoPE` table, residual
//! placement and `eps` are `Engine::compose_cols`'s (`src/engine/step/cols.rs`) and
//! `Engine::step_logits`'s exact path, so under the engine's Q32 `KvState` carrier [`forward`]'s
//! logits are bit-identical per position to `Engine::step_logits` over the same token sequence (see
//! the trust boundary: the default carrier is the `f32` one).
//!
//! Every weight is an `i64` Q32.32 row-major buffer loaded from an [`Engine`] with the engine's own
//! row dequantizer ([`dequant_row_fixed`]); the norm gains and attention biases are the engine's
//! resolved [`DecodeVectors`](crate::engine::loader::DecodeVectors) entries.
//!
//! # Evidence grade
//!
//! `tested`: the gate's `cargo test` on the build path. The test module asserts bit-equality of the
//! forward logits against `Engine::step_logits` and the backward's analytic gradient against a
//! central difference of the KL loss. No Lean refinement links this file yet (`proved` is open); it
//! is not on the extracted fragment.
//!
//! # Trust boundary
//!
//! The forward refines the engine's **Q32 `KvState` carrier** decode path, not the process default:
//! `carrier_from_env()` selects the native-binary32 carrier unless `APEIRON_F32_CARRIER=0`, and its
//! attention (`attn_head_f32`) is a float path this Q32 composition does not and cannot match. Gate
//! A runs in a child process with `APEIRON_F32_CARRIER=0` (the engine caches the carrier
//! process-wide) and there [`forward`]'s logits are
//! bit-identical to `Engine::step_logits` at every position. `i128` accumulation and the per-op
//! rounding are the per-op modules' contracts. Not covered here: partial rotary
//! (`rot_dims < head_dim`), MoE/routed experts, the affine mixer, the per-head QK `RMSNorm`, the
//! attention output gate, Hadamard-folded weights and the stochastic sampler. The residual and bias
//! adds are checked `i64` adds; every overflow returns [`ModelError::Overflow`]. No float and no
//! `unsafe` appear after the load boundary.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the f64→Q32.32 boundary is range-checked immediately before the `as i64`; the \
              single-letter names are the schedule's symbols (`t`, `e`, `hd`, `r`, `o`, `i`) and \
              the gradient carries `dh`/`dh2`/`dln1`/`dln2`/`dh_attn` name the same stage on the \
              backward and the forward; the layer loop and the forward/backward are each one linear \
              composition"
)]

use super::attn::{self, AttnError};
use super::dense::{self, DenseError};
use crate::engine::{dequant_row_fixed, Engine, EngineError, RopeTable};

/// A refused whole-model forward or backward evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// A dimension, a buffer length or a token id disagrees with the model geometry.
    Shape(String),
    /// A checked add (the residual or bias accumulation) left `i64`.
    Overflow,
    /// A named tensor is absent from the engine directory.
    Missing(String),
    /// The engine refused a payload read or a row dequantization.
    Engine(EngineError),
    /// The configuration cannot form the model (a non-finite or out-of-range `rms_eps`).
    Config(String),
    /// A dense forward/backward kernel refused.
    Dense(DenseError),
    /// An attention forward/backward kernel refused.
    Attn(AttnError),
}

impl From<EngineError> for ModelError {
    fn from(e: EngineError) -> Self {
        ModelError::Engine(e)
    }
}

impl From<DenseError> for ModelError {
    fn from(e: DenseError) -> Self {
        ModelError::Dense(e)
    }
}

impl From<AttnError> for ModelError {
    fn from(e: AttnError) -> Self {
        ModelError::Attn(e)
    }
}

/// The resolved model geometry plus the Q32.32 `RMSNorm` epsilon the engine's step uses.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Trunk layer count.
    pub n_layer: usize,
    /// Query heads.
    pub n_head: usize,
    /// KV heads (GQA); `n_head % n_head_kv == 0`.
    pub n_head_kv: usize,
    /// Residual width.
    pub n_embd: usize,
    /// Feed-forward width.
    pub n_ff: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// Per-head width.
    pub head_dim: usize,
    /// `round(rms_eps · 2^32)`, the engine's `eps_fixed` value.
    pub eps: i64,
    /// `rope_theta`, the `RoPE` table's load input.
    pub rope_theta: f64,
}

impl Config {
    /// `q_out = n_head · head_dim`, or [`ModelError::Overflow`].
    ///
    /// # Errors
    ///
    /// [`ModelError::Overflow`] when the product leaves `usize`.
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn q_out(&self) -> Result<usize, ModelError> {
        self.n_head
            .checked_mul(self.head_dim)
            .ok_or(ModelError::Overflow)
    }

    /// `kv_out = n_head_kv · head_dim`, or [`ModelError::Overflow`].
    ///
    /// # Errors
    ///
    /// [`ModelError::Overflow`] when the product leaves `usize`.
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn kv_out(&self) -> Result<usize, ModelError> {
        self.n_head_kv
            .checked_mul(self.head_dim)
            .ok_or(ModelError::Overflow)
    }

    /// The dimension sanity the forward and backward both assume.
    ///
    /// # Errors
    ///
    /// [`ModelError::Shape`] for a zero model dimension or GQA head counts that do not divide;
    /// [`ModelError::Overflow`] from [`Config::q_out`] or [`Config::kv_out`].
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn checked(&self) -> Result<(), ModelError> {
        if self.n_layer == 0 || self.n_embd == 0 || self.n_ff == 0 || self.head_dim == 0 {
            return Err(ModelError::Shape("zero model dimension".to_string()));
        }
        if self.n_head_kv == 0 || !self.n_head.is_multiple_of(self.n_head_kv) {
            return Err(ModelError::Shape("GQA head counts".to_string()));
        }
        let _ = self.q_out()?;
        let _ = self.kv_out()?;
        Ok(())
    }
}

/// One layer's Q32.32 parameters, row-major `o × i` with the input dimension fastest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerWeights {
    /// `attn_norm.weight`, `n_embd` wide.
    pub attn_norm: Vec<i64>,
    /// `attn_q.weight`, `(n_head·head_dim) × n_embd`.
    pub wq: Vec<i64>,
    /// `attn_q.bias`, `n_head·head_dim` wide, or `None`.
    pub bq: Option<Vec<i64>>,
    /// `attn_k.weight`, `(n_head_kv·head_dim) × n_embd`.
    pub wk: Vec<i64>,
    /// `attn_k.bias`, `n_head_kv·head_dim` wide, or `None`.
    pub bk: Option<Vec<i64>>,
    /// `attn_v.weight`, `(n_head_kv·head_dim) × n_embd`.
    pub wv: Vec<i64>,
    /// `attn_v.bias`, `n_head_kv·head_dim` wide, or `None`.
    pub bv: Option<Vec<i64>>,
    /// `attn_output.weight`, `n_embd × (n_head·head_dim)`.
    pub wo: Vec<i64>,
    /// `ffn_norm.weight`, `n_embd` wide.
    pub ffn_norm: Vec<i64>,
    /// `ffn_gate.weight`, `n_ff × n_embd`.
    pub w_gate: Vec<i64>,
    /// `ffn_up.weight`, `n_ff × n_embd`.
    pub w_up: Vec<i64>,
    /// `ffn_down.weight`, `n_embd × n_ff`.
    pub w_down: Vec<i64>,
}

/// The whole model's Q32.32 parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Weights {
    /// Per trunk layer.
    pub layers: Vec<LayerWeights>,
    /// `output_norm.weight`, `n_embd` wide.
    pub output_norm: Vec<i64>,
    /// `token_embd.weight`, `vocab × n_embd`.
    pub token_embd: Vec<i64>,
    /// The head (`output.weight` when present, else tied `token_embd.weight`), `vocab × n_embd`.
    pub lm_head: Vec<i64>,
}

/// `round(v · 2^32)` for a finite non-negative `v`, the engine's `eps_fixed` body.
///
/// # Complexity
/// `O(1)`.
fn eps_q32(v: f64) -> Result<i64, ModelError> {
    if !v.is_finite() || v < 0.0 {
        return Err(ModelError::Config("rms_eps".to_string()));
    }
    let q = (v * 4_294_967_296.0).round();
    if !(q.is_finite() && (0.0..9_223_372_036_854_775_808.0).contains(&q)) {
        return Err(ModelError::Config("rms_eps".to_string()));
    }
    Ok(q as i64)
}

/// The `[n_in, n_out]` quantized tensor `name` dequantized to a row-major `n_out × n_in` Q32.32
/// matrix, one engine row dequantizer call per output row.
///
/// # Complexity
/// `O(n_out·n_in)` dequantized elements; one contiguous read of the checked tensor extent.
fn load_matrix(
    engine: &Engine,
    name: &str,
    n_out: usize,
    n_in: usize,
) -> Result<Vec<i64>, ModelError> {
    let t = engine
        .table()
        .get(name)
        .ok_or_else(|| ModelError::Missing(name.to_string()))?;
    if t.dims.len() != 2 || t.dims[0] != n_in as u64 || t.dims[1] != n_out as u64 {
        return Err(ModelError::Shape(format!(
            "{name} dims {:?} != [{n_in}, {n_out}]",
            t.dims
        )));
    }
    let n_bytes = t
        .n_bytes
        .ok_or_else(|| ModelError::Shape(format!("{name} has no geometry")))?;
    let row_bytes = n_bytes / (n_out as u64);
    let raw = engine.payload(t.byte_range.start, n_bytes)?;
    let row_bytes = usize::try_from(row_bytes).map_err(|_| ModelError::Overflow)?;
    let mut out: Vec<i64> = Vec::with_capacity(n_out * n_in);
    for o in 0..n_out {
        let s = o * row_bytes;
        let end = s + row_bytes;
        if end > raw.len() {
            return Err(ModelError::Shape(format!("{name} row {o} past payload")));
        }
        dequant_row_fixed(t.ggml_type, &raw[s..end], n_in, &mut out)?;
    }
    if out.len() != n_out * n_in {
        return Err(ModelError::Shape(format!(
            "{name} produced {} of {} weights",
            out.len(),
            n_out * n_in
        )));
    }
    Ok(out)
}

impl Weights {
    /// Load every weight and the resolved [`Config`] from a loaded [`Engine`].
    ///
    /// # Errors
    ///
    /// [`ModelError::Missing`] for an absent norm, bias, embedding or head tensor;
    /// [`ModelError::Shape`] for a tensor whose stored dims disagree with the geometry or a
    /// dequantization that does not fill its row; [`ModelError::Config`] for a negative or
    /// out-of-range `rms_eps`; [`ModelError::Overflow`] on a length conversion; and
    /// [`ModelError::Engine`] for the engine's payload-read refusals.
    ///
    /// # Complexity
    /// `O(n_layer·(n_embd² + n_embd·n_ff) + vocab·n_embd)` dequantized weight elements.
    pub fn from_engine(engine: &Engine) -> Result<(Weights, Config), ModelError> {
        let ec = &engine.config;
        let cfg = Config {
            n_layer: ec.n_layer,
            n_head: ec.n_head,
            n_head_kv: ec.n_head_kv,
            n_embd: ec.n_embd,
            n_ff: ec.n_ff,
            vocab: ec.vocab_size,
            head_dim: ec.head_dim,
            eps: eps_q32(ec.rms_eps)?,
            rope_theta: ec.rope_theta,
        };
        cfg.checked()?;
        let e = cfg.n_embd;
        let q_out = cfg.q_out()?;
        let kv_out = cfg.kv_out()?;
        let vectors = engine.decode_vectors();
        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let norms = vectors
                .norms_of(l)
                .ok_or_else(|| ModelError::Missing(format!("blk.{l} norms")))?;
            let attn_norm = norms[0]
                .clone()
                .ok_or_else(|| ModelError::Missing(format!("blk.{l}.attn_norm.weight")))?;
            let ffn_norm = norms[1]
                .clone()
                .ok_or_else(|| ModelError::Missing(format!("blk.{l}.ffn_norm.weight")))?;
            let biases = vectors
                .biases_of(l)
                .ok_or_else(|| ModelError::Missing(format!("blk.{l} biases")))?;
            layers.push(LayerWeights {
                attn_norm,
                wq: load_matrix(engine, &format!("blk.{l}.attn_q.weight"), q_out, e)?,
                bq: biases[0].clone(),
                wk: load_matrix(engine, &format!("blk.{l}.attn_k.weight"), kv_out, e)?,
                bk: biases[1].clone(),
                wv: load_matrix(engine, &format!("blk.{l}.attn_v.weight"), kv_out, e)?,
                bv: biases[2].clone(),
                wo: load_matrix(engine, &format!("blk.{l}.attn_output.weight"), e, q_out)?,
                ffn_norm,
                w_gate: load_matrix(engine, &format!("blk.{l}.ffn_gate.weight"), cfg.n_ff, e)?,
                w_up: load_matrix(engine, &format!("blk.{l}.ffn_up.weight"), cfg.n_ff, e)?,
                w_down: load_matrix(engine, &format!("blk.{l}.ffn_down.weight"), e, cfg.n_ff)?,
            });
        }
        let output_norm = vectors
            .output_norm
            .clone()
            .ok_or_else(|| ModelError::Missing("output_norm.weight".to_string()))?;
        let token_embd = load_matrix(engine, "token_embd.weight", cfg.vocab, e)?;
        let head_name = engine.lm_head().name.clone();
        let lm_head = load_matrix(engine, &head_name, cfg.vocab, e)?;
        Ok((
            Weights {
                layers,
                output_norm,
                token_embd,
                lm_head,
            },
            cfg,
        ))
    }
}

/// `.len() == t · w` or [`ModelError::Shape`].
///
/// # Complexity
/// `O(1)`.
fn expect(t: usize, w: usize, len: usize, what: &str) -> Result<usize, ModelError> {
    match t.checked_mul(w) {
        Some(n) if n == len => Ok(n),
        Some(n) => Err(ModelError::Shape(format!("{what}: {len} != {t}·{w} = {n}"))),
        None => Err(ModelError::Overflow),
    }
}

/// Checked `a[i] += b[i]` over equal-width rows.
///
/// # Complexity
/// `O(len)`.
fn add_flat(a: &mut [i64], b: &[i64]) -> Result<(), ModelError> {
    if a.len() != b.len() {
        return Err(ModelError::Shape("add width".to_string()));
    }
    for (x, y) in a.iter_mut().zip(b) {
        *x = x.checked_add(*y).ok_or(ModelError::Overflow)?;
    }
    Ok(())
}

/// The column sums of a `t × w` row-major buffer: `out[c] = Σ_r x[r·w + c]`, checked.
///
/// # Complexity
/// `O(t·w)`.
fn col_sum(x: &[i64], t: usize, w: usize) -> Result<Vec<i64>, ModelError> {
    expect(t, w, x.len(), "col_sum")?;
    let mut out = vec![0i64; w];
    for r in 0..t {
        for c in 0..w {
            out[c] = out[c]
                .checked_add(x[r * w + c])
                .ok_or(ModelError::Overflow)?;
        }
    }
    Ok(out)
}

/// `RMSNorm` with gain over every row of a `t × w` buffer.
///
/// # Complexity
/// `O(t·w)`; one [`dense::rmsnorm_fwd`] per row.
fn rmsnorm_rows(
    x: &[i64],
    gamma: &[i64],
    t: usize,
    w: usize,
    eps: i64,
) -> Result<Vec<i64>, ModelError> {
    expect(t, w, x.len(), "rmsnorm_rows")?;
    if x.is_empty() {
        return Err(ModelError::Shape("rmsnorm_rows empty".to_string()));
    }
    let mut out = Vec::with_capacity(x.len());
    for r in 0..t {
        out.extend_from_slice(&dense::rmsnorm_fwd(&x[r * w..r * w + w], gamma, eps)?);
    }
    Ok(out)
}

/// `RMSNorm` backward over every row of a `t × w` buffer: `(dx, Σ_r dgamma_r)`.
///
/// # Complexity
/// `O(t·w)`; one [`dense::rmsnorm_bwd`] per row.
fn rmsnorm_bwd_rows(
    x: &[i64],
    gamma: &[i64],
    eps: i64,
    dy: &[i64],
    t: usize,
    w: usize,
) -> Result<(Vec<i64>, Vec<i64>), ModelError> {
    expect(t, w, x.len(), "rmsnorm_bwd_rows x")?;
    expect(t, w, dy.len(), "rmsnorm_bwd_rows dy")?;
    let mut dx = vec![0i64; x.len()];
    let mut dg = vec![0i64; w];
    for r in 0..t {
        let (dxr, dgr) =
            dense::rmsnorm_bwd(&x[r * w..r * w + w], gamma, eps, &dy[r * w..r * w + w])?;
        dx[r * w..r * w + w].copy_from_slice(&dxr);
        for c in 0..w {
            dg[c] = dg[c].checked_add(dgr[c]).ok_or(ModelError::Overflow)?;
        }
    }
    Ok((dx, dg))
}

/// `RoPE` forward on every `row`-wide row of a `t × row` buffer at the row's own position.
///
/// # Complexity
/// `O(t·row)`; one [`attn::rope_fwd`] per token.
fn rope_rows_fwd(
    x: &[i64],
    t: usize,
    row: usize,
    rope: &RopeTable,
) -> Result<Vec<i64>, ModelError> {
    expect(t, row, x.len(), "rope_rows_fwd")?;
    let mut out = Vec::with_capacity(x.len());
    for r in 0..t {
        let pos = u32::try_from(r).map_err(|_| ModelError::Overflow)?;
        out.extend_from_slice(&attn::rope_fwd(&x[r * row..r * row + row], pos, rope)?);
    }
    Ok(out)
}

/// `RoPE` backward on every `row`-wide row of a `t × row` gradient buffer.
///
/// # Complexity
/// `O(t·row)`; one [`attn::rope_bwd`] per token.
fn rope_rows_bwd(
    dy: &[i64],
    t: usize,
    row: usize,
    rope: &RopeTable,
) -> Result<Vec<i64>, ModelError> {
    expect(t, row, dy.len(), "rope_rows_bwd")?;
    let mut out = Vec::with_capacity(dy.len());
    for r in 0..t {
        let pos = u32::try_from(r).map_err(|_| ModelError::Overflow)?;
        out.extend_from_slice(&attn::rope_bwd(&dy[r * row..r * row + row], pos, rope)?);
    }
    Ok(out)
}

/// Add a `w`-wide bias vector to every row of a `t × w` buffer, checked.
///
/// # Complexity
/// `O(t·w)`.
fn add_bias(x: &mut [i64], bias: Option<&[i64]>, t: usize, w: usize) -> Result<(), ModelError> {
    let Some(b) = bias else {
        return Ok(());
    };
    if b.len() != w {
        return Err(ModelError::Shape("bias width".to_string()));
    }
    expect(t, w, x.len(), "add_bias")?;
    for r in 0..t {
        for c in 0..w {
            let i = r * w + c;
            x[i] = x[i].checked_add(b[c]).ok_or(ModelError::Overflow)?;
        }
    }
    Ok(())
}

/// Every stage input the backward reads for one layer, captured by [`forward`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerTape {
    /// Hidden before `attn_norm`, `t × n_embd`.
    pub ln1_in: Vec<i64>,
    /// `attn_norm` output (the Q/K/V input), `t × n_embd`.
    pub ln1_out: Vec<i64>,
    /// Query after `RoPE`, `t × q_out`.
    pub q: Vec<i64>,
    /// Key after `RoPE`, `t × kv_out`.
    pub k: Vec<i64>,
    /// Value, `t × kv_out`.
    pub v: Vec<i64>,
    /// Attention probabilities, `n_head × t × t`.
    pub p: Vec<i64>,
    /// Attention output, `t × q_out`.
    pub attn_out: Vec<i64>,
    /// Hidden before `ffn_norm`, `t × n_embd`.
    pub h2: Vec<i64>,
    /// `ffn_norm` output, `t × n_embd`.
    pub ln2_out: Vec<i64>,
    /// `SwiGLU` gate projection, `t × n_ff`.
    pub gate: Vec<i64>,
    /// `SwiGLU` up projection, `t × n_ff`.
    pub up: Vec<i64>,
    /// `SwiGLU` product, `t × n_ff`.
    pub m: Vec<i64>,
}

/// The forward tape: the token sequence, the per-layer stage inputs and the head's inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tape {
    /// Token count.
    pub t: usize,
    /// The token ids the forward ran (the embedding backward's scatter index).
    pub tokens: Vec<u32>,
    /// Per layer, in forward order.
    pub layers: Vec<LayerTape>,
    /// Hidden after every layer (the `output_norm` input), `t × n_embd`.
    pub final_hidden: Vec<i64>,
    /// `output_norm` output (the head input), `t × n_embd`.
    pub final_normed: Vec<i64>,
}

/// The whole-model Qwen2 forward over `tokens`, causal, returning `t × vocab` Q32.32 logits and the
/// [`Tape`] the backward consumes.
///
/// # Errors
///
/// [`ModelError::Shape`] for an empty token sequence or a token id past the vocabulary;
/// [`ModelError::Overflow`] when a checked add leaves `i64` or a length product leaves `usize`;
/// [`ModelError::Attn`] from the `RoPE` table or an attention kernel; [`ModelError::Dense`] from a
/// projection, norm or `SwiGLU` kernel.
///
/// # Complexity
/// Per layer `O(t·n_embd·(n_embd + q_out + 2·kv_out + n_ff) + n_head·t²·head_dim)`; the head
/// `O(t·vocab·n_embd)`. Space `O(t·(n_embd + q_out + kv_out + n_ff) + n_head·t²)` for the tape.
pub fn forward(w: &Weights, cfg: &Config, tokens: &[u32]) -> Result<(Vec<i64>, Tape), ModelError> {
    cfg.checked()?;
    let t = tokens.len();
    if t == 0 {
        return Err(ModelError::Shape("empty token sequence".to_string()));
    }
    let e = cfg.n_embd;
    let q_out = cfg.q_out()?;
    let kv_out = cfg.kv_out()?;
    let n_ff = cfg.n_ff;
    let rope = RopeTable::new(cfg.rope_theta, cfg.head_dim)
        .map_err(|err| ModelError::Attn(AttnError::Rope(err)))?;

    let mut h: Vec<i64> = Vec::with_capacity(t * e);
    for &tok in tokens {
        let id = usize::try_from(tok).map_err(|_| ModelError::Overflow)?;
        if id >= cfg.vocab {
            return Err(ModelError::Shape(format!(
                "token {tok} >= vocab {}",
                cfg.vocab
            )));
        }
        h.extend_from_slice(&w.token_embd[id * e..id * e + e]);
    }

    let mut layers: Vec<LayerTape> = Vec::with_capacity(cfg.n_layer);
    for lw in &w.layers {
        let ln1_in = h.clone();
        let ln1_out = rmsnorm_rows(&h, &lw.attn_norm, t, e, cfg.eps)?;
        let mut q = dense::linear_fwd(&ln1_out, &lw.wq, t, e, q_out)?;
        add_bias(&mut q, lw.bq.as_deref(), t, q_out)?;
        let mut k = dense::linear_fwd(&ln1_out, &lw.wk, t, e, kv_out)?;
        add_bias(&mut k, lw.bk.as_deref(), t, kv_out)?;
        let mut v = dense::linear_fwd(&ln1_out, &lw.wv, t, e, kv_out)?;
        add_bias(&mut v, lw.bv.as_deref(), t, kv_out)?;
        let q = rope_rows_fwd(&q, t, q_out, &rope)?;
        let k = rope_rows_fwd(&k, t, kv_out, &rope)?;
        let (attn_out, p) = attn::attn_fwd(&q, &k, &v, cfg.n_head, cfg.n_head_kv, cfg.head_dim)?;
        let o = dense::linear_fwd(&attn_out, &lw.wo, t, q_out, e)?;
        let mut h2 = h;
        add_flat(&mut h2, &o)?;
        let ln2_out = rmsnorm_rows(&h2, &lw.ffn_norm, t, e, cfg.eps)?;
        let gate = dense::linear_fwd(&ln2_out, &lw.w_gate, t, e, n_ff)?;
        let up = dense::linear_fwd(&ln2_out, &lw.w_up, t, e, n_ff)?;
        let m = dense::swiglu_fwd(&gate, &up)?;
        let d = dense::linear_fwd(&m, &lw.w_down, t, n_ff, e)?;
        let mut h3 = h2.clone();
        add_flat(&mut h3, &d)?;
        h = h3;
        layers.push(LayerTape {
            ln1_in,
            ln1_out,
            q,
            k,
            v,
            p,
            attn_out,
            h2,
            ln2_out,
            gate,
            up,
            m,
        });
    }

    let final_hidden = h;
    let final_normed = rmsnorm_rows(&final_hidden, &w.output_norm, t, e, cfg.eps)?;
    let logits = dense::linear_fwd(&final_normed, &w.lm_head, t, e, cfg.vocab)?;

    Ok((
        logits,
        Tape {
            t,
            tokens: tokens.to_vec(),
            layers,
            final_hidden,
            final_normed,
        },
    ))
}

/// The per-layer gradient buffers, mirroring [`LayerWeights`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerGrads {
    /// `dL/d attn_norm.weight`.
    pub attn_norm: Vec<i64>,
    /// `dL/d attn_q.weight`.
    pub wq: Vec<i64>,
    /// `dL/d attn_q.bias`.
    pub bq: Vec<i64>,
    /// `dL/d attn_k.weight`.
    pub wk: Vec<i64>,
    /// `dL/d attn_k.bias`.
    pub bk: Vec<i64>,
    /// `dL/d attn_v.weight`.
    pub wv: Vec<i64>,
    /// `dL/d attn_v.bias`.
    pub bv: Vec<i64>,
    /// `dL/d attn_output.weight`.
    pub wo: Vec<i64>,
    /// `dL/d ffn_norm.weight`.
    pub ffn_norm: Vec<i64>,
    /// `dL/d ffn_gate.weight`.
    pub w_gate: Vec<i64>,
    /// `dL/d ffn_up.weight`.
    pub w_up: Vec<i64>,
    /// `dL/d ffn_down.weight`.
    pub w_down: Vec<i64>,
}

/// The whole-model gradient buffers, mirroring [`Weights`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grads {
    /// Per trunk layer, in forward order.
    pub layers: Vec<LayerGrads>,
    /// `dL/d output_norm.weight`.
    pub output_norm: Vec<i64>,
    /// `dL/d token_embd.weight` (a scatter-add over the token ids).
    pub token_embd: Vec<i64>,
    /// `dL/d lm_head`.
    pub lm_head: Vec<i64>,
}

/// `dembedding[token[r], :] += dh[r, :]`, checked, over a `t × w` hidden gradient.
///
/// # Complexity
/// `O(t·w)`.
fn embedding_grad(
    dh: &[i64],
    tokens: &[u32],
    t: usize,
    w: usize,
    vocab: usize,
) -> Result<Vec<i64>, ModelError> {
    expect(t, w, dh.len(), "embedding_grad")?;
    let mut out = vec![0i64; vocab * w];
    for r in 0..t {
        let id = usize::try_from(tokens[r]).map_err(|_| ModelError::Overflow)?;
        if id >= vocab {
            return Err(ModelError::Shape(format!("token {} >= vocab", tokens[r])));
        }
        for c in 0..w {
            out[id * w + c] = out[id * w + c]
                .checked_add(dh[r * w + c])
                .ok_or(ModelError::Overflow)?;
        }
    }
    Ok(out)
}

/// The whole-model backward: `dL/dW` for every weight from the forward [`Tape`] and the logit
/// gradient `dlogits` (`t × vocab`).
///
/// # Errors
///
/// [`ModelError::Shape`] when the tape or `dlogits` disagrees with the geometry;
/// [`ModelError::Overflow`] on a checked add or a width mismatch's length product; and the
/// [`ModelError::Dense`] and [`ModelError::Attn`] refusals of the kernels, as in [`forward`].
///
/// # Complexity
/// Mirrors [`forward`]'s `O(t·(n_embd² + n_embd·n_ff) + n_head·t²·head_dim)` per layer plus the head.
pub fn backward(
    w: &Weights,
    cfg: &Config,
    tape: &Tape,
    dlogits: &[i64],
) -> Result<Grads, ModelError> {
    cfg.checked()?;
    let t = tape.t;
    let e = cfg.n_embd;
    let q_out = cfg.q_out()?;
    let kv_out = cfg.kv_out()?;
    let n_ff = cfg.n_ff;
    if tape.layers.len() != cfg.n_layer || tape.tokens.len() != t {
        return Err(ModelError::Shape("tape shape".to_string()));
    }
    expect(t, cfg.vocab, dlogits.len(), "dlogits")?;
    let rope = RopeTable::new(cfg.rope_theta, cfg.head_dim)
        .map_err(|err| ModelError::Attn(AttnError::Rope(err)))?;

    let d_normed = dense::linear_bwd_input(dlogits, &w.lm_head, t, e, cfg.vocab)?;
    let lm_head = dense::linear_bwd_weight(dlogits, &tape.final_normed, t, e, cfg.vocab)?;
    let (mut dh, output_norm) =
        rmsnorm_bwd_rows(&tape.final_hidden, &w.output_norm, cfg.eps, &d_normed, t, e)?;

    let mut grads: Vec<LayerGrads> = Vec::with_capacity(cfg.n_layer);
    for l in (0..cfg.n_layer).rev() {
        let lw = &w.layers[l];
        let lt = &tape.layers[l];
        // h3 = h2 + down(m): the down branch's gradient, and the residual identity that carries it
        // into h2.
        let dd = dense::linear_bwd_input(&dh, &lw.w_down, t, n_ff, e)?;
        let w_down = dense::linear_bwd_weight(&dh, &lt.m, t, n_ff, e)?;
        let (dgate, dup) = dense::swiglu_bwd(&lt.gate, &lt.up, &dd)?;
        let dln2_g = dense::linear_bwd_input(&dgate, &lw.w_gate, t, e, n_ff)?;
        let w_gate = dense::linear_bwd_weight(&dgate, &lt.ln2_out, t, e, n_ff)?;
        let dln2_u = dense::linear_bwd_input(&dup, &lw.w_up, t, e, n_ff)?;
        let w_up = dense::linear_bwd_weight(&dup, &lt.ln2_out, t, e, n_ff)?;
        let mut dln2 = dln2_g;
        add_flat(&mut dln2, &dln2_u)?;
        let (dh2_ffn, ffn_norm) = rmsnorm_bwd_rows(&lt.h2, &lw.ffn_norm, cfg.eps, &dln2, t, e)?;
        // The residual identity: h3's gradient flows unchanged into h2, plus the ffn-norm branch.
        let mut dh2 = dh;
        add_flat(&mut dh2, &dh2_ffn)?;
        let d_attn_out = dense::linear_bwd_input(&dh2, &lw.wo, t, q_out, e)?;
        let wo = dense::linear_bwd_weight(&dh2, &lt.attn_out, t, q_out, e)?;
        let (dq_rope, dk_rope, dv) = attn::attn_bwd(
            &lt.q,
            &lt.k,
            &lt.v,
            &lt.p,
            &d_attn_out,
            cfg.n_head,
            cfg.n_head_kv,
            cfg.head_dim,
        )?;
        let dq = rope_rows_bwd(&dq_rope, t, q_out, &rope)?;
        let dk = rope_rows_bwd(&dk_rope, t, kv_out, &rope)?;
        let bq = col_sum(&dq, t, q_out)?;
        let bk = col_sum(&dk, t, kv_out)?;
        let bv = col_sum(&dv, t, kv_out)?;
        let dln1_q = dense::linear_bwd_input(&dq, &lw.wq, t, e, q_out)?;
        let dln1_k = dense::linear_bwd_input(&dk, &lw.wk, t, e, kv_out)?;
        let dln1_v = dense::linear_bwd_input(&dv, &lw.wv, t, e, kv_out)?;
        let wq = dense::linear_bwd_weight(&dq, &lt.ln1_out, t, e, q_out)?;
        let wk = dense::linear_bwd_weight(&dk, &lt.ln1_out, t, e, kv_out)?;
        let wv = dense::linear_bwd_weight(&dv, &lt.ln1_out, t, e, kv_out)?;
        let mut dln1 = dln1_q;
        add_flat(&mut dln1, &dln1_k)?;
        add_flat(&mut dln1, &dln1_v)?;
        let (dh_attn, attn_norm) =
            rmsnorm_bwd_rows(&lt.ln1_in, &lw.attn_norm, cfg.eps, &dln1, t, e)?;
        add_flat(&mut dh2, &dh_attn)?;
        dh = dh2;
        grads.push(LayerGrads {
            attn_norm,
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            wo,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
        });
    }
    grads.reverse();

    let token_embd = embedding_grad(&dh, &tape.tokens, t, e, cfg.vocab)?;
    Ok(Grads {
        layers: grads,
        output_norm,
        token_embd,
        lm_head,
    })
}

#[cfg(test)]
#[path = "model/tests.rs"]
mod tests;
