//! The engine-reading load: the model's [`Config`] resolved from an [`Engine`], a layer's fitted
//! linears named by `GGUF` tensor suffix, and the whole-model [`Weights::from_engine`] projection.
//!
//! This file is IO glue and stays outside the extraction unit (`proof/extract/grad_solve_port.rs`
//! takes [`super::model`], never this module): every body here reads the engine's directory, its
//! payload or its resolved [`DecodeVectors`](crate::engine::loader::DecodeVectors), none of which the
//! pinned Aeneas translates. The backward and its integer helpers are in [`super::model`].
//!
//! # Evidence grade
//!
//! `tested`: `cargo test --test learn_hyb_load` on the `Qwen3.5-0.8B` proxy; the qwen2 load identity is
//! `tested` by `learn_fold_capture_engine` and `learn_fold_run_convert`.
//!
//! # Trust boundary
//!
//! Not covered: the engine's own `GGUF` parse and quantizer dispatch, which this module calls and does
//! not re-check. A stored dim that disagrees with the geometry is refused with
//! [`ModelError::Shape`]; no tensor extent is re-derived here.

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
              single-letter names are the schedule's symbols (`t`, `e`, `l`, `o`) and the layer \
              loader is one linear composition of the per-mixer name sets"
)]

use crate::engine::{dequant_row_fixed, Engine, EngineError, MixerKind};

use super::model::{Config, LayerWeights, ModelError, Weights};

impl From<EngineError> for ModelError {
    fn from(e: EngineError) -> Self {
        ModelError::Engine(e)
    }
}

impl Config {
    /// The engine's resolved geometry as a [`Config`]: the dimensions, `rms_eps` as a Q32.32 word
    /// and `rope_theta` as host `f64`.
    ///
    /// # Errors
    ///
    /// [`ModelError::Config`] for a negative or out-of-range `rms_eps`; [`ModelError::Shape`] from
    /// [`Config::checked`].
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn from_engine(engine: &Engine) -> Result<Config, ModelError> {
        let ec = &engine.config;
        let cfg = Config {
            n_layer: ec.n_layer,
            n_head: ec.n_head,
            n_head_kv: ec.n_head_kv,
            n_embd: ec.n_embd,
            n_ff: ec.n_ff,
            vocab: ec.vocab_size,
            head_dim: ec.head_dim,
            eps: eps_q32(ec.rms_eps.to_f64())?,
            rope_theta: ec.rope_theta.to_f64(),
        };
        cfg.checked()?;
        Ok(cfg)
    }
}

/// Stage slot of the attention-side `RMSNorm` output in the engine's tap rows: the input of every
/// Q/K/V and qkv/gate projection (`src/engine/step/tap.rs:STAGE_LN1_OUT`, private to the engine
/// step module).
pub const TAP_LN1_OUT: usize = 1;
/// Stage slot of the output projection's input: the attention output, or the affine mixer's value
/// output (`STAGE_ATTN_OUT`).
pub const TAP_ATTN_OUT: usize = 2;
/// Stage slot of the FFN-side `RMSNorm` output, the gate/up input (`STAGE_LN2_OUT`).
pub const TAP_LN2_OUT: usize = 3;
/// Stage slot of the `SwiGLU` product, the down projection's input (`STAGE_M`).
pub const TAP_M: usize = 4;

/// One fitted linear of a layer: the `GGUF` tensor suffix it is, its `rows × cols` shape, the tap
/// slot of its input and its Q32.32 words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Linear {
    /// Tensor-name suffix, e.g. `"attn_qkv"`; the stored tensor is `blk.{l}.{name}.weight`.
    pub name: &'static str,
    /// Output rows.
    pub rows: usize,
    /// Input width, the fastest-varying axis.
    pub cols: usize,
    /// Tap slot of its input: one of [`TAP_LN1_OUT`], [`TAP_ATTN_OUT`], [`TAP_LN2_OUT`], [`TAP_M`].
    pub tap: usize,
    /// Row-major `rows × cols`.
    pub w: Vec<i64>,
    /// `blk.{l}.{name}.bias`, `rows` wide, or `None`.
    pub bias: Option<Vec<i64>>,
}

/// `blk.{l}.{name}.weight` dequantized to a row-major `rows × cols` [`Linear`].
///
/// # Errors
///
/// As [`load_matrix`].
///
/// # Complexity
/// `O(rows·cols)` dequantized elements; one contiguous read of the checked tensor extent.
fn linear(
    engine: &Engine,
    l: usize,
    name: &'static str,
    rows: usize,
    cols: usize,
    tap: usize,
    bias: Option<Vec<i64>>,
) -> Result<Linear, ModelError> {
    let w = load_matrix(engine, &format!("blk.{l}.{name}.weight"), rows, cols)?;
    Ok(Linear {
        name,
        rows,
        cols,
        tap,
        w,
        bias,
    })
}

/// One layer's fitted linears keyed by `GGUF` tensor name, with the two norm gains and the mixer that
/// decides which names are present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerLinears {
    /// The layer's sequence mixer; it selects the attention-side names.
    pub mixer: MixerKind,
    /// `attn_norm.weight`, `n_embd` wide.
    pub attn_norm: Vec<i64>,
    /// `ffn_norm.weight` (`Qwen2`) or `post_attention_norm.weight` (`Qwen3.5`), `n_embd` wide.
    pub ffn_norm: Vec<i64>,
    /// The fitted linears, in the order the writer replaces them.
    pub linears: Vec<Linear>,
}

/// Layer `l`'s fitted linears, named by tensor suffix and shaped by its mixer.
///
/// A softmax layer's set is `attn_q` (rows `2·q_out` when the engine's `attn_gate` is set),
/// `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`, with the stored `Q/K/V`
/// biases. An affine (`Gated DeltaNet`) layer's set is `attn_qkv` (`2·n_k_heads·head_k +
/// n_v_heads·head_v` rows), `attn_gate` (`n_v_heads·head_v`) and `ssm_out` (`n_embd ×
/// n_v_heads·head_v`), with no bias. `ffn_{gate,up,down}` are the same on both. The `ssm_*`
/// vectors, `ssm_conv1d` and the per-head QK norms are not fitted and are not read here.
///
/// # Errors
///
/// [`ModelError::Missing`] for a layer past the mixer schedule, an absent norm or affine geometry,
/// or an absent tensor; [`ModelError::Shape`] for a stored tensor whose dims disagree with the
/// geometry; [`ModelError::Overflow`] on a length product; and [`ModelError::Engine`] for the
/// engine's payload refusals.
///
/// # Complexity
/// `O(Σ_name rows·cols)` dequantized elements: `O((ell + n_ff)·n_embd)` for the softmax set and
/// `O((n_embd + n_ff)·n_embd)` for the affine set, `ell` the attention width.
pub fn layer_from_engine(
    engine: &Engine,
    cfg: &Config,
    l: usize,
) -> Result<LayerLinears, ModelError> {
    if l >= cfg.n_layer {
        return Err(ModelError::Shape(format!(
            "layer {l} past the trunk's {} layers",
            cfg.n_layer
        )));
    }
    let ec = &engine.config;
    let mixer = *ec
        .mixers
        .get(l)
        .ok_or_else(|| ModelError::Missing(format!("blk.{l} mixer")))?;
    let vectors = engine.decode_vectors();
    let norms = vectors
        .norms_of(l)
        .ok_or_else(|| ModelError::Missing(format!("blk.{l} norms")))?;
    let attn_norm = norms[0]
        .clone()
        .ok_or_else(|| ModelError::Missing(format!("blk.{l}.attn_norm.weight")))?;
    let ffn_norm = norms[1]
        .clone()
        .ok_or_else(|| ModelError::Missing(format!("blk.{l}.ffn_norm.weight")))?;
    let e = cfg.n_embd;
    let n_ff = cfg.n_ff;
    let mut linears: Vec<Linear> = Vec::with_capacity(7);
    match mixer {
        MixerKind::Softmax => {
            let biases = vectors
                .biases_of(l)
                .ok_or_else(|| ModelError::Missing(format!("blk.{l} biases")))?;
            let q_out = cfg.q_out()?;
            let kv_out = cfg.kv_out()?;
            let q_rows = if ec.attn_gate {
                q_out.checked_mul(2).ok_or(ModelError::Overflow)?
            } else {
                q_out
            };
            linears.push(linear(
                engine,
                l,
                "attn_q",
                q_rows,
                e,
                TAP_LN1_OUT,
                biases[0].clone(),
            )?);
            linears.push(linear(
                engine,
                l,
                "attn_k",
                kv_out,
                e,
                TAP_LN1_OUT,
                biases[1].clone(),
            )?);
            linears.push(linear(
                engine,
                l,
                "attn_v",
                kv_out,
                e,
                TAP_LN1_OUT,
                biases[2].clone(),
            )?);
            linears.push(linear(
                engine,
                l,
                "attn_output",
                e,
                q_out,
                TAP_ATTN_OUT,
                None,
            )?);
        }
        MixerKind::Affine => {
            let g = ec
                .affine
                .ok_or_else(|| ModelError::Missing(format!("blk.{l} affine geometry")))?;
            let keys = g
                .n_k_heads
                .checked_mul(g.head_k)
                .ok_or(ModelError::Overflow)?;
            let vals = g
                .n_v_heads
                .checked_mul(g.head_v)
                .ok_or(ModelError::Overflow)?;
            let qkv_rows = keys
                .checked_mul(2)
                .and_then(|k2| k2.checked_add(vals))
                .ok_or(ModelError::Overflow)?;
            linears.push(linear(
                engine,
                l,
                "attn_qkv",
                qkv_rows,
                e,
                TAP_LN1_OUT,
                None,
            )?);
            linears.push(linear(engine, l, "attn_gate", vals, e, TAP_LN1_OUT, None)?);
            linears.push(linear(engine, l, "ssm_out", e, vals, TAP_ATTN_OUT, None)?);
        }
    }
    linears.push(linear(engine, l, "ffn_gate", n_ff, e, TAP_LN2_OUT, None)?);
    linears.push(linear(engine, l, "ffn_up", n_ff, e, TAP_LN2_OUT, None)?);
    linears.push(linear(engine, l, "ffn_down", e, n_ff, TAP_M, None)?);
    Ok(LayerLinears {
        mixer,
        attn_norm,
        ffn_norm,
        linears,
    })
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
/// # Errors
///
/// [`ModelError::Missing`] for an absent tensor, [`ModelError::Shape`] for stored dims other than
/// `[n_in, n_out]` or a payload short of a row, and [`ModelError::Engine`] for the engine's
/// payload and dequantizer refusals.
///
/// # Complexity
/// `O(n_out·n_in)` dequantized elements; one contiguous read of the checked tensor extent.
pub fn load_matrix(
    engine: &Engine,
    name: &str,
    n_out: usize,
    n_in: usize,
) -> Result<Vec<i64>, ModelError> {
    let rows: Vec<usize> = (0..n_out).collect();
    load_rows(engine, name, n_out, n_in, &rows)
}

/// Rows `rows` (in order, repeats allowed) of the `[n_in, n_out]` quantized tensor `name`,
/// dequantized to a row-major `rows.len() × n_in` Q32.32 matrix; an embedding lookup when `name`
/// is `token_embd.weight` and `rows` are token ids.
///
/// # Errors
///
/// As [`load_matrix`], plus [`ModelError::Shape`] for a row index `≥ n_out`.
///
/// # Complexity
/// `O(rows.len()·n_in)` dequantized elements; the tensor extent is mapped, not copied.
pub fn load_rows(
    engine: &Engine,
    name: &str,
    n_out: usize,
    n_in: usize,
    rows: &[usize],
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
    let mut out: Vec<i64> = Vec::with_capacity(rows.len() * n_in);
    for &o in rows {
        let s = o * row_bytes;
        let end = s + row_bytes;
        if o >= n_out || end > raw.len() {
            return Err(ModelError::Shape(format!("{name} row {o} past payload")));
        }
        dequant_row_fixed(t.ggml_type, &raw[s..end], n_in, &mut out)?;
    }
    if out.len() != rows.len() * n_in {
        return Err(ModelError::Shape(format!(
            "{name} produced {} of {} weights",
            out.len(),
            rows.len() * n_in
        )));
    }
    Ok(out)
}

impl Weights {
    /// Load every weight and the resolved [`Config`] from a loaded [`Engine`]: one
    /// [`layer_from_engine`] call per trunk layer, projected onto [`LayerWeights`] by tensor name.
    ///
    /// A trunk layer whose mixer is [`MixerKind::Affine`] is refused: the fitted backward below is
    /// the softmax-attention schedule, so an affine layer's `attn_q` does not exist and the refusal
    /// names it.
    ///
    /// # Errors
    ///
    /// [`ModelError::Missing`] for an affine layer, an absent norm, bias, embedding or head tensor;
    /// [`ModelError::Shape`] for a tensor whose stored dims disagree with the geometry or a
    /// dequantization that does not fill its row; [`ModelError::Config`] for a negative or
    /// out-of-range `rms_eps`; [`ModelError::Overflow`] on a length conversion; and
    /// [`ModelError::Engine`] for the engine's payload-read refusals.
    ///
    /// # Complexity
    /// `O(n_layer·(n_embd² + n_embd·n_ff) + vocab·n_embd)` dequantized weight elements.
    pub fn from_engine(engine: &Engine) -> Result<(Weights, Config), ModelError> {
        let cfg = Config::from_engine(engine)?;
        let e = cfg.n_embd;
        let vectors = engine.decode_vectors();
        let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let ll = layer_from_engine(engine, &cfg, l)?;
            if ll.mixer == MixerKind::Affine {
                return Err(ModelError::Missing(format!("blk.{l}.attn_q.weight")));
            }
            let mut linears = ll.linears;
            let wq = take(&mut linears, l, "attn_q")?;
            let wk = take(&mut linears, l, "attn_k")?;
            let wv = take(&mut linears, l, "attn_v")?;
            let wo = take(&mut linears, l, "attn_output")?;
            let w_gate = take(&mut linears, l, "ffn_gate")?;
            let w_up = take(&mut linears, l, "ffn_up")?;
            let w_down = take(&mut linears, l, "ffn_down")?;
            layers.push(LayerWeights {
                attn_norm: ll.attn_norm,
                wq: wq.w,
                bq: wq.bias,
                wk: wk.w,
                bk: wk.bias,
                wv: wv.w,
                bv: wv.bias,
                wo: wo.w,
                ffn_norm: ll.ffn_norm,
                w_gate: w_gate.w,
                w_up: w_up.w,
                w_down: w_down.w,
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

/// The linear named `name` of layer `l`, removed from `linears`, or [`ModelError::Missing`] naming
/// the stored tensor.
///
/// # Complexity
/// `O(linears.len())` name comparisons and one swap-remove.
fn take(linears: &mut Vec<Linear>, l: usize, name: &str) -> Result<Linear, ModelError> {
    let Some(i) = linears.iter().position(|x| x.name == name) else {
        return Err(ModelError::Missing(format!("blk.{l}.{name}.weight")));
    };
    Ok(linears.swap_remove(i))
}
