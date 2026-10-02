//! GGUF engine loader: Qwen2 config resolution, a named tensor table over the file bytes and the
//! bounds arithmetic the embedding reader relies on.
//!
//! The loader reads the header through [`crate::format::gguf`] and resolves the `qwen2.*` metadata
//! into an [`EngineConfig`]. It then walks the tensor directory once, records every entry as
//! `(name, ggml type, dims, quant kind, byte range)` and refuses the load if any entry's range
//! leaves the file. Quant unpacking lives in `crate::format::quant`; this module only records the
//! directory extent and the exact geometry where that module has it.
//!
//! Payloads are resident: [`Engine::open`] reads the whole file once into a `Box<[u8]>`, and
//! [`Engine::payload`] serves a borrowed slice of it with the same checked bounds as
//! [`Engine::read_range`]; no directory value is used before it is checked, and no failure is a
//! panic. [`Engine::read_range`] is retained for callers that need owned bytes. Models larger than
//! RAM (an mmap or streaming path) are a later task.

use core::ops::Range;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex};

use crate::format::gguf::{GVal, Gguf, GgufHeader};
use crate::format::quant;

pub use super::error::EngineError;
use super::f32_convert::dequant_f32_row;
use super::kern::lut::Pq2Lut;
use super::value::round_shift_32;

/// The `general.architecture` this loader resolves config keys for.
pub const QWEN2_ARCH: &str = "qwen2";
/// Default `RoPE` base when `*.rope.freq_base` is absent.
pub const DEFAULT_ROPE_THETA: f64 = 10_000.0;
/// Default `RMSNorm` epsilon when `*.attention.layer_norm_rms_epsilon` is absent.
pub const DEFAULT_RMS_EPS: f64 = 1e-6;

/// The ggml quant families this crate names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    /// ggml `F32`.
    F32,
    /// ggml `F16`.
    F16,
    /// ggml `Q4_K`, a 256-weight affine super-block.
    Q4K,
    /// ggml `Q6_K`, a 256-weight affine super-block.
    Q6K,
    /// ggml `Q8_0`, a 32-weight block with one `f16` scale.
    Q8_0,
    /// ggml `Q5_0`, a 32-weight block with one `f16` scale, a high-bit plane and nibbles.
    Q5_0,
    /// A ggml type `crate::format::quant` has no geometry for; the raw id is kept.
    Other(u32),
}

impl QuantKind {
    /// Classify a ggml type id.
    #[must_use]
    pub fn from_ggml(t: u32) -> QuantKind {
        match t {
            quant::GGML_TYPE_F32 => QuantKind::F32,
            quant::GGML_TYPE_F16 => QuantKind::F16,
            quant::GGML_TYPE_Q4_K => QuantKind::Q4K,
            quant::GGML_TYPE_Q6_K => QuantKind::Q6K,
            quant::GGML_TYPE_Q8_0 => QuantKind::Q8_0,
            quant::GGML_TYPE_Q5_0 => QuantKind::Q5_0,
            other => QuantKind::Other(other),
        }
    }

    /// The ggml name, for diagnostics.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            QuantKind::F32 => "F32",
            QuantKind::F16 => "F16",
            QuantKind::Q4K => "Q4_K",
            QuantKind::Q6K => "Q6_K",
            QuantKind::Q8_0 => "Q8_0",
            QuantKind::Q5_0 => "Q5_0",
            QuantKind::Other(_) => "OTHER",
        }
    }
}

/// The sequence mixer a layer runs: one of the two carriers of the mixer primitive.
///
/// `Softmax` keeps the append-only key/value list and reads it with softmax attention; `Affine`
/// keeps a fixed-size state updated by an affine map `S ← A_t·S + B_t` and read by `Sᵀq` (Gated
/// `DeltaNet` in `qwen35`). A model is a per-layer schedule of these, not a separate code path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixerKind {
    /// Softmax attention over the append-only key/value list.
    Softmax,
    /// The fixed-size affine-state mixer.
    Affine,
}

/// Geometry of the affine-state mixer (`*.ssm.*` metadata).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineGeom {
    /// `*.ssm.conv_kernel`: taps of the causal depthwise convolution.
    pub conv_kernel: usize,
    /// `*.ssm.group_count`: key heads.
    pub n_k_heads: usize,
    /// `*.ssm.time_step_rank`: value heads (one decay and one β each).
    pub n_v_heads: usize,
    /// `*.ssm.state_size`: key width per head.
    pub head_k: usize,
    /// `*.ssm.inner_size / time_step_rank`: value width per head.
    pub head_v: usize,
}

/// The resolved model configuration: dimensions, the per-layer mixer schedule and the attention
/// flags that distinguish one decoder from another.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig {
    /// `general.architecture`.
    pub architecture: String,
    /// `*.block_count`.
    pub n_layer: usize,
    /// `*.attention.head_count`.
    pub n_head: usize,
    /// `*.attention.head_count_kv`.
    pub n_head_kv: usize,
    /// `*.embedding_length`.
    pub n_embd: usize,
    /// `*.feed_forward_length`.
    pub n_ff: usize,
    /// Vocabulary size, from `token_embd.weight` dims and cross-checked against the token array.
    pub vocab_size: usize,
    /// `*.context_length`, or `0` when unstated.
    pub context_length: usize,
    /// `*.rope.freq_base`.
    pub rope_theta: f64,
    /// `*.attention.layer_norm_rms_epsilon`.
    pub rms_eps: f64,
    /// `*.attention.key_length` when stated, else `n_embd / n_head` (checked integral).
    pub head_dim: usize,
    /// Per trunk layer, the mixer it runs. `*.full_attention_interval` = `f` makes layer `l` softmax
    /// iff `(l + 1) % f == 0` and affine otherwise; without the key every layer is softmax.
    pub mixers: Vec<MixerKind>,
    /// Leading dimensions of each head that `RoPE` rotates: `*.rope.dimension_count`, else `head_dim`.
    pub rot_dims: usize,
    /// Whether `attn_q` also emits a per-head sigmoid output gate: its rows are `2·n_head·head_dim`.
    pub attn_gate: bool,
    /// The affine mixer's geometry, present iff some layer is [`MixerKind::Affine`].
    pub affine: Option<AffineGeom>,
    /// `*.nextn_predict_layers`: multi-token-prediction blocks after the trunk (`blk.{n_layer}..`),
    /// softmax-attention blocks the speculative decoder drafts with; `0` when absent.
    pub n_nextn: usize,
}

/// One tensor-directory entry with its resolved quant kind and checked file extent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    /// GGUF name, e.g. `token_embd.weight`.
    pub name: String,
    /// ggml type id as stored.
    pub ggml_type: u32,
    /// The named quant family.
    pub quant: QuantKind,
    /// Dimensions as stored, fastest-varying first.
    pub dims: Vec<u64>,
    /// Total element count, checked.
    pub n_elements: u64,
    /// Exact storage bytes when `crate::format::quant` has the geometry; `None` otherwise.
    pub n_bytes: Option<u64>,
    /// Absolute half-open byte range: from this entry's data offset to the next entry's data
    /// offset in directory-offset order (the file end for the last entry).
    pub byte_range: Range<u64>,
}

/// The named tensor directory.
#[derive(Debug, Clone)]
pub struct TensorTable {
    entries: Vec<TensorEntry>,
}

impl TensorTable {
    /// The entry with `name`, compared by bytes.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.entries
            .iter()
            .find(|t| t.name.as_bytes() == name.as_bytes())
    }

    /// Every entry, in file directory order.
    #[must_use]
    pub fn entries(&self) -> &[TensorEntry] {
        &self.entries
    }

    /// The number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the directory is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The per-layer `F32` gain and attention-bias vectors the decode step needs, resolved to Q32.32
/// once at load.
///
/// The step composes the fold `cpu_rms` gain (`attn_norm.weight`, `ffn_norm.weight`,
/// `output_norm.weight`) and the Qwen2 attention biases (`attn_{q,k,v}.bias`). Those are the only
/// `F32` tensors on the decode path; reading them per layer per token would be I/O-bound, so the
/// loader resolves them once. An absent optional bias stays `None` (the fold `if let Some(b)`
/// arms); a present tensor with the wrong shape or type refuses at load with
/// [`EngineError::BadVector`], and a non-finite or out-of-range value with [`EngineError::F32`].
#[derive(Debug, Clone, Default)]
pub struct DecodeVectors {
    /// Per layer: `[attn_norm.weight, ffn_norm.weight]`, each `n_embd` wide.
    pub norms: Vec<[Option<Vec<i64>>; 2]>,
    /// Per layer: `[attn_q.bias, attn_k.bias, attn_v.bias]`; Q is `n_head·head_dim` wide, K and V
    /// are `n_head_kv·head_dim`.
    pub biases: Vec<[Option<Vec<i64>>; 3]>,
    /// Per layer: `[attn_q_norm.weight, attn_k_norm.weight]`, the Qwen3 per-head QK `RMSNorm`
    /// gains, each `head_dim` wide. Both are `None` on a Qwen2 model, which has no such tensor.
    pub qk_norms: Vec<[Option<Vec<i64>>; 2]>,
    /// `output_norm.weight`, `n_embd` wide.
    pub output_norm: Option<Vec<i64>>,
    /// Per layer: the affine mixer's `F32` vectors, `Some` exactly on [`MixerKind::Affine`] layers.
    pub affine: Vec<Option<AffineVectors>>,
    /// Per multi-token-prediction block: `[enorm, hnorm, shared_head_norm]`, each `n_embd` wide.
    pub nextn: Vec<[Option<Vec<i64>>; 3]>,
}

/// One affine-mixer layer's `F32` parameters in Q32.32.
#[derive(Debug, Clone, Default)]
pub struct AffineVectors {
    /// `ssm_conv1d.weight`, channel-major: tap `t` of channel `c` at `c·conv_kernel + t`.
    pub conv: Vec<i64>,
    /// `ssm_dt.bias`, one per value head: added to the decay input before `softplus`.
    pub dt: Vec<i64>,
    /// `ssm_a`, one per value head: `−exp(A_log)`, the decay coefficient.
    pub a: Vec<i64>,
    /// `ssm_norm.weight`, `head_v` wide: the gated output norm's gain.
    pub norm: Vec<i64>,
}

impl DecodeVectors {
    /// `[attn_norm, ffn_norm]` for `layer`, or `None` past the layer count.
    #[must_use]
    pub fn norms_of(&self, layer: usize) -> Option<&[Option<Vec<i64>>; 2]> {
        self.norms.get(layer)
    }

    /// `[q, k, v]` biases for `layer`, or `None` past the layer count.
    #[must_use]
    pub fn biases_of(&self, layer: usize) -> Option<&[Option<Vec<i64>>; 3]> {
        self.biases.get(layer)
    }

    /// `[attn_q_norm, attn_k_norm]` for `layer`, or `None` past the layer count.
    #[must_use]
    pub fn qk_norms_of(&self, layer: usize) -> Option<&[Option<Vec<i64>>; 2]> {
        self.qk_norms.get(layer)
    }

    /// Apply `layer`'s per-head QK `RMSNorm` to `query` and `key`, before `RoPE`.
    ///
    /// A model with no QK-norm tensors (Qwen2) gets `None` gains and both vectors are unchanged;
    /// Qwen3 gets the `head_dim`-wide gains applied per `hd`-wide head by [`qk_norm_heads`].
    ///
    /// # Errors
    ///
    /// As [`qk_norm_heads`].
    pub(crate) fn apply_qk_norms(
        &self,
        layer: usize,
        query: &mut [i64],
        key: &mut [i64],
        hd: usize,
        eps: i64,
    ) -> Result<(), super::StepError> {
        let qk = self.qk_norms_of(layer);
        qk_norm_heads(qk.and_then(|n| n[0].as_deref()), query, hd, eps)?;
        qk_norm_heads(qk.and_then(|n| n[1].as_deref()), key, hd, eps)
    }
}

/// The Qwen3 per-head QK `RMSNorm` gain: apply [`rms_norm_fixed`](super::norm::rms_norm_fixed) to
/// each `hd`-wide head of `x` in place, then the `head_dim`-wide per-channel gain. `gain` is `None`
/// on a model with no QK-norm tensor, where `x` is unchanged.
///
/// A `None` gain is a no-op; a gain whose width is not `hd`, a non-multiple-of-`hd` vector and the
/// norm's own range refusals propagate through [`super::StepError`].
///
/// # Complexity
///
/// `O(x.len())` time for the norm and the gain, `O(hd)` scratch per head.
pub(crate) fn qk_norm_heads(
    gain: Option<&[i64]>,
    x: &mut [i64],
    hd: usize,
    eps: i64,
) -> Result<(), super::StepError> {
    let Some(gain) = gain else {
        return Ok(());
    };
    if hd == 0 || gain.len() != hd || !x.len().is_multiple_of(hd) {
        return Err(super::StepError::BadConfig("qk norm geometry"));
    }
    let heads = x.len() / hd;
    let mut h = 0usize;
    while h < heads {
        let base = h * hd;
        let normed = super::norm::rms_norm_fixed(&x[base..base + hd], eps)?;
        let mut i = 0usize;
        while i < hd {
            let prod = i128::from(normed[i])
                .checked_mul(i128::from(gain[i]))
                .ok_or(super::StepError::Overflow)?;
            x[base + i] =
                i64::try_from(round_shift_32(prod)).map_err(|_| super::StepError::Overflow)?;
            i += 1;
        }
        h += 1;
    }
    Ok(())
}

/// The `normalized-sylvester-walsh-hadamard` transform name the fork's runtime accepts.
const HADAMARD_TRANSFORM: &str = "normalized-sylvester-walsh-hadamard";
/// The `input-last-dimension` axis name the fork's runtime accepts.
const HADAMARD_AXIS: &str = "input-last-dimension";

/// The `PrismML` `prism.hadamard.*` fold contract (`llama-model.cpp:1197-1263`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HadamardMeta {
    /// `prism.hadamard.version`: `1` for an untied output, `2` for a tied output.
    pub version: u32,
    /// `prism.hadamard.block_size`: the rotation block width, a nonzero power of two.
    pub block_size: u32,
    /// `prism.hadamard.tied_output`: true iff the output head reuses the rotated embedding.
    pub tied_output: bool,
    /// `prism.hadamard.weight_names`: folded tensor names, never empty.
    pub weight_names: Vec<String>,
    /// `prism.hadamard.sign_widths` paired with `sign_values`; `None` for `sign_mode = "identity"`.
    pub signs: Option<Vec<(u32, Vec<i8>)>>,
}

/// `h.meta(key)` as an `i64`; `Ok(None)` when absent, a refusal when present but not an `INT` type
/// in `i64` range.
fn hadamard_i64(h: &GgufHeader, key: &str) -> Result<Option<i64>, EngineError> {
    match h.meta(key) {
        None => Ok(None),
        Some(GVal::Int(x, _)) => i64::try_from(*x)
            .map(Some)
            .map_err(|_| EngineError::BadMetadata(format!("{key} is not an integer"))),
        Some(_) => Err(EngineError::BadMetadata(format!("{key} is not an integer"))),
    }
}

fn hadamard_require_i64(h: &GgufHeader, key: &str) -> Result<i64, EngineError> {
    hadamard_i64(h, key)?.ok_or_else(|| EngineError::MissingMetadata(key.to_string()))
}

fn hadamard_require_string(h: &GgufHeader, key: &str) -> Result<String, EngineError> {
    match h.meta(key) {
        None => Err(EngineError::MissingMetadata(key.to_string())),
        Some(GVal::Str(s)) => Ok(String::from_utf8_lossy(s).into_owned()),
        Some(_) => Err(EngineError::BadMetadata(format!("{key} is not a string"))),
    }
}

/// A string-array key; `Ok(None)` when absent. Cost is O(elements) time and space.
fn hadamard_string_array(h: &GgufHeader, key: &str) -> Result<Option<Vec<String>>, EngineError> {
    match h.meta(key) {
        None => Ok(None),
        Some(GVal::Arr(_, elems)) => {
            let mut out: Vec<String> = Vec::with_capacity(elems.len());
            let mut i = 0usize;
            while i < elems.len() {
                match &elems[i] {
                    GVal::Str(s) => out.push(String::from_utf8_lossy(s).into_owned()),
                    _ => {
                        return Err(EngineError::BadMetadata(format!(
                            "{key}[{i}] is not a string"
                        )))
                    }
                }
                i += 1;
            }
            Ok(Some(out))
        }
        Some(_) => Err(EngineError::BadMetadata(format!("{key} is not an array"))),
    }
}

fn hadamard_int_array(h: &GgufHeader, key: &str) -> Result<Option<Vec<i64>>, EngineError> {
    match h.meta(key) {
        None => Ok(None),
        Some(GVal::Arr(_, elems)) => {
            let mut out: Vec<i64> = Vec::with_capacity(elems.len());
            let mut i = 0usize;
            while i < elems.len() {
                match elems[i].as_i64() {
                    Some(v) => out.push(v),
                    None => {
                        return Err(EngineError::BadMetadata(format!(
                            "{key}[{i}] is not an integer"
                        )))
                    }
                }
                i += 1;
            }
            Ok(Some(out))
        }
        Some(_) => Err(EngineError::BadMetadata(format!("{key} is not an array"))),
    }
}

/// Validate the `explicit` sign table into `(width, values)` pairs; cost is O(sum of widths).
fn hadamard_signs(h: &GgufHeader, block_size: u32) -> Result<Vec<(u32, Vec<i8>)>, EngineError> {
    let widths = hadamard_int_array(h, "prism.hadamard.sign_widths")?.unwrap_or_default();
    let values = hadamard_int_array(h, "prism.hadamard.sign_values")?.unwrap_or_default();
    if widths.is_empty() {
        return Err(EngineError::BadMetadata(
            "prism.hadamard.sign_mode is explicit but sign_widths is empty".to_string(),
        ));
    }
    let mut out: Vec<(u32, Vec<i8>)> = Vec::with_capacity(widths.len());
    let mut off = 0usize;
    let mut i = 0usize;
    while i < widths.len() {
        let width = widths[i];
        let width_u32 = u32::try_from(width)
            .ok()
            .filter(|w| *w != 0 && w % block_size == 0);
        let Some(width_u32) = width_u32 else {
            return Err(EngineError::BadMetadata(format!(
                "invalid prism.hadamard sign width {width}: not a positive multiple of block size {block_size}"
            )));
        };
        let width_usize = width_u32 as usize;
        if off + width_usize > values.len() {
            return Err(EngineError::BadMetadata(format!(
                "invalid prism.hadamard sign width {width}: sign_values is too short"
            )));
        }
        let mut vec: Vec<i8> = Vec::with_capacity(width_usize);
        let mut j = off;
        while j < off + width_usize {
            let v = values[j];
            if v != 1 && v != -1 {
                return Err(EngineError::BadMetadata(
                    "prism.hadamard sign values must be +/-1".to_string(),
                ));
            }
            vec.push(if v == 1 { 1i8 } else { -1i8 });
            j += 1;
        }
        out.push((width_u32, vec));
        off += width_usize;
        i += 1;
    }
    if off != values.len() {
        return Err(EngineError::BadMetadata(
            "prism.hadamard.sign_values length mismatch".to_string(),
        ));
    }
    Ok(out)
}

/// Read and validate the `prism.hadamard.*` metadata: an absent version key is `Ok(None)`, and
/// otherwise every value the fork's loader rejects is a refusal. Cost is O(metadata size).
///
/// # Errors
///
/// [`EngineError::MissingMetadata`] or [`EngineError::BadMetadata`] when the fork would refuse.
pub fn hadamard_meta(h: &GgufHeader) -> Result<Option<HadamardMeta>, EngineError> {
    let Some(version) = hadamard_i64(h, "prism.hadamard.version")? else {
        return Ok(None);
    };
    let version = u32::try_from(version).map_err(|_| {
        EngineError::BadMetadata(format!("prism.hadamard.version {version} is not 1 or 2"))
    })?;
    if version != 1 && version != 2 {
        return Err(EngineError::BadMetadata(format!(
            "prism.hadamard.version {version} is not 1 or 2"
        )));
    }

    let tied_output = match h.meta("prism.hadamard.tied_output") {
        None => false,
        Some(GVal::Bool(b)) => *b,
        Some(_) => {
            return Err(EngineError::BadMetadata(
                "prism.hadamard.tied_output is not a bool".to_string(),
            ))
        }
    };
    if (version == 2) != tied_output {
        return Err(EngineError::BadMetadata(
            "prism.hadamard version 2 requires tied_output=true; version 1 forbids it".to_string(),
        ));
    }

    let block_size = hadamard_require_i64(h, "prism.hadamard.block_size")?;
    let block_size = u32::try_from(block_size).map_err(|_| {
        EngineError::BadMetadata("prism.hadamard.block_size is not a positive integer".to_string())
    })?;
    if block_size == 0 || (block_size & (block_size - 1)) != 0 {
        return Err(EngineError::BadMetadata(format!(
            "prism.hadamard.block_size {block_size} is zero or not a power of two"
        )));
    }

    let transform = hadamard_require_string(h, "prism.hadamard.transform")?;
    if transform != HADAMARD_TRANSFORM {
        return Err(EngineError::BadMetadata(format!(
            "unsupported prism.hadamard.transform {transform:?}"
        )));
    }
    let axis = hadamard_require_string(h, "prism.hadamard.axis")?;
    if axis != HADAMARD_AXIS {
        return Err(EngineError::BadMetadata(format!(
            "unsupported prism.hadamard.axis {axis:?}"
        )));
    }
    let sign_mode = hadamard_require_string(h, "prism.hadamard.sign_mode")?;
    if sign_mode != "identity" && sign_mode != "explicit" {
        return Err(EngineError::BadMetadata(format!(
            "unsupported prism.hadamard.sign_mode {sign_mode:?}"
        )));
    }

    let weight_names = hadamard_string_array(h, "prism.hadamard.weight_names")?
        .ok_or_else(|| EngineError::MissingMetadata("prism.hadamard.weight_names".to_string()))?;
    if weight_names.is_empty() {
        return Err(EngineError::BadMetadata(
            "prism.hadamard.weight_names is empty".to_string(),
        ));
    }

    let signs = if sign_mode == "explicit" {
        Some(hadamard_signs(h, block_size)?)
    } else {
        None
    };

    Ok(Some(HadamardMeta {
        version,
        block_size,
        tied_output,
        weight_names,
        signs,
    }))
}

/// A loaded GGUF: resolved config, checked tensor directory and the resident file bytes.
#[derive(Debug)]
pub struct Engine {
    /// Path the file was opened from.
    pub path: String,
    /// Total file size in bytes.
    pub file_len: u64,
    /// The resolved Qwen2.5 config.
    pub config: EngineConfig,
    /// The validated `prism.hadamard.*` fold contract, or `None` for an unrotated model.
    pub hadamard: Option<HadamardMeta>,
    /// The whole file, read once at open; [`Engine::payload`] serves checked slices of it.
    data: Box<[u8]>,
    table: TensorTable,
    token_embd: usize,
    vectors: DecodeVectors,
    /// The `PQ2_0` lookup tables, built once per tensor on first projection and reused for every
    /// later token. A plain name-keyed map: a decode step reads each tensor's table on every token.
    pq2_luts: Mutex<HashMap<String, Arc<Pq2Lut>>>,
    /// The head split into bit planes, built on first greedy `Q8Act` step; `None` inside when the
    /// head is not `Q8_0` or the model carries a Hadamard fold.
    bitplane: std::sync::OnceLock<Option<Arc<super::kern::bitplane::BitplaneHead>>>,
}

/// `header.meta_i64("<arch>.<suffix>")`, refusing when absent or non-integral.
fn require_i64(h: &GgufHeader, arch: &str, suffix: &str) -> Result<i64, EngineError> {
    let key = format!("{arch}.{suffix}");
    h.meta_i64(&key).ok_or(EngineError::MissingMetadata(key))
}

/// `header.meta("<arch>.<suffix>")`, interpreted as an `f64` at this boundary, or `default` when
/// absent.
///
/// [`crate::format::gguf`] keeps `FLOAT32`/`FLOAT64` metadata as raw bits so its parse path stays
/// integer-only; the widening to `f64` happens here, where the engine consumes the value. An
/// integer limb is still accepted, matching the previous accessor's domain.
#[allow(
    clippy::cast_precision_loss,
    reason = "the integer-limb arm reproduces the format accessor's prior widening; GGUF stores \
              these keys as FLOAT32/FLOAT64, so the lossy arm is a nonconforming-file fallback"
)]
fn optional_f64(h: &GgufHeader, arch: &str, suffix: &str, default: f64) -> f64 {
    let key = format!("{arch}.{suffix}");
    match h.meta(&key) {
        Some(GVal::F32Bits(bits, _)) => f64::from(f32::from_bits(*bits)),
        Some(GVal::F64Bits(bits, _)) => f64::from_bits(*bits),
        Some(GVal::Int(v, _)) => *v as f64,
        _ => default,
    }
}

/// An `i64` metadata field as a positive `usize`.
fn positive_usize(v: i64, key: &str) -> Result<usize, EngineError> {
    if v <= 0 {
        return Err(EngineError::BadMetadata(key.to_string()));
    }
    usize::try_from(v).map_err(|_| EngineError::BadMetadata(key.to_string()))
}

/// Resolve the `qwen2.*` metadata and the `token_embd.weight` shape into an [`EngineConfig`].
fn resolve_config(h: &GgufHeader) -> Result<EngineConfig, EngineError> {
    let architecture = String::from_utf8_lossy(h.arch()).into_owned();
    let n_block = positive_usize(require_i64(h, &architecture, "block_count")?, "block_count")?;
    // Multi-token-prediction blocks follow the trunk and are not part of the decode step.
    let n_nextn = match h.meta_i64(&format!("{architecture}.nextn_predict_layers")) {
        Some(v) if v >= 0 => usize::try_from(v).map_err(|_| EngineError::LengthOverflow)?,
        Some(_) => return Err(EngineError::BadMetadata("nextn_predict_layers".to_string())),
        None => 0,
    };
    let n_layer = n_block
        .checked_sub(n_nextn)
        .filter(|&n| n > 0)
        .ok_or_else(|| EngineError::BadMetadata("block_count".to_string()))?;
    let n_head = positive_usize(
        require_i64(h, &architecture, "attention.head_count")?,
        "attention.head_count",
    )?;
    let n_head_kv = positive_usize(
        require_i64(h, &architecture, "attention.head_count_kv")?,
        "attention.head_count_kv",
    )?;
    let n_embd = positive_usize(
        require_i64(h, &architecture, "embedding_length")?,
        "embedding_length",
    )?;
    let n_ff = positive_usize(
        require_i64(h, &architecture, "feed_forward_length")?,
        "feed_forward_length",
    )?;
    let (head_dim, rot_dims) = resolve_head_geometry(h, &architecture, n_embd, n_head)?;
    let mixers = resolve_mixers(h, &architecture, n_layer)?;
    let affine = if mixers.contains(&MixerKind::Affine) {
        Some(resolve_affine(h, &architecture)?)
    } else {
        None
    };
    let attn_gate = resolve_attn_gate(h, &mixers, n_head, head_dim)?;
    let context_length = match h.meta_i64(&format!("{architecture}.context_length")) {
        Some(v) if v > 0 => usize::try_from(v).map_err(|_| EngineError::LengthOverflow)?,
        _ => 0,
    };
    let rope_theta = optional_f64(h, &architecture, "rope.freq_base", DEFAULT_ROPE_THETA);
    let rms_eps = optional_f64(
        h,
        &architecture,
        "attention.layer_norm_rms_epsilon",
        DEFAULT_RMS_EPS,
    );

    let td = h
        .tensor("token_embd.weight")
        .ok_or_else(|| EngineError::MissingTensor("token_embd.weight".to_string()))?;
    if td.dims.len() != 2 {
        return Err(EngineError::ConfigMismatch(format!(
            "token_embd.weight rank {} is not 2",
            td.dims.len()
        )));
    }
    let (row_len, vocab_dim) = (td.dims[0], td.dims[1]);
    if row_len != n_embd as u64 {
        return Err(EngineError::ConfigMismatch(format!(
            "token_embd.weight row {row_len} != embedding_length {n_embd}"
        )));
    }
    let vocab_size = usize::try_from(vocab_dim).map_err(|_| EngineError::LengthOverflow)?;
    if let Some(GVal::Arr(_, toks)) = h.meta("tokenizer.ggml.tokens") {
        if toks.len() != vocab_size {
            return Err(EngineError::ConfigMismatch(format!(
                "token array {} != token_embd rows {vocab_size}",
                toks.len()
            )));
        }
    }

    Ok(EngineConfig {
        architecture,
        n_layer,
        n_head,
        n_head_kv,
        n_embd,
        n_ff,
        vocab_size,
        context_length,
        rope_theta,
        rms_eps,
        head_dim,
        mixers,
        rot_dims,
        attn_gate,
        affine,
        n_nextn,
    })
}

/// `(head_dim, rot_dims)`: `*.attention.key_length` (checked equal to `value_length` when stated)
/// or `n_embd / n_head`, and `*.rope.dimension_count` or the whole head, even and at most `head_dim`.
fn resolve_head_geometry(
    h: &GgufHeader,
    arch: &str,
    n_embd: usize,
    n_head: usize,
) -> Result<(usize, usize), EngineError> {
    let head_dim = if let Some(k) = h.meta_i64(&format!("{arch}.attention.key_length")) {
        let k = positive_usize(k, "attention.key_length")?;
        if let Some(v) = h.meta_i64(&format!("{arch}.attention.value_length")) {
            if positive_usize(v, "attention.value_length")? != k {
                return Err(EngineError::ConfigMismatch(format!(
                    "attention.value_length {v} != key_length {k}"
                )));
            }
        }
        k
    } else {
        if !n_embd.is_multiple_of(n_head) {
            return Err(EngineError::ConfigMismatch(format!(
                "embedding_length {n_embd} not divisible by head_count {n_head}"
            )));
        }
        n_embd / n_head
    };
    // Only a stated partial width is checked here; the whole-head default keeps its refusals in
    // `RopeTable::new`, as before the schedule existed.
    let rot_dims = match h.meta_i64(&format!("{arch}.rope.dimension_count")) {
        Some(v) => {
            let r = positive_usize(v, "rope.dimension_count")?;
            if r > head_dim || !r.is_multiple_of(2) {
                return Err(EngineError::ConfigMismatch(format!(
                    "rope.dimension_count {r} is odd or exceeds head_dim {head_dim}"
                )));
            }
            r
        }
        None => head_dim,
    };
    Ok((head_dim, rot_dims))
}

/// The per-layer mixer schedule from `*.full_attention_interval` (llama.cpp `qwen35.cpp`
/// `load_arch_hparams`: layer `l` is recurrent iff `(l + 1) % interval != 0`).
///
/// # Complexity
///
/// `O(n_layer)` time and space: one entry per trunk layer.
fn resolve_mixers(
    h: &GgufHeader,
    arch: &str,
    n_layer: usize,
) -> Result<Vec<MixerKind>, EngineError> {
    let interval = match h.meta_i64(&format!("{arch}.full_attention_interval")) {
        Some(v) => Some(positive_usize(v, "full_attention_interval")?),
        None => None,
    };
    let mut out = Vec::with_capacity(n_layer);
    let mut l = 0usize;
    while l < n_layer {
        out.push(match interval {
            Some(f) if !(l + 1).is_multiple_of(f) => MixerKind::Affine,
            _ => MixerKind::Softmax,
        });
        l += 1;
    }
    Ok(out)
}

/// The affine mixer's geometry from `*.ssm.*`, refusing a missing key or an inner size that is not
/// a whole number of value heads.
fn resolve_affine(h: &GgufHeader, arch: &str) -> Result<AffineGeom, EngineError> {
    let get =
        |k: &str| -> Result<usize, EngineError> { positive_usize(require_i64(h, arch, k)?, k) };
    let conv_kernel = get("ssm.conv_kernel")?;
    let key_heads = get("ssm.group_count")?;
    let value_heads = get("ssm.time_step_rank")?;
    let head_k = get("ssm.state_size")?;
    let inner = get("ssm.inner_size")?;
    if !inner.is_multiple_of(value_heads) || !value_heads.is_multiple_of(key_heads) {
        return Err(EngineError::ConfigMismatch(format!(
            "ssm.inner_size {inner} / time_step_rank {value_heads} / group_count {key_heads}"
        )));
    }
    Ok(AffineGeom {
        conv_kernel,
        n_k_heads: key_heads,
        n_v_heads: value_heads,
        head_k,
        head_v: inner / value_heads,
    })
}

/// Whether the softmax layers' `attn_q` carries the per-head output gate, read from its row count:
/// `n_head·head_dim` (no gate) or `2·n_head·head_dim` (query and gate interleaved per head).
fn resolve_attn_gate(
    h: &GgufHeader,
    mixers: &[MixerKind],
    n_head: usize,
    head_dim: usize,
) -> Result<bool, EngineError> {
    let Some(l) = mixers.iter().position(|m| *m == MixerKind::Softmax) else {
        return Ok(false);
    };
    let name = format!("blk.{l}.attn_q.weight");
    let Some(t) = h.tensor(&name) else {
        return Ok(false);
    };
    let rows = t.dims.get(1).copied().unwrap_or(0);
    let q_out = (n_head as u64).saturating_mul(head_dim as u64);
    if rows == q_out {
        Ok(false)
    } else if rows == q_out.saturating_mul(2) {
        Ok(true)
    } else {
        Err(EngineError::ConfigMismatch(format!(
            "{name} rows {rows} are neither {q_out} nor {}",
            q_out.saturating_mul(2)
        )))
    }
}

/// Walk the directory into a [`TensorTable`], checking every extent against the file.
///
/// A tensor whose type has no geometry in `crate::format::quant` still gets a range: its extent
/// runs from its data offset to the next entry's data offset in directory-offset order (the file
/// end for the last). That keeps every entry bounds-checked without inventing geometry; the exact
/// byte count is recorded only when the quant layer supplies it.
fn build_table(header: &GgufHeader, file_len: u64) -> Result<TensorTable, EngineError> {
    let count = header.tensors.len();
    let mut starts: Vec<u64> = Vec::with_capacity(count);
    let mut idx = 0usize;
    while idx < count {
        let desc = &header.tensors[idx];
        let at = header
            .data0
            .checked_add(desc.offset)
            .ok_or(EngineError::LengthOverflow)?;
        if at > file_len {
            return Err(EngineError::TensorOutOfBounds {
                name: String::from_utf8_lossy(&desc.name).into_owned(),
                at,
                n: 0,
                file_len,
            });
        }
        starts.push(at);
        idx += 1;
    }
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by_key(|&ord| starts[ord]);
    let mut ends: Vec<u64> = vec![file_len; count];
    let mut rank = 0usize;
    while rank + 1 < count {
        ends[order[rank]] = starts[order[rank + 1]];
        rank += 1;
    }

    let mut entries: Vec<TensorEntry> = Vec::with_capacity(count);
    let mut entry_idx = 0usize;
    while entry_idx < count {
        let desc = &header.tensors[entry_idx];
        let name = String::from_utf8_lossy(&desc.name).into_owned();
        let n_elements = desc.n_elements()?;
        let n_bytes = desc.n_bytes()?;
        let (start, end) = (starts[entry_idx], ends[entry_idx]);
        if end < start {
            return Err(EngineError::ConfigMismatch(format!(
                "directory offsets out of order at {name}"
            )));
        }
        if let Some(bytes) = n_bytes {
            let exact = start
                .checked_add(bytes)
                .ok_or(EngineError::LengthOverflow)?;
            if exact > file_len {
                return Err(EngineError::TensorOutOfBounds {
                    name,
                    at: start,
                    n: bytes,
                    file_len,
                });
            }
            if exact > end {
                return Err(EngineError::ConfigMismatch(format!(
                    "tensor {name} storage {bytes} overlaps the next directory entry"
                )));
            }
        }
        entries.push(TensorEntry {
            name,
            ggml_type: desc.ggml_type,
            quant: QuantKind::from_ggml(desc.ggml_type),
            dims: desc.dims.clone(),
            n_elements,
            n_bytes,
            byte_range: start..end,
        });
        entry_idx += 1;
    }
    Ok(TensorTable { entries })
}

/// Keep freed per-call buffers (columns, packs, conv caches; each under 64 MiB) in the glibc heap
/// instead of returning them to the kernel. Under the default dynamic thresholds a freed buffer is
/// trimmed or unmapped and the next layer's call faults its pages in again (~77K minor
/// faults per pp200 run, ~3.7 faults per page; with the thresholds held, ~9K). Process-wide and
/// set once; a refused `mallopt` leaves the default and changes no result.
fn hold_heap() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            // SAFETY: `mallopt` only sets allocator parameters; it is called before any engine
            // buffer exists and takes plain integers.
            unsafe {
                libc::mallopt(libc::M_MMAP_THRESHOLD, 64 << 20);
                libc::mallopt(libc::M_TRIM_THRESHOLD, 1 << 30);
            }
        });
    }
}

impl Engine {
    /// Open a GGUF file, resolve the config and build the checked tensor table.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] on an I/O failure, a GGUF parse refusal, a missing or malformed
    /// config key, a refusal of the `prism.hadamard.*` fold contract ([`hadamard_meta`]), a
    /// vocabulary cross-check failure, a missing `token_embd.weight`, or a tensor extent that
    /// leaves the file.
    pub fn open(path: &str) -> Result<Engine, EngineError> {
        hold_heap();
        let g = Gguf::open(path)?;
        let config = resolve_config(&g.header)?;
        let hadamard = hadamard_meta(&g.header)?;
        let table = build_table(&g.header, g.file_len)?;
        // Read the file once; every payload slice borrows from this. A file that changed size
        // between the header parse and this read fails closed rather than serving stale extents.
        let data = std::fs::read(path).map_err(|e| EngineError::Io(e.to_string()))?;
        if data.len() as u64 != g.file_len {
            return Err(EngineError::ConfigMismatch(format!(
                "file length changed while opening: {} != {}",
                data.len(),
                g.file_len
            )));
        }
        let token_embd = table
            .entries()
            .iter()
            .position(|t| t.name == "token_embd.weight")
            .ok_or_else(|| EngineError::MissingTensor("token_embd.weight".to_string()))?;
        let mut engine = Engine {
            path: path.to_string(),
            file_len: g.file_len,
            config,
            hadamard,
            data: data.into_boxed_slice(),
            table,
            token_embd,
            vectors: DecodeVectors::default(),
            pq2_luts: Mutex::new(HashMap::new()),
            bitplane: std::sync::OnceLock::new(),
        };
        let vectors = engine.load_decode_vectors()?;
        engine.vectors = vectors;
        Ok(engine)
    }

    /// The decode `F32` gain/bias vectors resolved once at load.
    #[must_use]
    pub fn decode_vectors(&self) -> &DecodeVectors {
        &self.vectors
    }

    /// The validated `prism.hadamard.*` fold contract, or `None` for an unrotated model.
    #[must_use]
    pub fn hadamard(&self) -> Option<&HadamardMeta> {
        self.hadamard.as_ref()
    }

    /// Read the `F32` vector `name` of length `len` into Q32.32, or `None` when absent.
    ///
    /// A present tensor must be rank-1 with `dims == [len]` and ggml type `F32`; otherwise it
    /// refuses with [`EngineError::BadVector`]. The payload is converted by
    /// [`super::f32_convert::dequant_f32_row`], which refuses a non-finite or out-of-`i64` value.
    fn read_f32_vector(&self, name: &str, len: usize) -> Result<Option<Vec<i64>>, EngineError> {
        let Some(t) = self.table.get(name) else {
            return Ok(None);
        };
        let expected = u64::try_from(len).map_err(|_| EngineError::LengthOverflow)?;
        if t.dims.len() != 1 || t.dims[0] != expected {
            return Err(EngineError::BadVector {
                name: name.to_string(),
                reason: "dims",
            });
        }
        if t.ggml_type != quant::GGML_TYPE_F32 {
            return Err(EngineError::BadVector {
                name: name.to_string(),
                reason: "type",
            });
        }
        let n_bytes = t.n_bytes.ok_or(EngineError::BadVector {
            name: name.to_string(),
            reason: "geometry",
        })?;
        let raw = self.read_range(t.byte_range.start, n_bytes)?;
        let mut out: Vec<i64> = Vec::with_capacity(len);
        dequant_f32_row(&raw, len, &mut out)?;
        Ok(Some(out))
    }

    /// Each multi-token-prediction block's input norms and head norm.
    fn load_nextn_vectors(&self) -> Result<Vec<[Option<Vec<i64>>; 3]>, EngineError> {
        let cfg = &self.config;
        let mut out = Vec::with_capacity(cfg.n_nextn);
        let mut i = 0usize;
        while i < cfg.n_nextn {
            let l = cfg.n_layer + i;
            out.push([
                self.read_f32_vector(&format!("blk.{l}.nextn.enorm.weight"), cfg.n_embd)?,
                self.read_f32_vector(&format!("blk.{l}.nextn.hnorm.weight"), cfg.n_embd)?,
                self.read_f32_vector(
                    &format!("blk.{l}.nextn.shared_head_norm.weight"),
                    cfg.n_embd,
                )?,
            ]);
            i += 1;
        }
        Ok(out)
    }

    /// Every affine layer's `F32` parameters (`ssm_conv1d`, `ssm_dt.bias`, `ssm_a`, `ssm_norm`),
    /// refusing an absent one; `None` for softmax layers.
    ///
    /// # Complexity
    ///
    /// `O(n_affine · conv_kernel · channels)` reads and conversions, once at load.
    fn load_affine_vectors(&self) -> Result<Vec<Option<AffineVectors>>, EngineError> {
        let cfg = &self.config;
        let mut out = Vec::with_capacity(cfg.n_layer);
        let mut l = 0usize;
        while l < cfg.n_layer {
            if cfg.mixers[l] != MixerKind::Affine {
                out.push(None);
                l += 1;
                continue;
            }
            let g = cfg
                .affine
                .ok_or_else(|| EngineError::ConfigMismatch("affine geometry".to_string()))?;
            let channels = 2 * g.n_k_heads * g.head_k + g.n_v_heads * g.head_v;
            let need =
                |name: String, v: Option<Vec<i64>>| v.ok_or(EngineError::MissingTensor(name));
            let conv_name = format!("blk.{l}.ssm_conv1d.weight");
            let conv = self.read_f32_matrix(&conv_name, g.conv_kernel, channels)?;
            let dt_name = format!("blk.{l}.ssm_dt.bias");
            let a_name = format!("blk.{l}.ssm_a");
            let norm_name = format!("blk.{l}.ssm_norm.weight");
            out.push(Some(AffineVectors {
                conv: need(conv_name, conv)?,
                dt: need(
                    dt_name.clone(),
                    self.read_f32_vector(&dt_name, g.n_v_heads)?,
                )?,
                a: need(a_name.clone(), self.read_f32_vector(&a_name, g.n_v_heads)?)?,
                norm: need(
                    norm_name.clone(),
                    self.read_f32_vector(&norm_name, g.head_v)?,
                )?,
            }));
            l += 1;
        }
        Ok(out)
    }

    /// A two-dimensional `F32` tensor `[d0, d1]` (ggml order: `d0` contiguous) as one Q32.32 vector
    /// of `d0·d1` values, or `None` when absent; the same type and range refusals as
    /// `read_f32_vector`.
    fn read_f32_matrix(
        &self,
        name: &str,
        d0: usize,
        d1: usize,
    ) -> Result<Option<Vec<i64>>, EngineError> {
        let Some(t) = self.table.get(name) else {
            return Ok(None);
        };
        if t.dims.len() != 2 || t.dims[0] != d0 as u64 || t.dims[1] != d1 as u64 {
            return Err(EngineError::BadVector {
                name: name.to_string(),
                reason: "dims",
            });
        }
        if t.ggml_type != quant::GGML_TYPE_F32 {
            return Err(EngineError::BadVector {
                name: name.to_string(),
                reason: "type",
            });
        }
        let len = d0.checked_mul(d1).ok_or(EngineError::LengthOverflow)?;
        let n_bytes = t.n_bytes.ok_or(EngineError::BadVector {
            name: name.to_string(),
            reason: "geometry",
        })?;
        let raw = self.read_range(t.byte_range.start, n_bytes)?;
        let mut out: Vec<i64> = Vec::with_capacity(len);
        dequant_f32_row(&raw, len, &mut out)?;
        Ok(Some(out))
    }

    /// Resolve every decode gain and attention bias the step reads, once.
    ///
    /// The names and widths are exactly [`Engine::step_with_kv`]'s reads: `blk.{l}.attn_norm`,
    /// `blk.{l}.ffn_norm` (`n_embd`), `blk.{l}.attn_{q,k,v}.bias` (`n_head·head_dim`,
    /// `n_head_kv·head_dim`), and `output_norm.weight` (`n_embd`).
    fn load_decode_vectors(&self) -> Result<DecodeVectors, EngineError> {
        let cfg = &self.config;
        let q_out = cfg
            .n_head
            .checked_mul(cfg.head_dim)
            .ok_or(EngineError::LengthOverflow)?;
        let kv_out = cfg
            .n_head_kv
            .checked_mul(cfg.head_dim)
            .ok_or(EngineError::LengthOverflow)?;
        let mut norms: Vec<[Option<Vec<i64>>; 2]> = Vec::with_capacity(cfg.n_layer);
        let mut biases: Vec<[Option<Vec<i64>>; 3]> = Vec::with_capacity(cfg.n_layer);
        let mut qk_norms: Vec<[Option<Vec<i64>>; 2]> = Vec::with_capacity(cfg.n_layer);
        let mut layer = 0usize;
        while layer < cfg.n_layer + cfg.n_nextn {
            norms.push([
                self.read_f32_vector(&format!("blk.{layer}.attn_norm.weight"), cfg.n_embd)?,
                // The pre-FFN norm: `ffn_norm` (Qwen2), `post_attention_norm` (Qwen3.5).
                match self.read_f32_vector(&format!("blk.{layer}.ffn_norm.weight"), cfg.n_embd)? {
                    Some(v) => Some(v),
                    None => self.read_f32_vector(
                        &format!("blk.{layer}.post_attention_norm.weight"),
                        cfg.n_embd,
                    )?,
                },
            ]);
            biases.push([
                self.read_f32_vector(&format!("blk.{layer}.attn_q.bias"), q_out)?,
                self.read_f32_vector(&format!("blk.{layer}.attn_k.bias"), kv_out)?,
                self.read_f32_vector(&format!("blk.{layer}.attn_v.bias"), kv_out)?,
            ]);
            // Qwen3 only: the per-head QK `RMSNorm` gains, absent (`None`) on Qwen2.
            qk_norms.push([
                self.read_f32_vector(&format!("blk.{layer}.attn_q_norm.weight"), cfg.head_dim)?,
                self.read_f32_vector(&format!("blk.{layer}.attn_k_norm.weight"), cfg.head_dim)?,
            ]);
            layer += 1;
        }
        let output_norm = self.read_f32_vector("output_norm.weight", cfg.n_embd)?;
        let affine = self.load_affine_vectors()?;
        let nextn = self.load_nextn_vectors()?;
        Ok(DecodeVectors {
            norms,
            biases,
            qk_norms,
            output_norm,
            affine,
            nextn,
        })
    }

    /// The head's bit planes, built once on first use (`O(vocab·n_embd)` byte operations and as
    /// many bytes of planes); `None` when the head is not `Q8_0` or the model is Hadamard-folded.
    pub(crate) fn bitplane_head(&self) -> Option<Arc<super::kern::bitplane::BitplaneHead>> {
        self.bitplane
            .get_or_init(|| {
                if self.hadamard.is_some() {
                    return None;
                }
                let t = self.lm_head();
                let raw = self.payload(t.byte_range.start, t.n_bytes?).ok()?;
                super::kern::bitplane::BitplaneHead::from_q8_0(
                    t.ggml_type,
                    raw,
                    self.config.vocab_size,
                    self.config.n_embd,
                )
            })
            .clone()
    }

    /// The cached `PQ2_0` lookup table of tensor `name`, built once from `raw`.
    ///
    /// The first projection of a tensor repacks its payload with [`Pq2Lut::from_pq2_0`] and caches
    /// the result under the tensor name; every later token reuses the same table, so the
    /// `O(n_out·n_in)` repack is paid once per tensor per load. The map is a `Mutex` because the
    /// decode step shares `&Engine` across its projection tasks.
    ///
    /// # Errors
    ///
    /// The [`MatvecError`](super::MatvecError) `Pq2Lut::from_pq2_0` returns for a non-finite block
    /// scale, a short payload or a row width that is not a whole number of 128-weight blocks.
    pub(crate) fn pq2_lut(
        &self,
        name: &str,
        raw: &[u8],
        n_out: usize,
        n_in: usize,
    ) -> Result<Arc<Pq2Lut>, super::MatvecError> {
        {
            let cache = self
                .pq2_luts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(lut) = cache.get(name) {
                return Ok(Arc::clone(lut));
            }
        }
        let lut = Arc::new(Pq2Lut::from_pq2_0(raw, n_out, n_in)?);
        let mut cache = self
            .pq2_luts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(Arc::clone(cache.entry(name.to_string()).or_insert(lut)))
    }

    /// The named tensor directory.
    #[must_use]
    pub fn table(&self) -> &TensorTable {
        &self.table
    }

    /// The `token_embd.weight` directory entry.
    #[must_use]
    pub fn token_embd(&self) -> &TensorEntry {
        &self.table.entries()[self.token_embd]
    }

    /// Read `n` bytes of the file at absolute offset `at`, refusing past EOF.
    ///
    /// Opens the file and returns owned bytes; retained for callers that need a buffer they can
    /// hold independently of the [`Engine`]. The step path uses [`Engine::payload`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::RowOutOfBounds`] when the range leaves the file and
    /// [`EngineError::Io`] when the read itself fails.
    pub fn read_range(&self, at: u64, n: u64) -> Result<Vec<u8>, EngineError> {
        let end = at.checked_add(n).ok_or(EngineError::LengthOverflow)?;
        if end > self.file_len {
            return Err(EngineError::RowOutOfBounds {
                at,
                n,
                end: self.file_len,
            });
        }
        let len = usize::try_from(n).map_err(|_| EngineError::LengthOverflow)?;
        let mut f = std::fs::File::open(&self.path).map_err(|e| EngineError::Io(e.to_string()))?;
        f.seek(SeekFrom::Start(at))
            .map_err(|e| EngineError::Io(e.to_string()))?;
        let mut buf = vec![0u8; len];
        f.read_exact(&mut buf)
            .map_err(|e| EngineError::Io(e.to_string()))?;
        Ok(buf)
    }

    /// Borrow `n` bytes of the resident file at absolute offset `at`, refusing past EOF.
    ///
    /// This is the step path's read: the bytes were read once at [`Engine::open`], so the borrow is
    /// O(1) and copies nothing. The bounds checks and refusals are
    /// [`Engine::read_range`]'s: [`EngineError::RowOutOfBounds`] when `at + n` leaves the file and
    /// [`EngineError::LengthOverflow`] when the range does not fit `usize`. The resident bytes are
    /// exactly the file's bytes, so a payload slice equals the range `read_range` returns.
    ///
    /// # Complexity
    ///
    /// `O(1)` time and no allocation; the returned slice borrows `self`.
    ///
    /// # Errors
    ///
    /// [`EngineError::RowOutOfBounds`], [`EngineError::LengthOverflow`].
    pub fn payload(&self, at: u64, n: u64) -> Result<&[u8], EngineError> {
        let end = at.checked_add(n).ok_or(EngineError::LengthOverflow)?;
        if end > self.file_len {
            return Err(EngineError::RowOutOfBounds {
                at,
                n,
                end: self.file_len,
            });
        }
        let start = usize::try_from(at).map_err(|_| EngineError::LengthOverflow)?;
        let stop = usize::try_from(end).map_err(|_| EngineError::LengthOverflow)?;
        self.data
            .get(start..stop)
            .ok_or(EngineError::LengthOverflow)
    }
}

#[cfg(test)]
mod tests;
