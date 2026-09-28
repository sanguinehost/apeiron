//! The decode step: the rows 0-8 composition and the `Engine` entry points.

use std::borrow::Cow;
use std::sync::Arc;

use crate::format::quant;

use super::chart::ffn::{projections_q8act, FfnScratch, ProjSource, ProjSpec};
use super::dequant::dequant_row_fixed;
#[cfg(test)]
use super::kern::lut::q8act_projection;
use super::kern::lut::{Pq2Lut, LUT_ROW_ALIGN};
use super::loader::HadamardMeta;
use super::mixer::AffineState;
use super::value::round_shift_32;
use super::{
    argmax_fixed, fixed_vector_checksum, fwht_fixed, lm_head, rms_norm_fixed, AttnError, Engine,
    EngineError, F32Error, FfnError, HadamardError, KvState, LmHeadError, MatvecError, MixerKind,
    NormError, ProjWeights, RopeError, RopeTable, SamplerError, TensorEntry,
};

// Unchunked kernel: test-only `Sched::Sequential`; executing `Sched::Resident` is `par::matvec_rows_chunked`.
#[cfg(test)]
use super::matvec_fixed;

/// One greedy decode step's result (rows 0–8 composed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepResult {
    /// The chosen token: [`argmax_fixed`] over the step's Q32.32 logits.
    pub token: u32,
    /// [`fixed_vector_checksum`] of the step's Q32.32 logits.
    pub logits_checksum: u64,
}

/// A refused decode step (rows 0–8 composed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepError {
    /// The token id left the vocabulary, or the embedding read refused.
    Embedding(EngineError),
    /// A tensor payload read refused.
    Read(EngineError),
    /// `RoPE` refused.
    Rope(RopeError),
    /// A norm refused.
    Norm(NormError),
    /// The KV cache refused (capacity, position, shape, softmax window).
    Kv(AttnError),
    /// A projection refused.
    Matvec(MatvecError),
    /// The FFN refused.
    Ffn(FfnError),
    /// The LM head refused.
    LmHead(LmHeadError),
    /// The sampler refused.
    Sampler(SamplerError),
    /// The folded-weight activation rotation refused ([`fwht_fixed`]).
    Hadamard(HadamardError),
    /// An `explicit` fold contract carries no sign vector for the activation width.
    MissingHadamardSigns {
        /// Tensor being projected.
        tensor: String,
        /// Activation width `x.len()` that needs a sign vector.
        width: usize,
    },
    /// `prism.hadamard.version = 2` (tied output) is not executed: the tied embedding path needs the
    /// inverse of the signed transform, which the forward [`fwht_fixed`] does not provide.
    TiedOutputUnsupported,
    /// An `F32` gain or bias refused.
    Gain(F32Error),
    /// A required tensor is absent from the directory.
    MissingTensor(String),
    /// A tensor's quant family has no step reader.
    UnsupportedQuant {
        /// ggml type id.
        ggml_type: u32,
    },
    /// A tensor's stored dims are not the dims this stage requires.
    BadShape {
        /// Tensor name.
        tensor: String,
        /// Stored dims, fastest-varying first.
        dims: Vec<u64>,
        /// Required dims, fastest-varying first.
        expected: Vec<u64>,
    },
    /// The supplied KV cache geometry is not the engine config's geometry.
    KvGeometry {
        /// Supplied layer count.
        kv_layer: usize,
        /// Supplied query-head count.
        kv_head: usize,
        /// Supplied KV-head count.
        kv_head_kv: usize,
        /// Supplied head width.
        kv_head_dim: usize,
    },
    /// A config field cannot form a step quantity.
    BadConfig(&'static str),
    /// A length did not fit the target integer width.
    LengthOverflow,
    /// A checked fixed-point addition or narrowing left its range.
    Overflow,
}

impl core::fmt::Display for StepError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StepError::Embedding(e) => write!(f, "embedding: {e}"),
            StepError::Read(e) => write!(f, "read: {e}"),
            StepError::Rope(e) => write!(f, "rope: {e:?}"),
            StepError::Norm(e) => write!(f, "norm: {e:?}"),
            StepError::Kv(e) => write!(f, "kv: {e:?}"),
            StepError::Matvec(e) => write!(f, "matvec: {e:?}"),
            StepError::Ffn(e) => write!(f, "ffn: {e:?}"),
            StepError::LmHead(e) => write!(f, "lm head: {e:?}"),
            StepError::Sampler(e) => write!(f, "sampler: {e:?}"),
            StepError::Hadamard(e) => write!(f, "hadamard: {e:?}"),
            StepError::MissingHadamardSigns { tensor, width } => {
                write!(f, "hadamard: {tensor} needs a sign vector of width {width}")
            }
            StepError::TiedOutputUnsupported => {
                write!(f, "hadamard: version 2 tied output is not executed")
            }
            StepError::Gain(e) => write!(f, "gain: {e}"),
            StepError::MissingTensor(n) => write!(f, "no tensor {n}"),
            StepError::UnsupportedQuant { ggml_type } => {
                write!(f, "no step reader for ggml type {ggml_type}")
            }
            StepError::BadShape {
                tensor,
                dims,
                expected,
            } => write!(f, "tensor {tensor} dims {dims:?} != {expected:?}"),
            StepError::KvGeometry {
                kv_layer,
                kv_head,
                kv_head_kv,
                kv_head_dim,
            } => write!(
                f,
                "kv geometry {kv_layer}x{kv_head}x{kv_head_kv}x{kv_head_dim} != config"
            ),
            StepError::BadConfig(k) => write!(f, "bad config field {k}"),
            StepError::LengthOverflow => write!(f, "length overflow"),
            StepError::Overflow => write!(f, "fixed-point overflow"),
        }
    }
}

impl std::error::Error for StepError {}

impl From<RopeError> for StepError {
    fn from(e: RopeError) -> Self {
        StepError::Rope(e)
    }
}

impl From<NormError> for StepError {
    fn from(e: NormError) -> Self {
        StepError::Norm(e)
    }
}

impl From<AttnError> for StepError {
    fn from(e: AttnError) -> Self {
        StepError::Kv(e)
    }
}

impl From<MatvecError> for StepError {
    fn from(e: MatvecError) -> Self {
        StepError::Matvec(e)
    }
}

impl From<FfnError> for StepError {
    fn from(e: FfnError) -> Self {
        StepError::Ffn(e)
    }
}

impl From<LmHeadError> for StepError {
    fn from(e: LmHeadError) -> Self {
        StepError::LmHead(e)
    }
}

impl From<SamplerError> for StepError {
    fn from(e: SamplerError) -> Self {
        StepError::Sampler(e)
    }
}

impl From<F32Error> for StepError {
    fn from(e: F32Error) -> Self {
        StepError::Gain(e)
    }
}

/// Which projection kernel a decode step uses: [`KernelMode::Exact`] the pre-chart path,
/// [`KernelMode::Q8Act`] the int8-activation chart (`matvec ∘ quant_act`) or LUT kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KernelMode {
    /// The pre-change path over the dequantized activation.
    #[default]
    Exact,
    /// The int8-activation chart kernel, or the `PQ2_0` lookup-table kernel.
    Q8Act,
}

/// How a step sources each tensor payload: [`Sched::Resident`] the executing borrowed path,
/// [`Sched::Sequential`] the `cfg(test)` oracle (owned `read_range`, one [`matvec_fixed`]).
#[derive(Clone, Copy)]
pub(crate) enum Sched {
    Resident,
    #[cfg(test)]
    Sequential,
}

/// Checked `x[i] += y[i]` over two equal-width Q32.32 rows (the residual adds and bias add).
fn add_assign_fixed(x: &mut [i64], y: &[i64]) -> Result<(), StepError> {
    if x.len() != y.len() {
        return Err(StepError::BadConfig("residual row width"));
    }
    let mut i = 0usize;
    while i < x.len() {
        x[i] = x[i].checked_add(y[i]).ok_or(StepError::Overflow)?;
        i += 1;
    }
    Ok(())
}

/// [`rms_norm_fixed`] followed by the per-channel Q32.32 gain, the fold `cpu_rms` order. A width
/// mismatch refuses with [`StepError::BadConfig`].
fn rms_with_gain_slice(gain: &[i64], x: &[i64], eps: i64) -> Result<Vec<i64>, StepError> {
    if gain.len() != x.len() {
        return Err(StepError::BadConfig("norm gain width"));
    }
    let base = rms_norm_fixed(x, eps).map_err(StepError::Norm)?;
    let mut out: Vec<i64> = Vec::with_capacity(base.len());
    let mut i = 0usize;
    while i < base.len() {
        let prod = i128::from(base[i])
            .checked_mul(i128::from(gain[i]))
            .ok_or(StepError::Overflow)?;
        out.push(i64::try_from(round_shift_32(prod)).map_err(|_| StepError::Overflow)?);
        i += 1;
    }
    Ok(out)
}

/// The folded-weight activation rotation: `None` when `name` is not in the fold contract, otherwise
/// a fresh copy of `x` rotated by [`fwht_fixed`] (`x' = round(c·H_{2^k}·(s⊙x)/2^32)`); `x` is never
/// modified. In `explicit` mode the sign vector is the contract entry whose width equals `x.len()`
/// (a missing width refuses with [`StepError::MissingHadamardSigns`]); in `identity` mode none is
/// applied. A kernel refusal maps to [`StepError::Hadamard`].
///
/// # Complexity
///
/// `O(n log block)` time and `O(n)` space; the contract scan is `O(weight_names + signs)` bytes.
fn folded_activation(
    meta: Option<&HadamardMeta>,
    name: &str,
    x: &[i64],
) -> Result<Option<Vec<i64>>, StepError> {
    let Some(meta) = meta else {
        return Ok(None);
    };
    // Only the weights the contract lists are folded; every other tensor keeps `x` verbatim.
    let mut listed = false;
    let mut i = 0usize;
    while i < meta.weight_names.len() {
        if meta.weight_names[i].as_bytes() == name.as_bytes() {
            listed = true;
        }
        i += 1;
    }
    if !listed {
        return Ok(None);
    }
    // The sign vector, when the mode is explicit, is keyed by the activation width.
    let mut signs: Option<&[i8]> = None;
    if let Some(table) = meta.signs.as_ref() {
        let mut found = false;
        let mut j = 0usize;
        while j < table.len() {
            if table[j].0 as usize == x.len() {
                signs = Some(table[j].1.as_slice());
                found = true;
            }
            j += 1;
        }
        if !found {
            return Err(StepError::MissingHadamardSigns {
                tensor: name.to_string(),
                width: x.len(),
            });
        }
    }
    let mut folded: Vec<i64> = Vec::with_capacity(x.len());
    let mut k = 0usize;
    while k < x.len() {
        folded.push(x[k]);
        k += 1;
    }
    fwht_fixed(&mut folded, signs, meta.block_size as usize).map_err(StepError::Hadamard)?;
    Ok(Some(folded))
}

/// [`Engine::step_proj_sched`]'s and [`Engine::lm_head_fixed_sched`]'s resident `Q8Act` leaf.
impl Engine {
    /// One `Q8Act` projection over `x`: one pooled
    /// [`quant_act_pooled`](super::chart::ffn::quant_act_pooled) of `x`, then the cached `PQ2_0`
    /// lookup-table row kernel or the chart row kernel split by `chunks`. Same result and refusals
    /// as `kern::lut::q8act_projection`; taking the quantization here lets a caller that already
    /// holds `q` skip a second pass.
    #[allow(
        clippy::too_many_arguments,
        reason = "the engine, tensor identity, payload and shape, activation and dispatch width \
                  are the projection's full input"
    )]
    fn proj_q8act_quantified(
        &self,
        name: &str,
        ggml_type: u32,
        raw: &[u8],
        n_out: usize,
        n_in: usize,
        x: &[i64],
        chunks: usize,
    ) -> Result<Vec<i64>, MatvecError> {
        let q = super::chart::ffn::quant_act_pooled(x)
            .ok_or(MatvecError::Dequant(EngineError::FixedOverflow))?;
        if ggml_type == quant::GGML_TYPE_PQ2_0 {
            let lut = self.pq2_lut(name, raw, n_out, n_in)?;
            let spec = ProjSpec {
                n_out,
                row_align: LUT_ROW_ALIGN,
                source: ProjSource::Rows(&*lut),
            };
            let mut out = super::chart::ffn::projections_q8act_prepared(n_in, &q, &[spec], chunks)?;
            Ok(out.pop().expect("one projection"))
        } else {
            super::chart::ffn::matvec_q8act_quant_chunked(ggml_type, raw, n_out, n_in, &q, chunks)
        }
    }
}

/// Prefill block width: the columns one pass carries, so the `Q8_0` weights are read once per
/// block and the power-of-two GEMM (`kern::gemm`) reuses each weight row across them.
/// `APEIRON_PREFILL_COLS` overrides it (read once).
pub(crate) fn prefill_block() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("APEIRON_PREFILL_COLS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(64)
    })
}

/// The bit-plane head on the greedy path unless `APEIRON_BITPLANE=0`; read once per process.
fn bitplane_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("APEIRON_BITPLANE").map_or(true, |v| v != "0"))
}

/// Each layer's index in the softmax list carrier (`KvState`), `None` for an affine layer. With an
/// all-softmax schedule this is the identity.
///
/// # Complexity
///
/// `O(n_layer)` time and space.
fn softmax_slots(mixers: &[MixerKind]) -> Vec<Option<usize>> {
    let mut out = Vec::with_capacity(mixers.len());
    let mut next = 0usize;
    for m in mixers {
        if *m == MixerKind::Softmax {
            out.push(Some(next));
            next += 1;
        } else {
            out.push(None);
        }
    }
    out
}

/// Each layer's index in the [`AffineState`], `None` for a softmax layer.
///
/// # Complexity
///
/// `O(n_layer)` time and space.
fn affine_slots(mixers: &[MixerKind]) -> Vec<Option<usize>> {
    let mut out = Vec::with_capacity(mixers.len());
    let mut next = 0usize;
    for m in mixers {
        if *m == MixerKind::Affine {
            out.push(Some(next));
            next += 1;
        } else {
            out.push(None);
        }
    }
    out
}

/// `RoPE` on the first `rope.head_dim()` components of every `hd`-wide head of `x` (partial
/// rotary: Qwen3.5 rotates 64 of 256). When the table width equals `hd` this is one
/// [`RopeTable::rotate_one`] per head, the same arithmetic as rotating the whole vector.
///
/// # Complexity
///
/// `O(x.len() / hd · rope.head_dim())` multiplications, no allocation.
fn rotate_head_prefixes(
    rope: &RopeTable,
    x: &mut [i64],
    hd: usize,
    pos: u32,
) -> Result<(), StepError> {
    let rot = rope.head_dim();
    if hd == 0 || rot > hd || !x.len().is_multiple_of(hd) {
        return Err(StepError::Rope(RopeError::VectorLength));
    }
    let mut base = 0usize;
    while base < x.len() {
        rope.rotate_one(&mut x[base..base + rot], pos)
            .map_err(StepError::Rope)?;
        base += hd;
    }
    Ok(())
}

/// Split a gated `attn_q` output, laid out per head as `[query(hd) | gate(hd)]` (llama.cpp
/// `qwen35.cpp` `build_layer_attn` views), into the query and the gate, each `n_head·hd` wide.
///
/// # Complexity
///
/// `O(n_head·hd)` copies into two new vectors.
fn split_query_gate(
    full: &[i64],
    n_head: usize,
    hd: usize,
) -> Result<(Vec<i64>, Vec<i64>), StepError> {
    if full.len() != n_head * hd * 2 {
        return Err(StepError::BadConfig("gated query width"));
    }
    let mut q = Vec::with_capacity(n_head * hd);
    let mut g = Vec::with_capacity(n_head * hd);
    let mut h = 0usize;
    while h < n_head {
        let base = h * 2 * hd;
        q.extend_from_slice(&full[base..base + hd]);
        g.extend_from_slice(&full[base + hd..base + 2 * hd]);
        h += 1;
    }
    Ok((q, g))
}

/// `att[i] ← att[i] · sigmoid(gate[i])`, the attention output gate, through the one
/// [`mul_q32`](super::value::mul_q32).
///
/// # Complexity
///
/// `O(att.len())` sigmoid evaluations and products.
fn apply_sigmoid_gate(att: &mut [i64], gate: &[i64]) -> Result<(), StepError> {
    if att.len() != gate.len() {
        return Err(StepError::BadConfig("gate width"));
    }
    let mut i = 0usize;
    while i < att.len() {
        let s =
            crate::numeric::fixed_ln::fixed_sigmoid(gate[i]).map_err(|_| StepError::Overflow)?;
        att[i] = super::value::mul_q32(att[i], s).ok_or(StepError::Overflow)?;
        i += 1;
    }
    Ok(())
}

/// `vectors.apply_qk_norms` plus the two per-vector `RoPE` rotations, over the pool when
/// `parallel`: every `hd`-wide head of `query` (then of `key`) gets its QK `RMSNorm` and its
/// rotation in one pool task. Each head's norm and rotation are head-local, so the split is
/// bit-identical to the serial norm-then-rotate pair; a refusal keeps the first head index.
#[allow(
    clippy::too_many_arguments,
    reason = "the two QK gains, the rotation table, both vectors, the head geometry, the norm \
              epsilon, the position and the schedule are the call's full input"
)]
fn qk_norm_and_rope(
    vectors: &super::loader::DecodeVectors,
    layer: usize,
    rope: &RopeTable,
    query: &mut [i64],
    key: &mut [i64],
    hd: usize,
    eps: i64,
    pos: u32,
    parallel: bool,
) -> Result<(), StepError> {
    if !parallel || hd == 0 || !query.len().is_multiple_of(hd) || !key.len().is_multiple_of(hd) {
        // The composed geometry is a whole number of `hd`-wide heads; anything else keeps the
        // serial refusals unchanged.
        vectors.apply_qk_norms(layer, query, key, hd, eps)?;
        rotate_head_prefixes(rope, query, hd, pos)?;
        return rotate_head_prefixes(rope, key, hd, pos);
    }
    let qk = vectors.qk_norms_of(layer);
    let qk_q = qk.and_then(|n| n[0].as_deref());
    let qk_k = qk.and_then(|n| n[1].as_deref());
    let one = |gain: Option<&[i64]>, dst: &mut [i64]| -> Result<(), StepError> {
        super::loader::qk_norm_heads(gain, dst, dst.len(), eps)?;
        rotate_head_prefixes(rope, dst, dst.len(), pos)
    };
    if let Some((_, e)) =
        super::chart::ffn::attn_heads(query.len() / hd, hd, query, true, |_, dst| one(qk_q, dst))
    {
        return Err(e);
    }
    if let Some((_, e)) =
        super::chart::ffn::attn_heads(key.len() / hd, hd, key, true, |_, dst| one(qk_k, dst))
    {
        return Err(e);
    }
    Ok(())
}

/// The seven `blk.{layer}.*` Q8-active decode weights of one layer, resolved once per step.
///
/// The tensor directory's `get` is a linear scan; the composed step calls it once per projection
/// per layer, so a token pays `n_layer · 7` scans and seven `format!` allocations per layer. This
/// table is one pass over the directory; each accessor is then an index, and the tensor's own
/// `name` serves as the `pq2_lut` cache key without an allocation. Part order is
/// `[q, k, v, attn_output, ffn_gate, ffn_up, ffn_down]`, the order the composed step consumes them,
/// so a missing tensor reports the same [`StepError::MissingTensor`] the name-based path reports.
#[derive(Clone, Copy)]
struct LayerIndex {
    idx: [Option<usize>; N_PARTS],
}

/// Parts per layer in [`LayerIndex`].
const N_PARTS: usize = 13;

/// The `blk.{layer}.*` part suffixes of [`LayerIndex`], in consumption order.
const LAYER_PARTS: [&str; N_PARTS] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    // The affine mixer's projections: `[q | k | v]` into the convolution, the output gate `z`,
    // the per-head `β` and decay inputs, and the output.
    "attn_qkv",
    "attn_gate",
    "ssm_beta",
    "ssm_alpha",
    "ssm_out",
    // The multi-token-prediction block's input projection `[enorm(e) | hnorm(h)] → n_embd`.
    "nextn.eh_proj",
];

impl LayerIndex {
    /// The directory index of part `p`, or the same [`StepError::MissingTensor`] the name-based
    /// `tensor_payload` returns.
    fn at(&self, p: usize, layer: usize) -> Result<usize, StepError> {
        self.idx[p].ok_or_else(|| {
            StepError::MissingTensor(format!("blk.{layer}.{}.weight", LAYER_PARTS[p]))
        })
    }
}

impl Engine {
    /// The per-layer index table of the seven Q8-active weights, from one pass over the directory.
    ///
    /// # Complexity
    ///
    /// `O(directory bytes)` once per step, replacing `O(n_layer · 7 · directory bytes)` scans.
    fn resolve_layers(&self) -> Vec<LayerIndex> {
        let entries = self.table().entries();
        let n = self.config.n_layer + self.config.n_nextn;
        let mut out: Vec<LayerIndex> = vec![
            LayerIndex {
                idx: [None; N_PARTS]
            };
            n
        ];
        let mut i = 0usize;
        while i < entries.len() {
            let name = entries[i].name.as_str();
            if let Some(rest) = name.strip_prefix("blk.") {
                if let Some((l, tail)) = rest.split_once('.') {
                    if let Ok(l) = l.parse::<usize>() {
                        if l < n {
                            if let Some(part) = tail.strip_suffix(".weight") {
                                let mut p = 0usize;
                                while p < LAYER_PARTS.len() {
                                    if LAYER_PARTS[p] == part {
                                        out[l].idx[p] = Some(i);
                                    }
                                    p += 1;
                                }
                            }
                        }
                    }
                }
            }
            i += 1;
        }
        out
    }

    /// The checked payload of directory entry `idx` for a `[n_in, n_out]` tensor: the same bytes,
    /// refusals and borrowed lifetime as `tensor_payload`, without the name scan or the up-front
    /// `expected` allocation (built only on the refusal path).
    fn payload_at(
        &self,
        idx: usize,
        n_out: usize,
        n_in: usize,
        sched: Sched,
    ) -> Result<(u32, Cow<'_, [u8]>), StepError> {
        let t = &self.table().entries()[idx];
        if t.dims.len() != 2 || t.dims[0] != n_in as u64 || t.dims[1] != n_out as u64 {
            return Err(StepError::BadShape {
                tensor: t.name.clone(),
                dims: t.dims.clone(),
                expected: vec![n_in as u64, n_out as u64],
            });
        }
        let n_bytes = t.n_bytes.ok_or(StepError::UnsupportedQuant {
            ggml_type: t.ggml_type,
        })?;
        let raw = match sched {
            Sched::Resident => Cow::Borrowed(
                self.payload(t.byte_range.start, n_bytes)
                    .map_err(StepError::Read)?,
            ),
            #[cfg(test)]
            Sched::Sequential => Cow::Owned(
                self.read_range(t.byte_range.start, n_bytes)
                    .map_err(StepError::Read)?,
            ),
        };
        Ok((t.ggml_type, raw))
    }

    /// The directory entry name of `idx`, for the `pq2_lut` cache key and the fold contract.
    fn name_at(&self, idx: usize) -> &str {
        self.table().entries()[idx].name.as_str()
    }

    /// `step_proj_sched`'s resident `Q8Act` arm over a resolved directory index.
    fn proj_q8act_at(
        &self,
        idx: usize,
        n_out: usize,
        n_in: usize,
        x: &[i64],
        chunks: usize,
    ) -> Result<Vec<i64>, StepError> {
        let name = self.name_at(idx);
        let (ggml_type, raw) = self.payload_at(idx, n_out, n_in, Sched::Resident)?;
        let folded = folded_activation(self.hadamard(), name, x)?;
        let activation: &[i64] = folded.as_deref().unwrap_or(x);
        self.proj_q8act_quantified(
            name,
            ggml_type,
            raw.as_ref(),
            n_out,
            n_in,
            activation,
            chunks,
        )
        .map_err(StepError::Matvec)
    }

    /// One `Q8Act` projection group: the layer parts `parts` (indices into [`LAYER_PARTS`]) all
    /// read the same input `x`, so one pooled quantization of `x` feeds every row kernel. Used by
    /// the attention `q/k/v` group and the affine mixer's four input projections.
    ///
    /// A `Hadamard`-folded member keeps one name-based `proj_q8act_at` per part, as the name path
    /// keeps one `step_proj_sched` per part.
    ///
    /// # Complexity
    ///
    /// `O(Σ outs · n_in)` multiply-adds over the pool, plus one `O(n_in)` quantization.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer index, parts, shapes, activation and dispatch width are the group's input"
    )]
    fn group_q8act_at(
        &self,
        li: &LayerIndex,
        layer: usize,
        parts: &[usize],
        outs: &[usize],
        n_in: usize,
        x: &[i64],
        chunks: usize,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        if parts.len() != outs.len() {
            return Err(StepError::BadConfig("projection group shape"));
        }
        let n = parts.len();
        let mut idxs = Vec::with_capacity(n);
        for &p in parts {
            idxs.push(li.at(p, layer)?);
        }
        let mut types = Vec::with_capacity(n);
        let mut raws: Vec<Cow<'_, [u8]>> = Vec::with_capacity(n);
        for i in 0..n {
            if folded_activation(self.hadamard(), self.name_at(idxs[i]), x)?.is_some() {
                let mut out = Vec::with_capacity(n);
                for j in 0..n {
                    out.push(self.proj_q8act_at(idxs[j], outs[j], n_in, x, chunks)?);
                }
                return Ok(out);
            }
            let (t, raw) = self.payload_at(idxs[i], outs[i], n_in, Sched::Resident)?;
            types.push(t);
            raws.push(raw);
        }
        let mut luts: Vec<Option<Arc<Pq2Lut>>> = Vec::with_capacity(n);
        for i in 0..n {
            luts.push(if types[i] == quant::GGML_TYPE_PQ2_0 {
                Some(self.pq2_lut(self.name_at(idxs[i]), raws[i].as_ref(), outs[i], n_in)?)
            } else {
                None
            });
        }
        let mut specs: Vec<ProjSpec<'_>> = Vec::with_capacity(n);
        for i in 0..n {
            specs.push(ProjSpec {
                n_out: outs[i],
                row_align: if luts[i].is_some() { LUT_ROW_ALIGN } else { 1 },
                source: match &luts[i] {
                    Some(l) => ProjSource::Rows(&**l),
                    None => ProjSource::Chart {
                        ggml_type: types[i],
                        raw: raws[i].as_ref(),
                    },
                },
            });
        }
        projections_q8act(n_in, x, &specs, chunks).map_err(StepError::Matvec)
    }

    /// `ffn_q8act_step` over resolved directory indices: the same three projections and pooled
    /// quantizations, with the `pq2_lut` cache key taken from the directory.
    fn ffn_q8act_at(
        &self,
        li: &LayerIndex,
        layer: usize,
        n_ff: usize,
        n_embd: usize,
        x: &[i64],
        scratch: &mut FfnScratch,
    ) -> Result<Vec<i64>, StepError> {
        let idxs = [li.at(4, layer)?, li.at(5, layer)?, li.at(6, layer)?];
        let mut types = [0u32; 3];
        let mut raws: [Cow<'_, [u8]>; 3] = Default::default();
        for i in 0..3 {
            let (n_out, n_in) = if i < 2 {
                (n_ff, n_embd)
            } else {
                (n_embd, n_ff)
            };
            let (t, raw) = self.payload_at(idxs[i], n_out, n_in, Sched::Resident)?;
            types[i] = t;
            raws[i] = raw;
        }
        let mut luts: [Option<Arc<Pq2Lut>>; 3] = [None, None, None];
        for i in 0..3 {
            let (n_out, n_in) = if i < 2 {
                (n_ff, n_embd)
            } else {
                (n_embd, n_ff)
            };
            if types[i] == quant::GGML_TYPE_PQ2_0 {
                luts[i] =
                    Some(self.pq2_lut(self.name_at(idxs[i]), raws[i].as_ref(), n_out, n_in)?);
            }
        }
        let mut specs: Vec<ProjSpec<'_>> = Vec::with_capacity(3);
        for i in 0..3 {
            let n_out = if i < 2 { n_ff } else { n_embd };
            specs.push(ffn_spec(
                ProjWeights {
                    ggml_type: types[i],
                    raw: raws[i].as_ref(),
                },
                luts[i].as_deref(),
                n_out,
            ));
        }
        let mut it = specs.into_iter();
        let [gate, up, down]: [ProjSpec<'_>; 3] = [it.next(), it.next(), it.next()].map(|s| {
            s.ok_or(MatvecError::ShapeMismatch { x: 0, n_in: 0 })
                .expect("three specs")
        });
        super::chart::ffn::ffn_q8act_prepared_into(
            n_embd,
            x,
            gate,
            up,
            down,
            n_ff,
            super::par::row_chunk_budget(n_ff),
            super::par::row_chunk_budget(n_embd),
            scratch,
        )
        .map_err(StepError::Ffn)
    }
}

/// One FFN tensor's row kernel: its cached `PQ2_0` table as a
/// [`RowKernel`](super::chart::ffn::RowKernel), else the chart row core.
fn ffn_spec<'a>(w: ProjWeights<'a>, lut: Option<&'a Pq2Lut>, n_out: usize) -> ProjSpec<'a> {
    ProjSpec {
        n_out,
        row_align: if lut.is_some() { LUT_ROW_ALIGN } else { 1 },
        source: match lut {
            Some(k) => ProjSource::Rows(k),
            None => ProjSource::Chart {
                ggml_type: w.ggml_type,
                raw: w.raw,
            },
        },
    }
}

impl Engine {
    /// One `Q8Act` FFN block (`SwiGLU`) for `layer`: the `PQ2_0` tensors use their cached lookup
    /// tables as [`RowKernel`](super::chart::ffn::RowKernel)s, the others the chart row core, and
    /// [`ffn_q8act_prepared`](super::chart::ffn::ffn_q8act_prepared) takes one pooled `quant_act`
    /// of `x` and one of the hidden.
    ///
    /// # Errors
    ///
    /// As [`FfnError`]: the geometry refusal, a projection's payload/quant/accumulator refusal and
    /// the elementwise overflow.
    fn ffn_q8act_step(
        &self,
        layer: usize,
        gate: ProjWeights<'_>,
        up: ProjWeights<'_>,
        down: ProjWeights<'_>,
        n_ff: usize,
        x: &[i64],
    ) -> Result<Vec<i64>, FfnError> {
        let n_embd = x.len();
        let gate_lut = self.ffn_lut(layer, "ffn_gate", gate, n_ff, n_embd)?;
        let up_lut = self.ffn_lut(layer, "ffn_up", up, n_ff, n_embd)?;
        let down_lut = self.ffn_lut(layer, "ffn_down", down, n_embd, n_ff)?;
        super::chart::ffn::ffn_q8act_prepared(
            n_embd,
            x,
            ffn_spec(gate, gate_lut.as_deref(), n_ff),
            ffn_spec(up, up_lut.as_deref(), n_ff),
            ffn_spec(down, down_lut.as_deref(), n_embd),
            n_ff,
            super::par::row_chunk_budget(n_ff),
            super::par::row_chunk_budget(n_embd),
        )
    }

    /// `layer`'s cached `PQ2_0` table for the named FFN tensor, or `None` for another quant family.
    fn ffn_lut(
        &self,
        layer: usize,
        part: &str,
        w: ProjWeights<'_>,
        n_out: usize,
        n_in: usize,
    ) -> Result<Option<Arc<Pq2Lut>>, MatvecError> {
        if w.ggml_type == quant::GGML_TYPE_PQ2_0 {
            Ok(Some(self.pq2_lut(
                &format!("blk.{layer}.{part}.weight"),
                w.raw,
                n_out,
                n_in,
            )?))
        } else {
            Ok(None)
        }
    }

    /// The Q32.32 embedding vector of token `id`.
    ///
    /// Reads one row of `token_embd.weight` (row stride from the quant geometry) and dequantizes it
    /// with no float; the row range is checked against the tensor extent and the read against the
    /// file length.
    ///
    /// # Errors
    ///
    /// [`EngineError::TokenOutOfRange`], [`EngineError::UnsupportedEmbeddingQuant`],
    /// [`EngineError::RowOutOfBounds`], [`EngineError::NonFiniteScale`] and the read/quant errors.
    pub fn token_embedding(&self, id: u32) -> Result<Vec<i64>, EngineError> {
        self.embedding_sched(id, Sched::Resident)
    }

    /// [`Engine::token_embedding`]'s body with the payload source chosen by `sched`: a borrowed
    /// resident slice under [`Sched::Resident`], an owned `read_range` copy under the test-only
    /// [`Sched::Sequential`].
    fn embedding_sched(&self, id: u32, sched: Sched) -> Result<Vec<i64>, EngineError> {
        let t = self.token_embd();
        let vocab = self.config.vocab_size;
        let id_usize =
            usize::try_from(id).map_err(|_| EngineError::TokenOutOfRange { id, vocab })?;
        if id_usize >= vocab {
            return Err(EngineError::TokenOutOfRange { id, vocab });
        }
        let (qk, bb) =
            quant::block_geom(t.ggml_type).ok_or(EngineError::UnsupportedEmbeddingQuant {
                ggml_type: t.ggml_type,
            })?;
        let qk = u64::try_from(qk).map_err(|_| EngineError::LengthOverflow)?;
        let row_elems =
            u64::try_from(self.config.n_embd).map_err(|_| EngineError::LengthOverflow)?;
        let blocks = row_elems.div_ceil(qk);
        let row_bytes = blocks
            .checked_mul(u64::try_from(bb).map_err(|_| EngineError::LengthOverflow)?)
            .ok_or(EngineError::LengthOverflow)?;
        let at = t
            .byte_range
            .start
            .checked_add(
                u64::from(id)
                    .checked_mul(row_bytes)
                    .ok_or(EngineError::LengthOverflow)?,
            )
            .ok_or(EngineError::LengthOverflow)?;
        let end = at
            .checked_add(row_bytes)
            .ok_or(EngineError::LengthOverflow)?;
        if end > t.byte_range.end {
            return Err(EngineError::RowOutOfBounds {
                at,
                n: row_bytes,
                end: t.byte_range.end,
            });
        }
        let raw = match sched {
            Sched::Resident => Cow::Borrowed(self.payload(at, row_bytes)?),
            #[cfg(test)]
            Sched::Sequential => Cow::Owned(self.read_range(at, row_bytes)?),
        };
        let mut out: Vec<i64> = Vec::with_capacity(self.config.n_embd);
        dequant_row_fixed(t.ggml_type, raw.as_ref(), self.config.n_embd, &mut out)?;
        if out.len() != self.config.n_embd {
            return Err(EngineError::ConfigMismatch(format!(
                "embedding row produced {} of {} weights",
                out.len(),
                self.config.n_embd
            )));
        }
        Ok(out)
    }

    /// The load-time `RoPE` frequency table for this model (port-map row 3).
    ///
    /// Builds [`RopeTable`] once from the loader's `f64` `rope_theta`/`head_dim` metadata; a decode
    /// loop holds it and calls [`RopeTable::rotate`] per token.
    ///
    /// # Errors
    ///
    /// [`RopeError`] for a malformed head geometry, a non-positive or non-finite `rope_theta`, or a
    /// frequency that does not fit Q32.32.
    pub fn rope_table(&self) -> Result<RopeTable, RopeError> {
        RopeTable::new(self.config.rope_theta, self.config.rot_dims)
    }

    /// The output (LM-head) tensor entry (port-map row 7): `output.weight` when the directory
    /// carries it, otherwise the tied `token_embd.weight`.
    ///
    /// Both are `[n_embd, vocab]` row-major, so the head projection is the row-4 kernel with
    /// `n_out = vocab` and `n_in = n_embd`. The choice is read from the loaded tensor table, never
    /// assumed.
    #[must_use]
    pub fn lm_head(&self) -> &TensorEntry {
        self.table()
            .get("output.weight")
            .unwrap_or_else(|| self.token_embd())
    }

    /// The Q32.32 LM-head logits of a residual activation `x` (port-map row 7).
    ///
    /// The head tensor is [`Engine::lm_head`] (`output.weight` when present, else the tied
    /// `token_embd.weight`), read over its exact checked extent and projected by [`matvec_fixed`]
    /// with `n_out = vocab`, `n_in = n_embd`; output `o` is `round(Σ_i W[o,i]·x_i / 2^32)`. No
    /// float appears on the executing path.
    ///
    /// # Errors
    ///
    /// [`LmHeadError`]'s shape, quant, payload, length, read and matvec refusals; no clamping.
    ///
    /// # Complexity
    ///
    /// `O(vocab·n_embd)` checked multiply-adds plus one dequantization per weight element;
    /// `O(vocab·n_embd)` payload bytes and `O(n_embd)` row scratch.
    pub fn lm_head_fixed(&self, x: &[i64]) -> Result<Vec<i64>, LmHeadError> {
        self.lm_head_fixed_sched(x, Sched::Resident, KernelMode::Exact)
    }

    /// [`Engine::lm_head_fixed`]'s body with the payload source and row schedule chosen by
    /// `sched`: a borrowed resident slice projected by [`super::par::matvec_rows_chunked`] under
    /// [`Sched::Resident`], an owned `read_range` copy projected by the unchunked
    /// [`lm_head::lm_head_fixed`] under the test-only [`Sched::Sequential`]; `mode` selects the
    /// `Q8Act` chart kernel.
    fn lm_head_fixed_sched(
        &self,
        x: &[i64],
        sched: Sched,
        mode: KernelMode,
    ) -> Result<Vec<i64>, LmHeadError> {
        let n_embd = self.config.n_embd;
        let vocab = self.config.vocab_size;
        if x.len() != n_embd {
            return Err(LmHeadError::ShapeMismatch { x: x.len(), n_embd });
        }
        let t = self.lm_head();
        let need = lm_head::head_payload_len(t.ggml_type, &t.dims, n_embd, vocab)?;
        let extent = t.byte_range.end - t.byte_range.start;
        if need > extent {
            return Err(LmHeadError::ShortPayload { need, have: extent });
        }
        let raw = match sched {
            Sched::Resident => Cow::Borrowed(
                self.payload(t.byte_range.start, need)
                    .map_err(LmHeadError::Read)?,
            ),
            #[cfg(test)]
            Sched::Sequential => Cow::Owned(
                self.read_range(t.byte_range.start, need)
                    .map_err(LmHeadError::Read)?,
            ),
        };
        match (sched, mode) {
            (Sched::Resident, KernelMode::Exact) => {
                // One row chunk is exactly `lm_head_fixed`'s unchunked body; keep that leaf on
                // the executing path for the single-chunk case and chunk only when it pays.
                let budget = super::par::row_chunk_budget(vocab);
                if budget <= 1 {
                    lm_head::lm_head_fixed(t.ggml_type, raw.as_ref(), vocab, n_embd, x)
                } else {
                    super::par::matvec_rows_chunked(
                        t.ggml_type,
                        raw.as_ref(),
                        vocab,
                        n_embd,
                        x,
                        budget,
                    )
                    .map_err(LmHeadError::Matvec)
                }
            }
            (Sched::Resident, KernelMode::Q8Act) => self
                .proj_q8act_quantified(
                    &t.name,
                    t.ggml_type,
                    raw.as_ref(),
                    vocab,
                    n_embd,
                    x,
                    super::par::row_chunk_budget(vocab),
                )
                .map_err(LmHeadError::Matvec),
            #[cfg(test)]
            (Sched::Sequential, KernelMode::Exact) => {
                lm_head::lm_head_fixed(t.ggml_type, raw.as_ref(), vocab, n_embd, x)
            }
            #[cfg(test)]
            (Sched::Sequential, KernelMode::Q8Act) => q8act_projection(
                self,
                &t.name,
                t.ggml_type,
                raw.as_ref(),
                vocab,
                n_embd,
                x,
                1,
            )
            .map_err(LmHeadError::Matvec),
        }
    }

    /// `rms_eps` rounded once to Q32.32: `round(rms_eps · 2^32)`.
    ///
    /// This is the row-2 load-time float boundary for the norm addend. A negative, non-finite or
    /// out-of-`i64` epsilon refuses with [`StepError::BadConfig`].
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the finite range guard above the cast"
    )]
    fn eps_fixed(&self) -> Result<i64, StepError> {
        let v = self.config.rms_eps;
        if !v.is_finite() || v < 0.0 {
            return Err(StepError::BadConfig("rms_eps"));
        }
        let q = (v * 4_294_967_296.0).round();
        if !(q.is_finite() && (0.0..9_223_372_036_854_775_808.0).contains(&q)) {
            return Err(StepError::BadConfig("rms_eps"));
        }
        Ok(q as i64)
    }

    /// The checked payload of the `[n_in, n_out]` tensor `name` for `sched`.
    ///
    /// Refuses a missing tensor, a stored shape that is not `[n_in, n_out]`, and a quant family
    /// with no geometry (so no byte count). The bytes are a borrowed resident slice under
    /// [`Sched::Resident`] and an owned `read_range` copy under the test-only [`Sched::Sequential`];
    /// both are the same checked extent, so the dequantizers see identical bytes.
    fn tensor_payload<'a>(
        &'a self,
        name: &str,
        n_out: usize,
        n_in: usize,
        sched: Sched,
    ) -> Result<(u32, Cow<'a, [u8]>), StepError> {
        let t = self
            .table()
            .get(name)
            .ok_or_else(|| StepError::MissingTensor(name.to_string()))?;
        let expected = vec![
            u64::try_from(n_in).map_err(|_| StepError::LengthOverflow)?,
            u64::try_from(n_out).map_err(|_| StepError::LengthOverflow)?,
        ];
        if t.dims != expected {
            return Err(StepError::BadShape {
                tensor: name.to_string(),
                dims: t.dims.clone(),
                expected,
            });
        }
        let n_bytes = t.n_bytes.ok_or(StepError::UnsupportedQuant {
            ggml_type: t.ggml_type,
        })?;
        let raw = match sched {
            Sched::Resident => Cow::Borrowed(
                self.payload(t.byte_range.start, n_bytes)
                    .map_err(StepError::Read)?,
            ),
            #[cfg(test)]
            Sched::Sequential => Cow::Owned(
                self.read_range(t.byte_range.start, n_bytes)
                    .map_err(StepError::Read)?,
            ),
        };
        Ok((t.ggml_type, raw))
    }

    /// One projection stage with the payload source and row schedule chosen by `sched`.
    ///
    /// A tensor named in `prism.hadamard.weight_names` is projected with its activation rotated by
    /// [`folded_activation`] first; every other tensor projects `x` verbatim. [`Sched::Resident`]
    /// projects the borrowed payload with [`super::par::matvec_rows_chunked`] over [`super::par::row_chunk_budget`] rows;
    /// the test-only [`Sched::Sequential`] projects an owned copy with one unchunked
    /// [`matvec_fixed`].
    fn step_proj_sched(
        &self,
        name: &str,
        n_out: usize,
        n_in: usize,
        x: &[i64],
        sched: Sched,
        mode: KernelMode,
    ) -> Result<Vec<i64>, StepError> {
        let (ggml_type, raw) = self.tensor_payload(name, n_out, n_in, sched)?;
        let folded = folded_activation(self.hadamard(), name, x)?;
        let activation: &[i64] = match &folded {
            Some(v) => v.as_slice(),
            None => x,
        };
        let out = match (sched, mode) {
            (Sched::Resident, KernelMode::Exact) => super::par::matvec_rows_chunked(
                ggml_type,
                raw.as_ref(),
                n_out,
                n_in,
                activation,
                super::par::row_chunk_budget(n_out),
            ),
            (Sched::Resident, KernelMode::Q8Act) => self.proj_q8act_quantified(
                name,
                ggml_type,
                raw.as_ref(),
                n_out,
                n_in,
                activation,
                super::par::row_chunk_budget(n_out),
            ),
            #[cfg(test)]
            (Sched::Sequential, KernelMode::Exact) => {
                matvec_fixed(ggml_type, raw.as_ref(), n_out, n_in, activation)
            }
            #[cfg(test)]
            (Sched::Sequential, KernelMode::Q8Act) => q8act_projection(
                self,
                name,
                ggml_type,
                raw.as_ref(),
                n_out,
                n_in,
                activation,
                1,
            ),
        };
        out.map_err(StepError::Matvec)
    }

    /// Q/K/V projections for `layer` over `x` under the schedule and kernel `sched`/`mode` choose:
    /// under the resident `Q8Act` path, when none of the three is `Hadamard`-folded, they share one
    /// `quant_act(x)` and one [`projections_q8act`] dispatch; every other case keeps the three
    /// [`Engine::step_proj_sched`] calls. Errors as [`Engine::step_proj_sched`], Q/K/V in order.
    fn step_qkv_sched(
        &self,
        layer: usize,
        outs: [usize; 3],
        n_in: usize,
        x: &[i64],
        sched: Sched,
        mode: KernelMode,
    ) -> Result<[Vec<i64>; 3], StepError> {
        let names = ["q", "k", "v"].map(|p| format!("blk.{layer}.attn_{p}.weight"));
        let separate = |this: &Self| -> Result<_, StepError> {
            Ok([
                this.step_proj_sched(&names[0], outs[0], n_in, x, sched, mode)?,
                this.step_proj_sched(&names[1], outs[1], n_in, x, sched, mode)?,
                this.step_proj_sched(&names[2], outs[2], n_in, x, sched, mode)?,
            ])
        };
        if !matches!(sched, Sched::Resident) || mode != KernelMode::Q8Act {
            return separate(self);
        }
        let mut types = [0u32; 3];
        let mut raws: [Cow<'_, [u8]>; 3] = Default::default();
        for i in 0..3 {
            // A folded tensor rotates its own activation copy; the three then share no `quant_act`.
            if folded_activation(self.hadamard(), &names[i], x)?.is_some() {
                return separate(self);
            }
            let (t, raw) = self.tensor_payload(&names[i], outs[i], n_in, Sched::Resident)?;
            types[i] = t;
            raws[i] = raw;
        }
        let mut luts: [Option<Arc<Pq2Lut>>; 3] = [None, None, None];
        for i in 0..3 {
            if types[i] == quant::GGML_TYPE_PQ2_0 {
                luts[i] = Some(self.pq2_lut(&names[i], raws[i].as_ref(), outs[i], n_in)?);
            }
        }
        let spec = |i: usize| ProjSpec {
            n_out: outs[i],
            row_align: if luts[i].is_some() { LUT_ROW_ALIGN } else { 1 },
            source: match &luts[i] {
                Some(l) => ProjSource::Rows(&**l),
                None => ProjSource::Chart {
                    ggml_type: types[i],
                    raw: raws[i].as_ref(),
                },
            },
        };
        let specs = [spec(0), spec(1), spec(2)];
        let out = projections_q8act(n_in, x, &specs, super::par::row_chunk_budget(outs[0]))?;
        let [query, key, value]: [Vec<i64>; 3] = out.try_into().expect("three projections");
        Ok([query, key, value])
    }

    /// One projection stage over the resident payload (see [`Engine::step_proj_sched`]).
    ///
    /// Test-only: the executing [`Engine::compose_logits`] calls [`Engine::step_proj_sched`]
    /// directly; the existing projection tests keep this resident-path entry point.
    #[cfg(test)]
    fn step_proj(
        &self,
        name: &str,
        n_out: usize,
        n_in: usize,
        x: &[i64],
    ) -> Result<Vec<i64>, StepError> {
        self.step_proj_sched(name, n_out, n_in, x, Sched::Resident, KernelMode::Exact)
    }

    /// One greedy decode step with a caller-owned KV cache (the stateful rows 0–8 composition).
    ///
    /// The position is the cache's `KvState::occupied(0)`, and the layer order and residual
    /// placement are the fold order named in the module documentation.
    ///
    /// # Errors
    ///
    /// Each stage's typed refusal through [`StepError`] — the token, cache, tensor, geometry and
    /// kernel refusals; no value is clamped.
    pub fn step_with_kv(&self, kv: &mut KvState, token_id: u32) -> Result<StepResult, StepError> {
        let logits = self.step_logits(kv, token_id)?;
        let token = argmax_fixed(&logits).map_err(StepError::Sampler)?;
        Ok(StepResult {
            token,
            logits_checksum: fixed_vector_checksum(&logits),
        })
    }

    /// The Q32.32 logits of one decode step with a caller-owned KV cache (rows 0–8 composed).
    ///
    /// [`Engine::step_with_kv`]'s body separated from its sampler: it advances `kv` by one position
    /// for `token_id` and returns the logit vector [`Engine::lm_head_fixed`] produces from the final
    /// normed residual, at the cache's `KvState::occupied(0)`.
    ///
    /// # Errors
    ///
    /// Each stage's typed refusal through [`StepError`]; no value is clamped.
    ///
    /// # Complexity
    ///
    /// `O(n_layer·(n_embd² + n_embd·n_ff) + vocab·n_embd)` checked multiply-adds plus the LM-head
    /// dequantization; `O(vocab)` for the returned logits.
    pub fn step_logits(&self, kv: &mut KvState, token_id: u32) -> Result<Vec<i64>, StepError> {
        self.compose_logits(
            kv,
            None,
            token_id,
            Sched::Resident,
            KernelMode::Exact,
            false,
        )
    }

    /// The Q32.32 logits of one decode step under the projection kernel `mode`.
    ///
    /// `KernelMode::Exact` is [`Engine::step_logits`] byte-for-byte; `KernelMode::Q8Act` routes every
    /// projection through the pooled int8-activation kernels.
    ///
    /// # Errors
    /// As [`Engine::step_logits`].
    pub fn step_logits_with_mode(
        &self,
        kv: &mut KvState,
        token_id: u32,
        mode: KernelMode,
    ) -> Result<Vec<i64>, StepError> {
        self.compose_logits(
            kv,
            None,
            token_id,
            Sched::Resident,
            mode,
            mode == KernelMode::Q8Act,
        )
    }

    /// [`Engine::compose_final`] followed by the head: one step's logits.
    ///
    /// # Errors
    ///
    /// As [`Engine::compose_final`], plus the head's [`LmHeadError`].
    fn compose_logits(
        &self,
        kv: &mut KvState,
        affine: Option<&mut AffineState>,
        token_id: u32,
        sched: Sched,
        mode: KernelMode,
        parallel_heads: bool,
    ) -> Result<Vec<i64>, StepError> {
        let xf = self.compose_final(kv, affine, token_id, sched, mode, parallel_heads)?;
        self.lm_head_fixed_sched(&xf, sched, mode)
            .map_err(StepError::LmHead)
    }

    /// The trunk of one step: embedding, every block, and the output norm; returns the normed final
    /// state the head reads. `parallel_heads` selects the [`super::chart::ffn::attn_heads`] pool
    /// dispatch, the serial loop being the test oracle.
    #[allow(clippy::too_many_lines, reason = "one linear rows 0-8 composition")]
    pub(crate) fn compose_final(
        &self,
        kv: &mut KvState,
        affine: Option<&mut AffineState>,
        token_id: u32,
        sched: Sched,
        mode: KernelMode,
        parallel_heads: bool,
    ) -> Result<Vec<i64>, StepError> {
        let mut out =
            self.compose_cols(kv, affine, &[token_id], sched, mode, parallel_heads, None)?;
        out.pop().ok_or(StepError::BadConfig("empty column block"))
    }

    /// The pre-change sequential reference for the bit-identity test: [`Engine::step_logits`]'s
    /// composition with owned `read_range` payloads and one unchunked [`matvec_fixed`] per
    /// projection. Test-only; the executing path is [`Engine::step_logits`].
    #[cfg(test)]
    fn step_logits_sequential(
        &self,
        kv: &mut KvState,
        token_id: u32,
    ) -> Result<Vec<i64>, StepError> {
        self.compose_logits(
            kv,
            None,
            token_id,
            Sched::Sequential,
            KernelMode::Exact,
            false,
        )
    }

    /// One greedy decode step with a fresh one-position [`KvState`].
    ///
    /// The stateless single-token forward at position 0; a decode loop that accumulates context
    /// calls [`Engine::step_with_kv`].
    ///
    /// # Errors
    /// As [`Engine::step_with_kv`].
    pub fn step(&mut self, token_id: u32) -> Result<StepResult, StepError> {
        let mut kv = self.decode_state(1)?;
        self.step_with_kv(&mut kv, token_id)
    }

    /// A decode [`KvState`] with this engine's geometry and capacity `cap`.
    ///
    /// The loader-side wiring for [`Engine::step_with_kv`]: the caller asks for the capacity its
    /// decode loop needs, bounded by the config's `context_length` when the file states one.
    ///
    /// # Errors
    ///
    /// [`StepError::BadConfig`] for a zero or above-context capacity, and [`StepError::Kv`] with
    /// [`AttnError`] for a geometry [`KvState::new`] refuses.
    pub fn decode_state(&self, cap: usize) -> Result<KvState, StepError> {
        if cap == 0 {
            return Err(StepError::BadConfig("kv capacity"));
        }
        let ctx = self.config.context_length;
        if ctx != 0 && cap > ctx {
            return Err(StepError::BadConfig("kv capacity above context_length"));
        }
        KvState::new(
            self.config
                .mixers
                .iter()
                .filter(|m| **m == MixerKind::Softmax)
                .count(),
            self.config.n_head,
            self.config.n_head_kv,
            self.config.head_dim,
            cap,
        )
        .map_err(StepError::Kv)
    }

    /// A decode state for this engine's schedule: the softmax list carrier sized `cap`, and the
    /// affine carrier when any layer is affine.
    ///
    /// # Errors
    ///
    /// As [`Engine::decode_state`].
    ///
    /// # Complexity
    ///
    /// The [`KvState`] planes plus `O(n_affine · state words)`, zeroed once.
    pub fn state(&self, cap: usize) -> Result<DecodeState, StepError> {
        let kv = self.decode_state(cap)?;
        let n_affine = self
            .config
            .mixers
            .iter()
            .filter(|m| **m == MixerKind::Affine)
            .count();
        let affine = match self.config.affine {
            Some(g) if n_affine > 0 => Some(AffineState::new(&g, n_affine)),
            _ => None,
        };
        Ok(DecodeState { kv, affine })
    }

    /// One greedy decode step over any schedule; `mode` selects the projection kernel.
    ///
    /// # Errors
    ///
    /// As [`Engine::step_with_kv`], plus [`StepError::BadConfig`] when an affine layer has no
    /// affine state.
    pub fn step_state(
        &self,
        st: &mut DecodeState,
        token_id: u32,
        mode: KernelMode,
    ) -> Result<StepResult, StepError> {
        let logits = self.step_logits_state(st, token_id, mode)?;
        let token = argmax_fixed(&logits).map_err(StepError::Sampler)?;
        Ok(StepResult {
            token,
            logits_checksum: fixed_vector_checksum(&logits),
        })
    }

    /// The logits of one decode step over any schedule (the one executor).
    ///
    /// # Errors
    ///
    /// As [`Engine::step_state`].
    pub fn step_logits_state(
        &self,
        st: &mut DecodeState,
        token_id: u32,
        mode: KernelMode,
    ) -> Result<Vec<i64>, StepError> {
        self.compose_logits(
            &mut st.kv,
            st.affine.as_mut(),
            token_id,
            Sched::Resident,
            mode,
            mode == KernelMode::Q8Act,
        )
    }

    /// Prefill `tokens` in blocks of [`prefill_block`] columns (weights read once per block),
    /// returning every position's output-normed final state; exactly the states of one-token
    /// steps (`qwen35_two_columns_equal_two_single_steps`).
    ///
    /// # Errors
    ///
    /// As [`Engine::step_state`].
    ///
    /// # Complexity
    ///
    /// `⌈n / prefill_block()⌉` block passes; each reads the `Q8_0` weights once.
    pub(crate) fn prefill_cols(
        &self,
        st: &mut DecodeState,
        tokens: &[u32],
        mode: KernelMode,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let mut out = Vec::with_capacity(tokens.len());
        for block in tokens.chunks(prefill_block()) {
            out.extend(self.compose_cols(
                &mut st.kv,
                st.affine.as_mut(),
                block,
                Sched::Resident,
                mode,
                mode == KernelMode::Q8Act,
                None,
            )?);
        }
        Ok(out)
    }

    /// Prefill `tokens` in blocks and return the greedy next token (the prompt's continuation).
    ///
    /// # Errors
    ///
    /// As [`Engine::step_state`]; [`StepError::BadConfig`] for an empty prompt.
    pub fn prefill_state(
        &self,
        st: &mut DecodeState,
        tokens: &[u32],
        mode: KernelMode,
    ) -> Result<u32, StepError> {
        let (last, head) = tokens
            .split_last()
            .ok_or(StepError::BadConfig("empty prompt"))?;
        self.prefill_cols(st, head, mode)?;
        self.next_token_state(st, *last, mode)
    }

    /// The greedy next token of one decode step over any schedule. In `Q8Act` with a `Q8_0` head it
    /// reads the head through the bit-plane argmax ([`super::kern::bitplane`], licensed by
    /// `BitplaneArgmax.bitplane_argmax_exact`); it returns exactly `argmax_fixed` of
    /// [`Engine::step_logits_state`]'s logits, without forming them.
    ///
    /// # Errors
    ///
    /// As [`Engine::step_state`].
    ///
    /// # Complexity
    ///
    /// The trunk plus, on the bit-plane path, `18/34` of the head's bytes and the survivors' low
    /// planes; otherwise the dense head.
    pub fn next_token_state(
        &self,
        st: &mut DecodeState,
        token_id: u32,
        mode: KernelMode,
    ) -> Result<u32, StepError> {
        let xf = self.compose_final(
            &mut st.kv,
            st.affine.as_mut(),
            token_id,
            Sched::Resident,
            mode,
            mode == KernelMode::Q8Act,
        )?;
        let (vocab, n_embd) = (self.config.vocab_size, self.config.n_embd);
        let _head = super::trace::span(
            super::trace::Phase::Head,
            1,
            super::trace::q8_bytes(vocab, n_embd),
            super::trace::macs(vocab, n_embd, 1),
        );
        // The bit-plane head is exact and, with the batched stage 1, faster than the dense head
        // on Qwen3.5 (3.01 vs 3.28 ms, median 18 survivors); `APEIRON_BITPLANE=0` forces dense.
        if mode == KernelMode::Q8Act && bitplane_enabled() {
            if let Some(bp) = self.bitplane_head() {
                if let Some(q) = super::chart::ffn::quant_act_pooled(&xf) {
                    if super::chart::q8act_fast_bound_ok(xf.len(), 128, &q, true) {
                        if let Some((token, _)) = bp.argmax(&q) {
                            return Ok(token);
                        }
                    }
                }
            }
        }
        let logits = self
            .lm_head_fixed_sched(&xf, Sched::Resident, mode)
            .map_err(StepError::LmHead)?;
        argmax_fixed(&logits).map_err(StepError::Sampler)
    }
}

/// The decode state of any schedule: the softmax list carrier and, when the schedule has affine
/// layers, the affine carrier.
#[derive(Debug, Clone)]
pub struct DecodeState {
    /// Keys and values of the softmax layers.
    pub kv: KvState,
    /// The affine layers' fixed-size states.
    pub affine: Option<AffineState>,
}

#[path = "step/cols.rs"]
mod cols;
#[path = "step/spec.rs"]
mod spec;
pub use spec::{MtpState, SpecStats};

#[cfg(test)]
#[path = "step/tests.rs"]
mod tests;
