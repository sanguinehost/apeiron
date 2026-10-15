//! The one executor's layer loop over `k` token columns.
//!
//! [`Engine::compose_cols`] runs `k` consecutive tokens through every block. Column `c` is exactly
//! the single-token step at position `pos + c`, bit for bit:
//! - Every projection column is `matvec_q8act` (or the exact path) on that column's activation.
//!   For `Q8_0` weights with `k > 1` the pooled `matmat_q8act` kernel reads each weight row once
//!   for all columns (measured on DRAM-bound shapes: two columns cost 1.02× one).
//! - Softmax layers append column `c`'s key and value, then attend over positions `0..=pos + c`:
//!   causality within the block.
//! - Affine layers update their state column by column, in token order.
//!
//! `compose_final` is the `k = 1` case: one implementation of the layer loop for every model,
//! schedule and kernel mode. `k = 1` keeps the single-column projection groups, so its speed and
//! its bits are unchanged.

use super::{
    add_assign_fixed, affine_slots, apply_sigmoid_gate, count_mixers, qk_norm_and_rope,
    rms_with_gain_slice, softmax_slots, split_query_gate, LayerIndex, Sched, StepError,
};
use crate::engine::chart::ffn::{quant_acts_pooled, swiglu_cols};
use crate::engine::chart::task::{HeadTask, Task2};
use crate::engine::chart::QuantAct;
use crate::engine::error::{
    TensorKey, PART_ATTN_NORM, PART_FFN_NORM, PART_OUTPUT_NORM, PART_SSM_CONV1D,
};
use crate::engine::kern::gemm::{gemm_q8_0_group, gemm_q8_0_swiglu, GemmPart};
use crate::engine::kern::matmat::matmat_q8act;
use crate::engine::loader::{opt_slice, DecodeVectors};
use crate::engine::mixer::{AffineLayer, AffineState};
use crate::engine::trace::{self, Phase};
use crate::engine::{
    ffn_fixed, AttnError, Engine, KernelMode, KvState, MatvecError, MixerKind, ProjWeights,
};
use crate::format::quant;
use std::sync::OnceLock;

/// The `APEIRON_EXACT_SCAN` flag: the per-thread test override, else the process value parsed once
/// (`OnceLock`) and defaulting to off. [`Engine::softmax_core_block`] reads it on the calling
/// thread before any pool dispatch, so every worker in a dispatch sees one value.
fn exact_scan_enabled() -> bool {
    crate::engine::attn::exact_scan_override().unwrap_or_else(|| {
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| std::env::var("APEIRON_EXACT_SCAN").is_ok_and(|v| v == "1"))
    })
}

/// The key-tile width of the fused exact-scan attention read. The fold's tile boundaries only
/// re-parenthesise the exact `i128` `den`/`num` sums, so the value changes no result; 32 keys keep
/// the `head_dim`-wide numerator hot through a tile.
const EXACT_SCAN_TILE: usize = 32;

impl Engine {
    /// [`Engine::compose_cols_in`] with the optional affine state and rollback sink resolved once
    /// to values: an absent state is an empty [`AffineState`] (its `layer_mut` yields `None`, the
    /// same refusal as before) and an absent sink an empty slice (`get_mut` yields `None`).
    ///
    /// # Errors
    ///
    /// As [`Engine::compose_cols_in`].
    ///
    /// # Complexity
    ///
    /// `O(1)` words besides [`Engine::compose_cols_in`].
    #[allow(
        clippy::too_many_arguments,
        reason = "the caches, columns, schedule, kernel, dispatch and rollback sink are its input"
    )]
    pub(crate) fn compose_cols(
        &self,
        kv: &mut KvState,
        affine: Option<&mut AffineState>,
        tokens: &[u32],
        sched: Sched,
        mode: KernelMode,
        parallel_heads: bool,
        snapshot: Option<&mut [AffineLayer]>,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let mut none_state = AffineState::empty();
        let mut none_snap: [AffineLayer; 0] = [];
        match (affine, snapshot) {
            (Some(a), Some(s)) => {
                self.compose_cols_in(kv, a, tokens, sched, mode, parallel_heads, s)
            }
            (Some(a), None) => {
                self.compose_cols_in(kv, a, tokens, sched, mode, parallel_heads, &mut none_snap)
            }
            (None, Some(s)) => {
                self.compose_cols_in(kv, &mut none_state, tokens, sched, mode, parallel_heads, s)
            }
            (None, None) => self.compose_cols_in(
                kv,
                &mut none_state,
                tokens,
                sched,
                mode,
                parallel_heads,
                &mut none_snap,
            ),
        }
    }

    /// `k = tokens.len()` consecutive decode steps through the trunk, in one pass over the weights
    /// where the kernel allows; returns each column's output-normed final state.
    ///
    /// # Errors
    ///
    /// As the single-token step, for the first column that refuses; the state has then advanced
    /// through the columns before it.
    ///
    /// # Complexity
    ///
    /// The single step's cost per column for everything but the weight streams, which are read
    /// once per pass for `Q8_0` tensors (`O(bytes)` rather than `O(k·bytes)`).
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "one linear composition of the block; the caches, columns, schedule, kernel, \
                  dispatch and rollback sink are its input"
    )]
    pub(crate) fn compose_cols_in(
        &self,
        kv: &mut KvState,
        affine: &mut AffineState,
        tokens: &[u32],
        sched: Sched,
        mode: KernelMode,
        parallel_heads: bool,
        snapshot: &mut [AffineLayer],
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let cfg = &self.config;
        if tokens.is_empty() {
            return Err(StepError::BadConfig(String::from("empty token block")));
        }
        let k = tokens.len();
        let _pass = trace::span(Phase::Pass, k, 0, 0);
        let slots = softmax_slots(&cfg.mixers);
        let aslots = affine_slots(&cfg.mixers);
        let n_softmax = count_mixers(&cfg.mixers, MixerKind::Softmax);
        if kv.n_layer() != n_softmax
            || kv.n_head() != cfg.n_head
            || kv.n_head_kv() != cfg.n_head_kv
            || kv.head_dim() != cfg.head_dim
        {
            return Err(StepError::KvGeometry {
                kv_layer: kv.n_layer(),
                kv_head: kv.n_head(),
                kv_head_kv: kv.n_head_kv(),
                kv_head_dim: kv.head_dim(),
            });
        }
        // Version 2 ties the head to the rotated embedding; the un-rotation needs the inverse of
        // the signed transform, which forward `fwht_fixed` is not in `explicit` mode. Refuse.
        if let Some(meta) = self.hadamard() {
            if meta.version == 2 {
                return Err(StepError::TiedOutputUnsupported);
            }
        }
        let eps = self.eps_fixed()?;
        let vectors = self.decode_vectors();
        let mut hidden: Vec<Vec<i64>> = Vec::with_capacity(k);
        let t_embed = trace::span(Phase::Embed, k, 0, 0);
        let mut res: Result<(), StepError> = Ok(());
        let mut i = 0usize;
        while i < tokens.len() && res.is_ok() {
            res = self.embed_push(tokens[i], sched, &mut hidden);
            i += 1;
        }
        res?;
        drop(t_embed);
        // The resident `Q8Act` composition resolves the per-layer directory entries once per
        // pass (`LayerIndex`); every other schedule keeps the name-based lookups.
        let layers = if matches!(sched, Sched::Resident) && mode == KernelMode::Q8Act {
            self.resolve_layers()
        } else {
            Vec::new()
        };
        let mut ffn_scratch = crate::engine::chart::ffn::FfnScratch::default();

        let mut layer = 0usize;
        let mut res: Result<(), StepError> = Ok(());
        while layer < cfg.n_layer && res.is_ok() {
            res = self.layer_cols(
                kv,
                &mut *affine,
                &mut *snapshot,
                &layers,
                layer,
                slots[layer],
                aslots[layer],
                &mut hidden,
                &mut ffn_scratch,
                eps,
                sched,
                mode,
                parallel_heads,
            );
            layer += 1;
        }
        res?;

        let out_gain = opt_slice(&vectors.output_norm)
            .ok_or_else(|| StepError::MissingTensor(TensorKey::top(PART_OUTPUT_NORM)))?;
        let _norm = trace::span(Phase::Norm, k, 0, 0);
        norm_cols(out_gain, &hidden, eps)
    }

    /// Block `layer` over the columns: the attention-side `RMSNorm`, the mixer primitive (the
    /// softmax list carrier at `slot` in the `KvState`, or the affine carrier at `aslot` in the
    /// `AffineState`) and its residual add, then the FFN-side norm, FFN and residual add, all into
    /// `hidden`. One call per layer so the layer loop of [`Engine::compose_cols`] carries only the
    /// `Result`.
    ///
    /// # Errors
    ///
    /// The first refusal of the block, in the order above.
    ///
    /// # Complexity
    ///
    /// As [`Engine::compose_cols`] for one layer.
    #[allow(
        clippy::too_many_arguments,
        reason = "the caches, layer, its mixer slots, columns, scratch, epsilon, schedule, kernel \
                  and dispatch are the block's input"
    )]
    fn layer_cols(
        &self,
        kv: &mut KvState,
        affine: &mut AffineState,
        snapshot: &mut [AffineLayer],
        layers: &[LayerIndex],
        layer: usize,
        slot: Option<usize>,
        aslot: Option<usize>,
        hidden: &mut [Vec<i64>],
        ffn_scratch: &mut crate::engine::chart::ffn::FfnScratch,
        eps: i64,
        sched: Sched,
        mode: KernelMode,
        parallel_heads: bool,
    ) -> Result<(), StepError> {
        let k = hidden.len();
        let vectors = self.decode_vectors();
        let li = layers.get(layer);
        let attn_gain = norm_gain(vectors, layer, 0, PART_ATTN_NORM)?;
        let t_norm = trace::span(Phase::Norm, k, 0, 0);
        let normed = norm_cols(attn_gain, hidden, eps)?;
        drop(t_norm);
        let mixed = match (slot, aslot) {
            (Some(slot), _) => {
                self.softmax_cols(li, layer, slot, kv, &normed, sched, mode, parallel_heads)
            }
            (None, Some(aslot)) => {
                let state = affine.layer_mut(aslot);
                let snap = snapshot.get_mut(aslot);
                match state {
                    Some(st) => {
                        self.affine_cols(li, layer, st, &normed, sched, mode, parallel_heads, snap)
                    }
                    None => Err(StepError::BadConfig(String::from(
                        "affine state for an affine layer",
                    ))),
                }
            }
            (None, None) => Err(StepError::BadConfig(String::from("layer without a mixer"))),
        }?;
        let t_res = trace::span(Phase::Residual, k, 0, 0);
        add_cols(hidden, &mixed)?;
        drop(t_res);
        let ffn_gain = norm_gain(vectors, layer, 1, PART_FFN_NORM)?;
        let t_norm = trace::span(Phase::Norm, k, 0, 0);
        let h2 = norm_cols(ffn_gain, hidden, eps)?;
        drop(t_norm);
        let ffn_out = self.ffn_cols(li, layer, &h2, sched, mode, ffn_scratch)?;
        let t_res = trace::span(Phase::Residual, k, 0, 0);
        add_cols(hidden, &ffn_out)?;
        drop(t_res);
        Ok(())
    }

    /// Layer part `part`'s projection of every column: the single-column helpers for one column;
    /// for more, the pooled `matmat_q8act` kernel on a resident `Q8_0` tensor (weights read once),
    /// else one single-column projection per column. Every column equals the single path's.
    ///
    /// # Complexity
    ///
    /// `O(n_out·n_in·k)` multiply-adds; weight bytes read once (`Q8_0`, `k > 1`) or `k` times.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer index, part, shape, columns, schedule and kernel are the input"
    )]
    pub(super) fn proj_cols(
        &self,
        li: Option<&LayerIndex>,
        layer: usize,
        part: usize,
        n_out: usize,
        n_in: usize,
        xs: &[Vec<i64>],
        sched: Sched,
        mode: KernelMode,
        pre: Option<&[QuantAct]>,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let Some(li) = li else {
            let key = super::part_key(layer, part);
            let mut out = Vec::with_capacity(xs.len());
            let mut i = 0usize;
            while i < xs.len() {
                out.push(self.step_proj_sched(key, n_out, n_in, &xs[i], sched, mode)?);
                i += 1;
            }
            return Ok(out);
        };
        let idx = li.at(part, layer)?;
        let budget = crate::engine::par::row_chunk_budget(n_out);
        if xs.len() > 1 && self.hadamard().is_none() {
            let (ggml_type, at) = self.payload_at(idx, n_out, n_in, Sched::Resident)?;
            if ggml_type == quant::GGML_TYPE_Q8_0 {
                let owned;
                let qs = if let Some(q) = pre {
                    q
                } else {
                    owned = quant_cols(xs)?;
                    &owned
                };
                return matmat_q8act(
                    ggml_type,
                    &self.resident()[at.off..at.off + at.len],
                    n_out,
                    n_in,
                    qs,
                )
                .map_err(StepError::Matvec);
            }
        }
        let mut out = Vec::with_capacity(xs.len());
        let mut i = 0usize;
        while i < xs.len() {
            out.push(self.proj_q8act_at(
                idx,
                super::part_key(layer, part),
                n_out,
                n_in,
                &xs[i],
                budget,
            )?);
            i += 1;
        }
        Ok(out)
    }

    /// A group of parts that read the same input: the single-column group kernel for one column
    /// (one pooled quantization, one dispatch), [`Engine::proj_cols`] per part for more.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer index, parts, shapes, columns, schedule and kernel are the input"
    )]
    fn group_cols(
        &self,
        li: Option<&LayerIndex>,
        layer: usize,
        parts: &[usize],
        outs: &[usize],
        n_in: usize,
        xs: &[Vec<i64>],
        sched: Sched,
        mode: KernelMode,
    ) -> Result<Vec<Vec<Vec<i64>>>, StepError> {
        if xs.len() == 1 {
            if let Some(li) = li {
                let mut got = self.group_q8act_at(
                    li,
                    layer,
                    parts,
                    outs,
                    n_in,
                    &xs[0],
                    crate::engine::par::row_chunk_budget(outs[0]),
                )?;
                let mut out = Vec::with_capacity(got.len());
                let mut j = 0usize;
                while j < got.len() {
                    out.push(vec![std::mem::take(&mut got[j])]);
                    j += 1;
                }
                return Ok(out);
            }
        }
        // One quantization per column serves every part of the group.
        // The match keeps the extracted test typed: after the caller matched `li` as `None`, Aeneas
        // inlines `none` into `li.is_some()` and the call loses its type argument.
        #[allow(clippy::redundant_pattern_matching, reason = "see above")]
        let has_index = match li {
            Some(_) => true,
            None => false,
        };
        let packed = has_index && xs.len() > 1;
        let qs = if packed { quant_cols(xs)? } else { Vec::new() };
        // The parts of a resident `Q8_0` group share `n_in`, so one activation pack and one row
        // dispatch serve them all. The group defers (`None`) to the per-part loop below, which owns
        // every refusal; a payload error propagates here in part order, as `proj_cols` would.
        let pre: Option<&[QuantAct]> = if packed { Some(&qs) } else { None };
        if let (Some(li), Some(qs)) = (li, pre) {
            if xs.len() > 1 && self.hadamard().is_none() {
                let mut payloads = Vec::with_capacity(parts.len());
                let mut err: Option<StepError> = None;
                let mut i = 0usize;
                while i < parts.len() && err.is_none() {
                    let p = parts[i];
                    match li.at(p, layer) {
                        Ok(idx) => match self.payload_at(idx, outs[i], n_in, Sched::Resident) {
                            Ok((ggml_type, at)) => payloads.push((ggml_type, at, outs[i])),
                            Err(e) => err = Some(e),
                        },
                        Err(e) => err = Some(e),
                    }
                    i += 1;
                }
                if let Some(e) = err {
                    return Err(e);
                }
                let mut group: Vec<GemmPart> = Vec::with_capacity(payloads.len());
                let mut gi = 0usize;
                while gi < payloads.len() {
                    let (ggml_type, at, n_out) = payloads[gi];
                    group.push(GemmPart {
                        ggml_type,
                        at,
                        n_out,
                    });
                    gi += 1;
                }
                if let Some(cols) = gemm_q8_0_group(self.resident(), &group, n_in, qs) {
                    return Ok(cols);
                }
            }
        }
        self.group_each(li, layer, parts, outs, n_in, xs, sched, mode, pre)
    }

    /// [`Engine::proj_cols`] per part, in part order; the first refusal returns.
    ///
    /// # Complexity
    ///
    /// One `proj_cols` per part; the result holds `parts.len()` column sets. Split out of
    /// [`Engine::group_cols`] so the extracted function carries a single loop.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer index, parts, shapes, columns, schedule, kernel and activations are the input"
    )]
    fn group_each(
        &self,
        li: Option<&LayerIndex>,
        layer: usize,
        parts: &[usize],
        outs: &[usize],
        n_in: usize,
        xs: &[Vec<i64>],
        sched: Sched,
        mode: KernelMode,
        pre: Option<&[QuantAct]>,
    ) -> Result<Vec<Vec<Vec<i64>>>, StepError> {
        let mut out = Vec::with_capacity(parts.len());
        let mut err: Option<StepError> = None;
        let mut i = 0usize;
        while i < parts.len() && err.is_none() {
            match self.proj_cols(li, layer, parts[i], outs[i], n_in, xs, sched, mode, pre) {
                Ok(v) => out.push(v),
                Err(e) => err = Some(e),
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        Ok(out)
    }

    /// One softmax-attention layer over the columns: per column the bias, the optional output
    /// gate, q/k norm and `RoPE` at its own position, append, then attention over the prefix
    /// including itself.
    #[allow(
        clippy::too_many_arguments,
        clippy::needless_range_loop,
        reason = "the layer index, layer, slot, cache, columns, schedule, kernel and dispatch"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "this per-column attention path keeps bias, gate, normalization, cache append and attention in their required state order"
    )]
    pub(super) fn softmax_cols(
        &self,
        li: Option<&LayerIndex>,
        layer: usize,
        slot: usize,
        kv: &mut KvState,
        xs: &[Vec<i64>],
        sched: Sched,
        mode: KernelMode,
        parallel_heads: bool,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let cfg = &self.config;
        let (n_head, hd, n_embd) = (cfg.n_head, cfg.head_dim, cfg.n_embd);
        let q_out = n_head.checked_mul(hd).ok_or(StepError::LengthOverflow)?;
        let q_proj = if cfg.attn_gate {
            q_out.checked_mul(2).ok_or(StepError::LengthOverflow)?
        } else {
            q_out
        };
        let kv_out = cfg
            .n_head_kv
            .checked_mul(hd)
            .ok_or(StepError::LengthOverflow)?;
        let eps = self.eps_fixed()?;
        let table = self.rope_table().map_err(StepError::Rope)?;
        let k = xs.len();
        let t_in = trace::span(
            Phase::AttnProjIn,
            k,
            trace::q8_bytes(q_proj + 2 * kv_out, n_embd),
            trace::macs(q_proj + 2 * kv_out, n_embd, k),
        );
        let qkv = if xs.len() == 1 && li.is_none() {
            let mut r =
                self.step_qkv_sched(layer, [q_proj, kv_out, kv_out], n_embd, &xs[0], sched, mode)?;
            let q = std::mem::take(&mut r[0]);
            let k = std::mem::take(&mut r[1]);
            let v = std::mem::take(&mut r[2]);
            vec![vec![q], vec![k], vec![v]]
        } else {
            self.group_cols(
                li,
                layer,
                &[0, 1, 2],
                &[q_proj, kv_out, kv_out],
                n_embd,
                xs,
                sched,
                mode,
            )?
        };
        drop(t_in);
        let t_core = trace::span(Phase::AttnCore, k, 0, 0);
        if k > 1 && parallel_heads {
            let atts = self.softmax_core_block(layer, slot, kv, &qkv, eps, &table)?;
            drop(t_core);
            let _out = trace::span(
                Phase::AttnProjOut,
                k,
                trace::q8_bytes(n_embd, q_out),
                trace::macs(n_embd, q_out, k),
            );
            return self.proj_cols(li, layer, 3, n_embd, q_out, &atts, sched, mode, None);
        }
        let mut atts = Vec::with_capacity(k);
        let mut err: Option<StepError> = None;
        let mut c = 0usize;
        while c < k && err.is_none() {
            match self.softmax_col(layer, slot, kv, &qkv, c, eps, &table, parallel_heads) {
                Ok(att) => atts.push(att),
                Err(e) => err = Some(e),
            }
            c += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        drop(t_core);
        let _out = trace::span(
            Phase::AttnProjOut,
            k,
            trace::q8_bytes(n_embd, q_out),
            trace::macs(n_embd, q_out, k),
        );
        self.proj_cols(li, layer, 3, n_embd, q_out, &atts, sched, mode, None)
    }

    /// One column of [`Engine::softmax_cols`]: bias, gate split, q/k norm and `RoPE` at its position,
    /// append, then attention over the prefix including itself.
    #[allow(
        clippy::too_many_arguments,
        clippy::index_refutable_slice,
        reason = "the per-column inputs of softmax_cols: layer, slot, cache, columns, column, eps, table, dispatch; the three biases are read by constant index because an array pattern does not extract"
    )]
    fn softmax_col(
        &self,
        layer: usize,
        slot: usize,
        kv: &mut KvState,
        qkv: &[Vec<Vec<i64>>],
        c: usize,
        eps: i64,
        table: &crate::engine::rope::RopeTable,
        parallel_heads: bool,
    ) -> Result<Vec<i64>, StepError> {
        let cfg = &self.config;
        let (n_head, hd) = (cfg.n_head, cfg.head_dim);
        let q_out = n_head.checked_mul(hd).ok_or(StepError::LengthOverflow)?;
        let vectors = self.decode_vectors();
        let (mut query, mut key, mut value) =
            (qkv[0][c].clone(), qkv[1][c].clone(), qkv[2][c].clone());
        if let Some(bs) = vectors.biases_of(layer) {
            if let Some(bias) = opt_slice(&bs[0]) {
                add_assign_fixed(&mut query, bias)?;
            }
        }
        if let Some(bs) = vectors.biases_of(layer) {
            if let Some(bias) = opt_slice(&bs[1]) {
                add_assign_fixed(&mut key, bias)?;
            }
        }
        if let Some(bs) = vectors.biases_of(layer) {
            if let Some(bias) = opt_slice(&bs[2]) {
                add_assign_fixed(&mut value, bias)?;
            }
        }
        let gate = if cfg.attn_gate {
            let (q, g) = split_query_gate(&query, n_head, hd)?;
            query = q;
            Some(g)
        } else {
            None
        };
        let pos = kv.occupied(slot).map_err(StepError::Kv)?;
        let pos_u32 = u32::try_from(pos).map_err(|_| StepError::LengthOverflow)?;
        qk_norm_and_rope(
            vectors,
            layer,
            table,
            &mut query,
            &mut key,
            hd,
            eps,
            pos_u32,
            parallel_heads,
        )?;
        let appended = kv.append(slot, &key, &value).map_err(StepError::Kv)?;
        let mut att: Vec<i64> = vec![0i64; q_out];
        let kv_ro: &KvState = kv;
        let head = QueryHeadCtx {
            kv: kv_ro,
            slot,
            query: &query,
            hd,
            appended,
        };
        if let Some((_, e)) =
            crate::engine::chart::ffn::attn_heads(n_head, hd, &mut att, parallel_heads, &head)
        {
            return Err(StepError::Kv(e));
        }
        if let Some(g) = gate {
            apply_sigmoid_gate(&mut att, &g)?;
        }
        Ok(att)
    }

    /// The attention core of a block of `k > 1` columns in three pooled stages, each column equal
    /// to the per-column loop in [`Engine::softmax_cols`]:
    /// 1. per column (one task each): bias, gate split, q/k norm and `RoPE` at position
    ///    `pos₀ + c`;
    /// 2. in token order: append each column's key and value;
    /// 3. per (column, head) (`k·n_head` tasks): attention over positions `0..=pos₀ + c`, then
    ///    that head's slice of the sigmoid output gate. The pool task index runs over the fold
    ///    order ([`fold_col`]) so a contiguous speed-weighted range alternates cheap and expensive
    ///    columns; a gather restores the block order.
    ///
    /// Column `c` sees exactly the keys the per-column loop had appended when it attended, so the
    /// block is causal and bit-identical (`qwen35_two_columns_equal_two_single_steps`). With
    /// `APEIRON_EXACT_SCAN=1` stage 3 reads each head through [`KvState::attn_head_fused`]
    /// instead of [`KvState::attn_head`]; that read is not bit-identical to the per-position path,
    /// and the deviation is bounded (`qwen35_exact_scan_attention_block_matches_baseline_within_bound`).
    /// A refusal leaves the cache unspecified.
    ///
    /// # Complexity
    ///
    /// `O(k·n_head·(pos₀ + k)·head_dim)` for the attention over `k·n_head` pool tasks, plus
    /// `O(k·(q_proj + 2·kv_out))` for stages 1–2; three dispatches instead of `k`.
    #[allow(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        reason = "the three stages read in order; c, h, n, q are the column, head, fold and query \
                  words"
    )]
    fn softmax_core_block(
        &self,
        layer: usize,
        slot: usize,
        kv: &mut KvState,
        qkv: &[Vec<Vec<i64>>],
        eps: i64,
        table: &crate::engine::rope::RopeTable,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let cfg = &self.config;
        let (n_head, hd) = (cfg.n_head, cfg.head_dim);
        let q_out = n_head * hd;
        let kv_out = cfg.n_head_kv * hd;
        let k = qkv[0].len();
        let gated = cfg.attn_gate;
        let vectors = self.decode_vectors();
        let pos0 = kv.occupied(slot).map_err(StepError::Kv)?;
        // Stage 1: per column `[query | key | value | gate]`.
        let stride = 2 * q_out + 2 * kv_out;
        let mut prep = vec![0i64; k * stride];
        let one = PrepColCtx {
            qkv,
            vectors,
            layer,
            dims: [n_head, hd, q_out, kv_out, pos0],
            gated,
            eps,
            table,
        };
        if let Some((_, e)) =
            crate::engine::chart::ffn::attn_heads(k, stride, &mut prep, true, &one)
        {
            return Err(e);
        }
        // Stage 2: append in token order.
        let mut last = Vec::with_capacity(k);
        let mut err: Option<StepError> = None;
        let mut c = 0usize;
        while c < k && err.is_none() {
            let base = c * stride;
            let key = &prep[base + q_out..base + q_out + kv_out];
            let value = &prep[base + q_out + kv_out..base + q_out + 2 * kv_out];
            match kv.append(slot, key, value) {
                Ok(appended) => last.push(appended),
                Err(e) => err = Some(StepError::Kv(e)),
            }
            c += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        // Stage 3: per (column, head) attention and gate. With `APEIRON_EXACT_SCAN=1` the read is
        // the fused `(m, den, num)` fold; the flag is resolved here, on the calling thread, so the
        // pool workers run one path for the whole dispatch.
        //
        // Pool task `t = n·n_head + h` owns fold position `n`, whose key column is `fold_col(n, k)`.
        // A contiguous range of the speed-weighted split then alternates cheap and expensive columns
        // instead of taking a monotone run of the cheapest. Each `(c, h)` reads only `prep`, the
        // read-only KV cache and `last[c]`, so the permutation leaves every word bit-identical; the
        // gather after the dispatch restores the tip's block order.
        let fused = exact_scan_enabled();
        let mut att = vec![0i64; k * q_out];
        let kv_ro: &KvState = kv;
        // One status slot per task, written only by that task; the slots are reduced after the
        // dispatch (`first_by_key`), so no shared error cell crosses the pool boundary.
        let tasks = att.len().div_ceil(hd.max(1));
        let mut status: Vec<Option<StepError>> = Vec::with_capacity(tasks);
        let mut s = 0usize;
        while s < tasks {
            status.push(None);
            s += 1;
        }
        let dims = [n_head, hd, kv_out, q_out, stride, k];
        let ctx = AttnHeadCtx {
            kv: kv_ro,
            layer: slot,
            flags: [fused, gated],
            prep: &prep,
            last: &last,
            dims,
        };
        let dispatched: Result<(), ()> =
            crate::engine::chart::pool::for_chunks2(&mut att, hd, &mut status, 1, &ctx);
        if dispatched.is_err() {
            return Err(StepError::BadConfig(String::from("attention dispatch")));
        }
        if let Some(e) = first_by_key(&mut status, k, n_head) {
            return Err(e);
        }
        // Block `n` holds column `fold_col(n, k)`; gather it back to index `c` with the same `k`
        // `q_out`-wide contiguous copies the tip's `chunks(q_out).map(to_vec)` performed.
        let mut atts: Vec<Vec<i64>> = Vec::with_capacity(k);
        let mut j = 0usize;
        while j < k {
            atts.push(Vec::new());
            j += 1;
        }
        for n in 0..k {
            atts[fold_col(n, k)] = att[n * q_out..(n + 1) * q_out].to_vec();
        }
        Ok(atts)
    }

    /// One affine-mixer layer over the columns: the four input projections as one group (weights
    /// read once), then per column in token order the convolution and the per-head updates
    /// ([`crate::engine::mixer::delta_heads`]), then the output projection. With `snapshot`, the
    /// layer's state after column 0 is copied into it, in place, before column 1 runs (the
    /// speculative rollback point).
    ///
    /// # Complexity
    ///
    /// The projections' multiply-adds over `k` columns plus `k` times
    /// [`crate::engine::mixer::delta_heads`]'s `O(n_v_heads · head_k · head_v)`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer index, layer, state, columns, schedule, kernel, dispatch and snapshot"
    )]
    pub(super) fn affine_cols(
        &self,
        li: Option<&LayerIndex>,
        layer: usize,
        st: &mut AffineLayer,
        xs: &[Vec<i64>],
        sched: Sched,
        mode: KernelMode,
        parallel: bool,
        snapshot: Option<&mut AffineLayer>,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let cfg = &self.config;
        let geom = cfg
            .affine
            .ok_or(StepError::BadConfig(String::from("affine geometry")))?;
        let key_dim = geom.n_k_heads * geom.head_k;
        let val_dim = geom.n_v_heads * geom.head_v;
        let channels = 2 * key_dim + val_dim;
        let outs = [channels, val_dim, geom.n_v_heads, geom.n_v_heads];
        let n_embd = cfg.n_embd;
        let k = xs.len();
        let n_in_out: usize = outs[0] + outs[1] + outs[2] + outs[3];
        let t_in = trace::span(
            Phase::AffineProjIn,
            k,
            trace::q8_bytes(n_in_out, n_embd),
            trace::macs(n_in_out, n_embd, k),
        );
        let proj = self.group_cols(li, layer, &[7, 8, 9, 10], &outs, n_embd, xs, sched, mode)?;
        drop(t_in);
        let Some(Some(params)) = self.decode_vectors().affine.get(layer) else {
            return Err(StepError::MissingTensor(TensorKey::blk(
                layer,
                PART_SSM_CONV1D,
            )));
        };
        let eps = self.eps_fixed()?;
        // The convolutions run first, in token order (each shifts the window); the rollback
        // point's conv half is the window after column 0.
        let t_conv = trace::span(Phase::AffineConv, k, 0, 0);
        let snapshot = conv_snapshot(snapshot, k, &*st, &proj[0][0]);
        let convs = crate::engine::mixer::conv_silu_cols_vecs(
            st,
            &params.conv,
            geom.conv_kernel,
            &proj[0],
        )?;
        drop(t_conv);
        // Then one pooled pass carries every head through all columns while its state is hot; each
        // head's rollback copy is taken inside its own task.
        let t_heads = trace::span(Phase::AffineHeads, k, 0, 0);
        let os = crate::engine::mixer::delta_heads_cols_vecs(
            st, &geom, params, &convs, &proj[1], &proj[2], &proj[3], eps, parallel, snapshot,
        )?;
        drop(t_heads);
        let _out = trace::span(
            Phase::AffineProjOut,
            k,
            trace::q8_bytes(n_embd, val_dim),
            trace::macs(n_embd, val_dim, k),
        );
        self.proj_cols(li, layer, 11, n_embd, val_dim, &os, sched, mode, None)
    }

    /// The `SwiGLU` FFN over the columns: the single-column path for one column (its pooled
    /// grouped kernel and scratch); for more, gate and up as `k`-column projections, one
    /// [`swiglu_cols`] dispatch over all `k·n_ff` elements, and down as a `k`-column projection.
    pub(super) fn ffn_cols(
        &self,
        li: Option<&LayerIndex>,
        layer: usize,
        xs: &[Vec<i64>],
        sched: Sched,
        mode: KernelMode,
        scratch: &mut crate::engine::chart::ffn::FfnScratch,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let (n_ff, n_embd) = (self.config.n_ff, self.config.n_embd);
        let _ffn = trace::span(
            Phase::Ffn,
            xs.len(),
            3 * trace::q8_bytes(n_ff, n_embd),
            3 * trace::macs(n_ff, n_embd, xs.len()),
        );
        if xs.len() == 1 || li.is_none() {
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                out.push(match li {
                    Some(li) => self.ffn_q8act_at(li, layer, n_ff, n_embd, x, scratch)?,
                    None => self.ffn_single_sched(layer, x, sched, mode)?,
                });
            }
            return Ok(out);
        }
        let qs = quant_cols(xs)?;
        // One pooled `[W_gate; W_up]` GEMM with the `SwiGLU` epilogue when both weights are `Q8_0`
        // and the fused kernel applies; `None` keeps the separate gate/up/`SwiGLU` path, which is
        // also the refusal owner for any gate/up value the fused kernel cannot produce exactly.
        let hs = match self.ffn_fused_swiglu(li, layer, n_ff, n_embd, &qs) {
            Some(Ok(hs)) => hs,
            Some(Err(e)) => return Err(StepError::Ffn(e)),
            None => {
                let gate =
                    self.proj_cols(li, layer, 4, n_ff, n_embd, xs, sched, mode, Some(&qs))?;
                let up = self.proj_cols(li, layer, 5, n_ff, n_embd, xs, sched, mode, Some(&qs))?;
                let mut hs: Vec<Vec<i64>> = Vec::with_capacity(xs.len());
                let mut c = 0usize;
                while c < xs.len() {
                    hs.push(vec![0i64; n_ff]);
                    c += 1;
                }
                swiglu_cols(&gate, &up, &mut hs).map_err(StepError::Ffn)?;
                hs
            }
        };
        self.proj_cols(li, layer, 6, n_embd, n_ff, &hs, sched, mode, None)
    }

    /// The fused `[W_gate; W_up]` `Q8_0` GEMM with the `SwiGLU` epilogue, or `None` when the input
    /// is not eligible: a `Hadamard` fold, a missing or non-`Q8_0` tensor, or the kernel's own
    /// shape/host gate. A setup failure returns `None` rather than an error so the separate path
    /// owns the refusal, preserving its gate-before-up-before-`SwiGLU` order.
    ///
    /// # Complexity
    ///
    /// `O(n_ff·n_in·k)` multiply-adds in one pool dispatch plus the two resident payload resolves;
    /// the gate/up matrices are not built.
    fn ffn_fused_swiglu(
        &self,
        li: Option<&LayerIndex>,
        layer: usize,
        n_ff: usize,
        n_embd: usize,
        qs: &[QuantAct],
    ) -> Option<Result<Vec<Vec<i64>>, crate::engine::FfnError>> {
        let li = li?;
        if self.hadamard().is_some() {
            return None;
        }
        let (Ok(gidx), Ok(uidx)) = (li.at(4, layer), li.at(5, layer)) else {
            return None;
        };
        let (Ok((gt, gat)), Ok((ut, uat))) = (
            self.payload_at(gidx, n_ff, n_embd, Sched::Resident),
            self.payload_at(uidx, n_ff, n_embd, Sched::Resident),
        ) else {
            return None;
        };
        if gt != quant::GGML_TYPE_Q8_0 || ut != quant::GGML_TYPE_Q8_0 {
            return None;
        }
        let file = self.resident();
        gemm_q8_0_swiglu(
            &file[gat.off..gat.off + gat.len],
            &file[uat.off..uat.off + uat.len],
            n_ff,
            n_embd,
            qs,
        )
    }

    /// The name-resolved FFN of one column (the exact path and the test oracle schedule).
    fn ffn_single_sched(
        &self,
        layer: usize,
        x: &[i64],
        sched: Sched,
        mode: KernelMode,
    ) -> Result<Vec<i64>, StepError> {
        let (n_ff, n_embd) = (self.config.n_ff, self.config.n_embd);
        let (gi, gt, gat) = self.tensor_payload(super::part_key(layer, 4), n_ff, n_embd, sched)?;
        let (ui, ut, uat) = self.tensor_payload(super::part_key(layer, 5), n_ff, n_embd, sched)?;
        let (di, dt, dat) = self.tensor_payload(super::part_key(layer, 6), n_embd, n_ff, sched)?;
        match mode {
            KernelMode::Exact => {
                let file = self.resident();
                ffn_fixed(
                    ProjWeights {
                        ggml_type: gt,
                        raw: &file[gat.off..gat.off + gat.len],
                    },
                    ProjWeights {
                        ggml_type: ut,
                        raw: &file[uat.off..uat.off + uat.len],
                    },
                    ProjWeights {
                        ggml_type: dt,
                        raw: &file[dat.off..dat.off + dat.len],
                    },
                    n_ff,
                    x,
                )
            }
            KernelMode::Q8Act => {
                self.ffn_q8act_step([gi, ui, di], [(gt, gat), (ut, uat), (dt, dat)], n_ff, x)
            }
        }
        .map_err(StepError::Ffn)
    }
}

/// The stage-3 fold order: pool task `n` owns key column `fold(n)`.
///
/// `n -> c` is a bijection on `0..k`: the even `n = 2j` hit the lower columns `c = j` and the odd
/// `n = 2j + 1` the upper ones `c = k - 1 - j`. For `k = 2m` the images are `0..m-1` and `m..2m-1`,
/// for `k = 2m + 1` they are `0..m` and `m+1..2m`; disjoint and covering. A contiguous range of the
/// speed-weighted pool split therefore alternates cheap and expensive columns: the cost of the
/// adjacent pair `(2j, 2j+1)` is `(pos₀ + j + 1) + (pos₀ + k - 1 - j + 1) = 2·pos₀ + k + 1`,
/// independent of `j`, so every contiguous range averages the same key work. Each `(c, h)` result
/// is a pure function of the read-only inputs, so the permutation leaves all values bit-identical.
///
/// # Complexity
///
/// `O(1)`.
fn fold_col(n: usize, k: usize) -> usize {
    if n.is_multiple_of(2) {
        n / 2
    } else {
        k - 1 - n / 2
    }
}

/// Stage-3 pool task `t = n·n_head + h`: head `h` of key column `c = fold_col(n, k)`, written into
/// `dst` (`hd` words), gated when `flags[1]`. `flags = [fused, gated]`, `dims = [n_head, hd,
/// kv_out, q_out, stride, k]`. A refusal goes to `status[0]` and the task returns `Ok`, so the pool
/// never sees an error and every task runs once.
///
/// # Complexity
///
/// Time `O(hd · (pos + 1))` words of KV read for the head; no allocation.
#[allow(
    clippy::too_many_arguments,
    clippy::unnecessary_wraps,
    reason = "flat scalar parameters keep the pool closure one call (Aeneas R2) with no nested region; the pool closure type is Result"
)]
fn attn_head_task(
    kv: &KvState,
    layer: usize,
    flags: [bool; 2],
    prep: &[i64],
    last: &[usize],
    dims: [usize; 6],
    task: usize,
    dst: &mut [i64],
    status: &mut [Option<StepError>],
) -> Result<(), ()> {
    let [n_head, hd, kv_out, q_out, stride, cols] = dims;
    let (fold_pos, head) = (task / n_head.max(1), task % n_head.max(1));
    let col = fold_col(fold_pos, cols);
    let base = col * stride;
    let query = &prep[base + head * hd..base + (head + 1) * hd];
    let read = if flags[0] {
        kv.attn_head_fused(layer, head, query, last[col], EXACT_SCAN_TILE, dst)
            .map_err(StepError::Kv)
    } else {
        kv.attn_head(layer, head, query, last[col], dst)
            .map_err(StepError::Kv)
    };
    let done = match read {
        Ok(()) if flags[1] => {
            let g0 = base + 2 * kv_out + q_out + head * hd;
            apply_sigmoid_gate(dst, &prep[g0..g0 + hd])
        }
        Ok(()) => Ok(()),
        Err(e) => Err(e),
    };
    if let Err(e) = done {
        status[0] = Some(e);
    }
    Ok(())
}

/// The shared borrows of one attention-head dispatch; `run` is the single call to [`attn_head_task`].
struct AttnHeadCtx<'a> {
    kv: &'a KvState,
    layer: usize,
    flags: [bool; 2],
    prep: &'a [i64],
    last: &'a [usize],
    dims: [usize; 6],
}

impl Task2<i64, Option<StepError>, ()> for AttnHeadCtx<'_> {
    fn run(&self, t: usize, dst: &mut [i64], st: &mut [Option<StepError>]) -> Result<(), ()> {
        attn_head_task(
            self.kv, self.layer, self.flags, self.prep, self.last, self.dims, t, dst, st,
        )
    }
}

/// The shared borrows of one query-head attention read; `run` is the single call to [`query_head_attn`].
struct QueryHeadCtx<'a> {
    kv: &'a KvState,
    slot: usize,
    query: &'a [i64],
    hd: usize,
    appended: usize,
}

impl HeadTask<i64, AttnError> for QueryHeadCtx<'_> {
    fn run(&self, h: usize, dst: &mut [i64]) -> Result<(), AttnError> {
        query_head_attn(
            self.kv,
            self.slot,
            self.query,
            self.hd,
            self.appended,
            h,
            dst,
        )
    }
}

/// One head of the query read: the `hd`-wide slice of `query` for head `h`, attended over `appended` positions.
fn query_head_attn(
    kv: &KvState,
    slot: usize,
    query: &[i64],
    hd: usize,
    appended: usize,
    h: usize,
    dst: &mut [i64],
) -> Result<(), AttnError> {
    kv.attn_head(slot, h, &query[h * hd..(h + 1) * hd], appended, dst)
}

/// The shared borrows of one column's stage-1 preparation; `run` is the single call to [`prep_column`].
struct PrepColCtx<'a> {
    qkv: &'a [Vec<Vec<i64>>],
    vectors: &'a DecodeVectors,
    layer: usize,
    /// `[n_head, hd, q_out, kv_out, pos0]`.
    dims: [usize; 5],
    gated: bool,
    eps: i64,
    table: &'a crate::engine::rope::RopeTable,
}

impl HeadTask<i64, StepError> for PrepColCtx<'_> {
    fn run(&self, c: usize, dst: &mut [i64]) -> Result<(), StepError> {
        prep_column(
            c,
            dst,
            self.qkv,
            self.vectors,
            self.layer,
            self.dims,
            self.gated,
            self.eps,
            self.table,
        )
    }
}

/// Stage 1 for column `c`: the biases, the optional gate split, the QK-norm and rotation, written to
/// `dst` as `[query | key | value | gate]`. The body is the former pool closure, unchanged.
#[allow(
    clippy::too_many_arguments,
    clippy::index_refutable_slice,
    reason = "the column's inputs are the call's full state, the geometry is packed in dims; the three biases are read by constant index because an array pattern does not extract"
)]
fn prep_column(
    c: usize,
    dst: &mut [i64],
    qkv: &[Vec<Vec<i64>>],
    vectors: &DecodeVectors,
    layer: usize,
    dims: [usize; 5],
    gated: bool,
    eps: i64,
    table: &crate::engine::rope::RopeTable,
) -> Result<(), StepError> {
    let [n_head, hd, q_out, kv_out, pos0] = dims;
    let (mut query, mut key, mut value) = (qkv[0][c].clone(), qkv[1][c].clone(), qkv[2][c].clone());
    if let Some(bs) = vectors.biases_of(layer) {
        if let Some(bias) = opt_slice(&bs[0]) {
            add_assign_fixed(&mut query, bias)?;
        }
    }
    if let Some(bs) = vectors.biases_of(layer) {
        if let Some(bias) = opt_slice(&bs[1]) {
            add_assign_fixed(&mut key, bias)?;
        }
    }
    if let Some(bs) = vectors.biases_of(layer) {
        if let Some(bias) = opt_slice(&bs[2]) {
            add_assign_fixed(&mut value, bias)?;
        }
    }
    let gate = if gated {
        let (q, g) = split_query_gate(&query, n_head, hd)?;
        query = q;
        g
    } else {
        Vec::new()
    };
    let pos = u32::try_from(pos0 + c).map_err(|_| StepError::LengthOverflow)?;
    qk_norm_and_rope(
        vectors, layer, table, &mut query, &mut key, hd, eps, pos, false,
    )?;
    let (dq, rest) = dst.split_at_mut(q_out);
    let (dk, rest) = rest.split_at_mut(kv_out);
    let (dv, dg) = rest.split_at_mut(kv_out);
    if query.len() != q_out || key.len() != kv_out || value.len() != kv_out {
        return Err(StepError::BadConfig(String::from("attention block widths")));
    }
    dq.copy_from_slice(&query);
    dk.copy_from_slice(&key);
    dv.copy_from_slice(&value);
    if gated {
        dg.copy_from_slice(&gate);
    }
    Ok(())
}

/// The shared borrows of one column's RMS norm; `run` is the single call to [`norm_col`].
struct NormColCtx<'a> {
    gain: &'a [i64],
    xs: &'a [Vec<i64>],
    eps: i64,
}

impl HeadTask<i64, StepError> for NormColCtx<'_> {
    fn run(&self, c: usize, dst: &mut [i64]) -> Result<(), StepError> {
        norm_col(self.gain, self.xs, self.eps, c, dst)
    }
}

/// One column of [`norm_cols`]: `rms_with_gain_slice` of `xs[c]` copied into `dst`.
fn norm_col(
    gain: &[i64],
    xs: &[Vec<i64>],
    eps: i64,
    c: usize,
    dst: &mut [i64],
) -> Result<(), StepError> {
    dst.copy_from_slice(&rms_with_gain_slice(gain, &xs[c], eps)?);
    Ok(())
}

/// The error of the task with the lowest refusal key, taken out of `status`. The key of task `t` is
/// the pre-fold index `fold_col(t / n_head, k)·n_head + t % n_head`, the order the tip dispatch
/// reported refusals in; keys are distinct because `fold_col` is a bijection.
///
/// # Complexity
///
/// Time `O(status.len())` key comparisons in one pass; space `O(1)` words (best index and key).
fn first_by_key(status: &mut [Option<StepError>], k: usize, n_head: usize) -> Option<StepError> {
    match lowest_key(status, k, n_head) {
        Some(best) => status[best].take(),
        None => None,
    }
}

/// Index of the failing task with the lowest refusal key, or `None` when no slot holds an error.
///
/// # Complexity
///
/// Time `O(status.len())` key comparisons in one pass; space `O(1)` words (best index and key).
fn lowest_key(status: &[Option<StepError>], k: usize, n_head: usize) -> Option<usize> {
    let nh = n_head.max(1);
    let n = status.len();
    let mut best = n;
    let mut best_key = 0usize;
    let mut t = 0usize;
    while t < n {
        if status[t].is_some() {
            let key = fold_col(t / nh, k) * nh + t % nh;
            if best == n || key < best_key {
                best = t;
                best_key = key;
            }
        }
        t += 1;
    }
    if best < n {
        Some(best)
    } else {
        None
    }
}

/// The rollback point's conv half: with a snapshot slot and more than one column, `snap` receives
/// the window of `st` after column 0 (`x0`); the slot is returned shared for the heads pass.
/// O(window) words copied by `snapshot_conv_after`; nothing else.
fn conv_snapshot<'a>(
    snapshot: Option<&'a mut AffineLayer>,
    k: usize,
    st: &AffineLayer,
    x0: &[i64],
) -> Option<&'a AffineLayer> {
    match snapshot {
        Some(snap) => {
            if k > 1 {
                snap.snapshot_conv_after(st, x0);
            }
            Some(&*snap)
        }
        None => None,
    }
}

/// `layer`'s `which`-th norm gain (`0` attention, `1` FFN) of the decode vectors, or the
/// [`StepError::MissingTensor`] for `part`. O(1).
fn norm_gain(
    vectors: &DecodeVectors,
    layer: usize,
    which: usize,
    part: u8,
) -> Result<&[i64], StepError> {
    match vectors.norms_of(layer) {
        Some(n) => match &n[which] {
            Some(g) => Ok(g.as_slice()),
            None => Err(StepError::MissingTensor(TensorKey::blk(layer, part))),
        },
        None => Err(StepError::MissingTensor(TensorKey::blk(layer, part))),
    }
}

/// `hidden[i] += delta[i]` for the columns both hold, stopping at the first refusal.
///
/// # Complexity
///
/// `O(k·n)` fixed-point additions, `O(1)` space: one loop over the `k` columns, one `n`-wide
/// [`add_assign_fixed`] each.
fn add_cols(hidden: &mut [Vec<i64>], delta: &[Vec<i64>]) -> Result<(), StepError> {
    let mut err: Option<StepError> = None;
    let mut i = 0usize;
    while i < hidden.len() && i < delta.len() && err.is_none() {
        if let Err(e) = add_assign_fixed(&mut hidden[i], &delta[i]) {
            err = Some(e);
        }
        i += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Each column's `RMSNorm` with gain: serial for one column, one pool task per column for more
/// (each column's norm is column-local, so the split is bit-identical).
///
/// # Complexity
///
/// `O(k·n)` over the pool.
fn norm_cols(gain: &[i64], xs: &[Vec<i64>], eps: i64) -> Result<Vec<Vec<i64>>, StepError> {
    let k = xs.len();
    let n = if k == 0 { 0 } else { xs[0].len() };
    let mut ragged = false;
    let mut c = 0usize;
    while c < k {
        if xs[c].len() != n {
            ragged = true;
        }
        c += 1;
    }
    if k < 2 || ragged {
        let mut rows = Vec::with_capacity(k);
        let mut err: Option<StepError> = None;
        let mut c = 0usize;
        while c < k && err.is_none() {
            match rms_with_gain_slice(gain, &xs[c], eps) {
                Ok(v) => rows.push(v),
                Err(e) => err = Some(e),
            }
            c += 1;
        }
        return match err {
            Some(e) => Err(e),
            None => Ok(rows),
        };
    }
    let mut out = vec![0i64; k * n];
    let one = NormColCtx { gain, xs, eps };
    if let Some((_, e)) = crate::engine::chart::ffn::attn_heads(k, n, &mut out, true, &one) {
        return Err(e);
    }
    let mut cols: Vec<Vec<i64>> = Vec::with_capacity(k);
    let mut c = 0usize;
    while c < k {
        cols.push(out[c * n..(c + 1) * n].to_vec());
        c += 1;
    }
    Ok(cols)
}

/// The pooled int8 quantization of every column ([`quant_acts_pooled`], one dispatch over the `k`
/// columns), shared by every projection that reads the same columns.
///
/// # Complexity
///
/// `O(k·n)` over the pool.
fn quant_cols(xs: &[Vec<i64>]) -> Result<Vec<QuantAct>, StepError> {
    quant_acts_pooled(xs).ok_or(StepError::Matvec(MatvecError::Dequant(
        crate::engine::EngineError::FixedOverflow,
    )))
}

#[cfg(test)]
mod fold_order {
    use super::{first_by_key, fold_col, StepError};

    /// Two failing tasks whose task order and key order differ report the key-order error. With
    /// `k = 3`, `n_head = 1`: task 1 has column `fold_col(1, 3) = 2` (key 2) and task 2 has column
    /// `fold_col(2, 3) = 1` (key 1), so the lowest key is task 2 although task 1 comes first.
    /// Reducing in task order returns the other error and fails this test.
    #[test]
    fn first_error_is_by_key_not_by_task() {
        for n_head in [1usize, 2] {
            let k = 3usize;
            let mut status: Vec<Option<StepError>> = (0..k * n_head).map(|_| None).collect();
            let (a, b) = (n_head, 2 * n_head);
            assert!(
                fold_col(a / n_head, k) > fold_col(b / n_head, k),
                "task order and key order must differ"
            );
            status[a] = Some(StepError::LengthOverflow);
            status[b] = Some(StepError::BadConfig(String::from("second task, lower key")));
            let got = first_by_key(&mut status, k, n_head);
            assert_eq!(
                got,
                Some(StepError::BadConfig(String::from("second task, lower key")))
            );
            assert!(status[b].is_none(), "the winning slot is taken");
            assert!(status[a].is_some(), "the losing slot is left");
        }
        assert_eq!(first_by_key(&mut [], 3, 1), None);
    }

    /// `fold_col` is a bijection on `0..k` and each adjacent fold pair is one cheap and one
    /// expensive column: their key counts sum to `k + 1` (the exact balance the stage-3 dispatch
    /// relies on). Mutating the map to `n / 2` duplicates columns and fails this test.
    #[test]
    fn fold_col_is_a_balanced_bijection() {
        for k in 1..=9usize {
            let mut seen = vec![false; k];
            for n in 0..k {
                let c = fold_col(n, k);
                assert!(c < k, "k={k} n={n}: column {c} out of range");
                assert!(!seen[c], "k={k} n={n}: column {c} twice");
                seen[c] = true;
            }
            // cost(c) = pos₀ + c + 1 keys; the pos₀ cancels in the pair sum.
            for j in 0..k / 2 {
                let cost = fold_col(2 * j, k) + fold_col(2 * j + 1, k) + 2;
                assert_eq!(cost, k + 1, "k={k} pair {j}: key count");
            }
        }
    }
}
