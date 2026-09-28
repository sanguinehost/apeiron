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
    add_assign_fixed, affine_slots, apply_sigmoid_gate, qk_norm_and_rope, rms_with_gain_slice,
    softmax_slots, split_query_gate, LayerIndex, Sched, StepError, LAYER_PARTS,
};
use crate::engine::chart::ffn::{quant_acts_pooled, swiglu_cols};
use crate::engine::chart::QuantAct;
use crate::engine::kern::gemm::gemm_q8_0_swiglu;
use crate::engine::kern::matmat::matmat_q8act;
use crate::engine::mixer::{AffineLayer, AffineState};
use crate::engine::trace::{self, Phase};
use crate::engine::{ffn_fixed, Engine, KernelMode, KvState, MatvecError, MixerKind, ProjWeights};
use crate::format::quant;

impl Engine {
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
    pub(crate) fn compose_cols(
        &self,
        kv: &mut KvState,
        mut affine: Option<&mut AffineState>,
        tokens: &[u32],
        sched: Sched,
        mode: KernelMode,
        parallel_heads: bool,
        mut snapshot: Option<&mut [AffineLayer]>,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let cfg = &self.config;
        if tokens.is_empty() {
            return Err(StepError::BadConfig("empty token block"));
        }
        let k = tokens.len();
        let _pass = trace::span(Phase::Pass, k, 0, 0);
        let slots = softmax_slots(&cfg.mixers);
        let aslots = affine_slots(&cfg.mixers);
        let n_softmax = cfg
            .mixers
            .iter()
            .filter(|m| **m == MixerKind::Softmax)
            .count();
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
        for &t in tokens {
            hidden.push(
                self.embedding_sched(t, sched)
                    .map_err(StepError::Embedding)?,
            );
        }
        drop(t_embed);
        // The resident `Q8Act` composition resolves the per-layer directory entries once per
        // pass (`LayerIndex`); every other schedule keeps the name-based lookups.
        let layers = if matches!(sched, Sched::Resident) && mode == KernelMode::Q8Act {
            Some(self.resolve_layers())
        } else {
            None
        };
        let mut ffn_scratch = crate::engine::chart::ffn::FfnScratch::default();

        let mut layer = 0usize;
        while layer < cfg.n_layer {
            let li = layers.as_ref().map(|v| &v[layer]);
            let attn_gain = vectors
                .norms_of(layer)
                .and_then(|n| n[0].as_deref())
                .ok_or_else(|| StepError::MissingTensor(format!("blk.{layer}.attn_norm.weight")))?;
            let t_norm = trace::span(Phase::Norm, k, 0, 0);
            let normed = norm_cols(attn_gain, &hidden, eps)?;
            drop(t_norm);
            // The mixer primitive: the softmax list carrier (`slot` is the layer's place in the
            // `KvState`) or the affine carrier (`aslot` in the `AffineState`).
            let mixed = match (slots[layer], aslots[layer]) {
                (Some(slot), _) => {
                    self.softmax_cols(li, layer, slot, kv, &normed, sched, mode, parallel_heads)?
                }
                (None, Some(aslot)) => {
                    let st = affine
                        .as_deref_mut()
                        .and_then(|st| st.layer_mut(aslot))
                        .ok_or(StepError::BadConfig("affine state for an affine layer"))?;
                    self.affine_cols(
                        li,
                        layer,
                        st,
                        &normed,
                        sched,
                        mode,
                        parallel_heads,
                        snapshot.as_deref_mut().and_then(|s| s.get_mut(aslot)),
                    )?
                }
                (None, None) => return Err(StepError::BadConfig("layer without a mixer")),
            };
            let t_res = trace::span(Phase::Residual, k, 0, 0);
            for (h, m) in hidden.iter_mut().zip(&mixed) {
                add_assign_fixed(h, m)?;
            }
            drop(t_res);
            let ffn_gain = vectors
                .norms_of(layer)
                .and_then(|n| n[1].as_deref())
                .ok_or_else(|| StepError::MissingTensor(format!("blk.{layer}.ffn_norm.weight")))?;
            let t_norm = trace::span(Phase::Norm, k, 0, 0);
            let h2 = norm_cols(ffn_gain, &hidden, eps)?;
            drop(t_norm);
            let ffn_out = self.ffn_cols(li, layer, &h2, sched, mode, &mut ffn_scratch)?;
            let t_res = trace::span(Phase::Residual, k, 0, 0);
            for (h, f) in hidden.iter_mut().zip(&ffn_out) {
                add_assign_fixed(h, f)?;
            }
            drop(t_res);
            layer += 1;
        }

        let out_gain = vectors
            .output_norm
            .as_deref()
            .ok_or_else(|| StepError::MissingTensor("output_norm.weight".to_string()))?;
        let _norm = trace::span(Phase::Norm, k, 0, 0);
        norm_cols(out_gain, &hidden, eps)
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
        let name = || format!("blk.{layer}.{}.weight", LAYER_PARTS[part]);
        let Some(li) = li else {
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                out.push(self.step_proj_sched(&name(), n_out, n_in, x, sched, mode)?);
            }
            return Ok(out);
        };
        let idx = li.at(part, layer)?;
        let budget = crate::engine::par::row_chunk_budget(n_out);
        if xs.len() > 1 && self.hadamard().is_none() {
            let (ggml_type, raw) = self.payload_at(idx, n_out, n_in, Sched::Resident)?;
            if ggml_type == quant::GGML_TYPE_Q8_0 {
                let owned;
                let qs = if let Some(q) = pre {
                    q
                } else {
                    owned = quant_cols(xs)?;
                    &owned
                };
                return matmat_q8act(ggml_type, raw.as_ref(), n_out, n_in, qs)
                    .map_err(StepError::Matvec);
            }
        }
        let mut out = Vec::with_capacity(xs.len());
        for x in xs {
            out.push(self.proj_q8act_at(idx, n_out, n_in, x, budget)?);
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
                let got = self.group_q8act_at(
                    li,
                    layer,
                    parts,
                    outs,
                    n_in,
                    &xs[0],
                    crate::engine::par::row_chunk_budget(outs[0]),
                )?;
                return Ok(got.into_iter().map(|v| vec![v]).collect());
            }
        }
        // One quantization per column serves every part of the group.
        let qs = if li.is_some() && xs.len() > 1 {
            Some(quant_cols(xs)?)
        } else {
            None
        };
        let mut out = Vec::with_capacity(parts.len());
        for (i, &p) in parts.iter().enumerate() {
            out.push(self.proj_cols(
                li,
                layer,
                p,
                outs[i],
                n_in,
                xs,
                sched,
                mode,
                qs.as_deref(),
            )?);
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
        let rope = self.rope_table().map_err(StepError::Rope)?;
        let vectors = self.decode_vectors();
        let k = xs.len();
        let t_in = trace::span(
            Phase::AttnProjIn,
            k,
            trace::q8_bytes(q_proj + 2 * kv_out, n_embd),
            trace::macs(q_proj + 2 * kv_out, n_embd, k),
        );
        let qkv = if xs.len() == 1 && li.is_none() {
            let [q, k, v] =
                self.step_qkv_sched(layer, [q_proj, kv_out, kv_out], n_embd, &xs[0], sched, mode)?;
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
            let atts = self.softmax_core_block(layer, slot, kv, &qkv, eps, &rope)?;
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
        for c in 0..k {
            let (mut query, mut key, mut value) =
                (qkv[0][c].clone(), qkv[1][c].clone(), qkv[2][c].clone());
            for (v, b) in [(&mut query, 0usize), (&mut key, 1), (&mut value, 2)] {
                if let Some(bias) = vectors.biases_of(layer).and_then(|bs| bs[b].as_deref()) {
                    add_assign_fixed(v, bias)?;
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
                &rope,
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
            let head = |h: usize, dst: &mut [i64]| {
                kv_ro.attn_head(slot, h, &query[h * hd..(h + 1) * hd], appended, dst)
            };
            if let Some((_, e)) =
                crate::engine::chart::ffn::attn_heads(n_head, hd, &mut att, parallel_heads, head)
            {
                return Err(StepError::Kv(e));
            }
            if let Some(g) = gate {
                apply_sigmoid_gate(&mut att, &g)?;
            }
            atts.push(att);
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

    /// The attention core of a block of `k > 1` columns in three pooled stages, each column equal
    /// to the per-column loop in [`Engine::softmax_cols`]:
    /// 1. per column (one task each): bias, gate split, q/k norm and `RoPE` at position
    ///    `pos₀ + c`;
    /// 2. in token order: append each column's key and value;
    /// 3. per (column, head) (`k·n_head` tasks): attention over positions `0..=pos₀ + c`, then
    ///    that head's slice of the sigmoid output gate.
    ///
    /// Column `c` sees exactly the keys the per-column loop had appended when it attended, so the
    /// block is causal and bit-identical (`qwen35_block_prefill_equals_single_steps`). A refusal
    /// leaves the cache unspecified.
    ///
    /// # Complexity
    ///
    /// `O(k·n_head·(pos₀ + k)·head_dim)` for the attention over `k·n_head` pool tasks, plus
    /// `O(k·(q_proj + 2·kv_out))` for stages 1–2; three dispatches instead of `k`.
    #[allow(clippy::too_many_lines, reason = "the three stages read in order")]
    fn softmax_core_block(
        &self,
        layer: usize,
        slot: usize,
        kv: &mut KvState,
        qkv: &[Vec<Vec<i64>>],
        eps: i64,
        rope: &crate::engine::rope::RopeTable,
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
        let one = |c: usize, dst: &mut [i64]| -> Result<(), StepError> {
            let (mut query, mut key, mut value) =
                (qkv[0][c].clone(), qkv[1][c].clone(), qkv[2][c].clone());
            for (v, b) in [(&mut query, 0usize), (&mut key, 1), (&mut value, 2)] {
                if let Some(bias) = vectors.biases_of(layer).and_then(|bs| bs[b].as_deref()) {
                    add_assign_fixed(v, bias)?;
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
                vectors, layer, rope, &mut query, &mut key, hd, eps, pos, false,
            )?;
            let (dq, rest) = dst.split_at_mut(q_out);
            let (dk, rest) = rest.split_at_mut(kv_out);
            let (dv, dg) = rest.split_at_mut(kv_out);
            if query.len() != q_out || key.len() != kv_out || value.len() != kv_out {
                return Err(StepError::BadConfig("attention block widths"));
            }
            dq.copy_from_slice(&query);
            dk.copy_from_slice(&key);
            dv.copy_from_slice(&value);
            if gated {
                dg.copy_from_slice(&gate);
            }
            Ok(())
        };
        if let Some((_, e)) = crate::engine::chart::ffn::attn_heads(k, stride, &mut prep, true, one)
        {
            return Err(e);
        }
        // Stage 2: append in token order.
        let mut last = Vec::with_capacity(k);
        for c in 0..k {
            let base = c * stride;
            let key = &prep[base + q_out..base + q_out + kv_out];
            let value = &prep[base + q_out + kv_out..base + q_out + 2 * kv_out];
            last.push(kv.append(slot, key, value).map_err(StepError::Kv)?);
        }
        // Stage 3: per (column, head) attention and gate.
        let mut att = vec![0i64; k * q_out];
        let kv_ro: &KvState = kv;
        let head = |t: usize, dst: &mut [i64]| -> Result<(), StepError> {
            let (c, h) = (t / n_head, t % n_head);
            let base = c * stride;
            let q = &prep[base + h * hd..base + (h + 1) * hd];
            kv_ro
                .attn_head(slot, h, q, last[c], dst)
                .map_err(StepError::Kv)?;
            if gated {
                let g0 = base + 2 * kv_out + q_out + h * hd;
                apply_sigmoid_gate(dst, &prep[g0..g0 + hd])?;
            }
            Ok(())
        };
        if let Some((_, e)) =
            crate::engine::chart::ffn::attn_heads(k * n_head, hd, &mut att, true, head)
        {
            return Err(e);
        }
        Ok(att.chunks(q_out).map(<[i64]>::to_vec).collect())
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
        clippy::needless_range_loop,
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
        let geom = cfg.affine.ok_or(StepError::BadConfig("affine geometry"))?;
        let key_dim = geom.n_k_heads * geom.head_k;
        let val_dim = geom.n_v_heads * geom.head_v;
        let channels = 2 * key_dim + val_dim;
        let outs = [channels, val_dim, geom.n_v_heads, geom.n_v_heads];
        let n_embd = cfg.n_embd;
        let k = xs.len();
        let n_in_out: usize = outs.iter().sum();
        let t_in = trace::span(
            Phase::AffineProjIn,
            k,
            trace::q8_bytes(n_in_out, n_embd),
            trace::macs(n_in_out, n_embd, k),
        );
        let proj = self.group_cols(li, layer, &[7, 8, 9, 10], &outs, n_embd, xs, sched, mode)?;
        drop(t_in);
        let params = self
            .decode_vectors()
            .affine
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| StepError::MissingTensor(format!("blk.{layer}.ssm_conv1d.weight")))?;
        let eps = self.eps_fixed()?;
        // The convolutions run first, in token order (each shifts the window); the rollback
        // point's conv half is the window after column 0.
        let mut snapshot = snapshot;
        let t_conv = trace::span(Phase::AffineConv, k, 0, 0);
        if let Some(snap) = snapshot.as_deref_mut() {
            if k > 1 {
                snap.snapshot_conv_after(st, &proj[0][0]);
            }
        }
        let inputs: Vec<&[i64]> = proj[0].iter().map(Vec::as_slice).collect();
        let convs =
            crate::engine::mixer::conv_silu_cols(st, &params.conv, geom.conv_kernel, &inputs)?;
        drop(t_conv);
        // Then one pooled pass carries every head through all columns while its state is hot; each
        // head's rollback copy is taken inside its own task.
        let t_heads = trace::span(Phase::AffineHeads, k, 0, 0);
        let cols: Vec<crate::engine::mixer::AffineCol<'_>> = (0..xs.len())
            .map(|c| {
                (
                    convs[c].as_slice(),
                    proj[1][c].as_slice(),
                    proj[2][c].as_slice(),
                    proj[3][c].as_slice(),
                )
            })
            .collect();
        let os = crate::engine::mixer::delta_heads_cols(
            st,
            &geom,
            params,
            &cols,
            eps,
            parallel,
            snapshot.as_deref(),
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
                let mut hs: Vec<Vec<i64>> = (0..xs.len()).map(|_| vec![0i64; n_ff]).collect();
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
        let (Ok((gt, graw)), Ok((ut, uraw))) = (
            self.payload_at(gidx, n_ff, n_embd, Sched::Resident),
            self.payload_at(uidx, n_ff, n_embd, Sched::Resident),
        ) else {
            return None;
        };
        if gt != quant::GGML_TYPE_Q8_0 || ut != quant::GGML_TYPE_Q8_0 {
            return None;
        }
        gemm_q8_0_swiglu(graw.as_ref(), uraw.as_ref(), n_ff, n_embd, qs)
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
        let (gt, graw) =
            self.tensor_payload(&format!("blk.{layer}.ffn_gate.weight"), n_ff, n_embd, sched)?;
        let (ut, uraw) =
            self.tensor_payload(&format!("blk.{layer}.ffn_up.weight"), n_ff, n_embd, sched)?;
        let (dt, draw) =
            self.tensor_payload(&format!("blk.{layer}.ffn_down.weight"), n_embd, n_ff, sched)?;
        let (wg, wu, wd) = (
            ProjWeights {
                ggml_type: gt,
                raw: graw.as_ref(),
            },
            ProjWeights {
                ggml_type: ut,
                raw: uraw.as_ref(),
            },
            ProjWeights {
                ggml_type: dt,
                raw: draw.as_ref(),
            },
        );
        match mode {
            KernelMode::Exact => ffn_fixed(wg, wu, wd, n_ff, x),
            KernelMode::Q8Act => self.ffn_q8act_step(layer, wg, wu, wd, n_ff, x),
        }
        .map_err(StepError::Ffn)
    }
}

/// Each column's `RMSNorm` with gain: serial for one column, one pool task per column for more
/// (each column's norm is column-local, so the split is bit-identical).
///
/// # Complexity
///
/// `O(k·n)` over the pool.
fn norm_cols(gain: &[i64], xs: &[Vec<i64>], eps: i64) -> Result<Vec<Vec<i64>>, StepError> {
    let (k, n) = (xs.len(), xs.first().map_or(0, Vec::len));
    if k < 2 || xs.iter().any(|x| x.len() != n) {
        return xs
            .iter()
            .map(|h| rms_with_gain_slice(gain, h, eps))
            .collect();
    }
    let mut out = vec![0i64; k * n];
    let one = |c: usize, dst: &mut [i64]| -> Result<(), StepError> {
        dst.copy_from_slice(&rms_with_gain_slice(gain, &xs[c], eps)?);
        Ok(())
    };
    if let Some((_, e)) = crate::engine::chart::ffn::attn_heads(k, n, &mut out, true, one) {
        return Err(e);
    }
    Ok(out.chunks(n).map(<[i64]>::to_vec).collect())
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
