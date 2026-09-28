//! Self-speculative greedy decoding with the model's own multi-token-prediction (MTP) block.
//!
//! The emitted tokens are exactly greedy decoding's, for any drafter: a draft token is kept only
//! when the verifier's own greedy token at that position equals it. Proof sketch: the lead's
//! `Nous/Model/SpeculativeRollback.lean` (SPEC-L1) and `JacobiDecode.verify_accept_prefix`.
//! Drafter quality changes speed only.
//!
//! # One round
//!
//! `x` is the last verified token, not yet run through the trunk, at position `p`; `h` is the
//! trunk's output-normed state at `p − 1`.
//! 1. **Draft.** The MTP block at position `p` reads `[enorm(embed x) | hnorm(h)]` (llama.cpp
//!    `qwen35.cpp` `graph_mtp`). Its head argmax over the first `draft_rows` vocabulary rows is the
//!    draft `d`. A narrower draft head is cheaper and only lowers acceptance.
//! 2. **Verify.** One two-column trunk pass over `[x, d]`, which reads the weights once for both
//!    columns, then the head argmax of both columns in one pass: `t₀` after `x`, `t₁` after `d`.
//! 3. **Accept** (`t₀ = d`): emit `d, t₁`. Otherwise **reject**: emit `t₀`, truncate the KV cache
//!    to `p + 1`, and restore every affine layer to its state after column 0.
//!
//! The MTP block's own cache never needs rollback: its entry at position `p` reads only the
//! verified `x` and the verified `h`.

use super::{add_assign_fixed, rms_with_gain_slice, LayerIndex, Sched, StepError};
use crate::engine::chart::ffn::{matvec_q8act_quant_chunked, quant_act_pooled};
use crate::engine::kern::matmat::matmat_q8act;
use crate::engine::mixer::AffineLayer;
use crate::engine::trace::{self, Phase};
use crate::engine::{argmax_fixed, Engine, KernelMode, KvState, MatvecError};
use crate::format::quant;

/// Counts from one speculative decode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecStats {
    /// Draft-and-verify rounds.
    pub rounds: usize,
    /// Drafts the verifier accepted.
    pub accepted: usize,
}

/// The MTP block's own key/value cache (one softmax layer).
#[derive(Debug, Clone)]
pub struct MtpState {
    kv: KvState,
}

impl Engine {
    /// An empty MTP cache of capacity `cap`.
    ///
    /// # Errors
    ///
    /// [`StepError::BadConfig`] when the model has no MTP block, and the [`KvState::new`] refusals.
    pub fn mtp_state(&self, cap: usize) -> Result<MtpState, StepError> {
        let cfg = &self.config;
        if cfg.n_nextn == 0 {
            return Err(StepError::BadConfig(
                "model has no multi-token-prediction block",
            ));
        }
        let kv =
            KvState::new(1, cfg.n_head, cfg.n_head_kv, cfg.head_dim, cap).map_err(StepError::Kv)?;
        Ok(MtpState { kv })
    }

    /// The MTP block over columns `(tokens[c], hs[c])`, each at the next MTP-cache position:
    /// `x = eh_proj([enorm(embed t) | hnorm(h)])`, then a softmax-attention block (the executor's
    /// `softmax_cols` and `ffn_cols` at layer `n_layer`), then the head norm. Returns the
    /// head-normed states.
    ///
    /// # Complexity
    ///
    /// One block of the trunk plus the `2·n_embd → n_embd` projection, per pass over `k` columns.
    fn mtp_cols(
        &self,
        li: Option<&LayerIndex>,
        mtp: &mut MtpState,
        tokens: &[u32],
        hs: &[Vec<i64>],
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let cfg = &self.config;
        let (l, n_embd) = (cfg.n_layer, cfg.n_embd);
        let _mtp = trace::span(Phase::Mtp, tokens.len(), 0, 0);
        let (sched, mode) = (Sched::Resident, KernelMode::Q8Act);
        let eps = self.eps_fixed()?;
        let vectors = self.decode_vectors();
        let nv = vectors
            .nextn
            .first()
            .ok_or(StepError::BadConfig("no MTP vectors"))?;
        let missing = |what: &str| StepError::MissingTensor(format!("blk.{l}.nextn.{what}"));
        let enorm = nv[0].as_deref().ok_or_else(|| missing("enorm.weight"))?;
        let hnorm = nv[1].as_deref().ok_or_else(|| missing("hnorm.weight"))?;
        let mut cats = Vec::with_capacity(tokens.len());
        for (&t, h) in tokens.iter().zip(hs) {
            let e = self
                .embedding_sched(t, sched)
                .map_err(StepError::Embedding)?;
            let mut cat = rms_with_gain_slice(enorm, &e, eps)?;
            cat.extend(rms_with_gain_slice(hnorm, h, eps)?);
            cats.push(cat);
        }
        let mut x = self.proj_cols(li, l, 12, n_embd, 2 * n_embd, &cats, sched, mode, None)?;
        let norms = vectors
            .norms_of(l)
            .ok_or(StepError::BadConfig("no MTP block norms"))?;
        let attn_gain = norms[0]
            .as_deref()
            .ok_or_else(|| StepError::MissingTensor(format!("blk.{l}.attn_norm.weight")))?;
        let mut normed = Vec::with_capacity(x.len());
        for v in &x {
            normed.push(rms_with_gain_slice(attn_gain, v, eps)?);
        }
        let mixed = self.softmax_cols(li, l, 0, &mut mtp.kv, &normed, sched, mode, true)?;
        for (v, m) in x.iter_mut().zip(&mixed) {
            add_assign_fixed(v, m)?;
        }
        let ffn_gain = norms[1].as_deref().ok_or_else(|| {
            StepError::MissingTensor(format!("blk.{l}.post_attention_norm.weight"))
        })?;
        let mut h2 = Vec::with_capacity(x.len());
        for v in &x {
            h2.push(rms_with_gain_slice(ffn_gain, v, eps)?);
        }
        let mut scratch = crate::engine::chart::ffn::FfnScratch::default();
        let f = self.ffn_cols(li, l, &h2, sched, mode, &mut scratch)?;
        for (v, fv) in x.iter_mut().zip(&f) {
            add_assign_fixed(v, fv)?;
        }
        let head_gain = match nv[2].as_deref() {
            Some(g) => g,
            None => vectors
                .output_norm
                .as_deref()
                .ok_or_else(|| StepError::MissingTensor("output_norm.weight".to_string()))?,
        };
        let mut out = Vec::with_capacity(x.len());
        for v in &x {
            out.push(rms_with_gain_slice(head_gain, v, eps)?);
        }
        Ok(out)
    }

    /// The `Q8Act` head's greedy token for every column over the first `rows` vocabulary rows:
    /// one pass over those rows for all columns (`matmat_q8act`), each column equal to the single
    /// `Q8Act` head's logits on those rows.
    ///
    /// # Complexity
    ///
    /// `O(rows·n_embd·k)` multiply-adds; `rows·row_bytes` weight bytes read once.
    fn head_argmax_cols(&self, xs: &[Vec<i64>], rows: usize) -> Result<Vec<u32>, StepError> {
        let n_embd = self.config.n_embd;
        let rows = rows.min(self.config.vocab_size);
        let _head = trace::span(
            Phase::Head,
            xs.len(),
            trace::q8_bytes(rows, n_embd),
            trace::macs(rows, n_embd, xs.len()),
        );
        let t = self.lm_head();
        let (qk, bb) = quant::block_geom(t.ggml_type).ok_or(StepError::UnsupportedQuant {
            ggml_type: t.ggml_type,
        })?;
        let row_bytes = n_embd.div_ceil(qk) * bb;
        let raw = self
            .payload(t.byte_range.start, (rows * row_bytes) as u64)
            .map_err(StepError::Read)?;
        let mut qs = Vec::with_capacity(xs.len());
        for x in xs {
            qs.push(
                quant_act_pooled(x).ok_or(StepError::Matvec(MatvecError::Dequant(
                    crate::engine::EngineError::FixedOverflow,
                )))?,
            );
        }
        let logits = if qs.len() == 1 {
            vec![matvec_q8act_quant_chunked(
                t.ggml_type,
                raw,
                rows,
                n_embd,
                &qs[0],
                crate::engine::par::row_chunk_budget(rows),
            )
            .map_err(StepError::Matvec)?]
        } else {
            matmat_q8act(t.ggml_type, raw, rows, n_embd, &qs).map_err(StepError::Matvec)?
        };
        let mut out = Vec::with_capacity(logits.len());
        for l in &logits {
            out.push(argmax_fixed(l).map_err(StepError::Sampler)?);
        }
        Ok(out)
    }

    /// Greedy decoding of `n` tokens after `prompt`, drafted by the MTP block and verified two
    /// tokens per trunk pass. The ids are exactly [`Engine::next_token_state`]'s greedy ids.
    /// `draft_rows` bounds the draft head's vocabulary scan.
    ///
    /// # Errors
    ///
    /// The step, head and cache refusals; [`StepError::BadConfig`] without an MTP block or for an
    /// empty prompt.
    ///
    /// # Complexity
    ///
    /// Per round: one MTP block plus a `draft_rows`-row head, one two-column trunk pass, one
    /// two-column head pass, and on a reject an `O(affine state)` restore.
    #[allow(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        reason = "prefill, fill and the round loop in order; x, d, t, h, n, p are the round's symbols"
    )]
    pub fn generate_speculative(
        &self,
        prompt: &[u32],
        n: usize,
        draft_rows: usize,
    ) -> Result<(Vec<u32>, SpecStats), StepError> {
        if prompt.is_empty() {
            return Err(StepError::BadConfig("empty prompt"));
        }
        let (sched, mode) = (Sched::Resident, KernelMode::Q8Act);
        let cap = prompt.len() + n + 4;
        let mut st = self.state(cap)?;
        let mut mtp = self.mtp_state(cap)?;
        let layers = self.resolve_layers();
        let li = layers.get(self.config.n_layer);
        let vocab = self.config.vocab_size;
        // Prefill the trunk in blocks of `prefill_block()`, keeping every position's final state.
        let mut hs = self.prefill_cols(&mut st, prompt, mode)?;
        // Fill the MTP cache for the prompt: position p reads (prompt[p], h_{p−1}), h_{−1} = 0.
        let zero = vec![0i64; self.config.n_embd];
        let mut p = 0usize;
        while p < prompt.len() {
            let end = (p + super::prefill_block()).min(prompt.len());
            let prev: Vec<Vec<i64>> = (p..end)
                .map(|q| {
                    if q == 0 {
                        zero.clone()
                    } else {
                        hs[q - 1].clone()
                    }
                })
                .collect();
            self.mtp_cols(li, &mut mtp, &prompt[p..end], &prev)?;
            p = end;
        }
        let last = hs.pop().ok_or(StepError::BadConfig("empty prompt"))?;
        let mut x = self.head_argmax_cols(std::slice::from_ref(&last), vocab)?[0];
        let mut h_prev = last;
        let mut out = vec![x];
        let mut fill: Option<(u32, Vec<i64>)> = None;
        let mut stats = SpecStats::default();
        // The rollback buffers: one per affine layer, allocated once and overwritten in place.
        let mut snap: Vec<AffineLayer> = st
            .affine
            .as_ref()
            .map(|a| a.layers().to_vec())
            .unwrap_or_default();
        while out.len() < n {
            stats.rounds += 1;
            // Draft from (x, h_prev); first fill the MTP entry of an accepted draft if one is due.
            let (tokens, hidden) = match fill.take() {
                Some((fill_tok, fill_state)) => {
                    (vec![fill_tok, x], vec![fill_state, h_prev.clone()])
                }
                None => (vec![x], vec![h_prev.clone()]),
            };
            let y = self.mtp_cols(li, &mut mtp, &tokens, &hidden)?;
            let d = self.head_argmax_cols(&y[y.len() - 1..], draft_rows)?[0];
            // Verify [x, d] in one trunk pass.
            let pos = st.kv.occupied(0).map_err(StepError::Kv)?;
            let xf = self.compose_cols(
                &mut st.kv,
                st.affine.as_mut(),
                &[x, d],
                sched,
                mode,
                true,
                Some(&mut snap[..]),
            )?;
            let t = self.head_argmax_cols(&xf, vocab)?;
            let mut xf = xf.into_iter();
            let xf0 = xf.next().ok_or(StepError::BadConfig("verify column 0"))?;
            if t[0] == d {
                stats.accepted += 1;
                out.push(d);
                if out.len() < n {
                    out.push(t[1]);
                }
                fill = Some((d, xf0));
                x = t[1];
                h_prev = xf.next().ok_or(StepError::BadConfig("verify column 1"))?;
            } else {
                out.push(t[0]);
                let _restore = trace::span(Phase::Restore, 1, 0, 0);
                st.kv.truncate(pos + 1).map_err(StepError::Kv)?;
                if let Some(aff) = st.affine.as_mut() {
                    for (i, layer) in snap.iter().enumerate() {
                        aff.layer_mut(i)
                            .ok_or(StepError::BadConfig("affine rollback slot"))?
                            .clone_from(layer);
                    }
                }
                x = t[0];
                h_prev = xf0;
            }
        }
        out.truncate(n);
        Ok((out, stats))
    }
}
