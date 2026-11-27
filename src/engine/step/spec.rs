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

use std::collections::HashMap;

use super::{add_assign_fixed, rms_with_gain_slice, LayerIndex, Sched, StepError};
use crate::engine::chart::ffn::{matvec_q8act_quant_chunked, quant_act_pooled};
use crate::engine::error::{
    TensorKey, PART_ATTN_NORM, PART_NEXTN_ENORM, PART_NEXTN_HNORM, PART_OUTPUT_NORM,
    PART_POST_ATTENTION_NORM,
};
use crate::engine::kern::matmat::matmat_q8act;
use crate::engine::loader::opt_slice;
use crate::engine::mixer::AffineLayer;
use crate::engine::trace::{self, Phase};
use crate::engine::{argmax_fixed, Engine, KernelMode, KvState, MatvecError};
use crate::format::quant;

use crate::engine::chart::QuantAct;

/// Appends `rms_with_gain_slice(gain, x, eps)` to `out`.
fn rms_gain_push(
    gain: &[i64],
    x: &[i64],
    eps: i64,
    out: &mut Vec<Vec<i64>>,
) -> Result<(), StepError> {
    let v = rms_with_gain_slice(gain, x, eps)?;
    out.push(v);
    Ok(())
}

/// Appends the pooled `Q8Act` quantization of `x`; a fixed-point overflow refuses.
fn quant_push(x: &[i64], out: &mut Vec<QuantAct>) -> Result<(), StepError> {
    match quant_act_pooled(x) {
        Some(q) => {
            out.push(q);
            Ok(())
        }
        None => Err(StepError::Matvec(MatvecError::Dequant(
            crate::engine::EngineError::FixedOverflow,
        ))),
    }
}

/// Appends the greedy id of `logits`; a sampler refusal is [`StepError::Sampler`].
fn argmax_push(logits: &[i64], out: &mut Vec<u32>) -> Result<(), StepError> {
    match argmax_fixed(logits) {
        Ok(t) => {
            out.push(t);
            Ok(())
        }
        Err(e) => Err(StepError::Sampler(e)),
    }
}

/// Counts from one speculative decode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecStats {
    /// Draft-and-verify rounds.
    pub rounds: usize,
    /// Draft tokens the verifier accepted.
    pub accepted: usize,
    /// Draft tokens proposed; `accepted / drafted` is the per-draft acceptance rate.
    pub drafted: usize,
}

/// One verification pass's result: the ids it emitted, whether the whole draft matched, and every
/// column's output-normed state.
struct VerifyRound {
    /// On a full accept, the draft plus the free token the verifier produced after it; on a
    /// reject, column 0's token alone.
    emitted: Vec<u32>,
    /// Whether every draft token equalled the verifier's own greedy token at that position.
    accepted: bool,
    /// The columns' output-normed final states, in column order.
    states: Vec<Vec<i64>>,
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
            return Err(StepError::BadConfig(String::from(
                "model has no multi-token-prediction block",
            )));
        }
        let kv =
            KvState::new(1, cfg.n_head, cfg.n_head_kv, cfg.head_dim, cap).map_err(StepError::Kv)?;
        Ok(MtpState { kv })
    }

    /// Appends the embedding of `id` to `out`; a refusal is [`StepError::Embedding`].
    ///
    /// A named per-item body: the column loops carry its `Result` as a value.
    pub(super) fn embed_push(
        &self,
        id: u32,
        sched: Sched,
        out: &mut Vec<Vec<i64>>,
    ) -> Result<(), StepError> {
        match self.embedding_sched(id, sched) {
            Ok(h) => {
                out.push(h);
                Ok(())
            }
            Err(e) => Err(StepError::Embedding(e)),
        }
    }

    /// Appends `[enorm(embed id) | hnorm(h)]` to `cats` (the MTP input column).
    #[allow(
        clippy::too_many_arguments,
        reason = "one MTP input column: token, schedule, both gains, hidden state, eps and the sink"
    )]
    fn mtp_cat_push(
        &self,
        id: u32,
        sched: Sched,
        enorm: &[i64],
        hnorm: &[i64],
        h: &[i64],
        eps: i64,
        cats: &mut Vec<Vec<i64>>,
    ) -> Result<(), StepError> {
        let mut emb = Vec::new();
        self.embed_push_into(id, sched, &mut emb)?;
        let mut cat = rms_with_gain_slice(enorm, &emb, eps)?;
        let tail = rms_with_gain_slice(hnorm, h, eps)?;
        cat.extend(tail);
        cats.push(cat);
        Ok(())
    }

    /// The embedding of `id`; a refusal is [`StepError::Embedding`].
    fn embed_push_into(&self, id: u32, sched: Sched, out: &mut Vec<i64>) -> Result<(), StepError> {
        match self.embedding_sched(id, sched) {
            Ok(h) => {
                *out = h;
                Ok(())
            }
            Err(e) => Err(StepError::Embedding(e)),
        }
    }

    /// The MTP block over columns `(tokens[c], hs[c])`, each at the next MTP-cache position:
    /// `x = eh_proj([enorm(embed t) | hnorm(h)])`, then a softmax-attention block (the executor's
    /// `softmax_cols` and `ffn_cols` at layer `n_layer`), then the head norm. Returns the
    /// head-normed states.
    ///
    /// # Complexity
    ///
    /// One block of the trunk plus the `2·n_embd → n_embd` projection, per pass over `k` columns.
    #[allow(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        reason = "the MTP block in order: enorm, attention, FFN and head norms over one column set"
    )]
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
            .ok_or(StepError::BadConfig(String::from("no MTP vectors")))?;
        let enorm = opt_slice(&nv[0])
            .ok_or_else(|| StepError::MissingTensor(TensorKey::blk(l, PART_NEXTN_ENORM)))?;
        let hnorm = opt_slice(&nv[1])
            .ok_or_else(|| StepError::MissingTensor(TensorKey::blk(l, PART_NEXTN_HNORM)))?;
        let mut cats = Vec::with_capacity(tokens.len());
        let mut res: Result<(), StepError> = Ok(());
        let mut i = 0usize;
        while i < tokens.len() && i < hs.len() && res.is_ok() {
            res = self.mtp_cat_push(tokens[i], sched, enorm, hnorm, &hs[i], eps, &mut cats);
            i += 1;
        }
        res?;
        let mut x = self.proj_cols(li, l, 12, n_embd, 2 * n_embd, &cats, sched, mode, None)?;
        let norms = vectors
            .norms_of(l)
            .ok_or(StepError::BadConfig(String::from("no MTP block norms")))?;
        let attn_gain = opt_slice(&norms[0])
            .ok_or_else(|| StepError::MissingTensor(TensorKey::blk(l, PART_ATTN_NORM)))?;
        let mut normed = Vec::with_capacity(x.len());
        let mut res: Result<(), StepError> = Ok(());
        let mut i = 0usize;
        while i < x.len() && res.is_ok() {
            res = rms_gain_push(attn_gain, &x[i], eps, &mut normed);
            i += 1;
        }
        res?;
        let mixed = self.softmax_cols(li, l, 0, &mut mtp.kv, &normed, sched, mode, true)?;
        let mut err: Option<StepError> = None;
        let mut i = 0usize;
        while i < x.len() && i < mixed.len() && err.is_none() {
            if let Err(e) = add_assign_fixed(&mut x[i], &mixed[i]) {
                err = Some(e);
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        let ffn_gain = opt_slice(&norms[1])
            .ok_or_else(|| StepError::MissingTensor(TensorKey::blk(l, PART_POST_ATTENTION_NORM)))?;
        let mut h2 = Vec::with_capacity(x.len());
        let mut err: Option<StepError> = None;
        let mut i = 0usize;
        while i < x.len() && err.is_none() {
            match rms_with_gain_slice(ffn_gain, &x[i], eps) {
                Ok(v) => h2.push(v),
                Err(e) => err = Some(e),
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        let mut scratch = crate::engine::chart::ffn::FfnScratch::default();
        let f = self.ffn_cols(li, l, &h2, sched, mode, &mut scratch)?;
        let mut err: Option<StepError> = None;
        let mut i = 0usize;
        while i < x.len() && i < f.len() && err.is_none() {
            if let Err(e) = add_assign_fixed(&mut x[i], &f[i]) {
                err = Some(e);
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        let head_gain = match opt_slice(&nv[2]) {
            Some(g) => g,
            None => opt_slice(&vectors.output_norm)
                .ok_or_else(|| StepError::MissingTensor(TensorKey::top(PART_OUTPUT_NORM)))?,
        };
        let mut out = Vec::with_capacity(x.len());
        let mut err: Option<StepError> = None;
        let mut i = 0usize;
        while i < x.len() && err.is_none() {
            match rms_with_gain_slice(head_gain, &x[i], eps) {
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
        let raw = match self.payload(t.byte_range.start, (rows * row_bytes) as u64) {
            Ok(raw) => raw,
            Err(e) => return Err(StepError::Read(e)),
        };
        let mut qs = Vec::with_capacity(xs.len());
        let mut res: Result<(), StepError> = Ok(());
        let mut i = 0usize;
        while i < xs.len() && res.is_ok() {
            res = quant_push(&xs[i], &mut qs);
            i += 1;
        }
        res?;
        let logits = if qs.len() == 1 {
            vec![matvec_q8act_quant_chunked(
                t.ggml_type,
                raw,
                rows,
                n_embd,
                &qs[0],
                crate::engine::ops::par::row_chunk_budget(rows),
            )
            .map_err(StepError::Matvec)?]
        } else {
            matmat_q8act(t.ggml_type, raw, rows, n_embd, &qs).map_err(StepError::Matvec)?
        };
        let mut out = Vec::with_capacity(logits.len());
        let mut res: Result<(), StepError> = Ok(());
        let mut i = 0usize;
        while i < logits.len() && res.is_ok() {
            res = argmax_push(&logits[i], &mut out);
            i += 1;
        }
        res?;
        Ok(out)
    }

    /// Verify one draft round: one trunk pass over `[x, drafts...]`, one head pass over every
    /// column, and the exact accept-or-reject decision. A full accept keeps every column and
    /// emits the draft plus the free token the verifier produced after it; any mismatch rejects
    /// the whole draft, keeps column 0 (the verifier's own greedy token), truncates the cache to
    /// that column and restores every affine layer to its after-column-0 state. Emitting only
    /// column 0 on a partial match is exact; it forgoes the matched prefix and its speed.
    ///
    /// # Errors
    ///
    /// The compose, head and cache refusals; [`StepError::BadConfig`] for an empty draft.
    ///
    /// # Complexity
    ///
    /// One `drafts.len() + 1`-column trunk pass, one head pass over the same columns, and on a
    /// reject one `O(affine state)` restore.
    fn verify_round(
        &self,
        st: &mut super::DecodeState,
        x: u32,
        drafts: &[u32],
        snap: &mut [AffineLayer],
    ) -> Result<VerifyRound, StepError> {
        if drafts.is_empty() {
            return Err(StepError::BadConfig(String::from("empty draft")));
        }
        let vocab = self.config.vocab_size;
        let pos = st.kv.occupied(0).map_err(StepError::Kv)?;
        let mut tokens = Vec::with_capacity(drafts.len() + 1);
        tokens.push(x);
        tokens.extend_from_slice(drafts);
        let states = self.compose_cols(
            &mut st.kv,
            st.affine.as_mut(),
            &tokens,
            Sched::Resident,
            KernelMode::Q8Act,
            true,
            Some(snap),
        )?;
        let t = self.head_argmax_cols(&states, vocab)?;
        let mut accepted = t.len() == drafts.len() + 1;
        let mut i = 0usize;
        while accepted && i < drafts.len() {
            accepted = t[i] == drafts[i];
            i += 1;
        }
        if !accepted && std::env::var_os("APEIRON_LOOKUP_DEBUG").is_some() {
            let mut bad = 0usize;
            while bad < drafts.len() && t[bad] == drafts[bad] {
                bad += 1;
            }
            eprintln!("verify reject at {bad}: x={x} drafts={drafts:?} t={t:?}");
        }
        if accepted {
            let mut emitted = Vec::with_capacity(drafts.len() + 1);
            emitted.extend_from_slice(drafts);
            emitted.push(t[drafts.len()]);
            Ok(VerifyRound {
                emitted,
                accepted,
                states,
            })
        } else {
            let _restore = trace::span(Phase::Restore, 1, 0, 0);
            st.kv.truncate(pos + 1).map_err(StepError::Kv)?;
            if let Some(aff) = st.affine.as_mut() {
                let mut j = 0usize;
                while j < snap.len() {
                    match aff.layer_mut(j) {
                        Some(dst) => dst.clone_from(&snap[j]),
                        None => {
                            return Err(StepError::BadConfig(String::from("affine rollback slot")));
                        }
                    }
                    j += 1;
                }
            }
            Ok(VerifyRound {
                emitted: vec![t[0]],
                accepted: false,
                states,
            })
        }
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
            return Err(StepError::BadConfig(String::from("empty prompt")));
        }
        let mode = KernelMode::Q8Act;
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
        let mut err: Option<StepError> = None;
        while p < prompt.len() && err.is_none() {
            let end = (p + super::prefill_block()).min(prompt.len());
            let mut prev: Vec<Vec<i64>> = Vec::with_capacity(end.saturating_sub(p));
            let mut q = p;
            while q < end {
                prev.push(if q == 0 {
                    zero.clone()
                } else {
                    hs[q - 1].clone()
                });
                q += 1;
            }
            match self.mtp_cols(li, &mut mtp, &prompt[p..end], &prev) {
                Ok(_) => p = end,
                Err(e) => err = Some(e),
            }
        }
        if let Some(e) = err {
            return Err(e);
        }
        let last = hs
            .pop()
            .ok_or(StepError::BadConfig(String::from("empty prompt")))?;
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
        let mut err: Option<StepError> = None;
        while out.len() < n {
            stats.rounds += 1;
            // Draft from (x, h_prev); first fill the MTP entry of an accepted draft if one is due.
            let (tokens, hidden) = match fill.take() {
                Some((fill_tok, fill_state)) => {
                    (vec![fill_tok, x], vec![fill_state, h_prev.clone()])
                }
                None => (vec![x], vec![h_prev.clone()]),
            };
            let y = match self.mtp_cols(li, &mut mtp, &tokens, &hidden) {
                Ok(y) => y,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            let d = match self.head_argmax_cols(&y[y.len() - 1..], draft_rows) {
                Ok(v) => v[0],
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            // Verify [x, d] in one two-column trunk pass.
            let round = match self.verify_round(&mut st, x, std::slice::from_ref(&d), &mut snap) {
                Ok(r) => r,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            stats.drafted += 1;
            let mut col_states = round.states.into_iter();
            let s0 = match col_states.next() {
                Some(s) => s,
                None => {
                    err = Some(StepError::BadConfig(String::from("verify column 0 state")));
                    break;
                }
            };
            if round.accepted {
                stats.accepted += 1;
                out.push(d);
                if out.len() < n {
                    out.push(round.emitted[1]);
                }
                let s1 = match col_states.next() {
                    Some(s) => s,
                    None => {
                        err = Some(StepError::BadConfig(String::from("verify column 1 state")));
                        break;
                    }
                };
                fill = Some((d, s0));
                x = round.emitted[1];
                h_prev = s1;
            } else {
                out.push(round.emitted[0]);
                x = round.emitted[0];
                h_prev = s0;
            }
        }
        if let Some(e) = err {
            return Err(e);
        }
        out.truncate(n);
        Ok((out, stats))
    }

    /// Greedy decoding of `n` tokens after `prompt`, drafted by an exact prompt-lookup n-gram
    /// drafter and verified in one trunk pass over every proposed column. The ids are exactly
    /// [`Engine::next_token_state`]'s greedy ids.
    ///
    /// At each round the drafter takes the last `ngram` ids of the causal stream, finds the latest
    /// earlier occurrence that ends before them, and proposes up to `draft` ids that followed it.
    /// `[x, proposals...]` go through [`Engine::verify_round`]: a full match keeps every column
    /// and emits the proposals plus the verifier's free token after them; any mismatch keeps only
    /// column 0. A round with no match is one ordinary greedy step, so the emitted ids do not
    /// depend on the drafter.
    ///
    /// # Errors
    ///
    /// The step, head and cache refusals; [`StepError::BadConfig`] for an empty prompt or a zero
    /// `ngram`/`draft`.
    ///
    /// # Complexity
    ///
    /// Prefill plus, per round, one `m + 1`-column trunk and head pass for a draft of `m`, or one
    /// greedy step; the n-gram index is `O(stream)` entries and `O(stream · ngram)` hash work.
    #[allow(
        clippy::too_many_lines,
        reason = "prefill, index, draft, verify and emit in one round loop"
    )]
    pub fn generate_lookup(
        &self,
        prompt: &[u32],
        n: usize,
        ngram: usize,
        draft: usize,
    ) -> Result<(Vec<u32>, SpecStats), StepError> {
        if prompt.is_empty() {
            return Err(StepError::BadConfig(String::from("empty prompt")));
        }
        if ngram == 0 || draft == 0 {
            return Err(StepError::BadConfig(String::from(
                "lookup needs a positive n-gram and draft length",
            )));
        }
        let mode = KernelMode::Q8Act;
        let cap = prompt.len() + n + draft + 4;
        let mut st = self.state(cap)?;
        let mut hs = self.prefill_cols(&mut st, prompt, mode)?;
        let last = hs
            .pop()
            .ok_or(StepError::BadConfig(String::from("empty prompt")))?;
        let vocab = self.config.vocab_size;
        let mut x = self.head_argmax_cols(std::slice::from_ref(&last), vocab)?[0];
        let mut out = vec![x];
        let mut ctx: Vec<u32> = Vec::with_capacity(prompt.len() + n + 1);
        ctx.extend_from_slice(prompt);
        ctx.push(x);
        let mut index: HashMap<u64, usize> = HashMap::new();
        let mut upto = 0usize;
        index_ngrams(&ctx, ngram, &mut index, &mut upto);
        let mut stats = SpecStats::default();
        let mut snap: Vec<AffineLayer> = st
            .affine
            .as_ref()
            .map(|a| a.layers().to_vec())
            .unwrap_or_default();
        let mut err: Option<StepError> = None;
        while out.len() < n {
            stats.rounds += 1;
            let proposals = lookup_draft(&ctx, ngram, draft, &index);
            if proposals.is_empty() {
                match self.next_token_state(&mut st, x, mode) {
                    Ok(y) => {
                        out.push(y);
                        ctx.push(y);
                        x = y;
                    }
                    Err(e) => {
                        err = Some(e);
                        break;
                    }
                }
            } else {
                stats.drafted += proposals.len();
                match self.verify_round(&mut st, x, &proposals, &mut snap) {
                    Ok(round) => {
                        if round.accepted {
                            stats.accepted += proposals.len();
                            out.extend_from_slice(&round.emitted);
                        } else {
                            out.push(round.emitted[0]);
                        }
                        x = match round.emitted.last() {
                            Some(&t) => t,
                            None => {
                                err = Some(StepError::BadConfig(String::from("verify emit")));
                                break;
                            }
                        };
                        ctx.extend_from_slice(&round.emitted);
                    }
                    Err(e) => {
                        err = Some(e);
                        break;
                    }
                }
            }
            index_ngrams(&ctx, ngram, &mut index, &mut upto);
        }
        if let Some(e) = err {
            return Err(e);
        }
        out.truncate(n);
        Ok((out, stats))
    }
}

/// FNV-1a over an id n-gram; a hash collision only lowers a draft, never an emitted id.
fn ngram_hash(ids: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &id in ids {
        h ^= u64::from(id);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Insert every n-gram start that cannot overlap the trailing `ngram` ids: `start ≤ len − 2·ngram`.
/// `upto` is the next start to insert, so one stream appends work over a growing context.
fn index_ngrams(ctx: &[u32], ngram: usize, map: &mut HashMap<u64, usize>, upto: &mut usize) {
    let Some(span) = ngram.checked_mul(2) else {
        return;
    };
    let Some(last) = ctx.len().checked_sub(span) else {
        return;
    };
    let mut start = *upto;
    while start <= last {
        map.insert(ngram_hash(&ctx[start..start + ngram]), start);
        start += 1;
    }
    *upto = start;
}

/// The ids that followed the latest indexed occurrence of the trailing `ngram` ids, up to `draft`.
/// Empty when no non-overlapping occurrence is indexed; the proposal never reads past `ctx`.
fn lookup_draft(ctx: &[u32], ngram: usize, draft: usize, index: &HashMap<u64, usize>) -> Vec<u32> {
    let Some(suffix_start) = ctx.len().checked_sub(ngram) else {
        return Vec::new();
    };
    let suffix = &ctx[suffix_start..];
    let Some(&start) = index.get(&ngram_hash(suffix)) else {
        return Vec::new();
    };
    if start + ngram > suffix_start || ctx[start..start + ngram] != *suffix {
        return Vec::new();
    }
    let from = start + ngram;
    let take = draft.min(ctx.len() - from);
    ctx[from..from + take].to_vec()
}
