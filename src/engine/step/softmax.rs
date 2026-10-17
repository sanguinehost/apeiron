//! The attention (softmax-layer) half of the column loop: [`Engine::softmax_cols`]'s projections,
//! its serial per-column path and its pooled block path.
//!
//! Split out of `step/cols.rs` by moving items, not by changing them; module paths in the generated
//! extraction move with it (`step.cols.*` -> `step.softmax.*`, `proof/boundary-axioms.txt`).

use super::cols::{first_by_key, fold_col};
use super::{
    add_assign_fixed, apply_sigmoid_gate, qk_norm_and_rope, split_query_gate, LayerIndex, Sched,
    StepError,
};
use crate::engine::chart::task::{HeadTask, Task2};
use crate::engine::loader::{opt_slice, DecodeVectors};
use crate::engine::trace::{self, Phase};
use crate::engine::{AttnError, Engine, KernelMode, KvState};
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
    /// The `[query | key | value]` projection columns of [`Engine::softmax_cols`]: the single-column
    /// fast path when `xs` holds one token and there is no [`LayerIndex`], else the grouped
    /// projection over every column. `dims` is `[q_proj, kv_out, n_embd]`.
    ///
    /// # Complexity
    ///
    /// As [`Engine::step_qkv_sched`] (one column) or [`Engine::group_cols`] (the block).
    pub(super) fn softmax_qkv(
        &self,
        li: Option<&LayerIndex>,
        layer: usize,
        xs: &[Vec<i64>],
        dims: [usize; 3],
        sched: Sched,
        mode: KernelMode,
    ) -> Result<Vec<Vec<Vec<i64>>>, StepError> {
        let [q_proj, kv_out, n_embd] = dims;
        if xs.len() == 1 && li.is_none() {
            let mut r =
                self.step_qkv_sched(layer, [q_proj, kv_out, kv_out], n_embd, &xs[0], sched, mode)?;
            let q = std::mem::take(&mut r[0]);
            let k = std::mem::take(&mut r[1]);
            let v = std::mem::take(&mut r[2]);
            Ok(vec![vec![q], vec![k], vec![v]])
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
            )
        }
    }

    /// The serial per-column attention loop of [`Engine::softmax_cols`]: column `c` runs at position
    /// `occupied + c`, in token order, on the calling thread.
    ///
    /// # Complexity
    ///
    /// `O(Σ_c cost(softmax_col(c)))`; `O(k)` output rows.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer, slot, cache, projections, norm epsilon, rotation table and dispatch are the loop's input"
    )]
    pub(super) fn softmax_serial_cols(
        &self,
        layer: usize,
        slot: usize,
        kv: &mut KvState,
        qkv: &[Vec<Vec<i64>>],
        eps: i64,
        table: &crate::engine::rope::RopeTable,
        parallel_heads: bool,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let k = qkv[0].len();
        let mut atts = Vec::with_capacity(k);
        let mut err: Option<StepError> = None;
        let mut c = 0usize;
        while c < k && err.is_none() {
            match self.softmax_col(layer, slot, kv, qkv, c, eps, table, parallel_heads) {
                Ok(att) => atts.push(att),
                Err(e) => err = Some(e),
            }
            c += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        Ok(atts)
    }

    /// The attention output projection of [`Engine::softmax_cols`], after the `AttnCore` span.
    ///
    /// # Complexity
    ///
    /// As [`Engine::proj_cols`] on projection part 3.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer index, layer, embedding width, query output width, attention rows, schedule and dispatch are the projection's input"
    )]
    pub(super) fn softmax_proj_out(
        &self,
        li: Option<&LayerIndex>,
        layer: usize,
        n_embd: usize,
        q_out: usize,
        atts: &[Vec<i64>],
        sched: Sched,
        mode: KernelMode,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let k = atts.len();
        let _out = trace::span(
            Phase::AttnProjOut,
            k,
            trace::q8_bytes(n_embd, q_out),
            trace::macs(n_embd, q_out, k),
        );
        self.proj_cols(li, layer, 3, n_embd, q_out, atts, sched, mode, None)
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
        let (mut query, mut key, value) = Self::softmax_bias_add(qkv, vectors, layer, c)?;
        let gate = Self::softmax_gate_split(&mut query, n_head, hd, cfg.attn_gate)?;
        let pos = kv.occupied(slot).map_err(StepError::Kv)?;
        let pos_u32 = u32::try_from(pos).map_err(|_| StepError::LengthOverflow)?;
        Self::softmax_qk_rope(
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
        let appended = Self::softmax_kv_append(kv, slot, &key, &value)?;
        let kv_ro: &KvState = kv;
        let mut att = Self::softmax_head_loop(
            kv_ro,
            slot,
            &query,
            [n_head, hd, q_out, appended],
            parallel_heads,
        )?;
        Self::softmax_gate_apply(&mut att, opt_slice(&gate))?;
        Ok(att)
    }

    /// The column's three attention biases, added to the cloned `[query | key | value]` in that
    /// order. A bias entry of `None` leaves its vector unchanged, as the former inline blocks did.
    ///
    /// # Complexity
    ///
    /// `O(q_out + 2·kv_out)` checked adds, `O(1)` besides the three column clones.
    #[allow(
        clippy::type_complexity,
        clippy::index_refutable_slice,
        reason = "the three biased column vectors are the call's result; the three biases are read by constant index because an array pattern does not extract"
    )]
    fn softmax_bias_add(
        qkv: &[Vec<Vec<i64>>],
        vectors: &DecodeVectors,
        layer: usize,
        c: usize,
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>), StepError> {
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
        Ok((query, key, value))
    }

    /// The gated-query split: `query` becomes its `n_head·hd` query half and the gate half is
    /// returned; an ungated layer returns `None` and leaves `query` untouched.
    ///
    /// # Complexity
    ///
    /// `O(n_head·hd)` copies; `O(1)` otherwise.
    fn softmax_gate_split(
        query: &mut Vec<i64>,
        n_head: usize,
        hd: usize,
        gated: bool,
    ) -> Result<Option<Vec<i64>>, StepError> {
        if !gated {
            return Ok(None);
        }
        let (q, g) = split_query_gate(query, n_head, hd)?;
        *query = q;
        Ok(Some(g))
    }

    /// The QK-norm and `RoPE` of the column's query and key at `pos`.
    ///
    /// # Complexity
    ///
    /// As [`qk_norm_and_rope`].
    #[allow(
        clippy::too_many_arguments,
        reason = "the two QK gains come from vectors, the rotation table, both vectors, the head \
                  geometry, the norm epsilon, the position and the schedule are the call's input"
    )]
    fn softmax_qk_rope(
        vectors: &DecodeVectors,
        layer: usize,
        table: &crate::engine::rope::RopeTable,
        query: &mut [i64],
        key: &mut [i64],
        hd: usize,
        eps: i64,
        pos: u32,
        parallel: bool,
    ) -> Result<(), StepError> {
        qk_norm_and_rope(vectors, layer, table, query, key, hd, eps, pos, parallel)
    }

    /// Append the column's key and value to `kv` at `slot`, returning the position after the
    /// append, the prefix length this column attends over.
    ///
    /// # Complexity
    ///
    /// As [`KvState::append`].
    fn softmax_kv_append(
        kv: &mut KvState,
        slot: usize,
        key: &[i64],
        value: &[i64],
    ) -> Result<usize, StepError> {
        kv.append(slot, key, value).map_err(StepError::Kv)
    }

    /// The per-head attention read of one column into an `q_out`-wide output; `dims` is
    /// `[n_head, hd, q_out, appended]`.
    ///
    /// # Complexity
    ///
    /// `O(n_head·hd·appended)` word reads, `O(q_out)` output.
    fn softmax_head_loop(
        kv: &KvState,
        slot: usize,
        query: &[i64],
        dims: [usize; 4],
        parallel_heads: bool,
    ) -> Result<Vec<i64>, StepError> {
        let [n_head, hd, q_out, appended] = dims;
        let mut att: Vec<i64> = vec![0i64; q_out];
        let head = QueryHeadCtx {
            kv,
            slot,
            query,
            hd,
            appended,
        };
        if let Some((_, e)) =
            crate::engine::chart::ffn::attn_heads(n_head, hd, &mut att, parallel_heads, &head)
        {
            return Err(StepError::Kv(e));
        }
        Ok(att)
    }

    /// The attention output gate: `att[i] ← att[i] · sigmoid(gate[i])` when the layer is gated.
    ///
    /// # Complexity
    ///
    /// As [`apply_sigmoid_gate`].
    fn softmax_gate_apply(att: &mut [i64], gate: Option<&[i64]>) -> Result<(), StepError> {
        if let Some(g) = gate {
            apply_sigmoid_gate(att, g)?;
        }
        Ok(())
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
    pub(super) fn softmax_core_block(
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
/// `dst` as `[query | key | value | gate]`. The stages are the same calls [`Engine::softmax_col`]
/// makes, so the block and the per-column path share one body per stage.
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
    let (mut query, mut key, value) = Engine::softmax_bias_add(qkv, vectors, layer, c)?;
    let gate = prep_gate_split(&mut query, n_head, hd, gated)?;
    let pos = u32::try_from(pos0 + c).map_err(|_| StepError::LengthOverflow)?;
    Engine::softmax_qk_rope(
        vectors, layer, table, &mut query, &mut key, hd, eps, pos, false,
    )?;
    prep_copy_out(dst, &query, &key, &value, &gate, [q_out, kv_out], gated)
}

/// The gate half of [`prep_column`]'s query, or `Vec::new()` for an ungated layer. `Engine::softmax_gate_split`
/// returns the option; the pool writes a flat `[query | key | value | gate]` slot, so the absent case is empty.
///
/// # Complexity
///
/// `O(n_head·hd)` copies; `O(1)` otherwise.
fn prep_gate_split(
    query: &mut Vec<i64>,
    n_head: usize,
    hd: usize,
    gated: bool,
) -> Result<Vec<i64>, StepError> {
    match Engine::softmax_gate_split(query, n_head, hd, gated)? {
        Some(g) => Ok(g),
        None => Ok(Vec::new()),
    }
}

/// Write the column's `[query | key | value | gate]` into the pool slot `dst`; `dims` is
/// `[q_out, kv_out]`. A width mismatch is refused before any byte is written.
///
/// # Complexity
///
/// `O(q_out + 2·kv_out)` word copies; `O(1)` besides.
fn prep_copy_out(
    dst: &mut [i64],
    query: &[i64],
    key: &[i64],
    value: &[i64],
    gate: &[i64],
    dims: [usize; 2],
    gated: bool,
) -> Result<(), StepError> {
    let [q_out, kv_out] = dims;
    if query.len() != q_out || key.len() != kv_out || value.len() != kv_out {
        return Err(StepError::BadConfig(String::from("attention block widths")));
    }
    let (dq, rest) = dst.split_at_mut(q_out);
    let (dk, rest) = rest.split_at_mut(kv_out);
    let (dv, dg) = rest.split_at_mut(kv_out);
    dq.copy_from_slice(query);
    dk.copy_from_slice(key);
    dv.copy_from_slice(value);
    if gated {
        dg.copy_from_slice(gate);
    }
    Ok(())
}
