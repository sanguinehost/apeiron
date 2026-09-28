//! Row-chunked Q8-activation projections and the chart-native FFN: the `KernelMode::Q8Act`
//! executing path. 
//!
//! [`matvec_q8act_chunked`] is [`super::matvec_q8act`] over whole rows split into contiguous row
//! chunks run on scoped threads, concatenated in row order; it is the Q8-activation twin of
//! `par::matvec_rows_chunked`. [`projections_q8act`] generalizes its gate/up fusion to `n`
//! projections over one `quant_act(x)` and one pool dispatch, taking either the chart row core or a
//! caller-supplied [`RowKernel`] so the `PQ2_0` LUT can share the activation without this module
//! importing `kern`. [`ffn_q8act`] is [`ffn_fixed`](crate::engine::ffn::ffn_fixed)'s arithmetic with
//! each projection replaced by `matvec_q8act` over `quant_act` of its input; the gate and up
//! projections share one `quant_act` of `x`. The `Exact` path in `ffn.rs` and `par.rs` is not
//! touched.

use crate::format::quant;

use super::super::ffn::{silu_fixed, FfnError, ProjWeights};
use super::super::matvec::MatvecError;
use super::super::par::row_chunk_budget;
use super::super::value::round_shift_32;
use super::super::EngineError;
use super::pool::Pool;
use super::{matvec_q8act, q8act_plan, q8act_rows, quant_act, Q8ActPlan, QuantAct};
use std::sync::Mutex;

#[cfg(test)]
thread_local! {
    /// Test-only: when false, [`quant_act_pooled`] takes the single [`quant_act`] call, the
    /// equality oracle for the chunked schedule.
    static POOL_QUANT: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Test-only: select the single-call [`quant_act`] schedule inside [`quant_act_pooled`].
#[cfg(test)]
pub(crate) fn set_pool_quant(on: bool) {
    POOL_QUANT.with(|c| c.set(on));
}

/// Disjoint output ranges of one pooled quantization: task `t` writes only its own block run.
#[derive(Clone, Copy)]
struct ActPtr {
    scales: *mut i64,
    codes: *mut i8,
}
// SAFETY: every task writes a distinct block run of the two allocations, which outlive the
// dispatch; no task reads another's range.
unsafe impl Send for ActPtr {}
// SAFETY: as above; `quant_act_pooled` gives each task its own block range.
unsafe impl Sync for ActPtr {}

/// [`quant_act`] over the pool: `x` is cut into whole 32-element blocks and each contiguous run of
/// blocks is quantized by one pool task with the same [`quant_act`]. A block's scale and codes
/// depend only on that block, so the concatenation is exactly the single call's `scales`/`codes`
/// (and any chunk's refusal is the same `None`); `blocks <= 1` or a one-wide pool delegates.
///
/// # Complexity
///
/// `O(x.len())` time in the same two passes as [`quant_act`], split over `min(2·width, blocks)`
/// tasks; `O(x.len())` output, one allocation pair.
pub(crate) fn quant_act_pooled(x: &[i64]) -> Option<QuantAct> {
    /// The activation quantization block, as in [`quant_act`].
    const QK: usize = 32;
    #[cfg(test)]
    if !POOL_QUANT.with(std::cell::Cell::get) {
        return quant_act(x);
    }
    let blocks = x.len().div_ceil(QK);
    let pool = Pool::global();
    let tasks = (2 * pool.width()).min(blocks).max(1);
    if tasks <= 1 {
        return quant_act(x);
    }
    let per = blocks.div_ceil(tasks);
    let mut scales: Vec<i64> = vec![0; blocks];
    let mut codes: Vec<i8> = vec![0; x.len()];
    let ptr = ActPtr {
        scales: scales.as_mut_ptr(),
        codes: codes.as_mut_ptr(),
    };
    let act = &ptr;
    let refused = std::sync::atomic::AtomicBool::new(false);
    pool.run(tasks, &|t| {
        let lo_b = (t * per).min(blocks);
        let hi_b = ((t + 1) * per).min(blocks);
        if lo_b >= hi_b {
            return;
        }
        let lo = lo_b * QK;
        let hi = (hi_b * QK).min(x.len());
        match quant_act(&x[lo..hi]) {
            Some(qa) => {
                // SAFETY: task `t` is the only writer of blocks `[lo_b, hi_b)` and of codes
                // `[lo, hi)`; both writes stay inside the two allocations, which outlive the
                // dispatch.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        qa.scales().as_ptr(),
                        act.scales.add(lo_b),
                        hi_b - lo_b,
                    );
                    std::ptr::copy_nonoverlapping(qa.codes().as_ptr(), act.codes.add(lo), hi - lo);
                }
            }
            None => refused.store(true, std::sync::atomic::Ordering::Relaxed),
        }
    });
    if refused.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    Some(QuantAct { scales, codes })
}

/// [`quant_act_pooled`] as a refusal: `None` only for an activation block whose scale times code
/// leaves `i64`, which has no Q32.32 dequantization, so the projection refuses with the same
/// [`MatvecError::Dequant`]`(`[`EngineError::FixedOverflow`]`)` [`matvec_q8act`] returns for such a
/// [`QuantAct`].
fn quantize(x: &[i64]) -> Result<QuantAct, MatvecError> {
    quant_act_pooled(x).ok_or(MatvecError::Dequant(EngineError::FixedOverflow))
}

/// The `SwiGLU` element of one index: `round(silu(gate)·up / 2^32)` in checked integer arithmetic.
///
/// # Errors
///
/// [`FfnError::Overflow`] when the `i128` product or the `i64` narrowing leaves its range;
/// `silu_fixed`'s own refusal otherwise.
fn swiglu_one(gate: i64, up: i64) -> Result<i64, FfnError> {
    let s = silu_fixed(gate)?;
    let prod = i128::from(s)
        .checked_mul(i128::from(up))
        .ok_or(FfnError::Overflow)?;
    i64::try_from(round_shift_32(prod)).map_err(|_| FfnError::Overflow)
}

/// `h_i = round(silu(gate_i)·up_i / 2^32)` for every `i`, on the pool when `parallel`.
///
/// Every element is independent and [`swiglu_one`] is unchanged, so the pool split is bit-identical
/// to the serial loop. Returns the first refusal in element order. The serial loop (`parallel`
/// false) is the test oracle.
///
/// # Errors
///
/// As [`swiglu_one`], and [`FfnError::BadGeometry`] when the three slices disagree in length.
///
/// # Complexity
///
/// `O(n)` [`swiglu_one`] evaluations over `⌈n/(2·width)⌉`-element chunks, one pool dispatch;
/// `O(1)` scratch beyond `h`.
pub(crate) fn swiglu_hidden(
    gate: &[i64],
    up: &[i64],
    h: &mut [i64],
    parallel: bool,
) -> Result<(), FfnError> {
    if gate.len() != up.len() || gate.len() != h.len() {
        return Err(FfnError::BadGeometry);
    }
    if !parallel {
        let mut i = 0usize;
        while i < h.len() {
            h[i] = swiglu_one(gate[i], up[i])?;
            i += 1;
        }
        return Ok(());
    }
    let pool = Pool::global();
    let chunk = h.len().div_ceil(2 * pool.width()).max(1);
    let first: Mutex<Option<(usize, FfnError)>> = Mutex::new(None);
    pool.chunks_mut(h, chunk, &|_c, start, dst| {
        let mut j = 0usize;
        while j < dst.len() {
            match swiglu_one(gate[start + j], up[start + j]) {
                Ok(v) => dst[j] = v,
                Err(e) => {
                    let mut slot = first
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if slot.as_ref().is_none_or(|(k, _)| start + j < *k) {
                        *slot = Some((start + j, e));
                    }
                    return;
                }
            }
            j += 1;
        }
    });
    match first
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        Some((_, e)) => Err(e),
        None => Ok(()),
    }
}

/// Apply exact `SwiGLU` to rectangular columns and return the owned results on success.
///
/// For `k` columns of width `n`, validation takes `O(k)` length checks and evaluation takes
/// `O(k * n)` word work. The output uses `k` vector descriptors and `k * n` words. When
/// `parallel` and `k >= pool.width()`, one pool dispatch schedules whole columns; otherwise
/// per-column [`swiglu_hidden`] calls preserve its schedule.
///
/// # Errors
///
/// Returns [`FfnError::BadGeometry`] before evaluation if column counts or widths disagree, and
/// otherwise the first [`swiglu_one`] refusal in column-major order.
pub(crate) fn swiglu_hidden_cols(
    gate: &[Vec<i64>],
    up: &[Vec<i64>],
    parallel: bool,
) -> Result<Vec<Vec<i64>>, FfnError> {
    let columns = gate.len();
    if columns != up.len() {
        return Err(FfnError::BadGeometry);
    }
    let width = gate.first().map_or(0, Vec::len);
    if gate.iter().any(|x| x.len() != width) || up.iter().any(|x| x.len() != width) {
        return Err(FfnError::BadGeometry);
    }
    columns.checked_mul(width).ok_or(FfnError::BadGeometry)?;
    let mut hidden: Vec<Vec<i64>> = (0..columns).map(|_| vec![0; width]).collect();
    if width == 0 {
        return Ok(hidden);
    }

    if parallel && columns >= Pool::global().width() {
        let pool = Pool::global();
        // Bound task count independently of the input size; each task still owns whole columns.
        let task_chunk = columns.div_ceil(64).max(1);
        let first: Mutex<Option<(usize, FfnError)>> = Mutex::new(None);
        pool.chunks_mut(&mut hidden, task_chunk, &|_, start, dst| {
            for (offset, output) in dst.iter_mut().enumerate() {
                let column = start + offset;
                if let Err(error) = swiglu_hidden(&gate[column], &up[column], output, false) {
                    let mut slot = first
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if slot.as_ref().is_none_or(|(earliest, _)| column < *earliest) {
                        *slot = Some((column, error));
                    }
                    return;
                }
            }
        });
        if let Some((_, error)) = first
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            return Err(error);
        }
    } else {
        for column in 0..columns {
            swiglu_hidden(&gate[column], &up[column], &mut hidden[column], parallel)?;
        }
    }
    Ok(hidden)
}

/// The most scoped threads the Q8-activation row-chunked kernel uses.
///
/// `row_chunk_budget` resolves to all `available_parallelism` threads (32 here), so every projection
/// spawns and joins that many per call; on the hybrid 8-P+16-E host the spawn/join sys time and the
/// E-core/SMT contention outweigh the added parallelism. This cap was fixed by the chunk-count
/// sweep recorded in the W2 receipt: the decode tok/s for `[4, 8, 16, 32]` chunks.
const Q8ACT_MAX_CHUNKS: usize = 8;

/// [`matvec_q8act`] over `n_out` whole rows split into contiguous row chunks, run on the persistent
/// [`Pool`] and concatenated in row order; `q` is the already-quantized activation, so a
/// caller projecting several weight matrices with one `x` pays one `quant_act`.
///
/// The split is [`crate::engine::par::matvec_rows_chunked`]'s: `n_out / chunks` rows per chunk with
/// the first `n_out % chunks` chunks one row longer, so any `chunks` covers every row once. The
/// byte stride is [`quant::block_geom`]'s. `chunks <= 1` is exactly one [`matvec_q8act`] call over
/// the whole payload; when `ggml_type` has no block geometry the chunked split is not formed and
/// the whole call is one [`matvec_q8act`], which handles that family.
///
/// # Errors
///
/// The first chunk's refusal in row order: the chunks are joined in spawn order (ascending `lo`)
/// and the first `Err` is returned. A short payload, an unsupported quant family and a shape
/// mismatch are the same refusals [`matvec_q8act`] makes.
///
/// # Complexity
///
/// `O(n_out·n_in)` time and `O(n_out)` output — the same work as one [`matvec_q8act`] — plus
/// `O(chunks)` join handles and no copy of the weight payload.
pub(crate) fn matvec_q8act_quant_chunked(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    q: &QuantAct,
    chunks: usize,
) -> Result<Vec<i64>, MatvecError> {
    let upper = if n_out == 0 { 1 } else { n_out };
    let actual = chunks.min(Q8ACT_MAX_CHUNKS).clamp(1, upper);
    if actual == 1 {
        return matvec_q8act(ggml_type, raw, n_out, n_in, q);
    }
    let Some((qk, bb)) = quant::block_geom(ggml_type) else {
        return matvec_q8act(ggml_type, raw, n_out, n_in, q);
    };
    let row_bytes = n_in
        .div_ceil(qk)
        .checked_mul(bb)
        .ok_or(MatvecError::LengthOverflow)?;
    let need = n_out
        .checked_mul(row_bytes)
        .ok_or(MatvecError::LengthOverflow)?;
    if raw.len() < need {
        return Err(MatvecError::ShortPayload {
            need,
            have: raw.len(),
        });
    }
    if q.len() != n_in {
        return Err(MatvecError::ShapeMismatch { x: q.len(), n_in });
    }
    // One plan for every row chunk: the row-independent magnitude check, fast-row decision and
    // `Q4_K` sums are computed once for the call instead of once per task. A `Reference` plan is
    // one non-fused or non-Q32.32 activation, where each task is still one whole [`matvec_q8act`].
    let plan = q8act_plan(ggml_type, n_in, q);
    // Contiguous row chunks of `per` rows (the last may be shorter), one pool task each; every
    // row is independent, so the concatenation equals the single call for any split.
    let pool = Pool::global();
    let tasks = actual.max(2 * pool.width()).min(n_out);
    let per = n_out.div_ceil(tasks);
    let mut out: Vec<i64> = vec![0; n_out];
    let first_err: Mutex<Option<(usize, MatvecError)>> = Mutex::new(None);
    pool.chunks_mut(&mut out, per, &|c, lo, dst| {
        let rows = dst.len();
        let res = raw
            .get(lo * row_bytes..(lo + rows) * row_bytes)
            .ok_or(MatvecError::ShortPayload {
                need,
                have: raw.len(),
            })
            .and_then(|bytes| match &plan {
                Q8ActPlan::Fused(fast) => q8act_rows(
                    ggml_type,
                    bytes,
                    rows,
                    n_in,
                    row_bytes,
                    rows * row_bytes,
                    q,
                    fast,
                ),
                Q8ActPlan::Reference => matvec_q8act(ggml_type, bytes, rows, n_in, q),
            });
        match res {
            Ok(v) => dst.copy_from_slice(&v),
            Err(e) => {
                let mut slot = first_err
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if slot.as_ref().is_none_or(|(k, _)| c < *k) {
                    *slot = Some((c, e));
                }
            }
        }
    });
    match first_err
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        Some((_, e)) => Err(e),
        None => Ok(out),
    }
}

/// The int8-activation projection over a Q32.32 activation `x`: `quant_act(x)` once, then
/// [`matvec_q8act`] over `chunks` contiguous row chunks on scoped threads.
///
/// This is the `KernelMode::Q8Act` row kernel the decode step's Q/K/V/O, FFN and LM-head
/// projections call. By the `KernelMode::Q8Act` definition it equals `matvec_fixed(W,
/// dequant(quant_act(x)))` bit for bit, and for `chunks <= 1` it is exactly `matvec_q8act` of the
/// quantized activation.
///
/// # Errors
///
/// [`MatvecError::ShapeMismatch`] when `x.len() != n_in`; otherwise the [`matvec_q8act`] refusal
/// for the payload, the quant family or an out-of-range accumulator — an activation whose block
/// scale times code leaves `i64` refuses with
/// [`MatvecError::Dequant`]`(`[`EngineError::FixedOverflow`]`)`.
///
/// # Complexity
///
/// `O(n_in)` for the one `quant_act` plus `O(n_out·n_in)` in checked integer multiply-adds, and
/// `O(n_in + n_out)` output/scratch.
pub(crate) fn matvec_q8act_chunked(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    x: &[i64],
    chunks: usize,
) -> Result<Vec<i64>, MatvecError> {
    if x.len() != n_in {
        return Err(MatvecError::ShapeMismatch { x: x.len(), n_in });
    }
    let q = quantize(x)?;
    matvec_q8act_quant_chunked(ggml_type, raw, n_out, n_in, &q, chunks)
}

/// A raw output pointer that may cross threads; each pool task writes only the row range it owns,
/// and the ranges partition the allocation, which makes the concurrent writes sound.
#[derive(Clone, Copy)]
struct RowPtr(*mut i64);
// SAFETY: `matvec_q8act_pair_chunked` gives every task a disjoint row range of one `Vec<i64>`.
unsafe impl Send for RowPtr {}
// SAFETY: as above; the tasks partition the allocation and never alias.
unsafe impl Sync for RowPtr {}

/// One projection's dispatchable row-chunk plan in [`matvec_q8act_pair_chunked`].
struct ProjPrep {
    /// The chunk count [`matvec_q8act_quant_chunked`] would form (`> 1`).
    actual: usize,
    /// The row stride.
    row_bytes: usize,
    /// The call's row-independent kernel choice.
    plan: Q8ActPlan,
}

/// The outer checks and row-independent plan of one projection, or `Ok(None)` when
/// [`matvec_q8act_quant_chunked`] would not form a chunked dispatch (`actual == 1` or no block
/// geometry) and the pair must delegate.
fn proj_prep(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    q: &QuantAct,
    chunks: usize,
) -> Result<Option<ProjPrep>, MatvecError> {
    let upper = if n_out == 0 { 1 } else { n_out };
    let actual = chunks.min(Q8ACT_MAX_CHUNKS).clamp(1, upper);
    if actual == 1 {
        return Ok(None);
    }
    let Some((qk, bb)) = quant::block_geom(ggml_type) else {
        return Ok(None);
    };
    let row_bytes = n_in
        .div_ceil(qk)
        .checked_mul(bb)
        .ok_or(MatvecError::LengthOverflow)?;
    let need = n_out
        .checked_mul(row_bytes)
        .ok_or(MatvecError::LengthOverflow)?;
    if raw.len() < need {
        return Err(MatvecError::ShortPayload {
            need,
            have: raw.len(),
        });
    }
    if q.len() != n_in {
        return Err(MatvecError::ShapeMismatch { x: q.len(), n_in });
    }
    Ok(Some(ProjPrep {
        actual,
        row_bytes,
        plan: q8act_plan(ggml_type, n_in, q),
    }))
}

/// A per-dispatch prepared row kernel: `(lo, hi, out)`, with the activation already folded in. The
/// `PQ2_0` LUT builds its lookup tables once here instead of once per task.
pub(crate) type PreparedRows<'a> =
    Box<dyn Fn(usize, usize, &mut [i64]) -> Result<(), MatvecError> + Sync + 'a>;

/// A caller-supplied per-projection row kernel for [`projections_q8act`]: the `PQ2_0` LUT passes one
/// so this module runs its rows without importing `kern`.
pub(crate) trait RowKernel: Sync {
    /// Build this projection's per-dispatch kernel for `q`, or `None` when the unprepared
    /// [`RowKernel::rows`] path should be used (an activation with no Q32.32 value, or a call whose
    /// exact bound needs the reference fallback).
    ///
    /// # Errors
    ///
    /// This projection's own preparation refusal.
    fn prepare<'a>(&'a self, q: &'a QuantAct) -> Result<Option<PreparedRows<'a>>, MatvecError>;

    /// Write rows `[lo, hi)` of this projection into `out` (`out.len() == hi - lo`), reading the
    /// shared quantized activation `q`. Rows are independent, so the write is a pure function of
    /// `q` and the weights.
    ///
    /// # Errors
    ///
    /// This projection's own refusal for `[lo, hi)`.
    fn rows(&self, q: &QuantAct, lo: usize, hi: usize, out: &mut [i64]) -> Result<(), MatvecError>;
}

/// The row kernel of one projection in a [`projections_q8act`] dispatch.
pub(crate) enum ProjSource<'a> {
    /// The chart Q8-activation row core over `raw`.
    Chart { ggml_type: u32, raw: &'a [u8] },
    /// A caller-supplied row kernel (the tile-aligned `PQ2_0` LUT).
    Rows(&'a dyn RowKernel),
}

/// One projection of a [`projections_q8act`] dispatch.
pub(crate) struct ProjSpec<'a> {
    /// Output rows.
    pub(crate) n_out: usize,
    /// A chunk start must be a multiple of this (`1` for the chart row core, the tile height for a
    /// tile-aligned kernel).
    pub(crate) row_align: usize,
    /// The row kernel.
    pub(crate) source: ProjSource<'a>,
}

/// The dispatch plan of one projection: a chunkable chart projection, a caller-supplied row kernel,
/// or the sequential delegation [`proj_prep`] returns `None` for.
enum GroupPrep {
    Chart(ProjPrep),
    Rows,
    Delegate,
}

/// One projection's `(rows_per_chunk, task_count)` under [`projections_q8act`].
///
/// The chart split is [`matvec_q8act_quant_chunked`]'s (`tasks = actual.max(2·width)`, capped at the
/// row count); the row count is then re-derived as `⌈n_out / per⌉`, so a `per` that does not divide
/// `n_out` still covers every row exactly once. A tile-aligned kernel rounds `per` up to
/// [`ProjSpec::row_align`], so every chunk start is a tile boundary.
fn proj_split(prep: &GroupPrep, p: &ProjSpec<'_>, chunks: usize, width: usize) -> (usize, usize) {
    match prep {
        GroupPrep::Chart(pp) => {
            let t = pp.actual.max(2 * width).min(p.n_out.max(1));
            let per = p.n_out.div_ceil(t);
            let tasks = if per == 0 { 1 } else { p.n_out.div_ceil(per) };
            (per, tasks)
        }
        GroupPrep::Rows => {
            let align = p.row_align.max(1);
            if p.n_out == 0 {
                return (align, 0);
            }
            let tiles = p.n_out.div_ceil(align);
            let t = chunks.min(2 * width).min(tiles).max(1);
            let per = tiles.div_ceil(t) * align;
            (per, p.n_out.div_ceil(per))
        }
        GroupPrep::Delegate => unreachable!("delegated groups returned before the split"),
    }
}

/// `n` projections over one activation, run as one pool dispatch whose tasks span the row ranges of
/// all `n` projections in order; each task calls its projection's row kernel — the chart row core,
/// or the caller's [`RowKernel`] for a tile-aligned LUT.
///
/// `x` is quantized once with [`quantize`] and the same [`QuantAct`] is read by every task. A chart
/// projection uses [`matvec_q8act_quant_chunked`]'s row split and shared plan, so the concatenation
/// is that call bit for bit; a tile-aligned kernel keeps every chunk start on a [`ProjSpec::row_align`]
/// boundary. When a chart projection would not form a chunked dispatch, or its outer check refuses,
/// the group falls back to one sequential call per projection in order, which reproduces the
/// original results and first-refusal order exactly.
///
/// # Errors
///
/// The first refusal in projection-then-row order: the same value `n` separate
/// [`matvec_q8act_quant_chunked`] / [`RowKernel::rows`] calls joined left to right would return.
///
/// # Complexity
///
/// `O(Σ n_out_i·n_in)` time in checked multiply-adds with one `quant_act` instead of `n`, and
/// `O(Σ n_out_i)` output, in one synchronization instead of `n`.
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "the row-groups and their disjoint output pointers are the dispatch's full state; the \
              index names mirror the factorization"
)]
pub(crate) fn projections_q8act(
    n_in: usize,
    x: &[i64],
    projs: &[ProjSpec<'_>],
    chunks: usize,
) -> Result<Vec<Vec<i64>>, MatvecError> {
    if x.len() != n_in {
        return Err(MatvecError::ShapeMismatch { x: x.len(), n_in });
    }
    let q = quantize(x)?;
    projections_q8act_prepared(n_in, &q, projs, chunks)
}

/// [`projections_q8act`] over an already-quantized activation: the same dispatch, without the
/// `quant_act`. `q` must be [`quant_act_pooled`]'s or [`quant_act`]'s result for the activation.
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "the row-groups and their disjoint output pointers are the dispatch's full state; the \
              index names mirror the factorization"
)]
pub(crate) fn projections_q8act_prepared(
    n_in: usize,
    q: &QuantAct,
    projs: &[ProjSpec<'_>],
    chunks: usize,
) -> Result<Vec<Vec<i64>>, MatvecError> {
    let mut outs: Vec<Vec<i64>> = Vec::new();
    projections_q8act_prepared_into(n_in, q, projs, chunks, &mut outs)?;
    Ok(outs)
}

/// [`projections_q8act_prepared`] writing into caller-owned row buffers.
///
/// `outs` is resized to one vector per projection of `n_out` rows; a resize is a no-op when the
/// caller hands back the same buffers, so a decode step that composes the same widths every layer
/// pays no allocate-and-zero per layer. Every row is written before it is read, so a reused buffer
/// carries no stale value into the result.
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "the row-groups and their disjoint output pointers are the dispatch's full state; the \
              index names mirror the factorization"
)]
pub(crate) fn projections_q8act_prepared_into(
    n_in: usize,
    q: &QuantAct,
    projs: &[ProjSpec<'_>],
    chunks: usize,
    outs: &mut Vec<Vec<i64>>,
) -> Result<(), MatvecError> {
    if q.len() != n_in {
        return Err(MatvecError::ShapeMismatch { x: q.len(), n_in });
    }
    if outs.len() != projs.len() {
        outs.resize_with(projs.len(), Vec::new);
    }
    for (o, p) in outs.iter_mut().zip(projs) {
        if o.len() != p.n_out {
            o.resize(p.n_out, 0);
        }
    }
    let mut preps: Vec<GroupPrep> = Vec::with_capacity(projs.len());
    for p in projs {
        match &p.source {
            ProjSource::Chart { ggml_type, raw } => {
                match proj_prep(*ggml_type, raw, p.n_out, n_in, q, chunks)? {
                    Some(pp) => preps.push(GroupPrep::Chart(pp)),
                    None => preps.push(GroupPrep::Delegate),
                }
            }
            ProjSource::Rows(_) => preps.push(GroupPrep::Rows),
        }
    }
    // One chart projection that cannot chunk (or refuses) sends the whole group to the sequential
    // calls, whose join order is the first refusal order.
    if preps.iter().any(|p| matches!(p, GroupPrep::Delegate)) {
        for (i, p) in projs.iter().enumerate() {
            match &p.source {
                ProjSource::Chart { ggml_type, raw } => {
                    let v = matvec_q8act_quant_chunked(*ggml_type, raw, p.n_out, n_in, q, chunks)?;
                    outs[i].copy_from_slice(&v);
                }
                ProjSource::Rows(k) => k.rows(q, 0, p.n_out, &mut outs[i])?,
            }
        }
        return Ok(());
    }
    // The LUT's activation tables are per-dispatch, not per-task: build them once before the run.
    let prepared: Vec<Option<PreparedRows<'_>>> = projs
        .iter()
        .map(|p| match &p.source {
            ProjSource::Rows(k) => k.prepare(q),
            ProjSource::Chart { .. } => Ok(None),
        })
        .collect::<Result<Vec<_>, MatvecError>>()?;
    let pool = Pool::global();
    let width = pool.width();
    let split: Vec<(usize, usize)> = projs
        .iter()
        .zip(&preps)
        .map(|(p, prep)| proj_split(prep, p, chunks, width))
        .collect();
    let mut base: Vec<usize> = Vec::with_capacity(projs.len());
    let mut total = 0usize;
    for &(_, t) in &split {
        base.push(total);
        total += t;
    }
    let ptrs: Vec<RowPtr> = outs.iter_mut().map(|v| RowPtr(v.as_mut_ptr())).collect();
    let first_err: Mutex<Option<(usize, MatvecError)>> = Mutex::new(None);
    pool.run(total, &|t| {
        // `split` is ordered by projection, so the first element with `base <= t` is `t`'s
        // projection and the task order within it is ascending rows.
        let i = base.partition_point(|&b| b <= t) - 1;
        let c = t - base[i];
        let lo = c * split[i].0;
        let hi = (lo + split[i].0).min(projs[i].n_out);
        let rows = hi - lo;
        // SAFETY: task `t` owns `[lo, hi)` of projection `i`; the ranges partition every output and
        // the `Vec`s outlive the dispatch.
        let dst = unsafe { std::slice::from_raw_parts_mut(ptrs[i].0.add(lo), rows) };
        let res: Result<(), MatvecError> = match (&projs[i].source, &preps[i]) {
            (ProjSource::Chart { ggml_type, raw }, GroupPrep::Chart(pp)) => raw
                .get(lo * pp.row_bytes..hi * pp.row_bytes)
                .ok_or(MatvecError::ShortPayload {
                    need: rows * pp.row_bytes,
                    have: raw.len(),
                })
                .and_then(|bytes| match &pp.plan {
                    Q8ActPlan::Fused(fast) => q8act_rows(
                        *ggml_type,
                        bytes,
                        rows,
                        n_in,
                        pp.row_bytes,
                        rows * pp.row_bytes,
                        q,
                        fast,
                    ),
                    Q8ActPlan::Reference => matvec_q8act(*ggml_type, bytes, rows, n_in, q),
                })
                .map(|v| dst.copy_from_slice(&v)),
            (ProjSource::Rows(k), _) => match &prepared[i] {
                Some(f) => f(lo, hi, dst),
                None => k.rows(q, lo, hi, dst),
            },
            _ => unreachable!("delegated groups returned before the dispatch"),
        };
        if let Err(e) = res {
            let mut slot = first_err
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if slot.as_ref().is_none_or(|(k, _)| t < *k) {
                *slot = Some((t, e));
            }
        }
    });
    match first_err
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        Some((_, e)) => Err(e),
        None => Ok(()),
    }
}

/// Run one layer's query heads into `out`, split `hd` wide, calling `head(h, dst)` per head.
///
/// `parallel` dispatches the heads on the persistent [`Pool`] over the disjoint `out` slices;
/// otherwise the serial per-head loop runs. Both schedules call `head` with the same head index and
/// the same output slice, so their results are equal and the serial loop is the test oracle. Returns
/// the first refusal in head order (the pool path keeps the smallest head index among refusals),
/// matching the serial loop's first error.
///
/// # Complexity
///
/// `O(Σ_h cost(head(h)))` plus `O(n_head)` scheduling; no allocation and no copy of `out`.
pub(crate) fn attn_heads<E, F>(
    n_head: usize,
    hd: usize,
    out: &mut [i64],
    parallel: bool,
    head: F,
) -> Option<(usize, E)>
where
    F: Fn(usize, &mut [i64]) -> Result<(), E> + Sync,
    E: Send,
{
    if !parallel {
        let mut h = 0usize;
        while h < n_head {
            let lo = h * hd;
            if let Err(e) = head(h, &mut out[lo..lo + hd]) {
                return Some((h, e));
            }
            h += 1;
        }
        return None;
    }
    let pool = Pool::global();
    let first: Mutex<Option<(usize, E)>> = Mutex::new(None);
    pool.chunks_mut(out, hd, &|c, _start, dst| {
        if let Err(e) = head(c, dst) {
            let mut slot = first
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if slot.as_ref().is_none_or(|(k, _)| c < *k) {
                *slot = Some((c, e));
            }
        }
    });
    first
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// One token of the dense FFN (`SwiGLU`) block on the int8-activation chart path: `ffn_fixed`'s
/// arithmetic with each projection replaced by `matvec_q8act` on `quant_act(its input)`.
///
/// `gate`/`up` share one `quant_act(x)`; the elementwise `round(silu(gate_i)·up_i / 2^32)` pass is
/// [`ffn_fixed`](crate::engine::ffn::ffn_fixed)'s, and the down projection is `matvec_q8act` on
/// `quant_act(h)`. The gate is the `SiLU` argument and the `up` row the multiplier, as in
/// `ffn_fixed`. Each projection splits its rows by [`row_chunk_budget`], so the block uses the
/// machine's parallelism as `ffn_fixed`'s does.
///
/// # Errors
///
/// [`FfnError::BadGeometry`] for a zero `n_ff` or an empty `x`, [`FfnError::Matvec`] for a
/// projection shape, payload, quant or accumulator refusal, and [`FfnError::Overflow`] when an
/// elementwise product or a narrowing leaves its range. Gate logits saturate inside [`silu_fixed`];
/// no input is clamped.
///
/// # Complexity
///
/// `O(n_ff·n_embd)` time in checked integer multiply-adds over the three projections plus
/// `O(n_ff)` for the elementwise pass and the quantizations, `O(n_ff + n_embd)` scratch for the
/// gate, up and hidden vectors and their int8 activation codes, `O(n_embd)` for the output, and
/// `O(chunks)` scoped-thread handles owned by the chunked kernel; no weight payload is copied.
pub(crate) fn ffn_q8act(
    gate: ProjWeights<'_>,
    up: ProjWeights<'_>,
    down: ProjWeights<'_>,
    n_ff: usize,
    x: &[i64],
) -> Result<Vec<i64>, FfnError> {
    let n_embd = x.len();
    let g = ProjSpec {
        n_out: n_ff,
        row_align: 1,
        source: ProjSource::Chart {
            ggml_type: gate.ggml_type,
            raw: gate.raw,
        },
    };
    let u = ProjSpec {
        n_out: n_ff,
        row_align: 1,
        source: ProjSource::Chart {
            ggml_type: up.ggml_type,
            raw: up.raw,
        },
    };
    let d = ProjSpec {
        n_out: n_embd,
        row_align: 1,
        source: ProjSource::Chart {
            ggml_type: down.ggml_type,
            raw: down.raw,
        },
    };
    ffn_q8act_prepared(
        n_embd,
        x,
        g,
        u,
        d,
        n_ff,
        row_chunk_budget(n_ff),
        row_chunk_budget(n_embd),
    )
}

/// [`ffn_q8act`]'s composition over caller-built [`ProjSpec`]s: one pooled `quant_act` of `x`, the
/// gate/up pair in one [`projections_q8act_prepared`] dispatch, [`swiglu_hidden`], then the down
/// spec over a pooled `quant_act` of the hidden. A spec carries either the chart row core or a
/// caller's [`RowKernel`], so a `PQ2_0` lookup table runs its rows here without this module
/// importing `kern`.
///
/// # Errors
///
/// As [`ffn_q8act`]: [`FfnError::BadGeometry`] for a zero `n_ff`, an empty `x` or an `x` whose
/// length is not `n_in`; the projections' refusals; and [`FfnError::Overflow`] from the elementwise
/// pass.
#[allow(
    clippy::too_many_arguments,
    reason = "the three row kernels, their activation and the two dispatch widths are the \
              composition's full input"
)]
pub(crate) fn ffn_q8act_prepared(
    n_in: usize,
    x: &[i64],
    gate: ProjSpec<'_>,
    up: ProjSpec<'_>,
    down: ProjSpec<'_>,
    n_ff: usize,
    chunks_gate_up: usize,
    chunks_down: usize,
) -> Result<Vec<i64>, FfnError> {
    if n_ff == 0 || n_in == 0 || x.len() != n_in {
        return Err(FfnError::BadGeometry);
    }
    // Gate and up read the same activation and the same geometry, so both row sweeps are one pool
    // dispatch with one `quant_act(x)`; only their weight payloads differ.
    let q = quantize(x)?;
    let mut out = projections_q8act_prepared(n_in, &q, &[gate, up], chunks_gate_up)?;
    let up_vec = out.pop().expect("two projections");
    let gate_vec = out.pop().expect("two projections");
    let mut h: Vec<i64> = vec![0; n_ff];
    swiglu_hidden(&gate_vec, &up_vec, &mut h, true)?;
    let qh = quantize(&h)?;
    let mut d = projections_q8act_prepared(n_ff, &qh, &[down], chunks_down)?;
    Ok(d.pop().expect("one projection"))
}

/// Reusable row and hidden buffers for [`ffn_q8act_prepared_into`].
///
/// A decode step composes the same FFN width on every layer; the gate/up row outputs and the
/// `SwiGLU` hidden are `n_ff`-wide scratch whose allocate-and-zero fill is a per-layer serial cost
/// with no arithmetic. Holding them across layers leaves the steady state allocation-free. Every
/// element is written before it is read, so a reused buffer carries nothing into the result.
#[derive(Default)]
pub(crate) struct FfnScratch {
    /// The gate and up row outputs, indices 0 and 1.
    rows: Vec<Vec<i64>>,
    /// The `SwiGLU` hidden vector.
    hidden: Vec<i64>,
    /// The down projection's row output, taken by each call and re-formed empty.
    down_rows: Vec<Vec<i64>>,
}

/// [`ffn_q8act_prepared`] with caller-owned scratch buffers (`scratch`), returning the down
/// projection's rows.
///
/// The composition is unchanged: one pooled `quant_act` of `x`, the gate/up pair in one
/// [`projections_q8act_prepared_into`] dispatch, [`swiglu_hidden`], and the down spec over a pooled
/// `quant_act` of the hidden. Only the buffers are reused.
#[allow(
    clippy::too_many_arguments,
    reason = "the three row kernels, their activation, the two dispatch widths and the scratch \
              buffers are the composition's full input"
)]
pub(crate) fn ffn_q8act_prepared_into(
    n_in: usize,
    x: &[i64],
    gate: ProjSpec<'_>,
    up: ProjSpec<'_>,
    down: ProjSpec<'_>,
    n_ff: usize,
    chunks_gate_up: usize,
    chunks_down: usize,
    scratch: &mut FfnScratch,
) -> Result<Vec<i64>, FfnError> {
    if n_ff == 0 || n_in == 0 || x.len() != n_in {
        return Err(FfnError::BadGeometry);
    }
    let q = quantize(x)?;
    projections_q8act_prepared_into(n_in, &q, &[gate, up], chunks_gate_up, &mut scratch.rows)?;
    if scratch.hidden.len() != n_ff {
        scratch.hidden.resize(n_ff, 0);
    }
    swiglu_hidden(
        &scratch.rows[0],
        &scratch.rows[1],
        &mut scratch.hidden,
        true,
    )?;
    let qh = quantize(&scratch.hidden)?;
    if scratch.down_rows.is_empty() {
        scratch.down_rows.push(Vec::new());
    }
    projections_q8act_prepared_into(n_ff, &qh, &[down], chunks_down, &mut scratch.down_rows)?;
    Ok(std::mem::take(&mut scratch.down_rows[0]))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::many_single_char_names,
        clippy::needless_range_loop,
        reason = "test-only fixture/reference arithmetic on bounded small values; not extracted"
    )]

    use super::*;
    use crate::engine::matvec::matvec_fixed;
    use crate::engine::ONE;
    use crate::format::quant::{GGML_TYPE_F16, GGML_TYPE_Q8_0};

    #[test]
    fn swiglu_hidden_cols_matches_serial_columns_across_shapes() {
        for k in [1usize, 2, 3, 64, 129] {
            for n in [7usize, 31, 97] {
                let gate: Vec<Vec<i64>> = (0..k)
                    .map(|c| {
                        (0..n)
                            .map(|i| match (c + i) % 5 {
                                0 => -ONE,
                                1 => ONE,
                                2 => 0,
                                3 => 4 * ONE,
                                _ => -4 * ONE,
                            })
                            .collect()
                    })
                    .collect();
                let up: Vec<Vec<i64>> = (0..k)
                    .map(|c| {
                        (0..n)
                            .map(|i| match (c + i) % 5 {
                                0 => 2 * ONE,
                                2 => {
                                    if i % 2 == 0 {
                                        i64::MIN
                                    } else {
                                        i64::MAX
                                    }
                                }
                                3 => ONE,
                                _ => -ONE,
                            })
                            .collect()
                    })
                    .collect();
                let mut expected = vec![vec![0; n]; k];
                for c in 0..k {
                    swiglu_hidden(&gate[c], &up[c], &mut expected[c], false)
                        .expect("serial column succeeds");
                }

                for parallel in [false, true] {
                    assert_eq!(
                        swiglu_hidden_cols(&gate, &up, parallel),
                        Ok(expected.clone()),
                        "k={k}, n={n}, parallel={parallel}"
                    );
                }
            }
        }
    }

    #[test]
    fn swiglu_hidden_cols_checks_empty_and_malformed_geometry_upfront() {
        assert_eq!(swiglu_hidden_cols(&[], &[], true), Ok(vec![]));

        let empty_cols = vec![vec![], vec![]];
        assert_eq!(
            swiglu_hidden_cols(&empty_cols, &empty_cols, true),
            Ok(empty_cols.clone())
        );

        let gate = vec![vec![ONE; 3], vec![ONE; 2]];
        let up = vec![vec![ONE; 3], vec![ONE; 3]];
        assert_eq!(
            swiglu_hidden_cols(&gate, &up, true),
            Err(FfnError::BadGeometry)
        );

        let rect = vec![vec![ONE; 3], vec![ONE; 3]];
        let short_up = vec![vec![ONE; 3]];
        assert_eq!(
            swiglu_hidden_cols(&rect, &short_up, true),
            Err(FfnError::BadGeometry)
        );
        let ragged_up = vec![vec![ONE; 3], vec![ONE; 2]];
        assert_eq!(
            swiglu_hidden_cols(&rect, &ragged_up, true),
            Err(FfnError::BadGeometry)
        );
    }

    #[test]
    fn swiglu_hidden_cols_matches_overflow_refusal_without_publishing_output() {
        let k = 129;
        let n = 7;
        let mut gate = vec![vec![0; n]; k];
        let mut up = vec![vec![ONE; n]; k];
        gate[0][n - 1] = 2 * ONE;
        up[0][n - 1] = i64::MAX;
        gate[1][0] = 2 * ONE;
        up[1][0] = i64::MAX;
        let expected_error = FfnError::Overflow;
        let mut seen_error = None;
        for c in 0..k {
            let mut col = vec![0; n];
            if let Err(error) = swiglu_hidden(&gate[c], &up[c], &mut col, false) {
                seen_error = Some(error);
                break;
            }
        }
        assert_eq!(seen_error, Some(expected_error.clone()));

        assert_eq!(swiglu_hidden_cols(&gate, &up, true), Err(expected_error));
    }

    /// Finite `f16` bit patterns the fixture draws from: `0`, `±0.125`, `±0.25`, `±0.5`, `±1`,
    /// `±2` and `±4`.
    const PALETTE: [u16; 13] = [
        0x0000, 0x3000, 0xB000, 0x3400, 0xB400, 0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000,
        0x4200, 0xC200,
    ];

    /// A deterministic pseudo-random `u64` (Knuth's LCG), test-only.
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    /// A row-major `F16` payload of `rows × cols` drawn from the palette.
    fn f16_raw(rows: usize, cols: usize, salt: u64) -> Vec<u8> {
        let mut state = salt;
        let mut out: Vec<u8> = Vec::with_capacity(rows * cols * 2);
        let mut i = 0usize;
        while i < rows * cols {
            let idx = usize::try_from(lcg(&mut state) % 13).expect("index < 13");
            out.extend_from_slice(&PALETTE[idx].to_le_bytes());
            i += 1;
        }
        out
    }

    /// A `Q8_0` payload of `rows` rows over `cols` (a multiple of 32): every block scale is `1.0`
    /// and the codes lie in `[-10, 10]`.
    fn q8_rows(rows: usize, cols: usize) -> Vec<u8> {
        let mut b: Vec<u8> = Vec::with_capacity(rows * cols.div_ceil(32) * 34);
        let mut r = 0usize;
        while r < rows {
            let blocks = cols.div_ceil(32);
            let mut blk = 0usize;
            while blk < blocks {
                b.extend_from_slice(&0x3C00u16.to_le_bytes());
                let mut j = 0usize;
                while j < 32 {
                    let v = i32::try_from((r * 7 + blk * 5 + j * 3) % 21).expect("small") - 10;
                    b.push((v as i8) as u8);
                    j += 1;
                }
                blk += 1;
            }
            r += 1;
        }
        b
    }

    /// A Q32.32 activation with `±1/4` and `0` entries and a non-power-of-two magnitude, so the
    /// quantizer's scale is not trivial.
    fn activation(n: usize, salt: i64) -> Vec<i64> {
        let mut x: Vec<i64> = Vec::with_capacity(n);
        let mut i = 0usize;
        while i < n {
            let j = i64::try_from(i).expect("small") * 3 + salt;
            x.push((j.rem_euclid(7) - 3) * (ONE / 8));
            i += 1;
        }
        x
    }

    /// The chunked Q8-activation projection equals the unchunked `matvec_q8act` for every chunk
    /// budget, including budgets that do not divide the row count and one row per chunk, on a
    /// fused family (`Q8_0`) and the non-fused `F16` fallback.
    #[test]
    fn q8act_chunked_equals_the_unchunked_kernel() {
        const ROWS: usize = 10;
        const N_IN: usize = 64;
        let x = activation(N_IN, 0);
        let q = quant_act(&x).expect("quantizes");
        for (ty, raw) in [
            (GGML_TYPE_Q8_0, q8_rows(ROWS, N_IN)),
            (GGML_TYPE_F16, f16_raw(ROWS, N_IN, 0x5EED)),
        ] {
            let want = matvec_q8act(ty, &raw, ROWS, N_IN, &q).expect("unchunked");
            for chunks in [1usize, 2, 7, ROWS] {
                let got = matvec_q8act_chunked(ty, &raw, ROWS, N_IN, &x, chunks)
                    .unwrap_or_else(|e| panic!("type {ty} chunks {chunks}: {e:?}"));
                assert_eq!(got, want, "type {ty}, chunks {chunks}");
            }
        }
    }

    /// The chunked kernel refuses a shape mismatch before touching the payload.
    #[test]
    fn q8act_chunked_refuses_a_shape_mismatch() {
        let raw = q8_rows(4, 32);
        let got = matvec_q8act_chunked(GGML_TYPE_Q8_0, &raw, 4, 32, &activation(16, 1), 2);
        assert!(matches!(
            got,
            Err(MatvecError::ShapeMismatch { x: 16, n_in: 32 })
        ));
    }

    /// The pooled quantization equals the single [`quant_act`] call at and across block and chunk
    /// boundaries, and both refuse the same activations.
    #[test]
    fn quant_act_pooled_equals_the_single_quantization() {
        for n in [0usize, 1, 31, 32, 33, 64, 127, 128, 129, 2000, 6144] {
            let x = activation(n, 7);
            let single = quant_act(&x);
            let pooled = quant_act_pooled(&x);
            match (&single, &pooled) {
                (Some(w), Some(g)) => {
                    assert_eq!(w.scales(), g.scales(), "scales at n = {n}");
                    assert_eq!(w.codes(), g.codes(), "codes at n = {n}");
                }
                (None, None) => {}
                _ => panic!(
                    "n = {n}: single {}, pooled {}",
                    single.is_some(),
                    pooled.is_some()
                ),
            }
        }
    }

    /// The N-projection dispatch equals N separate unchunked `matvec_q8act` calls bit for bit on the
    /// `Q8_0` fused plan and the `F16` reference plan, across chunk budgets including the
    /// `chunks <= 1` delegation.
    #[test]
    fn projections_q8act_equals_separate_unchunked_calls() {
        const ROWS: usize = 40;
        const N_IN: usize = 64;
        let x = activation(N_IN, 5);
        let q = quant_act(&x).expect("quantizes");
        let q8 = q8_rows(ROWS, N_IN);
        let f16 = f16_raw(ROWS, N_IN, 0x77);
        let specs = [
            ProjSpec {
                n_out: ROWS,
                row_align: 1,
                source: ProjSource::Chart {
                    ggml_type: GGML_TYPE_Q8_0,
                    raw: &q8,
                },
            },
            ProjSpec {
                n_out: ROWS,
                row_align: 1,
                source: ProjSource::Chart {
                    ggml_type: GGML_TYPE_F16,
                    raw: &f16,
                },
            },
        ];
        let want = [
            matvec_q8act(GGML_TYPE_Q8_0, &q8, ROWS, N_IN, &q).expect("q8"),
            matvec_q8act(GGML_TYPE_F16, &f16, ROWS, N_IN, &q).expect("f16"),
        ];
        for chunks in [1usize, 2, 8] {
            let got = projections_q8act(N_IN, &x, &specs, chunks).expect("dispatch");
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(g, w, "projection {i}, chunks {chunks}");
            }
        }
    }

    /// The chart FFN equals the independent reference: `matvec_fixed` on `dequant(quant_act(·))`
    /// per projection and the same elementwise `round(silu(gate)·up / 2^32)`, bit for bit. The
    /// hidden width makes the gate/up/down projections chunk (`row_chunk_budget(128) >= 2` on a
    /// multi-core host), and `F16` exercises the non-fused fallback in `matvec_q8act`.
    #[test]
    fn ffn_q8act_equals_the_dequant_reference() {
        const N_EMBD: usize = 32;
        const N_FF: usize = 128;
        let x = activation(N_EMBD, 3);
        let g = f16_raw(N_FF, N_EMBD, 0xA1);
        let u = f16_raw(N_FF, N_EMBD, 0xB2);
        let d = f16_raw(N_EMBD, N_FF, 0xC3);
        let weights = || {
            (
                ProjWeights {
                    ggml_type: GGML_TYPE_F16,
                    raw: &g,
                },
                ProjWeights {
                    ggml_type: GGML_TYPE_F16,
                    raw: &u,
                },
                ProjWeights {
                    ggml_type: GGML_TYPE_F16,
                    raw: &d,
                },
            )
        };
        // Reference: one `matvec_fixed` per projection, each on the dequantized quantized input.
        let qx = quant_act(&x).expect("quantizes");
        let xh = qx.dequant().expect("dequantizes");
        let gate_vec = matvec_fixed(GGML_TYPE_F16, &g, N_FF, N_EMBD, &xh).expect("gate");
        let up_vec = matvec_fixed(GGML_TYPE_F16, &u, N_FF, N_EMBD, &xh).expect("up");
        let mut h: Vec<i64> = Vec::with_capacity(N_FF);
        for i in 0..N_FF {
            let s = silu_fixed(gate_vec[i]).expect("silu");
            let prod = i128::from(s)
                .checked_mul(i128::from(up_vec[i]))
                .expect("in range");
            h.push(i64::try_from(round_shift_32(prod)).expect("in range"));
        }
        let qh = quant_act(&h).expect("quantizes");
        let hh = qh.dequant().expect("dequantizes");
        let want = matvec_fixed(GGML_TYPE_F16, &d, N_EMBD, N_FF, &hh).expect("down");
        let (wg, wu, wd) = weights();
        assert_eq!(ffn_q8act(wg, wu, wd, N_FF, &x).expect("ffn_q8act"), want);
    }
}
