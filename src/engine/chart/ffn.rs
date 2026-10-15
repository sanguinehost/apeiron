//! Row-chunked Q8-activation projections and the chart-native FFN: the `KernelMode::Q8Act`
//! executing path.
//!
//! [`matvec_q8act_chunked`] is [`super::matvec_q8act`] over whole rows split into contiguous row
//! chunks run on scoped threads, concatenated in row order; it is the Q8-activation twin of
//! `par::matvec_rows_chunked`. [`projections_q8act`] generalizes its gate/up fusion to `n`
//! projections over one `quant_act(x)` and one pool dispatch, taking either the chart row core or a
//! [`Pq2Lut`] so the `PQ2_0` LUT can share the activation. [`ffn_q8act`] is [`ffn_fixed`](crate::engine::ffn::ffn_fixed)'s arithmetic with
//! each projection replaced by `matvec_q8act` over `quant_act` of its input; the gate and up
//! projections share one `quant_act` of `x`. The `Exact` path in `ffn.rs` and `par.rs` is not
//! touched.

#[cfg(target_arch = "x86_64")]
use crate::engine::fp::F32;
use crate::engine::mixer::f32_carrier::silu_f32_avx2;
use crate::engine::mixer::f32_carrier::{q32_to_f32, silu_f32_scalar};
use crate::engine::mixer::{carrier_from_env, Carrier};
use crate::format::quant;

use super::super::ffn::{silu_fixed, FfnError, ProjWeights};
use super::super::matvec::MatvecError;
use super::super::par::row_chunk_budget;
use super::super::value::round_shift_32;
use super::super::EngineError;
use super::{matvec_q8act, q8act_plan, q8act_rows, quant_act, Q8ActPlan, QuantAct};
use crate::engine::chart::task::{ChunkTask, HeadTask, IndexTask, Task1, VecTask};
use crate::engine::kern::lut::{Pq2Lut, PreparedLut};
use crate::engine::loader::PayloadRef;

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

impl ActPtr {
    #[inline]
    unsafe fn write_scales(&self, offset: usize, src: *const i64, count: usize) {
        std::ptr::copy_nonoverlapping(src, self.scales.add(offset), count);
    }

    #[inline]
    unsafe fn write_codes(&self, offset: usize, src: *const i8, count: usize) {
        std::ptr::copy_nonoverlapping(src, self.codes.add(offset), count);
    }
}

/// `dst[off + i] = src[i]` over `src`'s length, for a scale run.
fn copy_i64_at(dst: &mut [i64], off: usize, src: &[i64]) {
    let mut i = 0usize;
    while i < src.len() && off + i < dst.len() {
        dst[off + i] = src[i];
        i += 1;
    }
}

/// `dst[off + i] = src[i]` over `src`'s length, for a code run.
fn copy_i8_at(dst: &mut [i8], off: usize, src: &[i8]) {
    let mut i = 0usize;
    while i < src.len() && off + i < dst.len() {
        dst[off + i] = src[i];
        i += 1;
    }
}

/// [`quant_act`] of one block run into `out`; `false` is the [`quantize`] refusal, which the pooled
/// callers reduce to `None`. Takes one `&mut` output so the extraction sees no `&mut [i64]` and
/// `&mut [i8]` pair in a definition.
fn quant_fill(x: &[i64], out: &mut QuantAct) -> bool {
    match quant_act(x) {
        Some(qa) => {
            *out = qa;
            true
        }
        None => false,
    }
}

/// Pool context for [`quant_act_pooled`]: the activation and the codes each task quantizes.
struct QuantSlotsCtx<'a> {
    x: &'a [i64],
    per_codes: usize,
}

impl Task1<QuantAct, ()> for QuantSlotsCtx<'_> {
    fn run(&self, t: usize, slot: &mut [QuantAct]) -> Result<(), ()> {
        quant_slot_task(self.x, self.per_codes, t, slot)
    }
}

/// Task body of [`QuantSlotsCtx`]: quantizes the codes of task `t` into `slot[0]`.
fn quant_slot_task(x: &[i64], per_codes: usize, t: usize, slot: &mut [QuantAct]) -> Result<(), ()> {
    let lo = t * per_codes;
    let hi = (lo + per_codes).min(x.len());
    if quant_fill(&x[lo..hi], &mut slot[0]) {
        Ok(())
    } else {
        Err(())
    }
}

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
    let tasks = (2 * super::pool::width_global()).min(blocks).max(1);
    if tasks <= 1 {
        return quant_act(x);
    }
    let per = blocks.div_ceil(tasks);
    let per_codes = per * QK;
    let n_tasks = blocks.div_ceil(per);
    let mut slots: Vec<QuantAct> = Vec::with_capacity(n_tasks);
    let mut t = 0usize;
    while t < n_tasks {
        slots.push(QuantAct {
            scales: Vec::new(),
            codes: Vec::new(),
        });
        t += 1;
    }
    let refused = super::pool::for_chunks(&mut slots, 1, &QuantSlotsCtx { x, per_codes });
    if refused.is_err() {
        return None;
    }
    let mut scales: Vec<i64> = vec![0; blocks];
    let mut codes: Vec<i8> = vec![0; x.len()];
    let mut t = 0usize;
    while t < n_tasks {
        copy_i64_at(&mut scales, t * per, &slots[t].scales);
        copy_i8_at(&mut codes, t * per_codes, &slots[t].codes);
        t += 1;
    }
    Some(QuantAct { scales, codes })
}

/// Pool context for [`quant_acts_pooled`]: the columns; task `c` quantizes column `c`.
struct QuantColsCtx<'a> {
    xs: &'a [Vec<i64>],
}

impl Task1<QuantAct, ()> for QuantColsCtx<'_> {
    fn run(&self, c: usize, slot: &mut [QuantAct]) -> Result<(), ()> {
        quant_col_task(self.xs, c, slot)
    }
}

/// Task body of [`QuantColsCtx`]: quantizes column `c` into `slot[0]`.
fn quant_col_task(xs: &[Vec<i64>], c: usize, slot: &mut [QuantAct]) -> Result<(), ()> {
    if quant_fill(&xs[c], &mut slot[0]) {
        Ok(())
    } else {
        Err(())
    }
}

/// [`quant_act`] over every column at once: one pool task per column, each running the unchanged
/// single-column quantizer and copying its `scales`/`codes` into column `c`'s reservation of the
/// result. A block's scale and codes depend only on that column's own 32-element block, so the
/// batch is the per-column [`quant_act_pooled`] sequence bit for bit: same scales, same codes, and
/// a column's refusal is the same `None`. `xs.len() <= 1` and columns of differing length take the
/// per-column path. `k` is a prefill block width (`<= 64`), one dispatch instead of one per column.
///
/// # Complexity
///
/// `O(Σ_c xs[c].len())` time in the same two passes as [`quant_act`], split over `k` tasks in one
/// dispatch; `O(Σ_c xs[c].len())` output in `2k` allocations.
pub(crate) fn quant_acts_pooled(xs: &[Vec<i64>]) -> Option<Vec<QuantAct>> {
    /// The activation quantization block, as in [`quant_act`].
    const QK: usize = 32;
    let k = xs.len();
    let n = if xs.is_empty() { 0 } else { xs[0].len() };
    if k <= 1 || crate::engine::step::any_vec_len_ne(xs, n) {
        let mut out: Vec<QuantAct> = Vec::with_capacity(k);
        let mut i = 0usize;
        while i < k {
            out.push(quant_act_pooled(&xs[i])?);
            i += 1;
        }
        return Some(out);
    }
    let per_scales = n.div_ceil(QK);
    let mut out: Vec<QuantAct> = Vec::with_capacity(k);
    let mut col = 0usize;
    while col < k {
        out.push(QuantAct {
            scales: vec![0; per_scales],
            codes: vec![0; n],
        });
        col += 1;
    }
    // As `quant_act_pooled`: a refusal of any column is the whole result's `None`.
    let refused = super::pool::for_chunks(&mut out, 1, &QuantColsCtx { xs });
    if refused.is_err() {
        return None;
    }
    Some(out)
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

/// Convert an f32 readout value `o` back to Q32.32 fixed point `i64`.
///
/// Computes `round(o · 2^32)` using round-to-nearest.
/// Range check: non-finite (NaN, $\pm\infty$) or values exceeding the signed 64-bit integer
/// range $[-2^{63}, 2^{63}-1]$ return `Err(FfnError::Overflow)`. No silent clamping is performed.
#[inline]
pub(crate) fn f32_to_q32_checked(o: F32) -> Result<i64, FfnError> {
    if !o.is_finite() {
        return Err(FfnError::Overflow);
    }
    o.to_q32().ok_or(FfnError::Overflow)
}

/// Scalar `SwiGLU` in binary32: `SiLU(q32_to_f32(gate)) * q32_to_f32(up)` converted back to Q32.32.
#[inline]
pub(crate) fn swiglu_one_f32_scalar(gate: i64, up: i64) -> Result<i64, FfnError> {
    let g = q32_to_f32(gate);
    let u = q32_to_f32(up);
    let s = silu_f32_scalar(g);
    let prod = s * u;
    f32_to_q32_checked(prod)
}

/// Evaluate one `SwiGLU` element under the selected carrier.
#[inline]
pub(crate) fn swiglu_one_carrier(gate: i64, up: i64, carrier: Carrier) -> Result<i64, FfnError> {
    match carrier {
        Carrier::Q32 => swiglu_one(gate, up),
        Carrier::F32 => swiglu_one_f32_scalar(gate, up),
    }
}

/// AVX2 implementation of 8-lane `SwiGLU` in binary32.
/// Matches `swiglu_one_f32_scalar` bit for bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn swiglu_f32_avx2_8_into(
    gate: &[i64; 8],
    up: &[i64; 8],
    out: &mut [i64; 8],
) -> Result<(), (usize, FfnError)> {
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps};
    let mut g_f = [F32::ZERO; 8];
    let mut u_f = [F32::ZERO; 8];
    let mut l = 0usize;
    while l < 8 {
        g_f[l] = q32_to_f32(gate[l]);
        u_f[l] = q32_to_f32(up[l]);
        l += 1;
    }
    let vg = _mm256_loadu_ps(g_f.as_ptr().cast::<f32>());
    let vu = _mm256_loadu_ps(u_f.as_ptr().cast::<f32>());
    let vs = silu_f32_avx2(vg);
    let vp = _mm256_mul_ps(vs, vu);
    let mut prod_f = [F32::ZERO; 8];
    _mm256_storeu_ps(prod_f.as_mut_ptr().cast::<f32>(), vp);
    let mut l = 0usize;
    while l < 8 {
        match f32_to_q32_checked(prod_f[l]) {
            Ok(v) => out[l] = v,
            Err(e) => return Err((l, e)),
        }
        l += 1;
    }
    Ok(())
}

/// `h_i = round(silu(gate_i)·up_i / 2^32)` for every `i`, on the pool when `parallel`.
///
/// Resolves carrier from environment.
pub(crate) fn swiglu_hidden(
    gate: &[i64],
    up: &[i64],
    h: &mut [i64],
    parallel: bool,
) -> Result<(), FfnError> {
    swiglu_hidden_carrier(gate, up, h, parallel, carrier_from_env())
}

/// Evaluate `SwiGLU` elementwise over slices `gate` and `up`, writing into `out`, under explicit [`Carrier`].
/// Returns `Err((lane, err))` if a lane fails, reporting the relative failing index.
fn swiglu_slice(
    gate: &[i64],
    up: &[i64],
    out: &mut [i64],
    carrier: Carrier,
) -> Result<(), (usize, FfnError)> {
    let n = gate.len().min(up.len());
    let mut err: Option<(usize, FfnError)> = None;
    if carrier == Carrier::Q32 {
        let mut i = 0usize;
        while i < n && err.is_none() {
            match swiglu_one(gate[i], up[i]) {
                Ok(v) => out[i] = v,
                Err(e) => err = Some((i, e)),
            }
            i += 1;
        }
        return match err {
            Some(e) => Err(e),
            None => Ok(()),
        };
    }

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = std::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = false;

    let mut i = 0usize;
    let mut done = false;
    #[cfg(target_arch = "x86_64")]
    if has_avx2 {
        let mut buf = [0i64; 8];
        while i + 8 <= n && !done {
            let g8: &[i64; 8] = gate[i..i + 8].try_into().expect("8 elements");
            let u8: &[i64; 8] = up[i..i + 8].try_into().expect("8 elements");
            match unsafe { swiglu_f32_avx2_8_into(g8, u8, &mut buf) } {
                Ok(()) => {
                    out[i..i + 8].copy_from_slice(&buf);
                }
                Err((fail_lane, e)) => {
                    out[i..i + fail_lane].copy_from_slice(&buf[..fail_lane]);
                    err = Some((i + fail_lane, e));
                    done = true;
                }
            }
            i += 8;
        }
    }
    while i < n && !done {
        match swiglu_one_f32_scalar(gate[i], up[i]) {
            Ok(v) => out[i] = v,
            Err(e) => {
                err = Some((i, e));
                done = true;
            }
        }
        i += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Pool context for [`swiglu_hidden_carrier`]: the shared operands and the refusal slot.
struct SwigluHiddenCtx<'a> {
    gate: &'a [i64],
    up: &'a [i64],
    carrier: Carrier,
    first: &'a super::pool::FirstErr<usize, FfnError>,
}

impl ChunkTask<i64> for SwigluHiddenCtx<'_> {
    fn run(&self, _c: usize, start: usize, dst: &mut [i64]) {
        swiglu_hidden_task(self.gate, self.up, self.carrier, self.first, start, dst);
    }
}

/// Task body of [`SwigluHiddenCtx`]: `dst` is `h[start..start + dst.len()]`.
fn swiglu_hidden_task(
    gate: &[i64],
    up: &[i64],
    carrier: Carrier,
    first: &super::pool::FirstErr<usize, FfnError>,
    start: usize,
    dst: &mut [i64],
) {
    let len = dst.len();
    if let Err((lane, e)) = swiglu_slice(
        &gate[start..start + len],
        &up[start..start + len],
        dst,
        carrier,
    ) {
        first.record(start + lane, e);
    }
}

/// `h_i = round(silu(gate_i)·up_i / 2^32)` for every `i`, on the pool when `parallel`,
/// evaluated under explicit [`Carrier`].
pub(crate) fn swiglu_hidden_carrier(
    gate: &[i64],
    up: &[i64],
    h: &mut [i64],
    parallel: bool,
    carrier: Carrier,
) -> Result<(), FfnError> {
    if gate.len() != up.len() || gate.len() != h.len() {
        return Err(FfnError::BadGeometry);
    }
    if !parallel {
        return swiglu_slice(gate, up, h, carrier).map_err(|(_, e)| e);
    }
    let chunk = h.len().div_ceil(2 * super::pool::width_global()).max(1);
    let first: super::pool::FirstErr<usize, FfnError> = super::pool::FirstErr::new();
    super::pool::chunks_mut_global(
        h,
        chunk,
        &SwigluHiddenCtx {
            gate,
            up,
            carrier,
            first: &first,
        },
    );
    match first.take() {
        Some((_, e)) => Err(e),
        None => Ok(()),
    }
}

/// One pooled `SwiGLU` piece: `out[j] = swiglu(gate[offset + j], up[offset + j])` for `j` in
/// `0..out.len()`, stopping at the first refusal in index order.
fn swiglu_piece(
    gate: &[i64],
    up: &[i64],
    out: &mut [i64],
    offset: usize,
    carrier: Carrier,
) -> Result<(), FfnError> {
    let mut err: Option<FfnError> = None;
    let mut j = 0usize;
    while j < out.len() && err.is_none() {
        match swiglu_one_carrier(gate[offset + j], up[offset + j], carrier) {
            Ok(v) => out[j] = v,
            Err(e) => err = Some(e),
        }
        j += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// `out[c][j] = round(silu(gate[c][j])·up[c][j] / 2^32)` for every column `c` and every `j`: one
/// pool dispatch over the flat `k·n_ff` element count.
pub(crate) fn swiglu_cols(
    gate: &[Vec<i64>],
    up: &[Vec<i64>],
    out: &mut [Vec<i64>],
) -> Result<(), FfnError> {
    swiglu_cols_carrier(gate, up, out, carrier_from_env())
}

/// Pool context for [`swiglu_cols_carrier`]: the shared columns; task `c` is column `c`'s piece.
struct SwigluColsCtx<'a> {
    gate: &'a [Vec<i64>],
    up: &'a [Vec<i64>],
    carrier: Carrier,
}

impl VecTask<i64, FfnError> for SwigluColsCtx<'_> {
    fn run(&self, _t: usize, c: usize, offset: usize, piece: &mut [i64]) -> Result<(), FfnError> {
        swiglu_col_task(self.gate, self.up, c, offset, piece, self.carrier)
    }
}

/// Task body of [`SwigluColsCtx`]: column `c`'s piece at `offset`.
fn swiglu_col_task(
    gate: &[Vec<i64>],
    up: &[Vec<i64>],
    c: usize,
    offset: usize,
    piece: &mut [i64],
    carrier: Carrier,
) -> Result<(), FfnError> {
    swiglu_piece(&gate[c], &up[c], piece, offset, carrier)
}

/// `out[c][j] = round(silu(gate[c][j])·up[c][j] / 2^32)` for every column `c` and every `j` under
/// explicit [`Carrier`].
#[allow(
    clippy::many_single_char_names,
    reason = "k/n are the column and element counts and c/j an element's column and index, as in \
              the per-column loop this replaces"
)]
pub(crate) fn swiglu_cols_carrier(
    gate: &[Vec<i64>],
    up: &[Vec<i64>],
    out: &mut [Vec<i64>],
    carrier: Carrier,
) -> Result<(), FfnError> {
    let k = gate.len();
    if up.len() != k || out.len() != k {
        return Err(FfnError::BadGeometry);
    }
    #[allow(
        clippy::len_zero,
        reason = "slice len compares extract; first()/is_empty do not"
    )]
    let n = if gate.len() > 0 { gate[0].len() } else { 0 };
    if crate::engine::step::any_vec_len_ne(up, n) || crate::engine::step::any_vec_len_ne(out, n) {
        return Err(FfnError::BadGeometry);
    }
    let flat = k.checked_mul(n).ok_or(FfnError::BadGeometry)?;
    if flat == 0 {
        return Ok(());
    }
    // Each column is cut into `tasks_per_col` pieces, so the dispatch has about `2·width` tasks.
    let tasks_per_col = (2 * super::pool::width_global()).div_ceil(k).max(1);
    let chunk = n.div_ceil(tasks_per_col).max(1);
    let chunks = vec![chunk; k];
    super::pool::for_vec_chunks(out, &chunks, &SwigluColsCtx { gate, up, carrier })
}

/// The most scoped threads the Q8-activation row-chunked kernel uses.
///
/// `row_chunk_budget` resolves to all `available_parallelism` threads (32 here), so every projection
/// spawns and joins that many per call; on the hybrid 8-P+16-E host the spawn/join sys time and the
/// E-core/SMT contention outweigh the added parallelism. This cap was fixed by the chunk-count
/// sweep recorded in the W2 receipt: the decode tok/s for `[4, 8, 16, 32]` chunks.
const Q8ACT_MAX_CHUNKS: usize = 8;

/// Pool context for [`matvec_q8act_quant_chunked`]: the shared payload, geometry, plan and refusal slot.
struct Q8ChunkCtx<'a> {
    raw: &'a [u8],
    ggml_type: u32,
    n_in: usize,
    row_bytes: usize,
    need: usize,
    plan: &'a Q8ActPlan,
    q: &'a QuantAct,
    first_err: &'a super::pool::FirstErr<usize, MatvecError>,
}

impl ChunkTask<i64> for Q8ChunkCtx<'_> {
    fn run(&self, c: usize, lo: usize, dst: &mut [i64]) {
        q8act_chunk_task(
            self.raw,
            self.ggml_type,
            self.n_in,
            self.row_bytes,
            self.need,
            self.plan,
            self.q,
            self.first_err,
            c,
            lo,
            dst,
        );
    }
}

/// Task body of [`Q8ChunkCtx`]: rows `lo..lo + dst.len()` of the payload into `dst`; a refusal is
/// recorded at chunk `c`.
#[allow(
    clippy::too_many_arguments,
    reason = "each input is passed separately: the task body is extracted with its branches"
)]
fn q8act_chunk_task(
    raw: &[u8],
    ggml_type: u32,
    n_in: usize,
    row_bytes: usize,
    need: usize,
    plan: &Q8ActPlan,
    q: &QuantAct,
    first_err: &super::pool::FirstErr<usize, MatvecError>,
    c: usize,
    lo: usize,
    dst: &mut [i64],
) {
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
            first_err.record(c, e);
        }
    }
}

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
    let tasks = actual.max(2 * super::pool::width_global()).min(n_out);
    let per = n_out.div_ceil(tasks);
    let mut out: Vec<i64> = vec![0; n_out];
    let first_err: super::pool::FirstErr<usize, MatvecError> = super::pool::FirstErr::new();
    super::pool::chunks_mut_global(
        &mut out,
        per,
        &Q8ChunkCtx {
            raw,
            ggml_type,
            n_in,
            row_bytes,
            need,
            plan: &plan,
            q,
            first_err: &first_err,
        },
    );
    match first_err.take() {
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

impl RowPtr {
    #[inline]
    unsafe fn as_mut_slice<'a>(&self, offset: usize, len: usize) -> &'a mut [i64] {
        std::slice::from_raw_parts_mut(self.0.add(offset), len)
    }
}

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

/// One `PQ2_0` slot of the engine's open-time table, indexed by directory index: `None` for an
/// entry that is not a rank-2 `PQ2_0` tensor, else the repacked table or its stored refusal.
pub(crate) type LutSlot = Option<Result<Pq2Lut, MatvecError>>;

/// The `PQ2_0` table of slot `k`, or the stored refusal (`UnsupportedQuant` for an empty or absent
/// slot).
fn lut_at(luts: &[LutSlot], k: usize) -> Result<&Pq2Lut, MatvecError> {
    match luts.get(k) {
        Some(Some(Ok(l))) => Ok(l),
        Some(Some(Err(e))) => Err(e.clone()),
        _ => Err(MatvecError::UnsupportedQuant {
            ggml_type: quant::GGML_TYPE_PQ2_0,
        }),
    }
}

/// The row kernel of one projection in a [`projections_q8act`] dispatch.
#[derive(Clone, Copy)]
pub(crate) enum ProjSource {
    /// The chart Q8-activation row core over the payload `at` of the resident file.
    Chart { ggml_type: u32, at: PayloadRef },
    /// The tile-aligned `PQ2_0` lookup-table kernel ([`Pq2Lut::prepare`], [`Pq2Lut::rows`]) of the
    /// table in slot `k` of the open-time [`LutSlot`] table.
    Rows(usize),
}

/// One projection of a [`projections_q8act`] dispatch.
#[derive(Clone, Copy)]
pub(crate) struct ProjSpec {
    /// Output rows.
    pub(crate) n_out: usize,
    /// A chunk start must be a multiple of this (`1` for the chart row core, the tile height for a
    /// tile-aligned kernel).
    pub(crate) row_align: usize,
    /// The row kernel.
    pub(crate) source: ProjSource,
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
fn proj_split(prep: &GroupPrep, p: &ProjSpec, chunks: usize, width: usize) -> (usize, usize) {
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
/// all `n` projections in order; each task calls its projection's row kernel — the chart row core
/// over its payload in `file`, or the [`Pq2Lut`] rows of a tile-aligned LUT in `luts`.
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
/// [`matvec_q8act_quant_chunked`] / [`Pq2Lut::rows`] calls joined left to right would return.
///
/// # Complexity
///
/// `O(Σ n_out_i·n_in)` time in checked multiply-adds with one `quant_act` instead of `n`, and
/// `O(Σ n_out_i)` output, in one synchronization instead of `n`.
pub(crate) fn projections_q8act(
    file: &[u8],
    luts: &[LutSlot],
    n_in: usize,
    x: &[i64],
    projs: &[ProjSpec],
    chunks: usize,
) -> Result<Vec<Vec<i64>>, MatvecError> {
    if x.len() != n_in {
        return Err(MatvecError::ShapeMismatch { x: x.len(), n_in });
    }
    let q = quantize(x)?;
    projections_q8act_prepared(file, luts, n_in, &q, projs, chunks)
}

/// [`projections_q8act`] over an already-quantized activation: the same dispatch, without the
/// `quant_act`. `q` must be [`quant_act_pooled`]'s or [`quant_act`]'s result for the activation.
pub(crate) fn projections_q8act_prepared(
    file: &[u8],
    luts: &[LutSlot],
    n_in: usize,
    q: &QuantAct,
    projs: &[ProjSpec],
    chunks: usize,
) -> Result<Vec<Vec<i64>>, MatvecError> {
    let mut outs: Vec<Vec<i64>> = Vec::new();
    projections_q8act_prepared_into(file, luts, n_in, q, projs, chunks, &mut outs)?;
    Ok(outs)
}

/// One task of the [`projections_q8act_prepared_into`] dispatch: rows `[lo, lo + piece.len())` of
/// projection `spec` into `piece`. `prep` is the projection's plan and `prepared` its per-dispatch
/// LUT state (`None` for a chart projection or an unprepared LUT).
///
/// # Complexity
///
/// `O(piece.len()·n_in)` time in checked multiply-adds; `O(piece.len())` scratch for the chart row
/// core's result vector.
#[allow(
    clippy::too_many_arguments,
    reason = "the payload source, plan, activation and row range are the task's full input"
)]
fn proj_task(
    file: &[u8],
    luts: &[LutSlot],
    spec: &ProjSpec,
    prep: &GroupPrep,
    prepared: Option<&PreparedLut>,
    n_in: usize,
    q: &QuantAct,
    lo: usize,
    piece: &mut [i64],
) -> Result<(), MatvecError> {
    let rows = piece.len();
    let hi = lo + rows;
    match (&spec.source, prep) {
        (ProjSource::Chart { ggml_type, at }, GroupPrep::Chart(pp)) => {
            let raw = at.bytes(file);
            let bytes =
                raw.get(lo * pp.row_bytes..hi * pp.row_bytes)
                    .ok_or(MatvecError::ShortPayload {
                        need: rows * pp.row_bytes,
                        have: raw.len(),
                    })?;
            let v = match &pp.plan {
                Q8ActPlan::Fused(fast) => q8act_rows(
                    *ggml_type,
                    bytes,
                    rows,
                    n_in,
                    pp.row_bytes,
                    rows * pp.row_bytes,
                    q,
                    fast,
                )?,
                Q8ActPlan::Reference => matvec_q8act(*ggml_type, bytes, rows, n_in, q)?,
            };
            piece.copy_from_slice(&v);
            Ok(())
        }
        (ProjSource::Rows(k), _) => {
            let lut = lut_at(luts, *k)?;
            match prepared {
                Some(p) => p.rows(lut, q, lo, hi, piece),
                None => lut.rows(q, lo, hi, piece),
            }
        }
        _ => unreachable!("delegated groups returned before the dispatch"),
    }
}

/// The plan of one projection of a group: its chart plan, a delegation, or the LUT rows marker.
fn group_prep_of(
    file: &[u8],
    p: &ProjSpec,
    n_in: usize,
    q: &QuantAct,
    chunks: usize,
) -> Result<GroupPrep, MatvecError> {
    match &p.source {
        ProjSource::Chart { ggml_type, at } => {
            match proj_prep(*ggml_type, at.bytes(file), p.n_out, n_in, q, chunks)? {
                Some(pp) => Ok(GroupPrep::Chart(pp)),
                None => Ok(GroupPrep::Delegate),
            }
        }
        ProjSource::Rows(_) => Ok(GroupPrep::Rows),
    }
}

/// One projection of a delegated group through the sequential entry points, into `out`.
///
/// # Complexity
///
/// `O(p.n_out·n_in)` time in checked multiply-adds; `O(p.n_out)` scratch for the chart result.
fn delegate_one(
    file: &[u8],
    luts: &[LutSlot],
    n_in: usize,
    q: &QuantAct,
    p: &ProjSpec,
    chunks: usize,
    out: &mut [i64],
) -> Result<(), MatvecError> {
    match &p.source {
        ProjSource::Chart { ggml_type, at } => {
            let v =
                matvec_q8act_quant_chunked(*ggml_type, at.bytes(file), p.n_out, n_in, q, chunks)?;
            out.copy_from_slice(&v);
            Ok(())
        }
        ProjSource::Rows(k) => lut_at(luts, *k)?.rows(q, 0, p.n_out, out),
    }
}

/// Pool context for [`projections_q8act_prepared_into`]: the shared projections, plans and activation.
struct ProjTaskCtx<'a> {
    file: &'a [u8],
    luts: &'a [LutSlot],
    projs: &'a [ProjSpec],
    preps: &'a [GroupPrep],
    prepared: &'a [Option<PreparedLut>],
    n_in: usize,
    q: &'a QuantAct,
}

impl VecTask<i64, MatvecError> for ProjTaskCtx<'_> {
    fn run(&self, _t: usize, i: usize, lo: usize, piece: &mut [i64]) -> Result<(), MatvecError> {
        proj_task(
            self.file,
            self.luts,
            &self.projs[i],
            &self.preps[i],
            self.prepared[i].as_ref(),
            self.n_in,
            self.q,
            lo,
            piece,
        )
    }
}

/// [`projections_q8act_prepared`] writing into caller-owned row buffers.
///
/// `outs` is resized to one vector per projection of `n_out` rows; a resize is a no-op when the
/// caller hands back the same buffers, so a decode step that composes the same widths every layer
/// pays no allocate-and-zero per layer. Every row is written before it is read, so a reused buffer
/// carries no stale value into the result. The tasks are numbered projection-then-row by
/// [`super::pool::for_vec_chunks`], which returns the lowest-numbered refusal.
///
/// # Complexity
///
/// As [`projections_q8act`]; `O(projs.len())` extra space for the plans and the task table.
pub(crate) fn projections_q8act_prepared_into(
    file: &[u8],
    luts: &[LutSlot],
    n_in: usize,
    q: &QuantAct,
    projs: &[ProjSpec],
    chunks: usize,
    outs: &mut Vec<Vec<i64>>,
) -> Result<(), MatvecError> {
    if q.len() != n_in {
        return Err(MatvecError::ShapeMismatch { x: q.len(), n_in });
    }
    size_outs(projs, outs);
    let preps = group_preps(file, projs, n_in, q, chunks)?;
    // One chart projection that cannot chunk (or refuses) sends the whole group to the sequential
    // calls, whose join order is the first refusal order.
    if any_delegate(&preps) {
        return delegate_all(file, luts, n_in, q, projs, chunks, outs);
    }
    // The LUT's activation tables are per-dispatch, not per-task: build them once before the run.
    let prepared = prepare_luts(luts, projs, q)?;
    let per_chunks = split_counts(&preps, projs, chunks, super::pool::width_global());
    super::pool::for_vec_chunks(
        outs,
        &per_chunks,
        &ProjTaskCtx {
            file,
            luts,
            projs,
            preps: &preps,
            prepared: &prepared,
            n_in,
            q,
        },
    )
}

/// Resize `outs` to one vector of `n_out` rows per projection.
///
/// # Complexity
///
/// `O(projs.len())` time and, only when a buffer changes length, `O(n_out)` zeroed words.
fn size_outs(projs: &[ProjSpec], outs: &mut Vec<Vec<i64>>) {
    outs.resize(projs.len(), Vec::new());
    let mut oi = 0usize;
    while oi < projs.len() {
        if outs[oi].len() != projs[oi].n_out {
            outs[oi].resize(projs[oi].n_out, 0);
        }
        oi += 1;
    }
}

/// The plan of every projection, in order; the first refusal is returned.
///
/// # Complexity
///
/// `O(projs.len())` plans; `O(projs.len())` space.
fn group_preps(
    file: &[u8],
    projs: &[ProjSpec],
    n_in: usize,
    q: &QuantAct,
    chunks: usize,
) -> Result<Vec<GroupPrep>, MatvecError> {
    let mut preps: Vec<GroupPrep> = Vec::with_capacity(projs.len());
    let mut pj = 0usize;
    while pj < projs.len() {
        preps.push(group_prep_of(file, &projs[pj], n_in, q, chunks)?);
        pj += 1;
    }
    Ok(preps)
}

/// Whether any projection of the group delegates.
fn any_delegate(preps: &[GroupPrep]) -> bool {
    let mut delegate = false;
    let mut pi = 0usize;
    while pi < preps.len() {
        if matches!(preps[pi], GroupPrep::Delegate) {
            delegate = true;
        }
        pi += 1;
    }
    delegate
}

/// Every projection of a delegated group through the sequential entry points, in order.
///
/// # Complexity
///
/// `O(Σ n_out·n_in)` time in checked multiply-adds; `O(max n_out)` scratch.
fn delegate_all(
    file: &[u8],
    luts: &[LutSlot],
    n_in: usize,
    q: &QuantAct,
    projs: &[ProjSpec],
    chunks: usize,
    outs: &mut [Vec<i64>],
) -> Result<(), MatvecError> {
    let mut di = 0usize;
    while di < projs.len() {
        delegate_one(file, luts, n_in, q, &projs[di], chunks, &mut outs[di])?;
        di += 1;
    }
    Ok(())
}

/// The per-dispatch LUT state of every projection (`None` for a chart projection).
///
/// # Complexity
///
/// `O(projs.len())` LUT activation preparations; `O(projs.len())` space beyond the tables.
fn prepare_luts(
    luts: &[LutSlot],
    projs: &[ProjSpec],
    q: &QuantAct,
) -> Result<Vec<Option<PreparedLut>>, MatvecError> {
    let mut prepared: Vec<Option<PreparedLut>> = Vec::with_capacity(projs.len());
    let mut i = 0usize;
    while i < projs.len() {
        prepared.push(match &projs[i].source {
            ProjSource::Rows(k) => lut_at(luts, *k)?.prepare(q),
            ProjSource::Chart { .. } => None,
        });
        i += 1;
    }
    Ok(prepared)
}

/// The chunk count of every projection.
///
/// # Complexity
///
/// `O(projs.len())` time and space.
fn split_counts(
    preps: &[GroupPrep],
    projs: &[ProjSpec],
    chunks: usize,
    width: usize,
) -> Vec<usize> {
    let mut per_chunks: Vec<usize> = Vec::with_capacity(projs.len());
    let mut i = 0usize;
    while i < projs.len() {
        per_chunks.push(proj_split(&preps[i], &projs[i], chunks, width).0);
        i += 1;
    }
    per_chunks
}

/// Pool context for [`attn_heads`]: the head function and the refusal slot.
struct AttnHeadCtx<'a, E, K> {
    head: &'a K,
    first: &'a super::pool::FirstErr<usize, E>,
}

impl<E: Send, K: HeadTask<i64, E>> ChunkTask<i64> for AttnHeadCtx<'_, E, K> {
    fn run(&self, c: usize, _start: usize, dst: &mut [i64]) {
        attn_head_task(self.head, self.first, c, dst);
    }
}

/// Task body of [`AttnHeadCtx`]: head `c` writes `dst`; its refusal is recorded at `c`.
fn attn_head_task<E: Send, K: HeadTask<i64, E>>(
    head: &K,
    first: &super::pool::FirstErr<usize, E>,
    c: usize,
    dst: &mut [i64],
) {
    if let Err(e) = head.run(c, dst) {
        first.record(c, e);
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
pub(crate) fn attn_heads<E, K>(
    n_head: usize,
    hd: usize,
    out: &mut [i64],
    parallel: bool,
    head: &K,
) -> Option<(usize, E)>
where
    K: HeadTask<i64, E>,
    E: Send,
{
    if !parallel {
        let mut h = 0usize;
        while h < n_head {
            let lo = h * hd;
            if let Err(e) = head.run(h, &mut out[lo..lo + hd]) {
                return Some((h, e));
            }
            h += 1;
        }
        return None;
    }
    let first: super::pool::FirstErr<usize, E> = super::pool::FirstErr::new();
    super::pool::chunks_mut_global(
        out,
        hd,
        &AttnHeadCtx {
            head,
            first: &first,
        },
    );
    first.take()
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
/// `O(chunks)` scoped-thread handles owned by the chunked kernel. This entry takes borrowed
/// payloads, so it concatenates the three into one temporary file (`O(bytes)` copy) for the
/// offset-addressed dispatch; the engine path ([`ffn_q8act_prepared`]) copies no weight payload.
pub(crate) fn ffn_q8act(
    gate: ProjWeights<'_>,
    up: ProjWeights<'_>,
    down: ProjWeights<'_>,
    n_ff: usize,
    x: &[i64],
) -> Result<Vec<i64>, FfnError> {
    let n_embd = x.len();
    let mut file: Vec<u8> = Vec::with_capacity(gate.raw.len() + up.raw.len() + down.raw.len());
    let mut spec = |w: &ProjWeights<'_>, n_out: usize| {
        let at = PayloadRef {
            off: file.len(),
            len: w.raw.len(),
        };
        file.extend_from_slice(w.raw);
        ProjSpec {
            n_out,
            row_align: 1,
            source: ProjSource::Chart {
                ggml_type: w.ggml_type,
                at,
            },
        }
    };
    let g = spec(&gate, n_ff);
    let u = spec(&up, n_ff);
    let d = spec(&down, n_embd);
    ffn_q8act_prepared(
        &file,
        &[],
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

#[allow(
    clippy::too_many_arguments,
    reason = "projection parameters and output slice are passed together"
)]
fn chart_rows(
    ggml_type: u32,
    raw: &[u8],
    lo: usize,
    hi: usize,
    pp: &ProjPrep,
    n_in: usize,
    q: &QuantAct,
    dst: &mut [i64],
) -> Result<(), MatvecError> {
    let rows = hi - lo;
    let need = rows
        .checked_mul(pp.row_bytes)
        .ok_or(MatvecError::LengthOverflow)?;
    let bytes = raw
        .get(lo * pp.row_bytes..hi * pp.row_bytes)
        .ok_or(MatvecError::ShortPayload {
            need,
            have: raw.len(),
        })?;
    let v = match &pp.plan {
        Q8ActPlan::Fused(fast) => {
            q8act_rows(ggml_type, bytes, rows, n_in, pp.row_bytes, need, q, fast)
        }
        Q8ActPlan::Reference => matvec_q8act(ggml_type, bytes, rows, n_in, q),
    }?;
    dst.copy_from_slice(&v);
    Ok(())
}

/// [`ffn_q8act`]'s composition over caller-built [`ProjSpec`]s: one pooled `quant_act` of `x`, the
/// gate/up pair in one [`projections_q8act_prepared`] dispatch, [`swiglu_hidden`], then the down
/// spec over a pooled `quant_act` of the hidden. A spec carries either the chart row core or a
/// [`Pq2Lut`], so a `PQ2_0` lookup table runs its rows here.
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
    file: &[u8],
    luts: &[LutSlot],
    n_in: usize,
    x: &[i64],
    gate: ProjSpec,
    up: ProjSpec,
    down: ProjSpec,
    n_ff: usize,
    chunks_gate_up: usize,
    chunks_down: usize,
) -> Result<Vec<i64>, FfnError> {
    let mut scratch = FfnScratch::default();
    ffn_q8act_prepared_into(
        file,
        luts,
        n_in,
        x,
        gate,
        up,
        down,
        n_ff,
        chunks_gate_up,
        chunks_down,
        &mut scratch,
    )
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
    /// Pre-allocated [`QuantAct`] for the hidden activation; empty until the first fused call sizes it.
    hidden_quant: QuantAct,
}

/// Pool context for [`ffn_q8act_prepared_into`]: the fused gate/up/SwiGLU/quantize task's inputs.
struct FusedFfnCtx<'a> {
    blocks: usize,
    per_blocks: usize,
    gate_type: u32,
    gate_raw: &'a [u8],
    gate_pp: &'a ProjPrep,
    up_type: u32,
    up_raw: &'a [u8],
    up_pp: &'a ProjPrep,
    n_in: usize,
    q: &'a QuantAct,
    gate_ptr: RowPtr,
    up_ptr: RowPtr,
    hidden_ptr: RowPtr,
    act_ptr: ActPtr,
    carrier: Carrier,
    proj_err: &'a super::pool::FirstErr<u128, MatvecError>,
    swiglu_err: &'a super::pool::FirstErr<usize, FfnError>,
    quant_err: &'a super::pool::FirstErr<usize, ()>,
}

impl IndexTask for FusedFfnCtx<'_> {
    fn run(&self, t: usize) {
        fused_ffn_task(
            t,
            self.blocks,
            self.per_blocks,
            self.gate_type,
            self.gate_raw,
            self.gate_pp,
            self.up_type,
            self.up_raw,
            self.up_pp,
            self.n_in,
            self.q,
            self.gate_ptr,
            self.up_ptr,
            self.hidden_ptr,
            self.act_ptr,
            self.carrier,
            self.proj_err,
            self.swiglu_err,
            self.quant_err,
        );
    }
}

/// Task body of [`FusedFfnCtx`]: task `t`'s block run of the fused gate/up projections, `SwiGLU`
/// and activation quantization.
#[allow(
    clippy::too_many_arguments,
    reason = "each input is passed separately: the task body is extracted with its branches"
)]
fn fused_ffn_task(
    t: usize,
    blocks: usize,
    per_blocks: usize,
    gate_type: u32,
    gate_raw: &[u8],
    gate_pp: &ProjPrep,
    up_type: u32,
    up_raw: &[u8],
    up_pp: &ProjPrep,
    n_in: usize,
    q: &QuantAct,
    gate_ptr: RowPtr,
    up_ptr: RowPtr,
    hidden_ptr: RowPtr,
    act_ptr: ActPtr,
    carrier: Carrier,
    proj_err: &super::pool::FirstErr<u128, MatvecError>,
    swiglu_err: &super::pool::FirstErr<usize, FfnError>,
    quant_err: &super::pool::FirstErr<usize, ()>,
) {
    let lo_b = (t * per_blocks).min(blocks);
    let hi_b = ((t + 1) * per_blocks).min(blocks);
    if lo_b >= hi_b {
        return;
    }
    let lo = lo_b * 32;
    let hi = hi_b * 32;
    let rows = hi - lo;

    let gate_dst = unsafe { gate_ptr.as_mut_slice(lo, rows) };
    let up_dst = unsafe { up_ptr.as_mut_slice(lo, rows) };
    let h_dst = unsafe { hidden_ptr.as_mut_slice(lo, rows) };

    // Stage 1a: Gate projection
    if let Err(e) = chart_rows(gate_type, gate_raw, lo, hi, gate_pp, n_in, q, gate_dst) {
        proj_err.record(super::task::pair_key(0, lo), e);
        return;
    }

    // Stage 1b: Up projection
    if let Err(e) = chart_rows(up_type, up_raw, lo, hi, up_pp, n_in, q, up_dst) {
        proj_err.record(super::task::pair_key(1, lo), e);
        return;
    }

    // Stage 2: SwiGLU elementwise
    if let Err((lane, e)) = swiglu_slice(gate_dst, up_dst, h_dst, carrier) {
        let idx = lo + lane;
        swiglu_err.record(idx, e);
        return;
    }

    // Stage 3: Activation quantization on h_dst
    if let Some(qa) = quant_act(h_dst) {
        unsafe {
            act_ptr.write_scales(lo_b, qa.scales().as_ptr(), hi_b - lo_b);
            act_ptr.write_codes(lo, qa.codes().as_ptr(), rows);
        }
    } else {
        quant_err.record(lo_b, ());
    }
}

/// [`ffn_q8act_prepared`] with caller-owned scratch buffers (`scratch`), returning the down
/// projection's rows.
///
/// When `gate` and `up` are chart projections and `n_ff` is 32-aligned, gate and up projections,
/// `SwiGLU`, and hidden quantization run in ONE fused dispatch across the pool, followed by the down
/// projection. Refusal order is preserved: projection error, then `SwiGLU`, then quantization.
/// When not eligible, falls back to the separate-stage path.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the three row kernels, their activation, the two dispatch widths and the scratch \
              buffers are the composition's full input"
)]
pub(crate) fn ffn_q8act_prepared_into(
    file: &[u8],
    luts: &[LutSlot],
    n_in: usize,
    x: &[i64],
    gate: ProjSpec,
    up: ProjSpec,
    down: ProjSpec,
    n_ff: usize,
    chunks_gate_up: usize,
    chunks_down: usize,
    scratch: &mut FfnScratch,
) -> Result<Vec<i64>, FfnError> {
    if n_ff == 0 || n_in == 0 || x.len() != n_in {
        return Err(FfnError::BadGeometry);
    }
    let q = quantize(x)?;

    let can_fuse = n_ff.is_multiple_of(32)
        && gate.n_out == n_ff
        && up.n_out == n_ff
        && matches!(&gate.source, ProjSource::Chart { .. })
        && matches!(&up.source, ProjSource::Chart { .. });

    if can_fuse {
        let (gate_type, gate_raw) = match &gate.source {
            ProjSource::Chart { ggml_type, at } => (*ggml_type, at.bytes(file)),
            ProjSource::Rows(_) => unreachable!(),
        };
        let (up_type, up_raw) = match &up.source {
            ProjSource::Chart { ggml_type, at } => (*ggml_type, at.bytes(file)),
            ProjSource::Rows(_) => unreachable!(),
        };

        let gate_prep = proj_prep(gate_type, gate_raw, n_ff, n_in, &q, chunks_gate_up)?;
        let up_prep = proj_prep(up_type, up_raw, n_ff, n_in, &q, chunks_gate_up)?;

        if let (Some(gate_pp), Some(up_pp)) = (gate_prep, up_prep) {
            let blocks = n_ff / 32;
            if scratch.rows.len() < 2 {
                while scratch.rows.len() < 2 {
                    scratch.rows.push(Vec::new());
                }
            }
            if scratch.rows[0].len() != n_ff {
                scratch.rows[0].resize(n_ff, 0);
            }
            if scratch.rows[1].len() != n_ff {
                scratch.rows[1].resize(n_ff, 0);
            }
            if scratch.hidden.len() != n_ff {
                scratch.hidden.resize(n_ff, 0);
            }
            let qh = &mut scratch.hidden_quant;
            if qh.scales.len() != blocks {
                qh.scales.resize(blocks, 0);
            }
            if qh.codes.len() != n_ff {
                qh.codes.resize(n_ff, 0);
            }

            let width = super::pool::width_global();
            let target_tasks = (2 * width).min(blocks).max(1);
            let per_blocks = blocks.div_ceil(target_tasks);
            let tasks = blocks.div_ceil(per_blocks);

            let gate_ptr = RowPtr(scratch.rows[0].as_mut_ptr());
            let up_ptr = RowPtr(scratch.rows[1].as_mut_ptr());
            let hidden_ptr = RowPtr(scratch.hidden.as_mut_ptr());
            let act_ptr = ActPtr {
                scales: qh.scales.as_mut_ptr(),
                codes: qh.codes.as_mut_ptr(),
            };

            let proj_err: super::pool::FirstErr<u128, MatvecError> = super::pool::FirstErr::new();
            let swiglu_err: super::pool::FirstErr<usize, FfnError> = super::pool::FirstErr::new();
            let quant_err: super::pool::FirstErr<usize, ()> = super::pool::FirstErr::new();
            let carrier = carrier_from_env();

            super::pool::run_global(
                tasks,
                &FusedFfnCtx {
                    blocks,
                    per_blocks,
                    gate_type,
                    gate_raw,
                    gate_pp: &gate_pp,
                    up_type,
                    up_raw,
                    up_pp: &up_pp,
                    n_in,
                    q: &q,
                    gate_ptr,
                    up_ptr,
                    hidden_ptr,
                    act_ptr,
                    carrier,
                    proj_err: &proj_err,
                    swiglu_err: &swiglu_err,
                    quant_err: &quant_err,
                },
            );

            // Refusal precedence: projection, then SwiGLU, then quantization
            if let Some((_, e)) = proj_err.take() {
                return Err(FfnError::Matvec(e));
            }
            if let Some((_, e)) = swiglu_err.take() {
                return Err(e);
            }
            if quant_err.take().is_some() {
                return Err(FfnError::Matvec(MatvecError::Dequant(
                    EngineError::FixedOverflow,
                )));
            }

            let qh = &scratch.hidden_quant;
            if scratch.down_rows.is_empty() {
                scratch.down_rows.push(Vec::new());
            }
            projections_q8act_prepared_into(
                file,
                luts,
                n_ff,
                qh,
                &[down],
                chunks_down,
                &mut scratch.down_rows,
            )?;
            return Ok(std::mem::take(&mut scratch.down_rows[0]));
        }
    }

    // Fallback path
    projections_q8act_prepared_into(
        file,
        luts,
        n_in,
        &q,
        &[gate, up],
        chunks_gate_up,
        &mut scratch.rows,
    )?;
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
    projections_q8act_prepared_into(
        file,
        luts,
        n_ff,
        &qh,
        &[down],
        chunks_down,
        &mut scratch.down_rows,
    )?;
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

    /// The batched multi-column quantization equals one [`quant_act`] per column at 33-wide columns
    /// (the second block holds one lane), including the `k = 1` delegation and the differing-length
    /// fallback. Each column's values are scaled by its own factor, so its block scales differ from
    /// its neighbours' and a misdirected copy cannot pass by coincidence.
    #[test]
    fn quant_acts_pooled_equals_the_per_column_quantizations() {
        for k in [1usize, 2, 7] {
            let xs: Vec<Vec<i64>> = (0..k)
                .map(|c| {
                    let w = i64::try_from(c).expect("small") + 1;
                    activation(33, w).iter().map(|v| *v * w).collect()
                })
                .collect();
            let want: Vec<QuantAct> = xs
                .iter()
                .map(|x| quant_act(x).expect("quantizes"))
                .collect();
            let got = quant_acts_pooled(&xs).expect("batched quantization");
            assert_eq!(got.len(), k, "k = {k}");
            for (c, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(g.scales(), w.scales(), "scales, column {c}, k = {k}");
                assert_eq!(g.codes(), w.codes(), "codes, column {c}, k = {k}");
            }
        }
        // Columns of differing length take the per-column fallback.
        let mixed: Vec<Vec<i64>> = vec![activation(33, 1), activation(1, 2), activation(64, 3)];
        let want: Vec<QuantAct> = mixed
            .iter()
            .map(|x| quant_act(x).expect("quantizes"))
            .collect();
        let got = quant_acts_pooled(&mixed).expect("fallback");
        for (c, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g.scales(), w.scales(), "mixed scales, column {c}");
            assert_eq!(g.codes(), w.codes(), "mixed codes, column {c}");
        }
    }

    /// A column whose block scale times its code leaves `i64` refuses the whole batch, as the single
    /// call refuses that column.
    #[test]
    fn quant_acts_pooled_refuses_with_a_refusing_column() {
        let mut xs: Vec<Vec<i64>> = (0..3usize)
            .map(|c| activation(33, i64::try_from(c).expect("small") + 1))
            .collect();
        xs[1][0] = i64::MAX;
        assert!(quant_act(&xs[1]).is_none(), "the fixture column refuses");
        assert!(quant_acts_pooled(&xs).is_none(), "the batch refuses");
    }

    /// The pooled multi-column `SwiGLU` equals the serial per-column [`swiglu_hidden`] loop bit for
    /// bit over widths that do and do not divide the flat range, including a one-column batch.
    #[test]
    fn swiglu_cols_equals_the_serial_per_column_loop() {
        for k in [1usize, 2, 7] {
            for n_ff in [1usize, 33, 128] {
                let gate: Vec<Vec<i64>> = (0..k)
                    .map(|c| activation(n_ff, i64::try_from(c).expect("small") + 2))
                    .collect();
                let up: Vec<Vec<i64>> = (0..k)
                    .map(|c| activation(n_ff, i64::try_from(c).expect("small") + 5))
                    .collect();
                let want: Vec<Vec<i64>> = gate
                    .iter()
                    .zip(&up)
                    .map(|(g, u)| {
                        let mut h = vec![0i64; n_ff];
                        swiglu_hidden(g, u, &mut h, false).expect("serial");
                        h
                    })
                    .collect();
                let mut got = vec![vec![0i64; n_ff]; k];
                swiglu_cols(&gate, &up, &mut got)
                    .unwrap_or_else(|e| panic!("k = {k}, n_ff = {n_ff}: {e:?}"));
                assert_eq!(got, want, "k = {k}, n_ff = {n_ff}");
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
        let file = [q8.as_slice(), f16.as_slice()].concat();
        let specs = [
            ProjSpec {
                n_out: ROWS,
                row_align: 1,
                source: ProjSource::Chart {
                    ggml_type: GGML_TYPE_Q8_0,
                    at: PayloadRef {
                        off: 0,
                        len: q8.len(),
                    },
                },
            },
            ProjSpec {
                n_out: ROWS,
                row_align: 1,
                source: ProjSource::Chart {
                    ggml_type: GGML_TYPE_F16,
                    at: PayloadRef {
                        off: q8.len(),
                        len: f16.len(),
                    },
                },
            },
        ];
        let want = [
            matvec_q8act(GGML_TYPE_Q8_0, &q8, ROWS, N_IN, &q).expect("q8"),
            matvec_q8act(GGML_TYPE_F16, &f16, ROWS, N_IN, &q).expect("f16"),
        ];
        for chunks in [1usize, 2, 8] {
            let got = projections_q8act(&file, &[], N_IN, &x, &specs, chunks).expect("dispatch");
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

    /// The gate/up/down specs over a file laid out as `gate || up || down`, each payload `full`
    /// bytes (gate and up) or `d_full` bytes (down); `lens` are the lengths the specs claim, so a
    /// shorter claim models a truncated payload.
    fn make_test_specs(
        n_embd: usize,
        n_ff: usize,
        full: usize,
        lens: [usize; 3],
    ) -> (ProjSpec, ProjSpec, ProjSpec) {
        let spec = |n_out: usize, off: usize, len: usize| ProjSpec {
            n_out,
            row_align: 1,
            source: ProjSource::Chart {
                ggml_type: GGML_TYPE_Q8_0,
                at: PayloadRef { off, len },
            },
        };
        (
            spec(n_ff, 0, lens[0]),
            spec(n_ff, full, lens[1]),
            spec(n_embd, 2 * full, lens[2]),
        )
    }

    /// Fused `gate+up+SwiGLU+hidden_quant` equals the separate-stage execution on `Q8_0` weights,
    /// and preserves the refusal precedence: projection error, then `SwiGLU`, then quantization.
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "integration check of fused FFN nominal and refusal branches"
    )]
    fn ffn_fused_matches_unfused_and_preserves_refusal_order() {
        const N_EMBD: usize = 64;
        const N_FF: usize = 64;
        let x = activation(N_EMBD, 7);
        let g = q8_rows(N_FF, N_EMBD);
        let u = q8_rows(N_FF, N_EMBD);
        let d = q8_rows(N_EMBD, N_FF);
        let full = g.len();
        let file = [g.as_slice(), u.as_slice(), d.as_slice()].concat();
        let ok = [g.len(), u.len(), d.len()];

        // 1. Nominal path: fused matches unfused reference
        let (gate_spec, up_spec, down_spec) = make_test_specs(N_EMBD, N_FF, full, ok);
        let mut scratch_fused = FfnScratch::default();
        let got = ffn_q8act_prepared_into(
            &file,
            &[],
            N_EMBD,
            &x,
            gate_spec,
            up_spec,
            down_spec,
            N_FF,
            4,
            4,
            &mut scratch_fused,
        )
        .expect("fused ffn");

        // Unfused reference
        let q = quantize(&x).expect("quantize");
        let (gate_spec, up_spec, down_spec) = make_test_specs(N_EMBD, N_FF, full, ok);
        let mut scratch_unfused = FfnScratch::default();
        projections_q8act_prepared_into(
            &file,
            &[],
            N_EMBD,
            &q,
            &[gate_spec, up_spec],
            4,
            &mut scratch_unfused.rows,
        )
        .expect("projs");
        scratch_unfused.hidden.resize(N_FF, 0);
        swiglu_hidden(
            &scratch_unfused.rows[0],
            &scratch_unfused.rows[1],
            &mut scratch_unfused.hidden,
            true,
        )
        .expect("swiglu");
        let qh = quantize(&scratch_unfused.hidden).expect("quantize hidden");
        projections_q8act_prepared_into(
            &file,
            &[],
            N_FF,
            &qh,
            &[down_spec],
            4,
            &mut scratch_unfused.down_rows,
        )
        .expect("down");
        let want = std::mem::take(&mut scratch_unfused.down_rows[0]);

        assert_eq!(
            got, want,
            "fused FFN must be bit-identical to unfused reference"
        );

        // 2. Refusal precedence:
        // Case A: Gate projection error (truncated payload)
        let mut g_bad = g.clone();
        g_bad.truncate(10);
        let (gate_spec, up_spec, down_spec) =
            make_test_specs(N_EMBD, N_FF, full, [g_bad.len(), u.len(), d.len()]);
        let mut scratch = FfnScratch::default();
        let res = ffn_q8act_prepared_into(
            &file,
            &[],
            N_EMBD,
            &x,
            gate_spec,
            up_spec,
            down_spec,
            N_FF,
            4,
            4,
            &mut scratch,
        );
        assert!(
            matches!(res, Err(FfnError::Matvec(MatvecError::ShortPayload { .. }))),
            "expected ShortPayload for gate, got {res:?}"
        );

        // Case B: Up projection error (truncated payload)
        let mut u_bad = u.clone();
        u_bad.truncate(10);
        let (gate_spec, up_spec, down_spec) =
            make_test_specs(N_EMBD, N_FF, full, [g.len(), u_bad.len(), d.len()]);
        let mut scratch = FfnScratch::default();
        let res = ffn_q8act_prepared_into(
            &file,
            &[],
            N_EMBD,
            &x,
            gate_spec,
            up_spec,
            down_spec,
            N_FF,
            4,
            4,
            &mut scratch,
        );
        assert!(
            matches!(res, Err(FfnError::Matvec(MatvecError::ShortPayload { .. }))),
            "expected ShortPayload for up, got {res:?}"
        );

        // Case C: Both gate and up have errors -> gate error precedes up
        let (gate_spec, up_spec, down_spec) =
            make_test_specs(N_EMBD, N_FF, full, [g_bad.len(), u_bad.len(), d.len()]);
        let mut scratch = FfnScratch::default();
        let res = ffn_q8act_prepared_into(
            &file,
            &[],
            N_EMBD,
            &x,
            gate_spec,
            up_spec,
            down_spec,
            N_FF,
            4,
            4,
            &mut scratch,
        );
        assert!(
            matches!(res, Err(FfnError::Matvec(MatvecError::ShortPayload { .. }))),
            "expected ShortPayload for gate preceding up, got {res:?}"
        );

        // Case D: Quantization refusal on input x
        let mut x_bad = activation(N_EMBD, 1);
        x_bad[0] = i64::MAX;
        let (gate_spec, up_spec, down_spec) = make_test_specs(N_EMBD, N_FF, full, ok);
        let mut scratch = FfnScratch::default();
        let res = ffn_q8act_prepared_into(
            &file,
            &[],
            N_EMBD,
            &x_bad,
            gate_spec,
            up_spec,
            down_spec,
            N_FF,
            4,
            4,
            &mut scratch,
        );
        assert!(
            matches!(res, Err(FfnError::Matvec(MatvecError::Dequant(_)))),
            "expected Dequant overflow on x_bad, got {res:?}"
        );
    }

    #[test]
    fn f32_to_q32_checked_equals_the_f64_formula() {
        // The pre-conversion body, copied: `None` where it returned `Err`.
        fn old(o: f32) -> Option<i64> {
            if !o.is_finite() {
                return None;
            }
            let rounded = (f64::from(o) * 4_294_967_296.0).round();
            if !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&rounded) {
                return None;
            }
            Some(rounded as i64)
        }
        let new = |bits: u32| f32_to_q32_checked(F32::from_bits(bits)).ok();
        let edges: [f32; 22] = [
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(0x007f_ffff),
            2f32.powi(-33),
            -(2f32.powi(-33)),
            0.5,
            -0.5,
            2f32.powi(31),
            f32::from_bits(0x4eff_ffff),
            -(2f32.powi(31)),
            f32::from_bits(0xcf00_0001),
            2f32.powi(32),
            -(2f32.powi(32)),
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        for &e in &edges {
            assert_eq!(new(e.to_bits()), old(e), "edge bits {:#010x}", e.to_bits());
        }
        let mut seed = 0x51f0_0d5e_ed00_0001u64;
        let mut i = 0usize;
        while i < 200_000 {
            // Uniform bit patterns, then patterns whose exponent lands in the rounding range.
            let any = (lcg(&mut seed) >> 32) as u32;
            assert_eq!(new(any), old(f32::from_bits(any)), "bits {any:#010x}");
            let e = 100 + (lcg(&mut seed) % 60) as u32;
            let ranged = (any & 0x8000_0000) | (e << 23) | (any & 0x007f_ffff);
            assert_eq!(
                new(ranged),
                old(f32::from_bits(ranged)),
                "bits {ranged:#010x}"
            );
            i += 1;
        }
    }

    #[test]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_lossless,
        clippy::similar_names,
        reason = "test harnesses use standard RNG, float conversions and carrier variants"
    )]
    fn swiglu_f32_avx2_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx2") {
                return;
            }
            let mut seed = 0x5a17_c001u64;
            for _ in 0..1000 {
                let mut gate = [0i64; 8];
                let mut up = [0i64; 8];
                for l in 0..8 {
                    let raw_g = (lcg(&mut seed) % 40_000).cast_signed() - 20_000;
                    gate[l] = raw_g * (1i64 << 32) / 1000;
                    let raw_u = (lcg(&mut seed) % 40_000).cast_signed() - 20_000;
                    up[l] = raw_u * (1i64 << 32) / 1000;
                }
                let mut scalar_res = [0i64; 8];
                for l in 0..8 {
                    scalar_res[l] = swiglu_one_f32_scalar(gate[l], up[l]).expect("in range");
                }
                let mut avx2_res = [0i64; 8];
                unsafe {
                    swiglu_f32_avx2_8_into(&gate, &up, &mut avx2_res).expect("avx2 into");
                }
                assert_eq!(
                    scalar_res, avx2_res,
                    "AVX2 kernel must equal scalar twin bit for bit"
                );
            }
        }
    }

    #[test]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_lossless,
        reason = "test harnesses use standard RNG and float conversions"
    )]
    fn swiglu_f32_tracks_q32_oracle() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut max_abs_diff = 0.0f64;
        let mut max_rel_diff = 0.0f64;
        let mut count = 0usize;

        for _ in 0..2000 {
            let g_raw = ((lcg(&mut seed) % 12_000).cast_signed() - 6_000) * (1i64 << 32) / 1000;
            let u_raw = ((lcg(&mut seed) % 12_000).cast_signed() - 6_000) * (1i64 << 32) / 1000;

            let q32_val = swiglu_one(g_raw, u_raw).expect("q32");
            let f32_val = swiglu_one_f32_scalar(g_raw, u_raw).expect("f32");

            let q_flt = f64::from(f32::from_bits(q32_to_f32(q32_val).to_bits()));
            let f_flt = f64::from(f32::from_bits(q32_to_f32(f32_val).to_bits()));

            let abs_diff = (q_flt - f_flt).abs();
            let mag = q_flt.abs().max(f_flt.abs());
            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
            }
            if mag > 1e-4 {
                let rel_diff = abs_diff / mag;
                if rel_diff > max_rel_diff {
                    max_rel_diff = rel_diff;
                }
            }
            count += 1;
        }

        println!(
            "SwiGLU f32 vs Q32 oracle ({count} points): max |Δ| = {max_abs_diff:.6e}, max rel = {max_rel_diff:.6e}"
        );
        assert!(
            max_rel_diff < 1e-3,
            "relative deviation {max_rel_diff:e} exceeds 1e-3"
        );
    }

    #[test]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_lossless,
        clippy::similar_names,
        reason = "test harnesses use standard RNG, float conversions and timing buffers"
    )]
    #[ignore = "timing benchmark: run with cargo test --release --lib swiglu_f32_vs_q32_timing -- --ignored --nocapture"]
    fn swiglu_f32_vs_q32_timing() {
        let n_ff = 3456;
        let reps = 200;
        let runs = 5;

        let mut seed = 0xbeef_cafe_1234_5678u64;
        let gate: Vec<i64> = (0..n_ff)
            .map(|_| ((lcg(&mut seed) % 12_000).cast_signed() - 6_000) * (1i64 << 32) / 1000)
            .collect();
        let up: Vec<i64> = (0..n_ff)
            .map(|_| ((lcg(&mut seed) % 12_000).cast_signed() - 6_000) * (1i64 << 32) / 1000)
            .collect();

        let mut h_q32 = vec![0i64; n_ff];
        let mut h_f32 = vec![0i64; n_ff];

        // Warmup
        swiglu_hidden_carrier(&gate, &up, &mut h_q32, false, Carrier::Q32).expect("q32");
        swiglu_hidden_carrier(&gate, &up, &mut h_f32, false, Carrier::F32).expect("f32");

        let mut q32_best_ns = f64::INFINITY;
        for _ in 0..runs {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                swiglu_hidden_carrier(&gate, &up, &mut h_q32, false, Carrier::Q32).expect("q32");
                std::hint::black_box(&h_q32);
            }
            let el = t0.elapsed().as_nanos() as f64 / (f64::from(reps) * n_ff as f64);
            if el < q32_best_ns {
                q32_best_ns = el;
            }
        }

        let mut f32_best_ns = f64::INFINITY;
        for _ in 0..runs {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                swiglu_hidden_carrier(&gate, &up, &mut h_f32, false, Carrier::F32).expect("f32");
                std::hint::black_box(&h_f32);
            }
            let el = t0.elapsed().as_nanos() as f64 / (f64::from(reps) * n_ff as f64);
            if el < f32_best_ns {
                f32_best_ns = el;
            }
        }

        let speedup = q32_best_ns / f32_best_ns;
        println!(
            "\nSwiGLU timing (n_ff = {n_ff}, {reps} reps, single thread):\n\
             Q32 path: {q32_best_ns:.4} ns / element\n\
             F32 path: {f32_best_ns:.4} ns / element\n\
             Speedup:  {speedup:.2}x\n"
        );
    }
}
