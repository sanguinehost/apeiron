//! Exact k-column `Q8_0` projection under power-of-two activation scales: [`gemm_q8_0`], whose
//! column `c` is `matvec_q8act(Q8_0, raw, n_out, n_in, &qs[c])` bit for bit.
//!
//! The contract value of row `r`, column `c` is `round_shift_32(Σ_b d_{r,b}·S_{c,b}·D_{r,c,b})`
//! with `D` the block's integer dot `Σ_l q_l·xq_l`. Two facts make every block term a shift:
//! - a finite `f16` weight scale is exactly `d = m·2^k` in Q32.32 with `|m| < 2^11`, `k ∈ [8, 37]`
//!   ([`f16_split`]), and
//! - `quant_act` scales are powers of two, `S = 2^e`.
//!
//! So `d·S·D = (m·D)·2^(k+e)` with `|m·D| < 2^30`. Per row and column the kernel accumulates
//! `(m·D) << (k − k_min + e − e_min)` in `i64` lanes and applies `2^(k_min + e_min)` once in the
//! final rounding. That is exact while `30 + spread + ⌈log₂ blocks⌉ ≤ 63`
//! ([`spread_limit`]). A (row, column group) past the limit takes the exact `i128` scalar path
//! ([`exact_dot`]).
//!
//! **Layout.** Activation codes are packed per group of 8 columns as `x + 128` bytes, four
//! consecutive inputs of column `c` in 32-bit lane `c`. One `vpdpbusd` against a broadcast weight
//! word then adds four products to each of 8 columns: 8 instructions per block give the 8
//! columns' dots with no horizontal reduction. The `+128` bias is removed with the weight
//! block's code sum.
//!
//! **Refusals.** The kernel returns `None` whenever it cannot reproduce the contract without
//! deciding a refusal itself: a family other than `Q8_0`, a ragged width, a non-power-of-two
//! scale, a non-finite weight scale, a column [`matvec_q8act`] would not take its fast row for, an
//! output past `i64`, or a host without AVX-VNNI. The caller then runs the contract path
//! ([`super::matmat::matmat_q8act`]), which produces the value or the refusal.
//!
//! The AVX2/AVX-VNNI core is not extracted; its equality with [`exact_dot`] and with
//! [`matvec_q8act`] is `tested` (`gemm_equals_the_per_column_contract`).
//!
//! [`matvec_q8act`]: crate::engine::chart::matvec_q8act

use crate::engine::chart::{ActMagnitude, QuantAct};
use crate::engine::value::round_shift_32;
use crate::engine::{silu_fixed, FfnError};
use crate::format::quant::{GGML_TYPE_Q8_0, Q8_0_BLOCK_BYTES};

/// Lanes of one activation quantization block.
const ACT_QK: usize = 32;
/// Columns per packed group: one 32-bit lane each in a 256-bit register.
const LANES: usize = 8;
/// Weight rows per pool task.
const ROWS_PER_TASK: usize = 16;
/// The widest row the stack scratch holds, in blocks (`n_in ≤ 16384`).
const MAX_BLOCKS: usize = 512;

use super::pow2::{f16_split, finish};

/// The largest total exponent spread whose `i64` accumulation of `blocks` terms `|m·D| < 2^30`
/// cannot overflow: `30 + spread + ⌈log₂ blocks⌉ ≤ 63`.
fn spread_limit(blocks: usize) -> u32 {
    let log = usize::BITS - blocks.saturating_sub(1).leading_zeros();
    33u32.saturating_sub(log)
}

/// One call's activation columns, packed.
struct Packed {
    /// Column groups, `⌈k / 8⌉`.
    groups: usize,
    /// `[group][block][quad][lane]`: four `x + 128` bytes of column `8·group + lane`.
    words: Vec<u32>,
    /// `[group][block][lane]`: `e_{c,b} − e_min(c)`; `0` for a zero block and a padding lane.
    shift: Vec<i64>,
    /// `e_min(c)` per column.
    emin: Vec<u32>,
    /// The largest shift in each group.
    spread: Vec<u32>,
}

/// The packing's output pointers. Task `g` writes only group `g`'s `words`/`shift` slices, its
/// own columns' `emin` (columns `[8·g, min(8·g + LANES, k))`) and `spread[g]`; the four
/// allocations outlive the dispatch and no two tasks share a slot.
struct PackPtr {
    words: *mut u32,
    shift: *mut i64,
    emin: *mut u32,
    spread: *mut u32,
}
// SAFETY: `pack` gives each task one disjoint group, so no two tasks write the same `words` or
// `shift` entry, the same `emin` slot or `spread[g]`; the buffers are not reallocated while the
// dispatch runs.
unsafe impl Send for PackPtr {}
// SAFETY: as above.
unsafe impl Sync for PackPtr {}

/// Packs the columns, or `None` when a scale is not a power of two.
///
/// # Complexity
///
/// `O(k·n_in)` byte writes and `O(k·blocks)` exponents, split over `groups` pool tasks (one per
/// `LANES`-wide column group).
fn pack(qs: &[QuantAct], blocks: usize) -> Option<Packed> {
    let k = qs.len();
    let groups = k.div_ceil(LANES);
    let mut words = vec![0x8080_8080u32; groups * blocks * 8 * LANES];
    let mut shift = vec![0i64; groups * blocks * LANES];
    let mut emin = vec![0u32; k];
    let mut spread = vec![0u32; groups];
    let refused = std::sync::atomic::AtomicBool::new(false);
    let ptr = PackPtr {
        words: words.as_mut_ptr(),
        shift: shift.as_mut_ptr(),
        emin: emin.as_mut_ptr(),
        spread: spread.as_mut_ptr(),
    };
    let pk = &ptr;
    crate::engine::chart::pool::Pool::global().run(groups, &|g| {
        let lo_c = g * LANES;
        let hi_c = (lo_c + LANES).min(k);
        let mut group_spread = 0u32;
        for (c, q) in qs.iter().enumerate().take(hi_c).skip(lo_c) {
            let scales = q.scales();
            let mut lo = u32::MAX;
            for &s in scales {
                if s < 0 || (s != 0 && s.count_ones() != 1) {
                    refused.store(true, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
                if s != 0 {
                    lo = lo.min(s.trailing_zeros());
                }
            }
            let lo = if lo == u32::MAX { 0 } else { lo };
            // SAFETY: `c` lies in this task's group, so this is `emin[c]`'s only writer.
            unsafe { *pk.emin.add(c) = lo };
            let lane = c - lo_c;
            let codes = q.codes();
            for (b, &s) in scales.iter().enumerate().take(blocks) {
                let sh = if s == 0 { 0 } else { s.trailing_zeros() - lo };
                // SAFETY: `lane` is this task's lane of group `g`; the entry has no other writer.
                unsafe { *pk.shift.add((g * blocks + b) * LANES + lane) = i64::from(sh) };
                group_spread = group_spread.max(sh);
                for quad in 0..8 {
                    let at = b * ACT_QK + 4 * quad;
                    let mut w = 0u32;
                    for (i, &x) in codes[at..at + 4].iter().enumerate() {
                        w |= u32::from(x.cast_unsigned() ^ 0x80) << (8 * i);
                    }
                    // SAFETY: as for `shift`; the word has no other writer.
                    unsafe { *pk.words.add(((g * blocks + b) * 8 + quad) * LANES + lane) = w };
                }
            }
        }
        // SAFETY: `g` is this task's group, so this is `spread[g]`'s only writer. A group with no
        // real column cannot occur (`groups = ⌈k/LANES⌉`), so every `spread[g]` is assigned.
        unsafe { *pk.spread.add(g) = group_spread };
    });
    if refused.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    Some(Packed {
        groups,
        words,
        shift,
        emin,
        spread,
    })
}

/// The row kernel's output pointers, one per column, each into that column's `Vec<i64>` reserved
/// to `n_out`. Task `t` writes rows `[t·ROWS_PER_TASK, min((t+1)·ROWS_PER_TASK, n_out))` of every
/// column; the row ranges partition `0..n_out`, so no two tasks write the same element and no two
/// tasks alias.
#[cfg(target_arch = "x86_64")]
struct ColPtrs {
    ptrs: Vec<*mut std::mem::MaybeUninit<i64>>,
}
// SAFETY: `gemm_q8_0` gives each task a disjoint row range and one pointer per column into that
// column's own allocation; the kernel stores only rows in the task's range, so no two tasks
// alias.
#[cfg(target_arch = "x86_64")]
unsafe impl Send for ColPtrs {}
// SAFETY: as above.
#[cfg(target_arch = "x86_64")]
unsafe impl Sync for ColPtrs {}

/// The fused `SwiGLU` epilogue's output pointers, one per column, each into that column's
/// `Vec<i64>` pre-sized to `n_ff`. Task `t` writes hidden indices
/// `[t·ROWS_PER_TASK, min((t+1)·ROWS_PER_TASK, n_ff))` of every column; the row ranges partition
/// `0..n_ff`, so no two tasks write the same element.
#[cfg(target_arch = "x86_64")]
struct SwigluPtrs {
    ptrs: Vec<*mut i64>,
}
// SAFETY: `gemm_q8_0_swiglu` gives each task a disjoint hidden-index range and one pointer per
// column into that column's own allocation; no two tasks alias.
#[cfg(target_arch = "x86_64")]
unsafe impl Send for SwigluPtrs {}
// SAFETY: as above.
#[cfg(target_arch = "x86_64")]
unsafe impl Sync for SwigluPtrs {}

/// A row's per-block `(m, k − k_min, 128·Σ q)` and its `k_min` and exponent spread, or `None`
/// for a non-finite scale. The scalar definition and the oracle of `x86::row_scales_avx2`.
#[allow(
    dead_code,
    reason = "scalar oracle of x86::row_scales_avx2 and the non-x86_64 definition"
)]
fn row_scales(row: &[u8], blocks: usize, out: &mut [(i32, i64, i32)]) -> Option<(u32, u32)> {
    let mut kmin = u32::MAX;
    let mut kmax = 0u32;
    for (b, slot) in out.iter_mut().enumerate().take(blocks) {
        let off = b * Q8_0_BLOCK_BYTES;
        let (m, k) = f16_split(u16::from_le_bytes([row[off], row[off + 1]]))?;
        let mut sum = 0i32;
        for &c in &row[off + 2..off + Q8_0_BLOCK_BYTES] {
            sum += i32::from(c.cast_signed());
        }
        *slot = (m, i64::from(k), 128 * sum);
        kmin = kmin.min(k);
        kmax = kmax.max(k);
    }
    for slot in out.iter_mut().take(blocks) {
        slot.1 -= i64::from(kmin);
    }
    Some((kmin, kmax - kmin))
}

/// The exact `i128` contract value of one row against one column: `round_shift_32` of
/// `Σ_b d_b·S_b·D_b` with `d_b` the Q32.32 weight scale. The scalar reference for a (row, group)
/// past the spread limit and the test oracle of the vector core.
///
/// # Complexity
///
/// `O(n_in)` multiply-adds.
#[must_use]
pub(crate) fn exact_dot(row: &[u8], blocks: usize, q: &QuantAct) -> Option<i64> {
    let mut acc: i128 = 0;
    for b in 0..blocks {
        let off = b * Q8_0_BLOCK_BYTES;
        let (m, k) = f16_split(u16::from_le_bytes([row[off], row[off + 1]]))?;
        let mut dot = 0i32;
        for (l, &c) in row[off + 2..off + Q8_0_BLOCK_BYTES].iter().enumerate() {
            dot += i32::from(c.cast_signed()) * i32::from(q.codes()[b * ACT_QK + l]);
        }
        let d = i128::from(m) << k;
        acc = acc.checked_add(
            d.checked_mul(i128::from(q.scales()[b]))?
                .checked_mul(i128::from(dot))?,
        )?;
    }
    i64::try_from(round_shift_32(acc)).ok()
}

/// The k-column `Q8_0` projection, or `None` when the caller must run the contract path.
///
/// # Complexity
///
/// Temporal `O(n_out·n_in·k)` byte multiply-adds, 32 per `vpdpbusd`, plus `O(n_out·blocks·k/8)`
/// vector epilogue steps; the weights are read once per call, and each row's codes are reused
/// from L1 across the `⌈k/8⌉` groups. The activation pack runs over `⌈k/8⌉` pool tasks and the
/// row kernel over `⌈n_out/ROWS_PER_TASK⌉`, each task storing its rows straight into the
/// per-column vectors. Spatial `O(k·n_in)` packed activation bytes plus `O(n_out·k)` output.
#[must_use]
pub(crate) fn gemm_q8_0(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    qs: &[QuantAct],
) -> Option<Vec<Vec<i64>>> {
    #[cfg(target_arch = "x86_64")]
    {
        if ggml_type != GGML_TYPE_Q8_0
            || qs.is_empty()
            || n_out == 0
            || !n_in.is_multiple_of(ACT_QK)
            || n_in / ACT_QK > MAX_BLOCKS
            || !std::is_x86_feature_detected!("avx2")
            || !std::is_x86_feature_detected!("avxvnni")
        {
            return None;
        }
        let blocks = n_in / ACT_QK;
        let row_bytes = blocks * Q8_0_BLOCK_BYTES;
        if raw.len() < n_out.checked_mul(row_bytes)? {
            return None;
        }
        for q in qs {
            // The columns `matvec_q8act` computes on its fast row, whose value is the exact sum.
            if q.len() != n_in
                || q.scales().len() != blocks
                || !matches!(q.magnitude(), ActMagnitude::Exact(_))
                || !crate::engine::chart::q8act_fast_bound_ok(n_in, 128, q, true)
            {
                return None;
            }
        }
        let k = qs.len();
        let packed = pack(qs, blocks)?;
        let limit = spread_limit(blocks);
        // The deleted `inter` buffer refused when `n_out·k` overflowed `usize`; keep that refusal
        // without allocating it, so the overflow still routes to the contract path.
        n_out.checked_mul(k)?;
        let mut cols: Vec<Vec<i64>> = (0..k).map(|_| Vec::with_capacity(n_out)).collect();
        let ptrs = ColPtrs {
            ptrs: cols
                .iter_mut()
                .map(|v| v.spare_capacity_mut().as_mut_ptr())
                .collect(),
        };
        let bad = std::sync::atomic::AtomicBool::new(false);
        crate::engine::chart::pool::Pool::global().run(n_out.div_ceil(ROWS_PER_TASK), &|t| {
            let r0 = t * ROWS_PER_TASK;
            let r1 = ((t + 1) * ROWS_PER_TASK).min(n_out);
            // SAFETY: AVX2 and AVX-VNNI were detected above; `raw` holds `n_out` rows, the packing
            // covers `packed.groups` groups of `blocks` blocks, and `ptrs` addresses the rows
            // `[r0, r1)` of every column in that column's own `Vec`, reserved to `n_out`.
            let ok = unsafe {
                x86::rows(
                    raw,
                    row_bytes,
                    blocks,
                    r0,
                    r1 - r0,
                    qs,
                    &packed,
                    limit,
                    &ptrs,
                )
            };
            if !ok {
                bad.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        });
        if bad.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        for col in &mut cols {
            // SAFETY: every index `0..n_out` of every column was written exactly once above.
            unsafe { col.set_len(n_out) };
        }
        Some(cols)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (ggml_type, raw, n_out, n_in, qs);
        None
    }
}

/// Several `Q8_0` projections of the same activation with one shared pack and one row dispatch:
/// `parts[i] = (ggml_type, raw, n_out)` is projection `i` and every part reads the same input width
/// `n_in` and activation columns `qs`. Returns part `i`'s `k` columns, each equal to the value
/// [`gemm_q8_0`] produces for that part, or `None` when the caller must run the per-part contract
/// path (which owns every refusal).
///
/// The activation packing depends only on `(qs, blocks)` and every part shares `n_in`, so the pack
/// is computed once instead of once per part; the parts' row tasks form one pool dispatch through
/// the prefix sums of `⌈n_out_i / ROWS_PER_TASK⌉`. The same `x86::rows` kernel runs on the same
/// weight bytes and packed activation with the same `limit`, and each part's row ranges partition
/// that part's outputs, so the interleaving changes no word.
///
/// # Complexity
///
/// Temporal `O(Σ n_out_i·n_in·k)` byte multiply-adds plus `O(k·n_in)` packing, over one dispatch of
/// `Σ⌈n_out_i/ROWS_PER_TASK⌉` tasks; spatial `O(k·n_in)` packed bytes plus `O((Σ n_out_i)·k)`
/// output, against one pack and one dispatch per part for the per-part path.
#[must_use]
pub(crate) fn gemm_q8_0_group(
    parts: &[(u32, &[u8], usize)],
    n_in: usize,
    qs: &[QuantAct],
) -> Option<Vec<Vec<Vec<i64>>>> {
    #[cfg(target_arch = "x86_64")]
    {
        if qs.is_empty()
            || !n_in.is_multiple_of(ACT_QK)
            || n_in / ACT_QK > MAX_BLOCKS
            || !std::is_x86_feature_detected!("avx2")
            || !std::is_x86_feature_detected!("avxvnni")
        {
            return None;
        }
        let blocks = n_in / ACT_QK;
        let row_bytes = blocks * Q8_0_BLOCK_BYTES;
        for (ggml_type, raw, n_out) in parts {
            if *ggml_type != GGML_TYPE_Q8_0
                || *n_out == 0
                || raw.len() < n_out.checked_mul(row_bytes)?
            {
                return None;
            }
        }
        for q in qs {
            // The columns `matvec_q8act` computes on its fast row, whose value is the exact sum.
            if q.len() != n_in
                || q.scales().len() != blocks
                || !matches!(q.magnitude(), ActMagnitude::Exact(_))
                || !crate::engine::chart::q8act_fast_bound_ok(n_in, 128, q, true)
            {
                return None;
            }
        }
        let k = qs.len();
        // The deleted `inter` buffer refused when `n_out·k` overflowed `usize`; keep that refusal
        // without allocating it, and build the prefix sums the task-to-part map reads.
        let mut starts = Vec::with_capacity(parts.len() + 1);
        starts.push(0usize);
        for (_, _, n_out) in parts {
            n_out.checked_mul(k)?;
            let next = starts[starts.len() - 1] + n_out.div_ceil(ROWS_PER_TASK);
            starts.push(next);
        }
        let total_tasks = starts[parts.len()];
        let packed = pack(qs, blocks)?;
        let limit = spread_limit(blocks);
        let mut cols: Vec<Vec<Vec<i64>>> = parts
            .iter()
            .map(|(_, _, n_out)| (0..k).map(|_| Vec::with_capacity(*n_out)).collect())
            .collect();
        let col_ptrs: Vec<ColPtrs> = cols
            .iter_mut()
            .map(|part_cols| ColPtrs {
                ptrs: part_cols
                    .iter_mut()
                    .map(|v| v.spare_capacity_mut().as_mut_ptr())
                    .collect(),
            })
            .collect();
        let col_ptrs = &col_ptrs;
        let starts = &starts;
        let bad = std::sync::atomic::AtomicBool::new(false);
        crate::engine::chart::pool::Pool::global().run(total_tasks, &|t| {
            // `t` falls in the first part whose task prefix exceeds it; the row range is the
            // task's `ROWS_PER_TASK` slice of that part.
            let mut i = 0usize;
            while i + 1 < parts.len() && t >= starts[i + 1] {
                i += 1;
            }
            let r0 = (t - starts[i]) * ROWS_PER_TASK;
            let r1 = (r0 + ROWS_PER_TASK).min(parts[i].2);
            // SAFETY: AVX2 and AVX-VNNI were detected above; `parts[i].1` holds `n_out` rows of
            // `row_bytes`, the packing covers `blocks` blocks, and `col_ptrs[i]` addresses the rows
            // `[r0, r1)` of every column of part `i` in that column's own allocation, reserved to
            // `n_out`.
            let ok = unsafe {
                x86::rows(
                    parts[i].1,
                    row_bytes,
                    blocks,
                    r0,
                    r1 - r0,
                    qs,
                    &packed,
                    limit,
                    &col_ptrs[i],
                )
            };
            if !ok {
                bad.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        });
        if bad.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        for (i, part_cols) in cols.iter_mut().enumerate() {
            let n_out = parts[i].2;
            for col in part_cols.iter_mut() {
                // SAFETY: every index `0..n_out` of every column was written exactly once above.
                unsafe { col.set_len(n_out) };
            }
        }
        Some(cols)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (parts, n_in, qs);
        None
    }
}

/// `round(silu(gate)·up / 2^32)` in checked integer arithmetic: the fused epilogue's element,
/// identical to `chart::ffn`'s private `swiglu_one` (the same [`silu_fixed`], checked `i128`
/// product and one [`round_shift_32`] narrowing).
///
/// # Errors
///
/// [`FfnError::Overflow`] when the product or the `i64` narrowing leaves its range;
/// [`silu_fixed`]'s refusal otherwise.
fn swiglu_one(gate: i64, up: i64) -> Result<i64, FfnError> {
    let s = silu_fixed(gate)?;
    let prod = i128::from(s)
        .checked_mul(i128::from(up))
        .ok_or(FfnError::Overflow)?;
    i64::try_from(round_shift_32(prod)).map_err(|_| FfnError::Overflow)
}

/// The fused `[W_gate; W_up]` projection with the `SwiGLU` epilogue:
/// `h[c][i] = round(silu(gate[c][i])·up[c][i] / 2^32)`, where gate/up are row `i` of the two `Q8_0`
/// payloads against column `c`'s packed activation.
///
/// One pooled dispatch: each task owns a range of hidden indices `i`, computes gate row `i` and up
/// row `i` for every column with the same kernel `gemm_q8_0` uses (one [`pack`], the same `group2`
/// cores, [`finish`]/[`exact_dot`]), and writes `h[c][i]` directly. The gate matrix, the up matrix
/// and the separate `SwiGLU` pass are never materialised; the activation is packed once for both
/// channels.
///
/// `None` when the shape or host gate is not met, or when a gate/up value cannot be produced
/// exactly (a non-finite scale, a [`finish`]/[`exact_dot`] overflow, a non-`Exact` activation). The
/// caller then runs the separate gate GEMM, up GEMM and `swiglu_cols`, which owns those refusals in
/// the contract's gate-before-up-before-`SwiGLU` order.
///
/// `Some(Err)` is exactly a `SwiGLU` element refusal ([`silu_fixed`], the checked product or the
/// narrowing) at the minimal `(column, index)` in `swiglu_cols`' flat order.
///
/// # Complexity
///
/// Temporal `O(n_ff·n_in·k)` exact multiply-adds — the gate and up dots — plus `O(n_ff·k)`
/// `swiglu_one` evaluations over `⌈n_ff/ROWS_PER_TASK⌉` pool tasks in one dispatch. Spatial
/// `O(k·n_in)` packed activation bytes plus `O(n_ff·k)` output; no `n_ff·k` gate or up matrix.
#[must_use]
pub(crate) fn gemm_q8_0_swiglu(
    raw_gate: &[u8],
    raw_up: &[u8],
    n_ff: usize,
    n_in: usize,
    qs: &[QuantAct],
) -> Option<Result<Vec<Vec<i64>>, FfnError>> {
    #[cfg(target_arch = "x86_64")]
    {
        if qs.is_empty()
            || n_ff == 0
            || !n_in.is_multiple_of(ACT_QK)
            || n_in / ACT_QK > MAX_BLOCKS
            || !std::is_x86_feature_detected!("avx2")
            || !std::is_x86_feature_detected!("avxvnni")
        {
            return None;
        }
        let blocks = n_in / ACT_QK;
        let row_bytes = blocks * Q8_0_BLOCK_BYTES;
        let need = n_ff.checked_mul(row_bytes)?;
        if raw_gate.len() < need || raw_up.len() < need {
            return None;
        }
        let k = qs.len();
        // The output is `n_ff·k` words; guard the product as `gemm_q8_0` guards its interleaving
        // buffer.
        n_ff.checked_mul(k)?;
        for q in qs {
            // The same fast-row preconditions `gemm_q8_0` requires of every column.
            if q.len() != n_in
                || q.scales().len() != blocks
                || !matches!(q.magnitude(), ActMagnitude::Exact(_))
                || !crate::engine::chart::q8act_fast_bound_ok(n_in, 128, q, true)
            {
                return None;
            }
        }
        let packed = pack(qs, blocks)?;
        let limit = spread_limit(blocks);
        let mut cols: Vec<Vec<i64>> = (0..k).map(|_| vec![0i64; n_ff]).collect();
        let ptrs = SwigluPtrs {
            ptrs: cols.iter_mut().map(Vec::as_mut_ptr).collect(),
        };
        let out = &ptrs;
        let bad = std::sync::atomic::AtomicBool::new(false);
        let first: std::sync::Mutex<Option<(usize, usize, FfnError)>> = std::sync::Mutex::new(None);
        let first_ref = &first;
        crate::engine::chart::pool::Pool::global().run(n_ff.div_ceil(ROWS_PER_TASK), &|t| {
            let r0 = t * ROWS_PER_TASK;
            let rows = (n_ff - r0).min(ROWS_PER_TASK);
            // SAFETY: AVX2 and AVX-VNNI were detected above; both payloads hold `n_ff` rows and the
            // packing covers `packed.groups` groups of `blocks` blocks; task `t` owns hidden indices
            // `[r0, r0 + rows)` and writes each column's own allocation there.
            let ok = unsafe {
                x86::rows_swiglu(
                    raw_gate, raw_up, row_bytes, blocks, r0, rows, qs, &packed, limit, out,
                    first_ref,
                )
            };
            if !ok {
                bad.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        });
        if bad.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        match first
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            Some((_, _, e)) => Some(Err(e)),
            None => Some(Ok(cols)),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (raw_gate, raw_up, n_ff, n_in, qs);
        None
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{
        exact_dot, f16_split, finish, Packed, QuantAct, LANES, MAX_BLOCKS, Q8_0_BLOCK_BYTES,
    };
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_andnot_si256,
        _mm256_blendv_epi8, _mm256_castsi256_si128, _mm256_cmpeq_epi64, _mm256_cmpgt_epi64,
        _mm256_cvtepi32_epi64, _mm256_cvtepu32_epi64, _mm256_extracti128_si256, _mm256_loadu_si256,
        _mm256_mullo_epi32, _mm256_or_si256, _mm256_sad_epu8, _mm256_set1_epi32,
        _mm256_set1_epi64x, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_sllv_epi64,
        _mm256_srlv_epi64, _mm256_storeu_si256, _mm256_sub_epi32, _mm256_sub_epi64,
        _mm256_testc_si256, _mm256_xor_si256, _mm_add_epi64, _mm_cvtsi128_si64, _mm_extract_epi64,
        _mm_loadu_si128,
    };

    /// VEX-encoded `vpdpbusd`: four `u8·i8` products added into each `i32` lane.
    ///
    /// # Safety
    ///
    /// AVX-VNNI must be present.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dpbusd(mut acc: __m256i, u: __m256i, x: __m256i) -> __m256i {
        // SAFETY: the caller established AVX-VNNI.
        unsafe {
            core::arch::asm!(
                "{{vex}} vpdpbusd {acc}, {u}, {x}",
                acc = inout(ymm_reg) acc,
                u = in(ymm_reg) u,
                x = in(ymm_reg) x,
                options(pure, nomem, nostack)
            );
        }
        acc
    }

    /// The 8 columns of one group against two rows: each row's `i64` accumulators, low and high
    /// four lanes, of `Σ_b (m·D) << (k − k_min + e − e_min)`.
    ///
    /// # Safety
    ///
    /// AVX2 and AVX-VNNI; `ra`, `rb` hold `blocks` `Q8_0` blocks; `words` and `shift` point at
    /// the group's `blocks·64` words and `blocks·8` shifts.
    #[target_feature(enable = "avx2")]
    #[allow(
        clippy::cast_ptr_alignment,
        reason = "unaligned 256-bit loads and 32-bit weight-word reads"
    )]
    unsafe fn group2(
        ra: &[u8],
        rb: &[u8],
        sa: &[(i32, i64, i32)],
        sb: &[(i32, i64, i32)],
        blocks: usize,
        words: *const u32,
        shift: *const i64,
    ) -> [__m256i; 4] {
        let (mut a0, mut a1, mut b0, mut b1) = (
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
        );
        for b in 0..blocks {
            // SAFETY: in range by the caller's geometry.
            unsafe {
                let ca = ra.as_ptr().add(b * Q8_0_BLOCK_BYTES + 2);
                let cb = rb.as_ptr().add(b * Q8_0_BLOCK_BYTES + 2);
                let wb = words.add(b * 8 * LANES);
                let mut da = _mm256_setzero_si256();
                let mut db = _mm256_setzero_si256();
                for quad in 0..8 {
                    let act = _mm256_loadu_si256(wb.add(quad * LANES).cast::<__m256i>());
                    let wa = _mm256_set1_epi32(ca.add(4 * quad).cast::<i32>().read_unaligned());
                    let wbv = _mm256_set1_epi32(cb.add(4 * quad).cast::<i32>().read_unaligned());
                    da = dpbusd(da, act, wa);
                    db = dpbusd(db, act, wbv);
                }
                let sh0 = _mm256_loadu_si256(shift.add(b * LANES).cast::<__m256i>());
                let sh1 = _mm256_loadu_si256(shift.add(b * LANES + 4).cast::<__m256i>());
                let (ma, ka, ba) = sa[b];
                let ta = _mm256_mullo_epi32(
                    _mm256_sub_epi32(da, _mm256_set1_epi32(ba)),
                    _mm256_set1_epi32(ma),
                );
                let kav = _mm256_set1_epi64x(ka);
                a0 = _mm256_add_epi64(
                    a0,
                    _mm256_sllv_epi64(
                        _mm256_cvtepi32_epi64(_mm256_castsi256_si128(ta)),
                        _mm256_add_epi64(sh0, kav),
                    ),
                );
                a1 = _mm256_add_epi64(
                    a1,
                    _mm256_sllv_epi64(
                        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(ta, 1)),
                        _mm256_add_epi64(sh1, kav),
                    ),
                );
                let (mb, kb, bb) = sb[b];
                let tb = _mm256_mullo_epi32(
                    _mm256_sub_epi32(db, _mm256_set1_epi32(bb)),
                    _mm256_set1_epi32(mb),
                );
                let kbv = _mm256_set1_epi64x(kb);
                b0 = _mm256_add_epi64(
                    b0,
                    _mm256_sllv_epi64(
                        _mm256_cvtepi32_epi64(_mm256_castsi256_si128(tb)),
                        _mm256_add_epi64(sh0, kbv),
                    ),
                );
                b1 = _mm256_add_epi64(
                    b1,
                    _mm256_sllv_epi64(
                        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(tb, 1)),
                        _mm256_add_epi64(sh1, kbv),
                    ),
                );
            }
        }
        [a0, a1, b0, b1]
    }

    /// The scalar [`super::row_scales`] with each block's code sum taken as
    /// `Σ unsigned(c ^ 0x80) − 32·128`: the sign-bit flip is the two's-complement bijection
    /// `[-128, 127] → [0, 255]`, so `psadbw` sums the unsigned bytes into four 64-bit lanes (at
    /// most `8·255 = 2040` each, no cross-lane carry), and the `−4096` and the lane fold stay in
    /// `i32` (the scalar sum lies in `[-4096, 4064]`). The result feeds the same `128·sum` bias
    /// `group2` subtracts from its `vpdpbusd` accumulator.
    ///
    /// # Safety
    ///
    /// AVX2; `row` holds `blocks` `Q8_0` blocks and `out` has at least `blocks` slots.
    ///
    /// # Complexity
    ///
    /// `O(blocks)` vector steps, one block per iteration.
    #[target_feature(enable = "avx2")]
    #[allow(
        clippy::cast_ptr_alignment,
        reason = "unaligned 256-bit load of a block's 32 code bytes"
    )]
    pub(super) unsafe fn row_scales_avx2(
        row: &[u8],
        blocks: usize,
        out: &mut [(i32, i64, i32)],
    ) -> Option<(u32, u32)> {
        let mut kmin = u32::MAX;
        let mut kmax = 0u32;
        let flip = _mm256_set1_epi8(i8::MIN);
        let zero = _mm256_setzero_si256();
        for (b, slot) in out.iter_mut().enumerate().take(blocks) {
            let off = b * Q8_0_BLOCK_BYTES;
            let (m, k) = f16_split(u16::from_le_bytes([row[off], row[off + 1]]))?;
            // SAFETY: `off + 2 + 32 ≤ row.len()` for `b < blocks`.
            let sad = unsafe {
                let codes = _mm256_loadu_si256(row.as_ptr().add(off + 2).cast::<__m256i>());
                _mm256_sad_epu8(_mm256_xor_si256(codes, flip), zero)
            };
            // Fold the four `u64` `psadbw` lanes.
            let total = {
                let pair = _mm_add_epi64(
                    _mm256_castsi256_si128(sad),
                    _mm256_extracti128_si256(sad, 1),
                );
                _mm_cvtsi128_si64(pair) + _mm_extract_epi64(pair, 1)
            };
            // `total ∈ [0, 8160]`, so `total − 4096 ∈ [−4096, 4064]` narrows to `i32` exactly.
            let sum = i32::try_from(total - 4096).ok()?;
            *slot = (m, i64::from(k), 128 * sum);
            kmin = kmin.min(k);
            kmax = kmax.max(k);
        }
        for slot in out.iter_mut().take(blocks) {
            slot.1 -= i64::from(kmin);
        }
        Some((kmin, kmax - kmin))
    }

    /// [`super::pow2::finish`] of four lanes at once: `round_shift_32(acc·2^s)` with the scalar
    /// refusal set. Returns the four values and whether all four are defined.
    ///
    /// For `s < 32` the sum `acc + 2^(31−s)` is a 128-bit sign-extended pair `(hi, lo)`: `lo` is
    /// the 64-bit wrapping sum, and the carry out of bit 63, `(a & b) | ((a ^ b) & ~(a + b))`,
    /// widens `hi` from the accumulator's sign. The arithmetic shift `>> (32−s)` is then
    /// `srlv(lo) | sllv(hi)`. For `s ≥ 32` the value is `acc << (s − 32)`: it is defined exactly
    /// when shifting it back recovers `acc`, which excludes every `i64` overflow, and when
    /// `s − 32 > 63` only `acc = 0` stays defined. Any lane not on its branch computes a
    /// discarded value.
    ///
    /// # Safety
    ///
    /// AVX2.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn finish4(acc: __m256i, s: __m256i) -> (__m256i, bool) {
        let zero = _mm256_setzero_si256();
        let one = _mm256_set1_epi64x(1);
        let c31 = _mm256_set1_epi64x(31);
        let c32 = _mm256_set1_epi64x(32);
        let c64 = _mm256_set1_epi64x(64);
        // s < 32: `down = 32 − s > 0`.
        let use_a = _mm256_cmpgt_epi64(c32, s);
        let down = _mm256_sub_epi64(c32, s);
        let round = _mm256_sllv_epi64(one, _mm256_sub_epi64(down, one));
        let sum = _mm256_add_epi64(acc, round);
        let carry = {
            let both = _mm256_and_si256(acc, round);
            let diff = _mm256_andnot_si256(sum, _mm256_xor_si256(acc, round));
            _mm256_cmpgt_epi64(zero, _mm256_or_si256(both, diff))
        };
        let hi = _mm256_sub_epi64(_mm256_cmpgt_epi64(zero, acc), carry);
        let res_a = _mm256_or_si256(
            _mm256_srlv_epi64(sum, down),
            _mm256_sllv_epi64(hi, _mm256_sub_epi64(c64, down)),
        );
        // s ≥ 32: `up = s − 32`.
        let use_b = _mm256_cmpgt_epi64(s, c31);
        let up = _mm256_sub_epi64(s, c32);
        let shifted = _mm256_sllv_epi64(acc, up);
        let recovered = {
            let logical = _mm256_srlv_epi64(shifted, up);
            let fill =
                _mm256_sllv_epi64(_mm256_cmpgt_epi64(zero, shifted), _mm256_sub_epi64(c64, up));
            _mm256_or_si256(logical, fill)
        };
        let up_small = _mm256_cmpgt_epi64(c64, up);
        let valid_b = _mm256_or_si256(
            _mm256_and_si256(up_small, _mm256_cmpeq_epi64(recovered, acc)),
            _mm256_andnot_si256(up_small, _mm256_cmpeq_epi64(acc, zero)),
        );
        let valid = _mm256_or_si256(use_a, valid_b);
        let all = _mm256_testc_si256(valid, _mm256_set1_epi64x(-1)) == 1;
        (_mm256_blendv_epi8(res_a, shifted, use_b), all)
    }

    /// Eight `finish` values of one row's group, or `None` when any lane refuses. `s` holds at
    /// least eight lanes of `k_min + e_min(c)`; lanes at or past `k` are don't-cares.
    ///
    /// # Safety
    ///
    /// AVX2; `acc` holds eight lanes and `s` at least eight.
    #[target_feature(enable = "avx2")]
    #[allow(
        clippy::cast_ptr_alignment,
        reason = "unaligned load of the eight-lane accumulator and shift buffers"
    )]
    pub(super) unsafe fn finish8(acc: &[i64; 8], s: &[u32]) -> Option<[i64; 8]> {
        let mut out = [0i64; 8];
        let mut all = true;
        for half in 0..2 {
            // SAFETY: `half` reads lanes `4·half .. 4·half + 4` of both eight-lane buffers and
            // writes the same range of `out`.
            let ok = unsafe {
                let a = _mm256_loadu_si256(acc.as_ptr().add(4 * half).cast::<__m256i>());
                let sv = _mm256_cvtepu32_epi64(_mm_loadu_si128(
                    s.as_ptr().add(4 * half).cast::<__m128i>(),
                ));
                let (res, ok) = finish4(a, sv);
                _mm256_storeu_si256(out.as_mut_ptr().add(4 * half).cast::<__m256i>(), res);
                ok
            };
            all &= ok;
        }
        all.then_some(out)
    }

    /// Rows `r0..r0 + rows` against every column, each value stored into its own column's `Vec`
    /// at index `r0 + r + which`; `false` when an output leaves `i64` or a weight scale is
    /// non-finite (the caller then runs the contract path).
    ///
    /// # Safety
    ///
    /// AVX2 and AVX-VNNI; `raw` holds rows `r0..r0 + rows` of `row_bytes` bytes; every `cols.ptrs[c]`
    /// addresses column `c`'s allocation with room for `r0 + rows` elements, and the caller gives
    /// concurrent invocations disjoint row ranges.
    #[target_feature(enable = "avx2")]
    #[allow(
        clippy::too_many_arguments,
        clippy::many_single_char_names,
        clippy::cast_ptr_alignment,
        clippy::too_many_lines,
        reason = "p/k/r/c/v name the packed inputs, columns, row offset, output column and SIMD value; _mm256_storeu_si256 explicitly accepts the unaligned lane-buffer address; the scalar and vector epilogues read best together"
    )]
    pub(super) unsafe fn rows(
        raw: &[u8],
        row_bytes: usize,
        blocks: usize,
        r0: usize,
        rows: usize,
        qs: &[QuantAct],
        p: &Packed,
        limit: u32,
        cols: &super::ColPtrs,
    ) -> bool {
        let k = qs.len();
        let mut sa = [(0i32, 0i64, 0i32); MAX_BLOCKS];
        let mut sb = [(0i32, 0i64, 0i32); MAX_BLOCKS];
        let mut lanes = [0i64; 16];
        let mut r = 0usize;
        while r < rows {
            let pair = r + 1 < rows;
            let ia = r0 + r;
            let ib = if pair { ia + 1 } else { ia };
            let ra = &raw[ia * row_bytes..(ia + 1) * row_bytes];
            let rb = &raw[ib * row_bytes..(ib + 1) * row_bytes];
            let (Some((ka, spa)), Some((kb, spb))) = (
                // SAFETY: `ra`/`rb` hold `blocks` Q8_0 blocks and `sa`/`sb` have `MAX_BLOCKS`
                // slots (`blocks ≤ MAX_BLOCKS`).
                unsafe { row_scales_avx2(ra, blocks, &mut sa) },
                unsafe { row_scales_avx2(rb, blocks, &mut sb) },
            ) else {
                return false;
            };
            for g in 0..p.groups {
                // SAFETY: the group's words and shifts are in range of the packing.
                let acc = unsafe {
                    group2(
                        ra,
                        rb,
                        &sa[..blocks],
                        &sb[..blocks],
                        blocks,
                        p.words.as_ptr().add(g * blocks * 8 * LANES),
                        p.shift.as_ptr().add(g * blocks * LANES),
                    )
                };
                // SAFETY: `lanes` holds 16 `i64`.
                unsafe {
                    for (i, v) in acc.iter().enumerate() {
                        _mm256_storeu_si256(lanes.as_mut_ptr().add(4 * i).cast::<__m256i>(), *v);
                    }
                }
                let need_exact_a = spa + p.spread[g] > limit;
                let need_exact_b = spb + p.spread[g] > limit;
                let nvalid = (k - g * LANES).min(LANES);
                // Neither row's (row, group) needs the exact path: one vector `finish` per row.
                // A lane or group the vector refuses falls back to the scalar epilogue below, so
                // the refusal set and every stored value are the scalar ones.
                if !need_exact_a && (!pair || !need_exact_b) {
                    let mut shifts = [0u32; 2 * LANES];
                    {
                        let (lo, hi) = shifts.split_at_mut(LANES);
                        for (lane, (a, b)) in lo.iter_mut().zip(hi).enumerate().take(nvalid) {
                            let c = g * LANES + lane;
                            *a = ka + p.emin[c];
                            *b = kb + p.emin[c];
                        }
                    }
                    let mut acc_a = [0i64; LANES];
                    acc_a.copy_from_slice(&lanes[..LANES]);
                    // SAFETY: AVX2 is a target feature of `rows`; both buffers hold 8 lanes.
                    let fa = unsafe { finish8(&acc_a, &shifts[..LANES]) };
                    let fb = if pair {
                        let mut acc_b = [0i64; LANES];
                        acc_b.copy_from_slice(&lanes[LANES..]);
                        // SAFETY: as above.
                        unsafe { finish8(&acc_b, &shifts[LANES..]) }
                    } else {
                        Some([0i64; LANES])
                    };
                    if let (Some(va), Some(vb)) = (fa, fb) {
                        for lane in 0..nvalid {
                            let c = g * LANES + lane;
                            // SAFETY: `cols.ptrs[c]` addresses column `c`'s allocation, and task
                            // ownership makes `r0 + r + which` unique across tasks and in range.
                            unsafe {
                                cols.ptrs[c]
                                    .add(r0 + r)
                                    .write(std::mem::MaybeUninit::new(va[lane]));
                                if pair {
                                    cols.ptrs[c]
                                        .add(r0 + r + 1)
                                        .write(std::mem::MaybeUninit::new(vb[lane]));
                                }
                            }
                        }
                        continue;
                    }
                }
                for lane in 0..nvalid {
                    let c = g * LANES + lane;
                    for (which, (row, kmin, sp)) in
                        [(ra, ka, spa), (rb, kb, spb)].into_iter().enumerate()
                    {
                        if which == 1 && !pair {
                            continue;
                        }
                        let v = if sp + p.spread[g] > limit {
                            exact_dot(row, blocks, &qs[c])
                        } else {
                            finish(lanes[8 * which + lane], kmin + p.emin[c])
                        };
                        let Some(v) = v else {
                            return false;
                        };
                        // SAFETY: `cols.ptrs[c]` addresses column `c`'s allocation, and task
                        // ownership makes `r0 + r + which` unique across tasks and in range.
                        unsafe {
                            cols.ptrs[c]
                                .add(r0 + r + which)
                                .write(std::mem::MaybeUninit::new(v));
                        }
                    }
                }
            }
            r += if pair { 2 } else { 1 };
        }
        true
    }

    /// Hidden indices `r0..r0 + rows` of the fused `[W_gate; W_up]` projection: one packed
    /// activation dots gate row `i` and up row `i` for every column (`group2` shares the activation
    /// loads), the epilogue applies [`super::swiglu_one`], and `h[c][i]` is written directly.
    ///
    /// Returns `false` when a weight scale is non-finite or a gate/up value leaves `i64`; the caller
    /// then runs the separate gate/up/`SwiGLU` path, which owns the refusal. A refused `SwiGLU`
    /// element is recorded as the minimal `(column, index)` in `first`; every row is still scanned
    /// so the minimum over all tasks is the serial `swiglu_hidden` order.
    ///
    /// # Safety
    ///
    /// AVX2 and AVX-VNNI; `raw_gate`/`raw_up` hold indices `r0..r0 + rows` of `row_bytes` bytes;
    /// each `qs[c].codes()` holds `blocks·32` lanes and `qs[c].scales()` `blocks` values; `out` has
    /// one pointer per column into a distinct allocation of at least `n_ff` `i64`s.
    #[target_feature(enable = "avx2")]
    #[allow(
        clippy::too_many_arguments,
        clippy::many_single_char_names,
        clippy::cast_ptr_alignment,
        reason = "p/k/r/c/g/i name the packing, columns, row offset, output column, group and hidden \
                  index; _mm256_storeu_si256 accepts the unaligned lane buffer"
    )]
    pub(super) unsafe fn rows_swiglu(
        raw_gate: &[u8],
        raw_up: &[u8],
        row_bytes: usize,
        blocks: usize,
        r0: usize,
        rows: usize,
        qs: &[QuantAct],
        p: &Packed,
        limit: u32,
        out: &super::SwigluPtrs,
        first: &std::sync::Mutex<Option<(usize, usize, crate::engine::FfnError)>>,
    ) -> bool {
        let k = qs.len();
        let mut sa = [(0i32, 0i64, 0i32); MAX_BLOCKS];
        let mut sb = [(0i32, 0i64, 0i32); MAX_BLOCKS];
        let mut lanes = [0i64; 16];
        let mut r = 0usize;
        while r < rows {
            let i = r0 + r;
            let ga = &raw_gate[i * row_bytes..(i + 1) * row_bytes];
            let ua = &raw_up[i * row_bytes..(i + 1) * row_bytes];
            let (Some((ka, spa)), Some((kb, spb))) = (
                // SAFETY: `ga`/`ua` hold `blocks` Q8_0 blocks and `sa`/`sb` have `MAX_BLOCKS`
                // slots (`blocks ≤ MAX_BLOCKS`).
                unsafe { row_scales_avx2(ga, blocks, &mut sa) },
                unsafe { row_scales_avx2(ua, blocks, &mut sb) },
            ) else {
                return false;
            };
            for g in 0..p.groups {
                // SAFETY: the group's words and shifts are in range of the packing.
                let acc = unsafe {
                    group2(
                        ga,
                        ua,
                        &sa[..blocks],
                        &sb[..blocks],
                        blocks,
                        p.words.as_ptr().add(g * blocks * 8 * LANES),
                        p.shift.as_ptr().add(g * blocks * LANES),
                    )
                };
                // SAFETY: `lanes` holds 16 `i64`.
                unsafe {
                    for (idx, v) in acc.iter().enumerate() {
                        _mm256_storeu_si256(lanes.as_mut_ptr().add(4 * idx).cast::<__m256i>(), *v);
                    }
                }
                for lane in 0..LANES {
                    let c = g * LANES + lane;
                    if c >= k {
                        break;
                    }
                    let gate = if spa + p.spread[g] > limit {
                        exact_dot(ga, blocks, &qs[c])
                    } else {
                        finish(lanes[lane], ka + p.emin[c])
                    };
                    let up = if spb + p.spread[g] > limit {
                        exact_dot(ua, blocks, &qs[c])
                    } else {
                        finish(lanes[8 + lane], kb + p.emin[c])
                    };
                    let (Some(gate), Some(up)) = (gate, up) else {
                        return false;
                    };
                    match super::swiglu_one(gate, up) {
                        Ok(h) => {
                            // SAFETY: `c < k`; the task's hidden-index range is disjoint from every
                            // other task's, so this element has one writer.
                            unsafe { *out.ptrs[c].add(i) = h };
                        }
                        Err(e) => {
                            let mut slot = first
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            if slot.as_ref().is_none_or(|(c0, i0, _)| (c, i) < (*c0, *i0)) {
                                *slot = Some((c, i, e));
                            }
                        }
                    }
                }
            }
            r += 1;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test fixtures on bounded small values"
    )]

    use super::*;
    use crate::engine::chart::{matvec_q8act, quant_act};

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *seed >> 33
    }

    /// `rows` `Q8_0` rows of `n_in` with scale words drawn from `scales` (including subnormal and
    /// negative words) and random codes.
    fn payload(rows: usize, n_in: usize, scales: &[u16], seed: &mut u64) -> Vec<u8> {
        let mut out = Vec::new();
        for _ in 0..rows * (n_in / 32) {
            let s = scales[lcg(seed) as usize % scales.len()];
            out.extend_from_slice(&s.to_le_bytes());
            for _ in 0..32 {
                out.push((lcg(seed) % 256) as u8);
            }
        }
        out
    }

    /// Columns whose 32-wide blocks span `range` bits of magnitude (a wide range drives some
    /// (row, group) pairs past the spread limit onto the exact path), with some zero blocks.
    fn columns(k: usize, n_in: usize, range: u32, seed: &mut u64) -> Vec<QuantAct> {
        (0..k)
            .map(|_| {
                let x: Vec<i64> = (0..n_in)
                    .map(|j| {
                        if (j / 32) % 7 == 3 {
                            return 0;
                        }
                        let mag = 1i64 << (20 + (lcg(seed) as u32 % range.max(1)));
                        (lcg(seed).cast_signed() % (2 * mag + 1)) - mag
                    })
                    .collect();
                quant_act(&x).expect("activations quantize")
            })
            .collect()
    }

    #[test]
    fn f16_split_is_f16_to_fixed() {
        for h in 0..=u16::MAX {
            match (f16_split(h), crate::engine::f16_to_fixed(h)) {
                (Some((m, k)), Ok(v)) => assert_eq!(i64::from(m) << k, v, "{h:#06x}"),
                (None, Err(_)) => {}
                (a, b) => panic!("{h:#06x}: split {a:?} vs fixed {b:?}"),
            }
        }
    }

    /// Every column equals the per-column contract `matvec_q8act`, across widths, column counts
    /// (with ragged groups), weight-scale mixes (normal, subnormal, negative, zero) and activation
    /// ranges that exercise both the vector accumulation and the exact spread path.
    #[test]
    fn gemm_equals_the_per_column_contract() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni")) {
            return;
        }
        let mut seed = 0x6e6d_6d71u64;
        let scale_sets: [&[u16]; 3] = [
            &[0x2000, 0x2400, 0x1c00, 0xa000],
            &[0x0001, 0x03ff, 0x0400, 0x8200],
            &[0x0000, 0x3c00, 0x0010, 0x7bff],
        ];
        for &(n_out, n_in) in &[(1usize, 32usize), (17, 64), (33, 1024), (40, 3584)] {
            for &k in &[1usize, 2, 7, 8, 9, 17, 64] {
                for (si, scales) in scale_sets.iter().enumerate() {
                    for range in [1u32, 8, 30] {
                        let raw = payload(n_out, n_in, scales, &mut seed);
                        let qs = columns(k, n_in, range, &mut seed);
                        let want: Result<Vec<Vec<i64>>, _> = qs
                            .iter()
                            .map(|q| matvec_q8act(GGML_TYPE_Q8_0, &raw, n_out, n_in, q))
                            .collect();
                        match (gemm_q8_0(GGML_TYPE_Q8_0, &raw, n_out, n_in, &qs), want) {
                            (Some(got), Ok(want)) => assert_eq!(
                                got, want,
                                "{n_out}x{n_in} k {k} scales {si} range {range}"
                            ),
                            // A refusal is the contract path's to decide.
                            (None, _) => {}
                            (Some(_), Err(e)) => {
                                panic!("gemm produced values where the contract refuses: {e:?}")
                            }
                        }
                    }
                }
            }
        }
    }

    /// The kernel takes the fast path on ordinary inputs (a test that always fell back would pass
    /// vacuously).
    #[test]
    fn gemm_runs_on_ordinary_inputs() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni")) {
            return;
        }
        let mut seed = 7u64;
        let raw = payload(64, 1024, &[0x2000, 0x2400, 0xa000], &mut seed);
        let qs = columns(16, 1024, 4, &mut seed);
        assert!(gemm_q8_0(GGML_TYPE_Q8_0, &raw, 64, 1024, &qs).is_some());
    }

    /// Every part of [`gemm_q8_0_group`] equals [`gemm_q8_0`] on the same part: one shared pack and
    /// one dispatch produce the per-part values bit for bit, across widths, column counts and
    /// weight-scale mixes. A group refusal implies some part would refuse too, so the caller's
    /// per-part fallback owns the same values and refusals.
    #[test]
    fn gemm_group_equals_the_per_part_gemm() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni")) {
            return;
        }
        let mut seed = 0x6772_6f75u64;
        let scale_sets: [&[u16]; 3] = [
            &[0x2000, 0x2400, 0x1c00, 0xa000],
            &[0x0001, 0x03ff, 0x0400, 0x8200],
            &[0x0000, 0x3c00, 0x0010, 0x7bff],
        ];
        // Non-increasing output widths: a task always reads within its part's rows, so a mutation
        // that misroutes a part's weights stays in bounds and shows as a value difference.
        let outs = [40usize, 18, 5, 1];
        let mut saw_value = false;
        for &(n_in, k) in &[(32usize, 1usize), (64, 3), (256, 7), (1024, 9)] {
            for (si, scales) in scale_sets.iter().enumerate() {
                let raws: Vec<Vec<u8>> = outs
                    .iter()
                    .map(|&n| payload(n, n_in, scales, &mut seed))
                    .collect();
                let qs = columns(k, n_in, 8, &mut seed);
                let parts: Vec<(u32, &[u8], usize)> = raws
                    .iter()
                    .zip(outs)
                    .map(|(r, n)| (GGML_TYPE_Q8_0, r.as_slice(), n))
                    .collect();
                let want: Vec<Option<Vec<Vec<i64>>>> = parts
                    .iter()
                    .map(|&(t, r, n)| gemm_q8_0(t, r, n, n_in, &qs))
                    .collect();
                match gemm_q8_0_group(&parts, n_in, &qs) {
                    Some(cols) => {
                        saw_value = true;
                        for (i, w) in want.iter().enumerate() {
                            assert_eq!(
                                cols[i],
                                w.clone().expect("group value needs a per-part value"),
                                "part {i} n_in {n_in} k {k} scales {si}"
                            );
                        }
                    }
                    None => assert!(
                        want.iter().any(Option::is_none),
                        "group deferred where every part has a value: n_in {n_in} k {k} scales {si}"
                    ),
                }
            }
        }
        assert!(
            saw_value,
            "the group must take its fast path on ordinary inputs"
        );
    }

    /// A group defers (`None`) when one part's payload is short or carries a non-finite block
    /// scale, so the caller runs the per-part contract path; the same part alone still produces a
    /// value, so the refusal is not vacuous.
    #[test]
    fn gemm_group_defers_on_a_short_or_non_finite_part() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni")) {
            return;
        }
        let mut seed = 0x5f6f_7274u64;
        let (n_in, k, n_out) = (64usize, 2usize, 16usize);
        let scales = &[0x2000, 0x2400, 0x1c00, 0xa000];
        let good0 = payload(n_out, n_in, scales, &mut seed);
        let good1 = payload(n_out, n_in, scales, &mut seed);
        let qs = columns(k, n_in, 4, &mut seed);
        assert!(gemm_q8_0(GGML_TYPE_Q8_0, &good0, n_out, n_in, &qs).is_some());

        // One byte short of `n_out·row_bytes`.
        let short = &good1[..good1.len() - 1];
        let parts = [
            (GGML_TYPE_Q8_0, good0.as_slice(), n_out),
            (GGML_TYPE_Q8_0, short, n_out),
        ];
        assert!(gemm_q8_0_group(&parts, n_in, &qs).is_none(), "short part");

        // A non-finite (`f16` +inf) scale in the second part's first block; `rows` refuses it.
        let mut bad = good1.clone();
        bad[0..2].copy_from_slice(&0x7C00u16.to_le_bytes());
        let parts = [
            (GGML_TYPE_Q8_0, good0.as_slice(), n_out),
            (GGML_TYPE_Q8_0, bad.as_slice(), n_out),
        ];
        assert!(
            gemm_q8_0_group(&parts, n_in, &qs).is_none(),
            "non-finite scale"
        );
    }

    /// The vector `finish` (`x86::finish8`) equals the scalar `finish` on every lane: both
    /// rounding directions, `s == 32`, shifts past 63, and accumulators whose shifted value
    /// overflows `i64`.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn finish8_matches_the_scalar_finish() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let accs = [
            0i64,
            1,
            -1,
            2,
            -2,
            3,
            -3,
            1 << 40,
            -(1 << 40),
            i64::MAX,
            i64::MIN,
            i64::MAX / 3,
            (1 << 62) + 12345,
            -((1 << 62) + 12345),
        ];
        let shifts = [
            0u32, 1, 7, 8, 16, 31, 32, 33, 40, 47, 63, 64, 65, 66, 67, 96,
        ];
        for &a in &accs {
            for &s in &shifts {
                let got = unsafe { x86::finish8(&[a; 8], &[s; 8]) };
                let mut want = [0i64; 8];
                let mut ok = true;
                for w in &mut want {
                    match super::finish(a, s) {
                        Some(v) => *w = v,
                        None => ok = false,
                    }
                }
                assert_eq!(got, ok.then_some(want), "acc {a} s {s}");
            }
        }
        // Distinct accumulator and shift per lane: the per-lane blend and carry paths.
        let mut seed = 0x00dd_ccbb_aa99_8877u64;
        for _ in 0..4096 {
            let mut a = [0i64; 8];
            let mut s = [0u32; 8];
            let mut want = [0i64; 8];
            let mut ok = true;
            for l in 0..8 {
                let mag = 1i64 << (lcg(&mut seed) as u32 % 63);
                let delta = (lcg(&mut seed) % 1024).cast_signed();
                a[l] = if lcg(&mut seed) & 1 == 0 {
                    mag + delta
                } else {
                    mag.saturating_neg() + delta
                };
                s[l] = (lcg(&mut seed) as u32 % 72) + 1;
                match super::finish(a[l], s[l]) {
                    Some(v) => want[l] = v,
                    None => ok = false,
                }
            }
            assert_eq!(
                unsafe { x86::finish8(&a, &s) },
                ok.then_some(want),
                "a {a:?} s {s:?}"
            );
        }
    }

    /// The AVX2 code sum (`x86::row_scales_avx2`) equals the scalar `row_scales` for the extreme
    /// byte values, an alternating pattern, a ragged last block and thousands of `LCG` blocks.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn row_scales_simd_matches_the_scalar_oracle() {
        const SCALE: u16 = 0x3C00; // f16 1.0: finite, so both definitions return `Some`
        const B: usize = 5;

        fn row_of(blocks: usize, mut codes: impl FnMut(usize, usize) -> u8) -> Vec<u8> {
            let mut row = Vec::with_capacity(blocks * Q8_0_BLOCK_BYTES);
            for b in 0..blocks {
                row.extend_from_slice(&SCALE.to_le_bytes());
                for l in 0..32 {
                    row.push(codes(b, l));
                }
            }
            row
        }

        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let mut cases = vec![
            row_of(B, |_, _| 0x00),
            row_of(B, |_, _| 0x80),
            row_of(B, |_, _| 0x7F),
            row_of(B, |_, l| if l % 2 == 0 { 0x00 } else { 0xFF }),
            // The final block alone carries a mixed pattern: the ragged edge of the row.
            row_of(B, |b, l| if b + 1 == B { (0x40 + l) as u8 } else { 0x00 }),
        ];
        let mut seed = 0x0123_4567_89ab_cdefu64;
        for _ in 0..2048 {
            cases.push(row_of(B, |_, _| (lcg(&mut seed) % 256) as u8));
        }

        for row in &cases {
            let mut simd = vec![(0i32, 0i64, 0i32); B];
            let mut scalar = vec![(0i32, 0i64, 0i32); B];
            // SAFETY: AVX2 detected; the row holds `B` blocks and both buffers hold `B` slots.
            let got = unsafe { x86::row_scales_avx2(row, B, &mut simd) };
            let want = super::row_scales(row, B, &mut scalar);
            assert_eq!(got, want);
            assert_eq!(simd, scalar);
        }
    }

    /// A `rows × n_in` `Q8_0` payload whose every block has the largest finite scale `0x7BFF` and
    /// all code bytes `127`, so the row dot is maximal.
    fn max_payload(rows: usize, n_in: usize) -> Vec<u8> {
        let blocks = n_in / 32;
        let mut raw = vec![0u8; rows * blocks * Q8_0_BLOCK_BYTES];
        for r in 0..rows {
            for b in 0..blocks {
                let off = (r * blocks + b) * Q8_0_BLOCK_BYTES;
                raw[off..off + 2].copy_from_slice(&0x7BFFu16.to_le_bytes());
                for l in 0..32 {
                    raw[off + 2 + l] = 127;
                }
            }
        }
        raw
    }

    /// The fused `[W_gate; W_up]` + `SwiGLU` equals the separate gate GEMM, up GEMM and the serial
    /// `swiglu_hidden` loop bit for bit, over random `Q8_0` payloads with mixed `f16` exponents
    /// (normal, subnormal, negative, zero) and activation ranges that drive both the vector
    /// accumulation and the exact spread path. A gate/up GEMM refusal must defer (`None`) to the
    /// separate path, and a refused `SwiGLU` element must carry the same `FfnError`.
    #[test]
    fn swiglu_fused_equals_the_separate_path() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni")) {
            return;
        }
        let mut seed = 0x5a17_5a17u64;
        let scale_sets: [&[u16]; 3] = [
            &[0x2000, 0x2400, 0x1c00, 0xa000],
            &[0x0001, 0x03ff, 0x0400, 0x8200],
            &[0x0000, 0x3c00, 0x0010, 0x7bff],
        ];
        for &(n_ff, n_in) in &[(1usize, 32usize), (17, 64), (33, 128), (40, 256)] {
            for &k in &[1usize, 2, 7, 8, 9] {
                for (si, scales) in scale_sets.iter().enumerate() {
                    for range in [1u32, 8, 30] {
                        let g_raw = payload(n_ff, n_in, scales, &mut seed);
                        let u_raw = payload(n_ff, n_in, scales, &mut seed);
                        let qs = columns(k, n_in, range, &mut seed);
                        let gate: Result<Vec<Vec<i64>>, _> = qs
                            .iter()
                            .map(|q| matvec_q8act(GGML_TYPE_Q8_0, &g_raw, n_ff, n_in, q))
                            .collect();
                        let up: Result<Vec<Vec<i64>>, _> = qs
                            .iter()
                            .map(|q| matvec_q8act(GGML_TYPE_Q8_0, &u_raw, n_ff, n_in, q))
                            .collect();
                        let fused = gemm_q8_0_swiglu(&g_raw, &u_raw, n_ff, n_in, &qs);
                        match (gate, up) {
                            (Ok(gate), Ok(up)) => {
                                let mut want: Vec<Vec<i64>> =
                                    (0..k).map(|_| vec![0i64; n_ff]).collect();
                                let mut refused: Option<FfnError> = None;
                                for c in 0..k {
                                    if let Err(e) = crate::engine::chart::ffn::swiglu_hidden(
                                        &gate[c],
                                        &up[c],
                                        &mut want[c],
                                        false,
                                    ) {
                                        refused = Some(e);
                                        break;
                                    }
                                }
                                match (fused, refused) {
                                    (Some(Ok(got)), None) => assert_eq!(
                                        got, want,
                                        "n_ff {n_ff} n_in {n_in} k {k} scales {si} range {range}"
                                    ),
                                    (Some(Err(e)), Some(r)) => assert_eq!(
                                        e, r,
                                        "n_ff {n_ff} n_in {n_in} k {k} scales {si} range {range}"
                                    ),
                                    // A gate/up value the fused kernel cannot prove exact: the
                                    // separate path owns the value (covered by the ordinary test).
                                    (None, _) => {}
                                    (Some(Ok(_)), Some(r)) => {
                                        panic!("fused produced a value where SwiGLU refuses: {r:?}")
                                    }
                                    (Some(Err(e)), None) => {
                                        panic!("fused refused where the separate path is Ok: {e:?}")
                                    }
                                }
                            }
                            // A gate or up GEMM refusal: the fused path must defer so the separate
                            // path returns the contract's `MatvecError` in gate-before-up order.
                            _ => assert!(
                                fused.is_none(),
                                "a gate/up GEMM refusal must defer to the separate path"
                            ),
                        }
                    }
                }
            }
        }
    }

    /// The fused kernel takes its fast path on ordinary inputs (a test that always fell back would
    /// pass vacuously).
    #[test]
    fn swiglu_fused_runs_on_ordinary_inputs() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni")) {
            return;
        }
        let mut seed = 11u64;
        let g_raw = payload(64, 1024, &[0x2000, 0x2400, 0xa000], &mut seed);
        let u_raw = payload(64, 1024, &[0x2000, 0x2400, 0xa000], &mut seed);
        let qs = columns(8, 1024, 4, &mut seed);
        assert!(matches!(
            gemm_q8_0_swiglu(&g_raw, &u_raw, 64, 1024, &qs),
            Some(Ok(_))
        ));
    }

    /// A `SwiGLU` element whose product leaves `i64` refuses through the fused path with the same
    /// `FfnError` the serial `swiglu_hidden` returns, at the minimal `(column, index)`.
    #[test]
    fn swiglu_fused_reports_a_refusing_element() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("avxvnni")) {
            return;
        }
        let (n_ff, n_in) = (2usize, 32usize);
        let raw = max_payload(n_ff, n_in);
        // `127·2^24` quantizes on the power-of-two grid with code `127`: gate and up are ≈5.7e17,
        // so `silu(gate)·up / 2^32` leaves `i64`.
        let q = quant_act(&vec![127i64 << 24; n_in]).expect("dequantizes");
        let gate = matvec_q8act(GGML_TYPE_Q8_0, &raw, n_ff, n_in, &q).expect("gate contract");
        let up = matvec_q8act(GGML_TYPE_Q8_0, &raw, n_ff, n_in, &q).expect("up contract");
        let mut want = vec![0i64; n_ff];
        let refuse = crate::engine::chart::ffn::swiglu_hidden(&gate, &up, &mut want, false);
        assert_eq!(refuse, Err(FfnError::Overflow), "the fixture must refuse");
        assert_eq!(
            gemm_q8_0_swiglu(&raw, &raw, n_ff, n_in, &[q]),
            Some(Err(FfnError::Overflow))
        );
    }

    #[test]
    fn spread_limit_keeps_the_sum_in_i64() {
        assert_eq!(spread_limit(1), 33);
        assert_eq!(spread_limit(32), 28);
        assert_eq!(spread_limit(112), 26);
        for blocks in 1..=MAX_BLOCKS {
            let log = usize::BITS - (blocks - 1).leading_zeros();
            assert!(30 + spread_limit(blocks) + log <= 63, "{blocks}");
        }
    }
}
