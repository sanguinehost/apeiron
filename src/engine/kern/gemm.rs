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
/// for a non-finite scale.
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
        exact_dot, finish, row_scales, Packed, QuantAct, LANES, MAX_BLOCKS, Q8_0_BLOCK_BYTES,
    };
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_castsi256_si128, _mm256_cvtepi32_epi64,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_mullo_epi32, _mm256_set1_epi32,
        _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_sllv_epi64, _mm256_storeu_si256,
        _mm256_sub_epi32,
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
        reason = "p/k/r/c/v name the packed inputs, columns, row offset, output column and SIMD value; _mm256_storeu_si256 explicitly accepts the unaligned lane-buffer address"
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
                row_scales(ra, blocks, &mut sa),
                row_scales(rb, blocks, &mut sb),
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
                for lane in 0..LANES {
                    let c = g * LANES + lane;
                    if c >= k {
                        break;
                    }
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
                row_scales(ga, blocks, &mut sa),
                row_scales(ua, blocks, &mut sb),
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
