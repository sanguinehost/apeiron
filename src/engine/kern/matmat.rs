//! k-column int8-activation projection (L1): [`matmat_q8act`], the int8-activation matrix product
//! whose column `c` is [`matvec_q8act`] on `qs[c]` bit for bit.
//! Owned by APEIRON-DOM-L1-T1.
//!
//! For `Q8_0` and `Q5_0` each weight block is decoded once and dotted against all `k` activation
//! blocks while its bytes are in registers, so the weight payload is read once per call instead of
//! once per column. The additive decomposition is the one [`matvec_q8act`]'s fast row uses —
//! `Σ_b S_b·(d_b·Σ_l q_l·xq_{c,l})` in exact `i128` — so the integers are equal, the single
//! [`round_shift_32`] epilogue is equal, and the `i64` range check is equal. The fast kernel runs
//! only when [`matvec_q8act`] itself would take its fast row for every column (`Q8_0`/`Q5_0`, a
//! whole number of 32-wide activation blocks, an exact activation magnitude and a call-level bound
//! that rules out its `i64`/`i128` overflow); otherwise, and for every other family, the call is
//! the per-column [`matvec_q8act`], which is the contract itself.
//!
//! The AVX2 cores are reached only behind `is_x86_feature_detected!("avx2")` and are the only
//! unsafe code here; the portable twin is the fallback and the test oracle. Their equality with
//! the portable twin is `tested`, not extracted (trust boundary: `docs/epics/decode-dominance.md`
//! §L1).
#![allow(
    dead_code,
    reason = "the L1 entry point and its cores land in this wave; their W2 consumers (spec_greedy, \
              batched prefill) are still to be wired, and the tests call them under cfg(test)"
)]

use crate::engine::chart::{matvec_q8act, ActMagnitude, QuantAct};
use crate::engine::f16_to_fixed;
use crate::engine::value::round_shift_32;
use crate::engine::MatvecError;
use crate::format::quant::{
    GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, Q5_0_BLOCK_BYTES, Q5_0_QK, Q8_0_BLOCK_BYTES, Q8_0_QK,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256i, _mm256_abs_epi8, _mm256_add_epi8, _mm256_and_si256, _mm256_andnot_si256,
    _mm256_castsi256_si128, _mm256_cmpeq_epi8, _mm256_extracti128_si256, _mm256_loadu_si256,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_set1_epi8, _mm256_set_m128i,
    _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_sign_epi8,
    _mm256_slli_epi16, _mm256_sub_epi16, _mm_add_epi32, _mm_and_si128, _mm_cvtsi128_si32,
    _mm_loadu_si128, _mm_set1_epi8, _mm_shuffle_epi32, _mm_srli_epi16,
};

/// Lanes of one activation quantization block.
const ACT_QK: usize = 32;

/// The largest column count the fast kernel handles; larger `k` takes the per-column reference.
const MAX_K: usize = 16;

/// The largest finite `f16` magnitude in Q32.32: `65504 · 2^32`, as in the chart's bound.
const F16_MAX_FIXED: i128 = 65504i128 << 32;

/// The two families with a shared weight-block decode.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    /// `Q8_0`: `d` at 0, 32 signed code bytes at 2.
    Q8_0,
    /// `Q5_0`: `d` at 0, `qh` at 2, 16 packed `qs` bytes at 6.
    Q5_0,
}

/// The weight payload geometry of one call: `n_in` is a whole number of 32-wide activation blocks.
#[derive(Clone, Copy)]
struct Geom {
    /// The activation blocks per weight row, `n_in / 32`.
    blocks: usize,
    /// The raw bytes per weight block.
    block_bytes: usize,
    /// The raw bytes per weight row, `blocks · block_bytes`.
    row_bytes: usize,
}

/// The per-column terminal state of a fast-kernel call.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColumnState {
    /// Still matching [`matvec_q8act`]'s fast row and within `i64` so far.
    Live,
    /// A row's rounded value left `i64`: the column's refusal is [`MatvecError::OutputOverflow`].
    Overflow,
    /// A row's fast value was `None` (a non-finite block scale): the column falls back to
    /// [`matvec_q8act`] itself.
    Fallback,
}

/// The fast kernel's per-row outcome.
#[allow(
    clippy::large_enum_variant,
    reason = "the fixed `MAX_K`-wide `i128` array is stack scratch on the row hot path; boxing it \
              would add an allocation per weight row, and the unit variants carry no payload"
)]
enum RowOutcome {
    /// Every live column's exact `i128` accumulator for this row.
    Values([i128; MAX_K]),
    /// A block scale was `Inf`/`NaN`, so [`matvec_q8act`]'s fast row returns `None` for the row.
    NonFinite,
    /// A checked `i64`/`i128` step left range, which the call-level bound makes unreachable; the
    /// caller takes the per-column reference rather than guess.
    Bail,
}

/// The k-column int8-activation projection: `out[c] = matvec_q8act(ggml_type, raw, n_out, n_in,
/// &qs[c])` for every `c`, including the same `Err`.
///
/// `Q8_0` and `Q5_0` with a whole-block width, `k ≤ 16`, exact activation magnitudes and a
/// call-level overflow bound run the shared-decode kernel; every other input is the per-column
/// [`matvec_q8act`]. Either way column `c` equals the per-column call bit for bit, so a caller may
/// read the k columns as k independent projections.
///
/// # Errors
///
/// The first column's [`MatvecError`], in column order: the call is `Ok` exactly when every column
/// is `Ok`, and otherwise refuses with the refusal of the lowest-indexed refusing column — the
/// same [`MatvecError`] [`matvec_q8act`] returns for that column.
///
/// # Complexity
///
/// Temporal `O(n_out·n_in·k)` exact integer multiply-adds — the same count as `k` separate calls,
/// with the weight payload's `O(n_out·n_in)` bytes read once per call instead of `k` times and the
/// `k·n_out·n_in` activation code bytes read per column. Spatial `O(n_out·k)` for the output plus
/// `O(k)` column state; the kernel keeps no `O(n_in)` scratch.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the one `rounded as i64` narrowing is immediately preceded by an explicit \
              `i64::MIN`/`i64::MAX` range check; `TryFrom` emits an unmodeled external the pinned \
              fail-closed extraction audit rejects"
)]
pub(crate) fn matmat_q8act(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    qs: &[QuantAct],
) -> Result<Vec<Vec<i64>>, MatvecError> {
    // The power-of-two-scale GEMM (`super::gemm`) computes the same values when it can; it
    // defers every refusal to the path below.
    if let Some(cols) = super::gemm::gemm_q8_0(ggml_type, raw, n_out, n_in, qs) {
        return Ok(cols);
    }
    let k = qs.len();
    let Some(family) = fast_family(ggml_type, n_in, qs) else {
        return per_column(ggml_type, raw, n_out, n_in, qs);
    };
    let block_bytes = match family {
        Family::Q8_0 => Q8_0_BLOCK_BYTES,
        Family::Q5_0 => Q5_0_BLOCK_BYTES,
    };
    let blocks = n_in / ACT_QK;
    let row_bytes = blocks
        .checked_mul(block_bytes)
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
    let geom = Geom {
        blocks,
        block_bytes,
        row_bytes,
    };
    let avx = avx2_available();
    #[cfg(target_arch = "x86_64")]
    if avx && family == Family::Q8_0 {
        if let Some(cols) = pooled_q8_0(raw, n_out, geom, qs) {
            return Ok(cols);
        }
    }
    let mut columns: Vec<Vec<i64>> = (0..k).map(|_| Vec::with_capacity(n_out)).collect();
    let mut state = vec![ColumnState::Live; k];
    let mut o = 0usize;
    while o < n_out {
        if !state.contains(&ColumnState::Live) {
            break;
        }
        let start = o * row_bytes;
        let row = &raw[start..start + row_bytes];
        match row_accs(family, row, geom, qs, k, &state, avx) {
            RowOutcome::NonFinite => {
                let mut c = 0usize;
                while c < k {
                    if state[c] == ColumnState::Live {
                        state[c] = ColumnState::Fallback;
                    }
                    c += 1;
                }
            }
            RowOutcome::Bail => return per_column(ggml_type, raw, n_out, n_in, qs),
            RowOutcome::Values(accs) => {
                let mut c = 0usize;
                while c < k {
                    if state[c] == ColumnState::Live {
                        let rounded = round_shift_32(accs[c]);
                        if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
                            state[c] = ColumnState::Overflow;
                        } else {
                            columns[c].push(rounded as i64);
                        }
                    }
                    c += 1;
                }
            }
        }
        o += 1;
    }
    let mut result: Vec<Vec<i64>> = Vec::with_capacity(k);
    let mut c = 0usize;
    while c < k {
        match state[c] {
            ColumnState::Live => result.push(std::mem::take(&mut columns[c])),
            ColumnState::Overflow => return Err(MatvecError::OutputOverflow),
            ColumnState::Fallback => {
                result.push(matvec_q8act(ggml_type, raw, n_out, n_in, &qs[c])?);
            }
        }
        c += 1;
    }
    Ok(result)
}

/// The `Q8_0` fast path split by rows across the persistent pool: each task runs
/// [`q8_0_row_kcols_avx2`] over its rows for all `k` columns, so every weight row is read from
/// memory once per call. Column `c` equals [`matvec_q8act`] on `qs[c]` bit for bit (the same exact
/// `i128` row sums and one [`round_shift_32`]).
///
/// Returns `None` when any row hits a non-finite scale or an output leaves `i64`; the caller then
/// runs the serial kernel, which carries the exact refusal and fallback semantics.
///
/// # Complexity
///
/// `O(n_out·n_in·k)` multiply-adds over the pool; `O(n_out·k)` output words, interleaved once and
/// split into columns once.
#[cfg(target_arch = "x86_64")]
#[allow(
    clippy::many_single_char_names,
    reason = "k columns, r rows, o and c indices are the kernel's standard symbols"
)]
fn pooled_q8_0(raw: &[u8], n_out: usize, geom: Geom, qs: &[QuantAct]) -> Option<Vec<Vec<i64>>> {
    use std::sync::atomic::{AtomicBool, Ordering};
    let k = qs.len();
    let mut inter = vec![0i64; n_out.checked_mul(k)?];
    let bad = AtomicBool::new(false);
    let rows_per = n_out
        .div_ceil(crate::engine::par::row_chunk_budget(n_out))
        .max(1);
    crate::engine::chart::pool::Pool::global().chunks_mut(
        &mut inter,
        rows_per * k,
        &|_, start, dst| {
            let r0 = start / k;
            let mut acc = [0i128; MAX_K];
            let mut r = 0usize;
            while r < dst.len() / k {
                let o = r0 + r;
                let row = &raw[o * geom.row_bytes..(o + 1) * geom.row_bytes];
                acc[..k].fill(0);
                // SAFETY: the caller established AVX2, `raw.len() >= n_out·row_bytes`, and
                // `fast_family` checked every column's geometry and the narrow overflow bound.
                let ok = unsafe {
                    crate::engine::chart::q8_0_row_kcols_avx2(row, geom.blocks, qs, &mut acc[..k])
                };
                if ok.is_none() {
                    bad.store(true, Ordering::Relaxed);
                    return;
                }
                let mut c = 0usize;
                while c < k {
                    let Ok(v) = i64::try_from(round_shift_32(acc[c])) else {
                        bad.store(true, Ordering::Relaxed);
                        return;
                    };
                    dst[r * k + c] = v;
                    c += 1;
                }
                r += 1;
            }
        },
    );
    if bad.load(Ordering::Relaxed) {
        return None;
    }
    let mut cols: Vec<Vec<i64>> = (0..k).map(|_| Vec::with_capacity(n_out)).collect();
    let mut o = 0usize;
    while o < n_out {
        let mut c = 0usize;
        while c < k {
            cols[c].push(inter[o * k + c]);
            c += 1;
        }
        o += 1;
    }
    Some(cols)
}

/// The per-column [`matvec_q8act`] fallback, which is the contract itself.
///
/// Refuses with the lowest-indexed column's refusal, matching [`matmat_q8act`]'s error contract.
///
/// # Complexity
///
/// The sum of the per-column [`matvec_q8act`] costs, `O(k)` calls.
fn per_column(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    qs: &[QuantAct],
) -> Result<Vec<Vec<i64>>, MatvecError> {
    let mut out: Vec<Vec<i64>> = Vec::with_capacity(qs.len());
    let mut c = 0usize;
    while c < qs.len() {
        out.push(matvec_q8act(ggml_type, raw, n_out, n_in, &qs[c])?);
        c += 1;
    }
    Ok(out)
}

/// The fast kernel's family, or `None` when any column must take the per-column reference.
///
/// True exactly when [`matvec_q8act`]'s `fast` flag is true for every column: a `Q8_0`/`Q5_0`
/// type, a whole number of 32-wide activation blocks, `1 ≤ k ≤ 16`, each column's length matching
/// `n_in` with one scale per block, an exact [`ActMagnitude`] (`NoQ32` and the `i128` total
/// overflow both route [`matvec_q8act`] elsewhere), and [`fast_bound_ok`] — the chart's private
/// `q8act_fast_bound_ok(…, narrow = true)` predicate, re-derived here from the public `scales()`
/// accessor so this file edits no other module.
///
/// # Complexity
///
/// `O(k·n_in)` time in the per-column [`QuantAct::magnitude`], `O(1)` extra space.
fn fast_family(ggml_type: u32, n_in: usize, qs: &[QuantAct]) -> Option<Family> {
    let family = match ggml_type {
        GGML_TYPE_Q8_0 => Family::Q8_0,
        GGML_TYPE_Q5_0 => Family::Q5_0,
        _ => return None,
    };
    if !n_in.is_multiple_of(ACT_QK) || qs.is_empty() || qs.len() > MAX_K {
        return None;
    }
    let blocks = n_in / ACT_QK;
    let top: i128 = match family {
        Family::Q8_0 => 128,
        Family::Q5_0 => 16,
    };
    let mut c = 0usize;
    while c < qs.len() {
        let q = &qs[c];
        if q.len() != n_in || q.scales().len() != blocks {
            return None;
        }
        match q.magnitude() {
            ActMagnitude::Exact(_) => {}
            ActMagnitude::NoQ32 | ActMagnitude::TotalOverflow => return None,
        }
        if !fast_bound_ok(n_in, top, q) {
            return None;
        }
        c += 1;
    }
    Some(family)
}

/// The chart's `q8act_fast_bound_ok(n_in, top, q, narrow = true)`, re-derived here.
///
/// `S_max = max_b |S_b|`; the narrow condition `S_max·(32·top·127) ≤ i64::MAX` keeps each block's
/// `S_b·I_b` in `i64`, and `n_in·(65504·2^32)·top·127·S_max < i128::MAX` keeps every running
/// `i128` partial sum in range. The strict `<` mirrors the chart's saturated comparison.
///
/// # Complexity
///
/// `O(n_in/32)` time, `O(1)` extra space.
fn fast_bound_ok(n_in: usize, top: i128, q: &QuantAct) -> bool {
    let mut s_max: i128 = 0;
    let mut b = 0usize;
    while b < q.scales().len() {
        let a = i128::from(q.scales()[b]).abs();
        if a > s_max {
            s_max = a;
        }
        b += 1;
    }
    if s_max.saturating_mul(32 * top * 127) > i128::from(i64::MAX) {
        return false;
    }
    let bound = (n_in as i128)
        .saturating_mul(F16_MAX_FIXED)
        .saturating_mul(top)
        .saturating_mul(127)
        .saturating_mul(s_max);
    bound < i128::MAX
}

/// One weight row's `i128` accumulators for every live column: `Σ_b S_b·(d_b·Σ_l q_l·xq_{c,l})`.
///
/// The block loop decodes each weight block once ([`q8_0_block_kdots`]/[`q5_0_block_kdots`]) and
/// folds its contribution into every live column in the same `i64`/`i128` op order as
/// [`matvec_q8act`]'s fast row. A non-finite block scale returns [`RowOutcome::NonFinite`], and an
/// unreachable checked overflow returns [`RowOutcome::Bail`].
///
/// # Complexity
///
/// `O(blocks·k)` exact multiply-adds plus the per-block decode; `O(1)` extra space.
fn row_accs(
    family: Family,
    row: &[u8],
    geom: Geom,
    acts: &[QuantAct],
    k: usize,
    state: &[ColumnState],
    avx: bool,
) -> RowOutcome {
    let mut totals = [0i128; MAX_K];
    let mut blk = 0usize;
    while blk < geom.blocks {
        let off = blk * geom.block_bytes;
        let Ok(d) = f16_to_fixed(u16::from_le_bytes([row[off], row[off + 1]])) else {
            return RowOutcome::NonFinite;
        };
        let dots = match family {
            Family::Q8_0 => q8_0_block_kdots(avx, row, off + 2, acts, blk, k),
            Family::Q5_0 => q5_0_block_kdots(avx, row, off, acts, blk, k),
        };
        let mut c = 0usize;
        while c < k {
            if state[c] == ColumnState::Live {
                let scale = acts[c].scales()[blk];
                let Some(scaled) = scale.checked_mul(i64::from(dots[c])) else {
                    return RowOutcome::Bail;
                };
                let Some(term) = i128::from(d).checked_mul(i128::from(scaled)) else {
                    return RowOutcome::Bail;
                };
                match totals[c].checked_add(term) {
                    Some(total) => totals[c] = total,
                    None => return RowOutcome::Bail,
                }
            }
            c += 1;
        }
        blk += 1;
    }
    RowOutcome::Values(totals)
}

/// The `k` exact 32-lane dots of one `Q8_0` weight block: AVX2 when the host has it, else the
/// portable lane loop.
///
/// `weight_off` is the block's first code byte; the caller guarantees 32 code bytes there and 32
/// activation lanes per column.
///
/// # Complexity
///
/// `O(k)` exact 32-lane dots, `O(1)` extra space.
fn q8_0_block_kdots(
    avx: bool,
    row: &[u8],
    weight_off: usize,
    acts: &[QuantAct],
    blk: usize,
    k: usize,
) -> [i32; MAX_K] {
    #[cfg(target_arch = "x86_64")]
    if avx {
        // SAFETY: AVX2 is present, `weight_off + 32 ≤ row.len()` by the caller's geometry and each
        // `acts[c].codes()` holds the block's `blk·32 + 32` lanes by [`fast_family`].
        return unsafe { q8_0_block_kdots_avx2(row.as_ptr().add(weight_off), acts, blk, k) };
    }
    let _ = avx;
    q8_0_block_kdots_portable(&row[weight_off..weight_off + Q8_0_QK], acts, blk, k)
}

/// The `k` exact 32-lane dots of one `Q5_0` weight block: AVX2 when the host has it, else the
/// portable lane loop.
///
/// `block_off` is the block's `d` byte; the caller guarantees [`Q5_0_BLOCK_BYTES`] bytes there and
/// 32 activation lanes per column.
///
/// # Complexity
///
/// `O(k)` exact 32-lane dots, `O(1)` extra space.
fn q5_0_block_kdots(
    avx: bool,
    row: &[u8],
    block_off: usize,
    acts: &[QuantAct],
    blk: usize,
    k: usize,
) -> [i32; MAX_K] {
    #[cfg(target_arch = "x86_64")]
    if avx {
        // SAFETY: AVX2 is present, `block_off + Q5_0_BLOCK_BYTES ≤ row.len()` by the caller's
        // geometry and each `acts[c].codes()` holds the block's `blk·32 + 32` lanes by
        // [`fast_family`].
        return unsafe { q5_0_block_kdots_avx2(row.as_ptr().add(block_off), acts, blk, k) };
    }
    let _ = avx;
    q5_0_block_kdots_portable(&row[block_off..block_off + Q5_0_BLOCK_BYTES], acts, blk, k)
}

/// Exact `Σ_{l<32} (bytes[l] as i8)·xq[l]` for a `Q8_0` code window: the portable twin of the
/// AVX2 core and the no-AVX2 path.
///
/// # Complexity
///
/// `O(32)` time, `O(1)` extra space.
fn portable_u8_i8_dot(codes: &[u8], xq: &[i8]) -> i32 {
    let mut acc: i32 = 0;
    let mut l = 0usize;
    while l < codes.len() {
        acc += i32::from(i8::from_ne_bytes([codes[l]])) * i32::from(xq[l]);
        l += 1;
    }
    acc
}

/// Exact `Σ_{l<32} c_l·xq[l]` for one `Q5_0` block from its `qh`/`qs` bytes: the portable twin of
/// the AVX2 core. The lane order is `quant::q5_0_codes`'s: `qs` low nibbles are lanes `0..16`,
/// high nibbles lanes `16..32`, and `qh` bit `l` is lane `l`'s fifth bit.
///
/// # Complexity
///
/// `O(32)` time, `O(1)` extra space.
fn portable_q5_0_dot(qh: &[u8], qs: &[u8], xq: &[i8]) -> i32 {
    let qh_u32 = u32::from_le_bytes([qh[0], qh[1], qh[2], qh[3]]);
    let mut acc: i32 = 0;
    let mut j = 0usize;
    while j < 16 {
        let nib = qs[j];
        let lo_bit = ((qh_u32 >> j) & 1) as u8;
        let hi_bit = ((qh_u32 >> (16 + j)) & 1) as u8;
        let lo = i32::from((nib & 0x0F) | (lo_bit << 4)) - 16;
        let hi = i32::from((nib >> 4) | (hi_bit << 4)) - 16;
        acc += lo * i32::from(xq[j]);
        acc += hi * i32::from(xq[16 + j]);
        j += 1;
    }
    acc
}

/// The portable [`q8_0_block_kdots`]: one 32-byte window dotted against each column's 32 lanes.
///
/// # Complexity
///
/// `O(k)` exact 32-lane dots, `O(1)` extra space.
fn q8_0_block_kdots_portable(
    window: &[u8],
    acts: &[QuantAct],
    blk: usize,
    k: usize,
) -> [i32; MAX_K] {
    let xbase = blk * ACT_QK;
    let mut out = [0i32; MAX_K];
    let mut c = 0usize;
    while c < k {
        out[c] = portable_u8_i8_dot(window, &acts[c].codes()[xbase..xbase + ACT_QK]);
        c += 1;
    }
    out
}

/// The portable [`q5_0_block_kdots`]: one block decoded and dotted against each column's 32 lanes.
///
/// # Complexity
///
/// `O(k)` exact 32-lane dots, `O(1)` extra space.
fn q5_0_block_kdots_portable(
    block: &[u8],
    acts: &[QuantAct],
    blk: usize,
    k: usize,
) -> [i32; MAX_K] {
    let qh = &block[2..6];
    let qs = &block[6..6 + Q5_0_QK / 2];
    let xbase = blk * ACT_QK;
    let mut out = [0i32; MAX_K];
    let mut c = 0usize;
    while c < k {
        out[c] = portable_q5_0_dot(qh, qs, &acts[c].codes()[xbase..xbase + ACT_QK]);
        c += 1;
    }
    out
}

/// Whether the executing host exposes AVX2.
#[inline]
#[must_use]
fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Horizontal sum of the four `i32` lanes of `v`.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum128(v: __m128i) -> i32 {
    // SAFETY: register arithmetic only; no memory and no lanes outside `v`.
    unsafe {
        let s = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xB1));
        let t = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        _mm_cvtsi128_si32(t)
    }
}

/// Horizontal sum of the eight `i32` lanes of `v`.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum256(v: __m256i) -> i32 {
    // SAFETY: register arithmetic only; the lane extraction reads no memory.
    unsafe {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        hsum128(_mm_add_epi32(lo, hi))
    }
}

/// AVX2 [`q8_0_block_kdots`]: the weight vector is loaded and its magnitudes formed once, then
/// each column's activation vector is sign-folded into it with one `vpmaddubsw` pair sum.
///
/// # Safety
///
/// `weight` must point at 32 readable code bytes; each `acts[c].codes()` must hold at least
/// `blk·32 + 32` lanes; the caller must have AVX2.
///
/// # Complexity
///
/// `O(k)` exact 32-lane dots, `O(1)` extra space.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q8_0_block_kdots_avx2(
    weight: *const u8,
    acts: &[QuantAct],
    blk: usize,
    k: usize,
) -> [i32; MAX_K] {
    // SAFETY: the caller guarantees the 32-byte load and AVX2.
    unsafe {
        let w = _mm256_loadu_si256(weight.cast());
        let mag = _mm256_abs_epi8(w);
        let ones = _mm256_set1_epi16(1);
        let xbase = blk * ACT_QK;
        let mut out = [0i32; MAX_K];
        let mut c = 0usize;
        while c < k {
            let x = _mm256_loadu_si256(acts[c].codes().as_ptr().add(xbase).cast());
            let signed = _mm256_sign_epi8(x, w);
            let pair = _mm256_maddubs_epi16(mag, signed);
            out[c] = hsum256(_mm256_madd_epi16(pair, ones));
            c += 1;
        }
        out
    }
}

/// AVX2 [`q5_0_block_kdots`]: the fifth-bit decode (`nibble + 16·bit`, the lane bias removed with
/// `16·Σ xq`) forms the unsigned code vector once, then each column is one `vpmaddubsw` pair sum.
///
/// # Safety
///
/// `block` must point at [`Q5_0_BLOCK_BYTES`] readable bytes; each `acts[c].codes()` must hold at
/// least `blk·32 + 32` lanes; the caller must have AVX2.
///
/// # Complexity
///
/// `O(k)` exact 32-lane dots, `O(1)` extra space.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q5_0_block_kdots_avx2(
    block: *const u8,
    acts: &[QuantAct],
    blk: usize,
    k: usize,
) -> [i32; MAX_K] {
    // SAFETY: the caller guarantees the block's 22 readable bytes and AVX2.
    unsafe {
        let qh = block.add(2);
        let qs = block.add(6);
        let bytes = _mm_loadu_si128(qs.cast());
        let nib = _mm_set1_epi8(0x0F);
        let lo = _mm_and_si128(bytes, nib);
        let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), nib);
        let codes = _mm256_set_m128i(hi, lo); // lanes 0..16 low nibbles, 16..32 high nibbles

        // The fifth-bit plane: byte `l` of `hbit` is bit `l` of the little-endian `qh` u32.
        let lo16 = _mm256_set1_epi16(i16::from_le_bytes([*qh, *qh.add(1)]));
        let hi16 = _mm256_set1_epi16(i16::from_le_bytes([*qh.add(2), *qh.add(3)]));
        let pair = _mm256_set_m128i(_mm256_castsi256_si128(hi16), _mm256_castsi256_si128(lo16));
        let spread = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 1,
        );
        let spread_bytes = _mm256_shuffle_epi8(pair, spread);
        let bitmask = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64,
            -128, 1, 2, 4, 8, 16, 32, 64, -128,
        );
        let masked = _mm256_and_si256(spread_bytes, bitmask);
        let hbit = _mm256_andnot_si256(
            _mm256_cmpeq_epi8(masked, _mm256_setzero_si256()),
            _mm256_set1_epi8(1),
        );
        let bit16 = _mm256_slli_epi16(hbit, 4); // 0 or 16 per byte, no carry across a byte
        let unsigned = _mm256_add_epi8(codes, bit16); // 0..31

        let ones = _mm256_set1_epi16(1);
        let xbase = blk * ACT_QK;
        let mut out = [0i32; MAX_K];
        let mut c = 0usize;
        while c < k {
            let x = _mm256_loadu_si256(acts[c].codes().as_ptr().add(xbase).cast());
            let prod = _mm256_maddubs_epi16(unsigned, x);
            let xsum = _mm256_maddubs_epi16(_mm256_set1_epi8(1), x);
            let comb = _mm256_sub_epi16(prod, _mm256_slli_epi16(xsum, 4));
            out[c] = hsum256(_mm256_madd_epi16(comb, ones));
            c += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        clippy::cast_lossless,
        clippy::many_single_char_names,
        reason = "test-only fixture arithmetic on bounded small values; not extracted"
    )]

    use super::*;
    use crate::engine::chart::quant_act;
    use crate::engine::matvec_fixed;
    use crate::format::quant::{
        block_geom, GGML_TYPE_F32, GGML_TYPE_PQ2_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,
    };
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    /// xorshift64: a deterministic code/activation stream for the equality tests.
    fn next_rand(state: &mut u64) -> u64 {
        let mut s = *state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *state = s;
        s
    }

    /// A `rows × n_in` payload of random bytes with every block's `f16` scale(s) overwritten by
    /// `d` in every row, so each block is well formed while its codes and packed scales are random.
    fn random_payload(ggml_type: u32, rows: usize, n_in: usize, d: u16, seed: u64) -> Vec<u8> {
        let (qk, bb) = block_geom(ggml_type).expect("supported geometry");
        let blocks = n_in.div_ceil(qk);
        let row_bytes = blocks * bb;
        let mut raw = vec![0u8; rows * row_bytes];
        let mut state = seed;
        let mut i = 0usize;
        while i < raw.len() {
            raw[i] = (next_rand(&mut state) & 0xff) as u8;
            i += 1;
        }
        let mut r = 0usize;
        while r < rows {
            let mut block = 0usize;
            while block < blocks {
                let o = r * row_bytes + block * bb;
                match ggml_type {
                    GGML_TYPE_Q8_0 | GGML_TYPE_Q5_0 | GGML_TYPE_PQ2_0 => {
                        raw[o..o + 2].copy_from_slice(&d.to_le_bytes());
                    }
                    GGML_TYPE_Q4_K => {
                        raw[o..o + 2].copy_from_slice(&d.to_le_bytes());
                        raw[o + 2..o + 4].copy_from_slice(&d.to_le_bytes());
                    }
                    GGML_TYPE_Q6_K => {
                        raw[o + bb - 2..o + bb].copy_from_slice(&d.to_le_bytes());
                    }
                    _ => {}
                }
                block += 1;
            }
            r += 1;
        }
        raw
    }

    /// `n` small Q32.32 activations in `[-2048, 2047]`, from a deterministic stream.
    fn random_x(n: usize, seed: u64) -> Vec<i64> {
        let mut state = seed;
        let mut x: Vec<i64> = Vec::with_capacity(n);
        while x.len() < n {
            x.push((next_rand(&mut state) % 4096) as i64 - 2048);
        }
        x
    }

    /// `k` distinct small activation quantizations of length `n_in`.
    fn acts(n_in: usize, k: usize, seed: u64) -> Vec<QuantAct> {
        let mut out: Vec<QuantAct> = Vec::with_capacity(k);
        let mut c = 0usize;
        while c < k {
            let s = seed
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(c as u64 + 1);
            out.push(quant_act(&random_x(n_in, s)).expect("small activations dequantize"));
            c += 1;
        }
        out
    }

    /// `matmat_q8act` on `qs`: `Ok` and column-equal to the per-column [`matvec_q8act`] when every
    /// column is `Ok`, else the lowest-indexed column's refusal.
    fn check_columns(ggml_type: u32, raw: &[u8], n_out: usize, n_in: usize, qs: &[QuantAct]) {
        let expected: Vec<Result<Vec<i64>, MatvecError>> = qs
            .iter()
            .map(|q| matvec_q8act(ggml_type, raw, n_out, n_in, q))
            .collect();
        match matmat_q8act(ggml_type, raw, n_out, n_in, qs) {
            Ok(cols) => {
                assert_eq!(cols.len(), qs.len(), "column count");
                for (c, e) in expected.iter().enumerate() {
                    assert_eq!(
                        &cols[c],
                        e.as_ref().expect("every column must be Ok"),
                        "type {ggml_type} shape ({n_out},{n_in}) column {c} diverged"
                    );
                }
            }
            Err(e) => {
                let first = expected.iter().find_map(|r| r.as_ref().err());
                assert_eq!(
                    Some(&e),
                    first,
                    "type {ggml_type} shape ({n_out},{n_in}) refusal must be the first column's"
                );
            }
        }
    }

    /// `Q8_0`/`Q5_0` (shared-decode kernel) and `Q4_K`/`Q6_K`/`PQ2_0` (per-column fallback) over
    /// several shapes, including ragged widths, and `k ∈ {1, 2, 3, 8, 16}`: every column equals
    /// the per-column [`matvec_q8act`] bit for bit.
    #[test]
    fn columns_match_per_column_matvec_q8act() {
        const TYPES: [u32; 5] = [
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q5_0,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q6_K,
            GGML_TYPE_PQ2_0,
        ];
        const SHAPES: [(usize, usize); 6] =
            [(4, 64), (3, 96), (2, 33), (2, 300), (1, 128), (2, 256)];
        const KS: [usize; 5] = [1, 2, 3, 8, 16];
        let mut seed = 0x1234_5678_9abc_def0u64;
        for (ti, &ggml_type) in TYPES.iter().enumerate() {
            for &(rows, n_in) in &SHAPES {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let raw = random_payload(ggml_type, rows, n_in, 0x3800, seed);
                for &k in &KS {
                    let qs = acts(n_in, k, seed ^ ((k as u64) << 17) ^ ti as u64);
                    check_columns(ggml_type, &raw, rows, n_in, &qs);
                }
            }
        }
    }

    /// Extremes: maximal finite weight scales against `±127` codes at unit scale (the kernel's
    /// fast path), and activations that exhaust `i64` (the gate falls back to the per-column
    /// reference). Equality holds in both.
    #[test]
    fn columns_match_on_extreme_codes_and_scales() {
        const TYPES: [u32; 2] = [GGML_TYPE_Q8_0, GGML_TYPE_Q5_0];
        let n_in = 256usize;
        for (ti, &ggml_type) in TYPES.iter().enumerate() {
            let raw = random_payload(ggml_type, 2, n_in, 0x7BFF, 0x51ff_0000 + ti as u64);
            let x: Vec<i64> = (0..n_in)
                .map(|j| if j % 2 == 0 { 127 } else { -127 })
                .collect();
            let extreme = quant_act(&x).expect("±127 dequantizes");
            assert!(
                extreme.scales().iter().all(|&s| s == 1),
                "±127 must quantize at unit scale"
            );
            check_columns(ggml_type, &raw, 2, n_in, &[extreme]);

            let huge = quant_act(&vec![127i64 << 56; n_in])
                .expect("127·2^56 dequantizes on the power-of-two grid");
            check_columns(ggml_type, &raw, 2, n_in, &[huge]);
        }
    }

    /// The fast path's columns also equal `matvec_fixed` on the dequantized activation, the
    /// definition `matvec_q8act` itself refines.
    #[test]
    fn columns_match_matvec_fixed_on_dequantized_activations() {
        const TYPES: [u32; 2] = [GGML_TYPE_Q8_0, GGML_TYPE_Q5_0];
        for (ti, &ggml_type) in TYPES.iter().enumerate() {
            let (rows, n_in) = (3usize, 96usize);
            let raw = random_payload(ggml_type, rows, n_in, 0x3800, 0xbeef_0000 + ti as u64);
            let qs = acts(n_in, 4, 0x0dd0 + ti as u64);
            let cols = matmat_q8act(ggml_type, &raw, rows, n_in, &qs).expect("fast payload");
            for (c, q) in qs.iter().enumerate() {
                let xhat = q.dequant().expect("small activations dequantize");
                assert_eq!(
                    cols[c],
                    matvec_fixed(ggml_type, &raw, rows, n_in, &xhat).expect("reference"),
                    "type {ggml_type} column {c} vs matvec_fixed"
                );
            }
        }
    }

    /// Every refusal class agrees with the per-column reference, including the refusal value.
    #[test]
    fn refusals_match_per_column() {
        let q8 = GGML_TYPE_Q8_0;
        let raw = random_payload(q8, 2, 32, 0x3800, 1);
        let qs = acts(32, 1, 7);

        // Shape mismatch: the dequantized activation length differs from `n_in`.
        check_columns(q8, &raw, 2, 31, &qs);

        // Short payload.
        check_columns(q8, &raw[..raw.len() - 1], 2, 32, &qs);

        // Unsupported family.
        check_columns(GGML_TYPE_F32, &[], 1, 1, &qs);

        // Row-stride / payload length overflow.
        check_columns(q8, &raw, usize::MAX, 32, &qs);

        // Non-finite weight scale.
        let mut inf = random_payload(q8, 1, 32, 0x3800, 2);
        inf[0..2].copy_from_slice(&0x7C00u16.to_le_bytes());
        check_columns(q8, &inf, 1, 32, &qs);

        // Output overflow: weight 1.0 against `127·2^56` rounds past `i64`.
        let mut one = random_payload(q8, 1, 32, 0x3C00, 3);
        let mut l = 0usize;
        while l < 32 {
            one[2 + l] = 1;
            l += 1;
        }
        let huge = quant_act(&vec![127i64 << 56; 32])
            .expect("127·2^56 dequantizes on the power-of-two grid");
        check_columns(q8, &one, 1, 32, &[huge]);
    }

    /// The AVX2 block cores return the same exact `i32` dots as their portable twins on random
    /// weight bytes and random activations, for `k ∈ {1, 2, 3, 8, 16}`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_block_kdots_match_the_portable_twin() {
        const KS: [usize; 5] = [1, 2, 3, 8, 16];
        if !avx2_available() {
            return;
        }
        let n_in = 64usize;
        let mut seed = 0x243f_6a88_85a3_08d3u64;
        for &ggml_type in &[GGML_TYPE_Q8_0, GGML_TYPE_Q5_0] {
            let (qk, bb) = block_geom(ggml_type).expect("supported geometry");
            let blocks = n_in / qk;
            let raw = random_payload(ggml_type, 1, n_in, 0x3800, seed);
            for &k in &KS {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let qs = acts(n_in, k, seed);
                let mut blk = 0usize;
                while blk < blocks {
                    let off = blk * bb;
                    if ggml_type == GGML_TYPE_Q8_0 {
                        let port = q8_0_block_kdots_portable(&raw[off + 2..off + bb], &qs, blk, k);
                        // SAFETY: AVX2 is present, `raw[off + 2 .. off + 34]` holds 32 code bytes
                        // and each column's codes hold `blk·32 + 32` lanes.
                        let avx = unsafe {
                            q8_0_block_kdots_avx2(raw.as_ptr().add(off + 2), &qs, blk, k)
                        };
                        assert_eq!(&port[..k], &avx[..k], "Q8_0 block {blk} k {k} diverged");
                    } else {
                        let port = q5_0_block_kdots_portable(&raw[off..off + bb], &qs, blk, k);
                        // SAFETY: AVX2 is present, `raw[off .. off + 22]` holds the block and each
                        // column's codes hold `blk·32 + 32` lanes.
                        let avx =
                            unsafe { q5_0_block_kdots_avx2(raw.as_ptr().add(off), &qs, blk, k) };
                        assert_eq!(&port[..k], &avx[..k], "Q5_0 block {blk} k {k} diverged");
                    }
                    blk += 1;
                }
            }
        }
    }

    /// Timing probe (L1 deliverable 4): `time(k)/time(1)` for the two fast families at the
    /// 4864×896 projection shape. Run with
    /// `cargo test --release --all-features --locked kern -- --ignored --nocapture`.
    ///
    /// The epic's falsifier is `time(8) > 2·time(1)`. `KS` are sampled round-robin so that a
    /// transient load spike lands on every `k` alike, and each `k`'s estimator is its minimum
    /// over the rounds (the least-contended window). This is still a shared-host wall clock, not
    /// an idle-host measurement: the ratio is indicative, and a `measured` claim needs a quiesced
    /// host (`docs/epics/decode-dominance.md` §Invariants 3).
    /// k-column cost when the weights stream from DRAM (the decode regime): 256 MB of `Q8_0`
    /// matrices of the Qwen3.5 FFN shape, cycled so no call finds its matrix in cache.
    /// `cargo test --release --all-features --locked kern::matmat -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing measurement, not a correctness gate"]
    #[allow(clippy::cast_precision_loss, reason = "byte counts fit f64 exactly")]
    fn timing_k_scaling_dram() {
        let (n_out, n_in) = (3584usize, 1024usize);
        let row_bytes = n_in / 32 * Q8_0_BLOCK_BYTES;
        let copies = 256_000_000 / (n_out * row_bytes);
        let mats: Vec<Vec<u8>> = (0..copies)
            .map(|c| random_payload(GGML_TYPE_Q8_0, n_out, n_in, 0x2000, 0x5eed + c as u64))
            .collect();
        let pool = acts(n_in, 8, 0xabcd);
        let mut base = 0.0f64;
        for k in [1usize, 2, 3, 4, 6, 8] {
            let _ = matmat_q8act(GGML_TYPE_Q8_0, &mats[0], n_out, n_in, &pool[..k]).expect("warm");
            let reps = 2 * copies;
            let start = Instant::now();
            let mut i = 0usize;
            while i < reps {
                let out = matmat_q8act(GGML_TYPE_Q8_0, &mats[i % copies], n_out, n_in, &pool[..k])
                    .expect("timing");
                black_box(&out);
                i += 1;
            }
            let us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
            if k == 1 {
                base = us;
            }
            let gbs = (n_out * row_bytes) as f64 / us / 1e3;
            println!(
                "dram k={k}: {us:.1} µs/call, {:.2}× k=1, weights {gbs:.1} GB/s",
                us / base
            );
        }
    }

    #[test]
    #[ignore = "timing measurement, not a correctness gate"]
    fn timing_k_scaling() {
        const KS: [usize; 5] = [1, 2, 4, 8, 16];
        const ROUNDS: usize = 15;
        let n_out = 4864usize;
        let n_in = 896usize;
        println!("matmat_q8act timing: n_out={n_out} n_in={n_in} rounds={ROUNDS} (min per k)");
        println!("{:<6} {:>4} {:>14} {:>8}", "type", "k", "best_ms", "ratio");
        for &ggml_type in &[GGML_TYPE_Q8_0, GGML_TYPE_Q5_0] {
            let raw = random_payload(ggml_type, n_out, n_in, 0x3C00, 0x5eed);
            let pool = acts(n_in, 16, 0xabcd);
            let _ = matmat_q8act(ggml_type, &raw, n_out, n_in, &pool[..1]).expect("warmup");
            let mut best = [Duration::MAX; KS.len()];
            let mut round = 0usize;
            while round < ROUNDS {
                let mut i = 0usize;
                while i < KS.len() {
                    let start = Instant::now();
                    let out =
                        matmat_q8act(ggml_type, &raw, n_out, n_in, &pool[..KS[i]]).expect("timing");
                    let elapsed = start.elapsed();
                    black_box(&out);
                    if elapsed < best[i] {
                        best[i] = elapsed;
                    }
                    i += 1;
                }
                round += 1;
            }
            let base = best[0].as_secs_f64() * 1e3;
            let mut i = 0usize;
            while i < KS.len() {
                let ms = best[i].as_secs_f64() * 1e3;
                println!("{ggml_type:<6} {:>4} {ms:>14.3} {:>8.2}", KS[i], ms / base);
                i += 1;
            }
        }
    }
}
