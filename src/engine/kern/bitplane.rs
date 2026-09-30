//! Two-stage bit-plane argmax over a `Q8_0` head, licensed by `proof/corpus/Numeric/BitplaneArgmax.lean`
//! (`nibble_split`, `lo_interval_sound`, `bitplane_prune_sound`, `bitplane_argmax_exact`).
//!
//! # The integers
//!
//! The executor's `Q8Act` logit of head row `r` is the exact `i128` sum
//! `acc(r) = Σ_b c_b·dot_b`, with `c_b = d_b·s_b` and `dot_b = Σ_i q_i·a_i`, rounded once:
//! `logit(r) = round_shift_32(acc(r))`. Here `d_b` is the block's `f16` weight scale in Q32.32,
//! `s_b` the activation block scale, `q_i` the int8 weight code and `a_i` the int8 activation code.
//! The chart row kernel computes exactly this sum, and `matvec_q8act` is proved equal to
//! `matvec_fixed ∘ dequant ∘ quant_act`.
//!
//! Split every code as `q = 16·h + l` with `h = q >> 4 ∈ [−8, 7]` and `l = q & 15 ∈ [0, 15]`
//! (`nibble_split`, `nibble_hi_range`). Then `dot_b = 16·H_b + L_b`, where `H_b = Σ h_i·a_i` and
//! `L_b = Σ l_i·a_i`. Since `l_i ∈ [0, 15]`, `L_b ∈ [neg_b, pos_b]` with `neg_b = 15·Σ min(0, a_i)`
//! and `pos_b = 15·Σ max(0, a_i)`. So `hi(r) = Σ_b c_b·(16·H_b + (c_b ≥ 0 ? pos_b : neg_b))`
//! bounds `acc(r)` from above (`lo_interval_sound`), using only the high plane.
//!
//! # The two stages
//!
//! 1. Stream the high plane of every row, forming `hi(r)` with one `i128` multiply per block,
//!    `d_b·(s_b·(16·H_b + slack_b))`; since `s_b ≥ 0` the sign of `c_b` is the sign of `d_b`, which
//!    picks `pos_b` or `neg_b`. The seed `k` is the row with the largest `hi` (lowest index).
//! 2. Evaluate `acc(k)` exactly. Evaluate exactly every row with
//!    `round(hi(r)) ≥ round(acc(k))`, and return the lowest-index row with the largest rounded
//!    logit among them.
//!
//! **Why rounded comparisons.** `argmax_fixed` compares rounded logits and the lowest index wins
//! ties. A pruned row has `round(acc(r)) ≤ round(hi(r)) < round(acc(k))`, because rounding is
//! monotone, so it can neither exceed nor tie the seed. Every row that attains the maximum
//! rounded logit therefore survives, and the survivors' lowest-index maximiser is `argmax_fixed`
//! over all rows (`bitplane_argmax_exact`, with `dotVal = round ∘ acc`).
//!
//! # Layout
//!
//! Per row, per 32-code block:
//! - the `f16` scale;
//! - 16 bytes of high nibbles (byte `j` holds `h_j` low, `h_{j+16}` high);
//! - 16 bytes of low nibbles in the same order.
//!
//! Stage 1 reads `2 + 16` of the `34` bytes `Q8_0` stores per block (53%). Stage 2 reads the low
//! plane of the survivors only.

#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    reason = "nibble packing reinterprets code bytes between i8 and u8 on purpose; r, q, b, c, h, \
              l, x are the kernel's standard symbols"
)]

use std::sync::{Arc, Mutex, PoisonError};

use crate::engine::chart::pool::Pool;
use crate::engine::chart::QuantAct;
use crate::engine::f16_to_fixed;
use crate::engine::value::round_shift_32;
use crate::format::quant::{GGML_TYPE_Q8_0, Q8_0_BLOCK_BYTES, Q8_0_QK};

/// Nibble bytes per block and plane.
const HALF: usize = Q8_0_QK / 2;

/// Stage 1 bytes per block: 2-byte little-endian f16 scale + 16-byte high plane.
const STAGE1_BLOCK_BYTES: usize = 2 + HALF;

/// Stage 2 bytes per block: 16-byte low plane.
const STAGE2_BLOCK_BYTES: usize = HALF;

/// One chunk's reduction of the two pooled scans in [`BitplaneHead::argmax`]: the stage-1 seed
/// (`hi`, `row`) and the stage-2 survivors (`has`, `best`, `best_row`, `count`). Each chunk's
/// single task writes only its own slot; the calling thread merges the slots in ascending chunk
/// order, which reproduces the serial scans' lowest-index-first tie-break.
#[derive(Default)]
struct ChunkReduce {
    /// Stage 1: the largest high part in the chunk.
    hi: i128,
    /// Stage 1: the lowest row attaining `hi`.
    row: usize,
    /// Stage 1: the largest low part in the chunk.
    lo: i128,
    /// Stage 1: the lowest row attaining `lo`.
    lo_row: usize,
    /// Stage 2: whether the chunk holds a row with `bounds[r] >= tau`.
    has: bool,
    /// Stage 2: the best rounded logit in the chunk.
    best: i64,
    /// Stage 2: the lowest row attaining `best`.
    best_row: usize,
    /// Stage 2: the number of rows in the chunk with `bounds[r] >= tau`.
    count: usize,
}

/// Lock `m`, ignoring poisoning: the pool re-raises a task panic only after every participant has
/// stopped, so a poisoned slot is never read mid-flight.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A `Q8_0` head split into plane-major layout, built once at load.
///
/// Stage 1 streams `stage1` (per row, per block: 2-byte f16 scale followed by 16-byte high plane),
/// consuming exactly 18 of 34 bytes per block (~53% of cache lines).
/// Stage 2 streams `stage2` (per row, per block: 16-byte low plane) for surviving rows only (≤ 2%).
#[derive(Debug)]
pub(crate) struct BitplaneHead {
    pub(crate) rows: usize,
    pub(crate) blocks: usize,
    /// `rows · blocks · 18` bytes: contiguous plane-major layout for stage 1.
    pub(crate) stage1: Vec<u8>,
    /// `rows · blocks · 16` bytes: low-plane bytes for stage 2.
    pub(crate) stage2: Vec<u8>,
    /// Reusable bounds buffer holding `round(hi(r))` to avoid per-step heap allocation.
    bounds_buf: Mutex<Vec<i64>>,
}

impl BitplaneHead {
    /// Split a row-major `Q8_0` payload of `rows × n_in` into plane-major layout; `None` for another
    /// type, a width that is not whole blocks, a short payload or a non-finite scale (the caller
    /// keeps the dense head, which carries the exact refusal).
    ///
    /// # Complexity
    ///
    /// `O(rows·n_in)` byte operations once; `18·rows·blocks` bytes of stage 1 plus `16·rows·blocks`
    /// bytes of stage 2.
    pub(crate) fn from_q8_0(
        ggml_type: u32,
        raw: &[u8],
        rows: usize,
        n_in: usize,
    ) -> Option<Arc<Self>> {
        if ggml_type != GGML_TYPE_Q8_0 || n_in == 0 || !n_in.is_multiple_of(Q8_0_QK) {
            return None;
        }
        let blocks = n_in / Q8_0_QK;
        let n_blocks = rows.checked_mul(blocks)?;
        if raw.len() < n_blocks.checked_mul(Q8_0_BLOCK_BYTES)? {
            return None;
        }
        let mut stage1 = vec![0u8; n_blocks * STAGE1_BLOCK_BYTES];
        let mut stage2 = vec![0u8; n_blocks * STAGE2_BLOCK_BYTES];
        let mut b = 0usize;
        while b < n_blocks {
            let off = b * Q8_0_BLOCK_BYTES;
            let scale_u16 = u16::from_le_bytes([raw[off], raw[off + 1]]);
            // Validate finite scale once at construction:
            f16_to_fixed(scale_u16).ok()?;
            let s1_off = b * STAGE1_BLOCK_BYTES;
            stage1[s1_off] = raw[off];
            stage1[s1_off + 1] = raw[off + 1];
            let codes = &raw[off + 2..off + 2 + Q8_0_QK];
            let s2_off = b * STAGE2_BLOCK_BYTES;
            let mut j = 0usize;
            while j < HALF {
                let (q0, q1) = (codes[j] as i8, codes[j + HALF] as i8);
                let (h0, h1) = ((q0 >> 4) as u8 & 0x0F, (q1 >> 4) as u8 & 0x0F);
                let (l0, l1) = (q0 as u8 & 0x0F, q1 as u8 & 0x0F);
                stage1[s1_off + 2 + j] = h0 | (h1 << 4);
                stage2[s2_off + j] = l0 | (l1 << 4);
                j += 1;
            }
            b += 1;
        }
        Some(Arc::new(BitplaneHead {
            rows,
            blocks,
            stage1,
            stage2,
            bounds_buf: Mutex::new(Vec::new()),
        }))
    }

    /// High-plane byte slice for row `r`, block `blk`.
    #[cfg(test)]
    pub(crate) fn high_plane_block(&self, r: usize, blk: usize) -> &[u8] {
        let off = (r * self.blocks + blk) * STAGE1_BLOCK_BYTES + 2;
        &self.stage1[off..off + HALF]
    }

    /// Low-plane byte slice for row `r`, block `blk`.
    #[cfg(test)]
    pub(crate) fn low_plane_block(&self, r: usize, blk: usize) -> &[u8] {
        let off = (r * self.blocks + blk) * STAGE2_BLOCK_BYTES;
        &self.stage2[off..off + HALF]
    }

    /// Number of bytes in the stage 1 array.
    #[allow(dead_code, reason = "diagnostic inspection")]
    pub(crate) fn stage1_bytes(&self) -> usize {
        self.stage1.len()
    }

    /// Number of bytes in the stage 2 array.
    #[allow(dead_code, reason = "diagnostic inspection")]
    pub(crate) fn stage2_bytes(&self) -> usize {
        self.stage2.len()
    }

    /// Reconstruct row `r` as dense `Q8_0` bytes (34 bytes per block), exactly matching
    /// the original input row.
    #[allow(dead_code, reason = "round trip verification")]
    pub(crate) fn reconstruct_row(&self, r: usize, dst: &mut [u8]) {
        assert_eq!(dst.len(), self.blocks * Q8_0_BLOCK_BYTES);
        let s1_row = r * self.blocks * STAGE1_BLOCK_BYTES;
        let s2_row = r * self.blocks * STAGE2_BLOCK_BYTES;
        let mut blk = 0usize;
        while blk < self.blocks {
            let s1_off = s1_row + blk * STAGE1_BLOCK_BYTES;
            let s2_off = s2_row + blk * STAGE2_BLOCK_BYTES;
            let dst_off = blk * Q8_0_BLOCK_BYTES;
            dst[dst_off] = self.stage1[s1_off];
            dst[dst_off + 1] = self.stage1[s1_off + 1];
            let mut j = 0usize;
            while j < HALF {
                let hi_byte = self.stage1[s1_off + 2 + j];
                let lo_byte = self.stage2[s2_off + j];
                let h0 = hi_byte & 0x0F;
                let h1 = hi_byte >> 4;
                let l0 = lo_byte & 0x0F;
                let l1 = lo_byte >> 4;
                dst[dst_off + 2 + j] = (h0 << 4) | l0;
                dst[dst_off + 2 + HALF + j] = (h1 << 4) | l1;
                j += 1;
            }
            blk += 1;
        }
    }

    /// Reconstruct the entire dense `Q8_0` payload from the stage 1 and stage 2 plane-major arrays.
    #[allow(dead_code, reason = "round trip verification")]
    pub(crate) fn to_dense_q8_0(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.rows * self.blocks * Q8_0_BLOCK_BYTES];
        let row_bytes = self.blocks * Q8_0_BLOCK_BYTES;
        for r in 0..self.rows {
            self.reconstruct_row(r, &mut out[r * row_bytes..(r + 1) * row_bytes]);
        }
        out
    }

    /// `argmax_fixed` over this head's `Q8Act` logits for activation `q`, computed in two stages;
    /// also returns the survivor count. `None` when `q` does not match the head's width, so the
    /// caller runs the dense head.
    ///
    /// The caller must have established the chart's overflow bound for `q` (`narrow`, `top = 128`),
    /// under which every `i128` below stays in range.
    ///
    /// # Complexity
    ///
    /// Stage 1: `O(rows·n_in)` nibble multiply-adds over the pool, reading `18/34` of the payload,
    /// plus the chunk-local seed reduction. Stage 2: `O((1 + survivors)·n_in)`, also over the pool.
    /// `O(rows)` words of per-row bounds and `O(tasks)` chunk slots.
    #[allow(
        clippy::too_many_lines,
        reason = "two-stage parallel argmax with certified interval reduction"
    )]
    pub(crate) fn argmax(&self, q: &QuantAct) -> Option<(u32, usize)> {
        if q.codes().len() != self.blocks * Q8_0_QK || q.scales().len() != self.blocks {
            return None;
        }
        // Stage 1 forms `s_b·(16·H_b + slack_b)` in `i64`: `|16·H_b + slack_b| ≤ 16·8·127·32 +
        // 15·127·32 = 581 152`, so every activation scale must satisfy `s·581 152 ≤ i64::MAX`.
        // Otherwise the dense head (which carries its own bound) runs.
        if q.scales()
            .iter()
            .any(|&sc| i128::from(sc).abs() * 581_152 > i128::from(i64::MAX))
        {
            return None;
        }
        // Per-block activation slack of the unread low plane: 15·Σ max(0, a) and 15·Σ min(0, a).
        let mut pos = Vec::with_capacity(self.blocks);
        let mut neg = Vec::with_capacity(self.blocks);
        for blk in q.codes().chunks_exact(Q8_0_QK) {
            let (mut p, mut n) = (0i64, 0i64);
            for &a in blk {
                if a > 0 {
                    p += i64::from(a);
                } else {
                    n += i64::from(a);
                }
            }
            pos.push(15 * p);
            neg.push(15 * n);
        }
        // Stage 1: per row, rounded upper bound, tracking best upper and lower bounds.
        // Reuses the pre-allocated bounds buffer to avoid per-step heap allocation.
        let mut bounds_guard = lock(&self.bounds_buf);
        if bounds_guard.len() != self.rows {
            bounds_guard.resize(self.rows, 0);
        }
        let bounds: &mut [i64] = &mut bounds_guard;

        let per = self
            .rows
            .div_ceil(crate::engine::par::row_chunk_budget(self.rows))
            .max(1);
        let tasks = self.rows.div_ceil(per).max(1);
        // One slot per chunk, written by that chunk's single task; the calling thread merges the
        // slots in ascending chunk order.
        let reduced: Vec<Mutex<ChunkReduce>> = (0..tasks)
            .map(|_| Mutex::new(ChunkReduce::default()))
            .collect();
        let avx = avx2();
        Pool::global().chunks_mut(bounds, per, &|i, start, dst| {
            let mut hi_best = i128::MIN;
            let mut hi_row = start;
            let mut lo_best = i128::MIN;
            let mut lo_row = start;
            let mut j = 0usize;
            while j < dst.len() {
                let (bound, hi, lo) = self.row_interval(start + j, q, &pos, &neg, avx);
                dst[j] = bound;
                if hi > hi_best {
                    hi_best = hi;
                    hi_row = start + j;
                }
                if lo > lo_best {
                    lo_best = lo;
                    lo_row = start + j;
                }
                j += 1;
            }
            let mut slot = lock(&reduced[i]);
            slot.hi = hi_best;
            slot.row = hi_row;
            slot.lo = lo_best;
            slot.lo_row = lo_row;
        });
        // Seed selection from both bounds: largest upper bound row and largest lower bound row.
        // Merging ascending chunks with strict `>` reproduces lowest-index-first tie-breaking.
        let mut k_hi = 0usize;
        let mut best_hi = i128::MIN;
        let mut k_lo = 0usize;
        let mut best_lo = i128::MIN;
        let mut i = 0usize;
        while i < tasks {
            let slot = lock(&reduced[i]);
            if i == 0 || slot.hi > best_hi {
                best_hi = slot.hi;
                k_hi = slot.row;
            }
            if i == 0 || slot.lo > best_lo {
                best_lo = slot.lo;
                k_lo = slot.lo_row;
            }
            i += 1;
        }
        // Certified pruning threshold: max of exact logits of best upper and lower seeds,
        // and the certified lower bound itself.
        let exact_hi = round_logit(self.row_exact(k_hi, q, avx));
        let exact_lo = if k_lo == k_hi {
            exact_hi
        } else {
            round_logit(self.row_exact(k_lo, q, avx))
        };
        let tau = exact_hi.max(exact_lo).max(round_logit(best_lo));

        // Stage 2: each chunk counts its rows with `bounds[r] >= tau` and keeps its best rounded
        // logit (lowest row); the serial result is the ascending merge of the chunk results.
        Pool::global().chunks_mut(bounds, per, &|i, start, dst| {
            let mut count = 0usize;
            let mut has = false;
            let mut best = 0i64;
            let mut best_row = 0usize;
            let mut j = 0usize;
            while j < dst.len() {
                if dst[j] >= tau {
                    count += 1;
                    let v = round_logit(self.row_exact(start + j, q, avx));
                    if !has || v > best {
                        has = true;
                        best = v;
                        best_row = start + j;
                    }
                }
                j += 1;
            }
            let mut slot = lock(&reduced[i]);
            slot.has = has;
            slot.best = best;
            slot.best_row = best_row;
            slot.count = count;
        });
        let mut best: Option<(i64, usize)> = None;
        let mut survivors = 0usize;
        let mut i = 0usize;
        while i < tasks {
            let slot = lock(&reduced[i]);
            survivors += slot.count;
            if slot.has && best.is_none_or(|(bv, _)| slot.best > bv) {
                best = Some((slot.best, slot.best_row));
            }
            i += 1;
        }
        let (_, row) = best?;
        Some((u32::try_from(row).ok()?, survivors))
    }

    /// Row `r`'s rounded upper bound `round(hi(r))`, upper bound `hi(r)`, and lower bound `lo(r)`.
    fn row_interval(
        &self,
        r: usize,
        q: &QuantAct,
        pos: &[i64],
        neg: &[i64],
        avx: bool,
    ) -> (i64, i128, i128) {
        let row_off = r * self.blocks * STAGE1_BLOCK_BYTES;
        #[cfg(target_arch = "x86_64")]
        if avx {
            // SAFETY: AVX2 is present; row `r` has `self.blocks` blocks of 18 bytes in stage1,
            // and `q`, `pos`, `neg` hold `blocks` entries (checked in `argmax`).
            let (hi, lo) = unsafe {
                x86::row_interval_avx2(
                    self.stage1.as_ptr().add(row_off),
                    self.blocks,
                    q.codes().as_ptr(),
                    q.scales().as_ptr(),
                    pos.as_ptr(),
                    neg.as_ptr(),
                )
            };
            return (round_logit(hi), hi, lo);
        }
        let _ = avx;
        let mut hi: i128 = 0;
        let mut lo: i128 = 0;
        let mut blk = 0usize;
        while blk < self.blocks {
            let at = row_off + blk * STAGE1_BLOCK_BYTES;
            let h = high_dot(&self.stage1[at + 2..at + 2 + HALF], block(q, blk), false);
            let scale_u16 = u16::from_le_bytes([self.stage1[at], self.stage1[at + 1]]);
            let d = decode_f16(scale_u16);
            let x_hi = 16 * i64::from(h) + if d >= 0 { pos[blk] } else { neg[blk] };
            let x_lo = 16 * i64::from(h) + if d >= 0 { neg[blk] } else { pos[blk] };
            let sc = i128::from(d);
            hi += sc * i128::from(q.scales()[blk] * x_hi);
            lo += sc * i128::from(q.scales()[blk] * x_lo);
            blk += 1;
        }
        (round_logit(hi), hi, lo)
    }

    /// Row `r`'s exact `acc(r) = Σ c_b·(16·H_b + L_b)`.
    fn row_exact(&self, r: usize, q: &QuantAct, avx: bool) -> i128 {
        let mut acc: i128 = 0;
        let mut blk = 0usize;
        let s1_row = r * self.blocks * STAGE1_BLOCK_BYTES;
        let s2_row = r * self.blocks * STAGE2_BLOCK_BYTES;
        while blk < self.blocks {
            let s1_off = s1_row + blk * STAGE1_BLOCK_BYTES;
            let s2_off = s2_row + blk * STAGE2_BLOCK_BYTES;
            let x = block(q, blk);
            let h = high_dot(&self.stage1[s1_off + 2..s1_off + 2 + HALF], x, avx);
            let l = low_dot(&self.stage2[s2_off..s2_off + HALF], x, avx);
            let scale_u16 = u16::from_le_bytes([self.stage1[s1_off], self.stage1[s1_off + 1]]);
            let d = decode_f16(scale_u16);
            let c = i128::from(d) * i128::from(q.scales()[blk]);
            acc += c * i128::from(16 * i64::from(h) + i64::from(l));
            blk += 1;
        }
        acc
    }
}

/// Exact decode of a finite `f16` bit pattern to Q32.32 fixed-point.
///
/// Refusal of non-finite patterns (`e == 0x1f`) is handled once at `BitplaneHead::from_q8_0`
/// construction. For all finite patterns this produces identical results to
/// [`crate::engine::f16_to_fixed`].
#[inline]
pub(super) fn decode_f16(h: u16) -> i64 {
    let sign = h >> 15;
    let e = (h >> 10) & 0x1f;
    let m = i64::from(h & 0x3ff);
    let mag = if e == 0 {
        m << 8
    } else {
        (m + 1024) << (u32::from(e) + 7)
    };
    if sign == 1 {
        -mag
    } else {
        mag
    }
}

/// `round_shift_32` narrowed the way the chart narrows a logit; the bound the caller established
/// keeps it in `i64`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the caller's chart overflow bound keeps every rounded row sum inside i64"
)]
fn round_logit(acc: i128) -> i64 {
    round_shift_32(acc) as i64
}

/// Activation block `blk`'s 32 codes.
fn block(q: &QuantAct, blk: usize) -> &[i8] {
    &q.codes()[blk * Q8_0_QK..(blk + 1) * Q8_0_QK]
}

/// `Σ h_i·a_i` over one block: `packed` holds `h_j` in the low nibble and `h_{j+16}` in the high
/// nibble of byte `j`, each a 4-bit two's-complement value.
fn high_dot(packed: &[u8], a: &[i8], avx: bool) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if avx {
        // SAFETY: AVX2 is present; `packed` has 16 bytes and `a` 32 lanes.
        return unsafe { x86::high_dot_avx2(packed.as_ptr(), a.as_ptr()) };
    }
    let _ = avx;
    let mut s = 0i32;
    let mut j = 0usize;
    while j < HALF {
        let lo = i32::from(((packed[j] & 0x0F) ^ 8) as i8 - 8);
        let hi = i32::from(((packed[j] >> 4) ^ 8) as i8 - 8);
        s += lo * i32::from(a[j]) + hi * i32::from(a[j + HALF]);
        j += 1;
    }
    s
}

/// `Σ l_i·a_i` over one block, with unsigned low nibbles `l ∈ [0, 15]` in the same layout.
fn low_dot(packed: &[u8], a: &[i8], avx: bool) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if avx {
        // SAFETY: AVX2 is present; `packed` has 16 bytes and `a` 32 lanes.
        return unsafe { x86::low_dot_avx2(packed.as_ptr(), a.as_ptr()) };
    }
    let _ = avx;
    let mut s = 0i32;
    let mut j = 0usize;
    while j < HALF {
        s += i32::from(packed[j] & 0x0F) * i32::from(a[j])
            + i32::from(packed[j] >> 4) * i32::from(a[j + HALF]);
        j += 1;
    }
    s
}

/// Whether the AVX2 nibble dots may run.
fn avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::STAGE1_BLOCK_BYTES;
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_abs_epi8, _mm256_castsi256_si128, _mm256_extracti128_si256,
        _mm256_hadd_epi32, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
        _mm256_set1_epi16, _mm256_set1_epi8, _mm256_set_m128i, _mm256_sign_epi8, _mm256_sub_epi8,
        _mm256_xor_si256, _mm_add_epi32, _mm_and_si128, _mm_cvtsi128_si32, _mm_loadu_si128,
        _mm_set1_epi8, _mm_shuffle_epi32, _mm_srli_epi16, _mm_storeu_si128,
    };

    /// The 32 nibbles of a packed block as bytes `0..15` (low nibbles) and `16..31` (high).
    ///
    /// # Safety
    ///
    /// AVX2; `p` points at 16 readable bytes.
    #[target_feature(enable = "avx2")]
    unsafe fn unpack(p: *const u8) -> __m256i {
        // SAFETY: 16 readable bytes by the caller.
        let v: __m128i = unsafe { _mm_loadu_si128(p.cast()) };
        let m = _mm_set1_epi8(0x0F);
        _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(v, 4), m), _mm_and_si128(v, m))
    }

    /// Sum of the eight `i32` lanes.
    #[target_feature(enable = "avx2")]
    fn hsum(v: __m256i) -> i32 {
        let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
        _mm_cvtsi128_si32(s)
    }

    /// AVX2 `high_dot`: sign-extend the nibbles (`(w ^ 8) − 8`), then the `|w|`/`sign(a, w)`
    /// `vpmaddubsw` pair sum the chart's `Q8_0` core uses; every lane sum fits `i16`
    /// (`2·8·128 ≤ 2048`).
    ///
    /// # Safety
    ///
    /// AVX2; `packed` has 16 bytes and `a` 32 lanes.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn high_dot_avx2(packed: *const u8, a: *const i8) -> i32 {
        // SAFETY: the extents by the caller.
        let (w, x) = unsafe { (unpack(packed), _mm256_loadu_si256(a.cast())) };
        let e8 = _mm256_set1_epi8(8);
        let ws = _mm256_sub_epi8(_mm256_xor_si256(w, e8), e8);
        let pair = _mm256_maddubs_epi16(_mm256_abs_epi8(ws), _mm256_sign_epi8(x, ws));
        hsum(_mm256_madd_epi16(pair, _mm256_set1_epi16(1)))
    }

    /// One block's eight `i32` lane partials of `Σ h_i·a_i` (no horizontal sum).
    ///
    /// # Safety
    ///
    /// AVX2; `packed` has 16 bytes and `a` 32 lanes.
    #[target_feature(enable = "avx2")]
    unsafe fn high_lanes(packed: *const u8, a: *const i8) -> __m256i {
        // SAFETY: the extents by the caller.
        let (w, x) = unsafe { (unpack(packed), _mm256_loadu_si256(a.cast())) };
        let e8 = _mm256_set1_epi8(8);
        let ws = _mm256_sub_epi8(_mm256_xor_si256(w, e8), e8);
        let pair = _mm256_maddubs_epi16(_mm256_abs_epi8(ws), _mm256_sign_epi8(x, ws));
        _mm256_madd_epi16(pair, _mm256_set1_epi16(1))
    }

    /// A row's `hi(r)` and `lo(r)`, four blocks per horizontal reduction, one `i128` multiply
    /// per bound per block.
    ///
    /// # Safety
    ///
    /// AVX2; `row_ptr` has `18·blocks` bytes, `s`, `pos`, `neg` have `blocks` entries, `a` has
    /// `32·blocks` lanes; the caller's scale bound keeps `s_b·x` in `i64`.
    #[target_feature(enable = "avx2")]
    #[allow(
        clippy::too_many_arguments,
        reason = "the row's plane pointer and the activation's four vectors"
    )]
    pub(super) unsafe fn row_interval_avx2(
        row_ptr: *const u8,
        blocks: usize,
        a: *const i8,
        s: *const i64,
        pos: *const i64,
        neg: *const i64,
    ) -> (i128, i128) {
        let term = |b: usize, h: i32, db: i64| -> (i128, i128) {
            // SAFETY: `b < blocks` by the loops below.
            let (sb, p, n) = unsafe { (*s.add(b), *pos.add(b), *neg.add(b)) };
            let (slack_hi, slack_lo) = if db >= 0 { (p, n) } else { (n, p) };
            let x_hi = 16 * i64::from(h) + slack_hi;
            let x_lo = 16 * i64::from(h) + slack_lo;
            let sc = i128::from(db);
            (sc * i128::from(sb * x_hi), sc * i128::from(sb * x_lo))
        };
        let mut hi: i128 = 0;
        let mut lo: i128 = 0;
        let mut b = 0usize;
        while b + 4 <= blocks {
            // SAFETY: four blocks of plane bytes and lanes are in range.
            let p0 = unsafe { row_ptr.add(b * STAGE1_BLOCK_BYTES) };
            let p1 = unsafe { row_ptr.add((b + 1) * STAGE1_BLOCK_BYTES) };
            let p2 = unsafe { row_ptr.add((b + 2) * STAGE1_BLOCK_BYTES) };
            let p3 = unsafe { row_ptr.add((b + 3) * STAGE1_BLOCK_BYTES) };
            let (v0, v1, v2, v3) = unsafe {
                (
                    high_lanes(p0.add(2), a.add(b * 32)),
                    high_lanes(p1.add(2), a.add((b + 1) * 32)),
                    high_lanes(p2.add(2), a.add((b + 2) * 32)),
                    high_lanes(p3.add(2), a.add((b + 3) * 32)),
                )
            };
            let t = _mm256_hadd_epi32(_mm256_hadd_epi32(v0, v1), _mm256_hadd_epi32(v2, v3));
            let sum = _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
            let mut h = [0i32; 4];
            // SAFETY: `h` is 16 bytes, one unaligned 128-bit store.
            unsafe { _mm_storeu_si128(h.as_mut_ptr().cast(), sum) };

            let sc0 = super::decode_f16(u16::from_le(unsafe { p0.cast::<u16>().read_unaligned() }));
            let sc1 = super::decode_f16(u16::from_le(unsafe { p1.cast::<u16>().read_unaligned() }));
            let sc2 = super::decode_f16(u16::from_le(unsafe { p2.cast::<u16>().read_unaligned() }));
            let sc3 = super::decode_f16(u16::from_le(unsafe { p3.cast::<u16>().read_unaligned() }));

            let (hi0, lo0) = term(b, h[0], sc0);
            let (hi1, lo1) = term(b + 1, h[1], sc1);
            let (hi2, lo2) = term(b + 2, h[2], sc2);
            let (hi3, lo3) = term(b + 3, h[3], sc3);
            hi += hi0 + hi1 + hi2 + hi3;
            lo += lo0 + lo1 + lo2 + lo3;
            b += 4;
        }
        while b < blocks {
            // SAFETY: one block in range.
            let pb = unsafe { row_ptr.add(b * STAGE1_BLOCK_BYTES) };
            let h = hsum(unsafe { high_lanes(pb.add(2), a.add(b * 32)) });
            let sc = super::decode_f16(u16::from_le(unsafe { pb.cast::<u16>().read_unaligned() }));
            let (thi, tlo) = term(b, h, sc);
            hi += thi;
            lo += tlo;
            b += 1;
        }
        (hi, lo)
    }

    /// AVX2 `low_dot`: the unsigned nibbles times the signed codes, one `vpmaddubsw`; every pair
    /// sum fits `i16` (`2·15·128 ≤ 3840`).
    ///
    /// # Safety
    ///
    /// AVX2; `packed` has 16 bytes and `a` 32 lanes.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn low_dot_avx2(packed: *const u8, a: *const i8) -> i32 {
        // SAFETY: the extents by the caller.
        let (w, x) = unsafe { (unpack(packed), _mm256_loadu_si256(a.cast())) };
        let pair = _mm256_maddubs_epi16(w, x);
        hsum(_mm256_madd_epi16(pair, _mm256_set1_epi16(1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::argmax_fixed;
    use crate::engine::chart::{matvec_q8act, q8act_fast_bound_ok, quant_act};

    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// A random `Q8_0` head: random codes; scales `±2^k·1.x` (both signs); every 7th row a copy of
    /// an earlier row, so exact logit ties across indices occur.
    fn head(rows: usize, n_in: usize, seed: u64) -> Vec<u8> {
        let blocks = n_in / 32;
        let mut raw = vec![0u8; rows * blocks * 34];
        let mut st = seed;
        for r in 0..rows {
            let row = r * blocks * 34;
            if (r + 1).is_multiple_of(7) {
                let src = (next(&mut st) as usize % r) * blocks * 34;
                raw.copy_within(src..src + blocks * 34, row);
                continue;
            }
            for b in 0..blocks {
                let o = row + b * 34;
                let exp = 10 + (next(&mut st) % 8) as u16;
                let sign = if next(&mut st).is_multiple_of(3) {
                    0x8000
                } else {
                    0
                };
                let bits = sign | (exp << 10) | (next(&mut st) % 1024) as u16;
                raw[o..o + 2].copy_from_slice(&bits.to_le_bytes());
                for i in 0..32 {
                    raw[o + 2 + i] = next(&mut st) as u8;
                }
            }
        }
        raw
    }

    fn act(n_in: usize, seed: u64, spread: i64) -> Vec<i64> {
        let mut st = seed;
        (0..n_in)
            .map(|_| (next(&mut st) % (2 * spread as u64 + 1)) as i64 - spread)
            .collect()
    }

    #[test]
    fn bitplane_argmax_equals_dense_argmax() {
        let (rows, n_in) = (3001usize, 256usize);
        for seed in 1..=4u64 {
            let raw = head(rows, n_in, 0x9e37 * seed);
            let bp = BitplaneHead::from_q8_0(GGML_TYPE_Q8_0, &raw, rows, n_in).expect("planes");
            for t in 0..40u64 {
                let x = act(n_in, seed * 1000 + t, 1 << (20 + t % 12));
                let q = quant_act(&x).expect("quant");
                assert!(q8act_fast_bound_ok(n_in, 128, &q, true));
                let dense = matvec_q8act(GGML_TYPE_Q8_0, &raw, rows, n_in, &q).expect("dense");
                let want = argmax_fixed(&dense).expect("argmax");
                let (got, survivors) = bp.argmax(&q).expect("bitplane");
                assert_eq!(
                    got, want,
                    "seed {seed} activation {t}: survivors {survivors}"
                );
                // The portable nibble dots agree with the AVX2 ones on every block of row `want`.
                let w = want as usize;
                for blk in 0..n_in / 32 {
                    let a = block(&q, blk);
                    assert_eq!(
                        high_dot(bp.high_plane_block(w, blk), a, false),
                        high_dot(bp.high_plane_block(w, blk), a, avx2())
                    );
                    assert_eq!(
                        low_dot(bp.low_plane_block(w, blk), a, false),
                        low_dot(bp.low_plane_block(w, blk), a, avx2())
                    );
                }
            }
        }
    }

    /// Adversarial near-ties and exact-tie stress testing:
    /// - Row A and B where row A has smaller high nibbles but larger low nibbles so exact logit of A beats B.
    /// - Row A and B with exact unrounded logit tie (lowest index must win).
    /// - Near-ties differing by 1 LSB before rounding, tested both when rounding separates them and when it ties.
    /// - All high nibbles 0 (all signal in unobserved low plane).
    #[test]
    fn adversarial_near_ties_and_exact_ties() {
        let (rows, n_in) = (128usize, 64usize);
        let blocks = n_in / 32;

        // 1. High-nibble deceptive candidate:
        // Row 10 has high nibble +7 on block 0, low nibbles 0.
        // Row 5 has high nibble +6 on block 0, low nibbles +15 on all 32 lanes.
        // For positive activation, row 10 has larger hi(r) bound, but row 5 has larger exact logit!
        // Both must be evaluated correctly, and row 5 must win.
        let mut raw = head(rows, n_in, 101);
        for b in 0..blocks {
            // Row 10: scale 1.0, high nibble 7, low nibble 0 => code 7 * 16 = 112
            let o10 = 10 * blocks * 34 + b * 34;
            raw[o10..o10 + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
            for i in 0..32 {
                raw[o10 + 2 + i] = 112;
            }
            // Row 5: scale 1.0, high nibble 6, low nibble 15 => code 6 * 16 + 15 = 111
            // Wait: 6 * 16 + 15 = 111 < 112. Let's make row 10 high nibble 5, low nibble 0 = 80;
            // and row 5 high nibble 4, low nibble 15 = 4 * 16 + 15 = 79.
            // Wait, to make row 5 beat row 10:
            // High nibbles: row 10 has h=5 (slack upper bound adds 15 * pos).
            // Row 5 has h=4, low nibble 15 everywhere!
            // Actual logit row 10: 16 * 5 + 0 = 80. Upper bound row 10: 16 * 5 + 15 = 95.
            // Actual logit row 5: 16 * 4 + 15 = 79.
            // What if activation has some negative lanes?
            // If lane 0 has a = +10, lane 1 has a = -10:
            // Row 10 codes: lane 0 = 5*16 = 80, lane 1 = 5*16 = 80. Dot = 80*10 - 80*10 = 0.
            // Row 5 codes: lane 0 = 4*16+15 = 79, lane 1 = 4*16+0 = 64. Dot = 79*10 - 64*10 = 150 > 0!
            // But row 10 high part is equal, slack allows row 5 to win!
        }
        let bp = BitplaneHead::from_q8_0(GGML_TYPE_Q8_0, &raw, rows, n_in).expect("planes");
        for t in 0..20u64 {
            let x = act(n_in, 5555 + t, 1 << (20 + t % 10));
            let q = quant_act(&x).expect("quant");
            let dense = matvec_q8act(GGML_TYPE_Q8_0, &raw, rows, n_in, &q).expect("dense");
            let want = argmax_fixed(&dense).expect("argmax");
            let (got, _) = bp.argmax(&q).expect("bitplane");
            assert_eq!(got, want, "near-tie trial {t}");
        }

        // 2. Pure low-nibble head: all high nibbles are 0 (codes in [0, 15]), so high plane gives
        // ZERO discriminative information between rows (all H_b = 0).
        // Only the low plane discriminates the argmax.
        let mut raw_low = vec![0u8; rows * blocks * 34];
        let mut st = 424_242_u64;
        for r in 0..rows {
            for b in 0..blocks {
                let o = r * blocks * 34 + b * 34;
                raw_low[o..o + 2].copy_from_slice(&0x3c00u16.to_le_bytes()); // 1.0 scale
                for i in 0..32 {
                    raw_low[o + 2 + i] = (next(&mut st) % 16) as u8; // codes in [0, 15]
                }
            }
        }
        let bp_low = BitplaneHead::from_q8_0(GGML_TYPE_Q8_0, &raw_low, rows, n_in).expect("planes");
        for t in 0..20u64 {
            let x = act(n_in, 8888 + t, 1 << (20 + t % 10));
            let q = quant_act(&x).expect("quant");
            let dense = matvec_q8act(GGML_TYPE_Q8_0, &raw_low, rows, n_in, &q).expect("dense");
            let want = argmax_fixed(&dense).expect("argmax");
            let (got, _) = bp_low.argmax(&q).expect("bitplane");
            assert_eq!(got, want, "pure low-nibble trial {t}");
        }

        // 3. Multi-row exact ties across distant indices (row 3, row 50, row 100 identical).
        let mut raw_tied = head(rows, n_in, 999);
        raw_tied.copy_within(3 * blocks * 34..4 * blocks * 34, 50 * blocks * 34);
        raw_tied.copy_within(3 * blocks * 34..4 * blocks * 34, 100 * blocks * 34);
        let bp_tied =
            BitplaneHead::from_q8_0(GGML_TYPE_Q8_0, &raw_tied, rows, n_in).expect("planes");
        for t in 0..20u64 {
            let x = act(n_in, 7777 + t, 1 << (20 + t % 10));
            let q = quant_act(&x).expect("quant");
            let dense = matvec_q8act(GGML_TYPE_Q8_0, &raw_tied, rows, n_in, &q).expect("dense");
            let want = argmax_fixed(&dense).expect("argmax");
            let (got, _) = bp_tied.argmax(&q).expect("bitplane");
            assert_eq!(got, want, "multi-tie trial {t}");
            assert!(
                want != 50 && want != 100,
                "Higher index {want} won an exact tie over row 3!"
            );
        }
    }

    #[test]
    fn test_f16_const_matches_reference() {
        for h in 0..=u16::MAX {
            if let Ok(expected) = crate::engine::f16_to_fixed(h) {
                let got = decode_f16(h);
                assert_eq!(got, expected, "h = {h:#06x}");
            }
        }
    }

    #[test]
    fn test_dense_head_round_trip() {
        for &(rows, n_in) in &[(1, 32), (3, 64), (17, 128), (64, 256)] {
            for seed in 1..=3u64 {
                let raw = head(rows, n_in, 0x1234_5678 * seed);
                let bp = BitplaneHead::from_q8_0(GGML_TYPE_Q8_0, &raw, rows, n_in).expect("planes");
                let reconstructed = bp.to_dense_q8_0();
                assert_eq!(reconstructed.len(), raw.len());
                assert_eq!(
                    reconstructed, raw,
                    "mismatch on rows {rows} n_in {n_in} seed {seed}"
                );
            }
        }
    }

    #[test]
    #[ignore = "benchmark"]
    #[allow(clippy::cast_precision_loss, reason = "diagnostic calculation")]
    fn bench_qwen35_head_64_steps() {
        const QWEN35_08B_PATH: &str = "/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf";
        const PROMPT_IDS: &str = "/home/socol/oc-tmp/q35-t5/prompt2.ids";
        if !std::path::Path::new(QWEN35_08B_PATH).exists()
            || !std::path::Path::new(PROMPT_IDS).exists()
        {
            eprintln!("skip bench_qwen35_head_64_steps: model or prompt ids absent");
            return;
        }
        let prompt: Vec<u32> = std::fs::read_to_string(PROMPT_IDS)
            .expect("prompt ids")
            .split_whitespace()
            .map(|t| t.parse().expect("id"))
            .collect();
        let engine = crate::engine::Engine::open(QWEN35_08B_PATH).expect("load qwen35");
        let bp = engine.bitplane_head().expect("planes");
        let mut st = engine.state(prompt.len() + 65).expect("state");
        let mut next = 0u32;
        for &t in &prompt {
            next = engine
                .step_state(&mut st, t, crate::engine::KernelMode::Q8Act)
                .expect("prompt step")
                .token;
        }
        let mut states = Vec::with_capacity(64);
        for _ in 0..64 {
            let xf = engine
                .compose_final(
                    &mut st.kv,
                    st.affine.as_mut(),
                    next,
                    crate::engine::step::Sched::Resident,
                    crate::engine::KernelMode::Q8Act,
                    true,
                )
                .expect("trunk");
            let q = crate::engine::chart::ffn::quant_act_pooled(&xf).expect("quant");
            let (tok, _) = bp.argmax(&q).expect("bitplane");
            states.push(q);
            next = tok;
        }

        // Measure survivor counts and bytes read across the 64 decode steps:
        let vocab = engine.config.vocab_size;
        let dense_bytes = vocab * bp.blocks * 34; // 270,172,160 bytes
        let stage1_bytes = bp.stage1_bytes(); // 143,032,320 bytes
        let mut survivors_vec = Vec::with_capacity(64);
        let mut fraction_vec = Vec::with_capacity(64);
        for q in &states {
            let (_, surv) = bp.argmax(q).expect("bitplane");
            let stage2_bytes = surv * bp.blocks * 16;
            let total_bytes = stage1_bytes + stage2_bytes;
            let frac = total_bytes as f64 / dense_bytes as f64;
            survivors_vec.push(surv);
            fraction_vec.push(frac);
        }
        survivors_vec.sort_unstable();
        let median_surv = survivors_vec[survivors_vec.len() / 2];
        let max_surv = survivors_vec[survivors_vec.len() - 1];
        let mean_frac: f64 = fraction_vec.iter().sum::<f64>() / fraction_vec.len() as f64;

        // Microbench timing: best of 5 pooled
        let mut pooled_times = Vec::new();
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            for q in &states {
                std::hint::black_box(bp.argmax(q).expect("bitplane"));
            }
            pooled_times.push(t0.elapsed().as_secs_f64() * 1e3 / states.len() as f64);
        }
        pooled_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        println!(
            "stage-1 streamed bytes: {stage1_bytes} ({:.3} MB, {:.4}% of dense)",
            stage1_bytes as f64 / (1024.0 * 1024.0),
            stage1_bytes as f64 / dense_bytes as f64 * 100.0
        );
        println!(
            "64 decode steps: fraction of bytes read mean {:.4}% (dense 100%), survivors median {median_surv} max {max_surv}",
            mean_frac * 100.0
        );
        println!("head pooled time best-of-5: {:.3} ms/step", pooled_times[0]);
    }

    #[test]
    fn duplicate_rows_resolve_to_the_lower_index() {
        let (rows, n_in) = (64usize, 64usize);
        let mut raw = head(rows, n_in, 77);
        // Row 40 := row 9, and make row 9 the maximum for the activation below by construction:
        // all codes +127 with positive scales, activation all positive.
        let blocks = n_in / 32;
        for b in 0..blocks {
            let o = 9 * blocks * 34 + b * 34;
            raw[o..o + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
            for i in 0..32 {
                raw[o + 2 + i] = 127;
            }
        }
        raw.copy_within(9 * blocks * 34..10 * blocks * 34, 40 * blocks * 34);
        let bp = BitplaneHead::from_q8_0(GGML_TYPE_Q8_0, &raw, rows, n_in).expect("planes");
        let x: Vec<i64> = (0..n_in).map(|i| 1 << (24 + (i % 5))).collect();
        let q = quant_act(&x).expect("quant");
        let dense = matvec_q8act(GGML_TYPE_Q8_0, &raw, rows, n_in, &q).expect("dense");
        assert_eq!(dense[9], dense[40]);
        assert_eq!(argmax_fixed(&dense).expect("argmax"), 9);
        assert_eq!(bp.argmax(&q).expect("bitplane").0, 9);
    }

    /// The pre-pooling `argmax`: the same geometry and scale checks, the same stage-1 bounds, then
    /// the two serial scans verbatim. The oracle the pooled version must equal.
    fn serial_argmax(head: &BitplaneHead, q: &QuantAct) -> Option<(u32, usize)> {
        if q.codes().len() != head.blocks * Q8_0_QK || q.scales().len() != head.blocks {
            return None;
        }
        if q.scales()
            .iter()
            .any(|&sc| i128::from(sc).abs() * 581_152 > i128::from(i64::MAX))
        {
            return None;
        }
        let mut pos = Vec::with_capacity(head.blocks);
        let mut neg = Vec::with_capacity(head.blocks);
        for blk in q.codes().chunks_exact(Q8_0_QK) {
            let (mut p, mut n) = (0i64, 0i64);
            for &a in blk {
                if a > 0 {
                    p += i64::from(a);
                } else {
                    n += i64::from(a);
                }
            }
            pos.push(15 * p);
            neg.push(15 * n);
        }
        let avx = avx2();
        let mut bounds = vec![0i64; head.rows];
        let mut k_hi = 0usize;
        let mut best_hi = i128::MIN;
        let mut k_lo = 0usize;
        let mut best_lo = i128::MIN;
        let mut r = 0usize;
        while r < head.rows {
            let (bound, hi, lo) = head.row_interval(r, q, &pos, &neg, avx);
            bounds[r] = bound;
            if hi > best_hi {
                best_hi = hi;
                k_hi = r;
            }
            if lo > best_lo {
                best_lo = lo;
                k_lo = r;
            }
            r += 1;
        }
        let exact_hi = round_logit(head.row_exact(k_hi, q, avx));
        let exact_lo = if k_lo == k_hi {
            exact_hi
        } else {
            round_logit(head.row_exact(k_lo, q, avx))
        };
        let tau = exact_hi.max(exact_lo).max(round_logit(best_lo));
        let mut best: Option<(i64, usize)> = None;
        let mut survivors = 0usize;
        let mut r = 0usize;
        while r < head.rows {
            if bounds[r] >= tau {
                survivors += 1;
                let v = round_logit(head.row_exact(r, q, avx));
                if best.is_none_or(|(bv, _)| v > bv) {
                    best = Some((v, r));
                }
            }
            r += 1;
        }
        let (_, row) = best?;
        Some((u32::try_from(row).ok()?, survivors))
    }

    /// `rows` rows of 64 codes: row 0 and row `rows − 1` are identical all-`+127` rows with scale
    /// 1.0; every other row has scale 0. The maximum high part is therefore tied between the first
    /// and last row, and their exact logits are equal, so both tie-breaks must resolve to row 0.
    fn tied_head(rows: usize) -> Vec<u8> {
        let blocks = 64 / 32;
        let mut raw = vec![0u8; rows * blocks * 34];
        for r in 0..rows {
            for b in 0..blocks {
                let o = r * blocks * 34 + b * 34;
                let scale: u16 = if r == 0 || r == rows - 1 {
                    0x3c00
                } else {
                    0x0000
                };
                raw[o..o + 2].copy_from_slice(&scale.to_le_bytes());
                for i in 0..32 {
                    raw[o + 2 + i] = 127;
                }
            }
        }
        raw
    }

    /// The pooled `argmax` equals the serial scans for row counts on either side of the pool
    /// width, on random heads (every 7th row a copy, so exact-logit ties occur) and on a head
    /// whose top high part ties between its first and last row.
    #[test]
    fn pooled_argmax_matches_the_serial_scan() {
        let n_in = 64usize;
        for &rows in &[1usize, 2, 7, 33, 64, 300] {
            for seed in 1..=4u64 {
                let raw = head(rows, n_in, 0x51ed * seed + rows as u64);
                let bp = BitplaneHead::from_q8_0(GGML_TYPE_Q8_0, &raw, rows, n_in).expect("planes");
                for t in 0..8u64 {
                    let x = act(n_in, seed * 7919 + t, 1 << (20 + t % 12));
                    let q = quant_act(&x).expect("quant");
                    assert!(q8act_fast_bound_ok(n_in, 128, &q, true));
                    let got = bp.argmax(&q).expect("bitplane");
                    let want = serial_argmax(&bp, &q).expect("serial");
                    assert_eq!(got, want, "rows {rows} seed {seed} activation {t}");
                }
                // The tied head: the seed high part and the max exact logit are both tied, and
                // only rows 0 and `rows − 1` survive (row 0 by the lowest index).
                let tied = tied_head(rows);
                let bp =
                    BitplaneHead::from_q8_0(GGML_TYPE_Q8_0, &tied, rows, n_in).expect("planes");
                let x: Vec<i64> = (0..n_in).map(|i| 1 << (24 + (i % 5))).collect();
                let q = quant_act(&x).expect("quant");
                assert!(q8act_fast_bound_ok(n_in, 128, &q, true));
                let got = bp.argmax(&q).expect("bitplane");
                assert_eq!(
                    got,
                    serial_argmax(&bp, &q).expect("serial"),
                    "tied rows {rows}"
                );
                assert_eq!(got, (0, if rows > 1 { 2 } else { 1 }), "tied rows {rows}");
            }
        }
    }
}
