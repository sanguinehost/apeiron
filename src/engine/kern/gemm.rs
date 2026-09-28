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

/// Packs the columns, or `None` when a scale is not a power of two.
///
/// # Complexity
///
/// `O(k·n_in)` byte writes and `O(k·blocks)` exponents.
fn pack(qs: &[QuantAct], blocks: usize) -> Option<Packed> {
    let k = qs.len();
    let groups = k.div_ceil(LANES);
    let mut words = vec![0x8080_8080u32; groups * blocks * 8 * LANES];
    let mut shift = vec![0i64; groups * blocks * LANES];
    let mut emin = vec![0u32; k];
    let mut spread = vec![0u32; groups];
    for (c, q) in qs.iter().enumerate() {
        let (g, lane) = (c / LANES, c % LANES);
        let scales = q.scales();
        let mut lo = u32::MAX;
        for &s in scales {
            if s < 0 || (s != 0 && s.count_ones() != 1) {
                return None;
            }
            if s != 0 {
                lo = lo.min(s.trailing_zeros());
            }
        }
        let lo = if lo == u32::MAX { 0 } else { lo };
        emin[c] = lo;
        let codes = q.codes();
        for b in 0..blocks {
            let s = scales[b];
            let sh = if s == 0 { 0 } else { s.trailing_zeros() - lo };
            shift[(g * blocks + b) * LANES + lane] = i64::from(sh);
            spread[g] = spread[g].max(sh);
            for quad in 0..8 {
                let at = b * ACT_QK + 4 * quad;
                let mut w = 0u32;
                for (i, &x) in codes[at..at + 4].iter().enumerate() {
                    w |= u32::from(x.cast_unsigned() ^ 0x80) << (8 * i);
                }
                words[((g * blocks + b) * 8 + quad) * LANES + lane] = w;
            }
        }
    }
    Some(Packed {
        groups,
        words,
        shift,
        emin,
        spread,
    })
}

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
/// from L1 across the `⌈k/8⌉` groups. Spatial `O(k·n_in)` packed activation bytes plus
/// `O(n_out·k)` output.
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
        let mut inter = vec![0i64; n_out.checked_mul(k)?];
        let bad = std::sync::atomic::AtomicBool::new(false);
        crate::engine::chart::pool::Pool::global().chunks_mut(
            &mut inter,
            ROWS_PER_TASK * k,
            &|_, start, dst| {
                let r0 = start / k;
                let rows = dst.len() / k;
                // SAFETY: AVX2 and AVX-VNNI were detected above; `raw` holds `n_out` rows and
                // the packing covers `packed.groups` groups of `blocks` blocks.
                let ok =
                    unsafe { x86::rows(raw, row_bytes, blocks, r0, rows, qs, &packed, limit, dst) };
                if !ok {
                    bad.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            },
        );
        if bad.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let mut cols: Vec<Vec<i64>> = (0..k).map(|_| Vec::with_capacity(n_out)).collect();
        for r in 0..n_out {
            for (c, col) in cols.iter_mut().enumerate() {
                col.push(inter[r * k + c]);
            }
        }
        Some(cols)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (ggml_type, raw, n_out, n_in, qs);
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

    /// Rows `r0..r0 + rows` against every column into `dst[r·k + c]`; `false` when an output
    /// leaves `i64` or a weight scale is non-finite (the caller then runs the contract path).
    ///
    /// # Safety
    ///
    /// AVX2 and AVX-VNNI; `raw` holds rows `r0..r0 + rows` of `row_bytes` bytes.
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
        dst: &mut [i64],
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
                        dst[(r + which) * k + c] = v;
                    }
                }
            }
            r += if pair { 2 } else { 1 };
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
