//! Exact lookup-table row dot for `PQ2_0` weights (decode-dominance L6).
//!
//! `PQ2_0` stores one `f16` scale `d_b` per 128 weights and two-bit codes `code_l ∈ {0,1,2,3}` with
//! `w_l = (code_l − 1)·d_b`. [`Pq2Lut::from_pq2_0`] repacks the matrix once into bit planes (`lo`,
//! `hi`, `code = 2·hi + lo`), the nibbles of 32 consecutive rows for one 4-column group contiguous
//! (T-MAC, Wei et al., arXiv `2407.00088` §3.1–3.2).
//!
//! [`matvec_lut`] is bit for bit `matvec_fixed(142, raw, n_out, n_in, &dequant(q))`. Per
//! 4-activation group `g` it builds the exact 16-entry table `P_g[n] = Σ_{k<4} bit_k(n)·xq_{4g+k}`
//! (`|P_g| ≤ 508`) as two byte planes `P_g[n] + 508 = lo + 256·hi`, so a weight group is
//! `2·P_g[hiNib] + P_g[loNib] − Σ_k xq`: one lookup per plane instead of four multiplies, with a
//! 128-weight block contributing `d_b·Σ_{sb<4} S_{4b+sb}·inner_sb`. The fast path interleaves the
//! planes into `i16` sub-block inner and folds each sub-block into `i64` lanes; [`lut_i64_bound_ok`]
//! proves those lanes, [`lut_bound_ok`] the `i128` accumulation, else the call falls back to
//! [`matvec_fixed`]. The `AVX2` twin (`vpshufb`, 32 rows per instruction) is not extracted, and its
//! equality with the portable twin is `tested`.
#![allow(
    dead_code,
    reason = "[`INNER_ABS_MAX`] documents the sub-block bound, and [`matvec_lut`] / \
              [`matvec_lut_rows`] are the unchunked reference the tests compare against; the \
              executing path is `matvec_lut_chunked` via `q8act_projection`"
)]

use crate::engine::chart::task::{ChunkTask, Task3};
use crate::engine::chart::QuantAct;
use crate::engine::num::value::round_shift_32;
use crate::engine::{f16_to_fixed, matvec_fixed, EngineError, MatvecError};
use crate::format::quant::{self, PQ2_0_BLOCK_BYTES, PQ2_0_QK};

/// Lanes of one activation quantization block (the group of 4 divides 32 evenly).
const ACT_QK: usize = 32;

/// Rows per lookup tile; one `vpshufb` covers exactly this many rows.
const TILE_ROWS: usize = 32;

/// The row-boundary multiple the tile-aligned LUT rows need: [`Pq2Lut`] rows report it as the
/// [`crate::engine::chart::ffn::ProjSpec::row_align`] of their projection, so a dispatch never starts a chunk mid-tile.
pub(crate) const LUT_ROW_ALIGN: usize = TILE_ROWS;

/// Activation groups per 32-activation sub-block (`ACT_QK / 4`).
const GROUPS_PER_SUB: usize = ACT_QK / 4;

/// Activation sub-blocks (each with its own `S_b`) per 128-weight `PQ2_0` block.
const SUBS_PER_BLOCK: usize = PQ2_0_QK / ACT_QK;

/// The largest finite `f16` magnitude in Q32.32: `65504 · 2^32`.
const F16_MAX_FIXED: i128 = 65504i128 << 32;

/// The largest `|inner|` of one 32-activation sub-block: `ACT_QK = 32` lanes with `|xq| ≤ 127` and
/// weight codes in `[−1, 2]`, so `|inner| ≤ 32·508 = 16_256`.
const INNER_ABS_MAX: i64 = 16_256;

/// The largest `|Σ_{sb<4} S_sb·inner_sb|` of one 128-weight block divided by `max_b |S_b|`:
/// `SUBS_PER_BLOCK = 4` sub-blocks, each `|inner_sb| ≤ `[`INNER_ABS_MAX`], so `4·16_256 = 65_024`.
const BLOCK_INNER_ABS_MAX: i64 = 65_024;

/// The `+508` offsets folded into the `GROUPS_PER_SUB = 8` groups' lookups of one sub-block,
/// removed once in the reduction: `(2 + 1)·508·8 = 12_192`.
const LUT_COMBINE_OFFSET: i16 = 12_192;

/// A `PQ2_0` weight matrix repacked into the exact lookup-table layout by [`Pq2Lut::from_pq2_0`]
/// (requires `n_in % 128 == 0`). `codes` holds, per 32-row tile and 4-column group, the 32 rows'
/// `lo | hi << 4` bytes at `(tile·n_groups + g)·32 + lane`, padded past `n_out`; `scales` is `d_b`
/// in Q32.32 at `row·n_blocks + b`, padded past `n_out`; `raw` is kept for the checked fallback.
#[derive(Debug)]
pub(crate) struct Pq2Lut {
    /// Output rows.
    n_out: usize,
    /// Weight row width; a multiple of 128.
    n_in: usize,
    /// 128-weight blocks per row (`n_in / 128`).
    n_blocks: usize,
    /// 4-column groups per row (`n_in / 4`).
    n_groups: usize,
    /// 32-row tiles (`n_out.div_ceil(32)`).
    n_tiles: usize,
    /// Per `(row, block)` scale `d_b` in Q32.32, padded to whole tiles.
    scales: Vec<i64>,
    /// `max |d_b|` over every stored scale (including the tile padding), in Q32.32. It is the
    /// weight factor of [`lut_row_i64_bound_ok`]'s per-call bound, so the check is `O(n_in/32)`
    /// rather than `O(n_out·n_in)`.
    d_max: i128,
    /// Packed `lo | hi << 4` code bytes, `(tile, group, lane)` layout.
    codes: Vec<u8>,
    /// The original payload, retained only for the reference fallback.
    raw: Vec<u8>,
}

/// Per-call lookup tables: two byte planes encoding `P_g[n] + 508` and the group sums `Σ_k xq`.
struct LutTables {
    /// Low byte of `P_g[n] + 508`, `g·16 + n`.
    lo: Vec<u8>,
    /// High byte of `P_g[n] + 508`, `g·16 + n`.
    hi: Vec<u8>,
    /// `Σ_{k<4} xq_{4g+k}` per group.
    xs: Vec<i16>,
}

impl LutTables {
    /// `P_g[n] = Σ_{k<4} bit_k(n)·xq_{4g+k}` reconstructed from the two byte planes.
    #[inline]
    fn p(&self, g: usize, n: usize) -> i16 {
        i16::from(self.lo[g * 16 + n]) + i16::from(self.hi[g * 16 + n]) * 256 - 508
    }
}

/// Split one packed `PQ2_0` code byte into its low-plane and high-plane nibbles: the byte holds
/// the four column codes `code_c = lo_c + 2·hi_c` at bits `2c`/`2c+1`, and the returned values are
/// `Σ_c lo_c·2^c` and `Σ_c hi_c·2^c`, the 4-bit indices of the per-group tables. `O(1)` time and
/// space.
fn split_nibbles(byte: u8) -> (u8, u8) {
    let mut lo = 0u8;
    let mut hi = 0u8;
    let mut c = 0usize;
    while c < 4 {
        lo |= ((byte >> (2 * c)) & 1) << c;
        hi |= ((byte >> (2 * c + 1)) & 1) << c;
        c += 1;
    }
    (lo, hi)
}

/// The table body of [`build_tables`] for activation groups `[g0, g1)`, written into the matching
/// ranges of `lo`, `hi` and `xs`: `lo`/`hi` hold `16·(g1 - g0)` bytes and `xs` holds `g1 - g0`
/// entries, all indexed from group `g0`. Groups are independent, which is what lets
/// [`build_tables_pooled`] split the range across the pool.
///
/// `O(g1 - g0)` time and space (16 plus 4 bytes per group).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "`xsum ∈ [-508, 508]` is stored in `i16`, `p + 508 ∈ [0, 1016]` in the two byte \
              planes and each plane byte is masked to `8` bits immediately before the cast; the \
              bounds follow from `|xq| ≤ 127` and four lanes, so no cast can lose a bit. The loop \
              names (`g` group, `n` table entry, `k` lane, `p` partial sum, `x` the four codes) \
              mirror the 16-entry table recurrence they index"
)]
fn build_tables_range(
    codes: &[i8],
    g0: usize,
    g1: usize,
    lo: &mut [u8],
    hi: &mut [u8],
    xs: &mut [i16],
) {
    // The 16 subset sums of one group by doubling: after step `k`, `p[n]` is the sum over the
    // low `k + 1` codes selected by `n`'s low bits, so `p[15]` is the group sum. The first three
    // steps are unrolled into `p[0..8]` because `p` is fixed-size: the unrolled form has no
    // induction variable for the compiler to re-check, and the plane bytes are staged in
    // `lo16`/`hi16` so each plane leaves as one 16-byte copy instead of 16 bounds-checked stores.
    let mut p = [0i32; 16];
    let mut lo16 = [0u8; 16];
    let mut hi16 = [0u8; 16];
    let mut g = g0;
    while g < g1 {
        let base = 4 * g;
        let out = (g - g0) * 16;
        let c0 = i32::from(codes[base]);
        let c1 = i32::from(codes[base + 1]);
        let c2 = i32::from(codes[base + 2]);
        let c3 = i32::from(codes[base + 3]);
        p[0] = 0;
        p[1] = c0;
        p[2] = c1;
        p[3] = c0 + c1;
        p[4] = c2;
        p[5] = c0 + c2;
        p[6] = c1 + c2;
        p[7] = c0 + c1 + c2;
        let mut n = 0usize;
        while n < 8 {
            p[8 + n] = p[n] + c3;
            n += 1;
        }
        xs[g - g0] = p[15] as i16;
        let mut n = 0usize;
        while n < 16 {
            let enc = (p[n] + 508) as u32;
            lo16[n] = (enc & 0xff) as u8;
            hi16[n] = (enc >> 8) as u8;
            n += 1;
        }
        lo[out..out + 16].copy_from_slice(&lo16);
        hi[out..out + 16].copy_from_slice(&hi16);
        g += 1;
    }
}

/// Build the per-token tables from the quantized activation codes: one 16-entry table per
/// 4-activation group, stored as `P + 508 = lo + 256·hi` with `P ∈ [-508, 508]` so both planes fit
/// a `u8`, plus `Σ_{k<4} xq` per group (`|·| ≤ 508`, an `i16`). `O(n_in)` time and space (16 plus
/// 4 bytes per group).
fn build_tables(codes: &[i8]) -> LutTables {
    let n_groups = codes.len() / 4;
    let mut lo = vec![0u8; n_groups * 16];
    let mut hi = vec![0u8; n_groups * 16];
    let mut xs = vec![0i16; n_groups];
    build_tables_range(codes, 0, n_groups, &mut lo, &mut hi, &mut xs);
    LutTables { lo, hi, xs }
}

/// Pool task context for [`build_tables_pooled`]: the codes and the groups per task (FORM-PARALLEL D5).
struct BuildTablesTask<'a> {
    codes: &'a [i8],
    per: usize,
}

impl Task3<u8, u8, i16, ()> for BuildTablesTask<'_> {
    fn run(&self, t: usize, lo: &mut [u8], hi: &mut [u8], xs: &mut [i16]) -> Result<(), ()> {
        build_tables_task(self.codes, self.per, t, lo, hi, xs)
    }
}

/// The body of one [`BuildTablesTask`] piece: groups `[t·per, t·per + xs.len())`.
#[allow(
    clippy::unnecessary_wraps,
    reason = "the pool task type is Result; the body is infallible"
)]
fn build_tables_task(
    codes: &[i8],
    per: usize,
    t: usize,
    lo: &mut [u8],
    hi: &mut [u8],
    xs: &mut [i16],
) -> Result<(), ()> {
    build_tables_range(codes, t * per, t * per + xs.len(), lo, hi, xs);
    Ok(())
}

/// Groups a pool task must own for [`build_tables_pooled`]'s dispatch to be worth its wake-up;
/// below this the serial [`build_tables`] runs.
const PARALLEL_BUILD_GROUPS_PER_TASK: usize = 64;

/// [`build_tables`] with the groups split across the [`Pool`](crate::engine::chart::pool::Pool):
/// each task writes the disjoint group range `[t·per, t·per + per)` of `lo`, `hi` and `xs` with
/// [`build_tables_range`], so the three planes are byte-identical to the serial build (the per-group
/// recurrence is untouched). This takes the table build off the caller's serial path — the build's
/// wall time is `O(n_in / width)` instead of `O(n_in)` — while the row dispatch still sees the whole
/// table. `O(n_in)` total time and space.
fn build_tables_pooled(codes: &[i8]) -> LutTables {
    let n_groups = codes.len() / 4;
    let mut lo = vec![0u8; n_groups * 16];
    let mut hi = vec![0u8; n_groups * 16];
    let mut xs = vec![0i16; n_groups];
    let tasks =
        (n_groups / PARALLEL_BUILD_GROUPS_PER_TASK).min(crate::engine::chart::pool::width_global());
    if tasks <= 1 {
        build_tables_range(codes, 0, n_groups, &mut lo, &mut hi, &mut xs);
        return LutTables { lo, hi, xs };
    }
    let per = n_groups.div_ceil(tasks);
    let _: Result<(), ()> = crate::engine::chart::pool::for_chunks3(
        &mut lo,
        16 * per,
        &mut hi,
        16 * per,
        &mut xs,
        per,
        &BuildTablesTask { codes, per },
    );
    LutTables { lo, hi, xs }
}

/// Whether the exact per-row `i128` accumulation provably stays inside `i128` for this call.
///
/// With `|d_b| ≤ 65504·2^32`, `S_max = max_b |S_b|` and `|inner| ≤ 16256 = 32·508`, the reference
/// dot and this kernel's partial sums are bounded by `n_in·(65504·2^32)·S_max·508`. Below
/// `i128::MAX` no checked arithmetic can refuse and the fast path is exact; a saturated product
/// equals `i128::MAX`, indistinct from an exact bound at the maximum, so that case takes the
/// reference fallback. `O(n_in/32)` time to scan the scales, `O(1)` space.
fn lut_bound_ok(n_in: usize, act_scales: &[i64]) -> bool {
    let mut s_max: i128 = 0;
    let mut i = 0usize;
    while i < act_scales.len() {
        let a = i128::from(act_scales[i]).abs();
        if a > s_max {
            s_max = a;
        }
        i += 1;
    }
    let bound = (n_in as i128)
        .saturating_mul(F16_MAX_FIXED)
        .saturating_mul(s_max)
        .saturating_mul(508);
    bound < i128::MAX
}

/// Whether the fast path's per-block `i64` lane can hold every activation scale product of this
/// call: `|Σ_{sb<4} S_sb·inner_sb| ≤ max_b |S_b|·`[`BLOCK_INNER_ABS_MAX`]` ≤ i64::MAX`. When it
/// holds, each `S_b·inner_sb` (`|inner_sb| ≤ `[`INNER_ABS_MAX`]) and its four-term sum fit `i64`,
/// so the lane arithmetic truncates nothing and the sub-block `i16` reduction stays inside `i16`.
/// `O(n_in/32)` time to scan the scales, `O(1)` space.
fn lut_i64_bound_ok(act_scales: &[i64]) -> bool {
    let mut s_max: i128 = 0;
    let mut i = 0usize;
    while i < act_scales.len() {
        let a = i128::from(act_scales[i]).abs();
        if a > s_max {
            s_max = a;
        }
        i += 1;
    }
    s_max.saturating_mul(i128::from(BLOCK_INNER_ABS_MAX)) <= i128::from(i64::MAX)
}

/// Whether the exact per-row dot can be accumulated in two `i64` lanes instead of one `i128`: the
/// high half `W = Σ_b ⌊d_b·v_b / 2^32⌋` and the low half `L = Σ_b (d_b·v_b mod 2^32)`, whose
/// recombination `W·2^32 + L` is the row accumulator.
///
/// The derivation, from the proved `Numeric/Matvec/LutPQ2.lean` `planeSum_bound` (`|P_g| ≤ 508` for four
/// activation codes `|xq| ≤ 127`):
///
/// * one group's table entry is `|P_g| ≤ 508` (`planeSum_bound`);
/// * one 32-activation sub-block `inner = 2·Σ P_hi + Σ P_lo − Σ xq` over `GROUPS_PER_SUB = 8`
///   groups is `|inner| ≤ 2·8·508 + 8·508 + 8·508 = 32·508 = 16_256` = [`INNER_ABS_MAX`];
/// * one 128-weight block `v_b = Σ_{sb<4} S_sb·inner_sb` is `|v_b| ≤ 4·S_max·16_256`, i.e.
///   `S_max·`[`BLOCK_INNER_ABS_MAX`] with `S_max = max_b |S_b|`;
/// * one row `R = Σ_b d_b·v_b` is `|R| ≤ n_blocks·d_max·S_max·65_024`.
///
/// Since `d_b·v_b = w_b·2^32 + r_b` with `0 ≤ r_b < 2^32`, `R = W·2^32 + L` exactly and
/// `|W| ≤ n_blocks·d_max·S_max·65_024 / 2^32`, `0 ≤ L < n_blocks·2^32`. Both halves are in `i64`
/// exactly when that bound is at most `i64::MAX·2^32` and `n_blocks·2^32 ≤ i64::MAX`; the reduction
/// `round_shift_32(R) = W + ⌊(L + 2^31)/2^32⌋` then matches [`round_shift_32`] bit for bit. The
/// width is therefore chosen by the proved bound, not by a runtime overflow check. `O(n_in/32)`
/// time to scan the activation scales, `O(1)` space.
fn lut_row_i64_bound_ok(lut: &Pq2Lut, act_scales: &[i64]) -> bool {
    let mut s_max: i128 = 0;
    let mut i = 0usize;
    while i < act_scales.len() {
        let a = i128::from(act_scales[i]).abs();
        if a > s_max {
            s_max = a;
        }
        i += 1;
    }
    let per_row = (lut.n_blocks as i128)
        .saturating_mul(lut.d_max)
        .saturating_mul(s_max)
        .saturating_mul(i128::from(BLOCK_INNER_ABS_MAX));
    let high_ok = per_row <= (i128::from(i64::MAX) << 32);
    let low_ok = (lut.n_blocks as i128).saturating_mul(1i128 << 32) <= i128::from(i64::MAX);
    high_ok && low_ok
}

/// The per-block `i64` lanes of one 32-row tile: `v[lane] = Σ_{sb<4} S_{4b+sb}·inner_{b,sb,lane}`
/// on the portable lane loop. The two code bytes of each group are read with [`LutTables::p`] and
/// the scale is the sub-block's own `S_b`; `v` is exact in `i64` because the caller checked
/// [`lut_i64_bound_ok`]. `O(ACT_QK·4)` time per block, `O(1)` space.
#[allow(
    clippy::many_single_char_names,
    reason = "the index names (`b` 128-weight block, `sb` 32-activation sub-block, `g` group, `s` \
              activation scale) mirror the exact factorization they index; `acc` is exact because \
              |acc| ≤ max|S|·BLOCK_INNER_ABS_MAX ≤ i64::MAX by `lut_i64_bound_ok`"
)]
fn block_inners_portable(
    lut: &Pq2Lut,
    tables: &LutTables,
    q: &QuantAct,
    tile: usize,
    b: usize,
    v: &mut [i64; TILE_ROWS],
) {
    let mut lane = 0usize;
    while lane < TILE_ROWS {
        let mut acc: i64 = 0;
        let mut sb = 0usize;
        while sb < SUBS_PER_BLOCK {
            let s = q.scales()[SUBS_PER_BLOCK * b + sb];
            let gbase = (SUBS_PER_BLOCK * b + sb) * GROUPS_PER_SUB;
            let mut sum_hi: i64 = 0;
            let mut sum_lo: i64 = 0;
            let mut sum_x: i64 = 0;
            let mut g8 = 0usize;
            while g8 < GROUPS_PER_SUB {
                let g = gbase + g8;
                let idx = (tile * lut.n_groups + g) * TILE_ROWS + lane;
                sum_hi += i64::from(tables.p(g, usize::from(lut.codes[idx] >> 4)));
                sum_lo += i64::from(tables.p(g, usize::from(lut.codes[idx] & 0x0f)));
                sum_x += i64::from(tables.xs[g]);
                g8 += 1;
            }
            acc += s * (2 * sum_hi + sum_lo - sum_x);
            sb += 1;
        }
        v[lane] = acc;
        lane += 1;
    }
}

/// Dispatch one 32-row tile's block to the `AVX2` core or the portable twin.
fn block_inners(
    lut: &Pq2Lut,
    tables: &LutTables,
    q: &QuantAct,
    tile: usize,
    b: usize,
    v: &mut [i64; TILE_ROWS],
    avx: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if avx {
            // SAFETY: `avx` is only true when `is_x86_feature_detected!("avx2")` returned true,
            // and the caller's construction gives the plane, table and scale extents the core
            // reads.
            unsafe { block_inners_avx2(lut, tables, q, tile, b, v) };
            return;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx;
    block_inners_portable(lut, tables, q, tile, b, v);
}

/// Fold one eight-lane `i32` group of a sub-block's `i16` inner into its 64-bit lane accumulators.
/// With `S` split as `a` (the low 32 bits, taken as `i32`) plus `b` times `2^32`, the product of `S`
/// and `inner` is the product of `a` and `inner` plus the product of `b` and `inner` times `2^32`,
/// exactly. `acc_e` takes the even rows of `x` and `acc_o` the odd rows. The caller's
/// `lut_i64_bound_ok` keeps the whole sub-block sum inside `i64::MAX`, with `|inner|` at most
/// `INNER_ABS_MAX`, `|a|` at most `2^31` and `|b|` at most `2^15`, so `|a·inner|` is at most `2^46`,
/// `|b·inner|` at most `2^30`, and every partial sum is exact in `i64`. `AVX2` must be present.
/// `O(8)` time, `O(1)` space.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    reason = "`b` is `(S - low 32 bits) >> 32`, an exact multiple of `2^32` whose magnitude is at \
              most `2^15` because `S` is inside the checked [`lut_i64_bound_ok`] band, so the `as \
              i32` narrowing cannot wrap; the index names (`a` low half, `b` high half) mirror the \
              exact decomposition they carry"
)]
unsafe fn fold8(
    acc_e: &mut std::arch::x86_64::__m256i,
    acc_o: &mut std::arch::x86_64::__m256i,
    x: std::arch::x86_64::__m256i,
    a: std::arch::x86_64::__m256i,
    b: std::arch::x86_64::__m256i,
) {
    use std::arch::x86_64::{
        _mm256_add_epi64, _mm256_mul_epi32, _mm256_slli_epi64, _mm256_srli_epi64,
    };
    let t1 = _mm256_mul_epi32(a, x);
    let t2 = _mm256_mul_epi32(b, x);
    let xo = _mm256_srli_epi64::<32>(x);
    let u1 = _mm256_mul_epi32(a, xo);
    let u2 = _mm256_mul_epi32(b, xo);
    *acc_e = _mm256_add_epi64(*acc_e, _mm256_add_epi64(t1, _mm256_slli_epi64::<32>(t2)));
    *acc_o = _mm256_add_epi64(*acc_o, _mm256_add_epi64(u1, _mm256_slli_epi64::<32>(u2)));
}

/// The `AVX2` per-block 64-bit lanes of one 32-row tile: per group `vpshufb` selects the low and
/// high byte planes of `P_g + 508` by its nibbles and `unpacklo/hi_epi8` interleaves them into
/// `i16` `lo + 256·hi`; the high plane carries coefficient `2`, one [`LUT_COMBINE_OFFSET`] is
/// removed per sub-block, and each sub-block's `i16` inner is folded into `i64` by [`fold8`] with
/// that sub-block's `S_b`. `AVX2` must be present, and `lut.codes` holds `n_tiles·n_groups·32`
/// bytes, `tables.lo`/`tables.hi` `n_groups·16`, `tables.xs` `n_groups`, `q.scales()` at least
/// `4·n_blocks`, per the caller's construction and the `tile < n_tiles`, `b < n_blocks` guards.
/// `O(ACT_QK·4)` time per block, `O(1)` space.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    reason = "one `#[target_feature]` body keeps every intrinsic in scope with `AVX2` enabled; \
              splitting it would move them into functions without the feature. The index names \
              (`b` block, `g` group, `s` activation scale) mirror the exact factorization they \
              index, and the two `as i32` narrowings are deliberate: `a32` is `s`'s low 32 bits \
              and `b32` is the exact `(s - a32) >> 32`, whose magnitude [`lut_i64_bound_ok`] \
              bounds by `2^15`, so `a32 + b32·2^32 = s` and neither wrap loses a bit"
)]
unsafe fn block_inners_avx2(
    lut: &Pq2Lut,
    tables: &LutTables,
    q: &QuantAct,
    tile: usize,
    b: usize,
    v: &mut [i64; TILE_ROWS],
) {
    use std::arch::x86_64::{
        _mm256_add_epi16, _mm256_and_si256, _mm256_broadcastsi128_si256, _mm256_castsi256_si128,
        _mm256_cvtepi16_epi32, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_set1_epi16,
        _mm256_set1_epi32, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
        _mm256_slli_epi16, _mm256_srli_epi16, _mm256_sub_epi16, _mm256_unpackhi_epi64,
        _mm256_unpackhi_epi8, _mm256_unpacklo_epi64, _mm256_unpacklo_epi8, _mm_loadu_si128,
        _mm_storeu_si128,
    };
    // One `(even, odd)` 64-bit lane pair per eight-lane `i32` group of the tile; the sub-block
    // loop accumulates every sub-block into them, so the fold and the store happen once per block.
    let mut acc_e = [_mm256_setzero_si256(); 4];
    let mut acc_o = [_mm256_setzero_si256(); 4];
    let mut sb = 0usize;
    while sb < SUBS_PER_BLOCK {
        // `acc_hi_*` collects the coefficient-`2` high-plane lookups, `acc_lo_*` the low plane;
        // `u` holds rows 0..8 and 16..24, `v` holds 8..16 and 24..32 after the byte interleave.
        let mut acc_hi_u = _mm256_setzero_si256();
        let mut acc_hi_v = _mm256_setzero_si256();
        let mut acc_lo_u = _mm256_setzero_si256();
        let mut acc_lo_v = _mm256_setzero_si256();
        let mut xsum: i16 = 0;
        let gbase = (SUBS_PER_BLOCK * b + sb) * GROUPS_PER_SUB;
        let mut g8 = 0usize;
        while g8 < GROUPS_PER_SUB {
            let g = gbase + g8;
            xsum += tables.xs[g];
            let idx_base = (tile * lut.n_groups + g) * TILE_ROWS;
            let tab_base = g * 16;
            // SAFETY: the caller's construction gives `idx_base + 32 <= lut.codes.len()` and
            // `tab_base + 16 <= tables.lo.len()`; every load reads exactly those bytes. The packed
            // byte is `lo | hi << 4`; after the mask the two nibble planes are each `<= 15`, so the
            // high bit of every index byte is clear and `vpshufb` selects inside the duplicated
            // 16-entry table. Byte value `b` of a plane is the matching byte of `P_g + 508`, so
            // `lo + 256·hi` is that table entry.
            unsafe {
                let packed = _mm256_loadu_si256(lut.codes.as_ptr().add(idx_base).cast());
                let m0f = _mm256_set1_epi8(0x0f);
                let idx_lo = _mm256_and_si256(packed, m0f);
                let idx_hi = _mm256_and_si256(_mm256_srli_epi16::<4>(packed), m0f);
                let tab_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                    tables.lo.as_ptr().add(tab_base).cast(),
                ));
                let tab_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                    tables.hi.as_ptr().add(tab_base).cast(),
                ));
                let a_lo = _mm256_shuffle_epi8(tab_lo, idx_lo);
                let b_lo = _mm256_shuffle_epi8(tab_hi, idx_lo);
                let a_hi = _mm256_shuffle_epi8(tab_lo, idx_hi);
                let b_hi = _mm256_shuffle_epi8(tab_hi, idx_hi);
                acc_lo_u = _mm256_add_epi16(acc_lo_u, _mm256_unpacklo_epi8(a_lo, b_lo));
                acc_lo_v = _mm256_add_epi16(acc_lo_v, _mm256_unpackhi_epi8(a_lo, b_lo));
                acc_hi_u = _mm256_add_epi16(acc_hi_u, _mm256_unpacklo_epi8(a_hi, b_hi));
                acc_hi_v = _mm256_add_epi16(acc_hi_v, _mm256_unpackhi_epi8(a_hi, b_hi));
            }
            g8 += 1;
        }
        let cst = _mm256_set1_epi16(LUT_COMBINE_OFFSET);
        let xv = _mm256_set1_epi16(xsum);
        let inner_u = _mm256_sub_epi16(
            _mm256_sub_epi16(
                _mm256_add_epi16(_mm256_slli_epi16::<1>(acc_hi_u), acc_lo_u),
                cst,
            ),
            xv,
        );
        let inner_v = _mm256_sub_epi16(
            _mm256_sub_epi16(
                _mm256_add_epi16(_mm256_slli_epi16::<1>(acc_hi_v), acc_lo_v),
                cst,
            ),
            xv,
        );
        // `inner_u` lane 0 holds rows 0..8 and lane 1 rows 16..24; `inner_v` lane 0 rows 8..16 and
        // lane 1 rows 24..32. `s = a32 + b32·2^32` exactly, and `lut_i64_bound_ok` bounds every
        // partial sum, so the 64-bit lanes truncate nothing.
        let s = q.scales()[SUBS_PER_BLOCK * b + sb];
        let a32 = s as i32;
        let b32 = ((s - i64::from(a32)) >> 32) as i32;
        let av = _mm256_set1_epi32(a32);
        let bv = _mm256_set1_epi32(b32);
        // SAFETY: the four `cvtepi16_epi32` reads take the two halves of `inner_u`/`inner_v` in
        // turn, so every 8-`i16` group feeds exactly one `fold8`; the pointers are the caller's
        // `v`, whose `TILE_ROWS` lanes the four stores below cover exactly once.
        unsafe {
            let xa = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(inner_u));
            let xb = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(inner_u));
            let xc = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(inner_v));
            let xd = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(inner_v));
            fold8(&mut acc_e[0], &mut acc_o[0], xa, av, bv);
            fold8(&mut acc_e[1], &mut acc_o[1], xb, av, bv);
            fold8(&mut acc_e[2], &mut acc_o[2], xc, av, bv);
            fold8(&mut acc_e[3], &mut acc_o[3], xd, av, bv);
        }
        sb += 1;
    }
    // `acc_e[0,1,2,3]` cover rows 0..8, 16..24, 8..16 and 24..32, so the lane-0/1 pairs land at
    // row offsets `8·dest[j]`; within a group, `unpacklo_epi64` holds the even rows of the low
    // half and `unpackhi_epi64` the odd rows of the high half, each half's eight `i64` written as
    // four `i128` stores.
    let dest = [0usize, 2, 1, 3];
    let mut j = 0usize;
    while j < 4 {
        // SAFETY: `v` holds `TILE_ROWS = 32` `i64`; the four stores of iteration `j` write the
        // sixteen bytes at `v[8·dest[j]..]` through `v[8·dest[j] + 8..]` and the four `dest` values
        // partition `0..4`, so every lane is written exactly once.
        unsafe {
            let eo_lo = _mm256_unpacklo_epi64(acc_e[j], acc_o[j]);
            let eo_hi = _mm256_unpackhi_epi64(acc_e[j], acc_o[j]);
            let base = v.as_mut_ptr().add(8 * dest[j]);
            _mm_storeu_si128(base.cast(), _mm256_castsi256_si128(eo_lo));
            _mm_storeu_si128(base.add(2).cast(), _mm256_castsi256_si128(eo_hi));
            _mm_storeu_si128(base.add(4).cast(), _mm256_extracti128_si256::<1>(eo_lo));
            _mm_storeu_si128(base.add(6).cast(), _mm256_extracti128_si256::<1>(eo_hi));
        }
        j += 1;
    }
}

impl Pq2Lut {
    /// Repack an `n_out × n_in` `PQ2_0` payload into the exact lookup-table layout.
    ///
    /// The payload must be the row-major block stream of ggml type `PQ2_0` (`142`): per row
    /// `n_in / 128` blocks of 34 bytes, an `f16` scale at offset 0 and 32 code bytes at offset 2.
    /// [`MatvecError::RowLength`] refuses a width that is not a whole number of 128-weight blocks,
    /// [`MatvecError::LengthOverflow`] a size or offset leaving `usize`,
    /// [`MatvecError::ShortPayload`] fewer than `n_out` whole rows, and
    /// [`MatvecError::Dequant`]`(`[`EngineError::NonFiniteScale`]`)` an `Inf`/`NaN` scale.
    /// `O(n_out·n_in)` time and space once at load (two byte planes plus the retained payload).
    pub(crate) fn from_pq2_0(raw: &[u8], n_out: usize, n_in: usize) -> Result<Self, MatvecError> {
        if !n_in.is_multiple_of(PQ2_0_QK) {
            return Err(MatvecError::RowLength {
                row: 0,
                weights: n_in - n_in % PQ2_0_QK,
                n_in,
            });
        }
        let n_blocks = n_in / PQ2_0_QK;
        let n_groups = n_in / 4;
        let n_tiles = n_out.div_ceil(TILE_ROWS);
        let row_bytes = n_blocks
            .checked_mul(PQ2_0_BLOCK_BYTES)
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
        let padded_rows = n_tiles
            .checked_mul(TILE_ROWS)
            .ok_or(MatvecError::LengthOverflow)?;
        let plane_len = padded_rows
            .checked_mul(n_groups)
            .ok_or(MatvecError::LengthOverflow)?;
        let scale_len = padded_rows
            .checked_mul(n_blocks)
            .ok_or(MatvecError::LengthOverflow)?;
        let mut scales: Vec<i64> = vec![0i64; scale_len];
        let mut d_max: i128 = 0;
        let mut codes: Vec<u8> = vec![0u8; plane_len];
        let mut o = 0usize;
        let mut ok = true;
        let mut err: Option<MatvecError> = None;
        while o < n_out && ok {
            let tile = o / TILE_ROWS;
            let lane = o % TILE_ROWS;
            let row_base = o * row_bytes;
            let mut b = 0usize;
            while b < n_blocks && ok {
                let off = row_base + b * PQ2_0_BLOCK_BYTES;
                let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                match f16_to_fixed(bits) {
                    Ok(d) => {
                        scales[o * n_blocks + b] = d;
                        let a = i128::from(d).abs();
                        if a > d_max {
                            d_max = a;
                        }
                    }
                    Err(e) => {
                        err = Some(MatvecError::Dequant(e));
                        ok = false;
                    }
                }
                if ok {
                    let mut k = 0usize;
                    while k < 32 {
                        let (lon, hin) = split_nibbles(raw[off + 2 + k]);
                        let idx = (tile * n_groups + b * 32 + k) * TILE_ROWS + lane;
                        codes[idx] = lon | (hin << 4);
                        k += 1;
                    }
                }
                b += 1;
            }
            o += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        Ok(Self {
            n_out,
            n_in,
            n_blocks,
            n_groups,
            n_tiles,
            scales,
            d_max,
            codes,
            raw: raw[..need].to_vec(),
        })
    }
}

/// The exact row accumulator of one `(tile, lane)` via the tables, on the portable lane loop:
/// `O(n_in)` exact integer additions and one `i128` product per weight sub-block; `O(1)` space.
#[allow(
    clippy::many_single_char_names,
    reason = "the index names (`b` 128-weight block, `sb` 32-activation sub-block, `g` group, \
              `s` activation scale, `d` weight scale) mirror the exact factorization they index; \
              the parameter `q` is the quantized activation the handoff names"
)]
fn tile_accs_portable(
    lut: &Pq2Lut,
    tables: &LutTables,
    q: &QuantAct,
    tile: usize,
    accs: &mut [i128; TILE_ROWS],
) {
    let mut lane = 0usize;
    while lane < TILE_ROWS {
        let mut acc: i128 = 0;
        let mut b = 0usize;
        while b < lut.n_blocks {
            let d = i128::from(lut.scales[(tile * TILE_ROWS + lane) * lut.n_blocks + b]);
            let mut inner_block: i128 = 0;
            let mut sb = 0usize;
            while sb < SUBS_PER_BLOCK {
                let s = i128::from(q.scales()[SUBS_PER_BLOCK * b + sb]);
                let mut sum_hi: i64 = 0;
                let mut sum_lo: i64 = 0;
                let mut sum_x: i64 = 0;
                let gbase = (SUBS_PER_BLOCK * b + sb) * GROUPS_PER_SUB;
                let mut g8 = 0usize;
                while g8 < GROUPS_PER_SUB {
                    let g = gbase + g8;
                    let idx = (tile * lut.n_groups + g) * TILE_ROWS + lane;
                    sum_hi += i64::from(tables.p(g, usize::from(lut.codes[idx] >> 4)));
                    sum_lo += i64::from(tables.p(g, usize::from(lut.codes[idx] & 0x0f)));
                    sum_x += i64::from(tables.xs[g]);
                    g8 += 1;
                }
                let inner = 2 * sum_hi + sum_lo - sum_x;
                inner_block += s * i128::from(inner);
                sb += 1;
            }
            acc += d * inner_block;
            b += 1;
        }
        accs[lane] = acc;
        lane += 1;
    }
}

/// Dispatch one 32-row tile to the `AVX2` core or the portable twin.
fn tile_accs(
    lut: &Pq2Lut,
    tables: &LutTables,
    q: &QuantAct,
    tile: usize,
    accs: &mut [i128; TILE_ROWS],
    avx: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if avx {
            // SAFETY: `avx` is only true when `is_x86_feature_detected!("avx2")` returned true,
            // and the caller's checks give the plane, table and scale extents the core reads.
            unsafe { tile_accs_avx2(lut, tables, q, tile, accs) };
            return;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx;
    tile_accs_portable(lut, tables, q, tile, accs);
}

/// Whether the executing host exposes `AVX2`.
#[inline]
fn avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The `AVX2` exact row accumulator for one 32-row tile: the table is duplicated into both 128-bit
/// lanes so one `vpshufb` resolves 32 rows; the two planes of `P + 508` are looked up separately
/// and recombined as `lo + 256·hi − 508` in `i16`, and the `d·S` lift to `i128` is
/// [`tile_accs_portable`]'s exact arithmetic. `AVX2` must be present, and `lut.codes` holds
/// `n_tiles·n_groups·32` bytes, `tables.lo`/`tables.hi` `n_groups·16`, `tables.xs` `n_groups` and
/// `lut.scales` `n_tiles·32·n_blocks`, per the caller's construction and the `tile < n_tiles`
/// guard. `O(n_in)` time (four `vpshufb` per 4-activation group plus one scalar `i128` product per
/// weight sub-block per row); `O(1)` space.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::too_many_lines,
    clippy::many_single_char_names,
    reason = "one `#[target_feature]` body keeps the eight-group lookup loop, the two nibble \
              planes and the four-sub-block reduction in one scope, so every intrinsic is reached \
              with `AVX2` enabled; splitting it would move intrinsics into functions without the \
              feature. The index names (`b` block, `g` group, `s` activation scale, `d` weight \
              scale) mirror the exact factorization they index"
)]
unsafe fn tile_accs_avx2(
    lut: &Pq2Lut,
    tables: &LutTables,
    q: &QuantAct,
    tile: usize,
    accs: &mut [i128; TILE_ROWS],
) {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi16, _mm256_and_si256, _mm256_broadcastsi128_si256,
        _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_loadu_si256,
        _mm256_set1_epi16, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
        _mm256_slli_epi16, _mm256_srli_epi16, _mm256_storeu_si256, _mm256_sub_epi16,
        _mm_loadu_si128,
    };
    let mut b = 0usize;
    while b < lut.n_blocks {
        let mut sb = 0usize;
        while sb < SUBS_PER_BLOCK {
            let gbase = (SUBS_PER_BLOCK * b + sb) * GROUPS_PER_SUB;
            let mut sum_hi: [__m256i; 2] = [_mm256_setzero_si256(); 2];
            let mut sum_lo: [__m256i; 2] = [_mm256_setzero_si256(); 2];
            let mut xsum: i16 = 0;
            let mut g8 = 0usize;
            while g8 < GROUPS_PER_SUB {
                let g = gbase + g8;
                xsum += tables.xs[g];
                let idx_base = (tile * lut.n_groups + g) * TILE_ROWS;
                let tab_base = g * 16;
                // SAFETY: the caller's construction gives `idx_base + 32 <= lut.codes.len()` and
                // `tab_base + 16 <= tables.lo.len()`; every load reads exactly those bytes, and
                // the packed `lo | hi << 4` nibbles are `<= 15`, so the high bit of each index
                // byte is clear and `vpshufb` selects inside the duplicated 16-entry table.
                unsafe {
                    let packed = _mm256_loadu_si256(lut.codes.as_ptr().add(idx_base).cast());
                    let m0f = _mm256_set1_epi8(0x0f);
                    let idx_lo = _mm256_and_si256(packed, m0f);
                    let idx_hi = _mm256_and_si256(_mm256_srli_epi16::<4>(packed), m0f);
                    let tab_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                        tables.lo.as_ptr().add(tab_base).cast(),
                    ));
                    let tab_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                        tables.hi.as_ptr().add(tab_base).cast(),
                    ));
                    let lo_at_lo = _mm256_shuffle_epi8(tab_lo, idx_lo);
                    let hi_at_lo = _mm256_shuffle_epi8(tab_hi, idx_lo);
                    let lo_at_hi = _mm256_shuffle_epi8(tab_lo, idx_hi);
                    let hi_at_hi = _mm256_shuffle_epi8(tab_hi, idx_hi);
                    let c508 = _mm256_set1_epi16(508);
                    let mut p_lo: [__m256i; 2] = [_mm256_setzero_si256(); 2];
                    let mut p_hi: [__m256i; 2] = [_mm256_setzero_si256(); 2];
                    let mut half = 0usize;
                    while half < 2 {
                        let lo_lane = if half == 0 {
                            _mm256_castsi256_si128(lo_at_lo)
                        } else {
                            _mm256_extracti128_si256(lo_at_lo, 1)
                        };
                        let hi_lane = if half == 0 {
                            _mm256_castsi256_si128(hi_at_lo)
                        } else {
                            _mm256_extracti128_si256(hi_at_lo, 1)
                        };
                        p_lo[half] = _mm256_sub_epi16(
                            _mm256_add_epi16(
                                _mm256_cvtepu8_epi16(lo_lane),
                                _mm256_slli_epi16::<8>(_mm256_cvtepu8_epi16(hi_lane)),
                            ),
                            c508,
                        );
                        let lo_lane = if half == 0 {
                            _mm256_castsi256_si128(lo_at_hi)
                        } else {
                            _mm256_extracti128_si256(lo_at_hi, 1)
                        };
                        let hi_lane = if half == 0 {
                            _mm256_castsi256_si128(hi_at_hi)
                        } else {
                            _mm256_extracti128_si256(hi_at_hi, 1)
                        };
                        p_hi[half] = _mm256_sub_epi16(
                            _mm256_add_epi16(
                                _mm256_cvtepu8_epi16(lo_lane),
                                _mm256_slli_epi16::<8>(_mm256_cvtepu8_epi16(hi_lane)),
                            ),
                            c508,
                        );
                        sum_lo[half] = _mm256_add_epi16(sum_lo[half], p_lo[half]);
                        sum_hi[half] = _mm256_add_epi16(sum_hi[half], p_hi[half]);
                        half += 1;
                    }
                }
                g8 += 1;
            }
            let xv = _mm256_set1_epi16(xsum);
            let mut buf = [0i16; TILE_ROWS];
            // SAFETY: `buf` holds 32 `i16`; the two stores write exactly the 32 bytes of the two
            // 256-bit registers into it.
            unsafe {
                let inner0 = _mm256_sub_epi16(
                    _mm256_add_epi16(_mm256_slli_epi16::<1>(sum_hi[0]), sum_lo[0]),
                    xv,
                );
                let inner1 = _mm256_sub_epi16(
                    _mm256_add_epi16(_mm256_slli_epi16::<1>(sum_hi[1]), sum_lo[1]),
                    xv,
                );
                _mm256_storeu_si256(buf.as_mut_ptr().cast(), inner0);
                _mm256_storeu_si256(buf.as_mut_ptr().add(16).cast(), inner1);
            }
            let s = i128::from(q.scales()[SUBS_PER_BLOCK * b + sb]);
            let row_base = tile * TILE_ROWS;
            let mut lane = 0usize;
            while lane < TILE_ROWS {
                let d = i128::from(lut.scales[(row_base + lane) * lut.n_blocks + b]);
                accs[lane] += d * s * i128::from(buf[lane]);
                lane += 1;
            }
            sb += 1;
        }
        b += 1;
    }
}

/// The exact lookup-table projection: `matvec_fixed(142, raw, n_out, n_in, x̂)` bit for bit.
///
/// `lut` was built by [`Pq2Lut::from_pq2_0`]; `q` is the quantized activation whose dequantization
/// is `x̂`, and the row dot is the table decomposition in the module docs. When [`lut_bound_ok`]
/// cannot prove the exact `i128` accumulation stays in range the call falls back to
/// [`matvec_fixed`] on `x̂`, so no refusal can differ: the same [`MatvecError`] values as
/// `matvec_fixed(142, raw, n_out, n_in, &x̂)`, including [`MatvecError::ShapeMismatch`],
/// [`MatvecError::OutputOverflow`] and [`MatvecError::Dequant`] for an activation with no Q32.32
/// value. `O(n_out·n_in)` time and space once the tables are built (one `i16` lookup sum per
/// 4-activation group per row instead of four `i128` products).
#[allow(
    clippy::cast_possible_truncation,
    reason = "the one `rounded as i64` narrowing is immediately preceded by an explicit \
              `i64::MIN`/`i64::MAX` range check"
)]
pub(crate) fn matvec_lut(lut: &Pq2Lut, q: &QuantAct) -> Result<Vec<i64>, MatvecError> {
    matvec_lut_with(lut, q, avx2())
}

/// [`matvec_lut`] with the `AVX2` choice made explicit, so the equality test can force the
/// portable twin as its oracle; errors as [`matvec_lut`].
fn matvec_lut_with(lut: &Pq2Lut, q: &QuantAct, avx: bool) -> Result<Vec<i64>, MatvecError> {
    let mut out: Vec<i64> = vec![0i64; lut.n_out];
    matvec_lut_rows_with(lut, q, 0, lut.n_out, &mut out, avx)?;
    Ok(out)
}

/// The row range `[row_lo, row_hi)` of [`matvec_lut`], written to `out` at `o = row_lo`.
///
/// The arithmetic, the per-call bound check and the checked fallback are [`matvec_lut_with`]'s; a
/// tile's rows outside the range are still accumulated (the tables are per-tile dense) but not
/// written, and `out.len()` must equal `row_hi - row_lo`. Errors as [`matvec_lut`], plus a
/// row-range or output-length refusal before any payload work.
#[allow(
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    reason = "the one `rounded as i64` narrowing is immediately preceded by an explicit \
              `i64::MIN`/`i64::MAX` range check; the index names (`t` tile, `b` block, `v` the \
              block's `i32` lanes, `d` the weight scale) mirror the factorization they index"
)]
fn matvec_lut_rows_with(
    lut: &Pq2Lut,
    q: &QuantAct,
    row_lo: usize,
    row_hi: usize,
    out: &mut [i64],
    avx: bool,
) -> Result<(), MatvecError> {
    if q.len() != lut.n_in {
        return Err(MatvecError::ShapeMismatch {
            x: q.len(),
            n_in: lut.n_in,
        });
    }
    if row_lo > row_hi || row_hi > lut.n_out {
        return Err(MatvecError::ShortPayload {
            need: row_hi,
            have: lut.n_out,
        });
    }
    if out.len() != row_hi - row_lo {
        return Err(MatvecError::RowLength {
            row: row_lo,
            weights: out.len(),
            n_in: row_hi - row_lo,
        });
    }
    if !q.dequantizes() {
        return Err(MatvecError::Dequant(EngineError::FixedOverflow));
    }
    if !lut_bound_ok(lut.n_in, q.scales()) {
        let Some(xhat) = q.dequant() else {
            return Err(MatvecError::Dequant(EngineError::FixedOverflow));
        };
        let row_bytes = lut.n_blocks * PQ2_0_BLOCK_BYTES;
        let bytes = &lut.raw[row_lo * row_bytes..row_hi * row_bytes];
        let v = matvec_fixed(
            quant::GGML_TYPE_PQ2_0,
            bytes,
            row_hi - row_lo,
            lut.n_in,
            &xhat,
        )?;
        out.copy_from_slice(&v);
        return Ok(());
    }
    let tables = build_tables(q.codes());
    // The `i32` lane is exact when its bound holds; otherwise the checked portable `i128` path
    // runs, and [`lut_bound_ok`] has already proved its exactness.
    let fast = lut_i64_bound_ok(q.scales());
    rows_core(lut, &tables, q, row_lo, row_hi, out, fast, avx)
}

/// The tile loop shared by [`matvec_lut_rows_with`] and [`matvec_lut_chunked`]; the caller has
/// checked the geometry and the `i128` bound, chosen the `i32`-lane path with `fast`, built
/// `tables` once and passed the `AVX2` decision.
///
/// With `fast` it also asks [`lut_row_i64_bound_ok`]: when that proved bound holds the row
/// accumulates in the two `i64` halves `W`/`L` of the exact product split (no `i128` in the
/// per-block loop), otherwise it keeps the current checked `i128` accumulation. Refuses
/// [`MatvecError::OutputOverflow`] when a rounded row does not fit `i64`, and
/// [`MatvecError::RowLength`] for a start off a tile boundary.
#[allow(
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    reason = "the `rounded as i64` is range-checked first; `(p >> 32) as i64` and \
              `(p & 0xffff_ffff) as i64` are only reached under [`lut_row_i64_bound_ok`], whose \
              bound makes `|p|/2^32 ≤ i64::MAX` and `p mod 2^32 ∈ [0, 2^32)`; the index names \
              mirror the factorization; the eight parameters are the kernel's full input"
)]
fn rows_core(
    lut: &Pq2Lut,
    tables: &LutTables,
    q: &QuantAct,
    row_lo: usize,
    row_hi: usize,
    out: &mut [i64],
    fast: bool,
    avx: bool,
) -> Result<(), MatvecError> {
    // A row range starts on a tile boundary: [`matvec_lut_chunked`]'s split guarantees it, and the
    // accumulator index is the row's offset from that boundary.
    if !row_lo.is_multiple_of(TILE_ROWS) {
        return Err(MatvecError::RowLength {
            row: row_lo,
            weights: row_lo % TILE_ROWS,
            n_in: TILE_ROWS,
        });
    }
    let first_tile = row_lo / TILE_ROWS;
    let last_tile = row_hi.div_ceil(TILE_ROWS);
    // One width decision per call: the proved bound is `O(n_in/32)` and identical for every tile.
    let two_i64 = fast && lut_row_i64_bound_ok(lut, q.scales());
    let mut t = first_tile;
    let mut ok = true;
    while t < last_tile && ok {
        // A fresh accumulator per tile: the `AVX2` core accumulates into it, and rows of one tile
        // must not carry the previous tile's sums.
        let mut accs = [0i128; TILE_ROWS];
        let base = t * TILE_ROWS;
        if two_i64 {
            // The `i64` width proved by `lut_row_i64_bound_ok`: `p = d·v = W_b·2^32 + L_b` with
            // `0 ≤ L_b < 2^32`, so `W = Σ_b W_b` and `L = Σ_b L_b` hold the exact row accumulator
            // `R = W·2^32 + L` without any `i128` in the per-block loop. The reconstruction is one
            // `i128` add-shift per lane, and `round_shift_32` then rounds it exactly as the
            // `i128` path does.
            let mut acc_w = [0i64; TILE_ROWS];
            let mut acc_r = [0i64; TILE_ROWS];
            // One scratch buffer per tile, not per block: [`block_inners`] overwrites all
            // `TILE_ROWS` lanes on every call, so the caller's initialization is only needed once.
            let mut v = [0i64; TILE_ROWS];
            let mut b = 0usize;
            while b < lut.n_blocks {
                block_inners(lut, tables, q, t, b, &mut v, avx);
                let mut lane = 0usize;
                while lane < TILE_ROWS {
                    let d = i128::from(lut.scales[(base + lane) * lut.n_blocks + b]);
                    let p = d * i128::from(v[lane]);
                    acc_w[lane] += (p >> 32) as i64;
                    acc_r[lane] += (p & 0xffff_ffff) as i64;
                    lane += 1;
                }
                b += 1;
            }
            let mut lane = 0usize;
            while lane < TILE_ROWS {
                accs[lane] = (i128::from(acc_w[lane]) << 32) + i128::from(acc_r[lane]);
                lane += 1;
            }
        } else if fast {
            let mut v = [0i64; TILE_ROWS];
            let mut b = 0usize;
            while b < lut.n_blocks {
                block_inners(lut, tables, q, t, b, &mut v, avx);
                let mut lane = 0usize;
                while lane < TILE_ROWS {
                    let d = i128::from(lut.scales[(base + lane) * lut.n_blocks + b]);
                    accs[lane] += d * i128::from(v[lane]);
                    lane += 1;
                }
                b += 1;
            }
        } else {
            tile_accs(lut, tables, q, t, &mut accs, avx);
        }
        let mut lane = 0usize;
        while lane < TILE_ROWS && ok {
            let o = base + lane;
            if o >= row_lo && o < row_hi {
                let rounded = round_shift_32(accs[lane]);
                if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
                    ok = false;
                } else {
                    out[o - row_lo] = rounded as i64;
                }
            }
            lane += 1;
        }
        t += 1;
    }
    if !ok {
        return Err(MatvecError::OutputOverflow);
    }
    Ok(())
}

/// One 32-row-aligned row range of `lut` projected against the Q32.32 activation `x`, run on the
/// [`Pool`].
///
/// `chunks` bounds the dispatch width (`<= 1` is the single [`matvec_lut`] call); the rows split
/// into `⌈n_tiles / tasks⌉`-tile chunks, so every chunk start is a [`TILE_ROWS`] boundary and the
/// last chunk is the ragged remainder. The chunks concatenate in row order to [`matvec_lut`] =
/// `matvec_fixed(142, raw, n_out, n_in, &dequant(quant_act(x)))` bit for bit. Errors as
/// [`matvec_lut`], plus the shape and dequant refusals, with the first chunk's refusal in row order.
/// `O(n_in)` for the one `quant_act` and shared table build, then `O(n_out·n_in)`; `O(n_out)` output.
pub(crate) fn matvec_lut_chunked(
    lut: &Pq2Lut,
    x: &[i64],
    chunks: usize,
) -> Result<Vec<i64>, MatvecError> {
    if x.len() != lut.n_in {
        return Err(MatvecError::ShapeMismatch {
            x: x.len(),
            n_in: lut.n_in,
        });
    }
    let q = crate::engine::chart::quant_act(x)
        .ok_or(MatvecError::Dequant(EngineError::FixedOverflow))?;
    let n_tiles = lut.n_tiles;
    // The activation tables are per-call, not per-chunk: one build shared by every task.
    if chunks <= 1 || n_tiles <= 1 || !lut_bound_ok(lut.n_in, q.scales()) {
        return matvec_lut_with(lut, &q, avx2());
    }
    let tables = build_tables(q.codes());
    let fast = lut_i64_bound_ok(q.scales());
    let avx = avx2();
    let tasks = chunks
        .min(2 * crate::engine::chart::pool::width_global())
        .min(n_tiles);
    let per_rows = n_tiles.div_ceil(tasks) * TILE_ROWS;
    let mut out: Vec<i64> = vec![0i64; lut.n_out];
    let first_err: crate::engine::chart::pool::FirstErr<usize, MatvecError> =
        crate::engine::chart::pool::FirstErr::new();
    crate::engine::chart::pool::chunks_mut_global(
        &mut out,
        per_rows,
        &MatvecChunkTask {
            lut,
            tables: &tables,
            q: &q,
            first_err: &first_err,
            fast,
            avx,
        },
    );
    match first_err.take() {
        Some((_, e)) => Err(e),
        None => Ok(out),
    }
}

/// Pool task context for [`matvec_lut_chunked`]'s row split: shared borrows and flags (FORM-PARALLEL D5).
struct MatvecChunkTask<'a> {
    lut: &'a Pq2Lut,
    tables: &'a LutTables,
    q: &'a QuantAct,
    first_err: &'a crate::engine::chart::pool::FirstErr<usize, MatvecError>,
    fast: bool,
    avx: bool,
}

impl ChunkTask<i64> for MatvecChunkTask<'_> {
    fn run(&self, c: usize, lo: usize, dst: &mut [i64]) {
        matvec_chunk_task(
            self.lut,
            self.tables,
            self.q,
            self.first_err,
            self.fast,
            self.avx,
            c,
            lo,
            dst,
        );
    }
}

/// The body of one [`MatvecChunkTask`] piece: rows `[lo, lo + dst.len())`; a failure is recorded
/// under the chunk index `c` so the lowest-index error is the one returned.
#[allow(
    clippy::too_many_arguments,
    reason = "the lut, its tables, the activation, the error slot, the two kernel flags, and the chunk index, start and piece"
)]
fn matvec_chunk_task(
    lut: &Pq2Lut,
    tables: &LutTables,
    q: &QuantAct,
    first_err: &crate::engine::chart::pool::FirstErr<usize, MatvecError>,
    fast: bool,
    avx: bool,
    c: usize,
    lo: usize,
    dst: &mut [i64],
) {
    if let Err(e) = rows_core(lut, tables, q, lo, lo + dst.len(), dst, fast, avx) {
        first_err.record(c, e);
    }
}

/// [`matvec_lut_chunked`]'s single-row-range core; errors as [`matvec_lut_rows_with`].
pub(crate) fn matvec_lut_rows(
    lut: &Pq2Lut,
    q: &QuantAct,
    row_lo: usize,
    row_hi: usize,
    out: &mut [i64],
) -> Result<(), MatvecError> {
    matvec_lut_rows_with(lut, q, row_lo, row_hi, out, avx2())
}

/// One `PQ2_0` projection's per-dispatch state: the activation's lookup tables and the kernel
/// flags. [`PreparedLut::rows`] runs [`rows_core`] over it with the table and activation the
/// caller passes, so the chart N-projection dispatch builds the tables once per call rather than
/// once per task. It owns its data and borrows nothing.
pub(crate) struct PreparedLut {
    tables: LutTables,
    fast: bool,
    avx: bool,
}

impl PreparedLut {
    /// Write rows `[lo, hi)` of `lut` into `out` (`out.len() == hi - lo`), as [`Pq2Lut::rows`]
    /// does with the activation's tables already built. `q` is the activation this value was
    /// prepared from.
    ///
    /// # Errors
    ///
    /// The refusals of [`rows_core`] for `[lo, hi)`.
    pub(crate) fn rows(
        &self,
        lut: &Pq2Lut,
        q: &QuantAct,
        lo: usize,
        hi: usize,
        out: &mut [i64],
    ) -> Result<(), MatvecError> {
        rows_core(lut, &self.tables, q, lo, hi, out, self.fast, self.avx)
    }
}

/// A cached `PQ2_0` table as a row kernel of the chart N-projection dispatch
/// ([`crate::engine::chart::ffn::ProjSource::Rows`]). A chunk start must be a [`LUT_ROW_ALIGN`]
/// boundary.
impl Pq2Lut {
    /// Build this projection's per-dispatch kernel for `q`, or `None` when the unprepared
    /// [`Self::rows`] path should be used (an activation with no Q32.32 value, or a call whose exact
    /// bound needs the reference fallback).
    pub(crate) fn prepare(&self, q: &QuantAct) -> Option<PreparedLut> {
        // The reference fallback and the no-Q32 refusal keep the per-task [`Self::rows`] path; the
        // tables for the fast path are the call's, so they are built once here — across the pool by
        // [`build_tables_pooled`], off the caller's serial path — not once per task.
        if !q.dequantizes() || !lut_bound_ok(self.n_in, q.scales()) {
            return None;
        }
        Some(PreparedLut {
            tables: build_tables_pooled(q.codes()),
            fast: lut_i64_bound_ok(q.scales()),
            avx: avx2(),
        })
    }

    /// Write rows `[lo, hi)` of this projection into `out` (`out.len() == hi - lo`), reading the
    /// shared quantized activation `q`. Rows are independent, so the write is a pure function of `q`
    /// and the weights.
    ///
    /// # Errors
    ///
    /// This projection's own refusal for `[lo, hi)`.
    pub(crate) fn rows(
        &self,
        q: &QuantAct,
        lo: usize,
        hi: usize,
        out: &mut [i64],
    ) -> Result<(), MatvecError> {
        matvec_lut_rows(self, q, lo, hi, out)
    }
}

/// The `PQ2_0`-projection entry point of the `Q8Act` decode path: the cached lookup-table kernel
/// when `ggml_type` is `PQ2_0`, else [`crate::engine::chart::ffn::matvec_q8act_chunked`]. The
/// table is the [`Engine`](crate::engine::loader::Engine)'s open-time slot of directory index
/// `idx`, so the `O(n_out·n_in)` repack runs once per tensor per load and non-`PQ2_0` families
/// keep the chart kernel. Errors as those two kernels.
#[allow(
    clippy::too_many_arguments,
    reason = "the engine, tensor identity, payload and shape, activation and dispatch width \
              are the projection's full input"
)]
pub(crate) fn q8act_projection(
    engine: &crate::engine::loader::Engine,
    idx: usize,
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    x: &[i64],
    chunks: usize,
) -> Result<Vec<i64>, MatvecError> {
    if ggml_type == quant::GGML_TYPE_PQ2_0 {
        let lut = engine.pq2_lut(idx)?;
        return matvec_lut_chunked(lut, x, chunks);
    }
    crate::engine::chart::ffn::matvec_q8act_chunked(ggml_type, raw, n_out, n_in, x, chunks)
}

#[cfg(test)]
#[path = "lut_tests.rs"]
mod tests;
