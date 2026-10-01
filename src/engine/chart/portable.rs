//! The portable lane-loop twins of the AVX2 cores and of the fast row kernels: the no-AVX2
//! fallback and the equality oracle for `x86`. Every function here is a plain scalar loop whose
//! result is the exact value the corresponding `#[target_feature(enable = "avx2")]` core returns;
//! that equality is `tested` (`chart::tests`), not extracted.
use crate::format::quant::QK_K;

use super::ACT_QK;

/// Sentinel for a non-finite `f16` pattern in [`F16_FIXED_TABLE`]: an odd value.
///
/// No finite pattern attains it. A subnormal decodes to `mant << 8` and a normal to
/// `(1024 + mant) << (exp + 7)` with `exp ≥ 1`, so every decoded magnitude is a multiple of `2^8`
/// and its low bit is clear.
const F16_NONFINITE: i64 = 1;

/// The exact Q32.32 decode of every `f16` bit pattern, indexed by the pattern; [`F16_NONFINITE`]
/// marks the `Inf`/`NaN` exponent `0x1f`.
///
/// The row kernels read one entry per weight block. The branch decomposition it replaces (an
/// exponent test, a subnormal test and a byte reassembly of two `u16` halves) is about a third of
/// a scalar block's work in the previous build's inner loops.
static F16_FIXED_TABLE: [i64; 65536] = build_f16_fixed_table();

/// Build [`F16_FIXED_TABLE`]: normal `(1024 + mant) << (exp + 7)`, subnormal `mant << 8`, the sign
/// applied by negation, and [`F16_NONFINITE`] for `exp == 0x1f`. The exhaustive equality with
/// `crate::engine::f16_to_fixed` is `tested` (`fast_f16_decode_matches_reference_for_all_patterns`).
#[allow(
    clippy::large_stack_arrays,
    reason = "the array is the initializer of a `static`; it is materialized in const evaluation, \
              never on a runtime stack"
)]
const fn build_f16_fixed_table() -> [i64; 65536] {
    let mut table = [F16_NONFINITE; 65536];
    let mut h: u16 = 0;
    loop {
        let e = (h >> 10) & 0x1f;
        if e != 0x1f {
            let m = (h & 0x3ff) as i64;
            let mag = if e == 0 {
                m << 8
            } else {
                (m + 1024) << (e as u32 + 7)
            };
            table[h as usize] = if h & 0x8000 == 0 { mag } else { -mag };
        }
        if h == u16::MAX {
            break;
        }
        h += 1;
    }
    table
}

/// Exact `f16` → Q32.32 decode by table lookup, or `None` for an `Inf`/`NaN` exponent.
///
/// Identical to [`crate::engine::f16_to_fixed`] on every finite pattern; the `None` case routes the
/// caller to the reference and its `NonFiniteScale` refusal. The exhaustive equality with the
/// reference is `tested`.
#[inline]
pub(super) fn f16_to_fixed_fast(h: u16) -> Option<i64> {
    let v = F16_FIXED_TABLE[h as usize];
    if v & 1 == 0 {
        Some(v)
    } else {
        None
    }
}

/// [`f16_to_fixed_fast`] without the per-block branch: the raw table entry, whose low bit is
/// [`F16_NONFINITE`]'s.
///
/// A row kernel that ORs each decoded block scale into one accumulator and tests its low bit once
/// after the block loop refuses exactly the patterns [`f16_to_fixed_fast`] refuses — every finite
/// decode is a multiple of `2^8` — while paying one instruction per block instead of a test and a
/// branch.
#[inline]
pub(super) fn f16_to_fixed_fast_raw(h: u16) -> i64 {
    F16_FIXED_TABLE[h as usize]
}

/// Portable exact `Σ codes[l]·xq[l]` over `codes.len()` lanes.
///
/// `|Σ| ≤ 32·128·127 < 2^31` for every group the kernel forms (at most 32 lanes), so the `i32`
/// accumulator cannot leave range. The portable twin of the AVX2 cores and the test oracle.
///
/// # Complexity
///
/// `O(n)` time, `O(1)` extra space.
pub(super) fn portable_i8_dot(codes: &[i8], xq: &[i8]) -> i32 {
    let mut acc: i32 = 0;
    let mut lane = 0usize;
    while lane < codes.len() {
        acc += i32::from(codes[lane]) * i32::from(xq[lane]);
        lane += 1;
    }
    acc
}

/// Exact `Σ_{l<n} (codes[l] as i8)·xq[l]` for raw `Q8_0` code bytes: the portable twin of
/// [`super::x86::dot32_u8_i8`] and the no-AVX2 path.
///
/// # Complexity
///
/// `O(n)` time, `O(1)` extra space.
pub(super) fn portable_u8_i8_dot(codes: &[u8], xq: &[i8]) -> i32 {
    let mut acc: i32 = 0;
    let mut l = 0usize;
    while l < codes.len() {
        acc += i32::from(i8::from_ne_bytes([codes[l]])) * i32::from(xq[l]);
        l += 1;
    }
    acc
}

/// Exact `Σ_{l<32} c_l·xq[l]` for one `Q5_0` block from its raw `qh`/`qs` bytes: the portable twin
/// of [`super::x86::q5_0_dot32`] and the no-AVX2 path. The lane order matches
/// [`quant::q5_0_codes`](crate::format::quant::q5_0_codes).
///
/// # Complexity
///
/// `O(32)` time, `O(1)` extra space.
pub(super) fn portable_q5_0_dot(qh: &[u8], qs: &[u8], xq: &[i8]) -> i32 {
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

/// Exact `(Σ low_l·xq_lo[l], Σ high_l·xq_hi[l])` for one 32-byte `Q4_K` `qs` window: the portable
/// twin of [`super::x86::q4_k_window_dot`].
///
/// # Complexity
///
/// `O(32)` time, `O(1)` extra space.
pub(super) fn portable_q4_k_window_dots(qs: &[u8], xq_lo: &[i8], xq_hi: &[i8]) -> (i32, i32) {
    let mut lo = 0i32;
    let mut hi = 0i32;
    let mut l = 0usize;
    while l < 32 {
        lo += i32::from(qs[l] & 0x0F) * i32::from(xq_lo[l]);
        hi += i32::from(qs[l] >> 4) * i32::from(xq_hi[l]);
        l += 1;
    }
    (lo, hi)
}

/// Per-32-lane `Σ xq` of a `QuantAct`, computed once per call and reused by every row: the
/// `Q4_K` affine rows need `Σ xq` for each 32-lane group.
///
/// # Complexity
///
/// `O(n)` time and `O(n/32)` space.
pub(super) fn block_xsums(codes: &[i8]) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::with_capacity(codes.len() / ACT_QK);
    let mut base = 0usize;
    while base + ACT_QK <= codes.len() {
        let mut acc = 0i32;
        let mut l = 0usize;
        while l < ACT_QK {
            acc += i32::from(codes[base + l]);
            l += 1;
        }
        out.push(acc);
        base += ACT_QK;
    }
    out
}

/// Exact `Σ codes[l]·xq[l]` over 32 signed lanes, AVX2 when the host has it.
pub(super) fn dot32_signed(codes: &[i8], xq: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if super::avx2() {
            // SAFETY: AVX2 is present and both slices hold the 32 lanes the core loads.
            return unsafe { super::x86::dot32_i8(codes, xq) };
        }
    }
    portable_i8_dot(codes, xq)
}

/// Exact `Σ codes[l]·xq[l]` over 16 signed lanes, AVX2 when the host has it.
pub(super) fn dot16_signed(codes: &[i8], xq: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if super::avx2() {
            // SAFETY: AVX2 is present and both slices hold the 16 lanes the core loads.
            return unsafe { super::x86::dot16_i8(codes, xq) };
        }
    }
    portable_i8_dot(codes, xq)
}

/// Exact `Σ_{l<take} codes[cbase+l]·xq[xbase+l]` over one group window.
///
/// # Complexity
///
/// `O(take)` time, `O(1)` extra space.
pub(super) fn group_dot(codes: &[i8], cbase: usize, xq: &[i8], xbase: usize, take: usize) -> i32 {
    if take == 0 {
        return 0;
    }
    let code_window = &codes[cbase..cbase + take];
    let x_window = &xq[xbase..xbase + take];
    if take == 32 {
        dot32_signed(code_window, x_window)
    } else if take == 16 {
        dot16_signed(code_window, x_window)
    } else {
        portable_i8_dot(code_window, x_window)
    }
}

/// Exact `Σ_{l<take} xq[xbase+l]` in `i32` (`|·| ≤ 32·127`).
///
/// # Complexity
///
/// `O(take)` time, `O(1)` extra space.
pub(super) fn group_xsum(xq: &[i8], xbase: usize, take: usize) -> i32 {
    let mut acc: i32 = 0;
    let mut lane = 0usize;
    while lane < take {
        acc += i32::from(xq[xbase + lane]);
        lane += 1;
    }
    acc
}

/// Portable `Q8_0` fast row (no AVX2).
///
/// # Complexity
///
/// `O(n_in)` time in exact integer multiply-adds, `O(1)` extra space.
#[allow(
    dead_code,
    reason = "oracle and regression parity targets for chart::tests"
)]
pub(super) fn q8_0_row_portable(
    row: &[u8],
    blocks: usize,
    codes: &[i8],
    scales: &[i64],
) -> Option<i128> {
    super::q8act_accumulate_row_portable::<super::Q8_0Decoder>(
        row,
        blocks * ACT_QK,
        codes,
        scales,
        &[],
    )
}

/// Portable `Q5_0` fast row (no AVX2).
///
/// # Complexity
///
/// `O(n_in)` time in exact integer multiply-adds, `O(1)` extra space.
#[allow(
    dead_code,
    reason = "oracle and regression parity targets for chart::tests"
)]
pub(super) fn q5_0_row_portable(
    row: &[u8],
    blocks: usize,
    codes: &[i8],
    scales: &[i64],
) -> Option<i128> {
    super::q8act_accumulate_row_portable::<super::Q5_0Decoder>(
        row,
        blocks * ACT_QK,
        codes,
        scales,
        &[],
    )
}

/// Portable `Q4_K` fast row.
///
/// # Complexity
///
/// `O(n_in)` time, `O(1)` extra space.
#[allow(
    dead_code,
    reason = "oracle and regression parity targets for chart::tests"
)]
pub(super) fn q4_k_row_portable(
    row: &[u8],
    pairs: usize,
    codes: &[i8],
    act_scales: &[i64],
    xsums: &[i32],
) -> Option<i128> {
    super::q8act_accumulate_row_portable::<super::Q4_KDecoder>(
        row,
        pairs * 64,
        codes,
        act_scales,
        xsums,
    )
}

/// Portable `Q6_K` fast row.
///
/// # Complexity
///
/// `O(n_in)` time, `O(QK_K)` extra space per block.
#[allow(
    dead_code,
    reason = "oracle and regression parity targets for chart::tests"
)]
pub(super) fn q6_k_row_portable(
    row: &[u8],
    blocks: usize,
    codes: &[i8],
    act_scales: &[i64],
) -> Option<i128> {
    super::q8act_accumulate_row_portable::<super::Q6_KDecoder>(
        row,
        blocks * QK_K,
        codes,
        act_scales,
        &[],
    )
}
