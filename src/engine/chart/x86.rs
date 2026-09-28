//! AVX2 inner-product cores for the integer chart dot, ported from `fold`
//! (`fold/bedrock/src/serve/model/rowdot.rs`: `row_dot_q4_avx2`'s `vpmaddubsw`/`vpmaddwd` byte
//! plane, `row_dot_q2_avx2`'s packed-plane decode, `q1.rs`'s exact integer group).
//!
//! Only the integer group dot is ported; `fold`'s float epilogue is not. Every core returns the
//! same exact `i32` as the portable lane loop in the parent module. `vpmaddubsw` widens each pair
//! of products into a `i16`, and a 32-lane group sums at most `2·128·127 = 32512 < 2^15`, so the
//! widening add cannot saturate; there is no float and no reduction-order hazard.
//!
//! Each core is `#[target_feature(enable = "avx2")]` and is reached only behind
//! `is_x86_feature_detected!("avx2")` (see [`avx2`]); the parent module holds the portable twin
//! that is the fallback and the test oracle. The intrinsics are not extracted: their equality with
//! the portable twin is `tested` (trust boundary, the engine-throughput note §W2).
//!
//! The `*_rows2_*` cores are the cross-row tiled row bodies: two rows per pass share each weight
//! block's activation code vector and per-block activation scale `S_b`. Every row keeps its own
//! per-block fold order and its own single `round_shift_32`, so the outputs are the exact values
//! the single-row cores return.

use crate::format::quant::{
    self, Q4_K_BLOCK_BYTES, Q4_K_SCALE_SIZE, Q5_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES, QK_K,
};

use super::portable::f16_to_fixed_fast_raw;
use super::ACT_QK;
use crate::engine::value::round_shift_32;

use std::arch::x86_64::{
    __m128i, __m256i, _mm256_abs_epi8, _mm256_and_si256, _mm256_andnot_si256,
    _mm256_castsi256_si128, _mm256_cmpeq_epi8, _mm256_extracti128_si256, _mm256_hadd_epi32,
    _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_or_si256,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_set1_epi8, _mm256_set_m128i,
    _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_sign_epi8,
    _mm256_slli_epi16, _mm256_srli_epi16, _mm256_storeu_si256, _mm256_sub_epi8, _mm_abs_epi8,
    _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_madd_epi16, _mm_maddubs_epi16,
    _mm_set1_epi16, _mm_shuffle_epi32, _mm_sign_epi8, _mm_srli_epi16, _mm_storeu_si128,
    _mm_sub_epi32,
};

/// Whether the executing host exposes AVX2.
#[inline]
#[must_use]
pub(super) fn avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// Whether the executing host exposes AVX-VNNI (VEX-encoded `vpdpbusd`).
#[inline]
#[must_use]
pub(super) fn avxvnni() -> bool {
    static VNNI: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
    match VNNI.load(core::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let has = std::is_x86_feature_detected!("avxvnni");
            VNNI.store(
                if has { 1 } else { 2 },
                core::sync::atomic::Ordering::Relaxed,
            );
            has
        }
    }
}

/// VEX-encoded `vpdpbusd` accumulating four `u8·i8` products into each `i32` dword.
///
/// Inlined via asm under `#[target_feature(enable = "avx2")]` to avoid rustc/LLVM inlining
/// barriers between AVX2 callers and AVX-VNNI callees, and to maintain compatibility across
/// toolchain versions without clippy's `incompatible_msrv` warning.
///
/// # Safety
///
/// Caller must ensure AVX-VNNI is supported at runtime.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dpbusd_vex(mut acc: __m256i, u: __m256i, x: __m256i) -> __m256i {
    core::arch::asm!(
        "{{vex}} vpdpbusd {acc}, {u}, {x}",
        acc = inout(ymm_reg) acc,
        u = in(ymm_reg) u,
        x = in(ymm_reg) x,
        options(pure, nomem, nostack)
    );
    acc
}

/// The `f16` scale word at `off` in an AVX2 row loop, read as one unaligned `u16`.
///
/// The AVX2 lane is `x86_64`-only and therefore little-endian, so this equals
/// `u16::from_le_bytes([row[off], row[off + 1]])` while paying one load instead of two byte
/// loads, a shift and an `or`.
///
/// # Safety
///
/// `base` must point at 2 readable bytes at `off`; the caller's length checks establish that.
#[inline]
pub(super) unsafe fn scale_word(base: *const u8, off: usize) -> u16 {
    // SAFETY: the caller guarantees the two bytes at `off` are readable.
    unsafe { core::ptr::read_unaligned(base.add(off).cast::<u16>()) }
}

/// `quant::q4_k_scale_min` on a slice: the `j`-th of the eight packed 6-bit `(scale, min)` pairs.
///
/// `#[inline]` so the `#[target_feature(enable = "avx2")]` row loops keep it in the pair loop.
#[inline]
#[must_use]
pub(super) fn q4_k_scale_min_at(sc: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (sc[j] & 63, sc[j + 4] & 63)
    } else {
        (
            (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4),
            (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4),
        )
    }
}

/// Horizontal sum of the four `i32` lanes of `v`.
#[inline]
unsafe fn hsum128(v: __m128i) -> i32 {
    // SAFETY: register arithmetic only; no memory and no lanes outside `v`.
    unsafe {
        // [a+b, b+a, c+d, d+c] then add the two 64-bit halves.
        let s = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xB1));
        let t = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        _mm_cvtsi128_si32(t)
    }
}

/// Horizontal sum of the eight `i32` lanes of `v`.
#[inline]
unsafe fn hsum256(v: __m256i) -> i32 {
    // SAFETY: register arithmetic only; the lane extraction reads no memory.
    unsafe {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        hsum128(_mm_add_epi32(lo, hi))
    }
}

/// The four exact sums of the eight `i32` lanes of `p0..p3`, as `[Σp0, Σp1, Σp2, Σp3]`.
///
/// A three-`vphaddd` tree: the first `hadd` pairs the four-lane halves of `p0`/`p1` and `p2`/`p3`,
/// the second folds those pairs, and the third re-pairs the two 128-bit halves so that
/// `lo128 + hi128` holds the four totals in block order. Five instructions reduce four blocks; a
/// `hsum256` tree costs seven each. Every intermediate is exact `i32`: a lane vector sums at most
/// `32·128·127 < 2^21`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn reduce4(p0: __m256i, p1: __m256i, p2: __m256i, p3: __m256i) -> __m128i {
    // SAFETY: register arithmetic only; no memory and no lanes outside the four vectors.
    let h01 = _mm256_hadd_epi32(p0, p1);
    let h23 = _mm256_hadd_epi32(p2, p3);
    let h = _mm256_hadd_epi32(h01, h23);
    _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1))
}

/// The eight `i32` lane-pair sums of `c·x`: the register twin of the load in [`dot32_lanes_ptr`].
///
/// On AVX-VNNI hardware, `_mm256_dpbusd_avx_epi32` directly accumulates four `u8·i8` products
/// into each `i32` dword, replacing the `vpmaddubsw` + `vpmaddwd` pair.
///
/// **Equivalence proof for exact i32 semantics:**
/// `vpmaddubsw` computes `c0·x0 + c1·x1` as signed saturated `i16`.
/// Here `|c| <= 128` (or `<= 32` for `Q6_K`, `<= 16` for `Q5_0`) and `|x| <= 127`.
/// For any format:
/// - `Q8_0`: `|c| <= 128`, `|x| <= 127` -> `2 * 128 * 127 = 32512 < 32767 = 2^15 - 1`.
/// - `Q6_K`: `|c| <= 32`, `|x| <= 127` -> `2 * 32 * 127 = 8128 < 32767`.
/// - `Q5_0`: `|c| <= 16`, `|x| <= 127` -> `2 * 16 * 127 = 4064 < 32767`.
///
/// In all formats, `vpmaddubsw` never saturates. `vpmaddwd(pair, 1)` then sums adjacent pairs
/// into `i32`: `(c0·x0 + c1·x1) + (c2·x2 + c3·x3) = sum_{k=0..3} c_k · x_k`.
/// `vpdpbusd(0, abs(c), sign(x, c))` computes `0 + sum_{k=0..3} c_k · x_k` directly in `i32`.
/// Therefore, `vpdpbusd` and the `vpmaddubsw` + `vpmaddwd` chain produce identically bit-exact results.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot32_lanes(c: __m256i, x: __m256i) -> __m256i {
    // SAFETY: register arithmetic only; no memory and no lanes outside the two vectors.
    if avxvnni() {
        // SAFETY: AVX-VNNI detected at runtime.
        unsafe { dot32_lanes_vnni(c, x) }
    } else {
        unsafe { dot32_lanes_avx2(c, x) }
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot32_lanes_avx2(c: __m256i, x: __m256i) -> __m256i {
    _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_abs_epi8(c), _mm256_sign_epi8(x, c)),
        _mm256_set1_epi16(1),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot32_lanes_vnni(c: __m256i, x: __m256i) -> __m256i {
    dpbusd_vex(
        _mm256_setzero_si256(),
        _mm256_abs_epi8(c),
        _mm256_sign_epi8(x, c),
    )
}

/// The eight `i32` partial sums `Σ_{2k,2k+1} c·xq` of one 32-lane `vpmaddubsw`/`vpmaddwd` pass,
/// before any horizontal reduction. `#[inline]` so callers keep the loads and multiply fused.
///
/// # Safety
///
/// `codes` must point at 32 readable bytes and `xq` at 32 readable lanes; the caller must have
/// AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot32_lanes_ptr(codes: *const u8, xq: *const i8) -> __m256i {
    // SAFETY: the caller guarantees the two loads' extents and AVX2.
    unsafe {
        let c = _mm256_loadu_si256(codes.cast());
        let x = _mm256_loadu_si256(xq.cast());
        dot32_lanes(c, x)
    }
}

/// Two `vphaddd` levels over four lane vectors, splitting each into its low four-lane sum `lo` and
/// high four-lane sum `hi`: the eight results in the order
/// `[lo p1, lo p2, lo p3, lo p4, hi p1, hi p2, hi p3, hi p4]`. Three instructions replace eight
/// 128-bit horizontal sums. Every intermediate is exact `i32`: a half sums at most
/// `16·128·127 < 2^21`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn reduce4_split(p1: __m256i, p2: __m256i, p3: __m256i, p4: __m256i) -> __m256i {
    // SAFETY: register arithmetic only; no memory and no lanes outside the four vectors.
    _mm256_hadd_epi32(_mm256_hadd_epi32(p1, p2), _mm256_hadd_epi32(p3, p4))
}

/// The four exact 32-lane byte dots `Σ_{l<32} codes_i[l]·xq_i[l]` for four consecutive blocks:
/// four [`dot32_lanes_ptr`] passes reduced by one [`reduce4`] tree, returning `[dot0, dot1, dot2,
/// dot3]`. Block `i`'s 32 code bytes are at `codes + i·stride` (stride `Q8_0_BLOCK_BYTES` when a
/// per-block scale word separates them) and its activations at `xq + 32i`.
///
/// # Safety
///
/// `codes + 3·stride` must point at 32 readable bytes and `xq` at 128 readable lanes; the caller
/// must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn dot32x4_ptr(codes: *const u8, stride: usize, xq: *const i8) -> [i32; 4] {
    // SAFETY: the caller's four blocks give each pass its 32 bytes and 32 lanes.
    unsafe {
        let c0 = _mm256_loadu_si256(codes.cast());
        let c1 = _mm256_loadu_si256(codes.add(stride).cast());
        let c2 = _mm256_loadu_si256(codes.add(2 * stride).cast());
        let c3 = _mm256_loadu_si256(codes.add(3 * stride).cast());
        let x0 = _mm256_loadu_si256(xq.cast());
        let x1 = _mm256_loadu_si256(xq.add(32).cast());
        let x2 = _mm256_loadu_si256(xq.add(64).cast());
        let x3 = _mm256_loadu_si256(xq.add(96).cast());
        let r = if avxvnni() {
            reduce4(
                dot32_lanes_vnni(c0, x0),
                dot32_lanes_vnni(c1, x1),
                dot32_lanes_vnni(c2, x2),
                dot32_lanes_vnni(c3, x3),
            )
        } else {
            reduce4(
                dot32_lanes_avx2(c0, x0),
                dot32_lanes_avx2(c1, x1),
                dot32_lanes_avx2(c2, x2),
                dot32_lanes_avx2(c3, x3),
            )
        };
        let mut out = [0i32; 4];
        // SAFETY: `out` is four `i32` = 16 bytes, exactly one 128-bit store.
        _mm_storeu_si128(out.as_mut_ptr().cast(), r);
        out
    }
}

/// Four exact 32-lane byte dots for two rows against two blocks: lane `(r, b)` is row `r`'s codes
/// at `codes + r·stride + b·block_stride` against activation vector `x_b`. The result is
/// `[dot(0,0), dot(0,1), dot(1,0), dot(1,1)]`.
///
/// # Safety
///
/// `codes + stride + block_stride` must point at 32 readable bytes; the caller must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot32x2x2_rows(
    codes: *const u8,
    stride: usize,
    block_stride: usize,
    x0: __m256i,
    x1: __m256i,
) -> __m128i {
    // SAFETY: the caller's two rows and two blocks give each pass its 32 code bytes.
    unsafe {
        reduce4(
            dot32_lanes(_mm256_loadu_si256(codes.cast()), x0),
            dot32_lanes(_mm256_loadu_si256(codes.add(block_stride).cast()), x1),
            dot32_lanes(_mm256_loadu_si256(codes.add(stride).cast()), x0),
            dot32_lanes(
                _mm256_loadu_si256(codes.add(stride + block_stride).cast()),
                x1,
            ),
        )
    }
}

/// Exact `Σ_{l<32} codes[l]·xq[l]` for signed `i8` codes and int8 activations `xq`.
///
/// `abs_epi8` makes the code magnitudes the unsigned `vpmaddubsw` operand and `sign_epi8` folds the
/// code sign into the activation, so the pair sum is over the original products.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn dot32_i8(codes: &[i8], xq: &[i8]) -> i32 {
    // SAFETY: the caller guarantees at least 32 lanes in each slice; the two unaligned 256-bit
    // loads read exactly those lanes, and the target-feature guarantee makes the intrinsics valid.
    unsafe {
        let code_bytes = _mm256_loadu_si256(codes.as_ptr().cast());
        let act_bytes = _mm256_loadu_si256(xq.as_ptr().cast());
        let mag_bytes = _mm256_abs_epi8(code_bytes);
        let signed_act = _mm256_sign_epi8(act_bytes, code_bytes);
        let pair_sums = _mm256_maddubs_epi16(mag_bytes, signed_act);
        hsum256(_mm256_madd_epi16(pair_sums, _mm256_set1_epi16(1)))
    }
}

/// Exact `Σ_{l<16} codes[l]·xq[l]` for signed `i8` codes and int8 activations `xq`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn dot16_i8(codes: &[i8], xq: &[i8]) -> i32 {
    // SAFETY: the caller guarantees at least 16 lanes in each slice; the two unaligned 128-bit
    // loads read exactly those lanes.
    unsafe {
        let code_bytes = _mm_loadu_si128(codes.as_ptr().cast());
        let act_bytes = _mm_loadu_si128(xq.as_ptr().cast());
        let mag_bytes = _mm_abs_epi8(code_bytes);
        let signed_act = _mm_sign_epi8(act_bytes, code_bytes);
        let pair_sums = _mm_maddubs_epi16(mag_bytes, signed_act);
        hsum128(_mm_madd_epi16(pair_sums, _mm_set1_epi16(1)))
    }
}

/// Exact `(Σ_l low_l·xq_lo[l], Σ_l high_l·xq_hi[l])` for one 32-byte `Q4_K` `qs` window, where
/// `low_l`/`high_l` are the low/high nibbles of `qs[l]`.
///
/// The two nibble planes carry independent weights (`q4_k_codes`: low nibble lane `l`, high nibble
/// lane `l+32` of the 64-weight window), so each plane is multiplied by its own activation half.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn q4_k_window_dot(qs: &[u8], xq_lo: &[i8], xq_hi: &[i8]) -> (i32, i32) {
    // SAFETY: the caller guarantees 32 `qs` bytes and 32 lanes in each activation window.
    unsafe { q4_k_window_dot_ptr(qs.as_ptr(), xq_lo.as_ptr(), xq_hi.as_ptr()) }
}

/// [`q4_k_window_dot`] from raw pointers. `#[inline]` so a
/// `#[target_feature(enable = "avx2")]` caller keeps it in the row loop.
///
/// # Safety
///
/// `qs` must point at 32 readable bytes and `xq_lo`/`xq_hi` at 32 readable lanes each; the caller
/// must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn q4_k_window_dot_ptr(
    qs: *const u8,
    xq_lo: *const i8,
    xq_hi: *const i8,
) -> (i32, i32) {
    // SAFETY: the caller guarantees the four loads' extents and AVX2.
    unsafe {
        let bytes = _mm256_loadu_si256(qs.cast());
        let nib = _mm256_set1_epi8(0x0F);
        let lo = _mm256_and_si256(bytes, nib);
        let hi = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), nib);
        let xl = _mm256_loadu_si256(xq_lo.cast());
        let xh = _mm256_loadu_si256(xq_hi.cast());
        let ones = _mm256_set1_epi16(1);
        let pl = _mm256_maddubs_epi16(lo, xl);
        let ph = _mm256_maddubs_epi16(hi, xh);
        (
            hsum256(_mm256_madd_epi16(pl, ones)),
            hsum256(_mm256_madd_epi16(ph, ones)),
        )
    }
}

/// The eight exact 16-lane dots of one full 256-weight `Q4_K` block: its four 32-byte `qs` windows,
/// each contributing a low-nibble and a high-nibble dot, in the order
/// `[w0lo, w0hi, w1lo, w1hi, w2lo, w2hi, w3lo, w3hi]`.
///
/// `qs` points at window 0 (the first `qs` byte of the block) and `xq` at the block's first
/// activation group; window `i`'s low/high planes read `xq + 64i` and `xq + 64i + 32`. Two
/// [`reduce4`] trees replace eight 256-bit horizontal sums.
///
/// # Safety
///
/// `qs` must point at 128 readable bytes and `xq` at 256 readable lanes; the caller must have
/// AVX2.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn q4_k_block_dots(qs: *const u8, xq: *const i8) -> [i32; 8] {
    // SAFETY: the caller guarantees the eight loads' extents and AVX2.
    unsafe {
        let nib = _mm256_set1_epi8(0x0F);
        let b0 = _mm256_loadu_si256(qs.cast());
        let b1 = _mm256_loadu_si256(qs.add(32).cast());
        let b2 = _mm256_loadu_si256(qs.add(64).cast());
        let b3 = _mm256_loadu_si256(qs.add(96).cast());
        let q0l = _mm256_and_si256(b0, nib);
        let q1l = _mm256_and_si256(b1, nib);
        let q2l = _mm256_and_si256(b2, nib);
        let q3l = _mm256_and_si256(b3, nib);
        let q0h = _mm256_and_si256(_mm256_srli_epi16(b0, 4), nib);
        let q1h = _mm256_and_si256(_mm256_srli_epi16(b1, 4), nib);
        let q2h = _mm256_and_si256(_mm256_srli_epi16(b2, 4), nib);
        let q3h = _mm256_and_si256(_mm256_srli_epi16(b3, 4), nib);
        let lo = reduce4(
            dot32_lanes(q0l, _mm256_loadu_si256(xq.cast())),
            dot32_lanes(q1l, _mm256_loadu_si256(xq.add(64).cast())),
            dot32_lanes(q2l, _mm256_loadu_si256(xq.add(128).cast())),
            dot32_lanes(q3l, _mm256_loadu_si256(xq.add(192).cast())),
        );
        let hi = reduce4(
            dot32_lanes(q0h, _mm256_loadu_si256(xq.add(32).cast())),
            dot32_lanes(q1h, _mm256_loadu_si256(xq.add(96).cast())),
            dot32_lanes(q2h, _mm256_loadu_si256(xq.add(160).cast())),
            dot32_lanes(q3h, _mm256_loadu_si256(xq.add(224).cast())),
        );
        let mut ol = [0i32; 4];
        let mut oh = [0i32; 4];
        // SAFETY: each array is four `i32` = 16 bytes, exactly one 128-bit store.
        _mm_storeu_si128(ol.as_mut_ptr().cast(), lo);
        _mm_storeu_si128(oh.as_mut_ptr().cast(), hi);
        [ol[0], oh[0], ol[1], oh[1], ol[2], oh[2], ol[3], oh[3]]
    }
}

/// Exact `Σ_{l<32} codes[l]·xq[l]` for one `Q8_0` block from raw code pointers. `#[inline]` so a
/// `#[target_feature(enable = "avx2")]` caller keeps it in the row loop.
///
/// # Safety
///
/// `codes` must point at 32 readable bytes and `xq` at 32 readable lanes; the caller must have
/// AVX2.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn dot32_ptr(codes: *const u8, xq: *const i8) -> i32 {
    // SAFETY: the caller guarantees the 32-byte and 32-lane reads and AVX2.
    unsafe { hsum256(dot32_lanes_ptr(codes, xq)) }
}

/// Exact `Σ_{l<32} codes[l]·xq[l]` for one `Q8_0` block read straight from its raw bytes: the 32
/// signed codes are reinterpreted in place, so no `[i8; 32]` copy is formed. Test-only slice
/// wrapper around [`dot32_ptr`].
#[cfg(test)]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn dot32_u8_i8(codes: &[u8], xq: &[i8]) -> i32 {
    // SAFETY: the caller guarantees at least 32 code bytes and 32 activation lanes.
    unsafe { dot32_ptr(codes.as_ptr(), xq.as_ptr()) }
}

/// Test-only slice wrapper around [`q5_0_dot32_ptr`].
#[cfg(test)]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn q5_0_dot32(qh: &[u8], qs: &[u8], xq: &[i8]) -> i32 {
    // SAFETY: the caller guarantees 16 `qs` bytes, 4 `qh` bytes and 32 activation lanes.
    unsafe { q5_0_dot32_ptr(qh.as_ptr(), qs.as_ptr(), xq.as_ptr()) }
}

/// Exact `Σ_{l<32} c_l·xq_l` for one `Q5_0` block from raw `qh`/`qs` pointers, with
/// `c_l = (nibble_l | (bit_l·16)) − 16` in the order `quant::q5_0_codes` fixes: `qs` low nibbles
/// are lanes `0..16`, high nibbles lanes `16..32`, and `qh` bit `l` is the fifth bit of lane `l`.
///
/// The signed lane byte is assembled directly: `nibble` when the fifth bit is set is `+nibble`
/// (`c = nibble + 16 − 16`), and `nibble | 0xF0` when it is clear is the `i8` value
/// `nibble − 16`. No bias term and no second `vpmaddubsw` are needed.
///
/// The fifth-bit plane is one broadcast, one `vpshufb` and one all-ones test: each byte of the
/// little-endian `qh` word is spread to eight lanes, `BIT_FORCE` sets every bit except the one
/// that lane tests, and `vpcmpeqb` against `-1` is the test. Lane order is the same eight-bit
/// group per `qh` byte as before.
///
/// The code magnitude enters the unsigned `vpmaddubsw` operand and the code sign folds into the
/// activation, so the pair sum is over the signed products; `2·16·127 < 2^15`, so the widening add
/// cannot saturate. `#[inline]` so a `#[target_feature(enable = "avx2")]` caller keeps it in the
/// row loop.
///
/// # Safety
///
/// `qh` must point at 4 readable bytes, `qs` at 16 readable bytes and `xq` at 32 readable lanes;
/// the caller must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn q5_0_dot32_ptr(qh: *const u8, qs: *const u8, xq: *const i8) -> i32 {
    // SAFETY: the caller guarantees the four loads' extents and AVX2.
    unsafe { hsum256(q5_0_lanes_ptr(qh, qs, xq)) }
}

/// Four consecutive `Q5_0` block dots in one [`reduce4`] tree. `qh`/`qs` point at block 0's
/// `qh`/`qs` fields, `stride` is the block pitch (`Q5_0_BLOCK_BYTES`) and block `i`'s activations
/// are at `xq + 32i`; the result is `[dot0, dot1, dot2, dot3]`.
///
/// # Safety
///
/// `qh + 3·stride` must point at 4 readable bytes, `qs + 3·stride` at 16 readable bytes and `xq`
/// at 128 readable lanes; the caller must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn q5_0_dot32x4_ptr(
    qh: *const u8,
    qs: *const u8,
    stride: usize,
    xq: *const i8,
) -> [i32; 4] {
    // SAFETY: the caller's four blocks give each pass its 4 `qh` bytes, 16 `qs` bytes and 32 lanes.
    unsafe {
        let (c0, c1, c2, c3) = (
            q5_0_codes(qh, qs),
            q5_0_codes(qh.add(stride), qs.add(stride)),
            q5_0_codes(qh.add(2 * stride), qs.add(2 * stride)),
            q5_0_codes(qh.add(3 * stride), qs.add(3 * stride)),
        );
        let x0 = _mm256_loadu_si256(xq.cast());
        let x1 = _mm256_loadu_si256(xq.add(32).cast());
        let x2 = _mm256_loadu_si256(xq.add(64).cast());
        let x3 = _mm256_loadu_si256(xq.add(96).cast());
        let r = if avxvnni() {
            reduce4(
                dot32_lanes_vnni(c0, x0),
                dot32_lanes_vnni(c1, x1),
                dot32_lanes_vnni(c2, x2),
                dot32_lanes_vnni(c3, x3),
            )
        } else {
            reduce4(
                dot32_lanes_avx2(c0, x0),
                dot32_lanes_avx2(c1, x1),
                dot32_lanes_avx2(c2, x2),
                dot32_lanes_avx2(c3, x3),
            )
        };
        let mut out = [0i32; 4];
        // SAFETY: `out` is four `i32` = 16 bytes, exactly one 128-bit store.
        _mm_storeu_si128(out.as_mut_ptr().cast(), r);
        out
    }
}

/// Four `Q5_0` offset-representation dots for two rows against two blocks, the `Q5_0` twin of
/// [`dot32x2x2_rows`]: lane `(r, b)` reads row `r`'s `qh`/`qs` at `qh + r·stride + b·block_stride`
/// against `x_b`, in [`q5_0_lanes_x_offset`]'s representation, so the caller subtracts each block's
/// `16·Σxq` before scaling.
///
/// # Safety
///
/// `qh + stride + block_stride` must point at 4 readable bytes and `qs + stride + block_stride` at
/// 16; the caller must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q5_0_dot32x2x2_rows(
    qh: *const u8,
    qs: *const u8,
    stride: usize,
    block_stride: usize,
    x0: __m256i,
    x1: __m256i,
) -> __m128i {
    // SAFETY: the caller's two rows and two blocks give each pass its 4 `qh` and 16 `qs` bytes.
    unsafe {
        reduce4(
            q5_0_lanes_x_offset(qh, qs, x0),
            q5_0_lanes_x_offset(qh.add(block_stride), qs.add(block_stride), x1),
            q5_0_lanes_x_offset(qh.add(stride), qs.add(stride), x0),
            q5_0_lanes_x_offset(
                qh.add(stride + block_stride),
                qs.add(stride + block_stride),
                x1,
            ),
        )
    }
}

/// The eight `i32` partial sums of one `Q5_0` block dot, before horizontal reduction; the decode
/// is [`q5_0_dot32_ptr`]'s.
///
/// # Safety
///
/// `qh` must point at 4 readable bytes, `qs` at 16 readable bytes and `xq` at 32 readable lanes;
/// the caller must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q5_0_lanes_ptr(qh: *const u8, qs: *const u8, xq: *const i8) -> __m256i {
    // SAFETY: the caller guarantees `xq` holds the 32 lanes the load reads.
    unsafe { q5_0_lanes_x(qh, qs, _mm256_loadu_si256(xq.cast())) }
}

/// The two `Q5_0` per-lane planes of one block, from raw `qh`/`qs` pointers: the interleaved
/// nibble vector `nib` (`qs` low nibbles in lanes `0..16`, its high nibbles in lanes `16..32`) and
/// the fifth-bit mask `set` (`0xFF` exactly where the fifth bit is set).
///
/// `qh` is one `u32`, each byte spread to eight lanes by one `vpshufb`; `BIT_FORCE` leaves only the
/// tested bit clear, so `vpor` then `vpcmpeqb` against `-1` is the test. The two consumers differ
/// only in how they assemble the lane byte, so the expansion is shared.
///
/// # Safety
///
/// `qh` must point at 4 readable bytes and `qs` at 16 readable bytes; the caller must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q5_0_nib_and_set(qh: *const u8, qs: *const u8) -> (__m256i, __m256i) {
    // SAFETY: the caller guarantees the two loads' extents and AVX2.
    unsafe {
        let q = _mm_loadu_si128(qs.cast());
        let nib = _mm256_and_si256(
            _mm256_set_m128i(_mm_srli_epi16(q, 4), q),
            _mm256_set1_epi8(0x0F),
        );

        // SAFETY: `qh` points at 4 readable bytes; x86_64 is little-endian, so the unaligned load
        // is the same word `from_le_bytes` would build.
        let qh32 = qh.cast::<i32>().read_unaligned();
        let spread = _mm256_shuffle_epi8(
            _mm256_set1_epi32(qh32),
            _mm256_setr_epi8(
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                3, 3, 3, 3,
            ),
        );
        let set = _mm256_cmpeq_epi8(
            _mm256_or_si256(
                spread,
                _mm256_set1_epi64x(0x7fbf_dfef_f7fb_fdfeu64.cast_signed()),
            ),
            _mm256_set1_epi8(-1),
        );
        (nib, set)
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q5_0_codes(qh: *const u8, qs: *const u8) -> __m256i {
    let (nib, set) = unsafe { q5_0_nib_and_set(qh, qs) };
    _mm256_or_si256(
        nib,
        _mm256_andnot_si256(set, _mm256_set1_epi8(0xF0u8.cast_signed())),
    )
}

/// [`q5_0_lanes_ptr`] with the activation vector already loaded, so four rows can share one
/// activation load.
///
/// # Safety
///
/// `qh` must point at 4 readable bytes and `qs` at 16 readable bytes; the caller must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q5_0_lanes_x(qh: *const u8, qs: *const u8, x: __m256i) -> __m256i {
    // SAFETY: the caller guarantees the two loads' extents and AVX2.
    let codes = unsafe { q5_0_codes(qh, qs) };
    dot32_lanes(codes, x)
}

/// [`q5_0_lanes_x`] in the offset representation `u_l = c_l + 16 ∈ [0, 31]`: the unsigned
/// `vpmaddubsw` operand is `u` itself, so the block needs neither `abs_epi8` nor `sign_epi8`.
///
/// On AVX-VNNI hardware, `_mm256_dpbusd_avx_epi32` directly accumulates four `u8·i8` products
/// `u · x` into each `i32` dword, replacing `vpmaddubsw` + `vpmaddwd`.
///
/// **Equivalence proof for exact i32 semantics:**
/// For `Q5_0` offset representation: `u in [0, 31]`, `|x| <= 127`.
/// `2 * 31 * 127 = 7874 < 32767 = 2^15 - 1`, so `vpmaddubsw` never saturates.
/// `vpmaddwd(pair, 1)` sums pairs to `sum_{k=0..3} u_k · x_k`.
/// `vpdpbusd(0, u, x)` computes `0 + sum_{k=0..3} u_k · x_k` directly in `i32`.
/// Therefore, the results are bit-exact identical.
///
/// # Safety
///
/// `qh` must point at 4 readable bytes and `qs` at 16 readable bytes; the caller must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q5_0_lanes_x_offset(qh: *const u8, qs: *const u8, x: __m256i) -> __m256i {
    // SAFETY: the caller guarantees the two loads' extents and AVX2.
    let (nib, set) = unsafe { q5_0_nib_and_set(qh, qs) };
    // `bit·16` where the fifth bit is set, so `nib | (bit·16)` is `c + 16` in `[0, 31]`.
    let u = _mm256_or_si256(nib, _mm256_and_si256(set, _mm256_set1_epi8(0x10)));
    if avxvnni() {
        // SAFETY: AVX-VNNI detected at runtime.
        unsafe { q5_0_lanes_x_offset_vnni(u, x) }
    } else {
        _mm256_madd_epi16(_mm256_maddubs_epi16(u, x), _mm256_set1_epi16(1))
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q5_0_lanes_x_offset_vnni(u: __m256i, x: __m256i) -> __m256i {
    dpbusd_vex(_mm256_setzero_si256(), u, x)
}

/// The eight 16-lane dots `d_g = Σ_{l<16} code_l·xq_l` of one `Q6_K` block half, in group order
/// (`8n, 8n+1, …, 8n+7` for half `n`).
///
/// `ql` points at the 64 low-nibble bytes of half `n`, `qh` at its 32 two-bit high-plane bytes and
/// `xq` at that half's 128 activation codes. `Q6_K` is pure (`w = d·sc·q`), so the caller applies
/// the group scale and the activation block scale to each returned dot. The two-bit planes are
/// extracted with `srli_epi16` plus a `0x03` mask, which is exact per byte because a 16-bit shift
/// only carries the neighbour byte's high bits into bit positions the mask removes.
///
/// # Safety
///
/// `ql` must point at 64 readable bytes, `qh` at 32 readable bytes and `xq` at 128 readable lanes;
/// the caller must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn q6_k_half_dots(ql: *const u8, qh: *const u8, xq: *const i8) -> [i32; 8] {
    // SAFETY: the caller guarantees each load's extent and AVX2.
    unsafe {
        let a = _mm256_loadu_si256(ql.cast());
        let b = _mm256_loadu_si256(ql.add(32).cast());
        let h = _mm256_loadu_si256(qh.cast());
        let nib = _mm256_set1_epi8(0x0F);
        let two = _mm256_set1_epi8(0x03);
        let bias = _mm256_set1_epi8(32);
        let al = _mm256_and_si256(a, nib);
        let bl = _mm256_and_si256(b, nib);
        let ah = _mm256_and_si256(_mm256_srli_epi16(a, 4), nib);
        let bh = _mm256_and_si256(_mm256_srli_epi16(b, 4), nib);
        let h0 = _mm256_and_si256(h, two);
        let h1 = _mm256_and_si256(_mm256_srli_epi16(h, 2), two);
        let h2 = _mm256_and_si256(_mm256_srli_epi16(h, 4), two);
        let h3 = _mm256_and_si256(_mm256_srli_epi16(h, 6), two);
        let q1 = _mm256_sub_epi8(_mm256_or_si256(al, _mm256_slli_epi16(h0, 4)), bias);
        let q2 = _mm256_sub_epi8(_mm256_or_si256(bl, _mm256_slli_epi16(h1, 4)), bias);
        let q3 = _mm256_sub_epi8(_mm256_or_si256(ah, _mm256_slli_epi16(h2, 4)), bias);
        let q4 = _mm256_sub_epi8(_mm256_or_si256(bh, _mm256_slli_epi16(h3, 4)), bias);
        let x0 = _mm256_loadu_si256(xq.cast());
        let x1 = _mm256_loadu_si256(xq.add(32).cast());
        let x2 = _mm256_loadu_si256(xq.add(64).cast());
        let x3 = _mm256_loadu_si256(xq.add(96).cast());
        let h = if avxvnni() {
            reduce4_split(
                dot32_lanes_vnni(q1, x0),
                dot32_lanes_vnni(q2, x1),
                dot32_lanes_vnni(q3, x2),
                dot32_lanes_vnni(q4, x3),
            )
        } else {
            reduce4_split(
                dot32_lanes_avx2(q1, x0),
                dot32_lanes_avx2(q2, x1),
                dot32_lanes_avx2(q3, x2),
                dot32_lanes_avx2(q4, x3),
            )
        };
        let mut o = [0i32; 8];
        // SAFETY: `o` is eight `i32` = 32 bytes, exactly one 256-bit store.
        _mm256_storeu_si256(o.as_mut_ptr().cast(), h);
        // `h` holds the four low sums then the four high sums; the caller's group order is
        // `[q1lo, q1hi, q2lo, q2hi, …]`.
        [o[0], o[4], o[1], o[5], o[2], o[6], o[3], o[7]]
    }
}

/// `Some(Some(accs))` for a family with a two-row tiled core,
/// `Some(None)` on a non-finite weight scale, `None` when the family or width has no tiled core.
///
/// # Safety
///
/// A tiled core's accumulators, or the one refusal the single-row path also reports: a non-finite
/// weight scale in one of the tiled rows.
pub(super) enum Tiled<const N: usize> {
    /// The `N` rows' exact per-block folds, before rounding.
    Accs([i128; N]),
    /// A non-finite weight scale: the caller falls the whole payload back to the reference.
    NonFinite,
}

/// What [`try_tiled`] did: pushed `usize` rounded rows, refused for a non-finite scale, or found
/// no tiled core for the family and width.
pub(super) enum TileOutcome {
    /// That many rows were rounded and pushed onto the caller's output.
    Pushed(usize),
    /// A non-finite weight scale; the caller falls the whole payload back.
    NonFinite,
    /// No tiled core applies; the caller uses the single-row kernel.
    Unsupported,
}

/// `Some(accs)` for a family with a two-row tiled core, `None` when the family or width has no
/// tiled core; `Tiled::NonFinite` is a non-finite weight scale in one of the two rows.
///
/// # Safety
///
/// AVX2 must be present; `rows` must point at two consecutive rows of `row_bytes` bytes each with
/// `ggml_type`'s full block geometry for `n_in`, `codes` at `n_in` lanes, `scales` at the full
/// activation block count and `xsums` holding the family's per-block data: the `Q4_K` rows read
/// raw `Σxq` at `n_in / ACT_QK` entries, the `Q5_0` rows read one `16·Σxq` block-pair vector (see
/// [`q5_0_rows2_avx2`]), and the `Q8_0` row reads none.
pub(super) unsafe fn rows2_avx2(
    ggml_type: u32,
    rows: *const u8,
    row_bytes: usize,
    n_in: usize,
    codes: *const i8,
    scales: *const i64,
    xsums: *const i32,
) -> Option<Tiled<2>> {
    let q8 = ggml_type == quant::GGML_TYPE_Q8_0 && n_in.is_multiple_of(ACT_QK);
    let q5 = ggml_type == quant::GGML_TYPE_Q5_0 && n_in.is_multiple_of(ACT_QK);
    let q4 = ggml_type == quant::GGML_TYPE_Q4_K && n_in.is_multiple_of(QK_K);
    if !(q8 || q5 || q4) {
        return None;
    }
    // SAFETY: the caller's two rows and block geometry give the chosen core its extents.
    let accs = unsafe {
        if q8 {
            q8_0_rows2_avx2(rows, row_bytes, n_in / ACT_QK, codes, scales)
        } else if q5 {
            q5_0_rows2_avx2(rows, row_bytes, n_in / ACT_QK, codes, scales, xsums)
        } else {
            q4_k_rows2_avx2(rows, row_bytes, n_in / QK_K, codes, scales, xsums)
        }
    };
    Some(match accs {
        Some(values) => Tiled::Accs(values),
        None => Tiled::NonFinite,
    })
}

/// Apply the two-row tiled core when `remaining` rows and `avail` bytes cover it, pushing the
/// rounded results onto `out`.
///
/// Every core keeps its rows' own per-block fold order and one final rounding, so the pushed values
/// are the single-row kernels' bit for bit; an out-of-range row sets `*ok = false` (the caller then
/// returns `OutputOverflow` and discards `out`).
///
/// # Safety
///
/// AVX2 must be present; `rows` must point at `avail` readable bytes covering `remaining` whole
/// rows of `row_bytes` each with `ggml_type`'s full block geometry for `n_in`; `codes`, `scales`
/// and `xsums` carry the extents [`rows2_avx2`] states.
#[allow(
    clippy::too_many_arguments,
    reason = "the call's geometry (`ggml_type`, the row base, pitch, width, remaining rows and \
              available bytes), the shared activation plan and the output cursor are the tiled \
              core's full input"
)]
pub(super) unsafe fn try_tiled(
    ggml_type: u32,
    rows: *const u8,
    row_bytes: usize,
    n_in: usize,
    remaining: usize,
    avail: usize,
    out: &mut Vec<i64>,
    ok: &mut bool,
    codes: *const i8,
    scales: *const i64,
    xsums: *const i32,
) -> TileOutcome {
    if remaining >= 2 && avail >= 2 * row_bytes {
        // SAFETY: AVX2 is present and the caller's two rows are in range.
        if let Some(tiled) =
            unsafe { rows2_avx2(ggml_type, rows, row_bytes, n_in, codes, scales, xsums) }
        {
            return match tiled {
                Tiled::Accs(values) => {
                    push_rows(&values, out, ok);
                    TileOutcome::Pushed(2)
                }
                Tiled::NonFinite => TileOutcome::NonFinite,
            };
        }
    }
    TileOutcome::Unsupported
}

/// Round each tiled accumulator exactly as the single-row path does and push it, clearing `*ok` on
/// an `i64` range violation.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `rounded as i64` narrowing is immediately preceded by an explicit \
              `i64::MIN`/`i64::MAX` range check, as in `q8act_rows`"
)]
fn push_rows<const N: usize>(values: &[i128; N], out: &mut Vec<i64>, ok: &mut bool) {
    let mut r = 0usize;
    while r < N {
        let rounded = round_shift_32(values[r]);
        if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
            *ok = false;
        } else {
            out.push(rounded as i64);
        }
        r += 1;
    }
}

/// Two `Q8_0` fast rows in one pass over two blocks each: both blocks' activation vectors and
/// `S_b` are loaded once and shared by the two rows, while each row keeps its own per-block weight
/// scale and fold order. An odd trailing block is folded a block at a time.
///
/// # Safety
///
/// AVX2 must be present; `rows` must point at 2 consecutive rows of `blocks·Q8_0_BLOCK_BYTES` bytes
/// each, `codes` at `blocks·ACT_QK` lanes and `scales` at `blocks` entries. `None` is a non-finite
/// weight scale.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn q8_0_rows2_avx2(
    rows: *const u8,
    row_bytes: usize,
    blocks: usize,
    codes: *const i8,
    scales: *const i64,
) -> Option<[i128; 2]> {
    let mut acc = [0i128; 2];
    let mut bad: i64 = 0;
    let mut blk = 0usize;
    let stride = Q8_0_BLOCK_BYTES;
    while blk + 2 <= blocks {
        let off = blk * stride;
        // SAFETY: the four scale words at `off`/`off + stride` are inside the two rows.
        let (d00, d01, d10, d11) = unsafe {
            (
                f16_to_fixed_fast_raw(scale_word(rows, off)),
                f16_to_fixed_fast_raw(scale_word(rows, off + stride)),
                f16_to_fixed_fast_raw(scale_word(rows.add(row_bytes), off)),
                f16_to_fixed_fast_raw(scale_word(rows.add(row_bytes), off + stride)),
            )
        };
        bad |= d00 | d01 | d10 | d11;
        // SAFETY: `blk + 1 < blocks`, so both activation vectors are in range.
        let (x0, x1) = unsafe {
            (
                _mm256_loadu_si256(codes.add(blk * ACT_QK).cast()),
                _mm256_loadu_si256(codes.add((blk + 1) * ACT_QK).cast()),
            )
        };
        // SAFETY: both rows' 32 code bytes in both blocks are in range.
        let r = unsafe { dot32x2x2_rows(rows.add(off + 2), row_bytes, stride, x0, x1) };
        let mut dots = [0i32; 4];
        // SAFETY: `dots` is four `i32` = 16 bytes, exactly one 128-bit store.
        unsafe { _mm_storeu_si128(dots.as_mut_ptr().cast(), r) };
        // SAFETY: `scales` holds `blocks` entries and `blk + 1 < blocks`.
        let (s0, s1) = unsafe { (*scales.add(blk), *scales.add(blk + 1)) };
        acc[0] += i128::from(d00) * i128::from(s0 * i64::from(dots[0]));
        acc[0] += i128::from(d01) * i128::from(s1 * i64::from(dots[1]));
        acc[1] += i128::from(d10) * i128::from(s0 * i64::from(dots[2]));
        acc[1] += i128::from(d11) * i128::from(s1 * i64::from(dots[3]));
        blk += 2;
    }
    while blk < blocks {
        let off = blk * stride;
        // SAFETY: both rows' scale words at `off` are in range.
        let (d0, d1) = unsafe {
            (
                f16_to_fixed_fast_raw(scale_word(rows, off)),
                f16_to_fixed_fast_raw(scale_word(rows.add(row_bytes), off)),
            )
        };
        bad |= d0 | d1;
        // SAFETY: `blk < blocks`.
        let x = unsafe { _mm256_loadu_si256(codes.add(blk * ACT_QK).cast()) };
        // SAFETY: both rows' 32 code bytes at `off + 2` are in range.
        let dots = unsafe {
            [
                hsum256(dot32_lanes(_mm256_loadu_si256(rows.add(off + 2).cast()), x)),
                hsum256(dot32_lanes(
                    _mm256_loadu_si256(rows.add(row_bytes + off + 2).cast()),
                    x,
                )),
            ]
        };
        // SAFETY: `scales` holds `blocks` entries and `blk < blocks`.
        let s = unsafe { *scales.add(blk) };
        acc[0] += i128::from(d0) * i128::from(s * i64::from(dots[0]));
        acc[1] += i128::from(d1) * i128::from(s * i64::from(dots[1]));
        blk += 1;
    }
    if bad & 1 != 0 {
        return None;
    }
    Some(acc)
}

/// Two `Q5_0` fast rows in one pass over two blocks each; the `Q5_0` twin of [`q8_0_rows2_avx2`].
///
/// The dots are [`q5_0_lanes_x_offset`]'s `Σ (c + 16)·xq`; one `vpsubd` against the pair's
/// `[c(2j), c(2j+1), c(2j), c(2j+1)]` vector reduces all four to the exact signed block dots, with
/// `c = 16·Σxq` shared by the two rows. `dot_offset ∈ [−125984, 125984]` and `c ∈ [−65024, 65024]`,
/// so the difference is exact in `i32` and its range is the signed dot's `[−65024, 60960]`, the
/// `top = 16` magnitude the parent module's fast bound already covers.
///
/// A non-finite scale in either row sets the low bit of the `bad` accumulator
/// ([`f16_to_fixed_fast_raw`]) and `None` is returned after the block loop; the per-block test and
/// branch would otherwise sit in the row loop.
///
/// # Safety
///
/// The same obligations as [`q8_0_rows2_avx2`] with `Q5_0_BLOCK_BYTES`, plus `xsums` pointing at
/// `2·blocks` readable `i32`s: one `[c(2j), c(2j+1), c(2j), c(2j+1)]` per block pair `j`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn q5_0_rows2_avx2(
    rows: *const u8,
    row_bytes: usize,
    blocks: usize,
    codes: *const i8,
    scales: *const i64,
    xsums: *const i32,
) -> Option<[i128; 2]> {
    let mut acc = [0i128; 2];
    let mut bad: i64 = 0;
    let mut blk = 0usize;
    let stride = Q5_0_BLOCK_BYTES;
    while blk + 2 <= blocks {
        let off = blk * stride;
        // SAFETY: the four scale words at `off`/`off + stride` are inside the two rows.
        let (d00, d01, d10, d11) = unsafe {
            (
                f16_to_fixed_fast_raw(scale_word(rows, off)),
                f16_to_fixed_fast_raw(scale_word(rows, off + stride)),
                f16_to_fixed_fast_raw(scale_word(rows.add(row_bytes), off)),
                f16_to_fixed_fast_raw(scale_word(rows.add(row_bytes), off + stride)),
            )
        };
        bad |= d00 | d01 | d10 | d11;
        // SAFETY: `blk + 1 < blocks`, so both activation vectors are in range.
        let (x0, x1) = unsafe {
            (
                _mm256_loadu_si256(codes.add(blk * ACT_QK).cast()),
                _mm256_loadu_si256(codes.add((blk + 1) * ACT_QK).cast()),
            )
        };
        // SAFETY: both rows' `qh`/`qs` fields in both blocks are in range.
        let r = unsafe {
            q5_0_dot32x2x2_rows(
                rows.add(off + 2),
                rows.add(off + 6),
                row_bytes,
                stride,
                x0,
                x1,
            )
        };
        // SAFETY: `blk` is even, so the pair entry `[c(blk), c(blk+1), c(blk), c(blk+1)]` sits at
        // `2·blk` `i32`s and is in range for the `2` blocks this iteration consumes.
        let corr = unsafe { _mm_loadu_si128(xsums.add(2 * blk).cast()) };
        let mut dots = [0i32; 4];
        // SAFETY: `dots` is four `i32` = 16 bytes, exactly one 128-bit store.
        unsafe { _mm_storeu_si128(dots.as_mut_ptr().cast(), _mm_sub_epi32(r, corr)) };
        // SAFETY: `scales` holds `blocks` entries and `blk + 1 < blocks`.
        let (s0, s1) = unsafe { (*scales.add(blk), *scales.add(blk + 1)) };
        acc[0] += i128::from(d00) * i128::from(s0 * i64::from(dots[0]));
        acc[0] += i128::from(d01) * i128::from(s1 * i64::from(dots[1]));
        acc[1] += i128::from(d10) * i128::from(s0 * i64::from(dots[2]));
        acc[1] += i128::from(d11) * i128::from(s1 * i64::from(dots[3]));
        blk += 2;
    }
    while blk < blocks {
        let off = blk * stride;
        // SAFETY: both rows' scale words at `off` and the block's `16·Σxq` are in range.
        let (d0, d1) = unsafe {
            (
                f16_to_fixed_fast_raw(scale_word(rows, off)),
                f16_to_fixed_fast_raw(scale_word(rows.add(row_bytes), off)),
            )
        };
        bad |= d0 | d1;
        // SAFETY: `blk < blocks`.
        let x = unsafe { _mm256_loadu_si256(codes.add(blk * ACT_QK).cast()) };
        // SAFETY: both rows' `qh`/`qs` fields at `off + 2`/`off + 6` are in range.
        let dots = unsafe {
            [
                hsum256(q5_0_lanes_x_offset(rows.add(off + 2), rows.add(off + 6), x)),
                hsum256(q5_0_lanes_x_offset(
                    rows.add(row_bytes + off + 2),
                    rows.add(row_bytes + off + 6),
                    x,
                )),
            ]
        };
        // SAFETY: `scales` holds `blocks` entries; the tail's `blk` is even, so its pair entry at
        // `2·blk` is in range and lane 0 is `c(blk)`.
        let s = unsafe { *scales.add(blk) };
        let c = unsafe { *xsums.add(2 * blk) };
        acc[0] += i128::from(d0) * i128::from(s * i64::from(dots[0] - c));
        acc[1] += i128::from(d1) * i128::from(s * i64::from(dots[1] - c));
        blk += 1;
    }
    if bad & 1 != 0 {
        return None;
    }
    Some(acc)
}

/// Two `Q4_K` fast rows in one pass over whole 256-weight blocks, sharing the block's eight
/// activation groups and its `xsums`/`act_scales` reads; each row keeps its own `d`/`dmin`/`sc`
/// decode and per-pair fold order.
///
/// # Safety
///
/// AVX2 must be present; `rows` must point at 2 consecutive rows of `blocks·Q4_K_BLOCK_BYTES` bytes
/// each, `codes` at `blocks·QK_K` lanes, and `xsums`/`act_scales` at `8·blocks` entries each.
/// `None` is a non-finite weight scale.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn q4_k_rows2_avx2(
    rows: *const u8,
    row_bytes: usize,
    blocks: usize,
    codes: *const i8,
    act_scales: *const i64,
    xsums: *const i32,
) -> Option<[i128; 2]> {
    let mut acc = [0i128; 2];
    let mut bad: i64 = 0;
    let mut b = 0usize;
    while b < blocks {
        let off = b * Q4_K_BLOCK_BYTES;
        // SAFETY: both rows' header bytes at `off` are inside their own row extents.
        let (d, dmin, sc) = unsafe {
            let mut dv = [0i64; 2];
            let mut dmv = [0i64; 2];
            let mut scv = [[0u8; Q4_K_SCALE_SIZE]; 2];
            let mut r = 0usize;
            while r < 2 {
                let p = rows.add(r * row_bytes);
                dv[r] = f16_to_fixed_fast_raw(scale_word(p, off));
                dmv[r] = f16_to_fixed_fast_raw(scale_word(p, off + 2));
                scv[r]
                    .copy_from_slice(core::slice::from_raw_parts(p.add(off + 4), Q4_K_SCALE_SIZE));
                r += 1;
            }
            bad |= dv[0] | dv[1] | dmv[0] | dmv[1];
            (dv, dmv, scv)
        };
        // SAFETY: `codes` holds 256 lanes at `b·QK_K`.
        let dots = unsafe { q4_k_rows2_block_dots(rows, row_bytes, off + 16, codes.add(b * QK_K)) };
        let mut r = 0usize;
        while r < 2 {
            let mut c = 0usize;
            while c < 4 {
                let (sl, ml) = q4_k_scale_min_at(&sc[r], 2 * c);
                let (sh, mh) = q4_k_scale_min_at(&sc[r], 2 * c + 1);
                let slope_lo = d[r] * i64::from(sl);
                let intercept_lo = dmin[r] * i64::from(ml);
                let slope_hi = d[r] * i64::from(sh);
                let intercept_hi = dmin[r] * i64::from(mh);
                let g_lo = 8 * b + 2 * c;
                let g_hi = g_lo + 1;
                // SAFETY: `xsums`/`act_scales` hold `8·blocks` entries and `g_hi < 8·blocks`.
                let x_lo = unsafe { *xsums.add(g_lo) };
                let x_hi = unsafe { *xsums.add(g_hi) };
                let s_lo = unsafe { *act_scales.add(g_lo) };
                let s_hi = unsafe { *act_scales.add(g_hi) };
                let inner_lo = i128::from(slope_lo) * i128::from(dots[r][2 * c])
                    - i128::from(intercept_lo) * i128::from(x_lo);
                let inner_hi = i128::from(slope_hi) * i128::from(dots[r][2 * c + 1])
                    - i128::from(intercept_hi) * i128::from(x_hi);
                acc[r] += i128::from(s_lo) * inner_lo + i128::from(s_hi) * inner_hi;
                c += 1;
            }
            r += 1;
        }
        b += 1;
    }
    if bad & 1 != 0 {
        return None;
    }
    Some(acc)
}

/// The eight exact 16-lane dots of one full 256-weight `Q4_K` block for two rows, sharing the
/// block's eight activation groups: [`q4_k_block_dots`] twice against one activation load set.
///
/// # Safety
///
/// `rows + r·stride + qs_off` must point at 128 readable bytes for `r < 2` and `xq` at 256 readable
/// lanes; the caller must have AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q4_k_rows2_block_dots(
    rows: *const u8,
    stride: usize,
    qs_off: usize,
    xq: *const i8,
) -> [[i32; 8]; 2] {
    // SAFETY: `xq` holds the block's 256 lanes; the eight loads read them once.
    let (x0, x1, x2, x3, x4, x5, x6, x7) = unsafe {
        (
            _mm256_loadu_si256(xq.cast()),
            _mm256_loadu_si256(xq.add(32).cast()),
            _mm256_loadu_si256(xq.add(64).cast()),
            _mm256_loadu_si256(xq.add(96).cast()),
            _mm256_loadu_si256(xq.add(128).cast()),
            _mm256_loadu_si256(xq.add(160).cast()),
            _mm256_loadu_si256(xq.add(192).cast()),
            _mm256_loadu_si256(xq.add(224).cast()),
        )
    };
    let nib = _mm256_set1_epi8(0x0F);
    let mut out = [[0i32; 8]; 2];
    let mut r = 0usize;
    while r < 2 {
        let base = unsafe { rows.add(r * stride + qs_off) };
        // SAFETY: the caller's extents give this row's four 32-byte windows.
        let (b0, b1, b2, b3) = unsafe {
            (
                _mm256_loadu_si256(base.cast()),
                _mm256_loadu_si256(base.add(32).cast()),
                _mm256_loadu_si256(base.add(64).cast()),
                _mm256_loadu_si256(base.add(96).cast()),
            )
        };
        let lo = reduce4(
            dot32_lanes(_mm256_and_si256(b0, nib), x0),
            dot32_lanes(_mm256_and_si256(b1, nib), x2),
            dot32_lanes(_mm256_and_si256(b2, nib), x4),
            dot32_lanes(_mm256_and_si256(b3, nib), x6),
        );
        let hi = reduce4(
            dot32_lanes(_mm256_and_si256(_mm256_srli_epi16(b0, 4), nib), x1),
            dot32_lanes(_mm256_and_si256(_mm256_srli_epi16(b1, 4), nib), x3),
            dot32_lanes(_mm256_and_si256(_mm256_srli_epi16(b2, 4), nib), x5),
            dot32_lanes(_mm256_and_si256(_mm256_srli_epi16(b3, 4), nib), x7),
        );
        let mut ol = [0i32; 4];
        let mut oh = [0i32; 4];
        // SAFETY: each array is four `i32` = 16 bytes, exactly one 128-bit store.
        unsafe {
            _mm_storeu_si128(ol.as_mut_ptr().cast(), lo);
            _mm_storeu_si128(oh.as_mut_ptr().cast(), hi);
        }
        out[r] = [ol[0], oh[0], ol[1], oh[1], ol[2], oh[2], ol[3], oh[3]];
        r += 1;
    }
    out
}
