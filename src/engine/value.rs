//! The value-path epilogue helpers shared by every fixed-point kernel: round-to-nearest
//! division and the documented `>> 32` shift, plus the deterministic vector checksum.

/// Round-to-nearest `num / den` (`den > 0`, `num >= 0`) in `i128`, ties upward.
pub(super) fn div_round_nearest(num: i128, den: i128) -> i128 {
    (num + den / 2) / den
}

/// Round-to-nearest `v / 2^32` for a signed `i128`, ties upward.
pub(super) fn round_shift_32(v: i128) -> i128 {
    (v + (1_i128 << 31)) >> 32
}

/// A deterministic 64-bit checksum of a fixed-point vector: FNV-1a over the little-endian
/// two's-complement encoding of each `i64`.
#[must_use]
pub fn fixed_vector_checksum(v: &[i64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in v {
        for b in x.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The value-path Q32.32 product `round(a·b / 2^32)`: one `i128` product and one
/// [`round_shift_32`], `None` when the rounded value leaves `i64`. The one checked multiply every
/// elementwise stage (gates, decays, gains) uses.
pub(crate) fn mul_q32(a: i64, b: i64) -> Option<i64> {
    i64::try_from(round_shift_32(i128::from(a) * i128::from(b))).ok()
}
