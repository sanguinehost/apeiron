//! The scalar arithmetic of the power-of-two activation chart, in the extraction fragment
//! (`proof/extract/lib.rs` includes this file by `#[path]`):
//! - [`act_scale`]: the activation block scale `quant_act` uses;
//! - [`f16_split`]: a weight scale as `m·2^k`;
//! - [`finish`]: the kernel's final rounding.
//!
//! The vector kernel (`kern::gemm`) calls these bodies; their Lean refinements are in
//! `proof/corpus/Bridge/Pow2Chart.lean`. The loops are `while` loops carrying their values, with no
//! `TryFrom`, no iterators and no early return inside a loop, so Charon/Aeneas translate them.

/// The smallest power of two `S` with `127·S ≥ absmax`: `2^⌈log₂ ⌈absmax/127⌉⌉`, `0` for
/// `absmax = 0`, `None` when it would pass `2^62` (the largest power of two below `i64::MAX`).
///
/// # Complexity
///
/// At most 63 doublings.
#[must_use]
#[allow(
    clippy::manual_is_multiple_of,
    reason = "division and remainder keep this extraction body free of the unmodelled is_multiple_of method"
)]
pub fn act_scale(absmax: u64) -> Option<u64> {
    if absmax == 0 {
        return Some(0);
    }
    // ⌈absmax / 127⌉ without the `absmax + 126` overflow.
    let need = absmax / 127 + u64::from(absmax % 127 != 0);
    let mut s: u64 = 1;
    let mut steps: u32 = 0;
    while s < need && steps < 62 {
        s *= 2;
        steps += 1;
    }
    if s < need {
        None
    } else {
        Some(s)
    }
}

/// `f16_to_fixed(h) = m·2^k` for a finite `f16` word `h`: `m` signed with `|m| ≤ 2047` and
/// `k ∈ [8, 37]`; `None` for `Inf`/`NaN`.
///
/// A normal word is `(1024 + f)·2^(e − 25)`, so `(1024 + f)·2^(e + 7)` in Q32.32; a subnormal is
/// `f·2^−24`, so `f·2^8`.
#[must_use]
#[allow(
    clippy::many_single_char_names,
    clippy::cast_lossless,
    reason = "h/e/f/m/k are the standard half-word fields; the masked fraction is at most 1023, so the u16-to-i32 cast is exact and avoids Aeneas's unmodeled FromU16 external"
)]
pub fn f16_split(h: u16) -> Option<(i32, u32)> {
    let e = u32::from((h >> 10) & 0x1f);
    let f = (h & 0x3ff) as i32;
    if e == 31 {
        return None;
    }
    let (m, k) = if e == 0 { (f, 8) } else { (f + 1024, e + 7) };
    if h >> 15 == 1 {
        Some((-m, k))
    } else {
        Some((m, k))
    }
}

/// `round_shift_32(acc·2^s) = ⌊(acc·2^s + 2^31) / 2^32⌋` in `i64`, or `None` when it leaves
/// `i64`.
///
/// For `s ≥ 32` it is `acc·2^(s − 32)`; below, `⌊(acc + 2^(31 − s)) / 2^(32 − s)⌋`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `as i64` narrowing follows an explicit `i64::MIN..=i64::MAX` check; `TryFrom` \
              emits an unmodeled external the fail-closed extraction audit rejects"
)]
pub fn finish(acc: i64, s: u32) -> Option<i64> {
    let wide = i128::from(acc);
    let v: i128 = if s >= 32 {
        let up = s - 32;
        if acc == 0 {
            0
        } else if up > 63 {
            return None;
        } else {
            wide * (1i128 << up)
        }
    } else {
        let down = 32 - s;
        (wide + (1i128 << (down - 1))) >> down
    };
    if v < i128::from(i64::MIN) || v > i128::from(i64::MAX) {
        None
    } else {
        Some(v as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `act_scale` is the smallest power of two covering `absmax / 127`, and at most twice
    /// `⌈absmax / 127⌉`.
    #[test]
    fn act_scale_is_the_least_covering_power_of_two() {
        let samples = [
            0u64,
            1,
            126,
            127,
            128,
            254,
            255,
            1 << 20,
            (1 << 20) + 1,
            127 << 40,
            (127 << 40) + 1,
            127 << 56,
            (127 << 56) + 1,
            u64::MAX,
        ];
        for a in samples {
            let c = a.div_ceil(127);
            match act_scale(a) {
                Some(0) => assert_eq!(a, 0),
                Some(s) => {
                    assert!(s.is_power_of_two());
                    assert!(u128::from(s) * 127 >= u128::from(a), "{a}");
                    assert!(s == 1 || (s / 2) * 127 < a, "not least at {a}");
                    assert!(s < 2 * c.max(1), "{a}");
                }
                None => assert!(c > 1 << 62, "{a}"),
            }
        }
    }

    /// `finish` is `round_shift_32(acc·2^s)` wherever that fits `i128`.
    #[test]
    fn finish_is_round_shift_32_of_the_scaled_sum() {
        let accs = [
            0i64,
            1,
            -1,
            3,
            -3,
            1 << 40,
            -(1 << 40),
            i64::MAX,
            i64::MIN,
            12_345_678,
        ];
        for acc in accs {
            for s in 0..=95u32 {
                // For `s ≤ 64` the product fits `i128`; above it, `acc·2^(s − 32)` directly.
                let want = if s <= 64 {
                    let v = (i128::from(acc) << s) + (1i128 << 31);
                    i64::try_from(v >> 32).ok()
                } else {
                    i128::from(acc)
                        .checked_mul(1i128 << (s - 32))
                        .and_then(|v| i64::try_from(v).ok())
                };
                assert_eq!(finish(acc, s), want, "acc {acc} s {s}");
            }
        }
    }
}
