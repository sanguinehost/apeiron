//! Exact integer Walsh–Hadamard rotation for the ternary/binary folded weights
//!.
//!
//! A folded weight stores its activations in the Sylvester basis `H_{2^k} = H_2^{⊗k}` with
//! `H_2 = [[1, 1], [1, −1]]`. Inference multiplies the activation by the stored sign vector, then
//! by `H_{2^k}/√(2^k)`. Both steps are integer here: the unnormalized butterflies are exact `i64`
//! arithmetic and all normalization folds into one final rounding, replacing the fork's `k`
//! float rounding stages (`ggml_compute_forward_fwht_impl`, `ggml/src/ggml-cpu/ops.cpp:12066`;
//! sign flip and matmul in `src/llama-graph.cpp:1555-1578`). No float appears on the executing
//! path.
//!
//! The modulus `2^k` is bounded by `MAX_K`, so a block never leaves `i64` before the checked
//! normalization; an overflow refuses with [`HadamardError::Overflow`] and nothing is clamped.

use super::value::div_round_nearest;
use super::FIXED_FRAC_BITS;

/// The largest block exponent: `block = 2^k` is admitted for `1 ≤ k ≤ MAX_K`.
const MAX_K: usize = 16;

/// `C[k] = round(2^32 / √(2^k))`, the Q32.32 constant for `1/√(2^k)`; `C[0]` is unused.
///
/// For even `k`, `C[k] = 2^(32 − k/2)` exactly, so the normalization is one exact shift in that
/// case. The table is checked against the reference to half a unit in the last place in
/// `c_table_matches_reference`.
const C: [i64; MAX_K + 1] = [
    0,
    3_037_000_500,
    2_147_483_648,
    1_518_500_250,
    1_073_741_824,
    759_250_125,
    536_870_912,
    379_625_062,
    268_435_456,
    189_812_531,
    134_217_728,
    94_906_266,
    67_108_864,
    47_453_133,
    33_554_432,
    23_726_566,
    16_777_216,
];

/// A refused Walsh–Hadamard rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HadamardError {
    /// `block` is not `2^k` with `1 ≤ k ≤ 16`.
    NotPowerOfTwo,
    /// `x.len()` is not a multiple of `block`, or `signs` is present with a length other than
    /// `x.len()`.
    LengthMismatch,
    /// A sign entry is not `-1` or `+1`.
    BadSign,
    /// A checked negation, butterfly add/sub or normalization left `i64`/`i128`.
    Overflow,
}

/// The exponent `k` with `block == 2^k` and `1 ≤ k ≤ MAX_K`, or `None`.
///
/// The loop is bounded by `MAX_K`, so the function is O(1) time and space.
fn block_exponent(block: usize) -> Option<usize> {
    if block < 2 {
        return None;
    }
    let mut k = 0usize;
    while k < MAX_K && (1_usize << k) < block {
        k += 1;
    }
    if (1_usize << k) == block {
        Some(k)
    } else {
        None
    }
}

/// Round-to-nearest `num / 2^32` for a signed `i128`, ties away from zero.
///
/// [`div_round_nearest`] admits only `num ≥ 0`, so the magnitude is rounded with it and the sign
/// is reapplied. O(1) time and space. The product `num` cannot reach `i128::MIN`, so `-num` is
/// safe.
fn round_q32_signed(num: i128) -> i128 {
    let den = 1_i128 << FIXED_FRAC_BITS;
    if num >= 0 {
        div_round_nearest(num, den)
    } else {
        -div_round_nearest(-num, den)
    }
}

/// The unnormalized in-place FWHT core: `k` butterfly stages over each `block`-sized segment.
///
/// Stage `len = 1, 2, …, 2^(k−1)` maps `(u, v) ↦ (u + v, u − v)`; after the stages each segment
/// holds `H_{2^k} · x` exactly (the Sylvester entry is `(−1)^popcount(i & j)`). Signs and
/// normalization belong to [`fwht_fixed`].
///
/// The block loop and the butterfly loop carry a `bool` failure flag and stop on the first
/// refusal, so no `?` and no early `return` appears inside either; the pinned Charon/Aeneas
/// fragment lowers a return inside a loop to an unsupported control-flow pattern.
///
/// # Errors
///
/// [`HadamardError::NotPowerOfTwo`] for a `block` outside `2^k`, `1 ≤ k ≤ 16`;
/// [`HadamardError::LengthMismatch`] when `x.len()` is not a multiple of `block`;
/// [`HadamardError::Overflow`] on a checked add/sub.
///
/// # Complexity
///
/// `O(n log block)` checked `i64` add/sub and `O(1)` extra space: each of the `k = log2(block)`
/// stages touches every one of the `n = x.len()` elements once, in place.
#[allow(
    clippy::manual_let_else,
    reason = "the explicit `match` early return is written out because let-else desugars through \
              a construct the pinned Charon/Aeneas fragment does not model, as in \
              `src/engine/norm.rs`"
)]
pub fn fwht_int(x: &mut [i64], block: usize) -> Result<(), HadamardError> {
    let k = match block_exponent(block) {
        Some(k) => k,
        None => return Err(HadamardError::NotPowerOfTwo),
    };
    if !x.len().is_multiple_of(block) {
        return Err(HadamardError::LengthMismatch);
    }
    let mut ok = true;
    let mut err: Option<HadamardError> = None;
    let mut stage = 0usize;
    let mut span = 1usize;
    while stage < k && ok {
        let mut base = 0usize;
        while base < x.len() && ok {
            let mut lane = 0usize;
            while lane < span && ok {
                let top = x[base + lane];
                let bottom = x[base + lane + span];
                // `Result` arms, not `Option` arms, are the shape `dequant.rs` uses and the shape
                // the refusal flag carries; the `Err` arm binds its refusal for the flag.
                match top.checked_add(bottom).ok_or(HadamardError::Overflow) {
                    Ok(sum) => match top.checked_sub(bottom).ok_or(HadamardError::Overflow) {
                        Ok(diff) => {
                            x[base + lane] = sum;
                            x[base + lane + span] = diff;
                        }
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                        }
                    },
                    Err(e) => {
                        err = Some(e);
                        ok = false;
                    }
                }
                lane += 1;
            }
            base += 2 * span;
        }
        span *= 2;
        stage += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// The exact integer normalized rotation `x' = round(c · H_{2^k} · (s ⊙ x) / 2^32)` per block.
///
/// `x` is a row of `x.len() / block` consecutive blocks of `block = 2^k` elements. Each block is
/// multiplied by its signs (checked negation), transformed in place by [`fwht_int`], then
/// normalized once with `c = C[k] ≈ 2^32/√(2^k)` and the documented round-to-nearest `>> 32`
/// epilogue. For even `k`, `c = 2^(32 − k/2)` is exact and the block is a fixed-point isometry up
/// to the single rounding.
///
/// # Errors
///
/// [`HadamardError::NotPowerOfTwo`], [`HadamardError::LengthMismatch`] and
/// [`HadamardError::BadSign`] are checked before any write, so `x` is untouched on those
/// refusals. [`HadamardError::Overflow`] can occur after writes begin; `x` is then unspecified,
/// as in the spec's failure table.
///
/// # Complexity
///
/// Per block of `n = 2^k` elements: `O(n log n)` checked `i64` add/sub plus `n` `i128`
/// multiplications and roundings; `O(1)` extra space. `x` is transformed in place.
#[allow(
    clippy::cast_possible_truncation,
    clippy::manual_let_else,
    reason = "the one `as i64` narrowing is range-checked against `i64::MAX`/`MIN` immediately \
              before it; `TryFrom` and `?` inside a loop each emit an unmodeled external in the \
              pinned Charon/Aeneas extraction, and let-else desugars through an unmodeled construct \
              (see `src/engine/norm.rs`)"
)]
pub fn fwht_fixed(x: &mut [i64], signs: Option<&[i8]>, block: usize) -> Result<(), HadamardError> {
    // Validate block and lengths before any write.
    let k = match block_exponent(block) {
        Some(k) => k,
        None => return Err(HadamardError::NotPowerOfTwo),
    };
    if !x.len().is_multiple_of(block) {
        return Err(HadamardError::LengthMismatch);
    }
    let sign_len = match signs {
        Some(s) => s.len(),
        None => x.len(),
    };
    if sign_len != x.len() {
        return Err(HadamardError::LengthMismatch);
    }
    if let Some(s) = signs {
        let mut idx = 0usize;
        let mut signs_ok = true;
        while idx < s.len() && signs_ok {
            if s[idx] != 1 && s[idx] != -1 {
                signs_ok = false;
            }
            idx += 1;
        }
        if !signs_ok {
            return Err(HadamardError::BadSign);
        }
    }

    // Sign multiply: checked negation. `checked_neg` has no model in the pinned Aeneas fragment
    // (`CheckedOps/Neg.lean` is absent), so the negation is the modeled `0 - v` checked
    // subtraction, which accepts the same values.
    if let Some(s) = signs {
        let mut idx = 0usize;
        let mut neg_ok = true;
        let mut neg_err: Option<HadamardError> = None;
        while idx < x.len() && neg_ok {
            if s[idx] == -1 {
                match 0_i64.checked_sub(x[idx]).ok_or(HadamardError::Overflow) {
                    Ok(v) => x[idx] = v,
                    Err(e) => {
                        neg_err = Some(e);
                        neg_ok = false;
                    }
                }
            }
            idx += 1;
        }
        if let Some(e) = neg_err {
            return Err(e);
        }
    }

    match fwht_int(x, block) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }

    // One normalization for the whole row: `round(y * C[k] / 2^32)`, narrowed checked.
    let coeff = i128::from(C[k]);
    let mut idx = 0usize;
    let mut norm_ok = true;
    let mut norm_err: Option<HadamardError> = None;
    while idx < x.len() && norm_ok {
        match i128::from(x[idx])
            .checked_mul(coeff)
            .ok_or(HadamardError::Overflow)
        {
            Ok(prod) => {
                let rounded = round_q32_signed(prod);
                if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
                    norm_err = Some(HadamardError::Overflow);
                    norm_ok = false;
                } else {
                    x[idx] = rounded as i64;
                }
            }
            Err(e) => {
                norm_err = Some(e);
                norm_ok = false;
            }
        }
        idx += 1;
    }
    if let Some(e) = norm_err {
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "test-only reference arithmetic converts small bounded test indices and \
                  magnitudes to f64/i64; none of it is extracted or on the executing path"
    )]

    use super::*;

    /// A fixed 64-bit LCG (Knuth's MMIX constants): deterministic pseudo-random test inputs with
    /// no `rand` dependency.
    fn lcg_next(state: &mut u64) -> i64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 33) % 2001) as i64 - 1000
    }

    /// `H_{2^k} · (s ⊙ x)` by the entry formula `(−1)^popcount(i & j)`, O(n²) per block.
    fn naive_sylvester(x: &[i64], signs: Option<&[i8]>, k: usize) -> Vec<i64> {
        let n = 1_usize << k;
        let mut out = vec![0_i64; n];
        for (i, slot) in out.iter_mut().enumerate() {
            let mut acc: i128 = 0;
            for j in 0..n {
                let sign = match signs {
                    Some(s) => i128::from(s[j]),
                    None => 1,
                };
                let entry: i128 = if (i & j).count_ones() % 2 == 1 { -1 } else { 1 };
                acc += entry * sign * i128::from(x[j]);
            }
            *slot = acc as i64;
        }
        out
    }

    /// `2^(32 − k/2)`, the reference Q32.32 constant for `1/√(2^k)`.
    fn reference_c(k: usize) -> f64 {
        2_f64.powi(32) / 2_f64.powf(k as f64 / 2.0)
    }

    #[test]
    fn c_table_matches_reference() {
        for (k, &c_k) in C.iter().enumerate().take(MAX_K + 1).skip(1) {
            let want = reference_c(k);
            let diff = (c_k as f64 - want).abs();
            assert!(diff <= 0.5, "k={k} C={c_k} want={want} diff={diff}");
            if k % 2 == 0 {
                assert_eq!(c_k, 1_i64 << (32 - k / 2), "even k={k}");
            }
        }
    }

    #[test]
    fn fwht_int_matches_naive_sylvester() {
        let mut state = 0x1234_5678_9abc_def0_u64;
        for k in 1..=10 {
            let n = 1_usize << k;
            let x: Vec<i64> = (0..n).map(|_| lcg_next(&mut state)).collect();
            let want = naive_sylvester(&x, None, k);
            let mut got = x.clone();
            fwht_int(&mut got, n).expect("valid block");
            assert_eq!(got, want, "k={k}");
        }
    }

    #[test]
    fn fwht_fixed_matches_naive_signed_sylvester() {
        let mut state = 0xdead_beef_cafe_1234_u64;
        for (k, &c_k) in C.iter().enumerate().take(11).skip(1) {
            let n = 1_usize << k;
            let x: Vec<i64> = (0..n).map(|_| lcg_next(&mut state)).collect();
            let signs: Vec<i8> = (0..n).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
            let h = naive_sylvester(&x, Some(&signs), k);
            let mut got = x.clone();
            fwht_fixed(&mut got, Some(&signs), n).expect("valid block");
            let coeff_err = (c_k as f64 / 2_f64.powi(32) - 2_f64.powf(-(k as f64) / 2.0)).abs();
            for (i, &h_i) in h.iter().enumerate() {
                let want = h_i as f64 / 2_f64.powf(k as f64 / 2.0);
                let bound = 1.0 + (h_i as f64).abs() * coeff_err;
                let diff = (got[i] as f64 - want).abs();
                assert!(
                    diff <= bound + 1e-9,
                    "k={k} i={i} got={} want={want} diff={diff} bound={bound}",
                    got[i]
                );
            }
        }
    }

    #[test]
    fn fwht_fixed_involution_even_k() {
        let mut state = 0x0f0f_0f0f_1234_5678_u64;
        for k in (2..=16).step_by(2) {
            let n = 1_usize << k;
            let x: Vec<i64> = (0..n).map(|_| lcg_next(&mut state)).collect();
            let mut y = x.clone();
            fwht_fixed(&mut y, None, n).expect("valid block");
            fwht_fixed(&mut y, None, n).expect("valid block");
            for (i, (&a, &b)) in y.iter().zip(x.iter()).enumerate() {
                assert!((a - b).abs() <= 2, "k={k} i={i} y={a} x={b}");
            }
        }
    }

    #[test]
    fn refusals_leave_input_untouched() {
        // NotPowerOfTwo: 0, 1, non-powers, and k > 16.
        for block in [0_usize, 1, 3, 6, 1 << 17] {
            let original = vec![3_i64, -5, 7, 11];
            let mut x = original.clone();
            assert_eq!(
                fwht_fixed(&mut x, None, block),
                Err(HadamardError::NotPowerOfTwo),
                "block={block}"
            );
            assert_eq!(x, original, "block={block} mutated on refusal");
        }
        // LengthMismatch: x.len() not a multiple of block.
        let original = vec![3_i64, -5, 7];
        let mut x = original.clone();
        assert_eq!(
            fwht_fixed(&mut x, None, 4),
            Err(HadamardError::LengthMismatch)
        );
        assert_eq!(x, original);
        // LengthMismatch: signs length differs from x.
        let original = vec![3_i64, -5, 7, 11];
        let mut x = original.clone();
        assert_eq!(
            fwht_fixed(&mut x, Some(&[1_i8, -1]), 4),
            Err(HadamardError::LengthMismatch)
        );
        assert_eq!(x, original);
        // BadSign: before any write.
        for bad in [0_i8, 2, -2, i8::MIN, i8::MAX] {
            let mut x = original.clone();
            assert_eq!(
                fwht_fixed(&mut x, Some(&[1, bad, -1, 1]), 4),
                Err(HadamardError::BadSign),
                "sign={bad}"
            );
            assert_eq!(x, original);
        }
    }

    #[test]
    fn overflow_refusals() {
        // Butterfly add overflow.
        let mut x = vec![i64::MAX, 1];
        assert_eq!(fwht_fixed(&mut x, None, 2), Err(HadamardError::Overflow));
        // Butterfly subtract overflow: (0 − i64::MIN).
        let mut x = vec![0_i64, i64::MIN];
        assert_eq!(fwht_fixed(&mut x, None, 2), Err(HadamardError::Overflow));
        // Sign negation overflow: −i64::MIN.
        let mut x = vec![i64::MIN, 0];
        assert_eq!(
            fwht_fixed(&mut x, Some(&[-1, 1]), 2),
            Err(HadamardError::Overflow)
        );
    }
}
