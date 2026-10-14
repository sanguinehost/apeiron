//! E8 ∪ O8 256-entry codebook of the one-bit fold transfer, in Q32.32 (`i64`, `ONE = 2^32`).
//!
//! Port of `e8_o8_codebook` in `tools/onebit/fold_transfer.py` (lines 239–263). Nearest-row rule:
//! `encode` in `tools/onebit/pv_finish.py` (lines 87–96), the argmax of `2·B·Cᵀ − |C|²`, which is
//! the Euclidean argmin of `|B − C|²` with the first index on ties. Scaled rule: the per-row cost
//! `a²|C|² − 2a·x·C` of `gptq_e8` (`fold_transfer.py` 301–308) and `e8_icm_step` (608), i.e. the
//! nearest codeword to `a·C_j`. No prior Rust body exists for this codebook (epic W19, Subsumed).
//!
//! Row layout (the index is the stored format, `codeword(idx)`):
//! - `0..128`: the 128 even-parity half-cube spinors `(±1)^8` (an even number of `−1`), norm √8.
//!   Row `r` is the `r`-th integer `i` in `0..256` with even popcount, ascending. Coordinate `k` is
//!   `−1` iff bit `7−k` of `i` is set.
//! - `128..240`: the 112 D8 roots `±2e_a ±2e_b` (`a < b`), loop order `(a, b, s_a, s_b)` with
//!   `s ∈ (−2, +2)` each, norm √8.
//! - `240..256`: the 16 O8 vertices `±√8·e_i`, row `240 + 2i` is `−√8 e_i`, row `240 + 2i + 1` is `+√8 e_i`.
//!
//! Scale: the script's real values with no normalisation. A Q32.32 entry is `value · 2^32`. The 240
//! E8 rows are exactly the doubled coordinates `2x`, `x ∈ E8 = D8 ∪ (D8 + ½·1)`, so their `value/ONE`
//! is an integer with all coordinates of one parity and sum `≡ 0 (mod 4)`. The 16 O8 entries are
//! `±√8`, irrational, and are carried as [`SQRT8_Q32`]; they are not in E8.
//!
//! Rounding: a scaled codeword `round(a·c / ONE)` is round-half-up `(n + 2^31) >> 32`, the rule of
//! `round_shift_32` in `src/engine/value.rs`. Distances are exact `i128` sums of squares.
//!
//! This module is dependency-free apart from [`crate::engine::ONE`] so that it can be pulled in by
//! `#[path]` into the extraction.

use crate::engine::ONE;

/// Number of codebook rows: 128 spinors, 112 D8 roots, 16 O8 vertices.
pub const CODEBOOK_LEN: usize = 256;

/// `round(√8 · 2^32) = 12_148_002_000`, the Q32.32 value of `2√2`. Correct rounding is proved by the
/// test `sqrt8_q32_is_correct_rounding`: `(2s−1)² < 2^69 < (2s+1)²` in `i128`, i.e. `|s − √(8·2^64)| < ½`.
pub const SQRT8_Q32: i64 = 12_148_002_000;

const NEG_ONE: i64 = -ONE;
const TWO_ONE: i64 = 2 * ONE;
const NEG_TWO_ONE: i64 = -2 * ONE;
const NEG_SQRT8_Q32: i64 = -SQRT8_Q32;

/// Refusals of the codebook and the nearest-row search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum E8Error {
    /// `idx >= 256` passed to [`codeword`].
    IndexOutOfRange,
    /// `scale <= 0` passed to [`nearest_scaled`].
    NonPositiveScale,
    /// A checked `i128` operation in the squared distance left `i128`.
    Overflow,
}

/// Parity of the popcount of `i` (8-bit index): `true` iff the number of set bits is even.
///
/// # Complexity
/// `O(8)` single-bit steps for an 8-bit `i`: one XOR and one shift per bit.
fn even_parity(i: usize) -> bool {
    let mut x: usize = i;
    let mut p: usize = 0;
    while x > 0 {
        p ^= x & 1;
        x >>= 1;
    }
    p == 0
}

/// Row `r < 128` of the codebook: the `r`-th even-popcount sign vector in ascending index order.
///
/// # Complexity
/// `O(256)` loop steps, each an [`even_parity`] of cost `O(8)`: `O(2048)` bit operations. The loop
/// always finds its key for `r < 128`, since exactly 128 of the 256 indices have even popcount.
fn spinor_row(r: usize) -> [i64; 8] {
    let mut i: usize = 0;
    let mut count: usize = 0;
    let mut key: usize = 0;
    let mut hit: bool = false;
    while i < 256 && !hit {
        if even_parity(i) {
            if count == r {
                key = i;
                hit = true;
            } else {
                count += 1;
            }
        }
        i += 1;
    }
    let mut out = [0_i64; 8];
    let mut k: usize = 0;
    while k < 8 {
        out[k] = if ((key >> (7 - k)) & 1) == 1 {
            NEG_ONE
        } else {
            ONE
        };
        k += 1;
    }
    out
}

/// Row `128 + k` (`k < 112`) of the codebook: the `k`-th D8 root `±2e_a ±2e_b` in loop order.
///
/// # Complexity
/// `O(28)` pair steps to locate the pair `(a, b)` with `pos = k/4` in lexicographic order.
fn root_row(k: usize) -> [i64; 8] {
    let pair = k / 4;
    let sub = k % 4;
    let mut pos: usize = 0;
    let mut first: usize = 0;
    let mut second: usize = 1;
    let mut hit: bool = false;
    while first < 7 && !hit {
        second = first + 1;
        while second < 8 && !hit {
            if pos == pair {
                hit = true;
            } else {
                pos += 1;
                second += 1;
            }
        }
        if !hit {
            first += 1;
        }
    }
    let mut out = [0_i64; 8];
    // `sub = 2·(index of s_a) + (index of s_b)`, with index 0 for −2 and 1 for +2.
    out[first] = if sub / 2 == 0 { NEG_TWO_ONE } else { TWO_ONE };
    out[second] = if sub.is_multiple_of(2) {
        NEG_TWO_ONE
    } else {
        TWO_ONE
    };
    out
}

/// Row `240 + 2i + s` (`s = 0` for `−√8`, `1` for `+√8`) of the codebook: the O8 vertex `±√8 e_i`.
fn o8_row(k: usize) -> [i64; 8] {
    let mut out = [0_i64; 8];
    out[k / 2] = if k.is_multiple_of(2) {
        NEG_SQRT8_Q32
    } else {
        SQRT8_Q32
    };
    out
}

/// Row `j < 256` of the codebook, without the range check. Callers guarantee `j < 256`.
fn codeword_unchecked(j: usize) -> [i64; 8] {
    if j < 128 {
        spinor_row(j)
    } else if j < 240 {
        root_row(j - 128)
    } else {
        o8_row(j - 240)
    }
}

/// Codeword `idx` of the 256-entry E8 ∪ O8 codebook, as Q32.32 coordinates (`value · 2^32`).
///
/// For `idx < 240` the coordinates are doubled-integer coordinates `2x` of an E8 point (`value/ONE`
/// is an integer). For `240 ≤ idx < 256` they are `±√8·e_i` ([`SQRT8_Q32`]).
///
/// # Errors
/// [`E8Error::IndexOutOfRange`] if `idx >= 256`.
///
/// # Complexity
/// `O(256·8)` bit operations in the worst case (the spinor rows, see [`spinor_row`]); `O(28)` pair
/// steps for a root row; `O(1)` for an O8 row. Measure: single-bit operations on `usize`.
pub fn codeword(idx: usize) -> Result<[i64; 8], E8Error> {
    if idx < CODEBOOK_LEN {
        Ok(codeword_unchecked(idx))
    } else {
        Err(E8Error::IndexOutOfRange)
    }
}

/// The squared-distance term for one coordinate: `(x − round(scale·c / ONE))²` in `i128`, or `None`
/// on overflow. The rounding is round-half-up `(n + 2^31) >> 32`, the rule of `round_shift_32`.
fn scaled_term(x: i64, c: i64, scale: i64) -> Option<i128> {
    match i128::from(scale).checked_mul(i128::from(c)) {
        None => None,
        Some(sc) => match sc.checked_add(1_i128 << 31) {
            None => None,
            Some(t) => match i128::from(x).checked_sub(t >> 32) {
                None => None,
                Some(e) => e.checked_mul(e),
            },
        },
    }
}

/// Squared distance from `x` to the scaled row `scale · row` (Q32.32 rounding per coordinate), as an
/// exact `i128` sum of the eight [`scaled_term`]s.
///
/// # Errors
/// [`E8Error::Overflow`] if any term or the running sum leaves `i128`.
///
/// # Complexity
/// `O(8)` checked `i128` multiplications and additions.
fn scaled_dist2(x: &[i64; 8], row: &[i64; 8], scale: i64) -> Result<i128, E8Error> {
    let mut acc: i128 = 0;
    let mut i: usize = 0;
    let mut ok: bool = true;
    while i < 8 && ok {
        match scaled_term(x[i], row[i], scale) {
            None => {
                ok = false;
            }
            Some(t) => match acc.checked_add(t) {
                None => {
                    ok = false;
                }
                Some(s) => {
                    acc = s;
                }
            },
        }
        i += 1;
    }
    if ok {
        Ok(acc)
    } else {
        Err(E8Error::Overflow)
    }
}

/// Index of the codebook row nearest to `x` (Euclidean, exact `i128`), first index on ties. `x` is in
/// the codebook's own scale; equal to [`nearest_scaled`] with `scale = ONE`.
///
/// # Errors
/// [`E8Error::Overflow`] if a squared distance leaves `i128` (for example `x` near `i64::MIN`).
///
/// # Complexity
/// `O(256·8)` checked `i128` operations: an exhaustive scan of the codebook. `O(1)` in `x`, since the
/// codebook size is fixed.
pub fn nearest(x: &[i64; 8]) -> Result<usize, E8Error> {
    nearest_scaled(x, ONE)
}

/// Index of the codeword `j` minimising `|x − scale·C_j|²` (exact `i128` after per-coordinate
/// round-half-up of `scale·C_j`), first index on ties. `scale` is Q32.32 and must be positive.
///
/// # Errors
/// [`E8Error::NonPositiveScale`] if `scale <= 0`; [`E8Error::Overflow`] if a distance leaves `i128`.
///
/// # Complexity
/// `O(256·8)` checked `i128` operations: an exhaustive scan. Temporal: 256 rows × 8 coordinates
/// × (one multiply, one add, one subtract, one square). Spatial: `O(1)`.
pub fn nearest_scaled(x: &[i64; 8], scale: i64) -> Result<usize, E8Error> {
    if scale <= 0 {
        return Err(E8Error::NonPositiveScale);
    }
    let mut best: usize = 0;
    let mut best_d: i128 = 0;
    let mut ok: bool = true;
    let mut j: usize = 0;
    while j < CODEBOOK_LEN && ok {
        let row = codeword_unchecked(j);
        match scaled_dist2(x, &row, scale) {
            Err(_) => {
                ok = false;
            }
            Ok(d) => {
                if j == 0 || d < best_d {
                    best = j;
                    best_d = d;
                }
            }
        }
        j += 1;
    }
    if ok {
        Ok(best)
    } else {
        Err(E8Error::Overflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Doubled-integer E8 membership: `v/ONE` integral, all coordinates of one parity, sum `≡ 0 (mod 4)`.
    fn in_e8_doubled(v: &[i64; 8]) -> bool {
        let mut parity: i64 = 0;
        let mut sum: i64 = 0;
        let mut ok = true;
        for (k, &c) in v.iter().enumerate() {
            if c % ONE != 0 {
                return false;
            }
            let d = c / ONE;
            if k == 0 {
                parity = d.rem_euclid(2);
            } else if d.rem_euclid(2) != parity {
                ok = false;
            }
            sum += d;
        }
        ok && sum.rem_euclid(4) == 0
    }

    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// Exhaustive reference: first index of the minimal exact `i128` distance `Σ (x − C_j)²`.
    fn brute_nearest(x: &[i64; 8]) -> usize {
        let mut best = 0usize;
        let mut best_d: i128 = i128::MAX;
        for j in 0..CODEBOOK_LEN {
            let row = codeword(j).expect("row in range");
            let mut d: i128 = 0;
            for k in 0..8 {
                let e = i128::from(x[k]) - i128::from(row[k]);
                d += e * e;
            }
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        best
    }

    #[test]
    fn learn_code_e8_codebook_membership_and_shape() {
        let mut spinors = 0;
        let mut roots = 0;
        let mut o8 = 0;
        for j in 0..CODEBOOK_LEN {
            let row = codeword(j).expect("row in range");
            if j < 240 {
                assert!(in_e8_doubled(&row), "row {j} not in E8 (doubled)");
                if j < 128 {
                    spinors += 1;
                    assert!(row.iter().all(|&c| c == ONE || c == -ONE));
                    let minus = row.iter().filter(|&&c| c < 0).count();
                    assert_eq!(minus % 2, 0, "spinor row {j} odd minus count");
                } else {
                    roots += 1;
                    let nz = row.iter().filter(|&&c| c != 0).count();
                    assert_eq!(nz, 2, "root row {j}");
                    assert!(row.iter().all(|&c| c == 0 || c == 2 * ONE || c == -2 * ONE));
                }
            } else {
                o8 += 1;
                assert!(!in_e8_doubled(&row), "O8 row {j} must not be in E8");
                let nz = row.iter().filter(|&&c| c != 0).count();
                assert_eq!(nz, 1, "O8 row {j}");
                assert!(row
                    .iter()
                    .all(|&c| c == 0 || c == SQRT8_Q32 || c == NEG_SQRT8_Q32));
            }
        }
        assert_eq!((spinors, roots, o8), (128, 112, 16));
    }

    #[test]
    fn learn_code_e8_codewords_distinct() {
        for a in 0..CODEBOOK_LEN {
            for b in (a + 1)..CODEBOOK_LEN {
                assert_ne!(
                    codeword(a).expect("a"),
                    codeword(b).expect("b"),
                    "rows {a} and {b} coincide"
                );
            }
        }
    }

    #[test]
    fn learn_code_e8_sqrt8_q32_is_correct_rounding() {
        let s = i128::from(SQRT8_Q32);
        let four_x: i128 = 1_i128 << 69; // 4 · 8 · 2^64
        assert!((2 * s - 1) * (2 * s - 1) < four_x);
        assert!(four_x < (2 * s + 1) * (2 * s + 1));
    }

    #[test]
    fn learn_code_e8_nearest_roundtrip() {
        for i in 0..CODEBOOK_LEN {
            let c = codeword(i).expect("row in range");
            assert_eq!(nearest(&c), Ok(i), "row {i}");
        }
    }

    #[test]
    fn learn_code_e8_nearest_scaled_roundtrip() {
        for i in 0..CODEBOOK_LEN {
            let c = codeword(i).expect("row in range");
            let mut x = [0_i64; 8];
            for k in 0..8 {
                x[k] = 2 * c[k];
            }
            assert_eq!(nearest_scaled(&x, 2 * ONE), Ok(i), "row {i}");
        }
    }

    #[test]
    fn learn_code_e8_nearest_matches_exhaustive() {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut checked = 0usize;
        // 200 pseudo-random points, coordinates uniform in [−3, 3) in Q32.32.
        while checked < 200 {
            let mut x = [0_i64; 8];
            for k in 0..8 {
                let r = xorshift(&mut state) % (6 * ONE as u64);
                x[k] = r as i64 - 3 * ONE;
            }
            assert_eq!(nearest(&x), Ok(brute_nearest(&x)), "point {checked}");
            checked += 1;
        }
        // Boundary points: the origin (every row is at distance² 8 and the first index wins), and a
        // point on a codeword with a 1-ulp offset.
        let zero = [0_i64; 8];
        assert_eq!(nearest(&zero), Ok(0));
        assert_eq!(nearest(&zero), Ok(brute_nearest(&zero)));
        let mut near = codeword(200).expect("row");
        near[0] += 1;
        assert_eq!(nearest(&near), Ok(brute_nearest(&near)));
    }

    #[test]
    fn learn_code_e8_refusals() {
        assert_eq!(codeword(CODEBOOK_LEN), Err(E8Error::IndexOutOfRange));
        let x = [0_i64; 8];
        assert_eq!(nearest_scaled(&x, 0), Err(E8Error::NonPositiveScale));
        assert_eq!(nearest_scaled(&x, -1), Err(E8Error::NonPositiveScale));
        assert_eq!(nearest(&[i64::MIN; 8]), Err(E8Error::Overflow));
    }
}
