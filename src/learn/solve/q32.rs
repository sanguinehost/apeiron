//! The `f64` ↔ Q32.32 `i64` boundary and the pure-integer carriers the solve primitives share.
//!
//! A real `x` is the grid word `round(x·2³²)` and a word `v` is `v/2³²`. Every caller that hands an
//! `f64` fold quantity to [`super::chol`], [`super::syrk`] or [`super::gemm`] converts here, so the
//! range test, the rounding and the power-of-two scaling rule exist once.
//!
//! The integer carriers below ([`Fx`], [`Bfp`], [`SpdQ`]) and their primitives ([`ilog2_pos`],
//! [`rshift_rne`], [`round_q32_even`], [`round_f16`]) are pure integer: no `f32`/`f64`, no
//! `unsafe`, no `unwrap`, and only the shift and checked-op forms the pinned Aeneas fragment models.
//! They carry the `f64` fold-fit quantities that later lanes convert to `Q32.32`; the `f64`
//! boundary functions above are the older grid map.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::comparison_chain,
    clippy::implicit_saturating_sub,
    clippy::manual_let_else,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "the closure-free `match`/`if` forms are what the pinned Charon/Aeneas extraction \
              fragment models, so `match o { Some(x) => x, None => return .. }` stays in place of \
              `let .. else`, the sign chain in `scale_q32` stays in place of a `cmp`, and the \
              `if bits > 62` guard stays in place of `saturating_sub`; `Fx` is deliberately \
              `Clone`-not-`Copy` (the carrier spec fixes the derives), so `round_f16` takes it by \
              value; the `f64 as i64` in `to_grid` is the Q32.32 boundary rounding, range-checked \
              immediately before it; the `f64 as i32` in `panel_shift` and `fill_scale` is an \
              exponent clamped to a small range; the `i64 as f64` in `from_grid` is the \
              documented grid-to-real map; in the integer carriers every `as` is range-checked by \
              construction (`u32 as i32` exponents are <= 127, `i64 as u128` magnitudes are \
              positive, `i128 as i64` narrowings follow a checked bound), the single-letter names \
              are the significand/exponent/quotient/remainder symbols of the shift recurrence, and \
              the row-major loops index by `i/j`"
)]

/// The `Q32.32` scale `2³²`: the grid word of a real `x` is `round(x·2³²)` and a word `v` is `v/2³²`.
pub const GRID_SCALE: f64 = 4_294_967_296.0;

/// The `wsyrk` input bound in the reals: the primitive's documented bound is `|v| < 2³⁶` in raw
/// `Q32.32`, i.e. `2³⁶/2³² = 16` on the `f64` side.
pub const INPUT_BOUND: f64 = 16.0;

/// The grid's own real range bound `2³¹`: `|x| < 2³¹` keeps `round(x·2³²)` inside `i64`. The bound
/// for a value the primitives consume without a tighter input bound — a metric factor or a
/// [`super::chol::chol_solve`] operand.
pub const GRID_BOUND: f64 = 2_147_483_648.0;

/// `round(x·2³²)` as a `Q32.32` word, or `None` when `x` is not finite or `|x|` reaches `bound`.
/// `bound` is [`INPUT_BOUND`] for a `wsyrk` input and [`GRID_BOUND`] for a value the solve consumes.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn to_grid(x: f64, bound: f64) -> Option<i64> {
    let q = x * GRID_SCALE;
    if !x.is_finite() || x.abs() >= bound || q.abs() >= 9_223_372_036_854_775_808.0 {
        return None;
    }
    Some(q.round() as i64)
}

/// `v/2³²`, the `Q32.32` word as an `f64`.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn from_grid(v: i64) -> f64 {
    v as f64 / GRID_SCALE
}

/// The least `k ≥ 0` with `max·2⁻ᵏ ≤ 8` — the exact power-of-two shift that brings a panel below the
/// `wsyrk` input bound `16` with head-room; `k = 0` when `max ≤ 8`.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn panel_shift(max: f64) -> i32 {
    if max <= 8.0 {
        0
    } else {
        (max / 8.0).log2().ceil() as i32
    }
}

/// `2^e` for an integer `e` (exact, by the IEEE exponent field), or `None` outside `f64`'s finite
/// normal range `[-1022, 1023]`.
fn pow2(e: i32) -> Option<f64> {
    if (-1022..=1023).contains(&e) {
        Some(f64::from_bits(u64::try_from(e + 1023).ok()? << 52))
    } else {
        None
    }
}

/// The exact power of two `s` that scales a magnitude `max_mag` into `(2³⁰, 2³¹)`, the fill of the
/// grid range, so the scaled entries keep the most significant bits available; `1.0` when
/// `max_mag ≤ 0`. `max_mag` is a finite non-negative magnitude.
///
/// Two operands filled by separate powers of two `s_a`, `s_b` solve to `X_q = (s_b/s_a)·X`, which
/// lands the quotient near the top of the grid range rather than at the bottom; the caller undoes it
/// as `X = (s_a/s_b)·X_q`.
///
/// The `log2` estimate is corrected by exact halvings and doublings; the exponent is capped so
/// `s·2³²` stays a finite `f64`, which under-fills only for a magnitude below `2⁻⁸⁶⁹`.
///
/// # Complexity
///
/// `O(1)`: the correction loops run at most twice after the `log2` estimate.
#[must_use]
pub fn fill_scale(max_mag: f64) -> f64 {
    const MAX_VALUE: f64 = 2_147_483_647.0;
    if max_mag <= 0.0 {
        return 1.0;
    }
    let estimate = (MAX_VALUE / max_mag).log2();
    let exponent = if estimate.is_finite() {
        estimate.clamp(-1022.0, 900.0).floor() as i32
    } else {
        900
    };
    let mut scale = pow2(exponent).unwrap_or(1.0);
    while max_mag * scale > MAX_VALUE {
        scale *= 0.5;
    }
    while scale < 1.0e270 && max_mag * (scale * 2.0) <= MAX_VALUE {
        scale *= 2.0;
    }
    scale
}

/// A refusal from an integer-carrier conversion, scaling or shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q32Error {
    /// A checked shift or `i128 → i64` narrowing left the target integer range.
    Overflow,
    /// A [`SpdQ`] input has a non-positive diagonal entry, so its base-2 logarithm is undefined.
    NotSpd,
    /// A [`SpdQ`] input is not symmetric: `h[i·n + j] != h[j·n + i]`.
    Asymmetric,
    /// A slice length disagrees with the declared shape.
    ShapeMismatch,
}

/// `floor(log2 x)`, the index of the highest set bit of `x`, or `None` for `x == 0`.
///
/// # Complexity
///
/// `O(1)`: at most 128 halvings, the width of `u128`.
#[must_use]
pub fn ilog2_pos(x: u128) -> Option<u32> {
    if x == 0 {
        return None;
    }
    let mut v = x;
    let mut n = 0_u32;
    while v > 1 {
        v >>= 1;
        n += 1;
    }
    Some(n)
}

/// `v / 2^s` rounded to nearest, ties to even, or `None` when `s > 126`.
///
/// With `q = v / 2^s` (truncating toward zero) and `r = v mod 2^s` the signed remainder, the result
/// is `q + step`: `step` is `sign(r)` when `|r| > 2^(s−1)`, `0` when `|r| < 2^(s−1)`, and on the tie
/// `|r| = 2^(s−1)` it is `sign(r)` exactly when `q` is odd, so the even neighbour wins. `s > 126`
/// is refused because `2^s` would leave `i128`'s magnitude for a nonzero quotient.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn rshift_rne(v: i128, s: u32) -> Option<i128> {
    if s > 126 {
        return None;
    }
    if s == 0 {
        return Some(v);
    }
    let d: i128 = 1_i128 << s;
    let half: i128 = 1_i128 << (s - 1);
    let q = v / d;
    let r = v % d;
    if r == 0 {
        return Some(q);
    }
    let mag = if r > 0 { r } else { -r };
    let step: i128 = if mag > half {
        if r > 0 {
            1
        } else {
            -1
        }
    } else if mag < half || q % 2 == 0 {
        0
    } else if r > 0 {
        1
    } else {
        -1
    };
    Some(q + step)
}

/// `v` as `i64`, or `None` when it is outside `i64`.
fn narrow_i64(v: i128) -> Option<i64> {
    if v > i128::from(i64::MAX) || v < i128::from(i64::MIN) {
        None
    } else {
        Some(v as i64)
    }
}

/// `|v|` as a `u128` (the two's-complement negation for `v < 0`, `i128::MIN` included).
fn magnitude_i128(v: i128) -> u128 {
    if v < 0 {
        !(v as u128) + 1
    } else {
        v as u128
    }
}

/// [`rshift_rne`] of `v` by 32 then an `i128 → i64` narrowing: the Q64.64 value `v` as a Q32.32
/// word, rounded to nearest, ties to even.
///
/// # Errors
///
/// [`Q32Error::Overflow`] when the rounded value is outside `[i64::MIN, i64::MAX]` (the shift is a
/// constant 32, so only the narrowing refuses).
///
/// # Complexity
///
/// `O(1)`.
pub fn round_q32_even(v: i128) -> Result<i64, Q32Error> {
    match rshift_rne(v, 32) {
        Some(r) => narrow_i64(r).ok_or(Q32Error::Overflow),
        None => Err(Q32Error::Overflow),
    }
}

/// `num / den` rounded to nearest, ties to even, or `None` when `den <= 0` or the doubled remainder
/// leaves `i128`.
///
/// The one rounded quotient of the `Q32.32` path: a `Q32.32` division is `div_rne(num << 32, den)`
/// followed by one narrowing. The body is [`crate::numeric::fixed_ln::div_rne`] — the extraction
/// unit compiles `numeric` standalone (`proof/extract/lib.rs`), so the quotient is defined there and
/// re-exported here as the shared name the fold and solve code calls.
pub use crate::numeric::fixed_ln::div_rne;

/// A scaled-integer real `m · 2^e`: the scalar carrier a `f64` fold-fit quantity moves to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fx {
    /// The integer significand `m`; `|m| < 2^62` after [`Fx::from_i128`].
    pub m: i64,
    /// The base-2 exponent `e`.
    pub e: i32,
}

impl Fx {
    /// `v · 2^e` as an [`Fx`].
    ///
    /// A nonzero `v` that needs no more than 62 bits keeps `v` as the significand and `e` as the
    /// exponent. A wider `v` is right-shifted by `bit_length(v) − 62` with round-to-nearest, ties to
    /// even, and the exponent is raised by the same shift, so the value is preserved up to the
    /// rounding; a rounding that would reach `2^62` shifts one bit further so `|m| < 2^62` holds.
    /// Zero is canonicalised to `{ m: 0, e: 0 }`.
    ///
    /// # Errors
    ///
    /// [`Q32Error::Overflow`] when the narrowed significand leaves `i64` (unreachable by the
    /// 62-bit-bounded shift; kept for totality).
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    pub fn from_i128(v: i128, e: i32) -> Result<Fx, Q32Error> {
        if v == 0 {
            return Ok(Fx { m: 0, e: 0 });
        }
        let bits = match ilog2_pos(magnitude_i128(v)) {
            Some(b) => b + 1,
            None => return Err(Q32Error::Overflow),
        };
        let s = if bits > 62 { bits - 62 } else { 0 };
        let m = match rshift_rne(v, s) {
            Some(x) => x,
            None => return Err(Q32Error::Overflow),
        };
        let bound: i128 = 1_i128 << 62;
        if m >= bound || m <= -bound {
            let s2 = s + 1;
            let m2 = match rshift_rne(v, s2) {
                Some(x) => x,
                None => return Err(Q32Error::Overflow),
            };
            let mm = match narrow_i64(m2) {
                Some(x) => x,
                None => return Err(Q32Error::Overflow),
            };
            return Ok(Fx {
                m: mm,
                e: e + s2 as i32,
            });
        }
        let mm = match narrow_i64(m) {
            Some(x) => x,
            None => return Err(Q32Error::Overflow),
        };
        Ok(Fx {
            m: mm,
            e: e + s as i32,
        })
    }
}

/// A block floating-point vector `m[i] · 2^e`: one base-2 exponent shared by the whole block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bfp {
    /// The integer significands; `max |m[i]| ∈ [2^34, 2^35)` after [`Bfp::from_i128`] unless all
    /// inputs are zero.
    pub m: Vec<i64>,
    /// The shared base-2 exponent `e`.
    pub e: i32,
}

impl Bfp {
    /// The block floating-point form of `v[i] · 2^e` with `max |m[i]| ∈ [2^34, 2^35)`.
    ///
    /// The common shift follows the maximum magnitude `max |v[i]|`: when it needs more than 35
    /// bits the integers are right-shifted by `bit_length(max) − 35` with round-to-nearest, ties to
    /// even (one bit deeper if the rounded maximum would reach `2^35`, so the upper bound holds);
    /// when it needs fewer they are left-shifted exactly by `35 − bit_length(max)`. The exponent is
    /// raised by a right shift and left unchanged by a left shift. All-zero (or empty) input
    /// returns zeros with `e` unchanged.
    ///
    /// # Errors
    ///
    /// [`Q32Error::Overflow`] when a checked shift or narrowing leaves `i64` (unreachable by the
    /// shift choice; kept for totality).
    ///
    /// # Complexity
    ///
    /// Time `O(n)`, space `O(n)` for the output: the maximum scan and the shift pass each visit the
    /// `n` inputs once.
    pub fn from_i128(v: &[i128], e: i32) -> Result<Bfp, Q32Error> {
        let n = v.len();
        let mut max_mag: u128 = 0;
        let mut max_val: i128 = 0;
        let mut i = 0_usize;
        while i < n {
            let mag = magnitude_i128(v[i]);
            if mag > max_mag {
                max_mag = mag;
                max_val = v[i];
            }
            i += 1;
        }
        if max_mag == 0 {
            return Ok(Bfp {
                m: vec![0_i64; n],
                e,
            });
        }
        let bits = match ilog2_pos(max_mag) {
            Some(b) => b + 1,
            None => return Err(Q32Error::Overflow),
        };
        let mut out = vec![0_i64; n];
        if bits <= 35 {
            let k = 35 - bits;
            let mut i = 0_usize;
            while i < n {
                out[i] = (v[i] << k) as i64;
                i += 1;
            }
            return Ok(Bfp { m: out, e });
        }
        let mut s = bits - 35;
        let bound: i128 = 1_i128 << 35;
        let rounded_max = match rshift_rne(max_val, s) {
            Some(x) => x,
            None => return Err(Q32Error::Overflow),
        };
        if rounded_max >= bound || rounded_max <= -bound {
            s += 1;
        }
        let mut err: Option<Q32Error> = None;
        let mut i = 0_usize;
        while i < n && err.is_none() {
            match rshift_rne(v[i], s) {
                Some(x) => out[i] = x as i64,
                None => err = Some(Q32Error::Overflow),
            }
            i += 1;
        }
        if let Some(er) = err {
            return Err(er);
        }
        Ok(Bfp {
            m: out,
            e: e + s as i32,
        })
    }
}

/// `v / 2^shift` (a right shift, round-to-nearest ties-to-even, when `shift > 0`), `v · 2^(−shift)`
/// (an exact checked left shift when `shift < 0`) or the identity word when `shift == 0`.
fn scale_q32(v: i64, shift: i32) -> Result<i64, Q32Error> {
    if shift > 0 {
        match rshift_rne(i128::from(v), shift as u32) {
            Some(x) => narrow_i64(x).ok_or(Q32Error::Overflow),
            None => Err(Q32Error::Overflow),
        }
    } else if shift < 0 {
        let k = -shift;
        if k > 64 {
            return Err(Q32Error::Overflow);
        }
        narrow_i64(i128::from(v) << k).ok_or(Q32Error::Overflow)
    } else {
        Ok(v)
    }
}

/// A diagonal-equilibrated symmetric positive-definite matrix in Q32.32:
/// `H = diag(2^d) · hat · diag(2^d)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpdQ {
    /// The equilibrated row-major `[n, n]` Q32.32 matrix `hat`; `hat[i][i] ∈ [2^32, 2^34)`.
    pub hat: Vec<i64>,
    /// The per-row base-2 exponents `d`, length `n`.
    pub d: Vec<i32>,
}

impl SpdQ {
    /// Equilibrate the symmetric positive-definite Q32.32 matrix `h` (row-major `[n, n]`, a word
    /// `w` denoting the real `w / 2^32`) into an [`SpdQ`].
    ///
    /// `d[i] = floor((bit_length(h[i][i]) − 1 − 32) / 2)`, so `hat[i][i] = h[i][i] / 2^(2·d[i])`
    /// is the real value in `[1, 4)` (words in `[2^32, 2^34)`), and
    /// `hat[i][j] = h[i][j] / 2^(d[i]+d[j])` — a right shift, round-to-nearest ties-to-even, when
    /// `d[i]+d[j] > 0`, an exact checked left shift when `d[i]+d[j] < 0`, and the identity word when
    /// the sum is zero. Only power-of-two shifts are used, so `H = diag(2^d) · hat · diag(2^d)`
    /// holds up to the rounding of the off-diagonals.
    ///
    /// # Errors
    ///
    /// [`Q32Error::ShapeMismatch`] when `n == 0`, or `n·n` leaves `usize`, or `h.len() != n·n`;
    /// [`Q32Error::Asymmetric`] when some `h[i][j] != h[j][i]`;
    /// [`Q32Error::NotSpd`] when some diagonal entry is below `1` (its logarithm is undefined);
    /// [`Q32Error::Overflow`] when a checked left shift leaves `i64`.
    ///
    /// # Complexity
    ///
    /// Time `O(n²)`, space `O(n²)` for `hat` plus `O(n)` for `d`: the symmetry scan and the shift
    /// pass each visit the `n²` entries once.
    pub fn from_q32(h: &[i64], n: usize) -> Result<SpdQ, Q32Error> {
        if n == 0 {
            return Err(Q32Error::ShapeMismatch);
        }
        let nn = match n.checked_mul(n) {
            Some(v) => v,
            None => return Err(Q32Error::ShapeMismatch),
        };
        if h.len() != nn {
            return Err(Q32Error::ShapeMismatch);
        }
        let mut symmetric = true;
        let mut idx = 0_usize;
        while idx < nn {
            let i = idx / n;
            let j = idx % n;
            if i < j && h[idx] != h[j * n + i] {
                symmetric = false;
            }
            idx += 1;
        }
        if !symmetric {
            return Err(Q32Error::Asymmetric);
        }
        let mut d = vec![0_i32; n];
        let mut err: Option<Q32Error> = None;
        let mut i = 0_usize;
        while i < n && err.is_none() {
            let hii = h[i * n + i];
            if hii < 1 {
                err = Some(Q32Error::NotSpd);
            } else {
                match ilog2_pos(hii as u128) {
                    Some(bl) => d[i] = (bl as i32 - 32) >> 1,
                    None => err = Some(Q32Error::NotSpd),
                }
            }
            i += 1;
        }
        if let Some(er) = err {
            return Err(er);
        }
        let mut hat = vec![0_i64; nn];
        let mut err: Option<Q32Error> = None;
        let mut idx = 0_usize;
        while idx < nn && err.is_none() {
            let i = idx / n;
            let j = idx % n;
            match scale_q32(h[idx], d[i] + d[j]) {
                Ok(x) => hat[idx] = x,
                Err(er) => err = Some(er),
            }
            idx += 1;
        }
        if let Some(er) = err {
            return Err(er);
        }
        Ok(SpdQ { hat, d })
    }

    /// Invert [`SpdQ::from_q32`]: recover the Q32.32 symmetric matrix `h` with
    /// `h[i][j] = hat[i][j] · 2^(d[i]+d[j])` (a checked left shift when the sum is positive, a
    /// round-to-nearest ties-to-even right shift when it is negative, the identity at zero).
    ///
    /// # Errors
    ///
    /// [`Q32Error::ShapeMismatch`] when `hat.len() != d.len()²` (or `d.len()²` leaves `usize`);
    /// [`Q32Error::Overflow`] when a checked left shift leaves `i64`.
    ///
    /// # Complexity
    ///
    /// Time `O(n²)`, space `O(n²)` for the output.
    pub fn to_q32(&self) -> Result<Vec<i64>, Q32Error> {
        let n = self.d.len();
        let nn = match n.checked_mul(n) {
            Some(v) => v,
            None => return Err(Q32Error::ShapeMismatch),
        };
        if self.hat.len() != nn {
            return Err(Q32Error::ShapeMismatch);
        }
        let mut h = vec![0_i64; nn];
        let mut err: Option<Q32Error> = None;
        let mut idx = 0_usize;
        while idx < nn && err.is_none() {
            let i = idx / n;
            let j = idx % n;
            match scale_q32(self.hat[idx], -(self.d[i] + self.d[j])) {
                Ok(x) => h[idx] = x,
                Err(er) => err = Some(er),
            }
            idx += 1;
        }
        if let Some(er) = err {
            return Err(er);
        }
        Ok(h)
    }
}

/// The IEEE-754 binary16 round-to-nearest-even of the real `x.m · 2^(x.e)`, as an [`Fx`].
///
/// With `E = floor(log2 |x.m|) + x.e`, the quantum is `q = −24` on the subnormal range `E < −14`
/// and `q = E − 10` otherwise; the significand is `r = round(x.m · 2^(x.e−q))` (round-to-nearest,
/// ties to even; an exact left shift when `x.e ≥ q`), and the result is `{ m: r, e: q }`. A rounded
/// magnitude of `2^16` or more is [`Q32Error::Overflow`] (binary16 infinity). Zero is `{ m: 0, e: 0 }`.
///
/// This differs from [`crate::learn::fold::fit::quant::f16r`] — which narrows `f64 → f32 → f16`,
/// two roundings — only on inputs needing more than 24 significant bits; here the rounding is a
/// single step from the exact integer value.
///
/// # Errors
///
/// [`Q32Error::Overflow`] when the rounded magnitude reaches `2^16`.
///
/// # Complexity
///
/// `O(1)`.
pub fn round_f16(x: Fx) -> Result<Fx, Q32Error> {
    let m = x.m;
    let e = x.e;
    if m == 0 {
        return Ok(Fx { m: 0, e: 0 });
    }
    let il = match ilog2_pos(magnitude_i128(i128::from(m))) {
        Some(b) => b,
        None => return Err(Q32Error::Overflow),
    };
    let big_e = il as i32 + e;
    let q: i32 = if big_e < -14 { -24 } else { big_e - 10 };
    let shift = e - q;
    let r128: i128 = if shift >= 0 {
        i128::from(m) << shift
    } else {
        let s = -shift;
        if s > 126 {
            return Err(Q32Error::Overflow);
        }
        match rshift_rne(i128::from(m), s as u32) {
            Some(v) => v,
            None => return Err(Q32Error::Overflow),
        }
    };
    if big_e >= 16 {
        return Err(Q32Error::Overflow);
    }
    if big_e == 15 {
        let rm = if r128 < 0 { -r128 } else { r128 };
        if rm >= 2048 {
            return Err(Q32Error::Overflow);
        }
    }
    let r = match narrow_i64(r128) {
        Some(v) => v,
        None => return Err(Q32Error::Overflow),
    };
    Ok(Fx { m: r, e: q })
}
