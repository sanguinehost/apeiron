//! `RoPE`: the load-time frequency table and the per-token integer rotation (port-map row 3).
//!
//! Each rotary dimension `j` carries a turn frequency `turns[j]`, the `u64` nearest to
//! `theta^(-2j/head_dim) / (2pi)` in `Z/2^64` (the fraction of a full turn per position). The
//! per-token phase is `(turns[j] * pos) mod 2^64`, exact wrapping arithmetic with no radian
//! window: the signed lift [`turn_to_radians`] maps it into the `fixed_sin_cos` window for every
//! `u32` position, so the path admits any position the KV cache can hold.

use super::value::round_shift_32;
use crate::numeric::fixed_trig::{fixed_sin_cos, turn_to_radians};

/// A refused `RoPE` operation (port-map row 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeError {
    /// `head_dim` is zero, odd, or less than two, so the rotary pair `(j, j + head_dim/2)` is
    /// undefined.
    BadHeadDim,
    /// The frequency table length is not `head_dim / 2`.
    TableLength,
    /// The `q` and `k` vectors differ in length, or that length is not a multiple of `head_dim`.
    VectorLength,
    /// `rope_theta` is not a finite positive real, so no frequency is defined.
    BadTheta,
    /// A turn frequency is not finite or reaches `2^64`, or a rotated component left its admitted
    /// `i64` fixed-point range.
    OutOfRange,
}

/// The load-time `RoPE` turn table (port-map row 3).
///
/// `turns[j]` is `theta^(-2j/head_dim) / (2pi)` in `Z/2^64` for `j < head_dim / 2`, the fraction of
/// a full turn per position; the table is position-independent and is reused across every layer
/// and token. The per-token rotation is integer-only; see [`RopeTable::rotate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RopeTable {
    head_dim: usize,
    turns: Vec<u64>,
}

impl RopeTable {
    /// Build the turn table from `rope_theta` and `head_dim`.
    ///
    /// `turns[j] = round(theta^(-2j/head_dim) / (2pi) * 2^64)`, computed once at load with the
    /// host `f64` `powf` and rounded to nearest as a `u64` phase in `Z/2^64`. This is the row-3
    /// load-time float boundary, named in the module trust-boundary statement; every per-token
    /// operation below is integer.
    ///
    /// # Errors
    ///
    /// Returns [`RopeError::BadHeadDim`] for zero, odd or unit `head_dim`,
    /// [`RopeError::BadTheta`] for a non-finite or non-positive `theta`, and
    /// [`RopeError::OutOfRange`] when a turn is non-finite or reaches `2^64`.
    ///
    /// # Complexity
    ///
    /// `O(head_dim)` time and space: one `powf` and one rounding per rotary dimension.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the host f64 powf/round boundary is declared; the range check above `as u64` \
                  keeps the cast in the u64 phase domain, and `j`/`head_dim` are tiny"
    )]
    pub fn new(theta: f64, head_dim: usize) -> Result<RopeTable, RopeError> {
        if head_dim < 2 || !head_dim.is_multiple_of(2) {
            return Err(RopeError::BadHeadDim);
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(RopeError::BadTheta);
        }
        let n = head_dim / 2;
        let hd = head_dim as f64;
        let unit = 18_446_744_073_709_551_616.0_f64; // 2^64: one full turn in phases
        let two_pi = 2.0 * core::f64::consts::PI;
        let mut turns: Vec<u64> = Vec::with_capacity(n);
        let mut j = 0usize;
        while j < n {
            let exponent = -2.0 * (j as f64) / hd;
            let q = (theta.powf(exponent) * unit / two_pi).round();
            if !(q.is_finite() && (0.0..unit).contains(&q)) {
                return Err(RopeError::OutOfRange);
            }
            turns.push(q as u64);
            j += 1;
        }
        Ok(RopeTable { head_dim, turns })
    }

    /// Build the table from an explicit turn vector.
    ///
    /// This is the integer entry point for the same table: a caller that builds the turn
    /// frequencies on its own fixed-point path can bypass [`RopeTable::new`]'s load-time float
    /// boundary. Every `u64` is a valid phase; no value is clamped.
    ///
    /// # Errors
    ///
    /// Returns [`RopeError::BadHeadDim`] for zero, odd or unit `head_dim`, and
    /// [`RopeError::TableLength`] when `turns.len() != head_dim / 2`.
    pub fn from_turns(head_dim: usize, turns: Vec<u64>) -> Result<RopeTable, RopeError> {
        if head_dim < 2 || !head_dim.is_multiple_of(2) {
            return Err(RopeError::BadHeadDim);
        }
        if turns.len() != head_dim / 2 {
            return Err(RopeError::TableLength);
        }
        Ok(RopeTable { head_dim, turns })
    }

    /// The rotary head width this table is built for.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The turn frequencies, `head_dim / 2` of them, fastest rotary dimension first.
    #[must_use]
    pub fn turns(&self) -> &[u64] {
        &self.turns
    }

    /// Rotate `q` and `k` in place at token position `pos`.
    ///
    /// `q` and `k` hold the same number of `head_dim`-wide heads; each head rotates the pair
    /// `(j, j + head_dim/2)` by the angle `turn_to_radians((turns[j] * pos) mod 2^64)`. The phase
    /// product wraps exactly in `Z/2^64`, and each output component is
    /// `round_shift_32(x0 * cos - x1 * sin)` and `round_shift_32(x1 * cos + x0 * sin)`, the
    /// value-path round-to-nearest epilogue.
    ///
    /// # Errors
    ///
    /// Returns [`RopeError::VectorLength`] when `q` and `k` differ in length or that length is not
    /// a multiple of `head_dim`, and [`RopeError::OutOfRange`] when a rotated component does not
    /// fit `i64`. No input is clamped.
    ///
    /// # Complexity
    ///
    /// `O(q.len())` time in multiplications and trig calls, `O(1)` extra space per pair: the
    /// rotation is in place and one `fixed_sin_cos` evaluates a pair.
    pub fn rotate(&self, q: &mut [i64], k: &mut [i64], pos: u32) -> Result<(), RopeError> {
        let hd = self.head_dim;
        if q.len() != k.len() || !q.len().is_multiple_of(hd) {
            return Err(RopeError::VectorLength);
        }
        let heads = q.len() / hd;
        let mut h = 0usize;
        // No `return` inside the loop: the pinned Charon/Aeneas fragment rejects an early return in
        // one of two sequential loops. The mutation order and the propagated refusal are the same.
        let mut ok = true;
        while h < heads && ok {
            let base = h * hd;
            match self.rotate_head(&mut q[base..base + hd], pos) {
                Ok(()) => match self.rotate_head(&mut k[base..base + hd], pos) {
                    Ok(()) => {}
                    Err(_) => ok = false,
                },
                Err(_) => ok = false,
            }
            h += 1;
        }
        // `rotate_head` constructs only [`RopeError::OutOfRange`], so this is the error it propagates.
        if !ok {
            return Err(RopeError::OutOfRange);
        }
        Ok(())
    }

    /// Rotate a single `head_dim`-multiple vector in place at token position `pos`.
    ///
    /// The decode loop rotates Q (`n_head` heads) and K (`n_head_kv` heads) separately, so the
    /// two vectors are different lengths; this is [`RopeTable::rotate`]'s per-vector arm with the
    /// same angle product, trig call and round-to-nearest epilogue.
    ///
    /// # Errors
    ///
    /// Returns [`RopeError::VectorLength`] when `x.len()` is not a multiple of `head_dim`, and
    /// [`RopeError::OutOfRange`] with the same reach as [`RopeTable::rotate`].
    ///
    /// # Complexity
    ///
    /// `O(x.len())` time in multiplications and trig calls, `O(1)` extra space.
    pub fn rotate_one(&self, x: &mut [i64], pos: u32) -> Result<(), RopeError> {
        let hd = self.head_dim;
        if !x.len().is_multiple_of(hd) {
            return Err(RopeError::VectorLength);
        }
        let heads = x.len() / hd;
        let mut h = 0usize;
        // No `return` inside the loop; see [`RopeTable::rotate`].
        let mut ok = true;
        while h < heads && ok {
            let base = h * hd;
            match self.rotate_head(&mut x[base..base + hd], pos) {
                Ok(()) => {}
                Err(_) => ok = false,
            }
            h += 1;
        }
        if !ok {
            return Err(RopeError::OutOfRange);
        }
        Ok(())
    }

    /// Rotate one `head_dim`-wide head in place.
    ///
    /// The phase is `(turns[j] * pos) mod 2^64`, the exact product in `Z/2^64`; [`turn_to_radians`]
    /// then maps it into the [`fixed_sin_cos`] window. No `return` appears inside the loop, the
    /// pair write order and round-to-nearest epilogue are unchanged, and the only refusal left is a
    /// rotated component outside `i64`.
    #[allow(
        clippy::cast_lossless,
        clippy::cast_possible_truncation,
        reason = "plain i128 comparisons and `as` casts after an explicit range check extract \
                  through Charon/Aeneas, while TryFrom and a return inside a loop emit an unmodeled \
                  external or an unsupported control-flow pattern the fail-closed audit rejects"
    )]
    fn rotate_head(&self, x: &mut [i64], pos: u32) -> Result<(), RopeError> {
        let d2 = self.head_dim / 2;
        let mut j = 0usize;
        let mut ok = true;
        while j < d2 && ok {
            let phase = ((self.turns[j] as u128 * pos as u128) & u64::MAX as u128) as u64;
            match fixed_sin_cos(turn_to_radians(phase)) {
                Ok(sc) => {
                    let x0 = x[j];
                    let x1 = x[j + d2];
                    let p00 = i128::from(x0) * i128::from(sc.cos);
                    let p01 = i128::from(x1) * i128::from(sc.sin);
                    let p10 = i128::from(x1) * i128::from(sc.cos);
                    let p11 = i128::from(x0) * i128::from(sc.sin);
                    match p00.checked_sub(p01) {
                        Some(re) => match p10.checked_add(p11) {
                            Some(im) => {
                                let out0 = round_shift_32(re);
                                let out1 = round_shift_32(im);
                                if out0 > i64::MAX as i128
                                    || out0 < i64::MIN as i128
                                    || out1 > i64::MAX as i128
                                    || out1 < i64::MIN as i128
                                {
                                    ok = false;
                                } else {
                                    x[j] = out0 as i64;
                                    x[j + d2] = out1 as i64;
                                }
                            }
                            None => ok = false,
                        },
                        None => ok = false,
                    }
                }
                Err(_) => ok = false,
            }
            j += 1;
        }
        if !ok {
            return Err(RopeError::OutOfRange);
        }
        Ok(())
    }
}

/// Rotate `q` and `k` in place with a load-time [`RopeTable`] (port-map row 3).
///
/// The shape the decode loop calls per token; [`RopeTable::rotate`] holds the implementation.
///
/// # Errors
///
/// As [`RopeTable::rotate`].
pub fn rope_rotate_fixed(
    q: &mut [i64],
    k: &mut [i64],
    pos: u32,
    table: &RopeTable,
) -> Result<(), RopeError> {
    table.rotate(q, k, pos)
}
