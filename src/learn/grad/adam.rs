//! Adam on Q32.32 parameter vectors: the CPU oracle of the device update
//! `gpu::nn::bwd::opt::adam_update_dispatch` the KD step runs.
//!
//! Torch defaults (`beta = (0.9, 0.999)`, `eps = 1e-8`, no weight decay). The first moment is
//! Q32.32; the second is `Q64.64` in `i128`, because `v = (1−β₂)·g²` squares the gradient and at
//! Q32.32 a gradient below `2^-16` would round `v` to zero. Every product is rounded once, ties to
//! even ([`round_q32_even`]); every narrowing is checked.

use crate::learn::solve::q32::{narrow, round_q32_even, rshift_rne};
use crate::numeric::fixed_rsqrt::isqrt;

/// The Q32.32 unit: the word `1 << 32` denotes `1.0`.
const ONE: i64 = 1 << 32;

/// Adam `beta_1` (torch default), `round(0.9·2^32)`.
const BETA1_Q32: i64 = 3_865_470_566;

/// `1 − beta_1`, `round(0.1·2^32)`; with [`BETA1_Q32`] it sums to exactly `2^32`.
const ONE_MINUS_BETA1_Q32: i64 = 429_496_730;

/// Adam `beta_2` (torch default), `round(0.999·2^32)`.
const BETA2_Q32: i64 = 4_290_672_329;

/// `1 − beta_2`, `round(0.001·2^32)`.
const ONE_MINUS_BETA2_Q32: i64 = 4_294_967;

/// Adam `eps` (torch default), `round(1e-8·2^32) = 43`.
const ADAM_EPS_Q32: i64 = 43;

/// A refused Adam update.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdamError {
    /// The parameter, gradient and state widths disagree.
    Shape(String),
    /// A checked word left its carrier.
    Range(String),
}

impl core::fmt::Display for AdamError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Shape(s) => write!(f, "adam shape: {s}"),
            Self::Range(s) => write!(f, "adam range: {s}"),
        }
    }
}

impl std::error::Error for AdamError {}

/// `a·b` in Q32.32, ties to even, or `None` outside `i64`. `O(1)`.
fn mul(a: i64, b: i64) -> Option<i64> {
    round_q32_even(i128::from(a) * i128::from(b)).ok()
}

/// `round(n / d)`, ties to even, or `None` when `d == 0` or the result leaves `i128`. `O(1)`.
fn div_ne(n: i128, d: i128) -> Option<i128> {
    if d == 0 {
        return None;
    }
    let (q, r) = (n / d, n % d);
    if r == 0 {
        return Some(q);
    }
    let up = if r.abs().checked_mul(2)? > d.abs() || (r.abs() * 2 == d.abs() && q % 2 != 0) {
        if (r < 0) == (d < 0) {
            1
        } else {
            -1
        }
    } else {
        0
    };
    q.checked_add(up)
}

/// `round(a / b)` as a Q32.32 word, or `None` when `b == 0` or the result leaves `i64`. `O(1)`.
fn div(a: i64, b: i64) -> Option<i64> {
    narrow(div_ne(i128::from(a) << 32, i128::from(b))?)
}

/// `base^e` in Q32.32 by exponentiation by squaring; `O(log e)` multiplications.
fn pow_q32(base: i64, mut e: u64) -> Option<i64> {
    let mut acc = ONE;
    let mut b = base;
    while e > 0 {
        if e & 1 == 1 {
            acc = mul(acc, b)?;
        }
        e >>= 1;
        if e > 0 {
            b = mul(b, b)?;
        }
    }
    Some(acc)
}

/// `v` or [`AdamError::Range`] naming `what`.
fn ok64(v: Option<i64>, what: &str) -> Result<i64, AdamError> {
    v.ok_or_else(|| AdamError::Range(what.to_string()))
}

/// `v` or [`AdamError::Range`] naming `what`.
fn ok128(v: Option<i128>, what: &str) -> Result<i128, AdamError> {
    v.ok_or_else(|| AdamError::Range(what.to_string()))
}

/// One parameter vector's Adam state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Adam {
    m: Vec<i64>,
    v: Vec<i128>,
    t: u64,
}

impl Adam {
    /// Zero state for a parameter vector of `n` elements. `O(n)` time and space.
    #[must_use]
    pub fn zeros(n: usize) -> Self {
        Self {
            m: vec![0_i64; n],
            v: vec![0_i128; n],
            t: 0,
        }
    }

    /// The number of parameters this state covers. `O(1)`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.m.len()
    }

    /// Whether the state covers no parameter. `O(1)`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.m.is_empty()
    }

    /// One Adam update in place: `m ← β₁m + (1−β₁)g`, `v ← β₂v + (1−β₂)g²`,
    /// `p ← p − lr·m̂/(√v̂ + ε)`. Every operand is a Q32.32 word. On a refusal the elements before
    /// the refused one are already updated and the step counter has advanced.
    ///
    /// # Errors
    /// [`AdamError::Shape`] on a width disagreement; [`AdamError::Range`] on a refused word.
    ///
    /// # Complexity
    /// `O(p.len())` time, serial, `O(1)` extra space.
    pub fn apply(&mut self, lr: i64, p: &mut [i64], g: &[i64]) -> Result<(), AdamError> {
        if p.len() != g.len() || p.len() != self.m.len() || p.len() != self.v.len() {
            return Err(AdamError::Shape(format!(
                "width {} vs {} vs {}",
                p.len(),
                g.len(),
                self.m.len()
            )));
        }
        self.t = self.t.saturating_add(1);
        let b1 = ok64(pow_q32(BETA1_Q32, self.t), "beta_1^t")?;
        let b2 = ok64(pow_q32(BETA2_Q32, self.t), "beta_2^t")?;
        let bc1 = ok64(ONE.checked_sub(b1), "1 - beta_1^t")?;
        let bc2 = ok64(ONE.checked_sub(b2), "1 - beta_2^t")?;
        if bc2 <= 0 {
            return Err(AdamError::Range("1 - beta_2^t".to_string()));
        }
        // `√(1 − β₂^t)` in Q32.32, so the denominator is `√v/√bc2 + ε` rather than
        // `√(v/bc2) + ε`: the division waits until after the square root, where the exponent no
        // longer underflows.
        let root_bc2 = ok64(narrow(isqrt(i128::from(bc2) << 32)), "sqrt(1 - beta_2^t)")?;
        for i in 0..p.len() {
            let gi = g[i];
            let mi = ok64(mul(BETA1_Q32, self.m[i]), "q32 word")?;
            let m2 = ok64(mul(ONE_MINUS_BETA1_Q32, gi), "q32 word")?;
            let mi = ok64(mi.checked_add(m2), "q32 word")?;
            self.m[i] = mi;
            // Second moment in Q64.64: `β₂·v + (1−β₂)·g²`, `g²` the exact `i128` square.
            let g2 = i128::from(gi) * i128::from(gi);
            let t1 = ok128(
                i128::from(BETA2_Q32)
                    .checked_mul(self.v[i])
                    .and_then(|x| rshift_rne(x, 32)),
                "q64 word",
            )?;
            let t2 = ok128(
                i128::from(ONE_MINUS_BETA2_Q32)
                    .checked_mul(g2)
                    .and_then(|x| rshift_rne(x, 32)),
                "q64 word",
            )?;
            let vi = ok128(t1.checked_add(t2), "q64 word")?;
            if vi < 0 {
                return Err(AdamError::Range("second moment".to_string()));
            }
            self.v[i] = vi;
            let mhat = ok64(div(mi, bc1), "q32 word")?;
            // `√v̂ = √v / √bc2`, both square roots in Q32.32 from the Q64.64 word.
            let root_v = ok64(narrow(isqrt(vi)), "q32 word")?;
            let root = ok64(div(root_v, root_bc2), "q32 word")?;
            let denom = ok64(root.checked_add(ADAM_EPS_Q32), "q32 word")?;
            let num = ok64(mul(lr, mhat), "q32 word")?;
            let step = if denom == 0 {
                0
            } else {
                ok64(div(num, denom), "q32 word")?
            };
            p[i] = ok64(p[i].checked_sub(step), "q32 word")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_step_moves_by_lr_times_sign() {
        // Step 1: m̂ = g, √v̂ = |g|, so the move is lr·g/(|g| + ε) ≈ lr·sign(g).
        let mut a = Adam::zeros(2);
        let lr = ONE / 1000;
        let mut p = vec![ONE, -ONE];
        a.apply(lr, &mut p, &[ONE / 2, -(ONE / 4)]).expect("update");
        assert!((p[0] - (ONE - lr)).abs() <= 2, "{}", p[0]);
        assert!((p[1] - (-ONE + lr)).abs() <= 2, "{}", p[1]);
    }

    #[test]
    fn width_disagreement_is_refused() {
        let mut a = Adam::zeros(2);
        let mut p = vec![0_i64; 3];
        assert!(matches!(
            a.apply(ONE, &mut p, &[0; 3]),
            Err(AdamError::Shape(_))
        ));
    }
}
