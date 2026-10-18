//! The integer sampler: greedy argmax and the nucleus (top-p) filter (port-map row 8).

use crate::engine::num::value::div_round_nearest;
use crate::engine::{FIXED_FRAC_BITS, ONE};
use crate::numeric::fixed_exp::fixed_exp;

/// A refused sampler step (port-map row 8).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplerError {
    /// The logit vector is empty, so no index is defined.
    EmptyLogits,
    /// The logit count leaves the `u32` index range.
    LogitsTooLong,
    /// The nucleus threshold is not in `(0, ONE]`.
    BadThreshold {
        /// Threshold supplied, Q32.32.
        p: i64,
    },
    /// A maximum-shifted logit left the [`fixed_exp`] window `[-16·ONE, 21·ONE]`.
    ExpOutOfRange,
    /// The softmax total is zero, so the probability normalisation is undefined.
    ZeroSum,
    /// An `i128` accumulation or an `i64` narrowing left its range.
    Overflow,
}

/// Argmax over Q32.32 logits: the v0 greedy sampler (port-map row 8).
///
/// The result is the index of the maximal logit; an exact tie keeps the lowest index because the
/// scan updates only on a strict `>`. There is no division and no float. The index is returned as
/// `u32`; a logit vector longer than `u32::MAX` refuses rather than truncating.
///
/// # Errors
///
/// Returns [`SamplerError::EmptyLogits`] for an empty vector and [`SamplerError::LogitsTooLong`]
/// when the winning index does not fit `u32`.
///
/// # Complexity
///
/// `O(n)` time in `i64` comparisons, `O(1)` extra space; `n` is the vocabulary size.
pub fn argmax_fixed(logits: &[i64]) -> Result<u32, SamplerError> {
    if logits.is_empty() {
        return Err(SamplerError::EmptyLogits);
    }
    let mut best = 0usize;
    let mut i = 1usize;
    while i < logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
        i += 1;
    }
    u32::try_from(best).map_err(|_| SamplerError::LogitsTooLong)
}

/// The integer nucleus (top-p) filter over Q32.32 logits (port-map row 8).
///
/// The filter follows the port map's exp/divide contract with no float: subtract the exact maximum
/// logit, evaluate [`fixed_exp`] on each shifted logit (the window is `[-16·ONE, 21·ONE]`; the
/// shifted values are `<= 0`, so only the lower bound can refuse), accumulate the checked `i128`
/// total, and form each Q32.32 probability with exactly one round-to-nearest division
/// `round(e_i·2^32 / Σ e)` (ties upward). The entries are ordered by descending logit, ties to the
/// lowest index, and the shortest prefix whose cumulative probability reaches the Q32.32 threshold
/// `p` is returned in that order. `p = ONE` admits the whole vocabulary.
///
/// The stochastic draw from the returned nucleus and temperature scaling need a deterministic RNG
/// contract and are not on this path.
///
/// # Errors
///
/// Returns [`SamplerError::EmptyLogits`] for an empty vector, [`SamplerError::LogitsTooLong`] when
/// an index does not fit `u32`, [`SamplerError::BadThreshold`] when `p` is not in `(0, ONE]`,
/// [`SamplerError::ExpOutOfRange`] when a maximum-shifted logit is below the `fixed_exp` lower
/// window, [`SamplerError::ZeroSum`] when the total is zero, and [`SamplerError::Overflow`] when
/// an `i128` accumulation leaves its range. No value is clamped.
///
/// # Complexity
///
/// `O(n)` time in `fixed_exp` calls and checked integer arithmetic for the probabilities plus
/// `O(n log n)` for the ordering, with `O(n)` scratch space for the exponentials and the order.
pub fn top_p_fixed(logits: &[i64], p: i64) -> Result<Vec<u32>, SamplerError> {
    if logits.is_empty() {
        return Err(SamplerError::EmptyLogits);
    }
    if p <= 0 || p > ONE {
        return Err(SamplerError::BadThreshold { p });
    }
    let len = logits.len();
    u32::try_from(len - 1).map_err(|_| SamplerError::LogitsTooLong)?;

    // Exact maximum for the shift.
    let mut mx = logits[0];
    let mut i = 1usize;
    while i < len {
        if logits[i] > mx {
            mx = logits[i];
        }
        i += 1;
    }

    // Exponentials and the checked i128 total.
    let mut exps: Vec<i64> = Vec::with_capacity(len);
    let mut total: i128 = 0;
    i = 0;
    while i < len {
        let shifted = logits[i].checked_sub(mx).ok_or(SamplerError::Overflow)?;
        let e = fixed_exp(shifted).map_err(|_| SamplerError::ExpOutOfRange)?;
        total = total
            .checked_add(i128::from(e))
            .ok_or(SamplerError::Overflow)?;
        exps.push(e);
        i += 1;
    }
    if total <= 0 {
        return Err(SamplerError::ZeroSum);
    }

    // One round-to-nearest division per probability.
    let mut weights: Vec<i128> = Vec::with_capacity(len);
    i = 0;
    while i < len {
        let scaled = i128::from(exps[i]) << FIXED_FRAC_BITS;
        weights.push(div_round_nearest(scaled, total));
        i += 1;
    }

    // Descending logit; an exact tie keeps the lowest index.
    let mut order: Vec<usize> = Vec::with_capacity(len);
    i = 0;
    while i < len {
        order.push(i);
        i += 1;
    }
    order.sort_by(|&a, &b| logits[b].cmp(&logits[a]).then(a.cmp(&b)));

    // Shortest prefix whose cumulative probability reaches the threshold.
    let mut nucleus: Vec<u32> = Vec::new();
    let mut cum: i128 = 0;
    let mut k = 0usize;
    while k < len {
        let idx = order[k];
        nucleus.push(u32::try_from(idx).map_err(|_| SamplerError::LogitsTooLong)?);
        cum = cum
            .checked_add(weights[idx])
            .ok_or(SamplerError::Overflow)?;
        if cum >= i128::from(p) {
            break;
        }
        k += 1;
    }
    Ok(nucleus)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Control for the index fill and the descending sort: the top logit with the lowest index
    // (index 1 over the tied index 2) comes first, so the first prefix step selects it alone.
    // The full-mass nucleus is a prefix of the old order `[1, 2, 0, 3]`.
    #[test]
    fn top_p_fixed_orders_descending_with_lowest_index_on_ties() {
        assert_eq!(top_p_fixed(&[5, 9, 9, 1], 1), Ok(vec![1]));
        let full = top_p_fixed(&[5, 9, 9, 1], ONE).expect("top_p");
        assert!(!full.is_empty());
        assert!([1u32, 2, 0, 3].starts_with(&full));
    }
}
