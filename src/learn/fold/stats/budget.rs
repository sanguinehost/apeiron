//! All-in weight-byte accounting for a per-tensor-class bits-per-weight (bpw) assignment.
//!
//! Answers "does a target fit a byte budget" with exact integer arithmetic: every class carries its
//! weight count and its bpw as a rational `num/den`, so the total is a sum of exact rationals and no
//! floating point enters the accounting. [`total_bits`] returns the bit total; [`bytes_ceil`]
//! converts whole bits to whole bytes by rounding up; [`reference_ud_iq2_m`] is the class table of
//! llama.cpp's `UD-IQ2_M` Qwen3.8-27B GGUF as published in
//! `~/Workspace/fold/results/where-the-27b-bits-go.md`; and [`max_body_bpw`] returns the largest
//! uniform body rate (as an exact rational) whose residues keep the allocation inside a byte budget.
//!
//! # Anchors, not a GGUF read
//!
//! The class sizes are that document's table, not a byte-level read of a `.gguf`; no GGUF of the 27B
//! is on disk. They are an `anchor` for the class sizes, and the arithmetic over them is `tested`
//! (`tests/learn_fold_budget.rs`).
//!
//! # Complexity
//!
//! [`total_bits`] is O(n) time and O(1) space over `n` classes, with one `gcd` reduction per added
//! term; every other function is a fixed number of checked `u64`/`u128` operations and is O(1) in
//! time and space.

/// Number of weights one unit of a [`TensorClass::weights`] count stands for.
///
/// The reference table states weights in `G` (10^9) to three decimals, i.e. at `10^6`-weight
/// resolution; storing the count in units of `WEIGHT_UNIT` keeps every entry an exact integer.
pub const WEIGHT_UNIT: u64 = 1_000_000;

/// One quantized tensor class: a name, a weight count and a bits-per-weight rate `bpw_num / bpw_den`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorClass {
    /// Class label (for example `"IQ3_XXS"`).
    pub name: &'static str,
    /// Weight count in units of [`WEIGHT_UNIT`] weights.
    pub weights: u64,
    /// Numerator of the bits-per-weight rate.
    pub bpw_num: u64,
    /// Denominator of the bits-per-weight rate; `0` is refused by the accounting functions.
    pub bpw_den: u64,
}

/// The `UD-IQ2_M` class table for Qwen3.8-27B, from the `where-the-27b-bits-go.md` table.
///
/// The table's bpw figures are `3.0625 = 49/16`, `2.5625 = 41/16`, `3.4375 = 55/16`, `2.625 = 21/8`,
/// `4.25 = 17/4`, `1.75 = 7/4`. Weights are the table's `G` figures in units of [`WEIGHT_UNIT`]
/// (`15.435 G -> 15435`), rounded to the table's own `10^6`-weight resolution — no further rounding.
#[must_use]
pub fn reference_ud_iq2_m() -> [TensorClass; 7] {
    [
        TensorClass {
            name: "IQ3_XXS",
            weights: 15_435,
            bpw_num: 49,
            bpw_den: 16,
        },
        TensorClass {
            name: "IQ2_S",
            weights: 5_704,
            bpw_num: 41,
            bpw_den: 16,
        },
        TensorClass {
            name: "IQ3_S",
            weights: 3_203,
            bpw_num: 55,
            bpw_den: 16,
        },
        TensorClass {
            name: "Q3_K(output)",
            weights: 1_271,
            bpw_num: 55,
            bpw_den: 16,
        },
        TensorClass {
            name: "Q2_K(token_embd)",
            weights: 1_271,
            bpw_num: 21,
            bpw_den: 8,
        },
        TensorClass {
            name: "IQ4_XS",
            weights: 409,
            bpw_num: 17,
            bpw_den: 4,
        },
        TensorClass {
            name: "IQ1_M",
            weights: 24,
            bpw_num: 7,
            bpw_den: 4,
        },
    ]
}

/// Greatest common divisor of two `u128` values (Euclid); `gcd(0, x) == x`.
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// The exact bit contribution of one class as a rational `(num, den)`; `None` on a zero denominator
/// or a `u128` overflow.
fn class_term(c: &TensorClass) -> Option<(u128, u128)> {
    if c.bpw_den == 0 {
        return None;
    }
    let w = u128::from(c.weights).checked_mul(u128::from(WEIGHT_UNIT))?;
    Some((w.checked_mul(u128::from(c.bpw_num))?, u128::from(c.bpw_den)))
}

/// `weights (units of WEIGHT_UNIT) * bpw.0 / bpw.1` as an exact rational; `None` on a zero
/// denominator or a `u128` overflow.
fn term_of(weights: u64, bpw: (u64, u64)) -> Option<(u128, u128)> {
    if bpw.1 == 0 {
        return None;
    }
    let w = u128::from(weights).checked_mul(u128::from(WEIGHT_UNIT))?;
    Some((w.checked_mul(u128::from(bpw.0))?, u128::from(bpw.1)))
}

/// Add `term` (a rational) to `acc` (kept in lowest terms), reducing immediately; `None` on a `u128`
/// overflow in the common denominator or the numerator.
fn add_rational(acc: (u128, u128), term: (u128, u128)) -> Option<(u128, u128)> {
    let (tn, td) = term;
    let g = gcd(acc.1, td);
    let lcm = (acc.1 / g).checked_mul(td)?;
    let num = acc
        .0
        .checked_mul(lcm / acc.1)?
        .checked_add(tn.checked_mul(lcm / td)?)?;
    let den = lcm;
    let gr = gcd(num, den);
    if gr > 1 {
        Some((num / gr, den / gr))
    } else {
        Some((num, den))
    }
}

/// The exact total bits over `classes`, rounded up to the next whole bit.
///
/// Each class contributes `weights * WEIGHT_UNIT * bpw_num / bpw_den` as an exact rational; the
/// rational sum is reduced term by term and ceilinged once, so no floating point and no per-class
/// truncation enters the figure. Returns `None` on a zero `bpw_den` anywhere in `classes`, or on a
/// `u128` overflow.
///
/// # Complexity
///
/// O(n) time and O(1) space over `n = classes.len()`, with one `gcd` per added term.
#[must_use]
pub fn total_bits(classes: &[TensorClass]) -> Option<u128> {
    // Flag form rather than `?`: the pinned extraction fragment has no `?` inside a loop. A
    // refused term stops the sum, so the later iterations are skipped exactly as the `?` would.
    let mut acc = (0u128, 1u128);
    let mut ok = true;
    for c in classes {
        if ok {
            match class_term(c) {
                Some(term) => match add_rational(acc, term) {
                    Some(next) => acc = next,
                    None => ok = false,
                },
                None => ok = false,
            }
        }
    }
    if !ok {
        return None;
    }
    let (num, den) = acc;
    num.checked_add(den.checked_sub(1)?)?.checked_div(den)
}

/// Round a whole-bit count up to whole bytes: `ceil(bits / 8)`. `None` on `u128` overflow.
#[must_use]
pub fn bytes_ceil(bits: u128) -> Option<u128> {
    bits.checked_add(7).map(|b| b / 8)
}

/// The largest uniform body bits-per-weight rate that fits `budget_bytes`, as an exact rational
/// `(num, den)`, given fixed embedding and output-head residues.
///
/// `total_weights`, `embd_weights` and `head_weights` are counts in units of [`WEIGHT_UNIT`]
/// weights; `embd_bpw` and `head_bpw` are `(num, den)` rates (denominators non-zero); the body is
/// `total - embd - head`, charged at one uniform rate. The returned rate is the exact quotient
/// `(budget_bytes - reserve_bytes) * 8 - embd_bits - head_bits`, over the body weight count,
/// reduced to lowest terms — its total allocation equals the budget exactly, so it is the largest
/// rate that fits.
///
/// Returns `None` when the body is empty, when `reserve_bytes` exceeds `budget_bytes`, when the
/// fixed residues alone exceed the budget, when either rate has a zero denominator, or on a
/// `u128`/`u64` overflow.
#[must_use]
pub fn max_body_bpw(
    total_weights: u64,
    embd_weights: u64,
    head_weights: u64,
    embd_bpw: (u64, u64),
    head_bpw: (u64, u64),
    budget_bytes: u64,
    reserve_bytes: u64,
) -> Option<(u64, u64)> {
    let body = total_weights
        .checked_sub(embd_weights)?
        .checked_sub(head_weights)?;
    if body == 0 {
        return None;
    }
    let net = budget_bytes.checked_sub(reserve_bytes)?;
    let budget_bits = u128::from(net).checked_mul(8)?;
    let fixed = add_rational(
        term_of(embd_weights, embd_bpw)?,
        term_of(head_weights, head_bpw)?,
    )?;
    let fixed_bits = fixed
        .0
        .checked_add(fixed.1.checked_sub(1)?)?
        .checked_div(fixed.1)?;
    let remaining = budget_bits.checked_sub(fixed_bits)?;
    let body_raw = u128::from(body).checked_mul(u128::from(WEIGHT_UNIT))?;
    let g = gcd(remaining, body_raw);
    let (num, den) = if g > 1 {
        (remaining / g, body_raw / g)
    } else {
        (remaining, body_raw)
    };
    Some((u64_word(num)?, u64_word(den)?))
}

/// `v` as a `u64`, `None` when it exceeds `u64::MAX`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the bound test above is the only guard the cast needs, and `u64::try_from(u128)` is \
              an unmodeled external in the pinned extraction"
)]
fn u64_word(v: u128) -> Option<u64> {
    if v > u128::from(u64::MAX) {
        None
    } else {
        Some(v as u64)
    }
}
