//! `tests/learn_fold_budget.rs` — the 27B all-in weight-byte accounting: the `UD-IQ2_M` reference
//! table sums to its document's figures, the all-in bits-per-weight rates for 4 GiB and 4 GB are the
//! exact rationals, 1.18 b/w all-in is 4.029 GB / 3.753 GiB, the body rate that fits 4 GiB decides
//! whether 1.18 is a body or an all-in rate, and the overflow / zero-denominator refusals. Grade
//! `tested`. The class sizes are the `where-the-27b-bits-go.md` table — an `anchor` for the sizes,
//! not a byte-level GGUF read (no 27B GGUF is on disk).

#![allow(
    clippy::cast_precision_loss,
    reason = "the exact rationals are asserted as reduced (num, den) pairs; f64 appears only to state \
              the document's rounded decimal beside the exact value. The accounting itself is \
              integer-only."
)]

use apeiron::learn::fold::budget::{
    bytes_ceil, max_body_bpw, reference_ud_iq2_m, total_bits, TensorClass, WEIGHT_UNIT,
};

/// One gibibyte in bytes.
const GIB: u128 = 1_073_741_824;

/// The `UD-IQ2_M` table sums to the document's 27.318 G total (within 0.1%) and 10.31 GB of weights
/// (within 0.5%); each row is the table's own entry.
#[test]
fn reference_table_matches_the_document() {
    let r = reference_ud_iq2_m();

    // The seven rows, exactly as tabulated (weights in units of 1e6).
    assert_eq!(
        r[0],
        TensorClass {
            name: "IQ3_XXS",
            weights: 15_435,
            bpw_num: 49,
            bpw_den: 16
        }
    );
    assert_eq!(
        r[1],
        TensorClass {
            name: "IQ2_S",
            weights: 5_704,
            bpw_num: 41,
            bpw_den: 16
        }
    );
    assert_eq!(
        r[2],
        TensorClass {
            name: "IQ3_S",
            weights: 3_203,
            bpw_num: 55,
            bpw_den: 16
        }
    );
    assert_eq!(
        r[3],
        TensorClass {
            name: "Q3_K(output)",
            weights: 1_271,
            bpw_num: 55,
            bpw_den: 16
        }
    );
    assert_eq!(
        r[4],
        TensorClass {
            name: "Q2_K(token_embd)",
            weights: 1_271,
            bpw_num: 21,
            bpw_den: 8
        }
    );
    assert_eq!(
        r[5],
        TensorClass {
            name: "IQ4_XS",
            weights: 409,
            bpw_num: 17,
            bpw_den: 4
        }
    );
    assert_eq!(
        r[6],
        TensorClass {
            name: "IQ1_M",
            weights: 24,
            bpw_num: 7,
            bpw_den: 4
        }
    );

    // 27.317 G weights, within 0.1% of the document's 27.318 G.
    let total_e6: u64 = r.iter().map(|c| c.weights).sum();
    assert_eq!(total_e6, 27_317);
    assert!(
        total_e6 * 1000 <= 27_318 * 1001 && total_e6 * 1000 >= 27_318 * 999,
        "weight total {total_e6}e6 is not within 0.1% of 27.318e9"
    );

    // Exact bit total and byte total, and the 10.31 GB figure within 0.5%.
    let bits = total_bits(&r).unwrap();
    assert_eq!(bits, 82_382_187_500);
    let bytes = bytes_ceil(bits).unwrap();
    assert_eq!(bytes, 10_297_773_438);
    assert!(
        bytes * 1000 <= 10_310_000_000u128 * 1005 && bytes * 1000 >= 10_310_000_000 * 995,
        "weight bytes {bytes} is not within 0.5% of 10.31 GB"
    );

    // The unit the table is stored in is 1e6 weights.
    assert_eq!(WEIGHT_UNIT, 1_000_000);
}

/// The all-in rate over 27.318 G weights for 4 GiB and 4 GB is the exact rational the document
/// rounds to 1.2578 and 1.1714 b/w.
#[test]
fn all_in_rates_are_exact_rationals() {
    // The all-in rate is `max_body_bpw` with the whole weight mass in the body.
    let rate_at_4gib = max_body_bpw(27_318, 0, 0, (0, 1), (0, 1), 4_294_967_296, 0).unwrap();
    assert_eq!(rate_at_4gib, (268_435_456, 213_421_875));
    assert!(
        ((rate_at_4gib.0 as f64 / rate_at_4gib.1 as f64) - 1.2578).abs() < 5e-4,
        "4 GiB all-in rate {rate_at_4gib:?} is not ~1.2578"
    );

    let decimal_g = max_body_bpw(27_318, 0, 0, (0, 1), (0, 1), 4_000_000_000, 0).unwrap();
    assert_eq!(decimal_g, (16_000, 13_659));
    assert!(
        ((decimal_g.0 as f64 / decimal_g.1 as f64) - 1.1714).abs() < 5e-4,
        "4 GB all-in rate {decimal_g:?} is not ~1.1714"
    );
}

/// A uniform 1.18 b/w over 27.318 G weights is 4.029 GB = 3.753 GiB: it fits 4 GiB but not 4 GB.
#[test]
fn one_point_one_eight_all_in_is_four_point_oh_three_gb() {
    let uniform = TensorClass {
        name: "all-in-1.18",
        weights: 27_318,
        bpw_num: 118,
        bpw_den: 100,
    };
    let bits = total_bits(&[uniform]).unwrap();
    assert_eq!(bits, 32_235_240_000);
    let bytes = bytes_ceil(bits).unwrap();
    assert_eq!(bytes, 4_029_405_000); // 4.029405 GB

    // 3.7527 GiB, i.e. 3.753 GiB to the mill.
    assert_eq!((bytes * 1000 + GIB / 2) / GIB, 3753);

    // Fits 4 GiB, exceeds 4 GB.
    assert!(bytes < 4_294_967_296, "1.18 b/w should fit 4 GiB");
    assert!(bytes > 4_000_000_000, "1.18 b/w should exceed 4 GB");
}

/// With `token_embd` (`Q2_K`) and `output` (`Q3_K`) fixed, the largest uniform body rate that fits
/// 4 GiB is 6663575217/6194000000 b/w (~1.0758) — below 1.18, so 1.18 is the **all-in** rate, not a
/// body rate: a 1.18 body rate overflows 4 GiB.
#[test]
fn body_rate_that_fits_four_gib_decides_1_18() {
    let r = reference_ud_iq2_m();
    let embd = r[4]; // Q2_K(token_embd), 2.625 = 21/8
    let head = r[3]; // Q3_K(output), 3.4375 = 55/16

    let body = max_body_bpw(
        27_318,
        embd.weights,
        head.weights,
        (embd.bpw_num, embd.bpw_den),
        (head.bpw_num, head.bpw_den),
        4_294_967_296,
        0,
    )
    .unwrap();
    assert_eq!(body, (6_663_575_217, 6_194_000_000));
    let body_rate = body.0 as f64 / body.1 as f64;
    assert!(
        (body_rate - 1.0758).abs() < 5e-4,
        "4 GiB body rate {body:?} is not ~1.0758"
    );
    assert!(
        body_rate < 1.18,
        "body rate {body_rate} should be below 1.18"
    );

    // At body = 1.18 the whole allocation exceeds 4 GiB.
    let body_weights = 27_318u64 - embd.weights - head.weights; // 24_776e6
    let body_bits = u128::from(body_weights) * u128::from(WEIGHT_UNIT) * 118 / 100;
    let fixed_bits = total_bits(&[embd, head]).unwrap();
    assert_eq!(fixed_bits, 7_705_437_500);
    let total_bytes = bytes_ceil(body_bits + fixed_bits).unwrap();
    assert_eq!(total_bytes, 4_617_639_688);
    assert!(
        total_bytes > 4_294_967_296,
        "a 1.18 body rate must overflow 4 GiB, got {total_bytes} B"
    );
}

/// `reserve_bytes` shrinks the budget by exactly its size: reserving half leaves the same rate as
/// halving the budget outright.
#[test]
fn reserve_is_subtracted_from_the_budget() {
    let with_reserve = max_body_bpw(100, 0, 0, (0, 1), (0, 1), 200, 100).unwrap();
    let without = max_body_bpw(100, 0, 0, (0, 1), (0, 1), 100, 0).unwrap();
    assert_eq!(with_reserve, without);
}

/// Overflow and zero-denominator paths refuse with `None` rather than wrapping or dividing by zero.
#[test]
fn refuses_overflow_and_zero_denominator() {
    // A zero bpw denominator in the class table.
    assert_eq!(
        total_bits(&[TensorClass {
            name: "zero-den",
            weights: 1,
            bpw_num: 1,
            bpw_den: 0
        }]),
        None
    );

    // A weight * rate product past `u128`.
    assert_eq!(
        total_bits(&[TensorClass {
            name: "overflow",
            weights: u64::MAX,
            bpw_num: u64::MAX,
            bpw_den: 1,
        }]),
        None
    );

    // `bits + 7` past `u128::MAX`.
    assert_eq!(bytes_ceil(u128::MAX), None);

    // A reserve larger than the budget.
    assert_eq!(max_body_bpw(100, 0, 0, (0, 1), (0, 1), 10, 11), None);

    // Embedding + head weights exceeding the total.
    assert_eq!(max_body_bpw(10, 6, 6, (1, 1), (1, 1), 1000, 0), None);

    // An empty body.
    assert_eq!(max_body_bpw(10, 6, 4, (1, 1), (1, 1), 1000, 0), None);

    // A zero rate denominator on a non-empty residue.
    assert_eq!(max_body_bpw(10, 2, 0, (1, 0), (0, 1), 1000, 0), None);

    // Fixed residues alone exceeding the budget.
    assert_eq!(max_body_bpw(10, 8, 0, (4, 1), (0, 1), 1, 0), None);
}
