//! `tests/learn_onebit_bits.rs` — `onebit_bits` reproduces the five bit counters of the Qwen2.5-0.5B
//! arm `...-sink-fnorm-grow-res-qmlp-nb` exactly (the receipts/onebit/t3-nb-3seed note.json`,
//! `Receipts.t3_nb_3seed`), and the controls move the totals by the amounts the closed form
//! predicts. Grade `tested` (Python counter versus Rust closed form; not a refinement).

use apeiron::learn::onebit_bits::{
    bits_c, bits_corr, bits_d, bits_r, bits_sign, index_bits, params, total_bits, Cfg,
};

const RECEIPT: &str = include_str!("../the receipts/onebit/t3-nb-3seed note.json");

const SIGN: u64 = 400_490_496;
const R: u64 = 15_347_712;
const C: u64 = 2_128_896;
const D: u64 = 1_775_616;
const CORR: u64 = 1_798_656;
const TOTAL: u64 = 421_541_376;
const PARAMS: u64 = 357_826_560;

/// Headers per layer: `r` 2 x 12 calls, `c` 12, `d` 12, corrector 2 x (rank 2 + rank 1).
const HEADERS_PER_LAYER: u64 = 24 + 12 + 12 + 6;

#[test]
fn receipt_literals_are_in_the_receipt_file() {
    for (key, v) in [
        ("bits", TOTAL),
        ("params", PARAMS),
        ("bits_sign", SIGN),
        ("bits_r", R),
        ("bits_c", C),
        ("bits_d", D),
        ("bits_corr", CORR),
    ] {
        let needle = format!("\"{key}\": {v}");
        assert!(RECEIPT.contains(&needle), "receipt lacks {needle}");
    }
}

#[test]
fn qwen_components_equal_the_receipt() {
    let c = Cfg::qwen25_05b_nb();
    assert_eq!(bits_sign(&c), Some(SIGN));
    assert_eq!(bits_r(&c), Some(R));
    assert_eq!(bits_c(&c), Some(C));
    assert_eq!(bits_d(&c), Some(D));
    assert_eq!(bits_corr(&c), Some(CORR));
    assert_eq!(params(&c), Some(PARAMS));
}

#[test]
fn total_is_the_sum_of_the_five_components_and_nothing_else() {
    let c = Cfg::qwen25_05b_nb();
    assert_eq!(SIGN + R + C + D + CORR, TOTAL);
    assert_eq!(total_bits(&c), Some(TOTAL));
    let sum = bits_sign(&c).unwrap()
        + bits_r(&c).unwrap()
        + bits_c(&c).unwrap()
        + bits_d(&c).unwrap()
        + bits_corr(&c).unwrap();
    assert_eq!(total_bits(&c), Some(sum));
}

#[test]
fn per_layer_sign_is_the_hand_count() {
    let one = Cfg {
        layers: 1,
        ..Cfg::qwen25_05b_nb()
    };
    assert_eq!(bits_sign(&one), Some(16_687_104));
    assert_eq!(params(&one), Some(14_909_440));
    assert_eq!(bits_sign(&one).map(|s| s * 24), Some(SIGN));
}

#[test]
fn k_rows_417_adds_two_rows_of_sign_per_layer() {
    let base = Cfg::qwen25_05b_nb();
    let more = Cfg {
        k_rows: 417,
        ..base
    };
    assert_eq!(
        bits_sign(&more).unwrap() - bits_sign(&base).unwrap(),
        24 * 2 * 896
    );
    let one = Cfg { layers: 1, ..base };
    let one_more = Cfg { layers: 1, ..more };
    assert_eq!(
        bits_sign(&one_more).unwrap() - bits_sign(&one).unwrap(),
        2 * 896
    );
    // The extra row also carries r scale bits (4 * 7 levels + 8 per row, twice per layer).
    assert_eq!(
        bits_r(&more).unwrap() - bits_r(&base).unwrap(),
        24 * 2 * (4 * 7 + 8)
    );
    assert_eq!(bits_c(&more), bits_c(&base));
    assert_eq!(bits_d(&more), bits_d(&base));
    assert_eq!(bits_corr(&more), bits_corr(&base));
    assert_ne!(total_bits(&more), Some(TOTAL));
    // Params count plane 1 only, so k_rows does not move it.
    assert_eq!(params(&more), Some(PARAMS));
}

#[test]
fn dropping_the_header_removes_exactly_header_times_count() {
    let base = Cfg::qwen25_05b_nb();
    let none = Cfg {
        header_bits: 0,
        ..base
    };
    let removed = 24 * HEADERS_PER_LAYER * 32;
    assert_eq!(removed, 41_472);
    assert_eq!(total_bits(&none), Some(TOTAL - removed));
    assert_eq!(bits_sign(&none), Some(SIGN));
    assert_eq!(
        bits_r(&base).unwrap() - bits_r(&none).unwrap(),
        24 * 24 * 32
    );
    assert_eq!(
        bits_c(&base).unwrap() - bits_c(&none).unwrap(),
        24 * 12 * 32
    );
    assert_eq!(
        bits_d(&base).unwrap() - bits_d(&none).unwrap(),
        24 * 12 * 32
    );
    assert_eq!(
        bits_corr(&base).unwrap() - bits_corr(&none).unwrap(),
        24 * 6 * 32
    );
}

#[test]
fn overflow_and_bad_group_return_none() {
    let base = Cfg::qwen25_05b_nb();
    let big = Cfg {
        d_model: u64::MAX / 2,
        ..base
    };
    assert_eq!(bits_sign(&big), None);
    assert_eq!(params(&big), None);
    assert_eq!(bits_c(&big), None);
    assert_eq!(bits_d(&big), None);
    assert_eq!(bits_corr(&big), None);
    assert_eq!(total_bits(&big), None);
    let many = Cfg {
        layers: u64::MAX,
        ..base
    };
    assert_eq!(total_bits(&many), None);
    assert_eq!(index_bits(&many), None);
    // `n` must be a multiple of the scale group.
    for group in [0, 100] {
        let bad = Cfg { group, ..base };
        assert_eq!(bits_r(&bad), None, "group {group}");
        assert_eq!(total_bits(&bad), None, "group {group}");
    }
}

#[test]
fn index_term_is_reported_apart_from_the_receipt_total() {
    let c = Cfg::qwen25_05b_nb();
    assert_eq!(index_bits(&c), Some(48 * 2044));
    assert_eq!(total_bits(&c), Some(TOTAL));
    // log2 C(4864, 416) = sum_{i<416} log2((4864 - i) / (416 - i)); the per-set bound is its ceiling.
    let log2_c: f64 = (0..416_u32)
        .map(|i| (f64::from(4864 - i) / f64::from(416 - i)).log2())
        .sum();
    assert!(
        log2_c > 2043.0 && log2_c <= 2044.0,
        "log2 C(4864,416) = {log2_c}"
    );
    assert!(index_bits(&c).unwrap() >= 98_106);
}
