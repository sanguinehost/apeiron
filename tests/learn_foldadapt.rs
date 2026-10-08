//! Integration tests for forward-only finetuning adapter (`FoldAdapter`).
//!
//! Refines:
//! - `Sanguine.MLPrimitivesBridge.osFuse_is_add` (`proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:174`)
//! - `Sanguine.MLPrimitivesBridge.osFuse_grouping_free` (`proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:182`)
//! - `Sanguine.MLPrimitivesBridge.kn_floor_refinement` (`proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:385`)
#![allow(
    clippy::similar_names,
    reason = "distinct models in test setup (fa_ab, fa_ba, fa_a, fa_b)"
)]
#![allow(
    clippy::unreadable_literal,
    reason = "numeric floating point literal precision in tests"
)]
#![allow(
    clippy::doc_markdown,
    reason = "mathematical formulas and Lean theorem names in doc comments"
)]
#![allow(
    clippy::float_cmp,
    reason = "exact bit-for-bit float equality assertion in unlearning test"
)]

use apeiron::learn::foldadapt::{FoldAdaptError, FoldAdapter, DOORWAY};

#[test]
fn order_free_merge() {
    let stream_a = &[10u32, 20, 30, 40, 20, 30, 50, 10, 20, 30];
    let stream_b = &[20u32, 30, 40, 60, 70, 80, 20, 30, 50];

    // fold(A); fold(B)
    let mut fa_ab = FoldAdapter::new(3).expect("order 3 valid");
    fa_ab.fold(stream_a);
    fa_ab.fold(stream_b);

    // fold(B); fold(A)
    let mut fa_ba = FoldAdapter::new(3).expect("order 3 valid");
    fa_ba.fold(stream_b);
    fa_ba.fold(stream_a);

    // merge(fold A, fold B)
    let mut fa_a = FoldAdapter::new(3).expect("order 3 valid");
    fa_a.fold(stream_a);
    let mut fa_b = FoldAdapter::new(3).expect("order 3 valid");
    fa_b.fold(stream_b);

    let mut fa_merged = fa_a.clone();
    fa_merged.merge(&fa_b).expect("orders match");

    let fa_merged_fn = fa_a.merged(&fa_b).expect("orders match");

    // Exact count state (sorted entries), not floats
    assert_eq!(
        fa_ab.count_state(),
        fa_ba.count_state(),
        "fold(A);fold(B) must equal fold(B);fold(A)"
    );
    assert_eq!(
        fa_ab.count_state(),
        fa_merged.count_state(),
        "fold(A);fold(B) must equal merge(fold A, fold B)"
    );
    assert_eq!(
        fa_merged.count_state(),
        fa_merged_fn.count_state(),
        "merge(&mut) must match merged(&)"
    );
}

#[test]
fn grouping_free() {
    let s1 = &[1u32, 2, 3, 4, 5, 2, 3];
    let s2 = &[3u32, 4, 5, 6, 7, 4, 5];
    let s3 = &[5u32, 6, 7, 8, 9, 6, 7];
    let s4 = &[7u32, 8, 9, 10, 11, 8, 9];

    let mut fa1 = FoldAdapter::new(3).expect("order 3 valid");
    fa1.fold(s1);
    let mut fa2 = FoldAdapter::new(3).expect("order 3 valid");
    fa2.fold(s2);
    let mut fa3 = FoldAdapter::new(3).expect("order 3 valid");
    fa3.fold(s3);
    let mut fa4 = FoldAdapter::new(3).expect("order 3 valid");
    fa4.fold(s4);

    // merge(merge(a,b),c)
    let mut left_grouped = fa1.clone();
    left_grouped.merge(&fa2).expect("merge 1+2");
    left_grouped.merge(&fa3).expect("merge (1+2)+3");
    left_grouped.merge(&fa4).expect("merge ((1+2)+3)+4");

    // merge(a,merge(b,c))
    let mut right_inner = fa3.clone();
    right_inner.merge(&fa4).expect("merge 3+4");
    let mut right_mid = fa2.clone();
    right_mid.merge(&right_inner).expect("merge 2+(3+4)");
    let mut right_grouped = fa1.clone();
    right_grouped.merge(&right_mid).expect("merge 1+(2+(3+4))");

    // fold of the concatenation's per-stream counts
    let mut seq_folded = FoldAdapter::new(3).expect("order 3 valid");
    seq_folded.fold(s1);
    seq_folded.fold(s2);
    seq_folded.fold(s3);
    seq_folded.fold(s4);

    assert_eq!(
        left_grouped.count_state(),
        right_grouped.count_state(),
        "merge must be strictly grouping-free (associative)"
    );
    assert_eq!(
        left_grouped.count_state(),
        seq_folded.count_state(),
        "merge grouping must match sequential per-stream folding"
    );
}

#[test]
fn exact_unlearning() {
    let stream_a = &[1u32, 2, 3, 2, 3, 4, 5, 1, 2, 3, 6, 7];
    let stream_b = &[7u32, 8, 9, 8, 9, 10, 11, 12, 13, 2, 3, 99];

    let mut fa_a = FoldAdapter::new(3).expect("order 3 valid");
    fa_a.fold(stream_a);
    let state_a = fa_a.count_state();
    let sig_a = fa_a.signature();
    let keys_a = fa_a.keys().to_vec();

    let mut fa_ab = FoldAdapter::new(3).expect("order 3 valid");
    fa_ab.fold(stream_a);
    fa_ab.fold(stream_b);
    fa_ab.unfold(stream_b);

    // Exact count state match
    assert_eq!(
        fa_ab.count_state(),
        state_a,
        "fold(A);fold(B);unfold(B) must exactly match fold(A)"
    );
    assert_eq!(fa_ab.signature(), sig_a, "signature must match exactly");

    // An unfolded-to-zero entry leaves no key behind
    assert_eq!(
        fa_ab.keys().len(),
        keys_a.len(),
        "unfolded-to-zero entries must not leave keys behind"
    );
    for k in fa_ab.keys() {
        assert!(keys_a.contains(k), "key {k:?} must be present in fold(A)");
    }

    // refine() outputs bit-identical on 5+ contexts
    let contexts: &[&[u32]] = &[
        &[1, 2],
        &[2, 3],
        &[6],
        &[7, 8],
        &[99, 100],
        &[13],
        &[2, 3, 4],
    ];
    let base: Vec<f64> = vec![0.02, 0.08, 0.15, 0.20, 0.10, 0.05, 0.05, 0.10, 0.10, 0.15];

    for ctx in contexts {
        let r_a = fa_a.refine(&base, ctx).expect("refine fa_a succeeds");
        let r_ab = fa_ab.refine(&base, ctx).expect("refine fa_ab succeeds");
        assert_eq!(
            r_a, r_ab,
            "refine on context {ctx:?} must be bit-identical between fa_a and unlearned fa_ab"
        );
    }
}

#[test]
fn mass_conservation() {
    // Normalised arbitrary base (non-uniform)
    let base: Vec<f64> = vec![0.05, 0.20, 0.15, 0.10, 0.25, 0.05, 0.12, 0.08];
    let base_sum: f64 = base.iter().sum();
    assert!((base_sum - 1.0).abs() < 1e-12);

    let mut fa = FoldAdapter::new(3).expect("order 3 valid");
    // Decided context: [1, 2] followed by 3
    fa.fold(&[0, 1, 2, 3, 1, 2, 3, 1, 2, 3]);
    // Diffuse context: [4] followed by many different tokens
    fa.fold(&[4, 0, 4, 1, 4, 2, 4, 5, 4, 6]);

    let test_contexts: &[&[u32]] = &[
        &[1, 2],       // seen decided context (PR <= 2.0)
        &[4],          // seen diffuse context (PR > 2.0)
        &[99, 100],    // unseen context
        &[7],          // single-token seen context with no evidence
        &[],           // empty context
        &[0, 1, 2],    // order 3 context
        &[1, 2, 3, 4], // context longer than k=3
    ];

    for ctx in test_contexts {
        let r = fa.refine(&base, ctx).expect("refine succeeds");
        let sum: f64 = r.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "mass conservation failed on ctx {ctx:?}: sum = {sum}"
        );
        for (idx, &p) in r.iter().enumerate() {
            assert!(
                p >= 0.0,
                "negative probability at index {idx} on ctx {ctx:?}: {p}"
            );
        }
    }

    // Also test unconstrained refine_with (inf PR budget)
    let r_ungated = fa
        .refine_with(&base, &[4], fa.discounts())
        .expect("refine_with succeeds");
    let ungated_sum: f64 = r_ungated.iter().sum();
    assert!(
        (ungated_sum - 1.0).abs() < 1e-12,
        "ungated mass conservation failed: sum = {ungated_sum}"
    );
    for (idx, &p) in r_ungated.iter().enumerate() {
        assert!(p >= 0.0, "ungated negative probability at index {idx}: {p}");
    }
}

#[test]
fn evidence_lands() {
    let mut fa = FoldAdapter::new(3).expect("order 3 valid");
    // Context [1, 2] followed exclusively by next-token 3
    fa.fold(&[1, 2, 3, 9, 1, 2, 3, 9, 1, 2, 3]);

    let base = vec![0.1f64; 10]; // uniform base distribution
    let r = fa.refine(&base, &[1, 2]).expect("refine succeeds");

    assert!(
        r[3] > base[3],
        "evidence lands: r[3] ({}) must exceed base[3] ({})",
        r[3],
        base[3]
    );
}

#[test]
fn diffuse_preservation() {
    let mut fa = FoldAdapter::new(3).expect("order 3 valid");
    // Context [5] is followed by {0, 1, 2, 3, 4, 6} (PR = 6 > DOORWAY = 2)
    fa.fold(&[5, 0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 6]);

    let base = vec![0.1f64; 10];
    let r = fa.refine(&base, &[9, 5]).expect("refine succeeds");
    assert_eq!(
        r, base,
        "diffuse domain context (PR > 2.0) must preserve the base untouched"
    );
}

#[test]
fn refusals() {
    // Bad order returns Err, no panic
    assert_eq!(
        FoldAdapter::new(0).unwrap_err(),
        FoldAdaptError::InvalidOrder(0)
    );
    assert_eq!(
        FoldAdapter::new(5).unwrap_err(),
        FoldAdaptError::InvalidOrder(5)
    );
    assert_eq!(
        FoldAdapter::with_capacity(0, 16).unwrap_err(),
        FoldAdaptError::InvalidOrder(0)
    );
    assert_eq!(
        FoldAdapter::with_capacity(5, 16).unwrap_err(),
        FoldAdaptError::InvalidOrder(5)
    );

    // Empty base returns Err, no panic
    let fa = FoldAdapter::new(3).expect("order 3 valid");
    assert_eq!(
        fa.refine(&[], &[1, 2]).unwrap_err(),
        FoldAdaptError::EmptyBase
    );
    assert_eq!(
        fa.refine_with(&[], &[1, 2], (0.5, 0.5, 0.5)).unwrap_err(),
        FoldAdaptError::EmptyBase
    );
    assert_eq!(
        fa.refine_sparse(&[], &[1, 2], (0.5, 0.5, 0.5), DOORWAY)
            .unwrap_err(),
        FoldAdaptError::EmptyBase
    );

    // Mismatched order on merge returns Err, no panic
    let mut fa3 = FoldAdapter::new(3).expect("order 3 valid");
    let fa2 = FoldAdapter::new(2).expect("order 2 valid");
    assert_eq!(
        fa3.merge(&fa2).unwrap_err(),
        FoldAdaptError::OrderMismatch {
            expected: 3,
            actual: 2
        }
    );
    assert_eq!(
        fa3.merged(&fa2).unwrap_err(),
        FoldAdaptError::OrderMismatch {
            expected: 3,
            actual: 2
        }
    );
}
