//! Integration and parity tests for KN count-LM byte model (`CountLM`).
//!
//! Refines:
//! - `KneserNey.kn_normalizes` (`proof/corpus/Substrate/Algebra/Neural/KneserNey.lean:98`)
//! - `MirasCountLM.hebb_comm` (`proof/corpus/Substrate/Algebra/Neural/MirasCountLM.lean:89`)
//! - `MirasCountLM.forget_hebb` (`proof/corpus/Substrate/Algebra/Neural/MirasCountLM.lean:101`)
#![allow(
    clippy::similar_names,
    reason = "distinct models in test setup (m_ab, m_ba, m_a, m_b)"
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

use apeiron::learn::countlm::CountLM;
use std::fs::File;
use std::io::Read;

/// Test 1: Normalisation (refines KneserNey.kn_normalizes).
///
/// For many contexts after folding a mixed stream, |Σ_w dist(ctx)[w] − 1| < 1e-9, including unseen contexts.
#[test]
fn test_kn_normalisation() {
    let mut lm = CountLM::new(4);
    let training_stream = b"The quick brown fox jumps over the lazy dog. 0123456789! \
        A fast red fox jumped over a lazy hound. Kneser-Ney count language model normalization.";
    lm.fold(training_stream);

    let test_contexts: &[&[u8]] = &[
        b"",
        b" ",
        b"fox",
        b"The ",
        b"lazy",
        b"jump",
        b"quic",
        b"Knes",
        b"0123",
        // Unseen contexts:
        b"\xff\xfe\xfd\xfc",
        b"ZZZZ",
        b"zebra",
        b"unseen context here",
        b"\x00\x00\x00\x00",
        b"9876543210",
        // Shorter than order:
        b"a",
        b"ab",
        b"abc",
    ];

    let reader = lm.reader();
    for &ctx in test_contexts {
        let p = reader.dist(ctx);
        let mass: f64 = p.iter().sum();
        assert!(
            (mass - 1.0).abs() < 1e-9,
            "Context {ctx:?} mass {mass} deviates from 1.0 by >= 1e-9"
        );
        for (w, &pw) in p.iter().enumerate() {
            assert!(
                pw >= 0.0,
                "Negative probability {pw} at byte {w} for context {ctx:?}"
            );
        }
    }
}

/// Test 2: Order-free merge (refines MirasCountLM.hebb_comm).
///
/// fold(A) then fold(B) equals fold(B) then fold(A) equals merge(fold A, fold B),
/// compared on the count state (exact integers).
#[test]
fn test_order_free_merge() {
    let stream_a = b"The quick brown fox jumps over the lazy dog. Invariant preservation.";
    let stream_b = b"Pack my box with five dozen liquor jugs and sphinx of black quartz.";

    // Model 1: fold(A) then fold(B)
    let mut m_ab = CountLM::new(4);
    m_ab.fold(stream_a);
    m_ab.fold(stream_b);

    // Model 2: fold(B) then fold(A)
    let mut m_ba = CountLM::new(4);
    m_ba.fold(stream_b);
    m_ba.fold(stream_a);

    // Model 3: merge(fold A, fold B)
    let mut m_a = CountLM::new(4);
    m_a.fold(stream_a);
    let mut m_b = CountLM::new(4);
    m_b.fold(stream_b);
    let mut m_merged = m_a.clone();
    m_merged.merge(&m_b);

    assert_eq!(
        m_ab.signature(),
        m_ba.signature(),
        "fold(A) then fold(B) signature must equal fold(B) then fold(A)"
    );
    assert_eq!(
        m_ab.signature(),
        m_merged.signature(),
        "fold(A) then fold(B) signature must equal merge(fold A, fold B)"
    );

    let state_ab = m_ab.count_state();
    let state_ba = m_ba.count_state();
    let state_merged = m_merged.count_state();

    assert_eq!(
        state_ab, state_ba,
        "Exact count states must be identical for fold(A) then fold(B) vs fold(B) then fold(A)"
    );
    assert_eq!(
        state_ab, state_merged,
        "Exact count states must be identical for sequential fold vs merge"
    );
}

/// Test 3: Exact unlearning (refines MirasCountLM.forget_hebb).
///
/// unfold(A) after fold(A) on fold(B) state equals fold(B) state exactly.
#[test]
fn test_exact_unlearning() {
    let stream_a = b"Observation stream A: transient pattern to be unlearned cleanly.";
    let stream_b = b"Base knowledge stream B: permanent ledger state to preserve bit-for-bit.";

    let mut m_base = CountLM::new(4);
    m_base.fold(stream_b);
    let expected_state = m_base.count_state();
    let expected_sig = m_base.signature();

    let mut m_delta = CountLM::new(4);
    m_delta.fold(stream_a);

    // Fold A onto B
    let mut m = m_base.clone();
    m.fold(stream_a);
    assert_ne!(
        m.count_state(),
        expected_state,
        "Folding A must perturb the count state"
    );

    // Unfold A
    m.unfold(&m_delta);

    assert_eq!(
        m.count_state(),
        expected_state,
        "unfold(A) after fold(A) must restore the exact count state"
    );
    assert_eq!(
        m.signature(),
        expected_sig,
        "unfold(A) after fold(A) must restore the exact signature"
    );

    // Reader distributions must also be bit-identical
    let r_restored = m.reader();
    let r_base = m_base.reader();
    let contexts: &[&[u8]] = &[b"", b"Base", b"ledger", b"unlearned", b"cleanly"];
    for &ctx in contexts {
        assert_eq!(
            r_restored.dist(ctx),
            r_base.dist(ctx),
            "Reader distribution for context {ctx:?} must match base model"
        );
    }
}

/// Test 4: Parity with the spinor-ttl reference on held-out English text.
///
/// Corpus: `~/models/data/c4_en.txt`; train bytes `[0, 1_000_000)`, held-out bytes
/// `[1_000_000, 1_100_000)`, order 4. Both sides count and score only positions with a full
/// 4-byte context (`i >= 4` within each slice) and score the unrenormalised KN distribution.
/// Reference: spinor-ttl `a05bb1b` `src/spinor_ttl/countlm.py` (sha256 prefix `92b83507127b3c18`),
/// `tab_fold(rows, 4)` then `score(tab, held_rows, 4)`: 2.288994499294 b/B, 53443 argmax hits
/// over 99996 positions, discounts (0.637186868955899, 1.0462353033030585, 1.4168338668711173).
#[test]
fn test_parity() {
    const CORPUS: &str = "~/models/data/c4_en.txt";
    const PY_BPB: f64 = 2.288_994_499_294;
    const PY_HITS: usize = 53_443;
    const K: usize = 4;

    let mut buf = vec![0u8; 1_100_000];
    File::open(CORPUS)
        .and_then(|mut f| f.read_exact(&mut buf))
        .unwrap_or_else(|e| panic!("{CORPUS}: {e}"));
    let (train, held) = buf.split_at(1_000_000);

    let mut lm = CountLM::new(K);
    lm.fold_range(train, K, train.len());
    let reader = lm.reader();

    let (d1, d2, d3) = reader.discounts();
    assert!((d1 - 0.637186868955899).abs() < 1e-12, "D1 {d1}");
    assert!((d2 - 1.0462353033030585).abs() < 1e-12, "D2 {d2}");
    assert!((d3 - 1.4168338668711173).abs() < 1e-12, "D3 {d3}");

    let (mut bits, mut hits) = (0.0f64, 0usize);
    for i in K..held.len() {
        let p = reader.dist(&held[i - K..i]);
        bits -= p[usize::from(held[i])].log2();
        let first_max = (0..256).fold(0, |m, j| if p[j] > p[m] { j } else { m });
        hits += usize::from(first_max == usize::from(held[i]));
    }
    let n = u32::try_from(held.len() - K).expect("held-out positions fit u32");
    let bpb = bits / f64::from(n);
    println!("rust b/B {bpb:.12} hits {hits}; reference b/B {PY_BPB:.12} hits {PY_HITS}");
    assert!((bpb - PY_BPB).abs() < 1e-9, "b/B {bpb:.12} vs {PY_BPB:.12}");
    assert_eq!(hits, PY_HITS);
}
