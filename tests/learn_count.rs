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

/// `key_pack_is_injective`: [`pack_key`] has disjoint bit fields, so distinct
/// `(order, packed-context, byte)` triples map to distinct `u64` words, and [`unpack_key`]
/// round-trips. Exhaustive over order `1..=4`, a sample of contexts (each order's full byte width)
/// and all 256 bytes.
#[test]
fn key_pack_is_injective() {
    use apeiron::learn::countlm::{pack_key, unpack_key};
    use std::collections::{HashMap, HashSet};

    // Contexts spanning every order width: 1, 2, 3 and 4 bytes, oldest-highest.
    const CONTEXTS: &[u64] = &[
        0x00,
        0x01,
        0xff,
        0x01_02,
        0xfe_ed,
        0x00_00,
        0x01_02_03,
        0xff_ff_ff,
        0x01_02_03_04,
        0xde_ad_be_ef,
    ];

    let mut seen: HashSet<(u8, u64, u16)> = HashSet::new();
    let mut words: HashMap<u64, (u8, u64, u16)> = HashMap::new();
    for order in 1u8..=4 {
        let mask = if order == 4 {
            u64::MAX
        } else {
            (1u64 << (8 * u32::from(order))) - 1
        };
        for &c in CONTEXTS {
            let ctx = c & mask;
            for b in 0u16..256 {
                let k = pack_key(order, ctx, b);
                assert!(k < u64::MAX - 1, "packed key must not be a reserved marker");
                assert_eq!(
                    unpack_key(k),
                    (order, ctx, b),
                    "round-trip order={order} ctx={ctx:#x} b={b}"
                );
                if let Some(prev) = words.insert(k, (order, ctx, b)) {
                    assert_eq!(
                        prev,
                        (order, ctx, b),
                        "coordinate collision at order={order} ctx={ctx:#x} b={b}: {prev:?}"
                    );
                }
                seen.insert((order, ctx, b));
            }
        }
        // Every byte as its own 1-byte context, at every order.
        for b in 0u16..256 {
            let ctx = u64::from(u8::try_from(b).expect("byte label fits u8"));
            let k = pack_key(order, ctx, b);
            assert_eq!(unpack_key(k), (order, ctx, b));
            if let Some(prev) = words.insert(k, (order, ctx, b)) {
                assert_eq!(prev, (order, ctx, b));
            }
            seen.insert((order, ctx, b));
        }
    }
    assert_eq!(
        seen.len(),
        words.len(),
        "distinct triples must map to distinct words"
    );
}

/// Collision count on the c4 order-4 run: the legacy tuple keys addressed by the lossy tuple `Coord`
/// fold vs the injective packed words, and the stored-entry count of a real [`CountLM`].
#[test]
fn collision_count_c4() {
    use apeiron::freq::Coord;
    use apeiron::learn::countlm::{pack_key, unpack_key};
    use std::collections::HashSet;
    const CORPUS: &str = "/home/socol/models/data/c4_en.txt";
    const K: usize = 4;

    for train_len in [1_000_000usize, 8_000_000usize] {
        let mut buf = vec![0u8; train_len];
        File::open(CORPUS)
            .and_then(|mut f| f.read_exact(&mut buf))
            .unwrap_or_else(|e| panic!("{CORPUS}: {e}"));
        let train = &buf[..];

        let mut old_keys: HashSet<(u8, u64, u16)> = HashSet::new();
        let mut old_coords: HashSet<u64> = HashSet::new();
        let mut new_words: HashSet<u64> = HashSet::new();
        for i in K..train.len() {
            let label = u16::from(train[i]);
            for o in 1..=K.min(i) {
                let p = train[i - o..i]
                    .iter()
                    .fold(0u64, |p, &b| (p << 8) | u64::from(b));
                let o8 = u8::try_from(o).expect("order fits u8");
                let old = (o8, p, label);
                old_keys.insert(old);
                old_coords.insert(old.coord());
                new_words.insert(pack_key(o8, p, label));
            }
        }

        let mut lm = CountLM::new(K);
        lm.fold_range(train, K, train.len());
        let stored = lm.count_state().len();

        println!(
            "train_len={train_len}: legacy keys={} coords={} collapsed={}; packed words={} stored={}",
            old_keys.len(),
            old_coords.len(),
            old_keys.len() - old_coords.len(),
            new_words.len(),
            stored,
        );

        assert_eq!(
            new_words.len(),
            old_keys.len(),
            "injective pack: distinct words == distinct n-gram keys"
        );
        assert_eq!(
            stored,
            old_keys.len(),
            "distinct stored entries == distinct n-gram keys"
        );
        // The packed key unfolds back to the exact key that was folded.
        for &w in &new_words {
            let (o, p, b) = unpack_key(w);
            assert!(old_keys.contains(&(o, p, b)));
        }
    }
}

/// Test 4: Parity with the spinor-ttl reference on held-out English text.
///
/// Corpus: `/home/socol/models/data/c4_en.txt`; train bytes `[0, 1_000_000)`, held-out bytes
/// `[1_000_000, 1_100_000)`, order 4. Both sides count and score only positions with a full
/// 4-byte context (`i >= 4` within each slice) and score the unrenormalised KN distribution.
/// Reference: spinor-ttl `a05bb1b` `src/spinor_ttl/countlm.py` (sha256 prefix `92b83507127b3c18`),
/// `tab_fold(rows, 4)` then `score(tab, held_rows, 4)`: 2.288994499294 b/B, 53443 argmax hits
/// over 99996 positions, discounts (0.637186868955899, 1.0462353033030585, 1.4168338668711173).
#[test]
fn test_parity() {
    const CORPUS: &str = "/home/socol/models/data/c4_en.txt";
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
