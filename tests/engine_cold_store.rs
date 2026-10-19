//! `ColdStore`, the cold KV block store, grade `tested`.
//!
//! Checked here: sealed rows leave the dense plane and gather back bit-for-bit for the exact named
//! blocks; a missing block, a wrong width and a packed-key overflow refuse; the payload bytes are
//! `O(tokens)` at the 128k- and 1M-equivalent block counts while the index bytes are `O(blocks)`.
//! Not checked here: quality, any kv/attention read, disk spill, or GPU behaviour.

#![allow(
    clippy::arithmetic_side_effects,
    reason = "test fixture arithmetic on block ids and counts bounded far below i64::MAX; the store does the checked sums"
)]

use apeiron::engine::{ColdStore, ColdStoreError};

const BLOCK: usize = 64;

/// A block's rows: key row `r` is `base + r`, value row `r` is `-base - r`, each `head_dim` wide.
fn rows(base: i64, head_dim: usize, block: usize) -> (Vec<i64>, Vec<i64>) {
    let mut k = Vec::with_capacity(block * head_dim);
    let mut v = Vec::with_capacity(block * head_dim);
    let mut r = 0usize;
    while r < block {
        let word = base * 1000 + i64::try_from(r).expect("small");
        let mut i = 0usize;
        while i < head_dim {
            k.push(word);
            v.push(-word);
            i += 1;
        }
        r += 1;
    }
    (k, v)
}

/// Rows stored under one key come back exactly, and the three key fields stay distinct.
#[test]
fn sealed_rows_gather_back_outside_the_dense_plane() {
    let head_dim = 4usize;
    let mut store = ColdStore::new(BLOCK, head_dim).expect("geometry");
    let (k0, v0) = rows(7, head_dim, BLOCK);
    let (k1, v1) = rows(9, head_dim, BLOCK);
    store.put(3, 1, 0, &k0, &v0).expect("put");
    store.put(3, 1, 1, &k1, &v1).expect("put");
    let (gk, gv) = store.gather(3, 1, 0).expect("gather");
    assert_eq!(gk, &k0[..]);
    assert_eq!(gv, &v0[..]);
    let (gk, gv) = store.gather(3, 1, 1).expect("gather");
    assert_eq!(gk, &k1[..]);
    assert_eq!(gv, &v1[..]);
    // A different layer and a different kv-head are different records.
    assert_eq!(store.gather(4, 1, 0).err(), Some(ColdStoreError::Missing));
    assert_eq!(store.gather(3, 0, 0).err(), Some(ColdStoreError::Missing));
    // The named block list gathers exactly those rows, in order.
    let both = store.gather_many(3, 1, &[1, 0]).expect("many");
    assert_eq!(both.len(), 2);
    assert_eq!(both[0].0, &k1[..]);
    assert_eq!(both[1].0, &k0[..]);
    // A re-put replaces the record and does not double-count the stored positions.
    let (k2, v2) = rows(11, head_dim, BLOCK);
    store.put(3, 1, 0, &k2, &v2).expect("replace");
    assert_eq!(store.gather(3, 1, 0).expect("gather").0, &k2[..]);
    assert_eq!(store.blocks(), 2);
    assert_eq!(store.positions(), 2 * BLOCK);
    assert_eq!(store.payload_bytes(), 2 * BLOCK * 2 * head_dim * 8);
}

/// A missing block, a wrong width, an empty row and an out-of-range key all refuse.
#[test]
fn refusals_are_reported_not_substituted() {
    let head_dim = 4usize;
    let mut store = ColdStore::new(BLOCK, head_dim).expect("geometry");
    let (k, v) = rows(1, head_dim, BLOCK);
    store.put(0, 0, 0, &k, &v).expect("put");
    assert_eq!(store.gather(0, 0, 1).err(), Some(ColdStoreError::Missing));
    // One missing id refuses the whole list: no silent skip.
    assert_eq!(
        store.gather_many(0, 0, &[0, 1]).err(),
        Some(ColdStoreError::Missing)
    );
    // A row slice that is not a whole number of head_dim rows.
    assert_eq!(
        store
            .put(0, 1, 0, &k[..k.len() - 1], &v[..v.len() - 1])
            .err(),
        Some(ColdStoreError::BadGeometry)
    );
    // Key and value rows of different lengths.
    assert_eq!(
        store.put(0, 1, 0, &k, &v[..v.len() - head_dim]).err(),
        Some(ColdStoreError::BadGeometry)
    );
    // An empty block.
    assert_eq!(
        store.put(0, 1, 0, &[], &[]).err(),
        Some(ColdStoreError::BadGeometry)
    );
    // A block wider than the geometry, and a key field past 2^21.
    let (wide, wide_v) = rows(2, head_dim, BLOCK + 1);
    assert_eq!(
        store.put(0, 1, 0, &wide, &wide_v).err(),
        Some(ColdStoreError::BadGeometry)
    );
    assert_eq!(
        store.put(1 << 21, 0, 0, &k, &v).err(),
        Some(ColdStoreError::BadGeometry)
    );
    assert_eq!(
        ColdStore::new(0, head_dim).err(),
        Some(ColdStoreError::BadGeometry)
    );
}

/// 128k- and 1M-equivalent block counts: payload `O(tokens)` at an exact byte count, index
/// `O(blocks)`, both reported apart. Test width `head_dim = 2`, one plane; the production figure at
/// `head_dim = 64` is 32× these per-plane numbers, 49 152 B/token over 24 layers × 2 kv-heads.
#[test]
fn stored_bytes_are_o_tokens_at_128k_and_1m() {
    let head_dim = 2usize;
    let per_position = 2 * head_dim * 8;
    assert_eq!(per_position, 32);
    let cases: [(usize, usize); 2] = [(131_072, 4_194_304), (1_048_576, 33_554_432)];
    let mut c = 0usize;
    while c < cases.len() {
        let (tokens, want_payload) = cases[c];
        let blocks = tokens / BLOCK;
        let mut store = ColdStore::new(BLOCK, head_dim).expect("geometry");
        let (k, v) = rows(3, head_dim, BLOCK);
        let mut b = 0usize;
        while b < blocks {
            store.put(0, 0, b, &k, &v).expect("put");
            b += 1;
        }
        assert_eq!(store.blocks(), blocks);
        assert_eq!(store.positions(), tokens);
        assert_eq!(store.bytes_per_position(), per_position);
        assert_eq!(store.payload_bytes(), want_payload);
        assert_eq!(store.payload_bytes(), tokens * per_position);
        assert_eq!(store.index_bytes(), blocks * 40);
        assert!(store.index_bytes() * BLOCK < store.payload_bytes() * 2);
        // The 9 B/token token-store target is not met by a KV row store, and is not claimed.
        assert!(store.bytes_per_position() > 9);
        c += 1;
    }
}
