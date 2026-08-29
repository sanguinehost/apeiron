//! The pepper, from both ends: the floor a process sets and the ceiling a type enforces.
//!
//! This file is its own test binary, and that is load-bearing. The process pepper is a pair of
//! statics, and `set_process_pepper` deliberately refuses once a table has read it -- so a test
//! that sets it cannot share a process with a test that does not. Cargo gives each integration
//! test file its own binary, which is the isolation this needs. It is also why everything here
//! runs inside **one** `#[test]`: test functions within a binary run on threads that share those
//! statics, and ordering between them is not something to leave to the scheduler.

use apeiron::coord::spread;
use apeiron::map::{Exposed, Map, Words};
use apeiron::pepper::{
    peek_process_pepper, pepper_is_committed, process_pepper, set_process_pepper,
    try_set_process_pepper,
};
use std::num::NonZeroU64;

const PEPPER: u64 = 0x5EED_1234_ABCD_9876;

/// Keys that land in one group, found by **search** rather than by inversion.
///
/// This is the attack, and the shape of it is the whole result. The placement word is a discard --
/// there is no `unspread` for it -- so an attacker cannot solve for a target group. What they can
/// still do is *evaluate* the function, because it is public, and keep the keys that happen to
/// land where they want. That costs one trial per `2^bits` candidates.
///
/// `bits` is how many of the descent's leading windows the attacker matches. Matching more makes
/// the collision chain longer, and each extra window multiplies the search by `2^gbits`. That
/// trade is what `table::group_of` charges for, and until the identity and placement words were
/// split it could be skipped entirely: the address was a bijection, so matching every window at
/// once was a single call to `unspread`.
fn aimed_by_search(n: usize, bits: u32) -> Option<Vec<u64>> {
    let shift = 57 - bits;
    let mask = (1u64 << bits) - 1;
    let (mut out, mut k, mut tried) = (Vec::with_capacity(n), 1u64, 0u64);
    while out.len() < n {
        k = k.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        tried += 1;
        if (spread(k).place >> shift) & mask == 0 {
            out.push(k);
        }
        // A budget, so a test that would take a week fails instead of hanging. Exceeding it *is*
        // the result for the deepest chain: the attacker cannot afford it either.
        if tried > 400_000_000 {
            return None;
        }
    }
    Some(out)
}

fn honest(n: usize) -> Vec<u64> {
    (0..n as u64).map(|i| i.wrapping_mul(0x9E37_79B9)).collect()
}

#[test]
fn the_floor_the_ceiling_and_the_report() {
    // -- before anything reads it -------------------------------------------------------------
    assert!(!pepper_is_committed(), "something read the pepper before this test");
    assert_eq!(peek_process_pepper(), 0, "the default must be no pepper");

    // -- what a chain costs the attacker, and what it buys them ---------------------------------
    //
    // The claim `table::group_of` makes is that extending a collision chain by one step costs
    // another `gbits` of address control. This crate then corrected it: true, and not the same as
    // *expensive*, because the address was a public **bijection** -- matching every window was one
    // call to `unspread`, and a 200,000-key aimed build ran 988x slower than an honest one.
    //
    // Splitting identity from placement removed that shortcut, and the original claim is now the
    // real one. The attacker can still evaluate the placement word; they cannot invert it. So the
    // cost of a chain is exponential in its length while the damage is only linear, and this
    // asserts both halves of that -- the cheap chain must be shallow, and the deep chain must be
    // unaffordable.
    let n = 2_000usize;
    let mut sizing: Words<u64, u64> = Words::with_pepper(0);
    sizing.reserve(n);
    for i in 0..n as u64 {
        sizing.insert(i, i);
    }
    let gbits = (sizing.slots() / 16).trailing_zeros();
    let good = {
        let mut m: Words<u64, u64> = Words::with_pepper(0);
        for &k in &honest(n) {
            m.insert(k, k);
        }
        m.stats()
    };
    assert!(!good.looks_aimed(), "honest keys tripped the alarm: {good:?}");

    let mut damage = vec![good.mean_probe];
    for windows in 1..=3u32 {
        let Some(ks) = aimed_by_search(n, windows * gbits) else {
            // The search ran out of budget, which is the point: this is the depth an attacker
            // cannot reach, and it must be reached before the damage becomes interesting.
            assert!(windows >= 3, "gave up at only {windows} windows -- the descent is too cheap");
            break;
        };
        let mut m: Words<u64, u64> = Words::with_pepper(0);
        for &k in &ks {
            m.insert(k, k);
        }
        let s = m.stats();
        damage.push(s.mean_probe);
        assert_eq!(m.len(), n, "the aimed set lost keys");
    }

    // Linear in the windows matched. Each step of it cost `2^gbits` times the last.
    for w in damage.windows(2) {
        assert!(
            w[1] > w[0],
            "matching another window bought the attacker nothing: {damage:?}"
        );
        assert!(
            w[1] - w[0] < 2.0,
            "the damage per window is superlinear, so the descent is not bounding it: {damage:?}"
        );
    }
    assert!(
        damage.len() >= 2 && damage.len() <= 3,
        "expected the search to become unaffordable by the third window, got {damage:?}"
    );

    // -- the floor ----------------------------------------------------------------------------
    set_process_pepper(PEPPER);
    // The timing arm below is a real 200,000-key build four times over. Optimised that is a couple
    // of seconds; unoptimised it is two minutes, for a ratio that is just as visible at a tenth the
    // size. The attack does not need scale to show up -- it is quadratic.
    assert_eq!(peek_process_pepper(), PEPPER);

    // A map that says nothing about peppering now gets one anyway. That is the whole point: the
    // maps that need it are the ones written by someone thinking about parsing, not addressing.
    let plain: Map<String, u32> = Map::new();
    assert_eq!(plain.pepper(), PEPPER, "Map::new did not take the process pepper");
    assert!(pepper_is_committed(), "reading it must commit it");

    // And the pepper defends against the search too -- more completely than it defended against
    // the inversion, because the attacker's filter is now the thing that breaks. To keep a
    // candidate they must evaluate the placement word the *table* will use, and with a pepper they
    // cannot: the multiplier is the secret. Keys aimed with the public function land in a peppered
    // table exactly as randomly as honest ones do.
    let n = 2_000usize;
    let h = honest(n);
    let a = aimed_by_search(n, gbits).expect("one window is always affordable");
    let stats_of = |mk: fn() -> Words<u64, u64>, ks: &[u64]| {
        let mut m = mk();
        for &k in ks {
            m.insert(k, k);
        }
        m.stats()
    };
    let bare = stats_of(|| Words::with_pepper(0), &a);
    let floored = stats_of(Words::new, &a);
    let floored_honest = stats_of(Words::new, &h);
    println!(
        "  one aimed window: unpeppered mean probe {:.2}, peppered {:.2}, honest {:.2}",
        bare.mean_probe, floored.mean_probe, floored_honest.mean_probe
    );
    assert!(bare.mean_probe > 1.5, "the attack did not reproduce ({bare:?})");
    assert!(
        floored.mean_probe < bare.mean_probe * 0.75,
        "the process pepper did not defend it: peppered {floored:?} vs bare {bare:?}"
    );

    // -- setting it late is refused ------------------------------------------------------------
    //
    // Two maps in one process must agree on where a key lives. If the pepper moved after the first
    // table was built, half the process would disagree with the other half about every key, and it
    // would surface as an intermittent missing entry a long way from the cause.
    let late = try_set_process_pepper(0xDEAD_BEEF);
    assert!(late.is_err(), "the pepper was allowed to change under a live table");
    assert_eq!(late.unwrap_err().current, PEPPER);
    assert_eq!(peek_process_pepper(), PEPPER, "a refused set must change nothing");
    assert_eq!(process_pepper(), PEPPER);

    // -- the ceiling ---------------------------------------------------------------------------
    //
    // `Exposed` takes a `NonZeroU64`, so an unpeppered one is not a value that exists. There is no
    // runtime check here to forget, which is the difference between this and a convention.
    let key = NonZeroU64::new(PEPPER).expect("nonzero");
    let mut index: Exposed<String, u32> = Exposed::new(key);
    index.insert("nw_wswmg001".into(), 7);
    assert_eq!(index.get_at("nw_wswmg001"), Some(&7)); // the whole Map surface, through Deref
    assert_eq!(index.pepper(), PEPPER);
    assert!(index.stats().peppered);

    assert!(NonZeroU64::new(0).is_none(), "zero is not a pepper");
    let from_process: Option<Exposed<String, u32>> = Exposed::from_process_pepper();
    assert!(from_process.is_some(), "the process set one, so this must succeed");
}
