//! The pepper, from both ends: the floor a process sets and the ceiling a type enforces.
//!
//! This file is its own test binary, and that is load-bearing. The process pepper is a pair of
//! statics, and `set_process_pepper` deliberately refuses once a table has read it -- so a test
//! that sets it cannot share a process with a test that does not. Cargo gives each integration
//! test file its own binary, which is the isolation this needs. It is also why everything here
//! runs inside **one** `#[test]`: test functions within a binary run on threads that share those
//! statics, and ordering between them is not something to leave to the scheduler.

use apeiron::coord::unspread;
use apeiron::map::{Exposed, Map, Words};
use apeiron::pepper::{
    peek_process_pepper, pepper_is_committed, process_pepper, set_process_pepper,
    try_set_process_pepper,
};
use std::num::NonZeroU64;
use std::time::Instant;

const PEPPER: u64 = 0x5EED_1234_ABCD_9876;

/// Keys whose addresses agree on their top bits under the *unpeppered* bijection.
///
/// `unspread` is public and exact, so this is arithmetic and not search -- which is the whole
/// point. An attacker who can read this crate can write this function.
fn aimed(n: usize) -> Vec<u64> {
    (0..n as u64)
        .map(|i| unspread(0xABCD_E000_0000_0000 | i))
        .collect()
}

fn honest(n: usize) -> Vec<u64> {
    (0..n as u64).map(|i| i.wrapping_mul(0x9E37_79B9)).collect()
}

fn build_ms<F: Fn() -> Words<u64, u64>>(make: F, ks: &[u64]) -> f64 {
    let t = Instant::now();
    let mut m = make();
    for &k in ks {
        m.insert(k, k);
    }
    assert_eq!(m.len(), ks.len());
    t.elapsed().as_secs_f64() * 1e3
}

#[test]
fn the_floor_the_ceiling_and_the_report() {
    // -- before anything reads it -------------------------------------------------------------
    assert!(!pepper_is_committed(), "something read the pepper before this test");
    assert_eq!(peek_process_pepper(), 0, "the default must be no pepper");

    // -- the report makes an attack visible, peppered or not ----------------------------------
    //
    // This runs first, on an explicitly unpeppered map, because it is the diagnostic a server
    // operator has when nobody set a pepper. `stats()` must say so.
    let n = 20_000;
    let mut wide_open: Words<u64, u64> = Words::with_pepper(0);
    for &k in &aimed(n) {
        wide_open.insert(k, k);
    }
    let bad = wide_open.stats();
    assert!(!bad.peppered, "stats must report an unpeppered table");
    assert!(
        bad.looks_aimed(),
        "an aimed key set must be visible in the report, got {bad:?}"
    );

    let mut ordinary: Words<u64, u64> = Words::with_pepper(0);
    for &k in &honest(n) {
        ordinary.insert(k, k);
    }
    let good = ordinary.stats();
    assert!(
        !good.looks_aimed(),
        "an honest key set must not trip the alarm, got {good:?}"
    );
    // The report has to separate them, but not on `longest_probe`: asserting a 10x gap there
    // pinned this test to a probe layout rather than to the property, and broke when the descent
    // moved to bit 57 -- this aim's varying low bits now split the chain by the fifth window, so
    // its longest probe is seven. What an aimed set cannot avoid is walking shared groups and,
    // having agreed on the top of the word, sharing the tag: measured, mean probe 4.99 against
    // 1.00 and 69.5 address compares per hit against 1.04. Asserted at half and a tenth of those
    // margins, so the property has room to breathe and a collapse still fails.
    assert!(
        bad.mean_probe > good.mean_probe * 2.0,
        "the probe report is not separating them: aimed {bad:?} vs honest {good:?}"
    );
    assert!(
        bad.mean_compares > good.mean_compares * 6.0,
        "the compare report is not separating them: aimed {bad:?} vs honest {good:?}"
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

    // And it defends. Same keys, same construction, built through the process floor.
    let n = if cfg!(debug_assertions) { 20_000 } else { 200_000 };
    let (a, h) = (aimed(n), honest(n));
    let raw = build_ms(|| Words::with_pepper(0), &a) / build_ms(|| Words::with_pepper(0), &h);
    let floored = build_ms(Words::new, &a) / build_ms(Words::new, &h);
    println!("  aimed/honest: unpeppered {raw:.0}x, process pepper {floored:.1}x");
    assert!(raw > 20.0, "the attack did not reproduce ({raw:.1}x)");
    assert!(floored < 4.0, "the process pepper did not defend it ({floored:.1}x)");

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
