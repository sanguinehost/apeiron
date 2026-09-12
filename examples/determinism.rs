//! Same inputs, same state -- across processes.
//!
//! ARCHITECTURE.md §9 requires that identical initial state plus identical ordered inputs produce
//! identical state, or replay-based debugging and reconciliation are guesswork. A hasher seeded
//! from the stack pointer denies that by construction.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "this is a measurement harness, not library code, and each lint below is listed because this \
              file actually triggers it: sizes and indices narrowed to build keys, bounded by the loop that \
              produced them; signed world coordinates reinterpreted, which is what the packing does. Listed \
              individually rather than suppressed wholesale, so a pedantic lint this file does not yet \
              trigger still surfaces"
)]

use std::collections::HashMap;

use apeiron::coord::{i21, Packed};
use apeiron::table::CountTable;

fn keys() -> Vec<(i32, i32, i32)> {
    let mut r = 0x243F_6A88_85A3_08D3u64;
    let mut next = || {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        r
    };
    (0..2000)
        .map(|_| {
            let v = next();
            let c = |s: u32| ((v >> s) as i32 % 60) - 30;
            (c(0), c(20), c(40))
        })
        .collect()
}

/// A fingerprint of the container's *layout*, not just its contents: iteration order, which is
/// what a replay of anything that walks the map depends on.
fn order_hash<I: IntoIterator<Item = u64>>(it: I) -> u64 {
    it.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, v| {
        (h ^ v).wrapping_mul(0x1000_0000_01b3)
    })
}

fn main() {
    let ks = keys();
    let addrs: Vec<u64> = ks
        .iter()
        .map(|k| {
            (
                i21::new(k.0).unwrap(),
                i21::new(k.1).unwrap(),
                i21::new(k.2).unwrap(),
            )
                .packed()
        })
        .collect();

    let apeiron = {
        let t = CountTable::build(&addrs, 16);
        order_hash(addrs.iter().map(|&a| t.get(a) as u64))
    };

    let mut m: HashMap<(i32, i32, i32), i64, foldhash::fast::RandomState> = HashMap::default();
    for k in &ks {
        *m.entry(*k).or_insert(0) += 1;
    }
    let foldhash_order = order_hash(m.iter().map(|(k, v)| (k.0 as u64) ^ ((*v as u64) << 32)));

    let mut s: HashMap<(i32, i32, i32), i64> = HashMap::new();
    for k in &ks {
        *s.entry(*k).or_insert(0) += 1;
    }
    let std_order = order_hash(s.iter().map(|(k, v)| (k.0 as u64) ^ ((*v as u64) << 32)));

    println!("apeiron {apeiron:#018x}  foldhash {foldhash_order:#018x}  std {std_order:#018x}");
}
