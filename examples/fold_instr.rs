//! Instructions per fold, which a loaded machine cannot distort the way it distorts cycles.
//!
//! `FOLD_ARM=apeiron|foldhash FOLD_CORPUS=resref|short|path|long cargo run --release --example fold_instr`
//!
//! One `#[inline(never)]` loop per arm, for the reason `probe_cost.rs` documents: with both in one
//! `main` the inliner's budget is shared and a change to either moves both.
use apeiron::fold::{fold_bytes, fold_bytes_with};
use std::hash::BuildHasher;
use std::hint::black_box;

const N: usize = 1000;
const REPS: usize = 20_000;

#[inline(never)]
fn ap(ks: &[String]) -> u64 {
    let mut acc = 0u64;
    for _ in 0..REPS {
        for s in ks {
            acc = acc.wrapping_add(fold_bytes(s.as_bytes()));
        }
    }
    acc
}

/// The same fold under a real pepper, which is the shape a `Map` actually runs: `fold_bytes` folds
/// `NO_PEPPER`, so the seed pair collapses to two constants and the arm above measures a fold whose
/// seeds are free. This one pays for them.
#[inline(never)]
fn ap_peppered(ks: &[String], pepper: u64) -> u64 {
    let mut acc = 0u64;
    for _ in 0..REPS {
        for s in ks {
            acc = acc.wrapping_add(fold_bytes_with(pepper, s.as_bytes()));
        }
    }
    acc
}

#[inline(never)]
fn fh(ks: &[String]) -> u64 {
    let h = foldhash::fast::RandomState::default();
    let mut acc = 0u64;
    for _ in 0..REPS {
        for s in ks {
            acc = acc.wrapping_add(h.hash_one(s.as_str()));
        }
    }
    acc
}

fn main() {
    let kind = std::env::var("FOLD_CORPUS").unwrap_or_else(|_| "resref".into());
    let ks: Vec<String> = (0..N)
        .map(|i| match kind.as_str() {
            "short" => format!("{i}"),
            "path" => format!("data/meshes/creature/c_human_male_{i:04}.mdl"),
            "long" => "x".repeat(200 + i % 64),
            _ => format!("nw_wswmg{i:05}"),
        })
        .collect();
    let arm = std::env::var("FOLD_ARM").unwrap_or_else(|_| "apeiron".into());
    let acc = match arm.as_str() {
        "peppered" => ap_peppered(&ks, black_box(0x5EED_1234_ABCD_9876)),
        "foldhash" => fh(&ks),
        _ => ap(&ks),
    };
    println!("{arm}/{kind}: {}", black_box(acc));
}
