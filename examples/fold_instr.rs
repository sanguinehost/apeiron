//! Instructions per fold, which a loaded machine cannot distort the way it distorts cycles.
//!
//! `FOLD_ARM=apeiron|foldhash cargo run --release --example fold_instr`
use apeiron::fold::fold_bytes;
use std::hash::BuildHasher;
use std::hint::black_box;

const N: usize = 1000;
const REPS: usize = 20_000;

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
    let mut acc = 0u64;
    if arm == "apeiron" {
        for _ in 0..REPS {
            for s in &ks {
                acc = acc.wrapping_add(fold_bytes(s.as_bytes()));
            }
        }
    } else {
        let h = foldhash::fast::RandomState::default();
        for _ in 0..REPS {
            for s in &ks {
                acc = acc.wrapping_add(h.hash_one(s.as_str()));
            }
        }
    }
    println!("{arm}/{kind}: {}", black_box(acc));
}
