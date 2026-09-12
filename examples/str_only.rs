//! One string-lookup loop per arm, each in its own function, so the inliner's budget is per-arm.
//!
//! Written this way because it has to be. Sharing one `main` across arms made *our* number move
//! by 12 instructions (110.4 to 122.4) when a third arm was added, while `hashbrown`'s did not
//! move at all: our fold is a length ladder big enough that whether it inlines depends on how
//! large the calling function already is, and `foldhash`'s is small enough to always inline. A
//! harness with every arm in one `main` measures that, not the lookup.
//!
//! `STR_ARM=ap|zst|fold|fx cargo run --release --example str_only`

use apeiron::map::Map;
use std::hint::black_box;

const N: usize = 694;
const REPS: usize = 200_000;

/// `STR_KEYS=resref|path` -- 13 bytes, the shape a Neverwinter resref is, or ~42, the shape a
/// Lyrium `VPath` is. The two sit either side of the fold's 32-byte round, so they take different
/// paths through it and a change to one is not a change to the other.
fn keys() -> Vec<String> {
    match std::env::var("STR_KEYS")
        .unwrap_or_else(|_| "resref".into())
        .as_str()
    {
        "path" => (0..N)
            .map(|i| format!("modules/neverwinter/areas/nw_wswmg{i:05}.are"))
            .collect(),
        _ => (0..N).map(|i| format!("nw_wswmg{i:05}")).collect(),
    }
}

#[inline(never)]
fn ap(ks: &[String]) -> usize {
    let mut m: Map<String, u64> = Map::new();
    for k in ks {
        m.insert(k.clone(), 1);
    }
    ap_probe(&m, ks)
}

/// The lookup loop alone. **The build has to be in its own frame**, and that is worth more than
/// the inliner split above: sharing one with `insert` costs the lookup six callee-saved registers
/// and a 264-byte stack frame, and the register allocator answers by *rematerialising* the fold's
/// seed rather than keeping it live. Split, LLVM hoists `pepper * PHI + PHI` out of the loop by
/// itself. Measured, 13-byte resrefs: 107.73 instructions shared, **94.70** split, against
/// `hashbrown`'s 84.41 and 80.25 -- our number moved three times as far as theirs, so the shared
/// frame was not a fair harness either. A real caller writes `map.get(k)` from a function that is
/// not also building the map.
#[inline(never)]
fn ap_probe(m: &Map<String, u64>, ks: &[String]) -> usize {
    let mut hits = 0;
    for _ in 0..REPS {
        for k in ks {
            hits += usize::from(m.contains_at(black_box(k.as_str())));
        }
    }
    hits
}

/// `Stored<String>` is `{ id: u64, k: String }` = 32 bytes, so a ZST value makes the slot stride a
/// power of two: one `shl $5` for the slot address instead of `lea`-by-5 then `lea`-by-8.
#[inline(never)]
fn zst(ks: &[String]) -> usize {
    let mut m: Map<String, ()> = Map::new();
    for k in ks {
        m.insert(k.clone(), ());
    }
    zst_probe(&m, ks)
}

#[inline(never)]
fn zst_probe(m: &Map<String, ()>, ks: &[String]) -> usize {
    let mut hits = 0;
    for _ in 0..REPS {
        for k in ks {
            hits += usize::from(m.contains_at(black_box(k.as_str())));
        }
    }
    hits
}

#[inline(never)]
fn hb<S: std::hash::BuildHasher + Default>(ks: &[String]) -> usize {
    let mut m: hashbrown::HashMap<String, u64, S> = hashbrown::HashMap::default();
    for k in ks {
        m.insert(k.clone(), 1);
    }
    hb_probe(&m, ks)
}

#[inline(never)]
fn hb_probe<S: std::hash::BuildHasher>(
    m: &hashbrown::HashMap<String, u64, S>,
    ks: &[String],
) -> usize {
    let mut hits = 0;
    for _ in 0..REPS {
        for k in ks {
            hits += usize::from(m.contains_key(black_box(k.as_str())));
        }
    }
    hits
}

fn main() {
    let ks = keys();
    let arm = std::env::var("STR_ARM").unwrap_or_else(|_| "ap".into());
    let hits = match arm.as_str() {
        "zst" => zst(&ks),
        "fold" => hb::<foldhash::fast::RandomState>(&ks),
        "fx" => hb::<std::hash::BuildHasherDefault<rustc_hash::FxHasher>>(&ks),
        _ => ap(&ks),
    };
    println!("{arm}: {}", black_box(hits));
}
