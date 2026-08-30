//! One hit-lookup loop per arm, each in its own function, so `perf stat` measures the probe and
//! not the harness.
//!
//! `LOOKUP_ARM=apeiron|map|fold|fx|std|apstr|apstrzst|foldstr|fxstr cargo run --release --example probe_cost`
//!
//! # Why every arm is `#[inline(never)]`
//!
//! It did not used to be, and the cost of that was a **four-instruction phantom swing**. With all
//! arms in one `main`, `fafa229` -- a change to `fold.rs`, which a `u64` key never calls -- moved
//! this file's `apeiron` arm from 32.82 to 34.82 and its `fold` arm from 35.18 to 33.20. The map
//! did not change. What changed is that `main` got smaller when the long fold moved out of line,
//! so LLVM allocated registers differently across the whole function, and the two extra
//! instructions landed in the harness's own `hits +=` accumulator:
//!
//! | | 265fb9c | fafa229 |
//! |---|---|---|
//! | accumulate a hit | `mov $0x1,%ecx` / `add %rcx,%r14`, falling through | `mov %r8,%r12` / `add %rcx,%r12` / `mov %r12,%r8` / `mov $0x1,%ecx` / `jmp` |
//!
//! Five instructions where there were two, and the probe itself byte-identical either side. One
//! function per arm bounds the inliner's budget to one loop, which is the only way a delta here
//! means the map and not the register allocator. `str_only.rs` learned this first.
use apeiron::map::{Map, Words};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::hint::black_box;

const N: usize = 694; // one Neverwinter area's worth: everything is L1-resident
const REPS: usize = 200_000;

type Fx = BuildHasherDefault<rustc_hash::FxHasher>;
type Fold = foldhash::fast::RandomState;

fn keys() -> Vec<u64> {
    let mut s = 0x243F_6A88_85A3_08D3u64;
    (0..N)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s & 0xFFFF_FFFF
        })
        .collect()
}

fn strings() -> Vec<String> {
    (0..N).map(|i| format!("nw_wswmg{i:05}")).collect()
}

#[inline(never)]
fn ap_words(ks: &[u64]) -> usize {
    let mut m: Words<u64, u64> = Words::new();
    for &k in ks {
        m.insert(k, k);
    }
    let mut hits = 0;
    for _ in 0..REPS {
        for &k in ks {
            hits += usize::from(m.contains_key(&k));
        }
    }
    hits
}

#[inline(never)]
fn ap_map(ks: &[u64]) -> usize {
    let mut m: Map<u64, u64> = Map::new();
    for &k in ks {
        m.insert(k, k);
    }
    let mut hits = 0;
    for _ in 0..REPS {
        for &k in ks {
            hits += usize::from(m.contains_key(&k));
        }
    }
    hits
}

#[inline(never)]
fn hb_words<S: std::hash::BuildHasher + Default>(ks: &[u64]) -> usize {
    let mut m: hashbrown::HashMap<u64, u64, S> = hashbrown::HashMap::default();
    for &k in ks {
        m.insert(k, k);
    }
    let mut hits = 0;
    for _ in 0..REPS {
        for &k in ks {
            hits += usize::from(m.contains_key(&k));
        }
    }
    hits
}

#[inline(never)]
fn std_words(ks: &[u64]) -> usize {
    let mut m: HashMap<u64, u64> = HashMap::new();
    for &k in ks {
        m.insert(k, k);
    }
    let mut hits = 0;
    for _ in 0..REPS {
        for &k in ks {
            hits += usize::from(m.contains_key(&k));
        }
    }
    hits
}

/// `V = ()` brings `Stored<String>` + value to 32 bytes, the same as `hashbrown`'s `(String, u64)`.
/// If slot width explained the `String` gap this arm would close it. It does not: 119.86 against
/// 119.43.
#[inline(never)]
fn ap_str<V: Default + Clone>(ks: &[String]) -> usize {
    let mut m: Map<String, V> = Map::new();
    for k in ks {
        m.insert(k.clone(), V::default());
    }
    let mut hits = 0;
    for _ in 0..REPS {
        for k in ks {
            hits += usize::from(m.contains_at(k.as_str()));
        }
    }
    hits
}

/// `contains_key(&String)`, not `contains_at(&str)`: this is the path through `Stored::holds`,
/// where the identity word is an early-reject hint in front of a `memcmp`. `contains_at` goes
/// through a closure instead and does not share it.
#[inline(never)]
fn ap_map_str(ks: &[String]) -> usize {
    let mut m: Map<String, u64> = Map::new();
    for k in ks {
        m.insert(k.clone(), 1);
    }
    let mut hits = 0;
    for _ in 0..REPS {
        for k in ks {
            hits += usize::from(m.contains_key(k));
        }
    }
    hits
}

#[inline(never)]
fn hb_str<S: std::hash::BuildHasher + Default>(ks: &[String]) -> usize {
    let mut m: hashbrown::HashMap<String, u64, S> = hashbrown::HashMap::default();
    for k in ks {
        m.insert(k.clone(), 1);
    }
    let mut hits = 0;
    for _ in 0..REPS {
        for k in ks {
            hits += usize::from(m.contains_key(k.as_str()));
        }
    }
    hits
}

fn main() {
    let arm = std::env::var("LOOKUP_ARM").unwrap_or_else(|_| "apeiron".into());
    let hits = match arm.as_str() {
        "map" => ap_map(&keys()),
        "fold" => hb_words::<Fold>(&keys()),
        "fx" => hb_words::<Fx>(&keys()),
        "std" => std_words(&keys()),
        "apstr" => ap_str::<u64>(&strings()),
        "mapstr" => ap_map_str(&strings()),
        "apstrzst" => ap_str::<()>(&strings()),
        "foldstr" => hb_str::<Fold>(&strings()),
        "fxstr" => hb_str::<Fx>(&strings()),
        _ => ap_words(&keys()),
    };
    println!("{arm}: {} lookups", black_box(hits));
}
