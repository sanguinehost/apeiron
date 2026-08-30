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
//!
//! # Why the build is in a second function
//!
//! Same lesson, larger number. With the build loop in the measured frame, the lookup inherits six
//! callee-saved registers and a 264-byte stack frame from an `insert` that inlines a grow path, a
//! rehome and an allocation -- and the register allocator answers by **rematerialising** the fold's
//! seed on every key instead of keeping it live. Split, LLVM hoists `pepper * PHI + PHI` and
//! `pepper * PSI + PSI` out of the loop by itself, which is what they are: loop-invariant.
//!
//! | 13-byte resrefs | shared frame | split |
//! |---|---|---|
//! | apeiron `String` | 107.73 | **94.70** |
//! | hb + foldhash | 84.41 | 80.25 |
//!
//! Ours moved three times as far as theirs, so the shared frame was not a fair harness either. It
//! is also not the shape a caller writes: `map.get(k)` is called from a function that is not
//! simultaneously building the map.
//!
//! **This retires a planned change.** Precomputing the fold's two seed bases in `Salt` was on the
//! list, costing a defaulted method on `Coord` and `Locate` and sixteen bytes on every `Map`. It
//! was worth a measured 6.00 instructions -- all of which were this rematerialisation, and none of
//! which survive the split. There is nothing left for the interface change to buy.
use apeiron::map::{Map, Words};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::hint::black_box;

/// `LOOKUP_N` keys, default 694 -- one Neverwinter area's worth, where everything is L1-resident.
/// A whole module is nearer `100_000` and is not, which is where the two designs stop agreeing.
fn n() -> usize {
    std::env::var("LOOKUP_N").ok().and_then(|v| v.parse().ok()).unwrap_or(694)
}

/// Total lookups is held near constant across sizes so `perf stat` divides by a known number:
/// `REPS * N ~= 138_800_000`.
fn reps() -> usize {
    (138_800_000 / n()).max(1)
}

type Fx = BuildHasherDefault<rustc_hash::FxHasher>;
type Fold = foldhash::fast::RandomState;

fn keys() -> Vec<u64> {
    let mut s = 0x243F_6A88_85A3_08D3u64;
    (0..n())
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s & 0xFFFF_FFFF
        })
        .collect()
}

/// `LOOKUP_KEYS=resref|path` -- 13 bytes, or ~42 with a thirty-character **shared prefix**, which
/// is what a Lyrium `VPath` is. The prefix is the point: it is what makes a failed key compare
/// expensive, and so what the identity word's early reject is worth anything against.
fn strings() -> Vec<String> {
    match std::env::var("LOOKUP_KEYS").unwrap_or_else(|_| "resref".into()).as_str() {
        "path" => (0..n())
            .map(|i| format!("modules/neverwinter/areas/nw_wswmg{i:05}.are"))
            .collect(),
        _ => (0..n()).map(|i| format!("nw_wswmg{i:05}")).collect(),
    }
}

/// Built **pre-sized**. The lookup loop is identical either way -- growth happens before the clock
/// starts -- but at `LOOKUP_N` above the default the table would otherwise pass through a dozen
/// geometries on the way up, and a size sweep wants one geometry per point.
#[inline(never)]
fn ap_words(ks: &[u64]) -> usize {
    let mut m: Words<u64, u64> = Words::with_capacity(ks.len());
    for &k in ks {
        m.insert(k, k);
    }
    ap_words_probe(&m, ks)
}

#[inline(never)]
fn ap_words_probe(m: &Words<u64, u64>, ks: &[u64]) -> usize {
    let mut hits = 0;
    for _ in 0..reps() {
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
    ap_map_probe(&m, ks)
}

#[inline(never)]
fn ap_map_probe(m: &Map<u64, u64>, ks: &[u64]) -> usize {
    let mut hits = 0;
    for _ in 0..reps() {
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
    hb_words_probe(&m, ks)
}

#[inline(never)]
fn hb_words_probe<S: std::hash::BuildHasher>(
    m: &hashbrown::HashMap<u64, u64, S>,
    ks: &[u64],
) -> usize {
    let mut hits = 0;
    for _ in 0..reps() {
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
    std_words_probe(&m, ks)
}

#[inline(never)]
fn std_words_probe(m: &HashMap<u64, u64>, ks: &[u64]) -> usize {
    let mut hits = 0;
    for _ in 0..reps() {
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
    ap_str_probe(&m, ks)
}

/// The lookup loop **alone**, with the build in its own frame.
///
/// Not a cosmetic split. Sharing a frame with `insert` -- which inlines a grow path, a rehome and
/// an allocation -- costs the lookup six callee-saved registers and a 264-byte frame it never
/// wanted, and the register allocator pays for that by *rematerialising* the fold's seed instead
/// of keeping it live. A real caller writes `map.get(k)` from a function that is not also building
/// the map, so this is the faithful shape as well as the cheaper one.
#[inline(never)]
fn ap_str_probe<V>(m: &Map<String, V>, ks: &[String]) -> usize {
    let mut hits = 0;
    for _ in 0..reps() {
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
    ap_map_str_probe(&m, ks)
}

#[inline(never)]
fn ap_map_str_probe(m: &Map<String, u64>, ks: &[String]) -> usize {
    let mut hits = 0;
    for _ in 0..reps() {
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
    hb_str_probe(&m, ks)
}

#[inline(never)]
fn hb_str_probe<S: std::hash::BuildHasher>(
    m: &hashbrown::HashMap<String, u64, S>,
    ks: &[String],
) -> usize {
    let mut hits = 0;
    for _ in 0..reps() {
        for k in ks {
            hits += usize::from(m.contains_key(k.as_str()));
        }
    }
    hits
}

/// `get`, not `contains_key`: the value fetch is a separate question and this crate has been
/// behind on it. `get_vs_contains.rs` records what has already been ruled out.
#[inline(never)]
fn ap_words_get_probe(m: &Words<u64, u64>, ks: &[u64]) -> u64 {
    let mut acc = 0u64;
    for _ in 0..reps() {
        for &k in ks {
            acc = acc.wrapping_add(m.get(&k).copied().unwrap_or(0));
        }
    }
    acc
}

#[inline(never)]
fn ap_words_get(ks: &[u64]) -> usize {
    let mut m: Words<u64, u64> = Words::with_capacity(ks.len());
    for &k in ks {
        m.insert(k, k);
    }
    usize::try_from(ap_words_get_probe(&m, ks) & 0xFFFF_FFFF).unwrap_or(0)
}

#[inline(never)]
fn hb_words_get_probe<S: std::hash::BuildHasher>(
    m: &hashbrown::HashMap<u64, u64, S>,
    ks: &[u64],
) -> u64 {
    let mut acc = 0u64;
    for _ in 0..reps() {
        for &k in ks {
            acc = acc.wrapping_add(m.get(&k).copied().unwrap_or(0));
        }
    }
    acc
}

#[inline(never)]
fn hb_words_get<S: std::hash::BuildHasher + Default>(ks: &[u64]) -> usize {
    let mut m: hashbrown::HashMap<u64, u64, S> = hashbrown::HashMap::default();
    for &k in ks {
        m.insert(k, k);
    }
    usize::try_from(hb_words_get_probe(&m, ks) & 0xFFFF_FFFF).unwrap_or(0)
}

/// A **miss** loop: every key is absent. The probe stops at the first group holding an `EMPTY`
/// tag, so this measures the vacancy scan and not the compare.
#[inline(never)]
fn ap_miss_probe(m: &Words<u64, u64>, ks: &[u64]) -> usize {
    let mut hits = 0;
    for _ in 0..reps() {
        for &k in ks {
            hits += usize::from(m.contains_key(&k));
        }
    }
    hits
}

#[inline(never)]
fn ap_miss(ks: &[u64]) -> usize {
    let mut m: Words<u64, u64> = Words::new();
    for &k in ks {
        m.insert(k, k);
    }
    let absent: Vec<u64> = ks.iter().map(|k| k | 1 << 40).collect();
    ap_miss_probe(&m, &absent)
}

#[inline(never)]
fn hb_miss_probe<S: std::hash::BuildHasher>(
    m: &hashbrown::HashMap<u64, u64, S>,
    ks: &[u64],
) -> usize {
    let mut hits = 0;
    for _ in 0..reps() {
        for &k in ks {
            hits += usize::from(m.contains_key(&k));
        }
    }
    hits
}

#[inline(never)]
fn hb_miss<S: std::hash::BuildHasher + Default>(ks: &[u64]) -> usize {
    let mut m: hashbrown::HashMap<u64, u64, S> = hashbrown::HashMap::default();
    for &k in ks {
        m.insert(k, k);
    }
    let absent: Vec<u64> = ks.iter().map(|k| k | 1 << 40).collect();
    hb_miss_probe(&m, &absent)
}

fn main() {
    let arm = std::env::var("LOOKUP_ARM").unwrap_or_else(|_| "apeiron".into());
    let hits = match arm.as_str() {
        "map" => ap_map(&keys()),
        "apget" => ap_words_get(&keys()),
        "apmiss" => ap_miss(&keys()),
        "foldmiss" => hb_miss::<Fold>(&keys()),
        "fxmiss" => hb_miss::<Fx>(&keys()),
        "foldget" => hb_words_get::<Fold>(&keys()),
        "fxget" => hb_words_get::<Fx>(&keys()),
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
