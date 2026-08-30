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
//! # The 150k-400k band, and eight things it is not
//!
//! Between roughly 150,000 and 400,000 entries -- where the table fits in L3 but not L2 -- a
//! `Words<u64, u64>` lookup costs 8-21% more **cycles** than `hashbrown` + `foldhash` at equal
//! instructions and *fewer* cache misses at every level. Above and below that band we are at
//! parity. Measured at n=300,000: 15.44 cycles against 13.10.
//!
//! Eight explanations have been tested and refuted, each with the counter that kills it:
//!
//! | hypothesis | measurement |
//! |---|---|
//! | spilled loop invariants | constant-folding `gshift`/`gmask` takes L1 loads 7.07 -> 5.08 and cycles 15.44 -> **15.75** |
//! | outlining the cold descent | removes both spills, costs **+5.4 instructions**, cycles unmoved |
//! | bounding the group index by the shift | correct, **+1.01 instructions**, cycles worse at every size |
//! | TLB pressure | 0.0003 misses/lookup against `hashbrown`'s 0.0000; 0.013 walk cycles of 16.09 |
//! | probe displacement | **anti-correlated**: 0.3% displaced at 300k where the gap is 21%, 2.9% at 100k where it is 8% |
//! | aligned groups vs a sliding window | simulated: 0.37% -> 0.21% displacement, worth ~0.03 cycles |
//! | fill factor | 5/8 instead of 7/8 changes nothing (same slot count at this n) |
//! | branch mispredicts | 0.0336/lookup against 0.0069, but `int_misc.clear_resteer_cycles` prices them at **0.09 of 15.98 cycles** |
//!
//! What *is* established: top-down puts both arms at ~54% backend bound with near-identical memory
//! stalls (2.55 cycles against 2.34, and we are **lower** on L3 stalls), so the difference lives in
//! the ~13 non-stalled cycles. We issue **7.07 L1 loads per lookup against 4.06**, and
//! `topdown-heavy-ops` is 303.9M against 170.6M -- but deleting two of those loads changed nothing,
//! so they are absorbed by the out-of-order engine rather than serialising.
//!
//! # It is the **aligned group**, and the layout has nothing to do with it
//!
//! The single-allocation theory was tested and is **refuted**. A standalone table with the same
//! geometry and the same probe, built three ways -- two `Vec`s, one block with tags then slots, and
//! `hashbrown`'s own layout with slots *below* the ctrl pointer addressed backwards -- measures the
//! same to within noise at every size:
//!
//! | n | two allocations | one block | backwards from ctrl |
//! |---|---|---|---|
//! | 100,000 | 12.95 | 13.82 | 14.16 |
//! | 300,000 | 16.50 | 16.22 | 16.17 |
//! | 400,000 | 18.82 | 19.08 | 19.81 |
//!
//! Swapping the **probe** into that same harness moves it immediately, and splitting `hashbrown`'s
//! probe into its two halves says which half:
//!
//! | n = 400,000 | first group | stride after | cycles |
//! |---|---|---|---|
//! | ours | aligned window | aligned window | 18.82 |
//! | | aligned window | **triangular** | 21.71 (worse) |
//! | | **unaligned** | aligned window | **16.62** |
//! | `hashbrown` | unaligned | triangular | 14.72 |
//!
//! The triangular stride is not the win -- taking it alone is a **loss**. The win is that
//! `hashbrown`'s window can start at *any* slot, so a group is centred on the key's own position,
//! while ours starts only at multiples of sixteen and every key in a group reads the same sixteen
//! tags.
//!
//! **And it is separable from the reason the low-bit window was rejected.** `Geom::first`'s doc
//! records that reading the window from the bottom of the placement word fails the flatness gate --
//! a product's low bits are its least mixed. That is about *which bits*, not about alignment.
//! Keeping the top window and dropping only `& !(GROUP - 1)`:
//!
//! | n | aligned (ours) | top window, unaligned | `hashbrown` |
//! |---|---|---|---|
//! | 100,000 | 12.95 | 12.52 | 11.25 |
//! | 300,000 | 16.50 | **14.25** | 13.51 |
//! | 400,000 | 18.82 | **16.18** | 14.72 |
//!
//! That recovers about 62% of the gap while keeping the entropy source the crate proved it needs.
//!
//! # Implemented, gated, and reverted
//!
//! It was built. Two Kani harnesses were written *before* the unsafe code, and both prove:
//! `a_clamped_unaligned_window_stays_inside_the_region` (clamping the offset to `slots - GROUP`
//! keeps `line`'s sixteen-byte read in bounds, so `tags.len() == slots` stays true and no
//! iterator, `retain`, `Drain` or `stats` walk has to learn about a mirrored tail) and
//! `the_unaligned_window_is_the_aligned_one_plus_an_offset` (the unaligned window reads the
//! *same* top bits, plus four offset bits below them -- the obligation `Geom::first`'s doc
//! exists to enforce).
//!
//! **That second proof caught an error in the benchmark above.** The fast arm shifted by the same
//! amount and dropped the `* GROUP`, which reads a window four bits *lower* rather than the same
//! window unaligned -- exactly the least-mixed bits the flatness gate rejected. Corrected, the
//! standalone win was still real: 19.04 -> 15.64 cycles at n=400,000.
//!
//! Two things killed it anyway.
//!
//! **The flatness gate.** `i21hi` at n=1,500,000 -- a world varying along one axis, a real Lyrium
//! key shape and not a synthetic corner -- gives a longest probe of **23 against the bound of
//! 16**, with the mean untouched at 1.011. Unaligned windows overlap, so a full group spills into
//! a neighbour that is already partly full and the tail clusters in a way the aligned form cannot.
//! `hashbrown` answers this with its triangular stride; a simulation agrees that the stride owns
//! the tail. This crate cannot take that trade: the window descent is what makes aiming cost
//! `2^(j * gbits)` per step, and a stride that consumes no address bits gives that up.
//!
//! **And in the real crate the win did not survive contact.** The standalone harness has no
//! `Descent`, no tombstone bookkeeping and no `Salt`, so the clamp's `cmp`/`cmov` and the second
//! `Geom` field were free there and are not here: **+1.8 instructions**, and 15.98 cycles at
//! n=300,000 against the 15.84 it was meant to beat.
//!
//! The proofs are kept. They cost nothing, they are true of the geometry either way, and they are
//! what a second attempt would have had to write first.
//!
//! What is left of the diagnosis: the aligned group **is** the cause of the band, and the fix for
//! it needs a stride this crate will not adopt. That is a design boundary rather than an open
//! question.
//!
//! The remaining sketch, kept for the record: `line()` reads
//! `base..base + GROUP` with no wraparound, so an unaligned base near the end of the region reads
//! past the array; `hashbrown` pays for this by duplicating its first fifteen control bytes at the
//! end. Group alignment is also assumed by `Descent`, by `rehome`, and by the `stats` walk. That is
//! a real change to the unsafe surface, it must pass `tests/the_probe_stays_flat.rs` on every
//! distribution that has ever broken this crate, and it should be started fresh rather than
//! appended to the session that found it.
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
