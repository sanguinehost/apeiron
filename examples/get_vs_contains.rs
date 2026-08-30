//! Where `get` costs more than `contains_key`, and how much of that is ours.
//!
//! `perf stat` on this says the probe is **ahead** of `hashbrown` with `foldhash` on
//! `contains_key` and behind it on `get`, which is a narrower and more useful statement than "the
//! lookup is slower". At 2^20 entries, instructions per operation:
//!
//! | | apeiron `Words` | hb + foldhash |
//! |---|---|---|
//! | `contains_key` | **34.1** | 34.5 |
//! | `get` | 38.1 | **34.5** |
//!
//! foldhash pays the same for both; we pay four more for `get`. Ruled out by measurement: the
//! bounds check on the value fetch (`get_unchecked` changed nothing) and the addressing of it
//! (raw pointer arithmetic changed nothing).
//!
//! What the disassembly of the two hit loops actually shows, per lookup:
//!
//! - **foldhash**: `xor` + `mul` + `xor` for the hash, `shr $0x39` for the tag, and `and` for the
//!   group. One `and`, because its first group is `hash & bucket_mask`.
//! - **apeiron**: the widening multiply, then `rol` + `xor` for the placement word, then
//!   `or $0x80` for the tag's live bit, then a **variable** shift whose amount is loaded from the
//!   `Geom` on every iteration -- `mov 0x30(%rsp),%rcx` and `shr %cl,%rax` -- before the mask.
//!
//! Two of those are the design earning its keep: the `rol`/`xor` is the identity/placement split,
//! which is what makes the placement word non-invertible. The other two are not, and are the
//! places to look next -- the live bit could be an *absent* bit instead, needing no `or`, and the
//! first group need not come from a variable window.
//!
//! **`REPS` is 200 on purpose.** At `REPS = 8` the map's construction is a sixth of the work and
//! shows up as a flat ~6-instruction penalty at every size, which reads exactly like a lookup
//! difference and is not one. That mistake is why this file exists rather than a one-off script.
//!
//! `ARM=ap_get|ap_has|fold_get|fold_has|fx_get BITS=20 perf stat -e instructions ...`
use apeiron::map::Words;
use std::hash::BuildHasherDefault;
use std::hint::black_box;
type Fx = BuildHasherDefault<rustc_hash::FxHasher>;
const REPS: usize = 200;
fn main() {
    let bits: u32 = std::env::var("BITS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let n: usize = 1 << bits;
    let ks: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
    let probe: Vec<u64> = ks.iter().rev().copied().collect();
    // Absent keys: the same shape, disjoint from the built set.
    let missing: Vec<u64> = ks.iter().map(|k| k ^ 0x8000_0000_0000_0001).collect();
    let arm = std::env::var("ARM").unwrap_or_else(|_| "ap_get".into());
    let mut acc = 0u64;
    match arm.as_str() {
        "fold_get" => { let m: hashbrown::HashMap<u64,u64,foldhash::fast::RandomState> = ks.iter().map(|&k|(k,k)).collect();
            for _ in 0..REPS { for k in &probe { acc = acc.wrapping_add(m.get(k).copied().unwrap_or(0)); } } }
        "fold_has" => { let m: hashbrown::HashMap<u64,u64,foldhash::fast::RandomState> = ks.iter().map(|&k|(k,k)).collect();
            for _ in 0..REPS { for k in &probe { acc += u64::from(m.contains_key(k)); } } }
        "fx_get" => { let m: hashbrown::HashMap<u64,u64,Fx> = ks.iter().map(|&k|(k,k)).collect();
            for _ in 0..REPS { for k in &probe { acc = acc.wrapping_add(m.get(k).copied().unwrap_or(0)); } } }
        "ap_miss" => { let m: Words<u64,u64> = ks.iter().map(|&k|(k,k)).collect();
            for _ in 0..REPS { for k in &missing { acc = acc.wrapping_add(m.get(k).copied().unwrap_or(0)); } } }
        "fold_miss" => { let m: hashbrown::HashMap<u64,u64,foldhash::fast::RandomState> = ks.iter().map(|&k|(k,k)).collect();
            for _ in 0..REPS { for k in &missing { acc = acc.wrapping_add(m.get(k).copied().unwrap_or(0)); } } }
        "ap_has" => { let m: Words<u64,u64> = ks.iter().map(|&k|(k,k)).collect();
            for _ in 0..REPS { for k in &probe { acc += u64::from(m.contains_key(k)); } } }
        _ => { let m: Words<u64,u64> = ks.iter().map(|&k|(k,k)).collect();
            for _ in 0..REPS { for k in &probe { acc = acc.wrapping_add(m.get(k).copied().unwrap_or(0)); } } }
    }
    println!("{}", black_box(acc));
}
