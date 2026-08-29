//! One hit-lookup loop, nothing else, so `perf stat` measures the probe and not the harness.
//!
//! `LOOKUP_ARM=apeiron|fx|std cargo run --release --example probe_cost`
use apeiron::map::{Map, Words};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::hint::black_box;

const N: usize = 694; // one Neverwinter area's worth: everything is L1-resident
const REPS: usize = 200_000;

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

fn main() {
    let ks = keys();
    let arm = std::env::var("LOOKUP_ARM").unwrap_or_else(|_| "apeiron".into());
    let mut hits = 0usize;
    match arm.as_str() {
        "fx" => {
            type Fx = BuildHasherDefault<rustc_hash::FxHasher>;
            let mut m: hashbrown::HashMap<u64, u64, Fx> = hashbrown::HashMap::default();
            for &k in &ks { m.insert(k, k); }
            for _ in 0..REPS { for &k in &ks { hits += usize::from(m.contains_key(&k)); } }
        }
        "fold" => {
            let mut m: hashbrown::HashMap<u64, u64, foldhash::fast::RandomState> =
                hashbrown::HashMap::default();
            for &k in &ks { m.insert(k, k); }
            for _ in 0..REPS { for &k in &ks { hits += usize::from(m.contains_key(&k)); } }
        }
        "foldstr" => {
            let ks: Vec<String> = (0..N).map(|i| format!("nw_wswmg{i:05}")).collect();
            let mut m: hashbrown::HashMap<String, u64, foldhash::fast::RandomState> =
                hashbrown::HashMap::default();
            for k in &ks { m.insert(k.clone(), 1); }
            for _ in 0..REPS { for k in &ks { hits += usize::from(m.contains_key(k.as_str())); } }
        }
        "std" => {
            let mut m: HashMap<u64, u64> = HashMap::new();
            for &k in &ks { m.insert(k, k); }
            for _ in 0..REPS { for &k in &ks { hits += usize::from(m.contains_key(&k)); } }
        }
        "apstr" => {
            let ks: Vec<String> = (0..N).map(|i| format!("nw_wswmg{i:05}")).collect();
            let mut m: Map<String, u64> = Map::new();
            for k in &ks { m.insert(k.clone(), 1); }
            for _ in 0..REPS { for k in &ks { hits += usize::from(m.contains_at(k.as_str())); } }
        }
        // Slot width, isolated: `()` values bring `Stored<String>` + value to 32 bytes, the same
        // as hashbrown's `(String, u64)`. If slot width explained the `String` gap this arm would
        // close it.
        "apstrzst" => {
            let ks: Vec<String> = (0..N).map(|i| format!("nw_wswmg{i:05}")).collect();
            let mut m: Map<String, ()> = Map::new();
            for k in &ks { m.insert(k.clone(), ()); }
            for _ in 0..REPS { for k in &ks { hits += usize::from(m.contains_at(k.as_str())); } }
        }
        "fxstr" => {
            type Fx = BuildHasherDefault<rustc_hash::FxHasher>;
            let ks: Vec<String> = (0..N).map(|i| format!("nw_wswmg{i:05}")).collect();
            let mut m: hashbrown::HashMap<String, u64, Fx> = hashbrown::HashMap::default();
            for k in &ks { m.insert(k.clone(), 1); }
            for _ in 0..REPS { for k in &ks { hits += usize::from(m.contains_key(k.as_str())); } }
        }
        _ => {
            let mut m: Words<u64, u64> = Words::new();
            for &k in &ks { m.insert(k, k); }
            for _ in 0..REPS { for &k in &ks { hits += usize::from(m.contains_key(&k)); } }
        }
    }
    println!("{arm}: {} lookups", black_box(hits));
}
