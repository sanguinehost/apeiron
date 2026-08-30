//! The write path, isolated: insert into a table that has already reserved.
//!
//! `whole_surface` says `insert reserved` is the widest gap against `hashbrown` + `foldhash`.
//! Growth is excluded on purpose -- with capacity reserved up front, what is left is the address,
//! the probe for a vacant slot, and the write.
//!
//! `ARM=ap_ins|fold_ins|ap_entry|fold_entry|ap_str|fold_str BITS=17 perf stat -e instructions ...`

use apeiron::map::{Map, Words};
use foldhash::fast::RandomState;
use std::hint::black_box;

const REPS: usize = 200;

fn main() {
    let bits: u32 = std::env::var("BITS").ok().and_then(|v| v.parse().ok()).unwrap_or(17);
    let n: usize = 1 << bits;
    let mut s = 0x243F_6A88_85A3_08D3u64;
    let ks: Vec<u64> = (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s }).collect();
    let strs: Vec<String> = (0..n).map(|i| format!("nw_wswmg{i:07}")).collect();
    let arm = std::env::var("ARM").unwrap_or_else(|_| "ap_ins".into());
    let mut acc = 0usize;
    for _ in 0..REPS {
        match arm.as_str() {
            "fold_ins" => {
                let mut m: hashbrown::HashMap<u64, u64, RandomState> =
                    hashbrown::HashMap::with_capacity_and_hasher(n, RandomState::default());
                for &k in &ks { m.insert(k, k); }
                acc += m.len();
            }
            "ap_entry" => {
                let mut m: Words<u64, u64> = Words::with_capacity(n);
                for &k in &ks { *m.entry(k).or_insert(0) += 1; }
                acc += m.len();
            }
            "fold_entry" => {
                let mut m: hashbrown::HashMap<u64, u64, RandomState> =
                    hashbrown::HashMap::with_capacity_and_hasher(n, RandomState::default());
                for &k in &ks { *m.entry(k).or_insert(0) += 1; }
                acc += m.len();
            }
            "ap_str" => {
                let mut m: Map<String, u64> = Map::with_capacity(n);
                for k in &strs { m.insert(k.clone(), 1); }
                acc += m.len();
            }
            "fold_str" => {
                let mut m: hashbrown::HashMap<String, u64, RandomState> =
                    hashbrown::HashMap::with_capacity_and_hasher(n, RandomState::default());
                for k in &strs { m.insert(k.clone(), 1); }
                acc += m.len();
            }
            _ => {
                let mut m: Words<u64, u64> = Words::with_capacity(n);
                for &k in &ks { m.insert(k, k); }
                acc += m.len();
            }
        }
    }
    println!("{}", black_box(acc));
}
