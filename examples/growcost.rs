//! The build path *with* growth, which `write_path` deliberately excludes.
//!
//! `write_path` reserves up front, so it prices the address, the probe and the write. This one
//! does not reserve, so every rebuild pays a fresh tag region as well -- which is the only place
//! the empty sentinel's value can cost anything.
//!
//! It exists because the sentinel was zero on the argument that a fresh region is then a `calloc`
//! the kernel may hand over lazily, rather than a pass of writes. Measured at 2^22, five rounds
//! each: 113.7 ms mean with the zero sentinel against 105.8 ms with `0xFF`. The fill is real and
//! it is smaller than the two instructions per insert that the other polarity costs, so the
//! argument was true about the allocation and wrong about the total.
//!
//! `ARM=ap|fold BITS=22 REPS=6 cargo run --release --example growcost`

use apeiron::map::Words;
use foldhash::fast::RandomState;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let bits: u32 = std::env::var("BITS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let n = 1usize << bits;
    let mut s = 0x243F_6A88_85A3_08D3u64;
    let ks: Vec<u64> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        })
        .collect();
    let arm = std::env::var("ARM").unwrap_or_else(|_| "ap".into());

    let mut acc = 0usize;
    let t = Instant::now();
    for _ in 0..reps {
        if arm == "fold" {
            let mut m: hashbrown::HashMap<u64, u64, RandomState> = hashbrown::HashMap::default();
            for &k in &ks {
                m.insert(k, k);
            }
            acc += m.len();
        } else {
            let mut m: Words<u64, u64> = Words::new();
            for &k in &ks {
                m.insert(k, k);
            }
            acc += m.len();
        }
    }
    let per = t.elapsed().as_secs_f64() * 1000.0 / f64::from(u32::try_from(reps).unwrap());
    println!("{arm} 2^{bits}: {per:.2} ms/build  ({})", black_box(acc));
}
