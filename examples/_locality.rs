use apeiron::map::Words;
use std::hint::black_box;
fn main() {
    let n = 10_000usize;
    let which = std::env::var("DIST").unwrap_or_else(|_| "serial".into());
    let ks: Vec<u64> = match which.as_str() {
        "serial" => (0..n as u64).collect(),
        "highbits" => (0..n as u64).map(u64::swap_bytes).collect(),
        _ => { let mut s = 0u64; (0..n).map(|_| { s = s.wrapping_add(1).wrapping_mul(3_787_392_781); s }).collect() }
    };
    let arm = std::env::var("ARM").unwrap_or_else(|_| "ap".into());
    let mut acc = 0usize;
    if arm == "hb" {
        let m: hashbrown::HashMap<u64, usize, foldhash::fast::RandomState> = ks.iter().map(|&k|(k,1)).collect();
        for _ in 0..2000 { for k in &ks { acc += m.get(k).copied().unwrap_or(0); } }
    } else {
        let m: Words<u64, usize> = ks.iter().map(|&k|(k,1)).collect();
        for _ in 0..2000 { for k in &ks { acc += m.get(k).copied().unwrap_or(0); } }
    }
    println!("{}", black_box(acc));
}
