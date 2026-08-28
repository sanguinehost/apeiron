use apeiron::coord::{bucket, i21, Packed};
use apeiron::table::CountTable;
use std::time::Instant;

fn main() {
    let n = 1_000_000usize;
    let mut r = 0x243F_6A88_85A3_08D3u64;
    let mut next = || { r ^= r << 13; r ^= r >> 7; r ^= r << 17; r };
    let keys: Vec<u64> = (0..n).map(|_| {
        let v = next();
        let c = |s: u32| ((v >> s) as i32 % 400) - 200;
        (i21::new(c(0)).unwrap(), i21::new(c(20)).unwrap(), i21::new(c(40)).unwrap()).packed()
    }).collect();
    let mut occ = keys.clone(); occ.sort_unstable(); occ.dedup();
    let bits = (occ.len() * 2).next_power_of_two().trailing_zeros();

    let go = |label: &str, th: usize, f: &dyn Fn() -> (CountTable, f64)| {
        let (t, build) = f();
        let s = Instant::now();
        let acc = if th <= 1 { occ.iter().map(|&a| t.get(a)).sum::<i64>() }
                  else { t.sum_parallel(&occ, th) };
        let probe = s.elapsed().as_secs_f64() * 1000.0;
        println!("  {label:<24} build {build:>7.2} ms  probe {probe:>7.2} ms  total {:>7.2} ms  acc {acc}",
                 build + probe);
    };

    // unpartitioned baseline, for the comparison
    let s = Instant::now();
    let m = 1usize << bits;
    let mut flat = vec![(u64::MAX, 0i64); m];
    for &a in &keys {
        let mut i = bucket(a, bits) as usize;
        loop {
            if flat[i].0 == u64::MAX { flat[i] = (a, 1); break; }
            if flat[i].0 == a { flat[i].1 += 1; break; }
            i = (i + 1) & (m - 1);
        }
    }
    let fb = s.elapsed().as_secs_f64() * 1000.0;
    let s = Instant::now();
    let mut acc = 0i64;
    for &a in &occ {
        let mut i = bucket(a, bits) as usize;
        loop {
            if flat[i].0 == u64::MAX { break; }
            if flat[i].0 == a { acc += flat[i].1; break; }
            i = (i + 1) & (m - 1);
        }
    }
    let fp = s.elapsed().as_secs_f64() * 1000.0;
    println!("  {:<24} build {fb:>7.2} ms  probe {fp:>7.2} ms  total {:>7.2} ms  acc {acc}",
             "flat (unpartitioned)", fb + fp);

    go("partitioned, 1 thread", 1, &|| {
        let s = Instant::now();
        let t = CountTable::build(&keys, bits);
        (t, s.elapsed().as_secs_f64() * 1000.0)
    });
    for th in [8usize, 16, 32] {
        go(&format!("partitioned, {th} threads"), th, &|| {
            let s = Instant::now();
            let t = CountTable::build_parallel(&keys, bits, th);
            (t, s.elapsed().as_secs_f64() * 1000.0)
        });
    }
}

