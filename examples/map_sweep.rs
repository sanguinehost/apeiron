//! Where the keyed faces stand against the maps people actually use, at the sizes people use them.
//!
//! The million-key figure is the one that sells a table and the hundred-key figure is the one that
//! decides whether it belongs in a game loop. Both are here, and so is everything between.
//!
//! Four columns, because "beating hashbrown" means nothing without saying which hasher:
//! `std` is `HashMap` with SipHash-1-3 (seeded, so DoS-resistant and not deterministic across
//! runs), `hb+fold` is hashbrown with foldhash, `hb+fx` is hashbrown with fxhash (fast and
//! trivially attackable), and `htrin` is the fold crate's `Table`.

use apeiron::map::{Map, Words};
use hashbrown::HashMap as HbMap;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::hint::black_box;
use std::time::Instant;

type Fold = foldhash::fast::RandomState;
type Fx = BuildHasherDefault<rustc_hash::FxHasher>;

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn time(reps: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..reps.clamp(1, 3) {
        f();
    }
    let mut runs: Vec<f64> = (0..7)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            t.elapsed().as_secs_f64() * 1e9 / reps as f64
        })
        .collect();
    runs.sort_by(f64::total_cmp);
    runs[3]
}

/// Nanoseconds per key, so numbers three orders of magnitude apart stay comparable.
fn row(label: &str, n: usize, cols: &[f64]) {
    let best = cols.iter().copied().fold(f64::INFINITY, f64::min);
    print!("  {label:<10} {n:>8}");
    for &c in cols {
        let mark = if (c - best).abs() < f64::EPSILON { '*' } else { ' ' };
        print!(" {c:>9.2}{mark}");
    }
    println!();
}

fn head(what: &str) {
    println!("\n{what}   (ns per key, median of 7; * = best)");
    println!(
        "  {:<10} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "", "n", "apeiron", "std", "hb+fold", "hb+fx"
    );
}

fn int_keys(n: usize) -> Vec<u64> {
    let mut next = rng(0x243F_6A88_85A3_08D3);
    let mut ks: Vec<u64> = (0..n).map(|_| next() & 0xFFFF_FFFF).collect();
    ks.sort_unstable();
    ks.dedup();
    ks
}

/// Resource references: the shape a Neverwinter index is actually keyed by -- short, a shared
/// prefix, a numeric tail. A corpus of random hex would flatter every hasher equally and would not
/// be the workload.
fn resrefs(n: usize) -> Vec<String> {
    let stems = [
        "nw_wswmg", "nw_it_msmlmisc", "x2_it_drowcloth", "plc_barrel", "c_dragred",
        "nw_maarcl", "x0_maceflail", "tr_area", "pnl_chr_gen", "gui_icon",
    ];
    (0..n).map(|i| format!("{}{:03}", stems[i % stems.len()], i)).collect()
}

fn ints(sizes: &[usize]) {
    head("integer keys, u64 -> u64: build then look every key up");
    for &n in sizes {
        let ks = int_keys(n);
        let n = ks.len();
        let reps = (2_000_000 / n.max(1)).clamp(3, 2000);

        let ap = time(reps, || {
            let mut m: Words<u64, u64> = Words::with_capacity(ks.len());
            for &k in &ks {
                m.insert(k, k);
            }
            black_box(ks.iter().filter_map(|k| m.get(k)).count());
        });
        let st = time(reps, || {
            let mut m: HashMap<u64, u64> = HashMap::with_capacity(ks.len());
            for &k in &ks {
                m.insert(k, k);
            }
            black_box(ks.iter().filter_map(|k| m.get(k)).count());
        });
        let hf = time(reps, || {
            let mut m: HbMap<u64, u64, Fold> = HbMap::with_capacity_and_hasher(ks.len(), Fold::default());
            for &k in &ks {
                m.insert(k, k);
            }
            black_box(ks.iter().filter_map(|k| m.get(k)).count());
        });
        let hx = time(reps, || {
            let mut m: HbMap<u64, u64, Fx> = HbMap::with_capacity_and_hasher(ks.len(), Fx::default());
            for &k in &ks {
                m.insert(k, k);
            }
            black_box(ks.iter().filter_map(|k| m.get(k)).count());
        });
        row("build+get", n, &[
            ap / n as f64,
            st / n as f64,
            hf / n as f64,
            hx / n as f64,
        ]);
    }
}

fn lookups(sizes: &[usize]) {
    head("integer keys: lookup only, on a built table (hits)");
    for &n in sizes {
        let ks = int_keys(n);
        let n = ks.len();
        let reps = (4_000_000 / n.max(1)).clamp(3, 4000);

        let mut ap: Words<u64, u64> = Words::with_capacity(n);
        let mut st: HashMap<u64, u64> = HashMap::with_capacity(n);
        let mut hf: HbMap<u64, u64, Fold> = HbMap::with_capacity_and_hasher(n, Fold::default());
        let mut hx: HbMap<u64, u64, Fx> = HbMap::with_capacity_and_hasher(n, Fx::default());
        for &k in &ks {
            ap.insert(k, k);
            st.insert(k, k);
            hf.insert(k, k);
            hx.insert(k, k);
        }
        row("get", n, &[
            time(reps, || { black_box(ks.iter().filter_map(|k| ap.get(k)).count()); }) / n as f64,
            time(reps, || { black_box(ks.iter().filter_map(|k| st.get(k)).count()); }) / n as f64,
            time(reps, || { black_box(ks.iter().filter_map(|k| hf.get(k)).count()); }) / n as f64,
            time(reps, || { black_box(ks.iter().filter_map(|k| hx.get(k)).count()); }) / n as f64,
        ]);
    }
}

fn strings(sizes: &[usize]) {
    head("resref keys, String -> u32: build then look every key up");
    for &n in sizes {
        let ks = resrefs(n);
        let reps = (1_000_000 / n.max(1)).clamp(3, 1000);

        let ap = time(reps, || {
            let mut m: Map<String, u32> = Map::with_capacity(n);
            for (i, k) in ks.iter().enumerate() {
                m.insert_at(k.as_str(), i as u32);
            }
            black_box(ks.iter().filter_map(|k| m.get_at(k.as_str())).count());
        });
        let st = time(reps, || {
            let mut m: HashMap<String, u32> = HashMap::with_capacity(n);
            for (i, k) in ks.iter().enumerate() {
                m.insert(k.clone(), i as u32);
            }
            black_box(ks.iter().filter_map(|k| m.get(k.as_str())).count());
        });
        let hf = time(reps, || {
            let mut m: HbMap<String, u32, Fold> = HbMap::with_capacity_and_hasher(n, Fold::default());
            for (i, k) in ks.iter().enumerate() {
                m.insert(k.clone(), i as u32);
            }
            black_box(ks.iter().filter_map(|k| m.get(k.as_str())).count());
        });
        let hx = time(reps, || {
            let mut m: HbMap<String, u32, Fx> = HbMap::with_capacity_and_hasher(n, Fx::default());
            for (i, k) in ks.iter().enumerate() {
                m.insert(k.clone(), i as u32);
            }
            black_box(ks.iter().filter_map(|k| m.get(k.as_str())).count());
        });
        row("build+get", n, &[ap / n as f64, st / n as f64, hf / n as f64, hx / n as f64]);
    }
}

fn string_lookups(sizes: &[usize]) {
    head("resref keys: lookup only, on a built table (hits)");
    for &n in sizes {
        let ks = resrefs(n);
        let reps = (2_000_000 / n.max(1)).clamp(3, 2000);
        let mut ap: Map<String, u32> = Map::with_capacity(n);
        let mut st: HashMap<String, u32> = HashMap::with_capacity(n);
        let mut hf: HbMap<String, u32, Fold> = HbMap::with_capacity_and_hasher(n, Fold::default());
        let mut hx: HbMap<String, u32, Fx> = HbMap::with_capacity_and_hasher(n, Fx::default());
        for (i, k) in ks.iter().enumerate() {
            ap.insert_at(k.as_str(), i as u32);
            st.insert(k.clone(), i as u32);
            hf.insert(k.clone(), i as u32);
            hx.insert(k.clone(), i as u32);
        }
        row("get", n, &[
            time(reps, || { black_box(ks.iter().filter_map(|k| ap.get_at(k.as_str())).count()); }) / n as f64,
            time(reps, || { black_box(ks.iter().filter_map(|k| st.get(k.as_str())).count()); }) / n as f64,
            time(reps, || { black_box(ks.iter().filter_map(|k| hf.get(k.as_str())).count()); }) / n as f64,
            time(reps, || { black_box(ks.iter().filter_map(|k| hx.get(k.as_str())).count()); }) / n as f64,
        ]);
    }
}

fn main() {
    let sizes = [8usize, 32, 100, 694, 5_000, 50_000, 500_000];
    let which: Vec<String> = std::env::args().skip(1).collect();
    let all = which.is_empty();
    if all || which.iter().any(|a| a == "int") {
        ints(&sizes);
        lookups(&sizes);
    }
    if all || which.iter().any(|a| a == "str") {
        strings(&sizes);
        string_lookups(&sizes);
    }
    println!();
}
