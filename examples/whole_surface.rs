//! Every operation against `hashbrown` + `foldhash` and `std`, with the counter beside the clock.
//!
//! A table is not one function. Claiming it replaces `std::collections::HashMap` means claiming it
//! on insert, lookup, miss, entry, removal, iteration, clone, drain, retain and extend -- on word
//! keys and on string keys, at every size from sixteen entries to a million, and on the key
//! distributions that break addresses rather than only the one that flatters them. This runs all
//! of that and prints `LOSS` on any row more than 2% behind `hashbrown`.
//!
//! Run: `taskset -c 2 cargo run --release --example whole_surface [u64|str] [serial|highbits|random]`
//!
//! # Three things the previous version of this file could not tell you
//!
//! It reported ns/op alone, from `Instant` around a loop, and it was wrong in three ways that a
//! reader had no way to see:
//!
//! **It never checked the answers.** `best(r, || ks.iter().filter_map(|k| m.get(k)).sum())` scores
//! a map that returns `None` for every key as the fastest arm on the board. Every timed closure
//! here returns a checksum and `row` panics unless all three implementations agree, so an arm that
//! answers wrongly or gets deleted fails the run instead of winning it.
//!
//! **It had no counter.** `benches/against_the_field.rs` has a standing note that its wall clock
//! disagreed with `perf` by 3x and that the disagreement is unexplained; the honest response is to
//! read both from the same loop. `examples/support/counter.rs` opens `cpu_core` instructions and
//! cycles in-process, so every row below carries its own instruction count. Nanoseconds that move
//! without instructions or cycles moving are a harness artifact, and that is now visible per row
//! rather than requiring a second harness and a fork.
//!
//! **It sampled one key distribution.** Keys came from a single xorshift. `highbits` -- `0..n`
//! under `swap_bytes` -- is the distribution that has broken this crate's address twice, and it
//! was not here. It is a command-line argument now, and `serial` with it.
//!
//! # What is timed, exactly
//!
//! Timing and counting are separate passes over the same closure, because the `ioctl` that arms a
//! counter costs more than a thousand-key lookup loop. The clock pass takes the **minimum** of
//! many rounds: a busy machine can only make a round slower. The counter pass takes the minimum of
//! three, since retired instructions are near-deterministic and the spread is the page faults.
//!
//! `clone` and `insert fresh` include the drop of what they built, and are labelled for it. A
//! pre-built victim per round is how `remove`, `drain` and `retain` avoid the same problem, and at
//! a million keys the victims are the memory ceiling on the round count.

#![expect(
    clippy::cast_precision_loss,
    reason = "a measurement harness: counts become floats to report per-operation rates"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "`a`/`s`/`f` are the three arms, `n` the size, `c` the counter, `d` the divisor: \
              every one of them is a column heading in the table this file prints"
)]
#![expect(
    clippy::unnecessary_get_then_check,
    reason = "the `get miss` rows measure `get` on an absent key on purpose; `contains_key` is a \
              different function in all three implementations and would measure a different path"
)]

#[path = "support/counter.rs"]
mod counter;

use apeiron::map::{Map, Words};
use counter::Counter;
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

type Fold = foldhash::fast::RandomState;
type HbMap<K, V> = hashbrown::HashMap<K, V, Fold>;

/// One arm of one row: what it cost and what it answered.
#[derive(Clone, Copy)]
struct Arm {
    ns: f64,
    instructions: f64,
    cycles: f64,
    checksum: u64,
}

impl Arm {
    fn new() -> Self {
        Self { ns: f64::INFINITY, instructions: f64::INFINITY, cycles: f64::INFINITY, checksum: 0 }
    }
}

/// `#[inline(never)]` is the whole measurement, not a detail.
///
/// `run` is generic, so each closure gets its own instantiation -- but inlined, all of them land
/// in one frame, and the lookup loops then inherit the callee-saved registers and the stack frame
/// of an `insert` that inlines a grow path and an allocation. The register allocator pays for that
/// by rematerialising the fold's seed on every key. Isolated, `examples/probe_cost.rs` measures the
/// same 13-byte resref lookup at 92.5 instructions where the shared frame reads 107.7, and
/// `hashbrown`'s arm moves a third as far -- so the shared frame is not even a fair harness, let
/// alone an accurate one.
#[inline(never)]
fn run(reps: usize, f: &mut impl FnMut() -> u64) -> u64 {
    let mut acc = 0u64;
    for _ in 0..reps {
        acc = acc.wrapping_add(black_box(f()));
    }
    acc
}

/// Times one arm for one round and folds the result into its running minimum.
fn tick(arm: &mut Arm, reps: usize, f: &mut impl FnMut() -> u64) {
    let t = Instant::now();
    arm.checksum = black_box(run(reps, f));
    arm.ns = arm.ns.min(t.elapsed().as_secs_f64() * 1e9 / reps as f64);
}

/// Counts one arm for one round and folds the result into its running minimum.
fn count(c: Option<&Counter>, arm: &mut Arm, reps: usize, f: &mut impl FnMut() -> u64) {
    if let Some(c) = c {
        let (_, k) = c.count(|| black_box(run(reps, f)));
        arm.instructions = arm.instructions.min(k.instructions as f64 / reps as f64);
        arm.cycles = arm.cycles.min(k.cycles as f64 / reps as f64);
    }
}

/// Times and counts three repeatable arms with the arms **alternated inside each round**, then
/// prints the row.
///
/// Measuring one arm to completion and then the next is the mistake this crate has already paid
/// for once: the machine's frequency, its cache state and its neighbours all drift across the
/// seconds a full arm takes, and the drift lands entirely on whichever arm ran during it. The
/// first version of this rewrite did exactly that, and at a million keys it reported apeiron
/// slower in nanoseconds and *faster* in cycles on the same row -- a contradiction that only the
/// counter made visible. Alternating inside the round bounds the drift to one round.
///
/// # Panics
///
/// If any two arms disagree on their checksum. A disagreement means the row is measuring different
/// work in different arms, which is the one failure a timing harness cannot report as a number.
#[expect(clippy::too_many_arguments, reason = "three arms plus the row's own labels and budget; a struct per arm would move the same eight values behind a name that adds nothing")]
fn row(
    op: &str,
    n: usize,
    per: f64,
    c: Option<&Counter>,
    (rounds, reps): (usize, usize),
    mut fa: impl FnMut() -> u64,
    mut fs: impl FnMut() -> u64,
    mut ff: impl FnMut() -> u64,
) {
    let (mut a, mut s, mut f) = (Arm::new(), Arm::new(), Arm::new());
    // The order rotates every round. Alternating the arms bounds the machine's drift to one round,
    // but running the same arm first in every round hands that arm every cold start -- measured on
    // `examples/op_asm.rs`, a `u64` hit read 0.83x with apeiron always first and 0.96x with it
    // always second, and neither of those is the number.
    for r in 0..rounds * 3 {
        match r % 3 {
            0 => tick(&mut a, reps, &mut fa),
            1 => tick(&mut s, reps, &mut fs),
            _ => tick(&mut f, reps, &mut ff),
        }
    }
    for r in 0..9 {
        match r % 3 {
            0 => count(c, &mut a, reps, &mut fa),
            1 => count(c, &mut s, reps, &mut fs),
            _ => count(c, &mut f, reps, &mut ff),
        }
    }
    emit(op, n, per, a, s, f);
}

/// The same, for arms that consume a victim built before the clock starts.
///
/// `clone().drain()` measures a clone and a drain together, and at these sizes the clone is most
/// of it. One fresh victim per round is the only way to see the drain by itself.
///
/// # Panics
///
/// As `row`.
#[expect(clippy::too_many_arguments, reason = "three arms, each a maker and a consumer, plus the row's own labels; splitting them into a struct per arm moves the same nine values behind a name that adds nothing")]
fn row_consuming<A, S, F>(
    op: &str,
    n: usize,
    per: f64,
    c: Option<&Counter>,
    rounds: usize,
    ma: impl Fn() -> A,
    mut fa: impl FnMut(A) -> u64,
    ms: impl Fn() -> S,
    mut fs: impl FnMut(S) -> u64,
    mf: impl Fn() -> F,
    mut ff: impl FnMut(F) -> u64,
) {
    let (mut a, mut s, mut f) = (Arm::new(), Arm::new(), Arm::new());
    // The order rotates, as in `row` and for the same reason.
    for r in 0..rounds * 3 {
        match r % 3 {
            0 => {
                let v = ma();
                let t = Instant::now();
                a.checksum = black_box(fa(v));
                a.ns = a.ns.min(t.elapsed().as_secs_f64() * 1e9);
            }
            1 => {
                let v = ms();
                let t = Instant::now();
                s.checksum = black_box(fs(v));
                s.ns = s.ns.min(t.elapsed().as_secs_f64() * 1e9);
            }
            _ => {
                let v = mf();
                let t = Instant::now();
                f.checksum = black_box(ff(v));
                f.ns = f.ns.min(t.elapsed().as_secs_f64() * 1e9);
            }
        }
    }
    if let Some(c) = c {
        for r in 0..9 {
            match r % 3 {
                0 => {
                    let v = ma();
                    let (_, k) = c.count(|| black_box(fa(v)));
                    a.instructions = a.instructions.min(k.instructions as f64);
                    a.cycles = a.cycles.min(k.cycles as f64);
                }
                1 => {
                    let v = ms();
                    let (_, k) = c.count(|| black_box(fs(v)));
                    s.instructions = s.instructions.min(k.instructions as f64);
                    s.cycles = s.cycles.min(k.cycles as f64);
                }
                _ => {
                    let v = mf();
                    let (_, k) = c.count(|| black_box(ff(v)));
                    f.instructions = f.instructions.min(k.instructions as f64);
                    f.cycles = f.cycles.min(k.cycles as f64);
                }
            }
        }
    }
    emit(op, n, per, a, s, f);
}

/// Checks the three arms answered the same thing, then prints the row.
///
/// # Panics
///
/// If any two arms disagree.
fn emit(op: &str, n: usize, per: f64, a: Arm, s: Arm, f: Arm) {
    assert!(
        a.checksum == s.checksum && a.checksum == f.checksum,
        "{op} n={n}: arms disagree -- apeiron {}, std {}, hb {}. \
         The row is not measuring the same work in each arm.",
        a.checksum,
        s.checksum,
        f.checksum
    );
    let worst = s.ns.min(f.ns);
    let mut mark = if a.ns > worst * 1.02 { "  LOSS".to_string() } else { String::new() };
    // The clock and the counter must agree about who won. They are measuring the same loop, so a
    // row where apeiron is ahead in nanoseconds and behind in cycles -- or the reverse -- is not a
    // finding about the map, it is a finding about this harness. Measuring one arm to completion
    // before the next produced exactly that at a million keys, and nothing in the output said so.
    let (dn, dc) = (a.ns - f.ns, a.cycles - f.cycles);
    if dn.abs() > f.ns * 0.03 && dc.abs() > f.cycles * 0.03 && (dn < 0.0) != (dc < 0.0) {
        mark.push_str("  ?ns-vs-cycles");
    }
    println!(
        "  {op:<16} {n:>8} {:>8.2} {:>8.2} {:>8.2} {:>6.2}x | {:>7.1} {:>7.1} | {:>7.2} {:>7.2}{mark}",
        a.ns / per,
        s.ns / per,
        f.ns / per,
        worst / a.ns,
        a.instructions / per,
        f.instructions / per,
        a.cycles / per,
        f.cycles / per,
    );
}

fn head(what: &str) {
    println!("\n{what}");
    println!("  ns/op, then instructions/op and cycles/op for apeiron and hb+fold.");
    println!("  ratio is apeiron against the better rival's ns; >1 is a win.");
    println!(
        "  {:<16} {:>8} {:>8} {:>8} {:>8} {:>7} | {:>7} {:>7} | {:>7} {:>7}",
        "op", "n", "apeiron", "std", "hb+fold", "ratio", "ap ins", "hb ins", "ap cyc", "hb cyc"
    );
}

/// The three key distributions `hashbrown`'s own benchmarks use.
///
/// `highbits` is `0..n` under `swap_bytes`, so every key differs only in bits a table reads last.
/// It is the distribution that has broken this crate's address twice and the one a single xorshift
/// corpus never produces.
fn words(dist: &str, n: usize) -> Vec<u64> {
    match dist {
        "serial" => (0..n as u64).collect(),
        "highbits" => (0..n as u64).map(u64::swap_bytes).collect(),
        _ => {
            let mut s = 0x243F_6A88_85A3_08D3u64;
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    s
                })
                .collect()
        }
    }
}

/// String keys shaped like the real corpus: a resref is a name in a fixed-width field.
fn strings(dist: &str, n: usize) -> Vec<String> {
    words(dist, n).iter().map(|w| format!("nw_wswmg{w:016x}")).collect()
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "u64".into());
    let dist = std::env::args().nth(2).unwrap_or_else(|| "random".into());
    let c = Counter::open();
    if c.is_none() {
        println!("note: hardware counters unavailable; instruction columns will read inf");
    }
    let sizes = [16usize, 256, 1_000, 16_384, 100_000, 1_000_000];
    if which == "u64" {
        u64_surface(c.as_ref(), &dist, &sizes);
    } else {
        str_surface(c.as_ref(), &dist, &sizes);
    }
}

/// Rounds and reps for a size: enough repetitions that a timed region outlasts the clock's own
/// cost, and enough rounds that the minimum has seen an uncontended one.
fn budget(n: usize) -> (usize, usize) {
    let reps = (200_000 / n).max(1);
    let rounds = if n > 200_000 { 6 } else { 20 };
    (rounds, reps)
}

fn u64_surface(c: Option<&Counter>, dist: &str, sizes: &[usize]) {
    head(&format!("integer keys, {dist} distribution"));
    for &n in sizes {
        let ks = words(dist, n);
        // Misses, and checked to be misses: a "miss" that is really a hit measures the hit path.
        let miss: Vec<u64> = ks.iter().map(|k| !k ^ 0x5555_5555_5555_5555).collect();
        let seen: std::collections::HashSet<u64> = ks.iter().copied().collect();
        assert!(miss.iter().all(|m| !seen.contains(m)), "the miss keys are not all misses");
        let d = n as f64;
        let b = budget(n);

        let built_a: Words<u64, u64> = ks.iter().map(|&k| (k, k)).collect();
        let built_s: HashMap<u64, u64> = ks.iter().map(|&k| (k, k)).collect();
        let built_f: HbMap<u64, u64> = ks.iter().map(|&k| (k, k)).collect();

        row("insert+drop", n, d, c, b,
            || { let mut m: Words<u64,u64> = Words::new(); for &k in &ks { m.insert(k,k); } m.len() as u64 },
            || { let mut m: HashMap<u64,u64> = HashMap::new(); for &k in &ks { m.insert(k,k); } m.len() as u64 },
            || { let mut m: HbMap<u64,u64> = HbMap::default(); for &k in &ks { m.insert(k,k); } m.len() as u64 });

        row("insert reserved", n, d, c, b,
            || { let mut m: Words<u64,u64> = Words::with_capacity(n); for &k in &ks { m.insert(k,k); } m.len() as u64 },
            || { let mut m: HashMap<u64,u64> = HashMap::with_capacity(n); for &k in &ks { m.insert(k,k); } m.len() as u64 },
            || { let mut m: HbMap<u64,u64> = HbMap::with_capacity_and_hasher(n, Fold::default()); for &k in &ks { m.insert(k,k); } m.len() as u64 });

        row("extend", n, d, c, b,
            || { let mut m: Words<u64,u64> = Words::new(); m.extend(ks.iter().map(|&k| (k,k))); m.len() as u64 },
            || { let mut m: HashMap<u64,u64> = HashMap::new(); m.extend(ks.iter().map(|&k| (k,k))); m.len() as u64 },
            || { let mut m: HbMap<u64,u64> = HbMap::default(); m.extend(ks.iter().map(|&k| (k,k))); m.len() as u64 });

        row("get hit", n, d, c, b,
            || ks.iter().filter_map(|k| built_a.get(k)).sum(),
            || ks.iter().filter_map(|k| built_s.get(k)).sum(),
            || ks.iter().filter_map(|k| built_f.get(k)).sum());

        // `get(..).is_some()` and not `contains_key`: this row exists to measure the `get` path on
        // a key that is not there, and `contains_key` is a different function in all three
        // implementations -- apeiron's skips the value load, which is exactly what must not be
        // skipped here. `contains` has its own row on the string surface.
        row("get miss", n, d, c, b,
            || miss.iter().map(|k| u64::from(built_a.get(k).is_some())).sum(),
            || miss.iter().map(|k| u64::from(built_s.get(k).is_some())).sum(),
            || miss.iter().map(|k| u64::from(built_f.get(k).is_some())).sum());

        row("entry", n, d, c, b,
            || { let mut m: Words<u64,u64> = Words::with_capacity(n); for &k in &ks { *m.entry(k).or_insert(0) += 1; } m.values().fold(0u64,|a,v| a.wrapping_add(*v)) },
            || { let mut m: HashMap<u64,u64> = HashMap::with_capacity(n); for &k in &ks { *m.entry(k).or_insert(0) += 1; } m.values().fold(0u64,|a,v| a.wrapping_add(*v)) },
            || { let mut m: HbMap<u64,u64> = HbMap::with_capacity_and_hasher(n, Fold::default()); for &k in &ks { *m.entry(k).or_insert(0) += 1; } m.values().fold(0u64,|a,v| a.wrapping_add(*v)) });

        row_consuming("remove all", n, d, c, b.0,
            || built_a.clone(), |mut m| { let mut s=0u64; for &k in &ks { s = s.wrapping_add(m.remove(&k).unwrap_or(0)); } s },
            || built_s.clone(), |mut m| { let mut s=0u64; for &k in &ks { s = s.wrapping_add(m.remove(&k).unwrap_or(0)); } s },
            || built_f.clone(), |mut m| { let mut s=0u64; for &k in &ks { s = s.wrapping_add(m.remove(&k).unwrap_or(0)); } s });

        row("iterate", n, d, c, b,
            || built_a.values().fold(0u64, |a,v| a.wrapping_add(*v)),
            || built_s.values().fold(0u64, |a,v| a.wrapping_add(*v)),
            || built_f.values().fold(0u64, |a,v| a.wrapping_add(*v)));

        row("clone+drop", n, d, c, b,
            || built_a.clone().len() as u64,
            || built_s.clone().len() as u64,
            || built_f.clone().len() as u64);

        // The keys are summed, not counted: draining a type with no `Drop` and discarding the
        // pairs is dead code, and LLVM removes it -- which reads as an impossibly fast `drain`.
        row_consuming("drain", n, d, c, b.0,
            || built_a.clone(), |mut m| m.drain().fold(0u64, |a,(k,_)| a.wrapping_add(k)),
            || built_s.clone(), |mut m| m.drain().fold(0u64, |a,(k,_)| a.wrapping_add(k)),
            || built_f.clone(), |mut m| m.drain().fold(0u64, |a,(k,_)| a.wrapping_add(k)));

        row_consuming("retain half", n, d, c, b.0,
            || built_a.clone(), |mut m| { m.retain(|k,_| k % 2 == 0); m.values().fold(0u64,|a,v| a.wrapping_add(*v)) },
            || built_s.clone(), |mut m| { m.retain(|k,_| k % 2 == 0); m.values().fold(0u64,|a,v| a.wrapping_add(*v)) },
            || built_f.clone(), |mut m| { m.retain(|k,_| k % 2 == 0); m.values().fold(0u64,|a,v| a.wrapping_add(*v)) });
    }
}

fn str_surface(c: Option<&Counter>, dist: &str, sizes: &[usize]) {
    head(&format!("string keys, {dist} distribution"));
    for &n in sizes {
        let ks = strings(dist, n);
        let miss: Vec<String> = ks.iter().map(|k| format!("zz_{k}")).collect();
        let d = n as f64;
        let b = budget(n);
        let built_a: Map<String, u64> = ks.iter().map(|k| (k.clone(), 1)).collect();
        let built_s: HashMap<String, u64> = ks.iter().map(|k| (k.clone(), 1)).collect();
        let built_f: HbMap<String, u64> = ks.iter().map(|k| (k.clone(), 1)).collect();

        row("insert+drop", n, d, c, b,
            || { let mut m: Map<String,u64> = Map::new(); for k in &ks { m.insert(k.clone(),1); } m.len() as u64 },
            || { let mut m: HashMap<String,u64> = HashMap::new(); for k in &ks { m.insert(k.clone(),1); } m.len() as u64 },
            || { let mut m: HbMap<String,u64> = HbMap::default(); for k in &ks { m.insert(k.clone(),1); } m.len() as u64 });

        row("get hit", n, d, c, b,
            || ks.iter().filter_map(|k| built_a.get_at(k.as_str())).sum(),
            || ks.iter().filter_map(|k| built_s.get(k.as_str())).sum(),
            || ks.iter().filter_map(|k| built_f.get(k.as_str())).sum());

        // As the integer surface: the `get` path, deliberately not `contains_key`.
        row("get miss", n, d, c, b,
            || miss.iter().map(|k| u64::from(built_a.get_at(k.as_str()).is_some())).sum(),
            || miss.iter().map(|k| u64::from(built_s.get(k.as_str()).is_some())).sum(),
            || miss.iter().map(|k| u64::from(built_f.get(k.as_str()).is_some())).sum());

        row("contains", n, d, c, b,
            || ks.iter().map(|k| u64::from(built_a.contains_at(k.as_str()))).sum(),
            || ks.iter().map(|k| u64::from(built_s.contains_key(k.as_str()))).sum(),
            || ks.iter().map(|k| u64::from(built_f.contains_key(k.as_str()))).sum());

        row("iterate", n, d, c, b,
            || built_a.values().fold(0u64, |a,v| a.wrapping_add(*v)),
            || built_s.values().fold(0u64, |a,v| a.wrapping_add(*v)),
            || built_f.values().fold(0u64, |a,v| a.wrapping_add(*v)));

        row("clone+drop", n, d, c, b,
            || built_a.clone().len() as u64,
            || built_s.clone().len() as u64,
            || built_f.clone().len() as u64);

        row_consuming("remove all", n, d, c, b.0,
            || built_a.clone(), |mut m| { let mut s=0u64; for k in &ks { s = s.wrapping_add(m.remove_at(k.as_str()).unwrap_or(0)); } s },
            || built_s.clone(), |mut m| { let mut s=0u64; for k in &ks { s = s.wrapping_add(m.remove(k.as_str()).unwrap_or(0)); } s },
            || built_f.clone(), |mut m| { let mut s=0u64; for k in &ks { s = s.wrapping_add(m.remove(k.as_str()).unwrap_or(0)); } s });
    }
}
