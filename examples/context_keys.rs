//! Does the DIAL do anything, or is the win a ring fix and an avalanche?
//!
//! `coord.rs` is already explicit that hashtrinity's 49.2% → 0% is a plain `GF(2)` / `Z/2^64`
//! mismatch and "not the algebra showing through as a bug", and that `finish` is splitmix64's
//! finalizer — standard multiplicative-hashing hygiene, no dial in it. So a head-to-head of
//! hashtrinity against apeiron measures three changes at once and credits all of them to the
//! trichotomy. This separates them, on the key set that actually matters to us: `countlm`'s
//! order-`k` byte contexts, which pack losslessly into a `u64` at `k ≤ 8` and fall back to a
//! rolling hash above that.
//!
//! Four arms, each adding exactly one thing:
//!
//! | arm | fold | what it isolates |
//! |---|---|---|
//! | `xor` | `acc·PHI ^ next` | hashtrinity's fold, the baseline |
//! | `mix` | `acc·PHI + next` | the ring fix alone (`GF(2)` → `Z/2^64`) |
//! | `mix+finish` | `bucket(mix…)` | the ring fix plus the avalanche |
//! | `pack` | `(hi << shift) \| lo` | the DIAL: the elliptic face's bijection |
//!
//! The first three are the hyperbolic face — a discard, judged on spread, and the birthday floor
//! is the best any of them can do. `pack` is the elliptic face, where the address *determines*
//! the key. If the dial is real, `pack` is not "better at the floor", it is off the axis the floor
//! measures: exactly zero, on every key set, and **unavailable** above the width where the
//! bijection stops existing rather than silently degrading.
//!
//! Run: `cargo run --release --example context_keys -- <corpus path>`

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "this is a measurement harness, not library code, and each lint below is listed because this \
              file actually triggers it: sizes and indices narrowed to build keys, bounded by the loop that \
              produced them; the reverse of that reinterpretation; counts converted to floats to report \
              rates; a harness would need 2^53 samples to lose a bit. Listed individually rather than \
              suppressed wholesale, so a pedantic lint this file does not yet trigger still surfaces"
)]

use apeiron::coord::{bucket, finish, mix, pack, PHI};
use std::collections::HashSet;

fn xor_fold(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0u64, |a, &b| a.wrapping_mul(PHI) ^ u64::from(b))
}
fn mix_fold(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0u64, |a, &b| mix(a, u64::from(b)))
}
/// The elliptic-face address: concatenation, `width` bits per symbol. `None` when the key is too
/// wide for a bijection to exist — the refusal, not a fallback.
fn pack_fold(syms: &[u8], width: u32) -> Option<u64> {
    if syms.len() as u32 * width > 64 {
        return None;
    }
    Some(syms.iter().fold(0u64, |a, &s| pack(a, u64::from(s), width)))
}

fn collisions(addrs: &[u64]) -> f64 {
    let n = addrs.len();
    100.0 * (n - addrs.iter().copied().collect::<HashSet<_>>().len()) as f64 / n as f64
}
fn bucketed(addrs: &[u64], bits: u32) -> f64 {
    let n = addrs.len();
    let d = addrs.iter().map(|&a| bucket(a, bits)).collect::<HashSet<_>>().len();
    100.0 * (n - d) as f64 / n as f64
}
fn floor(n: usize, bits: u32) -> f64 {
    let m = 2f64.powi(bits as i32);
    let distinct = m * (1.0 - (-(n as f64) / m).exp());
    100.0 * (n as f64 - distinct) / n as f64
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "~/bdh/input.txt".to_string()
    });
    let data = std::fs::read(&path).expect("corpus");
    let data = &data[..data.len().min(2_000_000)];

    // the corpus alphabet decides how wide a symbol must be, and therefore how far the elliptic
    // face reaches: 64 / ceil(log2(|alphabet|)) symbols, not 8.
    let mut seen = [false; 256];
    for &b in data {
        seen[b as usize] = true;
    }
    let alpha: Vec<u8> = (0..=255u8).filter(|&b| seen[b as usize]).collect();
    let mut code = [0u8; 256];
    for (i, &b) in alpha.iter().enumerate() {
        code[b as usize] = u8::try_from(i).unwrap_or(0);
    }
    let width = (usize::BITS - (alpha.len() - 1).leading_zeros()).max(1);
    println!(
        "corpus {} bytes, alphabet {} symbols → {width} bits/symbol; elliptic face reaches order {} \
         (byte-wide it reaches 8)",
        data.len(),
        alpha.len(),
        64 / width
    );
    println!();
    println!(
        "{:<7} {:>9}   {:>11} {:>11} {:>11} {:>9}   {:>16}",
        "order", "distinct", "xor+finish", "mix+finish", "floor", "", "pack full/bucket"
    );

    for k in [4usize, 6, 8, 9, 10, 12, 16] {
        // every distinct order-k context in the corpus
        let mut ctxs: HashSet<&[u8]> = HashSet::new();
        for w in data.windows(k) {
            ctxs.insert(w);
        }
        let ctxs: Vec<&[u8]> = ctxs.into_iter().collect();
        let n = ctxs.len();
        let bits = (usize::BITS - n.leading_zeros()).min(30); // table sized to the key count
        let xs: Vec<u64> = ctxs.iter().map(|c| xor_fold(c)).collect();
        let ms: Vec<u64> = ctxs.iter().map(|c| mix_fold(c)).collect();
        let coded: Vec<Vec<u8>> =
            ctxs.iter().map(|c| c.iter().map(|&b| code[b as usize]).collect()).collect();
        let ps: Option<Vec<u64>> =
            coded.iter().map(|c| pack_fold(c, width)).collect::<Option<Vec<_>>>();

        let pack_cell = match &ps {
            Some(p) => {
                let c = collisions(p);
                let b = bucketed(p, bits);
                format!("{c:.2}% / {b:.2}%")
            }
            None => "refused".to_string(),
        };
        println!(
            "k={k:<5} {n:>9}   {:>10.2}% {:>10.2}% {:>10.2}% {:>9}   {pack_cell:>16}",
            bucketed(&xs, bits),
            bucketed(&ms, bits),
            floor(n, bits),
            "",
        );
        // mix+finish IS bucket(); the raw-top column is the one without the avalanche
        let raw_top = {
            let d = ms.iter().map(|&a| a >> (64 - bits)).collect::<HashSet<_>>().len();
            100.0 * (n - d) as f64 / n as f64
        };
        let raw_xor = {
            let d = xs.iter().map(|&a| a >> (64 - bits)).collect::<HashSet<_>>().len();
            100.0 * (n - d) as f64 / n as f64
        };
        println!(
            "        {:>9}   no avalanche (raw top bits):  xor {raw_xor:.2}%   mix {raw_top:.2}%",
            ""
        );
        let _ = finish(0);
    }
    println!();
    println!("pack column is `full-address collisions / bucketed collisions`.");
    println!("`refused` = the key is wider than 64 bits, so no bijection exists and the elliptic");
    println!("face is unavailable — the dial's answer, where a hash would silently degrade.");
}
