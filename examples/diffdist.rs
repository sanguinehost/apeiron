//! `SMHasher`'s `DiffDist` in seconds, so a fold change can be screened before an eleven-minute run.
//!
//! Two fold candidates were rejected today *by `SMHasher`* after `examples/top_bit.rs` said both
//! were free. `top_bit` samples three bit positions over 200,000 keys and counts distinct high-32
//! differences; `DiffDist` collides **windows** of 24 to 36 bits over 2^21 differences, and the two
//! do not agree. This is the real procedure, from `smhasher/DifferentialTest.h`:
//!
//! for each key bit, draw 2^21 random 8-byte keys, hash each, flip that bit, hash again, keep
//! `h1 ^ h2`, then count collisions among those differences in each window and compare against
//! `n*(n-1)/2 / 2^bits`.
//!
//! `DIFF_BITS=56-63` (default) picks the bits to test; `DIFF_BITS=0-63` does all of them.
//!
//! Calibration, against the runs in `fold-abi/SMHASHER.md`: the shipped fold reads **81/63 at bit
//! 63 in the 35-bit window** and passes; a variant seeding by rotation read 152/63 and failed. A
//! screen that does not reproduce those two numbers is not measuring `DiffDist`.
#![expect(
    clippy::cast_precision_loss,
    reason = "a measurement harness: collision counts become floats to report a ratio against the \
              birthday expectation"
)]

use apeiron::fold::fold_bytes_with;

const N: usize = 1 << 21;

fn collisions(diffs: &[u64], bits: u32, high: bool) -> usize {
    let shift = if high { 64 - bits } else { 0 };
    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let mut v: Vec<u64> = diffs.iter().map(|d| (d >> shift) & mask).collect();
    v.sort_unstable();
    let mut c = 0;
    for w in v.windows(2) {
        if w[0] == w[1] {
            c += 1;
        }
    }
    c
}

fn main() {
    let spec = std::env::var("DIFF_BITS").unwrap_or_else(|_| "56-63".into());
    let (lo, hi) = spec.split_once('-').unwrap_or(("0", "63"));
    let (lo, hi): (u32, u32) = (lo.parse().unwrap(), hi.parse().unwrap());

    println!("  bit   window            actual / expected   ratio");
    let mut worst = (0.0f64, String::new());
    for bit in lo..=hi {
        // Deterministic key stream, so two runs of this program are comparable.
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut diffs: Vec<u64> = Vec::with_capacity(N);
        for _ in 0..N {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let k = s.to_le_bytes();
            let h1 = fold_bytes_with(0, &k);
            let mut k2 = k;
            k2[(bit / 8) as usize] ^= 1 << (bit % 8);
            let h2 = fold_bytes_with(0, &k2);
            diffs.push(h1 ^ h2);
        }
        for high in [false, true] {
            for bits in [32u32, 35] {
                let c = collisions(&diffs, bits, high);
                let expected =
                    (N as f64) * ((N - 1) as f64) / 2.0 / 2f64.powi(i32::try_from(bits).unwrap_or(64));
                let ratio = c as f64 / expected;
                let label = format!("{} {bits}-bit", if high { "high" } else { "low " });
                if ratio > worst.0 {
                    worst = (ratio, format!("bit {bit} {label}: {c}/{expected:.0}"));
                }
                if ratio > 1.5 {
                    println!("  {bit:<5} {label:<16} {c:>7} / {expected:<8.0}  {ratio:.2}x  <-");
                } else if bits == 35 && !high {
                    println!("  {bit:<5} {label:<16} {c:>7} / {expected:<8.0}  {ratio:.2}x");
                }
            }
        }
    }
    println!("\n  WORST {:.2}x   {}", worst.0, worst.1);
}
