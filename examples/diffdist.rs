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
//! `DIFF_LEN=8` (default, and what `SMHasher` uses) picks the key length. Length matters to this
//! measurement more than anything else: at eight bytes and below the fold reads the same bytes
//! twice, so both `mix2` operands come from one word, and at nine and above they are two
//! independent words.
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

#[expect(
    clippy::many_single_char_names,
    reason = "`s`, `k`, `w`, `b`, `n` are the key stream, the key, the finalized word, its bytes \
              and a chunk length -- the conventional names for each, in a twenty-line loop"
)]
fn main() {
    let klen: usize =
        std::env::var("DIFF_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let spec = std::env::var("DIFF_BITS").unwrap_or_else(|_| "56-63".into());
    let (lo, hi) = spec.split_once('-').unwrap_or(("0", "63"));
    let (lo, hi): (u32, u32) = (lo.parse().unwrap(), hi.parse().unwrap());

    // A key of `klen` bytes has `8 * klen` bits, and flipping one it does not have is a no-op:
    // the "flipped" key equals the original, every difference is zero, and they all collide. That
    // reports as a catastrophic ratio -- 32768x, 2097151/64 -- for a fold with nothing wrong with
    // it. The shipped fold reads exactly that at `DIFF_LEN` 4, 5 and 7, which is how this guard
    // came to be written: a screen that manufactures a failure is worse than one that is silent,
    // because the next person spends the afternoon chasing it.
    // # This screen is not trustworthy below eight bytes
    //
    // It collides *differences* `h(k) ^ h(k ^ bit)`, and for a short key the reachable difference
    // set is structured rather than uniform: at six bytes the reads are `b[0..4]` and `b[2..6]`,
    // so a flip lands in one read or in both, and there are only a few shapes of difference to
    // collide. This reports 4.09x, 6.95x and 5.81x at five, six and seven bytes for a fold that
    // `examples/lowbits_probe.rs` measures at **1.00x -- exactly foldhash's** -- by counting real
    // collisions in the low bits of real hashes, and that `SMHasher`'s own short-key keysets
    // (TwoBytes, Cyclic, Sparse) put no higher than 1.24x.
    //
    // Two harness bugs were fixed here before that conclusion was reached, and neither was the
    // cause: flipping bits a short key does not have (guarded below), and truncating a raw
    // xorshift orbit for keys under eight bytes rather than finalizing it first. Read this screen
    // at eight bytes and up, where it was calibrated against `SMHasher` to within 2%.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "`klen` comes from `DIFF_LEN` and indexes a 256-byte buffer, so it is far below \
                  `u32::MAX`; the multiply saturates besides"
    )]
    let key_bits = (klen as u32).saturating_mul(8);
    if lo >= key_bits {
        println!("  key length {klen} bytes = {key_bits} bits; no bit in {lo}-{hi} exists. Nothing to test.");
        return;
    }
    let hi = hi.min(key_bits.saturating_sub(1));

    println!("  key length {klen} bytes ({key_bits} bits); testing bits {lo}-{hi}");
    println!("  bit   window            actual / expected   ratio");
    let mut worst = (0.0f64, String::new());
    for bit in lo..=hi {
        // Deterministic key stream, so two runs of this program are comparable.
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut diffs: Vec<u64> = Vec::with_capacity(N);
        for _ in 0..N {
            // Every byte of the key is random, as `SMHasher` draws them. Correlated tail bytes
            // would be measuring the harness, not the fold.
            let mut k = [0u8; 256];
            for chunk in k[..klen].chunks_mut(8) {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                // **Finalize before truncating.** A raw xorshift state is a good 64-bit stream and
                // a bad short one: consecutive states satisfy `s' = f(s)`, so taking the low five
                // bytes of each yields keys on a structured orbit rather than independent draws.
                // That is a property of the harness, and this screen reported it as the fold's --
                // 6.66x at six bytes, where `SMHasher`'s own short-key keysets (TwoBytes, Cyclic,
                // Sparse) see nothing above 1.24x. Splitmix's finalizer decorrelates the low bytes
                // so a truncation is as good as the whole word.
                let mut w = s;
                w = (w ^ (w >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                w = (w ^ (w >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                w ^= w >> 31;
                let b = w.to_le_bytes();
                let n = chunk.len();
                chunk.copy_from_slice(&b[..n]);
            }
            let h1 = fold_bytes_with(0, &k[..klen]);
            let mut k2 = k;
            k2[(bit / 8) as usize] ^= 1 << (bit % 8);
            let h2 = fold_bytes_with(0, &k2[..klen]);
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
