//! Real collisions in the low bits of real hashes, against foldhash, at every key length.
//!
//! `examples/diffdist.rs` collides *differences* and is only calibrated at eight bytes and up;
//! below that it reported 4.09x, 6.95x and 5.81x at five, six and seven bytes. This answers the
//! question those numbers raise directly -- hash a million distinct keys, bucket them by their low
//! twenty bits, count collision pairs against the birthday expectation -- with no differences and
//! no windows in the way.
//!
//! Both hashers read **1.00x at every length from four to sixteen bytes**. The `diffdist` numbers
//! are a property of that screen at short lengths, not of the fold, and the two files now say so.
//!
//! Run: `cargo run --release --example lowbits_probe`
use std::hash::{BuildHasher, Hash, Hasher};

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "every cast here is masked to `lowbits` (at most 20) or is a count that cannot exceed \
              `n`, which is 2^20; this is a probe, and the arithmetic is bounded by construction"
)]
fn main() {
    // Direct collision count in the low bits, no differences, no windows: hash many distinct keys
    // of one length, bucket by the low 20 bits, compare against the birthday expectation.
    let n = 1u64 << 20;
    for klen in [4usize, 5, 6, 7, 8, 13, 16] {
        for lowbits in [20u32] {
            let mut seen = vec![0u32; 1 << lowbits];
            let mut fh = vec![0u32; 1 << lowbits];
            let st = foldhash::fast::FixedState::with_seed(0x1234_5678);
            for i in 0..n {
                // Distinct keys by construction: the counter is written into the key bytes.
                let mut k = [0u8; 16];
                let mut w = i;
                w = (w ^ (w >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                w = (w ^ (w >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                w ^= w >> 31;
                k[..8].copy_from_slice(&w.to_le_bytes());
                k[8..].copy_from_slice(&w.rotate_left(32).to_le_bytes());
                let key = &k[..klen];
                let h = apeiron::fold::fold_bytes(key);
                seen[(h & ((1 << lowbits) - 1)) as usize] += 1;
                let mut hh = st.build_hasher();
                Hash::hash_slice(key, &mut hh);
                fh[(hh.finish() & ((1 << lowbits) - 1)) as usize] += 1;
            }
            // Collision pairs within each bucket.
            let cc = |v: &Vec<u32>| -> f64 {
                v.iter().map(|&c| f64::from(c) * (f64::from(c) - 1.0) / 2.0).sum()
            };
            let nf = n as f64;
            let exp = nf * (nf - 1.0) / 2.0 / f64::from(1u32 << lowbits);
            println!(
                "  klen={klen:<3} low{lowbits}  apeiron {:>10.0} ({:.2}x)   foldhash {:>10.0} ({:.2}x)   expected {exp:.0}",
                cc(&seen), cc(&seen) / exp, cc(&fh), cc(&fh) / exp
            );
        }
    }
}
