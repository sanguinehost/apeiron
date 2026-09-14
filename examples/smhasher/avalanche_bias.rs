//! `SMHasher`'s `Avalanche` in seconds: the worst per-bit bias, over every input bit.
//!
//! For each input bit, flip it across many random keys and count how often each *output* bit
//! changes. An unbiased hash flips each output bit half the time; the reported figure is the worst
//! `|2p - 1|` over every (input bit, output bit) pair. `SMHasher` fails above roughly 1%.
//!
//! **This exists because `examples/diffdist.rs` was not sufficient.** Routing the finalizer by
//! length passed `DiffDist` at 17, 24 and 32 bytes (1.25x, 1.27x, 1.22x) and then failed `Avalanche`
//! at twenty bytes with a **13.62%** bias. The two tests see different things: `DiffDist` looks at
//! the distribution of whole-word differences, Avalanche at the marginal bias of each output bit.
//! A fold change needs both screens and then `SMHasher` anyway.
//!
//! `AVA_LEN=20` picks the key length; the default sweeps the lengths `SMHasher` uses.
use apeiron::fold::fold_bytes_with;

/// `SMHasher`'s own count, so the numbers here are directly comparable to its output and the
/// noise floor is the same. At 40,000 the floor is about 2% and a 1% threshold flags nothing but
/// noise; at 300,000 it is about 0.7%, and the shipped fold reports 0.6-0.9% across every length.
const REPS: usize = 300_000;

fn worst_bias(klen: usize) -> (f64, usize, usize) {
    let kbits = klen * 8;
    let mut flips = vec![0u32; kbits * 64];
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut key = vec![0u8; klen];
    for _ in 0..REPS {
        for chunk in key.chunks_mut(8) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let n = chunk.len();
            chunk.copy_from_slice(&s.to_le_bytes()[..n]);
        }
        let h0 = fold_bytes_with(0, &key);
        for bit in 0..kbits {
            key[bit / 8] ^= 1 << (bit % 8);
            let d = h0 ^ fold_bytes_with(0, &key);
            key[bit / 8] ^= 1 << (bit % 8);
            let row = bit * 64;
            let mut m = d;
            while m != 0 {
                let j = m.trailing_zeros() as usize;
                flips[row + j] += 1;
                m &= m - 1;
            }
        }
    }
    let mut worst = (0.0f64, 0usize, 0usize);
    for bit in 0..kbits {
        for j in 0..64 {
            #[expect(clippy::cast_precision_loss, reason = "counts bounded by REPS")]
            let p = f64::from(flips[bit * 64 + j]) / REPS as f64;
            let bias = (2.0 * p - 1.0).abs();
            if bias > worst.0 {
                worst = (bias, bit, j);
            }
        }
    }
    worst
}

fn main() {
    let lens: Vec<usize> = match std::env::var("AVA_LEN") {
        Ok(v) => vec![v.parse().unwrap()],
        Err(_) => vec![3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32, 40, 64, 128],
    };
    println!("  len   worst bias   at (input bit -> output bit)");
    for klen in lens {
        let (b, i, j) = worst_bias(klen);
        let flag = if b > 0.01 { "  <- FAIL" } else { "" };
        println!("  {klen:<5} {:>8.4}%   bit {i} -> bit {j}{flag}", b * 100.0);
    }
}
