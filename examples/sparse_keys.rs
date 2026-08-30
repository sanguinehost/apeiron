//! SMHasher's Sparse keyset, locally: long keys with almost no bits set.
//!
//! 256-byte keys with up to two bits set -- 2,098,177 of them. A fold whose lanes lose their
//! length term or their constant offset collides here, which is how the first repair of the
//! pepper defect was caught.
//!
//! Run: `cargo run --release --example sparse_keys`

use apeiron::fold::fold_bytes;
use std::collections::HashSet;

fn main() {
    for len in [32usize, 256] {
        let mut seen: HashSet<u64> = HashSet::new();
        let mut n = 0usize;
        let mut key = vec![0u8; len];
        seen.insert(fold_bytes(&key));
        n += 1;
        for i in 0..len * 8 {
            key[i / 8] ^= 1 << (i % 8);
            seen.insert(fold_bytes(&key));
            n += 1;
            for j in i + 1..len * 8 {
                key[j / 8] ^= 1 << (j % 8);
                seen.insert(fold_bytes(&key));
                n += 1;
                key[j / 8] ^= 1 << (j % 8);
            }
            key[i / 8] ^= 1 << (i % 8);
        }
        println!("  len {len:>3}: {n} keys -> {} distinct ({} collisions)", seen.len(), n - seen.len());
    }
}
