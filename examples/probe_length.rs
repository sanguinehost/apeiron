//! Groups touched and full compares made per hit, against `hashbrown`'s slot count at the same n.
//!
//! Takes the `probe_cost` key set; prints apeiron's `Stats` beside the load factor each library
//! runs at. Instructions per probe can be equal while probes per lookup are not, and this is the
//! statistic that separates them.
#![expect(
    clippy::cast_precision_loss,
    reason = "a measurement harness: slot and entry counts become floats to report a load factor"
)]

use apeiron::map::Words;

const N: usize = 694;

fn keys() -> Vec<u64> {
    let mut s = 0x243F_6A88_85A3_08D3u64;
    (0..N)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s & 0xFFFF_FFFF
        })
        .collect()
}

fn main() {
    let ks = keys();
    let mut m: Words<u64, u64> = Words::new();
    for &k in &ks {
        m.insert(k, k);
    }
    let s = m.stats();
    let mut h: hashbrown::HashMap<u64, u64, foldhash::fast::RandomState> =
        hashbrown::HashMap::default();
    for &k in &ks {
        h.insert(k, k);
    }
    println!(
        "apeiron:   live {} slots {} load {:.3}",
        s.live,
        s.slots,
        s.live as f64 / s.slots as f64
    );
    println!(
        "           mean_probe {:.4}  mean_compares {:.4}  longest {}",
        s.mean_probe, s.mean_compares, s.longest_probe
    );
    println!(
        "hashbrown: live {} slots {} load {:.3}",
        h.len(),
        h.capacity(),
        h.len() as f64 / h.capacity() as f64
    );
}
