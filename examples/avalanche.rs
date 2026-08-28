//! Which reduction of the 128-bit product mixes best?
//!
//! `x * y` in 128 bits is the two-slot product with `x = 2^32`. Folding the halves back means
//! choosing what `2^64` is congruent to, which is choosing sigma:
//!
//!   lo + hi   sigma = +1   hyperbolic
//!   lo        sigma =  0   parabolic, the nilpotent discarded
//!   lo - hi   sigma = -1   elliptic
//!   lo ^ hi                not a face -- GF(2), what foldhash uses
//!
//! Measured by the strict avalanche criterion: flip one input bit, count how often each output
//! bit flips. Perfect is 50% for every one of the 64x64 pairs. Reported as the worst pair and the
//! mean absolute deviation, because an average near 50% hides a single bit that never moves.

const SAMPLES: usize = 20_000;

fn sac(mix: impl Fn(u64) -> u64) -> (f64, f64) {
    let mut flips = [[0u32; 64]; 64];
    let mut r = 0x243F_6A88_85A3_08D3u64;
    let mut next = || { r ^= r << 13; r ^= r >> 7; r ^= r << 17; r };
    for _ in 0..SAMPLES {
        let x = next();
        let h = mix(x);
        for i in 0..64 {
            let d = mix(x ^ (1u64 << i)) ^ h;
            for (o, f) in flips[i].iter_mut().enumerate() {
                *f += ((d >> o) & 1) as u32;
            }
        }
    }
    let mut worst = 0.0f64;
    let mut total = 0.0f64;
    for row in &flips {
        for &f in row {
            let p = f as f64 / SAMPLES as f64;
            let dev = (p - 0.5).abs();
            worst = worst.max(dev);
            total += dev;
        }
    }
    (worst, total / (64.0 * 64.0))
}

const C: u64 = 0x2d35_8dcc_aa6c_78a5; // foldhash's own ARBITRARY constant

fn main() {
    let prod = |x: u64| (x as u128).wrapping_mul(C as u128);
    let lo = |p: u128| p as u64;
    let hi = |p: u128| (p >> 64) as u64;

    // Cheap candidates that are still bijections, so `bucket` never merges what `packed` split.
    let arms: [(&str, Box<dyn Fn(u64) -> u64>); 10] = [
        ("parabolic  lo",     Box::new(move |x| lo(prod(x)))),
        ("hyperbolic lo+hi",  Box::new(move |x| lo(prod(x)).wrapping_add(hi(prod(x))))),
        ("elliptic   lo-hi",  Box::new(move |x| lo(prod(x)).wrapping_sub(hi(prod(x))))),
        ("foldhash   lo^hi",  Box::new(move |x| lo(prod(x)) ^ hi(prod(x)))),
        ("elliptic x2",       Box::new(move |x| {
            let a = lo(prod(x)).wrapping_sub(hi(prod(x)));
            let p = (a as u128).wrapping_mul(C as u128);
            lo(p).wrapping_sub(hi(p))
        })),
        ("splitmix64 finish", Box::new(|x| apeiron::coord::finish(x))),
        // mul by odd is a bijection; xorshift is a bijection; so are their compositions.
        ("mul                ", Box::new(|x| x.wrapping_mul(C))),
        ("mul,xs32           ", Box::new(|x| { let h = x.wrapping_mul(C); h ^ (h >> 32) })),
        ("xs32,mul,xs32      ", Box::new(|x| {
            let h = x ^ (x >> 32);
            let h = h.wrapping_mul(C);
            h ^ (h >> 32)
        })),
        ("xs29,mul,xs32      ", Box::new(|x| {
            let h = x ^ (x >> 29);
            let h = h.wrapping_mul(C);
            h ^ (h >> 32)
        })),
    ];

    println!("  {:<20} {:>12} {:>12}", "reduction", "worst dev", "mean dev");
    for (name, f) in arms {
        let (worst, mean) = sac(&f);
        println!("  {name:<20} {worst:>11.4} {mean:>12.5}");
    }
}
