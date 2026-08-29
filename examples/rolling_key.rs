//! Does `token_countlm`'s context key actually collide, on realistic token contexts?
//!
//! `arch::token_countlm` keys its count table by a polynomial rolling hash mod 2^64,
//! `h(ctx) = Σ ctx[j]·B^(o-1-j)` with `B = PHI`, and stores it in hashtrinity's **Regen** face —
//! the u64 IS the key, nothing is kept beside it to compare. So two contexts that collide do not
//! probe past each other, they MERGE their counts, silently and unrecoverably. The byte LM has a
//! lossless pack below order 9; this one has none at any order.
//!
//! That makes it the one place where a keying defect could plausibly explain a retrieval result,
//! so it should be measured rather than argued from the birthday bound — `coord.rs` already has a
//! worked case where a multiply-fold looked fine and was not injective (the Fibonacci pair
//! `(2971215073, 50920843)` folding to `(0,0)`).
//!
//! Two distributions, because structure is what breaks multiplicative folds:
//! uniform token ids, and Zipfian ones, which is what real text gives.

use std::collections::HashSet;

const B: u64 = 0x9E37_79B9_7F4A_7C15;

/// `token_countlm`'s key, transcribed: `Σ ctx[j]·B^(o-1-j)`.
fn roll(ctx: &[u32]) -> u64 {
    ctx.iter().fold(0u64, |h, &t| h.wrapping_mul(B).wrapping_add(u64::from(t)))
}

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed.max(1);
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn floor(n: usize, bits: u32) -> f64 {
    let m = 2f64.powi(bits as i32);
    let d = m * (1.0 - (-(n as f64) / m).exp());
    100.0 * (n as f64 - d) / n as f64
}

fn main() {
    let vocab: u32 = 8192; // the harness's BPE size
    let n: usize = 8_000_000; // distinct contexts, ~ a 40 MB corpus at order 6
    println!("vocab {vocab}, {n} distinct contexts, key = polynomial rolling hash mod 2^64 (B = PHI)");
    println!(
        "{:<10} {:>12} {:>12} {:>16} {:>16}",
        "order", "distribution", "distinct ctx", "collisions", "birthday floor"
    );

    for order in [4usize, 6, 8, 12] {
        for zipf in [false, true] {
            let mut r = rng(0xC0FFEE + order as u64 + u64::from(zipf));
            let mut set: HashSet<u64> = HashSet::with_capacity(n);
            // dedupe CONTEXTS first: a Zipfian draw repeats the same context often, and counting
            // those as collisions would report the draw's duplicates as a defect of the hash.
            let mut ctxs: HashSet<Vec<u32>> = HashSet::with_capacity(n);
            let mut ctx = vec![0u32; order];
            for _ in 0..n {
                for c in ctx.iter_mut() {
                    *c = if zipf {
                        // heavy head: most mass on a few hundred ids, the shape real text has
                        let u = (r() >> 11) as f64 / (1u64 << 53) as f64;
                        let x = (u.powf(4.0) * f64::from(vocab)) as u32;
                        x.min(vocab - 1)
                    } else {
                        (r() % u64::from(vocab)) as u32
                    };
                }
                if ctxs.insert(ctx.clone()) {
                    set.insert(roll(&ctx));
                }
            }
            let distinct_ctx = ctxs.len();
            let dup = distinct_ctx - set.len();
            println!(
                "k={order:<8} {:>12} {:>12} {:>15.6}% {:>15.6}%",
                if zipf { "zipf" } else { "uniform" },
                distinct_ctx,
                100.0 * dup as f64 / distinct_ctx as f64,
                floor(distinct_ctx, 64),
            );
        }
    }

    // the adversarial question: is the fold injective on SHORT contexts, where it must be?
    println!();
    for order in [2usize, 3, 4] {
        let total = u64::from(vocab).pow(order as u32).min(40_000_000);
        let mut set: HashSet<u64> = HashSet::with_capacity(total as usize);
        let mut ctx = vec![0u32; order];
        let mut count = 0usize;
        'outer: loop {
            set.insert(roll(&ctx));
            count += 1;
            if count as u64 >= total {
                break;
            }
            for i in (0..order).rev() {
                ctx[i] += 1;
                if ctx[i] < vocab {
                    continue 'outer;
                }
                ctx[i] = 0;
            }
            break;
        }
        println!(
            "exhaustive order {order}: {count} contexts → {} distinct keys ({} collisions)",
            set.len(),
            count - set.len()
        );
    }
}
