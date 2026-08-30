//! `SMHasher`'s `MomentChi2` derivative -- the crate's **last failing test** -- in seconds instead
//! of eleven minutes, and what causes it.
//!
//! The statistic is the fifth moment of `popcount(h(k_prev) ^ h(k))` over linearly increasing keys
//! with step 2. For a hash with no structure that popcount is `Binomial(64, 1/2)`: mean 32,
//! **variance 16**. `foldhash` measures 15.999, 16.005 and 15.997 at four, eight and sixteen bytes.
//! The shipped fold measures **16.559 and 16.777** at four and eight, and `SMHasher` reports the
//! resulting fifth moment as 145,171 against an ideal of 1.
//!
//! Four and eight bytes are exactly the lengths where `b[..k]` and `b[len-k..]` are the *same
//! bytes*, so the fold has one input word and both `mix2` operands are `x + seed`. Sixteen bytes
//! reads two disjoint halves and is clean. That is the whole shape of the failure, and it is not
//! visible in distinctness -- `h(x+2) ^ h(x)` takes 1,048,576 distinct values out of 1,048,576 at
//! every length. Only the moment sees it.
//!
//! # What the sweep below found
//!
//! Not the seed operation, not the read positions, and not bit placement. **The guard.**
//! `mix2` is `(hi ^ lo) ^ (a + rotl(b, 32))`, and with one input word that second term is an
//! affine function of the input riding on a well-mixed `p`. Sorting the variants by whether the
//! guard is *injective in the input*:
//!
//! | guard | injective? | var, len=4 |
//! |---|---|---|
//! | none at all | no | 15.961 |
//! | `x + y`, giving `2a + k` | no, loses the top bit | 15.899 |
//! | `x \| y` | no | 16.051 |
//! | `(x + y) ^ (x & y)` | no | 15.957 |
//! | **`x + rotl(y, 32)`, as shipped** | **yes** | **16.835** |
//! | `(x \| y) + x` | yes, `+ x` restores it | 16.915 |
//! | `x + rotl(y, 1)` / `rotl(y, 8)` / `rotl(y, 16)` | yes | 17.0 / 17.0 / 16.9 |
//!
//! Every guard that loses information about the input passes; every guard that keeps it fails.
//!
//! **And that is in direct tension with `mix2`'s own contract**, which asks the guard to be
//! "injective in each argument with the other held fixed" -- the property that makes it break ties
//! when `p` collides. Every candidate that fixes the moment gives that up, and the symmetric ones
//! give up more: a symmetric `mix2` hands an attacker a constructible collision in `fold_long`,
//! where `mix2(x + a0, NU + a1)` and the pepper-zero state let them solve `a0' = NU + a1 - x`,
//! `a1' = x + a0 - NU`. So this is a design decision about what the guard is for, not a free win,
//! and nothing here is applied to the fold.
//!
//! Run: `cargo run --release --example moment_chi2`

#![expect(
    clippy::cast_precision_loss,
    reason = "a measurement harness: histogram counts become floats to report a variance"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "the fold *is* the truncation, exactly as in `fold::mix2`: `hi ^ lo` of a 128-bit \
              product keeps 64 bits on purpose"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "`a`, `c`, `x`, `y`, `p`, `w` are `fold::mix2`'s own names; renaming them here would \
              make the variants harder to line up against the function they are variants of"
)]

use apeiron::coord::PHI;

const NU: u64 = 0xD6E8_FEB8_6659_FD93;
const PSI: u64 = 0x94D0_49BB_1331_11EB;

fn mix2(a: u64, b: u64) -> u64 {
    let w = u128::from(a).wrapping_mul(u128::from(b));
    let p = ((w >> 64) as u64) ^ (w as u64);
    p ^ a.wrapping_add(b.rotate_left(32))
}

fn settle(h: u64) -> u64 {
    let x = h.wrapping_mul(PHI);
    x ^ (x >> 32)
}

/// The `len <= 16`, `len >= 8` branch, with the second operand formed `variant` ways.
fn fold(variant: u8, len: usize, x: u64) -> u64 {
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&x.to_le_bytes());
    let b = &buf[..len];
    let seed = PHI.wrapping_add((len as u64).wrapping_mul(NU));
    let seed2 = PSI.wrapping_add((len as u64).wrapping_mul(PHI));
    let w = |s: &[u8]| u64::from_le_bytes(s.try_into().unwrap());
    let (a, c) = if len >= 8 {
        (w(&b[..8]), w(&b[len - 8..]))
    } else {
        (u64::from(u32::from_le_bytes(b[..4].try_into().unwrap())),
         u64::from(u32::from_le_bytes(b[len - 4..].try_into().unwrap())))
    };
    settle(match variant {
        0 => mix2(a.wrapping_add(seed), c.wrapping_add(seed2)),
        1 => mix2(a.wrapping_add(seed), c.rotate_left(32).wrapping_add(seed2)),
        2 => mix2(a.wrapping_add(seed), c ^ seed2),
        3 => mix2(a ^ seed, c.wrapping_add(seed2)),
        4 => mix2(a.wrapping_add(seed), c.wrapping_add(seed2).rotate_left(32)),
        // `foldhash`'s shape: the seed enters by **xor**, so `(s1^x)(s2^x)` is not a polynomial in
        // `x` and there is no quadratic left to differentiate.
        5 => mix2(a ^ seed, c ^ seed2),
        6 => mix2(a ^ seed, c.rotate_left(32) ^ seed2),
        // Isolating the terms. `mix2` is `(hi^lo) ^ (a + rotl(b,32))`; the second half is the
        // zero-annihilation guard. At four bytes `a` and `c` are the same zero-extended `u32`, so
        // `rotl(c+s2, 32)` puts the *input* straight into the high half of the guard and a near
        // constant into the low half -- an affine term riding on the output.
        7 => {
            let (x, y) = (a.wrapping_add(seed), c.wrapping_add(seed2));
            let w = u128::from(x).wrapping_mul(u128::from(y));
            ((w >> 64) as u64) ^ (w as u64) // no guard
        }
        8 => {
            let (x, y) = (a.wrapping_add(seed), c.wrapping_add(seed2));
            let w = u128::from(x).wrapping_mul(u128::from(y));
            let p = ((w >> 64) as u64) ^ (w as u64);
            p ^ x.wrapping_add(y) // guard, but symmetric and unrotated
        }
        // Asymmetric guards that keep the input in the *low* bits instead of lifting it.
        9..=13 => {
            let r = [1u32, 8, 16, 48, 63][(variant - 9) as usize];
            let (x, y) = (a.wrapping_add(seed), c.wrapping_add(seed2));
            let w = u128::from(x).wrapping_mul(u128::from(y));
            let p = ((w >> 64) as u64) ^ (w as u64);
            p ^ x.wrapping_add(y.rotate_left(r))
        }
        14 => {
            let (x, y) = (a.wrapping_add(seed), c.wrapping_add(seed2));
            let w = u128::from(x).wrapping_mul(u128::from(y));
            let p = ((w >> 64) as u64) ^ (w as u64);
            p ^ x.wrapping_add(y << 1) // asymmetric, input stays low
        }
        // `+` and `^` are both affine, so with one input word the guard is affine in it whatever
        // the bit positions. `|` is not affine, and it still guards: `0 | y == y`, `x | 0 == x`.
        _ => {
            let (x, y) = (a.wrapping_add(seed), c.wrapping_add(seed2));
            let w = u128::from(x).wrapping_mul(u128::from(y));
            let p = ((w >> 64) as u64) ^ (w as u64);
            match variant {
                15 => p ^ (x | y.rotate_left(32)),
                16 => p ^ (x | y),
                17 => p ^ (x.wrapping_add(y) ^ (x & y)),
                18 => p ^ (x ^ y ^ (x & y).rotate_left(32)),
                // Asymmetric *and* non-affine: `|` is neither linear nor symmetric once one
                // operand is added back. Guards: `x==0` gives `y`, `y==0` gives `2x`.
                19 => p ^ ((x | y).wrapping_add(x)),
                20 => p ^ ((x | y).wrapping_add(y)),
                21 => p ^ ((x | y) ^ (x & y).wrapping_add(x)),
                22 => p ^ (x.wrapping_add(y) | x),
                // Linear guards `a*x + b*y`. Per-argument injective iff `a` and `b` are both odd;
                // diagonal `(a+b)*t + k` non-injective iff `a+b` is even; asymmetric iff `a != b`.
                // Two odds sum to an even, so all three hold at once.
                23 => p ^ x.wrapping_add(y.wrapping_mul(3)),
                24 => p ^ x.wrapping_mul(3).wrapping_add(y),
                25 => p ^ x.wrapping_add(y.wrapping_mul(5)),
                26 => p ^ x.wrapping_mul(3).wrapping_add(y.wrapping_mul(5)),
                27 => p ^ x.wrapping_add(y.wrapping_mul(7)),
                // `alpha = 1, beta = -1`: both odd, sum zero (even), difference 2. One `sub`.
                _ => p ^ x.wrapping_sub(y),
            }
        }
    })
}

fn variance(len: usize, h: impl Fn(u64) -> u64) -> f64 {
    const N: u64 = 1 << 22;
    let _ = len;
    let mut hist = [0u64; 65];
    let mut prev = h(0);
    for i in 1..=N {
        let cur = h(i * 2);
        hist[(prev ^ cur).count_ones() as usize] += 1;
        prev = cur;
    }
    let total: u64 = hist.iter().sum();
    let mean: f64 =
        hist.iter().enumerate().map(|(k, &c)| k as f64 * c as f64).sum::<f64>() / total as f64;
    hist.iter().enumerate().map(|(k, &c)| (k as f64 - mean).powi(2) * c as f64).sum::<f64>()
        / total as f64
}

fn main() {
    println!("  derivative popcount variance, ideal 16.000 (foldhash measures 15.999)\n");
    for (v, name) in [
        (0u8, "0  as shipped:      (a+s1, c+s2)"),
        (1, "1  rotate c:        (a+s1, rotl(c,32)+s2)"),
        (2, "2  xor the seed:    (a+s1, c^s2)"),
        (3, "3  xor the first:   (a^s1, c+s2)"),
        (4, "4  rotate after:    (a+s1, rotl(c+s2,32))"),
        (5, "5  xor both:        (a^s1, c^s2)   <- foldhash's shape"),
        (6, "6  xor both, rot c: (a^s1, rotl(c,32)^s2)"),
        (7, "7  NO GUARD:        (hi^lo) only"),
        (8, "8  unrotated guard: p ^ (x+y)          [symmetric: forbidden]"),
        (9, "9  guard rotl 1:    p ^ (x + rotl(y,1))"),
        (10, "10 guard rotl 8:    p ^ (x + rotl(y,8))"),
        (11, "11 guard rotl 16:   p ^ (x + rotl(y,16))"),
        (12, "12 guard rotl 48:   p ^ (x + rotl(y,48))"),
        (13, "13 guard rotl 63:   p ^ (x + rotl(y,63))"),
        (14, "14 guard shl 1:     p ^ (x + (y<<1))"),
        (15, "15 OR, rotated:     p ^ (x | rotl(y,32))"),
        (16, "16 OR:              p ^ (x | y)"),
        (17, "17 add ^ and:       p ^ ((x+y) ^ (x&y))"),
        (18, "18 xor ^ rot-and:   p ^ (x ^ y ^ rotl(x&y,32))"),
        (19, "19 (x|y)+x:         p ^ ((x|y) + x)      [asym + non-affine]"),
        (20, "20 (x|y)+y:         p ^ ((x|y) + y)      [asym + non-affine]"),
        (21, "21 or ^ (and+x):    p ^ ((x|y) ^ ((x&y)+x))"),
        (22, "22 (x+y)|x:         p ^ ((x+y) | x)"),
        (23, "23 x + 3y:          both odd, sum even, asymmetric"),
        (24, "24 3x + y:          both odd, sum even, asymmetric"),
        (25, "25 x + 5y:          both odd, sum even, asymmetric"),
        (26, "26 3x + 5y:         both odd, sum even, asymmetric"),
        (27, "27 x + 7y:          both odd, sum even, asymmetric"),
        (28, "28 x - y:           both odd, sum 0, ONE instruction"),
    ] {
        let vs: Vec<String> =
            [4usize, 8].iter().map(|&l| format!("len={l}: {:7.3}", variance(l, |x| fold(v, l, x)))).collect();
        println!("  {name:<42} {}", vs.join("   "));
    }
}
