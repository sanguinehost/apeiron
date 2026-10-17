//! `tests/learn_fold_attn_speed.rs` — the bit-identity witness and the unit-cost bench for
//! `stats::parts::attn_a` (`src/learn/fold/stats/parts.rs`), at the production geometry of the
//! `Qwen2.5-0.5B` teacher (`n_head 14`, `n_kv_head 2`, `head_dim 64`, `hidden 896`) and the
//! production sequence length `t = 512`.
//!
//! Positive: the FNV-1a digests of the attention output on three seeded fixtures are pinned; one of
//! them (`t = 130`) has a length that is not a power of two and its last row is the causal boundary
//! row (it attends to all 130 keys). A perturbed final query row leaves every earlier row
//! bit-for-bit unchanged. `mul_rounding_identity` checks the floor-form round-half-to-even that
//! `parts::mul` records against the canonical `q32::round_q32_even` on random products and on the
//! sign and tie edges.
//! Measurement: `unit_cost` times one `attn_a` call at `t = 512`, three runs, and reports the mean
//! and spread in ns per multiply-add; the same test times `flat`, a serial `i128` dot over the same
//! `t(t+1)/2` triangular population, as the instrument's control for a loop that only multiplies
//! and adds.
//! Grade `tested` for the digests, `measured` for the timings.
//!
//! Mutation control: skipping the last key of the final query row in `AttnRows::rows` reddens
//! `attn_outputs_are_pinned` (the pinned digest of the `t = 130` fixture moves).
//!
//! The multiply-add count of one call is `n_head·head_dim·t·(t+1)`: the score contraction over the
//! causal keys (`t(t+1)/2` terms, `head_dim` each, `n_head` heads) plus the value mix over the same
//! keys. The box is shared, so the bench pairs its runs and reports the spread; it asserts nothing
//! about the rate.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    reason = "the fixture names the head geometry `nh`/`nkv`/`hd` as the module does and the bench \
              casts a word count and a duration into `f64` to report a per-multiply-add rate, and a               row index below `t·head_dim` into the `i64` of the control's panel"
)]

use std::time::Instant;

use apeiron::learn::fold::stats::parts::{attn_a, LayerCfg, RopeTables};
use apeiron::learn::solve::q32::{to_grid, GRID_BOUND};

/// Production query head count (`Qwen2.5-0.5B`).
const NH: usize = 14;
/// Production key/value head count.
const NKV: usize = 2;
/// Production head width.
const HD: usize = 64;
/// Production residual width, `n_head·head_dim`.
const HID: usize = NH * HD;

fn cfg() -> LayerCfg {
    LayerCfg {
        n_head: NH,
        n_kv_head: NKV,
        head_dim: HD,
        hidden: HID,
        inter: 4864,
        eps: 1e-6,
        theta: 1e4,
    }
}

/// Deterministic 64-bit LCG (Numerical Recipes constants).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// `n` `Q32.32` words of uniform `[-1, 1)` values.
    fn words(&mut self, n: usize) -> Vec<i64> {
        (0..n)
            .map(|_| {
                let u = (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
                to_grid(2.0 * u - 1.0, GRID_BOUND).expect("on grid")
            })
            .collect()
    }
}

/// FNV-1a over the little-endian bytes of the words.
fn digest(words: &[i64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for &w in words {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The `q`/`k`/`v` words and the output of one `attn_a` call at `t`.
fn run(seed: u64, t: usize) -> Vec<i64> {
    let c = cfg();
    let rope = RopeTables::new(&c, t).expect("rope");
    let mut rng = Lcg::new(seed);
    let q = rng.words(t * NH * HD);
    let k = rng.words(t * NKV * HD);
    let v = rng.words(t * NKV * HD);
    attn_a(&c, &rope, &q, &k, &v, t).expect("attention")
}

#[test]
fn attn_outputs_are_pinned() {
    // `t = 130` is not a multiple of a power-of-two tile and its row 129 is the causal boundary
    // row (it reads all 130 keys); `t = 512` is the production length.
    let d130a = digest(&run(0x5eed_0001, 130));
    let d130b = digest(&run(0x5eed_0002, 130));
    let d512 = digest(&run(0x5eed_0003, 512));
    println!("attn digests t=130 s1 {d130a:#018x} s2 {d130b:#018x} t=512 s3 {d512:#018x}");
    assert_eq!(d130a, 0x152e_4db8_a319_4787);
    assert_eq!(d130b, 0x64fc_8efe_6cf4_0029);
    assert_eq!(d512, 0x6325_7a9c_f867_00d8);
}

#[test]
fn last_row_is_the_causal_boundary() {
    let c = cfg();
    let t = 130;
    let rope = RopeTables::new(&c, t).expect("rope");
    let mut rng = Lcg::new(0x5eed_0004);
    let q = rng.words(t * NH * HD);
    let k = rng.words(t * NKV * HD);
    let v = rng.words(t * NKV * HD);
    let a = attn_a(&c, &rope, &q, &k, &v, t).expect("attention");
    let mut q2 = q.clone();
    for d in 0..NH * HD {
        q2[(t - 1) * NH * HD + d] = q2[(t - 1) * NH * HD + d].saturating_add(1 << 20);
    }
    let b = attn_a(&c, &rope, &q2, &k, &v, t).expect("attention");
    let w = NH * HD;
    assert_eq!(&a[..(t - 1) * w], &b[..(t - 1) * w], "prefix moved");
    assert_ne!(
        &a[(t - 1) * w..],
        &b[(t - 1) * w..],
        "final row did not move"
    );
}

/// Multiply-adds in one `attn_a` call: scores plus value mix over the causal keys.
fn mult_adds(t: usize) -> f64 {
    (NH * HD * t * (t + 1)) as f64
}

/// The attention score pass as plain `i128` dots: `t(t+1)/2` contiguous `head_dim` dots over
/// row-major panels, no key stride, no rounding, no `exp`. Its own multiply-add count is
/// `head_dim·t(t+1)/2`; the report divides by that, so the attention loop and a loop that only
/// multiplies and adds are on one scale.
fn flat(t: usize) -> i128 {
    let hd = HD;
    let a: Vec<i64> = (0..t * hd)
        .map(|i| i64::try_from(i % 1021).unwrap() - 510)
        .collect();
    let b: Vec<i64> = (0..t * hd)
        .map(|i| i64::try_from(i % 997).unwrap() - 498)
        .collect();
    let mut acc = 0_i128;
    for i in 0..t {
        let ar = &a[i * hd..][..hd];
        for j in 0..=i {
            let br = &b[j * hd..][..hd];
            for d in 0..hd {
                acc += i128::from(ar[d]) * i128::from(br[d]);
            }
        }
    }
    acc
}

#[test]
fn unit_cost() {
    let c = cfg();
    let t = 512;
    let rope = RopeTables::new(&c, t).expect("rope");
    let mut rng = Lcg::new(0x5eed_0005);
    let q = rng.words(t * NH * HD);
    let k = rng.words(t * NKV * HD);
    let v = rng.words(t * NKV * HD);
    let n = mult_adds(t);
    let mut ns = Vec::new();
    for _ in 0..3 {
        let t0 = Instant::now();
        let out = attn_a(&c, &rope, &q, &k, &v, t).expect("attention");
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(&out);
        ns.push(dt * 1e9 / n);
    }
    let mean = ns.iter().sum::<f64>() / ns.len() as f64;
    let lo = ns.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = ns.iter().copied().fold(0.0_f64, f64::max);
    println!(
        "attn unit cost t=512: {mean:.4} ns/mult-add (runs {:.4} {:.4} {:.4}, spread {:.4})",
        ns[0],
        ns[1],
        ns[2],
        hi - lo
    );

    let mut fns = Vec::new();
    let fn_count = (HD * t * (t + 1) / 2) as f64;
    for _ in 0..3 {
        let t0 = Instant::now();
        let acc = flat(t);
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(acc);
        fns.push(dt * 1e9 / fn_count);
    }
    let fmean = fns.iter().sum::<f64>() / fns.len() as f64;
    println!(
        "flat i128 control (serial, score-pass dots): {fmean:.4} ns/mult-add (runs {:.4} {:.4} {:.4})",
        fns[0], fns[1], fns[2]
    );
}

/// The formula `parts::mul` records — the floor-form round-half-to-even of a `Q64.64` product —
/// against the canonical `q32::round_q32_even`, on two million random products and on the sign and
/// tie edges. The source's `mul` is a copy of this expression inlined into the attention body, so
/// this test guards the derivation the digests of `attn_outputs_are_pinned` then exercise at 235 M
/// products per fixture; it does not reach the inlined body itself.
#[test]
fn mul_rounding_identity() {
    use apeiron::learn::solve::q32::round_q32_even;
    fn floor_form(p: i128) -> i64 {
        let q = p >> 32;
        let r = p & 0xffff_ffff;
        let half = 1_i128 << 31;
        (q + i128::from(r > half || (r == half && (q & 1) != 0))) as i64
    }
    let mut s = 0x1234_5678_9abc_def0_u64;
    let mut next = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        i64::try_from(s >> 11).unwrap() - (1 << 52)
    };
    for _ in 0..2_000_000 {
        let (a, b) = (next() % (1 << 31), next() % (1 << 31));
        let p = i128::from(a) * i128::from(b);
        assert_eq!(floor_form(p), round_q32_even(p).unwrap(), "a={a} b={b}");
    }
    // Ties round to the even neighbour on both sides of zero; 0, ±1 and the extreme words cross the
    // sign and narrowing edges.
    for &(a, b) in &[
        (1_i64 << 31, 1_i64),
        (3, 1_i64 << 31),
        (-(1_i64 << 31), 1),
        (-(1_i64 << 31), 3),
        (0, 1),
        (1, 1),
        (-1, 1),
        (i64::from(i32::MIN), 1),
        (1, i64::from(i32::MIN)),
        (i64::MAX, 1),
        (i64::MAX, i64::MAX),
        (i64::MIN, i64::MAX),
    ] {
        let p = i128::from(a) * i128::from(b);
        let want = round_q32_even(p);
        if (a == i64::MAX || a == i64::MIN) && b == i64::MAX {
            assert!(want.is_err(), "off the grid must stay off: a={a} b={b}");
        } else {
            assert_eq!(floor_form(p), want.unwrap(), "edge a={a} b={b}");
        }
    }
}
