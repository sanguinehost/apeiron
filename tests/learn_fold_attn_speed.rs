//! `tests/learn_fold_attn_speed.rs` — the pinned output, the accuracy witness and the unit-cost
//! bench for `stats::parts::attn_a` (`src/learn/fold/stats/parts.rs`), which contracts its scores
//! and its value mix on `engine::kern::limb::gemm_q32` at three limbs. Production geometry of the
//! `Qwen2.5-0.5B` teacher (`n_head 14`, `n_kv_head 2`, `head_dim 64`, `hidden 896`), production
//! sequence length `t = 512`.
//!
//! Positive: the FNV-1a digests of the attention output on three seeded fixtures are pinned; one of
//! them (`t = 130`) has a length that is not a power of two and its last row is the causal boundary
//! row (it attends to all 130 keys). A perturbed final query row leaves every earlier row
//! bit-for-bit unchanged. `mul_rounding_identity` checks the floor-form round-half-to-even that
//! `parts::mul` records against the canonical `q32::round_q32_even` on random products and on the
//! sign and tie edges.
//!
//! Accuracy: [`limb_path_agrees_with_the_exact_body`] holds the limb path against [`old_attn`], the
//! exact pre-port body (every score and every value accumulation an `i128` dot rounded once), and
//! asserts the entrywise difference is inside the limb module's propagated bound with a relative
//! error below `2⁻¹⁵`. Measured: `2.0e-6` absolute and `2.0e-6` relative at `t = 512`.
//!
//! Measurement: `unit_cost` times one `attn_a` call at `t = 512`, three runs, and reports the mean
//! and spread in ns per multiply-add; the same test times `flat`, a serial `i128` dot over the same
//! `t(t+1)/2` triangular population, as the instrument's control for a loop that only multiplies
//! and adds. `unit_cost_paired` is the ignored paired bench: five alternating limb and exact runs.
//! The body `attn_a` replaced is pooled (`engine::chart::pool::chunks_mut_global` over
//! `(position, head)` rows), and that pool is `pub(crate)`, so its paired arm lives in
//! `src/learn/fold/stats/parts.rs` (`mod k5`, `cargo test --release --lib k5_paired -- --ignored
//! --nocapture`): two campaigns of five alternating runs at `t = 512` give `0.228`/`0.240` against
//! `0.421`/`0.424` ns per word multiply-add, ratio `0.54`/`0.57`.
//! Grade `tested` for the digests and the accuracy bound, `measured` for the timings.
//!
//! Mutation controls, both run and restored: `parts::LIMBS` from 3 to 2 reddens
//! `limb_path_agrees_with_the_exact_body` and `attn_outputs_are_pinned`; dropping the causal mask
//! (`scores[r·extent..][..=i]` to `[..extent]`) reddens `attn_outputs_are_pinned`.
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
use apeiron::learn::solve::q32::{div_rne, narrow, round_q32_even, to_grid, GRID_BOUND};
use apeiron::numeric::fixed_exp::fixed_exp;

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
    let (c, rope, q, k, v) = fixture(seed, t);
    attn_a(&c, &rope, &q, &k, &v, t).expect("attention")
}

/// One seeded fixture: the geometry, its rope tables and the three word panels.
fn fixture(seed: u64, t: usize) -> (LayerCfg, RopeTables, Vec<i64>, Vec<i64>, Vec<i64>) {
    let c = cfg();
    let rope = RopeTables::new(&c, t).expect("rope");
    let mut rng = Lcg::new(seed);
    let q = rng.words(t * NH * HD);
    let k = rng.words(t * NKV * HD);
    let v = rng.words(t * NKV * HD);
    (c, rope, q, k, v)
}

/// `a*b` in `Q32.32`, rounded once to nearest, ties to even: the pre-port `parts::mul` expression,
/// written here so the reference body and the module's are compared on one rounding rule.
fn mul(a: i64, b: i64) -> Option<i64> {
    let p = i128::from(a) * i128::from(b);
    let q = p >> 32;
    let r = p & 0xffff_ffff;
    let half = 1_i128 << 31;
    narrow(q + i128::from(r > half || (r == half && (q & 1) != 0)))
}

/// Ratio of one `Q32.32` word: the pre-port `parts::div`.
fn div(a: i64, b: i64) -> Option<i64> {
    div_rne(i128::from(a) << 32, i128::from(b)).and_then(narrow)
}

/// The pre-port `parts::exp_sat`: `e^x` clamped at `e²¹`, flushed below `e⁻¹⁶`.
fn exp_sat(x: i64) -> Option<i64> {
    const ONE: i64 = 1 << 32;
    let x = x.min(21 * ONE);
    match fixed_exp(x) {
        Ok(v) => Some(v),
        Err(_) if x < -16 * ONE => Some(0),
        Err(_) => None,
    }
}

/// The floored integer square root of `n·2³²`, by the bit-by-bit digit algorithm: an independent
/// root for the reference body's attention scale (`parts` uses `numeric::fixed_rsqrt::isqrt`).
fn isqrt(n: i128) -> i128 {
    let mut bit = 1_i128 << 126;
    let mut rest = n;
    let mut root = 0_i128;
    while bit > rest {
        bit >>= 2;
    }
    while bit != 0 {
        if rest >= root + bit {
            rest -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}

/// The pre-port half-rotation `dst = src·cos(emb) + rot(src)·sin(emb)`, one row at a time.
fn rope_apply(dst: &mut [i64], src: &[i64], t: usize, n_head: usize, hd: usize, rope: &RopeTables) {
    let half = hd / 2;
    for p in 0..t {
        for h in 0..n_head {
            let base = p * n_head * hd + h * hd;
            for d in 0..hd {
                let r = if d < half {
                    -src[base + d + half]
                } else {
                    src[base + d - half]
                };
                dst[base + d] = mul(src[base + d], rope.cos[p * hd + d])
                    .and_then(|a| mul(r, rope.sin[p * hd + d]).and_then(|b| a.checked_add(b)))
                    .expect("rope");
            }
        }
    }
}

/// The pre-port attention body (`git show 71e766ed:src/learn/fold/stats/parts.rs`, `AttnRows::rows`
/// and `attn_words`): every score and every value accumulation an exact `i128` dot rounded once,
/// one `(position, head)` row at a time, serial. This is the body [`attn_a`] replaced and the
/// reference the limb path is graded against.
fn old_attn(
    c: &LayerCfg,
    rope: &RopeTables,
    q: &[i64],
    k: &[i64],
    v: &[i64],
    t: usize,
) -> Vec<i64> {
    const ONE: i64 = 1 << 32;
    let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
    let root = i64::try_from(isqrt(i128::try_from(hd).expect("head width") << 64)).expect("root");
    let scale = div(ONE, root).expect("scale");
    let (mut qr, mut kr) = (vec![0_i64; t * nh * hd], vec![0_i64; t * nkv * hd]);
    rope_apply(&mut qr, q, t, nh, hd, rope);
    rope_apply(&mut kr, k, t, nkv, hd, rope);
    let rep = nh / nkv;
    let mut out = vec![0_i64; t * nh * hd];
    for i in 0..t {
        for h in 0..nh {
            let gbase = (h / rep) * hd;
            let qrow = &qr[i * nh * hd + h * hd..][..hd];
            let mut scores = vec![0_i64; t];
            let mut row_max = i64::MIN;
            for (j, slot) in scores.iter_mut().enumerate().take(i + 1) {
                let krow = &kr[j * nkv * hd + gbase..][..hd];
                let mut dot: i128 = 0;
                for d in 0..hd {
                    dot += i128::from(qrow[d]) * i128::from(krow[d]);
                }
                *slot =
                    mul(round_q32_even(dot).expect("score in grid"), scale).expect("scaled score");
                row_max = row_max.max(*slot);
            }
            let mut acc = vec![0_i64; hd];
            let mut denom: i64 = 0;
            for (j, &s) in scores.iter().enumerate().take(i + 1) {
                let e = exp_sat(s - row_max).expect("weight");
                denom += e;
                let vbase = j * nkv * hd + gbase;
                for (a, &vj) in acc.iter_mut().zip(&v[vbase..][..hd]) {
                    *a += mul(e, vj).expect("mix term");
                }
            }
            for (o, &a) in out[i * nh * hd + h * hd..][..hd].iter_mut().zip(&acc) {
                *o = div(a, denom).expect("mix");
            }
        }
    }
    out
}

/// The limb module's entrywise bound `G(n, L, max|a|, max|b|)` in `Q32.32` words
/// (`src/engine/kern/limb.rs`, "Error bound"): `⌈P/2⌉ + 1 + B·(1 + ⌈C_L·2^{Ea+Eb−32}⌉)` with
/// `L = 3`, `P = L(L+1)/2`, `C_L = (33 + 16(L−1))·2^{−7L}`, `B = ⌈n/32⌉` and
/// `Ea = ⌊log₂ max|a|⌋ + 2`.
fn limb_bound(n: usize, max_a: f64, max_b: f64) -> f64 {
    let l = 3.0_f64;
    let p = l * (l + 1.0) / 2.0;
    let c_l = (33.0 + 16.0 * (l - 1.0)) * 2f64.powf(-7.0 * l);
    let ea = max_a.log2().floor() + 2.0;
    let eb = max_b.log2().floor() + 2.0;
    let b = (n as f64 / 32.0).ceil();
    (p / 2.0).ceil() + 1.0 + b * (1.0 + (c_l * 2f64.powf(ea + eb - 32.0)).ceil())
}

/// The absolute difference bound one `attn_a` output entry carries against [`old_attn`], in real
/// units, from the two contraction bounds `G` of `src/engine/kern/limb.rs` (both in `Q32.32`
/// words): the score error `g_s·2⁻³²` scales every weight by `exp`, so the mix numerator moves by
/// at most `g_s·2⁻³²·Σe_j·max|v|`, the mix's own limb error is `g_m` words, and the division adds
/// one quantum. `Σe_j ≥ 2³²` (the row maximum's weight is one), so the numerator error divided by
/// the denominator is bounded by these terms unchanged.
fn attn_bound(c: &LayerCfg, q: &[i64], k: &[i64], v: &[i64], t: usize) -> f64 {
    let qmax = q.iter().map(|v| v.unsigned_abs()).max().unwrap_or(1) as f64;
    let kmax = k.iter().map(|v| v.unsigned_abs()).max().unwrap_or(1) as f64;
    let vmax = v.iter().map(|v| v.unsigned_abs()).max().unwrap_or(1) as f64;
    let g_s = limb_bound(c.head_dim, qmax, kmax);
    let g_m = limb_bound(t, 4_294_967_296.0, vmax);
    let q32 = 2f64.powi(-32);
    (g_m + g_s * vmax * q32 + 2.0) * q32
}

/// The pinned digests of the limb path on the three fixtures; the limb bound and the tolerance
/// assertion that license these pins are in [`limb_path_agrees_with_the_exact_body`]. Re-derived
/// after the entrywise error against the exact pre-port body was measured; the `t = 512` pin moved
/// again, inside that same bound, when `parts::BAND` grew to the whole sequence (the value mix
/// contracts over the band's `extent`, so the limb blocks it truncates over change with the band;
/// `t = 130` is one band either way).
#[test]
fn attn_outputs_are_pinned() {
    // `t = 130` is not a multiple of a power-of-two tile and its row 129 is the causal boundary
    // row (it reads all 130 keys); `t = 512` is the production length.
    let d130a = digest(&run(0x5eed_0001, 130));
    let d130b = digest(&run(0x5eed_0002, 130));
    let d512 = digest(&run(0x5eed_0003, 512));
    println!("attn digests t=130 s1 {d130a:#018x} s2 {d130b:#018x} t=512 s3 {d512:#018x}");
    assert_eq!(d130a, 0xa5b7_21cd_6a32_1e2a);
    assert_eq!(d130b, 0x8aef_36cc_075f_c4b9);
    assert_eq!(d512, 0xc0d7_0ad2_660d_f0e8);
}

/// The limb path against the exact pre-port body [`old_attn`] on the three pinned fixtures: every
/// entry is inside the propagated limb bound, and the largest deviation relative to the output's
/// own scale `max|old|` stays within `2⁻¹⁵`. The limit is the block truncation the three limbs
/// carry, not a rounding-mode difference: `C₃ = 65·2⁻²¹` of the block exponent per block, summed
/// over the `t` value terms the mix contracts (`O(t)·2⁻²¹ ≈ 2⁻¹⁵` at `t = 512`). Reports the
/// observed maximum, absolute and relative, per fixture.
#[test]
fn limb_path_agrees_with_the_exact_body() {
    for (seed, t) in [
        (0x5eed_0001_u64, 130_usize),
        (0x5eed_0002, 130),
        (0x5eed_0003, 512),
    ] {
        let (c, rope, q, k, v) = fixture(seed, t);
        let new = attn_a(&c, &rope, &q, &k, &v, t).expect("attention");
        let old = old_attn(&c, &rope, &q, &k, &v, t);
        let bound = attn_bound(&c, &q, &k, &v, t);
        let worst = new
            .iter()
            .zip(&old)
            .map(|(&x, &y)| x.abs_diff(y))
            .max()
            .unwrap_or(0);
        let scale = old
            .iter()
            .map(|v| v.unsigned_abs())
            .max()
            .unwrap_or(1)
            .max(1);
        let (abs, rel) = (worst as f64 * 2f64.powi(-32), worst as f64 / scale as f64);
        println!("t={t} seed={seed:#x}: max |Δ| {abs:.3e} (bound {bound:.3e}), relative {rel:.3e}");
        assert!(abs <= bound, "t={t}: {abs:e} > bound {bound:e}");
        assert!(rel <= 2f64.powi(-15), "t={t}: relative {rel:e}");
    }
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

/// Paired port bench: five alternating runs of the exact pre-port body and the kernel limb path on
/// one production-shape fixture (`t = 512`), reporting ns per multiply-add for each arm and the
/// spread. Ignored so the gate run stays fast; run
/// `cargo test --release --test learn_fold_attn_speed -- --ignored unit_cost_paired --nocapture`.
///
/// The multiply-add count is [`mult_adds`]; the limb arm executes `limbs(limbs+1)/2 = 6` limb pairs
/// per word product, so its ns per *word* multiply-add is what the two columns compare.
#[test]
#[ignore = "measurement: release run, five alternating arms"]
fn unit_cost_paired() {
    let t = 512;
    let (c, rope, q, k, v) = fixture(0x5eed_0006, t);
    let n = mult_adds(t);
    let (mut new_ns, mut old_ns) = (Vec::new(), Vec::new());
    for _ in 0..5 {
        let t0 = Instant::now();
        let a = attn_a(&c, &rope, &q, &k, &v, t).expect("attention");
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(&a);
        new_ns.push(dt * 1e9 / n);

        let t0 = Instant::now();
        let b = old_attn(&c, &rope, &q, &k, &v, t);
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(&b);
        old_ns.push(dt * 1e9 / n);
    }
    let stats = |xs: &[f64]| {
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = xs.iter().copied().fold(0.0_f64, f64::max);
        (mean, hi - lo)
    };
    let (nm, ns_) = stats(&new_ns);
    let (om, os) = stats(&old_ns);
    println!("attn t=512 limb  {nm:.4} ns/MAC (spread {ns_:.4}; runs {new_ns:?})");
    println!("attn t=512 exact {om:.4} ns/MAC (spread {os:.4}; runs {old_ns:?})");
    println!("ratio limb/exact {:.3}", nm / om);
}
