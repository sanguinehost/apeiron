//! `tests/learn_fold_plane.rs` — the finishing stage's coded weight plane: the `encode` index round
//! trip and exactness flag, the frozen-code forward against the pre-migration `f64` oracle
//! (`ref_forward`), the straight-through gradient against central differences of the Q32.32
//! unquantized forward and against the `f64` oracle backward (`ref_backward`), the row-restricted
//! `add_into` / `backward`, the recorded word digests of every entry point on two seeds, and the
//! shape refusals. Grade `tested`; nothing here is `proved`.
//!
//! The two `ref_*` bodies are the `f64` implementation this lane replaced, kept verbatim in
//! structure as the oracle. They run on `f64` copies of the Q32.32 state (`from_grid`), so the
//! comparison isolates the migration's arithmetic; each assertion carries the bound derived in
//! `src/learn/fold/finish/plane.rs` (`8·2^-32` for the forward's four roundings) or `32·2^-32` for
//! the backward's longer chains, scaled by the tensor's largest magnitude.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    reason = "the fixed-seed LCG maps its 53 high bits to an f64 and its low bits to an index; the \
              short `o/i/j/k/m` indices and the `r`/`c`/`d` carrier names are the reference \
              arithmetic's vocabulary; the `f64` oracle is the pre-migration body and its \
              tolerance literals are the derived Q32.32 bounds; the refusal test enumerates every \
              shape arm in one body; the `ref_*` oracles take one parameter per plane field so \
              they read as the pre-migration signatures they copy."
)]

use apeiron::learn::fold::finish::plane::{encode, CodedPlane, PlaneError};
use apeiron::learn::fold::fit::icm::codebook;
use apeiron::learn::fold::fit::quant::{quant_c_u8, quant_r_hier};
use apeiron::learn::fold::fit::steps::{compose, GROUP};
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

#[path = "support/fwht_oracle.rs"]
mod fwht_oracle;

use fwht_oracle::fwht128;

/// The f64 fixture as Q32.32 words, the quantiser's input at the test's own data.
fn words(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&v| to_grid(v, GRID_BOUND).expect("fixture on the grid"))
        .collect()
}

/// The word grid as reals, the test's own read-back for its f64 arithmetic.
fn reals(x: &[i64]) -> Vec<f64> {
    x.iter().map(|&w| from_grid(w)).collect()
}

/// Out rows of the main fixtures.
const O: usize = 4;
/// In columns (a whole number of 128-blocks).
const N: usize = 256;
/// Group count along `n`.
const G: usize = N / GROUP;
/// Code-block width.
const BLOCK: usize = 8;
/// Blocks per row.
const NB: usize = N / BLOCK;
/// Out rows of the row-restricted fixture.
const O_P: usize = 2;
/// The central-difference step of the straight-through gradient check, as a Q32.32 word
/// (`round(1e-5·2^32)`).
const H_Q32: i64 = 42_950;
/// One Q32.32 unit.
const Q: f64 = 1.0 / 4_294_967_296.0;

/// Deterministic 64-bit linear congruential generator (fixed seed, no `rand` crate).
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

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Uniform in `[0.2, 1.0)`.
    fn pos(&mut self) -> f64 {
        0.2 + 0.8 * self.unit()
    }
}

/// The quantized base pack of one plane, in Q32.32: the `r` grid `(lo, sp)` and the quantized scales
/// `r`, `c`, `d`. `r` is a fixed point of the row-grid snap and `c`/`d` of `quant_c_u8`.
struct Pack {
    lo: Vec<i64>,
    sp: Vec<i64>,
    r: Vec<i64>,
    c: Vec<i64>,
    d: Vec<i64>,
}

fn build_pack(rng: &mut Lcg, o: usize) -> Pack {
    let r_raw: Vec<f64> = (0..o * G).map(|_| rng.pos()).collect();
    let rh = quant_r_hier(&words(&r_raw), o, G).unwrap();
    let c_raw: Vec<f64> = (0..N).map(|_| 0.5 + 0.5 * rng.unit()).collect();
    let c0 = quant_c_u8(&words(&c_raw), 6).unwrap().0;
    let d_raw: Vec<f64> = (0..N).map(|_| 0.5 + 0.5 * rng.unit()).collect();
    let d0 = quant_c_u8(&words(&d_raw), 5).unwrap().0;
    Pack {
        lo: rh.r_lo,
        sp: rh.r_sp,
        r: rh.rq,
        c: c0,
        d: d0,
    }
}

/// `o·(n/8)` codebook indices in `0..256`.
fn indices(rng: &mut Lcg, o: usize) -> Vec<u8> {
    (0..o * NB).map(|_| (rng.next_u64() % 256) as u8).collect()
}

/// The exact code field from a list of codebook indices, `o × n` row-major, as Q32.32 words.
fn code_field(idx: &[u8], o: usize, n: usize) -> Vec<i64> {
    let cb = codebook();
    let nb = n / BLOCK;
    let mut s = vec![0_i64; o * n];
    for oi in 0..o {
        for m in 0..nb {
            let j = idx[oi * nb + m] as usize;
            for k in 0..BLOCK {
                s[oi * n + m * BLOCK + k] = cb[j][k];
            }
        }
    }
    s
}

/// `s ⊙ d` over the rows of an `o × n` buffer (oracle).
fn scale_cols(s: &[f64], d: &[f64], o: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0.0_f64; o * n];
    for row in 0..o {
        for col in 0..n {
            t[row * n + col] = s[row * n + col] * d[col];
        }
    }
    t
}

/// The pre-migration `f64` forward `compose(fwht128(S ⊙ d), r, c)` over the Q32.32 state read back
/// as `f64`. The oracle the migrated [`CodedPlane::weight`] is asserted against.
fn ref_forward(s: &[i64], r: &[i64], c: &[i64], d: &[i64], o: usize, n: usize) -> Vec<f64> {
    let sf: Vec<f64> = s.iter().map(|&v| from_grid(v)).collect();
    let df: Vec<f64> = d.iter().map(|&v| from_grid(v)).collect();
    let y = fwht128(&scale_cols(&sf, &df, o, n)).unwrap();
    let yq: Vec<i64> = y.iter().map(|&v| to_grid(v, GRID_BOUND).unwrap()).collect();
    compose(&yq, r, c, o, n)
        .unwrap()
        .into_iter()
        .map(from_grid)
        .collect()
}

/// The pre-migration `f64` backward at the same state: the documented `dr`/`dc`/`dY`/`dd` formulas
/// with the quantized forward values.
fn ref_backward(
    s: &[i64],
    r_l: &[i64],
    c_l: &[i64],
    d_l: &[i64],
    lo: &[i64],
    sp: &[i64],
    grad_w: &[i64],
    o: usize,
    n: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let g = n / GROUP;
    let sf: Vec<f64> = s.iter().map(|&v| from_grid(v)).collect();
    let rl: Vec<f64> = r_l.iter().map(|&v| from_grid(v)).collect();
    let cl: Vec<f64> = c_l.iter().map(|&v| from_grid(v)).collect();
    let dl: Vec<f64> = d_l.iter().map(|&v| from_grid(v)).collect();
    let lof: Vec<f64> = lo.iter().map(|&v| from_grid(v)).collect();
    let spf: Vec<f64> = sp.iter().map(|&v| from_grid(v)).collect();
    let gp: Vec<f64> = grad_w.iter().map(|&v| from_grid(v)).collect();
    // The forward quantizers at the same state.
    let mut rq = vec![0.0_f64; o * g];
    for oi in 0..o {
        for gg in 0..g {
            let x = ((rl[oi * g + gg] - lof[oi]) / spf[oi] * 15.0).clamp(0.0, 15.0);
            let k = x.round_ties_even();
            rq[oi * g + gg] = lof[oi] + (k / 15.0) * spf[oi];
        }
    }
    let cq = reals(&quant_c_u8(&words(&cl), 6).unwrap().0);
    let dq = reals(&quant_c_u8(&words(&dl), 5).unwrap().0);
    let x = fwht128(&scale_cols(&sf, &dq, o, n)).unwrap();
    let mut gy = vec![0.0_f64; o * n];
    for k in 0..o * n {
        gy[k] = gp[k] * x[k];
    }
    let mut dr = vec![0.0_f64; o * g];
    for oi in 0..o {
        for gg in 0..g {
            let mut acc = 0.0_f64;
            for j in gg * GROUP..(gg + 1) * GROUP {
                acc += gy[oi * n + j] * cq[j];
            }
            dr[oi * g + gg] = acc;
        }
    }
    let mut dc = vec![0.0_f64; n];
    for oi in 0..o {
        for j in 0..n {
            dc[j] += gy[oi * n + j] * rq[oi * g + j / GROUP];
        }
    }
    let mut t = vec![0.0_f64; o * n];
    for oi in 0..o {
        for j in 0..n {
            t[oi * n + j] = gp[oi * n + j] * rq[oi * g + j / GROUP] * cq[j];
        }
    }
    let dy = fwht128(&t).unwrap();
    let mut dd = vec![0.0_f64; n];
    for oi in 0..o {
        for j in 0..n {
            dd[j] += sf[oi * n + j] * dy[oi * n + j];
        }
    }
    let mut dpa = vec![0.0_f64; o * g];
    for k in 0..o * g {
        dpa[k] = dr[k] * rl[k];
    }
    let mut dpb = vec![0.0_f64; n];
    for j in 0..n {
        dpb[j] = dc[j] * cl[j];
    }
    let mut dpd = vec![0.0_f64; n];
    for j in 0..n {
        dpd[j] = dd[j] * dl[j];
    }
    (dpa, dpb, dpd)
}

/// Assert `|q − ref| ≤ ulps·2^-32·max(1, max|ref|)`, the derived Q32.32 bound.
fn assert_close(q: &[i64], reference: &[f64], ulps: f64, name: &str) {
    assert_eq!(q.len(), reference.len(), "{name}: length");
    let scale = reference.iter().fold(1.0_f64, |m, &v| m.max(v.abs()));
    let bound = ulps * Q * scale;
    let mut worst = 0.0_f64;
    for k in 0..q.len() {
        let d = (from_grid(q[k]) - reference[k]).abs();
        worst = worst.max(d);
        assert!(
            d <= bound,
            "{name}[{k}]: {} vs {} (|d| {d:e} > bound {bound:e})",
            from_grid(q[k]),
            reference[k]
        );
    }
    println!("{name}: max |d| {worst:e} <= bound {bound:e}");
}

fn assert_bits(a: &[i64], b: &[i64]) {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    for k in 0..a.len() {
        assert_eq!(a[k], b[k], "word {k}: {} vs {}", a[k], b[k]);
    }
}

/// `count` distinct indices in `0..len`, drawn from the LCG.
fn pick(len: usize, count: usize, rng: &mut Lcg) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    while out.len() < count.min(len) {
        let v = (rng.next_u64() as usize) % len;
        if !out.contains(&v) {
            out.push(v);
        }
    }
    out
}

/// `0.5·Σ (W ⊙ M)²` on the unquantized Q32.32 forward.
fn loss(plane: &CodedPlane, m: &[f64]) -> f64 {
    let w = plane.weight_unquantized();
    let mut acc = 0.0_f64;
    for (a, b) in w.iter().zip(m) {
        let t = from_grid(*a) * b;
        acc += t * t;
    }
    0.5 * acc
}

/// Set the offsets to `base` with `delta` added to coordinate `t` of axis `which` (`0 = pa`,
/// `1 = pb`, `2 = pd`). The offset boundary is a Q32.32 word.
fn apply(
    plane: &mut CodedPlane,
    base: (&[i64], &[i64], &[i64]),
    which: usize,
    t: usize,
    delta: i64,
) {
    let (pa0, pb0, pd0) = base;
    let mut pa = pa0.to_vec();
    let mut pb = pb0.to_vec();
    let mut pd = pd0.to_vec();
    match which {
        0 => pa[t] += delta,
        1 => pb[t] += delta,
        _ => pd[t] += delta,
    }
    plane.set_offsets(&pa, &pb, &pd).unwrap();
}

/// The central difference of the loss in coordinate `t` of axis `which`.
fn central(
    plane: &mut CodedPlane,
    base: (&[i64], &[i64], &[i64]),
    which: usize,
    t: usize,
    m: &[f64],
) -> f64 {
    apply(plane, base, which, t, H_Q32);
    let plus = loss(plane, m);
    apply(plane, base, which, t, -H_Q32);
    let minus = loss(plane, m);
    apply(plane, base, which, t, 0);
    (plus - minus) / (2.0 * from_grid(H_Q32))
}

fn rel(fd: f64, an: i64, name: &str, idx: usize) {
    let an = from_grid(an);
    let d = (fd - an).abs();
    let scale = an.abs().max(1e-6);
    assert!(
        d / scale < 5e-3,
        "{name}[{idx}]: fd {fd:e} analytic {an:e} rel {:.3e}",
        d / scale
    );
}

/// FNV-1a over a word's eight little-endian bytes, the recorded-witness hash.
fn fnv_word(d: &mut u64, v: i64) {
    for b in v.to_le_bytes() {
        *d ^= u64::from(b);
        *d = d.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// FNV-1a over a buffer, length-prefixed so a truncation moves the digest.
fn fnv_buf(d: &mut u64, xs: &[i64]) {
    fnv_word(d, i64::try_from(xs.len()).unwrap_or(i64::MAX));
    for &v in xs {
        fnv_word(d, v);
    }
}

const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

/// The recorded word digests of every plane entry point on two seeds: the pack in, the parsed
/// fields, the forward, the offsets and backward, `add_into` and `encode`.
#[test]
fn recorded_plane_digests() {
    let names = ["pack", "fields", "forward", "offsets", "add_into", "encode"];
    for seed in [7_u64, 23] {
        // Recorded on the pre-migration f64 boundary; the words API reproduces them exactly.
        let want: [u64; 6] = if seed == 7 {
            [
                0xaa71_31af_72a6_fccc,
                0xc41e_5469_fc1e_3105,
                0xba74_5d70_d411_78b4,
                0x0c06_5213_743d_136f,
                0xfa5b_c32d_a185_724c,
                0xb675_ad3c_3aac_2124,
            ]
        } else {
            [
                0x1db1_49cc_4942_8912,
                0x12e6_b6ee_ff3a_dd7c,
                0x2a86_2462_6b41_5967,
                0x2edd_6683_d6ca_c5d3,
                0x70cf_bbaf_2326_e506,
                0x3b18_5e6d_0cfd_2024,
            ]
        };
        let mut got = [0_u64; 6];
        let mut rng = Lcg::new(seed);
        let pack = build_pack(&mut rng, O);
        let idx = indices(&mut rng, O);
        let s = code_field(&idx, O, N);
        let mut plane = CodedPlane::from_pack(
            &s, &pack.lo, &pack.sp, &pack.r, &pack.c, &pack.d, None, O, N,
        )
        .unwrap();
        let mut d = FNV0;
        fnv_buf(&mut d, &s);
        for &b in &idx {
            fnv_word(&mut d, i64::from(b));
        }
        for v in [&pack.lo, &pack.sp, &pack.r, &pack.c, &pack.d] {
            fnv_buf(&mut d, v);
        }
        got[0] = d;

        let mut d = FNV0;
        for &b in &plane.idx {
            fnv_word(&mut d, i64::from(b));
        }
        fnv_word(&mut d, i64::from(plane.s_fix.is_some()));
        for v in [&plane.lo, &plane.sp, &plane.r0, &plane.c0, &plane.d0] {
            fnv_buf(&mut d, v);
        }
        got[1] = d;

        let mut d = FNV0;
        fnv_buf(&mut d, &plane.code_field());
        let (r_l, c_l, d_l) = plane.latents();
        let (r_q, c_q, d_q) = plane.qvals();
        for v in [&r_l, &c_l, &d_l, &r_q, &c_q, &d_q] {
            fnv_buf(&mut d, v);
        }
        fnv_buf(&mut d, &plane.weight());
        fnv_buf(&mut d, &plane.weight_unquantized());
        got[2] = d;

        let pa_raw: Vec<f64> = (0..O * G).map(|_| 0.1 * (rng.unit() - 0.5)).collect();
        let pb_raw: Vec<f64> = (0..N).map(|_| 0.1 * (rng.unit() - 0.5)).collect();
        let pd_raw: Vec<f64> = (0..N).map(|_| 0.1 * (rng.unit() - 0.5)).collect();
        let (pa, pb, pd) = (words(&pa_raw), words(&pb_raw), words(&pd_raw));
        plane.set_offsets(&pa, &pb, &pd).unwrap();
        let g: Vec<i64> = (0..O * N)
            .map(|_| ((rng.unit() - 0.5) * 4_294_967_296.0) as i64)
            .collect();
        let (dpa, dpb, dpd) = plane.backward(&g).unwrap();
        let mut d = FNV0;
        for v in [&plane.pa, &plane.pb, &plane.pd] {
            fnv_buf(&mut d, v);
        }
        let (r_l, c_l, d_l) = plane.latents();
        let (r_q, c_q, d_q) = plane.qvals();
        for v in [&r_l, &c_l, &d_l, &r_q, &c_q, &d_q] {
            fnv_buf(&mut d, v);
        }
        fnv_buf(&mut d, &plane.weight());
        for v in [&dpa, &dpb, &dpd] {
            fnv_buf(&mut d, v);
        }
        got[3] = d;

        let pw = plane.weight();
        let mut w = vec![0_i64; 4 * N];
        plane.add_into(&mut w, &pw, 4).unwrap();
        let mut d = FNV0;
        fnv_buf(&mut d, &w);
        got[4] = d;

        let (enc, exact) = encode(&s, O, N).unwrap();
        let mut d = FNV0;
        for &b in &enc {
            fnv_word(&mut d, i64::from(b));
        }
        fnv_word(&mut d, i64::from(exact));
        got[5] = d;

        for (i, name) in names.iter().enumerate() {
            assert_eq!(got[i], want[i], "seed {seed} {name} word digest");
        }
    }
}

#[test]
fn from_pack_exact_field_reproduces_reference_within_the_derived_bound() {
    let mut rng = Lcg::new(7);
    let idx = indices(&mut rng, O);
    let pack = build_pack(&mut rng, O);
    let s = code_field(&idx, O, N);
    let plane = CodedPlane::from_pack(
        &s, &pack.lo, &pack.sp, &pack.r, &pack.c, &pack.d, None, O, N,
    )
    .unwrap();
    assert!(
        plane.s_fix.is_none(),
        "an exact code field keeps idx, not s_fix"
    );
    assert_eq!(plane.idx, idx);
    // The base scales are canonicalised through the plane's own quantizers, so the quantized and
    // unquantized latents agree bit-for-bit at zero offsets.
    let (r_l, c_l, d_l) = plane.latents();
    let (r_q, c_q, d_q) = plane.qvals();
    assert_bits(&r_q, &r_l);
    assert_bits(&c_q, &c_l);
    assert_bits(&d_q, &d_l);
    // The migrated forward against the pre-migration f64 body at the same Q32.32 state.
    let reference = ref_forward(&plane.code_field(), &plane.r0, &plane.c0, &plane.d0, O, N);
    assert_close(&plane.weight(), &reference, 8.0, "weight");
    let u = ref_forward(&plane.code_field(), &plane.r0, &plane.c0, &plane.d0, O, N);
    assert_close(&plane.weight_unquantized(), &u, 8.0, "weight_unquantized");
}

#[test]
fn backward_matches_finite_differences_of_the_unquantized_forward() {
    let mut rng = Lcg::new(11);
    let idx = indices(&mut rng, O);
    let pack = build_pack(&mut rng, O);
    let s = code_field(&idx, O, N);
    let mut plane = CodedPlane::from_pack(
        &s, &pack.lo, &pack.sp, &pack.r, &pack.c, &pack.d, None, O, N,
    )
    .unwrap();
    // A fixed quadratic field M; the loss gradient to W is W ⊙ M ⊙ M.
    let m: Vec<f64> = (0..O * N).map(|_| 0.5 + rng.unit()).collect();
    let w = plane.weight_unquantized();
    let gvec: Vec<i64> = w
        .iter()
        .zip(&m)
        .map(|(a, b)| (from_grid(*a) * b * b * 4_294_967_296.0) as i64)
        .collect();
    let (dpa, dpb, dpd) = plane.backward(&gvec).unwrap();

    let base = (plane.pa.clone(), plane.pb.clone(), plane.pd.clone());
    let base_ref = (base.0.as_slice(), base.1.as_slice(), base.2.as_slice());
    for &t in &pick(O * G, 6, &mut rng) {
        rel(central(&mut plane, base_ref, 0, t, &m), dpa[t], "pa", t);
    }
    for &t in &pick(N, 6, &mut rng) {
        rel(central(&mut plane, base_ref, 1, t, &m), dpb[t], "pb", t);
    }
    for &t in &pick(N, 6, &mut rng) {
        rel(central(&mut plane, base_ref, 2, t, &m), dpd[t], "pd", t);
    }
}

#[test]
fn backward_agrees_with_the_f64_oracle_within_the_derived_bound() {
    let mut rng = Lcg::new(13);
    let idx = indices(&mut rng, O);
    let pack = build_pack(&mut rng, O);
    let s = code_field(&idx, O, N);
    let mut plane = CodedPlane::from_pack(
        &s, &pack.lo, &pack.sp, &pack.r, &pack.c, &pack.d, None, O, N,
    )
    .unwrap();
    // Nonzero offsets so the `exp` chain and the latent products are exercised.
    let pa_raw: Vec<f64> = (0..O * G).map(|_| 0.1 * (rng.unit() - 0.5)).collect();
    let pb_raw: Vec<f64> = (0..N).map(|_| 0.1 * (rng.unit() - 0.5)).collect();
    let pd_raw: Vec<f64> = (0..N).map(|_| 0.1 * (rng.unit() - 0.5)).collect();
    let (pa, pb, pd) = (words(&pa_raw), words(&pb_raw), words(&pd_raw));
    plane.set_offsets(&pa, &pb, &pd).unwrap();
    let gvec: Vec<i64> = (0..O * N)
        .map(|_| (0.25 * (rng.unit() - 0.5) * 4_294_967_296.0) as i64)
        .collect();
    let (dpa, dpb, dpd) = plane.backward(&gvec).unwrap();
    let (r_l, c_l, d_l) = plane.latents();
    let (ra, rb, rd) = ref_backward(
        &plane.code_field(),
        &r_l,
        &c_l,
        &d_l,
        &plane.lo,
        &plane.sp,
        &gvec,
        O,
        N,
    );
    assert_close(&dpa, &ra, 32.0, "dpa");
    assert_close(&dpb, &rb, 32.0, "dpb");
    assert_close(&dpd, &rd, 32.0, "dpd");
}

#[test]
fn encode_round_trips_indices_and_falls_back_to_s_fix() {
    let mut rng = Lcg::new(23);
    let idx = indices(&mut rng, O);
    let s = code_field(&idx, O, N);
    let (enc, exact) = encode(&s, O, N).unwrap();
    assert!(
        exact,
        "a field built from codebook rows must encode exactly"
    );
    assert_eq!(enc, idx);

    // Perturb one 8-block by one grid quantum: it is no longer a codebook row, so encode is inexact
    // and from_pack keeps the whole field verbatim.
    let mut s2 = s.clone();
    s2[5 * BLOCK + 3] += 1;
    let (_, exact2) = encode(&s2, O, N).unwrap();
    assert!(!exact2, "a perturbed block must clear the exactness flag");

    let pack = build_pack(&mut rng, O);
    let plane = CodedPlane::from_pack(
        &s2, &pack.lo, &pack.sp, &pack.r, &pack.c, &pack.d, None, O, N,
    )
    .unwrap();
    assert!(
        plane.s_fix.is_some(),
        "an inexact field is carried in s_fix"
    );
    let reference = ref_forward(&plane.code_field(), &plane.r0, &plane.c0, &plane.d0, O, N);
    assert_close(&plane.weight(), &reference, 8.0, "s_fix weight");
}

#[test]
fn rows_restrict_add_into_and_backward() {
    let mut rng = Lcg::new(31);
    let o_p = 2usize;
    let pack = build_pack(&mut rng, o_p);
    let idx = indices(&mut rng, o_p);
    let s = code_field(&idx, o_p, N);
    let rows = [1usize, 3];
    let plane = CodedPlane::from_pack(
        &s,
        &pack.lo,
        &pack.sp,
        &pack.r,
        &pack.c,
        &pack.d,
        Some(&rows),
        o_p,
        N,
    )
    .unwrap();
    let pw = plane.weight();
    assert_eq!(pw.len(), o_p * N);

    let o_total = 4usize;
    let mut w = vec![1_i64 << 32; o_total * N];
    plane.add_into(&mut w, &pw, o_total).unwrap();
    for j in 0..N {
        assert_eq!(w[j], 1_i64 << 32, "row 0 untouched");
        assert_eq!(w[2 * N + j], 1_i64 << 32, "row 2 untouched");
        assert_eq!(
            w[N + j],
            (1_i64 << 32) + pw[j],
            "row 1 receives plane row 0"
        );
        assert_eq!(
            w[3 * N + j],
            (1_i64 << 32) + pw[N + j],
            "row 3 receives plane row 1"
        );
    }

    // backward over the full 4-row gradient must gather rows 1 and 3 only.
    let g_full: Vec<i64> = (0..o_total * N)
        .map(|_| ((rng.unit() - 0.5) * 4_294_967_296.0) as i64)
        .collect();
    let mut g_part = vec![0_i64; o_p * N];
    for j in 0..N {
        g_part[j] = g_full[N + j];
        g_part[N + j] = g_full[3 * N + j];
    }
    let none = CodedPlane::from_pack(
        &s, &pack.lo, &pack.sp, &pack.r, &pack.c, &pack.d, None, o_p, N,
    )
    .unwrap();
    let a = plane.backward(&g_full).unwrap();
    let b = none.backward(&g_part).unwrap();
    assert_bits(&a.0, &b.0);
    assert_bits(&a.1, &b.1);
    assert_bits(&a.2, &b.2);
}

#[test]
fn shape_refusals() {
    let mut rng = Lcg::new(41);
    let pack = build_pack(&mut rng, O);
    let idx = indices(&mut rng, O);
    let s = code_field(&idx, O, N);
    let mut plane = CodedPlane::from_pack(
        &s, &pack.lo, &pack.sp, &pack.r, &pack.c, &pack.d, None, O, N,
    )
    .unwrap();

    // s_rot length disagrees with o·n.
    let short = vec![0_i64; O * N - 1];
    assert_eq!(
        CodedPlane::from_pack(&short, &pack.lo, &pack.sp, &pack.r, &pack.c, &pack.d, None, O, N)
            .unwrap_err(),
        PlaneError::Shape
    );
    // n is not a multiple of GROUP (128).
    let n2 = 192usize;
    let s2 = vec![1_i64 << 31; O * n2];
    let lo2 = vec![1_i64 << 32; O];
    let sp2 = vec![1_i64 << 32; O];
    let r2 = vec![1_i64 << 32; O * (n2 / GROUP)];
    let c2 = vec![1_i64 << 32; n2];
    let d2 = vec![1_i64 << 32; n2];
    assert_eq!(
        CodedPlane::from_pack(&s2, &lo2, &sp2, &r2, &c2, &d2, None, O, n2).unwrap_err(),
        PlaneError::Shape
    );
    // rows length must equal o.
    assert_eq!(
        CodedPlane::from_pack(
            &s,
            &pack.lo,
            &pack.sp,
            &pack.r,
            &pack.c,
            &pack.d,
            Some(&[0, 1]),
            O,
            N
        )
        .unwrap_err(),
        PlaneError::Shape
    );
    assert_eq!(
        plane
            .set_offsets(&[0_i64; O * G - 1], &[0_i64; N], &[0_i64; N])
            .unwrap_err(),
        PlaneError::Shape
    );
    // backward refuses a gradient that is not a whole number of rows.
    assert_eq!(
        plane.backward(&[0_i64; O * N - 1]).unwrap_err(),
        PlaneError::Shape
    );
    // add_into refuses a receiving weight of the wrong shape.
    let pw = plane.weight();
    assert_eq!(
        plane.add_into(&mut [0_i64; 2], &pw, O).unwrap_err(),
        PlaneError::Shape
    );

    // A row-restricted plane refuses a row index outside the receiving weight.
    let packp = build_pack(&mut rng, O_P);
    let idxp = indices(&mut rng, O_P);
    let sp_field = code_field(&idxp, O_P, N);
    let rows = [1usize, 3];
    let pp = CodedPlane::from_pack(
        &sp_field,
        &packp.lo,
        &packp.sp,
        &packp.r,
        &packp.c,
        &packp.d,
        Some(&rows),
        O_P,
        N,
    )
    .unwrap();
    let pwp = pp.weight();
    assert_eq!(
        pp.add_into(&mut [0_i64; O_P * N], &pwp, O_P).unwrap_err(),
        PlaneError::RowOutOfRange
    );
    assert_eq!(
        pp.backward(&[0_i64; O_P * N]).unwrap_err(),
        PlaneError::RowOutOfRange
    );
}
