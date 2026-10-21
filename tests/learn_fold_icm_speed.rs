//! `tests/learn_fold_icm_speed.rs`  — the witness and microbench of the `E8 ∪ O8` ICM sweep
//! (`e8_icm_step`, `e8_joint_icm_2plane`) at `o = 64`, `n = 512` and at the production `down`
//! shape `(o, n) = (896, 4864)`. Grade `tested`.
//!
//! The sweep's residual (`sweep_blocks`) updates in lazy batches: one `GROUP`-wide column batch per
//! dispatch instead of one `BLOCK`-wide one, with the in-batch columns applied to the decisions from
//! a running prefix. The base arm is the per-block form (`icm.rs` at `84f2d8d3`), captured by the
//! constants below; the witness pins the two quantities the regrouping must preserve — `J` never
//! worse and the code decisions unchanged on at least 99 % of the blocks.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unreadable_literal,
    reason = "the witness table pairs a recorded `J` with its disagreement rows, the LCG maps integers to `f64` by division, the flat row-major index arithmetic is the reference payload format, the representability and argmin checks are exact float comparisons, and the test helpers mirror the ported option surface"
)]

use apeiron::learn::code::e8::codeword;
use apeiron::learn::fold::fit::icm::{
    e8_icm_step, e8_joint_icm_2plane, j_rows, plane_w, JointStats, Plane, PlaneRef,
};
use apeiron::learn::solve::q32;

/// Deterministic xorshift64 LCG.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[lo, hi)` from the top 53 bits.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
        lo + (hi - lo) * u
    }
}

const BLOCK: usize = 8;

const GROUP: usize = 128;

const CODE_LEN: usize = 256;

/// A fixture as the `Q32.32` words the module consumes: `round(x·2³²)` at the grid range.
fn q(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&v| q32::to_grid(v, q32::GRID_BOUND).expect("fixture inside the grid"))
        .collect()
}

/// The 256 codewords as the `Q32.32` words the module stores.
fn codebook() -> Vec<[i64; BLOCK]> {
    (0..CODE_LEN)
        .map(|j| codeword(j).expect("row in range"))
        .collect()
}

fn pref(p: &Plane) -> PlaneRef<'_> {
    PlaneRef {
        s: &p.s,
        idx: &p.idx,
        r: &p.r,
        c: &p.c,
        d: &p.d,
    }
}

/// A positive-definite Toeplitz metric `Hl[i,j] = rho^|i-j|` (the autocorrelation of an AR(1) plus
/// the `j = i` term), a stand-in for the calibration Gram.
fn toeplitz(n: usize, rho: f64) -> Vec<f64> {
    let mut h = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = rho.powi(i32::try_from(i.abs_diff(j)).expect("d < 2^31"));
        }
    }
    h
}

/// A plane with random codebook rows, random group scales in `[0.5, 1.5)` and unit `c`, `d`.
fn rand_plane(o: usize, n: usize, rng: &mut Lcg) -> Plane {
    let nb = n / BLOCK;
    let g = n / GROUP;
    let cb = codebook();
    let mut s = vec![0_i64; o * n];
    let mut idx = vec![0_u16; o * nb];
    for row in 0..o {
        for m in 0..nb {
            let j = (rng.next_u64() % 256) as usize;
            idx[row * nb + m] = j as u16;
            let at = row * n + m * BLOCK;
            s[at..at + BLOCK].copy_from_slice(&cb[j]);
        }
    }
    Plane {
        s,
        idx,
        r: q(&(0..o * g)
            .map(|_| rng.uniform(0.5, 1.5))
            .collect::<Vec<f64>>()),
        c: vec![1_i64 << 32; n],
        d: vec![1_i64 << 32; n],
    }
}

fn rand_mat(rows: usize, cols: usize, rng: &mut Lcg) -> Vec<f64> {
    (0..rows * cols).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

/// The two-plane arm with the options every test but the refusal and block-subset cases uses:
/// `g_row = None`, `row_chunk = 256`, all blocks.
fn joint(
    ws: &[i64],
    hl: &[i64],
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    o: usize,
    n: usize,
    sweeps: usize,
    ridge: i64,
) -> (Plane, Plane, JointStats) {
    e8_joint_icm_2plane(ws, hl, p1, p2, o, n, sweeps, ridge, None, 256, None).expect("joint")
}

/// FNV-1a over words' little-endian bytes.
fn fnv_words_witness(d: &mut u64, xs: &[i64]) {
    for &v in xs {
        for b in v.to_le_bytes() {
            *d ^= u64::from(b);
            *d = d.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// The shape of the bit-identity witness: `o = 64` rows, `n = 512` columns (4 groups, 64 blocks).
const WITNESS_SHAPE: (usize, usize) = (64, 512);

/// The three fixed LCG fixtures of the bit-identity witness.
const WITNESS_SEEDS: [u64; 3] = [0x1C_5EED, 0x2C_5EED, 0x3C_5EED];

/// `(ws, hl, p1, p2)` for one witness fixture: `rho = 0.37` metric, two random planes, a random
/// target.
fn witness_case(seed: u64) -> (Vec<i64>, Vec<i64>, Plane, Plane) {
    let (o, n) = WITNESS_SHAPE;
    let mut rng = Lcg(seed);
    let hl = q(&toeplitz(n, 0.37));
    let p1 = rand_plane(o, n, &mut rng);
    let p2 = rand_plane(o, n, &mut rng);
    let ws = q(&rand_mat(o, n, &mut rng));
    (ws, hl, p1, p2)
}

/// The production `down` fixture: `(o, n) = (896, 4864)`, the same `rho = 0.37` metric.
const DOWN_SHAPE: (usize, usize) = (896, 4864);

/// One seeded `(896, 4864)` fixture: `(ws, hl, p1)`.
fn down_case(seed: u64) -> (Vec<i64>, Vec<i64>, Plane) {
    let (o, n) = DOWN_SHAPE;
    let mut rng = Lcg(seed);
    let hl = q(&toeplitz(n, 0.37));
    let p1 = rand_plane(o, n, &mut rng);
    (q(&rand_mat(o, n, &mut rng)), hl, p1)
}

/// FNV-1a over a plane's code words then its code indices.
fn fnv_plane(p: &Plane) -> u64 {
    let mut d = 0xcbf2_9ce4_8422_2325u64;
    fnv_words_witness(&mut d, &p.s);
    for &v in &p.idx {
        for b in v.to_le_bytes() {
            d ^= u64::from(b);
            d = d.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    d
}

/// `J` of a returned step: the module's own objective at the plane's real weights.
fn j_of(ws: &[i64], hl: &[i64], p: &Plane, o: usize, n: usize) -> i64 {
    let w = plane_w(pref(p), o, n).expect("plane_w");
    j_rows(ws, &w, hl, o, n, None).expect("j_rows")
}

// ── the base arm at `84f2d8d3` ────────────────────────────────────────────────────────────────
//
// The base arm is the per-block residual of `icm.rs` at `84f2d8d3`
// (`nt_q(&chg[..moved·BLOCK], moved, &cols, n, BLOCK)` once per block per row band). Its own source
// is not kept reachable in `src/` (§4 obligation 1: no parallel path), so its observations are the
// captured constants below, taken by `LB_CAPTURE` on this file at that revision. Both arms agree bit
// for bit on all four fixtures, so every disagreement list below is empty.
/// The base arm's `e8_icm_step` / `e8_joint_icm_2plane` outputs at [`WITNESS_SHAPE`], `sweeps = 2`,
/// `ridge = 0`: `(step s+idx digest, joint plane-1 digest, joint plane-2 digest, moves)` per seed.
///
/// The lazy batch reproduces every one of these, the step digest included: the regrouping is exact at
/// the `Q32.32` word on these fixtures (no decision margin is inside the `2^-31`-relative limb
/// difference), so no digest is re-pinned. `e8_joint_icm_2plane` has its own sweep, which the
/// regrouping does not reach.
const WITNESS: [(u64, u64, u64, usize); 3] = [
    (
        0xc55a_bd47_15e7_9ff3,
        0x4d9d_1538_fe32_cfd1,
        0xb98f_ec11_3eda_8076,
        5811,
    ),
    (
        0x21a8_1e41_2c18_e424,
        0xc1cb_1938_74e6_a15d,
        0x9164_6950_cb4a_7a0f,
        5832,
    ),
    (
        0x492d_b452_26fb_3436,
        0x4a3c_903e_4a7d_d931,
        0x69ae_0967_a0f0_70af,
        5796,
    ),
];

#[test]
fn learn_fold_icm_sweep_outputs_are_bit_identical() {
    let (o, n) = WITNESS_SHAPE;
    let mut seen = Vec::new();
    for seed in WITNESS_SEEDS {
        let (ws, hl, p1, p2) = witness_case(seed);
        let step = e8_icm_step(&ws, &hl, pref(&p1), o, n, 2, 0).expect("step");
        let (j1, j2, st) = joint(&ws, &hl, pref(&p1), pref(&p2), o, n, 2, 0);
        let got = (fnv_plane(&step), fnv_plane(&j1), fnv_plane(&j2), st.moves);
        println!("witness seed {seed:#x}: {got:#x?}");
        seen.push(got);
    }
    assert_eq!(seen, WITNESS, "sweep outputs drifted");
}
/// The base arm's `J` and plane digest (`s` then `idx`) after
/// `e8_icm_step(..., sweeps = 2, ridge = 0)`, per witness fixture. The digest is the code-agreement
/// witness: it is equal exactly when every one of the `o·n/BLOCK` block codes is.
const BASE_WITNESS: [(i64, u64); 3] = [
    (52_156_689_818_262, 0xc55a_bd47_15e7_9ff3),
    (53_721_755_482_564, 0x21a8_1e41_2c18_e424),
    (50_206_656_982_787, 0x492d_b452_26fb_3436),
];

/// The base arm's `J` at the `(896, 4864)` fixture, `sweeps = 2`.
const BASE_DOWN_J: i64 = 6_998_698_622_758_382;

/// The base arm's plane digest at the `(896, 4864)` fixture, `sweeps = 2`.
const BASE_DOWN_DIGEST: u64 = 0x0a83_7c2c_e4f2_04de;

/// One lazy-batch witness row: `J` never worse than the base arm by more than one part in `1e6`, and
/// the plane digest equal to it, i.e. **every** block code agrees (agreement `1.00000`, not `>= 0.99`
/// — the bar the brief sets is the weaker one, and it is met with equality by construction of the
/// regrouped sum). A moved decision moves the digest and names itself here.
fn witness_arm(tag: &str, ws: &[i64], hl: &[i64], p: &Plane, o: usize, n: usize, base: (i64, u64)) {
    let (base_j, base_digest) = base;
    let step = e8_icm_step(ws, hl, pref(p), o, n, 2, 0).expect("step");
    let j_new = j_of(ws, hl, &step, o, n);
    let digest = fnv_plane(&step);
    println!(
        "lazy batch {tag}: J {j_new} vs base {base_j} ({:+.3e} relative), codes agree {}/{} = \
         {:.5}",
        (j_new as f64 - base_j as f64) / base_j as f64,
        usize::from(digest == base_digest) * step.idx.len(),
        step.idx.len(),
        f64::from(u8::from(digest == base_digest)),
    );
    assert!(
        j_new as f64 <= base_j as f64 * (1.0 + 1e-6),
        "{tag}: J {j_new} exceeds the base arm {base_j} by more than 1e-6 relative"
    );
    assert_eq!(
        digest, base_digest,
        "{tag}: the block codes differ from the base arm's — the in-batch prefix correction is the \
         term that holds them"
    );
}

#[test]
fn learn_fold_icm_lazy_batch_matches_base() {
    let (o, n) = WITNESS_SHAPE;
    for (i, seed) in WITNESS_SEEDS.iter().enumerate() {
        let (ws, hl, p1, _) = witness_case(*seed);
        witness_arm(
            &format!("fixture {i}"),
            &ws,
            &hl,
            &p1,
            o,
            n,
            BASE_WITNESS[i],
        );
    }
}

/// The `(896, 4864)` witness: `J` and the plane digest at the production `down` shape, two sweeps.
#[test]
fn learn_fold_icm_lazy_batch_down_shape() {
    let (o, n) = DOWN_SHAPE;
    let (ws, hl, p1) = down_case(8);
    witness_arm("down", &ws, &hl, &p1, o, n, (BASE_DOWN_J, BASE_DOWN_DIGEST));
}

/// The capture arm: the base observations the constants above hold. `LB_CAPTURE=1` prints them and
/// returns, so the constants are never guessed.
#[test]
#[ignore = "capture of the base arm's observations"]
fn icm_lazy_batch_capture() {
    if std::env::var("LB_CAPTURE").is_err() {
        return;
    }
    let (o, n) = WITNESS_SHAPE;
    for seed in WITNESS_SEEDS {
        let (ws, hl, p1, _) = witness_case(seed);
        let step = e8_icm_step(&ws, &hl, pref(&p1), o, n, 2, 0).expect("step");
        println!(
            "CAPTURE witness {seed:#x} digest={:#018x} j={} codes={}",
            fnv_plane(&step),
            j_of(&ws, &hl, &step, o, n),
            hex_codes(&step.idx)
        );
    }
    let (o, n) = DOWN_SHAPE;
    let (ws, hl, p1) = down_case(8);
    let step = e8_icm_step(&ws, &hl, pref(&p1), o, n, 2, 0).expect("step");
    println!(
        "CAPTURE down digest={:#018x} j={} codes={}",
        fnv_plane(&step),
        j_of(&ws, &hl, &step, o, n),
        hex_codes(&step.idx)
    );
}

/// A plane's block codes as two hex digits each, `u16` low byte (every code is `< 256`).
fn hex_codes(idx: &[u16]) -> String {
    const HEX: [u8; 16] = *b"0123456789abcdef";
    let mut s = String::with_capacity(idx.len() * 2);
    for &v in idx {
        s.push(HEX[usize::from(v >> 4)] as char);
        s.push(HEX[usize::from(v & 0xf)] as char);
    }
    s
}

/// The ICM sweep microbench, release only:
/// `cargo test --release --locked --test learn_fold_icm_speed -- --ignored --nocapture icm_lazy_batch_microbench`.
///
/// One `e8_icm_step` sweep is measured as `t(sweeps = 1) − t(sweeps = 0)`: the setup is the same
/// call in both and cancels. Three paired reps per shape; the mean and the observed `[min, max]`.
#[test]
#[ignore = "release-mode microbench of the ICM sweep"]
fn icm_lazy_batch_microbench() {
    use std::hint::black_box;
    use std::time::Instant;
    let report = |name: &str, o: usize, n: usize, xs: &[f64]| {
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let min = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let max = xs.iter().copied().fold(0.0, f64::max);
        println!(
            "icm sweep {name} o={o} n={n}: {mean:.3} s [{min:.3}, {max:.3}] over {} paired runs",
            xs.len()
        );
    };
    for (o, n, seed) in [(64usize, 512usize, 0u64), (896, 896, 1), (896, 4864, 8)] {
        let (ws, hl, p1) = if (o, n) == WITNESS_SHAPE {
            let (ws, hl, p1, _) = witness_case(WITNESS_SEEDS[0]);
            (ws, hl, p1)
        } else if (o, n) == DOWN_SHAPE {
            down_case(seed)
        } else {
            let mut rng = Lcg(seed);
            let hl = q(&toeplitz(n, 0.37));
            let p1 = rand_plane(o, n, &mut rng);
            (q(&rand_mat(o, n, &mut rng)), hl, p1)
        };
        let mut sweeps = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            black_box(e8_icm_step(&ws, &hl, pref(&p1), o, n, 0, 0).expect("step"));
            let setup = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            black_box(e8_icm_step(&ws, &hl, pref(&p1), o, n, 1, 0).expect("step"));
            let one = t1.elapsed().as_secs_f64();
            sweeps.push(one - setup);
        }
        report("step", o, n, &sweeps);
    }
}
