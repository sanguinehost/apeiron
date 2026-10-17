//! `tests/learn_fold_icm_speed.rs` — the bit-identity witness of the `E8 ∪ O8` ICM sweep
//! (`e8_icm_step`, `e8_joint_icm_2plane`) at `o = 64`, `n = 512` on three fixed LCG fixtures, and the
//! release-mode microbench of the sweep's unit cost. Grade `tested`.

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
    clippy::unreadable_literal,
    reason = "the LCG maps integers to `f64` by division, the flat row-major index arithmetic is the reference payload format, the representability and argmin checks are exact float comparisons, and the test helpers mirror the ported option surface"
)]

use apeiron::learn::code::e8::codeword;
use apeiron::learn::fold::fit::icm::{
    e8_icm_step, e8_joint_icm_2plane, JointStats, Plane, PlaneRef,
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

/// The recorded `e8_icm_step` / `e8_joint_icm_2plane` outputs at [`WITNESS_SHAPE`], `sweeps = 2`,
/// `ridge = 0`: `(step s+idx digest, joint plane-1 digest, joint plane-2 digest, moves)` per seed.
/// The optimization that hoists the row scale out of the residual update is bit-identical when
/// these reproduce exactly.
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

/// The ICM sweep microbench, release only:
/// `cargo test --release --locked --test learn_fold_icm_speed -- --ignored --nocapture icm_sweep_microbench`.
///
/// `sweeps = 0` isolates the setup (metric, three products, the quad tensors); the difference to
/// `sweeps = 9` is nine sweeps, so `ns / (row, block)` is the sweep's unit cost at `o = 64`,
/// `n = 512` (4096 row-blocks per sweep). Paired three-run means with the observed spread.
#[test]
#[ignore = "release-mode microbench of the ICM sweep"]
fn icm_sweep_microbench() {
    use std::hint::black_box;
    use std::time::Instant;
    // The sweep is measured as `t(sweeps = k) − t(sweeps = 0)`; the setup is the same call in both
    // and cancels. Production runs `sweeps = 2`; `k = 9` shows the converged regime the change
    // skips. A single run is a sample: three paired runs, mean and observed spread.
    let report = |name: &str, o: usize, n: usize, k: usize, xs: &[f64]| {
        let mean = xs.iter().sum::<f64>() / 3.0;
        let min = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let max = xs.iter().copied().fold(0.0, f64::max);
        println!(
            "icm {name} o={o} n={n} sweeps={k}: {mean:.2} ns/(row,block) [{min:.2}, {max:.2}] \
             over 3 paired runs, sweeps 0 vs {k}"
        );
    };
    // `n = 512` and the residual-dominated `n = 2048`; both pooled over the engine's width.
    for (o, n) in [(64usize, 512usize), (16, 2048)] {
        let (ws, hl, p1, p2) = if (o, n) == WITNESS_SHAPE {
            witness_case(WITNESS_SEEDS[0])
        } else {
            let mut rng = Lcg(WITNESS_SEEDS[0]);
            let hl = q(&toeplitz(n, 0.37));
            let p1 = rand_plane(o, n, &mut rng);
            let p2 = rand_plane(o, n, &mut rng);
            (q(&rand_mat(o, n, &mut rng)), hl, p1, p2)
        };
        let row_blocks = o * (n / BLOCK);
        let step = |sw: usize| {
            let t = Instant::now();
            black_box(e8_icm_step(&ws, &hl, pref(&p1), o, n, sw, 0).expect("step"));
            t.elapsed().as_nanos() as f64
        };
        let joint_run = |sw: usize| {
            let t = Instant::now();
            black_box(joint(&ws, &hl, pref(&p1), pref(&p2), o, n, sw, 0));
            t.elapsed().as_nanos() as f64
        };
        for k in [1usize, 2, 9] {
            let (mut step_ns, mut joint_ns) = (Vec::new(), Vec::new());
            for _ in 0..3 {
                let (s0, sk) = (step(0), step(k));
                step_ns.push((sk - s0) / (k as f64 * row_blocks as f64));
                let (j0, jk) = (joint_run(0), joint_run(k));
                joint_ns.push((jk - j0) / (k as f64 * row_blocks as f64));
            }
            report("step", o, n, k, &step_ns);
            report("joint", o, n, k, &joint_ns);
        }
    }
}
