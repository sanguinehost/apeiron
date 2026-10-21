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
    e8_icm_step, e8_joint_icm_2plane, e8_joint_icm_2plane_annealed, j_rows, plane_w, BetaSchedule,
    JointStats, Plane, PlaneRef,
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

// ── the joint two-plane sweep's lazy batch ────────────────────────────────────────────────────
//
// `joint_band`'s residual is batched by `GROUP` (`128` columns = `16` blocks) in the new form: a
// block's change accumulates in a `[rows, GROUP]` matrix, the block's decision reads the running
// prefix for its own eight columns, and one flush per group applies the whole group's change to
// every out-of-group column. The base arm is the per-block residual of `icm.rs` at `7e3c4dbd`.
//
// The base arm is **not** callable from here: its products run on `engine::kern::limb::gemm_q32`,
// which is `pub(crate)`, and a 5-limb reference body in `src/` is the parallel path §4 obligation 1
// refuses. Its observations are therefore the captured constants below — taken by `JOINT_CAPTURE` on
// this file at `7e3c4dbd` with the same harness — and every arm is judged against them.

/// One arm's observation: both planes' code digests, the pair's `J` word and the move count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct JointArm {
    d1: u64,
    d2: u64,
    j: i64,
    moves: usize,
}

/// One `e8_joint_icm_2plane_annealed` call at the production option surface
/// (`g_row = None`, `row_chunk = 256`, all blocks, `ridge = 0`).
fn joint_arm(
    ws: &[i64],
    hl: &[i64],
    p1: &Plane,
    p2: &Plane,
    o: usize,
    n: usize,
    sweeps: usize,
    schedule: Option<&BetaSchedule>,
) -> JointArm {
    let (a, b, st) = e8_joint_icm_2plane_annealed(
        ws,
        hl,
        pref(p1),
        pref(p2),
        o,
        n,
        sweeps,
        0,
        None,
        256,
        None,
        schedule,
    )
    .expect("joint");
    JointArm {
        d1: fnv_plane(&a),
        d2: fnv_plane(&b),
        j: st.j_after,
        moves: st.moves,
    }
}

/// The joint witness shape and its three seeds.
const JOINT_SHAPE: (usize, usize) = (64, 512);

const JOINT_SEEDS: [u64; 3] = [0x4C_5EED, 0x5C_5EED, 0x6C_5EED];

/// The production `o`-kind joint shape.
const JOINT_O_SHAPE: (usize, usize) = (896, 896);

/// The one annealing schedule of the witness: a two-sweep cooling ramp, seeded.
const JOINT_SEED: u64 = 0x00C0_FFEE;

fn joint_schedule() -> BetaSchedule {
    BetaSchedule::new(1_i64 << 28, 1_i64 << 32, 2, JOINT_SEED).expect("schedule")
}

/// `(ws, hl, p1, p2)` for one joint fixture: the `rho = 0.37` metric, two random planes, a random
/// target.
fn joint_case(seed: u64, o: usize, n: usize) -> (Vec<i64>, Vec<i64>, Plane, Plane) {
    let mut rng = Lcg(seed);
    let hl = q(&toeplitz(n, 0.37));
    let p1 = rand_plane(o, n, &mut rng);
    let p2 = rand_plane(o, n, &mut rng);
    (q(&rand_mat(o, n, &mut rng)), hl, p1, p2)
}

/// The base arm's observations at [`JOINT_SHAPE`], `sweeps = 2`, per seed: the `T = 0` arm and the
/// [`JOINT_SEED`] schedule's arm.
const JOINT_BASE: [(JointArm, JointArm); 3] = [
    (
        JointArm {
            d1: 0xf0cf_a451_13ed_af9d,
            d2: 0x4436_69ae_b687_9813,
            j: 16_709_021_102_793,
            moves: 5770,
        },
        JointArm {
            d1: 0x3a5b_9b51_e6fd_fc9a,
            d2: 0x7f53_6214_a27f_aaa3,
            j: 60_153_568_769_653,
            moves: 8192,
        },
    ),
    (
        JointArm {
            d1: 0x470f_7bec_b982_4fc8,
            d2: 0x6754_f2b1_874a_d69c,
            j: 17_464_437_631_886,
            moves: 5816,
        },
        JointArm {
            d1: 0x51e8_cfb7_f5ae_b75b,
            d2: 0xb35a_7dfe_88e7_b46e,
            j: 59_781_448_909_579,
            moves: 8192,
        },
    ),
    (
        JointArm {
            d1: 0x9807_92d3_0e13_c5e1,
            d2: 0x4d1f_bc77_03c6_6be6,
            j: 17_468_087_684_731,
            moves: 5814,
        },
        JointArm {
            d1: 0xc37f_e8eb_34a4_94c9,
            d2: 0xc510_ee96_56f0_2c11,
            j: 58_917_920_103_505,
            moves: 8191,
        },
    ),
];

/// The base arm's observations at [`JOINT_O_SHAPE`], `sweeps = 2`: `T = 0` and the schedule.
const JOINT_BASE_O: (JointArm, JointArm) = (
    JointArm {
        d1: 0xf0cf_b496_206d_a901,
        d2: 0xe40b_4b14_a503_a493,
        j: 432_176_896_298_905,
        moves: 142_478,
    },
    JointArm {
        d1: 0x21ed_8f20_e227_78ec,
        d2: 0x1ae8_5bf1_c238_4948,
        j: 1_475_866_660_403_872,
        moves: 200_699,
    },
);

/// One witness row: `J` never worse than the base arm by more than one part in `1e6`, and both
/// planes' digests equal to the base arm's — i.e. every block code of both planes agrees.
fn joint_witness(tag: &str, got: JointArm, base: JointArm) {
    joint_report(tag, got, base);
    assert!(
        got.j as f64 <= base.j as f64 * (1.0 + 1e-6),
        "{tag}: J {} exceeds the base arm {} by more than 1e-6 relative",
        got.j,
        base.j
    );
    assert_eq!(
        (got.d1, got.d2, got.moves),
        (base.d1, base.d2, base.moves),
        "{tag}: the block codes or the move count differ from the base arm's — the in-batch \
         prefix correction is the term that holds them"
    );
}

/// The same row without the identity assertion, for the scheduled arm (see
/// [`joint_schedule_witness`]).
fn joint_report(tag: &str, got: JointArm, base: JointArm) {
    println!(
        "joint lazy batch {tag}: J {} vs base {} ({:+.3e} relative), digests {:#018x}/{:#018x} vs \
         {:#018x}/{:#018x}, moves {} vs {}",
        got.j,
        base.j,
        (got.j as f64 - base.j as f64) / base.j as f64,
        got.d1,
        got.d2,
        base.d1,
        base.d2,
        got.moves,
        base.moves,
    );
}

/// The scheduled arm's witness: the same schedule on the same inputs reproduces the same planes
/// decision for decision (the row `moves` count and both digests), and the base arm's observation is
/// reported beside it.
///
/// Identity with the base arm is **not** the criterion here, and cannot be: the scheduled path draws
/// each block's pair from the `CODE_LEN²` Boltzmann table, so a cost table that differs in its last
/// word — which a regrouped `i128` sum does, inside the `2^-32` grid — flips a draw that no margin
/// protects, and one flipped draw changes the whole trajectory. The `T = 0` arm, where the decision
/// is an argmin and therefore has a margin, is the identity witness; this arm's is determinism.
fn joint_schedule_witness(tag: &str, first: JointArm, second: Option<JointArm>, base: JointArm) {
    joint_report(tag, first, base);
    if let Some(second) = second {
        assert_eq!(
            (first.d1, first.d2, first.moves),
            (second.d1, second.d2, second.moves),
            "{tag}: the scheduled arm is not reproducible on the same inputs"
        );
    }
}

#[test]
fn learn_fold_icm_joint_lazy_batch_matches_base() {
    let (o, n) = JOINT_SHAPE;
    let schedule = joint_schedule();
    for (i, seed) in JOINT_SEEDS.iter().enumerate() {
        let (ws, hl, p1, p2) = joint_case(*seed, o, n);
        joint_witness(
            &format!("fixture {i}"),
            joint_arm(&ws, &hl, &p1, &p2, o, n, 2, None),
            JOINT_BASE[i].0,
        );
    }
    for (i, seed) in JOINT_SEEDS.iter().enumerate() {
        let (ws, hl, p1, p2) = joint_case(*seed, o, n);
        let a = joint_arm(&ws, &hl, &p1, &p2, o, n, 2, Some(&schedule));
        let b = joint_arm(&ws, &hl, &p1, &p2, o, n, 2, Some(&schedule));
        joint_schedule_witness(
            &format!("fixture {i} scheduled"),
            a,
            Some(b),
            JOINT_BASE[i].1,
        );
    }
}

/// The `(896, 896)` witness: the production `o`-kind joint shape, two sweeps, both arms.
#[test]
fn learn_fold_icm_joint_lazy_batch_o_shape() {
    let (o, n) = JOINT_O_SHAPE;
    let (ws, hl, p1, p2) = joint_case(9, o, n);
    let schedule = joint_schedule();
    joint_witness(
        "o-shape",
        joint_arm(&ws, &hl, &p1, &p2, o, n, 2, None),
        JOINT_BASE_O.0,
    );
    joint_schedule_witness(
        "o-shape scheduled",
        joint_arm(&ws, &hl, &p1, &p2, o, n, 2, Some(&schedule)),
        None,
        JOINT_BASE_O.1,
    );
}

/// The capture arm: the base observations the constants above hold. `JOINT_CAPTURE=1` prints them
/// and the wall of each call, so neither a constant nor a unit cost is guessed.
#[test]
#[ignore = "capture of the joint base arm's observations"]
fn icm_joint_capture() {
    if std::env::var("JOINT_CAPTURE").is_err() {
        return;
    }
    let arm = |tag: &str, o: usize, n: usize, seed: u64| {
        let (ws, hl, p1, p2) = joint_case(seed, o, n);
        let schedule = joint_schedule();
        for (name, sched) in [("none", None), ("sched", Some(&schedule))] {
            let t0 = std::time::Instant::now();
            let a = joint_arm(&ws, &hl, &p1, &p2, o, n, 2, sched);
            let wall = t0.elapsed().as_secs_f64();
            println!(
                "CAPTURE {tag} {name} seed={seed:#x} d1={:#018x} d2={:#018x} j={} moves={} wall={wall:.3}s",
                a.d1, a.d2, a.j, a.moves
            );
        }
    };
    let (o, n) = JOINT_SHAPE;
    for seed in JOINT_SEEDS {
        arm("witness", o, n, seed);
    }
    let (o, n) = JOINT_O_SHAPE;
    arm("o-shape", o, n, 9);
    // The one-sweep unit cost of the two shapes the microbench measures.
    for (tag, o, n, seed) in [
        ("probe1", 896usize, 896usize, 9u64),
        ("probe128", 128, 896, 11),
    ] {
        let (ws, hl, p1, p2) = joint_case(seed, o, n);
        let t0 = std::time::Instant::now();
        let a = joint_arm(&ws, &hl, &p1, &p2, o, n, 1, None);
        println!(
            "CAPTURE {tag} one sweep o={o} n={n}: {:.3} s, moves {}",
            t0.elapsed().as_secs_f64(),
            a.moves
        );
    }
}

/// The joint sweep microbench: `cargo test --release --test learn_fold_icm_speed -- --ignored
/// --nocapture icm_joint_lazy_batch_microbench`.
///
/// One arm is `t(sweeps = 2) − t(sweeps = 0)` per rep (the setup cancels), so the number is two
/// sweeps; the table reports the per-sweep mean and `[min, max]` over three reps.
#[test]
#[ignore = "release-mode microbench of the joint two-plane sweep"]
fn icm_joint_lazy_batch_microbench() {
    use std::hint::black_box;
    use std::time::Instant;
    for (o, n, seed) in [(896usize, 896usize, 9u64), (128, 896, 11)] {
        let (ws, hl, p1, p2) = joint_case(seed, o, n);
        let mut per_sweep = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            black_box(joint_arm(&ws, &hl, &p1, &p2, o, n, 0, None));
            let setup = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            black_box(joint_arm(&ws, &hl, &p1, &p2, o, n, 2, None));
            let two = t1.elapsed().as_secs_f64();
            per_sweep.push((two - setup) / 2.0);
        }
        let mean = per_sweep.iter().sum::<f64>() / per_sweep.len() as f64;
        let min = per_sweep.iter().copied().fold(f64::INFINITY, f64::min);
        let max = per_sweep.iter().copied().fold(0.0_f64, f64::max);
        println!(
            "joint sweep o={o} n={n}: {mean:.3} s/sweep [{min:.3}, {max:.3}] over {} reps",
            per_sweep.len()
        );
    }
}

/// The pair-search cost of one `(896, 896)` joint sweep, isolated: `896` rows × `112` blocks calls of
/// the public [`best_pair_q`] against the block's own `CODE_LEN²` cross table. Upper bound on the
/// sweep's serial decision phase (the real sweep also builds `av`/`bv` and writes two planes'
/// codes), measured so a regression in `joint_band` can be attributed.
#[test]
#[ignore = "isolated cost of the joint sweep's pair search"]
fn icm_joint_pair_search_probe() {
    use apeiron::learn::fold::fit::icm::best_pair_q;
    use std::hint::black_box;
    use std::time::Instant;
    const O: usize = 896;
    const NB: usize = 112;
    let mut rng = Lcg(0xA11CE);
    let x_all: Vec<i64> = (0..NB * CODE_LEN * CODE_LEN)
        .map(|_| i64::from((rng.next_u64() >> 40) as u32))
        .collect();
    let av = [0_i128; CODE_LEN];
    let bv = [0_i128; CODE_LEN];
    let t0 = Instant::now();
    let mut acc = 0_usize;
    for _ in 0..O {
        for m in 0..NB {
            let xm = &x_all[m * CODE_LEN * CODE_LEN..(m + 1) * CODE_LEN * CODE_LEN];
            acc ^= black_box(best_pair_q(0, &av, xm, &bv).expect("pair"));
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    println!(
        "joint pair search (896, 896) one sweep: {wall:.3} s, {:.3}e9 costs, acc {acc}",
        (O * NB * CODE_LEN * CODE_LEN) as f64 / 1e9
    );
}
