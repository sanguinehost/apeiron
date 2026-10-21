//! `tests/learn_fold_gptq_e8_beam.rs` — the beam-quality property of `gptq_e8`, split from
//! `tests/learn_fold_gptq_e8.rs` at its 1,500-line cap: averaged over five correlated Hessians, the
//! whitened total error at beams 4 and 8 falls below beam 1. The error is recomputed from the
//! returned codes and scales against an independent `f64` replay of the feedback. Grade `tested`.
//!
//! The second subject is the separable 256-way cost scan of `beam_row` (`table_scan`): the pre-change
//! 256×8 multiply-square body lives **here only**, as [`base_scan`], and is run against the production
//! arm through the same row body (`beam_pass_probe`) and against the same `(G, H)` vectors
//! ([`scan_arms_are_identical`]). `gptq_e8_output_matches_the_pinned_base_digest` pins the whole
//! `gptq_e8` output of the pre-change build on four fixtures, so any divergence in the selection,
//! the feedback or the `wscale` pick is a mismatch, not a tolerance.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    reason = "the replay mirrors the row-major indices, single-letter names and Q32.32 casts of the ported kernel"
)]

use apeiron::engine::ONE;
use apeiron::learn::code::e8;
use apeiron::learn::fold::fit::gptq_e8::{
    beam_pass_probe, feedback_u, gptq_e8, table_scan, BlockScan, GptqE8, UFac, BLOCK, GROUP,
};
use apeiron::learn::solve::q32::{from_grid, to_grid, Fx, GRID_BOUND};
use std::sync::OnceLock;

/// The Q32.32 word of `x`: `round(x·2³²)`. `O(1)`.
fn word(x: f64) -> i64 {
    to_grid(x, GRID_BOUND).expect("value inside the Q32.32 grid range")
}

/// The words of a real buffer, the boundary conversion the callers of the flipped API apply. `O(len)`.
fn words(x: &[f64]) -> Vec<i64> {
    x.iter().map(|&v| word(v)).collect()
}

/// The `f64` image of a word buffer: `x = w·2⁻³²`. `O(len)`.
fn reals(x: &[i64]) -> Vec<f64> {
    x.iter().map(|&v| from_grid(v)).collect()
}

/// The `f64` view of a [`UFac`]: `U_ij = (hat_ij / 2^32)·2^-d_j` above the diagonal, `0` below.
fn ufac_to_f64(uf: &UFac) -> Vec<f64> {
    let n = uf.n;
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            out[i * n + j] = uf.hat[i * n + j] as f64 / ONE as f64 * 2f64.powi(-uf.d[j]);
        }
    }
    out
}

/// xorshift64: deterministic, no crate.
fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Deterministic uniform in `[−1, 1)`.
fn uniform(s: &mut u64) -> f64 {
    (xorshift(s) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// A deterministic uniform `o × n` weight.
fn random_w(o: usize, n: usize, seed: u64) -> Vec<f64> {
    let mut st = seed;
    (0..o * n).map(|_| uniform(&mut st)).collect()
}

/// `H = (1/t)·XᵀX + 0.25·I` for a deterministic uniform `X` (`t = n`) with a per-8-block column
/// gain, so the 8-block diagonal traces are distinct and no act-order tie arises.
fn correlated_h(n: usize, seed: u64) -> Vec<f64> {
    let t = n;
    let mut st = seed;
    let gain: Vec<f64> = (0..n).map(|j| 1.0 + (j / BLOCK) as f64 * 0.05).collect();
    let x: Vec<f64> = (0..t * n).map(|_| uniform(&mut st)).collect();
    let mut h = vec![0.0f64; n * n];
    for r in 0..t {
        for i in 0..n {
            let xi = x[r * n + i] * gain[i];
            for j in i..n {
                h[i * n + j] += xi * x[r * n + j] * gain[j];
            }
        }
    }
    let s = 1.0 / t as f64;
    for i in 0..n {
        for j in i..n {
            let v = h[i * n + j] * s;
            h[i * n + j] = v;
            h[j * n + i] = v;
        }
        h[i * n + i] += 0.25;
    }
    h
}

/// The four fixtures of the exactness witness: three seeded `(64, 512)` and one seeded `(896, 896)`.
const FIXTURES: [(usize, usize, u64); 4] = [
    (64, 512, 0x0BEA_0001),
    (64, 512, 0x0BEA_0002),
    (64, 512, 0x0BEA_0003),
    (896, 896, 0x0BEA_0004),
];

/// The pre-change codebook table: `e8::codeword(k)` for the 256 rows, built once. The pre-change
/// `beam_pass` built exactly this at `gptq_e8.rs:884-887` and passed it to `beam_row`.
fn pre_change_cw() -> &'static [[i64; BLOCK]; e8::CODEBOOK_LEN] {
    static CW: OnceLock<[[i64; BLOCK]; e8::CODEBOOK_LEN]> = OnceLock::new();
    CW.get_or_init(|| {
        let mut cw = [[0i64; BLOCK]; e8::CODEBOOK_LEN];
        for (k, row) in cw.iter_mut().enumerate() {
            *row = e8::codeword(k).expect("code index below 256");
        }
        cw
    })
}

/// The seven coordinate values of the codebook, as the witness reads them off [`pre_change_cw`]:
/// `−√8, −2, −1, 0, 1, 2, √8` in Q32.32 words, ascending. Written here independently of the kernel.
const SEVEN: [i64; 7] = [
    -e8::SQRT8_Q32,
    -2 * ONE,
    -ONE,
    0,
    ONE,
    2 * ONE,
    e8::SQRT8_Q32,
];

/// The `(t, v)` index of every codeword coordinate, from [`SEVEN`] and [`pre_change_cw`].
fn seven_index() -> [[u8; BLOCK]; e8::CODEBOOK_LEN] {
    let cw = pre_change_cw();
    let mut vi = [[0u8; BLOCK]; e8::CODEBOOK_LEN];
    for k in 0..e8::CODEBOOK_LEN {
        for t in 0..BLOCK {
            let v = SEVEN
                .iter()
                .position(|&x| x == cw[k][t])
                .unwrap_or_else(|| {
                    panic!(
                        "row {k} coordinate {t} value {} is not one of SEVEN",
                        cw[k][t]
                    )
                });
            vi[k][t] = v as u8;
        }
    }
    vi
}

/// **The pre-change 256-way scan, at `3b4758f7` `gptq_e8.rs:596-616`**, kept callable only here: the
/// squared whitened residual of every codeword, `Σ_t (cw[k][t]·G_t − H_t)²` in checked `i128` from a
/// `0` start over `t` ascending. The third argument is the separable index table the production arm
/// needs; this body ignores it and reads the codebook words, as the pre-change body did.
fn base_scan(
    gvec: &[i128; BLOCK],
    hvec: &[i128; BLOCK],
    _vi: &[[u8; BLOCK]; e8::CODEBOOK_LEN],
) -> Option<[i128; e8::CODEBOOK_LEN]> {
    let cw = pre_change_cw();
    let mut out = [0i128; e8::CODEBOOK_LEN];
    for (k, row) in cw.iter().enumerate() {
        let mut c: i128 = 0;
        for t in 0..BLOCK {
            let e = i128::from(row[t])
                .checked_mul(gvec[t])?
                .checked_sub(hvec[t])?;
            c = c.checked_add(e.checked_mul(e)?)?;
        }
        out[k] = c;
    }
    Some(out)
}

/// A 64-bit FNV-1a digest of a `gptq_e8` output: the codeword words, the group-scale words and the
/// codes, in that order.
fn digest(r: &GptqE8) -> u64 {
    let mut d: u64 = 0xcbf2_9ce4_8422_2325;
    let mut put = |b: u8| {
        d ^= u64::from(b);
        d = d.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for &v in r.s.iter().chain(r.scale.iter()) {
        for b in v.to_le_bytes() {
            put(b);
        }
    }
    for &v in &r.idx {
        for b in v.to_le_bytes() {
            put(b);
        }
    }
    d
}

/// The `gptq_e8` output digest of every fixture, beam and `wscale` setting, taken from the
/// **pre-change** build at `3b4758f7` with `act_order = true` and `damp = 0.01`: `(o, n, seed, beam,
/// wscale, digest)`. The fixtures and the Hessian seeds are those of [`FIXTURES`]; the weight and
/// the metric are the deterministic ones [`random_w`] and [`correlated_h`] build.
const BASE_DIGESTS: [(usize, usize, u64, usize, bool, u64); 16] = [
    (64, 512, 0x0BEA_0001, 1, false, 0x3322_06d4_82a3_1ee7),
    (64, 512, 0x0BEA_0001, 1, true, 0x7633_ee9b_d8df_76dc),
    (64, 512, 0x0BEA_0001, 4, false, 0xe4d6_475a_a354_d76a),
    (64, 512, 0x0BEA_0001, 4, true, 0xb3c6_31bb_c382_1bfa),
    (64, 512, 0x0BEA_0002, 1, false, 0x2736_a13a_7714_ec91),
    (64, 512, 0x0BEA_0002, 1, true, 0x6eca_f676_b017_bcf7),
    (64, 512, 0x0BEA_0002, 4, false, 0x8271_7cfc_9270_bbad),
    (64, 512, 0x0BEA_0002, 4, true, 0x99f0_a4dc_0b66_0b85),
    (64, 512, 0x0BEA_0003, 1, false, 0xc1d3_35f6_5eb4_bd1a),
    (64, 512, 0x0BEA_0003, 1, true, 0x9318_d4cf_062a_a9b4),
    (64, 512, 0x0BEA_0003, 4, false, 0xdbd5_4dca_eb8d_893b),
    (64, 512, 0x0BEA_0003, 4, true, 0xe0c1_6fa3_ebf6_7174),
    (896, 896, 0x0BEA_0004, 1, false, 0xffd8_cc96_0c74_198f),
    (896, 896, 0x0BEA_0004, 1, true, 0x36d7_30bd_8834_dd80),
    (896, 896, 0x0BEA_0004, 4, false, 0xd16a_837d_6a00_2b93),
    (896, 896, 0x0BEA_0004, 4, true, 0xc290_d542_21e3_8a7f),
];

/// The whitened total error of a `gptq_e8` result, recomputed from the returned codes, signs and
/// scales: for each group, `Err = (w0 − a·S)·U_blk⁻¹` block by block, accumulated as `Σ |err_block|²`
/// over every row and block, with the same intra- and inter-group feedback the kernel applies. This
/// equals the kernel's `cum_err2` at the winning beam (`step_err2 = |Z_w − a·Z_c|²`).
fn whitened_total_error(r: &GptqE8, w: &[f64], h: &[f64], o: usize, n: usize, damp: f64) -> f64 {
    let u = ufac_to_f64(&feedback_u(&words(h), n, word(damp)).expect("SPD factor"));
    let ng = n.div_ceil(GROUP);
    let nb = n / BLOCK;
    let mut wk = w.to_vec();
    let mut tot = 0.0f64;
    for gi in 0..ng {
        let g0 = gi * GROUP;
        let g1 = (g0 + GROUP).min(n);
        let gs = g1 - g0;
        let mut err_all = vec![0.0f64; o * gs];
        let scale = reals(&r.scale);
        for row in 0..o {
            let a = scale[row * ng + gi];
            let mut blk = 0;
            while blk < gs / BLOCK {
                let b0 = blk * BLOCK;
                let b1 = b0 + BLOCK;
                let mut zw = [0.0f64; BLOCK];
                for t in 0..BLOCK {
                    zw[t] = wk[row * n + g0 + b0 + t];
                }
                let mut j = BLOCK;
                while j > 0 {
                    j -= 1;
                    let mut s = zw[j];
                    for k in (j + 1)..BLOCK {
                        s -= zw[k] * u[(g0 + b0 + k) * n + g0 + b0 + j];
                    }
                    zw[j] = s / u[(g0 + b0 + j) * n + g0 + b0 + j];
                }
                let c = usize::from(r.idx[row * nb + (g0 + b0) / BLOCK]);
                let cw = e8::codeword(c).expect("code index");
                let mut err = [0.0f64; BLOCK];
                for t in 0..BLOCK {
                    err[t] = zw[t] - a * (cw[t] as f64 / ONE as f64);
                    err_all[row * gs + b0 + t] = err[t];
                    tot += err[t] * err[t];
                }
                let mut cc = b1;
                while cc < gs {
                    let mut acc = 0.0f64;
                    for t in 0..BLOCK {
                        acc += err[t] * u[(g0 + b0 + t) * n + g0 + cc];
                    }
                    wk[row * n + g0 + cc] -= acc;
                    cc += 1;
                }
                blk += 1;
            }
        }
        if g1 < n {
            for row in 0..o {
                let mut m = g1;
                while m < n {
                    let mut acc = 0.0f64;
                    for j in 0..gs {
                        acc += err_all[row * gs + j] * u[(g0 + j) * n + m];
                    }
                    wk[row * n + m] -= acc;
                    m += 1;
                }
            }
        }
    }
    tot
}

/// The first differing entry of two `Fx` rows, as `(index, left, right)`, or `None` when equal.
fn first_fx_diff(a: &[Fx], b: &[Fx]) -> Option<(usize, Fx, Fx)> {
    a.iter()
        .zip(b.iter())
        .enumerate()
        .find(|(_, (x, y))| x != y)
        .map(|(i, (x, y))| (i, x.clone(), y.clone()))
}

#[test]
fn codeword_coordinates_are_the_seven_values() {
    let vi = seven_index();
    for k in 0..e8::CODEBOOK_LEN {
        for t in 0..BLOCK {
            assert!(
                usize::from(vi[k][t]) < SEVEN.len(),
                "row {k} coordinate {t} index {} out of range",
                vi[k][t]
            );
        }
    }
    // Only the O8 rows carry ±√8: all 16 of them do, and no E8 row does.
    let neg = 0u8;
    let pos = (SEVEN.len() - 1) as u8;
    for k in 0..e8::CODEBOOK_LEN {
        let has = vi[k].contains(&neg) || vi[k].contains(&pos);
        let want = k >= 240;
        assert_eq!(has, want, "row {k}: ±√8 present = {has}, expected {want}");
    }
}

/// The two scans return the same `i128` for every codeword on every input: seeds across the
/// magnitude classes the block carrier produces (`gvec = 0`, small `H`, and both up to the `ALIGN`
/// ceiling `|cw·G|, |H| < 2^58` where the eight squares stay below `2^121`), plus inputs past
/// `i128` where both must refuse.
#[test]
fn scan_arms_are_identical() {
    let vi = seven_index();
    let mut st = 0x5EED_5CA1u64;
    for case in 0..400usize {
        let mut gvec = [0i128; BLOCK];
        let mut hvec = [0i128; BLOCK];
        for t in 0..BLOCK {
            let cls = usize::try_from(xorshift(&mut st) % 6).expect("below 6");
            let mag: i128 = match cls {
                0 => 0,
                1 => 1,
                2 => 3,
                3 => 1 << 10,
                4 => 1 << 20,
                _ => 1 << 23,
            };
            let sg = if xorshift(&mut st) & 1 == 0 { 1 } else { -1 };
            let sh = if xorshift(&mut st) & 1 == 0 { 1 } else { -1 };
            gvec[t] = sg * mag;
            hvec[t] = sh * mag * (1 + i128::from(xorshift(&mut st) % 5));
        }
        if case == 0 {
            gvec = [0i128; BLOCK];
        }
        let base = base_scan(&gvec, &hvec, &vi);
        let table = table_scan(&gvec, &hvec, &vi);
        assert_eq!(
            base.is_some(),
            table.is_some(),
            "case {case}: refusal differs, gvec {gvec:?} hvec {hvec:?}"
        );
        let (Some(b), Some(tb)) = (base, table) else {
            panic!("case {case}: an in-domain input was refused");
        };
        for k in 0..e8::CODEBOOK_LEN {
            assert_eq!(
                b[k], tb[k],
                "case {case}: cost mismatch at k = {k} ({} vs {}); gvec {gvec:?} hvec {hvec:?}",
                b[k], tb[k]
            );
        }
    }
    // Past `i128` both refuse: the same overflow, from the same products.
    let big = [i128::MAX / 2; BLOCK];
    let zero = [0i128; BLOCK];
    assert!(
        base_scan(&big, &zero, &vi).is_none(),
        "base accepted 2^126 squares"
    );
    assert!(
        table_scan(&big, &zero, &vi).is_none(),
        "table accepted 2^126 squares"
    );
    // And one class above the block bound but inside `i128`: 8·(2^40·2^34)² = 2^151 leaves `i128`,
    // so both refuse for the same term.
    let big = [1i128 << 40; BLOCK];
    assert!(
        base_scan(&big, &zero, &vi).is_none(),
        "base accepted 2^148 squares"
    );
    assert!(
        table_scan(&big, &zero, &vi).is_none(),
        "table accepted 2^148 squares"
    );
}

/// The two arms select the same codes and produce the same scales, codeword images and residuals on
/// every row of the four fixtures at `beam ∈ {1, 4}` and `wscale ∈ {false, true}`. Two full beams
/// through `beam_pass_probe` differ only in the scan body, so this covers the `select_k` order, the
/// rebuild (which carries `cum`), the feedback and the `wscale` pick.
#[test]
fn beam_pass_arms_agree() {
    let damp = word(0.01);
    for (o, n, seed) in FIXTURES {
        let w = words(&random_w(o, n, seed));
        let h = correlated_h(n, seed ^ 0x5A5A_5A5A);
        let u = feedback_u(&words(&h), n, damp).expect("SPD factor");
        for beam in [1usize, 4] {
            for wscale in [false, true] {
                let run = |scan: BlockScan| {
                    beam_pass_probe(&u, n, 0, GROUP, &w, o, beam, wscale, scan).expect("beam pass")
                };
                let (a1, s1, e1, i1) = run(base_scan);
                let (a2, s2, e2, i2) = run(table_scan);
                assert!(
                    i1 == i2,
                    "o={o} n={n} seed={seed:#x} beam={beam} wscale={wscale}: codes differ"
                );
                assert!(
                    s1 == s2,
                    "o={o} n={n} seed={seed:#x} beam={beam} wscale={wscale}: codeword images differ"
                );
                if let Some((idx, x, y)) = first_fx_diff(&e1, &e2) {
                    panic!(
                        "o={o} n={n} seed={seed:#x} beam={beam} wscale={wscale}: residual {idx} differs ({x:?} vs {y:?})"
                    );
                }
                if let Some((idx, x, y)) = first_fx_diff(&a1, &a2) {
                    panic!(
                        "o={o} n={n} seed={seed:#x} beam={beam} wscale={wscale}: scale {idx} differs ({x:?} vs {y:?})"
                    );
                }
            }
        }
    }
}

/// Every `gptq_e8` output on the four fixtures is bit-identical to the pre-change build: the codes,
/// the codeword words and the group scales hash to [`BASE_DIGESTS`]. `beam = 4` is the production
/// arm and runs the scan this lane changes.
#[test]
fn gptq_e8_output_matches_the_pinned_base_digest() {
    let damp = word(0.01);
    for (o, n, seed, beam, wscale, want) in BASE_DIGESTS {
        let w = words(&random_w(o, n, seed));
        let h = words(&correlated_h(n, seed ^ 0x5A5A_5A5A));
        let r = gptq_e8(&w, &h, o, n, damp, true, wscale, beam).expect("gptq_e8");
        assert_eq!(
            digest(&r),
            want,
            "o={o} n={n} seed={seed:#x} beam={beam} wscale={wscale}: output differs from the \
             pre-change build"
        );
    }
}

#[test]
fn beam_mean_objective_falls_below_beam_one() {
    let (o, n, damp) = (3usize, 256usize, 0.01);
    let beams = [1usize, 4, 8];
    let mut mean = [0.0f64; 3];
    let seeds = [
        0x0BAD_0001u64,
        0x0BAD_0002,
        0x0BAD_0003,
        0x0BAD_0004,
        0x0BAD_0005,
    ];
    for seed in seeds {
        let w = random_w(o, n, seed);
        let h = correlated_h(n, seed ^ 0x5A5A_5A5A);
        for (i, &beam) in beams.iter().enumerate() {
            let r = gptq_e8(&words(&w), &words(&h), o, n, word(damp), false, false, beam)
                .expect("gptq_e8");
            mean[i] += whitened_total_error(&r, &w, &h, o, n, damp) / 5.0;
        }
    }
    println!("beam mean objective (1,4,8): {mean:?}");
    assert!(
        mean[1] < mean[0],
        "beam 4 mean {} >= beam 1 mean {}",
        mean[1],
        mean[0]
    );
    assert!(
        mean[2] < mean[0],
        "beam 8 mean {} >= beam 1 mean {}",
        mean[2],
        mean[0]
    );
}

/// The interleaved speed arm, `--ignored`: `beam_pass` at `(896, 896)` and `(128, 896)` through
/// `beam_pass_probe`, one production group (`g0 = 0`, `gs = GROUP`), `beam = 4`, the pre-change scan
/// and the separable scan alternating inside one process. Nine runs per arm; the mean and the
/// `[min, max]` are printed, then the same two arms through the whole `gptq_e8` at `(896, 896)`
/// (`beam = 4`, `wscale = true`, the production settings): its wall carries `feedback_u` and the
/// rest of the call, which the scan does not reach.
#[test]
#[ignore = "timing, not a gate"]
fn bench_beam_pass_base_vs_table() {
    let damp = word(0.01);
    let runs = 9usize;
    for (o, n) in [(896usize, 896usize), (128usize, 896usize)] {
        let seed = 0x0BEA_0004 ^ (o as u64);
        let w = words(&random_w(o, n, seed));
        let h = correlated_h(n, seed ^ 0x5A5A_5A5A);
        let u = feedback_u(&words(&h), n, damp).expect("SPD factor");
        for wscale in [false, true] {
            let mut base = Vec::with_capacity(runs);
            let mut table = Vec::with_capacity(runs);
            for _ in 0..runs {
                for (arm, scan) in [
                    ("base", base_scan as BlockScan),
                    ("table", table_scan as BlockScan),
                ] {
                    let t0 = std::time::Instant::now();
                    let _ = beam_pass_probe(&u, n, 0, GROUP, &w, o, 4, wscale, scan)
                        .expect("beam pass");
                    let dt = t0.elapsed().as_secs_f64();
                    if arm == "base" {
                        base.push(dt);
                    } else {
                        table.push(dt);
                    }
                }
            }
            let stat = |v: &[f64]| {
                let mean = v.iter().sum::<f64>() / v.len() as f64;
                let min = v.iter().copied().fold(f64::MAX, f64::min);
                let max = v.iter().copied().fold(0.0f64, f64::max);
                (mean, min, max)
            };
            let (mb, lob, hib) = stat(&base);
            let (mt, lot, hit) = stat(&table);
            println!(
                "beam_pass o={o} n={n} wscale={wscale}: base {mb:.4} s [{lob:.4}, {hib:.4}] n={runs}; \
                 table {mt:.4} s [{lot:.4}, {hit:.4}] n={runs}; ratio {:.2}x",
                mb / mt
            );
        }
    }
    // The whole `gptq_e8` at (896, 896) under the production settings.
    let (o, n, seed) = (896usize, 896usize, 0x0BEA_0004u64);
    let w = words(&random_w(o, n, seed));
    let h = words(&correlated_h(n, seed ^ 0x5A5A_5A5A));
    let mut wall = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = std::time::Instant::now();
        let _ = gptq_e8(&w, &h, o, n, damp, true, true, 4).expect("gptq_e8");
        wall.push(t0.elapsed().as_secs_f64());
    }
    let mean = wall.iter().sum::<f64>() / wall.len() as f64;
    let min = wall.iter().copied().fold(f64::MAX, f64::min);
    let max = wall.iter().copied().fold(0.0f64, f64::max);
    println!("gptq_e8 o={o} n={n} beam=4 wscale=true: {mean:.4} s [{min:.4}, {max:.4}] n={runs}");
}
