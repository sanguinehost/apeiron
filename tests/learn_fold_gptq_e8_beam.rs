//! `tests/learn_fold_gptq_e8_beam.rs` — the beam-quality property of `gptq_e8`, split from
//! `tests/learn_fold_gptq_e8.rs` at its 1,500-line cap: averaged over five correlated Hessians, the
//! whitened total error at beams 4 and 8 falls below beam 1. The error is recomputed from the
//! returned codes and scales against an independent `f64` replay of the feedback. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    reason = "the replay mirrors the row-major indices, single-letter names and Q32.32 casts of the ported kernel"
)]

use apeiron::engine::ONE;
use apeiron::learn::code::e8;
use apeiron::learn::fold::fit::gptq_e8::{feedback_u, gptq_e8, GptqE8, UFac, BLOCK, GROUP};
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

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

/// The whitened total error is not monotone in the beam: the search is greedy per 8-block, and a
/// wider beam can drop the greedy path (measured, beams `1,2,4,8`, five matrices: matrix 0
/// `[100.067, 101.785, 101.031, 96.480]`, matrix 2 `[104.529, 107.620, 98.021, 97.278]`). The
/// tested property is the mean over the five matrices: beams 4 and 8 each fall below beam 1.
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
