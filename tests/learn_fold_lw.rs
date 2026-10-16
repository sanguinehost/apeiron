//! `tests/learn_fold_lw.rs` — the shrinkage-intensity estimators: [`lw_intensity`] against the
//! definition `Σ_{i≠j} Var̂(mean_t s_ti s_tj)` computed by a literal double loop, its `λ ∈ [0, 1]`
//! clamp and its `(0, 0)` on a diagonal scatter; [`block_off`] against an explicitly formed
//! `H_b = Σ_k S_kᵀ S_k`; [`lw_block_intensity`] against the literal batch-means formula and its
//! `B = 1` zero; and the shape/sample-count refusals. Grade `tested`.
//!
//! Mutation control (see the lane return): `(nf * (nf - 1.0))` in `lw_intensity` changed to
//! `(nf * nf)` turns `lw_intensity_matches_definition` red.

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::needless_range_loop,
    reason = "the tests compare `f64` results by exact equality where both sides are the same \
              literal zero, index the row-major `[n, o]` sample buffers by `i/j/t`, and name the \
              scalars `num`/`nf`/`off`/`lambda` as the reference does"
)]

use apeiron::learn::fold::stats::lw::{block_off, lw_block_intensity, lw_intensity, LwError};

/// Deterministic 64-bit linear congruential generator (Numerical Recipes constants); the top 53
/// bits map to `[0, 1)` exactly.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / 9_007_199_254_740_992.0
    }

    fn signed(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }
}

const O: usize = 6;
const N: usize = 12;

/// Row-major `[N, O]` sample matrix `s_t` from the LCG.
fn samples(seed: u64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    (0..N * O).map(|_| rng.signed()).collect()
}

/// `H = Σ_t s_t s_tᵀ` (row-major `[o, o]`), the raw scatter of `s` (`[rows, o]`).
fn scatter(s: &[f64], rows: usize, o: usize) -> Vec<f64> {
    let mut h = vec![0.0f64; o * o];
    for t in 0..rows {
        for i in 0..o {
            let si = s[t * o + i];
            for j in 0..o {
                h[i * o + j] += si * s[t * o + j];
            }
        }
    }
    h
}

/// `Σ_t ‖s_t‖⁴` and `Σ_t s_t⁴` from the raw samples.
fn moments(s: &[f64], rows: usize, o: usize) -> (f64, Vec<f64>) {
    let mut s4 = 0.0f64;
    let mut q = vec![0.0f64; o];
    for t in 0..rows {
        let mut norm2 = 0.0f64;
        for i in 0..o {
            let v = s[t * o + i];
            norm2 += v * v;
            q[i] += v * v * v * v;
        }
        s4 += norm2 * norm2;
    }
    (s4, q)
}

/// `off(H/n) = Σ_{i≠j} (H_ij/n)²`, the literal off-diagonal sum.
fn off_literal(h: &[f64], o: usize, n: usize) -> f64 {
    let nf = n as f64;
    let mut off = 0.0f64;
    for i in 0..o {
        for j in 0..o {
            if i != j {
                let m = h[i * o + j] / nf;
                off += m * m;
            }
        }
    }
    off
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + b.abs())
}

#[test]
fn lw_intensity_matches_definition() {
    let s = samples(0x1234_5678);
    let h = scatter(&s, N, O);
    let (s4, q) = moments(&s, N, O);
    let (num, lambda) = lw_intensity(s4, &q, &h, O, N).unwrap();

    // Independent num from the definition: Σ_{i≠j} Var̂(mean_t s_ti s_tj), a literal double loop.
    let nf = N as f64;
    let mut num_ref = 0.0f64;
    for i in 0..O {
        for j in 0..O {
            if i == j {
                continue;
            }
            let mut mean = 0.0f64;
            for t in 0..N {
                mean += s[t * O + i] * s[t * O + j];
            }
            mean /= nf;
            let mut var = 0.0f64;
            for t in 0..N {
                let a = s[t * O + i] * s[t * O + j] - mean;
                var += a * a;
            }
            num_ref += var / (nf * (nf - 1.0));
        }
    }
    assert!(close(num, num_ref), "num {num} vs definition {num_ref}");

    let off = off_literal(&h, O, N);
    assert!(off > 0.0);
    let lambda_ref = (num_ref / off).clamp(0.0, 1.0);
    assert!(close(lambda, lambda_ref), "lambda {lambda} vs {lambda_ref}");
}

#[test]
fn lw_intensity_lambda_clamped_and_diagonal_zero() {
    let s = samples(0x0BAD_F00D);
    let h = scatter(&s, N, O);
    let (s4, q) = moments(&s, N, O);
    let (_, lambda) = lw_intensity(s4, &q, &h, O, N).unwrap();
    assert!((0.0..=1.0).contains(&lambda), "lambda {lambda}");

    // Diagonal-only samples: each sample has exactly one nonzero coordinate, so H is diagonal and
    // off = 0 -> (0, 0).
    let mut d = vec![0.0f64; N * O];
    for t in 0..N {
        d[t * O + (t % O)] = 0.5 + t as f64;
    }
    let hd = scatter(&d, N, O);
    let (s4d, qd) = moments(&d, N, O);
    assert_eq!(lw_intensity(s4d, &qd, &hd, O, N).unwrap(), (0.0, 0.0));
}

#[test]
fn block_off_matches_explicit_scatter() {
    const K: usize = 3;
    const T: usize = 4;
    let mut rng = Lcg::new(0xBEEF_0001);
    let blocks: Vec<Vec<f64>> = (0..K)
        .map(|_| (0..T * O).map(|_| rng.signed()).collect())
        .collect();
    let refs: Vec<&[f64]> = blocks.iter().map(Vec::as_slice).collect();
    let got = block_off(&refs, T, O).unwrap();

    // Literal H_b = Σ_k S_kᵀ S_k, then ‖H_b‖_F² − Σ_i (H_b)_ii².
    let mut hb = vec![0.0f64; O * O];
    for blk in &blocks {
        for r in 0..T {
            for i in 0..O {
                let v = blk[r * O + i];
                for j in 0..O {
                    hb[i * O + j] += v * blk[r * O + j];
                }
            }
        }
    }
    let frob: f64 = hb.iter().map(|x| x * x).sum();
    let mut diag = 0.0f64;
    for i in 0..O {
        let d = hb[i * O + i];
        diag += d * d;
    }
    let want = frob - diag;
    assert!(close(got, want), "block_off {got} vs explicit {want}");
}

#[test]
fn lw_block_intensity_matches_literal() {
    const B: usize = 4;
    const K: usize = 2;
    const T: usize = 3;
    let nb = K * T;
    let mut rng = Lcg::new(0xF00D_0002);
    let blocks: Vec<Vec<Vec<f64>>> = (0..B)
        .map(|_| {
            (0..K)
                .map(|_| (0..T * O).map(|_| rng.signed()).collect())
                .collect()
        })
        .collect();

    let mut p = 0.0f64;
    let mut h = vec![0.0f64; O * O];
    for blk in &blocks {
        let refs: Vec<&[f64]> = blk.iter().map(Vec::as_slice).collect();
        p += block_off(&refs, T, O).unwrap() / (nb as f64 * nb as f64);
        for s in blk {
            for r in 0..T {
                for i in 0..O {
                    let v = s[r * O + i];
                    for j in 0..O {
                        h[i * O + j] += v * s[r * O + j];
                    }
                }
            }
        }
    }
    let n = B * nb;
    let (num, lambda) = lw_block_intensity(p, B, &h, O, n).unwrap();

    let off = off_literal(&h, O, n);
    assert!(off > 0.0);
    let bf = B as f64;
    let num_ref = (p - bf * off) / (bf * (bf - 1.0));
    let lambda_ref = (num_ref / off).clamp(0.0, 1.0);
    assert!(close(num, num_ref), "num {num} vs literal {num_ref}");
    assert!(close(lambda, lambda_ref), "lambda {lambda} vs {lambda_ref}");

    // B = 1: no between-block variance.
    assert_eq!(lw_block_intensity(p, 1, &h, O, n).unwrap(), (0.0, 0.0));
}

#[test]
fn refusals() {
    let s = samples(0x5EED);
    let h = scatter(&s, N, O);
    let (s4, q) = moments(&s, N, O);
    assert!(off_literal(&h, O, N) > 0.0);

    assert_eq!(
        lw_intensity(s4, &q, &h[..O * O - 1], O, N),
        Err(LwError::ShapeMismatch)
    );
    assert_eq!(
        lw_intensity(s4, &q[..O - 1], &h, O, N),
        Err(LwError::ShapeMismatch)
    );
    assert_eq!(lw_intensity(s4, &q, &h, O, 1), Err(LwError::TooFewSamples));

    assert_eq!(
        lw_block_intensity(0.0, 2, &h[..O * O - 1], O, N),
        Err(LwError::ShapeMismatch)
    );
    assert_eq!(
        lw_block_intensity(0.0, 2, &h, O, 0),
        Err(LwError::TooFewSamples)
    );

    let short = [0.0f64; 5];
    assert_eq!(block_off(&[&short[..]], 3, 2), Err(LwError::ShapeMismatch));
}
