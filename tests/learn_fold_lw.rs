//! `tests/learn_fold_lw.rs` — the shrinkage-intensity estimators on their `Q32.32` word boundary:
//! [`lw_intensity`] against the definition `Σ_{i≠j} Var̂(mean_t s_ti s_tj)` computed by a literal
//! double loop, its `λ ∈ [0, 1]` clamp and its `(0, 0)` on a diagonal scatter; [`block_off`] against
//! an explicitly formed `H_b = Σ_k S_kᵀ S_k`; [`lw_block_intensity`] against the literal batch-means
//! formula and its `B = 1` zero; the shape/sample-count refusals; and the recorded word digests.
//! Grade `tested`.
//!
//! The definitions are evaluated in `f64` over the same words (a test oracle, never the build
//! path). Bound, derived: every input is a grid word (`≤ 2⁻³²` relative from the real it encodes),
//! each reported scalar is one `Q32.32` rounding (`2⁻³²` relative), and the oracle's own loop adds
//! `n` `f64` terms (`≤ n·2⁻⁵³`). `2⁻³² ≈ 2.33e-10`, so the `1e-9` relative bound used below is about
//! four grid quanta — the bound the pre-migration test used on the same internal words, not a
//! loosened one.
//!
//! Witness: [`words_reproduce_the_recorded_face`] pins 16 FNV-1a digests of the output words (8 LE
//! bytes per word) recorded on base `57d21789` from the then-`f64` boundary as `to_grid(result)`,
//! over 2 seeds × 2 shapes, including a shape with `n` small and `B < 2`.
//!
//! Mutation control (see the lane return): `off_of`'s `if j != i` changed to `if j == i` turns
//! [`words_reproduce_the_recorded_face`] red; restoring it passes.

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::needless_range_loop,
    reason = "the oracles evaluate the published formulas in `f64` over the grid words, index the \
              row-major `[n, o]` buffers by `i/j/t`, and name the scalars `num`/`nf`/`off`/`lambda` \
              as the reference does"
)]

use apeiron::learn::fold::stats::lw::{block_off, lw_block_intensity, lw_intensity, LwError};
use apeiron::learn::solve::q32::{div_rne, from_grid, narrow, rshift_rne, to_grid, GRID_BOUND};

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

/// `Q32.32` one.
const ONE: i64 = 1 << 32;

/// One input literal on the grid, halved so a fourth power stays inside `i128` by construction
/// (`|v| ≤ 2³¹` → `v⁴ ≤ 2¹²⁴`).
fn word(x: f64) -> i64 {
    to_grid(x, GRID_BOUND).expect("on grid")
}

/// Row-major `[rows, o]` sample matrix `s_t` as grid words.
fn sample_words(seed: u64, rows: usize, o: usize) -> Vec<i64> {
    let mut rng = Lcg::new(seed);
    (0..rows * o).map(|_| word(rng.signed() * 0.5)).collect()
}

/// `H = Σ_t s_t s_tᵀ` (row-major `[o, o]`) as words: the `i128` sum of word products (scale `2⁶⁴`)
/// rounded once by `2⁻³²`.
fn scatter_words(s: &[i64], rows: usize, o: usize) -> Vec<i64> {
    let mut h = vec![0_i128; o * o];
    for t in 0..rows {
        for i in 0..o {
            let si = i128::from(s[t * o + i]);
            for j in 0..o {
                h[i * o + j] += si * i128::from(s[t * o + j]);
            }
        }
    }
    h.into_iter()
        .map(|v| i64::try_from(rshift_rne(v, 32).unwrap()).unwrap())
        .collect()
}

/// `Σ_t ‖s_t‖⁴` and `Σ_t s_t⁴` as words. `‖s_t‖²` and each `s_ti⁴` are rounded once to a word
/// before the sum, keeping every intermediate inside `i128` for `|v| ≤ 2³¹`.
fn word_moments(s: &[i64], rows: usize, o: usize) -> (i64, Vec<i64>) {
    let mut s4: i128 = 0;
    let mut q = vec![0_i128; o];
    for t in 0..rows {
        let mut n2: i128 = 0;
        for i in 0..o {
            let v = i128::from(s[t * o + i]);
            let v2 = v * v;
            n2 += v2;
            q[i] += rshift_rne(v2 * v2, 96).unwrap();
        }
        let n2w = rshift_rne(n2, 32).unwrap();
        s4 += rshift_rne(n2w * n2w, 32).unwrap();
    }
    (
        i64::try_from(s4).unwrap(),
        q.into_iter().map(|v| i64::try_from(v).unwrap()).collect(),
    )
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + b.abs())
}

/// `off(H/n) = Σ_{i≠j} (H_ij/n)²`, the literal off-diagonal sum over the words.
fn off_literal(h: &[i64], o: usize, n: usize) -> f64 {
    let nf = n as f64;
    let mut off = 0.0f64;
    for i in 0..o {
        for j in 0..o {
            if i != j {
                let m = from_grid(h[i * o + j]) / nf;
                off += m * m;
            }
        }
    }
    off
}

/// FNV-1a over the 8 little-endian bytes of each word.
fn fnv(ws: &[i64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for w in ws {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// One witness shape: `o` columns, `rows` samples for [`lw_intensity`], `k` label-draw stacks of
/// `t` rows for [`block_off`], and the two block counts fed to [`lw_block_intensity`].
struct Case {
    o: usize,
    rows: usize,
    k: usize,
    t: usize,
    bs: [usize; 2],
}

/// `0`: shape of the definitional tests (`o = 6`, `n = 12`, `B ≥ 3`); `1`: `n` small (`n = 4`) with
/// `B = 1` (`B < 2`) and `B = 2`.
const CASES: [Case; 2] = [
    Case {
        o: 6,
        rows: 12,
        k: 3,
        t: 4,
        bs: [4, 3],
    },
    Case {
        o: 3,
        rows: 4,
        k: 2,
        t: 2,
        bs: [1, 2],
    },
];

/// The 16 recorded digests: `(label, FNV-1a of the output words)`. Recorded on base `57d21789`
/// through the then-`f64` boundary as `to_grid(result)`; the words API must reproduce them exactly.
const WITNESS: [(&str, u64); 16] = [
    ("0x12345678/0/intensity", 0x8b96_fa92_691c_17a9),
    ("0x12345678/0/off", 0x09f9_10fc_6c7e_a94d),
    ("0x12345678/0/block0", 0x7005_ccc9_e53b_c5c2),
    ("0x12345678/0/block1", 0xae1b_fb34_77c3_11c3),
    ("0x12345678/1/intensity", 0xb389_0781_0433_0060),
    ("0x12345678/1/off", 0xf5ba_e651_b331_d651),
    ("0x12345678/1/block0", 0x8820_1fb9_60ff_6465),
    ("0x12345678/1/block1", 0xb669_007b_45b3_eead),
    ("0xbadf00d/0/intensity", 0xc0c5_e387_7d05_b69d),
    ("0xbadf00d/0/off", 0x0ce1_bcec_4deb_2e81),
    ("0xbadf00d/0/block0", 0xf413_9e2e_28ff_a256),
    ("0xbadf00d/0/block1", 0xef20_0c7e_6fd3_9463),
    ("0xbadf00d/1/intensity", 0x275a_c9f5_6ec1_f6b3),
    ("0xbadf00d/1/off", 0xb4de_32df_0a81_6963),
    ("0xbadf00d/1/block0", 0x8820_1fb9_60ff_6465),
    ("0xbadf00d/1/block1", 0xcd2f_09fb_6d3d_b7e3),
];

/// The four digest rows of one `(seed, shape)`: the intensity pair, the block `off`, and the block
/// intensity pair at each of the case's two block counts.
fn case_digests(seed: u64, ci: usize, c: &Case) -> Vec<(String, u64)> {
    let s = sample_words(seed, c.rows, c.o);
    let h = scatter_words(&s, c.rows, c.o);
    let (s4, q) = word_moments(&s, c.rows, c.o);
    let (num, lambda) = lw_intensity(s4, &q, &h, c.o, c.rows).unwrap();
    let mut rows = vec![(format!("{seed:#x}/{ci}/intensity"), fnv(&[num, lambda]))];

    let mut rng = Lcg::new(seed ^ 0x5a5a_5a5a);
    let stacks: Vec<Vec<i64>> = (0..c.k)
        .map(|_| (0..c.t * c.o).map(|_| word(rng.signed() * 0.5)).collect())
        .collect();
    let refs: Vec<&[i64]> = stacks.iter().map(Vec::as_slice).collect();
    let off = block_off(&refs, c.t, c.o).unwrap();
    rows.push((format!("{seed:#x}/{ci}/off"), fnv(&[off])));

    let all: Vec<i64> = stacks.iter().flat_map(|b| b.iter().copied()).collect();
    let ht = scatter_words(&all, c.rows, c.o);
    let nb = c.k * c.t;
    let mut p: i128 = 0;
    for stack in &stacks {
        let one = block_off(&[stack.as_slice()], c.t, c.o).unwrap();
        let q_word = div_rne(i128::from(one), i128::try_from(nb * nb).unwrap()).unwrap();
        p += i128::from(narrow(q_word).unwrap());
    }
    let p = i64::try_from(p).unwrap();
    for (bi, &b) in c.bs.iter().enumerate() {
        let (num, lambda) = lw_block_intensity(p, b, &ht, c.o, b * nb).unwrap();
        rows.push((format!("{seed:#x}/{ci}/block{bi}"), fnv(&[num, lambda])));
    }
    rows
}

#[test]
fn words_reproduce_the_recorded_face() {
    let mut got: Vec<(String, u64)> = Vec::new();
    for &seed in &[0x1234_5678_u64, 0x0BAD_F00D] {
        for (ci, c) in CASES.iter().enumerate() {
            got.extend(case_digests(seed, ci, c));
        }
    }
    assert_eq!(got.len(), WITNESS.len());
    for (row, (label, want)) in got.iter().zip(WITNESS) {
        assert_eq!(row.0, label);
        assert_eq!(row.1, want, "{}", row.0);
    }
}

#[test]
fn lw_intensity_matches_definition() {
    let s = sample_words(0x1234_5678, N, O);
    let h = scatter_words(&s, N, O);
    let (s4, q) = word_moments(&s, N, O);
    let (num, lambda) = lw_intensity(s4, &q, &h, O, N).unwrap();

    // Independent num from the definition: Σ_{i≠j} Var̂(mean_t s_ti s_tj), a literal double loop over
    // the same words.
    let sf: Vec<f64> = s.iter().map(|&v| from_grid(v)).collect();
    let nf = N as f64;
    let mut num_ref = 0.0f64;
    for i in 0..O {
        for j in 0..O {
            if i == j {
                continue;
            }
            let mut mean = 0.0f64;
            for t in 0..N {
                mean += sf[t * O + i] * sf[t * O + j];
            }
            mean /= nf;
            let mut var = 0.0f64;
            for t in 0..N {
                let a = sf[t * O + i] * sf[t * O + j] - mean;
                var += a * a;
            }
            num_ref += var / (nf * (nf - 1.0));
        }
    }
    assert!(
        close(from_grid(num), num_ref),
        "num {} vs definition {num_ref}",
        from_grid(num)
    );

    let off = off_literal(&h, O, N);
    assert!(off > 0.0);
    let lambda_ref = (num_ref / off).clamp(0.0, 1.0);
    assert!(
        close(from_grid(lambda), lambda_ref),
        "lambda {} vs {lambda_ref}",
        from_grid(lambda)
    );
}

#[test]
fn lw_intensity_lambda_clamped_and_diagonal_zero() {
    let s = sample_words(0x0BAD_F00D, N, O);
    let h = scatter_words(&s, N, O);
    let (s4, q) = word_moments(&s, N, O);
    let (_, lambda) = lw_intensity(s4, &q, &h, O, N).unwrap();
    assert!((0..=ONE).contains(&lambda), "lambda {lambda}");

    // Diagonal-only samples: each sample has exactly one nonzero coordinate, so H is diagonal and
    // off = 0 -> (0, 0).
    let mut d = vec![0_i64; N * O];
    for t in 0..N {
        d[t * O + (t % O)] = word(0.5 + t as f64);
    }
    let hd = scatter_words(&d, N, O);
    let (s4d, qd) = word_moments(&d, N, O);
    assert_eq!(lw_intensity(s4d, &qd, &hd, O, N).unwrap(), (0, 0));
}

#[test]
fn block_off_matches_explicit_scatter() {
    const K: usize = 3;
    const T: usize = 4;
    let mut rng = Lcg::new(0xBEEF_0001);
    let blocks: Vec<Vec<i64>> = (0..K)
        .map(|_| (0..T * O).map(|_| word(rng.signed() * 0.5)).collect())
        .collect();
    let refs: Vec<&[i64]> = blocks.iter().map(Vec::as_slice).collect();
    let got = block_off(&refs, T, O).unwrap();

    // Literal H_b = Σ_k S_kᵀ S_k in `f64`, then ‖H_b‖_F² − Σ_i (H_b)_ii².
    let mut hb = vec![0.0f64; O * O];
    for blk in &blocks {
        for r in 0..T {
            for i in 0..O {
                let v = from_grid(blk[r * O + i]);
                for j in 0..O {
                    hb[i * O + j] += v * from_grid(blk[r * O + j]);
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
    assert!(
        close(from_grid(got), want),
        "block_off {} vs explicit {want}",
        from_grid(got)
    );
}

#[test]
fn lw_block_intensity_matches_literal() {
    const B: usize = 4;
    const K: usize = 2;
    const T: usize = 3;
    let nb = K * T;
    let mut rng = Lcg::new(0xF00D_0002);
    let blocks: Vec<Vec<Vec<i64>>> = (0..B)
        .map(|_| {
            (0..K)
                .map(|_| (0..T * O).map(|_| word(rng.signed() * 0.5)).collect())
                .collect()
        })
        .collect();

    // p = Σ_b off(H_b)/n_b² on the words: one `Q32.32` quotient per block.
    let mut p: i128 = 0;
    for blk in &blocks {
        let refs: Vec<&[i64]> = blk.iter().map(Vec::as_slice).collect();
        let ob = block_off(&refs, T, O).unwrap();
        let q = div_rne(i128::from(ob), i128::try_from(nb * nb).unwrap()).unwrap();
        p += i128::from(narrow(q).unwrap());
    }
    let p = i64::try_from(p).unwrap();
    let all: Vec<i64> = blocks
        .iter()
        .flat_map(|b| b.iter().flat_map(|s| s.iter().copied()))
        .collect();
    let h = scatter_words(&all, B * nb, O);
    let n = B * nb;
    let (num, lambda) = lw_block_intensity(p, B, &h, O, n).unwrap();

    // The literal formula over the same words.
    let pf = blocks
        .iter()
        .map(|blk| {
            let refs: Vec<&[i64]> = blk.iter().map(Vec::as_slice).collect();
            from_grid(block_off(&refs, T, O).unwrap()) / (nb as f64 * nb as f64)
        })
        .sum::<f64>();
    let off = off_literal(&h, O, n);
    assert!(off > 0.0);
    let bf = B as f64;
    let num_ref = (pf - bf * off) / (bf * (bf - 1.0));
    let lambda_ref = (num_ref / off).clamp(0.0, 1.0);
    assert!(
        close(from_grid(num), num_ref),
        "num {} vs literal {num_ref}",
        from_grid(num)
    );
    assert!(
        close(from_grid(lambda), lambda_ref),
        "lambda {} vs {lambda_ref}",
        from_grid(lambda)
    );

    // B = 1: no between-block variance.
    assert_eq!(lw_block_intensity(p, 1, &h, O, n).unwrap(), (0, 0));
}

#[test]
fn refusals() {
    let s = sample_words(0x5EED, N, O);
    let h = scatter_words(&s, N, O);
    let (s4, q) = word_moments(&s, N, O);
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
        lw_block_intensity(0, 2, &h[..O * O - 1], O, N),
        Err(LwError::ShapeMismatch)
    );
    assert_eq!(
        lw_block_intensity(0, 2, &h, O, 0),
        Err(LwError::TooFewSamples)
    );

    let short = [0_i64; 5];
    assert_eq!(block_off(&[&short[..]], 3, 2), Err(LwError::ShapeMismatch));
}
