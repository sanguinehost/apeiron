//! `tests/learn_fold_corrector.rs` — the fp16/u8-rounded rank-`rank` corrector in the `Hl` metric
//! (`fold_transfer.py:560–582`, the `round_f16=True` and `u8_corr=True` branches).
//!
//! Positive: [`Mode::Exact`] equals [`lowrank_corrector_exact`]; the fp16 correction stays within the
//! half-ulp bound of the exact correction in the `Hl` metric (and no better than the exact optimum);
//! the u8 factors lie on the 8-bit grid (re-quantizing is a fixed point) with the exact bit count;
//! `Uh·Vh == Corr` bit-for-bit; a full-rank `rank = min(o, n)` recovers `ΔW` to each mode's
//! precision. Boundary: the shape, rank and non-finite refusals. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the fixed-seed LCG maps a 24-bit integer field to an f64 in [0, 1), and the test \
              kernels index row-major `[r, c]` buffers by `i/j/k/l` against the spec's \
              `DeltaW[i, j]`/`Uh`/`Vh` arithmetic; the `assert_eq!`s on f64 are the intended \
              bit-exact fixed-point and factor-product checks, not approximate equality."
)]

use apeiron::learn::fold::fit::corrector::{lowrank_corrector, CorrectorError, Mode};
use apeiron::learn::fold::fit::eigh::lowrank_corrector_exact;
use apeiron::learn::fold::fit::quant::quant_c_u8;

/// Deterministic 64-bit LCG (Numerical Recipes constants), the tests' only randomness source.
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

    /// Uniform in `[0, 1)`, 53 significant bits.
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Uniform in `[-1, 1)`.
    fn sym(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, &x| m.max(x.abs()))
}

/// `a · b` for row-major `[r, k] · [k, c]`, accumulating `Σ_t` per `(i, j)` in ascending `t`.
fn matmul(a: &[f64], b: &[f64], r: usize, k: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r * c];
    for i in 0..r {
        for j in 0..c {
            let mut acc = 0.0f64;
            for t in 0..k {
                acc += a[i * k + t] * b[t * c + j];
            }
            out[i * c + j] = acc;
        }
    }
    out
}

/// A random SPD matrix `BᵀB + ½I`, row-major `[n, n]`.
fn random_spd(rng: &mut Lcg, n: usize) -> Vec<f64> {
    let mut b = vec![0.0f64; n * n];
    for x in &mut b {
        *x = rng.sym();
    }
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += b[k * n + i] * b[k * n + j];
            }
            h[i * n + j] = acc + if i == j { 0.5 } else { 0.0 };
        }
    }
    h
}

/// `tr(R · Hl · Rᵀ) = Σ_ij (R·Hl)[i, j]·R[i, j]` for row-major `[o, n]` `r`, `[n, n]` `hl` — the
/// squared `Hl`-norm of `R`, the objective the corrector minimises.
fn trace_r_hl_rt(r: &[f64], hl: &[f64], o: usize, n: usize) -> f64 {
    let rhl = matmul(r, hl, o, n, n);
    let mut t = 0.0f64;
    for i in 0..o {
        for j in 0..n {
            t += rhl[i * n + j] * r[i * n + j];
        }
    }
    t
}

/// The `o × n` residual `ΔW − Corr`.
fn residual(dw: &[f64], corr: &[f64]) -> Vec<f64> {
    dw.iter().zip(corr).map(|(x, y)| x - y).collect()
}

/// The shared fixture: random `ΔW` (`o × n`) and SPD `Hl` (`n × n`) from one seed.
fn fixture(seed: u64, o: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let dw: Vec<f64> = (0..o * n).map(|_| rng.sym()).collect();
    let hl = random_spd(&mut rng, n);
    (dw, hl)
}

const O: usize = 6;
const N: usize = 10;

#[test]
fn exact_equals_reference() {
    let rank = 3usize;
    let (dw, hl) = fixture(0x1111_2222_3333_4444, O, N);
    let mine = lowrank_corrector(&dw, &hl, O, N, rank, Mode::Exact, true).expect("exact");
    let reference = lowrank_corrector_exact(&dw, &hl, O, N, rank).expect("reference");
    assert_eq!(mine.bits, (16 * rank * (O + N)) as u64, "exact bit count");
    assert_eq!(mine.corr, reference.corr, "Corr != reference Corr");
    assert!(
        (mine.top_sum - reference.top_sum).abs() <= 1e-12,
        "top_sum {} != {}",
        mine.top_sum,
        reference.top_sum
    );
    let f = mine.factors.as_ref().expect("factors requested");
    assert_eq!(f.u, reference.u, "Uh != reference U");
    assert_eq!(f.v, reference.v, "Vh != reference V");
}

#[test]
fn f16_reconstruction_bound() {
    let rank = 3usize;
    let (dw, hl) = fixture(0x5555_6666_7777_8888, O, N);
    let exact = lowrank_corrector(&dw, &hl, O, N, rank, Mode::Exact, true).expect("exact");
    let f16 = lowrank_corrector(&dw, &hl, O, N, rank, Mode::F16, true).expect("f16");
    assert_eq!(f16.bits, (16 * rank * (O + N)) as u64, "f16 bit count");
    assert!(
        (f16.top_sum - exact.top_sum).abs() <= 1e-12,
        "f16 top_sum {} != exact {}",
        f16.top_sum,
        exact.top_sum
    );
    let err = |corr: &[f64]| trace_r_hl_rt(&residual(&dw, corr), &hl, O, N);
    let e_exact = err(&exact.corr);
    let e_f16 = err(&f16.corr);
    // ||ΔW||²_Hl, the scale the half-ulp of the binary16 factors propagates at.
    let s = trace_r_hl_rt(&dw, &hl, O, N);
    let half_ulp = 2f64.powi(-11);
    let slack = 16.0 * half_ulp * s + 1e-12;
    assert!(
        e_f16 <= e_exact + slack,
        "fp16 error {e_f16} exceeds exact {e_exact} + half-ulp slack {slack}"
    );
    // The exact correction is the optimal rank-`rank` `Hl` projection, so fp16 cannot beat it.
    assert!(
        e_f16 >= e_exact - 1e-12,
        "fp16 error {e_f16} below the exact optimum {e_exact}"
    );
}

#[test]
fn u8_bits_and_grid() {
    let rank = 3usize;
    let (dw, hl) = fixture(0x9999_AAAA_BBBB_CCCC, O, N);
    let u8 = lowrank_corrector(&dw, &hl, O, N, rank, Mode::U8, true).expect("u8");
    assert_eq!(
        u8.bits,
        (rank as u64) * (8 * (O + N) as u64 + 64),
        "u8 bit count"
    );
    let f = u8.factors.as_ref().expect("factors requested");
    assert_eq!(f.u.len(), O * rank);
    assert_eq!(f.v.len(), rank * N);
    // Each column of Uh is on the 8-bit grid: re-quantizing it is a fixed point, bit-for-bit.
    for k in 0..rank {
        let col: Vec<f64> = (0..O).map(|i| f.u[i * rank + k]).collect();
        let (q, bits) = quant_c_u8(&col, 8).expect("column quantizes");
        assert_eq!(bits, 8 * O as u64 + 32, "column header bits");
        for i in 0..O {
            assert_eq!(q[i], f.u[i * rank + k], "Uh[{i}, {k}] off the u8 grid");
        }
    }
    // Each row of Vh is on the 8-bit grid.
    for k in 0..rank {
        let row = &f.v[k * N..k * N + N];
        let (q, bits) = quant_c_u8(row, 8).expect("row quantizes");
        assert_eq!(bits, 8 * N as u64 + 32, "row header bits");
        assert_eq!(q, row.to_vec(), "Vh row {k} off the u8 grid");
    }
}

#[test]
fn factors_product_bit_exact() {
    let rank = 3usize;
    let (dw, hl) = fixture(0x1234_5678_9ABC_DEF0, O, N);
    for mode in [Mode::Exact, Mode::F16, Mode::U8] {
        let c = lowrank_corrector(&dw, &hl, O, N, rank, mode, true).expect("corrector");
        let f = c.factors.as_ref().expect("factors requested");
        let prod = matmul(&f.u, &f.v, O, rank, N);
        assert_eq!(prod, c.corr, "Uh·Vh != Corr for {mode:?}");
    }
}

#[test]
fn full_rank_recovery() {
    let rank = O.min(N); // full row rank
    let (dw, hl) = fixture(0x0FED_CBA9_8765_4321, O, N);
    let scale = max_abs(&dw);
    // Per-mode precision of the recovery `Corr ≈ ΔW` at full rank.
    let tolerances = [(Mode::Exact, 1e-9), (Mode::F16, 1e-2), (Mode::U8, 5e-2)];
    for (mode, tol) in tolerances {
        let c = lowrank_corrector(&dw, &hl, O, N, rank, mode, false).expect("corrector");
        let e = max_abs(&residual(&dw, &c.corr));
        assert!(
            e <= tol * scale,
            "{mode:?}: full-rank recovery error {e} exceeds {tol}·scale {}",
            tol * scale
        );
    }
}

#[test]
fn refusals() {
    let (dw, hl) = fixture(0xBEEF_C0DE_1234_5678, O, N);
    // ΔW length disagrees with o·n.
    let dw_bad = vec![0.0f64; O * N + 1];
    assert_eq!(
        lowrank_corrector(&dw_bad, &hl, O, N, 3, Mode::F16, false).err(),
        Some(CorrectorError::ShapeMismatch)
    );
    // Hl length disagrees with n·n.
    let hl_bad = vec![0.0f64; N * N - 1];
    assert_eq!(
        lowrank_corrector(&dw, &hl_bad, O, N, 3, Mode::U8, false).err(),
        Some(CorrectorError::ShapeMismatch)
    );
    // rank > o, in each mode.
    for mode in [Mode::Exact, Mode::F16, Mode::U8] {
        assert_eq!(
            lowrank_corrector(&dw, &hl, O, N, O + 1, mode, false).err(),
            Some(CorrectorError::RankOutOfRange),
            "{mode:?}: rank > o not refused"
        );
    }
    // Non-finite ΔW entry.
    let mut dw_nan = dw.clone();
    dw_nan[0] = f64::NAN;
    assert_eq!(
        lowrank_corrector(&dw_nan, &hl, O, N, 3, Mode::F16, false).err(),
        Some(CorrectorError::NonFinite)
    );
    // Rank 0 returns the empty corrector.
    let empty = lowrank_corrector(&dw, &hl, O, N, 0, Mode::F16, true).expect("rank 0");
    assert_eq!(empty.bits, 0);
    assert_eq!(empty.top_sum, 0.0);
    assert!(empty.corr.iter().all(|&x| x == 0.0));
    let f = empty.factors.as_ref().expect("factors requested");
    assert!(f.u.is_empty() && f.v.is_empty());
}
