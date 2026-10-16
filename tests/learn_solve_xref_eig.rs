//! `tests/learn_solve_xref_eig.rs` — bit-identity cross-reference for the Q-xeig control-flow
//! restructure of `src/learn/solve/eig.rs`.
//!
//! Every public entry point (`top_eigh`) runs on deterministic fixtures and every output word and
//! every error variant is folded into one FNV-1a 64 hash. The asserted constants were taken from
//! the pre-change code at `29ae4148`; the restructure must reproduce them exactly.
//!
//! Fixtures: sizes 1, 3, 8, 9, 16, 17 (odd and around the oversampling block), an LCG-built
//! diagonal-dominant SPD, a planted-rank `A·Aᵀ` SPD, the all-zero matrix, and every refusal arm
//! (shape, empty, rank zero, rank out of range, input bound, dimension-product overflow,
//! non-symmetry, low-rank `DegenerateBasis`, `NotConverged`).
//!
//! Grade `tested`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the fixtures fold raw Q32.32 words, indices and the f64 tolerance-scale constant into \
              an FNV-1a 64 state; every cast is a deliberate bit reinterpretation in the test only."
)]

use apeiron::learn::solve::eig::EigError as E;

/// Raw Q32.32 `1.0`.
const ONE: i64 = 1 << 32;

/// Deterministic 64-bit xorshift (the fixtures' only randomness).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// An `i64` in `(−hi, hi)`.
    fn next_small(&mut self, hi: i64) -> i64 {
        let span = 2 * hi as u64;
        (self.next_u64() % span) as i64 - hi
    }
}

/// An `n × n` symmetric, diagonal-dominant matrix: the off-diagonals from the LCG in a modest
/// range, the diagonal `n · hi²` so the matrix is positive definite and converges.
fn spd(n: usize, hi: i64, seed: u64) -> Vec<i64> {
    let mut lcg = Lcg::new(seed);
    let mut m = vec![0i64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let v = lcg.next_small(hi);
            m[i * n + j] = v;
            m[j * n + i] = v;
        }
    }
    for i in 0..n {
        m[i * n + i] = (n as i64) * hi * hi;
    }
    m
}

/// `round32(v) = (v + 2³¹) >> 32`.
fn round32(v: i128) -> i128 {
    (v + (1i128 << 31)) >> 32
}

/// A Q32.32 `n × n` PSD matrix `A·Aᵀ` with `A` an `n × m` LCG matrix, one round per entry.
fn gram(n: usize, m: usize, hi: i64, seed: u64) -> Vec<i64> {
    let mut lcg = Lcg::new(seed);
    let mut a = vec![0i64; n * m];
    for e in &mut a {
        *e = lcg.next_small(hi);
    }
    let mut out = vec![0i64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc: i128 = 0;
            for k in 0..m {
                acc += i128::from(a[i * m + k]) * i128::from(a[j * m + k]);
            }
            out[i * n + j] = i64::try_from(round32(acc)).expect("i64");
        }
    }
    out
}

/// The fixtures: `(label, m, n, r, threads)`.
fn fixtures() -> Vec<(String, Vec<i64>, usize, usize, usize)> {
    let mut f: Vec<(String, Vec<i64>, usize, usize, usize)> = Vec::new();
    for &n in &[1usize, 3, 8, 9, 16, 17] {
        for &r in &[1usize, 2, 4] {
            if r > n {
                continue;
            }
            for &threads in &[1usize, 4] {
                f.push((
                    format!("spd n={n} r={r} t={threads}"),
                    spd(n, 1 << 14, 0x1234_5678 ^ n as u64),
                    n,
                    r,
                    threads,
                ));
            }
        }
    }
    for &n in &[8usize, 16, 32] {
        let m = gram(n, n.min(4), 1 << 12, 0xdead_beef ^ n as u64);
        f.push((format!("gram n={n}"), m.clone(), n, 1, 4));
        f.push((format!("gram n={n} r=2"), m, n, 2, 1));
    }
    // All-zero.
    f.push(("zero n=4".into(), vec![0i64; 16], 4, 1, 1));
    // Refusals.
    f.push(("shape".into(), vec![0i64; 5], 2, 1, 1));
    f.push(("empty".into(), vec![], 0, 1, 1));
    f.push(("rank zero".into(), vec![ONE; 4], 2, 0, 1));
    f.push(("rank range".into(), vec![ONE; 4], 2, 3, 1));
    f.push(("overflow n".into(), vec![], usize::MAX, 1, 1));
    let mut over = vec![0i64; 4];
    over[0] = 1i64 << 36;
    over[3] = ONE;
    f.push(("input bound".into(), over, 2, 1, 1));
    let mut bad = vec![0i64; 9];
    for i in 0..3 {
        bad[i * 3 + i] = ONE;
    }
    bad[1] = 2;
    f.push(("not symmetric".into(), bad, 3, 1, 1));
    let mut near = vec![0i64; 9];
    for i in 0..3 {
        near[i * 3 + i] = ONE;
    }
    near[1] = 1;
    f.push(("near symmetric".into(), near, 3, 1, 1));
    // Rank-2 diagonal of order 16: rank < s = 9 → DegenerateBasis.
    let mut low = vec![0i64; 256];
    low[0] = ONE;
    low[17] = ONE;
    f.push(("degenerate".into(), low, 16, 1, 1));
    // A near-degenerate spectrum that cannot reach tolerance at MAX_ITERS.
    let n = 16usize;
    let mut slow = vec![0i64; n * n];
    for k in 0..n {
        let v = match k {
            9 => (0.955 * 4_294_967_296.0) as i64,
            0..=8 => ((1.0 - 0.005 * (k as f64)) * 4_294_967_296.0) as i64,
            _ => ONE >> 7,
        };
        slow[k * n + k] = v;
    }
    f.push(("not converged".into(), slow, n, 1, 1));
    f
}

/// Fold a `u64` into the FNV-1a 64 state.
fn fold(h: &mut u64, v: u64) {
    for b in 0..8 {
        *h ^= (v >> (8 * b)) & 0xff;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

#[test]
fn top_eigh_bit_identity() {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (label, m, n, r, threads) in fixtures() {
        match apeiron::learn::solve::eig::top_eigh(&m, n, r, threads) {
            Ok(t) => {
                fold(&mut h, 0);
                fold(&mut h, t.evals.len() as u64);
                for &e in &t.evals {
                    fold(&mut h, e as u64);
                }
                for &e in &t.evecs {
                    fold(&mut h, e as u64);
                }
                fold(&mut h, t.iters as u64);
            }
            Err(e) => {
                fold(&mut h, 1);
                let d = match e {
                    E::ShapeMismatch => 1,
                    E::Empty => 2,
                    E::RankZero => 3,
                    E::RankOutOfRange => 4,
                    E::InputBound => 5,
                    E::NotSymmetric => 6,
                    E::Overflow => 7,
                    E::DegenerateBasis => 8,
                    E::NotConverged { residual } => {
                        fold(&mut h, residual as u64);
                        9
                    }
                };
                fold(&mut h, d);
            }
        }
        let _ = label;
    }
    println!("top_eigh FNV-1a 64 = {h:#018x}");
    // taken from the pre-change code at 29ae4148
    assert_eq!(h, 0x1d7c_dca4_6cbe_eca5, "bit-identity hash drift");
}
