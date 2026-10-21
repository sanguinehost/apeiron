//! `tests/learn_fold_nb_refine.rs` — the reduced joint block-ICM post-pass (`nb_refine`) on a
//! finished two-plane pair: `J` falls when a round is kept and never rises, a run on its own output
//! reaches a round that keeps nothing, the stored-format bounds hold (row scale a `quant_r_hier`
//! value, column scale on the `6`-bit grid), `rounds = 0` is the identity, the returned weights are
//! the returned packs' `compose`, and shape / width inputs are refused. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the fixed-seed LCG maps integers to `f64` by division and the flat row-major index \
              arithmetic is the reference payload format; the equality checks are exact float \
              comparisons by construction (the same expressions the port evaluates), and the \
              `o/n/m/j` names are the reference's vocabulary"
)]

use apeiron::learn::code::e8::codeword;
use apeiron::learn::fold::fit::icm::{plane_w, Plane, PlaneRef};
use apeiron::learn::fold::fit::nb_refine::{
    nb_refine, NbError, NbRefined, C_BITS, NB_SWEEPS, ROW_CHUNK,
};
use apeiron::learn::fold::fit::quant::{quant_c_u8, quant_r_hier};
use apeiron::learn::fold::fit::steps::{compose, objective};
use apeiron::learn::solve::q32::{from_grid, round_q32_even, to_grid, GRID_BOUND};

#[path = "support/fwht_oracle.rs"]
mod fwht_oracle;

use fwht_oracle::fwht128;

/// Deterministic xorshift64 LCG.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
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

/// The 256 codewords as `Q32.32` words, the table the module builds.
fn codebook() -> Vec<[i64; BLOCK]> {
    (0..CODE_LEN)
        .map(|j| codeword(j).expect("row in range"))
        .collect()
}

/// A value on the `Q32.32` grid: `round(x·2³²)` exists and is within the grid range.
fn q1(x: f64) -> i64 {
    to_grid(x, GRID_BOUND).expect("value inside the Q32.32 grid range")
}

/// A fixture as the `Q32.32` words the migrated step consumes.
fn qw(x: &[f64]) -> Vec<i64> {
    x.iter().copied().map(q1).collect()
}

/// A word slice read back through the grid map, the migrated boundary's read-out.
fn uw(x: &[i64]) -> Vec<f64> {
    x.iter().copied().map(from_grid).collect()
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

/// A random strictly diagonally dominant symmetric matrix: `O(n²)` time and space, SPD by
/// Gershgorin (`Hl[i,i] > Σ_{j≠i} |Hl[i,j]|`), a stand-in for the calibration Gram.
fn rand_spd(n: usize, rng: &mut Lcg) -> Vec<f64> {
    let mut h = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let v = rng.uniform(-1.0, 1.0);
            h[i * n + j] = v;
            h[j * n + i] = v;
        }
    }
    for i in 0..n {
        let mut off = 0.0_f64;
        for j in 0..n {
            if j != i {
                off += h[i * n + j].abs();
            }
        }
        h[i * n + i] = off + 1.0 + rng.uniform(0.0, 1.0);
    }
    h
}

/// A plane with random codebook rows, random group scales in `[0.5, 1.5)` and the given column
/// scale; `d` is one.
fn rand_plane(o: usize, n: usize, c: &[i64], rng: &mut Lcg) -> Plane {
    let nb = n / BLOCK;
    let g = n / GROUP;
    let cb = codebook();
    let mut s = vec![0_i64; o * n];
    let mut idx = vec![0_u16; o * nb];
    for row in 0..o {
        for m in 0..nb {
            let j = (rng.next_u64() % 256) as usize;
            idx[row * nb + m] = j as u16;
            for k in 0..BLOCK {
                s[row * n + m * BLOCK + k] = cb[j][k];
            }
        }
    }
    Plane {
        s,
        idx,
        r: (0..o * g).map(|_| q1(rng.uniform(0.5, 1.5))).collect(),
        c: c.to_vec(),
        d: vec![1_i64 << 32; n],
    }
}

/// The two-plane weight `plane_w(p1) + plane_w(p2)`, read back through the grid map.
fn pair_w(p1: &Plane, p2: &Plane, o: usize, n: usize) -> Vec<f64> {
    let mut w = uw(&plane_w(pref(p1), o, n).expect("plane_w"));
    let w2 = uw(&plane_w(pref(p2), o, n).expect("plane_w"));
    for k in 0..o * n {
        w[k] += w2[k];
    }
    w
}

/// `J(ws, w1 + w2, hl)`, the unweighted script objective.
fn pair_j(ws: &[f64], p1: &Plane, p2: &Plane, hl: &[f64], o: usize, n: usize) -> f64 {
    let w = pair_w(p1, p2, o, n);
    from_grid(objective(&qw(ws), &qw(&w), &qw(hl), o, n).expect("objective"))
}

/// The derived bound on `|J(ws, w) − J(ws, w̃)|` when `w̃` is `w` summed on the `Q32.32` grid.
///
/// `∂J/∂W = 2·(Ws − W)·Hl`, and each summed entry carries at most one grid unit `U = 2⁻³²` of
/// error, so `|ΔJ| ≤ 2·U·Σ|(Ws − W)·Hl|`; the factor `2` is absorbed into the full unit below.
fn j_grid_bound(ws: &[f64], w: &[f64], hl: &[f64], o: usize, n: usize) -> f64 {
    let u = 2f64.powi(-32);
    let mut total = 0.0_f64;
    for row in 0..o {
        for i in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += (ws[row * n + k] - w[row * n + k]) * hl[k * n + i];
            }
            total += acc.abs();
        }
    }
    u * total
}

/// The fixture: a random SPD metric, a target pair with a column-scale spread, `ws` its sum plus a
/// small perturbation, and two input planes with random codes and unit column scales.
struct Fixture {
    ws: Vec<f64>,
    hl: Vec<f64>,
    p1: Plane,
    p2: Plane,
    o: usize,
    n: usize,
}

impl Fixture {
    fn new(seed: u64, o: usize, n: usize) -> Self {
        let mut rng = Lcg::new(seed);
        let hl = rand_spd(n, &mut rng);
        let ct: Vec<i64> = (0..n).map(|_| q1(rng.uniform(0.5, 1.5))).collect();
        let truth1 = rand_plane(o, n, &ct, &mut rng);
        let truth2 = rand_plane(o, n, &ct, &mut rng);
        let mut ws = pair_w(&truth1, &truth2, o, n);
        for v in &mut ws {
            *v += 0.1 * rng.uniform(-1.0, 1.0);
        }
        let ones = vec![1_i64 << 32; n];
        let p1 = rand_plane(o, n, &ones, &mut rng);
        let p2 = rand_plane(o, n, &ones, &mut rng);
        Self {
            ws,
            hl,
            p1,
            p2,
            o,
            n,
        }
    }

    /// The post-pass from an explicit pack pair at the ported `sweeps = NB_SWEEPS`, `icm_ridge = 0`.
    fn run_from(&self, a: &Plane, b: &Plane, rounds: usize) -> NbRefined {
        nb_refine(
            &qw(&self.ws),
            &qw(&self.hl),
            pref(a),
            pref(b),
            self.o,
            self.n,
            None,
            0,
            NB_SWEEPS,
            rounds,
        )
        .expect("nb_refine")
    }

    fn run(&self, rounds: usize) -> NbRefined {
        self.run_from(&self.p1, &self.p2, rounds)
    }
}

#[test]
fn learn_fold_nb_refine_lowers_j_and_keeps_a_round() {
    let f = Fixture::new(0x4E42_5EED, 4, 256);
    let out = f.run(2);
    println!(
        "learn_fold_nb_refine: o={} n={} j_seq={} j_nb={} rounds_kept={} moves={} per_round={:?}",
        f.o,
        f.n,
        out.log.j_seq,
        out.log.j_nb,
        out.log.rounds_kept,
        out.log.moves,
        out.log.per_round
    );
    // The acceptance rule of the port: a round is kept only when J strictly falls, so the returned
    // pair's J can never exceed the input pair's, and a fixture with a kept round is strictly below.
    assert!(
        out.log.j_nb <= out.log.j_seq,
        "j_nb {} above j_seq {}",
        out.log.j_nb,
        out.log.j_seq
    );
    assert!(
        out.log.rounds_kept >= 1,
        "no round kept: per_round {:?}",
        out.log.per_round
    );
    assert!(
        out.log.j_nb < out.log.j_seq,
        "j_nb {} is not below j_seq {}",
        out.log.j_nb,
        out.log.j_seq
    );
    assert_eq!(out.log.per_round.len(), out.log.rounds_kept);
    assert!(out.log.moves > 0, "a kept round must have moved codes");
    // The log's j_nb is the objective of the returned pair measured independently. The port sums
    // the two weights on the `Q32.32` grid, so the two `J`s agree to `j_grid_bound`, not exactly.
    let j_ret = pair_j(&f.ws, &out.p1, &out.p2, &f.hl, f.o, f.n);
    let sum = pair_w(&out.p1, &out.p2, f.o, f.n);
    let bound = j_grid_bound(&f.ws, &sum, &f.hl, f.o, f.n);
    assert!(
        (j_ret - from_grid(out.log.j_nb)).abs() <= bound,
        "log j_nb {} != measured {j_ret} within {bound}",
        from_grid(out.log.j_nb)
    );
}

#[test]
fn learn_fold_nb_refine_reaches_a_round_that_keeps_nothing() {
    let f = Fixture::new(0x4E42_F1D0, 4, 256);
    let mut cur_a = f.p1.clone();
    let mut cur_b = f.p2.clone();
    let mut j_prev = pair_j(&f.ws, &cur_a, &cur_b, &f.hl, f.o, f.n);
    let mut settled = false;
    for _ in 0..8 {
        let out = f.run_from(&cur_a, &cur_b, 2);
        assert!(
            from_grid(out.log.j_nb) <= j_prev,
            "J rose {j_prev} -> {} across runs",
            from_grid(out.log.j_nb)
        );
        j_prev = from_grid(out.log.j_nb);
        if out.log.rounds_kept == 0 {
            // Nothing was kept, so the returned packs are the input packs: the run's map is at a
            // fixed point and j_nb is exactly j_seq.
            assert_eq!(out.p1.s, cur_a.s);
            assert_eq!(out.p1.idx, cur_a.idx);
            assert_eq!(out.p1.r, cur_a.r);
            assert_eq!(out.p1.c, cur_a.c);
            assert_eq!(out.p1.d, cur_a.d);
            assert_eq!(out.p2.s, cur_b.s);
            assert_eq!(out.p2.r, cur_b.r);
            assert_eq!(out.p2.c, cur_b.c);
            assert_eq!(out.log.j_nb, out.log.j_seq);
            settled = true;
            break;
        }
        cur_a = out.p1;
        cur_b = out.p2;
    }
    assert!(
        settled,
        "no run kept zero rounds within 8: the post-pass did not reach a fixed point"
    );
}

#[test]
fn learn_fold_nb_refine_output_stays_in_the_stored_formats() {
    let f = Fixture::new(0x4E42_B17E, 4, 256);
    let out = f.run(2);
    assert!(out.log.rounds_kept >= 1, "fixture kept no round");
    let g = f.n / GROUP;
    for p in [&out.p1, &out.p2] {
        // The column scale is a `6`-bit `quant_c_u8` value: re-quantising it is the identity.
        let (cq, _) = quant_c_u8(&p.c, C_BITS).expect("quant_c_u8");
        assert_eq!(cq.len(), p.c.len());
        for i in 0..p.c.len() {
            assert_eq!(cq[i], p.c[i], "c[{i}] is not on the {C_BITS}-bit grid");
        }
        // The returned row scale is a valid quantiser input. The quantiser is *not* idempotent on
        // its own output in general (a second pass rebuilds the row-mean grid from the decoded
        // values, so it can shift the per-row grid), which is why the format bound below is stated
        // on the values themselves, not on a second pass.
        assert!(quant_r_hier(&p.r, f.o, g).is_ok());
        // The row scale stays in the stored format: every row's entries are a subset of one equally
        // spaced grid of at most 16 points, `r_lo + (k / 15)·r_sp` with `k` an integer in `0..=15`
        // (4-bit codes), so the row writes no value finer than the format carries.
        for row in 0..f.o {
            let mut vals: Vec<f64> = (0..g).map(|j| from_grid(p.r[row * g + j])).collect();
            vals.sort_by(f64::total_cmp);
            vals.dedup();
            assert!(
                vals.len() <= 16,
                "row {row}: {} distinct values exceed the 16 of the 4-bit grid",
                vals.len()
            );
            if vals.len() < 2 {
                continue;
            }
            let lo = vals[0];
            let mut step = f64::INFINITY;
            for w in vals.windows(2) {
                let d = w[1] - w[0];
                if d > 0.0 && d < step {
                    step = d;
                }
            }
            assert!(
                step.is_finite() && step > 0.0,
                "row {row}: no positive step"
            );
            for &v in &vals {
                let k = (v - lo) / step;
                assert!(
                    (k - k.round()).abs() <= 1e-9 * (1.0 + k.abs()),
                    "row {row}: {v} is not on the equal-step grid lo={lo} step={step} (k = {k})"
                );
                assert!(
                    k.round() <= 15.0,
                    "row {row}: grid index {} exceeds 15",
                    k.round()
                );
            }
        }
    }
}

#[test]
fn learn_fold_nb_refine_rounds_zero_returns_the_input() {
    let f = Fixture::new(0x4E42_2E20, 4, 256);
    let out = f.run(0);
    assert_eq!(out.log.rounds_kept, 0);
    assert_eq!(out.log.moves, 0);
    assert!(out.log.per_round.is_empty());
    assert_eq!(out.log.j_nb, out.log.j_seq);
    for (got, want) in [(&out.p1, &f.p1), (&out.p2, &f.p2)] {
        assert_eq!(got.s, want.s);
        assert_eq!(got.idx, want.idx);
        assert_eq!(got.r, want.r);
        assert_eq!(got.c, want.c);
        assert_eq!(got.d, want.d);
    }
    assert_eq!(out.w1, plane_w(pref(&f.p1), f.o, f.n).expect("w1"));
    assert_eq!(out.w2, plane_w(pref(&f.p2), f.o, f.n).expect("w2"));
    let mut ws = out.w1.clone();
    for k in 0..f.o * f.n {
        ws[k] += out.w2[k];
    }
    assert_eq!(
        out.log.j_seq,
        objective(&qw(&f.ws), &ws, &qw(&f.hl), f.o, f.n).expect("J"),
        "j_seq is not the exact-sum objective"
    );
}

#[test]
fn learn_fold_nb_refine_weights_are_the_packs_compose() {
    let f = Fixture::new(0x4E42_C0DE, 4, 256);
    let out = f.run(2);
    assert!(out.log.rounds_kept >= 1, "fixture kept no round");
    // S_e = fwht128(S_rot ⊙ d) of the returned pack; its `compose` with the returned scales is the
    // returned weight — the same expressions the port evaluates when it keeps a round.
    //
    // The port evaluates `S_e` on the `Q32.32` grid: the `s ⊙ d` product is rounded once per entry
    // (`≤ U/2`, plus `|s|·U/2` for `d`'s own boundary conversion, `U = 2⁻³²`), the normalised
    // 128-block transform amplifies an input error by at most `√128` and rounds each output once,
    // and `compose`'s two scale multiplies carry the result by `max|r|·max|c|`. Derived bound:
    // `(√128·U·(1 + max|s·d|) + U)·max|r|·max|c|` — the grid quantum times the condition growth,
    // with `U` a full unit so the two half-unit roundings are covered.
    let u = 2f64.powi(-32);
    let (w1, w2) = (uw(&out.w1), uw(&out.w2));
    for (pack, w) in [(&out.p1, &w1), (&out.p2, &w2)] {
        let mut prod = vec![0.0_f64; f.o * f.n];
        let mut sd_max = 0.0_f64;
        for row in 0..f.o {
            for col in 0..f.n {
                // The port's `s ⊙ d` product: one `Q32.32` round-to-even per entry.
                let sd = from_grid(
                    round_q32_even(i128::from(pack.s[row * f.n + col]) * i128::from(pack.d[col]))
                        .expect("product on the grid"),
                );
                sd_max = sd_max.max(sd.abs());
                prod[row * f.n + col] = sd;
            }
        }
        let s_e = fwht128(&prod).expect("fwht128");
        let composed = uw(&compose(&qw(&s_e), &pack.r, &pack.c, f.o, f.n).expect("compose"));
        let r_max = pack
            .r
            .iter()
            .fold(0.0_f64, |m, &v| m.max(from_grid(v).abs()));
        let c_max = pack
            .c
            .iter()
            .fold(0.0_f64, |m, &v| m.max(from_grid(v).abs()));
        let bound = (128f64.sqrt() * u * (1.0 + sd_max) + u) * r_max * c_max;
        let mut worst = 0.0_f64;
        for (a, b) in composed.iter().zip(w) {
            worst = worst.max((a - b).abs());
        }
        assert!(
            worst <= bound,
            "compose(S_e, r, c) differs from the returned plane weight by {worst} > {bound}"
        );
    }
    let sum = pair_w(&out.p1, &out.p2, f.o, f.n);
    let bound = j_grid_bound(&f.ws, &sum, &f.hl, f.o, f.n);
    let j_meas = objective(&qw(&f.ws), &qw(&sum), &qw(&f.hl), f.o, f.n).expect("J");
    assert!(
        (out.log.j_nb - j_meas).unsigned_abs() as f64 <= bound,
        "log j_nb {} is not the exact-sum objective within {bound}",
        out.log.j_nb
    );
}

#[test]
fn learn_fold_nb_refine_refuses_bad_shapes_and_widths() {
    let f = Fixture::new(0x4E42_EA00, 4, 256);
    let (wsq, hlq) = (qw(&f.ws), qw(&f.hl));
    let good = |ws: &[i64], hl: &[i64], a: &Plane, b: &Plane| {
        nb_refine(ws, hl, pref(a), pref(b), f.o, f.n, None, 0, 2, 2)
    };
    assert!(good(&wsq, &hlq, &f.p1, &f.p2).is_ok());

    // A short `ws`, `hl`, `s`, `idx`, `r`, `c` or `d` is a shape refusal.
    assert_eq!(
        good(&wsq[1..], &hlq, &f.p1, &f.p2).err(),
        Some(NbError::Shape)
    );
    assert_eq!(
        good(&wsq, &hlq[1..], &f.p1, &f.p2).err(),
        Some(NbError::Shape)
    );
    let short_s = Plane {
        s: f.p1.s[1..].to_vec(),
        ..f.p1.clone()
    };
    assert_eq!(
        good(&wsq, &hlq, &short_s, &f.p2).err(),
        Some(NbError::Shape)
    );
    let short_idx = Plane {
        idx: f.p1.idx[1..].to_vec(),
        ..f.p1.clone()
    };
    assert_eq!(
        good(&wsq, &hlq, &short_idx, &f.p2).err(),
        Some(NbError::Shape)
    );
    let short_r = Plane {
        r: f.p1.r[1..].to_vec(),
        ..f.p1.clone()
    };
    assert_eq!(
        good(&wsq, &hlq, &short_r, &f.p2).err(),
        Some(NbError::Shape)
    );
    let short_c = Plane {
        c: f.p1.c[1..].to_vec(),
        ..f.p1.clone()
    };
    assert_eq!(
        good(&wsq, &hlq, &short_c, &f.p2).err(),
        Some(NbError::Shape)
    );
    let short_d = Plane {
        d: f.p1.d[1..].to_vec(),
        ..f.p1.clone()
    };
    assert_eq!(
        good(&wsq, &hlq, &short_d, &f.p2).err(),
        Some(NbError::Shape)
    );

    // A `g_row` of the wrong length is a shape refusal.
    let with_gr = |gr: &[i64]| {
        nb_refine(
            &wsq,
            &hlq,
            pref(&f.p1),
            pref(&f.p2),
            f.o,
            f.n,
            Some(gr),
            0,
            2,
            2,
        )
    };
    assert_eq!(with_gr(&[1_i64; 1]).err(), Some(NbError::Shape));
    assert!(with_gr(&[1_i64; 4]).is_ok());

    // `n` not a multiple of 128 is a width refusal, with the slices scaled to agree.
    let n2 = 100;
    let ws2 = vec![0_i64; f.o * n2];
    let hl2 = vec![0_i64; n2 * n2];
    let p2 = Plane {
        s: vec![0_i64; f.o * n2],
        idx: vec![0_u16; f.o * (n2 / BLOCK)],
        r: vec![0_i64; f.o * (n2 / GROUP)],
        c: vec![0_i64; n2],
        d: vec![0_i64; n2],
    };
    assert_eq!(
        nb_refine(&ws2, &hl2, pref(&p2), pref(&p2), f.o, n2, None, 0, 2, 2).err(),
        Some(NbError::Width)
    );
    // A zero row count is a shape refusal.
    assert_eq!(
        nb_refine(&[], &[], pref(&f.p1), pref(&f.p2), 0, f.n, None, 0, 2, 2).err(),
        Some(NbError::Shape)
    );

    // A pack scale outside the `i64` product range is refused. The word-domain `Plane` cannot carry
    // a NaN or an infinity, so the pack-scale refusal exercised here is an off-grid product: `r`
    // overflows inside `plane_w`'s group scaling, `c` survives `plane_w` and overflows in the entry
    // `J`, so the refusal is stated as the class (a step or ICM refusal), not one variant.
    let bad_r = Plane {
        r: {
            let mut r = f.p1.r.clone();
            r[0] = i64::MAX;
            r
        },
        ..f.p1.clone()
    };
    let err = good(&wsq, &hlq, &bad_r, &f.p2).unwrap_err();
    assert!(
        matches!(err, NbError::Icm(_) | NbError::Steps(_)),
        "an off-grid group scale must be refused, got {err:?}"
    );
    let bad_c = Plane {
        c: {
            let mut c = f.p2.c.clone();
            c[1] = i64::MAX;
            c
        },
        ..f.p2.clone()
    };
    let err = good(&wsq, &hlq, &f.p1, &bad_c).unwrap_err();
    assert!(
        matches!(err, NbError::Icm(_) | NbError::Steps(_)),
        "an off-grid column scale must be refused, got {err:?}"
    );
}

#[test]
fn learn_fold_nb_refine_row_chunk_is_the_documented_constant() {
    // The chunk is the script's `row_chunk` default; one chunk covers every `o` of the fixtures.
    assert_eq!(ROW_CHUNK, 256);
    // The sweep count is the script's `NB_SWEEPS = 2` default, and the column quantiser is `6`-bit.
    assert_eq!(NB_SWEEPS, 2);
    assert_eq!(C_BITS, 6);
    let f = Fixture::new(0x4E42_C4A2, 4, 256);
    assert!(f.o <= ROW_CHUNK);
}

/// `J(Ws, Wp, Hl) = tr(D·Hl·Dᵀ)` on the pre-port exact body (`src/learn/fold/fit/steps.rs` at base
/// `7a56eb47`, sha256 `8b43418206b454182b09372debb939c49d72d12f6e1311a0674b81d7e886c7c8`):
/// `D = Ws − Wp`, `D·Hl` by one exact `i128` contraction per entry after the operands' exact
/// power-of-two shifts, the trace one exact `i128` dot rounded once. Copied in, so the K5 report
/// below compares the limb port against a body that shares no product with `src`.
#[allow(clippy::cast_possible_wrap)] // `bits` is an `ilog2` of a `u128` magnitude, at most 128
fn exact_objective(ws: &[i64], wp: &[i64], hl: &[i64], o: usize, n: usize) -> i64 {
    use apeiron::learn::solve::q32::{ilog2_pos, narrow, round_q32_even, rshift_rne};
    fn sh(v: i64, k: i32) -> i64 {
        match k.signum() {
            1 => narrow(rshift_rne(i128::from(v), k as u32).expect("shift")).expect("narrow"),
            -1 => narrow(i128::from(v).checked_shl((-k) as u32).expect("shift")).expect("narrow"),
            _ => v,
        }
    }
    fn panel(x: &[i64]) -> i32 {
        let m = x
            .iter()
            .fold(0u128, |m, &v| m.max(u128::from(v.unsigned_abs())));
        match ilog2_pos(m) {
            Some(bits) => bits as i32 + 1 - 35,
            None => 0,
        }
    }
    let len = o * n;
    let mut d = vec![0_i64; len];
    for i in 0..len {
        d[i] = ws[i] - wp[i];
    }
    let (ka, kb) = (panel(&d), panel(hl));
    let ds: Vec<i64> = d.iter().map(|&v| sh(v, ka)).collect();
    let hs: Vec<i64> = hl.iter().map(|&v| sh(v, kb)).collect();
    let mut y = vec![0_i64; len];
    for i in 0..o {
        for j in 0..n {
            let mut acc = 0_i128;
            for l in 0..n {
                acc += i128::from(ds[i * n + l]) * i128::from(hs[l * n + j]);
            }
            let s = acc.checked_add(1_i128 << 31).expect("gemm");
            y[i * n + j] = i64::try_from(s >> 32).expect("gemm");
        }
    }
    let y: Vec<i64> = y.iter().map(|&v| sh(v, -(ka + kb))).collect();
    let mut acc = 0_i128;
    for i in 0..len {
        acc += i128::from(d[i]) * i128::from(y[i]);
    }
    round_q32_even(acc).expect("J")
}

/// FNV-1a over the little-endian bytes of a word buffer.
fn fnv_words(d: &mut u64, xs: &[i64]) {
    for &v in xs {
        for b in v.to_le_bytes() {
            *d ^= u64::from(b);
            *d = d.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// FNV-1a over the little-endian bytes of a scalar.
fn fnv_u64(d: &mut u64, v: u64) {
    for b in v.to_le_bytes() {
        *d ^= u64::from(b);
        *d = d.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// The `nb_refine` word digests, re-pinned under §0 K5 after the port of `steps.rs`'s products onto
/// the limb kernel. The pre-port digests were `0xd837_5ee3_af49_df9b` and
/// `0xa037_2afc_ee43_289a`; the K5 table behind the re-pin compares the port's own `J` against the
/// exact pre-port body ([`exact_objective`]) on the same words, per seed: `J(in)` relative
/// `2.9e-7` / `6.2e-7`, `J(out)` relative `3.8e-7` / `1.1e-6`, and the round's accept margin
/// `0.99` / `0.98` relative — six orders above the `J` error, so the kept/rejected decision is the
/// exact arm's. The fixture generator is the one the recording used, kept verbatim.
#[test]
fn learn_fold_nb_refine_words_are_bit_identical_to_the_recorded_wrapper() {
    /// The recorder's LCG (Numerical Recipes constants) and its `[-1, 1)` draw.
    struct Recorder(u64);
    impl Recorder {
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
        fn sym(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64 * 2.0 - 1.0
        }
    }
    /// A symmetric matrix with a dominant diagonal.
    fn spd(n: usize, rng: &mut Recorder) -> Vec<f64> {
        let mut h = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let v = rng.sym();
                h[i * n + j] = v;
                h[j * n + i] = v;
            }
        }
        for i in 0..n {
            let mut off = 0.0_f64;
            for j in 0..n {
                if j != i {
                    off += h[i * n + j].abs();
                }
            }
            h[i * n + i] = off + 1.0 + rng.sym().abs();
        }
        h
    }
    /// A random-code plane: codebook rows, `r = q(0.5 + |sym|)`, the given `c`, `d = 1`.
    fn plane(o: usize, n: usize, c: &[i64], rng: &mut Recorder) -> Plane {
        let nb = n / BLOCK;
        let cb = codebook();
        let mut s = vec![0_i64; o * n];
        let mut idx = vec![0_u16; o * nb];
        for row in 0..o {
            for m in 0..nb {
                let j = (rng.next_u64() % 256) as usize;
                idx[row * nb + m] = j as u16;
                for k in 0..BLOCK {
                    s[row * n + m * BLOCK + k] = cb[j][k];
                }
            }
        }
        Plane {
            s,
            idx,
            r: (0..o * n / GROUP)
                .map(|_| q1(0.5 + rng.sym().abs()))
                .collect(),
            c: c.to_vec(),
            d: vec![1_i64 << 32; n],
        }
    }
    let (o, n) = (8usize, 128usize);
    for (seed, want, moves) in [
        (0x9b0_0001u64, 0xeccb_fce5_c8c7_48a9u64, 170usize),
        (0x9b0_0002, 0xfdce_e4a8_3fa8_86df, 171),
    ] {
        let mut rng = Recorder::new(seed);
        let hl = spd(n, &mut rng);
        let ct: Vec<i64> = (0..n).map(|_| q1(0.5 + rng.sym().abs())).collect();
        let truth1 = plane(o, n, &ct, &mut rng);
        let truth2 = plane(o, n, &ct, &mut rng);
        let p1 = plane(o, n, &vec![1_i64 << 32; n], &mut rng);
        let p2 = plane(o, n, &vec![1_i64 << 32; n], &mut rng);
        let w1 = plane_w(pref(&truth1), o, n).expect("truth1");
        let w2 = plane_w(pref(&truth2), o, n).expect("truth2");
        let ws: Vec<f64> = w1
            .iter()
            .zip(&w2)
            .map(|(&a, &b)| from_grid(a) + from_grid(b) + 0.1 * rng.sym())
            .collect();
        let out = nb_refine(
            &qw(&ws),
            &qw(&hl),
            pref(&p1),
            pref(&p2),
            o,
            n,
            None,
            q1(0.2),
            1,
            2,
        )
        .expect("nb_refine");
        // K5 before the re-pin: `J` of the input pair and of the returned pair, the port's own words
        // against the exact arm, and the round's decision margin.
        let in1 = plane_w(pref(&p1), o, n).expect("input plane 1");
        let in2 = plane_w(pref(&p2), o, n).expect("input plane 2");
        let wp_in: Vec<i64> = in1.iter().zip(&in2).map(|(a, b)| a + b).collect();
        let wp_out: Vec<i64> = out.w1.iter().zip(&out.w2).map(|(a, b)| a + b).collect();
        let hlq = qw(&hl);
        let j_in = exact_objective(&qw(&ws), &wp_in, &hlq, o, n);
        let j_out = exact_objective(&qw(&ws), &wp_out, &hlq, o, n);
        let rel = |a: i64, b: i64| (a - b).abs() as f64 / (1.0 + (b as f64).abs());
        eprintln!(
            "seed={seed:x}: J(in) port {} exact {j_in} rel {:e}; J(out) port {} exact {j_out} rel {:e}; margin {} words ({:e} relative)",
            out.log.j_seq,
            rel(out.log.j_seq, j_in),
            out.log.j_nb,
            rel(out.log.j_nb, j_out),
            out.log.j_seq - out.log.j_nb,
            (out.log.j_seq - out.log.j_nb) as f64 / (1.0 + (j_in as f64).abs()),
        );
        assert!(
            rel(out.log.j_seq, j_in) <= 1e-4 && rel(out.log.j_nb, j_out) <= 1e-4,
            "seed={seed:x}: the port's J is beyond the K5 bound of the exact arm"
        );
        let mut d = 0xcbf2_9ce4_8422_2325u64;
        fnv_words(&mut d, &out.w1);
        fnv_words(&mut d, &out.w2);
        fnv_u64(&mut d, out.log.j_seq as u64);
        fnv_u64(&mut d, out.log.j_nb as u64);
        assert_eq!(d, want, "seed={seed:x}");
        assert_eq!(out.log.rounds_kept, 2, "seed={seed:x}");
        assert_eq!(out.log.moves, moves, "seed={seed:x}");
    }
}
