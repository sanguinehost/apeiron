//! `tests/learn_fold_resq.rs` — the `-resq` production row-scale step for the MLP `up` plane.
//! Grade `tested`.
//!
//! Covers: the fixed point `Wq == W0 ⇒ s = 1`; the `Wq_u = 0.5·W_u0` optimum near the `2.0` clamp
//! and the `[0.5, 2.0]` bound; the solved normal equations; slab-split invariance; the fold of the
//! multiplier into `quant_r_hier`'s row scale for the full and partial (`pk2_top`) planes; and the
//! shape/non-finite refusals.
//!
//! Mutation control (see the lane return): dropping the `+ lam` term from `rhs` in
//! [`resq_solve`](apeiron::learn::fold::layer::resq::resq_solve) turns `teacher_equals_student_gives_unit`
//! red.

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::suboptimal_flops,
    clippy::unreadable_literal,
    reason = "the tests index the row-major `[rows, n]`/`[o, g]` buffers by `i/j/r` and name the \
              carriers as the reference does; the RNG casts a u64 into the mantissa of an f64 in \
              `[0, 1)` by construction"
)]

use apeiron::learn::fold::fit::quant::{quant_r_hier, RHier};
use apeiron::learn::fold::layer::resq::{
    rescale_hier, rescale_hier_top, resq_scale, resq_solve, ResqError, ResqInputs,
};

const INTER: usize = 16;
const HIDDEN: usize = 8;
const N: usize = 128;
const SLAB_ROWS: usize = 24;

/// Deterministic xorshift64 PRNG, uniform in `[-1, 1)`.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 11) as f64 / 9_007_199_254_740_992.0) * 2.0 - 1.0
    }

    /// Row-major `[r, c]` with entries uniform in `[-1, 1)`.
    fn matrix(&mut self, r: usize, c: usize) -> Vec<f64> {
        let mut m = vec![0.0_f64; r * c];
        for v in &mut m {
            *v = self.next_f64();
        }
        m
    }
}

struct Fixture {
    x0: Vec<f64>,
    x1: Vec<f64>,
    w_g0: Vec<f64>,
    w_u0: Vec<f64>,
    w_down: Vec<f64>,
}

fn fixture(seed: u64) -> Fixture {
    let mut rng = Rng::new(seed);
    Fixture {
        x0: rng.matrix(SLAB_ROWS, N),
        x1: rng.matrix(SLAB_ROWS, N),
        w_g0: rng.matrix(INTER, N),
        w_u0: rng.matrix(INTER, N),
        w_down: rng.matrix(HIDDEN, INTER),
    }
}

/// Build inputs with the given student weights.
fn inputs<'a>(
    f: &'a Fixture,
    slabs: &'a [&'a [f64]],
    wq_g: &'a [f64],
    wq_u: &'a [f64],
) -> ResqInputs<'a> {
    ResqInputs {
        x_slabs: slabs,
        n: N,
        w_g0: &f.w_g0,
        w_u0: &f.w_u0,
        wq_g,
        wq_u,
        w_down: &f.w_down,
        inter: INTER,
        hidden: HIDDEN,
    }
}

/// (a) Student equal to teacher (`Wq == W0`): `P_q == P_0`, so `C_mat == P_mat`, `GoC == GoP`, and
/// `(GoP + lam·I)·1 = GoP·1 + lam·1` has the exact solution `s = 1`.
#[test]
fn teacher_equals_student_gives_unit() {
    let f = fixture(0x51ce_0001);
    let slabs: [&[f64]; 2] = [&f.x0, &f.x1];
    let s = resq_scale(&inputs(&f, &slabs, &f.w_g0, &f.w_u0)).expect("shape ok");
    assert_eq!(s.len(), INTER);
    for (i, &v) in s.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-6, "row {i}: s = {v}");
    }
}

/// (b) `Wq_u = 0.5·W_u0`: `P_q = 0.5·P_0`, so `GoC = 2·(G⊙P_mat)` and `s = 2` is the unclamped
/// optimum of the undamped normal system; the 5% ridge pulls the raw solution just below `2`, and the
/// clamp keeps `s_sol` in `[0.5, 2.0]`.
#[test]
fn half_up_weight_optimum_near_two() {
    let f = fixture(0x51ce_0002);
    let slabs: [&[f64]; 2] = [&f.x0, &f.x1];
    let wq_u: Vec<f64> = f.w_u0.iter().map(|v| 0.5 * v).collect();
    let sol = resq_solve(&inputs(&f, &slabs, &f.w_g0, &wq_u)).expect("shape ok");

    // Undamped system: `GoP_u = G⊙P_mat`, `rhs_u = GoC·1`; `GoC = 2·GoP_u`, so `GoP_u·2 = rhs_u`.
    let d = INTER;
    let mut residual = 0.0_f64;
    for i in 0..d {
        let mut lhs = 0.0_f64;
        for j in 0..d {
            let diag = if i == j { sol.lam } else { 0.0 };
            lhs += (sol.gop[i * d + j] - diag) * 2.0;
        }
        let rhs_u = sol.rhs[i] - sol.lam;
        residual = residual.max((lhs - rhs_u).abs());
    }
    assert!(residual < 1e-9, "undamped residual at s = 2: {residual}");

    for (i, &v) in sol.s_raw.iter().enumerate() {
        assert!((v - 2.0).abs() < 0.5, "row {i}: raw = {v}");
    }
    for (i, &v) in sol.s_clamped.iter().enumerate() {
        assert!((0.5..=2.0).contains(&v), "row {i}: clamped = {v}");
    }
}

/// (c) The clamp bound is respected on the raw solution path: the clamped vector is exactly the raw
/// vector clamped, and every entry is within `[0.5, 2.0]`.
#[test]
fn clamp_bound_respected() {
    let f = fixture(0x51ce_0003);
    let slabs: [&[f64]; 2] = [&f.x0, &f.x1];
    // A large down scale drives the raw optimum well past 2.0, so the upper clamp binds.
    let wq_u: Vec<f64> = f.w_u0.iter().map(|v| 0.1 * v).collect();
    let sol = resq_solve(&inputs(&f, &slabs, &f.w_g0, &wq_u)).expect("shape ok");
    let mut upper_binds = false;
    for (&raw, &cl) in sol.s_raw.iter().zip(sol.s_clamped.iter()) {
        assert_eq!(cl, raw.clamp(0.5, 2.0));
        assert!((0.5..=2.0).contains(&cl));
        if raw > 2.0 {
            upper_binds = true;
        }
    }
    assert!(
        upper_binds,
        "the upper clamp should bind for a 0.1x up scale"
    );
}

/// (d) The solved normal equations hold: `(GoP + lam·I)·s_raw == rhs`, residual `< 1e-9`.
#[test]
fn normal_equations_hold() {
    let f = fixture(0x51ce_0004);
    let slabs: [&[f64]; 2] = [&f.x0, &f.x1];
    let sol = resq_solve(&inputs(&f, &slabs, &f.w_g0, &f.w_u0)).expect("shape ok");
    for i in 0..INTER {
        let mut lhs = 0.0_f64;
        for j in 0..INTER {
            lhs += sol.gop[i * INTER + j] * sol.s_raw[j];
        }
        assert!(
            (lhs - sol.rhs[i]).abs() < 1e-9,
            "row {i}: residual = {}",
            lhs - sol.rhs[i]
        );
    }
}

/// (e) Slab split invariance: one 48-row slab and two 24-row slabs give the same solution.
#[test]
fn slab_split_invariance() {
    let f = fixture(0x51ce_0005);
    let split: [&[f64]; 2] = [&f.x0, &f.x1];
    let mut whole = Vec::with_capacity(2 * SLAB_ROWS * N);
    whole.extend_from_slice(&f.x0);
    whole.extend_from_slice(&f.x1);
    let one: [&[f64]; 1] = [&whole];

    let s_split = resq_scale(&inputs(&f, &split, &f.w_g0, &f.w_u0)).expect("shape ok");
    let s_one = resq_scale(&inputs(&f, &one, &f.w_g0, &f.w_u0)).expect("shape ok");
    for i in 0..INTER {
        assert!(
            (s_split[i] - s_one[i]).abs() < 1e-9,
            "row {i}: {} vs {}",
            s_split[i],
            s_one[i]
        );
    }
}

/// (f) `rescale_hier` folds the multiplier into the row scale: `rh = quant_r_hier(rh·s)`, matching a
/// direct call on the product, for the full plane and the gathered (`pk2_top`) partial plane.
#[test]
fn rescale_hier_folds_into_row_scale() {
    let f = fixture(0x51ce_0006);
    let slabs: [&[f64]; 2] = [&f.x0, &f.x1];
    let s = resq_scale(&inputs(&f, &slabs, &f.w_g0, &f.w_u0)).expect("shape ok");

    // Full plane: rh is [o, g] with g = 3.
    let o = INTER;
    let g = 3;
    let mut rng = Rng::new(0x51ce_0007);
    let rh: Vec<f64> = (0..o * g)
        .map(|_| 0.1 + 0.9 * rng.next_f64().abs())
        .collect();
    let folded = rescale_hier(&rh, &s, o, g).expect("shape ok");
    let mut product = vec![0.0_f64; o * g];
    for i in 0..o {
        for j in 0..g {
            product[i * g + j] = rh[i * g + j] * s[i];
        }
    }
    let direct = quant_r_hier(&product, o, g).expect("shape ok");
    assert_same_hier(&folded, &direct);

    // Partial (`pk2_top`) plane: same s gathered at the plane's global rows.
    let rows: [usize; 4] = [0, 3, 7, 15];
    let og = rows.len();
    let rh2: Vec<f64> = (0..og * g)
        .map(|_| 0.1 + 0.9 * rng.next_f64().abs())
        .collect();
    let folded2 = rescale_hier_top(&rh2, &s, &rows, g).expect("shape ok");
    let mut product2 = vec![0.0_f64; og * g];
    for (i, &r) in rows.iter().enumerate() {
        for j in 0..g {
            product2[i * g + j] = rh2[i * g + j] * s[r];
        }
    }
    let direct2 = quant_r_hier(&product2, og, g).expect("shape ok");
    assert_same_hier(&folded2, &direct2);
}

/// (g) Refusals: shape, empty and non-finite on the solve; shape, empty, non-finite and an
/// out-of-range gathered row on the rescale.
#[test]
fn refusals() {
    let f = fixture(0x51ce_0008);
    let slabs: [&[f64]; 2] = [&f.x0, &f.x1];

    // Wrong student weight length.
    let short = vec![0.0_f64; INTER * N - 1];
    assert_eq!(
        resq_scale(&inputs(&f, &slabs, &short, &f.w_u0)).unwrap_err(),
        ResqError::ShapeMismatch
    );

    // A slab whose length is not a multiple of n.
    let ragged = vec![0.0_f64; SLAB_ROWS * N + 5];
    let bad: [&[f64]; 1] = [&ragged];
    assert_eq!(
        resq_scale(&inputs(&f, &bad, &f.w_g0, &f.w_u0)).unwrap_err(),
        ResqError::ShapeMismatch
    );

    // Zero n is an empty domain.
    let empty = ResqInputs {
        x_slabs: &[],
        n: 0,
        w_g0: &f.w_g0,
        w_u0: &f.w_u0,
        wq_g: &f.w_g0,
        wq_u: &f.w_u0,
        w_down: &f.w_down,
        inter: INTER,
        hidden: HIDDEN,
    };
    assert_eq!(resq_scale(&empty).unwrap_err(), ResqError::Empty);

    // Non-finite teacher weight.
    let mut nan_u0 = f.w_u0.clone();
    nan_u0[3] = f64::NAN;
    assert_eq!(
        resq_scale(&inputs(&f, &slabs, &f.w_g0, &nan_u0)).unwrap_err(),
        ResqError::NonFinite
    );

    // rescale_hier: shape and non-finite.
    let rh = vec![1.0_f64; INTER * 2];
    let s = vec![1.0_f64; INTER];
    assert_eq!(
        rescale_hier(&rh, &s[..INTER - 1], INTER, 2).unwrap_err(),
        ResqError::ShapeMismatch
    );
    assert_eq!(rescale_hier(&[], &[], 0, 2).unwrap_err(), ResqError::Empty);
    let mut rh_nan = rh.clone();
    rh_nan[0] = f64::INFINITY;
    assert_eq!(
        rescale_hier(&rh_nan, &s, INTER, 2).unwrap_err(),
        ResqError::NonFinite
    );

    // rescale_hier_top: an out-of-range gathered row is refused.
    let rh2 = vec![1.0_f64; 2 * 2];
    let rows: [usize; 2] = [0, INTER + 5];
    assert_eq!(
        rescale_hier_top(&rh2, &s, &rows, 2).unwrap_err(),
        ResqError::ShapeMismatch
    );
}

/// Compare two hierarchical row-scale results field-for-field.
fn assert_same_hier(a: &RHier, b: &RHier) {
    assert_eq!(a.bits, b.bits);
    assert_eq!(a.k, b.k);
    assert_eq!(a.u6, b.u6);
    assert_eq!(a.sp_idx, b.sp_idx);
    assert_eq!(a.r_lo, b.r_lo);
    assert_eq!(a.r_sp, b.r_sp);
    assert_eq!(a.rq, b.rq);
}
