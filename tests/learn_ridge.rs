//! `tests/learn_ridge.rs` — the streaming Gram + direct SPD ridge solve: `gram_stack` and
//! `boundary_stack` (fold/merge order-freedom), exact unlearning, `ridgeHead_unique` on a
//! rank-deficient `X`, planted-map recovery, closed-form `loo` against a brute-force refit, and the
//! shape / non-SPD refusals. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "the fixed-seed xorshift maps a 24-bit integer field to an f32 in [-1, 1) and the \
              planted target is rounded to f32 by nature; the short `r/i/o` indices are the \
              reference arithmetic's vocabulary. The pedantic lints stay on for the crate."
)]

use apeiron::learn::ridge::{cholesky, solve_spd, FoldTrainer, RidgeError};

/// Deterministic xorshift64 PRNG, uniform in `[-1, 1)`.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

/// A row-major `[rows, di]` input and `[rows, out]` target of U(-1,1) values.
fn data(seed: u64, rows: usize, di: usize, out: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let x: Vec<f32> = (0..rows * di).map(|_| rng.next_f32()).collect();
    let y: Vec<f32> = (0..rows * out).map(|_| rng.next_f32()).collect();
    (x, y)
}

/// Largest relative difference, normalized by `max(|a|, |b|, 1)`.
fn max_rel_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs() / p.abs().max(q.abs()).max(1.0))
        .fold(0.0f64, f64::max)
}

/// `(stack X₁ X₂)ᵀ·stack X₁ X₂ = X₁ᵀX₁ + X₂ᵀX₂` and the boundary twin (`gram_stack`,
/// `boundary_stack`): the accumulate is a monoid morphism, so folding `A` then `B` equals folding
/// `A‖B`, and `merge` of the two fold states equals the same thing.
#[test]
fn gram_stack_is_a_monoid_morphism() {
    let (di, out) = (5usize, 2usize);
    let (na, nb) = (12usize, 9usize);
    let (xa, ya) = data(1, na, di, out);
    let (xb, yb) = data(2, nb, di, out);

    let mut ta = FoldTrainer::new(di, out, 1.0, true);
    ta.fold(&xa, &ya, na).unwrap();
    let mut tb = FoldTrainer::new(di, out, 1.0, true);
    tb.fold(&xb, &yb, nb).unwrap();

    let mut whole = FoldTrainer::new(di, out, 1.0, true);
    whole.fold(&xa, &ya, na).unwrap();
    whole.fold(&xb, &yb, nb).unwrap();

    let mut merged = ta.clone();
    merged.merge(&tb).unwrap();

    let mut gram_sum = ta.drag().to_vec();
    for (g, b) in gram_sum.iter_mut().zip(tb.drag()) {
        *g += b;
    }
    let mut cross_sum = ta.cross_term().to_vec();
    for (c, b) in cross_sum.iter_mut().zip(tb.cross_term()) {
        *c += b;
    }

    assert!(
        max_rel_diff(&gram_sum, whole.drag()) <= 1e-12,
        "gram_stack: fold(A)+fold(B) != fold(A‖B)"
    );
    assert!(
        max_rel_diff(&cross_sum, whole.cross_term()) <= 1e-12,
        "boundary_stack: cross terms are not additive"
    );
    assert!(
        max_rel_diff(merged.drag(), whole.drag()) <= 1e-12,
        "merge U differs from the stacked fold"
    );
    assert!(
        max_rel_diff(merged.cross_term(), whole.cross_term()) <= 1e-12,
        "merge C differs from the stacked fold"
    );
    let total = i64::try_from(na + nb).unwrap();
    assert_eq!(whole.count(), total);
    assert_eq!(merged.count(), total);
}

/// `fold(A); fold(B); unfold(B)` restores `fold(A)` — exact unlearning; the count returns to
/// `|A|`. State and solved head both compared.
#[test]
fn unfold_is_exactly_never_having_folded() {
    let (di, out) = (4usize, 3usize);
    let (na, nb) = (15usize, 7usize);
    let (xa, ya) = data(3, na, di, out);
    let (xb, yb) = data(4, nb, di, out);

    let mut t = FoldTrainer::new(di, out, 1.0, true);
    t.fold(&xa, &ya, na).unwrap();
    let u_ref = t.drag().to_vec();
    let c_ref = t.cross_term().to_vec();
    let head_ref = t.solve(None).unwrap();

    t.fold(&xb, &yb, nb).unwrap();
    t.unfold(&xb, &yb, nb).unwrap();

    assert!(
        max_rel_diff(&u_ref, t.drag()) <= 1e-9,
        "unfold left a U residue"
    );
    assert!(
        max_rel_diff(&c_ref, t.cross_term()) <= 1e-9,
        "unfold left a C residue"
    );
    assert_eq!(t.count(), i64::try_from(na).unwrap());
    let head = t.solve(None).unwrap();
    assert!(max_rel_diff(&head_ref.weight, &head.weight) <= 1e-9);
    assert!(max_rel_diff(&head_ref.bias, &head.bias) <= 1e-9);
}

/// `ridgeHead_unique` on a rank-deficient `X` (a duplicated column): with `λ > 0` the drag is SPD,
/// the solve succeeds, the normal equations `(XᵀX + λI)W = XᵀY` hold, and two different fold orders
/// give the same head.
#[test]
fn ridge_unique_on_rank_deficient_x() {
    let (di, out, rows) = (5usize, 2usize, 40usize);
    let (mut x, y) = data(5, rows, di, out);
    for r in 0..rows {
        x[r * di + 1] = x[r * di]; // duplicate column 0 → XᵀX rank-deficient
    }
    let lam = 0.5;

    let mut t = FoldTrainer::new(di, out, lam, true);
    t.fold(&x, &y, rows).unwrap();
    let head = t
        .solve(None)
        .expect("λ>0 makes the drag SPD, so the solve succeeds on rank-deficient X");

    let d = t.aug_dim();
    let mut a = t.drag().to_vec();
    for i in 0..d {
        a[i * d + i] += lam;
    }
    let c = t.cross_term();
    let mut w = vec![0.0f64; d * out];
    for o in 0..out {
        for i in 0..di {
            w[i * out + o] = head.weight[o * di + i];
        }
        w[di * out + o] = head.bias[o];
    }
    let mut max_res = 0.0f64;
    for i in 0..d {
        for o in 0..out {
            let mut acc = 0.0;
            for k in 0..d {
                acc += a[i * d + k] * w[k * out + o];
            }
            max_res = max_res.max((acc - c[i * out + o]).abs());
        }
    }
    assert!(
        max_res <= 1e-8,
        "normal-equations residual {max_res:e} exceeds 1e-8"
    );

    // The same head from a different fold order (rows reversed).
    let mut xr = vec![0.0f32; rows * di];
    let mut yr = vec![0.0f32; rows * out];
    for r in 0..rows {
        let src = rows - 1 - r;
        xr[r * di..r * di + di].copy_from_slice(&x[src * di..src * di + di]);
        yr[r * out..r * out + out].copy_from_slice(&y[src * out..src * out + out]);
    }
    let mut t2 = FoldTrainer::new(di, out, lam, true);
    t2.fold(&xr, &yr, rows).unwrap();
    let head2 = t2.solve(None).unwrap();
    assert!(
        max_rel_diff(&head.weight, &head2.weight) <= 1e-9
            && max_rel_diff(&head.bias, &head2.bias) <= 1e-9,
        "fold order changed the unique head"
    );
}

/// A noise-free planted linear map is recovered to ≤1e-6 with `λ = 1e-9`.
#[test]
fn recovers_planted_linear_map() {
    let (di, out, rows) = (6usize, 2usize, 1000usize);
    let mut rng = Rng::new(7);
    let wt: Vec<f64> = (0..di * out).map(|_| f64::from(rng.next_f32())).collect();
    let mut x = vec![0.0f32; rows * di];
    let mut y = vec![0.0f32; rows * out];
    for r in 0..rows {
        for i in 0..di {
            x[r * di + i] = rng.next_f32();
        }
        for o in 0..out {
            let mut acc = 0.0f64;
            for i in 0..di {
                acc += f64::from(x[r * di + i]) * wt[i * out + o];
            }
            y[r * out + o] = acc as f32;
        }
    }
    let head = FoldTrainer::fit(&x, &y, rows, di, out, 1e-9, false).unwrap();
    let mut max_err = 0.0f64;
    for o in 0..out {
        for i in 0..di {
            max_err = max_err.max((head.weight[o * di + i] - wt[i * out + o]).abs());
        }
    }
    assert!(max_err <= 1e-6, "planted recovery error {max_err:e}");
}

/// The closed-form `loo` equals a brute-force leave-one-out refit to ≤1e-8.
#[test]
fn closed_form_loo_matches_refit() {
    let (di, out, rows) = (3usize, 1usize, 8usize);
    let (x, y) = data(9, rows, di, out);
    let lam = 0.5;
    let mut t = FoldTrainer::new(di, out, lam, true);
    t.fold(&x, &y, rows).unwrap();
    let fast = t.loo(&x, &y, rows, None).unwrap();

    for r in 0..rows {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..rows {
            if i != r {
                xs.extend_from_slice(&x[i * di..(i + 1) * di]);
                ys.extend_from_slice(&y[i * out..(i + 1) * out]);
            }
        }
        let sub = FoldTrainer::fit(&xs, &ys, rows - 1, di, out, lam, true).unwrap();
        let want = sub.predict(&x[r * di..(r + 1) * di], 1).unwrap();
        for o in 0..out {
            let got = fast[r * out + o];
            assert!(
                (got - want[o]).abs() <= 1e-8,
                "row {r} out {o}: closed form {got} vs refit {}",
                want[o]
            );
        }
    }
}

/// A fixed-seed sparse design: distinct active features per row, plus the densified `[rows, di]`
/// batch whose only nonzeros are `x[r·di + idxs[r,k]] = vals[r,k]`.
fn sparse_design(
    seed: u64,
    rows: usize,
    di: usize,
    nnz: usize,
    out: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let mut idxs = vec![0u32; rows * nnz];
    let mut vals = vec![0.0f32; rows * nnz];
    let mut x = vec![0.0f32; rows * di];
    for r in 0..rows {
        let mut cnt = 0;
        while cnt < nnz {
            let cand = (((rng.next_f32() + 1.0) * 0.5 * di as f32) as u32).min(di as u32 - 1);
            if !idxs[r * nnz..r * nnz + cnt].contains(&cand) {
                idxs[r * nnz + cnt] = cand;
                cnt += 1;
            }
        }
        for k in 0..nnz {
            let v = rng.next_f32();
            vals[r * nnz + k] = v;
            x[r * di + idxs[r * nnz + k] as usize] = v;
        }
    }
    let y: Vec<f32> = (0..rows * out).map(|_| rng.next_f32()).collect();
    (idxs, vals, y, x)
}

/// `fold_sparse` on a sparse design equals `fold` on its densified rows: the same `drag`,
/// `cross_term` and solved head, to ≤1e-9 relative, with and without the bias column.
#[test]
fn fold_sparse_equals_fold_on_densified_rows() {
    for bias in [false, true] {
        for (seed, rows, di, nnz, out) in
            [(21u64, 64usize, 9usize, 4usize, 2usize), (22, 37, 6, 3, 1)]
        {
            let (idxs, vals, y, x) = sparse_design(seed, rows, di, nnz, out);
            let lam = 0.7;
            let mut sp = FoldTrainer::new(di, out, lam, bias);
            sp.fold_sparse(&idxs, &vals, &y, rows, nnz).unwrap();
            let mut dense = FoldTrainer::new(di, out, lam, bias);
            dense.fold(&x, &y, rows).unwrap();

            assert!(
                max_rel_diff(sp.drag(), dense.drag()) <= 1e-9,
                "drag differs (bias={bias})"
            );
            assert!(
                max_rel_diff(sp.cross_term(), dense.cross_term()) <= 1e-9,
                "cross differs (bias={bias})"
            );
            assert_eq!(sp.count(), dense.count());

            let hs = sp.solve(None).unwrap();
            let hd = dense.solve(None).unwrap();
            assert!(
                max_rel_diff(&hs.weight, &hd.weight) <= 1e-9,
                "weight differs (bias={bias})"
            );
            assert!(
                max_rel_diff(&hs.bias, &hd.bias) <= 1e-9,
                "bias differs (bias={bias})"
            );
        }
    }
}

/// `fold_sparse` refuses a wrong-length slice (`ShapeMismatch`) and an index at/over
/// `in_features` (`IndexOutOfRange`).
#[test]
fn fold_sparse_refuses_bad_shape_and_index() {
    let (di, out, rows, nnz) = (5usize, 2usize, 4usize, 3usize);
    let (idxs, vals, y, _x) = sparse_design(23, rows, di, nnz, out);
    let mut t = FoldTrainer::new(di, out, 1.0, true);

    assert_eq!(
        t.fold_sparse(&idxs[..idxs.len() - 1], &vals, &y, rows, nnz),
        Err(RidgeError::ShapeMismatch)
    );
    assert_eq!(
        t.fold_sparse(&idxs, &vals[..vals.len() - 1], &y, rows, nnz),
        Err(RidgeError::ShapeMismatch)
    );
    assert_eq!(
        t.fold_sparse(&idxs, &vals, &y[..y.len() - 1], rows, nnz),
        Err(RidgeError::ShapeMismatch)
    );

    let mut bad = idxs.clone();
    bad[0] = di as u32;
    assert_eq!(
        t.fold_sparse(&bad, &vals, &y, rows, nnz),
        Err(RidgeError::IndexOutOfRange)
    );
}

/// Bad shapes refuse with `ShapeMismatch`; a non-SPD pivot refuses with `NotPositiveDefinite`.
#[test]
fn refuses_bad_shapes_and_non_spd() {
    let (di, out) = (3usize, 2usize);
    let (x, y) = data(11, 6, di, out);

    let mut t = FoldTrainer::new(di, out, 1.0, true);
    assert_eq!(
        t.fold(&x[..x.len() - 1], &y, 6),
        Err(RidgeError::ShapeMismatch)
    );
    assert_eq!(
        t.fold(&x, &y[..y.len() - 1], 6),
        Err(RidgeError::ShapeMismatch)
    );
    let mut t2 = FoldTrainer::new(di, out, 1.0, true);
    assert_eq!(
        t2.merge(&FoldTrainer::new(di + 1, out, 1.0, true)),
        Err(RidgeError::ShapeMismatch)
    );

    // [[1,2],[2,1]] has an eigenvalue -1 → not positive-definite.
    let a = [1.0f64, 2.0, 2.0, 1.0];
    assert_eq!(cholesky(&a, 2), Err(RidgeError::NotPositiveDefinite));
    assert_eq!(cholesky(&a, 3), Err(RidgeError::ShapeMismatch));
    let b = [1.0f64, 0.0, 0.0, 1.0];
    assert_eq!(
        solve_spd(&a, &b, 2, 2),
        Err(RidgeError::NotPositiveDefinite)
    );

    let head = FoldTrainer::new(di, out, 1.0, false).solve(None).unwrap();
    assert_eq!(
        head.predict(&x[..x.len() - 1], 6),
        Err(RidgeError::ShapeMismatch)
    );
}
