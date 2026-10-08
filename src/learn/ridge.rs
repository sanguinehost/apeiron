//! Streaming Gram accumulation and direct SPD ridge solve — the closed-form linear head.
//!
//! Training a linear map `Y ≈ X·W` is a solve, not a search. [`FoldTrainer::fold`] accumulates the
//! normal equations — the drag `U = XᵀX` and the cross term `C = XᵀY` — as a `+`-monoid accumulate
//! over rows; [`FoldTrainer::solve`] returns the ridge head `W = (U + λI)⁻¹ C` in one `Cholesky`
//! solve. [`FoldTrainer::unfold`] is the group inverse of `fold` (exact unlearning, no retraining)
//! and [`FoldTrainer::merge`] adds two fold states (order-free sharding).
//!
//! # Lean statements refined
//!
//! This module is the port of `../fold/modelfold/src/foldtrain.rs` (sha256 prefix
//! `a5ff02895756a137`), the reference implementation of Sanguine paper 1's `Train.lean`. The
//! guarantees map to named theorems in
//! `proof/corpus/Substrate/Algebra/Neural/TrainingCollapse.lean` (namespace
//! `Sanguine.TrainingCollapse`), which is the kernel-checked statement each is a pointer to:
//!
//! - [`FoldTrainer::solve`] → `ridgeHead_solves`: `drag X λ * ridgeHead X Y λ = Xᵀ * Y`, and
//!   `ridgeHead_unique`: any `W'` with `drag X λ * W' = Xᵀ * Y` equals `ridgeHead X Y λ` — the
//!   normal-equations solution is the unique ridge head. `drag_posDef` gives positive-definiteness
//!   for `λ > 0`, so the `Cholesky` solve does not fail.
//! - [`FoldTrainer::fold`] / [`FoldTrainer::merge`] → `gram_stack`:
//!   `(stack X₁ X₂)ᵀ * stack X₁ X₂ = X₁ᵀ * X₁ + X₂ᵀ * X₂` — the accumulate is a monoid morphism,
//!   so the solved endpoint is independent of the order blocks were folded.
//!
//! The refinement is a citation to a kernel-checked theorem; the Rust bodies are not extracted or
//! linked, so the port's behavior is `tested` (`tests/learn_ridge.rs`), not `proved`.
//!
//! # Subsumption
//!
//! The empirical best for this role is `fold/modelfold/src/foldtrain.rs` at the cited digest; the
//! CPU [`FoldTrainer`] is that subsumed body. The `GramBackend` trait and its generic parameter are
//! dropped — this crate has no GPU seam here — and the source's `rayon` row parallelism is dropped
//! for a sequential CPU path (no `rayon` dependency). The f64 accumulation and the normal-equations
//! layout are unchanged.
//!
//! # Partiality
//!
//! [`cholesky`] refuses a non-positive pivot with [`RidgeError::NotPositiveDefinite`] rather than
//! panicking; the source's progressive-jitter retry is dropped. Every shape is checked and refused
//! with [`RidgeError::ShapeMismatch`]; no function in this module panics on caller input.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "the flat-slice numeric kernels index row-major `[rows, d]` buffers by `r/i/j/k/o`, \
              which reads closer to the reference `foldtrain.rs` arithmetic than enumerated \
              iterators; the pedantic lints stay on for the rest of the crate."
)]

/// A refusal from a ridge operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RidgeError {
    /// A slice length disagrees with the declared shape (`rows * features`).
    ShapeMismatch,
    /// The drag `U + λI` is not positive-definite at some `Cholesky` pivot. Reachable only with
    /// `λ ≤ 0` on rank-deficient data.
    NotPositiveDefinite,
}

/// A solved linear head `y = W·x + b`, weights row-major `[out, in]`.
#[derive(Clone, Debug)]
pub struct LinearHead {
    /// Input feature count.
    pub in_features: usize,
    /// Output count (targets / classes).
    pub out_features: usize,
    /// Weight matrix, row-major `[out_features, in_features]`.
    pub weight: Vec<f64>,
    /// Bias, length `out_features` (all zeros if the trainer had `bias = false`).
    pub bias: Vec<f64>,
}

impl LinearHead {
    /// Predict logits for a batch of rows `x` (row-major `[rows, in_features]`). Returns row-major
    /// `[rows, out_features]`. `O(rows·out_features·in_features)` time.
    ///
    /// # Errors
    /// [`RidgeError::ShapeMismatch`] if `x.len() != rows * in_features`.
    pub fn predict(&self, x: &[f32], rows: usize) -> Result<Vec<f64>, RidgeError> {
        let (di, out) = (self.in_features, self.out_features);
        if x.len() != rows * di {
            return Err(RidgeError::ShapeMismatch);
        }
        let mut y = vec![0.0f64; rows * out];
        for r in 0..rows {
            let xrow = &x[r * di..r * di + di];
            for o in 0..out {
                let wrow = &self.weight[o * di..o * di + di];
                let mut acc = self.bias[o];
                for i in 0..di {
                    acc += wrow[i] * f64::from(xrow[i]);
                }
                y[r * out + o] = acc;
            }
        }
        Ok(y)
    }

    /// Argmax class per row (the ridge / logistic classifier read-out).
    /// `O(rows·out_features·in_features)` time.
    ///
    /// # Errors
    /// [`RidgeError::ShapeMismatch`] if `x.len() != rows * in_features`.
    pub fn argmax(&self, x: &[f32], rows: usize) -> Result<Vec<usize>, RidgeError> {
        let logits = self.predict(x, rows)?;
        let out = self.out_features;
        Ok((0..rows)
            .map(|r| {
                let s = &logits[r * out..r * out + out];
                let mut best = 0;
                for o in 1..out {
                    if s[o] > s[best] {
                        best = o;
                    }
                }
                best
            })
            .collect())
    }
}

/// Closed-form trainer for a linear map `Y ≈ X·W + b` via the streaming fold.
///
/// Accumulate data with [`fold`](Self::fold) (order-free, exactly removable with
/// [`unfold`](Self::unfold)), then [`solve`](Self::solve) for the unique ridge head.
#[derive(Clone, Debug)]
pub struct FoldTrainer {
    in_features: usize,
    out_features: usize,
    bias: bool,
    d: usize,
    ridge: f64,
    u: Vec<f64>,
    c: Vec<f64>,
    n: i64,
}

impl FoldTrainer {
    /// A fresh trainer. `O(d² + d·out_features)` time and space for the two accumulators.
    #[must_use]
    pub fn new(in_features: usize, out_features: usize, ridge: f64, bias: bool) -> Self {
        let d = in_features + usize::from(bias);
        Self {
            in_features,
            out_features,
            bias,
            d,
            ridge,
            u: vec![0.0; d * d],
            c: vec![0.0; d * out_features],
            n: 0,
        }
    }

    /// Signed sample count folded in so far.
    #[must_use]
    pub fn count(&self) -> i64 {
        self.n
    }

    /// The drag `U = XᵀX` (row-major `[d, d]`, `d = in_features + bias`) — the fold's accumulated
    /// second moment.
    #[must_use]
    pub fn drag(&self) -> &[f64] {
        &self.u
    }

    /// The cross term `C = XᵀY` (row-major `[d, out_features]`) — the fold's other accumulated
    /// moment.
    #[must_use]
    pub fn cross_term(&self) -> &[f64] {
        &self.c
    }

    /// The augmented feature dim `d = in_features + (bias ? 1 : 0)`.
    #[must_use]
    pub fn aug_dim(&self) -> usize {
        self.d
    }

    /// Augment a batch with the bias column (a constant `1.0`) if `bias` is set.
    /// O(rows·d) time and space.
    fn augment(&self, x: &[f32], rows: usize) -> Vec<f64> {
        let (di, d) = (self.in_features, self.d);
        let mut xa = vec![0.0f64; rows * d];
        for r in 0..rows {
            let src = &x[r * di..r * di + di];
            let dst = &mut xa[r * d..r * d + d];
            for i in 0..di {
                dst[i] = f64::from(src[i]);
            }
            if self.bias {
                dst[d - 1] = 1.0;
            }
        }
        xa
    }

    /// Add (`sign = 1.0`) or remove (`sign = -1.0`) a batch's contribution; `dn` is the signed row
    /// delta for the sample count. `O(rows·d·(d+out_features))` time;
    /// `O(d² + d·out_features + rows·d)` space for the accumulators and the augmented batch.
    fn accumulate(
        &mut self,
        x: &[f32],
        y: &[f32],
        rows: usize,
        sign: f64,
        dn: i64,
    ) -> Result<(), RidgeError> {
        if x.len() != rows * self.in_features || y.len() != rows * self.out_features {
            return Err(RidgeError::ShapeMismatch);
        }
        let xa = self.augment(x, rows);
        let (d, out) = (self.d, self.out_features);
        // Gram `u[i*d+j] += sign * Σ_r xa[r*d+i] * xa[r*d+j]`.
        for r in 0..rows {
            let row = &xa[r * d..r * d + d];
            for i in 0..d {
                let xi = sign * row[i];
                let ui = &mut self.u[i * d..i * d + d];
                for j in 0..d {
                    ui[j] += xi * row[j];
                }
            }
        }
        // Cross `c[i*out+o] += sign * Σ_r xa[r*d+i] * y[r*out+o]`.
        for r in 0..rows {
            let xrow = &xa[r * d..r * d + d];
            let yrow = &y[r * out..r * out + out];
            for i in 0..d {
                let xi = sign * xrow[i];
                let ci = &mut self.c[i * out..i * out + out];
                for o in 0..out {
                    ci[o] += xi * f64::from(yrow[o]);
                }
            }
        }
        self.n += dn;
        Ok(())
    }

    /// Accumulate a batch into the normal equations (`gram_stack`, order-free).
    /// `O(rows·d·(d+out_features))` time.
    ///
    /// # Errors
    /// [`RidgeError::ShapeMismatch`] if `x.len() != rows * in_features` or
    /// `y.len() != rows * out_features`.
    pub fn fold(&mut self, x: &[f32], y: &[f32], rows: usize) -> Result<(), RidgeError> {
        let dn = i64::try_from(rows).map_err(|_| RidgeError::ShapeMismatch)?;
        self.accumulate(x, y, rows, 1.0, dn)
    }

    /// Exactly remove a batch — the group inverse of [`fold`](Self::fold). Exact unlearning:
    /// `fold(A); fold(B); unfold(B)` equals `fold(A)` to f64 round-off.
    /// `O(rows·d·(d+out_features))`.
    ///
    /// # Errors
    /// [`RidgeError::ShapeMismatch`] if `x.len() != rows * in_features` or
    /// `y.len() != rows * out_features`.
    pub fn unfold(&mut self, x: &[f32], y: &[f32], rows: usize) -> Result<(), RidgeError> {
        let dn = -i64::try_from(rows).map_err(|_| RidgeError::ShapeMismatch)?;
        self.accumulate(x, y, rows, -1.0, dn)
    }

    /// Combine another fold state by addition (order-free sharding, `gram_stack`).
    /// `O(d² + d·out_features)` time.
    ///
    /// # Errors
    /// [`RidgeError::ShapeMismatch`] if `other` has a different `d` or `out_features`.
    pub fn merge(&mut self, other: &Self) -> Result<(), RidgeError> {
        if self.d != other.d || self.out_features != other.out_features {
            return Err(RidgeError::ShapeMismatch);
        }
        for (a, b) in self.u.iter_mut().zip(&other.u) {
            *a += b;
        }
        for (a, b) in self.c.iter_mut().zip(&other.c) {
            *a += b;
        }
        self.n += other.n;
        Ok(())
    }

    /// The unique ridge head (`ridgeHead_unique`): `W = (U + λI)⁻¹ C`, one SPD solve.
    /// The drag `U + λI` is positive-definite for `λ > 0` (`drag_posDef`), so the `Cholesky` solve
    /// does not fail. `ridge` overrides the trainer's `λ` when `Some`.
    /// `O(d³ + d²·out_features)` time; `O(d² + d·out_features)` space.
    ///
    /// # Errors
    /// [`RidgeError::NotPositiveDefinite`] if `U + λI` is not positive-definite (only possible with
    /// `λ ≤ 0`).
    pub fn solve(&self, ridge: Option<f64>) -> Result<LinearHead, RidgeError> {
        let lam = ridge.unwrap_or(self.ridge);
        let (d, out) = (self.d, self.out_features);
        // A = U + λI (the drag), row-major [d, d].
        let mut a = self.u.clone();
        for i in 0..d {
            a[i * d + i] += lam;
        }
        // Solve A · W = C for W (row-major [d, out]).
        let w = solve_spd(&a, &self.c, d, out)?;
        // W's first `in_features` rows are the weights; the last row (if bias) is the bias.
        let di = self.in_features;
        let mut weight = vec![0.0f64; out * di];
        let mut bias = vec![0.0f64; out];
        for o in 0..out {
            for i in 0..di {
                weight[o * di + i] = w[i * out + o];
            }
            if self.bias {
                bias[o] = w[di * out + o];
            }
        }
        Ok(LinearHead {
            in_features: di,
            out_features: out,
            weight,
            bias,
        })
    }

    /// One-shot: fold all of `(x, y)` and solve. `x` is row-major `[rows, in_features]`, `y`
    /// row-major `[rows, out_features]` (one-hot for classification, continuous for regression).
    /// O(rows·d² + d³) time.
    ///
    /// # Errors
    /// [`RidgeError::ShapeMismatch`] if the slices disagree with the declared shape, or
    /// [`RidgeError::NotPositiveDefinite`] if `U + λI` is not positive-definite.
    pub fn fit(
        x: &[f32],
        y: &[f32],
        rows: usize,
        in_features: usize,
        out_features: usize,
        ridge: f64,
        bias: bool,
    ) -> Result<LinearHead, RidgeError> {
        let mut t = Self::new(in_features, out_features, ridge, bias);
        t.fold(x, y, rows)?;
        t.solve(None)
    }

    /// Exact leave-one-out predictions from ONE factorization: `ŷ_loo(i) = (xᵢᵀW − hᵢ·yᵢ)/(1−hᵢ)`
    /// with leverage `hᵢ = xᵢᵀG⁻¹xᵢ`. Fold once, factorize once, read every held-out prediction at
    /// its coordinate. `x`/`y` must be the same rows, in the same order, that were folded in.
    /// Returns row-major `[rows, out_features]`.
    /// `O(d³ + d²·(out_features+rows) + rows·d·out_features)` time.
    ///
    /// # Errors
    /// [`RidgeError::ShapeMismatch`] if `x`/`y` disagree with the shape;
    /// [`RidgeError::NotPositiveDefinite`] if `U + λI` is not positive-definite.
    pub fn loo(
        &self,
        x: &[f32],
        y: &[f32],
        rows: usize,
        ridge: Option<f64>,
    ) -> Result<Vec<f64>, RidgeError> {
        let (d, out, di) = (self.d, self.out_features, self.in_features);
        if x.len() != rows * di || y.len() != rows * out {
            return Err(RidgeError::ShapeMismatch);
        }
        let lam = ridge.unwrap_or(self.ridge);
        let xa = self.augment(x, rows);
        // A = U + λI, and ONE right-hand side block holding [XᵀY | Xᵀ]: the same `Cholesky` yields
        // both W and G⁻¹xᵢ for every i. One factorization, `out + rows` back-substitutions.
        let mut a = self.u.clone();
        for i in 0..d {
            a[i * d + i] += lam;
        }
        let cols = out + rows;
        let mut b = vec![0.0f64; d * cols];
        for i in 0..d {
            for o in 0..out {
                b[i * cols + o] = self.c[i * out + o];
            }
            for r in 0..rows {
                b[i * cols + out + r] = xa[r * d + i];
            }
        }
        let sol = solve_spd(&a, &b, d, cols)?;

        let mut pred = vec![0.0f64; rows * out];
        for r in 0..rows {
            // leverage hᵣ = xᵣᵀ (G⁻¹xᵣ)
            let mut h = 0.0f64;
            for i in 0..d {
                h += xa[r * d + i] * sol[i * cols + out + r];
            }
            // 1 − h is the shrink factor; it is 0 only for a row the fit reproduces exactly
            // (h = 1), where the held-out prediction is undefined — fall back to the in-sample
            // one rather than divide by zero.
            let denom = 1.0 - h;
            for o in 0..out {
                let mut fitted = 0.0f64;
                for i in 0..d {
                    fitted += xa[r * d + i] * sol[i * cols + o];
                }
                pred[r * out + o] = if denom.abs() < 1e-9 {
                    fitted
                } else {
                    (fitted - h * f64::from(y[r * out + o])) / denom
                };
            }
        }
        Ok(pred)
    }
}

/// Solve the SPD system `A · X = Bmat` for `X`, `a` row-major `[d, d]`, `bmat` row-major `[d, out]`;
/// returns `X` row-major `[d, out]`. `Cholesky` factor then a forward and back substitution per
/// right-hand-side column. O(d³ + d²·out) time; O(d² + d·out) space.
///
/// # Errors
/// [`RidgeError::ShapeMismatch`] if `a.len() != d*d` or `bmat.len() != d*out`;
/// [`RidgeError::NotPositiveDefinite`] if `a` has a non-positive pivot.
pub fn solve_spd(a: &[f64], bmat: &[f64], d: usize, out: usize) -> Result<Vec<f64>, RidgeError> {
    if a.len() != d * d || bmat.len() != d * out {
        return Err(RidgeError::ShapeMismatch);
    }
    let l = cholesky(a, d)?;
    let mut x = bmat.to_vec();
    // Forward: x[i] = (b[i] - Σ_{k<i} L[i,k] x[k]) / L[i,i].
    for i in 0..d {
        let lii = l[i * d + i];
        for o in 0..out {
            let mut s = x[i * out + o];
            for k in 0..i {
                s -= l[i * d + k] * x[k * out + o];
            }
            x[i * out + o] = s / lii;
        }
    }
    // Back: x[i] = (y[i] - Σ_{k>i} L[k,i] x[k]) / L[i,i].
    for i in (0..d).rev() {
        let lii = l[i * d + i];
        for o in 0..out {
            let mut s = x[i * out + o];
            for k in (i + 1)..d {
                s -= l[k * d + i] * x[k * out + o];
            }
            x[i * out + o] = s / lii;
        }
    }
    Ok(x)
}

/// `Cholesky` factor `L` (lower-triangular, row-major `[d, d]`) of SPD `a`. O(d³/3) time;
/// O(d²) space.
///
/// # Errors
/// [`RidgeError::ShapeMismatch`] if `a.len() != d*d`; [`RidgeError::NotPositiveDefinite`] if a
/// diagonal pivot is non-positive or NaN (`drag_posDef` guarantees positivity in exact arithmetic for
/// `λ > 0`, so this is reachable only with `λ ≤ 0` on rank-deficient data).
pub fn cholesky(a: &[f64], d: usize) -> Result<Vec<f64>, RidgeError> {
    if a.len() != d * d {
        return Err(RidgeError::ShapeMismatch);
    }
    let mut l = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if s.is_nan() || s <= 0.0 {
                    return Err(RidgeError::NotPositiveDefinite);
                }
                l[i * d + j] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    Ok(l)
}
