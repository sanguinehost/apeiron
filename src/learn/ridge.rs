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
    /// A sparse feature index is `>= in_features`.
    IndexOutOfRange,
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

    /// Accumulate a **sparse** batch into the normal equations without densifying it — the same
    /// `drag` and `cross_term` [`fold`](Self::fold) produces on the densified rows, in
    /// `O(rows·nnz² + rows·nnz·out_features)` multiply-adds and `O(1)` extra space.
    ///
    /// Row `r` carries `nnz` active features: `idxs[r·nnz + k] : u32` is a column in
    /// `0..in_features`, `vals[r·nnz + k] : f32` its value. The row's Gram contribution is the
    /// outer product of its sparse support, `u[i,j] += Σ_{k,l} vals[k]·vals[l]` placed at
    /// `(idxs[k], idxs[l])` — exactly `x_i·x_j` for the densified row when a row's indices are
    /// distinct (the `make_data` convention). The Gram is symmetric, so only the lower triangle
    /// (`i >= j`) is accumulated and [`mirror_lower`](Self::mirror_lower) fills the upper — half
    /// the scattered stores — and the feature indices are converted and range-checked once per row,
    /// not once per `(k,l)` pair. The bias column, when `bias` is set, gets `u[i,bias] += x_i`,
    /// `u[bias,i] += x_i`, `u[bias,bias] += 1`, and `c[bias,o] += y[o]`, the same additions
    /// `augment` makes (the bias row is the last row, so those are lower entries). No dense
    /// `[rows, d]` row is materialized.
    ///
    /// The `rows·nnz²` term is independent of `d`; the `rows·nnz·out_features` term is the cross
    /// term. Derivation: per row the `nnz×nnz` support outer product is `nnz²` products and the
    /// `nnz·out_features` cross is `nnz·out_features`; the accumulators are the trainer's existing
    /// `O(d² + d·out_features)` fields, so the extra space is one scalar plus an `O(nnz)` index
    /// scratch buffer.
    ///
    /// # Errors
    /// [`RidgeError::ShapeMismatch`] if `idxs.len() != rows·nnz`, `vals.len() != rows·nnz`, or
    /// `y.len() != rows·out_features`; [`RidgeError::IndexOutOfRange`] if some
    /// `idxs[r·nnz + k] >= in_features`.
    pub fn fold_sparse(
        &mut self,
        idxs: &[u32],
        vals: &[f32],
        y: &[f32],
        rows: usize,
        nnz: usize,
    ) -> Result<(), RidgeError> {
        self.sparse_accumulate(idxs, vals, y, rows, nnz)?;
        self.mirror_lower();
        Ok(())
    }

    /// The lower-triangle accumulation shared by [`fold_sparse`](Self::fold_sparse) and its parallel
    /// form: add each row's `nnz×nnz` support outer product into the **lower** triangle of `u` (the
    /// Gram is symmetric, so `j <= i` is written and the upper is filled once by
    /// [`mirror_lower`](Self::mirror_lower)). No mirror happens here, so a parallel partial stays
    /// cheap. `O(rows·(nnz² + nnz·out_features))` time; `O(nnz)` scratch for the converted indices.
    ///
    /// # Errors
    /// [`RidgeError::ShapeMismatch`] on a wrong-length slice; [`RidgeError::IndexOutOfRange`] if
    /// some `idxs[r·nnz + k] >= in_features`.
    fn sparse_accumulate(
        &mut self,
        idxs: &[u32],
        vals: &[f32],
        y: &[f32],
        rows: usize,
        nnz: usize,
    ) -> Result<(), RidgeError> {
        let (d, out, di) = (self.d, self.out_features, self.in_features);
        if idxs.len() != rows * nnz || vals.len() != rows * nnz || y.len() != rows * out {
            return Err(RidgeError::ShapeMismatch);
        }
        let bias_row = d - usize::from(self.bias);
        let mut idx_buf = vec![0usize; nnz];
        for r in 0..rows {
            let irow = &idxs[r * nnz..r * nnz + nnz];
            let vrow = &vals[r * nnz..r * nnz + nnz];
            let yrow = &y[r * out..r * out + out];
            // Convert and range-check the row's indices once; every `(k,l)` pair below reuses them.
            for k in 0..nnz {
                let ik = usize::try_from(irow[k]).map_err(|_| RidgeError::IndexOutOfRange)?;
                if ik >= di {
                    return Err(RidgeError::IndexOutOfRange);
                }
                idx_buf[k] = ik;
            }
            for k in 0..nnz {
                let ik = idx_buf[k];
                let vk = f64::from(vrow[k]);
                // Gram support outer product, lower triangle: `u[idx_k, idx_l] += v_k · v_l`.
                let uk = &mut self.u[ik * d..ik * d + d];
                for l in 0..nnz {
                    let il = idx_buf[l];
                    if il <= ik {
                        uk[il] += vk * f64::from(vrow[l]);
                    }
                }
                // Cross: `c[idx_k, o] += v_k · y_o` (not symmetric, full).
                let ck = &mut self.c[ik * out..ik * out + out];
                for o in 0..out {
                    ck[o] += vk * f64::from(yrow[o]);
                }
            }
            if self.bias {
                // The bias row is the last row, so `u[bias, i]` and `u[bias, bias]` are lower entries.
                let urow = &mut self.u[bias_row * d..bias_row * d + d];
                for k in 0..nnz {
                    urow[idx_buf[k]] += f64::from(vrow[k]);
                }
                urow[bias_row] += 1.0;
                let crow = &mut self.c[bias_row * out..bias_row * out + out];
                for o in 0..out {
                    crow[o] += f64::from(yrow[o]);
                }
            }
        }
        let dn = i64::try_from(rows).map_err(|_| RidgeError::ShapeMismatch)?;
        self.n += dn;
        Ok(())
    }

    /// Copy the lower triangle of `u` into its upper triangle (`u[j,i] = u[i,j]` for `j < i`), so
    /// the full symmetric Gram is available after a lower-triangle-only accumulation. `O(d²)`.
    fn mirror_lower(&mut self) {
        let d = self.d;
        for i in 0..d {
            for j in 0..i {
                self.u[j * d + i] = self.u[i * d + j];
            }
        }
    }

    /// Parallel [`fold_sparse`](Self::fold_sparse): split the `rows` into `threads` contiguous
    /// chunks, fold each into a private `U`/`C`/`n`, and merge the partials by entrywise addition.
    ///
    /// The merge is licensed by the Gram monoid. `Sanguine.TrainingCollapse.gram_stack` — the Gram
    /// of a stacked batch is the sum of its blocks' Grams, `(stack X₁ X₂)ᵀ·stack X₁ X₂ =
    /// X₁ᵀX₁ + X₂ᵀX₂` (the in-repo chain is `apeiron_gram.gram_fold_i8_stack`,
    /// `proof/corpus/Learn/GramExact.lean`) — so the chunk partials sum to the whole batch's normal
    /// equations, up to f64 summation order. With `threads <= 1` this delegates to
    /// [`fold_sparse`](Self::fold_sparse); a refusal is returned before any partial is merged, so a
    /// bad index never leaves `self` half-updated.
    ///
    /// Cost. Work `O(rows·(nnz² + nnz·out_features))`; span `O(rows/threads·(nnz² + nnz·out) +
    /// threads·(d² + d·out))` (the final merge is serial); extra space
    /// `threads·(d² + d·out_features)` for the private accumulators, with `d = aug_dim()`.
    ///
    /// # Errors
    /// [`RidgeError::ShapeMismatch`] if `idxs.len() != rows·nnz`, `vals.len() != rows·nnz`, or
    /// `y.len() != rows·out_features`; [`RidgeError::IndexOutOfRange`] if some
    /// `idxs[r·nnz + k] >= in_features`.
    ///
    /// # Panics
    /// If a worker thread panics; the workers call only [`fold_sparse`](Self::fold_sparse), which
    /// does not panic on caller input, so a panic here is an internal bug.
    pub fn fold_sparse_par(
        &mut self,
        idxs: &[u32],
        vals: &[f32],
        y: &[f32],
        rows: usize,
        nnz: usize,
        threads: usize,
    ) -> Result<(), RidgeError> {
        let (out, di) = (self.out_features, self.in_features);
        if idxs.len() != rows * nnz || vals.len() != rows * nnz || y.len() != rows * out {
            return Err(RidgeError::ShapeMismatch);
        }
        let threads = threads.max(1).min(rows.max(1));
        if threads <= 1 {
            return self.fold_sparse(idxs, vals, y, rows, nnz);
        }
        // Even split of the row range; the first `rows % threads` chunks take one extra row.
        let base = rows / threads;
        let rem = rows % threads;
        let mut bounds = Vec::with_capacity(threads + 1);
        bounds.push(0usize);
        let mut acc = 0usize;
        for t in 0..threads {
            acc += base + usize::from(t < rem);
            bounds.push(acc);
        }
        let (in_features, out_features, ridge, bias) = (di, out, self.ridge, self.bias);
        let partials = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let (lo, hi) = (bounds[t], bounds[t + 1]);
                    scope.spawn(move || {
                        let mut part = FoldTrainer::new(in_features, out_features, ridge, bias);
                        part.sparse_accumulate(
                            &idxs[lo * nnz..hi * nnz],
                            &vals[lo * nnz..hi * nnz],
                            &y[lo * out_features..hi * out_features],
                            hi - lo,
                            nnz,
                        )
                        .map(|()| part)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("fold_sparse_par worker panicked"))
                .collect::<Result<Vec<FoldTrainer>, RidgeError>>()
        })?;
        // Merge only after every chunk succeeded, so `self` is never half-updated on a refusal.
        // The reduction over the `T` partials is memory-bound, so fan it over index ranges too.
        self.merge_partials_par(&partials, threads);
        // The partials accumulated the lower triangle only; fill the upper once for the whole state.
        self.mirror_lower();
        Ok(())
    }

    /// Add `parts` into `self` entrywise (the `gram_stack` merge), splitting the `d² + d·out` index
    /// range over up to `threads` scoped workers. The serial `merge` re-reads and rewrites `self.u`
    /// once per partial (`T·d²` read-modify-writes); this reads each partial once and writes `self`
    /// once, in parallel. `O(T·(d² + d·out))` time, `O(T·(d² + d·out))` read traffic, `O(1)` extra
    /// space.
    fn merge_partials_par(&mut self, parts: &[Self], threads: usize) {
        let merge_threads = threads.max(1);
        let cu = self.u.len().div_ceil(merge_threads).max(1);
        std::thread::scope(|scope| {
            for (ci, dst) in self.u.chunks_mut(cu).enumerate() {
                let off = ci * cu;
                scope.spawn(move || {
                    for p in parts {
                        let n = dst.len();
                        for (a, b) in dst.iter_mut().zip(&p.u[off..off + n]) {
                            *a += b;
                        }
                    }
                });
            }
        });
        let cc = self.c.len().div_ceil(merge_threads).max(1);
        std::thread::scope(|scope| {
            for (ci, dst) in self.c.chunks_mut(cc).enumerate() {
                let off = ci * cc;
                scope.spawn(move || {
                    for p in parts {
                        let n = dst.len();
                        for (a, b) in dst.iter_mut().zip(&p.c[off..off + n]) {
                            *a += b;
                        }
                    }
                });
            }
        });
        let dn: i64 = parts.iter().map(|p| p.n).sum();
        self.n += dn;
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
    /// does not fail. `ridge` overrides the trainer's `λ` when `Some`. Uses the blocked, parallel
    /// `Cholesky` ([`solve_spd_blocked_par`]) with the process's available parallelism.
    /// `O(d³ + d²·out_features)` time; `O(d² + d·out_features)` space.
    ///
    /// # Errors
    /// [`RidgeError::NotPositiveDefinite`] if `U + λI` is not positive-definite (only possible with
    /// `λ ≤ 0`).
    pub fn solve(&self, ridge: Option<f64>) -> Result<LinearHead, RidgeError> {
        let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        self.solve_par(ridge, threads)
    }

    /// [`solve`](Self::solve) with an explicit thread count for the blocked `Cholesky` factor; the
    /// factorisation is thread-count-independent (each trailing row is updated by exactly one
    /// worker in a fixed order), so `threads` changes only the wall-clock cost, not the head.
    /// `O(d³ + d²·out_features)` time; `O(d² + d·out_features)` space.
    ///
    /// # Errors
    /// [`RidgeError::NotPositiveDefinite`] if `U + λI` is not positive-definite (only possible with
    /// `λ ≤ 0`).
    pub fn solve_par(&self, ridge: Option<f64>, threads: usize) -> Result<LinearHead, RidgeError> {
        let lam = ridge.unwrap_or(self.ridge);
        let (d, out) = (self.d, self.out_features);
        // A = U + λI (the drag), row-major [d, d].
        let mut a = self.u.clone();
        for i in 0..d {
            a[i * d + i] += lam;
        }
        // Solve A · W = C for W (row-major [d, out]).
        let w = solve_spd_blocked_par(&a, &self.c, d, out, threads)?;
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

/// Cache block width for the right-looking blocked `Cholesky` factor. 64 f64 = 512 B, one cache
/// line per row of a block.
const CHOLESKY_BLOCK: usize = 64;

/// The panel geometry shared by one trailing update: `l` is the factor so far, the panel `L21`
/// occupies `l[·, kbase .. kbase+kb]`, and the updated trailing block's columns start at `col0`.
struct TrailingCtx<'a> {
    l: &'a [f64],
    kbase: usize,
    col0: usize,
    d: usize,
    kb: usize,
}

/// Subtract `L21 · L21ᵀ` from the trailing block `rows` (the `nrows` rows of the working matrix
/// starting at global row `first`, row stride `d`). Every row updates its lower-triangle entries
/// `col0 ..= gi`, so splitting the rows across workers is a disjoint write with identical
/// arithmetic per entry (thread-count-independent). `O(nrows²·kb)` multiply-adds.
fn trailing_update(rows: &mut [f64], first: usize, nrows: usize, ctx: &TrailingCtx) {
    let (d, kb) = (ctx.d, ctx.kb);
    for ri in 0..nrows {
        let gi = first + ri;
        let row = &mut rows[ri * d..ri * d + d];
        for gj in ctx.col0..=gi {
            let mut s = 0.0f64;
            for p in 0..kb {
                s += ctx.l[gi * d + ctx.kbase + p] * ctx.l[gj * d + ctx.kbase + p];
            }
            row[gj] -= s;
        }
    }
}

/// Dispatch the trailing update `work[kend..d, kend..d] -= L21 · L21ᵀ` (lower triangle) to scoped
/// workers. Rows are split contiguously and weighted by their triangular work (`ri + 1`
/// multiply-adds), so no worker carries ~2x the mean and every entry is written once with
/// thread-count-independent arithmetic.
fn trailing_update_par(
    work: &mut [f64],
    l: &[f64],
    k: usize,
    kend: usize,
    d: usize,
    kb: usize,
    threads: usize,
) {
    let nr = d - kend;
    let nthreads = threads.min(nr);
    let ctx = TrailingCtx {
        l,
        kbase: k,
        col0: kend,
        d,
        kb,
    };
    let trailing = &mut work[kend * d..d * d];
    if nthreads == 1 {
        trailing_update(trailing, kend, nr, &ctx);
        return;
    }
    let total = nr * (nr + 1) / 2;
    let target = total.div_ceil(nthreads);
    let mut lens: Vec<usize> = Vec::with_capacity(nthreads);
    let (mut cur, mut acc) = (0usize, 0usize);
    for ri in 0..nr {
        cur += 1;
        acc += ri + 1;
        if acc >= target && lens.len() + 1 < nthreads {
            lens.push(cur);
            cur = 0;
            acc = 0;
        }
    }
    if cur > 0 {
        lens.push(cur);
    }
    let mut rest: &mut [f64] = trailing;
    let mut groups: Vec<(usize, &mut [f64])> = Vec::with_capacity(lens.len());
    let mut local = 0usize;
    for len in lens {
        let (head, tail) = rest.split_at_mut(len * d);
        groups.push((kend + local, head));
        local += len;
        rest = tail;
    }
    std::thread::scope(|scope| {
        for (first, group) in groups {
            let ctx = &ctx;
            scope.spawn(move || {
                trailing_update(group, first, group.len() / d, ctx);
            });
        }
    });
}

/// Forward-substitution panel solve `L[first .. first+nrows, k .. k+kb]` of the blocked
/// `Cholesky`: `L[gi, k+j] = (work[gi, k+j] − Σ_{p<j} L[gi, k+p]·L[k+j, k+p]) / L[k+j, k+j]` for
/// each row `gi`; the rows are independent. `work` is row-major `[d, d]`; `l11` holds the factor's
/// rows before `first` (the diagonal block rows `k..k+kb`). `O(nrows·kb²)` multiply-adds.
fn panel_solve_rows(
    rows: &mut [f64],
    first: usize,
    nrows: usize,
    work: &[f64],
    l11: &[f64],
    geo: (usize, usize, usize),
) {
    let (k, d, kb) = geo;
    for ri in 0..nrows {
        let gi = first + ri;
        let row = &mut rows[ri * d..ri * d + d];
        for j in 0..kb {
            let gj = k + j;
            let mut s = work[gi * d + gj];
            for p in 0..j {
                s -= row[k + p] * l11[gj * d + k + p];
            }
            row[gj] = s / l11[gj * d + gj];
        }
    }
}

/// Blocked, cache-friendly right-looking `Cholesky` factor `L` of SPD `a`, single-threaded.
/// Same factor as [`cholesky`] up to f64 summation order; same refusal semantics.
/// `O(d³)` time; `O(d²)` extra space for the working copy and `L`.
///
/// # Errors
/// [`RidgeError::ShapeMismatch`] if `a.len() != d*d`; [`RidgeError::NotPositiveDefinite`] if a
/// diagonal pivot of a block is non-positive or NaN.
pub fn cholesky_blocked(a: &[f64], d: usize) -> Result<Vec<f64>, RidgeError> {
    cholesky_blocked_par(a, d, 1)
}

/// Blocked right-looking `Cholesky` factor with the trailing-block update distributed over
/// `threads` scoped workers (one contiguous row range each, so the factorisation is
/// thread-count-independent). Blocks of [`CHOLESKY_BLOCK`]; `threads <= 1` runs serial.
/// `O(d³)` time; `O(d²)` extra space.
///
/// # Errors
/// [`RidgeError::ShapeMismatch`] if `a.len() != d*d`; [`RidgeError::NotPositiveDefinite`] if a
/// diagonal pivot of a block is non-positive or NaN.
pub fn cholesky_blocked_par(a: &[f64], d: usize, threads: usize) -> Result<Vec<f64>, RidgeError> {
    if a.len() != d * d {
        return Err(RidgeError::ShapeMismatch);
    }
    let mut work = a.to_vec();
    let mut l = vec![0.0f64; d * d];
    let threads = threads.max(1);
    let mut k = 0usize;
    while k < d {
        let kb = (d - k).min(CHOLESKY_BLOCK);
        // (1) Factor the diagonal block `L[k..k+kb, k..k+kb]` (lower triangle) from `work`.
        for i in 0..kb {
            let gi = k + i;
            for j in 0..=i {
                let gj = k + j;
                let mut s = work[gi * d + gj];
                for p in 0..j {
                    s -= l[gi * d + k + p] * l[gj * d + k + p];
                }
                if i == j {
                    if s.is_nan() || s <= 0.0 {
                        return Err(RidgeError::NotPositiveDefinite);
                    }
                    l[gi * d + gj] = s.sqrt();
                } else {
                    l[gi * d + gj] = s / l[gj * d + gj];
                }
            }
        }
        let kend = k + kb;
        // (2) Panel solve `L[kend..d, k..kend]`: rows are independent and each costs `kb(kb-1)/2`
        // multiply-adds, so split the rows evenly across workers.
        {
            let nr_p = d - kend;
            let np = threads.min(nr_p);
            let (diag, panel) = l.split_at_mut(kend * d);
            let l11: &[f64] = diag;
            if np <= 1 {
                panel_solve_rows(panel, kend, nr_p, &work, l11, (k, d, kb));
            } else {
                let cp = nr_p.div_ceil(np).max(1);
                std::thread::scope(|scope| {
                    for (ci, group) in panel.chunks_mut(cp * d).enumerate() {
                        let first = kend + ci * cp;
                        let l11: &[f64] = l11;
                        let work_ref: &[f64] = &work;
                        scope.spawn(move || {
                            panel_solve_rows(
                                group,
                                first,
                                group.len() / d,
                                work_ref,
                                l11,
                                (k, d, kb),
                            );
                        });
                    }
                });
            }
        }
        // (3) Trailing update `work[kend..d, kend..d] -= L21 · L21ᵀ` (lower triangle), fanned out.
        if kend < d {
            trailing_update_par(&mut work, &l, k, kend, d, kb, threads);
        }
        k = kend;
    }
    Ok(l)
}

/// Solve the SPD system `A · X = Bmat` with the blocked, single-threaded `Cholesky`
/// ([`cholesky_blocked`]). Same solution and refusal semantics as [`solve_spd`], up to f64
/// summation order. `O(d³ + d²·out)` time; `O(d² + d·out)` space.
///
/// # Errors
/// [`RidgeError::ShapeMismatch`] if `a.len() != d*d` or `bmat.len() != d*out`;
/// [`RidgeError::NotPositiveDefinite`] if `a` has a non-positive pivot.
pub fn solve_spd_blocked(
    a: &[f64],
    bmat: &[f64],
    d: usize,
    out: usize,
) -> Result<Vec<f64>, RidgeError> {
    solve_spd_blocked_par(a, bmat, d, out, 1)
}

/// Solve the SPD system `A · X = Bmat` with the blocked, parallel `Cholesky`
/// ([`cholesky_blocked_par`]) then a forward and a back substitution per right-hand-side column.
/// Same solution and refusal semantics as [`solve_spd`], up to f64 summation order.
/// `O(d³ + d²·out)` time; `O(d² + d·out)` space.
///
/// # Errors
/// [`RidgeError::ShapeMismatch`] if `a.len() != d*d` or `bmat.len() != d*out`;
/// [`RidgeError::NotPositiveDefinite`] if `a` has a non-positive pivot.
pub fn solve_spd_blocked_par(
    a: &[f64],
    bmat: &[f64],
    d: usize,
    out: usize,
    threads: usize,
) -> Result<Vec<f64>, RidgeError> {
    if a.len() != d * d || bmat.len() != d * out {
        return Err(RidgeError::ShapeMismatch);
    }
    let l = cholesky_blocked_par(a, d, threads)?;
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
