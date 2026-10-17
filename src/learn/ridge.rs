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

    /// The serial lower-triangle accumulation for [`fold_sparse`](Self::fold_sparse): add each row's
    /// `nnz×nnz` support outer product into the **lower** triangle of `u` (the Gram is symmetric, so
    /// `j <= i` is written and the upper is filled once by
    /// [`mirror_lower`](Self::mirror_lower)). The parallel form [`fold_sparse_par`](Self::fold_sparse_par)
    /// uses [`sparse_accumulate_owner`] instead. `O(rows·(nnz² + nnz·out_features))` time; `O(nnz)`
    /// scratch for the converted indices.
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

    /// The owner-partitioned inner kernel of [`fold_sparse_par`](Self::fold_sparse_par): for the `U`
    /// rows `[lo, hi)` and the matching `C` rows, scan samples `[r0, r1)` and apply the pairs whose
    /// first index `a` lies in the range. `u_rows` and `c_rows` are the worker's contiguous row
    /// slices (row `a` starts at local offset `a - lo`). The caller has range-checked every index, so
    /// the conversion cannot fail. Writes the **lower** triangle of `u` (`b <= a`) and the full `c`;
    /// `vals` and `y` are touched only for in-range pairs. Rows are processed in ascending order, so
    /// each entry accumulates in sample order regardless of how `[r0, r1)` is blocked.
    /// `O((r1−r0)·(nnz² + nnz·out))` time, `O(1)` scratch.
    #[allow(
        clippy::too_many_arguments,
        reason = "the kernel is handed the two disjoint accumulator slices plus the read-only sample \
                  arrays and geometry; bundling them into a struct would only move the fields."
    )]
    fn sparse_accumulate_owner(
        u_rows: &mut [f64],
        c_rows: &mut [f64],
        lo: usize,
        hi: usize,
        d: usize,
        out: usize,
        bias: bool,
        bias_row: usize,
        idxs: &[u32],
        vals: &[f32],
        y: &[f32],
        r0: usize,
        r1: usize,
        nnz: usize,
    ) {
        for r in r0..r1 {
            let base = r * nnz;
            let irow = &idxs[base..base + nnz];
            for k in 0..nnz {
                let ik = irow[k] as usize;
                if ik < lo || ik >= hi {
                    continue;
                }
                // `vals` and `y` are read only for the in-range pairs, so an out-of-range sample
                // costs one `idxs` element and nothing else.
                let vk = f64::from(vals[base + k]);
                let uoff = (ik - lo) * d;
                let urow = &mut u_rows[uoff..uoff + d];
                for l in 0..nnz {
                    let il = irow[l] as usize;
                    if il <= ik {
                        urow[il] += vk * f64::from(vals[base + l]);
                    }
                }
                let yrow = &y[r * out..r * out + out];
                let coff = (ik - lo) * out;
                let crow = &mut c_rows[coff..coff + out];
                for o in 0..out {
                    crow[o] += vk * f64::from(yrow[o]);
                }
            }
            if bias && lo <= bias_row && bias_row < hi {
                // The bias row is the last row, so `u[bias, i]` and `u[bias, bias]` are lower entries.
                let vrow = &vals[base..base + nnz];
                let uoff = (bias_row - lo) * d;
                let urow = &mut u_rows[uoff..uoff + d];
                for (k, &raw) in irow.iter().enumerate() {
                    urow[raw as usize] += f64::from(vrow[k]);
                }
                urow[bias_row] += 1.0;
                let yrow = &y[r * out..r * out + out];
                let coff = (bias_row - lo) * out;
                let crow = &mut c_rows[coff..coff + out];
                for o in 0..out {
                    crow[o] += f64::from(yrow[o]);
                }
            }
        }
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

    /// Parallel [`fold_sparse`](Self::fold_sparse). Two accumulator-disjoint strategies, chosen by
    /// [`owner_partition`](Self::owner_partition):
    ///
    /// - **Output-row partition** ([`fold_sparse_par_owner`](Self::fold_sparse_par_owner)): each
    ///   worker owns a contiguous range of `U` rows and the matching `C` rows, re-scans every sample,
    ///   and applies only the pairs whose first index falls in its range. The accumulators are written
    ///   in place — no partial, no merge — and every entry accumulates in sample order, so the result
    ///   is bit-identical to the serial [`fold_sparse`](Self::fold_sparse) and independent of the
    ///   worker count. Chosen when the sparse nonzeros are not large relative to the Gram: the
    ///   `T·rows·nnz` redundant index reads then cost less than the `T·d²`-entry partial zeroing and
    ///   merge they replace.
    /// - **Row partition with partials** ([`fold_sparse_par_partials`](Self::fold_sparse_par_partials)):
    ///   each worker folds a row chunk into a private `U`/`C`, and the partials are summed (the
    ///   `gram_stack` monoid). Chosen when `rows·nnz` dwarfs `d²`, where reading each sample once
    ///   beats re-scanning it `T` times.
    ///
    /// Both paths run on the engine pool, one row range per `width_global()` participant, and every
    /// index is checked before any task writes, so a refusal leaves `self` unmodified. Both paths are
    /// bit-identical to [`fold_sparse`](Self::fold_sparse) up to the documented f64 summation order.
    ///
    /// # Errors
    /// [`RidgeError::ShapeMismatch`] if `idxs.len() != rows·nnz`, `vals.len() != rows·nnz`, or
    /// `y.len() != rows·out_features`; [`RidgeError::IndexOutOfRange`] if some
    /// `idxs[r·nnz + k] >= in_features`.
    ///
    /// # Panics
    /// If a pool task panics it is re-raised by the dispatch; the tasks call only kernels that do not panic on caller input, so
    /// a panic here is an internal bug.
    pub fn fold_sparse_par(
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
        // Range-check every index before a task writes, preserving the serial refusal guarantee
        // (`self` stays unmodified when a bad index is refused). Fanned over the pool so the shared
        // pass is not a serial `rows·nnz` scan ahead of the pooled fold.
        if !Self::indices_in_range(idxs, di) {
            return Err(RidgeError::IndexOutOfRange);
        }
        if Self::owner_partition(d, rows, nnz) {
            self.fold_sparse_par_owner(idxs, vals, y, rows, nnz)
        } else {
            self.fold_sparse_par_partials(idxs, vals, y, rows, nnz)
        }
    }

    /// The accumulator-disjoint strategy for [`fold_sparse_par`](Self::fold_sparse_par). The
    /// output-row partition re-scans every sample once per worker (`T·rows·nnz` index reads) but
    /// removes the partial zeroing and merge (`~T·d²` accumulator entries); the row partition reads
    /// each sample once but pays that merge. The owner path is chosen when the sparse nonzeros are
    /// within `5·d²`. The constant is calibrated on the three race configs: `200k×512`
    /// (`rows·nnz = 1.6M > 5·d² = 1.3M` → row partition, measured 4–5 ms fold vs 7 ms owner);
    /// `500k×1024` (`4M < 5.2M` → owner, measured 20 ms vs 22–27 ms); `1M×256`
    /// (`12M > 0.33M` → row partition, measured 32 ms vs 61 ms owner). `O(1)` time.
    fn owner_partition(d: usize, rows: usize, nnz: usize) -> bool {
        rows.saturating_mul(nnz) <= 5usize.saturating_mul(d).saturating_mul(d)
    }

    /// `true` iff every `idxs` entry is `< di`, checked over the engine pool (each task scans a
    /// contiguous chunk and stops on the first out-of-range entry, clearing a shared flag). The
    /// result validates the whole input before any accumulator write, so a refusal leaves `self`
    /// unmodified. `O(idxs.len())` time.
    fn indices_in_range(idxs: &[u32], di: usize) -> bool {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct Scan<'a> {
            idxs: &'a [u32],
            di: usize,
            ok: &'a AtomicBool,
            chunk: usize,
        }
        impl crate::engine::chart::task::IndexTask for Scan<'_> {
            fn run(&self, t: usize) {
                let lo = t * self.chunk;
                let hi = (lo + self.chunk).min(self.idxs.len());
                for &raw in &self.idxs[lo..hi] {
                    if (raw as usize) >= self.di {
                        self.ok.store(false, Ordering::Relaxed);
                        return;
                    }
                }
            }
        }
        let n = idxs.len();
        if n == 0 {
            return true;
        }
        let ok = AtomicBool::new(true);
        let chunk = n
            .div_ceil(crate::engine::chart::pool::width_global())
            .max(1);
        let tasks = n.div_ceil(chunk);
        crate::engine::chart::pool::run_global(
            tasks,
            &Scan {
                idxs,
                di,
                ok: &ok,
                chunk,
            },
        );
        ok.load(Ordering::Relaxed)
    }

    /// Output-row-partitioned parallel fold ([`fold_sparse_par`](Self::fold_sparse_par)): split the
    /// `d` `U`/`C` rows into one contiguous range per `width_global()` pool participant, scan every
    /// sample per task, and apply the pairs whose first index is in the task's range, in place.
    ///
    /// Cost. Work `O(rows·(nnz² + nnz·out_features))` — the same total applied updates as the serial
    /// fold, split across owners. Scan cost `T·rows·nnz`: every owner re-reads every sample to test
    /// its indices against the owner's range, so the index scan is paid `T` times rather than once;
    /// `vals` and `y` are read only for the in-range pairs. Span
    /// `O(rows·nnz + rows·(nnz² + nnz·out)/T)`: each owner scans all `rows·nnz` indices and applies its
    /// `1/T` share of the in-range updates (uniform distinct-index convention, where the expected
    /// updates to any row are `rows·nnz²/(2d)`). Extra space `O(1)` beyond the trainer's
    /// `O(d² + d·out_features)` accumulators; no per-task partial.
    ///
    /// # Panics
    /// A pool task panic is re-raised by the dispatch; the body does not panic on caller input.
    ///
    /// # Errors
    /// [`RidgeError::ShapeMismatch`] if `rows` exceeds `i64::MAX` (unreachable for a slice input).
    fn fold_sparse_par_owner(
        &mut self,
        idxs: &[u32],
        vals: &[f32],
        y: &[f32],
        rows: usize,
        nnz: usize,
    ) -> Result<(), RidgeError> {
        struct OwnerTask<'a> {
            idxs: &'a [u32],
            vals: &'a [f32],
            y: &'a [f32],
            rows: usize,
            nnz: usize,
            d: usize,
            out: usize,
            bias: bool,
            bias_row: usize,
            per: usize,
        }
        impl crate::engine::chart::task::Task2<f64, f64, RidgeError> for OwnerTask<'_> {
            fn run(&self, t: usize, u: &mut [f64], c: &mut [f64]) -> Result<(), RidgeError> {
                let lo = t * self.per;
                let hi = lo + u.len() / self.d;
                FoldTrainer::sparse_accumulate_owner(
                    u,
                    c,
                    lo,
                    hi,
                    self.d,
                    self.out,
                    self.bias,
                    self.bias_row,
                    self.idxs,
                    self.vals,
                    self.y,
                    0,
                    self.rows,
                    self.nnz,
                );
                Ok(())
            }
        }
        let (d, out) = (self.d, self.out_features);
        // Contiguous row ranges of equal count: under the uniform distinct-index convention the
        // expected number of lower-triangle updates to a row is `rows·nnz²/(2d)`, the same for every
        // row, so equal row counts balance the expected work.
        let per = d
            .div_ceil(crate::engine::chart::pool::width_global())
            .max(1);
        let bias = self.bias;
        let bias_row = d - usize::from(bias);
        crate::engine::chart::pool::for_chunks2(
            &mut self.u,
            per * d,
            &mut self.c,
            per * out,
            &OwnerTask {
                idxs,
                vals,
                y,
                rows,
                nnz,
                d,
                out,
                bias,
                bias_row,
                per,
            },
        )?;
        let dn = i64::try_from(rows).map_err(|_| RidgeError::ShapeMismatch)?;
        self.n += dn;
        // The owners wrote the lower triangle only; fill the upper once for the whole state.
        self.mirror_lower();
        Ok(())
    }

    /// Row-partitioned parallel fold ([`fold_sparse_par`](Self::fold_sparse_par)): split the `rows`
    /// into one contiguous chunk per `width_global()` pool participant, fold each into a private
    /// `U`/`C`, and merge the partials by entrywise addition. The merge is licensed by the Gram
    /// monoid `Sanguine.TrainingCollapse.gram_stack` — the Gram of a stacked batch is the sum of its
    /// blocks' Grams, `(stack X₁ X₂)ᵀ·stack X₁ X₂ = X₁ᵀX₁ + X₂ᵀX₂` (the in-repo chain is
    /// `apeiron_gram.gram_fold_i8_stack`, `proof/corpus/Learn/GramExact.lean`) — so the chunk
    /// partials sum to the whole batch's normal equations, up to f64 summation order.
    ///
    /// Cost. Work `O(rows·(nnz² + nnz·out_features))`; span `O(rows/T·(nnz² + nnz·out) +
    /// T·(d² + d·out))` (the final merge is a pooled reduction); extra space
    /// `T·(d² + d·out_features)` for the private accumulators.
    ///
    /// # Panics
    /// A pool task panic is re-raised by the dispatch; the body does not panic on caller input.
    ///
    /// # Errors
    /// [`RidgeError::ShapeMismatch`] if `rows` exceeds `i64::MAX` (unreachable for a slice input).
    fn fold_sparse_par_partials(
        &mut self,
        idxs: &[u32],
        vals: &[f32],
        y: &[f32],
        rows: usize,
        nnz: usize,
    ) -> Result<(), RidgeError> {
        struct PartialsTask<'a> {
            idxs: &'a [u32],
            vals: &'a [f32],
            y: &'a [f32],
            rows: usize,
            nnz: usize,
            out: usize,
            per: usize,
            first: &'a crate::engine::chart::pool::FirstErr<usize, RidgeError>,
        }
        impl crate::engine::chart::task::ChunkTask<FoldTrainer> for PartialsTask<'_> {
            fn run(&self, t: usize, _start: usize, piece: &mut [FoldTrainer]) {
                let lo = t * self.per;
                let hi = (lo + self.per).min(self.rows);
                let Some(part) = piece.first_mut() else {
                    return;
                };
                if let Err(e) = part.sparse_accumulate(
                    &self.idxs[lo * self.nnz..hi * self.nnz],
                    &self.vals[lo * self.nnz..hi * self.nnz],
                    &self.y[lo * self.out..hi * self.out],
                    hi - lo,
                    self.nnz,
                ) {
                    self.first.record(t, e);
                }
            }
        }
        let (out, di) = (self.out_features, self.in_features);
        let per = rows
            .div_ceil(crate::engine::chart::pool::width_global())
            .max(1);
        let tasks = rows.div_ceil(per).max(1);
        let (in_features, out_features, ridge, bias) = (di, out, self.ridge, self.bias);
        let mut partials: Vec<FoldTrainer> = (0..tasks)
            .map(|_| FoldTrainer::new(in_features, out_features, ridge, bias))
            .collect();
        let first: crate::engine::chart::pool::FirstErr<usize, RidgeError> =
            crate::engine::chart::pool::FirstErr::new();
        crate::engine::chart::pool::chunks_mut_global(
            &mut partials,
            1,
            &PartialsTask {
                idxs,
                vals,
                y,
                rows,
                nnz,
                out,
                per,
                first: &first,
            },
        );
        if let Some((_, e)) = first.take() {
            return Err(e);
        }
        // The reduction over the `T` partials is memory-bound, so fan it over index ranges too.
        self.merge_partials_par(&partials);
        // The partials accumulated the lower triangle only; fill the upper once for the whole state.
        self.mirror_lower();
        Ok(())
    }

    /// Add `parts` into `self` entrywise (the `gram_stack` merge), splitting the `d² + d·out` index
    /// range over one range per `width_global()` pool participant. The serial `merge` re-reads and
    /// rewrites `self.u` once per partial (`T·d²` read-modify-writes); this reads each partial once
    /// and writes `self` once, on the pool. `O(T·(d² + d·out))` time, `O(T·(d² + d·out))` read
    /// traffic, `O(1)` extra space.
    fn merge_partials_par(&mut self, parts: &[Self]) {
        struct MergeTask<'a> {
            srcs: &'a [&'a [f64]],
        }
        impl crate::engine::chart::task::ChunkTask<f64> for MergeTask<'_> {
            fn run(&self, _c: usize, start: usize, dst: &mut [f64]) {
                let n = dst.len();
                for src in self.srcs {
                    for (a, b) in dst.iter_mut().zip(&src[start..start + n]) {
                        *a += b;
                    }
                }
            }
        }
        let us: Vec<&[f64]> = parts.iter().map(|p| p.u.as_slice()).collect();
        let cu = self
            .u
            .len()
            .div_ceil(crate::engine::chart::pool::width_global())
            .max(1);
        crate::engine::chart::pool::chunks_mut_global(&mut self.u, cu, &MergeTask { srcs: &us });
        let cs: Vec<&[f64]> = parts.iter().map(|p| p.c.as_slice()).collect();
        let cc = self
            .c
            .len()
            .div_ceil(crate::engine::chart::pool::width_global())
            .max(1);
        crate::engine::chart::pool::chunks_mut_global(&mut self.c, cc, &MergeTask { srcs: &cs });
        let dn: i64 = parts.iter().map(|p| p.n).sum();
        self.n = self.n.saturating_add(dn);
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
    /// does not fail. `ridge` overrides the trainer's `λ` when `Some`. Uses the blocked, pooled
    /// `Cholesky` ([`solve_spd_blocked`]); each trailing row is updated by exactly one task, so the
    /// head is independent of the pool width.
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
        let w = solve_spd_blocked(&a, &self.c, d, out)?;
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

/// Pack the panel `L21 = l[kend..d, kbase..kbase+kb]` transposed into `pk`, so that for a fixed
/// panel column `p` the panel entries of consecutive trailing rows are contiguous:
/// `pk[p·nr + r] = l[(kend+r)·d + kbase + p]`. This is the layout the register-blocked trailing
/// update reads: its inner loop is a contiguous `dst[j] -= a·src[j]` with a broadcast `a`, which
/// LLVM vectorizes — unlike the strided dot-product reduction it replaces, which the compiler cannot
/// reorder without `fast-math`. `O(kb·nr)` time.
fn pack_panel(
    pk: &mut [f64],
    l: &[f64],
    kbase: usize,
    kend: usize,
    d: usize,
    kb: usize,
    nr: usize,
) {
    for r in 0..nr {
        let src = &l[(kend + r) * d + kbase..(kend + r) * d + kbase + kb];
        for p in 0..kb {
            pk[p * nr + r] = src[p];
        }
    }
}

/// Subtract `L21 · L21ᵀ` from the trailing block `rows` (the `nrows` rows of the working matrix
/// starting at global row `first`, row stride `d`) using the packed panel `pk`. For each row
/// `gi = first + ri`, subtract `Σ_p pk[p·nr + (gi−col0)]·pk[p·nr + j]` from `row[col0 + j]` for
/// `j = 0..=(gi−col0)` — the lower-triangle entries. The inner loop is contiguous in `j` (both the
/// destination row and the packed source), so it lowers to vector multiply-and-subtract; the outer
/// `p` order is fixed, so splitting the rows across workers is a disjoint write with
/// pool-width-independent arithmetic. `O(nrows²·kb)` multiply-adds, the same as the dot-product
/// form, but feeding a vector operation per `(row, p)` instead of a scalar latency-bound reduction.
/// Time and space `O(nrows²·kb)`; `O(1)` scratch.
#[allow(
    clippy::too_many_arguments,
    reason = "the kernel is handed the packed panel plus the trailing-block geometry (row start, \
              row count, stride, column origin, panel width) and the total trailing-row count that \
              indexes the packed panel; a struct would only relocate the fields."
)]
fn trailing_update_packed(
    rows: &mut [f64],
    first: usize,
    nrows: usize,
    pk: &[f64],
    nr: usize,
    d: usize,
    col0: usize,
    kb: usize,
) {
    let off = first - col0;
    for ri in 0..nrows {
        let gi = first + ri;
        let lim = gi - col0 + 1;
        let row = &mut rows[ri * d..ri * d + d];
        let dst = &mut row[col0..col0 + lim];
        for p in 0..kb {
            let a = pk[p * nr + off + ri];
            let src = &pk[p * nr..p * nr + lim];
            for (w, b) in dst.iter_mut().zip(src) {
                *w -= a * b;
            }
        }
    }
}

/// Dispatch the packed trailing update `work[kend..d, kend..d] -= L21 · L21ᵀ` (lower triangle) over
/// the engine pool, one contiguous row range per `width_global()` participant. Every entry is
/// written once by exactly one task with a fixed `p` order, so the factor is bit-identical to the
/// serial body.
///
/// # Complexity
///
/// `O(nr²·kb)` multiply-subtracts split across the pool, `O(1)` extra space.
fn trailing_update_pool(work: &mut [f64], pk: &[f64], kend: usize, d: usize, kb: usize) {
    struct TrailingTask<'a> {
        pk: &'a [f64],
        nr: usize,
        d: usize,
        kend: usize,
        kb: usize,
        per: usize,
    }
    impl crate::engine::chart::task::ChunkTask<f64> for TrailingTask<'_> {
        fn run(&self, t: usize, start: usize, group: &mut [f64]) {
            let first = self.kend + t * self.per;
            let _ = start;
            trailing_update_packed(
                group,
                first,
                group.len() / self.d,
                self.pk,
                self.nr,
                self.d,
                self.kend,
                self.kb,
            );
        }
    }
    let nr = d - kend;
    let per = nr
        .div_ceil(crate::engine::chart::pool::width_global())
        .max(1);
    crate::engine::chart::pool::chunks_mut_global(
        &mut work[kend * d..d * d],
        per * d,
        &TrailingTask {
            pk,
            nr,
            d,
            kend,
            kb,
            per,
        },
    );
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

/// Blocked, cache-friendly right-looking `Cholesky` factor `L` of SPD `a`, with the panel solve and
/// the trailing-block update distributed over the engine pool (one contiguous row range per
/// `width_global()` participant, so the factorisation is independent of the pool width). Blocks of
/// [`CHOLESKY_BLOCK`]. Same factor as [`cholesky`] up to f64 summation order; same refusal
/// semantics. `O(d³)` time; `O(d²)` extra space.
///
/// # Errors
/// [`RidgeError::ShapeMismatch`] if `a.len() != d*d`; [`RidgeError::NotPositiveDefinite`] if a
/// diagonal pivot of a block is non-positive or NaN.
pub fn cholesky_blocked(a: &[f64], d: usize) -> Result<Vec<f64>, RidgeError> {
    if a.len() != d * d {
        return Err(RidgeError::ShapeMismatch);
    }
    let mut work = a.to_vec();
    let mut l = vec![0.0f64; d * d];
    // Scratch for the transposed panel of one block; reused across blocks so the packing is not a
    // per-block allocation (and its page faults).
    let mut pk = vec![0.0f64; CHOLESKY_BLOCK * d];
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
        // multiply-adds, so split the rows evenly across the pool.
        {
            struct PanelTask<'a> {
                work: &'a [f64],
                l11: &'a [f64],
                k: usize,
                d: usize,
                kb: usize,
                kend: usize,
                cp: usize,
            }
            impl crate::engine::chart::task::ChunkTask<f64> for PanelTask<'_> {
                fn run(&self, t: usize, _start: usize, group: &mut [f64]) {
                    panel_solve_rows(
                        group,
                        self.kend + t * self.cp,
                        group.len() / self.d,
                        self.work,
                        self.l11,
                        (self.k, self.d, self.kb),
                    );
                }
            }
            let nr_p = d - kend;
            let cp = nr_p
                .div_ceil(crate::engine::chart::pool::width_global())
                .max(1);
            let (diag, panel) = l.split_at_mut(kend * d);
            let l11: &[f64] = diag;
            crate::engine::chart::pool::chunks_mut_global(
                panel,
                cp * d,
                &PanelTask {
                    work: &work,
                    l11,
                    k,
                    d,
                    kb,
                    kend,
                    cp,
                },
            );
        }
        // (3) Pack the panel `L21 = L[kend..d, k..kend]` (transposed) and run the trailing update
        // `work[kend..d, kend..d] -= L21 · L21ᵀ` (lower triangle), fanned out.
        if kend < d {
            let nr = d - kend;
            pack_panel(&mut pk, &l, k, kend, d, kb, nr);
            trailing_update_pool(&mut work, &pk, kend, d, kb);
        }
        k = kend;
    }
    Ok(l)
}

/// Solve the SPD system `A · X = Bmat` with the blocked, pooled `Cholesky`
/// ([`cholesky_blocked`]) then a forward and a back substitution per right-hand-side column.
/// Same solution and refusal semantics as [`solve_spd`], up to f64 summation order.
/// `O(d³ + d²·out)` time; `O(d² + d·out)` space.
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
    if a.len() != d * d || bmat.len() != d * out {
        return Err(RidgeError::ShapeMismatch);
    }
    let l = cholesky_blocked(a, d)?;
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
