//! The per-stage statistics and the Tikhonov solve of the production `fold_gqa` fold-transfer arm.
//!
//! Port of the production configuration of `fold_transfer.py`'s per-stage statistics loop
//! (`~/lanes/port/spec/fold_transfer.py:1538-1690`) and the two helpers it calls,
//! [`ridge_solve`] (`:426-428`) and the metric shift (`:430-432`). The ported flags are the
//! production arm's only: the student-input path (`teacher_in = false`), mode `fold_gqa`,
//! `split_intra_inter = true`, `sink_gate = true`, `guided = false`, `block = true` with
//! `block_alpha = 1.0`, `asym = false`. The teacher-output and R1–R5 paths of the script (`yaqa`,
//! `rrr`, `zellner`, `gin3515`, the two-plane `icm` arms, `use_grow`/`use_res`/`use_align`/`use_skew`,
//! `qmlp`/`mlp2p`, `-nb`, `-glu`) are out of scope; only the branch this arm executes is present.
//!
//! | item | reference |
//! |---|---|
//! | [`sink_token_weights`] | `fold_transfer.py:1592-1597` (`sink_gate` token weights) |
//! | [`StageAccum::new`] / [`StageAccum::add_slab`] | `:1541-1648` (the slab gram loop) |
//! | [`solve_stage`] | `:1652-1682` (the `fold_gqa` solve, `split_intra_inter`) |
//! | [`SolveParams`] | `pv_finish.py:75-84` defaults (`a.*`) |
//! | [`ridge_solve`] | `:426-428` |
//! | [`metric`] | `:430-432` |
//! | [`bias_update`] | `:1880` (`q`/`k`/`v` bias after quantisation) |
//!
//! # What the port reproduces, and what it does not
//!
//! The accumulator is `f64`; the reference accumulates `fp32` (`dt = fp32`). `Hm = ΣᵀXᵀX` is the
//! unweighted input Gram, `Hm_w = Σ Xwᵀ Xw` the sink-weighted one with `Xw = X·w` applied **linearly**
//! (`:1598-1599`, not `√w` — the reference is a weighted least-squares `Σ_t w_t |y_t − W x_t|²`).
//! `Cs[j] = Σ Xwᵀ(Y_j·w)`, `Cts[j] = Σ Xwᵀ(Y_ts_j·w)`, `x_sum = Σ X` unweighted and `y_sum[j] = Σ Y_j`
//! unweighted (`:1610`, `:1646`). The `guided` first-moment weights `a_sl` (`:1587-1589`) are the
//! production arm's absent branch and are not ported.
//!
//! # Evidence grade
//!
//! `tested`: the gate's `cargo test` on the build path (`tests/learn_fold_stage.rs`). No Lean
//! refinement links this file and it is not on the extracted fragment, so nothing here is `proved`.
//!
//! # Trust boundary
//!
//! The layer forward (RMSNorm, attention, the SwiGLU product, the teacher taps `mid`/`out`/`in_mid`)
//! is another lane's port; this module takes its outputs as plain row-major slices and makes no claim
//! about them. Not covered: rustc/LLVM codegen, the fp32 accumulation the reference uses, and any
//! activation the caller does not supply.

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::suboptimal_flops,
    reason = "the flat-slice kernels index row-major `[rows, n]` and `[n, out]` buffers by `i/j/k/o/\
              r`, and convert `usize` counts to `f64`, reading closer to the reference \
              `fold_transfer.py` arithmetic than enumerated iterators would; `StageAccum`/\
              `StageError` in module `stage` repeat the module name by the brief's naming."
)]

use crate::learn::fold::stats::hocapture::Kind;
use crate::learn::ridge::{solve_spd, RidgeError};

/// A refusal from a per-stage statistics or solve operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageError {
    /// A slice length disagrees with the declared shape.
    Shape,
    /// An empty population: a slab with `t < 2` rows per sequence (no median), or a zero token count.
    Empty,
    /// An input carried a NaN or infinite value.
    NonFinite,
    /// The ridged matrix `H + ridge·mean(diag H)·I` was not positive-definite at a `Cholesky` pivot.
    NotPositiveDefinite,
    /// A split intra/inter solve was requested for an output with no `Cts` accumulator.
    MissingCts,
}

impl From<RidgeError> for StageError {
    fn from(e: RidgeError) -> Self {
        match e {
            RidgeError::NotPositiveDefinite => Self::NotPositiveDefinite,
            RidgeError::ShapeMismatch | RidgeError::IndexOutOfRange => Self::Shape,
        }
    }
}

/// The `fold_gqa` solve hyper-parameters, the production arm's `a.*` defaults
/// (`~/lanes/port/spec/pv_finish.py:75-84`, `arm_args`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveParams {
    /// `a.ridge_intra = 0.045` (`pv_finish.py:80`): ridge of the intra-residual solve.
    pub ridge_intra: f64,
    /// `a.ridge_inter = 0.025` (`pv_finish.py:80`): ridge of the inter-residual solve.
    pub ridge_inter: f64,
    /// `a.inter_gamma = 1.0` (`pv_finish.py:80`): weight of `dW_inter` added to `dW_intra`.
    pub inter_gamma: f64,
    /// `a.ridge_w0_gqa = 0.03` (`pv_finish.py:79`): ridge of the direct solve for q/k/v/gate/up.
    pub ridge_w0_gqa: f64,
    /// `a.inter_gamma_in` at the `-gin25` arm (`fold_transfer.py:1450`, `pv_finish.py:80`):
    /// the step on `dW_inter` for the direct outputs. The production arm runs `gin25`, `0.25`.
    pub gin: f64,
    /// `a.sink_tau = 1.5` (`pv_finish.py:81`): row-norm cap on `dW` relative to `W0`.
    pub sink_tau: f64,
    /// The width `896` the ridges are scaled by (`n / 896`, `fold_transfer.py:1670-1680`).
    pub n_ref: f64,
}

impl Default for SolveParams {
    fn default() -> Self {
        Self {
            ridge_intra: 0.045,
            ridge_inter: 0.025,
            inter_gamma: 1.0,
            ridge_w0_gqa: 0.03,
            gin: 0.25,
            sink_tau: 1.5,
            n_ref: 896.0,
        }
    }
}

/// `r_tok > thresh · r_med ? r_med / max(r_tok, 1e-6) : 1` for every token of the slab
/// (`fold_transfer.py:1592-1597`).
///
/// Per token `r`: `r_t = max(‖h_t‖, ‖p_mid‖, ‖p_out‖)`, `r_s = max(‖h_s‖, ‖p_in_mid‖)` (the second
/// term only when `p_in_mid` is given), `r_tok = max(r_t, r_s)`. `r_med` is the median of `r_t` over
/// positions `>= 1` of every sequence in the slab; the reference `torch.median`
/// (`fold_transfer.py:1596`) returns the **lower** middle element on an even count, so the sorted
/// population's element at index `(k-1)/2` (integer division) is taken. All row norms are the L2
/// norm over `width`.
///
/// Inputs are row-major `[seqs·t, width]`; `h_t` is the stage input, `p_mid`/`p_out` the teacher
/// taps of `mid`/`out`, `h_s` the student input and `p_in_mid` the optional student `mid` input. The
/// returned weights are length `seqs·t`.
///
/// # Complexity
///
/// Temporal `O(seqs·t·width + k·log k)` with `k = seqs·(t−1)` (the norms dominate; the sort is the
/// median). Spatial `O(seqs·t + k)`.
///
/// # Errors
///
/// [`StageError::Empty`] if `t < 2` (no median population); [`StageError::Shape`] if a slice length
/// disagrees with `seqs·t·width`; [`StageError::NonFinite`] if any input entry is NaN or infinite.
#[allow(
    clippy::too_many_arguments,
    reason = "the reference passes the three tapped tensors, the stage input, the optional mid \
              input, the slab geometry and the threshold as separate arguments \
              (`fold_transfer.py:1592-1597`); a struct would only relocate them."
)]
pub fn sink_token_weights(
    h_t: &[f64],
    p_mid: &[f64],
    p_out: &[f64],
    h_s: &[f64],
    p_in_mid: Option<&[f64]>,
    seqs: usize,
    t: usize,
    width: usize,
    thresh: f64,
) -> Result<Vec<f64>, StageError> {
    let rows = seqs * t;
    if rows == 0 || width == 0 || t < 2 {
        return Err(StageError::Empty);
    }
    if h_t.len() != rows * width
        || p_mid.len() != rows * width
        || p_out.len() != rows * width
        || h_s.len() != rows * width
    {
        return Err(StageError::Shape);
    }
    if let Some(p) = p_in_mid {
        if p.len() != rows * width {
            return Err(StageError::Shape);
        }
    }
    if !(finite(h_t)
        && finite(p_mid)
        && finite(p_out)
        && finite(h_s)
        && p_in_mid.is_none_or(finite))
    {
        return Err(StageError::NonFinite);
    }
    let mut r_t = vec![0.0f64; rows];
    let mut r_tok = vec![0.0f64; rows];
    for r in 0..rows {
        let base = r * width;
        let rt = norm(&h_t[base..base + width])
            .max(norm(&p_mid[base..base + width]))
            .max(norm(&p_out[base..base + width]));
        let mut rs = norm(&h_s[base..base + width]);
        if let Some(p) = p_in_mid {
            rs = rs.max(norm(&p[base..base + width]));
        }
        r_t[r] = rt;
        r_tok[r] = rt.max(rs);
    }
    // The median population: positions >= 1 of every sequence, `torch.median`'s lower middle.
    let mut med: Vec<f64> = Vec::with_capacity(seqs * (t - 1));
    for s in 0..seqs {
        for p in 1..t {
            med.push(r_t[s * t + p]);
        }
    }
    if med.is_empty() {
        return Err(StageError::Empty);
    }
    med.sort_by(f64::total_cmp);
    let r_med = med[(med.len() - 1) / 2];
    let mut w = vec![0.0f64; rows];
    for r in 0..rows {
        w[r] = if r_tok[r] > thresh * r_med {
            r_med / r_tok[r].max(1e-6)
        } else {
            1.0
        };
    }
    Ok(w)
}

/// The per-stage normal-equation accumulator over slabs of a calibration set
/// (`fold_transfer.py:1541-1648`).
#[derive(Clone, Debug, PartialEq)]
pub struct StageAccum {
    /// Input width `n`.
    pub n: usize,
    /// Output width per stage output (`names`), in stage order.
    pub outs: Vec<usize>,
    /// Whether output `j` carries a `Cts` accumulator (`split_intra_inter` on `o`/`down`).
    pub with_ts: Vec<bool>,
    /// `Hm = Σ XᵀX`, row-major `[n, n]`, unweighted.
    pub hm: Vec<f64>,
    /// `Hm_w = Σ XwᵀXw`, row-major `[n, n]`, `Xw = X·w` (linear weight).
    pub hm_w: Vec<f64>,
    /// `Cs[j] = Σ Xwᵀ(Y_j·w)`, row-major `[n, outs[j]]`.
    pub cs: Vec<Vec<f64>>,
    /// `Cts[j] = Σ Xwᵀ(Y_ts_j·w)`, row-major `[n, outs[j]]`; empty when `with_ts[j]` is `false`.
    pub cts: Vec<Vec<f64>>,
    /// `x_sum = Σ X`, length `n`, unweighted.
    pub x_sum: Vec<f64>,
    /// `y_sum[j] = Σ Y_j`, length `outs[j]`, unweighted.
    pub y_sum: Vec<Vec<f64>>,
    /// The token count `Σ rows` folded in so far.
    pub n_toks: f64,
}

impl StageAccum {
    /// A zeroed accumulator for input width `n` and the stage's output widths `outs`, with
    /// `with_ts[j]` naming the outputs that carry a `Cts` (`o`/`down` under `split_intra_inter`).
    ///
    /// # Complexity
    ///
    /// Temporal and spatial `O(n² + n·Σ outs)`.
    ///
    /// # Errors
    ///
    /// [`StageError::Shape`] if `outs` and `with_ts` differ in length, or `n == 0`.
    pub fn new(n: usize, outs: &[usize], with_ts: &[bool]) -> Result<Self, StageError> {
        if n == 0 || outs.len() != with_ts.len() {
            return Err(StageError::Shape);
        }
        let m = outs.len();
        let mut cs = Vec::with_capacity(m);
        let mut cts = Vec::with_capacity(m);
        let mut y_sum = Vec::with_capacity(m);
        for (&o, &ts) in outs.iter().zip(with_ts) {
            cs.push(vec![0.0f64; n * o]);
            cts.push(if ts { vec![0.0f64; n * o] } else { Vec::new() });
            y_sum.push(vec![0.0f64; o]);
        }
        Ok(Self {
            n,
            outs: outs.to_vec(),
            with_ts: with_ts.to_vec(),
            hm: vec![0.0f64; n * n],
            hm_w: vec![0.0f64; n * n],
            cs,
            cts,
            x_sum: vec![0.0f64; n],
            y_sum,
            n_toks: 0.0,
        })
    }

    /// Fold one slab: `x` is row-major `[rows, n]`, `w_tok` length `rows`, `ys[j]` row-major
    /// `[rows, outs[j]]` and `y_ts[j]` the `o`/`down` teacher-target block (`None` where
    /// `with_ts[j]` is `false`). Accumulates `Hm`, `Hm_w`, `Cs`, `Cts`, `x_sum`, `y_sum` and the
    /// token count as `fold_transfer.py:1590-1647`; the `rows` loop is the dominant cost.
    ///
    /// # Complexity
    ///
    /// Temporal `O(rows·(n² + n·Σ outs))` — the `Hm`/`Hm_w` scatter alone is `O(rows·n²)` and is the
    /// dominant term; spatial `O(1)` beyond the existing accumulators.
    ///
    /// # Errors
    ///
    /// [`StageError::Shape`] if a slice length disagrees with the declared geometry (including a
    /// missing `y_ts[j]` where `with_ts[j]` is `true`); [`StageError::NonFinite`] if any input entry
    /// is NaN or infinite.
    pub fn add_slab(
        &mut self,
        x: &[f64],
        w_tok: &[f64],
        ys: &[&[f64]],
        y_ts: &[Option<&[f64]>],
    ) -> Result<(), StageError> {
        let rows = w_tok.len();
        let n = self.n;
        let m = self.outs.len();
        if x.len() != rows * n || ys.len() != m || y_ts.len() != m {
            return Err(StageError::Shape);
        }
        for j in 0..m {
            if ys[j].len() != rows * self.outs[j] {
                return Err(StageError::Shape);
            }
            if self.with_ts[j] && y_ts[j].is_none_or(|s| s.len() != rows * self.outs[j]) {
                return Err(StageError::Shape);
            }
        }
        if !finite(x) || !finite(w_tok) || ys.iter().any(|s| !finite(s)) {
            return Err(StageError::NonFinite);
        }
        for j in 0..m {
            if self.with_ts[j] {
                if let Some(s) = y_ts[j] {
                    if !finite(s) {
                        return Err(StageError::NonFinite);
                    }
                }
            }
        }
        // Hm = Σ XᵀX and Hm_w = Σ Xwᵀ Xw, Xw = X·w with the weight applied linearly.
        for r in 0..rows {
            let xrow = &x[r * n..r * n + n];
            let w = w_tok[r];
            for i in 0..n {
                self.x_sum[i] += xrow[i];
            }
            for i in 0..n {
                let xi = xrow[i];
                let hmi = &mut self.hm[i * n..i * n + n];
                let hmwi = &mut self.hm_w[i * n..i * n + n];
                for j in 0..n {
                    let (xm_i, xm_j) = (xi * w, xrow[j] * w);
                    hmi[j] += xi * xrow[j];
                    hmwi[j] += xm_i * xm_j;
                }
            }
        }
        // Cs, Cts, y_sum.
        for j in 0..m {
            let out = self.outs[j];
            let yrow_all = ys[j];
            for r in 0..rows {
                let xrow = &x[r * n..r * n + n];
                let w = w_tok[r];
                let yrow = &yrow_all[r * out..r * out + out];
                for i in 0..n {
                    let xwi = xrow[i] * w;
                    let csi = &mut self.cs[j][i * out..i * out + out];
                    for o in 0..out {
                        csi[o] += xwi * (yrow[o] * w);
                    }
                }
            }
            if self.with_ts[j] {
                if let Some(yts) = y_ts[j] {
                    for r in 0..rows {
                        let xrow = &x[r * n..r * n + n];
                        let w = w_tok[r];
                        let yrow = &yts[r * out..r * out + out];
                        for i in 0..n {
                            let xwi = xrow[i] * w;
                            let cti = &mut self.cts[j][i * out..i * out + out];
                            for o in 0..out {
                                cti[o] += xwi * (yrow[o] * w);
                            }
                        }
                    }
                }
            }
            for o in 0..out {
                let mut s = 0.0f64;
                for r in 0..rows {
                    s += yrow_all[r * out + o];
                }
                self.y_sum[j][o] += s;
            }
        }
        self.n_toks += rows as f64;
        Ok(())
    }
}

/// The `fold_gqa` split intra/inter solve (`fold_transfer.py:1652-1682`) for one stage.
///
/// `H_solve = Hm_w`. For [`Kind::O`]/[`Kind::Down`] the residual against the anchor `W0` is split
/// into `C_intra = Cts − H_solve·W0ᵀ` and `C_inter = Cs − Cts`, each solved with
/// [`ridge_solve`] at `ridge_intra·(n/n_ref)` and `ridge_inter·(n/n_ref)`, combined as
/// `dW = dW_intra + inter_gamma·dW_inter` and, under `sink_tau > 0`, row-capped at
/// `sink_tau·max(‖W0_row‖, 1e-6)`. For the other kinds the direct residual `Cs − H_solve·W0ᵀ` is
/// solved at `ridge_w0_gqa·(n/n_ref)` and stepped by `gin`. `Ws[j] = W0[j] + dW` (`[outs[j], n]`).
///
/// # Complexity
///
/// Temporal `O(Σ_j (n³ + n²·outs[j]))` — one `Cholesky`/solve per residual; spatial
/// `O(n² + Σ_j n·outs[j])`.
///
/// # Errors
///
/// [`StageError::Shape`] if `w0`/`kinds` disagree with the accumulator's outputs or a `w0[j]` length
/// is wrong; [`StageError::MissingCts`] if an `o`/`down` output has no `Cts`; and the solve's
/// [`StageError::NotPositiveDefinite`].
pub fn solve_stage(
    accum: &StageAccum,
    w0: &[&[f64]],
    kinds: &[Kind],
    p: &SolveParams,
) -> Result<Vec<Vec<f64>>, StageError> {
    let n = accum.n;
    let m = accum.outs.len();
    if n == 0 || w0.len() != m || kinds.len() != m {
        return Err(StageError::Shape);
    }
    for j in 0..m {
        if w0[j].len() != accum.outs[j] * n {
            return Err(StageError::Shape);
        }
    }
    let h = &accum.hm_w;
    let lam_scale = n as f64 / p.n_ref;
    let mut ws = Vec::with_capacity(m);
    for j in 0..m {
        let out = accum.outs[j];
        let w0j = w0[j];
        let c0 = matmul_h_wt(h, w0j, n, out);
        match kinds[j] {
            Kind::O | Kind::Down => {
                if !accum.with_ts[j] {
                    return Err(StageError::MissingCts);
                }
                let cts = &accum.cts[j];
                let cs = &accum.cs[j];
                let mut c_intra = vec![0.0f64; n * out];
                let mut c_inter = vec![0.0f64; n * out];
                for k in 0..n * out {
                    c_intra[k] = cts[k] - c0[k];
                    c_inter[k] = cs[k] - cts[k];
                }
                let d_intra = ridge_solve(h, &c_intra, n, out, p.ridge_intra * lam_scale)?;
                let d_inter = ridge_solve(h, &c_inter, n, out, p.ridge_inter * lam_scale)?;
                let mut dw = vec![0.0f64; out * n];
                for k in 0..out * n {
                    dw[k] = d_intra[k] + p.inter_gamma * d_inter[k];
                }
                if p.sink_tau > 0.0 {
                    for o in 0..out {
                        let row = &mut dw[o * n..o * n + n];
                        let w0_row = &w0j[o * n..o * n + n];
                        let w0_rn = norm(w0_row).max(1e-6);
                        let dw_rn = norm(row).max(1e-6);
                        if dw_rn > p.sink_tau * w0_rn {
                            let scale = p.sink_tau * w0_rn / dw_rn;
                            for v in row.iter_mut() {
                                *v *= scale;
                            }
                        }
                    }
                }
                let mut wsj = vec![0.0f64; out * n];
                for k in 0..out * n {
                    wsj[k] = w0j[k] + dw[k];
                }
                ws.push(wsj);
            }
            _ => {
                let cs = &accum.cs[j];
                let mut c_inter = vec![0.0f64; n * out];
                for k in 0..n * out {
                    c_inter[k] = cs[k] - c0[k];
                }
                let d_inter = ridge_solve(h, &c_inter, n, out, p.ridge_w0_gqa * lam_scale)?;
                let mut wsj = vec![0.0f64; out * n];
                for k in 0..out * n {
                    wsj[k] = w0j[k] + p.gin * d_inter[k];
                }
                ws.push(wsj);
            }
        }
    }
    Ok(ws)
}

/// The ridge normal-equation solve `ridge_solve(H, C, ridge)` (`fold_transfer.py:426-428`):
/// `Hr = H` with `diag(Hr) += ridge·mean(diag H)`, then `Hr·X = C` by `Cholesky`, returning
/// `Xᵀ` (shape `[out, n]`).
///
/// `h` is row-major `[n, n]`, `c` row-major `[n, out]`; the returned matrix is row-major `[out, n]`.
///
/// # Complexity
///
/// Temporal `O(n³ + n²·out)`; spatial `O(n² + n·out)`.
///
/// # Errors
///
/// [`StageError::Shape`] if `h.len() != n²` or `c.len() != n·out`;
/// [`StageError::NotPositiveDefinite`] if `Hr` is not positive-definite.
pub fn ridge_solve(
    h: &[f64],
    c: &[f64],
    n: usize,
    out: usize,
    ridge: f64,
) -> Result<Vec<f64>, StageError> {
    if n == 0 || h.len() != n * n || c.len() != n * out {
        return Err(StageError::Shape);
    }
    let mean = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let mut hr = h.to_vec();
    let add = ridge * mean;
    for i in 0..n {
        hr[i * n + i] += add;
    }
    // x = Hr⁻¹ C, row-major [n, out]; the reference returns xᵀ ([out, n]).
    let x = solve_spd(&hr, c, n, out)?;
    let mut out_t = vec![0.0f64; out * n];
    for i in 0..n {
        for o in 0..out {
            out_t[o * n + i] = x[i * out + o];
        }
    }
    Ok(out_t)
}

/// The metric matrix `Hl = H` with `diag += ridge·mean(diag H)` (`fold_transfer.py:430-432`).
///
/// Distinct from [`crate::learn::fold::fit::icm::ridged_metric`], which additionally scales the shift
/// by `n / 896`; here the shift is `ridge·mean(diag H)` alone.
///
/// # Complexity
///
/// Temporal and spatial `O(n²)`.
///
/// # Errors
///
/// [`StageError::Shape`] if `hm.len() != n²` or `n == 0`.
pub fn metric(hm: &[f64], n: usize, ridge: f64) -> Result<Vec<f64>, StageError> {
    if n == 0 || hm.len() != n * n {
        return Err(StageError::Shape);
    }
    let mut h = hm.to_vec();
    let mean = (0..n).map(|i| hm[i * n + i]).sum::<f64>() / n as f64;
    let add = ridge * mean;
    for i in 0..n {
        h[i * n + i] += add;
    }
    Ok(h)
}

/// The post-quantisation bias update `db = (y_sum − Wp·x_sum) / n_toks` (`fold_transfer.py:1880`),
/// applied to `q`/`k`/`v` after the quantised weight `Wp` (`[out, n]`) is fixed.
///
/// # Complexity
///
/// Temporal `O(out·n)`; spatial `O(out)`.
///
/// # Errors
///
/// [`StageError::Shape`] if `wp.len() != out·n`, `x_sum`/`y_sum` are empty or `n_toks` is not
/// finite; [`StageError::Empty`] if `n_toks == 0`; [`StageError::NonFinite`] if an input entry is NaN
/// or infinite.
pub fn bias_update(
    y_sum: &[f64],
    x_sum: &[f64],
    wp: &[f64],
    n_toks: f64,
) -> Result<Vec<f64>, StageError> {
    let n = x_sum.len();
    let out = y_sum.len();
    if n == 0 || out == 0 || wp.len() != out * n {
        return Err(StageError::Shape);
    }
    if !n_toks.is_finite() {
        return Err(StageError::Shape);
    }
    if !finite(x_sum) || !finite(y_sum) || !finite(wp) {
        return Err(StageError::NonFinite);
    }
    if n_toks == 0.0 {
        return Err(StageError::Empty);
    }
    let mut db = vec![0.0f64; out];
    for o in 0..out {
        let row = &wp[o * n..o * n + n];
        let mut s = 0.0f64;
        for i in 0..n {
            s += row[i] * x_sum[i];
        }
        db[o] = (y_sum[o] - s) / n_toks;
    }
    Ok(db)
}

/// `H · W0ᵀ` for `h` row-major `[n, n]` and `w0` row-major `[out, n]`; result row-major `[n, out]`.
/// `O(n²·out)` time.
fn matmul_h_wt(h: &[f64], w0: &[f64], n: usize, out: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * out];
    for i in 0..n {
        let hrow = &h[i * n..i * n + n];
        let crow = &mut c[i * out..i * out + out];
        for o in 0..out {
            let wrow = &w0[o * n..o * n + n];
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += hrow[k] * wrow[k];
            }
            crow[o] = acc;
        }
    }
    c
}

/// The L2 norm of a row.
fn norm(row: &[f64]) -> f64 {
    row.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// `true` iff every entry is finite.
fn finite(a: &[f64]) -> bool {
    a.iter().all(|v| v.is_finite())
}
