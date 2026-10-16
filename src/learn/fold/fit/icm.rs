//! Block iterated-conditional-modes (ICM) refinement of the one-bit fold transfer's `E8 ∪ O8` planes
//! in the `(r, c, d)`-scaled Hadamard–Gram metric.
//!
//! Port of `tools/onebit/fold_transfer.py` (removed; at `2f15d6a9`) lines 585–720:
//! [`e8_icm_step`] (585–612), [`plane_w`] and [`j_rows`] (615–623), [`e8_joint_icm_2plane`] (626–690)
//! and [`r_step2`] (693–712). A plane is the pack `(s, r, c, d)` whose weight is
//! `plane_w = fwht128(s·d) · r_rep · c`; the stored code of every 8-column block is one of the 256
//! rows of the [`crate::learn::code::e8`] codebook, carried as [`Plane::idx`] (`u16`) next to its
//! materialised real values [`Plane::s`].
//!
//! The metric quantity is `J(Ws, Wp, Hl) = Σ_o (Ws_o − Wp_o)^T Hl (Ws_o − Wp_o)` ([`j_rows`]). Both
//! ICM routines are exact coordinate descent on `J`: block `m` is replaced by the codebook row
//! minimising the block-local cost `a² Q_m[j] − 2 a (f_m · C_j)` (one plane) or the pair cost
//! `2 a₁a₂ X_m[i₁,i₂] + A[i₁] + B[i₂]` (two planes), holding every other block fixed. Each accepted
//! move therefore does not increase `J`, which is what `tests/learn_fold_icm.rs` asserts per sweep.
//!
//! Scale: the codebook of `crate::learn::code::e8` is `Q32.32` (`i64`, `ONE = 2^32`); [`codebook`]
//! converts each row to its real value `v / ONE`, so the `s` plane entries are `±1`, `±2`, `±√8` as
//! in the script. The Euclidean nearest-codeword rule of the codebook (`e8::nearest`, the first index
//! on ties) is exposed as [`nearest_index`]; the ICM argmin is *not* that rule — it minimises the
//! `H_blk`-weighted cost above — and is computed here directly.
//!
//! Grade: `tested` (`tests/learn_fold_icm.rs`); nothing in this module is extracted or refined to the
//! canonical corpus, so no statement here is `proved`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the flat-slice kernels index row-major `[o, n]` buffers by `o/i/j/k/l/m` and convert \
              `usize`/`i64` scales and code indices to `f64`/`u16`, reading closer to the reference \
              `fold_transfer.py` arithmetic than enumerated iterators would; the move counter and \
              the block tie test are the reference's exact-`0.0` comparisons (`e1 != 0`); the \
              functions are long because each is one ported body"
)]

use crate::engine::ONE;
use crate::learn::code::e8;
use crate::learn::ridge::{solve_spd, RidgeError};
use crate::learn::solve::gemm::{gemm_nn_f64, GemmError};

/// Columns per `E8 ∪ O8` block: the code dimension.
pub const BLOCK: usize = 8;

/// Columns per `r` group scale. `GROUP = 128` unless the Python says otherwise.
pub const GROUP: usize = 128;

/// Codebook rows: the 256 `E8 ∪ O8` codes of [`crate::learn::code::e8`].
pub const CODE_LEN: usize = e8::CODEBOOK_LEN;

/// A refusal from an ICM refinement step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcmError {
    /// A slice length disagrees with the declared shape (`o`, `n` and the group/block widths).
    Shape,
    /// `n` is not a positive multiple of both [`BLOCK`] and [`GROUP`].
    Width,
    /// `row_chunk == 0`.
    ZeroChunk,
    /// A block index in `blocks` is `>= n / BLOCK`.
    BlockOutOfRange,
    /// A value could not be carried as a codebook index, or the codebook's `i64` distance
    /// arithmetic left its range.
    Overflow,
    /// The `r` normal system was not positive-definite (only reachable on a degenerate metric).
    NotPositiveDefinite,
}

/// A borrowed plane `(s, idx, r, c, d)`.
#[derive(Clone, Copy, Debug)]
pub struct PlaneRef<'a> {
    /// Plane code values (real, not `Q32.32`), row-major `[o, n]`; block `m` of row `o` is
    /// `s[o*n + 8m .. o*n + 8m + 8]`.
    pub s: &'a [f64],
    /// Codebook row per block, row-major `[o, n / 8]`, each in `0..256`.
    pub idx: &'a [u16],
    /// Group scale, row-major `[o, n / 128]`; column `i` uses `r[o*(n/128) + i/128]`.
    pub r: &'a [f64],
    /// Column scale, length `n`.
    pub c: &'a [f64],
    /// Hadamard column scale, length `n`.
    pub d: &'a [f64],
}

/// An owned plane: the result of a refinement, with `r`, `c`, `d` passed through unchanged.
#[derive(Clone, Debug)]
pub struct Plane {
    /// Plane code values, row-major `[o, n]`.
    pub s: Vec<f64>,
    /// Codebook row per block, row-major `[o, n / 8]`.
    pub idx: Vec<u16>,
    /// Group scale, row-major `[o, n / 128]`.
    pub r: Vec<f64>,
    /// Column scale, length `n`.
    pub c: Vec<f64>,
    /// Hadamard column scale, length `n`.
    pub d: Vec<f64>,
}

/// Counters from [`e8_joint_icm_2plane`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointStats {
    /// `J` of the two planes before the sweeps, in the ICM-ridged metric, row-weighted by `g_row`.
    pub j_before: f64,
    /// `J` after the sweeps, same metric.
    pub j_after: f64,
    /// Sum over blocks of the rows in which at least one plane's block code changed.
    pub moves: usize,
}

/// The 256 `E8 ∪ O8` codewords as real values: `codeword(j)[k] / 2^32`.
///
/// # Complexity
/// Temporal `O(256·8)` fixed-point conversions (one [`e8::codeword`] per row); spatial `O(256·8)`
/// `f64`. Both are constant in the caller's dimensions.
#[must_use]
pub fn codebook() -> [[f64; BLOCK]; CODE_LEN] {
    let one = ONE as f64;
    let mut out = [[0.0_f64; BLOCK]; CODE_LEN];
    for j in 0..CODE_LEN {
        // `j < CODE_LEN` is the caller-independent loop bound, so the row always exists.
        if let Ok(row) = e8::codeword(j) {
            for k in 0..BLOCK {
                out[j][k] = row[k] as f64 / one;
            }
        }
    }
    out
}

/// Euclidean nearest codebook row of a real-valued 8-vector, first index on ties, via the codebook's
/// own `Q32.32` rule ([`e8::nearest`]).
///
/// The ICM block argmin is not this rule: it minimises `a² Q_m[j] − 2 a (f_m · C_j)` in the
/// `H_blk`-weighted metric. This helper is the codebook's Euclidean rule, for a consumer that
/// initialises a plane from a target block.
///
/// # Errors
/// [`IcmError::Overflow`] if `code · 2^32` is not representable in `i64`, or the codebook's exact
/// `i128` distance arithmetic leaves its range.
///
/// # Complexity
/// Temporal `O(256·8)` exact `i128` operations; spatial `O(1)`.
pub fn nearest_index(code: &[f64; BLOCK]) -> Result<usize, IcmError> {
    let one = ONE as f64;
    let mut q = [0_i64; BLOCK];
    for k in 0..BLOCK {
        let v = code[k] * one;
        if !(v >= i64::MIN as f64 && v <= i64::MAX as f64) {
            return Err(IcmError::Overflow);
        }
        q[k] = v.round() as i64;
    }
    match e8::nearest(&q) {
        Ok(i) => Ok(i),
        Err(_) => Err(IcmError::Overflow),
    }
}

/// The metric the ICM cost is minimised in: `Hl`, or `Hl` plus the `icm_ridge` diagonal shift
/// `icm_ridge · (n / 896) · mean(diag(Hl))` when `icm_ridge > 0`.
///
/// # Errors
/// [`IcmError::Shape`] if `hl.len() != n*n` or `n == 0`.
///
/// # Complexity
/// Temporal `O(n²)`; spatial `O(n²)`.
pub fn ridged_metric(hl: &[f64], n: usize, icm_ridge: f64) -> Result<Vec<f64>, IcmError> {
    if n == 0 || hl.len() != n * n {
        return Err(IcmError::Shape);
    }
    let mut h = hl.to_vec();
    if icm_ridge > 0.0 {
        let mut mean = 0.0;
        for i in 0..n {
            mean += hl[i * n + i];
        }
        mean /= n as f64;
        let add = icm_ridge * (n as f64 / 896.0) * mean;
        for i in 0..n {
            h[i * n + i] += add;
        }
    }
    Ok(h)
}

/// `plane_w(P) = fwht128(s·d) · r_rep · c`, the plane's weight matrix, row-major `[o, n]`.
///
/// # Errors
/// [`IcmError::Shape`] on a slice length that disagrees with `o`/`n`; [`IcmError::Width`] if `n` is
/// not a positive multiple of [`BLOCK`] and [`GROUP`].
///
/// # Complexity
/// Temporal `O(o·n·log2(GROUP))` additions for the Walsh–Hadamard transform plus `O(o·n)` scalar
/// multiplies; spatial `O(o·n)`.
pub fn plane_w(p: PlaneRef<'_>, o: usize, n: usize) -> Result<Vec<f64>, IcmError> {
    check_plane(p, o, n)?;
    let g = n / GROUP;
    let mut t = p.s.to_vec();
    scale_cols(&mut t, o, n, p.d);
    fwht128(&mut t);
    scale_cols(&mut t, o, n, p.c);
    for row in 0..o {
        for col in 0..n {
            t[row * n + col] *= p.r[row * g + col / GROUP];
        }
    }
    Ok(t)
}

/// `J(Ws, W, Hl) = Σ_o g_row[o] · (D_o Hl · D_o)`, `D = Ws − W`; the unweighted sum when `g_row` is
/// `None`.
///
/// # Errors
/// [`IcmError::Shape`] if `ws`, `w` are not `o*n`, `hl` is not `n*n`, or `g_row` is not `o`.
///
/// # Complexity
/// Temporal `O(o·n²)` multiply-adds; spatial `O(1)` beyond the inputs.
pub fn j_rows(
    ws: &[f64],
    w: &[f64],
    hl: &[f64],
    o: usize,
    n: usize,
    g_row: Option<&[f64]>,
) -> Result<f64, IcmError> {
    if ws.len() != o * n || w.len() != o * n || hl.len() != n * n {
        return Err(IcmError::Shape);
    }
    if let Some(g) = g_row {
        if g.len() != o {
            return Err(IcmError::Shape);
        }
    }
    let mut total = 0.0;
    for row in 0..o {
        let mut j = 0.0;
        for i in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += (ws[row * n + k] - w[row * n + k]) * hl[k * n + i];
            }
            j += acc * (ws[row * n + i] - w[row * n + i]);
        }
        total += match g_row {
            Some(g) => g[row] * j,
            None => j,
        };
    }
    Ok(total)
}

/// One plane's iterated conditional modes: `sweeps` passes over the `n / 8` blocks, each replacing
/// one block's code by the codebook row minimising the block cost in the (optionally `icm_ridge`d)
/// metric, block order reversed on even sweeps.
///
/// `r`, `c` and `d` are returned unchanged; `s` and `idx` carry the refined codes.
///
/// # Errors
/// [`IcmError::Shape`] on a slice length that disagrees with `o`/`n`; [`IcmError::Width`] if `n` is
/// not a positive multiple of [`BLOCK`] and [`GROUP`].
///
/// # Complexity
/// Temporal `sweeps · (n/8) · O · (BLOCK² + CODE_LEN·BLOCK + BLOCK·n)` multiply-adds: per block and
/// row the residual rebuild (`BLOCK²`), the `CODE_LEN`-row argmin (`CODE_LEN·BLOCK`) and the
/// residual update (`BLOCK·n`), summed over the `n/8` independent blocks. Setup adds `O(o·n²)` for
/// `Ws@Hl`, the two `fwht128` passes, `B@H_rot` and `CODE_LEN` `Q` values per block. Spatial
/// `O(n²)` for `H_rot`/`hcc` plus `O(o·n)` for `s`, `F_res`, `G_rot`, `B`.
pub fn e8_icm_step(
    ws: &[f64],
    hl: &[f64],
    p: PlaneRef<'_>,
    o: usize,
    n: usize,
    sweeps: usize,
    icm_ridge: f64,
) -> Result<Plane, IcmError> {
    check_plane(p, o, n)?;
    if ws.len() != o * n || hl.len() != n * n {
        return Err(IcmError::Shape);
    }
    let metric = ridged_metric(hl, n, icm_ridge)?;
    let cb = codebook();
    let g = n / GROUP;
    let nb = n / BLOCK;

    // G_rot = fwht128((Ws @ Hl) * c) * d
    let mut g_rot = matmul(ws, &metric, o, n, n)?;
    scale_cols(&mut g_rot, o, n, p.c);
    fwht128(&mut g_rot);
    scale_cols(&mut g_rot, o, n, p.d);
    // H_rot = had_gram(Hl ∘ c c^T) ∘ d d^T
    let mut h_rot = had_gram(&pair_scaled(&metric, n, p.c, p.c), n);
    pair_scale_in_place(&mut h_rot, n, p.d, p.d);
    // B = s * r_rep, F_res = G_rot - B @ H_rot
    let mut b = p.s.to_vec();
    for row in 0..o {
        for col in 0..n {
            b[row * n + col] *= p.r[row * g + col / GROUP];
        }
    }
    let bh = matmul(&b, &h_rot, o, n, n)?;
    let mut f_res = g_rot;
    for k in 0..o * n {
        f_res[k] -= bh[k];
    }
    // Q_all[m][j] = C_j^T H_blk[m] C_j, with H_blk[m] the m-th 8x8 diagonal block of H_rot.
    let mut q_all = vec![0.0_f64; nb * CODE_LEN];
    for m in 0..nb {
        let hb = block_of(&h_rot, n, m);
        for j in 0..CODE_LEN {
            let mut acc = 0.0;
            for a in 0..BLOCK {
                for bcol in 0..BLOCK {
                    acc += cb[j][a] * hb[a][bcol] * cb[j][bcol];
                }
            }
            q_all[m * CODE_LEN + j] = acc;
        }
    }

    let mut s_cur = p.s.to_vec();
    let mut idx_cur = p.idx.to_vec();
    for sw in 0..sweeps {
        for step in 0..nb {
            let m = if sw % 2 == 0 { nb - 1 - step } else { step };
            let b0 = m * BLOCK;
            let gidx = b0 / GROUP;
            let hb = block_of(&h_rot, n, m);
            let mut a = vec![0.0_f64; o];
            for row in 0..o {
                a[row] = p.r[row * g + gidx];
            }
            let mut diff = vec![0.0_f64; o * BLOCK];
            let mut arg = vec![0_usize; o];
            for row in 0..o {
                let mut f_m = [0.0_f64; BLOCK];
                for k in 0..BLOCK {
                    let mut acc = f_res[row * n + b0 + k];
                    for l in 0..BLOCK {
                        acc += a[row] * s_cur[row * n + b0 + l] * hb[l][k];
                    }
                    f_m[k] = acc;
                }
                let mut best = 0_usize;
                let mut best_cost = f64::INFINITY;
                for j in 0..CODE_LEN {
                    let mut dot = 0.0;
                    for k in 0..BLOCK {
                        dot += f_m[k] * cb[j][k];
                    }
                    let cost = a[row] * a[row] * q_all[m * CODE_LEN + j] - 2.0 * a[row] * dot;
                    if cost < best_cost {
                        best_cost = cost;
                        best = j;
                    }
                }
                arg[row] = best;
                for k in 0..BLOCK {
                    diff[row * BLOCK + k] = cb[best][k] - s_cur[row * n + b0 + k];
                }
            }
            for row in 0..o {
                for k in 0..BLOCK {
                    s_cur[row * n + b0 + k] = cb[arg[row]][k];
                }
                idx_cur[row * nb + m] = arg[row] as u16;
            }
            for row in 0..o {
                for col in 0..n {
                    let mut acc = 0.0;
                    for k in 0..BLOCK {
                        acc += a[row] * diff[row * BLOCK + k] * h_rot[(b0 + k) * n + col];
                    }
                    f_res[row * n + col] -= acc;
                }
            }
        }
    }
    Ok(Plane {
        s: s_cur,
        idx: idx_cur,
        r: p.r.to_vec(),
        c: p.c.to_vec(),
        d: p.d.to_vec(),
    })
}

/// Joint block ICM over two planes: `sweeps` passes over the `n / 8` blocks (or the `blocks` subset,
/// in the given order), each replacing both planes' codes of one block by the pair `(i₁, i₂)`
/// minimising `2 a₁a₂ X_m[i₁,i₂] + A[i₁] + B[i₂]` in the (optionally `icm_ridge`d) coupled metric.
///
/// The `row_chunk` argument bounds the working set of the argmin loop (one chunk of rows at a time);
/// it does not change the result, because rows are independent given the residual field.
///
/// # Errors
/// [`IcmError::Shape`] on a slice length that disagrees with `o`/`n`; [`IcmError::Width`] if `n` is
/// not a positive multiple of [`BLOCK`] and [`GROUP`]; [`IcmError::ZeroChunk`] if `row_chunk == 0`;
/// [`IcmError::BlockOutOfRange`] if a block index in `blocks` is `>= n / BLOCK`.
///
/// # Complexity
/// Temporal `sweeps · (n/8) · O · (CODE_LEN² + 2·CODE_LEN·BLOCK + 2·BLOCK·n)` multiply-adds for the
/// pair argmin and the two residual updates, plus `O(n²·log2(GROUP))` for the three `had_gram`
/// passes and `(n/8)·CODE_LEN²·BLOCK` for the `X` tensor. Spatial `O(o·n)` for `s`, `F`, `G`, `B`,
/// `O(n²)` for `H11/H22/H12/H21`, `(n/8)·CODE_LEN²` `f64` for `X` (16 MB at `n = 256`, 57 MB at
/// `n = 896`) and `row_chunk·CODE_LEN²` transient in the argmin when the cost tensor is materialised
/// by a consumer (this body iterates the pair grid directly).
#[allow(
    clippy::too_many_arguments,
    reason = "the option surface (sweeps, icm_ridge, g_row, row_chunk, blocks) mirrors the Python \
              signature; a bag struct would hide which arguments are metric options and which are \
              shapes"
)]
pub fn e8_joint_icm_2plane(
    ws: &[f64],
    hl: &[f64],
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    o: usize,
    n: usize,
    sweeps: usize,
    icm_ridge: f64,
    g_row: Option<&[f64]>,
    row_chunk: usize,
    blocks: Option<&[usize]>,
) -> Result<(Plane, Plane, JointStats), IcmError> {
    check_plane(p1, o, n)?;
    check_plane(p2, o, n)?;
    if ws.len() != o * n || hl.len() != n * n {
        return Err(IcmError::Shape);
    }
    if let Some(g) = g_row {
        if g.len() != o {
            return Err(IcmError::Shape);
        }
    }
    if row_chunk == 0 {
        return Err(IcmError::ZeroChunk);
    }
    let metric = ridged_metric(hl, n, icm_ridge)?;
    let cb = codebook();
    let g = n / GROUP;
    let nb = n / BLOCK;
    if let Some(bs) = blocks {
        for &m in bs {
            if m >= nb {
                return Err(IcmError::BlockOutOfRange);
            }
        }
    }

    let w1 = plane_w(p1, o, n)?;
    let w2 = plane_w(p2, o, n)?;
    let mut wsum = w1;
    for k in 0..o * n {
        wsum[k] += w2[k];
    }
    let j_before = j_rows(ws, &wsum, &metric, o, n, g_row)?;

    let mut s1 = p1.s.to_vec();
    let mut s2 = p2.s.to_vec();
    let mut idx1 = p1.idx.to_vec();
    let mut idx2 = p2.idx.to_vec();
    let r1 = p1.r.to_vec();
    let r2 = p2.r.to_vec();
    let c1 = p1.c.to_vec();
    let c2 = p2.c.to_vec();
    let d1 = p1.d.to_vec();
    let d2 = p2.d.to_vec();

    let wh = matmul(ws, &metric, o, n, n)?;
    let mut g1 = wh.clone();
    scale_cols(&mut g1, o, n, &c1);
    fwht128(&mut g1);
    scale_cols(&mut g1, o, n, &d1);
    let mut g2 = wh;
    scale_cols(&mut g2, o, n, &c2);
    fwht128(&mut g2);
    scale_cols(&mut g2, o, n, &d2);

    let mut h11 = had_gram(&pair_scaled(&metric, n, &c1, &c1), n);
    pair_scale_in_place(&mut h11, n, &d1, &d1);
    let mut h22 = had_gram(&pair_scaled(&metric, n, &c2, &c2), n);
    pair_scale_in_place(&mut h22, n, &d2, &d2);
    let mut h12 = had_gram(&pair_scaled(&metric, n, &c1, &c2), n);
    pair_scale_in_place(&mut h12, n, &d1, &d2);
    let h21 = transpose(&h12, n);

    let mut b1 = s1.clone();
    let mut b2 = s2.clone();
    for row in 0..o {
        for col in 0..n {
            b1[row * n + col] *= r1[row * g + col / GROUP];
            b2[row * n + col] *= r2[row * g + col / GROUP];
        }
    }
    let b1h11 = matmul(&b1, &h11, o, n, n)?;
    let b2h21 = matmul(&b2, &h21, o, n, n)?;
    let b2h22 = matmul(&b2, &h22, o, n, n)?;
    let b1h12 = matmul(&b1, &h12, o, n, n)?;
    let mut f1 = g1;
    let mut f2 = g2;
    for k in 0..o * n {
        f1[k] = f1[k] - b1h11[k] - b2h21[k];
        f2[k] = f2[k] - b2h22[k] - b1h12[k];
    }

    let mut q1 = vec![0.0_f64; nb * CODE_LEN];
    let mut q2 = vec![0.0_f64; nb * CODE_LEN];
    let mut x_all = vec![0.0_f64; nb * CODE_LEN * CODE_LEN];
    for m in 0..nb {
        let hb11 = block_of(&h11, n, m);
        let hb22 = block_of(&h22, n, m);
        let hb12 = block_of(&h12, n, m);
        for j in 0..CODE_LEN {
            let mut a1v = 0.0;
            let mut a2v = 0.0;
            for a in 0..BLOCK {
                for bcol in 0..BLOCK {
                    a1v += cb[j][a] * hb11[a][bcol] * cb[j][bcol];
                    a2v += cb[j][a] * hb22[a][bcol] * cb[j][bcol];
                }
            }
            q1[m * CODE_LEN + j] = a1v;
            q2[m * CODE_LEN + j] = a2v;
        }
        for i1 in 0..CODE_LEN {
            let mut tmp = [0.0_f64; BLOCK];
            for a in 0..BLOCK {
                for bcol in 0..BLOCK {
                    tmp[bcol] += cb[i1][a] * hb12[a][bcol];
                }
            }
            for i2 in 0..CODE_LEN {
                let mut acc = 0.0;
                for bcol in 0..BLOCK {
                    acc += tmp[bcol] * cb[i2][bcol];
                }
                x_all[(m * CODE_LEN + i1) * CODE_LEN + i2] = acc;
            }
        }
    }

    let mut moves = 0_usize;
    for sw in 0..sweeps {
        let order: Vec<usize> = match blocks {
            Some(bs) => bs.to_vec(),
            None => {
                if sw % 2 == 0 {
                    (0..nb).rev().collect()
                } else {
                    (0..nb).collect()
                }
            }
        };
        for &m in &order {
            let b0 = m * BLOCK;
            let gidx = b0 / GROUP;
            let hb11 = block_of(&h11, n, m);
            let hb22 = block_of(&h22, n, m);
            let hb12 = block_of(&h12, n, m);
            let xm = &x_all[m * CODE_LEN * CODE_LEN..(m + 1) * CODE_LEN * CODE_LEN];
            for o0 in (0..o).step_by(row_chunk) {
                let o1 = (o0 + row_chunk).min(o);
                for row in o0..o1 {
                    let a1 = r1[row * g + gidx];
                    let a2 = r2[row * g + gidx];
                    let mut f1m = [0.0_f64; BLOCK];
                    let mut f2m = [0.0_f64; BLOCK];
                    for k in 0..BLOCK {
                        let mut acc1 = f1[row * n + b0 + k];
                        let mut acc2 = f2[row * n + b0 + k];
                        for l in 0..BLOCK {
                            let s1o = s1[row * n + b0 + l];
                            let s2o = s2[row * n + b0 + l];
                            acc1 += a1 * s1o * hb11[l][k] + a2 * s2o * hb12[k][l];
                            acc2 += a2 * s2o * hb22[l][k] + a1 * s1o * hb12[l][k];
                        }
                        f1m[k] = acc1;
                        f2m[k] = acc2;
                    }
                    let mut av = [0.0_f64; CODE_LEN];
                    let mut bv = [0.0_f64; CODE_LEN];
                    for j in 0..CODE_LEN {
                        let mut d1v = 0.0;
                        let mut d2v = 0.0;
                        for k in 0..BLOCK {
                            d1v += f1m[k] * cb[j][k];
                            d2v += f2m[k] * cb[j][k];
                        }
                        av[j] = a1 * a1 * q1[m * CODE_LEN + j] - 2.0 * a1 * d1v;
                        bv[j] = a2 * a2 * q2[m * CODE_LEN + j] - 2.0 * a2 * d2v;
                    }
                    let cross = 2.0 * a1 * a2;
                    let mut flat = 0_usize;
                    let mut best = f64::INFINITY;
                    for i1v in 0..CODE_LEN {
                        let base = i1v * CODE_LEN;
                        for i2v in 0..CODE_LEN {
                            let t = cross * xm[base + i2v] + av[i1v] + bv[i2v];
                            if t < best {
                                best = t;
                                flat = base + i2v;
                            }
                        }
                    }
                    let i1s = flat / CODE_LEN;
                    let i2s = flat % CODE_LEN;
                    let mut e1 = [0.0_f64; BLOCK];
                    let mut e2 = [0.0_f64; BLOCK];
                    let mut moved = false;
                    for k in 0..BLOCK {
                        let n1 = cb[i1s][k] - s1[row * n + b0 + k];
                        let n2 = cb[i2s][k] - s2[row * n + b0 + k];
                        e1[k] = a1 * n1;
                        e2[k] = a2 * n2;
                        if n1 != 0.0 || n2 != 0.0 {
                            moved = true;
                        }
                    }
                    if moved {
                        moves += 1;
                    }
                    for k in 0..BLOCK {
                        s1[row * n + b0 + k] = cb[i1s][k];
                        s2[row * n + b0 + k] = cb[i2s][k];
                    }
                    idx1[row * nb + m] = i1s as u16;
                    idx2[row * nb + m] = i2s as u16;
                    for col in 0..n {
                        let mut u1 = 0.0;
                        let mut u2 = 0.0;
                        for k in 0..BLOCK {
                            u1 += e1[k] * h11[(b0 + k) * n + col] + e2[k] * h21[(b0 + k) * n + col];
                            u2 += e2[k] * h22[(b0 + k) * n + col] + e1[k] * h12[(b0 + k) * n + col];
                        }
                        f1[row * n + col] -= u1;
                        f2[row * n + col] -= u2;
                    }
                }
            }
        }
    }

    let out1 = Plane {
        s: s1,
        idx: idx1,
        r: r1,
        c: c1,
        d: d1,
    };
    let out2 = Plane {
        s: s2,
        idx: idx2,
        r: r2,
        c: c2,
        d: d2,
    };
    let mut w1n = plane_w(
        PlaneRef {
            s: &out1.s,
            idx: &out1.idx,
            r: &out1.r,
            c: &out1.c,
            d: &out1.d,
        },
        o,
        n,
    )?;
    let w2n = plane_w(
        PlaneRef {
            s: &out2.s,
            idx: &out2.idx,
            r: &out2.r,
            c: &out2.c,
            d: &out2.d,
        },
        o,
        n,
    )?;
    for k in 0..o * n {
        w1n[k] += w2n[k];
    }
    let j_after = j_rows(ws, &w1n, &metric, o, n, g_row)?;
    Ok((
        out1,
        out2,
        JointStats {
            j_before,
            j_after,
            moves,
        },
    ))
}

/// Joint least-squares solve for both planes' group scales `(r1, r2)`: the exact minimiser of
/// `J(Ws, plane_w(s1,r1,c1,d1) + plane_w(s2,r2,c2,d2), Hl)` over `r1, r2 ∈ R^{o × (n/128)}`, per row
/// a `2G × 2G` normal system `M x = t` with a `1e-12 · mean(diag(M))` diagonal ridge.
///
/// `s1`/`c1` and `s2`/`c2` are the raw code values and column scales; `d` does not enter (the group
/// scale is applied after the transform, and a common `d` factor cancels in the normal equations
/// only in the plane it multiplies — this body is the ported reference, which passes the plane
/// weights `s_k · c_k` and the metric `Hl`).
///
/// # Errors
/// [`IcmError::Shape`] on a slice length that disagrees with `o`/`n`; [`IcmError::Width`] if `n` is
/// not a positive multiple of [`BLOCK`] and [`GROUP`]; [`IcmError::NotPositiveDefinite`] if the
/// per-row normal matrix is not positive-definite after the ridge.
///
/// # Complexity
/// Temporal `2G` row-panel products of `o × 128` by `128 × n` (`2G·o·128·n` multiply-adds) plus `o`
/// solves of `2G × 2G` (`o·(2G)³/3`); spatial `O(o·(2G)²)` for the per-row normal matrix plus
/// `O(o·n)` for the plane blocks.
pub fn r_step2(
    ws: &[f64],
    hl: &[f64],
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    o: usize,
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), IcmError> {
    check_plane(p1, o, n)?;
    check_plane(p2, o, n)?;
    if ws.len() != o * n || hl.len() != n * n {
        return Err(IcmError::Shape);
    }
    let g = n / GROUP;
    let two_g = 2 * g;
    let wh = matmul(ws, hl, o, n, n)?;
    let mut r1_out = vec![0.0_f64; o * g];
    let mut r2_out = vec![0.0_f64; o * g];
    let mut kvec = vec![0.0_f64; g * GROUP];
    for row in 0..o {
        // P[o, a_, j]: a_ < G is plane 1's block `a_`, a_ >= G plane 2's block `a_ - G`.
        let pv = |a_: usize, j: usize| -> f64 {
            if a_ < g {
                p1.s[row * n + a_ * GROUP + j] * p1.c[a_ * GROUP + j]
            } else {
                p2.s[row * n + (a_ - g) * GROUP + j] * p2.c[(a_ - g) * GROUP + j]
            }
        };
        let mut mmat = vec![0.0_f64; two_g * two_g];
        for a_ in 0..two_g {
            let ga = a_ % g;
            for gidx in 0..g {
                for j in 0..GROUP {
                    let mut acc = 0.0;
                    for t in 0..GROUP {
                        acc += pv(a_, t) * hl[(ga * GROUP + t) * n + gidx * GROUP + j];
                    }
                    kvec[gidx * GROUP + j] = acc;
                }
            }
            for bcol in 0..two_g {
                let gb = bcol % g;
                let mut acc = 0.0;
                for j in 0..GROUP {
                    acc += kvec[gb * GROUP + j] * pv(bcol, j);
                }
                mmat[a_ * two_g + bcol] = acc;
            }
        }
        let mut tvec = vec![0.0_f64; two_g];
        for a_ in 0..two_g {
            let ga = a_ % g;
            let mut acc = 0.0;
            for j in 0..GROUP {
                acc += pv(a_, j) * wh[row * n + ga * GROUP + j];
            }
            tvec[a_] = acc;
        }
        let mut mean_diag = 0.0;
        for k in 0..two_g {
            mean_diag += mmat[k * two_g + k];
        }
        mean_diag /= two_g as f64;
        for k in 0..two_g {
            mmat[k * two_g + k] += 1.0e-12 * mean_diag;
        }
        let x = match solve_spd(&mmat, &tvec, two_g, 1) {
            Ok(x) => x,
            Err(RidgeError::NotPositiveDefinite) => return Err(IcmError::NotPositiveDefinite),
            Err(_) => return Err(IcmError::Shape),
        };
        for k in 0..g {
            r1_out[row * g + k] = x[k];
            r2_out[row * g + k] = x[g + k];
        }
    }
    Ok((r1_out, r2_out))
}

/// Validate `o`, `n` and every slice length of a plane.
fn check_plane(p: PlaneRef<'_>, o: usize, n: usize) -> Result<(), IcmError> {
    if o == 0 || n == 0 {
        return Err(IcmError::Shape);
    }
    if n % BLOCK != 0 || n % GROUP != 0 {
        return Err(IcmError::Width);
    }
    if p.s.len() != o * n
        || p.idx.len() != o * (n / BLOCK)
        || p.r.len() != o * (n / GROUP)
        || p.c.len() != n
        || p.d.len() != n
    {
        return Err(IcmError::Shape);
    }
    Ok(())
}

/// The in-place Walsh–Hadamard transform of every consecutive [`GROUP`]-long chunk of `x`, times
/// `GROUP^{-1/2}` (the orthonormal transform of `fwht128`; `x.len()` must be a multiple of
/// [`GROUP`]).
///
/// # Complexity
/// Temporal `O(len·log2(GROUP))` additions and `O(len)` multiplies; spatial `O(1)`.
fn fwht128(x: &mut [f64]) {
    let mut h = 1_usize;
    while h < GROUP {
        let mut base = 0_usize;
        while base < x.len() {
            for i in 0..h {
                let a = x[base + i];
                let b = x[base + h + i];
                x[base + i] = a + b;
                x[base + h + i] = a - b;
            }
            base += 2 * h;
        }
        h *= 2;
    }
    let norm = 1.0 / (GROUP as f64).sqrt();
    for v in x.iter_mut() {
        *v *= norm;
    }
}

/// `had_gram(H) = W H W` with `W` the orthonormal per-128-block Walsh–Hadamard transform on both
/// indices (`fwht128(fwht128(H).T).T`).
///
/// # Complexity
/// Temporal `O(n²·log2(GROUP))`; spatial `O(n²)`.
fn had_gram(h: &[f64], n: usize) -> Vec<f64> {
    let mut t = h.to_vec();
    fwht128(&mut t);
    let mut t = transpose(&t, n);
    fwht128(&mut t);
    transpose(&t, n)
}

/// Row-major transpose of an `n × n` buffer.
///
/// # Complexity
/// Temporal `O(n²)`; spatial `O(n²)`.
fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

/// The worker count [`matmul`] hands to [`gemm_nn`]. `gemm_nn` is documented bit-identical across
/// thread counts, so this changes only wall-clock, not the returned vector.
const GEMM_THREADS: usize = 8;

/// Map a [`GemmError`] from [`gemm_nn`] onto an [`IcmError`], the module's existing refusal set: an
/// empty dimension or a length that disagrees with the already-checked shape is [`IcmError::Shape`],
/// and every magnitude or range refusal is the numeric-range refusal [`IcmError::Overflow`].
///
/// # Complexity
///
/// `O(1)`.
fn map_gemm(e: GemmError) -> IcmError {
    match e {
        GemmError::ZeroDim | GemmError::ShapeMismatch => IcmError::Shape,
        GemmError::InputBound | GemmError::Overflow => IcmError::Overflow,
    }
}

/// Row-major `[rows, inner] × [inner, cols]` product on the shared Q32.32 path
/// ([`gemm_nn_f64`]): the operand scale rule, the grid rounding and the exact undo are its.
///
/// # Errors
/// [`IcmError::Shape`] for an empty dimension or a length that disagrees with the shape;
/// [`IcmError::Overflow`] for a non-finite or ungriddable entry or a product that leaves `f64`.
///
/// # Complexity
/// As [`gemm_nn_f64`]: `O(rows·inner·cols)` `i128` multiply-accumulates split across
/// [`GEMM_THREADS`], `O(rows·inner + inner·cols)` conversions, `O(rows·cols)` space.
pub fn matmul(
    a: &[f64],
    b: &[f64],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<f64>, IcmError> {
    gemm_nn_f64(a, b, rows, inner, cols, GEMM_THREADS).map_err(map_gemm)
}

/// `x[row·n + col] *= v[col]` for every row of an `[o, n]` buffer.
///
/// # Complexity
/// Temporal `O(o·n)`; spatial `O(1)`.
fn scale_cols(x: &mut [f64], o: usize, n: usize, v: &[f64]) {
    for row in 0..o {
        for col in 0..n {
            x[row * n + col] *= v[col];
        }
    }
}

/// `out[i·n + j] = h[i·n + j] · x[i] · y[j]` of an `n × n` buffer.
///
/// # Complexity
/// Temporal `O(n²)`; spatial `O(n²)`.
fn pair_scaled(h: &[f64], n: usize, x: &[f64], y: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = h[i * n + j] * x[i] * y[j];
        }
    }
    out
}

/// In-place `h[i·n + j] *= x[i] · y[j]`.
///
/// # Complexity
/// Temporal `O(n²)`; spatial `O(1)`.
fn pair_scale_in_place(h: &mut [f64], n: usize, x: &[f64], y: &[f64]) {
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] *= x[i] * y[j];
        }
    }
}

/// The `m`-th `8 × 8` diagonal block of an `n × n` buffer.
///
/// # Complexity
/// Temporal `O(BLOCK²)`; spatial `O(BLOCK²)`.
fn block_of(h: &[f64], n: usize, m: usize) -> [[f64; BLOCK]; BLOCK] {
    let mut out = [[0.0_f64; BLOCK]; BLOCK];
    for a in 0..BLOCK {
        for b in 0..BLOCK {
            out[a][b] = h[(m * BLOCK + a) * n + m * BLOCK + b];
        }
    }
    out
}
