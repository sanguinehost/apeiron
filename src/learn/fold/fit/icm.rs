//! Iterated-conditional-modes (ICM) refinement of the one-bit fold transfer's `E8 ∪ O8` planes in the
//! `(r, c, d)`-scaled Hadamard–Gram metric.
//!
//! Port of `tools/onebit/fold_transfer.py` (removed; at `2f15d6a9`) lines 585–720. A plane is the pack
//! `(s, r, c, d)` with weight `plane_w = fwht128(s·d) · r_rep · c`, its block codes one of the 256 rows
//! of the [`crate::learn::code::e8`] codebook; both routines are exact coordinate descent on
//! `J(Ws, Wp, Hl) = Σ_o (Ws_o − Wp_o)ᵀ Hl (Ws_o − Wp_o)`.
//!
//! # Carrier and boundary
//!
//! Every value is Q32.32: `x` is `round(x·2³²)` in an `i64`, each product is exact in `i128` brought
//! back by one [`rshift_rne`] rounding, and block/pair costs compare as exact Q96.96 `i128`, so an
//! argmin is the integer argmin and a tie goes to the first index. The 128-point transform is the
//! engine's [`fwht_fixed`]; `gemm`/`chol` own the products and the solves; row bands of matrix-sized
//! work run on the pool and every elementwise/transform pass is a serial integer loop. Every `pub`
//! entry here takes and returns words: the planes, the metric and both `J` values are `Q32.32`, so
//! a caller that holds reals converts once at its own boundary ([`codebook`] and [`JointStats`] are
//! the read-out exceptions). No step pays a `from_grid`/`to_grid` round trip.
//!
//! Grade: `tested` (`tests/learn_fold_icm.rs`); nothing here is extracted or refined to the canonical
//! corpus, so no statement here is `proved`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the flat-slice kernels index row-major `[o, n]` buffers by `o/i/j/k/l/m` and narrow a \
              code index already bounded by `CODE_LEN` to `u16`, reading closer to the reference \
              `fold_transfer.py` arithmetic than enumerated iterators would"
)]

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::engine::chart::pool::{
    chunks_mut_global, for_chunks2, for_chunks3, width_global, FirstErr,
};
use crate::engine::chart::task::{ChunkTask, Task2, Task3};
use crate::engine::ONE;
use crate::engine::{fwht_fixed, HadamardError};
use crate::learn::code::e8;
use crate::learn::solve::chol::{chol_solve, SolveError};
use crate::learn::solve::gemm::{gemm_nn, gemm_nt, GemmError};
use crate::learn::solve::q32::{self, ilog2_pos, rshift_rne};

/// Columns per `E8 ∪ O8` block: the code dimension.
pub const BLOCK: usize = 8;
/// Columns per `r` group scale, and the transform block of [`fwht_rows`].
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
    /// A value left the `Q32.32` grid or a checked integer operation left its range.
    Overflow,
    /// The `r` normal system was not positive-definite (only reachable on a degenerate metric).
    NotPositiveDefinite,
}

/// A borrowed plane `(s, idx, r, c, d)` on the `Q32.32` grid.
#[derive(Clone, Copy, Debug)]
pub struct PlaneRef<'a> {
    /// Plane code values (`Q32.32` words), row-major `[o, n]`: block `m` of row `o` is `s[o*n+8m .. +8]`.
    pub s: &'a [i64],
    /// Codebook row per block, row-major `[o, n / 8]`, each in `0..256`.
    pub idx: &'a [u16],
    /// Group scale, row-major `[o, n / GROUP]`.
    pub r: &'a [i64],
    /// Column scale, length `n`.
    pub c: &'a [i64],
    /// Hadamard column scale, length `n`.
    pub d: &'a [i64],
}

/// An owned plane: the result of a refinement, with `r`, `c`, `d` passed through unchanged.
#[derive(Clone, Debug)]
pub struct Plane {
    /// Plane code values (`Q32.32` words), row-major `[o, n]`.
    pub s: Vec<i64>,
    /// Codebook row per block, row-major `[o, n / 8]`.
    pub idx: Vec<u16>,
    /// Group scale, row-major `[o, n / GROUP]`.
    pub r: Vec<i64>,
    /// Column scale, length `n`.
    pub c: Vec<i64>,
    /// Hadamard column scale, length `n`.
    pub d: Vec<i64>,
}

/// Counters from [`e8_joint_icm_2plane`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointStats {
    /// `J` of the two planes before the sweeps, in the ICM-ridged metric, row-weighted by `g_row`,
    /// as a Q32.32 word.
    pub j_before: i64,
    /// `J` after the sweeps, same metric, as a Q32.32 word.
    pub j_after: i64,
    /// Sum over blocks of the rows in which at least one plane's block code changed.
    pub moves: usize,
}

/// `round(v / 2^s)` as an `i64`, or [`IcmError::Overflow`] when the rounded value leaves `i64`.
fn round_shift(v: i128, s: u32) -> Result<i64, IcmError> {
    match rshift_rne(v, s) {
        Some(x) => i64::try_from(x).map_err(|_| IcmError::Overflow),
        None => Err(IcmError::Overflow),
    }
}

/// `a + b`, refusing an `i64` overflow.
fn add_q(a: i64, b: i64) -> Result<i64, IcmError> {
    a.checked_add(b).ok_or(IcmError::Overflow)
}

/// `a − b`, refusing an `i64` overflow.
fn sub_q(a: i64, b: i64) -> Result<i64, IcmError> {
    a.checked_sub(b).ok_or(IcmError::Overflow)
}

/// `round(a·b / 2³²)`.
pub(super) fn mul_q(a: i64, b: i64) -> Result<i64, IcmError> {
    round_shift(i128::from(a) * i128::from(b), 32)
}

/// Checked `i128` sum.
fn qsum(a: i128, b: i128) -> Result<i128, IcmError> {
    a.checked_add(b).ok_or(IcmError::Overflow)
}

/// Checked `i128` difference.
fn qsub(a: i128, b: i128) -> Result<i128, IcmError> {
    a.checked_sub(b).ok_or(IcmError::Overflow)
}

/// Checked `i128` product.
fn qmul(a: i128, b: i128) -> Result<i128, IcmError> {
    a.checked_mul(b).ok_or(IcmError::Overflow)
}

/// The exact `a·b·c` at Q96.96.
fn triple(a: i64, b: i64, c: i64) -> Result<i128, IcmError> {
    qmul(qmul(i128::from(a), i128::from(b))?, i128::from(c))
}

/// `Σ a[k]·b[k]` at Q64.64, exact; `O(len)` multiply-adds, `O(1)` space.
fn dot64(a: &[i64], b: &[i64]) -> Result<i128, IcmError> {
    let mut acc: i128 = 0;
    for k in 0..a.len() {
        acc = qsum(acc, i128::from(a[k]) * i128::from(b[k]))?;
    }
    Ok(acc)
}

/// The exact Q96.96 block cost `a²·q − 2·a·dot` (`dot` at Q64.64), compared unrounded.
fn block_cost(a: i64, q: i64, dot: i128) -> Result<i128, IcmError> {
    let t1 = qmul(i128::from(a) * i128::from(a), i128::from(q))?;
    qsub(t1, qmul(qmul(i128::from(a), dot)?, 2)?)
}

/// The exact Q96.96 pair cost `2·a₁a₂·x + A + B`.
fn pair_cost(a1: i64, a2: i64, x: i64, av: i128, bv: i128) -> Result<i128, IcmError> {
    let a12 = qmul(i128::from(a1), i128::from(a2))?;
    qsum(qsum(qmul(qmul(a12, 2)?, i128::from(x))?, av)?, bv)
}

/// The exact left shift of a Q32.32 word by `s` bits: the undo of a panel scaling.
fn shl_q32(v: i64, s: u32) -> Result<i64, IcmError> {
    let scale = 1_i128.checked_shl(s).ok_or(IcmError::Overflow)?;
    i64::try_from(qmul(i128::from(v), scale)?).map_err(|_| IcmError::Overflow)
}

/// An empty dimension or a length that disagrees with the checked shape is [`IcmError::Shape`]; a magnitude or range refusal is [`IcmError::Overflow`].
fn map_gemm(e: GemmError) -> IcmError {
    match e {
        GemmError::ZeroDim | GemmError::ShapeMismatch => IcmError::Shape,
        GemmError::InputBound | GemmError::Overflow => IcmError::Overflow,
    }
}

/// A non-positive pivot (or diagonal) is the normal-system refusal, every other cause shape.
fn map_solve(e: SolveError) -> IcmError {
    match e {
        SolveError::NotPositiveDefinite | SolveError::NonPositiveDiagonal => {
            IcmError::NotPositiveDefinite
        }
        SolveError::ShapeMismatch | SolveError::Empty | SolveError::Overflow => IcmError::Shape,
    }
}

/// The `gemm` product of two grid panels, with the common power-of-two panel scale undone.
/// # Complexity
/// `O(rows·inner·cols)` `i128` multiply-accumulates on the pool, `O(rows·cols)` space.
fn product_panels(
    a: &[i64],
    b: &[i64],
    rows: usize,
    inner: usize,
    cols: usize,
    nt: bool,
) -> Result<(Vec<i64>, u32), IcmError> {
    let (Some(ri), Some(bi)) = (rows.checked_mul(inner), cols.checked_mul(inner)) else {
        return Err(IcmError::Shape);
    };
    if rows == 0 || inner == 0 || cols == 0 || a.len() != ri || b.len() != bi {
        return Err(IcmError::Shape);
    }
    // The least shift bringing each panel's magnitude under the `gemm` input bound.
    let shift = |x: &[i64]| -> Result<(Vec<i64>, u32), IcmError> {
        let mut mx: u128 = 0;
        for &v in x {
            mx = mx.max(u128::from(v.unsigned_abs()));
        }
        let s = match ilog2_pos(mx) {
            Some(b) if b + 1 > 35 => b + 1 - 35,
            _ => 0,
        };
        let p = x
            .iter()
            .map(|&v| round_shift(i128::from(v), s))
            .collect::<Result<Vec<i64>, IcmError>>()?;
        Ok((p, s))
    };
    let ((pa, sa), (pb, sb)) = (shift(a)?, shift(b)?);
    let c = if nt {
        gemm_nt(&pa, &pb, rows, inner, cols)
    } else {
        gemm_nn(&pa, &pb, rows, inner, cols)
    }
    .map_err(map_gemm)?;
    Ok((c, sa + sb))
}

/// # Complexity
/// As [`product_panels`] plus `O(rows·cols)` shifts.
fn matmul_q(
    a: &[i64],
    b: &[i64],
    rows: usize,
    inner: usize,
    cols: usize,
    nt: bool,
) -> Result<Vec<i64>, IcmError> {
    let (c, back) = product_panels(a, b, rows, inner, cols, nt)?;
    c.into_iter().map(|v| shl_q32(v, back)).collect()
}

/// The 256 `E8 ∪ O8` codewords as Q32.32 words.
fn codebook_q() -> [[i64; BLOCK]; CODE_LEN] {
    core::array::from_fn(|j| e8::codeword(j).unwrap_or([0_i64; BLOCK]))
}

/// The `m`-th `8 × 8` diagonal block of a row-major `n × n` buffer, as `H[a][b]`.
fn block_of_q(h: &[i64], n: usize, m: usize) -> [[i64; BLOCK]; BLOCK] {
    core::array::from_fn(|a| core::array::from_fn(|b| h[(m * BLOCK + a) * n + m * BLOCK + b]))
}

/// `Σ_a c[a]·H[a, :]` at Q64.64: the row reduction reused across the 256 codebook rows.
fn mat_blk_row(c: &[i64; BLOCK], h: &[[i64; BLOCK]; BLOCK]) -> Result<[i128; BLOCK], IcmError> {
    let mut out = [0_i128; BLOCK];
    for a in 0..BLOCK {
        for b in 0..BLOCK {
            out[b] = qsum(out[b], i128::from(c[a]) * i128::from(h[a][b]))?;
        }
    }
    Ok(out)
}

/// `Σ_k t[k]·c[k]` at Q64.64, exact.
fn row_dot(t: &[i128; BLOCK], c: &[i64; BLOCK]) -> Result<i128, IcmError> {
    let mut acc: i128 = 0;
    for k in 0..BLOCK {
        acc = qsum(acc, qmul(t[k], i128::from(c[k]))?)?;
    }
    Ok(acc)
}

/// Rows per pool task of a pooled row band.
fn rows_per_task(rows: usize) -> usize {
    rows.div_ceil(width_global()).max(1)
}

/// The elementwise binary op of [`bin_into`].
#[derive(Clone, Copy)]
pub(super) enum BinOp {
    /// `dst + src`.
    Add,
    /// `dst − src`.
    Sub,
}

/// `dst op= src`, checked; `O(len)` adds, `O(1)` space.
pub(super) fn bin_into(dst: &mut [i64], src: &[i64], op: BinOp) -> Result<(), IcmError> {
    if dst.len() != src.len() {
        return Err(IcmError::Shape);
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = match op {
            BinOp::Add => add_q(*d, s)?,
            BinOp::Sub => sub_q(*d, s)?,
        };
    }
    Ok(())
}

/// What a [`scale_by`] multiplies a column by.
#[derive(Clone, Copy)]
pub(super) enum ScaleBy {
    /// `v[col]`, the column scale `c`.
    Column,
    /// `v[row·g + col/GROUP]`, the group scale `r`.
    Group,
    /// `v[row]`, the row scale `x`.
    Row,
}

/// `x[o, n] *= v` along the selected axis, checked; `O(o·n)` multiplies, `O(1)` space.
pub(super) fn scale_by(
    x: &mut [i64],
    o: usize,
    n: usize,
    v: &[i64],
    g: usize,
    by: ScaleBy,
) -> Result<(), IcmError> {
    let need = match by {
        ScaleBy::Column => n,
        ScaleBy::Group => o * g,
        ScaleBy::Row => o,
    };
    let group_ok = !matches!(by, ScaleBy::Group) || g * GROUP == n;
    if o == 0 || n == 0 || x.len() != o * n || v.len() != need || !group_ok {
        return Err(IcmError::Shape);
    }
    for row in 0..o {
        for (col, d) in x[row * n..(row + 1) * n].iter_mut().enumerate() {
            let f = match by {
                ScaleBy::Column => v[col],
                ScaleBy::Group => v[row * g + col / GROUP],
                ScaleBy::Row => v[row],
            };
            *d = mul_q(*d, f)?;
        }
    }
    Ok(())
}

/// `fwht128` over each [`GROUP`]-column block of `x`, in place: the engine's normalized integer
/// rotation `x' = round(H₁₂₈ · x / √128)`, the `1/√128` folded into one Q32.32 constant.
/// # Complexity
/// `O(o·n·log₂(GROUP))` checked adds and one `i128` multiply per entry, `O(1)` space in place.
fn fwht_rows(x: &mut [i64], o: usize, n: usize) -> Result<(), IcmError> {
    if o == 0 || n == 0 || !n.is_multiple_of(GROUP) || x.len() != o * n {
        return Err(IcmError::Shape);
    }
    fwht_fixed(x, None, GROUP).map_err(|e| match e {
        HadamardError::Overflow => IcmError::Overflow,
        _ => IcmError::Shape,
    })
}

/// The transpose of a row-major `[n, n]` buffer; `O(n²)` copies, `O(n²)` space.
fn transpose_q(h: &[i64], n: usize) -> Vec<i64> {
    let mut t = vec![0_i64; n * n];
    for r in 0..n {
        for c in 0..n {
            t[c * n + r] = h[r * n + c];
        }
    }
    t
}

/// `R·h·R` for `R = H₁₂₈/√128`: row transform, transpose, row transform, transpose back.
/// # Complexity
/// `O(n²·log₂(GROUP))` checked adds plus two `O(n²)` transposes, `O(n²)` space.
fn had_gram_q(h: &[i64], n: usize) -> Result<Vec<i64>, IcmError> {
    if n == 0 || !n.is_multiple_of(GROUP) || h.len() != n * n {
        return Err(IcmError::Width);
    }
    let mut t = h.to_vec();
    fwht_rows(&mut t, n, n)?;
    let mut u = transpose_q(&t, n);
    fwht_rows(&mut u, n, n)?;
    Ok(transpose_q(&u, n))
}

/// `H_jk = had_gram(Hl ∘ c_j c_kᵀ) ∘ d_j d_kᵀ`, the scaled Hadamard–Gram metric of one (or two) plane.
/// # Complexity
/// As [`had_gram_q`] plus four `O(n²)` scalings, `O(n²)` space.
fn cross_metric_q(
    hl: &[i64],
    n: usize,
    c1: &[i64],
    c2: &[i64],
    d1: &[i64],
    d2: &[i64],
) -> Result<Vec<i64>, IcmError> {
    if hl.len() != n * n {
        return Err(IcmError::Shape);
    }
    let mut h = hl.to_vec();
    scale_by(&mut h, n, n, c2, 1, ScaleBy::Column)?;
    scale_by(&mut h, n, n, c1, 1, ScaleBy::Row)?;
    let mut h = had_gram_q(&h, n)?;
    scale_by(&mut h, n, n, d2, 1, ScaleBy::Column)?;
    scale_by(&mut h, n, n, d1, 1, ScaleBy::Row)?;
    Ok(h)
}

/// Validate `ws`, `hl` and `g` against `o`/`n`.
fn check_inputs(
    ws: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    g: Option<&[i64]>,
) -> Result<(), IcmError> {
    if ws.len() != o * n || hl.len() != n * n || g.is_some_and(|g| g.len() != o) {
        return Err(IcmError::Shape);
    }
    Ok(())
}

/// An owned [`Plane`] from its refined code and the input's unchanged scales; `O(o·n)` time and space.
fn into_plane(s: &[i64], idx: Vec<u16>, p: PlaneRef<'_>) -> Plane {
    Plane {
        s: s.to_vec(),
        idx,
        r: p.r.to_vec(),
        c: p.c.to_vec(),
        d: p.d.to_vec(),
    }
}

/// The 256 `E8 ∪ O8` codewords as Q32.32 words `codeword(j)[k]`. Constant size.
#[must_use]
pub fn codebook() -> [[i64; BLOCK]; CODE_LEN] {
    codebook_q()
}

/// Euclidean nearest codebook row of a grid 8-vector, first index on ties, via the codebook's own
/// `Q32.32` rule ([`e8::nearest`]). The ICM block argmin is not this rule: it minimises the block cost
/// in the `H_blk`-weighted metric.
/// # Errors
/// [`IcmError::Overflow`] for an exact `i128` distance overflow.
/// # Complexity
/// `O(256·8)` exact `i128` operations, `O(1)` space: constant in the caller's dimensions.
pub fn nearest_index(code: &[i64; BLOCK]) -> Result<usize, IcmError> {
    e8::nearest(code).map_err(|_| IcmError::Overflow)
}

/// The metric the ICM cost is minimised in: `Hl`, or `Hl` plus the `icm_ridge` diagonal shift
/// `icm_ridge · (n / 896) · mean(diag(Hl))` on the grid when `icm_ridge > 0`; a positive shift that
/// rounds below one raw unit is kept at one raw unit, as `steps::c_step` does.
/// # Errors
/// [`IcmError::Shape`] if `hl.len() != n*n` or `n == 0`; [`IcmError::Overflow`] off the grid.
/// # Complexity
/// `O(n²)` time and space: one copy, then `n` diagonal adds.
pub fn ridged_metric(hl: &[i64], n: usize, icm_ridge: i64) -> Result<Vec<i64>, IcmError> {
    if n == 0 || hl.len() != n * n {
        return Err(IcmError::Shape);
    }
    ridged_metric_q(hl, n, icm_ridge)
}

/// # Complexity
/// `O(n²)` time and space: one copy, `n` checked diagonal adds.
fn ridged_metric_q(hl: &[i64], n: usize, icm_ridge: i64) -> Result<Vec<i64>, IcmError> {
    let mut h = hl.to_vec();
    if icm_ridge > 0 {
        let mut sum: i128 = 0;
        for i in 0..n {
            sum = qsum(sum, i128::from(hl[i * n + i]))?;
        }
        let mean = q32::narrow(div_round(sum, n as i128)?).ok_or(IcmError::Overflow)?;
        // `n / 896` on the grid, then `icm_ridge · (n/896) · mean`.
        let nq = i128::try_from(n).map_err(|_| IcmError::Overflow)? << 32;
        let frac = q32::narrow(div_round(nq, 896)?).ok_or(IcmError::Overflow)?;
        let add = mul_q(mul_q(icm_ridge, frac)?, mean)?;
        let add = if mean > 0 && add < 1 { ONE } else { add };
        for i in 0..n {
            h[i * n + i] = add_q(h[i * n + i], add)?;
        }
    }
    Ok(h)
}

/// Signed `round(v / d)` (ties away from zero) for `d > 0`; [`IcmError::Overflow`] if `d <= 0` or an
/// intermediate leaves `i128`.
fn div_round(v: i128, d: i128) -> Result<i128, IcmError> {
    if d <= 0 {
        return Err(IcmError::Overflow);
    }
    let half = d / 2;
    if v >= 0 {
        v.checked_add(half).map(|x| x / d)
    } else {
        v.checked_sub(half).map(|x| x / d)
    }
    .ok_or(IcmError::Overflow)
}

/// `plane_w(P) = fwht128(s·d) · r_rep · c`, the plane's weight matrix, row-major `[o, n]`, on the
/// `Q32.32` grid.
/// # Errors
/// [`IcmError::Shape`] on a slice length that disagrees with `o`/`n`, or a value the grid cannot carry;
/// [`IcmError::Width`] if `n` is not a positive multiple of [`BLOCK`] and [`GROUP`].
/// # Complexity
/// `O(o·n·log2(GROUP))` additions for the transform plus `O(o·n)` checked multiplies; `O(o·n)` space.
pub fn plane_w(p: PlaneRef<'_>, o: usize, n: usize) -> Result<Vec<i64>, IcmError> {
    check_plane(p, o, n)?;
    plane_w_q(p.s, p.r, p.c, p.d, o, n)
}

/// # Complexity
/// `O(o·n)` checked multiplies plus `O(o·n·log2(GROUP))` transform additions; `O(o·n)` space.
pub(super) fn code_transform_q(
    s: &[i64],
    d: &[i64],
    o: usize,
    n: usize,
) -> Result<Vec<i64>, IcmError> {
    let mut t = s.to_vec();
    scale_by(&mut t, o, n, d, n / GROUP, ScaleBy::Column)?;
    fwht_rows(&mut t, o, n)?;
    Ok(t)
}

/// # Complexity
/// As [`plane_w`].
fn plane_w_q(
    s: &[i64],
    r: &[i64],
    c: &[i64],
    d: &[i64],
    o: usize,
    n: usize,
) -> Result<Vec<i64>, IcmError> {
    let g = n / GROUP;
    let mut t = code_transform_q(s, d, o, n)?;
    scale_by(&mut t, o, n, c, g, ScaleBy::Column)?;
    scale_by(&mut t, o, n, r, g, ScaleBy::Group)?;
    Ok(t)
}

/// `J(Ws, W, Hl) = Σ_o g_row[o] · (D_o Hl · D_o)`, `D = Ws − W` on the grid, as the `Q32.32`
/// word of the sum; unweighted when `g_row` is `None`.
/// # Errors
/// [`IcmError::Shape`] on a slice length that disagrees with `o`/`n`; [`IcmError::Overflow`] off the grid.
/// # Complexity
/// `O(o·n²)` multiply-adds (the `D·Hl` product runs on the engine pool), `O(o·n)` space.
pub fn j_rows(
    ws: &[i64],
    w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
) -> Result<i64, IcmError> {
    check_inputs(ws, hl, o, n, g_row)?;
    if w.len() != o * n {
        return Err(IcmError::Shape);
    }
    j_rows_q(ws, w, hl, o, n, g_row)
}

/// # Complexity
/// `O(o·n)` multiply-adds in the row reduction (serial in row order, so the sum is deterministic); the
/// `D·Hl` product it consumes is the pooled [`matmul_q`].
fn j_rows_q(
    ws: &[i64],
    w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
) -> Result<i64, IcmError> {
    let mut d = ws.to_vec();
    bin_into(&mut d, w, BinOp::Sub)?;
    let prod = matmul_q(&d, hl, o, n, n, false)?;
    let mut total: i128 = 0;
    for row in 0..o {
        let (dr, pr) = (&d[row * n..(row + 1) * n], &prod[row * n..(row + 1) * n]);
        let j = round_shift(dot64(dr, pr)?, 32)?;
        let j = match g_row {
            Some(g) => mul_q(g[row], j)?,
            None => j,
        };
        total = qsum(total, i128::from(j))?;
    }
    q32::narrow(total).ok_or(IcmError::Overflow)
}

/// One plane's iterated conditional modes: `sweeps` passes over the `n / 8` blocks, each replacing one
/// block's code by the codebook row minimising the block cost in the (optionally `icm_ridge`d) metric,
/// block order reversed on even sweeps. `r`, `c` and `d` are returned unchanged.
/// # Errors
/// [`IcmError::Shape`] on a slice length that disagrees with `o`/`n`; [`IcmError::Width`] if `n` is not a positive multiple of [`BLOCK`] `*` [`GROUP`]; [`IcmError::Overflow`] off the grid.
/// # Complexity
/// `sweeps · (n/8) · O · (CODE_LEN·BLOCK + BLOCK² + BLOCK·n)` checked multiply-adds over the `n/8`
/// independent blocks of each row, plus `O(o·n²)` setup for `Ws@Hl` and the transforms; `O(n²)` for
/// `H_rot`/`q_all` plus `O(o·n)`.
pub fn e8_icm_step(
    ws: &[i64],
    hl: &[i64],
    p: PlaneRef<'_>,
    o: usize,
    n: usize,
    sweeps: usize,
    icm_ridge: i64,
) -> Result<Plane, IcmError> {
    check_inputs(ws, hl, o, n, None)?;
    check_plane(p, o, n)?;
    let metric = ridged_metric_q(hl, n, icm_ridge)?;
    let (g, nb, cb) = (n / GROUP, n / BLOCK, codebook_q());
    // G_rot = fwht128((Ws @ Hl) * c) * d, H_rot = had_gram(Hl ∘ c c^T) ∘ d d^T
    let mut g_rot = matmul_q(ws, &metric, o, n, n, false)?;
    scale_by(&mut g_rot, o, n, p.c, g, ScaleBy::Column)?;
    fwht_rows(&mut g_rot, o, n)?;
    scale_by(&mut g_rot, o, n, p.d, g, ScaleBy::Column)?;
    let h_rot = cross_metric_q(&metric, n, p.c, p.c, p.d, p.d)?;
    // B = s * r_rep, F_res = G_rot - B @ H_rot
    let mut b = p.s.to_vec();
    scale_by(&mut b, o, n, p.r, g, ScaleBy::Group)?;
    let mut f_res = g_rot;
    bin_into(
        &mut f_res,
        &matmul_q(&b, &h_rot, o, n, n, false)?,
        BinOp::Sub,
    )?;
    let q_all = block_quads_q(&h_rot, &cb, nb, n)?;
    let (mut s_cur, mut idx_cur) = (p.s.to_vec(), p.idx.to_vec());
    let cx = SweepCtx {
        h_rot: &h_rot,
        q_all: &q_all,
        cb: &cb,
        n,
        g,
        sweeps,
    };
    let per = rows_per_task(o);
    for_chunks3(
        &mut f_res,
        per * n,
        &mut s_cur,
        per * n,
        &mut idx_cur,
        per * nb,
        &SweepTask {
            cx: &cx,
            r: p.r,
            per,
        },
    )?;
    Ok(into_plane(&s_cur, idx_cur, p))
}

/// `Q_all[m][j] = C_jᵀ H_blk[m] C_j` over the `n/8` diagonal blocks; `O((n/8)·CODE_LEN·BLOCK²)`.
fn block_quads_q(
    h: &[i64],
    cb: &[[i64; BLOCK]; CODE_LEN],
    nb: usize,
    n: usize,
) -> Result<Vec<i64>, IcmError> {
    let mut out = vec![0_i64; nb * CODE_LEN];
    for m in 0..nb {
        let hb = block_of_q(h, n, m);
        for j in 0..CODE_LEN {
            let t = mat_blk_row(&cb[j], &hb)?;
            out[m * CODE_LEN + j] = round_shift(row_dot(&t, &cb[j])?, 64)?;
        }
    }
    Ok(out)
}

/// `X_m[i₁,i₂] = C_i₁ᵀ H_blk[m] C_i₂`; `O((n/8)·CODE_LEN²·BLOCK)` multiply-adds, `O((n/8)·CODE_LEN²)` space.
fn block_cross_q(
    h: &[i64],
    cb: &[[i64; BLOCK]; CODE_LEN],
    nb: usize,
    n: usize,
) -> Result<Vec<i64>, IcmError> {
    let mut out = vec![0_i64; nb * CODE_LEN * CODE_LEN];
    for m in 0..nb {
        let hb = block_of_q(h, n, m);
        for i1 in 0..CODE_LEN {
            let tmp = mat_blk_row(&cb[i1], &hb)?;
            for i2 in 0..CODE_LEN {
                out[(m * CODE_LEN + i1) * CODE_LEN + i2] =
                    round_shift(row_dot(&tmp, &cb[i2])?, 64)?;
            }
        }
    }
    Ok(out)
}

/// The read-only operands of [`sweep_rows`].
struct SweepCtx<'a> {
    h_rot: &'a [i64],
    q_all: &'a [i64],
    cb: &'a [[i64; BLOCK]; CODE_LEN],
    n: usize,
    g: usize,
    sweeps: usize,
}

/// Pool task of [`e8_icm_step`]'s row split: `cx`, the per-row scales `r` (`[o, g]`) and rows per task.
struct SweepTask<'a> {
    cx: &'a SweepCtx<'a>,
    r: &'a [i64],
    per: usize,
}

impl Task3<i64, i64, u16, IcmError> for SweepTask<'_> {
    fn run(&self, t: usize, f: &mut [i64], sc: &mut [i64], ix: &mut [u16]) -> Result<(), IcmError> {
        let (g, lo) = (self.cx.g, t * self.per);
        let hi = (lo + f.len() / self.cx.n) * g;
        sweep_rows(self.cx, f, sc, ix, &self.r[lo * g..hi])
    }
}

/// # Complexity
/// `sweeps · (n/8) · rows · (CODE_LEN·BLOCK + BLOCK² + 2·BLOCK·n)` checked multiply-adds: the residual
/// update dominates, `2·BLOCK·n` over eight contiguous `H_rot` rows. `O(BLOCK)` extra space.
fn sweep_rows(
    cx: &SweepCtx<'_>,
    f_res: &mut [i64],
    s_cur: &mut [i64],
    idx: &mut [u16],
    r: &[i64],
) -> Result<(), IcmError> {
    let (n, nb) = (cx.n, cx.n / BLOCK);
    let rows = r.len() / cx.g;
    for sw in 0..cx.sweeps {
        for step in 0..nb {
            let m = if sw % 2 == 0 { nb - 1 - step } else { step };
            let b0 = m * BLOCK;
            let hb = block_of_q(cx.h_rot, n, m);
            for row in 0..rows {
                let a = r[row * cx.g + b0 / GROUP];
                let f_row = &mut f_res[row * n..(row + 1) * n];
                let s_row = &mut s_cur[row * n..(row + 1) * n];
                let mut f_m = [0_i64; BLOCK];
                for k in 0..BLOCK {
                    let mut acc: i128 = 0;
                    for l in 0..BLOCK {
                        acc = qsum(acc, triple(a, s_row[b0 + l], hb[l][k])?)?;
                    }
                    f_m[k] = add_q(f_row[b0 + k], round_shift(acc, 64)?)?;
                }
                let (mut best, mut best_cost) = (0_usize, i128::MAX);
                for j in 0..CODE_LEN {
                    let cost = block_cost(a, cx.q_all[m * CODE_LEN + j], dot64(&f_m, &cx.cb[j])?)?;
                    if cost < best_cost {
                        best_cost = cost;
                        best = j;
                    }
                }
                let mut coef = [0_i128; BLOCK];
                for k in 0..BLOCK {
                    let diff = sub_q(cx.cb[best][k], s_row[b0 + k])?;
                    coef[k] = i128::from(a) * i128::from(diff);
                    s_row[b0 + k] = cx.cb[best][k];
                }
                idx[row * nb + m] = best as u16;
                for col in 0..n {
                    let mut acc: i128 = 0;
                    for k in 0..BLOCK {
                        acc = qsum(
                            acc,
                            qmul(coef[k], i128::from(cx.h_rot[(b0 + k) * n + col]))?,
                        )?;
                    }
                    f_row[col] = sub_q(f_row[col], round_shift(acc, 64)?)?;
                }
            }
        }
    }
    Ok(())
}

/// Joint block ICM over two planes: `sweeps` passes over the `n / 8` blocks (or the `blocks` subset, in
/// the given order), each replacing both planes' codes of one block by the pair `(i₁, i₂)` minimising
/// `2 a₁a₂ X_m[i₁,i₂] + A[i₁] + B[i₂]`. `row_chunk` bounds the rows one pool task works on.
/// # Errors
/// [`IcmError::Shape`] on a slice length that disagrees with `o`/`n`; [`IcmError::Width`] if `n` is not
/// a positive multiple of [`BLOCK`] and [`GROUP`]; [`IcmError::ZeroChunk`] if `row_chunk == 0`;
/// [`IcmError::BlockOutOfRange`] for a block index `>= n / BLOCK`; [`IcmError::Overflow`] off the grid.
/// # Complexity
/// `sweeps · (n/8) · O · (CODE_LEN² + 2·CODE_LEN·BLOCK + 2·BLOCK·n)` checked multiply-adds for the pair
/// argmin and the two residual updates, plus `O(n²·log2(GROUP))` for the three `had_gram` passes and
/// `(n/8)·CODE_LEN²·BLOCK` for `X`; rows split across the pool. `O(o·n)` for `s`, `F`, `G`, `B`,
/// `O(n²)` for `H11/H22/H12`, `(n/8)·CODE_LEN²` `i64` for `X`.
#[allow(
    clippy::too_many_arguments,
    reason = "the option surface (sweeps, icm_ridge, g_row, row_chunk, blocks) mirrors the Python \
              signature; a bag struct would hide which arguments are metric options and which are \
              shapes"
)]
pub fn e8_joint_icm_2plane(
    ws: &[i64],
    hl: &[i64],
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    o: usize,
    n: usize,
    sweeps: usize,
    icm_ridge: i64,
    g_row: Option<&[i64]>,
    row_chunk: usize,
    blocks: Option<&[usize]>,
) -> Result<(Plane, Plane, JointStats), IcmError> {
    check_inputs(ws, hl, o, n, g_row)?;
    check_plane(p1, o, n)?;
    check_plane(p2, o, n)?;
    if row_chunk == 0 {
        return Err(IcmError::ZeroChunk);
    }
    let nb = n / BLOCK;
    if let Some(bs) = blocks {
        for &m in bs {
            if m >= nb {
                return Err(IcmError::BlockOutOfRange);
            }
        }
    }
    let metric = ridged_metric_q(hl, n, icm_ridge)?;
    let (g, cb) = (n / GROUP, codebook_q());
    let mut wsum = plane_w_q(p1.s, p1.r, p1.c, p1.d, o, n)?;
    bin_into(
        &mut wsum,
        &plane_w_q(p2.s, p2.r, p2.c, p2.d, o, n)?,
        BinOp::Add,
    )?;
    let j_before = j_rows_q(ws, &wsum, &metric, o, n, g_row)?;
    let (mut s1, mut s2) = (p1.s.to_vec(), p2.s.to_vec());
    let (mut idx1, mut idx2) = (p1.idx.to_vec(), p2.idx.to_vec());
    // G_k = fwht128((Ws @ Hl) * c_k) * d_k
    let wh = matmul_q(ws, &metric, o, n, n, false)?;
    let mut g1 = wh.clone();
    scale_by(&mut g1, o, n, p1.c, g, ScaleBy::Column)?;
    fwht_rows(&mut g1, o, n)?;
    scale_by(&mut g1, o, n, p1.d, g, ScaleBy::Column)?;
    let mut g2 = wh;
    scale_by(&mut g2, o, n, p2.c, g, ScaleBy::Column)?;
    fwht_rows(&mut g2, o, n)?;
    scale_by(&mut g2, o, n, p2.d, g, ScaleBy::Column)?;
    // H_jk = had_gram(Hl ∘ c_j c_k^T) ∘ d_j d_k^T
    let h11 = cross_metric_q(&metric, n, p1.c, p1.c, p1.d, p1.d)?;
    let h22 = cross_metric_q(&metric, n, p2.c, p2.c, p2.d, p2.d)?;
    let h12 = cross_metric_q(&metric, n, p1.c, p2.c, p1.d, p2.d)?;
    // F_1 = G_1 - B_1 H_11 - B_2 H_12^T, F_2 = G_2 - B_2 H_22 - B_1 H_12
    let mut b1 = s1.clone();
    scale_by(&mut b1, o, n, p1.r, g, ScaleBy::Group)?;
    let mut b2 = s2.clone();
    scale_by(&mut b2, o, n, p2.r, g, ScaleBy::Group)?;
    let mut f1 = g1;
    bin_into(&mut f1, &matmul_q(&b1, &h11, o, n, n, false)?, BinOp::Sub)?;
    bin_into(&mut f1, &matmul_q(&b2, &h12, o, n, n, true)?, BinOp::Sub)?;
    let mut f2 = g2;
    bin_into(&mut f2, &matmul_q(&b2, &h22, o, n, n, false)?, BinOp::Sub)?;
    bin_into(&mut f2, &matmul_q(&b1, &h12, o, n, n, false)?, BinOp::Sub)?;
    let (q1v, q2v) = (
        block_quads_q(&h11, &cb, nb, n)?,
        block_quads_q(&h22, &cb, nb, n)?,
    );
    let x_all = block_cross_q(&h12, &cb, nb, n)?;
    let (moves, first) = (AtomicUsize::new(0), FirstErr::<usize, IcmError>::new());
    let mut rows: Vec<JointRow<'_>> = f1
        .chunks_mut(n)
        .zip(f2.chunks_mut(n))
        .zip(s1.chunks_mut(n))
        .zip(s2.chunks_mut(n))
        .zip(idx1.chunks_mut(nb))
        .zip(idx2.chunks_mut(nb))
        .map(|(((((f1, f2), s1), s2), idx1), idx2)| JointRow {
            f1,
            f2,
            s1,
            s2,
            idx1,
            idx2,
        })
        .collect();
    let per = row_chunk.min(rows_per_task(o)).max(1);
    let task = JointSweep {
        h11: &h11,
        h22: &h22,
        h12: &h12,
        q1: &q1v,
        q2: &q2v,
        x_all: &x_all,
        cb: &cb,
        r1: p1.r,
        r2: p2.r,
        blocks,
        n,
        g,
        nb,
        sweeps,
        first: &first,
        moves: &moves,
    };
    chunks_mut_global(&mut rows, per, &task);
    if let Some((_, e)) = first.take() {
        return Err(e);
    }
    let mut wsum = plane_w_q(&s1, p1.r, p1.c, p1.d, o, n)?;
    bin_into(
        &mut wsum,
        &plane_w_q(&s2, p2.r, p2.c, p2.d, o, n)?,
        BinOp::Add,
    )?;
    let stats = JointStats {
        j_before,
        j_after: j_rows_q(ws, &wsum, &metric, o, n, g_row)?,
        moves: moves.load(Ordering::Relaxed),
    };
    Ok((into_plane(&s1, idx1, p1), into_plane(&s2, idx2, p2), stats))
}

/// One row of the joint sweep's mutable state: two residual fields, two sign fields, two code rows.
struct JointRow<'a> {
    f1: &'a mut [i64],
    f2: &'a mut [i64],
    s1: &'a mut [i64],
    s2: &'a mut [i64],
    idx1: &'a mut [u16],
    idx2: &'a mut [u16],
}

/// The read-only operands and counters of [`JointSweep`].
struct JointSweep<'a> {
    h11: &'a [i64],
    h22: &'a [i64],
    h12: &'a [i64],
    q1: &'a [i64],
    q2: &'a [i64],
    x_all: &'a [i64],
    cb: &'a [[i64; BLOCK]; CODE_LEN],
    r1: &'a [i64],
    r2: &'a [i64],
    blocks: Option<&'a [usize]>,
    n: usize,
    g: usize,
    nb: usize,
    sweeps: usize,
    first: &'a FirstErr<usize, IcmError>,
    moves: &'a AtomicUsize,
}

impl ChunkTask<JointRow<'_>> for JointSweep<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [JointRow<'_>]) {
        let mut moved = 0_usize;
        for (rr, row) in piece.iter_mut().enumerate() {
            match joint_row(self, row, start + rr) {
                Ok(m) => moved += m,
                Err(e) => {
                    self.first.record(start, e);
                    break;
                }
            }
        }
        self.moves.fetch_add(moved, Ordering::Relaxed);
    }
}

/// # Complexity
/// `sweeps · nb · (CODE_LEN² + 2·CODE_LEN·BLOCK + 2·BLOCK·n)` checked multiply-adds; `O(CODE_LEN)`
/// space.
fn joint_row(cx: &JointSweep<'_>, row: &mut JointRow<'_>, rowno: usize) -> Result<usize, IcmError> {
    let (n, nb) = (cx.n, cx.nb);
    let mut moved = 0_usize;
    for sw in 0..cx.sweeps {
        let count = cx.blocks.map_or(nb, <[usize]>::len);
        for step in 0..count {
            let m = match cx.blocks {
                Some(bs) => bs[step],
                None if sw % 2 == 0 => nb - 1 - step,
                None => step,
            };
            let (b0, gidx) = (m * BLOCK, m * BLOCK / GROUP);
            let (hb11, hb22, hb12) = (
                block_of_q(cx.h11, n, m),
                block_of_q(cx.h22, n, m),
                block_of_q(cx.h12, n, m),
            );
            let xm = &cx.x_all[m * CODE_LEN * CODE_LEN..(m + 1) * CODE_LEN * CODE_LEN];
            let (a1, a2) = (cx.r1[rowno * cx.g + gidx], cx.r2[rowno * cx.g + gidx]);
            let (mut f1m, mut f2m) = ([0_i64; BLOCK], [0_i64; BLOCK]);
            for k in 0..BLOCK {
                let (mut acc1, mut acc2) = (0_i128, 0_i128);
                for l in 0..BLOCK {
                    let (s1o, s2o) = (row.s1[b0 + l], row.s2[b0 + l]);
                    acc1 = qsum(acc1, triple(a1, s1o, hb11[l][k])?)?;
                    acc1 = qsum(acc1, triple(a2, s2o, hb12[k][l])?)?;
                    acc2 = qsum(acc2, triple(a2, s2o, hb22[l][k])?)?;
                    acc2 = qsum(acc2, triple(a1, s1o, hb12[l][k])?)?;
                }
                f1m[k] = add_q(row.f1[b0 + k], round_shift(acc1, 64)?)?;
                f2m[k] = add_q(row.f2[b0 + k], round_shift(acc2, 64)?)?;
            }
            let (mut av, mut bv) = ([0_i128; CODE_LEN], [0_i128; CODE_LEN]);
            for j in 0..CODE_LEN {
                av[j] = block_cost(a1, cx.q1[m * CODE_LEN + j], dot64(&f1m, &cx.cb[j])?)?;
                bv[j] = block_cost(a2, cx.q2[m * CODE_LEN + j], dot64(&f2m, &cx.cb[j])?)?;
            }
            let (mut flat, mut best) = (0_usize, i128::MAX);
            for i1v in 0..CODE_LEN {
                for i2v in 0..CODE_LEN {
                    let t = pair_cost(a1, a2, xm[i1v * CODE_LEN + i2v], av[i1v], bv[i2v])?;
                    if t < best {
                        best = t;
                        flat = i1v * CODE_LEN + i2v;
                    }
                }
            }
            let (i1s, i2s) = (flat / CODE_LEN, flat % CODE_LEN);
            let (mut e1, mut e2) = ([0_i128; BLOCK], [0_i128; BLOCK]);
            let mut changed = false;
            for k in 0..BLOCK {
                let n1 = sub_q(cx.cb[i1s][k], row.s1[b0 + k])?;
                let n2 = sub_q(cx.cb[i2s][k], row.s2[b0 + k])?;
                e1[k] = i128::from(a1) * i128::from(n1);
                e2[k] = i128::from(a2) * i128::from(n2);
                changed |= n1 != 0 || n2 != 0;
                row.s1[b0 + k] = cx.cb[i1s][k];
                row.s2[b0 + k] = cx.cb[i2s][k];
            }
            if changed {
                moved += 1;
            }
            row.idx1[m] = i1s as u16;
            row.idx2[m] = i2s as u16;
            for col in 0..n {
                let (mut u1, mut u2) = (0_i128, 0_i128);
                for k in 0..BLOCK {
                    let h11v = i128::from(cx.h11[(b0 + k) * n + col]);
                    let h22v = i128::from(cx.h22[(b0 + k) * n + col]);
                    let h12a = i128::from(cx.h12[(b0 + k) * n + col]);
                    let h12b = i128::from(cx.h12[col * n + (b0 + k)]);
                    u1 = qsum(u1, qsum(qmul(e1[k], h11v)?, qmul(e2[k], h12b)?)?)?;
                    u2 = qsum(u2, qsum(qmul(e2[k], h22v)?, qmul(e1[k], h12a)?)?)?;
                }
                row.f1[col] = sub_q(row.f1[col], round_shift(u1, 64)?)?;
                row.f2[col] = sub_q(row.f2[col], round_shift(u2, 64)?)?;
            }
        }
    }
    Ok(moved)
}

/// Joint least-squares solve for both planes' group scales `(r1, r2)`: per row a `2G × 2G` normal
/// system `M x = t` minimising `J(Ws, plane_w(s1,r1,c1,d1) + plane_w(s2,r2,c2,d2), Hl)`, with a
/// `1e-12 · mean(diag M)` diagonal ridge (kept at one raw unit when it would round below the grid).
/// `s1`/`c1` and `s2`/`c2` are the raw code values and column scales; `d` does not enter (the group
/// scale is applied after the transform; the ported reference passes `s_k · c_k` and the metric `Hl`).
/// # Errors
/// [`IcmError::Shape`] on a slice length that disagrees with `o`/`n`; [`IcmError::Width`] if `n` is not
/// a positive multiple of [`BLOCK`] and [`GROUP`]; [`IcmError::NotPositiveDefinite`] if the per-row
/// normal matrix is not positive-definite after the ridge; [`IcmError::Overflow`] off the grid.
/// # Complexity
/// `2G` row-panel products of `o × 128` by `128 × n` (`2G·o·128·n` checked multiply-adds) plus `o`
/// solves of `2G × 2G` (`o·(2G)³/3`), rows split across the pool; `O(o·(2G)²)` for the per-row normal
/// matrix plus `O(o·n)`.
pub fn r_step2(
    ws: &[i64],
    hl: &[i64],
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    o: usize,
    n: usize,
) -> Result<(Vec<i64>, Vec<i64>), IcmError> {
    check_inputs(ws, hl, o, n, None)?;
    check_plane(p1, o, n)?;
    check_plane(p2, o, n)?;
    let (g, per) = (n / GROUP, rows_per_task(o));
    let wh = matmul_q(ws, hl, o, n, n, false)?;
    let mut r1_out = vec![0_i64; o * g];
    let mut r2_out = vec![0_i64; o * g];
    let task = RStepRows {
        hl,
        wh: &wh,
        p1,
        p2,
        n,
        g,
        per,
    };
    for_chunks2(&mut r1_out, per * g, &mut r2_out, per * g, &task)?;
    Ok((r1_out, r2_out))
}

/// The pool task of [`r_step2`]: one band of rows of the two normal systems.
struct RStepRows<'a> {
    hl: &'a [i64],
    wh: &'a [i64],
    p1: PlaneRef<'a>,
    p2: PlaneRef<'a>,
    n: usize,
    g: usize,
    per: usize,
}

impl Task2<i64, i64, IcmError> for RStepRows<'_> {
    fn run(&self, t: usize, r1: &mut [i64], r2: &mut [i64]) -> Result<(), IcmError> {
        let (g, row0) = (self.g, t * self.per);
        for rr in 0..r1.len() / g {
            let (a, b) = r_row_q(self.hl, self.wh, self.p1, self.p2, self.n, g, row0 + rr)?;
            r1[rr * g..(rr + 1) * g].copy_from_slice(&a);
            r2[rr * g..(rr + 1) * g].copy_from_slice(&b);
        }
        Ok(())
    }
}

/// The Q32.32 dot of a `GROUP`-word window against a `GROUP`-long window of stride `stride`, one rounding.
fn dot_window(a: &[i64], h: &[i64], stride: usize) -> Result<i64, IcmError> {
    let mut acc: i128 = 0;
    for k in 0..GROUP {
        acc = qsum(acc, i128::from(a[k]) * i128::from(h[k * stride]))?;
    }
    round_shift(acc, 32)
}

/// # Complexity
/// `2G·GROUP` exact panel products and one fill shift, `2G·G·GROUP²` checked multiply-adds for the metric products and the right-hand side, plus
/// the `2G × 2G` solve (`(2G)³/3`); `O(2G·n + g·GROUP)` space.
fn r_row_q(
    hl: &[i64],
    wh: &[i64],
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    n: usize,
    g: usize,
    row: usize,
) -> Result<(Vec<i64>, Vec<i64>), IcmError> {
    let two_g = 2 * g;
    // P[a_, j]: `a_ < G` is plane 1's block `a_`, `a_ >= G` plane 2's block `a_ - G`.
    // `P` is filled from below so `max|P| ∈ (2³⁴, 2³⁵]` raw (`steps`' panel rule): a top row's tiny
    // `s·c` would otherwise round `P·Hl·Pᵀ` under the quantum and refuse a positive-definite system.
    // `M' = 2²ᵏM` and `t` is read at `2²ᵏ` (`2ᵏP'·w`, one rounding), so the solve returns `x` itself
    // at the grid quantum.
    let mut prod = vec![0_i128; two_g * GROUP];
    for a_ in 0..two_g {
        let (s, c) = if a_ < g { (p1.s, p1.c) } else { (p2.s, p2.c) };
        let aa = a_ % g;
        for j in 0..GROUP {
            prod[a_ * GROUP + j] =
                i128::from(s[row * n + aa * GROUP + j]) * i128::from(c[aa * GROUP + j]);
        }
    }
    let max = prod.iter().fold(0_u128, |m, v| m.max(v.unsigned_abs()));
    let k = ilog2_pos(max).map_or(0, |l| 66_u32.saturating_sub(l));
    let fill = |v: i128| -> Result<i64, IcmError> {
        if k <= 32 {
            round_shift(v, 32 - k)
        } else {
            q32::narrow(qmul(v, 1_i128 << (k - 32))?).ok_or(IcmError::Overflow)
        }
    };
    let pv = prod
        .iter()
        .map(|&v| fill(v))
        .collect::<Result<Vec<i64>, IcmError>>()?;
    let mut kvec = vec![0_i64; g * GROUP];
    let mut mmat = vec![0_i64; two_g * two_g];
    for a_ in 0..two_g {
        let (ga, pa) = (a_ % g, &pv[a_ * GROUP..(a_ + 1) * GROUP]);
        for gidx in 0..g {
            for j in 0..GROUP {
                let h = &hl[(ga * GROUP) * n + gidx * GROUP + j..];
                kvec[gidx * GROUP + j] = dot_window(pa, h, n)?;
            }
        }
        for bcol in 0..two_g {
            let kb = &kvec[(bcol % g) * GROUP..(bcol % g) * GROUP + GROUP];
            mmat[a_ * two_g + bcol] = dot_window(kb, &pv[bcol * GROUP..(bcol + 1) * GROUP], 1)?;
        }
    }
    let mut tvec = vec![0_i64; two_g];
    for a_ in 0..two_g {
        let ga = a_ % g;
        let p = &pv[a_ * GROUP..(a_ + 1) * GROUP];
        let w = &wh[row * n + ga * GROUP..];
        let mut acc: i128 = 0;
        for j in 0..GROUP {
            acc = qsum(acc, i128::from(p[j]) * i128::from(w[j]))?;
        }
        tvec[a_] = fill(acc)?;
    }
    let mut mean: i128 = 0;
    for k in 0..two_g {
        mean = qsum(mean, i128::from(mmat[k * two_g + k]))?;
    }
    let mean = q32::narrow(div_round(mean, two_g as i128)?).ok_or(IcmError::Overflow)?;
    // `1e-12 · mean(diag)`, floored at one raw unit when a positive ridge would round to zero.
    let add =
        q32::narrow(div_round(i128::from(mean), 1_000_000_000_000)?).ok_or(IcmError::Overflow)?;
    let add = if mean > 0 && add < 1 { 1 } else { add };
    for k in 0..two_g {
        mmat[k * two_g + k] = add_q(mmat[k * two_g + k], add)?;
    }
    let solved = chol_solve(&mmat, &tvec, two_g, 1).map_err(map_solve)?;
    Ok((solved.x[..g].to_vec(), solved.x[g..].to_vec()))
}

/// Validate `o`, `n` and every slice length of a plane.
fn check_plane(p: PlaneRef<'_>, o: usize, n: usize) -> Result<(), IcmError> {
    if o == 0 || n == 0 {
        return Err(IcmError::Shape);
    }
    if !n.is_multiple_of(BLOCK) || !n.is_multiple_of(GROUP) {
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

/// Row-major `[rows, inner] × [inner, cols]` product on the shared `Q32.32` path. The panel scale
/// rule and the grid rounding are [`product_panels`]'s, and the exact power-of-two undo is
/// [`shl_q32`], so a product outside the `i64` grid range is still representable.
/// # Errors
/// [`IcmError::Shape`] for an empty dimension or a length that disagrees with the shape;
/// [`IcmError::Overflow`] for a product that leaves the grid.
/// # Complexity
/// As [`product_panels`]: `O(rows·inner·cols)` pool multiply-accumulates, `O(rows·cols)` space.
pub fn matmul(
    a: &[i64],
    b: &[i64],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<i64>, IcmError> {
    matmul_q(a, b, rows, inner, cols, false)
}
