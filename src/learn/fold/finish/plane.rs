//! The finishing stage's coded weight plane: the frozen `E8 ∪ O8` code field `S`, a frozen per-row
//! scale grid `(lo, sp)` and the trainable log-multiplicative latents `(pa, pb, pd)`, ported from
//! `spec/pv_finish.py:205-285` over the codebook of [`crate::learn::fold::fit::icm`]. A linear's
//! weight is `compose(fwht128(S ⊙ d), r, c)`.
//!
//! # Representation
//!
//! Every stored value and intermediate is Q32.32 `i64`; products are `i128` Q64.64 rounded once
//! ([`round_q32_even`], ties to even); a reduction accumulates in `i128` and rounds once. The pack,
//! the offsets and [`encode`]'s field are words too: the loader decodes the checkpoint to Q32.32
//! ([`crate::learn::fold::run::convert`] refuses a non-finite or off-grid decode) before a plane is
//! built, so no `f64` enters here. Against an `f64` evaluation the forward applies four roundings
//! (the `S ⊙ d` product, the transform's scale, the two scale products) while the butterfly is
//! exact addition, so `|w_q − w_ref| ≤ 8·2^-32·max(1, max|w_ref|)`; the `r` snap produces exactly
//! [`crate::learn::fold::fit::quant::r_level`]'s integer level. Bases are canonicalised through the
//! plane's own quantizers, so `qvals() == latents()` at zero offsets.
//!
//! # Execution
//!
//! Pooled work is the whole-tensor forward and backward: the `o × n` scale, compose and
//! Walsh–Hadamard passes ([`Pass`], `o·n·log2(128) ≥ 10^6` word ops at a fitted shape). Every
//! per-row, per-group and per-element vector loop is a serial integer loop, including the
//! construction-time codebook [`encode`] (`o·n·256` multiply-adds, run once per plane).
//!
//! # Partiality
//!
//! [`CodedPlane::from_pack`] and [`CodedPlane::set_offsets`] validate shape and finiteness with a
//! [`PlaneError`]; `n` must be a positive multiple of [`GROUP`]. An allocator that returns a bare
//! `Vec` signals a refusal with an empty vector, caught by the caller's length check. `encode`'s
//! argmax keeps the reference's lowest-index tie rule.
//!
//! Grade `tested` (`tests/learn_fold_plane.rs`); nothing here is extracted or refined.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::must_use_candidate,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::unreadable_literal,
    reason = "the kernels index row-major `[o, n]` buffers by `o/i/j/k/m/gg` and name the carriers \
              `r0`/`c0`/`d0`, `pa`/`pb`/`pd`, `lo`/`sp`; the `u8` code index cast is exact \
              (`best < 256`); every `i128 → i64` narrowing is checked by `narrow`."
)]

use crate::engine::chart::pool::{chunks_mut_global, width_global, FirstErr};
use crate::engine::chart::task::ChunkTask;
use crate::learn::code::e8::codeword;
use crate::learn::fold::fit::icm::{BLOCK, CODE_LEN};
use crate::learn::fold::fit::steps::GROUP;
use crate::learn::solve::q32::{narrow, round_f16, round_q32_even, rshift_rne, Fx};
use crate::numeric::fixed_exp::fixed_exp;

/// `ONE`, the Q32.32 unit: the word `1 << 32` denotes `1.0`.
pub(crate) const ONE: i64 = 1 << 32;

/// `round(2^32 / √128)`: the normalising scale of the 128-point Walsh–Hadamard transform.
const INV_SQRT128_Q32: i64 = 379_625_062;

/// `round(a·b / 2^32)`: the Q32.32 product, or `None` when it leaves `i64`. `O(1)`.
pub(crate) fn mul(a: i64, b: i64) -> Option<i64> {
    round_q32_even(i128::from(a) * i128::from(b)).ok()
}

/// `round(n / d)`, ties to even, or `None` when `d == 0` or the result leaves `i128`. `O(1)`.
pub(crate) fn div_ne(n: i128, d: i128) -> Option<i128> {
    if d == 0 {
        return None;
    }
    let (q, r) = (n / d, n % d);
    if r == 0 {
        return Some(q);
    }
    let up = if r.abs().checked_mul(2)? > d.abs() || (r.abs() * 2 == d.abs() && q % 2 != 0) {
        if (r < 0) == (d < 0) {
            1
        } else {
            -1
        }
    } else {
        0
    };
    q.checked_add(up)
}

/// `round(a / b)` as a Q32.32 word, or `None` when `b == 0` or the result leaves `i64`. `O(1)`.
pub(crate) fn div(a: i64, b: i64) -> Option<i64> {
    narrow(div_ne(i128::from(a) << 32, i128::from(b))?)
}

/// A refusal from a [`CodedPlane`] constructor or method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneError {
    /// A slice length disagrees with the declared shape, a `rows` list is not `o` long, or `n` is
    /// not a positive multiple of [`GROUP`].
    Shape,
    /// An output row index in `rows` is not below the receiving weight's row count.
    RowOutOfRange,
    /// A checked word left the Q32.32 carrier: a quantizer, `exp` or `i128 → i64` narrowing.
    Range,
}

/// The latent offset gradients `(∂L/∂pa, ∂L/∂pb, ∂L/∂pd)` of [`CodedPlane::backward`], in Q32.32.
pub type LatentGrad = (Vec<i64>, Vec<i64>, Vec<i64>);

/// One stored plane, every field Q32.32: a frozen code field (`idx` into the codebook, or `s_fix`),
/// an optional output-row restriction `rows`, the frozen grid `(lo, sp)`, the bases `(r0, c0, d0)`
/// and the latent offsets `(pa, pb, pd)`; the latents start at `0`.
#[derive(Clone, Debug)]
pub struct CodedPlane {
    /// Out rows (`=|rows|` when `rows` is set).
    pub o: usize,
    /// In columns; a positive multiple of [`GROUP`].
    pub n: usize,
    /// Codebook row per 8-column block, row-major `[o, n / 8]`; empty when `s_fix` is set.
    pub idx: Vec<u8>,
    /// The exact code field `[o, n]` when some block is not a codebook row.
    pub s_fix: Option<Vec<i64>>,
    /// Output rows of the receiving weight this plane writes, or `None` for every row.
    pub rows: Option<Vec<usize>>,
    /// Per-row lower endpoint of the frozen `r` grid, length `o`.
    pub lo: Vec<i64>,
    /// Per-row width of the frozen `r` grid, length `o`.
    pub sp: Vec<i64>,
    /// Base group scale, row-major `[o, n / GROUP]`.
    pub r0: Vec<i64>,
    /// Base column scale, length `n`.
    pub c0: Vec<i64>,
    /// Base Hadamard column scale, length `n`.
    pub d0: Vec<i64>,
    /// Group-scale log offset, row-major `[o, n / GROUP]`.
    pub pa: Vec<i64>,
    /// Column-scale log offset, length `n`.
    pub pb: Vec<i64>,
    /// Hadamard column-scale log offset, length `n`.
    pub pd: Vec<i64>,
}

impl CodedPlane {
    /// Build a plane from its pack: the rotated code field (`o × n`), the frozen `r` grid `(lo, sp)`,
    /// the scales `r` (`o × n/128`), `c`, `d` and an optional row restriction `rows`, canonicalised
    /// through the plane's own quantizers; the field becomes `idx` when every block equals its
    /// nearest codebook row, else `s_fix`. Every argument is a Q32.32 word.
    ///
    /// # Errors
    /// [`PlaneError::Shape`] on a shape or length disagreement; [`PlaneError::Range`] from a
    /// quantizer. `O(o·n·CODE_LEN)`.
    pub fn from_pack(
        s_rot: &[i64],
        lo: &[i64],
        sp: &[i64],
        r: &[i64],
        c: &[i64],
        d: &[i64],
        rows: Option<&[usize]>,
        o: usize,
        n: usize,
    ) -> Result<Self, PlaneError> {
        if o == 0 || n == 0 || !n.is_multiple_of(GROUP) {
            return Err(PlaneError::Shape);
        }
        let on = o.checked_mul(n).ok_or(PlaneError::Shape)?;
        let g = n / GROUP;
        let og = o.checked_mul(g).ok_or(PlaneError::Shape)?;
        let lens = [s_rot.len(), lo.len(), sp.len(), r.len(), c.len(), d.len()];
        if lens != [on, o, o, og, n, n] || rows.is_some_and(|rw| rw.len() != o) {
            return Err(PlaneError::Shape);
        }
        let (idx_all, exact) = encode(s_rot, o, n)?;
        let (idx, s_fix) = if exact {
            (idx_all, None)
        } else {
            (Vec::new(), Some(s_rot.to_vec()))
        };
        let mut plane = Self {
            o,
            n,
            idx,
            s_fix,
            rows: rows.map(<[usize]>::to_vec),
            lo: lo.to_vec(),
            sp: sp.to_vec(),
            r0: r.to_vec(),
            c0: c.to_vec(),
            d0: d.to_vec(),
            pa: vec![0_i64; og],
            pb: vec![0_i64; n],
            pd: vec![0_i64; n],
        };
        let (r_q, c_q, d_q) = plane.qvals();
        if r_q.len() != og || c_q.len() != n || d_q.len() != n {
            return Err(PlaneError::Range);
        }
        plane.r0 = r_q;
        plane.c0 = c_q;
        plane.d0 = d_q;
        Ok(plane)
    }

    /// Overwrite the log offsets `(pa, pb, pd)` with Q32.32 words.
    ///
    /// # Errors
    /// [`PlaneError::Shape`] on a length disagreement. Serial `O(o·n/128 + n)`.
    pub fn set_offsets(&mut self, pa: &[i64], pb: &[i64], pd: &[i64]) -> Result<(), PlaneError> {
        if self.o == 0 || self.n == 0 || !self.n.is_multiple_of(GROUP) {
            return Err(PlaneError::Shape);
        }
        let g = self.n / GROUP;
        if pa.len() != self.o * g || pb.len() != self.n || pd.len() != self.n {
            return Err(PlaneError::Shape);
        }
        self.pa.copy_from_slice(pa);
        self.pb.copy_from_slice(pb);
        self.pd.copy_from_slice(pd);
        Ok(())
    }

    /// The code field `S` (`o × n`, row-major): `s_fix` verbatim, else the codebook rows named by
    /// `idx`; empty on a shape disagreement. `O(o·n)`, serial.
    pub fn code_field(&self) -> Vec<i64> {
        if let Some(s) = &self.s_fix {
            return s.clone();
        }
        let nb = self.n / BLOCK;
        if self.idx.len() != self.o * nb {
            return Vec::new();
        }
        let cb = codebook_q32();
        let mut s = vec![0_i64; self.o * self.n];
        for (oi, row) in s.chunks_mut(self.n).enumerate() {
            for m in 0..nb {
                let src = &cb[usize::from(self.idx[oi * nb + m])];
                row[m * BLOCK..(m + 1) * BLOCK].copy_from_slice(src);
            }
        }
        s
    }

    /// The latents `(r0·exp(pa), c0·exp(pb), d0·exp(pd))`, Q32.32; empty on a refusal, serial
    /// `O(o·n/128 + n)`.
    pub fn latents(&self) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        (
            scale_exp(&self.r0, &self.pa),
            scale_exp(&self.c0, &self.pb),
            scale_exp(&self.d0, &self.pd),
        )
    }

    /// The quantized forward values `(r_q, c_q, d_q)`: `r` snapped to its frozen row grid, `c` and
    /// `d` through [`quant_c_q32`] at 6 and 5 bits; serial `O(o·n/128 + n)`.
    pub fn qvals(&self) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let (r_l, c_l, d_l) = self.latents();
        (
            snap_r(&r_l, &self.lo, &self.sp, self.o, self.n),
            quant_c_q32(&c_l, 6),
            quant_c_q32(&d_l, 5),
        )
    }

    /// The plane's weight `compose(fwht128(S ⊙ d_q), r_q, c_q)`, row-major `o × n`; empty on a
    /// shape violation. Temporal `O(o·n·log2(GROUP))`, the `o × n` passes pooled.
    pub fn weight(&self) -> Vec<i64> {
        let (r_q, c_q, d_q) = self.qvals();
        compose_plane(&self.code_field(), &r_q, &c_q, &d_q, self.o, self.n)
    }

    /// The unquantized forward `compose(fwht128(S ⊙ d_l), r_l, c_l)`, whose latent gradient is what
    /// [`CodedPlane::backward`] returns; complexity as [`CodedPlane::weight`].
    pub fn weight_unquantized(&self) -> Vec<i64> {
        let (r_l, c_l, d_l) = self.latents();
        compose_plane(&self.code_field(), &r_l, &c_l, &d_l, self.o, self.n)
    }

    /// The straight-through gradient: given `grad_w = ∂L/∂W`, return `(∂L/∂pa, ∂L/∂pb, ∂L/∂pd)`.
    /// With `H = fwht128`, `X = H(S ⊙ d_q)` and `G` the plane's gradient rows:
    /// `dr[o,g] = Σ_{j∈g} G·c_q·X`, `dc[j] = Σ_o G·r_q·X`, `dY = H(G ⊙ r_q ⊙ c_q)`,
    /// `dd[j] = Σ_o S[o,j]·dY[o,j]`; then `∂pa = dr·r_l`, `∂pb = dc·c_l`, `∂pd = dd·d_l`.
    ///
    /// # Errors
    /// [`PlaneError::Shape`] on a `grad_w` shape; [`PlaneError::RowOutOfRange`] for a `rows` entry
    /// past `grad_w`; [`PlaneError::Range`] on a checked word; `O(o·n·log2(GROUP))`, pooled.
    pub fn backward(&self, grad_w: &[i64]) -> Result<LatentGrad, PlaneError> {
        if self.o == 0 || self.n == 0 || !self.n.is_multiple_of(GROUP) {
            return Err(PlaneError::Shape);
        }
        let g = self.n / GROUP;
        let og = self.o * g;
        if !grad_w.len().is_multiple_of(self.n) {
            return Err(PlaneError::Shape);
        }
        let o_total = grad_w.len() / self.n;
        let mut gp = vec![0_i64; self.o * self.n];
        match &self.rows {
            None => {
                if o_total != self.o {
                    return Err(PlaneError::Shape);
                }
                gp.copy_from_slice(grad_w);
            }
            Some(rows) => {
                if rows.len() != self.o {
                    return Err(PlaneError::Shape);
                }
                for (i, &rr) in rows.iter().enumerate() {
                    if rr >= o_total {
                        return Err(PlaneError::RowOutOfRange);
                    }
                    gp[i * self.n..(i + 1) * self.n]
                        .copy_from_slice(&grad_w[rr * self.n..(rr + 1) * self.n]);
                }
            }
        }
        let (r_l, c_l, d_l) = self.latents();
        let r_q = snap_r(&r_l, &self.lo, &self.sp, self.o, self.n);
        let c_q = quant_c_q32(&c_l, 6);
        let d_q = quant_c_q32(&d_l, 5);
        if r_q.len() != og || c_q.len() != self.n || d_q.len() != self.n {
            return Err(PlaneError::Range);
        }
        let s = self.code_field();
        let x = fwht(&scale_by_cols(&s, &d_q, self.o, self.n));
        if s.len() != self.o * self.n || x.len() != self.o * self.n {
            return Err(PlaneError::Range);
        }
        let gy = pointwise(&gp, &x);
        let dr = row_group_dots(&gy, &c_q, self.o, self.n, g)?;
        let dc = dot_cols(&gy, &r_q, self.o, self.n, g, GROUP)?;
        let dy = fwht(&rows_pass(&gp, &r_q, &c_q, self.o, self.n));
        if dy.len() != self.o * self.n {
            return Err(PlaneError::Range);
        }
        let dd = dot_cols(&s, &dy, self.o, self.n, self.n, 1)?;
        Ok((
            pointwise(&dr, &r_l),
            pointwise(&dc, &c_l),
            pointwise(&dd, &d_l),
        ))
    }

    /// Sum the plane weight `plane_w` into `w` (`o_total × n`): every row when `rows` is `None`,
    /// else the rows in `rows`, a repeated destination summing in index order.
    ///
    /// # Errors
    /// [`PlaneError::Shape`] on a shape mismatch; [`PlaneError::RowOutOfRange`] for a `rows` entry
    /// past `o_total`; [`PlaneError::Range`] if a checked add leaves `i64`. Serial `O(o·n)`.
    pub fn add_into(
        &self,
        w: &mut [i64],
        plane_w: &[i64],
        o_total: usize,
    ) -> Result<(), PlaneError> {
        if self.o == 0 || self.n == 0 {
            return Err(PlaneError::Shape);
        }
        if plane_w.len() != self.o.checked_mul(self.n).ok_or(PlaneError::Shape)?
            || w.len() != o_total.checked_mul(self.n).ok_or(PlaneError::Shape)?
        {
            return Err(PlaneError::Shape);
        }
        match &self.rows {
            None if o_total != self.o => return Err(PlaneError::Shape),
            Some(rows) if rows.len() != self.o => return Err(PlaneError::Shape),
            Some(rows) if rows.iter().any(|&r| r >= o_total) => {
                return Err(PlaneError::RowOutOfRange)
            }
            _ => {}
        }
        for i in 0..self.o {
            let d = match &self.rows {
                None => i,
                Some(rows) => rows[i],
            };
            let src = &plane_w[i * self.n..(i + 1) * self.n];
            let row = &mut w[d * self.n..(d + 1) * self.n];
            for (a, b) in row.iter_mut().zip(src) {
                *a = a.checked_add(*b).ok_or(PlaneError::Range)?;
            }
        }
        Ok(())
    }
}

/// One whole-tensor word pass: the module's pooled forward and backward work.
enum Op<'a> {
    /// `out[i] = x[i]·v[i mod n]` (a whole-tensor column scale).
    Cols { x: &'a [i64], v: &'a [i64] },
    /// `out[i] = x[i]·r[oi·g + j/GROUP]·c[j]` (the compose scale of an `o × n` tensor).
    Rows {
        x: &'a [i64],
        r: &'a [i64],
        c: &'a [i64],
        n: usize,
        g: usize,
    },
    /// The normalised 128-point Walsh–Hadamard transform of each 128-chunk, in place.
    Fwht,
}

/// [`Op`] over one contiguous element range; the first refused word is recorded.
struct Pass<'a> {
    op: Op<'a>,
    bad: &'a FirstErr<usize, ()>,
}

impl ChunkTask<i64> for Pass<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [i64]) {
        let fail = |i: usize| self.bad.record(i, ());
        match &self.op {
            Op::Cols { x, v } => {
                for (k, d) in piece.iter_mut().enumerate() {
                    let i = start + k;
                    match mul(x[i], v[i % v.len()]) {
                        Some(w) => *d = w,
                        None => return fail(i),
                    }
                }
            }
            Op::Rows { x, r, c, n, g } => {
                for (k, d) in piece.iter_mut().enumerate() {
                    let i = start + k;
                    match mul(x[i], r[(i / n) * g + (i % n) / GROUP]).and_then(|a| mul(a, c[i % n]))
                    {
                        Some(w) => *d = w,
                        None => return fail(i),
                    }
                }
            }
            Op::Fwht => {
                for (bi, chunk) in piece.chunks_mut(GROUP).enumerate() {
                    let mut h = 1;
                    while h < GROUP {
                        let mut i = 0;
                        while i < GROUP {
                            for j in i..(i + h) {
                                let (a, b) = (chunk[j], chunk[j + h]);
                                match (a.checked_add(b), a.checked_sub(b)) {
                                    (Some(s), Some(d)) => (chunk[j], chunk[j + h]) = (s, d),
                                    _ => return fail(start + bi * GROUP + j),
                                }
                            }
                            i += 2 * h;
                        }
                        h *= 2;
                    }
                    for (k, v) in chunk.iter_mut().enumerate() {
                        match mul(*v, INV_SQRT128_Q32) {
                            Some(s) => *v = s,
                            None => return fail(start + bi * GROUP + k),
                        }
                    }
                }
            }
        }
    }
}

/// Run `op` over `out`, one contiguous piece per participant; `false` on a refused word.
fn apply(out: &mut [i64], chunk: usize, op: Op<'_>) -> bool {
    let bad: FirstErr<usize, ()> = FirstErr::new();
    chunks_mut_global(out, chunk.max(1), &Pass { op, bad: &bad });
    bad.take().is_none()
}

/// `x[i]·v[i mod n]` over an `o × n` buffer; empty on a refusal. Temporal `O(o·n)`: `≥ 10^6` at a
/// fitted shape, so pooled.
fn scale_by_cols(x: &[i64], v: &[i64], o: usize, n: usize) -> Vec<i64> {
    if n == 0 || v.len() != n || x.len() != o * n {
        return Vec::new();
    }
    let mut t = vec![0_i64; x.len()];
    let chunk = o.div_ceil(width_global()).max(1) * n;
    if !apply(&mut t, chunk, Op::Cols { x, v }) {
        return Vec::new();
    }
    t
}

/// `a[i]·b[i]` over equal-length buffers; empty on a refusal, pooled `O(len)`.
fn pointwise(a: &[i64], b: &[i64]) -> Vec<i64> {
    if a.len() != b.len() || a.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0_i64; a.len()];
    let chunk = a.len().div_ceil(width_global()).max(1);
    if !apply(&mut out, chunk, Op::Cols { x: a, v: b }) {
        return Vec::new();
    }
    out
}

/// [`Op::Rows`] over an `o × n` buffer; empty on a length or word refusal. `O(o·n)` on the pool.
fn rows_pass(x: &[i64], r: &[i64], c: &[i64], o: usize, n: usize) -> Vec<i64> {
    let g = n / GROUP;
    if g == 0 || x.len() != o * n || r.len() != o * g || c.len() != n {
        return Vec::new();
    }
    let mut w = vec![0_i64; o * n];
    let chunk = o.div_ceil(width_global()).max(1) * n;
    if !apply(&mut w, chunk, Op::Rows { x, r, c, n, g }) {
        return Vec::new();
    }
    w
}

/// The normalised 128-point Walsh–Hadamard transform `H₁₂₈/√128` over every contiguous 128-chunk;
/// the butterfly is exact `i64` addition, the scale multiplies with one rounding. Temporal
/// `O(o·n·log2(GROUP))` (seven butterfly stages per chunk), pooled.
fn fwht(x: &[i64]) -> Vec<i64> {
    if x.is_empty() || !x.len().is_multiple_of(GROUP) {
        return Vec::new();
    }
    let mut y = x.to_vec();
    let blocks = y.len() / GROUP;
    let chunk = blocks.div_ceil(width_global()).max(1) * GROUP;
    if !apply(&mut y, chunk, Op::Fwht) {
        return Vec::new();
    }
    y
}

/// `compose(fwht128(s ⊙ d), r, c)` over an `o × n` field; empty on a shape refusal. Temporal
/// `O(o·n·log2(GROUP))`, the passes pooled.
fn compose_plane(s: &[i64], r: &[i64], c: &[i64], d: &[i64], o: usize, n: usize) -> Vec<i64> {
    let g = n / GROUP;
    if g == 0 || s.len() != o * n || r.len() != o * g || c.len() != n {
        return Vec::new();
    }
    let y = fwht(&scale_by_cols(s, d, o, n));
    if y.len() != o * n {
        return Vec::new();
    }
    rows_pass(&y, r, c, o, n)
}

/// `dr[o,gg] = round(Σ_{j∈gg} gy[o,j]·c_q[j] / 2^32)`, one `i128` dot per `(row, group)` rounded
/// once, serial `O(o·n)`.
///
/// # Errors
/// [`PlaneError::Shape`] on a length disagreement or `g == 0`; [`PlaneError::Range`] on a word.
fn row_group_dots(
    gy: &[i64],
    c_q: &[i64],
    o: usize,
    n: usize,
    g: usize,
) -> Result<Vec<i64>, PlaneError> {
    if g == 0 || c_q.len() != n || gy.len() != o * n {
        return Err(PlaneError::Shape);
    }
    let mut dr = vec![0_i64; o * g];
    for (e, d) in dr.iter_mut().enumerate() {
        let (row, gg) = (e / g, e % g);
        let mut acc = 0_i128;
        for j in 0..GROUP {
            acc += i128::from(gy[row * n + gg * GROUP + j]) * i128::from(c_q[gg * GROUP + j]);
        }
        *d = round_q32_even(acc).map_err(|_| PlaneError::Range)?;
    }
    Ok(dr)
}

/// The column reduction `out[j] = round(Σ_o a[o,j]·b[o, j/per] / 2^32)`, each column accumulated in
/// `i128` and rounded once: the two column-scale gradients; serial `O(o·n)`.
///
/// # Errors
/// [`PlaneError::Shape`] for `a.len() != o·n` or `b.len() != o·g`; [`PlaneError::Range`] on a word.
fn dot_cols(
    a: &[i64],
    b: &[i64],
    o: usize,
    n: usize,
    g: usize,
    per: usize,
) -> Result<Vec<i64>, PlaneError> {
    if g == 0 || a.len() != o * n || b.len() != o * g {
        return Err(PlaneError::Shape);
    }
    let mut out = vec![0_i64; n];
    for (j, d) in out.iter_mut().enumerate() {
        let mut acc = 0_i128;
        for oi in 0..o {
            acc += i128::from(a[oi * n + j]) * i128::from(b[oi * g + j / per]);
        }
        *d = round_q32_even(acc).map_err(|_| PlaneError::Range)?;
    }
    Ok(out)
}

/// The `r` snap of [`CodedPlane::qvals`]: [`crate::learn::fold::fit::quant::r_level`]'s own level for
/// the code `k = round(clamp((r_l − lo)/sp·15, 0, 15))`, on the same grid endpoints. Empty on a
/// shape refusal; serial `O(o·n/128)`.
fn snap_r(r_l: &[i64], lo: &[i64], sp: &[i64], o: usize, n: usize) -> Vec<i64> {
    let g = n / GROUP;
    if g == 0 || r_l.len() != o * g || lo.len() != o || sp.len() != o {
        return Vec::new();
    }
    let level = |i: usize| -> Option<i64> {
        let oi = i / g;
        let x = i128::from(r_l[i].checked_sub(lo[oi])?);
        let k = div_ne(x * 15, i128::from(sp[oi]))?.clamp(0, 15);
        lo[oi].checked_add(narrow(div_ne(k * i128::from(sp[oi]), 15)?)?)
    };
    let out: Option<Vec<i64>> = (0..o * g).map(level).collect();
    out.unwrap_or_default()
}

/// Affine column-scale quantization in Q32.32: `c_min`/`c_max` rounded to binary16, a span floored
/// at one grid unit, each element rounded to the nearest of `2^bits` grid points on `[c_min, c_max]`;
/// a fixed point of the map. Empty for an empty slice, `bits` outside `{5, 6}` or a refusal; serial
/// `O(n)`.
fn quant_c_q32(x: &[i64], bits: u32) -> Vec<i64> {
    if x.is_empty() || !(5..=6).contains(&bits) {
        return Vec::new();
    }
    let mn = x.iter().copied().min().unwrap_or(0);
    let mx = x.iter().copied().max().unwrap_or(0);
    let (Some(c_min), Some(c_max)) = (f16_q32(mn), f16_q32(mx)) else {
        return Vec::new();
    };
    let span = (i128::from(c_max) - i128::from(c_min)).max(1);
    let levels: i128 = (1_i128 << bits) - 1;
    let round = |v: i64| -> Option<i64> {
        let t = div_ne((i128::from(v) - i128::from(c_min)) << 32, span)?;
        let code = rshift_rne(t.clamp(0, i128::from(ONE)) * levels, 32)?;
        c_min.checked_add(narrow(div_ne(code * span, levels)?)?)
    };
    let out: Option<Vec<i64>> = x.iter().copied().map(round).collect();
    out.unwrap_or_default()
}

/// The IEEE-754 binary16 round-to-nearest-even of a Q32.32 word, as a Q32.32 word. `O(1)`.
fn f16_q32(v: i64) -> Option<i64> {
    let f = round_f16(Fx { m: v, e: -32 }).ok()?;
    if f.m == 0 {
        return Some(0);
    }
    let shift = f.e + 32;
    let wide = if shift >= 0 {
        i128::from(f.m) << u32::try_from(shift).ok().filter(|&s| s <= 96)?
    } else {
        rshift_rne(
            i128::from(f.m),
            u32::try_from(-shift).ok().filter(|&s| s <= 126)?,
        )?
    };
    narrow(wide)
}

/// Nearest codebook row per 8-column block of an `o × n` Q32.32 field: for block `b` the index
/// `argmax_j (2·⟨b, C_j⟩ − |C_j|²)`, lowest index on an exact tie (torch `argmax` semantics); the
/// second value is the exactness flag.
///
/// # Errors
/// [`PlaneError::Shape`] for `o == 0`, `n` not a multiple of [`BLOCK`], or a length disagreement;
/// serial `O(o·n·CODE_LEN)` multiply-adds, construction-time only.
pub fn encode(s: &[i64], o: usize, n: usize) -> Result<(Vec<u8>, bool), PlaneError> {
    let on = o.checked_mul(n).ok_or(PlaneError::Shape)?;
    if o == 0 || n == 0 || !n.is_multiple_of(BLOCK) || s.len() != on {
        return Err(PlaneError::Shape);
    }
    let cb = codebook_q32();
    // `|C_j|²` is a sum of eight squared Q32.32 words, up to `8·(2.83·2^32)² ≈ 2^69`; it stays
    // `i128` because the score is compared, never stored as a word.
    let mut cn = [0_i128; CODE_LEN];
    for (j, slot) in cn.iter_mut().enumerate() {
        *slot = (0..BLOCK)
            .map(|k| i128::from(cb[j][k]) * i128::from(cb[j][k]))
            .sum();
    }
    let nb = n / BLOCK;
    let mut idx = vec![0_u8; o * nb];
    let mut exact = true;
    for (b, slot) in idx.iter_mut().enumerate() {
        let b0 = (b / nb) * n + (b % nb) * BLOCK;
        let (mut best, mut best_score) = (0_usize, i128::MIN);
        for j in 0..CODE_LEN {
            let dot: i128 = (0..BLOCK)
                .map(|t| i128::from(s[b0 + t]) * i128::from(cb[j][t]))
                .sum();
            let score = 2 * dot - cn[j];
            if score > best_score {
                best_score = score;
                best = j;
            }
        }
        *slot = best as u8;
        exact &= cb[best][..] == s[b0..b0 + BLOCK];
    }
    Ok((idx, exact))
}

/// `base[i] · exp(off[i])` over `min` of the two lengths; empty on an `exp` or word refusal.
/// `O(len)` serial multiplies and `exp` calls (one entry per parameter); spatial `O(len)`.
fn scale_exp(base: &[i64], off: &[i64]) -> Vec<i64> {
    let len = base.len().min(off.len());
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        match fixed_exp(off[i]).ok().and_then(|e| mul(base[i], e)) {
            Some(v) => out.push(v),
            None => return Vec::new(),
        }
    }
    out
}

/// The codebook as Q32.32 words: [`codeword`] carries them already; `O(CODE_LEN·BLOCK)` copies.
fn codebook_q32() -> [[i64; BLOCK]; CODE_LEN] {
    let mut out = [[0_i64; BLOCK]; CODE_LEN];
    for (j, row) in out.iter_mut().enumerate() {
        if let Ok(words) = codeword(j) {
            *row = words;
        }
    }
    out
}
