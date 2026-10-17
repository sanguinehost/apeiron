//! The finishing stage's coded weight plane: the frozen `E8 ∪ O8` code field `S`, a frozen per-row
//! scale grid `(lo, sp)` and the trainable log-multiplicative latents `(pa, pb, pd)`.
//!
//! Port of the production configuration of `spec/pv_finish.py:205-285` (the `Plane` class and
//! `plane_weight`) and `add_plane` (331-345), over the refit primitives of
//! [`crate::learn::fold::fit::steps`] (`compose`, `fwht128`) and the scale quantizers of
//! [`crate::learn::fold::fit::quant`] and the `E8 ∪ O8` codebook of [`crate::learn::fold::fit::icm`].
//! The one-bit arm stores one plane of a linear as
//! `weight = compose(fwht128(S ⊙ d), r, c)`, where `S` is a field of 8-column blocks, each a row of
//! the 256-row codebook.
//!
//! # What is ported
//!
//! Only the production configuration: the code field is **frozen** (no V-step, no straight-through
//! code latents), the rotation is `fwht128` (not the learned `kan` reflection register), and the
//! trainable subset is `r`, `c`, `d`. The `kan` rotation, the dense-corrector `corr` factors and the
//! `norm` zero-bit tunables, the `regrid` re-fit, the `obs` Optimal-Brain-Surgeon gain and the
//! `block-s` AQLM warm-up are out of scope and absent here.
//!
//! # Straight-through gradient
//!
//! [`CodedPlane::qvals`] returns the *forward* values of the three arm quantizers: `r` snapped to its
//! frozen row grid `lo + round(clamp((r_l − lo)/sp·15, 0, 15))/15·sp`, and `c`/`d` through
//! [`quant_c_u8`] at 6 and 5 bits. The quantizer Jacobian is the identity (straight-through), so the
//! gradient the finishing step needs is the gradient of the **unquantized** forward
//! `compose(fwht128(S ⊙ d_l), r_l, c_l)` on the latents `r_l = r0·exp(pa)`, `c_l = c0·exp(pb)`,
//! `d_l = d0·exp(pd)`, which is what [`CodedPlane::backward`] returns and what
//! [`CodedPlane::weight_unquantized`] evaluates. The two agree by construction whenever the
//! quantizers are the identity at the evaluation point; `tests/learn_fold_plane.rs` checks the
//! identity by central differences on a plane whose latents are exact quantizer fixed points.
//!
//! # Partiality
//!
//! [`CodedPlane::from_pack`] and [`CodedPlane::set_offsets`] validate every shape and refuse a
//! non-finite value with a [`PlaneError`]; `n` must be a positive multiple of [`GROUP`] (128).
//! [`CodedPlane::backward`] and [`CodedPlane::add_into`] refuse a shape or an out-of-range output row.
//! No method here panics on caller input.
//!
//! # Deviations from the Python
//!
//! - The forward runs in `f64` on the CPU, so the `encode` argmax and the `float32` tensor arithmetic
//!   of `pv_finish.py` can differ in a near-tie; the tie rule (lowest index) is preserved.
//! - `from_pack` returns a `Result` and refuses ragged slices and non-finite values, where the Python
//!   broadcasts or raises inside torch.
//!
//! # Grade
//!
//! `tested` (`tests/learn_fold_plane.rs`); nothing here is extracted or refined to the canonical
//! corpus, so no statement here is `proved`.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::must_use_candidate,
    clippy::suboptimal_flops,
    clippy::float_cmp,
    clippy::too_many_arguments,
    clippy::doc_markdown,
    reason = "the flat-slice kernels index row-major `[o, n]` buffers by `o/i/j/k/m/gg` and name the \
              carriers `r0`/`c0`/`d0`, `pa`/`pb`/`pd` and `lo`/`sp` as the reference does; the `u8` \
              code index cast is exact (`best < 256`); `encode`'s argmax and the exactness flag are \
              the reference's exact `>`/`==` comparisons against the codebook."
)]

use crate::learn::fold::fit::icm::{codebook, BLOCK, CODE_LEN};
use crate::learn::fold::fit::quant::{quant_c_u8, r_level};
use crate::learn::fold::fit::steps::{compose, fwht128, GROUP};

/// A refusal from a [`CodedPlane`] constructor or method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneError {
    /// A slice length disagrees with the declared `o`/`n` shape, a `rows` list does not carry `o`
    /// entries, or `n` is not a positive multiple of [`GROUP`].
    Shape,
    /// An output row index in `rows` is not below the receiving weight's row count.
    RowOutOfRange,
    /// A base value, a grid endpoint or an offset is not finite (`NaN` or `±∞`).
    NotFinite,
}

/// The latent offset gradients `(∂L/∂pa, ∂L/∂pb, ∂L/∂pd)` of [`CodedPlane::backward`].
pub type LatentGrad = (Vec<f64>, Vec<f64>, Vec<f64>);

/// One stored plane: a frozen code field (`idx` into the codebook, or `s_fix` when some block is not
/// a codebook row), an optional output-row restriction `rows`, the frozen per-row scale grid
/// `(lo, sp)`, the base values `(r0, c0, d0)` and the log-multiplicative latent offsets
/// `(pa, pb, pd)`.
///
/// All latents start at `0`, so `latents()` reproduces the base values at construction. The fields
/// are public state; construct through [`CodedPlane::from_pack`] and set offsets through
/// [`CodedPlane::set_offsets`] so the shape and finiteness invariants hold.
#[derive(Clone, Debug)]
pub struct CodedPlane {
    /// Out rows of this plane (`=|rows|` when `rows` is set).
    pub o: usize,
    /// In columns; a positive multiple of [`GROUP`].
    pub n: usize,
    /// Codebook row per 8-column block, row-major `[o, n / 8]`; empty when `s_fix` is set.
    pub idx: Vec<u8>,
    /// The exact code field `[o, n]` when some block is not a codebook row; `None` when `idx` is used.
    pub s_fix: Option<Vec<f64>>,
    /// Output rows of the receiving weight this plane writes, or `None` for every row.
    pub rows: Option<Vec<usize>>,
    /// Per-row lower grid endpoint `lo` of the frozen `r` grid, length `o`.
    pub lo: Vec<f64>,
    /// Per-row grid width `sp` of the frozen `r` grid, length `o`.
    pub sp: Vec<f64>,
    /// Base group scale `r0`, row-major `[o, n / GROUP]`.
    pub r0: Vec<f64>,
    /// Base column scale `c0`, length `n`.
    pub c0: Vec<f64>,
    /// Base Hadamard column scale `d0`, length `n`.
    pub d0: Vec<f64>,
    /// Group-scale log offset `pa`, row-major `[o, n / GROUP]`.
    pub pa: Vec<f64>,
    /// Column-scale log offset `pb`, length `n`.
    pub pb: Vec<f64>,
    /// Hadamard column-scale log offset `pd`, length `n`.
    pub pd: Vec<f64>,
}

impl CodedPlane {
    /// Build a plane from its stored pack: the rotated code field `s_rot` (`o × n`), the frozen
    /// `r` grid `(lo, sp)` (`o` each), the scales `r` (`o × n/128`), `c` (`n`) and `d` (`n`), and an
    /// optional output-row restriction `rows` (`o` entries). The code field is encoded to
    /// [`CodedPlane::idx`] when every block equals its nearest codebook row, and kept verbatim in
    /// [`CodedPlane::s_fix`] otherwise.
    ///
    /// # Errors
    ///
    /// [`PlaneError::Shape`] if `o == 0`, `n` is not a positive multiple of [`GROUP`], a slice length
    /// disagrees with the declared shape, or `rows` is present with a length other than `o`;
    /// [`PlaneError::NotFinite`] if any base value or grid endpoint is not finite.
    ///
    /// # Complexity
    ///
    /// Temporal: `O(o·n)` multiply-adds (the `o·(n/8)` blocks times the 256 codebook rows of
    /// [`encode`], constant in the caller's dimensions) plus the finiteness scan. Spatial: `O(o·n)`
    /// `f64` words for `s_fix` when the field is not exact, `O(o·n/8)` `u8` otherwise.
    pub fn from_pack(
        s_rot: &[f64],
        lo: &[f64],
        sp: &[f64],
        r: &[f64],
        c: &[f64],
        d: &[f64],
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
        if s_rot.len() != on
            || lo.len() != o
            || sp.len() != o
            || r.len() != og
            || c.len() != n
            || d.len() != n
        {
            return Err(PlaneError::Shape);
        }
        if let Some(rw) = rows {
            if rw.len() != o {
                return Err(PlaneError::Shape);
            }
        }
        for v in s_rot.iter().chain(lo).chain(sp).chain(r).chain(c).chain(d) {
            if !v.is_finite() {
                return Err(PlaneError::NotFinite);
            }
        }
        let (idx_all, exact) = encode(s_rot, o, n)?;
        let (idx, s_fix) = if exact {
            (idx_all, None)
        } else {
            (Vec::new(), Some(s_rot.to_vec()))
        };
        Ok(Self {
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
            pa: vec![0.0_f64; og],
            pb: vec![0.0_f64; n],
            pd: vec![0.0_f64; n],
        })
    }

    /// Overwrite the log offsets `(pa, pb, pd)` after validating their shapes and finiteness.
    ///
    /// # Errors
    ///
    /// [`PlaneError::Shape`] if the plane's own shape is degenerate or a slice length disagrees;
    /// [`PlaneError::NotFinite`] if any offset is not finite.
    ///
    /// # Complexity
    ///
    /// Temporal: `O(o·n/128 + n)` compares and copies. Spatial: `O(1)` (in place).
    pub fn set_offsets(&mut self, pa: &[f64], pb: &[f64], pd: &[f64]) -> Result<(), PlaneError> {
        if self.o == 0 || self.n == 0 || !self.n.is_multiple_of(GROUP) {
            return Err(PlaneError::Shape);
        }
        let g = self.n / GROUP;
        if pa.len() != self.o * g || pb.len() != self.n || pd.len() != self.n {
            return Err(PlaneError::Shape);
        }
        for v in pa.iter().chain(pb).chain(pd) {
            if !v.is_finite() {
                return Err(PlaneError::NotFinite);
            }
        }
        self.pa.copy_from_slice(pa);
        self.pb.copy_from_slice(pb);
        self.pd.copy_from_slice(pd);
        Ok(())
    }

    /// The code field `S` (`o × n`, row-major): `s_fix` verbatim, else the codebook rows named by
    /// `idx`.
    ///
    /// # Complexity
    ///
    /// Temporal: `O(o·n)` words copied. Spatial: `O(o·n)` `f64` words.
    pub fn code_field(&self) -> Vec<f64> {
        if let Some(s) = &self.s_fix {
            return s.clone();
        }
        let cb = codebook();
        let nb = self.n / BLOCK;
        let mut s = vec![0.0_f64; self.o * self.n];
        for oi in 0..self.o {
            for m in 0..nb {
                let j = usize::from(self.idx.get(oi * nb + m).copied().unwrap_or(0));
                let row = &cb[j];
                for k in 0..BLOCK {
                    s[oi * self.n + m * BLOCK + k] = row[k];
                }
            }
        }
        s
    }

    /// The latents `(r_l, c_l, d_l) = (r0·exp(pa), c0·exp(pb), d0·exp(pd))`.
    ///
    /// # Complexity
    ///
    /// Temporal: `O(o·n/128 + n)` multiplies and `exp` calls. Spatial: `O(o·n/128 + n)` `f64` words.
    pub fn latents(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            scale_exp(&self.r0, &self.pa),
            scale_exp(&self.c0, &self.pb),
            scale_exp(&self.d0, &self.pd),
        )
    }

    /// The quantized forward values `(r_q, c_q, d_q)`: `r` snapped to its frozen row grid, `c`
    /// through [`quant_c_u8`] at 6 bits and `d` through [`quant_c_u8`] at 5 bits.
    ///
    /// # Complexity
    ///
    /// Temporal: `O(o·n/128 + n)` (the two affine quantizers are one pass each). Spatial:
    /// `O(o·n/128 + n)` `f64` words.
    pub fn qvals(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (r_l, c_l, d_l) = self.latents();
        (
            snap_r(&r_l, &self.lo, &self.sp, self.o, self.n),
            quant_affine(&c_l, 6),
            quant_affine(&d_l, 5),
        )
    }

    /// The plane's weight `compose(fwht128(S ⊙ d_q), r_q, c_q)`, row-major `o × n`. A zero-length
    /// result signals an internal shape-invariant violation, which [`CodedPlane::from_pack`]
    /// and [`CodedPlane::set_offsets`] make unreachable.
    ///
    /// # Complexity
    ///
    /// Temporal: `O(o·n·log2(GROUP))` additions for the Walsh–Hadamard transform plus `O(o·n)`
    /// multiplies. Spatial: `O(o·n)` `f64` words.
    pub fn weight(&self) -> Vec<f64> {
        let s = self.code_field();
        let (r_q, c_q, d_q) = self.qvals();
        compose_plane(&s, &r_q, &c_q, &d_q, self.o, self.n)
    }

    /// The unquantized forward `compose(fwht128(S ⊙ d_l), r_l, c_l)` on the latents; the
    /// differentiable forward whose gradient (on the latents) is the straight-through gradient
    /// [`CodedPlane::backward`] returns. A zero-length result as in [`CodedPlane::weight`].
    ///
    /// # Complexity
    ///
    /// Temporal and spatial as [`CodedPlane::weight`].
    pub fn weight_unquantized(&self) -> Vec<f64> {
        let s = self.code_field();
        let (r_l, c_l, d_l) = self.latents();
        compose_plane(&s, &r_l, &c_l, &d_l, self.o, self.n)
    }

    /// The straight-through gradient of `0.5·Σ (W ⊙ M)²`-style losses: given `grad_w = ∂L/∂W`
    /// (`o × n`, or `o_total × n` when `rows` is set, of which only the plane's rows enter), return
    /// `(∂L/∂pa, ∂L/∂pb, ∂L/∂pd)`.
    ///
    /// With `H = fwht128`, `X = H(S ⊙ d_q)` and `G` the plane's gradient rows:
    /// `dr[o,g] = Σ_{j∈g} G·c_q·X`, `dc[j] = Σ_o G·r_q·X`, `dY = H(G ⊙ r_q ⊙ c_q)`,
    /// `dd[j] = Σ_o S[o,j]·dY[o,j]`; then `∂pa = dr·r_l`, `∂pb = dc·c_l`, `∂pd = dd·d_l`.
    ///
    /// # Errors
    ///
    /// [`PlaneError::Shape`] if `grad_w` is not a whole number of `n`-long rows or disagrees with the
    /// plane shape; [`PlaneError::RowOutOfRange`] if an entry of `rows` is at least the row count
    /// implied by `grad_w`.
    ///
    /// # Complexity
    ///
    /// Temporal: `O(o·n·log2(GROUP))` (`X` and the `dY` transform) plus `O(o·n)` multiply-adds and
    /// three reductions. Spatial: `O(o·n)` `f64` words (`X`, `dY` and the gathered gradient rows).
    pub fn backward(&self, grad_w: &[f64]) -> Result<LatentGrad, PlaneError> {
        if self.o == 0 || self.n == 0 || !self.n.is_multiple_of(GROUP) {
            return Err(PlaneError::Shape);
        }
        let g = self.n / GROUP;
        let og = self.o * g;
        if !grad_w.len().is_multiple_of(self.n) {
            return Err(PlaneError::Shape);
        }
        let o_total = grad_w.len() / self.n;
        let mut gp = vec![0.0_f64; self.o * self.n];
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
                    for j in 0..self.n {
                        gp[i * self.n + j] = grad_w[rr * self.n + j];
                    }
                }
            }
        }
        let (r_l, c_l, d_l) = self.latents();
        let r_q = snap_r(&r_l, &self.lo, &self.sp, self.o, self.n);
        let c_q = quant_affine(&c_l, 6);
        let d_q = quant_affine(&d_l, 5);
        if r_q.len() != og || c_q.len() != self.n || d_q.len() != self.n {
            return Err(PlaneError::Shape);
        }
        let s = self.code_field();
        let Ok(x) = fwht128(&scale_by_cols(&s, &d_q, self.o, self.n)) else {
            return Err(PlaneError::Shape);
        };
        let mut gy = vec![0.0_f64; self.o * self.n];
        for k in 0..self.o * self.n {
            gy[k] = gp[k] * x[k];
        }
        let mut dr = vec![0.0_f64; og];
        for oi in 0..self.o {
            for gg in 0..g {
                let mut acc = 0.0_f64;
                for j in gg * GROUP..(gg + 1) * GROUP {
                    acc += gy[oi * self.n + j] * c_q[j];
                }
                dr[oi * g + gg] = acc;
            }
        }
        let mut dc = vec![0.0_f64; self.n];
        for oi in 0..self.o {
            for j in 0..self.n {
                dc[j] += gy[oi * self.n + j] * r_q[oi * g + j / GROUP];
            }
        }
        let mut t = vec![0.0_f64; self.o * self.n];
        for oi in 0..self.o {
            for j in 0..self.n {
                t[oi * self.n + j] = gp[oi * self.n + j] * r_q[oi * g + j / GROUP] * c_q[j];
            }
        }
        let Ok(dy) = fwht128(&t) else {
            return Err(PlaneError::Shape);
        };
        let mut dd = vec![0.0_f64; self.n];
        for oi in 0..self.o {
            for j in 0..self.n {
                dd[j] += s[oi * self.n + j] * dy[oi * self.n + j];
            }
        }
        let mut dpa = vec![0.0_f64; og];
        for k in 0..og {
            dpa[k] = dr[k] * r_l[k];
        }
        let mut dpb = vec![0.0_f64; self.n];
        for j in 0..self.n {
            dpb[j] = dc[j] * c_l[j];
        }
        let mut dpd = vec![0.0_f64; self.n];
        for j in 0..self.n {
            dpd[j] = dd[j] * d_l[j];
        }
        Ok((dpa, dpb, dpd))
    }

    /// Sum the plane weight `plane_w` (`o × n`) into `w` (`o_total × n`): every row when `rows` is
    /// `None`, else the rows in `rows`. The port of `add_plane`
    /// (`spec/pv_finish.py:331-345`).
    ///
    /// # Errors
    ///
    /// [`PlaneError::Shape`] if the plane's own shape is degenerate, `plane_w` is not `o × n`, or `w`
    /// is not `o_total × n`; [`PlaneError::RowOutOfRange`] if an entry of `rows` is at least
    /// `o_total`.
    ///
    /// # Complexity
    ///
    /// Temporal: `O(o·n)` adds (rows restricted: `O(|rows|·n)`). Spatial: `O(1)` (in place).
    pub fn add_into(
        &self,
        w: &mut [f64],
        plane_w: &[f64],
        o_total: usize,
    ) -> Result<(), PlaneError> {
        if self.o == 0 || self.n == 0 {
            return Err(PlaneError::Shape);
        }
        let on = self.o.checked_mul(self.n).ok_or(PlaneError::Shape)?;
        let otn = o_total.checked_mul(self.n).ok_or(PlaneError::Shape)?;
        if plane_w.len() != on || w.len() != otn {
            return Err(PlaneError::Shape);
        }
        match &self.rows {
            None => {
                if o_total != self.o {
                    return Err(PlaneError::Shape);
                }
                for (dst, &src) in w.iter_mut().zip(plane_w) {
                    *dst += src;
                }
            }
            Some(rows) => {
                if rows.len() != self.o {
                    return Err(PlaneError::Shape);
                }
                for (i, &rr) in rows.iter().enumerate() {
                    if rr >= o_total {
                        return Err(PlaneError::RowOutOfRange);
                    }
                    for j in 0..self.n {
                        w[rr * self.n + j] += plane_w[i * self.n + j];
                    }
                }
            }
        }
        Ok(())
    }
}

/// Nearest codebook row per 8-column block of an `o × n` field `s`: for block `b` the index
/// `argmax_j (2·⟨b, C_j⟩ − |C_j|²)`, carrying the lowest index on an exact tie (torch `argmax`
/// semantics). The second value is the exactness flag: every block equals its chosen row bit for
/// bit.
///
/// # Errors
///
/// [`PlaneError::Shape`] if `o == 0`, `n` is not a positive multiple of [`BLOCK`], or `s` is not
/// `o·n`.
///
/// # Complexity
///
/// Temporal: `O(o·n)` multiply-adds — `o·(n/8)` blocks, each `CODE_LEN = 256` rows of [`BLOCK`] `= 8`
/// terms. Spatial: `O(o·n/8)` `u8` words for the indices.
pub fn encode(s: &[f64], o: usize, n: usize) -> Result<(Vec<u8>, bool), PlaneError> {
    if o == 0 || n == 0 || !n.is_multiple_of(BLOCK) {
        return Err(PlaneError::Shape);
    }
    if s.len() != o.checked_mul(n).ok_or(PlaneError::Shape)? {
        return Err(PlaneError::Shape);
    }
    let cb = codebook();
    let mut cn = [0.0_f64; CODE_LEN];
    for j in 0..CODE_LEN {
        let mut acc = 0.0_f64;
        for k in 0..BLOCK {
            acc += cb[j][k] * cb[j][k];
        }
        cn[j] = acc;
    }
    let nb = n / BLOCK;
    let mut idx = vec![0_u8; o * nb];
    let mut exact = true;
    for oi in 0..o {
        for m in 0..nb {
            let b0 = oi * n + m * BLOCK;
            let mut best = 0_usize;
            let mut best_score = f64::NEG_INFINITY;
            for j in 0..CODE_LEN {
                let mut dot = 0.0_f64;
                for k in 0..BLOCK {
                    dot += s[b0 + k] * cb[j][k];
                }
                let score = 2.0 * dot - cn[j];
                if score > best_score {
                    best_score = score;
                    best = j;
                }
            }
            idx[oi * nb + m] = best as u8;
            for k in 0..BLOCK {
                if s[b0 + k] != cb[best][k] {
                    exact = false;
                }
            }
        }
    }
    Ok((idx, exact))
}

/// `base[i] · exp(off[i])`; the length is `min` of the two (both are the validated shape at every
/// call site). `O(len)` time and space.
fn scale_exp(base: &[f64], off: &[f64]) -> Vec<f64> {
    base.iter().zip(off).map(|(&b, &p)| b * p.exp()).collect()
}

/// `col[i] *= v[i]` over the rows of an `o × n` buffer. `O(o·n)` time, `O(o·n)` space.
fn scale_by_cols(x: &[f64], v: &[f64], o: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0.0_f64; o * n];
    for row in 0..o {
        for col in 0..n {
            t[row * n + col] = x[row * n + col] * v[col];
        }
    }
    t
}

/// `compose(fwht128(s ⊙ d), r, c)` over an `o × n` field; a zero-length result signals a shape
/// refusal from a primitive, unreachable for validated shapes. `O(o·n·log2(GROUP))` time,
/// `O(o·n)` space.
fn compose_plane(s: &[f64], r: &[f64], c: &[f64], d: &[f64], o: usize, n: usize) -> Vec<f64> {
    let Ok(y) = fwht128(&scale_by_cols(s, d, o, n)) else {
        return Vec::new();
    };
    compose(&y, r, c, o, n).unwrap_or_default()
}

/// The `r` snap of [`CodedPlane::qvals`]: [`r_level`] `lo[o] + (k/15)·sp[o]` on the Q32.32 grid with
/// `k = round_ties_even(clamp((r_l − lo[o])/sp[o]·15, 0, 15))` per element of an `o × n/128` latent.
///
/// # Complexity
///
/// Temporal: `O(o·n/128)` rounds. Spatial: `O(o·n/128)` `f64` words.
fn snap_r(r_l: &[f64], lo: &[f64], sp: &[f64], o: usize, n: usize) -> Vec<f64> {
    let g = n / GROUP;
    let mut out = vec![0.0_f64; o * g];
    for oi in 0..o {
        for gg in 0..g {
            let x = ((r_l[oi * g + gg] - lo[oi]) / sp[oi] * 15.0).clamp(0.0, 15.0);
            let k = x.round_ties_even();
            out[oi * g + gg] =
                r_level(lo[oi], sp[oi], u8::try_from(k as i64).unwrap_or(0)).unwrap_or(f64::NAN);
        }
    }
    out
}

/// [`quant_c_u8`] at a validated `bits ∈ {5, 6}` over a non-empty slice, where the only refusals
/// (`Empty`, `BadBits`, `LengthOverflow`) are unreachable; a refusal yields an empty vector that a
/// later shape check catches. `O(len)` time and space.
fn quant_affine(x: &[f64], bits: u32) -> Vec<f64> {
    match quant_c_u8(x, bits) {
        Ok((v, _)) => v,
        Err(_) => Vec::new(),
    }
}
