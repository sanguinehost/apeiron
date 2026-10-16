//! `fold_fit` in its production configuration: the alternation of the discrete E8 `gptq_e8` step and
//! the exact continuous `c`/`r`/`d` steps plus the E8 `e8_icm_step` refinement, followed by the
//! fp16/u8 pack of the winning plane.
//!
//! Port of `fold_fit` in `tools/onebit/fold_transfer.py` (removed; at `2f15d6a9`) lines 839–977,
//! specialised to the call site the transfer actually runs (`fold_transfer.py` 1733–1736):
//! `refit=True`, `had=True`, `act_order=True`, `wscale=True`, `hier_scale=True`, `bscale=True`,
//! `e8_code=True`, `e8_icm=True`, `iters=1`, `sweeps=2`. The flags that are off there (`cgauge`,
//! `bgptq`, `hpre`) are absent, not stubbed, and `hier_scale` is reachable only through the
//! `bscale and had` arm, which is the arm taken. `iters > 1` (the `cs`/`Wc` requantisation at lines
//! 890–909) is outside the production path and refused with [`FoldFitError::Unsupported`].
//!
//! # The port
//!
//! 1. `Hl = metric(H, ridge)`: `H` with `ridge · mean(diag(H))` added to its diagonal (line 431).
//! 2. `gptq_e8(fwht128(Ws), had_gram(H), beam=e8_beam, damp, act_order=true, wscale=true)` gives the
//!    code plane `S_rot`, the group scale `r` and, through `S = fwht128(S_rot)`, the candidate
//!    `compose(S, r, ones)`; `J` of that candidate is `trace[0]`.
//! 3. One `refit0` candidate from `inner_iters` passes of `c_step → mean-|c| normalisation → r_step →
//!    d_step → 8-block sign parity fix → 128-group `d` normalisation → e8_icm_step → fwht128(S_rot·d)
//!    → r_step`, whose `J` is `trace[1]`.
//! 4. `best = argmin J` over the two candidates (first on a tie). The `bscale and had` arm quantises
//!    the winning `d` to 5 bits, refits `r` and `c` against the re-materialised `S_dh`, tries the
//!    `e8_icm_step` re-quantisation candidate and the un-`d` fallback candidate, keeps the smallest
//!    `J`, and returns the pack `(S_rot, rh, ch, dh)` whose recomposition must equal `Wp` bit-for-bit.
//!
//! # Deviations from the Python
//!
//! - The Python's `assert torch.equal(...)` on the winning pack is a typed
//!   [`FoldFitError::PackMismatch`], never a panic.
//! - Every refit step's shape refusal, the codebook refusal, the quantiser refusal and the
//!   non-positive-definite refusal are carried as typed [`FoldFitError`] arms.
//! - The `torch` `float64`/`float32` branch on the step ridge (`eps = 1e-6 if float32`) is the
//!   `f64` arm (`1e-12`), as in the ported [`crate::learn::fold::fit::steps`] and `icm` bodies.
//!
//! # Partiality and trust boundary
//!
//! The body runs in `f64` on the crate's build path and is `tested`
//! (`tests/learn_fold_fold_fit.rs`); it is not extracted and no refinement theorem links it to the
//! canonical object, so nothing here is `proved`. It makes no hardware or performance claim.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::float_cmp,
    clippy::must_use_candidate,
    reason = "the body indexes row-major `[o, n]` and `[o, g]` buffers by `r/o/i/j/k/m/g` and names \
              the carriers `c`/`r`/`d`/`s_rot`/`dh`/`rh`/`ch` as the reference does; the `usize` to \
              `f64` casts are the shape means the reference forms (`n`, `GROUP`); the recomposition \
              check and the candidate tie are the reference's exact comparisons (`torch.equal`, \
              `min`); the single long body is the ported `fold_fit`"
)]

use crate::learn::fold::fit::gptq_e8::{gptq_e8, GptqE8Error};
use crate::learn::fold::fit::icm::{e8_icm_step, IcmError, PlaneRef};
use crate::learn::fold::fit::quant::{quant_c_u8, quant_r_hier, QuantError};
use crate::learn::fold::fit::steps::{
    c_step, compose, d_step, fwht128, had_gram, objective, r_step, StepsError,
};

/// Columns per `r` group scale: the Python `GROUP`.
pub const GROUP: usize = 128;

/// Columns per `E8 ∪ O8` code: the Python block width.
pub const BLOCK: usize = 8;

/// The `f64` normal-equation ridge the refit steps add (`eps` in the script).
const EPS: f64 = 1e-12;

/// A refusal from [`fold_fit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldFitError {
    /// A slice length disagrees with the declared `o × n` / `n × n` shape, or an inner step refused
    /// the shape.
    ShapeMismatch,
    /// `o == 0` or `n == 0`.
    ZeroDim,
    /// `n` is not a positive multiple of [`GROUP`].
    Width,
    /// `iters != 1`: the `cs`/`Wc` requantisation arm is out of the ported domain.
    Unsupported,
    /// A weight, Hessian or row weight is NaN or infinite.
    NonFinite,
    /// A reduced normal system or the damped Hessian is not positive-definite.
    NotPositiveDefinite,
    /// A checked integer quantity left its range in a codebook or bit-budget helper.
    Overflow,
    /// A code index does not decode: a codebook refusal.
    Codebook,
    /// A quantisation carrier refused its bit width.
    BadBits,
    /// The winning pack does not recompose to `Wp` bit-for-bit.
    PackMismatch,
}

impl core::fmt::Display for FoldFitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShapeMismatch => write!(f, "matrix shape or buffer length mismatch"),
            Self::ZeroDim => write!(f, "output or column count must be non-zero"),
            Self::Width => write!(f, "column count is not a positive multiple of 128"),
            Self::Unsupported => write!(f, "only iters = 1 is ported"),
            Self::NonFinite => write!(f, "weight or Hessian entry is not finite"),
            Self::NotPositiveDefinite => write!(f, "a reduced system is not positive-definite"),
            Self::Overflow => write!(f, "a checked integer quantity left its range"),
            Self::Codebook => write!(f, "codebook search left its admitted range"),
            Self::BadBits => write!(f, "quantisation bit width is out of range"),
            Self::PackMismatch => write!(f, "the winning pack does not recompose to Wp"),
        }
    }
}

impl std::error::Error for FoldFitError {}

impl From<StepsError> for FoldFitError {
    fn from(e: StepsError) -> Self {
        match e {
            StepsError::ShapeMismatch => Self::ShapeMismatch,
            StepsError::NotPositiveDefinite => Self::NotPositiveDefinite,
        }
    }
}

impl From<IcmError> for FoldFitError {
    fn from(e: IcmError) -> Self {
        match e {
            IcmError::Shape | IcmError::ZeroChunk | IcmError::BlockOutOfRange => {
                Self::ShapeMismatch
            }
            IcmError::Width => Self::Width,
            IcmError::Overflow => Self::Overflow,
            IcmError::NotPositiveDefinite => Self::NotPositiveDefinite,
        }
    }
}

impl From<QuantError> for FoldFitError {
    fn from(e: QuantError) -> Self {
        match e {
            QuantError::Empty | QuantError::ShapeMismatch => Self::ShapeMismatch,
            QuantError::BadBits => Self::BadBits,
            QuantError::LengthOverflow => Self::Overflow,
        }
    }
}

impl From<GptqE8Error> for FoldFitError {
    fn from(e: GptqE8Error) -> Self {
        match e {
            GptqE8Error::ShapeMismatch
            | GptqE8Error::InvalidDamping
            | GptqE8Error::BeamOutOfRange => Self::ShapeMismatch,
            GptqE8Error::ZeroDim => Self::ZeroDim,
            GptqE8Error::BlockMisaligned => Self::Width,
            GptqE8Error::NotPositiveDefinite => Self::NotPositiveDefinite,
            GptqE8Error::NonFinite => Self::NonFinite,
            GptqE8Error::Overflow => Self::Overflow,
            GptqE8Error::Codebook => Self::Codebook,
        }
    }
}

/// The configuration of the production [`fold_fit`]. The fixed flags (`had`, `e8_code`, `bscale`,
/// `hier_scale`, `e8_icm`, `act_order`, `wscale`) are not parameters: the production arm sets them
/// all true.
#[derive(Clone, Copy, Debug)]
pub struct FoldFitConfig {
    /// Outer iterations of the alternation (line 863). Only `1` is ported.
    pub iters: usize,
    /// Metric ridge of `metric(H, ridge)`: `ridge · mean(diag(H))` on the diagonal.
    pub ridge: f64,
    /// Damping passed to [`gptq_e8`] (the Python `damp`).
    pub damp: f64,
    /// Inner refit passes per outer iteration (line 864; the production call site uses `3`).
    pub inner_iters: usize,
    /// Beam width passed to [`gptq_e8`] (the production call site uses `4`).
    pub e8_beam: usize,
    /// `icm_ridge` of [`e8_icm_step`] (`icm_ridge · (n/896) · mean(diag(H))`, line 590).
    pub icm_ridge: f64,
    /// `sweeps` of each [`e8_icm_step`] call (the production call site uses `2`).
    pub sweeps: usize,
}

/// Which candidate of the alternation supplied the winning plane's code, scale and Hadamard scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldFitTag {
    /// The plain `gptq_e8` candidate (`trace[0]`).
    Gptq,
    /// The `refit0` candidate (`trace[1]`).
    Refit0,
}

impl FoldFitTag {
    /// The Python tag string (`"gptq"` / `"refit0"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gptq => "gptq",
            Self::Refit0 => "refit0",
        }
    }
}

/// The stored plane of the winning branch: `Wp = compose(fwht128(s_rot · dh), rh, ch)`.
#[derive(Clone, Debug)]
pub struct PackOut {
    /// Hadamard-domain code plane, row-major `[o, n]`.
    pub s_rot: Vec<f64>,
    /// Per-group row scale from [`quant_r_hier`], row-major `[o, n/128]`.
    pub rh: Vec<f64>,
    /// Per-column scale from [`quant_c_u8`], length `n`.
    pub ch: Vec<f64>,
    /// Hadamard column scale from [`quant_c_u8`], length `n`; all ones on the fallback branch.
    pub dh: Vec<f64>,
}

/// The result of [`fold_fit`]: the dequantised weight and the pack that recomposes to it.
#[derive(Clone, Debug)]
pub struct FoldFit {
    /// The dequantised weight `Wp`, row-major `[o, n]`.
    pub wp: Vec<f64>,
    /// The winning plane `(s_rot, rh, ch, dh)`.
    pub pack: PackOut,
    /// The smallest candidate `J` (`min` over `trace`, the first one on a tie).
    pub j_best: f64,
    /// `J` per candidate: `[J(gptq), J(refit0)]`.
    pub trace: Vec<f64>,
    /// Bit budget of the winning `rh` (`4·o·g + 8·o + 64`).
    pub bits_r: u64,
    /// Bit budget of the winning `ch` (`6·n + 32`).
    pub bits_c: u64,
    /// Bit budget of `dh` (`5·n + 32`).
    pub bits_d: u64,
    /// The candidate that supplied the winning plane.
    pub tag: FoldFitTag,
}

/// A candidate of the alternation, holding the plane it produced.
#[derive(Clone, Debug)]
struct Candidate {
    j: f64,
    tag: FoldFitTag,
    s_rot: Vec<f64>,
    r: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

/// `metric(H, ridge) = H` with `ridge · mean(diag(H))` added to the diagonal (line 431).
///
/// # Complexity
/// Time `O(n²)` to copy and `O(n)` for the diagonal; space `O(n²)`.
fn metric(h: &[f64], n: usize, ridge: f64) -> Vec<f64> {
    let mut hl = h.to_vec();
    let mut mean = 0.0_f64;
    for i in 0..n {
        mean += h[i * n + i];
    }
    mean /= n as f64;
    let add = ridge * mean;
    for i in 0..n {
        hl[i * n + i] += add;
    }
    hl
}

/// `x[:, i] *= v[i]` for a row-major `[o, n]` buffer. Time `O(o·n)`, space `O(1)`.
fn scale_columns(x: &mut [f64], o: usize, n: usize, v: &[f64]) {
    for row in 0..o {
        for col in 0..n {
            x[row * n + col] *= v[col];
        }
    }
}

/// The per-column sign of `d` with the 8-block parity fix (lines 871–878): the sign vector has an
/// even number of `−1` in every 8-block, and a block with an odd count has the sign at its
/// least-`|d|` index (first on a tie) flipped.
///
/// # Complexity
/// Time `O(n)`; space `O(n)` for the sign vector.
fn parity_signs(d: &[f64], n: usize) -> Vec<f64> {
    let mut sign = vec![1.0_f64; n];
    for i in 0..n {
        if d[i] < 0.0 {
            sign[i] = -1.0;
        }
    }
    let nb = n / BLOCK;
    for m in 0..nb {
        let b0 = m * BLOCK;
        let mut negatives = 0_usize;
        for k in 0..BLOCK {
            if sign[b0 + k] < 0.0 {
                negatives += 1;
            }
        }
        if negatives % 2 == 1 {
            let mut min_i = 0_usize;
            let mut min_abs = f64::INFINITY;
            for k in 0..BLOCK {
                let a = d[b0 + k].abs();
                if a < min_abs {
                    min_abs = a;
                    min_i = k;
                }
            }
            sign[b0 + min_i] = -sign[b0 + min_i];
        }
    }
    sign
}

/// The production [`fold_fit`]: the `gptq_e8` candidate, the `inner_iters` refit passes, the candidate
/// selection and the `bscale and had` pack.
///
/// `ws` is the pristine weight, row-major `[o, n]`; `h` the calibration Gram, row-major `[n, n]`;
/// `g_row` the optional row weight (length `o`). `n` must be a positive multiple of [`GROUP`],
/// `iters` must be `1`.
///
/// # Errors
/// [`FoldFitError::Unsupported`] if `iters != 1`; [`FoldFitError::ZeroDim`] if `o == 0` or `n == 0`;
/// [`FoldFitError::Width`] if `n` is not a positive multiple of [`GROUP`];
/// [`FoldFitError::ShapeMismatch`] on a buffer length that disagrees with `o`/`n` or a wrong
/// `g_row` length; [`FoldFitError::NonFinite`] on a non-finite weight, Hessian or row weight;
/// [`FoldFitError::NotPositiveDefinite`] if a reduced system or the damped Hessian is not
/// positive-definite; [`FoldFitError::Overflow`], [`FoldFitError::Codebook`] or
/// [`FoldFitError::BadBits`] from a codebook or quantiser helper; [`FoldFitError::PackMismatch`] if
/// the winning pack does not recompose to `Wp` bit-for-bit.
///
/// # Complexity
/// Time `O(gptq + inner_iters·(o·n² + n³ + e8_icm) + q)`: the dominant terms are the `gptq_e8` beam
/// search (`O(n³ + o·n³/64·256·beam)`), the per-pass `c`/`d` normal systems (`O(o·n²)` to form,
/// `O(n³)` to factor each) with `r_step` at `O(o·n²)`, and the [`e8_icm_step`] sweeps; the pack
/// phase adds two more ICM calls and four `r`/`c` refits, so the total carries a small constant
/// multiple of the refit cost. Measure: f64 multiply-subtracts. Space `O(n² + o·n)` for the plane,
/// the metric and the reduced systems.
pub fn fold_fit(
    ws: &[f64],
    h: &[f64],
    o: usize,
    n: usize,
    g_row: Option<&[f64]>,
    cfg: &FoldFitConfig,
) -> Result<FoldFit, FoldFitError> {
    if cfg.iters != 1 {
        return Err(FoldFitError::Unsupported);
    }
    if o == 0 || n == 0 {
        return Err(FoldFitError::ZeroDim);
    }
    if !n.is_multiple_of(GROUP) {
        return Err(FoldFitError::Width);
    }
    if ws.len() != o * n || h.len() != n * n {
        return Err(FoldFitError::ShapeMismatch);
    }
    if let Some(gr) = g_row {
        if gr.len() != o {
            return Err(FoldFitError::ShapeMismatch);
        }
    }
    if ws.iter().any(|x| !x.is_finite()) || h.iter().any(|x| !x.is_finite()) {
        return Err(FoldFitError::NonFinite);
    }
    if let Some(gr) = g_row {
        if gr.iter().any(|x| !x.is_finite()) {
            return Err(FoldFitError::NonFinite);
        }
    }

    let g = n / GROUP;
    let nb = n / BLOCK;
    let hl = metric(h, n, cfg.ridge);
    let idx_zero = vec![0_u16; o * nb];

    // Step 2: the `gptq_e8` candidate (lines 843–860).
    let ws_rot = fwht128(ws)?;
    let h_gram = had_gram(h, n)?;
    let q = gptq_e8(&ws_rot, &h_gram, o, n, cfg.damp, true, true, cfg.e8_beam)?;
    let s_rot0 = q.s;
    let r0 = q.scale;
    let c0 = vec![1.0_f64; n];
    let d0 = vec![1.0_f64; n];
    let s0 = fwht128(&s_rot0)?;
    let wq0 = compose(&s0, &r0, &c0, o, n)?;
    let j0 = objective(ws, &wq0, &hl, o, n)?;

    let mut cands: Vec<Candidate> = vec![Candidate {
        j: j0,
        tag: FoldFitTag::Gptq,
        s_rot: s_rot0,
        r: r0,
        c: c0,
        d: d0,
    }];
    let mut trace = vec![j0];

    // Step 3: the single refit pass (`iters = 1`, lines 862–889). The inner loop works on owned
    // copies of the `gptq` candidate's plane; the candidate itself keeps the pristine values.
    let mut s_rot = cands[0].s_rot.clone();
    let mut s_cur = fwht128(&s_rot)?;
    let mut r_cur = cands[0].r.clone();
    let mut c_cur = cands[0].c.clone();
    let mut d_cur = cands[0].d.clone();
    for _ in 0..cfg.inner_iters {
        c_cur = c_step(ws, &hl, &s_cur, &r_cur, o, n, g_row)?;
        let smean = c_cur.iter().map(|x| x.abs()).sum::<f64>() / n as f64;
        for v in &mut c_cur {
            *v /= smean;
        }
        for v in &mut r_cur {
            *v *= smean;
        }
        r_cur = r_step(ws, &hl, &s_cur, &c_cur, o, n)?;

        let d_new = d_step(ws, &hl, &s_rot, &r_cur, &c_cur, o, n, g_row)?;
        let sign = parity_signs(&d_new, n);
        scale_columns(&mut s_rot, o, n, &sign);
        let mut group_max = vec![0.0_f64; g];
        for gi in 0..g {
            let mut mx = 0.0_f64;
            for k in 0..GROUP {
                let v = (d_new[gi * GROUP + k] * sign[gi * GROUP + k]).abs();
                if v > mx {
                    mx = v;
                }
            }
            group_max[gi] = mx.max(EPS);
        }
        for i in 0..n {
            d_cur[i] = d_new[i] * sign[i] / group_max[i / GROUP];
        }
        for r in 0..o {
            for gi in 0..g {
                r_cur[r * g + gi] *= group_max[gi];
            }
        }

        let plane = PlaneRef {
            s: &s_rot,
            idx: &idx_zero,
            r: &r_cur,
            c: &c_cur,
            d: &d_cur,
        };
        let refined = e8_icm_step(ws, &hl, plane, o, n, cfg.sweeps, cfg.icm_ridge)?;
        s_rot = refined.s;
        let mut rotated = s_rot.clone();
        scale_columns(&mut rotated, o, n, &d_cur);
        s_cur = fwht128(&rotated)?;
        r_cur = r_step(ws, &hl, &s_cur, &c_cur, o, n)?;
    }
    let wq1 = compose(&s_cur, &r_cur, &c_cur, o, n)?;
    let j1 = objective(ws, &wq1, &hl, o, n)?;
    cands.push(Candidate {
        j: j1,
        tag: FoldFitTag::Refit0,
        s_rot,
        r: r_cur,
        c: c_cur,
        d: d_cur,
    });
    trace.push(j1);

    // Step 4: `best = argmin J`, first on a tie (line 910).
    let mut best = 0_usize;
    for i in 1..cands.len() {
        if cands[i].j < cands[best].j {
            best = i;
        }
    }
    let j_best = cands[best].j;
    let tag = cands[best].tag;
    let s_rot_best = cands[best].s_rot.clone();
    let c_best = cands[best].c.clone();
    let d_best = cands[best].d.clone();

    // The `bscale and had` arm (lines 932–968).
    let (dh, bits_d) = quant_c_u8(&d_best, 5)?;
    let mut rotated = s_rot_best.clone();
    scale_columns(&mut rotated, o, n, &dh);
    let s_dh = fwht128(&rotated)?;
    let rh_hier = quant_r_hier(&r_step(ws, &hl, &s_dh, &c_best, o, n)?, o, g)?;
    let bits_r = rh_hier.bits;
    let rh = rh_hier.rq;
    let c_ref = c_step(ws, &hl, &s_dh, &rh, o, n, g_row)?;
    let (ch, bits_c) = quant_c_u8(&c_ref, 6)?;
    let mut wp = compose(&s_dh, &rh, &ch, o, n)?;
    let mut j_wp = objective(ws, &wp, &hl, o, n)?;
    let mut pack_s_rot = s_rot_best.clone();
    let mut pack_rh = rh;
    let mut pack_ch = ch;
    let mut pack_dh = dh;
    let mut s_rot_cur = s_rot_best;

    // The `e8_icm` re-quantisation candidate (lines 942–950).
    let plane = PlaneRef {
        s: &pack_s_rot,
        idx: &idx_zero,
        r: &pack_rh,
        c: &pack_ch,
        d: &pack_dh,
    };
    let s_rot_q = e8_icm_step(ws, &hl, plane, o, n, cfg.sweeps, cfg.icm_ridge)?.s;
    {
        let mut rotated = s_rot_q.clone();
        scale_columns(&mut rotated, o, n, &pack_dh);
        let s_dh_q = fwht128(&rotated)?;
        let rh_q = quant_r_hier(&r_step(ws, &hl, &s_dh_q, &pack_ch, o, n)?, o, g)?.rq;
        let ch_q = quant_c_u8(&c_step(ws, &hl, &s_dh_q, &rh_q, o, n, g_row)?, 6)?.0;
        let wp_q = compose(&s_dh_q, &rh_q, &ch_q, o, n)?;
        let j_q = objective(ws, &wp_q, &hl, o, n)?;
        if j_q < j_wp {
            j_wp = j_q;
            wp = wp_q;
            s_rot_cur.clone_from(&s_rot_q);
            pack_s_rot = s_rot_q;
            pack_rh = rh_q;
            pack_ch = ch_q;
        }
    }

    // The un-`d` fallback candidate (lines 960–967).
    {
        let s_1 = fwht128(&s_rot_cur)?;
        let r_1 = r_step(ws, &hl, &s_1, &c_best, o, n)?;
        let rh_1 = quant_r_hier(&r_1, o, g)?.rq;
        let ch_1 = quant_c_u8(&c_step(ws, &hl, &s_1, &rh_1, o, n, g_row)?, 6)?.0;
        let wp_1 = compose(&s_1, &rh_1, &ch_1, o, n)?;
        if objective(ws, &wp_1, &hl, o, n)? < j_wp {
            wp = wp_1;
            pack_s_rot = s_rot_cur;
            pack_rh = rh_1;
            pack_ch = ch_1;
            pack_dh = vec![1.0_f64; n];
        }
    }

    // The winning pack must recompose to `Wp` exactly (the Python's `assert`, line 968).
    let mut rotated = pack_s_rot.clone();
    scale_columns(&mut rotated, o, n, &pack_dh);
    let recomposed = compose(&fwht128(&rotated)?, &pack_rh, &pack_ch, o, n)?;
    for k in 0..o * n {
        if recomposed[k].to_bits() != wp[k].to_bits() {
            return Err(FoldFitError::PackMismatch);
        }
    }

    Ok(FoldFit {
        wp,
        pack: PackOut {
            s_rot: pack_s_rot,
            rh: pack_rh,
            ch: pack_ch,
            dh: pack_dh,
        },
        j_best,
        trace,
        bits_r,
        bits_c,
        bits_d,
        tag,
    })
}
