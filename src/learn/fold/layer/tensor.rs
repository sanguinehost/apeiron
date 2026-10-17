//! The per-tensor assembly after the stage solve: the two-plane fit, the `-nb` joint refinement, the
//! MLP-transpose second plane, the low-rank corrector, the post-quantisation bias and the `-resq`
//! row-scale fold, with the stored-plane reconstruction and its store check.
//!
//! Port of the production arm of `fold_transfer.py`'s per-tensor block (`:1733-1956`, `_sink_recon`
//! `:1042-1059`, `_sink_store` `:1061-1071`). The fixed flags are on: `act_order`, `wscale`,
//! `hier_scale`, `bscale`, `e8_code`, `e8_icm`, `backfit_2plane`, `nb_arm`, `q_mlp_trans`,
//! `use_corr`, `use_res`, `use_resq`; absent (not parameters): `yaqa`, `yaqa_refine`,
//! `drift_filter`, `rrr_mode`, `zellner`, `guided`, `glu_comp`, `use_align`, `use_skew`,
//! `mlp_2plane_k896`, `full_attn_2plane`, `cgauge`, `bgptq`, `hpre`.
//!
//! # Deviations from the Python
//!
//! - Every slice is validated and every helper refusal is a typed [`TensorError`] arm; nothing here
//!   panics on caller input.
//! - `ncorr` adds the corrector's bit count **unconditionally**, as `:1884` does (`ncorr += b_corr`
//!   sits outside the `if J(...) < J(...)`); a rejected corrector still spends its factor bits.
//!
//! # Grade
//!
//! `tested` (`tests/learn_fold_tensor.rs`). No extraction or refinement links this file to a Lean
//! theorem, so nothing here is `proved`.
//!
//! # Trust boundary
//!
//! The stage solve, the activation Gram `Hm`, the row weight `g_row` and the bias sums
//! `x_sum`/`y_sum` are another lane's outputs (`layer::stage`), taken as given. Not covered:
//! rustc/LLVM codegen, the reference's fp32 accumulation, and the model forward behind the
//! calibration inputs.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the row-major `[o, n]` kernels index by `o/i/j/k/r` and cast `usize` counts to `f64`, \
              reading closer to `fold_transfer.py` than enumerated iterators would; the public \
              types repeat the module name (`tensor::TensorKind`) by the brief's naming, and \
              `fit_tensor` takes the reference's eight arguments, measured in f64 multiply-adds"
)]

use self::carrier::{eval, grids, reals, Elem};
use crate::learn::fold::fit::corrector::{lowrank_corrector, CorrectorError, Factors, Mode};
use crate::learn::fold::fit::fold_fit::{fold_fit, FoldFitConfig, FoldFitError, PackOut, GROUP};
use crate::learn::fold::fit::icm::{nearest_index, IcmError, Plane, PlaneRef, BLOCK};
use crate::learn::fold::fit::nb_refine::{nb_refine, NbError, NbLog};
use crate::learn::fold::fit::quant::QuantError;
use crate::learn::fold::fit::steps::{compose, fwht_words, objective, StepsError};
use crate::learn::fold::layer::resq::{rescale_hier, rescale_hier_top, ResqError};
use crate::learn::fold::layer::stage::{bias_update, metric, StageError};
use crate::learn::solve::q32::{from_grid, Q32Error, GRID_SCALE};

/// `n/128`, the number of hierarchical row-scale groups.
const fn groups(n: usize) -> usize {
    n / GROUP
}

/// Which linear of the transformer a per-tensor fit serves. The kind selects the second-plane and
/// corrector arms of the script's `two_plane_nm`/`corr_targets`/`rk` dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorKind {
    /// The attention query projection.
    Q,
    /// The attention key projection.
    K,
    /// The attention value projection.
    V,
    /// The attention output projection.
    O,
    /// The MLP gate projection.
    Gate,
    /// The MLP up projection (the `-resq` row-scale target).
    Up,
    /// The MLP down projection.
    Down,
}

/// The caller-computed scalars of one tensor's fit.
///
/// `rg_q` is `quant_ridge · (n/896)` and governs `Hl_m` and `fold_fit`'s internal metric; `fit`'s
/// own `ridge` is ignored. `qmlp_rows` is `a.qmlp_rows` (`k_rows` for `gate`/`up`);
/// `nb_sweeps`/`nb_rounds` are the `-nb` pass's `sweeps`/`rounds` (the script's defaults are
/// [`crate::learn::fold::fit::nb_refine::NB_SWEEPS`] and `2`).
#[derive(Clone, Copy, Debug)]
pub struct TensorParams {
    /// `quant_ridge · (n/896)`, the metric ridge the script threads as `rg_q`.
    pub rg_q: f64,
    /// `a.qmlp_rows`: rows of the `q_mlp_trans` second plane for `gate`/`up`.
    pub qmlp_rows: usize,
    /// The `fold_fit` configuration (its `ridge` is ignored; `rg_q` governs).
    pub fit: FoldFitConfig,
    /// `sweeps` of the `-nb` joint ICM pass.
    pub nb_sweeps: usize,
    /// `rounds` of the `-nb` pass.
    pub nb_rounds: usize,
}

/// The unweighted first moments the `q`/`k`/`v` bias update needs: `x_sum` (`n`) and `y_sum`
/// (`o`) are the summed stage input and teacher output, `n_toks` the token count.
#[derive(Clone, Copy, Debug)]
pub struct BiasStats<'a> {
    /// `Σ_t x_t`, length `n`.
    pub x_sum: &'a [f64],
    /// `Σ_t y_t`, length `o`.
    pub y_sum: &'a [f64],
    /// The token count the moments are divided by.
    pub n_toks: f64,
}

/// The bit budget of one tensor's fit (`fold_transfer.py`'s `nsign`/`nr`/`nc`/`nd`/`ncorr`).
/// `nsign` counts the sign plane: `2·o·n` for `k`/`v`/`o`, `o·n + k_rows·n` for `gate`/`up`, `o·n`
/// for `q`/`down` before the corrector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TensorBits {
    /// Sign-plane bits.
    pub nsign: u64,
    /// Row-scale bits (`bits_r` of each accepted `fold_fit`).
    pub nr: u64,
    /// Column-scale bits (`bits_c`).
    pub nc: u64,
    /// Hadamard-scale bits (`bits_d`).
    pub nd: u64,
    /// Corrector stored-factor bits, added whether or not the corrector is applied.
    pub ncorr: u64,
}

/// One stored plane of a tensor's sink: a pack `s_rot, rh, ch, dh` (row-major) with an optional
/// [`Self::rows`] list, a dense [`Self::dense`] plane (the corrector), or neither beyond the
/// factors. A rows plane's `s_rot`/`rh` are `[rows.len(), n]`/`[rows.len(), n/128]`, a full plane's
/// `[o, n]`/`[o, n/128]`; `ch`/`dh` are always length `n`.
#[derive(Clone, Debug)]
pub struct SinkPlane {
    /// Hadamard-domain code plane, `[rows.len() or o, n]`.
    pub s_rot: Vec<f64>,
    /// Per-group row scale, `[rows.len() or o, n/128]`.
    pub rh: Vec<f64>,
    /// Column scale, length `n`.
    pub ch: Vec<f64>,
    /// Hadamard column scale, length `n`.
    pub dh: Vec<f64>,
    /// The global row indices of a partial (rows) plane; `None` for a full plane.
    pub rows: Option<Vec<usize>>,
    /// The dense corrector `Corr` `[o, n]`, present only on a corrector plane.
    pub dense: Option<Vec<f64>>,
    /// The corrector's stored factors `(Uh, Vh)`, present only on a corrector plane.
    pub factors: Option<Factors>,
}

/// The result of [`fit_tensor`]: the stored weight, its sink planes, the bit budget, the `-nb` log
/// (present only for `k`/`v`/`o`), the `q_mlp_trans` top rows (present only for `gate`/`up` with
/// `k_rows > 0`) and the bias update (present only for `q`/`k`/`v`).
#[derive(Clone, Debug)]
pub struct TensorFit {
    /// The kind this fit serves.
    pub kind: TensorKind,
    /// The dequantised weight `Wp`, row-major `[o, n]`; equal to `recon(planes)` within `1e-6`.
    pub wp: Vec<f64>,
    /// The sink planes, in the script's `_sink_recon` order.
    pub planes: Vec<SinkPlane>,
    /// The bit budget.
    pub bits: TensorBits,
    /// The `-nb` run log, for `k`/`v`/`o`.
    pub nb: Option<NbLog>,
    /// The `q_mlp_trans` second plane's global rows, for `gate`/`up` with `k_rows > 0`.
    pub top_rows: Option<Vec<usize>>,
    /// `db = (y_sum − Wp·x_sum)/n_toks`, for `q`/`k`/`v`.
    pub db: Option<Vec<f64>>,
}

/// A refusal from the per-tensor assembly.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorError {
    /// A slice length disagrees with the declared shape, or a helper saw a mismatched shape.
    ShapeMismatch,
    /// `o == 0` or `n == 0`.
    ZeroDim,
    /// `n` is not a positive multiple of `128`.
    Width,
    /// A weight, Gram, row weight, rescale or bias input is not finite.
    NonFinite,
    /// The requested `qmlp_rows` exceeds the row count `o`.
    QmlpRowsOutOfRange {
        /// The requested `k_rows`.
        k_rows: usize,
        /// The row count `o`.
        o: usize,
    },
    /// A [`BiasStats`] was supplied for a kind that does not update a bias (not `q`/`k`/`v`).
    BiasKind,
    /// [`fold_row_scale`] was called on a fit that is not an `up` fit.
    NotUp,
    /// A plane record disagrees with the declared shape, or the plane structure is not the one the
    /// operation accepts (a rows plane out of range, a duplicate row, a dense plane where a pack is
    /// required).
    PlaneShape,
    /// A value reaches the Q32.32 grid bound `2³¹`, or a checked carrier step leaves its range.
    Grid,
    /// The stored planes do not recompose to `Wp`: neither bit-for-bit nor within
    /// `max|diff| / max|Wp| <= 1e-6`.
    SinkMismatch {
        /// The observed `max|diff| / max|Wp|` (or `max|diff|` when `max|Wp| == 0`).
        rel: f64,
    },
    /// [`fold_fit`] refused.
    Fit(FoldFitError),
    /// [`nb_refine`] refused.
    Nb(NbError),
    /// [`lowrank_corrector`] refused.
    Corrector(CorrectorError),
    /// The `-resq` row-scale fold refused.
    Resq(ResqError),
    /// The stage metric or bias update refused.
    Stage(StageError),
    /// The ICM plane helper refused.
    Icm(IcmError),
    /// A closed-form step refused.
    Steps(StepsError),
    /// A scale quantiser refused.
    Quant(QuantError),
}

impl core::fmt::Display for TensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShapeMismatch => write!(f, "a slice length disagrees with the declared shape"),
            Self::ZeroDim => write!(f, "output or column count must be non-zero"),
            Self::Width => write!(f, "column count is not a positive multiple of 128"),
            Self::NonFinite => write!(f, "an input entry is not finite"),
            Self::QmlpRowsOutOfRange { k_rows, o } => {
                write!(f, "qmlp_rows {k_rows} exceeds the row count {o}")
            }
            Self::BiasKind => write!(f, "a bias update was supplied for a kind without a bias"),
            Self::NotUp => write!(f, "the row-scale fold applies only to an `up` fit"),
            Self::PlaneShape => write!(f, "a sink plane disagrees with the declared shape"),
            Self::Grid => write!(f, "a value or operation left the Q32.32 carrier's range"),
            Self::SinkMismatch { rel } => {
                write!(
                    f,
                    "the stored planes do not recompose to Wp (relative {rel:e})"
                )
            }
            Self::Fit(e) => write!(f, "fold_fit refused: {e:?}"),
            Self::Nb(e) => write!(f, "nb_refine refused: {e:?}"),
            Self::Corrector(e) => write!(f, "lowrank_corrector refused: {e:?}"),
            Self::Resq(e) => write!(f, "the row-scale fold refused: {e:?}"),
            Self::Stage(e) => write!(f, "the stage helper refused: {e:?}"),
            Self::Icm(e) => write!(f, "the ICM helper refused: {e:?}"),
            Self::Steps(e) => write!(f, "a closed-form step refused: {e:?}"),
            Self::Quant(e) => write!(f, "a scale quantiser refused: {e:?}"),
        }
    }
}

impl std::error::Error for TensorError {}

impl From<FoldFitError> for TensorError {
    fn from(e: FoldFitError) -> Self {
        Self::Fit(e)
    }
}
impl From<NbError> for TensorError {
    fn from(e: NbError) -> Self {
        Self::Nb(e)
    }
}
impl From<CorrectorError> for TensorError {
    fn from(e: CorrectorError) -> Self {
        Self::Corrector(e)
    }
}
impl From<ResqError> for TensorError {
    fn from(e: ResqError) -> Self {
        Self::Resq(e)
    }
}
impl From<StageError> for TensorError {
    fn from(e: StageError) -> Self {
        Self::Stage(e)
    }
}
impl From<IcmError> for TensorError {
    fn from(e: IcmError) -> Self {
        Self::Icm(e)
    }
}
impl From<StepsError> for TensorError {
    fn from(e: StepsError) -> Self {
        Self::Steps(e)
    }
}
impl From<QuantError> for TensorError {
    fn from(e: QuantError) -> Self {
        Self::Quant(e)
    }
}
impl From<Q32Error> for TensorError {
    fn from(e: Q32Error) -> Self {
        match e {
            Q32Error::ShapeMismatch => Self::ShapeMismatch,
            _ => Self::Grid,
        }
    }
}

/// `a − b`, one checked subtraction per entry, length the shorter input's (the callers pass equal
/// lengths and a wrong length is refused downstream); temporal and spatial `O(len)`.
fn sub_words(a: &[i64], b: &[i64]) -> Result<Vec<i64>, TensorError> {
    Ok(eval(a.len().min(b.len()), Elem::Sub(a, b))?)
}

/// `a + b`, length as [`sub_words`].
fn add_words(a: &[i64], b: &[i64]) -> Result<Vec<i64>, TensorError> {
    Ok(eval(a.len().min(b.len()), Elem::Add(a, b))?)
}

/// Whether every entry is finite.
fn all_finite(x: &[f64]) -> bool {
    x.iter().all(|v| v.is_finite())
}

/// `row_mass[i] = Σ_j ws[i, j]²`, the `(Ws_nm ** 2).sum(1)` the `q_mlp_trans` row choice ranks by;
/// the squares are carrier words and each row sum is exact in `i128`. Temporal and spatial `O(o·n)`.
fn row_mass(ws: &[f64], o: usize, n: usize) -> Result<Vec<i128>, TensorError> {
    if ws.len() != o.checked_mul(n).ok_or(TensorError::ShapeMismatch)? {
        return Err(TensorError::ShapeMismatch);
    }
    let wq = grids(ws)?;
    let sq = eval(wq.len(), Elem::Square(&wq))?;
    sq.chunks(n)
        .map(|row| {
            row.iter()
                .try_fold(0_i128, |s, &v| s.checked_add(i128::from(v)))
                .ok_or(TensorError::Grid)
        })
        .collect()
}

/// The `k` largest indices of `mass`, descending, ties to the lower index (`torch.topk(mass, k)`
/// order; the words' total order is exact). Temporal `O(o log o)`, spatial `O(o)`.
///
/// # Errors
/// [`TensorError::ShapeMismatch`] if `k > mass.len()`.
fn topk_desc(mass: &[i128], k: usize) -> Result<Vec<usize>, TensorError> {
    if k > mass.len() {
        return Err(TensorError::ShapeMismatch);
    }
    let mut idx: Vec<usize> = (0..mass.len()).collect();
    idx.sort_by(|&a, &b| mass[b].cmp(&mass[a]).then_with(|| a.cmp(&b)));
    idx.truncate(k);
    Ok(idx)
}

/// Gather `rows` of a row-major `[o, n]` Q32.32 word matrix into a dense `[rows.len(), n]` word
/// buffer; used to form the `q_mlp_trans` residual in words, with no `f64` crossing. Temporal and
/// spatial `O(rows.len()·n)`.
///
/// # Errors
/// [`TensorError::ShapeMismatch`] on a row index out of range or a short matrix.
fn gather_rows(m: &[i64], rows: &[usize], n: usize) -> Result<Vec<i64>, TensorError> {
    if n == 0 || rows.iter().any(|&r| r >= m.len() / n) {
        return Err(TensorError::ShapeMismatch);
    }
    let mut out = Vec::with_capacity(rows.len() * n);
    for &r in rows {
        out.extend_from_slice(&m[r * n..(r + 1) * n]);
    }
    Ok(out)
}

/// The codebook row per 8-block of a plane's `s_rot` (row-major `[rlen, n]`), the `idx` field a
/// [`PlaneRef`] carries: the Euclidean nearest codebook row of each block ([`nearest_index`]). The
/// stored `s_rot` value is a codeword, so this recovers the chosen index; the `-nb` joint ICM
/// re-derives it and only requires the length. Temporal `O(rlen·(n/8)·256·8)`, spatial
/// `O(rlen·n/8)`.
///
/// # Errors
/// [`TensorError::Icm`] if a block's values leave the codebook's admitted range.
fn plane_idx(s_rot: &[i64], rlen: usize, n: usize) -> Result<Vec<u16>, TensorError> {
    let nb = n / BLOCK;
    if s_rot.len() != rlen * n {
        return Err(TensorError::PlaneShape);
    }
    let mut idx = Vec::with_capacity(rlen * nb);
    let mut code = [0_i64; BLOCK];
    for row in 0..rlen {
        for m in 0..nb {
            let base = row * n + m * BLOCK;
            code.copy_from_slice(&s_rot[base..base + BLOCK]);
            idx.push(nearest_index(&code)? as u16);
        }
    }
    Ok(idx)
}

/// An owned [`Plane`] view of a fold pack (`s_rot`, `rh`, `ch`, `dh`) plus the derived `idx`.
/// Temporal `O(rlen·(n/8)·256·8 + rlen·n)`, spatial `O(rlen·n)`.
///
/// # Errors
/// [`TensorError::Icm`] from [`plane_idx`].
fn pack_to_plane(pack: &PackOut, rlen: usize, n: usize) -> Result<Plane, TensorError> {
    let s = pack.s_rot.clone();
    let idx = plane_idx(&s, rlen, n)?;
    Ok(Plane {
        s,
        idx,
        r: pack.rh.clone(),
        c: pack.ch.clone(),
        d: pack.dh.clone(),
    })
}

/// A full-plane [`SinkPlane`] (no `rows`, no `dense`, no factors) from its four stored buffers.
fn plane_of(s_rot: Vec<f64>, rh: Vec<f64>, ch: Vec<f64>, dh: Vec<f64>) -> SinkPlane {
    SinkPlane {
        s_rot,
        rh,
        ch,
        dh,
        rows: None,
        dense: None,
        factors: None,
    }
}

/// Whether `p`'s stored buffers have the `[rlen, n]`/`[rlen, n/128]`/`n`/`n` shape.
fn plane_shape(p: &SinkPlane, rlen: usize, n: usize) -> bool {
    let g = groups(n);
    p.s_rot.len() == rlen * n && p.rh.len() == rlen * g && p.ch.len() == n && p.dh.len() == n
}

/// A full pack plane (rows `None`) of a `fold_fit` pack.
fn pack_plane(pack: &PackOut) -> SinkPlane {
    plane_of(
        reals(&pack.s_rot),
        reals(&pack.rh),
        reals(&pack.ch),
        reals(&pack.dh),
    )
}

/// A pack plane from an owned [`Plane`] (the `-nb` output).
fn owned_plane(p: &Plane) -> SinkPlane {
    plane_of(reals(&p.s), reals(&p.r), reals(&p.c), reals(&p.d))
}

/// The full `[o, n]` weight of one sink plane: a dense plane is itself; a pack plane is
/// `compose(fwht128(s_rot·dh), rh, ch)`, scattered in at `rows` when present. Temporal
/// `O(rlen·n·log₂128 + rlen·n)`, spatial `O(o·n)`.
///
/// # Errors
/// [`TensorError::PlaneShape`] on a length or `rows` disagreement; [`TensorError::Width`] if `n` is
/// not a multiple of `128`; [`TensorError::Steps`] from the transform/compose.
fn plane_weight(p: &SinkPlane, o: usize, n: usize) -> Result<Vec<i64>, TensorError> {
    if let Some(dense) = &p.dense {
        if p.rows.is_some() || dense.len() != o * n {
            return Err(TensorError::PlaneShape);
        }
        return Ok(grids(dense)?);
    }
    if !n.is_multiple_of(GROUP) {
        return Err(TensorError::Width);
    }
    let g = groups(n);
    let (rlen, rows) = match &p.rows {
        Some(rs) => {
            if rs.iter().any(|&r| r >= o) {
                return Err(TensorError::PlaneShape);
            }
            (rs.len(), Some(rs.as_slice()))
        }
        None => (o, None),
    };
    if p.s_rot.len() != rlen * n || p.rh.len() != rlen * g || p.ch.len() != n || p.dh.len() != n {
        return Err(TensorError::PlaneShape);
    }
    // rot = s_rot ⊙ dh (repeated down the rows), one rounding per entry on the carrier; the
    // transform and the composition are the word forms, so no real is read back between them.
    let sq = grids(&p.s_rot)?;
    let mut rot = eval(sq.len(), Elem::MulCol(&sq, &grids(&p.dh)?, n))?;
    fwht_words(&mut rot)?;
    let wq = compose(&rot, &grids(&p.rh)?, &grids(&p.ch)?, rlen, n)?;
    match rows {
        None => Ok(wq),
        Some(rs) => {
            let mut full = vec![0_i64; o * n];
            scatter_add(&mut full, &wq, rs, n)?;
            Ok(full)
        }
    }
}

/// Add `src`'s `[rows.len(), n]` rows into `dst`'s `[o, n]` at `rows`, which must be distinct and
/// in range: a duplicate would put two adds in an unspecified order. Temporal
/// `O(rows.len()·n + o)`, spatial `O(o)`.
///
/// # Errors
/// [`TensorError::ShapeMismatch`] on a short `src` or `n == 0`; [`TensorError::PlaneShape`] on a
/// duplicate or out-of-range row; [`TensorError::Grid`] if a word sum leaves the carrier.
fn scatter_add(dst: &mut [i64], src: &[i64], rows: &[usize], n: usize) -> Result<(), TensorError> {
    if n == 0 || src.len() != rows.len() * n {
        return Err(TensorError::ShapeMismatch);
    }
    let o = dst.len() / n;
    let mut inv = vec![usize::MAX; o];
    for (i, &r) in rows.iter().enumerate() {
        if r >= o || inv[r] != usize::MAX {
            return Err(TensorError::PlaneShape);
        }
        inv[r] = i;
    }
    for (r, &i) in inv.iter().enumerate() {
        if i == usize::MAX {
            continue;
        }
        for j in 0..n {
            let k = r * n + j;
            dst[k] = dst[k]
                .checked_add(src[i * n + j])
                .ok_or(TensorError::Grid)?;
        }
    }
    Ok(())
}

/// The stored weight of a plane list: `Σ_p plane_weight(p)` in the script's `_sink_recon` order
/// (full planes add elementwise, a rows plane at its rows, a dense plane is itself). Temporal
/// `Σ_p O(rlen_p·n·log₂128)`, spatial `O(o·n)`.
///
/// # Errors
/// [`TensorError::ZeroDim`] for `o == 0` or `n == 0`; [`TensorError::Width`] if `n` is not a
/// multiple of `128`; [`TensorError::PlaneShape`] for an empty plane list or a plane shape that
/// disagrees with `o`/`n`; [`TensorError::Steps`] from a transform.
pub fn recon(planes: &[SinkPlane], o: usize, n: usize) -> Result<Vec<f64>, TensorError> {
    Ok(reals(&recon_words(planes, o, n)?))
}

/// [`recon`] on the carrier: `Σ_p plane_weight(p)` as Q32.32 words, one checked add per entry per
/// plane; temporal `Σ_p O(rlen_p·n·log₂128)`, spatial `O(o·n)`.
fn recon_words(planes: &[SinkPlane], o: usize, n: usize) -> Result<Vec<i64>, TensorError> {
    if o == 0 || n == 0 {
        return Err(TensorError::ZeroDim);
    }
    if !n.is_multiple_of(GROUP) {
        return Err(TensorError::Width);
    }
    if planes.is_empty() {
        return Err(TensorError::PlaneShape);
    }
    let mut w = plane_weight(&planes[0], o, n)?;
    for p in &planes[1..] {
        let pw = plane_weight(p, o, n)?;
        w = eval(w.len(), Elem::Add(&w, &pw))?;
    }
    Ok(w)
}

/// `_sink_store`'s check: `recon == wp` bit-for-bit, else `max|diff| / max|wp| <= 1e-6`.
///
/// # Errors
/// [`TensorError::SinkMismatch`] on a length mismatch or a relative deviation above `1e-6`;
/// temporal `O(o·n)`, spatial `O(1)`.
fn sink_check(recon_wp: &[i64], wp: &[i64]) -> Result<(), TensorError> {
    if recon_wp.len() != wp.len() {
        return Err(TensorError::SinkMismatch { rel: f64::INFINITY });
    }
    let (mut maxd, mut maxw) = (0_i128, 0_i128);
    for (&a, &b) in recon_wp.iter().zip(wp) {
        maxd = maxd.max((i128::from(a) - i128::from(b)).abs());
        maxw = maxw.max(i128::from(b).abs());
    }
    // The words' ratio is the reals' ratio (the common `2³²` scale cancels); equal buffers give 0.
    let rel = if maxw > 0 {
        maxd as f64 / maxw as f64
    } else {
        maxd as f64 / GRID_SCALE
    };
    if rel <= 1e-6 {
        Ok(())
    } else {
        Err(TensorError::SinkMismatch { rel })
    }
}

/// `J(Ws, W, Hl)` (unweighted, the script's `J`) on the carrier, read once to `f64` for the
/// caller's comparison.
fn j_words(ws: &[i64], w: &[i64], hl: &[i64], o: usize, n: usize) -> Result<f64, TensorError> {
    Ok(from_grid(objective(ws, w, hl, o, n)?))
}

/// The per-tensor assembly after the stage solve.
///
/// `ws_nm` is the stage solve's weight, row-major `[o, n]`; `hm` the calibration Gram `[n, n]`;
/// `g_row` the optional row weight (length `o`); `bias` is required for `q`/`k`/`v` only.
/// `Hl_m = metric(hm, rg_q)`, `fit`'s internal metric uses the same `rg_q`, and `fold_fit` dominates
/// the temporal cost: `3` calls per `k`/`v`/`o` plus one `nb_refine`, `1` for the other kinds plus a
/// corrector for `q`/`down`, each `O(gptq + inner_iters·(o·n² + n³ + e8_icm))`; each `J` is
/// `O(o·n²)` and the spatial peak is `O(n² + o·n)`.
///
/// # Errors
/// [`TensorError::ZeroDim`] for `o == 0` or `n == 0`; [`TensorError::Width`] if `n` is not a
/// positive multiple of `128`; [`TensorError::ShapeMismatch`] on a length, `g_row` or bias-sum
/// disagreement; [`TensorError::NonFinite`] on a non-finite input; [`TensorError::BiasKind`] if
/// `bias` is supplied for a kind without one; [`TensorError::QmlpRowsOutOfRange`] if
/// `qmlp_rows > o`; [`TensorError::SinkMismatch`] if the planes do not recompose to `Wp`; else the
/// failed helper's arm.
pub fn fit_tensor(
    kind: TensorKind,
    ws_nm: &[f64],
    hm: &[f64],
    o: usize,
    n: usize,
    g_row: Option<&[f64]>,
    params: &TensorParams,
    bias: Option<BiasStats<'_>>,
) -> Result<TensorFit, TensorError> {
    if o == 0 || n == 0 {
        return Err(TensorError::ZeroDim);
    }
    if !n.is_multiple_of(GROUP) {
        return Err(TensorError::Width);
    }
    let on = o.checked_mul(n).ok_or(TensorError::ShapeMismatch)?;
    let nn = n.checked_mul(n).ok_or(TensorError::ShapeMismatch)?;
    if ws_nm.len() != on || hm.len() != nn {
        return Err(TensorError::ShapeMismatch);
    }
    if let Some(g) = g_row {
        if g.len() != o {
            return Err(TensorError::ShapeMismatch);
        }
        if !all_finite(g) {
            return Err(TensorError::NonFinite);
        }
    }
    if !all_finite(ws_nm) || !all_finite(hm) {
        return Err(TensorError::NonFinite);
    }
    let has_bias = matches!(kind, TensorKind::Q | TensorKind::K | TensorKind::V);
    if bias.is_some() && !has_bias {
        return Err(TensorError::BiasKind);
    }
    if let Some(b) = &bias {
        if !all_finite(b.x_sum) || !all_finite(b.y_sum) || !b.n_toks.is_finite() {
            return Err(TensorError::NonFinite);
        }
        if b.x_sum.len() != n || b.y_sum.len() != o {
            return Err(TensorError::ShapeMismatch);
        }
    }

    // Hl_m = metric(Hm, rg_q); fold_fit's internal metric uses the same rg_q. `hlq` is the one
    // read-in to the carrier for the `J` evaluations below.
    let hl = metric(hm, n, params.rg_q)?;
    let hlq = grids(&hl)?;
    let cfg = FoldFitConfig {
        ridge: params.rg_q,
        ..params.fit
    };

    // The stage weight enters the carrier once; the residuals, plane sums, plane weights and `J`
    // evaluations below are word operations, while `fold_fit`, `nb_refine`, `lowrank_corrector` and
    // `bias_update` keep their declared `f64` boundaries.
    let wsq = grids(ws_nm)?;
    let hmq = grids(hm)?;
    let g_rowq: Option<Vec<i64>> = match g_row {
        Some(g) => Some(grids(g)?),
        None => None,
    };

    // Step 2: Wp1, the first full plane; `fold_fit` takes and returns words, so the code plane is
    // carried from here to the pack without a grid round trip.
    let f1 = fold_fit(&wsq, &hmq, o, n, g_rowq.as_deref(), &cfg)?;
    let wp1 = f1.wp.clone();
    let mut bits = TensorBits {
        nsign: on as u64,
        nr: f1.bits_r,
        nc: f1.bits_c,
        nd: f1.bits_d,
        ncorr: 0,
    };
    let kind_two_plane = matches!(kind, TensorKind::K | TensorKind::V | TensorKind::O);
    let kind_qmlp = matches!(kind, TensorKind::Gate | TensorKind::Up);
    let kind_corr = matches!(kind, TensorKind::Q | TensorKind::Down);

    let mut planes: Vec<SinkPlane> = vec![pack_plane(&f1.pack)];
    let mut wp: Vec<i64>;
    let mut nb_log: Option<NbLog> = None;
    let mut top_rows: Option<Vec<usize>> = None;

    if kind_two_plane {
        // Step 3: the second plane, the backfit candidate and the `-nb` pass.
        let f2 = fold_fit(&sub_words(&wsq, &wp1)?, &hmq, o, n, g_rowq.as_deref(), &cfg)?;
        let wp2 = f2.wp.clone();
        let mut p1 = f1.pack.clone();
        let mut p2 = f2.pack.clone();
        let mut info2 = (f2.bits_r, f2.bits_c, f2.bits_d);

        // backfit_2plane: Wp1_b against the frozen Wp2, then Wp2_b against the (possibly new) Wp1.
        // Only the accepted packs and the accepted Wp2 bits survive; the `-nb` pass re-derives the
        // weights from the packs, so the intermediate weight buffers are not kept.
        let j_cur = j_words(&wsq, &add_words(&wp1, &wp2)?, &hlq, o, n)?;
        let f1b = fold_fit(&sub_words(&wsq, &wp2)?, &hmq, o, n, g_rowq.as_deref(), &cfg)?;
        let wp1b = f1b.wp.clone();
        let j_1b = j_words(&wsq, &add_words(&wp1b, &wp2)?, &hlq, o, n)?;
        if j_1b < j_cur {
            p1 = f1b.pack.clone();
            let f2b = fold_fit(
                &sub_words(&wsq, &wp1b)?,
                &hmq,
                o,
                n,
                g_rowq.as_deref(),
                &cfg,
            )?;
            let j_2b = j_words(&wsq, &add_words(&wp1b, &f2b.wp)?, &hlq, o, n)?;
            if j_2b < j_1b {
                p2 = f2b.pack.clone();
                info2 = (f2b.bits_r, f2b.bits_c, f2b.bits_d);
            }
        }

        // nb_refine on the two packs; the returned packs and weights replace the pair.
        let pl1 = pack_to_plane(&p1, o, n)?;
        let pl2 = pack_to_plane(&p2, o, n)?;
        let refined = nb_refine(
            ws_nm,
            &hl,
            PlaneRef {
                s: &pl1.s,
                idx: &pl1.idx,
                r: &pl1.r,
                c: &pl1.c,
                d: &pl1.d,
            },
            PlaneRef {
                s: &pl2.s,
                idx: &pl2.idx,
                r: &pl2.r,
                c: &pl2.c,
                d: &pl2.d,
            },
            o,
            n,
            g_row,
            params.fit.icm_ridge,
            params.nb_sweeps,
            params.nb_rounds,
        )?;
        wp = add_words(&grids(&refined.w1)?, &grids(&refined.w2)?)?;
        planes = vec![owned_plane(&refined.p1), owned_plane(&refined.p2)];
        nb_log = Some(refined.log);
        // Bits of the accepted Wp2 fit (`info2`), plus the second sign plane.
        bits.nsign += on as u64;
        bits.nr += info2.0;
        bits.nc += info2.1;
        bits.nd += info2.2;
    } else if kind_qmlp {
        // Step 4: the q_mlp_trans rows plane on `gate`/`up`.
        let k_rows = params.qmlp_rows;
        if k_rows > o {
            return Err(TensorError::QmlpRowsOutOfRange { k_rows, o });
        }
        if k_rows == 0 {
            wp = wp1.clone();
        } else {
            let mass = row_mass(ws_nm, o, n)?;
            let top = topk_desc(&mass, k_rows)?;
            let ws_top = gather_rows(&wsq, &top, n)?;
            let wp1_top = gather_rows(&wp1, &top, n)?;
            let resid = sub_words(&ws_top, &wp1_top)?;
            let g_topq: Option<Vec<i64>> =
                g_rowq.as_ref().map(|g| top.iter().map(|&i| g[i]).collect());
            let f2 = fold_fit(&resid, &hmq, k_rows, n, g_topq.as_deref(), &cfg)?;
            wp = wp1.clone();
            scatter_add(&mut wp, &f2.wp, &top, n)?;
            planes = vec![pack_plane(&f1.pack), pack_plane(&f2.pack)];
            top_rows = Some(top.clone());
            // The rows plane carries a row list; record it on the second plane.
            if let Some(p2) = planes.get_mut(1) {
                p2.rows = Some(top);
            }
            bits.nsign += (k_rows * n) as u64;
            bits.nr += f2.bits_r;
            bits.nc += f2.bits_c;
            bits.nd += f2.bits_d;
        }
    } else {
        // `q`/`down`: the first plane alone before the corrector.
        wp = wp1.clone();
    }

    if kind_corr {
        // Step 5: the low-rank corrector (rank 2 for `q`, 1 for `down`), applied only if it lowers J.
        let rk = if kind == TensorKind::Q { 2 } else { 1 };
        let delta = reals(&sub_words(&wsq, &wp)?);
        let corr = lowrank_corrector(&delta, &hl, o, n, rk, Mode::U8, true)?;
        // The corrector's stored-factor bits are spent whether or not it is applied (`:1884`).
        bits.ncorr += corr.bits;
        let cand = add_words(&wp, &grids(&corr.corr)?)?;
        let j_new = j_words(&wsq, &cand, &hlq, o, n)?;
        let j_old = j_words(&wsq, &wp, &hlq, o, n)?;
        if j_new < j_old {
            wp = cand;
            planes.push(SinkPlane {
                s_rot: Vec::new(),
                rh: Vec::new(),
                ch: Vec::new(),
                dh: Vec::new(),
                rows: None,
                dense: Some(corr.corr.clone()),
                factors: corr.factors.clone(),
            });
        }
    }

    // Step 6: the q/k/v bias from the final Wp.
    let db = match (has_bias, &bias) {
        (true, Some(b)) => Some(bias_update(b.y_sum, b.x_sum, &reals(&wp), b.n_toks)?),
        _ => None,
    };

    // The store check: the planes must recompose to Wp.
    let recon_wp = recon_words(&planes, o, n)?;
    sink_check(&recon_wp, &wp)?;

    Ok(TensorFit {
        kind,
        wp: reals(&wp),
        planes,
        bits,
        nb: nb_log,
        top_rows,
        db,
    })
}

/// The `-resq` row-scale fold for an `up` fit: fold the caller's per-row multiplier `s_sol` into each
/// stored plane's hierarchical row scale (`rh = quant_r_hier(rh·s_sol)`), replace the planes and set
/// `fit.wp = recon(planes)`. The planes must be exactly `{full}` or `{full, rows}` with no dense
/// plane; every other fit is refused. `g = n/128`; temporal `O(o·n + rows.len()·n)`, spatial `O(o·n)`.
///
/// # Errors
/// [`TensorError::NotUp`] if `fit.kind != Up`; [`TensorError::ZeroDim`]/[`TensorError::Width`] on the
/// geometry; [`TensorError::ShapeMismatch`] if `s_sol.len() != o`; [`TensorError::NonFinite`] on a
/// non-finite `s_sol`; [`TensorError::PlaneShape`] on an unaccepted plane; else the rescale's arm.
pub fn fold_row_scale(
    fit: &mut TensorFit,
    s_sol: &[f64],
    o: usize,
    n: usize,
) -> Result<(), TensorError> {
    if fit.kind != TensorKind::Up {
        return Err(TensorError::NotUp);
    }
    if o == 0 || n == 0 {
        return Err(TensorError::ZeroDim);
    }
    if !n.is_multiple_of(GROUP) {
        return Err(TensorError::Width);
    }
    if s_sol.len() != o {
        return Err(TensorError::ShapeMismatch);
    }
    if !all_finite(s_sol) {
        return Err(TensorError::NonFinite);
    }
    let g = groups(n);
    if fit.planes.len() > 2 {
        return Err(TensorError::PlaneShape);
    }
    let full = fit.planes.first().ok_or(TensorError::PlaneShape)?;
    if full.dense.is_some() || full.rows.is_some() || !plane_shape(full, o, n) {
        return Err(TensorError::PlaneShape);
    }

    let rh1 = reals(&rescale_hier(&full.rh, s_sol, o, g)?.rq);
    let mut next = vec![plane_of(
        full.s_rot.clone(),
        rh1,
        full.ch.clone(),
        full.dh.clone(),
    )];

    if let Some(p2) = fit.planes.get(1) {
        let rows = p2.rows.clone().ok_or(TensorError::PlaneShape)?;
        if p2.dense.is_some() || rows.iter().any(|&r| r >= o) {
            return Err(TensorError::PlaneShape);
        }
        let rlen = rows.len();
        if !plane_shape(p2, rlen, n) {
            return Err(TensorError::PlaneShape);
        }
        let rh2 = reals(&rescale_hier_top(&p2.rh, s_sol, &rows, g)?.rq);
        let mut top = plane_of(p2.s_rot.clone(), rh2, p2.ch.clone(), p2.dh.clone());
        top.rows = Some(rows);
        next.push(top);
    }

    // `_sink_store` on the freshly built planes: the reconstruction is the stored weight by
    // construction, so the check is the identity here (kept for parity with the script).
    let recon_wp = recon_words(&next, o, n)?;
    sink_check(&recon_wp, &recon_wp)?;
    fit.planes = next;
    fit.wp = reals(&recon_wp);
    Ok(())
}

/// The layer's Q32.32 boundary and elementwise kernels, shared with [`super::driver`].
///
/// `f64` enters once through [`grids`], every elementwise step is one [`Elem`] pass over `i64` words
/// with `i128` products and one round-to-nearest-even, and [`reals`] is the boundary back to the
/// `f64` signatures the other modules still take. A refusal is a [`Q32Error`], mapped once at each
/// caller. One pass is `o·n` words, at most ~10⁷, so every kernel here is serial.
pub(crate) mod carrier {
    use crate::learn::solve::q32::{from_grid, round_q32_even, to_grid, Q32Error, GRID_BOUND};

    /// `a·b` as one rounded word.
    pub(crate) fn mul(a: i64, b: i64) -> Result<i64, Q32Error> {
        round_q32_even(i128::from(a) * i128::from(b))
    }

    /// `round(num/den)` to nearest, ties to even; `None` on `den == 0` or a quotient outside `i64`.
    /// `q32` carries no rounded division, so the one copy is here.
    fn div_rne(num: i128, den: i128) -> Option<i64> {
        let q = num.checked_div(den)?;
        let r = num.checked_rem(den)?;
        let dabs = den.checked_abs()?;
        let mag2 = r.checked_abs()?.checked_mul(2)?;
        let dir = if (r > 0) == (den > 0) { 1 } else { -1 };
        let q = if mag2 > dabs || (mag2 == dabs && q % 2 != 0) {
            q.checked_add(dir)?
        } else {
            q
        };
        i64::try_from(q).ok()
    }

    /// `num/den` as a Q32.32 word — the real ratio of two words — rounded once.
    pub(crate) fn ratio(num: i128, den: i128) -> Result<i64, Q32Error> {
        div_rne(num.checked_shl(32).ok_or(Q32Error::Overflow)?, den).ok_or(Q32Error::Overflow)
    }

    /// `x` as Q32.32 words; `Overflow` when an entry is not finite or reaches `2³¹`.
    pub(crate) fn grids(x: &[f64]) -> Result<Vec<i64>, Q32Error> {
        x.iter()
            .map(|&v| to_grid(v, GRID_BOUND))
            .collect::<Option<Vec<_>>>()
            .ok_or(Q32Error::Overflow)
    }

    /// `x` as reals, the boundary out of the carrier.
    pub(crate) fn reals(x: &[i64]) -> Vec<f64> {
        x.iter().map(|&v| from_grid(v)).collect()
    }

    /// `Σ x`, exact in `i128`; `Overflow` if the sum leaves `i128`.
    pub(crate) fn sum_i128(x: &[i64]) -> Result<i128, Q32Error> {
        x.iter()
            .try_fold(0_i128, |s, &v| s.checked_add(i128::from(v)))
            .ok_or(Q32Error::Overflow)
    }

    /// The column sums of squared words of a row-major `[rows, cols]` matrix, each square one rounded
    /// word and the sum exact in `i128`; temporal `O(rows·cols)`, spatial `O(cols)`.
    pub(crate) fn col_squares(w: &[i64], rows: usize, cols: usize) -> Result<Vec<i128>, Q32Error> {
        (0..cols)
            .map(|c| {
                (0..rows).try_fold(0_i128, |s, r| {
                    let v = w[r * cols + c];
                    s.checked_add(i128::from(mul(v, v)?))
                        .ok_or(Q32Error::Overflow)
                })
            })
            .collect()
    }

    /// `x[k]/den` as Q32.32 words, one rounding each; temporal and spatial `O(x.len())`.
    pub(crate) fn ratio_words(x: &[i128], den: i64) -> Result<Vec<i64>, Q32Error> {
        x.iter().map(|&v| ratio(v, i128::from(den))).collect()
    }

    /// The elementwise kernels: `eval` writes `out[i]` from the context's words.
    #[derive(Clone, Copy)]
    pub(crate) enum Elem<'a> {
        Sub(&'a [i64], &'a [i64]),
        Add(&'a [i64], &'a [i64]),
        /// `(a − b) + (b − c)`.
        TriSub(&'a [i64], &'a [i64], &'a [i64]),
        /// `a[i]·b[i mod w]`, a column scale broadcast over the rows.
        MulCol(&'a [i64], &'a [i64], usize),
        Square(&'a [i64]),
        /// `a[i] − b[i mod w]`, a column bias broadcast over the rows.
        SubBias(&'a [i64], &'a [i64], usize),
        /// `a[i]/d` as the real ratio of two words.
        Ratio(&'a [i64], i64),
    }

    /// The first `len` entries of `e`; temporal `O(len)`, spatial `O(len)`.
    pub(crate) fn eval(len: usize, e: Elem<'_>) -> Result<Vec<i64>, Q32Error> {
        (0..len)
            .map(|i| match &e {
                Elem::Sub(a, b) => a[i].checked_sub(b[i]).ok_or(Q32Error::Overflow),
                Elem::Add(a, b) => a[i].checked_add(b[i]).ok_or(Q32Error::Overflow),
                Elem::TriSub(a, b, c) => tri_sub(a[i], b[i], c[i]).ok_or(Q32Error::Overflow),
                Elem::MulCol(a, b, w) => mul(a[i], b[i % w]),
                Elem::Square(a) => mul(a[i], a[i]),
                Elem::SubBias(a, b, w) => a[i].checked_sub(b[i % w]).ok_or(Q32Error::Overflow),
                Elem::Ratio(a, d) => ratio(i128::from(a[i]), i128::from(*d)),
            })
            .collect()
    }

    /// `(a − b) + (b − c)`, every step checked.
    fn tri_sub(a: i64, b: i64, c: i64) -> Option<i64> {
        a.checked_sub(b)?.checked_add(b.checked_sub(c)?)
    }
}
