//! The per-tensor assembly after the stage solve: the two-plane fit, the `-nb` joint refinement, the
//! MLP-transpose second plane, the low-rank corrector, the post-quantisation bias and the `-resq`
//! row-scale fold, with the stored-plane reconstruction and its store check.
//!
//! Port of the production arm of the per-tensor block of `fold_transfer.py`'s `run_variant`
//! (`spec/fold_transfer.py:1733-1956`). The production flags are fixed: `act_order`, `wscale`,
//! `hier_scale`, `bscale`, `e8_code`, `e8_icm`, `backfit_2plane`, `nb_arm`, `q_mlp_trans`,
//! `use_corr`, `use_res`, `use_resq` are on; `yaqa`, `yaqa_refine`, `drift_filter`, `rrr_mode`,
//! `zellner`, `guided`, `glu_comp`, `use_align`, `use_skew`, `mlp_2plane_k896`, `full_attn_2plane`,
//! `cgauge`, `bgptq`, `hpre` are off and absent (not parameters). `Hl_m = metric(Hm, rg_q)`.
//!
//! | item | reference |
//! |---|---|
//! | [`fit_tensor`] | `:1733-1883` (the folds, two-plane, `-nb`, `q_mlp_trans`, corrector, bias) |
//! | [`fold_row_scale`] | `:1925-1949` (the `-resq` `up` row-scale fold) |
//! | [`recon`] | `_sink_recon`, `:1042-1059` |
//! | the store check | `_sink_store`, `:1061-1071` (`torch.equal`, else `max|diff|/max|W| <= 1e-6`) |
//!
//! The helpers reused unchanged are [`crate::learn::fold::fit::fold_fit::fold_fit`] (the pack fit),
//! [`crate::learn::fold::fit::nb_refine::nb_refine`] (the `-nb` post-pass),
//! [`crate::learn::fold::fit::corrector::lowrank_corrector`] (the `q`/`down` corrector),
//! [`crate::learn::fold::layer::resq::rescale_hier`] / `rescale_hier_top` (the row-scale fold) and
//! [`crate::learn::fold::layer::stage::bias_update`] / `metric`.
//!
//! # Deviations from the Python
//!
//! - Every slice is validated before use and every helper refusal is carried as a typed
//!   [`TensorError`] arm; the script's `.clone()`/fancy-index panics have no counterpart. No
//!   function here panics on caller input.
//! - `ncorr` adds the corrector's bit count **unconditionally**, exactly as the script does
//!   (`:1884`: `ncorr += b_corr` sits outside the `if J(...) < J(...)` that applies `Corr`). A
//!   rejected corrector still spends its stored-factor bits.
//! - [`TensorParams::fit`]'s `ridge` field is ignored; [`TensorParams::rg_q`] governs both the
//!   `Hl_m` metric and `fold_fit`'s internal metric, because the script has a single `rg_q`.
//!
//! # Grade
//!
//! `tested` (`tests/learn_fold_tensor.rs`). No extraction or refinement links this file to a Lean
//! theorem, so nothing here is `proved`.
//!
//! # Trust boundary
//!
//! The stage solve, the activation Gram `Hm`, the row weight `g_row` and the bias sums
//! `x_sum`/`y_sum` are another lane's outputs (`layer::stage`) and are taken as given; this module
//! makes no claim about them. Not covered: rustc/LLVM codegen, the fp32 accumulation the reference
//! uses, and the model forward that produces the calibration inputs.

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

use crate::learn::fold::fit::corrector::{lowrank_corrector, CorrectorError, Factors, Mode};
use crate::learn::fold::fit::fold_fit::{fold_fit, FoldFitConfig, FoldFitError, PackOut, GROUP};
use crate::learn::fold::fit::icm::{nearest_index, IcmError, Plane, PlaneRef, BLOCK};
use crate::learn::fold::fit::nb_refine::{nb_refine, NbError, NbLog};
use crate::learn::fold::fit::quant::QuantError;
use crate::learn::fold::fit::steps::{compose, fwht128, objective, StepsError};
use crate::learn::fold::layer::resq::{rescale_hier, rescale_hier_top, ResqError};
use crate::learn::fold::layer::stage::{bias_update, metric, StageError};

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
/// `rg_q` is `quant_ridge · (n/896)`; it governs `Hl_m` and `fold_fit`'s internal metric. `fit`
/// carries `damp` and the fixed production flags (`inner_iters` `3`, `e8_beam` `4`, `icm_ridge`
/// `0.20`, `iters` `1`, `sweeps`); its `ridge` field is ignored in favour of `rg_q`. `qmlp_rows`
/// is `a.qmlp_rows` (`k_rows` for `gate`/`up`). `nb_sweeps`/`nb_rounds` are the `-nb` pass's
/// `sweeps`/`rounds` (the script's defaults are [`crate::learn::fold::fit::nb_refine::NB_SWEEPS`]
/// and `2`).
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
///
/// `nr`/`nc`/`nd`/`ncorr` are the winning quantisers' own counts. `nsign` counts the sign plane:
/// `2·o·n` for `k`/`v`/`o` (two full planes), `o·n + k_rows·n` for `gate`/`up`, `o·n` for the one
/// full plane that `q`/`down` fit before the corrector.
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
/// [`Self::rows`] list, or a dense [`Self::dense`] plane (the corrector), or neither beyond the
/// factors.
///
/// A pack plane has [`Self::dense`] `None` and non-empty `s_rot`; a dense plane has all of
/// `s_rot`/`rh`/`ch`/`dh` empty. A rows plane's `s_rot`/`rh` are `[rows.len(), n]` /
/// `[rows.len(), n/128]`; a full plane's are `[o, n]` / `[o, n/128]`. `ch`/`dh` are always length
/// `n`.
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
    /// operation accepts (a rows plane out of range, a dense plane where a pack is required).
    PlaneShape,
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

/// `a − b`, elementwise; the result length is the shorter input's (the callers pass equal lengths,
/// and a wrong length is caught by the downstream shape checks).
fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// `a + b`, elementwise; length as [`sub`].
fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Whether every entry is finite.
fn all_finite(x: &[f64]) -> bool {
    x.iter().all(|v| v.is_finite())
}

/// `row_mass[i] = Σ_j ws[i, j]²`, the `(Ws_nm ** 2).sum(1)` the `q_mlp_trans` row choice ranks by.
///
/// # Complexity
/// Temporal `O(o·n)`; spatial `O(o)`.
fn row_mass(ws: &[f64], o: usize, n: usize) -> Vec<f64> {
    let mut mass = vec![0.0_f64; o];
    for i in 0..o {
        let mut s = 0.0_f64;
        for j in 0..n {
            let v = ws[i * n + j];
            s += v * v;
        }
        mass[i] = s;
    }
    mass
}

/// The `k` largest indices of `mass` in descending order, ties to the lower index (the script's
/// `torch.topk(mass, k).indices` order).
///
/// # Errors
/// [`TensorError::ShapeMismatch`] if `k > mass.len()`.
///
/// # Complexity
/// Temporal `O(o log o)`; spatial `O(o)`.
fn topk_desc(mass: &[f64], k: usize) -> Result<Vec<usize>, TensorError> {
    if k > mass.len() {
        return Err(TensorError::ShapeMismatch);
    }
    let mut idx: Vec<usize> = (0..mass.len()).collect();
    idx.sort_by(|&a, &b| mass[b].total_cmp(&mass[a]).then_with(|| a.cmp(&b)));
    idx.truncate(k);
    Ok(idx)
}

/// Gather `rows` of a row-major `[o, n]` matrix into a dense `[rows.len(), n]` buffer. `rows` is a
/// valid index set from [`topk_desc`], so the copies stay in bounds.
///
/// # Complexity
/// Temporal `O(rows.len()·n)`; spatial `O(rows.len()·n)`.
fn gather_rows(m: &[f64], rows: &[usize], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows.len() * n];
    for (i, &r) in rows.iter().enumerate() {
        out[i * n..i * n + n].copy_from_slice(&m[r * n..r * n + n]);
    }
    out
}

/// The codebook row per 8-block of a plane's `s_rot` (row-major `[rlen, n]`), the `idx` field a
/// [`PlaneRef`] carries: the Euclidean nearest codebook row of each block ([`nearest_index`]).
///
/// The value `s_rot` stores is a codebook codeword (a sign flip of one, per `fold_fit`'s `bscale`
/// step), so this recovers the chosen index; the `-nb` joint ICM does not read the incoming `idx`
/// (it re-derives every block's index from the cost tensor), it only requires the field's length.
///
/// # Errors
/// [`TensorError::Icm`] if a block's values leave the codebook's admitted range.
///
/// # Complexity
/// Temporal `O(rlen·(n/8)·256·8)`; spatial `O(rlen·n/8)`.
fn plane_idx(s_rot: &[f64], rlen: usize, n: usize) -> Result<Vec<u16>, TensorError> {
    let nb = n / BLOCK;
    let mut idx = vec![0_u16; rlen * nb];
    let mut code = [0.0_f64; BLOCK];
    for row in 0..rlen {
        for m in 0..nb {
            for k in 0..BLOCK {
                code[k] = s_rot[row * n + m * BLOCK + k];
            }
            let j = nearest_index(&code)?;
            idx[row * nb + m] = j as u16;
        }
    }
    Ok(idx)
}

/// An owned [`Plane`] view of a fold pack (`s_rot`, `rh`, `ch`, `dh`) plus the derived `idx`.
///
/// # Errors
/// [`TensorError::Icm`] from [`plane_idx`].
///
/// # Complexity
/// Temporal `O(rlen·(n/8)·256·8 + rlen·n)`; spatial `O(rlen·n)`.
fn pack_to_plane(pack: &PackOut, rlen: usize, n: usize) -> Result<Plane, TensorError> {
    let idx = plane_idx(&pack.s_rot, rlen, n)?;
    Ok(Plane {
        s: pack.s_rot.clone(),
        idx,
        r: pack.rh.clone(),
        c: pack.ch.clone(),
        d: pack.dh.clone(),
    })
}

/// A full pack plane (rows `None`) of a `fold_fit` pack.
fn pack_plane(pack: &PackOut) -> SinkPlane {
    SinkPlane {
        s_rot: pack.s_rot.clone(),
        rh: pack.rh.clone(),
        ch: pack.ch.clone(),
        dh: pack.dh.clone(),
        rows: None,
        dense: None,
        factors: None,
    }
}

/// A pack plane from an owned [`Plane`] (the `-nb` output).
fn owned_plane(p: &Plane) -> SinkPlane {
    SinkPlane {
        s_rot: p.s.clone(),
        rh: p.r.clone(),
        ch: p.c.clone(),
        dh: p.d.clone(),
        rows: None,
        dense: None,
        factors: None,
    }
}

/// The full `[o, n]` weight of one sink plane: a dense plane is itself; a pack plane is
/// `compose(fwht128(s_rot·dh), rh, ch)` (bitwise the same expression as `_sink_recon` and
/// `fold_fit`'s store check), scattered into `[o, n]` at `rows` when present.
///
/// # Errors
/// [`TensorError::PlaneShape`] on a length that disagrees with `o`/`n` or `rows`, or a rows index
/// out of range; [`TensorError::Width`] if `n` is not a multiple of `128`; [`TensorError::Steps`]
/// from the transform/compose.
///
/// # Complexity
/// Temporal `O(rlen·n·log2(128))` for the transform plus `O(rlen·n)`; spatial `O(o·n)`.
fn plane_weight(p: &SinkPlane, o: usize, n: usize) -> Result<Vec<f64>, TensorError> {
    if let Some(dense) = &p.dense {
        if p.rows.is_some() || dense.len() != o * n {
            return Err(TensorError::PlaneShape);
        }
        return Ok(dense.clone());
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
    // rot = s_rot ⊙ dh (repeated down the rows).
    let mut rot = p.s_rot.clone();
    for row in 0..rlen {
        for col in 0..n {
            rot[row * n + col] *= p.dh[col];
        }
    }
    let t = fwht128(&rot)?;
    let wi = compose(&t, &p.rh, &p.ch, rlen, n)?;
    match rows {
        None => Ok(wi),
        Some(rs) => {
            let mut full = vec![0.0_f64; o * n];
            for (i, &r) in rs.iter().enumerate() {
                for j in 0..n {
                    full[r * n + j] += wi[i * n + j];
                }
            }
            Ok(full)
        }
    }
}

/// The stored weight of a plane list: `Σ_p plane_weight(p)` in the script's `_sink_recon` order
/// (full planes add elementwise, a rows plane adds at its rows, a dense plane is itself).
///
/// # Errors
/// [`TensorError::ZeroDim`] for `o == 0` or `n == 0`; [`TensorError::Width`] if `n` is not a
/// multiple of `128`; [`TensorError::PlaneShape`] for an empty plane list or a plane shape that
/// disagrees with `o`/`n`; [`TensorError::Steps`] from a transform.
///
/// # Complexity
/// Temporal `Σ_p O(rlen_p·n·log2(128))`; spatial `O(o·n)`.
pub fn recon(planes: &[SinkPlane], o: usize, n: usize) -> Result<Vec<f64>, TensorError> {
    if o == 0 || n == 0 {
        return Err(TensorError::ZeroDim);
    }
    if !n.is_multiple_of(GROUP) {
        return Err(TensorError::Width);
    }
    if planes.is_empty() {
        return Err(TensorError::PlaneShape);
    }
    let on = o.checked_mul(n).ok_or(TensorError::ShapeMismatch)?;
    let mut w = vec![0.0_f64; on];
    for p in planes {
        let pw = plane_weight(p, o, n)?;
        for k in 0..on {
            w[k] += pw[k];
        }
    }
    Ok(w)
}

/// `_sink_store`'s check: `recon == wp` bit-for-bit, else `max|diff| / max|wp| <= 1e-6`.
///
/// # Errors
/// [`TensorError::SinkMismatch`] on a length mismatch or a relative deviation above `1e-6`.
///
/// # Complexity
/// Temporal `O(o·n)`; spatial `O(1)`.
fn sink_check(recon_wp: &[f64], wp: &[f64]) -> Result<(), TensorError> {
    if recon_wp.len() != wp.len() {
        return Err(TensorError::SinkMismatch { rel: f64::INFINITY });
    }
    let mut equal = true;
    let mut maxd = 0.0_f64;
    let mut maxw = 0.0_f64;
    for k in 0..recon_wp.len() {
        if recon_wp[k].to_bits() != wp[k].to_bits() {
            equal = false;
        }
        let d = (recon_wp[k] - wp[k]).abs();
        if d > maxd {
            maxd = d;
        }
        let a = wp[k].abs();
        if a > maxw {
            maxw = a;
        }
    }
    if equal {
        return Ok(());
    }
    let rel = if maxw > 0.0 { maxd / maxw } else { maxd };
    if rel <= 1e-6 {
        Ok(())
    } else {
        Err(TensorError::SinkMismatch { rel })
    }
}

/// `J(Ws, W, Hl)` (unweighted), the script's `J`.
fn j_of(ws: &[f64], w: &[f64], hl: &[f64], o: usize, n: usize) -> Result<f64, TensorError> {
    Ok(objective(ws, w, hl, o, n)?)
}

/// The per-tensor assembly after the stage solve.
///
/// `ws_nm` is the stage solve's weight, row-major `[o, n]`; `hm` the calibration Gram `[n, n]`;
/// `g_row` the optional row weight (length `o`). `Hl_m = metric(hm, rg_q)` is built here; `fit`'s
/// internal metric uses the same `rg_q`. `bias` is required for `q`/`k`/`v` only.
///
/// # Errors
/// [`TensorError::ZeroDim`] for `o == 0` or `n == 0`; [`TensorError::Width`] if `n` is not a
/// positive multiple of `128`; [`TensorError::ShapeMismatch`] on a slice length that disagrees with
/// `o`/`n` or a wrong `g_row`; [`TensorError::NonFinite`] on a non-finite weight, Gram, row weight
/// or bias input; [`TensorError::BiasKind`] if `bias` is supplied for a kind without a bias;
/// [`TensorError::QmlpRowsOutOfRange`] if `qmlp_rows > o` for `gate`/`up`; [`TensorError::SinkMismatch`]
/// if the planes do not recompose to `Wp`; otherwise the refusal of the [`fold_fit`], [`nb_refine`],
/// [`lowrank_corrector`] or [`bias_update`] helper.
///
/// # Complexity
/// Temporal dominated by the `fold_fit` calls: `3` per `k`/`v`/`o` (two-plane plus the backfit
/// candidate) plus one `nb_refine`, `1` per `gate`/`up`/`q`/`down` plus one corrector for `q`/`down`;
/// each `fold_fit` is `O(gptq + inner_iters·(o·n² + n³ + e8_icm))`, each `nb_refine` is
/// `rounds·(e8_joint_icm_2plane + steps)`, each `J` is `O(o·n²)`. Spatial `O(n² + o·n)`.
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

    // Hl_m = metric(Hm, rg_q); fold_fit's internal metric uses the same rg_q.
    let hl = metric(hm, n, params.rg_q)?;
    let cfg = FoldFitConfig {
        ridge: params.rg_q,
        ..params.fit
    };

    // Step 2: Wp1, the first full plane.
    let f1 = fold_fit(ws_nm, hm, o, n, g_row, &cfg)?;
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
    let mut wp: Vec<f64>;
    let mut nb_log: Option<NbLog> = None;
    let mut top_rows: Option<Vec<usize>> = None;

    if kind_two_plane {
        // Step 3: the second plane, the backfit candidate and the `-nb` pass.
        let f2 = fold_fit(&sub(ws_nm, &f1.wp), hm, o, n, g_row, &cfg)?;
        let mut p1 = f1.pack.clone();
        let mut p2 = f2.pack.clone();
        let mut info2 = (f2.bits_r, f2.bits_c, f2.bits_d);

        // backfit_2plane: Wp1_b against the frozen Wp2, then Wp2_b against the (possibly new) Wp1.
        // Only the accepted packs and the accepted Wp2 bits survive; the `-nb` pass re-derives the
        // weights from the packs, so the intermediate weight buffers are not kept.
        let j_cur = j_of(ws_nm, &add(&f1.wp, &f2.wp), &hl, o, n)?;
        let f1b = fold_fit(&sub(ws_nm, &f2.wp), hm, o, n, g_row, &cfg)?;
        let j_1b = j_of(ws_nm, &add(&f1b.wp, &f2.wp), &hl, o, n)?;
        if j_1b < j_cur {
            p1 = f1b.pack.clone();
            let f2b = fold_fit(&sub(ws_nm, &f1b.wp), hm, o, n, g_row, &cfg)?;
            let j_2b = j_of(ws_nm, &add(&f1b.wp, &f2b.wp), &hl, o, n)?;
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
        wp = add(&refined.w1, &refined.w2);
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
            wp = f1.wp.clone();
        } else {
            let mass = row_mass(ws_nm, o, n);
            let top = topk_desc(&mass, k_rows)?;
            let ws_top = gather_rows(ws_nm, &top, n);
            let wp1_top = gather_rows(&f1.wp, &top, n);
            let resid = sub(&ws_top, &wp1_top);
            let g_top: Option<Vec<f64>> = g_row.map(|g| top.iter().map(|&i| g[i]).collect());
            let f2 = fold_fit(&resid, hm, k_rows, n, g_top.as_deref(), &cfg)?;
            wp = f1.wp.clone();
            for (i, &r) in top.iter().enumerate() {
                for j in 0..n {
                    wp[r * n + j] += f2.wp[i * n + j];
                }
            }
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
        wp = f1.wp.clone();
    }

    if kind_corr {
        // Step 5: the low-rank corrector (rank 2 for `q`, 1 for `down`), applied only if it lowers J.
        let rk = if kind == TensorKind::Q { 2 } else { 1 };
        let delta = sub(ws_nm, &wp);
        let corr = lowrank_corrector(&delta, &hl, o, n, rk, Mode::U8, true)?;
        // The corrector's stored-factor bits are spent whether or not it is applied (`:1884`).
        bits.ncorr += corr.bits;
        let cand = add(&wp, &corr.corr);
        let j_new = j_of(ws_nm, &cand, &hl, o, n)?;
        let j_old = j_of(ws_nm, &wp, &hl, o, n)?;
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
        (true, Some(b)) => Some(bias_update(b.y_sum, b.x_sum, &wp, b.n_toks)?),
        _ => None,
    };

    // The store check: the planes must recompose to Wp.
    let recon_wp = recon(&planes, o, n)?;
    sink_check(&recon_wp, &wp)?;

    Ok(TensorFit {
        kind,
        wp,
        planes,
        bits,
        nb: nb_log,
        top_rows,
        db,
    })
}

/// The `-resq` row-scale fold for an `up` fit: fold the caller's per-row multiplier `s_sol` into each
/// stored plane's hierarchical row scale (`rh = quant_r_hier(rh·s_sol)`), replace the planes and set
/// `fit.wp = recon(planes)`.
///
/// The fit must be an `up` fit whose planes are exactly `{full}` or `{full, rows}` with no dense
/// plane; every other fit is refused. `g = n/128`.
///
/// # Errors
/// [`TensorError::NotUp`] if `fit.kind != Up`; [`TensorError::ZeroDim`] for `o == 0` or `n == 0`;
/// [`TensorError::Width`] if `n` is not a multiple of `128`; [`TensorError::ShapeMismatch`] if
/// `s_sol.len() != o`; [`TensorError::NonFinite`] on a non-finite `s_sol`; [`TensorError::PlaneShape`]
/// on a plane structure or shape the fold does not accept; [`TensorError::Resq`] from the rescale,
/// [`TensorError::Steps`] from the reconstruction.
///
/// # Complexity
/// Temporal `O(o·n + (rows.len())·n)` for the two rescales plus the reconstruction
/// `O(o·n·log2(128))`; spatial `O(o·n)`.
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
    if full.dense.is_some() || full.rows.is_some() {
        return Err(TensorError::PlaneShape);
    }
    if full.s_rot.len() != o * n
        || full.rh.len() != o * g
        || full.ch.len() != n
        || full.dh.len() != n
    {
        return Err(TensorError::PlaneShape);
    }

    let rh1 = rescale_hier(&full.rh, s_sol, o, g)?.rq;
    let mut next = vec![SinkPlane {
        s_rot: full.s_rot.clone(),
        rh: rh1,
        ch: full.ch.clone(),
        dh: full.dh.clone(),
        rows: None,
        dense: None,
        factors: None,
    }];

    if let Some(p2) = fit.planes.get(1) {
        let rows = p2.rows.clone().ok_or(TensorError::PlaneShape)?;
        if p2.dense.is_some() || rows.iter().any(|&r| r >= o) {
            return Err(TensorError::PlaneShape);
        }
        let rlen = rows.len();
        if p2.s_rot.len() != rlen * n
            || p2.rh.len() != rlen * g
            || p2.ch.len() != n
            || p2.dh.len() != n
        {
            return Err(TensorError::PlaneShape);
        }
        let rh2 = rescale_hier_top(&p2.rh, s_sol, &rows, g)?.rq;
        next.push(SinkPlane {
            s_rot: p2.s_rot.clone(),
            rh: rh2,
            ch: p2.ch.clone(),
            dh: p2.dh.clone(),
            rows: Some(rows),
            dense: None,
            factors: None,
        });
    }

    // `_sink_store` on the freshly built planes: the reconstruction is the stored weight by
    // construction, so the check is the identity here (kept for parity with the script).
    let recon_wp = recon(&next, o, n)?;
    sink_check(&recon_wp, &recon_wp)?;
    fit.planes = next;
    fit.wp = recon_wp;
    Ok(())
}
