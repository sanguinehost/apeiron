//! The run glue from the fold fit to the finishing P-step: the fitted [`TensorFit`] sink planes
//! become the finisher's [`CodedPlane`]s, the student's norms and biases become the zero-bit
//! tunables, and the calibration chunks drive [`Finisher::step`] against the teacher's logits.
//!
//! # Representation
//!
//! Every value is a Q32.32 `i64` word. A [`SinkPlane`] pack (`s_rot`, `rh`, `ch`, `dh`) enters
//! [`CodedPlane::from_pack`] verbatim; the pack does not retain the `r` hierarchy's `(lo, sp)`, so
//! [`row_grid`] recovers a grid on which every stored row value is a fixed point of the plane's own
//! snap. A dense corrector plane becomes [`Linear::extra`]. The teacher logit source is the
//! existing whole-model [`forward`] that `run/convert.rs` and `stats/heldout.rs` already call; this
//! module adds no forward.
//!
//! # Execution
//!
//! [`planes_from_fits`] is `Σ` over the `o × n` planes of [`CodedPlane::from_pack`]'s codebook
//! encode, pooled inside the plane; [`finish_model`] is `steps` teacher forwards plus `steps`
//! finisher steps. No loop here is tensor-sized beyond those calls, so none dispatches its own pool
//! work.
//!
//! # Evidence grade
//!
//! `tested` (`tests/learn_fold_run_finish.rs`); nothing here is extracted or refined.
//!
//! # Trust boundary
//!
//! Not covered: rustc/LLVM codegen, the `GGUF` write ([`crate::learn::fold::run::transfer`]), and
//! the teacher forward's own contract ([`crate::learn::grad::model::forward`]).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the plane kernels index row-major `[o, n]` buffers by `o/i/j/g` and name the pack \
              fields `rh`/`ch`/`dh`; the `usize → i128` casts are on validated shape counts"
)]

use crate::learn::fold::finish::plane::{div_ne, CodedPlane, PlaneError};
use crate::learn::fold::finish::train::{
    assemble, Finisher, Frozen, LayerTunables, LayerTuned, Linear, Planes, Schedule, TrainError,
    Tunables, N_KINDS,
};
use crate::learn::fold::fit::quant::quant_r_hier;
use crate::learn::fold::fit::steps::GROUP;
use crate::learn::fold::layer::driver::ModelResult;
use crate::learn::fold::layer::tensor::{TensorFit, TensorKind};
use crate::learn::grad::model::{forward, Config, LayerWeights, ModelError, Weights};
use crate::learn::solve::q32::narrow;

/// A refusal from the fold-fit-to-finisher glue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinishRunError {
    /// A fit, plane or tunable shape disagrees with the model geometry.
    Shape(String),
    /// A coded plane refused.
    Plane(PlaneError),
    /// The finishing step refused.
    Train(TrainError),
    /// The teacher forward refused.
    Model(ModelError),
    /// A fitted row scale has no 16-level grid on which its stored values are fixed points, so the
    /// plane cannot be rebuilt exactly.
    Grid(String),
}

impl From<PlaneError> for FinishRunError {
    fn from(e: PlaneError) -> Self {
        FinishRunError::Plane(e)
    }
}

impl From<TrainError> for FinishRunError {
    fn from(e: TrainError) -> Self {
        FinishRunError::Train(e)
    }
}

impl From<ModelError> for FinishRunError {
    fn from(e: ModelError) -> Self {
        FinishRunError::Model(e)
    }
}

impl core::fmt::Display for FinishRunError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Shape(m) => write!(f, "shape: {m}"),
            Self::Plane(e) => write!(f, "plane: {e:?}"),
            Self::Train(e) => write!(f, "finishing step: {e:?}"),
            Self::Model(e) => write!(f, "teacher forward: {e:?}"),
            Self::Grid(m) => write!(f, "row grid: {m}"),
        }
    }
}

impl std::error::Error for FinishRunError {}

/// `(o_total, n)` of linear `k`, in `q, k, v, o, gate, up, down` order.
///
/// # Errors
/// [`FinishRunError::Shape`] past [`N_KINDS`]; [`FinishRunError::Model`] from the `q_out`/`kv_out`
/// products.
fn kind_dims(cfg: &Config, k: usize) -> Result<(usize, usize), FinishRunError> {
    let e = cfg.n_embd;
    let q_out = cfg.q_out()?;
    let kv_out = cfg.kv_out()?;
    match k {
        0 => Ok((q_out, e)),
        1 | 2 => Ok((kv_out, e)),
        3 => Ok((e, q_out)),
        4 | 5 => Ok((cfg.n_ff, e)),
        6 => Ok((e, cfg.n_ff)),
        _ => Err(FinishRunError::Shape(format!(
            "linear kind {k} >= {N_KINDS}"
        ))),
    }
}

/// The [`TensorKind`] of `k` in [`KIND_NAMES`] order.
fn kind_of(k: usize) -> TensorKind {
    match k {
        0 => TensorKind::Q,
        1 => TensorKind::K,
        2 => TensorKind::V,
        3 => TensorKind::O,
        4 => TensorKind::Gate,
        5 => TensorKind::Up,
        _ => TensorKind::Down,
    }
}

/// The snap of one stored row value onto the grid `(lo, sp)`: `lo + round(k·sp/15)` with
/// `k = clamp(round((v − lo)·15/sp), 0, 15)`, the exact `CodedPlane::qvals` rule. `O(1)`.
fn snap_word(v: i64, lo: i64, sp: i64) -> Option<i64> {
    let x = i128::from(v.checked_sub(lo)?);
    let k = div_ne(x * 15, i128::from(sp))?.clamp(0, 15);
    lo.checked_add(narrow(div_ne(k * i128::from(sp), 15)?)?)
}

/// Snap every stored row value of an `o × g` scale table.
fn snap_rows(rh: &[i64], lo: &[i64], sp: &[i64], o: usize, g: usize) -> Option<Vec<i64>> {
    if o == 0 || g == 0 || rh.len() != o * g || lo.len() != o || sp.len() != o {
        return None;
    }
    let out: Option<Vec<i64>> = (0..rh.len())
        .map(|i| snap_word(rh[i], lo[i / g], sp[i / g]))
        .collect();
    out
}

/// An exact 16-level grid for one row of stored row scales: `lo = min` and the first
/// `sp = round(15·(max − min)/Δ)` for which every value is a fixed point of [`snap_word`].
///
/// # Errors
/// [`FinishRunError::Shape`] on an empty row; [`FinishRunError::Grid`] when no candidate of the
/// 15 steps reproduces the row.
fn exact_row(row: &[i64]) -> Result<(i64, i64), FinishRunError> {
    let mn = *row
        .iter()
        .min()
        .ok_or_else(|| FinishRunError::Shape("empty row scale".to_string()))?;
    let mx = *row
        .iter()
        .max()
        .ok_or_else(|| FinishRunError::Shape("empty row scale".to_string()))?;
    if mn == mx {
        return Ok((mn, 1));
    }
    let d = mx
        .checked_sub(mn)
        .ok_or_else(|| FinishRunError::Grid("row span".to_string()))?;
    for delta in 1..=15_i128 {
        let Some(sp) = div_ne(15 * i128::from(d), delta).and_then(narrow) else {
            continue;
        };
        if sp > 0 && row.iter().all(|&v| snap_word(v, mn, sp) == Some(v)) {
            return Ok((mn, sp));
        }
    }
    Err(FinishRunError::Grid(
        "no 16-level fixed-point grid".to_string(),
    ))
}

/// The `(lo, sp)` grid of a stored `o × g` row-scale table: [`quant_r_hier`]'s own grid when every
/// value is a fixed point of it, else the per-row [`exact_row`] search.
///
/// # Complexity
/// Temporal `O(o·g)` for the hierarchy plus `O(12·o·g)` for the fallback (twelve candidate spans
/// per row, one snap per entry); spatial `O(o·g)`.
fn row_grid(rh: &[i64], o: usize, g: usize) -> Result<(Vec<i64>, Vec<i64>), FinishRunError> {
    if o == 0 || g == 0 || rh.len() != o * g {
        return Err(FinishRunError::Shape("row scale shape".to_string()));
    }
    if let Ok(h) = quant_r_hier(rh, o, g) {
        if snap_rows(rh, &h.r_lo, &h.r_sp, o, g).is_some_and(|s| s == rh) {
            return Ok((h.r_lo, h.r_sp));
        }
    }
    let mut lo = Vec::with_capacity(o);
    let mut sp = Vec::with_capacity(o);
    for i in 0..o {
        let (l, s) = exact_row(&rh[i * g..(i + 1) * g])?;
        lo.push(l);
        sp.push(s);
    }
    Ok((lo, sp))
}

/// The grid and code plane of one [`SinkPlane`](crate::learn::fold::layer::tensor::SinkPlane),
/// with a `rows` plane's global indices taken from the sink record or, for the second plane of
/// `gate`/`up`, from the fit's `top_rows`.
///
/// # Errors
/// [`FinishRunError::Shape`] on a plane whose buffers disagree with `o_total × n`;
/// [`FinishRunError::Grid`] from [`row_grid`]; [`FinishRunError::Plane`] from
/// [`CodedPlane::from_pack`].
fn coded_plane(
    p: &crate::learn::fold::layer::tensor::SinkPlane,
    top_rows: Option<&[usize]>,
    o_total: usize,
    n: usize,
    k: usize,
    pi: usize,
) -> Result<CodedPlane, FinishRunError> {
    let full = o_total
        .checked_mul(n)
        .ok_or_else(|| FinishRunError::Shape("plane area".to_string()))?;
    let (o_p, rows) = match &p.rows {
        Some(rw) => (rw.len(), Some(rw.clone())),
        None if p.s_rot.len() == full => (o_total, None),
        None if (k == 4 || k == 5) && pi == 1 => {
            let top = top_rows
                .ok_or_else(|| FinishRunError::Shape("gate/up top_rows absent".to_string()))?;
            (top.len(), Some(top.to_vec()))
        }
        None => {
            return Err(FinishRunError::Shape(format!(
                "plane {pi} of kind {k}: {} words != {o_total}x{n}",
                p.s_rot.len()
            )))
        }
    };
    let g = n / GROUP;
    let (lo, sp) = row_grid(&p.rh, o_p, g)?;
    Ok(CodedPlane::from_pack(
        &p.s_rot,
        &lo,
        &sp,
        &p.rh,
        &p.ch,
        &p.dh,
        rows.as_deref(),
        o_p,
        n,
    )?)
}

/// The finisher's [`Planes`] from the fit's per-layer [`TensorFit`]s: each pack plane becomes a
/// [`CodedPlane`], a dense corrector plane becomes [`Linear::extra`], and every linear keeps the
/// model's `o_total × n`.
///
/// # Errors
/// [`FinishRunError::Shape`] on a layer/kind count or a buffer that disagrees with the geometry;
/// [`FinishRunError::Grid`]/[`FinishRunError::Plane`] from a plane.
///
/// # Complexity
/// Temporal `O(Σ_{l,k} o·n·CODE_LEN)` for the codebook encode inside `from_pack` plus
/// `O(Σ o·n)` for the row grids; spatial `O(Σ_{l,k} o·n)` for the rebuilt planes.
pub fn planes_from_fits(cfg: &Config, fits: &[Vec<TensorFit>]) -> Result<Planes, FinishRunError> {
    if fits.len() != cfg.n_layer {
        return Err(FinishRunError::Shape(format!(
            "fits {} != n_layer {}",
            fits.len(),
            cfg.n_layer
        )));
    }
    let mut layers: Vec<Vec<Linear>> = Vec::with_capacity(cfg.n_layer);
    for (l, fs) in fits.iter().enumerate() {
        if fs.len() != N_KINDS {
            return Err(FinishRunError::Shape(format!(
                "layer {l}: {} kinds != {N_KINDS}",
                fs.len()
            )));
        }
        let mut row: Vec<Linear> = Vec::with_capacity(N_KINDS);
        for (k, fit) in fs.iter().enumerate() {
            if fit.kind != kind_of(k) {
                return Err(FinishRunError::Shape(format!(
                    "layer {l} kind {k}: fit is {:?}",
                    fit.kind
                )));
            }
            let (o_total, n) = kind_dims(cfg, k)?;
            let mut lin = Linear::new(o_total, n);
            for (pi, p) in fit.planes.iter().enumerate() {
                if let Some(dense) = &p.dense {
                    lin.extra = Some(dense.clone());
                } else {
                    lin.planes
                        .push(coded_plane(p, fit.top_rows.as_deref(), o_total, n, k, pi)?);
                }
            }
            row.push(lin);
        }
        layers.push(row);
    }
    Ok(Planes::new(layers))
}

/// One bias base: the student's fitted bias, else the fit's `db`, else zeros.
///
/// # Errors
/// [`FinishRunError::Shape`] on a supplied width that disagrees with the linear.
fn bias_base(
    student: Option<&[i64]>,
    db: Option<&[i64]>,
    width: usize,
    what: &str,
) -> Result<Vec<i64>, FinishRunError> {
    match student.or(db) {
        Some(v) if v.len() == width => Ok(v.to_vec()),
        Some(v) => Err(FinishRunError::Shape(format!(
            "{what} width {} != {width}",
            v.len()
        ))),
        None => Ok(vec![0_i64; width]),
    }
}

/// The `db` of the fit of `kind` in a layer's seven fits.
fn fit_db(fits: &[TensorFit], kind: TensorKind) -> Option<&[i64]> {
    fits.iter()
        .find(|f| f.kind == kind)
        .and_then(|f| f.db.as_deref())
}

/// The finisher's [`Tunables`] from the student's layers: `ln1`/`ln2` from `attn_norm`/
/// `ffn_norm`, and the `q`/`k`/`v` bias bases from the student's fitted bias (`Lt.b + db` of
/// [`crate::learn::fold::layer::driver::apply_fit`]), falling back to the fit's `db` and then to
/// zeros; `fn_base = fnorm`.
///
/// # Errors
/// [`FinishRunError::Shape`] on a layer/fit count or a width disagreeing with `n_embd` and the
/// weight shapes.
///
/// # Complexity
/// Temporal `O(n_layer·(n_embd + q_out + kv_out))`, one copy per parameter; spatial the same.
pub fn tunables_from_weights(
    layers: &[LayerWeights],
    fnorm: &[i64],
    fits: &[Vec<TensorFit>],
) -> Result<Tunables, FinishRunError> {
    if layers.len() != fits.len() {
        return Err(FinishRunError::Shape(format!(
            "layers {} != fits {}",
            layers.len(),
            fits.len()
        )));
    }
    let mut out: Vec<LayerTunables> = Vec::with_capacity(layers.len());
    for (l, lw) in layers.iter().enumerate() {
        let fs = &fits[l];
        if fs.len() != N_KINDS {
            return Err(FinishRunError::Shape(format!(
                "layer {l}: {} fits != {N_KINDS}",
                fs.len()
            )));
        }
        let e = lw.attn_norm.len();
        if e == 0
            || lw.ffn_norm.len() != e
            || !lw.wq.len().is_multiple_of(e)
            || !lw.wk.len().is_multiple_of(e)
        {
            return Err(FinishRunError::Shape(format!(
                "layer {l}: norm or projection width"
            )));
        }
        let q_out = lw.wq.len() / e;
        let kv_out = lw.wk.len() / e;
        let qb = bias_base(lw.bq.as_deref(), fit_db(fs, TensorKind::Q), q_out, "qb")?;
        let kb = bias_base(lw.bk.as_deref(), fit_db(fs, TensorKind::K), kv_out, "kb")?;
        let vb = bias_base(lw.bv.as_deref(), fit_db(fs, TensorKind::V), kv_out, "vb")?;
        out.push(LayerTunables::new(
            &lw.attn_norm,
            &lw.ffn_norm,
            &qb,
            &kb,
            &vb,
        ));
    }
    Ok(Tunables::new(out, fnorm))
}

/// The finished student: the assembled Q32.32 [`Weights`] (the same shape
/// [`crate::learn::fold::run::convert::student_to_grad`] produces, so
/// [`crate::learn::fold::run::transfer::write_student_gguf`] consumes it), the settled norms and
/// biases, the final `output_norm` gain and the pre-update mean KL of each step.
#[derive(Clone, Debug)]
pub struct Finished {
    /// The assembled stored weights, norms and biases, ready for the `GGUF` write.
    pub weights: Weights,
    /// The settled per-layer norms and biases.
    pub tuned: Vec<LayerTuned>,
    /// The settled final `RMSNorm` gain.
    pub output_norm: Vec<i64>,
    /// The batch mean KL measured before each Adam update, Q32.32 words.
    pub loss: Vec<i64>,
}

/// Build the finisher from a [`ModelResult`] and run `steps` of [`Finisher::step`] over the
/// calibration chunks, cycling them, against the teacher's own logits; the teacher's embedding and
/// head are the frozen tensors.
///
/// # Errors
/// [`FinishRunError::Shape`] on an empty chunk list, an empty chunk or a fit/geometry disagreement;
/// [`FinishRunError::Model`]/[`FinishRunError::Train`] from the teacher forward or a step.
///
/// # Complexity
/// Temporal `O(steps·(cost(forward(teacher, chunk)) + Finisher::step))` plus one
/// `O(Σ o·n·CODE_LEN)` plane build and one `O(Σ o·n·log2(GROUP))` assemble; spatial
/// `O(t·vocab + Σ o·n)`.
pub fn finish_model(
    cfg: &Config,
    res: &ModelResult,
    teacher: &Weights,
    tokens: &[Vec<u32>],
    steps: u64,
    warmup: u64,
) -> Result<Finished, FinishRunError> {
    if tokens.is_empty() || tokens.iter().any(Vec::is_empty) {
        return Err(FinishRunError::Shape(
            "empty calibration chunk list".to_string(),
        ));
    }
    let planes = planes_from_fits(cfg, &res.fits)?;
    let tun = tunables_from_weights(&res.layers, &res.fnorm, &res.fits)?;
    let frozen = Frozen::new(&teacher.token_embd, &teacher.lm_head);
    let schedule = Schedule::new(warmup, steps.max(1));
    let mut fin = Finisher::new(cfg.clone(), planes, tun, frozen, schedule)?;
    let len =
        usize::try_from(steps).map_err(|_| FinishRunError::Shape("step count".to_string()))?;
    let mut loss: Vec<i64> = Vec::with_capacity(len);
    for i in 0..len {
        let chunk = &tokens[i % tokens.len()];
        let (logits, _tape) = forward(teacher, cfg, chunk)?;
        loss.push(fin.step(chunk, &logits)?);
    }
    let weights = assemble(cfg, &fin.planes, &fin.tun, &fin.frozen)?;
    let tuned = fin.tun.layers.iter().map(LayerTunables::tuned).collect();
    let output_norm = fin.tun.output_norm();
    Ok(Finished {
        weights,
        tuned,
        output_norm,
        loss,
    })
}
