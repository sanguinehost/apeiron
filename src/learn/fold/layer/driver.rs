//! The per-layer driver of the production `fold_gqa` fold transfer: the four stage folds, the
//! student-path propagation and the model loop.
//!
//! Port of the per-layer loop and final-norm tail of `run_variant`
//! (`~/lanes/port/spec/fold_transfer.py:1479-1690`, `:1692-2040`). The production
//! arm is the only one present: `fold_gqa`, the student-input path (`teacher_in = false`), `block`,
//! `sink_gate`, `split_intra_inter`, `use_grow`, `q_mlp_trans`, `nb_arm`, `use_corr`, `use_resq`,
//! `qronos_metric`, `bscale`, `e8_icm`; the `guided`, `yaqa*`, `drift_filter`, `rrr_mode`,
//! `zellner`, `glu_comp`, `use_align`, `asym`, `skew`, `teacher_in`, `low_vram` and `PACK_SINK`
//! branches are off and absent (no parameters, no stubs).
//!
//! One layer runs `qkv`, `o`, `ffn`, `down` (`STAGES`, `:49`): each accumulates the sink-weighted
//! normal equations over the slabs ([`StageAccum`]), solves them ([`solve_stage`]) and fits every
//! tensor ([`fit_tensor`]), writing `Lq[nm] = Wp` into the running student layer (`:1954`);
//! [`sink_token_weights`] takes its median over one slab's positions (`:1592-1597`).
//!
//! # The carrier
//!
//! Every computed value is a Q32.32 `i64` word, rounded once per step
//! ([`crate::learn::solve::q32`]); leaving the range is [`DriverError::Grid`]. The `f64` arguments,
//! returns and [`LayerWeights`] fields are the boundary, converted once per call; a kernel is one
//! serial pass, and a stage's tensors are fitted serially because the pool dispatcher is not
//! reentrant.
//!
//! # Grade
//!
//! `tested`: `tests/learn_fold_driver.rs`. No extraction or refinement links this file to a Lean
//! theorem, so nothing here is `proved`.
//!
//! # Trust boundary
//!
//! `layer::stage`, `layer::tensor`, `layer::resq`, `stats::parts` and `fit::fnorm` are other
//! modules' contracts, taken as given. Not covered: rustc/LLVM codegen, the reference's fp32
//! accumulation, and the model loader and calibration set behind the inputs.

#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::implicit_clone,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the driver is the ported `run_variant` body: it indexes row-major `[rows, width]` \
              buffers by `r/i/j/k/o`, names the tensors `q/k/v/o/gate/up/down` and the carriers \
              `lt`/`lq`/`hs`/`hs_mid`/`ht` as the reference does, casts dimension counts \
              `usize → f64`, and its stage functions take the reference's argument set; the public \
              types repeat `driver::` by the brief's naming"
)]

use crate::learn::fold::fit::fnorm::{fnorm_fit, FnormError};
use crate::learn::fold::fit::fold_fit::FoldFitConfig;
use crate::learn::fold::layer::resq::{resq_scale, ResqError, ResqInputs};
use crate::learn::fold::layer::stage::{
    sink_token_weights, solve_stage, SolveParams, StageAccum, StageError,
};
use crate::learn::fold::layer::tensor::carrier::{
    col_squares, eval, grids, ratio, ratio_words, reals, silu_grid, sum_i128, Elem,
};
use crate::learn::fold::layer::tensor::{
    fit_tensor, fold_row_scale, BiasStats, TensorBits, TensorError, TensorFit, TensorKind,
    TensorParams,
};
use crate::learn::fold::stats::hocapture::Kind;
use crate::learn::fold::stats::parts::{
    attn_a, linear, parts, rms, LayerCfg, LayerF64, PartsError, RopeTables,
};
use crate::learn::solve::q32::Q32Error;

/// A refusal from the layer driver.
#[derive(Clone, Debug, PartialEq)]
pub enum DriverError {
    /// A slice length disagrees with the declared geometry, a width is not a positive multiple of
    /// `128`, or a required teacher/student tensor is missing.
    Shape,
    /// `slab == 0`, `n_seq == 0`, or an invalid head/intermediate geometry.
    Geometry,
    /// `t < 2`: a slab of one token per sequence has no median population for the sink weights.
    TinySeq,
    /// An input entry (a weight, a hidden state or the `ln_next` source) is not finite.
    NonFinite,
    /// A value reaches the Q32.32 grid bound `|x| < 2³¹`, or a checked `i128`/`i64` operation
    /// leaves its range.
    Grid,
    /// The layer forward refused (`stats::parts`).
    Parts(PartsError),
    /// The stage statistics or solve refused (`layer::stage`).
    Stage(StageError),
    /// The per-tensor assembly refused (`layer::tensor`).
    Tensor(TensorError),
    /// The `-resq` row-scale step refused (`layer::resq`).
    Resq(ResqError),
    /// The final RMSNorm-gain fit refused (`fit::fnorm`).
    Fnorm(FnormError),
}

impl From<PartsError> for DriverError {
    fn from(e: PartsError) -> Self {
        Self::Parts(e)
    }
}

impl From<StageError> for DriverError {
    fn from(e: StageError) -> Self {
        Self::Stage(e)
    }
}

impl From<TensorError> for DriverError {
    fn from(e: TensorError) -> Self {
        Self::Tensor(e)
    }
}

impl From<ResqError> for DriverError {
    fn from(e: ResqError) -> Self {
        Self::Resq(e)
    }
}

impl From<FnormError> for DriverError {
    fn from(e: FnormError) -> Self {
        Self::Fnorm(e)
    }
}

impl From<Q32Error> for DriverError {
    fn from(e: Q32Error) -> Self {
        match e {
            Q32Error::ShapeMismatch => Self::Shape,
            _ => Self::Grid,
        }
    }
}

impl core::fmt::Display for DriverError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Shape => write!(f, "a slice length disagrees with the declared geometry"),
            Self::Geometry => write!(f, "invalid slab, sequence count or layer geometry"),
            Self::TinySeq => write!(f, "t < 2: no sink median population"),
            Self::NonFinite => write!(f, "an input entry is not finite"),
            Self::Grid => write!(f, "a value or operation left the Q32.32 carrier's range"),
            Self::Parts(e) => write!(f, "the layer forward refused: {e:?}"),
            Self::Stage(e) => write!(f, "the stage statistics refused: {e:?}"),
            Self::Tensor(e) => write!(f, "the per-tensor assembly refused: {e}"),
            Self::Resq(e) => write!(f, "the row-scale step refused: {e:?}"),
            Self::Fnorm(e) => write!(f, "the final-norm fit refused: {e:?}"),
        }
    }
}

impl std::error::Error for DriverError {}

/// The twelve owned `f64` weight tensors of one `Qwen2` decoder layer, each row-major `[out, in]`
/// (the Python transfer-layer dictionary `TL[l]`, `:117-122`).
#[derive(Clone, Debug, PartialEq)]
pub struct LayerWeights {
    /// `input_layernorm.weight`, length `hidden`.
    pub ln1: Vec<f64>,
    /// `post_attention_layernorm.weight`, length `hidden`.
    pub ln2: Vec<f64>,
    /// `self_attn.q_proj.weight`, row-major `[n_head·head_dim, hidden]`.
    pub q: Vec<f64>,
    /// `self_attn.q_proj.bias`, length `n_head·head_dim`.
    pub qb: Vec<f64>,
    /// `self_attn.k_proj.weight`, row-major `[n_kv_head·head_dim, hidden]`.
    pub k: Vec<f64>,
    /// `self_attn.k_proj.bias`, length `n_kv_head·head_dim`.
    pub kb: Vec<f64>,
    /// `self_attn.v_proj.weight`, row-major `[n_kv_head·head_dim, hidden]`.
    pub v: Vec<f64>,
    /// `self_attn.v_proj.bias`, length `n_kv_head·head_dim`.
    pub vb: Vec<f64>,
    /// `self_attn.o_proj.weight`, row-major `[hidden, n_head·head_dim]`.
    pub o: Vec<f64>,
    /// `mlp.gate_proj.weight`, row-major `[inter, hidden]`.
    pub gate: Vec<f64>,
    /// `mlp.up_proj.weight`, row-major `[inter, hidden]`.
    pub up: Vec<f64>,
    /// `mlp.down_proj.weight`, row-major `[hidden, inter]`.
    pub down: Vec<f64>,
}

impl LayerWeights {
    /// A borrowed [`LayerF64`] view of this layer, for the forward and the helper signatures that
    /// take `&[f64]`. `O(1)`.
    #[must_use]
    pub fn as_f64(&self) -> LayerF64<'_> {
        LayerF64 {
            ln1: &self.ln1,
            ln2: &self.ln2,
            q: &self.q,
            qb: &self.qb,
            k: &self.k,
            kb: &self.kb,
            v: &self.v,
            vb: &self.vb,
            o: &self.o,
            gate: &self.gate,
            up: &self.up,
            down: &self.down,
        }
    }
}

/// The driver's geometry, hyper-parameters and per-stage fit configuration.
///
/// For a stage of input width `n`, `rg_q = quant_ridge·(n/896)` and `damp = quant_damp·(n/896)`
/// (`fold_transfer.py:1692-1693`). `fit` carries the fixed flags (`inner_iters` 3, `e8_beam` 4,
/// `icm_ridge` 0.20, `iters` 1, `sweeps` 2); its `ridge`/`damp` are overwritten per stage.
#[derive(Clone, Copy, Debug)]
pub struct DriverCfg {
    /// The layer geometry.
    pub layer: LayerCfg,
    /// Tokens per sequence (`T`).
    pub t: usize,
    /// Sequences in the calibration batch.
    pub n_seq: usize,
    /// Sequences per slab (`a.slab`); the last slab may be short.
    pub slab: usize,
    /// `a.quant_ridge = 0.005` (`pv_finish.py:79`, `qronos_metric`).
    pub quant_ridge: f64,
    /// `a.quant_damp = 0.02` (`pv_finish.py:79`, `qronos_metric`).
    pub quant_damp: f64,
    /// `a.sink_thresh = 5.0` (`pv_finish.py:79`): the sink-weight threshold.
    pub sink_thresh: f64,
    /// `a.qmlp_rows = 416` (`pv_finish.py:78`): rows of the `gate`/`up` second plane.
    pub qmlp_rows: usize,
    /// The `fold_gqa` solve hyper-parameters (`SolveParams::default`).
    pub solve: SolveParams,
    /// The `fold_fit` configuration (fixed production flags; `ridge`/`damp` overwritten per stage).
    pub fit: FoldFitConfig,
    /// `NB_SWEEPS = 2`: sweeps of each `-nb` joint ICM pass.
    pub nb_sweeps: usize,
    /// `nb_refine`'s `rounds = 2`.
    pub nb_rounds: usize,
}

impl DriverCfg {
    /// The production arm's defaults for a layer geometry and calibration batch
    /// (`pv_finish.py:75-84`, `arm_args`): `slab` 10, `quant_ridge` 0.005, `quant_damp` 0.02,
    /// `sink_thresh` 5.0, `qmlp_rows` 416, [`SolveParams::default`], the fixed `fold_fit` flags,
    /// `nb_sweeps`/`nb_rounds` 2. `O(1)`.
    #[must_use]
    pub fn production(layer: LayerCfg, t: usize, n_seq: usize) -> Self {
        Self {
            layer,
            t,
            n_seq,
            slab: 10,
            quant_ridge: 0.005,
            quant_damp: 0.02,
            sink_thresh: 5.0,
            qmlp_rows: 416,
            solve: SolveParams::default(),
            fit: FoldFitConfig {
                iters: 1,
                ridge: 0.005,
                damp: 0.02,
                inner_iters: 3,
                e8_beam: 4,
                icm_ridge: 0.20,
                sweeps: 2,
            },
            nb_sweeps: 2,
            nb_rounds: 2,
        }
    }
}

/// The quantised layer and the evidence of one [`quantise_layer`] call.
#[derive(Clone, Debug)]
pub struct LayerResult {
    /// The quantised student layer (the fitted `Lq`).
    pub lq: LayerWeights,
    /// The seven tensor fits in `q, k, v, o, gate, up, down` order.
    pub fits: Vec<TensorFit>,
    /// The bit budget summed over the seven fits.
    pub bits: TensorBits,
    /// The teacher layer output (`Pt.out` concatenated over slabs), row-major `[n_seq·t, hidden]`.
    pub ht_next: Vec<f64>,
}

/// The resumable state of a model-level transfer run: the layers quantised so far, the teacher and
/// student hidden states they leave, the bit sum and the per-layer fits.
///
/// [`quantise_model`] is [`model_init`] + [`model_step`] + the untouched-layer copy +
/// [`model_finish`]; a checkpointed run is the same calls split across invocations. The checkpoint
/// ([`crate::learn::fold::run::transfer::save`]) writes the weights, the hidden states and `bits`,
/// not `fits`, so a loaded state carries one empty fit vector per finished layer.
#[derive(Clone, Debug)]
pub struct ModelState {
    /// Index of the next layer to quantise (`0 ..= layers.len()`).
    pub next: usize,
    /// The teacher hidden state entering layer `next` (the last quantised layer's `ht_next`),
    /// row-major `[n_seq·t, hidden]`.
    pub ht: Vec<f64>,
    /// The student hidden state entering layer `next`, row-major `[n_seq·t, hidden]`.
    pub hs: Vec<f64>,
    /// The quantised layers `0 .. next`, in order.
    pub layers: Vec<LayerWeights>,
    /// The bit budget summed over the quantised layers.
    pub bits: TensorBits,
    /// The tensor fits of each quantised layer (`fits[l]` for layer `l < next`), kept so that
    /// [`model_finish`] returns the same [`ModelResult`] as a one-shot run.
    pub fits: Vec<Vec<TensorFit>>,
}

/// The quantised model and the evidence of one [`quantise_model`] call.
#[derive(Clone, Debug)]
pub struct ModelResult {
    /// One entry per input layer; layers `>= n_layers` are copied unchanged.
    pub layers: Vec<LayerWeights>,
    /// The fitted final RMSNorm gain, length `hidden`.
    pub fnorm: Vec<f64>,
    /// The bit budget summed over the quantised layers.
    pub bits: TensorBits,
    /// The tensor fits of each layer (empty for a copied layer).
    pub fits: Vec<Vec<TensorFit>>,
}

/// One of the four stages of a layer (`STAGES`, `fold_transfer.py:49`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StageId {
    /// `q`, `k` and `v`.
    Qkv,
    /// `o`.
    O,
    /// `gate` and `up`.
    Ffn,
    /// `down`.
    Down,
}

impl StageId {
    /// The stage's tensors, in fit order.
    fn tensor_kinds(self) -> &'static [TensorKind] {
        match self {
            Self::Qkv => &[TensorKind::Q, TensorKind::K, TensorKind::V],
            Self::O => &[TensorKind::O],
            Self::Ffn => &[TensorKind::Gate, TensorKind::Up],
            Self::Down => &[TensorKind::Down],
        }
    }

    /// The stage's solve kinds (the `split_intra_inter`/`use_grow` dispatch keys).
    fn solve_kinds(self) -> &'static [Kind] {
        match self {
            Self::Qkv => &[Kind::Q, Kind::K, Kind::V],
            Self::O => &[Kind::O],
            Self::Ffn => &[Kind::Gate, Kind::Up],
            Self::Down => &[Kind::Down],
        }
    }

    /// The stage input width `n` (the first tensor's input width, `:1539`).
    fn input_width(self, cfg: &DriverCfg) -> usize {
        match self {
            Self::Qkv | Self::Ffn => cfg.layer.hidden,
            Self::O => q_out(cfg),
            Self::Down => cfg.layer.inter,
        }
    }

    /// The stage output widths, in fit order.
    fn outs(self, cfg: &DriverCfg) -> Vec<usize> {
        match self {
            Self::Qkv => vec![q_out(cfg), kv_out(cfg), kv_out(cfg)],
            Self::O | Self::Down => vec![cfg.layer.hidden],
            Self::Ffn => vec![cfg.layer.inter, cfg.layer.inter],
        }
    }

    /// Whether each output carries a `Cts` accumulator (`split_intra_inter` on `o`/`down`).
    fn with_ts(self) -> &'static [bool] {
        match self {
            Self::Qkv => &[false, false, false],
            Self::O | Self::Down => &[true],
            Self::Ffn => &[false, false],
        }
    }

    /// The teacher's stage tensors, the solve's `w0` (`:1658`).
    fn teacher_tensors(self, lt: &LayerWeights) -> Vec<&[f64]> {
        match self {
            Self::Qkv => vec![&lt.q, &lt.k, &lt.v],
            Self::O => vec![&lt.o],
            Self::Ffn => vec![&lt.gate, &lt.up],
            Self::Down => vec![&lt.down],
        }
    }
}

/// The stage output: the six [`TensorFit`] results and the per-slab stage inputs.
///
/// `slabs` is the `o` stage's attention output, the `ffn` stage's raw `Pin_in` (for the `-resq`
/// solve) and the `down` stage's `Pin_in`; it is empty for `qkv`.
#[derive(Clone, Debug)]
struct StageOut {
    /// The stage's tensor fits, in fit order.
    fits: Vec<TensorFit>,
    /// The stage inputs, one flattened `[rows, n]` buffer per slab.
    slabs: Vec<Vec<f64>>,
}

/// The teacher intermediates one slab of the forward keeps (`keep_keys`, `:1526`): the fields the
/// stage targets read. Row-major `[rows, width]`.
#[derive(Clone, Debug)]
struct SlabParts {
    /// `xa·qᵀ + qb`, `[rows, q_out]`.
    q: Vec<f64>,
    /// `xa·kᵀ + kb`, `[rows, kv_out]`.
    k: Vec<f64>,
    /// `xa·vᵀ + vb`, `[rows, kv_out]`.
    v: Vec<f64>,
    /// `h + a·oᵀ`, `[rows, hidden]`.
    mid: Vec<f64>,
    /// `xf·gateᵀ`, `[rows, inter]`.
    g: Vec<f64>,
    /// `xf·upᵀ`, `[rows, inter]`.
    u: Vec<f64>,
    /// the layer output, `[rows, hidden]`.
    out: Vec<f64>,
}

/// `n_head · head_dim`, the query/attention output width. `O(1)`.
fn q_out(cfg: &DriverCfg) -> usize {
    cfg.layer.n_head * cfg.layer.head_dim
}

/// `n_kv_head · head_dim`, the key/value output width. `O(1)`.
fn kv_out(cfg: &DriverCfg) -> usize {
    cfg.layer.n_kv_head * cfg.layer.head_dim
}

/// The slab partition of `n_seq` sequences: `(row0, rows, seqs)` per slab, in order. The last slab
/// may be short (`:1552`). `O(n_seq/slab)` time and space.
///
/// # Errors
/// [`DriverError::Geometry`] if `slab == 0` or `n_seq == 0`.
fn slab_geometry(
    n_seq: usize,
    t: usize,
    slab: usize,
) -> Result<Vec<(usize, usize, usize)>, DriverError> {
    if slab == 0 || n_seq == 0 {
        return Err(DriverError::Geometry);
    }
    let mut v = Vec::new();
    let mut s0 = 0;
    while s0 < n_seq {
        let seqs = slab.min(n_seq - s0);
        v.push((s0 * t, seqs * t, seqs));
        s0 += seqs;
    }
    Ok(v)
}

/// `true` iff every entry is finite.
fn finite(a: &[f64]) -> bool {
    a.iter().all(|v| v.is_finite())
}

/// The sum of two [`TensorBits`].
fn add_bits(a: TensorBits, b: &TensorBits) -> TensorBits {
    TensorBits {
        nsign: a.nsign + b.nsign,
        nr: a.nr + b.nr,
        nc: a.nc + b.nc,
        nd: a.nd + b.nd,
        ncorr: a.ncorr + b.ncorr,
    }
}

/// `g_row = sq / mean(sq)` (`fold_transfer.py:1706`, `:1710`): the per-row weight normalised to mean
/// `1`, floored at one grid quantum so an all-zero input gives zeros, never a NaN.
///
/// # Errors
/// [`DriverError::Shape`] if `v` is empty; [`DriverError::NonFinite`] on a non-finite entry;
/// [`DriverError::Grid`] if an entry or an operation leaves the carrier's range. Temporal `O(len)`
/// (one square pass, one `i128` sum, one divide, one ratio pass); spatial `O(len)`.
pub fn g_row_from_vec(v: &[f64]) -> Result<Vec<f64>, DriverError> {
    if v.is_empty() {
        return Err(DriverError::Shape);
    }
    if !finite(v) {
        return Err(DriverError::NonFinite);
    }
    let q = grids(v)?;
    let sq = eval(q.len(), Elem::Square(&q))?;
    let len = i128::try_from(q.len()).map_err(|_| DriverError::Shape)?;
    // `len` as a word, so the ratio is `Σ sq / len` (a mean) rather than `Σ sq / len·2³²`.
    let mean = ratio(sum_i128(&sq)?, len << 32)?;
    Ok(reals(&eval(sq.len(), Elem::Ratio(&sq, mean.max(1)))?))
}

/// `g_row` for the `gate`/`up` stage: the column sums of squares of the teacher `down`
/// (`(Lt["down"] ** 2).sum(0)`, `:1705`), normalised to mean `1`. `w` is row-major `[rows, cols]`;
/// the result has length `cols`, the column sums exact in `i128`.
///
/// # Errors
/// [`DriverError::Shape`] if `w.len() != rows·cols` or `cols == 0`; [`DriverError::NonFinite`] on a
/// non-finite entry; [`DriverError::Grid`] if an entry, a sum or a division leaves the range.
/// Temporal `O(rows·cols)`; spatial `O(cols)`.
pub fn g_row_from_matrix(w: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>, DriverError> {
    let rc = rows.checked_mul(cols).ok_or(DriverError::Shape)?;
    if cols == 0 || w.len() != rc {
        return Err(DriverError::Shape);
    }
    if !finite(w) {
        return Err(DriverError::NonFinite);
    }
    let wq = grids(w)?;
    let sums = col_squares(&wq, rows, cols)?;
    let mut total = 0_i128;
    for &s in &sums {
        total = total.checked_add(s).ok_or(DriverError::Grid)?;
    }
    let den = i128::try_from(cols).map_err(|_| DriverError::Grid)? << 32;
    Ok(reals(&ratio_words(&sums, ratio(total, den)?.max(1))?))
}

/// The batched teacher forward: [`parts`] per sequence of the slab, concatenated (`:1533`). `h` is
/// row-major `[seqs·t, hidden]`; the result fields are row-major `[seqs·t, width]`. Temporal
/// `O(seqs·t·(n_head·head_dim·hidden + inter·hidden) + seqs·t²·n_head·head_dim)`, one [`parts`] per
/// sequence; spatial `O(seqs·t·(hidden + head widths + inter))`.
///
/// # Errors
/// As [`parts`], wrapped: [`DriverError::Shape`] on a slice or geometry mismatch, and the forward's
/// refusals.
fn batch_parts(
    cfg: &DriverCfg,
    rope: &RopeTables,
    lt: &LayerWeights,
    h: &[f64],
    seqs: usize,
    t: usize,
) -> Result<SlabParts, DriverError> {
    let hid = cfg.layer.hidden;
    let qo = q_out(cfg);
    let kvo = kv_out(cfg);
    let inter = cfg.layer.inter;
    let rows = seqs * t;
    if h.len() != rows * hid {
        return Err(DriverError::Shape);
    }
    let lt_f = lt.as_f64();
    let mut out = SlabParts {
        q: Vec::with_capacity(rows * qo),
        k: Vec::with_capacity(rows * kvo),
        v: Vec::with_capacity(rows * kvo),
        mid: Vec::with_capacity(rows * hid),
        g: Vec::with_capacity(rows * inter),
        u: Vec::with_capacity(rows * inter),
        out: Vec::with_capacity(rows * hid),
    };
    // The pool dispatch is inside `parts` (its GEMMs); a sequence here is one serial call.
    for s in 0..seqs {
        let hb = &h[s * t * hid..(s + 1) * t * hid];
        let p = parts(&cfg.layer, rope, &lt_f, hb, t)?;
        out.q.extend_from_slice(&p.q);
        out.k.extend_from_slice(&p.k);
        out.v.extend_from_slice(&p.v);
        out.mid.extend_from_slice(&p.mid);
        out.g.extend_from_slice(&p.g);
        out.u.extend_from_slice(&p.u);
        out.out.extend_from_slice(&p.out);
    }
    Ok(out)
}

/// The batched attention: [`attn_a`] per sequence of the slab, concatenated (`:1567`, `:1613`).
/// `q` is `[seqs·t, q_out]`, `k`/`v` `[seqs·t, kv_out]`; the result is `[seqs·t, q_out]`. Temporal
/// `O(seqs·t²·n_head·head_dim)`; spatial `O(seqs·t·q_out)`.
///
/// # Errors
/// As [`attn_a`], wrapped (a shape or `RoPE`-length refusal).
fn attn_batch(
    cfg: &DriverCfg,
    rope: &RopeTables,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seqs: usize,
    t: usize,
) -> Result<Vec<f64>, DriverError> {
    let qo = q_out(cfg);
    let kvo = kv_out(cfg);
    if q.len() != seqs * t * qo || k.len() != seqs * t * kvo || v.len() != seqs * t * kvo {
        return Err(DriverError::Shape);
    }
    let mut out = Vec::with_capacity(seqs * t * qo);
    // Serial per sequence: the pool work is inside `attn_a`'s GEMMs.
    for s in 0..seqs {
        let (lo, hi) = (s * t, (s + 1) * t);
        out.extend_from_slice(&attn_a(
            &cfg.layer,
            rope,
            &q[lo * qo..hi * qo],
            &k[lo * kvo..hi * kvo],
            &v[lo * kvo..hi * kvo],
            t,
        )?);
    }
    Ok(out)
}

/// `y[r, j] − b[j]` broadcast over the rows: the `q`/`k`/`v` target `Pt[nm] − Lt[nm + "b"]`
/// (`:1630`).
/// # Errors
/// [`DriverError::Shape`] if a length disagrees with `rows`/`w`; [`DriverError::Grid`] from a
/// conversion or a checked subtraction. Temporal `O(rows·w)`, spatial `O(rows·w)`.
fn sub_bias(y: &[f64], b: &[f64], rows: usize, w: usize) -> Result<Vec<f64>, DriverError> {
    let rw = rows.checked_mul(w).ok_or(DriverError::Shape)?;
    if w == 0 || y.len() != rw || b.len() != w {
        return Err(DriverError::Shape);
    }
    let e = Elem::SubBias(&grids(y)?, &grids(b)?, w);
    Ok(reals(&eval(rw, e)?))
}

/// `(a − b) + (b − c)` on the carrier: the `o` stage's `(Pt.mid − Ht) + (Ht − Hs)` and the `down`
/// stage's `(Pt.out − Pt.mid) + (Pt.mid − mid)` residual (`:1630`).
/// # Errors
/// [`DriverError::Shape`] on a length mismatch; [`DriverError::Grid`] from a conversion or a checked
/// step. Temporal `O(len)`, spatial `O(len)`.
fn tri_sub(a: &[f64], b: &[f64], c: &[f64]) -> Result<Vec<f64>, DriverError> {
    if a.len() != b.len() || a.len() != c.len() {
        return Err(DriverError::Shape);
    }
    let e = Elem::TriSub(&grids(a)?, &grids(b)?, &grids(c)?);
    Ok(reals(&eval(a.len(), e)?))
}

/// `silu(g)·u` elementwise on the carrier: the `down` stage's gate/up product (`:1904`).
/// [`crate::learn::fold::stats::parts::silu`] is evaluated here, then rounded once.
/// # Errors
/// [`DriverError::Shape`] on a length mismatch; [`DriverError::Grid`] from a conversion or the
/// product. Temporal `O(len)`, spatial `O(len)`.
fn silu_mul(g: &[f64], u: &[f64]) -> Result<Vec<f64>, DriverError> {
    if g.len() != u.len() {
        return Err(DriverError::Shape);
    }
    let (sq, uq) = (silu_grid(g)?, grids(u)?);
    Ok(reals(&eval(g.len(), Elem::Mul(&sq, &uq))?))
}

/// `dst = base + add` on the carrier: the student-path accumulations `hs_mid = hs + a·Lq.oᵀ` and
/// `hs = hs_mid + m·Lq.downᵀ` (`:1960-1967`), and the bias write `Lq[nm + "b"] = Lt[nm + "b"] + db`
/// (`:1879-1881`).
/// # Errors
/// [`DriverError::Shape`] on a length mismatch; [`DriverError::Grid`] from a conversion or the
/// checked sum. Temporal `O(dst.len())`, spatial `O(dst.len())`.
fn add_into(dst: &mut [f64], base: &[f64], add: &[f64]) -> Result<(), DriverError> {
    if dst.len() != base.len() || dst.len() != add.len() {
        return Err(DriverError::Shape);
    }
    let (bq, aq) = (grids(base)?, grids(add)?);
    dst.copy_from_slice(&reals(&eval(dst.len(), Elem::Add(&bq, &aq))?));
    Ok(())
}

/// Write the fitted tensor into `lq`: `Lq[nm] = Wp`, and for `q`/`k`/`v` the bias
/// `Lq[nm + "b"] = Lt[nm + "b"] + db` (`:1879-1881`, `:1954`).
///
/// # Errors
/// [`DriverError::Shape`] if a bias is missing or a length disagrees.
fn apply_fit(
    lq: &mut LayerWeights,
    kind: TensorKind,
    lt: &LayerWeights,
    fit: &TensorFit,
) -> Result<(), DriverError> {
    match kind {
        TensorKind::Q => {
            lq.q.clone_from(&fit.wp);
            set_bias(&mut lq.qb, &lt.qb, fit.db.as_deref())?;
        }
        TensorKind::K => {
            lq.k.clone_from(&fit.wp);
            set_bias(&mut lq.kb, &lt.kb, fit.db.as_deref())?;
        }
        TensorKind::V => {
            lq.v.clone_from(&fit.wp);
            set_bias(&mut lq.vb, &lt.vb, fit.db.as_deref())?;
        }
        TensorKind::O => lq.o.clone_from(&fit.wp),
        TensorKind::Gate => lq.gate.clone_from(&fit.wp),
        TensorKind::Up => lq.up.clone_from(&fit.wp),
        TensorKind::Down => lq.down.clone_from(&fit.wp),
    }
    Ok(())
}

/// `dst[i] = base[i] + db[i]` on the carrier: the `Lq[nm + "b"]` write (`:1879-1881`).
///
/// # Errors
/// [`DriverError::Shape`] if a bias is missing or a length disagrees; else as [`add_into`].
fn set_bias(dst: &mut [f64], base: &[f64], db: Option<&[f64]>) -> Result<(), DriverError> {
    add_into(dst, base, db.ok_or(DriverError::Shape)?)
}

/// One stage of a layer: the slab Gram accumulation (`:1538-1648`), the `fold_gqa` solve
/// (`:1652-1682`) and the per-tensor fit (`:1692-1957`), writing each fit into `lq`. `hs_mid` feeds
/// the `ffn`/`down` stage input; the sink weights of every stage read `hs`. Temporal
/// `O(Σ_slabs rows·n² + Σ_j n³)`, dominated by the seven [`fit_tensor`] calls; spatial
/// `O(n² + Σ_j n·outs_j + Σ_slabs rows·n)`.
/// # Errors
/// [`DriverError::Shape`]/[`DriverError::NonFinite`] on a length or finiteness mismatch, and every
/// refusal of [`sink_token_weights`], [`StageAccum::add_slab`], [`solve_stage`], [`fit_tensor`],
/// [`resq_scale`] and [`fold_row_scale`].
fn run_stage(
    cfg: &DriverCfg,
    rope: &RopeTables,
    lt: &LayerWeights,
    lq: &mut LayerWeights,
    ln_next: &[f64],
    stage: StageId,
    pt_cache: &[SlabParts],
    ht: &[f64],
    hs: &[f64],
    hs_mid: &[f64],
) -> Result<StageOut, DriverError> {
    let hid = cfg.layer.hidden;
    let inter = cfg.layer.inter;
    let t = cfg.t;
    let eps = cfg.layer.eps;
    let qo = q_out(cfg);
    let kvo = kv_out(cfg);
    let n = stage.input_width(cfg);
    let outs = stage.outs(cfg);
    let kinds = stage.tensor_kinds();
    let with_ts = stage.with_ts();
    let geom = slab_geometry(cfg.n_seq, t, cfg.slab)?;
    let mut accum = StageAccum::new(n, &outs, with_ts)?;
    let mut slab_in: Vec<Vec<f64>> = Vec::with_capacity(geom.len());

    for (i, &(row0, rows, seqs)) in geom.iter().enumerate() {
        let pt = &pt_cache[i];
        let hts = &ht[row0 * hid..row0 * hid + rows * hid];
        let hss = &hs[row0 * hid..row0 * hid + rows * hid];
        let mut pin_mid: Option<Vec<f64>> = None;
        let mut pin_xa: Option<Vec<f64>> = None;
        let mut pin_xf: Option<Vec<f64>> = None;

        let pin_in: Vec<f64> = match stage {
            StageId::Qkv => rms(hss, &lq.ln1, eps, rows, hid)?,
            StageId::O => {
                let xa = rms(hss, &lq.ln1, eps, rows, hid)?;
                let q = linear(&xa, rows, hid, &lq.q, Some(&lq.qb), qo)?;
                let k = linear(&xa, rows, hid, &lq.k, Some(&lq.kb), kvo)?;
                let v = linear(&xa, rows, hid, &lq.v, Some(&lq.vb), kvo)?;
                let a = attn_batch(cfg, rope, &q, &k, &v, seqs, t)?;
                pin_xa = Some(xa);
                a
            }
            StageId::Ffn => {
                let mid = hs_mid[row0 * hid..row0 * hid + rows * hid].to_vec();
                let x = rms(&mid, &lq.ln2, eps, rows, hid)?;
                pin_mid = Some(mid);
                x
            }
            StageId::Down => {
                let mid = hs_mid[row0 * hid..row0 * hid + rows * hid].to_vec();
                let xf = rms(&mid, &lq.ln2, eps, rows, hid)?;
                let g = linear(&xf, rows, hid, &lq.gate, None, inter)?;
                let u = linear(&xf, rows, hid, &lq.up, None, inter)?;
                let m = silu_mul(&g, &u)?;
                pin_mid = Some(mid);
                pin_xf = Some(xf);
                m
            }
        };

        let w_tok = sink_token_weights(
            hts,
            &pt.mid,
            &pt.out,
            hss,
            pin_mid.as_deref(),
            seqs,
            t,
            hid,
            cfg.sink_thresh,
        )?;

        let (ys, y_ts): (Vec<Vec<f64>>, Vec<Option<Vec<f64>>>) = match stage {
            StageId::Qkv => (
                vec![
                    sub_bias(&pt.q, &lt.qb, rows, qo)?,
                    sub_bias(&pt.k, &lt.kb, rows, kvo)?,
                    sub_bias(&pt.v, &lt.vb, rows, kvo)?,
                ],
                vec![None, None, None],
            ),
            StageId::O => {
                let xa = pin_xa.as_deref().ok_or(DriverError::Shape)?;
                let y = tri_sub(&pt.mid, hts, hss)?;
                let qt = linear(xa, rows, hid, &lt.q, Some(&lt.qb), qo)?;
                let kt = linear(xa, rows, hid, &lt.k, Some(&lt.kb), kvo)?;
                let vt = linear(xa, rows, hid, &lt.v, Some(&lt.vb), kvo)?;
                let a_ts = attn_batch(cfg, rope, &qt, &kt, &vt, seqs, t)?;
                let y_ts = linear(&a_ts, rows, qo, &lt.o, None, hid)?;
                (vec![y], vec![Some(y_ts)])
            }
            StageId::Ffn => (vec![pt.g.clone(), pt.u.clone()], vec![None, None]),
            StageId::Down => {
                let xf = pin_xf.as_deref().ok_or(DriverError::Shape)?;
                let mid = pin_mid.as_deref().ok_or(DriverError::Shape)?;
                let y = tri_sub(&pt.out, &pt.mid, mid)?;
                let mg = linear(xf, rows, hid, &lt.gate, None, inter)?;
                let mu = linear(xf, rows, hid, &lt.up, None, inter)?;
                let m_ts = silu_mul(&mg, &mu)?;
                let y_ts = linear(&m_ts, rows, inter, &lt.down, None, hid)?;
                (vec![y], vec![Some(y_ts)])
            }
        };

        let ys_ref: Vec<&[f64]> = ys.iter().map(Vec::as_slice).collect();
        let yts_ref: Vec<Option<&[f64]>> = y_ts.iter().map(|o| o.as_deref()).collect();
        accum.add_slab(&pin_in, &w_tok, &ys_ref, &yts_ref)?;
        if stage != StageId::Qkv {
            slab_in.push(pin_in);
        }
    }

    let w0 = stage.teacher_tensors(lt);
    let ws = solve_stage(&accum, &w0, stage.solve_kinds(), &cfg.solve)?;

    let mut fits = Vec::with_capacity(outs.len());
    for (j, &kind) in kinds.iter().enumerate() {
        let o = outs[j];
        let rg_q = cfg.quant_ridge * (n as f64 / cfg.solve.n_ref);
        let damp = cfg.quant_damp * (n as f64 / cfg.solve.n_ref);
        let fit_cfg = FoldFitConfig {
            iters: cfg.fit.iters,
            ridge: rg_q,
            damp,
            inner_iters: cfg.fit.inner_iters,
            e8_beam: cfg.fit.e8_beam,
            icm_ridge: cfg.fit.icm_ridge,
            sweeps: cfg.fit.sweeps,
        };
        let g_row: Option<Vec<f64>> = match stage {
            StageId::Qkv => None,
            StageId::O => Some(g_row_from_vec(&lt.ln2)?),
            StageId::Ffn => Some(g_row_from_matrix(&lt.down, hid, inter)?),
            StageId::Down => Some(g_row_from_vec(ln_next)?),
        };
        let bias = match kind {
            TensorKind::Q | TensorKind::K | TensorKind::V => Some(BiasStats {
                x_sum: &accum.x_sum,
                y_sum: &accum.y_sum[j],
                n_toks: accum.n_toks,
            }),
            _ => None,
        };
        let params = TensorParams {
            rg_q,
            qmlp_rows: cfg.qmlp_rows,
            fit: fit_cfg,
            nb_sweeps: cfg.nb_sweeps,
            nb_rounds: cfg.nb_rounds,
        };
        let mut fit = fit_tensor(
            kind,
            &ws[j],
            &accum.hm,
            o,
            n,
            g_row.as_deref(),
            &params,
            bias,
        )?;
        if kind == TensorKind::Up {
            let refs: Vec<&[f64]> = slab_in.iter().map(Vec::as_slice).collect();
            let s_sol = resq_scale(&ResqInputs {
                x_slabs: &refs,
                n: hid,
                w_g0: &lt.gate,
                w_u0: &lt.up,
                wq_g: &lq.gate,
                wq_u: &fit.wp,
                w_down: &lt.down,
                inter,
                hidden: hid,
            })?;
            fold_row_scale(&mut fit, &s_sol, o, n)?;
        }
        apply_fit(lq, kind, lt, &fit)?;
        fits.push(fit);
    }

    Ok(StageOut {
        fits,
        slabs: slab_in,
    })
}

/// Validate the layer geometry, the stage widths and the driver scalars.
///
/// # Errors
/// [`DriverError::Geometry`] for an invalid head geometry, a zero `slab`/`n_seq` or `inter`;
/// [`DriverError::TinySeq`] if `t < 2`; [`DriverError::Shape`] if a stage input width is not a
/// positive multiple of `128`; [`DriverError::NonFinite`] if `eps` or `theta` is not a finite
/// positive value.
fn validate(cfg: &DriverCfg) -> Result<(), DriverError> {
    let l = &cfg.layer;
    if !l.eps.is_finite() || !l.theta.is_finite() || l.theta <= 0.0 {
        return Err(DriverError::NonFinite);
    }
    if l.n_head == 0
        || l.n_kv_head == 0
        || l.head_dim == 0
        || !l.head_dim.is_multiple_of(2)
        || l.inter == 0
        || !l.n_head.is_multiple_of(l.n_kv_head)
        || l.hidden != l.n_head * l.head_dim
    {
        return Err(DriverError::Geometry);
    }
    if cfg.slab == 0 || cfg.n_seq == 0 {
        return Err(DriverError::Geometry);
    }
    if cfg.t < 2 {
        return Err(DriverError::TinySeq);
    }
    // The fitted tensors' input widths (hidden for qkv/ffn, the query width for o, inter for down)
    // must be positive multiples of 128; `fold_fit`/`fit_tensor` require it.
    for n in [l.hidden, q_out(cfg), l.inter] {
        if n == 0 || !n.is_multiple_of(128) {
            return Err(DriverError::Shape);
        }
    }
    Ok(())
}

/// The owned length of every tensor of `lt` at `cfg`'s geometry, or a [`DriverError::Shape`] if a
/// field disagrees or is not finite.
fn check_layer(lt: &LayerWeights, cfg: &DriverCfg) -> Result<(), DriverError> {
    let l = &cfg.layer;
    let hid = l.hidden;
    let qo = q_out(cfg);
    let kvo = kv_out(cfg);
    let inter = l.inter;
    let ok = lt.ln1.len() == hid
        && lt.ln2.len() == hid
        && lt.q.len() == qo * hid
        && lt.qb.len() == qo
        && lt.k.len() == kvo * hid
        && lt.kb.len() == kvo
        && lt.v.len() == kvo * hid
        && lt.vb.len() == kvo
        && lt.o.len() == hid * qo
        && lt.gate.len() == inter * hid
        && lt.up.len() == inter * hid
        && lt.down.len() == hid * inter;
    if !ok {
        return Err(DriverError::Shape);
    }
    for slice in [
        &lt.ln1, &lt.ln2, &lt.q, &lt.qb, &lt.k, &lt.kb, &lt.v, &lt.vb, &lt.o, &lt.gate, &lt.up,
        &lt.down,
    ] {
        if !finite(slice) {
            return Err(DriverError::NonFinite);
        }
    }
    Ok(())
}

/// Quantise `lt` against the teacher/student hidden states, propagate the student path and return
/// the quantised layer, the seven fits, the bit budget and the teacher layer output.
///
/// `ht`/`hs` are row-major `[n_seq·t, hidden]`; `hs` is the student input and is replaced by the
/// student layer output. `ln_next` (length `hidden`) is the `down` stage's `g_row` source: the next
/// layer's `ln1`, or the final norm weight (`:1709`). Temporal `O(Σ_slabs t·parts + Σ_stages rows·n²
/// + Σ_7 fits)`, dominated by the teacher forward cache, the stage Gram and solve and the seven
/// [`fit_tensor`] calls; spatial `O(n_seq·t·hidden + Σ_slabs rows·hidden + n²)`.
///
/// # Errors
/// [`DriverError::Geometry`] for an invalid geometry or a zero `slab`; [`DriverError::TinySeq`] for
/// `t < 2`; [`DriverError::Shape`] for a length or width mismatch; [`DriverError::NonFinite`] for a
/// non-finite weight or hidden state; and every refusal of the stage, tensor, `-resq` and forward
/// helpers.
pub fn quantise_layer(
    cfg: &DriverCfg,
    rope: &RopeTables,
    lt: &LayerWeights,
    ln_next: &[f64],
    ht: &[f64],
    hs: &mut Vec<f64>,
) -> Result<LayerResult, DriverError> {
    validate(cfg)?;
    check_layer(lt, cfg)?;
    let hid = cfg.layer.hidden;
    let t = cfg.t;
    let rows_total = cfg.n_seq.checked_mul(t).ok_or(DriverError::Shape)?;
    let total = rows_total.checked_mul(hid).ok_or(DriverError::Shape)?;
    if ht.len() != total || hs.len() != total {
        return Err(DriverError::Shape);
    }
    if ln_next.len() != hid {
        return Err(DriverError::Shape);
    }
    if !finite(ht) || !finite(hs) || !finite(ln_next) {
        return Err(DriverError::NonFinite);
    }

    // The teacher forward cache, one `keep_keys` slab per slab (`:1532-1535`), and the teacher layer
    // output `ht_next`.
    let geom = slab_geometry(cfg.n_seq, t, cfg.slab)?;
    let mut pt_cache: Vec<SlabParts> = Vec::with_capacity(geom.len());
    let mut ht_next = Vec::with_capacity(total);
    for &(row0, rows, seqs) in &geom {
        let hts = &ht[row0 * hid..row0 * hid + rows * hid];
        let p = batch_parts(cfg, rope, lt, hts, seqs, t)?;
        ht_next.extend_from_slice(&p.out);
        pt_cache.push(p);
    }

    let mut lq = lt.clone();
    let hs_in = hs.clone();
    let mut hs_mid = vec![0.0_f64; total];
    let mut fits: Vec<TensorFit> = Vec::with_capacity(7);
    let mut bits = TensorBits::default();

    // Stage qkv: no student-path propagation.
    let s_qkv = run_stage(
        cfg,
        rope,
        lt,
        &mut lq,
        ln_next,
        StageId::Qkv,
        &pt_cache,
        ht,
        &hs_in,
        &hs_mid,
    )?;
    bits = add_bits(bits, &sum_bits(&s_qkv.fits));
    fits.extend(s_qkv.fits);

    // Stage o: fit `Lq.o`, then `hs_mid = hs + a·Lq.oᵀ` (`:1960-1965`).
    let s_o = run_stage(
        cfg,
        rope,
        lt,
        &mut lq,
        ln_next,
        StageId::O,
        &pt_cache,
        ht,
        &hs_in,
        &hs_mid,
    )?;
    for (i, &(row0, rows, _)) in geom.iter().enumerate() {
        let add = linear(&s_o.slabs[i], rows, q_out(cfg), &lq.o, None, hid)?;
        let lo = row0 * hid;
        add_into(
            &mut hs_mid[lo..lo + rows * hid],
            &hs_in[lo..lo + rows * hid],
            &add,
        )?;
    }
    bits = add_bits(bits, &sum_bits(&s_o.fits));
    fits.extend(s_o.fits);

    // Stage ffn: fit `Lq.gate` then `Lq.up` (with the `-resq` fold), on `hs_mid`.
    let s_ffn = run_stage(
        cfg,
        rope,
        lt,
        &mut lq,
        ln_next,
        StageId::Ffn,
        &pt_cache,
        ht,
        &hs_in,
        &hs_mid,
    )?;
    bits = add_bits(bits, &sum_bits(&s_ffn.fits));
    fits.extend(s_ffn.fits);

    // Stage down: fit `Lq.down`, then `hs = hs_mid + m·Lq.downᵀ` (`:1966-1967`).
    let s_down = run_stage(
        cfg,
        rope,
        lt,
        &mut lq,
        ln_next,
        StageId::Down,
        &pt_cache,
        ht,
        &hs_in,
        &hs_mid,
    )?;
    let mut hs_out = vec![0.0_f64; total];
    for (i, &(row0, rows, _)) in geom.iter().enumerate() {
        let add = linear(&s_down.slabs[i], rows, cfg.layer.inter, &lq.down, None, hid)?;
        let lo = row0 * hid;
        add_into(
            &mut hs_out[lo..lo + rows * hid],
            &hs_mid[lo..lo + rows * hid],
            &add,
        )?;
    }
    bits = add_bits(bits, &sum_bits(&s_down.fits));
    fits.extend(s_down.fits);

    *hs = hs_out;
    Ok(LayerResult {
        lq,
        fits,
        bits,
        ht_next,
    })
}

/// The bit budget summed over a slice of fits.
fn sum_bits(fits: &[TensorFit]) -> TensorBits {
    let mut b = TensorBits::default();
    for f in fits {
        b = add_bits(b, &f.bits);
    }
    b
}

/// The model driver: quantise the first `n_layers` layers, propagate the teacher and student hidden
/// states, and fit the final RMSNorm gain (`fold_transfer.py:1514-1531`, `:2034-2040`).
///
/// `h0` is `cfg.E[ids_cal]` (`[n_seq·t, hidden]`), `embed` the tied embedding `[vocab, hidden]`,
/// `fnorm_w` the teacher final norm weight. Layers `>= n_layers` are copied unchanged; the final fit
/// is `fnorm_fit(…, steps = 2, ridge = 0.10, lr = fnorm_lr, coh_prior = true)` (`:2037`). Temporal
/// `O(Σ_{l < n_layers} quantise_layer + fnorm_fit)`; spatial `O(layers.len()·hidden² + n_seq·t·hidden)`.
///
/// # Errors
/// [`DriverError::Shape`] if `n_layers > layers.len()`, `h0` has the wrong length, or a layer/embed
/// slice disagrees; and every refusal of [`quantise_layer`] and [`fnorm_fit`].
pub fn quantise_model(
    cfg: &DriverCfg,
    rope: &RopeTables,
    layers: &[LayerWeights],
    fnorm_w: &[f64],
    embed: &[f64],
    vocab: usize,
    h0: &[f64],
    n_layers: usize,
    fnorm_lr: f64,
) -> Result<ModelResult, DriverError> {
    validate(cfg)?;
    if n_layers > layers.len() {
        return Err(DriverError::Shape);
    }
    let hid = cfg.layer.hidden;
    let rows = cfg.n_seq.checked_mul(cfg.t).ok_or(DriverError::Shape)?;
    let total = rows.checked_mul(hid).ok_or(DriverError::Shape)?;
    if h0.len() != total || fnorm_w.len() != hid {
        return Err(DriverError::Shape);
    }
    if embed.len() != vocab.checked_mul(hid).ok_or(DriverError::Shape)? {
        return Err(DriverError::Shape);
    }
    if !finite(h0) || !finite(fnorm_w) || !finite(embed) {
        return Err(DriverError::NonFinite);
    }

    let mut st = model_init(h0);
    model_step(cfg, rope, layers, fnorm_w, &mut st, n_layers)?;
    let mut l = n_layers;
    while l < layers.len() {
        st.layers.push(layers[l].clone());
        st.fits.push(Vec::new());
        l += 1;
    }
    model_finish(cfg, st, fnorm_w, embed, vocab, fnorm_lr)
}

/// The state of a model transfer before its first layer: `next = 0`, both hidden states equal to the
/// embedded calibration batch `h0`, no layers, a zero bit budget. `O(h0.len())` time and space (the
/// two hidden-state copies); the geometry is checked by [`model_step`].
#[must_use]
pub fn model_init(h0: &[f64]) -> ModelState {
    ModelState {
        next: 0,
        ht: h0.to_vec(),
        hs: h0.to_vec(),
        layers: Vec::new(),
        bits: TensorBits::default(),
        fits: Vec::new(),
    }
}

/// Quantise the layers `st.next .. until` against the teacher slice `layers`, propagating the
/// teacher and student hidden states through `st` exactly as the [`quantise_model`] loop does.
///
/// `fnorm_w` is the final norm gain (the last layer's `down` `g_row` source); `until` must lie in
/// `st.next ..= layers.len()`, so a resumed state continues where its checkpoint left it and a
/// one-shot run passes `until = n_layers`. Temporal `O(Σ_{l = next}^{until-1} quantise_layer)`;
/// spatial `O(layers.len()·hidden² + n_seq·t·hidden)` plus one layer's fit scratch.
///
/// # Errors
/// [`DriverError::Shape`] for `until > layers.len()`, `until < st.next`, a hidden-state length that
/// disagrees with `n_seq·t·hidden`, or a `fnorm_w` that is not `hidden` wide;
/// [`DriverError::NonFinite`] for a non-finite hidden state or norm gain; and every refusal of
/// [`quantise_layer`].
pub fn model_step(
    cfg: &DriverCfg,
    rope: &RopeTables,
    layers: &[LayerWeights],
    fnorm_w: &[f64],
    st: &mut ModelState,
    until: usize,
) -> Result<(), DriverError> {
    validate(cfg)?;
    if until > layers.len() || until < st.next {
        return Err(DriverError::Shape);
    }
    let hid = cfg.layer.hidden;
    let rows = cfg.n_seq.checked_mul(cfg.t).ok_or(DriverError::Shape)?;
    let total = rows.checked_mul(hid).ok_or(DriverError::Shape)?;
    if st.ht.len() != total || st.hs.len() != total || fnorm_w.len() != hid {
        return Err(DriverError::Shape);
    }
    if !finite(&st.ht) || !finite(&st.hs) || !finite(fnorm_w) {
        return Err(DriverError::NonFinite);
    }
    let mut l = st.next;
    while l < until {
        let ln_next: &[f64] = if l + 1 < layers.len() {
            &layers[l + 1].ln1
        } else {
            fnorm_w
        };
        let r = quantise_layer(cfg, rope, &layers[l], ln_next, &st.ht, &mut st.hs)?;
        st.ht = r.ht_next;
        st.bits = add_bits(st.bits, &r.bits);
        st.layers.push(r.lq);
        st.fits.push(r.fits);
        l += 1;
        st.next = l;
    }
    Ok(())
}

/// Finish a transfer: the final RMSNorm-gain fit over the state's hidden states
/// (`fnorm_fit(…, steps = 2, ridge = 0.10, lr = fnorm_lr, coh_prior = true)`,
/// `fold_transfer.py:2037`), returning the quantised layers, `fnorm`, the bit sum and the fits.
///
/// `st.layers` must hold one entry per model layer: the quantised prefix `0 .. st.next` from
/// [`model_step`] and, from [`quantise_model`], the untouched tail copies. Temporal
/// `O(fnorm_fit + layers.len()·hidden²)`; spatial `O(vocab·hidden + n_seq·t·hidden)`.
///
/// # Errors
/// [`DriverError::Shape`] for a hidden-state or embedding length that disagrees with the geometry,
/// or a fit vector count that disagrees with the layer count; [`DriverError::NonFinite`] for a
/// non-finite input; and every refusal of [`fnorm_fit`].
pub fn model_finish(
    cfg: &DriverCfg,
    st: ModelState,
    fnorm_w: &[f64],
    embed: &[f64],
    vocab: usize,
    fnorm_lr: f64,
) -> Result<ModelResult, DriverError> {
    validate(cfg)?;
    let hid = cfg.layer.hidden;
    let rows = cfg.n_seq.checked_mul(cfg.t).ok_or(DriverError::Shape)?;
    let total = rows.checked_mul(hid).ok_or(DriverError::Shape)?;
    if st.ht.len() != total || st.hs.len() != total || fnorm_w.len() != hid {
        return Err(DriverError::Shape);
    }
    if embed.len() != vocab.checked_mul(hid).ok_or(DriverError::Shape)? {
        return Err(DriverError::Shape);
    }
    if st.fits.len() != st.layers.len() || st.next > st.layers.len() {
        return Err(DriverError::Shape);
    }
    if !finite(&st.ht) || !finite(&st.hs) || !finite(fnorm_w) || !finite(embed) {
        return Err(DriverError::NonFinite);
    }
    let fnorm = fnorm_fit(
        &st.ht,
        &st.hs,
        cfg.n_seq,
        cfg.t,
        hid,
        fnorm_w,
        embed,
        vocab,
        cfg.layer.eps,
        2,
        0.10,
        fnorm_lr,
        true,
    )?;
    Ok(ModelResult {
        layers: st.layers,
        fnorm,
        bits: st.bits,
        fits: st.fits,
    })
}
