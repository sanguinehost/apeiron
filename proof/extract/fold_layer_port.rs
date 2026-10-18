//! Extraction unit for the per-layer fold transfer (`src/learn/fold/layer/`). The module tree is
//! reached through its real `mod.rs` by `#[path]`, so the extracted bodies are the real sources;
//! each `word_layer_*` wrapper fixes the monomorphic root and calls the real function, restating no
//! formula.
//!
//! # Merge (`proof/extract/lib.rs`)
//!
//! ```text
//! #[path = "fold_layer_port.rs"]
//! pub mod fold_layer_port;
//! ```
//!
//! and the `learn::fold` mirror in `fold_port.rs` re-exports `layer`, so the sources' absolute
//! `crate::learn::fold::layer::…` paths resolve:
//!
//! ```text
//! pub use crate::fold_layer_port::layer;
//! ```
//!
//! # Boundary
//!
//! `layer/mod.rs` declares `stage`, `tensor`, `driver` and `resq`; the wrappers below root the
//! public entry points of `driver`, `tensor` and `resq` only. `stage` is not rooted: its bodies are
//! reached only through a driver root and are translated then, or refused and recorded.

/// The real `src/learn/fold/layer/`: the stage accumulator/solve, the per-tensor assembly, the
/// `-resq` row scale and the layer driver.
#[path = "../../src/learn/fold/layer/mod.rs"]
pub mod layer;

use crate::learn::fold::fit::quant::RHier;
use crate::learn::fold::stats::parts::{self, LayerCfg, LayerWords, RopeTables};
use layer::driver::{
    DriverCfg, DriverError, LayerProgress, LayerResult, LayerWeights, ModelResult, ModelState,
};
use layer::resq::{ResqError, ResqInputs, ResqSolution};
use layer::tensor::{BiasStats, SinkPlane, TensorError, TensorFit, TensorKind, TensorParams};

// ── `learn::fold::layer::driver` ─────────────────────────────────────────────────────────────

/// The twelve tensors of a layer as borrowed word slices.
pub fn word_layer_driver_layer_words(lt: &LayerWeights) -> Result<LayerWords<'_>, DriverError> {
    layer::driver::layer_words(lt)
}

/// The production arm's default driver configuration.
pub fn word_layer_driver_cfg_production(
    layer: LayerCfg,
    t: usize,
    n_seq: usize,
) -> DriverCfg {
    DriverCfg::production(layer, t, n_seq)
}

/// `g_row = sq / mean(sq)` for one vector.
pub fn word_layer_driver_g_row_from_vec(v: &[i64]) -> Result<Vec<i64>, DriverError> {
    layer::driver::g_row_from_vec(v)
}

/// `g_row` from the column sums of squares of a matrix.
pub fn word_layer_driver_g_row_from_matrix(
    w: &[i64],
    rows: usize,
    cols: usize,
) -> Result<Vec<i64>, DriverError> {
    layer::driver::g_row_from_matrix(w, rows, cols)
}

/// Quantise at most `max_stages` of a layer's remaining stages.
pub fn word_layer_driver_quantise_layer_staged(
    cfg: &DriverCfg,
    rope: &RopeTables,
    lt: &LayerWeights,
    ln_next: &[i64],
    ht: &[i64],
    hs: &mut Vec<i64>,
    progress: &mut Option<LayerProgress>,
    max_stages: usize,
) -> Result<Option<LayerResult>, DriverError> {
    layer::driver::quantise_layer_staged(cfg, rope, lt, ln_next, ht, hs, progress, max_stages)
}

/// Quantise one layer against the teacher/student hidden states.
pub fn word_layer_driver_quantise_layer(
    cfg: &DriverCfg,
    rope: &RopeTables,
    lt: &LayerWeights,
    ln_next: &[i64],
    ht: &[i64],
    hs: &mut Vec<i64>,
) -> Result<LayerResult, DriverError> {
    layer::driver::quantise_layer(cfg, rope, lt, ln_next, ht, hs)
}

/// Quantise the first `n_layers` layers and fit the final norm.
pub fn word_layer_driver_quantise_model(
    cfg: &DriverCfg,
    rope: &RopeTables,
    layers: &[LayerWeights],
    fnorm_w: &[i64],
    embed: &[i64],
    vocab: usize,
    h0: &[i64],
    n_layers: usize,
    fnorm_lr: i64,
) -> Result<ModelResult, DriverError> {
    layer::driver::quantise_model(cfg, rope, layers, fnorm_w, embed, vocab, h0, n_layers, fnorm_lr)
}

/// The state of a model transfer before its first layer.
pub fn word_layer_driver_model_init(h0: &[i64]) -> ModelState {
    layer::driver::model_init(h0)
}

/// Quantise the layers `st.next .. until`.
pub fn word_layer_driver_model_step(
    cfg: &DriverCfg,
    rope: &RopeTables,
    layers: &[LayerWeights],
    fnorm_w: &[i64],
    st: &mut ModelState,
    until: usize,
) -> Result<(), DriverError> {
    layer::driver::model_step(cfg, rope, layers, fnorm_w, st, until)
}

/// Quantise at most `max_stages` stages from `st`.
pub fn word_layer_driver_model_step_stages(
    cfg: &DriverCfg,
    rope: &RopeTables,
    layers: &[LayerWeights],
    fnorm_w: &[i64],
    st: &mut ModelState,
    max_stages: usize,
) -> Result<(), DriverError> {
    layer::driver::model_step_stages(cfg, rope, layers, fnorm_w, st, max_stages)
}

/// Finish a transfer with the final RMSNorm-gain fit.
pub fn word_layer_driver_model_finish(
    cfg: &DriverCfg,
    st: ModelState,
    fnorm_w: &[i64],
    embed: &[i64],
    vocab: usize,
    fnorm_lr: i64,
) -> Result<ModelResult, DriverError> {
    layer::driver::model_finish(cfg, st, fnorm_w, embed, vocab, fnorm_lr)
}

// ── `learn::fold::layer::tensor` ─────────────────────────────────────────────────────────────

/// The stored weight of a plane list.
pub fn word_layer_tensor_recon(
    planes: &[SinkPlane],
    o: usize,
    n: usize,
) -> Result<Vec<i64>, TensorError> {
    layer::tensor::recon(planes, o, n)
}

/// The per-tensor assembly after the stage solve.
pub fn word_layer_tensor_fit_tensor(
    kind: TensorKind,
    ws_nm: &[i64],
    hm: &[i64],
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
    params: &TensorParams,
    bias: Option<BiasStats<'_>>,
) -> Result<TensorFit, TensorError> {
    layer::tensor::fit_tensor(kind, ws_nm, hm, o, n, g_row, params, bias)
}

/// The `-resq` row-scale fold for an `up` fit.
pub fn word_layer_tensor_fold_row_scale(
    fit: &mut TensorFit,
    s_sol: &[i64],
    o: usize,
    n: usize,
) -> Result<(), TensorError> {
    layer::tensor::fold_row_scale(fit, s_sol, o, n)
}

// ── `learn::fold::layer::resq` ───────────────────────────────────────────────────────────────

/// The clamped per-row multiplier `s_sol`.
pub fn word_layer_resq_scale(inputs: &ResqInputs<'_>) -> Result<Vec<i64>, ResqError> {
    layer::resq::resq_scale(inputs)
}

/// Build and solve the `-resq` elementwise-Gram ridge system.
pub fn word_layer_resq_solve(inputs: &ResqInputs<'_>) -> Result<ResqSolution, ResqError> {
    layer::resq::resq_solve(inputs)
}

/// Fold a per-row multiplier into a plane's hierarchical row scale.
pub fn word_layer_resq_rescale_hier(
    rh: &[i64],
    s: &[i64],
    o: usize,
    g: usize,
) -> Result<RHier, ResqError> {
    layer::resq::rescale_hier(rh, s, o, g)
}

/// The partial-plane variant of `rescale_hier`.
pub fn word_layer_resq_rescale_hier_top(
    rh: &[i64],
    s: &[i64],
    rows: &[usize],
    g: usize,
) -> Result<RHier, ResqError> {
    layer::resq::rescale_hier_top(rh, s, rows, g)
}
