//! Extraction unit for the finishing P-step (`src/learn/fold/finish/`) and the run glue's
//! `convert` (`src/learn/fold/run/`). Each module is reached by `#[path]`, so the extracted bodies
//! are the real sources; each `word_*` wrapper fixes the monomorphic root and calls the real
//! function, restating no formula.
//!
//! # Merge (`proof/extract/lib.rs`)
//!
//! The unit is declared at the crate root, as `fold_port`/`fit_port` are:
//!
//! ```text
//! #[path = "fold_finish_port.rs"]
//! pub mod fold_finish_port;
//! ```
//!
//! and the `learn::fold` mirror in `fold_port.rs` re-exports `finish` and `run`, so the sources'
//! absolute `crate::learn::fold::{finish,run}` paths resolve:
//!
//! ```text
//! pub mod fold {
//!     pub use crate::fold_port::{fit, stats};
//!     pub use crate::fold_finish_port::{convert, finish};
//!     pub mod run { pub use crate::fold_finish_port::convert; }
//! }
//! ```
//!
//! # Boundary
//!
//! `src/learn/fold/run/{transfer,finish,npy}.rs` are not declared: `transfer.rs` imports
//! `crate::crypto::sha256::sha256`, and this unit's `crypto` module is `src/crypto/keccak.rs`
//! (`proof/extract/lib.rs`), which has no `sha256` child; `run/mod.rs` would pull all three in.

/// The real `src/learn/fold/finish/`: the frozen-code weight plane and the P-step over it.
#[path = "../../src/learn/fold/finish/mod.rs"]
pub mod finish;

/// The real `src/learn/fold/run/convert.rs`: the engine-weight/fold-driver conversion and the
/// `GGUF` scalar encoders.
#[path = "../../src/learn/fold/run/convert.rs"]
pub mod convert;

// ── `learn::fold::finish::train` ──────────────────────────────────────────────────────────────

use crate::learn;
use convert::ConvertError;
use finish::train::{
    Adam, Finisher, Frozen, LayerTunables, Linear, Planes, Schedule, TrainError, Tunables,
};

/// The empty coded linear of the given shape.
pub fn word_finish_train_linear_new(o_total: usize, n: usize) -> Linear {
    Linear::new(o_total, n)
}

/// The per-layer, per-kind coded-linear table.
pub fn word_finish_train_planes_new(layers: Vec<Vec<Linear>>) -> Planes {
    Planes::new(layers)
}

/// Linear `k` of layer `l`.
pub fn word_finish_train_planes_linear(
    p: &Planes,
    l: usize,
    k: usize,
) -> Result<&Linear, TrainError> {
    p.linear(l, k)
}

/// Linear `k` of layer `l`, mutable.
pub fn word_finish_train_planes_linear_mut(
    p: &mut Planes,
    l: usize,
    k: usize,
) -> Result<&mut Linear, TrainError> {
    p.linear_mut(l, k)
}

/// One layer's zero-bit bases, every latent zero.
pub fn word_finish_train_layer_tunables_new(
    ln1: &[i64],
    ln2: &[i64],
    qb: &[i64],
    kb: &[i64],
    vb: &[i64],
) -> LayerTunables {
    LayerTunables::new(ln1, ln2, qb, kb, vb)
}

/// The frozen bias scale of `qb`/`kb`/`vb` (`0`/`1`/`2`).
pub fn word_finish_train_layer_tunables_bias_scale(lt: &LayerTunables, which: usize) -> i64 {
    lt.bias_scale(which)
}

/// The resolved gains and biases of one layer.
pub fn word_finish_train_layer_tunables_tuned(lt: &LayerTunables) -> finish::train::LayerTuned {
    lt.tuned()
}

/// The whole model's zero-bit tunables.
pub fn word_finish_train_tunables_new(layers: Vec<LayerTunables>, fn_base: &[i64]) -> Tunables {
    Tunables::new(layers, fn_base)
}

/// The final `output_norm` gain `fn_base · exp(fn_d)`.
pub fn word_finish_train_tunables_output_norm(t: &Tunables) -> Vec<i64> {
    t.output_norm()
}

/// The frozen embedding and head words.
pub fn word_finish_train_frozen_new(token_embd: &[i64], lm_head: &[i64]) -> Frozen {
    Frozen::new(token_embd, lm_head)
}

/// The step-counted cosine schedule.
pub fn word_finish_train_schedule_new(warmup: u64, steps: u64) -> Schedule {
    Schedule::new(warmup, steps)
}

/// The learning-rate factor at `step`, Q32.32.
pub fn word_finish_train_schedule_factor(s: &Schedule, step: u64) -> i64 {
    s.factor(step)
}

/// Zero Adam state for `n` parameters.
pub fn word_finish_train_adam_zeros(n: usize) -> Adam {
    Adam::zeros(n)
}

/// The parameter count the Adam state covers.
pub fn word_finish_train_adam_len(a: &Adam) -> usize {
    a.len()
}

/// Whether the Adam state covers no parameter.
pub fn word_finish_train_adam_is_empty(a: &Adam) -> bool {
    a.is_empty()
}

/// One Adam update in place.
pub fn word_finish_train_adam_apply(
    a: &mut Adam,
    lr: i64,
    p: &mut [i64],
    g: &[i64],
) -> Result<(), TrainError> {
    a.apply(lr, p, g)
}

/// The completed P-step updates.
pub fn word_finish_train_finisher_step_index(f: &Finisher) -> u64 {
    f.step_index()
}

/// The batch mean KL and the row-summed logit gradient.
pub fn word_finish_train_kl_mean(
    cfg: &learn::grad::model::Config,
    logits: &[i64],
    teacher: &[i64],
) -> Result<(i64, Vec<i64>), TrainError> {
    finish::train::kl_mean(cfg, logits, teacher)
}

// ── `learn::fold::run::convert` ───────────────────────────────────────────────────────────────

/// Whole little-endian `f16`s as Q32.32 words.
pub fn word_run_convert_decode_f16(raw: &[u8]) -> Result<Vec<i64>, ConvertError> {
    convert::decode_f16(raw)
}

/// Whole little-endian `bf16`s as Q32.32 words.
pub fn word_run_convert_decode_bf16(raw: &[u8]) -> Result<Vec<i64>, ConvertError> {
    convert::decode_bf16(raw)
}

/// Whole little-endian `f32`s as Q32.32 words.
pub fn word_run_convert_decode_f32(raw: &[u8]) -> Result<Vec<i64>, ConvertError> {
    convert::decode_f32(raw)
}
