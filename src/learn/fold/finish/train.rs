//! The finishing P-step: Adam on the coded planes' continuous latents and on the zero-bit
//! norm/bias tunables, driven by the KL teacher objective.
//!
//! Port of the production training step of `spec/pv_finish.py` (the tunables at 1206-1218, `tuned`
//! at 1220-1223, the `Adam` groups at 1410-1496 and the learning-rate schedule at 1886-1889) over
//! the frozen-code plane of [`crate::learn::fold::finish::plane`] and the Q32.32 whole-model
//! forward/backward of [`crate::learn::grad::model`].
//!
//! # What is assembled
//!
//! The stored weight of each coded linear is the sum of its planes'
//! [`CodedPlane::weight`]`(s)` plus an optional frozen dense `extra` term (the accepted corrector
//! on `q`/`down`); the student's norm gains and attention biases come from the zero-bit tunables:
//! `gamma = base · exp(d)` for `attn_norm`, `ffn_norm` and `output_norm`, and
//! `b = base + d · max(mean|base|, 1e-3)` for `qb`, `kb`, `vb`. Every assembled value is rounded
//! once to Q32.32 (`round(v · 2^32)`) and refuses a non-finite or out-of-range value.
//!
//! # Gradient
//!
//! The objective is the batch mean KL of the teacher's next-token distribution against the
//! student's, `Σ_rows KL(p_t ‖ p_s) / rows`, from [`kl_row`] on the student's Q32.32 logits. The
//! teacher's logits are an input. [`Finisher::gradients`] runs the Q32.32 backward once per step
//! and maps `dL/dW` back to
//! * the plane latents through [`CodedPlane::backward`] (the straight-through Jacobian of the
//!   unquantized forward), with `dL/dW` taken in real weight units (`grad_q32 / 2^32`);
//! * `d_ln1`/`d_ln2`/`d_fn` through `∂gamma/∂d = base · exp(d) = gamma`, and `d_qb`/`d_kb`/`d_vb`
//!   through the frozen bias scale `max(mean|base|, 1e-3)`.
//!
//! [`Finisher::step`] applies one `Adam` update to each group and returns the mean KL measured
//! **before** the update (the Python's own per-step `loss`).
//!
//! # Deviations from the Python
//!
//! * The cosine learning-rate schedule uses the **step** count, `frac = step / steps`, where the
//!   Python uses wall-clock `frac = (time − t0) / (t_end − t0)` with a `max(·, step/steps)` floor.
//!   Step-counted fractions make the port deterministic; the brief states this deviation.
//! * `amp`, the V-step, the `ste` code latents, the `corr` factor group, the `kan` register, the
//!   `block-s` AQLM warm-up and the `regrid` re-fit are out of scope and absent: the codes are
//!   frozen, and the training groups are the latents and the zero-bit norm/bias tunables.
//! * The forward and backward run in f64/Q32.32 on the CPU, not `bfloat16` on CUDA, so the loss is
//!   the Q32.32 pipeline's, not the reference's bf16 sample.
//! * [`Finisher::gradients`] accumulates the per-row logit gradient as the **sum** over rows and
//!   divides the mapped `dL/dW` by the row count once, in f64, where the Python's autograd
//!   divides the loss before the graph; the two agree up to that single rounding.
//!
//! # Grade
//!
//! `tested` (`tests/learn_fold_train.rs`); nothing here is extracted or refined to the canonical
//! corpus, so no statement here is `proved`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::must_use_candidate,
    reason = "the flat-slice kernels index row-major `[rows, cols]` buffers by single letters and \
              use the reference's `pa`/`pb`/`pd`, `ln1`/`ln2`, `qb`/`kb`/`vb` and `fn` carrier \
              names; the `i64 → f64` and `i64 → usize` casts are the Q32.32 boundary and are \
              range-checked or dimension-checked before use; `gradients`/`apply`/`assemble` are \
              each one linear composition over the seven kinds."
)]

use crate::learn::fold::finish::plane::{CodedPlane, LatentGrad, PlaneError};
use crate::learn::fold::fit::steps::GROUP;
use crate::learn::grad::head::{kl_row, HeadError};
use crate::learn::grad::model::{
    backward, forward, Config, Grads, LayerGrads, LayerWeights, ModelError, Weights,
};

/// `2^32`, the Q32.32 unit.
const TWO32: f64 = 4_294_967_296.0;

/// `2^63`, the exclusive bound of an `i64` value.
const I64_LIMIT: f64 = 9_223_372_036_854_775_808.0;

/// The latent group's learning rate (the Python `--lr-lat`).
pub const LR_LAT: f64 = 1e-3;

/// The norm/bias group's learning rate (the Python `--lr-norm`).
pub const LR_NORM: f64 = 1e-4;

/// Adam `beta_1` (torch default).
const BETA1: f64 = 0.9;

/// Adam `beta_2` (torch default).
const BETA2: f64 = 0.999;

/// Adam `eps` (torch default).
const ADAM_EPS: f64 = 1e-8;

/// The floor of the bias scale (the Python `clamp_min(1e-3)`).
const BIAS_SCALE_MIN: f64 = 1e-3;

/// The number of coded linears per trunk layer.
pub const N_KINDS: usize = 7;

/// The name of each coded linear, in [`Planes::layers`] and [`Gradients::planes`] index order:
/// `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`.
pub const KIND_NAMES: [&str; N_KINDS] = ["q", "k", "v", "o", "gate", "up", "down"];

/// A refused finishing-step construction, gradient or update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainError {
    /// A dimension, a buffer length or a plane shape disagrees with the model geometry.
    Shape(String),
    /// A named value is not finite (`NaN` or `±∞`).
    NotFinite(String),
    /// A value does not fit the Q32.32 `i64` carrier.
    Range(String),
    /// A checked `i64` add or a length product left its range.
    Overflow,
    /// A coded plane refused.
    Plane(PlaneError),
    /// The Q32.32 whole-model forward or backward refused.
    Model(ModelError),
    /// The KL head refused.
    Head(HeadError),
}

impl From<PlaneError> for TrainError {
    fn from(e: PlaneError) -> Self {
        TrainError::Plane(e)
    }
}

impl From<ModelError> for TrainError {
    fn from(e: ModelError) -> Self {
        TrainError::Model(e)
    }
}

impl From<HeadError> for TrainError {
    fn from(e: HeadError) -> Self {
        TrainError::Head(e)
    }
}

/// `(o_total, n)` of linear `k` in the model geometry: `q` is `q_out × n_embd`, `k`/`v` are
/// `kv_out × n_embd`, `o` is `n_embd × q_out`, `gate`/`up` are `n_ff × n_embd` and `down` is
/// `n_embd × n_ff`.
///
/// # Errors
///
/// [`TrainError::Shape`] for an index at or past [`N_KINDS`]; [`TrainError::Model`] from the
/// `q_out`/`kv_out` products.
///
/// # Complexity
/// `O(1)`.
fn kind_dims(cfg: &Config, k: usize) -> Result<(usize, usize), TrainError> {
    let e = cfg.n_embd;
    let q_out = cfg.q_out()?;
    let kv_out = cfg.kv_out()?;
    match k {
        0 => Ok((q_out, e)),
        1 | 2 => Ok((kv_out, e)),
        3 => Ok((e, q_out)),
        4 | 5 => Ok((cfg.n_ff, e)),
        6 => Ok((e, cfg.n_ff)),
        _ => Err(TrainError::Shape(format!("linear kind {k} >= {N_KINDS}"))),
    }
}

/// The `k`th weight gradient of a layer, in [`KIND_NAMES`] order.
///
/// # Complexity
/// `O(1)`.
fn kind_grad(lg: &LayerGrads, k: usize) -> &Vec<i64> {
    match k {
        0 => &lg.wq,
        1 => &lg.wk,
        2 => &lg.wv,
        3 => &lg.wo,
        4 => &lg.w_gate,
        5 => &lg.w_up,
        _ => &lg.w_down,
    }
}

/// `round(v · 2^32)` as `i64`, refusing a non-finite or out-of-range value.
///
/// # Errors
///
/// [`TrainError::Range`] if `v` is not finite or `round(v · 2^32)` does not fit `i64`.
///
/// # Complexity
/// `O(1)`.
fn to_q32(v: f64, what: &str) -> Result<i64, TrainError> {
    let q = (v * TWO32).round();
    if !(q.is_finite() && q > -I64_LIMIT && q < I64_LIMIT) {
        return Err(TrainError::Range(format!("{what} = {v}")));
    }
    Ok(q as i64)
}

/// A row-major f64 buffer rounded elementwise to Q32.32.
///
/// # Errors
///
/// As [`to_q32`] for an element out of range; [`TrainError::NotFinite`] for a `NaN`.
///
/// # Complexity
///
/// Temporal `O(x.len())`; spatial `O(x.len())` `i64` words.
fn q32_vec(x: &[f64], what: &str) -> Result<Vec<i64>, TrainError> {
    let mut out = Vec::with_capacity(x.len());
    for (i, &v) in x.iter().enumerate() {
        if v.is_nan() {
            return Err(TrainError::NotFinite(format!("{what}[{i}]")));
        }
        out.push(to_q32(v, what)?);
    }
    Ok(out)
}

/// The bias scale `max(mean|base|, 1e-3)` of one zero-bit bias vector; the frozen multiplier of its
/// additive latent. An empty base gives the floor.
///
/// # Complexity
///
/// Temporal `O(base.len())`; spatial `O(1)`.
fn bias_scale(base: &[f64]) -> f64 {
    if base.is_empty() {
        return BIAS_SCALE_MIN;
    }
    let mut acc = 0.0_f64;
    for &v in base {
        acc += v.abs();
    }
    (acc / base.len() as f64).max(BIAS_SCALE_MIN)
}

/// One coded linear: the planes whose weights sum into its stored weight, the optional frozen dense
/// addend, and the expected `o_total × n` shape.
#[derive(Clone, Debug)]
pub struct Linear {
    /// The frozen-code planes, summed in order by [`CodedPlane::add_into`].
    pub planes: Vec<CodedPlane>,
    /// An optional frozen dense addend, row-major `o_total × n` (the accepted corrector on
    /// `q`/`down`); never trained.
    pub extra: Option<Vec<f64>>,
    /// Out rows of the stored weight.
    pub o_total: usize,
    /// In columns of the stored weight.
    pub n: usize,
}

impl Linear {
    /// An empty linear of the given shape; add planes with [`Linear::planes`] and set
    /// [`Linear::extra`] before use.
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn new(o_total: usize, n: usize) -> Self {
        Self {
            planes: Vec::new(),
            extra: None,
            o_total,
            n,
        }
    }
}

/// The coded linears of the whole model: `layers[l][k]` is linear [`KIND_NAMES`]`[k]` of trunk
/// layer `l`.
#[derive(Clone, Debug)]
pub struct Planes {
    /// Per trunk layer, [`N_KINDS`] entries in [`KIND_NAMES`] order.
    pub layers: Vec<Vec<Linear>>,
}

impl Planes {
    /// Wrap a per-layer, per-kind table.
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn new(layers: Vec<Vec<Linear>>) -> Self {
        Self { layers }
    }

    /// Linear `k` of layer `l`.
    ///
    /// # Errors
    ///
    /// [`TrainError::Shape`] if `l` or `k` is past the table.
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn linear(&self, l: usize, k: usize) -> Result<&Linear, TrainError> {
        if k >= N_KINDS {
            return Err(TrainError::Shape(format!("linear kind {k} >= {N_KINDS}")));
        }
        self.layers
            .get(l)
            .and_then(|row| row.get(k))
            .ok_or_else(|| TrainError::Shape(format!("plane layer {l} kind {k} absent")))
    }

    /// Linear `k` of layer `l`, mutable.
    ///
    /// # Errors
    ///
    /// As [`Planes::linear`].
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn linear_mut(&mut self, l: usize, k: usize) -> Result<&mut Linear, TrainError> {
        if k >= N_KINDS {
            return Err(TrainError::Shape(format!("linear kind {k} >= {N_KINDS}")));
        }
        self.layers
            .get_mut(l)
            .and_then(|row| row.get_mut(k))
            .ok_or_else(|| TrainError::Shape(format!("plane layer {l} kind {k} absent")))
    }
}

/// One layer's resolved zero-bit values: `gamma = base · exp(d)` for the two norm gains, and
/// `b = base + d · max(mean|base|, 1e-3)` for the three attention biases.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerTuned {
    /// `attn_norm` gain, `n_embd` wide.
    pub ln1: Vec<f64>,
    /// `ffn_norm` gain, `n_embd` wide.
    pub ln2: Vec<f64>,
    /// `attn_q.bias`, `n_head·head_dim` wide.
    pub qb: Vec<f64>,
    /// `attn_k.bias`, `n_head_kv·head_dim` wide.
    pub kb: Vec<f64>,
    /// `attn_v.bias`, `n_head_kv·head_dim` wide.
    pub vb: Vec<f64>,
}

/// One layer's zero-bit bases and their trainable latents.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerTunables {
    /// `attn_norm` base gain, `n_embd` wide.
    pub ln1: Vec<f64>,
    /// `attn_norm` log latent; `gamma = ln1 · exp(ln1_d)`.
    pub ln1_d: Vec<f64>,
    /// `ffn_norm` base gain, `n_embd` wide.
    pub ln2: Vec<f64>,
    /// `ffn_norm` log latent.
    pub ln2_d: Vec<f64>,
    /// `attn_q.bias` base.
    pub qb: Vec<f64>,
    /// `attn_q.bias` additive latent, scaled by [`LayerTunables::qb_scale`].
    pub qb_d: Vec<f64>,
    /// `attn_k.bias` base.
    pub kb: Vec<f64>,
    /// `attn_k.bias` additive latent.
    pub kb_d: Vec<f64>,
    /// `attn_v.bias` base.
    pub vb: Vec<f64>,
    /// `attn_v.bias` additive latent.
    pub vb_d: Vec<f64>,
}

impl LayerTunables {
    /// Build a layer's tunables from the frozen bases with every latent zero (the Python's
    /// `torch.zeros_like` initialisation).
    ///
    /// # Complexity
    ///
    /// Temporal `O(n_embd + q_out + kv_out)`; spatial the same.
    pub fn new(ln1: &[f64], ln2: &[f64], qb: &[f64], kb: &[f64], vb: &[f64]) -> Self {
        Self {
            ln1: ln1.to_vec(),
            ln1_d: vec![0.0_f64; ln1.len()],
            ln2: ln2.to_vec(),
            ln2_d: vec![0.0_f64; ln2.len()],
            qb: qb.to_vec(),
            qb_d: vec![0.0_f64; qb.len()],
            kb: kb.to_vec(),
            kb_d: vec![0.0_f64; kb.len()],
            vb: vb.to_vec(),
            vb_d: vec![0.0_f64; vb.len()],
        }
    }

    /// The frozen bias scale of `qb`/`kb`/`vb` (`0`/`1`/`2`).
    ///
    /// # Complexity
    /// `O(width)`.
    pub fn bias_scale(&self, which: usize) -> f64 {
        match which {
            0 => bias_scale(&self.qb),
            1 => bias_scale(&self.kb),
            _ => bias_scale(&self.vb),
        }
    }

    /// The resolved gains and biases `(ln1, ln2, qb, kb, vb)` of [`LayerTunables`]; the Python's
    /// `tuned(li)`.
    ///
    /// # Complexity
    ///
    /// Temporal `O(n_embd + q_out + kv_out)` multiplies and `exp` calls; spatial the same.
    pub fn tuned(&self) -> LayerTuned {
        let exp = |base: &[f64], d: &[f64]| -> Vec<f64> {
            base.iter().zip(d).map(|(&b, &p)| b * p.exp()).collect()
        };
        let add = |base: &[f64], d: &[f64], s: f64| -> Vec<f64> {
            base.iter().zip(d).map(|(&b, &p)| b + p * s).collect()
        };
        LayerTuned {
            ln1: exp(&self.ln1, &self.ln1_d),
            ln2: exp(&self.ln2, &self.ln2_d),
            qb: add(&self.qb, &self.qb_d, bias_scale(&self.qb)),
            kb: add(&self.kb, &self.kb_d, bias_scale(&self.kb)),
            vb: add(&self.vb, &self.vb_d, bias_scale(&self.vb)),
        }
    }
}

/// The whole model's zero-bit tunables: one [`LayerTunables`] per trunk layer and the final
/// `output_norm` gain `fn_base · exp(fn_d)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Tunables {
    /// Per trunk layer.
    pub layers: Vec<LayerTunables>,
    /// `output_norm` base gain, `n_embd` wide.
    pub fn_base: Vec<f64>,
    /// `output_norm` log latent.
    pub fn_d: Vec<f64>,
}

impl Tunables {
    /// Build the tunables from the frozen bases with every latent zero.
    ///
    /// # Complexity
    ///
    /// Temporal `O(n_layer·(n_embd + q_out + kv_out) + n_embd)`; spatial the same.
    pub fn new(layers: Vec<LayerTunables>, fn_base: &[f64]) -> Self {
        Self {
            layers,
            fn_base: fn_base.to_vec(),
            fn_d: vec![0.0_f64; fn_base.len()],
        }
    }

    /// The final `output_norm` gain `fn_base · exp(fn_d)`.
    ///
    /// # Complexity
    /// `O(n_embd)`.
    pub fn output_norm(&self) -> Vec<f64> {
        self.fn_base
            .iter()
            .zip(&self.fn_d)
            .map(|(&b, &p)| b * p.exp())
            .collect()
    }
}

/// The frozen model tensors outside the coded planes: the token embedding and the head, both
/// row-major `vocab × n_embd` in f64.
#[derive(Clone, Debug, PartialEq)]
pub struct Frozen {
    /// `token_embd.weight`.
    pub token_embd: Vec<f64>,
    /// `output.weight` (or the tied `token_embd.weight`).
    pub lm_head: Vec<f64>,
}

/// The step-counted cosine learning-rate schedule:
/// `f(step) = min(1, step/warmup) · (0.1 + 0.9·(1 + cos(π·step/steps))/2)`.
///
/// The Python's `frac` is `(time − t0)/(t_end − t0)` floored at `step/steps`; this port uses the
/// step fraction alone so the schedule is a function of the update index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// Linear ramp length in steps.
    pub warmup: u64,
    /// Cosine period in steps.
    pub steps: u64,
}

impl Schedule {
    /// The schedule with the given ramp and period.
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn new(warmup: u64, steps: u64) -> Self {
        Self { warmup, steps }
    }

    /// The learning-rate factor at `step` (`step = 0` is before the first update).
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn factor(&self, step: u64) -> f64 {
        let wu = if self.warmup == 0 {
            1.0
        } else {
            (step as f64 / self.warmup as f64).min(1.0)
        };
        let frac = if self.steps == 0 {
            1.0
        } else {
            step as f64 / self.steps as f64
        };
        wu.min(1.0) * (0.1 + 0.9 * 0.5 * (1.0 + (core::f64::consts::PI * frac).cos()))
    }
}

/// One parameter vector's Adam state (torch defaults: `beta = (0.9, 0.999)`, `eps = 1e-8`, no
/// weight decay).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
}

impl Adam {
    /// Zero state for a parameter vector of `n` elements.
    ///
    /// # Complexity
    /// `O(n)`.
    pub fn zeros(n: usize) -> Self {
        Self {
            m: vec![0.0_f64; n],
            v: vec![0.0_f64; n],
            t: 0,
        }
    }

    /// The number of parameters this state covers.
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn len(&self) -> usize {
        self.m.len()
    }

    /// Whether the state covers no parameter.
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn is_empty(&self) -> bool {
        self.m.is_empty()
    }

    /// The number of completed updates.
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn t(&self) -> u64 {
        self.t
    }

    /// One Adam update in place: `m ← β₁m + (1−β₁)g`, `v ← β₂v + (1−β₂)g²`,
    /// `p ← p − lr·m̂/(√v̂ + ε)`.
    ///
    /// # Errors
    ///
    /// [`TrainError::Shape`] if `p.len()`, `g.len()` and the state disagree.
    ///
    /// # Complexity
    ///
    /// Temporal `O(p.len())` (two moment updates and one parameter update per element); spatial
    /// `O(1)` in place.
    pub fn apply(&mut self, lr: f64, p: &mut [f64], g: &[f64]) -> Result<(), TrainError> {
        if p.len() != g.len() || p.len() != self.m.len() {
            return Err(TrainError::Shape(format!(
                "adam width {} vs {} vs {}",
                p.len(),
                g.len(),
                self.m.len()
            )));
        }
        self.t += 1;
        let bc1 = 1.0 - BETA1.powi(i32::try_from(self.t).unwrap_or(i32::MAX));
        let bc2 = 1.0 - BETA2.powi(i32::try_from(self.t).unwrap_or(i32::MAX));
        for i in 0..p.len() {
            let gi = g[i];
            self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * gi;
            self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * gi * gi;
            let mhat = self.m[i] / bc1;
            let vhat = self.v[i] / bc2;
            p[i] -= lr * mhat / (vhat.sqrt() + ADAM_EPS);
        }
        Ok(())
    }
}

/// The summed plane weight of one linear plus its optional frozen dense addend, in f64.
///
/// `quantized` selects the stored forward [`CodedPlane::weight`] (the training forward) or the
/// differentiable [`CodedPlane::weight_unquantized`] (the straight-through reference the
/// finite-difference control differentiates).
///
/// # Errors
///
/// [`TrainError::Shape`] for a plane width that disagrees with the linear or an `extra` of the
/// wrong length; [`TrainError::Plane`] from [`CodedPlane::add_into`].
///
/// # Complexity
///
/// Temporal `O(Σ_p o_p·n·log2(GROUP) + o_total·n)`; spatial `O(o_total·n)` f64 words.
fn linear_weight(lin: &Linear, quantized: bool) -> Result<Vec<f64>, TrainError> {
    let on = lin.o_total.checked_mul(lin.n).ok_or(TrainError::Overflow)?;
    let mut w = vec![0.0_f64; on];
    for p in &lin.planes {
        if p.n != lin.n || p.o == 0 || !p.n.is_multiple_of(GROUP) {
            return Err(TrainError::Shape(format!(
                "plane {}x{} in linear {}x{}",
                p.o, p.n, lin.o_total, lin.n
            )));
        }
        let pw = if quantized {
            p.weight()
        } else {
            p.weight_unquantized()
        };
        p.add_into(&mut w, &pw, lin.o_total)?;
    }
    if let Some(e) = &lin.extra {
        if e.len() != on {
            return Err(TrainError::Shape(format!(
                "extra {} != {}x{}",
                e.len(),
                lin.o_total,
                lin.n
            )));
        }
        for (a, b) in w.iter_mut().zip(e) {
            *a += b;
        }
    }
    Ok(w)
}

/// The Q32.32 `LayerWeights` of every kind of one layer, from the assembled linears and the tuned
/// zero-bit values.
///
/// # Complexity
///
/// Temporal `O(Σ_k (o_k·n_k·log2(GROUP) + o_k·n_k))`; spatial `O(Σ_k o_k·n_k)` Q32.32 words.
fn assemble_layer(
    cfg: &Config,
    planes: &Planes,
    tun: &LayerTunables,
    l: usize,
    quantized: bool,
) -> Result<LayerWeights, TrainError> {
    let mut weights: Vec<Vec<i64>> = Vec::with_capacity(N_KINDS);
    for k in 0..N_KINDS {
        let lin = planes.linear(l, k)?;
        let (o, n) = kind_dims(cfg, k)?;
        if lin.o_total != o || lin.n != n {
            return Err(TrainError::Shape(format!(
                "layer {l} kind {}: planes {}x{} != model {o}x{n}",
                KIND_NAMES[k], lin.o_total, lin.n
            )));
        }
        let w = linear_weight(lin, quantized)?;
        weights.push(q32_vec(&w, &format!("layer {l} {}", KIND_NAMES[k]))?);
    }
    let t = tun.tuned();
    let mut it = weights.into_iter();
    let mut next = |what: &str| -> Result<Vec<i64>, TrainError> {
        it.next()
            .ok_or_else(|| TrainError::Shape(format!("layer {l} {what} absent")))
    };
    let wq = next(KIND_NAMES[0])?;
    let wk = next(KIND_NAMES[1])?;
    let wv = next(KIND_NAMES[2])?;
    let wo = next(KIND_NAMES[3])?;
    let w_gate = next(KIND_NAMES[4])?;
    let w_up = next(KIND_NAMES[5])?;
    let w_down = next(KIND_NAMES[6])?;
    Ok(LayerWeights {
        attn_norm: q32_vec(&t.ln1, "attn_norm")?,
        wq,
        bq: Some(q32_vec(&t.qb, "attn_q.bias")?),
        wk,
        bk: Some(q32_vec(&t.kb, "attn_k.bias")?),
        wv,
        bv: Some(q32_vec(&t.vb, "attn_v.bias")?),
        wo,
        ffn_norm: q32_vec(&t.ln2, "ffn_norm")?,
        w_gate,
        w_up,
        w_down,
    })
}

/// Assemble the whole-model Q32.32 [`Weights`]: the stored forward weight of every coded linear
/// plus the tuned norms and biases, the frozen embedding and the frozen head.
///
/// # Errors
///
/// [`TrainError::Shape`] for a plane or tunable shape that disagrees with the model geometry;
/// [`TrainError::Plane`] from a plane's forward; [`TrainError::Range`] or [`TrainError::NotFinite`]
/// for a value outside the Q32.32 carrier.
///
/// # Complexity
///
/// Temporal `O(n_layer·Σ_k o_k·n_k·log2(GROUP) + vocab·n_embd)`; spatial
/// `O(n_layer·Σ_k o_k·n_k + vocab·n_embd)` Q32.32 words.
pub fn assemble(
    cfg: &Config,
    planes: &Planes,
    tun: &Tunables,
    frozen: &Frozen,
) -> Result<Weights, TrainError> {
    assemble_with(cfg, planes, tun, frozen, true)
}

/// As [`assemble`], over the differentiable [`CodedPlane::weight_unquantized`] forward: the
/// straight-through reference whose latents' Jacobian is the one the plane backward returns.
///
/// # Errors
///
/// As [`assemble`].
///
/// # Complexity
///
/// As [`assemble`].
pub fn assemble_unquantized(
    cfg: &Config,
    planes: &Planes,
    tun: &Tunables,
    frozen: &Frozen,
) -> Result<Weights, TrainError> {
    assemble_with(cfg, planes, tun, frozen, false)
}

/// [`assemble`]/[`assemble_unquantized`]'s shared body.
///
/// # Complexity
///
/// As [`assemble`].
fn assemble_with(
    cfg: &Config,
    planes: &Planes,
    tun: &Tunables,
    frozen: &Frozen,
    quantized: bool,
) -> Result<Weights, TrainError> {
    let e = cfg.n_embd;
    if planes.layers.len() != cfg.n_layer || tun.layers.len() != cfg.n_layer {
        return Err(TrainError::Shape(format!(
            "planes {} / tunables {} != n_layer {}",
            planes.layers.len(),
            tun.layers.len(),
            cfg.n_layer
        )));
    }
    let emb = cfg.vocab.checked_mul(e).ok_or(TrainError::Overflow)?;
    if frozen.token_embd.len() != emb || frozen.lm_head.len() != emb {
        return Err(TrainError::Shape(format!(
            "token_embd {} / lm_head {} != vocab·n_embd {emb}",
            frozen.token_embd.len(),
            frozen.lm_head.len()
        )));
    }
    let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.n_layer);
    for l in 0..cfg.n_layer {
        layers.push(assemble_layer(cfg, planes, &tun.layers[l], l, quantized)?);
    }
    Ok(Weights {
        layers,
        output_norm: q32_vec(&tun.output_norm(), "output_norm")?,
        token_embd: q32_vec(&frozen.token_embd, "token_embd")?,
        lm_head: q32_vec(&frozen.lm_head, "lm_head")?,
    })
}

/// The batch mean KL of the teacher's distribution against the student's Q32.32 logits, with the
/// row-summed logit gradient `dL/dz` (`t · vocab`, Q32.32).
///
/// The returned `(mean_kl, dz)` are the Python's `kl_batch` pair for one chunk: `mean_kl` is
/// `Σ_rows KL / rows` in real units and `dz` is the sum over rows of `p_s − p_t`.
///
/// # Errors
///
/// [`TrainError::Shape`] for a logits/teacher length that is not `t·vocab` with `t > 0`;
/// [`TrainError::Head`] from [`kl_row`]; [`TrainError::Overflow`] when the summed loss or gradient
/// leaves `i64`.
///
/// # Complexity
///
/// Temporal `O(t·vocab)` for `t` [`kl_row`] calls; spatial `O(t·vocab)` `i64` words for `dz`.
pub fn kl_mean(
    cfg: &Config,
    logits: &[i64],
    teacher: &[i64],
) -> Result<(f64, Vec<i64>), TrainError> {
    let v = cfg.vocab;
    if v == 0 || logits.len() != teacher.len() || !logits.len().is_multiple_of(v) {
        return Err(TrainError::Shape(format!(
            "logits {} / teacher {} not a multiple of vocab {v}",
            logits.len(),
            teacher.len()
        )));
    }
    let t = logits.len() / v;
    if t == 0 {
        return Err(TrainError::Shape("empty batch".to_string()));
    }
    let mut dz = vec![0_i64; logits.len()];
    let mut acc: i64 = 0;
    for r in 0..t {
        let (loss, g) = kl_row(&logits[r * v..r * v + v], &teacher[r * v..r * v + v])?;
        acc = acc.checked_add(loss).ok_or(TrainError::Overflow)?;
        for k in 0..v {
            dz[r * v + k] = dz[r * v + k]
                .checked_add(g[k])
                .ok_or(TrainError::Overflow)?;
        }
    }
    Ok((acc as f64 / (t as f64 * TWO32), dz))
}

/// The gradients of the batch mean KL with respect to one layer's zero-bit latents.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerNormGrad {
    /// `∂L/∂ attn_norm` log latent.
    pub ln1: Vec<f64>,
    /// `∂L/∂ ffn_norm` log latent.
    pub ln2: Vec<f64>,
    /// `∂L/∂ attn_q.bias` additive latent.
    pub qb: Vec<f64>,
    /// `∂L/∂ attn_k.bias` additive latent.
    pub kb: Vec<f64>,
    /// `∂L/∂ attn_v.bias` additive latent.
    pub vb: Vec<f64>,
}

/// The gradients of the batch mean KL with respect to every trainable latent of the finishing
/// step, plus the mean KL the gradients were taken at.
#[derive(Clone, Debug, PartialEq)]
pub struct Gradients {
    /// Per layer, per [`KIND_NAMES`] entry, per plane: the plane's
    /// `(∂L/∂pa, ∂L/∂pb, ∂L/∂pd)`.
    pub planes: Vec<Vec<Vec<LatentGrad>>>,
    /// Per layer, the zero-bit norm/bias latents' gradients.
    pub norms: Vec<LayerNormGrad>,
    /// `∂L/∂ output_norm` log latent.
    pub fn_d: Vec<f64>,
    /// The batch mean KL the gradients are taken at (real units).
    pub mean_kl: f64,
}

/// The P-step state: the coded planes, the zero-bit tunables, the frozen tensors, one Adam state
/// per parameter and the step-counted learning-rate schedule.
#[derive(Clone, Debug)]
pub struct Finisher {
    /// The model geometry.
    pub cfg: Config,
    /// The coded planes of every layer and kind.
    pub planes: Planes,
    /// The zero-bit bases and latents.
    pub tun: Tunables,
    /// The frozen embedding and head.
    pub frozen: Frozen,
    /// The learning-rate schedule.
    pub schedule: Schedule,
    /// The Adam state of every plane's `pa`/`pb`/`pd`.
    moments: Vec<Vec<Vec<PlaneMoments>>>,
    /// The Adam state of every layer's zero-bit latents.
    norm_moments: Vec<[Adam; 5]>,
    /// The Adam state of the final-norm latent.
    fn_moment: Adam,
    /// Completed updates.
    step: u64,
}

/// The Adam state of one plane's three latent axes.
#[derive(Clone, Debug, PartialEq)]
struct PlaneMoments {
    /// `pa` state.
    pa: Adam,
    /// `pb` state.
    pb: Adam,
    /// `pd` state.
    pd: Adam,
}

impl PlaneMoments {
    /// Zero state sized to the plane's three latents.
    ///
    /// # Complexity
    /// `O(o·n/128 + n)`.
    fn zeros(p: &CodedPlane) -> Self {
        Self {
            pa: Adam::zeros(p.pa.len()),
            pb: Adam::zeros(p.pb.len()),
            pd: Adam::zeros(p.pd.len()),
        }
    }
}

impl Finisher {
    /// Build the P-step: validate the geometry against `cfg` and allocate the Adam state.
    ///
    /// # Errors
    ///
    /// [`TrainError::Shape`] for a plane, tunable or frozen tensor that disagrees with the model
    /// geometry; [`TrainError::Model`] from [`Config::checked`].
    ///
    /// # Complexity
    ///
    /// Temporal `O(n_layer·(Σ_k o_k·n_k/128 + n_embd + q_out + kv_out))` for the validation and the
    /// state allocation; spatial the same.
    pub fn new(
        cfg: Config,
        planes: Planes,
        tun: Tunables,
        frozen: Frozen,
        schedule: Schedule,
    ) -> Result<Self, TrainError> {
        cfg.checked()?;
        let e = cfg.n_embd;
        let q_out = cfg.q_out()?;
        let kv_out = cfg.kv_out()?;
        if planes.layers.len() != cfg.n_layer {
            return Err(TrainError::Shape(format!(
                "planes {} != n_layer {}",
                planes.layers.len(),
                cfg.n_layer
            )));
        }
        if tun.layers.len() != cfg.n_layer || tun.fn_base.len() != e || tun.fn_d.len() != e {
            return Err(TrainError::Shape("tunable layer count".to_string()));
        }
        for (l, row) in planes.layers.iter().enumerate() {
            if row.len() != N_KINDS {
                return Err(TrainError::Shape(format!(
                    "layer {l}: {} kinds != {N_KINDS}",
                    row.len()
                )));
            }
            for (k, lin) in row.iter().enumerate() {
                let (o, n) = kind_dims(&cfg, k)?;
                if lin.o_total != o || lin.n != n {
                    return Err(TrainError::Shape(format!(
                        "layer {l} kind {}: {}x{} != {o}x{n}",
                        KIND_NAMES[k], lin.o_total, lin.n
                    )));
                }
                for p in &lin.planes {
                    if p.o == 0 || p.n != n || !p.n.is_multiple_of(GROUP) {
                        return Err(TrainError::Shape(format!(
                            "layer {l} kind {}: plane {}x{}",
                            KIND_NAMES[k], p.o, p.n
                        )));
                    }
                    if let Some(rows) = &p.rows {
                        if rows.len() != p.o || rows.iter().any(|&r| r >= o) {
                            return Err(TrainError::Shape(format!(
                                "layer {l} kind {}: plane rows",
                                KIND_NAMES[k]
                            )));
                        }
                    } else if p.o != o {
                        return Err(TrainError::Shape(format!(
                            "layer {l} kind {}: plane rows {o}",
                            KIND_NAMES[k]
                        )));
                    }
                    if p.pa.len() != p.o * (p.n / GROUP) || p.pb.len() != p.n || p.pd.len() != p.n {
                        return Err(TrainError::Shape(format!(
                            "layer {l} kind {}: plane latents",
                            KIND_NAMES[k]
                        )));
                    }
                }
            }
        }
        for (l, lt) in tun.layers.iter().enumerate() {
            let bad = lt.ln1.len() != e
                || lt.ln1_d.len() != e
                || lt.ln2.len() != e
                || lt.ln2_d.len() != e
                || lt.qb.len() != q_out
                || lt.qb_d.len() != q_out
                || lt.kb.len() != kv_out
                || lt.kb_d.len() != kv_out
                || lt.vb.len() != kv_out
                || lt.vb_d.len() != kv_out;
            if bad {
                return Err(TrainError::Shape(format!("layer {l} tunable widths")));
            }
        }
        let emb = cfg.vocab.checked_mul(e).ok_or(TrainError::Overflow)?;
        if frozen.token_embd.len() != emb || frozen.lm_head.len() != emb {
            return Err(TrainError::Shape(format!(
                "token_embd {} / lm_head {} != {emb}",
                frozen.token_embd.len(),
                frozen.lm_head.len()
            )));
        }
        let moments: Vec<Vec<Vec<PlaneMoments>>> = planes
            .layers
            .iter()
            .map(|row| {
                row.iter()
                    .map(|lin| lin.planes.iter().map(PlaneMoments::zeros).collect())
                    .collect()
            })
            .collect();
        let norm_moments: Vec<[Adam; 5]> = tun
            .layers
            .iter()
            .map(|lt| {
                [
                    Adam::zeros(lt.ln1_d.len()),
                    Adam::zeros(lt.ln2_d.len()),
                    Adam::zeros(lt.qb_d.len()),
                    Adam::zeros(lt.kb_d.len()),
                    Adam::zeros(lt.vb_d.len()),
                ]
            })
            .collect();
        let fn_moment = Adam::zeros(tun.fn_d.len());
        Ok(Self {
            cfg,
            planes,
            tun,
            frozen,
            schedule,
            moments,
            norm_moments,
            fn_moment,
            step: 0,
        })
    }

    /// The completed updates.
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn step_index(&self) -> u64 {
        self.step
    }

    /// The batch mean KL at the current latents, over the stored (quantized) forward.
    ///
    /// # Errors
    ///
    /// As [`assemble`], [`forward`] and [`kl_mean`].
    ///
    /// # Complexity
    ///
    /// As [`Finisher::gradients`] without the backward.
    pub fn loss(&self, tokens: &[u32], teacher: &[i64]) -> Result<f64, TrainError> {
        let w = assemble(&self.cfg, &self.planes, &self.tun, &self.frozen)?;
        let (logits, _) = forward(&w, &self.cfg, tokens)?;
        Ok(kl_mean(&self.cfg, &logits, teacher)?.0)
    }

    /// The gradients of the batch mean KL with respect to every trainable latent; no parameter is
    /// changed.
    ///
    /// # Errors
    ///
    /// [`TrainError::Shape`] for a teacher/logits geometry mismatch; [`TrainError::Model`],
    /// [`TrainError::Head`], [`TrainError::Plane`] from the forward, head and plane backward;
    /// [`TrainError::Overflow`] on a checked accumulate.
    ///
    /// # Complexity
    ///
    /// Temporal: per layer one Q32.32 forward and backward
    /// `O(t·(n_embd² + n_embd·n_ff) + n_head·t²·head_dim)` plus the head `O(t·vocab·n_embd)`; the
    /// latent map is `O(Σ plane o·n·log2(GROUP))`. Spatial `O(t·vocab + Σ plane o·n)`.
    pub fn gradients(&self, tokens: &[u32], teacher: &[i64]) -> Result<Gradients, TrainError> {
        let w = assemble(&self.cfg, &self.planes, &self.tun, &self.frozen)?;
        let (logits, tape) = forward(&w, &self.cfg, tokens)?;
        let (mean_kl, dz) = kl_mean(&self.cfg, &logits, teacher)?;
        let grads: Grads = backward(&w, &self.cfg, &tape, &dz)?;
        let t = tokens.len();
        if t == 0 {
            return Err(TrainError::Shape("empty batch".to_string()));
        }
        let inv = 1.0 / (t as f64 * TWO32);
        let e = self.cfg.n_embd;

        let mut planes: Vec<Vec<Vec<LatentGrad>>> = Vec::with_capacity(self.cfg.n_layer);
        let mut norms: Vec<LayerNormGrad> = Vec::with_capacity(self.cfg.n_layer);
        for l in 0..self.cfg.n_layer {
            let lg = grads
                .layers
                .get(l)
                .ok_or_else(|| TrainError::Shape(format!("grad layer {l} absent")))?;
            let lt = &self.tun.layers[l];
            let row = self
                .planes
                .layers
                .get(l)
                .ok_or_else(|| TrainError::Shape(format!("plane layer {l} absent")))?;
            let mut per_kind: Vec<Vec<LatentGrad>> = Vec::with_capacity(N_KINDS);
            for (k, lin) in row.iter().enumerate() {
                let gw = kind_grad(lg, k);
                let gf: Vec<f64> = gw.iter().map(|&x| x as f64 * inv).collect();
                let mut per_plane: Vec<LatentGrad> = Vec::with_capacity(lin.planes.len());
                for p in &lin.planes {
                    per_plane.push(p.backward(&gf)?);
                }
                per_kind.push(per_plane);
            }
            planes.push(per_kind);
            let mut dn = LayerNormGrad {
                ln1: vec![0.0_f64; e],
                ln2: vec![0.0_f64; e],
                qb: vec![0.0_f64; lt.qb.len()],
                kb: vec![0.0_f64; lt.kb.len()],
                vb: vec![0.0_f64; lt.vb.len()],
            };
            let s_qb = lt.bias_scale(0);
            let s_kb = lt.bias_scale(1);
            let s_vb = lt.bias_scale(2);
            for j in 0..e {
                dn.ln1[j] = lg.attn_norm[j] as f64 * inv * lt.ln1[j] * lt.ln1_d[j].exp();
                dn.ln2[j] = lg.ffn_norm[j] as f64 * inv * lt.ln2[j] * lt.ln2_d[j].exp();
            }
            for j in 0..dn.qb.len() {
                dn.qb[j] = lg.bq[j] as f64 * inv * s_qb;
            }
            for j in 0..dn.kb.len() {
                dn.kb[j] = lg.bk[j] as f64 * inv * s_kb;
            }
            for j in 0..dn.vb.len() {
                dn.vb[j] = lg.bv[j] as f64 * inv * s_vb;
            }
            norms.push(dn);
        }
        let mut fn_d = vec![0.0_f64; e];
        for j in 0..e {
            fn_d[j] =
                grads.output_norm[j] as f64 * inv * self.tun.fn_base[j] * self.tun.fn_d[j].exp();
        }
        Ok(Gradients {
            planes,
            norms,
            fn_d,
            mean_kl,
        })
    }

    /// One Adam update of every latent at the current step's learning-rate factor; nothing is
    /// changed on a shape disagreement.
    ///
    /// # Errors
    ///
    /// [`TrainError::Shape`] if a gradient width disagrees with its parameter.
    ///
    /// # Complexity
    ///
    /// Temporal `O(Σ plane (o·n/128 + n) + n_layer·(n_embd + q_out + kv_out))`; spatial `O(1)` in
    /// place.
    pub fn apply(&mut self, g: &Gradients, factor: f64) -> Result<(), TrainError> {
        let lr_lat = LR_LAT * factor;
        let lr_norm = LR_NORM * factor;
        for l in 0..self.cfg.n_layer {
            let row = self
                .planes
                .layers
                .get_mut(l)
                .ok_or_else(|| TrainError::Shape(format!("plane layer {l} absent")))?;
            let gk = g
                .planes
                .get(l)
                .ok_or_else(|| TrainError::Shape(format!("grad plane layer {l} absent")))?;
            let ml = self
                .moments
                .get_mut(l)
                .ok_or_else(|| TrainError::Shape(format!("moment layer {l} absent")))?;
            for (k, lin) in row.iter_mut().enumerate() {
                let gp = gk
                    .get(k)
                    .ok_or_else(|| TrainError::Shape(format!("grad kind {k} absent")))?;
                let mk = ml
                    .get_mut(k)
                    .ok_or_else(|| TrainError::Shape(format!("moment kind {k} absent")))?;
                for (i, p) in lin.planes.iter_mut().enumerate() {
                    let (dpa, dpb, dpd) = gp
                        .get(i)
                        .ok_or_else(|| TrainError::Shape(format!("grad plane {i} absent")))?;
                    let m = mk
                        .get_mut(i)
                        .ok_or_else(|| TrainError::Shape(format!("moment plane {i} absent")))?;
                    m.pa.apply(lr_lat, &mut p.pa, dpa)?;
                    m.pb.apply(lr_lat, &mut p.pb, dpb)?;
                    m.pd.apply(lr_lat, &mut p.pd, dpd)?;
                }
            }
            let gn = g
                .norms
                .get(l)
                .ok_or_else(|| TrainError::Shape(format!("grad norm layer {l} absent")))?;
            let lt = self
                .tun
                .layers
                .get_mut(l)
                .ok_or_else(|| TrainError::Shape(format!("tunable layer {l} absent")))?;
            let ml = self
                .norm_moments
                .get_mut(l)
                .ok_or_else(|| TrainError::Shape(format!("norm moment layer {l} absent")))?;
            ml[0].apply(lr_norm, &mut lt.ln1_d, &gn.ln1)?;
            ml[1].apply(lr_norm, &mut lt.ln2_d, &gn.ln2)?;
            ml[2].apply(lr_norm, &mut lt.qb_d, &gn.qb)?;
            ml[3].apply(lr_norm, &mut lt.kb_d, &gn.kb)?;
            ml[4].apply(lr_norm, &mut lt.vb_d, &gn.vb)?;
        }
        let mut fnd = self.tun.fn_d.clone();
        self.fn_moment.apply(lr_norm, &mut fnd, &g.fn_d)?;
        self.tun.fn_d = fnd;
        Ok(())
    }

    /// One finishing P-step: the mean KL at the current latents, then one Adam update of every
    /// latent at the step's learning-rate factor.
    ///
    /// # Errors
    ///
    /// As [`Finisher::gradients`] and [`Finisher::apply`].
    ///
    /// # Complexity
    ///
    /// As [`Finisher::gradients`] plus `O(Σ plane (o·n/128 + n))` for the update.
    pub fn step(&mut self, tokens: &[u32], teacher: &[i64]) -> Result<f64, TrainError> {
        let g = self.gradients(tokens, teacher)?;
        self.step += 1;
        let f = self.schedule.factor(self.step);
        self.apply(&g, f)?;
        Ok(g.mean_kl)
    }
}
