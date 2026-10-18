//! The finishing P-step: Adam on the coded planes' continuous latents and on the zero-bit
//! norm/bias tunables, driven by the KL teacher objective.
//!
//! Port of the production training step of `spec/pv_finish.py` (tunables 1206-1223, `Adam` groups
//! 1410-1496, schedule 1886-1889) over the plane of [`crate::learn::fold::finish::plane`] and the
//! Q32.32 whole-model forward/backward of [`crate::learn::grad::model`].
//!
//! Each coded linear's stored weight is the sum of its planes' [`CodedPlane::weight`] plus an
//! optional frozen dense `extra` (the corrector on `q`/`down`); the zero-bit tunables resolve as
//! `gamma = base · exp(d)` for the norms and `b = base + d · max(mean|base|, 1e-3)` for the
//! attention biases.
//!
//! The objective is the batch mean KL of the teacher's next-token distribution against the
//! student's, `Σ_rows KL(p_t ‖ p_s) / rows`, from [`kl_row`] on the student's logits.
//! [`Finisher::gradients`] runs the Q32.32 backward and maps `dL/dW` to the plane latents (the
//! straight-through Jacobian, per batch element as `grad_q32 / rows`), to `d_ln1`/`d_ln2`/`d_fn`
//! through `∂gamma/∂d = gamma`, and to `d_qb`/`d_kb`/`d_vb` through the frozen bias scale.
//! [`Finisher::step`] applies one Adam update and returns the mean KL measured **before** it.
//!
//! # Representation
//!
//! Latents, gradients, Adam moments, the schedule factor, the learning rate and the reported KL are
//! Q32.32 `i64`; products are `i128` Q64.64 rounded once
//! ([`crate::learn::solve::q32::round_q32_even`]); `exp` is
//! [`crate::numeric::fixed_exp::fixed_exp`], `sqrt` the exact [`crate::numeric::fixed_rsqrt::isqrt`],
//! `cos` [`crate::numeric::fixed_trig::fixed_sin_cos`]. The frozen tensors arrive as words from the
//! loader (whose decode refuses a non-finite or off-grid value), so the finisher's whole surface is
//! integer: nothing here converts a float.
//!
//! Agreement bounds against an `f64` evaluation of the same recurrence: a word operation rounds
//! once (`≤ 2^-33` relative, so `k` chained products `≤ k·2^-33`); `exp` is `≤ 1e-8` relative for
//! `x ≥ 0`; `sqrt` is floored (`2^-32`); `Adam`'s `eps = 1e-8` is 43 grid units, whose effect on a
//! step is `eps/(sqrt(v)+eps)`; the learning rate is carried at Q64.64 (`≤ 2^-64` from the Python's
//! decimal constant) and rounded onto the grid once per step; the schedule's cosine carries
//! `≤ 2^-20` plus three roundings.
//!
//! # Execution
//!
//! Every loop here is a serial integer loop: the per-row, per-parameter and per-element vector
//! updates are `≤ ~10^5` word ops per call, below the ~10^6 at which a pool dispatch pays. Tensor
//! work over `o × n` is the plane's, which pools it. Loops over layers, kinds and planes are
//! structural and serial.
//!
//! # Deviations from the Python
//!
//! * The schedule uses `frac = step / steps` (deterministic), where the Python uses wall-clock
//!   with a step floor.
//! * `amp`, the V-step, the `ste` code latents, the `corr` group, `kan`, `block-s` AQLM warm-up
//!   and the `regrid` refit are out of scope: the codes are frozen.
//! * The forward and backward run in Q32.32 on the CPU, not `bfloat16` on CUDA, and
//!   [`Finisher::gradients`] divides the summed logit gradient once by the row count.
//!
//! Grade `tested` (`tests/learn_fold_train.rs`); nothing here is extracted or refined.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::must_use_candidate,
    clippy::items_after_statements,
    clippy::unreadable_literal,
    reason = "the flat-slice kernels index row-major `[rows, cols]` buffers by single letters and \
              use the reference's `pa`/`pb`/`pd`, `ln1`/`ln2`, `qb`/`kb`/`vb` and `fn` carrier \
              names; the `i64` literals are Q32.32 constants; every `i128` narrowing is checked by \
              `narrow`; `gradients`/`apply`/`assemble` are each one linear composition over the \
              seven kinds."
)]

use crate::learn::fold::finish::plane::{
    div, div_ne, mul, CodedPlane, LatentGrad, PlaneError, ONE,
};
use crate::learn::fold::fit::steps::GROUP;
use crate::learn::grad::head::{kl_row, HeadError};
use crate::learn::grad::model::{
    backward, forward, Config, Grads, LayerGrads, LayerWeights, ModelError, Weights,
};
use crate::learn::solve::q32::{div_rne_pos, narrow, rshift_rne};
use crate::numeric::fixed_exp::fixed_exp;
use crate::numeric::fixed_rsqrt::isqrt;
use crate::numeric::fixed_trig::fixed_sin_cos;

/// The latent group's learning rate (the Python `--lr-lat`), `round(1e-3·2^64)`: carried at Q64.64
/// so one step's `lr·factor` product rounds once, onto Q32.32.
const LR_LAT_Q64: i64 = 18_446_744_073_709_552;

/// The norm/bias group's learning rate (the Python `--lr-norm`), `round(1e-4·2^64)`.
const LR_NORM_Q64: i64 = 1_844_674_407_370_955;

/// Adam `beta_1` (torch default), `round(0.9·2^32)`.
const BETA1_Q32: i64 = 3_865_470_566;

/// `1 − beta_1`, `round(0.1·2^32)`; with [`BETA1_Q32`] it sums to exactly `2^32`.
const ONE_MINUS_BETA1_Q32: i64 = 429_496_730;

/// Adam `beta_2` (torch default), `round(0.999·2^32)`.
const BETA2_Q32: i64 = 4_290_672_329;

/// `1 − beta_2`, `round(0.001·2^32)`.
const ONE_MINUS_BETA2_Q32: i64 = 4_294_967;

/// Adam `eps` (torch default), `round(1e-8·2^32) = 43`.
const ADAM_EPS_Q32: i64 = 43;

/// The floor of the bias scale (the Python `clamp_min(1e-3)`), `round(1e-3·2^32)`.
const BIAS_SCALE_MIN_Q32: i64 = 4_294_967;

/// `0.1`, `round(0.1·2^32)`.
const TENTH_Q32: i64 = 429_496_730;

/// `0.45 = 0.9/2`, `round(0.45·2^32)`.
const COS_AMPL_Q32: i64 = 1_932_735_283;

/// `round(π·2^32)`, the angle of one half cosine period.
const PI_Q32: i64 = 13_493_037_705;

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
    /// A value does not fit the Q32.32 carrier.
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

/// The step's learning rate `lr·factor` as a Q32.32 word: the rate is carried at Q64.64, so the
/// product is one rounded quotient by `2^64` onto the grid. `O(1)`.
fn lr_word(base_q64: i64, factor: i64) -> Result<i64, TrainError> {
    narrow(div_rne_pos(
        i128::from(base_q64) * i128::from(factor),
        1_i128 << 64,
    ))
    .ok_or_else(|| TrainError::Range("learning rate".to_string()))
}

/// `(o_total, n)` of linear `k`: `q` is `q_out × n_embd`, `k`/`v` are `kv_out × n_embd`, `o` is
/// `n_embd × q_out`, `gate`/`up` are `n_ff × n_embd` and `down` is `n_embd × n_ff`.
///
/// # Errors
/// [`TrainError::Shape`] past [`N_KINDS`]; [`TrainError::Model`] from the `q_out`/`kv_out` products.
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

/// The `k`th weight gradient of a layer, in [`KIND_NAMES`] order. `O(1)`.
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

/// `round(a[i] / t)` over a Q32.32 buffer; an empty result signals a refused division, serial
/// `O(a.len())`.
fn over(a: &[i64], t: i64) -> Vec<i64> {
    if a.is_empty() || t <= 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(a.len());
    for &v in a {
        match div_ne(i128::from(v), i128::from(t)).and_then(narrow) {
            Some(w) => out.push(w),
            None => return Vec::new(),
        }
    }
    out
}

/// `dst[j] = (a[j]/t)·b[j]·exp(c[j])` (a norm gain's latent gradient), Q32.32; an empty result
/// signals a length, `exp` or word refusal. Serial `O(len)`.
fn norm_grad(a: &[i64], b: &[i64], c: &[i64], t: i64) -> Vec<i64> {
    let len = a.len().min(b.len()).min(c.len());
    if len == 0 || t <= 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let q = div_ne(i128::from(a[i]), i128::from(t)).and_then(narrow);
        let e = fixed_exp(c[i]).ok();
        match q
            .zip(e)
            .and_then(|(q, e)| mul(q, b[i]).and_then(|x| mul(x, e)))
        {
            Some(v) => out.push(v),
            None => return Vec::new(),
        }
    }
    out
}

/// `dst[j] = (a[j]/t)·s` (a bias latent's gradient), Q32.32; an empty result signals a refusal,
/// serial `O(len)`.
fn bias_grad(a: &[i64], s: i64, t: i64) -> Vec<i64> {
    if a.is_empty() || t <= 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(a.len());
    for &v in a {
        match div_ne(i128::from(v), i128::from(t))
            .and_then(narrow)
            .and_then(|q| mul(q, s))
        {
            Some(w) => out.push(w),
            None => return Vec::new(),
        }
    }
    out
}

/// `Σ|base[i]|` in `i128`, serial `O(n)`.
fn abs_sum(base: &[i64]) -> i128 {
    let mut acc = 0_i128;
    let mut i = 0;
    while i < base.len() {
        acc += i128::from(base[i]).abs();
        i += 1;
    }
    acc
}

/// The bias scale `max(mean|base|, 1e-3)` of one zero-bit bias vector, Q32.32; an empty base gives
/// the floor.
///
/// # Complexity
/// Temporal `O(base.len())`, serial; spatial `O(1)`.
fn bias_scale(base: &[i64]) -> i64 {
    if base.is_empty() {
        return BIAS_SCALE_MIN_Q32;
    }
    let mean = div_ne(abs_sum(base), base.len() as i128).unwrap_or(0);
    match narrow(mean) {
        Some(v) if v > BIAS_SCALE_MIN_Q32 => v,
        _ => BIAS_SCALE_MIN_Q32,
    }
}

/// One coded linear: the planes whose weights sum into its stored weight, the optional frozen dense
/// addend, and the expected `o_total × n` shape.
#[derive(Clone, Debug)]
pub struct Linear {
    /// The frozen-code planes, summed in order by [`CodedPlane::add_into`].
    pub planes: Vec<CodedPlane>,
    /// An optional frozen dense addend, row-major `o_total × n` Q32.32 (the corrector on `q`/`down`).
    pub extra: Option<Vec<i64>>,
    /// Out rows of the stored weight.
    pub o_total: usize,
    /// In columns of the stored weight.
    pub n: usize,
}

impl Linear {
    /// An empty linear of the given shape. `O(1)`.
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
    /// Wrap a per-layer, per-kind table. `O(1)`.
    pub fn new(layers: Vec<Vec<Linear>>) -> Self {
        Self { layers }
    }

    /// Linear `k` of layer `l`.
    ///
    /// # Errors
    /// [`TrainError::Shape`] if `l` or `k` is past the table. `O(1)`.
    pub fn linear(&self, l: usize, k: usize) -> Result<&Linear, TrainError> {
        if k >= N_KINDS {
            return Err(TrainError::Shape(format!("linear kind {k} >= {N_KINDS}")));
        }
        match self.layers.get(l) {
            Some(row) => match row.get(k) {
                Some(lin) => Ok(lin),
                None => Err(TrainError::Shape(format!(
                    "plane layer {l} kind {k} absent"
                ))),
            },
            None => Err(TrainError::Shape(format!(
                "plane layer {l} kind {k} absent"
            ))),
        }
    }

    /// Linear `k` of layer `l`, mutable.
    ///
    /// # Errors
    /// As [`Planes::linear`]. `O(1)`.
    pub fn linear_mut(&mut self, l: usize, k: usize) -> Result<&mut Linear, TrainError> {
        if k >= N_KINDS {
            return Err(TrainError::Shape(format!("linear kind {k} >= {N_KINDS}")));
        }
        match self.layers.get_mut(l) {
            Some(row) => match row.get_mut(k) {
                Some(lin) => Ok(lin),
                None => Err(TrainError::Shape(format!(
                    "plane layer {l} kind {k} absent"
                ))),
            },
            None => Err(TrainError::Shape(format!(
                "plane layer {l} kind {k} absent"
            ))),
        }
    }
}

/// One layer's resolved zero-bit values: `gamma = base · exp(d)` for the two norm gains, and
/// `b = base + d · max(mean|base|, 1e-3)` for the three attention biases. Q32.32.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerTuned {
    /// `attn_norm` gain, `n_embd` wide.
    pub ln1: Vec<i64>,
    /// `ffn_norm` gain, `n_embd` wide.
    pub ln2: Vec<i64>,
    /// `attn_q.bias`, `n_head·head_dim` wide.
    pub qb: Vec<i64>,
    /// `attn_k.bias`, `n_head_kv·head_dim` wide.
    pub kb: Vec<i64>,
    /// `attn_v.bias`, `n_head_kv·head_dim` wide.
    pub vb: Vec<i64>,
}

/// One layer's zero-bit bases and their trainable latents; latents and bases are Q32.32.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerTunables {
    /// `attn_norm` base gain, `n_embd` wide.
    pub ln1: Vec<i64>,
    /// `attn_norm` log latent; `gamma = ln1 · exp(ln1_d)`.
    pub ln1_d: Vec<i64>,
    /// `ffn_norm` base gain, `n_embd` wide.
    pub ln2: Vec<i64>,
    /// `ffn_norm` log latent.
    pub ln2_d: Vec<i64>,
    /// `attn_q.bias` base.
    pub qb: Vec<i64>,
    /// `attn_q.bias` additive latent, scaled by [`LayerTunables::bias_scale`].
    pub qb_d: Vec<i64>,
    /// `attn_k.bias` base.
    pub kb: Vec<i64>,
    /// `attn_k.bias` additive latent.
    pub kb_d: Vec<i64>,
    /// `attn_v.bias` base.
    pub vb: Vec<i64>,
    /// `attn_v.bias` additive latent.
    pub vb_d: Vec<i64>,
}

impl LayerTunables {
    /// Build a layer's tunables from the frozen base words, every latent zero.
    ///
    /// `O(n_embd + q_out + kv_out)`, serial.
    pub fn new(ln1: &[i64], ln2: &[i64], qb: &[i64], kb: &[i64], vb: &[i64]) -> Self {
        let q_ln1 = ln1.to_vec();
        let q_ln2 = ln2.to_vec();
        let q_qb = qb.to_vec();
        let q_kb = kb.to_vec();
        let q_vb = vb.to_vec();
        Self {
            ln1_d: vec![0_i64; q_ln1.len()],
            ln2_d: vec![0_i64; q_ln2.len()],
            qb_d: vec![0_i64; q_qb.len()],
            kb_d: vec![0_i64; q_kb.len()],
            vb_d: vec![0_i64; q_vb.len()],
            ln1: q_ln1,
            ln2: q_ln2,
            qb: q_qb,
            kb: q_kb,
            vb: q_vb,
        }
    }

    /// The frozen bias scale of `qb`/`kb`/`vb` (`0`/`1`/`2`); `O(width)`, serial.
    pub fn bias_scale(&self, which: usize) -> i64 {
        match which {
            0 => bias_scale(&self.qb),
            1 => bias_scale(&self.kb),
            _ => bias_scale(&self.vb),
        }
    }

    /// The resolved gains and biases `(ln1, ln2, qb, kb, vb)`; a refused `exp` leaves an empty
    /// vector. `O(n_embd + q_out + kv_out)`, serial.
    pub fn tuned(&self) -> LayerTuned {
        LayerTuned {
            ln1: scale_exp(&self.ln1, &self.ln1_d),
            ln2: scale_exp(&self.ln2, &self.ln2_d),
            qb: add_scaled(&self.qb, &self.qb_d, bias_scale(&self.qb)),
            kb: add_scaled(&self.kb, &self.kb_d, bias_scale(&self.kb)),
            vb: add_scaled(&self.vb, &self.vb_d, bias_scale(&self.vb)),
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
    pub fn_base: Vec<i64>,
    /// `output_norm` log latent.
    pub fn_d: Vec<i64>,
}

impl Tunables {
    /// Build the tunables from the frozen base words, every latent zero.
    ///
    /// `O(n_layer·(n_embd + q_out + kv_out) + n_embd)`, serial.
    pub fn new(layers: Vec<LayerTunables>, fn_base: &[i64]) -> Self {
        let base = fn_base.to_vec();
        Self {
            fn_d: vec![0_i64; base.len()],
            layers,
            fn_base: base,
        }
    }

    /// The final `output_norm` gain `fn_base · exp(fn_d)`; `O(n_embd)`, serial.
    pub fn output_norm(&self) -> Vec<i64> {
        scale_exp(&self.fn_base, &self.fn_d)
    }
}

/// The frozen model tensors outside the coded planes: the token embedding and the head, both
/// row-major `vocab × n_embd` in Q32.32.
#[derive(Clone, Debug, PartialEq)]
pub struct Frozen {
    /// `token_embd.weight`.
    pub token_embd: Vec<i64>,
    /// `output.weight` (or the tied `token_embd.weight`).
    pub lm_head: Vec<i64>,
}

impl Frozen {
    /// Take the frozen embedding and head words.
    ///
    /// `O(vocab·n_embd)`, serial.
    pub fn new(token_embd: &[i64], lm_head: &[i64]) -> Self {
        Self {
            token_embd: token_embd.to_vec(),
            lm_head: lm_head.to_vec(),
        }
    }
}

/// The step-counted cosine schedule
/// `f(step) = min(1, step/warmup) · (0.1 + 0.9·(1 + cos(π·step/steps))/2)`; the Python's `frac`
/// is wall-clock floored at `step/steps`, and the step fraction alone is deterministic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// Linear ramp length in steps.
    pub warmup: u64,
    /// Cosine period in steps.
    pub steps: u64,
}

impl Schedule {
    /// The schedule with the given ramp and period. `O(1)`.
    pub fn new(warmup: u64, steps: u64) -> Self {
        Self { warmup, steps }
    }

    /// The learning-rate factor at `step` as a Q32.32 word. The operands are bounded, so no refusal
    /// is reachable; a refused word returns `0`. `O(1)`.
    pub fn factor(&self, step: u64) -> i64 {
        // `step/den` in Q32.32; a zero denominator is the full fraction.
        let wu = if self.warmup == 0 {
            ONE
        } else {
            match div_ne(i128::from(step) << 32, i128::from(self.warmup)) {
                Some(q) => match narrow(q) {
                    Some(v) => v,
                    None => ONE,
                },
                None => ONE,
            }
        };
        let wu = wu.clamp(0, ONE);
        let frac = if self.steps == 0 {
            ONE
        } else {
            match div_ne(i128::from(step) << 32, i128::from(self.steps)) {
                Some(q) => match narrow(q) {
                    Some(v) => v,
                    None => ONE,
                },
                None => ONE,
            }
        };
        let angle = match mul(PI_Q32, frac) {
            Some(v) => v,
            None => return 0,
        };
        let sc = match fixed_sin_cos(angle) {
            Ok(v) => v,
            Err(_) => return 0,
        };
        let one_cos = match ONE.checked_add(sc.cos) {
            Some(v) => v,
            None => return 0,
        };
        let term = match mul(COS_AMPL_Q32, one_cos) {
            Some(v) => v,
            None => return 0,
        };
        match TENTH_Q32.checked_add(term) {
            Some(t) => match mul(wu, t) {
                Some(v) => v,
                None => 0,
            },
            None => 0,
        }
    }
}

/// One parameter vector's Adam state (torch defaults: `beta = (0.9, 0.999)`, `eps = 1e-8`, no
/// weight decay): the first moment is Q32.32, the second `Q64.64` in `i128`.
///
/// The second moment carries twice the fractional width because `v = (1−β₂)·g²` squares the
/// gradient: at `Q32.32` a gradient below `2^-16` rounds `v` to zero and the update degenerates to
/// `lr·m̂/eps`. Taking the square root from the `Q64.64` word keeps `Q32.32` precision down to
/// gradients of order `2^-32`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Adam {
    m: Vec<i64>,
    v: Vec<i128>,
    t: u64,
}

impl Adam {
    /// Zero state for a parameter vector of `n` elements. `O(n)`.
    pub fn zeros(n: usize) -> Self {
        Self {
            m: vec![0_i64; n],
            v: vec![0_i128; n],
            t: 0,
        }
    }

    /// The number of parameters this state covers. `O(1)`.
    pub fn len(&self) -> usize {
        self.m.len()
    }

    /// Whether the state covers no parameter. `O(1)`.
    pub fn is_empty(&self) -> bool {
        self.m.is_empty()
    }

    /// One Adam update in place: `m ← β₁m + (1−β₁)g`, `v ← β₂v + (1−β₂)g²`,
    /// `p ← p − lr·m̂/(√v̂ + ε)`. Every operand is a Q32.32 word.
    ///
    /// # Errors
    /// [`TrainError::Shape`] on a width disagreement; [`TrainError::Range`] on a refused word.
    ///
    /// # Complexity
    /// Temporal `O(p.len())` serial (three words per element, below the dispatch threshold);
    /// spatial `O(1)` in place.
    pub fn apply(&mut self, lr: i64, p: &mut [i64], g: &[i64]) -> Result<(), TrainError> {
        if p.len() != g.len() || p.len() != self.m.len() || p.len() != self.v.len() {
            return Err(TrainError::Shape(format!(
                "adam width {} vs {} vs {}",
                p.len(),
                g.len(),
                self.m.len()
            )));
        }
        self.t = self.t.saturating_add(1);
        let b1 = match pow_q32(BETA1_Q32, self.t) {
            Some(v) => v,
            None => return Err(TrainError::Range("beta_1^t".to_string())),
        };
        let b2 = match pow_q32(BETA2_Q32, self.t) {
            Some(v) => v,
            None => return Err(TrainError::Range("beta_2^t".to_string())),
        };
        let bc1 = match ONE.checked_sub(b1) {
            Some(v) => v,
            None => return Err(TrainError::Range("1 - beta_1^t".to_string())),
        };
        let bc2 = match ONE.checked_sub(b2) {
            Some(v) => v,
            None => return Err(TrainError::Range("1 - beta_2^t".to_string())),
        };
        if bc2 <= 0 {
            return Err(TrainError::Range("1 - beta_2^t".to_string()));
        }
        // `√(1 − β₂^t)` in Q32.32, so the denominator is `√v/√bc2 + ε` rather than
        // `√(v/bc2) + ε`: the division is deferred until after the square root, where the
        // exponent no longer underflows.
        let root_bc2 = match narrow(isqrt(i128::from(bc2) << 32)) {
            Some(v) => v,
            None => return Err(TrainError::Range("sqrt(1 - beta_2^t)".to_string())),
        };
        let mut err: Option<TrainError> = None;
        let mut i = 0;
        while i < p.len() {
            let gi = g[i];
            let mi = match ok(mul(BETA1_Q32, self.m[i])) {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            let m2 = match ok(mul(ONE_MINUS_BETA1_Q32, gi)) {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            let mi = match ok(mi.checked_add(m2)) {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            self.m[i] = mi;
            // Second moment in Q64.64: `β₂·v + (1−β₂)·g²`, where `g²` is the exact `i128` square
            // of the Q32.32 word.
            let g2 = i128::from(gi) * i128::from(gi);
            let t1 = match ok(i128::from(BETA2_Q32)
                .checked_mul(self.v[i])
                .and_then(|x| rshift_rne(x, 32)))
            {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            let t2 = match ok(i128::from(ONE_MINUS_BETA2_Q32)
                .checked_mul(g2)
                .and_then(|x| rshift_rne(x, 32)))
            {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            let vi = match ok(t1.checked_add(t2)) {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            if vi < 0 {
                err = Some(TrainError::Range("adam second moment".to_string()));
                break;
            }
            self.v[i] = vi;
            let mhat = match ok(div(mi, bc1)) {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            // `√v̂ = √v / √bc2`, both square roots in Q32.32 from the Q64.64 word.
            let root_v = match ok(narrow(isqrt(vi))) {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            let root = match ok(div(root_v, root_bc2)) {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            let denom = match ok(root.checked_add(ADAM_EPS_Q32)) {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            let num = match ok(mul(lr, mhat)) {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            let step = if denom == 0 {
                0
            } else {
                match ok(div(num, denom)) {
                    Ok(v) => v,
                    Err(e) => {
                        err = Some(e);
                        break;
                    }
                }
            };
            let next = match ok(p[i].checked_sub(step)) {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            };
            p[i] = next;
            i += 1;
        }
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// `result` or [`TrainError::Range`], for the checked-word chains above.
fn ok<T>(v: Option<T>) -> Result<T, TrainError> {
    match v {
        Some(x) => Ok(x),
        None => Err(TrainError::Range("q32 word".to_string())),
    }
}

/// [`TrainError::Shape`] for an absent table entry `what` at `i`.
fn absent(what: &str, i: usize) -> TrainError {
    TrainError::Shape(format!("{what} {i} absent"))
}

/// `base^e` in Q32.32 by exponentiation by squaring; `O(log e)`.
fn pow_q32(base: i64, mut e: u64) -> Option<i64> {
    let mut acc = ONE;
    let mut b = base;
    let mut bad = false;
    while e > 0 && !bad {
        if e & 1 == 1 {
            match mul(acc, b) {
                Some(v) => acc = v,
                None => bad = true,
            }
        }
        e >>= 1;
        if e > 0 && !bad {
            match mul(b, b) {
                Some(v) => b = v,
                None => bad = true,
            }
        }
    }
    if bad {
        None
    } else {
        Some(acc)
    }
}

/// `base[i]·exp(off[i])` over `min` of the two lengths; an empty vector signals a refusal, serial
/// `O(len)`.
fn scale_exp(base: &[i64], off: &[i64]) -> Vec<i64> {
    let len = base.len().min(off.len());
    if len == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(len);
    let mut bad = false;
    let mut i = 0;
    while i < len && !bad {
        match fixed_exp(off[i]).ok().and_then(|e| mul(base[i], e)) {
            Some(v) => out.push(v),
            None => bad = true,
        }
        i += 1;
    }
    if bad {
        return Vec::new();
    }
    out
}

/// `base[i] + off[i]·s`; an empty vector signals a word refusal, serial `O(len)`.
fn add_scaled(base: &[i64], off: &[i64], s: i64) -> Vec<i64> {
    let len = base.len().min(off.len());
    if len == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(len);
    let mut bad = false;
    let mut i = 0;
    while i < len && !bad {
        match mul(off[i], s).and_then(|v| base[i].checked_add(v)) {
            Some(v) => out.push(v),
            None => bad = true,
        }
        i += 1;
    }
    if bad {
        return Vec::new();
    }
    out
}

/// The summed plane weight of one linear plus its optional frozen dense addend, in Q32.32;
/// `quantized` selects the stored [`CodedPlane::weight`] or the differentiable
/// [`CodedPlane::weight_unquantized`].
///
/// # Errors
/// [`TrainError::Shape`] on a plane or `extra` width, [`TrainError::Plane`] from a plane,
/// [`TrainError::Range`] on a plane refusal; `O(Σ_p o_p·n·log2(GROUP) + o_total·n)`, the plane
/// tensors pooled.
fn linear_weight(lin: &Linear, quantized: bool) -> Result<Vec<i64>, TrainError> {
    let on = lin.o_total.checked_mul(lin.n).ok_or(TrainError::Overflow)?;
    let mut w = vec![0_i64; on];
    for p in &lin.planes {
        if p.n != lin.n || p.o == 0 || !p.n.is_multiple_of(GROUP) {
            let m = format!("plane {}x{} in linear {}x{}", p.o, p.n, lin.o_total, lin.n);
            return Err(TrainError::Shape(m));
        }
        let pw = if quantized {
            p.weight()
        } else {
            p.weight_unquantized()
        };
        if pw.len() != p.o * p.n {
            return Err(TrainError::Range(format!(
                "plane {}x{} weight refused",
                p.o, p.n
            )));
        }
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
        for (d, &v) in w.iter_mut().zip(e) {
            *d = d.checked_add(v).ok_or(TrainError::Overflow)?;
        }
    }
    Ok(w)
}

/// The `LayerWeights` of every kind of one layer; `O(Σ_k (o_k·n_k·log2(GROUP) + o_k·n_k))`, the
/// kind loop serial over each weight's pooled plane tensors.
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
        weights.push(linear_weight(lin, quantized)?);
    }
    let t = tun.tuned();
    let [wq, wk, wv, wo, w_gate, w_up, w_down] = <[Vec<i64>; N_KINDS]>::try_from(weights)
        .map_err(|_| TrainError::Shape(format!("layer {l} kinds absent")))?;
    Ok(LayerWeights {
        attn_norm: t.ln1,
        wq,
        bq: Some(t.qb),
        wk,
        bk: Some(t.kb),
        wv,
        bv: Some(t.vb),
        wo,
        ffn_norm: t.ln2,
        w_gate,
        w_up,
        w_down,
    })
}

/// Assemble the whole-model Q32.32 [`Weights`]: the stored forward weight of every coded linear
/// plus the tuned norms and biases and the frozen embedding and head.
///
/// # Errors
/// [`TrainError::Shape`] on a plane or tunable shape, [`TrainError::Plane`] from a plane,
/// [`TrainError::Range`] outside the carrier; `O(n_layer·Σ_k o_k·n_k·log2(GROUP) + vocab·n_embd)`,
/// the layer loop serial over the pooled plane tensors.
pub fn assemble(
    cfg: &Config,
    planes: &Planes,
    tun: &Tunables,
    frozen: &Frozen,
) -> Result<Weights, TrainError> {
    assemble_with(cfg, planes, tun, frozen, true)
}

/// As [`assemble`], over the differentiable [`CodedPlane::weight_unquantized`] forward.
///
/// # Errors
/// As [`assemble`].
pub fn assemble_unquantized(
    cfg: &Config,
    planes: &Planes,
    tun: &Tunables,
    frozen: &Frozen,
) -> Result<Weights, TrainError> {
    assemble_with(cfg, planes, tun, frozen, false)
}

/// [`assemble`]'s shared body. `O(assemble)`.
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
    let output_norm = tun.output_norm();
    if output_norm.len() != e {
        return Err(TrainError::Range("output_norm refused".to_string()));
    }
    // The layer loop is structural: each layer's kinds and planes dispatch their own tensor work to
    // the pool, and the pool is not re-entrant (a task that dispatches deadlocks the dispatcher).
    let mut layers: Vec<LayerWeights> = Vec::with_capacity(cfg.n_layer);
    for l in 0..cfg.n_layer {
        let lt = tun
            .layers
            .get(l)
            .ok_or_else(|| TrainError::Shape(format!("tunable layer {l} absent")))?;
        layers.push(assemble_layer(cfg, planes, lt, l, quantized)?);
    }
    Ok(Weights {
        layers,
        output_norm,
        token_embd: frozen.token_embd.clone(),
        lm_head: frozen.lm_head.clone(),
    })
}

/// The batch mean KL of the teacher's distribution against the student's Q32.32 logits, with the
/// row-summed logit gradient `dL/dz` (`t · vocab`, Q32.32): `mean_kl` is the Q32.32 mean word, `dz`
/// the sum over rows of `p_s − p_t`. Serial `O(t·vocab)`.
///
/// # Errors
/// [`TrainError::Shape`] for a length that is not `t·vocab` with `t > 0`; [`TrainError::Head`] from
/// [`kl_row`]; [`TrainError::Range`] when the summed loss leaves `i64`.
pub fn kl_mean(
    cfg: &Config,
    logits: &[i64],
    teacher: &[i64],
) -> Result<(i64, Vec<i64>), TrainError> {
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
    let mut acc = 0_i128;
    let mut err: Option<TrainError> = None;
    let mut r = 0;
    while r < t && err.is_none() {
        let base = r * v;
        match kl_row(&logits[base..base + v], &teacher[base..base + v]) {
            Ok((loss, g)) => match acc.checked_add(i128::from(loss)) {
                Some(a) => {
                    acc = a;
                    dz[base..base + v].copy_from_slice(&g[..v]);
                }
                None => err = Some(TrainError::Overflow),
            },
            Err(e) => err = Some(TrainError::Head(e)),
        }
        r += 1;
    }
    match err {
        Some(e) => return Err(e),
        None => {}
    }
    let mean = div_ne(acc, t as i128).ok_or(TrainError::Overflow)?;
    let mean = narrow(mean).ok_or(TrainError::Overflow)?;
    Ok((mean, dz))
}

/// The gradients of the batch mean KL with respect to one layer's zero-bit latents, Q32.32.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayerNormGrad {
    /// `∂L/∂ attn_norm` log latent.
    pub ln1: Vec<i64>,
    /// `∂L/∂ ffn_norm` log latent.
    pub ln2: Vec<i64>,
    /// `∂L/∂ attn_q.bias` additive latent.
    pub qb: Vec<i64>,
    /// `∂L/∂ attn_k.bias` additive latent.
    pub kb: Vec<i64>,
    /// `∂L/∂ attn_v.bias` additive latent.
    pub vb: Vec<i64>,
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
    pub fn_d: Vec<i64>,
    /// The batch mean KL the gradients are taken at, Q32.32.
    pub mean_kl: i64,
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
    /// The learning rate's schedule, the Adam state per parameter and the step count.
    pub schedule: Schedule,
    moments: Vec<Vec<Vec<PlaneMoments>>>,
    norm_moments: Vec<[Adam; 5]>,
    fn_moment: Adam,
    step: u64,
}

/// The Adam state of one plane's three latent axes.
#[derive(Clone, Debug, PartialEq)]
struct PlaneMoments {
    pa: Adam,
    pb: Adam,
    pd: Adam,
}

impl PlaneMoments {
    /// Zero state sized to the plane's three latents; `O(o·n/128 + n)`.
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
    /// [`TrainError::Shape`] on a disagreeing plane, tunable or frozen tensor;
    /// [`TrainError::Model`] from [`Config::checked`]. `O(n_layer·Σ_k o_k·n_k/128)`, serial.
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
        let moments = planes
            .layers
            .iter()
            .map(|row| {
                row.iter()
                    .map(|lin| lin.planes.iter().map(PlaneMoments::zeros).collect())
                    .collect()
            })
            .collect();
        let norm_moments = tun
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

    /// The completed updates. `O(1)`.
    pub fn step_index(&self) -> u64 {
        self.step
    }

    /// The batch mean KL at the current latents over the stored forward, as a Q32.32 word.
    ///
    /// # Errors
    /// As [`assemble`], [`forward`] and [`kl_mean`]; cost as [`Finisher::gradients`] without the
    /// backward.
    pub fn loss(&self, tokens: &[u32], teacher: &[i64]) -> Result<i64, TrainError> {
        let w = assemble(&self.cfg, &self.planes, &self.tun, &self.frozen)?;
        let (logits, _) = forward(&w, &self.cfg, tokens)?;
        Ok(kl_mean(&self.cfg, &logits, teacher)?.0)
    }

    /// The gradients of the batch mean KL with respect to every trainable latent; nothing changes.
    ///
    /// # Errors
    /// [`TrainError::Shape`] on a teacher/logits geometry mismatch; [`TrainError::Model`],
    /// [`TrainError::Head`], [`TrainError::Plane`] from the forward, head and plane backward;
    /// [`TrainError::Overflow`]/[`TrainError::Range`] on a checked accumulate or word.
    ///
    /// # Complexity
    /// Per layer one forward and backward `O(t·(n_embd² + n_embd·n_ff) + n_head·t²·head_dim)` plus
    /// the head `O(t·vocab·n_embd)` and the latent map `O(Σ plane o·n·log2(GROUP))`; the tensor
    /// passes are the plane's pooled ones, the layer loop serial.
    pub fn gradients(&self, tokens: &[u32], teacher: &[i64]) -> Result<Gradients, TrainError> {
        let w = assemble(&self.cfg, &self.planes, &self.tun, &self.frozen)?;
        let (logits, tape) = forward(&w, &self.cfg, tokens)?;
        let (mean_kl, dz) = kl_mean(&self.cfg, &logits, teacher)?;
        let grads: Grads = backward(&w, &self.cfg, &tape, &dz)?;
        let t = tokens.len();
        if t == 0 {
            return Err(TrainError::Shape("empty batch".to_string()));
        }
        let rows = i64::try_from(t).map_err(|_| TrainError::Overflow)?;
        let e = self.cfg.n_embd;
        let mut planes: Vec<Vec<Vec<LatentGrad>>> =
            vec![vec![Vec::new(); N_KINDS]; self.cfg.n_layer];
        let mut norms = vec![LayerNormGrad::default(); self.cfg.n_layer];
        // Structural layer loop: the plane backwards pool, and the pool is not re-entrant.
        for l in 0..self.cfg.n_layer {
            let (per_kind, ln) = layer_grads(&grads, &self.tun, &self.planes, l, rows)?;
            planes[l] = per_kind;
            norms[l] = ln;
        }
        let fn_d = norm_grad(&grads.output_norm, &self.tun.fn_base, &self.tun.fn_d, rows);
        if fn_d.len() != e {
            return Err(TrainError::Range("output_norm gradient".to_string()));
        }
        Ok(Gradients {
            planes,
            norms,
            fn_d,
            mean_kl,
        })
    }

    /// One Adam update of every latent at the current step's factor word; nothing changes on a
    /// shape disagreement.
    ///
    /// # Errors
    /// [`TrainError::Shape`] on a gradient width, [`TrainError::Range`] on a refused gradient or
    /// word. `O(Σ plane (o·n/128 + n) + n_layer·(n_embd + q_out + kv_out))`, serial, in place.
    pub fn apply(&mut self, g: &Gradients, factor: i64) -> Result<(), TrainError> {
        let lr_lat = lr_word(LR_LAT_Q64, factor)?;
        let lr_norm = lr_word(LR_NORM_Q64, factor)?;
        for l in 0..self.cfg.n_layer {
            let row = self
                .planes
                .layers
                .get_mut(l)
                .ok_or_else(|| absent("plane layer", l))?;
            let gk = g
                .planes
                .get(l)
                .ok_or_else(|| absent("grad plane layer", l))?;
            let ml = self
                .moments
                .get_mut(l)
                .ok_or_else(|| absent("moment layer", l))?;
            for (k, lin) in row.iter_mut().enumerate() {
                let gp = gk.get(k).ok_or_else(|| absent("grad kind", k))?;
                let mk = ml.get_mut(k).ok_or_else(|| absent("moment kind", k))?;
                for (i, p) in lin.planes.iter_mut().enumerate() {
                    let (dpa, dpb, dpd) = gp.get(i).ok_or_else(|| absent("grad plane", i))?;
                    let m = mk.get_mut(i).ok_or_else(|| absent("moment plane", i))?;
                    m.pa.apply(lr_lat, &mut p.pa, dpa)?;
                    m.pb.apply(lr_lat, &mut p.pb, dpb)?;
                    m.pd.apply(lr_lat, &mut p.pd, dpd)?;
                }
            }
            let gn = g.norms.get(l).ok_or_else(|| absent("grad norm layer", l))?;
            let lt = self
                .tun
                .layers
                .get_mut(l)
                .ok_or_else(|| absent("tunable layer", l))?;
            let ml = self
                .norm_moments
                .get_mut(l)
                .ok_or_else(|| absent("norm moment layer", l))?;
            ml[0].apply(lr_norm, &mut lt.ln1_d, &gn.ln1)?;
            ml[1].apply(lr_norm, &mut lt.ln2_d, &gn.ln2)?;
            ml[2].apply(lr_norm, &mut lt.qb_d, &gn.qb)?;
            ml[3].apply(lr_norm, &mut lt.kb_d, &gn.kb)?;
            ml[4].apply(lr_norm, &mut lt.vb_d, &gn.vb)?;
        }
        self.fn_moment.apply(lr_norm, &mut self.tun.fn_d, &g.fn_d)?;
        Ok(())
    }

    /// One finishing P-step: the gradients, then one Adam update at the step's factor.
    ///
    /// # Errors
    /// As [`Finisher::gradients`] and [`Finisher::apply`]; cost as those plus
    /// `O(Σ plane (o·n/128 + n))`.
    pub fn step(&mut self, tokens: &[u32], teacher: &[i64]) -> Result<i64, TrainError> {
        let g = self.gradients(tokens, teacher)?;
        self.step += 1;
        let f = self.schedule.factor(self.step);
        self.apply(&g, f)?;
        Ok(g.mean_kl)
    }
}

/// The plane and zero-bit latent gradients of one layer, Q32.32.
///
/// # Errors
/// [`TrainError::Shape`] on an absent gradient or width, [`TrainError::Range`] on a refusal.
/// `O(Σ plane o·n·log2(GROUP) + n_embd + q_out + kv_out)`, the per-plane backward pooled.
fn layer_grads(
    grads: &Grads,
    tun: &Tunables,
    planes: &Planes,
    l: usize,
    rows: i64,
) -> Result<(Vec<Vec<LatentGrad>>, LayerNormGrad), TrainError> {
    let e = tun
        .layers
        .get(l)
        .ok_or_else(|| absent("tunable layer", l))?
        .ln1
        .len();
    let lg = grads.layers.get(l).ok_or_else(|| absent("grad layer", l))?;
    let lt = &tun.layers[l];
    let row = planes
        .layers
        .get(l)
        .ok_or_else(|| absent("plane layer", l))?;
    let mut per_kind: Vec<Vec<LatentGrad>> = Vec::with_capacity(N_KINDS);
    for (k, lin) in row.iter().enumerate() {
        let gw = kind_grad(lg, k);
        let gf = over(gw, rows);
        if gf.len() != gw.len() {
            return Err(TrainError::Range(format!(
                "layer {l} kind {k} gradient refused"
            )));
        }
        let mut per_plane: Vec<LatentGrad> = Vec::with_capacity(lin.planes.len());
        for p in &lin.planes {
            per_plane.push(p.backward(&gf)?);
        }
        per_kind.push(per_plane);
    }
    let dn = LayerNormGrad {
        ln1: norm_grad(&lg.attn_norm, &lt.ln1, &lt.ln1_d, rows),
        ln2: norm_grad(&lg.ffn_norm, &lt.ln2, &lt.ln2_d, rows),
        qb: bias_grad(&lg.bq, lt.bias_scale(0), rows),
        kb: bias_grad(&lg.bk, lt.bias_scale(1), rows),
        vb: bias_grad(&lg.bv, lt.bias_scale(2), rows),
    };
    if dn.ln1.len() != e
        || dn.ln2.len() != e
        || dn.qb.len() != lt.qb.len()
        || dn.kb.len() != lt.kb.len()
        || dn.vb.len() != lt.vb.len()
    {
        return Err(TrainError::Range(format!(
            "layer {l} norm gradient refused"
        )));
    }
    Ok((per_kind, dn))
}
