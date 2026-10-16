//! Output-side output-gradient Gram (`HO`) capture: the per-linear scatter `Σ_t w_t g_t g_tᵀ` of the
//! sampled-label loss, the Schäfer–Strimmer inputs `S4`/`Q`, and the mean-diagonal-1 finish.
//!
//! [`Hocapture::accumulate`] runs [`crate::learn::grad::model::forward`], draws `K` sampled targets
//! per token from the teacher softmax, and for each draw runs [`backward_with_taps`] once: the whole
//! backward of [`crate::learn::grad::model::backward`], returning the weight gradients *and* the
//! gradient of the sampled-label loss with respect to every linear's **output** at every token.
//! [`accumulate_kind`] folds one such `(input, output-gradient)` pair into
//! `HO = Σ_t w_t g_t g_tᵀ`, `S4 = Σ_t ‖s_t‖⁴` and `Q_i = Σ_t s_ti⁴` with `s_ti = √w_t · g_ti` and
//! `w_t = ‖x_t‖²` the squared norm of the linear's input row. The port is the `outer=True` half of
//! the reference `gq_capture` (`/home/socol/oc-tmp/lanes/port/spec/fold_transfer.py:1163-1341`) and
//! its `gq_stats` consumer (`:1343-1354`); the reference names the target as YAQA Sketch A, Eq. 11
//! (Tseng et al., arXiv 2505.22988), one round from `(H_I)₀ = I`.
//!
//! # The seven kinds and the four stage groups
//!
//! The reference keys `HO`, `S4`, `Q` and `ho_lw` per layer and per **linear**, `KINDS = {q, k, v,
//! o, gate, up, down}` (`fold_transfer.py:47-48`), and groups them into the four stage names of
//! `GQ_STAGE_OF` (`:1095`): `qkv`, `o`, `ffn` and `down`. This module keeps the reference's
//! per-linear granularity — the Schäfer–Strimmer inputs are per linear, so the grouped statistic is
//! not the same number — and exposes the grouping as [`Kind::stage`]. A grouped `HO` for `qkv` is
//! the block diagonal of the `q`, `k` and `v` matrices, because their output gradients are disjoint
//! coordinates of one concatenated vector.
//!
//! | [`Kind`] | linear | output gradient tapped | input weighting `w_t` | width `O` |
//! |---|---|---|---|---|
//! | `Q` | `attn_q` (pre-`RoPE` output) | `RoPE` adjoint of the attention query adjoint | `ln1_out` | `q_out` |
//! | `K` | `attn_k` (pre-`RoPE` output) | `RoPE` adjoint of the attention key adjoint | `ln1_out` | `kv_out` |
//! | `V` | `attn_v` | attention value adjoint | `ln1_out` | `kv_out` |
//! | `O` | `attn_output` | `∂L/∂o` at the residual split `h2 = h + o` | `attn_out` | `q_out` |
//! | `Gate` | `ffn_gate` | `SwiGLU` gate adjoint | `ln2_out` | `n_ff` |
//! | `Up` | `ffn_up` | `SwiGLU` up adjoint | `ln2_out` | `n_ff` |
//! | `Down` | `ffn_down` | incoming hidden gradient | `m` | `n_embd` |
//!
//! The input column is the reference's `GQ_IN_OF` (`fold_transfer.py:1098`), and the tap widths are
//! its `chan[nm]`.
//!
//! # Why `backward_with_taps` exists here
//!
//! [`crate::learn::grad::model::backward`] returns weight gradients only; the reference taps the
//! *activation* adjoints of the same graph (`torch.autograd.grad(loss, flat)`, `:1262`). No hook in
//! `grad/model.rs` exposes them, and that file is not owned by this task, so [`backward_with_taps`]
//! recomposes the identical chain in this file out of the public kernels
//! ([`dense::linear_bwd_input`], [`dense::linear_bwd_weight`], [`dense::swiglu_bwd`],
//! [`dense::rmsnorm_bwd`], [`attn::attn_bwd`], [`attn::rope_bwd`]) and returns both the taps and the
//! same [`Grads`]. Nothing is duplicated below the kernel boundary. The gate asserts the returned
//! [`Grads`] are bit-equal to `backward`'s, so the taps provably sit on the real chain.
//!
//! # Sampled targets: the sampler deviation
//!
//! The reference draws `s = torch.multinomial(probs, 1, generator=gen)[:, 0]` once per draw
//! (`:1260`), row-major over the `N·T` positions, from the teacher's own softmax. `torch.multinomial`
//! is not reproducible across builds and `rand` is not available on this path, so the draw is an
//! inline xorshift64\* [`sample_targets`]: one uniform `u ∈ [0, 1)` per position from the seed, and
//! the inverse-CDF walk over the same Q32.32 softmax row. **Deviation:** the target *distribution* is
//! the reference's (the teacher softmax, `K` draws per token, identical draw order: draw-major, then
//! token-major); the *stream* is not `torch`'s, and no cross-implementation agreement of the drawn
//! index is claimed. The draw is deterministic in `(logits, K, seed)`.
//!
//! # Accumulation precision, and what is not ported
//!
//! The reference accumulates `HO` in `fp32` on the GPU and `S4`/`Q` in `fp64` (`:1209-1213`); here
//! every accumulator is `f64` and the operands are the exact Q32.32 adjoints scaled by `2⁻³²`. That
//! is a deliberate rounding deviation: `f64` strict superset of the reference's fp32 accumulator, and
//! the capture is then a function of the tree alone.
//!
//! Not ported, and not stubbed: `A` and `B` (the first and second moments, `:1263-1285`),
//! `max_dlogit` (the teacher-forward check, `:1246-1249`), the bit-reproducibility assertion
//! (`:1252-1258`), the block/sequence-unit intensity `ho_lb` and `PB` (`blocks=True`, `:1274-1295`)
//! and the per-linear shrinkage intensity `ho_lw` itself: `lw_intensity` (`:1101-1121`) is the
//! caller's, in the module `src/learn/fold/lw.rs` this tree does not yet carry. This module returns
//! the raw scatter and the two intensity inputs `S4`/`Q`, before the mean-diagonal normalisation, so
//! the caller can form `λ` exactly as the reference does.
//!
//! # Evidence grade
//!
//! `tested`: the gate's `cargo test` on the build path (`tests/learn_fold_hocapture.rs`) — symmetry
//! and `Cholesky` positive-definiteness after the `+εI` finish, a central-difference check of one
//! linear's output gradient against the sampled-label loss, additivity of the accumulator, the
//! `‖x‖²` weighting law, the `Grads` identity against [`crate::learn::grad::model::backward`], and
//! the refusals. No Lean refinement links this file; it is not on the extracted fragment.
//!
//! # Trust boundary
//!
//! The taps are Q32.32 CPU values and the accumulators are `f64` CPU artifacts; nothing here is the
//! quantiser's own accumulation and nothing here is measured. The backward chain, the softmax rule
//! and the kernels are the contracts of `grad::model`, `grad::head` and `grad::dense`. Not covered:
//! GPU `fp32` accumulation, `torch.multinomial`'s stream, the `A`/`B` moments, `ho_lb`, `ho_lw`, and
//! any activation the [`Tape`] does not carry.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    reason = "the Q32.32 `i64` adjoints, the `u64` draw counter and the `u32` token ids are \
              converted to `f64` at the accumulator boundary, where the value ranges are the \
              activations' own; the single-letter names are the schedule's symbols (`t`, `e`, `o`, \
              `s`, `g`, `w`), `g` names the output gradient and `w` the squared input norm; the \
              backward is one linear composition with the tap widths and the layer loop passes them \
              through"
)]

use crate::engine::fp::F64;
use crate::engine::{RopeTable, ONE};
use crate::learn::grad::attn::{self, AttnError};
use crate::learn::grad::dense::{self, DenseError};
use crate::learn::grad::head;
use crate::learn::grad::model::{Config, Grads, LayerGrads, ModelError, Tape, Weights};
use crate::numeric::fixed_exp::fixed_exp;

/// `2⁻³²`: the Q32.32 → `f64` scale. Division by a power of two is exact in binary floating point.
const INV_ONE: f64 = 1.0 / 4_294_967_296.0;

/// The diagonal the reference adds after the mean-diagonal normalisation (`fold_transfer.py:1323`).
pub const DEFAULT_EPS: f64 = 0.01;

/// The clamp the reference applies to the mean diagonal before dividing (`fold_transfer.py:1322`).
const MEAN_DIAGONAL_FLOOR: f64 = 1e-30;

/// Lower edge of the `fixed_exp` window, `-16·ONE`. A shifted logit below it has `p = 0`, the same
/// rule [`head`] applies; the constant is that module's private `FLUSH`.
const FLUSH: i128 = -16 * (1_i128 << 32);

/// The kinds, in the reference's `KINDS` order (`fold_transfer.py:47-48`).
pub const N_KINDS: usize = 7;

/// `2⁵³`, the divisor that maps the 53 high bits of a xorshift64\* word to `[0, 1)`.
const TWO53: f64 = 9_007_199_254_740_992.0;

/// The seed used when the caller passes `0`, which is not a fixed point of xorshift64\*.
const SEED_BIAS: u64 = 0x9E37_79B9_7F4A_7C15;

/// One linear of the reference's `KINDS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `attn_q`, tapped before `RoPE`.
    Q,
    /// `attn_k`, tapped before `RoPE`.
    K,
    /// `attn_v`.
    V,
    /// `attn_output`.
    O,
    /// `ffn_gate`.
    Gate,
    /// `ffn_up`.
    Up,
    /// `ffn_down`.
    Down,
}

impl Kind {
    /// Every kind, in the reference's `KINDS` order: q, k, v, o, gate, up, down.
    pub const ALL: [Kind; N_KINDS] = [
        Kind::Q,
        Kind::K,
        Kind::V,
        Kind::O,
        Kind::Gate,
        Kind::Up,
        Kind::Down,
    ];

    /// The `0..N_KINDS` index of this kind, the order of [`Kind::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Kind::Q => 0,
            Kind::K => 1,
            Kind::V => 2,
            Kind::O => 3,
            Kind::Gate => 4,
            Kind::Up => 5,
            Kind::Down => 6,
        }
    }

    /// The stage group the reference's `GQ_STAGE_OF` (`fold_transfer.py:1095`) puts this kind in.
    #[must_use]
    pub const fn stage(self) -> Stage {
        match self {
            Kind::Q | Kind::K | Kind::V => Stage::Qkv,
            Kind::O => Stage::O,
            Kind::Gate | Kind::Up => Stage::GateUp,
            Kind::Down => Stage::Down,
        }
    }

    /// The linear's output width `O = chan[kind]` (`fold_transfer.py:1205`).
    ///
    /// # Errors
    ///
    /// [`HocaptureError::Overflow`] when a width product leaves `usize`.
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn width(self, cfg: &Config) -> Result<usize, HocaptureError> {
        Ok(match self {
            Kind::Q | Kind::O => cfg.q_out().map_err(HocaptureError::Model)?,
            Kind::K | Kind::V => cfg.kv_out().map_err(HocaptureError::Model)?,
            Kind::Gate | Kind::Up => cfg.n_ff,
            Kind::Down => cfg.n_embd,
        })
    }

    /// The width of the input whose squared norm weights this kind's rank-1 update: the reference's
    /// `GQ_IN_OF` (`fold_transfer.py:1098`).
    ///
    /// # Errors
    ///
    /// [`HocaptureError::Overflow`] when a width product leaves `usize`.
    ///
    /// # Complexity
    /// `O(1)`.
    pub fn input_width(self, cfg: &Config) -> Result<usize, HocaptureError> {
        Ok(match self {
            Kind::Q | Kind::K | Kind::V | Kind::Gate | Kind::Up => cfg.n_embd,
            Kind::O => cfg.q_out().map_err(HocaptureError::Model)?,
            Kind::Down => cfg.n_ff,
        })
    }

    /// The input row block of `layer` that weights this kind's update: `ln1_out` for q/k/v,
    /// `attn_out` for o, `ln2_out` for gate/up, `m` for down.
    ///
    /// # Complexity
    /// `O(1)`.
    #[must_use]
    pub fn input(self, layer: &crate::learn::grad::model::LayerTape) -> &[i64] {
        match self {
            Kind::Q | Kind::K | Kind::V => &layer.ln1_out,
            Kind::O => &layer.attn_out,
            Kind::Gate | Kind::Up => &layer.ln2_out,
            Kind::Down => &layer.m,
        }
    }
}

/// The four stage groups of the reference's `GQ_STAGE_OF`. `ffn` is named `GateUp` here, because the
/// group holds the `ffn_gate` and `ffn_up` linears and the stage name `ffn` also names their input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// `q`, `k` and `v`.
    Qkv,
    /// `o`.
    O,
    /// `gate` and `up`.
    GateUp,
    /// `down`.
    Down,
}

/// A refusal from the output-gradient capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HocaptureError {
    /// A dimension, a buffer length or a token id disagrees with the geometry.
    Shape(String),
    /// A width product left `usize`, or a sample counter left `u64`.
    Overflow,
    /// The forward, the backward or a kernel refused.
    Model(ModelError),
}

impl From<ModelError> for HocaptureError {
    fn from(e: ModelError) -> Self {
        HocaptureError::Model(e)
    }
}

impl From<DenseError> for HocaptureError {
    fn from(e: DenseError) -> Self {
        HocaptureError::Model(ModelError::Dense(e))
    }
}

impl From<AttnError> for HocaptureError {
    fn from(e: AttnError) -> Self {
        HocaptureError::Model(ModelError::Attn(e))
    }
}

/// `HocaptureError::Shape(what)`.
///
/// # Complexity
/// `O(what)`.
fn shape(what: &str) -> HocaptureError {
    HocaptureError::Shape(what.to_string())
}

/// `.len() == t · w` or [`HocaptureError::Shape`].
///
/// # Complexity
/// `O(1)`.
fn expect(t: usize, w: usize, len: usize, what: &str) -> Result<usize, HocaptureError> {
    match t.checked_mul(w) {
        Some(n) if n == len => Ok(n),
        Some(n) => Err(shape(&format!("{what}: {len} != {t}·{w} = {n}"))),
        None => Err(HocaptureError::Overflow),
    }
}

/// Checked `a[i] += b[i]` over equal-width buffers.
///
/// # Complexity
/// `O(len)`.
fn add_flat(a: &mut [i64], b: &[i64]) -> Result<(), HocaptureError> {
    if a.len() != b.len() {
        return Err(shape("add width"));
    }
    for (x, y) in a.iter_mut().zip(b) {
        *x = x.checked_add(*y).ok_or(HocaptureError::Overflow)?;
    }
    Ok(())
}

/// The column sums of a `t × w` row-major buffer, checked.
///
/// # Complexity
/// `O(t·w)`.
fn col_sum(x: &[i64], t: usize, w: usize) -> Result<Vec<i64>, HocaptureError> {
    expect(t, w, x.len(), "col_sum")?;
    let mut out = vec![0i64; w];
    for r in 0..t {
        for c in 0..w {
            out[c] = out[c]
                .checked_add(x[r * w + c])
                .ok_or(HocaptureError::Overflow)?;
        }
    }
    Ok(out)
}

/// `RMSNorm` backward over every row of a `t × w` buffer: `(dx, Σ_r dgamma_r)`.
///
/// The composition is `crate::learn::grad::model`'s private `rmsnorm_bwd_rows`, recomposed from the
/// public [`dense::rmsnorm_bwd`] with the same checked accumulation order.
///
/// # Complexity
/// `O(t·w)`; one [`dense::rmsnorm_bwd`] per row.
fn rmsnorm_bwd_rows(
    x: &[i64],
    gamma: &[i64],
    eps: i64,
    dy: &[i64],
    t: usize,
    w: usize,
) -> Result<(Vec<i64>, Vec<i64>), HocaptureError> {
    expect(t, w, x.len(), "rmsnorm_bwd_rows x")?;
    expect(t, w, dy.len(), "rmsnorm_bwd_rows dy")?;
    let mut dx = vec![0i64; x.len()];
    let mut dg = vec![0i64; w];
    for r in 0..t {
        let (dxr, dgr) =
            dense::rmsnorm_bwd(&x[r * w..r * w + w], gamma, eps, &dy[r * w..r * w + w])?;
        dx[r * w..r * w + w].copy_from_slice(&dxr);
        for c in 0..w {
            dg[c] = dg[c].checked_add(dgr[c]).ok_or(HocaptureError::Overflow)?;
        }
    }
    Ok((dx, dg))
}

/// `RoPE` backward on every `row`-wide row of a `t × row` gradient buffer, at the row's own position.
///
/// # Complexity
/// `O(t·row)`; one [`attn::rope_bwd`] per token.
fn rope_rows_bwd(
    dy: &[i64],
    t: usize,
    row: usize,
    rope: &RopeTable,
) -> Result<Vec<i64>, HocaptureError> {
    expect(t, row, dy.len(), "rope_rows_bwd")?;
    let mut out = Vec::with_capacity(dy.len());
    for r in 0..t {
        let pos = u32::try_from(r).map_err(|_| HocaptureError::Overflow)?;
        out.extend_from_slice(&attn::rope_bwd(&dy[r * row..r * row + row], pos, rope)?);
    }
    Ok(out)
}

/// One layer's seven tapped output gradients, each `t × O` row-major in Q32.32, `O` the kind's
/// [`Kind::width`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerTaps {
    /// `∂L/∂attn_q output`, `t × q_out`.
    pub q: Vec<i64>,
    /// `∂L/∂attn_k output`, `t × kv_out`.
    pub k: Vec<i64>,
    /// `∂L/∂attn_v output`, `t × kv_out`.
    pub v: Vec<i64>,
    /// `∂L/∂attn_output output`, `t × q_out`.
    pub o: Vec<i64>,
    /// `∂L/∂ffn_gate output`, `t × n_ff`.
    pub gate: Vec<i64>,
    /// `∂L/∂ffn_up output`, `t × n_ff`.
    pub up: Vec<i64>,
    /// `∂L/∂ffn_down output`, `t × n_embd`.
    pub down: Vec<i64>,
}

impl LayerTaps {
    /// This layer's tap of `kind`.
    ///
    /// # Complexity
    /// `O(1)`.
    #[must_use]
    pub fn get(&self, kind: Kind) -> &[i64] {
        match kind {
            Kind::Q => &self.q,
            Kind::K => &self.k,
            Kind::V => &self.v,
            Kind::O => &self.o,
            Kind::Gate => &self.gate,
            Kind::Up => &self.up,
            Kind::Down => &self.down,
        }
    }
}

/// Every layer's tapped output gradients, in forward layer order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Taps {
    /// Per trunk layer, in forward order; `taps[l].get(kind)` is `∂L/∂(kind output)` at layer `l`.
    pub layers: Vec<LayerTaps>,
}

impl Taps {
    /// Layer `l`'s tap of `kind`, or `None` when the layer index is out of range.
    ///
    /// # Complexity
    /// `O(1)`.
    #[must_use]
    pub fn get(&self, l: usize, kind: Kind) -> Option<&[i64]> {
        self.layers.get(l).map(|lt| lt.get(kind))
    }
}

/// The whole-model backward over `dlogits` (`t × vocab`), returning the same [`Grads`] as
/// [`crate::learn::grad::model::backward`] **and** the seven per-layer output gradients that
/// backward does not expose.
///
/// The chain, the kernel calls, the accumulation order and the single rounding per output are
/// [`crate::learn::grad::model::backward`]'s (`src/learn/grad/model.rs:739-817`); the taps are read
/// at the points the reference's `gq_capture` taps (`fold_transfer.py:1262`): `∂L/∂o = ∂L/∂h2` at the
/// residual split `h2 = h + o`, `∂L/∂(q,k,v)` after each `RoPE` adjoint, `∂L/∂(gate,up)` out of
/// `SwiGLU`, `∂L/∂(down output)` the incoming hidden gradient.
///
/// `dlogits` is the logit gradient of the loss the caller differentiates; for the sampled-label loss
/// of [`accumulate_kind`] it is `p − onehot(s)`, the [`logits_grad_ce`] of the sampled targets.
///
/// # Errors
///
/// [`HocaptureError::Shape`] when the tape, `dlogits` or a buffer disagrees with the geometry;
/// [`HocaptureError::Overflow`] on a checked add or a length product; and the
/// [`HocaptureError::Model`] refusals of the kernels, as in
/// [`crate::learn::grad::model::backward`].
///
/// # Complexity
/// Mirrors [`crate::learn::grad::model::backward`]: `O(t·(n_embd² + n_embd·n_ff) + n_head·t²·head_dim)`
/// per layer plus the head, with the taps as extra `O(t·Σ_kind O_kind)` words of output.
pub fn backward_with_taps(
    w: &Weights,
    cfg: &Config,
    tape: &Tape,
    dlogits: &[i64],
) -> Result<(Grads, Taps), HocaptureError> {
    cfg.checked().map_err(HocaptureError::Model)?;
    let t = tape.t;
    let e = cfg.n_embd;
    let q_out = cfg.q_out().map_err(HocaptureError::Model)?;
    let kv_out = cfg.kv_out().map_err(HocaptureError::Model)?;
    let n_ff = cfg.n_ff;
    if tape.layers.len() != cfg.n_layer || tape.tokens.len() != t {
        return Err(shape("tape shape"));
    }
    expect(t, cfg.vocab, dlogits.len(), "dlogits")?;
    let rope = RopeTable::new(F64::from_f64(cfg.rope_theta), cfg.head_dim)
        .map_err(|err| HocaptureError::Model(ModelError::Attn(AttnError::Rope(err))))?;

    let d_normed = dense::linear_bwd_input(dlogits, &w.lm_head, t, e, cfg.vocab)?;
    let lm_head = dense::linear_bwd_weight(dlogits, &tape.final_normed, t, e, cfg.vocab)?;
    let (mut dh, output_norm) =
        rmsnorm_bwd_rows(&tape.final_hidden, &w.output_norm, cfg.eps, &d_normed, t, e)?;

    let mut grads: Vec<LayerGrads> = Vec::with_capacity(cfg.n_layer);
    let mut taps: Vec<LayerTaps> = Vec::with_capacity(cfg.n_layer);
    for l in (0..cfg.n_layer).rev() {
        let lw = &w.layers[l];
        let lt = &tape.layers[l];
        // h3 = h2 + down(m): the incoming hidden gradient is the down linear's output gradient.
        let down = dh.clone();
        let dd = dense::linear_bwd_input(&dh, &lw.w_down, t, n_ff, e)?;
        let w_down = dense::linear_bwd_weight(&dh, &lt.m, t, n_ff, e)?;
        let (dgate, dup) = dense::swiglu_bwd(&lt.gate, &lt.up, &dd)?;
        let (gate, up) = (dgate.clone(), dup.clone());
        let dln2_g = dense::linear_bwd_input(&dgate, &lw.w_gate, t, e, n_ff)?;
        let w_gate = dense::linear_bwd_weight(&dgate, &lt.ln2_out, t, e, n_ff)?;
        let dln2_u = dense::linear_bwd_input(&dup, &lw.w_up, t, e, n_ff)?;
        let w_up = dense::linear_bwd_weight(&dup, &lt.ln2_out, t, e, n_ff)?;
        let mut dln2 = dln2_g;
        add_flat(&mut dln2, &dln2_u)?;
        let (dh2_ffn, ffn_norm) = rmsnorm_bwd_rows(&lt.h2, &lw.ffn_norm, cfg.eps, &dln2, t, e)?;
        // The residual identity: h3's gradient flows unchanged into h2, plus the ffn-norm branch.
        let mut dh2 = dh;
        add_flat(&mut dh2, &dh2_ffn)?;
        // h2 = h + o, and the attn-norm branch below feeds h, not o: this is ∂L/∂o.
        let o = dh2.clone();
        let d_attn_out = dense::linear_bwd_input(&dh2, &lw.wo, t, q_out, e)?;
        let wo = dense::linear_bwd_weight(&dh2, &lt.attn_out, t, q_out, e)?;
        let (dq_rope, dk_rope, dv) = attn::attn_bwd(
            &lt.q,
            &lt.k,
            &lt.v,
            &lt.p,
            &d_attn_out,
            cfg.n_head,
            cfg.n_head_kv,
            cfg.head_dim,
        )?;
        let q = rope_rows_bwd(&dq_rope, t, q_out, &rope)?;
        let k = rope_rows_bwd(&dk_rope, t, kv_out, &rope)?;
        let bq = col_sum(&q, t, q_out)?;
        let bk = col_sum(&k, t, kv_out)?;
        let bv = col_sum(&dv, t, kv_out)?;
        let dln1_q = dense::linear_bwd_input(&q, &lw.wq, t, e, q_out)?;
        let dln1_k = dense::linear_bwd_input(&k, &lw.wk, t, e, kv_out)?;
        let dln1_v = dense::linear_bwd_input(&dv, &lw.wv, t, e, kv_out)?;
        let wq = dense::linear_bwd_weight(&q, &lt.ln1_out, t, e, q_out)?;
        let wk = dense::linear_bwd_weight(&k, &lt.ln1_out, t, e, kv_out)?;
        let wv = dense::linear_bwd_weight(&dv, &lt.ln1_out, t, e, kv_out)?;
        let mut dln1 = dln1_q;
        add_flat(&mut dln1, &dln1_k)?;
        add_flat(&mut dln1, &dln1_v)?;
        let (dh_attn, attn_norm) =
            rmsnorm_bwd_rows(&lt.ln1_in, &lw.attn_norm, cfg.eps, &dln1, t, e)?;
        add_flat(&mut dh2, &dh_attn)?;
        dh = dh2;
        taps.push(LayerTaps {
            q,
            k,
            v: dv,
            o,
            gate,
            up,
            down,
        });
        grads.push(LayerGrads {
            attn_norm,
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            wo,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
        });
    }
    grads.reverse();
    taps.reverse();

    let token_embd = embedding_grad(&dh, &tape.tokens, t, e, cfg.vocab)?;
    Ok((
        Grads {
            layers: grads,
            output_norm,
            token_embd,
            lm_head,
        },
        Taps { layers: taps },
    ))
}

/// `dembedding[token[r], :] += dh[r, :]`, checked, over a `t × w` hidden gradient.
///
/// # Complexity
/// `O(t·w)`.
fn embedding_grad(
    dh: &[i64],
    tokens: &[u32],
    t: usize,
    w: usize,
    vocab: usize,
) -> Result<Vec<i64>, HocaptureError> {
    expect(t, w, dh.len(), "embedding_grad")?;
    let mut out = vec![0i64; vocab * w];
    for r in 0..t {
        let id = usize::try_from(tokens[r]).map_err(|_| HocaptureError::Overflow)?;
        if id >= vocab {
            return Err(shape("embedding token >= vocab"));
        }
        for c in 0..w {
            out[id * w + c] = out[id * w + c]
                .checked_add(dh[r * w + c])
                .ok_or(HocaptureError::Overflow)?;
        }
    }
    Ok(out)
}

/// `n / d` rounded to nearest, ties away from zero; `d > 0`.
///
/// `head.rs`'s `round_div` is private; this is its rule.
///
/// # Complexity
/// `O(1)`.
fn round_div(n: i128, d: i128) -> i128 {
    let q = n / d;
    let r = n % d;
    let mag = if r < 0 { 0 - r } else { r };
    if 2 * mag >= d {
        if n >= 0 {
            q + 1
        } else {
            q - 1
        }
    } else {
        q
    }
}

/// `x` narrowed to `i64`, or [`HocaptureError::Overflow`] when it leaves the range.
///
/// # Complexity
/// `O(1)`.
fn to64(x: i128) -> Result<i64, HocaptureError> {
    if x < i128::from(i64::MIN) || x > i128::from(i64::MAX) {
        Err(HocaptureError::Overflow)
    } else {
        Ok(x as i64)
    }
}

/// Softmax of one Q32.32 logit row, `pᵢ = round(eᵢ·ONE/S)`.
///
/// The rule is [`head`]'s private `softmax_row` (`src/learn/grad/head.rs:279-304`): shift by the row
/// maximum, `eᵢ = fixed_exp(uᵢ)` with `uᵢ < −16·ONE` flushed to `0`, `S = Σeᵢ`, one rounding per
/// `pᵢ`. (The reference row also forms `L = fixed_ln(S)` for its `log p`; this port needs `p` only.)
/// It is mirrored here over the same public [`fixed_exp`] because `head`'s function is private and
/// this task does not own `head.rs`; the gate pins the two to the same values through `kl_row`
/// (`softmax_matches_head`).
///
/// # Errors
///
/// [`HocaptureError::Shape`] for an empty row, or when [`fixed_exp`] refuses the shifted value
/// (unreachable: every shifted value is `≤ 0` and the flush catches the window's lower edge);
/// [`HocaptureError::Overflow`] when the weight sum or a probability leaves `i64`.
///
/// # Complexity
/// `O(v)` comparisons, `v` [`fixed_exp`] calls and `v` `i128` divisions; `O(v)` words of output.
pub fn softmax_row(z: &[i64]) -> Result<Vec<i64>, HocaptureError> {
    let v = z.len();
    if v == 0 {
        return Err(shape("empty softmax row"));
    }
    let mut m = z[0];
    for &zi in z {
        if zi > m {
            m = zi;
        }
    }
    let mut e = vec![0i64; v];
    let mut sum: i128 = 0;
    for (i, &zi) in z.iter().enumerate() {
        let u = i128::from(zi) - i128::from(m);
        if u < FLUSH {
            continue;
        }
        let ev = fixed_exp(to64(u)?).map_err(|_| shape("fixed_exp refused"))?;
        e[i] = ev;
        sum += i128::from(ev);
    }
    let mut p = vec![0i64; v];
    let one = i128::from(ONE);
    for i in 0..v {
        p[i] = to64(round_div(i128::from(e[i]) * one, sum))?;
    }
    Ok(p)
}

/// The chunked head's `Σ_t log p_t` read at the sampled targets, negated: the sampled-label
/// cross-entropy the taps differentiate, in Q32.32.
///
/// This is `head::log_softmax_row` per row, summed in exact `i128` and rounded once. It is the loss
/// whose central difference the gate compares a tap against, so the check does not go through
/// [`softmax_row`]'s own rounding.
///
/// # Errors
///
/// [`HocaptureError::Shape`] when `logits.len() != t·vocab`, `targets.len() != t`, or a target is
/// past the vocabulary; [`HocaptureError::Overflow`] on a checked sum or a narrowing.
///
/// # Complexity
/// `O(t·vocab)`; one softmax per row.
pub fn cross_entropy_sum(
    logits: &[i64],
    t: usize,
    vocab: usize,
    targets: &[u32],
) -> Result<i64, HocaptureError> {
    expect(t, vocab, logits.len(), "cross_entropy logits")?;
    if targets.len() != t {
        return Err(shape("cross_entropy targets"));
    }
    let mut acc: i128 = 0;
    for r in 0..t {
        let id = usize::try_from(targets[r]).map_err(|_| HocaptureError::Overflow)?;
        if id >= vocab {
            return Err(shape("cross_entropy target >= vocab"));
        }
        let lp = head::log_softmax_row(&logits[r * vocab..(r + 1) * vocab])
            .map_err(|_| shape("log_softmax_row refused"))?;
        acc -= i128::from(lp[id]);
    }
    to64(round_div(acc, 1))
}

/// The logit gradient of the sampled-label loss `Σ_t CE(z_t, s_t)`: `out[t·vocab + i] = p_ti −
/// [i = s_t]`, in Q32.32.
///
/// # Errors
///
/// [`HocaptureError::Shape`] when `logits.len() != t·vocab`, `out.len() != t·vocab`, `targets.len()
/// != t`, or a target is past the vocabulary.
///
/// # Complexity
/// `O(t·vocab)`; one [`softmax_row`] per row and one write per entry.
pub fn logits_grad_ce(
    logits: &[i64],
    targets: &[u32],
    vocab: usize,
    out: &mut [i64],
) -> Result<(), HocaptureError> {
    let t = targets.len();
    expect(t, vocab, logits.len(), "logits_grad_ce logits")?;
    expect(t, vocab, out.len(), "logits_grad_ce out")?;
    for r in 0..t {
        let id = usize::try_from(targets[r]).map_err(|_| HocaptureError::Overflow)?;
        if id >= vocab {
            return Err(shape("logits_grad_ce target >= vocab"));
        }
        let p = softmax_row(&logits[r * vocab..(r + 1) * vocab])?;
        let row = r * vocab;
        for i in 0..vocab {
            out[row + i] = p[i] - if i == id { ONE } else { 0 };
        }
    }
    Ok(())
}

/// One xorshift64\* step, returning the uniform `u ∈ [0, 1)` of its 53 high bits.
///
/// # Complexity
/// `O(1)`.
fn xorshift64s(state: &mut u64) -> f64 {
    let mut s = if *state == 0 { SEED_BIAS } else { *state };
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    *state = s;
    let r = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
    ((r >> 11) as f64) / TWO53
}

/// `K` sampled targets per token from the teacher softmax of `logits`, draw-major then token-major:
/// `out[j·t + i]` is draw `j`'s target at token `i`.
///
/// The distribution is the reference's (`torch.multinomial(probs, 1)` per draw over the `N·T`
/// rows, `fold_transfer.py:1260`); the stream is this file's inline xorshift64\* from `seed`. The
/// inverse-CDF walk reads the same Q32.32 softmax row the loss gradient uses, and falls back to the
/// last index with non-zero probability when the uniform lands past the row's rounding residue.
///
/// # Errors
///
/// [`HocaptureError::Shape`] when `K == 0`, `vocab == 0`, `logits.len() != t·vocab`, or `t == 0`.
///
/// # Complexity
/// `O(K·t·vocab)`: one [`softmax_row`] per token plus one inverse-CDF walk per draw.
pub fn sample_targets(
    logits: &[i64],
    t: usize,
    vocab: usize,
    k: usize,
    seed: u64,
) -> Result<Vec<u32>, HocaptureError> {
    if k == 0 || vocab == 0 || t == 0 {
        return Err(shape("sample_targets dimensions"));
    }
    expect(t, vocab, logits.len(), "sample_targets logits")?;
    let mut rows = Vec::with_capacity(t);
    for r in 0..t {
        rows.push(softmax_row(&logits[r * vocab..(r + 1) * vocab])?);
    }
    let mut state = if seed == 0 { SEED_BIAS } else { seed };
    let mut out = vec![0u32; k * t];
    for j in 0..k {
        for r in 0..t {
            let p = &rows[r];
            let draw = xorshift64s(&mut state) * (ONE as f64);
            let mut cum: i128 = 0;
            let mut pick = p.len() - 1;
            let mut found = false;
            for (i, &pi) in p.iter().enumerate() {
                cum += i128::from(pi);
                if draw < cum as f64 {
                    pick = i;
                    found = true;
                    break;
                }
            }
            if !found {
                for (i, &pi) in p.iter().enumerate() {
                    if pi > 0 {
                        pick = i;
                    }
                }
            }
            out[j * t + r] = u32::try_from(pick).map_err(|_| HocaptureError::Overflow)?;
        }
    }
    Ok(out)
}

/// One `(layer, kind)` raw accumulator: the scatter `Σ_t w_t g_t g_tᵀ` and the two Schäfer–Strimmer
/// inputs of the same samples.
#[derive(Debug, Clone, PartialEq)]
pub struct KindStats {
    /// `Σ_t w_t g_t g_tᵀ`, row-major `O × O`, full and symmetric.
    pub ho: Vec<f64>,
    /// `Σ_t ‖s_t‖⁴` with `s_ti = √w_t·g_ti` (`fold_transfer.py:1282`).
    pub s4: f64,
    /// `Σ_t s_ti⁴` per output coordinate (`fold_transfer.py:1283`), `O` long.
    pub q: Vec<f64>,
    /// `(token, draw)` pairs accumulated: the reference's `n = K·N·T` (`fold_transfer.py:1312`).
    pub samples: u64,
}

impl KindStats {
    /// The zero accumulator of width `O`.
    ///
    /// # Errors
    ///
    /// [`HocaptureError::Overflow`] when `O²` leaves `usize`.
    ///
    /// # Complexity
    /// Time `O(O²)`; space `O(O²)` `f64` words for `ho` and `O` for `q`.
    pub fn zeros(o: usize) -> Result<KindStats, HocaptureError> {
        let o2 = o.checked_mul(o).ok_or(HocaptureError::Overflow)?;
        Ok(KindStats {
            ho: vec![0.0; o2],
            s4: 0.0,
            q: vec![0.0; o],
            samples: 0,
        })
    }
}

/// Every kind's raw accumulator for one layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerStats {
    /// Indexed by [`Kind::index`].
    pub kinds: [KindStats; N_KINDS],
}

impl LayerStats {
    /// The accumulator of `kind`.
    ///
    /// # Complexity
    /// `O(1)`.
    #[must_use]
    pub fn kind(&self, kind: Kind) -> &KindStats {
        &self.kinds[kind.index()]
    }
}

/// Fold one token block of `(input, output-gradient)` pairs into `stats`: the rank-1 update
/// `ho += Σ_t ‖x_t‖² g_t g_tᵀ` and the `S4`/`Q` recurrences, both `O(O²)`/`O(O)` per token.
///
/// `x` is the linear's input, `t × w` row-major in Q32.32; `g` is the loss's gradient with respect to
/// the linear's output, `t × o` row-major in Q32.32. `w_t = ‖x_t‖²` in real units (`x/2³²` twice), and
/// `s_ti = √w_t·g_ti` in real units, exactly the reference's `s2 = g²·w` (`fold_transfer.py:1281`).
///
/// # Errors
///
/// [`HocaptureError::Shape`] when `x.len() != t·w`, `g.len() != t·o`, `o == 0`, or `stats` has the
/// wrong `ho`/`q` widths; [`HocaptureError::Overflow`] when the sample counter leaves `u64`.
///
/// # Complexity
/// Time `O(t·(w + O²))`: one `w`-term norm and `O(O+1)/2` rank-1 multiply-adds plus `O` recurrence
/// terms per token. Space `O(max(w, O))` `f64` scratch rows, no per-token allocation.
pub fn accumulate_kind(
    stats: &mut KindStats,
    x: &[i64],
    g: &[i64],
    t: usize,
    w: usize,
    o: usize,
) -> Result<(), HocaptureError> {
    if o == 0 || w == 0 {
        return Err(shape("accumulate_kind zero width"));
    }
    expect(t, w, x.len(), "accumulate_kind x")?;
    expect(t, o, g.len(), "accumulate_kind g")?;
    if o.checked_mul(o) != Some(stats.ho.len()) || stats.q.len() != o {
        return Err(shape("accumulate_kind accumulator width"));
    }
    let mut xf = vec![0.0f64; w];
    let mut gf = vec![0.0f64; o];
    let mut s2 = vec![0.0f64; o];
    for r in 0..t {
        for (d, v) in xf.iter_mut().zip(&x[r * w..r * w + w]) {
            *d = *v as f64 * INV_ONE;
        }
        let wt: f64 = xf.iter().map(|v| v * v).sum();
        for (d, v) in gf.iter_mut().zip(&g[r * o..r * o + o]) {
            *d = *v as f64 * INV_ONE;
        }
        for i in 0..o {
            s2[i] = wt * gf[i] * gf[i];
        }
        for i in 0..o {
            let gi = wt * gf[i];
            for j in i..o {
                let v = gi * gf[j];
                stats.ho[i * o + j] += v;
                if i != j {
                    stats.ho[j * o + i] += v;
                }
            }
        }
        let n2: f64 = s2.iter().sum();
        stats.s4 += n2 * n2;
        for i in 0..o {
            stats.q[i] += s2[i] * s2[i];
        }
    }
    let rows = u64::try_from(t).map_err(|_| HocaptureError::Overflow)?;
    stats.samples = stats
        .samples
        .checked_add(rows)
        .ok_or(HocaptureError::Overflow)?;
    Ok(())
}

/// The output-side (YAQA Sketch A) capture: one raw accumulator per `(layer, kind)`, plus the fixed
/// draw count and seed.
#[derive(Debug, Clone, PartialEq)]
pub struct Hocapture {
    /// Draws per token, the reference's `K`.
    k: usize,
    /// The sampler seed.
    seed: u64,
    /// Residual width, fixed by [`Hocapture::new`].
    n_embd: usize,
    /// `n_head·head_dim`, fixed by [`Hocapture::new`].
    q_out: usize,
    /// `n_head_kv·head_dim`, fixed by [`Hocapture::new`].
    kv_out: usize,
    /// Feed-forward width, fixed by [`Hocapture::new`].
    n_ff: usize,
    /// Vocabulary, fixed by [`Hocapture::new`].
    vocab: usize,
    /// Per trunk layer, in forward order.
    layers: Vec<LayerStats>,
}

impl Hocapture {
    /// The zero accumulator for `cfg`'s geometry, `k` draws per token and the sampler seed `seed`.
    ///
    /// # Errors
    ///
    /// [`HocaptureError::Shape`] for `k == 0` or a zero model dimension;
    /// [`HocaptureError::Overflow`] when `q_out`, `kv_out` or a squared width leaves `usize`.
    ///
    /// # Complexity
    /// Time `O(n_layer·(q_out² + kv_out² + n_ff² + n_embd²))`; the same in space, one
    /// [`KindStats::zeros`] per `(layer, kind)`.
    pub fn new(cfg: &Config, k: usize, seed: u64) -> Result<Hocapture, HocaptureError> {
        cfg.checked().map_err(HocaptureError::Model)?;
        if k == 0 || cfg.n_layer == 0 {
            return Err(shape("Hocapture dimensions"));
        }
        let q_out = cfg.q_out().map_err(HocaptureError::Model)?;
        let kv_out = cfg.kv_out().map_err(HocaptureError::Model)?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            let mut kinds: Vec<KindStats> = Vec::with_capacity(N_KINDS);
            for kind in Kind::ALL {
                kinds.push(KindStats::zeros(kind.width(cfg)?)?);
            }
            let kinds: [KindStats; N_KINDS] = kinds.try_into().map_err(|_| shape("kind count"))?;
            layers.push(LayerStats { kinds });
        }
        Ok(Hocapture {
            k,
            seed,
            n_embd: cfg.n_embd,
            q_out,
            kv_out,
            n_ff: cfg.n_ff,
            vocab: cfg.vocab,
            layers,
        })
    }

    /// Draws `K` targets per token from the model's own softmax at `tokens` (seed
    /// [`Hocapture::seed`]) and folds them in.
    ///
    /// # Errors
    ///
    /// As [`Hocapture::add_taps`], plus the forward's and the sampler's refusals.
    ///
    /// # Complexity
    /// One forward, `K` backward passes and `O(K·t·Σ_kind(w_kind + O_kind²))` accumulation terms; as
    /// [`crate::learn::grad::model::forward`] plus `K ×` [`backward_with_taps`].
    pub fn accumulate(
        &mut self,
        w: &Weights,
        cfg: &Config,
        tokens: &[u32],
    ) -> Result<(), HocaptureError> {
        let (logits, tape) =
            crate::learn::grad::model::forward(w, cfg, tokens).map_err(HocaptureError::Model)?;
        let targets = sample_targets(&logits, tape.t, cfg.vocab, self.k, self.seed)?;
        self.add_taps(w, cfg, &logits, &tape, &targets)
    }

    /// Runs the forward at `tokens` and folds in the caller's fixed `targets`, one target per token
    /// per draw, draw-major: `targets[j·t + i]` is draw `j`'s target at token `i`.
    ///
    /// # Errors
    ///
    /// As [`Hocapture::add_taps`], plus the forward's refusals.
    ///
    /// # Complexity
    /// One forward, `K` backward passes and the accumulation; as [`Hocapture::accumulate`].
    pub fn accumulate_with_targets(
        &mut self,
        w: &Weights,
        cfg: &Config,
        tokens: &[u32],
        targets: &[u32],
    ) -> Result<(), HocaptureError> {
        let (logits, tape) =
            crate::learn::grad::model::forward(w, cfg, tokens).map_err(HocaptureError::Model)?;
        self.add_taps(w, cfg, &logits, &tape, targets)
    }

    /// Folds `K` draws of fixed `targets` into the accumulator from an already-computed forward:
    /// `logits` (`t × vocab`) and `tape` as [`crate::learn::grad::model::forward`] returned them.
    ///
    /// One [`logits_grad_ce`] and one [`backward_with_taps`] per draw; the taps are folded by
    /// [`accumulate_kind`] against the tape's own input rows, so the weighting is the linear's real
    /// input norm, not a copy of it.
    ///
    /// # Errors
    ///
    /// [`HocaptureError::Shape`] when `cfg` disagrees with the geometry [`Hocapture::new`] fixed,
    /// when `logits.len() != t·vocab`, when `targets.len() != k·t`, when a target is past the
    /// vocabulary, or when a tape buffer is not `t × width`; [`HocaptureError::Overflow`] on a
    /// length product or a checked add; the kernels' refusals otherwise.
    ///
    /// # Complexity
    /// `K` backward passes, `O(K·(t·vocab + t·(n_embd² + n_embd·n_ff) + n_head·t²·head_dim))`, plus
    /// `O(K·t·Σ_kind(w_kind + O_kind²))` for the rank-1 updates.
    pub fn add_taps(
        &mut self,
        w: &Weights,
        cfg: &Config,
        logits: &[i64],
        tape: &Tape,
        targets: &[u32],
    ) -> Result<(), HocaptureError> {
        if cfg.n_embd != self.n_embd
            || cfg.n_ff != self.n_ff
            || cfg.vocab != self.vocab
            || cfg.q_out().map_err(HocaptureError::Model)? != self.q_out
            || cfg.kv_out().map_err(HocaptureError::Model)? != self.kv_out
            || cfg.n_layer != self.layers.len()
        {
            return Err(shape("config != Hocapture geometry"));
        }
        let t = tape.t;
        expect(t, cfg.vocab, logits.len(), "add_taps logits")?;
        if targets.len() != self.k * t {
            return Err(shape("add_taps targets != k·t"));
        }
        let mut dlogits = vec![0i64; t * cfg.vocab];
        for j in 0..self.k {
            let rows = &targets[j * t..(j + 1) * t];
            logits_grad_ce(logits, rows, cfg.vocab, &mut dlogits)?;
            let (_grads, taps) = backward_with_taps(w, cfg, tape, &dlogits)?;
            for l in 0..cfg.n_layer {
                for kind in Kind::ALL {
                    let o = kind.width(cfg)?;
                    let iw = kind.input_width(cfg)?;
                    let stats = &mut self.layers[l].kinds[kind.index()];
                    accumulate_kind(
                        stats,
                        kind.input(&tape.layers[l]),
                        taps.layers[l].get(kind),
                        t,
                        iw,
                        o,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Draws per token, the reference's `K`.
    ///
    /// # Complexity
    /// `O(1)`.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// The sampler seed.
    ///
    /// # Complexity
    /// `O(1)`.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The raw accumulators, per layer: `ho = Σ_t w_t g_t g_tᵀ`, `s4`, `q`, `samples`.
    ///
    /// # Complexity
    /// `O(1)`.
    #[must_use]
    pub fn stats(&self) -> &[LayerStats] {
        &self.layers
    }

    /// The normalised copy: per `(layer, kind)`, `ho / mean(diag(ho))` with the mean clamped up to
    /// `1e-30`, then `eps` added to the diagonal — the reference's `x / x.diagonal().mean().clamp_min(1e-30)`
    /// and `x.diagonal().add_(0.01)` (`fold_transfer.py:1322-1323`), with the caller supplying `eps`
    /// ([`DEFAULT_EPS`] is the reference's `0.01`). `s4`, `q` and `samples` pass through unchanged, so
    /// `ho_lw` can still be formed on the raw scatter as the reference does.
    ///
    /// # Errors
    ///
    /// [`HocaptureError::Shape`] when a kind has accumulated no samples (a mean over no samples is
    /// undefined, not zero).
    ///
    /// # Complexity
    /// `O(n_layer·Σ_kind O_kind²)` time, one `ho` copy per `(layer, kind)`.
    pub fn finish(&self, eps: f64) -> Result<Vec<LayerStats>, HocaptureError> {
        let mut out = Vec::with_capacity(self.layers.len());
        for ls in &self.layers {
            let mut kinds: Vec<KindStats> = Vec::with_capacity(N_KINDS);
            for kind in Kind::ALL {
                let ks = ls.kind(kind);
                if ks.samples == 0 {
                    return Err(shape("finish over zero samples"));
                }
                let o = kind.width_at(self);
                let mut ho = ks.ho.clone();
                let mut diag = 0.0f64;
                for i in 0..o {
                    diag += ho[i * o + i];
                }
                let mean = diag / o as f64;
                let denom = if mean > MEAN_DIAGONAL_FLOOR {
                    mean
                } else {
                    MEAN_DIAGONAL_FLOOR
                };
                for v in &mut ho {
                    *v /= denom;
                }
                for i in 0..o {
                    ho[i * o + i] += eps;
                }
                kinds.push(KindStats {
                    ho,
                    s4: ks.s4,
                    q: ks.q.clone(),
                    samples: ks.samples,
                });
            }
            let kinds: [KindStats; N_KINDS] = kinds.try_into().map_err(|_| shape("kind count"))?;
            out.push(LayerStats { kinds });
        }
        Ok(out)
    }
}

impl Kind {
    /// The kind's output width from the accumulator's own fixed geometry.
    ///
    /// # Complexity
    /// `O(1)`.
    fn width_at(self, cap: &Hocapture) -> usize {
        match self {
            Kind::Q | Kind::O => cap.q_out,
            Kind::K | Kind::V => cap.kv_out,
            Kind::Gate | Kind::Up => cap.n_ff,
            Kind::Down => cap.n_embd,
        }
    }
}
