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
//! inline xorshift64\* [`sample_targets`]: one uniform Q32.32 word in `[0, ONE)` per position from the
//! seed, and the inverse-CDF walk over the same Q32.32 softmax row. **Deviation:** the target
//! *distribution* is the reference's (the teacher softmax, `K` draws per token, identical draw
//! order: draw-major, then token-major); the *stream* is not `torch`'s, and no cross-implementation
//! agreement of the drawn index is claimed. The draw is deterministic in `(logits, K, seed)`.
//!
//! # Accumulation carrier and error
//!
//! `HO` is Q32.32 `i64`, the taps' own carrier. The reference accumulates `HO` in `fp32` on the GPU
//! and `S4`/`Q` in `fp64` (`:1209-1213`); here one term is `a = round_q32_even(w_t·g_ti)` — the exact
//! `i128` product of two words, rounded once — and the entry accumulates `Σ_t a·g_tj` in exact
//! `i128`, rounded once per entry, so an entry carries at most `t` half-quanta of intermediate
//! rounding (`t·2⁻³³`) plus its own half quantum. The mean-diagonal finish divides `HO` in fixed
//! point (`round(H·O·2³²/Σdiag)` words) and adds the caller's `eps` word on the diagonal; the floor
//! `1e-30` is sub-quantum, so only a zero diagonal is reachable and it has the zero quotient (a
//! nonzero scatter whose mean diagonal underflows the grid refuses [`HocaptureError::Overflow`]
//! rather than inventing a value).
//!
//! `S4` and `Q` are fourth powers of Q32.32 quantities, so their real magnitude reaches `2⁶⁴` times
//! the grid's `|x| < 2³¹` bound and they are carried by [`Fx`] (`m·2^e`, `|m| < 2⁶²`) rather than by
//! a Q32.32 word; the reference carries them in `fp64` for the same reason. Each [`accumulate_kind`]
//! call forms the exact `i128` Q64.64 total over its tokens and rounds it once into the carrier
//! (`Fx::from_i128(total, −64)`), and the per-call totals are added by [`fx_add`], so an accumulated
//! `S4`/`Q` carries one 62-bit rounding per call plus one per addition — a relative `2⁻⁶¹` per step,
//! nine orders below the `2⁻³²` rounding the same quantity inherits from its Q32.32 factors.
//!
//! The `HO` pass of [`accumulate_kind`] is the file's one matrix-sized pass (`t·O²/2` word ops per
//! call) and runs on the engine pool through [`chunks_mut_global`]: each output entry is written by
//! exactly one task and summed in ascending token order there, so results are schedule-independent.
//! The weight, `Q` and `S4` passes are vector loops (one value per token or per coordinate) and stay
//! serial, as do [`softmax_row`], [`sample_targets`] and [`logits_grad_ce`], whose row bodies a pool
//! task must not re-dispatch (`chart::pool::Pool`).
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
//! The taps, the scatter and the intensity inputs are Q32.32 CPU words; nothing here is the
//! quantiser's own accumulation and nothing here is measured. The thread pool is the concurrency
//! boundary. The backward chain, the softmax rule and the kernels are the contracts of `grad::model`,
//! `grad::head` and `grad::dense`. Not covered: GPU `fp32` accumulation, `torch.multinomial`'s
//! stream, the `A`/`B` moments, `ho_lb`, `ho_lw`, and any activation the [`Tape`] does not carry.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    reason = "the single-letter names are the schedule's symbols (`t`, `e`, `o`, `s`, `g`, `w`), `g` \
              names the output gradient and `w` the squared input norm; the `u64 as i64` narrows a \
              Q32.32 sampler word clamped below `ONE`; the backward is one linear composition \
              with the tap widths and the layer loop passes them through"
)]

use crate::engine::chart::pool::{chunks_mut_global, width_global, FirstErr};
use crate::engine::chart::task::ChunkTask;
use crate::engine::{RopeTable, ONE};
use crate::learn::grad::attn::{self, AttnError};
use crate::learn::grad::dense::{self, DenseError};
use crate::learn::grad::head;
use crate::learn::grad::model::{Config, Grads, LayerGrads, ModelError, Tape, Weights};
use crate::learn::solve::q32::{round_q32_even, rshift_rne, Fx};
use crate::numeric::fixed_exp::fixed_exp;

/// The diagonal the reference adds after the mean-diagonal normalisation (`fold_transfer.py:1323`):
/// `0.01` as a `Q32.32` word, `round(0.01·2³²) = 42_949_673`.
pub const DEFAULT_EPS: i64 = 42_949_673;

/// Lower edge of the `fixed_exp` window, `-16·ONE`: a shifted logit below it has `p = 0`.
const FLUSH: i128 = -16 * (1_i128 << 32);

/// The kinds, in the reference's `KINDS` order (`fold_transfer.py:47-48`).
pub const N_KINDS: usize = 7;

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
    /// [`HocaptureError::Overflow`] when a width product leaves `usize`.
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
    /// [`HocaptureError::Overflow`] when a width product leaves `usize`.
    pub fn input_width(self, cfg: &Config) -> Result<usize, HocaptureError> {
        Ok(match self {
            Kind::Q | Kind::K | Kind::V | Kind::Gate | Kind::Up => cfg.n_embd,
            Kind::O => cfg.q_out().map_err(HocaptureError::Model)?,
            Kind::Down => cfg.n_ff,
        })
    }

    /// The input row block of `layer` that weights this kind's update: `ln1_out` for q/k/v,
    /// `attn_out` for o, `ln2_out` for gate/up, `m` for down.
    #[must_use]
    pub fn input(self, layer: &crate::learn::grad::model::LayerTape) -> &[i64] {
        match self {
            Kind::Q | Kind::K | Kind::V => &layer.ln1_out,
            Kind::O => &layer.attn_out,
            Kind::Gate | Kind::Up => &layer.ln2_out,
            Kind::Down => &layer.m,
        }
    }

    /// The kind's output width from the accumulator's own fixed geometry.
    fn width_at(self, cap: &Hocapture) -> usize {
        match self {
            Kind::Q | Kind::O => cap.q_out,
            Kind::K | Kind::V => cap.kv_out,
            Kind::Gate | Kind::Up => cap.n_ff,
            Kind::Down => cap.n_embd,
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
    /// A width product left `usize`, a sample counter left `u64`, a checked product left `i128`, or a
    /// rounded word left the grid.
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

/// The checked column sums of a `t × w` row-major buffer.
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

/// `RMSNorm` backward over every row of a `t × w` buffer: `(dx, Σ_r dgamma_r)`, the composition
/// `crate::learn::grad::model`'s private `rmsnorm_bwd_rows` performs.
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
    #[must_use]
    pub fn get(&self, l: usize, kind: Kind) -> Option<&[i64]> {
        self.layers.get(l).map(|lt| lt.get(kind))
    }
}

/// The whole-model backward over `dlogits` (`t × vocab`), returning the same [`Grads`] as
/// [`crate::learn::grad::model::backward`] **and** the seven per-layer output gradients it does not
/// expose.
///
/// The chain, the kernel calls, the accumulation order and the single rounding per output are
/// [`crate::learn::grad::model::backward`]'s (`src/learn/grad/model.rs:739-817`); the taps are read
/// where the reference's `gq_capture` taps (`fold_transfer.py:1262`): `∂L/∂o` at the residual split
/// `h2 = h + o`, `∂L/∂(q,k,v)` after each `RoPE` adjoint, `∂L/∂(gate,up)` out of `SwiGLU`,
/// `∂L/∂(down output)` the incoming hidden gradient.
///
/// # Errors
/// [`HocaptureError::Shape`] when the tape, `dlogits` or a buffer disagrees with the geometry;
/// [`HocaptureError::Overflow`] on a checked add or a length product; and the
/// [`HocaptureError::Model`] refusals of the kernels.
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
    let rope = RopeTable::new(
        crate::engine::num::fp::F64::from_f64(cfg.rope_theta),
        cfg.head_dim,
    )
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

/// `n / d` rounded to nearest, ties away from zero; `d > 0`. `q32` has no general divide-and-round,
/// so this is the file's one such rule (it is `grad::head`'s private one).
fn round_div(n: i128, d: i128) -> i128 {
    let (q, r) = (n / d, n % d);
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

/// `x` narrowed to `i64`, or [`HocaptureError::Overflow`].
fn to64(x: i128) -> Result<i64, HocaptureError> {
    i64::try_from(x).map_err(|_| HocaptureError::Overflow)
}

/// `a + b` in the [`Fx`] carrier: the operand with the larger exponent is kept, the other is
/// right-shifted by the exponent gap (round to nearest, ties to even) and the significands are added
/// in `i128`, renormalised once by [`Fx::from_i128`]. A gap of 63 bits or more drops the smaller
/// term, whose magnitude is then under the larger significand's last place.
///
/// # Errors
/// [`HocaptureError::Overflow`] when a checked exponent difference, shift or add leaves its range
/// (unreachable for accumulators built from Q32.32 products; kept for totality).
fn fx_add(a: &Fx, b: &Fx) -> Result<Fx, HocaptureError> {
    if a.m == 0 || b.m == 0 {
        return Ok(if a.m == 0 { b.clone() } else { a.clone() });
    }
    let (hi, lo) = if a.e >= b.e { (a, b) } else { (b, a) };
    let gap = hi.e.checked_sub(lo.e).ok_or(HocaptureError::Overflow)?;
    if gap > 62 {
        return Ok(hi.clone());
    }
    let shift = u32::try_from(gap).map_err(|_| HocaptureError::Overflow)?;
    let shifted = rshift_rne(i128::from(lo.m), shift).ok_or(HocaptureError::Overflow)?;
    let sum = i128::from(hi.m)
        .checked_add(shifted)
        .ok_or(HocaptureError::Overflow)?;
    Fx::from_i128(sum, hi.e).map_err(|_| HocaptureError::Overflow)
}

/// Softmax of one Q32.32 logit row, `pᵢ = round(eᵢ·ONE/S)`.
///
/// The rule is [`head`]'s private `softmax_row` (`src/learn/grad/head.rs:279-304`): shift by the row
/// maximum, `eᵢ = fixed_exp(uᵢ)` with `uᵢ < −16·ONE` flushed to `0`, `S = Σeᵢ`, one rounding per
/// `pᵢ`. It is mirrored here over the same public [`fixed_exp`] because `head`'s function is private
/// and this task does not own `head.rs`; the gate pins the two through `kl_row`
/// (`softmax_matches_head`).
///
/// # Errors
/// [`HocaptureError::Shape`] for an empty row, or when [`fixed_exp`] refuses the shifted value
/// (unreachable: every shifted value is `≤ 0`); [`HocaptureError::Overflow`] when a sum or a
/// probability leaves `i64`.
///
/// # Complexity
/// `O(v)`: `v` comparisons, `v` [`fixed_exp`] calls and `v` `i128` divisions; `O(v)` words out.
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
    // One flat pass: the fragment carries no `continue` and no `?` inside a loop body, so
    // `softmax_term` holds the flush edge and the two refusals and each loop carries its exit.
    let mut err: Option<HocaptureError> = None;
    for i in 0..v {
        if err.is_none() {
            let u = i128::from(z[i]) - i128::from(m);
            match softmax_term(u) {
                Ok(ev) => {
                    e[i] = ev;
                    sum += i128::from(ev);
                }
                Err(e) => err = Some(e),
            }
        }
    }
    if let Some(e) = err {
        return Err(e);
    }
    let mut p = vec![0i64; v];
    let one = i128::from(ONE);
    let mut err = None;
    for i in 0..v {
        if err.is_none() {
            match to64(round_div(i128::from(e[i]) * one, sum)) {
                Ok(pi) => p[i] = pi,
                Err(e) => err = Some(e),
            }
        }
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(p)
}

/// `exp(z_i − m)` for one shifted Q32.32 logit, zero below the [`FLUSH`] edge. `O(1)`;
/// [`HocaptureError::Overflow`] when the shifted word leaves `i64`, the shape refusal when
/// [`fixed_exp`] refuses.
fn softmax_term(u: i128) -> Result<i64, HocaptureError> {
    if u < FLUSH {
        return Ok(0);
    }
    let zi = to64(u)?;
    fixed_exp(zi).map_err(|_| shape("fixed_exp refused"))
}

/// The chunked head's `Σ_t log p_t` read at the sampled targets, negated: the sampled-label
/// cross-entropy the taps differentiate, in Q32.32.
///
/// This is `head::log_softmax_row` per row, summed in exact `i128` and rounded once, so the gate's
/// central difference does not go through [`softmax_row`]'s own rounding.
///
/// # Errors
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

/// `p_i − [i = s_r]` for row `r`.
///
/// # Errors
/// `Shape` for a target past the vocabulary, `Overflow` from the narrowing.
///
/// # Complexity
/// `O(vocab)`: one [`softmax_row`] and one write per entry.
fn grad_row(
    logits: &[i64],
    targets: &[u32],
    vocab: usize,
    r: usize,
) -> Result<Vec<i64>, HocaptureError> {
    let target = *targets
        .get(r)
        .ok_or_else(|| shape("logits_grad_ce target"))?;
    let id = usize::try_from(target).map_err(|_| HocaptureError::Overflow)?;
    if id >= vocab {
        return Err(shape("logits_grad_ce target >= vocab"));
    }
    let p = softmax_row(&logits[r * vocab..(r + 1) * vocab])?;
    let mut out = p;
    out[id] = out[id].checked_sub(ONE).ok_or(HocaptureError::Overflow)?;
    Ok(out)
}

/// The logit gradient of the sampled-label loss `Σ_t CE(z_t, s_t)`: `out[t·vocab + i] = p_ti −
/// [i = s_t]`, in Q32.32.
///
/// # Errors
/// [`HocaptureError::Shape`] when `logits.len() != t·vocab`, `out.len() != t·vocab`, `targets.len()
/// != t`, or a target is past the vocabulary.
///
/// # Complexity
/// Time `O(t·vocab)`, once per draw (`K` per capture): a per-token vector loop, not a matrix
/// pass, so it stays serial and the Gram below takes the file's pool budget.
pub fn logits_grad_ce(
    logits: &[i64],
    targets: &[u32],
    vocab: usize,
    out: &mut [i64],
) -> Result<(), HocaptureError> {
    let t = targets.len();
    expect(t, vocab, logits.len(), "logits_grad_ce logits")?;
    expect(t, vocab, out.len(), "logits_grad_ce out")?;
    if vocab == 0 || t == 0 {
        return Err(shape("logits_grad_ce dimensions"));
    }
    for r in 0..t {
        let row = grad_row(logits, targets, vocab, r)?;
        out[r * vocab..(r + 1) * vocab].copy_from_slice(&row);
    }
    Ok(())
}

/// One xorshift64\* step, returning the uniform `u ∈ [0, ONE)` of its 53 high bits.
///
/// `u = (r mod 2⁵³)/2⁵³` in Q32.32 is `u·2³² = (r >> 32)` truncated; the +`2²⁰` before the shift
/// rounds to nearest instead, and the clamp keeps the word strictly below `ONE` so the inverse-CDF
/// walk can still pass a row whose probabilities sum to `ONE`.
fn xorshift64s(state: &mut u64) -> i64 {
    let mut s = if *state == 0 { SEED_BIAS } else { *state };
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    *state = s;
    let r = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
    let word = ((r >> 11).wrapping_add(1 << 20)) >> 21;
    let cap = (ONE as u64) - 1;
    let w = if word > cap { cap } else { word };
    w as i64
}

/// `K` sampled targets per token from the teacher softmax of `logits`, draw-major then token-major:
/// `out[j·t + i]` is draw `j`'s target at token `i`.
///
/// The distribution is the reference's (`torch.multinomial(probs, 1)` per draw over the `N·T`
/// rows, `fold_transfer.py:1260`); the stream is this file's inline xorshift64\* from `seed`. The
/// inverse-CDF walk reads the same Q32.32 softmax row the loss gradient uses, in exact `i128`, and
/// falls back to the last index with non-zero probability when the uniform lands past the row's
/// rounding residue.
///
/// # Errors
/// [`HocaptureError::Shape`] when `K == 0`, `vocab == 0`, `logits.len() != t·vocab`, or `t == 0`.
///
/// # Complexity
/// Time `O(K·t·vocab)`: one [`softmax_row`] per token (a serial per-token loop) plus one
/// inverse-CDF walk per draw; space `O(t·vocab)` softmax words plus `O(K·t)` output.
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
    let mut rows = vec![0i64; t * vocab];
    for r in 0..t {
        let p = softmax_row(&logits[r * vocab..(r + 1) * vocab])?;
        rows[r * vocab..(r + 1) * vocab].copy_from_slice(&p);
    }
    let mut state = if seed == 0 { SEED_BIAS } else { seed };
    let mut out = vec![0u32; k * t];
    // Flag form: the inverse-CDF walk is its own function (no `break` in a nested loop) and the
    // narrowing refusal is carried out of the draw loop instead of returned inside it.
    let mut failed = false;
    for j in 0..k {
        for r in 0..t {
            if !failed {
                let p = &rows[r * vocab..(r + 1) * vocab];
                let draw = i128::from(xorshift64s(&mut state));
                let pick = pick_from_row(p, draw);
                match u32::try_from(pick) {
                    Ok(v) => out[j * t + r] = v,
                    Err(_) => failed = true,
                }
            }
        }
    }
    if failed {
        return Err(HocaptureError::Overflow);
    }
    Ok(out)
}

/// The inverse-CDF draw on one Q32.32 row: the first index whose cumulative probability passes
/// `draw`, else the last index of positive probability. `O(v)` comparisons; space `O(1)`.
fn pick_from_row(p: &[i64], draw: i128) -> usize {
    let mut cum: i128 = 0;
    let mut pick = p.len() - 1;
    let mut found = false;
    let mut i = 0;
    while i < p.len() && !found {
        cum += i128::from(p[i]);
        if draw < cum {
            pick = i;
            found = true;
        }
        i += 1;
    }
    if !found {
        for i in 0..p.len() {
            if p[i] > 0 {
                pick = i;
            }
        }
    }
    pick
}

/// One `(layer, kind)` raw accumulator: the scatter `Σ_t w_t g_t g_tᵀ` and the two Schäfer–Strimmer
/// inputs of the same samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KindStats {
    /// `Σ_t w_t g_t g_tᵀ`, row-major `O × O`, **upper triangle only**, Q32.32 words.
    pub ho: Vec<i64>,
    /// `Σ_t ‖s_t‖⁴` with `s_ti = √w_t·g_ti` (`fold_transfer.py:1282`), in the [`Fx`] carrier: a
    /// fourth power leaves the Q32.32 grid on any real model.
    pub s4: Fx,
    /// `Σ_t s_ti⁴` per output coordinate (`fold_transfer.py:1283`), `O` long, the same carrier.
    pub q: Vec<Fx>,
    /// `(token, draw)` pairs accumulated: the reference's `n = K·N·T` (`fold_transfer.py:1312`).
    pub samples: u64,
}

impl KindStats {
    /// The zero accumulator of width `O`.
    ///
    /// # Errors
    /// [`HocaptureError::Overflow`] when `O²` leaves `usize`.
    ///
    /// # Complexity
    /// Time `O(O²)`; space `O(O²)` words for `ho` and `O` for `q`.
    pub fn zeros(o: usize) -> Result<KindStats, HocaptureError> {
        let o2 = o.checked_mul(o).ok_or(HocaptureError::Overflow)?;
        Ok(KindStats {
            ho: vec![0; o2],
            s4: Fx { m: 0, e: 0 },
            q: vec![Fx { m: 0, e: 0 }; o],
            samples: 0,
        })
    }
}

/// Every kind's raw accumulator for one layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerStats {
    /// Indexed by [`Kind::index`].
    pub kinds: [KindStats; N_KINDS],
}

impl LayerStats {
    /// The accumulator of `kind`.
    #[must_use]
    pub fn kind(&self, kind: Kind) -> &KindStats {
        &self.kinds[kind.index()]
    }
}

/// The squared input norm `‖x_t‖²` of one Q32.32 row, as one word: the exact `i128` sum of the
/// squares, rounded once.
///
/// # Complexity
/// `O(w)`; space `O(1)`.
fn token_weight(x: &[i64]) -> Result<i64, HocaptureError> {
    let mut acc: i128 = 0;
    for &v in x {
        let sq = i128::from(v)
            .checked_mul(i128::from(v))
            .ok_or(HocaptureError::Overflow)?;
        acc = acc.checked_add(sq).ok_or(HocaptureError::Overflow)?;
    }
    round_q32_even(acc).map_err(|_| HocaptureError::Overflow)
}

/// `a_t = round(w_t·g_ti)` in Q32.32: the input-weighted output gradient that both `HO` and `s²`
/// build on.
///
/// # Errors
/// [`HocaptureError::Overflow`] when the product leaves the grid.
fn weighted_grad(w_t: i64, g_ti: i64) -> Result<i64, HocaptureError> {
    round_q32_even(i128::from(w_t) * i128::from(g_ti)).map_err(|_| HocaptureError::Overflow)
}

/// The pool task of [`accumulate_kind`]'s `HO` pass: one band of upper-triangle output rows.
struct HoRows<'a> {
    weights: &'a [i64],
    g: &'a [i64],
    o: usize,
    t: usize,
    first: &'a FirstErr<usize, HocaptureError>,
}

impl ChunkTask<i64> for HoRows<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [i64]) {
        let row0 = start / self.o;
        let rows = piece.len() / self.o;
        for i in 0..rows {
            let row = &mut piece[i * self.o..(i + 1) * self.o];
            if let Err(e) = ho_row(row, row0 + i, self.weights, self.g, self.t, self.o) {
                self.first.record(row0 + i, e);
                return;
            }
        }
    }
}

/// `HO[gi][j] += Σ_t round(w_t·g_tgi)·g_tj` for `j ≥ gi`: exact `i128` per entry, rounded once,
/// added into the row's upper triangle (the lower one stays zero for [`Hocapture::finish`]).
///
/// # Errors
/// [`HocaptureError::Overflow`] when an intermediate or the entry's round leaves its range.
///
/// # Complexity
/// Time `O(t·(O − gi))` multiply-adds and `t` roundings (one per token, hoisted out of the `j`
/// loop); space `O(O)` `i128` accumulators.
fn ho_row(
    row: &mut [i64],
    gi: usize,
    weights: &[i64],
    g: &[i64],
    t: usize,
    o: usize,
) -> Result<(), HocaptureError> {
    let mut acc = vec![0i128; o];
    for r in 0..t {
        let gr = &g[r * o..(r + 1) * o];
        let a = weighted_grad(weights[r], gr[gi])?;
        for j in gi..o {
            acc[j] = acc[j]
                .checked_add(
                    i128::from(a)
                        .checked_mul(i128::from(gr[j]))
                        .ok_or(HocaptureError::Overflow)?,
                )
                .ok_or(HocaptureError::Overflow)?;
        }
    }
    for j in gi..o {
        let v = round_q32_even(acc[j]).map_err(|_| HocaptureError::Overflow)?;
        row[j] = row[j].checked_add(v).ok_or(HocaptureError::Overflow)?;
    }
    Ok(())
}

/// Fold one token block of `(input, output-gradient)` pairs into `stats`: the rank-1 update
/// `ho += Σ_t ‖x_t‖² g_t g_tᵀ` and the `S4`/`Q` recurrences.
///
/// `x` is the linear's input, `t × w` row-major in Q32.32; `g` is the loss's gradient with respect to
/// the linear's output, `t × o` row-major in Q32.32. `w_t = ‖x_t‖²` as one word, and
/// `s_ti² = round(round(w_t·g_ti)·g_ti)`, exactly the reference's `s2 = g²·w`
/// (`fold_transfer.py:1281`). `ho` gains the upper triangle only; [`Hocapture::finish`] mirrors it.
///
/// # Errors
/// [`HocaptureError::Shape`] when `x.len() != t·w`, `g.len() != t·o`, `o == 0` or `w == 0`, or
/// `stats` has the wrong `ho`/`q` widths; [`HocaptureError::Overflow`] when the sample counter
/// leaves `u64`, an exact `i128` sum leaves its range, or a rounded `ho` word leaves the grid.
/// (`s4`/`q` are carried in [`Fx`], so their fourth powers do not have to fit the grid.)
///
/// # Complexity
/// Time `O(t·(w + O²))`: one `w`-term norm and `O(O+1)/2` rank-1 multiply-adds plus `O` recurrence
/// terms per token. Unit cost `t·w + t·O²/2` word ops against `7·n_layer·K` calls per capture, so
/// the `t·O²/2` `HO` pass (≥ `10⁶` word ops at any model width) is split by output row across
/// [`width_global`] pool tasks; the `O(t)` weight, `Q` and `S4` passes are vector loops and serial.
/// Space `O(t)` weights plus `O(O)` per task.
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

    // `w_t = ‖x_t‖²` per token.
    let mut weights = vec![0i64; t];
    for (r, slot) in weights.iter_mut().enumerate() {
        *slot = token_weight(&x[r * w..(r + 1) * w])?;
    }

    // `HO` upper-triangle rows on the pool.
    let chunk_rows = o.div_ceil(width_global()).max(1);
    let first: FirstErr<usize, HocaptureError> = FirstErr::new();
    chunks_mut_global(
        &mut stats.ho,
        chunk_rows * o,
        &HoRows {
            weights: &weights,
            g,
            o,
            t,
            first: &first,
        },
    );
    if let Some((_, e)) = first.take() {
        return Err(e);
    }

    // `Q_i = Σ_t s_ti⁴` and `S4 = Σ_t (Σ_i s_ti²)²`, exact Q64.64 totals rounded once per call.
    let mut q = vec![0i128; o];
    let mut s4: i128 = 0;
    for r in 0..t {
        let gr = &g[r * o..(r + 1) * o];
        let mut n2: i128 = 0;
        for (i, &gi) in gr.iter().enumerate() {
            let s2 = weighted_grad(weighted_grad(weights[r], gi)?, gi)?;
            q[i] = q[i]
                .checked_add(
                    i128::from(s2)
                        .checked_mul(i128::from(s2))
                        .ok_or(HocaptureError::Overflow)?,
                )
                .ok_or(HocaptureError::Overflow)?;
            n2 = n2
                .checked_add(i128::from(s2))
                .ok_or(HocaptureError::Overflow)?;
        }
        s4 = s4
            .checked_add(n2.checked_mul(n2).ok_or(HocaptureError::Overflow)?)
            .ok_or(HocaptureError::Overflow)?;
    }
    for (dst, v) in stats.q.iter_mut().zip(&q) {
        let add = Fx::from_i128(*v, -64).map_err(|_| HocaptureError::Overflow)?;
        *dst = fx_add(dst, &add)?;
    }
    let total = Fx::from_i128(s4, -64).map_err(|_| HocaptureError::Overflow)?;
    stats.s4 = fx_add(&stats.s4, &total)?;

    let rows = u64::try_from(t).map_err(|_| HocaptureError::Overflow)?;
    stats.samples = stats
        .samples
        .checked_add(rows)
        .ok_or(HocaptureError::Overflow)?;
    Ok(())
}

/// The output-side (YAQA Sketch A) capture: one raw accumulator per `(layer, kind)`, plus the fixed
/// draw count and seed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hocapture {
    /// Draws per token, the reference's `K`.
    k: usize,
    /// The sampler seed.
    seed: u64,
    n_embd: usize,
    q_out: usize,
    kv_out: usize,
    n_ff: usize,
    vocab: usize,
    /// Per trunk layer, in forward order.
    layers: Vec<LayerStats>,
}

impl Hocapture {
    /// The zero accumulator for `cfg`'s geometry, `k` draws per token and the sampler seed `seed`.
    ///
    /// # Errors
    /// [`HocaptureError::Shape`] for `k == 0` or a zero model dimension;
    /// [`HocaptureError::Overflow`] when `q_out`, `kv_out` or a squared width leaves `usize`.
    ///
    /// # Complexity
    /// Time `O(n_layer·(q_out² + kv_out² + n_ff² + n_embd²))`; the same in space.
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
    /// As [`Hocapture::add_taps`], plus the forward's and the sampler's refusals.
    ///
    /// # Complexity
    /// One forward, `K` backward passes and `O(K·t·Σ_kind(w_kind + O_kind²))` accumulation terms.
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
    /// As [`Hocapture::add_taps`], plus the forward's refusals.
    ///
    /// # Complexity
    /// As [`Hocapture::accumulate`].
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
    /// input norm. The draw loop is serial because each iteration dispatches the engine pool and a
    /// pool task must not dispatch the pool.
    ///
    /// # Errors
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
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// The sampler seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The raw accumulators, per layer: `ho = Σ_t w_t g_t g_tᵀ` (upper triangle, Q32.32 words),
    /// `s4`, `q` (the [`Fx`] carrier) and the sample count.
    #[must_use]
    pub fn stats(&self) -> &[LayerStats] {
        &self.layers
    }

    /// The normalised copy: per `(layer, kind)`, `ho / mean(diag(ho))` — the fixed-point ratio
    /// `round(ho·O·2³²/Σdiag)` — then `eps` on the diagonal, the reference's
    /// `x / x.diagonal().mean().clamp_min(1e-30)` and `x.diagonal().add_(0.01)`
    /// (`fold_transfer.py:1322-1323`), with the caller supplying `eps` ([`DEFAULT_EPS`] is the
    /// reference's `0.01`). The floor `1e-30` is below the grid's quantum, so a zero diagonal is the
    /// only reachable branch and it yields zeros. `s4`, `q` and `samples` pass through unchanged, so
    /// `ho_lw` can still be formed on the raw scatter as the reference does.
    ///
    /// # Errors
    /// [`HocaptureError::Shape`] when a kind has accumulated no samples; [`HocaptureError::Overflow`]
    /// when `eps` leaves the grid or a scaled entry leaves `i64`.
    ///
    /// # Complexity
    /// `O(n_layer·Σ_kind O_kind²)` time, one `ho` copy per `(layer, kind)`.
    pub fn finish(&self, eps: i64) -> Result<Vec<LayerStats>, HocaptureError> {
        let mut out = Vec::with_capacity(self.layers.len());
        for ls in &self.layers {
            let mut kinds: Vec<KindStats> = Vec::with_capacity(N_KINDS);
            for kind in Kind::ALL {
                let ks = ls.kind(kind);
                if ks.samples == 0 {
                    return Err(shape("finish over zero samples"));
                }
                let o = kind.width_at(self);
                let diag: i128 = (0..o)
                    .filter_map(|i| ks.ho.get(i * o + i))
                    .map(|&v| i128::from(v))
                    .sum();
                let num = i128::try_from(o)
                    .ok()
                    .and_then(|d| d.checked_shl(32))
                    .ok_or(HocaptureError::Overflow)?;
                let mut ho = vec![0i64; o * o];
                for i in 0..o {
                    for j in i..o {
                        let w = if diag <= 0 {
                            0
                        } else {
                            let raw = i128::from(ks.ho[i * o + j])
                                .checked_mul(num)
                                .ok_or(HocaptureError::Overflow)?;
                            to64(round_div(raw, diag))?
                        };
                        let v = if i == j {
                            w.checked_add(eps).ok_or(HocaptureError::Overflow)?
                        } else {
                            w
                        };
                        ho[i * o + j] = v;
                        ho[j * o + i] = v;
                    }
                }
                kinds.push(KindStats {
                    ho,
                    s4: ks.s4.clone(),
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
