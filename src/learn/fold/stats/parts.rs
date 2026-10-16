//! `f64` CPU forward pass of ONE `Qwen2` decoder layer, returning every intermediate the
//! fold-transfer reads.
//!
//! Port of `parts`, `attn_a`, `rms` and `rope_tables` from the read-only spec
//! `~/lanes/port/spec/fold_transfer.py` (`rope_tables:60-64`, `rot:66-68`,
//! `rms:70-71`, `attn_a:73-81`, `parts:83-92`). On one sequence of `t` tokens and a residual input
//! `h` (row-major `[t, hidden]`), [`parts`] runs:
//!
//! ```text
//! xa  = rms(h, ln1)                       # [t, hidden]
//! q   = xa·qᵀ + qb                        # [t, n_head·head_dim]
//! k   = xa·kᵀ + kb,  v = xa·vᵀ + vb       # [t, n_kv_head·head_dim]
//! a   = attn_a(q, k, v)                   # rotate-half RoPE, GQA repeat_interleave, causal softmax
//! mid = h + a·oᵀ                          # [t, hidden]
//! xf  = rms(mid, ln2)
//! m   = silu(xf·gateᵀ) · (xf·upᵀ)         # [t, inter]
//! out = mid + m·downᵀ                     # [t, hidden]
//! ```
//!
//! Every weight is row-major `[out, in]` exactly as the Python transfer-layer dictionary `TL[l]`
//! holds it (`fold_transfer.py:117-122`). [`RopeTables`] reproduces `rope_tables`: with
//! `inv[i] = theta^(-2i/head_dim)` and `emb[t, j] = t · inv[j mod (head_dim/2)]`, the half-rotation
//! is `x ↦ x·cos(emb) + rot(x)·sin(emb)` with `rot(x) = cat(-x[upper], x[lower])` (`rot`, line 68).
//!
//! # Deviations from the Python
//!
//! - The reference computes in `torch.float64` on CUDA (`DEV = "cuda"`), which contracts some
//!   multiply-adds and evaluates `exp`/`cos`/`sin` in a device library. This port is a portable
//!   scalar `f64` CPU path with no FMA contraction and the host libm; the recurrence and operand
//!   order are the reference's, so results agree to `f64` round-off, not bit-for-bit across
//!   platforms.
//! - `rope_tables` builds `CHUNK = 512` rows once for the whole run; [`RopeTables::new`] builds
//!   `t_max` rows for the caller's sequence.
//! - Refusals are typed ([`PartsError`]) where the reference would raise an opaque `torch`
//!   broadcasting error or produce a non-finite value from a non-finite `eps`/`theta`. The
//!   admissible domain is otherwise unchanged.
//!
//! # Evidence grade
//!
//! `tested`: the gate's `cargo test` on the build path (`tests/learn_fold_parts.rs`). No Lean
//! refinement links this file; it is not on the extracted fragment.
//!
//! # Trust boundary
//!
//! Covers the `f64` arithmetic of [`rms`], [`attn_a`], [`parts`] and [`RopeTables::new`] on
//! `&[f64]` inputs. Not covered: the model loader that produces the weight tensors, any
//! `torch`/CUDA codegen, and the rest of the transfer (`gptq_*`, the refit steps, KL). Those are
//! other modules' contracts.

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    reason = "the dense kernels index the row-major `[t, width]` buffers by `r/i/j/k/o/d/h`, name the \
              head geometry `nh`/`nkv`/`hd` and the residual width `hid` as the reference does, \
              accumulate `Σ x²` as `x*x` (not `mul_add`) to keep the elementwise arithmetic the \
              reference's, and cast the position and dimension counts `usize → f64`, a deliberate \
              lossy cast of a dimension"
)]

/// The geometry of one `Qwen2` decoder layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerCfg {
    /// Query head count (`num_attention_heads`).
    pub n_head: usize,
    /// Key/value head count (`num_key_value_heads`); `n_head % n_kv_head == 0` is required.
    pub n_kv_head: usize,
    /// Per-head width (`head_dim`); even.
    pub head_dim: usize,
    /// Residual stream width; `hidden == n_head * head_dim` is required.
    pub hidden: usize,
    /// Feed-forward width (`intermediate_size`).
    pub inter: usize,
    /// `RMSNorm` epsilon.
    pub eps: f64,
    /// `RoPE` base (`rope_theta`); finite and positive.
    pub theta: f64,
}

impl LayerCfg {
    /// Refuse an invalid head geometry or a non-finite `eps`/`theta`. `O(1)`.
    fn validate(&self) -> Result<(), PartsError> {
        if !self.eps.is_finite() || !self.theta.is_finite() || self.theta <= 0.0 {
            return Err(PartsError::NonFinite);
        }
        if self.n_head == 0
            || self.n_kv_head == 0
            || self.head_dim == 0
            || !self.head_dim.is_multiple_of(2)
        {
            return Err(PartsError::Shape);
        }
        let wide = self
            .n_head
            .checked_mul(self.head_dim)
            .ok_or(PartsError::Shape)?;
        if wide != self.hidden || !self.n_head.is_multiple_of(self.n_kv_head) {
            return Err(PartsError::Shape);
        }
        Ok(())
    }
}

/// A refusal from the layer forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartsError {
    /// A slice length disagrees with the declared shape, a dimension product left `usize`, or the
    /// head geometry is invalid (`n_kv_head == 0`, `n_head % n_kv_head != 0`, an odd `head_dim`, or
    /// `hidden != n_head * head_dim`).
    Shape,
    /// `eps` is not finite, or `theta` is not a finite positive value.
    NonFinite,
    /// The sequence length `t` exceeds the `t_max` the [`RopeTables`] were built for.
    RopeTooShort,
}

/// The twelve `f64` weight tensors of one `Qwen2` decoder layer, each row-major `[out, in]`.
///
/// The fields mirror the Python transfer-layer dictionary keys (`fold_transfer.py:117-122`); the
/// output-projection, gate, up and down tensors have no bias in `Qwen2`, so only `q`/`k`/`v` carry
/// one.
#[derive(Clone, Copy, Debug)]
pub struct LayerF64<'a> {
    /// `input_layernorm.weight`, length `hidden`.
    pub ln1: &'a [f64],
    /// `post_attention_layernorm.weight`, length `hidden`.
    pub ln2: &'a [f64],
    /// `self_attn.q_proj.weight`, row-major `[n_head·head_dim, hidden]`.
    pub q: &'a [f64],
    /// `self_attn.q_proj.bias`, length `n_head·head_dim`.
    pub qb: &'a [f64],
    /// `self_attn.k_proj.weight`, row-major `[n_kv_head·head_dim, hidden]`.
    pub k: &'a [f64],
    /// `self_attn.k_proj.bias`, length `n_kv_head·head_dim`.
    pub kb: &'a [f64],
    /// `self_attn.v_proj.weight`, row-major `[n_kv_head·head_dim, hidden]`.
    pub v: &'a [f64],
    /// `self_attn.v_proj.bias`, length `n_kv_head·head_dim`.
    pub vb: &'a [f64],
    /// `self_attn.o_proj.weight`, row-major `[hidden, n_head·head_dim]`.
    pub o: &'a [f64],
    /// `mlp.gate_proj.weight`, row-major `[inter, hidden]`.
    pub gate: &'a [f64],
    /// `mlp.up_proj.weight`, row-major `[inter, hidden]`.
    pub up: &'a [f64],
    /// `mlp.down_proj.weight`, row-major `[hidden, inter]`.
    pub down: &'a [f64],
}

/// The half-rotation tables [`attn_a`] applies to `q` and `k`.
///
/// `cos` and `sin` are each row-major `[t_max, head_dim]`, with
/// `emb[t, j] = t · theta^(-2·(j mod head_dim/2)/head_dim)` (`rope_tables`, lines 60-64): the
/// frequency vector is concatenated with itself, so the second half of a head rotates with the same
/// angle as the first.
#[derive(Clone, Debug)]
pub struct RopeTables {
    /// `cos(emb)`, row-major `[t_max, head_dim]`.
    pub cos: Vec<f64>,
    /// `sin(emb)`, row-major `[t_max, head_dim]`.
    pub sin: Vec<f64>,
    head_dim: usize,
    t_max: usize,
}

impl RopeTables {
    /// Build the tables for `t_max` positions. `O(t_max·head_dim)` time and space.
    ///
    /// # Errors
    /// [`PartsError::Shape`] if the geometry is invalid or `t_max·head_dim` leaves `usize`;
    /// [`PartsError::NonFinite`] if `eps`/`theta` is not a finite positive value.
    pub fn new(cfg: &LayerCfg, t_max: usize) -> Result<Self, PartsError> {
        cfg.validate()?;
        let hd = cfg.head_dim;
        let half = hd / 2;
        let n = t_max.checked_mul(hd).ok_or(PartsError::Shape)?;
        let mut cos = vec![0.0f64; n];
        let mut sin = vec![0.0f64; n];
        let mut inv = vec![0.0f64; half];
        for i in 0..half {
            let e = (2.0 * i as f64) / (hd as f64);
            inv[i] = 1.0 / cfg.theta.powf(e);
        }
        for p in 0..t_max {
            let base = p * hd;
            for j in 0..hd {
                let emb = (p as f64) * inv[j % half];
                cos[base + j] = emb.cos();
                sin[base + j] = emb.sin();
            }
        }
        Ok(Self {
            cos,
            sin,
            head_dim: hd,
            t_max,
        })
    }

    /// The head width the tables were built for.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The position count the tables cover.
    #[must_use]
    pub fn t_max(&self) -> usize {
        self.t_max
    }
}

/// Every intermediate of one decoder layer, each row-major `[t, width]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Parts {
    /// `rms(h, ln1)`, `[t, hidden]`.
    pub xa: Vec<f64>,
    /// Query projection `xa·qᵀ + qb`, `[t, n_head·head_dim]`.
    pub q: Vec<f64>,
    /// Key projection `xa·kᵀ + kb`, `[t, n_kv_head·head_dim]`.
    pub k: Vec<f64>,
    /// Value projection `xa·vᵀ + vb`, `[t, n_kv_head·head_dim]`.
    pub v: Vec<f64>,
    /// Causal GQA attention output, `[t, n_head·head_dim]`.
    pub a: Vec<f64>,
    /// Residual after attention, `h + a·oᵀ`, `[t, hidden]`.
    pub mid: Vec<f64>,
    /// `rms(mid, ln2)`, `[t, hidden]`.
    pub xf: Vec<f64>,
    /// Gate projection `xf·gateᵀ`, `[t, inter]`.
    pub g: Vec<f64>,
    /// Up projection `xf·upᵀ`, `[t, inter]`.
    pub u: Vec<f64>,
    /// `silu(g)·u`, `[t, inter]`.
    pub m: Vec<f64>,
    /// Layer output `mid + m·downᵀ`, `[t, hidden]`.
    pub out: Vec<f64>,
}

/// `a·b`, refused when the product leaves `usize`.
fn area(a: usize, b: usize) -> Result<usize, PartsError> {
    a.checked_mul(b).ok_or(PartsError::Shape)
}

/// `silu(g) = g·sigmoid(g) = g / (1 + exp(-g))`, `O(1)`.
#[must_use]
pub fn silu(g: f64) -> f64 {
    g / (1.0 + (-g).exp())
}

/// Row-major dense projection `out[r, o] = bias[o] + Σ_i w[o, i]·x[r, i]`.
///
/// # Complexity
///
/// Time `O(t·out_w·in_w)` multiply-adds; space `O(t·out_w)` for the output. Derivation: per output
/// row `r` and column `o` the inner loop is a length-`in_w` dot product, and the `t·out_w` rows and
/// columns are independent.
///
/// # Errors
/// [`PartsError::Shape`] if any slice length disagrees with the declared shape.
pub fn linear(
    x: &[f64],
    t: usize,
    in_w: usize,
    w: &[f64],
    bias: Option<&[f64]>,
    out_w: usize,
) -> Result<Vec<f64>, PartsError> {
    if x.len() != area(t, in_w)? || w.len() != area(out_w, in_w)? {
        return Err(PartsError::Shape);
    }
    if let Some(b) = bias {
        if b.len() != out_w {
            return Err(PartsError::Shape);
        }
    }
    let mut out = vec![0.0f64; area(t, out_w)?];
    for r in 0..t {
        let xrow = &x[r * in_w..r * in_w + in_w];
        let orow = &mut out[r * out_w..r * out_w + out_w];
        for o in 0..out_w {
            let wrow = &w[o * in_w..o * in_w + in_w];
            let mut acc = bias.map_or(0.0, |b| b[o]);
            for i in 0..in_w {
                acc += wrow[i] * xrow[i];
            }
            orow[o] = acc;
        }
    }
    Ok(out)
}

/// `RMSNorm` over rows: `out[r, j] = w[j] · x[r, j] / sqrt(mean_j(x[r, :]²) + eps)` (`rms`, line 71).
///
/// # Complexity
///
/// Time `O(t·width)` multiply-adds plus `t` square roots and `t` divisions; space `O(t·width)` for
/// the output. Derivation: one length-`width` sum of squares and one `width` scale per row, rows
/// independent.
///
/// # Errors
/// [`PartsError::Shape`] if `x.len() != t·width`, `w.len() != width`, or `width == 0`;
/// [`PartsError::NonFinite`] if `eps` is not finite.
pub fn rms(x: &[f64], w: &[f64], eps: f64, t: usize, width: usize) -> Result<Vec<f64>, PartsError> {
    if !eps.is_finite() {
        return Err(PartsError::NonFinite);
    }
    if width == 0 {
        return Err(PartsError::Shape);
    }
    let n = area(t, width)?;
    if x.len() != n || w.len() != width {
        return Err(PartsError::Shape);
    }
    let mut out = vec![0.0f64; n];
    let wf = width as f64;
    for r in 0..t {
        let row = &x[r * width..r * width + width];
        let mut ss = 0.0f64;
        for &v in row {
            ss += v * v;
        }
        let inv = 1.0 / (ss / wf + eps).sqrt();
        let dst = &mut out[r * width..r * width + width];
        for j in 0..width {
            dst[j] = w[j] * (row[j] * inv);
        }
    }
    Ok(out)
}

/// Apply the `RoPE` half-rotation in place to `src` (row-major `[t, n_head·head_dim]`), writing
/// `dst`: `dst = src·cos(emb) + rot(src)·sin(emb)` with `rot(x) = cat(-x[upper], x[lower])`
/// (`rot`, line 68). `O(t·n_head·head_dim)`; `dst` and `src` are the same shape.
fn rope_apply(dst: &mut [f64], src: &[f64], t: usize, n_head: usize, hd: usize, rope: &RopeTables) {
    let half = hd / 2;
    for p in 0..t {
        let row = p * n_head * hd;
        let cb = p * hd;
        for h in 0..n_head {
            let base = row + h * hd;
            for d in 0..hd {
                let c = rope.cos[cb + d];
                let s = rope.sin[cb + d];
                let x = src[base + d];
                let r = if d < half {
                    -src[base + d + half]
                } else {
                    src[base + d - half]
                };
                dst[base + d] = x * c + r * s;
            }
        }
    }
}

/// Causal grouped-query attention with rotate-half `RoPE` (`attn_a`, lines 73-81).
///
/// `q` is row-major `[t, n_head·head_dim]`, `k` and `v` are `[t, n_kv_head·head_dim]`. Each query
/// and key head is rotated with [`RopeTables`]; the `n_kv_head` key/value heads are expanded to
/// `n_head` by `repeat_interleave(n_head / n_kv_head)`, so head `h` reads kv head
/// `h / (n_head / n_kv_head)`. Attention is the scaled dot product with scale
/// `1/sqrt(head_dim)` and a causal mask: query position `i` attends to key positions `j ≤ i` only.
/// Returns row-major `[t, n_head·head_dim]`.
///
/// # Complexity
///
/// Time `O(t·(n_head + n_kv_head)·head_dim)` for the two rotations plus
/// `O(t²·n_head·head_dim)` for the scores and the value accumulation; space `O(t·(n_head +
/// n_kv_head)·head_dim + t + head_dim)` beyond the output. Derivation: rotation is linear in the
/// token count; for each head the query `i` scores against `i + 1` keys, each a length-`head_dim`
/// dot product and a length-`head_dim` value update, and `Σ_{i<t}(i + 1) = t(t + 1)/2`.
///
/// # Errors
/// [`PartsError::Shape`] for an invalid geometry or a slice that is not the stated shape;
/// [`PartsError::NonFinite`] for non-finite `eps`/`theta`; [`PartsError::RopeTooShort`] if
/// `t > rope.t_max`, or if `rope.head_dim` differs from `cfg.head_dim`.
pub fn attn_a(
    cfg: &LayerCfg,
    rope: &RopeTables,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    t: usize,
) -> Result<Vec<f64>, PartsError> {
    cfg.validate()?;
    if rope.head_dim != cfg.head_dim {
        return Err(PartsError::Shape);
    }
    if t > rope.t_max {
        return Err(PartsError::RopeTooShort);
    }
    let nh = cfg.n_head;
    let nkv = cfg.n_kv_head;
    let hd = cfg.head_dim;
    let qt = area(area(t, nh)?, hd)?;
    let kt = area(area(t, nkv)?, hd)?;
    if q.len() != qt || k.len() != kt || v.len() != kt {
        return Err(PartsError::Shape);
    }
    let scale = 1.0 / (hd as f64).sqrt();
    let mut qr = vec![0.0f64; qt];
    let mut kr = vec![0.0f64; kt];
    rope_apply(&mut qr, q, t, nh, hd, rope);
    rope_apply(&mut kr, k, t, nkv, hd, rope);
    let rep = nh / nkv;
    let mut out = vec![0.0f64; qt];
    let mut scores = vec![0.0f64; t];
    let mut acc = vec![0.0f64; hd];
    for h in 0..nh {
        let g = h / rep;
        let gbase = g * hd;
        for i in 0..t {
            let qbase = i * nh * hd + h * hd;
            let mut row_max = f64::NEG_INFINITY;
            for j in 0..=i {
                let kbase = j * nkv * hd + gbase;
                let mut dot = 0.0f64;
                for d in 0..hd {
                    dot += qr[qbase + d] * kr[kbase + d];
                }
                let s = scale * dot;
                scores[j] = s;
                if s > row_max {
                    row_max = s;
                }
            }
            let mut denom = 0.0f64;
            acc.fill(0.0);
            for j in 0..=i {
                let e = (scores[j] - row_max).exp();
                denom += e;
                let vbase = j * nkv * hd + gbase;
                for d in 0..hd {
                    acc[d] += e * v[vbase + d];
                }
            }
            let obase = i * nh * hd + h * hd;
            for d in 0..hd {
                out[obase + d] = acc[d] / denom;
            }
        }
    }
    Ok(out)
}

/// One `Qwen2` decoder layer on a whole sequence (`parts`, lines 83-91): the sandwich of the two
/// pre-norms, causal GQA attention and the `SwiGLU` feed-forward, returning every intermediate.
///
/// `h` is row-major `[t, hidden]`. The weights are checked against the geometry before any
/// arithmetic, so a malformed layer is refused without touching an index.
///
/// # Complexity
///
/// Time `O(t·(n_head·head_dim·hidden + inter·hidden) + t²·n_head·head_dim)`: the four attention
/// projections each cost a `[t, hidden]·[w, hidden]ᵀ` product, the two `SwiGLU` projections a
/// `[t, hidden]·[inter, hidden]ᵀ` product, `down` a `[t, inter]·[hidden, inter]ᵀ` product, the two
/// norms `O(t·hidden)`, and [`attn_a`] as documented. Space `O(t·(hidden + n_head·head_dim +
/// n_kv_head·head_dim + inter))` for [`Parts`] plus the rope scratch inside [`attn_a`]. Derivation:
/// each product is a `t × out × in` triple loop; the intermediates are one `[t, width]` buffer each.
///
/// # Errors
/// [`PartsError::Shape`] for an invalid geometry or a weight/input slice that is not the stated
/// shape; [`PartsError::NonFinite`] for non-finite `eps`/`theta`; [`PartsError::RopeTooShort`] if
/// `t > rope.t_max` or `rope.head_dim` differs from `cfg.head_dim`.
pub fn parts(
    cfg: &LayerCfg,
    rope: &RopeTables,
    layer: &LayerF64,
    h: &[f64],
    t: usize,
) -> Result<Parts, PartsError> {
    cfg.validate()?;
    if rope.head_dim != cfg.head_dim {
        return Err(PartsError::Shape);
    }
    if t > rope.t_max {
        return Err(PartsError::RopeTooShort);
    }
    let hid = cfg.hidden;
    let nh = cfg.n_head;
    let nkv = cfg.n_kv_head;
    let hd = cfg.head_dim;
    let inter = cfg.inter;
    let qw = area(nh, hd)?;
    let kw = area(nkv, hd)?;
    if h.len() != area(t, hid)? {
        return Err(PartsError::Shape);
    }

    let xa = rms(h, layer.ln1, cfg.eps, t, hid)?;
    let q = linear(&xa, t, hid, layer.q, Some(layer.qb), qw)?;
    let k = linear(&xa, t, hid, layer.k, Some(layer.kb), kw)?;
    let v = linear(&xa, t, hid, layer.v, Some(layer.vb), kw)?;
    let a = attn_a(cfg, rope, &q, &k, &v, t)?;

    let attn_out = linear(&a, t, qw, layer.o, None, hid)?;
    let mut mid = vec![0.0f64; area(t, hid)?];
    for r in 0..t {
        for c in 0..hid {
            mid[r * hid + c] = h[r * hid + c] + attn_out[r * hid + c];
        }
    }

    let xf = rms(&mid, layer.ln2, cfg.eps, t, hid)?;
    let g = linear(&xf, t, hid, layer.gate, None, inter)?;
    let u = linear(&xf, t, hid, layer.up, None, inter)?;
    let mut m = vec![0.0f64; area(t, inter)?];
    for i in 0..t * inter {
        m[i] = silu(g[i]) * u[i];
    }

    let down_out = linear(&m, t, inter, layer.down, None, hid)?;
    let mut out = vec![0.0f64; area(t, hid)?];
    for r in 0..t {
        for c in 0..hid {
            out[r * hid + c] = mid[r * hid + c] + down_out[r * hid + c];
        }
    }

    Ok(Parts {
        xa,
        q,
        k,
        v,
        a,
        mid,
        xf,
        g,
        u,
        m,
        out,
    })
}
