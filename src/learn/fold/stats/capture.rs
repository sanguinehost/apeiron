//! Calibration activation-Gram (input-side Hessian) capture on the crate's own forward.
//!
//! [`Grams::accumulate`] reads the Q32.32 [`Tape`] captured by
//! [`crate::learn::grad::model::forward`] and accumulates, per trunk layer and per linear input, the
//! unweighted scatter `H = Σ_t x_t x_tᵀ` over the tokens, `x_t` the linear's input row.
//! [`Grams::finish`] mirrors the upper triangle to the lower one and applies the requested
//! normalisation. The port is the activation-Gram half of the reference `rstar_fast.py` `Gram`
//! (`~/lanes/port/spec/rstar_fast.py:225-254`), whose four `H:<key>` tensors are
//! exactly these four inputs (`rstar_fast.py:641-653`).
//!
//! | [`LayerGrams`] field | linear | [`Tape`] stage | width |
//! |---|---|---|---|
//! | `qkv` | Q/K/V projection | `ln1_out` (`attn_norm` output) | `n_embd` |
//! | `o` | output projection | `attn_out` | `q_out = n_head·head_dim` |
//! | `gate_up` | gate and up | `ln2_out` (`ffn_norm` output) | `n_embd` |
//! | `down` | down projection | `m` (`SwiGLU` product) | `n_ff` |
//!
//! # Sum or mean, weighting, accumulation precision
//!
//! The accumulator is the raw sum `Σ_t x_t x_tᵀ`; [`Normalisation`] selects what the consumer is
//! handed, and the doc comment on each variant names the reference site it reproduces. The
//! reference `Gram.add` adds one **unweighted** `aᵀa` per token batch (`rstar_fast.py:234-245`) —
//! there is no per-token weight — and `Gram.result(nt)` divides by the token count `nt = nc·CHUNK`
//! (`rstar_fast.py:247-254`, `:677`), so [`Normalisation::Mean`] is the reference's `H`; the
//! imatrix `diag(H) = (Σ_t x_t²)/nt` of `forward_all` (`rstar_fast.py:367`) confirms it.
//! [`Normalisation::MeanDiagonal`] reproduces the different normalisation `gq_capture` applies to
//! its output-gradient `HO` (mean diagonal 1, then `0.01` added to the diagonal,
//! `~/lanes/port/spec/fold_transfer.py:1322-1323`).
//!
//! Accumulation here is `f64`. The reference accumulates `fp32` (`Gram.acc`, `rstar_fast.py:231`);
//! its optional `fp16s` mode additionally fixes a per-column scale from the first batch
//! (`rstar_fast.py:238-241`). Neither rounding is reproduced: the operands are exact Q32.32→`f64`
//! (`x / 2^32`, an exact scaling by a power of two, so only `f64`'s 53-bit mantissa limits it) and
//! the f64 scatter removes the fp32 accumulation error the `--gram tf32` reference warns about.
//!
//! # Residual: what `gq_capture` captures that this module does not
//!
//! Only the activation Gram is ported. `gq_capture` (`fold_transfer.py:1163-1341`) additionally
//! captures: `A` (per layer and stage, the per-token mean-1 first-moment statistics), `B` (per
//! layer and kind, the mean-1 second moment of the output gradient), `HO` (the output-side
//! Hessian `Σ_t w_t g_t g_tᵀ`, `w_t = ‖x_t‖²` of the *input*, `g_t` the sampled-label gradient of
//! the linear's *output*), `S4`/`Q4` and `ho_lw` (the Schäfer–Strimmer intensity of `HO_raw`),
//! `PB` and `ho_lb` (the block/sequence-unit intensity), `max_dlogit` (the teacher-forward check)
//! and the `K` per-position sampled softmax targets. Those require a backward pass this forward-only
//! capture does not run; none is stubbed.
//!
//! # Symmetry and cost
//!
//! `H` is symmetric, so [`Grams::accumulate`] writes only the upper triangle (`i ≤ j`) of each
//! accumulator and [`Grams::finish`] mirrors it once. Per token and matrix that is `w·(w+1)/2`
//! multiply-adds instead of `w²`. Accumulation is `f64` and deterministic: fixed loop order, no
//! threading, no `rand`.
//!
//! # Evidence grade
//!
//! `tested`: the gate's `cargo test` on the build path (`tests/learn_fold_capture.rs`). No Lean
//! refinement links this file; it is not on the extracted fragment.
//!
//! # Trust boundary
//!
//! The capture reads the Q32.32 `i64` tape only; the `f64` Gram is a CPU artifact and is not the
//! quantiser's own accumulation. `rmsnorm`, attention and the dense kernels are
//! [`crate::learn::grad::model`]'s contracts. Not covered: the output-gradient Grams above, and any
//! activation the tape does not carry.

#![allow(
    clippy::cast_precision_loss,
    reason = "the Q32.32 `i64` operands and the `u64` token counter are converted to `f64` at the \
              capture boundary; the conversion is exact for the value ranges Q32.32 activations \
              carry and the accumulator is deliberately f64"
)]

use crate::learn::grad::model::{Config, Tape};

/// `2^-32`: the Q32.32 → `f64` scale. Division by a power of two is exact in binary floating point.
const INV_ONE: f64 = 1.0 / 4_294_967_296.0;

/// A refusal from the Gram capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureError {
    /// The tape or configuration disagrees with the accumulator's geometry.
    Shape(String),
    /// A length product left `usize`, or the token counter left `u64`.
    Overflow,
}

/// What [`Grams::finish`] hands back for one matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Normalisation {
    /// The raw scatter `Σ_t x_t x_tᵀ`, unscaled.
    Sum,
    /// `(Σ_t x_t x_tᵀ) / tokens`, the reference `Gram.result(nt)` (`rstar_fast.py:247-254`).
    Mean,
    /// `(Σ_t x_t x_tᵀ) / mean(diag) + eps·I` — the mean-diagonal-1 normalisation `gq_capture` applies
    /// to its output-gradient `HO` (`fold_transfer.py:1322-1323`). The mean diagonal is clamped up to
    /// `1e-30` before the division, as the reference does.
    MeanDiagonal {
        /// Added to every diagonal entry after the division.
        eps: f64,
    },
}

/// One layer's four activation Grams, each row-major `w × w` (`[LayerGrams]`'s field docs name the
/// width), plus the token count summed into them.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerGrams {
    /// `Σ_t ln1_out_t ln1_out_tᵀ`, row-major `n_embd × n_embd`.
    pub qkv: Vec<f64>,
    /// `Σ_t attn_out_t attn_out_tᵀ`, row-major `q_out × q_out`.
    pub o: Vec<f64>,
    /// `Σ_t ln2_out_t ln2_out_tᵀ`, row-major `n_embd × n_embd`.
    pub gate_up: Vec<f64>,
    /// `Σ_t m_t m_tᵀ`, row-major `n_ff × n_ff`.
    pub down: Vec<f64>,
    /// Tokens accumulated into this layer's Grams.
    pub tokens: u64,
}

/// The per-layer activation-Gram accumulator for one calibration pass.
#[derive(Debug, Clone, PartialEq)]
pub struct Grams {
    /// Per trunk layer, in forward order.
    pub layers: Vec<LayerGrams>,
    /// Residual width, fixed by [`Grams::new`].
    n_embd: usize,
    /// `n_head · head_dim`, fixed by [`Grams::new`].
    q_out: usize,
    /// Feed-forward width, fixed by [`Grams::new`].
    n_ff: usize,
}

impl Grams {
    /// Allocate the zero accumulator for `cfg`'s geometry: one [`LayerGrams`] per trunk layer, the
    /// four matrices zeroed and the token counter at zero.
    ///
    /// # Errors
    ///
    /// [`CaptureError::Shape`] for a zero model dimension; [`CaptureError::Overflow`] when
    /// `n_head · head_dim` or a squared width leaves `usize`.
    ///
    /// # Complexity
    /// `O(1)` time; space `O(n_layer·(n_embd² + q_out² + n_ff²))` `f64` words, one allocation per
    /// matrix.
    pub fn new(cfg: &Config) -> Result<Grams, CaptureError> {
        let n_embd = cfg.n_embd;
        let n_ff = cfg.n_ff;
        if cfg.n_layer == 0 || n_embd == 0 || n_ff == 0 || cfg.n_head == 0 || cfg.head_dim == 0 {
            return Err(CaptureError::Shape("zero model dimension".to_string()));
        }
        let q_out = cfg.q_out().map_err(|_| CaptureError::Overflow)?;
        let e2 = square(n_embd)?;
        let q2 = square(q_out)?;
        let f2 = square(n_ff)?;
        let blank = || LayerGrams {
            qkv: vec![0.0; e2],
            o: vec![0.0; q2],
            gate_up: vec![0.0; e2],
            down: vec![0.0; f2],
            tokens: 0,
        };
        Ok(Grams {
            layers: (0..cfg.n_layer).map(|_| blank()).collect(),
            n_embd,
            q_out,
            n_ff,
        })
    }

    /// Fold `tape`'s four per-layer linear inputs into the accumulator, upper triangle only.
    ///
    /// # Errors
    ///
    /// [`CaptureError::Shape`] when `cfg` disagrees with the geometry [`Grams::new`] fixed, when
    /// `tape.layers.len() != cfg.n_layer`, or when a stage buffer's length is not `tape.t · w`;
    /// [`CaptureError::Overflow`] when `tape.t · w` or the token counter leaves its range.
    ///
    /// # Complexity
    /// Per token `n_embd·(n_embd+1)/2 + q_out·(q_out+1)/2 + n_ff·(n_ff+1)/2` `f64` multiply-adds;
    /// `O(t)` conversions plus `O(w_max)` scratch space, no per-token allocation.
    pub fn accumulate(&mut self, tape: &Tape, cfg: &Config) -> Result<(), CaptureError> {
        let q_out = cfg.q_out().map_err(|_| CaptureError::Overflow)?;
        if cfg.n_embd != self.n_embd
            || q_out != self.q_out
            || cfg.n_ff != self.n_ff
            || cfg.n_layer != self.layers.len()
        {
            return Err(CaptureError::Shape(format!(
                "config {}×{}×{} / {} layers != Grams {}×{}×{} / {} layers",
                cfg.n_embd,
                q_out,
                cfg.n_ff,
                cfg.n_layer,
                self.n_embd,
                self.q_out,
                self.n_ff,
                self.layers.len()
            )));
        }
        if tape.layers.len() != cfg.n_layer {
            return Err(CaptureError::Shape(format!(
                "tape has {} layers, config {}",
                tape.layers.len(),
                cfg.n_layer
            )));
        }
        let t = tape.t;
        for (acc, lt) in self.layers.iter_mut().zip(&tape.layers) {
            expect(t, self.n_embd, lt.ln1_out.len(), "ln1_out")?;
            expect(t, self.q_out, lt.attn_out.len(), "attn_out")?;
            expect(t, self.n_embd, lt.ln2_out.len(), "ln2_out")?;
            expect(t, self.n_ff, lt.m.len(), "m")?;
            fold_rows_upper(&mut acc.qkv, &lt.ln1_out, t, self.n_embd);
            fold_rows_upper(&mut acc.o, &lt.attn_out, t, self.q_out);
            fold_rows_upper(&mut acc.gate_up, &lt.ln2_out, t, self.n_embd);
            fold_rows_upper(&mut acc.down, &lt.m, t, self.n_ff);
            let rows = u64::try_from(t).map_err(|_| CaptureError::Overflow)?;
            acc.tokens = acc.tokens.checked_add(rows).ok_or(CaptureError::Overflow)?;
        }
        Ok(())
    }

    /// The mirrored, normalised copy of the accumulator: one [`LayerGrams`] per layer, every matrix
    /// full and symmetric.
    ///
    /// # Errors
    ///
    /// [`CaptureError::Shape`] for [`Normalisation::Mean`] or [`Normalisation::MeanDiagonal`] on a
    /// zero token count (a mean over no tokens is undefined, not zero).
    ///
    /// # Complexity
    /// `O(n_layer·(n_embd² + q_out² + n_ff²))` time and one `f64` copy of the accumulator; one
    /// `d·(d+1)/2` pass per matrix.
    pub fn finish(&self, normalisation: Normalisation) -> Result<Vec<LayerGrams>, CaptureError> {
        let mut out = Vec::with_capacity(self.layers.len());
        for lg in &self.layers {
            if lg.tokens == 0 && normalisation != Normalisation::Sum {
                return Err(CaptureError::Shape(
                    "normalisation over zero tokens".to_string(),
                ));
            }
            let normalise = |a: &[f64], d: usize| -> Result<Vec<f64>, CaptureError> {
                let (scale, diag_add) = match normalisation {
                    Normalisation::Sum => (1.0, 0.0),
                    Normalisation::Mean => (1.0 / lg.tokens as f64, 0.0),
                    Normalisation::MeanDiagonal { eps } => {
                        let mean = mean_diagonal(a, d);
                        let denom = if mean > 1e-30 { mean } else { 1e-30 };
                        (1.0 / denom, eps)
                    }
                };
                mirror(a, d, scale, diag_add)
            };
            out.push(LayerGrams {
                qkv: normalise(&lg.qkv, self.n_embd)?,
                o: normalise(&lg.o, self.q_out)?,
                gate_up: normalise(&lg.gate_up, self.n_embd)?,
                down: normalise(&lg.down, self.n_ff)?,
                tokens: lg.tokens,
            });
        }
        Ok(out)
    }
}

/// `x · x`, or [`CaptureError::Overflow`].
///
/// # Complexity
/// `O(1)`.
fn square(x: usize) -> Result<usize, CaptureError> {
    x.checked_mul(x).ok_or(CaptureError::Overflow)
}

/// `.len() == t · w` or [`CaptureError::Shape`].
///
/// # Complexity
/// `O(1)`.
fn expect(t: usize, w: usize, len: usize, what: &str) -> Result<usize, CaptureError> {
    match t.checked_mul(w) {
        Some(n) if n == len => Ok(n),
        Some(n) => Err(CaptureError::Shape(format!(
            "{what}: {len} != {t}·{w} = {n}"
        ))),
        None => Err(CaptureError::Overflow),
    }
}

/// Add `Σ_{r<t} x_r x_rᵀ` to the upper triangle (`i ≤ j`) of the row-major `w × w` accumulator
/// `acc`, `x` row-major `t × w` with Q32.32 entries.
///
/// # Complexity
/// `O(t·w²)` upper-triangle writes, of which `t·w·(w+1)/2` are multiply-adds; `O(w)` scratch
/// `f64` rows.
fn fold_rows_upper(acc: &mut [f64], x: &[i64], t: usize, w: usize) {
    let mut row_f = vec![0.0f64; w];
    for r in 0..t {
        let row = &x[r * w..r * w + w];
        for (d, v) in row_f.iter_mut().zip(row) {
            *d = *v as f64 * INV_ONE;
        }
        for i in 0..w {
            let xi = row_f[i];
            for j in i..w {
                acc[i * w + j] += xi * row_f[j];
            }
        }
    }
}

/// The mean of the diagonal of the row-major `d × d` accumulator `a`, read from its upper triangle.
///
/// # Complexity
/// `O(d)`.
fn mean_diagonal(a: &[f64], d: usize) -> f64 {
    let mut s = 0.0f64;
    for i in 0..d {
        if let Some(v) = a.get(i * d + i) {
            s += *v;
        }
    }
    s / d as f64
}

/// The mirrored, scaled copy of the upper-triangle accumulator `a` (row-major `d × d`): every entry
/// `(i, j)` is `a[min·d+max] · scale`, plus `diag_add` on the diagonal.
///
/// # Errors
///
/// [`CaptureError::Shape`] when `a.len() != d·d` or `d·d` leaves `usize`.
///
/// # Complexity
/// `O(d²)` time and space.
fn mirror(a: &[f64], d: usize, scale: f64, diag_add: f64) -> Result<Vec<f64>, CaptureError> {
    if a.len() != square(d)? {
        return Err(CaptureError::Shape(format!("gram {} != {d}²", a.len())));
    }
    let mut out = vec![0.0f64; a.len()];
    for i in 0..d {
        for j in i..d {
            let mut v = a[i * d + j] * scale;
            if i == j {
                v += diag_add;
            }
            out[i * d + j] = v;
            out[j * d + i] = v;
        }
    }
    Ok(out)
}
