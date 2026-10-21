//! The opt-in per-layer activation tap. Board .
//!
//! [`Tap`] holds, per trunk layer and per column, the twelve stage inputs the activation-Gram
//! capture reads ([`crate::learn::fold::stats::capture`]) and the backward pass needs
//! ([`crate::learn::grad::model::backward_with_taps`]): the residual stream entering the layer, the
//! attention-side `RMSNorm` output (the Q/K/V projection input), the output projection's input
//! (the attention output, or the affine mixer's value output), the FFN-side `RMSNorm` output (the
//! gate/up input), the `SwiGLU` product (the down projection's input), the attention's own
//! post-`RoPE` query, key and value rows, the gate and up pre-activations the `SwiGLU` epilogue
//! consumes, the residual after the attention add (the `ffn_norm` input, the reference
//! `LayerTape.h2`), and the layer's output residual (the next layer's input; the last layer's row
//! is the `output_norm` input, the reference `Tape.final_hidden`).
//!
//! The three attention rows are the words the attention reads, so a backward pass recomputes the
//! attention probabilities from them instead of a recorded `p` (the reference saves `q, k, v,
//! output, logsumexp` and recomputes `attn <- exp(attn - normalizer)`; `PyTorch` `93537cd6472`
//! `aten/src/ATen/native/cpu/FlashAttentionKernel.cpp:1000-1005`).
//!
//! It is allocated only when a caller asks for it: the layer loop carries `Option<&mut Tap>` and
//! the off path allocates nothing and records nothing.

/// Stage slot of the residual stream entering a layer, in [`Tap::rows`].
pub(crate) const STAGE_HIDDEN_IN: usize = 0;
/// Stage slot of the attention-side `RMSNorm` output (the Q/K/V projection input).
pub(crate) const STAGE_LN1_OUT: usize = 1;
/// Stage slot of the output projection's input (the attention output).
pub(crate) const STAGE_ATTN_OUT: usize = 2;
/// Stage slot of the FFN-side `RMSNorm` output (the gate/up input).
pub(crate) const STAGE_LN2_OUT: usize = 3;
/// Stage slot of the `SwiGLU` product (the down projection input).
pub(crate) const STAGE_M: usize = 4;
/// Stage slot of the column's post-`RoPE` query, `n_head·head_dim` words wide (the attention's
/// score input).
pub(crate) const STAGE_Q: usize = 5;
/// Stage slot of the column's post-`RoPE` key, `n_head_kv·head_dim` words wide (the attention's
/// score input, and the row appended to [`crate::engine::KvState`]).
pub(crate) const STAGE_K: usize = 6;
/// Stage slot of the column's value, `n_head_kv·head_dim` words wide (the attention's value input,
/// and the row appended to [`crate::engine::KvState`]).
pub(crate) const STAGE_V: usize = 7;
/// Stage slot of the column's gate pre-activation, `n_ff` words wide: the `SiLU` argument the
/// `SwiGLU` epilogue consumes.
pub(crate) const STAGE_GATE: usize = 8;
/// Stage slot of the column's up pre-activation, `n_ff` words wide: the `SwiGLU` epilogue's
/// multiplier.
pub(crate) const STAGE_UP: usize = 9;
/// Stage slot of the residual stream after the attention add and before the FFN-side `RMSNorm`,
/// `n_embd` words wide: the `ffn_norm` input the backward pass needs (the reference
/// `LayerTape.h2`, `rmsnorm_bwd_rows(&lt.h2, ..)`).
pub(crate) const STAGE_H2: usize = 10;
/// Stage slot of the layer's output residual stream, `n_embd` words wide: the FFN add's result,
/// which is the next layer's input; the last layer's row is the `output_norm` input (the reference
/// `Tape.final_hidden`).
pub(crate) const STAGE_OUT: usize = 11;
/// The number of stage slots in [`Tap::rows`].
pub(crate) const STAGE_COUNT: usize = 12;

/// One pass's captured stage inputs, `[layer][stage][column][width]` Q32.32 rows.
///
/// Every stage starts empty; a stage a schedule does not route through a tap point stays empty. The
/// single-column FFN fuses its `SwiGLU` product inside `ffn_q8act_at`, so [`STAGE_M`] and the
/// [`STAGE_GATE`]/[`STAGE_UP`] pre-activations are empty there; the pooled multi-column path fills
/// every stage. The pre-activations let a backward pass recompute `silu` from the saved words
/// (`unsloth/kernels/fast_lora.py:110` saves `X, e, g`).
pub(crate) struct Tap {
    /// `[layer][stage][column][width]`, the stage slots named by [`STAGE_HIDDEN_IN`] and below.
    pub(crate) rows: Vec<Vec<Vec<Vec<i64>>>>,
}

impl Tap {
    /// A tap over `n_layer` layers with every stage empty.
    ///
    /// # Complexity
    ///
    /// `O(n_layer·STAGE_COUNT)` empty-vector allocations; nothing per column, because the column
    /// rows are filled by [`Tap::record`] from the stage the pass already computed.
    pub(crate) fn new(n_layer: usize) -> Tap {
        let mut rows: Vec<Vec<Vec<Vec<i64>>>> = Vec::with_capacity(n_layer);
        let mut l = 0usize;
        while l < n_layer {
            let mut stages: Vec<Vec<Vec<i64>>> = Vec::with_capacity(STAGE_COUNT);
            let mut s = 0usize;
            while s < STAGE_COUNT {
                stages.push(Vec::new());
                s += 1;
            }
            rows.push(stages);
            l += 1;
        }
        Tap { rows }
    }

    /// Copy `cols` into stage `stage` of `layer`.
    ///
    /// # Complexity
    ///
    /// `O(cols.len()·width)` words copied; `O(1)` besides the replaced rows.
    pub(crate) fn record(&mut self, layer: usize, stage: usize, cols: &[Vec<i64>]) {
        self.rows[layer][stage] = cols.to_vec();
    }

    /// Copy the column-major flat stage `flat` (`width` words per column) into stage `stage` of
    /// `layer`.
    ///
    /// # Complexity
    ///
    /// `O(flat.len())` words copied; `O(flat.len()/width)` row allocations. `width` must be
    /// non-zero (`n_ff` on every call), which the caller's config guarantees.
    pub(crate) fn record_flat(&mut self, layer: usize, stage: usize, flat: &[i64], width: usize) {
        let mut rows: Vec<Vec<i64>> = Vec::with_capacity(flat.len() / width.max(1));
        for col in flat.chunks(width.max(1)) {
            rows.push(col.to_vec());
        }
        self.rows[layer][stage] = rows;
    }

    /// Append one column `row` to stage `stage` of `layer`; the per-column stages ([`STAGE_Q`],
    /// [`STAGE_K`], [`STAGE_V`]) are filled as the pass computes each column.
    ///
    /// # Complexity
    ///
    /// `O(row.len())` words copied into the grown stage; `O(1)` besides, amortised.
    pub(crate) fn push(&mut self, layer: usize, stage: usize, row: &[i64]) {
        self.rows[layer][stage].push(row.to_vec());
    }
}
