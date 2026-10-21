//! The opt-in per-layer activation tap. Board .
//!
//! [`Tap`] holds, per trunk layer and per column, the five stage inputs the activation-Gram
//! capture reads ([`crate::learn::fold::stats::capture`]): the residual stream entering the layer,
//! the attention-side `RMSNorm` output (the Q/K/V projection input), the output projection's input
//! (the attention output, or the affine mixer's value output), the FFN-side `RMSNorm` output (the
//! gate/up input), and the `SwiGLU` product (the down projection's input).
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
/// The number of stage slots in [`Tap::rows`].
pub(crate) const STAGE_COUNT: usize = 5;

/// One pass's captured stage inputs, `[layer][stage][column][width]` Q32.32 rows.
///
/// Every stage starts empty; a stage a schedule does not route through a tap point stays empty. The
/// single-column FFN fuses its `SwiGLU` product inside `ffn_q8act_at`, so [`STAGE_M`] is empty
/// there; the pooled multi-column path fills every stage.
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
}
