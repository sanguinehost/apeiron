//! One block of the executor over externally supplied residual columns: the student-stream tap a
//! block-wise fit needs, where the layer input is a quantised model's own trunk state rather than
//! the teacher's.

use super::super::{affine_slots, prefill_block, softmax_slots, Sched, StepError};
use super::tap::{self, Tap};
use crate::engine::mixer::AffineState;
use crate::engine::{DecodeState, Engine, KernelMode};

impl Engine {
    /// Block `layer` alone over the residual columns `hidden` (positions `0..hidden.len()` of a
    /// fresh `st`), in the blocks of [`prefill_block`] and on the resident `Q8Act` schedule of
    /// [`Engine::prefill_tap_cols`]. Returns the stage rows `[stage][column][width]` of [`Tap`].
    /// Fed the teacher's recorded layer input, the rows equal `prefill_tap_cols`'s for that layer
    /// bit for bit.
    ///
    /// # Errors
    ///
    /// As [`Engine::prefill_tap_cols`]; [`StepError::BadConfig`] for no columns or a layer past
    /// the model.
    ///
    /// # Complexity
    ///
    /// One layer's cost of [`Engine::prefill_tap_cols`] per column, plus one `O(width)` copy per
    /// stage per column.
    pub fn layer_tap_cols(
        &self,
        st: &mut DecodeState,
        layer: usize,
        hidden: &[Vec<i64>],
    ) -> Result<Vec<Vec<Vec<i64>>>, StepError> {
        let cfg = &self.config;
        if hidden.is_empty() || layer >= cfg.n_layer {
            return Err(StepError::BadConfig(String::from(
                "layer tap: no columns or no layer",
            )));
        }
        let slots = softmax_slots(&cfg.mixers);
        let aslots = affine_slots(&cfg.mixers);
        let layers = self.resolve_layers();
        let eps = self.eps_fixed()?;
        let mut scratch = crate::engine::chart::ffn::FfnScratch::default();
        let mut none_state = AffineState::empty();
        let mut rows: Vec<Vec<Vec<i64>>> = vec![Vec::new(); tap::STAGE_COUNT];
        let width = prefill_block();
        let num_blocks = hidden.len().div_ceil(width);
        let base = hidden.len() / num_blocks;
        let rem = hidden.len() % num_blocks;
        let mut start = 0usize;
        for bi in 0..num_blocks {
            let size = if bi < rem { base + 1 } else { base };
            let mut cols = hidden[start..start + size].to_vec();
            start += size;
            let mut t = Some(Tap::new(cfg.n_layer));
            let affine = match st.affine.as_mut() {
                Some(a) => a,
                None => &mut none_state,
            };
            self.layer_cols(
                &mut st.kv,
                affine,
                &mut [],
                &layers,
                layer,
                slots[layer],
                aslots[layer],
                &mut cols,
                &mut scratch,
                eps,
                Sched::Resident,
                KernelMode::Q8Act,
                true,
                &mut t,
            )?;
            if let Some(t) = t {
                for (dst, src) in rows
                    .iter_mut()
                    .zip(t.rows.into_iter().nth(layer).unwrap_or_default())
                {
                    dst.extend(src);
                }
            }
        }
        Ok(rows)
    }
}
