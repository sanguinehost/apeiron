//! Per-layer steps of the fold transfer: stage statistics and solve, and the production scaling
//! steps.

/// The `-resq` production row-scale step for the MLP `up` plane: the elementwise-Gram ridge solve
/// for the per-row multiplier and its fold into the hierarchical row scale (`quant_r_hier`).
pub mod resq;

/// Sink-gated token weights, the stage Gram/`Cts`/`Cs` accumulator and the split intra/inter ridge
/// solve of the production `fold_gqa` arm (`student-input path`, `sink_gate`, `split_intra_inter`).
pub mod stage;
