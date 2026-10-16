//! Per-layer production scaling steps of the fold transfer.

/// The `-resq` production row-scale step for the MLP `up` plane: the elementwise-Gram ridge solve
/// for the per-row multiplier and its fold into the hierarchical row scale (`quant_r_hier`).
pub mod resq;
