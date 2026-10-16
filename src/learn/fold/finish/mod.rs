//! The finishing stage: the frozen-code weight plane carrying trainable continuous latents.

/// The coded plane: a frozen `E8 ∪ O8` code field, a frozen per-row scale grid and the trainable
/// log-multiplicative latents `r`, `c`, `d`.
pub mod plane;
