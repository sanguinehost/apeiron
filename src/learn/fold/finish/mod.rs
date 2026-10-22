//! The finishing stage: the frozen-code weight plane carrying trainable continuous latents.

/// The whole-model Q32.32 weights assembled from the coded planes, the tunables and the frozen
/// embedding and head.
pub mod assemble;

/// The coded plane: a frozen `E8 ∪ O8` code field, a frozen per-row scale grid and the trainable
/// log-multiplicative latents `r`, `c`, `d`.
pub mod plane;

/// The finishing P-step: Adam on the plane latents and the zero-bit norm/bias tunables, with the
/// Q32.32 KL teacher objective.
pub mod train;
