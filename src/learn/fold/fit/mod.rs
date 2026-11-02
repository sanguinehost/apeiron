//! Symmetric eigendecomposition, the low-rank corrector, and the scale quantisers.

/// The rounded rank-`rank` corrector (fp16, u8 and exact carriers) in the `Hl` metric.
pub mod corrector;
/// Symmetric eigendecomposition (cyclic Jacobi) and the exact rank-`rank` corrector in the `Hl` metric.
pub mod eigh;
/// Bit-exact binary16 rounding and the hierarchical row/column/diagonal scale quantisers.
pub mod quant;
