//! Weight-fit steps of the fold transfer.

/// The rounded rank-`rank` corrector (fp16, u8 and exact carriers) in the `Hl` metric.
pub mod corrector;
/// Symmetric eigendecomposition (cyclic Jacobi) and the exact rank-`rank` corrector in the `Hl` metric.
pub mod eigh;
/// Damped Newton-Fisher descent of the final RMSNorm gain against the readout KL.
pub mod fnorm;
/// Block-Cholesky GPTQ over the E8 codebook (one code index per eight columns, optional per-row scale and act-order).
pub mod gptq_e8;
/// Per-block nearest-codeword ICM passes over the E8 codes of one and two planes, and the two-plane `r` refit.
pub mod icm;
/// Reduced joint block-ICM post-pass on a finished two-plane pair (`nb_refine`).
pub mod nb_refine;
/// Bit-exact binary16 rounding and the hierarchical row/column/diagonal scale quantisers.
pub mod quant;
/// `fwht128`, `compose`, the refit objective and the closed-form `c`, `r`, `d` steps.
pub mod steps;
