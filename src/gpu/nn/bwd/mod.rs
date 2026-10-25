//! Backward products of the dense layer, as f32 SPIR-V twins of `learn::grad::dense`.
//!
//! `lin` carries the two Jacobian-transpose GEMMs (`dX` and `dW`); the forward maps and the norm
//! backward live in the sibling modules whose kernels they invert. All three dense GEMMs — forward,
//! `dX` and `dW` — run on the engine GEMM's bf16 face (`src/gpu/spirv/coopmat_gemm.comp`,
//! `-DOPF=1`); `lin` is the operand description of each.

pub mod opt;
pub use opt::*;
