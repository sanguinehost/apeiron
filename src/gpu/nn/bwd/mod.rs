//! Backward products of the dense layer, as f32 SPIR-V twins of `learn::grad::dense`.
//!
//! `lin` carries the two Jacobian-transpose GEMMs (`dX` and `dW`); the forward maps and the norm
//! backward live in the sibling modules whose kernels they invert.

pub mod lin;
pub use lin::*;
