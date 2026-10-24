//! Backward products of the dense layer, as f32 SPIR-V twins of `learn::grad::dense`.
//!
//! `lin` carries the two Jacobian-transpose GEMMs (`dX` and `dW`); the forward maps and the norm
//! backward live in the sibling modules whose kernels they invert. `coop` carries the three dense
//! GEMMs again on the tensor cores (`bf16` operands, f32 accumulator), the same signatures.

mod coop;
pub mod lin;
pub mod opt;
pub use coop::*;
pub use lin::*;
pub use opt::*;
