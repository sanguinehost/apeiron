//! Forward products of the dense layer, as f32 SPIR-V twins of `learn::grad::dense`.
//!
//! `lin` carries the forward map `y = x·Wᵀ`; the backward adjoints that invert it live in the
//! sibling `bwd` module.

pub mod lin;
pub use lin::*;
