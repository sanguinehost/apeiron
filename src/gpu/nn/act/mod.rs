//! The SwiGLU activation face as f32 SPIR-V twins of `learn::grad::dense`.
//!
//! `swiglu` carries the forward map `out = silu(gate)·up` and its adjoint pair
//! `(dgate, dup)`; the dense forward and backward products it sits between live in `fwd` and
//! `bwd`.

pub mod swiglu;
pub use swiglu::*;
