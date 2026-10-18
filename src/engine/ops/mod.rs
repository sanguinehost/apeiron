//! The per-token model operators, one module per port-map row: `RMSNorm`, `RoPE`, the dense and
//! routed FFN, the projection kernels (plain, fused, row-parallel), the Walsh–Hadamard rotation,
//! the LM head and the sampler.

pub(super) mod ffn;
pub(super) mod fused;
pub(super) mod hadamard;
pub(super) mod lm_head;
pub(super) mod matvec;
pub(super) mod moe;
pub(super) mod norm;
pub(super) mod par;
pub(super) mod rope;
pub(super) mod sampler;
