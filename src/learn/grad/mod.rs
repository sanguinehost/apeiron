//! Backward passes of the ported model (W19): dense adjoints (linear, `RMSNorm`, `SwiGLU`), causal
//! GQA attention with `RoPE`, and the KL/TV vocabulary heads. Q32.32 `i64`, checked arithmetic.

pub mod attn;
pub mod dense;
pub mod head;
