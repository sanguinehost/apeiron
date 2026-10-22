//! Backward passes of the ported model (W19): dense adjoints (linear, `RMSNorm`, `SwiGLU`), causal
//! GQA attention with `RoPE`, and the KL/TV vocabulary heads. Q32.32 `i64`, checked arithmetic.
//!
//! [`model`] is the extraction unit's subject; [`load`] is the engine-reading IO glue that fills it
//! and stays outside `proof/extract/`.

pub mod attn;
pub mod dense;
pub mod head;
pub mod load;
pub mod model;
