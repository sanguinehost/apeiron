//! The attention kernels: the flat and tiled prefill passes and the glue that joins them to the
//! decode path, the sparse-attention (knot) kernels, the DP4A score builders and the KV delta fill.
//! `use super::*;` gives every file below the parent's name set, private items included.

use super::*;

mod attn_dp4a;
mod bwd;
mod pool;
mod prefill;
mod prefill_glue;
mod sparse;

pub use bwd::*;
pub use pool::*;
pub use prefill::*;
pub use prefill_glue::*;
pub use sparse::*;
