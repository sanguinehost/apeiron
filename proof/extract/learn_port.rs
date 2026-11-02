//! Extraction unit for the one-bit backward passes (`src/learn/grad/`). Each module is reached by
//! `#[path]` (relative to this file's directory), so the extracted bodies are the real source; each
//! `word_<module>_<fn>` wrapper fixes the monomorphic root and calls the real function, restating no
//! formula.

#[path = "../../src/learn/grad/dense.rs"]
pub mod dense;

/// The real linear forward body, rooted at the concrete Q32.32 `i64` word.
pub fn word_dense_linear_fwd(
    x: &[i64],
    w: &[i64],
    n: usize,
    i: usize,
    o: usize,
) -> Result<Vec<i64>, dense::DenseError> {
    dense::linear_fwd(x, w, n, i, o)
}

/// The real linear input-adjoint body, rooted at the concrete Q32.32 `i64` word.
pub fn word_dense_linear_bwd_input(
    dy: &[i64],
    w: &[i64],
    n: usize,
    i: usize,
    o: usize,
) -> Result<Vec<i64>, dense::DenseError> {
    dense::linear_bwd_input(dy, w, n, i, o)
}

/// The real linear weight-adjoint body, rooted at the concrete Q32.32 `i64` word.
pub fn word_dense_linear_bwd_weight(
    dy: &[i64],
    x: &[i64],
    n: usize,
    i: usize,
    o: usize,
) -> Result<Vec<i64>, dense::DenseError> {
    dense::linear_bwd_weight(dy, x, n, i, o)
}

/// The real `RMSNorm` forward body, rooted at the concrete Q32.32 `i64` word.
pub fn word_dense_rmsnorm_fwd(
    x: &[i64],
    gamma: &[i64],
    eps: i64,
) -> Result<Vec<i64>, dense::DenseError> {
    dense::rmsnorm_fwd(x, gamma, eps)
}

/// The real `RMSNorm` backward body, rooted at the concrete Q32.32 `i64` word.
pub fn word_dense_rmsnorm_bwd(
    x: &[i64],
    gamma: &[i64],
    eps: i64,
    dy: &[i64],
) -> Result<(Vec<i64>, Vec<i64>), dense::DenseError> {
    dense::rmsnorm_bwd(x, gamma, eps, dy)
}

/// The real `SwiGLU` forward body, rooted at the concrete Q32.32 `i64` word.
pub fn word_dense_swiglu_fwd(gate: &[i64], up: &[i64]) -> Result<Vec<i64>, dense::DenseError> {
    dense::swiglu_fwd(gate, up)
}

/// The real `SwiGLU` backward body, rooted at the concrete Q32.32 `i64` word.
pub fn word_dense_swiglu_bwd(
    gate: &[i64],
    up: &[i64],
    dy: &[i64],
) -> Result<(Vec<i64>, Vec<i64>), dense::DenseError> {
    dense::swiglu_bwd(gate, up, dy)
}

/// The causal GQA attention and RoPE backward passes (`src/learn/grad/attn.rs`).
#[path = "../../src/learn/grad/attn.rs"]
pub mod attn;

/// Real `learn::grad::attn::rope_fwd`.
pub fn word_attn_rope_fwd(
    x: &[i64],
    pos: u32,
    table: &crate::engine::RopeTable,
) -> Result<Vec<i64>, attn::AttnError> {
    attn::rope_fwd(x, pos, table)
}

/// Real `learn::grad::attn::rope_bwd`.
pub fn word_attn_rope_bwd(
    dy: &[i64],
    pos: u32,
    table: &crate::engine::RopeTable,
) -> Result<Vec<i64>, attn::AttnError> {
    attn::rope_bwd(dy, pos, table)
}

/// Real `learn::grad::attn::attn_fwd`.
pub fn word_attn_attn_fwd(
    q: &[i64],
    k: &[i64],
    v: &[i64],
    nh: usize,
    nkv: usize,
    hd: usize,
) -> Result<(Vec<i64>, Vec<i64>), attn::AttnError> {
    attn::attn_fwd(q, k, v, nh, nkv, hd)
}

/// Real `learn::grad::attn::attn_bwd`.
pub fn word_attn_attn_bwd(
    q: &[i64],
    k: &[i64],
    v: &[i64],
    p: &[i64],
    dout: &[i64],
    nh: usize,
    nkv: usize,
    hd: usize,
) -> Result<attn::AttnGrads, attn::AttnError> {
    attn::attn_bwd(q, k, v, p, dout, nh, nkv, hd)
}

#[path = "../../src/learn/grad/head.rs"]
pub mod head;

/// Real `learn::grad::head::log_softmax_row`.
pub fn word_head_log_softmax_row(z: &[i64]) -> Result<Vec<i64>, head::HeadError> {
    head::log_softmax_row(z)
}

/// Real `learn::grad::head::kl_row`.
pub fn word_head_kl_row(z_s: &[i64], z_t: &[i64]) -> Result<(i64, Vec<i64>), head::HeadError> {
    head::kl_row(z_s, z_t)
}

/// Real `learn::grad::head::tv_row`.
pub fn word_head_tv_row(z_s: &[i64], z_t: &[i64]) -> Result<(i64, Vec<i64>), head::HeadError> {
    head::tv_row(z_s, z_t)
}

/// Real `learn::grad::head::head_chunked`.
pub fn word_head_head_chunked(
    hidden: &[i64],
    w_head: &[i64],
    z_t: &[i64],
    dims: head::Dims,
    chunk: usize,
    objective: head::Objective,
) -> Result<(i64, Vec<i64>, Vec<i64>), head::HeadError> {
    head::head_chunked(hidden, w_head, z_t, dims, chunk, objective)
}
