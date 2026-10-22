//! Extraction unit for the GGUF writer (`src/learn/gguf.rs`) and the serial cores of the Q32.32
//! solve primitives (`src/learn/solve/{gemm,inv,q32,syrk}.rs`) and the exact Gram fold
//! (`src/learn/gram.rs`). Each module is the real source by `#[path]` or the `solve_port` copy of
//! it; each `word_*` wrapper fixes the concrete root and calls the real body, restating no formula.
//!
//! The thread drivers stay opaque in `proof/regen.sh` (`block_rows_pool`, `invert_columns_pool`,
//! `trailing_update_pool`). Their per-worker bodies stay private to `learn::solve` and are not
//! rooted: staging those files in the main crate meets the kernel-route refusal until the products
//! move onto `engine::kern::limb`, which is the KERN epic, not this extraction unit.

#[path = "../../src/learn/gguf.rs"]
pub mod gguf;

/// The real `learn::grad::{dense,attn}` bodies, re-exported so the real `grad/model.rs`
/// `super::{dense,attn}` imports resolve to the modules `learn_port` already includes.
pub use crate::learn_port::{attn, dense};

/// The whole-model backward module. Only its integer bodies below are rooted: `backward`
/// constructs a `RopeTable` from a host `f64` (`F64::from_f64`), which the pinned Aeneas does not
/// translate, so that body stays outside this extraction unit.
#[path = "../../src/learn/grad/model.rs"]
pub mod model;

/// The `solve` modules `solve_port` already includes by `#[path]`; re-exported here so the serial
/// cores below name the same real bodies rather than a second copy of the module.
pub use crate::solve_port::{chol, gemm, inv, q32, syrk};

/// Real `learn::gguf::write_header`: magic, version, counts, metadata, tensor directory and the
/// alignment padding, all checked.
pub fn word_learn_gguf_write_header(
    header: &crate::format::gguf::GgufHeader,
) -> Result<Vec<u8>, crate::format::gguf::GgufError> {
    gguf::write_header(header)
}

/// Real `learn::gguf::write_file`: the header, the tensor-extent check and the data section.
pub fn word_learn_gguf_write_file(
    header: &crate::format::gguf::GgufHeader,
    data: &[u8],
) -> Result<Vec<u8>, crate::format::gguf::GgufError> {
    gguf::write_file(header, data)
}

/// Real `learn::gram::gram_accumulator_clear`: `|g[idx]| ≤ i64::MAX − rows·16_384` for every entry.
pub fn word_learn_gram_accumulator_clear(g: &[i64], rows: usize) -> bool {
    crate::gram::gram_accumulator_clear(g, rows)
}

/// Real `learn::gram::gram_accumulate`: `g[i*d+j] += cols[j][i]`, checked `i64`.
pub fn word_learn_gram_accumulate(
    g: Vec<i64>,
    cols: &[Vec<i64>],
    d: usize,
) -> Result<Vec<i64>, crate::gram::GramError> {
    crate::gram::gram_accumulate(g, cols, d)
}

/// Real `learn::solve::q32::div_rne_pos`: `num / den` rounded to nearest, ties to even, positive
/// divisor.
pub fn word_solve_q32_div_rne_pos(num: i128, den: i128) -> i128 {
    q32::div_rne_pos(num, den)
}

/// Real `learn::solve::q32::SpdQ::to_q32`: `h[i][j] = hat[i][j] · 2^(d[i]+d[j])`, checked.
pub fn word_solve_q32_spd_to_q32(h: &q32::SpdQ) -> Result<Vec<i64>, q32::Q32Error> {
    h.to_q32()
}

/// Real `learn::grad::model::Config::q_out`: `n_head · head_dim`, checked.
pub fn word_model_config_q_out(cfg: &model::Config) -> Result<usize, model::ModelError> {
    cfg.q_out()
}

/// Real `learn::grad::model::Config::kv_out`: `n_head_kv · head_dim`, checked.
pub fn word_model_config_kv_out(cfg: &model::Config) -> Result<usize, model::ModelError> {
    cfg.kv_out()
}

/// Real `learn::grad::model::Config::checked`: the dimension and GQA-divisibility sanity.
pub fn word_model_config_checked(cfg: &model::Config) -> Result<(), model::ModelError> {
    cfg.checked()
}

/// Real `learn::grad::model::expect`: `.len() == t · w`, checked.
pub fn word_model_expect(
    t: usize,
    w: usize,
    len: usize,
    what: &str,
) -> Result<usize, model::ModelError> {
    model::expect(t, w, len, what)
}

/// Real `learn::grad::model::add_flat`: checked `a[i] += b[i]`.
pub fn word_model_add_flat(a: &mut [i64], b: &[i64]) -> Result<(), model::ModelError> {
    model::add_flat(a, b)
}

/// Real `learn::grad::model::col_sum`: the checked column sums of a `t × w` buffer.
pub fn word_model_col_sum(x: &[i64], t: usize, w: usize) -> Result<Vec<i64>, model::ModelError> {
    model::col_sum(x, t, w)
}

/// Real `learn::grad::model::rmsnorm_rows`: `RMSNorm` over every row, one `dense::rmsnorm_fwd` each.
pub fn word_model_rmsnorm_rows(
    x: &[i64],
    gamma: &[i64],
    t: usize,
    w: usize,
    eps: i64,
) -> Result<Vec<i64>, model::ModelError> {
    model::rmsnorm_rows(x, gamma, t, w, eps)
}

/// Real `learn::grad::model::rmsnorm_bwd_rows`: the row-wise `RMSNorm` backward and gain sum.
pub fn word_model_rmsnorm_bwd_rows(
    x: &[i64],
    gamma: &[i64],
    eps: i64,
    dy: &[i64],
    t: usize,
    w: usize,
) -> Result<(Vec<i64>, Vec<i64>), model::ModelError> {
    model::rmsnorm_bwd_rows(x, gamma, eps, dy, t, w)
}

/// Real `learn::grad::model::rope_rows_fwd`: `RoPE` on every row at its own position.
pub fn word_model_rope_rows_fwd(
    x: &[i64],
    t: usize,
    row: usize,
    rope: &crate::engine::RopeTable,
) -> Result<Vec<i64>, model::ModelError> {
    model::rope_rows_fwd(x, t, row, rope)
}

/// Real `learn::grad::model::rope_rows_bwd`: the row-wise `RoPE` adjoint.
pub fn word_model_rope_rows_bwd(
    dy: &[i64],
    t: usize,
    row: usize,
    rope: &crate::engine::RopeTable,
) -> Result<Vec<i64>, model::ModelError> {
    model::rope_rows_bwd(dy, t, row, rope)
}

/// Real `learn::grad::model::add_bias`: checked bias add over every row.
pub fn word_model_add_bias(
    x: &mut [i64],
    bias: Option<&[i64]>,
    t: usize,
    w: usize,
) -> Result<(), model::ModelError> {
    model::add_bias(x, bias, t, w)
}

/// Real `learn::grad::model::embedding_grad`: the scatter-add `dembedding[token[r],:] += dh[r,:]`.
pub fn word_model_embedding_grad(
    dh: &[i64],
    tokens: &[u32],
    t: usize,
    w: usize,
    vocab: usize,
) -> Result<Vec<i64>, model::ModelError> {
    model::embedding_grad(dh, tokens, t, w, vocab)
}

/// Real `learn::onebit_bits::Cfg::qwen25_05b_nb`: the Qwen2.5-0.5B `-nb` arm's fixed dimensions.
pub fn word_onebit_qwen25_05b_nb() -> crate::onebit_bits::Cfg {
    crate::onebit_bits::Cfg::qwen25_05b_nb()
}
