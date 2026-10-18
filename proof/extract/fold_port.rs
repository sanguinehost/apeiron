//! Extraction unit for the fold statistics (`src/learn/fold/stats/`) and the closed-form weight-fit
//! steps (`src/learn/fold/fit/`). Each module is reached through its real `mod.rs` by `#[path]`, so
//! the extracted bodies are the real sources; each `word_*` wrapper fixes the monomorphic root and
//! calls the real function, restating no formula.
//!
//! # Merge (`proof/extract/lib.rs`)
//!
//! The unit is declared at the crate root, as `learn_port`/`solve_port` are:
//!
//! ```text
//! #[path = "fold_port.rs"]
//! pub mod fold_port;
//! ```
//!
//! and the `learn` mirror carries the paths these sources name absolutely (a `#[path]` inside an
//! inline module cannot traverse `..`, so every module is declared at the crate root and only
//! re-exported here):
//!
//! ```text
//! pub mod learn {
//!     pub mod solve { pub use crate::solve_port::{chol, eig, gemm, inv, q32, syrk}; }
//!     pub mod fold {
//!         pub use crate::fold_port::{fit, stats};
//!     }
//!     pub mod grad {
//!         pub use crate::grad_solve_port::model;
//!         pub use crate::learn_port::{attn, dense, head};
//!     }
//!     pub mod code { pub use crate::learn_port::e8; }
//!     pub use crate::grad_solve_port::gguf;
//! }
//! ```
//!
//! `src/learn/fold/fit/steps.rs` also reads `crate::engine::{fwht_fixed, HadamardError}`, the real
//! `src/engine/mod.rs` re-exports; the engine mirror adds those two names.
//!
//! # Boundary
//!
//! The layer/run/finish surface (`fold/layer/`, `fold/run/`, `fold/finish/`) is not declared here:
//! those bodies use `Iterator` adapters and function items the pinned Aeneas does not translate
//! (the signature translation aborts on `Iterator`), so they are a rewrite lane, not a port.
//!
//! Five fit entry points stay unrooted for the same class of reason, one body each: `fnorm_fit`,
//! `fold_fit`, `gptq_e8`, `feedback_u` and `nb_refine` carry a `?`/early return inside a loop or a
//! `break` out of an outer loop. Their callees that are clean are still reached through the roots
//! below.

/// The real `src/learn/fold/fit/`: the corrector and its eigendecomposition, the closed-form
/// `c`/`r`/`d` steps, the hierarchical quantisers and the `-nb` post-pass.
#[path = "../../src/learn/fold/fit/mod.rs"]
pub mod fit;

/// The real `src/learn/fold/stats/`: the activation Grams, the shrinkage intensities and the
/// `f64` forward the transfer consumes.
#[path = "../../src/learn/fold/stats/mod.rs"]
pub mod stats;

/// Mirror of `crate::learn` for the extraction unit: the real `learn` sources name
/// `crate::learn::…`, which resolves here to the `solve_port`/`fold_port`/`learn_port` copies.
pub mod learn {
    pub mod solve {
        pub use crate::solve_port::{chol, eig, gemm, inv, q32, syrk};
    }
    pub mod fold {
        pub use crate::fold_port::{fit, stats};
    }
    pub mod grad {
        pub use crate::grad_solve_port::model;
        pub use crate::learn_port::{attn, dense, head};
    }
    pub mod code {
        pub use crate::learn_port::e8;
    }
    pub use crate::grad_solve_port::gguf;
}

// ── `learn::fold::stats` ─────────────────────────────────────────────────────────────────────

/// The real `ceil(bits / 8)`, rooted at the concrete `u128`.
pub fn word_stats_budget_bytes_ceil(bits: u128) -> Option<u128> {
    stats::budget::bytes_ceil(bits)
}

/// The real largest-fitting uniform body rate, rooted at the concrete `u64` rationals.
pub fn word_stats_budget_max_body_bpw(
    total_weights: u64,
    embd_weights: u64,
    head_weights: u64,
    embd_bpw: (u64, u64),
    head_bpw: (u64, u64),
    budget_bytes: u64,
    reserve_bytes: u64,
) -> Option<(u64, u64)> {
    stats::budget::max_body_bpw(
        total_weights,
        embd_weights,
        head_weights,
        embd_bpw,
        head_bpw,
        budget_bytes,
        reserve_bytes,
    )
}

/// The real Schäfer–Strimmer intensity, rooted at the concrete Q32.32 words.
pub fn word_stats_lw_intensity(
    s4: i64,
    q: &[i64],
    h: &[i64],
    o: usize,
    n: usize,
) -> Result<(i64, i64), stats::lw::LwError> {
    stats::lw::lw_intensity(s4, q, h, o, n)
}

/// The real batch-means intensity, rooted at the concrete Q32.32 words.
pub fn word_stats_lw_block_intensity(
    p: i64,
    b: usize,
    h: &[i64],
    o: usize,
    n: usize,
) -> Result<(i64, i64), stats::lw::LwError> {
    stats::lw::lw_block_intensity(p, b, h, o, n)
}

/// The real `a·b` Q32.32 product, rooted at the concrete words.
pub fn word_stats_parts_mul(a: i64, b: i64) -> Option<i64> {
    stats::parts::mul(a, b)
}

/// The real floored Q32.32 square root, rooted at the concrete word.
pub fn word_stats_parts_sqrt(x: i64) -> Option<i64> {
    stats::parts::sqrt(x)
}

/// The real `(sin t, cos t)` turn reduction, rooted at the concrete word.
pub fn word_stats_parts_sin_cos(t: i64) -> Option<(i64, i64)> {
    stats::parts::sin_cos(t)
}

/// The real power-of-two grid shift, rooted at the concrete word slice.
pub fn word_stats_parts_shift(a: &mut [i64], k: i32) -> Option<()> {
    stats::parts::shift(a, k)
}

/// The real panel normalisation, rooted at the concrete word slice.
pub fn word_stats_parts_panel(v: &[i64]) -> Option<(Vec<i64>, i32)> {
    stats::parts::panel(v)
}

/// The real `RMSNorm` row pass, rooted at the concrete Q32.32 words.
pub fn word_stats_parts_rms(
    x: &[i64],
    w: &[i64],
    eps: i64,
    t: usize,
    width: usize,
) -> Result<Vec<i64>, stats::parts::PartsError> {
    stats::parts::rms(x, w, eps, t, width)
}

/// The real `RoPE` half-rotation, rooted at the concrete Q32.32 words and rope tables.
pub fn word_stats_parts_rope_apply(
    dst: &mut [i64],
    src: &[i64],
    t: usize,
    n_head: usize,
    hd: usize,
    rope: &stats::parts::RopeTables,
) -> Result<(), stats::parts::PartsError> {
    stats::parts::rope_apply(dst, src, t, n_head, hd, rope)
}

/// The real `silu` on one Q32.32 word, rooted at the concrete word.
pub fn word_stats_parts_silu(g: i64) -> Option<i64> {
    stats::parts::silu(g)
}

/// The real `silu(g)·u` on words, rooted at the concrete words.
pub fn word_stats_parts_silu_mul(g: i64, u: i64) -> Option<i64> {
    stats::parts::silu_mul(g, u)
}

/// The real [`stats::parts::RopeTables::head_dim`] accessor, rooted at the concrete table.
pub fn word_stats_parts_rope_head_dim(rope: &stats::parts::RopeTables) -> usize {
    rope.head_dim()
}

/// The real [`stats::parts::RopeTables::t_max`] accessor, rooted at the concrete table.
pub fn word_stats_parts_rope_t_max(rope: &stats::parts::RopeTables) -> usize {
    rope.t_max()
}

// ── `learn::fold::fit` ─────────────────────────────────────────────────────────────────────

/// The rounded low-rank corrector of `ΔW` in the metric `Hl`.
pub fn word_fit_lowrank_corrector(
    delta_w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    rank: usize,
    mode: fit::corrector::Mode,
    factors: bool,
) -> Result<fit::corrector::LowrankCorrector, fit::corrector::CorrectorError> {
    fit::corrector::lowrank_corrector(delta_w, hl, o, n, rank, mode, factors)
}

/// The exact eigenstage `M = (ΔW·Hl)·ΔWᵀ` and its top-`rank` eigenvalue sum.
pub fn word_fit_eigen_parts(
    delta_w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    rank: usize,
) -> Result<(Vec<i64>, i64), fit::corrector::CorrectorError> {
    fit::corrector::eigen_parts(delta_w, hl, o, n, rank)
}

/// The symmetric eigendecomposition.
pub fn word_fit_eigh(
    a: &[i64],
    n: usize,
) -> Result<(Vec<i64>, Vec<i64>), fit::eigh::EighError> {
    fit::eigh::eigh(a, n)
}

/// The `o × o` Gram `(ΔW·Hl)·ΔWᵀ` in the word carrier.
pub fn word_fit_eigen_gram(
    delta_w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
) -> Result<(Vec<i64>, i32), fit::eigh::EighError> {
    fit::eigh::eigen_gram(delta_w, hl, o, n)
}

/// The exact rank-`rank` corrector.
pub fn word_fit_lowrank_corrector_exact(
    delta_w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    rank: usize,
) -> Result<fit::eigh::Corrector, fit::eigh::EighError> {
    fit::eigh::lowrank_corrector_exact(delta_w, hl, o, n, rank)
}

/// The hierarchical group-scale quantiser.
pub fn word_fit_quant_r_hier(
    r: &[i64],
    o: usize,
    g: usize,
) -> Result<fit::quant::RHier, fit::quant::QuantError> {
    fit::quant::quant_r_hier(r, o, g)
}

/// Affine column-scale quantization.
pub fn word_fit_quant_c_u8(
    c: &[i64],
    bits: u32,
) -> Result<(Vec<i64>, u64), fit::quant::QuantError> {
    fit::quant::quant_c_u8(c, bits)
}

/// Bit-exact binary16 rounding of a word.
pub fn word_fit_f16r(w: i64) -> Result<i64, fit::quant::QuantError> {
    fit::quant::f16r(w)
}

/// The stored-grid level of code `k` on `(lo, sp)`: `None` when it leaves the word.
pub fn word_fit_r_level(lo: i64, sp: i64, k: u8) -> Option<i64> {
    fit::quant::r_level(lo, sp, k)
}

/// [`RHier::reconstruct`]: decode `rq` from the stored per-row grid and 4-bit codes.
pub fn word_fit_r_hier_reconstruct(r: &fit::quant::RHier) -> Vec<i64> {
    r.reconstruct()
}

/// The exact power-of-two exponent [`eigen_parts`] applies to the raw Gram words.
pub fn word_fit_gram_scale(m: &[i64]) -> i32 {
    fit::eigh::gram_scale(m)
}
