#![warn(missing_docs)]
#![warn(clippy::allow_attributes_without_reason)]

//! The two-slot algebra `R[x]/(x^2 - sigma)`, the growable tables built on it, and an integer
//! inference engine.
//!
//! [`Cell`] is the algebra: a pair over a ring with multiplication and norm form, and `Sigma`
//! selecting the elliptic, parabolic or hyperbolic face. The `Cell<i64>` word bodies are extracted
//! to Lean and refined to the canonical definitions (`proof/`). [`freq`] holds the executed tables
//! (`Table`, `CoordMap`, `FreqMap`); they are tested, and their refinement to the canonical
//! `Apeiron.Table` is open.
//!
//! | area | modules |
//! |---|---|
//! | algebra | `cell`, `scalar`, `numeric`, `kepler`, `carrier`, `exchange` |
//! | addressing and tables | `coord`, `fold`, `freq`, `table`, `keystore`, `merge`, `pepper` |
//! | engine | `format` (GGUF, quant unpacking, BPE), `engine`, `attn_cut`, `learn` |
//! | GPU | `simt`, `simt_nn`, `spirv`, `gpu` (feature `gpu`) |
//! | policy | `policy` |
//! | unaudited crypto | `aes`, `chacha`, `crypto`, `sha256` |
//!
//! `learn`, `attn_cut`, `carrier`, `exchange` and `kepler` are experiments: tested, with no
//! stability promise.

/// AES-128 (FIPS 197). Unaudited: tested against the SP 800-38A known-answer vector only. Not
/// side-channel resistant: the S-box is indexed by secret bytes. Do not use it to protect real data;
/// its rooting onto `R[x]/(x² − σ)` is in progress.
pub mod aes;
pub mod attn_cut;
pub mod carrier;
pub mod cell;
/// ChaCha20-Poly1305 (RFC 8439). Unaudited: tested against the RFC 8439 known-answer vectors only.
/// Its constant-time construction is not measured, so treat it as not side-channel resistant. Do not
/// use it to protect real data; its rooting onto `R[x]/(x² − σ)` is in progress.
pub mod chacha;
pub mod coord;
/// SHA-3 and SHAKE (FIPS 202). Unaudited: tested against the NIST known-answer vectors only, with no
/// side-channel review. Do not use it to protect real data; its rooting onto `R[x]/(x² − σ)` is in
/// progress.
pub mod crypto;
pub mod engine;
pub mod exchange;
pub mod fold;
pub mod format;
/// The incumbent's exact NB count face (`FreqMap`) and map face (`Table`), ported into this crate.
pub mod freq;
// `hashtrinity`, ported from the author's `fold` repository at rev `4cc14471` with its upstream
// layout. The host lint policy is scoped off it by the module-level `#![allow]` in
// `src/hashtrinity/mod.rs`; `rustfmt` is scoped off it by the `#[rustfmt::skip]` below, which also
// covers the neighbouring `mod` declarations in this file.
/// The subsumed incumbent: `hashtrinity` at rev `4cc14471`, ported as this crate's own module.
/// Private, because `freq` is its public face. The external `hashtrinity` git dependency is
/// removed, so the doctests in the submodules are marked `ignore`:
/// they document the upstream crate's API, and this crate's public path for the port is `freq`.
#[rustfmt::skip]
mod hashtrinity;
/// The GPU runtime (dlopen Vulkan device + dispatch) over the ported `simt`
/// kernels. Behind the `gpu` feature; the kernel builders themselves stay always-on.
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod kepler;
pub mod keystore;
/// Training, finetuning and quantization.
pub mod learn;
pub mod merge;
pub mod numeric;
pub mod pepper;
pub mod policy;
pub mod scalar;
/// SHA-256 (FIPS 180-4). Unaudited: tested against the FIPS 180-4 known-answer vectors only, with no
/// side-channel review. Do not use it to protect real data; its rooting onto `R[x]/(x² − σ)` is in
/// progress.
pub mod sha256;
/// The ported SIMT (GPU) substrate: the 12-op alphabet encoded to SPIR-V, the carrier-dial GEMV
/// fold and the integer-dot (DP4A) chart GEMV. Ported from the author's `fold` repository.
pub mod simt;
/// The ported SIMT neural network glue kernels: RMSNorm, RoPE, attention, SwiGLU, argmax,
/// prefill tiles. Ported from the author's `fold` repository.
pub mod simt_nn;
pub mod spirv;
pub mod table;

/// Coordinate and packed-key derives, available with the `derive` feature.
#[cfg(feature = "derive")]
pub use apeiron_derive::{Coord, Packed};

pub use carrier::{
    conservative_density_step, diffusive_flux_half, diffusive_line3_step, point_density,
    point_flux, DensityLine3,
};
pub use cell::{Cell, Sigma};
pub use exchange::{accept_derived_impulse, elastic_exchange, exchange_impulse, Exchange};
pub use pepper::{process_pepper, set_process_pepper};
pub use table::{StorageError, Table};
