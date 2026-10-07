#![warn(missing_docs)]
#![warn(clippy::allow_attributes_without_reason)]

//! The two-slot algebra `R[x]/(x^2 - sigma)` and one corpus-specified table.
//!
//! [`Table`] is the sole storage type. It is tested; its source refinement is pending.

pub mod aes;
pub mod attn_cut;
pub mod carrier;
pub mod cell;
pub mod chacha;
pub mod coord;
pub mod crypto;
pub mod engine;
pub mod exchange;
pub mod fold;
pub mod format;
/// The incumbent's exact NB count face (`FreqMap`) and map face (`Table`), vendored into this crate.
pub mod freq;
// Vendored third-party source, kept as imported from fold rev `4cc14471`. The host lint policy is
// scoped off it by the module-level `#![allow]` in `src/hashtrinity/mod.rs` (measured: 1241
// diagnostics -> 0); `rustfmt` is scoped off it by the `#[rustfmt::skip]` below.
//
// Measured on this tree, the attribute alone is the whole exemption: removing it restores all 90
// formatting diffs across 14 vendored files. A nested `src/hashtrinity/.rustfmt.toml` is *not*
// needed and is deliberately absent -- adding `skip_children = true` there changes nothing while the
// attribute is present, and `ignore` in a nested config is stdin-only per rustfmt, so it would be
// dead configuration.
//
// Cost, stated rather than hidden: the attribute covers the neighbouring `mod` declarations in this
// file as well. That costs nothing today -- this list is rustfmt-canonical, checked in place with the
// subtree stubbed to an empty file, 0 diffs here -- but a misordered `mod` added next to this one
// would no longer be reported.
/// The incumbent, vendored: `hashtrinity` at rev `4cc14471` as this crate's own module.
/// Private, because `freq` is its public face. The external `hashtrinity` git dependency is
/// removed (owner direction, 2026-10-05), so the doctests in the submodules are marked `ignore`:
/// they document the upstream crate's API, and this crate's public path for the port is `freq`.
#[rustfmt::skip]
mod hashtrinity;
/// The sovereign GPU runtime (dlopen Vulkan device + dispatch) over the ported `simt`
/// kernels. Behind the `gpu` feature; the kernel builders themselves stay always-on.
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod kepler;
pub mod keystore;
pub mod merge;
pub mod numeric;
pub mod pepper;
pub mod policy;
pub mod scalar;
pub mod sha256;
/// The ported SIMT (GPU) substrate: the 12-op alphabet encoded to SPIR-V, the carrier-dial GEMV
/// fold and the integer-dot (DP4A) chart GEMV. Faithful port of fold's `modelfold::simt`.
pub mod simt;
/// The ported SIMT neural network glue kernels: RMSNorm, RoPE, attention, SwiGLU, argmax,
/// prefill tiles. Faithful port of fold's `modelfold::simt_nn`.
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
