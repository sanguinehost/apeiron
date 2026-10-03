#![warn(missing_docs)]
#![warn(clippy::allow_attributes_without_reason)]

//! The two-slot algebra `R[x]/(x^2 - sigma)` and one corpus-specified table.
//!
//! [`Table`] is the sole storage type. It is tested; its source refinement is pending.

pub mod aes;
pub mod carrier;
pub mod cell;
pub mod chacha;
pub mod coord;
pub mod crypto;
pub mod engine;
pub mod exchange;
pub mod fold;
pub mod format;
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
