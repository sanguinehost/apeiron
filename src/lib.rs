#![warn(missing_docs)]
#![warn(clippy::allow_attributes_without_reason)]
#![allow(
    unexpected_cfgs,
    reason = "`cfg(kani)` is set by `cargo kani`, which cargo itself does not know about"
)]

//! The two-slot algebra `R[x]/(x^2 - sigma)` and one corpus-specified table.
//!
//! [`Table`] is the sole storage type. Its implementation and source refinement are pending;
//! the removed containers' proofs and measurements do not establish this type's behavior.

pub mod carrier;
pub mod cell;
pub mod coord;
pub mod exchange;
pub mod fold;
pub mod keystore;
pub mod merge;
pub mod pepper;
pub mod proofs;
pub mod scalar;
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
pub use table::Table;
