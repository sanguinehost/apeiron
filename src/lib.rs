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

pub mod attn_cut;
pub mod carrier;
pub mod cell;
/// AES-128, ChaCha20-Poly1305, SHA-3/SHAKE and SHA-256. Unaudited; see each module.
pub mod crypto;
pub mod engine;
pub mod exchange;
pub mod format;
/// The GPU stack: the SPIR-V kernel builders (`simt`, `nn`), the SPIR-V fragment checker
/// (`spirv`), and behind the `gpu` feature the Vulkan runtime and the decode path.
pub mod gpu;
pub mod kepler;
/// Training, finetuning and quantization.
pub mod learn;
/// Every `K -> V` map in the crate and the key addressing it rests on.
pub mod map;
pub mod numeric;
pub mod policy;
pub mod scalar;

/// Coordinate and packed-key derives, available with the `derive` feature.
#[cfg(feature = "derive")]
pub use apeiron_derive::{Coord, Packed};

pub use carrier::{
    conservative_density_step, diffusive_flux_half, diffusive_line3_step, point_density,
    point_flux, DensityLine3,
};
pub use cell::{Cell, Sigma};
pub use exchange::{accept_derived_impulse, elastic_exchange, exchange_impulse, Exchange};
pub use map::pepper::{process_pepper, set_process_pepper};
pub use map::table::{StorageError, Table};
