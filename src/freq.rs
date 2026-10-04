//! The incumbent's NB count face (`FreqMap`), called directly from the pinned `hashtrinity` crate.
//!
//! A byte-for-byte copy of `hashtrinity/src/freq.rs` provably cannot compile here, with path lines
//! as the only change:
//!
//! - `E0603`: `bits_for_slots` (defined `hashtrinity/src/lib.rs:501`) and `EMPTY` (defined
//!   `hashtrinity/src/lib.rs:298`) are private and not re-exported, so `freq.rs:5` cannot import
//!   them from outside the crate.
//! - `E0116`: `freq.rs:112` writes an inherent `impl ... Table<K, i64, Regen<K>>`, and
//!   `hashtrinity::Table` is a foreign type; an inherent impl may only be written in the crate that
//!   defines the type.
//!
//! Each refusal is a privacy or coherence fact, not a path spelling, so no `crate::`→`hashtrinity::`
//! rewrite reaches it without editing the copied logic. The incumbent is therefore referenced
//! directly, unmodified, at the exact pin in `Cargo.toml`.

pub use hashtrinity::FreqMap;
