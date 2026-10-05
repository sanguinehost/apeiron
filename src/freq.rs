//! The incumbent's NB count face (`FreqMap`) and map face (`Table`), vendored into this crate.
//!
//! `src/hashtrinity/` is `hashtrinity` at rev `4cc14471ae24f3f204fb0b0e91736ef75c909358`,
//! copied from the fold Git object at that revision and compiled here as this crate's own module.
//! The table core comes across (`mod.rs`, `algebra`, `freq`, `par`, `face`, `seed`), together with
//! the `tregex`, `codec` and `crypto` modules webfold still imports from the vendored tree, not
//! just `freq.rs`, because:
//!
//! - `E0603`: `bits_for_slots` (defined `hashtrinity/src/lib.rs:501`) and `EMPTY` (defined
//!   `hashtrinity/src/lib.rs:298`) are private, so `freq.rs:5` cannot import them from outside the
//!   crate. Inside `crate::hashtrinity` they are in scope.
//! - `E0116`: `freq.rs:112` writes an inherent `impl ... Table<K, i64, Regen<K>>`. An inherent impl
//!   may only be written in the crate that defines the type; here the type is
//!   `crate::hashtrinity::Table`, defined in the same crate.
//!
//! So the two refusals that made a bare file copy impossible are answered by vendoring the crate,
//! not by rewriting its logic. The changes to the copied tree are path lines — 66 `crate::`
//! references rewritten to `crate::hashtrinity::`, the `hex`/`base64` module gates removed (this
//! crate has no `hex`/`base64` feature and `crypto.rs` is unconditional), `lib.rs` renamed
//! `mod.rs` — and the removal of every module no retained consumer reaches. This in-tree tree is
//! the only copy; the external `hashtrinity` git dependency is gone.
//!
//! Both faces are re-exported here under one private path so the vendored and the external
//! comparator are two spellings of the same contract that a reader can name side by side:
//! `apeiron::freq::Table` is the type `src/hashtrinity/` defines, `hashtrinity::Table` is the one
//! the git dependency defines, and `std::any::type_name` tells them apart.

pub use crate::hashtrinity::{par, Coord, FreqMap, Table, PHI};
