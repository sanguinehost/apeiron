#![warn(missing_docs)]
#![allow(unexpected_cfgs)] // `cfg(kani)` is set by `cargo kani`, which cargo does not know about.

//! The two-slot algebra, and what it licenses.
//!
//! One primitive: `R[x]/(x^2 - sigma)`. A cell is an ordered pair holding two values, each
//! readable back independently, and `sigma` is a single real dial. Every 2-dimensional unital
//! commutative associative algebra over the reals is this for some `sigma`, and the sign of
//! `sigma` -- the discriminant -- is the only thing that varies:
//!
//! | sigma | name | channels | nontrivial idempotents | discard exists |
//! |---|---|---|---|---|
//! | `< 0` | elliptic | none | no | no |
//! | `= 0` | parabolic | one (`a`) | no | no |
//! | `> 0` | hyperbolic | two (`a+b`, `a-b`) | yes | **yes** |
//!
//! That table is the whole crate. Each row is a theorem in `../sanguine-apeiron`
//! (`proof/Apeiron/Trichotomy/`), and each column becomes something a table can or cannot do:
//! two channels are two lanes that accumulate without coordination, and a discard is what a
//! lossy key fold, an interest cull or an LOD demotion all are.
//!
//! # What the dial builds
//!
//! | face | keystore | the thing | in |
//! |---|---|---|---|
//! | elliptic | **B** -- the address is the key | [`Words`], a map that stores no key | [`map`] |
//! | parabolic | **B/U** -- the key rides beside it | [`Map`], a map for opaque and string keys | [`map`] |
//! | hyperbolic | **NB** -- the key is discarded | [`CountTable`], counts per address | [`table`] |
//!
//! and one merge over all three ([`merge`]), because commutativity and associativity hold on every
//! face with no hypothesis on the dial while the keystore varies with it. That asymmetry is the
//! result; the three faces are not a naming scheme laid over one implementation.
//!
//! [`fold`] is what makes the parabolic face necessary rather than optional: it addresses keys that
//! are not already addresses, that is a discard, and a discard has fibers only the key itself can
//! separate.
//!
//! **The point of the crate is that the dial is real.** Naming three storage faces after the
//! trichotomy while branching on nothing is a vocabulary, not an implementation. Here `Sigma` is
//! a value, the operations take it, and the things only hyperbolic can do are unavailable
//! elsewhere by construction rather than by comment.

pub mod cell;
pub mod coord;
pub mod fold;
pub mod keystore;
pub mod map;
pub mod merge;
pub mod pepper;
pub mod proofs;
pub mod scalar;
pub mod table;

pub use cell::{Cell, Sigma};
pub use keystore::Keystore;
pub use map::{Exposed, ExposedWords, Map, Stats, Words};
pub use pepper::{process_pepper, set_process_pepper};
pub use table::CountTable;
