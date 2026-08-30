#![warn(missing_docs)]
// Every lint suppression states why. This is the crate's own rule -- a claim carries its
// justification -- applied to the one place it was not being applied.
//
// The companion lint `clippy::allow_attributes`, which would additionally force every suppression
// to be `#[expect]`, is deliberately **not** enabled. `#[expect]` is the better tool and is used
// everywhere it works: it warns when the lint it silences stops firing, so a suppression whose
// cause has been engineered away removes itself rather than accruing. Turning it on found four
// such cases here immediately. But it cannot express "fires for some expansions of this macro and
// not others", which is exactly the shape of the four remaining `#[allow]`s, and the exemption
// would have to nest inside the macro body to take effect. A rule with a self-referential escape
// hatch in it is worse than a rule plus a convention, so: `expect` by default, `allow` with a
// reason where `expect` cannot apply, and that constraint is enforced.
#![warn(clippy::allow_attributes_without_reason)]
#![allow(
    unexpected_cfgs,
    reason = "`cfg(kani)` is set by `cargo kani`, which cargo itself does not know about"
)]

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
//! | elliptic | **regenerated** -- the address is the key | [`Words`], a map that stores no key | [`map`] |
//! | parabolic | **carried** -- the key rides beside it | [`Map`], a map for opaque and string keys | [`map`] |
//! | hyperbolic | **discarded** -- the key is gone | [`CountTable`], counts per address | [`table`] |
//!
//! and one merge over all three ([`merge`]), because commutativity and associativity hold on every
//! face with no hypothesis on the dial while the keystore varies with it. That asymmetry is the
//! result; the three faces are not a naming scheme laid over one implementation.
//!
//! [`fold`] is what makes the parabolic face necessary rather than optional: it addresses keys that
//! are not already addresses, that is a discard, and a discard has fibers only the key itself can
//! separate.
//!
//! # Choose the key before you choose the table
//!
//! The largest number in this crate's measurements is not in this crate. Copying a 13-byte `String`
//! key into a table costs **376.9 instructions** -- three times the insert it precedes, and the
//! same for `hashbrown` -- and it is the allocator rather than the copy: 376.9 at 13 bytes, 375.6
//! at 42, 399.4 at 200, flat across a 3x range in length. A key that fits in a fixed array does not
//! pay it at all, and [`Coord`](coord::Coord) is implemented for `[u8; N]`.
//!
//! Measured over 694 Neverwinter resrefs, instructions per key (`examples/three_costs.rs`):
//!
//! | | `Map<String, V>` | `Map<[u8; 16], V>` | |
//! |---|---|---|---|
//! | copy the key in | 376.90 | **0.07** | a `malloc` against a `memcpy` |
//! | build the table | 500.91 | **86.87** | **5.8x** |
//! | look one key up | 97.64 | **50.80** | **1.9x** |
//!
//! A `ResRef` is at most sixteen bytes by the format's own definition, so `[u8; 16]` is not a
//! compromise there -- it is the type. Nothing this crate could win against another hash table is
//! worth a fifth of that difference, and the honest advice is to stop owning each key: use a fixed
//! array where the key has a bound, and an interned handle where it does not.
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
pub mod set;
pub mod table;

/// `#[derive(Coord)]` and `#[derive(Packed)]`, behind the `derive` feature.
///
/// Off by default so the ordinary build keeps zero dependencies; turning it on is what a codebase
/// with its own key types wants.
#[cfg(feature = "derive")]
pub use apeiron_derive::{Coord, Packed};

pub use cell::{Cell, Sigma};
pub use map::{Exposed, ExposedWords, Map, Stats, Words};
pub use set::{ExposedSet, Set, WordSet};
pub use pepper::{process_pepper, set_process_pepper};
pub use table::CountTable;
