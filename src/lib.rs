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
//! **The point of the crate is that the dial is real.** Naming three storage faces after the
//! trichotomy while branching on nothing is a vocabulary, not an implementation. Here `Sigma` is
//! a value, the operations take it, and the things only hyperbolic can do are unavailable
//! elsewhere by construction rather than by comment.

pub mod cell;
pub mod keystore;
pub mod proofs;
pub mod scalar;
pub mod coord;

pub use cell::{Cell, Sigma};
