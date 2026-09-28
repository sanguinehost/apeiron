//! Bounded authorization decision core: [`decide`] and its rule vocabulary.
//!
//! Canonical target `Apeiron.Policy.decide` (`proof/corpus/Policy/Core.lean`).

pub mod decide;

pub use decide::{decide, Atom, Cause, Effect, Op, Rule, Verdict};
