//! The four merges, and why there is only one of them.
//!
//! A table is a probe and a merge. The probe varies with the dial -- three faces, three keystores,
//! [`crate::keystore`] derives them from the discriminant. The merge does not, and that asymmetry
//! is the other half of the result.
//!
//! Order-free accumulation needs commutativity and associativity. Both hold on **every** face of
//! `R[x]/(x^2 - sigma)` with no hypothesis on `sigma` at all (`crate::cell::fold`, and
//! `tests/three_and_one.rs` checks it over all 120 orderings on all three faces). So splitting a
//! stream across threads and recombining is licensed everywhere, and nothing about which face a
//! table sits on changes which merges are available to it. One merge, three keystores.
//!
//! What varies is not the algebra but the *value monoid* the caller supplies, and there the count
//! is four because there are four shapes of thing a program wants a keyed accumulator to be:
//!
//! | face | merge | identity | order-free |
//! |---|---|---|---|
//! | set | [`keep`] | `false` | yes |
//! | map | [`set`] | absent | **no** -- last write wins |
//! | counter | [`count`] | `Monoid::identity` | yes |
//! | graph | [`link`] | `[]` | **no** -- append is not commutative |
//!
//! The counter row is order-free only because [`Monoid`] promises associativity. Floating-point
//! addition does not, so `f32`/`f64` are [`OrderedMonoid`] and count through
//! [`approx_count`]/[`approx_count_into`], which fixes the stream order and is [`Species::Ordered`],
//! not [`Species::Free`]. The property belongs to the value type, not the face.
//!
//! The four are not derived from the trichotomy and this module does not claim they are. Two are
//! commutative and two are directional, and [`Species`] records which, because a table built with
//! a directional merge cannot be built in parallel and recombined -- that is a real precondition
//! and it belongs beside the merge rather than in a caller's head.
//!
//! # Table integration
//!
//! [`keep`], [`set`], [`count`] and [`link`] are the merges *as functions*, `(new, old) -> V`.
//! That is the form the table above names and the form the commutativity and associativity
//! arguments are stated about, so they are the definitions.
//!
//! The single table's required merge signature is `(new, old) -> V`, including the default on
//! the first write. The standalone `*_into` helpers below are not a second table path and do not
//! define the table's mutation or panic contract.
//!
//! ```
//! use apeiron::coord::address::{
//!     EllipticPacking, FaceCoord, FaceHome, FaceResidueCapability, ParabolicDual, Residue,
//!     WordFaceHome, H3_40,
//! };
//! use apeiron::table::{Home, TableAddress};
//! use apeiron::Table;
//! use apeiron::Sigma;
//!
//! /// The named explicit home: physical placement plus the word semantic capability. The face
//! /// map is derived from the key, never from the placement.
//! #[derive(Clone, Copy)]
//! struct WordKeyHome;
//!
//! impl Home<u64> for WordKeyHome {
//!     fn home(&self, key: &u64) -> u64 {
//!         *key
//!     }
//!
//!     fn address(&self, _sigma: Sigma, key: &u64) -> TableAddress {
//!         TableAddress {
//!             place: self.home(key),
//!             direct_word: None,
//!         }
//!     }
//! }
//!
//! impl FaceResidueCapability<u64> for WordKeyHome {
//!     fn parabolic_dual(&self, key: &u64) -> ParabolicDual {
//!         WordFaceHome.parabolic_dual(key)
//!     }
//!
//!     fn hyperbolic_scalar(&self, profile: H3_40, key: &u64) -> Residue {
//!         WordFaceHome.hyperbolic_scalar(profile, key)
//!     }
//! }
//!
//! impl EllipticPacking<u64> for WordKeyHome {
//!     fn elliptic_pack(&self, key: &u64) -> u64 {
//!         WordFaceHome.elliptic_pack(key)
//!     }
//!
//!     fn elliptic_unpack(&self, word: u64) -> Option<u64> {
//!         WordFaceHome.elliptic_unpack(word)
//!     }
//! }
//!
//! impl FaceCoord<u64> for WordKeyHome {
//!     type Faces = Sigma;
//! }
//!
//! impl FaceHome<u64> for WordKeyHome {}
//!
//! // The explicit placement must be the home word, which is what makes the removal of the trait
//! // default observable.
//! for k in [7u64, 9] {
//!     let address = Home::address(&WordKeyHome, Sigma::default(), &k);
//!     assert_eq!(address.place, WordKeyHome.home(&k));
//!     assert_eq!(address.direct_word, None);
//! }
//!
//! let mut counts = Table::<u64, u32, WordKeyHome>::new(0, WordKeyHome);
//! for k in [7, 7, 9] {
//!     counts.insert_with(&apeiron::merge::count, k, 1).unwrap();
//! }
//! assert_eq!(*counts.get(&7), 2);
//! assert_eq!(*counts.get(&9), 1);
//! ```

/// Which merge a table is built with.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Face {
    /// Set membership (`||`).
    Keep,
    /// Map: last write wins.
    Set,
    /// Counter: a commutative monoid.
    Count,
    /// Graph: append.
    Link,
}

/// Whether a merge is order-free.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Species {
    /// Commutative, so a build may be split across threads and recombined.
    Free,
    /// Directional: the result depends on the order the stream arrived in.
    Ordered,
}

impl Face {
    /// Whether a build with this merge may be split and recombined.
    #[must_use]
    pub const fn species(self) -> Species {
        match self {
            Self::Keep | Self::Count => Species::Free,
            Self::Set | Self::Link => Species::Ordered,
        }
    }

    /// Whether a build with this merge may be split and recombined.
    #[must_use]
    pub const fn is_fusible(self) -> bool {
        matches!(self.species(), Species::Free)
    }
}

/// Set membership. Identity `false`.
#[inline]
#[must_use]
pub const fn keep(new: bool, old: bool) -> bool {
    new || old
}

/// Map: the newer value wins. Identity is the absent key.
#[inline]
#[must_use]
pub fn set<V>(new: V, _old: V) -> V {
    new
}

/// Counter: combine through the monoid. Identity is what an absent key reads as.
#[inline]
#[must_use]
pub fn count<V: Monoid>(new: V, old: V) -> V {
    new.combine(old)
}

/// [`count`] for an ordered, approximate accumulator.
///
/// The order is the caller's, so the result is only reproducible if the stream is. This is the
/// honest form of a float sum: it is not [`Monoid`] because the arithmetic is not associative, and
/// calling it `count` would hide that a split-and-recombine build is unsound here.
#[inline]
#[must_use]
pub fn approx_count<V: OrderedMonoid>(new: V, old: V) -> V {
    new.combine(old)
}

/// Graph: newest first.
///
/// Temporal `O(old.len())` amortized element moves: `extend` pushes each element of `old` once, and
/// the `Vec` grows geometrically, so a reallocation copies the prefix already present at most a
/// constant number of times per element; the count is in elements moved, not comparisons. Spatial
/// `O(new.len() + old.len())` elements owned by the returned `Vec`, plus the slack the geometric
/// growth leaves in its capacity. `new` is consumed, so the input's allocation is reused when it
/// has room.
#[inline]
#[must_use]
pub fn link<T>(mut new: Vec<T>, old: Vec<T>) -> Vec<T> {
    new.extend(old);
    new
}

// -- the same four, in place ---------------------------------------------------------------------
//
// Standalone in-place helpers; the table does not use these as its write contract.

/// [`keep`], applied in place.
#[inline]
pub fn keep_into(slot: &mut bool, new: bool) {
    *slot = keep(new, *slot);
}

/// [`set`], applied in place.
#[inline]
pub fn set_into<V>(slot: &mut V, new: V) {
    *slot = new;
}

/// [`count`], applied in place.
///
/// The old value is swapped out for the monoid's identity rather than read out and abandoned, so
/// the slot always holds a live `V`. If `combine` panics the entry survives holding the identity,
/// which is a value the caller can see and reason about -- not an uninitialised slot.
#[inline]
pub fn count_into<V: Monoid>(slot: &mut V, new: V) {
    let old = core::mem::replace(slot, V::identity());
    *slot = count(new, old);
}

/// [`approx_count`], applied in place, with the same identity-swap failure atomicity as
/// [`count_into`].
#[inline]
pub fn approx_count_into<V: OrderedMonoid>(slot: &mut V, new: V) {
    let old = core::mem::replace(slot, V::identity());
    *slot = approx_count(new, old);
}

/// [`link`], applied in place: `new` is spliced onto the front, newest first, without rebuilding
/// the tail.
///
/// Temporal `O(slot.len() + new.len())` element moves: `splice(0..0, new)` inserts every element of
/// `new` at the front and shifts the existing `slot.len()` elements up by that many, so each element
/// of the result is moved at most once; the count is in elements moved, not comparisons. Spatial
/// `O(new.len())` additional elements held by `slot`, plus any capacity reserved for the insertion.
#[inline]
pub fn link_into<T>(slot: &mut Vec<T>, new: Vec<T>) {
    slot.splice(0..0, new);
}

/// A commutative monoid: what makes a counting build order-free, and therefore splittable.
pub trait Monoid {
    /// What an absent key reads as.
    fn identity() -> Self;
    /// Combine two values for one key. Must be commutative and associative -- `count` is the merge
    /// that a parallel build reorders, and reordering is only invisible if this is.
    #[must_use]
    fn combine(self, other: Self) -> Self;
}

macro_rules! sum_monoid {
    ($($t:ty),*) => {$(
        impl Monoid for $t {
            #[inline]
            fn identity() -> Self { 0 }
            #[inline]
            fn combine(self, other: Self) -> Self { self.wrapping_add(other) }
        }
    )*};
}
sum_monoid!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

macro_rules! wide_sum_monoid {
    ($($t:ty),*) => {$(
        impl Monoid for $t {
            #[inline]
            fn identity() -> Self { 0 }
            #[inline]
            fn combine(self, other: Self) -> Self { self.wrapping_add(other) }
        }
    )*};
}
wide_sum_monoid!(u128, i128);

/// An ordered, approximate accumulator: commutative, but **not** associative, which is why it is
/// not [`Monoid`] and cannot license a split-and-recombine build. Floating-point addition is the
/// motivating case -- `(1e20 + -1e20) + 1 = 1` but `1e20 + (-1e20 + 1) = 0` -- so a count over it
/// must fix the order, as [`approx_count_into`] does when a table keeps a key's occurrences in
/// stream order.
pub trait OrderedMonoid {
    /// What an absent key reads as.
    fn identity() -> Self;
    /// Combine two values for one key, in the order given. Commutative but not associative.
    #[must_use]
    fn combine(self, other: Self) -> Self;
}

impl OrderedMonoid for f32 {
    #[inline]
    fn identity() -> Self {
        0.0
    }
    #[inline]
    fn combine(self, other: Self) -> Self {
        self + other
    }
}

impl OrderedMonoid for f64 {
    #[inline]
    fn identity() -> Self {
        0.0
    }
    #[inline]
    fn combine(self, other: Self) -> Self {
        self + other
    }
}

impl Monoid for bool {
    #[inline]
    fn identity() -> Self {
        false
    }
    #[inline]
    fn combine(self, other: Self) -> Self {
        self || other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exactly_two_of_the_four_are_fusible() {
        let all = [Face::Keep, Face::Set, Face::Count, Face::Link];
        assert_eq!(all.iter().filter(|f| f.is_fusible()).count(), 2);
        assert!(Face::Keep.is_fusible() && Face::Count.is_fusible());
        assert!(!Face::Set.is_fusible() && !Face::Link.is_fusible());
    }

    /// Each in-place merge must be the pure one it is named after. They are separate code -- one
    /// returns a value, the other writes through a reference -- so agreeing is a property to check,
    /// not a thing to assume.
    #[test]
    fn every_in_place_merge_agrees_with_its_pure_twin() {
        for &new in &[false, true] {
            for &old in &[false, true] {
                let mut slot = old;
                keep_into(&mut slot, new);
                assert_eq!(slot, keep(new, old), "keep({new}, {old})");
            }
        }
        for new in -50i64..50 {
            for old in -50i64..50 {
                let mut slot = old;
                count_into(&mut slot, new);
                assert_eq!(slot, count(new, old), "count({new}, {old})");

                let mut slot = old;
                set_into(&mut slot, new);
                assert_eq!(slot, set(new, old), "set({new}, {old})");
            }
        }
        for n in 0..6usize {
            for o in 0..6usize {
                let (new, old): (Vec<usize>, Vec<usize>) =
                    ((0..n).collect(), (100..100 + o).collect());
                let mut slot = old.clone();
                link_into(&mut slot, new.clone());
                assert_eq!(slot, link(new, old), "link at {n}/{o}");
            }
        }
    }

    /// The claim `Species::Free` makes, checked rather than asserted.
    #[test]
    fn the_fusible_merges_are_commutative_and_the_others_are_not() {
        assert_eq!(keep(true, false), keep(false, true));
        assert_eq!(count(3i64, 5), count(5i64, 3));
        assert_ne!(set(1, 2), set(2, 1));
        assert_ne!(link(vec![1], vec![2]), link(vec![2], vec![1]));
    }

    /// The law [`Monoid`] promises and `count` relies on: `combine` is associative, which is what
    /// makes reordering a split build invisible. Checked on the exact implementing types.
    #[test]
    fn monoid_combine_is_associative() {
        fn agrees<V: Monoid + Copy + PartialEq + core::fmt::Debug>(a: V, b: V, c: V) {
            assert_eq!(
                a.combine(b).combine(c),
                a.combine(b.combine(c)),
                "combine is not associative"
            );
        }
        agrees(1u64, 2, 3);
        agrees(-1i64, 2, 3);
        agrees(true, false, true);
    }

    /// The counterexample that keeps the floats out of [`Monoid`]: floating addition is commutative
    /// but not associative, so it is an [`OrderedMonoid`] and only an ordered reduction over it is
    /// sound.
    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "the test is about the exact floating-point results, not approximate equality"
    )]
    fn ordered_combine_is_commutative_but_not_associative() {
        let (a, b, c) = (1e20f64, -1e20f64, 1.0f64);
        assert_eq!(a.combine(b), b.combine(a), "combine is commutative");
        assert_ne!(
            a.combine(b).combine(c),
            a.combine(b.combine(c)),
            "floating-point combine must not be advertised as associative"
        );
    }
}
