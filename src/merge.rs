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
//! The four are not derived from the trichotomy and this module does not claim they are. Two are
//! commutative and two are directional, and [`Species`] records which, because a table built with
//! a directional merge cannot be built in parallel and recombined -- that is a real precondition
//! and it belongs beside the merge rather than in a caller's head.
//!
//! # Why each merge appears twice
//!
//! [`keep`], [`set`], [`count`] and [`link`] are the merges *as functions*, `(new, old) -> V`.
//! That is the form the table above names and the form the commutativity and associativity
//! arguments are stated about, so they are the definitions.
//!
//! [`keep_into`], [`set_into`], [`count_into`] and [`link_into`] are the same four with the
//! signature [`crate::map::Map::insert_with`] actually takes, `(&mut V, V)`. **Use these.** The
//! in-place form is the crate's insertion primitive rather than a convenience over the by-value
//! one, and the reason is failure atomicity: a merge that panics has to leave the map exactly as
//! it was, which a `(new, old) -> V` merge cannot promise because it has already moved the old
//! value out of the slot. `a_merge_that_panics_leaves_the_map_intact`, in
//! `tests/a_map_is_a_map.rs`, is what holds that.
//!
//! ```
//! use apeiron::map::Map;
//!
//! let mut counts: Map<u64, u32> = Map::new();
//! for k in [7, 7, 9] {
//!     counts.insert_with(k, 1, apeiron::merge::count_into);
//! }
//! assert_eq!(counts.get(&7), Some(&2));
//! assert_eq!(counts.get(&9), Some(&1));
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

/// Graph: newest first.
#[inline]
#[must_use]
pub fn link<T>(mut new: Vec<T>, old: Vec<T>) -> Vec<T> {
    new.extend(old);
    new
}

// -- the same four, in place ---------------------------------------------------------------------
//
// `Map::insert_with` takes `FnOnce(&mut V, V)` rather than `FnOnce(V, V) -> V`, because a merge
// that has to move the old value out of the slot to call the closure leaves the slot holding
// nothing for the duration -- and a panic in there loses the entry. The four above are the
// *statement*; these are how a table applies them. Each is checked against its pure twin.

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

/// [`link`], applied in place: `new` is spliced onto the front, newest first, without rebuilding
/// the tail.
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

impl Monoid for f32 {
    #[inline]
    fn identity() -> Self {
        0.0
    }
    /// As [`f64`]'s: not associative in the reals' sense, so a parallel build over it is
    /// deterministic only because the partition is.
    #[inline]
    fn combine(self, other: Self) -> Self {
        self + other
    }
}

impl Monoid for f64 {
    #[inline]
    fn identity() -> Self {
        0.0
    }
    /// Not associative in the reals' sense -- floating-point addition is not -- so a parallel build
    /// over this monoid is deterministic only if the partition is. It is: `crate::table` partitions
    /// by key, and a key's occurrences stay in stream order, so the sum for a key is summed in the
    /// same order however many threads run.
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
                let (new, old): (Vec<usize>, Vec<usize>) = ((0..n).collect(), (100..100 + o).collect());
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
}
