//! A set, which is a map whose values are nothing.
//!
//! `Set<K>` is `Map<K, (), H>` with the value hidden, and that is the whole implementation. `()` is
//! zero-sized, so a slot costs exactly what its [`Held`] costs and not a byte more -- an elliptic
//! set stores one `u64` per entry and no key at all.
//!
//! It exists because a set is not a map, at the call site. `map.insert(k, ())` returning
//! `Option<()>` is the wrong answer to "was it already there", `map.keys()` is the wrong name for
//! the elements, and there is nowhere to hang `is_subset`. Every one of those is a small thing that
//! a person has to translate in their head on every line.
//!
//! # Ordering
//!
//! Iteration is in address order: not insertion order, not sorted, but **the same every run** on
//! every machine, for the same elements under the same pepper. That is worth naming because it is
//! the property a `BTreeSet<u32>` is usually bought for -- thirteen of Lyrium's fourteen are
//! `BTreeSet<u32>` holding jump targets and block ids, ordered so that decompiler output can be
//! fingerprinted in a test rather than because anything queries a range. A set that iterates
//! reproducibly without paying for a tree serves that directly.

use crate::cell::Sigma;
use crate::coord::Coord;
use crate::map::{Absorb, Drain, Held, IntoKeys, Keys, Locate, Map, Regen, Salt, Stats, Stored};
use std::num::NonZeroU64;

/// A set on one of the two keyed faces.
///
/// `Set<K>` is the parabolic default and takes any `K: Coord + Eq`. [`WordSet`] is the elliptic
/// alias, for elements that fit in a word and are therefore stored as nothing but their address.
pub struct Set<K, H = Stored<K>>(Map<K, (), H>);

/// The elliptic face: elements that fit in a word, addressed bijectively, stored as the address.
pub type WordSet<K> = Set<K, Regen<K>>;

impl<K: Coord + Eq, H: Held<K>> Default for Set<K, H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Coord + Eq, H: Held<K>> Set<K, H> {
    /// An empty set, taking the process pepper. See [`crate::pepper`].
    #[must_use]
    pub fn new() -> Self {
        Self(Map::new())
    }

    /// An empty set sized for `n` elements.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self(Map::with_capacity(n))
    }

    /// An empty set with an explicit pepper, ignoring the process one.
    #[must_use]
    pub fn with_pepper(pepper: u64) -> Self {
        Self(Map::with_pepper(pepper))
    }

    /// An empty set with a given address bijection.
    #[must_use]
    pub fn with_salt(salt: Salt) -> Self {
        Self(Map::with_salt(salt))
    }

    /// Membership for every element of `ks`, in order.
    pub fn contains_each<'a>(&'a self, ks: &'a [K], mut visit: impl FnMut(&'a K, bool)) {
        self.0.get_each(ks, |k, v| visit(k, v.is_some()));
    }

    /// How many of `ks` this set holds.
    #[must_use]
    pub fn count_present(&self, ks: &[K]) -> usize {
        self.0.get_reduce(ks, 0usize, |()| 1usize, |a, b| a + b)
    }

    /// The map underneath, for the operations a set does not name.
    ///
    /// `Set` is deliberately thinner than [`Map`] -- it has no `entry`, no `extract_if`, no
    /// `Index` -- because nothing has needed them yet. This is the escape hatch that keeps that
    /// decision cheap to have got wrong: a reader that a set does not expose is one deref away.
    ///
    /// ```
    /// use apeiron::set::Set;
    ///
    /// let s: Set<u64> = [1u64, 2, 3].into_iter().collect();
    /// assert_eq!(s.as_map().len(), 3);
    /// assert!(s.as_map().contains_key(&2));
    /// ```
    #[must_use]
    pub fn as_map(&self) -> &Map<K, (), H> {
        &self.0
    }

    /// Live elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the set holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Elements this set holds before it must grow.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    /// Which keyed face this set is on.
    #[must_use]
    pub fn face(&self) -> Sigma {
        H::FACE
    }

    /// This set's pepper.
    #[must_use]
    pub fn pepper(&self) -> u64 {
        self.0.pepper()
    }

    /// The report. See [`Map::stats`].
    #[must_use]
    pub fn stats(&self) -> Stats {
        self.0.stats()
    }

    /// Add `k`. **True when it was not already there**, which is the answer a set is asked for and
    /// the one `Map::insert`'s `Option<()>` cannot give without a squint.
    pub fn insert(&mut self, k: K) -> bool {
        self.0.insert(k, ()).is_none()
    }

    /// Add `k`, returning the element it displaced.
    ///
    /// Different from [`Set::insert`] exactly when the two are equal but distinguishable -- a
    /// `String` element replacing an equal one held in a different allocation, say.
    pub fn replace(&mut self, k: K) -> Option<K> {
        let old = self.0.remove_entry(&k).map(|(k, ())| k);
        self.0.insert(k, ());
        old
    }

    /// Whether `k` is present.
    #[must_use]
    pub fn contains(&self, k: &K) -> bool {
        self.0.contains_key(k)
    }

    /// The stored element equal to `k`.
    #[must_use]
    pub fn get(&self, k: &K) -> Option<H::Ref<'_>> {
        self.0.get_key_value(k).map(|(k, ())| k)
    }

    /// Remove `k`. True when it was there.
    pub fn remove(&mut self, k: &K) -> bool {
        self.0.remove(k).is_some()
    }

    /// Remove `k` and hand back the stored element.
    pub fn take(&mut self, k: &K) -> Option<K> {
        self.0.remove_entry(k).map(|(k, ())| k)
    }

    /// The elements, in address order.
    #[must_use]
    pub fn iter(&self) -> Keys<'_, K, (), H> {
        self.0.keys()
    }

    /// Take every element out, keeping the allocation.
    pub fn drain(&mut self) -> SetDrain<'_, K, H> {
        SetDrain(self.0.drain())
    }

    /// Keep the elements `keep` accepts.
    pub fn retain(&mut self, mut keep: impl FnMut(H::Ref<'_>) -> bool) {
        self.0.retain(|k, ()| keep(k));
    }

    /// Remove and yield the elements `pred` accepts, leaving the rest.
    ///
    /// The std shape: `HashSet::extract_if`. Elements are removed as the iterator yields them, and
    /// dropping it early leaves the untested elements in place -- the same contract [`Map`] has.
    ///
    /// ```
    /// use apeiron::set::Set;
    ///
    /// let mut s: Set<u64> = (0u64..10).collect();
    /// let mut even: Vec<u64> = s.extract_if(|k| k % 2 == 0).collect();
    /// even.sort_unstable();
    /// assert_eq!(even, [0, 2, 4, 6, 8]);
    /// assert_eq!(s.len(), 5);
    /// ```
    pub fn extract_if<'a, F: FnMut(H::Ref<'_>) -> bool + 'a>(
        &'a mut self,
        mut pred: F,
    ) -> impl Iterator<Item = K> + 'a {
        self.0.extract_if(move |k, ()| pred(k)).map(|(k, ())| k)
    }

    /// Drop every element, keeping the allocation.
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Make room for `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    /// [`Set::reserve`], reporting an allocation failure rather than aborting on it.
    ///
    /// # Errors
    ///
    /// [`crate::map::AllocFailed`] when the region cannot be allocated.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), crate::map::AllocFailed> {
        self.0.try_reserve(additional)
    }

    /// Give back every slot the elements do not need.
    pub fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit();
    }

    /// Shrink, keeping room for at least `min`.
    pub fn shrink_to(&mut self, min: usize) {
        self.0.shrink_to(min);
    }

    // -- set algebra -----------------------------------------------------------------------------
    //
    // These return sets, where `std` returns lazy iterators. The reason is the faces: `std`'s
    // `intersection` yields `&T` because there is exactly one thing an element can be on the way
    // out, and here there are two -- a borrow on the parabolic face, a value rebuilt from the
    // address on the elliptic one. A lazy iterator would have to name one and so would work on only
    // one face. The predicates below need no such choice, and stay lazy and allocation-free.

    /// Everything in either.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self
    where
        K: Clone,
    {
        let mut out = Self::with_salt(self.0.salt());
        out.reserve(self.len() + other.len());
        self.0.for_each_key(|k| {
            out.insert(k.clone());
        });
        other.0.for_each_key(|k| {
            out.insert(k.clone());
        });
        out
    }

    /// Everything in both.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self
    where
        K: Clone,
    {
        // Walk the smaller: `contains` costs the same either way, so the loop length is the cost.
        let (small, big) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut out = Self::with_salt(self.0.salt());
        small.0.for_each_key(|k| {
            if big.contains(k) {
                out.insert(k.clone());
            }
        });
        out
    }

    /// Everything in `self` that is not in `other`.
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self
    where
        K: Clone,
    {
        let mut out = Self::with_salt(self.0.salt());
        self.0.for_each_key(|k| {
            if !other.contains(k) {
                out.insert(k.clone());
            }
        });
        out
    }

    /// Everything in exactly one of them.
    #[must_use]
    pub fn symmetric_difference(&self, other: &Self) -> Self
    where
        K: Clone,
    {
        let mut out = self.difference(other);
        other.0.for_each_key(|k| {
            if !self.contains(k) {
                out.insert(k.clone());
            }
        });
        out
    }

    /// Whether the two share nothing.
    #[must_use]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        let (small, big) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        small.0.all_keys(|k| !big.contains(k))
    }

    /// Whether every element of `self` is in `other`.
    #[must_use]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.len() <= other.len() && self.0.all_keys(|k| other.contains(k))
    }

    /// Whether every element of `other` is in `self`.
    #[must_use]
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }
}

impl<K: Coord + Eq + Clone, H: Held<K>> std::ops::BitOr<&Set<K, H>> for &Set<K, H> {
    type Output = Set<K, H>;
    fn bitor(self, rhs: &Set<K, H>) -> Set<K, H> {
        self.union(rhs)
    }
}

impl<K: Coord + Eq + Clone, H: Held<K>> std::ops::BitAnd<&Set<K, H>> for &Set<K, H> {
    type Output = Set<K, H>;
    fn bitand(self, rhs: &Set<K, H>) -> Set<K, H> {
        self.intersection(rhs)
    }
}

impl<K: Coord + Eq + Clone, H: Held<K>> std::ops::BitXor<&Set<K, H>> for &Set<K, H> {
    type Output = Set<K, H>;
    fn bitxor(self, rhs: &Set<K, H>) -> Set<K, H> {
        self.symmetric_difference(rhs)
    }
}

impl<K: Coord + Eq + Clone, H: Held<K>> std::ops::Sub<&Set<K, H>> for &Set<K, H> {
    type Output = Set<K, H>;
    fn sub(self, rhs: &Set<K, H>) -> Set<K, H> {
        self.difference(rhs)
    }
}

/// [`Set::drain`].
pub struct SetDrain<'a, K: Coord + Eq, H: Held<K>>(Drain<'a, K, (), H>);

impl<K: Coord + Eq, H: Held<K>> Iterator for SetDrain<'_, K, H> {
    type Item = K;
    fn next(&mut self) -> Option<K> {
        self.0.next().map(|(k, ())| k)
    }
}

/// [`Set::into_iter`].
pub struct SetIntoIter<K: Coord + Eq, H: Held<K>>(IntoKeys<K, (), H>);

impl<K: Coord + Eq, H: Held<K>> Iterator for SetIntoIter<K, H> {
    type Item = K;
    fn next(&mut self) -> Option<K> {
        self.0.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<K: Coord + Eq, H: Held<K>> ExactSizeIterator for SetIntoIter<K, H> {}

impl<K: Coord + Eq, H: Held<K>> IntoIterator for Set<K, H> {
    type Item = K;
    type IntoIter = SetIntoIter<K, H>;
    fn into_iter(self) -> Self::IntoIter {
        SetIntoIter(self.0.into_keys())
    }
}

impl<'a, K: Coord + Eq, H: Held<K>> IntoIterator for &'a Set<K, H> {
    type Item = H::Ref<'a>;
    type IntoIter = Keys<'a, K, (), H>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// -- the borrowed-element surface, parabolic face only ----------------------------------------------

impl<K: Coord + Eq> Set<K, Stored<K>> {
    /// `set.contains_at("resref")` on a `Set<String>`, allocating nothing.
    #[must_use]
    pub fn contains_at<Q: Locate<K>>(&self, q: Q) -> bool {
        self.0.contains_at(q)
    }

    /// The stored element a view names, without building one to look it up.
    ///
    /// `contains_at` answers whether; this hands back the element itself, so a `Set<String>`
    /// probed with `&str` yields the `&String` it is holding.
    #[must_use]
    pub fn get_at<Q: Locate<K>>(&self, q: Q) -> Option<&K> {
        self.0.get_key_value_at(q).map(|(k, ())| k)
    }

    /// Add through a view, taking ownership only on a miss.
    pub fn insert_at<Q: Absorb<K>>(&mut self, q: Q) -> bool {
        self.0.insert_at(q, ()).is_none()
    }

    /// Remove through a view.
    pub fn remove_at<Q: Locate<K>>(&mut self, q: Q) -> bool {
        self.0.remove_at(q).is_some()
    }
}

// -- a set that cannot exist unpeppered --------------------------------------------------------------

/// [`Set`] with a mandatory pepper. The set counterpart of [`crate::map::Exposed`].
pub struct ExposedSet<K, H = Stored<K>>(Set<K, H>);

impl<K: Coord + Eq, H: Held<K>> ExposedSet<K, H> {
    /// An empty set with a pepper that cannot be zero.
    #[must_use]
    pub fn new(pepper: NonZeroU64) -> Self {
        Self(Set::with_pepper(pepper.get()))
    }

    /// Take the process pepper, if the process set one.
    #[must_use]
    pub fn from_process_pepper() -> Option<Self> {
        NonZeroU64::new(crate::pepper::process_pepper()).map(Self::new)
    }

    /// The plain set underneath.
    #[must_use]
    pub fn into_inner(self) -> Set<K, H> {
        self.0
    }
}

impl<K: Coord + Eq, H: Held<K>> std::ops::Deref for ExposedSet<K, H> {
    type Target = Set<K, H>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<K: Coord + Eq, H: Held<K>> std::ops::DerefMut for ExposedSet<K, H> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// -- the rest of the shape ----------------------------------------------------------------------------

impl<K: Coord + Eq, H: Held<K>> FromIterator<K> for Set<K, H> {
    fn from_iter<T: IntoIterator<Item = K>>(it: T) -> Self {
        let it = it.into_iter();
        let mut s = Self::with_capacity(it.size_hint().0);
        for k in it {
            s.insert(k);
        }
        s
    }
}

impl<K: Coord + Eq, H: Held<K>> Extend<K> for Set<K, H> {
    /// Reserves from the iterator's lower bound, then inserts. See `Map`'s impl for the cost of
    /// not doing so.
    fn extend<T: IntoIterator<Item = K>>(&mut self, it: T) {
        let it = it.into_iter();
        let hint = it.size_hint().0;
        self.reserve(if self.is_empty() { hint } else { hint / 2 });
        for k in it {
            self.insert(k);
        }
    }
}

impl<K: Coord + Eq, H: Held<K>, const N: usize> From<[K; N]> for Set<K, H> {
    fn from(items: [K; N]) -> Self {
        items.into_iter().collect()
    }
}

impl<K: Coord + Eq, H: Held<K> + Clone> Clone for Set<K, H> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<K: Coord + Eq, H: Held<K>> PartialEq for Set<K, H> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K: Coord + Eq, H: Held<K>> Eq for Set<K, H> {}

impl<K: Coord + Eq, H: Held<K>> std::fmt::Debug for Set<K, H>
where
    for<'a> H::Ref<'a>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}
