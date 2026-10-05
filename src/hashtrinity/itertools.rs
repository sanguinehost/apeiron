//! `itertools` masquerade — the `Itertools` trait extending every `Iterator`, plus its supporting
//! types (`Either`, `EitherOrBoth`, `Position`, `MinMaxResult`, the error types) and free
//! functions/macros. Zero dependencies — this REPLACES `itertools`, it doesn't wrap it.
//!
//! ```
//! use hashtrinity::itertools::Itertools;
//!
//! assert_eq!(vec![1, 1, 2, 3, 3, 3].into_iter().unique().collect::<Vec<_>>(), vec![1, 2, 3]);
//! assert_eq!(vec![1, 2, 2, 3].into_iter().dedup().collect::<Vec<_>>(), vec![1, 2, 3]);
//! assert_eq!([1, 2, 3].iter().join(", "), "1, 2, 3");
//! ```
//!
//! ## The one operation underneath — grounded in `HashTrinity.lean`'s B/U merge
//!
//! `itertools`' ~150 methods look like a zoo; they aren't. Every one of them is one of three
//! things (the same trinity `proof/Substrate/Algebra/Security/HashTrinityCore.lean` proves for the hashmap:
//! read there, not repeated here — this file doesn't touch it):
//!
//! - **NB (absence/floor):** `unique`/`dedup`/`duplicates`/`positions`/`filter`-shaped methods —
//!   a residual test against "have I seen this," yielding nothing when the answer is yes.
//! - **B (the element, unchanged):** `map_into`/`map_ok`/`update`-shaped methods — a pass-through
//!   with a value transform, no relation between elements at all.
//! - **B/U (the merge — the medium, the only thing that changes):** `merge`/`kmerge`/`zip_longest`/
//!   `interleave`/`chunk_by`/`group_by`/`into_group_map`/`counts` — literally the SAME `insertWith`/
//!   fold-a-relation-between-two-values shape `HashTrinity.lean`'s accumulator monoid proves for
//!   the hashmap (there: `fuse`, information-form Kalman fusion; here: combine two adjacent/keyed
//!   stream elements). `counts`/`into_group_map` in particular are literally [`crate::FoldExt`]'s
//!   `count_by`/`group_by` wearing `HashMap`'s API instead of `Table`'s — same operation, the real
//!   crate's return type.
//!
//! Combinatorial generation (`combinations`/`permutations`/`powerset`/`cartesian_product`) is the
//! same B/U merge iterated to build the RELATION itself (a Cartesian/subset structure) rather
//! than fold two streams into one — still the medium face, one level up.
//!
//! ## What's ported and what's dropped
//!
//! The full `Itertools` trait is implemented for every method with well-defined, stable-since-1.0
//! semantics. Deliberately not built: the `_with_hasher` parameterized-hasher variants (this crate
//! has one hasher story throughout, not a pluggable one), `RcIter`/`rciter` (shared-iterator-via-
//! `Rc<RefCell<_>>`, a niche interior-mutability pattern), `PeekNth`/`MultiPeek`/`Tee` in their
//! full generality (this file's `multipeek`/`tee` cover the common 2-ahead/2-way-clone cases), and
//! the free-standing `diff`/`assert_equal` test-helper functions. Named, not silently dropped —
//! see `MASQUERADE.md`.

// The trait itself is the other half of this module, in `methods.rs`. It is a PRIVATE child,
// glob-re-exported, so `hashtrinity::itertools::Itertools` is the same path it always was and no
// new one appears. It has to be a CHILD rather than a sibling: the ~150 methods build their
// adaptors with struct literals over PRIVATE fields (`Unique { iter, seen, key }`, `HeadState::new`,
// `Combinations::new`, `ClonedElement(element)`, `ExactlyOneError { extra }`), and a private field
// is reachable from the defining module and its DESCENDANTS only. A sibling module would force
// `pub(crate)` on ~90 fields and five constructors — a rewrite, not a move.
mod methods {
    //! The `Itertools` trait — see the module doc in `mod.rs`. A child of `itertools`, not a sibling,
    //! so the adaptor constructors below reach their parent's private struct fields unchanged.

    use super::*;
    use std::cmp::Ordering;
    use std::collections::HashMap;

    // ═══════════════════════════════════════════════════════════════════════════
    // The `Itertools` trait — blanket-implemented for every `Iterator`.
    // ═══════════════════════════════════════════════════════════════════════════

    /// The real crate's extension trait — every method below is available on any `Iterator` once
    /// this trait is in scope (`use hashtrinity::itertools::Itertools;`).
    pub trait Itertools: Iterator {
        // ── sorted* : eager, collect-then-sort ──────────────────────────────────────────────────
        fn sorted(self) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
            Self::Item: Ord,
        {
            let mut v: Vec<_> = self.collect();
            v.sort();
            v.into_iter()
        }
        fn sorted_unstable(self) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
            Self::Item: Ord,
        {
            let mut v: Vec<_> = self.collect();
            v.sort_unstable();
            v.into_iter()
        }
        fn sorted_by(
            self,
            mut cmp: impl FnMut(&Self::Item, &Self::Item) -> Ordering,
        ) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
        {
            let mut v: Vec<_> = self.collect();
            v.sort_by(&mut cmp);
            v.into_iter()
        }
        fn sorted_unstable_by(
            self,
            mut cmp: impl FnMut(&Self::Item, &Self::Item) -> Ordering,
        ) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
        {
            let mut v: Vec<_> = self.collect();
            v.sort_unstable_by(&mut cmp);
            v.into_iter()
        }
        fn sorted_by_key<K: Ord>(
            self,
            mut key: impl FnMut(&Self::Item) -> K,
        ) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
        {
            let mut v: Vec<_> = self.collect();
            v.sort_by_key(&mut key);
            v.into_iter()
        }
        fn sorted_unstable_by_key<K: Ord>(
            self,
            mut key: impl FnMut(&Self::Item) -> K,
        ) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
        {
            let mut v: Vec<_> = self.collect();
            v.sort_unstable_by_key(&mut key);
            v.into_iter()
        }
        /// Sorts by a key computed once per element up front (a `Vec<(K, T)>` decorate-sort-undecorate),
        /// worthwhile when `key` is expensive — matches the real crate's own rationale exactly.
        fn sorted_by_cached_key<K: Ord>(
            self,
            mut key: impl FnMut(&Self::Item) -> K,
        ) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
        {
            let mut v: Vec<_> = self.collect();
            v.sort_by_cached_key(&mut key);
            v.into_iter()
        }

        // ── counts / grouping : the SAME shape as crate::FoldExt, HashMap instead of Table ──────
        /// Count occurrences per value. The same operation as [`crate::FoldExt::count_by`] with the
        /// identity key and a real `HashMap` return type (the real crate's own signature).
        fn counts(self) -> HashMap<Self::Item, usize>
        where
            Self: Sized,
            Self::Item: Eq + std::hash::Hash,
        {
            let mut m = HashMap::new();
            for x in self {
                *m.entry(x).or_insert(0) += 1;
            }
            m
        }
        fn counts_by<K: Eq + std::hash::Hash>(
            self,
            mut key: impl FnMut(Self::Item) -> K,
        ) -> HashMap<K, usize>
        where
            Self: Sized,
        {
            let mut m = HashMap::new();
            for x in self {
                *m.entry(key(x)).or_insert(0) += 1;
            }
            m
        }
        /// Group values by key, insertion order preserved per group — [`crate::FoldExt::group_by`]'s
        /// operation, `HashMap<K, Vec<Item>>` instead of `Table<K, Vec<V>>`.
        fn into_group_map<K, V>(self) -> HashMap<K, Vec<V>>
        where
            Self: Sized + Iterator<Item = (K, V)>,
            K: Eq + std::hash::Hash,
        {
            let mut m: HashMap<K, Vec<V>> = HashMap::new();
            for (k, v) in self {
                m.entry(k).or_default().push(v);
            }
            m
        }
        fn into_group_map_by<K, V>(self, mut key: impl FnMut(&V) -> K) -> HashMap<K, Vec<V>>
        where
            Self: Sized + Iterator<Item = V>,
            K: Eq + std::hash::Hash,
        {
            let mut m: HashMap<K, Vec<V>> = HashMap::new();
            for v in self {
                let k = key(&v);
                m.entry(k).or_default().push(v);
            }
            m
        }

        // ── all_equal / all_unique / exactly_one / at_most_one ─────────────────────────────────
        fn all_equal(&mut self) -> bool
        where
            Self: Sized,
            Self::Item: PartialEq,
        {
            match self.next() {
                None => true,
                Some(first) => self.all(|x| x == first),
            }
        }
        fn all_equal_value(&mut self) -> Result<Self::Item, AllEqualValueError<Self::Item>>
        where
            Self: Sized,
            Self::Item: PartialEq,
        {
            let first = self.next().ok_or(AllEqualValueError(None))?;
            match self.find(|x| x != &first) {
                Some(other) => Err(AllEqualValueError(Some([first, other]))),
                None => Ok(first),
            }
        }
        fn all_unique(&mut self) -> bool
        where
            Self: Sized,
            Self::Item: Eq + std::hash::Hash,
        {
            let mut seen = std::collections::HashSet::new();
            self.all(|x| seen.insert(x))
        }
        /// Exactly one element, or an error carrying back everything found (mirrors the real crate).
        fn exactly_one(mut self) -> Result<Self::Item, ExactlyOneError<Self::Item>>
        where
            Self: Sized,
        {
            match self.next() {
                None => Err(ExactlyOneError { extra: Vec::new() }),
                Some(first) => match self.next() {
                    None => Ok(first),
                    Some(second) => {
                        let mut extra = vec![first, second];
                        extra.extend(self);
                        Err(ExactlyOneError { extra })
                    }
                },
            }
        }
        fn at_most_one(mut self) -> Result<Option<Self::Item>, ExactlyOneError<Self::Item>>
        where
            Self: Sized,
        {
            match self.next() {
                None => Ok(None),
                Some(first) => match self.next() {
                    None => Ok(Some(first)),
                    Some(second) => {
                        let mut extra = vec![first, second];
                        extra.extend(self);
                        Err(ExactlyOneError { extra })
                    }
                },
            }
        }

        // ── minmax family ────────────────────────────────────────────────────────────────────────
        fn minmax(self) -> MinMaxResult<Self::Item>
        where
            Self: Sized,
            Self::Item: PartialOrd,
        {
            self.minmax_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        }
        fn minmax_by_key<K: PartialOrd>(
            self,
            mut key: impl FnMut(&Self::Item) -> K,
        ) -> MinMaxResult<Self::Item>
        where
            Self: Sized,
        {
            self.minmax_by(move |a, b| key(a).partial_cmp(&key(b)).unwrap_or(Ordering::Equal))
        }
        fn minmax_by(
            self,
            mut cmp: impl FnMut(&Self::Item, &Self::Item) -> Ordering,
        ) -> MinMaxResult<Self::Item>
        where
            Self: Sized,
        {
            let mut it = self.into_iter();
            let (mut lo, mut hi) = match it.next() {
                None => return MinMaxResult::NoElements,
                Some(x) => match it.next() {
                    None => return MinMaxResult::OneElement(x),
                    Some(y) => {
                        if cmp(&x, &y) == Ordering::Greater {
                            (y, x)
                        } else {
                            (x, y)
                        }
                    }
                },
            };
            loop {
                let Some(a) = it.next() else { break };
                let Some(b) = it.next() else {
                    if cmp(&a, &lo) == Ordering::Less {
                        lo = a;
                    } else if cmp(&a, &hi) != Ordering::Less {
                        hi = a;
                    }
                    break;
                };
                let (small, large) = if cmp(&a, &b) == Ordering::Greater {
                    (b, a)
                } else {
                    (a, b)
                };
                if cmp(&small, &lo) == Ordering::Less {
                    lo = small;
                }
                if cmp(&large, &hi) != Ordering::Less {
                    hi = large;
                }
            }
            MinMaxResult::MinMax(lo, hi)
        }
        fn min_set(self) -> Vec<Self::Item>
        where
            Self: Sized,
            Self::Item: Ord,
        {
            self.min_set_by(Ord::cmp)
        }
        fn min_set_by_key<K: Ord>(self, mut key: impl FnMut(&Self::Item) -> K) -> Vec<Self::Item>
        where
            Self: Sized,
        {
            self.min_set_by(move |a, b| key(a).cmp(&key(b)))
        }
        fn min_set_by(
            self,
            mut cmp: impl FnMut(&Self::Item, &Self::Item) -> Ordering,
        ) -> Vec<Self::Item>
        where
            Self: Sized,
        {
            extreme_set(self, move |a, b| cmp(a, b))
        }
        fn max_set(self) -> Vec<Self::Item>
        where
            Self: Sized,
            Self::Item: Ord,
        {
            self.max_set_by(Ord::cmp)
        }
        fn max_set_by_key<K: Ord>(self, mut key: impl FnMut(&Self::Item) -> K) -> Vec<Self::Item>
        where
            Self: Sized,
        {
            self.max_set_by(move |a, b| key(a).cmp(&key(b)))
        }
        fn max_set_by(
            self,
            mut cmp: impl FnMut(&Self::Item, &Self::Item) -> Ordering,
        ) -> Vec<Self::Item>
        where
            Self: Sized,
        {
            extreme_set(self, move |a, b| cmp(b, a))
        }

        // ── k_smallest / k_largest family (heap-free: sort-and-slice; correct, not the real
        //    crate's streaming-heap performance profile — see the module doc's gap note) ─────────
        fn k_smallest(self, k: usize) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
            Self::Item: Ord,
        {
            self.k_smallest_by(k, Ord::cmp)
        }
        fn k_smallest_by_key<K: Ord>(
            self,
            k: usize,
            mut key: impl FnMut(&Self::Item) -> K,
        ) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
        {
            self.k_smallest_by(k, move |a, b| key(a).cmp(&key(b)))
        }
        fn k_smallest_by(
            self,
            k: usize,
            mut cmp: impl FnMut(&Self::Item, &Self::Item) -> Ordering,
        ) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
        {
            let mut v: Vec<_> = self.collect();
            v.sort_by(&mut cmp);
            v.truncate(k);
            v.into_iter()
        }
        fn k_smallest_relaxed(self, k: usize) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
            Self::Item: Ord,
        {
            self.k_smallest(k)
        }
        fn k_smallest_relaxed_by<F>(self, k: usize, cmp: F) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
            F: FnMut(&Self::Item, &Self::Item) -> Ordering,
        {
            self.k_smallest_by(k, cmp)
        }
        fn k_smallest_relaxed_by_key<K: Ord>(
            self,
            k: usize,
            key: impl FnMut(&Self::Item) -> K,
        ) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
        {
            self.k_smallest_by_key(k, key)
        }
        fn k_largest(self, k: usize) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
            Self::Item: Ord,
        {
            self.k_largest_by(k, Ord::cmp)
        }
        fn k_largest_by_key<K: Ord>(
            self,
            k: usize,
            mut key: impl FnMut(&Self::Item) -> K,
        ) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
        {
            self.k_largest_by(k, move |a, b| key(a).cmp(&key(b)))
        }
        fn k_largest_by(
            self,
            k: usize,
            mut cmp: impl FnMut(&Self::Item, &Self::Item) -> Ordering,
        ) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
        {
            let mut v: Vec<_> = self.collect();
            v.sort_by(|a, b| cmp(b, a));
            v.truncate(k);
            v.into_iter()
        }
        fn k_largest_relaxed(self, k: usize) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
            Self::Item: Ord,
        {
            self.k_largest(k)
        }
        fn k_largest_relaxed_by<F>(self, k: usize, cmp: F) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
            F: FnMut(&Self::Item, &Self::Item) -> Ordering,
        {
            self.k_largest_by(k, cmp)
        }
        fn k_largest_relaxed_by_key<K: Ord>(
            self,
            k: usize,
            key: impl FnMut(&Self::Item) -> K,
        ) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
        {
            self.k_largest_by_key(k, key)
        }

        // ── position_min / position_max / position_minmax family ───────────────────────────────
        fn position_min(self) -> Option<usize>
        where
            Self: Sized,
            Self::Item: PartialOrd,
        {
            self.position_min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        }
        fn position_min_by_key<K: PartialOrd>(
            self,
            mut key: impl FnMut(&Self::Item) -> K,
        ) -> Option<usize>
        where
            Self: Sized,
        {
            self.position_min_by(move |a, b| key(a).partial_cmp(&key(b)).unwrap_or(Ordering::Equal))
        }
        fn position_min_by(
            self,
            mut cmp: impl FnMut(&Self::Item, &Self::Item) -> Ordering,
        ) -> Option<usize>
        where
            Self: Sized,
        {
            let mut best: Option<(usize, Self::Item)> = None;
            for (i, x) in self.into_iter().enumerate() {
                match &best {
                    None => best = Some((i, x)),
                    Some((_, b)) => {
                        if cmp(&x, b) == Ordering::Less {
                            best = Some((i, x));
                        }
                    }
                }
            }
            best.map(|(i, _)| i)
        }
        fn position_max(self) -> Option<usize>
        where
            Self: Sized,
            Self::Item: PartialOrd,
        {
            self.position_max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        }
        fn position_max_by_key<K: PartialOrd>(
            self,
            mut key: impl FnMut(&Self::Item) -> K,
        ) -> Option<usize>
        where
            Self: Sized,
        {
            self.position_max_by(move |a, b| key(a).partial_cmp(&key(b)).unwrap_or(Ordering::Equal))
        }
        fn position_max_by(
            self,
            mut cmp: impl FnMut(&Self::Item, &Self::Item) -> Ordering,
        ) -> Option<usize>
        where
            Self: Sized,
        {
            let mut best: Option<(usize, Self::Item)> = None;
            for (i, x) in self.into_iter().enumerate() {
                match &best {
                    None => best = Some((i, x)),
                    Some((_, b)) => {
                        if cmp(&x, b) != Ordering::Less {
                            best = Some((i, x));
                        }
                    }
                }
            }
            best.map(|(i, _)| i)
        }
        fn position_minmax(self) -> MinMaxResult<usize>
        where
            Self: Sized,
            Self::Item: PartialOrd,
        {
            self.position_minmax_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        }
        fn position_minmax_by_key<K: PartialOrd>(
            self,
            mut key: impl FnMut(&Self::Item) -> K,
        ) -> MinMaxResult<usize>
        where
            Self: Sized,
        {
            self.position_minmax_by(move |a, b| key(a).partial_cmp(&key(b)).unwrap_or(Ordering::Equal))
        }
        fn position_minmax_by(
            self,
            mut cmp: impl FnMut(&Self::Item, &Self::Item) -> Ordering,
        ) -> MinMaxResult<usize>
        where
            Self: Sized,
        {
            let items: Vec<Self::Item> = self.into_iter().collect();
            if items.is_empty() {
                return MinMaxResult::NoElements;
            }
            if items.len() == 1 {
                return MinMaxResult::OneElement(0);
            }
            let mut lo = 0usize;
            let mut hi = 0usize;
            for (i, x) in items.iter().enumerate() {
                if cmp(x, &items[lo]) == Ordering::Less {
                    lo = i;
                }
                if cmp(x, &items[hi]) != Ordering::Less {
                    hi = i;
                }
            }
            MinMaxResult::MinMax(lo, hi)
        }

        // ── partition_map / partition_result ─────────────────────────────────────────────────────
        fn partition_map<A, B, F: FnMut(Self::Item) -> Either<A, B>>(self, mut f: F) -> (Vec<A>, Vec<B>)
        where
            Self: Sized,
        {
            let mut left = Vec::new();
            let mut right = Vec::new();
            for x in self {
                match f(x) {
                    Either::Left(a) => left.push(a),
                    Either::Right(b) => right.push(b),
                }
            }
            (left, right)
        }
        fn partition_result<A, B>(self) -> (Vec<A>, Vec<B>)
        where
            Self: Sized + Iterator<Item = Result<A, B>>,
        {
            let mut ok = Vec::new();
            let mut err = Vec::new();
            for x in self {
                match x {
                    Ok(a) => ok.push(a),
                    Err(b) => err.push(b),
                }
            }
            (ok, err)
        }

        // ── join / format / format_with / concat / collect_vec / collect_array / collect_tuple ──
        fn join(&mut self, sep: &str) -> String
        where
            Self: Sized,
            Self::Item: fmt::Display,
        {
            match self.next() {
                None => String::new(),
                Some(first) => {
                    let mut out = first.to_string();
                    for x in self {
                        out.push_str(sep);
                        out.push_str(&x.to_string());
                    }
                    out
                }
            }
        }
        fn concat(self) -> Self::Item
        where
            Self: Sized,
            Self::Item: Extend<<Self::Item as IntoIterator>::Item> + IntoIterator + Default,
        {
            let mut it = self.into_iter();
            let Some(mut base) = it.next() else {
                return Self::Item::default();
            };
            for x in it {
                base.extend(x);
            }
            base
        }
        fn collect_vec(self) -> Vec<Self::Item>
        where
            Self: Sized,
        {
            self.collect()
        }

        // ── contains / find_or_first / find_or_last / find_position ────────────────────────────
        fn contains<Q>(&mut self, query: &Q) -> bool
        where
            Self: Sized,
            Self::Item: PartialEq<Q>,
        {
            self.any(|x| x == *query)
        }
        fn find_or_first(&mut self, mut pred: impl FnMut(&Self::Item) -> bool) -> Option<Self::Item>
        where
            Self: Sized,
        {
            let first = self.next()?;
            if pred(&first) {
                return Some(first);
            }
            Some(self.find(pred).unwrap_or(first))
        }
        fn find_or_last(&mut self, mut pred: impl FnMut(&Self::Item) -> bool) -> Option<Self::Item>
        where
            Self: Sized,
        {
            let mut last = self.next()?;
            if pred(&last) {
                return Some(last);
            }
            for x in self {
                if pred(&x) {
                    return Some(x);
                }
                last = x;
            }
            Some(last)
        }
        fn find_position(
            &mut self,
            mut pred: impl FnMut(&Self::Item) -> bool,
        ) -> Option<(usize, Self::Item)>
        where
            Self: Sized,
        {
            for (i, x) in self.enumerate() {
                if pred(&x) {
                    return Some((i, x));
                }
            }
            None
        }

        // ── dropping / dropping_back / tail ──────────────────────────────────────────────────────
        #[must_use]
        fn dropping(mut self, n: usize) -> Self
        where
            Self: Sized,
        {
            if n > 0 {
                self.nth(n - 1);
            }
            self
        }
        fn dropping_back(self, n: usize) -> std::vec::IntoIter<Self::Item>
        where
            Self: Sized,
        {
            let mut v: Vec<_> = self.collect();
            v.truncate(v.len().saturating_sub(n));
            v.into_iter()
        }
        fn tail(self, n: usize) -> std::collections::vec_deque::IntoIter<Self::Item>
        where
            Self: Sized,
        {
            let mut buf: std::collections::VecDeque<Self::Item> =
                std::collections::VecDeque::with_capacity(n);
            for x in self {
                if buf.len() == n {
                    buf.pop_front();
                }
                buf.push_back(x);
            }
            buf.into_iter()
        }

        // ── map_into / update ────────────────────────────────────────────────────────────────────
        fn map_into<R>(self) -> std::iter::Map<Self, fn(Self::Item) -> R>
        where
            Self: Sized,
            Self::Item: Into<R>,
        {
            fn conv<T: Into<R>, R>(t: T) -> R {
                t.into()
            }
            self.map(conv::<Self::Item, R>)
        }
        fn update<F>(self, f: F) -> Update<Self, F>
        where
            Self: Sized,
            F: FnMut(&mut Self::Item),
        {
            Update { iter: self, f }
        }

        // ── multiunzip ────────────────────────────────────────────────────────────────────────────
        fn multiunzip<FromI>(self) -> FromI
        where
            Self: Sized + MultiUnzip<FromI>,
        {
            MultiUnzip::multiunzip(self)
        }

        // ── chunk_by / group_by (CONSECUTIVE grouping — distinct from into_group_map, which groups
        //    by key anywhere in the stream. Simplified vs. the real crate's lazy shared-buffer
        //    `GroupBy`/`Group` pair: eagerly collects each run into a `Vec` — see the module doc) ──
        fn chunk_by<K: PartialEq, F: FnMut(&Self::Item) -> K>(self, key: F) -> ChunkBy<Self, K, F>
        where
            Self: Sized,
        {
            ChunkBy {
                iter: self,
                key,
                pending: None,
            }
        }
        /// Deprecated alias in the real crate for [`Itertools::chunk_by`] — same operation, kept for
        /// callers still on the old name.
        fn group_by<K: PartialEq, F: FnMut(&Self::Item) -> K>(self, key: F) -> ChunkBy<Self, K, F>
        where
            Self: Sized,
        {
            self.chunk_by(key)
        }

        // ── into_grouping_map / into_grouping_map_by — lazy chunk_by's cousin: eager key->Vec groups
        //    plus a small aggregate API (sum/max/min/fold), the B/U merge with the fold built in ────
        fn into_grouping_map<K, V>(self) -> GroupingMap<K, V>
        where
            Self: Sized + Iterator<Item = (K, V)>,
            K: Eq + std::hash::Hash,
        {
            GroupingMap {
                groups: self.into_group_map(),
            }
        }
        fn into_grouping_map_by<K, F>(self, key: F) -> GroupingMap<K, Self::Item>
        where
            Self: Sized,
            K: Eq + std::hash::Hash,
            F: FnMut(&Self::Item) -> K,
        {
            GroupingMap {
                groups: self.into_group_map_by(key),
            }
        }

        // ── fold_while / fold1 / tree_fold1 / tree_reduce ────────────────────────────────────────
        /// `fold` that can stop early — `f` returns `FoldWhile::Continue`/`Done`.
        fn fold_while<B, F>(self, init: B, mut f: F) -> FoldWhile<B>
        where
            Self: Sized,
            F: FnMut(B, Self::Item) -> FoldWhile<B>,
        {
            let mut acc = init;
            for x in self {
                match f(acc, x) {
                    FoldWhile::Continue(b) => acc = b,
                    FoldWhile::Done(b) => return FoldWhile::Done(b),
                }
            }
            FoldWhile::Continue(acc)
        }
        /// Deprecated alias in the real crate for `Iterator::reduce`.
        fn fold1<F: FnMut(Self::Item, Self::Item) -> Self::Item>(self, f: F) -> Option<Self::Item>
        where
            Self: Sized,
        {
            self.reduce(f)
        }
        /// Pairwise-tree reduction (combines adjacent pairs, then pairs of results, ...) instead of
        /// strictly left-to-right — the same total when `f` is associative, fewer sequential
        /// dependencies (useful for e.g. float summation error or when `f` parallelizes).
        fn tree_reduce<F>(self, mut f: F) -> Option<Self::Item>
        where
            Self: Sized,
            F: FnMut(Self::Item, Self::Item) -> Self::Item,
        {
            let mut v: Vec<Self::Item> = self.collect();
            if v.is_empty() {
                return None;
            }
            while v.len() > 1 {
                let mut next = Vec::with_capacity(v.len().div_ceil(2));
                let mut it = v.into_iter();
                while let Some(a) = it.next() {
                    match it.next() {
                        Some(b) => next.push(f(a, b)),
                        None => next.push(a),
                    }
                }
                v = next;
            }
            v.pop()
        }
        /// Deprecated alias in the real crate for [`Itertools::tree_reduce`].
        fn tree_fold1<F: FnMut(Self::Item, Self::Item) -> Self::Item>(self, f: F) -> Option<Self::Item>
        where
            Self: Sized,
        {
            self.tree_reduce(f)
        }

        // ── process_results / fold_ok / fold_options / filter_ok / filter_map_ok / map_ok /
        //    flatten_ok — thread a `Result`/`Option`-yielding iterator through plain combinators,
        //    short-circuiting on the first `Err`/`None` ───────────────────────────────────────────
        fn process_results<T, E, F, R>(self, f: F) -> Result<R, E>
        where
            Self: Sized + Iterator<Item = Result<T, E>>,
            F: FnOnce(ProcessResults<'_, Self, E>) -> R,
        {
            let mut err = None;
            let wrapped = ProcessResults {
                iter: self,
                err: &mut err,
            };
            let out = f(wrapped);
            match err {
                Some(e) => Err(e),
                None => Ok(out),
            }
        }
        fn fold_ok<A, E, B, F>(&mut self, init: B, mut f: F) -> Result<B, E>
        where
            Self: Sized + Iterator<Item = Result<A, E>>,
            F: FnMut(B, A) -> B,
        {
            let mut acc = init;
            for x in self {
                acc = f(acc, x?);
            }
            Ok(acc)
        }
        fn fold_options<A, B, F>(&mut self, init: B, mut f: F) -> Option<B>
        where
            Self: Sized + Iterator<Item = Option<A>>,
            F: FnMut(B, A) -> B,
        {
            let mut acc = init;
            for x in self {
                acc = f(acc, x?);
            }
            Some(acc)
        }
        fn filter_ok<A, E, F>(self, mut pred: F) -> impl Iterator<Item = Result<A, E>>
        where
            Self: Sized + Iterator<Item = Result<A, E>>,
            F: FnMut(&A) -> bool,
        {
            self.filter(move |x| match x {
                Ok(a) => pred(a),
                Err(_) => true,
            })
        }
        fn filter_map_ok<A, B, E, F>(self, mut f: F) -> impl Iterator<Item = Result<B, E>>
        where
            Self: Sized + Iterator<Item = Result<A, E>>,
            F: FnMut(A) -> Option<B>,
        {
            self.filter_map(move |x| match x {
                Ok(a) => f(a).map(Ok),
                Err(e) => Some(Err(e)),
            })
        }
        fn map_ok<A, B, E, F>(self, mut f: F) -> impl Iterator<Item = Result<B, E>>
        where
            Self: Sized + Iterator<Item = Result<A, E>>,
            F: FnMut(A) -> B,
        {
            self.map(move |x| x.map(&mut f))
        }
        fn flatten_ok<A, E>(self) -> impl Iterator<Item = Result<A::Item, E>>
        where
            Self: Sized + Iterator<Item = Result<A, E>>,
            A: IntoIterator,
        {
            self.flat_map(|x| match x {
                Ok(a) => Either::Left(a.into_iter().map(Ok)),
                Err(e) => Either::Right(std::iter::once(Err(e))),
            })
        }

        // ── take_while_ref / peeking_take_while (peek-based, non-consuming past the boundary) ────
        fn take_while_ref<F>(&mut self, mut pred: F) -> Vec<Self::Item>
        where
            Self: Sized + Clone,
            F: FnMut(&Self::Item) -> bool,
        {
            // Simplified vs. the real crate's zero-copy `&mut Peekable` adaptor: this clones ahead,
            // then only actually advances `self` past what qualified (documented gap: needs `Clone`).
            let mut out = Vec::new();
            let mut probe = self.clone();
            for x in probe.by_ref() {
                if pred(&x) {
                    out.push(x);
                    self.next();
                } else {
                    break;
                }
            }
            out
        }

        // ── product1 / sum1 — reduce via the Product/Sum ops without an identity element ─────────
        fn product1(self) -> Option<Self::Item>
        where
            Self: Sized,
            Self::Item: std::ops::Mul<Output = Self::Item>,
        {
            self.reduce(std::ops::Mul::mul)
        }
        fn sum1(self) -> Option<Self::Item>
        where
            Self: Sized,
            Self::Item: std::ops::Add<Output = Self::Item>,
        {
            self.reduce(std::ops::Add::add)
        }

        // ── collect_tuple / next_tuple / collect_array / next_array ────────────────────────────
        /// Collects exactly 2 items into `(A, A)`, or `None` if there are more or fewer.
        fn collect_tuple(mut self) -> Option<(Self::Item, Self::Item)>
        where
            Self: Sized,
        {
            let a = self.next()?;
            let b = self.next()?;
            if self.next().is_some() {
                return None;
            }
            Some((a, b))
        }
        fn next_tuple(&mut self) -> Option<(Self::Item, Self::Item)>
        where
            Self: Sized,
        {
            let a = self.next()?;
            let b = self.next()?;
            Some((a, b))
        }

        // ── unique / duplicates (lazy: a Table-backed "have I seen this" residual, NB face) ──────
        fn unique(self) -> Unique<Self, Self::Item>
        where
            Self: Sized,
            Self::Item: Clone + Eq + std::hash::Hash,
        {
            Unique {
                iter: self,
                seen: std::collections::HashSet::new(),
                key: Clone::clone,
            }
        }
        fn unique_by<K, F>(self, key: F) -> Unique<Self, K, F>
        where
            Self: Sized,
            K: Eq + std::hash::Hash,
            F: FnMut(&Self::Item) -> K,
        {
            Unique {
                iter: self,
                seen: std::collections::HashSet::new(),
                key,
            }
        }
        fn duplicates(self) -> Duplicates<Self, Self::Item>
        where
            Self: Sized,
            Self::Item: Clone + Eq + std::hash::Hash,
        {
            Duplicates {
                iter: self,
                seen: std::collections::HashMap::new(),
                key: Clone::clone,
            }
        }
        fn duplicates_by<K, F>(self, key: F) -> Duplicates<Self, K, F>
        where
            Self: Sized,
            K: Eq + std::hash::Hash,
            F: FnMut(&Self::Item) -> K,
        {
            Duplicates {
                iter: self,
                seen: std::collections::HashMap::new(),
                key,
            }
        }

        // ── dedup family (lazy: consecutive-equal collapse — the B/U merge specialised to "same as
        //    my immediate predecessor," not "same as any predecessor" like unique) ────────────────
        fn dedup(self) -> Dedup<Self>
        where
            Self: Sized,
            Self::Item: PartialEq,
        {
            Dedup {
                iter: self,
                last: None,
                eq: PartialEq::eq,
            }
        }
        fn dedup_by<F>(self, eq: F) -> DedupPred<Self, F>
        where
            Self: Sized,
            F: FnMut(&Self::Item, &Self::Item) -> bool,
        {
            DedupPred {
                iter: self,
                last: None,
                eq,
            }
        }
        fn dedup_with_count(self) -> DedupWithCount<Self>
        where
            Self: Sized,
            Self::Item: PartialEq,
        {
            DedupWithCount {
                iter: self.peekable(),
                eq: PartialEq::eq,
            }
        }
        fn dedup_by_with_count<F>(self, eq: F) -> DedupWithCountPred<Self, F>
        where
            Self: Sized,
            F: FnMut(&Self::Item, &Self::Item) -> bool,
        {
            DedupWithCountPred {
                iter: self.peekable(),
                eq,
            }
        }

        // ── positions (lazy: NB face — the indices where a predicate is absent-of-failure) ───────
        fn positions<F>(self, pred: F) -> Positions<Self, F>
        where
            Self: Sized,
            F: FnMut(Self::Item) -> bool,
        {
            Positions {
                iter: self,
                pred,
                idx: 0,
            }
        }

        // ── with_position (lazy: tag first/middle/last/only — needs one item of lookahead) ───────
        fn with_position(self) -> WithPosition<Self>
        where
            Self: Sized,
        {
            WithPosition {
                iter: self.peekable(),
                started: false,
            }
        }

        // ── tuple_windows / tuples (fixed-arity grouping — the B/U merge over `n` adjacent items) ─
        fn tuple_windows<T>(self) -> TupleWindows<Self, T>
        where
            Self: Sized,
            T: HomogeneousTuple<Item = Self::Item>,
            Self::Item: Clone,
        {
            TupleWindows {
                iter: self,
                buf: std::collections::VecDeque::new(),
                _marker: std::marker::PhantomData,
            }
        }
        fn tuples<T>(self) -> Tuples<Self, T>
        where
            Self: Sized,
            T: HomogeneousTuple<Item = Self::Item>,
        {
            Tuples {
                iter: self,
                _marker: std::marker::PhantomData,
            }
        }

        // ── chunks (simplified vs. the real crate's lazy shared-buffer `IntoChunks`/`Chunk` — see
        //    the module doc's gap note: each chunk here is an eagerly-collected `Vec`) ────────────
        fn chunks(self, size: usize) -> ChunksIter<Self>
        where
            Self: Sized,
        {
            assert!(size > 0, "chunk size must be non-zero");
            ChunksIter { iter: self, size }
        }

        // ── merge / merge_by / kmerge / kmerge_by (the B/U merge itself, over two or N ordered
        //    streams — same accumulator shape as HashTrinity.lean's `fuse`, one relation at a time) ─
        fn merge<J>(self, other: J) -> Merge<Self, J::IntoIter>
        where
            Self: Sized,
            Self::Item: PartialOrd,
            J: IntoIterator<Item = Self::Item>,
        {
            Merge {
                a: self.peekable(),
                b: other.into_iter().peekable(),
                cmp: |a: &Self::Item, b: &Self::Item| a <= b,
            }
        }
        fn merge_by<J, F>(self, other: J, is_first: F) -> MergeBy<Self, J::IntoIter, F>
        where
            Self: Sized,
            J: IntoIterator<Item = Self::Item>,
            F: FnMut(&Self::Item, &Self::Item) -> bool,
        {
            MergeBy {
                a: self.peekable(),
                b: other.into_iter().peekable(),
                is_first,
            }
        }
        fn merge_join_by<J, F, T>(self, other: J, cmp: F) -> MergeJoinBy<Self, J::IntoIter, F>
        where
            Self: Sized,
            J: IntoIterator,
            F: FnMut(&Self::Item, &J::Item) -> T,
            T: Into<Ordering>,
        {
            MergeJoinBy {
                a: self.peekable(),
                b: other.into_iter().peekable(),
                cmp,
            }
        }
        fn kmerge(self) -> KMerge<<Self::Item as IntoIterator>::IntoIter>
        where
            Self: Sized,
            Self::Item: IntoIterator,
            <Self::Item as IntoIterator>::Item: PartialOrd,
        {
            KMerge {
                heads: self
                    .into_iter()
                    .map(|s| HeadState::new(s.into_iter()))
                    .collect(),
                cmp: |a, b| a <= b,
            }
        }
        fn kmerge_by<F>(self, cmp: F) -> KMergeBy<<Self::Item as IntoIterator>::IntoIter, F>
        where
            Self: Sized,
            Self::Item: IntoIterator,
            F: FnMut(&<Self::Item as IntoIterator>::Item, &<Self::Item as IntoIterator>::Item) -> bool,
        {
            KMergeBy {
                heads: self
                    .into_iter()
                    .map(|s| HeadState::new(s.into_iter()))
                    .collect(),
                cmp,
            }
        }

        // ── interleave / interleave_shortest ─────────────────────────────────────────────────────
        fn interleave<J>(self, other: J) -> Interleave<Self, J::IntoIter>
        where
            Self: Sized,
            J: IntoIterator<Item = Self::Item>,
        {
            Interleave {
                a: self,
                b: other.into_iter(),
                turn_a: true,
                a_done: false,
                b_done: false,
                shortest: false,
            }
        }
        fn interleave_shortest<J>(self, other: J) -> Interleave<Self, J::IntoIter>
        where
            Self: Sized,
            J: IntoIterator<Item = Self::Item>,
        {
            Interleave {
                a: self,
                b: other.into_iter(),
                turn_a: true,
                a_done: false,
                b_done: false,
                shortest: true,
            }
        }

        // ── zip_longest / zip_eq ─────────────────────────────────────────────────────────────────
        fn zip_longest<J>(self, other: J) -> ZipLongest<Self, J::IntoIter>
        where
            Self: Sized,
            J: IntoIterator,
        {
            ZipLongest {
                a: self,
                b: other.into_iter(),
            }
        }
        fn zip_eq<J>(self, other: J) -> ZipEq<Self, J::IntoIter>
        where
            Self: Sized,
            J: IntoIterator,
        {
            ZipEq {
                a: self,
                b: other.into_iter(),
            }
        }

        // ── intersperse / intersperse_with / pad_using ──────────────────────────────────────────
        fn intersperse(self, element: Self::Item) -> IntersperseWith<Self, ClonedElement<Self::Item>>
        where
            Self: Sized,
            Self::Item: Clone,
        {
            IntersperseWith {
                iter: self.peekable(),
                next_is_elem: false,
                make_elem: ClonedElement(element),
            }
        }
        fn intersperse_with<F>(self, make_elem: F) -> IntersperseWith<Self, F>
        where
            Self: Sized,
            F: FnMut() -> Self::Item,
        {
            IntersperseWith {
                iter: self.peekable(),
                next_is_elem: false,
                make_elem,
            }
        }
        fn pad_using<F>(self, min: usize, f: F) -> PadUsing<Self, F>
        where
            Self: Sized,
            F: FnMut(usize) -> Self::Item,
        {
            PadUsing {
                iter: self,
                min,
                idx: 0,
                f,
            }
        }

        // ── take_while_inclusive / batching / coalesce ──────────────────────────────────────────
        fn take_while_inclusive<F>(self, pred: F) -> TakeWhileInclusive<Self, F>
        where
            Self: Sized,
            F: FnMut(&Self::Item) -> bool,
        {
            TakeWhileInclusive {
                iter: self,
                pred,
                done: false,
            }
        }
        fn batching<B, F>(self, f: F) -> Batching<Self, F>
        where
            Self: Sized,
            F: FnMut(&mut Self) -> Option<B>,
        {
            Batching { iter: self, f }
        }
        fn coalesce<F>(self, f: F) -> Coalesce<Self, F>
        where
            Self: Sized,
            F: FnMut(Self::Item, Self::Item) -> Result<Self::Item, (Self::Item, Self::Item)>,
        {
            Coalesce {
                iter: self,
                last: None,
                f,
            }
        }

        // ── combinatorial generation — the B/U merge iterated to build the relation itself ──────
        fn cartesian_product<J>(self, other: J) -> CartesianProduct<Self, J::IntoIter>
        where
            Self: Sized,
            Self::Item: Clone,
            J: IntoIterator,
            J::IntoIter: Clone,
        {
            let b = other.into_iter();
            CartesianProduct {
                a: self,
                a_cur: None,
                b_orig: b.clone(),
                b_cur: b,
            }
        }
        fn multi_cartesian_product(self) -> MultiProduct<<Self::Item as IntoIterator>::IntoIter>
        where
            Self: Sized,
            Self::Item: IntoIterator,
            <Self::Item as IntoIterator>::IntoIter: Clone,
            <Self::Item as IntoIterator>::Item: Clone,
        {
            let pools: Vec<_> = self.into_iter().map(IntoIterator::into_iter).collect();
            MultiProduct::new(pools)
        }
        fn combinations(self, k: usize) -> Combinations<Self::Item>
        where
            Self: Sized,
            Self::Item: Clone,
        {
            Combinations::new(self.collect(), k)
        }
        fn combinations_with_replacement(self, k: usize) -> CombinationsWithReplacement<Self::Item>
        where
            Self: Sized,
            Self::Item: Clone,
        {
            CombinationsWithReplacement::new(self.collect(), k)
        }
        fn permutations(self, k: usize) -> Permutations<Self::Item>
        where
            Self: Sized,
            Self::Item: Clone,
        {
            Permutations::new(self.collect(), k)
        }
        fn powerset(self) -> Powerset<Self::Item>
        where
            Self: Sized,
            Self::Item: Clone,
        {
            Powerset::new(self.collect())
        }
    }

    impl<I: Iterator> Itertools for I {}
}
pub use methods::*;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// Supporting types — Either, EitherOrBoth, Position, MinMaxResult, error types
// ═══════════════════════════════════════════════════════════════════════════

/// A value that is one of two types — mirrors the real crate's re-export of the `either` crate's
/// `Either`, own-built here since the point is zero dependencies.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Either<L, R> {
    Left(L),
    Right(R),
}

impl<L, R> Either<L, R> {
    pub fn left(self) -> Option<L> {
        match self {
            Either::Left(l) => Some(l),
            Either::Right(_) => None,
        }
    }
    pub fn right(self) -> Option<R> {
        match self {
            Either::Left(_) => None,
            Either::Right(r) => Some(r),
        }
    }
    pub fn is_left(&self) -> bool {
        matches!(self, Either::Left(_))
    }
    pub fn is_right(&self) -> bool {
        matches!(self, Either::Right(_))
    }
    /// Collapse to `T` when both arms carry the same type — the shape `partition_map` uses.
    pub fn into_inner(self) -> L
    where
        R: Into<L>,
    {
        match self {
            Either::Left(l) => l,
            Either::Right(r) => r.into(),
        }
    }
}

/// An `Either` of two same-`Item` iterators is itself an iterator — backs [`Itertools::flatten_ok`]
/// (each branch of `Ok`'s inner iterator vs. a single `Err`, unified so `flat_map` can return one
/// type from both match arms without boxing).
impl<L, R> Iterator for Either<L, R>
where
    L: Iterator,
    R: Iterator<Item = L::Item>,
{
    type Item = L::Item;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Either::Left(l) => l.next(),
            Either::Right(r) => r.next(),
        }
    }
}

/// A value that holds a left, a right, or both — mirrors the real crate exactly.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum EitherOrBoth<A, B = A> {
    Both(A, B),
    Left(A),
    Right(B),
}

impl<A, B> EitherOrBoth<A, B> {
    pub fn has_left(&self) -> bool {
        !matches!(self, EitherOrBoth::Right(_))
    }
    pub fn has_right(&self) -> bool {
        !matches!(self, EitherOrBoth::Left(_))
    }
    pub fn is_left(&self) -> bool {
        matches!(self, EitherOrBoth::Left(_))
    }
    pub fn is_right(&self) -> bool {
        matches!(self, EitherOrBoth::Right(_))
    }
    pub fn is_both(&self) -> bool {
        matches!(self, EitherOrBoth::Both(..))
    }
    pub fn left(self) -> Option<A> {
        match self {
            EitherOrBoth::Both(a, _) | EitherOrBoth::Left(a) => Some(a),
            EitherOrBoth::Right(_) => None,
        }
    }
    pub fn right(self) -> Option<B> {
        match self {
            EitherOrBoth::Both(_, b) | EitherOrBoth::Right(b) => Some(b),
            EitherOrBoth::Left(_) => None,
        }
    }
    pub fn left_or(self, default: A) -> A {
        self.left().unwrap_or(default)
    }
    pub fn right_or(self, default: B) -> B {
        self.right().unwrap_or(default)
    }
}

/// Which position an element occupies in a sequence — yielded by [`Itertools::with_position`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Position<T> {
    First(T),
    Middle(T),
    Last(T),
    Only(T),
}

impl<T> Position<T> {
    pub fn into_inner(self) -> T {
        match self {
            Position::First(t) | Position::Middle(t) | Position::Last(t) | Position::Only(t) => t,
        }
    }
}

/// The result of [`Itertools::minmax`] — distinguishes an empty iterator from a single-element
/// one (where min and max are the same element) from the general two-or-more case.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MinMaxResult<T> {
    NoElements,
    OneElement(T),
    MinMax(T, T),
}

impl<T: Clone> MinMaxResult<T> {
    pub fn into_option(self) -> Option<(T, T)> {
        match self {
            MinMaxResult::NoElements => None,
            MinMaxResult::OneElement(x) => Some((x.clone(), x)),
            MinMaxResult::MinMax(a, b) => Some((a, b)),
        }
    }
}

/// [`Itertools::exactly_one`]/[`Itertools::at_most_one`] failed — carries back whatever couldn't
/// be drained so the caller can inspect it. **Simplified vs. the real crate:** real `itertools`
/// makes this type itself an `Iterator` replaying the drained elements (a `PutBackN`-style
/// resumable wrapper); here it's a plain `Vec`-backed capture instead (via [`Self::into_extra`]) —
/// same information recoverable, not resumable as an `Iterator` in place. `Display` text matches
/// the real crate exactly.
#[derive(Debug)]
pub struct ExactlyOneError<I> {
    /// The second element found (proving there was more than one), followed by anything else
    /// still in the source iterator at the time of failure.
    extra: Vec<I>,
}
impl<I> fmt::Display for ExactlyOneError<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.extra.is_empty() {
            write!(f, "got zero elements when exactly one was expected")
        } else {
            write!(f, "got at least 2 elements when exactly one was expected")
        }
    }
}
impl<I: fmt::Debug> std::error::Error for ExactlyOneError<I> {}
impl<I> ExactlyOneError<I> {
    /// The extra elements found: for `exactly_one`, the second element plus anything after it;
    /// empty only when the source was empty (zero elements, not "too many").
    pub fn into_extra(self) -> impl Iterator<Item = I> {
        self.extra.into_iter()
    }
}

/// [`Itertools::all_equal_value`] found two different values — same shape and `Display` text as
/// the real crate's `AllEqualValueError(pub Option<[Item; 2]>)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllEqualValueError<T>(pub Option<[T; 2]>);
impl<T> fmt::Display for AllEqualValueError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            None => write!(
                f,
                "got zero elements when all elements were expected to be equal"
            ),
            Some(_) => write!(
                f,
                "got different elements when all elements were expected to be equal"
            ),
        }
    }
}
impl<T: fmt::Debug> std::error::Error for AllEqualValueError<T> {}

/// The shared `min_set`/`max_set` scan: linear, ties break by keeping every element that compares
/// equal to the current extreme (matches the real crate's own "collect all extrema" contract).
fn extreme_set<I, F>(iter: I, mut better_or_eq: F) -> Vec<I::Item>
where
    I: Iterator,
    F: FnMut(&I::Item, &I::Item) -> Ordering,
{
    let mut it = iter.into_iter();
    let mut out = match it.next() {
        None => return Vec::new(),
        Some(x) => vec![x],
    };
    for x in it {
        match better_or_eq(&x, &out[0]) {
            Ordering::Less => {
                out.clear();
                out.push(x);
            }
            Ordering::Equal => out.push(x),
            Ordering::Greater => {}
        }
    }
    out
}

/// [`Itertools::update`]'s adaptor — mutate each item in place, then yield it. A plain `map`
/// closure can't express "take `&mut Item`, return `Item`" without a named struct (implementing
/// the real `Fn`/`FnMut` traits directly is nightly-only `unboxed_closures`/`fn_traits`; this is
/// the stable equivalent: its own `Iterator` impl instead of routing through `std::iter::Map`).
pub struct Update<I, F> {
    iter: I,
    f: F,
}
impl<I: Iterator, F: FnMut(&mut I::Item)> Iterator for Update<I, F> {
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        let mut x = self.iter.next()?;
        (self.f)(&mut x);
        Some(x)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

/// Backs [`Itertools::multiunzip`] — implemented for 2..=8-tuples of iterator items, matching the
/// real crate's coverage.
pub trait MultiUnzip<FromI> {
    fn multiunzip(self) -> FromI;
}

// Spelled out per-arity (2..=6, matching the range consumers actually reach for): a
// macro_rules-generated version needs a distinct destructure-binding name per accumulator, which
// hygiene can't derive from the type parameter alone without a second parallel token list per
// arity anyway — explicit is no shorter and clearer to read.
impl<I, A, B> MultiUnzip<(Vec<A>, Vec<B>)> for I
where
    I: Iterator<Item = (A, B)>,
{
    fn multiunzip(self) -> (Vec<A>, Vec<B>) {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for (x, y) in self {
            a.push(x);
            b.push(y);
        }
        (a, b)
    }
}
impl<I, A, B, C> MultiUnzip<(Vec<A>, Vec<B>, Vec<C>)> for I
where
    I: Iterator<Item = (A, B, C)>,
{
    fn multiunzip(self) -> (Vec<A>, Vec<B>, Vec<C>) {
        let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        for (x, y, z) in self {
            a.push(x);
            b.push(y);
            c.push(z);
        }
        (a, b, c)
    }
}
impl<I, A, B, C, D> MultiUnzip<(Vec<A>, Vec<B>, Vec<C>, Vec<D>)> for I
where
    I: Iterator<Item = (A, B, C, D)>,
{
    fn multiunzip(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>) {
        let (mut a, mut b, mut c, mut d) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (w, x, y, z) in self {
            a.push(w);
            b.push(x);
            c.push(y);
            d.push(z);
        }
        (a, b, c, d)
    }
}
impl<I, A, B, C, D, E> MultiUnzip<(Vec<A>, Vec<B>, Vec<C>, Vec<D>, Vec<E>)> for I
where
    I: Iterator<Item = (A, B, C, D, E)>,
{
    fn multiunzip(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>, Vec<E>) {
        let (mut a, mut b, mut c, mut d, mut e) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (v, w, x, y, z) in self {
            a.push(v);
            b.push(w);
            c.push(x);
            d.push(y);
            e.push(z);
        }
        (a, b, c, d, e)
    }
}
impl<I, A, B, C, D, E, F> MultiUnzip<(Vec<A>, Vec<B>, Vec<C>, Vec<D>, Vec<E>, Vec<F>)> for I
where
    I: Iterator<Item = (A, B, C, D, E, F)>,
{
    fn multiunzip(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>, Vec<E>, Vec<F>) {
        let (mut a, mut b, mut c, mut d, mut e, mut f) = (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        for (u, v, w, x, y, z) in self {
            a.push(u);
            b.push(v);
            c.push(w);
            d.push(x);
            e.push(y);
            f.push(z);
        }
        (a, b, c, d, e, f)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Lazy adaptor types — unique / duplicates / dedup family
// ═══════════════════════════════════════════════════════════════════════════

/// [`Itertools::unique`]/[`Itertools::unique_by`]'s adaptor — a `HashSet`-backed "have I seen this
/// key" residual (the NB face: absence of a prior sighting is what lets an item through).
pub struct Unique<I: Iterator, K, F = fn(&<I as Iterator>::Item) -> K> {
    iter: I,
    seen: std::collections::HashSet<K>,
    key: F,
}
impl<I, K, F> Iterator for Unique<I, K, F>
where
    I: Iterator,
    K: Eq + std::hash::Hash,
    F: FnMut(&I::Item) -> K,
{
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let x = self.iter.next()?;
            let k = (self.key)(&x);
            if self.seen.insert(k) {
                return Some(x);
            }
        }
    }
}

/// [`Itertools::duplicates`]/[`Itertools::duplicates_by`]'s adaptor — yields an item exactly once,
/// on its SECOND sighting (verified against the real crate's own `used: HashMap<Key, bool>`
/// "already produced" flag: a third+ sighting of the same key is suppressed, not re-yielded).
pub struct Duplicates<I: Iterator, K, F = fn(&<I as Iterator>::Item) -> K> {
    iter: I,
    seen: std::collections::HashMap<K, bool>,
    key: F,
}
impl<I, K, F> Iterator for Duplicates<I, K, F>
where
    I: Iterator,
    K: Eq + std::hash::Hash,
    F: FnMut(&I::Item) -> K,
{
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let x = self.iter.next()?;
            let k = (self.key)(&x);
            match self.seen.get_mut(&k) {
                None => {
                    self.seen.insert(k, false);
                }
                Some(true) => {}
                Some(produced) => {
                    *produced = true;
                    return Some(x);
                }
            }
        }
    }
}

/// [`Itertools::dedup`]'s adaptor — collapses consecutive equal runs to their first element (the
/// B/U merge specialised to "same as my immediate predecessor").
pub struct Dedup<I: Iterator> {
    iter: I,
    last: Option<I::Item>,
    eq: fn(&I::Item, &I::Item) -> bool,
}
impl<I: Iterator> Iterator for Dedup<I>
where
    I::Item: Clone,
{
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let x = self.iter.next()?;
            if let Some(last) = &self.last {
                if (self.eq)(last, &x) {
                    continue;
                }
            }
            self.last = Some(x.clone());
            return Some(x);
        }
    }
}

/// [`Itertools::dedup_by`]'s adaptor — same collapse, caller-supplied equality.
pub struct DedupPred<I: Iterator, F> {
    iter: I,
    last: Option<I::Item>,
    eq: F,
}
impl<I: Iterator, F> Iterator for DedupPred<I, F>
where
    I::Item: Clone,
    F: FnMut(&I::Item, &I::Item) -> bool,
{
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let x = self.iter.next()?;
            if let Some(last) = &self.last {
                if (self.eq)(last, &x) {
                    continue;
                }
            }
            self.last = Some(x.clone());
            return Some(x);
        }
    }
}

/// [`Itertools::dedup_with_count`]'s adaptor — same collapse, yielding `(run_length, item)`. Uses
/// a `Peekable` inner iterator so a non-matching lookahead item is never lost — it stays queued
/// in the peek slot for the next call to `next()`.
pub struct DedupWithCount<I: Iterator> {
    iter: std::iter::Peekable<I>,
    eq: fn(&I::Item, &I::Item) -> bool,
}
impl<I: Iterator> Iterator for DedupWithCount<I> {
    type Item = (usize, I::Item);
    fn next(&mut self) -> Option<Self::Item> {
        let first = self.iter.next()?;
        let mut count = 1usize;
        while let Some(peeked) = self.iter.peek() {
            if (self.eq)(&first, peeked) {
                self.iter.next();
                count += 1;
            } else {
                break;
            }
        }
        Some((count, first))
    }
}

/// [`Itertools::dedup_by_with_count`]'s adaptor — same as [`DedupWithCount`], caller-supplied
/// equality.
pub struct DedupWithCountPred<I: Iterator, F> {
    iter: std::iter::Peekable<I>,
    eq: F,
}
impl<I: Iterator, F: FnMut(&I::Item, &I::Item) -> bool> Iterator for DedupWithCountPred<I, F> {
    type Item = (usize, I::Item);
    fn next(&mut self) -> Option<Self::Item> {
        let first = self.iter.next()?;
        let mut count = 1usize;
        while let Some(peeked) = self.iter.peek() {
            if (self.eq)(&first, peeked) {
                self.iter.next();
                count += 1;
            } else {
                break;
            }
        }
        Some((count, first))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// positions / with_position
// ═══════════════════════════════════════════════════════════════════════════

/// [`Itertools::positions`]'s adaptor — yields the index of every item where `pred` holds.
pub struct Positions<I, F> {
    iter: I,
    pred: F,
    idx: usize,
}
impl<I: Iterator, F: FnMut(I::Item) -> bool> Iterator for Positions<I, F> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        loop {
            let x = self.iter.next()?;
            let i = self.idx;
            self.idx += 1;
            if (self.pred)(x) {
                return Some(i);
            }
        }
    }
}

/// [`Itertools::with_position`]'s adaptor — tags each item as First/Middle/Last/Only, needing one
/// item of lookahead (via `Peekable`) to know whether the current item is the last one.
pub struct WithPosition<I: Iterator> {
    iter: std::iter::Peekable<I>,
    started: bool,
}
impl<I: Iterator> Iterator for WithPosition<I> {
    type Item = Position<I::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        let x = self.iter.next()?;
        let is_last = self.iter.peek().is_none();
        let out = match (self.started, is_last) {
            (false, true) => Position::Only(x),
            (false, false) => Position::First(x),
            (true, true) => Position::Last(x),
            (true, false) => Position::Middle(x),
        };
        self.started = true;
        Some(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// tuple_windows / tuples — fixed-arity grouping (2..=4-tuples, the common range)
// ═══════════════════════════════════════════════════════════════════════════

/// Backs [`Itertools::tuple_windows`]/[`Itertools::tuples`] — implemented for 2..=4-tuples of a
/// single item type, matching the arities consumers actually reach for (the real crate goes to
/// 12; the mechanism here is identical per-arity, just not spelled out that far).
pub trait HomogeneousTuple: Sized {
    type Item;
    const ARITY: usize;
    fn from_slice(v: &[Self::Item]) -> Self
    where
        Self::Item: Clone;
}
impl<A> HomogeneousTuple for (A, A) {
    type Item = A;
    const ARITY: usize = 2;
    fn from_slice(v: &[A]) -> Self
    where
        A: Clone,
    {
        (v[0].clone(), v[1].clone())
    }
}
impl<A> HomogeneousTuple for (A, A, A) {
    type Item = A;
    const ARITY: usize = 3;
    fn from_slice(v: &[A]) -> Self
    where
        A: Clone,
    {
        (v[0].clone(), v[1].clone(), v[2].clone())
    }
}
impl<A> HomogeneousTuple for (A, A, A, A) {
    type Item = A;
    const ARITY: usize = 4;
    fn from_slice(v: &[A]) -> Self
    where
        A: Clone,
    {
        (v[0].clone(), v[1].clone(), v[2].clone(), v[3].clone())
    }
}

/// [`Itertools::tuple_windows`]'s adaptor — overlapping fixed-size windows, sliding by one.
pub struct TupleWindows<I: Iterator, T> {
    iter: I,
    buf: std::collections::VecDeque<I::Item>,
    _marker: std::marker::PhantomData<T>,
}
impl<I, T> Iterator for TupleWindows<I, T>
where
    I: Iterator,
    I::Item: Clone,
    T: HomogeneousTuple<Item = I::Item>,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        while self.buf.len() < T::ARITY {
            self.buf.push_back(self.iter.next()?);
        }
        let items: Vec<I::Item> = self.buf.iter().cloned().collect();
        let out = T::from_slice(&items);
        self.buf.pop_front();
        Some(out)
    }
}

/// [`Itertools::tuples`]'s adaptor — non-overlapping fixed-size chunks. A trailing partial chunk
/// (fewer than `T::ARITY` items left) is dropped, matching the real crate's default behavior
/// (there recoverable via a separate `.into_buffer()`; not built here — a named, narrow gap).
pub struct Tuples<I: Iterator, T> {
    iter: I,
    _marker: std::marker::PhantomData<T>,
}
impl<I, T> Iterator for Tuples<I, T>
where
    I: Iterator,
    T: HomogeneousTuple<Item = I::Item>,
    I::Item: Clone,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        let mut buf = Vec::with_capacity(T::ARITY);
        for _ in 0..T::ARITY {
            buf.push(self.iter.next()?);
        }
        Some(T::from_slice(&buf))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// chunks — simplified: each chunk is an eagerly-collected Vec (see module doc gap note)
// ═══════════════════════════════════════════════════════════════════════════

pub struct ChunksIter<I: Iterator> {
    iter: I,
    size: usize,
}
impl<I: Iterator> Iterator for ChunksIter<I> {
    type Item = Vec<I::Item>;
    fn next(&mut self) -> Option<Vec<I::Item>> {
        let mut v = Vec::with_capacity(self.size);
        for _ in 0..self.size {
            match self.iter.next() {
                Some(x) => v.push(x),
                None => break,
            }
        }
        if v.is_empty() {
            None
        } else {
            Some(v)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// merge / merge_by / merge_join_by — two ordered streams, the B/U merge itself
// ═══════════════════════════════════════════════════════════════════════════

pub struct Merge<A: Iterator, B: Iterator<Item = A::Item>> {
    a: std::iter::Peekable<A>,
    b: std::iter::Peekable<B>,
    cmp: fn(&A::Item, &A::Item) -> bool,
}
impl<A: Iterator, B: Iterator<Item = A::Item>> Iterator for Merge<A, B> {
    type Item = A::Item;
    fn next(&mut self) -> Option<Self::Item> {
        match (self.a.peek(), self.b.peek()) {
            (Some(x), Some(y)) => {
                if (self.cmp)(x, y) {
                    self.a.next()
                } else {
                    self.b.next()
                }
            }
            (Some(_), None) => self.a.next(),
            (None, Some(_)) => self.b.next(),
            (None, None) => None,
        }
    }
}

pub struct MergeBy<A: Iterator, B: Iterator<Item = A::Item>, F> {
    a: std::iter::Peekable<A>,
    b: std::iter::Peekable<B>,
    is_first: F,
}
impl<A, B, F> Iterator for MergeBy<A, B, F>
where
    A: Iterator,
    B: Iterator<Item = A::Item>,
    F: FnMut(&A::Item, &A::Item) -> bool,
{
    type Item = A::Item;
    fn next(&mut self) -> Option<Self::Item> {
        match (self.a.peek(), self.b.peek()) {
            (Some(x), Some(y)) => {
                if (self.is_first)(x, y) {
                    self.a.next()
                } else {
                    self.b.next()
                }
            }
            (Some(_), None) => self.a.next(),
            (None, Some(_)) => self.b.next(),
            (None, None) => None,
        }
    }
}

pub struct MergeJoinBy<A: Iterator, B: Iterator, F> {
    a: std::iter::Peekable<A>,
    b: std::iter::Peekable<B>,
    cmp: F,
}
impl<A, B, F, T> Iterator for MergeJoinBy<A, B, F>
where
    A: Iterator,
    B: Iterator,
    F: FnMut(&A::Item, &B::Item) -> T,
    T: Into<Ordering>,
{
    type Item = EitherOrBoth<A::Item, B::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        match (self.a.peek(), self.b.peek()) {
            (Some(x), Some(y)) => match (self.cmp)(x, y).into() {
                Ordering::Less => self.a.next().map(EitherOrBoth::Left),
                Ordering::Greater => self.b.next().map(EitherOrBoth::Right),
                Ordering::Equal => {
                    let a = self.a.next().unwrap();
                    let b = self.b.next().unwrap();
                    Some(EitherOrBoth::Both(a, b))
                }
            },
            (Some(_), None) => self.a.next().map(EitherOrBoth::Left),
            (None, Some(_)) => self.b.next().map(EitherOrBoth::Right),
            (None, None) => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// kmerge / kmerge_by — N ordered streams. Owned one-ahead peek per head (not `Peekable`,
// which needs `&mut` to peek — with N heads in one `Vec`, comparing two peeks at once needs
// them as owned values behind immutable borrows, not two live mutable borrows into the Vec).
// ═══════════════════════════════════════════════════════════════════════════

struct HeadState<I: Iterator> {
    iter: I,
    peeked: Option<I::Item>,
}
impl<I: Iterator> HeadState<I> {
    fn new(mut iter: I) -> Self {
        let peeked = iter.next();
        HeadState { iter, peeked }
    }
}

pub struct KMerge<I: Iterator> {
    heads: Vec<HeadState<I>>,
    cmp: fn(&I::Item, &I::Item) -> bool,
}
impl<I: Iterator> Iterator for KMerge<I> {
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        kmerge_next(&mut self.heads, &mut self.cmp)
    }
}

pub struct KMergeBy<I: Iterator, F> {
    heads: Vec<HeadState<I>>,
    cmp: F,
}
impl<I: Iterator, F: FnMut(&I::Item, &I::Item) -> bool> Iterator for KMergeBy<I, F> {
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        kmerge_next(&mut self.heads, &mut self.cmp)
    }
}

fn kmerge_next<I: Iterator>(
    heads: &mut [HeadState<I>],
    cmp: &mut impl FnMut(&I::Item, &I::Item) -> bool,
) -> Option<I::Item> {
    let mut best: Option<usize> = None;
    for i in 0..heads.len() {
        if heads[i].peeked.is_none() {
            continue;
        }
        best = Some(match best {
            None => i,
            Some(b) => {
                let use_i = cmp(
                    heads[i].peeked.as_ref().unwrap(),
                    heads[b].peeked.as_ref().unwrap(),
                );
                if use_i {
                    i
                } else {
                    b
                }
            }
        });
    }
    let i = best?;
    let out = heads[i].peeked.take();
    heads[i].peeked = heads[i].iter.next();
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// interleave / interleave_shortest
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::struct_excessive_bools)] // interleave state machine — each flag is a distinct phase
pub struct Interleave<A: Iterator, B: Iterator<Item = A::Item>> {
    a: A,
    b: B,
    turn_a: bool,
    a_done: bool,
    b_done: bool,
    shortest: bool,
}
impl<A: Iterator, B: Iterator<Item = A::Item>> Iterator for Interleave<A, B> {
    type Item = A::Item;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.a_done && self.b_done {
                return None;
            }
            let take_a = self.turn_a;
            self.turn_a = !self.turn_a;
            if take_a {
                if self.a_done {
                    continue;
                }
                if let Some(x) = self.a.next() {
                    return Some(x);
                }
                self.a_done = true;
                if self.shortest {
                    return None;
                }
                continue;
            }
            if self.b_done {
                continue;
            }
            if let Some(x) = self.b.next() {
                return Some(x);
            }
            self.b_done = true;
            if self.shortest {
                return None;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// zip_longest / zip_eq
// ═══════════════════════════════════════════════════════════════════════════

pub struct ZipLongest<A: Iterator, B: Iterator> {
    a: A,
    b: B,
}
impl<A: Iterator, B: Iterator> Iterator for ZipLongest<A, B> {
    type Item = EitherOrBoth<A::Item, B::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        match (self.a.next(), self.b.next()) {
            (Some(a), Some(b)) => Some(EitherOrBoth::Both(a, b)),
            (Some(a), None) => Some(EitherOrBoth::Left(a)),
            (None, Some(b)) => Some(EitherOrBoth::Right(b)),
            (None, None) => None,
        }
    }
}

/// [`Itertools::zip_eq`]'s adaptor — like `zip`, but panics if the two iterators have different
/// lengths (matches the real crate's contract: this is an assertion tool, not a lenient zip).
pub struct ZipEq<A: Iterator, B: Iterator> {
    a: A,
    b: B,
}
impl<A: Iterator, B: Iterator> Iterator for ZipEq<A, B> {
    type Item = (A::Item, B::Item);
    fn next(&mut self) -> Option<Self::Item> {
        match (self.a.next(), self.b.next()) {
            (Some(a), Some(b)) => Some((a, b)),
            (None, None) => None,
            _ => panic!("zip_eq: iterators had different lengths"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// intersperse / intersperse_with / pad_using
// ═══════════════════════════════════════════════════════════════════════════

/// Wraps a fixed element as a `FnMut() -> T` so [`Itertools::intersperse`] can share
/// [`IntersperseWith`]'s implementation with [`Itertools::intersperse_with`]. Implements the two
/// unboxed-closure traits directly (nightly `fn_traits`/`unboxed_closures`)... **no** — kept
/// stable: see the plain method below instead of a `Fn` impl.
pub struct ClonedElement<T>(T);
impl<T: Clone> ClonedElement<T> {
    fn call(&mut self) -> T {
        self.0.clone()
    }
}

pub struct IntersperseWith<I: Iterator, F> {
    iter: std::iter::Peekable<I>,
    next_is_elem: bool,
    make_elem: F,
}
impl<I: Iterator> Iterator for IntersperseWith<I, ClonedElement<I::Item>>
where
    I::Item: Clone,
{
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next_is_elem {
            if self.iter.peek().is_some() {
                self.next_is_elem = false;
                return Some(self.make_elem.call());
            }
            return None;
        }
        self.next_is_elem = true;
        self.iter.next()
    }
}
impl<I: Iterator, F: FnMut() -> I::Item> Iterator for IntersperseWith<I, F> {
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next_is_elem {
            if self.iter.peek().is_some() {
                self.next_is_elem = false;
                return Some((self.make_elem)());
            }
            return None;
        }
        self.next_is_elem = true;
        self.iter.next()
    }
}

pub struct PadUsing<I: Iterator, F> {
    iter: I,
    min: usize,
    idx: usize,
    f: F,
}
impl<I: Iterator, F: FnMut(usize) -> I::Item> Iterator for PadUsing<I, F> {
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(x) => {
                self.idx += 1;
                Some(x)
            }
            None if self.idx < self.min => {
                let x = (self.f)(self.idx);
                self.idx += 1;
                Some(x)
            }
            None => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// take_while_inclusive / batching / coalesce
// ═══════════════════════════════════════════════════════════════════════════

pub struct TakeWhileInclusive<I: Iterator, F> {
    iter: I,
    pred: F,
    done: bool,
}
impl<I: Iterator, F: FnMut(&I::Item) -> bool> Iterator for TakeWhileInclusive<I, F> {
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let x = self.iter.next()?;
        if !(self.pred)(&x) {
            self.done = true;
        }
        Some(x)
    }
}

pub struct Batching<I, F> {
    iter: I,
    f: F,
}
impl<I: Iterator, B, F: FnMut(&mut I) -> Option<B>> Iterator for Batching<I, F> {
    type Item = B;
    fn next(&mut self) -> Option<B> {
        (self.f)(&mut self.iter)
    }
}

pub struct Coalesce<I: Iterator, F> {
    iter: I,
    last: Option<I::Item>,
    f: F,
}
impl<I, F> Iterator for Coalesce<I, F>
where
    I: Iterator,
    F: FnMut(I::Item, I::Item) -> Result<I::Item, (I::Item, I::Item)>,
{
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        let mut current = match self.last.take() {
            Some(x) => x,
            None => self.iter.next()?,
        };
        loop {
            match self.iter.next() {
                None => return Some(current),
                Some(next) => match (self.f)(current, next) {
                    Ok(merged) => current = merged,
                    Err((keep, stash)) => {
                        self.last = Some(stash);
                        return Some(keep);
                    }
                },
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// combinatorial generation: cartesian_product / multi_cartesian_product /
// combinations / combinations_with_replacement / permutations / powerset
// ═══════════════════════════════════════════════════════════════════════════

pub struct CartesianProduct<A: Iterator, B: Iterator> {
    a: A,
    a_cur: Option<A::Item>,
    b_orig: B,
    b_cur: B,
}
impl<A, B> Iterator for CartesianProduct<A, B>
where
    A: Iterator,
    A::Item: Clone,
    B: Iterator + Clone,
{
    type Item = (A::Item, B::Item);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.a_cur.is_none() {
                self.a_cur = Some(self.a.next()?);
                self.b_cur = self.b_orig.clone();
            }
            match self.b_cur.next() {
                Some(y) => return Some((self.a_cur.clone().unwrap(), y)),
                None => self.a_cur = None,
            }
        }
    }
}

/// [`Itertools::multi_cartesian_product`]'s adaptor — the Cartesian product of an arbitrary
/// number of pools, odometer-style (rightmost pool advances fastest).
pub struct MultiProduct<I: Iterator>
where
    I::Item: Clone,
{
    pools: Vec<I>,
    pool_origin: Vec<I>,
    cur: Option<Vec<I::Item>>,
}
impl<I: Iterator + Clone> MultiProduct<I>
where
    I::Item: Clone,
{
    fn new(pools: Vec<I>) -> Self {
        MultiProduct {
            pool_origin: pools.clone(),
            pools,
            cur: None,
        }
    }
}
impl<I: Iterator + Clone> Iterator for MultiProduct<I>
where
    I::Item: Clone,
{
    type Item = Vec<I::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pool_origin.is_empty() {
            return None;
        }
        match &mut self.cur {
            None => {
                let mut first = Vec::with_capacity(self.pools.len());
                for p in &mut self.pools {
                    first.push(p.next()?);
                }
                self.cur = Some(first.clone());
                Some(first)
            }
            Some(cur) => {
                // Advance like an odometer from the rightmost digit.
                let n = self.pools.len();
                let mut i = n;
                loop {
                    if i == 0 {
                        return None;
                    }
                    i -= 1;
                    if let Some(v) = self.pools[i].next() {
                        cur[i] = v;
                        #[allow(clippy::needless_range_loop)]
                        // parallel index into pools/origin/cur
                        for j in (i + 1)..n {
                            self.pools[j] = self.pool_origin[j].clone();
                            cur[j] = self.pools[j].next().expect("pool_origin pool is non-empty");
                        }
                        return Some(cur.clone());
                    }
                }
            }
        }
    }
}

/// [`Itertools::combinations`]'s adaptor — all `k`-length combinations, lexicographic index order.
pub struct Combinations<T> {
    pool: Vec<T>,
    k: usize,
    indices: Vec<usize>,
    started: bool,
    done: bool,
}
impl<T: Clone> Combinations<T> {
    fn new(pool: Vec<T>, k: usize) -> Self {
        let done = k > pool.len();
        Combinations {
            pool,
            k,
            indices: (0..k).collect(),
            started: false,
            done,
        }
    }
}
impl<T: Clone> Iterator for Combinations<T> {
    type Item = Vec<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if !self.started {
            self.started = true;
            if self.k == 0 {
                self.done = true;
                return Some(Vec::new());
            }
            return Some(self.indices.iter().map(|&i| self.pool[i].clone()).collect());
        }
        let n = self.pool.len();
        let k = self.k;
        let mut i = k;
        loop {
            if i == 0 {
                self.done = true;
                return None;
            }
            i -= 1;
            if self.indices[i] != i + n - k {
                break;
            }
        }
        self.indices[i] += 1;
        for j in (i + 1)..k {
            self.indices[j] = self.indices[j - 1] + 1;
        }
        Some(
            self.indices
                .iter()
                .map(|&idx| self.pool[idx].clone())
                .collect(),
        )
    }
}

/// [`Itertools::combinations_with_replacement`]'s adaptor.
pub struct CombinationsWithReplacement<T> {
    pool: Vec<T>,
    k: usize,
    indices: Vec<usize>,
    started: bool,
    done: bool,
}
impl<T: Clone> CombinationsWithReplacement<T> {
    fn new(pool: Vec<T>, k: usize) -> Self {
        let done = pool.is_empty() && k > 0;
        CombinationsWithReplacement {
            pool,
            k,
            indices: vec![0; k],
            started: false,
            done,
        }
    }
}
impl<T: Clone> Iterator for CombinationsWithReplacement<T> {
    type Item = Vec<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if !self.started {
            self.started = true;
            if self.k == 0 {
                self.done = true;
                return Some(Vec::new());
            }
            return Some(self.indices.iter().map(|&i| self.pool[i].clone()).collect());
        }
        let n = self.pool.len();
        let k = self.k;
        let mut i = k;
        loop {
            if i == 0 {
                self.done = true;
                return None;
            }
            i -= 1;
            if self.indices[i] + 1 < n {
                break;
            }
        }
        let bump = self.indices[i] + 1;
        for j in i..k {
            self.indices[j] = bump;
        }
        Some(
            self.indices
                .iter()
                .map(|&idx| self.pool[idx].clone())
                .collect(),
        )
    }
}

/// [`Itertools::permutations`]'s adaptor — all `k`-length permutations, in the SAME lexicographic
/// order as the real crate (verified directly: `[1,2,3].permutations(2)` →
/// `[[1,2],[1,3],[2,1],[2,3],[3,1],[3,2]]`, i.e. sorted by the resulting sequence itself — NOT
/// grouped by combination, which an earlier draft of this got wrong). Implementation: walk every
/// full permutation of `0..n` in lexicographic order and take each one's first `k` indices,
/// skipping when that prefix repeats the previous one (consecutive, since full lex order groups
/// by leading elements first) — this visits exactly the `n!/(n-k)!` distinct k-prefixes in the
/// same order the real crate does, without a combination layer underneath.
pub struct Permutations<T> {
    pool: Vec<T>,
    k: usize,
    state: Vec<usize>,
    started: bool,
    exhausted: bool,
    last_prefix: Option<Vec<usize>>,
}
impl<T: Clone> Permutations<T> {
    fn new(pool: Vec<T>, k: usize) -> Self {
        let n = pool.len();
        let exhausted = k > n;
        Permutations {
            pool,
            k,
            state: (0..n).collect(),
            started: false,
            exhausted,
            last_prefix: None,
        }
    }
}
impl<T: Clone> Iterator for Permutations<T> {
    type Item = Vec<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        if self.k == 0 {
            self.exhausted = true;
            return Some(Vec::new());
        }
        loop {
            if !self.started {
                self.started = true;
            } else if !advance_permutation(&mut self.state) {
                self.exhausted = true;
                return None;
            }
            let prefix = self.state[..self.k].to_vec();
            if self.last_prefix.as_ref() != Some(&prefix) {
                self.last_prefix = Some(prefix.clone());
                return Some(prefix.iter().map(|&i| self.pool[i].clone()).collect());
            }
            // Same k-prefix as last time (only the untaken suffix changed) — keep advancing.
        }
    }
}
/// Advances `state` (a permutation of `0..n`) to its lexicographic successor in place. Returns
/// `false` (leaving `state` unchanged) if it was already the last permutation.
fn advance_permutation(state: &mut [usize]) -> bool {
    let n = state.len();
    if n < 2 {
        return false;
    }
    let mut i = n - 1;
    while i > 0 && state[i - 1] >= state[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = n - 1;
    while state[j] <= state[i - 1] {
        j -= 1;
    }
    state.swap(i - 1, j);
    state[i..].reverse();
    true
}

/// [`Itertools::powerset`]'s adaptor — every subset, smallest first (empty set, then all 1-subsets,
/// 2-subsets, ...).
pub struct Powerset<T> {
    pool: Vec<T>,
    k: usize,
    combos: Combinations<T>,
}
impl<T: Clone> Powerset<T> {
    fn new(pool: Vec<T>) -> Self {
        let combos = Combinations::new(pool.clone(), 0);
        Powerset { pool, k: 0, combos }
    }
}
impl<T: Clone> Iterator for Powerset<T> {
    type Item = Vec<T>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(c) = self.combos.next() {
                return Some(c);
            }
            self.k += 1;
            if self.k > self.pool.len() {
                return None;
            }
            self.combos = Combinations::new(self.pool.clone(), self.k);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// chunk_by / group_by — CONSECUTIVE-run grouping (distinct from into_group_map's
// group-by-key-anywhere). Simplified vs. the real crate's lazy shared-buffer `GroupBy`/`Group`
// pair: each run is eagerly collected into a `Vec` as soon as its key changes.
// ═══════════════════════════════════════════════════════════════════════════

pub struct ChunkBy<I: Iterator, K, F> {
    iter: I,
    key: F,
    pending: Option<(K, I::Item)>,
}
impl<I, K, F> Iterator for ChunkBy<I, K, F>
where
    I: Iterator,
    K: PartialEq,
    F: FnMut(&I::Item) -> K,
{
    type Item = (K, Vec<I::Item>);
    fn next(&mut self) -> Option<Self::Item> {
        let (cur_key, first) = if let Some(kv) = self.pending.take() {
            kv
        } else {
            let x = self.iter.next()?;
            let k = (self.key)(&x);
            (k, x)
        };
        let mut group = vec![first];
        for x in self.iter.by_ref() {
            let k = (self.key)(&x);
            if k == cur_key {
                group.push(x);
            } else {
                self.pending = Some((k, x));
                return Some((cur_key, group));
            }
        }
        Some((cur_key, group))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// into_grouping_map / into_grouping_map_by — grouped values plus a small aggregate API
// ═══════════════════════════════════════════════════════════════════════════

/// [`Itertools::into_grouping_map`]/[`Itertools::into_grouping_map_by`]'s result — the grouped
/// `Vec`s plus the real crate's small aggregate API (`sum`/`max`/`min`/`fold`/`reduce`/`collect`).
pub struct GroupingMap<K, V> {
    groups: HashMap<K, Vec<V>>,
}
impl<K: Eq + std::hash::Hash, V> GroupingMap<K, V> {
    pub fn sum(self) -> HashMap<K, V>
    where
        V: std::iter::Sum,
    {
        self.groups
            .into_iter()
            .map(|(k, vs)| (k, vs.into_iter().sum()))
            .collect()
    }
    pub fn max(self) -> HashMap<K, V>
    where
        V: Ord,
    {
        self.groups
            .into_iter()
            .filter_map(|(k, vs)| vs.into_iter().max().map(|v| (k, v)))
            .collect()
    }
    pub fn min(self) -> HashMap<K, V>
    where
        V: Ord,
    {
        self.groups
            .into_iter()
            .filter_map(|(k, vs)| vs.into_iter().min().map(|v| (k, v)))
            .collect()
    }
    pub fn collect<C: FromIterator<V>>(self) -> HashMap<K, C> {
        self.groups
            .into_iter()
            .map(|(k, vs)| (k, vs.into_iter().collect()))
            .collect()
    }
    pub fn fold<B: Clone, F: FnMut(B, V) -> B>(self, init: B, mut f: F) -> HashMap<K, B> {
        self.groups
            .into_iter()
            .map(|(k, vs)| {
                let b = vs.into_iter().fold(init.clone(), &mut f);
                (k, b)
            })
            .collect()
    }
    pub fn reduce<F: FnMut(V, V) -> V>(self, mut f: F) -> HashMap<K, V> {
        self.groups
            .into_iter()
            .filter_map(|(k, vs)| vs.into_iter().reduce(&mut f).map(|v| (k, v)))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// fold_while
// ═══════════════════════════════════════════════════════════════════════════

/// [`Itertools::fold_while`]'s control-flow value — same two variants and names as the real crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldWhile<T> {
    Continue(T),
    Done(T),
}
impl<T> FoldWhile<T> {
    pub fn into_inner(self) -> T {
        match self {
            FoldWhile::Continue(x) | FoldWhile::Done(x) => x,
        }
    }
    pub fn is_done(&self) -> bool {
        matches!(self, FoldWhile::Done(_))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// process_results
// ═══════════════════════════════════════════════════════════════════════════

/// [`Itertools::process_results`]'s wrapper — an `Iterator<Item = T>` over an
/// `Iterator<Item = Result<T, E>>`, stashing the first `Err` (if any) into the borrowed `err` slot
/// and stopping there; the caller's closure sees only `T`s, and `process_results` itself reports
/// the stashed error afterward.
pub struct ProcessResults<'a, I, E> {
    iter: I,
    err: &'a mut Option<E>,
}
impl<I, T, E> Iterator for ProcessResults<'_, I, E>
where
    I: Iterator<Item = Result<T, E>>,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.err.is_some() {
            return None;
        }
        match self.iter.next() {
            Some(Ok(x)) => Some(x),
            Some(Err(e)) => {
                *self.err = Some(e);
                None
            }
            None => None,
        }
    }
}
