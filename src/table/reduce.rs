// The whole-table reduce: incumbent `fold hashtrinity/src/lib.rs:891` (`Table::encircle`).
// Included textually by `include!("table/reduce.rs")` from `src/table.rs`, so these items stay
// definitions of the `table` module and keep their generated type identities. Bodies, layout,
// field privacy, signatures and visibility are unchanged; this is a physical-file cut only.
//
// The ported body is `encircle`'s `fold_range`, generalised from closures to a [`Reduce`] witness
// for the same reason `Merge<V>` is a witness and not a closure: the extraction translates a trait
// instance and rejects a closure. The incumbent's serial branch, `if pool.width() <= 1 || n < 1<<14
// { return fold_range(0, n) }`, is the same code path as its per-task chunk, so porting
// `fold_range` alone ports the body that both branches run.
//
// Two deltas from the incumbent, both forced by the storage rather than chosen:
//
//   * The incumbent folds `self.coords` -- one flat array. Apeiron's table is resident-then-spill
//     (`Table::resident` then `Table::shards`), so the reduce walks resident slots then every
//     shard's slots, in that order. That is the order [`live_locations_of`](live_locations_of)
//     already documents and the order the relocation pass moves values along.
//   * `Reduce` takes `residual` and `combine` as methods on one witness rather than two closures,
//     so the extracted `word_table_reduce_all` calls the real body with a concrete `Reduce`.
//
// **Schedule-freedom is not asserted here.** The incumbent's parallel branch folds one partial per
// task and merges them in task order, and its doc comment claims "any task->executor assignment and
// any merge order give the same answer". That claim is a statement about the monoid, so it lives in
// the algebra, not in this loop: `Apeiron.Trichotomy.AnyRing.ringAffineCompose_assoc` and
// `ringAffineCompose_id` are what make it true for [`AffineFoldStep`], and
// `proof/corpus/TableRefinement/EncircleFold.lean` derives it from those two corpus theorems rather
// than from a comment. This loop only has to preserve storage order.

/// A whole-table reduce over the table's stored values: one witness carrying the identity, the
/// per-value residual, and the combine.
///
/// This is the incumbent's `encircle` closure triple `(&self, id, residual, combine)` with the
/// closure type erased to a witness, for the reason [`Merge`](crate::merge::Merge) is a witness:
/// `Merge` is the table's required merge signature, and the extraction translates
/// `table::WordMerge` while it rejects a captured closure.
///
/// The contract is [`Merge`](crate::merge::Merge)'s contract at the level of the accumulator:
/// `combine` is applied as `combine(acc, next)` so the accumulator is always the left argument, and
/// `identity()` is the neutral element. Associativity is what makes the chunked merge sound; the
/// table does not require commutativity, and nothing here assumes it.
pub trait Reduce<V> {
    /// The accumulator the whole-table reduce threads through the table's live slots.
    type Acc;

    /// The neutral element the reduce starts from: the incumbent's `id`.
    fn identity(&self) -> Self::Acc;

    /// The contribution of one live value: the incumbent's `residual`.
    fn residual(&self, value: &V) -> Self::Acc;

    /// The accumulator's binary operation, `combine(acc, next)`: the incumbent's `combine`.
    fn combine(&self, acc: Self::Acc, next: Self::Acc) -> Self::Acc;
}

/// One affine step over `R[x]/(x^2 - sigma)`: a coefficient cell and an injection cell.
///
/// This is `Apeiron.Trichotomy.AnyRing.RingAffineStep R` at `proof/Apeiron.lean` in the pinned
/// canonical corpus, field for field: `coefficient : RingCell R` and `injection : RingCell R`. The
/// methods below are [`ringAffineId`](AnyRing.ringAffineId), [`ringAffineApply`](AnyRing.ringAffineApply)
/// and [`ringAffineCompose`](AnyRing.ringAffineCompose) term for term; the linking theorems are
/// `AnyRing.ringAffineCompose_apply` and `AnyRing.ringAffineFold_apply`, proved in
/// `proof/corpus/TableRefinement/EncircleFold.lean`.
///
/// Every method takes the dial as an argument and none of them constrains it. `RingAffineStep` and
/// its three operations are stated in the corpus over `[CommRing R]` with a bare `sigma : R`, so the
/// laws below hold on all three faces with no sign hypothesis. The corpus theorems that *do* carry a
/// face, `AnyRing.ringUpper_ringAffineFold` and `AnyRing.ringLower_ringAffineFold`, each take
/// `hr : r ^ 2 = sigma` and project onto a channel; this reduce never projects, so no sub-step of it
/// forces a face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AffineStep<R> {
    /// The cell the step multiplies by: an ordinary weight, not a rotor.
    pub coefficient: Cell<R>,
    /// The cell the step adds after multiplying.
    pub injection: Cell<R>,
}

impl<R: Ring> AffineStep<R> {
    /// The identity step: coefficient one, injection zero.
    ///
    /// `AnyRing.ringAffineId R`.
    #[inline]
    #[must_use]
    pub fn id() -> Self {
        Self {
            coefficient: Cell::one(),
            injection: Cell::zero(),
        }
    }

    /// Sends `z` to `coefficient * z + injection`.
    ///
    /// `AnyRing.ringAffineApply sigma step z`. Temporal `O(1)`: two `Cell` operations, each a fixed
    /// number of scalar operations.
    #[inline]
    #[must_use]
    pub fn apply(self, sigma: Sigma, z: Cell<R>) -> Cell<R> {
        self.coefficient.mul(sigma, z).add(self.injection)
    }

    /// Applies `inner` first, then `self`.
    ///
    /// `AnyRing.ringAffineCompose sigma outer inner`: the composed coefficient is
    /// `outer.coefficient * inner.coefficient` and the composed injection carries the inner
    /// injection forward by the outer coefficient, `outer.coefficient * inner.injection +
    /// outer.injection`. Temporal `O(1)`, two `Cell` operations and one addition.
    #[inline]
    #[must_use]
    pub fn compose(self, sigma: Sigma, inner: Self) -> Self {
        Self {
            coefficient: self.coefficient.mul(sigma, inner.coefficient),
            injection: self.coefficient.mul(sigma, inner.injection).add(self.injection),
        }
    }
}

/// The [`Reduce`] instance the corpus fold law is about: every stored cell becomes the injection of
/// the step that translates by it, and the steps compose chronologically.
///
/// With `residual(v) = <1, v>` the applied point is `1 * z + v`, the translate. `combine(acc, next)`
/// is `next.compose(sigma, acc)`, so the step the table reaches later is composed on the *outside*,
/// which is the order `AnyRing.ringAffineFold` uses: `steps.foldl (fun acc step =>
/// ringAffineCompose sigma step acc) (ringAffineId R)`. The two agree term for term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AffineFoldStep {
    /// The dial the whole reduce runs at. One value for the whole table, not one per slot.
    pub sigma: Sigma,
}

impl AffineFoldStep {
    /// The corpus fold of `steps`, composed in list order.
    ///
    /// `AnyRing.ringAffineFold sigma steps`. Temporal `O(steps.len())` in the list length, `O(1)`
    /// retained accumulator steps: one `compose` per element, each `O(1)`.
    #[inline]
    #[must_use]
    pub fn fold(self, steps: &[AffineStep<i64>]) -> AffineStep<i64> {
        steps
            .iter()
            .fold(AffineStep::id(), |acc, step| step.compose(self.sigma, acc))
    }
}

impl Reduce<Cell<i64>> for AffineFoldStep {
    type Acc = AffineStep<i64>;

    #[inline]
    fn identity(&self) -> AffineStep<i64> {
        AffineStep::id()
    }

    #[inline]
    fn residual(&self, value: &Cell<i64>) -> AffineStep<i64> {
        AffineStep {
            coefficient: Cell::one(),
            injection: *value,
        }
    }

    #[inline]
    fn combine(&self, acc: AffineStep<i64>, next: AffineStep<i64>) -> AffineStep<i64> {
        next.compose(self.sigma, acc)
    }
}

/// The same chronology with a coefficient read from the value, so the witness is not commutative.
///
/// [`AffineFoldStep`]'s residual is always `<1, v>`, and composition between two unit-coefficient
/// steps carries only injections: `compose(<1,a>, <1,b>)` has injection `b + a`, so it is
/// commutative and a reduce over it agrees with its own mirror image. That is a fact about *that*
/// witness, not about `ringAffineCompose`, and a test built only on it cannot see a `combine` whose
/// arguments are swapped. The same holds for any residual whose coefficient is an affine function of
/// the injection: `compose(s, t)` and `compose(t, s)` agree exactly when `s.coefficient - 1` is a
/// ring multiple of `s.injection`, so `⟨1 + v, v⟩` is commutative for the same reason.
///
/// This witness reads the coefficient from the value's first slot and the injection from the whole
/// cell, which are independent, so `(s.c - 1) * t.d = a_s * t.d` and `(t.c - 1) * s.d = a_t * s.d`
/// are `⟨a_s a_t, a_s b_t⟩` and `⟨a_t a_s, a_t b_s⟩` -- equal only when the stored cells are
/// proportional. `compose` is therefore the non-commutative operation the corpus states, and a
/// reversed `combine` is observable.
///
/// `residual(v) = <1 + v.a, v>`: the step scales by one more than the stored first slot and then
/// translates by the whole stored cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AffineMixStep {
    /// The dial the whole reduce runs at. One value for the whole table, not one per slot.
    pub sigma: Sigma,
}

impl AffineMixStep {
    /// The corpus fold of `steps`, composed in list order, at this witness's chronology.
    ///
    /// The same shape as [`AffineFoldStep::fold`]; the two differ only in the step list, because
    /// both compose each later step on the outside. Kept separate so each witness's chronology is
    /// stated against a fold built from the same corpus operation.
    #[inline]
    #[must_use]
    pub fn fold(self, steps: &[AffineStep<i64>]) -> AffineStep<i64> {
        steps
            .iter()
            .fold(AffineStep::id(), |acc, step| step.compose(self.sigma, acc))
    }
}

impl Reduce<Cell<i64>> for AffineMixStep {
    type Acc = AffineStep<i64>;

    #[inline]
    fn identity(&self) -> AffineStep<i64> {
        AffineStep::id()
    }

    #[inline]
    fn residual(&self, value: &Cell<i64>) -> AffineStep<i64> {
        AffineStep {
            coefficient: Cell::of(value.a.wrapping_add(1), 0),
            injection: *value,
        }
    }

    #[inline]
    fn combine(&self, acc: AffineStep<i64>, next: AffineStep<i64>) -> AffineStep<i64> {
        next.compose(self.sigma, acc)
    }
}

impl<K, V, H: Home<K>, const N: usize, R: KeyRep<K, H>> Table<K, V, H, N, R> {
    /// Reduces every live value of the whole table into one accumulator.
    ///
    /// This is `fold hashtrinity/src/lib.rs:891` (`Table::encircle`) reduced to the body both of its
    /// branches share. Live slots are visited in resident-then-spill storage order -- the order
    /// `Table::live_locations` yields -- so the result depends on the storage layout and not on
    /// any traversal the caller chose.
    ///
    /// Temporal `O(lanes + sum(shard slot counts))`: one liveness test and, on a live slot, one
    /// `residual` and one `combine`. Spatial `O(1)`: one accumulator, no allocation, no
    /// intermediate list.
    #[expect(
        clippy::explicit_iter_loop,
        reason = "the implicit `for x in &vec` form reaches the extraction through \
                  `SharedAVec.Insts.CoreIterTraitsCollectIntoIteratorSharedATIter.into_iter`, which the \
                  pinned Aeneas Lean model does not define; `proof/regen.sh` fail-closed audit \
                  rejects it as an unlisted axiom. The explicit `.iter()` is the form \
                  `table::live_locations_of` already uses and is inside the closed cone."
    )]
    pub fn reduce_all<T: Reduce<V>>(&self, reduce: &T) -> T::Acc {
        let mut acc = reduce.identity();
        for slot in self.resident.slots.iter() {
            if slot.live {
                acc = reduce.combine(acc, reduce.residual(&slot.value));
            }
        }
        for region in self.shards.iter() {
            for slot in region.slots.iter() {
                if slot.live() {
                    acc = reduce.combine(acc, reduce.residual(&slot.value));
                }
            }
        }
        acc
    }

    /// The stored values of the whole table, in resident-then-spill storage order.
    ///
    /// The residual stream [`Table::reduce_all`] folds, exposed so a caller can build its own
    /// accumulator over the same order without re-deriving it. Temporal `O(lanes + slots)`;
    /// spatial `O(live)`.
    #[must_use]
    #[expect(
        clippy::explicit_iter_loop,
        reason = "the implicit `for x in &vec` form reaches the extraction through \
                  `SharedAVec.Insts.CoreIterTraitsCollectIntoIteratorSharedATIter.into_iter`, which the \
                  pinned Aeneas Lean model does not define; `proof/regen.sh` fail-closed audit \
                  rejects it as an unlisted axiom. The explicit `.iter()` is the form \
                  `table::live_locations_of` already uses and is inside the closed cone."
    )]
    pub fn values_in_order(&self) -> Vec<&V> {
        let mut values = Vec::new();
        for slot in self.resident.slots.iter() {
            if slot.live {
                values.push(&slot.value);
            }
        }
        for region in self.shards.iter() {
            for slot in region.slots.iter() {
                if slot.live() {
                    values.push(&slot.value);
                }
            }
        }
        values
    }
}