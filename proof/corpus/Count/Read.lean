-- Comment-only: 18 dangling/stale citations corrected.
import Count.Rep
import Growable.Search

/-!
# The count face's read refinement and aggregate

The extracted read body of the count face is `apeiron_word::word_stored_apply_get`
(`proof/ApeironWord.lean:26978-26988`, source `proof/extract/lib.rs:450-452`), which calls the real
`table.Table.apply_get`.  It is a function of an already-computed `table.SearchDecision`; the decision
itself is not extracted here, so its bridge contract is a premise.

This module states and proves the refinement of that extracted body against the canonical object, and
the count face's aggregate over it:

* `CountDecisionCorrect` is the count face's name for the decision contract
  `Apeiron.GrowableSearch.SearchDecisionCorrect` at its own types.
* `searchDecisionCorrect_of_countRep` proves the two canonical-`get` fields of that contract
  (`foundDenotes`, `absentDenotesDefault`) from `CountRep`: the live-representative clause is
  `count_get_of_live_representative` and the refusal clause is `count_get_default_of_absent`.  The
  wrapped contract already carries those two fields, and `AbsenceCertificate.coverage` mentions
  `Apeiron.Table.get` independently, so this is a statement that the two fields follow from the
  representation, not that the contract omits them.
* `count_get_refines_canonical_get` gives the refinement itself: the extracted read body **is**
  `Apeiron.Table.get` of the represented map, on the admitted domain and outside it alike.
* `countMerge_is_generated` names the merge the extracted bodies call: the generated monomorphic
  callback `table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge` computes `CountMerge`, the additive
  (`wrapping_add`) merge, new-first (`Apeiron.GrowableSearch.wordMerge_newFirst`).
* `count_get_build` gives the aggregate's read at an arbitrary merge and default, over the count stream:
  reading a map built from `countStream keys` at a key is the fold of the merge over that key's `1`s.
* `count_build_perm` gives the aggregate's schedule with **no** commutativity or associativity: the
  count stream carries the constant `1`, so any two orderings of the keys build the same map under
  every merge and default (`Apeiron.Table.build_eq_of_key_streams_eq`).  `canonicalCount_eq_of_perm`
  is that at the count merge and zero default.
* `countAggregate_get`, `countMerge_comm`, `countMerge_assoc` and `countAggregate_eq_of_perm` are the
  merge-algebra half: the count aggregate over an **arbitrary** value stream, its reading law, and its
  gauge invariance under a permutation of that stream (`Apeiron.Table.build_gauge_invariance`).
* `count_get_agrees_with_countAggregate_off_stream`, `countAggregate_get_zero_off_stream` and
  `count_get_agrees_with_canonical_build_off_stream` link the extracted read to the built map away from
  the build stream, where both sides read the refusal default.
* `count_get_refines_aggregate_get` is **conditional**: it composes the refinement with a named
  hypothesis `haggregate` that the represented map *is* the aggregate.  See the residual below.

## The residual: the executed body is not the extracted body

The count face's **executed** body is `crate::hashtrinity::FreqMap::count_par`
(`src/hashtrinity/freq.rs:21`), followed by `FreqMap::get_each` (`src/hashtrinity/mod.rs:785`).  Neither is
extracted:
`proof/ApeironWord.lean` carries no `FreqMap`, `count_par` or `get_each` symbol, `proof/extract/lib.rs`
names no `freq` module, and `count_par` drives an OS thread pool over raw-pointer writes and closures,
which the extraction fragment does not model.  So no `word_count_get` exists and none is invented here;
that symbol was a fabricated precedent in the earlier count work and is not restored.  What is proved is
the strongest true statement over what **is** reachable: the generic extracted read body
`word_stored_apply_get` refines the canonical read, and the canonical count aggregate is
`Apeiron.Table.build` with merge `new + old` — including its partition schedule.  The missing symbol — a
refinement of `FreqMap::count_par`/`get_each` itself, and the per-step whole-table insert-preservation
that `Growable/BuildRefine.foldList_logical` carries as `hstep` and never discharges — is the residual,
named, not assumed.

## `build`: subject, and what is claimed

The count face **does** have a `build` subject: the extraction carries the generated builders
`table.Table.build_at` (`proof/ApeironWord.lean:26597`), `table.Table.build_at_sequential` (`:26178`)
and their loops, and the count face's parallel arm calls the real `Table::build_at`
(`SpatialTable::build_at_parallel_phased`, `examples/reg1/adapters/spatial.rs:772` at `c9a2d0c^`, whose
short-input branch is `build_at_sequential`).  `Growable/BuildRefine.foldList_logical`
carries the generated fold to `Apeiron.Table.build`, modulo the per-step `hstep` premise that is never
discharged.  No theorem here claims the extracted `build_at` loop produces `countAggregate`; the
aggregate theorems are statements about the canonical object at the abstraction the crate exposes, and
the one that would compose them with the extracted read is stated as the conditional
`count_get_refines_aggregate_get`.
-/

namespace Apeiron.CountFace

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

noncomputable local instance : DecidableEq CountKey := Classical.decEq _

/-- The count merge: new-first addition, the merge `Apeiron.Table.insertWith` applies, which takes the
new value first (`insertWith f k v m = fun x => if x = k then f v (m x) else m x`,
`Apeiron/Table/Basic.lean:80-81`).  At `I64` the addition is the **extracted** wrapping addition
`core.num.I64.wrapping_add`, the monomorphic callback the generated merge
`table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge` computes (`Growable/Search.lean`,
`wordMerge_newFirst`); the crate's checked counter (`src/table/count.rs`, `CountRefusal::CountOverflow`)
refuses exactly where this wraps. -/
def CountMerge (new old : CountValue) : CountValue := core.num.I64.wrapping_add new old

/-- The count stream: one `1` at each key.  A key's subsequence of values is therefore the constant list
of its multiplicity, which is what makes the count aggregate order-independent. -/
def countStream (keys : List CountKey) : List (CountKey × CountValue) :=
  keys.map (fun k => (k, 1#i64))

/-- **The count face's aggregate over an arbitrary value stream.**  `Apeiron.Table.build` at the count
merge and the zero default.  The selected serial arm builds the count one `insert_with(&merge, k, 1)` at
a time (`examples/reg1/adapters/spatial.rs:594-596` at `c9a2d0c^`) and the parallel arm builds the same
stream at once (`examples/reg1/adapters/spatial.rs:761-772` at `c9a2d0c^`);
this is the canonical object that stream denotes, stated over an arbitrary stream so the merge algebra
below applies to it.  `noncomputable` because `Apeiron.Table.insertWith` compares keys through this
package's `DecidableEq CountKey`, which is `Classical.decEq`. -/
noncomputable def countAggregate (kvs : List (CountKey × CountValue)) : CanonicalCount :=
  Apeiron.Table.build CountMerge 0#i64 kvs

/-- **The canonical count map: the aggregate over the count stream.**  `countStream keys` is one `1` at
each key, so this is `Apeiron.Table.build` of the count merge from the zero default over the face's
increment stream; it is `countAggregate` at the count stream, named once for the schedule theorems
below. -/
noncomputable def canonicalCount (keys : List CountKey) : CanonicalCount :=
  countAggregate (countStream keys)

/-- **The count face's merge is the extracted generated merge.**  The monomorphic callback the generated
table invokes, `table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge`, computes `CountMerge` new-first
(`Apeiron.GrowableSearch.wordMerge_newFirst`).  So the count aggregate's merge is the extraction's own
`i64 +`, not an invented operation. -/
theorem countMerge_is_generated (new old : CountValue) :
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge () new old = ok (CountMerge new old) := by
  simpa [CountMerge] using Apeiron.GrowableSearch.wordMerge_newFirst new old

/-- **Reading a built map is the merge fold over exactly that key's occurrences.**  `Apeiron.Table.get_build_acc`
(`Apeiron/Table/Basic.lean:134`) at the constant default: reading
`Apeiron.Table.build f d (countStream keys)` at `k` folds `f` new-first over the values of `keys` at `k`,
which for the count stream are the `1`s of its multiplicity. -/
theorem count_get_build (f : CountValue → CountValue → CountValue) (d : CountValue)
    (keys : List CountKey) (k : CountKey) :
    Apeiron.Table.get (Apeiron.Table.build f d (countStream keys)) k
      = (((countStream keys).filter (fun kv => kv.1 = k)).map Prod.snd).foldl
          (fun acc v => f v acc) d := by
  have h := Apeiron.Table.get_build_acc f k (countStream keys) (fun _ => d)
  simpa [Apeiron.Table.build, Apeiron.Table.get] using h

/-- The count stream's values at a key are the constant `1`s of its multiplicity. -/
private theorem countStream_filter_snd (keys : List CountKey) (k : CountKey) :
    ((countStream keys).filter (fun kv => kv.1 = k)).map Prod.snd
      = List.replicate ((keys.filter (fun k' => k' = k)).length) 1#i64 := by
  unfold countStream
  rw [List.filter_map, List.map_map]
  simp [Function.comp_def]

/-- **Any two orderings of the key stream build the same count map, for every merge and every default.**
The count stream carries the constant `1`, so a rearrangement preserves each key's value subsequence as
a list, and `Apeiron.Table.build_eq_of_key_streams_eq` (`Apeiron/Table/Basic.lean:169`) applies with no
commutativity or associativity of the merge.  This is the licence `FreqMap::count_par`'s
partition-and-scatter schedule needs: partition the stream by home bucket, aggregate each bucket
independently, and the per-bucket maps denote the same count function as the whole. -/
theorem count_build_perm {f : CountValue → CountValue → CountValue} (d : CountValue)
    {keys₁ keys₂ : List CountKey} (h : keys₁.Perm keys₂) :
    Apeiron.Table.build f d (countStream keys₁) = Apeiron.Table.build f d (countStream keys₂) := by
  refine Apeiron.Table.build_eq_of_key_streams_eq f d ?_
  intro k
  rw [countStream_filter_snd keys₁ k, countStream_filter_snd keys₂ k,
    (h.filter (fun k' => k' = k)).length_eq]

/-- **The canonical count of a stream is invariant under its partition.**  Specialising `count_build_perm`
at the count merge and default: the canonical count map (`canonicalCount`) computed on one ordering of
the keys equals the canonical count map computed on any permutation — the count `FreqMap::count_par`
returns per bucket, read off the whole stream.  Unlike `countAggregate_eq_of_perm` this needs no
commutativity or associativity of the merge. -/
theorem canonicalCount_eq_of_perm {keys₁ keys₂ : List CountKey} (h : keys₁.Perm keys₂) :
    canonicalCount keys₁ = canonicalCount keys₂ := by
  simpa only [canonicalCount, countAggregate] using count_build_perm (f := CountMerge) 0#i64 h

/-- **Reading the count aggregate at a key is the additive fold of that key's increments.**  Exactly
`Apeiron.Table.get_build_acc` (`Apeiron/Table/Basic.lean:134`) at the count face's merge and zero
default over an **arbitrary** value stream: everything written at other keys is invisible at `key`, and
the merge fold over the key's own values is the count.  This is the reading law of the count face against
the canonical object; `count_get_build` is the same law restated at the count stream with the merge and
default general. -/
theorem countAggregate_get (kvs : List (CountKey × CountValue)) (key : CountKey) :
    Apeiron.Table.get (countAggregate kvs) key
      = ((kvs.filter (fun kv => kv.1 = key)).map Prod.snd).foldl
          (fun acc v => CountMerge v acc) 0#i64 :=
  Apeiron.Table.get_build_acc CountMerge key kvs (fun _ => 0#i64)

/-- **The count face's merge is commutative on the extracted value type.**  `wrapping_add` on `BitVec 64`
is `BitVec.add`, whose commutativity is `BitVec.add_comm`.  Instantiating at `CountMerge` is what lets
`countAggregate_eq_of_perm` apply, and it is what makes the count independent of the order in which the
parallel arms happen to land their increments. -/
theorem countMerge_comm (a b : CountValue) : CountMerge a b = CountMerge b a := by
  simp only [CountMerge, core.num.I64.wrapping_add, IScalar.wrapping_add]
  exact congrArg _ (BitVec.add_comm a.bv b.bv)

/-- **The count face's merge is associative on the extracted value type.**  `BitVec.add_assoc`. -/
theorem countMerge_assoc (a b c : CountValue) :
    CountMerge (CountMerge a b) c = CountMerge a (CountMerge b c) := by
  simp only [CountMerge, core.num.I64.wrapping_add, IScalar.wrapping_add]
  exact congrArg _ (BitVec.add_assoc a.bv b.bv c.bv)

/-- **The count is gauge-invariant: a rearranged increment stream counts the same.**
`Apeiron.Table.build_gauge_invariance` (`Apeiron/Table/Basic.lean:191`) at the count face's merge, whose
hypotheses `countMerge_comm` and `countMerge_assoc` discharge.  This is the licence the count face's
parallel and sharded arms run on for an **arbitrary** increment stream: partition the stream by home
address, count each shard, and the table denoted is the same function.  `canonicalCount_eq_of_perm`
covers the count stream without these hypotheses, through the constant `1`. -/
theorem countAggregate_eq_of_perm {kvs₁ kvs₂ : List (CountKey × CountValue)}
    (hperm : kvs₁.Perm kvs₂) :
    countAggregate kvs₁ = countAggregate kvs₂ :=
  Apeiron.Table.build_gauge_invariance countMerge_comm countMerge_assoc 0#i64 hperm

/-- **The count face's decision contract.**  This is `Apeiron.GrowableSearch.SearchDecisionCorrect`
(`Growable/Search.lean:168`) at the count face's types under its own name.  Its `search` field is the
full decision contract, including the two canonical-`get` fields `foundDenotes` and
`absentDenotesDefault`; `searchDecisionCorrect_of_countRep` proves those two from `CountRep`.
`AbsenceCertificate.coverage` (`Growable/Search.lean:159`) also mentions `Apeiron.Table.get`, so the
decision contract is not free of the canonical read; what this wrapper provides is the count face's name
for the contract, not a contract with the read clauses removed. -/
structure CountDecisionCorrect (state : StoredTable) (query : CountKey)
    (decision : table.SearchDecision) (direct : CountKey → Option CountLocation)
    (live : CountKey → Prop) (count : CanonicalCount)
    (order prefixOrder sweep checked : List CountLocation) : Prop where
  /-- The extracted decision's complete contract. -/
  search : Apeiron.GrowableSearch.SearchDecisionCorrect state CountDecode count query decision direct
    order prefixOrder sweep checked

/-- **`CountRep` proves the two canonical-`get` fields of the decision contract.**  The live-representative
clause `foundDenotes` is `count_get_of_live_representative` and the refusal clause `absentDenotesDefault`
is `count_get_default_of_absent`.  The wrapped contract `hdec.search` already carries both fields, so this
is a consistency statement: it shows the two fields follow from the representation alone, not that the
hypothesis omits them. -/
theorem searchDecisionCorrect_of_countRep
    (state : StoredTable) (query : CountKey) (decision : table.SearchDecision)
    (direct : CountKey → Option CountLocation) (live : CountKey → Prop) (count : CanonicalCount)
    (order prefixOrder sweep checked : List CountLocation)
    (hrep : CountRep state direct live count) (hdec : CountDecisionCorrect state query decision direct
      live count order prefixOrder sweep checked) :
    Apeiron.GrowableSearch.SearchDecisionCorrect state CountDecode count query decision direct order
      prefixOrder sweep checked := by
  refine { hdec.search with
    foundDenotes := ?_
    absentDenotesDefault := ?_ }
  · intro location value hentry
    exact count_get_of_live_representative state direct live count hrep location query value hentry
  · exact count_get_default_of_absent state direct live count hrep query

/-- **The extracted count-face read body refines the canonical `Apeiron.Table.get`.**  For every decision
the contract admits, at every face and every key,

    word_stored_apply_get state decision = ok (Apeiron.Table.get count query)

where `count` is the canonical map the extracted state represents under `CountRep`.  The extracted body is
`proof/ApeironWord.lean:26978-26988` over the real `table.Table.apply_get`, and the canonical object is
`Apeiron.Table.get` at `Apeiron/Table/Basic.lean:71`.  The proof uses
`Apeiron.GrowableSearch.word_stored_search_decision_simulates` at the contract
`searchDecisionCorrect_of_countRep` builds from the representation, whose two canonical-`get` fields come
from `CountRep`. -/
theorem count_get_refines_canonical_get
    (state : StoredTable) (query : CountKey) (decision : table.SearchDecision)
    (direct : CountKey → Option CountLocation) (live : CountKey → Prop) (count : CanonicalCount)
    (order prefixOrder sweep checked : List CountLocation)
    (hrep : CountRep state direct live count) (hdec : CountDecisionCorrect state query decision direct
      live count order prefixOrder sweep checked) :
    word_stored_apply_get state decision = ok (Apeiron.Table.get count query) :=
  Apeiron.GrowableSearch.word_stored_search_decision_simulates state query decision direct order
    prefixOrder sweep checked
    (searchDecisionCorrect_of_countRep state query decision direct live count order prefixOrder sweep
      checked hrep hdec)

/-- **A fold of `Apeiron.Table.insertWith` over a stream that never mentions the key reads the starting
map's value there.**  `Apeiron.Table.get_insertWith_ne` (`Apeiron/Table/Basic.lean:97`) leaves a read at
every other key untouched, so each step of the fold is invisible at `key`.  Stated over the canonical
`insertWith`/`get` alone, at an arbitrary starting map. -/
private theorem foldl_get_of_stream_ne {K V : Type} [DecidableEq K] (f : V → V → V) :
    ∀ (D : Apeiron.Table.HMap K V) (kvs : List (K × V)) (key : K),
      (∀ kv, kv ∈ kvs → kv.1 ≠ key) →
      Apeiron.Table.get (kvs.foldl (fun m kv => Apeiron.Table.insertWith f kv.1 kv.2 m) D) key
        = Apeiron.Table.get D key := by
  intro D kvs
  induction kvs generalizing D with
  | nil =>
      intro key _
      simp [List.foldl, Apeiron.Table.get]
  | cons kv rest ih =>
      intro key hstream
      rw [List.foldl_cons]
      have hne : key ≠ kv.1 := Ne.symm (hstream kv (List.mem_cons_self))
      have hrest := ih (Apeiron.Table.insertWith f kv.1 kv.2 D) key
        (fun q hq => hstream q (by simp [hq]))
      rw [hrest, Apeiron.Table.get_insertWith_ne f kv.1 kv.2 D hne]

/-- **A canonical build over a stream that never mentions the key reads the default there.**
`Apeiron.Table.build` (`Apeiron/Table/Basic.lean:121`) is the fold of `Apeiron.Table.insertWith` over the
stream from the constant default, so `foldl_get_of_stream_ne` reads it as the starting map's value. -/
private theorem build_get_of_stream_ne {K V : Type} [DecidableEq K] (f : V → V → V) (default : V) :
    ∀ (kvs : List (K × V)) (key : K), (∀ kv, kv ∈ kvs → kv.1 ≠ key) →
      Apeiron.Table.get (Apeiron.Table.build f default kvs) key = default := by
  intro kvs key hstream
  have h := foldl_get_of_stream_ne f (fun _ => default) kvs key hstream
  simpa only [Apeiron.Table.build, Apeiron.Table.get] using h

/-- **The count face's read agrees with the counting aggregate away from its stream.**  Under `CountRep`, a
key the extracted table does not represent as live is read by the extracted read body as the counting
aggregate reads it: the extracted default, which at the count face's zero default is the count zero.  This
links `countAggregate` at `get` to the represented map without restating the canonical `get_build_acc`,
whose `List.filter` is stated over the canonical corpus's own `DecidableEq` instance rather than this
package's — absence is the default in both, so a build over an unrelated stream is invisible to the
refusal.  `hzero` is an explicit hypothesis rather than inferred: `countAggregate` is the build from
`0#i64`, so the agreement is at the zero default and not at an arbitrary `state.default`. -/
theorem count_get_agrees_with_countAggregate_off_stream
    (state : StoredTable) (direct : CountKey → Option CountLocation) (live : CountKey → Prop)
    (count : CanonicalCount) (hrep : CountRep state direct live count) (key : CountKey)
    (habsent : Apeiron.GrowableSearch.Absent state CountDecode key) (hzero : state.default = 0#i64)
    (kvs : List (CountKey × CountValue)) (hstream : ∀ kv, kv ∈ kvs → kv.1 ≠ key) :
    Apeiron.Table.get (countAggregate kvs) key = Apeiron.Table.get count key := by
  have hagg : Apeiron.Table.get (countAggregate kvs) key = 0#i64 := by
    exact build_get_of_stream_ne CountMerge 0#i64 kvs key hstream
  calc Apeiron.Table.get (countAggregate kvs) key = 0#i64 := hagg
    _ = state.default := hzero.symm
    _ = Apeiron.Table.get count key :=
      (count_get_default_of_absent state direct live count hrep key habsent).symm

/-- **At the zero default, a key outside the aggregate's stream counts zero on both sides.**  The aggregate
side of `count_get_agrees_with_countAggregate_off_stream` taken at the extracted default: the key's count
is zero through the counting aggregate and through the extracted read.  Named separately from
`Apeiron.CountFace.count_get_zero_of_absent_at_zero_default` because that one states the count zero on the
represented-map side only. -/
theorem countAggregate_get_zero_off_stream
    (state : StoredTable) (direct : CountKey → Option CountLocation) (live : CountKey → Prop)
    (count : CanonicalCount) (hrep : CountRep state direct live count) (key : CountKey)
    (hzero : state.default = 0#i64)
    (habsent : Apeiron.GrowableSearch.Absent state CountDecode key)
    (kvs : List (CountKey × CountValue)) (hstream : ∀ kv, kv ∈ kvs → kv.1 ≠ key) :
    Apeiron.Table.get (countAggregate kvs) key = 0#i64 :=
  (count_get_agrees_with_countAggregate_off_stream state direct live count hrep key habsent hzero kvs
    hstream).trans
    (count_get_zero_of_absent_at_zero_default state direct live count hrep key hzero habsent)

/-- **The count face's read agrees with the canonical `Apeiron.Table.build` away from its stream.**  Under
`CountRep`, a key that no physical location of the extracted table decodes to reads the same value through
the extracted read body and through a map built by the canonical constructor `Apeiron.Table.build`
(`Apeiron/Table/Basic.lean:121`) from any merge, any stream that does not mention the key, and the
extracted default.  This is the `get_build_acc` content at the count face's refusal value, proved directly
from the canonical `build`, `insertWith` and `get` because the canonical `get_build_acc` is stated over the
corpus's own `List.filter`, whose `DecidableEq` instance is not this package's: absence is the default in
both, so a build over an unrelated stream is invisible to the refusal. -/
theorem count_get_agrees_with_canonical_build_off_stream
    (state : StoredTable) (direct : CountKey → Option CountLocation) (live : CountKey → Prop)
    (count : CanonicalCount) (hrep : CountRep state direct live count) (key : CountKey)
    (habsent : Apeiron.GrowableSearch.Absent state CountDecode key)
    (f : CountValue → CountValue → CountValue) (kvs : List (CountKey × CountValue))
    (hstream : ∀ kv, kv ∈ kvs → kv.1 ≠ key) :
    Apeiron.Table.get (Apeiron.Table.build f state.default kvs) key
      = Apeiron.Table.get count key :=
  (build_get_of_stream_ne f state.default kvs key hstream).trans
    (count_get_default_of_absent state direct live count hrep key habsent).symm

/-- **The count face's read at a key equals the counting aggregate at that key — CONDITIONAL.**  Under
`CountRep`, given the hypothesis `haggregate` that the represented map is the aggregate, the extracted read
body and the counting aggregate agree at every key:

    word_stored_apply_get state decision = ok (Apeiron.Table.get (countAggregate kvs) query)

`haggregate` is the residual this module does not discharge: the per-step whole-table
insert-preservation that `Growable/BuildRefine.foldList_logical` (`Growable/BuildRefine.lean:346`) carries
as `hstep` and never discharges.  Without `haggregate` the statement is not a link from the extracted body
to the aggregate; it is a conditional refinement, and it is labelled one.  The unconditional statements
that do connect the extracted read to a built map are
`count_get_agrees_with_canonical_build_off_stream` and `count_get_agrees_with_countAggregate_off_stream`,
both of which are at a refused key. -/
theorem count_get_refines_aggregate_get
    (state : StoredTable) (query : CountKey) (decision : table.SearchDecision)
    (direct : CountKey → Option CountLocation) (live : CountKey → Prop) (count : CanonicalCount)
    (order prefixOrder sweep checked : List CountLocation) (kvs : List (CountKey × CountValue))
    (hrep : CountRep state direct live count)
    (hdec : CountDecisionCorrect state query decision direct live count order prefixOrder sweep checked)
    (haggregate : ∀ key, Apeiron.Table.get count key = Apeiron.Table.get (countAggregate kvs) key) :
    word_stored_apply_get state decision = ok (Apeiron.Table.get (countAggregate kvs) query) := by
  rw [count_get_refines_canonical_get state query decision direct live count order prefixOrder sweep
    checked hrep hdec, haggregate query]

/-! ## Headline audit rows

Each row is the closed-cone gate of `Apeiron.AuditClosed`: `#audit` collects the declaration's transitive
axiom dependencies and fails the build on anything outside `{propext, Classical.choice, Quot.sound}`, so a
`sorryAx` anywhere in the cone is a build failure rather than a line in a log.  They sit in this module
rather than in `AxiomAudit/Core.lean` so they run in the scoped build of `Count.Read` itself. -/

#audit Apeiron.CountFace.countMerge_is_generated
#audit Apeiron.CountFace.count_get_build
#audit Apeiron.CountFace.count_build_perm
#audit Apeiron.CountFace.canonicalCount_eq_of_perm
#audit Apeiron.CountFace.countAggregate_get
#audit Apeiron.CountFace.countMerge_comm
#audit Apeiron.CountFace.countMerge_assoc
#audit Apeiron.CountFace.countAggregate_eq_of_perm
#audit Apeiron.CountFace.searchDecisionCorrect_of_countRep
#audit Apeiron.CountFace.count_get_refines_canonical_get
#audit Apeiron.CountFace.count_get_agrees_with_countAggregate_off_stream
#audit Apeiron.CountFace.countAggregate_get_zero_off_stream
#audit Apeiron.CountFace.count_get_agrees_with_canonical_build_off_stream
#audit Apeiron.CountFace.count_get_refines_aggregate_get

end Apeiron.CountFace
