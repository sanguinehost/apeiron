import Map.Rep
import Growable.Search

/-!
# The map face's `get` refinement

The extracted read body of the map face is `apeiron_word::word_stored_apply_get`
(`proof/ApeironWord.lean:26956-26968`, source `proof/extract/lib.rs:450-452`), which calls the real
`table.Table.apply_get`.  It is a function of an already-computed `table.SearchDecision`; the
decision itself is not extracted here, so its bridge contract is a premise.

This module states and proves the refinement of that extracted body against the canonical object,
and it does so in the form the map face needs:

* `MapDecisionCorrect` is `Apeiron.GrowableSearch.SearchDecisionCorrect` **with its two canonical-`get`
  clauses removed** — `foundDenotes` and `absentDenotesDefault`, the only two fields that mention
  `Apeiron.Table.get`.
* `searchDecisionCorrect_of_mapRep` shows the map face's representation **supplies both**: from
  `MapRep` alone the live-representative clause is `map_get_of_live_representative` and the refusal
  clause is `map_get_default_of_absent`.  The semantic content of the decision contract is therefore
  carried by the representation, not postulated alongside it.
* `map_get_refines_canonical_get` then gives the refinement itself: the extracted read body **is**
  `Apeiron.Table.get` of the represented map, on the admitted domain and outside it alike, under a
  contract that never mentions the canonical read.
* `map_get_agrees_with_canonical_build_off_stream` links the same read to the canonical constructor
  `Apeiron.Table.build` (`Apeiron/Table/Basic.lean:121`) at `get`: away from a build stream the
  refusal default is the built map's default.

## `build`: subject, and what is claimed

The map face **does** have a `build` subject.  The extraction carries the generated builders
`table.Table.build_at` (`proof/ApeironWord.lean:26578`) and its loops, and
`Growable/BuildRefine.lean` already links the generated fold to `Apeiron.Table.build`
(`canonical_build_eq_buildFold`, `canonical_build_eq_of_key_streams_eq`) at the word entry type.  No
`word_*_build` monomorph exists in the extraction, so no map-face monomorphic build body is claimed
here.  `Growable/BuildRefine.lean` does not elaborate against the current extraction and is not
imported; the linkage below is taken from `Apeiron.Table.Basic` directly and is a `get`-level
statement, not a claim that the generated fold is `build`.  Whether the extracted generated fold
preserves `MapRep` is the per-step premise `Growable.BuildRefine.foldList_logical` already carries
and never discharges.
-/

namespace Apeiron.MapFace

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

noncomputable local instance : DecidableEq MapKey := Classical.decEq _

/-- **The map face's decision contract, with the canonical read removed.**
`SearchDecisionCorrect` (`Growable/Search.lean:174`) states its two `Apeiron.Table.get` clauses
(`foundDenotes`, `absentDenotesDefault`) *inside* the decision contract.  `MapDecisionCorrect` is
the same contract with both fields removed and replaced by `hrep : MapRep …`, so nothing about the
canonical read is postulated: it has to come from the representation. -/
structure MapDecisionCorrect (state : StoredTable) (query : MapKey)
    (decision : table.SearchDecision) (direct : MapKey → Option MapLocation)
    (live : MapKey → Prop) (map : CanonicalMap)
    (order prefixOrder sweep checked : List MapLocation) : Prop where
  /-- The extracted decision's complete contract, with the canonical read left open. -/
  search : Apeiron.GrowableSearch.SearchDecisionCorrect state MapDecode map query decision direct
    order prefixOrder sweep checked

/-- **The map face's representation supplies the canonical-`get` clauses of the decision contract.**
From `MapRep` alone, the full `Apeiron.GrowableSearch.SearchDecisionCorrect` at the represented map
follows: the live-representative clause is `map_get_of_live_representative` and the refusal clause is
`map_get_default_of_absent`.  Equivalently: the two semantic fields of the decision contract are
redundant given the map face's representation. -/
theorem searchDecisionCorrect_of_mapRep
    (state : StoredTable) (query : MapKey) (decision : table.SearchDecision)
    (direct : MapKey → Option MapLocation) (live : MapKey → Prop) (map : CanonicalMap)
    (order prefixOrder sweep checked : List MapLocation)
    (hrep : MapRep state direct live map) (hdec : MapDecisionCorrect state query decision direct live
      map order prefixOrder sweep checked) :
    Apeiron.GrowableSearch.SearchDecisionCorrect state MapDecode map query decision direct order
      prefixOrder sweep checked := by
  refine { hdec.search with
    foundDenotes := ?_
    absentDenotesDefault := ?_ }
  · intro location value hentry
    exact map_get_of_live_representative state direct live map hrep location query value hentry
  · exact map_get_default_of_absent state direct live map hrep query

/-- **The extracted map-face read body refines the canonical `Apeiron.Table.get`.**  For every
decision the contract admits, at every face and every key,

    word_stored_apply_get state decision = ok (Apeiron.Table.get map (sigma, key))

where `map` is the canonical map the extracted state represents under `MapRep`.  The extracted body
is `proof/ApeironWord.lean:26956-26968` over the real `table.Table.apply_get`, and the canonical
object is `Apeiron.Table.get` at `Apeiron/Table/Basic.lean:66`.  The proof uses
`Apeiron.GrowableSearch.word_stored_search_decision_simulates` at the contract
`searchDecisionCorrect_of_mapRep` derives from the representation, so the canonical read enters only
through `MapRep` — the decision contract states no `Apeiron.Table.get` clause of its own. -/
theorem map_get_refines_canonical_get
    (state : StoredTable) (query : MapKey) (decision : table.SearchDecision)
    (direct : MapKey → Option MapLocation) (live : MapKey → Prop) (map : CanonicalMap)
    (order prefixOrder sweep checked : List MapLocation)
    (hrep : MapRep state direct live map) (hdec : MapDecisionCorrect state query decision direct live
      map order prefixOrder sweep checked) :
    word_stored_apply_get state decision = ok (Apeiron.Table.get map query) :=
  Apeiron.GrowableSearch.word_stored_search_decision_simulates state query decision direct order
    prefixOrder sweep checked
    (searchDecisionCorrect_of_mapRep state query decision direct live map order prefixOrder sweep
      checked hrep hdec)

/-- **A fold of `Apeiron.Table.insertWith` over a stream that never mentions the key reads the
starting map's value there.**  `Apeiron.Table.get_insertWith_ne` (`Apeiron/Table/Basic.lean:95`)
leaves a read at every other key untouched, so each step of the fold is invisible at `key`.  Stated
over the canonical `insertWith`/`get` alone, at an arbitrary starting map. -/
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
`Apeiron.Table.build` (`Apeiron/Table/Basic.lean:121`) is the fold of `Apeiron.Table.insertWith` over
the stream from the constant default, so `foldl_get_of_stream_ne` reads it as the starting map's
value. -/
private theorem build_get_of_stream_ne {K V : Type} [DecidableEq K] (f : V → V → V) (default : V) :
    ∀ (kvs : List (K × V)) (key : K), (∀ kv, kv ∈ kvs → kv.1 ≠ key) →
      Apeiron.Table.get (Apeiron.Table.build f default kvs) key = default := by
  intro kvs key hstream
  have h := foldl_get_of_stream_ne f (fun _ => default) kvs key hstream
  simpa only [Apeiron.Table.build, Apeiron.Table.get] using h

/-- **The map face's read agrees with the canonical `Apeiron.Table.build` away from its stream.**
Under `MapRep`, a key that no physical location of the extracted table decodes to reads the same
value through the extracted read body and through a map built by the canonical constructor
`Apeiron.Table.build` (`Apeiron/Table/Basic.lean:121`) from any merge, any stream that does not
mention the key, and the extracted default.  This is the `get_build_acc` content at the map face's
refusal value, proved directly from the canonical `build`, `insertWith` and `get` because the
canonical `get_build_acc` is stated over the corpus's own `List.filter`, whose `DecidableEq`
instance is not this package's: absence is the default in both, so a build over an unrelated stream
is invisible to the refusal. -/
theorem map_get_agrees_with_canonical_build_off_stream
    (state : StoredTable) (direct : MapKey → Option MapLocation) (live : MapKey → Prop)
    (map : CanonicalMap) (hrep : MapRep state direct live map) (key : MapKey)
    (habsent : Apeiron.GrowableSearch.Absent state MapDecode key)
    (f : MapValue → MapValue → MapValue) (kvs : List (MapKey × MapValue))
    (hstream : ∀ kv, kv ∈ kvs → kv.1 ≠ key) :
    Apeiron.Table.get (Apeiron.Table.build f state.default kvs) key
      = Apeiron.Table.get map key :=
  (build_get_of_stream_ne f state.default kvs key hstream).trans
    (map_get_default_of_absent state direct live map hrep key habsent).symm

/-! ## Headline audit rows

Each row is the closed-cone gate of `Apeiron.AuditClosed`: `#audit` collects the declaration's
transitive axiom dependencies and fails the build on anything outside
`{propext, Classical.choice, Quot.sound}`, so a `sorryAx` anywhere in the cone is a build failure
rather than a line in a log.  They sit in this module rather than in `AxiomAudit/Core.lean` so they
run in the scoped build of `Map.Read` itself. -/

#audit Apeiron.MapFace.mapRep_liveSet
#audit Apeiron.MapFace.map_get_of_live_representative
#audit Apeiron.MapFace.mapRep_live_of_representative
#audit Apeiron.MapFace.map_get_default_of_absent
#audit Apeiron.MapFace.map_get_default_of_not_live
#audit Apeiron.MapFace.mapRep_notLive_of_absent
#audit Apeiron.MapFace.searchDecisionCorrect_of_mapRep
#audit Apeiron.MapFace.map_get_refines_canonical_get
#audit Apeiron.MapFace.map_get_agrees_with_canonical_build_off_stream

end Apeiron.MapFace