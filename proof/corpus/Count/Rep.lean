import ApeironWord
import Apeiron.Table.Basic
import Apeiron.AuditClosed
import Growable.Identity
import Growable.Geometry
import Growable.Representation
import Growable.Search

/-!
# The count face's representation

The canonical object is `Apeiron.Table.{HMap,get,insertWith,build}` (`Apeiron/Table/Basic.lean`,
corpus rev `880aa57e47e782ed7a9d9ee1974cfc87360b446b`), a *total* function from keys to values with
absence carried by the default.  The **count face** is the frequency-map face: the key is the opaque
64-bit word `I64` carried together with its face `cell.Sigma`, and the value is the merged count.  The
represented canonical object is therefore

    Apeiron.Table.HMap (cell.Sigma × I64) I64

— `HMap` at `Apeiron/Table/Basic.lean:57`, `get` at `:66`.  This is the one canonical object; counting
is not a second canonical definition.  The count face's own reading of it is fixed in `Count/Read.lean`
by the additive merge `core.num.I64.wrapping_add` on `I64` and the zero default, entering through
`Apeiron.Table.{build,insertWith}`.

This module states that representation as one named object, `CountRep`, and proves the denotation
clauses that make it a refinement of `get`:

* on the represented domain — a live physical representative of the extracted table — the canonical
  `get` at that key is the extracted value (`count_get_of_live_representative`);
* outside it — a key with no representative anywhere — the canonical `get` is the extracted default
  (`count_get_default_of_absent`, `count_get_default_of_not_live`), which at the count face's zero
  default is the count zero (`count_get_zero_of_absent_at_zero_default`).

`CountRep` is the count face's named instance of the landed whole-table representation
`Apeiron.GrowableRepresentation.RepR` (`Growable/Representation.lean:299`) at the **stored** carrier:
the face's own key and value types, its own extracted decoder `Apeiron.GrowableSearch.decodeStoredOption`,
and the canonical map it denotes.  Nothing is added to or removed from `RepR`; what this module adds is
the face's name for it and its statement in canonical `Apeiron.Table.get` form.

## The carrier, reconciled

`StoredTable` is the generated `table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize`,
the type of the extracted read body `word_stored_apply_get` (`proof/ApeironWord.lean:26959`, from
`proof/extract/lib.rs:450-452`).  The two count refinements named the same type differently: one
`CountTable`, one `StoredTable`.  This module keeps **`StoredTable`**, the name the landed map face uses
for the identical generated type (`Map/Rep.lean:75`), so the two faces' `RepR` instances sit at one
type and one name; the synonym `CountTable` is dropped.

## What the count face is, and what it is not

The count face's **executed** body is `hashtrinity::FreqMap::count_par` (`src/hashtrinity/freq.rs:30`;
the selected arm `examples/reg1/adapters/spatial.rs:792` reaches the git dependency at rev `4cc14471`),
followed by `FreqMap::get_each`.  Neither is extracted: `proof/ApeironWord.lean` carries no `FreqMap`,
`count_par` or `get_each` symbol, `proof/extract/lib.rs` names no `freq` module, and `count_par` drives
an OS thread pool over raw-pointer writes and closures, which the extraction fragment does not model.
So no `word_count_get` body exists and none is invented here; that symbol was a fabricated precedent in
the earlier count work and is not restored.  What *is* extracted, and is this crate's own code, is the
generic stored-carrier read body `word_stored_apply_get` and the counting aggregate the adapter runs on
the apeiron side of every spatial pair:

- `run_selected` (`spatial.rs:589-604`) is `SpatialTable::new(0, StrataHome)` then
  `table.insert_with(&merge, StrataKey { tag: k }, 1)` per entity and `table.get(...)` per probe, with
  `merge = |new, old| new.wrapping_add(old)`;
- `registry_bucket_split` (`spatial.rs:756-772`) is
  `SpatialTable::build_at_parallel_phased(&merge, 0, &qualified, ...)`, the same additive merge from
  the zero default.

Those extracted bodies are refined in `Count/Read.lean`, against the canonical `Apeiron.Table.{get,build}`
and the generated monomorphic merge `table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge`
(`proof/ApeironWord.lean:24348-24357`, from `src/table.rs:200-205`), whose body is
`core.num.I64.wrapping_add`.

## What is not here

`CountRep` is carried as a premise everywhere it is used.  No theorem here derives `CountRep` for a
non-empty extracted table: the only landed producer is
`Apeiron.GrowableRepresentation.word_stored_table_new_repR`, whose state carries the empty live set and
the empty direct map.  Non-empty constructor reachability is the open part of the count map, as it is
for the read map.  The per-step insert-preservation linking the extracted `build_at` loop to
`Apeiron.Table.build` is likewise not discharged here; `Count/Read.lean` states the aggregate theorems
at the abstraction the crate exposes and names the conditional one as conditional.
-/

namespace Apeiron.CountFace

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-- The count face's key: the opaque 64-bit word carried together with its face. -/
abbrev CountKey := cell.Sigma × I64

/-- The count face's value type: the merged count, an `i64`. -/
abbrev CountValue := I64

/-- A physical representative of the count face: a flat lane index or a spill shard and index. -/
abbrev CountLocation := Apeiron.GrowableSearch.Location

/-- The extracted count-face table: the generated `table.Table` at the stored carrier and the fixed
`N = 16` group width, exactly the type of the extracted read body `word_stored_apply_get`
(`proof/ApeironWord.lean:26959`, from `proof/extract/lib.rs:450-452`).  This is the name the landed map
face uses for the identical type (`Map/Rep.lean:75`); the count face's earlier synonym `CountTable` is
dropped. -/
abbrev StoredTable := table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize

/-- The canonical object the count face represents: `Apeiron.Table.HMap` at `(Sigma × I64)` keys.
`Apeiron/Table/Basic.lean:57`. -/
abbrev CanonicalCount := Apeiron.Table.HMap CountKey CountValue

/-- The count face's decoder: a stored carrier retains its opaque key exactly, at every face
(`Apeiron.GrowableIdentity.decodeStored`, `Growable/Identity.lean:52`). -/
abbrev CountDecode : cell.Sigma → table.keyrep.Stored I64 → Option CountValue :=
  Apeiron.GrowableSearch.decodeStoredOption

/-- **The count face's representation.**  `CountRep state direct live count` says the extracted
count-face table `state` *represents* the canonical count map `count : Apeiron.Table.HMap
(cell.Sigma × I64) I64`: the logical keys its live physical entries decode to are exactly the declared
live set `live`, every live observation denotes `Apeiron.Table.get count` at its own key, every key
outside `live` reads the extracted default, no logical key has two representatives, every
occupancy-authoritative slot decodes, each representative sits at the address the table's own routing
computes, the direct index is that routing, and the flat extent, spill directory and `len` have the
checked lengths.

This is the landed whole-table representation `Apeiron.GrowableRepresentation.RepR`
(`Growable/Representation.lean:299`) at the stored carrier, which the header of that module already
names as "exactly `Apeiron.Table.{HMap,get,insertWith,build}` on `(Sigma × I64)` keys".  `CountRep` is
that relation under the count face's own names; the canonical object it refines is `Apeiron.Table.HMap`
and the only operation it constrains is `Apeiron.Table.get`. -/
def CountRep (state : StoredTable) (direct : CountKey → Option CountLocation)
    (live : CountKey → Prop) (count : CanonicalCount) : Prop :=
  Apeiron.GrowableRepresentation.RepR state CountDecode direct live count

/-- The represented domain is `live`: the set of logical keys the extracted table represents as live,
from `RepR`'s live-set equality. -/
theorem countRep_liveSet {state : StoredTable} {direct : CountKey → Option CountLocation}
    {live : CountKey → Prop} {count : CanonicalCount} (hrep : CountRep state direct live count) :
    ∀ key, live key ↔ ∃ location value,
      Apeiron.GrowableSearch.entryAt state CountDecode location = some (key, value) :=
  hrep.liveSet

/-- **On the represented domain, the canonical `get` is the extracted value.**  A live physical
representative of the extracted count-face table at `(key, value)` forces `Apeiron.Table.get count key
= value`: the extracted read body and the canonical read agree wherever the table represents the key as
live.  This is `RepR.observations`'s live half, in the canonical `Apeiron.Table.get` form the count face
refines. -/
theorem count_get_of_live_representative
    (state : StoredTable) (direct : CountKey → Option CountLocation) (live : CountKey → Prop)
    (count : CanonicalCount) (hrep : CountRep state direct live count)
    (location : CountLocation) (key : CountKey) (value : CountValue)
    (hentry : Apeiron.GrowableSearch.entryAt state CountDecode location = some (key, value)) :
    Apeiron.Table.get count key = value :=
  hrep.observations.1 location key value hentry

/-- A live representative puts its key in the represented domain. -/
theorem countRep_live_of_representative
    (state : StoredTable) (direct : CountKey → Option CountLocation) (live : CountKey → Prop)
    (count : CanonicalCount) (hrep : CountRep state direct live count)
    (location : CountLocation) (key : CountKey) (value : CountValue)
    (hentry : Apeiron.GrowableSearch.entryAt state CountDecode location = some (key, value)) :
    live key :=
  (hrep.liveSet key).2 ⟨location, value, hentry⟩

/-- **Outside the represented domain, the canonical `get` is the extracted default.**  A key that no
physical location of the extracted table decodes to reads `state.default` through `Apeiron.Table.get`:
the canonical table is total, so a refusal is propagated as the default rather than as a separate "not
found", and the value propagated is the extracted one.  Premises are the count face's representation
and the pointwise exclusion only — no decision contract. -/
theorem count_get_default_of_absent
    (state : StoredTable) (direct : CountKey → Option CountLocation) (live : CountKey → Prop)
    (count : CanonicalCount) (hrep : CountRep state direct live count) (key : CountKey)
    (habsent : Apeiron.GrowableSearch.Absent state CountDecode key) :
    Apeiron.Table.get count key = state.default := by
  refine hrep.observations.2 key ?_
  intro hlive
  obtain ⟨location, value, hentry⟩ := (hrep.liveSet key).mp hlive
  exact habsent location value hentry

/-- **The represented map is the constant-default map off the live set.**  Every key the representation
does not represent as live reads the extracted default, which is the same clause as
`count_get_default_of_absent` read off `live` instead of off the pointwise exclusion.  Stated separately
because the two premises are what a caller has: a caller holding a live-set predicate uses this one, and
a caller holding a pointwise exclusion uses the other. -/
theorem count_get_default_of_not_live
    (state : StoredTable) (direct : CountKey → Option CountLocation) (live : CountKey → Prop)
    (count : CanonicalCount) (hrep : CountRep state direct live count) (key : CountKey)
    (hnotLive : ¬ live key) :
    Apeiron.Table.get count key = state.default :=
  hrep.observations.2 key hnotLive

/-- **At the count face's zero default, an unrepresented key counts zero.**  The count face's selected
serial arm builds `SpatialTable::new(0, StrataHome)` (`spatial.rs:592`) and its parallel arm builds from
default `0` (`spatial.rs:769`), so with `state.default = 0#i64` the refusal value *is* the count zero.
This is the "absence is the zero count" clause of the face. -/
theorem count_get_zero_of_absent_at_zero_default
    (state : StoredTable) (direct : CountKey → Option CountLocation) (live : CountKey → Prop)
    (count : CanonicalCount) (hrep : CountRep state direct live count) (key : CountKey)
    (hzero : state.default = 0#i64)
    (habsent : Apeiron.GrowableSearch.Absent state CountDecode key) :
    Apeiron.Table.get count key = 0#i64 :=
  (count_get_default_of_absent state direct live count hrep key habsent).trans hzero

/-- Absence in the extracted table and absence in the represented domain coincide: a key is not
represented as live exactly when no location decodes to it.  The reverse direction of `RepR.liveSet`,
isolated. -/
theorem countRep_notLive_of_absent
    (state : StoredTable) (direct : CountKey → Option CountLocation) (live : CountKey → Prop)
    (count : CanonicalCount) (hrep : CountRep state direct live count) (key : CountKey)
    (habsent : Apeiron.GrowableSearch.Absent state CountDecode key) :
    ¬ live key := by
  intro hlive
  obtain ⟨location, value, hentry⟩ := (hrep.liveSet key).mp hlive
  exact habsent location value hentry

/-! ## Headline audit rows

Each row is the closed-cone gate of `Apeiron.AuditClosed`: `#audit` collects the declaration's
transitive axiom dependencies and fails the build on anything outside
`{propext, Classical.choice, Quot.sound}`, so a `sorryAx` anywhere in the cone is a build failure
rather than a line in a log.  They sit in this module rather than in `AxiomAudit/Core.lean` so they run
in the scoped build of `Count.Rep` itself. -/

#audit Apeiron.CountFace.countRep_liveSet
#audit Apeiron.CountFace.count_get_of_live_representative
#audit Apeiron.CountFace.countRep_live_of_representative
#audit Apeiron.CountFace.count_get_default_of_absent
#audit Apeiron.CountFace.count_get_default_of_not_live
#audit Apeiron.CountFace.count_get_zero_of_absent_at_zero_default
#audit Apeiron.CountFace.countRep_notLive_of_absent

end Apeiron.CountFace
