import ApeironWord
import Apeiron.Table.Basic
import Apeiron.AuditClosed
import Growable.Identity
import Growable.Geometry
import Growable.Representation
import Growable.Search

/-!
# The map face's representation

The canonical object is `Apeiron.Table.{HMap,get,insertWith,build}` (`Apeiron/Table/Basic.lean`,
corpus rev `880aa57e47e782ed7a9d9ee1974cfc87360b446b`), a *total* function from keys to values with
absence carried by the default.  The **map face** is the opaque-key face: the key is the opaque
64-bit word `I64` carried together with its face `cell.Sigma`, so the represented canonical object
is

    Apeiron.Table.HMap (cell.Sigma × I64) I64

— `HMap` at `Apeiron/Table/Basic.lean:57`, `get` at `:66`.

This module states that representation as one named object, `MapRep`, and proves the two denotation
clauses that make it a refinement of `get`:

* on the represented domain — a live physical representative of the extracted table — the canonical
  `get` at that key is the extracted value (`map_get_of_live_representative`);
* outside it — a key with no representative anywhere — the canonical `get` is the extracted default
  (`map_get_default_of_absent`, `map_get_default_of_not_live`).

`MapRep` is the map face's named instance of the landed whole-table representation
`Apeiron.GrowableRepresentation.RepR` (`Growable/Representation.lean:299`) at the **stored** carrier:
the face's own key and value types, its own extracted decoder `Apeiron.GrowableSearch.decodeStoredOption`,
and the canonical map it denotes.  Nothing is added to or removed from `RepR`; what this module adds
is the face's name for it and its statement in canonical `Apeiron.Table.get` form.

`Growable/ReadComposition.lean` carries the public `get` theorem for the whole table at the same
`RepR`; that module is a root of the bridge but its own import chain (`Growable.Read` →
`Growable.ReadLoop`) does not elaborate against the current extraction, so it cannot be imported
here.  `Growable/GaplessRead.lean` carries the executed-body `some`/`none` obligations for
`table.probe.spill_read`; it too does not elaborate against the current extraction (three unknown
identifiers, named in `docs/evidence` receipt of this task), so it is not imported here either.
`Growable/BuildRefine.lean` carries the generated fold's link to `Apeiron.Table.build`; the
`build` linkage this module needs at `get` is taken from `Apeiron.Table.Basic` directly, in
`Map/Read.lean`.

## What is not here

`MapRep` is carried as a premise everywhere it is used.  No theorem here derives `MapRep` for a
non-empty extracted table: the only landed producer is
`Apeiron.GrowableRepresentation.word_stored_table_new_repR`, whose state carries the empty live set
and the empty direct map.  Non-empty constructor reachability is the open part of the read map
(`Growable/ReadComposition.lean:376-388`), not a missing lemma in this module.
-/

namespace Apeiron.MapFace

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-- The map face's key: the opaque 64-bit word carried together with its face. -/
abbrev MapKey := cell.Sigma × I64

/-- The map face's value type. -/
abbrev MapValue := I64

/-- A physical representative of the map face: a flat lane index or a spill shard and index. -/
abbrev MapLocation := Apeiron.GrowableSearch.Location

/-- The extracted map-face table: the generated `table.Table` at the stored carrier and the fixed
`N = 16` group width, exactly the type of the extracted read body `word_stored_apply_get`
(`proof/ApeironWord.lean:26959`, from `proof/extract/lib.rs:450-452`). -/
abbrev StoredTable := table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize

/-- The canonical object the map face represents: `Apeiron.Table.HMap` at `(Sigma × I64)` keys.
`Apeiron/Table/Basic.lean:57`. -/
abbrev CanonicalMap := Apeiron.Table.HMap MapKey MapValue

/-- The map face's decoder: a stored carrier retains its opaque key exactly, at every face
(`Apeiron.GrowableIdentity.decodeStored`, `Growable/Identity.lean:52`). -/
abbrev MapDecode : cell.Sigma → table.keyrep.Stored I64 → Option MapValue :=
  Apeiron.GrowableSearch.decodeStoredOption

/-- **The map face's representation.**  `MapRep state direct live map` says the extracted map-face
table `state` *represents* the canonical map `map : Apeiron.Table.HMap (cell.Sigma × I64) I64`: the
logical keys its live physical entries decode to are exactly the declared live set `live`, every
live observation denotes `Apeiron.Table.get map` at its own key, every key outside `live` reads the
extracted default, no logical key has two representatives, every occupancy-authoritative slot
decodes, each representative sits at the address the table's own routing computes, the direct index
is that routing, and the flat extent, spill directory and `len` have the checked lengths.

This is the landed whole-table representation `Apeiron.GrowableRepresentation.RepR`
(`Growable/Representation.lean:299`) at the stored carrier, which the header of that module already
names as "exactly `Apeiron.Table.{HMap,get,insertWith,build}` on `(Sigma × I64)` keys".  `MapRep` is
that relation under the map face's own names; the canonical object it refines is `Apeiron.Table.HMap`
and the only operation it constrains is `Apeiron.Table.get`. -/
def MapRep (state : StoredTable) (direct : MapKey → Option MapLocation)
    (live : MapKey → Prop) (map : CanonicalMap) : Prop :=
  Apeiron.GrowableRepresentation.RepR state MapDecode direct live map

/-- The represented domain is `live`: the set of logical keys the extracted table represents as
live, from `RepR`'s live-set equality. -/
theorem mapRep_liveSet {state : StoredTable} {direct : MapKey → Option MapLocation}
    {live : MapKey → Prop} {map : CanonicalMap} (hrep : MapRep state direct live map) :
    ∀ key, live key ↔ ∃ location value,
      Apeiron.GrowableSearch.entryAt state MapDecode location = some (key, value) :=
  hrep.liveSet

/-- **On the represented domain, the canonical `get` is the extracted value.**  A live physical
representative of the extracted map-face table at `(key, value)` forces
`Apeiron.Table.get map key = value`: the extracted read body and the canonical read agree wherever
the table represents the key as live.  This is `RepR.observations`'s live half, in the canonical
`Apeiron.Table.get` form the map face refines. -/
theorem map_get_of_live_representative
    (state : StoredTable) (direct : MapKey → Option MapLocation) (live : MapKey → Prop)
    (map : CanonicalMap) (hrep : MapRep state direct live map)
    (location : MapLocation) (key : MapKey) (value : MapValue)
    (hentry : Apeiron.GrowableSearch.entryAt state MapDecode location = some (key, value)) :
    Apeiron.Table.get map key = value :=
  hrep.observations.1 location key value hentry

/-- A live representative puts its key in the represented domain. -/
theorem mapRep_live_of_representative
    (state : StoredTable) (direct : MapKey → Option MapLocation) (live : MapKey → Prop)
    (map : CanonicalMap) (hrep : MapRep state direct live map)
    (location : MapLocation) (key : MapKey) (value : MapValue)
    (hentry : Apeiron.GrowableSearch.entryAt state MapDecode location = some (key, value)) :
    live key :=
  (hrep.liveSet key).2 ⟨location, value, hentry⟩

/-- **Outside the represented domain, the canonical `get` is the extracted default.**  A key that
no physical location of the extracted table decodes to reads `state.default` through
`Apeiron.Table.get`: the canonical table is total, so a refusal is propagated as the default rather
than as a separate "not found", and the value propagated is the extracted one.  Premises are the
map face's representation and the pointwise exclusion only — no decision contract. -/
theorem map_get_default_of_absent
    (state : StoredTable) (direct : MapKey → Option MapLocation) (live : MapKey → Prop)
    (map : CanonicalMap) (hrep : MapRep state direct live map) (key : MapKey)
    (habsent : Apeiron.GrowableSearch.Absent state MapDecode key) :
    Apeiron.Table.get map key = state.default := by
  refine hrep.observations.2 key ?_
  intro hlive
  obtain ⟨location, value, hentry⟩ := (hrep.liveSet key).mp hlive
  exact habsent location value hentry

/-- **The represented map is the constant-default map off the live set.**  Every key the
representation does not represent as live reads the extracted default, which is the same clause as
`map_get_default_of_absent` read off `live` instead of off the pointwise exclusion.  Stated
separately because the two premises are what a caller has: a caller holding a live-set predicate
uses this one, and a caller holding a pointwise exclusion uses the other. -/
theorem map_get_default_of_not_live
    (state : StoredTable) (direct : MapKey → Option MapLocation) (live : MapKey → Prop)
    (map : CanonicalMap) (hrep : MapRep state direct live map) (key : MapKey)
    (hnotLive : ¬ live key) :
    Apeiron.Table.get map key = state.default :=
  hrep.observations.2 key hnotLive

/-- Absence in the extracted table and absence in the represented domain coincide: a key is not
represented as live exactly when no location decodes to it.  The reverse direction of
`RepR.liveSet`, isolated. -/
theorem mapRep_notLive_of_absent
    (state : StoredTable) (direct : MapKey → Option MapLocation) (live : MapKey → Prop)
    (map : CanonicalMap) (hrep : MapRep state direct live map) (key : MapKey)
    (habsent : Apeiron.GrowableSearch.Absent state MapDecode key) :
    ¬ live key := by
  intro hlive
  obtain ⟨location, value, hentry⟩ := (hrep.liveSet key).mp hlive
  exact habsent location value hentry

end Apeiron.MapFace