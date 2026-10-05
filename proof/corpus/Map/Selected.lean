import Map.Read

/-!
# The executed map face's representation

`Map/Rep.lean` and `Map/Read.lean` refine the **extracted** body `apeiron_word::word_stored_apply_get`
(`proof/ApeironWord.lean:26959`), which reads the growable flat/spill `table.Table` rooted at
`apeiron::Table` (`proof/extract/lib.rs:450`).  `main` at `b519047` **executes** a different body:
the selected map arm constructs `apeiron::hashtrinity::Table<StrataKey, i64>`
(`src/hashtrinity/mod.rs`, reached as `apeiron::freq::Table`, `src/freq.rs:29`) and calls its `get`
and `insert_with`.  This module states the executed face's representation and reduces its `get` to
the canonical `Apeiron.Table.get`.

The executed body is `hashtrinity` at rev `4cc14471ae24f3f204fb0b0e91736ef75c909358`, vendored at
`src/hashtrinity/`:

* carrier `Table<K, V, Stored<K>>` — `coords : Box<[u64]>` (identity and occupancy at once),
  `vals : Box<[MaybeUninit<V>]>`, `keys : Stored<K>`, `bits : u32`, `len : usize`, `tombs : usize`
  (`src/hashtrinity/mod.rs:441-453`);
* `home` — the key's slot, `coord < len ? coord : coord·PHI >> (64 − bits)`, the coordinate nudged
  off the two sentinels first (`src/hashtrinity/mod.rs:591-611`);
* `get` — read `coords[home]`; equal coordinate returns `vals[home]`, `EMPTY` returns `None`, otherwise
  the collision tail (`src/hashtrinity/mod.rs:700-778`);
* the one mutator `insert_with` — `src/hashtrinity/mod.rs:1064-1098`.

## Where the executed body differs from `apeiron::Table`

1. **No per-insert locator reservation.**  The extracted `apeiron::Table` insert path materializes a
   reserved relocation phase before the write — `Apeiron.GrowableRepresentation.ReservedRepR`
   (`Growable/Representation.lean:526`), the state a generated `insert` holds between address
   selection and commit.  The executed `insert_with` has no such phase: `src/hashtrinity/mod.rs:1064`
   probes once and writes `coords`/`vals`/`keys` in the same step.  `SelectedInsertRep` below carries
   no reserved-state field.
2. **A vacant slot makes no merge call.**  On a vacant slot the executed body stores `new` directly
   (`src/hashtrinity/mod.rs:1092-1096`), never calling `f`.  The canonical `Apeiron.Table.insertWith`
   reads `f value (get map key)` at the written key even when the key was absent, i.e. `f value
   default`; `selected_insert_vacant_differs_from_canonical` records the disagreeing pair.

## What is stated, and the residual

The extracted `word_hashtrinity_get` / `word_hashtrinity_insert_with` wrappers are staged in
`proof/extract/lib.rs` by a sibling lane and are **not yet regenerated** into
`proof/ApeironWord.lean`, so no Lean declaration names the executed body and no theorem here can be
about it directly.  What this module states instead is the executed face's representation contract
(`SelectedReadRep`, `SelectedInsertRep`, `SelectedRoutingRep` over the carrier `HTCarrier`) and the
strongest true statement over the reachable bodies: `selectedReadRep_of_mapRep` derives the executed
read contract from the reachable whole-table representation `MapRep` (`RepR` at the stored carrier)
plus the single named premise `hbridge`, that the executed read's `some`-keys are exactly the live
entries `Apeiron.GrowableSearch.entryAt` decodes.

Residual (the missing symbol): `apeiron_word::word_hashtrinity_get` and
`apeiron_word::word_hashtrinity_insert_with`, owned by the `proof/extract/lib.rs` lane; with them
extracted, `hbridge` is the live-set agreement of the executed coordinate read and the reachable
`entryAt`, and the executed body's `get` is `Apeiron.Table.get` of the represented map on the
admitted domain, with the refusal propagated outside it as the default.
-/

namespace Apeiron.MapFace

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

noncomputable local instance : DecidableEq MapKey := Classical.decEq _

/-- The executed map face's carrier, `hashtrinity::Table<K, V, Stored<K>>` (`src/hashtrinity/mod.rs:441`):
the coordinate/identity array, the payload array, the stored-key column, the slot-granularity `bits`,
the live count, and the tombstone count.  The read touches `coords`, `vals` and `bits`; `keys` and
`tombs` are carried for fidelity to the struct. -/
structure HTCarrier where
  coords : List U64
  vals : List I64
  keys : List I64
  bits : Nat
  len : Nat
  tombs : Nat

/-- The executed map face's interface at a carrier `State`: its read `State → MapKey → Option MapValue`
(`hashtrinity::Table::get`, `src/hashtrinity/mod.rs:700`), its home routing and slot count
(`hashtrinity::Table::home`, `src/hashtrinity/mod.rs:603`), and its one mutator
(`hashtrinity::Table::insert_with`, `src/hashtrinity/mod.rs:1064`).  The functions are the executed
bodies; they are the extraction's to supply and are not restated here. -/
structure SelectedFace (State : Type) where
  read : State → MapKey → Option MapValue
  home : State → MapKey → Nat
  slots : State → Nat
  insert : State → MapKey → MapValue → (MapValue → MapValue → MapValue) → State

/-- The executed map face: `SelectedFace` at its own carrier. -/
abbrev HTFace := SelectedFace HTCarrier

/-- **The executed map face's read representation.**  `SelectedReadRep F s live map default` says the
executed table `s` represents the canonical map `map` under the face `F`: the keys its `get` answers
`some` for are exactly the declared live set `live`, every live observation denotes
`Apeiron.Table.get` of `map`, and every key outside `live` reads `none` while the canonical read is the
default.  The clause shapes are `Apeiron.GrowableRepresentation.RepR`'s
(`Growable/Representation.lean:299`) collapsed to the executed face's `Option` read; `RepR` itself is
used directly in `selectedReadRep_of_repR`. -/
structure SelectedReadRep {State : Type} (F : SelectedFace State) (s : State)
    (live : MapKey → Prop) (map : CanonicalMap) (default : MapValue) : Prop where
  liveSet : ∀ key, live key ↔ (F.read s key).isSome
  get_live : ∀ key, live key → F.read s key = some (Apeiron.Table.get map key)
  get_absent : ∀ key, ¬ live key → F.read s key = none ∧ Apeiron.Table.get map key = default

/-- The executed read's live set is the represented live set. -/
theorem selected_liveSet {State : Type} {F : SelectedFace State} {s : State}
    {live : MapKey → Prop} {map : CanonicalMap} {default : MapValue}
    (h : SelectedReadRep F s live map default) :
    ∀ key, live key ↔ (F.read s key).isSome :=
  h.liveSet

/-- On the represented domain the executed read is the canonical read. -/
theorem selected_get_of_live {State : Type} {F : SelectedFace State} {s : State}
    {live : MapKey → Prop} {map : CanonicalMap} {default : MapValue}
    (h : SelectedReadRep F s live map default) {key : MapKey} (hlive : live key) :
    F.read s key = some (Apeiron.Table.get map key) :=
  h.get_live key hlive

/-- Outside the represented domain the executed read refuses. -/
theorem selected_get_none_of_not_live {State : Type} {F : SelectedFace State} {s : State}
    {live : MapKey → Prop} {map : CanonicalMap} {default : MapValue}
    (h : SelectedReadRep F s live map default) {key : MapKey} (hnot : ¬ live key) :
    F.read s key = none :=
  (h.get_absent key hnot).1

/-- Outside the represented domain, the canonical read is the extracted default: the refusal is
propagated as the default, not as a separate "not found". -/
theorem selected_get_default_of_not_live {State : Type} {F : SelectedFace State} {s : State}
    {live : MapKey → Prop} {map : CanonicalMap} {default : MapValue}
    (h : SelectedReadRep F s live map default) {key : MapKey} (hnot : ¬ live key) :
    Apeiron.Table.get map key = default :=
  (h.get_absent key hnot).2

/-- **The executed read is the canonical read, on the domain and outside it alike.**  The two
branches are exhaustive: `F.read s key = some value` exactly when `key` is represented live and the
canonical read at `key` is `value`. -/
theorem selected_get_refines_canonical_get {State : Type} {F : SelectedFace State} {s : State}
    {live : MapKey → Prop} {map : CanonicalMap} {default : MapValue}
    (h : SelectedReadRep F s live map default) (key : MapKey) (value : MapValue) :
    F.read s key = some value ↔ live key ∧ Apeiron.Table.get map key = value := by
  constructor
  · intro hread
    have hsome : (F.read s key).isSome := by rw [hread]; rfl
    have hlive : live key := (h.liveSet key).mpr hsome
    have hget : F.read s key = some (Apeiron.Table.get map key) := h.get_live key hlive
    exact ⟨hlive, Option.some.inj (hget.symm.trans hread)⟩
  · rintro ⟨hlive, hget⟩
    rw [h.get_live key hlive, hget]

/-- The executed home routing lands inside the executed table's slot array
(`src/hashtrinity/mod.rs:603`, `& (len - 1)` after the fold). -/
structure SelectedRoutingRep {State : Type} (F : SelectedFace State) (s : State) : Prop where
  home_lt : ∀ key, F.home s key < F.slots s

/-- The executed routing's slot bound. -/
theorem selected_home_lt_slots {State : Type} {F : SelectedFace State} {s : State}
    (h : SelectedRoutingRep F s) (key : MapKey) : F.home s key < F.slots s :=
  h.home_lt key

/-- **The executed mutator's contract.**  `SelectedInsertRep F s live map` says the one mutator reads
back as the canonical `Apeiron.Table.insertWith` on a live key, and as the written value on a vacant
key: the vacant slot makes no merge call. -/
structure SelectedInsertRep {State : Type} (F : SelectedFace State) (s : State)
    (live : MapKey → Prop) (map : CanonicalMap) : Prop where
  insert_live : ∀ key value f, live key →
    F.read (F.insert s key value f) key
      = some (Apeiron.Table.get (Apeiron.Table.insertWith f key value map) key)
  insert_absent : ∀ key value f, ¬ live key →
    F.read (F.insert s key value f) key = some value

/-- On a represented-live key the executed mutator reads back as the canonical `insertWith`. -/
theorem selected_insert_of_live {State : Type} {F : SelectedFace State} {s : State}
    {live : MapKey → Prop} {map : CanonicalMap}
    (h : SelectedInsertRep F s live map) {key : MapKey} (value : MapValue)
    (f : MapValue → MapValue → MapValue) (hlive : live key) :
    F.read (F.insert s key value f) key
      = some (Apeiron.Table.get (Apeiron.Table.insertWith f key value map) key) :=
  h.insert_live key value f hlive

/-- On a vacant key the executed mutator stores the written value and makes no merge call. -/
theorem selected_insert_absent_no_merge {State : Type} {F : SelectedFace State} {s : State}
    {live : MapKey → Prop} {map : CanonicalMap}
    (h : SelectedInsertRep F s live map) {key : MapKey} (value : MapValue)
    (f : MapValue → MapValue → MapValue) (hnot : ¬ live key) :
    F.read (F.insert s key value f) key = some value :=
  h.insert_absent key value f hnot

/-- **The vacant-slot difference, as an equation.**  On a key outside the represented domain the
executed mutator reads `value`, while the canonical `Apeiron.Table.insertWith` reads `f value default`:
the executed vacant slot makes no merge call and the canonical write does.  The two agree only when
`f value default = value`. -/
theorem selected_insert_vacant_differs_from_canonical {State : Type} {F : SelectedFace State}
    {s : State} {live : MapKey → Prop} {map : CanonicalMap} {default : MapValue}
    (hread : SelectedReadRep F s live map default) (hins : SelectedInsertRep F s live map)
    {key : MapKey} (value : MapValue) (f : MapValue → MapValue → MapValue) (hnot : ¬ live key) :
    F.read (F.insert s key value f) key = some value ∧
      Apeiron.Table.get (Apeiron.Table.insertWith f key value map) key = f value default := by
  refine ⟨hins.insert_absent key value f hnot, ?_⟩
  rw [Apeiron.Table.get_insertWith_self]
  rw [show map key = default from (hread.get_absent key hnot).2]

/-- **The executed read contract follows from the reachable whole-table representation.**  From
`Apeiron.GrowableRepresentation.RepR` at the extracted stored carrier and the single premise that the
executed read's `some`-keys are exactly the live entries `Apeiron.GrowableSearch.entryAt` decodes,
`SelectedReadRep` holds: the executed read is `Apeiron.Table.get` on the domain and refuses outside
it.  This is the executed face's contract, reduced to the reachable bodies. -/
theorem selectedReadRep_of_repR
    (F : HTFace) (s : HTCarrier)
    (state : StoredTable) (direct : MapKey → Option MapLocation)
    (live : MapKey → Prop) (map : CanonicalMap)
    (hrep : Apeiron.GrowableRepresentation.RepR state MapDecode direct live map)
    (hbridge : ∀ key value, F.read s key = some value ↔
      ∃ location, Apeiron.GrowableSearch.entryAt state MapDecode location = some (key, value)) :
    SelectedReadRep F s live map state.default := by
  refine ⟨?_, ?_, ?_⟩
  · intro key
    constructor
    · intro hlive
      obtain ⟨location, value, hentry⟩ := (hrep.liveSet key).mp hlive
      exact Option.isSome_iff_exists.mpr ⟨value, (hbridge key value).mpr ⟨location, hentry⟩⟩
    · intro hsome
      obtain ⟨value, hval⟩ := Option.isSome_iff_exists.mp hsome
      obtain ⟨location, hentry⟩ := (hbridge key value).mp hval
      exact (hrep.liveSet key).mpr ⟨location, value, hentry⟩
  · intro key hlive
    obtain ⟨location, value, hentry⟩ := (hrep.liveSet key).mp hlive
    have hget : Apeiron.Table.get map key = value := hrep.observations.1 location key value hentry
    rw [hget]
    exact (hbridge key value).mpr ⟨location, hentry⟩
  · intro key hnot
    refine ⟨?_, ?_⟩
    · cases hval : F.read s key with
      | none => rfl
      | some value =>
          obtain ⟨location, hentry⟩ := (hbridge key value).mp hval
          exact absurd ((hrep.liveSet key).mpr ⟨location, value, hentry⟩) hnot
    · exact hrep.observations.2 key hnot

/-- `MapRep` is `RepR` at the map face's decoder, so the executed read contract follows from the map
face's representation by the same reduction. -/
theorem selectedReadRep_of_mapRep
    (F : HTFace) (s : HTCarrier)
    (state : StoredTable) (direct : MapKey → Option MapLocation)
    (live : MapKey → Prop) (map : CanonicalMap)
    (hrep : MapRep state direct live map)
    (hbridge : ∀ key value, F.read s key = some value ↔
      ∃ location, Apeiron.GrowableSearch.entryAt state MapDecode location = some (key, value)) :
    SelectedReadRep F s live map state.default :=
  selectedReadRep_of_repR F s state direct live map hrep hbridge

/-! ## Headline audit rows -/

#audit Apeiron.MapFace.selected_liveSet
#audit Apeiron.MapFace.selected_get_of_live
#audit Apeiron.MapFace.selected_get_none_of_not_live
#audit Apeiron.MapFace.selected_get_default_of_not_live
#audit Apeiron.MapFace.selected_get_refines_canonical_get
#audit Apeiron.MapFace.selected_home_lt_slots
#audit Apeiron.MapFace.selected_insert_of_live
#audit Apeiron.MapFace.selected_insert_absent_no_merge
#audit Apeiron.MapFace.selected_insert_vacant_differs_from_canonical
#audit Apeiron.MapFace.selectedReadRep_of_repR
#audit Apeiron.MapFace.selectedReadRep_of_mapRep

end Apeiron.MapFace
