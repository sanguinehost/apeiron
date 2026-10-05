-- Row: `p1-primitive-refine-per-FfI4Rd4tkS5goRFZJVgsAR` (APEIRON-READ-PREMISE): the executed read's bridge premise, named and split.
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

/-! ## The premise `hbridge`, named and characterised

`selectedReadRep_of_repR` and `selectedReadRep_of_mapRep` each carry their last premise as an
anonymous binder, written out twice.  This section names it, splits it into its two directions, and
characterises it exactly, so the residual of the executed face's read refinement is a named
declaration with a known content rather than an open `∀ key value`.

**`SelectedReadBridge` is not a consequence of `Apeiron.GrowableRepresentation.RepR`.**  It quantifies
over an arbitrary `F : SelectedFace State` and an arbitrary `s : State`; `HTCarrier` is a hand-written
mirror of `hashtrinity::Table`'s fields (`src/hashtrinity/mod.rs:448-452`) and no clause in the tree
relates it to the extracted `StoredTable`, whose own read body is
`apeiron_word::word_stored_apply_get` (`proof/ApeironWord.lean:26956`).  The executed read
`hashtrinity::Table::get` (`src/hashtrinity/mod.rs:702`) has no declaration in the extraction at all:
`proof/extract/lib.rs:1227` stages the wrapper `word_hashtrinity_get`, and `proof/ApeironWord.lean`
contains no `hashtrinity` declaration.  There are therefore **two** independent unreachable causes,
and the wrappers landing discharge only the first:

1. the executed body `word_hashtrinity_get` is not in the extraction, so nothing states what
   `hashtrinity::Table::get` computes; and
2. no carrier agreement relates `s : HTCarrier` to `state : StoredTable`, which the bridge premise
   quantifies over side by side.

`selectedReadRep_iff_bridge` below pins the premise's content exactly: under `RepR` it is
*equivalent* to the executed face's read contract itself, so it is neither over- nor under-stated,
and what remains is the pair (1) and (2). -/

/-- **The executed read's entry agreement with the extracted table.**  `SelectedReadBridge F s state`
is the premise of `selectedReadRep_of_repR`, named: at the executed carrier `s` under the executed
read `F.read`, the keys answered `some` are exactly the live physical entries
`Apeiron.GrowableSearch.entryAt` decodes in the extracted stored table `state`, with the value pinned.
Its two directions are `SelectedReadSound` (from the extracted table) and `SelectedReadComplete`
(from the executed read). -/
structure SelectedReadBridge {State : Type} (F : SelectedFace State) (s : State)
    (state : StoredTable) : Prop where
  bridge : ∀ key value, F.read s key = some value ↔
    ∃ location, Apeiron.GrowableSearch.entryAt state MapDecode location = some (key, value)

/-- **Soundness: the extracted table's live entries are answered.**  A location whose slot decodes to
`(key, value)` forces the executed read to answer `some value` at `key`. -/
structure SelectedReadSound {State : Type} (F : SelectedFace State) (s : State)
    (state : StoredTable) : Prop where
  sound : ∀ key value, (∃ location,
      Apeiron.GrowableSearch.entryAt state MapDecode location = some (key, value)) →
    F.read s key = some value

/-- **Completeness: the executed read invents nothing.**  An answer `some value` at `key` forces a
live physical representative of `key` carrying `value`. -/
structure SelectedReadComplete {State : Type} (F : SelectedFace State) (s : State)
    (state : StoredTable) : Prop where
  complete : ∀ key value, F.read s key = some value →
    ∃ location, Apeiron.GrowableSearch.entryAt state MapDecode location = some (key, value)

/-- The bridge premise's direction from the extracted table. -/
theorem selectedReadSound_of_bridge {State : Type} {F : SelectedFace State} {s : State}
    {state : StoredTable} (h : SelectedReadBridge F s state) : SelectedReadSound F s state :=
  ⟨fun _ _ hentry => (h.bridge _ _).mpr hentry⟩

/-- The bridge premise's direction from the executed read. -/
theorem selectedReadComplete_of_bridge {State : Type} {F : SelectedFace State} {s : State}
    {state : StoredTable} (h : SelectedReadBridge F s state) : SelectedReadComplete F s state :=
  ⟨fun _ _ hread => (h.bridge _ _).mp hread⟩

/-- The two named directions are the bridge premise. -/
theorem selectedReadBridge_of_sound_complete {State : Type} {F : SelectedFace State} {s : State}
    {state : StoredTable} (hs : SelectedReadSound F s state) (hc : SelectedReadComplete F s state) :
    SelectedReadBridge F s state := by
  refine ⟨?_⟩
  intro key value
  exact ⟨fun hread => hc.complete key value hread, fun hentry => hs.sound key value hentry⟩

/-- **The bridge premise is exactly its two directions**, with no further content. -/
theorem selectedReadBridge_iff {State : Type} {F : SelectedFace State} {s : State}
    {state : StoredTable} :
    SelectedReadBridge F s state ↔
      SelectedReadSound F s state ∧ SelectedReadComplete F s state := by
  constructor
  · intro h
    exact ⟨selectedReadSound_of_bridge h, selectedReadComplete_of_bridge h⟩
  · rintro ⟨hs, hc⟩
    exact selectedReadBridge_of_sound_complete hs hc

/-- **Partial discharge: soundness alone puts the executed read at `Apeiron.Table.get` on the
represented domain.**  From `RepR` and `SelectedReadSound` alone — completeness nowhere used — every
represented-live key is answered the canonical value.  The canonical read enters only through
`RepR.observations`, which is what pins the value of the live representative. -/
theorem selected_get_of_live_of_sound
    (F : HTFace) (s : HTCarrier)
    (state : StoredTable) (direct : MapKey → Option MapLocation)
    (live : MapKey → Prop) (map : CanonicalMap)
    (hrep : Apeiron.GrowableRepresentation.RepR state MapDecode direct live map)
    (hsound : SelectedReadSound F s state) :
    ∀ key, live key → F.read s key = some (Apeiron.Table.get map key) := by
  intro key hlive
  obtain ⟨location, value, hentry⟩ := (hrep.liveSet key).mp hlive
  have hget : Apeiron.Table.get map key = value :=
    hrep.observations.1 location key value hentry
  rw [hsound.sound key value ⟨location, hentry⟩, hget]

/-- **Partial discharge: completeness alone forbids invention.**  From `RepR` and
`SelectedReadComplete` alone — soundness nowhere used — every answer `some value` of the executed read
is a represented-live key whose canonical read is `value`.  This is the direction in which the
executed read could have been unsound, and it is discharged without touching `SelectedReadSound`. -/
theorem selected_read_never_invents_of_complete
    (F : HTFace) (s : HTCarrier)
    (state : StoredTable) (direct : MapKey → Option MapLocation)
    (live : MapKey → Prop) (map : CanonicalMap)
    (hrep : Apeiron.GrowableRepresentation.RepR state MapDecode direct live map)
    (hcomplete : SelectedReadComplete F s state) :
    ∀ key value, F.read s key = some value →
      live key ∧ Apeiron.Table.get map key = value := by
  intro key value hread
  obtain ⟨location, hentry⟩ := hcomplete.complete key value hread
  exact ⟨(hrep.liveSet key).mpr ⟨location, value, hentry⟩,
    hrep.observations.1 location key value hentry⟩

/-- **The executed read refines the canonical `Apeiron.Table.get` from the two named directions
alone.**  From `RepR` plus `SelectedReadSound` and `SelectedReadComplete`, at every key: the executed
read answers `some value` exactly when `key` is represented live and the canonical read there is
`value`.  No clause of the conclusion is postulated by either premise. -/
theorem selected_read_refines_of_sound_complete
    (F : HTFace) (s : HTCarrier)
    (state : StoredTable) (direct : MapKey → Option MapLocation)
    (live : MapKey → Prop) (map : CanonicalMap)
    (hrep : Apeiron.GrowableRepresentation.RepR state MapDecode direct live map)
    (hsound : SelectedReadSound F s state) (hcomplete : SelectedReadComplete F s state) :
    ∀ key value, F.read s key = some value ↔ live key ∧ Apeiron.Table.get map key = value := by
  intro key value
  constructor
  · exact selected_read_never_invents_of_complete F s state direct live map hrep hcomplete key value
  · rintro ⟨hlive, hget⟩
    obtain ⟨location, value₀, hentry⟩ := (hrep.liveSet key).mp hlive
    have hcanonical : F.read s key = some (Apeiron.Table.get map key) :=
      selected_get_of_live_of_sound F s state direct live map hrep hsound key hlive
    rw [hcanonical, hget]

/-- The whole-table representation reduction, over the named premise. -/
theorem selectedReadRep_of_bridge
    (F : HTFace) (s : HTCarrier)
    (state : StoredTable) (direct : MapKey → Option MapLocation)
    (live : MapKey → Prop) (map : CanonicalMap)
    (hrep : Apeiron.GrowableRepresentation.RepR state MapDecode direct live map)
    (hbridge : SelectedReadBridge F s state) :
    SelectedReadRep F s live map state.default :=
  selectedReadRep_of_repR F s state direct live map hrep hbridge.bridge

/-- **The executed face's read contract, under `RepR`, is equivalent to the bridge premise.**  The
residual is therefore characterised, not merely named: `RepR` together with `SelectedReadBridge` is
exactly `SelectedReadRep`, and conversely a face satisfying `SelectedReadRep` under `RepR` satisfies
the bridge premise.  Nothing is added and nothing is dropped, so the remaining obligation is the
bridge premise itself and nothing behind it. -/
theorem selectedReadRep_iff_bridge
    (F : HTFace) (s : HTCarrier)
    (state : StoredTable) (direct : MapKey → Option MapLocation)
    (live : MapKey → Prop) (map : CanonicalMap)
    (hrep : Apeiron.GrowableRepresentation.RepR state MapDecode direct live map) :
    SelectedReadRep F s live map state.default ↔ SelectedReadBridge F s state := by
  constructor
  · intro hface
    refine { bridge := fun key value => ?_ }
    constructor
    · -- from the executed read to a live physical representative
      intro hread
      have hsome : (F.read s key).isSome := by rw [hread]; rfl
      have hlive : live key := (hface.liveSet key).mpr hsome
      have hget : F.read s key = some (Apeiron.Table.get map key) := hface.get_live key hlive
      obtain ⟨location, value₀, hentry⟩ := (hrep.liveSet key).mp hlive
      have hsame : Apeiron.Table.get map key = value₀ :=
        hrep.observations.1 location key value₀ hentry
      have hread' : F.read s key = some value₀ := hsame ▸ hget
      have hval : value₀ = value := Option.some.inj (hread'.symm.trans hread)
      exact ⟨location, by simpa [hval] using hentry⟩
    · -- from a live physical representative to the executed read's answer
      intro hentry
      obtain ⟨location, hentry'⟩ := hentry
      have hlive : live key := (hrep.liveSet key).mpr ⟨location, value, hentry'⟩
      have hget : F.read s key = some (Apeiron.Table.get map key) := hface.get_live key hlive
      have hval : Apeiron.Table.get map key = value :=
        hrep.observations.1 location key value hentry'
      rw [hget, hval]
  · intro hbridge
    exact selectedReadRep_of_bridge F s state direct live map hrep hbridge

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
#audit Apeiron.MapFace.selectedReadSound_of_bridge
#audit Apeiron.MapFace.selectedReadComplete_of_bridge
#audit Apeiron.MapFace.selectedReadBridge_of_sound_complete
#audit Apeiron.MapFace.selectedReadBridge_iff
#audit Apeiron.MapFace.selected_get_of_live_of_sound
#audit Apeiron.MapFace.selected_read_never_invents_of_complete
#audit Apeiron.MapFace.selected_read_refines_of_sound_complete
#audit Apeiron.MapFace.selectedReadRep_of_bridge
#audit Apeiron.MapFace.selectedReadRep_iff_bridge

end Apeiron.MapFace
