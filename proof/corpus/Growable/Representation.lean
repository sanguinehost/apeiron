import Growable.Geometry
import Growable.Refinement

/-!
# Whole-table growable representation

This ratifies the whole-table contract required by `docs/spec/storage-refinement.md`, extending the
selected-entry relations in `Growable.Relocation` to the complete generated flat/spill table.  The
represented semantic object is exactly `Apeiron.Table.{HMap,get,insertWith,build}` on
`(Sigma × I64)` keys.  An explicit logical live set is retained because a live override may equal the
default value; the support of a total function cannot recover that physical fact.

`ReservedRepR` denotes materialized relocation storage before the pending `insertWith` is applied.
`PendingCommittedR` states the exact old-map/new-map boundary.  Constructing it from the generated
commit roots remains a separate theorem requiring `SearchDecisionCorrect` and callback laws.
-/

namespace Apeiron.GrowableRepresentation

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

abbrev LogicalKey := cell.Sigma × I64
abbrev LogicalEntry := LogicalKey × I64
abbrev WordTable (R : Type) := table.Table I64 I64 table.WordHome R 16#usize
abbrev Location := Apeiron.GrowableSearch.Location

noncomputable local instance : DecidableEq LogicalKey := Classical.decEq _

/-- A flat live bit or nonzero spill tag is the sole occupancy authority. -/
def Occupied {R : Type} (state : WordTable R) : Location → Prop
  | .flat index => ∃ slot,
      state.resident.slots.val[index.val]? = some slot ∧ slot.live = true
  | .spill shard index => ∃ region tag payload,
      state.shards.val[shard.val]? = some region ∧
      region.tags.val[index.val]? = some tag ∧
      region.slots.val[index.val]? = some payload ∧ tag ≠ 0#u8

/-- The logical keys represented by live physical entries are exactly the declared live set. -/
def LiveSetEqual {R : Type} (state : WordTable R) (decode : cell.Sigma → R → Option I64)
    (live : LogicalKey → Prop) : Prop :=
  ∀ key, live key ↔ ∃ location value,
    Apeiron.GrowableSearch.entryAt state decode location = some (key, value)

/-- Every live observation denotes the canonical total map; every non-live key reads the default. -/
def ObservationCorrect {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64) : Prop :=
  (∀ location key value,
      Apeiron.GrowableSearch.entryAt state decode location = some (key, value) →
      Apeiron.Table.get logical key = value) ∧
  (∀ key, ¬live key → Apeiron.Table.get logical key = state.default)

/-- No logical key has two physical representatives in the composite flat/spill layout. -/
def UniqueRepresentation {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) : Prop :=
  ∀ location₁ location₂ key value₁ value₂,
    Apeiron.GrowableSearch.entryAt state decode location₁ = some (key, value₁) →
    Apeiron.GrowableSearch.entryAt state decode location₂ = some (key, value₂) →
    location₁ = location₂ ∧ value₁ = value₂

/-- Every occupancy-authoritative live slot decodes, and only such slots decode as live entries. -/
def PhysicalCoverage {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) : Prop :=
  ∀ location, Occupied state location ↔ ∃ entry,
    Apeiron.GrowableSearch.entryAt state decode location = some entry

/-- The direct-index function is exactly the generated `WordHome` address and shard-major index. -/
def DirectIndexCorrect {R : Type} (state : WordTable R)
    (direct : LogicalKey → Option Location) : Prop :=
  ∀ key location, direct key = some location ↔ ∃ address index,
    Apeiron.GrowableIdentity.wordHomeInst.address state.home key.1 key.2 = ok address ∧
    Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val =
      some index.val ∧
    location = .flat index

/-- Each live representative agrees with generated addressing at its flat index or spill shard. -/
def AddressCorrect {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) : Prop :=
  ∀ location key value,
    Apeiron.GrowableSearch.entryAt state decode location = some (key, value) →
    match location with
    | .flat index => ∃ address,
        Apeiron.GrowableIdentity.wordHomeInst.address state.home key.1 key.2 = ok address ∧
        Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = some index.val
    | .spill shard _ => ∃ address,
        Apeiron.GrowableIdentity.wordHomeInst.address state.home key.1 key.2 = ok address ∧
        table.shard address = ok shard

/-- Flat extent, fixed spill directory, and each spill tag/payload directory have matching lengths. -/
def SpillDirectoryConsistent {R : Type} (state : WordTable R) : Prop :=
  state.resident.slots.length =
      Apeiron.GrowableGeometry.shardCount * state.resident.lanes.val ∧
  state.shards.length = Apeiron.GrowableGeometry.shardCount ∧
  ∀ region ∈ state.shards.val, region.tags.length = region.slots.length

/-- `len` counts the finite logical live set, including live entries whose value equals the default. -/
def LengthCorrect {R : Type} (state : WordTable R) (live : LogicalKey → Prop) : Prop :=
  ∃ keys : List LogicalKey, keys.Nodup ∧
    (∀ key, key ∈ keys ↔ live key) ∧ keys.length = state.len.val

/--
The whole generated table represents one canonical `Apeiron.Table.HMap (Sigma × I64) I64`.
`DirectCover` is the composite direct-empty certificate from `Growable.Search`.
-/
structure RepR {R : Type} (state : WordTable R) (decode : cell.Sigma → R → Option I64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64) : Prop where
  liveSet : LiveSetEqual state decode live
  observations : ObservationCorrect state decode live logical
  unique : UniqueRepresentation state decode
  coverage : PhysicalCoverage state decode
  address : AddressCorrect state decode
  directIndex : DirectIndexCorrect state direct
  directCover : Apeiron.GrowableSearch.DirectCover state decode direct
  directory : SpillDirectoryConsistent state
  length : LengthCorrect state live

/-- Materialized relocation storage represents the old map before the pending write is committed. -/
def ReservedRepR {R : Type} (state : WordTable R) (reserved : table.RebuildReserved R I64)
    (decode : cell.Sigma → R → Option I64) (direct : LogicalKey → Option Location)
    (live : LogicalKey → Prop) (logical : Apeiron.Table.HMap LogicalKey I64) : Prop :=
  RepR (Apeiron.GrowableRefinement.rebuiltView state reserved) decode direct live logical

/--
The pending commit links the materialized old map to the canonical new-first successor.  The run
equation is explicit; this definition does not assert that an unextracted decision is correct.
-/
structure PendingCommittedR {R : Type} (state : WordTable R)
    (reserved : table.RebuildReserved R I64) (committed : WordTable R)
    (decode : cell.Sigma → R → Option I64)
    (directBefore directAfter : LogicalKey → Option Location)
    (liveBefore liveAfter : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64) (merge : I64 → I64 → I64)
    (pending : LogicalKey) (value : I64) (commitRun : Result (WordTable R)) : Prop where
  run : commitRun = ok committed
  reservedRep : ReservedRepR state reserved decode directBefore liveBefore logical
  committedRep : RepR committed decode directAfter liveAfter
    (Apeiron.Table.insertWith merge pending value logical)

/-- Whole-table live-set projection used by rebuild preservation. -/
theorem repR_liveSet {R state decode direct live logical}
    (h : @RepR R state decode direct live logical) : LiveSetEqual state decode live := h.liveSet

/-- Whole-table decoded uniqueness projection used by rebuild preservation. -/
theorem repR_unique {R state decode direct live logical}
    (h : @RepR R state decode direct live logical) : UniqueRepresentation state decode := h.unique

/-- Whole-table occupancy coverage projection used by rebuild preservation. -/
theorem repR_coverage {R state decode direct live logical}
    (h : @RepR R state decode direct live logical) : PhysicalCoverage state decode := h.coverage

/-- Composite `DirectCover` projection used by search after relocation. -/
theorem repR_directCover {R state decode direct live logical}
    (h : @RepR R state decode direct live logical) :
    Apeiron.GrowableSearch.DirectCover state decode direct := h.directCover

/-- Spill-directory shape projection used to frame disjoint relocation writes. -/
theorem repR_directory {R state decode direct live logical}
    (h : @RepR R state decode direct live logical) : SpillDirectoryConsistent state := h.directory

/-- Construct the reserved whole-table relation from certificates about the target rebuilt geometry. -/
theorem reservedRepR_of_components {R state reserved decode direct live logical}
    (hliveSet : LiveSetEqual (Apeiron.GrowableRefinement.rebuiltView state reserved) decode live)
    (hobservations : ObservationCorrect
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode live logical)
    (hunique : UniqueRepresentation
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode)
    (hcoverage : PhysicalCoverage
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode)
    (haddress : AddressCorrect
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode)
    (hdirectIndex : DirectIndexCorrect
      (Apeiron.GrowableRefinement.rebuiltView state reserved) direct)
    (hdirectCover : Apeiron.GrowableSearch.DirectCover
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode direct)
    (hdirectory : SpillDirectoryConsistent
      (Apeiron.GrowableRefinement.rebuiltView state reserved))
    (hlength : LengthCorrect
      (Apeiron.GrowableRefinement.rebuiltView state reserved) live) :
    @ReservedRepR R state reserved decode direct live logical :=
  ⟨hliveSet, hobservations, hunique, hcoverage, haddress, hdirectIndex, hdirectCover, hdirectory,
    hlength⟩

/-- The rebuilt target's live set is the one named by its reserved relation. -/
theorem reservedRepR_liveSet {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    LiveSetEqual (Apeiron.GrowableRefinement.rebuiltView state reserved) decode live := h.liveSet

/-- Decoded keys are unique in the rebuilt target geometry. -/
theorem reservedRepR_unique {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    UniqueRepresentation (Apeiron.GrowableRefinement.rebuiltView state reserved) decode := h.unique

/-- Occupancy-authoritative slots exactly cover the rebuilt target's decoded entries. -/
theorem reservedRepR_coverage {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    PhysicalCoverage (Apeiron.GrowableRefinement.rebuiltView state reserved) decode := h.coverage

/-- Direct-empty absence uses the direct relation for the rebuilt target geometry. -/
theorem reservedRepR_directCover {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    Apeiron.GrowableSearch.DirectCover
      (Apeiron.GrowableRefinement.rebuiltView state reserved) decode direct := h.directCover

/-- The rebuilt target has the checked flat extent and fixed spill directory. -/
theorem reservedRepR_directory {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    SpillDirectoryConsistent (Apeiron.GrowableRefinement.rebuiltView state reserved) := h.directory

/-- The rebuilt target's committed length counts its declared live set. -/
theorem reservedRepR_length {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical) :
    LengthCorrect (Apeiron.GrowableRefinement.rebuiltView state reserved) live := h.length

/-- Reserved storage is precisely the whole-table relation on its pre-pending table view. -/
theorem reservedRepR_iff_rebuiltView {R state reserved decode direct live logical} :
    @ReservedRepR R state reserved decode direct live logical ↔
      RepR (Apeiron.GrowableRefinement.rebuiltView state reserved) decode direct live logical :=
  Iff.rfl

/-- Project the old-map relation from the pending/committed boundary. -/
theorem pendingCommitted_reserved {R state reserved committed decode directBefore directAfter
    liveBefore liveAfter logical merge pending value commitRun}
    (h : @PendingCommittedR R state reserved committed decode directBefore directAfter liveBefore
      liveAfter logical merge pending value commitRun) :
    ReservedRepR state reserved decode directBefore liveBefore logical := h.reservedRep

/-- Project the canonical `insertWith` successor from the pending/committed boundary. -/
theorem pendingCommitted_committed {R state reserved committed decode directBefore directAfter
    liveBefore liveAfter logical merge pending value commitRun}
    (h : @PendingCommittedR R state reserved committed decode directBefore directAfter liveBefore
      liveAfter logical merge pending value commitRun) :
    RepR committed decode directAfter liveAfter
      (Apeiron.Table.insertWith merge pending value logical) := h.committedRep

/-- A rebuild source and the search location with the same generated indexes observe the same slot. -/
def sourceLocation : table.RebuildSource → Location
  | .Resident index => .flat index
  | .Spill shard index => .spill shard index

theorem sourceObservation_eq_entryAt {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) (source : table.RebuildSource) :
    Apeiron.GrowableRelocation.sourceObservation state decode source =
      Apeiron.GrowableSearch.entryAt state decode (sourceLocation source) := by
  cases source with
  | Resident index =>
      cases hslot : state.resident.slots.val[index.val]? with
      | none => simp [Apeiron.GrowableRelocation.sourceObservation,
          Apeiron.GrowableSearch.entryAt, sourceLocation, hslot]
      | some slot =>
          cases hlive : slot.live <;>
            cases hdecode : decode slot.sigma slot.held <;>
            simp [Apeiron.GrowableRelocation.sourceObservation,
              Apeiron.GrowableSearch.entryAt, sourceLocation, hslot, hlive, hdecode]
  | Spill shard index =>
      cases hregion : state.shards.val[shard.val]? with
      | none => simp [Apeiron.GrowableRelocation.sourceObservation,
          Apeiron.GrowableSearch.entryAt, sourceLocation, hregion]
      | some region =>
          cases htag : region.tags.val[index.val]? with
          | none => simp [Apeiron.GrowableRelocation.sourceObservation,
              Apeiron.GrowableSearch.entryAt, sourceLocation, hregion, htag]
          | some tag =>
              cases hslot : region.slots.val[index.val]? with
              | none => simp [Apeiron.GrowableRelocation.sourceObservation,
                  Apeiron.GrowableSearch.entryAt, sourceLocation, hregion, htag, hslot]
              | some payload =>
                  rcases payload with ⟨sigma, held, value⟩
                  by_cases hzero : tag = 0#u8 <;>
                    cases hdecode : decode sigma held <;>
                    simp [Apeiron.GrowableRelocation.sourceObservation,
                      Apeiron.GrowableSearch.entryAt, sourceLocation, hregion, htag, hslot,
                      hzero, hdecode]

/-- Whole-table representation agrees with the existing selected-source `RepR`. -/
theorem repR_singleSource {R state decode direct live logical}
    (h : @RepR R state decode direct live logical) (source : table.RebuildSource)
    (entry : LogicalEntry)
    (hentry : Apeiron.GrowableRelocation.sourceObservation state decode source = some entry) :
    Apeiron.GrowableRelocation.RepR state decode source logical := by
  refine ⟨entry, hentry, ?_⟩
  have hobserve := h.observations.1 (sourceLocation source) entry.1 entry.2
  apply hobserve
  rw [← sourceObservation_eq_entryAt]
  exact hentry

/-- A reserved destination and the rebuilt table view observe the same materialized slot. -/
def destinationLocation : Apeiron.GrowableRelocation.Destination → Location
  | .flat index => .flat index
  | .spill shard index => .spill shard index

theorem reservedObservation_eq_entryAt {R : Type} (state : WordTable R)
    (reserved : table.RebuildReserved R I64) (decode : cell.Sigma → R → Option I64)
    (destination : Apeiron.GrowableRelocation.Destination) :
    Apeiron.GrowableRelocation.reservedObservation reserved decode destination =
      Apeiron.GrowableSearch.entryAt (Apeiron.GrowableRefinement.rebuiltView state reserved) decode
        (destinationLocation destination) := by
  cases destination with
  | flat index =>
      cases hslot : reserved.resident.slots.val[index.val]? with
      | none => simp [Apeiron.GrowableRelocation.reservedObservation,
          Apeiron.GrowableSearch.entryAt, Apeiron.GrowableRefinement.rebuiltView,
          destinationLocation, hslot]
      | some slot =>
          cases hlive : slot.live <;>
            cases hdecode : decode slot.sigma slot.held <;>
            simp [Apeiron.GrowableRelocation.reservedObservation,
              Apeiron.GrowableSearch.entryAt, Apeiron.GrowableRefinement.rebuiltView,
              destinationLocation, hslot, hlive, hdecode]
  | spill shard index =>
      cases hregion : reserved.regions.val[shard.val]? with
      | none => simp [Apeiron.GrowableRelocation.reservedObservation,
          Apeiron.GrowableSearch.entryAt, Apeiron.GrowableRefinement.rebuiltView,
          destinationLocation, hregion]
      | some region =>
          cases htag : region.tags.val[index.val]? with
          | none => simp [Apeiron.GrowableRelocation.reservedObservation,
              Apeiron.GrowableSearch.entryAt, Apeiron.GrowableRefinement.rebuiltView,
              destinationLocation, hregion, htag]
          | some tag =>
              cases hslot : region.slots.val[index.val]? with
              | none => simp [Apeiron.GrowableRelocation.reservedObservation,
                  Apeiron.GrowableSearch.entryAt, Apeiron.GrowableRefinement.rebuiltView,
                  destinationLocation, hregion, htag, hslot]
              | some payload =>
                  rcases payload with ⟨sigma, held, value⟩
                  by_cases hzero : tag = 0#u8 <;>
                    cases hdecode : decode sigma held <;>
                    simp [Apeiron.GrowableRelocation.reservedObservation,
                      Apeiron.GrowableSearch.entryAt, Apeiron.GrowableRefinement.rebuiltView,
                      destinationLocation, hregion, htag, hslot, hzero, hdecode]

/-- Whole reserved representation agrees with the existing selected-destination relation. -/
theorem reservedRepR_singleDestination {R state reserved decode direct live logical}
    (h : @ReservedRepR R state reserved decode direct live logical)
    (destination : Apeiron.GrowableRelocation.Destination) (entry : LogicalEntry)
    (hentry : Apeiron.GrowableRelocation.reservedObservation reserved decode destination =
      some entry) :
    Apeiron.GrowableRelocation.ReservedRepR reserved decode destination logical := by
  refine ⟨entry, hentry, ?_⟩
  have hobserve := h.observations.1 (destinationLocation destination) entry.1 entry.2
  apply hobserve
  rw [← reservedObservation_eq_entryAt]
  exact hentry

end Apeiron.GrowableRepresentation
