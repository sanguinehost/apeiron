import ApeironWord
import Apeiron.Table.Basic
import Growable.Geometry
import Growable.Identity

/-!
# Search decision and extracted apply

The probe loops are extracted. `SearchDecisionCorrect` states their complete contract over the
physical entries supplied to the real straight-line apply, but this module does not prove that the
generated probe bodies establish that premise. Its theorems unfold the extracted stored and
regenerating apply bodies; they do not prove relocation or a whole-table representation invariant.
-/

namespace Apeiron.GrowableSearch


open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

abbrev LogicalKey := cell.Sigma × I64

noncomputable local instance : DecidableEq LogicalKey := Classical.decEq _

/-- A physical candidate consumed by the extracted apply. -/
inductive Location where
  | flat (index : Usize)
  | spill (shard index : Usize)
  deriving DecidableEq

/-- The extracted pair has only five well-formed class/location combinations. -/
def DecisionWellFormed (decision : table.SearchDecision) : Prop :=
  match decision.class, decision.location with
  | .Found, .Flat _ | .Vacant, .Flat _ | .Found, .Spill _ _ | .Vacant, .Spill _ _
  | .Exhausted, .NoIndex => True
  | _, _ => False

/-- Remove `NoIndex` from a well-formed found or vacant decision. -/
def decisionLocation (decision : table.SearchDecision) : Option Location :=
  match decision.location with
  | .Flat index => some (.flat index)
  | .Spill shard index => some (.spill shard index)
  | .NoIndex => none

/-- Decode a retained stored carrier. -/
def decodeStoredOption (_sigma : cell.Sigma) (held : table.keyrep.Stored I64) : Option I64 :=
  some held.key

/-- Decode a regenerating carrier only when the extracted codec succeeds. -/
def decodeRegenOption (sigma : cell.Sigma)
    (held : table.keyrep.Regen table.keyrep.I64WordCodec) : Option I64 :=
  match (Apeiron.GrowableIdentity.decodeRegen sigma held).match with
  | .ok key => some key
  | _ => none

/-- The live logical entry at a physical location. Vacant payload bytes are never observed. -/
def entryAt {R : Type} (state : table.Table I64 I64 table.WordHome R 16#usize)
    (decode : cell.Sigma → R → Option I64) (location : Location) : Option (LogicalKey × I64) :=
  match location with
  | .flat index =>
      match state.resident.slots.val[index.val]? with
      | some slot => if slot.live then
          match decode slot.sigma slot.held with
          | some key => some ((slot.sigma, key), slot.value)
          | none => none
        else none
      | none => none
  | .spill shard index =>
      match state.shards.val[shard.val]? with
      | some region =>
          match region.slots.val[index.val]? with
          | some slot => if slot.tag = 0#u8 then none else
              match decode slot.sigma slot.held with
              | some key => some ((slot.sigma, key), slot.value)
              | none => none
          | none => none
      | none => none

/-- A location is vacant exactly through its occupancy authority. -/
def VacantAt {R : Type} (state : table.Table I64 I64 table.WordHome R 16#usize) :
    Location → Prop
  | .flat index => ∃ slot, state.resident.slots.val[index.val]? = some slot ∧ slot.live = false
  | .spill shard index => ∃ region slot,
      state.shards.val[shard.val]? = some region ∧ region.slots.val[index.val]? = some slot ∧
        slot.tag = 0#u8

/-- No live physical representative has the queried full logical key. -/
def Absent {R : Type} (state : table.Table I64 I64 table.WordHome R 16#usize)
    (decode : cell.Sigma → R → Option I64) (query : LogicalKey) : Prop :=
  ∀ location value, entryAt state decode location ≠ some (query, value)

/-- The returned vacancy is the first vacancy in the finite decision order. -/
def FirstFit {R : Type} (state : table.Table I64 I64 table.WordHome R 16#usize)
    (order : List Location) (location : Location) : Prop :=
  VacantAt state location ∧ location ∈ order ∧
    ∀ earlier, earlier ∈ order.take (order.idxOf location) → ¬VacantAt state earlier

/-- Every possible live representative is included in the mandatory finite coverage. -/
def CompleteCoverage {R : Type} (state : table.Table I64 I64 table.WordHome R 16#usize)
    (decode : cell.Sigma → R → Option I64) (checked : List Location) : Prop :=
  ∀ location entry, entryAt state decode location = some entry → location ∈ checked

/-- Direct-empty absence is licensed only when every present direct-admitted key covers its slot. -/
def DirectCover {R : Type} (state : table.Table I64 I64 table.WordHome R 16#usize)
    (decode : cell.Sigma → R → Option I64) (direct : LogicalKey → Option Location) : Prop :=
  ∀ query location, direct query = some location → VacantAt state location →
    Absent state decode query

/--
A sound routed shard miss. The generated route selects the shard and no live representative in that
shard decodes to the query. The retired summary lanes are gone: the executing spill representation
is a linear tag probe (`table.probe.spill_read`), so a miss is the pointwise exclusion recorded in
`excludes`, not a negative summary lane. `Search` cannot import `GrowableRepresentation` (cycle), so
this is the local mirror of `GrowableRepresentation.SpillSummaryMissSound`; a producer instantiates
it from `RepR`'s shard inventory and the generated address route. It is state evidence, not the
desired `Absent` result: the routing and flat-exclusion consequences are separate structure fields
on `SearchDecisionCorrect`.
-/
structure SummaryMissSound {R : Type}
    (state : table.Table I64 I64 table.WordHome R 16#usize)
    (decode : cell.Sigma → R → Option I64) (query : LogicalKey) where
  address : table.TableAddress
  shard : Usize
  generatedAddress :
    Apeiron.GrowableIdentity.wordHomeInst.address state.home query.1 query.2 = ok address
  shardEq : table.shard address = ok shard
  region : table.Region R I64
  regionAt : state.shards.val[shard.val]? = some region
  excludes : ∀ index value, entryAt state decode (.spill shard index) ≠ some (query, value)

/--
The three-way non-`Found` absence certificate. Every non-`Found` decision licenses logical absence
through exactly one route:
* `directEmpty` — the decision's location is direct-admitted, so `DirectCover` at its first-fit
  vacancy yields `Absent`;
* `directExcludedSummaryMiss` — the query is not direct-admitted and a sound routed summary miss
  excludes every representative; `SearchDecisionCorrect.flatMatchAdmitted` and
  `SearchDecisionCorrect.spillMatchRouted` carry the generated route that turns the two exclusions
  into absence;
* `coverage` — every live representative lies in `checked` and none of the checked locations decodes
  to the query's canonical value.
No constructor carries a supplied `Found` or `Absent` result.
-/
inductive AbsenceCertificate {R : Type}
    (state : table.Table I64 I64 table.WordHome R 16#usize)
    (decode : cell.Sigma → R → Option I64) (logical : Apeiron.Table.HMap LogicalKey I64)
    (query : LogicalKey) (decision : table.SearchDecision)
    (direct : LogicalKey → Option Location) (checked : List Location) : Prop where
  | directEmpty (location : Location)
      (hlocation : decisionLocation decision = some location)
      (hdirect : direct query = some location) :
      AbsenceCertificate state decode logical query decision direct checked
  | directExcludedSummaryMiss (hexcluded : direct query = none)
      (hmiss : SummaryMissSound state decode query) :
      AbsenceCertificate state decode logical query decision direct checked
  | coverage (hcovered : ∀ location, location ∈ checked →
      entryAt state decode location ≠ some (query, Apeiron.Table.get logical query)) :
      AbsenceCertificate state decode logical query decision direct checked

/--
The unextracted decision's complete premise. Presence is exact and unique; vacancy is first-fit;
exhaustion is exact; direct absence uses `DirectCover`; every other absence uses the three-way
`AbsenceCertificate`; and a recorded vacancy never truncates the mandatory match sweep.
-/
structure SearchDecisionCorrect {R : Type}
    (state : table.Table I64 I64 table.WordHome R 16#usize)
    (decode : cell.Sigma → R → Option I64) (logical : Apeiron.Table.HMap LogicalKey I64)
    (query : LogicalKey) (decision : table.SearchDecision) (direct : LogicalKey → Option Location)
    (order prefixOrder sweep checked : List Location) : Prop where
  wellFormed : DecisionWellFormed decision
  unique : ∀ location₁ location₂ value₁ value₂,
    entryAt state decode location₁ = some (query, value₁) →
    entryAt state decode location₂ = some (query, value₂) → location₁ = location₂ ∧ value₁ = value₂
  foundExact : decision.class = .Found ↔ ∃ location value,
    decisionLocation decision = some location ∧
      entryAt state decode location = some (query, value)
  foundDenotes : ∀ location value, entryAt state decode location = some (query, value) →
    Apeiron.Table.get logical query = value
  vacantExact : decision.class = .Vacant ↔ ∃ location,
    decisionLocation decision = some location ∧ FirstFit state order location
  exhaustedExact : decision.class = .Exhausted ↔
    Absent state decode query ∧ ∀ location, location ∈ order → ¬VacantAt state location
  directCover : DirectCover state decode direct
  directAbsence : ∀ location, decision.class = .Vacant →
    decisionLocation decision = some location → direct query = some location →
      Absent state decode query
  complete : CompleteCoverage state decode checked
  absenceCertificate : decision.class ≠ .Found →
    AbsenceCertificate state decode logical query decision direct checked
  /-- Every live flat representative's key is direct-admitted: the forward direction of the generated
      `WordHome.address`/`preparedFlatIndex` route. It turns `direct query = none` into flat absence. -/
  flatMatchAdmitted : ∀ index value,
    entryAt state decode (.flat index) = some (query, value) →
      ∃ location, direct query = some location
  /-- Every live spill representative routes to its own shard through the generated address. It
      connects a routed `SummaryMissSound` exclusion to every spill location. -/
  spillMatchRouted : ∀ shard index value,
    entryAt state decode (.spill shard index) = some (query, value) →
      ∃ address, Apeiron.GrowableIdentity.wordHomeInst.address state.home query.1 query.2 =
        ok address ∧ table.shard address = ok shard
  vacancyDoesNotStopSweep : decision.class = .Vacant → ∀ location,
    location ∈ sweep → location ∈ checked
  prefixBeforeSweep : ∀ location, location ∈ prefixOrder → location ∈ order
  sweepComplete : ∀ location entry, entryAt state decode location = some entry →
    location ∈ prefixOrder ∨ location ∈ sweep
  absentDenotesDefault : Absent state decode query →
    Apeiron.Table.get logical query = state.default

/--
The generated query address selects the returned vacant flat index or spill shard.
`SearchDecisionCorrect` establishes first-fit occupancy but does not connect that location to
`WordHome.address`.  This is an explicit behavioural premise, not an input-only certificate;
discharging it requires extraction or another source binding for the decision loop.
-/
structure QueryAddressCorrect {R : Type}
    (state : table.Table I64 I64 table.WordHome R 16#usize) (query : LogicalKey)
    (address : table.TableAddress) (decision : table.SearchDecision) : Prop where
  generatedAddress :
    Apeiron.GrowableIdentity.wordHomeInst.address state.home query.1 query.2 = ok address
  vacantLocation : decision.class = .Vacant →
    match decision.location with
    | .Flat index =>
        Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = some index.val
    | .Spill shard _ => table.shard address = ok shard
    | .NoIndex => False

/-- The exact generated checked-add side condition on a vacant commit. -/
def FreshLengthBound {R : Type}
    (state : table.Table I64 I64 table.WordHome R 16#usize)
    (decision : table.SearchDecision) : Prop :=
  decision.class = .Vacant → state.len.val + 1 < Usize.max

/-- The two carrier instantiations admitted by the extracted growable word-table roots. -/
def ProductionWordCarrier (R : Type) : Prop :=
  R = table.keyrep.Stored I64 ∨
    R = table.keyrep.Regen table.keyrep.I64WordCodec

/--
The monomorphic extracted callback denotes the statement's arbitrary `merge`, receives the new
value first, and is invoked exactly once on the committing branch.  The call count is explicit:
pure callback extensionality alone cannot establish an execution-count claim.  `generatedIdentity`
is trivial for wrapping addition; `exactlyOnce` requires a source-bound commit trace.
-/
structure GrowableCallbackLaws (R : Type) (merge : I64 → I64 → I64)
    (mergeCalls : Nat) : Prop where
  carrier : ProductionWordCarrier R
  generatedIdentity : ∀ new old,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge () new old = ok (merge new old)
  newFirst : ∀ new old, merge new old = core.num.I64.wrapping_add new old
  identity : merge = core.num.I64.wrapping_add
  exactlyOnce : mergeCalls = 1

/-- A vacant decision exposes both its first-fit witness and its generated address location. -/
theorem queryAddressCorrect_vacant_firstFit {R : Type}
    {state : table.Table I64 I64 table.WordHome R 16#usize} {decode logical query decision direct
      order prefixOrder sweep checked address}
    (hsearch : SearchDecisionCorrect state decode logical query decision direct order prefixOrder
      sweep checked)
    (haddress : QueryAddressCorrect state query address decision)
    (hvacant : decision.class = .Vacant) :
    (∃ location, decisionLocation decision = some location ∧ FirstFit state order location) ∧
      (match decision.location with
      | .Flat index =>
          Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = some index.val
      | .Spill shard _ => table.shard address = ok shard
      | .NoIndex => False) :=
  ⟨hsearch.vacantExact.mp hvacant, haddress.vacantLocation hvacant⟩

/-- Project the generated address call used by a query-address certificate. -/
theorem queryAddressCorrect_generatedAddress {R : Type}
    {state : table.Table I64 I64 table.WordHome R 16#usize} {query address decision}
    (h : QueryAddressCorrect state query address decision) :
    Apeiron.GrowableIdentity.wordHomeInst.address state.home query.1 query.2 = ok address :=
  h.generatedAddress

/-- Project the exact generated checked-add condition for the vacant branch. -/
theorem freshLengthBound_vacant {R : Type}
    {state : table.Table I64 I64 table.WordHome R 16#usize} {decision}
    (h : FreshLengthBound state decision) (hvacant : decision.class = .Vacant) :
    state.len.val + 1 < Usize.max := h hvacant

/-- Project callback identity with the extracted wrapping-add operation. -/
theorem growableCallbackLaws_identity {R : Type} {merge : I64 → I64 → I64} {mergeCalls : Nat}
    (h : GrowableCallbackLaws R merge mergeCalls) :
    merge = core.num.I64.wrapping_add := h.identity

/-- Project the generated new-first callback equation and its single invocation. -/
theorem growableCallbackLaws_generated_once {R : Type} {merge : I64 → I64 → I64}
    {mergeCalls : Nat} (h : GrowableCallbackLaws R merge mergeCalls) :
    (∀ new old,
      table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge () new old = ok (merge new old)) ∧
      mergeCalls = 1 :=
  ⟨h.generatedIdentity, h.exactlyOnce⟩

/-- The generated monomorphic callback itself is new-first wrapping addition. -/
theorem wordMerge_newFirst (new old : I64) :
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge () new old =
      ok (core.num.I64.wrapping_add new old) := rfl

private theorem absent_of_not_found {R : Type}
    {state : table.Table I64 I64 table.WordHome R 16#usize} {decode logical query decision direct
      order prefixOrder sweep checked}
    (h : SearchDecisionCorrect state decode logical query decision direct order prefixOrder sweep
      checked)
    (hnfound : decision.class ≠ .Found) : Absent state decode query := by
  cases hclass : decision.class with
  | Found => exact False.elim (hnfound hclass)
  | Exhausted => exact (h.exhaustedExact.mp hclass).1
  | Vacant =>
      rcases h.absenceCertificate hnfound with
        ⟨location, hlocation, hdirect⟩ | ⟨hexcluded, hmiss⟩ | hcovered
      · obtain ⟨vacantLocation, hvacantLocation, hfirst⟩ := h.vacantExact.mp hclass
        have heq : location = vacantLocation := by
          rw [hlocation] at hvacantLocation
          exact Option.some.inj hvacantLocation
        subst vacantLocation
        exact h.directCover query location hdirect hfirst.1
      · intro location value hentry
        cases location with
        | flat index =>
            obtain ⟨location, hdirect⟩ := h.flatMatchAdmitted index value hentry
            rw [hexcluded] at hdirect
            cases hdirect
        | spill shard index =>
            obtain ⟨address, haddress, hshard⟩ := h.spillMatchRouted shard index value hentry
            rw [hmiss.generatedAddress] at haddress
            have haddrEq : address = hmiss.address := (Result.ok_injective haddress).symm
            rw [haddrEq] at hshard
            rw [hmiss.shardEq] at hshard
            have hshardEq : shard = hmiss.shard := (Result.ok_injective hshard).symm
            rw [hshardEq] at hentry
            exact hmiss.excludes index value hentry
      · intro location value hentry
        have hin := h.complete location (query, value) hentry
        apply hcovered location hin
        exact hentry.trans (congrArg (fun v => some (query, v))
          (h.foundDenotes location value hentry).symm)

private theorem apply_get_of_entry_stored
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (decision : table.SearchDecision) (query : LogicalKey) (location : Location) (value : I64)
    (hclass : decision.class = .Found) (hlocation : decisionLocation decision = some location)
    (hentry : entryAt state decodeStoredOption location = some (query, value)) :
    word_stored_apply_get state decision = ok value := by
  cases location with
  | flat index =>
      simp only [decisionLocation] at hlocation
      cases hloc : decision.location <;> simp_all
      simp only [entryAt, decodeStoredOption] at hentry
      split at hentry <;> simp_all
      rename_i slot hslot
      rcases hentry with ⟨hlive, hkey, hvalue⟩
      simp only [word_stored_apply_get, table.Table.apply_get, hloc, hclass,
        table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq,
        table.SearchClass.read_discriminant]
      simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hslot, hvalue]
  | spill shard index =>
      simp only [decisionLocation] at hlocation
      cases hloc : decision.location <;> simp_all
      simp only [entryAt, decodeStoredOption] at hentry
      split at hentry
      rename_i region hregion
      · split at hentry
        rename_i slot hslot
        · split at hentry
          · simp at hentry
          · rcases hentry with ⟨hkey, hvalue⟩
            simp only [word_stored_apply_get, table.Table.apply_get, hloc, hclass,
              table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq,
              table.SearchClass.read_discriminant]
            simp_all [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize]
        · simp at hentry
      · simp at hentry

/-- Stored-carrier found apply returns the exact canonical override. Refines `Apeiron.Table.get`. -/
theorem word_stored_apply_get_found {logical : Apeiron.Table.HMap LogicalKey I64}
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (query : LogicalKey) (decision : table.SearchDecision) (direct : LogicalKey → Option Location)
    (order prefixOrder sweep checked : List Location)
    (h : SearchDecisionCorrect state decodeStoredOption logical query decision direct order prefixOrder
      sweep checked) (hfound : decision.class = .Found) :
    word_stored_apply_get state decision = ok (Apeiron.Table.get logical query) := by
  obtain ⟨location, value, hlocation, hentry⟩ := h.foundExact.mp hfound
  rw [h.foundDenotes location value hentry]
  exact apply_get_of_entry_stored state decision query location value hfound hlocation hentry

/-- Stored-carrier absence returns the canonical default under `DirectCover`. Refines `get`. -/
theorem word_stored_apply_get_absent {logical : Apeiron.Table.HMap LogicalKey I64}
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (query : LogicalKey) (decision : table.SearchDecision) (direct : LogicalKey → Option Location)
    (order prefixOrder sweep checked : List Location)
    (h : SearchDecisionCorrect state decodeStoredOption logical query decision direct order prefixOrder
      sweep checked) (hnfound : decision.class ≠ .Found) :
    word_stored_apply_get state decision = ok (Apeiron.Table.get logical query) := by
  have habsent := absent_of_not_found h hnfound
  rw [h.absentDenotesDefault habsent]
  have hwf := h.wellFormed
  cases hc : decision.class <;> cases hl : decision.location <;>
    simp_all [word_stored_apply_get, table.Table.apply_get, DecisionWellFormed,
      table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq,
      table.SearchClass.read_discriminant]

/-- Stored-carrier decision/apply composition refines total-by-default `Apeiron.Table.get`. -/
theorem word_stored_search_decision_simulates {logical : Apeiron.Table.HMap LogicalKey I64}
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (query : LogicalKey) (decision : table.SearchDecision) (direct : LogicalKey → Option Location)
    (order prefixOrder sweep checked : List Location)
    (h : SearchDecisionCorrect state decodeStoredOption logical query decision direct order prefixOrder
      sweep checked) :
    word_stored_apply_get state decision = ok (Apeiron.Table.get logical query) := by
  by_cases hfound : decision.class = .Found
  · exact word_stored_apply_get_found state query decision direct order prefixOrder sweep checked h hfound
  · exact word_stored_apply_get_absent state query decision direct order prefixOrder sweep checked h
      hfound

private theorem apply_get_of_entry_regen
    (state : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (decision : table.SearchDecision) (query : LogicalKey) (location : Location) (value : I64)
    (hclass : decision.class = .Found) (hlocation : decisionLocation decision = some location)
    (hentry : entryAt state decodeRegenOption location = some (query, value)) :
    word_regen_apply_get state decision = ok value := by
  cases location with
  | flat index =>
      simp only [decisionLocation] at hlocation
      cases hloc : decision.location <;> simp_all
      simp only [entryAt] at hentry
      split at hentry <;> simp_all
      rename_i slot hslot
      split at hentry
      · rename_i decoded hdecode
        rcases hentry with ⟨hlive, hkey, hvalue⟩
        simp only [word_regen_apply_get, table.Table.apply_get, hloc, hclass,
          table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq,
          table.SearchClass.read_discriminant]
        simp_all [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize]
      · simp at hentry
  | spill shard index =>
      simp only [decisionLocation] at hlocation
      cases hloc : decision.location <;> simp_all
      simp only [entryAt] at hentry
      split at hentry
      rename_i region hregion
      · split at hentry
        rename_i slot hslot
        · split at hentry
          · simp at hentry
          · split at hentry
            · rename_i key hdecode
              rcases hentry with ⟨hkey, hvalue⟩
              simp only [word_regen_apply_get, table.Table.apply_get, hloc, hclass,
                table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq,
                table.SearchClass.read_discriminant]
              simp_all [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize]
            · simp at hentry
        · simp at hentry
      · simp at hentry

/-- Regenerating-carrier decision/apply composition refines total-by-default `Apeiron.Table.get`. -/
theorem word_regen_search_decision_simulates {logical : Apeiron.Table.HMap LogicalKey I64}
    (state : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (query : LogicalKey) (decision : table.SearchDecision) (direct : LogicalKey → Option Location)
    (order prefixOrder sweep checked : List Location)
    (h : SearchDecisionCorrect state decodeRegenOption logical query decision direct order prefixOrder
      sweep checked) :
    word_regen_apply_get state decision = ok (Apeiron.Table.get logical query) := by
  by_cases hfound : decision.class = .Found
  · obtain ⟨location, value, hlocation, hentry⟩ := h.foundExact.mp hfound
    rw [h.foundDenotes location value hentry]
    exact apply_get_of_entry_regen state decision query location value hfound hlocation hentry
  · have habsent := absent_of_not_found h hfound
    rw [h.absentDenotesDefault habsent]
    have hwf := h.wellFormed
    cases hc : decision.class <;> cases hl : decision.location <;>
      simp_all [word_regen_apply_get, table.Table.apply_get, DecisionWellFormed,
        table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq,
        table.SearchClass.read_discriminant]

/-- Replace one selected flat value without changing the other table fields. -/
def replaceFlatValue {R : Type} (state : table.Table I64 I64 table.WordHome R 16#usize)
    (index : Usize) (slot : table.FlatSlot R I64) :
    table.Table I64 I64 table.WordHome R 16#usize :=
  { default := state.default
    home := state.home
    resident := { lanes := state.resident.lanes, lane_bits := state.resident.lane_bits, slots := state.resident.slots.set index slot }
    shards := state.shards
    len := state.len
    key := state.key
    locator := state.locator }

/-- A stored found-flat apply commits the new-first merge to the selected real slot. -/
theorem word_stored_apply_insert_found_flat
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (index : Usize)
    (slot : table.FlatSlot (table.keyrep.Stored I64) I64)
    (hslot : state.resident.slots.val[index.val]? = some slot) :
    word_stored_apply_insert state { «class» := .Found, location := .Flat index }
        sigma key value address =
      ok (replaceFlatValue state index
        { slot with value := core.num.I64.wrapping_add value slot.value }, none) := by
  simp [word_stored_apply_insert, table.Table.apply_insert,
    replaceFlatValue,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize, hslot,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge]

/-- Reading the selected flat slot after the extracted update returns its replacement value. -/
theorem word_stored_apply_get_after_found_flat
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (index : Usize) (old replacement : table.FlatSlot (table.keyrep.Stored I64) I64)
    (hslot : state.resident.slots.val[index.val]? = some old) :
    word_stored_apply_get (replaceFlatValue state index replacement)
        { «class» := .Found, location := .Flat index } = ok replacement.value := by
  have hbound : index.val < state.resident.slots.length := by
    simpa using (List.getElem?_eq_some_iff.mp hslot).1
  simp [word_stored_apply_get, table.Table.apply_get, replaceFlatValue,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hbound,
    table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq,
    table.SearchClass.read_discriminant]

/-- A found flat commit composes with canonical `Apeiron.Table.get_insertWith_self`. -/
theorem word_stored_apply_insert_found_flat_self
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (index : Usize)
    (slot : table.FlatSlot (table.keyrep.Stored I64) I64)
    (hslot : state.resident.slots.val[index.val]? = some slot)
    (_hold : (slot.sigma, slot.held.key) = (sigma, key))
    (hvalue : slot.value = Apeiron.Table.get logical (sigma, key)) :
    word_stored_apply_insert state { «class» := .Found, location := .Flat index }
        sigma key value address ⦃ result =>
      result.2 = none ∧
      word_stored_apply_get result.1 { «class» := .Found, location := .Flat index } =
        ok (Apeiron.Table.get
          (Apeiron.Table.insertWith core.num.I64.wrapping_add (sigma, key) value logical)
          (sigma, key)) ⦄ := by
  rw [word_stored_apply_insert_found_flat state sigma key value address index slot hslot]
  simp only [WP.spec_ok, true_and]
  rw [Apeiron.Table.get_insertWith_self]
  simpa [Apeiron.Table.get, hvalue] using word_stored_apply_get_after_found_flat state index slot
    { slot with value := core.num.I64.wrapping_add value slot.value } hslot

/-- A vacant flat apply commits a new-first default merge and makes the real slot readable. -/
theorem word_stored_apply_insert_vacant_flat
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (index : Usize)
    (slot : table.FlatSlot (table.keyrep.Stored I64) I64)
    (hslot : state.resident.slots.val[index.val]? = some slot) (_hlive : slot.live = false)
    (nextLocator : table.TableLocator)
    (_hpublish : table.TableLocator.publish_for_query core.clone.CloneI64
      Apeiron.GrowableIdentity.wordHomeInst Apeiron.GrowableIdentity.storedKeyRep
      state.locator state.home sigma key (.Resident index) = ok nextLocator)
    (hlen : state.len.val + 1 < Usize.max) :
    word_stored_apply_insert state { «class» := .Vacant, location := .Flat index }
        sigma key value address ⦃ result =>
      result.2 = none ∧
      word_stored_apply_get result.1 { «class» := .Found, location := .Flat index } =
        ok (core.num.I64.wrapping_add value state.default) ⦄ := by
  have hbound : index.val < state.resident.slots.length := by
    simpa using (List.getElem?_eq_some_iff.mp hslot).1
  unfold word_stored_apply_insert table.Table.apply_insert
  simp_all [table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.hold,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge,
    alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
    word_stored_apply_get, table.Table.apply_get,
    table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq,
    table.SearchClass.read_discriminant]
  step
  simp_all

/-- A vacant flat commit composes with canonical `Apeiron.Table.get_insertWith_self`. -/
theorem word_stored_apply_insert_vacant_flat_self
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (index : Usize)
    (slot : table.FlatSlot (table.keyrep.Stored I64) I64)
    (hslot : state.resident.slots.val[index.val]? = some slot) (hlive : slot.live = false)
    (nextLocator : table.TableLocator)
    (hpublish : table.TableLocator.publish_for_query core.clone.CloneI64
      Apeiron.GrowableIdentity.wordHomeInst Apeiron.GrowableIdentity.storedKeyRep
      state.locator state.home sigma key (.Resident index) = ok nextLocator)
    (hlen : state.len.val + 1 < Usize.max)
    (hdefault : Apeiron.Table.get logical (sigma, key) = state.default) :
    word_stored_apply_insert state { «class» := .Vacant, location := .Flat index }
        sigma key value address ⦃ result =>
      result.2 = none ∧
      word_stored_apply_get result.1 { «class» := .Found, location := .Flat index } =
        ok (Apeiron.Table.get
          (Apeiron.Table.insertWith core.num.I64.wrapping_add (sigma, key) value logical)
          (sigma, key)) ⦄ := by
  apply WP.spec_mono (word_stored_apply_insert_vacant_flat state sigma key value address index slot
    hslot hlive nextLocator hpublish hlen)
  intro result hresult
  rw [Apeiron.Table.get_insertWith_self]
  have hd : logical (sigma, key) = state.default := by
    simpa [Apeiron.Table.get] using hdefault
  rw [hd]
  exact hresult

/-- A regenerating vacant-flat apply commits the default merge to the selected real slot. -/
theorem word_regen_apply_insert_vacant_flat
    (state : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (index : Usize)
    (slot : table.FlatSlot (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (hslot : state.resident.slots.val[index.val]? = some slot) (_hlive : slot.live = false)
    (nextLocator : table.TableLocator)
    (_hpublish : table.TableLocator.publish_for_query core.clone.CloneI64
      Apeiron.GrowableIdentity.wordHomeInst Apeiron.GrowableIdentity.regenKeyRep
      state.locator state.home sigma key (.Resident index) = ok nextLocator)
    (hlen : state.len.val + 1 < Usize.max) :
    word_regen_apply_insert state { «class» := .Vacant, location := .Flat index }
        sigma key value address ⦃ result =>
      result.2 = none ∧
      word_regen_apply_get result.1 { «class» := .Found, location := .Flat index } =
        ok (core.num.I64.wrapping_add value state.default) ⦄ := by
  have hbound : index.val < state.resident.slots.length := by
    simpa using (List.getElem?_eq_some_iff.mp hslot).1
  unfold word_regen_apply_insert table.Table.apply_insert
  simp_all [table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge,
    table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.hold,
    alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize,
    alloc.vec.Vec.index_usize]
  step with Apeiron.GrowableIdentity.codec_encode_at_identity sigma key address
  step
  simp_all [word_regen_apply_get, table.Table.apply_get,
    table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq,
    table.SearchClass.read_discriminant, alloc.vec.Vec.index_slice_index,
    alloc.vec.Vec.index_usize]

/-- A regenerating vacant flat commit composes with canonical `get_insertWith_self`. -/
theorem word_regen_apply_insert_vacant_flat_self
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (state : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (index : Usize)
    (slot : table.FlatSlot (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (hslot : state.resident.slots.val[index.val]? = some slot) (hlive : slot.live = false)
    (nextLocator : table.TableLocator)
    (hpublish : table.TableLocator.publish_for_query core.clone.CloneI64
      Apeiron.GrowableIdentity.wordHomeInst Apeiron.GrowableIdentity.regenKeyRep
      state.locator state.home sigma key (.Resident index) = ok nextLocator)
    (hlen : state.len.val + 1 < Usize.max)
    (hdefault : Apeiron.Table.get logical (sigma, key) = state.default) :
    word_regen_apply_insert state { «class» := .Vacant, location := .Flat index }
        sigma key value address ⦃ result =>
      result.2 = none ∧
      word_regen_apply_get result.1 { «class» := .Found, location := .Flat index } =
        ok (Apeiron.Table.get
          (Apeiron.Table.insertWith core.num.I64.wrapping_add (sigma, key) value logical)
          (sigma, key)) ⦄ := by
  apply WP.spec_mono (word_regen_apply_insert_vacant_flat state sigma key value address index slot
    hslot hlive nextLocator hpublish hlen)
  intro result hresult
  rw [Apeiron.Table.get_insertWith_self]
  have hd : logical (sigma, key) = state.default := by
    simpa [Apeiron.Table.get] using hdefault
  rw [hd]
  exact hresult

/-- A regenerating found-flat apply commits the new-first merge to the selected real slot. -/
theorem word_regen_apply_insert_found_flat
    (state : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (index : Usize)
    (slot : table.FlatSlot (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (hslot : state.resident.slots.val[index.val]? = some slot) :
    word_regen_apply_insert state { «class» := .Found, location := .Flat index }
        sigma key value address =
      ok (replaceFlatValue state index
        { slot with value := core.num.I64.wrapping_add value slot.value }, none) := by
  simp [word_regen_apply_insert, table.Table.apply_insert, replaceFlatValue,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize, hslot,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge]

/-- A regenerating found flat commit composes with canonical `get_insertWith_self`. -/
theorem word_regen_apply_insert_found_flat_self
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (state : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (index : Usize)
    (slot : table.FlatSlot (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (hslot : state.resident.slots.val[index.val]? = some slot)
    (_hold : entryAt state decodeRegenOption (.flat index) = some (((sigma, key)), slot.value))
    (hvalue : slot.value = Apeiron.Table.get logical (sigma, key)) :
    word_regen_apply_insert state { «class» := .Found, location := .Flat index }
        sigma key value address ⦃ result =>
      result.2 = none ∧
      word_regen_apply_get result.1 { «class» := .Found, location := .Flat index } =
        ok (Apeiron.Table.get
          (Apeiron.Table.insertWith core.num.I64.wrapping_add (sigma, key) value logical)
          (sigma, key)) ⦄ := by
  rw [word_regen_apply_insert_found_flat state sigma key value address index slot hslot]
  simp only [WP.spec_ok, true_and]
  rw [Apeiron.Table.get_insertWith_self]
  have hbound : index.val < state.resident.slots.length := by
    simpa using (List.getElem?_eq_some_iff.mp hslot).1
  simp [word_regen_apply_get, table.Table.apply_get, replaceFlatValue,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hbound,
    table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq,
    table.SearchClass.read_discriminant, Apeiron.Table.get, hvalue]

/-- Replace one selected spill slot without changing unrelated table fields. -/
def replaceSpillValue {R : Type} (state : table.Table I64 I64 table.WordHome R 16#usize)
    (shard : Usize) (region : table.Region R I64) (index : Usize)
    (slot : table.SpillSlot R I64) : table.Table I64 I64 table.WordHome R 16#usize :=
  { default := state.default
    home := state.home
    resident := state.resident
    shards := state.shards.set shard
      { slots := region.slots.set index slot, coords := region.coords,
        bits := region.bits, live := region.live }
    len := state.len
    key := state.key
    locator := state.locator }

/-- A prepared spill publication succeeds and exposes the written payload fields at its selected
index. The generated live tag is written independently of the payload fields. -/
def PublishReadable {R : Type} (region : table.Region R I64) (index : Usize)
    (payload : cell.Sigma × R × I64) (place : U64) : Prop :=
  ∃ nextRegion slot, table.Region.publish_live region index payload place = ok nextRegion ∧
    nextRegion.slots.val[index.val]? = some slot ∧
    slot.sigma = payload.1 ∧ slot.held = payload.2.1 ∧ slot.value = payload.2.2

/-- A stored found-spill apply commits the new-first merge to the selected real slot. -/
theorem word_stored_apply_insert_found_spill
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (shard index : Usize)
    (region : table.Region (table.keyrep.Stored I64) I64)
    (slot : table.SpillSlot (table.keyrep.Stored I64) I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (hslot : region.slots.val[index.val]? = some slot) :
    word_stored_apply_insert state { «class» := .Found, location := .Spill shard index }
        sigma key value address =
      ok (replaceSpillValue state shard region index
        { slot with value := core.num.I64.wrapping_add value slot.value }, none) := by
  simp [word_stored_apply_insert, table.Table.apply_insert, replaceSpillValue,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize, hregion, hslot,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge]

/-- A regenerating found-spill apply commits the new-first merge to the selected real slot. -/
theorem word_regen_apply_insert_found_spill
    (state : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (shard index : Usize)
    (region : table.Region (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (slot : table.SpillSlot (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (hslot : region.slots.val[index.val]? = some slot) :
    word_regen_apply_insert state { «class» := .Found, location := .Spill shard index }
        sigma key value address =
      ok (replaceSpillValue state shard region index
        { slot with value := core.num.I64.wrapping_add value slot.value }, none) := by
  simp [word_regen_apply_insert, table.Table.apply_insert, replaceSpillValue,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize, hregion, hslot,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge]

/-- A stored found spill commit composes with canonical `get_insertWith_self`. -/
theorem word_stored_apply_insert_found_spill_self
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (shard index : Usize)
    (region : table.Region (table.keyrep.Stored I64) I64)
    (slot : table.SpillSlot (table.keyrep.Stored I64) I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (hslot : region.slots.val[index.val]? = some slot)
    (_hold : (slot.sigma, slot.held.key) = (sigma, key))
    (hvalue : slot.value = Apeiron.Table.get logical (sigma, key)) :
    word_stored_apply_insert state { «class» := .Found, location := .Spill shard index }
        sigma key value address ⦃ result =>
      result.2 = none ∧
      word_stored_apply_get result.1 { «class» := .Found, location := .Spill shard index } =
        ok (Apeiron.Table.get
          (Apeiron.Table.insertWith core.num.I64.wrapping_add (sigma, key) value logical)
          (sigma, key)) ⦄ := by
  rw [word_stored_apply_insert_found_spill state sigma key value address shard index region slot
    hregion hslot]
  simp only [WP.spec_ok, true_and]
  rw [Apeiron.Table.get_insertWith_self]
  have hshard : shard.val < state.shards.length := by
    simpa using (List.getElem?_eq_some_iff.mp hregion).1
  have hindex : index.val < region.slots.length := by
    simpa using (List.getElem?_eq_some_iff.mp hslot).1
  simp [word_stored_apply_get, table.Table.apply_get, replaceSpillValue,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hshard, hindex,
    table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq,
    table.SearchClass.read_discriminant, Apeiron.Table.get, hvalue]

/-- A regenerating found spill commit composes with canonical `get_insertWith_self`. -/
theorem word_regen_apply_insert_found_spill_self
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (state : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (shard index : Usize)
    (region : table.Region (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (slot : table.SpillSlot (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (hslot : region.slots.val[index.val]? = some slot)
    (_hold : decodeRegenOption slot.sigma slot.held = some key)
    (hvalue : slot.value = Apeiron.Table.get logical (sigma, key)) :
    word_regen_apply_insert state { «class» := .Found, location := .Spill shard index }
        sigma key value address ⦃ result =>
      result.2 = none ∧
      word_regen_apply_get result.1 { «class» := .Found, location := .Spill shard index } =
        ok (Apeiron.Table.get
          (Apeiron.Table.insertWith core.num.I64.wrapping_add (sigma, key) value logical)
          (sigma, key)) ⦄ := by
  rw [word_regen_apply_insert_found_spill state sigma key value address shard index region slot
    hregion hslot]
  simp only [WP.spec_ok, true_and]
  rw [Apeiron.Table.get_insertWith_self]
  have hshard : shard.val < state.shards.length := by
    simpa using (List.getElem?_eq_some_iff.mp hregion).1
  have hindex : index.val < region.slots.length := by
    simpa using (List.getElem?_eq_some_iff.mp hslot).1
  simp [word_regen_apply_get, table.Table.apply_get, replaceSpillValue,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hshard, hindex,
    table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq,
    table.SearchClass.read_discriminant, Apeiron.Table.get, hvalue]

/-- A stored vacant-spill apply commits the default merge and makes the real slot readable. -/
theorem word_stored_apply_insert_vacant_spill
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (shard index : Usize)
    (region : table.Region (table.keyrep.Stored I64) I64)
    (slot : table.SpillSlot (table.keyrep.Stored I64) I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (_htag : slot.tag = 0#u8)
    (hslot : region.slots.val[index.val]? = some slot)
    (nextLocator : table.TableLocator)
    (_hpublish : table.TableLocator.publish_for_query core.clone.CloneI64
      Apeiron.GrowableIdentity.wordHomeInst Apeiron.GrowableIdentity.storedKeyRep
      state.locator state.home sigma key (.Spill shard index) = ok nextLocator)
    (hpublishPayload : PublishReadable region index
      (sigma, { key := key }, core.num.I64.wrapping_add value state.default) address.place)
    (hlen : state.len.val + 1 < Usize.max) :
    word_stored_apply_insert state { «class» := .Vacant, location := .Spill shard index }
        sigma key value address ⦃ result =>
      result.2 = none ∧
      word_stored_apply_get result.1 { «class» := .Found, location := .Spill shard index } =
        ok (core.num.I64.wrapping_add value state.default) ⦄ := by
  have hshard : shard.val < state.shards.length := by
    simpa using (List.getElem?_eq_some_iff.mp hregion).1
  have hslotBound : index.val < region.slots.length := by
    simpa using (List.getElem?_eq_some_iff.mp hslot).1
  unfold word_stored_apply_insert table.Table.apply_insert
  simp_all [table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.hold,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge,
    alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize,
    alloc.vec.Vec.index_usize]
  obtain ⟨nextRegion, nextSlot, hnextRegion, hnextSlot, hsigma, hheld, hvalue⟩ :=
    hpublishPayload
  rw [hnextRegion]
  step
  simp_all [word_stored_apply_get, table.Table.apply_get,
    table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq,
    table.SearchClass.read_discriminant, alloc.vec.Vec.index_slice_index,
    alloc.vec.Vec.index_usize]

/-- A regenerating vacant-spill apply commits the default merge and makes the slot readable. -/
theorem word_regen_apply_insert_vacant_spill
    (state : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (shard index : Usize)
    (region : table.Region (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (slot : table.SpillSlot (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (_htag : slot.tag = 0#u8)
    (hslot : region.slots.val[index.val]? = some slot)
    (nextLocator : table.TableLocator)
    (_hpublish : table.TableLocator.publish_for_query core.clone.CloneI64
      Apeiron.GrowableIdentity.wordHomeInst Apeiron.GrowableIdentity.regenKeyRep
      state.locator state.home sigma key (.Spill shard index) = ok nextLocator)
    (hpublishPayload : ∀ identity,
      PublishReadable region index
        (sigma, { identity := identity, codec := () },
          core.num.I64.wrapping_add value state.default) address.place)
    (hlen : state.len.val + 1 < Usize.max) :
    word_regen_apply_insert state { «class» := .Vacant, location := .Spill shard index }
        sigma key value address ⦃ result =>
      result.2 = none ∧
      word_regen_apply_get result.1 { «class» := .Found, location := .Spill shard index } =
        ok (core.num.I64.wrapping_add value state.default) ⦄ := by
  have hshard : shard.val < state.shards.length := by
    simpa using (List.getElem?_eq_some_iff.mp hregion).1
  have hslotBound : index.val < region.slots.length := by
    simpa using (List.getElem?_eq_some_iff.mp hslot).1
  obtain ⟨identity, hencode, _⟩ := WP.spec_imp_exists
    (Apeiron.GrowableIdentity.codec_encode_at_identity sigma key address)
  unfold word_regen_apply_insert table.Table.apply_insert
  simp_all [table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.hold,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge,
    alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize,
    alloc.vec.Vec.index_usize]
  obtain ⟨nextRegion, nextSlot, hnextRegion, hnextSlot, hsigma, hheld, hvalue⟩ :=
    hpublishPayload identity
  rw [hnextRegion]
  step
  simp_all [word_regen_apply_get, table.Table.apply_get,
    table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq,
    table.SearchClass.read_discriminant, alloc.vec.Vec.index_slice_index,
    alloc.vec.Vec.index_usize]

/-- A stored vacant spill commit composes with canonical `get_insertWith_self`. -/
theorem word_stored_apply_insert_vacant_spill_self
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (shard index : Usize)
    (region : table.Region (table.keyrep.Stored I64) I64)
    (slot : table.SpillSlot (table.keyrep.Stored I64) I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (htag : slot.tag = 0#u8)
    (hslot : region.slots.val[index.val]? = some slot)
    (nextLocator : table.TableLocator)
    (hpublish : table.TableLocator.publish_for_query core.clone.CloneI64
      Apeiron.GrowableIdentity.wordHomeInst Apeiron.GrowableIdentity.storedKeyRep
      state.locator state.home sigma key (.Spill shard index) = ok nextLocator)
    (hpublishPayload : PublishReadable region index
      (sigma, { key := key }, core.num.I64.wrapping_add value state.default) address.place)
    (hlen : state.len.val + 1 < Usize.max)
    (hdefault : Apeiron.Table.get logical (sigma, key) = state.default) :
    word_stored_apply_insert state { «class» := .Vacant, location := .Spill shard index }
        sigma key value address ⦃ result =>
      result.2 = none ∧
      word_stored_apply_get result.1 { «class» := .Found, location := .Spill shard index } =
        ok (Apeiron.Table.get
          (Apeiron.Table.insertWith core.num.I64.wrapping_add (sigma, key) value logical)
          (sigma, key)) ⦄ := by
  apply WP.spec_mono (word_stored_apply_insert_vacant_spill state sigma key value address shard index
    region slot hregion htag hslot nextLocator hpublish hpublishPayload hlen)
  intro result hresult
  rw [Apeiron.Table.get_insertWith_self]
  have hd : logical (sigma, key) = state.default := by
    simpa [Apeiron.Table.get] using hdefault
  rw [hd]
  exact hresult

/-- A regenerating vacant spill commit composes with canonical `get_insertWith_self`. -/
theorem word_regen_apply_insert_vacant_spill_self
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (state : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (shard index : Usize)
    (region : table.Region (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (slot : table.SpillSlot (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (htag : slot.tag = 0#u8)
    (hslot : region.slots.val[index.val]? = some slot)
    (nextLocator : table.TableLocator)
    (hpublish : table.TableLocator.publish_for_query core.clone.CloneI64
      Apeiron.GrowableIdentity.wordHomeInst Apeiron.GrowableIdentity.regenKeyRep
      state.locator state.home sigma key (.Spill shard index) = ok nextLocator)
    (hpublishPayload : ∀ identity,
      PublishReadable region index
        (sigma, { identity := identity, codec := () },
          core.num.I64.wrapping_add value state.default) address.place)
    (hlen : state.len.val + 1 < Usize.max)
    (hdefault : Apeiron.Table.get logical (sigma, key) = state.default) :
    word_regen_apply_insert state { «class» := .Vacant, location := .Spill shard index }
        sigma key value address ⦃ result =>
      result.2 = none ∧
      word_regen_apply_get result.1 { «class» := .Found, location := .Spill shard index } =
        ok (Apeiron.Table.get
          (Apeiron.Table.insertWith core.num.I64.wrapping_add (sigma, key) value logical)
          (sigma, key)) ⦄ := by
  apply WP.spec_mono (word_regen_apply_insert_vacant_spill state sigma key value address shard index
    region slot hregion htag hslot nextLocator hpublish hpublishPayload hlen)
  intro result hresult
  rw [Apeiron.Table.get_insertWith_self]
  have hd : logical (sigma, key) = state.default := by
    simpa [Apeiron.Table.get] using hdefault
  rw [hd]
  exact hresult

/-- An exhausted stored insert performs no commit and returns the pending entry unchanged. -/
theorem word_stored_apply_insert_needsGrowth
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) :
    word_stored_apply_insert state { «class» := .Exhausted, location := .NoIndex }
        sigma key value address = ok (state, some (key, value)) := by
  simp [word_stored_apply_insert, table.Table.apply_insert]

/-- An exhausted regenerating insert performs no commit and returns the pending entry unchanged. -/
theorem word_regen_apply_insert_needsGrowth
    (state : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) :
    word_regen_apply_insert state { «class» := .Exhausted, location := .NoIndex }
        sigma key value address = ok (state, some (key, value)) := by
  simp [word_regen_apply_insert, table.Table.apply_insert]

/--
Both extracted read applies and the canonical write observations compose under the explicit decision
premises. The branch theorems above establish the real `Committed` and `NeedsGrowth` apply effects;
this theorem names the canonical `get`, `get_insertWith_self`, and `get_insertWith_ne` roots.
-/
theorem search_decision_simulates
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (stored : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (regen : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (query : LogicalKey) (storedDecision regenDecision : table.SearchDecision)
    (direct : LogicalKey → Option Location) (order prefixOrder sweep checked : List Location)
    (hstored : SearchDecisionCorrect stored decodeStoredOption logical query storedDecision direct
      order prefixOrder sweep checked)
    (hregen : SearchDecisionCorrect regen decodeRegenOption logical query regenDecision direct order
      prefixOrder sweep checked) :
    word_stored_apply_get stored storedDecision = ok (Apeiron.Table.get logical query) ∧
    word_regen_apply_get regen regenDecision = ok (Apeiron.Table.get logical query) ∧
    (∀ value, Apeiron.Table.get
      (Apeiron.Table.insertWith core.num.I64.wrapping_add query value logical) query =
        core.num.I64.wrapping_add value (Apeiron.Table.get logical query)) ∧
    (∀ value other, other ≠ query → Apeiron.Table.get
      (Apeiron.Table.insertWith core.num.I64.wrapping_add query value logical) other =
        Apeiron.Table.get logical other) := by
  refine ⟨word_stored_search_decision_simulates stored query storedDecision direct order prefixOrder
      sweep checked hstored,
    word_regen_search_decision_simulates regen query regenDecision direct order prefixOrder sweep
      checked hregen, ?_, ?_⟩
  · exact fun value => Apeiron.Table.get_insertWith_self core.num.I64.wrapping_add query value logical
  · exact fun value other hne =>
      Apeiron.Table.get_insertWith_ne core.num.I64.wrapping_add query value logical hne

/-- Canonical readback at the committed key is `get_insertWith_self`. -/
theorem committed_insertWith_self (logical : Apeiron.Table.HMap LogicalKey I64)
    (query : LogicalKey) (value : I64) :
    Apeiron.Table.get
        (Apeiron.Table.insertWith core.num.I64.wrapping_add query value logical) query =
      core.num.I64.wrapping_add value (Apeiron.Table.get logical query) := by
  exact Apeiron.Table.get_insertWith_self core.num.I64.wrapping_add query value logical

/-- Canonical readback at every unequal key is `get_insertWith_ne`. -/
theorem committed_insertWith_ne (logical : Apeiron.Table.HMap LogicalKey I64)
    (query other : LogicalKey) (value : I64) (hne : other ≠ query) :
    Apeiron.Table.get
        (Apeiron.Table.insertWith core.num.I64.wrapping_add query value logical) other =
      Apeiron.Table.get logical other := by
  exact Apeiron.Table.get_insertWith_ne core.num.I64.wrapping_add query value logical hne

end Apeiron.GrowableSearch
