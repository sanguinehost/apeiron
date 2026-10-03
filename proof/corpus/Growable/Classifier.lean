import Growable.Representation
import Lean

/-!
# Row 7 producer bridges: `RepR` certificates to `Growable.Search` interfaces

Row 7 of the read map (`docs/epics/read-proof.md`) splits into the `Growable.Search`
interface (landed) and this producer module.  Each bridge below derives a `Growable.Search`
certificate from `Apeiron.GrowableRepresentation.RepR` fields and the generated route facts.  No
premise is a caller-supplied `Found`/`Absent` result and no desired conclusion is assumed:

1. `flatMatchAdmitted_of_repR` sources flat direct-admission from `RepR.address` (flat case via
   `GrowableGeometry.preparedFlatIndex`) and `RepR.directIndex`;
2. `spillMatchRouted_of_repR` sources the routed shard from `RepR.address` (spill case).

The retired bridges `summaryMissSound_of_repR`, `repR_absenceCertificate_of_summaryMiss` and
`repR_absent_of_summaryMiss` are gone with `GrowableRepresentation.SpillSummaryMissSound`: their
premise was a negative read of the generated occupancy summary, and `table.Region.summary_maybe`,
`table.summary_trailing_zeros` and `table.scan.*` no longer exist in the extraction.  Their
replacement is `Apeiron.GrowableSearch.AbsenceCertificate.coverage` together with
`Apeiron.GrowableSearch.CompleteCoverage`, fed by `Apeiron.GrowableScan.spill_probe_bound_absent_sound`
over a completed bounded cycle; the producer of that `checked` list is `Growable.Read`/`Growable.ReadLoop`.

The remaining `SearchDecisionCorrect` fields (exact first fit, `DecisionSymbolicCorrespondence`,
`FreshLengthBound`, callback laws) are downstream producer work, not asserted here.
-/

namespace Apeiron.GrowableClassifier

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

abbrev LogicalKey := Apeiron.GrowableSearch.LogicalKey
abbrev Location := Apeiron.GrowableSearch.Location
abbrev WordTable := Apeiron.GrowableRepresentation.WordTable

/--
2. A live flat representative's key is direct-admitted: `RepR.address` places the entry at the
generated `preparedFlatIndex` for its own key, and `RepR.directIndex` turns that route into
`direct query = some location`.
-/
theorem flatMatchAdmitted_of_repR {R : Type}
    {state : WordTable R} {decode : cell.Sigma → R → Option I64}
    {direct : LogicalKey → Option Location} {live : LogicalKey → Prop}
    {logical : Apeiron.Table.HMap LogicalKey I64}
    (hrep : Apeiron.GrowableRepresentation.RepR state decode direct live logical)
    (query : LogicalKey) (index : Usize) (value : I64)
    (hentry : Apeiron.GrowableSearch.entryAt state decode (.flat index) = some (query, value)) :
    ∃ location, direct query = some location := by
  obtain ⟨address, haddress, hflat⟩ := hrep.address (.flat index) query value hentry
  exact ⟨.flat index,
    (hrep.directIndex query (.flat index)).2 ⟨address, index, haddress, hflat, rfl⟩⟩

/--
3. A live spill representative routes to its own shard through the generated address: this is the
spill case of `RepR.address`.
-/
theorem spillMatchRouted_of_repR {R : Type}
    {state : WordTable R} {decode : cell.Sigma → R → Option I64}
    {direct : LogicalKey → Option Location} {live : LogicalKey → Prop}
    {logical : Apeiron.Table.HMap LogicalKey I64}
    (hrep : Apeiron.GrowableRepresentation.RepR state decode direct live logical)
    (query : LogicalKey) (shard index : Usize) (value : I64)
    (hentry : Apeiron.GrowableSearch.entryAt state decode (.spill shard index) =
      some (query, value)) :
    ∃ address, Apeiron.GrowableIdentity.wordHomeInst.address state.home query.1 query.2 =
        ok address ∧ table.shard address = ok shard :=
  hrep.address (.spill shard index) query value hentry

/-! ## Generated decision provenance

The real `table.Table.search_decision` produces its decision through exactly three branches:
`found_flat` on the direct route with the body's own live/`Sigma`/`matches` guards, `vacant_flat` on
a non-live direct slot, and the `spill_locate` route through `SearchDecision.from_spill`.  The
provenance below carries those branch witnesses, so downstream exactness claims can consume the
generated guards instead of assuming a decision shape.
-/

/-- The provenance of a generated search decision: the real `table.Table.search_decision` branch
that produced it, with that branch's `Result` successes and guards carried. -/
inductive DecisionProvenance {K V H R : Type} {N : Usize}
    (keyrep : table.keyrep.KeyRep R K H) (self : table.Table K V H R N) (sigma : cell.Sigma)
    (key : K) (address : table.TableAddress) : table.SearchDecision → Prop where
  | foundFlat (index : Usize) (slot : table.FlatSlot R V)
      (hslot : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot R V))
        self.resident.slots index = ok slot)
      (hlive : slot.live = true)
      (hsigma : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slot.sigma sigma = ok true)
      (hmatch : keyrep.«matches» sigma slot.held key address = ok true) :
      DecisionProvenance keyrep self sigma key address
        { «class» := table.SearchClass.Found, location := table.SearchLocation.Flat index }
  | vacantFlat (index : Usize) (slot : table.FlatSlot R V)
      (hslot : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot R V))
        self.resident.slots index = ok slot)
      (hlive : slot.live = false) :
      DecisionProvenance keyrep self sigma key address
        { «class» := table.SearchClass.Vacant, location := table.SearchLocation.Flat index }
  | spillRoute (shard : Usize) (region : table.Region R V) (site : table.SpillSite)
      (decision : table.SearchDecision)
      (hshard : table.shard address = ok shard)
      (hregion : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.Region R V))
        self.shards shard = ok region)
      (hsite : table.probe.spill_locate keyrep region sigma address key N = ok site)
      (hfrom : table.SearchDecision.from_spill shard site = ok decision) :
      DecisionProvenance keyrep self sigma key address decision

/-- The spill route body shared by the `none` direct word, absent flat index and guard-mismatch
branches of `table.Table.search_decision`. -/
private theorem spillRoute_provenance {K V H R : Type} {N : Usize}
    (keyrep : table.keyrep.KeyRep R K H) (self : table.Table K V H R N) (sigma : cell.Sigma)
    (key : K) (address : table.TableAddress) (decision : table.SearchDecision)
    (hrun : (do
        let shard_at ← table.shard address
        let r ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.Region R V))
          self.shards shard_at
        let ss ← table.probe.spill_locate keyrep r sigma address key N
        table.SearchDecision.from_spill shard_at ss) = ok decision) :
    DecisionProvenance keyrep self sigma key address decision := by
  cases hshard : table.shard address with
  | vis eff k => simp [hshard] at hrun
  | div => simp [hshard] at hrun
  | ret shard =>
      simp only [hshard, bind_tc_ok] at hrun
      cases hregion : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.Region R V))
          self.shards shard with
      | vis eff k => simp [hregion] at hrun
      | div => simp [hregion] at hrun
      | ret r =>
          simp only [hregion, bind_tc_ok] at hrun
          cases hsite : table.probe.spill_locate keyrep r sigma address key N with
          | vis eff k => simp [hsite] at hrun
          | div => simp [hsite] at hrun
          | ret ss =>
              simp only [hsite, bind_tc_ok] at hrun
              exact .spillRoute shard r ss decision hshard hregion hsite hrun

/-- Every successful generated `search_decision` run carries its real branch provenance. -/
theorem search_decision_provenance {K V H R F : Type} {N : Usize}
    (HomeInst : table.Home H K F) (keyrep : table.keyrep.KeyRep R K H)
    (self : table.Table K V H R N) (sigma : cell.Sigma) (key : K) (address : table.TableAddress)
    (decision : table.SearchDecision)
    (hrun : table.Table.search_decision HomeInst keyrep self sigma key address = ok decision) :
    DecisionProvenance keyrep self sigma key address decision := by
  simp only [table.Table.search_decision] at hrun
  cases hdirect : address.direct_word with
  | none =>
      simp only [hdirect] at hrun
      exact spillRoute_provenance keyrep self sigma key address decision hrun
  | some word =>
      simp only [hdirect] at hrun
      cases hflat : table.Flat.index self.resident word with
      | vis eff k => simp [hflat] at hrun
      | div => simp [hflat] at hrun
      | ret o =>
          simp only [hflat, bind_tc_ok] at hrun
          cases o with
          | none =>
              exact spillRoute_provenance keyrep self sigma key address decision hrun
          | some index =>
              cases hslot : alloc.vec.Vec.index
                  (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot R V))
                  self.resident.slots index with
              | vis eff k => simp [hslot] at hrun
              | div => simp [hslot] at hrun
              | ret slot =>
                  simp only [hslot, bind_tc_ok] at hrun
                  cases hlive : slot.live with
                  | false =>
                      simp only [hlive] at hrun
                      simp only [table.SearchDecision.vacant_flat] at hrun
                      have hdec : table.SearchDecision.mk table.SearchClass.Vacant (table.SearchLocation.Flat index) = decision := by
                        simpa using Result.ok_injective hrun
                      subst hdec
                      exact .vacantFlat index slot hslot hlive
                  | true =>
                      simp only [hlive, ↓reduceIte] at hrun
                      cases hsigma : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slot.sigma sigma with
                      | vis eff k => simp [hsigma] at hrun
                      | div => simp [hsigma] at hrun
                      | ret b =>
                          simp only [hsigma, bind_tc_ok] at hrun
                          cases hb : b with
                          | false =>
                              simp only [hb] at hrun
                              exact spillRoute_provenance keyrep self sigma key address decision
                                hrun
                          | true =>
                              simp only [hb, ↓reduceIte] at hrun
                              rw [hb] at hsigma
                              cases hmatch : keyrep.«matches» sigma slot.held key address with
                              | vis eff k => simp [hmatch] at hrun
                              | div => simp [hmatch] at hrun
                              | ret b1 =>
                                  simp only [hmatch, bind_tc_ok] at hrun
                                  cases hb1 : b1 with
                                  | false =>
                                      simp only [hb1] at hrun
                                      exact spillRoute_provenance keyrep self sigma key address
                                        decision hrun
                                  | true =>
                                      simp only [hb1, ↓reduceIte] at hrun
                                      rw [hb1] at hmatch
                                      simp only [table.SearchDecision.found_flat] at hrun
                                      have hdec : table.SearchDecision.mk table.SearchClass.Found
                                          (table.SearchLocation.Flat index) = decision := by
                                        simpa using Result.ok_injective hrun
                                      subst hdec
                                      exact .foundFlat index slot hslot hlive hsigma hmatch

/-- Every successful generated `search_decision` result is well formed. -/
theorem search_decision_wellFormed {K V H R F : Type} {N : Usize}
    (HomeInst : table.Home H K F) (keyrep : table.keyrep.KeyRep R K H)
    (self : table.Table K V H R N) (sigma : cell.Sigma) (key : K) (address : table.TableAddress)
    (decision : table.SearchDecision)
    (hrun : table.Table.search_decision HomeInst keyrep self sigma key address = ok decision) :
    Apeiron.GrowableSearch.DecisionWellFormed decision := by
  cases search_decision_provenance HomeInst keyrep self sigma key address decision hrun with
  | foundFlat _ _ _ _ _ _ => simp [Apeiron.GrowableSearch.DecisionWellFormed]
  | vacantFlat _ _ _ _ => simp [Apeiron.GrowableSearch.DecisionWellFormed]
  | spillRoute shard region site decision hshard hregion hsite hfrom =>
      cases site with
      | Found i =>
          simp only [table.SearchDecision.from_spill] at hfrom
          have hdec : table.SearchDecision.mk table.SearchClass.Found (table.SearchLocation.Spill shard i) = decision := by
            simpa using Result.ok_injective hfrom
          subst hdec
          simp [Apeiron.GrowableSearch.DecisionWellFormed]
      | Vacant i =>
          simp only [table.SearchDecision.from_spill] at hfrom
          have hdec : table.SearchDecision.mk table.SearchClass.Vacant (table.SearchLocation.Spill shard i) = decision := by
            simpa using Result.ok_injective hfrom
          subst hdec
          simp [Apeiron.GrowableSearch.DecisionWellFormed]
      | Grow =>
          simp only [table.SearchDecision.from_spill] at hfrom
          have hdec : table.SearchDecision.mk table.SearchClass.Exhausted
              table.SearchLocation.NoIndex = decision := by
            simpa using Result.ok_injective hfrom
          subst hdec
          simp [Apeiron.GrowableSearch.DecisionWellFormed]

/-- Forward found exactness on the flat route: a `.Found` decision at `.Flat index` is produced only
by the `found_flat` branch, whose own live bit, `Sigma` guard and `matches` guard are carried. -/
theorem search_decision_found_flat_guards {K V H R : Type} {N : Usize}
    {keyrep : table.keyrep.KeyRep R K H} {self : table.Table K V H R N} {sigma : cell.Sigma}
    {key : K} {address : table.TableAddress} {decision : table.SearchDecision}
    (index : Usize)
    (h : DecisionProvenance keyrep self sigma key address decision)
    (hclass : decision.class = table.SearchClass.Found)
    (hloc : decision.location = table.SearchLocation.Flat index) :
    ∃ slot, alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot R V))
        self.resident.slots index = ok slot ∧
      slot.live = true ∧
      cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slot.sigma sigma = ok true ∧
      keyrep.«matches» sigma slot.held key address = ok true := by
  cases h with
  | foundFlat index' slot hslot hlive hsigma hmatch =>
      simp only at hloc
      have hinj : index' = index := by injection hloc
      rw [← hinj]
      exact ⟨slot, hslot, hlive, hsigma, hmatch⟩
  | vacantFlat index' slot hslot hlive =>
      simp only at hclass
      cases hclass
  | spillRoute shard region site decision hshard hregion hsite hfrom =>
      cases site with
      | Found i =>
          simp only [table.SearchDecision.from_spill] at hfrom
          have hdec : table.SearchDecision.mk table.SearchClass.Found (table.SearchLocation.Spill shard i) = decision := by
            simpa using Result.ok_injective hfrom
          subst hdec
          simp only at hloc
          cases hloc
      | Vacant i =>
          simp only [table.SearchDecision.from_spill] at hfrom
          have hdec : table.SearchDecision.mk table.SearchClass.Vacant (table.SearchLocation.Spill shard i) = decision := by
            simpa using Result.ok_injective hfrom
          subst hdec
          simp only at hclass
          cases hclass
      | Grow =>
          simp only [table.SearchDecision.from_spill] at hfrom
          have hdec : table.SearchDecision.mk table.SearchClass.Exhausted
              table.SearchLocation.NoIndex = decision := by
            simpa using Result.ok_injective hfrom
          subst hdec
          simp only at hclass
          cases hclass

/-- The exact missing spill-route lemma, stated but not proved: a `.Found` hit returned by the
generated `table.probe.spill_locate` carries a full-key match certificate. `SpillSite.Found index`
transports no payload, so the certificate must come from the loop invariant of
`table.probe.spill_locate_loop0_loop0.body`, whose `matched` flag is exactly
`table.matches keyrep payload sigma key address = ok true` at the returned lane. Needed premises:
the inner loop's completion invariant (`done (true, i1)` only from a `matched = true` step), the
tag-hit lane's payload index in bounds (`base + least_set_lane hits < slots.length`) and full-key
`matches` exactness at that payload. -/
def SpillLocateFoundFullMatch {K V H R : Type} {N : Usize}
    (keyrep : table.keyrep.KeyRep R K H) (region : table.Region R V) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : K) (index : Usize) : Prop :=
  table.probe.spill_locate keyrep region sigma address key N = ok (table.SpillSite.Found index) →
    ∃ slot : table.SpillSlot R V, region.slots.val[index.val]? = some slot ∧
      table.matches keyrep slot sigma key address = ok true

/-- A successful generated slice index lookup reads exactly the underlying list element. -/
theorem vec_index_readback {T : Type} (slots : alloc.vec.Vec T) (i : Usize) (entry : T)
    (hindex : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice T) slots i = ok entry) :
    slots.val[i.val]? = some entry := by
  rw [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize] at hindex
  have hindex' : (match slots.val[i.val]? with
      | none => (fail .arrayOutOfBounds : Result T)
      | some x => ok x) = ok entry := hindex
  cases hget : slots.val[i.val]? with
  | none => simp [hget] at hindex'
  | some payload =>
      simp only [hget] at hindex'
      exact congrArg Option.some (Result.ok_injective hindex')

/-- The generated `matches` and `Sigma` equality guards compose to the full `table.matches` call. The
generated matcher takes the whole `table.SpillSlot`, so the guard pair is composed on a slot whose
`sigma` and `held` are the guarded ones and whose `tag` and `value` are arbitrary. -/
theorem table_matches_of_guards {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (slotSigma : cell.Sigma) (held : R) (value : V)
    (slotTag : U8)
    (sigma : cell.Sigma) (key : K) (address : table.TableAddress)
    (hsigma : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slotSigma sigma = ok true)
    (hmatch : keyrep.«matches» sigma held key address = ok true) :
    table.matches keyrep { tag := slotTag, sigma := slotSigma, held := held, value := value }
      sigma key address = ok true := by
  unfold table.matches
  simp [hsigma, hmatch]

/-- Flat found packaging for the Stored carrier: a live flat slot whose generated `matches` and `Sigma`
equality guards succeed denotes `((sigma, key), slot.value)` under `decodeStoredOption`. -/
theorem stored_entryAt_flat
    (state : WordTable Apeiron.GrowableIdentity.StoredI64)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress) (index : Usize)
    (slot : table.FlatSlot Apeiron.GrowableIdentity.StoredI64 I64)
    (hslot : state.resident.slots.val[index.val]? = some slot)
    (hlive : slot.live = true)
    (hsigma : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slot.sigma sigma = ok true)
    (hmatch : Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma slot.held key address = ok true) :
    Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption (.flat index) =
      some ((sigma, key), slot.value) := by
  have hmatches : table.matches Apeiron.GrowableIdentity.storedKeyRep
      { tag := 0#u8, sigma := slot.sigma, held := slot.held, value := slot.value }
      sigma key address = ok true :=
    table_matches_of_guards Apeiron.GrowableIdentity.storedKeyRep slot.sigma slot.held slot.value
      0#u8 sigma key address hsigma hmatch
  have hfull : Apeiron.GrowableIdentity.logicalStored slot.sigma slot.held = (sigma, key) :=
    (Apeiron.GrowableIdentity.stored_matches_iff slot.sigma sigma slot.held slot.value key address).mp
      hmatches
  have hsigmaEq : slot.sigma = sigma := by
    have hp : (slot.sigma, Apeiron.GrowableIdentity.decodeStored slot.sigma slot.held) = (sigma, key) :=
      hfull
    exact (Prod.mk.inj hp).1
  have hkeyEq : slot.held.key = key := by
    have hp : (slot.sigma, Apeiron.GrowableIdentity.decodeStored slot.sigma slot.held) = (sigma, key) :=
      hfull
    exact (Prod.mk.inj hp).2
  simp [Apeiron.GrowableSearch.entryAt, hslot, hlive, Apeiron.GrowableSearch.decodeStoredOption,
    hsigmaEq, hkeyEq]

/-- Flat found packaging for the Regen carrier: a live flat slot on a reachable carrier whose generated
`matches` and `Sigma` equality guards succeed denotes `((sigma, key), slot.value)` under
`decodeRegenOption`. -/
theorem regen_entryAt_flat
    (state : WordTable Apeiron.GrowableIdentity.RegenI64)
    (sigma : cell.Sigma) (key original : I64) (address : table.TableAddress) (index : Usize)
    (slot : table.FlatSlot Apeiron.GrowableIdentity.RegenI64 I64)
    (hslot : state.resident.slots.val[index.val]? = some slot)
    (hlive : slot.live = true)
    (hheld : word_regen_hold slot.sigma original = ok slot.held)
    (hsigma : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slot.sigma sigma = ok true)
    (hmatch : Apeiron.GrowableIdentity.regenKeyRep.«matches» sigma slot.held key address = ok true) :
    Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeRegenOption (.flat index) =
      some ((sigma, key), slot.value) := by
  have hmatches : table.matches Apeiron.GrowableIdentity.regenKeyRep
      { tag := 0#u8, sigma := slot.sigma, held := slot.held, value := slot.value }
      sigma key address = ok true :=
    table_matches_of_guards Apeiron.GrowableIdentity.regenKeyRep slot.sigma slot.held slot.value
      0#u8 sigma key address hsigma hmatch
  have hfull : (slot.sigma, original) = (sigma, key) :=
    (Apeiron.GrowableIdentity.regen_slot_matches_iff_of_reachable slot.sigma sigma slot.held
      original key slot.value address hheld).mp hmatches
  have hsigmaEq : slot.sigma = sigma := (Prod.mk.inj hfull).1
  have hkeyEq : original = key := (Prod.mk.inj hfull).2
  have hdecodeOriginal := Apeiron.GrowableIdentity.regen_hold_decodes slot.sigma original
  rw [hheld] at hdecodeOriginal
  simp only [WP.spec_ok] at hdecodeOriginal
  have hdecode : Apeiron.GrowableSearch.decodeRegenOption slot.sigma slot.held = some key := by
    simp [Apeiron.GrowableSearch.decodeRegenOption, hdecodeOriginal, hkeyEq]
  simp only [Apeiron.GrowableSearch.entryAt, hslot, hlive]
  rw [hdecode]
  simp [hsigmaEq]

/-- Logical denotation of any live entry: `RepR.observations.1` matches the
`SearchDecisionCorrect.foundDenotes` contract for any query. -/
theorem foundDenotes {R : Type}
    {state : WordTable R} {decode : cell.Sigma → R → Option I64}
    {direct : LogicalKey → Option Location} {live : LogicalKey → Prop}
    {logical : Apeiron.Table.HMap LogicalKey I64}
    (hrep : Apeiron.GrowableRepresentation.RepR state decode direct live logical)
    (query : LogicalKey) :
    ∀ location value, Apeiron.GrowableSearch.entryAt state decode location = some (query, value) →
      Apeiron.Table.get logical query = value :=
  fun location value hentry => hrep.observations.1 location query value hentry

/-- Local bridge alias matching the `_of_repR` naming convention of bridges 1–3. -/
theorem foundDenotes_of_repR {R : Type}
    {state : WordTable R} {decode : cell.Sigma → R → Option I64}
    {direct : LogicalKey → Option Location} {live : LogicalKey → Prop}
    {logical : Apeiron.Table.HMap LogicalKey I64}
    (hrep : Apeiron.GrowableRepresentation.RepR state decode direct live logical)
    (query : LogicalKey) :
    ∀ location value, Apeiron.GrowableSearch.entryAt state decode location = some (query, value) →
      Apeiron.Table.get logical query = value :=
  foundDenotes hrep query

/-- A `.Found` decision at `.Flat index` produced by `search_decision` on the stored carrier
denotes `((sigma, key), slot.value)` in `entryAt`, which refines canonical `Apeiron.Table.get`. -/
theorem search_decision_found_flat_denotes_stored
    {self : WordTable Apeiron.GrowableIdentity.StoredI64}
    {direct : LogicalKey → Option Location} {live : LogicalKey → Prop}
    {logical : Apeiron.Table.HMap LogicalKey I64}
    (hrep : Apeiron.GrowableRepresentation.RepR self Apeiron.GrowableSearch.decodeStoredOption direct live logical)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress) (decision : table.SearchDecision)
    (hrun : table.Table.search_decision Apeiron.GrowableIdentity.wordHomeInst Apeiron.GrowableIdentity.storedKeyRep
      self sigma key address = ok decision)
    (index : Usize)
    (hclass : decision.class = table.SearchClass.Found)
    (hloc : decision.location = table.SearchLocation.Flat index) :
    ∃ slot : table.FlatSlot Apeiron.GrowableIdentity.StoredI64 I64,
      self.resident.slots.val[index.val]? = some slot ∧
      slot.live = true ∧
      Apeiron.GrowableSearch.entryAt self Apeiron.GrowableSearch.decodeStoredOption (.flat index) =
        some ((sigma, key), slot.value) ∧
      Apeiron.Table.get logical (sigma, key) = slot.value := by
  have hprov := search_decision_provenance Apeiron.GrowableIdentity.wordHomeInst
    Apeiron.GrowableIdentity.storedKeyRep self sigma key address decision hrun
  obtain ⟨slot, hindex, hlive, hsigma, hmatch⟩ :=
    search_decision_found_flat_guards index hprov hclass hloc
  have hslot := vec_index_readback self.resident.slots index slot hindex
  have hentry := stored_entryAt_flat self sigma key address index slot hslot hlive hsigma hmatch
  have hdenotes := foundDenotes_of_repR hrep (sigma, key) (.flat index) slot.value hentry
  exact ⟨slot, hslot, hlive, hentry, hdenotes⟩

/-- A `.Found` decision at `.Flat index` produced by `search_decision` on the regen carrier with
a reachable flat slot denotes `((sigma, key), slot.value)` in `entryAt`, refining canonical
`Apeiron.Table.get`. -/
theorem search_decision_found_flat_denotes_regen
    {self : WordTable Apeiron.GrowableIdentity.RegenI64}
    {direct : LogicalKey → Option Location} {live : LogicalKey → Prop}
    {logical : Apeiron.Table.HMap LogicalKey I64}
    (hrep : Apeiron.GrowableRepresentation.RepR self Apeiron.GrowableSearch.decodeRegenOption direct live logical)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress) (decision : table.SearchDecision)
    (hrun : table.Table.search_decision Apeiron.GrowableIdentity.wordHomeInst Apeiron.GrowableIdentity.regenKeyRep
      self sigma key address = ok decision)
    (index : Usize)
    (hclass : decision.class = table.SearchClass.Found)
    (hloc : decision.location = table.SearchLocation.Flat index)
    (slot : table.FlatSlot Apeiron.GrowableIdentity.RegenI64 I64)
    (hindex : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot Apeiron.GrowableIdentity.RegenI64 I64))
      self.resident.slots index = ok slot)
    (original : I64)
    (hheld : word_regen_hold slot.sigma original = ok slot.held) :
    slot.live = true ∧
    Apeiron.GrowableSearch.entryAt self Apeiron.GrowableSearch.decodeRegenOption (.flat index) =
      some ((sigma, key), slot.value) ∧
    Apeiron.Table.get logical (sigma, key) = slot.value := by
  have hprov := search_decision_provenance Apeiron.GrowableIdentity.wordHomeInst
    Apeiron.GrowableIdentity.regenKeyRep self sigma key address decision hrun
  obtain ⟨slot', hindex', hlive, hsigma, hmatch⟩ :=
    search_decision_found_flat_guards index hprov hclass hloc
  have heq : slot' = slot := Result.ok_injective (hindex'.symm.trans hindex)
  cases heq
  have hslot := vec_index_readback self.resident.slots index slot hindex
  have hentry := regen_entryAt_flat self sigma key original address index slot hslot hlive hheld hsigma hmatch
  have hdenotes := foundDenotes_of_repR hrep (sigma, key) (.flat index) slot.value hentry
  exact ⟨hlive, hentry, hdenotes⟩

/-- Spill found packaging for the Stored carrier: a live spill slot whose generated `matches`
succeeds denotes `((sigma, key), value)` under `decodeStoredOption`. -/
theorem stored_entryAt_spill
    (state : WordTable Apeiron.GrowableIdentity.StoredI64)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress)
    (shard index : Usize) (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (tag : U8) (slotSigma : cell.Sigma) (held : Apeiron.GrowableIdentity.StoredI64) (value : I64)
    (hshard : state.shards.val[shard.val]? = some region)
    (hpayload : region.slots.val[index.val]? =
      some { tag := tag, sigma := slotSigma, held := held, value := value })
    (htagNonzero : tag ≠ 0#u8)
    (hmatch : table.matches Apeiron.GrowableIdentity.storedKeyRep
      { tag := tag, sigma := slotSigma, held := held, value := value } sigma key address = ok true) :
    Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.spill shard index) = some ((sigma, key), value) := by
  have hfull : Apeiron.GrowableIdentity.logicalStored slotSigma held = (sigma, key) :=
    (Apeiron.GrowableIdentity.stored_matches_iff slotSigma sigma held value key address).mp hmatch
  have hsigmaEq : slotSigma = sigma := by
    have hp : (slotSigma, Apeiron.GrowableIdentity.decodeStored slotSigma held) = (sigma, key) := hfull
    exact (Prod.mk.inj hp).1
  have hkeyEq : held.key = key := by
    have hp : (slotSigma, Apeiron.GrowableIdentity.decodeStored slotSigma held) = (sigma, key) := hfull
    exact (Prod.mk.inj hp).2
  simp [Apeiron.GrowableSearch.entryAt, hshard, hpayload, htagNonzero,
    Apeiron.GrowableSearch.decodeStoredOption, hsigmaEq, hkeyEq]

/-- Spill found packaging for the Regen carrier: a live spill slot on a reachable carrier whose
generated `matches` succeeds denotes `((sigma, key), value)` under `decodeRegenOption`. -/
theorem regen_entryAt_spill
    (state : WordTable Apeiron.GrowableIdentity.RegenI64)
    (sigma : cell.Sigma) (key original : I64) (address : table.TableAddress)
    (shard index : Usize) (region : table.Region Apeiron.GrowableIdentity.RegenI64 I64)
    (tag : U8) (slotSigma : cell.Sigma) (held : Apeiron.GrowableIdentity.RegenI64) (value : I64)
    (hshard : state.shards.val[shard.val]? = some region)
    (hpayload : region.slots.val[index.val]? =
      some { tag := tag, sigma := slotSigma, held := held, value := value })
    (htagNonzero : tag ≠ 0#u8)
    (hheld : word_regen_hold slotSigma original = ok held)
    (hmatch : table.matches Apeiron.GrowableIdentity.regenKeyRep
      { tag := tag, sigma := slotSigma, held := held, value := value } sigma key address = ok true) :
    Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeRegenOption
        (.spill shard index) = some ((sigma, key), value) := by
  have hfull : (slotSigma, original) = (sigma, key) :=
    (Apeiron.GrowableIdentity.regen_slot_matches_iff_of_reachable slotSigma sigma held
      original key value address hheld).mp hmatch
  have hsigmaEq : slotSigma = sigma := (Prod.mk.inj hfull).1
  have hkeyEq : original = key := (Prod.mk.inj hfull).2
  have hdecodeOriginal := Apeiron.GrowableIdentity.regen_hold_decodes slotSigma original
  rw [hheld] at hdecodeOriginal
  simp only [WP.spec_ok] at hdecodeOriginal
  have hdecode : Apeiron.GrowableSearch.decodeRegenOption slotSigma held = some key := by
    simp [Apeiron.GrowableSearch.decodeRegenOption, hdecodeOriginal, hkeyEq]
  simp only [Apeiron.GrowableSearch.entryAt, hshard, hpayload, htagNonzero, ite_false]
  rw [hdecode]
  simp [hsigmaEq]

-- Local fail-closed cone check until the integration owner registers these declarations in
-- AxiomAudit.lean. The shared audit file is outside this task's ownership.
open Lean Elab Command in
run_cmd do
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  let external : Array Lean.Name := #[``Apeiron.GrowableRepresentation.RepR,
    ``Apeiron.GrowableSearch.AbsenceCertificate]
  let localNames : Array Lean.Name := #[`flatMatchAdmitted_of_repR,
    `spillMatchRouted_of_repR,
    `DecisionProvenance, `search_decision_provenance, `search_decision_wellFormed,
    `search_decision_found_flat_guards, `SpillLocateFoundFullMatch,
    `vec_index_readback, `table_matches_of_guards, `stored_entryAt_flat, `regen_entryAt_flat,
    `foundDenotes, `foundDenotes_of_repR, `search_decision_found_flat_denotes_stored,
    `search_decision_found_flat_denotes_regen, `stored_entryAt_spill, `regen_entryAt_spill]
  for decl in external ++ localNames.map (Lean.Name.append `Apeiron.GrowableClassifier) do
    let axioms ← liftTermElabM <| Lean.collectAxioms decl
    let bad := axioms.filter (fun ax => !allowed.contains ax)
    unless bad.isEmpty do
      throwError m!"'{decl}' depends on axioms outside the closed cone: {bad.toList}"
    logInfo m!"'{decl}' depends on axioms: {axioms.toList}"

/-- `SpillSite.Grow` and `SpillSite.Found` are distinct constructors, so the probe's
`Grow` terminator never presents as a found hit. -/
private theorem spillSite_grow_ne_found {i : Usize}
    (h : table.SpillSite.Grow = table.SpillSite.Found i) : False := by
  cases h

/-- Generic bind inversion on the generated `do` form: a successful bind ran its head to
`ok`. -/
private theorem bind_ret_inv {α β : Type} (x : Result α) (f : α → Result β) (r : β)
    (hrun : (do let y ← x; f y) = ok r) : ∃ y, x = ok y ∧ f y = ok r := by
  cases hx : x with
  | vis eff k =>
      rw [hx] at hrun
      simp only [bind_tc_vis] at hrun
      exact (vis_not_ok hrun).elim
  | div =>
      rw [hx] at hrun
      simp only [bind_tc_div] at hrun
      exact (div_not_ok hrun).elim
  | ret y =>
      rw [hx] at hrun
      simp only [bind_tc_ok] at hrun
      exact ⟨y, rfl, hrun⟩

/-- One step of the generated `loop`: a successful run either stops at its first step's
`done`, or takes a `cont` step and continues from there. The `loop` unfolds one step because
the generated loop is definitionally `bind (body x) (fun r => match r with
cont x' => loop body x' | done z => ok z)`. -/
private theorem loop_run_cases {α β : Type} (body : α → Result (ControlFlow α β)) (x : α)
    (y : β) (hrun : loop body x = ok y) :
    ∃ r, body x = ok r ∧
      match r with
      | ControlFlow.done z => z = y
      | ControlFlow.cont x' => loop body x' = ok y := by
  unfold loop at hrun
  obtain ⟨r, hr, hfin⟩ := bind_ret_inv (body x) (fun r =>
      match r with
      | ControlFlow.cont x' => loop body x'
      | ControlFlow.done z => ok z) y hrun
  cases r with
  | done z =>
      exact ⟨ControlFlow.done z, hr, Result.ok_injective hfin⟩
  | cont x' =>
      exact ⟨ControlFlow.cont x', hr, hfin⟩

/-- A generated locate body step that reports a `Found` hit at `slot`. The generated body
reads the lane's coordinate from `region.coords`, tests the vacancy marker
`found = table.COORD_EMPTY`, tests the identity guard `found = wanted`, and only then reads
`region.slots` and runs `table.matches`, so the found step carries the coordinate readback,
its occupancy and identity guards and the full-key `matches` success in the same step. The
body never reads `ss.tag`. -/
private theorem spill_locate_body_found {K V H R : Type}
    (keyrep : table.keyrep.KeyRep R K H) (region : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (i home : Usize) (wanted : U64) (st slot : Usize)
    (hbody : table.probe.spill_locate_loop.body keyrep region sigma place direct key i home
      wanted st = ok (ControlFlow.done (table.SpillSite.Found slot))) :
    ∃ coord : U64, ∃ ss : table.SpillSlot R V,
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
        region.coords slot = ok coord ∧
      coord ≠ table.COORD_EMPTY ∧ coord = wanted ∧
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
        region.slots slot = ok ss ∧
      table.matches keyrep ss sigma key { place := place, direct_word := direct } = ok true := by
  unfold table.probe.spill_locate_loop.body at hbody
  by_cases hlt : st < i
  · rw [if_pos hlt] at hbody
    cases hadd : (home + st) with
    | vis eff k =>
        rw [hadd, bind_tc_vis] at hbody
        exact (vis_not_ok hbody).elim
    | div =>
        rw [hadd, bind_tc_div] at hbody
        exact (div_not_ok hbody).elim
    | ret i1 =>
        rw [hadd, bind_tc_ok] at hbody
        cases hsub : (i - 1#usize) with
        | vis eff k =>
            rw [hsub, bind_tc_vis] at hbody
            exact (vis_not_ok hbody).elim
        | div =>
            rw [hsub, bind_tc_div] at hbody
            exact (div_not_ok hbody).elim
        | ret i2 =>
            rw [hsub, bind_tc_ok] at hbody
            have hlift : (lift (i1 &&& i2) : Result Usize) = ok (i1 &&& i2) := rfl
            rw [hlift, bind_tc_ok] at hbody
            cases hidx : alloc.vec.Vec.index
                (core.slice.index.SliceIndexUsizeSlice Std.U64)
                region.coords (i1 &&& i2) with
            | vis eff k =>
                rw [hidx, bind_tc_vis] at hbody
                exact (vis_not_ok hbody).elim
            | div =>
                rw [hidx, bind_tc_div] at hbody
                exact (div_not_ok hbody).elim
            | ret coord =>
                rw [hidx, bind_tc_ok] at hbody
                by_cases hzero : coord = table.COORD_EMPTY
                · rw [if_pos hzero] at hbody
                  exact False.elim (by
                    have h := Result.ok_injective hbody
                    cases h)
                · rw [if_neg hzero] at hbody
                  by_cases hmatch : coord = wanted
                  · rw [if_pos hmatch] at hbody
                    cases hss : alloc.vec.Vec.index
                        (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
                        region.slots (i1 &&& i2) with
                    | vis eff k =>
                        rw [hss, bind_tc_vis] at hbody
                        exact (vis_not_ok hbody).elim
                    | div =>
                        rw [hss, bind_tc_div] at hbody
                        exact (div_not_ok hbody).elim
                    | ret ss =>
                        rw [hss, bind_tc_ok] at hbody
                        cases hmt : table.matches keyrep ss sigma key
                            { place := place, direct_word := direct } with
                        | vis eff k =>
                            rw [hmt, bind_tc_vis] at hbody
                            exact (vis_not_ok hbody).elim
                        | div =>
                            rw [hmt, bind_tc_div] at hbody
                            exact (div_not_ok hbody).elim
                        | ret b =>
                            rw [hmt, bind_tc_ok] at hbody
                            by_cases hb : b = true
                            · rw [if_pos hb] at hbody
                              have h := Result.ok_injective hbody
                              have hslot : slot = i1 &&& i2 := by
                                cases h
                                rfl
                              have hcoord : alloc.vec.Vec.index
                                  (core.slice.index.SliceIndexUsizeSlice Std.U64)
                                  region.coords slot = ok coord := by
                                rw [hslot]
                                exact hidx
                              have hread : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                                  (table.SpillSlot R V)) region.slots slot = ok ss := by
                                rw [hslot]
                                exact hss
                              exact ⟨coord, ss, hcoord, hzero, hmatch, hread,
                                hmt.trans (congrArg ok hb)⟩
                            · rw [if_neg hb] at hbody
                              obtain ⟨s1, hninc, hfin⟩ := bind_ret_inv (st + 1#usize)
                                (fun s1 => ok (ControlFlow.cont s1))
                                (ControlFlow.done (table.SpillSite.Found slot)) hbody
                              exact False.elim (by
                                have h := Result.ok_injective hfin
                                cases h)
                  · rw [if_neg hmatch] at hbody
                    obtain ⟨s1, hninc, hfin⟩ := bind_ret_inv (st + 1#usize)
                      (fun s1 => ok (ControlFlow.cont s1))
                            (ControlFlow.done (table.SpillSite.Found slot)) hbody
                    exact False.elim (by
                      have h := Result.ok_injective hfin
                      cases h)
  · rw [if_neg hlt] at hbody
    exact False.elim (by
      have h := Result.ok_injective hbody
      cases h)

/-- A generated locate body step that reports a `cont` successor `step1`. Every `cont`
branch computes `step1 ← st + 1`, so the successor raises the counter by exactly one, and
the step occurred strictly below the bound. -/
private theorem spill_locate_body_cont {K V H R : Type}
    (keyrep : table.keyrep.KeyRep R K H) (region : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (i home : Usize) (wanted : U64)
    (st step1 : Usize)
    (hbody : table.probe.spill_locate_loop.body keyrep region sigma place direct key i home
      wanted st = ok (ControlFlow.cont step1)) :
    st < i ∧ step1.val = st.val + 1 := by
  have incr (hrun : (do let s1 ← st + 1#usize; ok (ControlFlow.cont s1 : ControlFlow Usize table.SpillSite)) = ok (ControlFlow.cont step1)) :
      step1.val = st.val + 1 := by
    obtain ⟨s1, hninc, hfin⟩ := bind_ret_inv (st + 1#usize)
      (fun s1 => ok (ControlFlow.cont (s1 : Usize) : ControlFlow Usize table.SpillSite))
      (ControlFlow.cont step1) hrun
    have hs1 : s1.val = st.val + 1 := by
      have hadd := UScalar.add_equiv st (1#usize)
      rw [hninc] at hadd
      simpa using hadd.2.1
    have hcontra : (ControlFlow.cont s1) = (ControlFlow.cont step1) :=
      Result.ok_injective hfin
    cases hcontra
    exact hs1
  unfold table.probe.spill_locate_loop.body at hbody
  by_cases hlt : st < i
  · rw [if_pos hlt] at hbody
    cases hadd : (home + st) with
    | vis eff k =>
        rw [hadd, bind_tc_vis] at hbody
        exact (vis_not_ok hbody).elim
    | div =>
        rw [hadd, bind_tc_div] at hbody
        exact (div_not_ok hbody).elim
    | ret i1 =>
        rw [hadd, bind_tc_ok] at hbody
        cases hsub : (i - 1#usize) with
        | vis eff k =>
            rw [hsub, bind_tc_vis] at hbody
            exact (vis_not_ok hbody).elim
        | div =>
            rw [hsub, bind_tc_div] at hbody
            exact (div_not_ok hbody).elim
        | ret i2 =>
            rw [hsub, bind_tc_ok] at hbody
            have hlift : (lift (i1 &&& i2) : Result Usize) = ok (i1 &&& i2) := rfl
            rw [hlift, bind_tc_ok] at hbody
            cases hidx : alloc.vec.Vec.index
                (core.slice.index.SliceIndexUsizeSlice Std.U64)
                region.coords (i1 &&& i2) with
            | vis eff k =>
                rw [hidx, bind_tc_vis] at hbody
                exact (vis_not_ok hbody).elim
            | div =>
                rw [hidx, bind_tc_div] at hbody
                exact (div_not_ok hbody).elim
            | ret coord =>
                rw [hidx, bind_tc_ok] at hbody
                by_cases hzero : coord = table.COORD_EMPTY
                · rw [if_pos hzero] at hbody
                  exact False.elim (by
                    have h := Result.ok_injective hbody
                    cases h)
                · rw [if_neg hzero] at hbody
                  by_cases hmatch : coord = wanted
                  · rw [if_pos hmatch] at hbody
                    cases hss : alloc.vec.Vec.index
                        (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
                        region.slots (i1 &&& i2) with
                    | vis eff k =>
                        rw [hss, bind_tc_vis] at hbody
                        exact (vis_not_ok hbody).elim
                    | div =>
                        rw [hss, bind_tc_div] at hbody
                        exact (div_not_ok hbody).elim
                    | ret ss =>
                        rw [hss, bind_tc_ok] at hbody
                        cases hmt : table.matches keyrep ss sigma key
                            { place := place, direct_word := direct } with
                        | vis eff k =>
                            rw [hmt, bind_tc_vis] at hbody
                            exact (vis_not_ok hbody).elim
                        | div =>
                            rw [hmt, bind_tc_div] at hbody
                            exact (div_not_ok hbody).elim
                        | ret b =>
                            rw [hmt, bind_tc_ok] at hbody
                            by_cases hb : b = true
                            · rw [if_pos hb] at hbody
                              exact False.elim (by
                                have h := Result.ok_injective hbody
                                cases h)
                            · rw [if_neg hb] at hbody
                              exact ⟨hlt, incr hbody⟩
                  · rw [if_neg hmatch] at hbody
                    exact ⟨hlt, incr hbody⟩
  · rw [if_neg hlt] at hbody
    exact False.elim (by
      have h := Result.ok_injective hbody
      cases h)

/-- The locate loop's found hit carries a full-key match certificate at the returned lane.
Each step is read with `spill_locate_body_step`; the `cont` successor raises the step counter by
one below the bound, so the loop strictly decreases the remaining step budget `i.val - st.val`
and induction on that budget reaches the terminating step. -/
private theorem loop_hit_of_loop_run {K V H R : Type}
    (keyrep : table.keyrep.KeyRep R K H) (region : table.Region R V) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : K) (i home : Usize) (wanted : U64) (index : Usize)
    (hrun : table.probe.spill_locate_loop keyrep region sigma address.place address.direct_word
      key i home wanted 0#usize = ok (table.SpillSite.Found index)) :
    ∃ slot : table.SpillSlot R V, region.slots.val[index.val]? = some slot ∧
      table.matches keyrep slot sigma key address = ok true := by
  have step0 : ∀ (st : Usize),
      (table.probe.spill_locate_loop keyrep region sigma address.place address.direct_word
          key i home wanted st
        = ok (table.SpillSite.Found index)) →
      (∃ slot : table.SpillSlot R V, region.slots.val[index.val]? = some slot ∧
          table.matches keyrep slot sigma key address = ok true) ∨
        (∃ st1 : Usize, st1.val ≤ i.val ∧ i.val - st1.val < i.val - st.val ∧
            table.probe.spill_locate_loop keyrep region sigma address.place
              address.direct_word key i home wanted st1
            = ok (table.SpillSite.Found index)) := by
    intro st hloop
    obtain ⟨r, hbody, hred⟩ := loop_run_cases
      (fun st1 => table.probe.spill_locate_loop.body keyrep region sigma address.place
        address.direct_word key i home wanted st1) st (table.SpillSite.Found index) hloop
    cases r with
    | cont step1 =>
        obtain ⟨hlt, hs1⟩ := spill_locate_body_cont keyrep region sigma address.place
          address.direct_word key i home wanted st step1 hbody
        have hltv : st.val < i.val := hlt
        exact Or.inr ⟨step1, by omega, by rw [hs1]; omega, hred⟩
    | done site =>
        cases site with
        | Grow =>
            exact False.elim (spillSite_grow_ne_found hred)
        | Vacant slot =>
            exact False.elim (by cases hred)
        | Found slot =>
            have hslot : slot = index := by injection hred
            obtain ⟨coord, ss, hcoord, hlive, htag, hread, hmatch⟩ :=
              spill_locate_body_found keyrep region
              sigma address.place address.direct_word key i home wanted st slot hbody
            rw [← hslot]
            exact Or.inl ⟨ss, vec_index_readback region.slots slot ss hread, by
              cases address
              exact hmatch⟩
  have aux : ∀ n : Nat, ∀ st : Usize, i.val - st.val = n →
      (table.probe.spill_locate_loop keyrep region sigma address.place address.direct_word
          key i home wanted st
        = ok (table.SpillSite.Found index)) →
      ∃ slot : table.SpillSlot R V, region.slots.val[index.val]? = some slot ∧
        table.matches keyrep slot sigma key address = ok true := by
    intro n
    induction n using Nat.strong_induction_on with
    | _ n ih =>
        intro st hn hloop
        cases step0 st hloop with
        | inl hfound =>
            obtain ⟨w, hread, hmatch⟩ := hfound
            exact ⟨w, hread, hmatch⟩
        | inr hdown =>
            obtain ⟨st1, hst1, hdec, hloop1⟩ := hdown
            cases n with
            | zero =>
                rw [hn] at hdec
                exact False.elim ((Nat.not_lt_zero _) hdec)
            | succ m =>
                exact ih (i.val - st1.val) (by simpa [hn] using hdec) st1 rfl hloop1
  have hzero : (0#usize).val = 0 := rfl
  exact aux (i.val - 0#usize) 0#usize (by rw [hzero]) hrun

/-- L6: a generated `spill_locate` `Found` hit carries a full-key match certificate at
the returned lane. The readback and the real matcher's `ok true` come from the executed match
step: the locate body tests the vacancy marker on the lane coordinate and the identity guard
`found = wanted` before reading the slot, so the step that returns `Found` is the step that
read the slot and ran `table.matches` on it. `loop_run_cases`
reads the generated `loop` one step at a time and `spill_locate_body_found` /
`spill_locate_body_cont` read the step's own guards, so the proof reads the executed body
rather than a summary of it. `Growable/LocateLoop.lean` remains red and is not imported here. -/
theorem spill_locate_found_full_match {K V H R : Type} {N : Usize}
    (keyrep : table.keyrep.KeyRep R K H) (region : table.Region R V) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : K) (index : Usize) :
    SpillLocateFoundFullMatch (N := N) keyrep region sigma address key index := by
  intro hloc
  unfold table.probe.spill_locate at hloc
  cases hnb : table.valid_n N with
  | vis eff k =>
      rw [hnb, bind_tc_vis] at hloc
      exact (vis_not_ok hloc).elim
  | div =>
      rw [hnb, bind_tc_div] at hloc
      exact (div_not_ok hloc).elim
  | ret b =>
      rw [hnb, bind_tc_ok] at hloc
      by_cases hb : b = true
      · rw [if_pos hb] at hloc
        cases hbd : table.Region.bound region with
        | vis eff k =>
            rw [hbd, bind_tc_vis] at hloc
            exact (vis_not_ok hloc).elim
        | div =>
            rw [hbd, bind_tc_div] at hloc
            exact (div_not_ok hloc).elim
        | ret i =>
            rw [hbd, bind_tc_ok] at hloc
            by_cases hz : i = 0#usize
            · rw [if_pos hz] at hloc
              exact False.elim (spillSite_grow_ne_found (Result.ok_injective hloc))
            · rw [if_neg hz] at hloc
              cases hh : table.Region.home region address.place with
              | vis eff k =>
                  rw [hh, bind_tc_vis] at hloc
                  exact (vis_not_ok hloc).elim
              | div =>
                  rw [hh, bind_tc_div] at hloc
                  exact (div_not_ok hloc).elim
              | ret home =>
                  rw [hh, bind_tc_ok] at hloc
                  cases hw : table.Region.coord_of R V address.place with
                  | vis eff k =>
                      rw [hw, bind_tc_vis] at hloc
                      exact (vis_not_ok hloc).elim
                  | div =>
                      rw [hw, bind_tc_div] at hloc
                      exact (div_not_ok hloc).elim
                  | ret wanted =>
                      rw [hw, bind_tc_ok] at hloc
                      exact loop_hit_of_loop_run keyrep region sigma address key i home wanted
                        index hloc
      · rw [if_neg hb] at hloc
        exact False.elim (spillSite_grow_ne_found (Result.ok_injective hloc))
-- The cone check above runs before `spill_locate_found_full_match`, so the headline and the
-- loop-reading helpers it depends on are audited here. `AxiomAudit/Core.lean` registers the
-- headline's `#audit`, but that module imports `Growable.ReadLoop`, so this local check is the
-- one that executes while `Growable/ReadLoop.lean` is red.
open Lean Elab Command in
run_cmd do
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  let headline : Lean.Name := `Apeiron.GrowableClassifier.spill_locate_found_full_match
  let axioms ← liftTermElabM <| Lean.collectAxioms headline
  let bad := axioms.filter (fun ax => !allowed.contains ax)
  unless bad.isEmpty do
    throwError m!"'{headline}' depends on axioms outside the closed cone: {bad.toList}"
  logInfo m!"'{headline}' depends on axioms: {axioms.toList}"

end Apeiron.GrowableClassifier
