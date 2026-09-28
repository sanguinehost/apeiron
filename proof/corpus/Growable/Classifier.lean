import Growable.Representation
import Growable.LocateLoop
import Lean

/-!
# Row 7 producer bridges: `RepR` certificates to `Growable.Search` interfaces

Row 7 of the read map (`docs/epics/read-proof.md`) splits into the `Growable.Search`
interface (landed) and this producer module.  Each bridge below derives a `Growable.Search`
certificate from `Apeiron.GrowableRepresentation.RepR` fields and the generated route facts.  No
premise is a caller-supplied `Found`/`Absent` result and no desired conclusion is assumed:

1. `summaryMissSound_of_repR` bridges `GrowableRepresentation.SpillSummaryMissSound` (from
   `repR_spillSummaryMissSound`) plus the generated address/shard/region/summary route into the
   `Search`-local `Apeiron.GrowableSearch.SummaryMissSound`;
2. `flatMatchAdmitted_of_repR` sources flat direct-admission from `RepR.address` (flat case via
   `GrowableGeometry.preparedFlatIndex`) and `RepR.directIndex`;
3. `spillMatchRouted_of_repR` sources the routed shard from `RepR.address` (spill case);
4. `repR_absenceCertificate_of_summaryMiss` assembles the `directExcludedSummaryMiss` certificate,
   and `repR_absent_of_summaryMiss` derives logical absence from that certificate.

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
1. A generated routed summary miss plus the representation's summary shape/recordedness yields the
`Search`-local `SummaryMissSound`. The exclusion is `repR_spillSummaryMissSound`, not an assumption.
-/
def summaryMissSound_of_repR {R : Type}
    {state : WordTable R} {decode : cell.Sigma → R → Option I64}
    {direct : LogicalKey → Option Location} {live : LogicalKey → Prop}
    {logical : Apeiron.Table.HMap LogicalKey I64}
    (hrep : Apeiron.GrowableRepresentation.RepR state decode direct live logical)
    (query : LogicalKey) (address : table.TableAddress) (shard : Usize)
    (region : table.Region R I64)
    (haddress :
      Apeiron.GrowableIdentity.wordHomeInst.address state.home query.1 query.2 = ok address)
    (hshard : table.shard address = ok shard)
    (hregion : state.shards.val[shard.val]? = some region)
    (hmiss : table.Region.summary_maybe region address.place = ok false) :
    Apeiron.GrowableSearch.SummaryMissSound state decode query :=
  { address := address
    shard := shard
    generatedAddress := haddress
    shardEq := hshard
    region := region
    regionAt := hregion
    miss := hmiss
    excludes := (Apeiron.GrowableRepresentation.repR_spillSummaryMissSound hrep query address shard
        region haddress hshard hregion hmiss).excludes }

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

/--
4. The `directExcludedSummaryMiss` certificate for any decision and checked set: a
direct-excluded query plus a routed summary miss (bridge 1).
-/
theorem repR_absenceCertificate_of_summaryMiss {R : Type}
    {state : WordTable R} {decode : cell.Sigma → R → Option I64}
    {direct : LogicalKey → Option Location} {live : LogicalKey → Prop}
    {logical : Apeiron.Table.HMap LogicalKey I64}
    (hrep : Apeiron.GrowableRepresentation.RepR state decode direct live logical)
    (query : LogicalKey) (decision : table.SearchDecision) (checked : List Location)
    (address : table.TableAddress) (shard : Usize) (region : table.Region R I64)
    (haddress :
      Apeiron.GrowableIdentity.wordHomeInst.address state.home query.1 query.2 = ok address)
    (hshard : table.shard address = ok shard)
    (hregion : state.shards.val[shard.val]? = some region)
    (hmiss : table.Region.summary_maybe region address.place = ok false)
    (hexcluded : direct query = none) :
    Apeiron.GrowableSearch.AbsenceCertificate state decode logical query decision direct checked :=
  .directExcludedSummaryMiss hexcluded
    (summaryMissSound_of_repR hrep query address shard region haddress hshard hregion hmiss)

/--
Logical absence from a sound routed summary miss on a direct-excluded query. The flat side is
excluded by bridge 2 against `direct query = none`; the spill side by bridge 3 aligning every
representative with the routed shard and the exclusion of bridge 1. No absence is assumed.
-/
theorem repR_absent_of_summaryMiss {R : Type}
    {state : WordTable R} {decode : cell.Sigma → R → Option I64}
    {direct : LogicalKey → Option Location} {live : LogicalKey → Prop}
    {logical : Apeiron.Table.HMap LogicalKey I64}
    (hrep : Apeiron.GrowableRepresentation.RepR state decode direct live logical)
    (query : LogicalKey)
    (address : table.TableAddress) (shard : Usize) (region : table.Region R I64)
    (haddress :
      Apeiron.GrowableIdentity.wordHomeInst.address state.home query.1 query.2 = ok address)
    (hshard : table.shard address = ok shard)
    (hregion : state.shards.val[shard.val]? = some region)
    (hmiss : table.Region.summary_maybe region address.place = ok false)
    (hexcluded : direct query = none) :
    Apeiron.GrowableSearch.Absent state decode query := by
  have hmissSound := summaryMissSound_of_repR hrep query address shard region haddress hshard
    hregion hmiss
  intro location value hentry
  cases location with
  | flat index =>
      obtain ⟨location, hdirect⟩ := flatMatchAdmitted_of_repR hrep query index value hentry
      rw [hexcluded] at hdirect
      cases hdirect
  | spill spillShard index =>
      obtain ⟨address', haddress', hshard'⟩ :=
        spillMatchRouted_of_repR hrep query spillShard index value hentry
      rw [hmissSound.generatedAddress] at haddress'
      have haddrEq : address' = hmissSound.address := (Result.ok_injective haddress').symm
      rw [haddrEq] at hshard'
      rw [hmissSound.shardEq] at hshard'
      have hshardEq : spillShard = hmissSound.shard := (Result.ok_injective hshard').symm
      rw [hshardEq] at hentry
      exact hmissSound.excludes index value hentry

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
    ∃ payload : cell.Sigma × R × V, region.slots.val[index.val]? = some payload ∧
      table.matches keyrep payload sigma key address = ok true

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

/-- The generated `matches` and `Sigma` equality guards compose to the full `table.matches` call. -/
theorem table_matches_of_guards {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (slotSigma : cell.Sigma) (held : R) (value : V)
    (sigma : cell.Sigma) (key : K) (address : table.TableAddress)
    (hsigma : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slotSigma sigma = ok true)
    (hmatch : keyrep.«matches» sigma held key address = ok true) :
    table.matches keyrep (slotSigma, held, value) sigma key address = ok true := by
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
      (slot.sigma, slot.held, slot.value) sigma key address = ok true :=
    table_matches_of_guards Apeiron.GrowableIdentity.storedKeyRep slot.sigma slot.held slot.value
      sigma key address hsigma hmatch
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
      (slot.sigma, slot.held, slot.value) sigma key address = ok true :=
    table_matches_of_guards Apeiron.GrowableIdentity.regenKeyRep slot.sigma slot.held slot.value
      sigma key address hsigma hmatch
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
    (htag : region.tags.val[index.val]? = some tag)
    (htagNonzero : tag ≠ 0#u8)
    (hpayload : region.slots.val[index.val]? = some (slotSigma, held, value))
    (hmatch : table.matches Apeiron.GrowableIdentity.storedKeyRep (slotSigma, held, value)
      sigma key address = ok true) :
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
  simp [Apeiron.GrowableSearch.entryAt, hshard, htag, hpayload, htagNonzero,
    Apeiron.GrowableSearch.decodeStoredOption, hsigmaEq, hkeyEq]

/-- Spill found packaging for the Regen carrier: a live spill slot on a reachable carrier whose
generated `matches` succeeds denotes `((sigma, key), value)` under `decodeRegenOption`. -/
theorem regen_entryAt_spill
    (state : WordTable Apeiron.GrowableIdentity.RegenI64)
    (sigma : cell.Sigma) (key original : I64) (address : table.TableAddress)
    (shard index : Usize) (region : table.Region Apeiron.GrowableIdentity.RegenI64 I64)
    (tag : U8) (slotSigma : cell.Sigma) (held : Apeiron.GrowableIdentity.RegenI64) (value : I64)
    (hshard : state.shards.val[shard.val]? = some region)
    (htag : region.tags.val[index.val]? = some tag)
    (htagNonzero : tag ≠ 0#u8)
    (hpayload : region.slots.val[index.val]? = some (slotSigma, held, value))
    (hheld : word_regen_hold slotSigma original = ok held)
    (hmatch : table.matches Apeiron.GrowableIdentity.regenKeyRep (slotSigma, held, value)
      sigma key address = ok true) :
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
  simp only [Apeiron.GrowableSearch.entryAt, hshard, htag, hpayload, htagNonzero, ite_false]
  rw [hdecode]
  simp [hsigmaEq]

-- Local fail-closed cone check until the integration owner registers these declarations in
-- AxiomAudit.lean. The shared audit file is outside this task's ownership.
open Lean Elab Command in
run_cmd do
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  let external : Array Lean.Name := #[``Apeiron.GrowableRepresentation.RepR,
    ``Apeiron.GrowableRepresentation.repR_spillSummaryMissSound,
    ``Apeiron.GrowableSearch.SummaryMissSound, ``Apeiron.GrowableSearch.AbsenceCertificate]
  let localNames : Array Lean.Name := #[`summaryMissSound_of_repR, `flatMatchAdmitted_of_repR,
    `spillMatchRouted_of_repR, `repR_absenceCertificate_of_summaryMiss, `repR_absent_of_summaryMiss,
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

/-- L6: a generated `spill_locate` `Found` hit carries a full-key match certificate at
the returned lane. The payload readback and the real matcher's `ok true` come from the
executed match step, extracted through `Apeiron.GrowableLocateLoop`. -/
theorem spill_locate_found_full_match {K V H R : Type} {N : Usize}
    (keyrep : table.keyrep.KeyRep R K H) (region : table.Region R V) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : K) (index : Usize) :
    SpillLocateFoundFullMatch (N := N) keyrep region sigma address key index := by
  intro hloc
  obtain ⟨wanted, may_match, groups, mask, b, i1, _htag, _hsum, hloop⟩ :=
    Apeiron.GrowableLocateLoop.generated_spill_locate_unfold_run keyrep region sigma
      address key N index hloc
  have hcert := Apeiron.GrowableLocateLoop.generated_locate_outer_found keyrep
    region.tags region.slots sigma address.place address.direct_word key N groups mask
    wanted may_match b i1 index hloop
  rcases hcert with ⟨entry, hidx, hmatch⟩
  exact ⟨entry, vec_index_readback region.slots index entry hidx, hmatch⟩

end Apeiron.GrowableClassifier
