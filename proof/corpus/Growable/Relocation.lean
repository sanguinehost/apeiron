import ApeironWord
import Apeiron.Table.Basic
import Growable.Identity

/-!
# Per-entry relocation refinement

The extracted roots call the production `Table::relocate_entry` once.  This module proves that one
call copies one selected source observation to its planned destination for both concrete carriers.
The source table is unchanged, and the operation has no merge argument.  The unextracted traversal,
pending insertion, and commit are not represented as functions here, so no theorem in this module
claims the final `Apeiron.Table.insertWith` state.
-/

namespace Apeiron.GrowableRelocation

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

abbrev LogicalKey := cell.Sigma × I64
abbrev LogicalEntry := LogicalKey × I64
abbrev StoredI64 := table.keyrep.Stored I64
abbrev RegenI64 := table.keyrep.Regen table.keyrep.I64WordCodec

/-- A source selected by a rebuild-plan entry, decoded as one canonical key/value observation. -/
def sourceObservation {R : Type}
    (state : table.Table I64 I64 table.WordHome R 16#usize)
    (decode : cell.Sigma → R → Option I64) : table.RebuildSource → Option LogicalEntry
  | .Resident index =>
      match state.resident.slots.val[index.val]? with
      | some slot =>
          if slot.live then
            (decode slot.sigma slot.held).map fun key => ((slot.sigma, key), slot.value)
          else none
      | none => none
  | .Spill shard index =>
      match state.shards.val[shard.val]? with
      | some region =>
          match region.tags.val[index.val]?, region.slots.val[index.val]? with
          | some tag, some (sigma, held, value) =>
              if tag = 0#u8 then none
              else (decode sigma held).map fun key => ((sigma, key), value)
          | _, _ => none
      | none => none

/-- The selected source entry represents its value in the canonical total map. -/
def RepR {R : Type} (state : table.Table I64 I64 table.WordHome R 16#usize)
    (decode : cell.Sigma → R → Option I64) (source : table.RebuildSource)
    (logical : Apeiron.Table.HMap LogicalKey I64) : Prop :=
  ∃ entry, sourceObservation state decode source = some entry ∧ logical entry.1 = entry.2

/-- A destination occupied by one relocated canonical observation. -/
inductive Destination where
  | flat (index : Usize)
  | spill (shard index : Usize)

/-- Read one materialized relocation destination through its actual occupancy authority. -/
def reservedObservation {R : Type} (reserved : table.RebuildReserved R I64)
    (decode : cell.Sigma → R → Option I64) : Destination → Option LogicalEntry
  | .flat index =>
      match reserved.resident.slots.val[index.val]? with
      | some slot =>
          if slot.live then
            (decode slot.sigma slot.held).map fun key => ((slot.sigma, key), slot.value)
          else none
      | none => none
  | .spill shard index =>
      match reserved.regions.val[shard.val]? with
      | some region =>
          match region.tags.val[index.val]?, region.slots.val[index.val]? with
          | some tag, some (sigma, held, value) =>
              if tag = 0#u8 then none
              else (decode sigma held).map fun key => ((sigma, key), value)
          | _, _ => none
      | none => none

/-- The selected reserved destination represents its value in the same canonical total map. -/
def ReservedRepR {R : Type} (reserved : table.RebuildReserved R I64)
    (decode : cell.Sigma → R → Option I64) (destination : Destination)
    (logical : Apeiron.Table.HMap LogicalKey I64) : Prop :=
  ∃ entry, reservedObservation reserved decode destination = some entry ∧
    logical entry.1 = entry.2

/-- Relocation may retarget the locator but preserves every payload-bearing table field. -/
def PayloadStateEq {R : Type}
    (before after : table.Table I64 I64 table.WordHome R 16#usize) : Prop :=
  after.default = before.default ∧ after.home = before.home ∧
    after.resident = before.resident ∧ after.shards = before.shards ∧
    after.len = before.len ∧ after.key = before.key

/-- Read the physical payload selected by a relocation plan entry through its occupancy authority. -/
def sourcePayload {R : Type}
    (state : table.Table I64 I64 table.WordHome R 16#usize) :
    table.RebuildSource → Option (cell.Sigma × R × I64)
  | .Resident index =>
      match state.resident.slots.val[index.val]? with
      | some slot => if slot.live then some (slot.sigma, slot.held, slot.value) else none
      | none => none
  | .Spill shard index =>
      match state.shards.val[shard.val]? with
      | some region =>
          match region.tags.val[index.val]?, region.slots.val[index.val]? with
          | some tag, some payload => if tag = 0#u8 then none else some payload
          | _, _ => none
      | none => none

/-- The materialized input certificate needed by one extracted relocation call. -/
structure PreparedEntry {R : Type}
    (state : table.Table I64 I64 table.WordHome R 16#usize)
    (decode : cell.Sigma → R → Option I64) (plan : table.RebuildPlan)
    (reserved : table.RebuildReserved R I64) (entryAt : Usize)
    (logical : Apeiron.Table.HMap LogicalKey I64) where
  entry : table.RebuildPlanEntry
  entry_eq : plan.entries.val[entryAt.val]? = some entry
  source_rep : RepR state decode entry.source logical

/-- A planned flat write has a materialized in-bounds destination. -/
def FlatPrepared {R : Type} (entry : table.RebuildPlanEntry)
    (reserved : table.RebuildReserved R I64) (index : Usize) : Prop :=
  entry.flat = some index ∧ index.val < reserved.resident.slots.length

/--
The input-side certificate for the generated spill branch.  It identifies the selected source
payload, reconstructed address, vacant destination, and initialized destination bounds.
-/
structure SpillPrepared {R : Type}
    (state : table.Table I64 I64 table.WordHome R 16#usize)
    (entry : table.RebuildPlanEntry) (reserved : table.RebuildReserved R I64)
    (cloneHeld : R → R)
    (addressHeld : cell.Sigma → R → Result table.TableAddress)
    (locateHeld : table.Region R I64 → cell.Sigma → table.TableAddress → R →
      Result table.SpillSite)
    (shard index : Usize) where
  sigma : cell.Sigma
  held : R
  value : I64
  address : table.TableAddress
  region : table.Region R I64
  source_eq : sourcePayload state entry.source = some (sigma, held, value)
  flat_eq : entry.flat = none
  spill_eq : entry.spill = shard
  address_eq : addressHeld sigma (cloneHeld held) = ok address
  region_eq : reserved.regions.val[shard.val]? = some region
  vacant_eq : locateHeld region sigma address (cloneHeld held) = ok (.Vacant index)
  tag_bound : index.val < region.tags.length
  slot_bound : index.val < region.slots.length

/-- Convert one rebuild source to the locator's runtime payload location. -/
def sourceLocation : table.RebuildSource → table.PayloadLocation
  | .Resident index => .Resident index
  | .Spill shard index => .Spill shard index

/-- Convert one certified destination to the locator's runtime payload location. -/
def destinationLocation : Destination → table.PayloadLocation
  | .flat index => .Resident index
  | .spill shard index => .Spill shard index

/--
Fallible side effects required by one relocation.  The premise states successful locator retargeting;
for a spill it additionally states successful publication and the resulting occupancy readback.
-/
structure RelocationEffectsPrepared {R : Type}
    (keyRep : table.keyrep.KeyRep R I64 table.WordHome)
    (state : table.Table I64 I64 table.WordHome R 16#usize)
    (entry : table.RebuildPlanEntry) (cloneHeld : R → R) (destination : Destination) : Prop where
  retarget : ∀ sigma held value,
    sourcePayload state entry.source = some (sigma, held, value) →
    ∃ nextLocator,
      table.TableLocator.retarget_held core.clone.CloneI64
        Apeiron.GrowableIdentity.wordHomeInst keyRep state.locator state.home sigma
        (cloneHeld held) (sourceLocation entry.source) (destinationLocation destination) =
          ok nextLocator
  publish : match destination with
    | .flat _ => True
    | .spill _ index => ∀ (sigma : cell.Sigma) (held : R) (value : I64)
        (address : table.TableAddress) (region : table.Region R I64),
        ∃ published,
          table.Region.publish_live region index (sigma, cloneHeld held, value) address.place =
              ok published ∧
            published.slots.val[index.val]? = some (sigma, cloneHeld held, value) ∧
            ∃ tag, published.tags.val[index.val]? = some tag ∧ tag ≠ 0#u8

/-- The two generated outer-vector updates retain the payload write while adding the tag write. -/
def writeSpill {R : Type} (reserved : table.RebuildReserved R I64) (shard index : Usize)
    (region : table.Region R I64) (tag : U8) (payload : cell.Sigma × R × I64) :
    table.RebuildReserved R I64 :=
  let slots := region.slots.set index payload
  let regions := reserved.regions.set shard { region with slots := slots }
  let updated := { region with tags := region.tags.set index tag, slots := slots }
  { reserved with regions := regions.set shard updated }

/-- The generated flat branch replaces exactly one initialized resident slot. -/
def writeFlat {R : Type} (reserved : table.RebuildReserved R I64) (index : Usize)
    (slot : table.FlatSlot R I64) : table.RebuildReserved R I64 :=
  { reserved with resident :=
      { reserved.resident with slots := reserved.resident.slots.set index slot } }

private theorem getElem?_set_eq_of_ne {α : Type} (xs : List α) (written read : Nat)
    (value : α) (hne : read ≠ written) : (xs.set written value)[read]? = xs[read]? := by
  simp [hne]

/-- Read back the inner payload update through the outer generated region-vector update. -/
theorem nested_region_payload_set_readback {R : Type}
    (reserved : table.RebuildReserved R I64) (shard index : Usize)
    (region : table.Region R I64) (payload : cell.Sigma × R × I64)
    (hregion : reserved.regions.val[shard.val]? = some region) :
    (reserved.regions.set shard { region with slots := region.slots.set index payload }).val[shard.val]? =
      some { region with slots := region.slots.set index payload } := by
  have hshardBound : shard.val < reserved.regions.length := by
    simpa using (List.getElem?_eq_some_iff.mp hregion).1
  simp [hshardBound]

/-- Readback for the generated payload-then-tag nested `Vec::set` update. -/
theorem nested_region_set_readback {R : Type} (reserved : table.RebuildReserved R I64)
    (decode : cell.Sigma → R → Option I64) (shard index : Usize)
    (region : table.Region R I64) (sigma : cell.Sigma) (held : R) (value : I64) (tag : U8)
    (hregion : reserved.regions.val[shard.val]? = some region)
    (htagBound : index.val < region.tags.length)
    (hslotBound : index.val < region.slots.length) (htag : tag.val ≠ 0) :
    reservedObservation (writeSpill reserved shard index region tag (sigma, held, value)) decode
        (.spill shard index) =
      (decode sigma held).map fun key => ((sigma, key), value) := by
  have hshardBound : shard.val < reserved.regions.length := by
    simpa using (List.getElem?_eq_some_iff.mp hregion).1
  have htag' : tag ≠ 0#u8 := by
    intro hzero
    exact htag (congrArg UScalar.val hzero)
  simp [reservedObservation, writeSpill, hshardBound, htagBound, hslotBound, htag']

/-!
## Disjoint-write frames

These lemmas state the concrete frame rule used by a relocation prefix: a generated write changes
only its named destination.  In particular, a later write cannot invalidate an observation already
established at a pairwise-disjoint earlier destination.
-/

/-- A flat write retains every distinct earlier flat observation. -/
theorem writeFlat_frame_flat {R : Type} (reserved : table.RebuildReserved R I64)
    (decode : cell.Sigma → R → Option I64) (written earlier : Usize)
    (slot : table.FlatSlot R I64) (hne : earlier.val ≠ written.val) :
    reservedObservation (writeFlat reserved written slot) decode (.flat earlier) =
      reservedObservation reserved decode (.flat earlier) := by
  simp only [reservedObservation, writeFlat, alloc.vec.Vec.set_val_eq]
  rw [getElem?_set_eq_of_ne _ _ _ _ hne]

/-- A flat write retains every spill observation. -/
theorem writeFlat_frame_spill {R : Type} (reserved : table.RebuildReserved R I64)
    (decode : cell.Sigma → R → Option I64) (written shard index : Usize)
    (slot : table.FlatSlot R I64) :
    reservedObservation (writeFlat reserved written slot) decode (.spill shard index) =
      reservedObservation reserved decode (.spill shard index) := by
  rfl

/-- A spill write retains every flat observation. -/
theorem writeSpill_frame_flat {R : Type} (reserved : table.RebuildReserved R I64)
    (decode : cell.Sigma → R → Option I64) (shard written earlier : Usize)
    (region : table.Region R I64) (tag : U8) (payload : cell.Sigma × R × I64) :
    reservedObservation (writeSpill reserved shard written region tag payload) decode
        (.flat earlier) = reservedObservation reserved decode (.flat earlier) := by
  rfl

/-- A nested spill payload/tag write retains an observation in every other spill shard. -/
theorem writeSpill_frame_other_shard {R : Type}
    (reserved : table.RebuildReserved R I64) (decode : cell.Sigma → R → Option I64)
    (writtenShard writtenIndex earlierShard earlierIndex : Usize) (region : table.Region R I64)
    (tag : U8) (payload : cell.Sigma × R × I64)
    (hne : earlierShard.val ≠ writtenShard.val) :
    reservedObservation
        (writeSpill reserved writtenShard writtenIndex region tag payload) decode
        (.spill earlierShard earlierIndex) =
      reservedObservation reserved decode (.spill earlierShard earlierIndex) := by
  simp only [reservedObservation, writeSpill, alloc.vec.Vec.set_val_eq]
  rw [getElem?_set_eq_of_ne _ _ _ _ hne, getElem?_set_eq_of_ne _ _ _ _ hne]

/-- A nested spill payload/tag write retains every distinct index in the written shard. -/
theorem writeSpill_frame_same_shard {R : Type}
    (reserved : table.RebuildReserved R I64) (decode : cell.Sigma → R → Option I64)
    (shard written earlier : Usize) (region : table.Region R I64) (tag : U8)
    (payload : cell.Sigma × R × I64)
    (hregion : reserved.regions.val[shard.val]? = some region)
    (hne : earlier.val ≠ written.val) :
    reservedObservation (writeSpill reserved shard written region tag payload) decode
        (.spill shard earlier) =
      reservedObservation reserved decode (.spill shard earlier) := by
  have hshardBound : shard.val < reserved.regions.length := by
    simpa using (List.getElem?_eq_some_iff.mp hregion).1
  have hshardBound' : shard.val <
      (reserved.regions.val.set shard.val
        { region with slots := region.slots.set written payload }).length := by
    simpa using hshardBound
  simp only [reservedObservation, writeSpill, alloc.vec.Vec.set_val_eq]
  rw [List.getElem?_set_self hshardBound']
  simp only [alloc.vec.Vec.set_val_eq]
  rw [getElem?_set_eq_of_ne _ _ _ _ hne, getElem?_set_eq_of_ne _ _ _ _ hne]
  rw [hregion]

/-- Stored carriers decode without failure. -/
def decodeStored (_sigma : cell.Sigma) (held : StoredI64) : Option I64 := some held.key

/-- Regenerating carriers decode through the extracted exact codec. -/
def decodeRegen (sigma : cell.Sigma) (held : RegenI64) : Option I64 :=
  match (Apeiron.GrowableIdentity.decodeRegen sigma held).match with
  | .ok key => some key
  | _ => none

/--
One production stored-carrier relocation to a planned flat slot preserves the selected canonical
observation.  The returned source table is unchanged; no merge can occur because the rooted body has
no merge parameter.
-/
theorem word_stored_relocate_entry_flat_preserves
    (state : table.Table I64 I64 table.WordHome StoredI64 16#usize)
    (plan : table.RebuildPlan) (reserved : table.RebuildReserved StoredI64 I64)
    (entryAt index : Usize) (logical : Apeiron.Table.HMap LogicalKey I64)
    (prepared : PreparedEntry state decodeStored plan reserved entryAt logical)
    (hflat : FlatPrepared prepared.entry reserved index)
    (effects : RelocationEffectsPrepared Apeiron.GrowableIdentity.storedKeyRep state
      prepared.entry (fun held => ({ key := held.key } : StoredI64)) (.flat index)) :
    word_stored_relocate_entry state plan reserved entryAt ⦃ result =>
      PayloadStateEq state result.1 ∧
        ReservedRepR result.2 decodeStored (.flat index) logical ⦄ := by
  rcases prepared.source_rep with ⟨entry, hsource, hlogical⟩
  rcases hflat with ⟨hflat, hbound⟩
  cases hsrc : prepared.entry.source with
  | Resident sourceIndex =>
      simp only [sourceObservation, hsrc] at hsource
      split at hsource <;> simp_all
      rename_i sourceSlot hsourceSlot
      obtain ⟨hlive, key, hkey, rfl⟩ := hsource
      have hkeyeq : sourceSlot.held.key = key := by simpa [decodeStored] using hkey
      subst key
      have hpayload : sourcePayload state prepared.entry.source =
          some (sourceSlot.sigma, sourceSlot.held, sourceSlot.value) := by
        simp [sourcePayload, hsrc, hsourceSlot, hlive]
      obtain ⟨nextLocator, hretarget⟩ := effects.retarget sourceSlot.sigma sourceSlot.held
        sourceSlot.value hpayload
      rw [hsrc] at hretarget
      have hretarget' :
          table.TableLocator.retarget_held core.clone.CloneI64
            table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
            (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
              Apeiron.GrowableIdentity.wordExactIdentityInst core.clone.CloneI64
              table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
            state.locator state.home sourceSlot.sigma { key := sourceSlot.held.key }
            (.Resident sourceIndex) (.Resident index) = ok nextLocator := by
        simpa [Apeiron.GrowableIdentity.storedKeyRep,
          Apeiron.GrowableIdentity.wordHomeInst, sourceLocation, destinationLocation] using hretarget
      unfold word_stored_relocate_entry table.Table.relocate_entry
      simp [prepared.entry_eq, hsrc, hsourceSlot, hflat,
        hretarget',
        table.keyrep.Stored.Insts.CoreCloneClone.clone,
        alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
        alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize,
        PayloadStateEq, ReservedRepR, reservedObservation, decodeStored, hbound, hlogical]
  | Spill shard sourceIndex =>
      simp only [sourceObservation, hsrc] at hsource
      split at hsource <;> simp_all
      rename_i sourceRegion hsourceRegion
      split at hsource <;> simp_all
      rename_i tag sourceSigma sourceHeld sourceValue sourceTag hsourceTag
      obtain ⟨htagLive, key, hkey, rfl⟩ := hsource
      have hkeyeq : sourceHeld.key = key := by simpa [decodeStored] using hkey
      subst key
      have hpayload : sourcePayload state prepared.entry.source =
          some (sourceSigma, sourceHeld, sourceValue) := by
        simp [sourcePayload, hsrc, hsourceRegion, sourceTag, hsourceTag, htagLive]
      obtain ⟨nextLocator, hretarget⟩ := effects.retarget sourceSigma sourceHeld sourceValue
        hpayload
      rw [hsrc] at hretarget
      have hretarget' :
          table.TableLocator.retarget_held core.clone.CloneI64
            table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
            (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
              Apeiron.GrowableIdentity.wordExactIdentityInst core.clone.CloneI64
              table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
            state.locator state.home sourceSigma { key := sourceHeld.key }
            (.Spill shard sourceIndex) (.Resident index) = ok nextLocator := by
        simpa [Apeiron.GrowableIdentity.storedKeyRep,
          Apeiron.GrowableIdentity.wordHomeInst, sourceLocation, destinationLocation] using hretarget
      unfold word_stored_relocate_entry table.Table.relocate_entry
      simp [prepared.entry_eq, hsrc, hsourceRegion, hsourceTag, hflat,
        hretarget',
        table.keyrep.Stored.Insts.CoreCloneClone.clone,
        alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
        alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize,
        PayloadStateEq, ReservedRepR, reservedObservation, decodeStored, hbound, hlogical]

/--
One production regenerating-carrier relocation to a planned flat slot preserves the selected
canonical observation and the separately stored `Sigma`.
-/
theorem word_regen_relocate_entry_flat_preserves
    (state : table.Table I64 I64 table.WordHome RegenI64 16#usize)
    (plan : table.RebuildPlan) (reserved : table.RebuildReserved RegenI64 I64)
    (entryAt index : Usize) (logical : Apeiron.Table.HMap LogicalKey I64)
    (prepared : PreparedEntry state decodeRegen plan reserved entryAt logical)
    (hflat : FlatPrepared prepared.entry reserved index)
    (effects : RelocationEffectsPrepared Apeiron.GrowableIdentity.regenKeyRep state
      prepared.entry (fun held => ({ identity := held.identity, codec := () } : RegenI64))
      (.flat index)) :
    word_regen_relocate_entry state plan reserved entryAt ⦃ result =>
      PayloadStateEq state result.1 ∧
        ReservedRepR result.2 decodeRegen (.flat index) logical ⦄ := by
  rcases prepared.source_rep with ⟨entry, hsource, hlogical⟩
  rcases hflat with ⟨hflat, hbound⟩
  cases hsrc : prepared.entry.source with
  | Resident sourceIndex =>
      simp only [sourceObservation, hsrc] at hsource
      split at hsource <;> simp_all
      rename_i sourceSlot hsourceSlot
      obtain ⟨hlive, key, hkey, rfl⟩ := hsource
      have hcloneDecode :
          decodeRegen sourceSlot.sigma { identity := sourceSlot.held.identity, codec := () } =
            some key := by
        simpa [decodeRegen] using hkey
      have hpayload : sourcePayload state prepared.entry.source =
          some (sourceSlot.sigma, sourceSlot.held, sourceSlot.value) := by
        simp [sourcePayload, hsrc, hsourceSlot, hlive]
      obtain ⟨nextLocator, hretarget⟩ := effects.retarget sourceSlot.sigma sourceSlot.held
        sourceSlot.value hpayload
      rw [hsrc] at hretarget
      have hretarget' :
          table.TableLocator.retarget_held core.clone.CloneI64
            table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
            (table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep
              Apeiron.GrowableIdentity.wordExactIdentityInst
              table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
              Apeiron.GrowableIdentity.wordCodecInst)
            state.locator state.home sourceSlot.sigma
            { identity := sourceSlot.held.identity, codec := () }
            (.Resident sourceIndex) (.Resident index) = ok nextLocator := by
        simpa [Apeiron.GrowableIdentity.regenKeyRep,
          Apeiron.GrowableIdentity.wordHomeInst, sourceLocation, destinationLocation] using hretarget
      unfold word_regen_relocate_entry table.Table.relocate_entry
      simp [prepared.entry_eq, hsrc, hsourceSlot, hflat,
        hretarget',
        table.keyrep.Regen.Insts.CoreCloneClone.clone,
        alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
        alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize,
        PayloadStateEq, ReservedRepR, reservedObservation, hbound, hcloneDecode, hlogical]
  | Spill shard sourceIndex =>
      simp only [sourceObservation, hsrc] at hsource
      split at hsource <;> simp_all
      rename_i sourceRegion hsourceRegion
      split at hsource <;> simp_all
      rename_i tag sourceSigma sourceHeld sourceValue sourceTag hsourceTag
      obtain ⟨htagLive, key, hkey, rfl⟩ := hsource
      have hcloneDecode :
          decodeRegen sourceSigma { identity := sourceHeld.identity, codec := () } = some key := by
        simpa [decodeRegen] using hkey
      have hpayload : sourcePayload state prepared.entry.source =
          some (sourceSigma, sourceHeld, sourceValue) := by
        simp [sourcePayload, hsrc, hsourceRegion, sourceTag, hsourceTag, htagLive]
      obtain ⟨nextLocator, hretarget⟩ := effects.retarget sourceSigma sourceHeld sourceValue
        hpayload
      rw [hsrc] at hretarget
      have hretarget' :
          table.TableLocator.retarget_held core.clone.CloneI64
            table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
            (table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep
              Apeiron.GrowableIdentity.wordExactIdentityInst
              table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
              Apeiron.GrowableIdentity.wordCodecInst)
            state.locator state.home sourceSigma
            { identity := sourceHeld.identity, codec := () }
            (.Spill shard sourceIndex) (.Resident index) = ok nextLocator := by
        simpa [Apeiron.GrowableIdentity.regenKeyRep,
          Apeiron.GrowableIdentity.wordHomeInst, sourceLocation, destinationLocation] using hretarget
      unfold word_regen_relocate_entry table.Table.relocate_entry
      simp [prepared.entry_eq, hsrc, hsourceRegion, hsourceTag, hflat,
        hretarget',
        table.keyrep.Regen.Insts.CoreCloneClone.clone,
        alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
        alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize,
        PayloadStateEq, ReservedRepR, reservedObservation, hbound, hcloneDecode, hlogical]

/--
One production stored-carrier relocation to a planned spill slot preserves the selected canonical
observation.  The source table is unchanged and the generated live tag is the occupancy authority.
-/
theorem word_stored_relocate_entry_spill_preserves
    (state : table.Table I64 I64 table.WordHome StoredI64 16#usize)
    (plan : table.RebuildPlan) (reserved : table.RebuildReserved StoredI64 I64)
    (entryAt shard index : Usize) (logical : Apeiron.Table.HMap LogicalKey I64)
    (prepared : PreparedEntry state decodeStored plan reserved entryAt logical)
    (hspill : SpillPrepared state prepared.entry reserved
      (fun held => ({ key := held.key } : StoredI64))
      (fun sigma held =>
        Apeiron.GrowableIdentity.storedKeyRep |>.address_held
            state.home sigma held)
      (fun region sigma address held =>
        table.probe.spill_locate_held Apeiron.GrowableIdentity.storedKeyRep
          region sigma address held 16#usize)
      shard index)
    (effects : RelocationEffectsPrepared Apeiron.GrowableIdentity.storedKeyRep state
      prepared.entry (fun held => ({ key := held.key } : StoredI64)) (.spill shard index)) :
    word_stored_relocate_entry state plan reserved entryAt ⦃ result =>
      PayloadStateEq state result.1 ∧
        ReservedRepR result.2 decodeStored (.spill shard index) logical ⦄ := by
  rcases prepared.source_rep with ⟨logicalEntry, hsource, hlogical⟩
  have hpayload := hspill.source_eq
  have hpublish := effects.publish
  obtain ⟨published, hpublished, hslotPublished, tag, htagPublished, htagNonzero⟩ :=
    hpublish hspill.sigma hspill.held hspill.value hspill.address hspill.region
  have hshardBound : shard.val < reserved.regions.length := by
    simpa using (List.getElem?_eq_some_iff.mp hspill.region_eq).1
  have hregionGet : reserved.regions.val[shard.val] = hspill.region :=
    (List.getElem?_eq_some_iff.mp hspill.region_eq).2
  have haddress' :
      table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.address_held
        Apeiron.GrowableIdentity.wordExactIdentityInst core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma state.home hspill.sigma
        { key := hspill.held.key } = ok hspill.address := by
    simpa [Apeiron.GrowableIdentity.storedKeyRep,
      Apeiron.GrowableIdentity.wordHomeInst] using hspill.address_eq
  have hvacant' :
      table.probe.spill_locate_held
        (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
          Apeiron.GrowableIdentity.wordExactIdentityInst core.clone.CloneI64
          table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
        hspill.region hspill.sigma hspill.address { key := hspill.held.key } 16#usize =
          ok (.Vacant index) := by
    simpa [Apeiron.GrowableIdentity.storedKeyRep] using hspill.vacant_eq
  cases hsrc : prepared.entry.source with
  | Resident sourceIndex =>
      simp only [sourceObservation, hsrc] at hsource
      simp only [sourcePayload, hsrc] at hpayload
      split at hsource <;> simp_all
      rename_i sourceSlot hsourceSlot
      obtain ⟨key, hkey, rfl⟩ := hsource
      rcases hpayload with ⟨_, hsigma, hheld, hvalue⟩
      have hkeyeq : hspill.held.key = key := by simpa [decodeStored] using hkey
      subst key
      obtain ⟨nextLocator, hretarget⟩ := effects.retarget hspill.sigma hspill.held
        hspill.value hspill.source_eq
      rw [hsrc] at hretarget
      have hretarget' :
          table.TableLocator.retarget_held core.clone.CloneI64
            table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
            (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
              Apeiron.GrowableIdentity.wordExactIdentityInst core.clone.CloneI64
              table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
            state.locator state.home hspill.sigma { key := hspill.held.key }
            (.Resident sourceIndex) (.Spill shard index) = ok nextLocator := by
        simpa [Apeiron.GrowableIdentity.storedKeyRep,
          Apeiron.GrowableIdentity.wordHomeInst, sourceLocation, destinationLocation] using hretarget
      unfold word_stored_relocate_entry table.Table.relocate_entry
      simp [prepared.entry_eq, hsrc, hsourceSlot, hspill.flat_eq, hspill.spill_eq,
        hsigma, hheld, hvalue,
        haddress', hvacant', hretarget', hregionGet, hpublished,
        table.keyrep.Stored.Insts.CoreCloneClone.clone,
        alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
        alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize,
        PayloadStateEq, ReservedRepR, reservedObservation, hshardBound,
        hslotPublished, htagPublished, htagNonzero, decodeStored, hlogical]
  | Spill sourceShard sourceIndex =>
      simp only [sourceObservation, hsrc] at hsource
      simp only [sourcePayload, hsrc] at hpayload
      split at hsource <;> simp_all
      rename_i sourceRegion hsourceRegion
      split at hsource <;> simp_all
      rename_i sourceTag sourceSigma sourceHeld sourceValue sourceTagEq hsourceSlot
      obtain ⟨key, hkey, rfl⟩ := hsource
      rcases hpayload with ⟨_, hsigma, hheld, hvalue⟩
      have hkeyeq : hspill.held.key = key := by simpa [decodeStored] using hkey
      subst key
      obtain ⟨nextLocator, hretarget⟩ := effects.retarget hspill.sigma hspill.held
        hspill.value hspill.source_eq
      rw [hsrc] at hretarget
      have hretarget' :
          table.TableLocator.retarget_held core.clone.CloneI64
            table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
            (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
              Apeiron.GrowableIdentity.wordExactIdentityInst core.clone.CloneI64
              table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
            state.locator state.home hspill.sigma { key := hspill.held.key }
            (.Spill sourceShard sourceIndex) (.Spill shard index) = ok nextLocator := by
        simpa [Apeiron.GrowableIdentity.storedKeyRep,
          Apeiron.GrowableIdentity.wordHomeInst, sourceLocation, destinationLocation] using hretarget
      unfold word_stored_relocate_entry table.Table.relocate_entry
      simp [prepared.entry_eq, hsrc, hsourceRegion, hsourceSlot, hspill.flat_eq,
        hspill.spill_eq,
        haddress', hvacant', hretarget', hregionGet, hpublished,
        table.keyrep.Stored.Insts.CoreCloneClone.clone,
        alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
        alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize,
        PayloadStateEq, ReservedRepR, reservedObservation, hshardBound,
        hslotPublished, htagPublished, htagNonzero, decodeStored, hlogical]

/--
One production regenerating-carrier relocation to a planned spill slot preserves the decoded key,
value, and separately stored `Sigma`; the live tag makes the copied payload observable.
-/
theorem word_regen_relocate_entry_spill_preserves
    (state : table.Table I64 I64 table.WordHome RegenI64 16#usize)
    (plan : table.RebuildPlan) (reserved : table.RebuildReserved RegenI64 I64)
    (entryAt shard index : Usize) (logical : Apeiron.Table.HMap LogicalKey I64)
    (prepared : PreparedEntry state decodeRegen plan reserved entryAt logical)
    (hspill : SpillPrepared state prepared.entry reserved
      (fun held => ({ identity := held.identity, codec := () } : RegenI64))
      (fun sigma held =>
        Apeiron.GrowableIdentity.regenKeyRep |>.address_held state.home sigma held)
      (fun region sigma address held =>
        table.probe.spill_locate_held Apeiron.GrowableIdentity.regenKeyRep
          region sigma address held 16#usize)
      shard index)
    (effects : RelocationEffectsPrepared Apeiron.GrowableIdentity.regenKeyRep state
      prepared.entry (fun held => ({ identity := held.identity, codec := () } : RegenI64))
      (.spill shard index)) :
    word_regen_relocate_entry state plan reserved entryAt ⦃ result =>
      PayloadStateEq state result.1 ∧
        ReservedRepR result.2 decodeRegen (.spill shard index) logical ⦄ := by
  rcases prepared.source_rep with ⟨logicalEntry, hsource, hlogical⟩
  have hpayload := hspill.source_eq
  have hpublish := effects.publish
  obtain ⟨published, hpublished, hslotPublished, tag, htagPublished, htagNonzero⟩ :=
    hpublish hspill.sigma hspill.held hspill.value hspill.address hspill.region
  have hshardBound : shard.val < reserved.regions.length := by
    simpa using (List.getElem?_eq_some_iff.mp hspill.region_eq).1
  have hregionGet : reserved.regions.val[shard.val] = hspill.region :=
    (List.getElem?_eq_some_iff.mp hspill.region_eq).2
  have haddress' :
      table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.address_held
        Apeiron.GrowableIdentity.wordExactIdentityInst
        table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
        Apeiron.GrowableIdentity.wordCodecInst state.home hspill.sigma
        { identity := hspill.held.identity, codec := () } = ok hspill.address := by
    simpa [Apeiron.GrowableIdentity.regenKeyRep,
      Apeiron.GrowableIdentity.wordHomeInst] using hspill.address_eq
  have hvacant' :
      table.probe.spill_locate_held
        (table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep
          Apeiron.GrowableIdentity.wordExactIdentityInst
          table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
          Apeiron.GrowableIdentity.wordCodecInst)
        hspill.region hspill.sigma hspill.address
        { identity := hspill.held.identity, codec := () } 16#usize = ok (.Vacant index) := by
    simpa [Apeiron.GrowableIdentity.regenKeyRep] using hspill.vacant_eq
  cases hsrc : prepared.entry.source with
  | Resident sourceIndex =>
      simp only [sourceObservation, hsrc] at hsource
      simp only [sourcePayload, hsrc] at hpayload
      split at hsource <;> simp_all
      rename_i sourceSlot hsourceSlot
      obtain ⟨key, hkey, rfl⟩ := hsource
      rcases hpayload with ⟨_, hsigma, hheld, hvalue⟩
      have hcloneDecode :
          decodeRegen hspill.sigma { identity := hspill.held.identity, codec := () } = some key := by
        simpa [decodeRegen] using hkey
      obtain ⟨nextLocator, hretarget⟩ := effects.retarget hspill.sigma hspill.held
        hspill.value hspill.source_eq
      rw [hsrc] at hretarget
      have hretarget' :
          table.TableLocator.retarget_held core.clone.CloneI64
            table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
            (table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep
              Apeiron.GrowableIdentity.wordExactIdentityInst
              table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
              Apeiron.GrowableIdentity.wordCodecInst)
            state.locator state.home hspill.sigma
            { identity := hspill.held.identity, codec := () }
            (.Resident sourceIndex) (.Spill shard index) = ok nextLocator := by
        simpa [Apeiron.GrowableIdentity.regenKeyRep,
          Apeiron.GrowableIdentity.wordHomeInst, sourceLocation, destinationLocation] using hretarget
      unfold word_regen_relocate_entry table.Table.relocate_entry
      simp [prepared.entry_eq, hsrc, hsourceSlot, hspill.flat_eq, hspill.spill_eq,
        hsigma, hheld, hvalue,
        haddress', hvacant', hretarget', hregionGet, hpublished,
        table.keyrep.Regen.Insts.CoreCloneClone.clone,
        alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
        alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize,
        PayloadStateEq, ReservedRepR, reservedObservation, hshardBound,
        hslotPublished, htagPublished, htagNonzero, hcloneDecode, hlogical]
  | Spill sourceShard sourceIndex =>
      simp only [sourceObservation, hsrc] at hsource
      simp only [sourcePayload, hsrc] at hpayload
      split at hsource <;> simp_all
      rename_i sourceRegion hsourceRegion
      split at hsource <;> simp_all
      rename_i sourceTag sourceSigma sourceHeld sourceValue sourceTagEq hsourceSlot
      obtain ⟨key, hkey, rfl⟩ := hsource
      rcases hpayload with ⟨_, hsigma, hheld, hvalue⟩
      have hcloneDecode :
          decodeRegen hspill.sigma { identity := hspill.held.identity, codec := () } = some key := by
        simpa [decodeRegen] using hkey
      obtain ⟨nextLocator, hretarget⟩ := effects.retarget hspill.sigma hspill.held
        hspill.value hspill.source_eq
      rw [hsrc] at hretarget
      have hretarget' :
          table.TableLocator.retarget_held core.clone.CloneI64
            table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
            (table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep
              Apeiron.GrowableIdentity.wordExactIdentityInst
              table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
              Apeiron.GrowableIdentity.wordCodecInst)
            state.locator state.home hspill.sigma
            { identity := hspill.held.identity, codec := () }
            (.Spill sourceShard sourceIndex) (.Spill shard index) = ok nextLocator := by
        simpa [Apeiron.GrowableIdentity.regenKeyRep,
          Apeiron.GrowableIdentity.wordHomeInst, sourceLocation, destinationLocation] using hretarget
      unfold word_regen_relocate_entry table.Table.relocate_entry
      simp [prepared.entry_eq, hsrc, hsourceRegion, hsourceSlot, hspill.flat_eq,
        hspill.spill_eq,
        haddress', hvacant', hretarget', hregionGet, hpublished,
        table.keyrep.Regen.Insts.CoreCloneClone.clone,
        alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
        alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize,
        PayloadStateEq, ReservedRepR, reservedObservation, hshardBound,
        hslotPublished, htagPublished, htagNonzero, hcloneDecode, hlogical]

/-!
## Rooted traversal composition

The traversal itself has no callback argument.  The following two induction principles expose the
exact remaining semantic obligation: an input-side certificate must show that each rooted entry
step preserves the chosen prefix invariant.  The four entry theorems above discharge the selected
destination observation in such a certificate; disjointness, old/new live-set equality, uniqueness,
and direct coverage require the still-unmaterialized `RebuildPlanCorrect` predicate.
-/

/-- The stored-carrier rooted traversal preserves every invariant preserved by each rooted step. -/
theorem word_stored_relocate_all_preserves
    (state : table.Table I64 I64 table.WordHome StoredI64 16#usize)
    (plan : table.RebuildPlan) (reserved : table.RebuildReserved StoredI64 I64)
    (inv : table.RebuildReserved StoredI64 I64 → Prop) (hinitial : inv reserved)
    (hstep : ∀ current entryAt, entryAt.val < plan.entries.length → inv current →
      word_stored_relocate_entry state plan current entryAt ⦃ result =>
        result.1 = state ∧ inv result.2 ⦄) :
    word_stored_relocate_all state plan reserved ⦃ result =>
      result.1 = state ∧ inv result.2 ⦄ := by
  unfold word_stored_relocate_all table.Table.relocate_all
  unfold table.Table.relocate_all_loop
  apply loop.spec_decr_nat
    (measure := fun p :
      table.Table I64 I64 table.WordHome StoredI64 16#usize ×
        table.RebuildReserved StoredI64 I64 × Usize => plan.entries.length - p.2.2.val)
    (inv := fun p => p.1 = state ∧ p.2.2.val ≤ plan.entries.length ∧ inv p.2.1)
    (post := fun result :
      table.Table I64 I64 table.WordHome StoredI64 16#usize ×
        table.RebuildReserved StoredI64 I64 => result.1 = state ∧ inv result.2)
  · intro current hcurrent
    rcases current with ⟨currentState, currentReserved, entryAt⟩
    rcases hcurrent with ⟨hstate, hbound, hinv⟩
    dsimp only at hstate hbound hinv ⊢
    subst currentState
    unfold table.Table.relocate_all_loop.body
    by_cases hlt : entryAt < alloc.vec.Vec.len plan.entries
    · rw [if_pos hlt]
      have hlt' : entryAt.val < plan.entries.length := by simpa [alloc.vec.Vec.len] using hlt
      have hrelocate := hstep currentReserved entryAt hlt' hinv
      unfold word_stored_relocate_entry at hrelocate
      step with hrelocate
      step
      refine ⟨self1_post, ?_, self1_post1, ?_⟩ <;> omega
    · rw [if_neg hlt, WP.spec_ok]
      exact ⟨rfl, hinv⟩
  · exact ⟨rfl, Nat.zero_le _, hinitial⟩

/-- The regenerating-carrier rooted traversal preserves every invariant preserved by each step. -/
theorem word_regen_relocate_all_preserves
    (state : table.Table I64 I64 table.WordHome RegenI64 16#usize)
    (plan : table.RebuildPlan) (reserved : table.RebuildReserved RegenI64 I64)
    (inv : table.RebuildReserved RegenI64 I64 → Prop) (hinitial : inv reserved)
    (hstep : ∀ current entryAt, entryAt.val < plan.entries.length → inv current →
      word_regen_relocate_entry state plan current entryAt ⦃ result =>
        result.1 = state ∧ inv result.2 ⦄) :
    word_regen_relocate_all state plan reserved ⦃ result =>
      result.1 = state ∧ inv result.2 ⦄ := by
  unfold word_regen_relocate_all table.Table.relocate_all
  unfold table.Table.relocate_all_loop
  apply loop.spec_decr_nat
    (measure := fun p :
      table.Table I64 I64 table.WordHome RegenI64 16#usize ×
        table.RebuildReserved RegenI64 I64 × Usize => plan.entries.length - p.2.2.val)
    (inv := fun p => p.1 = state ∧ p.2.2.val ≤ plan.entries.length ∧ inv p.2.1)
    (post := fun result :
      table.Table I64 I64 table.WordHome RegenI64 16#usize ×
        table.RebuildReserved RegenI64 I64 => result.1 = state ∧ inv result.2)
  · intro current hcurrent
    rcases current with ⟨currentState, currentReserved, entryAt⟩
    rcases hcurrent with ⟨hstate, hbound, hinv⟩
    dsimp only at hstate hbound hinv ⊢
    subst currentState
    unfold table.Table.relocate_all_loop.body
    by_cases hlt : entryAt < alloc.vec.Vec.len plan.entries
    · rw [if_pos hlt]
      have hlt' : entryAt.val < plan.entries.length := by simpa [alloc.vec.Vec.len] using hlt
      have hrelocate := hstep currentReserved entryAt hlt' hinv
      unfold word_regen_relocate_entry at hrelocate
      step with hrelocate
      step
      refine ⟨self1_post, ?_, self1_post1, ?_⟩ <;> omega
    · rw [if_neg hlt, WP.spec_ok]
      exact ⟨rfl, hinv⟩
  · exact ⟨rfl, Nat.zero_le _, hinitial⟩

end Apeiron.GrowableRelocation
