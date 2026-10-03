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
          match region.slots.val[index.val]? with
          | some slot =>
              if slot.tag = 0#u8 then none
              else (decode slot.sigma slot.held).map fun key => ((slot.sigma, key), slot.value)
          | none => none
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
          match region.slots.val[index.val]? with
          | some slot =>
              if slot.tag = 0#u8 then none
              else (decode slot.sigma slot.held).map fun key => ((slot.sigma, key), slot.value)
          | none => none
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
          match region.slots.val[index.val]? with
          | some slot =>
              if slot.tag = 0#u8 then none
              else some (slot.sigma, slot.held, slot.value)
          | none => none
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
    (entry : table.RebuildPlanEntry) (reserved : table.RebuildReserved R I64)
    (cloneHeld : R → R) (destination : Destination) : Prop where
  retarget : ∀ sigma held value,
    sourcePayload state entry.source = some (sigma, held, value) →
    ∃ nextLocator,
      table.TableLocator.retarget_held core.clone.CloneI64
        Apeiron.GrowableIdentity.wordHomeInst keyRep state.locator state.home sigma
        (cloneHeld held) (sourceLocation entry.source) (destinationLocation destination) =
          ok nextLocator
  publish : match destination with
    | .flat _ => True
    | .spill shard index => ∀ (sigma : cell.Sigma) (held : R) (value : I64)
        (address : table.TableAddress) (region : table.Region R I64),
        sourcePayload state entry.source = some (sigma, held, value) →
        keyRep.address_held state.home sigma (cloneHeld held) = ok address →
        reserved.regions.val[shard.val]? = some region →
        ∃ (published : table.Region R I64) (tag : U8),
          table.Region.publish_live region index (sigma, cloneHeld held, value) address.place =
              ok published ∧
            published.slots.val[index.val]? =
              some ⟨tag, sigma, cloneHeld held, value⟩ ∧
            tag ≠ 0#u8

/-- The two generated outer-vector updates retain the payload write while adding the tag write. -/
def writeSpill {R : Type} (reserved : table.RebuildReserved R I64) (shard : Usize)
    (published : table.Region R I64) : table.RebuildReserved R I64 :=
  { reserved with regions := reserved.regions.set shard published }

/-- The generated flat branch replaces exactly one initialized resident slot. -/
def writeFlat {R : Type} (reserved : table.RebuildReserved R I64) (index : Usize)
    (slot : table.FlatSlot R I64) : table.RebuildReserved R I64 :=
  { reserved with resident :=
      { reserved.resident with slots := reserved.resident.slots.set index slot } }

private theorem getElem?_set_eq_of_ne {α : Type} (xs : List α) (written read : Nat)
    (value : α) (hne : read ≠ written) : (xs.set written value)[read]? = xs[read]? := by
  simp [hne]

/-- Read back the generated flat destination slot through the outer resident-vector update. -/
theorem reservedObservation_writeFlat_same {R : Type} (reserved : table.RebuildReserved R I64)
    (decode : cell.Sigma → R → Option I64) (index : Usize) (slot : table.FlatSlot R I64)
    (hbound : index.val < reserved.resident.slots.length) :
    reservedObservation (writeFlat reserved index slot) decode (.flat index) =
      (if slot.live then
        (decode slot.sigma slot.held).map fun key => ((slot.sigma, key), slot.value)
      else none) := by
  simp only [reservedObservation, writeFlat, alloc.vec.Vec.set_val_eq]
  rw [List.getElem?_set_self (by simpa using hbound)]

/-- Read back the generated spill destination slot through the outer shard-vector update. -/
theorem reservedObservation_writeSpill_same {R : Type} (reserved : table.RebuildReserved R I64)
    (decode : cell.Sigma → R → Option I64) (shard index : Usize)
    (published : table.Region R I64) (sigma : cell.Sigma) (held : R) (value : I64) (tag : U8)
    (hbound : shard.val < reserved.regions.length)
    (hslot : published.slots.val[index.val]? = some ⟨tag, sigma, held, value⟩)
    (htag : tag ≠ 0#u8) :
    reservedObservation (writeSpill reserved shard published) decode (.spill shard index) =
      (decode sigma held).map fun key => ((sigma, key), value) := by
  simp only [reservedObservation, writeSpill, alloc.vec.Vec.set_val_eq]
  rw [List.getElem?_set_self (by simpa using hbound)]
  dsimp only
  rw [hslot]
  simp [htag]

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
    (decode : cell.Sigma → R → Option I64) (shard earlier : Usize)
    (published : table.Region R I64) :
    reservedObservation (writeSpill reserved shard published) decode (.flat earlier) =
      reservedObservation reserved decode (.flat earlier) := by
  rfl

/-- A spill write retains an observation in every other spill shard. -/
theorem writeSpill_frame_other_shard {R : Type}
    (reserved : table.RebuildReserved R I64) (decode : cell.Sigma → R → Option I64)
    (writtenShard earlierShard earlierIndex : Usize) (published : table.Region R I64)
    (hne : earlierShard.val ≠ writtenShard.val) :
    reservedObservation (writeSpill reserved writtenShard published) decode
        (.spill earlierShard earlierIndex) =
      reservedObservation reserved decode (.spill earlierShard earlierIndex) := by
  simp only [reservedObservation, writeSpill, alloc.vec.Vec.set_val_eq]
  rw [getElem?_set_eq_of_ne _ _ _ _ hne]

/-- A spill write retains every untouched slot of the written shard. -/
theorem writeSpill_frame_same_shard {R : Type}
    (reserved : table.RebuildReserved R I64) (decode : cell.Sigma → R → Option I64)
    (shard earlier : Usize) (region published : table.Region R I64)
    (hregion : reserved.regions.val[shard.val]? = some region)
    (hbound : shard.val < reserved.regions.length)
    (hagree : published.slots.val[earlier.val]? = region.slots.val[earlier.val]?) :
    reservedObservation (writeSpill reserved shard published) decode (.spill shard earlier) =
      reservedObservation reserved decode (.spill shard earlier) := by
  simp only [reservedObservation, writeSpill, alloc.vec.Vec.set_val_eq]
  rw [List.getElem?_set_self (by simpa using hbound), hregion]
  dsimp only
  simp only [hagree]

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
      prepared.entry reserved (fun held => ({ key := held.key } : StoredI64)) (.flat index)) :
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
      rename_i sourceSlot hsourceSlot
      obtain ⟨htagLive, key, hkey, rfl⟩ := hsource
      have hkeyeq : sourceSlot.held.key = key := by simpa [decodeStored] using hkey
      subst key
      have hpayload : sourcePayload state prepared.entry.source =
          some (sourceSlot.sigma, sourceSlot.held, sourceSlot.value) := by
        simp [sourcePayload, hsrc, hsourceRegion, hsourceSlot, htagLive]
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
            (.Spill shard sourceIndex) (.Resident index) = ok nextLocator := by
        simpa [Apeiron.GrowableIdentity.storedKeyRep,
          Apeiron.GrowableIdentity.wordHomeInst, sourceLocation, destinationLocation] using hretarget
      unfold word_stored_relocate_entry table.Table.relocate_entry
      simp [prepared.entry_eq, hsrc, hsourceRegion, hsourceSlot, hflat,
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
      prepared.entry reserved (fun held => ({ identity := held.identity, codec := () } : RegenI64))
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
      rename_i sourceSlot hsourceSlot
      obtain ⟨htagLive, key, hkey, rfl⟩ := hsource
      have hcloneDecode :
          decodeRegen sourceSlot.sigma { identity := sourceSlot.held.identity, codec := () } =
            some key := by
        simpa [decodeRegen] using hkey
      have hpayload : sourcePayload state prepared.entry.source =
          some (sourceSlot.sigma, sourceSlot.held, sourceSlot.value) := by
        simp [sourcePayload, hsrc, hsourceRegion, hsourceSlot, htagLive]
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
            (.Spill shard sourceIndex) (.Resident index) = ok nextLocator := by
        simpa [Apeiron.GrowableIdentity.regenKeyRep,
          Apeiron.GrowableIdentity.wordHomeInst, sourceLocation, destinationLocation] using hretarget
      unfold word_regen_relocate_entry table.Table.relocate_entry
      simp [prepared.entry_eq, hsrc, hsourceRegion, hsourceSlot, hflat,
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
      prepared.entry reserved (fun held => ({ key := held.key } : StoredI64)) (.spill shard index)) :
    word_stored_relocate_entry state plan reserved entryAt ⦃ result =>
      PayloadStateEq state result.1 ∧
        ReservedRepR result.2 decodeStored (.spill shard index) logical ⦄ := by
  rcases prepared.source_rep with ⟨logicalEntry, hsource, hlogical⟩
  have hpayload := hspill.source_eq
  have haddress' :
      table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.address_held
        Apeiron.GrowableIdentity.wordExactIdentityInst core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma state.home hspill.sigma
        { key := hspill.held.key } = ok hspill.address := by
    simpa [Apeiron.GrowableIdentity.storedKeyRep,
      Apeiron.GrowableIdentity.wordHomeInst] using hspill.address_eq
  obtain ⟨published, tag, hpublished, hslotPublished, htagNonzero⟩ :=
    effects.publish hspill.sigma hspill.held hspill.value hspill.address hspill.region
      hspill.source_eq haddress' hspill.region_eq
  have hshardBound : shard.val < reserved.regions.length := by
    simpa using (List.getElem?_eq_some_iff.mp hspill.region_eq).1
  have hregionGet : reserved.regions.val[shard.val] = hspill.region :=
    (List.getElem?_eq_some_iff.mp hspill.region_eq).2
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
        hslotPublished, htagNonzero, decodeStored, hlogical]
  | Spill sourceShard sourceIndex =>
      simp only [sourceObservation, hsrc] at hsource
      simp only [sourcePayload, hsrc] at hpayload
      split at hsource <;> simp_all
      rename_i sourceRegion hsourceRegion
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
            (.Spill sourceShard sourceIndex) (.Spill shard index) = ok nextLocator := by
        simpa [Apeiron.GrowableIdentity.storedKeyRep,
          Apeiron.GrowableIdentity.wordHomeInst, sourceLocation, destinationLocation] using hretarget
      unfold word_stored_relocate_entry table.Table.relocate_entry
      simp [prepared.entry_eq, hsrc, hsourceRegion, hsourceSlot, hspill.flat_eq,
        hspill.spill_eq,
        hsigma, hheld, hvalue,
        haddress', hvacant', hretarget', hregionGet, hpublished,
        table.keyrep.Stored.Insts.CoreCloneClone.clone,
        alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
        alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize,
        PayloadStateEq, ReservedRepR, reservedObservation, hshardBound,
        hslotPublished, htagNonzero, decodeStored, hlogical]

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
      prepared.entry reserved (fun held => ({ identity := held.identity, codec := () } : RegenI64))
      (.spill shard index)) :
    word_regen_relocate_entry state plan reserved entryAt ⦃ result =>
      PayloadStateEq state result.1 ∧
        ReservedRepR result.2 decodeRegen (.spill shard index) logical ⦄ := by
  rcases prepared.source_rep with ⟨logicalEntry, hsource, hlogical⟩
  have hpayload := hspill.source_eq
  have haddress' :
      table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.address_held
        Apeiron.GrowableIdentity.wordExactIdentityInst
        table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
        Apeiron.GrowableIdentity.wordCodecInst state.home hspill.sigma
        { identity := hspill.held.identity, codec := () } = ok hspill.address := by
    simpa [Apeiron.GrowableIdentity.regenKeyRep,
      Apeiron.GrowableIdentity.wordHomeInst] using hspill.address_eq
  obtain ⟨published, tag, hpublished, hslotPublished, htagNonzero⟩ :=
    effects.publish hspill.sigma hspill.held hspill.value hspill.address hspill.region
      hspill.source_eq haddress' hspill.region_eq
  have hshardBound : shard.val < reserved.regions.length := by
    simpa using (List.getElem?_eq_some_iff.mp hspill.region_eq).1
  have hregionGet : reserved.regions.val[shard.val] = hspill.region :=
    (List.getElem?_eq_some_iff.mp hspill.region_eq).2
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
        hslotPublished, htagNonzero, hcloneDecode, hlogical]
  | Spill sourceShard sourceIndex =>
      simp only [sourceObservation, hsrc] at hsource
      simp only [sourcePayload, hsrc] at hpayload
      split at hsource <;> simp_all
      rename_i sourceRegion hsourceRegion
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
            (.Spill sourceShard sourceIndex) (.Spill shard index) = ok nextLocator := by
        simpa [Apeiron.GrowableIdentity.regenKeyRep,
          Apeiron.GrowableIdentity.wordHomeInst, sourceLocation, destinationLocation] using hretarget
      unfold word_regen_relocate_entry table.Table.relocate_entry
      simp [prepared.entry_eq, hsrc, hsourceRegion, hsourceSlot, hspill.flat_eq,
        hspill.spill_eq,
        hsigma, hheld, hvalue,
        haddress', hvacant', hretarget', hregionGet, hpublished,
        table.keyrep.Regen.Insts.CoreCloneClone.clone,
        alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
        alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize,
        PayloadStateEq, ReservedRepR, reservedObservation, hshardBound,
        hslotPublished, htagNonzero, hcloneDecode, hlogical]

/-!
## Rooted traversal composition

The traversal itself has no callback argument.  The following two induction principles expose the
exact remaining semantic obligation: an input-side certificate must show that each rooted entry
step preserves the chosen prefix invariant.  The four entry theorems above discharge the selected
destination observation in such a certificate; disjointness, old/new live-set equality, uniqueness,
and direct coverage require the still-unmaterialized `RebuildPlanCorrect` predicate.
-/

/-!
## Generated `retarget_held` success and failure

Read from the generated body at `proof/ApeironWord.lean` (digest `94edd03fda64e8b037073bf5209bd0535e3f94e0cab86ead09f38db550e5891c`),
`table.TableLocator.retarget_held` has exactly five outcome branches and one failure channel:

1. `table.same_location old new = ok true` → `ok self`. `same_location` is total (below).
2. `same_location old new = ok false`, `linked_handle … = ok none` → `ok self`.
3. `same_location old new = ok false`, `linked_handle … = ok (some handle)` and
   `UScalar.cast .Usize handle ≥ alloc.vec.Vec.len self.links` → `ok self`.
4. `same_location old new = ok false`, `linked_handle … = ok (some handle)`, the link index is in
   bounds, the stored `current` location satisfies `same_location current old = ok false` →
   `ok self`.
5. Same as 4 with `same_location current old = ok true` → `ok` of `self` with `links` updated at
   `UScalar.cast .Usize handle` to `(identity_node, new)` (`alloc.vec.Vec.set`; the generated
   `index_mut` continuation is `Vec.set` at that index).

Failure: the body's only fallible call whose success is not decided by its own guards is
`table.TableLocator.linked_handle`. `table.same_location`, `lift (UScalar.cast …)` (pure
truncating cast), and `alloc.vec.Vec.index`/`index_mut` (bounds discharged by the `handle <
len links` guard) never fail. So `retarget_held` returns `ok` exactly when `same_location old new =
ok true` or `linked_handle` returns `ok`, and otherwise returns the `linked_handle` failure
unchanged. `linked_handle` itself is total for `directory = none` and `sigma = Elliptic` (both
`ok none`, the `Option` residual maps to `ok none`), but not on arbitrary locator states: its
`Directory.walk` chain performs bounds-checked `read_word` reads (`residue_identity_root`,
`find_edge`, the `walk_identity` terminal read), a `massert (i < arity)`, and checked `Usize`
arithmetic on node-derived indexes, any of which fails on a malformed `directory`
(`Error.arrayOutOfBounds`, `Error.assertionFailure`, `Error.integerOverflow`).

Consequently `sourcePayload state entry.source = some (sigma, held, value)` does **not** entail
`∃ next, retarget_held … = ok next`: `sourcePayload` constrains payload fields only, not
`state.locator`, and a `linked_handle` refusal refutes the totality demand
(`retarget_held_refuses_of_linked_handle_fail` below). The traversal induction's `hstep` supply
therefore needs a locator-side success certificate; the exact success condition is
`retarget_held_succeeds_iff`.
-/

/-- `same_location` is total: it returns `ok b` for a Bool `b` on every input pair. -/
theorem generated_same_location_total (left right : table.PayloadLocation) :
    ∃ b, table.same_location left right = ok b := by
  cases left with
  | Resident li =>
    cases right with
    | Resident ri => exact ⟨_, rfl⟩
    | Spill rs ri => exact ⟨_, rfl⟩
  | Spill ls li =>
    cases right with
    | Resident ri => exact ⟨_, rfl⟩
    | Spill rs ri =>
      simp only [table.same_location]
      split <;> exact ⟨_, rfl⟩

/-- Bind inversion at `ok`: a `bind m k` that returns `ok` forces `m = ok`. -/
private theorem ok_of_bind_ok {α β : Type} (m : Result α) (k : α → Result β) (b : β)
    (h : m >>= k = ok b) : ∃ a, m = ok a := by
  cases hm : m.match with
  | ok a => exact ⟨a, Result.match.isOk.mp hm⟩
  | vis i kk =>
    rw [Result.match.isVis.mp hm] at h
    simp at h
  | div =>
    rw [Result.match.isDiv.mp hm] at h
    simp at h

/-- Branch 1: a true `same_location` guard short-circuits to the unchanged locator. -/
theorem retarget_held_eq_ok_self_of_same {K H R Faces : Type} (cloneInst : core.clone.Clone K)
    (homeInst : table.Home H K Faces) (keyRep : table.keyrep.KeyRep R K H)
    (self : table.TableLocator) (home : H) (sigma : cell.Sigma) (held : R)
    (old new : table.PayloadLocation) (hsame : table.same_location old new = ok true) :
    table.TableLocator.retarget_held cloneInst homeInst keyRep self home sigma held old new =
      ok self := by
  simp [table.TableLocator.retarget_held, hsame]

/-- Branch 2: a `none` from `linked_handle` leaves the locator unchanged. -/
theorem retarget_held_eq_ok_self_of_no_handle {K H R Faces : Type} (cloneInst : core.clone.Clone K)
    (homeInst : table.Home H K Faces) (keyRep : table.keyrep.KeyRep R K H)
    (self : table.TableLocator) (home : H) (sigma : cell.Sigma) (held : R)
    (old new : table.PayloadLocation) (hsame : table.same_location old new = ok false)
    (hlinked : table.TableLocator.linked_handle cloneInst homeInst keyRep self home sigma held =
      ok none) :
    table.TableLocator.retarget_held cloneInst homeInst keyRep self home sigma held old new =
      ok self := by
  simp [table.TableLocator.retarget_held, hsame, hlinked]

/-- Branch 3: a link handle at or past `links` length leaves the locator unchanged. -/
theorem retarget_held_eq_ok_self_of_link_out_of_bounds {K H R Faces : Type}
    (cloneInst : core.clone.Clone K) (homeInst : table.Home H K Faces)
    (keyRep : table.keyrep.KeyRep R K H) (self : table.TableLocator) (home : H)
    (sigma : cell.Sigma) (held : R) (old new : table.PayloadLocation) (handle : Std.U32)
    (hsame : table.same_location old new = ok false)
    (hlinked : table.TableLocator.linked_handle cloneInst homeInst keyRep self home sigma held =
      ok (some handle))
    (hoob : ¬ UScalar.cast .Usize handle < alloc.vec.Vec.len self.links) :
    table.TableLocator.retarget_held cloneInst homeInst keyRep self home sigma held old new =
      ok self := by
  simp only [table.TableLocator.retarget_held, hsame, hlinked, lift, bind_tc_ok, if_neg hoob]
  simp

/-- Branch 4: an in-bounds link whose stored location is not `old` leaves the locator unchanged. -/
theorem retarget_held_eq_ok_self_of_current_distinct {K H R Faces : Type}
    (cloneInst : core.clone.Clone K) (homeInst : table.Home H K Faces)
    (keyRep : table.keyrep.KeyRep R K H) (self : table.TableLocator) (home : H)
    (sigma : cell.Sigma) (held : R) (old new : table.PayloadLocation) (handle : Std.U32)
    (identity_node : Std.U32) (current : table.PayloadLocation)
    (hsame : table.same_location old new = ok false)
    (hlinked : table.TableLocator.linked_handle cloneInst homeInst keyRep self home sigma held =
      ok (some handle))
    (hlt : UScalar.cast .Usize handle < alloc.vec.Vec.len self.links)
    (hidx : alloc.vec.Vec.index_usize self.links (UScalar.cast .Usize handle) =
      ok (identity_node, current))
    (hcur : table.same_location current old = ok false) :
    table.TableLocator.retarget_held cloneInst homeInst keyRep self home sigma held old new =
      ok self := by
  simp only [table.TableLocator.retarget_held, hsame, hlinked, lift, bind_tc_ok, if_pos hlt,
    alloc.vec.Vec.index_slice_index, hidx]
  simp [hcur]

/--
Branch 5: an in-bounds link whose stored location is `old` is rewritten to `new`; the resulting
locator is `self` with `links.set (UScalar.cast .Usize handle) (identity_node, new)`.
-/
theorem retarget_held_eq_ok_set_of_current_matches {K H R Faces : Type}
    (cloneInst : core.clone.Clone K) (homeInst : table.Home H K Faces)
    (keyRep : table.keyrep.KeyRep R K H) (self : table.TableLocator) (home : H)
    (sigma : cell.Sigma) (held : R) (old new : table.PayloadLocation) (handle : Std.U32)
    (identity_node : Std.U32) (current : table.PayloadLocation)
    (hsame : table.same_location old new = ok false)
    (hlinked : table.TableLocator.linked_handle cloneInst homeInst keyRep self home sigma held =
      ok (some handle))
    (hlt : UScalar.cast .Usize handle < alloc.vec.Vec.len self.links)
    (hidx : alloc.vec.Vec.index_usize self.links (UScalar.cast .Usize handle) =
      ok (identity_node, current))
    (hcur : table.same_location current old = ok true) :
    table.TableLocator.retarget_held cloneInst homeInst keyRep self home sigma held old new =
      ok { self with links := alloc.vec.Vec.set self.links (UScalar.cast .Usize handle) (identity_node, new) } := by
  simp only [table.TableLocator.retarget_held, hsame, hlinked, lift, bind_tc_ok, if_pos hlt,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_mut_usize, hidx]
  simp [hcur]

/-- Failure propagation: with distinct locations, a `linked_handle` refusal is the whole result. -/
theorem retarget_held_fails_of_linked_handle_fail {K H R Faces : Type}
    (cloneInst : core.clone.Clone K) (homeInst : table.Home H K Faces)
    (keyRep : table.keyrep.KeyRep R K H) (self : table.TableLocator) (home : H)
    (sigma : cell.Sigma) (held : R) (old new : table.PayloadLocation) (e : Error)
    (hsame : table.same_location old new = ok false)
    (hlinked : table.TableLocator.linked_handle cloneInst homeInst keyRep self home sigma held =
      fail e) :
    table.TableLocator.retarget_held cloneInst homeInst keyRep self home sigma held old new =
      fail e := by
  simp [table.TableLocator.retarget_held, hsame, hlinked]

/--
The exact success condition: `retarget_held` returns `ok` if and only if either location guard is
true (branch 1) or `linked_handle` returns `ok` (branches 2-5, whose tail is total under the body's
own bounds guard).
-/
theorem retarget_held_succeeds_iff {K H R Faces : Type} (cloneInst : core.clone.Clone K)
    (homeInst : table.Home H K Faces) (keyRep : table.keyrep.KeyRep R K H)
    (self : table.TableLocator) (home : H) (sigma : cell.Sigma) (held : R)
    (old new : table.PayloadLocation) :
    (∃ next, table.TableLocator.retarget_held cloneInst homeInst keyRep self home sigma held old
        new = ok next) ↔
      (table.same_location old new = ok true ∨
        ∃ h, table.TableLocator.linked_handle cloneInst homeInst keyRep self home sigma held =
          ok h) := by
  obtain ⟨b, hb⟩ := generated_same_location_total old new
  constructor
  · intro hex
    obtain ⟨next, hnext⟩ := hex
    cases b with
    | true => exact Or.inl hb
    | false =>
      right
      simp only [table.TableLocator.retarget_held, hb, bind_tc_ok,
        if_neg Bool.false_ne_true] at hnext
      exact ok_of_bind_ok _ _ _ hnext
  · rintro (hsame | ⟨h, hlinked⟩)
    · exact ⟨_, retarget_held_eq_ok_self_of_same cloneInst homeInst keyRep self home sigma held
        old new hsame⟩
    · cases b with
      | true =>
        exact ⟨_, retarget_held_eq_ok_self_of_same cloneInst homeInst keyRep self home sigma held
          old new hb⟩
      | false =>
        cases h with
        | none =>
          exact ⟨_, retarget_held_eq_ok_self_of_no_handle cloneInst homeInst keyRep self home
            sigma held old new hb hlinked⟩
        | some handle =>
          by_cases hlt : UScalar.cast .Usize handle < alloc.vec.Vec.len self.links
          · have hbound' : (UScalar.cast .Usize handle).val < self.links.val.length := by
              simpa [alloc.vec.Vec.len] using hlt
            have hidx : alloc.vec.Vec.index_usize self.links (UScalar.cast .Usize handle) =
                ok (self.links.val[(UScalar.cast .Usize handle).val].1,
                  self.links.val[(UScalar.cast .Usize handle).val].2) := by
              rw [show alloc.vec.Vec.index_usize self.links (UScalar.cast .Usize handle) =
                  ok self.links.val[(UScalar.cast .Usize handle).val] by
                simp only [alloc.vec.Vec.index_usize]
                rw [show self.links[(UScalar.cast .Usize handle).val]? =
                    some self.links.val[(UScalar.cast .Usize handle).val] from
                    List.getElem?_eq_getElem hbound']]
            obtain ⟨b2, hb2⟩ := generated_same_location_total
              self.links.val[(UScalar.cast .Usize handle).val].2 old
            cases b2 with
            | false =>
              exact ⟨_, retarget_held_eq_ok_self_of_current_distinct cloneInst homeInst keyRep self
                home sigma held old new handle _ _ hb hlinked hlt hidx hb2⟩
            | true =>
              exact ⟨_, retarget_held_eq_ok_set_of_current_matches cloneInst homeInst keyRep self
                home sigma held old new handle _ _ hb hlinked hlt hidx hb2⟩
          · exact ⟨_, retarget_held_eq_ok_self_of_link_out_of_bounds cloneInst homeInst keyRep self
              home sigma held old new handle hb hlinked hlt⟩

/--
Counterexample shape to the traversal induction's totality demand: under a `sourcePayload`
observation and distinct source/destination locations, one `linked_handle` refusal (any `Error`
from the `Directory.walk` chain) refutes `∃ next, retarget_held … = ok next`. No totality theorem
in the shape `sourcePayload … → ∃ next, retarget_held … = ok next` is provable without a
locator-side success certificate.
-/
theorem retarget_held_refuses_of_linked_handle_fail {R : Type}
    (keyRep : table.keyrep.KeyRep R I64 table.WordHome)
    (state : table.Table I64 I64 table.WordHome R 16#usize) (entry : table.RebuildPlanEntry)
    (cloneHeld : R → R) (destination : Destination) (sigma : cell.Sigma) (held : R) (value : I64)
    (e : Error)
    (_hsource : sourcePayload state entry.source = some (sigma, held, value))
    (hne : table.same_location (sourceLocation entry.source) (destinationLocation destination) =
      ok false)
    (hlinked : table.TableLocator.linked_handle core.clone.CloneI64
      Apeiron.GrowableIdentity.wordHomeInst keyRep state.locator state.home sigma (cloneHeld held) =
        fail e) :
    ¬ ∃ next, table.TableLocator.retarget_held core.clone.CloneI64
      Apeiron.GrowableIdentity.wordHomeInst keyRep state.locator state.home sigma (cloneHeld held)
      (sourceLocation entry.source) (destinationLocation destination) = ok next := by
  intro hex
  obtain ⟨next, hnext⟩ := hex
  have hfail := retarget_held_fails_of_linked_handle_fail core.clone.CloneI64
    Apeiron.GrowableIdentity.wordHomeInst keyRep state.locator state.home sigma (cloneHeld held)
    (sourceLocation entry.source) (destinationLocation destination) e hne hlinked
  rw [hfail] at hnext
  exact fail_not_ok hnext

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

/-!
## Locator well-formedness and `hstep` totality

Read from the generated bodies at `proof/ApeironWord.lean` (digest
`94edd03fda64e8b037073bf5209bd0535e3f94e0cab86ead09f38db550e5891c`).  `linked_handle` derives
every index it feeds to `read_word` from words stored in `residue_words`, `identity_words` and
`edge_words`; the predicates below spell out the exact bounds those derived indexes need, and the
totality theorems discharge the traversal's `hstep` premise from them.

Record layout: a residue record is `RESIDUE_STRIDE = 4` words (`residue_words[4n + 3]` is the
identity root), an identity node is `IDENTITY_STRIDE = 2` words (`identity_words[2n]` is the
terminal handle, `identity_words[2n + 1]` the first edge), an edge is `EDGE_STRIDE = 3` words
(`edge_words[3e .. 3e + 2]` = label, child, next).  A residue *node value* `v` indexes
`residue_words` at `4v + 3`; an *identity node value* `v` indexes `identity_words` at `2v` and
`2v + 1`; an *edge value* `v` indexes `edge_words` at `3v .. 3v + 2`.
-/

/-- A residue node index: `NO_CHILD` or its identity-root read `residue_words[4v + 3]` is in
bounds. -/
def ResidueNodeWF (d : coord.locator.locator_core.Directory) (node : Std.U32) : Prop :=
  node = coord.locator.locator_core.NO_CHILD ∨
    4 * node.val + 3 < d.residue_words.val.length

/-- An identity node index: both derived identity reads `identity_words[2v]` (terminal) and
`identity_words[2v + 1]` (first edge) are in bounds. -/
def IdentityNodeWF (d : coord.locator.locator_core.Directory) (node : Std.U32) : Prop :=
  2 * node.val + 1 < d.identity_words.val.length

/-- An edge index: `NO_CHILD` or all three edge reads are in bounds and `EDGE_NEXT` strictly
decreases (so `find_edge`'s loop terminates). -/
def EdgeIndexWF (d : coord.locator.locator_core.Directory) (e : Std.U32) : Prop :=
  e = coord.locator.locator_core.NO_CHILD ∨
    (3 * e.val + 2 < d.edge_words.val.length ∧
      ∀ n, d.edge_words.val[3 * e.val + 2]? = some n →
        n = coord.locator.locator_core.NO_CHILD ∨ n.val < e.val)

/-- Word-array well-formedness of a locator directory: each stored word class carries exactly the
bounds its derived reads need. -/
structure DirectoryWF (d : coord.locator.locator_core.Directory) : Prop where
  /-- A residue child word (`4n + digit`, `digit < 3`) is a residue node value or `NO_CHILD`. -/
  residue_child_bound :
    ∀ (p : Nat) (v : Std.U32), d.residue_words.val[p]? = some v → p % 4 ≠ 3 →
      v ≠ coord.locator.locator_core.NO_CHILD → 4 * v.val + 3 < d.residue_words.val.length
  /-- A residue identity root (`4n + 3`) is an identity node value or `NO_CHILD`. -/
  residue_identity_bound :
    ∀ (p : Nat) (v : Std.U32), d.residue_words.val[p]? = some v → p % 4 = 3 →
      v ≠ coord.locator.locator_core.NO_CHILD → 2 * v.val + 1 < d.identity_words.val.length
  /-- An identity first-edge word (`2n + 1`) is an `EdgeIndexWF` edge value. -/
  identity_edge_bound :
    ∀ (p : Nat) (v : Std.U32), d.identity_words.val[p]? = some v → p % 2 = 1 →
      EdgeIndexWF d v
  /-- An edge child word (`3e + 1`) is an identity node value or `NO_CHILD`. -/
  edge_child_bound :
    ∀ (p : Nat) (v : Std.U32), d.edge_words.val[p]? = some v → p % 3 = 1 →
      v ≠ coord.locator.locator_core.NO_CHILD → 2 * v.val + 1 < d.identity_words.val.length
  /-- An edge next word (`3e + 2`) is an `EdgeIndexWF` edge value. -/
  edge_next_bound :
    ∀ (p : Nat) (v : Std.U32), d.edge_words.val[p]? = some v → p % 3 = 2 →
      EdgeIndexWF d v

/-- Well-formedness of one `ResidueIndex` instance against one directory: total and bounded
`LEVELS`/`ARITY`/`ARENA`, digits below `ARITY` (the `massert (i < arity)` in `walk_residue`), and
the arena root in bounds whenever a zero-level walk can return it unchanged. -/
structure ResidueIndexWF {I : Type} (inst : coord.locator.locator_core.ResidueIndex I)
    (d : coord.locator.locator_core.Directory) (levels arity : Std.Usize) (root : Std.U32) : Prop where
  /-- `LEVELS` succeeds and never exceeds the 64-entry digit buffer. -/
  levels_eq : inst.LEVELS = ok levels
  levels_le : levels.val ≤ 64
  /-- `ARITY` succeeds, is positive (the 0 digit passes `massert`) and at most `MAX_ARITY = 3`
  (so every residue child read lands on a child word, `4n + digit` with `digit < 3`). -/
  arity_eq : inst.ARITY = ok arity
  arity_pos : 0 < arity.val
  arity_le : arity.val ≤ 3
  /-- `ARENA` succeeds with `root`. -/
  arena_eq : inst.ARENA = ok root
  /-- With zero levels `walk_residue` returns the root unchanged, so `residue_identity_root` reads
  `residue_words[4 * root + 3]`; with any levels the returned node is a stored child value instead. -/
  root_bound : levels.val = 0 → ResidueNodeWF d root
  /-- Every digit written below `levels` is strictly below `ARITY`: the `massert (i < arity)` in
  `walk_residue_loop.body` cannot fail. -/
  digits_bounded : ∀ (index : I) (buf : Slice Std.U8), buf.val.length = levels.val →
    ∃ out, inst.digits index buf = ok out ∧ out.val.length = buf.val.length ∧
      ∀ (k : Nat) (dv : Std.U8), k < levels.val → out.val[k]? = some dv →
        core.convert.num.FromUsizeU8.from dv < arity

/-- Totality of the `ExactIdentity` methods `walk_identity` calls. -/
structure ExactIdentityWF {K Ret Res : Type}
    (inst : coord.locator.locator_core.ExactIdentity K Ret Res) : Prop where
  encoding_len_total : ∀ key, ∃ len, inst.encoding_len key = ok len
  encoding_byte_total : ∀ key (index : Std.Usize), ∃ b, inst.encoding_byte key index = ok b

/-- `EdgeIndexWF` depends only on `edge_words`. -/
theorem EdgeIndexWF.of_words_eq {d d' : coord.locator.locator_core.Directory} {e : Std.U32}
    (he : d'.edge_words = d.edge_words) (h : EdgeIndexWF d e) : EdgeIndexWF d' e := by
  simp only [EdgeIndexWF, he]
  exact h

/-- `IdentityNodeWF` depends only on `identity_words`. -/
theorem IdentityNodeWF.of_words_eq {d d' : coord.locator.locator_core.Directory} {node : Std.U32}
    (hi : d'.identity_words = d.identity_words) (h : IdentityNodeWF d node) :
    IdentityNodeWF d' node := by
  simp only [IdentityNodeWF, hi]
  exact h

/-- `ResidueNodeWF` depends only on `residue_words`. -/
theorem ResidueNodeWF.of_words_eq {d d' : coord.locator.locator_core.Directory} {node : Std.U32}
    (hr : d'.residue_words = d.residue_words) (h : ResidueNodeWF d node) :
    ResidueNodeWF d' node := by
  simp only [ResidueNodeWF, hr]
  exact h

/-- `DirectoryWF` depends only on the three word vectors. -/
theorem DirectoryWF.of_words_eq {d d' : coord.locator.locator_core.Directory}
    (h : d'.residue_words = d.residue_words ∧ d'.identity_words = d.identity_words ∧
      d'.edge_words = d.edge_words) : DirectoryWF d → DirectoryWF d' := by
  intro wf
  rcases h with ⟨hr, hi, he⟩
  constructor
  · intro p v hget hp hno
    rw [hr] at hget
    rw [hr]
    exact wf.residue_child_bound p v hget hp hno
  · intro p v hget hp hno
    rw [hr] at hget
    rw [hi]
    exact wf.residue_identity_bound p v hget hp hno
  · intro p v hget hp
    rw [hi] at hget
    exact EdgeIndexWF.of_words_eq he (wf.identity_edge_bound p v hget hp)
  · intro p v hget hp hno
    rw [he] at hget
    exact IdentityNodeWF.of_words_eq hi (wf.edge_child_bound p v hget hp hno)
  · intro p v hget hp
    rw [he] at hget
    exact EdgeIndexWF.of_words_eq he (wf.edge_next_bound p v hget hp)


/-- `read_word` succeeds in bounds and returns the stored word. -/
theorem read_word_ok (words : Slice Std.U32) (i : Std.Usize)
    (hbound : i.val < words.val.length) :
    coord.locator.locator_core.read_word words i = ok words.val[i.val] := by
  simp only [coord.locator.locator_core.read_word, Slice.index_usize]
  rw [show words[i]? = some words.val[i.val] by simp [hbound]]

/-- First-edge words are `EdgeIndexWF` edge values. -/
theorem DirectoryWF.edgeIndexWF_of_first_edge {d : coord.locator.locator_core.Directory}
    (wf : DirectoryWF d) {p : Nat} {v : Std.U32}
    (hget : d.identity_words.val[p]? = some v) (hp : p % 2 = 1) : EdgeIndexWF d v :=
  wf.identity_edge_bound p v hget hp

/-- Edge next words are `EdgeIndexWF` edge values. -/
theorem DirectoryWF.edgeIndexWF_of_next {d : coord.locator.locator_core.Directory}
    (wf : DirectoryWF d) {p : Nat} {v : Std.U32}
    (hget : d.edge_words.val[p]? = some v) (hp : p % 3 = 2) : EdgeIndexWF d v :=
  wf.edge_next_bound p v hget hp

/-- Edge child words are identity node values. -/
theorem DirectoryWF.identityNodeWF_of_edge_child {d : coord.locator.locator_core.Directory}
    (wf : DirectoryWF d) {p : Nat} {v : Std.U32}
    (hget : d.edge_words.val[p]? = some v) (hp : p % 3 = 1)
    (hno : v ≠ coord.locator.locator_core.NO_CHILD) : IdentityNodeWF d v :=
  wf.edge_child_bound p v hget hp hno

/-- Residue identity roots are identity node values. -/
theorem DirectoryWF.identityNodeWF_of_identity_root {d : coord.locator.locator_core.Directory}
    (wf : DirectoryWF d) {p : Nat} {v : Std.U32}
    (hget : d.residue_words.val[p]? = some v) (hp : p % 4 = 3)
    (hno : v ≠ coord.locator.locator_core.NO_CHILD) : IdentityNodeWF d v :=
  wf.residue_identity_bound p v hget hp hno

/-- Residue child words are residue node values. -/
theorem DirectoryWF.residueNodeWF_of_child {d : coord.locator.locator_core.Directory}
    (wf : DirectoryWF d) {p : Nat} {v : Std.U32}
    (hget : d.residue_words.val[p]? = some v) (hp : p % 4 ≠ 3) : ResidueNodeWF d v := by
  by_cases hno : v = coord.locator.locator_core.NO_CHILD
  · exact Or.inl hno
  · exact Or.inr (wf.residue_child_bound p v hget hp hno)


/-- `Directory.new` establishes `DirectoryWF` vacuously: all three word vectors are empty, so every
word-class field has an empty population. -/
theorem directory_new_wf :
    ∃ d, coord.locator.locator_core.Directory.new = ok d ∧ DirectoryWF d := by
  refine ⟨_, rfl, ?_⟩
  constructor
  · intro p v hget _ _
    simp [alloc.vec.Vec.new] at hget
  · intro p v hget _ _
    simp [alloc.vec.Vec.new] at hget
  · intro p v hget _
    simp [alloc.vec.Vec.new] at hget
  · intro p v hget _ _
    simp [alloc.vec.Vec.new] at hget
  · intro p v hget _
    simp [alloc.vec.Vec.new] at hget

/-- `recycle_scratch` changes only `staged_scratch` and preserves `DirectoryWF`. -/
theorem recycle_scratch_wf (d : coord.locator.locator_core.Directory)
    (path : alloc.vec.Vec Std.U32) (wf : DirectoryWF d) :
    ∃ d', coord.locator.locator_core.Directory.recycle_scratch d path = ok d' ∧ DirectoryWF d' :=
  ⟨{d with staged_scratch := some path}, rfl,
    DirectoryWF.of_words_eq (d := d) (d' := {d with staged_scratch := some path})
      ⟨rfl, rfl, rfl⟩ wf⟩

/-- `recycle_staged` changes only `staged_scratch` and preserves `DirectoryWF`. -/
theorem recycle_staged_wf (d : coord.locator.locator_core.Directory)
    (staged : coord.locator.locator_core.StagedPath) (wf : DirectoryWF d) :
    ∃ d', coord.locator.locator_core.Directory.recycle_staged d staged = ok d' ∧ DirectoryWF d' :=
  ⟨{d with staged_scratch := some staged.identity_path}, rfl,
    DirectoryWF.of_words_eq (d := d) (d' := {d with staged_scratch := some staged.identity_path})
      ⟨rfl, rfl, rfl⟩ wf⟩

end Apeiron.GrowableRelocation
