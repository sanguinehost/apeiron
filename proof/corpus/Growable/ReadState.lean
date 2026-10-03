import Growable.Representation
import Growable.Resources
import Lean

/-!
# Stored tag/address invariant and reserved relocation preservation

Split from `Growable.ReadLoop` to satisfy the repository's physical-line bound. Declarations keep
their exact names and statements in `Apeiron.GrowableReadLoop`. `StoredLiveTagAddress` records the
stored tag/address law on occupancy-authoritative live spill slots: the tag lives in the
`table.SpillSlot` at the lane. `StoredTaggedRepR` conjoins it with the canonical whole-table
`RepR`. The commit bridges preserve it across the generated apply/publication bodies.
`ReservedStoredLiveTagAddress` carries the analog on the rebuilt view of reserved storage, with the
per-step relocation results and the traversal induction. `Growable.ReadLoop` imports this module,
so every existing importer sees the same names transitively.
-/

namespace Apeiron.GrowableReadLoop

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-- Every occupancy-authoritative live spill tag identifies the generated
fingerprint and shard of its *stored* full key. Slots behind zero tags are
excluded. This is an additional state invariant, not a field of `RepR`. -/
def StoredLiveTagAddress
    (state : Apeiron.GrowableRepresentation.WordTable
      Apeiron.GrowableIdentity.StoredI64) : Prop :=
  ∀ (shard index : Usize)
      (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
      (slot : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64),
    state.shards.val[shard.val]? = some region →
    region.slots.val[index.val]? = some slot → slot.tag ≠ 0#u8 →
    ∃ address : table.TableAddress,
      Apeiron.GrowableIdentity.wordHomeInst.address state.home slot.sigma slot.held.key =
        ok address ∧ table.shard address = ok shard ∧
      table.tag address.place = ok slot.tag

/-- The separately tracked tag law and the canonical whole-table relation
are jointly required at a represented stored-word state. -/
def StoredTaggedRepR
    (state : Apeiron.GrowableRepresentation.WordTable
      Apeiron.GrowableIdentity.StoredI64)
    (direct : Apeiron.GrowableRepresentation.LogicalKey →
      Option Apeiron.GrowableRepresentation.Location)
    (live : Apeiron.GrowableRepresentation.LogicalKey → Prop)
    (logical : Apeiron.Table.HMap Apeiron.GrowableRepresentation.LogicalKey I64) : Prop :=
  Apeiron.GrowableRepresentation.RepR state Apeiron.GrowableSearch.decodeStoredOption
    direct live logical ∧ StoredLiveTagAddress state

/-- No tag can be live when `RepR.liveSet` is empty and the Stored carrier
always decodes. This proves initialization without treating `RepR` as the
missing nonempty-state tag law. -/
theorem stored_live_tag_address_of_empty_repR
    (state : Apeiron.GrowableRepresentation.WordTable
      Apeiron.GrowableIdentity.StoredI64)
    (default : I64)
    (hrep : Apeiron.GrowableRepresentation.RepR state
      Apeiron.GrowableSearch.decodeStoredOption (fun _ => none)
      (fun _ => False) (fun _ => default)) :
    StoredLiveTagAddress state := by
  intro shard index region slot hregion hslot htagNonzero
  have hentry : Apeiron.GrowableSearch.entryAt state
      Apeiron.GrowableSearch.decodeStoredOption (.spill shard index) =
        some ((slot.sigma, slot.held.key), slot.value) := by
    simp [Apeiron.GrowableSearch.entryAt, hregion, hslot,
      htagNonzero, Apeiron.GrowableSearch.decodeStoredOption]
  have hempty : False :=
    (hrep.liveSet (slot.sigma, slot.held.key)).mpr ⟨.spill shard index, slot.value, hentry⟩
  exact False.elim hempty

/-- The real generated constructor establishes both parts of the stored-state
invariant, at the same empty state returned by the source body. -/
theorem word_stored_table_new_tagged_repR (default : I64) :
    table.Table.new 16#usize Apeiron.GrowableIdentity.wordHomeInst
      Apeiron.GrowableIdentity.storedKeyRep default () ⦃ state =>
      StoredTaggedRepR state (fun _ => none) (fun _ => False) (fun _ => default) ⦄ := by
  apply WP.spec_mono (Apeiron.GrowableRepresentation.word_stored_table_new_repR default)
  intro state hrep
  exact ⟨hrep, stored_live_tag_address_of_empty_repR state default hrep⟩

/-- A materialized target with only zero spill tags has no live destination;
this is the actual pre-traversal property named by the resource certificate. -/
theorem stored_live_tags_initial_reserved
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (reserved : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64)
    (hvacant : Apeiron.GrowableResources.MaterializedVacant reserved) :
    StoredLiveTagAddress (Apeiron.GrowableRefinement.rebuiltView state reserved) := by
  intro shard index region slot hregion hslot hnonzero
  have hstored : reserved.regions.val[shard.val]? = some region := by
    simpa [Apeiron.GrowableRefinement.rebuiltView] using hregion
  have hmem : region ∈ reserved.regions.val := List.mem_of_getElem? hstored
  have hslotMem : slot ∈ region.slots.val := List.mem_of_getElem? hslot
  exact False.elim (hnonzero ((hvacant.2 region hmem) slot hslotMem))

/-- The input-side prepared certificate initializes the same invariant on
its own exact state and reserved storage; it does not prove adapter realization. -/
theorem prepared_stored_live_tags_initial
    (certificate : Apeiron.GrowableResources.PreparedResourceClauses
      Apeiron.GrowableIdentity.StoredI64) :
    StoredLiveTagAddress (Apeiron.GrowableRefinement.rebuiltView
      certificate.state certificate.storage) :=
  stored_live_tags_initial_reserved certificate.state certificate.storage
    certificate.materializedVacant

/-- Flat-only and scalar-field updates frame the spill-tag invariant when
their source-linked post-state retains both the home dictionary and shards. -/
theorem stored_live_tag_address_of_home_shards_eq
    (before after : Apeiron.GrowableRepresentation.WordTable
      Apeiron.GrowableIdentity.StoredI64)
    (hbefore : StoredLiveTagAddress before)
    (hhome : after.home = before.home)
    (hshards : after.shards = before.shards) :
    StoredLiveTagAddress after := by
  intro shard index region slot hregion hslot hnonzero
  have hregionBefore : before.shards.val[shard.val]? = some region := by
    simpa only [hshards] using hregion
  obtain ⟨address, haddress, hroute, htagAddress⟩ :=
    hbefore shard index region slot hregionBefore hslot hnonzero
  exact ⟨address, by simpa only [hhome] using haddress, hroute, htagAddress⟩

/-- A found-spill value-only update retains every tag and every held key. This
is the exact `replaceSpillValue` post-state used by the generated found-spill
apply theorem; it does not establish found classification or `RepR` preservation. -/
theorem stored_live_tag_address_replace_spill_value
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveTagAddress state)
    (shard index : Usize)
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (oldSlot : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)
    (newValue : I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (hslot : region.slots.val[index.val]? = some oldSlot) :
    StoredLiveTagAddress
      (Apeiron.GrowableSearch.replaceSpillValue state shard region index
        { oldSlot with value := newValue }) := by
  intro readShard readIndex readRegion readSlot hreadRegion hreadSlot htagNonzero
  have hshardBound : shard.val < state.shards.length :=
    (List.getElem?_eq_some_iff.mp hregion).1
  have hslotBound : index.val < region.slots.length :=
    (List.getElem?_eq_some_iff.mp hslot).1
  by_cases hsameShard : readShard.val = shard.val
  · have hshardEq : readShard = shard := UScalar.eq_of_val_eq hsameShard
    subst readShard
    have hnewRegion : readRegion =
        { region with slots := region.slots.set index { oldSlot with value := newValue } } := by
      simpa only [Apeiron.GrowableSearch.replaceSpillValue,
        alloc.vec.Vec.set_val_eq, List.getElem?_set_self hshardBound,
        Option.some.injEq] using hreadRegion.symm
    subst readRegion
    by_cases hsameIndex : readIndex.val = index.val
    · have hindexEq : readIndex = index := UScalar.eq_of_val_eq hsameIndex
      subst readIndex
      have hslotNew : readSlot = { oldSlot with value := newValue } := by
        simpa only [alloc.vec.Vec.set_val_eq,
          List.getElem?_set_self hslotBound, Option.some.injEq] using hreadSlot.symm
      subst readSlot
      exact hstate shard index region oldSlot hregion hslot htagNonzero
    · have hslotOld : region.slots.val[readIndex.val]? =
          some readSlot := by
        simpa [alloc.vec.Vec.set_val_eq, hsameIndex] using hreadSlot
      exact hstate shard readIndex region readSlot hregion hslotOld htagNonzero
  · have hregionOld : state.shards.val[readShard.val]? = some readRegion := by
      simpa [Apeiron.GrowableSearch.replaceSpillValue,
        alloc.vec.Vec.set_val_eq, hsameShard] using hreadRegion
    exact hstate readShard readIndex readRegion readSlot hregionOld hreadSlot htagNonzero

/-- The actual generated slot publication writes only its selected lane; every
other slot of the region is framed. -/
theorem generated_publish_live_frames {R V : Type}
    (before after : table.Region R V) (index read : Usize)
    (payload : cell.Sigma × R × V) (place : U64)
    (hpub : table.Region.publish_live before index payload place = ok after)
    (hne : read.val ≠ index.val) :
    after.slots.val[read.val]? = before.slots.val[read.val]? :=
  Apeiron.GrowableRepresentation.publish_live_slot_frame before after index read
    payload place hpub hne

/-- The generated found-spill apply refines the value-only replacement above;
its *returned table* retains the global tag/address invariant. -/
theorem word_stored_apply_insert_found_spill_keeps_tags
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveTagAddress state)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress)
    (shard index : Usize)
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (slot : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (hslot : region.slots.val[index.val]? = some slot) :
    ∃ next : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64,
      word_stored_apply_insert state
        { «class» := .Found, location := .Spill shard index }
        sigma key value address = ok (next, none) ∧
      StoredLiveTagAddress next := by
  refine ⟨Apeiron.GrowableSearch.replaceSpillValue state shard region index
      { slot with value := core.num.I64.wrapping_add value slot.value }, ?_, ?_⟩
  · exact Apeiron.GrowableSearch.word_stored_apply_insert_found_spill state
      sigma key value address shard index region slot hregion hslot
  · exact stored_live_tag_address_replace_spill_value state hstate shard index region
      slot (core.num.I64.wrapping_add value slot.value) hregion hslot

/-- The generated found-flat return changes no home or spill field. -/
theorem word_stored_apply_insert_found_flat_keeps_tags
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveTagAddress state) (sigma : cell.Sigma)
    (key value : I64) (address : table.TableAddress) (index : Usize)
    (slot : table.FlatSlot Apeiron.GrowableIdentity.StoredI64 I64)
    (hslot : state.resident.slots.val[index.val]? = some slot) :
    ∃ next, word_stored_apply_insert state
      { «class» := .Found, location := .Flat index } sigma key value address =
        ok (next, none) ∧ StoredLiveTagAddress next := by
  refine ⟨Apeiron.GrowableSearch.replaceFlatValue state index
    { slot with value := core.num.I64.wrapping_add value slot.value },
    Apeiron.GrowableSearch.word_stored_apply_insert_found_flat state
      sigma key value address index slot hslot, ?_⟩
  exact stored_live_tag_address_of_home_shards_eq _ _ hstate rfl rfl

/-- The generated vacant-flat return changes no home or spill field. -/
theorem word_stored_apply_insert_vacant_flat_keeps_tags
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveTagAddress state) (sigma : cell.Sigma)
    (key value : I64) (address : table.TableAddress) (index : Usize)
    (slot : table.FlatSlot Apeiron.GrowableIdentity.StoredI64 I64)
    (hslot : state.resident.slots.val[index.val]? = some slot) (_hfree : slot.live = false)
    (nextLocator : table.TableLocator)
    (hlocator : table.TableLocator.publish_for_query core.clone.CloneI64
      Apeiron.GrowableIdentity.wordHomeInst Apeiron.GrowableIdentity.storedKeyRep
      state.locator state.home sigma key (.Resident index) = ok nextLocator)
    (hlen : state.len.val + 1 < Usize.max) :
    ∃ next, word_stored_apply_insert state
      { «class» := .Vacant, location := .Flat index } sigma key value address =
        ok (next, none) ∧ StoredLiveTagAddress next := by
  obtain ⟨next, hrun, hhome, hshards⟩ :=
    Apeiron.GrowableRepresentation.word_stored_vacant_flat_return_frame state
      sigma key value address index slot hslot nextLocator hlocator hlen
  exact ⟨next, hrun, stored_live_tag_address_of_home_shards_eq state next
    hstate hhome hshards⟩

/-- A fresh generated publication preserves the whole-state invariant when
the source operation frames every other tag and payload lane. The frame is
explicit: the landed publication lemmas prove the selected lane, not this
untouched-lane property or the public commit's shard update. -/
theorem stored_live_tag_address_publish_spill_of_frame
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveTagAddress state)
    (shard index : Usize)
    (region published : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (sigma : cell.Sigma) (held : Apeiron.GrowableIdentity.StoredI64)
    (value : I64) (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma held.key =
      ok address)
    (hshard : table.shard address = ok shard)
    (hpublish : table.Region.publish_live region index (sigma, held, value)
      address.place = ok published)
    (hframe : ∀ other : Usize, other.val ≠ index.val →
      published.slots.val[other.val]? = region.slots.val[other.val]?) :
    StoredLiveTagAddress { state with shards := state.shards.set shard published } := by
  intro readShard readIndex readRegion readSlot hreadRegion hreadSlot htagNonzero
  have hshardBound : shard.val < state.shards.length :=
    (List.getElem?_eq_some_iff.mp hregion).1
  by_cases hsameShard : readShard.val = shard.val
  · have hshardEq : readShard = shard := UScalar.eq_of_val_eq hsameShard
    subst readShard
    have hnewRegion : readRegion = published := by
      simpa only [alloc.vec.Vec.set_val_eq,
        List.getElem?_set_self hshardBound, Option.some.injEq] using hreadRegion.symm
    subst readRegion
    by_cases hsameIndex : readIndex.val = index.val
    · have hindexEq : readIndex = index := UScalar.eq_of_val_eq hsameIndex
      subst readIndex
      obtain ⟨tag, htagGenerated, _htagNz, hslots⟩ :=
        Apeiron.GrowableRepresentation.publish_live_slots_exact region published index
          (sigma, held, value) address.place hpublish
      have hbound : index.val < region.slots.length := by
        have hle : index.val < (region.slots.val.set index.val
            ⟨tag, sigma, held, value⟩).length := by
          rw [← hslots]
          exact (List.getElem?_eq_some_iff.mp hreadSlot).1
        rw [List.length_set] at hle
        exact hle
      have hx : published.slots.val[index.val]? =
          some ⟨tag, sigma, held, value⟩ := by
        rw [hslots]
        exact List.getElem?_set_self hbound
      have hreadEq : readSlot = ⟨tag, sigma, held, value⟩ :=
        Option.some.inj (hreadSlot.symm.trans hx)
      subst readSlot
      exact ⟨address, haddress, hshard, htagGenerated⟩
    · have hreadOld : region.slots.val[readIndex.val]? = some readSlot := by
        rw [← hframe readIndex hsameIndex]
        exact hreadSlot
      exact hstate shard readIndex region readSlot hregion hreadOld htagNonzero
  · have hregionOld : state.shards.val[readShard.val]? = some readRegion := by
      simpa [alloc.vec.Vec.set_val_eq, hsameShard] using hreadRegion
    exact hstate readShard readIndex readRegion readSlot hregionOld hreadSlot htagNonzero

/-- Discharge the frame premise of the fresh-spill state update from the
generated publication itself. The address/shard equalities remain explicit
input-side routing obligations, not post-state assumptions. -/
theorem stored_live_tag_address_publish_spill
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveTagAddress state)
    (shard index : Usize)
    (region published : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (sigma : cell.Sigma) (held : Apeiron.GrowableIdentity.StoredI64)
    (value : I64) (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma held.key =
      ok address)
    (hshard : table.shard address = ok shard)
    (hpublish : table.Region.publish_live region index (sigma, held, value)
      address.place = ok published) :
    StoredLiveTagAddress { state with shards := state.shards.set shard published } := by
  apply stored_live_tag_address_publish_spill_of_frame state hstate shard index
    region published hregion sigma held value address haddress hshard hpublish
  intro other hne
  exact generated_publish_live_frames region published index other
    (sigma, held, value) address.place hpublish hne

/-- The *returned* table of the generated vacant-spill apply preserves tags,
provided query routing identifies its address and shard. -/
theorem word_stored_apply_insert_vacant_spill_keeps_tags
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveTagAddress state) (sigma : cell.Sigma)
    (key value : I64) (address : table.TableAddress) (shard index : Usize)
    (region published : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (nextLocator : table.TableLocator)
    (hlocator : table.TableLocator.publish_for_query core.clone.CloneI64
      Apeiron.GrowableIdentity.wordHomeInst Apeiron.GrowableIdentity.storedKeyRep
      state.locator state.home sigma key (.Spill shard index) = ok nextLocator)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (hshard : table.shard address = ok shard)
    (hpublish : table.Region.publish_live region index
      (sigma, { key := key }, core.num.I64.wrapping_add value state.default)
      address.place = ok published)
    (hlen : state.len.val + 1 < Usize.max) :
    ∃ next : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64,
      word_stored_apply_insert state
        { «class» := .Vacant, location := .Spill shard index }
        sigma key value address = ok (next, none) ∧ StoredLiveTagAddress next := by
  obtain ⟨next, hrun, hhome, hshards⟩ :=
    Apeiron.GrowableRepresentation.word_stored_vacant_spill_return_shards
      state sigma key value address shard index region published hregion
      nextLocator hlocator hpublish hlen
  refine ⟨next, hrun, ?_⟩
  exact stored_live_tag_address_of_home_shards_eq _ next
    (stored_live_tag_address_publish_spill state hstate shard index region published
      hregion sigma { key := key } (core.num.I64.wrapping_add value state.default)
      address haddress hshard hpublish) hhome hshards

/-!
## Reserved tag/address invariant across the rebuild traversal

`ReservedStoredLiveTagAddress` is the `ReservedRepR`-wide `StoredLiveTagAddress` analog on the
rebuilt table view of reserved relocation storage.  The per-step results discharge the spill and
flat relocation steps of the generated body; the traversal result threads them over
`table.Table.relocate_all`.  The per-step premises at every visited prefix state remain an
explicit caller obligation: the plan-side certificate binding to the visited states is not
established here.
-/

/-- The `ReservedRepR`-wide `StoredLiveTagAddress` analog: the stored tag/address invariant on
the rebuilt table view of reserved relocation storage. -/
def ReservedStoredLiveTagAddress
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (reserved : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64) : Prop :=
  StoredLiveTagAddress (Apeiron.GrowableRefinement.rebuiltView state reserved)

/-- One certified generated spill relocation step preserves the reserved tag/address invariant.
Composes the step's exact post-state equation with the landed publication framing chain
(`stored_live_tag_address_publish_spill`, `stored_live_tag_address_of_home_shards_eq`). -/
theorem stored_relocate_entry_spill_keeps_reserved_tags
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (plan : table.RebuildPlan)
    (current : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64)
    (entryAt shard index : Usize)
    (entry : table.RebuildPlanEntry)
    (sigma : cell.Sigma) (held : Apeiron.GrowableIdentity.StoredI64) (value : I64)
    (address : table.TableAddress)
    (region published : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (nextLocator : table.TableLocator)
    (hstate : StoredLiveTagAddress (Apeiron.GrowableRefinement.rebuiltView state current))
    (hentry : plan.entries.val[entryAt.val]? = some entry)
    (hsource : Apeiron.GrowableRelocation.sourcePayload state entry.source =
      some (sigma, held, value))
    (hflat : entry.flat = none)
    (hspill : entry.spill = shard)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma held.key =
      ok address)
    (hregion : current.regions.val[shard.val]? = some region)
    (hvacant : table.probe.spill_locate_held Apeiron.GrowableIdentity.storedKeyRep region sigma
      address { key := held.key } 16#usize = ok (.Vacant index))
    (hretarget : table.TableLocator.retarget_held core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
      Apeiron.GrowableIdentity.storedKeyRep state.locator state.home sigma { key := held.key }
      (Apeiron.GrowableRelocation.sourceLocation entry.source)
      (Apeiron.GrowableRelocation.destinationLocation (.spill shard index)) = ok nextLocator)
    (hpublish : table.Region.publish_live region index (sigma, { key := held.key }, value)
      address.place = ok published)
    (hshard : table.shard address = ok shard) :
    word_stored_relocate_entry state plan current entryAt ⦃ result =>
      ReservedStoredLiveTagAddress result.1 result.2 ⦄ := by
  rw [Apeiron.GrowableRepresentation.word_stored_relocate_entry_spill_run state plan current
    entryAt shard index entry sigma held value address region published nextLocator hentry hsource
    hflat hspill haddress hregion hvacant hretarget hpublish, WP.spec_ok]
  have hregionView :
      (Apeiron.GrowableRefinement.rebuiltView state current).shards.val[shard.val]? =
        some region := by
    simpa [Apeiron.GrowableRefinement.rebuiltView] using hregion
  have haddressView :
      Apeiron.GrowableIdentity.wordHomeInst.address
        (Apeiron.GrowableRefinement.rebuiltView state current).home sigma held.key = ok address := by
    simpa [Apeiron.GrowableRefinement.rebuiltView] using haddress
  have hpublished : StoredLiveTagAddress
      { Apeiron.GrowableRefinement.rebuiltView state current with
        shards := (Apeiron.GrowableRefinement.rebuiltView state current).shards.set shard published } :=
    stored_live_tag_address_publish_spill (Apeiron.GrowableRefinement.rebuiltView state current)
      hstate shard index region published hregionView sigma { key := held.key } value address
      haddressView hshard hpublish
  exact stored_live_tag_address_of_home_shards_eq _ _ hpublished rfl rfl

/-- One certified generated flat relocation step preserves the reserved tag/address invariant:
it changes no home or spill field of the rebuilt view. -/
theorem stored_relocate_entry_flat_keeps_reserved_tags
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (plan : table.RebuildPlan)
    (current : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64)
    (entryAt index : Usize)
    (entry : table.RebuildPlanEntry)
    (sigma : cell.Sigma) (held : Apeiron.GrowableIdentity.StoredI64) (value : I64)
    (nextLocator : table.TableLocator)
    (hstate : StoredLiveTagAddress (Apeiron.GrowableRefinement.rebuiltView state current))
    (hentry : plan.entries.val[entryAt.val]? = some entry)
    (hsource : Apeiron.GrowableRelocation.sourcePayload state entry.source =
      some (sigma, held, value))
    (hflat : entry.flat = some index)
    (old : table.FlatSlot Apeiron.GrowableIdentity.StoredI64 I64)
    (hdst : current.resident.slots.val[index.val]? = some old)
    (hretarget : table.TableLocator.retarget_held core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
      Apeiron.GrowableIdentity.storedKeyRep state.locator state.home sigma { key := held.key }
      (Apeiron.GrowableRelocation.sourceLocation entry.source)
      (Apeiron.GrowableRelocation.destinationLocation (.flat index)) = ok nextLocator) :
    word_stored_relocate_entry state plan current entryAt ⦃ result =>
      ReservedStoredLiveTagAddress result.1 result.2 ⦄ := by
  rw [Apeiron.GrowableRepresentation.word_stored_relocate_entry_flat_run state plan current
    entryAt index entry sigma held value nextLocator hentry hsource hflat old hdst hretarget,
    WP.spec_ok]
  exact stored_live_tag_address_of_home_shards_eq _ _ hstate rfl rfl

/-- Traversal induction for the reserved tag/address invariant: the generated stored-carrier
destination traversal preserves the invariant when every visited step preserves it.  The initial
state is the landed materialized-vacancy invariant; the per-step premise at every visited prefix
state is the explicit remaining certificate obligation. -/
theorem word_stored_relocate_all_keeps_reserved_tags
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (plan : table.RebuildPlan)
    (reserved : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64)
    (hinitial : ReservedStoredLiveTagAddress state reserved)
    (hstep : ∀ (s : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
      (current : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64) (entryAt : Usize),
      entryAt.val < plan.entries.length →
      ReservedStoredLiveTagAddress s current →
      word_stored_relocate_entry s plan current entryAt ⦃ result =>
        ReservedStoredLiveTagAddress result.1 result.2 ⦄) :
    word_stored_relocate_all state plan reserved ⦃ result =>
      ReservedStoredLiveTagAddress result.1 result.2 ⦄ :=
  Apeiron.GrowableRepresentation.word_stored_relocate_all_preserves_reserved state plan reserved
    ReservedStoredLiveTagAddress hinitial hstep

/-- The traversal starts from the landed initial reserved invariant of a prepared-resource
certificate; the per-step premise remains explicit. -/
theorem word_stored_relocate_all_keeps_reserved_tags_from_prepared
    (certificate : Apeiron.GrowableResources.PreparedResourceClauses
      Apeiron.GrowableIdentity.StoredI64)
    (plan : table.RebuildPlan)
    (hstep : ∀ (s : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
      (current : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64) (entryAt : Usize),
      entryAt.val < plan.entries.length →
      ReservedStoredLiveTagAddress s current →
      word_stored_relocate_entry s plan current entryAt ⦃ result =>
        ReservedStoredLiveTagAddress result.1 result.2 ⦄) :
    word_stored_relocate_all certificate.state plan certificate.storage ⦃ result =>
      ReservedStoredLiveTagAddress result.1 result.2 ⦄ :=
  word_stored_relocate_all_keeps_reserved_tags certificate.state plan certificate.storage
    (prepared_stored_live_tags_initial certificate) hstep

-- Local fail-closed cone check for the traversal block. The shared audit file is outside this
-- task's ownership; the integration owner registers these names in AxiomAudit.lean.
open Lean Elab Command in
run_cmd do
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  let external : Array Lean.Name := #[``Apeiron.GrowableRepresentation.word_stored_relocate_entry_spill_run,
    ``Apeiron.GrowableRepresentation.word_stored_relocate_entry_flat_run,
    ``Apeiron.GrowableRepresentation.word_stored_relocate_entry_spill_publishes_tag_address,
    ``Apeiron.GrowableRepresentation.word_stored_relocate_entry_spill_entryAt,
    ``Apeiron.GrowableRepresentation.reservedRepR_singleDestination_of_relocated,
    ``Apeiron.GrowableRepresentation.repR_singleSource_logical,
    ``Apeiron.GrowableRepresentation.word_stored_relocate_all_preserves_reserved]
  let localNames : Array Lean.Name := #[`ReservedStoredLiveTagAddress,
    `stored_relocate_entry_spill_keeps_reserved_tags, `stored_relocate_entry_flat_keeps_reserved_tags,
    `word_stored_relocate_all_keeps_reserved_tags,
    `word_stored_relocate_all_keeps_reserved_tags_from_prepared]
  for decl in external ++ localNames.map (Lean.Name.append `Apeiron.GrowableReadLoop) do
    let axioms ← liftTermElabM <| Lean.collectAxioms decl
    let bad := axioms.filter (fun ax => !allowed.contains ax)
    unless bad.isEmpty do
      throwError m!"'{decl}' depends on axioms outside the closed cone: {bad.toList}"
    logInfo m!"'{decl}' depends on axioms: {axioms.toList}"

end Apeiron.GrowableReadLoop
