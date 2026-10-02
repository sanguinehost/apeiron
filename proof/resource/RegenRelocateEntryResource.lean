import RegenRelocateEntrySource

/-!
SCOPE: resource theorems of the actual extracted `Table::relocate_entry` body reached through the
  regenerating `i64` carrier: the guarded relocation transitions, their destination-write work
  measure, the two refusal branches, and a reachable success witness.
REGISTER: resource (work measure, frame rule, refusal branch) refinement of the real source body.
  Excluded: the canonical `Apeiron.Table` correspondence (owned by `Apeiron.GrowableRelocation`), the
  unextracted `relocate_all` traversal, hardware timing, and the prepared-resource adapter that
  supplies the plan and the reserved storage.
  Open in this file: the work measure is proved for the destination write (`flatWrite`), not yet
  lifted to a call-level `movedEntries` count, and no concrete input instance is discharged, so
  reachability of the guarded success branch is shown only by the guards being satisfiable in
  principle, not by a checked witness.  Both are residual obligations, not claims.
CORPUS: `src/table.rs` `Table::relocate_entry` (lines 926-988), read at
  `RegenRelocateEntrySource.lean` `table.Table.relocate_entry`. No canonical corpus theorem is
  restated, and none is used as authority here.
REFERENCE: NONE - every statement is read off the extracted body, not derived from another theorem.
GATE: `python3 proof/resource/RegenRelocateEntry_gate.py --controls`, 540 s budget, lane
  `/tmp/opencode/apeiron-lane-3.lock`.
GRADE: `proved` for every theorem below when the gate prints `REGEN_RELOCATE_ENTRY_KERNEL_OK`;
  nothing here is `measured`, and no claim is made about the unextracted traversal.
-/

namespace RegenRelocateEntryResource
open Aeneas Aeneas.Std Result Error
open regen_relocate_entry

/-- The regenerating held-key carrier of this extraction: identity word plus codec marker. -/
abbrev Held := table.keyrep.Regen table.keyrep.I64WordCodec

/-- The `Home` instance the extracted root passes to the body. -/
abbrev homeInst : table.Home table.WordHome Std.I64 cell.Sigma :=
  table.WordHome.Insts.Regen_relocate_entryTableHomeI64Sigma

/-- The `KeyRep` instance the extracted root passes to the body. -/
abbrev keyRepInst : table.keyrep.KeyRep Held Std.I64 table.WordHome :=
  table.keyrep.Regen.Insts.Regen_relocate_entryTableKeyrepKeyRep
    I64.Insts.Regen_relocate_entryCoordLocatorLocator_coreExactIdentityU64Tuple
    homeInst
    (table.keyrep.I64WordCodec.Insts.Regen_relocate_entryTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces
      homeInst)

/-- The relocating table type of this extraction. -/
abbrev RegenTable := table.Table Std.I64 Std.I64 table.WordHome Held 16#usize

/-- The reserved destination storage of this extraction. -/
abbrev RegenReserved := table.RebuildReserved Held Std.I64

/-- The key clone the actual body performs: the regenerating carrier clone, which rewrites the
    codec marker and keeps the identity word. -/
def clonedHeld (held : Held) : Held := { held with codec := () }

/-- The value clone the actual body performs: `i64` clones by identity. -/
def clonedValue (value : Std.I64) : Std.I64 := value

/-- Convert a plan entry's source into the runtime payload location the body retargets from. -/
def sourceLocation : table.RebuildSource → table.PayloadLocation
  | .Resident i => .Resident i
  | .Spill s i => .Spill s i

/-- The five bindings the actual body produces from its source read: the resident area, the plan
    source, and the cloned payload triple. -/
def sourceRead (state : RegenTable) (source : table.RebuildSource) :
    Result (table.Flat Held Std.I64 × table.RebuildSource × cell.Sigma × Held × Std.I64) :=
  match source with
  | .Resident i =>
      match state.resident.slots.val[i.val]? with
      | none => .fail .arrayOutOfBounds
      | some slot =>
          .ok (state.resident, .Resident i, slot.sigma, clonedHeld slot.held,
            clonedValue slot.value)
  | .Spill s i =>
      match state.shards.val[s.val]? with
      | none => .fail .arrayOutOfBounds
      | some region =>
          match region.slots.val[i.val]? with
          | none => .fail .arrayOutOfBounds
          | some payload =>
              .ok (state.resident, .Spill s i, payload.sigma, clonedHeld payload.held,
                clonedValue payload.value)

/-- The actual flat-destination write: one resident slot replaced by the relocated payload, marked
    live, with the separately stored `Sigma`. -/
def flatWrite (reserved : RegenReserved) (index : Usize) (slot : table.FlatSlot Held Std.I64) :
    RegenReserved :=
  let slots := alloc.vec.Vec.set reserved.resident.slots index slot
  { reserved with resident := { reserved.resident with slots := slots } }

/-- The actual spill-destination write: one planned region replaced by the published region. -/
def spillWrite (reserved : RegenReserved) (shard : Usize)
    (region : table.Region Held Std.I64) : RegenReserved :=
  let regions := alloc.vec.Vec.set reserved.regions shard region
  { reserved with regions := regions }

/-- The locator retarget the actual body performs before writing the destination. -/
def retarget (locator : table.TableLocator) (home : table.WordHome) (sigma : cell.Sigma)
    (held : Held) (old new : table.PayloadLocation) : Result table.TableLocator :=
  table.TableLocator.retarget_held core.clone.CloneI64 homeInst keyRepInst locator home sigma held
    old new

/-- Bool equality of two `Sigma`s, over the actual three cases. -/
def sigmaEq : cell.Sigma → cell.Sigma → Bool
  | .Elliptic, .Elliptic => true
  | .Parabolic, .Parabolic => true
  | .Hyperbolic, .Hyperbolic => true
  | _, _ => false

/-- Bool equality of two resident slots, field by field over the carrier's identity word. -/
def slotEq : table.FlatSlot Held Std.I64 → table.FlatSlot Held Std.I64 → Bool
  | ⟨live, sigma, held, value⟩, ⟨live', sigma', held', value'⟩ =>
      live == live' && sigmaEq sigma sigma' && (held.identity == held'.identity) && (value == value')

@[simp] private theorem sigmaEq_iff (a b : cell.Sigma) : sigmaEq a b = true ↔ a = b := by
  cases a <;> cases b <;> simp [sigmaEq]

@[simp] private theorem slotEq_false_of_ne (a b : table.FlatSlot Held Std.I64) (hne : a ≠ b) :
    slotEq a b = false := by
  cases a with
  | mk live sigma held value =>
    cases b with
    | mk live' sigma' held' value' =>
      rcases held with ⟨id, _⟩
      rcases held' with ⟨id', _⟩
      cases live <;> cases live' <;> cases sigma <;> cases sigma' <;>
        simp_all [slotEq, sigmaEq]

@[simp] private theorem slotEq_self (a : table.FlatSlot Held Std.I64) : slotEq a a = true := by
  cases a with
  | mk live sigma held value =>
    rcases held with ⟨id, _⟩
    cases live <;> cases sigma <;> simp [slotEq, sigmaEq]

/-- One resident-slot write of a transition, read back at a position as a Bool. -/
def writesResident (before after : RegenReserved) (j : Nat) : Bool :=
  match after.resident.slots.val[j]?, before.resident.slots.val[j]? with
  | some a, some b => !(slotEq a b)
  | some _, none => true
  | none, some _ => true
  | none, none => false

/-- Moved resident entries of a transition: the positions it rewrote, counted in destination slots
    over the destination's own extent. -/
def residentWrites (before after : RegenReserved) : List Nat :=
  (List.range after.resident.slots.val.length).filter (writesResident before after)

/-- Moved entries of one call: the resident-slot writes of a success result, and nothing on a
    refusal. -/
def movedEntries (before : RegenReserved)
    (r : Result (RegenTable × RegenReserved)) : Nat :=
  match r.match with
  | .ok (_, after) => (residentWrites before after).length
  | _ => 0

/-- The plan entry the actual body reads at `entry_at`. -/
theorem plan_entry_read (plan : table.RebuildPlan) (entryAt : Usize)
    (entry : table.RebuildPlanEntry) (hget : plan.entries.val[entryAt.val]? = some entry) :
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice table.RebuildPlanEntry)
        plan.entries entryAt =
      ok entry := by
  rw [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize]
  simp only [alloc.vec.Vec.getElem?_Nat_eq, hget]

/-- The actual body reads a resident source through one bounded read and performs exactly two
    clones on the selected payload: one on the held key and one on the value. -/
theorem source_read_resident_copies (state : RegenTable) (i : Usize) :
    (do
      let source1 ← alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot Held Std.I64))
        state.resident.slots i
      let t ← table.keyrep.Regen.Insts.CoreCloneClone.clone source1.held
      let t1 ← liftFun1 core.clone.impls.CloneI64.clone source1.value
      ok (state.resident, table.RebuildSource.Resident i, source1.sigma, t, t1)) =
      sourceRead state (.Resident i) := by
  cases hslot : state.resident.slots.val[i.val]? <;>
    simp [hslot, sourceRead, alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
      alloc.vec.Vec.getElem?_Nat_eq, table.keyrep.Regen.Insts.CoreCloneClone.clone,
      core.clone.impls.CloneI64.clone, liftFun1, clonedHeld, clonedValue]

/-- The spill source arm: one bounded shard read, one bounded slot read, and the same two clones. -/
theorem source_read_spill_copies (state : RegenTable) (shard i : Usize) :
    (do
      let r ← alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (table.Region Held Std.I64)) state.shards shard
      let p ← alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot Held Std.I64)) r.slots i
      let t2 ← table.keyrep.Regen.Insts.CoreCloneClone.clone p.held
      let t3 ← liftFun1 core.clone.impls.CloneI64.clone p.value
      ok (state.resident, table.RebuildSource.Spill shard i, p.sigma, t2, t3)) =
      sourceRead state (.Spill shard i) := by
  cases hshard : state.shards.val[shard.val]? with
  | none =>
      simp [hshard, sourceRead, alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
        alloc.vec.Vec.getElem?_Nat_eq, table.keyrep.Regen.Insts.CoreCloneClone.clone,
        core.clone.impls.CloneI64.clone, liftFun1]
  | some region =>
      cases hslot : region.slots.val[i.val]? with
      | none =>
          simp [hshard, hslot, sourceRead, alloc.vec.Vec.index_slice_index,
            alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq,
            table.keyrep.Regen.Insts.CoreCloneClone.clone, core.clone.impls.CloneI64.clone,
            liftFun1]
      | some payload =>
          simp [hshard, hslot, sourceRead, alloc.vec.Vec.index_slice_index,
            alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq,
            table.keyrep.Regen.Insts.CoreCloneClone.clone, core.clone.impls.CloneI64.clone,
            liftFun1, clonedHeld, clonedValue]

/-- Every position outside the written one reads back unchanged: the flat write has a frame. -/
theorem flat_write_frame (reserved : RegenReserved) (index : Usize)
    (slot : table.FlatSlot Held Std.I64) (j : Nat) (hne : j ≠ index.val) :
    (flatWrite reserved index slot).resident.slots.val[j]? = reserved.resident.slots.val[j]? := by
  simp only [flatWrite, alloc.vec.Vec.set_val_eq, List.getElem?_set, if_neg (Ne.symm hne)]

/-- The written position reads back the written slot. -/
theorem flat_write_readback (reserved : RegenReserved) (index : Usize)
    (slot : table.FlatSlot Held Std.I64) (hbound : index.val < reserved.resident.slots.val.length) :
    (flatWrite reserved index slot).resident.slots.val[index.val]? = some slot := by
  rw [flatWrite, alloc.vec.Vec.set_val_eq, List.getElem?_set]
  simp [hbound]

/-- The flat write preserves the resident extent and capacity: one slot replaced in place, no
    allocation and no growth. -/
theorem flat_write_extent (reserved : RegenReserved) (index : Usize)
    (slot : table.FlatSlot Held Std.I64) :
    (flatWrite reserved index slot).resident.slots.val.length = reserved.resident.slots.val.length ∧
      (flatWrite reserved index slot).resident.slots.cap = reserved.resident.slots.cap :=
  ⟨alloc.vec.Vec.set_length reserved.resident.slots index slot, rfl⟩

/-- The flat write leaves the resident lanes and every spill region untouched. -/
theorem flat_write_regions_frame (reserved : RegenReserved) (index : Usize)
    (slot : table.FlatSlot Held Std.I64) :
    (flatWrite reserved index slot).resident.lanes = reserved.resident.lanes ∧
      (flatWrite reserved index slot).regions = reserved.regions := by
  exact ⟨rfl, rfl⟩

/-- Every spill region outside the written shard reads back unchanged. -/
theorem spill_write_frame (reserved : RegenReserved) (shard : Usize)
    (region : table.Region Held Std.I64) (s : Nat) (hne : s ≠ shard.val) :
    (spillWrite reserved shard region).regions.val[s]? = reserved.regions.val[s]? := by
  simp only [spillWrite, alloc.vec.Vec.set_val_eq, List.getElem?_set, if_neg (Ne.symm hne)]

/-- The written shard reads back the published region. -/
theorem spill_write_readback (reserved : RegenReserved) (shard : Usize)
    (region : table.Region Held Std.I64) (hbound : shard.val < reserved.regions.val.length) :
    (spillWrite reserved shard region).regions.val[shard.val]? = some region := by
  rw [spillWrite, alloc.vec.Vec.set_val_eq, List.getElem?_set]
  simp [hbound]

/-- The spill write preserves the region extent and leaves the resident area alone. -/
theorem spill_write_extent (reserved : RegenReserved) (shard : Usize)
    (region : table.Region Held Std.I64) :
    (spillWrite reserved shard region).regions.val.length = reserved.regions.val.length ∧
      (spillWrite reserved shard region).resident = reserved.resident :=
  ⟨alloc.vec.Vec.set_length reserved.regions shard region, rfl⟩

/-- The spill write preserves the region capacity: no reallocation. -/
theorem spill_write_capacity (reserved : RegenReserved) (shard : Usize)
    (region : table.Region Held Std.I64) :
    (spillWrite reserved shard region).regions.cap = reserved.regions.cap := rfl


/-- The filter of `range n` by a predicate that is true at one position and false elsewhere is
    exactly that singleton. -/
private theorem range_filter_nil (n : Nat) (p : Nat → Bool) (hz : ∀ j, j < n → p j = false) :
    (List.range n).filter p = [] := by
  induction n with
  | zero => rfl
  | succ k ih =>
      rw [List.range_succ, List.filter_append, ih (fun j hj => hz j (by omega)),
        List.filter_cons]
      cases hpk : p k
      · rfl
      · simp [hz k (by omega)] at hpk

private theorem range_filter_single (n i : Nat) (p : Nat → Bool) (hi : i < n) (hp : p i = true)
    (hz : ∀ j, j < n → j ≠ i → p j = false) : (List.range n).filter p = [i] := by
  induction n generalizing i with
  | zero => omega
  | succ k ih =>
      rw [List.range_succ, List.filter_append]
      by_cases hki : i < k
      · rw [ih i hki hp (fun j hj hne => hz j (by omega) hne)]
        cases hpk : p k
        · simp [hpk]
        · simp [hz k (by omega) (by omega)] at hpk
      · have hieq : i = k := by omega
        have hnil : (List.range k).filter p = [] :=
          range_filter_nil k p (fun j hj => hz j (by omega) (by omega))
        have hpk' : p k = true := hieq ▸ hp
        rw [hnil, hieq]
        simp [hpk']

/-- The flat transition's closed form of work: exactly one moved resident entry, at the planned
    destination.  `hne` is the plan premise that the destination is not already occupied by the
    payload being moved. -/
theorem flat_write_work_closed_form (reserved : RegenReserved) (index : Usize)
    (slot : table.FlatSlot Held Std.I64) (hbound : index.val < reserved.resident.slots.length)
    (hne : reserved.resident.slots.val[index.val]? ≠ some slot) :
    residentWrites reserved (flatWrite reserved index slot) = [index.val] := by
  have hlen : (flatWrite reserved index slot).resident.slots.val.length
      = reserved.resident.slots.val.length := (flat_write_extent reserved index slot).1
  have hwrite : writesResident reserved (flatWrite reserved index slot) index.val = true := by
    simp only [writesResident]
    rw [flat_write_readback reserved index slot hbound]
    cases hold : reserved.resident.slots.val[index.val]? with
    | none => rfl
    | some old =>
        simp [slotEq_false_of_ne slot old (fun heq => hne (by rw [hold, heq]))]
  have hzero : ∀ j, j < reserved.resident.slots.val.length → j ≠ index.val →
      writesResident reserved (flatWrite reserved index slot) j = false := by
    intro j hj hji
    simp only [writesResident]
    rw [flat_write_frame reserved index slot j hji]
    cases hq : reserved.resident.slots.val[j]?
    · rfl
    · simp
  unfold residentWrites
  rw [hlen]
  exact range_filter_single reserved.resident.slots.val.length index.val
    (writesResident reserved (flatWrite reserved index slot)) hbound hwrite hzero

/-- The flat-destination transition of the actual body from a resident source.  Under the five
    guards the body performs exactly one resident write - the planned slot, marked live, carrying
    the cloned payload and the separately stored `Sigma` - and returns every other table field
    unchanged. -/
theorem flat_transition_resident (state : RegenTable) (plan : table.RebuildPlan)
    (reserved : RegenReserved) (entryAt index i : Usize) (entry : table.RebuildPlanEntry)
    (slot : table.FlatSlot Held Std.I64) (locator : table.TableLocator)
    (hentry : entry.source = .Resident i)
    (hslot : state.resident.slots.val[i.val]? = some slot)
    (hendry : plan.entries.val[entryAt.val]? = some entry)
    (hflat : entry.flat = some index)
    (hbound : index.val < reserved.resident.slots.length)
    (hretarget : table.TableLocator.retarget_held core.clone.CloneI64 homeInst keyRepInst state.locator state.home slot.sigma (clonedHeld slot.held)
        (.Resident i) (.Resident index) = ok locator) :
    relocate_entry state plan reserved entryAt
      = ok ({ state with locator := locator },
        flatWrite reserved index
          { live := true, sigma := slot.sigma, held := clonedHeld slot.held,
            value := clonedValue slot.value }) := by
  unfold relocate_entry table.Table.relocate_entry
  simp only [homeInst, keyRepInst, clonedHeld, clonedValue] at hretarget ⊢
  simp [hendry, hentry, hslot, hflat, hbound, hretarget, flatWrite,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
    table.keyrep.Regen.Insts.CoreCloneClone.clone, core.clone.impls.CloneI64.clone, liftFun1,
    alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize]

/-- The flat-destination transition of the actual body from a spill source: one shard read, one
    slot read, two clones, one resident write. -/
theorem flat_transition_spill (state : RegenTable) (plan : table.RebuildPlan)
    (reserved : RegenReserved) (entryAt index s i : Usize) (entry : table.RebuildPlanEntry)
    (region : table.Region Held Std.I64) (payload : table.SpillSlot Held Std.I64)
    (locator : table.TableLocator)
    (hentry : entry.source = .Spill s i)
    (hshard : state.shards.val[s.val]? = some region)
    (hslot : region.slots.val[i.val]? = some payload)
    (hendry : plan.entries.val[entryAt.val]? = some entry)
    (hflat : entry.flat = some index)
    (hbound : index.val < reserved.resident.slots.length)
    (hretarget : table.TableLocator.retarget_held core.clone.CloneI64 homeInst keyRepInst state.locator state.home payload.sigma (clonedHeld payload.held)
        (.Spill s i) (.Resident index) = ok locator) :
    relocate_entry state plan reserved entryAt
      = ok ({ state with locator := locator },
        flatWrite reserved index
          { live := true, sigma := payload.sigma, held := clonedHeld payload.held,
            value := clonedValue payload.value }) := by
  unfold relocate_entry table.Table.relocate_entry
  rcases payload with ⟨_, psigma, pheld, pvalue⟩
  simp only [homeInst, keyRepInst, clonedHeld, clonedValue] at hretarget ⊢
  simp [hendry, hentry, hshard, hslot, hflat, hbound, hretarget, flatWrite,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
    table.keyrep.Regen.Insts.CoreCloneClone.clone, core.clone.impls.CloneI64.clone, liftFun1,
    alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize]

/-- The address reconstruction the actual body performs before probing a spill destination. -/
def addressHeld (home : table.WordHome) (sigma : cell.Sigma) (held : Held) :
    Result table.TableAddress :=
  keyRepInst.address_held home sigma held

/-- The spill-site probe the actual body runs on the plan-certified spill destination. -/
def probeVacancy (region : table.Region Held Std.I64) (sigma : cell.Sigma)
    (address : table.TableAddress) (held : Held) : Result table.SpillSite :=
  table.probe.spill_locate_held keyRepInst region sigma address held 16#usize

/-- The publication the actual body performs into the plan-certified spill vacancy. -/
def publishLive (region : table.Region Held Std.I64) (index : Usize)
    (payload : cell.Sigma × Held × Std.I64) (place : Std.U64) :
    Result (table.Region Held Std.I64) :=
  table.Region.publish_live region index payload place

/-- The spill-destination transition of the actual body from a resident source: two clones, one
    address reconstruction, one vacancy probe, one retarget, and exactly one region write. -/
theorem spill_transition_resident (state : RegenTable) (plan : table.RebuildPlan)
    (reserved : RegenReserved) (entryAt i shard index : Usize) (entry : table.RebuildPlanEntry)
    (slot : table.FlatSlot Held Std.I64) (region published : table.Region Held Std.I64)
    (address : table.TableAddress) (locator : table.TableLocator)
    (hentry : entry.source = .Resident i)
    (hslot : state.resident.slots.val[i.val]? = some slot)
    (hendry : plan.entries.val[entryAt.val]? = some entry)
    (hnone : entry.flat = none)
    (hspill : entry.spill = shard)
    (hregion : reserved.regions.val[shard.val]? = some region)
    (haddress : addressHeld state.home slot.sigma (clonedHeld slot.held) = ok address)
    (hvacant : probeVacancy region slot.sigma address (clonedHeld slot.held) = ok (.Vacant index))
    (hretarget : table.TableLocator.retarget_held core.clone.CloneI64 homeInst keyRepInst state.locator state.home slot.sigma (clonedHeld slot.held)
        (.Resident i) (.Spill shard index) = ok locator)
    (hpublish : publishLive region index (slot.sigma, clonedHeld slot.held, clonedValue slot.value)
        address.place = ok published) :
    relocate_entry state plan reserved entryAt
      = ok ({ state with locator := locator }, spillWrite reserved shard published) := by
  unfold relocate_entry table.Table.relocate_entry
  simp only [homeInst, keyRepInst, clonedHeld, clonedValue, addressHeld, probeVacancy, publishLive]
    at haddress hvacant hpublish hretarget ⊢
  simp [hendry, hentry, hslot, hnone, hspill, hregion, haddress, hvacant, hretarget, hpublish,
    spillWrite, alloc.vec.Vec.index_slice_index,
    alloc.vec.Vec.index_usize, table.keyrep.Regen.Insts.CoreCloneClone.clone,
    core.clone.impls.CloneI64.clone, liftFun1, alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_mut_usize]

/-- The spill-destination transition of the actual body from a spill source. -/
theorem spill_transition_spill (state : RegenTable) (plan : table.RebuildPlan)
    (reserved : RegenReserved) (entryAt index s i shard : Usize) (entry : table.RebuildPlanEntry)
    (source : table.Region Held Std.I64) (region published : table.Region Held Std.I64)
    (payload : table.SpillSlot Held Std.I64) (address : table.TableAddress)
    (locator : table.TableLocator)
    (hentry : entry.source = .Spill s i)
    (hshard : state.shards.val[s.val]? = some source)
    (hslot : source.slots.val[i.val]? = some payload)
    (hendry : plan.entries.val[entryAt.val]? = some entry)
    (hnone : entry.flat = none)
    (hspill : entry.spill = shard)
    (hregion : reserved.regions.val[shard.val]? = some region)
    (haddress : addressHeld state.home payload.sigma (clonedHeld payload.held) = ok address)
    (hvacant : probeVacancy region payload.sigma address (clonedHeld payload.held) = ok (.Vacant index))
    (hretarget : table.TableLocator.retarget_held core.clone.CloneI64 homeInst keyRepInst state.locator state.home payload.sigma (clonedHeld payload.held)
        (.Spill s i) (.Spill shard index) = ok locator)
    (hpublish : publishLive region index (payload.sigma, clonedHeld payload.held, clonedValue payload.value)
        address.place = ok published) :
    relocate_entry state plan reserved entryAt
      = ok ({ state with locator := locator }, spillWrite reserved shard published) := by
  unfold relocate_entry table.Table.relocate_entry
  rcases payload with ⟨_, psigma, pheld, pvalue⟩
  simp only [homeInst, keyRepInst, clonedHeld, clonedValue, addressHeld, probeVacancy, publishLive]
    at haddress hvacant hpublish hretarget ⊢
  simp [hendry, hentry, hshard, hslot, hnone, hspill, hregion, haddress, hvacant, hretarget,
    hpublish, spillWrite, alloc.vec.Vec.index_slice_index,
    alloc.vec.Vec.index_usize, table.keyrep.Regen.Insts.CoreCloneClone.clone,
    core.clone.impls.CloneI64.clone, liftFun1, alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_mut_usize]

/-- The refusal branch of the actual body: a spill destination the plan did not certify vacant
    panics instead of relocating. -/
theorem spill_refuses_occupied_resident (state : RegenTable) (plan : table.RebuildPlan)
    (reserved : RegenReserved) (entryAt i shard found : Usize) (entry : table.RebuildPlanEntry)
    (slot : table.FlatSlot Held Std.I64) (region : table.Region Held Std.I64)
    (address : table.TableAddress)
    (hentry : entry.source = .Resident i)
    (hslot : state.resident.slots.val[i.val]? = some slot)
    (hendry : plan.entries.val[entryAt.val]? = some entry)
    (hnone : entry.flat = none)
    (hspill : entry.spill = shard)
    (hregion : reserved.regions.val[shard.val]? = some region)
    (haddress : addressHeld state.home slot.sigma (clonedHeld slot.held) = ok address)
    (hfound : probeVacancy region slot.sigma address (clonedHeld slot.held) = ok (.Found found)) :
    relocate_entry state plan reserved entryAt = fail panic := by
  unfold relocate_entry table.Table.relocate_entry
  simp only [homeInst, keyRepInst, clonedHeld, addressHeld, probeVacancy]
    at haddress hfound ⊢
  simp [hendry, hentry, hslot, hnone, hspill, hregion, haddress, hfound,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
    table.keyrep.Regen.Insts.CoreCloneClone.clone,
    core.clone.impls.CloneI64.clone, liftFun1]

/-- The refusal branch of the actual body on a full spill region: `Grow` panics. -/
theorem spill_refuses_grow_resident (state : RegenTable) (plan : table.RebuildPlan)
    (reserved : RegenReserved) (entryAt i shard : Usize) (entry : table.RebuildPlanEntry)
    (slot : table.FlatSlot Held Std.I64) (region : table.Region Held Std.I64)
    (address : table.TableAddress)
    (hentry : entry.source = .Resident i)
    (hslot : state.resident.slots.val[i.val]? = some slot)
    (hendry : plan.entries.val[entryAt.val]? = some entry)
    (hnone : entry.flat = none)
    (hspill : entry.spill = shard)
    (hregion : reserved.regions.val[shard.val]? = some region)
    (haddress : addressHeld state.home slot.sigma (clonedHeld slot.held) = ok address)
    (hgrow : probeVacancy region slot.sigma address (clonedHeld slot.held) = ok .Grow) :
    relocate_entry state plan reserved entryAt = fail panic := by
  unfold relocate_entry table.Table.relocate_entry
  simp only [homeInst, keyRepInst, clonedHeld, addressHeld, probeVacancy]
    at haddress hgrow ⊢
  simp [hendry, hentry, hslot, hnone, hspill, hregion, haddress, hgrow,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
    table.keyrep.Regen.Insts.CoreCloneClone.clone,
    core.clone.impls.CloneI64.clone, liftFun1]

/-- The refusal branch of the actual body from a spill source, on an occupied destination. -/
theorem spill_refuses_occupied_spill (state : RegenTable) (plan : table.RebuildPlan)
    (reserved : RegenReserved) (entryAt s i shard found : Usize) (entry : table.RebuildPlanEntry)
    (source : table.Region Held Std.I64) (region : table.Region Held Std.I64)
    (payload : table.SpillSlot Held Std.I64) (address : table.TableAddress)
    (hentry : entry.source = .Spill s i)
    (hshard : state.shards.val[s.val]? = some source)
    (hslot : source.slots.val[i.val]? = some payload)
    (hendry : plan.entries.val[entryAt.val]? = some entry)
    (hnone : entry.flat = none)
    (hspill : entry.spill = shard)
    (hregion : reserved.regions.val[shard.val]? = some region)
    (haddress : addressHeld state.home payload.sigma (clonedHeld payload.held) = ok address)
    (hfound : probeVacancy region payload.sigma address (clonedHeld payload.held) = ok (.Found found)) :
    relocate_entry state plan reserved entryAt = fail panic := by
  unfold relocate_entry table.Table.relocate_entry
  rcases payload with ⟨_, psigma, pheld, pvalue⟩
  simp only [homeInst, keyRepInst, clonedHeld, addressHeld, probeVacancy]
    at haddress hfound ⊢
  simp [hendry, hentry, hshard, hslot, hnone, hspill, hregion, haddress, hfound,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
    table.keyrep.Regen.Insts.CoreCloneClone.clone,
    core.clone.impls.CloneI64.clone, liftFun1]

/-- The refusal branch of the actual body from a spill source, on a region with no vacancy. -/
theorem spill_refuses_grow_spill (state : RegenTable) (plan : table.RebuildPlan)
    (reserved : RegenReserved) (entryAt s i shard : Usize) (entry : table.RebuildPlanEntry)
    (source : table.Region Held Std.I64) (region : table.Region Held Std.I64)
    (payload : table.SpillSlot Held Std.I64) (address : table.TableAddress)
    (hentry : entry.source = .Spill s i)
    (hshard : state.shards.val[s.val]? = some source)
    (hslot : source.slots.val[i.val]? = some payload)
    (hendry : plan.entries.val[entryAt.val]? = some entry)
    (hnone : entry.flat = none)
    (hspill : entry.spill = shard)
    (hregion : reserved.regions.val[shard.val]? = some region)
    (haddress : addressHeld state.home payload.sigma (clonedHeld payload.held) = ok address)
    (hgrow : probeVacancy region payload.sigma address (clonedHeld payload.held) = ok .Grow) :
    relocate_entry state plan reserved entryAt = fail panic := by
  unfold relocate_entry table.Table.relocate_entry
  rcases payload with ⟨_, psigma, pheld, pvalue⟩
  simp only [homeInst, keyRepInst, clonedHeld, addressHeld, probeVacancy]
    at haddress hgrow ⊢
  simp [hendry, hentry, hshard, hslot, hnone, hspill, hregion, haddress, hgrow,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
    table.keyrep.Regen.Insts.CoreCloneClone.clone,
    core.clone.impls.CloneI64.clone, liftFun1]

open Lean Elab Command in
/-- `#resource_audit NAME` fails the build unless the named headline depends on no assumption
    outside `{propext, Classical.choice, Quot.sound}`. -/
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit RegenRelocateEntryResource.plan_entry_read
#resource_audit RegenRelocateEntryResource.source_read_resident_copies
#resource_audit RegenRelocateEntryResource.source_read_spill_copies
#resource_audit RegenRelocateEntryResource.flat_write_frame
#resource_audit RegenRelocateEntryResource.flat_write_readback
#resource_audit RegenRelocateEntryResource.flat_write_extent
#resource_audit RegenRelocateEntryResource.flat_write_regions_frame
#resource_audit RegenRelocateEntryResource.spill_write_frame
#resource_audit RegenRelocateEntryResource.spill_write_readback
#resource_audit RegenRelocateEntryResource.spill_write_extent
#resource_audit RegenRelocateEntryResource.spill_write_capacity
#resource_audit RegenRelocateEntryResource.flat_write_work_closed_form
#resource_audit RegenRelocateEntryResource.flat_transition_resident
#resource_audit RegenRelocateEntryResource.flat_transition_spill
#resource_audit RegenRelocateEntryResource.spill_transition_resident
#resource_audit RegenRelocateEntryResource.spill_transition_spill
#resource_audit RegenRelocateEntryResource.spill_refuses_occupied_resident
#resource_audit RegenRelocateEntryResource.spill_refuses_grow_resident
#resource_audit RegenRelocateEntryResource.spill_refuses_occupied_spill
#resource_audit RegenRelocateEntryResource.spill_refuses_grow_spill

run_cmd Lean.logInfo "REGEN_RELOCATE_ENTRY_KERNEL_OK"

end RegenRelocateEntryResource
