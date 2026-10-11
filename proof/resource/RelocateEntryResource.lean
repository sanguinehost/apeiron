import RelocateEntrySource
import Aeneas

/- SCOPE: the real `Table::relocate_entry` body reached through the extracted root
   `word_stored_relocate_entry`: its two guarded transitions, its closed-form work measure, its
   exit branches, and one reachable success. Extracted through this gate's own narrow forwarding
   `#[path]` wrapper over `src/table.rs`; the shared `proof/extract` unit is not used and does not
   compile.
   REGISTER: resource and source refinement of one per-entry relocation. Excluded: the
   flat/spill canonical `HMap` correspondence, the rebuild plan's own checked-sizing claim, the
   `relocate_all` traversal bound, and any hardware timing.
   CORPUS: `src/table.rs` `Table::relocate_entry` (lines 926-988, as the extraction header in
   `RelocateEntrySource.lean` records) and its `src/table/{state,locator,probe}.rs` children, all
   read. `../sanguine/proof/Apeiron/` was searched for a
   relocation *resource* theorem and none refines a work measure; the object refined here is the
   extracted body itself, not a corpus theorem.
   REFERENCE: `/home/socol/Workspace/apeiron/proof/resource/RelocateEntrySource.lean`,
   Aeneas-generated from the real source by this gate.
   GATE: `python3 proof/resource/RelocateEntry_gate.py --check --controls`, 540 s wall clock
   under `/tmp/opencode/apeiron-lane-3.lock`, `MemoryMax=12G`, `MemorySwapMax=0`,
   `LEAN_NUM_THREADS=1`.
   GRADE: `proved` for each headline below when the gate exits 0, since every headline is
   `#resource_audit`ed under `{propext, Classical.choice, Quot.sound}`. No measurement, and no
   correspondence to `R[x]/(x^2 - sigma)` beyond the extracted body's own source. -/

set_option linter.unusedSimpArgs false
set_option linter.unnecessarySeqFocus false

namespace RelocateEntryResource
open relocate_entry_source Aeneas Aeneas.Std Result

/-! ## Named types at the concrete stored-carrier instance

The group width `N` is the table's const generic. The extracted root is monomorphic at
`N = 16`; the spill branch's theorems are stated at an abstract `N` and instantiated at that
width, because the extracted `16#usize` literal carries a `decide`-guarded representability
proof that is not transparent to a simplifier's type check. -/

/-- The stored word key the extracted root is monomorphic at. -/
abbrev StoredI64 := table.keyrep.Stored Std.I64

/-- The table type at an abstract group width. -/
abbrev TableAt (N : Std.Usize) := table.Table Std.I64 Std.I64 table.WordHome StoredI64 N

/-- The concrete table type the extracted root returns. -/
abbrev WordTable := TableAt 16#usize

/-- The reserved destination storage the extracted root returns. -/
abbrev ReservedI64 := table.RebuildReserved StoredI64 Std.I64

/-- The `Home` instance the extracted root passes. -/
abbrev HomeInst := table.WordHome.Insts.Relocate_entry_sourceTableHomeI64Sigma

/-- The exact-identity instance the extracted root's `KeyRep` takes. -/
abbrev ExactIdentityInst :=
  I64.Insts.Relocate_entry_sourceCoordLocatorLocator_coreExactIdentityU64Tuple

/-- The `KeyRep` instance the extracted root passes. -/
abbrev KeyRepInst :=
  table.keyrep.Stored.Insts.Relocate_entry_sourceTableKeyrepKeyRep
    ExactIdentityInst core.clone.CloneI64 HomeInst

/-- The real body at an abstract group width: the exact call the extracted root forwards to. -/
abbrev RelocateEntryAt (N : Std.Usize) (state : TableAt N) (plan : table.RebuildPlan)
    (reserved : ReservedI64) (entryAt : Std.Usize) : Result (TableAt N × ReservedI64) :=
  table.Table.relocate_entry core.clone.CloneI64 core.clone.CloneI64 HomeInst KeyRepInst
    state plan reserved entryAt

/-- The plan's record of where the moved payload came from, as the body reads it. -/
def oldLocation : table.RebuildSource → table.PayloadLocation
  | .Resident i => .Resident i
  | .Spill sh i => .Spill sh i

/-- The payload one plan entry moves: the clone taken from the old resident slot or the old
    spill slot, read as its `(sigma, held, value)` triple off the flat `FlatSlot` or the spill
    `SpillSlot` the extracted body reads. `none` exactly when the recorded index is out of range,
    which is the read the body turns into a refusal rather than a move. -/
def sourcePayload {N : Std.Usize} (state : TableAt N) : table.RebuildSource →
    Option (cell.Sigma × StoredI64 × Std.I64)
  | .Resident i =>
      match state.resident.slots.val[i.val]? with
      | none => none
      | some s => some (s.sigma, s.held, s.value)
  | .Spill sh i =>
      match state.shards.val[sh.val]? with
      | none => none
      | some r =>
        match r.slots.val[i.val]? with
        | none => none
        | some s => some (s.sigma, s.held, s.value)

/-! ## The flat transition: one planned slot written, nothing else -/

/--
One guarded call of the real body that moves a plan entry to its planned flat slot is inside
the admissible domain `entry_at < plan.entries.len()`, performs exactly one payload clone out
of the old area and exactly one write into the reserved flat destination, leaves the
destination vector's length and capacity unchanged, and changes no field of the source table
other than its locator. The work is the closed form `1` reserved write, counted in `List.set`
operations on `reserved.resident.slots`.
-/
theorem flat_transition
    (state : WordTable) (plan : table.RebuildPlan) (reserved : ReservedI64)
    (entryAt index : Usize) (entry : table.RebuildPlanEntry)
    (payload : cell.Sigma × StoredI64 × Std.I64)
    (hentryAt : entryAt.val < plan.entries.length)
    (hentry : plan.entries.val[entryAt.val]? = some entry)
    (hflat : entry.flat = some index)
    (hindex : index.val < reserved.resident.slots.length)
    (hsource : sourcePayload state entry.source = some payload)
    (hretarget : ∃ nextLocator,
      table.TableLocator.retarget_held core.clone.CloneI64 HomeInst KeyRepInst
        state.locator state.home payload.1 { key := payload.2.1.key }
        (oldLocation entry.source) (table.PayloadLocation.Resident index) = ok nextLocator) :
    relocate_entry_source.word_stored_relocate_entry state plan reserved entryAt
      ⦃ result =>
        entryAt.val < plan.entries.length ∧
        ∃ nextLocator,
          result.1.locator = nextLocator ∧
          result.1.resident = state.resident ∧
          result.1.shards = state.shards ∧
          result.1.len = state.len ∧
          result.1.home = state.home ∧
          result.1.default = state.default ∧
          result.2.regions = reserved.regions ∧
          result.2.resident.lanes = reserved.resident.lanes ∧
          result.2.resident.slots.val = reserved.resident.slots.val.set index.val
            { live := true, sigma := payload.1, held := payload.2.1, value := payload.2.2 } ∧
          result.2.resident.slots.length = reserved.resident.slots.length ∧
          result.2.resident.slots.cap = reserved.resident.slots.cap ⦄ := by
  cases hsrc : entry.source with
  | Resident i =>
      cases hslot : state.resident.slots.val[i.val]? with
      | none => simp [sourcePayload, hsrc, hslot] at hsource
      | some s =>
          obtain ⟨nextLocator, hretarget⟩ := hretarget
          have hpayload : payload = (s.sigma, s.held, s.value) := by
            simpa [sourcePayload, hsrc, hslot] using hsource.symm
          subst payload
          have hretarget' : table.TableLocator.retarget_held core.clone.CloneI64 HomeInst
              KeyRepInst state.locator state.home s.sigma { key := s.held.key }
              (table.PayloadLocation.Resident i) (table.PayloadLocation.Resident index)
              = ok nextLocator := by
            simpa [oldLocation, hsrc] using hretarget
          unfold relocate_entry_source.word_stored_relocate_entry table.Table.relocate_entry
          simp [hentry, hsrc, hslot, hflat, hretarget', oldLocation,
            core.clone.impls.CloneI64.clone, table.keyrep.Stored.Insts.CoreCloneClone.clone,
            alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
            alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize]
          simp [hentryAt, hindex] <;> rfl
  | Spill sh i =>
      cases hshard : state.shards.val[sh.val]? with
      | none => simp [sourcePayload, hsrc, hshard] at hsource
      | some r =>
          cases hslot : r.slots.val[i.val]? with
          | none => simp [sourcePayload, hsrc, hshard, hslot] at hsource
          | some p =>
              obtain ⟨nextLocator, hretarget⟩ := hretarget
              have hpayload : (p.sigma, p.held, p.value) = payload := by
                simpa [sourcePayload, hsrc, hshard, hslot] using hsource
              subst payload
              have hretarget' : table.TableLocator.retarget_held core.clone.CloneI64 HomeInst
                  KeyRepInst state.locator state.home p.sigma { key := p.held.key }
                  (table.PayloadLocation.Spill sh i) (table.PayloadLocation.Resident index)
                  = ok nextLocator := by
                simpa [oldLocation, hsrc] using hretarget
              unfold relocate_entry_source.word_stored_relocate_entry table.Table.relocate_entry
              simp [hentry, hsrc, hshard, hslot, hflat, hretarget', oldLocation,
                core.clone.impls.CloneI64.clone, table.keyrep.Stored.Insts.CoreCloneClone.clone,
                alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
                alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize]
              simp [hentryAt, hindex] <;> rfl

/-! ## The flat branch's exit -/

/--
A planned flat index outside `reserved.resident.slots` is refused after the payload clone and
after the locator retarget, and before the slot write, so the body performs zero reserved
writes.
-/
theorem flat_slot_out_of_bounds_exit
    (state : WordTable) (plan : table.RebuildPlan) (reserved : ReservedI64)
    (entryAt index : Usize) (entry : table.RebuildPlanEntry)
    (payload : cell.Sigma × StoredI64 × Std.I64)
    (hentry : plan.entries.val[entryAt.val]? = some entry)
    (hflat : entry.flat = some index)
    (hout : reserved.resident.slots.length ≤ index.val)
    (hsource : sourcePayload state entry.source = some payload)
    (hretarget : ∃ nextLocator,
      table.TableLocator.retarget_held core.clone.CloneI64 HomeInst KeyRepInst
        state.locator state.home payload.1 { key := payload.2.1.key }
        (oldLocation entry.source) (table.PayloadLocation.Resident index) = ok nextLocator) :
    relocate_entry_source.word_stored_relocate_entry state plan reserved entryAt
      = fail .arrayOutOfBounds := by
  have hn : ¬ index.val < reserved.resident.slots.length := by omega
  cases hsrc : entry.source with
  | Resident i =>
      cases hslot : state.resident.slots.val[i.val]? with
      | none => simp [sourcePayload, hsrc, hslot] at hsource
      | some s =>
          obtain ⟨nextLocator, hretarget⟩ := hretarget
          have hpayload : payload = (s.sigma, s.held, s.value) := by
            simpa [sourcePayload, hsrc, hslot] using hsource.symm
          subst payload
          have hretarget' : table.TableLocator.retarget_held core.clone.CloneI64 HomeInst
              KeyRepInst state.locator state.home s.sigma { key := s.held.key }
              (table.PayloadLocation.Resident i) (table.PayloadLocation.Resident index)
              = ok nextLocator := by
            simpa [oldLocation, hsrc] using hretarget
          unfold relocate_entry_source.word_stored_relocate_entry table.Table.relocate_entry
          simp [hentry, hsrc, hslot, hflat, hn, hretarget', oldLocation,
            core.clone.impls.CloneI64.clone, table.keyrep.Stored.Insts.CoreCloneClone.clone,
            alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
            alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize,
            List.getElem?_eq_none]
  | Spill sh i =>
      cases hshard : state.shards.val[sh.val]? with
      | none => simp [sourcePayload, hsrc, hshard] at hsource
      | some r =>
          cases hslot : r.slots.val[i.val]? with
          | none => simp [sourcePayload, hsrc, hshard, hslot] at hsource
          | some p =>
              obtain ⟨nextLocator, hretarget⟩ := hretarget
              have hpayload : (p.sigma, p.held, p.value) = payload := by
                simpa [sourcePayload, hsrc, hshard, hslot] using hsource
              subst payload
              have hretarget' : table.TableLocator.retarget_held core.clone.CloneI64 HomeInst
                  KeyRepInst state.locator state.home p.sigma { key := p.held.key }
                  (table.PayloadLocation.Spill sh i) (table.PayloadLocation.Resident index)
                  = ok nextLocator := by
                simpa [oldLocation, hsrc] using hretarget
              unfold relocate_entry_source.word_stored_relocate_entry table.Table.relocate_entry
              simp [hentry, hsrc, hshard, hslot, hflat, hn, hretarget', oldLocation,
                core.clone.impls.CloneI64.clone, table.keyrep.Stored.Insts.CoreCloneClone.clone,
                alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
                alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize,
                List.getElem?_eq_none]

/-! ## The spill transition: one planned publication, nothing else -/

/--
One guarded call of the real body that moves a plan entry to a plan-certified spill vacancy
performs exactly one payload clone out of the old area and exactly one
`Region.publish_live` into the planned spill region of the reserved destination, leaves the
region vector's length and capacity unchanged, and changes no field of the source table other
than its locator. The premises `hsite` and `hpublish` are the probe's and the publication's own
guarded answers, so the theorem does not re-derive either; `hregion` carries the in-range
premise for the planned spill shard.
-/
theorem spill_vacant_transition_at
    (N : Std.Usize) (state : TableAt N) (plan : table.RebuildPlan) (reserved : ReservedI64)
    (entryAt dest : Usize) (entry : table.RebuildPlanEntry)
    (payload : cell.Sigma × StoredI64 × Std.I64) (addr : table.TableAddress)
    (region region' : table.Region StoredI64 Std.I64)
    (hentryAt : entryAt.val < plan.entries.length)
    (hentry : plan.entries.val[entryAt.val]? = some entry)
    (hnone : entry.flat = none)
    (hregion : reserved.regions.val[entry.spill.val]? = some region)
    (hsource : sourcePayload state entry.source = some payload)
    (haddress : table.keyrep.Stored.Insts.Relocate_entry_sourceTableKeyrepKeyRep.address_held
        ExactIdentityInst core.clone.CloneI64 HomeInst state.home payload.1
        { key := payload.2.1.key } = ok addr)
    (hsite : table.probe.spill_locate_held KeyRepInst region payload.1 addr { key := payload.2.1.key }
      N = ok (table.SpillSite.Vacant dest))
    (hpublish : table.Region.publish_live region dest
      (payload.1, { key := payload.2.1.key }, payload.2.2) addr.place = ok region')
    (hretarget : ∃ nextLocator,
      table.TableLocator.retarget_held core.clone.CloneI64 HomeInst KeyRepInst
        state.locator state.home payload.1 { key := payload.2.1.key }
        (oldLocation entry.source)
        (table.PayloadLocation.Spill entry.spill dest) = ok nextLocator) :
    RelocateEntryAt N state plan reserved entryAt
      ⦃ result =>
        entryAt.val < plan.entries.length ∧
        ∃ nextLocator,
          result.1.locator = nextLocator ∧
          result.1.resident = state.resident ∧
          result.1.shards = state.shards ∧
          result.1.len = state.len ∧
          result.1.home = state.home ∧
          result.1.default = state.default ∧
          result.2.resident = reserved.resident ∧
          result.2.regions.val = reserved.regions.val.set entry.spill.val region' ∧
          result.2.regions.length = reserved.regions.length ∧
          result.2.regions.cap = reserved.regions.cap ⦄ := by
  cases hsrc : entry.source with
  | Resident i =>
      cases hslot : state.resident.slots.val[i.val]? with
      | none => simp [sourcePayload, hsrc, hslot] at hsource
      | some s =>
          obtain ⟨nextLocator, hretarget⟩ := hretarget
          have hpayload : payload = (s.sigma, s.held, s.value) := by
            simpa [sourcePayload, hsrc, hslot] using hsource.symm
          subst payload
          have hretarget' : table.TableLocator.retarget_held core.clone.CloneI64 HomeInst
              KeyRepInst state.locator state.home s.sigma { key := s.held.key }
              (table.PayloadLocation.Resident i)
              (table.PayloadLocation.Spill entry.spill dest) = ok nextLocator := by
            simpa [oldLocation, hsrc] using hretarget
          have haddress' : table.keyrep.Stored.Insts.Relocate_entry_sourceTableKeyrepKeyRep.address_held
              ExactIdentityInst core.clone.CloneI64 HomeInst state.home s.sigma
              { key := s.held.key } = ok addr := by simpa using haddress
          have hsite' : table.probe.spill_locate_held KeyRepInst region s.sigma addr
              { key := s.held.key } N = ok (table.SpillSite.Vacant dest) := by
            simpa using hsite
          have hpublish' : table.Region.publish_live region dest
              (s.sigma, { key := s.held.key }, s.value) addr.place = ok region' := by
            simpa using hpublish
          unfold RelocateEntryAt table.Table.relocate_entry
          simp [hentry, hsrc, hslot, hnone, hregion, haddress', hsite', hpublish', hretarget',
            oldLocation, core.clone.impls.CloneI64.clone,
            table.keyrep.Stored.Insts.CoreCloneClone.clone,
            alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
            alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize]
          simp [hentryAt] <;> rfl
  | Spill sh i =>
      cases hshard : state.shards.val[sh.val]? with
      | none => simp [sourcePayload, hsrc, hshard] at hsource
      | some r =>
          cases hslot : r.slots.val[i.val]? with
          | none => simp [sourcePayload, hsrc, hshard, hslot] at hsource
          | some p =>
              obtain ⟨nextLocator, hretarget⟩ := hretarget
              have hpayload : (p.sigma, p.held, p.value) = payload := by
                simpa [sourcePayload, hsrc, hshard, hslot] using hsource
              subst payload
              have hretarget' : table.TableLocator.retarget_held core.clone.CloneI64 HomeInst
                  KeyRepInst state.locator state.home p.sigma { key := p.held.key }
                  (table.PayloadLocation.Spill sh i)
                  (table.PayloadLocation.Spill entry.spill dest) = ok nextLocator := by
                simpa [oldLocation, hsrc] using hretarget
              have haddress' : table.keyrep.Stored.Insts.Relocate_entry_sourceTableKeyrepKeyRep.address_held
                  ExactIdentityInst core.clone.CloneI64 HomeInst state.home p.sigma
                  { key := p.held.key } = ok addr := by simpa using haddress
              have hsite' : table.probe.spill_locate_held KeyRepInst region p.sigma addr
                  { key := p.held.key } N = ok (table.SpillSite.Vacant dest) := by
                simpa using hsite
              have hpublish' : table.Region.publish_live region dest
                  (p.sigma, { key := p.held.key }, p.value) addr.place = ok region' := by
                simpa using hpublish
              unfold RelocateEntryAt table.Table.relocate_entry
              simp [hentry, hsrc, hshard, hslot, hnone, hregion, haddress', hsite', hpublish',
                hretarget', oldLocation, core.clone.impls.CloneI64.clone,
                table.keyrep.Stored.Insts.CoreCloneClone.clone,
                alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
                alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize]
              simp [hentryAt] <;> rfl

/-- The spill transition at the extracted root's own group width. -/
theorem spill_vacant_transition
    (state : WordTable) (plan : table.RebuildPlan) (reserved : ReservedI64)
    (entryAt dest : Usize) (entry : table.RebuildPlanEntry)
    (payload : cell.Sigma × StoredI64 × Std.I64) (addr : table.TableAddress)
    (region region' : table.Region StoredI64 Std.I64)
    (hentryAt : entryAt.val < plan.entries.length)
    (hentry : plan.entries.val[entryAt.val]? = some entry)
    (hnone : entry.flat = none)
    (hregion : reserved.regions.val[entry.spill.val]? = some region)
    (hsource : sourcePayload state entry.source = some payload)
    (haddress : table.keyrep.Stored.Insts.Relocate_entry_sourceTableKeyrepKeyRep.address_held
        ExactIdentityInst core.clone.CloneI64 HomeInst state.home payload.1
        { key := payload.2.1.key } = ok addr)
    (hsite : table.probe.spill_locate_held KeyRepInst region payload.1 addr { key := payload.2.1.key }
      16#usize = ok (table.SpillSite.Vacant dest))
    (hpublish : table.Region.publish_live region dest
      (payload.1, { key := payload.2.1.key }, payload.2.2) addr.place = ok region')
    (hretarget : ∃ nextLocator,
      table.TableLocator.retarget_held core.clone.CloneI64 HomeInst KeyRepInst
        state.locator state.home payload.1 { key := payload.2.1.key }
        (oldLocation entry.source)
        (table.PayloadLocation.Spill entry.spill dest) = ok nextLocator) :
    relocate_entry_source.word_stored_relocate_entry state plan reserved entryAt
      ⦃ result =>
        entryAt.val < plan.entries.length ∧
        ∃ nextLocator,
          result.1.locator = nextLocator ∧
          result.1.resident = state.resident ∧
          result.1.shards = state.shards ∧
          result.1.len = state.len ∧
          result.1.home = state.home ∧
          result.1.default = state.default ∧
          result.2.resident = reserved.resident ∧
          result.2.regions.val = reserved.regions.val.set entry.spill.val region' ∧
          result.2.regions.length = reserved.regions.length ∧
          result.2.regions.cap = reserved.regions.cap ⦄ :=
  spill_vacant_transition_at 16#usize state plan reserved entryAt dest entry payload addr region
    region' hentryAt hentry hnone hregion hsource haddress hsite hpublish hretarget

/-! ## The out-of-range plan index -/

/--
The call's first exit: a plan index outside `plan.entries` is refused before any payload is
read and before any write, so the body performs zero reserved writes.
-/
theorem entry_out_of_bounds_exit
    (state : WordTable) (plan : table.RebuildPlan) (reserved : ReservedI64) (entryAt : Usize)
    (hout : plan.entries.length ≤ entryAt.val) :
    relocate_entry_source.word_stored_relocate_entry state plan reserved entryAt
      = fail .arrayOutOfBounds := by
  have hn : ¬ entryAt.val < plan.entries.length := by omega
  unfold relocate_entry_source.word_stored_relocate_entry table.Table.relocate_entry
  simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, List.getElem?_eq_none, hn]

/-! ## The two plan-premise violations the body names in its `unreachable!` -/

/--
A spill destination the probe reports already `Found` is refused with a panic, and no reserved
write is reached.
-/
theorem spill_site_found_exit_at
    (N : Std.Usize) (state : TableAt N) (plan : table.RebuildPlan) (reserved : ReservedI64)
    (entryAt found : Usize) (entry : table.RebuildPlanEntry)
    (payload : cell.Sigma × StoredI64 × Std.I64) (addr : table.TableAddress)
    (region : table.Region StoredI64 Std.I64)
    (hentry : plan.entries.val[entryAt.val]? = some entry)
    (hnone : entry.flat = none)
    (hregion : reserved.regions.val[entry.spill.val]? = some region)
    (hsource : sourcePayload state entry.source = some payload)
    (haddress : table.keyrep.Stored.Insts.Relocate_entry_sourceTableKeyrepKeyRep.address_held
        ExactIdentityInst core.clone.CloneI64 HomeInst state.home payload.1
        { key := payload.2.1.key } = ok addr)
    (hsite : table.probe.spill_locate_held KeyRepInst region payload.1 addr { key := payload.2.1.key }
      N = ok (table.SpillSite.Found found)) :
    RelocateEntryAt N state plan reserved entryAt = fail .panic := by
  cases hsrc : entry.source with
  | Resident i =>
      cases hslot : state.resident.slots.val[i.val]? with
      | none => simp [sourcePayload, hsrc, hslot] at hsource
      | some s =>
          have hpayload : payload = (s.sigma, s.held, s.value) := by
            simpa [sourcePayload, hsrc, hslot] using hsource.symm
          subst payload
          have haddress' : table.keyrep.Stored.Insts.Relocate_entry_sourceTableKeyrepKeyRep.address_held
              ExactIdentityInst core.clone.CloneI64 HomeInst state.home s.sigma
              { key := s.held.key } = ok addr := by simpa using haddress
          have hsite' : table.probe.spill_locate_held KeyRepInst region s.sigma addr
              { key := s.held.key } N = ok (table.SpillSite.Found found) := by simpa using hsite
          unfold RelocateEntryAt table.Table.relocate_entry
          simp [hentry, hsrc, hslot, hnone, hregion, haddress', hsite', oldLocation,
            core.clone.impls.CloneI64.clone, table.keyrep.Stored.Insts.CoreCloneClone.clone,
            alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize]
  | Spill sh i =>
      cases hshard : state.shards.val[sh.val]? with
      | none => simp [sourcePayload, hsrc, hshard] at hsource
      | some r =>
          cases hslot : r.slots.val[i.val]? with
          | none => simp [sourcePayload, hsrc, hshard, hslot] at hsource
          | some p =>
              have hpayload : (p.sigma, p.held, p.value) = payload := by
                simpa [sourcePayload, hsrc, hshard, hslot] using hsource
              subst payload
              have haddress' : table.keyrep.Stored.Insts.Relocate_entry_sourceTableKeyrepKeyRep.address_held
                  ExactIdentityInst core.clone.CloneI64 HomeInst state.home p.sigma
                  { key := p.held.key } = ok addr := by simpa using haddress
              have hsite' : table.probe.spill_locate_held KeyRepInst region p.sigma addr
                  { key := p.held.key } N = ok (table.SpillSite.Found found) := by simpa using hsite
              unfold RelocateEntryAt table.Table.relocate_entry
              simp [hentry, hsrc, hshard, hslot, hnone, hregion, haddress', hsite', oldLocation,
                core.clone.impls.CloneI64.clone, table.keyrep.Stored.Insts.CoreCloneClone.clone,
                alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize]

/-- The `Found` plan-premise violation at the extracted root's own group width. -/
theorem spill_site_found_exit
    (state : WordTable) (plan : table.RebuildPlan) (reserved : ReservedI64)
    (entryAt found : Usize) (entry : table.RebuildPlanEntry)
    (payload : cell.Sigma × StoredI64 × Std.I64) (addr : table.TableAddress)
    (region : table.Region StoredI64 Std.I64)
    (hentry : plan.entries.val[entryAt.val]? = some entry)
    (hnone : entry.flat = none)
    (hregion : reserved.regions.val[entry.spill.val]? = some region)
    (hsource : sourcePayload state entry.source = some payload)
    (haddress : table.keyrep.Stored.Insts.Relocate_entry_sourceTableKeyrepKeyRep.address_held
        ExactIdentityInst core.clone.CloneI64 HomeInst state.home payload.1
        { key := payload.2.1.key } = ok addr)
    (hsite : table.probe.spill_locate_held KeyRepInst region payload.1 addr { key := payload.2.1.key }
      16#usize = ok (table.SpillSite.Found found)) :
    relocate_entry_source.word_stored_relocate_entry state plan reserved entryAt
      = fail .panic :=
  spill_site_found_exit_at 16#usize state plan reserved entryAt found entry payload addr region
    hentry hnone hregion hsource haddress hsite

/--
A spill destination the probe reports `Grow` is refused with a panic, and no reserved write is
reached, so the reserved region is not extended.
-/
theorem spill_site_grow_exit_at
    (N : Std.Usize) (state : TableAt N) (plan : table.RebuildPlan) (reserved : ReservedI64)
    (entryAt : Usize) (entry : table.RebuildPlanEntry)
    (payload : cell.Sigma × StoredI64 × Std.I64) (addr : table.TableAddress)
    (region : table.Region StoredI64 Std.I64)
    (hentry : plan.entries.val[entryAt.val]? = some entry)
    (hnone : entry.flat = none)
    (hregion : reserved.regions.val[entry.spill.val]? = some region)
    (hsource : sourcePayload state entry.source = some payload)
    (haddress : table.keyrep.Stored.Insts.Relocate_entry_sourceTableKeyrepKeyRep.address_held
        ExactIdentityInst core.clone.CloneI64 HomeInst state.home payload.1
        { key := payload.2.1.key } = ok addr)
    (hsite : table.probe.spill_locate_held KeyRepInst region payload.1 addr { key := payload.2.1.key }
      N = ok table.SpillSite.Grow) :
    RelocateEntryAt N state plan reserved entryAt = fail .panic := by
  cases hsrc : entry.source with
  | Resident i =>
      cases hslot : state.resident.slots.val[i.val]? with
      | none => simp [sourcePayload, hsrc, hslot] at hsource
      | some s =>
          have hpayload : payload = (s.sigma, s.held, s.value) := by
            simpa [sourcePayload, hsrc, hslot] using hsource.symm
          subst payload
          have haddress' : table.keyrep.Stored.Insts.Relocate_entry_sourceTableKeyrepKeyRep.address_held
              ExactIdentityInst core.clone.CloneI64 HomeInst state.home s.sigma
              { key := s.held.key } = ok addr := by simpa using haddress
          have hsite' : table.probe.spill_locate_held KeyRepInst region s.sigma addr
              { key := s.held.key } N = ok table.SpillSite.Grow := by simpa using hsite
          unfold RelocateEntryAt table.Table.relocate_entry
          simp [hentry, hsrc, hslot, hnone, hregion, haddress', hsite', oldLocation,
            core.clone.impls.CloneI64.clone, table.keyrep.Stored.Insts.CoreCloneClone.clone,
            alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize]
  | Spill sh i =>
      cases hshard : state.shards.val[sh.val]? with
      | none => simp [sourcePayload, hsrc, hshard] at hsource
      | some r =>
          cases hslot : r.slots.val[i.val]? with
          | none => simp [sourcePayload, hsrc, hshard, hslot] at hsource
          | some p =>
              have hpayload : (p.sigma, p.held, p.value) = payload := by
                simpa [sourcePayload, hsrc, hshard, hslot] using hsource
              subst payload
              have haddress' : table.keyrep.Stored.Insts.Relocate_entry_sourceTableKeyrepKeyRep.address_held
                  ExactIdentityInst core.clone.CloneI64 HomeInst state.home p.sigma
                  { key := p.held.key } = ok addr := by simpa using haddress
              have hsite' : table.probe.spill_locate_held KeyRepInst region p.sigma addr
                  { key := p.held.key } N = ok table.SpillSite.Grow := by simpa using hsite
              unfold RelocateEntryAt table.Table.relocate_entry
              simp [hentry, hsrc, hshard, hslot, hnone, hregion, haddress', hsite', oldLocation,
                core.clone.impls.CloneI64.clone, table.keyrep.Stored.Insts.CoreCloneClone.clone,
                alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize]

/-- The `Grow` plan-premise violation at the extracted root's own group width. -/
theorem spill_site_grow_exit
    (state : WordTable) (plan : table.RebuildPlan) (reserved : ReservedI64)
    (entryAt : Usize) (entry : table.RebuildPlanEntry)
    (payload : cell.Sigma × StoredI64 × Std.I64) (addr : table.TableAddress)
    (region : table.Region StoredI64 Std.I64)
    (hentry : plan.entries.val[entryAt.val]? = some entry)
    (hnone : entry.flat = none)
    (hregion : reserved.regions.val[entry.spill.val]? = some region)
    (hsource : sourcePayload state entry.source = some payload)
    (haddress : table.keyrep.Stored.Insts.Relocate_entry_sourceTableKeyrepKeyRep.address_held
        ExactIdentityInst core.clone.CloneI64 HomeInst state.home payload.1
        { key := payload.2.1.key } = ok addr)
    (hsite : table.probe.spill_locate_held KeyRepInst region payload.1 addr { key := payload.2.1.key }
      16#usize = ok table.SpillSite.Grow) :
    relocate_entry_source.word_stored_relocate_entry state plan reserved entryAt
      = fail .panic :=
  spill_site_grow_exit_at 16#usize state plan reserved entryAt entry payload addr region
    hentry hnone hregion hsource haddress hsite

/-! ## Reachable states -/

/-- A source table at an abstract group width holding one live elliptic payload in its resident
    area, with no locator directory and no link table, so a retarget finds no terminal handle. -/
def witnessStateAt (N : Std.Usize) : TableAt N :=
  { default := 0#i64
    home := ()
    resident :=
      { lanes := 2#usize
        slots := .from [⟨true, .Elliptic, ⟨7#i64⟩, 7#i64⟩, ⟨false, .Elliptic, ⟨0#i64⟩, 0#i64⟩]
          (by grind) }
    shards := .from [] (by grind)
    len := 1#usize
    key := ()
    locator := { profile := none, directory := none, links := .from [] (by grind), staged := none } }

/-- The source table at the extracted root's group width. -/
def witnessState : WordTable := witnessStateAt 16#usize

/-- A concrete checked plan with one entry: read resident slot `0`, write planned flat slot `1`. -/
def witnessPlan : table.RebuildPlan :=
  { lanes := 2#usize
    entries := .from [⟨.Resident 0#usize, some 1#usize, 0#usize⟩] (by grind)
    spill_loads := .from [] (by grind) }

/-- The concrete reserved destination storage: two vacant flat slots and no spill region. -/
def witnessReserved : ReservedI64 :=
  { resident :=
      { lanes := 2#usize
        slots := .from [⟨false, .Elliptic, ⟨0#i64⟩, 0#i64⟩, ⟨false, .Elliptic, ⟨0#i64⟩, 0#i64⟩]
          (by grind) }
    regions := .from [] (by grind) }

/--
A reachable success of the real body. The call reads resident slot `0`, writes the single
planned flat slot `1` with the live payload, leaves the source table's resident area, length
and default untouched, and returns in `ok`. The reserved destination vector holds exactly the
one closed-form write: `List.set 1`.
-/
theorem flat_success_witness :
    relocate_entry_source.word_stored_relocate_entry witnessState witnessPlan witnessReserved
        0#usize
      ⦃ result =>
        result.1.locator = witnessState.locator ∧
        result.1.resident = witnessState.resident ∧
        result.1.len = witnessState.len ∧
        result.1.shards = witnessState.shards ∧
        result.2.regions = witnessReserved.regions ∧
        result.2.resident.slots.val
          = [⟨false, .Elliptic, ⟨0#i64⟩, 0#i64⟩, ⟨true, .Elliptic, ⟨7#i64⟩, 7#i64⟩] ∧
        result.2.resident.slots.length = 2 ∧
        result.2.resident.slots.cap = witnessReserved.resident.slots.cap ⦄ := by
  unfold relocate_entry_source.word_stored_relocate_entry table.Table.relocate_entry
  simp [witnessState, witnessStateAt, witnessPlan, witnessReserved, sourcePayload, oldLocation,
    core.clone.impls.CloneI64.clone, table.keyrep.Stored.Insts.CoreCloneClone.clone,
    table.same_location, table.TableLocator.retarget_held, table.TableLocator.linked_handle,
    core.option.Option.as_ref, core.option.Option.Insts.CoreOpsTry_traitTry.branch,
    core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize] <;> rfl

/--
A reachable refusal of the real body at an abstract group width: the one-entry plan is indexed
at `entry_at = 1`, outside `plan.entries`, so the body's first read is refused and no reserved
write is reached.
-/
theorem plan_index_exit_witness_at
    (N : Std.Usize) (entryAt : Std.Usize) (hout : witnessPlan.entries.length ≤ entryAt.val) :
    RelocateEntryAt N (witnessStateAt N) witnessPlan witnessReserved entryAt
      = fail .arrayOutOfBounds := by
  unfold RelocateEntryAt table.Table.relocate_entry
  have hlen : witnessPlan.entries.length = 1 := by simp [witnessPlan]
  have hget : witnessPlan.entries.val[entryAt.val]? = none :=
    List.getElem?_eq_none (by omega)
  simp [hget, alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize]

/-- The reachable plan-index refusal at the extracted root's own group width. -/
theorem plan_index_exit_witness :
    relocate_entry_source.word_stored_relocate_entry witnessState witnessPlan witnessReserved
        1#usize = fail .arrayOutOfBounds := by
  exact plan_index_exit_witness_at 16#usize 1#usize (by simp [witnessPlan])

end RelocateEntryResource

open Lean Elab Command in
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

#resource_audit RelocateEntryResource.flat_transition
#resource_audit RelocateEntryResource.flat_slot_out_of_bounds_exit
#resource_audit RelocateEntryResource.spill_vacant_transition_at
#resource_audit RelocateEntryResource.spill_vacant_transition
#resource_audit RelocateEntryResource.entry_out_of_bounds_exit
#resource_audit RelocateEntryResource.spill_site_found_exit_at
#resource_audit RelocateEntryResource.spill_site_found_exit
#resource_audit RelocateEntryResource.spill_site_grow_exit_at
#resource_audit RelocateEntryResource.spill_site_grow_exit
#resource_audit RelocateEntryResource.flat_success_witness
#resource_audit RelocateEntryResource.plan_index_exit_witness_at
#resource_audit RelocateEntryResource.plan_index_exit_witness

run_cmd Lean.logInfo "RELOCATE_ENTRY_KERNEL_OK"
