import Growable.Read
import Lean

/-!
# Root composition of the public read function: `word_stored_table_get_at`

Completes row 6 of the read map (`docs/epics/read-proof.md`).
Assembles all landed partitions of `table.Table.get_at`:
1. Direct hit (`word_stored_table_get_at_direct_hit`)
2. Vacant flat slot (`word_stored_table_get_at_flat_vacant`)
3. Direct word none with summary miss (`word_stored_table_get_at_direct_none`)
4. Direct word none with sweep miss (`word_stored_table_get_at_direct_none_sweep_none`)
5. Direct word none with sweep hit (`word_stored_table_get_at_direct_none_hit`)
6. Flat index none with summary miss (`word_stored_table_get_at_flat_none`)
7. Flat index none with sweep miss (`word_stored_table_get_at_flat_none_sweep_none`)
8. Flat index none with sweep hit (`word_stored_table_get_at_flat_none_hit`)
9. Flat occupied miss with summary miss (`word_stored_table_get_at_flat_occupied_miss`)
10. Flat occupied miss with sweep miss (`word_stored_table_get_at_flat_occupied_miss_sweep_none`)
11. Flat occupied miss with sweep hit (`word_stored_table_get_at_flat_occupied_miss_hit`)

The parameter `table.valid_n 16#usize = ok true` is proved directly by `valid_n_16` using `rfl`
after unfolding generated definitions, so it discharges unconditionally.
-/

namespace Apeiron.GrowableRead

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-- `table.valid_n 16#usize = ok true` discharges by definition. -/
theorem valid_n_16 : table.valid_n 16#usize = ok true := by
  unfold table.valid_n table.MAX_GROUP core.num.Usize.is_power_of_two
  rfl

/-- Partition 5 (spill hit), direct word none: the public read returns the spill slot value
at a positive summary sweep hit, which denotes the canonical `Apeiron.Table.get`. -/
theorem word_stored_table_get_at_direct_none_hit
    (state : WordTable StoredI64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (hreach : ConstructorReachableRepR state Apeiron.GrowableSearch.decodeStoredOption
      direct live logical)
    (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (hword : address.direct_word = none)
    (shardAt : Usize) (hshard : table.shard address = ok shardAt)
    (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shardAt.val]? = some region)
    (groups mask limit slot : Usize)
    (hadmit : Apeiron.GrowableReadLoop.GroupedReadAdmission region address.place 16#usize groups mask limit)
    (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
    (hgroups2 : 2 ≤ groups.val)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok (some slot)) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  have hmem : region ∈ state.shards.val := List.mem_of_getElem? hregion
  have htagSlots : region.tags.length = region.slots.length := hreach.directory.2.2 region hmem
  obtain ⟨value, hentryAt, hslotIdx⟩ := spill_read_some_entryAt state shardAt region hregion
    sigma address key 16#usize groups mask limit slot hadmit htagSlots (by decide) hgroups2
    valid_n_16 hpowCheck hresult
  have hobs := hreach.observations.1 (.spill shardAt slot) (sigma, key) value hentryAt
  have hrun : word_stored_table_get_at state sigma key = ok value := by
    simp only [word_stored_table_get_at]
    unfold table.Table.get_at
    have haddr : table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address state.home sigma key = ok address := haddress
    simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddr, hword, hshard]
    have hvec : alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (table.Region StoredI64 I64))
        state.shards shardAt = ok region := by
      simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hregion]
    simp only [hvec, bind_tc_ok, hresult, hslotIdx]
    rfl
  rw [hrun, hobs]

/-- Partition 5 (spill hit), flat index none: the public read returns the spill slot value
at a positive summary sweep hit, which denotes the canonical `Apeiron.Table.get`. -/
theorem word_stored_table_get_at_flat_none_hit
    (state : WordTable StoredI64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (hreach : ConstructorReachableRepR state Apeiron.GrowableSearch.decodeStoredOption
      direct live logical)
    (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (word : U32) (hword : address.direct_word = some word)
    (hflat : table.Flat.index state.resident word = ok none)
    (shardAt : Usize) (hshard : table.shard address = ok shardAt)
    (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shardAt.val]? = some region)
    (groups mask limit slot : Usize)
    (hadmit : Apeiron.GrowableReadLoop.GroupedReadAdmission region address.place 16#usize groups mask limit)
    (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
    (hgroups2 : 2 ≤ groups.val)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok (some slot)) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  have hmem : region ∈ state.shards.val := List.mem_of_getElem? hregion
  have htagSlots : region.tags.length = region.slots.length := hreach.directory.2.2 region hmem
  obtain ⟨value, hentryAt, hslotIdx⟩ := spill_read_some_entryAt state shardAt region hregion
    sigma address key 16#usize groups mask limit slot hadmit htagSlots (by decide) hgroups2
    valid_n_16 hpowCheck hresult
  have hobs := hreach.observations.1 (.spill shardAt slot) (sigma, key) value hentryAt
  have hrun : word_stored_table_get_at state sigma key = ok value := by
    simp only [word_stored_table_get_at]
    unfold table.Table.get_at
    have haddr : table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address state.home sigma key = ok address := haddress
    simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddr, hword, hflat, hshard]
    have hvec : alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (table.Region StoredI64 I64))
        state.shards shardAt = ok region := by
      simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hregion]
    simp only [hvec, bind_tc_ok, hresult, hslotIdx]
    rfl
  rw [hrun, hobs]

/-- Partition 5 (spill hit), occupied non-match: the public read returns the spill slot value
at a positive summary sweep hit, which denotes the canonical `Apeiron.Table.get`. -/
theorem word_stored_table_get_at_flat_occupied_miss_hit
    (state : WordTable StoredI64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (hreach : ConstructorReachableRepR state Apeiron.GrowableSearch.decodeStoredOption
      direct live logical)
    (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (word : U32) (hword : address.direct_word = some word)
    (index : Usize) (flat : table.FlatSlot StoredI64 I64)
    (hflat : table.Flat.index state.resident word = ok (some index))
    (hslot : state.resident.slots.val[index.val]? = some flat)
    (hlive : flat.live = true)
    (hmismatch : (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok false) ∨
      (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true ∧
       Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok false))
    (shardAt : Usize) (hshard : table.shard address = ok shardAt)
    (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shardAt.val]? = some region)
    (groups mask limit slot : Usize)
    (hadmit : Apeiron.GrowableReadLoop.GroupedReadAdmission region address.place 16#usize groups mask limit)
    (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
    (hgroups2 : 2 ≤ groups.val)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok (some slot)) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  have hmem : region ∈ state.shards.val := List.mem_of_getElem? hregion
  have htagSlots : region.tags.length = region.slots.length := hreach.directory.2.2 region hmem
  obtain ⟨value, hentryAt, hslotIdx⟩ := spill_read_some_entryAt state shardAt region hregion
    sigma address key 16#usize groups mask limit slot hadmit htagSlots (by decide) hgroups2
    valid_n_16 hpowCheck hresult
  have hobs := hreach.observations.1 (.spill shardAt slot) (sigma, key) value hentryAt
  have hrun : word_stored_table_get_at state sigma key = ok value := by
    simp only [word_stored_table_get_at]
    unfold table.Table.get_at
    have haddr : table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address state.home sigma key = ok address := haddress
    simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddr, hword, hflat]
    have hidx : alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot StoredI64 I64))
        state.resident.slots index = ok flat := by
      simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hslot]
    simp only [hidx, bind_tc_ok, hlive]
    have hvec : alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (table.Region StoredI64 I64))
        state.shards shardAt = ok region := by
      simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hregion]
    rcases hmismatch with hsigma | ⟨hsigma, hkey⟩
    · simp only [hsigma, bind_tc_ok, hshard, hvec, hresult, hslotIdx, ↓reduceIte]
      rfl
    · simp only [hsigma, bind_tc_ok, hshard, hvec, hresult, hslotIdx, ↓reduceIte]
      have hkey' : table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.matches
          I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple core.clone.CloneI64
          table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma sigma flat.held key address = ok false := hkey
      rw [hkey']
      simp only [bind_tc_ok]
      rfl
  rw [hrun, hobs]

/-- Partition 5 (spill miss), direct word none: a positive summary sweep miss returns default,
which denotes the canonical absent `Apeiron.Table.get`. -/
theorem word_stored_table_get_at_direct_none_sweep_none
    (state : WordTable StoredI64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (hreach : ConstructorReachableRepR state Apeiron.GrowableSearch.decodeStoredOption
      direct live logical)
    (hstate : Apeiron.GrowableReadLoop.StoredLiveTagAddress state)
    (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (hword : address.direct_word = none)
    (shardAt : Usize) (hshard : table.shard address = ok shardAt)
    (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shardAt.val]? = some region)
    (groups mask limit : Usize)
    (hadmit : Apeiron.GrowableReadLoop.GroupedReadAdmission region address.place 16#usize groups mask limit)
    (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
    (hgroups2 : 2 ≤ groups.val)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok none) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  have hmem : region ∈ state.shards.val := List.mem_of_getElem? hregion
  have htagSlots : region.tags.length = region.slots.length := hreach.directory.2.2 region hmem
  have hnoneEntryAt := spill_read_none_entryAt_ne state hstate shardAt region hregion sigma key address haddress
    16#usize groups mask limit hadmit htagSlots (by decide) hgroups2
    valid_n_16 hpowCheck hresult
  exact word_stored_table_get_at_direct_none state direct live logical hreach sigma key address haddress
    hword shardAt hshard region hregion (Or.inr ⟨hresult, hnoneEntryAt⟩)

/-- Partition 5 (spill miss), flat index none: a positive summary sweep miss returns default,
which denotes the canonical absent `Apeiron.Table.get`. -/
theorem word_stored_table_get_at_flat_none_sweep_none
    (state : WordTable StoredI64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (hreach : ConstructorReachableRepR state Apeiron.GrowableSearch.decodeStoredOption
      direct live logical)
    (hstate : Apeiron.GrowableReadLoop.StoredLiveTagAddress state)
    (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (word : U32) (hword : address.direct_word = some word)
    (hflat : table.Flat.index state.resident word = ok none)
    (hlane : state.resident.lanes.val ≤ word.val / 256 ∨
      ∀ index value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.flat index) ≠ some ((sigma, key), value))
    (shardAt : Usize) (hshard : table.shard address = ok shardAt)
    (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shardAt.val]? = some region)
    (groups mask limit : Usize)
    (hadmit : Apeiron.GrowableReadLoop.GroupedReadAdmission region address.place 16#usize groups mask limit)
    (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
    (hgroups2 : 2 ≤ groups.val)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok none) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  have hmem : region ∈ state.shards.val := List.mem_of_getElem? hregion
  have htagSlots : region.tags.length = region.slots.length := hreach.directory.2.2 region hmem
  have hnoneEntryAt := spill_read_none_entryAt_ne state hstate shardAt region hregion sigma key address haddress
    16#usize groups mask limit hadmit htagSlots (by decide) hgroups2
    valid_n_16 hpowCheck hresult
  exact word_stored_table_get_at_flat_none state direct live logical hreach sigma key address haddress
    word hword hflat hlane shardAt hshard region hregion (Or.inr ⟨hresult, hnoneEntryAt⟩)

/-- Partition 5 (spill miss), occupied non-match: a positive summary sweep miss returns default,
which denotes the canonical absent `Apeiron.Table.get`. -/
theorem word_stored_table_get_at_flat_occupied_miss_sweep_none
    (state : WordTable StoredI64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (hreach : ConstructorReachableRepR state Apeiron.GrowableSearch.decodeStoredOption
      direct live logical)
    (hstate : Apeiron.GrowableReadLoop.StoredLiveTagAddress state)
    (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (word : U32) (hword : address.direct_word = some word)
    (index : Usize) (flat : table.FlatSlot StoredI64 I64)
    (hflat : table.Flat.index state.resident word = ok (some index))
    (hslot : state.resident.slots.val[index.val]? = some flat)
    (hlive : flat.live = true)
    (hmismatch : (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok false) ∨
      (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true ∧
       Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok false))
    (shardAt : Usize) (hshard : table.shard address = ok shardAt)
    (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shardAt.val]? = some region)
    (groups mask limit : Usize)
    (hadmit : Apeiron.GrowableReadLoop.GroupedReadAdmission region address.place 16#usize groups mask limit)
    (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
    (hgroups2 : 2 ≤ groups.val)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok none)
    (hroute : Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = some index.val ∨
      ∀ idx value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.flat idx) ≠ some ((sigma, key), value)) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  have hmem : region ∈ state.shards.val := List.mem_of_getElem? hregion
  have htagSlots : region.tags.length = region.slots.length := hreach.directory.2.2 region hmem
  have hnoneEntryAt := spill_read_none_entryAt_ne state hstate shardAt region hregion sigma key address haddress
    16#usize groups mask limit hadmit htagSlots (by decide) hgroups2
    valid_n_16 hpowCheck hresult
  have hflatExcl : ∀ idx value,
      Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption (.flat idx) ≠
        some ((sigma, key), value) := by
    cases hroute with
    | inl hr =>
        have hm : (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok false) ∨
            (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true ∧
             Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok false) ∨
            (table.matches Apeiron.GrowableIdentity.storedKeyRep
              (flat.sigma, flat.held, flat.value) sigma key address = ok false) := by
          rcases hmismatch with h | h
          · exact Or.inl h
          · exact Or.inr (Or.inl h)
        exact flat_occupied_exclusion_of_prepared state direct live logical hreach sigma key address haddress
          index flat hslot hlive hm hr
    | inr hexcl => exact hexcl
  have habsent := absent_of_exclusions hreach sigma key address haddress shardAt hshard hflatExcl hnoneEntryAt
  have hdefault := default_of_absent hreach (sigma, key) habsent
  have hrun : word_stored_table_get_at state sigma key = ok state.default := by
    simp only [word_stored_table_get_at]
    unfold table.Table.get_at
    have haddr : table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address state.home sigma key = ok address := haddress
    simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddr, hword, hflat]
    have hidx : alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot StoredI64 I64))
        state.resident.slots index = ok flat := by
      simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hslot]
    simp only [hidx, bind_tc_ok, hlive]
    have hvec : alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (table.Region StoredI64 I64))
        state.shards shardAt = ok region := by
      simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hregion]
    rcases hmismatch with hsigma | ⟨hsigma, hkey⟩
    · simp only [hsigma, bind_tc_ok, hshard, hvec, hresult, ↓reduceIte]
      rfl
    · simp only [hsigma, bind_tc_ok, hshard, hvec, hresult, ↓reduceIte]
      have hkey' : table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.matches
          I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple core.clone.CloneI64
          table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma sigma flat.held key address = ok false := hkey
      rw [hkey']
      simp only [bind_tc_ok]
      rfl
  rw [hrun, hdefault]

/-- The execution route of a generated `table.Table.get_at` call partitioned over the control flow. -/
inductive StoredReadRoute (state : WordTable StoredI64) (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress) : Prop where
  | directHit (word : U32) (index : Usize) (flat : table.FlatSlot StoredI64 I64)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok (some index))
      (hslot : state.resident.slots.val[index.val]? = some flat)
      (hlive : flat.live = true)
      (hsigmaEq : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true)
      (hraw : Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok true)
      (hmatch : table.matches Apeiron.GrowableIdentity.storedKeyRep
        (flat.sigma, flat.held, flat.value) sigma key address = ok true)
  | flatVacant (word : U32) (index : Usize) (flat : table.FlatSlot StoredI64 I64)
      (shardAt : Usize)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok (some index))
      (hslot : state.resident.slots.val[index.val]? = some flat)
      (hlive : flat.live = false)
      (hshard : table.shard address = ok shardAt)
      (hflatExcl : ∀ idx value,
        Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
          (.flat idx) ≠ some ((sigma, key), value))
      (hspillExcl : ∀ idx value,
        Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
          (.spill shardAt idx) ≠ some ((sigma, key), value))
  | directNoneSummaryMiss (shardAt : Usize) (region : table.Region StoredI64 I64)
      (hword : address.direct_word = none)
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hsummary : region.summary_maybe address.place = ok false)
  | directNoneSweepNone (shardAt : Usize) (region : table.Region StoredI64 I64)
      (groups mask limit : Usize)
      (hword : address.direct_word = none)
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hadmit : Apeiron.GrowableReadLoop.GroupedReadAdmission region address.place 16#usize groups mask limit)
      (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
      (hgroups2 : 2 ≤ groups.val)
      (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key 16#usize = ok none)
  | directNoneSweepHit (shardAt : Usize) (region : table.Region StoredI64 I64)
      (groups mask limit slot : Usize)
      (hword : address.direct_word = none)
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hadmit : Apeiron.GrowableReadLoop.GroupedReadAdmission region address.place 16#usize groups mask limit)
      (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
      (hgroups2 : 2 ≤ groups.val)
      (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key 16#usize = ok (some slot))
  | flatNoneSummaryMiss (word : U32) (shardAt : Usize) (region : table.Region StoredI64 I64)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok none)
      (hlane : state.resident.lanes.val ≤ word.val / 256 ∨
        ∀ index value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
          (.flat index) ≠ some ((sigma, key), value))
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hsummary : region.summary_maybe address.place = ok false)
  | flatNoneSweepNone (word : U32) (shardAt : Usize) (region : table.Region StoredI64 I64)
      (groups mask limit : Usize)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok none)
      (hlane : state.resident.lanes.val ≤ word.val / 256 ∨
        ∀ index value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
          (.flat index) ≠ some ((sigma, key), value))
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hadmit : Apeiron.GrowableReadLoop.GroupedReadAdmission region address.place 16#usize groups mask limit)
      (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
      (hgroups2 : 2 ≤ groups.val)
      (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key 16#usize = ok none)
  | flatNoneSweepHit (word : U32) (shardAt : Usize) (region : table.Region StoredI64 I64)
      (groups mask limit slot : Usize)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok none)
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hadmit : Apeiron.GrowableReadLoop.GroupedReadAdmission region address.place 16#usize groups mask limit)
      (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
      (hgroups2 : 2 ≤ groups.val)
      (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key 16#usize = ok (some slot))
  | flatOccupiedMissSummaryMiss (word : U32) (index : Usize) (flat : table.FlatSlot StoredI64 I64)
      (shardAt : Usize) (region : table.Region StoredI64 I64)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok (some index))
      (hslot : state.resident.slots.val[index.val]? = some flat)
      (hlive : flat.live = true)
      (hmismatch : (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok false) ∨
        (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true ∧
         Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok false))
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hsummary : region.summary_maybe address.place = ok false)
      (hroute : Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = some index.val ∨
        ∀ idx value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
          (.flat idx) ≠ some ((sigma, key), value))
  | flatOccupiedMissSweepNone (word : U32) (index : Usize) (flat : table.FlatSlot StoredI64 I64)
      (shardAt : Usize) (region : table.Region StoredI64 I64)
      (groups mask limit : Usize)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok (some index))
      (hslot : state.resident.slots.val[index.val]? = some flat)
      (hlive : flat.live = true)
      (hmismatch : (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok false) ∨
        (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true ∧
         Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok false))
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hadmit : Apeiron.GrowableReadLoop.GroupedReadAdmission region address.place 16#usize groups mask limit)
      (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
      (hgroups2 : 2 ≤ groups.val)
      (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key 16#usize = ok none)
      (hroute : Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = some index.val ∨
        ∀ idx value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
          (.flat idx) ≠ some ((sigma, key), value))
  | flatOccupiedMissSweepHit (word : U32) (index : Usize) (flat : table.FlatSlot StoredI64 I64)
      (shardAt : Usize) (region : table.Region StoredI64 I64)
      (groups mask limit slot : Usize)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok (some index))
      (hslot : state.resident.slots.val[index.val]? = some flat)
      (hlive : flat.live = true)
      (hmismatch : (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok false) ∨
        (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true ∧
         Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok false))
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hadmit : Apeiron.GrowableReadLoop.GroupedReadAdmission region address.place 16#usize groups mask limit)
      (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
      (hgroups2 : 2 ≤ groups.val)
      (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key 16#usize = ok (some slot))

/-- Deliverable 2: Root composition of the public read function `word_stored_table_get_at`
under constructor-reachable `RepR` and live tag/address invariants. Assembles all landed partitions
(direct hit, direct empty, occupied non-match, summary miss, and complete positive-summary sweep). -/
theorem word_stored_table_get_at_root
    (state : WordTable StoredI64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (hreach : ConstructorReachableRepR state Apeiron.GrowableSearch.decodeStoredOption
      direct live logical)
    (hstate : Apeiron.GrowableReadLoop.StoredLiveTagAddress state)
    (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (hroute : StoredReadRoute state sigma key address) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  cases hroute with
  | directHit word index flat hword hflat hslot hlive hsigmaEq hraw hmatch =>
      exact word_stored_table_get_at_direct_hit state direct live logical hreach sigma key address haddress word hword index flat hflat hslot hlive hsigmaEq hraw hmatch
  | flatVacant word index flat shardAt hword hflat hslot hlive hshard hflatExcl hspillExcl =>
      exact word_stored_table_get_at_flat_vacant state direct live logical hreach sigma key address haddress word hword index flat hflat hslot hlive shardAt hshard hflatExcl hspillExcl
  | directNoneSummaryMiss shardAt region hword hshard hregion hsummary =>
      exact word_stored_table_get_at_direct_none state direct live logical hreach sigma key address haddress hword shardAt hshard region hregion (Or.inl hsummary)
  | directNoneSweepNone shardAt region groups mask limit hword hshard hregion hadmit hpowCheck hgroups2 hresult =>
      exact word_stored_table_get_at_direct_none_sweep_none state direct live logical hreach hstate sigma key address haddress hword shardAt hshard region hregion groups mask limit hadmit hpowCheck hgroups2 hresult
  | directNoneSweepHit shardAt region groups mask limit slot hword hshard hregion hadmit hpowCheck hgroups2 hresult =>
      exact word_stored_table_get_at_direct_none_hit state direct live logical hreach sigma key address haddress hword shardAt hshard region hregion groups mask limit slot hadmit hpowCheck hgroups2 hresult
  | flatNoneSummaryMiss word shardAt region hword hflat hlane hshard hregion hsummary =>
      exact word_stored_table_get_at_flat_none state direct live logical hreach sigma key address haddress word hword hflat hlane shardAt hshard region hregion (Or.inl hsummary)
  | flatNoneSweepNone word shardAt region groups mask limit hword hflat hlane hshard hregion hadmit hpowCheck hgroups2 hresult =>
      exact word_stored_table_get_at_flat_none_sweep_none state direct live logical hreach hstate sigma key address haddress word hword hflat hlane shardAt hshard region hregion groups mask limit hadmit hpowCheck hgroups2 hresult
  | flatNoneSweepHit word shardAt region groups mask limit slot hword hflat hshard hregion hadmit hpowCheck hgroups2 hresult =>
      exact word_stored_table_get_at_flat_none_hit state direct live logical hreach sigma key address haddress word hword hflat shardAt hshard region hregion groups mask limit slot hadmit hpowCheck hgroups2 hresult
  | flatOccupiedMissSummaryMiss word index flat shardAt region hword hflat hslot hlive hmismatch hshard hregion hsummary hroute =>
      exact word_stored_table_get_at_flat_occupied_miss state direct live logical hreach sigma key address haddress word hword index flat hflat hslot hlive hmismatch shardAt hshard region hregion hsummary hroute
  | flatOccupiedMissSweepNone word index flat shardAt region groups mask limit hword hflat hslot hlive hmismatch hshard hregion hadmit hpowCheck hgroups2 hresult hroute =>
      exact word_stored_table_get_at_flat_occupied_miss_sweep_none state direct live logical hreach hstate sigma key address haddress word hword index flat hflat hslot hlive hmismatch shardAt hshard region hregion groups mask limit hadmit hpowCheck hgroups2 hresult hroute
  | flatOccupiedMissSweepHit word index flat shardAt region groups mask limit slot hword hflat hslot hlive hmismatch hshard hregion hadmit hpowCheck hgroups2 hresult =>
      exact word_stored_table_get_at_flat_occupied_miss_hit state direct live logical hreach sigma key address haddress word hword index flat hflat hslot hlive hmismatch shardAt hshard region hregion groups mask limit slot hadmit hpowCheck hgroups2 hresult

/-!
## Concrete composition fixtures and paired read controls

`word_stored_table_get_at_root` and its six route theorems all carry `ConstructorReachableRepR`
at the query state. No concrete value of that premise is available in this environment: the only
landed producer, `Growable.GrowableRepresentation.word_stored_table_new_repR`, is a WP triple over
the generated `table.Table.new` loop whose state and `shards` list are exposed only through the
private `Growable.GrowableRepresentation.table_new_empty_repR`, and that empty state carries
`direct = fun _ => none`, `live = fun _ => False`; non-empty reachability is row 4/8 of
`docs/epics/read-proof.md`, open. The route and root theorems therefore cannot be
instantiated at a concrete state here. The fixtures below exercise the same generated operations the
route theorems partition — generated address, `table.Flat.index`, flat liveness/`Sigma`/matcher
dispatch, `table.Region.summary_maybe`, `table.probe.spill_read` and `entryAt` — at concrete
stored-carrier states, each positive paired with a rejecting negative. `Result I64` has no `BEq` in
this environment, so each check matches on `Result.match` (`MatchResult`).
-/

private theorem compositionControlNatBound (n : Nat) (h : n ≤ 1000) : n ≤ Usize.max := by
  rcases Usize.bounds_eq with hb | hb
  · rw [hb]; exact le_trans h (by norm_num [U32.max, U32.numBits])
  · rw [hb]; exact le_trans h (by norm_num [U64.max, U64.numBits])

private def compositionControlRepBound {α : Type} (n : Nat) (x : α) (h : n ≤ 1000) :
    (List.replicate n x).length ≤ Usize.max := by
  rw [List.length_replicate]
  exact compositionControlNatBound n h

private def compositionControlTagBound {α : Type} (n : Nat) (x : α) (t : α) (h : n + 1 ≤ 1000) :
    (List.replicate n x ++ [t]).length ≤ Usize.max := by
  rw [List.length_append, List.length_replicate, List.length_singleton]
  exact compositionControlNatBound (n + 1) h

/-- An empty spill region: no tags, no payload slots, no summary words. Its `summary_maybe` is the
conservative `ok true`, so it is a routed region with no recorded fingerprint. -/
def compositionControlRegion : table.Region StoredI64 I64 :=
  { tags := alloc.vec.Vec.new _, slots := alloc.vec.Vec.new _, summary := alloc.vec.Vec.new _ }

/-- A live direct flat slot holding `(Parabolic, 5) ↦ 42`. -/
def compositionControlSlot : table.FlatSlot StoredI64 I64 :=
  { live := true, sigma := cell.Sigma.Parabolic, held := { key := 5#i64 }, value := 42#i64 }

/-- A vacant direct flat slot with the same attempted contents. -/
def compositionControlVacantSlot : table.FlatSlot StoredI64 I64 :=
  { live := false, sigma := cell.Sigma.Parabolic, held := { key := 5#i64 }, value := 42#i64 }

/-- A `lanes = 1` stored table whose 256 direct slots are all live `(Parabolic, 5) ↦ 42`. -/
def compositionControlLiveTable : WordTable StoredI64 :=
  { default := 7#i64, home := (),
    resident := { lanes := 1#usize, slots := alloc.vec.Vec.from (List.replicate 256 compositionControlSlot) (compositionControlRepBound 256 compositionControlSlot (by norm_num)) },
    shards := alloc.vec.Vec.from (List.replicate 256 compositionControlRegion)
      (compositionControlRepBound 256 compositionControlRegion (by norm_num)),
    len := 0#usize, key := (),
    locator := { profile := none, directory := none, links := alloc.vec.Vec.new _, staged := none } }

/-- The same table with every direct slot vacant. -/
def compositionControlVacantTable : WordTable StoredI64 :=
  { default := 7#i64, home := (),
    resident := { lanes := 1#usize, slots := alloc.vec.Vec.from (List.replicate 256 compositionControlVacantSlot) (compositionControlRepBound 256 compositionControlVacantSlot (by norm_num)) },
    shards := alloc.vec.Vec.from (List.replicate 256 compositionControlRegion)
      (compositionControlRepBound 256 compositionControlRegion (by norm_num)),
    len := 0#usize, key := (),
    locator := { profile := none, directory := none, links := alloc.vec.Vec.new _, staged := none } }

/-- One tag slot, one payload slot and a single zero summary word: a negative recorded bit. -/
def compositionControlZeroSummaryRegion : table.Region StoredI64 I64 :=
  { tags := alloc.vec.Vec.from [0#u8] (compositionControlRepBound 1 (0#u8) (by norm_num))
    slots := alloc.vec.Vec.from
      [(cell.Sigma.Parabolic, ({ key := 5#i64 } : StoredI64), 42#i64)]
      (compositionControlRepBound 1 _ (by norm_num))
    summary := alloc.vec.Vec.from [0#u64] (compositionControlRepBound 1 (0#u64) (by norm_num)) }

/-- The generated address of the fixture query `(Parabolic, 5)`. -/
def compositionControlStoredAddress : table.TableAddress :=
  match (Apeiron.GrowableIdentity.wordHomeInst.address () cell.Sigma.Parabolic 5#i64).match with
  | .ok address => address
  | _ => { place := 0#u64, direct_word := none }

/-- The generated fingerprint tag `table.tag` of the fixture query place. -/
def compositionControlStoredTag : U8 :=
  match (table.tag compositionControlStoredAddress.place).match with
  | .ok tag => tag
  | _ => 0#u8

/-- A 32-slot routed region with the query's generated tag at one lane and the query payload in
every slot; the empty summary is the conservative `ok true`. -/
def compositionControlSpillRegion : table.Region StoredI64 I64 :=
  { tags := alloc.vec.Vec.from
      (List.replicate 31 0#u8 ++ [compositionControlStoredTag])
      (compositionControlTagBound 31 (0#u8) compositionControlStoredTag (by norm_num))
    slots := alloc.vec.Vec.from
      (List.replicate 32 (cell.Sigma.Parabolic, ({ key := 5#i64 } : StoredI64), 42#i64))
      (compositionControlRepBound 32 _ (by norm_num))
    summary := alloc.vec.Vec.new _ }

/-- A `lanes = 0` stored table whose 256 routed shards all carry the spill fixture region, so the
public read reaches `spill_read` on whichever shard the generated address selects. -/
def compositionControlSpillTable : WordTable StoredI64 :=
  { default := 7#i64, home := (),
    resident := { lanes := 0#usize, slots := alloc.vec.Vec.new _ },
    shards := alloc.vec.Vec.from (List.replicate 256 compositionControlSpillRegion)
      (compositionControlRepBound 256 compositionControlSpillRegion (by norm_num)),
    len := 0#usize, key := (),
    locator := { profile := none, directory := none, links := alloc.vec.Vec.new _, staged := none } }

-- Generated address is total and carries a direct word for the fixture query.
#guard match (Apeiron.GrowableIdentity.wordHomeInst.address () cell.Sigma.Parabolic 5#i64).match with
  | .ok address => address.direct_word.isSome
  | _ => false

-- Positive: the public read returns the live direct value at the full query.
#guard match (word_stored_table_get_at compositionControlLiveTable cell.Sigma.Parabolic 5#i64).match with
  | .ok v => v == 42#i64
  | _ => false

-- Negative: a wrong key, a wrong `Sigma`, and a vacant slot all return the default.
#guard match (word_stored_table_get_at compositionControlLiveTable cell.Sigma.Parabolic 6#i64).match with
  | .ok v => v == 7#i64
  | _ => false

#guard match (word_stored_table_get_at compositionControlLiveTable cell.Sigma.Hyperbolic 5#i64).match with
  | .ok v => v == 7#i64
  | _ => false

#guard match (word_stored_table_get_at compositionControlVacantTable cell.Sigma.Parabolic 5#i64).match with
  | .ok v => v == 7#i64
  | _ => false

-- `entryAt` agrees with the direct route: live slot positive, vacant and out-of-range negatives.
#guard match (Apeiron.GrowableSearch.entryAt compositionControlLiveTable
    Apeiron.GrowableSearch.decodeStoredOption (.flat 0#usize)) with
  | some (_, value) => value == 42#i64
  | none => false

#guard match (Apeiron.GrowableSearch.entryAt compositionControlVacantTable
    Apeiron.GrowableSearch.decodeStoredOption (.flat 0#usize)) with
  | some _ => false
  | none => true

#guard match (Apeiron.GrowableSearch.entryAt compositionControlLiveTable
    Apeiron.GrowableSearch.decodeStoredOption (.flat 256#usize)) with
  | some _ => false
  | none => true

-- `Flat.index` in-range positive and out-of-range (lane `1` of `lanes = 1`) negative.
#guard match (table.Flat.index compositionControlLiveTable.resident (0#u32)).match with
  | .ok o => o.isSome
  | _ => false

#guard match (table.Flat.index compositionControlLiveTable.resident (256#u32)).match with
  | .ok o => o.isNone
  | _ => false

-- Summary words: the empty summary is conservative `true`, the single zero word is `false`.
#guard match (table.Region.summary_maybe compositionControlRegion 0#u64).match with
  | .ok b => b
  | _ => false

#guard match (table.Region.summary_maybe compositionControlZeroSummaryRegion 0#u64).match with
  | .ok b => !b
  | _ => false

-- Routed spill read: positive query hit at lane 31, negative wrong key, wrong `Sigma`, wrong place.
#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep compositionControlSpillRegion
    cell.Sigma.Parabolic
    { place := compositionControlStoredAddress.place, direct_word := none } 5#i64 16#usize).match with
  | .ok (some slot) => slot == 31#usize
  | _ => false

#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep compositionControlSpillRegion
    cell.Sigma.Parabolic
    { place := compositionControlStoredAddress.place, direct_word := none } 6#i64 16#usize).match with
  | .ok none => true
  | _ => false

#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep compositionControlSpillRegion
    cell.Sigma.Hyperbolic
    { place := compositionControlStoredAddress.place, direct_word := none } 5#i64 16#usize).match with
  | .ok none => true
  | _ => false

#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep compositionControlSpillRegion
    cell.Sigma.Parabolic { place := 256#u64, direct_word := none } 5#i64 16#usize).match with
  | .ok none => true
  | _ => false

-- The same spill route through the public read: positive at the routed place, default otherwise.
#guard match (word_stored_table_get_at compositionControlSpillTable cell.Sigma.Parabolic 5#i64).match with
  | .ok v => v == 42#i64
  | _ => false

#guard match (word_stored_table_get_at compositionControlSpillTable cell.Sigma.Parabolic 6#i64).match with
  | .ok v => v == 7#i64
  | _ => false

#guard match (word_stored_table_get_at compositionControlSpillTable cell.Sigma.Hyperbolic 5#i64).match with
  | .ok v => v == 7#i64
  | _ => false

-- Local fail-closed cone check
open Lean Elab Command in
run_cmd do
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  let localNames : Array Lean.Name := #[
    `valid_n_16,
    `word_stored_table_get_at_direct_none_hit,
    `word_stored_table_get_at_flat_none_hit,
    `word_stored_table_get_at_flat_occupied_miss_hit,
    `word_stored_table_get_at_direct_none_sweep_none,
    `word_stored_table_get_at_flat_none_sweep_none,
    `word_stored_table_get_at_flat_occupied_miss_sweep_none,
    `word_stored_table_get_at_root,
    `compositionControlRegion,
    `compositionControlSlot,
    `compositionControlVacantSlot,
    `compositionControlLiveTable,
    `compositionControlVacantTable,
    `compositionControlZeroSummaryRegion,
    `compositionControlStoredAddress,
    `compositionControlStoredTag,
    `compositionControlSpillRegion,
    `compositionControlSpillTable
  ]
  for decl in localNames.map (Lean.Name.append `Apeiron.GrowableRead) do
    let axioms ← liftTermElabM <| Lean.collectAxioms decl
    let bad := axioms.filter (fun ax => !allowed.contains ax)
    unless bad.isEmpty do
      throwError m!"'{decl}' depends on axioms outside the closed cone: {bad.toList}"
    logInfo m!"'{decl}' depends on axioms: {axioms.toList}"

end Apeiron.GrowableRead
