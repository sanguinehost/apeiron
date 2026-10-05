-- Row: `p0-readloop-green-the-so-HieItT7wKt4SKVXrMBgWgF` (P0 · READLOOP: green the
-- corpus so `AxiomAudit.olean` exists and its rows evaluate). With `Growable/Read` green (`d6197bc`)
-- and the `Flux.Discrete` root registered, `lake build AxiomAudit` is exit 0 (9089 jobs, zero
-- `sorryAx`). `SpillCoordPayloadSync` is threaded here as the one named premise the composed read
-- carries — the same premise `b0191c8` landed on the read path, not a new weakening.
import Growable.Read
import Lean

/-!
# Root composition of the public read function: `word_stored_table_get_at`

Completes row 6 of the read map.
Assembles all landed partitions of `table.Table.get_at` under the linear spill probe:
1. Direct hit (`word_stored_table_get_at_direct_hit`)
2. Vacant flat slot (`word_stored_table_get_at_flat_vacant`)
3. Direct word none with probe miss (`word_stored_table_get_at_direct_none`)
4. Direct word none with probe hit (`word_stored_table_get_at_direct_none_hit`)
5. Flat index none with probe miss (`word_stored_table_get_at_flat_none`)
6. Flat index none with probe hit (`word_stored_table_get_at_flat_none_hit`)
7. Flat occupied miss with probe miss (`word_stored_table_get_at_flat_occupied_miss_sweep_none`)
8. Flat occupied miss with probe hit (`word_stored_table_get_at_flat_occupied_miss_hit`)

The coordinate lane is live: `table.Region` carries `slots`, `coords`, `bits` and `live`, and
`table.probe.spill_read` reads the `coords` word and compares it with `table.Region.coord_of`, so no
tag/summary lane remains.

The parameter `table.valid_n 16#usize = ok true` is proved directly by `Growable.Read`'s
`valid_n_16` using `rfl` after unfolding generated definitions, so it discharges unconditionally.
Every probe miss carries the chain-placement premise `hchain` (a vacant coordinate before a live
full match on the cycle makes the miss unsound; see `Growable.Read`'s module note). Every probe hit
carries the named lane-agreement premise `SpillCoordPayloadSync region` (the coordinate and payload
lanes select the same occupied indexes), because `entryAt` tests the payload tag while the probe
reads the coordinate: this is the premise `Growable.Read`'s `spill_read_some_entryAt` consumes, and
the hit route constructors carry it rather than discharging it.
-/

namespace Apeiron.GrowableRead

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

/-- Partition 4 (probe hit), direct word none: the public read returns the spill slot value
at a positive linear-probe hit, which denotes the canonical `Apeiron.Table.get`. No readiness
premise is needed: the run itself witnesses each executed step. -/
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
    (slot : Usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok (some slot))
    (hsync : SpillCoordPayloadSync region) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  obtain ⟨entry, hslotIdx, _hsigma, _hheld, hentryAt⟩ := spill_read_some_entryAt state shardAt
    region hregion sigma address key 16#usize slot hresult hsync
  have hobs := hreach.observations.1 (.spill shardAt slot) (sigma, key) entry.value hentryAt
  have hrun : word_stored_table_get_at state sigma key = ok entry.value := by
    have hvec : state.shards.index_usize shardAt = ok region := by
      simp [alloc.vec.Vec.index_usize, hregion]
    have hslotU : region.slots.index_usize slot = ok entry := by
      simpa [alloc.vec.Vec.index_slice_index] using hslotIdx
    have hshard' : table.shard { place := address.place, direct_word := none } = ok shardAt := by
      rw [← hword]; exact hshard
    have hresult' : table.probe.spill_read
        (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
          I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple core.clone.CloneI64
          table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
        region sigma { place := address.place, direct_word := none } key 16#usize =
        ok (some slot) := by
      rw [← hword]; exact hresult
    have hres : table.Table.resident_exact
        table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
        (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
          I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple core.clone.CloneI64
          table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
        state sigma address key = ok table.ResidentRead.Miss := by
      simp [table.Table.resident_exact, hword]
    simp only [word_stored_table_get_at]
    unfold table.Table.get_at
    have haddr : table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address state.home sigma key = ok address := haddress
    simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddr, hword, hres]
    simp [table.Table.get_spill, table.Table.get_exact_spill,
      core.option.Option.Insts.CoreOpsTry_traitTry.branch,
      hshard', hvec, hresult', hslotU]
  rw [hrun, hobs]

/-- Partition 6 (probe hit), flat index none: the public read returns the spill slot value
at a positive linear-probe hit, which denotes the canonical `Apeiron.Table.get`. No readiness
premise is needed: the run itself witnesses each executed step. -/
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
    (slot : Usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok (some slot))
    (hsync : SpillCoordPayloadSync region) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  obtain ⟨entry, hslotIdx, _hsigma, _hheld, hentryAt⟩ := spill_read_some_entryAt state shardAt
    region hregion sigma address key 16#usize slot hresult hsync
  have hobs := hreach.observations.1 (.spill shardAt slot) (sigma, key) entry.value hentryAt
  have hrun : word_stored_table_get_at state sigma key = ok entry.value := by
    have hvec : state.shards.index_usize shardAt = ok region := by
      simp [alloc.vec.Vec.index_usize, hregion]
    have hslotU : region.slots.index_usize slot = ok entry := by
      simpa [alloc.vec.Vec.index_slice_index] using hslotIdx
    have hshard' : table.shard { place := address.place, direct_word := some word } = ok shardAt := by
      rw [← hword]; exact hshard
    have hresult' : table.probe.spill_read
        (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
          I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple core.clone.CloneI64
          table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
        region sigma { place := address.place, direct_word := some word } key 16#usize =
        ok (some slot) := by
      rw [← hword]; exact hresult
    have hres : table.Table.resident_exact
        table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
        (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
          I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple core.clone.CloneI64
          table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
        state sigma address key = ok table.ResidentRead.Miss := by
      simp [table.Table.resident_exact, hword, hflat]
    simp only [word_stored_table_get_at]
    unfold table.Table.get_at
    have haddr : table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address state.home sigma key = ok address := haddress
    simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddr, hword, hres]
    simp [table.Table.get_spill, table.Table.get_exact_spill,
      core.option.Option.Insts.CoreOpsTry_traitTry.branch,
      hshard', hvec, hresult', hslotU]
  rw [hrun, hobs]

/-- Partition 8 (probe hit), occupied non-match: the public read returns the spill slot value
at a positive linear-probe hit, which denotes the canonical `Apeiron.Table.get`. No readiness
premise is needed: the run itself witnesses each executed step. -/
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
    (slot : Usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok (some slot))
    (hsync : SpillCoordPayloadSync region) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  obtain ⟨entry, hslotIdx, _hsigma, _hheld, hentryAt⟩ := spill_read_some_entryAt state shardAt
    region hregion sigma address key 16#usize slot hresult hsync
  have hobs := hreach.observations.1 (.spill shardAt slot) (sigma, key) entry.value hentryAt
  have hrun : word_stored_table_get_at state sigma key = ok entry.value := by
    have hidx : alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot StoredI64 I64))
        state.resident.slots index = ok flat := by
      simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hslot]
    have hvec : state.shards.index_usize shardAt = ok region := by
      simp [alloc.vec.Vec.index_usize, hregion]
    have hslotU : region.slots.index_usize slot = ok entry := by
      simpa [alloc.vec.Vec.index_slice_index] using hslotIdx
    have hshard' : table.shard { place := address.place, direct_word := some word } = ok shardAt := by
      rw [← hword]; exact hshard
    have hresult' : table.probe.spill_read
        (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
          I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple core.clone.CloneI64
          table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
        region sigma { place := address.place, direct_word := some word } key 16#usize =
        ok (some slot) := by
      rw [← hword]; exact hresult
    have hres : table.Table.resident_exact
        table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
        (table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
          I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple core.clone.CloneI64
          table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma)
        state sigma address key = ok table.ResidentRead.Miss := by
      rcases hmismatch with hσ | ⟨hσ, hk⟩
      · simp only [table.Table.resident_exact, hword, hflat, hidx, hlive, hσ,
          Bool.false_eq_true, ↓reduceIte, bind_tc_ok]
      · have hk' : table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.matches
            I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple core.clone.CloneI64
            table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma sigma flat.held key address = ok false := hk
        simp only [table.Table.resident_exact, hword, hflat, hidx, hlive, hσ,
          Bool.false_eq_true, ↓reduceIte, bind_tc_ok]
        rw [hk']
        simp only [bind_tc_ok, Bool.false_eq_true, ↓reduceIte]
    simp only [word_stored_table_get_at]
    unfold table.Table.get_at
    have haddr : table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address state.home sigma key = ok address := haddress
    simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddr, hword, hres]
    simp [table.Table.get_spill, table.Table.get_exact_spill,
      core.option.Option.Insts.CoreOpsTry_traitTry.branch,
      hshard', hvec, hresult', hslotU]
  rw [hrun, hobs]

/-- Partition 3 (probe miss), direct word none: a linear-probe miss returns default,
which denotes the canonical absent `Apeiron.Table.get`. Contract change: the explicit `hchain`
premise is required for the miss; without it a zero tag before a live full match on the cycle
refutes it. -/
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
    (hchain : Apeiron.GrowableReadLoop.StoredProbeChainSafe region address.place sigma key)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok none) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  exact word_stored_table_get_at_direct_none state direct live logical hreach hstate sigma key
    address haddress hword shardAt hshard region hregion hchain hresult

/-- Partition 5 (probe miss), flat index none: a linear-probe miss returns default,
which denotes the canonical absent `Apeiron.Table.get`. Contract change: the explicit `hchain`
premise is required for the miss; without it a zero tag before a live full match on the cycle
refutes it. -/
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
    (hchain : Apeiron.GrowableReadLoop.StoredProbeChainSafe region address.place sigma key)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok none) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  exact word_stored_table_get_at_flat_none state direct live logical hreach sigma key address haddress
    word hword hflat hlane shardAt hshard region hregion hstate hchain hresult

/-- Partition 7 (probe miss), occupied non-match: a linear-probe miss returns default,
which denotes the canonical absent `Apeiron.Table.get`. Contract change: the explicit `hchain`
premise is required for the miss; without it a zero tag before a live full match on the cycle
refutes it. Delegates to `word_stored_table_get_at_flat_occupied_miss`. -/
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
    (hchain : Apeiron.GrowableReadLoop.StoredProbeChainSafe region address.place sigma key)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok none)
    (hroute : Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = some index.val ∨
      ∀ idx value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.flat idx) ≠ some ((sigma, key), value)) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  exact word_stored_table_get_at_flat_occupied_miss state direct live logical hreach sigma key
    address haddress word hword index flat hflat hslot hlive hmismatch shardAt hshard region hregion
    hstate hchain hresult hroute

/-- The execution route of a generated `table.Table.get_at` call partitioned over the control flow.
Every miss route is a generated linear-probe `ok none`; the retired summary lanes have no route.
Contract change: every probe miss carries the chain-placement premise `hchain` (a zero tag before a
live full match on the cycle makes the miss unsound). -/
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
        { tag := 0#u8, sigma := flat.sigma, held := flat.held, value := flat.value }
        sigma key address = ok true)
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
  | directNoneSweepNone (shardAt : Usize) (region : table.Region StoredI64 I64)
      (hword : address.direct_word = none)
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hchain : Apeiron.GrowableReadLoop.StoredProbeChainSafe region address.place sigma key)
      (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key 16#usize = ok none)
  | directNoneSweepHit (shardAt : Usize) (region : table.Region StoredI64 I64)
      (slot : Usize)
      (hword : address.direct_word = none)
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key 16#usize = ok (some slot))
      (hsync : SpillCoordPayloadSync region)
  | flatNoneSweepNone (word : U32) (shardAt : Usize) (region : table.Region StoredI64 I64)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok none)
      (hlane : state.resident.lanes.val ≤ word.val / 256 ∨
        ∀ index value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
          (.flat index) ≠ some ((sigma, key), value))
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hchain : Apeiron.GrowableReadLoop.StoredProbeChainSafe region address.place sigma key)
      (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key 16#usize = ok none)
  | flatNoneSweepHit (word : U32) (shardAt : Usize) (region : table.Region StoredI64 I64)
      (slot : Usize)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok none)
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key 16#usize = ok (some slot))
      (hsync : SpillCoordPayloadSync region)
  | flatOccupiedMissSweepNone (word : U32) (index : Usize) (flat : table.FlatSlot StoredI64 I64)
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
      (hchain : Apeiron.GrowableReadLoop.StoredProbeChainSafe region address.place sigma key)
      (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key 16#usize = ok none)
      (hroute : Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = some index.val ∨
        ∀ idx value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
          (.flat idx) ≠ some ((sigma, key), value))
  | flatOccupiedMissSweepHit (word : U32) (index : Usize) (flat : table.FlatSlot StoredI64 I64)
      (shardAt : Usize) (region : table.Region StoredI64 I64)
      (slot : Usize)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok (some index))
      (hslot : state.resident.slots.val[index.val]? = some flat)
      (hlive : flat.live = true)
      (hmismatch : (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok false) ∨
        (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true ∧
         Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok false))
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key 16#usize = ok (some slot))
      (hsync : SpillCoordPayloadSync region)

/-- Deliverable 2: Root composition of the public read function `word_stored_table_get_at`
under constructor-reachable `RepR` and the live tag/address invariant. Assembles every landed
partition (direct hit, direct and occupied emptiness, and the complete linear-probe sweep). The
probe miss routes carry the chain-placement premise `hchain` as a `StoredReadRoute` field. -/
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
  | directNoneSweepNone shardAt region hword hshard hregion hchain hresult =>
      exact word_stored_table_get_at_direct_none_sweep_none state direct live logical hreach hstate sigma key address haddress hword shardAt hshard region hregion hchain hresult
  | directNoneSweepHit shardAt region slot hword hshard hregion hresult hsync =>
      exact word_stored_table_get_at_direct_none_hit state direct live logical hreach sigma key address haddress hword shardAt hshard region hregion slot hresult hsync
  | flatNoneSweepNone word shardAt region hword hflat hlane hshard hregion hchain hresult =>
      exact word_stored_table_get_at_flat_none_sweep_none state direct live logical hreach hstate sigma key address haddress word hword hflat hlane shardAt hshard region hregion hchain hresult
  | flatNoneSweepHit word shardAt region slot hword hflat hshard hregion hresult hsync =>
      exact word_stored_table_get_at_flat_none_hit state direct live logical hreach sigma key address haddress word hword hflat shardAt hshard region hregion slot hresult hsync
  | flatOccupiedMissSweepNone word index flat shardAt region hword hflat hslot hlive hmismatch hshard hregion hchain hresult hroute =>
      exact word_stored_table_get_at_flat_occupied_miss_sweep_none state direct live logical hreach hstate sigma key address haddress word hword index flat hflat hslot hlive hmismatch shardAt hshard region hregion hchain hresult hroute
  | flatOccupiedMissSweepHit word index flat shardAt region slot hword hflat hslot hlive hmismatch hshard hregion hresult hsync =>
      exact word_stored_table_get_at_flat_occupied_miss_hit state direct live logical hreach sigma key address haddress word hword index flat hflat hslot hlive hmismatch shardAt hshard region hregion slot hresult hsync

/-!
## Concrete composition fixtures and paired read controls

`word_stored_table_get_at_root` and its route theorems all carry `ConstructorReachableRepR`
at the query state. No concrete value of that premise is available in this environment: the only
landed producer, `Growable.GrowableRepresentation.word_stored_table_new_repR`, is a WP triple over
the generated `table.Table.new` loop whose state and `shards` list are exposed only through the
private `Growable.GrowableRepresentation.table_new_empty_repR`, and that empty state carries
`direct = fun _ => none`, `live = fun _ => False`; non-empty reachability is row 4/8 of
the read-proof note, open. The route and root theorems therefore cannot be instantiated at a
concrete state here. The fixtures below exercise the same generated operations the route theorems
partition — generated address, `table.Flat.index`, flat liveness/`Sigma`/matcher dispatch, the
linear `table.probe.spill_read` and `entryAt` — at concrete stored-carrier states, each positive
paired with a rejecting negative. `Result I64` has no `BEq` in this environment, so each check
matches on `Result.match` (`MatchResult`).
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

private def compositionControlTagPairBound {α : Type} (n : Nat) (x t y : α) (h : n + 2 ≤ 1000) :
    (List.replicate n x ++ [t, y]).length ≤ Usize.max := by
  have hlen : (List.replicate n x ++ [t, y]).length = n + 2 := by simp
  rw [hlen]
  exact compositionControlNatBound (n + 2) h

/-- An empty spill region: no slots, no live entries. Its linear probe answers `ok none`. -/
def compositionControlRegion : table.Region StoredI64 I64 :=
  { slots := alloc.vec.Vec.new _, coords := alloc.vec.Vec.new _, bits := 0#u32, live := 0#usize }

/-- A live direct flat slot holding `(Parabolic, 5) ↦ 42`. -/
def compositionControlSlot : table.FlatSlot StoredI64 I64 :=
  { live := true, sigma := cell.Sigma.Parabolic, held := { key := 5#i64 }, value := 42#i64 }

/-- A vacant direct flat slot with the same attempted contents. -/
def compositionControlVacantSlot : table.FlatSlot StoredI64 I64 :=
  { live := false, sigma := cell.Sigma.Parabolic, held := { key := 5#i64 }, value := 42#i64 }

/-- A `lanes = 1` stored table whose 256 direct slots are all live `(Parabolic, 5) ↦ 42`. -/
def compositionControlLiveTable : WordTable StoredI64 :=
  { default := 7#i64, home := (),
    resident := { lanes := 1#usize, lane_bits := 0#u32, slots := alloc.vec.Vec.from (List.replicate 256 compositionControlSlot) (compositionControlRepBound 256 compositionControlSlot (by norm_num)) },
    shards := alloc.vec.Vec.from (List.replicate 256 compositionControlRegion)
      (compositionControlRepBound 256 compositionControlRegion (by norm_num)),
    len := 0#usize, key := (),
    locator := { profile := none, directory := none, links := alloc.vec.Vec.new _, admitted := 0#usize, staged := none } }

/-- The same table with every direct slot vacant. -/
def compositionControlVacantTable : WordTable StoredI64 :=
  { default := 7#i64, home := (),
    resident := { lanes := 1#usize, lane_bits := 0#u32, slots := alloc.vec.Vec.from (List.replicate 256 compositionControlVacantSlot) (compositionControlRepBound 256 compositionControlVacantSlot (by norm_num)) },
    shards := alloc.vec.Vec.from (List.replicate 256 compositionControlRegion)
      (compositionControlRepBound 256 compositionControlRegion (by norm_num)),
    len := 0#usize, key := (),
    locator := { profile := none, directory := none, links := alloc.vec.Vec.new _, admitted := 0#usize, staged := none } }

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

/-- The empty-tag spill slot fixture payload: `(Parabolic, 5) ↦ 42` in the query's slot layout. -/
def compositionControlSpillSlot : table.SpillSlot StoredI64 I64 :=
  { tag := 0#u8, sigma := cell.Sigma.Parabolic, held := { key := 5#i64 }, value := 42#i64 }

/-- The generated coordinate word `table.Region.coord_of` of the fixture query place. -/
def compositionControlSpillWanted : U64 :=
  match (table.Region.coord_of StoredI64 I64 compositionControlStoredAddress.place).match with
  | .ok wanted => wanted
  | _ => table.COORD_EMPTY

/-- A 32-slot routed region: the query payload occupies lane 30, the remaining 31 slots are the
empty-tag fixture payload; the coordinate lane carries the query's `coord_of` word at lane 30 and
`COORD_EMPTY` elsewhere. `bits := 15` is chosen so the generated `table.Region.home` of the fixture
place (the `coord_of * HOME_SPREAD >>> (64 - bits)` branch, since the coordinate word exceeds the
coordinate-vector length) lands on lane 30; the generated coordinate probe therefore reads the
query coordinate at its first step and hits. A lane with `COORD_EMPTY` terminates the probe. -/
def compositionControlSpillRegion : table.Region StoredI64 I64 :=
  { slots := alloc.vec.Vec.from
      (List.replicate 30 compositionControlSpillSlot ++
        [{ compositionControlSpillSlot with tag := compositionControlStoredTag },
         compositionControlSpillSlot])
      (compositionControlTagPairBound 30 compositionControlSpillSlot
        { compositionControlSpillSlot with tag := compositionControlStoredTag }
        compositionControlSpillSlot (by norm_num))
    coords := alloc.vec.Vec.from
      (List.replicate 30 table.COORD_EMPTY ++
        [compositionControlSpillWanted, table.COORD_EMPTY])
      (compositionControlTagPairBound 30 table.COORD_EMPTY compositionControlSpillWanted
        table.COORD_EMPTY (by norm_num))
    bits := 15#u32
    live := 32#usize }

/-- A `lanes = 0` stored table whose 256 routed shards all carry the spill fixture region, so the
public read reaches the linear `spill_read` on whichever shard the generated address selects. -/
def compositionControlSpillTable : WordTable StoredI64 :=
  { default := 7#i64, home := (),
    resident := { lanes := 0#usize, lane_bits := 0#u32, slots := alloc.vec.Vec.new _ },
    shards := alloc.vec.Vec.from (List.replicate 256 compositionControlSpillRegion)
      (compositionControlRepBound 256 compositionControlSpillRegion (by norm_num)),
    len := 0#usize, key := (),
    locator := { profile := none, directory := none, links := alloc.vec.Vec.new _, admitted := 0#usize, staged := none } }

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

-- Linear probe: the empty region answers `none`; the spill fixture hits at lane 31.
#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep compositionControlRegion
    cell.Sigma.Parabolic { place := 0#u64, direct_word := none } 5#i64 16#usize).match with
  | .ok none => true
  | _ => false

-- Routed spill read: positive query hit at the generated home lane 30, negative wrong key, wrong
-- `Sigma`, wrong place.
#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep compositionControlSpillRegion
    cell.Sigma.Parabolic
    { place := compositionControlStoredAddress.place, direct_word := none } 5#i64 16#usize).match with
  | .ok (some slot) => slot == 30#usize
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
    `compositionControlStoredAddress,
    `compositionControlStoredTag,
    `compositionControlSpillSlot,
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
