import Growable.Read
import Lean

/-!
# Root composition of the public read function: `word_stored_table_get_at`

Completes row 6 of the read map.
Assembles all landed partitions of `table.Table.get_at`:
1. Direct hit (`word_stored_table_get_at_direct_hit`)
2. Vacant flat slot (`word_stored_table_get_at_flat_vacant`)
3. Direct word none, routed spill `none` (`word_stored_table_get_at_direct_none_sweep_none`)
4. Direct word none, routed spill hit (`word_stored_table_get_at_direct_none_hit`)
5. Flat index none, routed spill `none` (`word_stored_table_get_at_flat_none_sweep_none`)
6. Flat index none, routed spill hit (`word_stored_table_get_at_flat_none_hit`)
7. Occupied non-match, routed spill `none`
   (`word_stored_table_get_at_flat_occupied_miss_sweep_none`)
8. Occupied non-match, routed spill hit (`word_stored_table_get_at_flat_occupied_miss_hit`)

The parameter `table.valid_n 16#usize = ok true` is proved directly by `valid_n_16` using `rfl`
after unfolding generated definitions, so it discharges unconditionally.

## What the flat spill port changed

The flat spill port merged `Region.tags` and the occupancy summary into one `Region.slots`
array of `table.SpillSlot` records (`{tags, slots, summary}` → `{slots, coords, bits, live}`),
so the routed read is one bounded linear probe cycle over the capacity
(`proof/ApeironWord.lean:17485-17527`) and the grouped sweep (`WINDOWS + groups`, a
power-of-two mask over `groups - 1`) has no denotation.  Every premise this module passed to
the retired grouped sweep is restated over the current body or retired with its replacement
named.  Nothing is silently dropped and no name is kept under a shape the current body cannot
satisfy.

| stale site | disposition | replacement |
|---|---|---|
| `Apeiron.GrowableReadLoop.GroupedReadAdmission` (13 sites) | **obsolete, not restated** | its five conjuncts were the grouped sizing `region.slots.length = groups.val * n.val`, a power-of-two `groups`, `2 ≤ groups.val`, a `groups - 1` mask and the positive summary word. The current body admits on `alloc.vec.Vec.len region.slots ≠ 0#usize` and `table.valid_n n` and takes its bound from `alloc.vec.Vec.len region.slots` itself (`table.Region.bound` is that length by definition, `proof/ApeironWord.lean:17412-17414`). The admission chain is `SpillReadHitAdmission` below, which is the premise list of `Apeiron.GrowableReadLoop.generated_spill_read_branch`; the divisor obligation survives as its `valid` field, discharged by `valid_n_16` |
| `hpowCheck : core.num.Usize.is_power_of_two groups = ok true`, `hgroups2 : 2 ≤ groups.val`, and the `groups`/`mask`/`limit` binders (13 sites) | **obsolete, not restated** | no power-of-two and no group bound is a premise of the current body: the mask is `alloc.vec.Vec.len region.slots - 1#usize` and the cycle runs to that length. `SpillReadHitAdmission.reads` carries the two reads of every executed lane, quantified over whichever `bound`/`mask` `table.Region.bound`/`bound - 1` produce |
| `htagSlots : region.tags.length = region.slots.length` from `hreach.directory.2.2 region hmem` (13 sites) | **obsolete, not restated** | there is one payload array, so the length agreement is tautological. `Apeiron.GrowableRepresentation.SpillDirectoryConsistent`'s third conjunct is now `∀ region ∈ state.shards.val, Apeiron.GrowableScan.SpillPlacement region` (`Representation.lean:102-106`), which has no length agreement to project. The tag fact it supported is now region-level occupancy, carried as the explicit premise `SpillRegionOccupied region`; see `OPS-5` |
| `region.tags` / `htagNonzero` (the tag half of the retired `spill_read_some_tagged_full_match`) | **restated** | the occupancy marker is the slot record's own `tag`, supplied by `Apeiron.GrowableRead.SpillRegionOccupied region`. The conclusion keeps its shape with one extra binder (`∃ value tag, …`) |
| `obtain ⟨value, hentryAt, hslotIdx⟩ := spill_read_some_entryAt …` (3 sites) | **restated** | the new conclusion is `∃ (value : I64) (tag : U8), hentryAt ∧ hindex`; the call is `obtain ⟨value, tag, hentryAt, hindex⟩` and the premise list is `SpillReadHitAdmission` plus `SpillRegionOccupied` |
| `Apeiron.GrowableReadLoop.spill_read_none_entryAt_ne` (3 sites) | **retired; the obligation it delivered is recorded as OPEN** | it concluded `∀ index value, entryAt (.spill shardAt index) ≠ some ((sigma, key), value)` from a `none` run. That needs the whole shard read by the run, which `OPS-1`/`OPS-3` in `Growable.Read`'s header leave open (`table.Region.home` may return an index at or above the capacity). The routed-spill **exclusion** is now the explicit premise `hspillExcl`, which is also `Apeiron.GrowableRead.SpillReadNoneSound.excludes`; recorded as `OPS-6` |
| `hsummary : region.summary_maybe address.place = ok false` (3 route arms, 2 `#guard`s) | **obsolete, not restated** | `table.Region.summary_maybe` is absent from the extraction. The three `*SummaryMiss` route arms are merged into their `*SweepNone` counterparts, which now differ only by fields that have a denotation. The two summary `#guard`s are replaced by the current admission-gate check `compositionControlEmptyRegionAnswersNone` and the coordinate-branch fixture guards |
| `StoredReadRoute.directHit.hmatch` | **deleted as a premise and derived** | `Apeiron.GrowableRead.word_stored_table_get_at_direct_hit` derives the record-shaped `table.matches` equation from `hsigmaEq` and `hraw` (`stored_flat_match_unfold`); `table.Table.resident_exact` calls the carrier matcher on the flat slot's own `held` and never builds a `table.SpillSlot` (`proof/ApeironWord.lean:17295-17326`). One hypothesis fewer, same conclusion |
| `hstate : Apeiron.GrowableReadLoop.StoredLiveTagAddress state` (4 sites, incl. `word_stored_table_get_at_root`) | **removed from every theorem in this module** | the tag law existed to read a `U8` back out of the parallel `region.tags` array. The occupancy authority is now the slot record's own `tag`, supplied by `SpillRegionOccupied`, and the routed-spill exclusion is a premise. `word_stored_table_get_at_root` therefore has **one premise fewer** than the pre-image: a strengthening, recorded here rather than hidden |

### Open items this module does not close

* **`OPS-5` (opened here).** `Apeiron.GrowableRead.SpillRegionOccupied region` is an explicit
  premise of the three routed-spill-hit partitions and of the three `*SpillHit` route arms. Its
  producer — `Apeiron.GrowableScan.SpillPlacement.liveExact` together with
  `region.live.val = region.slots.val.length` (the shard is full) — is `OPS-4` in `Growable.Read`
  and is not closed here. No `RepR` field supplies the fullness half.
* **`OPS-6` (opened here).** The routed-spill **exclusion**
  `∀ index value, entryAt state decode (.spill shardAt index) ≠ some ((sigma, key), value)` is an
  explicit premise of the three routed-spill-`none` partitions and of the three `*SpillNone` route
  arms. The pre-image derived it from `table.Region.summary_maybe` and
  `Apeiron.GrowableRepresentation.repR_spillSummaryMissSound`; neither exists. This is the same
  obligation as `OPS-3` in `Growable.Read`, restated at the composition boundary; the producer
  route is `Apeiron.GrowableScan.spill_probe_bound_absent_sound`, whose own premise needs `OPS-1`.
* **`OPS-7` (opened here).** The pre-image's three `word_stored_table_get_at
  compositionControlSpillTable` guards are **not restated**. The generated address's `place` for
  `(Parabolic, 5)` is not pinned numerically anywhere in this file, so the fixture region cannot be
  keyed at the *query's own* coordinate lane without pinning it; the routed fixture is therefore
  checked at `table.probe.spill_read` and `Apeiron.GrowableSearch.entryAt` directly, at
  `place := 0#u64`, which is the `Growable.ReadLoop` fixture idiom
  (`ReadLoop.lean:975-1029`). `entryAt` reads no address, so its spill guards are exact.

Full `RepR` preservation across writes and rebuild remains row 4's open part of the read map.
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

/-- **The current admission chain of `table.probe.spill_read`.**  This is the premise list of
`Apeiron.GrowableReadLoop.generated_spill_read_branch`, packaged so the three routed-spill-hit
partitions share one statement.  It replaces `Apeiron.GrowableReadLoop.GroupedReadAdmission`, whose
five grouped-sizing conjuncts the merged schema removed: the current body gates on
`alloc.vec.Vec.len region.slots ≠ 0#usize` (`proof/ApeironWord.lean:17490-17492`), on
`table.valid_n n` (`:17493-17494`), reads `region.coords home` (`:17495-17497`), on a coordinate
hit reads `region.slots home` and calls `table.matches` (`:17503-17510`), and otherwise runs the
bounded cycle over `table.Region.bound region` with mask `bound - 1#usize` (`:17511-17519`).
`reads` is `OPS-2` of `Growable.Read`: the two reads of every executed lane are carried, not
derived from the loop's own recursion. -/
structure SpillReadHitAdmission (region : table.Region StoredI64 I64)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : I64) (n : Usize)
    (home : Usize) (wanted found : U64) (slot : Usize) : Prop where
  /-- The generated `alloc.vec.Vec.len region.slots != 0#usize` gate admits the region. -/
  nonempty : alloc.vec.Vec.len region.slots ≠ 0#usize
  /-- The generated `table.valid_n n` gate admits the routed divisor. -/
  valid : table.valid_n n = ok true
  /-- The computed home lane of the query's place. -/
  homeLane : table.Region.home region address.place = ok home
  /-- The wanted identity coordinate word of the query's place. -/
  wantedCoord : table.Region.coord_of StoredI64 I64 address.place = ok wanted
  /-- The coordinate read at the home lane succeeded. -/
  foundCoord : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
    region.coords home = ok found
  /-- On a coordinate hit at the home lane, the payload read succeeded. -/
  homeSlot : found ≠ table.COORD_EMPTY → found = wanted →
    ∃ ss : StoredSpillSlot, alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice StoredSpillSlot) region.slots home = ok ss
  /-- On a coordinate hit at the home lane, the generated matcher call succeeded. -/
  homeMatch : found ≠ table.COORD_EMPTY → found = wanted →
    ∀ ss : StoredSpillSlot,
      alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice StoredSpillSlot) region.slots home =
        ok ss →
      ∃ matched : Bool,
        table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address = ok matched
  /-- The two reads of every executed cycle lane succeeded, at whichever `bound`/`mask` the
  generated `table.Region.bound region` and `bound - 1#usize` produce. -/
  reads : ∀ (bound mask : Usize), table.Region.bound region = ok bound →
    bound - 1#usize = ok mask →
    Apeiron.GrowableReadLoop.TailReadSuccess Apeiron.GrowableIdentity.storedKeyRep region
      sigma address.place address.direct_word key home bound mask wanted 1#usize

/-- The payload index equation `spill_read_some_entryAt` concludes with. -/
private theorem spillHitValue (state : WordTable StoredI64) (shard : Usize)
    (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (hoccupied : SpillRegionOccupied region)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : I64) (n : Usize)
    (home : Usize) (wanted found : U64) (slot : Usize)
    (adm : SpillReadHitAdmission region sigma address key n home wanted found slot)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok (some slot)) :
    ∃ (value : I64) (tag : U8),
      Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.spill shard slot) = some ((sigma, key), value) ∧
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice StoredSpillSlot)
        region.slots slot =
          ok { tag := tag, sigma := sigma, held := { key := key }, value := value } :=
  spill_read_some_entryAt state shard region hregion hoccupied sigma address key n home wanted found
    slot adm.nonempty adm.valid adm.homeLane adm.wantedCoord adm.foundCoord adm.homeSlot
    adm.homeMatch adm.reads hresult

/-- The generated payload read of a routed-spill hit returns that slot's value.

This is the `get_at → get_spill → get_exact_spill → spill_read → Vec.index region.slots →
ss.value` chain at a positive routed hit, stated once and supplied the same two equations by each
of the three routed-spill-hit partitions, together with their own direct-route `ResidentRead.Miss`
equation.  The body is `proof/ApeironWord.lean:21766-21789` and `17538-17552`. -/
private theorem spillHitReadsValue (state : WordTable StoredI64)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (hresident : table.Table.resident_exact StoredHomeInst StoredKeyRepInst state sigma address key
      = ok table.ResidentRead.Miss)
    (shardAt : Usize) (hshard : table.shard address = ok shardAt)
    (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shardAt.val]? = some region)
    (slot : Usize) (tag : U8) (value : I64)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok (some slot))
    (hindex : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice StoredSpillSlot)
      region.slots slot = ok { tag := tag, sigma := sigma, held := { key := key }, value := value }) :
    word_stored_table_get_at state sigma key = ok value := by
  have haddr : table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address state.home sigma key =
      ok address := haddress
  have hvec : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (table.Region StoredI64 I64))
      state.shards shardAt = ok region := by
    simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hregion]
  simp only [word_stored_table_get_at]
  unfold table.Table.get_at
  simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddr,
    hresident, bind_tc_ok, table.Table.get_spill, table.Table.get_exact_spill,
    bind_tc_ok, hshard, hvec, bind_tc_ok, hresult,
    hindex, core.option.Option.Insts.CoreOpsTry_traitTry.branch,
    core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual,
    core.option.Option.unwrap_or]

/-- **The direct route of the routed-spill-hit partitions reads `ResidentRead.Miss`.**  One
statement of the `table.Table.resident_exact` dispatch: the generated body tests
`address.direct_word` and then `table.Flat.index`, and reports `Miss` on both paths the direct-none
and flat-index-none partitions supply. -/
private theorem spillHitResidentMissNone (state : WordTable StoredI64)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress)
    (hflat : ∀ w : U32, address.direct_word = some w →
      table.Flat.index state.resident w = ok none) :
    table.Table.resident_exact StoredHomeInst StoredKeyRepInst state sigma address key
      = ok table.ResidentRead.Miss := by
  cases hdw : address.direct_word with
  | none =>
      unfold table.Table.resident_exact
      rw [hdw]
  | some word =>
      unfold table.Table.resident_exact
      simp [hdw, hflat word hdw]

/-- Partition 4 (routed spill hit), direct word none: the public read returns the spill slot value
at a positive sweep hit, which denotes the canonical `Apeiron.Table.get`.

The pre-image took `hpowCheck`, `hgroups2` and the `groups`/`mask`/`limit` binders to feed
`spill_read_some_entryAt` through the grouped sweep, and derived the region occupancy from
`hreach.directory.2.2`. The current body takes `SpillReadHitAdmission` and `SpillRegionOccupied`
instead; see the disposition table in the module header. -/
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
    (hoccupied : SpillRegionOccupied region)
    (home : Usize) (wanted found : U64) (slot : Usize)
    (adm : SpillReadHitAdmission region sigma address key 16#usize home wanted found slot)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok (some slot)) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  obtain ⟨value, tag, hentryAt, hindex⟩ :=
    spillHitValue state shardAt region hregion hoccupied sigma address key 16#usize home wanted found
      slot adm hresult
  have hobs := hreach.observations.1 (.spill shardAt slot) (sigma, key) value hentryAt
  have hresident := spillHitResidentMissNone state sigma key address
    (fun _ hw => absurd (hword.symm.trans hw) (by simp))
  rw [spillHitReadsValue state sigma key address haddress hresident shardAt hshard region hregion
    slot tag value hresult hindex, hobs]

/-- Partition 6 (routed spill hit), flat index none: the public read returns the spill slot value
at a positive sweep hit, which denotes the canonical `Apeiron.Table.get`. -/
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
    (hoccupied : SpillRegionOccupied region)
    (home : Usize) (wanted found : U64) (slot : Usize)
    (adm : SpillReadHitAdmission region sigma address key 16#usize home wanted found slot)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok (some slot)) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  obtain ⟨value, tag, hentryAt, hindex⟩ :=
    spillHitValue state shardAt region hregion hoccupied sigma address key 16#usize home wanted found
      slot adm hresult
  have hobs := hreach.observations.1 (.spill shardAt slot) (sigma, key) value hentryAt
  have hresident := spillHitResidentMissNone state sigma key address
    (fun w hw => (Option.some.inj (hw.symm.trans hword)) ▸ hflat)
  rw [spillHitReadsValue state sigma key address haddress hresident shardAt hshard region hregion
    slot tag value hresult hindex, hobs]

/-- Partition 8 (routed spill hit), occupied non-match: the public read returns the spill slot
value at a positive sweep hit, which denotes the canonical `Apeiron.Table.get`. -/
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
    (hoccupied : SpillRegionOccupied region)
    (home : Usize) (wanted found : U64) (slot : Usize)
    (adm : SpillReadHitAdmission region sigma address key 16#usize home wanted found slot)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok (some slot)) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  obtain ⟨value, tag, hentryAt, hindex⟩ :=
    spillHitValue state shardAt region hregion hoccupied sigma address key 16#usize home wanted found
      slot adm hresult
  have hobs := hreach.observations.1 (.spill shardAt slot) (sigma, key) value hentryAt
  have hidx : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot StoredI64 I64))
      state.resident.slots index = ok flat := by
    simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hslot]
  have hresident : table.Table.resident_exact StoredHomeInst StoredKeyRepInst state sigma address
      key = ok table.ResidentRead.Miss := by
    unfold table.Table.resident_exact
    simp only [bind_tc_ok, hword, bind_tc_ok, hflat, bind_tc_ok, hidx, bind_tc_ok, hlive,
      bind_tc_ok]
    rcases hmismatch with hsigmaFalse | ⟨hsigmaTrue, hkeyFalse⟩
    · simp only [hsigmaFalse, bind_tc_ok, ↓reduceIte]
      rfl
    · have hkeyFalse' : table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.matches
          I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple core.clone.CloneI64
          table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma sigma flat.held key address = ok false :=
        hkeyFalse
      simp only [hsigmaTrue, bind_tc_ok, ↓reduceIte, hkeyFalse', bind_tc_ok, ↓reduceIte]
      rfl
  rw [spillHitReadsValue state sigma key address haddress hresident shardAt hshard region hregion
    slot tag value hresult hindex, hobs]

/-- Partition 3 (routed spill `none`), direct word none: the public read returns the default,
which denotes the canonical absent `Apeiron.Table.get`.

The pre-image derived the routed-spill exclusion from
`Apeiron.GrowableReadLoop.spill_read_none_entryAt_ne`. That theorem is retired: the whole-shard
absence it concluded needs every slot read by the run, which `OPS-1`/`OPS-3` in `Growable.Read`
leave open. The exclusion is therefore the explicit premise `hspillExcl` (`OPS-6`), and it is
exactly `Apeiron.GrowableRead.SpillReadNoneSound.excludes`. -/
theorem word_stored_table_get_at_direct_none_sweep_none
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
    (hread : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok none)
    (hspillExcl : ∀ index value,
      Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.spill shardAt index) ≠ some ((sigma, key), value)) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  exact word_stored_table_get_at_direct_none state direct live logical hreach sigma key address
    haddress hword shardAt hshard region hregion ⟨hread, hspillExcl⟩

/-- Partition 5 (routed spill `none`), flat index none: the public read returns the default,
which denotes the canonical absent `Apeiron.Table.get`. -/
theorem word_stored_table_get_at_flat_none_sweep_none
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
    (hlane : state.resident.lanes.val ≤ word.val / 256 ∨
      ∀ index value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.flat index) ≠ some ((sigma, key), value))
    (shardAt : Usize) (hshard : table.shard address = ok shardAt)
    (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shardAt.val]? = some region)
    (hread : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok none)
    (hspillExcl : ∀ index value,
      Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.spill shardAt index) ≠ some ((sigma, key), value)) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  exact word_stored_table_get_at_flat_none state direct live logical hreach sigma key address
    haddress word hword hflat hlane shardAt hshard region hregion ⟨hread, hspillExcl⟩

/-- Partition 7 (routed spill `none`), occupied non-match: the public read returns the default,
which denotes the canonical absent `Apeiron.Table.get`. -/
theorem word_stored_table_get_at_flat_occupied_miss_sweep_none
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
    (hread : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok none)
    (hspillExcl : ∀ index value,
      Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.spill shardAt index) ≠ some ((sigma, key), value))
    (hroute : Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = some index.val ∨
      ∀ idx value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.flat idx) ≠ some ((sigma, key), value)) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  exact word_stored_table_get_at_flat_occupied_miss state direct live logical hreach sigma key address
    haddress word hword index flat hflat hslot hlive hmismatch shardAt hshard region hregion
    ⟨hread, hspillExcl⟩ hroute

/-- The execution route of a generated `table.Table.get_at` call partitioned over the control flow.

The pre-image had eleven arms: three `*SummaryMiss`, three `*SweepNone`, three `*SweepHit`, plus
direct hit and flat vacant. The three `*SummaryMiss` arms are gone with `table.Region.summary_maybe`;
their fields are the `*SweepNone` fields, so each pair is one arm here. `directHit` no longer carries
`hmatch`: `word_stored_table_get_at_direct_hit` derives it from `hsigmaEq` and `hraw`, because the
generated direct path never builds a `table.SpillSlot`. -/
inductive StoredReadRoute (state : WordTable StoredI64) (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress) : Prop where
  | directHit (word : U32) (index : Usize) (flat : table.FlatSlot StoredI64 I64)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok (some index))
      (hslot : state.resident.slots.val[index.val]? = some flat)
      (hlive : flat.live = true)
      (hsigmaEq : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true)
      (hraw : Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok true)
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
  | directNoneSpillNone (shardAt : Usize) (region : table.Region StoredI64 I64)
      (hword : address.direct_word = none)
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hread : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
        key 16#usize = ok none)
      (hspillExcl : ∀ idx value,
        Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
          (.spill shardAt idx) ≠ some ((sigma, key), value))
  | directNoneSpillHit (shardAt : Usize) (region : table.Region StoredI64 I64)
      (home : Usize) (wanted found : U64) (slot : Usize)
      (hword : address.direct_word = none)
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hoccupied : SpillRegionOccupied region)
      (adm : SpillReadHitAdmission region sigma address key 16#usize home wanted found slot)
      (hread : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
        key 16#usize = ok (some slot))
  | flatNoneSpillNone (word : U32) (shardAt : Usize) (region : table.Region StoredI64 I64)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok none)
      (hlane : state.resident.lanes.val ≤ word.val / 256 ∨
        ∀ index value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
          (.flat index) ≠ some ((sigma, key), value))
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hread : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
        key 16#usize = ok none)
      (hspillExcl : ∀ idx value,
        Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
          (.spill shardAt idx) ≠ some ((sigma, key), value))
  | flatNoneSpillHit (word : U32) (shardAt : Usize) (region : table.Region StoredI64 I64)
      (home : Usize) (wanted found : U64) (slot : Usize)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok none)
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hoccupied : SpillRegionOccupied region)
      (adm : SpillReadHitAdmission region sigma address key 16#usize home wanted found slot)
      (hread : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
        key 16#usize = ok (some slot))
  | flatOccupiedMissSpillNone (word : U32) (index : Usize) (flat : table.FlatSlot StoredI64 I64)
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
      (hread : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
        key 16#usize = ok none)
      (hspillExcl : ∀ idx value,
        Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
          (.spill shardAt idx) ≠ some ((sigma, key), value))
      (hroute : Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = some index.val ∨
        ∀ idx value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
          (.flat idx) ≠ some ((sigma, key), value))
  | flatOccupiedMissSpillHit (word : U32) (index : Usize) (flat : table.FlatSlot StoredI64 I64)
      (shardAt : Usize) (region : table.Region StoredI64 I64)
      (home : Usize) (wanted found : U64) (slot : Usize)
      (hword : address.direct_word = some word)
      (hflat : table.Flat.index state.resident word = ok (some index))
      (hslot : state.resident.slots.val[index.val]? = some flat)
      (hlive : flat.live = true)
      (hmismatch : (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok false) ∨
        (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true ∧
         Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok false))
      (hshard : table.shard address = ok shardAt)
      (hregion : state.shards.val[shardAt.val]? = some region)
      (hoccupied : SpillRegionOccupied region)
      (adm : SpillReadHitAdmission region sigma address key 16#usize home wanted found slot)
      (hread : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
        key 16#usize = ok (some slot))

/-- Deliverable 2: Root composition of the public read function `word_stored_table_get_at`
under constructor-reachable `RepR`.  Assembles all landed partitions (direct hit, vacant direct
record, occupied non-match, routed spill `none`, and routed spill hit).

The pre-image carried `Apeiron.GrowableReadLoop.StoredLiveTagAddress state` here; it is **removed**.
The tag law existed to read the occupancy marker back out of the parallel `region.tags` array, and
the marker is now the slot record's own `tag`, supplied by `SpillRegionOccupied`, with the
routed-spill exclusion as a premise.  This theorem therefore has one premise fewer than the
pre-image — a strengthening. -/
theorem word_stored_table_get_at_root
    (state : WordTable StoredI64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (hreach : ConstructorReachableRepR state Apeiron.GrowableSearch.decodeStoredOption
      direct live logical)
    (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (hroute : StoredReadRoute state sigma key address) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  cases hroute with
  | directHit word index flat hword hflat hslot hlive hsigmaEq hraw =>
      exact word_stored_table_get_at_direct_hit state direct live logical hreach sigma key address
        haddress word hword index flat hflat hslot hlive hsigmaEq hraw
  | flatVacant word index flat shardAt hword hflat hslot hlive hshard hflatExcl hspillExcl =>
      exact word_stored_table_get_at_flat_vacant state direct live logical hreach sigma key address
        haddress word hword index flat hflat hslot hlive shardAt hshard hflatExcl hspillExcl
  | directNoneSpillNone shardAt region hword hshard hregion hread hspillExcl =>
      exact word_stored_table_get_at_direct_none_sweep_none state direct live logical hreach sigma
        key address haddress hword shardAt hshard region hregion hread hspillExcl
  | @directNoneSpillHit shardAt region home wanted found slot hword hshard hregion hoccupied adm hread =>
      exact word_stored_table_get_at_direct_none_hit state direct live logical hreach sigma key
        address haddress hword shardAt hshard region hregion hoccupied home wanted found slot adm hread
  | flatNoneSpillNone word shardAt region hword hflat hlane hshard hregion hread hspillExcl =>
      exact word_stored_table_get_at_flat_none_sweep_none state direct live logical hreach sigma key
        address haddress word hword hflat hlane shardAt hshard region hregion hread hspillExcl
  | @flatNoneSpillHit word shardAt region home wanted found slot hword hflat hshard hregion hoccupied adm hread =>
      exact word_stored_table_get_at_flat_none_hit state direct live logical hreach sigma key address
        haddress word hword hflat shardAt hshard region hregion hoccupied home wanted found slot adm hread
  | flatOccupiedMissSpillNone word index flat shardAt region hword hflat hslot hlive hmismatch hshard
      hregion hread hspillExcl hroute =>
      exact word_stored_table_get_at_flat_occupied_miss_sweep_none state direct live logical hreach
        sigma key address haddress word hword index flat hflat hslot hlive hmismatch shardAt hshard
        region hregion hread hspillExcl hroute
  | @flatOccupiedMissSpillHit word index flat shardAt region home wanted found slot hword hflat hslot
      hlive hmismatch hshard hregion hoccupied adm hread =>
      exact word_stored_table_get_at_flat_occupied_miss_hit state direct live logical hreach sigma key
        address haddress word hword index flat hflat hslot hlive hmismatch shardAt hshard region
        hregion hoccupied home wanted found slot adm hread

/-!
## Concrete composition fixtures and paired read controls

`word_stored_table_get_at_root` and its six route theorems all carry `ConstructorReachableRepR`
at the query state. No concrete value of that premise is available in this environment: the only
landed producer, `Growable.GrowableRepresentation.word_stored_table_new_repR`, is a WP triple over
the generated `table.Table.new` loop whose state and `shards` list are exposed only through the
private `Growable.GrowableRepresentation.table_new_empty_repR`, and that empty state carries
`direct = fun _ => none`, `live = fun _ => False`; non-empty reachability is row 4/8 of
the read-proof note, open. The route and root theorems therefore cannot be
instantiated at a concrete state here.

The fixtures below exercise the generated operations the route theorems partition. They are
rebuilt on the merged `{slots, coords, bits, live}` region schema. The routed fixture is keyed at
`place := 0#u64`, whose `table.Region.coord_of` is `0` and whose `table.Region.home` is lane `0`,
so the home probe is the first coordinate read — the `Growable.ReadLoop` fixture idiom
(`ReadLoop.lean:975-1029`). `Result I64` has no `BEq` in this environment, so each check matches on
`Result.match` (`MatchResult`).

Fixture dispositions against the pre-image:

* `compositionControlZeroSummaryRegion` — **retired, not restated.** It existed to make
  `table.Region.summary_maybe` answer `ok false`; that word has no denotation. Its two `#guard`s
  are replaced by `compositionControlEmptyRegionAnswersNone` (the current first admission gate, a
  kernel-checked theorem) and by the coordinate-branch guards on `compositionControlSpillRegion`.
* `compositionControlStoredTag` — **retired, not restated.** It filled the parallel
  `region.tags` array. The occupancy marker is now the slot record's own `tag`; the fixture's
  nonzero tags are literals, and the *identity* coordinate word the generated probe compares
  against is `region.coords`, which the fixture supplies directly.
* the three `word_stored_table_get_at compositionControlSpillTable` guards — **not restated**,
  reason `OPS-7` in the module header. The spill fixture's positive check is at
  `table.probe.spill_read` and at `Apeiron.GrowableSearch.entryAt` instead, both at
  `place := 0#u64`; `entryAt` reads no address, so its guards are exact.
-/

private theorem compositionControlNatBound (n : Nat) (h : n ≤ 1000) : n ≤ Usize.max := by
  rcases Usize.bounds_eq with hb | hb
  · rw [hb]; exact le_trans h (by norm_num [U32.max, U32.numBits])
  · rw [hb]; exact le_trans h (by norm_num [U64.max, U64.numBits])

private def compositionControlRepBound {α : Type} (n : Nat) (x : α) (h : n ≤ 1000) :
    (List.replicate n x).length ≤ Usize.max := by
  rw [List.length_replicate]
  exact compositionControlNatBound n h

private def compositionControlSlotBound {α : Type} (n : Nat) (x : α) (h : n ≤ 1000) :
    (List.replicate n x).length ≤ Usize.max := by
  rw [List.length_replicate]
  exact compositionControlNatBound n h

private def compositionControlHeadBound {α : Type} (x y : α) (n : Nat) (h : n + 1 ≤ 1000) :
    (x :: List.replicate n y).length ≤ Usize.max := by
  rw [List.length_cons, List.length_replicate]
  exact compositionControlNatBound (n + 1) h

/-- The generated locator of the concrete fixtures: no profile, no directory, no staged path, and
no admitted keys.  `table.TableLocator` gained its `admitted : Std.Usize` field
(`proof/ApeironWord.lean:17169-17174`), which the pre-image's fixtures did not carry. -/
def compositionControlLocator : table.TableLocator :=
  { profile := none, directory := none, links := alloc.vec.Vec.new _,
    admitted := 0#usize, staged := none }

/-- An empty spill region: no payload slots and no coordinate words. Its generated
`alloc.vec.Vec.len region.slots` is `0#usize`, so the first admission gate of
`table.probe.spill_read` refuses it. -/
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
    resident := { lanes := 1#usize, slots := alloc.vec.Vec.from (List.replicate 256 compositionControlSlot) (compositionControlSlotBound 256 compositionControlSlot (by norm_num)) },
    shards := alloc.vec.Vec.from (List.replicate 256 compositionControlRegion)
      (compositionControlRepBound 256 compositionControlRegion (by norm_num)),
    len := 0#usize, key := (),
    locator := compositionControlLocator }

/-- The same table with every direct slot vacant. -/
def compositionControlVacantTable : WordTable StoredI64 :=
  { default := 7#i64, home := (),
    resident := { lanes := 1#usize, slots := alloc.vec.Vec.from (List.replicate 256 compositionControlVacantSlot) (compositionControlSlotBound 256 compositionControlVacantSlot (by norm_num)) },
    shards := alloc.vec.Vec.from (List.replicate 256 compositionControlRegion)
      (compositionControlRepBound 256 compositionControlRegion (by norm_num)),
    len := 0#usize, key := (),
    locator := compositionControlLocator }

/-- The generated address of the fixture query `(Parabolic, 5)`. -/
def compositionControlStoredAddress : table.TableAddress :=
  match (Apeiron.GrowableIdentity.wordHomeInst.address () cell.Sigma.Parabolic 5#i64).match with
  | .ok address => address
  | _ => { place := 0#u64, direct_word := none }

/-- The occupied spill slot of the routed fixture: the query's full key `(Parabolic, 5)` with
payload `42`, carrying the nonzero occupancy marker `128#u8`. -/
def compositionControlSpillSlot : StoredSpillSlot :=
  { tag := 128#u8, sigma := cell.Sigma.Parabolic, held := { key := 5#i64 }, value := 42#i64 }

/-- A vacant spill slot: the same attempted payload with the zero occupancy marker. -/
def compositionControlVacantSpillSlot : StoredSpillSlot :=
  { tag := 0#u8, sigma := cell.Sigma.Parabolic, held := { key := 0#i64 }, value := 0#i64 }

/-- Thirty-two capacity, one live slot at the home lane.  `table.Region.home` of `place := 0#u64` is
lane `0`: `alloc.vec.Vec.len self.coords` is `32`, `table.Region.coord_of … 0#u64 = ok 0#u64`
because `0#u64 >>> table.SHARD_BITS = 0`, and `0 < 32` selects the direct branch
`i1 &&& mask` with `mask = 32 - 1`.  Lane `0` of `coords` is that wanted coordinate word; every
later lane is the vacancy terminator `table.COORD_EMPTY`. -/
def compositionControlSpillRegion : table.Region StoredI64 I64 :=
  { slots := alloc.vec.Vec.from
      (compositionControlSpillSlot :: List.replicate 31 compositionControlVacantSpillSlot)
      (compositionControlHeadBound _ _ 31 (by norm_num))
    coords := alloc.vec.Vec.from
      (0#u64 :: List.replicate 31 table.COORD_EMPTY)
      (compositionControlHeadBound _ _ 31 (by norm_num))
    bits := 0#u32
    live := 1#usize }

/-- A `lanes = 0` stored table whose 256 routed shards all carry the spill fixture region, so the
public read reaches the routed probe on whichever shard the generated address selects. -/
def compositionControlSpillTable : WordTable StoredI64 :=
  { default := 7#i64, home := (),
    resident := { lanes := 0#usize, slots := alloc.vec.Vec.new _ },
    shards := alloc.vec.Vec.from (List.replicate 256 compositionControlSpillRegion)
      (compositionControlRepBound 256 compositionControlSpillRegion (by norm_num)),
    len := 0#usize, key := (),
    locator := compositionControlLocator }

/-- **The first admission gate of the current `table.probe.spill_read` body is live.**  The
generated body tests `alloc.vec.Vec.len region.slots != 0#usize` before it computes a home lane
(`proof/ApeironWord.lean:17490-17492`); the empty fixture region has that length `0#usize`, so the
routed read answers `ok none` without probing.  This is a kernel check of the extracted body at a
reachable state, replacing the pre-image's two `table.Region.summary_maybe` guards. -/
theorem compositionControlEmptyRegionAnswersNone :
    table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep compositionControlRegion
      cell.Sigma.Parabolic { place := 0#u64, direct_word := none } 0#i64 16#usize = ok none :=
  spill_read_none_of_empty_region _ _ _ _ _ _ rfl

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

-- `entryAt` at the routed route: the slot record's own occupancy marker is the authority. The
-- occupied slot reads the query's payload back; a zero-tag slot is vacant; a shard past the
-- fixture's 256 regions is absent.
#guard match (Apeiron.GrowableSearch.entryAt compositionControlSpillTable
    Apeiron.GrowableSearch.decodeStoredOption (.spill 0#usize 0#usize)) with
  | some (_, value) => value == 42#i64
  | none => false

#guard match (Apeiron.GrowableSearch.entryAt compositionControlSpillTable
    Apeiron.GrowableSearch.decodeStoredOption (.spill 0#usize 5#usize)) with
  | some _ => false
  | none => true

#guard match (Apeiron.GrowableSearch.entryAt compositionControlSpillTable
    Apeiron.GrowableSearch.decodeStoredOption (.spill 256#usize 0#usize)) with
  | some _ => false
  | none => true

-- `Flat.index` in-range positive and out-of-range (lane `1` of `lanes = 1`) negative.
#guard match (table.Flat.index compositionControlLiveTable.resident (0#u32)).match with
  | .ok o => o.isSome
  | _ => false

#guard match (table.Flat.index compositionControlLiveTable.resident (256#u32)).match with
  | .ok o => o.isNone
  | _ => false

-- Routed spill read at `place := 0#u64`: positive at the home lane `0`; wrong key, wrong `Sigma`
-- and a different place all answer `ok none` — the first through the generated matcher returning
-- `cont` and the vacancy terminator at lane `1`, the third through the home lane's own coordinate
-- word.
#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep compositionControlSpillRegion
    cell.Sigma.Parabolic
    { place := 0#u64, direct_word := none } 5#i64 16#usize).match with
  | .ok (some slot) => slot == 0#usize
  | _ => false

#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep compositionControlSpillRegion
    cell.Sigma.Parabolic
    { place := 0#u64, direct_word := none } 6#i64 16#usize).match with
  | .ok none => true
  | _ => false

#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep compositionControlSpillRegion
    cell.Sigma.Hyperbolic
    { place := 0#u64, direct_word := none } 5#i64 16#usize).match with
  | .ok none => true
  | _ => false

#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep compositionControlSpillRegion
    cell.Sigma.Parabolic { place := 256#u64, direct_word := none } 5#i64 16#usize).match with
  | .ok none => true
  | _ => false

-- Local fail-closed cone check
open Lean Elab Command in
run_cmd do
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  let localNames : Array Lean.Name := #[
    `valid_n_16,
    `SpillReadHitAdmission,
    `word_stored_table_get_at_direct_none_hit,
    `word_stored_table_get_at_flat_none_hit,
    `word_stored_table_get_at_flat_occupied_miss_hit,
    `word_stored_table_get_at_direct_none_sweep_none,
    `word_stored_table_get_at_flat_none_sweep_none,
    `word_stored_table_get_at_flat_occupied_miss_sweep_none,
    `StoredReadRoute,
    `word_stored_table_get_at_root,
    `compositionControlRegion,
    `compositionControlSlot,
    `compositionControlVacantSlot,
    `compositionControlLiveTable,
    `compositionControlVacantTable,
    `compositionControlStoredAddress,
    `compositionControlLocator,
    `compositionControlSpillSlot,
    `compositionControlVacantSpillSlot,
    `compositionControlSpillRegion,
    `compositionControlSpillTable,
    `compositionControlEmptyRegionAnswersNone
  ]
  for decl in localNames.map (Lean.Name.append `Apeiron.GrowableRead) do
    let axioms ← liftTermElabM <| Lean.collectAxioms decl
    let bad := axioms.filter (fun ax => !allowed.contains ax)
    unless bad.isEmpty do
      throwError m!"'{decl}' depends on axioms outside the closed cone: {bad.toList}"
    logInfo m!"'{decl}' depends on axioms: {axioms.toList}"

end Apeiron.GrowableRead