import Growable.ReadLoop
import Lean

/-!
# Public read refinement: `word_{stored,regen}_table_get_at` / `Table.get_at`

Row 6 of the read map (`docs/epics/read-proof.md`). The generated public read
roots are refined to `ok (Apeiron.Table.get logical (sigma, key))` under explicit premises,
partitioned over the generated `table.Table.get_at` case tree:

1. **direct hit** — flat record live, `Sigma` equal, generated matcher `ok true`;
2. **direct empty** — no usable direct record (direct word absent, record out of extent, or
   vacant record), answered from the routed spill search;
3. **occupied non-match** — flat record live but `Sigma`/matcher mismatch, spill fallback;
4. **routed spill `none`** — the generated `table.probe.spill_read` answered `ok none` and no
   routed spill entry represents the query;
5. **routed spill hit** — `table.probe.spill_read` answered `ok (some slot)`, which denotes a
   live entry reading back through `entryAt` (`spill_read_some_entryAt`).

No caller-supplied found/absent result, desired location or public-read equality is a premise:
every premise is a state-side certificate (`RepR`, the live tag/address invariant
`StoredLiveTagAddress`, the live coordinate invariant `StoredLiveCoordAddress`), an equation of a
generated intermediate computation (the generated branch tests `Flat.index`, live bit, `Sigma`
comparison, carrier matcher call, `spill_read` run), or a bound on that run.
Full `RepR` preservation across writes and rebuild is row 4's open part of the read map, so the
reachability of the represented state is carried as the explicit named premise
`ConstructorReachableRepR`; it is never assumed. The rebuild traversal carries its own
certificate (`word_stored_relocate_all_keeps_reserved_tags_from_prepared` under its exact
`hstep`).

## What replaced what

The flat spill port merged `Region.tags` and the occupancy summary into the single
`Region.slots` array of `table.SpillSlot` records, so the read path is now one bounded linear
probe cycle over the capacity (`proof/ApeironWord.lean:17420-17527`). Six names this file
imported have no denotation in the current extraction. Each is **restated over the current body**
or **retired with its replacement named**; none is silently dropped, and none is kept under a name
the body cannot satisfy.

| retired name | disposition | replacement |
|---|---|---|
| `GroupedReadAdmission` | **obsolete, not restated** | its five conjuncts were the grouped sizing `region.slots.length = groups.val * n.val`, `2 ≤ groups.val`, a power-of-two `groups`, a `groups - 1` mask and the positive summary word. The current body admits on `alloc.vec.Vec.len region.slots ≠ 0#usize` and `table.valid_n n` and takes its bound from `alloc.vec.Vec.len region.slots` itself. The admission chain is the premise list of `Apeiron.GrowableReadLoop.generated_spill_read_branch`; the two `ok none` gates it now exposes are `spill_read_none_of_empty_region` and `spill_read_none_of_invalid_n` below |
| `OuterStepReady` | **obsolete, not restated** | it stated the grouped sweep's per-step read readiness over `region.tags` and `region.slots` at a group-relative lane. The current body reads `region.coords[(home + step) &&& mask]` and, on a coordinate hit, `region.slots[(home + step) &&& mask]`; neither read is total. The current form is `Apeiron.GrowableReadLoop.TailReadSuccess` (the two reads) with `Apeiron.GrowableReadLoop.TailLaneObserved` (what a lane recorded) |
| `repR_summary_read_branch` | **obsolete, not restated** | parameterized by `table.Region.summary_maybe region address.place`, a constant absent from the extraction. No caller in this file |
| `repR_live_spill_summary_positive` | **obsolete, not restated** | parameterized by `table.Region.summary_maybe region address.place = ok true`. No caller in this file |
| `spill_read_none_implies_no_full_match` | **restated at the granularity the current body supports, and recorded as open at the rest** | per-lane: `home_vacancy_excludes_live` (the home lane) and `observed_lane_excludes_live_full_match` (every lane a run observed); composed over a `none` run by `spill_read_none_excludes_read_lanes`. Whole-region absence is **not** delivered and is recorded as residual `OPS-1`/`OPS-3` below |
| `spill_read_none_implies_no_shard_full_match` | **obsolete, not restated** | the whole-shard form of the same obligation, blocked by the same two open items. No caller in this file |

### Open items this file does not close

* **`OPS-1` (inherited, unclosed).** `table.Region.home` can return an index at or above
  `alloc.vec.Vec.len region.slots`, and nothing in the corpus rules it out; there is no
  `region.coords.val.length = region.slots.val.length` fact and no proof that
  `table.Region.home region place < table.Region.bound region`. Consequently the lane
  `home &&& mask` is covered by neither the home probe nor the tail cycle on such a region, and
  **no theorem here concludes whole-region or whole-shard absence**. The per-lane theorems are
  stated at the lanes a run actually touched, which is all the current body supports.
* **`OPS-2` (inherited, unclosed).** `Apeiron.GrowableReadLoop.TailReadSuccess` — the two
  coordinate/slot reads succeeded at every executed lane — is carried as an explicit premise of the
  read-path theorems, not derived from a run's own equation.
* **`OPS-3` (opened here).** `Apeiron.GrowableRepresentation.repR_spillSummaryMissSound` is gone
  from `Representation.lean`, so the three "public read answers the default" partitions below take
  the routed-spill **exclusion** as an explicit premise where they previously derived it from a
  summary word. `Apeiron.GrowableScan.spill_probe_bound_absent_sound` and
  `Apeiron.GrowableSearch.AbsenceCertificate.coverage` are the named replacement route; its
  producer needs `OPS-1`, so it is **not** written here. Stated as a hypothesis change, not hidden.

Full `RepR` preservation across writes and rebuild remains row 4's open part of the read map.
-/

namespace Apeiron.GrowableRead

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

abbrev LogicalKey := Apeiron.GrowableRepresentation.LogicalKey
abbrev Location := Apeiron.GrowableRepresentation.Location
abbrev StoredI64 := Apeiron.GrowableIdentity.StoredI64
abbrev RegenI64 := Apeiron.GrowableIdentity.RegenI64
abbrev WordTable := Apeiron.GrowableRepresentation.WordTable

abbrev StoredSpillSlot := table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64

/-- The named reachability premise of the public read refinement: `RepR` at a state reachable from
the generated constructor `table.Table.new` through the landed write/rebuild certificate chain
(`word_stored_table_new_repR`, `word_regen_table_new_repR`, the `*keeps_tags` publication
facts, and the rebuild traversal certificate). Full `RepR` preservation across writes and
rebuild is row 4's open part of the read map, so this premise is carried explicitly by every
read theorem and is not assumed derivable here. -/
def ConstructorReachableRepR {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) (direct : LogicalKey → Option Location)
    (live : LogicalKey → Prop) (logical : Apeiron.Table.HMap LogicalKey I64) : Prop :=
  Apeiron.GrowableRepresentation.RepR state decode direct live logical

/-- **The generated flat-path matcher equation.**  `table.Table.resident_exact` does not build a
`table.SpillSlot` record: on a `Sigma` equality it calls the carrier matcher on the flat slot's own
`held` and compares `flat.sigma` (`proof/ApeironWord.lean:17295-17326`).  This states that equation
for the record-shaped `table.matches`, which is how the identity facts are phrased.  It is what
makes the retired `hmatch` premise of the two direct-hit partitions derivable rather than assumed. -/
private theorem stored_flat_match_unfold (slotSigma querySigma : cell.Sigma)
    (held : Apeiron.GrowableIdentity.StoredI64) (value : I64) (query : I64)
    (address : table.TableAddress)
    (hsigmaEq : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slotSigma querySigma = ok true) :
    table.matches Apeiron.GrowableIdentity.storedKeyRep
      { tag := 0#u8, sigma := slotSigma, held := held, value := value } querySigma query address =
      Apeiron.GrowableIdentity.storedKeyRep.«matches» querySigma held query address := by
  unfold table.matches
  simp [hsigmaEq]

/-- The regen carrier's analogue of `stored_flat_match_unfold`. -/
private theorem regen_flat_match_unfold (slotSigma querySigma : cell.Sigma)
    (held : Apeiron.GrowableIdentity.RegenI64) (value : I64) (query : I64)
    (address : table.TableAddress)
    (hsigmaEq : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq slotSigma querySigma = ok true) :
    table.matches Apeiron.GrowableIdentity.regenKeyRep
      { tag := 0#u8, sigma := slotSigma, held := held, value := value } querySigma query address =
      Apeiron.GrowableIdentity.regenKeyRep.«matches» querySigma held query address := by
  unfold table.matches
  simp [hsigmaEq]

/-- The stored carrier matcher at the instance terms the generated direct route calls, so a
premise stated with the `Apeiron.GrowableIdentity` abbreviation can be used at the generated body. -/
private theorem stored_keyrep_matches_expanded (sigma : cell.Sigma) (held : StoredI64)
    (key : I64) (address : table.TableAddress) :
    Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma held key address =
      table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.matches
        I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma sigma held key address := rfl

/-- Partition 1 (direct hit), Stored carrier: the generated public read returns the flat slot
value at a full generated matcher hit, and that value is the canonical `Apeiron.Table.get`.
Premises are the generated direct-route branch tests (`hword`, `hflat`, `hslot`, `hlive`,
`hsigmaEq`, `hraw`), which are exactly the tests `table.Table.resident_exact` performs: the
`hmatch` premise of the pre-image is **derived** from them by `stored_flat_match_unfold`, because
the generated direct path never builds a `table.SpillSlot`. -/
theorem word_stored_table_get_at_direct_hit
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
    (hsigmaEq : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true)
    (hraw : Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok true) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  have hmatch : table.matches Apeiron.GrowableIdentity.storedKeyRep
      { tag := 0#u8, sigma := flat.sigma, held := flat.held, value := flat.value } sigma key address =
      ok true :=
    (stored_flat_match_unfold flat.sigma sigma flat.held flat.value key address hsigmaEq).trans hraw
  have hfull : Apeiron.GrowableIdentity.logicalStored flat.sigma flat.held = (sigma, key) :=
    (Apeiron.GrowableIdentity.stored_matches_iff flat.sigma sigma flat.held flat.value key
      address).mp hmatch
  have hsigma : flat.sigma = sigma := by
    have hp : (flat.sigma, Apeiron.GrowableIdentity.decodeStored flat.sigma flat.held) =
        (sigma, key) := hfull
    exact (Prod.mk.inj hp).1
  have hkey : flat.held.key = key := by
    have hp : (flat.sigma, Apeiron.GrowableIdentity.decodeStored flat.sigma flat.held) =
        (sigma, key) := hfull
    exact (Prod.mk.inj hp).2
  have hentry : Apeiron.GrowableSearch.entryAt state
      Apeiron.GrowableSearch.decodeStoredOption (.flat index) = some ((sigma, key), flat.value) := by
    simp [Apeiron.GrowableSearch.entryAt, hslot, hlive,
      Apeiron.GrowableSearch.decodeStoredOption, hsigma, hkey]
  have hget : Apeiron.Table.get logical (sigma, key) = flat.value :=
    hreach.observations.1 (.flat index) (sigma, key) flat.value hentry
  rw [hget]
  have hindex : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot StoredI64 I64))
      state.resident.slots index = ok flat := by
    simpa [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hslot] using
      (rfl : (ok flat : Result (table.FlatSlot StoredI64 I64)) = ok flat)
  have haddr : table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address state.home sigma key = ok address := haddress
  have hraw' : table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.matches
      I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma sigma flat.held key address = ok true :=
    (stored_keyrep_matches_expanded sigma flat.held key address).symm.trans hraw
  simp only [word_stored_table_get_at]
  unfold table.Table.get_at
  simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddr,
    table.Table.resident_exact, bind_tc_ok, hword, hflat, hindex, hlive, hsigmaEq,
    hraw', bind_tc_ok, ↓reduceIte]

/-- The regen carrier matcher at the instance terms the generated direct route calls. -/
private theorem regen_keyrep_matches_expanded (sigma : cell.Sigma) (held : RegenI64)
    (key : I64) (address : table.TableAddress) :
    Apeiron.GrowableIdentity.regenKeyRep.«matches» sigma held key address =
      table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.matches
        Apeiron.GrowableIdentity.wordExactIdentityInst Apeiron.GrowableIdentity.wordHomeInst
        Apeiron.GrowableIdentity.wordCodecInst sigma held key address := rfl

/-- Partition 1 (direct hit), Regen carrier: the generated public read returns the flat slot
value at a full generated matcher hit on a reachable held key, and that value is the canonical
`Apeiron.Table.get`. The held-key reachability premise (`word_regen_hold`) is carried
explicitly and named (`hheld`); the pre-image's `hmatch` premise is derived from `hsigmaEq` and
`hraw`, for the same reason as in the Stored carrier. -/
theorem word_regen_table_get_at_direct_hit
    (state : WordTable RegenI64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (hreach : ConstructorReachableRepR state Apeiron.GrowableSearch.decodeRegenOption
      direct live logical)
    (sigma : cell.Sigma) (key original : I64)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (word : U32) (hword : address.direct_word = some word)
    (index : Usize) (flat : table.FlatSlot RegenI64 I64)
    (hflat : table.Flat.index state.resident word = ok (some index))
    (hslot : state.resident.slots.val[index.val]? = some flat)
    (hlive : flat.live = true)
    (hheld : word_regen_hold flat.sigma original = ok flat.held)
    (hsigmaEq : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true)
    (hraw : Apeiron.GrowableIdentity.regenKeyRep.«matches» sigma flat.held key address = ok true) :
    word_regen_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  have hmatch : table.matches Apeiron.GrowableIdentity.regenKeyRep
      { tag := 0#u8, sigma := flat.sigma, held := flat.held, value := flat.value } sigma key address =
      ok true :=
    (regen_flat_match_unfold flat.sigma sigma flat.held flat.value key address hsigmaEq).trans hraw
  have hfull : (flat.sigma, original) = (sigma, key) :=
    (Apeiron.GrowableIdentity.regen_slot_matches_iff_of_reachable flat.sigma sigma flat.held
      original key flat.value address hheld).mp hmatch
  have hsigma : flat.sigma = sigma := (Prod.mk.inj hfull).1
  have hkey : original = key := (Prod.mk.inj hfull).2
  have hdecodeOriginal := Apeiron.GrowableIdentity.regen_hold_decodes flat.sigma original
  rw [hheld] at hdecodeOriginal
  simp only [WP.spec_ok] at hdecodeOriginal
  have hdecode : Apeiron.GrowableSearch.decodeRegenOption flat.sigma flat.held = some key := by
    simp [Apeiron.GrowableSearch.decodeRegenOption, hdecodeOriginal, hkey]
  have hentry : Apeiron.GrowableSearch.entryAt state
      Apeiron.GrowableSearch.decodeRegenOption (.flat index) = some ((sigma, key), flat.value) := by
    simp only [Apeiron.GrowableSearch.entryAt, hslot, hlive]
    rw [hdecode]
    simp [hsigma]
  have hget : Apeiron.Table.get logical (sigma, key) = flat.value :=
    hreach.observations.1 (.flat index) (sigma, key) flat.value hentry
  rw [hget]
  have hindex : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot RegenI64 I64))
      state.resident.slots index = ok flat := by
    simpa [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hslot] using
      (rfl : (ok flat : Result (table.FlatSlot RegenI64 I64)) = ok flat)
  have haddr : table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address state.home sigma key = ok address := haddress
  have hraw' : table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.matches
      Apeiron.GrowableIdentity.wordExactIdentityInst Apeiron.GrowableIdentity.wordHomeInst
      Apeiron.GrowableIdentity.wordCodecInst sigma flat.held key address = ok true :=
    (regen_keyrep_matches_expanded sigma flat.held key address).symm.trans hraw
  simp only [word_regen_table_get_at]
  unfold table.Table.get_at
  simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddr,
    table.Table.resident_exact, bind_tc_ok, hword, hflat, hindex, hlive, hsigmaEq,
    hraw', bind_tc_ok, ↓reduceIte]

/-!
## Localization and absence helpers for the read partitions

Every physical representative of a queried full key is either at the query's direct flat
index or in its routed shard. Together with per-route flat exclusion and routed spill
exclusion this yields `Absent`, and `ObservationCorrect.2` then fixes the canonical `get`
at `state.default` ("none represented as the default"). These are the semantic halves of
partitions 2-4.
-/

/-- A physical representative of the queried full key sits at the query's direct flat index
(given by `preparedFlatIndex`) or in the query's routed shard. Follows `RepR.address`. -/
theorem query_rep {R : Type} {state : WordTable R}
    {decode : cell.Sigma → R → Option I64}
    {direct : LogicalKey → Option Location} {live : LogicalKey → Prop}
    {logical : Apeiron.Table.HMap LogicalKey I64}
    (hreach : ConstructorReachableRepR state decode direct live logical)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (location : Location) (value : I64)
    (hentry : Apeiron.GrowableSearch.entryAt state decode location = some ((sigma, key), value)) :
    (∃ index : Usize, location = .flat index ∧
        Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val =
          some index.val) ∨
    (∃ shard index : Usize, location = .spill shard index ∧ table.shard address = ok shard) := by
  have hcorr := hreach.address location (sigma, key) value hentry
  cases location with
  | flat index =>
      rcases hcorr with ⟨address', haddr', hprepared⟩
      have haddr'' : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key =
          ok address' := haddr'
      have heq : address' = address := Result.ok_injective (haddr''.symm.trans haddress)
      subst address'
      exact Or.inl ⟨index, rfl, hprepared⟩
  | spill shard index =>
      rcases hcorr with ⟨address', haddr', hshard'⟩
      have haddr'' : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key =
          ok address' := haddr'
      have heq : address' = address := Result.ok_injective (haddr''.symm.trans haddress)
      subst address'
      exact Or.inr ⟨shard, index, rfl, hshard'⟩

/-- Per-route flat exclusion plus routed-shard spill exclusion gives full logical absence. -/
theorem absent_of_exclusions {R : Type} {state : WordTable R}
    {decode : cell.Sigma → R → Option I64}
    {direct : LogicalKey → Option Location} {live : LogicalKey → Prop}
    {logical : Apeiron.Table.HMap LogicalKey I64}
    (hreach : ConstructorReachableRepR state decode direct live logical)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (shardAt : Usize) (hshard : table.shard address = ok shardAt)
    (hflat : ∀ index value,
      Apeiron.GrowableSearch.entryAt state decode (.flat index) ≠ some ((sigma, key), value))
    (hspill : ∀ index value,
      Apeiron.GrowableSearch.entryAt state decode (.spill shardAt index) ≠ some ((sigma, key), value)) :
    Apeiron.GrowableSearch.Absent state decode (sigma, key) := by
  intro location value hentry
  rcases query_rep hreach sigma key address haddress location value hentry with h | h
  · obtain ⟨index, hlocation, _⟩ := h
    subst location
    exact hflat index value hentry
  · obtain ⟨shard, index, hlocation, hshard'⟩ := h
    subst location
    have hshardEq : shard = shardAt := Result.ok_injective (hshard'.symm.trans hshard)
    subst shard
    exact hspill index value hentry

/-- A logically absent query reads the canonical default: "none represented as the default". -/
theorem default_of_absent {R : Type} {state : WordTable R}
    {decode : cell.Sigma → R → Option I64}
    {direct : LogicalKey → Option Location} {live : LogicalKey → Prop}
    {logical : Apeiron.Table.HMap LogicalKey I64}
    (hreach : ConstructorReachableRepR state decode direct live logical)
    (query : LogicalKey) (hab : Apeiron.GrowableSearch.Absent state decode query) :
    Apeiron.Table.get logical query = state.default := by
  have hnotLive : ¬ live query := by
    intro hl
    obtain ⟨location, value, hentry⟩ := (hreach.liveSet query).mp hl
    exact hab location value hentry
  exact hreach.observations.2 query hnotLive

/-! ## The generated `spill_read` admission gates

These replace the retired `spill_read_none_of_summary_miss`, whose premise
`table.Region.summary_maybe region address.place = ok false` was a constant of the removed summary
word. The current body has **two** `ok none` gates before it touches the region, and both are
restated: an empty region (`alloc.vec.Vec.len region.slots = 0#usize`) and an invalid divisor
(`table.valid_n n = ok false`). Neither replaces a conclusion of the retired theorem — the retired
theorem's conclusion was `ok none`, which the current body also produces at a vacancy and from the
bounded cycle — but the retired theorem's *subject* is gone and these are the gates that now decide
the same outcome before the probe starts. -/

/-- **Admission gate 1.**  A region with no slots answers `ok none` without entering the probe: the
generated body tests `alloc.vec.Vec.len region.slots != 0#usize` first
(`proof/ApeironWord.lean:17490-17492`). -/
theorem spill_read_none_of_empty_region {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (region : table.Region R V) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : K) (n : Usize)
    (hempty : alloc.vec.Vec.len region.slots = 0#usize) :
    table.probe.spill_read keyRep region sigma address key n = ok none := by
  have hne : (alloc.vec.Vec.len region.slots != 0#usize) = false := by simpa using hempty
  have hlen : region.slots.val.length = 0 := by
    have := congrArg UScalar.val hempty
    rw [alloc.vec.Vec.len_val] at this
    simpa using this
  unfold table.probe.spill_read
  simp [hne, hlen]

/-- **Admission gate 2.**  A divisor the generated `table.valid_n` refuses answers `ok none` before
the home lane is computed (`proof/ApeironWord.lean:17493-17494`). -/
theorem spill_read_none_of_invalid_n {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (region : table.Region R V) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : K) (n : Usize)
    (hvalid : table.valid_n n = ok false) :
    table.probe.spill_read keyRep region sigma address key n = ok none := by
  by_cases hempty : alloc.vec.Vec.len region.slots = 0#usize
  · exact spill_read_none_of_empty_region keyRep region sigma address key n hempty
  · have hne : (alloc.vec.Vec.len region.slots != 0#usize) = true := by simp [hempty]
    unfold table.probe.spill_read
    simp [hne, hvalid]

/-- `table.valid_n 16#usize = ok true` by definition: the routed divisor the public read uses is
admitted. (Kept here under a distinct name; `Growable.ReadComposition.valid_n_16` states the same
fact in its own namespace scope.) -/
theorem spill_read_valid_n_16 : table.valid_n 16#usize = ok true := by
  unfold table.valid_n table.MAX_GROUP core.num.Usize.is_power_of_two
  rfl

/-- The generated flat index rejects any direct word whose flat lane exceeds resident capacity.
Relates `table.Flat.index self word` to `word.val / 256` and `lanes`. -/
theorem flat_index_none_pin {R V : Type} (self : table.Flat R V) (word : U32)
    (hge : self.lanes.val ≤ word.val / 256) :
    table.Flat.index self word = ok none := by
  have hspec : (word >>> table.SHARD_BITS) ⦃ z => z.val = word.val >>> table.SHARD_BITS.val ∧ z.bv = word.bv >>> table.SHARD_BITS.val ⦄ := by
    apply @UScalar.ShiftRight_spec .U32 .U32 word table.SHARD_BITS
    unfold table.SHARD_BITS table.partition.SHARD_BITS
    decide
  obtain ⟨i, hi, hval, _hbv⟩ := WP.spec_imp_exists hspec
  have hsb : table.SHARD_BITS.val = 8 := by
    unfold table.SHARD_BITS table.partition.SHARD_BITS
    rfl
  have hival : i.val = word.val / 256 := by
    rw [hval, hsb, Nat.shiftRight_eq_div_pow]
  unfold table.Flat.index
  simp only [hi, bind_tc_ok, lift]
  have hlanesLt : self.lanes.val < 2 ^ 64 := by
    cases System.Platform.numBits_eq <;> scalar_tac
  have hcast : (UScalar.cast .U64 self.lanes).val = self.lanes.val := by
    rw [UScalar.cast_val_eq]
    exact Nat.mod_eq_of_lt hlanesLt
  have hle : (UScalar.cast .U64 self.lanes : U64).val ≤ (core.convert.num.FromU64U32.from i).val := by
    rw [core.convert.num.FromU64U32.from_val_eq, hival, hcast]
    exact hge
  have hcond : (UScalar.cast .U64 self.lanes : U64) ≤ core.convert.num.FromU64U32.from i := hle
  simp only [ge_iff_le, hcond, ↓reduceIte]

/-- Every successful generated flat index requires the direct word lane to be within resident lanes. -/
theorem flat_index_some_pin {R V : Type} (self : table.Flat R V) (word : U32)
    (index : Usize) (hidx : table.Flat.index self word = ok (some index)) :
    word.val / 256 < self.lanes.val := by
  by_contra h
  have hge : self.lanes.val ≤ word.val / 256 := by omega
  have hnone := flat_index_none_pin self word hge
  rw [hnone] at hidx
  have heq := Result.ok_injective hidx
  cases heq

/-- The generated flat index computation relates `table.Flat.index self word` to `word.val / 256`
and `lanes`: out-of-range lanes yield `none`, and successful index lookups require in-range lanes. -/
theorem flat_index_compute {R V : Type} (self : table.Flat R V) (word : U32) :
    (self.lanes.val ≤ word.val / 256 → table.Flat.index self word = ok none) ∧
    (∀ index, table.Flat.index self word = ok (some index) → word.val / 256 < self.lanes.val) :=
  ⟨flat_index_none_pin self word, flat_index_some_pin self word⟩

/-- **Routed spill read, negative run.**  The current replacement for the retired
`htail : (region.summary_maybe address.place = ok false) ∨ …` disjunction.  The retired first
disjunct had no counterpart: `table.Region.summary_maybe` is absent from the extraction, so no
current state satisfies it, and nothing is lost by dropping it.  The second disjunct survives
unchanged, with the two admission gates of the section above as new ways to discharge its
`spill_read` equation.  The routed-spill **exclusion** is a premise, not a derived fact: see
`OPS-3` in the module header. -/
structure SpillReadNoneSound (state : WordTable Apeiron.GrowableIdentity.StoredI64)
    (sigma : cell.Sigma)
    (key : I64) (address : table.TableAddress) (shardAt : Usize)
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64) (n : Usize) : Prop where
  /-- The generated routed spill read answered `ok none`. -/
  read : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
    key n = ok none
  /-- No routed spill entry of that shard represents the queried full key. -/
  excludes : ∀ index value,
    Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
      (.spill shardAt index) ≠ some ((sigma, key), value)

/-- The stored `Home`/`KeyRep` instances the generated public read uses. -/
abbrev StoredHomeInst := table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma
abbrev StoredKeyRepInst : table.keyrep.KeyRep StoredI64 I64 table.WordHome :=
  table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep
    I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple core.clone.CloneI64
    table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma

/-- **The generated public read answers the default once the direct route reported a miss.**  The
body is `table.Table.get_at`: the face domain's `Sigma`, the address computation, the direct-route
enum, and — on `ResidentRead.Miss` — the routed spill read through
`table.Table.get_spill`/`table.Table.get_exact_spill`, whose `ok none` is turned into
`self.default` by `core.option.Option.unwrap_or`
(`proof/ApeironWord.lean:21754-21789`, `17483-17527`).  The pre-image re-derived this chain inside
each partition; it is stated once here, and each partition supplies only its own
`resident_exact` equation and its own `spill_read` equation. -/
private theorem get_at_default_of_resident_miss
    (state : WordTable StoredI64) (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (hresident : table.Table.resident_exact StoredHomeInst StoredKeyRepInst state sigma address key
      = ok table.ResidentRead.Miss)
    (shardAt : Usize) (hshard : table.shard address = ok shardAt)
    (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shardAt.val]? = some region)
    (hspillNone : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma
      address key 16#usize = ok none) :
    word_stored_table_get_at state sigma key = ok state.default := by
  have haddr : table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address state.home sigma key = ok address := haddress
  have hshard' : table.shard ⟨address.place, address.direct_word⟩ = ok shardAt := hshard
  have hvec : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (table.Region StoredI64 I64))
      state.shards shardAt = ok region := by
    simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hregion]
  simp only [word_stored_table_get_at]
  unfold table.Table.get_at
  simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddr,
    hresident, bind_tc_ok, ↓reduceIte, table.Table.get_spill, table.Table.get_exact_spill,
    bind_tc_ok, hshard', hvec, bind_tc_ok, Apeiron.GrowableIdentity.storedKeyRep, hspillNone,
    bind_tc_ok, core.option.Option.Insts.CoreOpsTry_traitTry.branch,
    core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual,
    core.option.Option.unwrap_or]

/-- **The same, for the `ResidentRead.Vacant` route** (`proof/ApeironWord.lean:17295-17326`): a
vacant direct record answers `ok self.default` without consulting the shard. -/
private theorem get_at_default_of_resident_vacant
    (state : WordTable StoredI64) (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (hresident : table.Table.resident_exact StoredHomeInst StoredKeyRepInst state sigma address key
      = ok table.ResidentRead.Vacant) :
    word_stored_table_get_at state sigma key = ok state.default := by
  have haddr : table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address state.home sigma key = ok address := haddress
  simp only [word_stored_table_get_at]
  unfold table.Table.get_at
  simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddr,
    hresident, bind_tc_ok, ↓reduceIte]

/-- Partition 2 (direct empty, shape A: direct word absent), Stored carrier: the generated public
read returns `ok state.default` when `address.direct_word = none`, and that value is the canonical
`Apeiron.Table.get`.  The routed spill read is supplied as `SpillReadNoneSound`: a `none` answer
together with the absence of every routed spill entry for the query. -/
theorem word_stored_table_get_at_direct_none
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
    (htail : SpillReadNoneSound state sigma key address shardAt region 16#usize) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  have hflat : ∀ index value,
      Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption (.flat index) ≠
        some ((sigma, key), value) := by
    intro index value hentry
    rcases query_rep hreach sigma key address haddress (.flat index) value hentry with h | h
    · obtain ⟨index', _hloc, hprepared⟩ := h
      simp only [Apeiron.GrowableGeometry.preparedFlatIndex, hword] at hprepared
      cases hprepared
    · obtain ⟨shard, index', hloc, _⟩ := h
      cases hloc
  have hresident : table.Table.resident_exact StoredHomeInst StoredKeyRepInst state sigma address key
      = ok table.ResidentRead.Miss := by
    unfold table.Table.resident_exact
    simp only [bind_tc_ok, hword, bind_tc_ok, ↓reduceIte]
  have habsent := absent_of_exclusions hreach sigma key address haddress shardAt hshard hflat htail.excludes
  have hdefault := default_of_absent hreach (sigma, key) habsent
  rw [get_at_default_of_resident_miss state sigma key address haddress hresident shardAt hshard
    region hregion htail.read, hdefault]

/-- Partition 2 (direct empty, shape B: flat index miss), Stored carrier: the generated public
read returns `ok state.default` when `address.direct_word = some word` and `Flat.index = ok none`,
and that value is the canonical `Apeiron.Table.get`. -/
theorem word_stored_table_get_at_flat_none
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
    (htail : SpillReadNoneSound state sigma key address shardAt region 16#usize) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  have hflatExcl : ∀ index value,
      Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption (.flat index) ≠
        some ((sigma, key), value) := by
    cases hlane with
    | inl hge =>
        intro index value hentry
        rcases query_rep hreach sigma key address haddress (.flat index) value hentry with h | h
        · obtain ⟨index', _hloc, hprepared⟩ := h
          have hprepNone : Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = none := by
            simp only [Apeiron.GrowableGeometry.preparedFlatIndex, hword]
            have hnotLt : ¬ Apeiron.GrowableGeometry.directLane word < state.resident.lanes.val := by
              change ¬ (word.val / 256 < state.resident.lanes.val)
              omega
            simp [hnotLt]
          rw [hprepNone] at hprepared
          cases hprepared
        · obtain ⟨shard, index', hloc, _⟩ := h
          cases hloc
    | inr hexcl => exact hexcl
  have hresident : table.Table.resident_exact StoredHomeInst StoredKeyRepInst state sigma address key
      = ok table.ResidentRead.Miss := by
    unfold table.Table.resident_exact
    simp only [bind_tc_ok, hword, bind_tc_ok, hflat, bind_tc_ok, ↓reduceIte]
  have habsent := absent_of_exclusions hreach sigma key address haddress shardAt hshard hflatExcl htail.excludes
  have hdefault := default_of_absent hreach (sigma, key) habsent
  rw [get_at_default_of_resident_miss state sigma key address haddress hresident shardAt hshard
    region hregion htail.read, hdefault]

/-- A vacant flat slot holds no entry for any full key. -/
theorem flat_vacant_entry_none
    (state : WordTable StoredI64)
    (sigma : cell.Sigma) (key : I64)
    (index : Usize) (flat : table.FlatSlot StoredI64 I64)
    (hslot : state.resident.slots.val[index.val]? = some flat)
    (hlive : flat.live = false)
    (value : I64) :
    Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
      (.flat index) ≠ some ((sigma, key), value) := by
  simp [Apeiron.GrowableSearch.entryAt, hslot, hlive]

/-- A prepared direct route to a vacant slot excludes the query key from all flat positions. -/
theorem flat_vacant_exclusion_of_prepared
    (state : WordTable StoredI64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (hreach : ConstructorReachableRepR state Apeiron.GrowableSearch.decodeStoredOption
      direct live logical)
    (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (index : Usize) (flat : table.FlatSlot StoredI64 I64)
    (hslot : state.resident.slots.val[index.val]? = some flat)
    (hlive : flat.live = false)
    (hroute : Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = some index.val) :
    ∀ idx value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
      (.flat idx) ≠ some ((sigma, key), value) := by
  intro idx value hentry
  rcases query_rep hreach sigma key address haddress (.flat idx) value hentry with h | h
  · obtain ⟨idx', _hloc, hprepared⟩ := h
    have heq : idx'.val = index.val := by
      rw [hroute] at hprepared
      exact Option.some.inj hprepared.symm
    have heqIdx : idx' = index := UScalar.eq_of_val_eq heq
    have hentryIdx' : Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.flat idx') = some ((sigma, key), value) := by
      cases _hloc
      exact hentry
    rw [heqIdx] at hentryIdx'
    simp [Apeiron.GrowableSearch.entryAt, hslot, hlive] at hentryIdx'
  · obtain ⟨shard, idx', hloc, _⟩ := h
    cases hloc

/-- Partition 2 (direct empty, shape C: vacant flat slot), Stored carrier: the generated public
read returns `ok state.default` when the slot is vacant (`flat.live = false`), and that value is
the canonical `Apeiron.Table.get`. -/
theorem word_stored_table_get_at_flat_vacant
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
    (hlive : flat.live = false)
    (shardAt : Usize) (hshard : table.shard address = ok shardAt)
    (hflatExcl : ∀ idx value,
      Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.flat idx) ≠ some ((sigma, key), value))
    (hspillExcl : ∀ idx value,
      Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.spill shardAt idx) ≠ some ((sigma, key), value)) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  have hresident : table.Table.resident_exact StoredHomeInst StoredKeyRepInst state sigma address key
      = ok table.ResidentRead.Vacant := by
    have hidx : alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot StoredI64 I64))
        state.resident.slots index = ok flat := by
      simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hslot]
    unfold table.Table.resident_exact
    simp only [bind_tc_ok, hword, bind_tc_ok, hflat, bind_tc_ok, hidx, bind_tc_ok, hlive,
      bind_tc_ok, ↓reduceIte]
    rfl
  have habsent := absent_of_exclusions hreach sigma key address haddress shardAt hshard hflatExcl hspillExcl
  have hdefault := default_of_absent hreach (sigma, key) habsent
  rw [get_at_default_of_resident_vacant state sigma key address haddress hresident, hdefault]

/-- When a flat slot is occupied (`flat.live = true`) but does not match the query, the slot does
not hold an entry for `(sigma, key)`.

The pre-image carried a third disjunct stating a `table.matches` equation over the flat slot's
triple.  That equation is **derived** from the first two disjuncts by `stored_flat_match_unfold`
(`table.matches` is `if eq slot.sigma sigma then storedKeyRep.matches … else ok false`), and the
generated direct path never builds the record the third disjunct named.  Dropping it therefore
**strengthens** the hypothesis and changes no conclusion. -/
theorem flat_occupied_ne_exclusion
    (state : WordTable StoredI64)
    (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (index : Usize) (flat : table.FlatSlot StoredI64 I64)
    (hslot : state.resident.slots.val[index.val]? = some flat)
    (hlive : flat.live = true)
    (hmismatch : (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok false) ∨
      (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true ∧
       Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok false)) :
    Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
      (.flat index) ≠ some ((sigma, key), flat.value) := by
  intro hentry
  have heq : some ((flat.sigma, flat.held.key), flat.value) = some ((sigma, key), flat.value) := by
    simpa [Apeiron.GrowableSearch.entryAt, hslot, hlive,
      Apeiron.GrowableSearch.decodeStoredOption] using hentry
  have hinj := Option.some.inj heq
  have hsigma : flat.sigma = sigma := (Prod.mk.inj (Prod.mk.inj hinj).1).1
  have hkey : flat.held.key = key := (Prod.mk.inj (Prod.mk.inj hinj).1).2
  rcases hmismatch with hsigmaEq | ⟨hsigmaTrue, hkeyMatch⟩
  · rw [hsigma] at hsigmaEq
    have hsigOk : cell.Sigma.Insts.CoreCmpPartialEqSigma.eq sigma sigma = ok true :=
      (Apeiron.GrowableIdentity.sigma_eq_true_iff sigma sigma).mpr rfl
    rw [hsigOk] at hsigmaEq
    have heqRes := Result.ok_injective hsigmaEq
    cases heqRes
  · have hfull : Apeiron.GrowableIdentity.logicalStored flat.sigma flat.held = (sigma, key) := by
      simp [Apeiron.GrowableIdentity.logicalStored, Apeiron.GrowableIdentity.decodeStored,
        hsigma, hkey]
    have hmatchTrue :=
      (Apeiron.GrowableIdentity.stored_matches_iff flat.sigma sigma flat.held flat.value key address).mpr hfull
    have hmatchUnfold : table.matches Apeiron.GrowableIdentity.storedKeyRep
        { tag := 0#u8, sigma := flat.sigma, held := flat.held, value := flat.value } sigma key address =
        Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address :=
      stored_flat_match_unfold flat.sigma sigma flat.held flat.value key address hsigmaTrue
    rw [hmatchUnfold] at hmatchTrue
    rw [hmatchTrue] at hkeyMatch
    have heqRes := Result.ok_injective hkeyMatch
    cases heqRes

/-- A prepared direct route to an occupied non-matching slot excludes the query key
from all flat positions. -/
theorem flat_occupied_exclusion_of_prepared
    (state : WordTable StoredI64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (hreach : ConstructorReachableRepR state Apeiron.GrowableSearch.decodeStoredOption
      direct live logical)
    (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (index : Usize) (flat : table.FlatSlot StoredI64 I64)
    (hslot : state.resident.slots.val[index.val]? = some flat)
    (hlive : flat.live = true)
    (hmismatch : (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok false) ∨
      (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true ∧
       Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok false))
    (hroute : Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = some index.val) :
    ∀ idx value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
      (.flat idx) ≠ some ((sigma, key), value) := by
  intro idx value hentry
  rcases query_rep hreach sigma key address haddress (.flat idx) value hentry with h | h
  · obtain ⟨idx', _hloc, hprepared⟩ := h
    have heq : idx'.val = index.val := by
      rw [hroute] at hprepared
      exact Option.some.inj hprepared.symm
    have heqIdx : idx' = index := UScalar.eq_of_val_eq heq
    have hentryIdx' : Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.flat idx') = some ((sigma, key), value) := by
      cases _hloc
      exact hentry
    rw [heqIdx] at hentryIdx'
    have heqVal : some ((flat.sigma, flat.held.key), flat.value) = some ((sigma, key), value) := by
      simpa [Apeiron.GrowableSearch.entryAt, hslot, hlive,
        Apeiron.GrowableSearch.decodeStoredOption] using hentryIdx'
    have hinj := Option.some.inj heqVal
    have hval : flat.value = value := (Prod.mk.inj hinj).2
    subst value
    exact flat_occupied_ne_exclusion state sigma key address index flat hslot hlive hmismatch hentryIdx'
  · obtain ⟨shard, idx', hloc, _⟩ := h
    cases hloc

/-- Partition 3 (occupied non-match at the stored root), Stored carrier: when the direct flat slot
is occupied (`flat.live = true`) but does not match the query (Case (i) `Sigma` mismatch or Case (ii)
key mismatch), and the routed spill read answered `ok none` with no routed spill entry for the
query, the generated public read returns `ok state.default`, which is canonically
`ok (Apeiron.Table.get logical (sigma, key))`.

The pre-image took `hsummary : region.summary_maybe address.place = ok false` and derived the
routed-spill exclusion from `repR_spillSummaryMissSound`.  Both are gone: the summary word is
absent from the extraction and the certificate is absent from `Representation.lean`.  The `none`
answer and the exclusion are therefore premises (`SpillReadNoneSound`); this is `OPS-3` in the
module header, recorded, not hidden. -/
theorem word_stored_table_get_at_flat_occupied_miss
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
    (htail : SpillReadNoneSound state sigma key address shardAt region 16#usize)
    (hroute : Apeiron.GrowableGeometry.preparedFlatIndex address state.resident.lanes.val = some index.val ∨
      ∀ idx value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.flat idx) ≠ some ((sigma, key), value)) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
  have hflatExcl : ∀ idx value,
      Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption (.flat idx) ≠
        some ((sigma, key), value) := by
    cases hroute with
    | inl hr =>
        have hm : (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok false) ∨
            (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true ∧
             Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok false) := by
          rcases hmismatch with h | h
          · exact Or.inl h
          · exact Or.inr h
        exact flat_occupied_exclusion_of_prepared state direct live logical hreach sigma key address haddress
          index flat hslot hlive hm hr
    | inr hexcl => exact hexcl
  have hidx : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot StoredI64 I64))
      state.resident.slots index = ok flat := by
    simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hslot]
  have hresident : table.Table.resident_exact StoredHomeInst StoredKeyRepInst state sigma address key
      = ok table.ResidentRead.Miss := by
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
  have habsent := absent_of_exclusions hreach sigma key address haddress shardAt hshard hflatExcl htail.excludes
  have hdefault := default_of_absent hreach (sigma, key) habsent
  rw [get_at_default_of_resident_miss state sigma key address haddress hresident shardAt hshard
    region hregion htail.read, hdefault]

/-! ## Deliverable 1 at the generated spill read

The pre-image's `spill_read_some_tagged_full_match` read a `U8` tag back from a parallel
`region.tags` array and used `table.scan.match_tag` to locate the query's lane inside a group.
Neither exists: the marker now lives in the slot record and the identity guard is a `U64`
coordinate comparison.  The theorem is restated over `table.SpillSlot` and over the current
admission chain, with the same conclusion modulo the tag array, which is now a field of the entry. -/

/-- **Deliverable 1 at the generated spill read.**  On a run the current body admits — a non-empty
region, a valid divisor, a computed `home` and `wanted`, a successful home coordinate read, and the
two reads of the home lane and the cycle — a generated `spill_read` hit returns an in-bounds slot
whose payload is a complete `(Sigma, key)` match for the query.  This is
`Apeiron.GrowableReadLoop.spill_read_some_implies_full_match`; it is re-exported here under the
name the downstream composition reads. -/
theorem spill_read_some_full_match_readback
    (region : table.Region StoredI64 I64)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : I64)
    (n : Usize) (home : Usize) (wanted found : U64) (slot : Usize)
    (hnonempty : alloc.vec.Vec.len region.slots ≠ 0#usize)
    (hvalid : table.valid_n n = ok true)
    (hhome : table.Region.home region address.place = ok home)
    (hwanted : table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place =
      ok wanted)
    (hfound : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
      region.coords home = ok found)
    (hhomeSlot : found ≠ table.COORD_EMPTY → found = wanted →
      ∃ ss : StoredSpillSlot, alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice StoredSpillSlot) region.slots home = ok ss)
    (hhomeMatch : found ≠ table.COORD_EMPTY → found = wanted →
      ∀ ss : StoredSpillSlot,
        alloc.vec.Vec.index
          (core.slice.index.SliceIndexUsizeSlice StoredSpillSlot) region.slots home =
          ok ss →
        ∃ matched : Bool,
          table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address = ok matched)
    (hreads : ∀ (bound mask : Usize), table.Region.bound region = ok bound →
      bound - 1#usize = ok mask →
      Apeiron.GrowableReadLoop.TailReadSuccess Apeiron.GrowableIdentity.storedKeyRep region
        sigma address.place address.direct_word key home bound mask wanted 1#usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma
      address key n = ok (some slot)) :
    slot.val < region.slots.length ∧
      ∃ (entry : StoredSpillSlot),
        region.slots.val[slot.val]? = some entry ∧
        Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held = (sigma, key) :=
  Apeiron.GrowableReadLoop.spill_read_some_implies_full_match region sigma address key n home wanted
    found slot hnonempty hvalid hhome hwanted hfound hhomeSlot hhomeMatch hreads hresult

/-- A successful generated lookup is in bounds and reads exactly the stored list entry. -/
private theorem spill_slot_readback {T : Type} (v : alloc.vec.Vec T) (i : Usize) (entry : T)
    (hindex : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice T) v i = ok entry) :
    i.val < v.length ∧ v.val[i.val]? = some entry := by
  have hiu : alloc.vec.Vec.index_usize v i = ok entry := by
    simpa only [alloc.vec.Vec.index_slice_index] using hindex
  rw [alloc.vec.Vec.index_usize] at hiu
  generalize hscrut : (v : List T)[i.val]? = g at hiu
  cases g with
  | none => simp at hiu
  | some payload =>
      have he : payload = entry := Result.ok_injective hiu
      refine ⟨(List.getElem?_eq_some_iff.mp hscrut).1, ?_⟩
      rw [← he]

/-- **Region-level occupancy.**  Every stored payload of the region carries a nonzero occupancy
marker.  This is the current replacement for the pre-image's `htagSlots`/`htagNonzero` pair, which
read a tag out of the parallel `region.tags` array: the marker is now the slot record's own `tag`,
so one region-level fact replaces the per-slot readback.  It is exactly the premise
`Apeiron.GrowableScan.spill_probe_bound_absent_sound` names as `occupied`, and it follows from
`Apeiron.GrowableScan.SpillPlacement.liveExact` together with
`region.live.val = region.slots.val.length` (the shard is full), which `Growable.Read` does not
assume; see `OPS-4` in the module header. -/
structure SpillRegionOccupied (region : table.Region StoredI64 I64) : Prop where
  /-- Every in-bounds stored slot of the region carries a nonzero occupancy marker. -/
  occupied : ∀ (index : Usize) (entry : StoredSpillSlot), index.val < region.slots.val.length →
    region.slots.val[index.val]? = some entry → entry.tag ≠ 0#u8

/-- Deliverable 1: on an admitted run, a generated `spill_read` hit returns an entry which reads
back through `Apeiron.GrowableSearch.entryAt` with the exact query and payload value, and the same
read-back slot is the one `table.probe.spill_read`'s consumer indexes.

The pre-image's tag half (`region.tags.val[slot.val]? = some tag ∧ tag ≠ 0#u8`) is restated as the
two facts the merged schema makes true: the occupancy marker is the `table.SpillSlot` record's own
`tag`, supplied by `SpillRegionOccupied`, and the record indexed is the whole payload.  The
`htag/htagNonzero` pair is a *state* fact now, not a run fact: the current `table.matches` consults
`slot.sigma`, `slot.held` and `slot.value` and never reads `slot.tag`. -/
theorem spill_read_some_entryAt
    (state : WordTable StoredI64)
    (shard : Usize) (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (hoccupied : SpillRegionOccupied region)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : I64)
    (n : Usize) (home : Usize) (wanted found : U64) (slot : Usize)
    (hnonempty : alloc.vec.Vec.len region.slots ≠ 0#usize)
    (hvalid : table.valid_n n = ok true)
    (hhome : table.Region.home region address.place = ok home)
    (hwanted : table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place =
      ok wanted)
    (hfound : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
      region.coords home = ok found)
    (hhomeSlot : found ≠ table.COORD_EMPTY → found = wanted →
      ∃ ss : StoredSpillSlot, alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice StoredSpillSlot) region.slots home = ok ss)
    (hhomeMatch : found ≠ table.COORD_EMPTY → found = wanted →
      ∀ ss : StoredSpillSlot,
        alloc.vec.Vec.index
          (core.slice.index.SliceIndexUsizeSlice StoredSpillSlot) region.slots home =
          ok ss →
        ∃ matched : Bool,
          table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address = ok matched)
    (hreads : ∀ (bound mask : Usize), table.Region.bound region = ok bound →
      bound - 1#usize = ok mask →
      Apeiron.GrowableReadLoop.TailReadSuccess Apeiron.GrowableIdentity.storedKeyRep region
        sigma address.place address.direct_word key home bound mask wanted 1#usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma
      address key n = ok (some slot)) :
    ∃ (value : I64) (tag : U8),
      Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.spill shard slot) = some ((sigma, key), value) ∧
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice StoredSpillSlot)
        region.slots slot =
          ok { tag := tag, sigma := sigma, held := { key := key }, value := value } := by
  obtain ⟨hinBounds, entry, hslot, hlogical⟩ :=
    spill_read_some_full_match_readback region sigma address key n home wanted found slot
      hnonempty hvalid hhome hwanted hfound hhomeSlot hhomeMatch hreads hresult
  have hp : (entry.sigma, entry.held.key) = (sigma, key) := by
    simpa [Apeiron.GrowableIdentity.logicalStored, Apeiron.GrowableIdentity.decodeStored] using
      hlogical
  have hsigma : entry.sigma = sigma := (Prod.mk.inj hp).1
  have hkey : entry.held.key = key := (Prod.mk.inj hp).2
  have hheld : entry.held = { key := key } := by
    cases h : entry.held with
    | mk k =>
        have hk : k = key := by
          have hk' := congrArg (fun x : StoredI64 => x.key) h
          rw [hk'] at hkey
          exact hkey
        subst hk
        rfl
  have htagNonzero : entry.tag ≠ 0#u8 :=
    hoccupied.occupied slot entry (by simpa using hinBounds) hslot
  refine ⟨entry.value, entry.tag, ?_, ?_⟩
  · simp [Apeiron.GrowableSearch.entryAt, hregion, hslot, htagNonzero,
      Apeiron.GrowableSearch.decodeStoredOption, hsigma, hkey]
  · have hentry' : entry = ⟨entry.tag, sigma, ⟨key⟩, entry.value⟩ := by
      obtain ⟨entryTag, slotSigma, entryHeld, entryValue⟩ := entry
      congr 1
    rw [hentry'] at hslot
    simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hslot]

/-! ## Deliverable 2: a `none` result excludes the lanes the run read

The retired `spill_read_none_implies_no_full_match` (and its whole-shard form
`spill_read_none_implies_no_shard_full_match`) concluded `∀ index, entryAt (.spill shard index) ≠ …`
from a `none` run.  That conclusion is **not** restated: it needs every slot of the region to be
read by the run, and `OPS-1` in the module header leaves `table.Region.home region place` possibly
at or above the capacity, so the current body need not touch every slot.  What *is* restated is
the per-lane form the body does support, at both places a `none` run can end:

* the **home lane**, when the coordinate word there is the vacancy marker
  (`home_vacancy_excludes_live`), and
* every lane the bounded cycle **observed** (`observed_lane_excludes_live_full_match`, general in
  `sigma` and `key`), composed by `spill_read_none_excludes_read_lanes`.

The composition keeps the two `none` sources apart, because they license different obligations:
the home vacancy excludes exactly one lane and the exhausted cycle excludes the lanes it read. -/

/-- A `do`-bind whose source is a `Result` records its value and its continuation. -/
private theorem spill_bind_inv {α β : Type} (x : Result α) (f : α → Result β) (b : β)
    (h : (do let y ← x; f y) = ok b) : ∃ y, x = ok y ∧ f y = ok b := by
  cases hx : x with
  | vis eff k =>
      rw [hx] at h
      simp only [bind_tc_vis] at h
      exact (vis_not_ok h).elim
  | div =>
      rw [hx] at h
      simp only [bind_tc_div] at h
      exact (div_not_ok h).elim
  | ret y =>
      rw [hx] at h
      simp only [bind_tc_ok] at h
      exact ⟨y, rfl, h⟩

/-- **No place is the vacancy marker.**  The generated `table.Region.coord_of` maps `place >>> 8` to
itself, except that the one value equal to `core.num.U64.MAX` is replaced by one less; so it never
returns `table.COORD_EMPTY` (`proof/ApeironWord.lean:17375-17382`). -/
theorem region_coord_of_ne_coord_empty {R V : Type} (place : U64) :
    table.Region.coord_of R V place ≠ ok table.COORD_EMPTY := by
  intro heq
  unfold table.Region.coord_of at heq
  obtain ⟨c, _hc, hrest⟩ := spill_bind_inv (place >>> table.SHARD_BITS)
    (fun c => if c = table.COORD_EMPTY then table.COORD_EMPTY - 1#u64 else ok c)
    table.COORD_EMPTY heq
  by_cases h : c = table.COORD_EMPTY
  · rw [if_pos h] at hrest
    have hspec := UScalar.sub_equiv table.COORD_EMPTY (1#u64)
    rw [hrest] at hspec
    have hone : (1#u64 : U64).val = 1 := rfl
    obtain ⟨_hle, heq, _hbv⟩ := hspec
    rw [hone] at heq
    omega
  · rw [if_neg h] at hrest
    exact h (by simpa using hrest)

/-- **Home-lane absence, by the vacancy marker.**  Under the live coordinate law, a live slot at the
home lane carries `table.Region.coord_of` of its own stored address' place at `region.coords`, and
`region_coord_of_ne_coord_empty` says that is never the vacancy marker.  So a coordinate word of
`table.COORD_EMPTY` at the home lane leaves no live slot there, and in particular no live full
match for the query.  No run equation is needed: the marker is what the generated home probe
returns on its `found = table.COORD_EMPTY` branch. -/
theorem home_vacancy_excludes_live
    (state : WordTable StoredI64)
    (shard : Usize) (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (hcoord : Apeiron.GrowableReadLoop.StoredLiveCoordAddress state shard)
    (home : Usize) (found : U64) (sigma : cell.Sigma) (key : I64)
    (hfound : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
      region.coords home = ok found)
    (hvacant : found = table.COORD_EMPTY) :
    ∀ (entry : StoredSpillSlot),
      region.slots.val[home.val]? = some entry → entry.tag ≠ 0#u8 →
      Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held ≠ (sigma, key) := by
  obtain ⟨_hfoundB, hfoundB⟩ := spill_slot_readback region.coords home found hfound
  rw [hvacant] at hfoundB
  intro entry hslot hlive
  obtain ⟨address, coord, haddress, hshard, hcoordOf, hcoordB⟩ :=
    hcoord home region entry.tag entry.sigma entry.held entry.value hregion hslot hlive
  rw [hfoundB] at hcoordB
  have hcoordEq : table.COORD_EMPTY = coord := Option.some.inj hcoordB
  have hne := region_coord_of_ne_coord_empty (R := Apeiron.GrowableIdentity.StoredI64)
    (V := I64) address.place
  rw [hcoordOf] at hne
  exact fun _ => (hne (congrArg ok hcoordEq.symm)).elim

/-- **Per-observed-lane absence, general in the query.**  A lane the bounded cycle recorded an
observation at holds no live full match for the query: the record either refused the generated
matcher call or held a coordinate word other than the query's.  This is
`Apeiron.GrowableReadLoop.observed_lane_excludes_live_full_match` at an arbitrary `sigma`, `key`
and lane — that theorem is stated at the concrete carrier `cell.Sigma.Parabolic` / key `0`, so the
general form is proved here rather than by instantiation. -/
theorem observed_lane_excludes_live_full_match
    (region : table.Region StoredI64 I64)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64)
    (wanted : U64) (lane : Usize)
    (hobs : Apeiron.GrowableReadLoop.TailLaneObserved Apeiron.GrowableIdentity.storedKeyRep
      region sigma place direct key lane wanted)
    (hlivecoord : ∀ (entry : StoredSpillSlot),
      region.slots.val[lane.val]? = some entry → entry.tag ≠ 0#u8 →
      Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held = (sigma, key) →
      region.coords.val[lane.val]? = some wanted) :
    ∀ (entry : StoredSpillSlot),
      region.slots.val[lane.val]? = some entry → entry.tag ≠ 0#u8 →
      Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held ≠ (sigma, key) := by
  obtain ⟨coord, hcoord, hzero, hexcl⟩ := hobs
  obtain ⟨_hcoordB0, hcoordB⟩ := spill_slot_readback region.coords lane coord hcoord
  intro entry hslot hlive hfull
  have hcoordEq : coord = wanted :=
    Option.some.inj (hcoordB.symm.trans (hlivecoord entry hslot hlive hfull))
  rcases hexcl with hne | ⟨ss, hss, hmt⟩
  · exact (hne hcoordEq).elim
  · obtain ⟨_hssB, hssB⟩ := spill_slot_readback region.slots lane ss hss
    have hssEq : ss = entry := by
      rw [hslot] at hssB
      exact (Option.some.inj hssB).symm
    subst ss
    have hp : (entry.sigma, entry.held.key) = (sigma, key) := by
      simpa [Apeiron.GrowableIdentity.logicalStored, Apeiron.GrowableIdentity.decodeStored]
        using hfull
    obtain ⟨hps, hpk⟩ := Prod.mk.inj hp
    have hmatchTrue : table.matches Apeiron.GrowableIdentity.storedKeyRep
        { tag := entry.tag, sigma := entry.sigma, held := entry.held, value := entry.value }
        sigma key { place := place, direct_word := direct } = ok true :=
      (Apeiron.GrowableIdentity.stored_matches_iff entry.sigma sigma entry.held entry.value key
        { place := place, direct_word := direct }).mpr (by
          simpa [Apeiron.GrowableIdentity.logicalStored, Apeiron.GrowableIdentity.decodeStored,
            hps, hpk] using hfull)
    have hcontra : (false : Bool) = true := Result.ok_injective (hmt.symm.trans hmatchTrue)
    exact absurd hcontra (by simp)

/-- **Every lane a run observed excludes live full matches.**  The `TailSweepInv` of a `none` run
records a lane observation at every executed step below its stopping point; each observation
excludes live full matches by `observed_lane_excludes_live_full_match`.

The live-coordinate step needs the query's own address: a live slot at an observed lane carries
`table.Region.coord_of` of *its own* stored key's address place, so when that slot is the query's
key the two address equations agree by `Result.ok_injective` and the coordinate words agree.  This
is why the premise is stated at the conditional `logicalStored … = (sigma, key)` rather than at the
lane. -/
theorem observed_lanes_exclude_live
    (state : WordTable StoredI64)
    (shard : Usize) (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (hcoord : Apeiron.GrowableReadLoop.StoredLiveCoordAddress state shard)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : I64)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (place : U64) (direct : Option U32) (home bound mask : Usize) (wanted : U64)
    (hwanted : table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place = ok wanted)
    (start stop : Usize)
    (hinv : Apeiron.GrowableReadLoop.TailSweepInv Apeiron.GrowableIdentity.storedKeyRep
      region sigma place direct key home bound mask wanted start stop) :
    ∀ (lane : Usize),
      (∃ k : Nat, start.val ≤ k ∧ k < stop.val ∧
        ∃ st i1 : Usize, st.val = k ∧ home + st = ok i1 ∧ i1 &&& mask = lane) →
      ∀ (entry : StoredSpillSlot),
        region.slots.val[lane.val]? = some entry → entry.tag ≠ 0#u8 →
        Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held ≠ (sigma, key) := by
  intro lane hvisited entry hslot hlive
  obtain ⟨k, hk0, hk1, st, i1, hst, hadd, hilane⟩ := hvisited
  obtain ⟨st', i1', hst', hadd', coord, hcoordIdx, hzero, hexcl⟩ := hinv k hk0 hk1
  have hstEq : st = st' := UScalar.eq_of_val_eq (hst.trans hst'.symm)
  have hadd'' : home + st = ok i1' := by rw [hstEq]; exact hadd'
  have hiEq : i1 = i1' := Result.ok_injective (hadd.symm.trans hadd'')
  have hslotAt : region.slots.val[(i1' &&& mask).val]? = some entry := by
    rw [← hilane, hiEq] at hslot
    exact hslot
  have hlivecoord : ∀ (e : StoredSpillSlot),
      region.slots.val[(i1' &&& mask).val]? = some e → e.tag ≠ 0#u8 →
      Apeiron.GrowableIdentity.logicalStored e.sigma e.held = (sigma, key) →
      region.coords.val[(i1' &&& mask).val]? = some wanted := by
    intro e hslothe hlivee hfull
    obtain ⟨address', coord', haddress', _hshard', hcoordOf, hw⟩ :=
      hcoord (i1' &&& mask) region e.tag e.sigma e.held e.value hregion hslothe hlivee
    have hp : (e.sigma, e.held.key) = (sigma, key) := by
      simpa [Apeiron.GrowableIdentity.logicalStored, Apeiron.GrowableIdentity.decodeStored] using
        hfull
    obtain ⟨hps, hpk⟩ := Prod.mk.inj hp
    have haddEq : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address' := by
      rw [← hps, ← hpk]
      exact haddress'
    have hAddrEq : address' = address := Result.ok_injective (haddEq.symm.trans haddress)
    have hcw : coord' = wanted := by
      rw [hAddrEq] at hcoordOf
      exact Result.ok_injective (hcoordOf.symm.trans hwanted)
    rw [hcw] at hw
    exact hw
  exact observed_lane_excludes_live_full_match region sigma place direct key wanted (i1' &&& mask)
    ⟨coord, hcoordIdx, hzero, hexcl⟩ hlivecoord entry hslotAt hlive

/-- **A `none` run excludes live full matches at the lanes it read.**  The generated
`table.probe.spill_read` returns `none` on exactly two live paths once admitted: the vacancy marker
already at the home lane, or a bounded cycle that ran and returned `none`
(`Apeiron.GrowableReadLoop.generated_spill_read_branch`).  The first excludes the home lane
(`home_vacancy_excludes_live`); the second records a `TailSweepInv` up to its stopping step, whose
every observed lane excludes live full matches (`observed_lanes_exclude_live`).  The two are kept
apart because they license different obligations: the first covers one lane, the second only the
lanes read.  This is the current replacement for `spill_read_none_implies_no_full_match`; the
whole-region form is `OPS-1`/`OPS-3` and is **not** claimed. -/
theorem spill_read_none_excludes_read_lanes
    (state : WordTable StoredI64)
    (shard : Usize) (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (hcoord : Apeiron.GrowableReadLoop.StoredLiveCoordAddress state shard)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : I64)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (n : Usize) (home : Usize) (wanted found : U64)
    (hnonempty : alloc.vec.Vec.len region.slots ≠ 0#usize)
    (hvalid : table.valid_n n = ok true)
    (hhome : table.Region.home region address.place = ok home)
    (hwanted : table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place =
      ok wanted)
    (hfound : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
      region.coords home = ok found)
    (hhomeSlot : found ≠ table.COORD_EMPTY → found = wanted →
      ∃ ss : StoredSpillSlot, alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice StoredSpillSlot) region.slots home = ok ss)
    (hhomeMatch : found ≠ table.COORD_EMPTY → found = wanted →
      ∀ ss : StoredSpillSlot,
        alloc.vec.Vec.index
          (core.slice.index.SliceIndexUsizeSlice StoredSpillSlot) region.slots home =
          ok ss →
        ∃ matched : Bool,
          table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address = ok matched)
    (hreads : ∀ (bound mask : Usize), table.Region.bound region = ok bound →
      bound - 1#usize = ok mask →
      Apeiron.GrowableReadLoop.TailReadSuccess Apeiron.GrowableIdentity.storedKeyRep region
        sigma address.place address.direct_word key home bound mask wanted 1#usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma
      address key n = ok none) :
    (found = table.COORD_EMPTY ∧
        ∀ (entry : StoredSpillSlot), region.slots.val[home.val]? = some entry →
          entry.tag ≠ 0#u8 →
          Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held ≠ (sigma, key)) ∨
    (∃ (bound mask stop : Usize), table.Region.bound region = ok bound ∧
        bound - 1#usize = ok mask ∧
        table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep region sigma
          address.place address.direct_word key home bound mask wanted 1#usize = ok none ∧
        ∀ (lane : Usize),
          (∃ k : Nat, 1 ≤ k ∧ k < stop.val ∧
            ∃ st i1 : Usize, st.val = k ∧ home + st = ok i1 ∧ i1 &&& mask = lane) →
          ∀ (entry : StoredSpillSlot), region.slots.val[lane.val]? = some entry →
            entry.tag ≠ 0#u8 →
            Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held ≠ (sigma, key)) := by
  have hbranch := Apeiron.GrowableReadLoop.generated_spill_read_branch region sigma address key n
    home wanted found none hnonempty hvalid hhome hwanted hfound hhomeSlot hhomeMatch hresult
  rcases hbranch with hfirst | hrestB
  · rcases hfirst with ⟨hnone, hzero⟩
    exact Or.inl ⟨hzero, home_vacancy_excludes_live state shard region hregion hcoord home found
      sigma key hfound hzero⟩
  · rcases hrestB with hsecond | hthird
    · rcases hsecond with ⟨ss, hss, hmt, hslotEq⟩
      exact absurd hslotEq (by simp)
    · obtain ⟨bound, mask, hbound, hmask, hsweep⟩ := hthird
      rcases Apeiron.GrowableReadLoop.generated_tail_sweep_outcome
        Apeiron.GrowableIdentity.storedKeyRep region sigma address.place address.direct_word
        key home bound mask wanted 1#usize none
        (hreads bound mask hbound hmask).coord (hreads bound mask hbound hmask).slot
        (fun _ ss _ => Apeiron.GrowableReadLoop.stored_matcher_total ss sigma key address) hsweep with
        ⟨stop, _hr, hinv, _hstop⟩ | ⟨stop, i1, lane, ss, hr, hinv, hadd, hilane, hss, hmt⟩
      · refine Or.inr ⟨bound, mask, stop, hbound, hmask, hsweep, ?_⟩
        intro lane0 hvisited entry hslothe hlivee
        exact observed_lanes_exclude_live state shard region hregion hcoord sigma address key
          haddress address.place address.direct_word home bound mask wanted hwanted 1#usize stop hinv
          lane0 hvisited entry hslothe hlivee
      · exact absurd hr (by simp)

def emptyControlRegion : table.Region (table.keyrep.Stored Std.I64) Std.I64 := {
  slots := alloc.vec.Vec.new _
  coords := alloc.vec.Vec.new _
  bits := 0#u32
  live := 0#usize
}

/-- Control fixture for Partition 2 with `resident.lanes = 0#usize` (shape B). -/
def controlFixture : table.Table Std.I64 Std.I64 table.WordHome (table.keyrep.Stored Std.I64) 16#usize := {
  default := 0#i64,
  home := (),
  resident := { lanes := 0#usize, slots := alloc.vec.Vec.new _ },
  shards := alloc.vec.Vec.from (List.replicate 256 emptyControlRegion) (by scalar_tac),
  len := 0#usize,
  key := (),
  locator := {
    profile := none,
    directory := none,
    links := alloc.vec.Vec.new _,
    admitted := 0#usize,
    staged := none
  }
}

-- The control fixture is exercised by the composition module's controls; the guard form
-- `word_stored_table_get_at .. == ok ..` needs `BEq (Result I64)`, which this environment has not.

/-- The admission gates of the current `table.probe.spill_read` body are live at the routed
divisor the public read uses: the divisor `16#usize` is admitted (`spill_read_valid_n_16`) and
the empty control region answers `ok none` through the generated
`len region.slots != 0#usize` gate (`empty_region_answers_none`, a kernel check, not a sampled
value).  Without them neither gate has a reachable state at this divisor. -/
theorem empty_region_answers_none :
    table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep emptyControlRegion
      cell.Sigma.Parabolic { place := 0#u64, direct_word := none } 0#i64 16#usize = ok none :=
  spill_read_none_of_empty_region _ _ _ _ _ _ rfl

-- Local fail-closed cone check until the integration owner registers these declarations in
-- AxiomAudit.lean. The shared audit file is outside this task's ownership.
open Lean Elab Command in
run_cmd do
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  let external : Array Lean.Name := #[``Apeiron.GrowableIdentity.stored_matches_iff,
    ``Apeiron.GrowableIdentity.regen_slot_matches_iff_of_reachable,
    ``Apeiron.GrowableIdentity.regen_hold_decodes,
    ``Apeiron.GrowableRepresentation.RepR]
  let localNames : Array Lean.Name := #[`ConstructorReachableRepR,
    `word_stored_table_get_at_direct_hit, `word_regen_table_get_at_direct_hit,
    `query_rep, `absent_of_exclusions, `default_of_absent,
    `spill_read_none_of_empty_region, `spill_read_none_of_invalid_n, `spill_read_valid_n_16,
    `SpillReadNoneSound,
    `flat_index_none_pin, `flat_index_some_pin, `flat_index_compute,
    `word_stored_table_get_at_direct_none,
    `word_stored_table_get_at_flat_none,
    `flat_vacant_entry_none, `flat_vacant_exclusion_of_prepared,
    `word_stored_table_get_at_flat_vacant,
    `flat_occupied_ne_exclusion, `flat_occupied_exclusion_of_prepared,
    `word_stored_table_get_at_flat_occupied_miss,
    `spill_read_some_full_match_readback,
    `SpillRegionOccupied,
    `spill_read_some_entryAt,
    `region_coord_of_ne_coord_empty,
    `home_vacancy_excludes_live,
    `observed_lane_excludes_live_full_match,
    `observed_lanes_exclude_live,
    `spill_read_none_excludes_read_lanes, `empty_region_answers_none]
  for decl in external ++ localNames.map (Lean.Name.append `Apeiron.GrowableRead) do
    let axioms ← liftTermElabM <| Lean.collectAxioms decl
    let bad := axioms.filter (fun ax => !allowed.contains ax)
    unless bad.isEmpty do
      throwError m!"'{decl}' depends on axioms outside the closed cone: {bad.toList}"
    logInfo m!"'{decl}' depends on axioms: {axioms.toList}"

end Apeiron.GrowableRead