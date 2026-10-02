import Growable.ReadLoop
import Lean

/-!
# Public read refinement: `word_{stored,regen}_table_get_at` / `Table.get_at`

Row 6 of the read map (`docs/epics/read-proof.md`). The generated public read
roots are refined to `ok (Apeiron.Table.get logical (sigma, key))` under explicit premises,
partitioned over the generated `table.Table.get_at` case tree:

1. **direct hit** — flat record live, `Sigma` equal, generated matcher `ok true`;
2. **direct empty** — no usable direct record (direct word absent, record out of extent, or
   vacant record), answered from the routed linear-probe spill read or the `DirectCover` default;
3. **occupied non-match** — flat record live but `Sigma`/matcher mismatch, spill fallback;
4. **routed spill miss** — the routed region's `table.probe.spill_read` returns `ok none`
   under the live tag/address invariant `StoredLiveTagAddress`;
5. **linear-probe spill hit** — `spill_read = ok (some slot)` with the landed slot readback
   (`spill_read_some_implies_full_match`) and the live tag/address invariant
   `StoredLiveTagAddress`.

No caller-supplied found/absent result, desired location or public-read equality is a premise:
every premise is a state-side certificate (`RepR`, the live tag/address invariant) or an
equation of a generated intermediate computation (the generated branch tests `Flat.index`, live
bit, `Sigma` comparison, carrier matcher call, `spill_read` run).
Full `RepR` preservation across writes and rebuild is row 4's open part of the read map, so the
reachability of the represented state is carried as the explicit named premise
`ConstructorReachableRepR`; it is never assumed. The rebuild traversal carries its own
certificate (`word_stored_relocate_all_keeps_reserved_tags_from_prepared` under its exact
`hstep`). The `RelocationEffectsPrepared.publish` spill theorems are audit targets
(`AxiomAudit.lean:530-531`); their publish obligation is pinned to the prepared spill facts
(`sourcePayload`, `address_held`, `reserved.regions[shard]`) and discharged from the consumer's
`hspill` facts, not quantified over an arbitrary region. This read traversal does not use them.

The linear-probe sweep needs state-side premises beyond the live tag/address invariant, because
the invariant does not order occupied slots relative to the cycle. Counterexample: with `home = 0`
and `slots[0].tag = 0#u8` in front of a live full `(Sigma, key)` match at `slots[1]`, the generated
`table.probe.spill_read` returns `ok none` while `StoredLiveTagAddress` holds. Every miss theorem
therefore carries `hchain : Apeiron.GrowableReadLoop.StoredProbeChainSafe region address.place sigma
key` (a live full match lies on its cycle before any zero tag); `hvalid : table.valid_n n = ok true`
stays explicit on the generic miss lemma `spill_read_none_entryAt_ne` and is discharged by
`valid_n_16` at the fixed `n = 16#usize` call sites. The hit direction needs no readiness premise:
the run-anchored `generated_probe_run_some` certificate reads each executed step's admission off the
run itself.
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

/--
The named reachability premise of the public read refinement: `RepR` at a state reachable from
the generated constructor `table.Table.new` through the landed write/rebuild certificate chain
(`word_stored_table_new_repR`, `word_regen_table_new_repR`, the `*keeps_tags` publication
facts, and the rebuild traversal certificate). Full `RepR` preservation across writes and
rebuild is row 4's open part of the read map, so this premise is carried explicitly by every
read theorem and is not assumed derivable here. -/
def ConstructorReachableRepR {R : Type} (state : WordTable R)
    (decode : cell.Sigma → R → Option I64) (direct : LogicalKey → Option Location)
    (live : LogicalKey → Prop) (logical : Apeiron.Table.HMap LogicalKey I64) : Prop :=
  Apeiron.GrowableRepresentation.RepR state decode direct live logical

/-- `table.valid_n 16#usize = ok true` discharges by definition. Defined here so the fixed
`n = 16#usize` read partitions can discharge the generic miss lemma's `hvalid` premise. -/
theorem valid_n_16 : table.valid_n 16#usize = ok true := by
  unfold table.valid_n table.MAX_GROUP core.num.Usize.is_power_of_two
  rfl

/-- Partition 1 (direct hit), Stored carrier: the generated public read returns the flat slot
value at a full generated matcher hit, and that value is the canonical `Apeiron.Table.get`.
Premises are the generated direct-route branch tests (`hword`, `hflat`, `hslot`, `hlive`,
`hsigmaEq`, `hraw`) and the full generated matcher equation `hmatch` they compose. -/
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
    (hraw : Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok true)
    (hmatch : table.matches Apeiron.GrowableIdentity.storedKeyRep
      { tag := 0#u8, sigma := flat.sigma, held := flat.held, value := flat.value }
      sigma key address = ok true) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
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
  simp only [word_stored_table_get_at]
  unfold table.Table.get_at
  simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddress,
    hword, hflat]
  have hindex : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot StoredI64 I64))
      state.resident.slots index = ok flat := by
    simpa [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hslot] using
      (rfl : (ok flat : Result (table.FlatSlot StoredI64 I64)) = ok flat)
  simp_all

/-- Partition 1 (direct hit), Regen carrier: the generated public read returns the flat slot
value at a full generated matcher hit on a reachable held key, and that value is the canonical
`Apeiron.Table.get`. The held-key reachability premise (`word_regen_hold`) is carried
explicitly and named (`hheld`). -/
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
    (hraw : Apeiron.GrowableIdentity.regenKeyRep.«matches» sigma flat.held key address = ok true)
    (hmatch : table.matches Apeiron.GrowableIdentity.regenKeyRep
      { tag := 0#u8, sigma := flat.sigma, held := flat.held, value := flat.value }
      sigma key address = ok true) :
    word_regen_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key)) := by
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
  simp only [word_regen_table_get_at]
  unfold table.Table.get_at
  simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddress,
    hword, hflat]
  have hindex : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (table.FlatSlot RegenI64 I64))
      state.resident.slots index = ok flat := by
    simpa [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hslot] using
      (rfl : (ok flat : Result (table.FlatSlot RegenI64 I64)) = ok flat)
  simp_all

/-!
## Localization and absence helpers for the read partitions

Every physical representative of a queried full key is either at the query's direct flat
index or in its routed shard. Together with per-route flat exclusion and routed spill
exclusion this yields `Absent`, and `ObservationCorrect.2` then fixes the canonical `get`
at `state.default` ("none represented as the default"). These are the semantic halves of
partitions 2-4; the `preparedFlatIndex`/`table.Flat.index` pinning and the sweep-tail
replay build on them (open, see the task report).
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

/- The retired summary lanes have no successor: the routed read is the bounded linear probe
`table.probe.spill_read`, which inspects the single `SpillSlot` vector and answers `ok none`
only at a zero tag or an exhausted cycle. Its miss soundness is stated in
`spill_read_none_entryAt_ne` below from the ported `Growable.ReadLoop` sweep facts. -/

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

/-- A generated `spill_read` hit returns an in-bounds slot carrying a nonzero tag and the full
query key. This is the tag-carrying readback of the ported
`Growable.ReadLoop.spill_read_some_implies_full_match` over the single `SpillSlot` vector; the run
itself witnesses each executed step, so no readiness premise is needed. -/
theorem spill_read_some_tagged_full_match
    (region : table.Region StoredI64 I64)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : I64)
    (n slot : Usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok (some slot)) :
    slot.val < region.slots.length ∧
    ∃ (entry : table.SpillSlot StoredI64 I64) (tag : U8),
      region.slots.val[slot.val]? = some entry ∧
      entry.tag = tag ∧
      tag ≠ 0#u8 ∧
      Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held = (sigma, key) := by
  obtain ⟨hinBounds, entry, hslot, htagNonzero, hlogical⟩ :=
    Apeiron.GrowableReadLoop.spill_read_some_implies_full_match region sigma address key n slot
      hresult
  exact ⟨hinBounds, entry, entry.tag, hslot, rfl, htagNonzero, hlogical⟩

/-- Deliverable 1: a generated `spill_read` hit returns an in-bounds slot entry which reads back
through `entryAt` with the exact query and whose payload field is the returned value. -/
theorem spill_read_some_entryAt
    (state : WordTable StoredI64)
    (shard : Usize) (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : I64)
    (n slot : Usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok (some slot)) :
    ∃ entry : table.SpillSlot StoredI64 I64,
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
        (table.SpillSlot StoredI64 I64)) region.slots slot = ok entry ∧
      entry.sigma = sigma ∧
      entry.held = { key := key } ∧
      Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
        (.spill shard slot) = some ((sigma, key), entry.value) := by
  obtain ⟨_hinBounds, entry, tag, hslot, htag, htagNonzero, hlogical⟩ :=
    spill_read_some_tagged_full_match region sigma address key n slot hresult
  have hsigma : entry.sigma = sigma := by
    have hp : (entry.sigma, entry.held.key) = (sigma, key) := hlogical
    exact congrArg Prod.fst hp
  have hkey : entry.held.key = key := by
    have hp : (entry.sigma, entry.held.key) = (sigma, key) := hlogical
    exact congrArg Prod.snd hp
  have hheld : entry.held = { key := key } := by
    cases h : entry.held with
    | mk k =>
        have hk : k = key := by
          have hk' := congrArg (fun x : StoredI64 => x.key) h
          rw [hk'] at hkey
          exact hkey
        subst hk
        rfl
  have hentryTagNz : entry.tag ≠ 0#u8 := by
    rw [htag]
    exact htagNonzero
  have hindex : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
      (table.SpillSlot StoredI64 I64)) region.slots slot = ok entry := by
    simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hslot]
  have hentryAt : Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
      (.spill shard slot) = some ((sigma, key), entry.value) := by
    simp only [Apeiron.GrowableSearch.entryAt, hregion, hslot]
    rw [if_neg hentryTagNz]
    simp [Apeiron.GrowableSearch.decodeStoredOption, hsigma, hkey]
  exact ⟨entry, hindex, hsigma, hheld, hentryAt⟩

/-- Under the live tag/address invariant, a generated `spill_read` miss implies that no live slot
in that shard decodes to the query. The pointwise exclusion is sourced from the ported
`Growable.ReadLoop.spill_read_none_implies_no_full_match`, which needs `hvalid` and `hchain`: with
`slots[0].tag = 0#u8` before a live full match at `slots[1]` and `home = 0`, the probe returns
`ok none` while `StoredLiveTagAddress` holds, and `valid_n n = ok false` returns `ok none` without
scanning. -/
theorem spill_read_none_entryAt_ne
    (state : WordTable StoredI64)
    (hstate : Apeiron.GrowableReadLoop.StoredLiveTagAddress state)
    (shard : Usize) (region : table.Region StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (n : Usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok none)
    (hvalid : table.valid_n n = ok true)
    (hchain : Apeiron.GrowableReadLoop.StoredProbeChainSafe region address.place sigma key) :
    ∀ index value, Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
      (.spill shard index) ≠ some ((sigma, key), value) := by
  intro index value hentry
  have hnoMatch := Apeiron.GrowableReadLoop.spill_read_none_implies_no_full_match state hstate
    shard region hregion sigma key address haddress n hresult hvalid hchain
  unfold Apeiron.GrowableSearch.entryAt at hentry
  simp only [hregion] at hentry
  cases hslot : region.slots.val[index.val]? with
  | none => simp [hslot] at hentry
  | some entry =>
      simp only [hslot] at hentry
      by_cases hzero : entry.tag = 0#u8
      · simp [hzero] at hentry
      · simp only [if_neg hzero, Apeiron.GrowableSearch.decodeStoredOption] at hentry
        have hsome : ((entry.sigma, entry.held.key), entry.value) = ((sigma, key), value) :=
          Option.some.inj hentry
        have hpair : (entry.sigma, entry.held.key) = (sigma, key) := congrArg Prod.fst hsome
        have hfull : Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held = (sigma, key) :=
          hpair
        exact hnoMatch index entry hslot hzero hfull


/-- Partition 2 (direct empty, shape A: direct word absent), Stored carrier: the generated public
read returns `ok state.default` when `address.direct_word = none` and the routed linear-probe
spill read returns `ok none` under the live tag/address invariant, and that value is the
canonical `Apeiron.Table.get`. Contract change: the explicit `hchain` premise is required; without
it a zero tag before a live full match on the cycle makes the miss unsound and `RepR` does not
supply it. -/
theorem word_stored_table_get_at_direct_none
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
    (hmiss : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok none) :
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
  have hspillExcl := spill_read_none_entryAt_ne state hstate shardAt region hregion sigma key
    address haddress 16#usize hmiss valid_n_16 hchain
  have habsent := absent_of_exclusions hreach sigma key address haddress shardAt hshard hflat hspillExcl
  have hdefault := default_of_absent hreach (sigma, key) habsent
  have hrun : word_stored_table_get_at state sigma key = ok state.default := by
    simp only [word_stored_table_get_at]
    unfold table.Table.get_at
    have haddr : table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address state.home sigma key = ok address := haddress
    simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddr, hword, hshard]
    have hvec : alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (table.Region StoredI64 I64))
        state.shards shardAt = ok region := by
      simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hregion]
    simp only [hvec, bind_tc_ok, hmiss]
  rw [hrun, hdefault]

/-- Partition 2 (direct empty, shape B: flat index miss), Stored carrier: the generated public
read returns `ok state.default` when `address.direct_word = some word` and `Flat.index = ok none`,
and that value is the canonical `Apeiron.Table.get`. Contract change: the explicit `hchain` premise
is required for the probe miss (zero tag before a live full match), as in Partition 2 shape A. -/
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
    (hstate : Apeiron.GrowableReadLoop.StoredLiveTagAddress state)
    (hchain : Apeiron.GrowableReadLoop.StoredProbeChainSafe region address.place sigma key)
    (hmiss : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok none) :
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
  have hspillExcl := spill_read_none_entryAt_ne state hstate shardAt region hregion sigma key
    address haddress 16#usize hmiss valid_n_16 hchain
  have habsent := absent_of_exclusions hreach sigma key address haddress shardAt hshard hflatExcl hspillExcl
  have hdefault := default_of_absent hreach (sigma, key) habsent
  have hrun : word_stored_table_get_at state sigma key = ok state.default := by
    simp only [word_stored_table_get_at]
    unfold table.Table.get_at
    have haddr : table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address state.home sigma key = ok address := haddress
    simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_tc_ok, haddr, hword, hflat, hshard]
    have hvec : alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice (table.Region StoredI64 I64))
        state.shards shardAt = ok region := by
      simp [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, hregion]
    simp only [hvec, bind_tc_ok, hmiss]
  rw [hrun, hdefault]

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
  have habsent := absent_of_exclusions hreach sigma key address haddress shardAt hshard hflatExcl hspillExcl
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
    rfl
  rw [hrun, hdefault]

/-- When a flat slot is occupied (`flat.live = true`) but does not match the query (either by
`Sigma` mismatch or by key mismatch), the slot does not hold an entry for `(sigma, key)`. -/
theorem flat_occupied_ne_exclusion
    (state : WordTable StoredI64)
    (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (index : Usize) (flat : table.FlatSlot StoredI64 I64)
    (hslot : state.resident.slots.val[index.val]? = some flat)
    (hlive : flat.live = true)
    (hmismatch : (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok false) ∨
      (cell.Sigma.Insts.CoreCmpPartialEqSigma.eq flat.sigma sigma = ok true ∧
       Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok false) ∨
      (table.matches Apeiron.GrowableIdentity.storedKeyRep
        { tag := 0#u8, sigma := flat.sigma, held := flat.held, value := flat.value }
        sigma key address = ok false)) :
    Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
      (.flat index) ≠ some ((sigma, key), flat.value) := by
  intro hentry
  have heq : some ((flat.sigma, flat.held.key), flat.value) = some ((sigma, key), flat.value) := by
    simpa [Apeiron.GrowableSearch.entryAt, hslot, hlive,
      Apeiron.GrowableSearch.decodeStoredOption] using hentry
  have hinj := Option.some.inj heq
  have hsigma : flat.sigma = sigma := (Prod.mk.inj (Prod.mk.inj hinj).1).1
  have hkey : flat.held.key = key := (Prod.mk.inj (Prod.mk.inj hinj).1).2
  rcases hmismatch with hsigmaEq | ⟨hsigmaTrue, hkeyMatch⟩ | hmatches
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
        { tag := 0#u8, sigma := flat.sigma, held := flat.held, value := flat.value }
        sigma key address =
        Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address := by
      unfold table.matches
      simp [hsigmaTrue]
    rw [hmatchUnfold] at hmatchTrue
    rw [hmatchTrue] at hkeyMatch
    have heqRes := Result.ok_injective hkeyMatch
    cases heqRes
  · have hfull : Apeiron.GrowableIdentity.logicalStored flat.sigma flat.held = (sigma, key) := by
      simp [Apeiron.GrowableIdentity.logicalStored, Apeiron.GrowableIdentity.decodeStored,
        hsigma, hkey]
    have hmatchTrue :=
      (Apeiron.GrowableIdentity.stored_matches_iff flat.sigma sigma flat.held flat.value key address).mpr hfull
    rw [hmatchTrue] at hmatches
    have heqRes := Result.ok_injective hmatches
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
       Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok false) ∨
      (table.matches Apeiron.GrowableIdentity.storedKeyRep
        { tag := 0#u8, sigma := flat.sigma, held := flat.held, value := flat.value }
        sigma key address = ok false))
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
key mismatch), and the routed linear-probe spill read returns `ok none` under the live tag/address
invariant, the generated public read returns `ok state.default`, which is canonically
`ok (Apeiron.Table.get logical (sigma, key))`. Contract change: the explicit `hchain` premise is
required for the probe miss; without it a zero tag before a live full match on the cycle refutes the
conclusion and `RepR` does not supply it. -/
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
    (hstate : Apeiron.GrowableReadLoop.StoredLiveTagAddress state)
    (hchain : Apeiron.GrowableReadLoop.StoredProbeChainSafe region address.place sigma key)
    (hmiss : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key 16#usize = ok none)
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
             Apeiron.GrowableIdentity.storedKeyRep.«matches» sigma flat.held key address = ok false) ∨
            (table.matches Apeiron.GrowableIdentity.storedKeyRep
              { tag := 0#u8, sigma := flat.sigma, held := flat.held, value := flat.value }
              sigma key address = ok false) := by
          rcases hmismatch with h | h
          · exact Or.inl h
          · exact Or.inr (Or.inl h)
        exact flat_occupied_exclusion_of_prepared state direct live logical hreach sigma key address haddress
          index flat hslot hlive hm hr
    | inr hexcl => exact hexcl
  have hspillExcl := spill_read_none_entryAt_ne state hstate shardAt region hregion sigma key
    address haddress 16#usize hmiss valid_n_16 hchain
  have habsent := absent_of_exclusions hreach sigma key address haddress shardAt hshard hflatExcl hspillExcl
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
    · simp only [hsigma, bind_tc_ok, hshard, hvec, hmiss, ↓reduceIte]
      rfl
    · simp only [hsigma, bind_tc_ok, hshard, hvec, hmiss, ↓reduceIte]
      have hkey' : table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.matches
          I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple core.clone.CloneI64
          table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma sigma flat.held key address = ok false := hkey
      rw [hkey']
      simp only [bind_tc_ok]
      rfl
  rw [hrun, hdefault]


def emptyControlRegion : table.Region (table.keyrep.Stored Std.I64) Std.I64 := {
  slots := alloc.vec.Vec.new _,
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
    staged := none
  }
}

-- The control fixture is exercised by the composition module's controls; the guard form
-- `word_stored_table_get_at .. == ok ..` needs `BEq (Result I64)`, which this environment has not.

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
  let localNames : Array Lean.Name := #[`ConstructorReachableRepR, `valid_n_16,
    `word_stored_table_get_at_direct_hit, `word_regen_table_get_at_direct_hit,
    `query_rep, `absent_of_exclusions, `default_of_absent,
    `flat_index_none_pin, `flat_index_some_pin, `flat_index_compute,
    `word_stored_table_get_at_direct_none,
    `word_stored_table_get_at_flat_none,
    `flat_vacant_entry_none, `flat_vacant_exclusion_of_prepared,
    `word_stored_table_get_at_flat_vacant,
    `flat_occupied_ne_exclusion, `flat_occupied_exclusion_of_prepared,
    `word_stored_table_get_at_flat_occupied_miss,
    `spill_read_some_tagged_full_match,
    `spill_read_some_entryAt,
    `spill_read_none_entryAt_ne]
  for decl in external ++ localNames.map (Lean.Name.append `Apeiron.GrowableRead) do
    let axioms ← liftTermElabM <| Lean.collectAxioms decl
    let bad := axioms.filter (fun ax => !allowed.contains ax)
    unless bad.isEmpty do
      throwError m!"'{decl}' depends on axioms outside the closed cone: {bad.toList}"
    logInfo m!"'{decl}' depends on axioms: {axioms.toList}"

end Apeiron.GrowableRead
