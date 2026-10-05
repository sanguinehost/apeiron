-- Row: `apeiron-vv-v1-c-the-entry-BhpTnZsIcBnlxn1BVXRCcI` (/ READLOOP-HCOORD): hcoord is a THEOREM here; Read.lean's :417 cites it.
import Growable.Representation
-- Row: `apeiron-bridge-close-on-CCc44JmAFbofK6aRQJbemn`.
import Growable.ReadState
import Growable.Resources
import Lean

/-!
# Generated spill read over the coordinate-reading extraction

`table.probe.spill_read` (`proof/ApeironWord.lean:20521-20559`, `src/table/probe.rs:291-335`) is a
resident step at `table.Region.home`, followed — when that step certifies no hit — by
`table.probe.spill_read_tail` (`:20507-20517`, `src/table/probe.rs:345-388`), which runs the single
extracted loop `table.probe.spill_read_tail_loop` / `.body` (`:20455-20503`).

The loop is a bounded linear probe over **one `u64` coordinate lane**, not over the payload tag
lane. For `step < bound` it computes `slot = (home + step) &&& mask`, reads `r.coords[slot]`, and

* on `r.coords[slot] = table.COORD_EMPTY`, the sentinel `U64.MAX`, returns `done none`: the cycle is
  terminated by a vacant coordinate;
* on `r.coords[slot] = wanted`, where `wanted = table.Region.coord_of R V address.place`
  (`:20407-20412`, a `U64`), reads `r.slots[slot]` and runs the extracted full-key matcher
  `table.matches`, returning `done (some slot)` on success;
* otherwise advances `step + 1`.

Once `step ≥ bound` the body returns `done none`. There is no carried-hit argument and no second tag
lane: the tag of `table.SpillSlot` is never read by the probe. Every vacancy, clearance and hit
witness below is stated over the coordinate read, and `wanted` is the query's coordinate word.

The generated read writes nothing, so the hit and clearance facts are properties of a run alone.
The absence direction additionally needs the chain placement premise `StoredProbeChainSafe`: the
resident coordinate is live, a live full `(Sigma, key)` match lies on the query's own cycle at a
tail offset, and every earlier offset of that cycle reads a live coordinate word. A vacant
coordinate in front of a live full match makes the generated read return `ok none`, so this premise
is not derivable from the tag/address invariant alone.

**The dropped tag conjunct.** The previous headline `spill_read_some_implies_full_match` also
concluded `entry.tag ≠ 0#u8`. On the tag-lane extraction that conjunct followed from the executed
body, which compared `entry.tag` with `wanted = table.tag place`, and every generated tag carries
`table.TAG_LIVE`. This body never reads the tag lane, so the conjunct is not a consequence of the
run: a `table.Region` whose `coords` word at `slot` is the query's while `slots[slot].tag = 0` makes
`spill_read` return `ok (some slot)` with a zero tag. The conjunct is therefore **dropped** from
`spill_read_some_implies_full_match`. Re-establishing it needs a read-side coordinate/tag coherence
premise, which is a contract question for this module's consumers (the headline's statement below is
premise-free), not a fact derivable from the run.

**The occupied-coordinate conjunct.** `spill_read_some_implies_full_match` *does* conclude that the hit
slot's coordinate lane reads back a word other than `table.COORD_EMPTY`. Both cases of the run supply
it with no premise: the resident case off `generated_spill_read_some_run`'s coordinate readback at the
swept home slot, the tail case off `StoredProbeHit`'s `StoredProbeVisit`; in each the word is
`table.Region.coord_of place`, which `coord_of_result_ne_empty` shows is never the vacant sentinel. So
the conjunct is a theorem here, not a caller-supplied premise.
-/

namespace Apeiron.GrowableReadLoop

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-- The stored-word spill slot the read headlines are stated over. -/
abbrev StoredSpillSlot : Type :=
  table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64

/-- The stored-word spill slot vector the read headlines are stated over. -/
abbrev StoredSpillSlots : Type :=
  alloc.vec.Vec StoredSpillSlot

/-! ## Generic readers of the generated `do` form -/

/-- Generic bind inversion on the generated `do` form: a successful bind ran its head to `ok`. -/
private theorem bind_ret_inv {α β : Type} (x : Result α) (f : α → Result β) (r : β)
    (hrun : (do let y ← x; f y) = ok r) : ∃ y, x = ok y ∧ f y = ok r := by
  cases hx : x with
  | vis eff k =>
      rw [hx] at hrun
      simp only [bind_tc_vis] at hrun
      exact (vis_not_ok hrun).elim
  | div =>
      rw [hx] at hrun
      simp only [bind_tc_div] at hrun
      exact (div_not_ok hrun).elim
  | ret y =>
      rw [hx] at hrun
      simp only [bind_tc_ok] at hrun
      exact ⟨y, rfl, hrun⟩

/-- One step of the generated `loop`, read one `done`/`cont` at a time. -/
private theorem loop_run_cases {α β : Type} (body : α → Result (ControlFlow α β)) (x : α)
    (y : β) (hrun : loop body x = ok y) :
    ∃ rr, body x = ok rr ∧
      match rr with
      | ControlFlow.done z => z = y
      | ControlFlow.cont x' => loop body x' = ok y := by
  unfold loop at hrun
  obtain ⟨rr, hr, hfin⟩ := bind_ret_inv (body x) (fun rr =>
      match rr with
      | ControlFlow.cont x' => loop body x'
      | ControlFlow.done z => ok z) y hrun
  cases rr with
  | done z => exact ⟨ControlFlow.done z, hr, Result.ok_injective hfin⟩
  | cont x' => exact ⟨ControlFlow.cont x', hr, hfin⟩

/-- A successful generated payload lookup reads exactly the underlying list element. -/
theorem vec_index_readback {T : Type} (slots : alloc.vec.Vec T) (i : Usize) (entry : T)
    (hindex : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice T) slots i = ok entry) :
    slots.val[i.val]? = some entry := by
  rw [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize] at hindex
  have hindex' : (match slots.val[i.val]? with
      | none => (fail .arrayOutOfBounds : Result T)
      | some x => ok x) = ok entry := hindex
  cases hget : slots.val[i.val]? with
  | none => simp [hget] at hindex'
  | some payload =>
      simp only [hget] at hindex'
      exact congrArg Option.some (Result.ok_injective hindex')

/-- The generated `ok none` is not the generated `ok (some slot)`. -/
private theorem ok_none_ne_ok_some {α : Type} (x : α)
    (h : (ok (none : Option α)) = ok (some x)) : False :=
  Option.some_ne_none x (Result.ok_injective h).symm

/-- A non-vacant `hit` makes the generated `else` branch return `hit` unchanged, so the tail does
not run. -/
private theorem isNone_some_returns {α : Type} (hit : Option α) (x : α) (h : hit = some x)
    (tail : Result (Option α)) :
    (do let b1 := core.option.Option.is_none hit
        if b1 = true then tail else ok hit) = ok (some x) := by
  simp [core.option.Option.is_none, h]

/-- A vacant `hit` makes the generated `then` branch run the tail, and the tail's result is the
read's result. -/
private theorem isNone_none_returns {α : Type} (hit : Option α) (h : hit = none)
    (tail : Result (Option α)) :
    (do let b1 := core.option.Option.is_none hit
        if b1 = true then tail else ok hit) = tail := by
  simp [core.option.Option.is_none, h]

/-- `lift` is the infallible coercion the generated code uses: a successful `lift x = ok y` is
`x = y`. This is what turns the swept slot's generated `lift (i &&& mask)` into the arithmetic
`slot = i &&& mask`. -/
private theorem lift_eq_ok_iff {α : Type} {x y : α} (h : lift x = ok y) : x = y := by
  unfold lift at h
  exact Result.ok_injective h

/-- The generated `(Vec.len v != 0#usize) = true` guard reads back as `Vec.len v ≠ 0#usize` through
`bne_iff_ne`. -/
private theorem vec_ne_of_guard_true {α : Type} {v : alloc.vec.Vec α}
    (hB : (alloc.vec.Vec.len v != 0#usize) = true) : alloc.vec.Vec.len v ≠ 0#usize :=
  bne_iff_ne.mp hB

/-- The generated `(Vec.len v != 0#usize) = false` guard reads back as `Vec.len v = 0#usize`. -/
private theorem vec_eq_of_guard_false {α : Type} {v : alloc.vec.Vec α}
    (hB : (alloc.vec.Vec.len v != 0#usize) = false) : alloc.vec.Vec.len v = 0#usize := by
  by_contra hcon
  have htrue : (alloc.vec.Vec.len v != 0#usize) = true := bne_iff_ne.mpr hcon
  rw [htrue] at hB
  exact Bool.noConfusion hB

/-- A region's spill vector is nonempty when the generated empty-region guard passed. -/
private theorem region_slots_length_ne_zero_of_guard {R V : Type} (region : table.Region R V)
    (hB : (alloc.vec.Vec.len region.slots != 0#usize) = true) :
    region.slots.length ≠ 0 := by
  intro hz
  have hv : (alloc.vec.Vec.len region.slots).val = 0 := by
    rw [alloc.vec.Vec.len_val, hz]
  have hone : (0#usize : Usize).val = 0 := rfl
  exact vec_ne_of_guard_true hB (UScalar.eq_of_val_eq (by rw [hone, hv])).symm

/-- A region's spill vector is empty when the generated empty-region guard failed. -/
private theorem region_slots_length_eq_zero_of_guard_false {R V : Type} (region : table.Region R V)
    (hB : (alloc.vec.Vec.len region.slots != 0#usize) = false) :
    region.slots.length = 0 := by
  have hv := congrArg UScalar.val (vec_eq_of_guard_false hB)
  simpa [alloc.vec.Vec.len_val] using hv

/-- The generated occupancy re-test with a vacant `hit` runs the tail: the result is the tail's. -/
private theorem isNone_of_none_runs_tail (tail : Result (Option Usize)) :
    (do let hit ← ok (none : Option Usize)
        let b1 := core.option.Option.is_none hit
        if b1 then tail else ok hit) = tail := by
  simp [core.option.Option.is_none]

/-! ## The coordinate lane -/

/-- `COORD_EMPTY - 1` is not `COORD_EMPTY`: the sentinel's value is `2 ^ 64 - 1`. -/
private theorem coord_empty_sub_one_ok_ne (c : U64)
    (h : table.COORD_EMPTY - 1#u64 = ok c) : c ≠ table.COORD_EMPTY := by
  have hmax : table.COORD_EMPTY.val = 2 ^ 64 - 1 := by
    unfold table.COORD_EMPTY
    decide
  have h1 : (1#u64 : U64).val = 1 := by decide
  have hspec : (table.COORD_EMPTY - 1#u64) ⦃ z => z.val = table.COORD_EMPTY.val - 1 ⦄ :=
    WP.spec_mono (UScalar.sub_spec (ty := .U64) (x := table.COORD_EMPTY) (y := 1#u64)
      (by rw [h1, hmax]; omega)) (fun z hz => hz.1)
  rw [h, WP.spec_ok] at hspec
  intro hc
  rw [hc] at hspec
  omega

/-- `table.Region.coord_of` never returns the vacant sentinel: the nudge maps the one colliding
value to `COORD_EMPTY - 1` and the untouched value is unequal by the branch condition. -/
theorem coord_of_result_ne_empty {R V : Type} (place : U64) (c : U64)
    (h : table.Region.coord_of R V place = ok c) : c ≠ table.COORD_EMPTY := by
  unfold table.Region.coord_of at h
  obtain ⟨raw, _hraw, hrest⟩ := bind_ret_inv (place >>> table.SHARD_BITS)
    (fun raw => if raw = table.COORD_EMPTY then table.COORD_EMPTY - 1#u64 else ok raw) c h
  by_cases hraweq : raw = table.COORD_EMPTY
  · rw [if_pos hraweq] at hrest
    exact coord_empty_sub_one_ok_ne c hrest
  · rw [if_neg hraweq] at hrest
    have hcr : c = raw := Result.ok_injective hrest.symm
    rw [hcr]
    exact hraweq

/-! ## The real stored matcher -/

/-- The generated stored matcher is exactly the conjunction of the `Sigma` comparison and the
carrier key comparison, at any tag. -/
theorem stored_matches_slot_iff (slot : StoredSpillSlot) (querySigma : cell.Sigma)
    (query : I64) (address : table.TableAddress) :
    table.matches Apeiron.GrowableIdentity.storedKeyRep slot querySigma query address = ok true ↔
      Apeiron.GrowableIdentity.logicalStored slot.sigma slot.held = (querySigma, query) := by
  cases slot with
  | mk tag slotSigma held value =>
      cases slotSigma <;> cases querySigma <;> cases held <;>
        simp [table.matches, cell.Sigma.Insts.CoreCmpPartialEqSigma.eq,
          cell.Sigma.read_discriminant,
          table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.matches,
          Apeiron.GrowableIdentity.logicalStored, Apeiron.GrowableIdentity.decodeStored]

/-- The generated carried-key matcher terminates for every stored payload and query. -/
theorem stored_matcher_slot_total (slot : StoredSpillSlot) (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress) :
    ∃ matched : Bool,
      table.matches Apeiron.GrowableIdentity.storedKeyRep slot sigma key address = ok matched := by
  rcases slot with ⟨tag, slotSigma, held, value⟩
  cases slotSigma <;> cases sigma <;> cases held <;>
    simp [table.matches, cell.Sigma.Insts.CoreCmpPartialEqSigma.eq,
      cell.Sigma.read_discriminant,
      table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.matches]

/-- A successful generated payload lookup followed by the real stored matcher implies
in-bounds, complete `(Sigma,key)` equality. -/
theorem stored_candidate_full_match (slots : StoredSpillSlots) (index : Usize)
    (entry : StoredSpillSlot) (sigma : cell.Sigma) (key : I64)
    (address : table.TableAddress)
    (hslot : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice StoredSpillSlot) slots index = ok entry)
    (hmatch : table.matches Apeiron.GrowableIdentity.storedKeyRep entry sigma key address =
      ok true) :
    index.val < slots.length ∧
      Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held = (sigma, key) := by
  rw [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize] at hslot
  cases hread : slots.val[index.val]? with
  | none => simp [hread] at hslot
  | some payload =>
      refine ⟨(List.getElem?_eq_some_iff.mp hread).1, ?_⟩
      exact (stored_matches_slot_iff entry sigma key address).mp hmatch

/-! ## The visited-step witnesses, on the coordinate lane -/

/-- One executed tail step at `step`: the swept slot `(home + step) &&& mask` and the coordinate
word the generated body read there. The payload lane is untouched unless the coordinate is the
query's, so no payload clause appears. -/
def StoredProbeVisit (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (home mask : Usize) (j : Nat) (step slot : Usize) (found : U64) : Prop :=
  step.val = j ∧ ∃ i : Usize,
    home + step = ok i ∧ slot = i &&& mask ∧ r.coords.val[slot.val]? = some found

/-- A full-key hit at probe offset `j`: the swept coordinate word is the query's `wanted`, the
payload read at the same slot succeeds, and the real stored matcher accepts the query. -/
def StoredProbeHit (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (home mask : Usize) (wanted : U64) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : I64) (j : Nat) (step slot : Usize) (found : U64) : Prop :=
  StoredProbeVisit r home mask j step slot found ∧ found = wanted ∧
  ∃ (entry : StoredSpillSlot),
    r.slots.val[slot.val]? = some entry ∧
    table.matches Apeiron.GrowableIdentity.storedKeyRep entry sigma key
      { place := place, direct_word := direct } = ok true

/-- A visited step that cannot hide a full match: its coordinate word is the vacant sentinel, its
coordinate word is some other word, or the coordinate is the query's and the real matcher rejects
it. -/
def StoredProbeClear (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (home mask : Usize) (wanted : U64) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : I64) (j : Nat) (step slot : Usize) (found : U64) : Prop :=
  StoredProbeVisit r home mask j step slot found ∧
    (found = table.COORD_EMPTY ∨ found ≠ wanted ∨
      (found = wanted ∧
        ∃ (entry : StoredSpillSlot),
          r.slots.val[slot.val]? = some entry ∧
          table.matches Apeiron.GrowableIdentity.storedKeyRep entry sigma key
            { place := place, direct_word := direct } = ok false))

/-- A visited step whose coordinate word is the vacant sentinel: the generated
cycle-termination witness. -/
def StoredProbeZero (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (home mask : Usize) (j : Nat) (step slot : Usize) (found : U64) : Prop :=
  StoredProbeVisit r home mask j step slot found ∧ found = table.COORD_EMPTY

/-- Every swept offset below the bound reads a live coordinate word and a payload the real matcher
decides. This is the admission premise the tagged sweep statements carry; the run-anchored
certificates do not need it, because each executed step's own body evaluation supplies its
admission. -/
def StoredProbeReady (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64)
    (home mask bound : Usize) : Prop :=
  ∀ step : Usize, step.val < bound.val →
    ∃ (i slot : Usize) (found : U64),
      home + step = ok i ∧ slot = i &&& mask ∧
      r.coords.val[slot.val]? = some found ∧ found ≠ table.COORD_EMPTY ∧
      ∃ (entry : StoredSpillSlot) (matched : Bool),
        r.slots.val[slot.val]? = some entry ∧
        table.matches Apeiron.GrowableIdentity.storedKeyRep entry sigma key
          { place := place, direct_word := direct } = ok matched

/-- The sweep invariant. Either the visited prefix below the cursor is clear and the cursor is
still within the bound, or a hit has been recorded at an offset below the bound. -/
def StoredProbeInv (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64)
    (home mask : Usize) (wanted : U64) (bound : Usize) (step0 step : Usize) : Prop :=
  (step0.val ≤ step.val ∧ step.val ≤ bound.val ∧
    ∀ j, step0.val ≤ j → j < step.val →
      ∃ (st slot : Usize) (found : U64),
        StoredProbeClear r home mask wanted sigma place direct key j st slot found) ∨
  (∃ (j : Nat) (st slot : Usize) (found : U64),
    step0.val ≤ j ∧ j < bound.val ∧
      StoredProbeHit r home mask wanted sigma place direct key j st slot found)

/-- The sweep outcome. Either the loop returned `none` after a clear visited prefix terminated by
the bound or by a vacant coordinate, or it returned a slot carrying a full-key hit. -/
def StoredProbeOutcome (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64)
    (home mask : Usize) (wanted : U64) (bound : Usize) (step0 : Usize) (result : Option Usize) :
    Prop :=
  (result = none ∧ ∃ stop : Nat, step0.val ≤ stop ∧ stop ≤ bound.val ∧
    (∀ j, step0.val ≤ j → j < stop →
      ∃ (st slot : Usize) (found : U64),
        StoredProbeClear r home mask wanted sigma place direct key j st slot found) ∧
    (stop = bound.val ∨
      ∃ (st slot : Usize) (found : U64),
        StoredProbeZero r home mask stop st slot found)) ∨
  (∃ (j : Nat) (st slot : Usize) (found : U64),
    step0.val ≤ j ∧ j < bound.val ∧
      StoredProbeHit r home mask wanted sigma place direct key j st slot found ∧
    result = some slot)

/-- Two visited offsets of one cycle determine the same swept slot and the same coordinate word. -/
theorem stored_probe_visit_unique (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (home mask : Usize) (j : Nat) (step₁ slot₁ : Usize) (found₁ : U64)
    (step₂ slot₂ : Usize) (found₂ : U64)
    (h₁ : StoredProbeVisit r home mask j step₁ slot₁ found₁)
    (h₂ : StoredProbeVisit r home mask j step₂ slot₂ found₂) :
    slot₁ = slot₂ ∧ found₁ = found₂ := by
  rcases h₁ with ⟨hval₁, i₁, hadd₁, hslot₁, hfound₁⟩
  rcases h₂ with ⟨hval₂, i₂, hadd₂, hslot₂, hfound₂⟩
  have hstepEq : step₁ = step₂ := UScalar.eq_of_val_eq (by omega)
  subst hstepEq
  have hiEq : i₁ = i₂ := Result.ok_injective (hadd₁.symm.trans hadd₂)
  subst hiEq
  rw [hslot₁] at hfound₁
  rw [hslot₂] at hfound₂
  exact ⟨by rw [hslot₁, hslot₂], (Option.some.inj (hfound₂.symm.trans hfound₁)).symm⟩

/-! ## The generated tail-loop step, classified -/

/-- `table.probe.spill_read_tail_loop.body` (`proof/ApeironWord.lean:20455-20487`) has exactly four
outcomes: the guard `step < bound` failing returns `done none`; a vacant coordinate at the swept
slot returns `done none`; the coordinate being `wanted` with a successful full-key matcher returns
`done (some slot)`; every other live coordinate advances `step` by one, and that `cont` case
carries why the swept step could not hide a full match. The swept slot is the generated
`lift (i1 &&& mask)`, discharged here through `lift_eq_ok_iff`, so no part of this module reasons
about the mask arithmetic. -/
private theorem generated_probe_body_cases {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (home bound mask step : Usize) (wanted : U64)
    (out : ControlFlow Usize (Option Usize))
    (hbody : table.probe.spill_read_tail_loop.body keyRep r sigma place direct key home bound mask
      wanted step = ok out) :
    (out = ControlFlow.done none ∧ step.val ≥ bound.val) ∨
    (out = ControlFlow.done none ∧ ∃ (slot : Usize) (found : U64),
      step.val < bound.val ∧ r.coords.val[slot.val]? = some found ∧
      found = table.COORD_EMPTY ∧
      ∃ i : Usize, home + step = ok i ∧ slot = i &&& mask) ∨
    (step.val < bound.val ∧ ∃ (slot : Usize) (found : U64) (ss : table.SpillSlot R V),
      out = ControlFlow.done (some slot) ∧ r.coords.val[slot.val]? = some found ∧ found = wanted ∧
      r.slots.val[slot.val]? = some ss ∧
      ∃ i : Usize, home + step = ok i ∧ slot = i &&& mask ∧
        table.matches keyRep ss sigma key { place := place, direct_word := direct } = ok true) ∨
    (step.val < bound.val ∧ ∃ (slot : Usize) (found : U64) (next : Usize),
      out = ControlFlow.cont next ∧ step + 1#usize = ok next ∧
      r.coords.val[slot.val]? = some found ∧
      ∃ i : Usize, home + step = ok i ∧ slot = i &&& mask ∧
        (found = table.COORD_EMPTY ∨ found ≠ wanted ∨
          (found = wanted ∧
            ∃ (ss : table.SpillSlot R V),
              r.slots.val[slot.val]? = some ss ∧
              table.matches keyRep ss sigma key { place := place, direct_word := direct } =
                ok false))) := by
  unfold table.probe.spill_read_tail_loop.body at hbody
  by_cases hlt : step < bound
  · rw [if_pos hlt] at hbody
    have hltv : step.val < bound.val := hlt
    obtain ⟨i1, hi1, hrest⟩ := bind_ret_inv (home + step) (fun i1 => do
        let slot0 ← lift (i1 &&& mask)
        let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U64)
          r.coords slot0
        if found = table.COORD_EMPTY then ok (ControlFlow.done none)
        else if found = wanted then
          do
            let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
              (table.SpillSlot R V)) r.slots slot0
            let m ← table.matches keyRep ss sigma key { place := place, direct_word := direct }
            if m then ok (ControlFlow.done (some slot0))
            else
              do
                let step1 ← step + 1#usize
                ok (ControlFlow.cont step1)
        else
          do
            let step1 ← step + 1#usize
            ok (ControlFlow.cont step1)) out hbody
    obtain ⟨slot0, hlift0, hrest'⟩ := bind_ret_inv (lift (i1 &&& mask)) (fun slot0 => do
        let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U64)
          r.coords slot0
        if found = table.COORD_EMPTY then ok (ControlFlow.done none)
        else if found = wanted then
          do
            let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
              (table.SpillSlot R V)) r.slots slot0
            let m ← table.matches keyRep ss sigma key { place := place, direct_word := direct }
            if m then ok (ControlFlow.done (some slot0))
            else
              do
                let step1 ← step + 1#usize
                ok (ControlFlow.cont step1)
        else
          do
            let step1 ← step + 1#usize
            ok (ControlFlow.cont step1)) out hrest
    obtain ⟨found, hfound, hrest''⟩ := bind_ret_inv
      (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U64) r.coords slot0)
      (fun found =>
        if found = table.COORD_EMPTY then ok (ControlFlow.done none)
        else if found = wanted then
          do
            let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
              (table.SpillSlot R V)) r.slots slot0
            let m ← table.matches keyRep ss sigma key { place := place, direct_word := direct }
            if m then ok (ControlFlow.done (some slot0))
            else
              do
                let step1 ← step + 1#usize
                ok (ControlFlow.cont step1)
        else
          do
            let step1 ← step + 1#usize
            ok (ControlFlow.cont step1)) out hrest'
    have hslot0 : slot0 = i1 &&& mask := (lift_eq_ok_iff hlift0).symm
    have hcoord : r.coords.val[slot0.val]? = some found :=
      vec_index_readback r.coords slot0 found hfound
    by_cases hEmpty : found = table.COORD_EMPTY
    · rw [if_pos hEmpty] at hrest''
      refine Or.inr (Or.inl ?_)
      exact ⟨(Result.ok_injective hrest'').symm, slot0, found, hltv, hcoord, hEmpty,
        i1, hi1, hslot0⟩
    · rw [if_neg hEmpty] at hrest''
      by_cases hWant : found = wanted
      · rw [if_pos hWant] at hrest''
        obtain ⟨ss, hss, hrest3⟩ := bind_ret_inv
          (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
            (table.SpillSlot R V)) r.slots slot0) (fun ss => do
              let m ← table.matches keyRep ss sigma key { place := place, direct_word := direct }
              if m then ok (ControlFlow.done (some slot0))
              else
                do
                  let step1 ← step + 1#usize
                  ok (ControlFlow.cont step1)) out hrest''
        obtain ⟨m, hm, hrest4⟩ := bind_ret_inv
          (table.matches keyRep ss sigma key { place := place, direct_word := direct }) (fun m =>
            if m then ok (ControlFlow.done (some slot0))
            else
              do
                let step1 ← step + 1#usize
                ok (ControlFlow.cont step1)) out hrest3
        have hssList : r.slots.val[slot0.val]? = some ss := vec_index_readback r.slots slot0 ss hss
        by_cases hmT : m = true
        · rw [if_pos hmT] at hrest4
          refine Or.inr (Or.inr (Or.inl ?_))
          exact ⟨hltv, slot0, found, ss, (Result.ok_injective hrest4).symm, hcoord, hWant,
            hssList, i1, hi1, hslot0, hm.trans (congrArg ok hmT)⟩
        · rw [if_neg hmT] at hrest4
          have hmF : m = false := Bool.eq_false_of_not_eq_true hmT
          obtain ⟨step1, hstep1, hout⟩ := bind_ret_inv (step + 1#usize)
            (fun step1 => ok (ControlFlow.cont step1)) out hrest4
          have hwhy : found = table.COORD_EMPTY ∨ found ≠ wanted ∨
              (found = wanted ∧ ∃ (ss' : table.SpillSlot R V),
                r.slots.val[slot0.val]? = some ss' ∧
                table.matches keyRep ss' sigma key { place := place, direct_word := direct } =
                  ok false) := by
            refine Or.inr (Or.inr ?_)
            exact ⟨hWant, ss, hssList, hm.trans (congrArg ok hmF)⟩
          refine Or.inr (Or.inr (Or.inr ?_))
          exact ⟨hltv, slot0, found, step1, (Result.ok_injective hout).symm, hstep1, hcoord,
            i1, hi1, hslot0, hwhy⟩
      · rw [if_neg hWant] at hrest''
        obtain ⟨step1, hstep1, hout⟩ := bind_ret_inv (step + 1#usize)
          (fun step1 => ok (ControlFlow.cont step1)) out hrest''
        have hwhy : found = table.COORD_EMPTY ∨ found ≠ wanted ∨
            (found = wanted ∧ ∃ (ss' : table.SpillSlot R V),
              r.slots.val[slot0.val]? = some ss' ∧
              table.matches keyRep ss' sigma key { place := place, direct_word := direct } =
                ok false) := by
          refine Or.inr (Or.inl ?_)
          exact hWant
        refine Or.inr (Or.inr (Or.inr ?_))
        exact ⟨hltv, slot0, found, step1, (Result.ok_injective hout).symm, hstep1, hcoord,
          i1, hi1, hslot0, hwhy⟩
  · rw [if_neg hlt] at hbody
    refine Or.inl ⟨(Result.ok_injective hbody).symm, ?_⟩
    exact Nat.le_of_not_gt (fun h => hlt ((UScalar.lt_equiv step bound).mpr h))

/-! ## Loop-body facts -/

/-- Exhausting the bound returns `none`; the tail loop synthesizes no hit. -/
theorem generated_probe_stops_at_bound {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (home bound mask : Usize) (wanted : U64) (step : Usize) (hstep : ¬ step < bound) :
    table.probe.spill_read_tail_loop.body keyRep r sigma place direct key home bound mask wanted
      step = ok (ControlFlow.done none) := by
  unfold table.probe.spill_read_tail_loop.body
  rw [if_neg hstep]

/-- The generated hit branch, read off the body equation. This extraction's body has no
carried-hit argument: the hit is recognised from the coordinate word at the swept slot rather than
from a carried `Option`. So the "stops on a hit" fact is that a `done (some slot)` result at
`step` is certified, with `slot` the swept slot at that offset, its coordinate word equal to
`wanted`, and the payload read at it decided by the real matcher. -/
theorem generated_probe_stops_on_hit {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (home bound mask : Usize) (wanted : U64) (step i slot : Usize) (found : U64)
    (ss : table.SpillSlot R V)
    (hbody : table.probe.spill_read_tail_loop.body keyRep r sigma place direct key home bound mask
      wanted step = ok (ControlFlow.done (some slot)))
    (_hadd : home + step = ok i) (_hslot : slot = i &&& mask)
    (hfound : r.coords.val[slot.val]? = some found) (hfoundEq : found = wanted)
    (hss : r.slots.val[slot.val]? = some ss) :
    step.val < bound.val ∧
      ∃ matched : Bool,
        table.matches keyRep ss sigma key { place := place, direct_word := direct } = ok matched ∧
      table.probe.spill_read_tail_loop.body keyRep r sigma place direct key home bound mask wanted
        step = ok (ControlFlow.done (some slot)) := by
  rcases generated_probe_body_cases keyRep r sigma place direct key home bound mask step wanted
    (ControlFlow.done (some slot)) hbody with hstopBig | hstopSmall | hhit | hcont
  · obtain ⟨hout, -⟩ := hstopBig
    cases hout
  · obtain ⟨hout, slot', found', hltv, hcoord', heq, i', hi', hslot'⟩ := hstopSmall
    cases hout
  · obtain ⟨hltv, slot', found', ss', hout, hcoord', hfound', hss', i', hi', hslot', hmatch⟩ := hhit
    have hslotEq : slot' = slot := Option.some.inj (ControlFlow.done.inj hout).symm
    subst hslotEq
    have hfoundEq2 : found' = found := Option.some.inj (hcoord'.symm.trans hfound)
    subst hfoundEq2
    have hssEq : ss' = ss := Option.some.inj (hss'.symm.trans hss)
    subst hssEq
    exact ⟨hltv, ⟨true, hmatch, hbody⟩⟩
  · obtain ⟨-, slot', found', next, hout, -⟩ := hcont
    cases hout

/-- The generated increment succeeds under the loop guard and stays within the bound. -/
theorem generated_probe_increment_checked (step bound : Usize) (hstep : step < bound) :
    ∃ next : Usize, step + 1#usize = ok next ∧
      next.val = step.val + 1 ∧ next.val ≤ bound.val := by
  have hlt := (UScalar.lt_equiv step bound).mp hstep
  have hcap : step.val + 1 ≤ Usize.max := by
    have hbound := bound.hBounds
    simp only [UScalarTy.Usize_numBits_eq] at hbound
    have hmax : Usize.max = 2 ^ System.Platform.numBits - 1 := by
      simp [Usize.max, Usize.numBits]
    omega
  have hone : (1#usize).val = 1 := rfl
  obtain ⟨next, hnext, hnextVal⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := step) (y := 1#usize) (by omega))
  refine ⟨next, hnext, ?_, ?_⟩
  · simpa using hnextVal
  · omega

/-! ## The run-anchored sweep certificate

`generated_probe_body_cases` gives each executed step its own admission and its clear/hit witness.
A `cont` successor raises `step` by one below `bound`, so the remaining budget
`bound.val - step.val` strictly decreases and strong induction on it reaches the terminating step.
No readiness premise for unexecuted offsets is ever needed. -/

/-- Run-anchored certificate for a tail run from any start offset. Each executed step contributes
its own admission and its clear/hit witness, so no readiness premise is required. -/
private theorem generated_probe_run_certificate
    (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64)
    (home mask bound : Usize) (wanted : U64) (result : Option Usize) (start : Usize)
    (hle : start.val ≤ bound.val)
    (hrun : table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep r sigma place
      direct key home bound mask wanted start = ok result) :
    (result = none ∧ ∃ stop : Nat, start.val ≤ stop ∧ stop ≤ bound.val ∧
      (∀ j, start.val ≤ j → j < stop →
        ∃ (st slot : Usize) (found : U64),
          StoredProbeClear r home mask wanted sigma place direct key j st slot found) ∧
      (stop = bound.val ∨
        ∃ (st slot : Usize) (found : U64),
          StoredProbeZero r home mask stop st slot found)) ∨
    (∃ (j : Nat) (st slot : Usize) (found : U64),
      start.val ≤ j ∧ j < bound.val ∧
        StoredProbeHit r home mask wanted sigma place direct key j st slot found ∧
      result = some slot) := by
  have aux : ∀ n : Nat, ∀ (start : Usize), start.val ≤ bound.val →
      bound.val - start.val = n →
      (table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep r sigma place
        direct key home bound mask wanted start = ok result) →
      (result = none ∧ ∃ stop : Nat, start.val ≤ stop ∧ stop ≤ bound.val ∧
        (∀ j, start.val ≤ j → j < stop →
          ∃ (st slot : Usize) (found : U64),
            StoredProbeClear r home mask wanted sigma place direct key j st slot found) ∧
        (stop = bound.val ∨
          ∃ (st slot : Usize) (found : U64),
            StoredProbeZero r home mask stop st slot found)) ∨
      (∃ (j : Nat) (st slot : Usize) (found : U64),
        start.val ≤ j ∧ j < bound.val ∧
          StoredProbeHit r home mask wanted sigma place direct key j st slot found ∧
        result = some slot) := by
    intro n
    refine Nat.strong_induction_on n ?_
    intro n ih start hle hn hrun
    by_cases hltVal : start.val < bound.val
    · have hlt : start < bound := (UScalar.lt_equiv start bound).mpr hltVal
      obtain ⟨rr, hbody, hred⟩ := loop_run_cases
        (fun st1 => table.probe.spill_read_tail_loop.body
          Apeiron.GrowableIdentity.storedKeyRep r sigma place direct key home bound mask wanted
          st1) start result hrun
      rcases generated_probe_body_cases Apeiron.GrowableIdentity.storedKeyRep r sigma place direct
          key home bound mask start wanted rr hbody with
        hstopBig | hstopSmall | hhit | hcont
      · obtain ⟨hout, hge⟩ := hstopBig
        rw [hout] at hred
        refine Or.inl ⟨hred.symm, start.val, le_rfl, by omega, ?_, Or.inl (by omega)⟩
        intro j hj1 hj2
        omega
      · obtain ⟨hout, slot0, found, hltv, hcoord, hempty, i0, hi0, hslot0⟩ := hstopSmall
        rw [hout] at hred
        refine Or.inl ⟨hred.symm, start.val, le_rfl, by omega, ?_, ?_⟩
        · intro j hj1 hj2
          omega
        · have hvisit : StoredProbeVisit r home mask start.val start slot0 found :=
            ⟨rfl, i0, hi0, hslot0, hcoord⟩
          have hzero : StoredProbeZero r home mask start.val start slot0 found :=
            ⟨hvisit, hempty⟩
          exact Or.inr ⟨start, slot0, found, hzero⟩
      · obtain ⟨hltv, slot0, found, ss, hout, hcoord, hwant, hssList, i0, hi0, hslot0,
          hmatch⟩ := hhit
        rw [hout] at hred
        have hvisit : StoredProbeVisit r home mask start.val start slot0 found :=
          ⟨rfl, i0, hi0, hslot0, hcoord⟩
        have hhit : StoredProbeHit r home mask wanted sigma place direct key start.val start
            slot0 found := ⟨hvisit, hwant, ⟨ss, hssList, hmatch⟩⟩
        exact Or.inr ⟨start.val, start, slot0, found, le_rfl, hltv, hhit, hred.symm⟩
      · obtain ⟨hltv, slot0, found, next, hout, hnext, hcoord, i0, hi0, hslot0,
          hwhy⟩ := hcont
        rw [hout] at hred
        have hcap : start.val + 1 ≤ Usize.max := by
          have hb := bound.hBounds
          simp only [UScalarTy.Usize_numBits_eq] at hb
          have hmax : Usize.max = 2 ^ System.Platform.numBits - 1 := by
            simp [Usize.max, Usize.numBits]
          omega
        have hspec := Usize.add_spec (x := start) (y := 1#usize) hcap
        rw [hnext] at hspec
        rw [WP.spec_ok] at hspec
        have hnextVal : next.val = start.val + 1 := hspec
        have hnextLe : next.val ≤ bound.val := by omega
        have hvisit : StoredProbeVisit r home mask start.val start slot0 found :=
          ⟨rfl, i0, hi0, hslot0, hcoord⟩
        have hclearStart : StoredProbeClear r home mask wanted sigma place direct key start.val
            start slot0 found :=
          ⟨hvisit, hwhy⟩
        have hmeasure : bound.val - next.val < n := by omega
        have hcert' := ih (bound.val - next.val) hmeasure next hnextLe rfl hred
        rcases hcert' with
          ⟨hnone, stop, hstopStart, hstopLe, hprefix', hstopcase⟩ |
          ⟨j, st', slot', found', hjStart, hjBound, hhit', hres⟩
        · refine Or.inl ⟨hnone, stop, by omega, hstopLe, ?_, hstopcase⟩
          intro j hjStart hjStop
          by_cases hjNext : j < next.val
          · have hjEq : j = start.val := by omega
            subst hjEq
            exact ⟨start, slot0, found, hclearStart⟩
          · exact hprefix' j (by omega) hjStop
        · exact Or.inr ⟨j, st', slot', found', by omega, hjBound, hhit', hres⟩
    · have hnot : ¬ start < bound := by
        intro hb
        exact hltVal ((UScalar.lt_equiv start bound).mp hb)
      have hboundNone := generated_probe_stops_at_bound
        Apeiron.GrowableIdentity.storedKeyRep r sigma place direct key home bound mask wanted start
        hnot
      obtain ⟨rr, hbody, hred⟩ := loop_run_cases
        (fun st1 => table.probe.spill_read_tail_loop.body
          Apeiron.GrowableIdentity.storedKeyRep r sigma place direct key home bound mask wanted
          st1) start result hrun
      have hrr : rr = ControlFlow.done none :=
        Result.ok_injective (hbody.symm.trans hboundNone)
      rw [hrr] at hred
      refine Or.inl ⟨hred.symm, start.val, le_rfl, by omega, ?_, Or.inl (by omega)⟩
      intro j hj1 hj2
      omega
  exact aux (bound.val - start.val) start hle rfl hrun

/-- Project the hit witness from a generated `ok (some slot)` tail run, with no readiness premise. -/
private theorem generated_probe_run_some
    (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64)
    (home mask bound : Usize) (wanted : U64) (slot : Usize) (hbound1 : 1 ≤ bound.val)
    (hrun : table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep r sigma place
      direct key home bound mask wanted 1#usize = ok (some slot)) :
    ∃ (j : Nat) (st : Usize) (found : U64),
      1 ≤ j ∧ j < bound.val ∧
        StoredProbeHit r home mask wanted sigma place direct key j st slot found := by
  have hone : (1#usize : Usize).val = 1 := rfl
  have hcert := generated_probe_run_certificate r sigma place direct key home mask bound wanted
    (some slot) 1#usize hbound1 hrun
  rcases hcert with ⟨hnone, _⟩ | ⟨j, st, slot0, found, hjstart, hjbound, hhit, hres⟩
  · cases hnone
  · have hslotEq : slot = slot0 := Option.some.inj hres
    subst hslotEq
    have hpos : (0 : Nat) < j := by omega
    exact ⟨j, st, found, Nat.succ_le_of_lt hpos, hjbound, hhit⟩

/-- Project the clear visited prefix from a generated `ok none` tail run, with no readiness
premise. -/
private theorem generated_probe_run_none
    (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64)
    (home mask bound : Usize) (wanted : U64) (hbound1 : 1 ≤ bound.val)
    (hrun : table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep r sigma place
      direct key home bound mask wanted 1#usize = ok none) :
    ∃ stop : Nat, 1 ≤ stop ∧ stop ≤ bound.val ∧
      (∀ j, 1 ≤ j → j < stop →
        ∃ (st slot : Usize) (found : U64),
          StoredProbeClear r home mask wanted sigma place direct key j st slot found) ∧
      (stop = bound.val ∨
        ∃ (st slot : Usize) (found : U64),
          StoredProbeZero r home mask stop st slot found) := by
  have hone : (1#usize : Usize).val = 1 := rfl
  have hcert := generated_probe_run_certificate r sigma place direct key home mask bound wanted
    none 1#usize hbound1 hrun
  rcases hcert with ⟨hnone, stop, hstopStart, hstopLe, hprefix, hstopcase⟩ |
    ⟨j, st, slot0, found, hjstart, hjbound, hhit, hres⟩
  · have hpos : (0 : Nat) < stop := by omega
    exact ⟨stop, Nat.succ_le_of_lt hpos, hstopLe, hprefix, hstopcase⟩
  · cases hres

/-! ## The tagged sweep surface -/

/-- The generated tail loop reaches its outcome from `step0`, given the coordinate-admission
premise `hready`, a clear visited prefix below `step0`, and the budget `step0.val ≤ bound.val`.
The run-anchored certificate discharges the same shape without `hready`; `hready` is retained on
this audited surface and is not consumed by this proof. -/
theorem generated_probe_sweep_outcome
    (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64)
    (home mask bound : Usize) (wanted : U64) (step0 : Usize) (result : Option Usize)
    (_hready : StoredProbeReady r sigma place direct key home mask bound)
    (_hpre : ∀ j, j < step0.val →
      ∃ (st slot : Usize) (found : U64),
        StoredProbeClear r home mask wanted sigma place direct key j st slot found)
    (hle : step0.val ≤ bound.val)
    (hrun : table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep r sigma place
      direct key home bound mask wanted step0 = ok result) :
    StoredProbeOutcome r sigma place direct key home mask wanted bound step0 result :=
  generated_probe_run_certificate r sigma place direct key home mask bound wanted result step0
    hle hrun

/-- Project the full-key witness from a generated `ok (some slot)` sweep. -/
theorem generated_probe_sweep_some
    (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64)
    (home mask bound : Usize) (wanted : U64) (step0 : Usize) (slot : Usize)
    (_hready : StoredProbeReady r sigma place direct key home mask bound)
    (hpre : ∀ j, j < step0.val →
      ∃ (st slot : Usize) (found : U64),
        StoredProbeClear r home mask wanted sigma place direct key j st slot found)
    (hle : step0.val ≤ bound.val)
    (hrun : table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep r sigma place
      direct key home bound mask wanted step0 = ok (some slot)) :
    ∃ (j : Nat) (st : Usize) (found : U64),
      step0.val ≤ j ∧ j < bound.val ∧
        StoredProbeHit r home mask wanted sigma place direct key j st slot found := by
  have hpost := generated_probe_sweep_outcome r sigma place direct key home mask bound wanted
    step0 (some slot) _hready hpre hle hrun
  rcases hpost with ⟨hnone, _⟩ | ⟨j, st, slot0, found, hj, hjbound, hhit, heq⟩
  · cases hnone
  · have hslotEq : slot0 = slot := Option.some.inj heq.symm
    subst hslotEq
    exact ⟨j, st, found, hj, hjbound, hhit⟩

/-- Project the clear visited prefix from a generated `ok none` sweep. -/
theorem generated_probe_sweep_none
    (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64)
    (home mask bound : Usize) (wanted : U64) (step0 : Usize)
    (_hready : StoredProbeReady r sigma place direct key home mask bound)
    (hpre : ∀ j, j < step0.val →
      ∃ (st slot : Usize) (found : U64),
        StoredProbeClear r home mask wanted sigma place direct key j st slot found)
    (hle : step0.val ≤ bound.val)
    (hrun : table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep r sigma place
      direct key home bound mask wanted step0 = ok none) :
    ∃ stop : Nat, step0.val ≤ stop ∧ stop ≤ bound.val ∧
      (∀ j, step0.val ≤ j → j < stop →
        ∃ (st slot : Usize) (found : U64),
          StoredProbeClear r home mask wanted sigma place direct key j st slot found) ∧
      (stop = bound.val ∨
        ∃ (st slot : Usize) (found : U64),
          StoredProbeZero r home mask stop st slot found) := by
  have hpost := generated_probe_sweep_outcome r sigma place direct key home mask bound wanted
    step0 none _hready hpre hle hrun
  rcases hpost with ⟨-, stop, hstopStart, hstopLe, hprefix, hstopcase⟩ |
    ⟨j, st, slot0, found, hjstart, hjbound, hhit, hres⟩
  · exact ⟨stop, hstopStart, hstopLe, hprefix, hstopcase⟩
  · cases hres

/-! ## The generated tail function

`table.probe.spill_read_tail` (`proof/ApeironWord.lean:20507-20517`) reads `Region.bound`,
`bound - 1` and `Region.coord_of address.place` before entering the loop at offset `1`. Those are
the binds the geometry facts come from; the loop's own `wanted` is the same coordinate word. -/

/-- The three binds of the generated `spill_read_tail`, with `Region.bound`'s length equation
discharged through `alloc.vec.Vec.len_val`. -/
private theorem spill_read_tail_binds
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : I64) (home : Usize)
    (res : Option Usize)
    (hresult : table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma
      address key home = ok res) :
    ∃ (bound mask : Usize) (wanted : U64),
      table.Region.bound region = ok bound ∧
      bound.val = region.slots.length ∧
      bound - 1#usize = ok mask ∧
      table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place = ok wanted ∧
      table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep region sigma
        address.place address.direct_word key home bound mask wanted 1#usize = ok res := by
  unfold table.probe.spill_read_tail at hresult
  obtain ⟨bound, hbound, hrest⟩ := bind_ret_inv (table.Region.bound region) (fun bound => do
      let mask ← bound - 1#usize
      let wanted ← table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place
      table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep region sigma
        address.place address.direct_word key home bound mask wanted 1#usize) res hresult
  obtain ⟨mask, hmask, hrest'⟩ := bind_ret_inv (bound - 1#usize) (fun mask => do
      let wanted ← table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place
      table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep region sigma
        address.place address.direct_word key home bound mask wanted 1#usize) res hrest
  obtain ⟨wanted, hwanted, hrest''⟩ := bind_ret_inv
    (table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place)
    (fun wanted => table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep region
      sigma address.place address.direct_word key home bound mask wanted 1#usize) res hrest'
  have hlenEq : alloc.vec.Vec.len region.slots = bound := by
    unfold table.Region.bound at hbound
    exact Result.ok_injective hbound
  have hboundVal : bound.val = region.slots.length := by
    have hv := congrArg UScalar.val hlenEq
    exact hv.symm
  exact ⟨bound, mask, wanted, hbound, hboundVal, hmask, hwanted, hrest''⟩

/-! ## The top-level generated read -/

/-- The generated `spill_read` (`proof/ApeironWord.lean:20521-20559`) reports a hit either from its
resident step at `Region.home` — where the swept slot's coordinate word is the query's
`Region.coord_of` and the payload read at it passes `table.matches` — or from the tail, in which
case the tail's own binds supply `Region.bound`, `bound - 1` and `Region.coord_of`. -/
theorem generated_spill_read_some_run
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : I64) (n : Usize) (slot : Usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok (some slot)) :
    region.slots.length ≠ 0 ∧ table.valid_n n = ok true ∧
    ∃ home : Usize,
      table.Region.home region address.place = ok home ∧
      ((∃ (wanted : U64) (entry : StoredSpillSlot),
          region.coords.val[home.val]? = some wanted ∧ home = slot ∧
          table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place = ok wanted ∧
          region.slots.val[home.val]? = some entry ∧
          table.matches Apeiron.GrowableIdentity.storedKeyRep entry sigma key address = ok true) ∨
       (∃ (bound mask : Usize) (wanted : U64),
          table.Region.bound region = ok bound ∧
          bound.val = region.slots.length ∧
          bound - 1#usize = ok mask ∧
          table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place = ok wanted ∧
          table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma address
            key home = ok (some slot) ∧
          table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep region sigma
            address.place address.direct_word key home bound mask wanted 1#usize = ok (some slot))) := by
  unfold table.probe.spill_read at hresult
  by_cases hB : (alloc.vec.Vec.len region.slots != 0#usize) = true
  · rw [if_pos hB] at hresult
    obtain ⟨b, hvalid, hresult⟩ := bind_ret_inv (table.valid_n n) (fun b =>
      if b then do
        let home ← table.Region.home region address.place
        let wanted ← table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place
        let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U64)
          region.coords home
        if found = table.COORD_EMPTY then ok none
        else
          do
            let hit ←
              if found = wanted then
                do
                  let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                    (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home
                  let b1 ← table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address
                  if b1 then ok (some home) else ok none
              else ok none
            let b1 := core.option.Option.is_none hit
            if b1 then table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma address key home
            else ok hit
      else ok none) (some slot) hresult
    by_cases hbT : b = true
    · rw [if_pos hbT] at hresult
      obtain ⟨home, hhome, hresult⟩ := bind_ret_inv (table.Region.home region address.place)
        (fun home => do
          let wanted ← table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place
          let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U64)
            region.coords home
          if found = table.COORD_EMPTY then ok none
          else
            do
              let hit ←
                if found = wanted then
                  do
                    let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                      (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home
                    let b1 ← table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key
                      address
                    if b1 then ok (some home) else ok none
                  else ok none
              let b1 := core.option.Option.is_none hit
              if b1 then table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma address key home
              else ok hit) (some slot) hresult
      obtain ⟨wanted, hwanted, hresult⟩ := bind_ret_inv
        (table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place) (fun wanted => do
          let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U64)
            region.coords home
          if found = table.COORD_EMPTY then ok none
          else
            do
              let hit ←
                if found = wanted then
                  do
                    let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                      (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home
                    let b1 ← table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key
                      address
                    if b1 then ok (some home) else ok none
                  else ok none
              let b1 := core.option.Option.is_none hit
              if b1 then table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma address key home
              else ok hit) (some slot) hresult
      obtain ⟨found, hfound, hresult⟩ := bind_ret_inv
        (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U64) region.coords home)
        (fun found =>
          if found = table.COORD_EMPTY then ok none
          else
            do
              let hit ←
                if found = wanted then
                  do
                    let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                      (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home
                    let b1 ← table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key
                      address
                    if b1 then ok (some home) else ok none
                  else ok none
              let b1 := core.option.Option.is_none hit
              if b1 then table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma address key home
              else ok hit) (some slot) hresult
      by_cases hEmpty : found = table.COORD_EMPTY
      · rw [if_pos hEmpty] at hresult
        exact False.elim (ok_none_ne_ok_some slot hresult)
      · rw [if_neg hEmpty] at hresult
        by_cases hWant : found = wanted
        · rw [if_pos hWant] at hresult
          obtain ⟨hit, hhitEq, hrest'⟩ := bind_ret_inv
            (do
              let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home
              let b1 ← table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address
              if b1 then ok (some home) else ok none) (fun hit => do
                let b1 := core.option.Option.is_none hit
                if b1 then table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma address key home
                else ok hit) (some slot) hresult
          obtain ⟨ss, hss, hhit⟩ := bind_ret_inv
            (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
              (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home)
            (fun ss => do
              let b1 ← table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address
              if b1 then ok (some home) else ok none) hit hhitEq
          obtain ⟨m, hm, hhit'⟩ := bind_ret_inv
            (table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address)
            (fun m => if m then ok (some home) else ok none) hit hhit
          by_cases hmT : m = true
          · rw [if_pos hmT] at hhit'
            have hhitSome : (some home : Option Usize) = hit := Result.ok_injective hhit'
            rw [isNone_some_returns hit home hhitSome.symm
              (table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma
                address key home)] at hrest'
            have hhomeSlot : home = slot := Option.some.inj (Result.ok_injective hrest')
            have hcoord : region.coords.val[home.val]? = some wanted := by
              have hh := vec_index_readback region.coords home found hfound
              rwa [hWant] at hh
            have hssList : region.slots.val[home.val]? = some ss :=
              vec_index_readback region.slots home ss hss
            have hvalidTrue : table.valid_n n = ok true := hvalid.trans (by rw [hbT])
            refine ⟨?_, hvalidTrue, home, hhome, ?_⟩
            · exact region_slots_length_ne_zero_of_guard region hB
            · exact Or.inl ⟨wanted, ss, hcoord, hhomeSlot, hwanted, hssList,
                hm.trans (congrArg ok hmT)⟩
          · rw [if_neg hmT] at hhit'
            have hitNone : hit = none := Result.ok_injective hhit'.symm
            rw [isNone_none_returns hit hitNone
              (table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma
                address key home)] at hrest'
            obtain ⟨bound, mask, wanted', hbound, hboundVal, hmask, hwanted', hloop⟩ :=
              spill_read_tail_binds region sigma address key home (some slot) hrest'
            have hvalidTrue : table.valid_n n = ok true := hvalid.trans (by rw [hbT])
            refine ⟨?_, hvalidTrue, home, hhome, Or.inr ⟨bound, mask, wanted', hbound,
              hboundVal, hmask, hwanted', hrest', hloop⟩⟩
            · exact region_slots_length_ne_zero_of_guard region hB
        · rw [if_neg hWant] at hresult
          rw [isNone_of_none_runs_tail
            (table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma
              address key home)] at hresult
          obtain ⟨bound, mask, wanted', hbound, hboundVal, hmask, hwanted', hloop⟩ :=
            spill_read_tail_binds region sigma address key home (some slot) hresult
          have hvalidTrue : table.valid_n n = ok true := hvalid.trans (by rw [hbT])
          refine ⟨?_, hvalidTrue, home, hhome, Or.inr ⟨bound, mask, wanted', hbound, hboundVal,
            hmask, hwanted', hresult, hloop⟩⟩
          · exact region_slots_length_ne_zero_of_guard region hB
    · rw [if_neg hbT] at hresult
      exact False.elim (ok_none_ne_ok_some slot hresult)
  · rw [if_neg hB] at hresult
    exact False.elim (ok_none_ne_ok_some slot hresult)

/-- A generated `spill_read` `ok none` result either left the empty-region or `valid_n` guard early,
found a vacant coordinate at the resident slot, or ran the tail loop to `ok none`. -/
theorem generated_spill_read_none_run
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : I64) (n : Usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok none) :
    region.slots.length = 0 ∨ table.valid_n n = ok false ∨
    (∃ (home : Usize),
      table.Region.home region address.place = ok home ∧
        region.coords.val[home.val]? = some table.COORD_EMPTY) ∨
    (∃ (home bound mask : Usize) (wanted : U64),
      table.Region.home region address.place = ok home ∧
        table.Region.bound region = ok bound ∧
        bound.val = region.slots.length ∧
        bound - 1#usize = ok mask ∧
        table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place = ok wanted ∧
        table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma address key
          home = ok none) := by
  unfold table.probe.spill_read at hresult
  by_cases hB : (alloc.vec.Vec.len region.slots != 0#usize) = true
  · rw [if_pos hB] at hresult
    obtain ⟨b, hvalid, hresult⟩ := bind_ret_inv (table.valid_n n) (fun b =>
      if b then do
        let home ← table.Region.home region address.place
        let wanted ← table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place
        let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U64)
          region.coords home
        if found = table.COORD_EMPTY then ok none
        else
          do
            let hit ←
              if found = wanted then
                do
                  let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                    (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home
                  let b1 ← table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address
                  if b1 then ok (some home) else ok none
              else ok none
            let b1 := core.option.Option.is_none hit
            if b1 then table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma address key home
            else ok hit
      else ok none) none hresult
    by_cases hbT : b = true
    · rw [if_pos hbT] at hresult
      obtain ⟨home, hhome, hresult⟩ := bind_ret_inv (table.Region.home region address.place)
        (fun home => do
          let wanted ← table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place
          let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U64)
            region.coords home
          if found = table.COORD_EMPTY then ok none
          else
            do
              let hit ←
                if found = wanted then
                  do
                    let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                      (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home
                    let b1 ← table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key
                      address
                    if b1 then ok (some home) else ok none
                  else ok none
              let b1 := core.option.Option.is_none hit
              if b1 then table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma address key home
              else ok hit) none hresult
      obtain ⟨wanted, _hwanted, hresult⟩ := bind_ret_inv
        (table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place) (fun wanted => do
          let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U64)
            region.coords home
          if found = table.COORD_EMPTY then ok none
          else
            do
              let hit ←
                if found = wanted then
                  do
                    let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                      (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home
                    let b1 ← table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key
                      address
                    if b1 then ok (some home) else ok none
                  else ok none
              let b1 := core.option.Option.is_none hit
              if b1 then table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma address key home
              else ok hit) none hresult
      obtain ⟨found, hfound, hresult⟩ := bind_ret_inv
        (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U64) region.coords home)
        (fun found =>
          if found = table.COORD_EMPTY then ok none
          else
            do
              let hit ←
                if found = wanted then
                  do
                    let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                      (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home
                    let b1 ← table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key
                      address
                    if b1 then ok (some home) else ok none
                  else ok none
              let b1 := core.option.Option.is_none hit
              if b1 then table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma address key home
              else ok hit) none hresult
      by_cases hEmpty : found = table.COORD_EMPTY
      · rw [if_pos hEmpty] at hresult
        have hread : region.coords.val[home.val]? = some table.COORD_EMPTY := by
          have hh := vec_index_readback region.coords home found hfound
          rwa [hEmpty] at hh
        exact Or.inr (Or.inr (Or.inl ⟨home, hhome, hread⟩))
      · rw [if_neg hEmpty] at hresult
        by_cases hWant : found = wanted
        · rw [if_pos hWant] at hresult
          obtain ⟨hit, hhitEq, hrest'⟩ := bind_ret_inv
            (do
              let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home
              let b1 ← table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address
              if b1 then ok (some home) else ok none) (fun hit => do
                let b1 := core.option.Option.is_none hit
                if b1 then table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma address key home
                else ok hit) none hresult
          obtain ⟨ss, _hss, hhit⟩ := bind_ret_inv
            (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
              (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home)
            (fun ss => do
              let b1 ← table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address
              if b1 then ok (some home) else ok none) hit hhitEq
          obtain ⟨m, _hm, hhit'⟩ := bind_ret_inv
            (table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address)
            (fun m => if m then ok (some home) else ok none) hit hhit
          by_cases hmT : m = true
          · rw [if_pos hmT] at hhit'
            have hhitSome : hit = some home := Result.ok_injective hhit'.symm
            rw [isNone_some_returns hit home hhitSome
              (table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma
                address key home)] at hrest'
            exact False.elim (ok_none_ne_ok_some home hrest'.symm)
          · rw [if_neg hmT] at hhit'
            have hitNone : hit = none := Result.ok_injective hhit'.symm
            rw [isNone_none_returns hit hitNone
              (table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma
                address key home)] at hrest'
            obtain ⟨bound, mask, wanted', hbound, hboundVal, hmask, hwanted', hloop⟩ :=
              spill_read_tail_binds region sigma address key home none hrest'
            exact Or.inr (Or.inr (Or.inr ⟨home, bound, mask, wanted', hhome, hbound, hboundVal,
              hmask, hwanted', hrest'⟩))
        · rw [if_neg hWant] at hresult
          rw [isNone_of_none_runs_tail
            (table.probe.spill_read_tail Apeiron.GrowableIdentity.storedKeyRep region sigma
              address key home)] at hresult
          obtain ⟨bound, mask, wanted', hbound, hboundVal, hmask, hwanted', hloop⟩ :=
            spill_read_tail_binds region sigma address key home none hresult
          exact Or.inr (Or.inr (Or.inr ⟨home, bound, mask, wanted', hhome, hbound, hboundVal,
            hmask, hwanted', hresult⟩))
    · have hbF : b = false := Bool.eq_false_of_not_eq_true hbT
      exact Or.inr (Or.inl (hvalid.trans (by rw [hbF])))
  · exact Or.inl (region_slots_length_eq_zero_of_guard_false region
      (Bool.of_not_eq_true hB))

/-! ## The hit headline -/

/-- A generated `spill_read` `ok (some slot)` result names a slot inside the region's spill
vector whose payload holds the canonical `(Sigma, key)`, and the same slot's coordinate lane reads
back a word that is not the vacant sentinel. This is the coordinate-surface form of the
hit headline: the generated body observes `Region.coords`, so the hit is certified by the swept
coordinate word equalling the query's `Region.coord_of address.place` and the real stored matcher
accepting the payload. The payload tag is not observed by the body, so no nonzero-tag conjunct is
stated (see the module header). The occupied-coordinate conjunct is read off the run in both cases;
`coord_of_result_ne_empty` carries its non-emptiness. -/
theorem spill_read_some_implies_full_match
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : I64) (n : Usize) (slot : Usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok (some slot)) :
    slot.val < region.slots.length ∧
    ∃ entry : StoredSpillSlot,
      region.slots.val[slot.val]? = some entry ∧
      Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held = (sigma, key) ∧
      (∃ coord : U64, region.coords.val[slot.val]? = some coord ∧
        coord ≠ table.COORD_EMPTY) := by
  obtain ⟨hne, _hvalid, home, _hhome, hcases⟩ :=
    generated_spill_read_some_run region sigma address key n slot hresult
  rcases hcases with
    ⟨wantedR, entry, hcoordR, hhomeSlot, hwantedR, hssList, hmatch⟩ | htail
  · subst hhomeSlot
    refine ⟨(List.getElem?_eq_some_iff.mp hssList).1, entry, hssList, ?_,
      ⟨wantedR, hcoordR, coord_of_result_ne_empty address.place wantedR hwantedR⟩⟩
    exact (stored_matches_slot_iff entry sigma key address).mp hmatch
  · obtain ⟨bound, mask, wanted, _hbound, hboundVal, _hmask, hwanted, _htailRun, hloop⟩ := htail
    have hbound1 : 1 ≤ bound.val := by
      rw [hboundVal]
      omega
    obtain ⟨_, _, _, _, _, hhit⟩ :=
      generated_probe_run_some region sigma address.place address.direct_word key home mask bound
        wanted slot hbound1 hloop
    obtain ⟨hvisit, hfoundWanted, entry, hssList, hmatch⟩ := hhit
    obtain ⟨_, _, _, _, hcoordF⟩ := hvisit
    refine ⟨(List.getElem?_eq_some_iff.mp hssList).1, entry, hssList, ?_,
      ⟨wanted, hfoundWanted ▸ hcoordF,
        coord_of_result_ne_empty address.place wanted hwanted⟩⟩
    exact (stored_matches_slot_iff entry sigma key address).mp hmatch

/-! ## The absence headline -/

/-- The placement premise that turns a `done none` from the generated loop into an absence
certificate. Two conjuncts, split so that the resident-live fact is not conditioned on the loop
geometry: the resident coordinate `Region.home region place` is occupied, and a live full
`(Sigma, key)` match lies on the query's own cycle at a tail offset `j ≥ 1` whose coordinate is the
query's and every earlier offset of the cycle is live. It is not derivable from the tag/address
invariant alone, because a vacant coordinate in front of a live full match makes the generated read
return `ok none` while that invariant holds. -/
def StoredProbeChainSafe (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (place : U64) (sigma : cell.Sigma) (key : I64) : Prop :=
  (∀ (home : Usize), table.Region.home region place = ok home →
    region.coords.val[home.val]? ≠ some table.COORD_EMPTY) ∧
  (∀ (home bound mask : Usize) (wanted : U64) (index : Usize) (entry : StoredSpillSlot),
    table.Region.home region place = ok home →
    bound.val = region.slots.length →
    bound - 1#usize = ok mask →
    table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 place = ok wanted →
    region.slots.val[index.val]? = some entry →
    Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held = (sigma, key) →
    ∃ (coord : U64) (j : Nat) (st : Usize),
      region.coords.val[index.val]? = some coord ∧ coord = wanted ∧
      1 ≤ j ∧ j < bound.val ∧
      StoredProbeVisit region home mask j st index coord ∧
      (∀ k, k < j → ∃ (st' slot : Usize) (found : U64),
        StoredProbeVisit region home mask k st' slot found ∧
          found ≠ table.COORD_EMPTY))

/-- Deliverable 1: a generated `spill_read` `ok none` result excludes every live full `(Sigma, key)`
match in the region. `hvalid` and `hchain` are required beyond the run equation:
`valid_n = ok false` returns `none` without scanning, and the tag/address invariant does not place
live slots before a vacant coordinate. The clear-prefix witness is read off the run itself by
`generated_probe_run_none`. -/
theorem spill_read_none_implies_no_full_match
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (_hstate : StoredLiveTagAddress state)
    (shard : Usize) (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (_hregion : state.shards.val[shard.val]? = some region)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress)
    (_haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (n : Usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok none)
    (hvalid : table.valid_n n = ok true)
    (hchain : StoredProbeChainSafe region address.place sigma key) :
    ∀ (index : Usize) (entry : StoredSpillSlot),
      region.slots.val[index.val]? = some entry → entry.tag ≠ 0#u8 →
      Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held ≠ (sigma, key) := by
  intro index entry hslot htagNz hfull
  rcases hchain with ⟨hnotVac, hplace⟩
  rcases generated_spill_read_none_run region sigma address key n hresult with
    hlenZero | hvalidFalse | ⟨homeVac, hhomeVac, hvac⟩ | ⟨home, _, _, _, hhome, _, _, _, _,
      htailRun⟩
  · have hlt : index.val < region.slots.length := (List.getElem?_eq_some_iff.mp hslot).1
    omega
  · exact Bool.noConfusion (Result.ok_injective (hvalid.symm.trans hvalidFalse))
  · exact absurd hvac (hnotVac homeVac hhomeVac)
  · obtain ⟨bound, mask, wanted, _hbound, hboundVal, hmask, hwanted, htailLoop⟩ :=
      spill_read_tail_binds region sigma address key home none htailRun
    have hbound1 : 1 ≤ bound.val := by
      have h1le : (1#usize).val ≤ bound.val := by
        have h := @UScalar.sub_equiv .Usize bound 1#usize
        rw [hmask, Result.match.ok] at h
        exact h.1
      simpa using h1le
    obtain ⟨stop, _hstop1, _hstopLe, hprefix, hstopcase⟩ :=
      generated_probe_run_none region sigma address.place address.direct_word key home mask bound
        wanted hbound1 htailLoop
    obtain ⟨coord, j, st, _hcoordIndex, hcoordWanted, hj1, hj2, hvisit, hpreds⟩ :=
      hplace home bound mask wanted index entry hhome hboundVal hmask hwanted hslot hfull
    have hwantedNe : wanted ≠ table.COORD_EMPTY :=
      coord_of_result_ne_empty address.place wanted hwanted
    have hjstop : j < stop := by
      rcases hstopcase with hstopBound | ⟨zst, zslot, zfound, hzvisit, hzempty⟩
      · omega
      · by_contra hnot
        have hle : stop ≤ j := by omega
        by_cases heq : stop = j
        · have hzvisit' : StoredProbeVisit region home mask j zst zslot zfound := by
            simpa [heq] using hzvisit
          have hpair := stored_probe_visit_unique region home mask j st index coord zst zslot
            zfound hvisit hzvisit'
          rw [← hpair.2] at hzempty
          exact hwantedNe (hcoordWanted ▸ hzempty)
        · have hjstop' : stop < j := by omega
          obtain ⟨kst, kslot, kfound, hkvisit, hkne⟩ := hpreds stop hjstop'
          have hpair := stored_probe_visit_unique region home mask stop kst kslot kfound zst zslot
            zfound hkvisit hzvisit
          rw [← hpair.2] at hzempty
          exact hkne hzempty
    obtain ⟨cst, cslot, cfound, hclearVisit, hclearCond⟩ := hprefix j hj1 hjstop
    have hpair := stored_probe_visit_unique region home mask j st index coord cst cslot cfound
      hvisit hclearVisit
    rw [← hpair.1, ← hpair.2] at hclearCond
    rcases hclearCond with hempty | hneWant | ⟨_hw, centry, hcentrySlot, hcentryMatch⟩
    · exact hwantedNe (hcoordWanted ▸ hempty)
    · exact hneWant hcoordWanted
    · have hentryEq : centry = entry := by
        exact (Option.some.inj (hslot.symm.trans hcentrySlot)).symm
      have hmatchTrue : table.matches Apeiron.GrowableIdentity.storedKeyRep entry sigma key
          { place := address.place, direct_word := address.direct_word } = ok true :=
        (stored_matches_slot_iff entry sigma key address).mpr hfull
      rw [hentryEq] at hcentryMatch
      exact Bool.false_ne_true (Result.ok_injective (hmatchTrue.symm.trans hcentryMatch)).symm

end Apeiron.GrowableReadLoop