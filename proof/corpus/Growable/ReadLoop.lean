import Growable.Representation
import Growable.ReadState
import Growable.Resources
import Lean

/-!
# Generated spill-read sweep over the single linear probe

The extracted `table.probe.spill_read` is one bounded linear-probe cycle over the single
`table.SpillSlot` vector: `slot = (home + step) &&& mask`, a zero tag terminates the cycle,
a tag equal to the query's generated tag runs the real stored matcher, and `bound` is the
region capacity. There is no group/summary tree and no second tag lane.

These facts concern the extracted loop body and its successful runs, not `Table.get_at`.
The absence direction of the read is only as strong as the probe's chain premise: a `none`
result excludes a live full match because a live slot is placed before any vacancy on its
own cycle. That placement fact is stated explicitly (`StoredProbeChainSafe`) and is not
derivable from the tag/address invariant alone: a zero tag preceding a live full-match slot
on the cycle makes the generated read return `ok none` while a live representative exists.
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

/-! ## Loop-body facts -/

/-- A carried hit terminates the generated body immediately, whatever the remaining bound. -/
theorem generated_probe_stops_on_hit {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (slots : alloc.vec.Vec (table.SpillSlot R V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (bound mask home : Usize) (wanted : U8) (index step : Usize) :
    table.probe.spill_read_loop.body keyRep slots sigma place direct key bound mask home
      wanted (some index) step = ok (ControlFlow.done (some index)) := by
  unfold table.probe.spill_read_loop.body
  by_cases hlt : step < bound
  · simp only [if_pos hlt, core.option.Option.is_none, Option.isNone_some,
      Bool.false_eq_true, if_false]
  · simp only [if_neg hlt]

/-- Exhausting the bound returns the accumulated hit; no hit is synthesized. -/
theorem generated_probe_stops_at_bound {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (slots : alloc.vec.Vec (table.SpillSlot R V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (bound mask home : Usize) (wanted : U8) (hit : Option Usize) (step : Usize)
    (hstep : ¬ step < bound) :
    table.probe.spill_read_loop.body keyRep slots sigma place direct key bound mask home
      wanted hit step = ok (ControlFlow.done hit) := by
  unfold table.probe.spill_read_loop.body
  simp only [if_neg hstep]

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

/-- One executed probe step: the checked index sum, the cyclic mask and the checked payload
lookup all succeed at probe offset `j`. -/
def StoredProbeVisit (slots : StoredSpillSlots) (home mask : Usize) (j : Nat)
    (slot : Usize) (entry : StoredSpillSlot) : Prop :=
  ∃ (step hstep : Usize),
    step.val = j ∧ home + step = ok hstep ∧ slot = hstep &&& mask ∧
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice StoredSpillSlot)
      slots slot = ok entry

/-- A full-key hit at probe offset `j`: the visited slot's tag is the wanted generated tag and
the real stored matcher accepts the query. -/
def StoredProbeHit (slots : StoredSpillSlots) (home mask : Usize) (wanted : U8)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64) (j : Nat)
    (slot : Usize) : Prop :=
  ∃ entry : StoredSpillSlot,
    StoredProbeVisit slots home mask j slot entry ∧ entry.tag = wanted ∧
    table.matches Apeiron.GrowableIdentity.storedKeyRep entry sigma key
      { place := place, direct_word := direct } = ok true

/-- A visited step that cannot hide a full match: its tag is zero, or the real matcher
rejects the query. -/
def StoredProbeClear (slots : StoredSpillSlots) (home mask : Usize) (wanted : U8)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64) (j : Nat)
    (slot : Usize) (entry : StoredSpillSlot) : Prop :=
  StoredProbeVisit slots home mask j slot entry ∧
    (entry.tag = 0#u8 ∨
      (entry.tag = wanted →
        table.matches Apeiron.GrowableIdentity.storedKeyRep entry sigma key
          { place := place, direct_word := direct } = ok false))

/-- A visited step whose tag is zero: the generated cycle-termination witness. -/
def StoredProbeZero (slots : StoredSpillSlots) (home mask : Usize) (j : Nat)
    (slot : Usize) : Prop :=
  ∃ entry : StoredSpillSlot,
    StoredProbeVisit slots home mask j slot entry ∧ entry.tag = 0#u8

/-- Every reached probe step below the bound has a successful checked index and a total
matcher call. This is the admission premise of the sweep outcome. -/
def StoredProbeReady (slots : StoredSpillSlots) (sigma : cell.Sigma) (place : U64)
    (direct : Option U32) (key : I64) (home mask bound : Usize) : Prop :=
  ∀ step : Usize, step.val < bound.val →
    ∃ (slot hstep : Usize) (entry : StoredSpillSlot),
      home + step = ok hstep ∧ slot = hstep &&& mask ∧
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice StoredSpillSlot)
        slots slot = ok entry ∧
      ∃ matched : Bool,
        table.matches Apeiron.GrowableIdentity.storedKeyRep entry sigma key
          { place := place, direct_word := direct } = ok matched

/-- The probe sweep outcome: either no hit and a clear visited prefix terminated by a zero tag
or the bound, or a full-key hit at a step below the bound. -/
def StoredProbeOutcome (slots : StoredSpillSlots) (sigma : cell.Sigma) (place : U64)
    (direct : Option U32) (key : I64) (home mask : Usize) (wanted : U8) (bound : Usize)
    (result : Option Usize) : Prop :=
  (result = none ∧ ∃ stop : Nat, stop ≤ bound.val ∧
    (∀ j, j < stop → ∃ slot entry,
      StoredProbeClear slots home mask wanted sigma place direct key j slot entry) ∧
    (stop = bound.val ∨ ∃ slot, StoredProbeZero slots home mask stop slot)) ∨
  (∃ (j : Nat) (slot : Usize), j < bound.val ∧
    StoredProbeHit slots home mask wanted sigma place direct key j slot ∧ result = some slot)

/-- The probe sweep invariant. A walk state records the clear visited prefix; a stopped state
records the cycle-termination witness; a hit state records the full-key witness. -/
def StoredProbeInv (slots : StoredSpillSlots) (sigma : cell.Sigma) (place : U64)
    (direct : Option U32) (key : I64) (home mask : Usize) (wanted : U8) (bound : Usize)
    (state : Option Usize × Usize) : Prop :=
  (state.1 = none ∧ state.2.val ≤ bound.val ∧
    ∀ j, j < state.2.val → ∃ slot entry,
      StoredProbeClear slots home mask wanted sigma place direct key j slot entry) ∨
  (state.1 = none ∧ state.2.val = bound.val ∧
    ∃ stop : Nat, stop ≤ bound.val ∧
      (∀ j, j < stop → ∃ slot entry,
        StoredProbeClear slots home mask wanted sigma place direct key j slot entry) ∧
      (stop = bound.val ∨ ∃ slot, StoredProbeZero slots home mask stop slot)) ∨
  (∃ (j : Nat) (slot : Usize), state.1 = some slot ∧ j < state.2.val ∧ j < bound.val ∧
    StoredProbeHit slots home mask wanted sigma place direct key j slot)

/-- A live full-match slot lies on the query's own cycle before any vacancy: every earlier
offset of the cycle is occupied. This placement premise is what turns a zero-tag termination
into an absence certificate; the tag/address invariant alone does not supply it. -/
def StoredProbeChainSafe (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (place : U64) (sigma : cell.Sigma) (key : I64) : Prop :=
  ∀ (bound mask home : Usize) (wanted : U8),
    bound.val = region.slots.length →
    bound - 1#usize = ok mask →
    table.Region.home region place = ok home →
    table.tag place = ok wanted →
    ∀ (index : Usize) (entry : StoredSpillSlot),
      region.slots.val[index.val]? = some entry → entry.tag ≠ 0#u8 →
      Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held = (sigma, key) →
      ∃ j : Nat, j < bound.val ∧
        StoredProbeVisit region.slots home mask j index entry ∧
        (∀ k, k < j → ∃ (slot : Usize) (entry : StoredSpillSlot),
          StoredProbeVisit region.slots home mask k slot entry ∧ entry.tag ≠ 0#u8)

/-- Two visited offsets of one cycle determine the same slot and entry. -/
theorem stored_probe_visit_unique (slots : StoredSpillSlots) (home mask : Usize) (j : Nat)
    (slot₁ slot₂ : Usize) (entry₁ entry₂ : StoredSpillSlot)
    (h₁ : StoredProbeVisit slots home mask j slot₁ entry₁)
    (h₂ : StoredProbeVisit slots home mask j slot₂ entry₂) :
    slot₁ = slot₂ ∧ entry₁ = entry₂ := by
  rcases h₁ with ⟨step₁, hstep₁, hval₁, hadd₁, hslot₁, hindex₁⟩
  rcases h₂ with ⟨step₂, hstep₂, hval₂, hadd₂, hslot₂, hindex₂⟩
  have hstepEq : step₁ = step₂ := UScalar.eq_of_val_eq (by omega)
  subst hstepEq
  have hokEq : hstep₁ = hstep₂ := Result.ok_injective (hadd₁.symm.trans hadd₂)
  subst hokEq
  have hslotEq : slot₁ = slot₂ := by rw [hslot₁, hslot₂]
  subst hslotEq
  refine ⟨rfl, ?_⟩
  rw [hindex₁] at hindex₂
  exact Result.ok_injective hindex₂

/-! ## The sweep outcome -/

/-- One generated body step with no carried hit. The successor is the zero-tag stop, the
matcher hit, or the next offset. -/
private theorem generated_probe_body_none_eval {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (slots : alloc.vec.Vec (table.SpillSlot R V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (bound mask home : Usize) (wanted : U8) (step hstep probe : Usize)
    (entry : table.SpillSlot R V) (matched : Bool) (next : Usize)
    (hlt : step < bound) (hadd : home + step = ok hstep) (hslot : probe = hstep &&& mask)
    (hindex : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V)) slots probe = ok entry)
    (hmatch : table.matches keyRep entry sigma key
      { place := place, direct_word := direct } = ok matched)
    (hnext : step + 1#usize = ok next) :
    table.probe.spill_read_loop.body keyRep slots sigma place direct key bound mask home
        wanted none step =
      ok (if entry.tag = 0#u8 then ControlFlow.cont (none, bound)
        else ControlFlow.cont
          ((if entry.tag = wanted then (if matched = true then some probe else none) else none),
            next)) := by
  unfold table.probe.spill_read_loop.body
  rw [if_pos hlt]
  simp only [core.option.Option.is_none, Option.isNone_none, if_true]
  rw [hadd]
  simp only [bind_tc_ok]
  rw [← hslot]
  simp only [lift, bind_tc_ok]
  rw [hindex]
  simp only [bind_tc_ok]
  rw [hmatch, hnext]
  simp only [bind_tc_ok]
  by_cases hzero : entry.tag = 0#u8
  · simp only [hzero, if_true]
  · simp only [if_neg hzero]
    by_cases hw : entry.tag = wanted
    · simp only [hw, if_true]
      cases matched <;> simp only [Bool.false_eq_true, if_true, if_false, bind_tc_ok]
    · simp only [if_neg hw, bind_tc_ok]

/-- The generated probe loop reaches its outcome from `none`. -/
theorem generated_probe_sweep_outcome
    (slots : StoredSpillSlots) (sigma : cell.Sigma) (place : U64)
    (direct : Option U32) (key : I64) (home mask bound : Usize) (wanted : U8)
    (hready : StoredProbeReady slots sigma place direct key home mask bound) :
    table.probe.spill_read_loop Apeiron.GrowableIdentity.storedKeyRep slots sigma place
      direct key none bound mask home wanted 0#usize ⦃ result =>
      StoredProbeOutcome slots sigma place direct key home mask wanted bound result ⦄ := by
  unfold table.probe.spill_read_loop
  apply loop.spec_decr_nat
    (measure := fun state : Option Usize × Usize => bound.val - state.2.val)
    (inv := StoredProbeInv slots sigma place direct key home mask wanted bound)
    (post := StoredProbeOutcome slots sigma place direct key home mask wanted bound)
  · intro state hstate
    rcases state with ⟨hit, step⟩
    rcases hstate with hwalk | hstopped | hfound
    · rcases hwalk with ⟨hnone, hle, hclear⟩
      subst hnone
      dsimp only at hle hclear
      by_cases hlt : step < bound
      · have hslt : step.val < bound.val := (UScalar.lt_equiv step bound).mp hlt
        obtain ⟨slot, hstep, entry, hadd, hslot, hindex, matched, hmatch⟩ := hready step hslt
        obtain ⟨next, hnext, hnextVal, _⟩ := generated_probe_increment_checked step bound hlt
        dsimp only
        rw [generated_probe_body_none_eval Apeiron.GrowableIdentity.storedKeyRep slots sigma
            place direct key bound mask home wanted step hstep slot entry matched next
            hlt hadd hslot hindex hmatch hnext,
          WP.spec_ok]
        by_cases hzero : entry.tag = 0#u8
        · simp only [hzero, if_true]
          refine ⟨Or.inr (Or.inl ⟨rfl, rfl, step.val, le_of_lt hslt, ?_, ?_⟩), ?_⟩
          · intro j hj
            exact hclear j hj
          · exact Or.inr ⟨slot, entry, ⟨step, hstep, rfl, hadd, hslot, hindex⟩, hzero⟩
          · omega
        · simp only [if_neg hzero]
          by_cases hw : entry.tag = wanted
          · simp only [hw, if_true]
            cases matched with
            | false =>
                simp only [Bool.false_eq_true, if_false]
                have hnextLe : next.val ≤ bound.val := by omega
                refine ⟨Or.inl ⟨rfl, by simpa using hnextLe, ?_⟩, ?_⟩
                · intro j hj
                  have hj' : j < next.val := by simpa using hj
                  by_cases hjl : j < step.val
                  · exact hclear j hjl
                  · have hjEq : j = step.val := by omega
                    subst hjEq
                    exact ⟨slot, entry, ⟨step, hstep, rfl, hadd, hslot, hindex⟩,
                      Or.inr (fun _ => hmatch)⟩
                · omega
            | true =>
                simp only [if_true]
                have hstepNext : step.val < next.val := by omega
                refine ⟨Or.inr (Or.inr ⟨step.val, slot, rfl,
                  by simpa using hstepNext, hslt, entry,
                  ⟨step, hstep, rfl, hadd, hslot, hindex⟩, hw, hmatch⟩), ?_⟩
                · omega
          · simp only [if_neg hw]
            have hnextLe : next.val ≤ bound.val := by omega
            refine ⟨Or.inl ⟨rfl, by simpa using hnextLe, ?_⟩, ?_⟩
            · intro j hj
              have hj' : j < next.val := by simpa using hj
              by_cases hjl : j < step.val
              · exact hclear j hjl
              · have hjEq : j = step.val := by omega
                subst hjEq
                exact ⟨slot, entry, ⟨step, hstep, rfl, hadd, hslot, hindex⟩,
                  Or.inr (fun htag => absurd htag hw)⟩
            · omega
      · have hnotVal : ¬ step.val < bound.val := by
          intro hval
          exact hlt ((UScalar.lt_equiv step bound).mpr hval)
        have hstop : step.val = bound.val := by omega
        dsimp only
        rw [generated_probe_stops_at_bound Apeiron.GrowableIdentity.storedKeyRep slots sigma
          place direct key bound mask home wanted none step hlt, WP.spec_ok]
        refine Or.inl ⟨rfl, bound.val, le_rfl, ?_, Or.inl rfl⟩
        intro j hj
        exact hclear j (by omega)
    · rcases hstopped with ⟨hnone, hstepBound, stop, hstopLe, hprefix, hstopcase⟩
      subst hnone
      dsimp only at hstepBound
      have hnot : ¬ step < bound := by
        intro hlt
        have hval := (UScalar.lt_equiv step bound).mp hlt
        omega
      dsimp only
      rw [generated_probe_stops_at_bound Apeiron.GrowableIdentity.storedKeyRep slots sigma
        place direct key bound mask home wanted none step hnot, WP.spec_ok]
      exact Or.inl ⟨rfl, stop, hstopLe, hprefix, hstopcase⟩
    · rcases hfound with ⟨j, slot, hsome, hjstep, hjbound, hhit⟩
      dsimp only at hsome
      subst hsome
      dsimp only
      rw [generated_probe_stops_on_hit Apeiron.GrowableIdentity.storedKeyRep slots sigma
        place direct key bound mask home wanted slot step, WP.spec_ok]
      exact Or.inr ⟨j, slot, hjbound, hhit, rfl⟩
  · exact Or.inl ⟨rfl, by simp, by intro j hj; simp at hj⟩

/-- Project the full-key witness from a generated `ok (some slot)` sweep. -/
theorem generated_probe_sweep_some
    (slots : StoredSpillSlots) (sigma : cell.Sigma) (place : U64)
    (direct : Option U32) (key : I64) (home mask bound : Usize) (wanted : U8)
    (hready : StoredProbeReady slots sigma place direct key home mask bound)
    (slot : Usize)
    (hrun : table.probe.spill_read_loop Apeiron.GrowableIdentity.storedKeyRep slots sigma place
      direct key none bound mask home wanted 0#usize = ok (some slot)) :
    ∃ j : Nat, j < bound.val ∧
      StoredProbeHit slots home mask wanted sigma place direct key j slot := by
  have hpost := generated_probe_sweep_outcome slots sigma place direct key home mask bound
    wanted hready
  rw [hrun, WP.spec_ok] at hpost
  rcases hpost with ⟨hnone, _⟩ | ⟨j, slot', hj, hhit, heq⟩
  · cases hnone
  · have hslotEq : slot = slot' := Option.some.inj heq
    subst hslotEq
    exact ⟨j, hj, hhit⟩

/-- Project the clear visited prefix from a generated `ok none` sweep. -/
theorem generated_probe_sweep_none
    (slots : StoredSpillSlots) (sigma : cell.Sigma) (place : U64)
    (direct : Option U32) (key : I64) (home mask bound : Usize) (wanted : U8)
    (hready : StoredProbeReady slots sigma place direct key home mask bound)
    (hrun : table.probe.spill_read_loop Apeiron.GrowableIdentity.storedKeyRep slots sigma place
      direct key none bound mask home wanted 0#usize = ok none) :
    ∃ stop : Nat, stop ≤ bound.val ∧
      (∀ j, j < stop → ∃ slot entry,
        StoredProbeClear slots home mask wanted sigma place direct key j slot entry) ∧
      (stop = bound.val ∨ ∃ slot, StoredProbeZero slots home mask stop slot) := by
  have hpost := generated_probe_sweep_outcome slots sigma place direct key home mask bound
    wanted hready
  rw [hrun, WP.spec_ok] at hpost
  rcases hpost with ⟨_, stop, hstopLe, hprefix, hstopcase⟩ | ⟨j, slot, hj, _, hsome⟩
  · exact ⟨stop, hstopLe, hprefix, hstopcase⟩
  · cases hsome

/-- A successful generated payload lookup reads exactly the stored list entry. -/
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

/-! ## The top-level generated read -/

/-- A generated `spill_read` `ok (some slot)` result runs the bounded linear probe: the length
guard and `valid_n` passed, and the loop ran to that hit. -/
theorem generated_spill_read_some_run
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : I64) (n : Usize) (slot : Usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok (some slot)) :
    region.slots.length ≠ 0 ∧ table.valid_n n = ok true ∧
    ∃ (bound mask home : Usize) (wanted : U8),
      table.Region.bound region = ok bound ∧
      bound.val = region.slots.length ∧
      bound - 1#usize = ok mask ∧
      table.Region.home region address.place = ok home ∧
      table.tag address.place = ok wanted ∧
      table.probe.spill_read_loop Apeiron.GrowableIdentity.storedKeyRep region.slots sigma
        address.place address.direct_word key none bound mask home wanted 0#usize =
        ok (some slot) := by
  unfold table.probe.spill_read at hresult
  dsimp only at hresult
  split at hresult
  · rename_i hne
    unfold table.valid_n at hresult
    split at hresult
    · rename_i hnz
      split at hresult
      · rename_i hnle
        cases hpow : core.num.Usize.is_power_of_two n with
        | ret b =>
          rw [hpow, bind_tc_ok] at hresult
          split at hresult
          · rename_i hb
            cases hbound : table.Region.bound region with
            | ret bound =>
              rw [hbound, bind_tc_ok] at hresult
              cases hmask : bound - 1#usize with
              | ret mask =>
                rw [hmask, bind_tc_ok] at hresult
                cases hhome : table.Region.home region address.place with
                | ret home =>
                  rw [hhome, bind_tc_ok] at hresult
                  cases htag : table.tag address.place with
                  | ret wanted =>
                    rw [htag, bind_tc_ok] at hresult
                    cases hloop : table.probe.spill_read_loop
                        Apeiron.GrowableIdentity.storedKeyRep region.slots sigma address.place
                        address.direct_word key none bound mask home wanted 0#usize with
                    | ret hit =>
                      rw [hloop, bind_tc_ok] at hresult
                      have hhit : hit = some slot := by
                        by_cases hb1 : core.option.Option.is_none hit = true <;>
                          simpa [hb1] using hresult
                      subst hhit
                      have hvalid : table.valid_n n = ok true := by
                        unfold table.valid_n
                        rw [if_pos hnz, if_pos hnle, hpow, hb]
                      have hboundEq : alloc.vec.Vec.len region.slots = bound := by
                        simpa [table.Region.bound] using hbound
                      have hboundVal : bound.val = region.slots.length := by
                        rw [← hboundEq, alloc.vec.Vec.len_val]
                      refine ⟨?_, hvalid, bound, mask, home, wanted, rfl, hboundVal, hmask,
                        rfl, rfl, hloop⟩
                      intro hlenZero
                      have hval0 : (alloc.vec.Vec.len region.slots).val = 0 := by
                        simp [hlenZero]
                      have hlenEq : alloc.vec.Vec.len region.slots = 0#usize :=
                        UScalar.eq_of_val_eq (by simpa using hval0)
                      rw [hlenEq] at hne
                      simp at hne
                    | vis eff k =>
                      rw [hloop, bind_tc_vis] at hresult
                      exact (vis_not_ok hresult).elim
                    | div =>
                      rw [hloop, bind_tc_div] at hresult
                      exact (div_not_ok hresult).elim
                  | vis eff k => simp [htag] at hresult
                  | div => simp [htag] at hresult
                | vis eff k => simp [hhome] at hresult
                | div => simp [hhome] at hresult
              | vis eff k => simp [hmask] at hresult
              | div => simp [hmask] at hresult
            | vis eff k => simp [hbound] at hresult
            | div => simp [hbound] at hresult
          · simp at hresult
        | vis eff k => simp [hpow] at hresult
        | div => simp [hpow] at hresult
      · simp at hresult
    · simp at hresult
  · simp at hresult

/-- A generated `spill_read` `ok none` result either left the empty-region or `valid_n` guard
early, or ran the bounded linear probe to `ok none`. -/
theorem generated_spill_read_none_run
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : I64) (n : Usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok none) :
    region.slots.length = 0 ∨ table.valid_n n = ok false ∨
    ∃ (bound mask home : Usize) (wanted : U8),
      table.Region.bound region = ok bound ∧
      bound.val = region.slots.length ∧
      bound - 1#usize = ok mask ∧
      table.Region.home region address.place = ok home ∧
      table.tag address.place = ok wanted ∧
      table.probe.spill_read_loop Apeiron.GrowableIdentity.storedKeyRep region.slots sigma
        address.place address.direct_word key none bound mask home wanted 0#usize = ok none := by
  unfold table.probe.spill_read at hresult
  dsimp only at hresult
  split at hresult
  · rename_i hne
    unfold table.valid_n at hresult
    split at hresult
    · rename_i hnz
      split at hresult
      · rename_i hnle
        cases hpow : core.num.Usize.is_power_of_two n with
        | ret b =>
          rw [hpow, bind_tc_ok] at hresult
          split at hresult
          · rename_i hb
            cases hbound : table.Region.bound region with
            | ret bound =>
              rw [hbound, bind_tc_ok] at hresult
              cases hmask : bound - 1#usize with
              | ret mask =>
                rw [hmask, bind_tc_ok] at hresult
                cases hhome : table.Region.home region address.place with
                | ret home =>
                  rw [hhome, bind_tc_ok] at hresult
                  cases htag : table.tag address.place with
                  | ret wanted =>
                    rw [htag, bind_tc_ok] at hresult
                    cases hloop : table.probe.spill_read_loop
                        Apeiron.GrowableIdentity.storedKeyRep region.slots sigma address.place
                        address.direct_word key none bound mask home wanted 0#usize with
                    | ret hit =>
                      rw [hloop, bind_tc_ok] at hresult
                      have hhit : hit = none := by
                        by_cases hb1 : core.option.Option.is_none hit = true <;>
                          simpa [hb1] using hresult
                      subst hhit
                      have hboundEq : alloc.vec.Vec.len region.slots = bound := by
                        simpa [table.Region.bound] using hbound
                      have hboundVal : bound.val = region.slots.length := by
                        rw [← hboundEq, alloc.vec.Vec.len_val]
                      exact Or.inr (Or.inr ⟨bound, mask, home, wanted, rfl, hboundVal, hmask,
                        rfl, rfl, hloop⟩)
                    | vis eff k =>
                      rw [hloop, bind_tc_vis] at hresult
                      exact (vis_not_ok hresult).elim
                    | div =>
                      rw [hloop, bind_tc_div] at hresult
                      exact (div_not_ok hresult).elim
                  | vis eff k => simp [htag] at hresult
                  | div => simp [htag] at hresult
                | vis eff k => simp [hhome] at hresult
                | div => simp [hhome] at hresult
              | vis eff k => simp [hmask] at hresult
              | div => simp [hmask] at hresult
            | vis eff k => simp [hbound] at hresult
            | div => simp [hbound] at hresult
          · rename_i hbf
            refine Or.inr (Or.inl ?_)
            unfold table.valid_n
            rw [if_pos hnz, if_pos hnle, hpow]
            have hbEq : b = false := by
              cases b <;> simp_all
            rw [hbEq]
        | vis eff k => simp [hpow] at hresult
        | div => simp [hpow] at hresult
      · rename_i hnleFalse
        refine Or.inr (Or.inl ?_)
        unfold table.valid_n
        rw [if_pos hnz, if_neg hnleFalse]
    · rename_i hnzFalse
      refine Or.inr (Or.inl ?_)
      unfold table.valid_n
      rw [if_neg hnzFalse]
  · rename_i hz
    have hlenZero : region.slots.length = 0 := by
      by_contra hlen
      apply hz
      have hne' : alloc.vec.Vec.len region.slots ≠ 0#usize := by
        intro heq
        apply hlen
        have := congrArg UScalar.val heq
        simpa [alloc.vec.Vec.len_val] using this
      simpa using hne'
    exact Or.inl hlenZero

/-- Deliverable 2: a generated `spill_read` hit returns an in-bounds slot entry holding the full
query `(Sigma, key)` and a nonzero generated tag. -/
theorem spill_read_some_implies_full_match
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : I64) (n : Usize) (slot : Usize)
    (hprobe : ∀ (bound mask home : Usize) (wanted : U8),
      table.Region.bound region = ok bound →
      bound - 1#usize = ok mask →
      table.Region.home region address.place = ok home →
      table.tag address.place = ok wanted →
      StoredProbeReady region.slots sigma address.place address.direct_word key home mask bound)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok (some slot)) :
    slot.val < region.slots.length ∧
    ∃ entry : StoredSpillSlot,
      region.slots.val[slot.val]? = some entry ∧
      entry.tag ≠ 0#u8 ∧
      Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held = (sigma, key) := by
  obtain ⟨-, -, bound, mask, home, wanted, hbound, -, hmask, hhome, htag, hrun⟩ :=
    generated_spill_read_some_run region sigma address key n slot hresult
  obtain ⟨j, -, entry, hvisit, htagEq, hmatch⟩ :=
    generated_probe_sweep_some region.slots sigma address.place address.direct_word key home mask
      bound wanted (hprobe bound mask home wanted hbound hmask hhome htag) slot hrun
  rcases hvisit with ⟨step, hstep, -, -, -, hindex⟩
  obtain ⟨hinBounds, hlogical⟩ := stored_candidate_full_match region.slots slot entry sigma key
    { place := address.place, direct_word := address.direct_word } hindex hmatch
  refine ⟨hinBounds, entry, vec_index_readback region.slots slot entry hindex, ?_, hlogical⟩
  rw [htagEq]
  exact Apeiron.GrowableScan.tag_ne_zero address.place wanted htag

/-- Deliverable 1: on an admitted region under the live tag/address invariant and the probe
chain placement premise, a generated `spill_read` `ok none` result excludes every live full
`(Sigma, key)` match in the region. The probe chain premise is not derivable from the
tag/address invariant: a zero tag before a live full-match slot makes the read return `none`
while a live representative exists. -/
theorem spill_read_none_implies_no_full_match
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveTagAddress state)
    (shard : Usize) (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (n : Usize)
    (hvalid : table.valid_n n = ok true)
    (hprobe : ∀ (bound mask home : Usize) (wanted : U8),
      table.Region.bound region = ok bound →
      bound - 1#usize = ok mask →
      table.Region.home region address.place = ok home →
      table.tag address.place = ok wanted →
      StoredProbeReady region.slots sigma address.place address.direct_word key home mask bound)
    (hchain : StoredProbeChainSafe region address.place sigma key)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok none) :
    ∀ (index : Usize) (entry : StoredSpillSlot),
      region.slots.val[index.val]? = some entry → entry.tag ≠ 0#u8 →
      Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held ≠ (sigma, key) := by
  intro index entry hslot htagNz hfull
  rcases generated_spill_read_none_run region sigma address key n hresult with
    hlenZero | hvalidFalse | ⟨bound, mask, home, wanted, hbound, hboundVal, hmask, hhome, htag,
      hrun⟩
  · have hbound' : index.val < region.slots.length := (List.getElem?_eq_some_iff.mp hslot).1
    omega
  · rw [hvalid] at hvalidFalse
    exact Bool.false_ne_true (Result.ok_injective hvalidFalse).symm
  · have hready := hprobe bound mask home wanted hbound hmask hhome htag
    obtain ⟨stop, hstopLe, hprefix, hstopcase⟩ :=
      generated_probe_sweep_none region.slots sigma address.place address.direct_word key home
        mask bound wanted hready hrun
    have hpair : (entry.sigma, entry.held.key) = (sigma, key) := hfull
    have hsigma : entry.sigma = sigma := congrArg Prod.fst hpair
    have hkey : entry.held.key = key := congrArg Prod.snd hpair
    obtain ⟨addr, haddrEntry, -, htagEntry⟩ :=
      hstate shard index region entry hregion hslot htagNz
    rw [hsigma, hkey] at haddrEntry
    have haddrEq : addr = address := Result.ok_injective (haddrEntry.symm.trans haddress)
    have hentryTag : entry.tag = wanted := by
      rw [haddrEq] at htagEntry
      exact Result.ok_injective (htagEntry.symm.trans htag)
    have hmatchTrue : table.matches Apeiron.GrowableIdentity.storedKeyRep entry sigma key
        { place := address.place, direct_word := address.direct_word } = ok true :=
      (stored_matches_slot_iff entry sigma key address).mpr hfull
    obtain ⟨j, hjlt, hvisit, hpreds⟩ :=
      hchain bound mask home wanted hboundVal hmask hhome htag index entry hslot htagNz hfull
    have hjstop : j < stop := by
      rcases hstopcase with hstopBound | ⟨zslot, zentry, zvisit, ztag0⟩
      · omega
      · by_contra hnot
        have hle : stop ≤ j := by omega
        by_cases heq : stop = j
        · have hvisit' : StoredProbeVisit region.slots home mask stop index entry :=
            heq.symm ▸ hvisit
          have hentryEq := (stored_probe_visit_unique region.slots home mask stop index zslot
            entry zentry hvisit' zvisit).2
          rw [← hentryEq] at ztag0
          exact htagNz ztag0
        · have hjstop' : stop < j := by omega
          obtain ⟨sk, ek, hvisitk, hek⟩ := hpreds stop hjstop'
          have hentryEq := (stored_probe_visit_unique region.slots home mask stop sk zslot ek
            zentry hvisitk zvisit).2
          rw [hentryEq] at hek
          exact hek ztag0
    obtain ⟨slot, cEntry, hclearVisit, hclearCond⟩ := hprefix j hjstop
    have hentryEq := (stored_probe_visit_unique region.slots home mask j slot index cEntry entry
      hclearVisit hvisit).2
    rw [hentryEq] at hclearCond
    rcases hclearCond with hzero | hfalse
    · exact htagNz hzero
    · have hcontra : (true : Bool) = false :=
        Result.ok_injective (hmatchTrue.symm.trans (hfalse hentryTag))
      exact Bool.false_ne_true hcontra.symm

end Apeiron.GrowableReadLoop
