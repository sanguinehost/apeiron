import Growable.Refinement
import Growable.InsertRefine
import Growable.Representation

/-!
# Gapless-linear-insert read lemma for the ported `table.probe::spill_read`

The flat spill port replaced `Region.tags` and the occupancy summary by one contiguous `u64`
coordinate word per slot with the single vacant sentinel `table.COORD_EMPTY`, and split the read
into `table.probe.spill_read` (the straight-line resident step at `Region.home`) and
`table.probe.spill_read_tail` (the cold collision tail).  `proof/ApeironWord.lean` extracts both
bodies verbatim from `src/table/probe.rs`:

| generated body | `proof/ApeironWord.lean` | `src/table/probe.rs` |
|---|---|---|
| `table.probe.spill_read_tail_loop.body` | 20486-20518 | 314:4-342:1 |
| `table.probe.spill_read_tail_loop` | 20523-20534 | 314:4-342:1 |
| `table.probe.spill_read_tail` | 20538-20548 | 303:0-342:1 |
| `table.probe.spill_read` | 20552-20591 | 252:0-293:1 |
| `table.get_prepared_shard` | 20594-20613 | 94:0-105:1 |

This module proves the `some` half of the gapless-linear-insert read obligation of
`~/ht-port/receipt.md` §4 obligation 1, on those bodies and for every face `sigma`
and every key: a returned `Some i` is certified to satisfy *both* conjuncts — `coords[i]` is the
query's coordinate word `Region.coord_of address.place`, and slot `i` holds the merged value under
the extracted full-key matcher `table.matches`.  No insert-path invariant is assumed; every
statement is a consequence of the executed body alone.

The swept-slot cycle is enumerated by `tail_slot_cycle_enumerates`: for a power-of-two
`Region.bound r` and `mask = bound - 1`, the map `step ↦ (home + step) &&& mask` is a bijection
of `[0, bound)`, so the tail step range `[1, bound)` visits every index except the resident `home`.

`publish_live_coordinate_is_publication_point` closes the ordering half of *gapless publication*:
the write side (`table.Region.publish_live`, `proof/ApeironWord.lean:20687-20719`) stores the
payload and the occupancy `tag` before it stores `coords[index] = Region.coord_of place`.  The
coordinate lane is the word the read compares against, so the coordinate store is the publication
point; the returned coordinate lane is the pre-publication lane updated at `index` with a live
word.  Turning an executed `none` into "no live slot holds the query" still needs the whole-table
first-fit discipline — that a live slot never sits past a vacancy in probe order — which this
module does not prove.  That is the remaining residual named in the receipt.
-/

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron
namespace GrowableRepresentation

/-- **Destination readback of a generated publication.**  The slot read back at the destination
index carries the published triple's `sigma`, `held` and `value`.  The generated
`Apeiron.GrowableRepresentation.publish_live_slots_exact` already gives the whole destination slot;
this exposes the index readback and the payload equations the coordinate lane consumes.  The
in-bounds obligation is not a premise: a successful publication implies the destination `index_mut`
succeeded, hence the bound. -/
theorem publish_live_slot_readback {R V : Type} (before after : table.Region R V) (index : Usize)
    (sigma : cell.Sigma) (held : R) (value : V) (place : U64)
    (hpub : table.Region.publish_live before index (sigma, held, value) place = ok after) :
    ∃ slot, after.slots.val[index.val]? = some slot ∧
      slot.sigma = sigma ∧ slot.held = held ∧ slot.value = value := by
  obtain ⟨tag, _htag, _htagNe, hslots⟩ :=
    Apeiron.GrowableRepresentation.publish_live_slots_exact before after index
      (sigma, held, value) place hpub
  have hbound : index.val < before.slots.val.length := by
    unfold table.Region.publish_live at hpub
    rw [alloc.vec.Vec.index_mut_slice_index] at hpub
    cases h1 : alloc.vec.Vec.index_mut_usize before.slots index with
    | vis eff k => simp [h1] at hpub
    | div => simp [h1] at hpub
    | ret r1 =>
        rcases r1 with ⟨_ss, _back1⟩
        simp only [h1, bind_tc_ok] at hpub
        cases hget : before.slots.val[index.val]? with
        | none => simp [alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize, hget] at h1
        | some _old => exact (List.getElem?_eq_some_iff.mp hget).1
  refine ⟨⟨tag, sigma, held, value⟩, ?_, rfl, rfl, rfl⟩
  rw [hslots]
  simpa using List.getElem?_set_self hbound

end GrowableRepresentation

namespace GrowableScan

/-- **The bounded cycle is surjective on the slots.**  For a power-of-two capacity the bounded
cycle `step ↦ (home + step) &&& (bound - 1)` visits every slot below the capacity.  This is the
arithmetic the `none` half of `GaplessRead` enumerates the probe over; it carries no generated-body
premise beyond the power-of-two bound. -/
theorem spill_cycle_covers {bound home j k : Nat} (hp : bound = 2 ^ k) (hj : j < bound) :
    ∃ step, step < bound ∧ (home + step) &&& (bound - 1) = j := by
  subst hp
  have hk : 0 < 2 ^ k := Nat.pow_pos (by decide)
  set step : Nat := (j + 2 ^ k - home % 2 ^ k) % 2 ^ k with hstepDef
  have hstepLt : step < 2 ^ k := by rw [hstepDef]; exact Nat.mod_lt _ hk
  have happ : (home + step) % 2 ^ k = (home % 2 ^ k + step) % 2 ^ k := by
    have hmod : step % 2 ^ k = step := Nat.mod_eq_of_lt hstepLt
    rw [Nat.add_mod, hmod]
  refine ⟨step, hstepLt, ?_⟩
  rw [Nat.and_two_pow_sub_one_eq_mod, happ]
  by_cases hle : home % 2 ^ k ≤ j
  · have hstepEq : step = j - home % 2 ^ k := by
      have hdiff : j - home % 2 ^ k < 2 ^ k :=
        lt_of_le_of_lt (Nat.sub_le j (home % 2 ^ k)) hj
      have hform : j + 2 ^ k - home % 2 ^ k = (j - home % 2 ^ k) + 2 ^ k := by omega
      rw [hstepDef, hform, Nat.add_mod_right, Nat.mod_eq_of_lt hdiff]
    rw [hstepEq]
    have hsum : home % 2 ^ k + (j - home % 2 ^ k) = j := Nat.add_sub_of_le hle
    rw [hsum]
    exact Nat.mod_eq_of_lt hj
  · have hlt : j < home % 2 ^ k := Nat.lt_of_not_ge hle
    have hstepEq : step = j + 2 ^ k - home % 2 ^ k := by
      rw [hstepDef]
      rw [Nat.mod_eq_of_lt (by omega)]
    rw [hstepEq]
    have hsum : home % 2 ^ k + (j + 2 ^ k - home % 2 ^ k) = j + 2 ^ k := by
      have h1 : 2 ^ k - home % 2 ^ k + home % 2 ^ k = 2 ^ k := by omega
      omega
    rw [hsum]
    have hmod : (j + 2 ^ k) % 2 ^ k = j := by
      rw [Nat.add_mod, Nat.mod_eq_of_lt hj, Nat.mod_self, Nat.add_zero,
        Nat.mod_eq_of_lt hj]
    exact hmod

end GrowableScan
end Apeiron

namespace Apeiron.GrowableGaplessRead

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-! ## Generic readers of generated `do` code -/

/-- Generic bind inversion on the generated `do` form: a successful bind ran its head to `ok`. -/
private theorem bind_ret_inv {α β : Type} (x : Result α) (f : α → Result β) (r : β)
    (hrun : (do let y ← x; f y) = ok r) : ∃ y, x = ok y ∧ f y = ok r := by
  cases hx : x with
  | vis eff k =>
      rw [hx, bind_tc_vis] at hrun
      exact (vis_not_ok hrun).elim
  | div =>
      rw [hx, bind_tc_div] at hrun
      exact (div_not_ok hrun).elim
  | ret y =>
      rw [hx] at hrun
      rw [bind_tc_ok] at hrun
      exact ⟨y, rfl, hrun⟩

/-- The generated `ok none` is not the generated `ok (some slot)`. -/
private theorem ok_none_ne_ok_some {α : Type} (x : α)
    (h : (ok (none : Option α)) = ok (some x)) : False :=
  Option.some_ne_none x (Result.ok_injective h).symm

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
private theorem vec_index_readback {T : Type} (slots : alloc.vec.Vec T) (i : Usize) (entry : T)
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

/-- The generated occupancy predicate `core.option.Option.is_none` at its two values. -/
private theorem isNone_none {α : Type} : core.option.Option.is_none (none : Option α) = true := by
  simp [core.option.Option.is_none]

private theorem isNone_some {α : Type} (x : α) :
    ¬ (core.option.Option.is_none (some x) = true) := by
  simp [core.option.Option.is_none]

/-- The generated occupancy re-test of `spill_read`, decided by its own `hit` value: a
non-vacant `hit` makes the generated `else` branch return `hit` unchanged, so the tail does not
run. -/
private theorem isNone_some_returns {α : Type} (hit : Option α) (x : α) (h : hit = some x)
    (tail : Result (Option α)) :
    (do let b1 := core.option.Option.is_none hit
        if b1 = true then tail else ok hit) = ok (some x) := by
  simp [core.option.Option.is_none, h]

/-- The same re-test with a vacant `hit`: the generated `then` branch runs the tail and its result
is the read's result. -/
private theorem isNone_none_returns {α : Type} (hit : Option α) (h : hit = none)
    (tail : Result (Option α)) :
    (do let b1 := core.option.Option.is_none hit
        if b1 = true then tail else ok hit) = tail := by
  simp [core.option.Option.is_none, h]

/-! ## The generated tail-loop step

`table.probe.spill_read_tail_loop.body` (`proof/ApeironWord.lean:20486-20518`) has exactly three
outcomes: the step guard `step < bound` failing returns `done none`; a vacant coordinate at the
swept slot returns `done none`; a coordinate equal to `wanted` whose full-key matcher succeeds
returns `done (some slot)`; every other live coordinate advances `step` by one.  The swept slot is
the generated `lift (i1 &&& mask)`, carried here as the `ok` value of that lift, so no part of this
module reasons about the mask arithmetic. -/
private theorem tail_body_cases {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (home bound mask step : Usize) (wanted : U64)
    (out : ControlFlow Usize (Option Usize))
    (hbody : table.probe.spill_read_tail_loop.body keyrep r sigma place direct key home bound mask
      wanted step = ok out) :
    (out = ControlFlow.done none ∧ step.val ≥ bound.val) ∨
    (out = ControlFlow.done none ∧ step.val < bound.val) ∨
    (step.val < bound.val ∧ ∃ slot, out = ControlFlow.done (some slot) ∧
      r.coords.val[slot.val]? = some wanted ∧ ∃ ss, r.slots.val[slot.val]? = some ss ∧
        table.matches keyrep ss sigma key { place := place, direct_word := direct } = ok true) ∨
    (step.val < bound.val ∧ ∃ st1, out = ControlFlow.cont st1 ∧ step + 1#usize = ok st1) := by
  unfold table.probe.spill_read_tail_loop.body at hbody
  by_cases hlt : step < bound
  · rw [if_pos hlt] at hbody
    have hltv : step.val < bound.val := hlt
    obtain ⟨i1, _, hrest⟩ := bind_ret_inv (home + step) (fun i1 => do
        let slot0 ← lift (i1 &&& mask)
        let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
          r.coords slot0
        if found = table.COORD_EMPTY then ok (ControlFlow.done none)
        else if found = wanted then
          do
            let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
              (table.SpillSlot R V)) r.slots slot0
            let m ← table.matches keyrep ss sigma key { place := place, direct_word := direct }
            if m then ok (ControlFlow.done (some slot0))
            else
              do
                let step1 ← step + 1#usize
                ok (ControlFlow.cont step1)
        else
          do
            let step1 ← step + 1#usize
            ok (ControlFlow.cont step1)) out hbody
    obtain ⟨slot0, _, hrest'⟩ := bind_ret_inv (lift (i1 &&& mask)) (fun slot0 => do
        let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
          r.coords slot0
        if found = table.COORD_EMPTY then ok (ControlFlow.done none)
        else if found = wanted then
          do
            let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
              (table.SpillSlot R V)) r.slots slot0
            let m ← table.matches keyrep ss sigma key { place := place, direct_word := direct }
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
      (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64) r.coords slot0)
      (fun found =>
        if found = table.COORD_EMPTY then ok (ControlFlow.done none)
        else if found = wanted then
          do
            let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
              (table.SpillSlot R V)) r.slots slot0
            let m ← table.matches keyrep ss sigma key { place := place, direct_word := direct }
            if m then ok (ControlFlow.done (some slot0))
            else
              do
                let step1 ← step + 1#usize
                ok (ControlFlow.cont step1)
        else
          do
            let step1 ← step + 1#usize
            ok (ControlFlow.cont step1)) out hrest'
    by_cases hEmpty : found = table.COORD_EMPTY
    · rw [if_pos hEmpty] at hrest''
      exact Or.inr (Or.inl ⟨(Result.ok_injective hrest'').symm, hltv⟩)
    · rw [if_neg hEmpty] at hrest''
      by_cases hWant : found = wanted
      · rw [if_pos hWant] at hrest''
        obtain ⟨ss, hss, hrest3⟩ := bind_ret_inv
          (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
            (table.SpillSlot R V)) r.slots slot0) (fun ss => do
              let m ← table.matches keyrep ss sigma key { place := place, direct_word := direct }
              if m then ok (ControlFlow.done (some slot0))
              else
                do
                  let step1 ← step + 1#usize
                  ok (ControlFlow.cont step1)) out hrest''
        obtain ⟨m, hm, hrest4⟩ := bind_ret_inv
          (table.matches keyrep ss sigma key { place := place, direct_word := direct }) (fun m =>
            if m then ok (ControlFlow.done (some slot0))
            else
              do
                let step1 ← step + 1#usize
                ok (ControlFlow.cont step1)) out hrest3
        by_cases hmT : m = true
        · rw [if_pos hmT] at hrest4
          have hcoord := vec_index_readback r.coords slot0 found hfound
          rw [hWant] at hcoord
          exact Or.inr (Or.inr (Or.inl ⟨hltv, slot0,
            (Result.ok_injective hrest4).symm, hcoord, ss,
            vec_index_readback r.slots slot0 ss hss, hm.trans (congrArg ok hmT)⟩))
        · rw [if_neg hmT] at hrest4
          obtain ⟨step1, hstep1, hout⟩ := bind_ret_inv (step + 1#usize)
            (fun step1 => ok (ControlFlow.cont step1)) out hrest4
          exact Or.inr (Or.inr (Or.inr ⟨hltv, step1,
            (Result.ok_injective hout).symm, hstep1⟩))
      · rw [if_neg hWant] at hrest''
        obtain ⟨step1, hstep1, hout⟩ := bind_ret_inv (step + 1#usize)
          (fun step1 => ok (ControlFlow.cont step1)) out hrest''
        exact Or.inr (Or.inr (Or.inr ⟨hltv, step1,
          (Result.ok_injective hout).symm, hstep1⟩))
  · rw [if_neg hlt] at hbody
    refine Or.inl ⟨(Result.ok_injective hbody).symm, ?_⟩
    exact Nat.le_of_not_gt (fun h => hlt ((UScalar.lt_equiv step bound).mpr h))

/-! ## The generated tail loop

`tail_step_cases` reads one loop step; the `cont` successor raises `step` by one below `bound`, so
the remaining budget `bound.val - step.val` strictly decreases and strong induction on it reaches
the terminating step.  `hbound` is the arithmetic side condition `bound.val ≤ Usize.max` needed by
the generated `step + 1` addition; it is discharged from `Region.bound` at the call site. -/
private theorem tail_loop_some {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (home bound mask step slot : Usize)
    (wanted : U64) (hbound : bound.val ≤ Usize.max)
    (hrun : table.probe.spill_read_tail_loop keyrep r sigma place direct key home bound mask wanted
      step = ok (some slot)) :
    ∃ ss,
      r.coords.val[slot.val]? = some wanted ∧
      r.slots.val[slot.val]? = some ss ∧
      table.matches keyrep ss sigma key { place := place, direct_word := direct } = ok true := by
  have stepCases : ∀ (st : Usize),
      (table.probe.spill_read_tail_loop keyrep r sigma place direct key home bound mask wanted st
        = ok (some slot)) →
      (∃ ss, r.coords.val[slot.val]? = some wanted ∧ r.slots.val[slot.val]? = some ss ∧
          table.matches keyrep ss sigma key { place := place, direct_word := direct } = ok true) ∨
        (∃ st1, bound.val - st1.val < bound.val - st.val ∧
          table.probe.spill_read_tail_loop keyrep r sigma place direct key home bound mask
            wanted st1 = ok (some slot)) := by
    intro st hloop
    obtain ⟨rr, hbody, hred⟩ := loop_run_cases
      (fun st1 => table.probe.spill_read_tail_loop.body keyrep r sigma place direct key home
        bound mask wanted st1) st (some slot) hloop
    obtain hcases := tail_body_cases keyrep r sigma place direct key home bound mask st wanted rr
      hbody
    rcases hcases with hstopBig | hstopSmall | hhit | hcont
    · obtain ⟨hout, _⟩ := hstopBig
      rw [hout] at hred
      exact False.elim ((Option.some_ne_none slot) hred.symm)
    · obtain ⟨hout, _⟩ := hstopSmall
      rw [hout] at hred
      exact False.elim ((Option.some_ne_none slot) hred.symm)
    · obtain ⟨hltv, s, houts, hcoord, ss, hslot, hm⟩ := hhit
      rw [houts] at hred
      have hse : s = slot := Option.some.inj hred
      refine Or.inl ⟨ss, ?_, ?_, ?_⟩
      · rw [← hse]; exact hcoord
      · rw [← hse]; exact hslot
      · exact hm
    · obtain ⟨hltv, st1, hout, hstep1⟩ := hcont
      rw [hout] at hred
      have hcap : st.val + 1 ≤ Usize.max := by omega
      have hval : st1.val = st.val + 1 := by
        obtain ⟨sum, hsumRun, hsumVal⟩ := WP.spec_imp_exists
          (Usize.add_spec (x := st) (y := 1#usize) hcap)
        have hstep1' : ok sum = ok st1 := hsumRun.symm.trans hstep1
        have hsumEq : sum = st1 := Result.ok_injective hstep1'
        exact hsumEq ▸ hsumVal
      refine Or.inr ⟨st1, ?_, hred⟩
      omega
  have aux : ∀ n : Nat, ∀ st : Usize, bound.val - st.val = n →
      (table.probe.spill_read_tail_loop keyrep r sigma place direct key home bound mask wanted st
        = ok (some slot)) →
      ∃ ss, r.coords.val[slot.val]? = some wanted ∧ r.slots.val[slot.val]? = some ss ∧
        table.matches keyrep ss sigma key { place := place, direct_word := direct } = ok true := by
    intro n
    induction n using Nat.strong_induction_on with
    | _ n ih =>
        intro st hn hloop
        cases stepCases st hloop with
        | inl hfound => exact hfound
        | inr hdown =>
            obtain ⟨st1, hdec, hloop1⟩ := hdown
            cases n with
            | zero => rw [hn] at hdec; exact False.elim ((Nat.not_lt_zero _) hdec)
            | succ m => exact ih (bound.val - st1.val) (by simpa [hn] using hdec) st1 rfl hloop1
  exact aux (bound.val - step.val) step rfl hrun

/-! ## The generated tail function -/

/-- The generated `spill_read_tail` (`proof/ApeironWord.lean:20538-20548`) reports a hit only at a
slot whose coordinate word is `Region.coord_of address.place` and whose payload is a full
`(sigma, key)` matcher success. -/
private theorem spill_read_tail_some {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : K) (home slot : Usize)
    (hresult : table.probe.spill_read_tail keyrep r sigma address key home = ok (some slot)) :
    ∃ wanted ss,
      table.Region.coord_of R V address.place = ok wanted ∧
      r.coords.val[slot.val]? = some wanted ∧
      r.slots.val[slot.val]? = some ss ∧
      table.matches keyrep ss sigma key address = ok true := by
  unfold table.probe.spill_read_tail at hresult
  obtain ⟨bound, hbound, hrest⟩ := bind_ret_inv (table.Region.bound r) (fun bound => do
      let mask ← bound - 1#usize
      let wanted ← table.Region.coord_of R V address.place
      table.probe.spill_read_tail_loop keyrep r sigma address.place address.direct_word key home
        bound mask wanted 1#usize) (some slot) hresult
  obtain ⟨mask, _, hrest'⟩ := bind_ret_inv (bound - 1#usize) (fun mask => do
      let wanted ← table.Region.coord_of R V address.place
      table.probe.spill_read_tail_loop keyrep r sigma address.place address.direct_word key home
        bound mask wanted 1#usize) (some slot) hrest
  obtain ⟨wanted, hwanted, hrest''⟩ := bind_ret_inv (table.Region.coord_of R V address.place)
    (fun wanted => table.probe.spill_read_tail_loop keyrep r sigma address.place
      address.direct_word key home bound mask wanted 1#usize) (some slot) hrest'
  have hprop := r.slots.property
  change r.slots.length ≤ Usize.max at hprop
  have hboundVal : bound.val = r.slots.length := by
    have hlen : alloc.vec.Vec.len r.slots = bound := by
      unfold table.Region.bound at hbound
      exact Result.ok_injective hbound
    have hval := congrArg UScalar.val hlen
    rw [alloc.vec.Vec.len_val] at hval
    exact hval.symm
  obtain ⟨ss, hcoord, hslot, hmatch⟩ :=
    tail_loop_some keyrep r sigma address.place address.direct_word key home bound mask 1#usize
      slot wanted (by rw [hboundVal]; exact hprop) hrest''
  exact ⟨wanted, ss, hwanted, hcoord, hslot, hmatch⟩

/-! ## The generated read -/

/-- The generated `spill_read` (`proof/ApeironWord.lean:20552-20591`) reaches its tail exactly when
its straight-line resident step did not certify a hit.  A reported hit is either the resident home
step, carrying that slot's coordinate word and its full-key matcher success, or a tail hit. -/
private theorem spill_read_some_cases {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : K) (n slot : Usize)
    (hresult : table.probe.spill_read keyrep r sigma address key n = ok (some slot)) :
    ∃ home wanted,
      table.Region.home r address.place = ok home ∧
      table.Region.coord_of R V address.place = ok wanted ∧
      ((r.coords.val[home.val]? = some wanted ∧ slot = home ∧
          ∃ ss, r.slots.val[home.val]? = some ss ∧
            table.matches keyrep ss sigma key address = ok true) ∨
       table.probe.spill_read_tail keyrep r sigma address key home = ok (some slot)) := by
  unfold table.probe.spill_read at hresult
  by_cases hB : (alloc.vec.Vec.len r.slots != 0#usize) = true
  · rw [if_pos hB] at hresult
    obtain ⟨b, _, hresult⟩ := bind_ret_inv (table.valid_n n) (fun b =>
      if b then do
        let home ← table.Region.home r address.place
        let wanted ← table.Region.coord_of R V address.place
        let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
          r.coords home
        if found = table.COORD_EMPTY then ok none
        else
          do
            let hit ←
              if found = wanted then
                do
                  let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                    (table.SpillSlot R V)) r.slots home
                  let b1 ← table.matches keyrep ss sigma key address
                  if b1 then ok (some home) else ok none
              else ok none
            let b1 := core.option.Option.is_none hit
            if b1 then table.probe.spill_read_tail keyrep r sigma address key home
            else ok hit
      else ok none) (some slot) hresult
    by_cases hbT : b = true
    · rw [if_pos hbT] at hresult
      obtain ⟨home, hhome, hresult⟩ := bind_ret_inv (table.Region.home r address.place)
        (fun home => do
          let wanted ← table.Region.coord_of R V address.place
          let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
            r.coords home
          if found = table.COORD_EMPTY then ok none
          else
            do
              let hit ←
                if found = wanted then
                  do
                    let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                      (table.SpillSlot R V)) r.slots home
                    let b1 ← table.matches keyrep ss sigma key address
                    if b1 then ok (some home) else ok none
                else ok none
              let b1 := core.option.Option.is_none hit
              if b1 then table.probe.spill_read_tail keyrep r sigma address key home
              else ok hit) (some slot) hresult
      obtain ⟨wanted, hwanted, hresult⟩ := bind_ret_inv
        (table.Region.coord_of R V address.place) (fun wanted => do
          let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
            r.coords home
          if found = table.COORD_EMPTY then ok none
          else
            do
              let hit ←
                if found = wanted then
                  do
                    let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                      (table.SpillSlot R V)) r.slots home
                    let b1 ← table.matches keyrep ss sigma key address
                    if b1 then ok (some home) else ok none
                else ok none
              let b1 := core.option.Option.is_none hit
              if b1 then table.probe.spill_read_tail keyrep r sigma address key home
              else ok hit) (some slot) hresult
      obtain ⟨found, hfound, hresult⟩ := bind_ret_inv
        (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64) r.coords home)
        (fun found =>
          if found = table.COORD_EMPTY then ok none
          else
            do
              let hit ←
                if found = wanted then
                  do
                    let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                      (table.SpillSlot R V)) r.slots home
                    let b1 ← table.matches keyrep ss sigma key address
                    if b1 then ok (some home) else ok none
                else ok none
              let b1 := core.option.Option.is_none hit
              if b1 then table.probe.spill_read_tail keyrep r sigma address key home
              else ok hit) (some slot) hresult
      refine ⟨home, wanted, hhome, hwanted, ?_⟩
      by_cases hEmpty : found = table.COORD_EMPTY
      · rw [if_pos hEmpty] at hresult
        exact False.elim (ok_none_ne_ok_some slot hresult)
      · rw [if_neg hEmpty] at hresult
        by_cases hWant : found = wanted
        · rw [if_pos hWant] at hresult
          obtain ⟨hit, hhitEq, hrest'⟩ := bind_ret_inv
            (do
              let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                (table.SpillSlot R V)) r.slots home
              let b1 ← table.matches keyrep ss sigma key address
              if b1 then ok (some home) else ok none) (fun hit => do
                let b1 := core.option.Option.is_none hit
                if b1 then table.probe.spill_read_tail keyrep r sigma address key home
                else ok hit) (some slot) hresult
          obtain ⟨ss, hss, hhit⟩ := bind_ret_inv
            (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
              (table.SpillSlot R V)) r.slots home) (fun ss => do
                let b1 ← table.matches keyrep ss sigma key address
                if b1 then ok (some home) else ok none) hit hhitEq
          obtain ⟨m, hm, hhit'⟩ := bind_ret_inv (table.matches keyrep ss sigma key address)
            (fun m => if m then ok (some home) else ok none) hit hhit
          by_cases hmT : m = true
          · rw [if_pos hmT] at hhit'
            have hhitSome : (some home : Option Usize) = hit := Result.ok_injective hhit'
            rw [isNone_some_returns hit home hhitSome.symm
              (table.probe.spill_read_tail keyrep r sigma address key home)] at hrest'
            have hslotEq : some home = some slot := Result.ok_injective hrest'
            have hcoord := vec_index_readback r.coords home found hfound
            rw [hWant] at hcoord
            exact Or.inl ⟨hcoord, (Option.some.inj hslotEq).symm, ss,
              vec_index_readback r.slots home ss hss, hm.trans (congrArg ok hmT)⟩
          · rw [if_neg hmT] at hhit'
            have hitNone : hit = none := (Result.ok_injective hhit').symm
            rw [isNone_none_returns hit hitNone
              (table.probe.spill_read_tail keyrep r sigma address key home)] at hrest'
            exact Or.inr hrest'
        · rw [if_neg hWant, bind_tc_ok] at hresult
          rw [isNone_none_returns (none : Option _) rfl
            (table.probe.spill_read_tail keyrep r sigma address key home)] at hresult
          exact Or.inr hresult
    · rw [if_neg hbT] at hresult
      exact False.elim (ok_none_ne_ok_some slot hresult)
  · rw [if_neg hB] at hresult
    exact False.elim (ok_none_ne_ok_some slot hresult)

/-! ## Headline: the `some` half of the gapless-linear-insert read lemma -/

/-- **The `some` half of the gapless-linear-insert read lemma, for every face `sigma` and every
key.**  On the generated `table.probe.spill_read` body (`proof/ApeironWord.lean:20552-20591`), a
reported `Some slot` is certified to satisfy both conjuncts of the obligation: the coordinate word
read at `slot` is `Region.coord_of` of the query's routed `place`, and the payload read at `slot`
is a success of the extracted full-key matcher `table.matches` at the query's `(sigma, key)`.  No
insert-path invariant is assumed. -/
theorem spill_read_some_certifies_coordinate_and_payload {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : K) (n slot : Usize)
    (hresult : table.probe.spill_read keyrep r sigma address key n = ok (some slot)) :
    ∃ wanted ss,
      table.Region.coord_of R V address.place = ok wanted ∧
      r.coords.val[slot.val]? = some wanted ∧
      r.slots.val[slot.val]? = some ss ∧
      table.matches keyrep ss sigma key address = ok true := by
  obtain ⟨home, wanted, _, hwanted, hcase⟩ :=
    spill_read_some_cases keyrep r sigma address key n slot hresult
  rcases hcase with hhit | htail
  · obtain ⟨hcoord, hslotEq, ss, hss, hm⟩ := hhit
    rw [hslotEq]
    exact ⟨wanted, ss, hwanted, hcoord, hss, hm⟩
  · obtain ⟨w, ss, hw, hcoord, hslot, hm⟩ :=
    spill_read_tail_some keyrep r sigma address key home slot htail
    exact ⟨w, ss, hw, hcoord, hslot, hm⟩

/-- The extracted `table.matches` at the carried-key face is canonical logical-key equality at every
tag, not only at the vacant tag the existing `Apeiron.GrowableIdentity.stored_matches_iff` uses. -/
private theorem stored_matches_slot_iff (slotTag : U8) (slotSigma : cell.Sigma)
    (held : Apeiron.GrowableIdentity.StoredI64) (value : I64) (querySigma : cell.Sigma)
    (query : I64) (address : table.TableAddress) :
    table.matches Apeiron.GrowableIdentity.storedKeyRep
      { tag := slotTag, sigma := slotSigma, held := held, value := value } querySigma query address
        = ok true ↔
      Apeiron.GrowableIdentity.logicalStored slotSigma held = (querySigma, query) := by
  simpa only [table.matches] using
    (Apeiron.GrowableIdentity.stored_matches_iff slotSigma querySigma held value query address)

/-- **The `some` half at the carried-key face, stated on the canonical logical key.**  For every
face `sigma` and key, the generated `spill_read` returns `Some slot` only when `coords[slot]` is the
query's coordinate word and slot `slot` holds the canonical `(sigma, key)` entry, i.e. the slot's
own merged value. -/
theorem stored_spill_read_some_is_canonical_key
    (r : table.Region Apeiron.GrowableIdentity.StoredI64 I64) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : I64) (n slot : Usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep r sigma address key n
      = ok (some slot)) :
    ∃ wanted ss,
      table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place = ok wanted ∧
      r.coords.val[slot.val]? = some wanted ∧
      r.slots.val[slot.val]? = some ss ∧
      Apeiron.GrowableIdentity.logicalStored ss.sigma ss.held = (sigma, key) := by
  obtain ⟨wanted, ss, hwanted, hcoord, hslot, hm⟩ :=
    spill_read_some_certifies_coordinate_and_payload Apeiron.GrowableIdentity.storedKeyRep r
      sigma address key n slot hresult
  exact ⟨wanted, ss, hwanted, hcoord, hslot,
    (stored_matches_slot_iff ss.tag ss.sigma ss.held ss.value sigma key address).mp hm⟩

/-! ## The read path: the located slot's own merged value is what is returned -/

/-- **The generated read path returns the located slot's own merged payload.**
`table.get_prepared_shard` (`proof/ApeironWord.lean:20594-20613`, `src/table/prepared.rs:94-105`)
composed with `spill_read_some_certifies_coordinate_and_payload`: a returned value is exactly
`ss.value` of a slot whose coordinate word is the query's and whose payload is the extracted
full-key matcher success at the face the generated `FaceDomain` selected. -/
theorem get_prepared_shard_returns_located_payload {K H R V Clause0_Faces : Type}
    (coordaddressFaceCoordInst : coord.address.FaceCoord H K Clause0_Faces)
    (keyrep : table.keyrep.KeyRep R K H) (region : table.Region R V) (face : Clause0_Faces)
    (address : table.TableAddress) (key : K) (n : Usize) (value : V)
    (hresult : table.get_prepared_shard coordaddressFaceCoordInst keyrep region face address key n
      = ok (some value)) :
    ∃ sigma wanted ss,
      ∃ slot : Usize,
        table.Region.coord_of R V address.place = ok wanted ∧
        region.coords.val[slot.val]? = some wanted ∧
        region.slots.val[slot.val]? = some ss ∧ ss.value = value ∧
        table.matches keyrep ss sigma key address = ok true := by
  unfold table.get_prepared_shard at hresult
  obtain ⟨sigma, _, hrest⟩ := bind_ret_inv
    (coordaddressFaceCoordInst.FaceDomainInst.sigma face) (fun sigma => do
      let o ← table.probe.spill_read keyrep region sigma address key n
      match o with
      | none => ok none
      | some index =>
          do
            let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
              (table.SpillSlot R V)) region.slots index
            ok (some ss.value)) (some value) hresult
  cases hread : (table.probe.spill_read keyrep region sigma address key n) with
  | vis eff k =>
      rw [hread, bind_tc_vis] at hrest
      exact (vis_not_ok hrest).elim
  | div => rw [hread, bind_tc_div] at hrest; exact (div_not_ok hrest).elim
  | ret o =>
      rw [hread, bind_tc_ok] at hrest
      cases o with
      | none =>
          have hne : (ok (none : Option V)) = ok (some value) := hrest
          exact False.elim (ok_none_ne_ok_some value hne)
      | some slot =>
          obtain ⟨ss, hss, hrest'⟩ := bind_ret_inv
            (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
              region.slots slot) (fun ss => ok (some ss.value)) (some value) hrest
          have hvalue : ss.value = value := Option.some.inj (Result.ok_injective hrest')
          obtain ⟨hw, hss'', hwanted, hcoord, hslots, hm⟩ :=
            spill_read_some_certifies_coordinate_and_payload keyrep region sigma address key n
              slot hread
          have hss' := vec_index_readback region.slots slot ss hss
          have hssEq : hss'' = ss := Option.some.inj (hslots.symm.trans hss')
          subst hss''
          refine ⟨sigma, hw, ss, slot, hwanted, hcoord, hslots, ?_, hm⟩
          exact hvalue


/-! ## Cycle enumeration of the swept slot

The tail loop reads `let i1 ← home + step` then `let slot ← lift (i1 &&& mask)`, with
`mask ← bound - 1#usize` and `step` starting at `1#usize` (`proof/ApeironWord.lean:20486-20548`,
`src/table/probe.rs:303-342`).  `table.Region.bound r` is a power of two — `table.valid_n` admits
only `is_power_of_two` extents (`ApeironWord.lean:20431`) — so the sweep
`step ↦ (home + step) &&& mask` is the cycle `step ↦ (home + step) mod bound`.
`tail_slot_cycle_enumerates` proves that this cycle is a bijection of `[0, bound)`; the tail's own
range `step ∈ [1, bound)` is that bijection with the `step = 0` slot `home` removed, and the
straight-line resident step reads `home`.

The statement uses `UScalar.val`, `UScalar.and` and the total scalar addition alone: no `Region`, no
`Region.home`, no `Result`.  The generated body's `home + step` is the checked addition; under
`home.val < bound.val` and `step.val < bound.val` with `bound.val = 2^exponent ≤ Usize.max` the sum
cannot overflow, so its value is `core.num.Usize.wrapping_add`'s, which is what the statement
uses. -/

/-- Masking a scalar with `2^e - 1` reads its residue class modulo `2^e`. -/
private theorem scalar_and_mask_val (x mask : Std.Usize) (e : Nat)
    (hmask : mask.val = 2 ^ e - 1) :
    (UScalar.and x mask).val = x.val % 2 ^ e := by
  change (x &&& mask).val = x.val % 2 ^ e
  rw [UScalar.val_and, hmask, Nat.and_two_pow_sub_one_eq_mod]

/-- The swept slot `(home + step) &&& mask`, read as a value, is the residue of `home + step`
modulo the power-of-two extent `bound.val`. -/
private theorem swept_slot_val (home bound mask step : Std.Usize) (exponent : Nat)
    (hbound : bound.val = 2 ^ exponent) (hmask : mask.val = bound.val - 1) :
    (UScalar.and (core.num.Usize.wrapping_add home step) mask).val =
      (home.val + step.val) % bound.val := by
  change (core.num.Usize.wrapping_add home step &&& mask).val =
      (home.val + step.val) % bound.val
  have hwidth : exponent ≤ System.Platform.numBits := by
    by_contra h
    have hlarge := Nat.pow_lt_pow_right (show 1 < (2 : Nat) by omega)
      (show System.Platform.numBits < exponent by omega)
    have hbounds := bound.hBounds
    simp only [UScalarTy.Usize_numBits_eq] at hbounds
    omega
  rw [UScalar.val_and, core.num.Usize.wrapping_add_val_eq, hmask, hbound]
  have hsize : UScalar.size .Usize = 2 ^ System.Platform.numBits := by
    simp [Usize.size, Usize.numBits]
  rw [hsize, Nat.and_two_pow_sub_one_eq_mod]
  exact Nat.mod_mod_of_dvd _ (Nat.pow_dvd_pow 2 hwidth)

/-- For a target index inside the extent, the step that reaches it: `(home + j) mod bound = target`
with `j < bound`. -/
private theorem cycle_step_exists (home target bound : Nat)
    (hhome : home < bound) (htarget : target < bound) :
    ∃ j, j < bound ∧ (home + j) % bound = target := by
  by_cases hlt : target < home
  · refine ⟨target + bound - home, by omega, ?_⟩
    have hsum : home + (target + bound - home) = target + bound := by omega
    rw [hsum, Nat.add_mod_right, Nat.mod_eq_of_lt htarget]
  · refine ⟨target - home, by omega, ?_⟩
    have hsum : home + (target - home) = target := by omega
    rw [hsum, Nat.mod_eq_of_lt htarget]

/-- **Cycle enumeration of the ported `spill_read` tail.**  For a power-of-two extent `bound`
(`bound.val = 2^exponent`) masked to `mask.val = bound.val - 1`, and a resident index
`home.val < bound.val`, the swept slot `(home + step) &&& mask` runs over every index of the region
exactly once as `step` ranges over `[0, bound)`: every target below `bound` is reached by some step
below `bound`, and no two steps below `bound` reach the same target.  Stated over `UScalar.val`,
`UScalar.and` and the total scalar addition alone — no `Region`, no `Region.home`, no `Result`. -/
theorem tail_slot_cycle_enumerates (home bound mask : Std.Usize) (exponent : Nat)
    (hbound : bound.val = 2 ^ exponent)
    (hmask : mask.val = bound.val - 1)
    (hhome : home.val < bound.val) :
    (∀ target : Std.Usize, target.val < bound.val →
        ∃ step : Std.Usize, step.val < bound.val ∧
          (UScalar.and (core.num.Usize.wrapping_add home step) mask).val = target.val) ∧
    (∀ step₁ step₂ : Std.Usize,
        step₁.val < bound.val → step₂.val < bound.val →
        (UScalar.and (core.num.Usize.wrapping_add home step₁) mask).val =
          (UScalar.and (core.num.Usize.wrapping_add home step₂) mask).val →
        step₁.val = step₂.val) := by
  refine ⟨?_, ?_⟩
  · intro target htarget
    obtain ⟨j, hjlt, hjmod⟩ := cycle_step_exists home.val target.val bound.val hhome htarget
    have hjbound : j < 2 ^ UScalarTy.Usize.numBits := lt_trans hjlt bound.hBounds
    refine ⟨UScalar.ofNatCore j hjbound, ?_, ?_⟩
    · rw [UScalar.ofNatCore_val_eq]; exact hjlt
    · rw [swept_slot_val home bound mask _ exponent hbound hmask,
          UScalar.ofNatCore_val_eq, hjmod]
  · intro step₁ step₂ hs₁ hs₂ heq
    rw [swept_slot_val home bound mask step₁ exponent hbound hmask,
        swept_slot_val home bound mask step₂ exponent hbound hmask] at heq
    exact Nat.ModEq.eq_of_lt_of_lt
      (Nat.ModEq.add_left_cancel' home.val heq) hs₁ hs₂

/-! ## The coordinate store is the publication point

The write side and the read side meet here.  `spill_read` keys a resident slot on `coords[i]`
(`proof/ApeironWord.lean:20552-20591`) and the tail's swept slot on `(home + step) &&& mask`;
`publish_live` (`proof/ApeironWord.lean:20687-20719`, `src/table/state.rs:194-205`) writes the
payload triple `(sigma, held, value)` and the occupancy `tag` into `slots[index]`, and only then
writes `coords[index] = Region.coord_of place`.  The coordinate lane is the search-visible word, so
the coordinate store is the publication point: until it happens `index` is not search-visible, and
the payload lane is already final.  A reader that keys on the coordinate word therefore never
observes a live coordinate over an unpublished payload.

The final-state shape is `Growable.InsertRefine.publish_live_publish_ordered`; the theorem below
reuses it and adds the *order*: the returned coordinate lane is the pre-publication lane
`before.coords` updated at `index` with the live word `Region.coord_of place`, and the payload lane
`after.slots` is already complete in the intermediate region whose coordinate lane is still
`before.coords`.  It is all-`sigma`: it reads the word lanes and the `coord_of` map, not the ring
identity, so it carries no face boundary. -/

/-- The executed body of `Region.publish_live` rewrites exactly the coordinate lane at `index`, and
that lane is the pre-publication lane `before.coords` updated with the live word
`Region.coord_of place`: the coordinate store is the publication point. -/
private theorem publish_live_coords_eq_set {R V : Type}
    (before after : table.Region R V) (index : Usize) (sigma : cell.Sigma) (held : R)
    (value : V) (place : U64) (c : U64)
    (hpub : table.Region.publish_live before index (sigma, held, value) place = ok after)
    (hcoord : table.Region.coord_of R V place = ok c) :
    after.coords = before.coords.set index c := by
  unfold table.Region.publish_live at hpub
  rw [alloc.vec.Vec.index_mut_slice_index] at hpub
  obtain ⟨_, -, hpub⟩ := bind_ret_inv _ _ _ hpub
  obtain ⟨_, -, hpub⟩ := bind_ret_inv _ _ _ hpub
  obtain ⟨_, -, hpub⟩ := bind_ret_inv _ _ _ hpub
  obtain ⟨_, -, hpub⟩ := bind_ret_inv _ _ _ hpub
  obtain ⟨_, -, hpub⟩ := bind_ret_inv _ _ _ hpub
  rw [hcoord] at hpub
  obtain ⟨w1, hw1, hpub⟩ := bind_ret_inv _ _ _ hpub
  obtain ⟨r5, h5, hpub⟩ := bind_ret_inv _ _ _ hpub
  obtain ⟨_, -, hpub⟩ := bind_ret_inv _ _ _ hpub
  have hw1c : w1 = c := (Result.ok_injective hw1).symm
  subst w1
  simp at hpub
  subst after
  rw [alloc.vec.Vec.index_mut_slice_index] at h5
  have hbound4 : index.val < before.coords.val.length := by
    cases hget : before.coords.val[index.val]? with
    | none => simp [alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize, hget] at h5
    | some old => exact (List.getElem?_eq_some_iff.mp hget).1
  have h5spec := alloc.vec.Vec.index_mut_usize_spec before.coords index hbound4
  rw [h5, WP.spec_ok] at h5spec
  show r5.2 c = before.coords.set index c
  rw [h5spec.2]

/-- **The coordinate store is `Region.publish_live`'s publication point.**  On the generated body
the payload and occupancy stores precede the coordinate store, so the returned coordinate lane is
the pre-publication lane updated at `index` with the live word `Region.coord_of place`; the payload
lane is already final in the intermediate region whose coordinate lane is still `before.coords`; and
every other coordinate and slot is unchanged.  A reader keyed on the coordinate word cannot see a
live coordinate over an unpublished payload at `index`, and sees no change elsewhere.  All-`sigma`;
no insert-path invariant is assumed. -/
theorem publish_live_coordinate_is_publication_point {R V : Type}
    (before after : table.Region R V) (index : Usize) (sigma : cell.Sigma)
    (held : R) (value : V) (place : U64)
    (hpub : table.Region.publish_live before index (sigma, held, value) place = ok after) :
    ∃ (staged : table.Region R V) (slot : table.SpillSlot R V) (c : U64),
      staged.coords = before.coords ∧
      staged.slots = after.slots ∧
      after.coords = before.coords.set index c ∧
      table.Region.coord_of R V place = ok c ∧
      c ≠ table.COORD_EMPTY ∧
      after.slots.val[index.val]? = some slot ∧
      slot.sigma = sigma ∧ slot.held = held ∧ slot.value = value ∧
      (∀ j : Nat, j ≠ index.val → after.coords.val[j]? = before.coords.val[j]?) ∧
      (∀ j : Nat, j ≠ index.val → after.slots.val[j]? = before.slots.val[j]?) := by
  obtain ⟨⟨c, hcoord, _, hne⟩, ⟨slot, hslot, hsigma, hheld, hvalue⟩, hcf, hsf⟩ :=
    Apeiron.GrowableInsertRefine.publish_live_publish_ordered
      before after index sigma held value place hpub
  have hset := publish_live_coords_eq_set before after index sigma held value place c hpub hcoord
  exact ⟨{ before with slots := after.slots }, slot, c, rfl, rfl, hset, hcoord, hne,
    hslot, hsigma, hheld, hvalue, hcf, hsf⟩

/-! ## Coordinate soundness of a live slot

The read side of the port treats the coordinate word `coords[i]` as the occupancy *and* identity
prefilter: `table.probe.spill_read` reads `coords[home]` and `table.probe.spill_read_tail_loop.body`
reads `coords[slot]`, then reads `slots[slot]` and runs the full-key matcher
(`proof/ApeironWord.lean:20579-20619` and `20512-20547`).  The tag analogue of the invariant below is
`Apeiron.GrowableReadLoop.StoredLiveTagAddress` (`Growable/ReadState.lean:46`): it reads the slot's
own `tag` and says nothing about `coords`.  This section states and proves the `coords` analogue —
at every live slot the coordinate word is the coordinate word of the slot's *own* stored key, and it
is never the vacant sentinel `table.COORD_EMPTY`.

The two facts it rests on are the constructor and the publication, not an assumed invariant:

* `table.Region.publish_live` (`proof/ApeironWord.lean:20687-20719`, `src/table/state.rs:194-205`)
  writes the payload, then the occupancy `tag`, then `coords[index] = Region.coord_of place` last.
  `publish_live_coords` below is the executed-body read of that call: the located coordinate is
  `Region.coord_of place`, live, and every other coordinate is framed.  `Region.coord_of`
  (`proof/ApeironWord.lean:20479-20484`) never returns `COORD_EMPTY` — the nudge moves the one
  colliding value to `COORD_EMPTY - 1`.
* `table.Table.new` (`proof/ApeironWord.lean:21559-21579`) pushes only `table.Region.empty` shards;
  its regions carry no slots at all, so no slot is live and the invariant holds vacuously.  The
  certificate route is `word_stored_table_new_repR` (`Growable/Representation.lean:591`) with the
  empty live set, the same route the tag analogue uses.

The statement is all-`σ`: it reads the word-level `coords` array, `Region.coord_of`, `table.shard`
and the routing `wordHomeInst.address`, and it never cases on `cell.Sigma` or touches the ring
identity.  So it holds at the elliptic `σ<0`, parabolic `σ=0` and hyperbolic `σ>0` faces alike. -/

/-- **Coordinate soundness of a live slot.**  Every live spill slot — read-back `tag ≠ 0` — carries
a coordinate word, that word is not the vacant sentinel `table.COORD_EMPTY`, and it is exactly
`Region.coord_of` of the `place` the table's own routing `wordHomeInst.address` computes from the
slot's stored `(sigma, held.key)`, at the slot's own shard.  This is the `coords` analogue of
`Apeiron.GrowableReadLoop.StoredLiveTagAddress`; that invariant reads `tag` and says nothing about
`coords`. -/
def StoredLiveCoordAddress
    (state : Apeiron.GrowableRepresentation.WordTable
      Apeiron.GrowableIdentity.StoredI64) : Prop :=
  ∀ (shard index : Usize)
      (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
      (tag : U8) (sigma : cell.Sigma)
      (held : Apeiron.GrowableIdentity.StoredI64) (value : I64),
    state.shards.val[shard.val]? = some region →
    region.slots.val[index.val]? =
      some { tag := tag, sigma := sigma, held := held, value := value } → tag ≠ 0#u8 →
    ∃ (coord : U64) (address : table.TableAddress),
      region.coords.val[index.val]? = some coord ∧ coord ≠ table.COORD_EMPTY ∧
      Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma held.key = ok address ∧
      table.shard address = ok shard ∧
      table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place = ok coord

/-- A live slot's coordinate word is never the vacant sentinel.  Immediate from the routing
conclusion of `StoredLiveCoordAddress` and the nudge of `Region.coord_of`.  `Region.coord_of` folds
`COORD_EMPTY` to `COORD_EMPTY - 1`; a scalar is never both the sentinel and the sentinel minus one. -/
theorem stored_live_coord_address_ne_empty
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveCoordAddress state)
    (shard index : Usize)
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (coord : U64) (tag : U8) (sigma : cell.Sigma)
    (held : Apeiron.GrowableIdentity.StoredI64) (value : I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (hslot : region.slots.val[index.val]? =
      some { tag := tag, sigma := sigma, held := held, value := value })
    (hcoord : region.coords.val[index.val]? = some coord) (htag : tag ≠ 0#u8) :
    coord ≠ table.COORD_EMPTY := by
  obtain ⟨coord', addr, hcoord', hcNe, _⟩ :=
    hstate shard index region tag sigma held value hregion hslot htag
  have hc : coord' = coord := Option.some.inj (hcoord'.symm.trans hcoord)
  rw [hc] at hcNe
  exact hcNe

/-- The input-side empty certificate initializes the coordinate invariant on its exact state: an
empty live set admits no live slot, so the conclusion is vacuous.  This is the `coords` companion of
`Apeiron.GrowableReadLoop.stored_live_tag_address_of_empty_repR`; it carries the same
`RepR`/`entryAt` route and no coordinate fact is assumed. -/
theorem stored_live_coord_address_of_empty_repR
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (default : I64)
    (hrep : Apeiron.GrowableRepresentation.RepR state
      Apeiron.GrowableSearch.decodeStoredOption (fun _ => none)
      (fun _ => False) (fun _ => default)) :
    StoredLiveCoordAddress state := by
  intro shard index region tag sigma held value hregion hslot htagNonzero
  have hentry : Apeiron.GrowableSearch.entryAt state
      Apeiron.GrowableSearch.decodeStoredOption (.spill shard index) =
        some ((sigma, held.key), value) := by
    simp [Apeiron.GrowableSearch.entryAt, hregion, hslot,
      htagNonzero, Apeiron.GrowableSearch.decodeStoredOption]
  exact False.elim ((hrep.liveSet (sigma, held.key)).mpr
    ⟨.spill shard index, value, hentry⟩)

/-- The real generated constructor establishes coordinate soundness: `table.Table.new`
(`proof/ApeironWord.lean:21559-21579`) builds its shards from `table.Region.empty` alone, and the
empty canonical relation carries no live slot.  The `coords` companion of
`Apeiron.GrowableReadLoop.word_stored_table_new_tagged_repR`. -/
theorem word_stored_table_new_coord_address (default : I64) :
    table.Table.new 16#usize Apeiron.GrowableIdentity.wordHomeInst
      Apeiron.GrowableIdentity.storedKeyRep default () ⦃ state =>
      StoredLiveCoordAddress state ⦄ := by
  apply WP.spec_mono
    (Apeiron.GrowableRepresentation.word_stored_table_new_repR default)
  intro state hrep
  exact stored_live_coord_address_of_empty_repR state default hrep

/-- `COORD_EMPTY - 1` is not `COORD_EMPTY`: the sentinel is `2 ^ 64 - 1`. -/
private theorem coord_empty_sub_one_ne (c : U64)
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

/-- The generated `Region.coord_of` never returns the vacant sentinel: it folds the one colliding
value to `COORD_EMPTY - 1`. -/
private theorem coord_of_result_ne_empty {R V : Type} (place : U64) (c : U64)
    (h : table.Region.coord_of R V place = ok c) : c ≠ table.COORD_EMPTY := by
  unfold table.Region.coord_of at h
  obtain ⟨raw, hraw, hrest⟩ := bind_ret_inv (place >>> table.SHARD_BITS)
    (fun raw => if raw = table.COORD_EMPTY then table.COORD_EMPTY - 1#u64 else ok raw) c h
  by_cases hraweq : raw = table.COORD_EMPTY
  · rw [if_pos hraweq] at hrest
    exact coord_empty_sub_one_ne c hrest
  · rw [if_neg hraweq] at hrest
    have hcr : c = raw := Result.ok_injective hrest.symm
    rw [hcr]
    exact hraweq

private theorem getElem?_set_ne {α : Type} (xs : List α) (written read : Nat)
    (value : α) (hne : read ≠ written) : (xs.set written value)[read]? = xs[read]? := by
  simp [hne]

/-- The coordinate lane of the generated `table.Region.publish_live`
(`proof/ApeironWord.lean:20687-20719`, `src/table/state.rs:194-205`): one call writes
`Region.coord_of place` — live — at `index` and frames every other coordinate.  Read directly from
the body, so the coordinate store is this module's own obligation; the slot lane is the landed
`Apeiron.GrowableRepresentation.publish_live_slot_readback`. -/
theorem publish_live_coords {R V : Type} (before after : table.Region R V) (index : Usize)
    (sigma : cell.Sigma) (held : R) (value : V) (place : U64)
    (hpub : table.Region.publish_live before index (sigma, held, value) place = ok after) :
    (∃ c, table.Region.coord_of R V place = ok c ∧
        after.coords.val[index.val]? = some c ∧ c ≠ table.COORD_EMPTY) ∧
      (∀ j : Nat, j ≠ index.val → after.coords.val[j]? = before.coords.val[j]?) := by
  unfold table.Region.publish_live at hpub
  rw [alloc.vec.Vec.index_mut_slice_index] at hpub
  cases h1 : alloc.vec.Vec.index_mut_usize before.slots index with
  | vis eff k => simp [h1] at hpub
  | div => simp [h1] at hpub
  | ret r1 =>
      rcases r1 with ⟨ss, back1⟩
      simp only [h1, bind_tc_ok] at hpub
      simp at hpub
      have hbound0 : index.val < before.slots.val.length := by
        cases hget : before.slots.val[index.val]? with
        | none => simp [alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize, hget] at h1
        | some old => exact (List.getElem?_eq_some_iff.mp hget).1
      have h1spec := alloc.vec.Vec.index_mut_usize_spec before.slots index hbound0
      rw [h1, WP.spec_ok] at h1spec
      cases h2 : alloc.vec.Vec.index_mut_usize (back1 { ss with sigma := sigma }) index with
      | vis eff k => simp [h2] at hpub
      | div => simp [h2] at hpub
      | ret r2 =>
          rcases r2 with ⟨ss1, back2⟩
          simp only [h2, bind_tc_ok] at hpub
          simp at hpub
          have hbound1 : index.val < (back1 { ss with sigma := sigma }).val.length := by
            rw [h1spec.2, alloc.vec.Vec.set_val_eq]
            simpa using hbound0
          have h2spec := alloc.vec.Vec.index_mut_usize_spec
            (back1 { ss with sigma := sigma }) index hbound1
          rw [h2, WP.spec_ok] at h2spec
          cases h3 : alloc.vec.Vec.index_mut_usize (back2 { ss1 with held := held }) index with
          | vis eff k => simp [h3] at hpub
          | div => simp [h3] at hpub
          | ret r3 =>
              rcases r3 with ⟨ss2, back3⟩
              simp only [h3, bind_tc_ok] at hpub
              simp at hpub
              have hbound2 : index.val < (back2 { ss1 with held := held }).val.length := by
                rw [h2spec.2, alloc.vec.Vec.set_val_eq]
                simpa using hbound1
              have h3spec := alloc.vec.Vec.index_mut_usize_spec
                (back2 { ss1 with held := held }) index hbound2
              rw [h3, WP.spec_ok] at h3spec
              generalize htag : table.tag place = tagResult at hpub
              cases tagResult with
              | vis eff k => simp at hpub
              | div => simp at hpub
              | ret tag =>
                  cases h4 : alloc.vec.Vec.index_mut_usize (back3 { ss2 with value := value }) index with
                  | vis eff k => simp [h4] at hpub
                  | div => simp [h4] at hpub
                  | ret r4 =>
                      rcases r4 with ⟨ss3, back4⟩
                      simp only [h4, bind_tc_ok] at hpub
                      simp at hpub
                      have hbound3 :
                          index.val < (back3 { ss2 with value := value }).val.length := by
                        rw [h3spec.2, alloc.vec.Vec.set_val_eq]
                        simpa using hbound2
                      have h4spec := alloc.vec.Vec.index_mut_usize_spec
                        (back3 { ss2 with value := value }) index hbound3
                      rw [h4, WP.spec_ok] at h4spec
                      cases h5 : table.Region.coord_of R V place with
                      | vis eff k => simp [h5] at hpub
                      | div => simp [h5] at hpub
                      | ret coord =>
                          cases h6 : alloc.vec.Vec.index_mut_usize before.coords index with
                          | vis eff k => simp [h5, h6] at hpub
                          | div => simp [h5, h6] at hpub
                          | ret r5 =>
                              rcases r5 with ⟨coordSlot, coordBack⟩
                              simp only [h5, h6, bind_tc_ok] at hpub
                              simp at hpub
                              generalize hadd : before.live + 1#usize = addResult at hpub
                              cases addResult with
                              | vis eff k => simp at hpub
                              | div => simp at hpub
                              | ret live1 =>
                                  simp at hpub
                                  subst after
                                  have hbound4 : index.val < before.coords.val.length := by
                                    cases hget : before.coords.val[index.val]? with
                                    | none =>
                                        simp [alloc.vec.Vec.index_mut_usize,
                                          alloc.vec.Vec.index_usize, hget] at h6
                                    | some old => exact (List.getElem?_eq_some_iff.mp hget).1
                                  have h6spec := alloc.vec.Vec.index_mut_usize_spec before.coords
                                    index hbound4
                                  rw [h6, WP.spec_ok] at h6spec
                                  refine ⟨⟨coord, rfl, ?_, coord_of_result_ne_empty place coord h5⟩, ?_⟩
                                  · rw [h6spec.2, alloc.vec.Vec.set_val_eq]
                                    exact List.getElem?_set_self hbound4
                                  · intro j hj
                                    rw [h6spec.2, alloc.vec.Vec.set_val_eq]
                                    exact getElem?_set_ne before.coords.val index.val j coord hj

/-- **The real generated publication preserves coordinate soundness.**  At the published `index` the
read-back slot carries the published `(sigma, held, value)` and `coords[index] = Region.coord_of
address.place`, live; every other coordinate and slot is framed by the same call.  Stated on
`table.Region.publish_live` (`proof/ApeironWord.lean:20687-20719`) itself, so it is the executed-body
fact and no returned-state coordinate equation is assumed. -/
theorem stored_live_coord_address_publish_spill
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveCoordAddress state)
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
    StoredLiveCoordAddress { state with shards := state.shards.set shard published } := by
  obtain ⟨⟨c, hc, hcoordPub, hcNe⟩, hcoordsFrame⟩ :=
    publish_live_coords region published index sigma held value address.place hpublish
  obtain ⟨slot, hslotPub, hslotSigma, hslotHeld, _⟩ :=
    Apeiron.GrowableRepresentation.publish_live_slot_readback region published index sigma held
      value address.place hpublish
  intro readShard readIndex readRegion readTag readSigma readHeld readValue
    hreadRegion hreadSlot hreadTagNonzero
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
      have hslotEq : slot =
          ({ tag := readTag, sigma := readSigma, held := readHeld, value := readValue } :
            table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64) :=
        Option.some.inj (hslotPub.symm.trans hreadSlot)
      have hsigma : readSigma = sigma :=
        (congrArg table.SpillSlot.sigma hslotEq).symm.trans hslotSigma
      have hheld : readHeld = held :=
        (congrArg table.SpillSlot.held hslotEq).symm.trans hslotHeld
      subst readSigma
      subst readHeld
      exact ⟨c, address, hcoordPub, hcNe, haddress, hshard, hc⟩
    · have hslotBefore : region.slots.val[readIndex.val]? =
          some { tag := readTag, sigma := readSigma, held := readHeld, value := readValue } :=
        (Apeiron.GrowableRepresentation.publish_live_slot_frame region published index readIndex
          (sigma, held, value) address.place hpublish hsameIndex).symm.trans hreadSlot
      obtain ⟨coord, addr, hcoord, hcNe, haddr, hsh, hcf⟩ :=
        hstate shard readIndex region readTag readSigma readHeld readValue
          hregion hslotBefore hreadTagNonzero
      exact ⟨coord, addr, (hcoordsFrame readIndex.val hsameIndex).trans hcoord,
        hcNe, haddr, hsh, hcf⟩
  · have hregionOld : state.shards.val[readShard.val]? = some readRegion := by
      simpa [alloc.vec.Vec.set_val_eq, hsameShard] using hreadRegion
    exact hstate readShard readIndex readRegion readTag readSigma readHeld readValue
      hregionOld hreadSlot hreadTagNonzero

/-- The coordinate invariant is framed by any state update that leaves the home routing and the
shard vector unchanged; the `coords` companion of
`Apeiron.GrowableReadLoop.stored_live_tag_address_of_home_shards_eq`. -/
theorem stored_live_coord_address_of_home_shards_eq
    (before after : Apeiron.GrowableRepresentation.WordTable
      Apeiron.GrowableIdentity.StoredI64)
    (hbefore : StoredLiveCoordAddress before)
    (hhome : after.home = before.home)
    (hshards : after.shards = before.shards) :
    StoredLiveCoordAddress after := by
  intro shard index region tag sigma held value hregion hslot htagNonzero
  have hregionBefore : before.shards.val[shard.val]? = some region := by
    simpa only [hshards] using hregion
  obtain ⟨coord, addr, hcoord, hcNe, haddr, hsh, hcf⟩ :=
    hbefore shard index region tag sigma held value hregionBefore hslot htagNonzero
  exact ⟨coord, addr, hcoord, hcNe, by simpa only [hhome] using haddr, hsh, hcf⟩

/-! ## The `None` half of the gapless-linear-insert read obligation

The read is a linear probe that stops at the first vacant coordinate.  `table.probe.spill_read`
(`proof/ApeironWord.lean:17486-17524`) reads `coords[home]` and returns `ok none` on the vacant
sentinel; otherwise it runs `table.probe.spill_read_tail` (`proof/ApeironWord.lean:17472-17482`),
whose loop body (`proof/ApeironWord.lean:17420-17452`) advances while the swept slot
`(home + step) &&& mask` is neither vacant nor a full hit, and stops at the first vacancy or on
exhaustion.  The `some` half above says a reported hit is certified; this half proves the negative
structure: a `none` reports that every slot the probe *reaches* is either vacant or fails the
coordinate/matcher test, and — given the first-fit gap discipline that a live representative never
sits past a vacancy in probe order — no slot anywhere in the region is a hit.  The search covers the
region by `tail_slot_cycle_enumerates`. -/

/-- A coord-hit at a natural slot index: the slot's coordinate word is the query's routed word and
its payload is an extracted full-key matcher success.  This is exactly the pair of facts the
generated probe accepts at a slot — `table.probe.spill_read` reads `coords`, then `slots`, then
`table.matches` (`proof/ApeironWord.lean:17486-17524`, `17420-17452`). -/
def ProbeHitAt {K H R V : Type} (keyrep : table.keyrep.KeyRep R K H) (r : table.Region R V)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : K) (wanted : U64)
    (j : Nat) : Prop :=
  r.coords.val[j]? = some wanted ∧
  ∃ ss, r.slots.val[j]? = some ss ∧ table.matches keyrep ss sigma key address = ok true

/-- The slot at a natural index is vacant: its coordinate word is the sentinel `COORD_EMPTY`, the
probe's own termination marker (`proof/ApeironWord.lean:17435`, `17503`). -/
def ProbeVacantAt {R V : Type} (r : table.Region R V) (j : Nat) : Prop :=
  r.coords.val[j]? = some table.COORD_EMPTY

/-- The natural value of the swept slot generated by the *checked* addition `home + step` and the
mask `mask = bound - 1`, when `bound` is the power-of-two extent: the residue of `home + step`
modulo `bound`.  This is the bridge from the generated body's
`let i1 ← home + step; let slot ← lift (i1 &&& mask)` to the arithmetic the cycle enumeration is
stated over.  The checked addition is a total `Result`, so a successful `home + step = ok i1`
already gives `i1.val = home.val + step.val` through `UScalar.add_equiv`; no overflow hypothesis is
needed. -/
private theorem swept_val_of_checked
    (home bound mask step i1 : Usize) (exponent : Nat)
    (hpow : bound.val = 2 ^ exponent) (hmask : mask.val = bound.val - 1)
    (hadd : home + step = ok i1) :
    (UScalar.and i1 mask).val = (home.val + step.val) % bound.val := by
  have hval : i1.val = home.val + step.val := by
    have h := UScalar.add_equiv (ty := .Usize) home step
    rw [hadd] at h
    exact h.2.1
  have hmask' : mask.val = 2 ^ exponent - 1 := by rw [hmask, hpow]
  rw [scalar_and_mask_val i1 mask exponent hmask', hval, hpow]

/-- The four outcomes of one generated tail-loop step, with the swept slot named: the guard fails
(`step ≥ bound`); the swept coordinate is vacant; the swept coordinate is the query's and the
matcher succeeds; or the step advances with the swept slot being *not* a hit.  This is
`tail_body_cases` with the swept slot `UScalar.and i1 mask` and the negative content of the
advancing branch exposed, so the negative half reads the body's own termination. -/
private theorem tail_body_cases_swept {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (home bound mask step : Usize) (wanted : U64)
    (out : ControlFlow Usize (Option Usize))
    (hbody : table.probe.spill_read_tail_loop.body keyrep r sigma place direct key home bound mask
      wanted step = ok out) :
    (out = ControlFlow.done none ∧ step.val ≥ bound.val) ∨
    (out = ControlFlow.done none ∧ step.val < bound.val ∧ ∃ i1, home + step = ok i1 ∧
      r.coords.val[(UScalar.and i1 mask).val]? = some table.COORD_EMPTY) ∨
    (step.val < bound.val ∧ ∃ i1, home + step = ok i1 ∧
      out = ControlFlow.done (some (UScalar.and i1 mask)) ∧
      r.coords.val[(UScalar.and i1 mask).val]? = some wanted ∧
      ∃ ss, r.slots.val[(UScalar.and i1 mask).val]? = some ss ∧
        table.matches keyrep ss sigma key { place := place, direct_word := direct } = ok true) ∨
    (step.val < bound.val ∧ ∃ i1 st1, home + step = ok i1 ∧
      out = ControlFlow.cont st1 ∧ step + 1#usize = ok st1 ∧
      ¬ (r.coords.val[(UScalar.and i1 mask).val]? = some wanted ∧
        ∃ ss, r.slots.val[(UScalar.and i1 mask).val]? = some ss ∧
          table.matches keyrep ss sigma key { place := place, direct_word := direct } = ok true)) := by
  unfold table.probe.spill_read_tail_loop.body at hbody
  by_cases hlt : step < bound
  · rw [if_pos hlt] at hbody
    have hltv : step.val < bound.val := hlt
    obtain ⟨i1, hadd, hrest⟩ := bind_ret_inv (home + step) (fun i1 => do
        let slot0 ← lift (i1 &&& mask)
        let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
          r.coords slot0
        if found = table.COORD_EMPTY then ok (ControlFlow.done none)
        else if found = wanted then
          do
            let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
              (table.SpillSlot R V)) r.slots slot0
            let m ← table.matches keyrep ss sigma key { place := place, direct_word := direct }
            if m then ok (ControlFlow.done (some slot0))
            else
              do
                let step1 ← step + 1#usize
                ok (ControlFlow.cont step1)
        else
          do
            let step1 ← step + 1#usize
            ok (ControlFlow.cont step1)) out hbody
    obtain ⟨slot0, hslot0, hrest'⟩ := bind_ret_inv (lift (i1 &&& mask)) (fun slot0 => do
        let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
          r.coords slot0
        if found = table.COORD_EMPTY then ok (ControlFlow.done none)
        else if found = wanted then
          do
            let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
              (table.SpillSlot R V)) r.slots slot0
            let m ← table.matches keyrep ss sigma key { place := place, direct_word := direct }
            if m then ok (ControlFlow.done (some slot0))
            else
              do
                let step1 ← step + 1#usize
                ok (ControlFlow.cont step1)
        else
          do
            let step1 ← step + 1#usize
            ok (ControlFlow.cont step1)) out hrest
    have hslot0eq : slot0 = UScalar.and i1 mask := by
      have hraw : i1 &&& mask = slot0 := Result.ok_injective hslot0
      rw [← hraw]
      rfl
    obtain ⟨found, hfound, hrest''⟩ := bind_ret_inv
      (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64) r.coords slot0)
      (fun found =>
        if found = table.COORD_EMPTY then ok (ControlFlow.done none)
        else if found = wanted then
          do
            let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
              (table.SpillSlot R V)) r.slots slot0
            let m ← table.matches keyrep ss sigma key { place := place, direct_word := direct }
            if m then ok (ControlFlow.done (some slot0))
            else
              do
                let step1 ← step + 1#usize
                ok (ControlFlow.cont step1)
        else
          do
            let step1 ← step + 1#usize
            ok (ControlFlow.cont step1)) out hrest'
    by_cases hEmpty : found = table.COORD_EMPTY
    · rw [if_pos hEmpty] at hrest''
      refine Or.inr (Or.inl ⟨(Result.ok_injective hrest'').symm, hltv, i1, hadd, ?_⟩)
      have hcoord := vec_index_readback r.coords slot0 found hfound
      rw [hslot0eq, hEmpty] at hcoord
      exact hcoord
    · rw [if_neg hEmpty] at hrest''
      by_cases hWant : found = wanted
      · rw [if_pos hWant] at hrest''
        obtain ⟨ss, hss, hrest3⟩ := bind_ret_inv
          (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
            (table.SpillSlot R V)) r.slots slot0) (fun ss => do
              let m ← table.matches keyrep ss sigma key { place := place, direct_word := direct }
              if m then ok (ControlFlow.done (some slot0))
              else
                do
                  let step1 ← step + 1#usize
                  ok (ControlFlow.cont step1)) out hrest''
        obtain ⟨m, hm, hrest4⟩ := bind_ret_inv
          (table.matches keyrep ss sigma key { place := place, direct_word := direct }) (fun m =>
            if m then ok (ControlFlow.done (some slot0))
            else
              do
                let step1 ← step + 1#usize
                ok (ControlFlow.cont step1)) out hrest3
        by_cases hmT : m = true
        · rw [if_pos hmT] at hrest4
          have hout : out = ControlFlow.done (some slot0) := (Result.ok_injective hrest4).symm
          refine Or.inr (Or.inr (Or.inl ⟨hltv, i1, hadd, ?_, ?_, ss, ?_, ?_⟩))
          · rw [hout, hslot0eq]
          · have hcoord := vec_index_readback r.coords slot0 found hfound
            rw [hslot0eq, hWant] at hcoord
            exact hcoord
          · have hslot := vec_index_readback r.slots slot0 ss hss
            rw [hslot0eq] at hslot
            exact hslot
          · exact hm.trans (congrArg ok hmT)
        · rw [if_neg hmT] at hrest4
          obtain ⟨step1, hstep1, hout⟩ := bind_ret_inv (step + 1#usize)
            (fun step1 => ok (ControlFlow.cont step1)) out hrest4
          refine Or.inr (Or.inr (Or.inr ⟨hltv, i1, step1, hadd,
            (Result.ok_injective hout).symm, hstep1, ?_⟩))
          intro ⟨_, ss', hsl', hm'⟩
          have hmf : m = false := by cases m <;> simp_all
          have hmFalse : table.matches keyrep ss sigma key
              { place := place, direct_word := direct } = ok false := by simpa [hmf] using hm
          have hslot : r.slots.val[(UScalar.and i1 mask).val]? = some ss := by
            have h := vec_index_readback r.slots slot0 ss hss
            rwa [hslot0eq] at h
          have hssEq : ss' = ss := Option.some.inj (hsl'.symm.trans hslot)
          rw [hssEq, hmFalse] at hm'
          simp at hm'
      · rw [if_neg hWant] at hrest''
        obtain ⟨step1, hstep1, hout⟩ := bind_ret_inv (step + 1#usize)
          (fun step1 => ok (ControlFlow.cont step1)) out hrest''
        refine Or.inr (Or.inr (Or.inr ⟨hltv, i1, step1, hadd,
          (Result.ok_injective hout).symm, hstep1, ?_⟩))
        intro ⟨hco, _⟩
        have hcd := vec_index_readback r.coords slot0 found hfound
        rw [hslot0eq] at hcd
        exact hWant (Option.some.inj (hcd.symm.trans hco))
  · rw [if_neg hlt] at hbody
    refine Or.inl ⟨(Result.ok_injective hbody).symm, ?_⟩
    exact Nat.le_of_not_gt (fun h => hlt ((UScalar.lt_equiv step bound).mpr h))

/-- **The negative scan of the ported tail.**  If the generated tail loop returns `ok none` starting
at `step`, then there is a stop position `stop ∈ [step, bound]` such that every swept slot before
`stop` is not a coord-hit, and if `stop` is strictly below the extent the swept slot there is vacant
(the probe's own termination).  This is the read's *negative* content and assumes no insert-path
invariant: the `none` is a property of the executed scan alone. -/
private theorem tail_loop_none_absent {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (home bound mask step : Usize) (wanted : U64)
    (exponent : Nat) (hpow : bound.val = 2 ^ exponent) (hmask : mask.val = bound.val - 1)
    (hstle : step.val ≤ bound.val)
    (hrun : table.probe.spill_read_tail_loop keyrep r sigma place direct key home bound mask wanted
      step = ok none) :
    ∃ stop : Nat, step.val ≤ stop ∧ stop ≤ bound.val ∧
      (∀ t : Nat, step.val ≤ t → t < stop →
        ¬ ProbeHitAt keyrep r sigma { place := place, direct_word := direct } key wanted
          ((home.val + t) % bound.val)) ∧
      (stop < bound.val → ProbeVacantAt r ((home.val + stop) % bound.val)) := by
  have aux : ∀ n : Nat, (∀ st : Usize, bound.val - st.val = n → st.val ≤ bound.val →
      table.probe.spill_read_tail_loop keyrep r sigma place direct key home bound mask wanted st
        = ok none →
      ∃ stop : Nat, st.val ≤ stop ∧ stop ≤ bound.val ∧
        (∀ t : Nat, st.val ≤ t → t < stop →
          ¬ ProbeHitAt keyrep r sigma { place := place, direct_word := direct } key wanted
            ((home.val + t) % bound.val)) ∧
        (stop < bound.val → ProbeVacantAt r ((home.val + stop) % bound.val))) := by
    intro n
    induction n using Nat.strong_induction_on with
    | _ n ih =>
      intro st hn hstle' hloop
      obtain ⟨out, hb, hred⟩ := loop_run_cases
        (fun st1 => table.probe.spill_read_tail_loop.body keyrep r sigma place direct key home bound
          mask wanted st1) st none hloop
      obtain hcases := tail_body_cases_swept keyrep r sigma place direct key home bound mask st wanted
        out hb
      rcases hcases with ⟨hout, hge⟩ | ⟨hout, hlt, i1, hadd, hvac⟩ |
        ⟨hlt, i1, hadd, hout, hcoord, ss, hslot, hm⟩ |
        ⟨hlt, i1, st1, hadd, hout, hstep1, hnotHit⟩
      · refine ⟨st.val, le_refl _, by omega, ?_, ?_⟩
        · intro t ht hlt'; omega
        · intro h; omega
      · have hsweep := swept_val_of_checked home bound mask st i1 exponent hpow hmask hadd
        refine ⟨st.val, le_refl _, by omega, ?_, ?_⟩
        · intro t ht hlt'; omega
        · intro _; rw [← hsweep]; exact hvac
      · rw [hout] at hred
        simp at hred
      · rw [hout] at hred
        have hst1val : st1.val = st.val + 1 := by
          have h := UScalar.add_equiv (ty := .Usize) st (1#usize)
          rw [hstep1] at h
          simpa using h.2.1
        have hdec : bound.val - st1.val < n := by omega
        have hst1le : st1.val ≤ bound.val := by omega
        obtain ⟨stop, hle, hlebound, hmiss, hvacstop⟩ :=
          ih (bound.val - st1.val) hdec st1 rfl hst1le hred
        have hsweep := swept_val_of_checked home bound mask st i1 exponent hpow hmask hadd
        refine ⟨stop, by omega, hlebound, ?_, hvacstop⟩
        intro t ht hltstop
        by_cases ht : t = st.val
        · subst ht
          rw [← hsweep]
          exact hnotHit
        · exact hmiss t (by omega) hltstop
  exact aux (bound.val - step.val) step rfl hstle hrun

/-- **The negative read inverted.**  A reported `ok none` of the generated `table.probe.spill_read`
(`proof/ApeironWord.lean:17486-17524`) comes from exactly one of: an empty region; a rejected
`table.valid_n`; a vacant resident coordinate; or a resident coordinate that is neither vacant nor a
certified hit, with the tail also reporting `ok none`. -/
private theorem spill_read_none_cases {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : K) (n : Usize)
    (hresult : table.probe.spill_read keyrep r sigma address key n = ok none) :
    (r.slots.val.length = 0) ∨
    (table.valid_n n = ok false) ∨
    (∃ home wanted found,
      table.Region.home r address.place = ok home ∧
      table.Region.coord_of R V address.place = ok wanted ∧
      r.coords.val[home.val]? = some found ∧
      ((found = table.COORD_EMPTY) ∨
       (found ≠ table.COORD_EMPTY ∧
        (found ≠ wanted ∨
          (∃ ss, r.slots.val[home.val]? = some ss ∧
            table.matches keyrep ss sigma key address = ok false)) ∧
        table.probe.spill_read_tail keyrep r sigma address key home = ok none))) := by
  unfold table.probe.spill_read at hresult
  by_cases hB : (alloc.vec.Vec.len r.slots != 0#usize) = true
  · rw [if_pos hB] at hresult
    obtain ⟨b, hb, hresult⟩ := bind_ret_inv (table.valid_n n) (fun b =>
      if b then do
        let home ← table.Region.home r address.place
        let wanted ← table.Region.coord_of R V address.place
        let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
          r.coords home
        if found = table.COORD_EMPTY then ok none
        else
          do
            let hit ←
              if found = wanted then
                do
                  let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                    (table.SpillSlot R V)) r.slots home
                  let b1 ← table.matches keyrep ss sigma key address
                  if b1 then ok (some home) else ok none
              else ok none
            let b1 := core.option.Option.is_none hit
            if b1 then table.probe.spill_read_tail keyrep r sigma address key home
            else ok hit
      else ok none) none hresult
    by_cases hbT : b = true
    · rw [if_pos hbT] at hresult
      obtain ⟨home, hhome, hresult⟩ := bind_ret_inv (table.Region.home r address.place)
        (fun home => do
          let wanted ← table.Region.coord_of R V address.place
          let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
            r.coords home
          if found = table.COORD_EMPTY then ok none
          else
            do
              let hit ←
                if found = wanted then
                  do
                    let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                      (table.SpillSlot R V)) r.slots home
                    let b1 ← table.matches keyrep ss sigma key address
                    if b1 then ok (some home) else ok none
                else ok none
              let b1 := core.option.Option.is_none hit
              if b1 then table.probe.spill_read_tail keyrep r sigma address key home
              else ok hit) none hresult
      obtain ⟨wanted, hwanted, hresult⟩ := bind_ret_inv
        (table.Region.coord_of R V address.place) (fun wanted => do
          let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
            r.coords home
          if found = table.COORD_EMPTY then ok none
          else
            do
              let hit ←
                if found = wanted then
                  do
                    let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                      (table.SpillSlot R V)) r.slots home
                    let b1 ← table.matches keyrep ss sigma key address
                    if b1 then ok (some home) else ok none
                else ok none
              let b1 := core.option.Option.is_none hit
              if b1 then table.probe.spill_read_tail keyrep r sigma address key home
              else ok hit) none hresult
      obtain ⟨found, hfound, hresult⟩ := bind_ret_inv
        (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64) r.coords home)
        (fun found =>
          if found = table.COORD_EMPTY then ok none
          else
            do
              let hit ←
                if found = wanted then
                  do
                    let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                      (table.SpillSlot R V)) r.slots home
                    let b1 ← table.matches keyrep ss sigma key address
                    if b1 then ok (some home) else ok none
                else ok none
              let b1 := core.option.Option.is_none hit
              if b1 then table.probe.spill_read_tail keyrep r sigma address key home
              else ok hit) none hresult
      refine Or.inr (Or.inr ⟨home, wanted, found, hhome, hwanted,
        vec_index_readback r.coords home found hfound, ?_⟩)
      by_cases hEmpty : found = table.COORD_EMPTY
      · exact Or.inl hEmpty
      · rw [if_neg hEmpty] at hresult
        obtain ⟨hit, hhit, hrest⟩ := bind_ret_inv
          (if found = wanted then
            do
              let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                (table.SpillSlot R V)) r.slots home
              let b1 ← table.matches keyrep ss sigma key address
              if b1 then ok (some home) else ok none
            else ok none) (fun hit => do
              let b1 := core.option.Option.is_none hit
              if b1 then table.probe.spill_read_tail keyrep r sigma address key home
              else ok hit) none hresult
        by_cases hb1 : core.option.Option.is_none hit = true
        · dsimp only at hrest
          rw [if_pos hb1] at hrest
          refine Or.inr ⟨hEmpty, ?_, hrest⟩
          by_cases hWant : found = wanted
          · right
            rw [if_pos hWant] at hhit
            obtain ⟨ss, hss, hhit2⟩ := bind_ret_inv
              (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                (table.SpillSlot R V)) r.slots home) (fun ss => do
                  let b1 ← table.matches keyrep ss sigma key address
                  if b1 then ok (some home) else ok none) hit hhit
            obtain ⟨m, hm, hhit3⟩ := bind_ret_inv
              (table.matches keyrep ss sigma key address) (fun m =>
                if m then ok (some home) else ok none) hit hhit2
            by_cases hmT : m = true
            · rw [if_pos hmT] at hhit3
              have hsome : hit = some home := (Result.ok_injective hhit3).symm
              rw [hsome] at hb1
              exact (isNone_some home hb1).elim
            · have hmf : m = false := by
                cases m with
                | false => rfl
                | true => exact absurd rfl hmT
              exact ⟨ss, vec_index_readback r.slots home ss hss, by simpa [hmf] using hm⟩
          · exact Or.inl hWant
        · dsimp only at hrest
          rw [if_neg hb1] at hrest
          have hhitNone : hit = none := Result.ok_injective hrest
          exact absurd (by rw [hhitNone]; exact isNone_none) hb1
    · have hbFalse : b = false := by
        cases b with
        | false => rfl
        | true => exact absurd rfl hbT
      exact Or.inr (Or.inl (by rw [hbFalse] at hb; exact hb))
  · left
    have hlen0 : alloc.vec.Vec.len r.slots = 0#usize := by
      by_contra h
      have hne : (alloc.vec.Vec.len r.slots != 0#usize) = true := by
        rw [bne_iff_ne]; exact h
      exact hB hne
    have h := congrArg UScalar.val hlen0
    rw [alloc.vec.Vec.len_val] at h
    simpa using h

/-! ## The source fix `OPS-1`: `Region.home` lands in `[0, bound)`

`Region.home` (`src/table/state.rs`) reduces both of its branches through the probe cycle mask
`bound - 1`, where `bound = Region.bound r = r.slots.len()`.  The returned value lies in `[0, bound)`
for every nonempty region, independent of any relation between the `coords` and `slots` lengths: the
direct branch's `coord_of` value and the spread branch's top-`bits` value are both `&&&` the mask,
and `x &&& m ≤ m < bound`.

This is the fact the absence claims below previously carried as the hypothesis
`home.val < bound.val`.  It is proved here on the regenerated body (`proof/ApeironWord.lean`,
`table.Region.home`), assuming no well-formedness of the region beyond `bound ≠ 0`, so those claims
no longer assume it.  `Apeiron.GrowableHomeBound.home_lt_bound` re-exports this proof for the `OPS-1`
receipt. -/

/-- `x &&& mask ≤ mask` at the `Usize` value level. -/
private theorem and_val_le (x mask : Std.Usize) : (x &&& mask).val ≤ mask.val := by
  rw [UScalar.val_and]
  exact Nat.and_le_right

/-- **`OPS-1`, resolved by the source fix.**  Every value the fixed `table.Region.home` returns is
    strictly below `table.Region.bound`, for every nonempty region and every placement. -/
theorem home_lt_bound {R V : Type} (r : table.Region R V) (place : Std.U64)
    (home bound : Std.Usize)
    (hhome : table.Region.home r place = ok home)
    (hbound : table.Region.bound r = ok bound) (hne : bound.val ≠ 0) :
    home.val < bound.val := by
  have hb : r.slots.val.length = bound.val := by
    have h1 : alloc.vec.Vec.len r.slots = bound := by
      have := Result.ok_injective (by simpa only [table.Region.bound] using hbound)
      simpa using this
    simpa [alloc.vec.Vec.len_val] using congrArg (fun x : Std.Usize => x.val) h1
  have hbne : alloc.vec.Vec.len r.slots ≠ 0#usize := by
    intro h
    apply hne
    rw [← hb]
    have := congrArg (fun x : Std.Usize => x.val) h
    simpa [alloc.vec.Vec.len_val] using this
  have hbpos : 1 ≤ bound.val := by omega
  unfold table.Region.home at hhome
  dsimp only at hhome
  rw [if_neg hbne] at hhome
  obtain ⟨mask, hmask, hhome⟩ := bind_ret_inv (alloc.vec.Vec.len r.slots - 1#usize)
    (fun mask => do
      let c ← table.Region.coord_of R V place
      let len := alloc.vec.Vec.len r.coords
      if len = 0#usize then ok 0#usize
      else
        let i ← lift (UScalar.cast .U64 len)
        if c < i then
          let i1 ← lift (UScalar.cast .Usize c)
          ok (i1 &&& mask)
        else
          let i1 ← lift (core.num.U64.wrapping_mul c table.HOME_SPREAD)
          let i2 ← 64#u32 - r.bits
          let i3 ← i1 >>> i2
          let i4 ← lift (UScalar.cast .Usize i3)
          ok (i4 &&& mask)) home hhome
  have hmaskval : mask.val = bound.val - 1 := by
    have h := @UScalar.sub_equiv .Usize (alloc.vec.Vec.len r.slots) 1#usize
    rw [hmask, Result.match.ok] at h
    simp at h
    omega
  obtain ⟨c, _hc, hhome⟩ := bind_ret_inv (table.Region.coord_of R V place)
    (fun c => do
      let len := alloc.vec.Vec.len r.coords
      if len = 0#usize then ok 0#usize
      else
        let i ← lift (UScalar.cast .U64 len)
        if c < i then
          let i1 ← lift (UScalar.cast .Usize c)
          ok (i1 &&& mask)
        else
          let i1 ← lift (core.num.U64.wrapping_mul c table.HOME_SPREAD)
          let i2 ← 64#u32 - r.bits
          let i3 ← i1 >>> i2
          let i4 ← lift (UScalar.cast .Usize i3)
          ok (i4 &&& mask)) home hhome
  by_cases hlen : alloc.vec.Vec.len r.coords = 0#usize
  · dsimp only at hhome
    rw [if_pos hlen] at hhome
    have hz : home = 0#usize := (Result.ok_injective hhome).symm
    rw [hz]
    exact Nat.pos_of_ne_zero hne
  · dsimp only at hhome
    rw [if_neg hlen] at hhome
    obtain ⟨i, _hi, hhome⟩ := bind_ret_inv
      (lift (UScalar.cast .U64 (alloc.vec.Vec.len r.coords)))
      (fun i => if c < i then
        (do let i1 ← lift (UScalar.cast .Usize c); ok (i1 &&& mask))
      else
        (do
          let i1 ← lift (core.num.U64.wrapping_mul c table.HOME_SPREAD)
          let i2 ← 64#u32 - r.bits
          let i3 ← i1 >>> i2
          let i4 ← lift (UScalar.cast .Usize i3)
          ok (i4 &&& mask))) home hhome
    by_cases hci : c < i
    · rw [if_pos hci] at hhome
      obtain ⟨i1, _hi1, hhome⟩ := bind_ret_inv (lift (UScalar.cast .Usize c))
        (fun i1 => ok (i1 &&& mask)) home hhome
      have he : home = i1 &&& mask := (Result.ok_injective hhome).symm
      rw [he]
      exact lt_of_le_of_lt (and_val_le i1 mask) (by omega)
    · rw [if_neg hci] at hhome
      obtain ⟨w, _hw, hhome⟩ := bind_ret_inv
        (lift (core.num.U64.wrapping_mul c table.HOME_SPREAD))
        (fun w => do
          let i2 ← 64#u32 - r.bits
          let i3 ← w >>> i2
          let i4 ← lift (UScalar.cast .Usize i3)
          ok (i4 &&& mask)) home hhome
      obtain ⟨sh, _hsh, hhome⟩ := bind_ret_inv (64#u32 - r.bits)
        (fun sh => do
          let i3 ← w >>> sh
          let i4 ← lift (UScalar.cast .Usize i3)
          ok (i4 &&& mask)) home hhome
      obtain ⟨z, _hz, hhome⟩ := bind_ret_inv (w >>> sh)
        (fun z => do
          let i4 ← lift (UScalar.cast .Usize z)
          ok (i4 &&& mask)) home hhome
      obtain ⟨i4, _hi4, hhome⟩ := bind_ret_inv (lift (UScalar.cast .Usize z))
        (fun i4 => ok (i4 &&& mask)) home hhome
      have he : home = i4 &&& mask := (Result.ok_injective hhome).symm
      rw [he]
      exact lt_of_le_of_lt (and_val_le i4 mask) (by omega)

/-- **The first-fit gap discipline.**  In the probe order `t ↦ (home + t) % bound`, once a swept
slot is vacant, no later swept slot is a coord-hit.  This is the property the insert path owes the
read: a live representative is never placed past a vacancy it would have occupied.  It is stated
over the read's own words (`ProbeVacantAt`, `ProbeHitAt`) and is the sole insert-side premise the
`None` half consumes. -/
def GapDiscipline {K H R V : Type} (keyrep : table.keyrep.KeyRep R K H) (r : table.Region R V)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : K) (wanted : U64)
    (home bound : Usize) : Prop :=
  ∀ t : Nat, t < bound.val →
    ProbeVacantAt r ((home.val + t) % bound.val) →
    ∀ t' : Nat, t ≤ t' → t' < bound.val →
      ¬ ProbeHitAt keyrep r sigma address key wanted ((home.val + t') % bound.val)

/-- **The `None` half, conditional on the gap discipline.**  On the generated
`table.probe.spill_read` (`proof/ApeironWord.lean:17486-17524`), for every face `sigma` and every
key, if the read reports `ok none` and the first-fit gap discipline holds at the routed
`home`/`wanted`, then no slot of the region is a coord-hit: there is no index whose coordinate word
is the query's and whose payload passes the extracted matcher.  The only remaining insert-side
premise is the gap discipline itself; the region's home-in-range obligation is discharged by
`home_lt_bound` from the power-of-two capacity, and the read's negative content is unconditional. -/
theorem spill_read_none_no_probe_hit_of_gap {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : K) (n bound home : Usize) (wanted : U64)
    (exponent : Nat)
    (hvalid : table.valid_n n = ok true)
    (hbound : table.Region.bound r = ok bound)
    (hhome : table.Region.home r address.place = ok home)
    (hwanted : table.Region.coord_of R V address.place = ok wanted)
    (hpow : bound.val = 2 ^ exponent)
    (hgap : GapDiscipline keyrep r sigma address key wanted home bound)
    (hresult : table.probe.spill_read keyrep r sigma address key n = ok none) :
    ∀ j : Nat, j < bound.val → ¬ ProbeHitAt keyrep r sigma address key wanted j := by
  have hhomeLt : home.val < bound.val :=
    home_lt_bound r address.place home bound hhome hbound (by rw [hpow]; exact pow_ne_zero _ (by norm_num))
  have hboundVal : bound.val = r.slots.val.length := by
    have h := congrArg UScalar.val (Result.ok_injective hbound)
    rw [alloc.vec.Vec.len_val] at h
    exact h.symm
  obtain hcases := spill_read_none_cases keyrep r sigma address key n hresult
  rcases hcases with hempty | hvalidFalse | ⟨home', wanted', found, hhome', hwanted', hfound, hsub⟩
  · intro j hj
    omega
  · exact Bool.noConfusion (Result.ok_injective (hvalid.symm.trans hvalidFalse))
  · have hhomeEq : home' = home := Result.ok_injective (hhome'.symm.trans hhome)
    have hwantedEq : wanted' = wanted := Result.ok_injective (hwanted'.symm.trans hwanted)
    subst home'
    subst wanted'
    rcases hsub with hfoundEmpty | ⟨hfoundNe, hnotResident, htail⟩
    · -- resident vacancy: the gap discipline at step 0
      have hvac0 : ProbeVacantAt r ((home.val + 0) % bound.val) := by
        rw [Nat.add_zero, Nat.mod_eq_of_lt hhomeLt]
        rw [hfoundEmpty] at hfound
        exact hfound
      intro j hj
      obtain ⟨step, hstepLt, hstepEq⟩ := Apeiron.GrowableScan.spill_cycle_covers
        (home := home.val) (bound := bound.val) (j := j) (k := exponent) hpow hj
      have hjEq : j = (home.val + step) % bound.val := by
        rw [← hstepEq, hpow, Nat.and_two_pow_sub_one_eq_mod]
      intro hhit
      rw [hjEq] at hhit
      exact (hgap 0 (by omega) hvac0 step (by omega) hstepLt) hhit
    · -- resident non-hit and the tail also misses
      unfold table.probe.spill_read_tail at htail
      obtain ⟨boundT, hboundT, htail⟩ := bind_ret_inv (table.Region.bound r) (fun boundT => do
          let mask ← boundT - 1#usize
          let wantedT ← table.Region.coord_of R V address.place
          table.probe.spill_read_tail_loop keyrep r sigma address.place address.direct_word key home
            boundT mask wantedT 1#usize) none htail
      obtain ⟨mask, hmaskRun, htail⟩ := bind_ret_inv (boundT - 1#usize) (fun mask => do
          let wantedT ← table.Region.coord_of R V address.place
          table.probe.spill_read_tail_loop keyrep r sigma address.place address.direct_word key home
            boundT mask wantedT 1#usize) none htail
      obtain ⟨wantedT, hwantedT, htail⟩ := bind_ret_inv (table.Region.coord_of R V address.place)
        (fun wantedT => table.probe.spill_read_tail_loop keyrep r sigma address.place
          address.direct_word key home boundT mask wantedT 1#usize) none htail
      have hboundTEq : boundT = bound := Result.ok_injective (hboundT.symm.trans hbound)
      have hwantedTEq : wantedT = wanted := Result.ok_injective (hwantedT.symm.trans hwanted)
      subst boundT
      subst wantedT
      have h1le : (1#usize).val ≤ bound.val := by
        show 1 ≤ bound.val
        omega
      have hmaskVal : mask.val = bound.val - 1 := by
        obtain ⟨z, hzrun, hzval⟩ := WP.spec_imp_exists
          (UScalar.sub_spec (ty := .Usize) (x := bound) (y := 1#usize) (by simpa using h1le))
        have hz : z = mask := Result.ok_injective (hzrun.symm.trans hmaskRun)
        rw [← hz]
        simpa using hzval.1
      have htailAbs := tail_loop_none_absent keyrep r sigma address.place address.direct_word key home
        bound mask 1#usize wanted exponent hpow hmaskVal h1le htail
      obtain ⟨stop, hstop1, hstopBound, hmiss, hvacstop⟩ := htailAbs
      have hresNot : ¬ ProbeHitAt keyrep r sigma address key wanted home.val := by
        intro hhit
        unfold ProbeHitAt at hhit
        obtain ⟨hco, ss, hslot, hmatch⟩ := hhit
        have hfw : found = wanted := Option.some.inj (hfound.symm.trans hco)
        rcases hnotResident with hne | ⟨ss2, hslot2, hmatch2⟩
        · exact hne hfw
        · have hssEq : ss = ss2 := Option.some.inj (hslot.symm.trans hslot2)
          rw [hssEq] at hmatch
          rw [hmatch2] at hmatch
          exact Bool.noConfusion (Result.ok_injective hmatch)
      intro j hj
      obtain ⟨step, hstepLt, hstepEq⟩ := Apeiron.GrowableScan.spill_cycle_covers
        (home := home.val) (bound := bound.val) (j := j) (k := exponent) hpow hj
      have hjEq : j = (home.val + step) % bound.val := by
        rw [← hstepEq, hpow, Nat.and_two_pow_sub_one_eq_mod]
      intro hhit
      rw [hjEq] at hhit
      by_cases hstop : stop ≤ step
      · have hstopLt : stop < bound.val := by omega
        exact (hgap stop hstopLt (hvacstop hstopLt) step hstop hstepLt) hhit
      · have hstepLtStop : step < stop := by omega
        by_cases hstep0 : step = 0
        · subst hstep0
          rw [Nat.add_zero, Nat.mod_eq_of_lt hhomeLt] at hhit
          exact hresNot hhit
        · exact (hmiss step (Nat.pos_of_ne_zero hstep0) hstepLtStop) hhit

/-- **The `None` half for the carried-key face, under coordinate soundness.**  Composing
`spill_read_none_no_probe_hit_of_gap` with the proved `StoredLiveCoordAddress` invariant: a live
slot that holds the query's canonical logical key has coordinate word `Region.coord_of address.place`
(its own routed coordinate), so it is a coord-hit and the read could not have reported `ok none`.
Hence `ok none` certifies that no live slot of the routed region holds the key.  The gap discipline
is the only remaining insert-side premise; the in-range `home` is discharged by `home_lt_bound`. -/
theorem stored_spill_read_none_no_live_key
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveCoordAddress state)
    (sigma : cell.Sigma) (key : I64) (n bound home : Usize) (wanted : U64) (exponent : Nat)
    (shard : Usize) (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (_hshard : table.shard address = ok shard)
    (hvalid : table.valid_n n = ok true)
    (hbound : table.Region.bound region = ok bound)
    (hhome : table.Region.home region address.place = ok home)
    (hwanted : table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place = ok wanted)
    (hpow : bound.val = 2 ^ exponent)
    (hgap : GapDiscipline Apeiron.GrowableIdentity.storedKeyRep region sigma address key wanted
      home bound)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key n
      = ok none) :
    ∀ (index : Usize) (ss : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64),
      region.slots.val[index.val]? = some ss → ss.tag ≠ 0#u8 →
      Apeiron.GrowableIdentity.logicalStored ss.sigma ss.held ≠ (sigma, key) := by
  have hprobe := spill_read_none_no_probe_hit_of_gap Apeiron.GrowableIdentity.storedKeyRep region
    sigma address key n bound home wanted exponent hvalid hbound hhome hwanted hpow hgap
    hresult
  have hboundVal : bound.val = region.slots.val.length := by
    have h := congrArg UScalar.val (Result.ok_injective hbound)
    rw [alloc.vec.Vec.len_val] at h
    exact h.symm
  intro index ss hslot htag hlogical
  obtain ⟨coord, addr, hcoord, _hcNe, haddr', _hshard', hcf⟩ :=
    hstate shard index region ss.tag ss.sigma ss.held ss.value hregion hslot htag
  have hssigma : ss.sigma = sigma := by
    simpa [Apeiron.GrowableIdentity.logicalStored] using congrArg Prod.fst hlogical
  have hskey : ss.held.key = key := by
    simpa [Apeiron.GrowableIdentity.logicalStored, Apeiron.GrowableIdentity.decodeStored]
      using congrArg Prod.snd hlogical
  rw [hssigma, hskey] at haddr'
  have haddrEq : addr = address := Result.ok_injective (haddr'.symm.trans haddress)
  have hcoordWanted : region.coords.val[index.val]? = some wanted := by
    rw [haddrEq] at hcf
    have hcw : coord = wanted := Result.ok_injective (hcf.symm.trans hwanted)
    rw [hcw] at hcoord
    exact hcoord
  have hmatch : table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address
      = ok true :=
    (stored_matches_slot_iff ss.tag ss.sigma ss.held ss.value sigma key address).mpr hlogical
  have hlt : index.val < bound.val := by
    rw [hboundVal]
    exact (List.getElem?_eq_some_iff.mp hslot).1
  exact (hprobe index.val hlt) ⟨hcoordWanted, ss, hslot, hmatch⟩

/-! ### Whole-capacity and whole-shard absence, restated

The W3 redesign retired the whole-capacity absence theorem `spill_read_none_implies_no_full_match`
(and its whole-shard corollary) because `Region.home` could land at or above the capacity.  With the
source fix, `home_lt_bound` closes that gap, so the two forms are restated here on the current
coordinate schema: a `none` read under the first-fit gap discipline excludes every live slot of the
routed region — and, by coordinate-sound routing, of every shard — from holding the query under full
`(Sigma, key)` equality.  They are the `StoredLiveCoordAddress` analogues of the retired
`StoredLiveTagAddress` theorems and carry no `home < bound` hypothesis. -/

/-- **Whole-capacity absence, restated under its historical name.**  The region-level no-live-match
form of the retired `spill_read_none_implies_no_full_match`: a generated `ok none` read at the
routed region, under `StoredLiveCoordAddress` and the gap discipline, certifies that no live slot of
that region holds the query under full `(Sigma, key)` equality. -/
theorem spill_read_none_implies_no_full_match
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveCoordAddress state)
    (sigma : cell.Sigma) (key : I64) (n bound home : Usize) (wanted : U64) (exponent : Nat)
    (shard : Usize) (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (hshard : table.shard address = ok shard)
    (hvalid : table.valid_n n = ok true)
    (hbound : table.Region.bound region = ok bound)
    (hhome : table.Region.home region address.place = ok home)
    (hwanted : table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place = ok wanted)
    (hpow : bound.val = 2 ^ exponent)
    (hgap : GapDiscipline Apeiron.GrowableIdentity.storedKeyRep region sigma address key wanted
      home bound)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key n
      = ok none) :
    ∀ (index : Usize) (ss : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64),
      region.slots.val[index.val]? = some ss → ss.tag ≠ 0#u8 →
      Apeiron.GrowableIdentity.logicalStored ss.sigma ss.held ≠ (sigma, key) :=
  stored_spill_read_none_no_live_key state hstate sigma key n bound home wanted exponent shard
    region hregion address haddress hshard hvalid hbound hhome hwanted hpow hgap hresult

/-- **Whole-shard absence, restated under its historical name.**  The whole-shard corollary
`spill_read_none_implies_no_shard_full_match`: the same `ok none` read excludes a live full match at
*every* shard, since `StoredLiveCoordAddress` routes any live slot's stored `(Sigma, key)` to the
query's own address, hence to the routed shard. -/
theorem spill_read_none_implies_no_shard_full_match
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveCoordAddress state)
    (sigma : cell.Sigma) (key : I64) (n bound home : Usize) (wanted : U64) (exponent : Nat)
    (shard : Usize) (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (hshard : table.shard address = ok shard)
    (hvalid : table.valid_n n = ok true)
    (hbound : table.Region.bound region = ok bound)
    (hhome : table.Region.home region address.place = ok home)
    (hwanted : table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place = ok wanted)
    (hpow : bound.val = 2 ^ exponent)
    (hgap : GapDiscipline Apeiron.GrowableIdentity.storedKeyRep region sigma address key wanted
      home bound)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key n
      = ok none) :
    ∀ (shard' : Usize) (region' : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
      (index : Usize) (ss : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64),
      state.shards.val[shard'.val]? = some region' →
      region'.slots.val[index.val]? = some ss → ss.tag ≠ 0#u8 →
      Apeiron.GrowableIdentity.logicalStored ss.sigma ss.held ≠ (sigma, key) := by
  have habsent := spill_read_none_implies_no_full_match state hstate sigma key n bound home wanted
    exponent shard region hregion address haddress hshard hvalid hbound hhome hwanted hpow hgap
    hresult
  intro shard' region' index ss hregion' hslot htag hfull
  obtain ⟨_coord, addr', _hcoord', _hcNe, haddr', hshard', _hcf⟩ :=
    hstate shard' index region' ss.tag ss.sigma ss.held ss.value hregion' hslot htag
  have hssigma : ss.sigma = sigma := by
    simpa [Apeiron.GrowableIdentity.logicalStored] using congrArg Prod.fst hfull
  have hskey : ss.held.key = key := by
    simpa [Apeiron.GrowableIdentity.logicalStored, Apeiron.GrowableIdentity.decodeStored]
      using congrArg Prod.snd hfull
  rw [hssigma, hskey] at haddr'
  have haddrEq : addr' = address := Result.ok_injective (haddr'.symm.trans haddress)
  rw [haddrEq] at hshard'
  have hshardEq : shard' = shard := Result.ok_injective (hshard'.symm.trans hshard)
  subst shard'
  have hregionEq : region' = region := Option.some.inj (hregion'.symm.trans hregion)
  subst region'
  exact habsent index ss hslot htag hfull

/-- Converse decomposition of a spill `entryAt` under the carried-key decode: a live spill entry
names its shard's region, its slot, and the slot's own stored fields. -/
private theorem entryAt_spill_stored_inv
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (sigma : cell.Sigma) (key : I64) (shard index : Usize) (value : I64)
    (h : Apeiron.GrowableSearch.entryAt state Apeiron.GrowableSearch.decodeStoredOption
      (.spill shard index) = some ((sigma, key), value)) :
    ∃ (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
      (slot : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64),
      state.shards.val[shard.val]? = some region ∧
      region.slots.val[index.val]? = some slot ∧ slot.tag ≠ 0#u8 ∧
      slot.sigma = sigma ∧ slot.held.key = key ∧ slot.value = value := by
  simp only [Apeiron.GrowableSearch.entryAt, Apeiron.GrowableSearch.decodeStoredOption] at h
  split at h
  · rename_i _x region hregion
    split at h
    · rename_i _x2 slot hslot
      by_cases htag : slot.tag = 0#u8
      · rw [if_pos htag] at h
        cases h
      · rw [if_neg htag] at h
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        exact ⟨region, slot, hregion, hslot, htag, h.1.1, h.1.2, h.2⟩
    · cases h
  · cases h

/-- **Composition to the canonical `Apeiron.Table.get`.**  Under the whole-table representation
relation, coordinate soundness, and the same read hypotheses as
`spill_read_none_no_probe_hit_of_gap`, a reported `ok none` at the query's routed region — with no
live flat representative of the query — denotes the canonical absent entry, so
`Apeiron.Table.get logical (sigma, key)` is the table default.  This is the `None` half composed with
the canonical object. -/
theorem word_stored_spill_read_none_get_default
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (direct : Apeiron.GrowableRepresentation.LogicalKey → Option Apeiron.GrowableSearch.Location)
    (live : Apeiron.GrowableRepresentation.LogicalKey → Prop)
    (logical : Apeiron.Table.HMap Apeiron.GrowableRepresentation.LogicalKey I64)
    (hreach : Apeiron.GrowableRepresentation.RepR state Apeiron.GrowableSearch.decodeStoredOption
      direct live logical)
    (hcoords : StoredLiveCoordAddress state)
    (sigma : cell.Sigma) (key : I64) (n bound home : Usize) (wanted : U64) (exponent : Nat)
    (shard : Usize) (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (hshard : table.shard address = ok shard)
    (hvalid : table.valid_n n = ok true)
    (hbound : table.Region.bound region = ok bound)
    (hhome : table.Region.home region address.place = ok home)
    (hwanted : table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place = ok wanted)
    (hpow : bound.val = 2 ^ exponent)
    (hgap : GapDiscipline Apeiron.GrowableIdentity.storedKeyRep region sigma address key wanted
      home bound)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address key n
      = ok none)
    (hflat : ∀ index value, Apeiron.GrowableSearch.entryAt state
      Apeiron.GrowableSearch.decodeStoredOption (.flat index) ≠ some ((sigma, key), value)) :
    Apeiron.Table.get logical (sigma, key) = state.default := by
  have habsent : Apeiron.GrowableSearch.Absent state Apeiron.GrowableSearch.decodeStoredOption
      (sigma, key) := by
    intro location value h
    cases location with
    | flat index => exact hflat index value h
    | spill shard' index =>
        obtain ⟨region', slot, hregion', hslot', htag, hsigma, hkey, _hval⟩ :=
          entryAt_spill_stored_inv state sigma key shard' index value h
        obtain ⟨addr, haddr, hshr⟩ := hreach.address (.spill shard' index) (sigma, key) value h
        have haddrEq : addr = address := Result.ok_injective (haddr.symm.trans haddress)
        rw [haddrEq] at hshr
        have hshEq : shard' = shard := Result.ok_injective (hshr.symm.trans hshard)
        subst shard'
        have hregionEq : region' = region := Option.some.inj (hregion'.symm.trans hregion)
        subst region'
        have hlogical : Apeiron.GrowableIdentity.logicalStored slot.sigma slot.held = (sigma, key) := by
          simp only [Apeiron.GrowableIdentity.logicalStored, Apeiron.GrowableIdentity.decodeStored,
            hsigma, hkey]
        exact (stored_spill_read_none_no_live_key state hcoords sigma key n bound home wanted
          exponent shard region hregion address haddress hshard hvalid hbound hhome hwanted hpow
          hgap hresult index slot hslot' htag) hlogical
  have hnotlive : ¬ live (sigma, key) := by
    intro hl
    obtain ⟨location, value, hentry⟩ := (hreach.liveSet (sigma, key)).mp hl
    exact habsent location value hentry
  exact hreach.observations.2 (sigma, key) hnotlive

/-! ## The first-fit gap discipline of the insert side

The read side's `None` half (`spill_read_none_no_probe_hit_of_gap`) consumes one insert-side
premise, `GapDiscipline`.  This section proves the placement fact the vacant-spill commit owes it,
over the real generated bodies:

* `table.probe.spill_locate` (`proof/ApeironWord.lean:20886-20905`) returns `SpillSite.Vacant index`
  only at the *first* vacant slot of its probe order: the generated loop advances past every slot
  whose coordinate word is not the sentinel, so `index` is reached at some step `iv < bound` and
  every earlier swept slot is occupied (`VacantPlacement`);
* `table.Region.publish_live` (`proof/ApeironWord.lean:20801-20832`) writes the live coordinate
  `Region.coord_of place` at `index` and frames every other coordinate and slot, so the one
  coordinate it makes live is at `index` and every earlier probe slot stays occupied.

`gap_discipline_of_vacant_frame` composes them: the pre-state gap discipline together with the
first-fit placement yields the post-state gap discipline.  The placement is unconditional; the
pre-state discipline is the step's induction hypothesis (base: the empty region, where no slot is a
hit).  `spill_read_none_no_probe_hit_of_publish_vacant` feeds the result back into the read theorem,
so the read's `hgap` premise is discharged from the executed insert bodies.

One premise remains named, not assumed: the pre-state `GapDiscipline` — the induction hypothesis.
The region's home-in-range obligation (`home.val < bound.val`) that the read composition consumes is
discharged from the power-of-two capacity by `home_lt_bound`, so it is no longer a premise of this
face. -/

/-- The first-fit placement fact of one generated `spill_locate` `Vacant` return: `index` is the
probe slot at some step `iv < bound`, it is vacant in the state, and every earlier probe slot is
occupied, i.e. carries a coordinate word other than the sentinel.  This is the property the read's
`GapDiscipline` consumes: a live representative is never placed past a vacancy. -/
def VacantPlacement {R V : Type} (r : table.Region R V) (home bound : Usize) (index : Usize) :
    Prop :=
  ∃ iv : Nat, iv < bound.val ∧ (home.val + iv) % bound.val = index.val ∧
    r.coords.val[index.val]? = some table.COORD_EMPTY ∧
    ∀ t : Nat, t < iv → r.coords.val[((home.val + t) % bound.val)]? ≠ some table.COORD_EMPTY

/-- `SpillSite.Grow` and `SpillSite.Vacant` are distinct constructors, so a `Grow` terminator never
presents as a vacant stop. -/
private theorem spill_site_grow_ne_vacant {i : Usize}
    (h : table.SpillSite.Grow = table.SpillSite.Vacant i) : False := by
  cases h

/-- The four outcomes of one generated `spill_locate` loop step, read from the body
(`proof/ApeironWord.lean:20837-20869`): the step guard fails (`Grow`); the swept coordinate is
vacant (`Vacant`, with the coordinate readback); the swept coordinate is the query's and the matcher
succeeds (`Found`); or the step advances with the swept slot occupied.  This is the locate analogue
of `tail_body_cases_swept`. -/
private theorem spill_locate_body_cases {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (bound home : Usize) (wanted : U64) (step : Usize)
    (out : ControlFlow Usize table.SpillSite)
    (hbody : table.probe.spill_locate_loop.body keyrep r sigma place direct key bound home wanted
      step = ok out) :
    (out = ControlFlow.done table.SpillSite.Grow ∧ bound.val ≤ step.val) ∨
    (step.val < bound.val ∧ ∃ (i1 i2 : Usize),
      home + step = ok i1 ∧ bound - 1#usize = ok i2 ∧
      out = ControlFlow.done (table.SpillSite.Vacant (i1 &&& i2)) ∧
      r.coords.val[(i1 &&& i2).val]? = some table.COORD_EMPTY) ∨
    (step.val < bound.val ∧ ∃ (i1 i2 : Usize) (ss : table.SpillSlot R V),
      home + step = ok i1 ∧ bound - 1#usize = ok i2 ∧
      out = ControlFlow.done (table.SpillSite.Found (i1 &&& i2)) ∧
      r.coords.val[(i1 &&& i2).val]? = some wanted ∧
      r.slots.val[(i1 &&& i2).val]? = some ss ∧
      table.matches keyrep ss sigma key { place := place, direct_word := direct } = ok true) ∨
    (step.val < bound.val ∧ ∃ (i1 i2 st1 : Usize),
      home + step = ok i1 ∧ bound - 1#usize = ok i2 ∧
      out = ControlFlow.cont st1 ∧ step + 1#usize = ok st1 ∧
      r.coords.val[(i1 &&& i2).val]? ≠ some table.COORD_EMPTY) := by
  unfold table.probe.spill_locate_loop.body at hbody
  by_cases hlt : step < bound
  · rw [if_pos hlt] at hbody
    obtain ⟨i1, hadd, hbody⟩ := bind_ret_inv (home + step) (fun i1 => do
        let i2 ← bound - 1#usize
        let slot0 ← lift (i1 &&& i2)
        let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
          r.coords slot0
        if found = table.COORD_EMPTY then ok (ControlFlow.done (table.SpillSite.Vacant slot0))
        else
          if found = wanted then
            do
              let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                (table.SpillSlot R V)) r.slots slot0
              let b ← table.matches keyrep ss sigma key { place := place, direct_word := direct }
              if b then ok (ControlFlow.done (table.SpillSite.Found slot0))
              else
                do
                  let step1 ← step + 1#usize
                  ok (ControlFlow.cont step1)
          else
            do
              let step1 ← step + 1#usize
              ok (ControlFlow.cont step1)) out hbody
    obtain ⟨i2, hsub, hbody⟩ := bind_ret_inv (bound - 1#usize) (fun i2 => do
        let slot0 ← lift (i1 &&& i2)
        let found ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
          r.coords slot0
        if found = table.COORD_EMPTY then ok (ControlFlow.done (table.SpillSite.Vacant slot0))
        else
          if found = wanted then
            do
              let ss ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                (table.SpillSlot R V)) r.slots slot0
              let b ← table.matches keyrep ss sigma key { place := place, direct_word := direct }
              if b then ok (ControlFlow.done (table.SpillSite.Found slot0))
              else
                do
                  let step1 ← step + 1#usize
                  ok (ControlFlow.cont step1)
          else
            do
              let step1 ← step + 1#usize
              ok (ControlFlow.cont step1)) out hbody
    rw [show (lift (i1 &&& i2) : Result Usize) = ok (i1 &&& i2) from rfl,
      bind_tc_ok] at hbody
    cases hidx : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
        r.coords (i1 &&& i2) with
    | vis eff k =>
        rw [hidx, bind_tc_vis] at hbody
        exact (vis_not_ok hbody).elim
    | div =>
        rw [hidx, bind_tc_div] at hbody
        exact (div_not_ok hbody).elim
    | ret coord =>
        rw [hidx, bind_tc_ok] at hbody
        have hcoord := vec_index_readback r.coords (i1 &&& i2) coord hidx
        by_cases hzero : coord = table.COORD_EMPTY
        · rw [if_pos hzero] at hbody
          exact Or.inr (Or.inl ⟨hlt, i1, i2, hadd, hsub,
            (Result.ok_injective hbody).symm, hcoord.trans (congrArg some hzero)⟩)
        · rw [if_neg hzero] at hbody
          by_cases hWant : coord = wanted
          · rw [if_pos hWant] at hbody
            cases hss : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
                (table.SpillSlot R V)) r.slots (i1 &&& i2) with
            | vis eff k =>
                rw [hss, bind_tc_vis] at hbody
                exact (vis_not_ok hbody).elim
            | div =>
                rw [hss, bind_tc_div] at hbody
                exact (div_not_ok hbody).elim
            | ret ss =>
                rw [hss, bind_tc_ok] at hbody
                cases hmt : table.matches keyrep ss sigma key
                    { place := place, direct_word := direct } with
                | vis eff k =>
                    rw [hmt, bind_tc_vis] at hbody
                    exact (vis_not_ok hbody).elim
                | div =>
                    rw [hmt, bind_tc_div] at hbody
                    exact (div_not_ok hbody).elim
                | ret b =>
                    rw [hmt, bind_tc_ok] at hbody
                    by_cases hb : b = true
                    · rw [if_pos hb] at hbody
                      exact Or.inr (Or.inr (Or.inl ⟨hlt, i1, i2, ss, hadd, hsub,
                        (Result.ok_injective hbody).symm,
                        hcoord.trans (congrArg some hWant),
                        vec_index_readback r.slots (i1 &&& i2) ss hss,
                        hmt.trans (congrArg ok hb)⟩))
                    · rw [if_neg hb] at hbody
                      obtain ⟨st1, hst1, hfin⟩ := bind_ret_inv (step + 1#usize)
                        (fun s1 => ok (ControlFlow.cont (s1 : Usize) :
                          ControlFlow Usize table.SpillSite)) out hbody
                      exact Or.inr (Or.inr (Or.inr ⟨hlt, i1, i2, st1, hadd, hsub,
                        (Result.ok_injective hfin).symm, hst1,
                        fun hcontra => hzero (Option.some.inj (hcoord.symm.trans hcontra))⟩))
          · rw [if_neg hWant] at hbody
            obtain ⟨st1, hst1, hfin⟩ := bind_ret_inv (step + 1#usize)
              (fun s1 => ok (ControlFlow.cont (s1 : Usize) :
                ControlFlow Usize table.SpillSite)) out hbody
            exact Or.inr (Or.inr (Or.inr ⟨hlt, i1, i2, st1, hadd, hsub,
              (Result.ok_injective hfin).symm, hst1,
              fun hcontra => hzero (Option.some.inj (hcoord.symm.trans hcontra))⟩))
  · rw [if_neg hlt] at hbody
    refine Or.inl ⟨(Result.ok_injective hbody).symm, ?_⟩
    exact Nat.le_of_not_gt (fun h => hlt ((UScalar.lt_equiv step bound).mpr h))

/-- **The first-fit scan of the ported locate loop.**  If the generated `spill_locate` loop run from
`0` reports `Vacant index`, then `index` is the swept slot at some step `iv < bound`, it is vacant,
and every earlier swept slot is occupied.  Derived from the executed loop body alone; the only
arithmetic input is the power-of-two extent, which turns `(home + step) &&& (bound - 1)` into
`(home + step) % bound`. -/
private theorem spill_locate_loop_vacant_scan {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (bound home : Usize) (wanted : U64)
    (exponent : Nat) (hpow : bound.val = 2 ^ exponent) (index : Usize)
    (hrun : table.probe.spill_locate_loop keyrep r sigma place direct key bound home wanted 0#usize
      = ok (table.SpillSite.Vacant index)) :
    VacantPlacement r home bound index := by
  have aux : ∀ n : Nat, (∀ st : Usize, bound.val - st.val = n → st.val ≤ bound.val →
      table.probe.spill_locate_loop keyrep r sigma place direct key bound home wanted st
        = ok (table.SpillSite.Vacant index) →
      ∃ iv : Nat, st.val ≤ iv ∧ iv < bound.val ∧
        (home.val + iv) % bound.val = index.val ∧
        r.coords.val[index.val]? = some table.COORD_EMPTY ∧
        ∀ t : Nat, st.val ≤ t → t < iv →
          r.coords.val[((home.val + t) % bound.val)]? ≠ some table.COORD_EMPTY) := by
    intro n
    induction n using Nat.strong_induction_on with
    | _ n ih =>
      intro st hn hstle hloop
      obtain ⟨out, hb, hred⟩ := loop_run_cases
        (fun st1 => table.probe.spill_locate_loop.body keyrep r sigma place direct key bound home
          wanted st1) st (table.SpillSite.Vacant index) hloop
      obtain hcases :=
        spill_locate_body_cases keyrep r sigma place direct key bound home wanted st out hb
      rcases hcases with ⟨hout, _hge⟩ | ⟨hlt, i1, i2, hadd, hsub, hout, hvac⟩ |
        ⟨hlt, i1, i2, ss, hadd, hsub, hout, _hcoordW, _hslot, _hmatch⟩ |
        ⟨hlt, i1, i2, st1, hadd, hsub, hout, hstep1, hne⟩
      · rw [hout] at hred
        cases hred
      · rw [hout] at hred
        have hslotEq : UScalar.and i1 i2 = index := by
          have hraw : i1 &&& i2 = index := by injection hred
          rw [← hraw]
          rfl
        have h1le : (1#usize).val ≤ bound.val := by
          show 1 ≤ bound.val
          omega
        obtain ⟨z, hzrun, hzval⟩ := WP.spec_imp_exists
          (UScalar.sub_spec (ty := .Usize) (x := bound) (y := 1#usize) (by simpa using h1le))
        have hz : z = i2 := Result.ok_injective (hzrun.symm.trans hsub)
        have hmask : i2.val = bound.val - 1 := by
          rw [← hz]
          simpa using hzval.1
        have hsweep := swept_val_of_checked home bound i2 st i1 exponent hpow hmask hadd
        have hidxEq : (home.val + st.val) % bound.val = index.val := by
          rw [← hsweep, hslotEq]
        refine ⟨st.val, le_refl _, hlt, hidxEq, ?_, ?_⟩
        · rw [← hslotEq]
          exact hvac
        · intro t _ ht
          omega
      · rw [hout] at hred
        cases hred
      · rw [hout] at hred
        have hst1val : st1.val = st.val + 1 := by
          have h := UScalar.add_equiv (ty := .Usize) st (1#usize)
          rw [hstep1] at h
          simpa using h.2.1
        have hdec : bound.val - st1.val < n := by omega
        have hst1le : st1.val ≤ bound.val := by omega
        obtain ⟨iv, hst1iv, hivlt, hivEq, hvac, hocc⟩ :=
          ih (bound.val - st1.val) hdec st1 rfl hst1le hred
        have h1le : (1#usize).val ≤ bound.val := by
          show 1 ≤ bound.val
          omega
        obtain ⟨z, hzrun, hzval⟩ := WP.spec_imp_exists
          (UScalar.sub_spec (ty := .Usize) (x := bound) (y := 1#usize) (by simpa using h1le))
        have hz : z = i2 := Result.ok_injective (hzrun.symm.trans hsub)
        have hmask : i2.val = bound.val - 1 := by
          rw [← hz]
          simpa using hzval.1
        have hsweep := swept_val_of_checked home bound i2 st i1 exponent hpow hmask hadd
        refine ⟨iv, by omega, hivlt, hivEq, hvac, ?_⟩
        intro t ht hltiv
        by_cases hts : t = st.val
        · subst hts
          rw [← hsweep]
          exact hne
        · exact hocc t (by omega) hltiv
  obtain ⟨iv, _hst0, hivlt, hivEq, hvac, hocc⟩ :=
    aux (bound.val - 0#usize) 0#usize rfl (Nat.zero_le _) hrun
  exact ⟨iv, hivlt, hivEq, hvac, fun t ht => hocc t (Nat.zero_le _) ht⟩

/-- **The first-fit placement of the ported `spill_locate`.**  On the generated
`table.probe.spill_locate` (`proof/ApeironWord.lean:20886-20905`), for the region's own `bound` and
`home` and a power-of-two extent, a `SpillSite.Vacant index` return is a first-fit placement:
`index` is vacant and every earlier probe slot is occupied.  No insert-path invariant is assumed;
the fact is a consequence of the executed scan. -/
theorem spill_locate_vacant_placement {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : K) (n : Usize) (index : Usize)
    (bound home : Usize) (exponent : Nat)
    (hbound : table.Region.bound r = ok bound)
    (hhome : table.Region.home r address.place = ok home)
    (hpow : bound.val = 2 ^ exponent)
    (hloc : table.probe.spill_locate keyrep r sigma address key n = ok (table.SpillSite.Vacant index)) :
    VacantPlacement r home bound index := by
  unfold table.probe.spill_locate at hloc
  cases hb : table.valid_n n with
  | vis eff k =>
      rw [hb, bind_tc_vis] at hloc
      exact (vis_not_ok hloc).elim
  | div =>
      rw [hb, bind_tc_div] at hloc
      exact (div_not_ok hloc).elim
  | ret b =>
      rw [hb, bind_tc_ok] at hloc
      by_cases hbt : b = true
      · rw [if_pos hbt] at hloc
        cases hbd : table.Region.bound r with
        | vis eff k =>
            rw [hbd, bind_tc_vis] at hloc
            exact (vis_not_ok hloc).elim
        | div =>
            rw [hbd, bind_tc_div] at hloc
            exact (div_not_ok hloc).elim
        | ret i =>
            rw [hbd, bind_tc_ok] at hloc
            by_cases hz : i = 0#usize
            · rw [if_pos hz] at hloc
              exact (spill_site_grow_ne_vacant (Result.ok_injective hloc)).elim
            · rw [if_neg hz] at hloc
              cases hhh : table.Region.home r address.place with
              | vis eff k =>
                  rw [hhh, bind_tc_vis] at hloc
                  exact (vis_not_ok hloc).elim
              | div =>
                  rw [hhh, bind_tc_div] at hloc
                  exact (div_not_ok hloc).elim
              | ret home' =>
                  rw [hhh, bind_tc_ok] at hloc
                  cases hww : table.Region.coord_of R V address.place with
                  | vis eff k =>
                      rw [hww, bind_tc_vis] at hloc
                      exact (vis_not_ok hloc).elim
                  | div =>
                      rw [hww, bind_tc_div] at hloc
                      exact (div_not_ok hloc).elim
                  | ret wanted =>
                      rw [hww, bind_tc_ok] at hloc
                      have hbeq : i = bound := Result.ok_injective (hbd.symm.trans hbound)
                      have hheq : home' = home := Result.ok_injective (hhh.symm.trans hhome)
                      rw [hbeq, hheq] at hloc
                      exact spill_locate_loop_vacant_scan keyrep r sigma address.place
                        address.direct_word key bound home wanted exponent hpow index hloc
      · rw [if_neg hbt] at hloc
        exact (spill_site_grow_ne_vacant (Result.ok_injective hloc)).elim

/-- **The gap discipline is carried by the vacant publish.**  If `index` is a first-fit vacancy of
`before` (`VacantPlacement`) and `publish_live` writes the live coordinate at `index` — framing every
other coordinate and slot — then the pre-state `GapDiscipline` implies it in `after`: the new live
word sits at the head of the occupied prefix, so every slot that stays vacant is later in probe order
than every hit.  The `publish_live` framing is read from the executed body, not assumed. -/
theorem gap_discipline_of_vacant_frame {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (before after : table.Region R V)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : K) (wanted : U64)
    (home bound : Usize) (index : Usize)
    (hframeC : ∀ j : Nat, j ≠ index.val → after.coords.val[j]? = before.coords.val[j]?)
    (hframeS : ∀ j : Nat, j ≠ index.val → after.slots.val[j]? = before.slots.val[j]?)
    (hne : after.coords.val[index.val]? ≠ some table.COORD_EMPTY)
    (hplace : VacantPlacement before home bound index)
    (hprev : GapDiscipline keyrep before sigma address key wanted home bound) :
    GapDiscipline keyrep after sigma address key wanted home bound := by
  obtain ⟨iv, hivlt, hivEq, _hvac, hprefix⟩ := hplace
  intro t htB hvacT t' ht'le ht'B hhit
  unfold ProbeVacantAt at hvacT
  unfold ProbeHitAt at hhit
  have hnePos : (home.val + t) % bound.val ≠ index.val := by
    intro heq
    exact hne (by rw [← heq]; exact hvacT)
  have hvacB : ProbeVacantAt before ((home.val + t) % bound.val) := by
    show before.coords.val[((home.val + t) % bound.val)]? = some table.COORD_EMPTY
    rw [← hframeC ((home.val + t) % bound.val) hnePos]
    exact hvacT
  have htgt : iv < t := by
    by_contra hle
    rcases Nat.lt_or_eq_of_le (Nat.le_of_not_gt hle) with hlt | heq
    · exact (hprefix t hlt) hvacB
    · exact hnePos (by rw [heq, hivEq])
  have ht'ne : (home.val + t') % bound.val ≠ index.val := by
    intro heq
    have heqiv : t' = iv := Nat.ModEq.eq_of_lt_of_lt
      (Nat.ModEq.add_left_cancel' home.val (heq.trans hivEq.symm)) ht'B hivlt
    omega
  obtain ⟨hco, ss, hslot, hmatch⟩ := hhit
  have hcoB : before.coords.val[((home.val + t') % bound.val)]? = some wanted := by
    rw [← hframeC ((home.val + t') % bound.val) ht'ne]
    exact hco
  have hslotB : before.slots.val[((home.val + t') % bound.val)]? = some ss := by
    rw [← hframeS ((home.val + t') % bound.val) ht'ne]
    exact hslot
  exact (hprev t htB hvacB t' ht'le ht'B) ⟨hcoB, ss, hslotB, hmatch⟩

/-- **The vacant-spill commit establishes the gap discipline.**  Composing
`spill_locate_vacant_placement` with the `publish_live` frame: the pre-state gap discipline implies
the post-state gap discipline at the region the commit returns.  The placement is derived from the
executed `spill_locate`, the framing from the executed `publish_live`; the only insert-side input is
the pre-state discipline itself. -/
theorem publish_vacant_establishes_gap {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (before after : table.Region R V) (index : Usize)
    (sigma : cell.Sigma) (held : R) (value : V) (place : U64)
    (address : table.TableAddress) (key : K) (wanted : U64) (home bound : Usize)
    (hplace : VacantPlacement before home bound index)
    (hpub : table.Region.publish_live before index (sigma, held, value) place = ok after)
    (hprev : GapDiscipline keyrep before sigma address key wanted home bound) :
    GapDiscipline keyrep after sigma address key wanted home bound := by
  obtain ⟨⟨c, _hcoord, hcoordAt, hcNe⟩, hcf⟩ :=
    publish_live_coords before after index sigma held value place hpub
  obtain ⟨_, _, _, hsf⟩ :=
    Apeiron.GrowableInsertRefine.publish_live_publish_ordered before after index sigma held value
      place hpub
  refine gap_discipline_of_vacant_frame keyrep before after sigma address key wanted home bound index
    hcf hsf ?_ hplace hprev
  rw [hcoordAt]
  exact fun h => hcNe (Option.some.inj h)

/-- **The `None` half with the read's gap premise discharged from the insert bodies.**  If
`spill_locate` located the first vacancy `index` of `before`, `publish_live` committed the query's
live word there producing `after`, and the pre-state gap discipline holds, then a read of `after`
reporting `ok none` certifies that no slot of `after` is a coord-hit.  Every premise of
`spill_read_none_no_probe_hit_of_gap` is met: its `hgap` is `publish_vacant_establishes_gap`. -/
theorem spill_read_none_no_probe_hit_of_publish_vacant {K H R V : Type}
    (keyrep : table.keyrep.KeyRep R K H) (before after : table.Region R V) (sigma : cell.Sigma)
    (address : table.TableAddress) (key : K) (n bound home : Usize) (wanted : U64)
    (index : Usize) (exponent : Nat) (held : R) (value : V)
    (hvalid : table.valid_n n = ok true)
    (hboundB : table.Region.bound before = ok bound)
    (hhomeB : table.Region.home before address.place = ok home)
    (hboundA : table.Region.bound after = ok bound)
    (hhomeA : table.Region.home after address.place = ok home)
    (hwanted : table.Region.coord_of R V address.place = ok wanted)
    (hpow : bound.val = 2 ^ exponent)
    (hloc : table.probe.spill_locate keyrep before sigma address key n =
      ok (table.SpillSite.Vacant index))
    (hpub : table.Region.publish_live before index (sigma, held, value) address.place = ok after)
    (hprev : GapDiscipline keyrep before sigma address key wanted home bound)
    (hresult : table.probe.spill_read keyrep after sigma address key n = ok none) :
    ∀ j : Nat, j < bound.val → ¬ ProbeHitAt keyrep after sigma address key wanted j := by
  have hplace := spill_locate_vacant_placement keyrep before sigma address key n index bound home
    exponent hboundB hhomeB hpow hloc
  have hgap := publish_vacant_establishes_gap keyrep before after index sigma held value
    address.place address key wanted home bound hplace hpub hprev
  exact spill_read_none_no_probe_hit_of_gap keyrep after sigma address key n bound home wanted
    exponent hvalid hboundA hhomeA hwanted hpow hgap hresult

/-! ## Cone placement

The placement facts, the strengthened read headlines and the restated whole-capacity/whole-shard
forms all have their transitive axiom sets read directly, since the shared `#audit` root
(`AxiomAudit/Core.lean`) is owned by another lane; each must be within the closed cone
`{propext, Classical.choice, Quot.sound}`.  No `sorryAx` is present. -/

#print axioms spill_locate_vacant_placement
#print axioms gap_discipline_of_vacant_frame
#print axioms publish_vacant_establishes_gap
#print axioms spill_read_none_no_probe_hit_of_publish_vacant

#print axioms home_lt_bound
#print axioms spill_read_none_no_probe_hit_of_gap
#print axioms stored_spill_read_none_no_live_key
#print axioms word_stored_spill_read_none_get_default
#print axioms spill_read_none_implies_no_full_match
#print axioms spill_read_none_implies_no_shard_full_match

end Apeiron.GrowableGaplessRead
