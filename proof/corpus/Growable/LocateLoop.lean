import Growable.Scan
import Lean
import Mathlib.Tactic

/-!
# Locate-loop completion facts for `table.probe.spill_locate`

The extracted locate probe is a single linear loop (`table.probe.spill_locate_loop`,
`ApeironWord.lean`): for `step < bound` it reads `slot = (home + step) &&& (bound - 1)`,
returns `Vacant slot` on an empty tag, returns `Found slot` when the tag equals `wanted` and
`table.matches` succeeds, continues with `step + 1` otherwise, and returns `Grow` once the
bound is reached. The mask/bit-scan scanner (`table.least_set_lane`), the grouped outer sweep
(`table.WINDOWS`, `table.scan.match_tag`) and the summary admission (`summary_maybe`) are gone
from the artifact; this module is rewritten against that single loop and no longer imports
`Growable.ReadLoop` (its own `spill_read_loop0*` surface is stale and outside this file's
ownership).

The certification chain:
* `LocateSlotCertified` is the payload-level full-key match at a slot;
* `LocateMatchWitness`/`LocateVacantWitness` add the executed step that read the slot;
* `LocateOuterStepReady` is the body's per-step classification (continue, match, vacancy,
  bound exhausted), proved by `locate_inner_body_shape`;
* `LocateInnerOutcome` is the run outcome over the loop, proved by
  `generated_locate_inner_outcome`;
* `generated_locate_inner_found`/`generated_locate_outer_found` extract the full-key
  certificate from a `Found` run;
* `generated_spill_locate_unfold*` invert the top-level `table.probe.spill_locate` to the
  executed loop run and its certificate.

The former two-level naming (inner mask scan, outer group sweep) collapses to the one live
loop: `LocateOuterInv`/`LocateOuterOutcome` are the same predicates as their inner
counterparts. The audit rows for `least_set_lane_loop_selects`, `least_set_lane_loop_in_range`
and `scan_lane_of_loop_lane` in `AxiomAudit/Core.lean` must be dropped by the audit owner
(that file is outside this task's ownership).
-/

namespace Apeiron.GrowableLocateLoop

open Aeneas Aeneas.Std Result
open apeiron_word

/-! ## L2: the locate probe body step -/

/-- A certified match slot: the slot read yields a spill slot whose full-key comparison
against `key` at `place`/`direct` succeeds. -/
abbrev LocateSlotCertified {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (r : table.Region R V) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : K) (slot : Usize) : Prop :=
  ∃ ss : table.SpillSlot R V,
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
      r.slots slot = ok ss ∧
    table.matches keyRep ss sigma key { place := place, direct_word := direct } = ok true

/-- A recorded vacancy: the executed step read an empty-tag spill slot at `slot`. -/
abbrev LocateVacantWitness {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (r : table.Region R V) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : K) (bound home : Usize) (wanted : U8) (slot : Usize) : Prop :=
  ∃ (ss : table.SpillSlot R V) (step : Usize),
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
      r.slots slot = ok ss ∧
    ss.tag = 0#u8 ∧
    step.val < bound.val ∧
    table.probe.spill_locate_loop.body keyRep r sigma place direct key bound home wanted step
      = ok (ControlFlow.done (table.SpillSite.Vacant slot))

/-- One executed miss step: the body at `step` continues with `step + 1`. -/
abbrev LocateOuterStepMiss {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (r : table.Region R V) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : K) (bound home : Usize) (wanted : U8) (step : Usize) : Prop :=
  step.val < bound.val ∧
  ∃ next : Usize,
    table.probe.spill_locate_loop.body keyRep r sigma place direct key bound home wanted step
      = ok (ControlFlow.cont next) ∧
    next.val = step.val + 1

/-- A recorded match: the executed step returned `Found slot` and the slot's full-key
comparison succeeded. -/
abbrev LocateMatchWitness {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (r : table.Region R V) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : K) (bound home : Usize) (wanted : U8) (slot : Usize) : Prop :=
  ∃ step : Usize,
    step.val < bound.val ∧
    table.probe.spill_locate_loop.body keyRep r sigma place direct key bound home wanted step
      = ok (ControlFlow.done (table.SpillSite.Found slot)) ∧
    LocateSlotCertified keyRep r sigma place direct key slot

/-- The body's per-step classification: continue, match, vacancy, or bound exhausted. -/
abbrev LocateOuterStepReady {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (r : table.Region R V) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : K) (bound home : Usize) (wanted : U8) (step : Usize)
    (cf : ControlFlow Usize table.SpillSite) : Prop :=
  (∃ next : Usize, cf = ControlFlow.cont next ∧ step.val < bound.val ∧
    next.val = step.val + 1) ∨
  (cf = ControlFlow.done table.SpillSite.Grow ∧ bound.val ≤ step.val) ∨
  (∃ slot : Usize, cf = ControlFlow.done (table.SpillSite.Vacant slot) ∧
    step.val < bound.val ∧
    LocateVacantWitness keyRep r sigma place direct key bound home wanted slot) ∨
  (∃ slot : Usize, cf = ControlFlow.done (table.SpillSite.Found slot) ∧
    step.val < bound.val ∧
    LocateSlotCertified keyRep r sigma place direct key slot)

/-- Generic bind inversion on the generated `do` form: a successful bind ran its head
to `ok`. -/
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

/-- Left inverse of `ControlFlow.done` on the locate result type; used to read the done
value out of a body equality without `injection`. -/
private def doneProject (cf : ControlFlow Usize table.SpillSite) : table.SpillSite :=
  match cf with
  | ControlFlow.cont _ => table.SpillSite.Grow
  | ControlFlow.done y => y

private theorem doneProject_done (y : table.SpillSite) : doneProject (ControlFlow.done y) = y :=
  rfl

/-- Left inverse of `table.SpillSite.Found`; used to compare recorded slots. -/
private def foundProject (site : table.SpillSite) : Usize :=
  match site with
  | table.SpillSite.Found x => x
  | table.SpillSite.Vacant x => x
  | table.SpillSite.Grow => 0#usize

private theorem foundProject_found (x : Usize) : foundProject (table.SpillSite.Found x) = x :=
  rfl

/-- Locate-body inversion: a successful body run is one of the four step classifications.
The proof peels the generated binds and the tag/match tests; no mask or bit-scan state
appears because the extracted body has none. -/
theorem locate_inner_body_shape {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (bound home : Usize) (wanted : U8)
    (step : Usize) (cf : ControlFlow Usize table.SpillSite)
    (hrun : table.probe.spill_locate_loop.body keyRep r sigma place direct key bound home
      wanted step = ok cf) :
    LocateOuterStepReady keyRep r sigma place direct key bound home wanted step cf := by
  have hrun0 := hrun
  unfold table.probe.spill_locate_loop.body at hrun
  by_cases hlt : step < bound
  · rw [if_pos hlt] at hrun
    obtain ⟨i1, _hadd, hrun⟩ := bind_ret_inv _ _ _ hrun
    obtain ⟨i2, _hsub, hrun⟩ := bind_ret_inv _ _ _ hrun
    have hlift : lift (i1 &&& i2) = ok (i1 &&& i2) := rfl
    rw [hlift, bind_tc_ok] at hrun
    obtain ⟨ss, hidx, hrun⟩ := bind_ret_inv _ _ _ hrun
    by_cases htag0 : ss.tag = 0#u8
    · rw [if_pos htag0] at hrun
      have hcf : cf = ControlFlow.done (table.SpillSite.Vacant (i1 &&& i2)) :=
        (Result.ok_injective hrun).symm
      have hbody : table.probe.spill_locate_loop.body keyRep r sigma place direct key bound
          home wanted step =
          ok (ControlFlow.done (table.SpillSite.Vacant (i1 &&& i2))) := by
        rw [hcf] at hrun0
        exact hrun0
      exact Or.inr (Or.inr (Or.inl ⟨i1 &&& i2, hcf, hlt,
        ⟨ss, step, hidx, htag0, hlt, hbody⟩⟩))
    · rw [if_neg htag0] at hrun
      by_cases htagw : ss.tag = wanted
      · rw [if_pos htagw] at hrun
        obtain ⟨b, hmatch, hrun⟩ := bind_ret_inv _ _ _ hrun
        by_cases hb : b = true
        · rw [if_pos hb] at hrun
          have hcert : LocateSlotCertified keyRep r sigma place direct key (i1 &&& i2) := by
            refine ⟨ss, hidx, ?_⟩
            rw [hmatch, hb]
          exact Or.inr (Or.inr (Or.inr
            ⟨i1 &&& i2, (Result.ok_injective hrun).symm, hlt, hcert⟩))
        · rw [if_neg hb] at hrun
          obtain ⟨next, hnext, hrun⟩ := bind_ret_inv _ _ _ hrun
          have hval : next.val = step.val + 1 := by
            have hadd := UScalar.add_equiv (x := step) (y := 1#usize)
            rw [hnext] at hadd
            simp only [Result.match.ok] at hadd
            obtain ⟨_, hval, _⟩ := hadd
            simpa using hval
          exact Or.inl ⟨next, (Result.ok_injective hrun).symm, hlt, hval⟩
      · rw [if_neg htagw] at hrun
        obtain ⟨next, hnext, hrun⟩ := bind_ret_inv _ _ _ hrun
        have hval : next.val = step.val + 1 := by
          have hadd := UScalar.add_equiv (x := step) (y := 1#usize)
          rw [hnext] at hadd
          simp only [Result.match.ok] at hadd
          obtain ⟨_, hval, _⟩ := hadd
          simpa using hval
        exact Or.inl ⟨next, (Result.ok_injective hrun).symm, hlt, hval⟩
  · have hnlt : ¬ step.val < bound.val :=
      fun h => hlt ((UScalar.lt_equiv step bound).mpr h)
    have hge : bound.val ≤ step.val := by omega
    rw [if_neg hlt] at hrun
    exact Or.inr (Or.inl ⟨(Result.ok_injective hrun).symm, hge⟩)

/-- The `Found` step certifies its slot: the executed match branch reads the slot and its
full-key comparison returned `true`. -/
theorem generated_spill_locate_clears_selected {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (bound home : Usize) (wanted : U8)
    (step slot : Usize)
    (hrun : table.probe.spill_locate_loop.body keyRep r sigma place direct key bound home
      wanted step = ok (ControlFlow.done (table.SpillSite.Found slot))) :
    LocateSlotCertified keyRep r sigma place direct key slot := by
  rcases locate_inner_body_shape keyRep r sigma place direct key bound home wanted step
      (ControlFlow.done (table.SpillSite.Found slot)) hrun with
    ⟨next, hcont, _, _⟩ | ⟨hgrow, _⟩ | ⟨vslot, hvac, _, _⟩ | ⟨fslot, hfnd, _, hcert⟩
  · cases hcont
  · cases hgrow
  · cases hvac
  · have hFF : table.SpillSite.Found slot = table.SpillSite.Found fslot := by
      simpa only [doneProject_done] using congrArg doneProject hfnd
    have hslot : fslot = slot := by
      simpa only [foundProject_found] using (congrArg foundProject hFF).symm
    subst hslot
    exact hcert

/-! ## L3: the locate probe completion outcome -/

/-- Prefix invariant of the probe loop from `step0` to the current `step`: `step` did not
fall below `step0` and every executed step below it missed. -/
abbrev LocateInnerInv {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (r : table.Region R V) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : K) (bound home : Usize) (wanted : U8) (step0 step : Usize) : Prop :=
  step0.val ≤ step.val ∧
  ∀ s : Usize, step0.val ≤ s.val → s.val < step.val →
    LocateOuterStepMiss keyRep r sigma place direct key bound home wanted s

/-- The probe loop's outcome from `step0`: a certified match, a recorded vacancy, or
exhaustion of the bound with every step from `step0` on recorded as a miss. -/
abbrev LocateInnerOutcome {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (r : table.Region R V) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : K) (bound home : Usize) (wanted : U8) (step0 : Usize) (result : table.SpillSite) : Prop :=
  (∃ slot : Usize, result = table.SpillSite.Found slot ∧
    LocateMatchWitness keyRep r sigma place direct key bound home wanted slot) ∨
  (∃ slot : Usize, result = table.SpillSite.Vacant slot ∧
    LocateVacantWitness keyRep r sigma place direct key bound home wanted slot) ∨
  (result = table.SpillSite.Grow ∧
    ∀ s : Usize, step0.val ≤ s.val → s.val < bound.val →
      LocateOuterStepMiss keyRep r sigma place direct key bound home wanted s)

/-- The two-level (inner/outer) naming collapses to the single live loop; this is the same
predicate as `LocateInnerInv` under the outer-loop name retained for the audit surface. -/
abbrev LocateOuterInv {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (r : table.Region R V) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : K) (bound home : Usize) (wanted : U8) (step0 step : Usize) : Prop :=
  LocateInnerInv keyRep r sigma place direct key bound home wanted step0 step

/-- The two-level (inner/outer) naming collapses to the single live loop; this is the same
predicate as `LocateInnerOutcome` under the outer-loop name retained for the audit
surface. -/
abbrev LocateOuterOutcome {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (r : table.Region R V) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : K) (bound home : Usize) (wanted : U8) (step0 : Usize) (result : table.SpillSite) : Prop :=
  LocateInnerOutcome keyRep r sigma place direct key bound home wanted step0 result

/-- One step of the prefix invariant: a miss at the cursor extends the sweep. -/
theorem locate_inner_inv_snoc {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (bound home : Usize) (wanted : U8)
    {step0 step next : Usize}
    (hinv : LocateInnerInv keyRep r sigma place direct key bound home wanted step0 step)
    (hmiss : LocateOuterStepMiss keyRep r sigma place direct key bound home wanted step)
    (hnext : next.val = step.val + 1) :
    LocateInnerInv keyRep r sigma place direct key bound home wanted step0 next := by
  rcases hinv with ⟨hle, hsweep⟩
  refine ⟨by omega, ?_⟩
  intro s hs hlt
  rcases lt_trichotomy s.val step.val with hlt' | heq | hgt'
  · exact hsweep s hs hlt'
  · have hseq : s = step := UScalar.eq_of_val_eq heq
    subst hseq
    exact hmiss
  · exact (by omega : False).elim

/-- The generated loop wrapper equals its `loop` presentation with a projection lambda. -/
theorem wrapper_loop_form {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (bound home : Usize) (wanted : U8)
    (step : Usize) :
    table.probe.spill_locate_loop keyRep r sigma place direct key bound home wanted step =
      loop (fun step1 => table.probe.spill_locate_loop.body keyRep r sigma place direct key
        bound home wanted step1) step := by
  unfold table.probe.spill_locate_loop
  rfl

/-- The outer-loop name of `wrapper_loop_form`: the locate sweep is the one live loop. -/
theorem outer_wrapper_loop_form {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (bound home : Usize) (wanted : U8)
    (step : Usize) :
    table.probe.spill_locate_loop keyRep r sigma place direct key bound home wanted step =
      loop (fun step1 => table.probe.spill_locate_loop.body keyRep r sigma place direct key
        bound home wanted step1) step :=
  wrapper_loop_form keyRep r sigma place direct key bound home wanted step

/-- With the linear probe the readiness discharge is the body classification itself: the
bounds geometry that the grouped sweep needed no longer appears, so this is
`locate_inner_body_shape` under the retained readiness name. -/
theorem locate_outer_step_ready_of_geometry {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (bound home : Usize) (wanted : U8)
    (step : Usize) (cf : ControlFlow Usize table.SpillSite)
    (hrun : table.probe.spill_locate_loop.body keyRep r sigma place direct key bound home
      wanted step = ok cf) :
    LocateOuterStepReady keyRep r sigma place direct key bound home wanted step cf :=
  locate_inner_body_shape keyRep r sigma place direct key bound home wanted step cf hrun

/-- The outer-loop name of `locate_inner_body_shape`: one live body. -/
theorem locate_outer_body_shape {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (bound home : Usize) (wanted : U8)
    (step : Usize) (cf : ControlFlow Usize table.SpillSite)
    (hrun : table.probe.spill_locate_loop.body keyRep r sigma place direct key bound home
      wanted step = ok cf) :
    LocateOuterStepReady keyRep r sigma place direct key bound home wanted step cf :=
  locate_inner_body_shape keyRep r sigma place direct key bound home wanted step cf hrun

/-! ## L4: the locate probe run outcome -/

/-- The generated probe loop lands `LocateInnerOutcome` from `step0`. The induction is on
`bound - step`, so each `cont` advances the cursor by exactly one; the body inversion
supplies the per-step certificates, and the run itself is the readiness witness (no
unsatisfiable premise at a `Found` step). -/
theorem generated_locate_inner_outcome {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (bound home : Usize) (wanted : U8)
    (step0 : Usize) (res : table.SpillSite)
    (hrun : table.probe.spill_locate_loop keyRep r sigma place direct key bound home wanted
      step0 = ok res) :
    LocateInnerOutcome keyRep r sigma place direct key bound home wanted step0 res := by
  have hgen : ∀ k : Nat, ∀ (step0 : Usize) (res : table.SpillSite),
      bound.val - step0.val ≤ k →
      table.probe.spill_locate_loop keyRep r sigma place direct key bound home wanted step0
        = ok res →
      LocateInnerOutcome keyRep r sigma place direct key bound home wanted step0 res := by
    intro k
    induction k with
    | zero =>
        intro step0 res hk hrun
        have hge : bound.val ≤ step0.val := by omega
        have hbody : table.probe.spill_locate_loop.body keyRep r sigma place direct key
            bound home wanted step0 = ok (ControlFlow.done table.SpillSite.Grow) := by
          unfold table.probe.spill_locate_loop.body
          have hnot : ¬ step0 < bound := by
            intro hlt
            have := (UScalar.lt_equiv step0 bound).mp hlt
            omega
          rw [if_neg hnot]
        rw [wrapper_loop_form keyRep r sigma place direct key bound home wanted step0] at hrun
        rw [loop.eq_def] at hrun
        have hrunA : (Std.bind (table.probe.spill_locate_loop.body keyRep r sigma place
            direct key bound home wanted step0)
            (fun r' => match r' with
              | ControlFlow.cont x =>
                  table.probe.spill_locate_loop keyRep r sigma place direct key bound
                    home wanted x
              | ControlFlow.done y => ok y)) = ok res := by
          convert hrun using 2
          funext r'
          cases r' with
          | cont st => rfl
          | done y => rfl
        obtain ⟨cf, hbodycf, hrest⟩ := bind_ret_inv _ _ _ hrunA
        rw [hbody] at hbodycf
        have hcf : cf = ControlFlow.done table.SpillSite.Grow :=
          (Result.ok_injective hbodycf).symm
        rw [hcf] at hrest
        have hres : res = table.SpillSite.Grow := by
          have hrest' : ok table.SpillSite.Grow = ok res := hrest
          exact (Result.ok_injective hrest').symm
        subst hres
        refine Or.inr (Or.inr ⟨rfl, ?_⟩)
        intro s hs hb
        exact (by omega : False).elim
    | succ k ih =>
        intro step0 res hk hrun
        rw [wrapper_loop_form keyRep r sigma place direct key bound home wanted step0] at hrun
        rw [loop.eq_def] at hrun
        have hrunA : (Std.bind (table.probe.spill_locate_loop.body keyRep r sigma place
            direct key bound home wanted step0)
            (fun r' => match r' with
              | ControlFlow.cont x =>
                  table.probe.spill_locate_loop keyRep r sigma place direct key bound
                    home wanted x
              | ControlFlow.done y => ok y)) = ok res := by
          convert hrun using 2
          funext r'
          cases r' with
          | cont st => rfl
          | done y => rfl
        obtain ⟨cf, hbodycf, hrest⟩ := bind_ret_inv _ _ _ hrunA
        rcases locate_inner_body_shape keyRep r sigma place direct key bound home wanted
            step0 cf hbodycf with
          ⟨next, hcf, hsb, hnext⟩ | ⟨hcfG, hge⟩ | ⟨slot, hcfV, _hsbV, hwitV⟩
          | ⟨slot, hcfF, hsbF, hcertF⟩
        · -- the step missed and the loop continues at `next`
          have hmiss : LocateOuterStepMiss keyRep r sigma place direct key bound home wanted
              step0 := by
            refine ⟨hsb, next, ?_, hnext⟩
            rw [hcf] at hbodycf
            exact hbodycf
          have hloopNext : table.probe.spill_locate_loop keyRep r sigma place direct key
              bound home wanted next = ok res := by
            rw [hcf] at hrest
            exact hrest
          have hk' : bound.val - next.val ≤ k := by omega
          rcases ih next res hk' hloopNext with
            ⟨slot', hresF, hwitF⟩ | ⟨slot', hresV, hwitV⟩ | ⟨hresG, hsweep⟩
          · exact Or.inl ⟨slot', hresF, hwitF⟩
          · exact Or.inr (Or.inl ⟨slot', hresV, hwitV⟩)
          · refine Or.inr (Or.inr ⟨hresG, ?_⟩)
            intro s hs hb
            by_cases hlt : s.val < next.val
            · have hvalEq : s.val = step0.val := by omega
              have hseq : s = step0 := UScalar.eq_of_val_eq hvalEq
              subst hseq
              exact hmiss
            · exact hsweep s (by omega) hb
        · -- the bound was exhausted at `step0`
          have hresG : res = table.SpillSite.Grow := by
            rw [hcfG] at hrest
            have hrest' : ok table.SpillSite.Grow = ok res := hrest
            exact (Result.ok_injective hrest').symm
          subst hresG
          refine Or.inr (Or.inr ⟨rfl, ?_⟩)
          intro s hs hb
          exact (by omega : False).elim
        · -- a vacancy stopped the loop
          have hresV : res = table.SpillSite.Vacant slot := by
            rw [hcfV] at hrest
            have hrest' : ok (table.SpillSite.Vacant slot) = ok res := hrest
            exact (Result.ok_injective hrest').symm
          subst hresV
          exact Or.inr (Or.inl ⟨slot, rfl, hwitV⟩)
        · -- a match stopped the loop
          have hresF : res = table.SpillSite.Found slot := by
            rw [hcfF] at hrest
            have hrest' : ok (table.SpillSite.Found slot) = ok res := hrest
            exact (Result.ok_injective hrest').symm
          subst hresF
          have hmatch : LocateMatchWitness keyRep r sigma place direct key bound home wanted
              slot := by
            refine ⟨step0, hsbF, ?_, hcertF⟩
            rw [hcfF] at hbodycf
            exact hbodycf
          exact Or.inl ⟨slot, rfl, hmatch⟩
  exact hgen (bound.val - step0.val) step0 res (Nat.le_refl _) hrun

/-- The outer-loop name of `generated_locate_inner_outcome`. -/
theorem generated_locate_outer_outcome {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (bound home : Usize) (wanted : U8)
    (step0 : Usize) (res : table.SpillSite)
    (hrun : table.probe.spill_locate_loop keyRep r sigma place direct key bound home wanted
      step0 = ok res) :
    LocateOuterOutcome keyRep r sigma place direct key bound home wanted step0 res :=
  generated_locate_inner_outcome keyRep r sigma place direct key bound home wanted step0
    res hrun

/-- Run-relative found extraction: a probe run that stopped at `Found slot` certifies the
slot's full-key comparison. -/
theorem generated_locate_inner_found {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (bound home : Usize) (wanted : U8)
    (step0 : Usize) (slot : Usize)
    (hrun : table.probe.spill_locate_loop keyRep r sigma place direct key bound home wanted
      step0 = ok (table.SpillSite.Found slot)) :
    LocateSlotCertified keyRep r sigma place direct key slot := by
  rcases generated_locate_inner_outcome keyRep r sigma place direct key bound home wanted
      step0 (table.SpillSite.Found slot) hrun with
    ⟨fslot, hres, hwit⟩ | ⟨vslot, hres, _⟩ | ⟨hres, _⟩
  · have hslot : fslot = slot := by
      simpa only [foundProject_found] using (congrArg foundProject hres).symm
    subst hslot
    rcases hwit with ⟨st, _, _, hcert⟩
    exact hcert
  · cases hres
  · cases hres

/-- The outer-loop name of `generated_locate_inner_found` from the initial cursor. -/
theorem generated_locate_outer_found {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (bound home : Usize) (wanted : U8)
    (slot : Usize)
    (hrun : table.probe.spill_locate_loop keyRep r sigma place direct key bound home wanted
      0#usize = ok (table.SpillSite.Found slot)) :
    LocateSlotCertified keyRep r sigma place direct key slot :=
  generated_locate_inner_found keyRep r sigma place direct key bound home wanted 0#usize
    slot hrun

/-! ## L5: the top-level `spill_locate` unfold -/

/-- A generated `spill_locate` `Found` inverts to the executed linear probe: the region
bound, probe home and wanted tag are the run's parameters, and the returned index carries
the full-key certificate. -/
theorem generated_spill_locate_unfold {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (region : table.Region R V) (sigma : cell.Sigma) (address : table.TableAddress)
    (key : K) (n : Usize) (index : Usize)
    (hresult : table.probe.spill_locate keyRep region sigma address key n =
      ok (table.SpillSite.Found index)) :
    ∃ (bound home : Usize) (wanted : U8),
      table.Region.bound region = ok bound ∧
      table.Region.home region address.place = ok home ∧
      table.tag address.place = ok wanted ∧
      table.probe.spill_locate_loop keyRep region sigma address.place address.direct_word
        key bound home wanted 0#usize = ok (table.SpillSite.Found index) ∧
      LocateSlotCertified keyRep region sigma address.place address.direct_word key index := by
  unfold table.probe.spill_locate at hresult
  obtain ⟨b, _hvalid, hresult⟩ := bind_ret_inv _ _ _ hresult
  by_cases hb : b = true
  · rw [if_pos hb] at hresult
    obtain ⟨bound, hbound, hresult⟩ := bind_ret_inv _ _ _ hresult
    by_cases hzero : bound = 0#usize
    · rw [if_pos hzero] at hresult
      cases Result.ok_injective hresult
    · rw [if_neg hzero] at hresult
      obtain ⟨home, hhome, hresult⟩ := bind_ret_inv _ _ _ hresult
      obtain ⟨wanted, htag, hresult⟩ := bind_ret_inv _ _ _ hresult
      have hcert : LocateSlotCertified keyRep region sigma address.place address.direct_word
          key index :=
        generated_locate_inner_found keyRep region sigma address.place address.direct_word
          key bound home wanted 0#usize index hresult
      exact ⟨bound, home, wanted, hbound, hhome, htag, hresult, hcert⟩
  · rw [if_neg hb] at hresult
    cases Result.ok_injective hresult

/-- Run-relative unfold: the same inversion as `generated_spill_locate_unfold`, kept as
the name the classification layer consumes. -/
theorem generated_spill_locate_unfold_run {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (region : table.Region R V) (sigma : cell.Sigma) (address : table.TableAddress)
    (key : K) (n : Usize) (index : Usize)
    (hresult : table.probe.spill_locate keyRep region sigma address key n =
      ok (table.SpillSite.Found index)) :
    ∃ (bound home : Usize) (wanted : U8),
      table.Region.bound region = ok bound ∧
      table.Region.home region address.place = ok home ∧
      table.tag address.place = ok wanted ∧
      table.probe.spill_locate_loop keyRep region sigma address.place address.direct_word
        key bound home wanted 0#usize = ok (table.SpillSite.Found index) ∧
      LocateSlotCertified keyRep region sigma address.place address.direct_word key index :=
  generated_spill_locate_unfold keyRep region sigma address key n index hresult

end Apeiron.GrowableLocateLoop
