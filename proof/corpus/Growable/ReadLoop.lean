import ApeironWord
import Growable.Representation
import Growable.ReadState
import Growable.Resources
import Growable.Scan
import Lean

/-!
# Generated bounded spill-read sweep: bounded step facts

These facts concern the extracted spill-read bodies, not `Table.get_at`.

The flat spill port replaced the group/window descent with one bounded linear probe cycle over
the capacity, and merged the parallel `Region.tags` array and the occupancy summary into
`Region.slots` and `Region.coords`.

* `table.probe.spill_read` (`proof/ApeironWord.lean:20581-20621`) admits on
  `len region.slots ≠ 0` and `table.valid_n n`, computes `home ← table.Region.home region
  place`, `wanted ← table.Region.coord_of region place`, reads the coordinate word at
  `region.coords[home]`, and either answers `some home` on a full `(Sigma, key)` match or
  enters `table.probe.spill_read_tail`.
* `table.probe.spill_read_tail` (`proof/ApeironWord.lean:20567-20579`) fixes
  `bound ← table.Region.bound region` (the capacity), `mask ← bound - 1`, re-derives the same
  `wanted`, and runs `table.probe.spill_read_tail_loop` from first step `1`.
* `table.probe.spill_read_tail_loop.body` (`proof/ApeironWord.lean:20515-20550`) reads
  `region.coords[(home + step) &&& mask]` and stops on the vacancy marker
  `table.COORD_EMPTY`, on a full-key match, or at the bound.

`Growable.Scan` already fixes the lane visited at step `k` as `SpillCycleSlot bound home k` and
proves the cycle is a permutation of the slots under a power-of-two capacity.  This module is the
read-path half: what the body records at each executed lane, what the whole sweep therefore
establishes, and what the read composition delivers.

## What replaced what

| retired subject | replacement | section |
|---|---|---|
| `table.WINDOWS`, `table.group`, `table.DESCENT_MASK`, `table.WINDOW_MASK` | none: the `WINDOWS + groups` descent has no denotation | 1 |
| `table.scan.match_tag`, `least_set_lane`, `InnerCompared`, `InnerReady`, `InnerSweepInv`, `InnerSweepOutcome` | `TailLaneObserved`, `TailSweepInv`, `TailSweepOutcome` over the coordinate lane | 2 |
| `table.probe.spill_read_loop0`, `spill_read_loop0_loop0` | `table.probe.spill_read_tail_loop`, `.body` | 1, 2 |
| `generated_group_bounded`, `generated_tail_group_surjective` | `Growable.Scan`'s `SpillCycleSlot` and `spill_cycle_covers_of_placement` | 2 |
| `table.Region.summary_maybe` positive-summary admission | the generated admission chain `len ≠ 0` and `valid_n n`, plus the coordinate identity guard | 6 |
| the summary-word readback reaching `StoredLiveTagAddress` | `StoredLiveCoordAddress`: the coordinate word read back at a live lane | 3 |

`Growable.Read` and `Growable.ReadComposition` imported the retired `OuterStepReady`,
`GroupedReadAdmission`, `repR_summary_read_branch`, `repR_live_spill_summary_positive`,
`spill_read_none_implies_no_full_match` and `spill_read_none_implies_no_shard_full_match`; those
names are **not** re-declared here and their restatements are the open obligations below.
-/

namespace Apeiron.GrowableReadLoop

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-! ## 1 · Local arithmetic of the generated checked operations

The body computes `home + step` and `step + 1` through the generated checked `Usize` operations,
so every lane fact carries the checked equation that produced it. -/

/-- A `do`-bind whose source is a `Result` records its value and its continuation. -/
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

/-- The generated `&&&` under `lift` is the scalar `&&&`. -/
private theorem lift_and (x y : Usize) : (lift (x &&& y) : Result Usize) = ok (x &&& y) := rfl

/-- The generated checked `step + 1` of a `cont` branch succeeds whenever the guard held, and its
component is one past the guard's step and no past the bound.  This is the read-loop analogue of
the retired `generated_outer_increment_checked`, whose statement it keeps. -/
theorem generated_tail_increment_checked (st bound : Usize) (hst : st < bound) :
    ∃ next : Usize, st + 1#usize = ok next ∧
      next.val = st.val + 1 ∧ next.val ≤ bound.val := by
  have hltVal := (UScalar.lt_equiv st bound).mp hst
  have hboundCap : bound.val ≤ Usize.max := by
    have hbound := bound.hBounds
    simp only [UScalarTy.Usize_numBits_eq] at hbound
    have hmax : Usize.max = 2 ^ System.Platform.numBits - 1 := by
      simp [Usize.max, Usize.numBits]
    rw [hmax]
    omega
  have hone : (1#usize).val = 1 := rfl
  obtain ⟨next, hnext, hnextVal⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := st) (y := 1#usize) (by omega))
  refine ⟨next, hnext, ?_, ?_⟩
  · simpa using hnextVal
  · omega

/-! ## 2 · Local facts about the generated read-tail body -/

/-- A generated `spill_read_tail_loop.body` step that reports a full match at `lane` carries, in
the same step, the coordinate readback from `region.coords`, both guards it tested (the vacancy
marker and the identity guard), the successful `region.slots` read and the successful
`table.matches` call.  The body never reads `ss.tag`: the occupancy marker lives in
`region.coords` as the vacancy word. -/
private theorem tail_body_found {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (region : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (home bound mask : Usize)
    (wanted : U64) (st lane : Usize)
    (hbody : table.probe.spill_read_tail_loop.body keyRep region sigma place direct key home
      bound mask wanted st = ok (ControlFlow.done (some lane))) :
    ∃ (coord : U64) (ss : table.SpillSlot R V) (i1 : Usize),
      st < bound ∧ home + st = ok i1 ∧ i1 &&& mask = lane ∧
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
        region.coords lane = ok coord ∧
      coord ≠ table.COORD_EMPTY ∧ coord = wanted ∧
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
        region.slots lane = ok ss ∧
      table.matches keyRep ss sigma key { place := place, direct_word := direct } = ok true := by
  unfold table.probe.spill_read_tail_loop.body at hbody
  by_cases hlt : st < bound
  · rw [if_pos hlt] at hbody
    cases hadd : (home + st) with
    | vis eff k =>
        rw [hadd, bind_tc_vis] at hbody
        exact (vis_not_ok hbody).elim
    | div =>
        rw [hadd, bind_tc_div] at hbody
        exact (div_not_ok hbody).elim
    | ret i1 =>
        rw [hadd, bind_tc_ok] at hbody
        rw [lift_and, bind_tc_ok] at hbody
        cases hidx : alloc.vec.Vec.index
            (core.slice.index.SliceIndexUsizeSlice Std.U64) region.coords (i1 &&& mask) with
        | vis eff k =>
            rw [hidx, bind_tc_vis] at hbody
            exact (vis_not_ok hbody).elim
        | div =>
            rw [hidx, bind_tc_div] at hbody
            exact (div_not_ok hbody).elim
        | ret coord =>
            rw [hidx, bind_tc_ok] at hbody
            by_cases hzero : coord = table.COORD_EMPTY
            · rw [if_pos hzero] at hbody
              exact False.elim (by
                have h := Result.ok_injective hbody
                cases h)
            · rw [if_neg hzero] at hbody
              by_cases hmatch : coord = wanted
              · rw [if_pos hmatch] at hbody
                cases hss : alloc.vec.Vec.index
                    (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
                    region.slots (i1 &&& mask) with
                | vis eff k =>
                    rw [hss, bind_tc_vis] at hbody
                    exact (vis_not_ok hbody).elim
                | div =>
                    rw [hss, bind_tc_div] at hbody
                    exact (div_not_ok hbody).elim
                | ret ss =>
                    rw [hss, bind_tc_ok] at hbody
                    cases hmt : table.matches keyRep ss sigma key
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
                          have hinj := Result.ok_injective hbody
                          have hinj' : some (i1 &&& mask) = some lane := by
                            cases hinj
                            rfl
                          have hilane : i1 &&& mask = lane := Option.some.inj hinj'
                          rw [hilane] at hidx hss
                          exact ⟨coord, ss, i1, hlt, rfl, hilane, hidx, hzero, hmatch, hss,
                            hmt.trans (congrArg ok hb)⟩
                        · rw [if_neg hb] at hbody
                          obtain ⟨s1, hninc, hfin⟩ := bind_ret_inv (st + 1#usize)
                            (fun s1 => ok (ControlFlow.cont s1))
                            (ControlFlow.done (some lane)) hbody
                          exact False.elim (by
                            have h := Result.ok_injective hfin
                            cases h)
              · rw [if_neg hmatch] at hbody
                obtain ⟨s1, hninc, hfin⟩ := bind_ret_inv (st + 1#usize)
                  (fun s1 => ok (ControlFlow.cont s1))
                  (ControlFlow.done (some lane)) hbody
                exact False.elim (by
                  have h := Result.ok_injective hfin
                  cases h)
  · rw [if_neg hlt] at hbody
    exact False.elim (by
      have h := Result.ok_injective hbody
      cases h)

/-- The generated read-tail body stops with the accumulated `none` once the cycle's bound is
reached, and that stop reads no coordinate at all.  This is the read-loop replacement for the
retired `generated_outer_stops_at_limit`. -/
theorem generated_tail_stops_at_bound {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (region : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (home bound mask : Usize)
    (wanted : U64) (st : Usize) (hst : ¬ st < bound) :
    table.probe.spill_read_tail_loop.body keyRep region sigma place direct key home
      bound mask wanted st = ok (ControlFlow.done none) := by
  unfold table.probe.spill_read_tail_loop.body
  rw [if_neg hst]

/-- A generated read-tail body step that stopped with the vacancy terminator carries the
coordinate readback at its lane and the vacancy guard.  This is the "stop short of the bound"
outcome the retired grouped sweep had no denotation for: the bounded cycle terminates at the
first vacant lane, so a `none` result by itself does **not** witness that every lane was read. -/
private theorem tail_body_vacant {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (region : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (home bound mask : Usize)
    (wanted : U64) (st : Usize)
    (hst : st < bound)
    (hbody : table.probe.spill_read_tail_loop.body keyRep region sigma place direct key home
      bound mask wanted st = ok (ControlFlow.done none)) :
    ∃ (coord : U64) (i1 : Usize),
      home + st = ok i1 ∧
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
        region.coords (i1 &&& mask) = ok coord ∧
      coord = table.COORD_EMPTY := by
  unfold table.probe.spill_read_tail_loop.body at hbody
  rw [if_pos hst] at hbody
  cases hadd : (home + st) with
  | vis eff k =>
      rw [hadd, bind_tc_vis] at hbody
      exact (vis_not_ok hbody).elim
  | div =>
      rw [hadd, bind_tc_div] at hbody
      exact (div_not_ok hbody).elim
  | ret i1 =>
      rw [hadd, bind_tc_ok] at hbody
      rw [lift_and, bind_tc_ok] at hbody
      cases hidx : alloc.vec.Vec.index
          (core.slice.index.SliceIndexUsizeSlice Std.U64) region.coords (i1 &&& mask) with
      | vis eff k =>
          rw [hidx, bind_tc_vis] at hbody
          exact (vis_not_ok hbody).elim
      | div =>
          rw [hidx, bind_tc_div] at hbody
          exact (div_not_ok hbody).elim
      | ret coord =>
          rw [hidx, bind_tc_ok] at hbody
          by_cases hzero : coord = table.COORD_EMPTY
          · rw [if_pos hzero] at hbody
            exact ⟨coord, i1, rfl, hidx, hzero⟩
          · rw [if_neg hzero] at hbody
            by_cases hmatch : coord = wanted
            · rw [if_pos hmatch] at hbody
              cases hss : alloc.vec.Vec.index
                  (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
                  region.slots (i1 &&& mask) with
              | vis eff k =>
                  rw [hss, bind_tc_vis] at hbody
                  exact (vis_not_ok hbody).elim
              | div =>
                  rw [hss, bind_tc_div] at hbody
                  exact (div_not_ok hbody).elim
              | ret ss =>
                  rw [hss, bind_tc_ok] at hbody
                  cases hmt : table.matches keyRep ss sigma key
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
                        exact False.elim (by
                          have h := Result.ok_injective hbody
                          cases h)
                      · rw [if_neg hb] at hbody
                        obtain ⟨s1, hninc, hfin⟩ := bind_ret_inv (st + 1#usize)
                          (fun s1 => ok (ControlFlow.cont s1))
                          (ControlFlow.done none) hbody
                        exact False.elim (by
                          have h := Result.ok_injective hfin
                          cases h)
            · rw [if_neg hmatch] at hbody
              obtain ⟨s1, hninc, hfin⟩ := bind_ret_inv (st + 1#usize)
                (fun s1 => ok (ControlFlow.cont s1))
                (ControlFlow.done none) hbody
              exact False.elim (by
                have h := Result.ok_injective hfin
                cases h)

/-- A generated read-tail body step that continued past its lane carries the coordinate readback
at that lane and the exclusion its two guards established there: either the lane's coordinate word
is not the wanted coordinate, or it is, and the body read the lane's slot and the full-key matcher
refused it.  This is the read-loop replacement for the retired `InnerCompared` and the 64-lane
`InnerSweepInv`/`InnerSweepOutcome` apparatus, which located the identity guard by scanning a
`U8` tag array instead of comparing a `U64` coordinate word. -/
private theorem tail_body_cont {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (region : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (home bound mask : Usize)
    (wanted : U64) (st next : Usize)
    (hbody : table.probe.spill_read_tail_loop.body keyRep region sigma place direct key home
      bound mask wanted st = ok (ControlFlow.cont next)) :
    st < bound ∧ st + 1#usize = ok next ∧
      ∃ (coord : U64) (i1 : Usize),
        home + st = ok i1 ∧
        alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
          region.coords (i1 &&& mask) = ok coord ∧
        coord ≠ table.COORD_EMPTY ∧
        (coord ≠ wanted ∨
          ∃ ss : table.SpillSlot R V,
            alloc.vec.Vec.index
              (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
              region.slots (i1 &&& mask) = ok ss ∧
            table.matches keyRep ss sigma key
              { place := place, direct_word := direct } = ok false) := by
  unfold table.probe.spill_read_tail_loop.body at hbody
  by_cases hlt : st < bound
  · rw [if_pos hlt] at hbody
    cases hadd : (home + st) with
    | vis eff k =>
        rw [hadd, bind_tc_vis] at hbody
        exact (vis_not_ok hbody).elim
    | div =>
        rw [hadd, bind_tc_div] at hbody
        exact (div_not_ok hbody).elim
    | ret i1 =>
        rw [hadd, bind_tc_ok] at hbody
        rw [lift_and, bind_tc_ok] at hbody
        cases hidx : alloc.vec.Vec.index
            (core.slice.index.SliceIndexUsizeSlice Std.U64) region.coords (i1 &&& mask) with
        | vis eff k =>
            rw [hidx, bind_tc_vis] at hbody
            exact (vis_not_ok hbody).elim
        | div =>
            rw [hidx, bind_tc_div] at hbody
            exact (div_not_ok hbody).elim
        | ret coord =>
            rw [hidx, bind_tc_ok] at hbody
            by_cases hzero : coord = table.COORD_EMPTY
            · rw [if_pos hzero] at hbody
              exact False.elim (by
                have h := Result.ok_injective hbody
                cases h)
            · rw [if_neg hzero] at hbody
              by_cases hmatch : coord = wanted
              · rw [if_pos hmatch] at hbody
                cases hss : alloc.vec.Vec.index
                    (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
                    region.slots (i1 &&& mask) with
                | vis eff k =>
                    rw [hss, bind_tc_vis] at hbody
                    exact (vis_not_ok hbody).elim
                | div =>
                    rw [hss, bind_tc_div] at hbody
                    exact (div_not_ok hbody).elim
                | ret ss =>
                    rw [hss, bind_tc_ok] at hbody
                    cases hmt : table.matches keyRep ss sigma key
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
                          exact False.elim (by
                            have h := Result.ok_injective hbody
                            cases h)
                        · rw [if_neg hb] at hbody
                          obtain ⟨s1, hninc, hfin⟩ := bind_ret_inv (st + 1#usize)
                            (fun s1 => ok (ControlFlow.cont s1))
                            (ControlFlow.cont next) hbody
                          have hinj := Result.ok_injective hfin
                          have heq : s1 = next := by
                            cases hinj
                            rfl
                          exact ⟨hlt, by rw [← heq]; exact hninc, coord, i1, rfl, hidx, hzero,
                            Or.inr ⟨ss, hss, hmt.trans
                              (congrArg ok (Bool.eq_false_of_not_eq_true hb))⟩⟩
              · rw [if_neg hmatch] at hbody
                obtain ⟨s1, hninc, hfin⟩ := bind_ret_inv (st + 1#usize)
                  (fun s1 => ok (ControlFlow.cont s1))
                  (ControlFlow.cont next) hbody
                have hinj := Result.ok_injective hfin
                have heq : s1 = next := by
                  cases hinj
                  rfl
                exact ⟨hlt, by rw [← heq]; exact hninc, coord, i1, rfl, hidx, hzero,
                  Or.inl hmatch⟩
  · rw [if_neg hlt] at hbody
    exact False.elim (by
      have h := Result.ok_injective hbody
      cases h)

/-! ## The lane observation and the sweep -/

/-- What the generated read-tail body observed at one lane: the coordinate word it read from
`region.coords`, the vacancy guard that word passed, and — in the only case where the identity
guard also passed — the lane's slot and the full-key matcher call that refused it.  A
`COORD_EMPTY` word is *not* part of this: it terminates the cycle instead of recording an
observation. -/
def TailLaneObserved {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (region : table.Region R V) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : K) (lane : Usize) (wanted : U64) : Prop :=
  ∃ coord : U64,
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
      region.coords lane = ok coord ∧
    coord ≠ table.COORD_EMPTY ∧
    (coord ≠ wanted ∨
      ∃ ss : table.SpillSlot R V,
        alloc.vec.Vec.index
          (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
          region.slots lane = ok ss ∧
        table.matches keyRep ss sigma key
          { place := place, direct_word := direct } = ok false)

/-- Every step of `[start, state.val)` has been executed, and at each the generated `home + k` and
`(home + k) &&& mask` computations and that step's lane observation are recorded. -/
def TailSweepInv {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (region : table.Region R V) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : K) (home _bound mask : Usize) (wanted : U64) (start state : Usize) : Prop :=
  ∀ (k : Nat), start.val ≤ k → k < state.val →
    ∃ (st i1 : Usize), st.val = k ∧ home + st = ok i1 ∧
      TailLaneObserved keyRep region sigma place direct key (i1 &&& mask) wanted

/-- The three outcomes the real generated bounded cycle has.  A `none` result is *either* the
bound being reached with every lane of `[start, bound)` observed, *or* the vacancy terminator
firing at a lane below the bound.  A `some lane` result is a full-key match at a lane whose own
generated reads are recorded. -/
def TailSweepOutcome {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (region : table.Region R V) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : K) (home bound mask : Usize) (wanted : U64) (start : Usize)
    (result : Option Usize) : Prop :=
  (∃ stop : Usize, result = none ∧
      TailSweepInv keyRep region sigma place direct key home bound mask wanted start stop ∧
      (bound.val ≤ stop.val ∨
        ∃ (i1 : Usize) (coord : U64), home + stop = ok i1 ∧
          alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
            region.coords (i1 &&& mask) = ok coord ∧
          coord = table.COORD_EMPTY)) ∨
  (∃ (stop i1 lane : Usize) (ss : table.SpillSlot R V), result = some lane ∧
    TailSweepInv keyRep region sigma place direct key home bound mask wanted start stop ∧
    home + stop = ok i1 ∧ i1 &&& mask = lane ∧
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
      region.slots lane = ok ss ∧
    table.matches keyRep ss sigma key { place := place, direct_word := direct } = ok true)

/-- A `cont` step's two obligations, given the body's equation: the invariant at the next step, and
the decrease of the distance-to-the-bound measure.  The body's own equation fixes the next step as
the generated `step + 1` and records that step's lane observation. -/
private theorem tail_cont_obligation {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (region : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (home bound mask : Usize) (wanted : U64)
    (start st next : Usize)
    (hst : TailSweepInv keyRep region sigma place direct key home bound mask wanted start st)
    (hlt : st < bound)
    (hbody : table.probe.spill_read_tail_loop.body keyRep region sigma place direct key home
      bound mask wanted st = ok (ControlFlow.cont next)) :
    TailSweepInv keyRep region sigma place direct key home bound mask wanted start next ∧
      bound.val - next.val < bound.val - st.val := by
  obtain ⟨hlt', hnext', coord, i1, hadd, hidx, hzero, hexcl⟩ :=
    tail_body_cont keyRep region sigma place direct key home bound mask wanted st next hbody
  have hnextVal : next.val = st.val + 1 ∧ next.val ≤ bound.val := by
    obtain ⟨next'', hn, hp⟩ := generated_tail_increment_checked st bound hlt
    have hEq : ok next'' = ok next := hn.symm.trans hnext'
    have hinj : next'' = next := Result.ok_injective hEq
    rw [← hinj]
    exact ⟨hp.1, hp.2⟩
  have hltVal : st.val < bound.val := (UScalar.lt_equiv st bound).mp hlt
  refine ⟨?_, by rw [hnextVal.1]; omega⟩
  intro k hk0 hk1
  by_cases hke : k < st.val
  · exact hst k hk0 hke
  · have hks : k = st.val := by
      rw [hnextVal.1] at hk1
      omega
    subst k
    exact ⟨st, ⟨_, rfl, hadd, ⟨coord, hidx, hzero, hexcl⟩⟩⟩

/-- **The generated bounded cycle's outcome.**  The real `spill_read_tail_loop`, started at
`start`, either exhausted its bound having observed every lane of `[start, bound)`, or stopped at
a vacancy, or returned a lane whose full key matched.  The measure is the distance to the bound,
which decreases exactly on the `cont` branches because those only occur below the bound.  This
replaces the retired `generated_outer_sweep_outcome` and `generated_inner_sweep_outcome`. -/
theorem generated_tail_sweep_spec {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (region : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (home bound mask : Usize)
    (wanted : U64) (start : Usize)
    (hCoord : ∀ (st : Usize),
      TailSweepInv keyRep region sigma place direct key home bound mask wanted start st →
      st < bound →
      ∃ (coord : U64) (i1 : Usize), home + st = ok i1 ∧
        alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
          region.coords (i1 &&& mask) = ok coord)
    (hSlot : ∀ (st : Usize) (coord : U64) (i1 : Usize), st < bound → home + st = ok i1 →
        alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
          region.coords (i1 &&& mask) = ok coord → coord = wanted →
      ∃ (ss : table.SpillSlot R V),
        alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
          region.slots (i1 &&& mask) = ok ss)
    (hMatch : ∀ (st : Usize) (ss : table.SpillSlot R V), st < bound →
      ∃ matched : Bool, table.matches keyRep ss sigma key
        { place := place, direct_word := direct } = ok matched)
    (hAdd : ∀ st : Usize, st < bound → ∃ next : Usize, st + 1#usize = ok next) :
    table.probe.spill_read_tail_loop keyRep region sigma place direct key home
      bound mask wanted start ⦃ result =>
        TailSweepOutcome keyRep region sigma place direct key home bound mask wanted
          start result ⦄ := by
  unfold table.probe.spill_read_tail_loop
  apply loop.spec_decr_nat
    (measure := fun st : Usize => bound.val - st.val)
    (inv := TailSweepInv keyRep region sigma place direct key home bound mask wanted start)
    (post := TailSweepOutcome keyRep region sigma place direct key home bound mask wanted start)
  · intro st hst
    by_cases hlt : st < bound
    · obtain ⟨coord, i1, hadd, hidx⟩ := hCoord st hst hlt
      by_cases hzero : coord = table.COORD_EMPTY
      · have hbody : table.probe.spill_read_tail_loop.body keyRep region sigma place direct key
            home bound mask wanted st = ok (ControlFlow.done none) := by
          unfold table.probe.spill_read_tail_loop.body
          rw [if_pos hlt, hadd, bind_tc_ok, lift_and, bind_tc_ok, hidx, bind_tc_ok,
            if_pos hzero]
        obtain ⟨coord', i1', hadd', hidx', hzero'⟩ :=
          tail_body_vacant keyRep region sigma place direct key home bound mask wanted st hlt
            hbody
        refine WP.exists_imp_spec ⟨_, hbody, ?_⟩
        exact Or.inl ⟨st, rfl, hst, Or.inr ⟨i1', coord', hadd', hidx', hzero'⟩⟩
      · by_cases hmatch : coord = wanted
        · obtain ⟨ss, hss⟩ := hSlot st coord i1 hlt hadd hidx hmatch
          obtain ⟨b, hmt⟩ := hMatch st ss hlt
          by_cases hb : b = true
          · have hbody : table.probe.spill_read_tail_loop.body keyRep region sigma place
                direct key home bound mask wanted st =
                ok (ControlFlow.done (some (i1 &&& mask))) := by
              unfold table.probe.spill_read_tail_loop.body
              rw [if_pos hlt, hadd, bind_tc_ok, lift_and, bind_tc_ok, hidx, bind_tc_ok,
                if_neg hzero, if_pos hmatch, hss, bind_tc_ok, hmt, bind_tc_ok, if_pos hb]
            obtain ⟨_, _, _, _, _, _, _, _, _, _, hmtTrue⟩ :=
              tail_body_found keyRep region sigma place direct key home bound mask wanted
                st (i1 &&& mask) hbody
            refine WP.exists_imp_spec ⟨_, hbody, ?_⟩
            exact Or.inr ⟨st, i1, i1 &&& mask, ss, rfl, hst, hadd, rfl, hss,
              hmt.trans (congrArg ok hb)⟩
          · obtain ⟨next, hnext⟩ := hAdd st hlt
            have hbody : table.probe.spill_read_tail_loop.body keyRep region sigma place
                direct key home bound mask wanted st = ok (ControlFlow.cont next) := by
              unfold table.probe.spill_read_tail_loop.body
              rw [if_pos hlt, hadd, bind_tc_ok, lift_and, bind_tc_ok, hidx, bind_tc_ok,
                if_neg hzero, if_pos hmatch, hss, bind_tc_ok, hmt, bind_tc_ok, if_neg hb,
                hnext, bind_tc_ok]
            exact WP.exists_imp_spec ⟨_, hbody,
              tail_cont_obligation keyRep region sigma place direct key home bound mask wanted
                start st next hst hlt hbody⟩
        · obtain ⟨next, hnext⟩ := hAdd st hlt
          have hbody : table.probe.spill_read_tail_loop.body keyRep region sigma place
              direct key home bound mask wanted st = ok (ControlFlow.cont next) := by
            unfold table.probe.spill_read_tail_loop.body
            rw [if_pos hlt, hadd, bind_tc_ok, lift_and, bind_tc_ok, hidx, bind_tc_ok,
              if_neg hzero, if_neg hmatch, hnext, bind_tc_ok]
          exact WP.exists_imp_spec ⟨_, hbody,
            tail_cont_obligation keyRep region sigma place direct key home bound mask wanted
              start st next hst hlt hbody⟩
    · have hbody : table.probe.spill_read_tail_loop.body keyRep region sigma place direct key
          home bound mask wanted st = ok (ControlFlow.done none) := by
        unfold table.probe.spill_read_tail_loop.body
        rw [if_neg hlt]
      refine WP.exists_imp_spec ⟨_, hbody, ?_⟩
      refine Or.inl ⟨st, rfl, hst, Or.inl ?_⟩
      have hltVal : ¬ st.val < bound.val := ((UScalar.lt_equiv st bound).not).mp hlt
      omega
  · intro k hk0 hk1
    omega

/-- The equation form of `generated_tail_sweep_spec`: what a run that returned `result` recorded. -/
theorem generated_tail_sweep_outcome {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (region : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (home bound mask : Usize)
    (wanted : U64) (start : Usize) (result : Option Usize)
    (hCoord : ∀ (st : Usize),
      TailSweepInv keyRep region sigma place direct key home bound mask wanted start st →
      st < bound →
      ∃ (coord : U64) (i1 : Usize), home + st = ok i1 ∧
        alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
          region.coords (i1 &&& mask) = ok coord)
    (hSlot : ∀ (st : Usize) (coord : U64) (i1 : Usize), st < bound → home + st = ok i1 →
        alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
          region.coords (i1 &&& mask) = ok coord → coord = wanted →
      ∃ (ss : table.SpillSlot R V),
        alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
          region.slots (i1 &&& mask) = ok ss)
    (hMatch : ∀ (st : Usize) (ss : table.SpillSlot R V), st < bound →
      ∃ matched : Bool, table.matches keyRep ss sigma key
        { place := place, direct_word := direct } = ok matched)
    (hresult : table.probe.spill_read_tail_loop keyRep region sigma place direct key home
      bound mask wanted start = ok result) :
    TailSweepOutcome keyRep region sigma place direct key home bound mask wanted
      start result := by
  have hpost := generated_tail_sweep_spec keyRep region sigma place direct key home bound mask
    wanted start hCoord hSlot hMatch (fun st hlt => by
      obtain ⟨n, hn, _⟩ := generated_tail_increment_checked st bound hlt
      exact ⟨n, hn⟩)
  rw [hresult, WP.spec_ok] at hpost
  exact hpost

/-! ## 3 · Readback and the live coordinate law -/

/-- A successful generated lookup is in bounds and reads exactly the stored list entry. -/
private theorem vec_index_readback {T : Type} (v : alloc.vec.Vec T) (i : Usize) (entry : T)
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

/-- **The live coordinate law of a scanned region.**  An occupied slot at `index` carries, in
`region.coords`, the coordinate word of its own home address in that shard.  This is the
coordinate-level replacement for the retired positive summary word `table.Region.summary_maybe`
together with the tag readback that reached it: the generated read path's identity guard is now
`coord = wanted` with `wanted = table.Region.coord_of place`, and `table.Region.publish_live`
writes exactly `table.Region.coord_of place` into `region.coords` at the index it publishes
(`proof/ApeironWord.lean:20711-20718`).  `StoredLiveTagAddress` records the tag-level
correspondence for the same lanes; this records the coordinate-level one the probe compares. -/
def StoredLiveCoordAddress
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (shard : Usize) : Prop :=
  ∀ (index : Usize) (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
      (tag : U8) (sigma : cell.Sigma) (held : Apeiron.GrowableIdentity.StoredI64) (value : I64),
    state.shards.val[shard.val]? = some region →
    region.slots.val[index.val]? =
      some { tag := tag, sigma := sigma, held := held, value := value } → tag ≠ 0#u8 →
    ∃ (address : table.TableAddress) (coord : U64),
      Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma held.key = ok address ∧
      table.shard address = ok shard ∧
      table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place = ok coord ∧
      region.coords.val[index.val]? = some coord

/-- A generated matcher call terminates for every payload and query; the generic `KeyRep` trait
alone does not supply this law.  This replaces the retired `stored_matcher_total`, whose statement
it keeps. -/
theorem stored_matcher_total
    (entry : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress) :
    ∃ matched : Bool,
      table.matches Apeiron.GrowableIdentity.storedKeyRep entry sigma key address =
        ok matched := by
  rcases entry with ⟨tag, slotSigma, held, value⟩
  cases slotSigma <;> cases sigma <;> cases held <;>
    simp [table.matches, cell.Sigma.Insts.CoreCmpPartialEqSigma.eq,
      cell.Sigma.read_discriminant,
      table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.matches]

/-- A successful generated payload lookup followed by the *real stored-key matcher* implies
in-bounds, complete `(Sigma, key)` equality.  A coordinate hit by itself supplies neither fact.
This replaces the retired `stored_candidate_full_match`, whose statement it keeps. -/
theorem stored_candidate_full_match
    (slots : alloc.vec.Vec (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64))
    (index : Usize) (entry : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress)
    (hslot : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice
        (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) slots index = ok entry)
    (hmatch : table.matches Apeiron.GrowableIdentity.storedKeyRep entry sigma key address =
      ok true) :
    index.val < slots.length ∧ slots.val[index.val]? = some entry ∧
      Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held = (sigma, key) := by
  obtain ⟨hinB, hreadB⟩ := vec_index_readback slots index entry hslot
  exact ⟨hinB, hreadB,
    (Apeiron.GrowableIdentity.stored_matches_iff
      entry.sigma sigma entry.held entry.value key address).mp hmatch⟩

/-! ## 4 · Deliverable 2: a returned lane is a full match -/

/-- **The generated reads of the bounded cycle, as a named premise.**  The tail body reads
`region.coords[(home + step) &&& mask]`, and on a coordinate hit also
`region.slots[(home + step) &&& mask]`.  Neither read is total: `region.coords` may be shorter than
`bound`, and `home + step` may overflow, so either `Vec.index` can return a `vis` residue and the
whole `spill_read_tail_loop` becomes a `vis` rather than an `ok`.  A consumer holding
`spill_read_tail_loop … = ok result` therefore knows both reads succeeded at every executed step,
but extracting that needs a well-founded induction over the loop's own recursion which is **not**
written here; see the receipt's open obligation `OPS-2`.  `hMatch` and `hAdd` *are* discharged
inside `generated_tail_sweep_outcome` — the latter from `generated_tail_increment_checked` — so only
the two reads are carried. -/
structure TailReadSuccess {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (region : table.Region R V) (sigma : cell.Sigma) (place : U64) (direct : Option U32)
    (key : K) (home bound mask : Usize) (wanted : U64) (start : Usize) : Prop where
  /-- The coordinate read of every executed lane below the bound succeeded. -/
  coord : ∀ (st : Usize),
    TailSweepInv keyRep region sigma place direct key home bound mask wanted start st →
    st < bound →
    ∃ (c : U64) (i1 : Usize), home + st = ok i1 ∧
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
        region.coords (i1 &&& mask) = ok c
  /-- On a coordinate hit, the payload read of that lane succeeded too. -/
  slot : ∀ (st : Usize) (c : U64) (i1 : Usize), st < bound → home + st = ok i1 →
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
      region.coords (i1 &&& mask) = ok c → c = wanted →
    ∃ (ss : table.SpillSlot R V),
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
        region.slots (i1 &&& mask) = ok ss

/-- **Deliverable 2 at the sweep.**  A generated bounded cycle that returned `some lane` did so on
a successful `table.matches` at that lane, so the lane is in bounds and its payload is a full
`(Sigma, key)` match.  The in-bounds fact is free: it is the bound of the generated read that
returned the slot. -/
theorem generated_tail_some_full_match {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (region : table.Region R V) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (home bound mask : Usize) (wanted : U64)
    (slot : Usize) (hMatchSlot : ∀ (ss : table.SpillSlot R V),
      ∃ matched : Bool, table.matches keyRep ss sigma key
        { place := place, direct_word := direct } = ok matched)
    (hreads : TailReadSuccess keyRep region sigma place direct key home bound mask wanted 1#usize)
    (hresult : table.probe.spill_read_tail_loop keyRep region sigma place direct key home
      bound mask wanted 1#usize = ok (some slot)) :
    slot.val < region.slots.length ∧
      ∃ (entry : table.SpillSlot R V),
        alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
          region.slots slot = ok entry ∧
        region.slots.val[slot.val]? = some entry ∧
        table.matches keyRep entry sigma key { place := place, direct_word := direct } =
          ok true := by
  rcases generated_tail_sweep_outcome keyRep region sigma place direct key home bound mask
    wanted 1#usize (some slot) hreads.coord hreads.slot (fun _ ss _ => hMatchSlot ss) hresult with
    ⟨_, hnone, _, _⟩ | ⟨_, _, lane, ss, hsome, _, _, _, hss, hmt⟩
  · cases hnone
  · have hEq : slot = lane := Option.some.inj hsome
    rw [← hEq] at hss
    obtain ⟨hinB, hreadB⟩ := vec_index_readback region.slots slot ss hss
    exact ⟨hinB, ss, hss, hreadB, hmt⟩

/-- **Deliverable 2 at the stored carrier.**  A returned lane's payload is a complete
`(Sigma, key)` match for the query. -/
theorem stored_tail_some_full_match
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64)
    (home bound mask : Usize) (wanted : U64) (slot : Usize)
    (hreads : TailReadSuccess Apeiron.GrowableIdentity.storedKeyRep region sigma place direct
      key home bound mask wanted 1#usize)
    (hresult : table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep region
      sigma place direct key home bound mask wanted 1#usize = ok (some slot)) :
    slot.val < region.slots.length ∧
      ∃ (entry : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64),
        region.slots.val[slot.val]? = some entry ∧
        Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held = (sigma, key) := by
  obtain ⟨hinB, ss, hindex, hread, hmt⟩ :=
    generated_tail_some_full_match Apeiron.GrowableIdentity.storedKeyRep region sigma place
      direct key home bound mask wanted slot
      (fun ss => stored_matcher_total ss sigma key { place := place, direct_word := direct })
      hreads hresult
  obtain ⟨_, hread', hlogical⟩ := stored_candidate_full_match region.slots slot ss sigma key
    { place := place, direct_word := direct } hindex hmt
  exact ⟨hinB, ss, hread', hlogical⟩

/-! ## 5 · Deliverable 1: a `none` result excludes the lanes it observed -/

/-- **An observed lane excludes the live slots at that lane.**  Under the live coordinate law an
occupied slot at the lane whose coordinate word is the wanted coordinate is a full `(Sigma, key)`
match; so a lane whose recorded observation either refused the match or held a different
coordinate word holds no live full match. -/
theorem observed_lane_excludes_live_full_match
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (place : U64) (direct : Option U32) (wanted : U64) (lane : Usize)
    (hobs : TailLaneObserved Apeiron.GrowableIdentity.storedKeyRep region cell.Sigma.Parabolic
      place direct 0#i64 lane wanted)
    (hlivecoord : ∀ (slot : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64),
      region.slots.val[lane.val]? = some slot → slot.tag ≠ 0#u8 →
      region.coords.val[lane.val]? = some wanted) :
    ∀ (slot : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64),
      region.slots.val[lane.val]? = some slot → slot.tag ≠ 0#u8 →
      table.matches Apeiron.GrowableIdentity.storedKeyRep slot cell.Sigma.Parabolic 0#i64
        { place := place, direct_word := direct } ≠ ok true := by
  obtain ⟨coord, hcoord, hzero, hexcl⟩ := hobs
  intro slot hslot hlive
  have hcoordEq : coord = wanted := by
    obtain ⟨_, hreadB⟩ := vec_index_readback region.coords lane coord hcoord
    have hliveRead := hlivecoord slot hslot hlive
    rw [hreadB] at hliveRead
    exact Option.some.inj hliveRead
  rcases hexcl with hne | ⟨ss, hss, hmt⟩
  · exact False.elim (hne hcoordEq)
  · obtain ⟨_, hssB⟩ := vec_index_readback region.slots lane ss hss
    have hssEq : ss = slot := by
      rw [hslot] at hssB
      exact (Option.some.inj hssB).symm
    rw [hssEq] at hmt
    cases hcoordEq
    exact fun hm => by
      have hcontra : (false : Bool) = true := Result.ok_injective (hmt.symm.trans hm)
      exact absurd hcontra (by simp)

/-! ## 6 · The generated `spill_read` admission chain

`generated_spill_read_branch` restates the retired `generated_spill_read_unfold`.  Its premise
list changed because the retired one carried the grouped-sizing equality
`region.slots.length = groups.val * n.val` and a power-of-two mask over `groups - 1`, neither of
which the current body computes: the current body admits on `len region.slots ≠ 0` and
`table.valid_n n`, and the sweep's bound is `alloc.vec.Vec.len region.slots` itself.

A `none` result has two distinct sources and the statement keeps them apart: the vacancy marker
already sitting at the home lane, and a bounded sweep that ran and returned `none`.  A `some home`
result is a full-key match at the home lane.
-/

/-- **The generated `spill_read` admission chain and its two exits.** -/
theorem generated_spill_read_branch
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : I64) (n : Usize)
    (home : Usize) (wanted : U64) (found : U64) (result : Option Usize)
    (hnonempty : alloc.vec.Vec.len region.slots ≠ 0#usize)
    (hvalid : table.valid_n n = ok true)
    (hhome : table.Region.home region address.place = ok home)
    (hwanted : table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place =
      ok wanted)
    (hfound : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
      region.coords home = ok found)
    (hhomeSlot : found ≠ table.COORD_EMPTY → found = wanted →
      ∃ ss : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64, alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice
          (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home = ok ss)
    (hhomeMatch : found ≠ table.COORD_EMPTY → found = wanted →
      ∀ ss : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64,
        alloc.vec.Vec.index
          (core.slice.index.SliceIndexUsizeSlice
            (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home =
          ok ss →
        ∃ matched : Bool,
          table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address = ok matched)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma
      address key n = ok result) :
    (result = none ∧ found = table.COORD_EMPTY) ∨
    (∃ ss : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64,
        alloc.vec.Vec.index
          (core.slice.index.SliceIndexUsizeSlice
            (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home =
          ok ss ∧
        table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address = ok true ∧
        result = some home) ∨
    (∃ (bound mask : Usize), table.Region.bound region = ok bound ∧
        bound - 1#usize = ok mask ∧
        table.probe.spill_read_tail_loop Apeiron.GrowableIdentity.storedKeyRep region sigma
          address.place address.direct_word key home bound mask wanted 1#usize = ok result) := by
  have hneZeroB : (alloc.vec.Vec.len region.slots != 0#usize) = true := by
    simpa using hnonempty
  unfold table.probe.spill_read at hresult
  rw [if_pos hneZeroB, hvalid, bind_tc_ok, hhome, bind_tc_ok, hwanted, bind_tc_ok, hfound,
    bind_tc_ok] at hresult
  by_cases hzero : found = table.COORD_EMPTY
  · rw [if_pos hzero, if_pos rfl] at hresult
    exact Or.inl ⟨Result.ok_injective hresult.symm, hzero⟩
  · rw [if_neg hzero] at hresult
    by_cases hmatch : found = wanted
    · rw [if_pos hmatch] at hresult
      obtain ⟨ss, hss⟩ := hhomeSlot hzero hmatch
      rw [hss, bind_tc_ok] at hresult
      obtain ⟨matched, hmt⟩ := hhomeMatch hzero hmatch ss hss
      rw [hmt, bind_tc_ok] at hresult
      by_cases hb : matched = true
      · rw [if_pos hb] at hresult
        refine Or.inr (Or.inl ⟨ss, hss, hmt.trans (congrArg ok hb), ?_⟩)
        simp at hresult
        first
        | exact hresult
        | exact hresult.symm
      · rw [if_neg hb] at hresult
        simp at hresult
        refine Or.inr (Or.inr ?_)
        unfold table.probe.spill_read_tail at hresult
        have hbound : table.Region.bound region =
            ok (alloc.vec.Vec.len region.slots) := rfl
        rw [hbound, bind_tc_ok] at hresult
        cases hsub : (alloc.vec.Vec.len region.slots - 1#usize) with
        | vis eff k =>
            rw [hsub, bind_tc_vis] at hresult
            exact (vis_not_ok hresult).elim
        | div =>
            rw [hsub, bind_tc_div] at hresult
            exact (div_not_ok hresult).elim
        | ret mask =>
            rw [hsub, bind_tc_ok, hwanted, bind_tc_ok] at hresult
            refine ⟨alloc.vec.Vec.len region.slots, mask, hbound, hsub, hresult⟩
    · rw [if_neg hmatch] at hresult
      simp at hresult
      refine Or.inr (Or.inr ?_)
      unfold table.probe.spill_read_tail at hresult
      have hbound : table.Region.bound region =
          ok (alloc.vec.Vec.len region.slots) := rfl
      rw [hbound, bind_tc_ok] at hresult
      cases hsub : (alloc.vec.Vec.len region.slots - 1#usize) with
      | vis eff k =>
          rw [hsub, bind_tc_vis] at hresult
          exact (vis_not_ok hresult).elim
      | div =>
          rw [hsub, bind_tc_div] at hresult
          exact (div_not_ok hresult).elim
      | ret mask =>
          rw [hsub, bind_tc_ok, hwanted, bind_tc_ok] at hresult
          refine ⟨alloc.vec.Vec.len region.slots, mask, hbound, hsub, hresult⟩

/-! ## 7 · Deliverable 2 at the generated read -/

/-- **Deliverable 2 at `table.probe.spill_read`.**  A generated read that returned `some slot`
returned a slot that is in bounds and whose payload is a full `(Sigma, key)` match for the query.
Both live exits of `generated_spill_read_branch` deliver that, so the conclusion is stated once.
No state invariant is assumed. -/
theorem spill_read_some_implies_full_match
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : I64) (n : Usize)
    (home : Usize) (wanted : U64) (found : U64) (slot : Usize)
    (hnonempty : alloc.vec.Vec.len region.slots ≠ 0#usize)
    (hvalid : table.valid_n n = ok true)
    (hhome : table.Region.home region address.place = ok home)
    (hwanted : table.Region.coord_of Apeiron.GrowableIdentity.StoredI64 I64 address.place =
      ok wanted)
    (hfound : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
      region.coords home = ok found)
    (hhomeSlot : found ≠ table.COORD_EMPTY → found = wanted →
      ∃ ss : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64, alloc.vec.Vec.index
        (core.slice.index.SliceIndexUsizeSlice
          (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home = ok ss)
    (hhomeMatch : found ≠ table.COORD_EMPTY → found = wanted →
      ∀ ss : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64,
        alloc.vec.Vec.index
          (core.slice.index.SliceIndexUsizeSlice
            (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64)) region.slots home =
          ok ss →
        ∃ matched : Bool,
          table.matches Apeiron.GrowableIdentity.storedKeyRep ss sigma key address = ok matched)
    (hreads : ∀ (bound mask : Usize), table.Region.bound region = ok bound →
      bound - 1#usize = ok mask →
      TailReadSuccess Apeiron.GrowableIdentity.storedKeyRep region sigma address.place
        address.direct_word key home bound mask wanted 1#usize)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma
      address key n = ok (some slot)) :
    slot.val < region.slots.length ∧
      ∃ (entry : table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64),
        region.slots.val[slot.val]? = some entry ∧
        Apeiron.GrowableIdentity.logicalStored entry.sigma entry.held = (sigma, key) := by
  rcases generated_spill_read_branch region sigma address key n home wanted found (some slot)
    hnonempty hvalid hhome hwanted hfound hhomeSlot hhomeMatch hresult with
    ⟨hnone, hzero⟩ | ⟨ss, hss, hmt, hslotEq⟩ | ⟨bound, mask, hbound, hmask, hsweep⟩
  · cases hnone
  · have hEq : slot = home := Option.some.inj hslotEq
    rw [← hEq] at hss
    obtain ⟨hinB, hreadB, hlogical⟩ :=
      stored_candidate_full_match region.slots slot ss sigma key address hss hmt
    exact ⟨hinB, ss, hreadB, hlogical⟩
  · obtain ⟨hinB, entry, hread, hlogical⟩ :=
      stored_tail_some_full_match region sigma address.place address.direct_word key home bound
        mask wanted slot (hreads bound mask hbound hmask) hsweep
    exact ⟨hinB, entry, hread, hlogical⟩

/-! ## 8 · Fixtures against the current `{slots, coords, bits, live}` schema

Capacity four, coordinate words `0` (the wanted coordinate of place `0`) and `table.COORD_EMPTY`.
`Region.home` of place `0` is lane `0`, so the home probe is the first coordinate read and the tail
cycle starts at lane `1`.
-/

private def fixtureSlots (t0 : U8) (k0 : I64) : alloc.vec.Vec
    (table.SpillSlot Apeiron.GrowableIdentity.StoredI64 I64) :=
  alloc.vec.Vec.from
    ([{ tag := t0, sigma := cell.Sigma.Parabolic, held := { key := k0 }, value := 0#i64 },
      { tag := 128#u8, sigma := cell.Sigma.Parabolic, held := { key := 0#i64 }, value := 0#i64 },
      { tag := 0#u8, sigma := cell.Sigma.Parabolic, held := { key := 0#i64 }, value := 0#i64 },
      { tag := 0#u8, sigma := cell.Sigma.Parabolic, held := { key := 0#i64 }, value := 0#i64 }])
    (by scalar_tac)

private def fixtureCoords : alloc.vec.Vec U64 :=
  alloc.vec.Vec.from ([0#u64, 0#u64, table.COORD_EMPTY, table.COORD_EMPTY]) (by scalar_tac)

/-- The matching entry sits at the home lane, so the home probe answers it. -/
private def homeHitRegion : table.Region Apeiron.GrowableIdentity.StoredI64 I64 :=
  { slots := fixtureSlots 128#u8 0#i64, coords := fixtureCoords, bits := 0#u32, live := 1#usize }

/-- The matching entry sits one lane past a *different* live key, so only the tail cycle finds it. -/
private def tailHitRegion : table.Region Apeiron.GrowableIdentity.StoredI64 I64 :=
  { slots := fixtureSlots 128#u8 7#i64, coords := fixtureCoords, bits := 0#u32, live := 2#usize }

-- The home probe answers the query from the home lane.
#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep homeHitRegion
    cell.Sigma.Parabolic { place := 0#u64, direct_word := none } 0#i64 1#usize).match with
  | .ok (some slot) => slot == 0#usize
  | _ => false

-- The same region answers `none` for a key no live entry holds, and it does so by the vacancy
-- terminator at lane two: the cycle stops before reaching the bound.  This is the outcome the
-- retired `WINDOWS + groups` sweep had no denotation for.
#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep homeHitRegion
    cell.Sigma.Parabolic { place := 0#u64, direct_word := none } 1#i64 1#usize).match with
  | .ok none => true
  | _ => false

-- A live entry that is *not* the query's key does not stop the tail cycle: it moves to the next
-- lane and finds the query there.  The sweep must treat a non-matching identity guard as `cont`,
-- not as exhaustion.
#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep tailHitRegion
    cell.Sigma.Parabolic { place := 0#u64, direct_word := none } 0#i64 1#usize).match with
  | .ok (some slot) => slot == 1#usize
  | _ => false

-- An entry past the vacancy is unreachable: the same region returns `none` for the query of lane
-- one, because lane two already terminated the cycle.
#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep tailHitRegion
    cell.Sigma.Parabolic { place := 0#u64, direct_word := none } 1#i64 1#usize).match with
  | .ok none => true
  | _ => false

-- A vacancy already at the home lane answers `none` without reading the sweep.
#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep
    { slots := fixtureSlots 0#u8 0#i64
      coords := alloc.vec.Vec.from
        (List.replicate 4 table.COORD_EMPTY) (by scalar_tac)
      bits := 0#u32
      live := 0#usize }
    cell.Sigma.Parabolic { place := 0#u64, direct_word := none } 0#i64 1#usize).match with
  | .ok none => true
  | _ => false

-- A generated publication writes the generated tag and the coordinate word of the published place,
-- and counts the lane live. This is the producer of the live coordinate law of section 3.
#guard match (table.Region.publish_live homeHitRegion 3#usize
    (cell.Sigma.Parabolic, { key := 5#i64 }, 7#i64) 0#u64).match with
  | .ok published =>
      match (table.tag 0#u64).match with
      | .ok expected =>
          published.live.val == 2 && published.coords.val[3]? == some 0#u64 &&
            published.slots.val[3]?.map (fun slot => slot.tag) == some expected
      | _ => false
  | _ => false

-- Local fail-closed cone check until the integration owner registers these declarations in
-- AxiomAudit.lean. The shared audit file is outside this task's ownership.
open Lean Elab Command in
run_cmd do
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  let external : Array Lean.Name := #[``Apeiron.GrowableScan.spill_cycle_slot_lt,
    ``Apeiron.GrowableScan.spill_cycle_covers,
    ``Apeiron.GrowableScan.spill_cycle_covers_of_placement,
    ``Apeiron.GrowableIdentity.stored_matches_iff]
  let localNames : Array Lean.Name := #[`generated_tail_increment_checked,
    `generated_tail_stops_at_bound, `generated_tail_sweep_spec,
    `generated_tail_sweep_outcome, `TailLaneObserved,
    `TailSweepInv, `TailSweepOutcome, `StoredLiveCoordAddress, `stored_matcher_total,
    `stored_candidate_full_match, `generated_tail_some_full_match, `TailReadSuccess,
    `stored_tail_some_full_match, `observed_lane_excludes_live_full_match,
    `generated_spill_read_branch, `spill_read_some_implies_full_match]
  for decl in external ++ localNames.map (Lean.Name.append `Apeiron.GrowableReadLoop) do
    let axioms ← liftTermElabM <| Lean.collectAxioms decl
    let bad := axioms.filter (fun ax => !allowed.contains ax)
    unless bad.isEmpty do
      throwError m!"'{decl}' depends on axioms outside the closed cone: {bad.toList}"
    logInfo m!"'{decl}' depends on axioms: {axioms.toList}"

end Apeiron.GrowableReadLoop