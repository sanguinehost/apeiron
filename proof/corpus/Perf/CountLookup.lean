import Mathlib.Tactic

/-!
# Count lookup: the overlap ceiling and the tag-collision count

Two general theorems behind the count-lookup cost model
 and the lever admission rule, plus their
instances at the measured constants of P74.

* `overlap_gain`: when a lookup's time is `max a b`, with `a` and `b` two overlapped components (the
  form `Perf.Overlap.overlap_full_of_disjoint` gives for disjoint phases),
  relieving `a` by `d` cycles gains exactly `min d (a - b)`. A counter's active cycles are additive
  only up to the component's slack over the other.
* `lever_closes_only_if`: a lever closes a gap `g` only if both the relieved amount and the slack are
  at least `g`.
* `tag_collisions_sum`: over all `n ^ L` assignments of `L` tags drawn from `n` values, the number of
  lanes whose tag equals a fixed `t` sums to `L * n ^ (L - 1)`. The mean number of tag matches ahead
  of the true lane is therefore `L / n`; this is BEAT-18's `mean_lane / 128` for 7-bit tags. Uniform,
  independent tags are the premise; dense and high-bits keys break it through `f * PHI`
 .

The P74 constants are `model` inputs in tenths of a cycle per lookup, copied from
`~/count-model/p74ab.txt` (medians of 5 interleaved pairs, i9-14900KF core 4). No
theorem here produces a cycle: the `max` form is a model of the lookup, not a theorem about the CPU.
rustc/LLVM codegen, the microarchitecture and the page-walk hardware are outside every statement.
-/

namespace Perf

namespace CountLookup

/-! ## 1 · The overlap ceiling -/

/-- **Overlap gain.** Relieving the overlapped component `a` of `max a b` by `d` lowers the maximum by
exactly `min d (a - b)`, in truncated `Nat` subtraction. -/
theorem overlap_gain (a b d : ℕ) : max a b - max (a - d) b = min d (a - b) := by
  omega

/-- **The drop identifies the slack.** If relieving `d` gains `g < d`, the relieved component was
critical by exactly `g`. -/
theorem slack_of_partial_gain {a b d g : ℕ} (h : max a b - max (a - d) b = g) (hg : g < d) :
    a - b = g := by
  rw [overlap_gain] at h
  omega

/-- **Lever admission.** A lever that relieves `d` cycles of the overlapped component `a` closes a
gap `gap` only if `d ≥ gap` and the slack `a - b ≥ gap`. A lever whose relieved amount is below the
cell's gap is refused before dispatch. -/
theorem lever_closes_only_if {a b d gap : ℕ} (h : gap ≤ max a b - max (a - d) b) :
    gap ≤ d ∧ gap ≤ a - b := by
  rw [overlap_gain] at h
  omega

/-! ## 2 · Tag collisions -/

/-- **Tag-collision count.** Over all assignments `v : Fin L → Fin n`, the lanes whose tag equals `t`
number `L * n ^ (L - 1)` in total. -/
theorem tag_collisions_sum (L n : ℕ) (t : Fin n) :
    ∑ v : Fin L → Fin n, (Finset.univ.filter (fun i => v i = t)).card = L * n ^ (L - 1) := by
  classical
  calc
    ∑ v : Fin L → Fin n, (Finset.univ.filter (fun i => v i = t)).card
        = ∑ v : Fin L → Fin n, ∑ i : Fin L, if v i = t then 1 else 0 :=
          Finset.sum_congr rfl fun v _ => Finset.card_filter _ _
    _ = ∑ i : Fin L, ∑ v : Fin L → Fin n, if v i = t then 1 else 0 := Finset.sum_comm
    _ = ∑ i : Fin L, (Finset.univ.filter (fun v : Fin L → Fin n => v i = t)).card :=
          Finset.sum_congr rfl fun i _ => (Finset.card_filter _ _).symm
    _ = ∑ _i : Fin L, n ^ (L - 1) := by
          refine Finset.sum_congr rfl fun i _ => ?_
          have h := Fintype.card_filter_piFinset_const_eq_of_mem
            (Finset.univ : Finset (Fin n)) i (Finset.mem_univ t)
          simpa [Fintype.piFinset_univ] using h
    _ = L * n ^ (L - 1) := by simp

/-! ## 3 · P74 at the measured constants (`model`) -/

/-- P74 baseline `dtlb_load_misses.walk_active` at `1e7 all` dense / high-bits / random, tenths of a
cycle per lookup. `model`, `p74ab.txt`. -/
def p74Walk : List ℕ := [498, 466, 460]

/-- P74 measured cycle drop with THP at the same cells, tenths of a cycle per lookup. `model`. -/
def p74Drop : List ℕ := [147, 66, 113]

/-- P74 baseline gap `ap_par - hb_fx` at the same cells, tenths of a cycle per lookup
(113.6 - 56.2, 105.1 - 56.7, 103.4 - 63.9). `model`. -/
def p74Gap : List ℕ := [574, 484, 395]

/-- **P74 clause (2) refuted.** At every `1e7 all` cell the drop is below half the relieved walk
cycles, the registered floor. With `slack_of_partial_gain`, the `max` model reads each drop as the
walk component's slack: 14.7, 6.6 and 11.3 cycles. -/
theorem p74_clause2_refuted :
    List.Forall₂ (fun d w => 2 * d < w) p74Drop p74Walk := by decide

/-- **P74 admission, before the run.** At `1e7 all` dense and high-bits the whole walk is below the
gap to `hb_fx`, so by `lever_closes_only_if` removing it cannot close either cell; at random it is not
excluded by this bound. -/
theorem p74_walk_cannot_close :
    p74Walk[0]! < p74Gap[0]! ∧ p74Walk[1]! < p74Gap[1]! ∧ p74Gap[2]! ≤ p74Walk[2]! := by decide

-- Non-vacuity: the overlap gain at a critical, a partly critical and a hidden component.
#guard max 100 40 - max (100 - 30) 40 = 30
#guard max 100 90 - max (100 - 30) 90 = 10
#guard max 50 90 - max (50 - 30) 90 = 0

/-! ## 4 · P75: the registered rule, applied

P75 registered two hypotheses and a bar: a hypothesis holds when the median of 5 interleaved pairs is
within ±15 % in at least 22 of 27 cells, and if neither holds the header is not the cause. The counts
are those of the perf-predictions note §P75 result (`model` inputs). -/

/-- The registered bar: cells (of 27) that must agree. -/
def p75Bar : ℕ := 22

/-- Cells (of 27) in which H-W (`ap_flat` = `ap_par` × 33/42) was within ±15 %. -/
def p75HW : ℕ := 21

/-- Cells (of 27) in which H-S (`ap_flat` = `hb_fx` at the same working set) was within ±15 %. -/
def p75HS : ℕ := 5

/-- **P75 refuted by its own rule.** Neither hypothesis reaches the bar, so the header is not the cause.
H-W misses by one cell. -/
theorem p75_refuted : ¬ p75Bar ≤ p75HW ∧ ¬ p75Bar ≤ p75HS ∧ p75HW + 1 = p75Bar := by decide

/-! ## 5 · P76: the mispredict clause

One row per excess cell, in the order 1e5 10 % dense, 1e6 1 % dense, 1e6 all dense, 1e7 10 % dense,
1e7 all high-bits. Mispredicts (`br_misp_retired.all_branches`) and retired instructions
(`inst_retired.any`) per lookup, in thousandths. `model` inputs, `~/count-header/p76.txt`,
medians of 5 interleaved rounds, `ap_par` and `ap_flat`. -/

/-- One P76 cell: `ap_par` and `ap_flat` mispredicts per lookup, `ap_par` retired instructions. -/
structure P76Cell where
  parMisp : ℕ
  flatMisp : ℕ
  parInst : ℕ

def p76Cells : List P76Cell :=
  [⟨106, 12, 56555⟩, ⟨108, 12, 56542⟩, ⟨148, 11, 56274⟩, ⟨148, 11, 56274⟩, ⟨113, 3, 55982⟩]

/-- `ap_par` retired instructions per lookup in the controls 1e5 all dense and 1e6 10 % dense. -/
def p76CtrlInst : List ℕ := [55001, 55003]

/-- **P76 clause (1) refuted.** The `ap_par` − `ap_flat` mispredict difference is 0.094–0.137 per
lookup in every excess cell, below the registered band [0.30, 0.70]. -/
theorem p76_clause1_refuted :
    ∀ c ∈ p76Cells, c.flatMisp ≤ c.parMisp ∧ c.parMisp - c.flatMisp < 300 := by decide

/-- **P76 clause (2) holds.** Retired instructions in the excess cells are within ±10 % of the controls. -/
theorem p76_clause2 :
    ∀ c ∈ p76Cells, ∀ k ∈ p76CtrlInst, c.parInst * 10 ≤ k * 11 ∧ k * 9 ≤ c.parInst * 10 := by decide

/-- **P76 clauses (3) and (4) hold.** `ap_par` mispredicts at most 0.10 per lookup in the controls
(measured 0.000, 0.000) and `ap_flat` in all 7 cells. -/
theorem p76_clause34 :
    (∀ m ∈ ([0, 0] : List ℕ), m ≤ 100) ∧ (∀ c ∈ p76Cells, c.flatMisp ≤ 100) ∧
      (∀ m ∈ ([0, 0] : List ℕ), m ≤ 100) := by decide

/-! ## 6 · P77: issued against retired uops

Per lookup, in thousandths, same cells and order. `model` inputs,
`~/count-header/p77.txt`, medians of 5 interleaved rounds. `Iss` = `uops_issued.any`,
`Ret` = `uops_retired.slots`, `Cyc` = `cycles`, `Misp` = `br_misp_retired.all_branches`;
`machine_clears.count` was 0.000 in all 14 arm-cells. -/

/-- One P77 cell. -/
structure P77Cell where
  parIss : ℕ
  parRet : ℕ
  flatIss : ℕ
  flatRet : ℕ
  parMisp : ℕ
  parCyc : ℕ
  flatCyc : ℕ

def p77Cells : List P77Cell :=
  [⟨88562, 53398, 45522, 42447, 102, 20836, 9712⟩,
   ⟨90706, 53376, 45594, 42375, 108, 21259, 9324⟩,
   ⟨102178, 53139, 46375, 42138, 147, 61758, 43786⟩,
   ⟨102104, 53140, 46481, 42138, 147, 60081, 42493⟩,
   ⟨93178, 52883, 43041, 41880, 113, 101429, 69834⟩]

/-- The two controls (1e5 all dense, 1e6 10 % dense): `ap_par` issued and retired, `ap_flat` issued
and retired. -/
def p77Ctrl : List (ℕ × ℕ × ℕ × ℕ) := [(52007, 52003, 41006, 41003), (52013, 52009, 41006, 41003)]

/-- **P77 clause (1) holds.** `ap_par` retired uops in the excess cells are within ±10 % of the controls. -/
theorem p77_clause1 :
    ∀ c ∈ p77Cells, ∀ k ∈ p77Ctrl, c.parRet * 10 ≤ k.2.1 * 11 ∧ k.2.1 * 9 ≤ c.parRet * 10 := by
  decide

/-- **P77 clause (2) holds.** Issued − retired is at least 30 uops per lookup for `ap_par` in every
excess cell, at most 2 in the controls, and at most 6 for `ap_flat` everywhere. -/
theorem p77_clause2 :
    (∀ c ∈ p77Cells, c.parRet ≤ c.parIss ∧ 30000 ≤ c.parIss - c.parRet) ∧
    (∀ k ∈ p77Ctrl, k.2.1 ≤ k.1 ∧ k.1 - k.2.1 ≤ 2000) ∧
    (∀ c ∈ p77Cells, c.flatRet ≤ c.flatIss ∧ c.flatIss - c.flatRet ≤ 6000) ∧
    (∀ k ∈ p77Ctrl, k.2.2.2 ≤ k.2.2.1 ∧ k.2.2.1 - k.2.2.2 ≤ 6000) := by decide

/-- **P77 clause (3) holds.** `machine_clears.count` per lookup is below 0.02 in all 14 arm-cells. -/
theorem p77_clause3 : ∀ m ∈ List.replicate 14 (0 : ℕ), m < 20 := by decide

/-- **P77 clause (4) holds.** At least 250 non-retired uops per mispredict for `ap_par`. -/
theorem p77_clause4 : ∀ c ∈ p77Cells, 250 * c.parMisp ≤ c.parIss - c.parRet := by decide

/-- Issue-width input of the cycle model (`model`, a design assumption, not a measurement). -/
def p77Width : ℕ := 6

/-- **A lever that removes `d` cycles of throughput gains at most `d`.** Whatever the slack. -/
theorem lever_gain_le (a b d : ℕ) : max a b - max (a - d) b ≤ d := by
  rw [overlap_gain]; exact Nat.min_le_left _ _

/-- **A lever relieving fewer cycles than a gap cannot close it.** -/
theorem lever_cannot_close {a b d gap : ℕ} (h : d < gap) : ¬ gap ≤ max a b - max (a - d) b :=
  fun hg => absurd (lever_closes_only_if hg).1 (by omega)

/-- **The retired-uop difference is under half of the `ap_par` − `ap_flat` cycle gap.** At the issue
width the retired-uop difference is worth at most `(parRet − flatRet) / 6` cycles; the cycle gap is more
than twice that in every excess cell. Under the `max` model the rest is latency. -/
theorem p77_uop_difference_under_half_gap :
    ∀ c ∈ p77Cells, (c.parRet - c.flatRet) / p77Width * 2 < c.parCyc - c.flatCyc := by decide

/-- **No lever that removes the retired-uop difference closes the cycle gap to `ap_flat`.** -/
theorem p77_retired_uop_lever_cannot_close (c : P77Cell) (hc : c ∈ p77Cells) (a b : ℕ) :
    ¬ c.parCyc - c.flatCyc ≤ max a b - max (a - (c.parRet - c.flatRet) / p77Width) b := by
  have h := p77_uop_difference_under_half_gap c hc
  exact lever_cannot_close (by omega)

end CountLookup

end Perf
