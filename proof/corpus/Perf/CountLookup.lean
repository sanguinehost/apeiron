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

end CountLookup

end Perf
