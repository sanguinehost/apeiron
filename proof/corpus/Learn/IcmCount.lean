import Mathlib
import ApeironWord
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Learn.IcmError

/-!
# The per-entry `sub_q` step count of `sweep_rows`

`src/learn/fold/fit/icm.rs` `sweep_rows` runs `sweeps` passes over `nb = n / BLOCK` blocks (`:909`,
`:921-923`), `BLOCK = 8` (`:46`). Inside one block:

* the row loop `:960-1023` visits each row once and appends it to `moved_row[0..moved_n]` when its
  block code changed (`if changed`, `:1017-1021`), so the block's moved rows are a set — a row
  appears at most once;
* for each moved row `i` the residual loop `:1040-1047` runs `col < n`, applying exactly one
  `sub_q` to each entry `(row, col)` of that row.

So an entry `(row, col)` receives at most one `sub_q` per block, hence at most `nb` per sweep and at
most `sweeps · nb` over the pass (`stepCount_sweeps_le`). The bound is `≤`, not `=`: `changed`
`:1014-1016` fires only on a code change, so a converged row costs no step (`stepCount_eq_zero`); it
is attained when a row moves in every block (`stepCount_sweeps_eq`), so it is the exact maximum of
the move structure.

At the Down witness (`n = 4864`, `sweeps = 2`) `nb = 4864 / 8 = 608` and the maximum is
`2 · 608 = 1216` (`down_blocks`, `down_step_count`), the count the margin instantiates at
(`word_argmin_cost_bound_end2end_derived_Ea35_count_down`). The measured average is `524.7` moves
per entry per sweep, below the maximum, so the measured count is covered by the structural bound and
is not equal to it.

## Boundary

This counts the loop's move structure; it is not a refinement of the extracted `sweep_rows` body. The
model hypotheses are the two readings above: `nb = n / BLOCK` blocks per sweep, and one `sub_q` per
entry per block. `Ea ≤ 35` remains a measured hypothesis (`Learn.IcmError.limb_error_chg_Ea35`); the
`Ea ≤ 36` chain there is the input-general fallback. The gap between the chain's margin and the
measured `1.75e-7` is itself a theorem (`chain_margin_gt_measured`): the chain's bound is a
sufficient condition and does not certify the measured margin.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

noncomputable section

namespace Learn.IcmCount

variable {ι : Type} [DecidableEq ι]

/-- One block's moved rows: `moved_row[0..moved_n]` (`icm.rs:1017-1021`). The row loop visits each
row once, so a row appears at most once and the `Finset` is the exact image. -/
abbrev Moves (ι : Type) [DecidableEq ι] (blocks : ℕ) : Type := Fin blocks → Finset ι

/-- **The per-entry step count of one sweep.** The entry `(row, col)` is updated once for each block
whose move set contains `row`: `icm.rs:1040-1047` updates all `n` columns of a moved row. -/
def stepCount {blocks : ℕ} (moved : Moves ι blocks) (row : ι) : ℕ :=
  (Finset.univ.filter (fun b => row ∈ moved b)).card

/-- **One sweep: at most one `sub_q` per entry per block.** -/
theorem stepCount_le_blocks {blocks : ℕ} (moved : Moves ι blocks) (row : ι) :
    stepCount moved row ≤ blocks :=
  (Finset.card_filter_le _ _).trans_eq (by simp)

/-- **The per-entry count over `sweeps` sweeps is at most `sweeps · blocks`.** -/
theorem stepCount_sweeps_le {sweeps blocks : ℕ} (moved : Fin sweeps → Moves ι blocks) (row : ι) :
    (∑ s, stepCount (moved s) row) ≤ sweeps * blocks := by
  calc (∑ s, stepCount (moved s) row) ≤ ∑ _s : Fin sweeps, blocks :=
        Finset.sum_le_sum (fun s _ => stepCount_le_blocks (moved s) row)
    _ = sweeps * blocks := by
        simp [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]

/-- A row that never moves costs no step: `changed` is false in every block. -/
theorem stepCount_eq_zero {blocks : ℕ} (moved : Moves ι blocks) (row : ι)
    (h : ∀ b, row ∉ moved b) : stepCount moved row = 0 := by
  unfold stepCount
  rw [Finset.filter_false_of_mem (fun b _ => h b), Finset.card_empty]

/-- The bound is the exact maximum of the move structure: a row that moves in every block reaches
`sweeps · blocks`. -/
theorem stepCount_sweeps_eq {sweeps blocks : ℕ} (moved : Fin sweeps → Moves ι blocks) (row : ι)
    (h : ∀ s b, row ∈ moved s b) :
    (∑ s, stepCount (moved s) row) = sweeps * blocks := by
  have hone : ∀ s, stepCount (moved s) row = blocks := by
    intro s
    unfold stepCount
    rw [Finset.filter_true_of_mem (fun b _ => h s b), Finset.card_univ, Fintype.card_fin]
  simp [hone, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]

/-! ## The chain: count → `εf` → `εd` → margin -/

/-- `epsF35` is monotone in the step count, so the structural maximum bounds the actual count. -/
theorem epsF35_le_of_le {εres0 : ℝ} {N M : ℕ} (hNM : N ≤ M) :
    Learn.IcmError.epsF35 εres0 N ≤ Learn.IcmError.epsF35 εres0 M := by
  have hcast : (N : ℝ) ≤ (M : ℝ) := by exact_mod_cast hNM
  unfold Learn.IcmError.epsF35 Learn.IcmError.epsStep35
  nlinarith [hcast]

/-- `epsDEnd` is monotone in the `f_blk` error. -/
theorem epsDEnd_le_of_le {εf εg : ℝ} (h : εf ≤ εg) :
    Learn.IcmError.epsDEnd εf ≤ Learn.IcmError.epsDEnd εg := by
  have hc : (0 : ℝ) ≤ 8 * (12100000000 / 2 ^ 32) := by norm_num
  unfold Learn.IcmError.epsDEnd
  calc 24842 / 2 ^ 32 + 8 * εf * (12100000000 / 2 ^ 32)
      = 24842 / 2 ^ 32 + (8 * (12100000000 / 2 ^ 32)) * εf := by ring
    _ ≤ 24842 / 2 ^ 32 + (8 * (12100000000 / 2 ^ 32)) * εg := by
        have hm := mul_le_mul_of_nonneg_left h hc
        linarith
    _ = 24842 / 2 ^ 32 + 8 * εg * (12100000000 / 2 ^ 32) := by ring

/-- **The end-to-end bound at the actual per-entry count.** `N ≤ sweeps · blocks` from
`stepCount_sweeps_le`; `epsF35` and `epsDEnd` are monotone, so the margin at the structural maximum
bounds the decision at the actual count. Exact margin:
`3088700432769024/2^64 + sweeps·blocks·1205934400000000/2^64 + εres0·387200000000/2^32`. -/
theorem word_argmin_cost_bound_end2end_derived_Ea35_count
    (q d qh dh : Learn.FitIcm.CodeIndex → ℝ) (sweeps blocks N : ℕ) (εres0 : ℝ)
    (hN : N ≤ sweeps * blocks)
    (hq : ∀ c, |qh c - q c| ≤ Learn.IcmError.epsQEnd)
    (hd : ∀ c, |dh c - d c| ≤ Learn.IcmError.epsDEnd (Learn.IcmError.epsF35 εres0 N))
    (cstar cold : Learn.FitIcm.CodeIndex)
    (hmin : ∀ c, qh cstar - 2 * dh cstar ≤ qh c - 2 * dh c) :
    q cstar - 2 * d cstar
      ≤ q cold - 2 * d cold
        + (3088700432769024 / 2 ^ 64
            + ((sweeps * blocks : ℕ) : ℝ) * (1205934400000000 / 2 ^ 64)
            + εres0 * (387200000000 / 2 ^ 32)) := by
  have hmono : Learn.IcmError.epsF35 εres0 N ≤ Learn.IcmError.epsF35 εres0 (sweeps * blocks) :=
    epsF35_le_of_le hN
  have hd' : ∀ c, |dh c - d c|
      ≤ Learn.IcmError.epsDEnd (Learn.IcmError.epsF35 εres0 (sweeps * blocks)) :=
    fun c => (hd c).trans (epsDEnd_le_of_le hmono)
  exact Learn.IcmError.word_argmin_cost_bound_end2end_derived_Ea35 q d qh dh
    (sweeps * blocks) εres0 hq hd' cstar cold hmin

/-! ## The Down witness: `blocks = n / 8 = 608`, `N = 2 · 608 = 1216` -/

/-- `blocks = n / BLOCK` at the Down witness: `n = 4864`, `BLOCK = 8` (`icm.rs:46`, `:909`). -/
theorem down_blocks : 4864 / 8 = 608 := by norm_num

/-- The structural maximum per entry at the Down witness: `sweeps = 2`, `blocks = 608`. -/
theorem down_step_count : (2 : ℕ) * 608 = 1216 := by norm_num

/-- **The Down instantiation at the derived count.** `sweeps = 2`, `blocks = 608`: `N = 1216`, the
structural maximum (`stepCount_sweeps_le`, `down_step_count`), so the margin
`1469504930832769024/2^64 ≈ 7.966e-2` now comes from the loop structure rather than from a measured
count. The measured average is `524.7` moves per entry per sweep, below this maximum. -/
theorem word_argmin_cost_bound_end2end_derived_Ea35_count_down
    (q d qh dh : Learn.FitIcm.CodeIndex → ℝ)
    (hq : ∀ c, |qh c - q c| ≤ Learn.IcmError.epsQEnd)
    (hd : ∀ c, |dh c - d c| ≤ Learn.IcmError.epsDEnd (Learn.IcmError.epsF35 0 1216))
    (cstar cold : Learn.FitIcm.CodeIndex)
    (hmin : ∀ c, qh cstar - 2 * dh cstar ≤ qh c - 2 * dh c) :
    q cstar - 2 * d cstar ≤ q cold - 2 * d cold + 1469504930832769024 / 2 ^ 64 := by
  have h := word_argmin_cost_bound_end2end_derived_Ea35_count q d qh dh 2 608 1216 0
    (by norm_num) hq hd cstar cold hmin
  have hcast : ((2 * 608 : ℕ) : ℝ) = ((1216 : ℕ) : ℝ) := by norm_num
  rw [hcast, Learn.IcmError.derived_const_value_Ea35_N1216] at h
  exact h

/-! ## The chain's floor relative to the measured argmin margin

The measured argmin margin at the Down witness is `1.75e-7` (`175/10^9`). The chain's margin term is
`3088700432769024/2^64 ≈ 1.674388e-4` at `εres0 = 0`, `N = 0`, and grows by
`1205934400000000/2^64 ≈ 6.537383e-5` per step (`chain_increment_pos`). So every instantiation at
`N ≥ 1` has a margin above the measured value (`chain_margin_gt_measured`); at `N = 1216` it is
`1469504930832769024/2^64 ≈ 7.966202e-2` (`chain_margin_down_gt_measured`), `4.55212e5` times the
measured margin (`chain_margin_down_ratio_gt` proves the `4 · 10^5` form), and the base alone is
`956.79` times it (`chain_base_ratio_gt` proves the `900 ·` form).

Read correctly: the argmin decision is proved at the chain margin as a **sufficient** condition
(`word_argmin_cost_bound_end2end_derived_Ea35_count{,_down}`). The measured margin lying below that
margin means the chain does not certify the measured value, not that the decision is wrong. The
looseness is carried by the per-move kernel bound — `Ea ≤ 35` is measured, the step bound
`εe + ½/2^32` is a worst case over the whole block — and by `εres0`, which remains free. -/

/-- The per-step margin increment is positive, so the margin is strictly increasing in `N`. -/
theorem chain_increment_pos : (0 : ℝ) < 1205934400000000 / 2 ^ 64 := by norm_num

/-- The base margin (`εres0 = 0`, `N = 0`) alone exceeds the measured argmin margin `175/10^9`
(`3088700432769024/2^64 ≈ 1.674388e-4` against `1.75e-7`, `956.79` times it). -/
theorem chain_base_gt_measured : (175 : ℝ) / 10 ^ 9 < 3088700432769024 / 2 ^ 64 := by
  rw [div_lt_div_iff₀ (by norm_num : (0 : ℝ) < 10 ^ 9) (by norm_num : (0 : ℝ) < 2 ^ 64)]
  norm_num

/-- **The chain's margin floor.** For every `N ≥ 1` the chain margin exceeds the measured argmin
margin `175/10^9`, so no instantiation of the derived bound delivers a margin at the measured value:
the bound is a sufficient condition, not a tight one. -/
theorem chain_margin_gt_measured (N : ℕ) (hN : 1 ≤ N) :
    (175 : ℝ) / 10 ^ 9
      < 3088700432769024 / 2 ^ 64 + (N : ℝ) * (1205934400000000 / 2 ^ 64) := by
  have hNpos : (0 : ℝ) < (N : ℝ) := Nat.cast_pos.mpr (by omega)
  have hpos : (0 : ℝ) < (N : ℝ) * (1205934400000000 / 2 ^ 64) :=
    mul_pos hNpos chain_increment_pos
  linarith [chain_base_gt_measured]

/-- The Down margin term at `N = 1216`, `εres0 = 0`, as `word_argmin_cost_bound_end2end_derived_Ea35_count_down`
writes it: `3088700432769024/2^64 + 1216·1205934400000000/2^64`. -/
theorem chain_margin_down_eq :
    3088700432769024 / 2 ^ 64 + ((1216 : ℕ) : ℝ) * (1205934400000000 / 2 ^ 64)
      = 1469504930832769024 / 2 ^ 64 := by
  norm_num

/-- The Down instantiation's margin (`N = 1216`) exceeds the measured argmin margin. -/
theorem chain_margin_down_gt_measured :
    (175 : ℝ) / 10 ^ 9 < 1469504930832769024 / 2 ^ 64 := by
  have h := chain_margin_gt_measured 1216 (by norm_num)
  rw [chain_margin_down_eq] at h
  exact h

/-- The base margin exceeds `900 · 175/10^9`; the exact ratio is `956.79`. -/
theorem chain_base_ratio_gt :
    (900 : ℝ) * ((175 : ℝ) / 10 ^ 9) < 3088700432769024 / 2 ^ 64 := by
  have heq : (900 : ℝ) * ((175 : ℝ) / 10 ^ 9) = (900 * 175 : ℝ) / 10 ^ 9 := by ring
  rw [heq, div_lt_div_iff₀ (by norm_num : (0 : ℝ) < 10 ^ 9) (by norm_num : (0 : ℝ) < 2 ^ 64)]
  norm_num

/-- The Down margin exceeds `4 · 10^5 · 175/10^9`; the exact ratio is `4.55212e5`. -/
theorem chain_margin_down_ratio_gt :
    (4 * 10 ^ 5 : ℝ) * ((175 : ℝ) / 10 ^ 9) < 1469504930832769024 / 2 ^ 64 := by
  have heq : (4 * 10 ^ 5 : ℝ) * ((175 : ℝ) / 10 ^ 9) = (4 * 10 ^ 5 * 175 : ℝ) / 10 ^ 9 := by ring
  rw [heq, div_lt_div_iff₀ (by norm_num : (0 : ℝ) < 10 ^ 9) (by norm_num : (0 : ℝ) < 2 ^ 64)]
  norm_num

end Learn.IcmCount
