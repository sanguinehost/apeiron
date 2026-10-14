/-
Apeiron — bound-pruned exact argmax and the 32-wide block residual bound.

A greedy LM head may compute exact logits only on a candidate set `C` selected from an
approximation `A` with a sound per-index error bound `E`. This module proves the selection
is exact: every index attaining the maximum of `L` survives the bound test, and the
lowest-index maximiser over `C` is the lowest-index maximiser over the whole index set.
The extracted `sampler.argmax_fixed` returns exactly that index, so pruning never changes
the greedy token.

The domain is `Fin n → ℤ`, not `List ℤ`: indices are the tying object here (the lowest
index among maximisers), `Fin n` carries in-range proofs in the type, and the concrete
`Slice I64` form is reached once, in `argmax_fixed_returns_lowest`, through the
`Numeric.Engine.Sampler` `argmaxAt` law rather than by re-deriving the scan.

Reused, not restated: `Numeric.EngineSamplerRefinement` (`logitAt`, `argmaxAt`,
`argmax_fixed_loop_eq`). Every headline is a kernel-checked theorem in the closed axiom
cone `{propext, Classical.choice, Quot.sound}` (`Classical` enters through `Finset` order
selection).
-/
import Numeric.Engine.Sampler

set_option autoImplicit false
set_option warningAsError true

open Classical
open Aeneas Aeneas.Std Result
open apeiron_word

namespace PrunedArgmax

/-! ## The lowest-index maximiser over a finite set -/

/-- `i` maximises `f` over `s` (membership is carried separately by the filter). -/
def IsMaxOn {n : ℕ} (f : Fin n → ℤ) (s : Finset (Fin n)) (i : Fin n) : Prop :=
  ∀ j ∈ s, f j ≤ f i

/-- `Fin n` is nonempty when `n` is positive. -/
theorem univ_nonempty_of_pos {n : ℕ} (hn : 0 < n) :
    (Finset.univ : Finset (Fin n)).Nonempty :=
  ⟨⟨0, hn⟩, Finset.mem_univ _⟩

/-- A nonempty finite set has a maximiser, so its maximiser set is nonempty. -/
theorem maxOn_nonempty {n : ℕ} (f : Fin n → ℤ) (s : Finset (Fin n)) (hs : s.Nonempty) :
    (s.filter (IsMaxOn f s)).Nonempty := by
  obtain ⟨i, hi, hmax⟩ := Finset.exists_max_image s f hs
  exact ⟨i, Finset.mem_filter.mpr ⟨hi, hmax⟩⟩

/-- The lowest index maximising `f` over `s`; `hs` is the nonemptiness of `s`. -/
noncomputable def lowestMaxOn {n : ℕ} (f : Fin n → ℤ) (s : Finset (Fin n))
    (hs : s.Nonempty) : Fin n :=
  (s.filter (IsMaxOn f s)).min' (maxOn_nonempty f s hs)

/-- The selected index lies in `s`. -/
theorem lowestMaxOn_mem {n : ℕ} (f : Fin n → ℤ) (s : Finset (Fin n))
    (hs : s.Nonempty) : lowestMaxOn f s hs ∈ s :=
  (Finset.mem_filter.mp (Finset.min'_mem _ _)).1

/-- The selected index maximises over `s`. -/
theorem lowestMaxOn_isMax {n : ℕ} (f : Fin n → ℤ) (s : Finset (Fin n))
    (hs : s.Nonempty) : IsMaxOn f s (lowestMaxOn f s hs) :=
  (Finset.mem_filter.mp (Finset.min'_mem _ _)).2

/-- No maximiser over `s` precedes the selected index. -/
theorem lowestMaxOn_le {n : ℕ} (f : Fin n → ℤ) (s : Finset (Fin n))
    (hs : s.Nonempty) (k : Fin n) (hk : k ∈ s) (hmax : IsMaxOn f s k) :
    lowestMaxOn f s hs ≤ k :=
  (Finset.isLeast_min' _ _).2 (Finset.mem_filter.mpr ⟨hk, hmax⟩)

/-! ## Candidate selection from a sound per-index bound -/

/-- The lower envelope of the approximation: `max_j (A j - E j)`. -/
noncomputable def lowerBound {n : ℕ} (A E : Fin n → ℤ) (hn : 0 < n) : ℤ :=
  Finset.univ.sup' (univ_nonempty_of_pos hn) (fun j => A j - E j)

/-- The candidate set `C = {i | A i + E i ≥ lowerBound}`. -/
noncomputable def candSet {n : ℕ} (A E : Fin n → ℤ) (hn : 0 < n) : Finset (Fin n) :=
  Finset.univ.filter (fun i => lowerBound A E hn ≤ A i + E i)

/-- **Bound-pruned exact argmax.**

Given `L`, `A`, `E : Fin n → ℤ` with `|L i − A i| ≤ E i` for all `i`, let
`m = max_j (A j − E j)` and `C = {i | A i + E i ≥ m}`. Then

* every index attaining `max L` lies in `C`, and
* for every proof that `C` is nonempty, the lowest-index maximiser of `L` over `C` equals
  the lowest-index maximiser of `L` over all of `Fin n`.

The second clause quantifies over the nonemptiness proof because the least index of `C`
is independent of that proof; clause one supplies the membership that makes `C` nonempty
whenever `L` has any maximiser. -/
theorem pruned_argmax_exact {n : ℕ} (hn : 0 < n) (L A E : Fin n → ℤ)
    (hbound : ∀ i, |L i - A i| ≤ E i) :
    (∀ i, (∀ j, L j ≤ L i) → i ∈ candSet A E hn) ∧
    (∀ hC : (candSet A E hn).Nonempty,
      lowestMaxOn L (candSet A E hn) hC = lowestMaxOn L Finset.univ (univ_nonempty_of_pos hn)) := by
  have hsup_le : ∀ i, (∀ j, L j ≤ L i) → lowerBound A E hn ≤ A i + E i := by
    intro i hi
    rw [lowerBound]
    refine (Finset.sup'_le_iff _ _).mpr fun j _ => ?_
    have hj1 : -(E j) ≤ L j - A j := (abs_le.mp (hbound j)).1
    have hj2 : L j - A j ≤ E j := (abs_le.mp (hbound j)).2
    have hi2 : L i - A i ≤ E i := (abs_le.mp (hbound i)).2
    have hji : L j ≤ L i := hi j
    omega
  have hmem : ∀ i, (∀ j, L j ≤ L i) → i ∈ candSet A E hn := by
    intro i hi
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, hsup_le i hi⟩
  refine ⟨hmem, ?_⟩
  intro hC
  have hglobal : ∀ j, L j ≤ L (lowestMaxOn L Finset.univ (univ_nonempty_of_pos hn)) :=
    fun j => (lowestMaxOn_isMax L Finset.univ (univ_nonempty_of_pos hn)) j (Finset.mem_univ _)
  obtain ⟨g, _, hgmax⟩ :=
    Finset.exists_max_image (Finset.univ : Finset (Fin n)) L (univ_nonempty_of_pos hn)
  have hgmax' : ∀ j, L j ≤ L g := fun j => hgmax j (Finset.mem_univ _)
  have hgC : g ∈ candSet A E hn := hmem g hgmax'
  apply le_antisymm
  · refine lowestMaxOn_le L (candSet A E hn) hC _ (hmem _ hglobal) fun j _ => hglobal j
  · refine lowestMaxOn_le L Finset.univ (univ_nonempty_of_pos hn) _ (Finset.mem_univ _) fun j _ => ?_
    exact le_trans (hgmax' j) ((lowestMaxOn_isMax L (candSet A E hn) hC) g hgC)

/-! ## The 32-wide block residual bound -/

/-- The block index of a position: `j / 32`. -/
def blockOf {n : ℕ} (j : Fin n) : ℕ := j.val / 32

/-- Number of 32-wide blocks covering `n` positions: `⌈n/32⌉`. -/
def blockCount (n : ℕ) : ℕ := (n + 31) / 32

/-- **Block residual bound (Hölder, one constant per block).**

Let `w`, `wh`, `x : Fin n → ℤ` and `r : ℕ → ℤ` with `|w j − wh j| ≤ r (j / 32)` on every
position. Then the residual of the whole row is bounded by the per-block bounds weighted
by the block absolute sums:

`|Σ_j (w j − wh j)·x j| ≤ Σ_{b < ⌈n/32⌉} r b · Σ_{j : j/32 = b} |x j|`.

Block size is the fixed `32`; `r` is total and only the first `⌈n/32⌉` values occur. No
nonnegativity of `r` is required: the bound enters only through `|w j − wh j| ≤ r _` and
`|x j| ≥ 0`. -/
theorem holder_block_bound (n : ℕ) (w wh x : Fin n → ℤ) (r : ℕ → ℤ)
    (h : ∀ j, |w j - wh j| ≤ r (blockOf j)) :
    |∑ j : Fin n, (w j - wh j) * x j| ≤
      ∑ b ∈ Finset.range (blockCount n),
        r b * ∑ j ∈ Finset.univ.filter (fun j : Fin n => blockOf j = b), |x j| := by
  have htri : |∑ j : Fin n, (w j - wh j) * x j| ≤
      ∑ j : Fin n, |(w j - wh j) * x j| :=
    Finset.abs_sum_le_sum_abs _ _
  have hstep : (∑ j : Fin n, |(w j - wh j) * x j|) ≤
      ∑ j : Fin n, r (blockOf j) * |x j| := by
    refine Finset.sum_le_sum fun j _ => ?_
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_right (h j) (abs_nonneg _)
  have hmaps : ∀ j ∈ (Finset.univ : Finset (Fin n)), blockOf j ∈ Finset.range (blockCount n) := by
    intro j _
    rw [Finset.mem_range, blockCount, blockOf]
    omega
  have hfiber : (∑ j : Fin n, r (blockOf j) * |x j|) =
      ∑ b ∈ Finset.range (blockCount n),
        ∑ j ∈ Finset.univ.filter (fun j : Fin n => blockOf j = b), r b * |x j| := by
    rw [← Finset.sum_fiberwise_of_maps_to hmaps (fun j => r (blockOf j) * |x j|)]
    refine Finset.sum_congr rfl fun b _ => Finset.sum_congr rfl fun j hj => ?_
    rw [(Finset.mem_filter.mp hj).2]
  calc |∑ j : Fin n, (w j - wh j) * x j|
      ≤ ∑ j : Fin n, |(w j - wh j) * x j| := htri
    _ ≤ ∑ j : Fin n, r (blockOf j) * |x j| := hstep
    _ = ∑ b ∈ Finset.range (blockCount n),
          ∑ j ∈ Finset.univ.filter (fun j : Fin n => blockOf j = b), r b * |x j| := hfiber
    _ = ∑ b ∈ Finset.range (blockCount n),
          r b * ∑ j ∈ Finset.univ.filter (fun j : Fin n => blockOf j = b), |x j| := by
        refine Finset.sum_congr rfl fun b _ => ?_
        exact (Finset.mul_sum _ _ _).symm

/-! ## The link to the extracted argmax

`argmaxAt logits best` is the exact law of `sampler.argmax_fixed` (`Numeric.EngineSamplerRefinement`):
`best` is in range, dominates every in-range logit, and strictly dominates every earlier one.
The strict clause is precisely "lowest index among maximisers", so the scan's `best.val` is
the `lowestMaxOn` index of the pruned-argmax statement, with `L` read as the integer logits. -/

/-- `argmaxAt` characterises the lowest-index maximiser: if `best` dominates all positions
weakly and all earlier positions strictly, it is the least index of the maximum. -/
theorem lowestMaxOn_eq_of_least {n : ℕ} (f : Fin n → ℤ) (best : ℕ) (hbest : best < n)
    (hmax : ∀ j : Fin n, f j ≤ f ⟨best, hbest⟩)
    (hstrict : ∀ j : Fin n, j.val < best → f j < f ⟨best, hbest⟩)
    (hs : (Finset.univ : Finset (Fin n)).Nonempty) :
    best = (lowestMaxOn f Finset.univ hs).val := by
  have hle : lowestMaxOn f Finset.univ hs ≤ (⟨best, hbest⟩ : Fin n) :=
    lowestMaxOn_le f Finset.univ hs ⟨best, hbest⟩ (Finset.mem_univ _) fun j _ => hmax j
  have hnotlt : ¬ (lowestMaxOn f Finset.univ hs).val < best := by
    intro hlt
    have hlt' : f (lowestMaxOn f Finset.univ hs) < f ⟨best, hbest⟩ :=
      hstrict (lowestMaxOn f Finset.univ hs) hlt
    have hge : f ⟨best, hbest⟩ ≤ f (lowestMaxOn f Finset.univ hs) :=
      lowestMaxOn_isMax f Finset.univ hs ⟨best, hbest⟩ (Finset.mem_univ _)
    omega
  have hle' : (lowestMaxOn f Finset.univ hs).val ≤ best := hle
  omega

/-- The pruned-argmax index of a logit slice: the least index maximising the integer logits
`(logitAt logits j).val`. -/
noncomputable def lowestLogitArgmax (logits : Slice I64) (hne : 0 < logits.val.length) :
    Fin logits.val.length :=
  lowestMaxOn (fun i : Fin logits.val.length => (logitAt logits i.val).val) Finset.univ
    (univ_nonempty_of_pos hne)

/-- **The extracted argmax returns the pruned exact maximiser.**

On nonempty logits the extracted `sampler.argmax_fixed_loop` returns a `best` whose value is the
lowest index maximising the integer logits, i.e. exactly the `lowestMaxOn` index of
`pruned_argmax_exact` when `L` is the logit reading. -/
theorem argmax_fixed_returns_lowest (logits : Slice I64) (hne : 0 < logits.val.length) :
    ∃ best : Usize,
      sampler.argmax_fixed_loop logits 0#usize 1#usize = ok best ∧
      best.val = (lowestLogitArgmax logits hne).val := by
  obtain ⟨b, hb, hloop⟩ := argmax_fixed_loop_eq logits hne
  refine ⟨b, hloop, ?_⟩
  obtain ⟨hblen, hmax, hstrict⟩ := hb
  exact lowestMaxOn_eq_of_least (fun i : Fin logits.val.length => (logitAt logits i.val).val)
    b.val hblen (fun j => hmax j.val j.isLt) (fun j hj => hstrict j.val hj) _

/-- **The public extracted argmax returns the pruned exact maximiser.**

`sampler.argmax_fixed::word_argmax_fixed` narrows the loop index to `u32`
(`Numeric.EngineSamplerRefinement.word_argmax_fixed_spec`); when the pruned index fits, the
returned `u32` is that index. The `u32`-overflow refusal arm is the spec's, not restated here. -/
theorem word_argmax_fixed_returns_lowest (logits : Slice I64) (hne : 0 < logits.val.length)
    (hle : (lowestLogitArgmax logits hne).val ≤ U32.max) :
    ∃ u : U32,
      word_argmax_fixed logits = ok (core.result.Result.Ok u) ∧
      u.val = (lowestLogitArgmax logits hne).val := by
  obtain ⟨b, hb, hsuccess, _⟩ := word_argmax_fixed_spec logits hne
  obtain ⟨hblen, hmax, hstrict⟩ := hb
  have hbval : b = (lowestLogitArgmax logits hne).val :=
    lowestMaxOn_eq_of_least (fun i : Fin logits.val.length => (logitAt logits i.val).val)
      b hblen (fun j => hmax j.val j.isLt) (fun j hj => hstrict j.val hj) _
  obtain ⟨u, hur, huv⟩ := hsuccess (by rw [hbval]; exact hle)
  exact ⟨u, hur, by rw [huv, hbval]⟩

/-! ## Joint non-vacuity witness

Four positions; `L = [10,0,0,0]`, `A = [10,0,0,0]`, `E = [1,1,1,1]`. The lower envelope is
`9`, so `A 0 + E 0 = 11 ≥ 9` but `A i + E i = 1 < 9` for `i ≠ 0`: `C = {0}` is a strict
subset of `Fin 4`. The exact-argmax equality holds, and both sides are `0`. -/

namespace Witness

/-- `L = [10,0,0,0]`. -/
def L4 : Fin 4 → ℤ := ![10, 0, 0, 0]

/-- `A = [10,0,0,0]`. -/
def A4 : Fin 4 → ℤ := ![10, 0, 0, 0]

/-- `E = [1,1,1,1]`. -/
def E4 : Fin 4 → ℤ := ![1, 1, 1, 1]

theorem hn4 : 0 < 4 := by norm_num

theorem hbound4 : ∀ i, |L4 i - A4 i| ≤ E4 i := by
  intro i
  fin_cases i <;> norm_num [L4, A4, E4]

theorem lowerBound_eq : lowerBound A4 E4 hn4 = 9 := by
  apply le_antisymm
  · rw [lowerBound]
    refine (Finset.sup'_le_iff _ _).mpr fun j _ => ?_
    fin_cases j <;> norm_num [A4, E4]
  · rw [lowerBound]
    refine (Finset.le_sup'_iff _).mpr ⟨(⟨0, hn4⟩ : Fin 4), Finset.mem_univ _, ?_⟩
    norm_num [A4, E4]

theorem candSet_eq : candSet A4 E4 hn4 = {0} := by
  ext i
  rw [candSet, lowerBound_eq]
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
  fin_cases i <;> norm_num [A4, E4]

/-- The candidate set is a strict subset of the index set, and the pruned selection is exact. -/
theorem witness :
    (∀ hC : (candSet A4 E4 hn4).Nonempty,
      lowestMaxOn L4 (candSet A4 E4 hn4) hC =
        lowestMaxOn L4 Finset.univ (univ_nonempty_of_pos hn4)) ∧
    (candSet A4 E4 hn4) ≠ Finset.univ := by
  refine ⟨(pruned_argmax_exact hn4 L4 A4 E4 hbound4).2, ?_⟩
  rw [candSet_eq]
  decide

end Witness

end PrunedArgmax
