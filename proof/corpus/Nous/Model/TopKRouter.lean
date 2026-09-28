/-
Copyright (c) 2026 Apeiron contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Apeiron contributors
-/

import Mathlib.Data.List.Sort
import Mathlib.Data.List.FinRange
import Mathlib.Data.Prod.Lex
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Tactic

/-!
# Top-`k` expert routing: exact integer selection and renormalized softmax

Fine-grained mixture-of-experts routers select the `k` largest router logits and renormalize the
gate weights over the selection (DeepSeekMoE, Dai et al., arXiv `2401.06066`; Qwen3-MoE
`softmax` → `topk` → optional `norm_topk_prob`). This module proves the two facts the fixed-point
router relies on:

* **Selection is exact and storage-order independent.** `topkSel` sorts the expert indices by the
  key `(logit descending, index ascending)`, so it uses no `exp` and no rounding. `topkSel_spec`
  states that a selected index dominates every unselected index (larger logit, or equal logit and
  smaller index), and `topkSel_perm_invariant` states that permuting expert storage relabels the
  selected set by the same permutation and preserves the canonical index set.

* **Renormalized full softmax is the softmax over the selection.** The common partition function
  cancels (`softmax_topk_renorm`), also after subtracting a shift such as the selection maximum
  (`softmax_topk_renorm_shift`, `softmax_topk_renorm_max`). Because `Real.exp` is strictly
  monotone, the top-`k` set is the same set computed on logits and on softmax probabilities
  (`topk_monotone_softmax`).

The real-valued identities are stated for `Fin n → ℝ`. Fixing the round-to-nearest fixed-point
execution of `exp` to these reals, and refining a Rust `route_topk` to `topkSel`, are separate
obligations.
-/

namespace Nous
namespace Model

open scoped BigOperators
open List

/-! ## 1. Exact selection on integer logits -/

/-- The selection key of an expert: `(OrderDual.toDual v, i)`, i.e. the logit `v` in the opposite
order followed by the expert index `i`. Ordering keys by `≤` lists the largest logit first and,
among equal logits, the smallest index first. -/
abbrev RouterKey : Type := ℤᵒᵈ ×ₗ ℕ

/-- The lexicographic selection key `(logit descending, index ascending)`. -/
def routerKey (v : ℤ) (i : ℕ) : RouterKey := toLex (OrderDual.toDual v, i)

@[simp] theorem ofLex_routerKey (v : ℤ) (i : ℕ) :
    ofLex (routerKey v i) = (OrderDual.toDual v, i) := rfl

/-- The index component of a key is `routerKey`'s second argument. -/
theorem routerKey_snd (v : ℤ) (i : ℕ) : (ofLex (routerKey v i)).2 = i := by
  simp

/-- Comparison of two selection keys: the larger logit is smaller in the key order, ties are
broken by the smaller index. -/
theorem routerKey_le_iff {v w : ℤ} {i j : ℕ} :
    routerKey v i ≤ routerKey w j ↔ w < v ∨ (v = w ∧ i ≤ j) := by
  rw [routerKey, routerKey, Prod.Lex.toLex_le_toLex]
  simp [OrderDual.toDual_lt_toDual]

/-- Selection keys separate experts: two keys are equal iff both the logit and the index agree. -/
theorem routerKey_inj {v w : ℤ} {i j : ℕ} :
    routerKey v i = routerKey w j ↔ v = w ∧ i = j := by
  rw [routerKey, routerKey, toLex_inj]
  simp [Prod.mk.injEq]

/-- The list of `(logit, index)` keys of `logits`, index `i` carrying the logit at position `i`. -/
def keyList (logits : List ℤ) : List RouterKey :=
  List.ofFn (fun i : Fin logits.length => routerKey (logits.get i) i)

/-- `keyList logits` sorted by the selection order `(logit descending, index ascending)`. -/
def sortedKeys (logits : List ℤ) : List RouterKey :=
  (keyList logits).mergeSort (· ≤ ·)

/-- The canonical index set of a list of selection keys: sort by the selection order and take the
first `k` indices. -/
def selKeys (ks : List RouterKey) (k : ℕ) : List ℕ :=
  ((ks.mergeSort (· ≤ ·)).take k).map (fun p => (ofLex p).2)

/-- `topkSel logits k` selects the `k` largest logits, ties broken by the lower index. The result
is a list of positions in `logits`, in selection order. -/
def topkSel (logits : List ℤ) (k : ℕ) : List ℕ :=
  selKeys (keyList logits) k

theorem keyList_mem {logits : List ℤ} {p : RouterKey} :
    p ∈ keyList logits ↔ ∃ i : Fin logits.length, routerKey (logits.get i) (i : ℕ) = p := by
  rw [keyList, List.mem_ofFn]

theorem keyAt_mem (logits : List ℤ) (i : ℕ) (h : i < logits.length) :
    routerKey (logits.get ⟨i, h⟩) i ∈ keyList logits := by
  rw [keyList, List.mem_ofFn]
  exact ⟨⟨i, h⟩, rfl⟩

theorem sortedKeys_perm (logits : List ℤ) : sortedKeys logits ~ keyList logits :=
  List.mergeSort_perm _ _

theorem mergeSort_sortedLE (ks : List RouterKey) : (ks.mergeSort (· ≤ ·)).SortedLE :=
  List.sortedLE_mergeSort

theorem mergeSort_perm' (ks : List RouterKey) : ks.mergeSort (· ≤ ·) ~ ks :=
  List.mergeSort_perm _ _

theorem sortedKeys_sorted (logits : List ℤ) : (sortedKeys logits).SortedLE :=
  mergeSort_sortedLE (keyList logits)

theorem keyList_nodup (logits : List ℤ) : (keyList logits).Nodup := by
  rw [keyList, List.nodup_ofFn]
  intro i j hij
  have h : (OrderDual.toDual (logits.get i), (i : ℕ))
      = (OrderDual.toDual (logits.get j), (j : ℕ)) := by
    simpa [routerKey] using hij
  exact Fin.ext (congrArg Prod.snd h)

theorem sortedKeys_nodup (logits : List ℤ) : (sortedKeys logits).Nodup :=
  (sortedKeys_perm logits).nodup_iff.mpr (keyList_nodup logits)

theorem keyAt_mem_sorted (logits : List ℤ) (i : ℕ) (h : i < logits.length) :
    routerKey (logits.get ⟨i, h⟩) i ∈ sortedKeys logits :=
  (sortedKeys_perm logits).mem_iff.mpr (keyAt_mem logits i h)

theorem topkSel_length (logits : List ℤ) (k : ℕ) :
    (topkSel logits k).length = min k logits.length := by
  simp [topkSel, selKeys, keyList, List.length_take]

theorem topkSel_nodup (logits : List ℤ) (k : ℕ) : (topkSel logits k).Nodup := by
  rw [topkSel, selKeys]
  apply List.Nodup.map_on
  · intro p hp q hq hpq
    have hp' : p ∈ keyList logits :=
      (sortedKeys_perm logits).mem_iff.mp (List.mem_of_mem_take hp)
    have hq' : q ∈ keyList logits :=
      (sortedKeys_perm logits).mem_iff.mp (List.mem_of_mem_take hq)
    rw [keyList_mem] at hp' hq'
    obtain ⟨ip, hip⟩ := hp'
    obtain ⟨iq, hiq⟩ := hq'
    have hval : (ip : ℕ) = (iq : ℕ) := by
      rw [← hip, ← hiq] at hpq
      simpa [routerKey] using hpq
    rw [← hip, ← hiq, Fin.ext hval]
  · exact (sortedKeys_nodup logits).sublist (List.take_sublist k _)

/-- Indices selected by `topkSel` lie in the valid range. -/
theorem topkSel_lt (logits : List ℤ) (k : ℕ) {i : ℕ} (hi : i ∈ topkSel logits k) :
    i < logits.length := by
  rw [topkSel, selKeys, List.mem_map] at hi
  obtain ⟨p, hp, hpi⟩ := hi
  have hpk : p ∈ keyList logits :=
    (sortedKeys_perm logits).mem_iff.mp (List.mem_of_mem_take hp)
  rw [keyList_mem] at hpk
  obtain ⟨q, hq⟩ := hpk
  rw [← hpi, ← hq]
  simp [routerKey]

/-- Membership in `topkSel` is exactly membership of the canonical key in the sorted prefix. -/
theorem mem_topkSel_iff (logits : List ℤ) (k i : ℕ) (h : i < logits.length) :
    i ∈ topkSel logits k ↔ routerKey (logits.get ⟨i, h⟩) i ∈ (sortedKeys logits).take k := by
  constructor
  · intro hi
    rw [topkSel, selKeys, List.mem_map] at hi
    obtain ⟨p, hp, hpi⟩ := hi
    have hps : p ∈ sortedKeys logits := List.mem_of_mem_take hp
    have hpk : p ∈ keyList logits := (sortedKeys_perm logits).mem_iff.mp hps
    rw [keyList_mem] at hpk
    obtain ⟨q, hq⟩ := hpk
    have hval : (q : ℕ) = i := by
      have h2 : (ofLex (routerKey (logits.get q) (q : ℕ))).2 = i := by rw [hq, hpi]
      simpa [routerKey] using h2
    obtain rfl : q = ⟨i, h⟩ := Fin.ext hval
    rw [hq]
    exact hp
  · intro hmem
    exact List.mem_map.mpr ⟨_, hmem, by simp [routerKey]⟩

/-- **Selection spec.** A selected index dominates every unselected index: its logit is larger, or
the logits are equal and its index is smaller. -/
theorem topkSel_spec (logits : List ℤ) (k : ℕ) {i j : ℕ} (hi : i ∈ topkSel logits k)
    (hj : j ∉ topkSel logits k) (hjn : j < logits.length) :
    logits.get ⟨j, hjn⟩ < logits.get ⟨i, topkSel_lt logits k hi⟩ ∨
      (logits.get ⟨j, hjn⟩ = logits.get ⟨i, topkSel_lt logits k hi⟩ ∧ i < j) := by
  have hin := topkSel_lt logits k hi
  have hki : routerKey (logits.get ⟨i, hin⟩) i ∈ (sortedKeys logits).take k :=
    (mem_topkSel_iff logits k i hin).mp hi
  have hkj_not : routerKey (logits.get ⟨j, hjn⟩) j ∉ (sortedKeys logits).take k :=
    fun hmem => hj ((mem_topkSel_iff logits k j hjn).mpr hmem)
  have hkj_mem : routerKey (logits.get ⟨j, hjn⟩) j ∈ sortedKeys logits :=
    keyAt_mem_sorted logits j hjn
  have hkj : routerKey (logits.get ⟨j, hjn⟩) j ∈ (sortedKeys logits).drop k := by
    have h := hkj_mem
    rw [← List.take_append_drop k (sortedKeys logits)] at h
    exact (List.mem_append.mp h).resolve_left hkj_not
  have hle : routerKey (logits.get ⟨i, hin⟩) i ≤ routerKey (logits.get ⟨j, hjn⟩) j :=
    ((sortedKeys_sorted logits).pairwise).rel_of_mem_take_of_mem_drop hki hkj
  rcases (routerKey_le_iff.mp hle) with hlt | ⟨heq, hij⟩
  · exact Or.inl hlt
  · exact Or.inr ⟨heq.symm, lt_of_le_of_ne hij fun h => hj (h ▸ hi)⟩

/-- **Order independence.** Permuting the list of selection keys leaves the canonical index set of
the selection unchanged. -/
theorem selKeys_perm_invariant {ks ls : List RouterKey} (h : ls ~ ks) (k : ℕ) :
    (selKeys ls k).toFinset = (selKeys ks k).toFinset := by
  have hp : ls.mergeSort (· ≤ ·) ~ ks.mergeSort (· ≤ ·) :=
    ((mergeSort_perm' ls).trans h).trans (mergeSort_perm' ks).symm
  have heq : ls.mergeSort (· ≤ ·) = ks.mergeSort (· ≤ ·) :=
    List.Perm.eq_of_sortedLE (mergeSort_sortedLE ls) (mergeSort_sortedLE ks) hp
  rw [selKeys, selKeys, heq]

/-- **Order independence over expert storage.** For a permutation `π` of expert storage, the
router reads expert `π i` at storage position `i`; selecting on the storage order and mapping each
position by `π` back to its canonical expert leaves the selected canonical index set unchanged. -/
theorem topkSel_perm_invariant (logits : List ℤ) (k : ℕ) (π : Equiv.Perm (Fin logits.length)) :
    (selKeys (List.ofFn (fun i : Fin logits.length => routerKey (logits.get (π i)) (π i))) k).toFinset
      = (topkSel logits k).toFinset :=
  selKeys_perm_invariant (ks := keyList logits)
    (Equiv.Perm.ofFn_comp_perm π (fun i : Fin logits.length => routerKey (logits.get i) i)) k

/-! ## 2. Renormalized softmax over the selection -/

/-- The softmax partition function `∑ j, exp (z j)`. -/
noncomputable def partition {n : ℕ} (z : Fin n → ℝ) : ℝ := ∑ j, Real.exp (z j)

/-- The softmax probability of component `i`. -/
noncomputable def softmax {n : ℕ} (z : Fin n → ℝ) (i : Fin n) : ℝ :=
  Real.exp (z i) / partition z

theorem partition_pos {n : ℕ} (z : Fin n → ℝ) (i : Fin n) : 0 < partition z :=
  Finset.sum_pos (fun j _ => Real.exp_pos (z j)) ⟨i, Finset.mem_univ i⟩

theorem partition_ne_zero {n : ℕ} (z : Fin n → ℝ) (i : Fin n) : partition z ≠ 0 :=
  ne_of_gt (partition_pos z i)

theorem sum_exp_pos {n : ℕ} (z : Fin n → ℝ) {S : Finset (Fin n)} (hS : S.Nonempty) :
    0 < ∑ s ∈ S, Real.exp (z s) :=
  Finset.sum_pos (fun s _ => Real.exp_pos (z s)) hS

/-- Softmax probabilities sum to one. -/
theorem softmax_sum_eq_one {n : ℕ} (z : Fin n → ℝ) (i : Fin n) : ∑ j, softmax z j = 1 := by
  simp only [softmax]
  rw [← Finset.sum_div]
  exact div_self (partition_ne_zero z i)

/-- **Renormalized full softmax is the softmax over the selection.** The partition function
`∑ j, exp (z j)` cancels, so for `i ∈ S` the renormalized full-softmax weight equals
`exp (z i) / ∑ s ∈ S, exp (z s)`. -/
theorem softmax_topk_renorm {n : ℕ} (z : Fin n → ℝ) (S : Finset (Fin n)) {i : Fin n}
    (hi : i ∈ S) :
    (Real.exp (z i) / partition z) / ((∑ s ∈ S, Real.exp (z s)) / partition z) =
      Real.exp (z i) / (∑ s ∈ S, Real.exp (z s)) := by
  have hD : partition z ≠ 0 := partition_ne_zero z i
  have hB : (∑ s ∈ S, Real.exp (z s)) ≠ 0 := ne_of_gt (sum_exp_pos z ⟨i, hi⟩)
  field_simp [hD, hB]

/-- Shifting every logit by `c` multiplies the partition function by `exp (-c)`. -/
theorem sum_exp_shift {n : ℕ} (z : Fin n → ℝ) (S : Finset (Fin n)) (c : ℝ) :
    (∑ s ∈ S, Real.exp (z s - c)) = Real.exp (-c) * ∑ s ∈ S, Real.exp (z s) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro s _
  rw [sub_eq_add_neg, Real.exp_add]
  ring

/-- **Shift invariance of softmax.** Subtracting a common shift `c` (for instance the maximum over
the selection) leaves the softmax probability unchanged. -/
theorem softmax_shift_invariant {n : ℕ} (z : Fin n → ℝ) {i : Fin n} (c : ℝ) :
    Real.exp (z i - c) / partition (fun j => z j - c) = Real.exp (z i) / partition z := by
  rw [div_eq_div_iff (partition_ne_zero (fun j => z j - c) i) (partition_ne_zero z i)]
  simp only [partition]
  rw [sum_exp_shift z Finset.univ c, sub_eq_add_neg, Real.exp_add]
  ring

/-- **Shift-invariant renormalized softmax.** The renormalized softmax over `S` is unchanged when a
common shift `c` is subtracted from all logits; the right-hand side is the unshifted ratio. -/
theorem softmax_topk_renorm_shift {n : ℕ} (z : Fin n → ℝ) (S : Finset (Fin n)) {i : Fin n}
    (hi : i ∈ S) (c : ℝ) :
    (Real.exp (z i - c) / partition (fun j => z j - c)) /
        ((∑ s ∈ S, Real.exp (z s - c)) / partition (fun j => z j - c)) =
      Real.exp (z i) / (∑ s ∈ S, Real.exp (z s)) := by
  rw [softmax_topk_renorm (fun j => z j - c) S hi, sum_exp_shift z S c, sub_eq_add_neg, Real.exp_add]
  have hQ : (∑ s ∈ S, Real.exp (z s)) ≠ 0 := ne_of_gt (sum_exp_pos z ⟨i, hi⟩)
  rw [div_eq_div_iff (mul_ne_zero (Real.exp_ne_zero (-c)) hQ) hQ]
  ring

/-- The renormalized softmax with the selection maximum subtracted. -/
theorem softmax_topk_renorm_max {n : ℕ} (z : Fin n → ℝ) (S : Finset (Fin n)) {i : Fin n}
    (hi : i ∈ S) :
    (Real.exp (z i - S.sup' ⟨i, hi⟩ z) / partition (fun j => z j - S.sup' ⟨i, hi⟩ z)) /
        ((∑ s ∈ S, Real.exp (z s - S.sup' ⟨i, hi⟩ z)) /
          partition (fun j => z j - S.sup' ⟨i, hi⟩ z)) =
      Real.exp (z i) / (∑ s ∈ S, Real.exp (z s)) :=
  softmax_topk_renorm_shift z S hi (S.sup' ⟨i, hi⟩ z)

/-! ## 3. Monotonicity of softmax and the top-`k` set -/

/-- Softmax preserves the strict order of the logits. -/
theorem softmax_lt_iff {n : ℕ} (z : Fin n → ℝ) {i j : Fin n} :
    softmax z i < softmax z j ↔ z i < z j := by
  rw [softmax, softmax, div_lt_div_iff_of_pos_right (partition_pos z i)]
  exact Real.exp_lt_exp

/-- A softmax value determines its logit: equal probabilities imply equal logits. -/
theorem softmax_eq_iff {n : ℕ} (z : Fin n → ℝ) {i j : Fin n} :
    softmax z i = softmax z j ↔ z i = z j := by
  rw [softmax, softmax, div_eq_div_iff (partition_ne_zero z i) (partition_ne_zero z j)]
  constructor
  · intro h
    exact Real.exp_injective (mul_right_cancel₀ (partition_ne_zero z i) h)
  · intro h
    rw [h]

/-- A set `S` is a top-`k` selection for the score function `v`: it has `min k n` elements and
every selected index strictly dominates every unselected index, ties broken by the lower index. -/
def IsTopKSet {n : ℕ} (v : Fin n → ℝ) (k : ℕ) (S : Finset (Fin n)) : Prop :=
  S.card = min k n ∧ ∀ i ∈ S, ∀ j, j ∉ S → v j < v i ∨ (v j = v i ∧ i < j)

/-- **Top-`k` sets agree on logits and on softmax probabilities.** Since softmax is strictly
monotone in the logits and injective on them (`softmax_lt_iff`, `softmax_eq_iff`), the same
selection predicate holds for `z` and for `softmax z`. -/
theorem topk_monotone_softmax {n : ℕ} (z : Fin n → ℝ) (k : ℕ) (S : Finset (Fin n)) :
    IsTopKSet z k S ↔ IsTopKSet (softmax z) k S := by
  constructor
  · rintro ⟨hcard, hspec⟩
    refine ⟨hcard, ?_⟩
    intro i hi j hj
    rcases hspec i hi j hj with hlt | ⟨heq, hij⟩
    · exact Or.inl ((softmax_lt_iff z).mpr hlt)
    · exact Or.inr ⟨(softmax_eq_iff z).mpr heq, hij⟩
  · rintro ⟨hcard, hspec⟩
    refine ⟨hcard, ?_⟩
    intro i hi j hj
    rcases hspec i hi j hj with hlt | ⟨heq, hij⟩
    · exact Or.inl ((softmax_lt_iff z).mp hlt)
    · exact Or.inr ⟨(softmax_eq_iff z).mp heq, hij⟩

/-- The top-`k` selection is unique: the selection predicate determines the set. Hence the top-`k`
set by logits and the top-`k` set by softmax probabilities are the same set. -/
theorem IsTopKSet.unique {n : ℕ} {v : Fin n → ℝ} {k : ℕ} {S T : Finset (Fin n)}
    (hS : IsTopKSet v k S) (hT : IsTopKSet v k T) : S = T := by
  have asymm : ∀ {i j : Fin n},
      (v j < v i ∨ (v j = v i ∧ i < j)) → ¬ (v i < v j ∨ (v i = v j ∧ j < i)) := by
    rintro i j (h | ⟨he, hlt⟩) (h' | ⟨he', hlt'⟩)
    · exact absurd h' (not_lt_of_gt h)
    · exact absurd he'.symm (ne_of_lt h)
    · exact absurd he.symm (ne_of_lt h')
    · exact absurd hlt' (not_lt_of_gt hlt)
  have key : ∀ {U W : Finset (Fin n)}, IsTopKSet v k U → IsTopKSet v k W →
      ∀ {i : Fin n}, i ∈ U → i ∉ W → W ⊆ U := by
    intro U W hU hW i hiU hiW j hjW
    by_contra hjU
    exact asymm (hU.2 i hiU j hjU) (hW.2 j hjW i hiW)
  apply Finset.Subset.antisymm
  · intro i hiS
    by_contra hiT
    have hTS := key hS hT hiS hiT
    have hcard : S.card ≤ T.card := by rw [hS.1, hT.1]
    exact hiT ((Finset.eq_of_subset_of_card_le hTS hcard).symm ▸ hiS)
  · intro i hiT
    by_contra hiS
    have hST := key hT hS hiT hiS
    have hcard : T.card ≤ S.card := by rw [hS.1, hT.1]
    exact hiS ((Finset.eq_of_subset_of_card_le hST hcard).symm ▸ hiT)

end Model
end Nous
