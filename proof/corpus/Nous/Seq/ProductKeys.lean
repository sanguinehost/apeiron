/-
Copyright (c) 2026 Apeiron contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Apeiron contributors
-/
import Mathlib.Data.Fin.Basic
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Prod
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Group.Defs
import Mathlib.Algebra.Order.Monoid.Defs

/-!
# Product-Key Memory (PKM): Certified Top-k Search on Unfolded Codebook Tori

Product-key memory (Lample et al. 2019, "Large Memory Layers with Product Keys"; anchor)
splits a query `q = (q₁, q₂)` and key codebooks into two sub-codebooks `C₁, C₂` of size `n`.
The `N = n²` slots are pairs `(i, j)` scored by additive score `s(i, j) = a(i) + b(j)`,
where `a(i) = q₁ · c₁_i` and `b(j) = q₂ · c₂_j`.

The foundational structural claim that reduces search complexity from `O(N) = O(n²)` to `O(√N) = O(n)`
is:
  **The top-`k` pairs of the product grid lie entirely inside `topk(a, k) × topk(b, k)`.**

This module mechanizes and certifies that claim and its exactness:
1. `product_topk_subset`: for any additive scores `a : Fin n → R`, `b : Fin m → R`, every element
   of the top-`k` of `(i, j) ↦ a i + b j` lies in `topk a k × topk b k` under consistent tie-breaking.
2. `product_topk_exact`: the top-`k` of the full `n · m` grid equals the top-`k` of the `k · k`
   candidate grid, establishing that evaluating `n + m` keys and `k²` candidates is exact.
3. `multi_product_topk`: generalized to `d` codebooks, the top-`k` of `∑ t, a t (x t)` lies in
   `∏ t, topk (a t) k`, requiring only `d · n + k^d` evaluations instead of `n^d`.
4. `slot_address_bijective`: the address map `(i, j) ↦ i · m + j` is a bijection
   `Fin n × Fin m ≃ Fin (n · m)` via `finProdFinEquiv` (the 2-torus unfolds onto a linear table).
5. Concrete witness: for `n = m = 4`, `k = 2`, brute-force top-2 over 16 slots equals top-2 over
   the 4 candidates, certified by `decide`.
-/

namespace Nous
namespace Seq
namespace ProductKeys

open scoped BigOperators
open Classical

/-! ## 1. Ranking, Tie-breaking, and Top-`k` over Finset -/

/-- Strict ranking relation induced by score and index tie-breaking (lowest index wins ties). -/
def prec {X R : Type*} [LinearOrder X] [LT R] [DecidableEq R]
    (score : X → R) (x y : X) : Prop :=
  score y < score x ∨ (score x = score y ∧ x < y)

instance {X R : Type*} [LinearOrder X] [LT R] [DecidableEq R]
    [DecidableRel (· < · : X → X → Prop)] [DecidableRel (· < · : R → R → Prop)]
    (score : X → R) : DecidableRel (prec score) :=
  fun x y => inferInstanceAs (Decidable (score y < score x ∨ (score x = score y ∧ x < y)))

/-- Predecessors of `x` in finset `s`: elements ranked strictly ahead of `x`. -/
def predecessors {X : Type*} [DecidableEq X] (s : Finset X) (precRel : X → X → Prop)
    [DecidableRel precRel] (x : X) : Finset X :=
  s.filter (fun y => precRel y x)

/-- The top-`k` elements of `s` under `precRel`: elements with strictly fewer than `k` predecessors. -/
def topk {X : Type*} [DecidableEq X] (s : Finset X) (precRel : X → X → Prop)
    [DecidableRel precRel] (k : ℕ) : Finset X :=
  s.filter (fun x => (predecessors s precRel x).card < k)

/-! ## 2. Product Scores and Lexicographic Tie-breaking -/

/-- Lexicographic index comparison on pairs (row index first, column index second). -/
def lexLT {α β : Type*} [LinearOrder α] [LinearOrder β] (p1 p2 : α × β) : Prop :=
  p1.1 < p2.1 ∨ (p1.1 = p2.1 ∧ p1.2 < p2.2)

instance {α β : Type*} [LinearOrder α] [LinearOrder β]
    [DecidableRel (· < · : α → α → Prop)] [DecidableRel (· < · : β → β → Prop)] :
    DecidableRel (lexLT (α := α) (β := β)) :=
  fun p1 p2 => inferInstanceAs (Decidable (p1.1 < p2.1 ∨ (p1.1 = p2.1 ∧ p1.2 < p2.2)))

/-- Joint additive score of a pair: `a i + b j`. -/
def pairScore {α β R : Type*} [Add R] (a : α → R) (b : β → R) (p : α × β) : R :=
  a p.1 + b p.2

/-- Ranking relation on pairs: higher additive score wins; ties broken lexicographically. -/
def precPair {α β R : Type*} [LinearOrder α] [LinearOrder β] [Add R] [LT R] [DecidableEq R]
    (a : α → R) (b : β → R) (p1 p2 : α × β) : Prop :=
  pairScore a b p2 < pairScore a b p1 ∨
    (pairScore a b p1 = pairScore a b p2 ∧ lexLT p1 p2)

instance {α β R : Type*} [LinearOrder α] [LinearOrder β] [Add R] [LT R] [DecidableEq R]
    [DecidableRel (· < · : α → α → Prop)] [DecidableRel (· < · : β → β → Prop)]
    [DecidableRel (· < · : R → R → Prop)] (a : α → R) (b : β → R) :
    DecidableRel (precPair a b) :=
  fun p1 p2 => inferInstanceAs (Decidable (pairScore a b p2 < pairScore a b p1 ∨
    (pairScore a b p1 = pairScore a b p2 ∧ lexLT p1 p2)))

/-- Monotonicity on the second coordinate: for any fixed row `i`, ranking along column `j`
is identical to the ranking of `(i, j)` in the product space. -/
theorem prec_snd_iff {α β R : Type*} [LinearOrder α] [LinearOrder β]
    [AddCommGroup R] [LinearOrder R] [IsOrderedAddMonoid R]
    [DecidableRel (· < · : α → α → Prop)] [DecidableRel (· < · : β → β → Prop)]
    [DecidableEq R] [DecidableRel (· < · : R → R → Prop)]
    (a : α → R) (b : β → R) (i : α) (j1 j2 : β) :
    prec b j1 j2 ↔ precPair a b (i, j1) (i, j2) := by
  dsimp [prec, precPair, pairScore, lexLT]
  rw [add_lt_add_iff_left, add_right_inj]
  simp

/-- Monotonicity on the first coordinate: for any fixed column `j`, ranking along row `i`
is identical to the ranking of `(i, j)` in the product space. -/
theorem prec_fst_iff {α β R : Type*} [LinearOrder α] [LinearOrder β]
    [AddCommGroup R] [LinearOrder R] [IsOrderedAddMonoid R]
    [DecidableRel (· < · : α → α → Prop)] [DecidableRel (· < · : β → β → Prop)]
    [DecidableEq R] [DecidableRel (· < · : R → R → Prop)]
    (a : α → R) (b : β → R) (i1 i2 : α) (j : β) :
    prec a i1 i2 ↔ precPair a b (i1, j) (i2, j) := by
  dsimp [prec, precPair, pairScore, lexLT]
  rw [add_lt_add_iff_right, add_left_inj]
  simp

/-! ## 3. Headline 1: `product_topk_subset` -/

/-- **Headline 1: Product top-k subset inclusion.**
For any additive scores `a : Fin n → R`, `b : Fin m → R` and integer `k`, every element of the
top-`k` of `(i, j) ↦ a i + b j` lies in `topk a k × topk b k` under consistent tie-breaking. -/
theorem product_topk_subset {n m : ℕ} {R : Type*}
    [AddCommGroup R] [LinearOrder R] [IsOrderedAddMonoid R]
    [DecidableEq R] [DecidableRel (· < · : R → R → Prop)]
    (a : Fin n → R) (b : Fin m → R) (k : ℕ) :
    ∀ p ∈ topk (Finset.univ : Finset (Fin n × Fin m)) (precPair a b) k,
      p ∈ (topk (Finset.univ : Finset (Fin n)) (prec a) k) ×ˢ
          (topk (Finset.univ : Finset (Fin m)) (prec b) k) := by
  rintro ⟨i, j⟩ hp
  rw [topk, Finset.mem_filter] at hp
  obtain ⟨_, hcard⟩ := hp
  rw [Finset.mem_product]
  constructor
  · rw [topk, Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    by_contra! hle
    let S := predecessors (Finset.univ : Finset (Fin n)) (prec a) i
    have hSle : k ≤ S.card := hle
    let g : Fin n → Fin n × Fin m := fun i' => (i', j)
    have hginj : Function.Injective g := fun _ _ h => (Prod.mk.inj h).1
    have hsub : S.image g ⊆ predecessors (Finset.univ : Finset (Fin n × Fin m)) (precPair a b) (i, j) := by
      intro p hp'
      rw [Finset.mem_image] at hp'
      obtain ⟨i', hi', rfl⟩ := hp'
      dsimp [S, predecessors] at hi' ⊢
      rw [Finset.mem_filter] at hi' ⊢
      refine ⟨Finset.mem_univ _, ?_⟩
      exact (prec_fst_iff a b i' i j).mp hi'.2
    have hcard_le : (S.image g).card ≤ (predecessors (Finset.univ : Finset (Fin n × Fin m)) (precPair a b) (i, j)).card :=
      Finset.card_le_card hsub
    rw [Finset.card_image_of_injective S hginj] at hcard_le
    omega
  · rw [topk, Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    by_contra! hle
    let S := predecessors (Finset.univ : Finset (Fin m)) (prec b) j
    have hSle : k ≤ S.card := hle
    let f : Fin m → Fin n × Fin m := fun j' => (i, j')
    have hfinj : Function.Injective f := fun _ _ h => (Prod.mk.inj h).2
    have hsub : S.image f ⊆ predecessors (Finset.univ : Finset (Fin n × Fin m)) (precPair a b) (i, j) := by
      intro p hp'
      rw [Finset.mem_image] at hp'
      obtain ⟨j', hj', rfl⟩ := hp'
      dsimp [S, predecessors] at hj' ⊢
      rw [Finset.mem_filter] at hj' ⊢
      refine ⟨Finset.mem_univ _, ?_⟩
      exact (prec_snd_iff a b i j' j).mp hj'.2
    have hcard_le : (S.image f).card ≤ (predecessors (Finset.univ : Finset (Fin n × Fin m)) (precPair a b) (i, j)).card :=
      Finset.card_le_card hsub
    rw [Finset.card_image_of_injective S hfinj] at hcard_le
    omega

/-! ## 4. Strict Order Properties and Exactness -/

theorem lexLT_irrefl {α β : Type*} [LinearOrder α] [LinearOrder β] (p : α × β) : ¬ lexLT p p := by
  rintro (h1 | ⟨_, h2⟩)
  · exact lt_irrefl _ h1
  · exact lt_irrefl _ h2

theorem lexLT_trans {α β : Type*} [LinearOrder α] [LinearOrder β] {p1 p2 p3 : α × β}
    (h12 : lexLT p1 p2) (h23 : lexLT p2 p3) : lexLT p1 p3 := by
  dsimp [lexLT] at *
  rcases h12 with h1 | ⟨heq1, h2⟩
  · rcases h23 with h3 | ⟨heq2, _⟩
    · left; exact lt_trans h1 h3
    · left; rw [← heq2]; exact h1
  · rcases h23 with h3 | ⟨heq2, h4⟩
    · left; rw [heq1]; exact h3
    · right; exact ⟨heq1.trans heq2, lt_trans h2 h4⟩

theorem lexLT_trichotomy {α β : Type*} [LinearOrder α] [LinearOrder β] (p1 p2 : α × β) :
    p1 = p2 ∨ lexLT p1 p2 ∨ lexLT p2 p1 := by
  dsimp [lexLT]
  rcases lt_trichotomy p1.1 p2.1 with h1 | heq1 | h2
  · right; left; left; exact h1
  · rcases lt_trichotomy p1.2 p2.2 with h3 | heq2 | h4
    · right; left; right; exact ⟨heq1, h3⟩
    · left; exact Prod.ext heq1 heq2
    · right; right; right; exact ⟨heq1.symm, h4⟩
  · right; right; left; exact h2

theorem precPair_irrefl {α β R : Type*} [LinearOrder α] [LinearOrder β] [Add R] [LinearOrder R]
    (a : α → R) (b : β → R) (p : α × β) : ¬ precPair a b p p := by
  dsimp [precPair]
  rintro (h1 | ⟨_, h2⟩)
  · exact lt_irrefl _ h1
  · exact lexLT_irrefl p h2

theorem precPair_trans {α β R : Type*} [LinearOrder α] [LinearOrder β] [Add R] [LinearOrder R]
    (a : α → R) (b : β → R) {p1 p2 p3 : α × β}
    (h12 : precPair a b p1 p2) (h23 : precPair a b p2 p3) :
    precPair a b p1 p3 := by
  dsimp [precPair] at *
  rcases h12 with hs1 | ⟨heqs1, hl1⟩
  · rcases h23 with hs2 | ⟨heqs2, _⟩
    · left; exact lt_trans hs2 hs1
    · left; rw [← heqs2]; exact hs1
  · rcases h23 with hs2 | ⟨heqs2, hl2⟩
    · left; rw [heqs1]; exact hs2
    · right; exact ⟨heqs1.trans heqs2, lexLT_trans hl1 hl2⟩

theorem precPair_trichotomy {α β R : Type*} [LinearOrder α] [LinearOrder β] [Add R] [LinearOrder R]
    (a : α → R) (b : β → R) (p1 p2 : α × β) :
    p1 = p2 ∨ precPair a b p1 p2 ∨ precPair a b p2 p1 := by
  dsimp [precPair]
  rcases lt_trichotomy (pairScore a b p1) (pairScore a b p2) with hs1 | hseq | hs2
  · right; right; left; exact hs1
  · rcases lexLT_trichotomy p1 p2 with heq | hl1 | hl2
    · left; exact heq
    · right; left; right; exact ⟨hseq, hl1⟩
    · right; right; right; exact ⟨hseq.symm, hl2⟩
  · right; left; left; exact hs2

/-- Existence of a minimum element in any nonempty finset under a strict total order. -/
theorem exists_min_prec {X : Type*} [DecidableEq X] (precRel : X → X → Prop)
    (htrans : ∀ x y z, precRel x y → precRel y z → precRel x z)
    (htrich : ∀ x y, x = y ∨ precRel x y ∨ precRel y x)
    (s : Finset X) (hs : s.Nonempty) :
    ∃ m ∈ s, ∀ y ∈ s, y ≠ m → precRel m y := by
  induction s using Finset.induction_on with
  | empty => exact (Finset.nonempty_iff_ne_empty.mp hs rfl).elim
  | @insert a s _ ih =>
    by_cases hs_empty : s = ∅
    · subst hs_empty
      refine ⟨a, Finset.mem_insert_self a ∅, ?_⟩
      intro y hy hne
      simp at hy
      subst hy
      exact (hne rfl).elim
    · have hs_nonempty : s.Nonempty := Finset.nonempty_iff_ne_empty.mpr hs_empty
      obtain ⟨m, hm, hmin⟩ := ih hs_nonempty
      rcases htrich a m with heq | hlt | hgt
      · subst heq
        refine ⟨a, Finset.mem_insert_self a s, ?_⟩
        intro y hy hne
        rcases Finset.mem_insert.mp hy with rfl | hy'
        · exact (hne rfl).elim
        · exact hmin y hy' hne
      · refine ⟨a, Finset.mem_insert_self a s, ?_⟩
        intro y hy hne
        rcases Finset.mem_insert.mp hy with rfl | hy'
        · exact (hne rfl).elim
        · by_cases hym : y = m
          · subst hym; exact hlt
          · exact htrans a m y hlt (hmin y hy' hym)
      · refine ⟨m, Finset.mem_insert_of_mem hm, ?_⟩
        intro y hy hne
        rcases Finset.mem_insert.mp hy with rfl | hy'
        · exact hgt
        · exact hmin y hy' hne

/-- Pruning exactness: filtering top-`k` over any candidate subset `c` containing the true top-`k`
yields the exact top-`k` of the full universe `u`. -/
theorem topk_exact_of_subset {X : Type*} [DecidableEq X] (precRel : X → X → Prop)
    [DecidableRel precRel]
    (hirr : ∀ x, ¬ precRel x x)
    (htrans : ∀ x y z, precRel x y → precRel y z → precRel x z)
    (htrich : ∀ x y, x = y ∨ precRel x y ∨ precRel y x)
    (u c : Finset X) (hcu : c ⊆ u) (k : ℕ)
    (hsub : topk u precRel k ⊆ c) :
    topk c precRel k = topk u precRel k := by
  ext x
  constructor
  · intro hxc
    rw [topk, Finset.mem_filter] at hxc ⊢
    obtain ⟨hxc_mem, hxc_card⟩ := hxc
    have hxu : x ∈ u := hcu hxc_mem
    refine ⟨hxu, ?_⟩
    by_contra! hle
    have hyt : ∀ y ∈ topk u precRel k, precRel y x := by
      intro y hy
      rw [topk, Finset.mem_filter] at hy
      obtain ⟨_, hy_card⟩ := hy
      rcases htrich x y with rfl | hxy | hyx
      · omega
      · have hsub_pred : predecessors u precRel x ⊆ predecessors u precRel y := by
          intro z hz
          rw [predecessors, Finset.mem_filter] at hz ⊢
          refine ⟨hz.1, htrans z x y hz.2 hxy⟩
        have h_insert : insert x (predecessors u precRel x) ⊆ predecessors u precRel y := by
          intro z hz
          rcases Finset.mem_insert.mp hz with rfl | hz
          · rw [predecessors, Finset.mem_filter]
            exact ⟨hxu, hxy⟩
          · exact hsub_pred hz
        have hx_not_mem : x ∉ predecessors u precRel x := by
          intro hx
          rw [predecessors, Finset.mem_filter] at hx
          exact hirr x hx.2
        have hcard_le := Finset.card_le_card h_insert
        rw [Finset.card_insert_of_notMem hx_not_mem] at hcard_le
        omega
      · exact hyx
    have hsub_c : topk u precRel k ⊆ predecessors c precRel x := by
      intro y hy
      rw [predecessors, Finset.mem_filter]
      exact ⟨hsub hy, hyt y hy⟩
    have hcard_tk_le : (topk u precRel k).card ≤ (predecessors c precRel x).card :=
      Finset.card_le_card hsub_c
    have htk_lt_k : (topk u precRel k).card < k := by omega
    let W := u \ topk u precRel k
    have hW_nonempty : W.Nonempty := by
      refine ⟨x, ?_⟩
      rw [Finset.mem_sdiff, topk, Finset.mem_filter, not_and]
      exact ⟨hxu, fun _ _ => by omega⟩
    obtain ⟨m, hm, hmin⟩ := exists_min_prec precRel htrans htrich W hW_nonempty
    rw [Finset.mem_sdiff, topk, Finset.mem_filter, not_and] at hm
    obtain ⟨hmu, hm_not_tk⟩ := hm
    have hm_le : k ≤ (predecessors u precRel m).card := by
      by_contra!
      exact hm_not_tk hmu this
    have ⟨z, hzm, hz_not_tk⟩ : ∃ z ∈ predecessors u precRel m, z ∉ topk u precRel k := by
      by_contra! hsub_preds
      have := Finset.card_le_card hsub_preds
      omega
    rw [predecessors, Finset.mem_filter] at hzm
    have hz_in_W : z ∈ W := by
      rw [Finset.mem_sdiff]
      exact ⟨hzm.1, hz_not_tk⟩
    have hz_ne_m : z ≠ m := by
      rintro rfl
      exact hirr z hzm.2
    have hm_prec_z := hmin z hz_in_W hz_ne_m
    have hcontra := htrans z m z hzm.2 hm_prec_z
    exact (hirr z hcontra).elim
  · intro hxu
    rw [topk, Finset.mem_filter] at hxu ⊢
    obtain ⟨hxu_mem, hxu_card⟩ := hxu
    refine ⟨hsub (by rw [topk, Finset.mem_filter]; exact ⟨hxu_mem, hxu_card⟩), ?_⟩
    have hsub_preds : predecessors c precRel x ⊆ predecessors u precRel x := by
      intro y hy
      rw [predecessors, Finset.mem_filter] at hy ⊢
      exact ⟨hcu hy.1, hy.2⟩
    have := Finset.card_le_card hsub_preds
    omega

/-! ## 5. Headline 2: `product_topk_exact` -/

/-- **Headline 2: Product top-k exactness.**
The top-`k` of the full `n · m` grid equals the top-`k` of the `k · k` candidate grid:
searching `n + m` keys and `k²` candidates is exact. -/
theorem product_topk_exact {n m : ℕ} {R : Type*}
    [AddCommGroup R] [LinearOrder R] [IsOrderedAddMonoid R]
    [DecidableEq R] [DecidableRel (· < · : R → R → Prop)]
    (a : Fin n → R) (b : Fin m → R) (k : ℕ) :
    topk ((topk (Finset.univ : Finset (Fin n)) (prec a) k) ×ˢ
          (topk (Finset.univ : Finset (Fin m)) (prec b) k))
      (precPair a b) k =
    topk (Finset.univ : Finset (Fin n × Fin m)) (precPair a b) k := by
  have hsub : topk (Finset.univ : Finset (Fin n × Fin m)) (precPair a b) k ⊆
      (topk (Finset.univ : Finset (Fin n)) (prec a) k) ×ˢ
      (topk (Finset.univ : Finset (Fin m)) (prec b) k) := by
    intro p hp
    exact product_topk_subset a b k p hp
  have hcu : (topk (Finset.univ : Finset (Fin n)) (prec a) k) ×ˢ
      (topk (Finset.univ : Finset (Fin m)) (prec b) k) ⊆ Finset.univ :=
    Finset.subset_univ _
  exact topk_exact_of_subset (precPair a b)
    (precPair_irrefl a b)
    (fun _ _ _ => precPair_trans a b)
    (precPair_trichotomy a b)
    Finset.univ _ hcu k hsub

/-! ## 6. Headline 3: Multi-product Top-`k` over `d` Codebooks -/

/-- Sum of scores over `d` codebooks: `∑ t, a t (x t)`. -/
def multiScore {d n : ℕ} {R : Type*} [AddCommMonoid R]
    (a : Fin d → Fin n → R) (x : Fin d → Fin n) : R :=
  ∑ t, a t (x t)

/-- Dictionary order on tuples: first coordinate of disagreement has smaller index. -/
def tupleLexLT {d n : ℕ} (x y : Fin d → Fin n) : Prop :=
  ∃ t : Fin d, x t < y t ∧ ∀ t' : Fin d, t' < t → x t' = y t'

/-- Ranking relation on `d`-tuples: higher sum score wins; ties broken by dictionary order. -/
def precMulti {d n : ℕ} {R : Type*} [AddCommGroup R] [LinearOrder R]
    (a : Fin d → Fin n → R) (x y : Fin d → Fin n) : Prop :=
  multiScore a y < multiScore a x ∨
    (multiScore a x = multiScore a y ∧ tupleLexLT x y)

noncomputable instance {d n : ℕ} {R : Type*} [AddCommGroup R] [LinearOrder R]
    (a : Fin d → Fin n → R) : DecidableRel (precMulti a) :=
  fun _ _ => Classical.dec _

theorem multi_score_update {d n : ℕ} {R : Type*} [AddCommGroup R]
    (a : Fin d → Fin n → R) (x : Fin d → Fin n) (t : Fin d) (y : Fin n) :
    multiScore a (Function.update x t y) - multiScore a x = a t y - a t (x t) := by
  dsimp [multiScore]
  rw [← Finset.sum_sub_distrib]
  have h_eq : ∀ t' ∈ Finset.univ, t' ≠ t →
      a t' (Function.update x t y t') - a t' (x t') = 0 := by
    intro t' _ hne
    rw [Function.update_of_ne hne, sub_self]
  rw [Finset.sum_eq_single t]
  · rw [Function.update_self]
  · intro t' ht' hne
    exact h_eq t' ht' hne
  · intro ht
    exact (ht (Finset.mem_univ t)).elim

theorem multi_score_update_lt {d n : ℕ} {R : Type*} [AddCommGroup R] [LinearOrder R] [IsOrderedAddMonoid R]
    (a : Fin d → Fin n → R) (x : Fin d → Fin n) (t : Fin d) (y : Fin n) :
    a t (x t) < a t y ↔ multiScore a x < multiScore a (Function.update x t y) := by
  have h := multi_score_update a x t y
  constructor
  · intro hlt
    have : 0 < a t y - a t (x t) := sub_pos.mpr hlt
    rw [← h] at this
    exact sub_pos.mp this
  · intro hlt
    have : 0 < multiScore a (Function.update x t y) - multiScore a x := sub_pos.mpr hlt
    rw [h] at this
    exact sub_pos.mp this

theorem multi_score_update_eq {d n : ℕ} {R : Type*} [AddCommGroup R] [LinearOrder R]
    (a : Fin d → Fin n → R) (x : Fin d → Fin n) (t : Fin d) (y : Fin n) :
    a t y = a t (x t) ↔ multiScore a (Function.update x t y) = multiScore a x := by
  have h := multi_score_update a x t y
  constructor
  · intro heq
    have : a t y - a t (x t) = 0 := sub_eq_zero.mpr heq
    rw [← h] at this
    exact sub_eq_zero.mp this
  · intro heq
    have : multiScore a (Function.update x t y) - multiScore a x = 0 := sub_eq_zero.mpr heq
    rw [h] at this
    exact sub_eq_zero.mp this

theorem tupleLexLT_update {d n : ℕ} (x : Fin d → Fin n) (t : Fin d) {y : Fin n} (hy : y < x t) :
    tupleLexLT (Function.update x t y) x := by
  refine ⟨t, ?_, ?_⟩
  · rw [Function.update_self]; exact hy
  · intro t' ht'
    rw [Function.update_of_ne (ne_of_lt ht')]

theorem prec_multi_update {d n : ℕ} {R : Type*} [AddCommGroup R] [LinearOrder R] [IsOrderedAddMonoid R]
    [DecidableEq R] [DecidableRel (· < · : R → R → Prop)]
    (a : Fin d → Fin n → R) (x : Fin d → Fin n) (t : Fin d) (y : Fin n)
    (hy : prec (a t) y (x t)) :
    precMulti a (Function.update x t y) x := by
  rcases hy with hlt | ⟨heq, hlt_idx⟩
  · left
    exact (multi_score_update_lt a x t y).mp hlt
  · right
    refine ⟨(multi_score_update_eq a x t y).mp heq, tupleLexLT_update x t hlt_idx⟩

/-- **Headline 3: Multi-product top-k subset inclusion.**
For scores summed across `d` codebooks, the top-`k` of `∑ t, a t (x t)` lies in `∏ t, topk (a t) k`. -/
theorem multi_product_topk {d n : ℕ} {R : Type*}
    [AddCommGroup R] [LinearOrder R] [IsOrderedAddMonoid R]
    [DecidableEq R] [DecidableRel (· < · : R → R → Prop)]
    (a : Fin d → Fin n → R) (k : ℕ) :
    ∀ x ∈ topk (Finset.univ : Finset (Fin d → Fin n)) (precMulti a) k,
      ∀ t : Fin d, x t ∈ topk (Finset.univ : Finset (Fin n)) (prec (a t)) k := by
  intro x hx t
  rw [topk, Finset.mem_filter] at hx ⊢
  obtain ⟨_, hx_card⟩ := hx
  refine ⟨Finset.mem_univ _, ?_⟩
  by_contra! hle
  let S := predecessors (Finset.univ : Finset (Fin n)) (prec (a t)) (x t)
  have hSle : k ≤ S.card := hle
  let g : Fin n → (Fin d → Fin n) := fun y => Function.update x t y
  have hginj : Function.Injective g := Function.update_injective x t
  have hsub : S.image g ⊆ predecessors (Finset.univ : Finset (Fin d → Fin n)) (precMulti a) x := by
    intro p hp
    rw [Finset.mem_image] at hp
    obtain ⟨y, hy, rfl⟩ := hp
    dsimp [S, predecessors] at hy
    rw [Finset.mem_filter] at hy
    rw [predecessors, Finset.mem_filter]
    refine ⟨Finset.mem_univ _, prec_multi_update a x t y hy.2⟩
  have hcard_le : (S.image g).card ≤ (predecessors (Finset.univ : Finset (Fin d → Fin n)) (precMulti a) x).card :=
    Finset.card_le_card hsub
  rw [Finset.card_image_of_injective S hginj] at hcard_le
  omega

/-- Total slot count for `d` codebooks of size `n`: `N = n^d`. -/
theorem multi_slot_card (d n : ℕ) :
    Fintype.card (Fin d → Fin n) = n ^ d := by
  rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_fin]

/-- Candidate grid size for `d` codebooks of size `n` with top-`k` selection: `k^d`. -/
theorem multi_candidate_card (d : ℕ) {n : ℕ} (S : Fin d → Finset (Fin n)) (k : ℕ)
    (hS : ∀ t, (S t).card = k) :
    Fintype.card (∀ t : Fin d, S t) = k ^ d := by
  rw [Fintype.card_pi]
  have hS' : ∀ t : Fin d, Fintype.card (S t) = k := fun t => by
    rw [Fintype.card_coe, hS t]
  simp [hS']

/-- Search cost comparison: evaluating `d` codebooks of size `n` plus `k^d` candidate slots
takes `d · n + k^d` key evaluations, whereas full search over `N = n^d` slots takes `n^d`. -/
def pkm_candidate_cost (d n k : ℕ) : ℕ := d * n + k ^ d

theorem pkm_cost_formula (d n k : ℕ) :
    pkm_candidate_cost d n k = d * n + k ^ d := rfl

/-! ## 7. Headline 4: `slot_address_bijective` -/

/-- **Headline 4: Bijective unfolding of the torus.**
The address map `(i, j) ↦ i · m + j` is a bijection `Fin n × Fin m ≃ Fin (n · m)`:
the 2-torus unfolds into a linear table. -/
def slot_address_bijective (n m : ℕ) : Fin n × Fin m ≃ Fin (n * m) :=
  finProdFinEquiv

/-- Value formula for slot addressing: `(i, j)` maps to index `i · m + j`. -/
theorem slot_address_val (n m : ℕ) (p : Fin n × Fin m) :
    ((slot_address_bijective n m p : Fin (n * m)) : ℕ) = p.1.val * m + p.2.val := by
  dsimp [slot_address_bijective, finProdFinEquiv]
  rw [Nat.add_comm, Nat.mul_comm]

/-! ## 8. Witness: `n = m = 4, k = 2` on `ℤ` -/

def a_wit : Fin 4 → ℤ
  | ⟨0, _⟩ => 10
  | ⟨1, _⟩ => 40
  | ⟨2, _⟩ => 20
  | ⟨3, _⟩ => 30

def b_wit : Fin 4 → ℤ
  | ⟨0, _⟩ => 1
  | ⟨1, _⟩ => 4
  | ⟨2, _⟩ => 2
  | ⟨3, _⟩ => 3

def cand_wit : Finset (Fin 4 × Fin 4) :=
  (topk Finset.univ (prec a_wit) 2) ×ˢ (topk Finset.univ (prec b_wit) 2)

theorem witness_cand_card : cand_wit.card = 4 := by
  decide

/-- **Witness**: brute-force top-2 over 16 slots equals top-2 over the 4 candidates (decide). -/
theorem witness_top2_eq :
    topk (Finset.univ : Finset (Fin 4 × Fin 4)) (precPair a_wit b_wit) 2 =
    topk cand_wit (precPair a_wit b_wit) 2 := by
  decide

end ProductKeys
end Seq
end Nous
