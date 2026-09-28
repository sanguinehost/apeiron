/-
Copyright (c) 2026 Apeiron contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Apeiron Formal Methods Group

Certified sparse softmax: exact attention without reading every key.
Task APEIRON-SPARSE-T1 (id sparse-t1). Contract: docs/epics/subquadratic-decode.md.

## Specification

In the fixed-point attention kernel (`src/engine/attn.rs:95-200`, `Numeric.EngineAttnRefinement`),
the windowed exponential `exp_at` returns exactly `0` whenever `score - max < -16·ONE`
(`exp_at_window`). Keys whose scores fall below this underflow threshold contribute nothing to the
denominator or the value mix accumulation. Skipping these keys leaves the softmax output bit-identical.

This module formalises integer semantics over `ℤ` with an abstract exponential `E : ℤ → ℤ` and
underflow window `W : ℤ` (covering `fixed_exp` with `W = 16·ONE`), proving:

1. `prune_max_preserved`: if every pruned key in `P ⊆ S` has score `< M - W` for some `M ≤ max S`
   attained by a kept key `k_kept ∈ S \ P`, and `0 ≤ W`, then `max (S \ P) = max S`.
2. `prune_exact`: for any `P ⊆ S` with scores `< m - W` (where `m = max S`), the windowed denominator,
   every weight of a kept key, and the value mix computed over `S \ P` equal those over `S`, while
   pruned weights are zero.
3. `box_bound_sound`: for vectors `q, k` with `lo ≤ k ≤ hi` componentwise, `q · k ≤ ∑_i max(q_i·lo_i, q_i·hi_i)`.
4. `ball_bound_sound`: over `ℝ`, `‖k - c‖ ≤ r → q · k ≤ q · c + r ‖q‖` (Cauchy–Schwarz).
5. `block_prune_exact`: runtime certificate skipping blocks whose box bound is `< M - W`, preserving
   the maximum score and leaving the attention output unchanged.
6. `witness_six_keys_sparse_softmax_exact`: concrete 6-key instance with `W = 16` where two keys are
   pruned and outputs are equal.

## Mathematical Attribution

- Box bound: coordinate-wise interval arithmetic (Moore, *Interval Analysis*, 1966); used for approximate
  pruning in Quest (Tang et al., 2024; anchor, not retrieved), here certified for exact underflow pruning.
- Ball bound / Cauchy–Schwarz inequality: Cauchy, *Cours d'Analyse* (1821); Schwarz (1888).
- Round-to-nearest integer division: Gauss, *Disquisitiones Arithmeticae* (1801).

## Dominant-Kernel Complexity

Dense softmax attention over `t` keys in dimension `d` evaluates all keys in `O(t · d)` word operations.
When keys are partitioned into `B` blocks, evaluating the block summaries takes `O(B · d)` operations.
If `L` keys survive the bound test (`L ≤ t`), evaluating live keys takes `O(L · d)` operations.
When `L ≪ t`, total evaluation cost drops from `O(t · d)` to `O((B + L) · d)`.
-/

import Mathlib

open scoped BigOperators
open scoped InnerProductSpace

namespace Nous.Seq.CertifiedSoftmax

/-! ## Integer Semantics and Softmax Definitions -/

/-- Half-biased integer division `(n + d / 2) / d`, matching `div_round_nearest` in `attn.rs`
and `Numeric.EngineNormRefinement.divRoundNearest`. -/
def divRoundNearest (n d : ℤ) : ℤ := (n + d / 2) / d

/-- Dividing zero by any positive denominator yields zero. -/
theorem divRoundNearest_zero_of_pos {d : ℤ} (hd : 0 < d) : divRoundNearest 0 d = 0 := by
  unfold divRoundNearest
  rw [zero_add]
  exact Int.ediv_eq_zero_of_lt (by omega) (by omega)

/-- Windowed exponential numerator: returns 0 below the underflow window `-W`,
and `E (score - m)` within the window, matching `exp_at` in `attn.rs` where `W = 16·ONE`. -/
def windowedExp (W : ℤ) (E : ℤ → ℤ) (score_val m : ℤ) : ℤ :=
  if score_val - m < -W then 0 else E (score_val - m)

/-- Softmax denominator: sum of windowed numerators over a key set `S`. -/
def totalDenom {α : Type*} [DecidableEq α] (W : ℤ) (E : ℤ → ℤ) (score : α → ℤ) (m : ℤ) (S : Finset α) : ℤ :=
  ∑ s ∈ S, windowedExp W E (score s) m

/-- Softmax weight for key `k`: scaled numerator divided by total denominator using round-to-nearest. -/
def softmaxWeight {α : Type*} [DecidableEq α] (W : ℤ) (E : ℤ → ℤ) (score : α → ℤ) (m : ℤ) (S : Finset α)
    (scale : ℤ) (k : α) : ℤ :=
  divRoundNearest (windowedExp W E (score k) m * scale) (totalDenom W E score m S)

/-- Value accumulation before output epilogue: `∑_s w_s · v_s`. -/
def valueMixRaw {α : Type*} [DecidableEq α] (W : ℤ) (E : ℤ → ℤ) (score : α → ℤ) (m : ℤ) (S : Finset α)
    (scale : ℤ) (v : α → ℤ) : ℤ :=
  ∑ s ∈ S, softmaxWeight W E score m S scale s * v s

/-- Value mix with output round-to-nearest epilogue against `outScale`. -/
def valueMix {α : Type*} [DecidableEq α] (W : ℤ) (E : ℤ → ℤ) (score : α → ℤ) (m : ℤ) (S : Finset α)
    (scale outScale : ℤ) (v : α → ℤ) : ℤ :=
  divRoundNearest (valueMixRaw W E score m S scale v) outScale

/-- Maximum score attained over a nonempty finite key set. -/
def maxScore {α : Type*} (s : Finset α) (hs : s.Nonempty) (score : α → ℤ) : ℤ :=
  s.sup' hs score

/-! ## Maximum Preservation under Bound Pruning -/

/-- **Maximum preservation.**
If every pruned key in `P ⊆ S` has score strictly less than `M - W` for some `M ≤ max S`
attained by a kept key `k_kept ∈ S \ P`, and `W ≥ 0`, then the maximum score over the
kept keys `S \ P` equals the maximum score over the full set `S`. -/
theorem prune_max_preserved {α : Type*} [DecidableEq α]
    (s p : Finset α) (score : α → ℤ) (W : ℤ) (hW : 0 ≤ W)
    (hs : s.Nonempty) (_hp_sub : p ⊆ s)
    (M : ℤ) (k_kept : α) (hk_kept : k_kept ∈ s \ p) (hM : score k_kept = M)
    (hprune : ∀ k ∈ p, score k < M - W)
    (hsp : (s \ p).Nonempty := ⟨k_kept, hk_kept⟩) :
    maxScore (s \ p) hsp score = maxScore s hs score := by
  unfold maxScore
  apply le_antisymm
  · apply Finset.sup'_le
    intro x hx
    have hxs : x ∈ s := (Finset.mem_sdiff.mp hx).1
    exact Finset.le_sup' score hxs
  · obtain ⟨m_elem, hm_elem_s, hm_elem_max⟩ := Finset.exists_mem_eq_sup' hs score
    have hm_not_in_p : m_elem ∉ p := by
      intro h_in_p
      have hlt := hprune m_elem h_in_p
      have hle : score k_kept ≤ score m_elem := by
        have hk_s : k_kept ∈ s := (Finset.mem_sdiff.mp hk_kept).1
        rw [← hm_elem_max]
        exact Finset.le_sup' score hk_s
      rw [hM] at hle
      omega
    have hm_in_sp : m_elem ∈ s \ p := Finset.mem_sdiff.mpr ⟨hm_elem_s, hm_not_in_p⟩
    rw [hm_elem_max]
    exact Finset.le_sup' score hm_in_sp

/-! ## Exact Pruning of Sub-Window Keys -/

/-- Pruning sub-window keys leaves the windowed denominator identical. -/
theorem totalDenom_prune_eq {α : Type*} [DecidableEq α] (W : ℤ) (E : ℤ → ℤ) (score : α → ℤ) (m : ℤ)
    (S P : Finset α) (hP : P ⊆ S) (hprune : ∀ s ∈ P, score s < m - W) :
    totalDenom W E score m (S \ P) = totalDenom W E score m S := by
  unfold totalDenom
  have hsplit : S = (S \ P) ∪ P := (Finset.sdiff_union_of_subset hP).symm
  have hdisj : Disjoint (S \ P) P := disjoint_sdiff_self_left
  have hsum : (∑ s ∈ S, windowedExp W E (score s) m) =
      (∑ s ∈ S \ P, windowedExp W E (score s) m) +
      (∑ s ∈ P, windowedExp W E (score s) m) := by
    nth_rw 1 [hsplit]
    exact Finset.sum_union hdisj
  rw [hsum]
  have hPzero : ∑ s ∈ P, windowedExp W E (score s) m = 0 := by
    apply Finset.sum_eq_zero
    intro s hs
    have hlt := hprune s hs
    unfold windowedExp
    have hdiff : score s - m < -W := by omega
    rw [if_pos hdiff]
  rw [hPzero, add_zero]

/-- Softmax weights of pruned keys are exactly zero. -/
theorem weight_pruned_zero {α : Type*} [DecidableEq α] (W : ℤ) (E : ℤ → ℤ) (score : α → ℤ) (m : ℤ)
    (S : Finset α) (scale : ℤ) (s : α) (hprune : score s < m - W)
    (htot : 0 < totalDenom W E score m S) :
    softmaxWeight W E score m S scale s = 0 := by
  unfold softmaxWeight
  have hdiff : score s - m < -W := by omega
  have hnum : windowedExp W E (score s) m = 0 := by
    unfold windowedExp; rw [if_pos hdiff]
  rw [hnum, zero_mul]
  exact divRoundNearest_zero_of_pos htot

/-- Softmax weights of kept keys computed over `S \ P` equal those computed over `S`. -/
theorem weight_kept_eq {α : Type*} [DecidableEq α] (W : ℤ) (E : ℤ → ℤ) (score : α → ℤ) (m : ℤ)
    (S P : Finset α) (scale : ℤ) (k : α) (_hk : k ∈ S \ P)
    (hP : P ⊆ S) (hprune : ∀ s ∈ P, score s < m - W) :
    softmaxWeight W E score m (S \ P) scale k = softmaxWeight W E score m S scale k := by
  unfold softmaxWeight
  rw [totalDenom_prune_eq W E score m S P hP hprune]

/-- Raw value mix computed over `S \ P` equals that over `S`. -/
theorem valueMixRaw_prune_eq {α : Type*} [DecidableEq α] (W : ℤ) (E : ℤ → ℤ) (score : α → ℤ) (m : ℤ)
    (S P : Finset α) (scale : ℤ) (v : α → ℤ)
    (hP : P ⊆ S) (hprune : ∀ s ∈ P, score s < m - W)
    (htot : 0 < totalDenom W E score m S) :
    valueMixRaw W E score m (S \ P) scale v = valueMixRaw W E score m S scale v := by
  unfold valueMixRaw
  have hsplit : S = (S \ P) ∪ P := (Finset.sdiff_union_of_subset hP).symm
  have hdisj : Disjoint (S \ P) P := disjoint_sdiff_self_left
  have hsum : (∑ s ∈ S, softmaxWeight W E score m S scale s * v s) =
      (∑ s ∈ S \ P, softmaxWeight W E score m S scale s * v s) +
      (∑ s ∈ P, softmaxWeight W E score m S scale s * v s) := by
    nth_rw 1 [hsplit]
    exact Finset.sum_union hdisj
  rw [hsum]
  have hsum_pruned : ∑ s ∈ P, softmaxWeight W E score m S scale s * v s = 0 := by
    apply Finset.sum_eq_zero
    intro s hs
    have hw := weight_pruned_zero W E score m S scale s (hprune s hs) htot
    rw [hw, zero_mul]
  rw [hsum_pruned, add_zero]
  apply Finset.sum_congr rfl
  intro s hs
  rw [weight_kept_eq W E score m S P scale s hs hP hprune]

/-- Value mix with epilogue computed over `S \ P` equals that over `S`. -/
theorem valueMix_prune_eq {α : Type*} [DecidableEq α] (W : ℤ) (E : ℤ → ℤ) (score : α → ℤ) (m : ℤ)
    (S P : Finset α) (scale outScale : ℤ) (v : α → ℤ)
    (hP : P ⊆ S) (hprune : ∀ s ∈ P, score s < m - W)
    (htot : 0 < totalDenom W E score m S) :
    valueMix W E score m (S \ P) scale outScale v = valueMix W E score m S scale outScale v := by
  unfold valueMix
  rw [valueMixRaw_prune_eq W E score m S P scale v hP hprune htot]

/-- **Certified sparse softmax: exact pruning.**
Let `S` be the key set, `m = max S`. For any subset `P ⊆ S` of keys with `score s < m - W`,
the windowed denominator, every weight of a kept key, and the value mix computed over
`S \ P` equal those over `S`, and pruned weights are zero. -/
theorem prune_exact {α : Type*} [DecidableEq α] (W : ℤ) (E : ℤ → ℤ) (score : α → ℤ) (m : ℤ)
    (S P : Finset α) (scale outScale : ℤ) (v : α → ℤ)
    (hP : P ⊆ S) (hprune : ∀ s ∈ P, score s < m - W)
    (htot : 0 < totalDenom W E score m S) :
    totalDenom W E score m (S \ P) = totalDenom W E score m S ∧
    (∀ s ∈ P, softmaxWeight W E score m S scale s = 0) ∧
    (∀ k ∈ S \ P, softmaxWeight W E score m (S \ P) scale k = softmaxWeight W E score m S scale k) ∧
    valueMix W E score m (S \ P) scale outScale v = valueMix W E score m S scale outScale v :=
  ⟨totalDenom_prune_eq W E score m S P hP hprune,
   fun s hs => weight_pruned_zero W E score m S scale s (hprune s hs) htot,
   fun k hk => weight_kept_eq W E score m S P scale k hk hP hprune,
   valueMix_prune_eq W E score m S P scale outScale v hP hprune htot⟩

/-! ## Geometric Bounds: Box Bound and Ball Bound -/

/-- Dot product of two integer vectors. -/
def dotProd {d : ℕ} (q k : Fin d → ℤ) : ℤ :=
  ∑ i : Fin d, q i * k i

/-- Box upper bound on query-key dot product: `∑_i max(q_i·lo_i, q_i·hi_i)`. -/
def boxBound {d : ℕ} (q lo hi : Fin d → ℤ) : ℤ :=
  ∑ i : Fin d, max (q i * lo i) (q i * hi i)

/-- **Box bound soundness.**
For vectors `q, k` with `lo ≤ k ≤ hi` componentwise, `q · k ≤ ∑_i max(q_i·lo_i, q_i·hi_i)`. -/
theorem box_bound_sound {d : ℕ} (q k lo hi : Fin d → ℤ)
    (hlo : ∀ i, lo i ≤ k i) (hhi : ∀ i, k i ≤ hi i) :
    dotProd q k ≤ boxBound q lo hi := by
  unfold dotProd boxBound
  apply Finset.sum_le_sum
  intro i _
  by_cases hq : 0 ≤ q i
  · have hmul : q i * k i ≤ q i * hi i := mul_le_mul_of_nonneg_left (hhi i) hq
    exact le_trans hmul (le_max_right _ _)
  · have hmul : q i * k i ≤ q i * lo i := by
      have : q i ≤ 0 := by omega
      nlinarith [hlo i]
    exact le_trans hmul (le_max_left _ _)

/-- **Ball bound soundness over ℝ (Cauchy–Schwarz).**
If `‖k - c‖ ≤ r`, then `q · k ≤ q · c + r ‖q‖`. -/
theorem ball_bound_sound {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (q k c : E) (r : ℝ) (hr : ‖k - c‖ ≤ r) :
    inner ℝ q k ≤ inner ℝ q c + r * ‖q‖ := by
  have heq : k = (k - c) + c := by abel
  rw [heq, inner_add_right]
  have hcs := real_inner_le_norm q (k - c)
  have hmul : ‖q‖ * ‖k - c‖ ≤ ‖q‖ * r := mul_le_mul_of_nonneg_left hr (norm_nonneg q)
  linarith [mul_comm ‖q‖ r]

/-! ## Block Certificate and Block Pruning -/

/-- A key block carrying coordinate-wise bounding intervals `[lo, hi]`. -/
structure Block (α : Type*) (d : ℕ) where
  keys : Finset α
  lo   : Fin d → ℤ
  hi   : Fin d → ℤ

/-- Soundness of a block summary with respect to key embeddings: every key lies in the box. -/
def Block.sound {α : Type*} {d : ℕ} (b : Block α d) (keyVec : α → Fin d → ℤ) : Prop :=
  ∀ k ∈ b.keys, (∀ i, b.lo i ≤ keyVec k i) ∧ (∀ i, keyVec k i ≤ b.hi i)

/-- Any key in a block whose box bound is below `M - W` has score strictly below `M - W`. -/
theorem block_keys_score_lt {α : Type*} {d : ℕ} (q : Fin d → ℤ) (keyVec : α → Fin d → ℤ)
    (b : Block α d) (hb : b.sound keyVec)
    (M W : ℤ) (hskip : boxBound q b.lo b.hi < M - W) :
    ∀ k ∈ b.keys, dotProd q (keyVec k) < M - W := by
  intro k hk
  have hsound := hb k hk
  have hbound := box_bound_sound q (keyVec k) b.lo b.hi hsound.1 hsound.2
  omega

/-- **Runtime block-skipping certificate.**
Blocks whose box bound is strictly below `M - W`, where `M` is a score attained by a visited key,
can be skipped entirely: the true maximum is preserved, and the softmax output is unchanged. -/
theorem block_prune_exact {α : Type*} [DecidableEq α] {d : ℕ}
    (W : ℤ) (hW : 0 ≤ W) (E : ℤ → ℤ)
    (q : Fin d → ℤ) (keyVec : α → Fin d → ℤ)
    (S : Finset α) (hs : S.Nonempty)
    (skipped : Finset (Block α d))
    (hsound : ∀ b ∈ skipped, b.sound keyVec)
    (M : ℤ) (k_visited : α)
    (hk_visited_s : k_visited ∈ S)
    (hk_not_skipped : k_visited ∉ skipped.biUnion Block.keys)
    (hM : dotProd q (keyVec k_visited) = M)
    (hskip : ∀ b ∈ skipped, boxBound q b.lo b.hi < M - W)
    (hsub : skipped.biUnion Block.keys ⊆ S)
    (scale outScale : ℤ) (v : α → ℤ)
    (htot : 0 < totalDenom W E (fun k => dotProd q (keyVec k))
      (maxScore S hs (fun k => dotProd q (keyVec k))) S) :
    let P := skipped.biUnion Block.keys
    let score := fun k => dotProd q (keyVec k)
    let m := maxScore S hs score
    haveI hsp : (S \ P).Nonempty := ⟨k_visited, Finset.mem_sdiff.mpr ⟨hk_visited_s, hk_not_skipped⟩⟩
    maxScore (S \ P) hsp score = m ∧
    valueMix W E score m (S \ P) scale outScale v = valueMix W E score m S scale outScale v := by
  have hk_kept : k_visited ∈ S \ (skipped.biUnion Block.keys) :=
    Finset.mem_sdiff.mpr ⟨hk_visited_s, hk_not_skipped⟩
  have hsp : (S \ (skipped.biUnion Block.keys)).Nonempty := ⟨k_visited, hk_kept⟩
  set P := skipped.biUnion Block.keys
  set score := fun k => dotProd q (keyVec k)
  set m := maxScore S hs score
  have hprune : ∀ k ∈ P, score k < M - W := by
    intro k hk
    rw [Finset.mem_biUnion] at hk
    obtain ⟨b, hb, hkb⟩ := hk
    have hb_sound := hsound b hb k hkb
    have hbound := box_bound_sound q (keyVec k) b.lo b.hi hb_sound.1 hb_sound.2
    have hbb := hskip b hb
    change dotProd q (keyVec k) < M - W
    omega
  have hmax_pres : maxScore (S \ P) hsp score = m :=
    prune_max_preserved S P score W hW hs hsub M k_visited hk_kept hM hprune hsp
  have hprune_m : ∀ k ∈ P, score k < m - W := by
    intro k hk
    have hlt := hprune k hk
    have hM_le_m : M ≤ m := by
      unfold m maxScore
      rw [← hM]
      exact Finset.le_sup' score hk_visited_s
    omega
  have hvm := valueMix_prune_eq W E score m S P scale outScale v hsub hprune_m htot
  exact ⟨hmax_pres, hvm⟩

/-! ## Non-Vacuity Witness: 6-Key Instance with W = 16 -/

/-- Underflow window matching Q32.32 `-16·ONE`. -/
def witW : ℤ := 16

/-- Abstract exponential function for the concrete witness. -/
def witE (x : ℤ) : ℤ := x + 20

/-- Six integer scores: keys 0..3 survive (`score ≥ 85 > 84`), keys 4..5 pruned (`score < 84`). -/
def witScore : Fin 6 → ℤ
  | 0 => 100
  | 1 => 95
  | 2 => 90
  | 3 => 85
  | 4 => 80
  | 5 => 70

/-- Six integer values. -/
def witV : Fin 6 → ℤ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5
  | 5 => 6

/-- The full 6-key set. -/
def witS : Finset (Fin 6) := Finset.univ

/-- The two pruned keys {4, 5}. -/
def witP : Finset (Fin 6) := {4, 5}

/-- Maximum score attained by kept key 0: `witScore 0 = 100`. -/
def witM : ℤ := 100

/-- Softmax numerator scale. -/
def witScale : ℤ := 100

/-- Value epilogue output scale. -/
def witOutScale : ℤ := 10

/-- Exactly two keys are pruned in the witness. -/
theorem witness_six_keys_pruned_count : witP.card = 2 := by decide

/-- Both pruned keys have score strictly below `M - W = 100 - 16 = 84`. -/
theorem witness_six_keys_pruned_condition :
    ∀ s ∈ witP, witScore s < witM - witW := by decide

/-- Windowed denominator over kept keys equals denominator over all 6 keys. -/
theorem witness_six_keys_denom_eq :
    totalDenom witW witE witScore witM (witS \ witP) =
    totalDenom witW witE witScore witM witS := by decide

/-- Output value mix over kept keys equals output value mix over all 6 keys. -/
theorem witness_six_keys_output_eq :
    valueMix witW witE witScore witM (witS \ witP) witScale witOutScale witV =
    valueMix witW witE witScore witM witS witScale witOutScale witV := by decide

/-- **Joint 6-key non-vacuity witness.**
A 6-key instance over `ℤ` with `W = 16` where exactly two keys are pruned and the output
value mix is bit-identical. -/
theorem witness_six_keys_sparse_softmax_exact :
    witP.card = 2 ∧
    (∀ s ∈ witP, witScore s < witM - witW) ∧
    totalDenom witW witE witScore witM (witS \ witP) = totalDenom witW witE witScore witM witS ∧
    valueMix witW witE witScore witM (witS \ witP) witScale witOutScale witV =
      valueMix witW witE witScore witM witS witScale witOutScale witV := by
  decide

end Nous.Seq.CertifiedSoftmax
