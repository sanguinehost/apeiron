import Mathlib

namespace Nous.Core

variable {n : Nat} {α : Type*}

/-- The comparison key of candidate `i` under the cost vector `c`: the cost first, the index
second, compared lexicographically. -/
def rankKey [LinearOrder α] (c : Fin n → α) (i : Fin n) : Lex (α × Fin n) :=
  toLex (c i, i)

/-- `stableRank c i = 1 + #{j | rankKey c j < rankKey c i}`: the position of `i` in the ascending
order of `(cost, index)`. Equal costs are broken by the lower index. -/
def stableRank [LinearOrder α] (c : Fin n → α) (i : Fin n) : Nat :=
  (Finset.univ.filter (fun j => rankKey c j < rankKey c i)).card + 1

section Basic

variable [LinearOrder α]

theorem rankKey_inj {c : Fin n → α} {i j : Fin n} :
    rankKey c i = rankKey c j ↔ c i = c j ∧ i = j := by
  simp [rankKey, Prod.mk.injEq]

theorem rankKey_injective (c : Fin n → α) : Function.Injective (rankKey c) := by
  intro i j h
  exact (rankKey_inj.mp h).2

theorem stableRank_pos (c : Fin n → α) (i : Fin n) : 1 ≤ stableRank c i := by
  simp only [stableRank]; omega

theorem stableRank_le (c : Fin n → α) (i : Fin n) : stableRank c i ≤ n := by
  have hn : 0 < n := lt_of_le_of_lt (Nat.zero_le _) i.isLt
  have hsub : Finset.univ.filter (fun j => rankKey c j < rankKey c i) ⊆ Finset.univ.erase i := by
    intro j hj
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj
    simp only [Finset.mem_erase, Finset.mem_univ, and_true]
    rintro rfl
    exact absurd hj (lt_irrefl _)
  have hle : (Finset.univ.filter (fun j => rankKey c j < rankKey c i)).card ≤ (Finset.univ.erase i).card :=
    Finset.card_le_card hsub
  have hcard : (Finset.univ.erase i).card = n - 1 := by
    rw [Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ, Fintype.card_fin]
  simp only [stableRank]
  omega

theorem stableRank_lt_of_lt {c : Fin n → α} {i j : Fin n}
    (h : rankKey c i < rankKey c j) : stableRank c i < stableRank c j := by
  have hsub : (Finset.univ.filter (fun k => rankKey c k < rankKey c i))
      ⊆ (Finset.univ.filter (fun k => rankKey c k < rankKey c j)) := by
    intro k hk
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk ⊢
    exact lt_trans hk h
  have hi : i ∈ Finset.univ.filter (fun k => rankKey c k < rankKey c j) := by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]; exact h
  have hii : i ∉ Finset.univ.filter (fun k => rankKey c k < rankKey c i) := by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_lt]
    exact le_rfl
  have hssub : (Finset.univ.filter (fun k => rankKey c k < rankKey c i))
      ⊂ (Finset.univ.filter (fun k => rankKey c k < rankKey c j)) :=
    Finset.ssubset_iff_subset_ne.mpr ⟨hsub, fun heq => hii (heq ▸ hi)⟩
  have := Finset.card_lt_card hssub
  simp only [stableRank]
  omega

theorem stableRank_injective (c : Fin n → α) : Function.Injective (stableRank c) := by
  intro i j hij
  rcases lt_trichotomy (rankKey c i) (rankKey c j) with h | h | h
  · exact absurd hij (ne_of_lt (stableRank_lt_of_lt h))
  · exact rankKey_injective c h
  · exact absurd hij.symm (ne_of_lt (stableRank_lt_of_lt h))

theorem stableRank_scale_free {β : Type*} [LinearOrder β] (c : Fin n → α) (φ : α → β)
    (hφ : StrictMono φ) : stableRank (fun i => φ (c i)) = stableRank c := by
  funext i
  simp only [stableRank]
  congr 2
  apply Finset.filter_congr
  intro k _
  simp only [rankKey, Prod.Lex.toLex_lt_toLex, hφ.lt_iff_lt, hφ.injective.eq_iff]

end Basic

section Sum

variable [LinearOrder α]

/-- Candidates ordered by `rankKey`; twice the number of strictly ordered pairs is `n(n-1)`. -/
theorem two_mul_card_lt_pairs (c : Fin n → α) :
    2 * (Finset.univ.filter (fun p : Fin n × Fin n => rankKey c p.1 < rankKey c p.2)).card
      = n * (n - 1) := by
  classical
  set A := Finset.univ.filter (fun p : Fin n × Fin n => rankKey c p.1 < rankKey c p.2) with hA
  set B := Finset.univ.filter (fun p : Fin n × Fin n => rankKey c p.2 < rankKey c p.1) with hB
  set D := Finset.univ.filter (fun p : Fin n × Fin n => rankKey c p.1 = rankKey c p.2) with hD
  have hAB : A.card = B.card := by
    refine Finset.card_bij (fun p _ => (p.2, p.1)) ?_ ?_ ?_
    · intro p hp
      simp only [hA, hB, Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      exact hp
    · intro p _ q _ h
      simp only [Prod.mk.injEq] at h
      exact Prod.ext h.2 h.1
    · intro q hq
      exact ⟨(q.2, q.1), by simpa only [hA, hB, Finset.mem_filter, Finset.mem_univ, true_and] using hq, rfl⟩
  have hDcard : D.card = n := by
    have hb : (Finset.univ : Finset (Fin n)).card = D.card := by
      refine Finset.card_bij (fun i _ => (i, i)) ?_ ?_ ?_
      · intro i _
        rw [hD]
        exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, rfl⟩
      · intro i _ j _ h
        exact congrArg Prod.fst h
      · rintro ⟨a, b⟩ hp
        simp only [hD, Finset.mem_filter, Finset.mem_univ, true_and] at hp
        refine ⟨a, Finset.mem_univ _, ?_⟩
        have hab : a = b := (rankKey_inj.mp hp).2
        subst hab
        rfl
    rw [← hb, Finset.card_univ, Fintype.card_fin]
  have hdisjAB : Disjoint A B := by
    rw [Finset.disjoint_left]
    intro p hp hq
    simp only [hA, hB, Finset.mem_filter, Finset.mem_univ, true_and] at hp hq
    exact absurd hp (not_lt.mpr (le_of_lt hq))
  have hdisjD : Disjoint (A ∪ B) D := by
    rw [Finset.disjoint_left]
    intro p hp hq
    simp only [hD, Finset.mem_filter, Finset.mem_univ, true_and] at hq
    simp only [Finset.mem_union, hA, hB, Finset.mem_filter, Finset.mem_univ, true_and] at hp
    rcases hp with h | h
    · exact absurd hq (ne_of_lt h)
    · exact absurd hq (ne_of_gt h)
  have hunion : A ∪ B ∪ D = Finset.univ := by
    ext p
    simp only [Finset.mem_union, hA, hB, hD, Finset.mem_filter, Finset.mem_univ, true_and]
    refine ⟨fun _ => trivial, fun _ => ?_⟩
    rcases lt_trichotomy (rankKey c p.1) (rankKey c p.2) with h | h | h
    · exact Or.inl (Or.inl h)
    · exact Or.inr h
    · exact Or.inl (Or.inr h)
  have hcard : A.card + B.card + D.card = n * n := by
    rw [← Finset.card_union_of_disjoint hdisjAB, ← Finset.card_union_of_disjoint hdisjD, hunion]
    simp [Finset.card_univ, Fintype.card_prod, Fintype.card_fin]
  have hnn : n * (n - 1) = n * n - n := by
    rcases n with _ | m
    · simp
    · rw [Nat.add_sub_cancel, Nat.mul_succ, Nat.add_sub_cancel]
  have h2A : 2 * A.card = n * n - n := by omega
  omega

theorem stableRank_sum_two_mul (c : Fin n → α) :
    2 * (∑ i, stableRank c i) = n * (n + 1) := by
  have hmain : ∑ i, stableRank c i
      = (Finset.univ.filter (fun p : Fin n × Fin n => rankKey c p.1 < rankKey c p.2)).card + n := by
    have step1 : ∀ i : Fin n,
        stableRank c i = (∑ k : Fin n, if rankKey c k < rankKey c i then 1 else 0) + 1 := by
      intro i
      rw [stableRank, Finset.card_filter]
    have hpair : (∑ i : Fin n, ∑ k : Fin n, if rankKey c k < rankKey c i then 1 else 0)
        = (Finset.univ.filter (fun p : Fin n × Fin n => rankKey c p.1 < rankKey c p.2)).card := by
      rw [Finset.card_filter, Fintype.sum_prod_type, Finset.sum_comm]
    calc ∑ i, stableRank c i
        = ∑ i, ((∑ k : Fin n, if rankKey c k < rankKey c i then 1 else 0) + 1) :=
          Finset.sum_congr rfl (fun i _ => step1 i)
      _ = (∑ i, ∑ k : Fin n, if rankKey c k < rankKey c i then 1 else 0) + n := by
          rw [Finset.sum_add_distrib]; simp [Finset.card_univ, Fintype.card_fin]
      _ = (Finset.univ.filter (fun p : Fin n × Fin n => rankKey c p.1 < rankKey c p.2)).card + n := by
          rw [hpair]
  rw [hmain, Nat.mul_add, two_mul_card_lt_pairs c]
  have h1 : n * (n + 1) = n * (n - 1) + 2 * n := by
    rcases n with _ | m
    · simp
    · rw [Nat.add_sub_cancel]; ring
  rw [h1]

theorem stableRank_sum (c : Fin n → α) : ∑ i, stableRank c i = n * (n + 1) / 2 := by
  refine Nat.eq_div_of_mul_eq_right (by norm_num) ?_
  simpa [mul_comm] using stableRank_sum_two_mul c

/-- The mean-rank exclusion law: on every nonempty candidate set some candidate has
`2 * rank ≥ n + 1`, i.e. rank at least `(n+1)/2`. -/
theorem exists_stableRank_ge (c : Fin n → α) (hn : 0 < n) :
    ∃ i, 2 * stableRank c i ≥ n + 1 := by
  by_contra h
  push Not at h
  have hsum2 : 2 * (∑ i, stableRank c i) = ∑ i, 2 * stableRank c i := by rw [Finset.mul_sum]
  have hlt : ∑ i : Fin n, 2 * stableRank c i < ∑ _i : Fin n, (n + 1) := by
    refine Finset.sum_lt_sum_of_nonempty ⟨⟨0, hn⟩, Finset.mem_univ _⟩ ?_
    intro i _
    have := h i
    omega
  have hrhs : (∑ _i : Fin n, (n + 1)) = n * (n + 1) := by
    simp [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
  rw [hrhs, ← hsum2, stableRank_sum_two_mul] at hlt
  omega

end Sum

end Nous.Core
