import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Nous.Seq.HyperbolicAddress
import Nous.Seq.ProductKeys
import Mathlib.Algebra.Order.Ring.Defs
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Tactic

/-!
# Threshold Algorithms for Rank-r Hyperbolic Cell Memories

This module mechanizes exact top-`k` retrieval for memory scores that decompose as a
nonnegative linear combination of `r` hyperbolic cells `R[x]/(x² - σ)` in SVD form.
This corresponds to the Tucker core of UltraMem (Huang et al., 2024, arXiv:2411.12364)
over rank-`r` cores `S_rowᵀ C S_col`, which we make exact by lifting the Fagin–Lotem–Naor
threshold algorithm (Fagin, Lotem, Naor, PODS 2001, "Optimal aggregation algorithms for
middleware") to product-channel spaces.

## Attribution and Lineage

1. **Threshold Algorithm:**
   Fagin, Lotem, and Naor (PODS 2001). The classic TA maintains sorted access to each
   stream and halts when the buffer top-`k` scores exceed a threshold `T` derived from
   the latest seen values. Here, because each channel is multiplicative (`A_l(i) * B_l(j)`),
   the threshold on unseen product pairs `(i, j) ∉ R_l(D) × C_l(D)` requires the joint bound
   `max (α_l * Bmax_l) (Amax_l * β_l)`, not the naive product `α_l * β_l`.

2. **UltraMem Core Representation:**
   UltraMem (Huang et al., 2024) uses Tucker decomposition for fast approximate memory
   addressing. For rank-1 cores, two-phase candidate generation is exact (`rank1_core_is_cell`).
   For rank-`r` cores (`r ≥ 2`), naive marginal search fails (`tucker_rank2_breaks`).
   This module provides the certified exact search mechanism for rank-`r` cores with an
   explicit evaluation bound.

3. **Canonical Trichotomy Dependencies:**
   - `Apeiron.Trichotomy.AnyRing.RingCell`
   - `Apeiron.Trichotomy.AnyRing.ringNormForm`
   - `Apeiron.Trichotomy.AnyRing.ringMul`
   - `Apeiron.Trichotomy.ChannelsAnyRing.ringUpper`
   - `Apeiron.Trichotomy.ChannelsAnyRing.ringLower`
   - `Nous.Seq.HyperbolicAddress.slotCell`
   - `Nous.Seq.HyperbolicAddress.slot_normForm`
   - `Nous.Seq.HyperbolicAddress.slot_mul`
   - `Nous.Seq.ProductKeys.topk`
   - `Nous.Seq.ProductKeys.prec`
   - `Nous.Seq.ProductKeys.precPair`
   - `Nous.Seq.ProductKeys.topk_exact_of_subset`

## Complexity (AGENTS.md §8)

At scan depth `D`:
- Channel evaluations: exactly `r * (n + m)` evaluations of row and column channel vectors.
- Candidate set size: `(Seen D).card ≤ r * D^2` candidate pairs (proved in `seen_card_le`).
- Product cell evaluations: at most `r * D^2` full hyperbolic norm form evaluations.
- Threshold calculation: `O(r)` scalar multiplications and comparisons per depth step.
Total temporal cost: `r * (n + m) + r * D^2` comparisons and multiplications, avoiding
exhaustive `n * m` grid evaluation when `r * D^2 < n * m`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedVariables false
set_option linter.unusedSectionVars false

namespace Nous.Seq.ThresholdCells

open Apeiron.Trichotomy.AnyRing
open Nous.Seq.HyperbolicAddress (slotCell slot_normForm slot_mul)
open Nous.Seq.ProductKeys

/-! ## 1. Strict Total Order Properties of Channel Ranking -/

/-- Irreflexivity of `prec`. -/
theorem prec_irrefl {X R : Type*} [LinearOrder X] [LinearOrder R] (f : X → R) (x : X) :
    ¬ prec f x x := by
  dsimp [prec]
  rintro (h | ⟨_, h2⟩)
  · exact lt_irrefl _ h
  · exact lt_irrefl _ h2

/-- Transitivity of `prec`. -/
theorem prec_trans {X R : Type*} [LinearOrder X] [LinearOrder R] (f : X → R) {x y z : X}
    (hxy : prec f x y) (hyz : prec f y z) : prec f x z := by
  dsimp [prec] at *
  rcases hxy with h1 | ⟨heq1, h2⟩
  · rcases hyz with h3 | ⟨heq2, _⟩
    · left; exact lt_trans h3 h1
    · left; rw [← heq2]; exact h1
  · rcases hyz with h3 | ⟨heq2, h4⟩
    · left; rw [heq1]; exact h3
    · right; exact ⟨heq1.trans heq2, lt_trans h2 h4⟩

/-- Trichotomy of `prec`. -/
theorem prec_trichotomy {X R : Type*} [LinearOrder X] [LinearOrder R] (f : X → R) (x y : X) :
    x = y ∨ prec f x y ∨ prec f y x := by
  dsimp [prec]
  rcases lt_trichotomy (f x) (f y) with h1 | heq | h2
  · right; right; left; exact h1
  · rcases lt_trichotomy x y with hx | heqx | hy
    · right; left; right; exact ⟨heq, hx⟩
    · left; exact heqx
    · right; right; right; exact ⟨heq.symm, hy⟩
  · right; left; left; exact h2

/-- Cardinality bound for `topk` under any strict total order `prec f`.
Every element in `topk s (prec f) k` has `< k` predecessors, so `topk` contains at most `k` elements.
Derivation: if `k < card(topk)`, the minimal element under the reversed order has `≥ k` predecessors
in `topk`, contradicting membership in `topk`. -/
theorem topk_card_le {X R : Type*} [DecidableEq X] [LinearOrder X] [LinearOrder R]
    (s : Finset X) (f : X → R) (k : ℕ) :
    (topk s (prec f) k).card ≤ k := by
  by_contra! hgt
  set tk := topk s (prec f) k with htk_def
  have htk_ne : tk.Nonempty := by
    rw [Finset.nonempty_iff_ne_empty]
    intro he
    have : tk.card = 0 := congrArg Finset.card he
    omega
  let rel : X → X → Prop := fun x y => prec f y x
  have hrel_trans : ∀ x y z, rel x y → rel y z → rel x z := by
    intro x y z hxy hyz
    exact prec_trans f hyz hxy
  have hrel_trich : ∀ x y, x = y ∨ rel x y ∨ rel y x := by
    intro x y
    rcases prec_trichotomy f x y with heq | hxy | hyx
    · left; exact heq
    · right; right; exact hxy
    · right; left; exact hyx
  obtain ⟨m, hm_in, hm_min⟩ := exists_min_prec rel hrel_trans hrel_trich tk htk_ne
  have hm_tk := hm_in
  rw [htk_def, topk, Finset.mem_filter] at hm_tk
  obtain ⟨_, hpred_lt⟩ := hm_tk
  have hsub : tk.erase m ⊆ predecessors s (prec f) m := by
    intro y hy
    rw [Finset.mem_erase] at hy
    have hym : y ≠ m := hy.1
    have hrel := hm_min y hy.2 hym
    dsimp [rel] at hrel
    rw [predecessors, Finset.mem_filter]
    have hy_in_s : y ∈ s := by
      have : y ∈ tk := hy.2
      rw [htk_def, topk, Finset.mem_filter] at this
      exact this.1
    exact ⟨hy_in_s, hrel⟩
  have hcard_sub := Finset.card_le_card hsub
  rw [Finset.card_erase_of_mem hm_in] at hcard_sub
  omega

/-! ## 2. Core Definitions: Score, Candidate Sets, Threshold -/

variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

variable {n m r : ℕ}
variable (w : Fin r → K)
variable (A : Fin r → Fin n → K)
variable (B : Fin r → Fin m → K)
variable (σ ρ : K)

/-- The total memory score of a product slot `p = (i, j)` as a nonnegative weighted sum
of `r` hyperbolic cell norm forms. -/
def score (p : Fin n × Fin m) : K :=
  ∑ l : Fin r, w l * ringNormForm σ (slotCell ρ (A l p.1) (B l p.2))

/-- Top-`D` rows for channel `l` under strict ranking with lowest-index tie-breaking. -/
def R (l : Fin r) (D : ℕ) : Finset (Fin n) :=
  topk (Finset.univ : Finset (Fin n)) (prec (A l)) D

/-- Top-`D` columns for channel `l` under strict ranking with lowest-index tie-breaking. -/
def C (l : Fin r) (D : ℕ) : Finset (Fin m) :=
  topk (Finset.univ : Finset (Fin m)) (prec (B l)) D

/-- Evaluated candidate set `Seen D`: the union over all `r` channels of the 2-D candidate
grids `R l D × C l D`. -/
def Seen (D : ℕ) : Finset (Fin n × Fin m) :=
  (Finset.univ : Finset (Fin r)).biUnion (fun l => (R A l D) ×ˢ (C B l D))

/-- Fagin–Lotem–Naor threshold for rank-`r` multiplicative channels.
For unseen pairs outside `R l D × C l D`, at least one factor is bounded by the respective
cutoff (`α l` or `β l`), yielding upper bound `max (α l * Bmax l) (Amax l * β l)`. -/
def threshold (α β Amax Bmax : Fin r → K) : K :=
  ∑ l : Fin r, w l * max (α l * Bmax l) (Amax l * β l)

/-! ## 3. Headline 1: `score_as_cells` -/

/-- **Headline 1: Score Representation via Canonical Cells.**
The score of slot `(i, j)` is identically `∑_l w_l * (A_l(i) * B_l(j))`, derived
directly from the canonical norm form of `slotCell ρ (A_l i) (B_l j)`. -/
theorem score_as_cells (hr : ρ ^ 2 = σ) (hr0 : ρ ≠ 0) (i : Fin n) (j : Fin m) :
    score w A B σ ρ (i, j) = ∑ l : Fin r, w l * (A l i * B l j) := by
  unfold score
  dsimp only
  congr 1
  ext l
  rw [slot_normForm σ ρ hr hr0]

theorem score_as_cells_pair (hr : ρ ^ 2 = σ) (hr0 : ρ ≠ 0) (p : Fin n × Fin m) :
    score w A B σ ρ p = ∑ l : Fin r, w l * (A l p.1 * B l p.2) := by
  rcases p with ⟨i, j⟩
  exact score_as_cells w A B σ ρ hr hr0 i j

/-! ## 4. Headline 2: `unseen_le_threshold` -/

/-- **Headline 2: Threshold Upper Bound on Unseen Pairs.**
For any pair `p ∉ Seen D`, every channel `l` has `p.1 ∉ R l D` or `p.2 ∉ C l D`.
By nonnegativity, `A_l(p.1) * B_l(p.2) ≤ max (α_l * Bmax_l) (Amax_l * β_l)`.
Summing over `l` with weights `w_l ≥ 0` bounds `score p ≤ threshold`. -/
theorem unseen_le_threshold (hr : ρ ^ 2 = σ) (hr0 : ρ ≠ 0)
    (hw : ∀ l, 0 ≤ w l)
    (hA_pos : ∀ l i, 0 ≤ A l i) (hB_pos : ∀ l j, 0 ≤ B l j)
    (α β Amax Bmax : Fin r → K) (D : ℕ)
    (hAmax : ∀ l i, A l i ≤ Amax l)
    (hBmax : ∀ l j, B l j ≤ Bmax l)
    (hα : ∀ l i, i ∉ R A l D → A l i ≤ α l)
    (hβ : ∀ l j, j ∉ C B l D → B l j ≤ β l)
    (p : Fin n × Fin m) (hp : p ∉ Seen A B D) :
    score w A B σ ρ p ≤ threshold w α β Amax Bmax := by
  rw [score_as_cells_pair w A B σ ρ hr hr0 p]
  unfold threshold
  apply Finset.sum_le_sum
  intro l _
  have hnot_both : ¬ (p.1 ∈ R A l D ∧ p.2 ∈ C B l D) := by
    intro hboth
    apply hp
    rw [Seen, Finset.mem_biUnion]
    refine ⟨l, Finset.mem_univ l, ?_⟩
    rw [Finset.mem_product]
    exact hboth
  rw [not_and_or] at hnot_both
  apply mul_le_mul_of_nonneg_left _ (hw l)
  rcases hnot_both with hrow | hcol
  · have h1 : A l p.1 ≤ α l := hα l p.1 hrow
    have h2 : B l p.2 ≤ Bmax l := hBmax l p.2
    have hprod : A l p.1 * B l p.2 ≤ α l * Bmax l := by
      apply mul_le_mul h1 h2 (hB_pos l p.2)
      exact le_trans (hA_pos l p.1) h1
    exact le_trans hprod (le_max_left _ _)
  · have h1 : A l p.1 ≤ Amax l := hAmax l p.1
    have h2 : B l p.2 ≤ β l := hβ l p.2 hcol
    have hprod : A l p.1 * B l p.2 ≤ Amax l * β l := by
      apply mul_le_mul h1 h2 (hB_pos l p.2)
      exact le_trans (hA_pos l p.1) h1
    exact le_trans hprod (le_max_right _ _)

/-! ## 5. Headline 4: `seen_card_le` (Evaluation Count Bound) -/

/-- **Headline 4: Evaluation Count Bound.**
The evaluated candidate set `Seen D` has cardinality at most `r * D^2`.
Derivation: `Seen D` is the union of `r` products `R l D × C l D`.
By `topk_card_le`, each `R l D` and `C l D` has size `≤ D`, so each product has size `≤ D^2`.
The union has size at most `∑_l D^2 = r * D^2`. -/
theorem seen_card_le (D : ℕ) :
    (Seen A B D).card ≤ r * D ^ 2 := by
  have h1 : (Seen A B D).card ≤ ∑ l : Fin r, ((R A l D) ×ˢ (C B l D)).card := by
    unfold Seen
    exact Finset.card_biUnion_le
  have h2 : (∑ l : Fin r, ((R A l D) ×ˢ (C B l D)).card) ≤ ∑ l : Fin r, D ^ 2 := by
    apply Finset.sum_le_sum
    intro l _
    rw [Finset.card_product]
    have hR : (R A l D).card ≤ D := topk_card_le _ _ _
    have hC : (C B l D).card ≤ D := topk_card_le _ _ _
    nlinarith
  have h3 : (∑ l : Fin r, D ^ 2) = r * D ^ 2 := by
    simp
  exact le_trans h1 (le_trans h2 (le_of_eq h3))

/-! ## 6. Headline 3: `threshold_stop_exact` -/

/-- Strict ranking relation on grid pairs: higher score wins; ties broken lexicographically. -/
def precGrid (sc : Fin n × Fin m → K) (p1 p2 : Fin n × Fin m) : Prop :=
  sc p2 < sc p1 ∨ (sc p1 = sc p2 ∧ lexLT p1 p2)

instance (sc : Fin n × Fin m → K) : DecidableRel (precGrid sc) :=
  fun p1 p2 => inferInstanceAs (Decidable (sc p2 < sc p1 ∨ (sc p1 = sc p2 ∧ lexLT p1 p2)))

theorem precGrid_irrefl (sc : Fin n × Fin m → K) (p : Fin n × Fin m) : ¬ precGrid sc p p := by
  dsimp [precGrid]
  rintro (h1 | ⟨_, h2⟩)
  · exact lt_irrefl _ h1
  · exact lexLT_irrefl p h2

theorem precGrid_trans (sc : Fin n × Fin m → K) {p1 p2 p3 : Fin n × Fin m}
    (h12 : precGrid sc p1 p2) (h23 : precGrid sc p2 p3) : precGrid sc p1 p3 := by
  dsimp [precGrid] at *
  rcases h12 with hs1 | ⟨heqs1, hl1⟩
  · rcases h23 with hs2 | ⟨heqs2, _⟩
    · left; exact lt_trans hs2 hs1
    · left; rw [← heqs2]; exact hs1
  · rcases h23 with hs2 | ⟨heqs2, hl2⟩
    · left; rw [heqs1]; exact hs2
    · right; exact ⟨heqs1.trans heqs2, lexLT_trans hl1 hl2⟩

theorem precGrid_trichotomy (sc : Fin n × Fin m → K) (p1 p2 : Fin n × Fin m) :
    p1 = p2 ∨ precGrid sc p1 p2 ∨ precGrid sc p2 p1 := by
  dsimp [precGrid]
  rcases lt_trichotomy (sc p1) (sc p2) with hs1 | hseq | hs2
  · right; right; left; exact hs1
  · rcases lexLT_trichotomy p1 p2 with heq | hl1 | hl2
    · left; exact heq
    · right; left; right; exact ⟨hseq, hl1⟩
    · right; right; right; exact ⟨hseq.symm, hl2⟩
  · right; left; left; exact hs2

/-- Stopping criterion: a subset `S ⊆ seen` of size `≥ k` whose elements all score
strictly above the threshold `T`. -/
def StopCondition (T : K) (sc : Fin n × Fin m → K) (seen : Finset (Fin n × Fin m)) (k : ℕ) : Prop :=
  ∃ S : Finset (Fin n × Fin m), S ⊆ seen ∧ k ≤ S.card ∧ ∀ q ∈ S, T < sc q

/-- **Headline 3: Exactness of Threshold Stopping.**
If at scan depth `D`, at least `k` elements in `Seen D` score strictly above the threshold `T D`,
then the top-`k` of `Seen D` under lexicographic tie-breaking equals the global top-`k` of the
entire `n × m` grid.
Derivation: no unseen pair can enter the top-`k` because each element of `S` strictly precedes it,
giving at least `k` predecessors in the global grid. Then `ProductKeys.topk_exact_of_subset`
delivers exactness. -/
theorem threshold_stop_exact (hr : ρ ^ 2 = σ) (hr0 : ρ ≠ 0)
    (hw : ∀ l, 0 ≤ w l)
    (hA_pos : ∀ l i, 0 ≤ A l i) (hB_pos : ∀ l j, 0 ≤ B l j)
    (α β Amax Bmax : Fin r → K) (D k : ℕ)
    (hAmax : ∀ l i, A l i ≤ Amax l)
    (hBmax : ∀ l j, B l j ≤ Bmax l)
    (hα : ∀ l i, i ∉ R A l D → A l i ≤ α l)
    (hβ : ∀ l j, j ∉ C B l D → B l j ≤ β l)
    (S : Finset (Fin n × Fin m)) (hS_sub : S ⊆ Seen A B D) (hS_card : k ≤ S.card)
    (hS_strict : ∀ q ∈ S, threshold w α β Amax Bmax < score w A B σ ρ q) :
    topk (Seen A B D) (precGrid (score w A B σ ρ)) k =
    topk (Finset.univ : Finset (Fin n × Fin m)) (precGrid (score w A B σ ρ)) k := by
  set sc := score w A B σ ρ
  set rel := precGrid sc
  have hsub : topk (Finset.univ : Finset (Fin n × Fin m)) rel k ⊆ Seen A B D := by
    intro p hp
    rw [topk, Finset.mem_filter] at hp
    obtain ⟨_, hpred_card⟩ := hp
    by_contra hp_not_seen
    have h_le := unseen_le_threshold w A B σ ρ hr hr0 hw hA_pos hB_pos α β Amax Bmax D hAmax hBmax hα hβ p hp_not_seen
    have hS_pred : S ⊆ predecessors Finset.univ rel p := by
      intro q hq
      rw [predecessors, Finset.mem_filter]
      refine ⟨Finset.mem_univ q, ?_⟩
      dsimp [rel, precGrid]
      left
      have hq_gt := hS_strict q hq
      exact lt_of_le_of_lt h_le hq_gt
    have hcard_le := Finset.card_le_card hS_pred
    omega
  exact topk_exact_of_subset rel
    (precGrid_irrefl sc)
    (fun _ _ _ => precGrid_trans sc)
    (precGrid_trichotomy sc)
    Finset.univ (Seen A B D) (Finset.subset_univ _) k hsub

/-! ## 7. Headline 5: `rank1_candidate_exact` -/

/-- For `r = 1`, the evaluated candidate set `Seen A B D` is the product grid `R A 0 D × C B 0 D`. -/
theorem seen_rank1 (A : Fin 1 → Fin n → K) (B : Fin 1 → Fin m → K) (D : ℕ) :
    Seen A B D = (R A 0 D) ×ˢ (C B 0 D) := by
  unfold Seen
  ext ⟨i, j⟩
  simp

/-- **Headline 5: Rank-1 Candidate Grid Exactness.**
For `r = 1`, the evaluated candidate set `Seen A B k` is identically the product grid
`R A 0 k × C B 0 k`. By `Nous.Seq.ProductKeys.product_topk_exact`, searching the `k²`
candidate grid over the top-`k` marginal channels equals the global top-`k` over the entire
`n × m` grid. -/
theorem rank1_candidate_exact (A : Fin 1 → Fin n → K) (B : Fin 1 → Fin m → K) (k : ℕ) :
    topk (Seen A B k) (precGrid (pairScore (A 0) (B 0))) k =
    topk (Finset.univ : Finset (Fin n × Fin m)) (precGrid (pairScore (A 0) (B 0))) k := by
  rw [seen_rank1]
  unfold R C
  exact product_topk_exact (A 0) (B 0) k

/-! ## 8. Headline 6: Concrete Witnesses -/

namespace Witness

/-- Row channels from `tucker_rank2_breaks` (rank 2, 3×3). -/
def A_tucker (l : Fin 2) (i : Fin 3) : ℚ :=
  match l.val, i.val with
  | 0, 0 => 10
  | 0, 1 => 6
  | 0, 2 => 0
  | 1, 0 => 0
  | 1, 1 => 6
  | 1, 2 => 10
  | _, _ => 0

/-- Column channels from `tucker_rank2_breaks` (rank 2, 3×3). -/
def B_tucker (l : Fin 2) (j : Fin 3) : ℚ :=
  match l.val, j.val with
  | 0, 0 => 10
  | 0, 1 => 6
  | 0, 2 => 0
  | 1, 0 => 0
  | 1, 1 => 6
  | 1, 2 => 10
  | _, _ => 0

/-- Unit weights for rank-2 Tucker witness. -/
def w_tucker (l : Fin 2) : ℚ := 1

/-- Full grid score for the rank-2 Tucker witness. -/
def sc_tucker (p : Fin 3 × Fin 3) : ℚ :=
  A_tucker 0 p.1 * B_tucker 0 p.2 + A_tucker 1 p.1 * B_tucker 1 p.2

/-- Exact scores across the 3×3 grid. -/
theorem tucker_scores :
    sc_tucker (0, 0) = 100 ∧
    sc_tucker (1, 1) = 72 ∧
    sc_tucker (2, 2) = 100 ∧
    sc_tucker (0, 1) = 60 ∧
    sc_tucker (1, 0) = 60 ∧
    sc_tucker (0, 2) = 0 ∧
    sc_tucker (2, 0) = 0 ∧
    sc_tucker (1, 2) = 60 ∧
    sc_tucker (2, 1) = 60 := by
  refine ⟨by norm_num [sc_tucker, A_tucker, B_tucker],
          by norm_num [sc_tucker, A_tucker, B_tucker],
          by norm_num [sc_tucker, A_tucker, B_tucker],
          by norm_num [sc_tucker, A_tucker, B_tucker],
          by norm_num [sc_tucker, A_tucker, B_tucker],
          by norm_num [sc_tucker, A_tucker, B_tucker],
          by norm_num [sc_tucker, A_tucker, B_tucker],
          by norm_num [sc_tucker, A_tucker, B_tucker],
          by norm_num [sc_tucker, A_tucker, B_tucker]⟩

/-- Maximum row channel values for Tucker witness (10 for both channels). -/
def Amax_tucker : Fin 2 → ℚ := fun _ => 10

/-- Maximum column channel values for Tucker witness (10 for both channels). -/
def Bmax_tucker : Fin 2 → ℚ := fun _ => 10

/-- Row cutoffs at scan depth D = 1 (6 for both channels). -/
def α1_tucker : Fin 2 → ℚ := fun _ => 6

/-- Column cutoffs at scan depth D = 1 (6 for both channels). -/
def β1_tucker : Fin 2 → ℚ := fun _ => 6

/-- Row cutoffs at scan depth D = 2 (0 for both channels). -/
def α2_tucker : Fin 2 → ℚ := fun _ => 0

/-- Column cutoffs at scan depth D = 2 (0 for both channels). -/
def β2_tucker : Fin 2 → ℚ := fun _ => 0

/-- Unit weights are nonnegative. -/
theorem tucker_hw : ∀ l : Fin 2, 0 ≤ w_tucker l := by
  intro l; fin_cases l <;> (dsimp [w_tucker]; norm_num)

/-- Row channel values are nonnegative. -/
theorem tucker_hA_pos : ∀ (l : Fin 2) (i : Fin 3), 0 ≤ A_tucker l i := by
  decide

/-- Column channel values are nonnegative. -/
theorem tucker_hB_pos : ∀ (l : Fin 2) (j : Fin 3), 0 ≤ B_tucker l j := by
  decide

/-- Row values bounded by Amax. -/
theorem tucker_hAmax : ∀ (l : Fin 2) (i : Fin 3), A_tucker l i ≤ Amax_tucker l := by
  decide

/-- Column values bounded by Bmax. -/
theorem tucker_hBmax : ∀ (l : Fin 2) (j : Fin 3), B_tucker l j ≤ Bmax_tucker l := by
  decide

/-- Row cutoff side condition at depth D = 1 using the module's R. -/
theorem tucker_hα1 : ∀ (l : Fin 2) (i : Fin 3), i ∉ R A_tucker l 1 → A_tucker l i ≤ α1_tucker l := by
  decide

/-- Column cutoff side condition at depth D = 1 using the module's C. -/
theorem tucker_hβ1 : ∀ (l : Fin 2) (j : Fin 3), j ∉ C B_tucker l 1 → B_tucker l j ≤ β1_tucker l := by
  decide

/-- Row cutoff side condition at depth D = 2 using the module's R. -/
theorem tucker_hα2 : ∀ (l : Fin 2) (i : Fin 3), i ∉ R A_tucker l 2 → A_tucker l i ≤ α2_tucker l := by
  decide

/-- Column cutoff side condition at depth D = 2 using the module's C. -/
theorem tucker_hβ2 : ∀ (l : Fin 2) (j : Fin 3), j ∉ C B_tucker l 2 → B_tucker l j ≤ β2_tucker l := by
  decide

/-- **Witness 1(a): Threshold at Scan Depth D = 1 is 120.** -/
theorem tucker_threshold_d1 :
    threshold w_tucker α1_tucker β1_tucker Amax_tucker Bmax_tucker = 120 := by
  unfold threshold
  rw [Fin.sum_univ_two]
  dsimp [w_tucker, α1_tucker, β1_tucker, Amax_tucker, Bmax_tucker]
  norm_num

/-- **Witness 1(b): Threshold at Scan Depth D = 2 is 0.** -/
theorem tucker_threshold_d2 :
    threshold w_tucker α2_tucker β2_tucker Amax_tucker Bmax_tucker = 0 := by
  unfold threshold
  rw [Fin.sum_univ_two]
  dsimp [w_tucker, α2_tucker, β2_tucker, Amax_tucker, Bmax_tucker]
  norm_num

/-- Memory score coincides with sc_tucker for any root ρ of σ. -/
theorem tucker_score_eq_sc (σ ρ : ℚ) (hr : ρ ^ 2 = σ) (hr0 : ρ ≠ 0) (p : Fin 3 × Fin 3) :
    score w_tucker A_tucker B_tucker σ ρ p = sc_tucker p := by
  rcases p with ⟨i, j⟩
  rw [score_as_cells w_tucker A_tucker B_tucker σ ρ hr hr0 i j]
  rw [Fin.sum_univ_two]
  dsimp [w_tucker, sc_tucker]
  ring

/-- **Witness 2: Stop Condition Fails at D = 1.**
At scan depth `D = 1`, every candidate in `Seen 1 = {(0,0), (2,2)}` has score `100 ≤ threshold = 120`,
so the threshold stop condition cannot fire at D = 1. -/
theorem tucker_stop_d1_fails (σ ρ : ℚ) (hr : ρ ^ 2 = σ) (hr0 : ρ ≠ 0) :
    ∀ q ∈ Seen A_tucker B_tucker 1,
      score w_tucker A_tucker B_tucker σ ρ q ≤
      threshold w_tucker α1_tucker β1_tucker Amax_tucker Bmax_tucker := by
  intro q _
  rw [tucker_score_eq_sc σ ρ hr hr0]
  rw [tucker_threshold_d1]
  rcases q with ⟨i, j⟩
  fin_cases i <;> fin_cases j <;> (dsimp [sc_tucker, A_tucker, B_tucker]; norm_num)

/-- Candidate (0, 0) strictly exceeds the D = 2 threshold T(2) = 0. -/
theorem tucker_stop_d2_holds (σ ρ : ℚ) (hr : ρ ^ 2 = σ) (hr0 : ρ ≠ 0) :
    threshold w_tucker α2_tucker β2_tucker Amax_tucker Bmax_tucker <
      score w_tucker A_tucker B_tucker σ ρ (0, 0) := by
  rw [tucker_threshold_d2, tucker_score_eq_sc σ ρ hr hr0]
  dsimp [sc_tucker, A_tucker, B_tucker]
  norm_num

/-- **Witness 3: Joint Exact Witness at D = 2.**
At scan depth `D = 2`, the singleton set `S = {(0, 0)} ⊆ Seen 2` has cardinality `1 ≥ 1` and
scores strictly above the threshold `T(2) = 0`. Applying `threshold_stop_exact` with the D = 2
side conditions proves that the top-1 of `Seen 2` equals the global top-1 of the entire 3×3 grid,
and `(0, 0)` is indeed in the global top-1. -/
theorem tucker_exact_d2 (σ ρ : ℚ) (hr : ρ ^ 2 = σ) (hr0 : ρ ≠ 0) :
    topk (Seen A_tucker B_tucker 2) (precGrid (score w_tucker A_tucker B_tucker σ ρ)) 1 =
    topk (Finset.univ : Finset (Fin 3 × Fin 3)) (precGrid (score w_tucker A_tucker B_tucker σ ρ)) 1 ∧
    (0, 0) ∈ topk (Finset.univ : Finset (Fin 3 × Fin 3)) (precGrid (score w_tucker A_tucker B_tucker σ ρ)) 1 := by
  have hS_sub : ({(0, 0)} : Finset (Fin 3 × Fin 3)) ⊆ Seen A_tucker B_tucker 2 := by
    decide
  have hS_card : 1 ≤ ({(0, 0)} : Finset (Fin 3 × Fin 3)).card := by
    decide
  have hS_strict : ∀ q ∈ ({(0, 0)} : Finset (Fin 3 × Fin 3)),
      threshold w_tucker α2_tucker β2_tucker Amax_tucker Bmax_tucker <
        score w_tucker A_tucker B_tucker σ ρ q := by
    intro q hq
    rw [Finset.mem_singleton] at hq
    subst hq
    exact tucker_stop_d2_holds σ ρ hr hr0
  have hexact := threshold_stop_exact w_tucker A_tucker B_tucker σ ρ hr hr0
    tucker_hw tucker_hA_pos tucker_hB_pos
    α2_tucker β2_tucker Amax_tucker Bmax_tucker 2 1
    tucker_hAmax tucker_hBmax tucker_hα2 tucker_hβ2
    {(0, 0)} hS_sub hS_card hS_strict
  refine ⟨hexact, ?_⟩
  rw [topk, Finset.mem_filter]
  refine ⟨Finset.mem_univ _, ?_⟩
  have hnot_pred : ∀ p : Fin 3 × Fin 3, p ∉ predecessors Finset.univ (precGrid (score w_tucker A_tucker B_tucker σ ρ)) (0, 0) := by
    intro p hp
    rw [predecessors, Finset.mem_filter] at hp
    obtain ⟨_, hprec⟩ := hp
    dsimp [precGrid] at hprec
    rw [tucker_score_eq_sc σ ρ hr hr0, tucker_score_eq_sc σ ρ hr hr0] at hprec
    rcases p with ⟨i, j⟩
    fin_cases i <;> fin_cases j
    · rcases hprec with hlt | ⟨_, hlex⟩
      · dsimp [sc_tucker, A_tucker, B_tucker] at hlt; norm_num at hlt
      · dsimp [lexLT] at hlex; revert hlex; decide
    · rcases hprec with hlt | ⟨heq, _⟩
      · dsimp [sc_tucker, A_tucker, B_tucker] at hlt; norm_num at hlt
      · dsimp [sc_tucker, A_tucker, B_tucker] at heq; norm_num at heq
    · rcases hprec with hlt | ⟨heq, _⟩
      · dsimp [sc_tucker, A_tucker, B_tucker] at hlt; norm_num at hlt
      · dsimp [sc_tucker, A_tucker, B_tucker] at heq; norm_num at heq
    · rcases hprec with hlt | ⟨heq, _⟩
      · dsimp [sc_tucker, A_tucker, B_tucker] at hlt; norm_num at hlt
      · dsimp [sc_tucker, A_tucker, B_tucker] at heq; norm_num at heq
    · rcases hprec with hlt | ⟨heq, _⟩
      · dsimp [sc_tucker, A_tucker, B_tucker] at hlt; norm_num at hlt
      · dsimp [sc_tucker, A_tucker, B_tucker] at heq; norm_num at heq
    · rcases hprec with hlt | ⟨heq, _⟩
      · dsimp [sc_tucker, A_tucker, B_tucker] at hlt; norm_num at hlt
      · dsimp [sc_tucker, A_tucker, B_tucker] at heq; norm_num at heq
    · rcases hprec with hlt | ⟨heq, _⟩
      · dsimp [sc_tucker, A_tucker, B_tucker] at hlt; norm_num at hlt
      · dsimp [sc_tucker, A_tucker, B_tucker] at heq; norm_num at heq
    · rcases hprec with hlt | ⟨heq, _⟩
      · dsimp [sc_tucker, A_tucker, B_tucker] at hlt; norm_num at hlt
      · dsimp [sc_tucker, A_tucker, B_tucker] at heq; norm_num at heq
    · rcases hprec with hlt | ⟨_, hlex⟩
      · dsimp [sc_tucker, A_tucker, B_tucker] at hlt; norm_num at hlt
      · dsimp [lexLT] at hlex; revert hlex; decide
  have hpred_empty : predecessors Finset.univ (precGrid (score w_tucker A_tucker B_tucker σ ρ)) (0, 0) = ∅ := by
    ext x
    simp [hnot_pred x]
  rw [hpred_empty, Finset.card_empty]
  exact Nat.zero_lt_one

/-- **Witness: Parameter σ Enters Only Through slotCell.**
At `σ = 4` with root `ρ = 2`, the cell norm form factors into the channel product
via `slot_normForm`, recovering identically the Tucker product score `sc_tucker (i, j)`.
Thus the quadratic parameter `σ` is purely internal to the cell representation. -/
theorem sigma4_enters_only_through_slotCell (i : Fin 3) (j : Fin 3) :
    let σ : ℚ := 4
    let ρ : ℚ := 2
    (∑ l : Fin 2, w_tucker l * ringNormForm σ (slotCell ρ (A_tucker l i) (B_tucker l j))) =
    sc_tucker (i, j) := by
  intro σ ρ
  have hr : (2 : ℚ) ^ 2 = 4 := by norm_num
  have hr0 : (2 : ℚ) ≠ 0 := by norm_num
  rw [Fin.sum_univ_two]
  dsimp only [w_tucker]
  rw [slot_normForm 4 2 hr hr0]
  rw [slot_normForm 4 2 hr hr0]
  unfold sc_tucker
  ring

end Witness

end Nous.Seq.ThresholdCells
