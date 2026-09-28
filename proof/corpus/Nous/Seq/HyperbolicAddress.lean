import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Apeiron.Trichotomy.Idempotents
import Apeiron.Trichotomy.IdempotentsAnyRing
import Apeiron.Trichotomy.NullBasis
import Apeiron.Trichotomy.SplitComplex
import Nous.Model.Decomposition
import Nous.Model.TopKRouter
import Nous.Seq.ProductKeys
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.Order.Ring.Defs
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Tactic

/-!
# Hyperbolic Cell Addressing and Product-Key Two-Phase Exact Search

This module derives product-key addressing directly from the canonical cell
`R[x]/(x² - σ)` and its trichotomy, establishing that two-part separable search
is a structural consequence of the hyperbolic algebra (`σ > 0`), not an ad-hoc
additive approximation.

## Mathematical Setting and Trichotomy Boundary

1. **Hyperbolic face (`σ > 0`):**
   The quadratic polynomial `x² - σ` splits into linear factors with roots `±r` (`r² = σ`).
   By `Apeiron.Trichotomy.ChannelsAnyRing`, the two coordinate channels
   `ringUpper r z = z.1 + r * z.2` and `ringLower r z = z.1 - r * z.2`
   form an algebra isomorphism `RingCell K ≃ K × K` (`hyperbolic_decomposition`).
   Multiplication is componentwise: `ringUpper r (z · w) = ringUpper r z * ringUpper r w`
   and `ringLower r (z · w) = ringLower r z * ringLower r w`.
   The quadratic norm form factors into the product of the two channels:
   `ringNormForm σ z = ringUpper r z * ringLower r z` (`ringNormForm_eq_upper_mul_lower`).
   Every slot `(i, j)` in a 2-D memory table is the product of a row cell `slotCell a_i 1`
   and a column cell `slotCell 1 b_j`: `slot_mul` proves this equals `slotCell a_i b_j`.
   For positive channels, the norm form is strictly monotone in each channel separately
   (`normForm_mono_channels`), which guarantees exact two-phase top-`k` pruning:
   `two_phase_exact_hyperbolic`.

2. **Parabolic boundary (`σ = 0`):**
   The two channels collapse into one (`ringUpper 0 z = ringLower 0 z = z.1`), and the norm
   form degenerates to a single square (`ringNormForm 0 z = (ringUpper 0 z)²`).
   Two-part separable addressing collapses to single-coordinate indexing.

3. **Elliptic face (`σ < 0`):**
   The norm form is strictly positive definite and does not factor into real linear forms
   (`elliptic_no_channels`). Two-phase separable search has no real channel coordinates;
   instead, the elliptic face addresses by rotation on the torus (see `Nous/Model/RopeOrbit.lean`
   `rope_orbit` and Ramallo's boundary action).

4. **UltraMem Boundary:**
   UltraMem (Huang et al., ICLR 2025, 2411.12364) hits an exact-search ceiling with rank-`r`
   Tucker cores `S_rowᵀ C S_col`. We prove that a rank-1 core is isomorphic to a single
   hyperbolic cell (`rank1_core_is_cell`), making two-phase search exact. For rank-2 cores,
   we prove non-vacuity via an explicit 3×3 instance over `ℚ` (`tucker_rank2_breaks`) where
   the true top-1 grid element is strictly outside the product of the marginal top-1 sets.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedVariables false
set_option linter.unusedSectionVars false

noncomputable section

namespace Nous.Seq.HyperbolicAddress

open Apeiron.Trichotomy
open Apeiron.Trichotomy.AnyRing
open Nous.Seq.ProductKeys

/-! ## 1. Headline 1: `slotCell`, `slot_channels`, and `slot_normForm` -/

section SlotCell

variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]
variable (σ r : K)

/-- A cell whose two channel coordinates under `ringUpper r` and `ringLower r` are `u` and `v`.
Reconstructed via `ringOfHalves` with root inverse `r⁻¹` and half `2⁻¹`. -/
def slotCell (u v : K) : RingCell K :=
  ringOfHalves u v r⁻¹ 2⁻¹

/-- The channel coordinates of `slotCell r u v` are exactly `(u, v)`. -/
theorem slot_channels (hr0 : r ≠ 0) (u v : K) :
    ringUpper r (slotCell r u v) = u ∧ ringLower r (slotCell r u v) = v := by
  have hrs : r * r⁻¹ = 1 := mul_inv_cancel₀ hr0
  have hh : 2 * (2⁻¹ : K) = 1 := by norm_num
  exact ⟨ringUpper_ringOfHalves hrs hh, ringLower_ringOfHalves hrs hh⟩

/-- The norm form of `slotCell r u v` is the product of its channels `u * v`.
Derived directly from canonical `ringNormForm_eq_upper_mul_lower`. -/
theorem slot_normForm (hr : r ^ 2 = σ) (hr0 : r ≠ 0) (u v : K) :
    ringNormForm σ (slotCell r u v) = u * v := by
  rw [ringNormForm_eq_upper_mul_lower hr]
  have ⟨hu, hv⟩ := slot_channels r hr0 u v
  rw [hu, hv]

/-! ## 2. Headline 2: `slot_mul` -/

/-- Every slot `(a, b)` in a 2-D table is the product of a row cell and a column cell:
`ringMul σ (slotCell a 1) (slotCell 1 b) = slotCell a b`.
Channels multiply componentwise under canonical `ringUpper_mul` and `ringLower_mul`. -/
theorem slot_mul (hr : r ^ 2 = σ) (hr0 : r ≠ 0) (a b : K) :
    ringMul σ (slotCell r a 1) (slotCell r 1 b) = slotCell r a b := by
  have hrs : r * r⁻¹ = 1 := mul_inv_cancel₀ hr0
  have hh : 2 * (2⁻¹ : K) = 1 := by norm_num
  set z := ringMul σ (slotCell r a 1) (slotCell r 1 b)
  have hu : ringUpper r z = a := by
    dsimp [z, slotCell]
    rw [ringUpper_mul hr]
    have hrs' : r * r⁻¹ = 1 := mul_inv_cancel₀ hr0
    have hh' : 2 * (2⁻¹ : K) = 1 := by norm_num
    rw [ringUpper_ringOfHalves hrs' hh', ringUpper_ringOfHalves hrs' hh', _root_.mul_one]
  have hv : ringLower r z = b := by
    dsimp [z, slotCell]
    rw [ringLower_mul hr]
    have hrs' : r * r⁻¹ = 1 := mul_inv_cancel₀ hr0
    have hh' : 2 * (2⁻¹ : K) = 1 := by norm_num
    rw [ringLower_ringOfHalves hrs' hh', ringLower_ringOfHalves hrs' hh', _root_.one_mul]
  have hrec := ringOfHalves_upper_lower hrs hh z
  rw [hu, hv] at hrec
  exact hrec.symm

/-! ## 3. Headline 3: `normForm_mono_channels` -/

/-- For positive channel `v > 0`, the norm form is strictly monotone in the first channel. -/
theorem normForm_mono_channels_left (hr : r ^ 2 = σ) (hr0 : r ≠ 0)
    {u₁ u₂ v : K} (hv : 0 < v) (hu : u₁ < u₂) :
    ringNormForm σ (slotCell r u₁ v) < ringNormForm σ (slotCell r u₂ v) := by
  rw [slot_normForm σ r hr hr0 u₁ v, slot_normForm σ r hr hr0 u₂ v]
  exact mul_lt_mul_of_pos_right hu hv

/-- For positive channel `u > 0`, the norm form is strictly monotone in the second channel. -/
theorem normForm_mono_channels_right (hr : r ^ 2 = σ) (hr0 : r ≠ 0)
    {u v₁ v₂ : K} (hu : 0 < u) (hv : v₁ < v₂) :
    ringNormForm σ (slotCell r u v₁) < ringNormForm σ (slotCell r u v₂) := by
  rw [slot_normForm σ r hr hr0 u v₁, slot_normForm σ r hr hr0 u v₂]
  exact mul_lt_mul_of_pos_left hv hu

/-- Strict monotonicity of the quadratic norm form in both channels separately. -/
theorem normForm_mono_channels (hr : r ^ 2 = σ) (hr0 : r ≠ 0) :
    (∀ {u₁ u₂ v : K}, 0 < v → u₁ < u₂ → ringNormForm σ (slotCell r u₁ v) < ringNormForm σ (slotCell r u₂ v)) ∧
    (∀ {u v₁ v₂ : K}, 0 < u → v₁ < v₂ → ringNormForm σ (slotCell r u v₁) < ringNormForm σ (slotCell r u v₂)) :=
  ⟨fun hv hu => normForm_mono_channels_left σ r hr hr0 hv hu,
   fun hu hv => normForm_mono_channels_right σ r hr hr0 hu hv⟩

end SlotCell

/-! ## 4. Headline 4: `two_phase_exact_hyperbolic` and `log_additive` -/

section TwoPhase

variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]
variable {n m : ℕ} (a : Fin n → K) (b : Fin m → K)

/-- The score of a product slot `(i, j)` in terms of cell product norm form. -/
def hyperbolicSlotScore (σ r : K) (hr : r ^ 2 = σ) (hr0 : r ≠ 0) (i : Fin n) (j : Fin m) : K :=
  ringNormForm σ (ringMul σ (slotCell r (a i) 1) (slotCell r 1 (b j)))

/-- The hyperbolic slot score simplifies to the product `a i * b j`. -/
theorem hyperbolicSlotScore_eq (σ r : K) (hr : r ^ 2 = σ) (hr0 : r ≠ 0) (i : Fin n) (j : Fin m) :
    hyperbolicSlotScore a b σ r hr hr0 i j = a i * b j := by
  unfold hyperbolicSlotScore
  rw [slot_mul σ r hr hr0, slot_normForm σ r hr hr0]

/-- Physical linear address of slot `(i, j)`. -/
def gridAddr (p : Fin n × Fin m) : ℕ := p.1.val * m + p.2.val

/-- Ranking relation on grid slots: higher norm form score wins; ties broken by lower address `i · m + j`. -/
def gridBeats (score : Fin n × Fin m → K) (p q : Fin n × Fin m) : Prop :=
  score q < score p ∨ (score q = score p ∧ gridAddr p < gridAddr q)

theorem gridBeats_asymm (score : Fin n × Fin m → K) (p q : Fin n × Fin m) :
    gridBeats score p q → ¬ gridBeats score q p := by
  unfold gridBeats
  rintro (hlt | ⟨heq, haddr⟩) (hlt' | ⟨heq', haddr'⟩)
  · exact lt_asymm hlt hlt'
  · exact ne_of_gt hlt heq'
  · exact ne_of_lt hlt' heq.symm
  · omega

/-- Top-`k` set of row scores with lower index tie-breaking. -/
def IsRowTopK (k : ℕ) (Sa : Finset (Fin n)) : Prop :=
  Sa.card = min k n ∧ ∀ i ∈ Sa, ∀ i' ∉ Sa, a i' < a i ∨ (a i' = a i ∧ i < i')

/-- Top-`k` set of column scores with lower index tie-breaking. -/
def IsColTopK (k : ℕ) (Sb : Finset (Fin m)) : Prop :=
  Sb.card = min k m ∧ ∀ j ∈ Sb, ∀ j' ∉ Sb, b j' < b j ∨ (b j' = b j ∧ j < j')

/-- Top-`k` set of grid slots under `gridBeats`. -/
def IsGridTopK (score : Fin n × Fin m → K) (k : ℕ) (S : Finset (Fin n × Fin m)) : Prop :=
  S.card = min k (n * m) ∧ ∀ p ∈ S, ∀ q ∉ S, gridBeats score p q

theorem gridAddr_lt_of_fst_lt {p q : Fin n × Fin m} (h : p.1 < q.1) :
    gridAddr p < gridAddr q := by
  unfold gridAddr
  have hp : p.1.val + 1 ≤ q.1.val := h
  have hp2 : p.2.val < m := p.2.isLt
  nlinarith

theorem gridAddr_lt_of_snd_lt {p q : Fin n × Fin m} (heq : p.1 = q.1) (h : p.2 < q.2) :
    gridAddr p < gridAddr q := by
  unfold gridAddr
  have h1 : p.1.val = q.1.val := congrArg Fin.val heq
  rw [h1]
  exact Nat.add_lt_add_left h _

/-- **Headline 4: Two-Phase Exact Hyperbolic Search.**
For positive row and column channel vectors, ranking the `n · m` slots by the hyperbolic cell
norm form `ringNormForm σ (ringMul σ (slotCell (a i) 1) (slotCell 1 (b j)))` with ties broken
by lowest address `i · m + j`, every top-`k` element belongs to the candidate grid
`topk(a) × topk(b)`. -/
theorem two_phase_exact_hyperbolic (σ r : K) (hr : r ^ 2 = σ) (hr0 : r ≠ 0)
    (ha : ∀ i, 0 < a i) (hb : ∀ j, 0 < b j)
    {k : ℕ} {Sa : Finset (Fin n)} {Sb : Finset (Fin m)} {S : Finset (Fin n × Fin m)}
    (hSa : IsRowTopK a k Sa) (hSb : IsColTopK b k Sb)
    (hS : IsGridTopK (fun p => hyperbolicSlotScore a b σ r hr hr0 p.1 p.2) k S) :
    ∀ p ∈ S, p.1 ∈ Sa ∧ p.2 ∈ Sb := by
  set score := fun p : Fin n × Fin m => hyperbolicSlotScore a b σ r hr hr0 p.1 p.2
  have hscore_eq : ∀ i j, score (i, j) = a i * b j := fun i j =>
    hyperbolicSlotScore_eq a b σ r hr hr0 i j
  intro p hp
  constructor
  · by_contra hi
    have hcard_a : Sa.card = k := by
      have : Sa.card ≤ n := by
        have := Sa.card_le_univ
        rwa [Fintype.card_fin] at this
      have : Sa.card < n := by
        have hne : Sa ≠ Finset.univ := fun h => hi (h ▸ Finset.mem_univ p.1)
        have hle : Sa.card ≤ Fintype.card (Fin n) := Sa.card_le_univ
        rw [Fintype.card_fin] at hle
        exact lt_of_le_of_ne hle (fun hcard => hne (Finset.eq_univ_of_card _ (by rwa [Fintype.card_fin])))
      have := hSa.1
      omega
    have hbeats : ∀ i' ∈ Sa, gridBeats score (i', p.2) p := by
      intro i' hi'
      have hspec := hSa.2 i' hi' p.1 hi
      unfold gridBeats
      rw [hscore_eq, hscore_eq]
      rcases hspec with hlt | ⟨heq, hlt⟩
      · left
        exact mul_lt_mul_of_pos_right hlt (hb p.2)
      · right
        constructor
        · rw [heq]
        · exact gridAddr_lt_of_fst_lt hlt
    set T := Sa.image (fun i' => (i', p.2)) with hT
    have hTcard : T.card = k := by
      rw [hT, Finset.card_image_of_injective]
      · exact hcard_a
      · intro x y hxy
        exact Prod.ext_iff.mp hxy |>.1
    have hpT : p ∉ T := by
      rw [hT, Finset.mem_image]
      rintro ⟨i', hi', hpair⟩
      have : p.1 = i' := (Prod.ext_iff.mp hpair).1.symm
      exact hi (this ▸ hi')
    have hnot_sub : ¬ T ⊆ S := by
      intro hsub
      have hcard_union : (insert p T).card = k + 1 := by
        rw [Finset.card_insert_of_notMem hpT, hTcard]
      have hsub_all : insert p T ⊆ S := by
        rw [Finset.insert_subset_iff]
        exact ⟨hp, hsub⟩
      have hle := Finset.card_le_card hsub_all
      rw [hcard_union, hS.1] at hle
      have : min k (n * m) ≤ k := min_le_left k (n * m)
      omega
    have ⟨q, hqT, hqS⟩ : ∃ q ∈ T, q ∉ S := by
      by_contra! h
      exact hnot_sub h
    rw [hT, Finset.mem_image] at hqT
    obtain ⟨i', hi', rfl⟩ := hqT
    have hq_beats_p := hbeats i' hi'
    have hp_beats_q := hS.2 p hp (i', p.2) hqS
    exact gridBeats_asymm score (i', p.2) p hq_beats_p hp_beats_q
  · by_contra hj
    have hcard_b : Sb.card = k := by
      have : Sb.card < m := by
        have hne : Sb ≠ Finset.univ := fun h => hj (h ▸ Finset.mem_univ p.2)
        have hle : Sb.card ≤ Fintype.card (Fin m) := Sb.card_le_univ
        rw [Fintype.card_fin] at hle
        exact lt_of_le_of_ne hle (fun hcard => hne (Finset.eq_univ_of_card _ (by rwa [Fintype.card_fin])))
      have := hSb.1
      omega
    have hbeats : ∀ j' ∈ Sb, gridBeats score (p.1, j') p := by
      intro j' hj'
      have hspec := hSb.2 j' hj' p.2 hj
      unfold gridBeats
      rw [hscore_eq, hscore_eq]
      rcases hspec with hlt | ⟨heq, hlt⟩
      · left
        exact mul_lt_mul_of_pos_left hlt (ha p.1)
      · right
        constructor
        · rw [heq]
        · exact gridAddr_lt_of_snd_lt rfl hlt
    set T := Sb.image (fun j' => (p.1, j')) with hT
    have hTcard : T.card = k := by
      rw [hT, Finset.card_image_of_injective]
      · exact hcard_b
      · intro x y hxy
        exact Prod.ext_iff.mp hxy |>.2
    have hpT : p ∉ T := by
      rw [hT, Finset.mem_image]
      rintro ⟨j', hj', hpair⟩
      have : p.2 = j' := (Prod.ext_iff.mp hpair).2.symm
      exact hj (this ▸ hj')
    have hnot_sub : ¬ T ⊆ S := by
      intro hsub
      have hcard_union : (insert p T).card = k + 1 := by
        rw [Finset.card_insert_of_notMem hpT, hTcard]
      have hsub_all : insert p T ⊆ S := by
        rw [Finset.insert_subset_iff]
        exact ⟨hp, hsub⟩
      have hle := Finset.card_le_card hsub_all
      rw [hcard_union, hS.1] at hle
      have : min k (n * m) ≤ k := min_le_left k (n * m)
      omega
    have ⟨q, hqT, hqS⟩ : ∃ q ∈ T, q ∉ S := by
      by_contra! h
      exact hnot_sub h
    rw [hT, Finset.mem_image] at hqT
    obtain ⟨j', hj', rfl⟩ := hqT
    have hq_beats_p := hbeats j' hj'
    have hp_beats_q := hS.2 p hp (p.1, j') hqS
    exact gridBeats_asymm score (p.1, j') p hq_beats_p hp_beats_q

end TwoPhase

/-! ## 5. Log-Additivity and Bridge to `ProductKeys.lean` -/

section LogAdditive

variable {n m : ℕ} (s : Fin n → ℝ) (t : Fin m → ℝ)
variable (σ r : ℝ) (hr : r ^ 2 = σ) (hr0 : r ≠ 0)

/-- **Corollary `log_additive`:**
With exponential channel weights `a i = e^{s_i}` and `b j = e^{t_j}` over `ℝ`,
the logarithm of the hyperbolic cell norm form equals the additive product-key score
`s_i + t_j` of Lample et al. (2019). -/
theorem log_additive (hr : r ^ 2 = σ) (hr0 : r ≠ 0) (i : Fin n) (j : Fin m) :
    Real.log (ringNormForm σ (ringMul σ (slotCell r (Real.exp (s i)) 1) (slotCell r 1 (Real.exp (t j))))) =
      s i + t j := by
  rw [slot_mul σ r hr hr0]
  rw [slot_normForm σ r hr hr0]
  rw [← Real.exp_add]
  exact Real.log_exp (s i + t j)

/-- **Bridge to `Nous.Seq.ProductKeys`:**
The ranking relation on product keys induced by `precPair s t` is identical to the ranking
induced by the hyperbolic norm form of the cell product. -/
theorem hyperbolic_precPair_eq (hr : r ^ 2 = σ) (hr0 : r ≠ 0) (p q : Fin n × Fin m) :
    precPair s t p q ↔
      (ringNormForm σ (ringMul σ (slotCell r (Real.exp (s q.1)) 1) (slotCell r 1 (Real.exp (t q.2)))) <
       ringNormForm σ (ringMul σ (slotCell r (Real.exp (s p.1)) 1) (slotCell r 1 (Real.exp (t p.2))))) ∨
      (ringNormForm σ (ringMul σ (slotCell r (Real.exp (s p.1)) 1) (slotCell r 1 (Real.exp (t p.2)))) =
       ringNormForm σ (ringMul σ (slotCell r (Real.exp (s q.1)) 1) (slotCell r 1 (Real.exp (t q.2)))) ∧
       lexLT p q) := by
  rw [slot_mul σ r hr hr0, slot_mul σ r hr hr0]
  rw [slot_normForm σ r hr hr0, slot_normForm σ r hr hr0]
  rw [← Real.exp_add, ← Real.exp_add]
  rw [Real.exp_lt_exp, Real.exp_eq_exp]
  dsimp [precPair, pairScore]
  rfl

/-- **Bridge Theorem:**
Hyperbolic cell product-key search coincides with `Nous.Seq.ProductKeys.product_topk_exact`. -/
theorem hyperbolic_product_topk_bridge (k : ℕ) :
    topk ((topk (Finset.univ : Finset (Fin n)) (prec s) k) ×ˢ
          (topk (Finset.univ : Finset (Fin m)) (prec t) k))
      (precPair s t) k =
    topk (Finset.univ : Finset (Fin n × Fin m)) (precPair s t) k :=
  product_topk_exact s t k

end LogAdditive

/-! ## 6. Headline 5: The Trichotomy Boundary over ℝ -/

section TrichotomyBoundary

/-- A real quadratic form factors as a product of two real linear forms with independent
coefficient vectors. -/
def FactorsIntoIndependentLinearForms (σ : ℝ) : Prop :=
  ∃ (c11 c12 c21 c22 : ℝ),
    (c11 * c22 - c12 * c21 ≠ 0) ∧
    (∀ z : Cell, ringNormForm σ z = (c11 * z.1 + c12 * z.2) * (c21 * z.1 + c22 * z.2))

/-- **`split_iff_hyperbolic`:**
The norm form `a² - σ b²` factors as a product of two real linear forms with linearly
independent coefficient vectors iff `0 < σ`. -/
theorem split_iff_hyperbolic (σ : ℝ) :
    FactorsIntoIndependentLinearForms σ ↔ 0 < σ := by
  constructor
  · rintro ⟨c11, c12, c21, c22, hdet, hfac⟩
    by_contra! hσle
    rcases hσle.lt_or_eq with hσneg | rfl
    · have hz : ringNormForm σ (-c12, c11) = 0 := by
        rw [hfac]
        have : c11 * (-c12) + c12 * c11 = 0 := by ring
        rw [this, zero_mul]
      simp only [ringNormForm] at hz
      have h12 : c12 = 0 := by
        have hsq : (-c12)^2 ≤ 0 := by
          have : (-c12)^2 = - (-σ * c11^2) := by linarith [hz]
          have : 0 ≤ -σ * c11^2 := by
            have : 0 < -σ := by linarith
            positivity
          linarith
        have : 0 ≤ (-c12)^2 := sq_nonneg _
        have : (-c12)^2 = 0 := by linarith
        linear_combination (neg_eq_zero.mp (sq_eq_zero_iff.mp this))
      have h11 : c11 = 0 := by
        subst h12
        simp only [neg_zero, sq, mul_zero] at hz
        have : -σ * c11^2 = 0 := by linarith [hz]
        have hneg : -σ ≠ 0 := by linarith
        have hc11sq : c11^2 = 0 := by
          cases mul_eq_zero.mp this with
          | inl h => exact False.elim (hneg h)
          | inr h => exact h
        exact sq_eq_zero_iff.mp hc11sq
      subst h11 h12
      simp only [zero_mul, sub_self, ne_eq, not_true_eq_false] at hdet
    · have hz01 : ringNormForm (0 : ℝ) (0, 1) = 0 := by simp [ringNormForm]
      rw [hfac] at hz01
      simp only [mul_zero, zero_add, mul_one] at hz01
      rcases mul_eq_zero.mp hz01 with hc12 | hc22
      · subst hc12
        have hz11 : ringNormForm (0 : ℝ) (1, 1) = 1 := by simp [ringNormForm]
        have hz1m1 : ringNormForm (0 : ℝ) (1, -1) = 1 := by simp [ringNormForm]
        rw [hfac (1, 1)] at hz11
        rw [hfac (1, -1)] at hz1m1
        simp only [mul_one, zero_mul, add_zero] at hz11 hz1m1
        have hdiff : 2 * (c11 * c22) = 0 := by linear_combination hz11 - hz1m1
        have : c11 * c22 = 0 := by linarith [hdiff]
        simp only [zero_mul, sub_zero] at hdet
        exact hdet this
      · subst hc22
        have hz11 : ringNormForm (0 : ℝ) (1, 1) = 1 := by simp [ringNormForm]
        have hz1m1 : ringNormForm (0 : ℝ) (1, -1) = 1 := by simp [ringNormForm]
        rw [hfac (1, 1)] at hz11
        rw [hfac (1, -1)] at hz1m1
        simp only [mul_one, add_zero] at hz11 hz1m1
        have hdiff : 2 * (c12 * c21) = 0 := by linear_combination hz11 - hz1m1
        have : c12 * c21 = 0 := by linarith [hdiff]
        simp only [mul_zero, zero_sub] at hdet
        exact hdet (neg_eq_zero.mpr this)
  · intro hσ
    refine ⟨1, Real.sqrt σ, 1, -Real.sqrt σ, ?_, ?_⟩
    · have hrpos : 0 < Real.sqrt σ := Real.sqrt_pos.mpr hσ
      have : 1 * -Real.sqrt σ - Real.sqrt σ * 1 = -2 * Real.sqrt σ := by ring
      rw [this]
      linarith
    · intro z
      simp only [ringNormForm]
      have hsq : Real.sqrt σ ^ 2 = σ := Real.sq_sqrt hσ.le
      nlinarith [hsq]

/-- Link between linear factorization and the existence of nontrivial idempotents
from canonical `exists_nontrivial_idempotent_iff_disc_pos`. -/
theorem split_iff_nontrivial_idempotent (σ : ℝ) :
    FactorsIntoIndependentLinearForms σ ↔
      (∃ z : Cell, Idempotent σ z ∧ z ≠ 0 ∧ z ≠ one) := by
  rw [split_iff_hyperbolic, exists_nontrivial_idempotent_iff_disc_pos]

/-- **`parabolic_one_channel`:**
At `σ = 0`, the two channels collapse into one (`ringUpper 0 = ringLower 0`), and the norm
form is a single channel squared: two-part addressing degenerates to one part. -/
theorem parabolic_one_channel (z : Cell) :
    ringUpper (0 : ℝ) z = ringLower (0 : ℝ) z ∧
    ringNormForm 0 z = (ringUpper (0 : ℝ) z) ^ 2 := by
  constructor
  · simp [ringUpper, ringLower]
  · simp [ringNormForm, ringUpper]

/-- Positive definiteness on the elliptic face: `ringNormForm σ z ≥ 0`, with equality iff `z = 0`. -/
theorem elliptic_normForm_pos_def {σ : ℝ} (hσ : σ < 0) (z : Cell) :
    0 ≤ ringNormForm σ z ∧ (ringNormForm σ z = 0 ↔ z = 0) := by
  have hpos : 0 < -σ := by linarith
  constructor
  · simp only [ringNormForm]
    have : z.1 ^ 2 - σ * z.2 ^ 2 = z.1 ^ 2 + (-σ) * z.2 ^ 2 := by ring
    rw [this]
    positivity
  · constructor
    · intro hz
      simp only [ringNormForm] at hz
      have h1 : z.1 = 0 := by
        have : z.1 ^ 2 ≤ 0 := by
          have h_eq : z.1 ^ 2 = - (-σ * z.2 ^ 2) := by linarith [hz]
          have : 0 ≤ -σ * z.2 ^ 2 := by positivity
          linarith
        have : 0 ≤ z.1 ^ 2 := sq_nonneg _
        have : z.1 ^ 2 = 0 := by linarith
        exact sq_eq_zero_iff.mp this
      have h2 : z.2 = 0 := by
        rw [h1] at hz
        have : -σ * z.2 ^ 2 = 0 := by linarith [hz]
        have hneg : -σ ≠ 0 := by linarith
        have hsq : z.2 ^ 2 = 0 := by
          cases mul_eq_zero.mp this with
          | inl h => exact False.elim (hneg h)
          | inr h => exact h
        exact sq_eq_zero_iff.mp hsq
      exact Prod.ext h1 h2
    · rintro rfl
      simp [ringNormForm]

/-- **`elliptic_no_channels`:**
At `σ < 0`, the norm form has no real linear factorization with independent coefficient vectors.
Two-phase separable search has no real channel coordinates.
(The elliptic face addresses instead by rotation on the unit circle `T`, generated by `rho θ`;
see `Nous/Model/RopeOrbit.lean` `rope_orbit`). -/
theorem elliptic_no_channels {σ : ℝ} (hσ : σ < 0) :
    ¬ FactorsIntoIndependentLinearForms σ := by
  intro hfac
  have hpos := (split_iff_hyperbolic σ).mp hfac
  linarith

end TrichotomyBoundary

/-! ## 7. Headline 6: The UltraMem Boundary -/

section UltraMemBoundary

variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]
variable (σ r : K) (hr : r ^ 2 = σ) (hr0 : r ≠ 0)

/-- **`rank1_core_is_cell`:**
A rank-1 Tucker score `(u · a_i) * (t · b_j)` with positive factors is the norm form of
the hyperbolic cell `slotCell r (u * a_i) (t * b_j)`, which equals the product of row and
column cells. Hence rank-1 cores are two-phase exact. -/
theorem rank1_core_is_cell (hr : r ^ 2 = σ) (hr0 : r ≠ 0) (u t a_i b_j : K) :
    (u * a_i) * (t * b_j) = ringNormForm σ (slotCell r (u * a_i) (t * b_j)) ∧
    ringNormForm σ (slotCell r (u * a_i) (t * b_j)) =
      ringNormForm σ (ringMul σ (slotCell r (u * a_i) 1) (slotCell r 1 (t * b_j))) := by
  have h1 := (slot_normForm σ r hr hr0 (u * a_i) (t * b_j)).symm
  have h2 : slotCell r (u * a_i) (t * b_j) = ringMul σ (slotCell r (u * a_i) 1) (slotCell r 1 (t * b_j)) :=
    (slot_mul σ r hr hr0 (u * a_i) (t * b_j)).symm
  exact ⟨h1, by rw [← h2]⟩

/-- Concrete row features for a rank-2 core over `ℚ` with `n = 3`. -/
def A_rank2 (i : Fin 3) (r_idx : Fin 2) : ℚ :=
  match i.val, r_idx.val with
  | 0, 0 => 10
  | 0, 1 => 0
  | 1, 0 => 6
  | 1, 1 => 6
  | 2, 0 => 0
  | 2, 1 => 10
  | _, _ => 0

/-- Concrete column features for a rank-2 core over `ℚ` with `m = 3`. -/
def B_rank2 (j : Fin 3) (c_idx : Fin 2) : ℚ :=
  match j.val, c_idx.val with
  | 0, 0 => 10
  | 0, 1 => 0
  | 1, 0 => 6
  | 1, 1 => 6
  | 2, 0 => 0
  | 2, 1 => 10
  | _, _ => 0

/-- The full grid score under an identity rank-2 core `C = I₂`. -/
def S_tucker (i : Fin 3) (j : Fin 3) : ℚ :=
  A_rank2 i 0 * B_rank2 j 0 + A_rank2 i 1 * B_rank2 j 1

/-- Row marginal score used by two-phase Tucker retrieval. -/
def M_row (i : Fin 3) : ℚ := S_tucker i 0 + S_tucker i 1 + S_tucker i 2

/-- Column marginal score used by two-phase Tucker retrieval. -/
def M_col (j : Fin 3) : ℚ := S_tucker 0 j + S_tucker 1 j + S_tucker 2 j

/-- **`tucker_rank2_breaks` (Non-vacuity of the rank-1 boundary):**
An explicit rank-2 core over `ℚ` with `n = m = 3` where row 1 and column 1 uniquely dominate
the marginals (`M_row 1 > M_row 0`, `M_row 1 > M_row 2`, `M_col 1 > M_col 0`, `M_col 1 > M_col 2`),
so the two-phase candidate set for top-1 is strictly `{(1, 1)}`.
Yet in the full grid, `(0, 0)` strictly beats `(1, 1)`: `S_tucker 1 1 < S_tucker 0 0` (72 < 100).
Thus the global top-1 is not in the candidate grid. -/
theorem tucker_rank2_breaks :
    M_row 0 < M_row 1 ∧ M_row 2 < M_row 1 ∧
    M_col 0 < M_col 1 ∧ M_col 2 < M_col 1 ∧
    S_tucker 1 1 < S_tucker 0 0 := by
  unfold M_row M_col S_tucker A_rank2 B_rank2
  norm_num

end UltraMemBoundary

/-! ## 8. Headline 7: Stretch `rank_r_threshold_exact` -/

section ThresholdAlgorithm

variable {K : Type*} [CommRing K] [LinearOrder K] [IsStrictOrderedRing K]
variable {α : Type*} [DecidableEq α]
variable {r : ℕ} (w : Fin r → K) (channelScore : Fin r → α → K)

/-- Total score as a weighted sum of `r` channels / hyperbolic cell norm forms. -/
def totalScore (p : α) : K :=
  ∑ l : Fin r, w l * channelScore l p

/-- Threshold value `T = ∑_l w_l * τ_l` from per-channel upper bounds. -/
def threshold (τ : Fin r → K) : K :=
  ∑ l : Fin r, w l * τ l

/-- Every unseen candidate's score is bounded by the threshold `T`. -/
theorem threshold_bound (hw : ∀ l, 0 ≤ w l) (τ : Fin r → K) (Seen : Finset α)
    (h_unseen : ∀ p ∉ Seen, ∀ l : Fin r, channelScore l p ≤ τ l)
    (p : α) (hp : p ∉ Seen) :
    totalScore w channelScore p ≤ threshold w τ := by
  unfold totalScore threshold
  apply Finset.sum_le_sum
  intro l _
  exact mul_le_mul_of_nonneg_left (h_unseen p hp l) (hw l)

/-- **`rank_r_threshold_exact`:**
For a score that is a nonnegative sum of `r` channel scores, if the threshold algorithm
encounters a subset `Top ⊆ Seen` of size `k` such that every element of `Top` scores at least
the threshold `T`, then no unseen candidate `p ∉ Seen` can exceed any element in `Top`. -/
theorem rank_r_threshold_exact (hw : ∀ l, 0 ≤ w l) (τ : Fin r → K) (Seen : Finset α)
    (h_unseen : ∀ p ∉ Seen, ∀ l : Fin r, channelScore l p ≤ τ l)
    {k : ℕ} {Top : Finset α} (_hTop_sub : Top ⊆ Seen) (_hTop_card : Top.card = k)
    (h_stopping : ∀ q ∈ Top, threshold w τ ≤ totalScore w channelScore q)
    (p : α) (hp : p ∉ Seen) :
    ∀ q ∈ Top, totalScore w channelScore p ≤ totalScore w channelScore q := by
  intro q hq
  have h1 := threshold_bound w channelScore hw τ Seen h_unseen p hp
  have h2 := h_stopping q hq
  exact le_trans h1 h2

end ThresholdAlgorithm

/-! ## 9. Concrete Witnesses -/

namespace Witness

-- Concrete instances at σ = 1 (r = 1)
theorem slot_mul_sigma1 :
    ringMul 1 (slotCell 1 (3 : ℚ) 1) (slotCell 1 1 (5 : ℚ)) = slotCell 1 3 5 := by
  apply slot_mul 1 1 (by norm_num) (by norm_num)

theorem slot_normForm_sigma1 :
    ringNormForm 1 (slotCell 1 (3 : ℚ) 5) = 15 := by
  dsimp [slotCell, ringOfHalves, ringNormForm]
  norm_num

-- Concrete instances at σ = 4 (r = 2)
theorem slot_mul_sigma4 :
    ringMul 4 (slotCell 2 (3 : ℚ) 1) (slotCell 2 1 (5 : ℚ)) = slotCell 2 3 5 := by
  apply slot_mul 4 2 (by norm_num) (by norm_num)

theorem slot_normForm_sigma4 :
    ringNormForm 4 (slotCell 2 (3 : ℚ) 5) = 15 := by
  dsimp [slotCell, ringOfHalves, ringNormForm]
  norm_num

-- 3×3 two-phase instance where top-1 is in the candidate grid
theorem two_phase_3x3_instance :
    let a : Fin 3 → ℚ := fun i => match i.val with | 0 => 10 | 1 => 6 | _ => 2
    let b : Fin 3 → ℚ := fun j => match j.val with | 0 => 8 | 1 => 5 | _ => 1
    (∀ i, 0 < a i) ∧ (∀ j, 0 < b j) ∧
    -- Top product score is at (0, 0)
    a 0 * b 0 = 80 ∧
    (∀ i j, a i * b j ≤ a 0 * b 0) := by
  dsimp
  refine ⟨?_, ?_, by norm_num, ?_⟩
  · intro ⟨i, hi⟩
    interval_cases i <;> norm_num
  · intro ⟨j, hj⟩
    interval_cases j <;> norm_num
  · intro ⟨i, hi⟩ ⟨j, hj⟩
    interval_cases i <;> interval_cases j <;> norm_num

-- Parabolic boundary witness (σ = 0)
theorem parabolic_boundary_witness (z : Cell) :
    ringUpper (0 : ℝ) z = ringLower (0 : ℝ) z ∧
    ringNormForm 0 z = (ringUpper (0 : ℝ) z) ^ 2 :=
  parabolic_one_channel z

-- Elliptic face witness (σ = -1)
theorem elliptic_boundary_witness :
    ¬ FactorsIntoIndependentLinearForms (-1) ∧
    (∀ z : Cell, 0 ≤ ringNormForm (-1) z ∧ (ringNormForm (-1) z = 0 ↔ z = 0)) :=
  ⟨elliptic_no_channels (by norm_num : (-1 : ℝ) < 0),
   fun z => elliptic_normForm_pos_def (by norm_num : (-1 : ℝ) < 0) z⟩

end Witness

end Nous.Seq.HyperbolicAddress
