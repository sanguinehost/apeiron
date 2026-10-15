import Nous.Seq.DeltaCarrier

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-!
# The two ends of the delta-rule family

. The canonical gated delta column step is `deltaCol α β v k s =
α • corr β k s + (β v) • k` with `corr β k s = s − β⟪s, k⟫ k` (`deltaCol_eq`). For a unit key
`k` the operator `corr β k` acts as `1 − β` on the line `k` and as the identity on `k⊥`.
Its endpoints are two classical objects:

- `β = 1`: the orthogonal projective measurement `I − kkᵀ` (von Neumann (1932); Lüders (1951)),
  with Born weight `⟪s, k⟫²` (`measure_idem`, `measure_born`).
- `β = 2`: the Householder reflection `I − 2kkᵀ` (Householder (1958)) (`reflect_invol`,
  `reflect_isometry`, `reflect_k`).

Within the admissible range `β ∈ [0, 2]` the step is nonexpansive (`nonexpansive_iff`), its only
idempotent is `β = 1` and its only isometry is `β = 2`, both besides the identity `β = 0`
(`idem_iff`, `invol_iff`, `isometry_iff`). Outside `[0, 2]` the step expands. The gated delta
rule is Yang, Kautz and Hatamizadeh (arXiv:2412.06464); the references above are anchors and are
not proved here. `verdict_l7` conjoins the headlines for `E := EuclideanSpace ℝ (Fin 2)` and
`k := e0`; `verdict_l7_general` states the same conjunction for every real inner product space
with a unit key.
-/

namespace Quantum.DeltaEnds

open Nous.Seq.DeltaCarrier
open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The rank-one correction `s ↦ s − β⟪s, k⟫ k` of the canonical step. -/
noncomputable def corr (β : ℝ) (k s : E) : E := s - (β * ⟪s, k⟫_ℝ) • k

theorem deltaCol_eq (α β v : ℝ) (k s : E) :
    deltaCol α β v k s = α • corr β k s + (β * v) • k := rfl

theorem coeff_eq_of_smul_k {a b : ℝ} (k : E) (hk : ‖k‖ = 1) (h : a • k = b • k) : a = b := by
  have hne : k ≠ 0 := by
    intro h0
    rw [h0, norm_zero] at hk
    exact zero_ne_one hk
  have h2 : (a - b) • k = 0 := by rw [sub_smul, h, sub_self]
  rcases smul_eq_zero.mp h2 with h3 | h3
  · linarith
  · exact absurd h3 hne

theorem inner_kk (k : E) (hk : ‖k‖ = 1) : ⟪k, k⟫_ℝ = 1 := by
  rw [real_inner_self_eq_norm_sq, hk]
  norm_num

theorem inner_corr_k (β : ℝ) (k s : E) (hk : ‖k‖ = 1) :
    ⟪corr β k s, k⟫_ℝ = (1 - β) * ⟪s, k⟫_ℝ := by
  unfold corr
  rw [inner_sub_left, real_inner_smul_left, inner_kk k hk]
  ring

theorem corr_k (β : ℝ) (k : E) (hk : ‖k‖ = 1) : corr β k k = (1 - β) • k := by
  unfold corr
  rw [inner_kk k hk, mul_one, sub_smul, one_smul]

theorem corr_orth (β : ℝ) (k w : E) (h : ⟪w, k⟫_ℝ = 0) : corr β k w = w := by
  unfold corr
  rw [h, mul_zero, zero_smul, sub_zero]

theorem corr_zero (k s : E) : corr 0 k s = s := by
  unfold corr
  rw [zero_mul, zero_smul, sub_zero]

theorem corr_comp (β γ : ℝ) (k s : E) (hk : ‖k‖ = 1) :
    corr β k (corr γ k s) = corr (β + γ - β * γ) k s := by
  unfold corr
  rw [inner_sub_left, real_inner_smul_left, inner_kk k hk]
  module

theorem corr_selfadj (β : ℝ) (k s t : E) :
    ⟪corr β k s, t⟫_ℝ = ⟪s, corr β k t⟫_ℝ := by
  unfold corr
  rw [inner_sub_left, inner_sub_right, real_inner_smul_left, real_inner_smul_right,
    real_inner_comm k t]
  ring

theorem norm_sq_corr (β : ℝ) (k s : E) (hk : ‖k‖ = 1) :
    ‖corr β k s‖ ^ 2 = ‖s‖ ^ 2 - β * (2 - β) * ⟪s, k⟫_ℝ ^ 2 := by
  unfold corr
  rw [norm_sub_sq_real, real_inner_smul_right, norm_smul, hk, mul_one, Real.norm_eq_abs,
    sq_abs]
  ring

theorem measure_idem (k s : E) (hk : ‖k‖ = 1) : corr 1 k (corr 1 k s) = corr 1 k s := by
  rw [corr_comp 1 1 k s hk]
  norm_num

theorem measure_born (k s : E) (hk : ‖k‖ = 1) :
    ‖corr 1 k s‖ ^ 2 + ⟪s, k⟫_ℝ ^ 2 = ‖s‖ ^ 2 := by
  rw [norm_sq_corr 1 k s hk]
  ring

theorem reflect_invol (k s : E) (hk : ‖k‖ = 1) : corr 2 k (corr 2 k s) = s := by
  rw [corr_comp 2 2 k s hk, show (2 : ℝ) + 2 - 2 * 2 = 0 by norm_num, corr_zero]

theorem reflect_isometry (k s : E) (hk : ‖k‖ = 1) : ‖corr 2 k s‖ = ‖s‖ := by
  have h := norm_sq_corr 2 k s hk
  have h2 : ‖corr 2 k s‖ ^ 2 = ‖s‖ ^ 2 := by
    rw [h]
    ring
  exact (pow_left_inj₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero).mp h2

theorem reflect_k (k : E) (hk : ‖k‖ = 1) : corr 2 k k = -k := by
  rw [corr_k 2 k hk, show (1 : ℝ) - 2 = -1 by norm_num, neg_one_smul]

theorem nonexpansive_iff (β : ℝ) (k : E) (hk : ‖k‖ = 1) :
    (∀ s : E, ‖corr β k s‖ ≤ ‖s‖) ↔ 0 ≤ β ∧ β ≤ 2 := by
  constructor
  · intro h
    have h1 := h k
    rw [corr_k β k hk, norm_smul, hk, mul_one, Real.norm_eq_abs] at h1
    obtain ⟨h1a, h1b⟩ := abs_le.mp h1
    exact ⟨by linarith, by linarith⟩
  · rintro ⟨h0, h2⟩ s
    exact householder_nonexpansive β h0 h2 k s hk

theorem idem_iff (β : ℝ) (k : E) (hk : ‖k‖ = 1) :
    (∀ s : E, corr β k (corr β k s) = corr β k s) ↔ β = 0 ∨ β = 1 := by
  constructor
  · intro h
    have hk0 := h k
    have e1 : corr (β + β - β * β) k k = (1 - (β + β - β * β)) • k := corr_k _ k hk
    have e2 : corr β k k = (1 - β) • k := corr_k β k hk
    rw [corr_comp β β k k hk, e1, e2] at hk0
    have hc := coeff_eq_of_smul_k k hk hk0
    have hm : β * (β - 1) = 0 := by linear_combination hc
    rcases mul_eq_zero.mp hm with h0 | h1
    · exact Or.inl h0
    · exact Or.inr (by linarith)
  · rintro (rfl | rfl) s
    · simp only [corr_zero]
    · exact measure_idem k s hk

theorem invol_iff (β : ℝ) (k : E) (hk : ‖k‖ = 1) :
    (∀ s : E, corr β k (corr β k s) = s) ↔ β = 0 ∨ β = 2 := by
  constructor
  · intro h
    have hk0 := h k
    have e1 : corr (β + β - β * β) k k = (1 - (β + β - β * β)) • k := corr_k _ k hk
    rw [corr_comp β β k k hk, e1] at hk0
    have hc := coeff_eq_of_smul_k k hk (hk0.trans (one_smul ℝ k).symm)
    have hm : β * (β - 2) = 0 := by linear_combination hc
    rcases mul_eq_zero.mp hm with h0 | h2
    · exact Or.inl h0
    · exact Or.inr (by linarith)
  · rintro (rfl | rfl) s
    · simp only [corr_zero]
    · exact reflect_invol k s hk

theorem isometry_iff (β : ℝ) (k : E) (hk : ‖k‖ = 1) :
    (∀ s : E, ‖corr β k s‖ = ‖s‖) ↔ β = 0 ∨ β = 2 := by
  constructor
  · intro h
    have h1 := h k
    rw [corr_k β k hk, norm_smul, hk, mul_one, Real.norm_eq_abs] at h1
    rcases (abs_eq (by norm_num : (0 : ℝ) ≤ 1)).mp h1 with h2 | h2
    · exact Or.inl (by linarith)
    · exact Or.inr (by linarith)
  · rintro (rfl | rfl) s
    · simp only [corr_zero]
    · exact reflect_isometry k s hk

/-- The conjunction of the headlines for a general unit key, with the four iffs for every `β`. -/
theorem verdict_l7_general {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    (k : F) (hk : ‖k‖ = 1) :
    (∀ s : F, corr 1 k (corr 1 k s) = corr 1 k s) ∧
    (∀ s : F, ‖corr 1 k s‖ ^ 2 + ⟪s, k⟫_ℝ ^ 2 = ‖s‖ ^ 2) ∧
    (∀ s : F, corr 2 k (corr 2 k s) = s) ∧
    (∀ s : F, ‖corr 2 k s‖ = ‖s‖) ∧
    corr 2 k k = -k ∧
    ∀ β : ℝ, ((∀ s : F, ‖corr β k s‖ ≤ ‖s‖) ↔ 0 ≤ β ∧ β ≤ 2) ∧
      ((∀ s : F, corr β k (corr β k s) = corr β k s) ↔ β = 0 ∨ β = 1) ∧
      ((∀ s : F, corr β k (corr β k s) = s) ↔ β = 0 ∨ β = 2) ∧
      ((∀ s : F, ‖corr β k s‖ = ‖s‖) ↔ β = 0 ∨ β = 2) :=
  ⟨fun s => measure_idem k s hk, fun s => measure_born k s hk, fun s => reflect_invol k s hk,
    fun s => reflect_isometry k s hk, reflect_k k hk,
    fun β => ⟨nonexpansive_iff β k hk, idem_iff β k hk, invol_iff β k hk, isometry_iff β k hk⟩⟩

/-- `verdict_l7_general` at the first basis vector of `ℝ²`. -/
theorem verdict_l7 :
    (∀ s : EuclideanSpace ℝ (Fin 2), corr 1 e0 (corr 1 e0 s) = corr 1 e0 s) ∧
    (∀ s : EuclideanSpace ℝ (Fin 2), ‖corr 1 e0 s‖ ^ 2 + ⟪s, e0⟫_ℝ ^ 2 = ‖s‖ ^ 2) ∧
    (∀ s : EuclideanSpace ℝ (Fin 2), corr 2 e0 (corr 2 e0 s) = s) ∧
    (∀ s : EuclideanSpace ℝ (Fin 2), ‖corr 2 e0 s‖ = ‖s‖) ∧
    corr 2 e0 e0 = -e0 ∧
    ∀ β : ℝ, ((∀ s : EuclideanSpace ℝ (Fin 2), ‖corr β e0 s‖ ≤ ‖s‖) ↔ 0 ≤ β ∧ β ≤ 2) ∧
      ((∀ s : EuclideanSpace ℝ (Fin 2), corr β e0 (corr β e0 s) = corr β e0 s) ↔
        β = 0 ∨ β = 1) ∧
      ((∀ s : EuclideanSpace ℝ (Fin 2), corr β e0 (corr β e0 s) = s) ↔ β = 0 ∨ β = 2) ∧
      ((∀ s : EuclideanSpace ℝ (Fin 2), ‖corr β e0 s‖ = ‖s‖) ↔ β = 0 ∨ β = 2) :=
  verdict_l7_general e0 e0_norm

end Quantum.DeltaEnds
