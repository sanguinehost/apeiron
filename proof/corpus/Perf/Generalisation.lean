import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Tactic

/-!
# Loewner sandwich and argmin optimism for a quadratic code score

Score: `J A R = tr(R A Rᵀ)` for `A : d × d` and a code `R : m × d` (real matrices).
Order: the Loewner order, `B ≽ A` iff `B − A` is positive semidefinite.

Proved here (`proved`):
- `J_nonneg`, `J_mono`, `J_smul`: positivity, monotonicity and homogeneity of `J`.
- `loewner_sandwich`: if `Hh − lam·H ≽ 0` and `Lam·H − Hh ≽ 0` with `lam > 0`, then
  `lam·J H R ≤ J Hh R ≤ Lam·J H R` for every `R`.
- `argmin_optimism`: a minimiser `Rh` of `J Hh` over a finite code set `C` satisfies
  `J H Rh / J Hh Rh ≤ 1/lam` and `J H Rh ≤ (Lam/lam)·J H R'` for every `R' ∈ C`.
- `mp_floor_gt`: arithmetic on `√(19/200)`.
- `w17_c_sandwich_verdict`: the conjunction for the `ffn_down` shape `d = 4864`, `N = 51200`,
  `c = d/N = 19/200`.

Input, not a theorem: the Marchenko–Pastur edge `(1 − √c)²` for the aspect ratio `c = d/N` is
the anchor of Marchenko and Pastur (1967), with the almost-sure edge convergence of Bai and Yin
(1993). `mp_floor_gt` uses that edge only as a numeric bound; no random-matrix statement is
proved here.

Hypothesis added beyond the lane brief: `argmin_optimism` takes `H.PosSemidef`. The sandwich alone
does not fix the sign of `J Hh Rh`, so `J Hh Rh ≠ 0` does not give the positivity the division
needs; `H ≽ 0` gives `J H Rh ≥ 0`, and `J Hh Rh ≥ lam·J H Rh ≥ 0`.

Grades: `proved` for the Lean statements; the MP reading is an `anchor`.
Scope: real matrices; no statement about a trained weight matrix or a quantizer.
-/

noncomputable section

namespace Perf.Generalisation

open Matrix

variable {d m : ℕ}

/-- The quadratic code score `J A R = tr(R A Rᵀ)`. -/
def J (A : Matrix (Fin d) (Fin d) ℝ) (R : Matrix (Fin m) (Fin d) ℝ) : ℝ :=
  (R * A * R.transpose).trace

theorem J_nonneg {A : Matrix (Fin d) (Fin d) ℝ} (hA : A.PosSemidef)
    (R : Matrix (Fin m) (Fin d) ℝ) : 0 ≤ J A R := by
  unfold J
  apply Matrix.PosSemidef.trace_nonneg
  have h := hA.mul_mul_conjTranspose_same R
  rwa [Matrix.conjTranspose_eq_transpose_of_trivial] at h

theorem J_sub (A B : Matrix (Fin d) (Fin d) ℝ) (R : Matrix (Fin m) (Fin d) ℝ) :
    J (A - B) R = J A R - J B R := by
  unfold J
  rw [Matrix.mul_sub, Matrix.sub_mul, Matrix.trace_sub]

theorem J_mono {A B : Matrix (Fin d) (Fin d) ℝ} (hAB : (B - A).PosSemidef)
    (R : Matrix (Fin m) (Fin d) ℝ) : J A R ≤ J B R := by
  have h := J_nonneg hAB R
  rw [J_sub] at h
  linarith

theorem J_smul (c : ℝ) (A : Matrix (Fin d) (Fin d) ℝ) (R : Matrix (Fin m) (Fin d) ℝ) :
    J (c • A) R = c * J A R := by
  unfold J
  rw [Matrix.mul_smul, Matrix.smul_mul, Matrix.trace_smul, smul_eq_mul]

/-- **Loewner sandwich.** Under `lam > 0`, `Hh − lam·H ≽ 0` and `Lam·H − Hh ≽ 0`, the score of
`Hh` is bracketed by `lam` and `Lam` times the score of `H`, for every code `R`. -/
theorem loewner_sandwich {H Hh : Matrix (Fin d) (Fin d) ℝ} {lam Lam : ℝ} (_hlam : 0 < lam)
    (hlow : (Hh - lam • H).PosSemidef) (hup : (Lam • H - Hh).PosSemidef)
    (R : Matrix (Fin m) (Fin d) ℝ) :
    lam * J H R ≤ J Hh R ∧ J Hh R ≤ Lam * J H R := by
  have h1 := J_nonneg hlow R
  have h2 := J_nonneg hup R
  rw [J_sub, J_smul] at h1 h2
  constructor <;> linarith

/-- **Argmin optimism.** `Rh` minimises `J Hh` over the finite nonempty code set `C`, and `H ≽ 0`
with the sandwich. Then `J H Rh / J Hh Rh ≤ 1/lam` and `J H Rh ≤ (Lam/lam)·J H R'` for every
`R' ∈ C`. -/
theorem argmin_optimism {H Hh : Matrix (Fin d) (Fin d) ℝ} {lam Lam : ℝ}
    (C : Finset (Matrix (Fin m) (Fin d) ℝ)) (_hC : C.Nonempty)
    (hH : H.PosSemidef) (hlam : 0 < lam)
    (hlow : (Hh - lam • H).PosSemidef) (hup : (Lam • H - Hh).PosSemidef)
    {Rh : Matrix (Fin m) (Fin d) ℝ} (_hRh : Rh ∈ C)
    (hmin : ∀ R' ∈ C, J Hh Rh ≤ J Hh R')
    (hne : J Hh Rh ≠ 0) :
    J H Rh / J Hh Rh ≤ 1 / lam ∧
    ∀ R' ∈ C, J H Rh ≤ (Lam / lam) * J H R' := by
  have hHRh : 0 ≤ J H Rh := J_nonneg hH Rh
  have hsand : ∀ R : Matrix (Fin m) (Fin d) ℝ,
      lam * J H R ≤ J Hh R ∧ J Hh R ≤ Lam * J H R :=
    fun R => loewner_sandwich hlam hlow hup R
  have hpos : 0 < J Hh Rh := by
    have := (hsand Rh).1
    rcases lt_or_gt_of_ne hne with h | h
    · nlinarith
    · exact h
  have hlow_Rh : lam * J H Rh ≤ J Hh Rh := (hsand Rh).1
  constructor
  · rw [div_le_iff₀ hpos]
    have : J H Rh ≤ (1 / lam) * J Hh Rh := by
      rw [one_div, ← div_eq_inv_mul, le_div_iff₀ hlam]
      linarith
    linarith
  · intro R' hR'
    have hmR := hmin R' hR'
    have hup_R' := (hsand R').2
    have hdiv : J H Rh ≤ J Hh Rh / lam := by
      rw [le_div_iff₀ hlam]; linarith [hlow_Rh]
    have hdiv2 : J Hh Rh / lam ≤ Lam * J H R' / lam :=
      div_le_div_of_nonneg_right (le_trans hmR hup_R') hlam.le
    calc J H Rh ≤ J Hh Rh / lam := hdiv
      _ ≤ Lam * J H R' / lam := hdiv2
      _ = (Lam / lam) * J H R' := by ring

/-- Arithmetic of the anchor edge `(1 − √c)²` at `c = 19/200`: with `√c ∈ (0.308, 0.3083)`,
`(1 − √c)² > 47/100` and `1/(1 − √c)² < 21/10`. -/
theorem mp_floor_gt :
    (1 - Real.sqrt (19 / 200)) ^ 2 > 47 / 100 ∧
    1 / (1 - Real.sqrt (19 / 200)) ^ 2 < 21 / 10 := by
  have hs2 : Real.sqrt (19 / 200) ^ 2 = 19 / 200 :=
    Real.sq_sqrt (by norm_num)
  have hs0 : 0 ≤ Real.sqrt (19 / 200) := Real.sqrt_nonneg _
  have hlo : 0.308 < Real.sqrt (19 / 200) := by nlinarith
  have hhi : Real.sqrt (19 / 200) < 0.3083 := by nlinarith
  have hq : (1 - Real.sqrt (19 / 200)) ^ 2 > 47 / 100 := by nlinarith
  refine ⟨hq, ?_⟩
  rw [div_lt_iff₀ (by linarith)]
  nlinarith

/-- **Verdict for the `ffn_down` shape** (`d = 4864`, `N = 51200`, `c = d/N = 19/200`): the
sandwich, argmin optimism and the anchor arithmetic, as one conjunction. -/
theorem w17_c_sandwich_verdict :
    (4864 : ℚ) / 51200 = 19 / 200 ∧
    (∀ (m : ℕ) (H Hh : Matrix (Fin 4864) (Fin 4864) ℝ) (lam Lam : ℝ), 0 < lam →
      (Hh - lam • H).PosSemidef → (Lam • H - Hh).PosSemidef →
      ∀ R : Matrix (Fin m) (Fin 4864) ℝ,
        lam * J H R ≤ J Hh R ∧ J Hh R ≤ Lam * J H R) ∧
    (∀ (m : ℕ) (H Hh : Matrix (Fin 4864) (Fin 4864) ℝ) (lam Lam : ℝ)
      (C : Finset (Matrix (Fin m) (Fin 4864) ℝ)) (Rh : Matrix (Fin m) (Fin 4864) ℝ),
      C.Nonempty → H.PosSemidef → 0 < lam →
      (Hh - lam • H).PosSemidef → (Lam • H - Hh).PosSemidef →
      Rh ∈ C → (∀ R' ∈ C, J Hh Rh ≤ J Hh R') → J Hh Rh ≠ 0 →
      J H Rh / J Hh Rh ≤ 1 / lam ∧ ∀ R' ∈ C, J H Rh ≤ (Lam / lam) * J H R') ∧
    ((1 - Real.sqrt (19 / 200)) ^ 2 > 47 / 100 ∧
      1 / (1 - Real.sqrt (19 / 200)) ^ 2 < 21 / 10) := by
  refine ⟨by norm_num, ?_, ?_, mp_floor_gt⟩
  · intro m H Hh lam Lam hlam hlow hup R
    exact loewner_sandwich hlam hlow hup R
  · intro m H Hh lam Lam C Rh hC hH hlam hlow hup hRh hmin hne
    exact argmin_optimism C hC hH hlam hlow hup hRh hmin hne

end Perf.Generalisation
