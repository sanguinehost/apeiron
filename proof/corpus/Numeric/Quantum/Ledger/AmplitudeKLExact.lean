import Numeric.Quantum.Ledger.AmplitudeKL

/-!
# The exact (all-orders) identity relating the KL divergence to the amplitude distance

For positive normalised distributions `p`, `q` on a finite type `ι`, write the amplitude ratio
displacement `u i = √(q i)/√(p i) − 1`, so that `q i = p i * (1 + u i)^2` and `1 + u i > 0`, and the
per-term correction `h x = 2 x − x^2 − 2 log (1 + x)`. Then

`KL p q = 2 * ampDistSq p q + ∑ i, p i * h (u i)`.

`h` is antitone on `(-1, ∞)` with `h 0 = 0`, so the sign of `h (u i)` is the sign of `−u i`; this
gives `KL ≤ 2 * ampDistSq` when every `u i ≥ 0` and `KL ≥ 2 * ampDistSq` when every `u i ≤ 0`.
-/

namespace Quantum.AmplitudeKL

open Finset

set_option linter.unusedSectionVars false

variable {ι : Type*} [Fintype ι]

noncomputable section

/-- The amplitude ratio displacement `u i = √(q i)/√(p i) − 1`. -/
def u (p q : ι → ℝ) : ι → ℝ := fun i => Real.sqrt (q i) / Real.sqrt (p i) - 1

/-- The exact per-term correction `h x = 2 x − x^2 − 2 log (1 + x)`. -/
def h (x : ℝ) : ℝ := 2 * x - x ^ 2 - 2 * Real.log (1 + x)

/-- `1 + u` is the amplitude ratio `√q / √p`. -/
private lemma one_add_u_eq (p q : ι → ℝ) (i : ι) :
    1 + u p q i = Real.sqrt (q i) / Real.sqrt (p i) := by
  unfold u; ring

/-- `(1 + u)^2 = q / p`. -/
private lemma u_sq_eq_ratio {p q : ι → ℝ} (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i) (i : ι) :
    (1 + u p q i) ^ 2 = q i / p i := by
  rw [one_add_u_eq, div_pow, Real.sq_sqrt (le_of_lt (hq i)), Real.sq_sqrt (le_of_lt (hp i))]

/-- `q = p * (1 + u)^2`. -/
private lemma q_eq_mul {p q : ι → ℝ} (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i) (i : ι) :
    q i = p i * (1 + u p q i) ^ 2 := by
  rw [u_sq_eq_ratio hp hq i]
  have hpi : p i ≠ 0 := ne_of_gt (hp i)
  field_simp

/-- `√q = √p * (1 + u)`. -/
private lemma sqrt_q_eq {p q : ι → ℝ} (hp : ∀ i, 0 < p i) (i : ι) :
    Real.sqrt (q i) = Real.sqrt (p i) * (1 + u p q i) := by
  rw [one_add_u_eq]
  have hne : Real.sqrt (p i) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr (hp i))
  field_simp

/-- `-1 < u i` for positive `p`, `q`. -/
private lemma neg_one_lt_u {p q : ι → ℝ} (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i) (i : ι) :
    -1 < u p q i := by
  have hr : 0 < Real.sqrt (q i) / Real.sqrt (p i) :=
    div_pos (Real.sqrt_pos.mpr (hq i)) (Real.sqrt_pos.mpr (hp i))
  unfold u
  linarith

/-- `h 0 = 0`. -/
lemma h_zero : h 0 = 0 := by simp [h]

/-- The derivative of `h`: `h' x = −2 x^2 / (1 + x)` for `x > -1`. -/
theorem h_deriv (x : ℝ) (hx : -1 < x) :
    HasDerivAt h (-2 * x ^ 2 / (1 + x)) x := by
  have hne : (1 + x) ≠ 0 := by linarith
  unfold h
  have h1 : HasDerivAt (fun y : ℝ => 2 * y) 2 x := hasDerivAt_const_mul (x := x) 2
  have h2 : HasDerivAt (fun y : ℝ => y ^ 2) (2 * x) x := by
    simpa using hasDerivAt_pow (x := x) 2
  have h3 : HasDerivAt (fun y : ℝ => 2 * Real.log (1 + y)) (2 * (1 / (1 + x))) x :=
    (((hasDerivAt_id x).const_add 1).log hne).const_mul 2
  refine ((h1.sub h2).sub h3).congr_deriv ?_
  field_simp
  ring

/-- `h` is antitone on `(-1, ∞)`. -/
theorem h_antitone : AntitoneOn h (Set.Ioi (-1)) := by
  have hdiff : DifferentiableOn ℝ h (Set.Ioi (-1)) := fun x hx =>
    (h_deriv x hx).differentiableAt.differentiableWithinAt
  refine antitoneOn_of_hasDerivWithinAt_nonpos (f' := fun x : ℝ => -2 * x ^ 2 / (1 + x))
    (convex_Ioi (-1)) hdiff.continuousOn ?_ ?_
  · intro x hx
    rw [interior_Ioi] at hx
    exact (h_deriv x hx).hasDerivWithinAt
  · intro x hx
    rw [interior_Ioi] at hx
    have hxlt : -1 < x := hx
    exact div_nonpos_of_nonpos_of_nonneg (by nlinarith [sq_nonneg x]) (by linarith)

/-- `0 ≤ h x` for `-1 < x ≤ 0`. -/
theorem h_nonneg_of_le_zero (x : ℝ) (hx : -1 < x) (hx0 : x ≤ 0) : 0 ≤ h x := by
  have h0 : (0 : ℝ) ∈ Set.Ioi (-1) := by norm_num
  simpa [h_zero] using h_antitone hx h0 hx0

/-- `h x ≤ 0` for `0 ≤ x`. -/
theorem h_nonpos_of_ge_zero (x : ℝ) (hx0 : 0 ≤ x) : h x ≤ 0 := by
  have h0 : (0 : ℝ) ∈ Set.Ioi (-1) := by norm_num
  have hx' : x ∈ Set.Ioi (-1) := by show -1 < x; linarith
  simpa [h_zero] using h_antitone h0 hx' hx0

/-- **The exact identity (all orders).** For positive normalised `p`, `q`, the KL divergence equals
twice the squared amplitude distance plus the mean of `h` under `p` at the amplitude ratio `u`. -/
theorem kl_eq_amp_sq_add_h {p q : ι → ℝ} (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i)
    (hp1 : ∑ i, p i = 1) (hq1 : ∑ i, q i = 1) :
    KL p q = 2 * ampDistSq p q + ∑ i, p i * h (u p q i) := by
  have hsum_zero : ∑ i, p i * ((u p q i) ^ 2 + 2 * u p q i) = 0 := by
    have hterm : ∀ i, p i * ((u p q i) ^ 2 + 2 * u p q i) = q i - p i := by
      intro i; rw [q_eq_mul hp hq i]; ring
    rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_sub_distrib, hq1, hp1]
    ring
  have hkl : KL p q = -2 * ∑ i, p i * Real.log (1 + u p q i) := by
    have hterm : ∀ i, p i * Real.log (p i / q i) = -2 * (p i * Real.log (1 + u p q i)) := by
      intro i
      have hpq : p i / q i = ((1 + u p q i) ^ 2)⁻¹ := by
        rw [u_sq_eq_ratio hp hq i, inv_div]
      rw [hpq, Real.log_inv, Real.log_pow]
      ring
    unfold KL
    rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.mul_sum]
  have hamp : ampDistSq p q = ∑ i, p i * (u p q i) ^ 2 := by
    have hterm : ∀ i, (amp p i - amp q i) ^ 2 = p i * (u p q i) ^ 2 := by
      intro i
      have hqeq := sqrt_q_eq (p := p) (q := q) hp i
      unfold amp
      rw [hqeq]
      rw [show (Real.sqrt (p i) - Real.sqrt (p i) * (1 + u p q i)) ^ 2
            = (Real.sqrt (p i)) ^ 2 * (u p q i) ^ 2 by ring,
        Real.sq_sqrt (le_of_lt (hp i))]
    unfold ampDistSq
    exact Finset.sum_congr rfl fun i _ => hterm i
  have hRHS : 2 * ampDistSq p q + ∑ i, p i * h (u p q i)
      = -2 * ∑ i, p i * Real.log (1 + u p q i) := by
    have step1 : 2 * ampDistSq p q + ∑ i, p i * h (u p q i)
        = ∑ i, (2 * (p i * (u p q i) ^ 2) + p i * h (u p q i)) := by
      rw [hamp, Finset.mul_sum, ← Finset.sum_add_distrib]
    have step2 : ∀ i, 2 * (p i * (u p q i) ^ 2) + p i * h (u p q i)
        = p i * ((u p q i) ^ 2 + 2 * u p q i) - 2 * (p i * Real.log (1 + u p q i)) := by
      intro i; unfold h; ring
    rw [step1, Finset.sum_congr rfl fun i _ => step2 i, Finset.sum_sub_distrib,
      ← Finset.mul_sum, hsum_zero]
    ring
  rw [hkl, hRHS]

/-- When every amplitude ratio satisfies `u i ≥ 0`, the KL divergence is at most twice the squared
amplitude distance. -/
theorem kl_le_two_amp_sq_of_u_nonneg {p q : ι → ℝ} (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i)
    (hp1 : ∑ i, p i = 1) (hq1 : ∑ i, q i = 1) (huv : ∀ i, 0 ≤ u p q i) :
    KL p q ≤ 2 * ampDistSq p q := by
  rw [kl_eq_amp_sq_add_h hp hq hp1 hq1]
  have hle : ∑ i, p i * h (u p q i) ≤ 0 :=
    Finset.sum_nonpos fun i _ =>
      mul_nonpos_of_nonneg_of_nonpos (le_of_lt (hp i))
        (h_nonpos_of_ge_zero (u p q i) (huv i))
  linarith

/-- When every amplitude ratio satisfies `u i ≤ 0`, twice the squared amplitude distance is at most
the KL divergence. -/
theorem kl_ge_two_amp_sq_of_u_nonpos {p q : ι → ℝ} (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i)
    (hp1 : ∑ i, p i = 1) (hq1 : ∑ i, q i = 1) (huv : ∀ i, u p q i ≤ 0) :
    2 * ampDistSq p q ≤ KL p q := by
  rw [kl_eq_amp_sq_add_h hp hq hp1 hq1]
  have hge : 0 ≤ ∑ i, p i * h (u p q i) :=
    Finset.sum_nonneg fun i _ =>
      mul_nonneg (le_of_lt (hp i))
        (h_nonneg_of_le_zero (u p q i) (neg_one_lt_u hp hq i) (huv i))
  linarith

/-- Non-vacuity: the hypotheses of `kl_eq_amp_sq_add_h` hold for `p = ![1/2, 1/2]`,
`q = ![1/4, 3/4]` on `Fin 2`. -/
example :
    (∀ i : Fin 2, 0 < (![1 / 2, 1 / 2] : Fin 2 → ℝ) i) ∧
    (∀ i : Fin 2, 0 < (![1 / 4, 3 / 4] : Fin 2 → ℝ) i) ∧
    (∑ i : Fin 2, (![1 / 2, 1 / 2] : Fin 2 → ℝ) i) = 1 ∧
    (∑ i : Fin 2, (![1 / 4, 3 / 4] : Fin 2 → ℝ) i) = 1 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i; fin_cases i <;> norm_num
  · intro i; fin_cases i <;> norm_num
  · rw [Fin.sum_univ_two]; norm_num
  · rw [Fin.sum_univ_two]; norm_num

end

end Quantum.AmplitudeKL
