import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.MeanValue
import Perf.KLTilt

/-!
# Pinsker's inequality on a finite vocabulary

For probability vectors `p`, `q : Fin n → ℝ` with `q > 0`: `(∑ |p - q|)² ≤ 2 · KL(p ‖ q)`, i.e.
`TV(p, q) ≤ √(KL(p ‖ q) / 2)` (Pinsker 1960; Csiszár 1967; Kullback 1967).

Proof (Csiszár–Kemperman route, no binary reduction): with `t = p / q` and
`h t = t log t - t + 1`, the scalar inequality `3 (t - 1)² ≤ (2 t + 4) h t` holds for `t ≥ 0`
(`scalar_ineq`, by monotonicity of its difference quotient: `F' = 4 g`, `g' = log t + 1/t - 1 ≥ 0`).
Cauchy–Schwarz on `|p - q| = q |t - 1|` with `∑ q (2 t + 4) / 3 = 2` and `∑ q h t = KL` gives the bound.

Standard names: Kullback–Leibler divergence, total variation, Cauchy–Schwarz. No `sorry`, closed
cone `{propext, Classical.choice, Quot.sound}`.
-/

namespace Perf.Pinsker

open Finset Real

/-- `h t = t log t - t + 1`, the integrand of `KL` as an `f`-divergence (`h 0 = 1` since `0 log 0 = 0`). -/
noncomputable def hfun (t : ℝ) : ℝ := t * Real.log t - t + 1

/-- Difference `(2 t + 4) h t - 3 (t - 1)²`; nonnegative on `t ≥ 0`. -/
noncomputable def F (t : ℝ) : ℝ := (2 * t + 4) * hfun t - 3 * (t - 1) ^ 2

/-- `g t = (t + 1) log t - 2 (t - 1)`; `F' = 4 g`. -/
noncomputable def g (t : ℝ) : ℝ := (t + 1) * Real.log t - 2 * (t - 1)

theorem F_hasDerivAt {t : ℝ} (ht : 0 < t) : HasDerivAt F (4 * g t) t := by
  have hl : HasDerivAt Real.log t⁻¹ t := Real.hasDerivAt_log ht.ne'
  have h1 : HasDerivAt (fun s => s * Real.log s) (1 * Real.log t + t * t⁻¹) t :=
    (hasDerivAt_id t).mul hl
  have h2 : HasDerivAt hfun (1 * Real.log t + t * t⁻¹ - 1) t :=
    (h1.sub (hasDerivAt_id t)).add_const 1
  have h3 : HasDerivAt (fun s => 2 * s + 4) 2 t := by
    simpa using ((hasDerivAt_id t).const_mul 2).add_const 4
  have h4 : HasDerivAt (fun s => 3 * (s - 1) ^ 2) (3 * (2 * (t - 1))) t := by
    have := (((hasDerivAt_id t).sub_const 1).pow 2).const_mul 3
    simpa using this
  have h5 := (h3.mul h2).sub h4
  refine HasDerivAt.congr_deriv h5 ?_
  unfold g hfun
  have : t * t⁻¹ = 1 := mul_inv_cancel₀ ht.ne'
  rw [this]
  ring

theorem g_hasDerivAt {t : ℝ} (ht : 0 < t) : HasDerivAt g (Real.log t + t⁻¹ - 1) t := by
  have hl : HasDerivAt Real.log t⁻¹ t := Real.hasDerivAt_log ht.ne'
  have h1 : HasDerivAt (fun s => (s + 1) * Real.log s) (1 * Real.log t + (t + 1) * t⁻¹) t :=
    ((hasDerivAt_id t).add_const 1).mul hl
  have h2 : HasDerivAt (fun s => 2 * (s - 1)) 2 t := by
    simpa using ((hasDerivAt_id t).sub_const 1).const_mul 2
  have h3 := h1.sub h2
  refine HasDerivAt.congr_deriv h3 ?_
  have : (t + 1) * t⁻¹ = 1 + t⁻¹ := by field_simp
  rw [this]
  ring

theorem g_deriv_nonneg {t : ℝ} (ht : 0 < t) : 0 ≤ Real.log t + t⁻¹ - 1 := by
  have := Real.one_sub_inv_le_log_of_pos ht
  linarith

theorem g_monotone : MonotoneOn g (Set.Ioi 0) := by
  apply monotoneOn_of_deriv_nonneg (convex_Ioi 0)
  · intro t ht
    exact (g_hasDerivAt ht).continuousAt.continuousWithinAt
  · intro t ht
    rw [interior_Ioi] at ht
    exact (g_hasDerivAt ht).differentiableAt.differentiableWithinAt
  · intro t ht
    rw [interior_Ioi] at ht
    rw [(g_hasDerivAt ht).deriv]
    exact g_deriv_nonneg ht

theorem g_one : g 1 = 0 := by simp [g]

theorem g_nonneg_of_one_le {t : ℝ} (ht : 1 ≤ t) : 0 ≤ g t := by
  have := g_monotone (show (1:ℝ) ∈ Set.Ioi 0 by norm_num) (show t ∈ Set.Ioi 0 by
    simp only [Set.mem_Ioi]; linarith) ht
  linarith [g_one]

theorem g_nonpos_of_le_one {t : ℝ} (h0 : 0 < t) (ht : t ≤ 1) : g t ≤ 0 := by
  have := g_monotone (show t ∈ Set.Ioi 0 from h0) (show (1:ℝ) ∈ Set.Ioi 0 by norm_num) ht
  linarith [g_one]

theorem F_one : F 1 = 0 := by simp [F, hfun]

theorem F_nonneg_of_one_le {t : ℝ} (ht : 1 ≤ t) : 0 ≤ F t := by
  have hm : MonotoneOn F (Set.Ici 1) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici 1)
    · intro s hs
      exact (F_hasDerivAt (lt_of_lt_of_le one_pos hs)).continuousAt.continuousWithinAt
    · intro s hs
      rw [interior_Ici] at hs
      exact (F_hasDerivAt (lt_trans one_pos hs)).differentiableAt.differentiableWithinAt
    · intro s hs
      rw [interior_Ici] at hs
      rw [(F_hasDerivAt (lt_trans one_pos hs)).deriv]
      have := g_nonneg_of_one_le (le_of_lt (Set.mem_Ioi.mp hs))
      linarith
  have := hm (show (1:ℝ) ∈ Set.Ici 1 by simp) (show t ∈ Set.Ici 1 from ht) ht
  linarith [F_one]

theorem F_nonneg_of_le_one {t : ℝ} (h0 : 0 < t) (ht : t ≤ 1) : 0 ≤ F t := by
  have hm : AntitoneOn F (Set.Ioc 0 1) := by
    apply antitoneOn_of_deriv_nonpos (convex_Ioc 0 1)
    · intro s hs
      exact (F_hasDerivAt hs.1).continuousAt.continuousWithinAt
    · intro s hs
      rw [interior_Ioc] at hs
      exact (F_hasDerivAt hs.1).differentiableAt.differentiableWithinAt
    · intro s hs
      rw [interior_Ioc] at hs
      rw [(F_hasDerivAt hs.1).deriv]
      have := g_nonpos_of_le_one hs.1 (le_of_lt hs.2)
      linarith
  have := hm (show t ∈ Set.Ioc 0 1 from ⟨h0, ht⟩) (show (1:ℝ) ∈ Set.Ioc 0 1 by simp) ht
  linarith [F_one]

/-- **Scalar inequality.** `3 (t - 1)² ≤ (2 t + 4) (t log t - t + 1)` for `t ≥ 0`. -/
theorem scalar_ineq {t : ℝ} (ht : 0 ≤ t) :
    3 * (t - 1) ^ 2 ≤ (2 * t + 4) * (t * Real.log t - t + 1) := by
  rcases ht.eq_or_lt with h | h
  · subst h; norm_num [Real.log_zero]
  · have : 0 ≤ F t := by
      rcases le_total t 1 with h1 | h1
      · exact F_nonneg_of_le_one h h1
      · exact F_nonneg_of_one_le h1
    unfold F hfun at this
    linarith

theorem hfun_nonneg {t : ℝ} (ht : 0 ≤ t) : 0 ≤ hfun t := by
  have h := scalar_ineq ht
  have h4 : 0 < 2 * t + 4 := by linarith
  have h0 : 0 ≤ 3 * (t - 1) ^ 2 := by positivity
  unfold hfun
  by_contra hneg
  have hneg := not_le.mp hneg
  nlinarith

variable {n : ℕ}

/-- `KL(p ‖ q) = ∑ q h(p / q)` for `q > 0`, `∑ p = ∑ q = 1`. -/
theorem kl_eq_sum_hfun (p q : Fin n → ℝ) (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i)
    (hps : ∑ i, p i = 1) (hqs : ∑ i, q i = 1) :
    Perf.KLTilt.kl p q = ∑ i, q i * hfun (p i / q i) := by
  have hpt : ∀ i, q i * hfun (p i / q i) =
      p i * (Real.log (p i) - Real.log (q i)) - p i + q i := by
    intro i
    have hqi := (hq i).ne'
    unfold hfun
    rcases (hp i).eq_or_lt with h | h
    · rw [← h]; simp
    · rw [Real.log_div h.ne' hqi]
      field_simp
  rw [sum_congr rfl (fun i _ => hpt i), sum_add_distrib, sum_sub_distrib, hps, hqs]
  unfold Perf.KLTilt.kl
  ring

/-- **Pinsker's inequality.** `(∑ |p - q|)² ≤ 2 KL(p ‖ q)`. -/
theorem pinsker (p q : Fin n → ℝ) (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i)
    (hps : ∑ i, p i = 1) (hqs : ∑ i, q i = 1) :
    (∑ i, |p i - q i|) ^ 2 ≤ 2 * Perf.KLTilt.kl p q := by
  set t : Fin n → ℝ := fun i => p i / q i with ht
  have ht0 : ∀ i, 0 ≤ t i := fun i => div_nonneg (hp i) (hq i).le
  have hpq : ∀ i, p i = q i * t i := fun i => by have := (hq i).ne'; simp only [ht]; field_simp
  set A : Fin n → ℝ := fun i => q i * (2 * t i + 4) / 3 with hA
  set B : Fin n → ℝ := fun i => q i * hfun (t i) with hB
  have hA0 : ∀ i, 0 ≤ A i := fun i => by
    have := ht0 i; have := (hq i).le; simp only [hA]; positivity
  have hB0 : ∀ i, 0 ≤ B i := fun i => mul_nonneg (hq i).le (hfun_nonneg (ht0 i))
  have hu : ∀ i, |p i - q i| ≤ Real.sqrt (A i) * Real.sqrt (B i) := by
    intro i
    rw [← Real.sqrt_mul (hA0 i)]
    apply Real.abs_le_sqrt
    have hd : p i - q i = q i * (t i - 1) := by rw [hpq i]; ring
    rw [hd, mul_pow]
    have h := scalar_ineq (ht0 i)
    have hqq : 0 ≤ q i ^ 2 := sq_nonneg _
    have e : A i * B i = q i ^ 2 * ((2 * t i + 4) * hfun (t i)) / 3 := by
      simp only [hA, hB]; ring
    rw [e]
    unfold hfun at h ⊢
    nlinarith
  have hCS := Finset.sum_mul_sq_le_sq_mul_sq univ (fun i => Real.sqrt (A i)) (fun i => Real.sqrt (B i))
  have hsA : ∑ i, A i = 2 := by
    have : ∑ i, A i = (2 * ∑ i, q i * t i + 4 * ∑ i, q i) / 3 := by
      simp only [hA, ← sum_div, mul_sum, ← sum_add_distrib]
      congr 1
      exact sum_congr rfl (fun i _ => by ring)
    have e : ∑ i, q i * t i = 1 := by
      rw [← hps]; exact sum_congr rfl (fun i _ => (hpq i).symm)
    rw [this, hqs, e]; norm_num
  have hsB : ∑ i, B i = Perf.KLTilt.kl p q := (kl_eq_sum_hfun p q hp hq hps hqs).symm
  have hsq : ∀ i, Real.sqrt (A i) ^ 2 = A i := fun i => Real.sq_sqrt (hA0 i)
  have hsq' : ∀ i, Real.sqrt (B i) ^ 2 = B i := fun i => Real.sq_sqrt (hB0 i)
  simp only [hsq, hsq', hsA, hsB] at hCS
  have hle : ∑ i, |p i - q i| ≤ ∑ i, Real.sqrt (A i) * Real.sqrt (B i) := sum_le_sum (fun i _ => hu i)
  have hnn : 0 ≤ ∑ i, |p i - q i| := sum_nonneg (fun i _ => abs_nonneg _)
  calc (∑ i, |p i - q i|) ^ 2 ≤ (∑ i, Real.sqrt (A i) * Real.sqrt (B i)) ^ 2 :=
        pow_le_pow_left₀ hnn hle 2
    _ ≤ 2 * Perf.KLTilt.kl p q := hCS

/-- Pinsker in total-variation form: `TV := (1/2) ∑ |p - q|` satisfies `TV² ≤ KL / 2`. -/
theorem tv_sq_le (p q : Fin n → ℝ) (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i)
    (hps : ∑ i, p i = 1) (hqs : ∑ i, q i = 1) :
    ((1 / 2) * ∑ i, |p i - q i|) ^ 2 ≤ Perf.KLTilt.kl p q / 2 := by
  have := pinsker p q hp hq hps hqs
  nlinarith

end Perf.Pinsker
