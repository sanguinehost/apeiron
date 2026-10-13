import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Taylor

/-!
# KL divergence between softmax tilts: half-variance identity and range bound

Finite vocabulary `Fin n` with `[NeZero n]`; logits `z`, tilt direction `δ : Fin n → ℝ`.

- `A z δ t` is the log-partition function of the tilt `t ↦ z + t δ` (a log-sum-exp).
- `sm z` is the softmax of `z`.
- `kl p q = ∑ p (log p − log q)` is the Kullback–Leibler divergence of probability vectors.
- `wvar w δ = ∑ w δ² − (∑ w δ)²` is the variance of `δ` under the probability vector `w`.

Proved (`proved`, closed cone `{propext, Classical.choice, Quot.sound}`):

- `kl_sm_eq`: `KL(sm z ‖ sm (z+δ)) = A 1 − A 0 − E_{sm z}[δ]`.
- `A_hasDerivAt`, `A_second`: `A'(t) = E_{sm(z+tδ)}[δ]`, `A''(t) = Var_{sm(z+tδ)}(δ)`.
- `kl_eq_half_var`: `KL = ½ Var_{sm(z+ξδ)}(δ)` for some `ξ ∈ (0,1)`, by two applications of Rolle's theorem.
- `wvar_le_range`: `Var_w(δ) ≤ (M − m)²/4` when `δ ∈ [m, M]` (Popoviciu).
- `kl_le_range_sq_div_8`: `KL ≤ (M − m)²/8`.

Standard names: softmax, log-partition function, variance, exponential family (Gibbs, 1902).
The identity `KL = ½ Var` is the second-order Taylor remainder of the log-partition function.

Non-vacuity: `witness_half_var` and `witness_range` instantiate both at `n = 2`, `z = (0, 0)`,
`δ = (1, −1)`.
-/

namespace Perf.KLTilt

open Finset

variable {n : ℕ} [NeZero n]

/-- Log-partition function of the tilt `t ↦ z + t δ`. -/
noncomputable def A (z δ : Fin n → ℝ) (t : ℝ) : ℝ :=
  Real.log (∑ j, Real.exp (z j + t * δ j))

/-- Softmax. -/
noncomputable def sm (z : Fin n → ℝ) (i : Fin n) : ℝ :=
  Real.exp (z i) / ∑ j, Real.exp (z j)

/-- Kullback–Leibler divergence of probability vectors. -/
noncomputable def kl (p q : Fin n → ℝ) : ℝ :=
  ∑ i, p i * (Real.log (p i) - Real.log (q i))

/-- Variance of `δ` under the weights `w`. -/
noncomputable def wvar (w δ : Fin n → ℝ) : ℝ :=
  ∑ i, w i * δ i ^ 2 - (∑ i, w i * δ i) ^ 2

lemma sum_exp_pos (z : Fin n → ℝ) : 0 < ∑ j, Real.exp (z j) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨0, Finset.mem_univ _⟩

lemma sum_exp_pos' (z δ : Fin n → ℝ) (t : ℝ) : 0 < ∑ j, Real.exp (z j + t * δ j) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨0, Finset.mem_univ _⟩

lemma sm_nonneg (z : Fin n → ℝ) (i : Fin n) : 0 ≤ sm z i :=
  div_nonneg (Real.exp_nonneg _) (sum_exp_pos z).le

lemma sm_sum (z : Fin n → ℝ) : ∑ i, sm z i = 1 := by
  unfold sm
  rw [← Finset.sum_div]
  exact div_self (sum_exp_pos z).ne'

lemma log_sm (z : Fin n → ℝ) (i : Fin n) :
    Real.log (sm z i) = z i - Real.log (∑ j, Real.exp (z j)) := by
  unfold sm
  rw [Real.log_div (Real.exp_pos _).ne' (sum_exp_pos z).ne', Real.log_exp]

omit [NeZero n] in
lemma A_zero (z δ : Fin n → ℝ) : A z δ 0 = Real.log (∑ j, Real.exp (z j)) := by
  simp [A]

omit [NeZero n] in
lemma A_one (z δ : Fin n → ℝ) : A z δ 1 = Real.log (∑ j, Real.exp (z j + δ j)) := by
  simp [A]

omit [NeZero n] in
lemma sum_tilt (w δ : Fin n → ℝ) (C : ℝ) (hw : ∑ i, w i = 1) :
    ∑ i, w i * (-(δ i) + C) = -(∑ i, w i * δ i) + C := by
  have e : ∀ i, w i * (-(δ i) + C) = -(w i * δ i) + C * w i := fun i => by ring
  simp only [e, Finset.sum_add_distrib, Finset.sum_neg_distrib, ← Finset.mul_sum, hw, mul_one]

/-- The KL divergence between two softmax tilts is the log-partition increment minus the mean tilt. -/
theorem kl_sm_eq (z δ : Fin n → ℝ) :
    kl (sm z) (sm (z + δ)) = A z δ 1 - A z δ 0 - ∑ i, sm z i * δ i := by
  have hterm : ∀ i, sm z i * (Real.log (sm z i) - Real.log (sm (z + δ) i))
      = sm z i * (-(δ i) + (Real.log (∑ j, Real.exp (z j + 1 * δ j)) - Real.log (∑ j, Real.exp (z j)))) := by
    intro i
    rw [log_sm, log_sm]
    have h1 : ∑ j, Real.exp ((z + δ) j) = ∑ j, Real.exp (z j + 1 * δ j) := by simp
    rw [h1]
    simp only [Pi.add_apply]
    ring
  unfold kl
  rw [Finset.sum_congr rfl (fun i _ => hterm i), sum_tilt (sm z) δ _ (sm_sum z), A_one, A_zero]
  ring

omit [NeZero n] in
lemma hasDerivAt_expt (z δ : Fin n → ℝ) (j : Fin n) (t : ℝ) :
    HasDerivAt (fun s => Real.exp (z j + s * δ j)) (Real.exp (z j + t * δ j) * δ j) t := by
  have h := (((hasDerivAt_id t).mul_const (δ j)).const_add (z j)).exp
  simpa using h

lemma hasDerivAt_A (z δ : Fin n → ℝ) (t : ℝ) :
    HasDerivAt (A z δ) ((∑ j, Real.exp (z j + t * δ j) * δ j) / ∑ j, Real.exp (z j + t * δ j)) t := by
  have hD : HasDerivAt (fun s => ∑ j, Real.exp (z j + s * δ j))
      (∑ j, Real.exp (z j + t * δ j) * δ j) t :=
    HasDerivAt.fun_sum (fun j _ => hasDerivAt_expt z δ j t)
  have h := hD.log (sum_exp_pos' z δ t).ne'
  exact h

omit [NeZero n] in
lemma sm_dot (z δ : Fin n → ℝ) (t : ℝ) :
    ∑ i, sm (z + t • δ) i * δ i
      = (∑ j, Real.exp (z j + t * δ j) * δ j) / ∑ j, Real.exp (z j + t * δ j) := by
  unfold sm
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- First derivative of the log-partition function is the mean tilt under the tilted softmax. -/
theorem A_hasDerivAt (z δ : Fin n → ℝ) (t : ℝ) :
    deriv (A z δ) t = ∑ i, sm (z + t • δ) i * δ i := by
  rw [(hasDerivAt_A z δ t).deriv, sm_dot]

omit [NeZero n] in
/-- Variance of `δ` under a softmax, as a quotient. -/
lemma wvar_softmax (e δ : Fin n → ℝ) (hS : (∑ j, e j) ≠ 0) :
    wvar (fun i => e i / ∑ j, e j) δ
      = ((∑ i, e i * δ i * δ i) * (∑ j, e j) - (∑ i, e i * δ i) * (∑ i, e i * δ i)) / (∑ j, e j) ^ 2 := by
  simp only [wvar]
  have h1 : ∑ i, (e i / ∑ j, e j) * δ i ^ 2 = (∑ i, e i * δ i * δ i) / ∑ j, e j := by
    rw [Finset.sum_div]
    apply Finset.sum_congr rfl
    intro i _
    ring
  have h2 : ∑ i, (e i / ∑ j, e j) * δ i = (∑ i, e i * δ i) / ∑ j, e j := by
    rw [Finset.sum_div]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [h1, h2]
  field_simp

lemma hasDerivAt_a1 (z δ : Fin n → ℝ) (t : ℝ) :
    HasDerivAt (fun s => (∑ j, Real.exp (z j + s * δ j) * δ j) / ∑ j, Real.exp (z j + s * δ j))
      (wvar (sm (z + t • δ)) δ) t := by
  have hN : HasDerivAt (fun s => ∑ j, Real.exp (z j + s * δ j) * δ j)
      (∑ j, Real.exp (z j + t * δ j) * δ j * δ j) t :=
    HasDerivAt.fun_sum (fun j _ => (hasDerivAt_expt z δ j t).mul_const (δ j))
  have hD : HasDerivAt (fun s => ∑ j, Real.exp (z j + s * δ j))
      (∑ j, Real.exp (z j + t * δ j) * δ j) t :=
    HasDerivAt.fun_sum (fun j _ => hasDerivAt_expt z δ j t)
  have h := hN.div hD (sum_exp_pos' z δ t).ne'
  have hsm : sm (z + t • δ) = fun i => Real.exp (z i + t * δ i) / ∑ j, Real.exp (z j + t * δ j) := by
    funext i
    simp [sm]
  exact h.congr_deriv (by rw [hsm, wvar_softmax _ _ (sum_exp_pos' z δ t).ne'])

/-- Second derivative of the log-partition function is the variance of `δ` under the tilted softmax. -/
theorem A_second (z δ : Fin n → ℝ) (t : ℝ) :
    deriv (deriv (A z δ)) t = wvar (sm (z + t • δ)) δ := by
  have hfun : deriv (A z δ) = fun s => (∑ j, Real.exp (z j + s * δ j) * δ j) / ∑ j, Real.exp (z j + s * δ j) :=
    funext (fun s => (hasDerivAt_A z δ s).deriv)
  rw [hfun]
  exact (hasDerivAt_a1 z δ t).deriv

/-- Half-variance identity: the KL divergence between a softmax and its tilt is `½ Var` at an
interior point. Route: Rolle on `A t − A 0 − t A'(0) − KL t²`, then Rolle on its derivative. -/
theorem kl_eq_half_var (z δ : Fin n → ℝ) :
    ∃ ξ ∈ Set.Ioo (0:ℝ) 1, kl (sm z) (sm (z + δ)) = (1/2) * wvar (sm (z + ξ • δ)) δ := by
  set g1 := kl (sm z) (sm (z + δ)) with hg1
  set a1 : ℝ → ℝ := fun t => ∑ i, sm (z + t • δ) i * δ i with ha1
  have hk : g1 = A z δ 1 - A z δ 0 - a1 0 := by
    rw [hg1, kl_sm_eq]
    simp [a1]
  have hAd : ∀ t, HasDerivAt (A z δ) (a1 t) t := by
    intro t
    show HasDerivAt (A z δ) (∑ i, sm (z + t • δ) i * δ i) t
    rw [sm_dot]
    exact hasDerivAt_A z δ t
  have ha1eq : a1 = fun s => (∑ j, Real.exp (z j + s * δ j) * δ j) / ∑ j, Real.exp (z j + s * δ j) :=
    funext (fun s => sm_dot z δ s)
  have hA2 : ∀ t, HasDerivAt a1 (wvar (sm (z + t • δ)) δ) t := by
    intro t
    rw [ha1eq]
    exact hasDerivAt_a1 z δ t
  -- Rolle 1
  have hf : ∀ t, HasDerivAt (fun x => A z δ x - A z δ 0 - x * a1 0 - g1 * x ^ 2)
      (a1 t - a1 0 - 2 * g1 * t) t := by
    intro t
    have h := (((hAd t).sub_const (A z δ 0)).sub ((hasDerivAt_id t).mul_const (a1 0))).sub
      ((hasDerivAt_pow 2 t).const_mul g1)
    exact h.congr_deriv (by rw [show (2:ℕ) - 1 = 1 from rfl]; ring)
  obtain ⟨η, hη, hηd⟩ := exists_hasDerivAt_eq_zero (f := fun x => A z δ x - A z δ 0 - x * a1 0 - g1 * x ^ 2)
    (f' := fun t => a1 t - a1 0 - 2 * g1 * t) (by norm_num : (0:ℝ) < 1)
    (fun x _ => (hf x).continuousAt.continuousWithinAt)
    (by
      show A z δ 0 - A z δ 0 - 0 * a1 0 - g1 * 0 ^ 2 = A z δ 1 - A z δ 0 - 1 * a1 0 - g1 * 1 ^ 2
      rw [hk]
      ring)
    (fun x _ => hf x)
  -- Rolle 2 on [0, η]
  have hF : ∀ t, HasDerivAt (fun x => a1 x - a1 0 - 2 * g1 * x) (wvar (sm (z + t • δ)) δ - 2 * g1) t := by
    intro t
    have h := ((hA2 t).sub_const (a1 0)).sub ((hasDerivAt_id t).const_mul (2 * g1))
    exact h.congr_deriv (by ring)
  obtain ⟨ξ, hξ, hξd⟩ := exists_hasDerivAt_eq_zero (f := fun x => a1 x - a1 0 - 2 * g1 * x)
    (f' := fun t => wvar (sm (z + t • δ)) δ - 2 * g1) hη.1
    (fun x _ => (hF x).continuousAt.continuousWithinAt)
    (by
      show a1 0 - a1 0 - 2 * g1 * 0 = a1 η - a1 0 - 2 * g1 * η
      linarith [hηd])
    (fun x _ => hF x)
  refine ⟨ξ, ⟨hξ.1, hξ.2.trans hη.2⟩, ?_⟩
  have hξd' : wvar (sm (z + ξ • δ)) δ - 2 * g1 = 0 := hξd
  linarith [hξd']

omit [NeZero n] in
/-- Popoviciu: the variance of `δ ∈ [m, M]` under a probability vector is at most `(M − m)²/4`. -/
theorem wvar_le_range (w δ : Fin n → ℝ) (m M : ℝ) (hw : ∀ i, 0 ≤ w i) (hs : ∑ i, w i = 1)
    (hm : ∀ i, m ≤ δ i) (hM : ∀ i, δ i ≤ M) : wvar w δ ≤ (M - m) ^ 2 / 4 := by
  have key : ∀ c : ℝ, wvar w δ ≤ ∑ i, w i * (δ i - c) ^ 2 := by
    intro c
    have e : ∀ i, w i * (δ i - c) ^ 2 = w i * δ i ^ 2 - 2 * c * (w i * δ i) + c ^ 2 * w i :=
      fun i => by ring
    simp only [wvar, e, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hs, mul_one]
    nlinarith [sq_nonneg (∑ i, w i * δ i - c)]
  have hbound : ∑ i, w i * (δ i - (M + m) / 2) ^ 2 ≤ (M - m) ^ 2 / 4 := by
    have hK : ∀ i, w i * (δ i - (M + m) / 2) ^ 2 ≤ w i * ((M - m) / 2) ^ 2 := by
      intro i
      refine mul_le_mul_of_nonneg_left ?_ (hw i)
      apply sq_le_sq'
      · linarith [hm i, hM i]
      · linarith [hm i, hM i]
    calc ∑ i, w i * (δ i - (M + m) / 2) ^ 2
        ≤ ∑ i, w i * ((M - m) / 2) ^ 2 := Finset.sum_le_sum (fun i _ => hK i)
      _ = (M - m) ^ 2 / 4 := by rw [← Finset.sum_mul, hs]; ring
  exact (key _).trans hbound

/-- Range bound: `KL(sm z ‖ sm (z+δ)) ≤ (M − m)²/8` when `δ ∈ [m, M]`. -/
theorem kl_le_range_sq_div_8 (z δ : Fin n → ℝ) (m M : ℝ) (hm : ∀ i, m ≤ δ i) (hM : ∀ i, δ i ≤ M) :
    kl (sm z) (sm (z + δ)) ≤ (M - m) ^ 2 / 8 := by
  obtain ⟨ξ, _, hξ⟩ := kl_eq_half_var z δ
  rw [hξ]
  have h := wvar_le_range (sm (z + ξ • δ)) δ m M (fun i => sm_nonneg _ i) (sm_sum _) hm hM
  linarith

/-- Non-vacuity: the half-variance identity at `n = 2`, `z = (0, 0)`, `δ = (1, −1)`. -/
theorem witness_half_var :
    ∃ ξ ∈ Set.Ioo (0:ℝ) 1,
      kl (sm (![0, 0] : Fin 2 → ℝ)) (sm ((![0, 0] : Fin 2 → ℝ) + ![1, -1]))
        = (1/2) * wvar (sm ((![0, 0] : Fin 2 → ℝ) + ξ • (![1, -1] : Fin 2 → ℝ))) ![1, -1] :=
  kl_eq_half_var _ _

/-- Non-vacuity: the range bound at `n = 2`, `z = (0, 0)`, `δ = (1, −1)`, `m = −1`, `M = 1`. -/
theorem witness_range :
    kl (sm (![0, 0] : Fin 2 → ℝ)) (sm ((![0, 0] : Fin 2 → ℝ) + ![1, -1]))
      ≤ ((1:ℝ) - (-1)) ^ 2 / 8 := by
  apply kl_le_range_sq_div_8 _ _ (-1) 1
  · intro i
    fin_cases i <;> norm_num
  · intro i
    fin_cases i <;> norm_num

end Perf.KLTilt
