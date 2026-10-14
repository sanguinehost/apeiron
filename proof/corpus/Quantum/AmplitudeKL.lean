import Mathlib

/-!
# The amplitude form of the KL divergence

Distributions on a finite type `ι` are carried by their amplitudes `ψ i = √(p i)`; this module
relates the Kullback--Leibler divergence `KL p q = ∑ p i * log (p i / q i)` to the Euclidean
distance of the amplitudes.

* `kl_ge_amplitude_sq`: `‖ψ − φ‖² ≤ KL p q`, the Hellinger-type first-order bound.
* `kl_second_order`: along the amplitude curve `q_s i = (√(p i) + s * v i)² / Z s` through `p` in
  the direction `v` with `∑ √(p i) * v i = 0`, the curves `s ↦ KL p (q_s)` and
  `s ↦ 2 ‖√(q_s) − √p‖²` agree in value and in first and second derivative at `s = 0`
  (both second derivatives equal `4 * ∑ v i ^ 2`).
* `logistic_eq_softmax_two`, `logistic_amplitude_add`: the logistic gate is the two-way softmax
  against a zero logit, and its two amplitudes sum to one in squares.
-/

namespace Quantum.AmplitudeKL

open Finset

open scoped Topology

variable {ι : Type*} [Fintype ι]

noncomputable section

/-- The Kullback--Leibler divergence `∑ p i * log (p i / q i)`. -/
def KL (p q : ι → ℝ) : ℝ := ∑ i, p i * Real.log (p i / q i)

/-- The amplitude `ψ i = √(p i)` of a distribution `p`. -/
def amp (p : ι → ℝ) : ι → ℝ := fun i => Real.sqrt (p i)

/-- The squared amplitude distance `‖ψ − φ‖² = ∑ (√(p i) − √(q i))²`. -/
def ampDistSq (p q : ι → ℝ) : ℝ := ∑ i, (amp p i - amp q i) ^ 2

/-- The squared norm `∑ v i ^ 2` of a displacement. -/
def sqNorm (v : ι → ℝ) : ℝ := ∑ i, (v i) ^ 2

/-- The logistic gate `σ x = 1 / (1 + e^{-x})`. -/
def logistic (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

lemma sqNorm_nonneg (v : ι → ℝ) : 0 ≤ sqNorm v :=
  Finset.sum_nonneg fun i _ => sq_nonneg (v i)

/-- The per-term inequality `2 a − 2 √(a b) ≤ a log (a / b)` for `a, b > 0`, obtained from
`log y ≤ y − 1` at `y = √(b / a)`. -/
private lemma two_mul_sub_two_mul_sqrt_le {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    2 * a - 2 * Real.sqrt (a * b) ≤ a * Real.log (a / b) := by
  set u := Real.sqrt (a / b) with hu
  have hba : 0 < a / b := div_pos ha hb
  have hu_pos : 0 < u := Real.sqrt_pos.mpr hba
  have hu_sq : u ^ 2 = a / b := Real.sq_sqrt (le_of_lt hba)
  have ha_eq : a = u ^ 2 * b := by rw [hu_sq]; field_simp
  have hsqrt : Real.sqrt (a * b) = u * b := by
    have h : a * b = (u * b) ^ 2 := by rw [ha_eq]; ring
    rw [h, Real.sqrt_sq_eq_abs, abs_of_pos (mul_pos hu_pos hb)]
  have hlog : Real.log (a / b) = 2 * Real.log u := by
    rw [← hu_sq, Real.log_pow]
    norm_num
  have hkey : u - 1 ≤ u * Real.log u := by
    have h := Real.log_le_sub_one_of_pos (inv_pos.mpr hu_pos)
    rw [Real.log_inv] at h
    have h' : 1 - u⁻¹ ≤ Real.log u := by linarith
    have hmul := mul_le_mul_of_nonneg_left h' (le_of_lt hu_pos)
    have heq : u * (1 - u⁻¹) = u - 1 := by field_simp
    linarith [hmul, heq]
  rw [hsqrt, hlog, ha_eq]
  have hnn : 0 ≤ 2 * u * b := by positivity
  nlinarith [mul_le_mul_of_nonneg_left hkey hnn]

/-- **Clause 1.** `‖ψ − φ‖² ≤ KL p q` for distributions `p`, `q` with full support. -/
theorem kl_ge_amplitude_sq {p q : ι → ℝ} (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i)
    (hp1 : ∑ i, p i = 1) (hq1 : ∑ i, q i = 1) :
    ampDistSq p q ≤ KL p q := by
  have hterm : ∀ i, (amp p i - amp q i) ^ 2 = p i + q i - 2 * Real.sqrt (p i * q i) := by
    intro i
    have h1 : (amp p i) ^ 2 = p i := Real.sq_sqrt (le_of_lt (hp i))
    have h2 : (amp q i) ^ 2 = q i := Real.sq_sqrt (le_of_lt (hq i))
    unfold amp at h1 h2 ⊢
    rw [Real.sqrt_mul (le_of_lt (hp i)) (q i), sub_sq, h1, h2]
    ring
  have hleft : ∑ i, (p i + q i - 2 * Real.sqrt (p i * q i))
      = 2 - 2 * ∑ i, Real.sqrt (p i * q i) := by
    rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, hp1, hq1, ← Finset.mul_sum]
    ring
  have hright : ∑ i, (2 * p i - 2 * Real.sqrt (p i * q i))
      = 2 - 2 * ∑ i, Real.sqrt (p i * q i) := by
    rw [Finset.sum_sub_distrib, ← Finset.mul_sum, hp1, ← Finset.mul_sum]
    ring
  calc ampDistSq p q = ∑ i, (p i + q i - 2 * Real.sqrt (p i * q i)) := by
        unfold ampDistSq
        exact Finset.sum_congr rfl fun i _ => hterm i
    _ = ∑ i, (2 * p i - 2 * Real.sqrt (p i * q i)) := hleft.trans hright.symm
    _ ≤ KL p q := Finset.sum_le_sum fun i _ => two_mul_sub_two_mul_sqrt_le (hp i) (hq i)

/-- **Clause 3.** The logistic gate is the two-way softmax against a zero logit. -/
theorem logistic_eq_softmax_two (x : ℝ) :
    1 / (1 + Real.exp (-x)) = Real.exp x / (Real.exp x + Real.exp 0) := by
  have hx : Real.exp x ≠ 0 := Real.exp_ne_zero x
  have hden : 1 + Real.exp (-x) ≠ 0 := by
    have := Real.exp_pos (-x)
    positivity
  have h : Real.exp x * Real.exp (-x) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  rw [Real.exp_zero, div_eq_div_iff hden (by positivity)]
  rw [one_mul, mul_add, mul_one, h]

/-- The amplitude form of the logistic gate: the squared amplitudes of the two directions sum to one. -/
theorem logistic_amplitude_add (x : ℝ) :
    (Real.sqrt (logistic x)) ^ 2 + (Real.sqrt (logistic (-x))) ^ 2 = 1 := by
  have hnonneg : 0 ≤ logistic x := by
    unfold logistic
    have := Real.exp_pos (-x)
    positivity
  have hnonneg' : 0 ≤ logistic (-x) := by
    unfold logistic
    have := Real.exp_pos x
    positivity
  rw [Real.sq_sqrt hnonneg, Real.sq_sqrt hnonneg']
  unfold logistic
  rw [neg_neg]
  have h : Real.exp (-x) * Real.exp x = 1 := by
    rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]
  have hden : (1 + Real.exp (-x)) * (1 + Real.exp x) ≠ 0 := by
    have h1 := Real.exp_pos (-x)
    have h2 := Real.exp_pos x
    positivity
  field_simp
  nlinarith [h]

/-! ## Clause 2: the amplitude curve through a distribution -/

/-- The normalising partition of the amplitude curve `√(p i) + s * v i`. -/
def Z (p v : ι → ℝ) (s : ℝ) : ℝ := ∑ i, (amp p i + s * v i) ^ 2

/-- The amplitude curve through `p` in the direction `v`: `q_s i = (√(p i) + s v i)² / Z s`. -/
def qs (p v : ι → ℝ) (s : ℝ) : ι → ℝ := fun i => (amp p i + s * v i) ^ 2 / Z p v s

/-- `‖√(q_s) − √p‖²`, the squared displacement of the amplitudes of the curve. -/
def ampDevSq (p v : ι → ℝ) (s : ℝ) : ℝ := ∑ i, (Real.sqrt (qs p v s i) - amp p i) ^ 2

/-- The open set of parameters on which the amplitude curve is positive. -/
def PosSet (p v : ι → ℝ) : Set ℝ := {s | ∀ i, 0 < amp p i + s * v i}

/-- The explicit form of `s ↦ KL p (q_s)` on `PosSet p v`. -/
def F1 (p v : ι → ℝ) (s : ℝ) : ℝ :=
  (∑ i, p i * Real.log (p i)) + Real.log (1 + s ^ 2 * sqNorm v)
    - 2 * ∑ i, p i * Real.log (amp p i + s * v i)

/-- The explicit form of `s ↦ ‖√(q_s) − √p‖²` on `PosSet p v`. -/
def F2 (_p v : ι → ℝ) (s : ℝ) : ℝ := 2 - 2 / Real.sqrt (1 + s ^ 2 * sqNorm v)

/-- The derivative of the explicit curve `F1`. -/
def KLderiv (p v : ι → ℝ) (s : ℝ) : ℝ :=
  2 * s * sqNorm v / (1 + s ^ 2 * sqNorm v) - 2 * ∑ i, p i * (v i / (amp p i + s * v i))

/-- The derivative of the explicit curve `F2`. -/
def Sderiv (_p v : ι → ℝ) (s : ℝ) : ℝ :=
  2 * s * sqNorm v / (Real.sqrt (1 + s ^ 2 * sqNorm v)) ^ 3

/-- `2 * Sderiv`, the derivative of `s ↦ 2 ‖√(q_s) − √p‖²`. -/
def Sderiv2 (_p v : ι → ℝ) (s : ℝ) : ℝ :=
  4 * s * sqNorm v / (Real.sqrt (1 + s ^ 2 * sqNorm v)) ^ 3

lemma isOpen_posSet (p v : ι → ℝ) : IsOpen (PosSet p v) := by
  have h : PosSet p v = ⋂ i, {s : ℝ | 0 < amp p i + s * v i} := by
    ext s; simp [PosSet]
  rw [h]
  exact isOpen_iInter_of_finite fun i =>
    isOpen_lt continuous_const (continuous_const.add (continuous_id.mul continuous_const))

omit [Fintype ι] in
lemma mem_posSet_zero {p v : ι → ℝ} (hp : ∀ i, 0 < p i) : (0 : ℝ) ∈ PosSet p v := by
  intro i
  have h := Real.sqrt_pos.mpr (hp i)
  simpa [amp] using h

lemma Z_eq {p v : ι → ℝ} (hp : ∀ i, 0 < p i) (hsum : ∑ i, p i = 1)
    (hv : ∑ i, amp p i * v i = 0) (s : ℝ) : Z p v s = 1 + s ^ 2 * sqNorm v := by
  have hterm : ∀ i, (amp p i + s * v i) ^ 2 = p i + 2 * s * (amp p i * v i) + s ^ 2 * (v i) ^ 2 := by
    intro i
    have h1 : (amp p i) ^ 2 = p i := Real.sq_sqrt (le_of_lt (hp i))
    rw [add_sq, h1]
    ring
  calc Z p v s = ∑ i, (p i + 2 * s * (amp p i * v i) + s ^ 2 * (v i) ^ 2) := by
        unfold Z
        exact Finset.sum_congr rfl fun i _ => hterm i
    _ = 1 + s ^ 2 * sqNorm v := by
        rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
          hsum, hv, sqNorm]
        ring

lemma Z_pos {p v : ι → ℝ} (hp : ∀ i, 0 < p i) (hsum : ∑ i, p i = 1)
    (hv : ∑ i, amp p i * v i = 0) (s : ℝ) : 0 < Z p v s := by
  rw [Z_eq hp hsum hv s]
  have := sqNorm_nonneg v
  positivity

lemma KL_eq_F1 {p v : ι → ℝ} (hp : ∀ i, 0 < p i) (hsum : ∑ i, p i = 1)
    (hv : ∑ i, amp p i * v i = 0) {s : ℝ} (hs : PosSet p v s) :
    KL p (qs p v s) = F1 p v s := by
  have hZpos : 0 < Z p v s := Z_pos hp hsum hv s
  have hZne : Z p v s ≠ 0 := ne_of_gt hZpos
  have hterm : ∀ i, p i * Real.log (p i / qs p v s i)
      = p i * Real.log (p i) + p i * Real.log (Z p v s)
        - 2 * (p i * Real.log (amp p i + s * v i)) := by
    intro i
    have hpos : 0 < amp p i + s * v i := hs i
    have hnum : (amp p i + s * v i) ^ 2 ≠ 0 := pow_ne_zero 2 (ne_of_gt hpos)
    have hpi : p i ≠ 0 := ne_of_gt (hp i)
    have hq : qs p v s i ≠ 0 := by
      unfold qs
      exact div_ne_zero hnum hZne
    have hqs : qs p v s i = (amp p i + s * v i) ^ 2 / Z p v s := rfl
    rw [Real.log_div hpi hq, hqs, Real.log_div hnum hZne, Real.log_pow]
    ring
  calc KL p (qs p v s)
      = ∑ i, (p i * Real.log (p i) + p i * Real.log (Z p v s)
          - 2 * (p i * Real.log (amp p i + s * v i))) := by
        unfold KL
        exact Finset.sum_congr rfl fun i _ => hterm i
    _ = F1 p v s := by
        unfold F1
        rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.sum_mul, ← Finset.mul_sum,
          hsum, Z_eq hp hsum hv s]
        ring

lemma ampDevSq_eq_F2 {p v : ι → ℝ} (hp : ∀ i, 0 < p i) (hsum : ∑ i, p i = 1)
    (hv : ∑ i, amp p i * v i = 0) {s : ℝ} (hs : PosSet p v s) :
    ampDevSq p v s = F2 p v s := by
  have hZpos : 0 < Z p v s := Z_pos hp hsum hv s
  set z := Real.sqrt (Z p v s) with hz
  have hz_pos : 0 < z := Real.sqrt_pos.mpr hZpos
  have hz_ne : z ≠ 0 := ne_of_gt hz_pos
  have hz_sq : z ^ 2 = Z p v s := Real.sq_sqrt (le_of_lt hZpos)
  have hsqr : ∀ i, Real.sqrt (qs p v s i) = (amp p i + s * v i) / z := by
    intro i
    have hpos : 0 < amp p i + s * v i := hs i
    have hsq : qs p v s i = ((amp p i + s * v i) / z) ^ 2 := by
      unfold qs
      rw [div_pow, hz_sq]
    rw [hsq, Real.sqrt_sq (div_nonneg (le_of_lt hpos) (le_of_lt hz_pos))]
  have h1 : ∑ i, ((amp p i + s * v i) / z) ^ 2 = 1 := by
    have hstep : ∀ i, ((amp p i + s * v i) / z) ^ 2 = (amp p i + s * v i) ^ 2 / z ^ 2 := by
      intro i; rw [div_pow]
    rw [Finset.sum_congr rfl fun i _ => hstep i, ← Finset.sum_div, hz_sq]
    exact div_self (ne_of_gt hZpos)
  have h2 : ∑ i, ((amp p i + s * v i) / z) * amp p i = 1 / z := by
    have hstep : ∀ i, ((amp p i + s * v i) / z) * amp p i
        = (1 / z) * ((amp p i + s * v i) * amp p i) := by
      intro i; ring
    rw [Finset.sum_congr rfl fun i _ => hstep i, ← Finset.mul_sum]
    have hinner : ∑ i, (amp p i + s * v i) * amp p i = 1 := by
      calc ∑ i, (amp p i + s * v i) * amp p i
          = ∑ i, ((amp p i) ^ 2 + s * (amp p i * v i)) := by
            apply Finset.sum_congr rfl
            intro i _
            ring
        _ = ∑ i, (amp p i) ^ 2 + s * ∑ i, amp p i * v i := by
            rw [Finset.sum_add_distrib, ← Finset.mul_sum]
        _ = 1 := by
            rw [hv]
            have hsq : ∑ i, (amp p i) ^ 2 = 1 := by
              rw [← hsum]
              exact Finset.sum_congr rfl fun i _ => Real.sq_sqrt (le_of_lt (hp i))
            rw [hsq]; ring
    rw [hinner]; ring
  have h3 : ∑ i, (amp p i) ^ 2 = 1 := by
    rw [← hsum]
    exact Finset.sum_congr rfl fun i _ => Real.sq_sqrt (le_of_lt (hp i))
  have hmain : ampDevSq p v s = ∑ i, (((amp p i + s * v i) / z) ^ 2
      - 2 * (((amp p i + s * v i) / z) * amp p i) + (amp p i) ^ 2) := by
    unfold ampDevSq
    apply Finset.sum_congr rfl
    intro i _
    rw [hsqr i]
    ring
  rw [hmain, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, h1, h2, h3]
  unfold F2
  rw [← Z_eq hp hsum hv s, ← hz]
  ring

lemma value_eq {p v : ι → ℝ} (hp : ∀ i, 0 < p i) (hsum : ∑ i, p i = 1)
    (hv : ∑ i, amp p i * v i = 0) : KL p (qs p v 0) = 2 * ampDevSq p v 0 := by
  have h0 := mem_posSet_zero (v := v) hp
  have hlogsum : ∑ i, p i * Real.log (amp p i) = (∑ i, p i * Real.log (p i)) / 2 := by
    have hterm : ∀ i, p i * Real.log (amp p i) = p i * Real.log (p i) / 2 := by
      intro i
      unfold amp
      rw [Real.log_sqrt (le_of_lt (hp i))]
      ring
    rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_div]
  have hone : (1 : ℝ) + 0 ^ 2 * sqNorm v = 1 := by norm_num
  have hz : ∀ i, p i * Real.log (amp p i + (0 : ℝ) * v i) = p i * Real.log (amp p i) := by
    intro i
    rw [zero_mul, add_zero]
  have hF1 : F1 p v 0 = 0 := by
    unfold F1
    rw [Finset.sum_congr rfl fun i _ => hz i, hlogsum, hone, Real.log_one]
    ring
  have hF2 : F2 p v 0 = 0 := by
    unfold F2
    rw [hone, Real.sqrt_one]
    norm_num
  rw [KL_eq_F1 hp hsum hv h0, ampDevSq_eq_F2 hp hsum hv h0, hF1, hF2]
  norm_num

/-- The identity function has derivative one (stated with `fun t => t` rather than `id`). -/
private lemma hasDerivAt_self (x : ℝ) : HasDerivAt (fun t : ℝ => t) 1 x :=
  hasDerivAt_id x

/-- Transport a `HasDerivAt` fact to a pointwise-equal function with an equal derivative. -/
private lemma hasDerivAt_congr_ptwise {f g : ℝ → ℝ} {f' g' x : ℝ} (h : HasDerivAt f f' x)
    (hfun : ∀ y, g y = f y) (hder : f' = g') : HasDerivAt g g' x :=
  (h.congr_of_eventuallyEq (Filter.EventuallyEq.of_eq (funext hfun))).congr_deriv hder

lemma hasDerivAt_affine (v : ι → ℝ) (c : ℝ) (s : ℝ) :
    HasDerivAt (fun t : ℝ => c + t ^ 2 * sqNorm v) (2 * s * sqNorm v) s := by
  have h1 : HasDerivAt (fun t : ℝ => t * t) (1 * s + s * 1) s :=
    (hasDerivAt_self s).mul (hasDerivAt_self s)
  have h2 : HasDerivAt (fun t : ℝ => t * t * sqNorm v) ((1 * s + s * 1) * sqNorm v) s :=
    h1.mul_const (sqNorm v)
  exact hasDerivAt_congr_ptwise (h2.const_add c) (fun t => by ring) (by ring)

lemma hasDerivAt_KLderiv (p v : ι → ℝ) (hp : ∀ i, 0 < p i) :
    HasDerivAt (KLderiv p v) (4 * sqNorm v) 0 := by
  have hg : HasDerivAt (fun t : ℝ => 1 + t ^ 2 * sqNorm v) 0 0 :=
    hasDerivAt_congr_ptwise (hasDerivAt_affine v 1 0) (fun t => by ring) (by ring)
  have hnum : HasDerivAt (fun t : ℝ => 2 * t * sqNorm v) (2 * sqNorm v) 0 :=
    hasDerivAt_congr_ptwise ((hasDerivAt_self 0).mul_const (2 * sqNorm v)) (fun t => by ring)
      (by ring)
  have hA : HasDerivAt (fun t : ℝ => 2 * t * sqNorm v / (1 + t ^ 2 * sqNorm v))
      (2 * sqNorm v) 0 := by
    have hgt : (1 + (0 : ℝ) ^ 2 * sqNorm v) ≠ 0 := by
      have := sqNorm_nonneg v
      positivity
    refine (hnum.div hg hgt).congr_deriv ?_
    simp
  have hterm : ∀ i, HasDerivAt (fun t : ℝ => p i * (v i / (amp p i + t * v i)))
      (-(v i ^ 2)) 0 := by
    intro i
    have hlin : HasDerivAt (fun t : ℝ => amp p i + t * v i) (v i) 0 :=
      hasDerivAt_congr_ptwise (((hasDerivAt_self 0).mul_const (v i)).const_add (amp p i))
        (fun t => by ring) (by ring)
    have hne : amp p i + (0 : ℝ) * v i ≠ 0 := by
      have h := Real.sqrt_pos.mpr (hp i)
      simpa [amp] using h.ne'
    refine (((hasDerivAt_const (0 : ℝ) (v i)).div hlin hne).const_mul (p i)).congr_deriv ?_
    have hsq : (amp p i) ^ 2 = p i := Real.sq_sqrt (le_of_lt (hp i))
    simp only [zero_mul, add_zero]
    rw [hsq, ← mul_div_assoc, mul_div_cancel_left₀ (0 - v i * v i) (ne_of_gt (hp i))]
    ring
  have hsum : HasDerivAt (fun t : ℝ => ∑ i, p i * (v i / (amp p i + t * v i)))
      (-(sqNorm v)) 0 := by
    have h := HasDerivAt.fun_sum (u := Finset.univ) fun i _ => hterm i
    refine hasDerivAt_congr_ptwise h (fun t => by simp) ?_
    simp [sqNorm, Finset.sum_neg_distrib]
  have hB : HasDerivAt (fun t : ℝ => 2 * ∑ i, p i * (v i / (amp p i + t * v i)))
      (2 * (-(sqNorm v))) 0 := hsum.const_mul 2
  exact hasDerivAt_congr_ptwise (hA.sub hB) (fun t => rfl) (by ring)

lemma hasDerivAt_Sderiv2 (p v : ι → ℝ) : HasDerivAt (Sderiv2 p v) (4 * sqNorm v) 0 := by
  have hg : HasDerivAt (fun t : ℝ => 1 + t ^ 2 * sqNorm v) 0 0 :=
    hasDerivAt_congr_ptwise (hasDerivAt_affine v 1 0) (fun t => by ring) (by ring)
  have hgpos : 0 < 1 + (0 : ℝ) ^ 2 * sqNorm v := by
    have := sqNorm_nonneg v
    positivity
  have hroot : HasDerivAt (fun t : ℝ => Real.sqrt (1 + t ^ 2 * sqNorm v)) 0 0 := by
    refine (hg.sqrt (ne_of_gt hgpos)).congr_deriv ?_
    simp
  have hcube : HasDerivAt (fun t : ℝ => (Real.sqrt (1 + t ^ 2 * sqNorm v)) ^ 3) 0 0 := by
    refine (hroot.pow 3).congr_deriv ?_
    simp
  have hcube_ne : (Real.sqrt (1 + (0 : ℝ) ^ 2 * sqNorm v)) ^ 3 ≠ 0 := by
    have h := Real.sqrt_pos.mpr hgpos
    positivity
  have hnum : HasDerivAt (fun t : ℝ => 4 * t * sqNorm v) (4 * sqNorm v) 0 :=
    hasDerivAt_congr_ptwise ((hasDerivAt_self 0).mul_const (4 * sqNorm v)) (fun t => by ring)
      (by ring)
  exact hasDerivAt_congr_ptwise (hnum.div hcube hcube_ne) (fun t => rfl) (by simp)

lemma hasDerivAt_F1 {p v : ι → ℝ} {s : ℝ} (hs : PosSet p v s) :
    HasDerivAt (F1 p v) (KLderiv p v s) s := by
  have hg := hasDerivAt_affine v 1 s
  have hgpos : 0 < 1 + s ^ 2 * sqNorm v := by
    have := sqNorm_nonneg v
    positivity
  have hlog : HasDerivAt (fun t : ℝ => Real.log (1 + t ^ 2 * sqNorm v))
      (2 * s * sqNorm v / (1 + s ^ 2 * sqNorm v)) s :=
    hg.log (ne_of_gt hgpos)
  have hsum : HasDerivAt (fun t : ℝ => ∑ i, p i * Real.log (amp p i + t * v i))
      (∑ i, p i * (v i / (amp p i + s * v i))) s := by
    have hterm : ∀ i, HasDerivAt (fun t : ℝ => p i * Real.log (amp p i + t * v i))
        (p i * (v i / (amp p i + s * v i))) s := by
      intro i
      have hlin : HasDerivAt (fun t : ℝ => amp p i + t * v i) (v i) s :=
        hasDerivAt_congr_ptwise (((hasDerivAt_self s).mul_const (v i)).const_add (amp p i))
          (fun t => by ring) (by ring)
      exact (hlin.log (ne_of_gt (hs i))).const_mul (p i)
    have h := HasDerivAt.fun_sum (u := Finset.univ) fun i _ => hterm i
    refine hasDerivAt_congr_ptwise h (fun t => by simp) ?_
    simp
  have hC : HasDerivAt (fun _ : ℝ => ∑ i, p i * Real.log (p i)) 0 s :=
    hasDerivAt_const s _
  exact hasDerivAt_congr_ptwise ((hC.add hlog).sub (hsum.const_mul 2))
    (fun t => rfl) (by unfold KLderiv; ring)

lemma hasDerivAt_F2 (p v : ι → ℝ) (s : ℝ) : HasDerivAt (F2 p v) (Sderiv p v s) s := by
  have hg := hasDerivAt_affine v 1 s
  have hgpos : 0 < 1 + s ^ 2 * sqNorm v := by
    have := sqNorm_nonneg v
    positivity
  have hroot : HasDerivAt (fun t : ℝ => Real.sqrt (1 + t ^ 2 * sqNorm v))
      ((2 * s * sqNorm v) / (2 * Real.sqrt (1 + s ^ 2 * sqNorm v))) s :=
    hg.sqrt (ne_of_gt hgpos)
  have hdiv := (hasDerivAt_const s 2).div hroot (ne_of_gt (Real.sqrt_pos.mpr hgpos))
  refine hasDerivAt_congr_ptwise (hdiv.const_sub 2) (fun t => rfl) ?_
  unfold Sderiv
  set r := Real.sqrt (1 + s ^ 2 * sqNorm v) with hr
  have hrne : r ≠ 0 := by rw [hr]; exact ne_of_gt (Real.sqrt_pos.mpr hgpos)
  field_simp
  ring

lemma hasDerivAt_KL {p v : ι → ℝ} (hp : ∀ i, 0 < p i) (hsum : ∑ i, p i = 1)
    (hv : ∑ i, amp p i * v i = 0) {s : ℝ} (hs : PosSet p v s) :
    HasDerivAt (fun t => KL p (qs p v t)) (KLderiv p v s) s :=
  (hasDerivAt_F1 hs).congr_of_eventuallyEq
    (Filter.eventually_of_mem ((isOpen_posSet p v).mem_nhds hs)
      fun t ht => KL_eq_F1 (s := t) hp hsum hv ht)

lemma hasDerivAt_ampDevSq {p v : ι → ℝ} (hp : ∀ i, 0 < p i) (hsum : ∑ i, p i = 1)
    (hv : ∑ i, amp p i * v i = 0) {s : ℝ} (hs : PosSet p v s) :
    HasDerivAt (fun t => 2 * ampDevSq p v t) (Sderiv2 p v s) s := by
  have heq : (fun t => 2 * ampDevSq p v t) =ᶠ[𝓝 s] (fun t => 2 * F2 p v t) := by
    filter_upwards [(isOpen_posSet p v).mem_nhds hs] with t ht
    rw [ampDevSq_eq_F2 hp hsum hv ht]
  have h : HasDerivAt (fun t => 2 * F2 p v t) (2 * Sderiv p v s) s :=
    (hasDerivAt_F2 p v s).const_mul 2
  have h' : HasDerivAt (fun t => 2 * F2 p v t) (Sderiv2 p v s) s :=
    h.congr_deriv (by unfold Sderiv Sderiv2; ring)
  exact h'.congr_of_eventuallyEq heq

lemma KLderiv_zero {p v : ι → ℝ} (hp : ∀ i, 0 < p i) (hv : ∑ i, amp p i * v i = 0) :
    KLderiv p v 0 = 0 := by
  have hterm : ∀ i, p i * (v i / (amp p i + (0 : ℝ) * v i)) = amp p i * v i := by
    intro i
    rw [zero_mul, add_zero]
    have hsq : (amp p i) ^ 2 = p i := Real.sq_sqrt (le_of_lt (hp i))
    have hne : amp p i ≠ 0 := by
      unfold amp
      exact ne_of_gt (Real.sqrt_pos.mpr (hp i))
    field_simp
    rw [hsq]
    ring
  have hsum : ∑ i, p i * (v i / (amp p i + (0 : ℝ) * v i)) = 0 := by
    rw [Finset.sum_congr rfl fun i _ => hterm i, hv]
  unfold KLderiv
  rw [hsum]
  norm_num

lemma Sderiv2_zero (p v : ι → ℝ) : Sderiv2 p v 0 = 0 := by
  unfold Sderiv2
  simp

/-- **Clause 2.** Along the amplitude curve `q_s` through a full-support distribution `p` in a
direction `v` that is orthogonal to `√p`, the curves `s ↦ KL p (q_s)` and `s ↦ 2 ‖√(q_s) − √p‖²`
agree in value and in first and second derivative at `s = 0`; both second derivatives are
`4 ∑ v i ^ 2`. -/
theorem kl_second_order {p v : ι → ℝ} (hp : ∀ i, 0 < p i) (hsum : ∑ i, p i = 1)
    (hv : ∑ i, amp p i * v i = 0) :
    KL p (qs p v 0) = 2 * ampDevSq p v 0 ∧
    HasDerivAt (fun s : ℝ => KL p (qs p v s)) 0 0 ∧
    HasDerivAt (fun s : ℝ => 2 * ampDevSq p v s) 0 0 ∧
    HasDerivAt (deriv (fun s : ℝ => KL p (qs p v s))) (4 * sqNorm v) 0 ∧
    HasDerivAt (deriv (fun s : ℝ => 2 * ampDevSq p v s)) (4 * sqNorm v) 0 := by
  have h0 := mem_posSet_zero (v := v) hp
  have hmem : PosSet p v ∈ 𝓝 (0 : ℝ) := (isOpen_posSet p v).mem_nhds h0
  refine ⟨value_eq hp hsum hv, ?_, ?_, ?_, ?_⟩
  · exact (hasDerivAt_KL hp hsum hv h0).congr_deriv (KLderiv_zero hp hv)
  · exact (hasDerivAt_ampDevSq hp hsum hv h0).congr_deriv (Sderiv2_zero p v)
  · have heq : deriv (fun s : ℝ => KL p (qs p v s)) =ᶠ[𝓝 (0 : ℝ)] KLderiv p v := by
      filter_upwards [hmem] with s hs
      exact (hasDerivAt_KL hp hsum hv hs).deriv
    exact heq.hasDerivAt_iff.mpr (hasDerivAt_KLderiv p v hp)
  · have heq : deriv (fun s : ℝ => 2 * ampDevSq p v s) =ᶠ[𝓝 (0 : ℝ)] Sderiv2 p v := by
      filter_upwards [hmem] with s hs
      exact (hasDerivAt_ampDevSq hp hsum hv hs).deriv
    exact heq.hasDerivAt_iff.mpr (hasDerivAt_Sderiv2 p v)

/-- **The unit verdict.** The Hellinger-type lower bound (`kl_ge_amplitude_sq`), the second-order
agreement of `s ↦ KL p (q_s)` with `s ↦ 2 ‖√(q_s) − √p‖²` at the base distribution
(`kl_second_order`), the logistic/softmax identity and its amplitude form all hold. -/
theorem verdict_l6 :
    (∀ {p q : ι → ℝ}, (∀ i, 0 < p i) → (∀ i, 0 < q i) → (∑ i, p i = 1) → (∑ i, q i = 1) →
        ampDistSq p q ≤ KL p q) ∧
    (∀ {p v : ι → ℝ}, (∀ i, 0 < p i) → (∑ i, p i = 1) → (∑ i, amp p i * v i = 0) →
        KL p (qs p v 0) = 2 * ampDevSq p v 0 ∧
        HasDerivAt (fun s : ℝ => KL p (qs p v s)) 0 0 ∧
        HasDerivAt (fun s : ℝ => 2 * ampDevSq p v s) 0 0 ∧
        HasDerivAt (deriv (fun s : ℝ => KL p (qs p v s))) (4 * sqNorm v) 0 ∧
        HasDerivAt (deriv (fun s : ℝ => 2 * ampDevSq p v s)) (4 * sqNorm v) 0) ∧
    (∀ x : ℝ, 1 / (1 + Real.exp (-x)) = Real.exp x / (Real.exp x + Real.exp 0)) ∧
    (∀ x : ℝ, (Real.sqrt (logistic x)) ^ 2 + (Real.sqrt (logistic (-x))) ^ 2 = 1) :=
  ⟨fun hp hq hp1 hq1 => kl_ge_amplitude_sq hp hq hp1 hq1,
    fun hp hsum hv => kl_second_order hp hsum hv,
    logistic_eq_softmax_two,
    logistic_amplitude_add⟩

end

end Quantum.AmplitudeKL
