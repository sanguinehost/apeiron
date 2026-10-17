/-
Taylor-series refinement of the extracted `sin_cos_series` of `src/numeric/fixed_trig.rs`
.

The body is the series stage of the `Q32.32` trigonometric evaluation: twelve rounded products
`q_mul`, thirteen rounded divisions `div_fac` by exact factorials, and thirteen `i128` sums, through
`r^13/13!` (sin) and `r^12/12!` (cos). This file proves the absolute budget of that stage against
`Real.sin`/`Real.cos` on `|toReal r| ≤ 1`:

  `|toReal sc.sin - sin (toReal r)| ≤ 2^-23`,  `|toReal sc.cos - cos (toReal r)| ≤ 2^-23`.

The two halves of the budget:

* **analytic remainder** (`sin_remainder`, `cos_remainder`): the partial sums through `r^13/13!` and
  `r^12/12!` are the real and imaginary parts of the degree-13 partial exponential at `r·i`, so
  `Complex.exp_bound` bounds both remainders by `|r|^14 · 15/(14!·14)`, which is below `trig_mu`.
* **accumulated rounding** (`hb*` in `sin_cos_series_spec`): `q_mul_step`/`div_fac_step` charge
  `(k+3)·trig_mu` and `(k+1)·trig_mu` per operation from the previous operation's budget; the six
  sin terms reach `69·trig_mu` and the six cos terms `57·trig_mu`, and the truncation above is
  charged once more, giving `70·trig_mu < 2^-23` and `58·trig_mu < 2^-23`.

The measured error of the body is `1.1e-9` relative; the proved budget is `1.2e-7` absolute, so
every step below is deliberately coarse.

## Scope

This module is the series stage only. The range reduction, the fold and the quadrant assembly of
`fixed_sin_cos` are not re-established here; the previous content of this file proved the degree-7
Taylor/Horner form of the *removed* `fixed_trig.taylor_sin_cos`/`fold1`/`fold2` bodies, which the
extraction no longer contains.
-/
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Real.Pi.Bounds
import Numeric.Scale
import Numeric.Rope.Part2

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Representation helpers -/

/-- A Q32.32 `i64` whose real value is at most `2` in magnitude has `|val| ≤ 2^40`: the magnitude
bound the `q_mul`/`div_fac` specifications require of their operands. -/
lemma val_abs_le_of_abs_toReal_le (u : I64) (h : |toReal u| ≤ 2) : |u.val| ≤ 2 ^ 40 := by
  have hq : q32 = (2 : ℝ) ^ 32 := by unfold q32; norm_num
  have h2 : (0 : ℝ) < (2 : ℝ) ^ 32 := by positivity
  rw [toReal, hq, abs_div, abs_of_pos h2, div_le_iff₀ h2] at h
  have h' : |(u.val : ℝ)| ≤ (2 : ℝ) ^ 33 := by nlinarith
  have hz : |u.val| ≤ (2 : ℤ) ^ 33 := by exact_mod_cast h'
  have hle : (2 : ℤ) ^ 33 ≤ (2 : ℤ) ^ 40 := by norm_num
  omega

/-- An operand within `e ≤ 1` of `R^n` with `|R| ≤ 1` has real value at most `2` in magnitude. -/
lemma abs_toReal_le_two (u : I64) (R : ℝ) (n : ℕ) (e : ℝ)
    (hR : |R| ≤ 1) (he : e ≤ 1) (h : |toReal u - R ^ n| ≤ e) : |toReal u| ≤ 2 := by
  have hpow : |R ^ n| ≤ 1 := by
    rw [abs_pow]
    exact pow_le_one₀ (abs_nonneg R) hR
  have h1 : |toReal u| ≤ e + 1 := by
    calc |toReal u| = |(toReal u - R ^ n) + R ^ n| := by ring_nf
      _ ≤ |toReal u - R ^ n| + |R ^ n| := abs_add_le _ _
      _ ≤ e + 1 := by linarith
  linarith

/-- The extracted scale `ONE` denotes `1`. -/
lemma toReal_ONE : toReal (4294967296#i64) = 1 := by
  unfold toReal
  have h : (4294967296#i64).val = 4294967296 := by norm_num
  rw [h]
  unfold q32
  norm_num

/-- `|a + b + c + d + e + f| ≤ |a| + |b| + |c| + |d| + |e| + |f|`. -/
lemma abs_six_le (a b c d e f : ℝ) :
    |a + b + c + d + e + f| ≤ |a| + |b| + |c| + |d| + |e| + |f| := by
  have h1 := abs_add_le (a + b + c + d + e) f
  have h2 := abs_add_le (a + b + c + d) e
  have h3 := abs_add_le (a + b + c) d
  have h4 := abs_add_le (a + b) c
  have h5 := abs_add_le a b
  linarith [abs_nonneg (a + b + c + d), abs_nonneg (a + b + c), abs_nonneg (a + b),
    abs_nonneg a, abs_nonneg b]

/-! ## Per-operation budgets

One rounded product multiplies the previous error by `1 + trig_mu` and adds two roundings; one
rounded division divides the previous error by `m!` and adds one rounding. Both close with the
previous budget `k·trig_mu` at `k ≤ 30`. -/

/-- One `q_mul(x, v)` of a value within `k·trig_mu` of `R^m` by a value within `trig_mu` of `R^2`
lands within `(k+3)·trig_mu` of `R^(m+2)`. -/
lemma q_mul_step (x v : I64) (R : ℝ) (m k : ℕ)
    (hR : |R| ≤ 1) (hk : k ≤ 30)
    (hx : |toReal x - R ^ m| ≤ k * trig_mu) (hv : |toReal v - R ^ 2| ≤ trig_mu) :
    ∃ w : I64, fixed_trig.q_mul x v = ok (some w) ∧
      |toReal w - R ^ (m + 2)| ≤ (k + 3) * trig_mu := by
  have he1 : (k : ℝ) * trig_mu ≤ 1 := by
    have hk' : (k : ℝ) ≤ 30 := by exact_mod_cast hk
    have h30 : (30 : ℝ) * trig_mu ≤ 1 := by unfold trig_mu; norm_num
    nlinarith [trig_mu_pos]
  have he0 : (0 : ℝ) ≤ (k : ℝ) * trig_mu :=
    mul_nonneg (Nat.cast_nonneg k) (le_of_lt trig_mu_pos)
  have hxa : |toReal x| ≤ 2 := abs_toReal_le_two x R m _ hR he1 hx
  have hva : |toReal v| ≤ 2 :=
    abs_toReal_le_two v R 2 trig_mu hR (by unfold trig_mu; norm_num) hv
  obtain ⟨w, hw, hwB⟩ := q_mul_spec x v (val_abs_le_of_abs_toReal_le x hxa)
    (val_abs_le_of_abs_toReal_le v hva)
  refine ⟨w, hw, ?_⟩
  have hR2 : |R ^ 2| ≤ 1 := by
    rw [abs_pow]; exact pow_le_one₀ (abs_nonneg R) hR
  have htv : |toReal v| ≤ 1 + trig_mu := by
    have h1 : |toReal v| ≤ trig_mu + 1 := by
      calc |toReal v| = |(toReal v - R ^ 2) + R ^ 2| := by ring_nf
        _ ≤ |toReal v - R ^ 2| + |R ^ 2| := abs_add_le _ _
        _ ≤ trig_mu + 1 := by linarith
    linarith
  have hRm : |R ^ m| ≤ 1 := by
    rw [abs_pow]; exact pow_le_one₀ (abs_nonneg R) hR
  have hkey : toReal x * toReal v - R ^ (m + 2)
      = (toReal x - R ^ m) * toReal v + R ^ m * (toReal v - R ^ 2) := by
    rw [pow_add]; ring
  have hb1 : |(toReal x - R ^ m) * toReal v| ≤ (k : ℝ) * trig_mu * (1 + trig_mu) := by
    rw [abs_mul]
    exact mul_le_mul hx htv (abs_nonneg _) he0
  have hb2 : |R ^ m * (toReal v - R ^ 2)| ≤ 1 * trig_mu := by
    rw [abs_mul]
    exact mul_le_mul hRm hv (abs_nonneg _) zero_le_one
  have hstep : |toReal x * toReal v - R ^ (m + 2)|
      ≤ (k : ℝ) * trig_mu + trig_mu + (k : ℝ) * trig_mu * trig_mu := by
    rw [hkey]
    have := abs_add_le ((toReal x - R ^ m) * toReal v) (R ^ m * (toReal v - R ^ 2))
    nlinarith
  have hsmall : (k : ℝ) * trig_mu * trig_mu ≤ trig_mu := by
    have : (k : ℝ) * trig_mu ≤ 1 := he1
    nlinarith [trig_mu_pos]
  calc |toReal w - R ^ (m + 2)|
      ≤ |toReal w - toReal x * toReal v| + |toReal x * toReal v - R ^ (m + 2)| :=
        abs_sub_le _ _ _
    _ ≤ trig_mu + ((k : ℝ) * trig_mu + trig_mu + (k : ℝ) * trig_mu * trig_mu) := by
        linarith
    _ ≤ (k + 3) * trig_mu := by
        have htot : (k : ℝ) * trig_mu + trig_mu + (k : ℝ) * trig_mu * trig_mu + trig_mu
            ≤ (k : ℝ) * trig_mu + 3 * trig_mu := by linarith [hsmall]
        calc trig_mu + ((k : ℝ) * trig_mu + trig_mu + (k : ℝ) * trig_mu * trig_mu)
            = (k : ℝ) * trig_mu + trig_mu + (k : ℝ) * trig_mu * trig_mu + trig_mu := by ring
          _ ≤ (k : ℝ) * trig_mu + 3 * trig_mu := htot
          _ = (k + 3) * trig_mu := by ring

/-- One `div_fac(p, m!)` of a value within `k·trig_mu` of `R^m` lands within `(k+1)·trig_mu` of
`R^m/m!`. -/
lemma div_fac_step (p : I64) (R : ℝ) (m k : ℕ) (f : I64) (hf : 0 < f.val)
    (hf2 : f.val ≤ 2 ^ 62) (hfv : (f.val : ℝ) = (m.factorial : ℝ))
    (hR : |R| ≤ 1) (hk : k ≤ 30) (hp : |toReal p - R ^ m| ≤ k * trig_mu) :
    ∃ w : I64, fixed_trig.div_fac p f = ok (some w) ∧
      |toReal w - R ^ m / (m.factorial : ℝ)| ≤ (k + 1) * trig_mu := by
  have he1 : (k : ℝ) * trig_mu ≤ 1 := by
    have hk' : (k : ℝ) ≤ 30 := by exact_mod_cast hk
    have h30 : (30 : ℝ) * trig_mu ≤ 1 := by unfold trig_mu; norm_num
    nlinarith [trig_mu_pos]
  have hpa : |toReal p| ≤ 2 := by
    have h1 : |toReal p| ≤ (k : ℝ) * trig_mu + 1 := by
      calc |toReal p| = |(toReal p - R ^ m) + R ^ m| := by ring_nf
        _ ≤ |toReal p - R ^ m| + |R ^ m| := abs_add_le _ _
        _ ≤ (k : ℝ) * trig_mu + 1 := by
            have hRm : |R ^ m| ≤ 1 := by
              rw [abs_pow]; exact pow_le_one₀ (abs_nonneg R) hR
            linarith
    linarith
  obtain ⟨w, hw, hwB⟩ := div_fac_spec p f hf (by
    have := val_abs_le_of_abs_toReal_le p hpa
    omega) hf2
  refine ⟨w, hw, ?_⟩
  have hkey : toReal p / (f.val : ℝ) - R ^ m / (m.factorial : ℝ)
      = (toReal p - R ^ m) / (m.factorial : ℝ) := by
    rw [← hfv]
    field_simp
  have hstep : |toReal p / (f.val : ℝ) - R ^ m / (m.factorial : ℝ)| ≤ (k : ℝ) * trig_mu := by
    have hfac1 : (0 : ℝ) < (m.factorial : ℝ) := by positivity
    rw [hkey, abs_div, abs_of_pos hfac1, div_le_iff₀ hfac1]
    have hfac : (1 : ℝ) ≤ (m.factorial : ℝ) := by
      exact_mod_cast Nat.one_le_iff_ne_zero.mpr (Nat.factorial_ne_zero m)
    nlinarith [abs_nonneg (toReal p - R ^ m)]
  calc |toReal w - R ^ m / (m.factorial : ℝ)|
      ≤ |toReal w - toReal p / (f.val : ℝ)| +
        |toReal p / (f.val : ℝ) - R ^ m / (m.factorial : ℝ)| := abs_sub_le _ _ _
    _ ≤ trig_mu + (k : ℝ) * trig_mu := by linarith
    _ = (k + 1) * trig_mu := by ring

/-! ## The Taylor partial sums

`sinSer` and `cosSer` are the exact partial sums the body computes in fixed point: the odd terms
through `s^13/13!` and the even terms through `s^12/12!`. -/

/-- `s - s^3/3! + s^5/5! - s^7/7! + s^9/9! - s^11/11! + s^13/13!`. -/
def sinSer (s : ℝ) : ℝ :=
  s - s ^ 3 / 6 + s ^ 5 / 120 - s ^ 7 / 5040 + s ^ 9 / 362880 - s ^ 11 / 39916800 +
    s ^ 13 / 6227020800

/-- `1 - s^2/2! + s^4/4! - s^6/6! + s^8/8! - s^10/10! + s^12/12!`. -/
def cosSer (s : ℝ) : ℝ :=
  1 - s ^ 2 / 2 + s ^ 4 / 24 - s ^ 6 / 720 + s ^ 8 / 40320 - s ^ 10 / 3628800 +
    s ^ 12 / 479001600

/-- The degree-13 partial exponential at `s·i` has `sinSer s` as its imaginary part. -/
lemma sinSer_eq_im (s : ℝ) :
    ((∑ m ∈ Finset.range 14, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ)).im) = sinSer s := by
  unfold sinSer
  norm_num [Finset.sum_range_succ, Complex.I_mul_I]
  simp only [mul_pow, ← Complex.ofReal_pow, Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im,
    zero_mul, add_zero]
  norm_num [pow_succ, pow_two, Complex.I_mul_I]
  ring

/-- The same partial exponential has `cosSer s` as its real part. -/
lemma cosSer_eq_re (s : ℝ) :
    ((∑ m ∈ Finset.range 14, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ)).re) = cosSer s := by
  unfold cosSer
  norm_num [Finset.sum_range_succ, Complex.I_mul_I]
  simp only [mul_pow, ← Complex.ofReal_pow, Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im,
    zero_mul]
  norm_num [pow_succ, pow_two, Complex.I_mul_I]
  ring

/-- `Complex.exp_bound` at degree 14, in the form both remainders use. -/
lemma exp_partial_bound (s : ℝ) (hs : |s| ≤ 1) :
    ‖Complex.exp ((s : ℂ) * Complex.I) -
        ∑ m ∈ Finset.range 14, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ)‖ ≤
      |s| ^ 14 * (15 * (((Nat.factorial 14 : ℝ)) * 14)⁻¹) := by
  have hzn : ‖(s : ℂ) * Complex.I‖ = |s| := by
    rw [norm_mul, Complex.norm_I, mul_one, Complex.norm_def, Complex.normSq_ofReal, ← sq,
      Real.sqrt_sq_eq_abs]
  have hz : ‖(s : ℂ) * Complex.I‖ ≤ 1 := by rw [hzn]; exact hs
  have hb := Complex.exp_bound hz (n := 14) (by norm_num)
  convert hb using 2
  · rw [hzn]
  · norm_num

/-- The truncation error of the `sin` series on `|s| ≤ 1` is below one per-operation budget. -/
theorem sin_remainder (s : ℝ) (hs : |s| ≤ 1) : |sinSer s - Real.sin s| ≤ trig_mu := by
  have hb := exp_partial_bound s hs
  calc |sinSer s - Real.sin s|
      = |((∑ m ∈ Finset.range 14, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ)) -
            Complex.exp ((s : ℂ) * Complex.I)).im| := by
          rw [← Complex.exp_ofReal_mul_I_im, ← sinSer_eq_im]
          simp only [← Complex.sub_im]
    _ ≤ ‖(∑ m ∈ Finset.range 14, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ)) -
            Complex.exp ((s : ℂ) * Complex.I)‖ := Complex.abs_im_le_norm _
    _ = ‖Complex.exp ((s : ℂ) * Complex.I) -
            (∑ m ∈ Finset.range 14, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ))‖ :=
          norm_sub_rev _ _
    _ ≤ |s| ^ 14 * (15 * (((Nat.factorial 14 : ℝ)) * 14)⁻¹) := hb
    _ ≤ 1 * (15 * (((Nat.factorial 14 : ℝ)) * 14)⁻¹) := by
          have h1 : |s| ^ 14 ≤ 1 := pow_le_one₀ (abs_nonneg s) hs
          have h2 : (0 : ℝ) ≤ 15 * (((Nat.factorial 14 : ℝ)) * 14)⁻¹ := by positivity
          exact mul_le_mul_of_nonneg_right h1 h2
    _ ≤ trig_mu := by unfold trig_mu; norm_num

/-- The truncation error of the `cos` series on `|s| ≤ 1` is below one per-operation budget. -/
theorem cos_remainder (s : ℝ) (hs : |s| ≤ 1) : |cosSer s - Real.cos s| ≤ trig_mu := by
  have hb := exp_partial_bound s hs
  calc |cosSer s - Real.cos s|
      = |((∑ m ∈ Finset.range 14, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ)) -
            Complex.exp ((s : ℂ) * Complex.I)).re| := by
          rw [← Complex.exp_ofReal_mul_I_re, ← cosSer_eq_re]
          simp only [← Complex.sub_re]
    _ ≤ ‖(∑ m ∈ Finset.range 14, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ)) -
            Complex.exp ((s : ℂ) * Complex.I)‖ := Complex.abs_re_le_norm _
    _ = ‖Complex.exp ((s : ℂ) * Complex.I) -
            (∑ m ∈ Finset.range 14, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ))‖ :=
          norm_sub_rev _ _
    _ ≤ |s| ^ 14 * (15 * (((Nat.factorial 14 : ℝ)) * 14)⁻¹) := hb
    _ ≤ 1 * (15 * (((Nat.factorial 14 : ℝ)) * 14)⁻¹) := by
          have h1 : |s| ^ 14 ≤ 1 := pow_le_one₀ (abs_nonneg s) hs
          have h2 : (0 : ℝ) ≤ 15 * (((Nat.factorial 14 : ℝ)) * 14)⁻¹ := by positivity
          exact mul_le_mul_of_nonneg_right h1 h2
    _ ≤ trig_mu := by unfold trig_mu; norm_num

/-! ## The extracted body -/

/-- The hypothesis `|toReal r| ≤ 1` of `sin_cos_series_spec` is inhabitable: `r = 0` satisfies it. -/
theorem sin_cos_series_domain_witness : ∃ r : I64, |toReal r| ≤ 1 :=
  ⟨0#i64, by rw [toReal]; norm_num [q32]⟩

/-- The series stage of the `Q32.32` trigonometric evaluation, on `|toReal r| ≤ 1`: the extracted
body succeeds and both outputs are within `2^-23` of the real functions.

The budget is `70·trig_mu` for `sin` (69 accumulated + 1 truncation) and `58·trig_mu` for `cos`
(57 + 1), both below `2^-23`. -/
theorem sin_cos_series_spec (r : I64) (hr : |toReal r| ≤ 1) :
    ∃ sc : fixed_trig.SinCos,
      fixed_trig.sin_cos_series r = ok (core.result.Result.Ok sc) ∧
        |toReal sc.sin - Real.sin (toReal r)| ≤ (2 : ℝ) ^ (-23 : ℤ) ∧
        |toReal sc.cos - Real.cos (toReal r)| ≤ (2 : ℝ) ^ (-23 : ℤ) := by
  set R : ℝ := toReal r with hRdef
  have hR : |R| ≤ 1 := hr
  have hrv : |r.val| ≤ 2 ^ 40 := val_abs_le_of_abs_toReal_le r (by linarith)
  have hmain : fixed_trig.sin_cos_series r ⦃ z => ∃ sc : fixed_trig.SinCos,
      z = core.result.Result.Ok sc ∧
      |toReal sc.sin - Real.sin R| ≤ (2 : ℝ) ^ (-23 : ℤ) ∧
      |toReal sc.cos - Real.cos R| ≤ (2 : ℝ) ^ (-23 : ℤ) ⦄ := by
    unfold fixed_trig.sin_cos_series
    simp only [lift, bind_tc_ok]
    rw [fixed_trig_ONE_eq]
    simp only [bind_tc_ok]
    -- r^2
    obtain ⟨v, hq, hqB⟩ := q_mul_spec r r hrv hrv
    rw [hq]
    simp only [bind_tc_ok]
    have hV : |toReal v - R ^ 2| ≤ trig_mu := by
      rw [← hRdef] at hqB
      simpa [pow_two] using hqB
    -- r^3
    obtain ⟨v1, hq1, hq1B⟩ := q_mul_step r v R 1 0 hR (by norm_num) (by
      rw [pow_one, ← hRdef]; simp) hV
    rw [hq1]
    simp only [bind_tc_ok]
    have hb3 : |toReal v1 - R ^ 3| ≤ 3 * trig_mu := by norm_num at hq1B ⊢; exact hq1B
    -- r^3/3!
    obtain ⟨v2, hq2, hq2B⟩ := div_fac_step v1 R 3 3 (6#i64) (by norm_num) (by norm_num)
      (by norm_num) hR (by norm_num) hb3
    rw [hq2]
    simp only [bind_tc_ok]
    have hb2 : |toReal v2 - R ^ 3 / 6| ≤ 4 * trig_mu := by norm_num at hq2B ⊢; exact hq2B
    step as ⟨s1, hs1⟩
    have hs1R : (s1.val : ℝ) / q32 = R - toReal v2 := by
      rw [hs1]
      simp only [core.convert.num.FromI128I64.from_val_eq, toReal]
      push_cast
      rw [hRdef]
      simp only [toReal]
      ring
    -- r^2/2!
    obtain ⟨v3, hq3, hq3B⟩ := div_fac_step v R 2 1 (2#i64) (by norm_num) (by norm_num)
      (by norm_num) hR (by norm_num) (by simpa using hV)
    rw [hq3]
    simp only [bind_tc_ok]
    have hb4 : |toReal v3 - R ^ 2 / 2| ≤ 2 * trig_mu := by norm_num at hq3B ⊢; exact hq3B
    step as ⟨c1, hc1⟩
    have hc1R' : (c1.val : ℝ) / q32 = toReal (4294967296#i64) - toReal v3 := by
      rw [hc1]
      simp only [core.convert.num.FromI128I64.from_val_eq, toReal]
      push_cast
      ring
    have hc1R : (c1.val : ℝ) / q32 = 1 - toReal v3 := by rw [hc1R', toReal_ONE]
    -- r^4
    obtain ⟨v4, hq4, hq4B⟩ := q_mul_step v v R 2 1 hR (by norm_num) (by simpa using hV) hV
    rw [hq4]
    simp only [bind_tc_ok]
    have hb5 : |toReal v4 - R ^ 4| ≤ 4 * trig_mu := by norm_num at hq4B ⊢; exact hq4B
    -- r^5
    obtain ⟨v5, hq5, hq5B⟩ := q_mul_step v1 v R 3 3 hR (by norm_num) hb3 hV
    rw [hq5]
    simp only [bind_tc_ok]
    have hb6 : |toReal v5 - R ^ 5| ≤ 6 * trig_mu := by norm_num at hq5B ⊢; exact hq5B
    -- r^5/5!
    obtain ⟨v6, hq6, hq6B⟩ := div_fac_step v5 R 5 6 (120#i64) (by norm_num) (by norm_num)
      (by norm_num) hR (by norm_num) hb6
    rw [hq6]
    simp only [bind_tc_ok]
    have hb7 : |toReal v6 - R ^ 5 / 120| ≤ 7 * trig_mu := by norm_num at hq6B ⊢; exact hq6B
    step as ⟨s2, hs2⟩
    have hs2R : (s2.val : ℝ) / q32 = (s1.val : ℝ) / q32 + toReal v6 := by
      rw [hs2]
      simp only [core.convert.num.FromI128I64.from_val_eq, toReal]
      push_cast
      ring
    -- r^4/4!
    obtain ⟨v7, hq7, hq7B⟩ := div_fac_step v4 R 4 4 (24#i64) (by norm_num) (by norm_num)
      (by norm_num) hR (by norm_num) hb5
    rw [hq7]
    simp only [bind_tc_ok]
    have hb8 : |toReal v7 - R ^ 4 / 24| ≤ 5 * trig_mu := by norm_num at hq7B ⊢; exact hq7B
    step as ⟨c2, hc2⟩
    have hc2R : (c2.val : ℝ) / q32 = (c1.val : ℝ) / q32 + toReal v7 := by
      rw [hc2]
      simp only [core.convert.num.FromI128I64.from_val_eq, toReal]
      push_cast
      ring
    -- r^6
    obtain ⟨v8, hq8, hq8B⟩ := q_mul_step v4 v R 4 4 hR (by norm_num) hb5 hV
    rw [hq8]
    simp only [bind_tc_ok]
    have hb9 : |toReal v8 - R ^ 6| ≤ 7 * trig_mu := by norm_num at hq8B ⊢; exact hq8B
    -- r^7
    obtain ⟨v9, hq9, hq9B⟩ := q_mul_step v5 v R 5 6 hR (by norm_num) hb6 hV
    rw [hq9]
    simp only [bind_tc_ok]
    have hb10 : |toReal v9 - R ^ 7| ≤ 9 * trig_mu := by norm_num at hq9B ⊢; exact hq9B
    -- r^7/7!
    obtain ⟨v10, hq10, hq10B⟩ := div_fac_step v9 R 7 9 (5040#i64) (by norm_num) (by norm_num)
      (by norm_num) hR (by norm_num) hb10
    rw [hq10]
    simp only [bind_tc_ok]
    have hb11 : |toReal v10 - R ^ 7 / 5040| ≤ 10 * trig_mu := by
      norm_num at hq10B ⊢; exact hq10B
    step as ⟨s3, hs3⟩
    have hs3R : (s3.val : ℝ) / q32 = (s2.val : ℝ) / q32 - toReal v10 := by
      rw [hs3]
      simp only [core.convert.num.FromI128I64.from_val_eq, toReal]
      push_cast
      ring
    -- r^6/6!
    obtain ⟨v11, hq11, hq11B⟩ := div_fac_step v8 R 6 7 (720#i64) (by norm_num) (by norm_num)
      (by norm_num) hR (by norm_num) hb9
    rw [hq11]
    simp only [bind_tc_ok]
    have hb12 : |toReal v11 - R ^ 6 / 720| ≤ 8 * trig_mu := by norm_num at hq11B ⊢; exact hq11B
    step as ⟨c3, hc3⟩
    have hc3R : (c3.val : ℝ) / q32 = (c2.val : ℝ) / q32 - toReal v11 := by
      rw [hc3]
      simp only [core.convert.num.FromI128I64.from_val_eq, toReal]
      push_cast
      ring
    -- r^8
    obtain ⟨v12, hq12, hq12B⟩ := q_mul_step v8 v R 6 7 hR (by norm_num) hb9 hV
    rw [hq12]
    simp only [bind_tc_ok]
    have hb13 : |toReal v12 - R ^ 8| ≤ 10 * trig_mu := by norm_num at hq12B ⊢; exact hq12B
    -- r^9
    obtain ⟨v13, hq13, hq13B⟩ := q_mul_step v9 v R 7 9 hR (by norm_num) hb10 hV
    rw [hq13]
    simp only [bind_tc_ok]
    have hb14 : |toReal v13 - R ^ 9| ≤ 12 * trig_mu := by norm_num at hq13B ⊢; exact hq13B
    -- r^9/9!
    obtain ⟨v14, hq14, hq14B⟩ := div_fac_step v13 R 9 12 (362880#i64) (by norm_num)
      (by norm_num) (by norm_num) hR (by norm_num) hb14
    rw [hq14]
    simp only [bind_tc_ok]
    have hb15 : |toReal v14 - R ^ 9 / 362880| ≤ 13 * trig_mu := by
      norm_num at hq14B ⊢; exact hq14B
    step as ⟨s4, hs4⟩
    have hs4R : (s4.val : ℝ) / q32 = (s3.val : ℝ) / q32 + toReal v14 := by
      rw [hs4]
      simp only [core.convert.num.FromI128I64.from_val_eq, toReal]
      push_cast
      ring
    -- r^8/8!
    obtain ⟨v15, hq15, hq15B⟩ := div_fac_step v12 R 8 10 (40320#i64) (by norm_num)
      (by norm_num) (by norm_num) hR (by norm_num) hb13
    rw [hq15]
    simp only [bind_tc_ok]
    have hb16 : |toReal v15 - R ^ 8 / 40320| ≤ 11 * trig_mu := by
      norm_num at hq15B ⊢; exact hq15B
    step as ⟨c4, hc4⟩
    have hc4R : (c4.val : ℝ) / q32 = (c3.val : ℝ) / q32 + toReal v15 := by
      rw [hc4]
      simp only [core.convert.num.FromI128I64.from_val_eq, toReal]
      push_cast
      ring
    -- r^10
    obtain ⟨v16, hq16, hq16B⟩ := q_mul_step v12 v R 8 10 hR (by norm_num) hb13 hV
    rw [hq16]
    simp only [bind_tc_ok]
    have hb17 : |toReal v16 - R ^ 10| ≤ 13 * trig_mu := by norm_num at hq16B ⊢; exact hq16B
    -- r^11
    obtain ⟨v17, hq17, hq17B⟩ := q_mul_step v13 v R 9 12 hR (by norm_num) hb14 hV
    rw [hq17]
    simp only [bind_tc_ok]
    have hb18 : |toReal v17 - R ^ 11| ≤ 15 * trig_mu := by norm_num at hq17B ⊢; exact hq17B
    -- r^11/11!
    obtain ⟨v18, hq18, hq18B⟩ := div_fac_step v17 R 11 15 (39916800#i64) (by norm_num)
      (by norm_num) (by norm_num) hR (by norm_num) hb18
    rw [hq18]
    simp only [bind_tc_ok]
    have hb19 : |toReal v18 - R ^ 11 / 39916800| ≤ 16 * trig_mu := by
      norm_num at hq18B ⊢; exact hq18B
    step as ⟨s5, hs5⟩
    have hs5R : (s5.val : ℝ) / q32 = (s4.val : ℝ) / q32 - toReal v18 := by
      rw [hs5]
      simp only [core.convert.num.FromI128I64.from_val_eq, toReal]
      push_cast
      ring
    -- r^10/10!
    obtain ⟨v19, hq19, hq19B⟩ := div_fac_step v16 R 10 13 (3628800#i64) (by norm_num)
      (by norm_num) (by norm_num) hR (by norm_num) hb17
    rw [hq19]
    simp only [bind_tc_ok]
    have hb20 : |toReal v19 - R ^ 10 / 3628800| ≤ 14 * trig_mu := by
      norm_num at hq19B ⊢; exact hq19B
    step as ⟨c5, hc5⟩
    have hc5R : (c5.val : ℝ) / q32 = (c4.val : ℝ) / q32 - toReal v19 := by
      rw [hc5]
      simp only [core.convert.num.FromI128I64.from_val_eq, toReal]
      push_cast
      ring
    -- r^12
    obtain ⟨v20, hq20, hq20B⟩ := q_mul_step v16 v R 10 13 hR (by norm_num) hb17 hV
    rw [hq20]
    simp only [bind_tc_ok]
    have hb21 : |toReal v20 - R ^ 12| ≤ 16 * trig_mu := by norm_num at hq20B ⊢; exact hq20B
    -- r^13
    obtain ⟨v21, hq21, hq21B⟩ := q_mul_step v17 v R 11 15 hR (by norm_num) hb18 hV
    rw [hq21]
    simp only [bind_tc_ok]
    have hb22 : |toReal v21 - R ^ 13| ≤ 18 * trig_mu := by norm_num at hq21B ⊢; exact hq21B
    -- r^13/13!
    obtain ⟨v22, hq22, hq22B⟩ := div_fac_step v21 R 13 18 (6227020800#i64) (by norm_num)
      (by norm_num) (by norm_num) hR (by norm_num) hb22
    rw [hq22]
    simp only [bind_tc_ok]
    have hb23 : |toReal v22 - R ^ 13 / 6227020800| ≤ 19 * trig_mu := by
      norm_num at hq22B ⊢; exact hq22B
    step as ⟨s6, hs6⟩
    have hs6R : (s6.val : ℝ) / q32 = (s5.val : ℝ) / q32 + toReal v22 := by
      rw [hs6]
      simp only [core.convert.num.FromI128I64.from_val_eq, toReal]
      push_cast
      ring
    -- r^12/12!
    obtain ⟨v23, hq23, hq23B⟩ := div_fac_step v20 R 12 16 (479001600#i64) (by norm_num)
      (by norm_num) (by norm_num) hR (by norm_num) hb21
    rw [hq23]
    simp only [bind_tc_ok]
    have hb24 : |toReal v23 - R ^ 12 / 479001600| ≤ 17 * trig_mu := by
      norm_num at hq23B ⊢; exact hq23B
    step as ⟨c6, hc6⟩
    have hc6R : (c6.val : ℝ) / q32 = (c5.val : ℝ) / q32 + toReal v23 := by
      rw [hc6]
      simp only [core.convert.num.FromI128I64.from_val_eq, toReal]
      push_cast
      ring
    -- the accumulated sums, and the two budgets
    have hs6tot : (s6.val : ℝ) / q32
        = R - toReal v2 + toReal v6 - toReal v10 + toReal v14 - toReal v18 + toReal v22 := by
      rw [hs6R, hs5R, hs4R, hs3R, hs2R, hs1R]
    have hc6tot : (c6.val : ℝ) / q32
        = 1 - toReal v3 + toReal v7 - toReal v11 + toReal v15 - toReal v19 + toReal v23 := by
      rw [hc6R, hc5R, hc4R, hc3R, hc2R, hc1R]
    have hsinE : |(s6.val : ℝ) / q32 - sinSer R| ≤ 69 * trig_mu := by
      have hdec : (s6.val : ℝ) / q32 - sinSer R
          = (R ^ 3 / 6 - toReal v2) + (toReal v6 - R ^ 5 / 120) +
            (R ^ 7 / 5040 - toReal v10) + (toReal v14 - R ^ 9 / 362880) +
            (R ^ 11 / 39916800 - toReal v18) + (toReal v22 - R ^ 13 / 6227020800) := by
        rw [hs6tot]; unfold sinSer; ring
      have hf2 : |R ^ 3 / 6 - toReal v2| ≤ 4 * trig_mu := by rw [abs_sub_comm]; exact hb2
      have hf10 : |R ^ 7 / 5040 - toReal v10| ≤ 10 * trig_mu := by
        rw [abs_sub_comm]; exact hb11
      have hf18 : |R ^ 11 / 39916800 - toReal v18| ≤ 16 * trig_mu := by
        rw [abs_sub_comm]; exact hb19
      have h6 := abs_six_le (R ^ 3 / 6 - toReal v2) (toReal v6 - R ^ 5 / 120)
        (R ^ 7 / 5040 - toReal v10) (toReal v14 - R ^ 9 / 362880)
        (R ^ 11 / 39916800 - toReal v18) (toReal v22 - R ^ 13 / 6227020800)
      rw [hdec]
      linarith [h6, hf2, hb7, hf10, hb15, hf18, hb23]
    have hcosE : |(c6.val : ℝ) / q32 - cosSer R| ≤ 57 * trig_mu := by
      have hdec : (c6.val : ℝ) / q32 - cosSer R
          = (R ^ 2 / 2 - toReal v3) + (toReal v7 - R ^ 4 / 24) +
            (R ^ 6 / 720 - toReal v11) + (toReal v15 - R ^ 8 / 40320) +
            (R ^ 10 / 3628800 - toReal v19) + (toReal v23 - R ^ 12 / 479001600) := by
        rw [hc6tot]; unfold cosSer; ring
      have hf3 : |R ^ 2 / 2 - toReal v3| ≤ 2 * trig_mu := by rw [abs_sub_comm]; exact hb4
      have hf11 : |R ^ 6 / 720 - toReal v11| ≤ 8 * trig_mu := by rw [abs_sub_comm]; exact hb12
      have hf19 : |R ^ 10 / 3628800 - toReal v19| ≤ 14 * trig_mu := by
        rw [abs_sub_comm]; exact hb20
      have h6 := abs_six_le (R ^ 2 / 2 - toReal v3) (toReal v7 - R ^ 4 / 24)
        (R ^ 6 / 720 - toReal v11) (toReal v15 - R ^ 8 / 40320)
        (R ^ 10 / 3628800 - toReal v19) (toReal v23 - R ^ 12 / 479001600)
      rw [hdec]
      linarith [h6, hf3, hb8, hf11, hb16, hf19, hb24]
    have hsinT : |(s6.val : ℝ) / q32 - Real.sin R| ≤ 70 * trig_mu := by
      have htri := abs_sub_le ((s6.val : ℝ) / q32) (sinSer R) (Real.sin R)
      have htr := sin_remainder R hR
      linarith
    have hcosT : |(c6.val : ℝ) / q32 - Real.cos R| ≤ 58 * trig_mu := by
      have htri := abs_sub_le ((c6.val : ℝ) / q32) (cosSer R) (Real.cos R)
      have htr := cos_remainder R hR
      linarith
    -- both sums are far inside the `i64` range
    have hsinv : |(s6.val : ℝ)| ≤ (2 : ℝ) * q32 := by
      have h1 : |(s6.val : ℝ) / q32| ≤ 2 := by
        have hs1' := Real.abs_sin_le_one R
        have h70 : 70 * trig_mu ≤ 1 := by unfold trig_mu; norm_num
        calc |(s6.val : ℝ) / q32|
            = |((s6.val : ℝ) / q32 - Real.sin R) + Real.sin R| := by ring_nf
          _ ≤ |(s6.val : ℝ) / q32 - Real.sin R| + |Real.sin R| := abs_add_le _ _
          _ ≤ 2 := by linarith
      rwa [abs_div, abs_of_pos q32_pos, div_le_iff₀ q32_pos] at h1
    have hcosv : |(c6.val : ℝ)| ≤ (2 : ℝ) * q32 := by
      have h1 : |(c6.val : ℝ) / q32| ≤ 2 := by
        have hc1' := Real.abs_cos_le_one R
        have h58 : 58 * trig_mu ≤ 1 := by unfold trig_mu; norm_num
        calc |(c6.val : ℝ) / q32|
            = |((c6.val : ℝ) / q32 - Real.cos R) + Real.cos R| := by ring_nf
          _ ≤ |(c6.val : ℝ) / q32 - Real.cos R| + |Real.cos R| := abs_add_le _ _
          _ ≤ 2 := by linarith
      rwa [abs_div, abs_of_pos q32_pos, div_le_iff₀ q32_pos] at h1
    have hsinz : |s6.val| ≤ (2 : ℤ) ^ 33 := by
      have hq2' : (2 : ℝ) * q32 = (2 : ℝ) ^ 33 := by unfold q32; norm_num
      rw [hq2'] at hsinv
      exact_mod_cast hsinv
    have hcosz : |c6.val| ≤ (2 : ℤ) ^ 33 := by
      have hq2' : (2 : ℝ) * q32 = (2 : ℝ) ^ 33 := by unfold q32; norm_num
      rw [hq2'] at hcosv
      exact_mod_cast hcosv
    obtain ⟨hlo6, hhi6⟩ := abs_le.mp hsinz
    obtain ⟨hlo6', hhi6'⟩ := abs_le.mp hcosz
    obtain ⟨w, hw, hwv⟩ := narrow_spec s6 (by rw [I64.min_eq]; omega)
      (by rw [I64.max_eq]; omega)
    rw [hw]
    simp only [bind_tc_ok]
    obtain ⟨w', hw', hw'v⟩ := narrow_spec c6 (by rw [I64.min_eq]; omega)
      (by rw [I64.max_eq]; omega)
    rw [hw']
    simp only [bind_tc_ok]
    simp only [Aeneas.Std.WP.spec_ok]
    refine ⟨{ sin := w, cos := w' }, rfl, ?_, ?_⟩
    · rw [toReal, hwv]
      refine le_trans hsinT ?_
      unfold trig_mu; norm_num
    · rw [toReal, hw'v]
      refine le_trans hcosT ?_
      unfold trig_mu; norm_num
  obtain ⟨z, hz, sc, hsc, hP⟩ := (Aeneas.Std.WP.spec_equiv_exists _ _).1 hmain
  exact ⟨sc, by rw [hz, hsc], hP.1, hP.2⟩

end

end apeiron_word
