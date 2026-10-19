import Mathlib
import ApeironWord

/-!
# The real-value spec of `basis.log2` (`fixed_ln` at Q32.32, then a `div_rne` by `ln 2`)

`ApeironWord.lean:89077` `fold_port.stats.basis.log2` computes, for `x : i64 > 0`:

```text
r = fixed_ln x                            // Q32.32 of `ln (x / 2^32)`, or `LnError`
v = div_rne (r <<< 32) fixed_ln.LN2_Q32   // Q32.32 of `ln (x/2^32) / ln 2`
narrow v                                  // `none` outside `i64`
```

and `WaterFill.rateAt` consumes it as `Σ (log2 l − log2 θ)` with `hlog` a call hypothesis. This
module states that spec on the water domain (`0 < x`, `x ≤ 2^62`) and proves the pieces that do not
depend on the extracted monad: the real-value constants, the truncation remainder, the normalization
algebra and the `s`-range. The composition is named at the end.

**This spec is made here, not ported.** The lineage search
`rg --glob '*.lean' '^\s*(def|theorem|lemma|abbrev)\s+(fixed_ln|log2|exp2|ln2|logb)\b'
../sanguine/proof ../sanguine-ref` returns nothing, and the live tree has no such declaration
outside the extracted bodies. Mathlib has no `Real.atanh`, so `2·atanh s` is written
`log ((1+s)/(1−s))`, the form the source evaluates; Mathlib's `Real.sum_range_sub_log_div_le`
(`Analysis/SpecialFunctions/Log/Deriv.lean:283`) is the truncation bound.

## What `fixed_ln` computes

For `x : ℤ`, `0 < x ≤ 2^62`: `k = ⌊log₂ x⌋`, `x_norm = x <<< (lz−1)` (mantissa in `[2^62, 2^63)`),
and `p = k+1` when `x_norm ≥ SQRT2_Q62`, else `p = k`. Then `base = 2^p`,
`s = round_div_i128((x − base)·2^32, x + base)`, `ln_m = ln_atanh_poly s`, and
`fixed_ln x = (p − 32)·LN2_Q32 + ln_m`, the Q32.32 of `ln (x/2^32)`, because
`ln x = p·ln2 + 2·atanh s`. The `p − 32` is the Q32.32 scaling: `x` is a Q32.32 word, so the
value's exponent is `p − 32`.

## The error budget and the spec's `ε`

`fixed_ln`'s error is the truncation remainder `9/(4·3^15) ≈ 5.14e-9` (`two_atanh_remainder`), the
six fixed-point roundings and the `s` rounding (`≤ 7·2^-32 ≈ 1.6e-9`), and
`|p − 32|·|LN2_Q32 − 2^32 ln 2|/2^32 ≤ 32·(1/2)/2^32 ≈ 3.7e-9` (`ln2_q32_sharp`), giving
`η = 1/2^26`. The `div_rne` stage then yields `ε = 1/2^25 = 128/2^32`, from
`η/ln2 + |ln(x/2^32)|·|LN2_Q32 − 2^32 ln2|/(LN2_Q32·ln2) + (1/2)/2^32` with
`|ln(x/2^32)| ≤ 23`, i.e. `64·0.693 + 23·0.5 = 55.9` over `2977044471·0.693 = 2.06e9`.
Both constants are proved with the margin the certified bounds actually give, not with the true
errors (`ln 2`'s constant carries a `1.8e-10` error but is certified to `1/2`).

## Boundary

No theorem here refines the extracted `fixed_ln`/`log2` bodies. The remaining theorems, with their
exact statements:

* `round_div2p32_half : ∀ v : ℤ, |(roundDiv2p32 v : ℝ)/2^32 − (v:ℝ)/2^32| ≤ 1/2^33`, and
  `round_div_half : ∀ num den : ℤ, 0 < den → |(roundDiv num den : ℝ) − num/den| ≤ 1/2` — the two
  integer roundings (`ApeironWord.lean:47473`, `:47372`), modelled as `roundDiv2p32`/`roundDiv`.
* `poly_error : ∀ s : ℝ, |s| ≤ 1/3 → |(polyVal s : ℝ)/2^32 − polyBody s| ≤ 1/2^26` — the six-step
  Horner evaluation of `ln_atanh_poly` at Q32.32 (`polyVal` the integer Horner value, `coef_error`
  the coefficient error). With `two_atanh_remainder` this gives `|ln_m/2^32 − 2·atanh s| ≤ 1/2^26`.
* `fixed_ln_spec : FixedLnBound (1/2^26)` — the composition of the above with the `s` rounding, the
  exponent bookkeeping and the `i128`→`i64` range case.
* `log2_spec : Log2Spec (1/2^25)` — `FixedLnBound (1/2^26)`, `round_div_half` at
  `LN2_Q32 = 2977044472`, and `narrow`.
-/

noncomputable section

namespace Learn.Log2

/-! ## The constants -/

/-- **`LN2_Q32 = 2977044472` is `2^32·ln 2` to within `1/2`** (the true error is `0.18`). The
partial sum of `1/2·log((1+x)/(1−x)) = Σ x^{2k+1}/(2k+1)` at `x = 1/3` has tail
`9 (1/3)^25/8` by `Real.sum_range_sub_log_div_le`, so `2^32·ln 2` sits within `0.192` of `LN2_Q32`. This
is the integer-scale bound the `div_rne` stage consumes. -/
theorem ln2_q32_sharp : |(2 : ℝ) ^ 32 * Real.log 2 - 2977044472| ≤ 1 / 2 := by
  have htail := Real.sum_range_sub_log_div_le (x := 1 / 3) (by norm_num : |(1 / 3 : ℝ)| < 1) 12
  have harg : (1 + (1 : ℝ) / 3) / (1 - 1 / 3) = 2 := by norm_num
  rw [harg] at htail
  have hb : |(1 / 3 : ℝ)| ^ (2 * 12 + 1) / (1 - (1 / 3) ^ 2) = 9 / (8 * 3 ^ 25) := by norm_num
  rw [hb] at htail
  set S : ℝ := Finset.sum (Finset.range 12) (fun i : ℕ => (1 / 3 : ℝ) ^ (2 * i + 1) / (2 * i + 1))
    with hS
  have hlog : |Real.log 2 - 2 * S| ≤ 9 / (4 * 3 ^ 25) := by
    have h2 : Real.log 2 - 2 * S = 2 * (1 / 2 * Real.log 2 - S) := by ring
    rw [h2, abs_mul, show |(2 : ℝ)| = 2 by norm_num]
    linarith
  have hSval : |(2 : ℝ) ^ 32 * (2 * S) - 2977044472| ≤ 1 / 4 := by
    rw [hS]
    norm_num [Finset.sum_range_succ]
  have hsplit : (2 : ℝ) ^ 32 * Real.log 2 - 2977044472
      = (2 : ℝ) ^ 32 * (Real.log 2 - 2 * S) + ((2 : ℝ) ^ 32 * (2 * S) - 2977044472) := by ring
  rw [hsplit]
  have h32 : (0 : ℝ) ≤ (2 : ℝ) ^ 32 := by positivity
  have habs : |(2 : ℝ) ^ 32 * (Real.log 2 - 2 * S)| = (2 : ℝ) ^ 32 * |Real.log 2 - 2 * S| := by
    rw [abs_mul, abs_of_nonneg h32]
  have hmul : (2 : ℝ) ^ 32 * |Real.log 2 - 2 * S| ≤ (2 : ℝ) ^ 32 * (9 / (4 * 3 ^ 25)) :=
    mul_le_mul_of_nonneg_left hlog h32
  have hnum : (2 : ℝ) ^ 32 * (9 / (4 * 3 ^ 25)) + 1 / 4 ≤ 1 / 2 := by norm_num
  have htri := abs_add_le ((2 : ℝ) ^ 32 * (Real.log 2 - 2 * S))
    ((2 : ℝ) ^ 32 * (2 * S) - 2977044472)
  rw [habs] at htri
  exact htri.trans (by linarith)

/-- **`SQRT2_Q62 = 6521908912666391553` is `2^62·√2` to within `1/2^53`** (`≈ 9.69e-17`; the constant
carries 53 bits of `√2`, not 62). Proved by squaring: `(c − 2^-53)^2 < 2 < (c + 2^-53)^2`. -/
theorem sqrt2_q62_bound :
    |(6521908912666391553 : ℝ) / 2 ^ 62 - Real.sqrt 2| ≤ 1 / 2 ^ 53 := by
  have hlo : (6521908912666391553 : ℝ) / 2 ^ 62 - 1 / 2 ^ 53 < Real.sqrt 2 := by
    rw [Real.lt_sqrt (by positivity)]
    norm_num
  have hhi : Real.sqrt 2 < (6521908912666391553 : ℝ) / 2 ^ 62 + 1 / 2 ^ 53 := by
    rw [Real.sqrt_lt (by norm_num) (by positivity)]
    norm_num
  rw [abs_le]
  constructor <;> linarith

/-- The seven Q32.32 Horner coefficients of `ln_atanh_poly` (`ApeironWord.lean:47488`): `coef j`
multiplies `s^{2j+1}`, so `coef 0 = 8589934592 = 2^33` (the constant term the source adds last) and
`coef 6 = 660764199` (the innermost, `s^13`). -/
def coef : ℕ → ℤ
  | 0 => 8589934592
  | 1 => 2863311531
  | 2 => 1717986918
  | 3 => 1227133513
  | 4 => 954437177
  | 5 => 780903145
  | 6 => 660764199
  | _ => 0

/-- **Each coefficient is the nearest Q32.32 word to `2^32·2/(2j+1)`**: the coefficient error is at
most half a unit. -/
theorem coef_error (j : ℕ) (hj : j < 7) :
    |(coef j : ℝ) - (2 : ℝ) ^ 32 * (2 / (2 * (j : ℝ) + 1))| ≤ 1 / 2 := by
  interval_cases j <;> norm_num [coef]

/-! ## The integer roundings and the polynomial, as models

The two roundings and the Horner evaluation below mirror the source bodies word for word; their
error lemmas are the named remainder (`round_div2p32_half`, `round_div_half`, `poly_error`). -/

/-- `round_div_2p32` (`ApeironWord.lean:47473`): `value >>> 32` after adding `2^31` to the magnitude,
i.e. nearest Q32.32 with ties away from zero. -/
def roundDiv2p32 (v : ℤ) : ℤ :=
  if 0 ≤ v then (v + 2 ^ 31) / 2 ^ 32 else -(((-v) + 2 ^ 31) / 2 ^ 32)

/-- `round_div_i128` (`ApeironWord.lean:47372`): nearest `num/den`, `den > 0`, ties away from zero. -/
def roundDiv (num den : ℤ) : ℤ :=
  if 0 ≤ num then (num + den / 2) / den else -(((-num) + den / 2) / den)

/-- The Horner value `ln_atanh_poly` computes at Q32.32 `s` (`ApeironWord.lean:47488`): six
`roundDiv2p32` steps over `y = roundDiv2p32 (s·s)` and the seven `coef`, then `·s` and one more
rounding. -/
def polyVal (s : ℤ) : ℤ :=
  let y := roundDiv2p32 (s * s)
  let a0 := roundDiv2p32 (660764199 * y) + 780903145
  let a1 := roundDiv2p32 (a0 * y) + 954437177
  let a2 := roundDiv2p32 (a1 * y) + 1227133513
  let a3 := roundDiv2p32 (a2 * y) + 1717986918
  let a4 := roundDiv2p32 (a3 * y) + 2863311531
  let a5 := roundDiv2p32 (a4 * y) + 8589934592
  roundDiv2p32 (a5 * s)

/-! ## The truncation remainder of `ln_atanh_poly`

`ln_atanh_poly` evaluates the odd series at `s` up to `s^13` (`k < 7`, coefficients `coef`, each
rounded to Q32.32). The remainder is the `k ≥ 7` tail. -/

/-- The `k < 7` partial sum: the polynomial body of `ln_atanh_poly` at real `s`, i.e. `2·atanh s`
truncated at `s^13`. -/
def polyBody (s : ℝ) : ℝ :=
  2 * Finset.sum (Finset.range 7) (fun k : ℕ => s ^ (2 * k + 1) / (2 * k + 1))

/-- **The truncation remainder at `|s| ≤ 1/3` is at most `9/(4·3^15)`** (`≈ 5.14e-9`, `≈ 22.1`
Q32.32 units), from `Real.sum_range_sub_log_div_le`: the tail is at most `|s|^15/(1−s²) ≤
9/(8·3^15)`, doubled for `2·atanh s`. -/
theorem two_atanh_remainder {s : ℝ} (h0 : 0 ≤ s) (h1 : s ≤ 1 / 3) :
    |Real.log ((1 + s) / (1 - s)) - polyBody s| ≤ 9 / (4 * 3 ^ 15) := by
  have hs1 : |s| < 1 := by rw [abs_of_nonneg h0]; linarith
  have h := Real.sum_range_sub_log_div_le (x := s) hs1 7
  have hb : |s| ^ (2 * 7 + 1) / (1 - s ^ 2) ≤ 9 / (8 * 3 ^ 15) := by
    rw [abs_of_nonneg h0]
    have hs15 : s ^ 15 ≤ (1 / 3 : ℝ) ^ 15 := pow_le_pow_left₀ h0 h1 15
    have hden : (1 : ℝ) - s ^ 2 ≥ 1 - 1 / 9 := by nlinarith
    have h1pos : (0 : ℝ) < 1 - s ^ 2 := by linarith
    rw [div_le_iff₀ h1pos]
    have hstep : s ^ 15 * (1 - 1 / 9) ≤ (1 / 3 : ℝ) ^ 15 * (1 - s ^ 2) := by nlinarith [hs15, hden]
    have hval : ((1 / 3 : ℝ) ^ 15 * (1 - s ^ 2)) / (1 - s ^ 2) = (1 / 3 : ℝ) ^ 15 := by field_simp
    nlinarith [hstep, hval, hs15]
  have hbody : polyBody s = 2 * Finset.sum (Finset.range 7) (fun k : ℕ => s ^ (2 * k + 1) / (2 * k + 1)) :=
    rfl
  have hmul : |Real.log ((1 + s) / (1 - s)) - polyBody s|
      = 2 * |1 / 2 * Real.log ((1 + s) / (1 - s))
          - Finset.sum (Finset.range 7) (fun k : ℕ => s ^ (2 * k + 1) / (2 * k + 1))| := by
    have h2 : Real.log ((1 + s) / (1 - s)) - polyBody s
        = 2 * (1 / 2 * Real.log ((1 + s) / (1 - s))
          - Finset.sum (Finset.range 7) (fun k : ℕ => s ^ (2 * k + 1) / (2 * k + 1))) := by
      rw [hbody]; ring
    rw [h2, abs_mul, show |(2 : ℝ)| = 2 by norm_num]
  rw [hmul]
  linarith

/-! ## Normalization: `ln x = ln p + log((1+s)/(1−s))` and `|s| ≤ 1/3` -/

/-- **The normalization identity.** For `0 < x`, `0 < p` and `s = (x − p)/(x + p)`:
`ln x = ln p + log((1+s)/(1−s))`. This is the identity the source's
`(p − 32)·LN2_Q32 + ln_m` computes. -/
theorem log_normalization {x p s : ℝ} (hx : 0 < x) (hp : 0 < p)
    (hs : s = (x - p) / (x + p)) :
    Real.log x = Real.log p + Real.log ((1 + s) / (1 - s)) := by
  have h1 : 1 + s = 2 * x / (x + p) := by
    rw [hs]; field_simp; ring
  have h2 : 1 - s = 2 * p / (x + p) := by
    rw [hs]; field_simp; ring
  have hsp : (1 + s) / (1 - s) = x / p := by
    rw [h1, h2]
    field_simp
  have hdiv : Real.log (x / p) = Real.log x - Real.log p := Real.log_div hx.ne' hp.ne'
  rw [hsp, hdiv]
  ring

/-- **`|s| ≤ 1/3` from the two linear range facts.** `(x−p)/(x+p)` lies in `[-1/3, 1/3]` when
`p/2 ≤ x ≤ 2p`. -/
theorem s_abs_le_third {x p : ℝ} (hp : 0 < p) (hlo : p / 2 ≤ x) (hhi : x ≤ 2 * p) :
    |(x - p) / (x + p)| ≤ 1 / 3 := by
  have hden : (0 : ℝ) < x + p := by linarith
  rw [abs_div, abs_of_pos hden, div_le_iff₀ hden, abs_le]
  constructor <;> linarith

/-- **The `p`-selection to `|s| ≤ 1/3`, at the source's own bounds.** The source picks `p` so that
`p/√2 ≤ x ≤ 3p/2`; `√2 ≤ 2` makes this `p/2 ≤ x ≤ 2p`, hence `|s| ≤ 1/3`. -/
theorem s_abs_le_third_of_sqrt2 {x p : ℝ} (hp : 0 < p)
    (hlo : p / Real.sqrt 2 ≤ x) (hhi : x ≤ 3 * p / 2) :
    |(x - p) / (x + p)| ≤ 1 / 3 := by
  have hsqrt2 : Real.sqrt 2 ≤ 2 := by
    have h := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)
    have hnn := Real.sqrt_nonneg 2
    nlinarith
  have hlo' : p / 2 ≤ x := by
    have hspos : (0 : ℝ) < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
    have : p / 2 ≤ p / Real.sqrt 2 := by
      rw [div_le_div_iff₀ (by norm_num : (0 : ℝ) < 2) hspos]
      nlinarith
    linarith
  have hhi' : x ≤ 2 * p := by linarith
  exact s_abs_le_third hp hlo' hhi'

/-! ## The spec, as a `Prop` -/

/-- **The spec.** On the water domain (`0 < x`, `x ≤ 2^62`) `basis.log2` returns `some v` with
`|v/2^32 − log₂(x/2^32)| ≤ ε`. This is what `WaterFill.rateAt`'s `hlog` needs. -/
def Log2Spec (ε : ℝ) : Prop :=
  ∀ x : ℤ, 0 < x → x ≤ 2 ^ 62 →
    ∃ v : ℤ, |(v : ℝ) / 2 ^ 32 - Real.log ((x : ℝ) / 2 ^ 32) / Real.log 2| ≤ ε

/-- The `fixed_ln` real bound the `log2` stage consumes: `w` is the Q32.32 of `ln (x/2^32)` within
`η`. -/
def FixedLnBound (η : ℝ) : Prop :=
  ∀ x : ℤ, 0 < x → x ≤ 2 ^ 62 →
    ∃ w : ℤ, |(w : ℝ) / 2 ^ 32 - Real.log ((x : ℝ) / 2 ^ 32)| ≤ η

end Learn.Log2
