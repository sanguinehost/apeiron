/-
Refinement of the fixed-point `sin`/`cos` body against the real trigonometric functions
.

The Rust body is `src/numeric/fixed_trig.rs`: `i64` Q32.32, window `[-4096, 4096]` radians,
`2π` reduction plus quadrant fold and degree-8 Taylor polynomials, `Result` refusal outside the
window. It is extracted into `ApeironWord` by `proof/regen.sh`; the reduction lemma, the
representation relation and the bounds are built here.

## Representation

An `i64` `v` denotes the real number `toReal v = v.val / 2^32`, the signed Q32.32 value
(`Numeric.Scale`). `trig_window x` is the documented admissible domain `[-4096·2^32, 4096·2^32]`.
The refusal correspondence is `fixed_sin_cos_refuses`: outside `trig_window` the extracted body
returns `Err OutOfRange`.

## Statement choice

The full window carries the combined bound, one clause per output, mirroring the exp and rsqrt
modules (`Numeric.ExpRefinement`, `Numeric.RsqrtRefinement`):

  `|toReal sc.sin - sin (toReal x)| <= 2^-20 · |sin (toReal x)| + 2^-20`,
  `|toReal sc.cos - cos (toReal x)| <= 2^-20 · |cos (toReal x)| + 2^-20`.

The relative term is the `2^-20` target. The absolute floor is required near the zeros of the two
outputs, where the output itself vanishes and a relative bound alone is unprovable: the dominant
fixed-point error there is the range reduction, whose rounded `TWO_PI_Q32` displaces `r` by up to
`|k| · 2^-33 <= 652 · 2^-33 ≈ 7.6e-8`. `trig_floor = 2^-20 ≈ 9.5e-7` dominates that displacement
by a factor `≈ 12`, so the floor is provable with repair room rather than pinned to the measured
`1.42e-8`. The Rust unit test asserts the tighter `5e-7 · |f| + 2^-24` on its dense grid; that
grid result is a measurement, not the universal statement. `fixed_sin_cos_refines` states this one.
-/
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Real.Pi.Bounds
import Numeric.Scale

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Representation relation -/

/-- The documented admissible window `[-4096·2^32, 4096·2^32]`, as a predicate on the signed Q32.32
value. `4096·2^32 = 17592186044416 = 2^44`. -/
def trig_window (x : I64) : Prop :=
  (-17592186044416 : ℤ) ≤ x.val ∧ x.val ≤ 17592186044416

/-! ## Refusal and bound statements -/

/-- Refusal correspondence: outside the window the extracted body returns `Err OutOfRange`. -/
def fixed_sin_cos_refuses : Prop :=
  ∀ x : I64,
    ¬ trig_window x →
      word_fixed_sin_cos x = ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)

/-- The absolute floor near the zeros of the two outputs, `2^-20`. It dominates the range-reduction
displacement `|k| · 2^-33 <= 652 · 2^-33 ≈ 7.6e-8`. -/
def trig_floor : ℝ := (2 : ℝ) ^ (-20 : ℤ)

/-- The target refinement bound on the window: success, with the combined absolute/relative error
bound `2^-20 · |f (toReal x)| + 2^-20` for each output. -/
def fixed_sin_cos_refines : Prop :=
  ∀ x : I64,
    trig_window x →
      ∃ sc : fixed_trig.SinCos,
        word_fixed_sin_cos x = ok (core.result.Result.Ok sc) ∧
          |toReal sc.sin - Real.sin (toReal x)| ≤
            epsExp * |Real.sin (toReal x)| + trig_floor ∧
          |toReal sc.cos - Real.cos (toReal x)| ≤
            epsExp * |Real.cos (toReal x)| + trig_floor

/-! ## Refusal: structural lemmas and endpoint evaluation

The extracted body opens with `let lower ← LOWER; if x < lower then Err else …`, with the mirrored
upper test in the second branch. `word_fixed_sin_cos_refuses_of_lt`/`_of_gt` are proved from that
structure alone, parameterised by the values of the extracted endpoint `const`s: when the local
comparison holds, the body returns `Err OutOfRange`.

The specialised Aeneas shift and multiply specifications evaluate the extracted `const`s:
`ONE = 1 <<< 32 = 2^32`, `LOWER = -4096 · ONE = -17592186044416`,
`UPPER = 4096 · ONE = 17592186044416`. `fixed_sin_cos_refuses_proved` instantiates the
parameterised lemmas at those values and discharges `fixed_sin_cos_refuses` over the complement of
`trig_window`.
-/

/-- Below the evaluated lower endpoint the extracted body refuses. -/
theorem word_fixed_sin_cos_refuses_of_lt (x c : I64)
    (hL : fixed_trig.fixed_sin_cos.LOWER = ok c) (h : x < c) :
    word_fixed_sin_cos x = ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange) := by
  rw [word_fixed_sin_cos, fixed_trig.fixed_sin_cos, hL]
  simp only [bind_tc_ok, if_pos h]

/-- Above the evaluated upper endpoint the extracted body refuses. -/
theorem word_fixed_sin_cos_refuses_of_gt (x cL cU : I64)
    (hL : fixed_trig.fixed_sin_cos.LOWER = ok cL)
    (hU : fixed_trig.fixed_sin_cos.UPPER = ok cU)
    (hn : ¬ x < cL) (h : x > cU) :
    word_fixed_sin_cos x = ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange) := by
  rw [word_fixed_sin_cos, fixed_trig.fixed_sin_cos, hL, hU]
  simp only [bind_tc_ok, if_neg hn, if_pos h]

/-! ## Endpoint evaluation and instantiation -/

/-- The extracted scale constant `ONE` evaluates to `2^32`. -/
theorem fixed_trig_ONE_eq : fixed_trig.ONE = ok (4294967296#i64) := by
  have h : fixed_trig.ONE ⦃ z => z = (4294967296#i64) ⦄ := by
    unfold fixed_trig.ONE
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I64.size, IScalar.size, I64.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted lower endpoint evaluates to `-4096 · 2^32`. -/
theorem fixed_trig_LOWER_eq :
    fixed_trig.fixed_sin_cos.LOWER = ok ((-17592186044416)#i64) := by
  have h : fixed_trig.fixed_sin_cos.LOWER ⦃ z => z = ((-17592186044416)#i64) ⦄ := by
    unfold fixed_trig.fixed_sin_cos.LOWER
    rw [fixed_trig_ONE_eq]
    simp only [bind_tc_ok]
    apply Aeneas.Std.WP.spec_mono
      (I64.mul_spec (x := (-4096)#i64) (y := 4294967296#i64)
        (by norm_num [I64.min, I64.max, I64.numBits])
        (by norm_num [I64.min, I64.max, I64.numBits]))
    intro z hz
    apply IScalar.eq_imp
    rw [hz]
    norm_num
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted upper endpoint evaluates to `4096 · 2^32`. -/
theorem fixed_trig_UPPER_eq :
    fixed_trig.fixed_sin_cos.UPPER = ok (17592186044416#i64) := by
  have h : fixed_trig.fixed_sin_cos.UPPER ⦃ z => z = (17592186044416#i64) ⦄ := by
    unfold fixed_trig.fixed_sin_cos.UPPER
    rw [fixed_trig_ONE_eq]
    simp only [bind_tc_ok]
    apply Aeneas.Std.WP.spec_mono
      (I64.mul_spec (x := 4096#i64) (y := 4294967296#i64)
        (by norm_num [I64.min, I64.max, I64.numBits])
        (by norm_num [I64.min, I64.max, I64.numBits]))
    intro z hz
    apply IScalar.eq_imp
    rw [hz]
    norm_num
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- Refusal correspondence: outside `trig_window` the extracted body returns `Err OutOfRange`. -/
theorem fixed_sin_cos_refuses_proved : fixed_sin_cos_refuses := by
  intro x hx
  rw [trig_window] at hx
  by_cases hlow : x.val < (-17592186044416 : ℤ)
  · exact word_fixed_sin_cos_refuses_of_lt x _ fixed_trig_LOWER_eq (by simpa using hlow)
  · have hlowle : (-17592186044416 : ℤ) ≤ x.val := le_of_not_gt hlow
    have hhigh : (17592186044416 : ℤ) < x.val := by
      by_contra hc
      exact hx ⟨hlowle, le_of_not_gt hc⟩
    refine word_fixed_sin_cos_refuses_of_gt x _ _ fixed_trig_LOWER_eq fixed_trig_UPPER_eq ?_ ?_
    · simp only [IScalar.lt_equiv]
      exact not_lt.mpr hlowle
    · simpa using hhigh

/-! ## Success branch: window guards pass

`trig_window x` is the admissible domain. On it, the two evaluated endpoint comparisons of the
extracted body are false, so the body falls through to the reduction and the quadrant fold.
`word_fixed_sin_cos_window_past_guards` records that fall-through as a rewrite: the left shape is
the extracted `do` block with both `if` guards already resolved to their `else` branches. The three
remaining refusals inside the reduction (the `k_wide`, `r_wide` `i64`-range tests) are not
discharged here; this is the structural first slice, not the value statement.
-/

theorem fixed_trig_lower_test_false (x : I64) (hx : trig_window x) :
    ¬ x < ((-17592186044416)#i64) := by
  intro h
  rw [IScalar.lt_equiv] at h
  exact not_lt.mpr hx.1 h

theorem fixed_trig_upper_test_false (x : I64) (hx : trig_window x) :
    ¬ x > (17592186044416#i64) := by
  intro h
  rw [gt_iff_lt] at h
  rw [IScalar.lt_equiv] at h
  exact not_lt.mpr hx.2 h

theorem word_fixed_sin_cos_window_past_guards (x : I64) (hx : trig_window x) :
    word_fixed_sin_cos x =
      (do
        let i2 ← lift (core.convert.num.FromI128I64.from x)
        let i3 ←
          lift (core.convert.num.FromI128I64.from fixed_trig.TWO_PI_RECIP_Q32)
        let product ← i2 * i3
        let i4 ← 1#i128 <<< 63#i32
        let i5 ← product + i4
        let k_wide ← i5 >>> 64#i32
        let i6 ← lift (IScalar.cast .I128 core.num.I64.MAX)
        if k_wide > i6
        then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
        else
          let i7 ← lift (IScalar.cast .I128 core.num.I64.MIN)
          if k_wide < i7
          then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
          else
            let k ← lift (IScalar.cast .I64 k_wide)
            let i8 ← lift (core.convert.num.FromI128I64.from x)
            let i9 ← lift (core.convert.num.FromI128I64.from k)
            let i10 ←
              lift (core.convert.num.FromI128I64.from fixed_trig.TWO_PI_Q32)
            let i11 ← i9 * i10
            let r_wide ← i8 - i11
            let i12 ← lift (IScalar.cast .I128 core.num.I64.MAX)
            if r_wide > i12
            then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
            else
              let i13 ← lift (IScalar.cast .I128 core.num.I64.MIN)
              if r_wide < i13
              then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
              else
                let r ← lift (IScalar.cast .I64 r_wide)
                let (s, negate) ←
                  if r > fixed_trig.HALF_PI_Q32
                  then do
                       let i14 ← r - fixed_trig.PI_Q32
                       ok (i14, true)
                  else
                    do
                    let i14 ← -. fixed_trig.HALF_PI_Q32
                    if r < i14
                    then let i15 ← r + fixed_trig.PI_Q32
                         ok (i15, true)
                    else ok (r, false)
                let (arg, swap) ←
                  if s > fixed_trig.QUARTER_PI_Q32
                  then do
                       let i14 ← fixed_trig.HALF_PI_Q32 - s
                       ok (i14, 1#i32)
                  else
                    do
                    let i14 ← -. fixed_trig.QUARTER_PI_Q32
                    if s < i14
                    then let i15 ← s + fixed_trig.HALF_PI_Q32
                         ok (i15, 2#i32)
                    else ok (s, 0#i32)
                let (sin_t, cos_t) ← fixed_trig.taylor_sin_cos arg
                let (sin_out, cos_out) ←
                  if swap = 1#i32
                  then ok (cos_t, sin_t)
                  else
                    if swap = 2#i32
                    then do
                         let i14 ← -. cos_t
                         ok (i14, sin_t)
                    else ok (sin_t, cos_t)
                if negate
                then
                  let sin_out1 ← -. sin_out
                  let cos_out1 ← -. cos_out
                  ok (core.result.Result.Ok { sin := sin_out1, cos := cos_out1 })
                else
                  ok (core.result.Result.Ok { sin := sin_out, cos := cos_out })) := by
  rw [word_fixed_sin_cos, fixed_trig.fixed_sin_cos, fixed_trig_LOWER_eq, fixed_trig_UPPER_eq]
  simp only [bind_tc_ok]
  rw [if_neg (fixed_trig_lower_test_false x hx),
      if_neg (fixed_trig_upper_test_false x hx)]

/-! ## Constant values and the reduction defect

The exact `toReal` images of the extracted constants, and the integer defect of the reciprocal
approximation. The reduction computes `k` by rounding `x · TWO_PI_RECIP_Q32 / 2^64` and then
`r = x - k · TWO_PI_Q32`. The product `TWO_PI_Q32 · TWO_PI_RECIP_Q32` misses `2^64` by
`two_pi_recip_defect`, the multiplicative error budget of the reduction. The remainder bound itself
needs the value-level unfold and is not stated here.
-/

theorem toReal_two_pi : toReal fixed_trig.TWO_PI_Q32 = 26986075409 / q32 := by
  unfold toReal fixed_trig.TWO_PI_Q32
  norm_num

theorem toReal_two_pi_recip : toReal fixed_trig.TWO_PI_RECIP_Q32 = 683565276 / q32 := by
  unfold toReal fixed_trig.TWO_PI_RECIP_Q32
  norm_num

theorem toReal_pi : toReal fixed_trig.PI_Q32 = 13493037705 / q32 := by
  unfold toReal fixed_trig.PI_Q32
  norm_num

theorem toReal_half_pi : toReal fixed_trig.HALF_PI_Q32 = 6746518852 / q32 := by
  unfold toReal fixed_trig.HALF_PI_Q32
  norm_num

theorem toReal_quarter_pi : toReal fixed_trig.QUARTER_PI_Q32 = 3373259426 / q32 := by
  unfold toReal fixed_trig.QUARTER_PI_Q32
  norm_num

theorem two_pi_recip_defect :
    (26986075409 : ℤ) * 683565276 - (2 : ℤ) ^ 64 = 11400346268 := by
  norm_num

/-! ## Integer range facts for the reduction

The reduction rounds `k = (x · TWO_PI_RECIP_Q32 + 2^63) >>> 64` and forms `r = x - k · TWO_PI_Q32`.
The extracted body tests `k_wide` and `r_wide` against the `i64` endpoints; these lemmas supply the
arithmetic those refusal tests need, independently of the `step` context:

* `fixed_trig_shift_bound` bounds the shift `k` on the window (`-652 <= k <= 652`), so
  `k · TWO_PI_Q32 ∈ [i64.min, i64.max]` and the `i128→i64` cast of `k` is the identity;
* `fixed_trig_shift_bound_window` lifts that to the `i128` product range seen by the guard.

The proof splits `x · 683565276 + 2^63 = k · 2^64 + t` with `t ∈ [0, 2^64)` and bounds `k` by
`-2^44 · 683565276 + 2^63` and `2^44 · 683565276 + 2^63` after the floor division by `2^64`. The
`r`-range and the quadrant-fold value characterization are the remaining value obligations.
-/

/-- The reduction shift `k = (x · TWO_PI_RECIP_Q32 + 2^63) >>> 64` lies in `[-652, 652]` on the
window. -/
theorem fixed_trig_shift_bound (x s : ℤ)
    (hlo : -17592186044416 ≤ x) (hhi : x ≤ 17592186044416)
    (hs : s = (x * 683565276 + 2 ^ 63) >>> 64) :
    -652 ≤ s ∧ s ≤ 652 := by
  have hsdiv : s = (x * 683565276 + 2 ^ 63) / 2 ^ 64 := by
    rw [hs, Int.shiftRight_eq_div_pow]
    norm_num
  have hA : 2 ^ 64 * s + (x * 683565276 + 2 ^ 63) % 2 ^ 64
      = x * 683565276 + 2 ^ 63 := by
    have := Int.mul_ediv_add_emod (x * 683565276 + 2 ^ 63) (2 ^ 64)
    rw [← hsdiv] at this
    exact this
  have ht0 : 0 ≤ (x * 683565276 + 2 ^ 63) % 2 ^ 64 := Int.emod_nonneg _ (by norm_num)
  have htlt : (x * 683565276 + 2 ^ 63) % 2 ^ 64 < 2 ^ 64 := Int.emod_lt_of_pos _ (by norm_num)
  have hmin : -17592186044416 * 683565276 + 2 ^ 63 ≤ x * 683565276 + 2 ^ 63 := by nlinarith [hlo]
  have hmax : x * 683565276 + 2 ^ 63 ≤ 17592186044416 * 683565276 + 2 ^ 63 := by nlinarith [hhi]
  have hslo : -652 ≤ s := by
    by_contra hc
    have hsle : s ≤ -653 := by omega
    have hlt : x * 683565276 + 2 ^ 63 < (s + 1) * 2 ^ 64 := by nlinarith [hA, htlt]
    nlinarith [hlt, hmin, hsle]
  have hshi : s ≤ 652 := by
    by_contra hc
    have hsge : 653 ≤ s := by omega
    have hle : 653 * 2 ^ 64 ≤ x * 683565276 + 2 ^ 63 := by nlinarith [hA, ht0, hsge]
    nlinarith [hle, hmax]
  exact ⟨hslo, hshi⟩

/-- `k · TWO_PI_Q32` lies in the `i64` range on the window, so the `i128→i64` cast and the following
subtraction see the true integer product. -/
theorem fixed_trig_shift_bound_window (x s : ℤ)
    (hlo : -17592186044416 ≤ x) (hhi : x ≤ 17592186044416)
    (hs : s = (x * 683565276 + 2 ^ 63) >>> 64) :
    -2 ^ 63 ≤ s * 26986075409 ∧ s * 26986075409 ≤ 2 ^ 63 - 1 := by
  obtain ⟨hslo, hshi⟩ := fixed_trig_shift_bound x s hlo hhi hs
  constructor <;> nlinarith

/-- The reduction remainder `r = x - k · TWO_PI_Q32` is bounded by `2^36`, hence lies in the `i64`
range. Unlike the `ln 2` reduction the remainder is not bounded by one `2^32` unit: `r` is the
residue modulo `2π`, so `|r| <= π · 2^32 ≈ 1.35e10 = 2^33.65`. The `2^36` slack covers that plus
the `|k| · two_pi_recip_defect <= 652 · 11400346268 ≈ 7.4e12 = 2^42.75` reciprocal defect. -/
theorem fixed_trig_reduction_remainder_range (x s : ℤ)
    (hshift : s = (x * 683565276 + 2 ^ 63) >>> 64)
    (hk : -2 ^ 63 ≤ s * 26986075409 ∧ s * 26986075409 ≤ 2 ^ 63 - 1) :
    -2 ^ 36 ≤ x - s * 26986075409 ∧ x - s * 26986075409 ≤ 2 ^ 36 := by
  have hshift' : (x * 683565276 + 2 ^ 63) >>> 64 = s := hshift.symm
  have hdiv : (x * 683565276 + 2 ^ 63) / 2 ^ 64 = s := by
    rw [Int.shiftRight_eq_div_pow] at hshift'
    norm_num at hshift'
    exact hshift'
  have hdecomp : 2 ^ 64 * s + (x * 683565276 + 2 ^ 63) % 2 ^ 64
      = x * 683565276 + 2 ^ 63 := by
    have := Int.mul_ediv_add_emod (x * 683565276 + 2 ^ 63) (2 ^ 64)
    rw [hdiv] at this
    exact this
  have ht0 : 0 ≤ (x * 683565276 + 2 ^ 63) % 2 ^ 64 := Int.emod_nonneg _ (by norm_num)
  have htlt : (x * 683565276 + 2 ^ 63) % 2 ^ 64 < 2 ^ 64 := Int.emod_lt_of_pos _ (by norm_num)
  have hmR : (26986075409 : ℤ) * 683565276 = 2 ^ 64 + 11400346268 := by norm_num
  have hDle : (11400346268 : ℤ) ≤ 26986075409 := by norm_num
  have hDnn : (0 : ℤ) ≤ 11400346268 := by norm_num
  have hsD_ub : s * 11400346268 ≤ 2 ^ 63 := by nlinarith [hk.1, hk.2, hDle, hDnn]
  have hsD_lb : -2 ^ 63 ≤ s * 11400346268 := by nlinarith [hk.1, hk.2, hDle, hDnn]
  have hrR : (x - s * 26986075409) * 683565276
      = (x * 683565276 + 2 ^ 63) % 2 ^ 64 - 2 ^ 63 - s * 11400346268 := by
    nlinarith [hdecomp, hmR]
  have hlow : -2 ^ 64 ≤ (x - s * 26986075409) * 683565276 := by
    rw [hrR]; nlinarith [ht0, htlt, hsD_ub, hsD_lb]
  have hhigh : (x - s * 26986075409) * 683565276 < 2 ^ 64 := by
    rw [hrR]; nlinarith [ht0, htlt, hsD_ub, hsD_lb]
  have hub : x - s * 26986075409 ≤ 2 ^ 36 := by
    by_contra hc
    have hrge : (2 : ℤ) ^ 36 ≤ x - s * 26986075409 := by omega
    have hmul : (2 : ℤ) ^ 36 * 683565276
        ≤ (x - s * 26986075409) * 683565276 :=
      mul_le_mul_of_nonneg_right hrge (by norm_num)
    nlinarith [hhigh, hmul]
  have hlb : -(2 : ℤ) ^ 36 ≤ x - s * 26986075409 := by
    by_contra hc
    have hrle : x - s * 26986075409 ≤ -(2 : ℤ) ^ 36 := by omega
    have hmul : (x - s * 26986075409) * 683565276
        ≤ (-(2 : ℤ) ^ 36) * 683565276 :=
      mul_le_mul_of_nonneg_right hrle (by norm_num)
    nlinarith [hlow, hmul]
  exact ⟨hlb, hub⟩

/-- Tight reduction remainder bound: over the window the remainder `r = x - k · TWO_PI_Q32` lies in
`[-2^34, 2^34]`. The true magnitude is `π·2^32 + 2e-5·2^32 ≈ 2^33.65`; `|r| <= 2^32` is *false*
(`x = HALF_PI_Q32` gives `k = 0` and `r = HALF_PI_Q32 = 6746518852 > 2^32`), so `2^34` is the sharp
power-of-two bound, and `fold_arg_abs_le` then gives `|arg| <= 2^32`. Uses `|k| <= 652`
(`fixed_trig_shift_bound`) so the reciprocal defect contributes at most `652 · D < 2^43`. -/
theorem fixed_trig_reduction_remainder_range_tight (x s : ℤ)
    (hlo : -17592186044416 ≤ x) (hhi : x ≤ 17592186044416)
    (hshift : s = (x * 683565276 + 2 ^ 63) >>> 64) :
    -2 ^ 34 ≤ x - s * 26986075409 ∧ x - s * 26986075409 ≤ 2 ^ 34 := by
  have hsb := fixed_trig_shift_bound x s hlo hhi hshift
  have hshift' : (x * 683565276 + 2 ^ 63) >>> 64 = s := hshift.symm
  have hdiv : (x * 683565276 + 2 ^ 63) / 2 ^ 64 = s := by
    rw [Int.shiftRight_eq_div_pow] at hshift'
    norm_num at hshift'
    exact hshift'
  have hdecomp : 2 ^ 64 * s + (x * 683565276 + 2 ^ 63) % 2 ^ 64
      = x * 683565276 + 2 ^ 63 := by
    have := Int.mul_ediv_add_emod (x * 683565276 + 2 ^ 63) (2 ^ 64)
    rw [hdiv] at this
    exact this
  have ht0 : 0 ≤ (x * 683565276 + 2 ^ 63) % 2 ^ 64 := Int.emod_nonneg _ (by norm_num)
  have htlt : (x * 683565276 + 2 ^ 63) % 2 ^ 64 < 2 ^ 64 := Int.emod_lt_of_pos _ (by norm_num)
  have hmR : (26986075409 : ℤ) * 683565276 = 2 ^ 64 + 11400346268 := by norm_num
  have hsD_ub : s * 11400346268 ≤ 2 ^ 43 := by
    have h := mul_le_mul_of_nonneg_right hsb.2 (by norm_num : (0:ℤ) ≤ 11400346268)
    have h2 : (652:ℤ) * 11400346268 ≤ 2 ^ 43 := by norm_num
    linarith
  have hsD_lb : -2 ^ 43 ≤ s * 11400346268 := by
    have h := mul_le_mul_of_nonneg_right hsb.1 (by norm_num : (0:ℤ) ≤ 11400346268)
    have h2 : -(2:ℤ) ^ 43 ≤ (-652:ℤ) * 11400346268 := by norm_num
    linarith
  have hrR : (x - s * 26986075409) * 683565276
      = (x * 683565276 + 2 ^ 63) % 2 ^ 64 - 2 ^ 63 - s * 11400346268 := by
    nlinarith [hdecomp, hmR]
  have hlow : -2 ^ 63 - 2 ^ 43 ≤ (x - s * 26986075409) * 683565276 := by
    rw [hrR]; nlinarith [ht0, htlt, hsD_ub, hsD_lb]
  have hhigh : (x - s * 26986075409) * 683565276 < 2 ^ 63 + 2 ^ 43 := by
    rw [hrR]; nlinarith [ht0, htlt, hsD_ub, hsD_lb]
  have hub : x - s * 26986075409 ≤ 2 ^ 34 := by
    by_contra hc
    have hrge : (2 : ℤ) ^ 34 ≤ x - s * 26986075409 := by omega
    have hmul : (2 : ℤ) ^ 34 * 683565276
        ≤ (x - s * 26986075409) * 683565276 :=
      mul_le_mul_of_nonneg_right hrge (by norm_num)
    nlinarith [hhigh, hmul]
  have hlb : -(2 : ℤ) ^ 34 ≤ x - s * 26986075409 := by
    by_contra hc
    have hrle : x - s * 26986075409 ≤ -(2 : ℤ) ^ 34 := by omega
    have hmul : (x - s * 26986075409) * 683565276
        ≤ (-(2 : ℤ) ^ 34) * 683565276 :=
      mul_le_mul_of_nonneg_right hrle (by norm_num)
    nlinarith [hlow, hmul]
  exact ⟨hlb, hub⟩

/-! ## Reduction value

The post-guard prefix computes `k_wide`, the `i128→i64` cast `k`, the wide product `k · TWO_PI_Q32`,
and `r_wide = x - k · TWO_PI_Q32`, then casts `r`. `word_fixed_sin_cos_reduction_value` returns
`Ok r` (no `OutOfRange`) with the value characterization of `k` and `r`. The two interior `i64`-range
guards are refuted from `fixed_trig_shift_bound` and `fixed_trig_reduction_remainder_range`. The
quadrant-fold outputs `(arg, swap, negate)` are the next value obligation.
-/

/-- Lower bound of the balanced `bmod 2^64`, independent of the value. -/
theorem int_bmod_64_lower (a : ℤ) : -2 ^ 63 ≤ a.bmod (2 ^ 64) := by
  have h := Int.le_bmod (x := a) (m := 2 ^ 64) (by norm_num)
  norm_num at h
  exact h

/-- Upper bound of the balanced `bmod 2^64`, independent of the value. -/
theorem int_bmod_64_upper (a : ℤ) : a.bmod (2 ^ 64) < 2 ^ 63 := by
  have h := Int.bmod_lt (x := a) (m := 2 ^ 64) (by norm_num)
  norm_num at h
  exact h

/-- `bmod 2^64` is the identity on the `i64` range. -/
theorem int_bmod_self_64 (a : ℤ) (hlo : -2 ^ 63 ≤ a) (hhi : a ≤ 2 ^ 63 - 1) :
    a.bmod (2 ^ 64) = a := by
  by_cases h : 0 ≤ a
  · rw [Int.bmod_eq_emod_of_lt]
    · exact Int.emod_eq_of_lt h (by norm_num; omega)
    · rw [Int.emod_eq_of_lt h (by norm_num; omega)]; norm_num; omega
  · rw [Int.bmod_neg]
    · have hem : a % ((2 ^ 64 : ℕ) : ℤ) = a + ((2 ^ 64 : ℕ) : ℤ) := by
        have h1 : (a + ((2 ^ 64 : ℕ) : ℤ)) % ((2 ^ 64 : ℕ) : ℤ)
            = a + ((2 ^ 64 : ℕ) : ℤ) :=
          Int.emod_eq_of_lt (by omega) (by omega)
        have h2 : (a + ((2 ^ 64 : ℕ) : ℤ)) % ((2 ^ 64 : ℕ) : ℤ)
            = a % ((2 ^ 64 : ℕ) : ℤ) := Int.add_emod_right a ((2 ^ 64 : ℕ) : ℤ)
        omega
      rw [hem]; omega
    · have hem : a % ((2 ^ 64 : ℕ) : ℤ) = a + ((2 ^ 64 : ℕ) : ℤ) := by
        have h1 : (a + ((2 ^ 64 : ℕ) : ℤ)) % ((2 ^ 64 : ℕ) : ℤ)
            = a + ((2 ^ 64 : ℕ) : ℤ) :=
          Int.emod_eq_of_lt (by omega) (by omega)
        have h2 : (a + ((2 ^ 64 : ℕ) : ℤ)) % ((2 ^ 64 : ℕ) : ℤ)
            = a % ((2 ^ 64 : ℕ) : ℤ) := Int.add_emod_right a ((2 ^ 64 : ℕ) : ℤ)
        omega
      rw [hem]; norm_num; omega

/-- On the window the post-guard reduction prefix returns `Ok r` (no `OutOfRange`), with `k.val` the
arithmetic shift `(x.val · TWO_PI_RECIP_Q32 + 2^63) >>> 64` and `r.val = x.val - k.val · TWO_PI_Q32`.
Witness `k` is the extracted `IScalar.cast .I64 k_wide`, `r` the extracted
`IScalar.cast .I64 r_wide`. -/
theorem word_fixed_sin_cos_reduction_value (x : I64) (hx : trig_window x) :
    (do
        let i2 ← lift (core.convert.num.FromI128I64.from x)
        let i3 ←
          lift (core.convert.num.FromI128I64.from fixed_trig.TWO_PI_RECIP_Q32)
        let product ← i2 * i3
        let i4 ← 1#i128 <<< 63#i32
        let i5 ← product + i4
        let k_wide ← i5 >>> 64#i32
        let i6 ← lift (IScalar.cast .I128 core.num.I64.MAX)
        if k_wide > i6
        then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
        else
          let i7 ← lift (IScalar.cast .I128 core.num.I64.MIN)
          if k_wide < i7
          then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
          else
            let k ← lift (IScalar.cast .I64 k_wide)
            let i8 ← lift (core.convert.num.FromI128I64.from x)
            let i9 ← lift (core.convert.num.FromI128I64.from k)
            let i10 ←
              lift (core.convert.num.FromI128I64.from fixed_trig.TWO_PI_Q32)
            let i11 ← i9 * i10
            let r_wide ← i8 - i11
            let i12 ← lift (IScalar.cast .I128 core.num.I64.MAX)
            if r_wide > i12
            then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
            else
              let i13 ← lift (IScalar.cast .I128 core.num.I64.MIN)
              if r_wide < i13
              then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
              else
                let r ← lift (IScalar.cast .I64 r_wide)
                ok (core.result.Result.Ok r)) ⦃ z =>
      ∃ k r : I64, z = core.result.Result.Ok r ∧
        k.val = (x.val * 683565276 + 2 ^ 63) >>> 64 ∧
        r.val = x.val - k.val * 26986075409 ⦄ := by
  rw [trig_window] at hx
  simp only [lift, bind_tc_ok]
  step*
  all_goals (try norm_num at *)
  · -- mul x · TWO_PI_RECIP_Q32 lower bound
    norm_num [I128.min, I128.numBits, fixed_trig.TWO_PI_RECIP_Q32] at *
    nlinarith [hx.1, hx.2]
  · -- mul x · TWO_PI_RECIP_Q32 upper bound
    norm_num [I128.max, I128.numBits, fixed_trig.TWO_PI_RECIP_Q32] at *
    nlinarith [hx.1, hx.2]
  · -- i5 = product + i4 upper bound
    have hi4 : (↑i4 : ℤ) = 2 ^ 63 := by
      rw [i4_post]; norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
    rw [product_post, hi4]
    norm_num [I128.max, I128.numBits, fixed_trig.TWO_PI_RECIP_Q32]
    nlinarith [hx.1, hx.2]
  · -- guard: k_wide <= I64.max
    have hsv : k_wide.val = (x.val * 683565276 + 2 ^ 63) >>> 64 := by
      have h1 : (↑k_wide : ℤ) = (↑x * 683565276 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [k_wide_post, i5_post, i4_post, product_post, Int.shiftRight_eq_div_pow]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod, fixed_trig.TWO_PI_RECIP_Q32]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_trig_shift_bound x.val k_wide.val hx.1 hx.2 hsv
    have hkmax : (↑k_wide : ℤ) ≤ 652 := hsb.2
    norm_num [I64.rMax, I64.numBits] at *
    exfalso; omega
  · -- guard: k_wide >= I64.min
    have hsv : k_wide.val = (x.val * 683565276 + 2 ^ 63) >>> 64 := by
      have h1 : (↑k_wide : ℤ) = (↑x * 683565276 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [k_wide_post, i5_post, i4_post, product_post, Int.shiftRight_eq_div_pow]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod, fixed_trig.TWO_PI_RECIP_Q32]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_trig_shift_bound x.val k_wide.val hx.1 hx.2 hsv
    have hkmin : -652 ≤ (↑k_wide : ℤ) := hsb.1
    norm_num [I64.rMin, I64.numBits] at *
    exfalso; omega
  · -- i11 multiplier lower bound
    norm_num [I128.min, I128.numBits, fixed_trig.TWO_PI_Q32] at *
    have h1 := int_bmod_64_lower (k_wide : ℤ)
    have h2 := int_bmod_64_upper (k_wide : ℤ)
    norm_num at h1 h2
    nlinarith [h1, h2]
  · -- i11 multiplier upper bound
    norm_num [I128.max, I128.numBits, fixed_trig.TWO_PI_Q32] at *
    have h1 := int_bmod_64_lower (k_wide : ℤ)
    have h2 := int_bmod_64_upper (k_wide : ℤ)
    norm_num at h1 h2
    nlinarith [h1, h2]
  · -- r_wide lower bound
    rw [i11_post]
    norm_num [I128.min, I128.numBits, fixed_trig.TWO_PI_Q32] at *
    have h1 := int_bmod_64_lower (k_wide : ℤ)
    have h2 := int_bmod_64_upper (k_wide : ℤ)
    norm_num at h1 h2
    nlinarith [h1, h2, hx.1, hx.2]
  · -- r_wide upper bound
    rw [i11_post]
    norm_num [I128.max, I128.numBits, fixed_trig.TWO_PI_Q32] at *
    have h1 := int_bmod_64_lower (k_wide : ℤ)
    have h2 := int_bmod_64_upper (k_wide : ℤ)
    norm_num at h1 h2
    nlinarith [h1, h2, hx.1, hx.2]
  · -- guard: r_wide <= I64.max
    have hsv : k_wide.val = (x.val * 683565276 + 2 ^ 63) >>> 64 := by
      have h1 : (↑k_wide : ℤ) = (↑x * 683565276 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [k_wide_post, i5_post, i4_post, product_post, Int.shiftRight_eq_div_pow]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod, fixed_trig.TWO_PI_RECIP_Q32]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_trig_shift_bound x.val k_wide.val hx.1 hx.2 hsv
    have hsr := fixed_trig_shift_bound_window x.val k_wide.val hx.1 hx.2 hsv
    have hkmod : (↑k_wide : ℤ).bmod 18446744073709551616 = k_wide.val := by
      have h := int_bmod_self_64 k_wide.val (by omega) (by omega)
      simpa [show (18446744073709551616 : ℤ) = 2 ^ 64 by norm_num] using h
    have hrv : r_wide.val = x.val - k_wide.val * 26986075409 := by
      rw [r_wide_post, i11_post]
      norm_num [fixed_trig.TWO_PI_Q32]
      rw [hkmod]
    have hrr := fixed_trig_reduction_remainder_range x.val k_wide.val hsv hsr
    norm_num [I64.rMax, I64.numBits] at *
    exfalso; omega
  · -- guard: r_wide >= I64.min
    have hsv : k_wide.val = (x.val * 683565276 + 2 ^ 63) >>> 64 := by
      have h1 : (↑k_wide : ℤ) = (↑x * 683565276 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [k_wide_post, i5_post, i4_post, product_post, Int.shiftRight_eq_div_pow]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod, fixed_trig.TWO_PI_RECIP_Q32]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_trig_shift_bound x.val k_wide.val hx.1 hx.2 hsv
    have hsr := fixed_trig_shift_bound_window x.val k_wide.val hx.1 hx.2 hsv
    have hkmod : (↑k_wide : ℤ).bmod 18446744073709551616 = k_wide.val := by
      have h := int_bmod_self_64 k_wide.val (by omega) (by omega)
      simpa [show (18446744073709551616 : ℤ) = 2 ^ 64 by norm_num] using h
    have hrv : r_wide.val = x.val - k_wide.val * 26986075409 := by
      rw [r_wide_post, i11_post]
      norm_num [fixed_trig.TWO_PI_Q32]
      rw [hkmod]
    have hrr := fixed_trig_reduction_remainder_range x.val k_wide.val hsv hsr
    norm_num [I64.rMin, I64.numBits] at *
    exfalso; omega
  · -- post: value characterization with k = cast .I64 k_wide
    have hsv : k_wide.val = (x.val * 683565276 + 2 ^ 63) >>> 64 := by
      have h1 : (↑k_wide : ℤ) = (↑x * 683565276 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [k_wide_post, i5_post, i4_post, product_post, Int.shiftRight_eq_div_pow]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod, fixed_trig.TWO_PI_RECIP_Q32]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_trig_shift_bound x.val k_wide.val hx.1 hx.2 hsv
    have hsr := fixed_trig_shift_bound_window x.val k_wide.val hx.1 hx.2 hsv
    have hrr := fixed_trig_reduction_remainder_range x.val k_wide.val hsv hsr
    have hkmod : (↑k_wide : ℤ).bmod 18446744073709551616 = k_wide.val := by
      have h := int_bmod_self_64 k_wide.val (by omega) (by omega)
      simpa [show (18446744073709551616 : ℤ) = 2 ^ 64 by norm_num] using h
    have hrval : r_wide.val = x.val - k_wide.val * 26986075409 := by
      rw [r_wide_post, i11_post]
      norm_num [fixed_trig.TWO_PI_Q32]
      rw [hkmod]
    have hrrv : -2 ^ 36 ≤ r_wide.val ∧ r_wide.val ≤ 2 ^ 36 := by
      rw [hrval]; exact hrr
    have hrmod : (↑r_wide : ℤ).bmod 18446744073709551616 = r_wide.val := by
      have h := int_bmod_self_64 r_wide.val (by omega) (by omega)
      simpa [show (18446744073709551616 : ℤ) = 2 ^ 64 by norm_num] using h
    have hkcast : (↑(IScalar.cast IScalarTy.I64 k_wide) : ℤ) = k_wide.val := by
      rw [IScalar.cast_val_eq]; norm_num
      exact int_bmod_self_64 k_wide.val (by omega) (by omega)
    refine ⟨IScalar.cast IScalarTy.I64 k_wide, ?_, ?_⟩
    · rw [hkcast]
      rw [Int.shiftRight_eq_div_pow] at hsv ⊢
      norm_num at hsv ⊢
      exact hsv
    · rw [hrmod, hrval, hkcast]

/-! ## Quadrant fold

The success branch folds the reduction remainder `r` into `[-pi/2, pi/2]` (`s = r - pi` for
`r > pi/2`, `r + pi` for `r < -pi/2`, else `r`, with `negate` recording the sign flip) and then into
`[-pi/4, pi/4]` by the quadrant swap (`arg = pi/2 - s` with `swap = 1` for `s > pi/4`,
`arg = s + pi/2` with `swap = 2` for `s < -pi/4`, else `arg = s` with `swap = 0`). The quadratic
functions `fixed_trig.fold1`/`fold2` and the flags `fold1_negate`/`fold2_swap` name those values;
`word_fixed_sin_cos_fold_value` is the WP spec of the extracted fold block on `|r| <= 2^36`. The
Taylor call on `arg` is the next slice.
-/

/-- `sin` fold of `r` into `[-pi/2, pi/2]` in Q32.32 units. -/
def fixed_trig.fold1 (r : ℤ) : ℤ :=
  if 6746518852 < r then r - 13493037705
  else if r < -6746518852 then r + 13493037705
  else r

/-- The sign flip recorded by the `sin` fold. -/
def fixed_trig.fold1_negate (r : ℤ) : Bool :=
  if 6746518852 < r then true else if r < -6746518852 then true else false

/-- Quadrant fold of `s` into `[-pi/4, pi/4]` in Q32.32 units. -/
def fixed_trig.fold2 (s : ℤ) : ℤ :=
  if 3373259426 < s then 6746518852 - s
  else if s < -3373259426 then s + 6746518852
  else s

/-- The quadrant index recorded by the `cos`/`sin` swap: `1` for the upper fold, `2` for the lower,
`0` for no swap. -/
def fixed_trig.fold2_swap (s : ℤ) : ℤ :=
  if 3373259426 < s then 1 else if s < -3373259426 then 2 else 0

/-- Fold `r` into `[-HALF_PI, HALF_PI]`; in particular `|fold1 r| <= 2^36` when `|r| <= 2^36`. -/
theorem fixed_trig.fold1_abs_le (r : ℤ) (hr : |r| ≤ 2 ^ 36) :
    |fixed_trig.fold1 r| ≤ 2 ^ 36 := by
  have hr' := abs_le.mp hr
  rw [abs_le]
  unfold fixed_trig.fold1
  split_ifs with h1 h2 <;> omega

/-- Folded argument bound: if `|r| <= 2^34` then `|fold2 (fold1 r)| <= 2^32`, so the Taylor block's
`|arg| <= 2^32` hypothesis is met once the reduction delivers `|r| <= 2^34`
(`fixed_trig_reduction_remainder_range_tight`). `2^34` is necessary: `|r| = π·2^32` is reachable. -/
theorem fixed_trig.fold_arg_abs_le (r : ℤ) (hr : |r| ≤ 2 ^ 34) :
    |fixed_trig.fold2 (fixed_trig.fold1 r)| ≤ 2 ^ 32 := by
  have hr' := abs_le.mp hr
  rw [abs_le]
  unfold fixed_trig.fold1 fixed_trig.fold2
  split_ifs <;> omega

/-- Tight folded-argument bound: with `|r| <= 2^34` the two folds land `arg` in
`[-QUARTER_PI, QUARTER_PI]` (`3373259426 = floor(pi/4 * 2^32)`), so `|toReal arg| < pi/4`. This is
what makes the degree-7/8 Taylor remainder small enough for the `2^-20` floor; the loose
`fold_arg_abs_le` (`|arg| <= 2^32`) is only used to meet the Taylor block's hypothesis. -/
theorem fixed_trig.fold_arg_abs_le_quarter (r : ℤ) (hr : |r| ≤ 2 ^ 34) :
    |fixed_trig.fold2 (fixed_trig.fold1 r)| ≤ 3373259426 := by
  have hr' := abs_le.mp hr
  rw [abs_le]
  unfold fixed_trig.fold1 fixed_trig.fold2
  split_ifs <;> omega

/-- WP spec of the extracted `sin` fold on its own, before the quadrant swap. -/
theorem fixed_trig_fold1_spec (r : I64) (hr : |r.val| ≤ 2 ^ 36) :
    (if r > fixed_trig.HALF_PI_Q32
     then do
          let i14 ← r - fixed_trig.PI_Q32
          ok (i14, true)
     else
       do
       let i14 ← -. fixed_trig.HALF_PI_Q32
       if r < i14
       then let i15 ← r + fixed_trig.PI_Q32
            ok (i15, true)
       else ok (r, false)) ⦃ z =>
      z.1.val = fixed_trig.fold1 r.val ∧ z.2 = fixed_trig.fold1_negate r.val ⦄ := by
  have hr' := abs_le.mp hr
  by_cases h1 : r > fixed_trig.HALF_PI_Q32
  · rw [if_pos h1]
    step*
    · -- lower bound: r - pi
      norm_num [fixed_trig.PI_Q32, I64.min, I64.numBits] at *
      omega
    · -- upper bound: r - pi
      norm_num [fixed_trig.PI_Q32, I64.max, I64.numBits] at *
      omega
    · -- upper branch value
      have h1z : (6746518852 : ℤ) < r.val := by
        have h := h1
        rw [gt_iff_lt, IScalar.lt_equiv] at h
        simp only [fixed_trig.HALF_PI_Q32] at h
        norm_num at h
        exact h
      constructor
      · rw [i14_post, fixed_trig.fold1, if_pos h1z]
        norm_num [fixed_trig.PI_Q32]
      · rw [fixed_trig.fold1_negate, if_pos h1z]
  · rw [if_neg h1]
    step*
    · -- lower bound: r + pi
      norm_num [fixed_trig.PI_Q32, I64.min, I64.numBits] at *
      omega
    · -- lower branch value
      rename_i hb
      have h2z : r.val < -6746518852 := by
        have h := hb
        rw [IScalar.lt_equiv] at h
        rw [i14_post] at h
        simp only [fixed_trig.HALF_PI_Q32] at h
        norm_num at h
        exact h
      constructor
      · rw [i15_post, fixed_trig.fold1, if_neg (by omega), if_pos h2z]
        norm_num [fixed_trig.PI_Q32]
      · rw [fixed_trig.fold1_negate, if_neg (by omega), if_pos h2z]
    · -- no-fold value
      rename_i hb
      have hle : r.val ≤ 6746518852 := by
        have h := h1
        rw [gt_iff_lt, not_lt] at h
        rw [IScalar.le_equiv] at h
        simp only [fixed_trig.HALF_PI_Q32] at h
        norm_num at h
        exact h
      have hge : -6746518852 ≤ r.val := by
        have h := hb
        rw [not_lt] at h
        rw [IScalar.le_equiv] at h
        rw [i14_post] at h
        simp only [fixed_trig.HALF_PI_Q32] at h
        norm_num at h
        exact h
      constructor
      · rw [fixed_trig.fold1, if_neg (by omega), if_neg (by omega)]
      · rw [fixed_trig.fold1_negate, if_neg (by omega), if_neg (by omega)]

/-- WP spec of the extracted quadrant swap on its own. -/
theorem fixed_trig_fold2_spec (s : I64) (hs : |s.val| ≤ 2 ^ 36) :
    (if s > fixed_trig.QUARTER_PI_Q32
     then do
          let i14 ← fixed_trig.HALF_PI_Q32 - s
          ok (i14, 1#i32)
     else
       do
       let i14 ← -. fixed_trig.QUARTER_PI_Q32
       if s < i14
       then let i15 ← s + fixed_trig.HALF_PI_Q32
            ok (i15, 2#i32)
       else ok (s, 0#i32)) ⦃ z =>
      z.1.val = fixed_trig.fold2 s.val ∧ z.2.val = fixed_trig.fold2_swap s.val ⦄ := by
  have hs' := abs_le.mp hs
  by_cases h1 : s > fixed_trig.QUARTER_PI_Q32
  · rw [if_pos h1]
    step*
    · -- lower bound: pi/2 - s
      norm_num [fixed_trig.HALF_PI_Q32, I64.min, I64.numBits] at *
      omega
    · -- upper bound: pi/2 - s
      norm_num [fixed_trig.HALF_PI_Q32, I64.max, I64.numBits] at *
      omega
    · -- upper branch value
      have h1z : (3373259426 : ℤ) < s.val := by
        have h := h1
        rw [gt_iff_lt, IScalar.lt_equiv] at h
        simp only [fixed_trig.QUARTER_PI_Q32] at h
        norm_num at h
        exact h
      constructor
      · rw [i14_post, fixed_trig.fold2, if_pos h1z]
        norm_num [fixed_trig.HALF_PI_Q32, fixed_trig.QUARTER_PI_Q32]
      · rw [fixed_trig.fold2_swap, if_pos h1z]
        norm_num [fixed_trig.QUARTER_PI_Q32]
  · rw [if_neg h1]
    step*
    · -- lower bound: s + pi/2
      norm_num [fixed_trig.HALF_PI_Q32, I64.min, I64.numBits] at *
      omega
    · -- lower branch value
      rename_i hb
      have h2z : s.val < -3373259426 := by
        have h := hb
        rw [IScalar.lt_equiv] at h
        rw [i14_post] at h
        simp only [fixed_trig.QUARTER_PI_Q32] at h
        norm_num at h
        exact h
      constructor
      · rw [i15_post, fixed_trig.fold2, if_neg (by omega), if_pos h2z]
        norm_num [fixed_trig.HALF_PI_Q32, fixed_trig.QUARTER_PI_Q32]
      · rw [fixed_trig.fold2_swap, if_neg (by omega), if_pos h2z]
        norm_num [fixed_trig.QUARTER_PI_Q32]
    · -- no-swap value
      rename_i hb
      have hle : s.val ≤ 3373259426 := by
        have h := h1
        rw [gt_iff_lt, not_lt] at h
        rw [IScalar.le_equiv] at h
        simp only [fixed_trig.QUARTER_PI_Q32] at h
        norm_num at h
        exact h
      have hge : -3373259426 ≤ s.val := by
        have h := hb
        rw [not_lt] at h
        rw [IScalar.le_equiv] at h
        rw [i14_post] at h
        simp only [fixed_trig.QUARTER_PI_Q32] at h
        norm_num at h
        exact h
      constructor
      · rw [fixed_trig.fold2, if_neg (by omega), if_neg (by omega)]
      · rw [fixed_trig.fold2_swap, if_neg (by omega), if_neg (by omega)]
        norm_num

/-- WP spec of the extracted quadrant-fold block on `|r.val| <= 2^36`: no refusal, and the returned
`s`, `negate`, `arg`, `swap` are exactly `fold1`, `fold1_negate`, `fold2`, `fold2_swap`. -/
theorem word_fixed_sin_cos_fold_value (r : I64) (hr : |r.val| ≤ 2 ^ 36) :
    (do
        let (s, negate) ←
          if r > fixed_trig.HALF_PI_Q32
          then do
               let i14 ← r - fixed_trig.PI_Q32
               ok (i14, true)
          else
            do
            let i14 ← -. fixed_trig.HALF_PI_Q32
            if r < i14
            then let i15 ← r + fixed_trig.PI_Q32
                 ok (i15, true)
            else ok (r, false)
        let (arg, swap) ←
          if s > fixed_trig.QUARTER_PI_Q32
          then do
               let i14 ← fixed_trig.HALF_PI_Q32 - s
               ok (i14, 1#i32)
          else
            do
            let i14 ← -. fixed_trig.QUARTER_PI_Q32
            if s < i14
            then let i15 ← s + fixed_trig.HALF_PI_Q32
                 ok (i15, 2#i32)
            else ok (s, 0#i32)
        ok (r, s, negate, arg, swap)) ⦃ z =>
      ∃ s arg : I64, ∃ negate : Bool, ∃ swap : I32,
        z = (r, s, negate, arg, swap) ∧
        s.val = fixed_trig.fold1 r.val ∧
        negate = fixed_trig.fold1_negate r.val ∧
        arg.val = fixed_trig.fold2 (fixed_trig.fold1 r.val) ∧
        swap.val = fixed_trig.fold2_swap (fixed_trig.fold1 r.val) ⦄ := by
  apply Aeneas.Std.WP.spec_bind (fixed_trig_fold1_spec r hr)
  intro p hp
  obtain ⟨s, negate⟩ := p
  have hs : |s.val| ≤ 2 ^ 36 := by
    rw [hp.1]
    exact fixed_trig.fold1_abs_le r.val hr
  apply Aeneas.Std.WP.spec_bind (fixed_trig_fold2_spec s hs)
  intro q hq
  obtain ⟨arg, swap⟩ := q
  exact ⟨s, arg, negate, swap, rfl, hp.1, hp.2, by rw [hq.1, hp.1], by rw [hq.2, hp.1]⟩

/-- The extracted quadrant-fold block as a named call, so `step*` can treat it as a specified step.
Defeq to the inline fold block. -/
def fixed_trig.fold_step (r : I64) : Result (I64 × I64 × Bool × I64 × I32) :=
  (do
    let (s, negate) ←
      if r > fixed_trig.HALF_PI_Q32
      then do
           let i14 ← r - fixed_trig.PI_Q32
           ok (i14, true)
      else
        do
        let i14 ← -. fixed_trig.HALF_PI_Q32
        if r < i14
        then let i15 ← r + fixed_trig.PI_Q32
             ok (i15, true)
        else ok (r, false)
    let (arg, swap) ←
      if s > fixed_trig.QUARTER_PI_Q32
      then do
           let i14 ← fixed_trig.HALF_PI_Q32 - s
           ok (i14, 1#i32)
      else
        do
        let i14 ← -. fixed_trig.QUARTER_PI_Q32
        if s < i14
        then let i15 ← s + fixed_trig.HALF_PI_Q32
             ok (i15, 2#i32)
        else ok (s, 0#i32)
    ok (r, s, negate, arg, swap))

/-- WP spec of the named fold block, restating `word_fixed_sin_cos_fold_value` on `fold_step`. -/
theorem fixed_trig_fold_step_spec (r : I64) (hr : |r.val| ≤ 2 ^ 36) :
    fixed_trig.fold_step r ⦃ z =>
      ∃ s arg : I64, ∃ negate : Bool, ∃ swap : I32,
        z = (r, s, negate, arg, swap) ∧
        s.val = fixed_trig.fold1 r.val ∧
        negate = fixed_trig.fold1_negate r.val ∧
        arg.val = fixed_trig.fold2 (fixed_trig.fold1 r.val) ∧
        swap.val = fixed_trig.fold2_swap (fixed_trig.fold1 r.val) ⦄ := by
  unfold fixed_trig.fold_step
  exact word_fixed_sin_cos_fold_value r hr

/-- Composed WP spec of the extracted post-guard body through the quadrant fold, in `Result`
bind/match form: the reduction prefix `word_fixed_sin_cos_reduction_value` returns `Ok r`, then the
fold block (defeq to the inline fold) returns `(r, fold1 r, fold1_negate r, fold2 (fold1 r),
fold2_swap (fold1 r))`. The ranges are supplied by `fixed_trig_shift_bound_window` and
`fixed_trig_reduction_remainder_range`; the Taylor call is the next slice. -/
theorem word_fixed_sin_cos_reduction_fold_value (x : I64) (hx : trig_window x) :
    (do
        let res ← (do
          let i2 ← lift (core.convert.num.FromI128I64.from x)
          let i3 ←
            lift (core.convert.num.FromI128I64.from fixed_trig.TWO_PI_RECIP_Q32)
          let product ← i2 * i3
          let i4 ← 1#i128 <<< 63#i32
          let i5 ← product + i4
          let k_wide ← i5 >>> 64#i32
          let i6 ← lift (IScalar.cast .I128 core.num.I64.MAX)
          if k_wide > i6
          then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
          else
            let i7 ← lift (IScalar.cast .I128 core.num.I64.MIN)
            if k_wide < i7
            then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
            else
              let k ← lift (IScalar.cast .I64 k_wide)
              let i8 ← lift (core.convert.num.FromI128I64.from x)
              let i9 ← lift (core.convert.num.FromI128I64.from k)
              let i10 ←
                lift (core.convert.num.FromI128I64.from fixed_trig.TWO_PI_Q32)
              let i11 ← i9 * i10
              let r_wide ← i8 - i11
              let i12 ← lift (IScalar.cast .I128 core.num.I64.MAX)
              if r_wide > i12
              then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
              else
                let i13 ← lift (IScalar.cast .I128 core.num.I64.MIN)
                if r_wide < i13
                then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
                else
                  let r ← lift (IScalar.cast .I64 r_wide)
                  ok (core.result.Result.Ok r))
        match res with
        | core.result.Result.Err e => ok (core.result.Result.Err e)
        | core.result.Result.Ok r => do
            let u ← fixed_trig.fold_step r
            ok (core.result.Result.Ok u)) ⦃ z =>
      ∃ k r s arg : I64, ∃ negate : Bool, ∃ swap : I32,
        z = core.result.Result.Ok (r, s, negate, arg, swap) ∧
        k.val = (x.val * 683565276 + 2 ^ 63) >>> 64 ∧
        r.val = x.val - k.val * 26986075409 ∧
        s.val = fixed_trig.fold1 r.val ∧
        negate = fixed_trig.fold1_negate r.val ∧
        arg.val = fixed_trig.fold2 (fixed_trig.fold1 r.val) ∧
        swap.val = fixed_trig.fold2_swap (fixed_trig.fold1 r.val) ⦄ := by
  apply Aeneas.Std.WP.spec_bind (word_fixed_sin_cos_reduction_value x hx)
  intro v hv
  obtain ⟨k, r, rfl, hk, hr⟩ := hv
  dsimp only
  have hsr := fixed_trig_shift_bound_window x.val k.val hx.1 hx.2 hk
  have hrr := fixed_trig_reduction_remainder_range x.val k.val hk hsr
  have hr36 : |r.val| ≤ 2 ^ 36 := by
    rw [abs_le]
    exact ⟨by rw [hr]; exact hrr.1, by rw [hr]; exact hrr.2⟩
  apply Aeneas.Std.WP.spec_bind (fixed_trig_fold_step_spec r hr36)
  intro u hu
  obtain ⟨s, arg, negate, swap, rfl, hs, hneg, harg, hswap⟩ := hu
  exact ⟨k, r, s, arg, negate, swap, rfl, hk, hr, hs, hneg, harg, hswap⟩


end

end apeiron_word
