/-
Refinement of the fixed-point `sin`/`cos` body against the real trigonometric functions
(`APEIRON-INT-TRANSCENDENTAL-W2-T1`; spec `docs/epics/inference-parity.md`).

The Rust body is `src/numeric/fixed_trig.rs`: `i64` Q32.32, window `[-4096, 4096]` radians,
range reduction by `k = q_rne(x << 32, HALF_PI_Q64)` and `r = q_rne(x << 32 - k·HALF_PI_Q64, 2^32)`,
a Taylor series through `r^13/13!` and `r^12/12!`, and a quadrant swap/negate on `k % 4`; `Result`
refusal outside the window. It is extracted into `ApeironWord` by `proof/regen.sh`.

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

The relative term is the `2^-20` target. The absolute floor `trig_floor = 2^-20` makes the bound
one-sided-sufficient: both right-hand sides are at least `2^-20`, so the refinement follows from the
absolute budget `2^-20` proven in `Numeric.Rope.Part4`. The measured error of the body is `1.1e-9`
relative; the proved absolute budget is `2e-9`, so the statement is loose by three orders of
magnitude and every analytic step below is deliberately coarse.
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

/-- The absolute floor near the zeros of the two outputs, `2^-20`. -/
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
structure alone, parameterised by the values of the extracted endpoint `const`s.
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

/-- Refusal correspondence proved: outside `trig_window` the extracted body returns
`Err OutOfRange`. -/
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

/-! ## Success branch: window guards pass -/

/-- On the window the extracted body's lower guard is false. -/
theorem fixed_trig_lower_test_false (x : I64) (hx : trig_window x) :
    ¬ x < ((-17592186044416)#i64) := by
  intro h
  rw [IScalar.lt_equiv] at h
  exact not_lt.mpr hx.1 h

/-- On the window the extracted body's upper guard is false. -/
theorem fixed_trig_upper_test_false (x : I64) (hx : trig_window x) :
    ¬ x > (17592186044416#i64) := by
  intro h
  rw [gt_iff_lt] at h
  rw [IScalar.lt_equiv] at h
  exact not_lt.mpr hx.2 h

/-- On the window both endpoint guards fall through to the reduction-and-series tail. -/
def trigPastGuards (x : I64) :
    Result (core.result.Result fixed_trig.SinCos fixed_trig.TrigError) :=
      (do
        let i ← lift (core.convert.num.FromI128I64.from x)
        let t64 ← i <<< 32#i32
        let o ← fixed_trig.q_rne t64 fixed_trig.HALF_PI_Q64
        match o with
        | none => ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
        | some v =>
          let i1 ← v * fixed_trig.HALF_PI_Q64
          let i2 ← t64 - i1
          let i3 ← 1#i128 <<< 32#i32
          let o1 ← fixed_trig.q_rne i2 i3
          match o1 with
          | none => ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
          | some v1 =>
            let o2 ← fixed_trig.narrow v1
            match o2 with
            | none => ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
            | some v2 =>
              let r ← fixed_trig.sin_cos_series v2
              match r with
              | core.result.Result.Ok v3 =>
                let i4 ← v % 4#i128
                let i5 ← i4 + 4#i128
                let quarter ← i5 % 4#i128
                match quarter with
                | 0#iscalar => ok r
                | 1#iscalar =>
                  let i6 ← -. v3.sin
                  ok (core.result.Result.Ok { sin := v3.cos, cos := i6 })
                | 2#iscalar =>
                  let i6 ← -. v3.sin
                  let i7 ← -. v3.cos
                  ok (core.result.Result.Ok { sin := i6, cos := i7 })
                | _ =>
                  let i6 ← -. v3.cos
                  ok (core.result.Result.Ok { sin := i6, cos := v3.sin })
              | core.result.Result.Err _ => ok r)

theorem word_fixed_sin_cos_window_past_guards (x : I64) (hx : trig_window x) :
    word_fixed_sin_cos x = trigPastGuards x := by
  rw [word_fixed_sin_cos, fixed_trig.fixed_sin_cos, fixed_trig_LOWER_eq, fixed_trig_UPPER_eq,
    trigPastGuards]
  simp only [bind_tc_ok, if_neg (fixed_trig_lower_test_false x hx),
    if_neg (fixed_trig_upper_test_false x hx)]
  rfl

end

end apeiron_word
