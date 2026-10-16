/-
  `Serve/Mixer3.lean` — the positive refinement theorems for `src/engine/mixer.rs` that `Serve/Mixer.lean`
  and `Serve/Mixer2.lean` leave open: `mixer.div_q32` (the whole body, both arms), the
  `mixer.delta_head_row` value theorem with its `div_q32` leaf discharged, and `mixer.l2_normalize`
  (the `isqrt` scale and the two loops).

  References are plain-Lean `Int` definitions over the *meaning* (shift-left, half bias, divide,
  integer square root, round), never a restatement of the Rust loop.  Names are file-local (`mixer3_`)
  so this module registers beside `Serve/Mixer.lean` and `Serve/Mixer2.lean`.

  Boundary: the `den <= 0` arm of `mixer.div_q32` converts a `&str` message, so every theorem whose
  statement mentions `mixer.div_q32` carries `alloc.string.String.Insts.CoreConvertFromShared0Str.from`
  (an opaque host boundary, `proof/boundary-axioms.txt`) in its `#print axioms` cone.  That axiom is in
  the *definition's* closure, not in the proof; the arithmetic the theorems state is independent of it.
-/

import Serve.Block.Mixer2

set_option autoImplicit false
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## §1 · References and numeric helpers -/

/-- Q32.32 quotient reference: the numerator shifted up by `32` with a half-divisor bias, divided by
`den`, and negated for a negative numerator (the Rust body biases, divides toward zero, and negates). -/
def mixer3_divQ32Ref (num den : Int) : Int :=
  if 0 ≤ num then Int.ediv (num * 4294967296 + Int.ediv den 2) den
  else -Int.ediv ((-num) * 4294967296 + Int.ediv den 2) den

theorem mixer3_divQ32Ref_eq : mixer3_divQ32Ref = mixer2_divQ32Ref := rfl

/-- The literal forms of the two `i128` sentinels. -/
theorem mixer3_i128Min_eq : mixer2_i128Min = -(2 : ℤ) ^ (127 : ℕ) := by
  simp only [mixer2_i128Min]
  norm_num

theorem mixer3_i128Max_eq : mixer2_i128Max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
  simp only [mixer2_i128Max]
  norm_num

/-- Every `i64` word lies in `[-2^63, 2^63]`. -/
theorem mixer3_i64_abs_le (a : Std.I64) : |a.val| ≤ (2 : ℤ) ^ (63 : ℕ) := by
  obtain ⟨h1, h2⟩ := mixer2_i64_val_bounds a
  rw [abs_le]
  constructor
  · have h : -(2 : ℤ) ^ (63 : ℕ) = -9223372036854775808 := by norm_num
    rw [h]; exact h1
  · have h : (2 : ℤ) ^ (63 : ℕ) = 9223372036854775808 := by norm_num
    rw [h]; omega

/-- `i64 → i128` has the integer value. -/
theorem mixer3_from_val (a : Std.I64) : (core.convert.num.FromI128I64.from a).val = a.val :=
  core.convert.num.FromI128I64.from_val_eq a

/-- `0 ≤ a · 2^32` gives `0 ≤ a`. -/
theorem mixer3_nonneg_of_mul_nonneg {a : ℤ} (h : 0 ≤ a * 4294967296) : 0 ≤ a := by
  by_contra hneg
  push Not at hneg
  have hc : (0 : ℤ) < 4294967296 := by norm_num
  have hm := mul_neg_of_neg_of_pos hneg hc
  linarith

/-- `a · 2^32 < 0` gives `a < 0`. -/
theorem mixer3_neg_of_mul_neg {a : ℤ} (h : a * 4294967296 < 0) : a < 0 := by
  by_contra h'
  push Not at h'
  have hc : (0 : ℤ) < 4294967296 := by norm_num
  have hm := mul_nonneg h' (le_of_lt hc)
  linarith

/-- `x ≥ 0` for an `i128` is the value comparison. -/
theorem mixer3_i128_ge_zero (x : Std.I128) : (x ≥ (0#i128)) ↔ (0 : ℤ) ≤ x.val := by
  rw [ge_iff_le, IScalar.le_equiv]
  simp

/-- The `i128` left shift by `32` is the multiplication by `2^32`: the `128`-bit container does not
wrap a value below `2^96`. -/
theorem mixer3_shiftLeft32_val (i : Std.I128) (h : |i.val| ≤ (2 : ℤ) ^ (95 : ℕ) - 1) :
    ∃ z : Std.I128, i <<< (32#i32) = ok z ∧ z.val = i.val * 4294967296 := by
  obtain ⟨z, hz, hzv, -⟩ := WP.spec_imp_exists
    (IScalar.ShiftLeft_IScalar_spec (x := i) (y := (32#i32)) (by norm_num)
      (by norm_num [I128.numBits]))
  refine ⟨z, hz, ?_⟩
  obtain ⟨hl, hu⟩ := abs_le.mp h
  have hlo : mixer2_i128Min ≤ i.val * 2^32 := by
    rw [mixer3_i128Min_eq]
    have hm := mul_le_mul_of_nonneg_right hl (by norm_num : (0 : ℤ) ≤ 2^32)
    have h2 : -(2 : ℤ) ^ (127 : ℕ) ≤ -((2 : ℤ) ^ (95 : ℕ) - 1) * 2^32 := by norm_num
    linarith [hm, h2]
  have hhi : i.val * 2^32 ≤ mixer2_i128Max := by
    rw [mixer3_i128Max_eq]
    have hm := mul_le_mul_of_nonneg_right hu (by norm_num : (0 : ℤ) ≤ 2^32)
    have h2 : ((2 : ℤ) ^ (95 : ℕ) - 1) * 2^32 ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by norm_num
    linarith [hm, h2]
  rw [hzv]
  rw [show (32#i32).toNat = 32 from by decide, Int.shiftLeft_eq]
  rw [show IScalarTy.I128.numBits = 128 from by decide]
  rw [mixer2_bmod_i128 (x := i.val * 2^32) hlo hhi]
  norm_num

/-- The `i64` arithmetic right shift by `1` is the floor half. -/
theorem mixer3_shiftRight1_val (d : Std.I64) :
    ∃ z : Std.I64, d >>> (1#i32) = ok z ∧ z.val = Int.ediv d.val 2 := by
  obtain ⟨z, hz, hzv, -⟩ := WP.spec_imp_exists
    (IScalar.ShiftRight_IScalar_spec (x := d) (y := (1#i32)) (by norm_num)
      (by norm_num [I64.numBits]))
  refine ⟨z, hz, ?_⟩
  rw [hzv, show (1#i32).toNat = 1 from by decide, Int.shiftRight_eq_div_pow]
  simp only [pow_one, Nat.cast_ofNat]
  rfl

/-- The `i128` negation of an in-range value. -/
theorem mixer3_neg_val (x : Std.I128) (hlo : IScalar.min IScalarTy.I128 ≤ -x.val)
    (hhi : -x.val ≤ IScalar.max IScalarTy.I128) :
    ∃ z : Std.I128, -.x = ok z ∧ z.val = -x.val := by
  have hne : (↑x : ℤ) ≠ IScalar.min IScalarTy.I128 := by
    intro hc
    rw [mixer2_i128_max_eq, mixer3_i128Max_eq] at hhi
    rw [hc, mixer2_i128_min_eq, mixer3_i128Min_eq] at hhi
    linarith [hhi]
  obtain ⟨z, hz, hzv⟩ := WP.spec_imp_exists (HNeg.hNeg.step (x := x) hne)
  exact ⟨z, hz, hzv⟩

/-! ## §2 · `mixer.div_q32` — the biased Q32.32 quotient

`div_q32 num den` is `num·2^32 / den` with a half-divisor bias, rounded toward zero and negated for a
negative numerator, narrowed to `i64`.  It refuses on `den ≤ 0` (`BadConfig`) and on a quotient outside
`[i64::MIN, i64::MAX]` (`Overflow`). -/

/-- **`div_q32`, success arm.** For a positive divisor and a reference value inside `i64`, the body
returns that value. `proved`, relative to the `Str → String` boundary in the `den ≤ 0` arm. -/
theorem mixer3_div_q32_val (num den : Std.I64)
    (hden : 0 < den.val)
    (hfit : I64.min ≤ mixer3_divQ32Ref num.val den.val ∧
      mixer3_divQ32Ref num.val den.val ≤ I64.max) :
    ∃ w : Std.I64, mixer.div_q32 num den = ok (.Ok w) ∧
      w.val = mixer3_divQ32Ref num.val den.val := by
  have h95abs : |num.val| ≤ (2 : ℤ) ^ (95 : ℕ) - 1 := by
    have h := mixer3_i64_abs_le num
    have h2 : (2 : ℤ) ^ (63 : ℕ) ≤ (2 : ℤ) ^ (95 : ℕ) - 1 := by norm_num
    linarith
  obtain ⟨h95l, h95u⟩ := abs_le.mp h95abs
  have hnumhi := (mixer2_i64_val_bounds num).2
  unfold mixer.div_q32
  rw [if_neg (by rw [IScalar.le_equiv]; exact not_le.mpr hden)]
  simp only [lift, bind_tc_ok]
  set i : Std.I128 := core.convert.num.FromI128I64.from num with hi
  have hiv : i.val = num.val := mixer3_from_val num
  obtain ⟨num128, hshift, hshiftv⟩ := mixer3_shiftLeft32_val i (by rw [hiv]; exact h95abs)
  rw [hshift]
  simp only [bind_tc_ok]
  have hnum128 : num128.val = num.val * 4294967296 := by rw [hshiftv, hiv]
  obtain ⟨i1, hi1, hi1v⟩ := mixer3_shiftRight1_val den
  rw [hi1]
  simp only [bind_tc_ok]
  set den128 : Std.I128 := core.convert.num.FromI128I64.from den with hden128
  have hden128v : den128.val = den.val := mixer3_from_val den
  have hfrm1 : (core.convert.num.FromI128I64.from i1).val = i1.val := mixer3_from_val i1
  have hi1nn : 0 ≤ i1.val := by
    rw [hi1v]
    exact Int.ediv_nonneg (by omega) (by norm_num)
  have hi1le : i1.val ≤ (2 : ℤ) ^ (63 : ℕ) := by
    have hdenhi : den.val ≤ 9223372036854775807 := (mixer2_i64_val_bounds den).2
    rw [hi1v]
    have h2 : Int.ediv den.val 2 ≤ den.val := Int.ediv_le_self 2 (by omega)
    have h3 : (2 : ℤ) ^ (63 : ℕ) = 9223372036854775808 := by norm_num
    rw [h3]
    omega
  have hnum128lo : -(2 : ℤ) ^ (95 : ℕ) ≤ num128.val := by
    rw [hnum128]
    have hnlo : -(2 : ℤ) ^ (63 : ℕ) ≤ num.val := by
      have h := (mixer2_i64_val_bounds num).1
      have h2 : (2 : ℤ) ^ (63 : ℕ) = 9223372036854775808 := by norm_num
      linarith [h, h2]
    have hm := mul_le_mul_of_nonneg_right hnlo (by norm_num : (0 : ℤ) ≤ 4294967296)
    have h2 : -(2 : ℤ) ^ (63 : ℕ) * 4294967296 = -(2 : ℤ) ^ (95 : ℕ) := by norm_num
    linarith [hm, h2]
  have hnum128hi : num128.val ≤ (2 : ℤ) ^ (95 : ℕ) := by
    rw [hnum128]
    have hu : num.val ≤ (2 : ℤ) ^ (63 : ℕ) := by
      have h : (2 : ℤ) ^ (63 : ℕ) = 9223372036854775808 := by norm_num
      rw [h]; omega
    have hm := mul_le_mul_of_nonneg_right hu (by norm_num : (0 : ℤ) ≤ 4294967296)
    have h2 : (2 : ℤ) ^ (63 : ℕ) * 4294967296 = (2 : ℤ) ^ (95 : ℕ) := by norm_num
    linarith [hm, h2]
  by_cases hge : num128 ≥ (0#i128)
  · have hnn : (0 : ℤ) ≤ num128.val := (mixer3_i128_ge_zero num128).1 hge
    have hnn' : (0 : ℤ) ≤ num.val := by
      rw [hnum128] at hnn
      exact mixer3_nonneg_of_mul_nonneg hnn
    rw [if_pos hge]
    have hmin : IScalar.min IScalarTy.I128 ≤
        num128.val + (core.convert.num.FromI128I64.from i1).val := by
      rw [hfrm1, mixer2_i128_min_eq, mixer3_i128Min_eq]
      have h1 : -(2 : ℤ) ^ (95 : ℕ) ≤ num128.val := hnum128lo
      have h2 : -(2 : ℤ) ^ (127 : ℕ) ≤ -(2 : ℤ) ^ (95 : ℕ) := by norm_num
      linarith [h1, hi1nn, h2]
    have hmax : num128.val + (core.convert.num.FromI128I64.from i1).val ≤
        IScalar.max IScalarTy.I128 := by
      rw [hfrm1, mixer2_i128_max_eq, mixer3_i128Max_eq]
      have h1 : num128.val ≤ (2 : ℤ) ^ (95 : ℕ) := hnum128hi
      have h2 : (2 : ℤ) ^ (95 : ℕ) + (2 : ℤ) ^ (63 : ℕ) ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by
        norm_num
      linarith [h1, hi1le, h2]
    obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists
      (IScalar.add_spec (x := num128) (y := core.convert.num.FromI128I64.from i1) hmin hmax)
    rw [hi2]
    simp only [bind_tc_ok]
    obtain ⟨q, hq, hqv⟩ := IScalar.div_spec (x := i2) (y := den128)
      (by rw [hden128v]; omega)
      (by intro hc; rw [hden128v] at hc; omega)
    rw [hq]
    simp only [bind_tc_ok]
    have hqval : q.val = mixer3_divQ32Ref num.val den.val := by
      rw [hqv, hi2v, hfrm1, hnum128, hi1v, hden128v, mixer3_divQ32Ref, if_pos hnn']
      have hA : (0 : ℤ) ≤ num.val * 4294967296 + Int.ediv den.val 2 := by
        have h1 : (0 : ℤ) ≤ num.val * 4294967296 := mul_nonneg hnn' (by norm_num)
        have h2 : (0 : ℤ) ≤ Int.ediv den.val 2 := Int.ediv_nonneg (by omega) (by norm_num)
        linarith
      rw [Int.tdiv_eq_ediv_of_nonneg hA]
      rfl
    have hfitq : I64.min ≤ q.val ∧ q.val ≤ I64.max := by rw [hqval]; exact hfit
    obtain ⟨w, hw, hwv⟩ := mixer2_try_from_val q hfitq.1 hfitq.2
    refine ⟨w, ?_, ?_⟩
    · rw [hw]
      simp only [bind_tc_ok, core.result.Result.map_err_ok]
    · rw [hwv, hqval]
  · have hneg : num.val < 0 := by
      have hneg' : num128.val < 0 := by
        have h : ¬ ((0 : ℤ) ≤ num128.val) := fun hc => hge ((mixer3_i128_ge_zero num128).2 hc)
        omega
      rw [hnum128] at hneg'
      exact mixer3_neg_of_mul_neg hneg'
    have hnneg : ¬ ((0 : ℤ) ≤ num.val) := by omega
    have hnum128neg : num128.val < 0 := by rw [hnum128]; exact mul_neg_of_neg_of_pos hneg (by norm_num)
    rw [if_neg hge]
    have hmin_neg : IScalar.min IScalarTy.I128 ≤ -num128.val := by
      rw [mixer2_i128_min_eq, mixer3_i128Min_eq]
      have h4 : -(2 : ℤ) ^ (127 : ℕ) ≤ -(2 : ℤ) ^ (95 : ℕ) := by norm_num
      linarith [h4, hnum128hi]
    have hmax_neg : -num128.val ≤ IScalar.max IScalarTy.I128 := by
      rw [mixer2_i128_max_eq, mixer3_i128Max_eq]
      have h2 : (2 : ℤ) ^ (95 : ℕ) ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by norm_num
      linarith [hnum128lo, h2]
    obtain ⟨i2, hi2, hi2v⟩ := mixer3_neg_val num128 hmin_neg hmax_neg
    rw [hi2]
    simp only [bind_tc_ok]
    have hi2val : i2.val = -num128.val := hi2v
    have hi2nn : 0 ≤ i2.val := by rw [hi2val]; omega
    have hi2hi : i2.val ≤ (2 : ℤ) ^ (95 : ℕ) := by rw [hi2val]; omega
    have hmin2 : IScalar.min IScalarTy.I128 ≤
        i2.val + (core.convert.num.FromI128I64.from i1).val := by
      rw [hfrm1, mixer2_i128_min_eq, mixer3_i128Min_eq]
      have h2 : -(2 : ℤ) ^ (127 : ℕ) ≤ -(2 : ℤ) ^ (95 : ℕ) := by norm_num
      linarith [hi2nn, hi1nn, h2]
    have hmax2 : i2.val + (core.convert.num.FromI128I64.from i1).val ≤
        IScalar.max IScalarTy.I128 := by
      rw [hfrm1, mixer2_i128_max_eq, mixer3_i128Max_eq]
      have h2 : (2 : ℤ) ^ (95 : ℕ) + (2 : ℤ) ^ (63 : ℕ) ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by
        norm_num
      linarith [hi2hi, hi1le, h2]
    obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists
      (IScalar.add_spec (x := i2) (y := core.convert.num.FromI128I64.from i1) hmin2 hmax2)
    rw [hi3]
    simp only [bind_tc_ok]
    have hi3nn : 0 ≤ i3.val := by
      rw [hi3v, hi2val, hfrm1]; linarith [hnum128neg, hi1nn]
    obtain ⟨i5, hi5, hi5v⟩ := IScalar.div_spec (x := i3) (y := den128)
      (by rw [hden128v]; omega)
      (by intro hc; rw [hden128v] at hc; omega)
    rw [hi5]
    simp only [bind_tc_ok]
    have hi5eq : i5.val = i3.val / den128.val := by
      rw [hi5v, Int.tdiv_eq_ediv_of_nonneg hi3nn]
    have hi5nn : 0 ≤ i5.val := by
      rw [hi5eq]
      exact Int.ediv_nonneg hi3nn (by rw [hden128v]; exact le_of_lt hden)
    have hmin5 : IScalar.min IScalarTy.I128 ≤ -i5.val := by
      rw [mixer2_i128_min_eq, mixer3_i128Min_eq]
      have h3 := IScalar.hmax i5
      rw [show IScalarTy.I128.numBits = 128 from by decide] at h3
      norm_num at h3
      linarith [h3]
    have hmax5 : -i5.val ≤ IScalar.max IScalarTy.I128 := by
      rw [mixer2_i128_max_eq, mixer3_i128Max_eq]
      linarith [hi5nn]
    obtain ⟨q, hq, hqv⟩ := mixer3_neg_val i5 hmin5 hmax5
    rw [hq]
    simp only [bind_tc_ok]
    have hqval : q.val = mixer3_divQ32Ref num.val den.val := by
      rw [hqv, hi5eq, hi3v, hi2val, hfrm1, hnum128, hi1v, hden128v, mixer3_divQ32Ref,
        if_neg hnneg]
      have hAB : -(num.val * 4294967296) + Int.ediv den.val 2 =
          -num.val * 4294967296 + Int.ediv den.val 2 := by rw [neg_mul]
      exact congrArg (fun t => -Int.ediv t den.val) hAB
    have hfitq : I64.min ≤ q.val ∧ q.val ≤ I64.max := by rw [hqval]; exact hfit
    obtain ⟨w, hw, hwv⟩ := mixer2_try_from_val q hfitq.1 hfitq.2
    refine ⟨w, ?_, ?_⟩
    · rw [hw]
      simp only [bind_tc_ok, core.result.Result.map_err_ok]
    · rw [hwv, hqval]

/-! ## §3 · `mixer.delta_head_row` — the `div_q32` leaf discharged

`Serve/Mixer2.lean`'s `mixer2_delta_head_row_ok` states the refusal-free value theorem with the
`div_q32` step as a *hypothesis* `h_div`.  It is discharged here by `mixer3_div_q32_val`; the other
ten leaves keep their Mixer2 hypotheses, and the joint-inhabitation witness at the end of the
section exhibits all eleven at the concrete types. -/

/-- The `0` quotient and the unit divisor: `div_q32 0 1`'s reference value. -/
theorem mixer3_divQ32Ref_zero : mixer3_divQ32Ref 0 1 = 0 := by
  rw [mixer3_divQ32Ref, if_pos (by norm_num)]
  norm_num

/-- The unit Q32.32 multiplier leaves `0` at `0`. -/
theorem mixer3_mulQ32Ref_one_zero : mixer2_mulQ32Ref 1 0 = 0 := by
  rw [mixer2_mulQ32Ref, mul_zero, mixer2_roundShift32Ref_zero]

/-- The literal forms of the two `i64` sentinels (`IScalar.min`/`max` are irreducible). -/
theorem mixer3_i64_min_eq : IScalar.min IScalarTy.I64 = I64.min := by
  rw [I64.min_eq]
  simp only [IScalar.min, IScalarTy.numBits]
  norm_num

theorem mixer3_i64_max_eq : IScalar.max IScalarTy.I64 = I64.max := by
  rw [I64.max_eq]
  simp only [IScalar.max, IScalarTy.numBits]
  norm_num

/-- **`I64::checked_sub` succeeds exactly on the difference inside `i64`.** -/
theorem mixer3_checked_sub_val (a b : Std.I64)
    (h : I64.min ≤ a.val - b.val ∧ a.val - b.val ≤ I64.max) :
    ∃ c : Std.I64, I64.checked_sub a b = some c ∧ c.val = a.val - b.val := by
  have hmin : IScalar.min IScalarTy.I64 ≤ a.val - b.val := by
    rw [mixer3_i64_min_eq]; exact h.1
  have hmax : a.val - b.val ≤ IScalar.max IScalarTy.I64 := by
    rw [mixer3_i64_max_eq]; exact h.2
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists (IScalar.sub_spec (x := a) (y := b) hmin hmax)
  exact ⟨c, by rw [I64.checked_sub, core.num.checked_sub_IScalar, hc]; rfl, hcv⟩

/-- **`I64::checked_add` succeeds exactly on the sum inside `i64`.** -/
theorem mixer3_checked_add_val (a b : Std.I64)
    (h : I64.min ≤ a.val + b.val ∧ a.val + b.val ≤ I64.max) :
    ∃ c : Std.I64, I64.checked_add a b = some c ∧ c.val = a.val + b.val := by
  have hmin : IScalar.min IScalarTy.I64 ≤ a.val + b.val := by
    rw [mixer3_i64_min_eq]; exact h.1
  have hmax : a.val + b.val ≤ IScalar.max IScalarTy.I64 := by
    rw [mixer3_i64_max_eq]; exact h.2
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists (IScalar.add_spec (x := a) (y := b) hmin hmax)
  exact ⟨c, by rw [I64.checked_add, core.num.checked_add_IScalar, hc]; rfl, hcv⟩

/-- **`delta_head_row`, the `div_q32` leaf discharged.** Identical to `mixer2_delta_head_row_ok`
except that the divisor leaf `h_div` is produced by `mixer3_div_q32_val` from `0 < c_val` and the
fitted reference quotient, so the returned `d'` carries its value. -/
theorem mixer3_delta_head_row_val (c_val beta v_j : Std.I64) (acc_m acc_o : Std.I128)
    (k_dot_q : Std.I64) (xm xo : Std.I128)
    (m_tilde o_tilde : Std.I64) (m dj term1 term2 d o : Std.I64)
    (h_rs_m : value.round_shift_32 acc_m = ok xm)
    (h_narrow_m : I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from xm = ok (.Ok m_tilde))
    (h_m : value.mul_q32 c_val m_tilde = ok (some m))
    (h_sub : I64.checked_sub v_j m = some d)
    (h_dj : value.mul_q32 beta d = ok (some dj))
    (hcd : 0 < c_val.val)
    (hfitd : I64.min ≤ mixer3_divQ32Ref dj.val c_val.val ∧
      mixer3_divQ32Ref dj.val c_val.val ≤ I64.max)
    (h_rs_o : value.round_shift_32 acc_o = ok xo)
    (h_narrow_o : I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from xo = ok (.Ok o_tilde))
    (h_t1 : value.mul_q32 c_val o_tilde = ok (some term1))
    (h_t2 : value.mul_q32 dj k_dot_q = ok (some term2))
    (h_add : I64.checked_add term1 term2 = some o) :
    ∃ d_prime : Std.I64,
      mixer.delta_head_row c_val beta v_j acc_m acc_o k_dot_q = ok (.Ok (d_prime, o)) ∧
        d_prime.val = mixer3_divQ32Ref dj.val c_val.val := by
  obtain ⟨d_prime, hdiv, hdivv⟩ := mixer3_div_q32_val dj c_val hcd hfitd
  exact ⟨d_prime, mixer2_delta_head_row_ok (c_val := c_val) (beta := beta) (v_j := v_j)
    (acc_m := acc_m) (acc_o := acc_o) (k_dot_q := k_dot_q) (xm := xm) (xo := xo)
    (m_tilde := m_tilde) (d_prime := d_prime) (o_tilde := o_tilde) (o := o)
    (m := m) (dj := dj) (term1 := term1) (term2 := term2) (d := d)
    h_rs_m h_narrow_m h_m h_sub h_dj hdiv h_rs_o h_narrow_o h_t1 h_t2 h_add, hdivv⟩

/-- **The eleven hypotheses of `delta_head_row` are jointly inhabited at the stated types.** With
`c = β = 1`, `v_j = k·q = 0` and both read accumulators `0`, every leaf lands on the reference `0`,
so the extracted body returns `Ok (0, 0)`.  This is the non-vacuity witness for the domain above. -/
example : ∃ (d_prime o : Std.I64),
    mixer.delta_head_row 1#i64 1#i64 0#i64 0#i128 0#i128 0#i64 = ok (.Ok (d_prime, o)) ∧
      d_prime.val = 0 ∧ o.val = 0 := by
  have h0i : (0#i128).val = 0 := by decide
  have h0 : (0#i64).val = 0 := by decide
  have hone : (1#i64).val = 1 := by decide
  have hmax : (0#i128).val + 2147483648 ≤ IScalar.max IScalarTy.I128 := by
    rw [h0i, mixer2_i128_max_eq]; simp only [mixer2_i128Max]; norm_num
  obtain ⟨xm, hxm, hxmv⟩ := mixer2_round_shift_32_ok 0#i128 hmax
  have hxmv0 : xm.val = 0 := by rw [hxmv, h0i, mixer2_roundShift32Ref_zero]
  obtain ⟨m_tilde, hmt, hmtv⟩ := mixer2_try_from_val xm
    (by rw [hxmv0, I64.min_eq]; norm_num) (by rw [hxmv0, I64.max_eq]; norm_num)
  have hmtv0 : m_tilde.val = 0 := by rw [hmtv, hxmv0]
  obtain ⟨m, hm, hmv⟩ := mixer2_mul_q32_val 1#i64 m_tilde (by
    rw [hone, hmtv0, mixer3_mulQ32Ref_one_zero, I64.min_eq, I64.max_eq]; constructor <;> norm_num)
  have hmv0 : m.val = 0 := by rw [hmv, hone, hmtv0, mixer3_mulQ32Ref_one_zero]
  obtain ⟨d, hd, hdv⟩ := mixer3_checked_sub_val 0#i64 m (by
    rw [h0, hmv0, I64.min_eq, I64.max_eq]; constructor <;> norm_num)
  have hdv0 : d.val = 0 := by rw [hdv, h0, hmv0]; norm_num
  obtain ⟨dj, hdj, hdjv⟩ := mixer2_mul_q32_val 1#i64 d (by
    rw [hone, hdv0, mixer3_mulQ32Ref_one_zero, I64.min_eq, I64.max_eq]; constructor <;> norm_num)
  have hdjv0 : dj.val = 0 := by rw [hdjv, hone, hdv0, mixer3_mulQ32Ref_one_zero]
  obtain ⟨d_prime, hdp, hdpv⟩ := mixer3_div_q32_val dj 1#i64 (by rw [hone]; norm_num) (by
    rw [hdjv0, hone, mixer3_divQ32Ref_zero, I64.min_eq, I64.max_eq]; constructor <;> norm_num)
  have hdpv0 : d_prime.val = 0 := by rw [hdpv, hdjv0, hone, mixer3_divQ32Ref_zero]
  obtain ⟨xo, hxo, hxov⟩ := mixer2_round_shift_32_ok 0#i128 hmax
  have hxov0 : xo.val = 0 := by rw [hxov, h0i, mixer2_roundShift32Ref_zero]
  obtain ⟨o_tilde, hot, hotv⟩ := mixer2_try_from_val xo
    (by rw [hxov0, I64.min_eq]; norm_num) (by rw [hxov0, I64.max_eq]; norm_num)
  have hotv0 : o_tilde.val = 0 := by rw [hotv, hxov0]
  obtain ⟨term1, ht1, ht1v⟩ := mixer2_mul_q32_val 1#i64 o_tilde (by
    rw [hone, hotv0, mixer3_mulQ32Ref_one_zero, I64.min_eq, I64.max_eq]; constructor <;> norm_num)
  have ht1v0 : term1.val = 0 := by rw [ht1v, hone, hotv0, mixer3_mulQ32Ref_one_zero]
  obtain ⟨term2, ht2, ht2v⟩ := mixer2_mul_q32_val dj 0#i64 (by
    rw [hdjv0, h0, mixer2_mulQ32Ref_zero, I64.min_eq, I64.max_eq]; constructor <;> norm_num)
  have ht2v0 : term2.val = 0 := by rw [ht2v, hdjv0, h0, mixer2_mulQ32Ref_zero]
  obtain ⟨o, ho, hov⟩ := mixer3_checked_add_val term1 term2 (by
    rw [ht1v0, ht2v0, I64.min_eq, I64.max_eq]; constructor <;> norm_num)
  have hov0 : o.val = 0 := by rw [hov, ht1v0, ht2v0]; norm_num
  exact ⟨d_prime, o,
    mixer2_delta_head_row_ok (c_val := 1#i64) (beta := 1#i64) (v_j := 0#i64)
      (acc_m := 0#i128) (acc_o := 0#i128) (k_dot_q := 0#i64) (xm := xm) (xo := xo)
      (m_tilde := m_tilde) (d_prime := d_prime) (o_tilde := o_tilde) (o := o)
      (m := m) (dj := dj) (term1 := term1) (term2 := term2) (d := d)
      hxm hmt hm hd hdj hdp hxo hot ht1 ht2 ho, hdpv0, hov0⟩

#print axioms mixer3_div_q32_val
#print axioms mixer3_divQ32Ref_eq
#print axioms mixer3_delta_head_row_val
#print axioms mixer3_checked_sub_val
#print axioms mixer3_checked_add_val

end apeiron_word
