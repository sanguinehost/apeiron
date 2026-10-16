/-
  `Serve/Mixer2.lean` — the positive refinement theorems for the Q32.32 `mixer` value kernels:
  what the extracted bodies *compute*, against a plain-Lean reference that states the meaning
  (`Int` arithmetic: the biased floor shift, the narrowed product, the half-biased quotient),
  not the Rust loop.

  Group: `mixer` (`src/engine/mixer.rs`).  Round 3: the `i128 → i64` narrowing is a *model*
  definition (`I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from`), so the narrowed value
  is knowable and the positive theorems are provable.  `Serve/Mixer.lean` holds the refusal
  theorems and the `hist_q32`/`chunked_route` positives; this file holds the positives the
  boundary axiom blocked, plus their reusable leaf specs.

  Names are file-local (`mixer2_`) so this module registers beside `Serve/Mixer.lean`.
-/

import ApeironWord

set_option autoImplicit false
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## §1 · References and the two word ranges

The plain-Lean meaning of each kernel: `Int`-level arithmetic with the exact floor/round of the
Q32.32 convention, independent of the extraction. -/

/-- Q32.32 rounding to the nearest multiple of `2^32` with ties up: `(v + 2^31) / 2^32` floored
(the Rust body shifts arithmetically). -/
def mixer2_roundShift32Ref (v : Int) : Int := Int.fdiv (v + 2147483648) 4294967296

/-- Q32.32 multiply reference: the narrowed product of the two words. -/
def mixer2_mulQ32Ref (a b : Int) : Int := mixer2_roundShift32Ref (a * b)

/-- Q32.32 divide reference on the admissible domain `0 < den`: half-biased, quotient truncated
toward zero (the Rust `/`), negated for a negative numerator. -/
def mixer2_divQ32Ref (num den : Int) : Int :=
  if 0 ≤ num then Int.ediv (num * 4294967296 + Int.ediv den 2) den
  else -Int.ediv ((-num) * 4294967296 + Int.ediv den 2) den

def mixer2_i128Min : Int := -170141183460469231731687303715884105728
def mixer2_i128Max : Int := 170141183460469231731687303715884105727

theorem mixer2_i128_min_eq : IScalar.min IScalarTy.I128 = mixer2_i128Min := by
  simp only [IScalar.min, IScalarTy.numBits, mixer2_i128Min]
  norm_num

theorem mixer2_i128_max_eq : IScalar.max IScalarTy.I128 = mixer2_i128Max := by
  simp only [IScalar.max, IScalarTy.numBits, mixer2_i128Max]
  norm_num

/-- Every `i64` word is its `Int` value in `[i64::MIN, i64::MAX]`. -/
theorem mixer2_i64_val_bounds (a : Std.I64) :
    -9223372036854775808 ≤ a.val ∧ a.val ≤ 9223372036854775807 := by
  constructor
  · have h := IScalar.hmin a
    simp only [IScalarTy.numBits] at h
    norm_num at h
    exact h
  · have h := IScalar.hmax a
    simp only [IScalarTy.numBits] at h
    norm_num at h
    omega

/-- Every `i128` accumulator is its `Int` value in `[i128::MIN, i128::MAX]`. -/
theorem mixer2_i128_val_bounds (a : Std.I128) : mixer2_i128Min ≤ a.val ∧ a.val ≤ mixer2_i128Max := by
  constructor
  · have h := IScalar.hmin a
    simp only [IScalarTy.numBits, mixer2_i128Min] at h ⊢
    norm_num at h
    exact h
  · have h := IScalar.hmax a
    simp only [IScalarTy.numBits, mixer2_i128Max] at h ⊢
    norm_num at h
    omega

/-- `Int.bmod` is the identity for a value inside the `i128` range: the `128`-bit container does
not wrap a value that fits. -/
theorem mixer2_bmod_i128 {x : Int} (h1 : mixer2_i128Min ≤ x) (h2 : x ≤ mixer2_i128Max) :
    Int.bmod x (2^128) = x := by
  apply Int.bmod_eq_of_le_mul_two
  · simp only [mixer2_i128Min] at h1
    norm_num
    omega
  · simp only [mixer2_i128Max] at h2
    norm_num
    omega

/-! ## §2 · The `i128 → i64` narrowing

`I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from` is
`if I64.min ≤ x.val ∧ x.val ≤ I64.max then ok (.Ok ⟨BitVec.ofInt 64 x.val⟩) else ok (.Err ())`.
The value it produces is the integer value; it refuses exactly outside `[i64::MIN, i64::MAX]`. -/

/-- **The narrowing succeeds with the integer value.** -/
theorem mixer2_try_from_val (x : Std.I128) (h1 : I64.min ≤ x.val) (h2 : x.val ≤ I64.max) :
    ∃ w : Std.I64, I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from x = ok (.Ok w) ∧
      w.val = x.val := by
  refine ⟨⟨BitVec.ofInt 64 x.val⟩, ?_, ?_⟩
  · unfold I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from
    rw [if_pos ⟨h1, h2⟩]
  · rw [I64.min_eq] at h1
    rw [I64.max_eq] at h2
    exact BitVec.toInt_ofInt_eq_self (by norm_num) h1 (by
      rw [show (2 : Int) ^ (IScalarTy.I64.numBits - 1) = 9223372036854775808 by
        simp only [IScalarTy.numBits]; norm_num]
      omega)

/-- **A successful narrowing returns the integer value.** -/
theorem mixer2_try_from_ok (x : Std.I128) (w : Std.I64)
    (h : I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from x = ok (.Ok w)) :
    I64.min ≤ x.val ∧ x.val ≤ I64.max ∧ w.val = x.val := by
  unfold I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from at h
  split at h
  · rename_i hb
    simp only [ok.injEq, core.result.Result.Ok.injEq] at h
    subst h
    refine ⟨hb.1, hb.2, ?_⟩
    rw [I64.min_eq] at hb
    rw [I64.max_eq] at hb
    exact BitVec.toInt_ofInt_eq_self (by norm_num) hb.1 (by
      rw [show (2 : Int) ^ (IScalarTy.I64.numBits - 1) = 9223372036854775808 by
        simp only [IScalarTy.numBits]; norm_num]
      omega)
  · simp at h

/-- **The narrowing refuses exactly outside `[i64::MIN, i64::MAX]`.** -/
theorem mixer2_try_from_refuse (x : Std.I128) (e : core.num.error.TryFromIntError)
    (h : I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from x = ok (.Err e)) :
    ¬ (I64.min ≤ x.val ∧ x.val ≤ I64.max) := by
  unfold I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from at h
  split at h
  · simp at h
  · rename_i hb
    exact hb

/-- **The `Result::ok` projection of a successful narrowing returns the value.** For every
`i128` and `i64` word whose `try_from`-then-`.ok` equals `ok (some w)`, the narrowing succeeded
inside `[i64::MIN, i64::MAX]` and `w` is the integer value. -/
theorem mixer2_tryfrom_bind_ok (x : Std.I128) (w : Std.I64)
    (h : (do let r ← I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from x
             core.result.Result.ok r) = ok (some w)) :
    I64.min ≤ x.val ∧ x.val ≤ I64.max ∧ w.val = x.val := by
  unfold I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from at h
  split at h
  · rename_i hb
    simp only [bind_tc_ok, core.result.Result.ok, ok.injEq, Option.some.injEq] at h
    refine ⟨hb.1, hb.2, ?_⟩
    rw [← h]
    rw [I64.min_eq] at hb
    rw [I64.max_eq] at hb
    exact BitVec.toInt_ofInt_eq_self (by norm_num) hb.1 (by
      rw [show (2 : Int) ^ (IScalarTy.I64.numBits - 1) = 9223372036854775808 by
        simp only [IScalarTy.numBits]; norm_num]
      omega)
  · exfalso
    simp only [bind_tc_ok, core.result.Result.ok, ok.injEq] at h
    exact absurd h.symm (Option.some_ne_none w)

/-! ## §3 · `value.round_shift_32` — round-to-nearest at `2^-32` -/

/-- The tie bias `1#i128 <<< 31#i32` has value `2^31`. -/
theorem mixer2_p31 : ∃ z : Std.I128, (1#i128 <<< 31#i32) = ok z ∧ z.val = 2147483648 := by
  obtain ⟨z, hz, hzv, -⟩ := WP.spec_imp_exists
    (IScalar.ShiftLeft_IScalar_spec (1#i128) (31#i32) (by scalar_tac) (by scalar_tac))
  refine ⟨z, hz, ?_⟩
  rw [hzv, show (1#i128).val = 1 from by simp [IScalar.val, I128.ofInt_bv]]
  decide

/-- **`round_shift_32` is the biased floor shift.** For every accumulator whose biased sum does
not overflow `i128`, the extracted body returns the reference value. -/
theorem mixer2_round_shift_32_ok (v : Std.I128)
    (hv : v.val + 2147483648 ≤ IScalar.max IScalarTy.I128) :
    ∃ w : Std.I128, value.round_shift_32 v = ok w ∧ w.val = mixer2_roundShift32Ref v.val := by
  unfold value.round_shift_32
  obtain ⟨z, hz, hzv⟩ := mixer2_p31
  rw [hz]
  simp only [bind_tc_ok]
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (IScalar.add_spec (x := v) (y := z)
      (by rw [hzv, mixer2_i128_min_eq]
          have hb := (mixer2_i128_val_bounds v).1
          simp only [mixer2_i128Min] at hb ⊢
          omega)
      (by rw [hzv]; exact hv))
  rw [hi1]
  simp only [bind_tc_ok]
  obtain ⟨w, hw, hwv, -⟩ := WP.spec_imp_exists
    (IScalar.ShiftRight_IScalar_spec (x := i1) (y := (32#i32)) (by scalar_tac) (by scalar_tac))
  refine ⟨w, by rw [hw], ?_⟩
  rw [hwv, hi1v, hzv]
  rw [show (32#i32).toNat = 32 from by simp [IScalar.toNat, I32.ofInt_bv]]
  rw [Int.shiftRight_eq_div_pow]
  rw [mixer2_roundShift32Ref, Int.fdiv_eq_ediv]
  norm_num

/-! ## §4 · `value.mul_q32` — the narrowed product

`mul_q32 a b` is `a·b` rounded at `2^-32` and narrowed to `i64`, `none` when the rounded product
does not fit.  The `i64·i64` product is bounded by `2^126 < 2^127`, so the `i128` multiply and the
biased sum never overflow; only the `i64` narrowing can refuse. -/

/-- **`mul_q32` returns the reference product whenever the reference value fits in `i64`.** -/
theorem mixer2_mul_q32_val (a b : Std.I64)
    (h : I64.min ≤ mixer2_mulQ32Ref a.val b.val ∧ mixer2_mulQ32Ref a.val b.val ≤ I64.max) :
    ∃ w : Std.I64, value.mul_q32 a b = ok (some w) ∧ w.val = mixer2_mulQ32Ref a.val b.val := by
  obtain ⟨ha1, ha2⟩ := mixer2_i64_val_bounds a
  obtain ⟨hb1, hb2⟩ := mixer2_i64_val_bounds b
  unfold value.mul_q32
  simp only [lift, bind_tc_ok]
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists
    (IScalar.mul_spec (x := core.convert.num.FromI128I64.from a)
      (y := core.convert.num.FromI128I64.from b)
      (by rw [mixer2_i128_min_eq, core.convert.num.FromI128I64.from_val_eq,
        core.convert.num.FromI128I64.from_val_eq]
          simp only [mixer2_i128Min]
          nlinarith)
      (by rw [mixer2_i128_max_eq, core.convert.num.FromI128I64.from_val_eq,
        core.convert.num.FromI128I64.from_val_eq]
          simp only [mixer2_i128Max]
          nlinarith))
  rw [hi2]
  simp only [bind_tc_ok]
  have hi2m : i2.val = a.val * b.val := by
    rw [hi2v, core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
  obtain ⟨i3, hi3, hi3v⟩ := mixer2_round_shift_32_ok i2
    (by rw [hi2m, mixer2_i128_max_eq]; simp only [mixer2_i128Max]; nlinarith)
  rw [hi3]
  simp only [bind_tc_ok]
  obtain ⟨w, hw, hwv⟩ := mixer2_try_from_val i3
    (by rw [hi3v, hi2m]; exact h.1) (by rw [hi3v, hi2m]; exact h.2)
  refine ⟨w, ?_, ?_⟩
  · rw [hw]; simp only [bind_tc_ok, core.result.Result.ok]
  · rw [hwv, hi3v, hi2m, mixer2_mulQ32Ref]

/-- **A successful `mul_q32` returns the reference product and its range.** -/
theorem mixer2_mul_q32_ok (a b : Std.I64) (w : Std.I64)
    (h : value.mul_q32 a b = ok (some w)) :
    I64.min ≤ mixer2_mulQ32Ref a.val b.val ∧ mixer2_mulQ32Ref a.val b.val ≤ I64.max ∧
      w.val = mixer2_mulQ32Ref a.val b.val := by
  obtain ⟨ha1, ha2⟩ := mixer2_i64_val_bounds a
  obtain ⟨hb1, hb2⟩ := mixer2_i64_val_bounds b
  unfold value.mul_q32 at h
  simp only [lift, bind_tc_ok] at h
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists
    (IScalar.mul_spec (x := core.convert.num.FromI128I64.from a)
      (y := core.convert.num.FromI128I64.from b)
      (by rw [mixer2_i128_min_eq, core.convert.num.FromI128I64.from_val_eq,
        core.convert.num.FromI128I64.from_val_eq]
          simp only [mixer2_i128Min]
          nlinarith)
      (by rw [mixer2_i128_max_eq, core.convert.num.FromI128I64.from_val_eq,
        core.convert.num.FromI128I64.from_val_eq]
          simp only [mixer2_i128Max]
          nlinarith))
  rw [hi2] at h
  simp only [bind_tc_ok] at h
  have hi2m : i2.val = a.val * b.val := by
    rw [hi2v, core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
  obtain ⟨i3, hi3, hi3v⟩ := mixer2_round_shift_32_ok i2
    (by rw [hi2m, mixer2_i128_max_eq]; simp only [mixer2_i128Max]; nlinarith)
  rw [hi3] at h
  simp only [bind_tc_ok] at h
  obtain ⟨hb1', hb2', hwv⟩ := mixer2_tryfrom_bind_ok i3 w h
  refine ⟨?_, ?_, ?_⟩
  · rw [hi3v, hi2m, ← mixer2_mulQ32Ref] at hb1'
    exact hb1'
  · rw [hi3v, hi2m, ← mixer2_mulQ32Ref] at hb2'
    exact hb2'
  · rw [hwv, hi3v, hi2m, mixer2_mulQ32Ref]

/-! ## §5 · `mixer.delta_head_row` — the delta-net head step

The read pass' value column `j`: the narrowed accumulated reads give `m̃ = round(S̃ᵀk)`,
`m = c·m̃`, `dj = β·(v_j − m)`, `d' = dj / c`; the output column is
`o_j = c·round(S̃ᵀq̃) + dj·(k·q)`.  The theorem is stated with each leaf's result as its
hypothesis; the leaf specs above discharge those hypotheses at value level. -/

/-- **`delta_head_row` returns the reference column pair.** Given each leaf produces its named
value (the `try_from` narrowing, the two `mul_q32` chains, the `div_q32`, the `checked_sub` /
`checked_add`), the extracted body returns `Ok (d', o)`. -/
theorem mixer2_delta_head_row_ok (c_val beta v_j : Std.I64) (acc_m acc_o : Std.I128)
    (k_dot_q : Std.I64) (xm xo : Std.I128)
    (m_tilde d_prime o_tilde o : Std.I64) (m dj term1 term2 d : Std.I64)
    (h_rs_m : value.round_shift_32 acc_m = ok xm)
    (h_narrow_m : I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from xm = ok (.Ok m_tilde))
    (h_m : value.mul_q32 c_val m_tilde = ok (some m))
    (h_sub : I64.checked_sub v_j m = some d)
    (h_dj : value.mul_q32 beta d = ok (some dj))
    (h_div : mixer.div_q32 dj c_val = ok (.Ok d_prime))
    (h_rs_o : value.round_shift_32 acc_o = ok xo)
    (h_narrow_o : I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from xo = ok (.Ok o_tilde))
    (h_t1 : value.mul_q32 c_val o_tilde = ok (some term1))
    (h_t2 : value.mul_q32 dj k_dot_q = ok (some term2))
    (h_add : I64.checked_add term1 term2 = some o) :
    mixer.delta_head_row c_val beta v_j acc_m acc_o k_dot_q = ok (.Ok (d_prime, o)) := by
  unfold mixer.delta_head_row
  simp only [h_rs_m, h_narrow_m, h_m, h_sub, h_dj, h_div, h_rs_o, h_narrow_o, h_t1, h_t2, h_add,
    lift, core.result.Result.map_err_ok, core.option.Option.ok_or_some,
    core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok]

/-! ## §6 · Non-vacuity — checked witnesses for each stated domain

The reference point `0` in each of the two carriers, and the two word-range facts, make the
hypotheses of the leaf specs satisfiable at the stated types. -/

theorem mixer2_roundShift32Ref_zero : mixer2_roundShift32Ref 0 = 0 := by
  rw [mixer2_roundShift32Ref, Int.fdiv_eq_ediv]
  norm_num

theorem mixer2_mulQ32Ref_zero : mixer2_mulQ32Ref 0 0 = 0 := by
  simp only [mixer2_mulQ32Ref, zero_mul, mixer2_roundShift32Ref_zero]

/-- **`round_shift_32`'s domain is inhabited and yields the reference.** -/
example : ∃ w : Std.I128, value.round_shift_32 0#i128 = ok w ∧ w.val = 0 := by
  obtain ⟨w, hw, hwv⟩ := mixer2_round_shift_32_ok 0#i128 (by
    have h0 : (0#i128).val = 0 := by decide
    rw [h0, mixer2_i128_max_eq]
    simp only [mixer2_i128Max]; norm_num)
  refine ⟨w, hw, ?_⟩
  have h0 : (0#i128).val = 0 := by decide
  rw [hwv, h0, mixer2_roundShift32Ref_zero]

/-- **The narrowing succeeds for an in-range value and returns it.** -/
example : ∃ w : Std.I64,
    I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from 0#i128 = ok (.Ok w) ∧ w.val = 0 := by
  have h0 : (0#i128).val = 0 := by decide
  obtain ⟨w, hw, hwv⟩ := mixer2_try_from_val 0#i128
    (by rw [h0, I64.min_eq]; norm_num)
    (by rw [h0, I64.max_eq]; norm_num)
  refine ⟨w, hw, ?_⟩
  rw [hwv, h0]

/-- **The narrowing refuses outside the `i64` range.** -/
example : ∃ e : core.num.error.TryFromIntError,
    I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from 9223372036854775808#i128
      = ok (.Err e) := by
  refine ⟨(), ?_⟩
  unfold I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from
  rw [if_neg (by
    intro hc
    have h2 := hc.2
    rw [I64.max_eq] at h2
    have hv : (9223372036854775808#i128).val = 9223372036854775808 := by decide
    omega)]

/-- **`mul_q32`'s domain is inhabited and yields the reference product.** -/
example : ∃ w : Std.I64, value.mul_q32 0#i64 0#i64 = ok (some w) ∧ w.val = 0 := by
  obtain ⟨w, hw, hwv⟩ := mixer2_mul_q32_val 0#i64 0#i64 (by
    have h0 : (0#i64).val = 0 := by decide
    rw [h0, mixer2_mulQ32Ref_zero, I64.min_eq, I64.max_eq]
    norm_num)
  refine ⟨w, hw, ?_⟩
  have h0 : (0#i64).val = 0 := by decide
  rw [hwv, h0, mixer2_mulQ32Ref_zero]

#print axioms mixer2_try_from_val
#print axioms mixer2_try_from_ok
#print axioms mixer2_try_from_refuse
#print axioms mixer2_tryfrom_bind_ok
#print axioms mixer2_round_shift_32_ok
#print axioms mixer2_mul_q32_val
#print axioms mixer2_mul_q32_ok
#print axioms mixer2_delta_head_row_ok

end apeiron_word
