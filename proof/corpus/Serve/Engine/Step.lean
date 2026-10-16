/-
Refinement of the extracted `step.attn_block.*` and `step.*` serving kernels (`SERVE3-step`).

The Rust bodies are `src/engine/step/attn_block.rs` (`mul_round_2p32` 39-42, `recip_est` 68-79,
`sigmoid_lanes` 104-159, `sigmoid_gate` 172-206, `gate_product` 213-215, `rotate_pair_prefixes`
237-307) and `src/engine/step.rs` (`affine_slots` 442-454, `apply_sigmoid_gate` 516-518,
`bytes_at` 686-699, `any_vec_len_ne` 706-715, `bytes_eq` 722-724). Every function here is on the
serving path reachable from `Engine::next_token_state`.

This module states WHAT each extracted body computes, as a reference over `Int`/`Nat`/`List`, and
proves the refinement (`word_<fn>_ok`) and, where the body can refuse, the refusal
(`word_<fn>_refuses`). Reused, not restated: `roundShift32` and `value_round_shift_32_spec`
(`Numeric/Engine/Rope.lean`), `I128_mul_q32_some` (`Numeric/Engine/Silu.lean`).

## Boundary

`sigmoid_lanes`, `sigmoid_gate`, `apply_sigmoid_gate`, `rotate_pair_prefixes`, `affine_slots` and
`softmax_slots` carry no refinement theorem in this pass; each blocker is named with its artefact in
the "Boundary" section at the end of the file. The float carrier (`F32`/`F64`) and the AVX2 paths are
out of scope (they are boundaries).
-/
import ApeironWord
import Numeric.Engine.Rope
import Numeric.Engine.Silu

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## Generic helpers -/

/-- A total extracted body's `ok` value, from a WP triple that pins it. -/
theorem stp_spec_eq {α : Type} {f : Result α} {v : α} (h : f ⦃ z => z = v ⦄) : f = ok v := by
  rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-! ## The extracted constants -/

theorem stp_attn_ONE_eq : step.attn_block.ONE = ok (4294967296#i64) := by
  have h : step.attn_block.ONE ⦃ z => z = (4294967296#i64) ⦄ := by
    unfold step.attn_block.ONE
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I64.size, IScalar.size, I64.numBits, Int.shiftLeft_eq, Int.bmod]
  exact stp_spec_eq h

theorem stp_attn_LOWER_eq : step.attn_block.LOWER = ok ((-68719476736)#i64) := by
  have h : step.attn_block.LOWER ⦃ z => z = ((-68719476736)#i64) ⦄ := by
    unfold step.attn_block.LOWER
    rw [stp_attn_ONE_eq]
    simp only [bind_tc_ok]
    apply WP.spec_mono
      (I64.mul_spec (x := (-16)#i64) (y := 4294967296#i64)
        (by norm_num [I64.min, I64.max, I64.numBits])
        (by norm_num [I64.min, I64.max, I64.numBits]))
    intro z hz
    apply IScalar.eq_imp
    rw [hz]
    norm_num
  exact stp_spec_eq h

theorem stp_attn_LN2_Q32_eq : step.attn_block.LN2_Q32 = 2977044472#i64 := by
  unfold step.attn_block.LN2_Q32
  rfl

theorem stp_attn_LN2_RECIP_Q32_eq : step.attn_block.LN2_RECIP_Q32 = 6196328019#i64 := by
  unfold step.attn_block.LN2_RECIP_Q32
  rfl

theorem stp_attn_LANES_eq : step.attn_block.LANES = 8#usize := by
  unfold step.attn_block.LANES
  rfl

theorem stp_attn_MAX_PAIRS_eq : step.attn_block.MAX_PAIRS = 128#usize := by
  unfold step.attn_block.MAX_PAIRS
  rfl

/-! ## `mul_round_2p32`

`src/engine/step/attn_block.rs:39-42`: `round(a · r / 2^32)`, ties away from zero, without a sign
branch: `(v + 2^31 + (v >> 127)) >> 32` with `v = a · r`, where `>>` is the arithmetic shift (floor
division by the power of two). The reference below is the rounding itself, not the shift chain. -/

/-- `round(y / 2^32)`, ties away from zero. -/
def stp_roundDiv2p32 (y : ℤ) : ℤ :=
  if 0 ≤ y then (y + 2147483648) / 4294967296 else -(((-y) + 2147483648) / 4294967296)

/-- The extracted branch-free form at `a · r`: `(a·r + 2^31 + (a·r >>> 127)) >>> 32`. -/
def stp_mulRoundRef (a r : ℤ) : ℤ := (a * r + 2147483648 + ((a * r) >>> 127)) >>> 32

/-- The branch-free form is the ties-away-from-zero rounding of `a · r / 2^32`: for `|a·r| < 2^127`
the sign shift `a·r >>> 127` is `0` (`a·r ≥ 0`) or `-1` (`a·r < 0`), and one floor division by
`2^32` of the half-offset numerator is the whole rounding. -/
theorem stp_mulRoundRef_eq (a r : ℤ) (h : |a * r| < 2^127) :
    stp_mulRoundRef a r = stp_roundDiv2p32 (a * r) := by
  have hlt : a * r < 2 ^ 127 := by
    have := le_abs_self (a * r)
    omega
  have hge : -(2 ^ 127) ≤ a * r := by
    have := neg_abs_le (a * r)
    omega
  unfold stp_mulRoundRef stp_roundDiv2p32
  simp only [Int.shiftRight_eq_div_pow, Nat.cast_ofNat, Nat.cast_pow]
  by_cases h0 : 0 ≤ a * r
  · rw [if_pos h0]
    have hsh : (a * r) / 2 ^ 127 = 0 := by omega
    rw [hsh, add_zero]
    norm_num
  · rw [if_neg h0]
    have hsh : (a * r) / 2 ^ 127 = -1 := by omega
    rw [hsh]
    have hd : (a * r + 2147483648 + -1) / 2 ^ 32
        = -((-(a * r) + 2147483648) / 2 ^ 32) := by omega
    rw [hd]
    norm_num

/-- `I128.size` is `2^128`. -/
theorem stp_I128_size_eq : I128.size = 2 ^ 128 := by
  norm_num [I128.size, IScalar.size, I128.numBits]

/-- The `i128` arithmetic shift by 127 of a value in the `2^127` window is its sign, `0` or `-1`. -/
theorem stp_shift127_val (y : ℤ) (h : |y| < 2 ^ 127) :
    y >>> 127 = if 0 ≤ y then 0 else -1 := by
  have h1 : y >>> 127 = y / (2 : ℤ) ^ 127 := by
    rw [Int.shiftRight_eq_div_pow]
    norm_num
  rw [h1]
  by_cases h0 : 0 ≤ y
  · rw [if_pos h0]
    have hlt : y < 2 ^ 127 := lt_of_le_of_lt (le_abs_self y) h
    omega
  · rw [if_neg h0]
    have hgt : -(2 ^ 127) < y := by
      have := neg_abs_le y
      omega
    omega

/-- The `i128` arithmetic shift by 32 of a value inside the `2^95` window fits `i64`. -/
theorem stp_shift32_fits (y : ℤ) (hlo : -(2 ^ 95) ≤ y) (hhi : y < 2 ^ 95) :
    -(2 ^ 63) ≤ y >>> 32 ∧ y >>> 32 < 2 ^ 63 := by
  have h1 : y >>> 32 = y / (2 : ℤ) ^ 32 := by
    rw [Int.shiftRight_eq_div_pow]
    norm_num
  rw [h1]
  constructor <;> omega

/-- **`mul_round_2p32` refinement.** On the documented operand window (`|a| < 2^34`, `|r| ≤ 2^32`,
the window `sigmoid_lanes` reduces into) the extracted body returns the ties-away-from-zero
rounding of the exact rational `a · r / 2^32`. -/
theorem word_mul_round_2p32_ok (a r : I64) (ha : |a.val| < 2 ^ 34) (hr : |r.val| ≤ 2 ^ 32) :
    ∃ z : I64, step.attn_block.mul_round_2p32 a r = ok z ∧
      z.val = stp_roundDiv2p32 (a.val * r.val) := by
  have h66 : |a.val * r.val| < 2 ^ 66 := by
    rw [abs_mul]
    calc |a.val| * |r.val| ≤ |a.val| * 2 ^ 32 := by
          exact mul_le_mul_of_nonneg_left hr (abs_nonneg _)
      _ < 2 ^ 34 * 2 ^ 32 := by
          exact mul_lt_mul_of_pos_right ha (by norm_num)
      _ = 2 ^ 66 := by norm_num
  have h127 : |a.val * r.val| < 2 ^ 127 := lt_trans h66 (by norm_num)
  have hspec : step.attn_block.mul_round_2p32 a r ⦃ z =>
      z.val = stp_roundDiv2p32 (a.val * r.val) ⦄ := by
    unfold step.attn_block.mul_round_2p32
    simp only [lift, bind_tc_ok]
    step*
    · have hv : (↑v : ℤ) = a.val * r.val := by
        rw [v_post, core.convert.num.FromI128I64.from_val_eq,
          core.convert.num.FromI128I64.from_val_eq]
      have hs : (↑i2 : ℤ) = 2147483648 := by
        rw [i2_post, stp_I128_size_eq, Int.shiftLeft_eq]
        norm_num [Int.bmod]
      have hm : I128.max = 2 ^ 127 - 1 := by norm_num [I128.max, IScalar.max, I128.numBits]
      rw [hv, hs, hm]
      have hb := abs_le.mp (le_of_lt h66)
      omega
    · have hv : (↑v : ℤ) = a.val * r.val := by
        rw [v_post, core.convert.num.FromI128I64.from_val_eq,
          core.convert.num.FromI128I64.from_val_eq]
      have hs : (↑i2 : ℤ) = 2147483648 := by
        rw [i2_post, stp_I128_size_eq, Int.shiftLeft_eq]
        norm_num [Int.bmod]
      have h4 : (↑i4 : ℤ) = (if 0 ≤ a.val * r.val then 0 else -1) := by
        rw [i4_post, hv, stp_shift127_val _ h127]
      have hm : I128.min = -(2 ^ 127) := by norm_num [I128.min, IScalar.min, I128.numBits]
      rw [i3_post, hv, hs, h4, hm]
      have hb := abs_le.mp (le_of_lt h66)
      split <;> omega
    · have hv : (↑v : ℤ) = a.val * r.val := by
        rw [v_post, core.convert.num.FromI128I64.from_val_eq,
          core.convert.num.FromI128I64.from_val_eq]
      have hs : (↑i2 : ℤ) = 2147483648 := by
        rw [i2_post, stp_I128_size_eq, Int.shiftLeft_eq]
        norm_num [Int.bmod]
      have h4 : (↑i4 : ℤ) = (if 0 ≤ a.val * r.val then 0 else -1) := by
        rw [i4_post, hv, stp_shift127_val _ h127]
      have hm : I128.max = 2 ^ 127 - 1 := by norm_num [I128.max, IScalar.max, I128.numBits]
      rw [i3_post, hv, hs, h4, hm]
      have hb := abs_le.mp (le_of_lt h66)
      split <;> omega
    · have hv : (↑v : ℤ) = a.val * r.val := by
        rw [v_post, core.convert.num.FromI128I64.from_val_eq,
          core.convert.num.FromI128I64.from_val_eq]
      have hs : (↑i2 : ℤ) = 2147483648 := by
        rw [i2_post, stp_I128_size_eq, Int.shiftLeft_eq]
        norm_num [Int.bmod]
      have h4 : (↑i4 : ℤ) = (if 0 ≤ a.val * r.val then 0 else -1) := by
        rw [i4_post, hv, stp_shift127_val _ h127]
      have h6 : (↑i6 : ℤ) =
          ((a.val * r.val + 2147483648 + (if 0 ≤ a.val * r.val then 0 else -1)) >>> 32) := by
        rw [i6_post, i5_post, i3_post, hv, hs, h4]
      have h6lo : -(2 ^ 63) ≤ (↑i6 : ℤ) := by
        rw [h6]
        exact (stp_shift32_fits _
          (by have hb := abs_le.mp (le_of_lt h66); split <;> omega)
          (by have hb := abs_le.mp (le_of_lt h66); split <;> omega)).1
      have h6hi : (↑i6 : ℤ) < 2 ^ 63 := by
        rw [h6]
        exact (stp_shift32_fits _
          (by have hb := abs_le.mp (le_of_lt h66); split <;> omega)
          (by have hb := abs_le.mp (le_of_lt h66); split <;> omega)).2
      have h6eq : (↑i6 : ℤ) = stp_mulRoundRef a.val r.val := by
        rw [h6]
        unfold stp_mulRoundRef
        rw [← stp_shift127_val (a.val * r.val) h127]
        norm_num
      have hcast : (↑(IScalar.cast IScalarTy.I64 i6) : ℤ) = (↑i6 : ℤ) := by
        rw [IScalar.val_mod_pow_inBounds]
        · exact h6lo
        · exact h6hi
      rw [hcast, h6eq, ← stp_mulRoundRef_eq a.val r.val h127]
  exact (WP.spec_equiv_exists _ _).1 hspec

/-- Non-vacuity: the reference is exercised on both signs and the branch-free form agrees. -/
example : stp_roundDiv2p32 15 = 0 := by decide

example : stp_mulRoundRef 3 2147483648 = 2 := by decide

example : stp_mulRoundRef (-3) 2147483648 = -2 := by decide

/-! ## `recip_est`

`src/engine/step/attn_block.rs:68-79`: `floor(n_wide / den)` saturated to `u64` (`u64::MAX` for
`den = 0` and for a quotient above `u64::MAX`). -/

/-- `floor(n / d)` saturated to `u64::MAX`. -/
def stp_recipRef (n d : ℕ) : ℕ := if d = 0 then 2 ^ 64 - 1 else min (n / d) (2 ^ 64 - 1)

/-- **`recip_est` refinement.** The extracted body returns the saturated `floor(n_wide / den)` in
both arms: `u64::MAX` for `den = 0` and for a quotient above `u64::MAX`, the exact quotient
otherwise. -/
theorem word_recip_est_ok (n_wide : U128) (den : U64) :
    ∃ z : U64, step.attn_block.recip_est n_wide den = ok z ∧
      z.val = stp_recipRef n_wide.val den.val := by
  have hspec : step.attn_block.recip_est n_wide den ⦃ z =>
      z.val = stp_recipRef n_wide.val den.val ⦄ := by
    unfold step.attn_block.recip_est
    by_cases h0 : den = 0#u64
    · have hden0 : den.val = 0 := by
        have h := (UScalar.eq_equiv den (0#u64)).1 h0
        simpa using h
      rw [if_pos h0, WP.spec_ok]
      rw [u64_max_val, stp_recipRef, if_pos hden0]
    · have hden0 : den.val ≠ 0 := by
        intro hc
        exact h0 ((UScalar.eq_equiv den (0#u64)).2 (by simpa using hc))
      rw [if_neg h0]
      simp only [lift, bind_tc_ok]
      step*
      · rename_i hgt
        have h1 : (core.convert.num.FromU128U64.from core.num.U64.MAX).val = 2 ^ 64 - 1 := by
          rw [core.convert.num.FromU128U64.from_val_eq, u64_max_val]
        have h2 : (core.convert.num.FromU128U64.from den).val = den.val :=
          core.convert.num.FromU128U64.from_val_eq den
        rw [h2] at qq_post
        have hgt' : 2 ^ 64 - 1 < qq.val := by
          have h := (UScalar.lt_equiv _ _).1 hgt
          rw [h1] at h
          exact h
        rw [u64_max_val, stp_recipRef, if_neg hden0, min_eq_right (by omega)]
      · rename_i hnle
        have h1 : (core.convert.num.FromU128U64.from core.num.U64.MAX).val = 2 ^ 64 - 1 := by
          rw [core.convert.num.FromU128U64.from_val_eq, u64_max_val]
        have h2 : (core.convert.num.FromU128U64.from den).val = den.val :=
          core.convert.num.FromU128U64.from_val_eq den
        rw [h2] at qq_post
        have hle : qq.val ≤ 2 ^ 64 - 1 := by
          have h : ¬ (core.convert.num.FromU128U64.from core.num.U64.MAX < qq) := hnle
          simp only [UScalar.lt_equiv, not_lt] at h
          rw [h1] at h
          exact h
        rw [stp_recipRef, if_neg hden0,
          UScalar.cast_val_mod_pow_of_inBounds_eq UScalarTy.U64 qq
            (by rw [UScalarTy.U64_numBits_eq]; omega),
          min_eq_left (by omega), qq_post]
  exact (WP.spec_equiv_exists _ _).1 hspec

/-- Non-vacuity: the reference is exercised on all three arms. -/
example : stp_recipRef 10 3 = 3 := by decide

example : stp_recipRef 10 0 = 2 ^ 64 - 1 := by decide

example : stp_recipRef (2 ^ 70) 1 = 2 ^ 64 - 1 := by decide

/-! ## `value.mul_q32` and `gate_product`

`src/engine/value.rs:31-33`: `mul_q32 a b = i64::try_from(round_shift_32 (a · b)).ok()`, the `None`
arm when the rounded product leaves `i64`. `src/engine/step/attn_block.rs:213-215`: `gate_product a s`
maps that `Option` to `Ok` / `Err Overflow`. The `i64` product is bounded by `2^126 < 2^127`, so the
`i128` multiply and the biased sum never overflow; only the `i64` narrowing can refuse. -/

/-- The `i128` lower bound as an integer literal. -/
def stp_i128Min : ℤ := -170141183460469231731687303715884105728

/-- The `i128` upper bound as an integer literal. -/
def stp_i128Max : ℤ := 170141183460469231731687303715884105727

theorem stp_i128_min_eq : IScalar.min IScalarTy.I128 = stp_i128Min := by
  simp only [IScalar.min, IScalarTy.numBits, stp_i128Min]; norm_num

theorem stp_i128_max_eq : IScalar.max IScalarTy.I128 = stp_i128Max := by
  simp only [IScalar.max, IScalarTy.numBits, stp_i128Max]; norm_num

/-- Every `i64` word is its integer value in `[i64::MIN, i64::MAX]`. -/
theorem stp_i64_val_bounds (a : I64) :
    -9223372036854775808 ≤ a.val ∧ a.val ≤ 9223372036854775807 := by
  constructor
  · have h := IScalar.hmin a
    simp only [IScalarTy.numBits] at h; norm_num at h; exact h
  · have h := IScalar.hmax a
    simp only [IScalarTy.numBits] at h; norm_num at h; omega

/-- The narrowed word of an integer in the `i64` window. -/
def stp_i64Of (v : ℤ) : I64 := ⟨BitVec.ofInt 64 v⟩

theorem stp_i64Of_val {v : ℤ} (h1 : -(2 ^ 63) ≤ v) (h2 : v < 2 ^ 63) :
    (stp_i64Of v).val = v := by
  unfold stp_i64Of
  exact BitVec.toInt_ofInt_eq_self (by norm_num) h1 h2

/-- `mul_q32`'s reference: the biased floor shift of the exact product, `some` when it fits `i64`. -/
def stp_mulQ32Ref (a b : ℤ) : Option ℤ :=
  if -(2 ^ 63) ≤ roundShift32 (a * b) ∧ roundShift32 (a * b) < 2 ^ 63
  then some (roundShift32 (a * b)) else none

/-- **`value.mul_q32` refinement.** The extracted body returns `some` of the reference product
exactly when the reference product fits `i64`, and `none` otherwise. -/
theorem word_mul_q32_ok (a b : I64) :
    value.mul_q32 a b = ok ((stp_mulQ32Ref a.val b.val).map stp_i64Of) := by
  have ha := stp_i64_val_bounds a
  have hb := stp_i64_val_bounds b
  unfold value.mul_q32
  simp only [lift, bind_tc_ok]
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists
    (IScalar.mul_spec (x := core.convert.num.FromI128I64.from a)
      (y := core.convert.num.FromI128I64.from b)
      (by rw [stp_i128_min_eq, core.convert.num.FromI128I64.from_val_eq,
        core.convert.num.FromI128I64.from_val_eq]
          simp only [stp_i128Min]; nlinarith [ha.1, ha.2, hb.1, hb.2])
      (by rw [stp_i128_max_eq, core.convert.num.FromI128I64.from_val_eq,
        core.convert.num.FromI128I64.from_val_eq]
          simp only [stp_i128Max]; nlinarith [ha.1, ha.2, hb.1, hb.2]))
  rw [hi2]
  simp only [bind_tc_ok]
  have hi2m : i2.val = a.val * b.val := by
    rw [hi2v, core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
  obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists
    (value_round_shift_32_spec i2
      (by rw [hi2m, I128.min_eq]; nlinarith [ha.1, ha.2, hb.1, hb.2])
      (by rw [hi2m, I128.max_eq]; nlinarith [ha.1, ha.2, hb.1, hb.2]))
  rw [hi3]
  simp only [bind_tc_ok]
  unfold I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from
  rw [hi3v, hi2m]
  by_cases hc : -(2 ^ 63) ≤ roundShift32 (a.val * b.val) ∧
      roundShift32 (a.val * b.val) < 2 ^ 63
  · have hcond : I64.min ≤ roundShift32 (a.val * b.val) ∧
        roundShift32 (a.val * b.val) ≤ I64.max := by
      rw [I64.min_eq, I64.max_eq]; exact ⟨hc.1, by omega⟩
    rw [if_pos hcond, stp_mulQ32Ref, if_pos hc]
    simp only [bind_tc_ok, core.result.Result.ok, Option.map, stp_i64Of]
  · have hcond : ¬ (I64.min ≤ roundShift32 (a.val * b.val) ∧
        roundShift32 (a.val * b.val) ≤ I64.max) := by
      rw [I64.min_eq, I64.max_eq]; omega
    rw [if_neg hcond, stp_mulQ32Ref, if_neg hc]
    simp only [bind_tc_ok, core.result.Result.ok, Option.map]

/-- `gate_product`'s reference: the narrowed product as `Ok`, `Err Overflow` when it leaves `i64`. -/
def stp_gateProdRef (a s : ℤ) : core.result.Result I64 step.StepError :=
  if -(2 ^ 63) ≤ roundShift32 (a * s) ∧ roundShift32 (a * s) < 2 ^ 63
  then core.result.Result.Ok (stp_i64Of (roundShift32 (a * s)))
  else core.result.Result.Err step.StepError.Overflow

/-- **`gate_product` refinement.** The extracted body returns the narrowed Q32.32 product, or the
`Overflow` refusal when it leaves `i64`. -/
theorem word_gate_product_ok (a s : I64) :
    step.attn_block.gate_product a s = ok (stp_gateProdRef a.val s.val) := by
  unfold step.attn_block.gate_product
  rw [word_mul_q32_ok a s]
  simp only [bind_tc_ok]
  by_cases hc : -(2 ^ 63) ≤ roundShift32 (a.val * s.val) ∧
      roundShift32 (a.val * s.val) < 2 ^ 63
  · rw [stp_mulQ32Ref, if_pos hc, Option.map, core.option.Option.ok_or_some,
      stp_gateProdRef, if_pos hc]
  · rw [stp_mulQ32Ref, if_neg hc, Option.map, core.option.Option.ok_or_none,
      stp_gateProdRef, if_neg hc]

/-- **`gate_product` refusal.** When the reference product leaves `i64` the body returns `Overflow`. -/
theorem word_gate_product_refuses (a s : I64) (h : stp_mulQ32Ref a.val s.val = none) :
    step.attn_block.gate_product a s = ok (core.result.Result.Err step.StepError.Overflow) := by
  have hc : ¬ (-(2 ^ 63) ≤ roundShift32 (a.val * s.val) ∧
      roundShift32 (a.val * s.val) < 2 ^ 63) := by
    intro hcc
    rw [stp_mulQ32Ref, if_pos hcc] at h
    exact absurd h (by simp)
  rw [word_gate_product_ok, stp_gateProdRef, if_neg hc]

/-- Non-vacuity: `0 · 0` narrows to `0` (`Ok`); the in-range arm is exercised. -/
example : stp_gateProdRef 0 0 = core.result.Result.Ok (stp_i64Of 0) := by
  unfold stp_gateProdRef roundShift32
  norm_num [Int.shiftRight_eq_div_pow]

/-- Non-vacuity: a product past `i64` refuses with `Overflow`. -/
example : stp_gateProdRef (2 ^ 62) (2 ^ 62) =
    core.result.Result.Err step.StepError.Overflow := by
  unfold stp_gateProdRef roundShift32
  norm_num [Int.shiftRight_eq_div_pow]

/-! ## `bytes_at` and `bytes_eq`

`src/engine/step.rs:686-699`: `bytes_at hay at needle` is `true` iff the `needle.length` bytes at
`hay[at..]` equal `needle`, guarded by `at + needle.length ≤ hay.length` (else `false`). The loop
accumulates the byte comparisons with `&&`. `:722-724`: `bytes_eq a b` is `bytes_at a 0 b` when the
lengths agree, `false` otherwise. -/

/-- The loop accumulator's reference: `c` byte comparisons starting at `hay[off + j]` / `needle[j]`. -/
def stp_bytesAtPrefix (hay needle : List U8) (off j : ℕ) : ℕ → Bool
  | 0 => true
  | (c + 1) => (decide (hay[off + j + c]? = needle[j + c]?) &&
                stp_bytesAtPrefix hay needle off j c)

/-- `bytes_at`'s loop from index `k`: the comparisons from `k` to `needle.length - 1`. -/
def stp_bytesAtAll (hay needle : List U8) (off k : ℕ) : Bool :=
  stp_bytesAtPrefix hay needle off k (needle.length - k)

/-- `bytes_at`'s reference: the guard arm (`false` when `needle` cannot fit) else the byte match. -/
def stp_bytesAt (hay needle : List U8) (off : ℕ) : Bool :=
  if off + needle.length ≤ hay.length then stp_bytesAtAll hay needle off 0 else false

theorem stp_bytesAtPrefix_succ (hay needle : List U8) (off j c : ℕ) :
    stp_bytesAtPrefix hay needle off j (c + 1) =
      (decide (hay[off + j + c]? = needle[j + c]?) &&
        stp_bytesAtPrefix hay needle off j c) := rfl

theorem stp_bytesAtPrefix_succ' (hay needle : List U8) (off k k' : ℕ) (h : k ≤ k') :
    stp_bytesAtPrefix hay needle off k (k' + 1 - k) =
      (decide (hay[off + k']? = needle[k']?) && stp_bytesAtPrefix hay needle off k (k' - k)) := by
  rw [show k' + 1 - k = (k' - k) + 1 from by omega, stp_bytesAtPrefix_succ]
  simp only [show off + k + (k' - k) = off + k' from by omega,
    show k + (k' - k) = k' from by omega]

theorem stp_bytesAtPrefix_step (hay needle : List U8) (off k k' : ℕ) (h : k ≤ k')
    (heq : decide (hay[off + k']? = needle[k']?) = true) :
    stp_bytesAtPrefix hay needle off k (k' + 1 - k) =
      stp_bytesAtPrefix hay needle off k (k' - k) := by
  rw [stp_bytesAtPrefix_succ' hay needle off k k' h, heq, Bool.true_and]

theorem stp_bytesAtPrefix_step_false (hay needle : List U8) (off k k' : ℕ) (h : k ≤ k')
    (hne : decide (hay[off + k']? = needle[k']?) = false) :
    stp_bytesAtPrefix hay needle off k (k' + 1 - k) = false := by
  rw [stp_bytesAtPrefix_succ' hay needle off k k' h, hne, Bool.false_and]

/-- The `bytes_at_loop` invariant: index in range, accumulator equals the start value conjoined with
the comparisons over `[k, st.1)`. -/
def stp_bytesAtInv (hay needle : List U8) (off k : ℕ) (same : Bool) (st : Usize × Bool) : Prop :=
  k ≤ st.1.val ∧ st.1.val ≤ needle.length ∧
    st.2 = (same && stp_bytesAtPrefix hay needle off k (st.1.val - k))

/-- Equal `Slice.index_usize` results give equal `getElem?` bytes at `off + k'` / `k'`. -/
theorem stp_byte_eq_decide (hay needle : Slice U8) (off k' i1 : Usize) (i2 i3 : U8)
    (hk' : k'.val < needle.val.length) (hfit : off.val + needle.val.length ≤ hay.val.length)
    (hi1 : i1.val = off.val + k'.val) (h2 : i2 = hay.val[i1.val]'(by omega))
    (h3 : i3 = needle.val[k'.val]'(by omega)) (heq : i2 = i3) :
    decide (hay.val[off.val + k'.val]? = needle.val[k'.val]?) = true := by
  have hA : hay.val[off.val + k'.val]? = some i2 := by
    have h := List.getElem?_eq_getElem (show i1.val < hay.val.length from by omega)
    rw [← h2] at h
    rw [hi1] at h
    exact h
  have hB : needle.val[k'.val]? = some i3 := by
    have h := List.getElem?_eq_getElem hk'
    rw [← h3] at h
    exact h
  rw [hA, hB, decide_eq_true_eq]
  exact congrArg some heq

/-- Differing `Slice.index_usize` results give differing `getElem?` bytes. -/
theorem stp_byte_ne_decide (hay needle : Slice U8) (off k' i1 : Usize) (i2 i3 : U8)
    (hk' : k'.val < needle.val.length) (hfit : off.val + needle.val.length ≤ hay.val.length)
    (hi1 : i1.val = off.val + k'.val) (h2 : i2 = hay.val[i1.val]'(by omega))
    (h3 : i3 = needle.val[k'.val]'(by omega)) (hne : i2 ≠ i3) :
    decide (hay.val[off.val + k'.val]? = needle.val[k'.val]?) = false := by
  have hA : hay.val[off.val + k'.val]? = some i2 := by
    have h := List.getElem?_eq_getElem (show i1.val < hay.val.length from by omega)
    rw [← h2] at h
    rw [hi1] at h
    exact h
  have hB : needle.val[k'.val]? = some i3 := by
    have h := List.getElem?_eq_getElem hk'
    rw [← h3] at h
    exact h
  rw [hA, hB, decide_eq_false_iff_not]
  exact fun heq => hne (Option.some.inj heq)

/-- **`bytes_at` loop body.** One iteration refines the invariant and decreases the measure. -/
theorem stp_bytesAtBody_spec (hay needle : Slice U8) (off : Usize) (k : Usize) (same : Bool)
    (hb : off.val + needle.val.length ≤ hay.val.length)
    (st : Usize × Bool) (hst : stp_bytesAtInv hay.val needle.val off.val k.val same st) :
    step.bytes_at_loop.body hay off needle st.1 st.2 ⦃ r =>
      match r with
      | .done y => y = (same && stp_bytesAtAll hay.val needle.val off.val k.val)
      | .cont st' => stp_bytesAtInv hay.val needle.val off.val k.val same st' ∧
          needle.val.length - st'.1.val < needle.val.length - st.1.val ⦄ := by
  rcases st with ⟨k', s'⟩
  obtain ⟨hge, hle, hs⟩ := hst
  dsimp only at hge hle hs
  unfold step.bytes_at_loop.body
  dsimp only
  by_cases hlt : k' < Slice.len needle
  · rw [if_pos hlt]
    have hlt' : k'.val < needle.val.length := by
      have := (UScalar.lt_equiv _ _).1 hlt
      simpa [Slice.len_val, Slice.length] using this
    step*
    by_cases hcmp : (i2 != i3) = true
    · have hne : i2 ≠ i3 := bne_iff_ne.mp hcmp
      have hpf : decide (hay.val[off.val + k'.val]? = needle.val[k'.val]?) = false :=
        stp_byte_ne_decide hay needle off k' i1 i2 i3 hlt' hb i1_post i2_post i3_post hne
      rw [if_pos hcmp]
      step*
      refine ⟨?_, ?_⟩
      · unfold stp_bytesAtInv
        dsimp only
        refine ⟨?_, ?_, ?_⟩
        · omega
        · omega
        · rw [show k1.val - k.val = k'.val + 1 - k.val from by omega,
              stp_bytesAtPrefix_step_false hay.val needle.val off.val k.val k'.val hge hpf]
          simp only [Bool.and_false]
      · omega
    · have heq : i2 = i3 := by by_contra h; exact hcmp (bne_iff_ne.mpr h)
      have hpt : decide (hay.val[off.val + k'.val]? = needle.val[k'.val]?) = true :=
        stp_byte_eq_decide hay needle off k' i1 i2 i3 hlt' hb i1_post i2_post i3_post heq
      rw [if_neg hcmp]
      step*
      refine ⟨?_, ?_⟩
      · unfold stp_bytesAtInv
        dsimp only
        refine ⟨?_, ?_, ?_⟩
        · omega
        · omega
        · rw [show k1.val - k.val = k'.val + 1 - k.val from by omega,
              stp_bytesAtPrefix_step hay.val needle.val off.val k.val k'.val hge hpt]
          exact hs
      · omega
  · rw [if_neg hlt]
    have hge' : needle.val.length ≤ k'.val := by
      have := (UScalar.le_equiv _ _).1 (le_of_not_gt hlt)
      simpa [Slice.len_val, Slice.length] using this
    have hkn : k'.val = needle.val.length := by omega
    step*
    rw [hs, hkn]
    rfl

/-- **`bytes_at` loop.** From any state `(k, same)` with `k ≤ needle.length` and the fit guard, the
scan returns `same` conjoined with the reference comparison. -/
theorem stp_bytesAtLoop_spec (hay needle : Slice U8) (off k : Usize) (same : Bool)
    (hk : k.val ≤ needle.val.length) (hb : off.val + needle.val.length ≤ hay.val.length) :
    step.bytes_at_loop hay off needle k same ⦃ b =>
      b = (same && stp_bytesAtAll hay.val needle.val off.val k.val) ⦄ := by
  unfold step.bytes_at_loop
  apply loop.spec_decr_nat
    (measure := fun (st : Usize × Bool) => needle.val.length - st.1.val)
    (inv := stp_bytesAtInv hay.val needle.val off.val k.val same)
    (post := fun (b : Bool) => b = (same && stp_bytesAtAll hay.val needle.val off.val k.val))
  · intro st hst
    convert stp_bytesAtBody_spec hay needle off k same hb st hst using 1
    funext r
    cases r with
    | done y => rfl
    | cont st' => rfl
  · unfold stp_bytesAtInv
    refine ⟨le_refl _, hk, ?_⟩
    simp [stp_bytesAtPrefix]

/-- **`bytes_at` refinement.** With the fit guard satisfied (or not) the scan returns the reference;
the two length guards and the checked `hay.len - at` are exactly `stp_bytesAt`'s guard. -/
theorem word_bytes_at_ok (hay : Slice U8) (p : Usize) (needle : Slice U8) :
    step.bytes_at hay p needle = ok (stp_bytesAt hay.val needle.val p.val) := by
  have h : step.bytes_at hay p needle ⦃ b => b = stp_bytesAt hay.val needle.val p.val ⦄ := by
    unfold step.bytes_at
    dsimp only
    by_cases hgt : p > Slice.len hay
    · rw [if_pos hgt, WP.spec_ok]
      rw [stp_bytesAt, if_neg (by
        have := (UScalar.lt_equiv (Slice.len hay) p).1 hgt
        simp only [Slice.len_val, Slice.length] at this
        omega)]
    · rw [if_neg hgt]
      have hlev : p.val ≤ hay.val.length := by
        have := (UScalar.le_equiv p (Slice.len hay)).1 (le_of_not_gt hgt)
        simp only [Slice.len_val, Slice.length] at this
        exact this
      step*
      · rename_i hif
        have hlt : i2.val < needle.val.length := by
          have := (UScalar.lt_equiv i2 (Slice.len needle)).1 hif
          simpa [Slice.len_val, Slice.length] using this
        have hs1 : (Slice.len hay).val = hay.val.length := Slice.len_val hay
        have hcon : ¬ (p.val + needle.val.length ≤ hay.val.length) := by omega
        rw [stp_bytesAt, if_neg hcon]
      · rename_i hnif
        have hs1 : (Slice.len hay).val = hay.val.length := Slice.len_val hay
        have hnle : needle.val.length ≤ i2.val := by
          have := (UScalar.le_equiv (Slice.len needle) i2).1 (le_of_not_gt hnif)
          simpa [Slice.len_val, Slice.length] using this
        have hb : p.val + needle.val.length ≤ hay.val.length := by omega
        have hloop := stp_bytesAtLoop_spec hay needle p 0#usize true (by simp) hb
        have hconv : (true && stp_bytesAtAll hay.val needle.val p.val (0#usize).val) =
            stp_bytesAt hay.val needle.val p.val := by
          rw [stp_bytesAt, if_pos hb, Bool.true_and]
          rfl
        apply WP.spec_mono hloop
        intro z hz
        exact hz.trans hconv
  exact stp_spec_eq h

/-- **`bytes_eq` reference.** The bytes agree iff the lengths agree and the `bytes_at` scan from `0`
agrees. -/
def stp_bytesEq (a b : List U8) : Bool :=
  if a.length = b.length then stp_bytesAt a b 0 else false

/-- **`bytes_eq` refinement.** The length test chooses between the `bytes_at` scan and `false`;
the second scan's checked `a.len - 0` is the identity, so the guard is the length equality. -/
theorem word_bytes_eq_ok (a b : Slice U8) :
    step.bytes_eq a b = ok (stp_bytesEq a.val b.val) := by
  unfold step.bytes_eq
  dsimp only
  by_cases hlen : Slice.len a = Slice.len b
  · rw [if_pos hlen, word_bytes_at_ok]
    have hlenv : a.val.length = b.val.length := by
      have := congrArg UScalar.val hlen
      simpa [Slice.len_val, Slice.length] using this
    rw [show (0#usize : Std.Usize).val = 0 from rfl, stp_bytesEq, if_pos hlenv]
  · rw [if_neg hlen]
    have hlenv : a.val.length ≠ b.val.length := by
      intro hc
      exact hlen ((UScalar.eq_equiv (Slice.len a) (Slice.len b)).2 (by
        simpa [Slice.len_val, Slice.length] using hc))
    rw [stp_bytesEq, if_neg hlenv]

/-- Non-vacuity: equal bytes agree, a differing byte disagrees, and unequal lengths disagree. -/
example : stp_bytesEq [1#u8, 2#u8, 3#u8] [1#u8, 2#u8, 3#u8] = true := by decide

example : stp_bytesEq [1#u8, 2#u8, 3#u8] [1#u8, 9#u8, 3#u8] = false := by decide

example : stp_bytesEq [1#u8] [1#u8, 2#u8] = false := by decide

/-! ## `any_vec_len_ne`

`src/engine/step.rs:706-715`: `true` iff some vector of `xs` has length other than `n`. The
reference scans the length list `stp_vecLens xs` for an entry `≠ n`. -/

/-- The lengths of the vectors of `xs`. -/
def stp_vecLens {T : Type} (xs : Slice (alloc.vec.Vec T)) : List ℕ :=
  xs.val.map (fun v => (alloc.vec.Vec.len v).val)

/-- The `any_vec_len_ne` reference: `true` iff some vector of `xs` has length other than `n`. -/
def stp_anyVecLenNe {T : Type} (xs : Slice (alloc.vec.Vec T)) (n : Usize) : Bool :=
  (stp_vecLens xs).any (fun m => decide (m ≠ n.val))

theorem stp_any_decide_true {l : List ℕ} {n m : ℕ} (hm : m ∈ l) (hne : m ≠ n) :
    l.any (fun m => decide (m ≠ n)) = true := by
  rw [List.any_eq_true]
  exact ⟨m, hm, by rw [decide_eq_true_eq]; exact hne⟩

theorem stp_any_decide_false {l : List ℕ} {n : ℕ} (h : ∀ m ∈ l, m = n) :
    l.any (fun m => decide (m ≠ n)) = false := by
  rw [List.any_eq_false]
  intro m hm hmt
  rw [decide_eq_true_eq] at hmt
  exact hmt (h m hm)

theorem stp_vecLens_getElem? {T : Type} (xs : Slice (alloc.vec.Vec T)) (j : ℕ)
    (hj : j < xs.val.length) :
    (stp_vecLens xs)[j]? = some ((alloc.vec.Vec.len (xs.val[j])).val) := by
  unfold stp_vecLens
  rw [List.getElem?_map, List.getElem?_eq_getElem hj]
  rfl

theorem stp_vecLens_length {T : Type} (xs : Slice (alloc.vec.Vec T)) :
    (stp_vecLens xs).length = xs.val.length := by
  simp [stp_vecLens]

theorem stp_replicate_append (n : ℕ) (a : ℕ) :
    List.replicate n a ++ [a] = a :: List.replicate n a := by
  induction n with
  | zero => rfl
  | succ k ih => rw [List.replicate_succ, List.cons_append, ih]

theorem stp_replicate_succ (n : ℕ) (a : ℕ) :
    List.replicate (n + 1) a = List.replicate n a ++ [a] := by
  rw [List.replicate_succ, stp_replicate_append]

/-- **`any_vec_len_ne` loop.** From `0` the scan returns `stp_anyVecLenNe`. -/
theorem stp_anyVecLenNe_loop_spec {T : Type} (xs : Slice (alloc.vec.Vec T)) (n : Usize) :
    step.any_vec_len_ne_loop xs n 0#usize ⦃ b => b = stp_anyVecLenNe xs n ⦄ := by
  unfold step.any_vec_len_ne_loop
  apply loop.spec_decr_nat
    (measure := fun (st : Usize) => xs.val.length - st.val)
    (inv := fun (st : Usize) => st.val ≤ xs.val.length ∧
      (stp_vecLens xs).take st.val = List.replicate st.val n.val)
    (post := fun (b : Bool) => b = stp_anyVecLenNe xs n)
  · intro st hinv'
    obtain ⟨hle, htake⟩ := hinv'
    unfold step.any_vec_len_ne_loop.body
    dsimp only
    by_cases hlt : st < Slice.len xs
    · rw [if_pos hlt]
      have hlt' : st.val < xs.val.length := by
        have := (UScalar.lt_equiv _ _).1 hlt
        simpa [Slice.len_val, Slice.length] using this
      step*
      · rename_i hbne
        have hmem : (alloc.vec.Vec.len v).val ∈ stp_vecLens xs := by
          rw [stp_vecLens]
          refine List.mem_map.mpr ⟨xs.val[st.val], List.getElem_mem hlt', ?_⟩
          rw [← v_post]
        have hne : (alloc.vec.Vec.len v).val ≠ n.val := by
          intro hc
          exact (bne_iff_ne.mp hbne) ((UScalar.eq_equiv _ _).2 hc)
        exact (stp_any_decide_true hmem hne).symm
      · rename_i hbnf
        refine ⟨by omega, ?_, by omega⟩
        have hvlen : (alloc.vec.Vec.len v).val = n.val := by
          rw [bne_iff_ne] at hbnf
          exact (UScalar.eq_equiv _ _).1 (not_not.mp hbnf)
        have hlast : (alloc.vec.Vec.len (xs.val[st.val])).val = n.val := by
          rw [← v_post]; exact hvlen
        rw [i3_post, List.take_add_one, htake, stp_vecLens_getElem? xs st.val hlt',
          Option.toList_some, stp_replicate_succ, hlast]
    · rw [if_neg hlt]
      have hge : xs.val.length ≤ st.val := by
        have := (UScalar.le_equiv _ _).1 (le_of_not_gt hlt)
        simpa [Slice.len_val, Slice.length] using this
      have hlens : stp_vecLens xs = List.replicate st.val n.val := by
        have h1 : (stp_vecLens xs).take st.val = stp_vecLens xs :=
          List.take_of_length_le (by rw [stp_vecLens_length]; omega)
        rw [← h1]; exact htake
      have hdrop : ∀ m ∈ stp_vecLens xs, m = n.val := by
        intro m hm
        rw [hlens] at hm
        exact List.eq_of_mem_replicate hm
      rw [WP.spec_ok, stp_anyVecLenNe]
      exact (stp_any_decide_false hdrop).symm
  · constructor
    · rw [show (0#usize : Std.Usize).val = 0 from rfl]; exact Nat.zero_le _
    · rw [show (0#usize : Std.Usize).val = 0 from rfl]; rfl

/-- **`any_vec_len_ne` refinement.** The scan returns whether some vector differs in length. -/
theorem word_any_vec_len_ne_ok {T : Type} (xs : Slice (alloc.vec.Vec T)) (n : Usize) :
    step.any_vec_len_ne xs n = ok (stp_anyVecLenNe xs n) := by
  have h : step.any_vec_len_ne xs n ⦃ b => b = stp_anyVecLenNe xs n ⦄ := by
    unfold step.any_vec_len_ne
    exact stp_anyVecLenNe_loop_spec xs n
  exact stp_spec_eq h

/-- Non-vacuity: with `n = 0` both vectors differ (`true`); with `n = 1` the first agrees
and the second differs (`true`), while equal lengths give `false`. -/
example : stp_anyVecLenNe (Slice.from [alloc.vec.Vec.from [1#u64] (by scalar_tac),
    alloc.vec.Vec.from [2#u64] (by scalar_tac)] (by scalar_tac)) 0#usize = true := by decide

example : stp_anyVecLenNe (Slice.from [alloc.vec.Vec.from [1#u64] (by scalar_tac),
    alloc.vec.Vec.from [2#u64, 3#u64] (by scalar_tac)] (by scalar_tac)) 1#usize = true := by decide

example : stp_anyVecLenNe (Slice.from [alloc.vec.Vec.from [7#u64] (by scalar_tac),
    alloc.vec.Vec.from [7#u64] (by scalar_tac)] (by scalar_tac)) 1#usize = false := by decide

/-! ## Boundary: functions with no refinement theorem in this pass

Each entry names the exact artefact that blocks a refinement here; none of them is a claim that the
Rust is wrong, and none is a `sorry`.

* **`sigmoid_lanes`** (`attn_block.rs:104-159`; extracted `step.attn_block.sigmoid_lanes`), and its
  two consumers **`sigmoid_gate`** (`:172-206`) and **`apply_sigmoid_gate`** (`step.rs:516-518`,
  extracted as a direct call of `sigmoid_gate`). `sigmoid_lanes`'s middle stage enumerates seven
  constants through `core.array.iter.IntoIter`, extracted as an **opaque type axiom**
  (`ApeironWord.lean:162`) with an **opaque `next` axiom** (`ApeironWord.lean:180`) and no model in
  `target/aeneas-model/lean/Aeneas/Std/`. The value returned by that iterator, and therefore by
  `sigmoid_lanes`, is not determined by the extraction; no theorem about its output is derivable
  from these bodies. The other three stages of `sigmoid_lanes` are ordinary loops.

* **`rotate_pair_prefixes`** (`attn_block.rs:237-307`). All callees are `def`s
  (`fixed_trig.fixed_sin_cos` `ApeironWord.lean:56930`, `rope.RopeTable.impl.turns` `:56761`,
  `value.round_shift_32`), so this is not an axiom boundary, but it is a five-level nested loop
  whose inner levels (`rotate_pair_prefixes_loop0_loop0_loop1/loop2`) mutate a sub-slice obtained by
  `core.slice.index.Slice.index_mut` and write it back through the returned `index_mut_back` closure.
  A refinement needs a frame specification for that borrow combinator; the loop is not attempted to
  closure in this pass, and no partial statement is presented as its result.

* **`affine_slots`** (`step.rs:442-454`) and the identically shaped **`softmax_slots`**
  (`step.rs:423-433`). The loop is over `core.slice.iter.Iter` (`SliceIter.lean:14`); its step
  `core.slice.iter.IteratorSliceIter.next` is a model `def` (`SliceIter.lean:62`) but carries no
  `@[step]` spec, and the branch value comes from `read_discriminant`. Measured here: the
  `read_discriminant` instance projection does not unfold at the transparency `rfl` reaches from
  `ApeironWord` (`rfl` cannot close `(read_discriminant MixerKind.Softmax).val = 0`), the same
  blocker recorded at `Numeric/Engine/Attn.lean:100`; so the branch taken at a given `MixerKind` is
  not derivable, and the iterator carries a bespoke invariant. Not attempted to closure in this
  pass.

The float carriers (`F32`/`F64`) and the AVX2 paths are out of scope (boundaries). -/

-- Axiom cone of each headline refinement (must be ⊆ {propext, Classical.choice, Quot.sound}).

end apeiron_word
