import ApeironWord
import Learn.Grad.DenseLinear
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The `src/learn/grad/dense.rs` linear passes: branch witnesses

The refinements in `Learn.Grad.DenseLinear` are conditional on an explicit admissible domain. This
module shows the domain and each refusal arm are inhabited, so neither the headline theorems nor the
refusals are vacuous:

* one reachable success-branch input per pass (`n = i = o = 1`, all-zero operands), by instantiating
  the headline theorem;
* one reachable input per refusal arm the three passes can construct: `Empty` (`expect_len` on a
  zero dimension, any operands), `Shape` (`expect_len` on a length mismatch), `Overflow`
  (`expect_len` on a dimension product above `Usize.max`, any operands);
* `Range` is unreachable on these paths: `expect_len` never constructs it, and the accumulators and
  the Q32.32 rounder construct only `Overflow` (`round_all_spec`, `linear_*_acc_spec`). The
  `RMSNorm`/`SiLU` bodies that do construct `Range` are not called by these three passes.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## 8. Branch witnesses -/

/-- `Try::branch` on `Err` yields `Break (Err e)`. -/
lemma dense_branch_err {T ε : Type} (e : ε) :
    core.result.Result.Insts.CoreOpsTry.branch (T := T) (E := ε) (core.result.Result.Err e) =
      ok (core.ops.control_flow.ControlFlow.Break (core.result.Result.Err e)) := rfl

/-- The all-zero one-element `i64` slice: the concrete witness operand. -/
def denseOne : Slice I64 :=
  Slice.from [0#i64] (by have h := dense_usize_le_max (1#usize); simpa using h)

/-- The empty `i64` slice. -/
def denseNil : Slice I64 := Slice.new I64

theorem dense_one_ne_zero : (1#usize : Usize) ≠ 0#usize := by
  intro h
  have h1 := congrArg UScalar.val h
  simp at h1

theorem dense_usize_max_ne_zero : core.num.Usize.MAX ≠ 0#usize := by
  intro h
  have h1 : core.num.Usize.MAX.val = (0#usize).val := congrArg UScalar.val h
  have h2 : core.num.Usize.MAX.val = Usize.max := rfl
  rw [h2] at h1
  have h3 : 1 ≤ Usize.max := by have h4 := dense_usize_le_max (1#usize); simpa using h4
  simp at h1
  omega

theorem dense_two_ne_zero : (2#usize : Usize) ≠ 0#usize := by
  intro h
  have h1 := congrArg UScalar.val h
  simp at h1

theorem dense_one_mul_le_max : (1#usize : Usize).val * (1#usize).val ≤ Usize.max := by
  have h := dense_usize_le_max (1#usize); simpa using h

/-- `Usize::MAX · 2` is not a `usize`, so `checked_mul` refuses it. -/
theorem dense_usize_max_mul_two_none : core.num.Usize.MAX.checked_mul 2#usize = none := by
  cases h : core.num.Usize.MAX.checked_mul 2#usize with
  | none => rfl
  | some z =>
    have hspec := Usize.checked_mul_bv_spec (x := core.num.Usize.MAX) (y := 2#usize)
    rw [h] at hspec
    have hx : core.num.Usize.MAX.val = Usize.max := rfl
    have h2 : (2#usize).val = 2 := rfl
    rw [hx, h2] at hspec
    have h1 : 1 ≤ Usize.max := by have h3 := dense_usize_le_max (1#usize); simpa using h3
    nlinarith [hspec.1, h1]

/-- A pair of all-zero operands makes every accumulation zero. -/
theorem dense_dsum_zero (L R : List ℤ) (gL gR : ℕ → ℕ → ℕ → ℕ) (a b : ℕ)
    (hL : ∀ j, L.getD j 0 = 0) (hR : ∀ j, R.getD j 0 = 0) : ∀ t, dsum L R gL gR a b t = 0 := by
  intro t
  induction t with
  | zero => rfl
  | succ k ih => rw [dsum_succ, ih, hL, hR]; rfl

theorem dense_qvals_one : qvals denseOne = [0] := by
  simp [qvals, denseOne, Slice.from_val]

theorem dense_zero_op : ∀ j, (qvals denseOne).getD j 0 = 0 := by
  intro j
  rw [dense_qvals_one]
  cases j <;> rfl

theorem dense_i128_zero_range : I128.min ≤ (0 : ℤ) ∧ (0 : ℤ) ≤ I128.max := by
  rw [I128.min_eq, I128.max_eq]
  norm_num

theorem dense_i128_range_zero : I128.min ≤ (0 : ℤ) + 2 ^ 31 ∧ (0 : ℤ) + 2 ^ 31 ≤ I128.max := by
  rw [I128.min_eq, I128.max_eq]
  norm_num

theorem dense_nar_zero :
    -((2 : ℤ) ^ 63) ≤ rnd32 (0 : ℤ) ∧ rnd32 (0 : ℤ) ≤ (2 : ℤ) ^ 63 - 1 := by
  rw [rnd32_eq_div]
  norm_num

/-- `expect_len` never constructs `Range`. -/
theorem dense_expect_len_ne_range (rows cols len : Usize) (e : learn_port.dense.DenseError)
    (h : learn_port.dense.expect_len rows cols len = ok (core.result.Result.Err e)) :
    e ≠ learn_port.dense.DenseError.Range := by
  unfold learn_port.dense.expect_len at h
  simp only [lift, bind_tc_ok] at h
  split at h
  · rw [Result.ok.injEq] at h; cases h; intro hc; cases hc
  · split at h
    · rw [Result.ok.injEq] at h; cases h; intro hc; cases hc
    · split at h
      · rw [Result.ok.injEq] at h; cases h; intro hc; cases hc
      · split at h
        · simp at h
        · rw [Result.ok.injEq] at h; cases h; intro hc; cases hc

/-! ### `linear_fwd` -/

/-- `linear_fwd` refuses a zero `n` with `Empty` (any operands). -/
theorem linear_fwd_witness_empty (x w : Slice I64) (i o : Usize) :
    learn_port.dense.linear_fwd x w 0#usize i o =
      ok (core.result.Result.Err learn_port.dense.DenseError.Empty) := by
  have h := dense_expect_len_empty 0#usize i (Slice.len x) rfl
  unfold learn_port.dense.linear_fwd
  simp only [h, dense_branch_err, dense_from_residual_err, bind_tc_ok]

/-- `linear_fwd` refuses a length mismatch with `Shape`. -/
theorem linear_fwd_witness_shape :
    learn_port.dense.linear_fwd denseNil denseOne 1#usize 1#usize 1#usize =
      ok (core.result.Result.Err learn_port.dense.DenseError.Shape) := by
  have h := dense_expect_len_shape 1#usize 1#usize (Slice.len denseNil) dense_one_ne_zero
    dense_one_ne_zero dense_one_mul_le_max (by simp [denseNil, Slice.new_val])
  unfold learn_port.dense.linear_fwd
  simp only [h, dense_branch_err, dense_from_residual_err, bind_tc_ok]

/-- `linear_fwd` refuses a dimension product above `Usize.max` with `Overflow` (any operands). -/
theorem linear_fwd_witness_overflow (x w : Slice I64) (o : Usize) :
    learn_port.dense.linear_fwd x w core.num.Usize.MAX 2#usize o =
      ok (core.result.Result.Err learn_port.dense.DenseError.Overflow) := by
  have h := dense_expect_len_overflow core.num.Usize.MAX 2#usize (Slice.len x) dense_usize_max_ne_zero
    dense_two_ne_zero dense_usize_max_mul_two_none
  unfold learn_port.dense.linear_fwd
  simp only [h, dense_branch_err, dense_from_residual_err, bind_tc_ok]

/-- The all-zero `1 × 1` input reaches `linear_fwd`'s success branch. -/
theorem linear_fwd_witness_ok :
    ∃ out, learn_port.dense.linear_fwd denseOne denseOne 1#usize 1#usize 1#usize =
      ok (core.result.Result.Ok out) := by
  obtain ⟨out, hout, _⟩ := linear_fwd_ok denseOne denseOne 1#usize 1#usize 1#usize
    dense_one_ne_zero dense_one_ne_zero dense_one_ne_zero
    (by simp [denseOne, Slice.from_val]) (by simp [denseOne, Slice.from_val])
    dense_one_mul_le_max dense_one_mul_le_max dense_one_mul_le_max
    (by
      intro r' hr' c' hc' t ht
      have h1v : (1#usize : Usize).val = 1 := rfl
      rw [h1v] at hr' hc' ht ⊢
      obtain rfl : r' = 0 := by omega
      obtain rfl : c' = 0 := by omega
      have hz : fwdK (qvals denseOne) (qvals denseOne) 1 0 0 t = 0 :=
        dense_dsum_zero _ _ _ _ 0 0 dense_zero_op dense_zero_op t
      rw [hz]
      exact dense_i128_zero_range)
    (by
      intro t ht
      have h1v : (1#usize : Usize).val = 1 := rfl
      rw [h1v] at ht ⊢
      obtain rfl : t = 0 := by omega
      have hz : fwdK (qvals denseOne) (qvals denseOne) 1 0 0 1 = 0 :=
        dense_dsum_zero _ _ _ _ 0 0 dense_zero_op dense_zero_op 1
      have hg : (fwdRows (qvals denseOne) (qvals denseOne) 1 1 1).getD 0 0 = 0 := by
        simp [fwdRows, fwdCols, pushN, hz]
      rw [hg]
      exact dense_i128_range_zero)
    (by
      intro t ht
      have h1v : (1#usize : Usize).val = 1 := rfl
      rw [h1v] at ht ⊢
      obtain rfl : t = 0 := by omega
      have hz : fwdK (qvals denseOne) (qvals denseOne) 1 0 0 1 = 0 :=
        dense_dsum_zero _ _ _ _ 0 0 dense_zero_op dense_zero_op 1
      have hg : (fwdRows (qvals denseOne) (qvals denseOne) 1 1 1).getD 0 0 = 0 := by
        simp [fwdRows, fwdCols, pushN, hz]
      rw [hg]
      exact dense_nar_zero)
  exact ⟨out, hout⟩

/-! ### `linear_bwd_input` -/

/-- `linear_bwd_input` refuses a zero `n` with `Empty` (any operands). -/
theorem linear_bwd_input_witness_empty (dy w : Slice I64) (i o : Usize) :
    learn_port.dense.linear_bwd_input dy w 0#usize i o =
      ok (core.result.Result.Err learn_port.dense.DenseError.Empty) := by
  have h := dense_expect_len_empty 0#usize o (Slice.len dy) rfl
  unfold learn_port.dense.linear_bwd_input
  simp only [h, dense_branch_err, dense_from_residual_err, bind_tc_ok]

/-- `linear_bwd_input` refuses a length mismatch with `Shape`. -/
theorem linear_bwd_input_witness_shape :
    learn_port.dense.linear_bwd_input denseNil denseOne 1#usize 1#usize 1#usize =
      ok (core.result.Result.Err learn_port.dense.DenseError.Shape) := by
  have h := dense_expect_len_shape 1#usize 1#usize (Slice.len denseNil) dense_one_ne_zero
    dense_one_ne_zero dense_one_mul_le_max (by simp [denseNil, Slice.new_val])
  unfold learn_port.dense.linear_bwd_input
  simp only [h, dense_branch_err, dense_from_residual_err, bind_tc_ok]

/-- `linear_bwd_input` refuses `n · o` above `Usize.max` with `Overflow` (any operands). -/
theorem linear_bwd_input_witness_overflow (dy w : Slice I64) (i : Usize) :
    learn_port.dense.linear_bwd_input dy w core.num.Usize.MAX i 2#usize =
      ok (core.result.Result.Err learn_port.dense.DenseError.Overflow) := by
  have h := dense_expect_len_overflow core.num.Usize.MAX 2#usize (Slice.len dy) dense_usize_max_ne_zero
    dense_two_ne_zero dense_usize_max_mul_two_none
  unfold learn_port.dense.linear_bwd_input
  simp only [h, dense_branch_err, dense_from_residual_err, bind_tc_ok]

/-- The all-zero `1 × 1` input reaches `linear_bwd_input`'s success branch. -/
theorem linear_bwd_input_witness_ok :
    ∃ out, learn_port.dense.linear_bwd_input denseOne denseOne 1#usize 1#usize 1#usize =
      ok (core.result.Result.Ok out) := by
  obtain ⟨out, hout, _⟩ := linear_bwd_input_ok denseOne denseOne 1#usize 1#usize 1#usize
    dense_one_ne_zero dense_one_ne_zero dense_one_ne_zero
    (by simp [denseOne, Slice.from_val]) (by simp [denseOne, Slice.from_val])
    dense_one_mul_le_max dense_one_mul_le_max dense_one_mul_le_max
    (by
      intro r' hr' k' hk' t ht
      have h1v : (1#usize : Usize).val = 1 := rfl
      rw [h1v] at hr' hk' ht
      obtain rfl : r' = 0 := by omega
      obtain rfl : k' = 0 := by omega
      have hz : bwdIAcc denseOne denseOne 1#usize 1#usize 0 0 t = 0 :=
        dense_dsum_zero _ _ _ _ 0 0 dense_zero_op dense_zero_op t
      rw [hz]
      exact dense_i128_zero_range)
    (by
      intro t ht
      have h1v : (1#usize : Usize).val = 1 := rfl
      rw [h1v] at ht ⊢
      obtain rfl : t = 0 := by omega
      have hz : bwdIAcc denseOne denseOne 1#usize 1#usize 0 0 1 = 0 :=
        dense_dsum_zero _ _ _ _ 0 0 dense_zero_op dense_zero_op 1
      have hg : (rowsOf 1 (bwdIElem denseOne denseOne 1#usize 1#usize) 1).getD 0 0 = 0 := by
        simp [rowsOf, pushN, bwdIElem, hz]
      rw [hg]
      exact dense_i128_range_zero)
    (by
      intro t ht
      have h1v : (1#usize : Usize).val = 1 := rfl
      rw [h1v] at ht ⊢
      obtain rfl : t = 0 := by omega
      have hz : bwdIAcc denseOne denseOne 1#usize 1#usize 0 0 1 = 0 :=
        dense_dsum_zero _ _ _ _ 0 0 dense_zero_op dense_zero_op 1
      have hg : (rowsOf 1 (bwdIElem denseOne denseOne 1#usize 1#usize) 1).getD 0 0 = 0 := by
        simp [rowsOf, pushN, bwdIElem, hz]
      rw [hg]
      exact dense_nar_zero)
  exact ⟨out, hout⟩

/-! ### `linear_bwd_weight` -/

/-- `linear_bwd_weight` refuses a zero `n` with `Empty` (any operands). -/
theorem linear_bwd_weight_witness_empty (dy x : Slice I64) (i o : Usize) :
    learn_port.dense.linear_bwd_weight dy x 0#usize i o =
      ok (core.result.Result.Err learn_port.dense.DenseError.Empty) := by
  have h := dense_expect_len_empty 0#usize o (Slice.len dy) rfl
  unfold learn_port.dense.linear_bwd_weight
  simp only [h, dense_branch_err, dense_from_residual_err, bind_tc_ok]

/-- `linear_bwd_weight` refuses a length mismatch with `Shape`. -/
theorem linear_bwd_weight_witness_shape :
    learn_port.dense.linear_bwd_weight denseNil denseOne 1#usize 1#usize 1#usize =
      ok (core.result.Result.Err learn_port.dense.DenseError.Shape) := by
  have h := dense_expect_len_shape 1#usize 1#usize (Slice.len denseNil) dense_one_ne_zero
    dense_one_ne_zero dense_one_mul_le_max (by simp [denseNil, Slice.new_val])
  unfold learn_port.dense.linear_bwd_weight
  simp only [h, dense_branch_err, dense_from_residual_err, bind_tc_ok]

/-- `linear_bwd_weight` refuses `n · o` above `Usize.max` with `Overflow` (any operands). -/
theorem linear_bwd_weight_witness_overflow (dy x : Slice I64) (i : Usize) :
    learn_port.dense.linear_bwd_weight dy x core.num.Usize.MAX i 2#usize =
      ok (core.result.Result.Err learn_port.dense.DenseError.Overflow) := by
  have h := dense_expect_len_overflow core.num.Usize.MAX 2#usize (Slice.len dy) dense_usize_max_ne_zero
    dense_two_ne_zero dense_usize_max_mul_two_none
  unfold learn_port.dense.linear_bwd_weight
  simp only [h, dense_branch_err, dense_from_residual_err, bind_tc_ok]

/-- The all-zero `1 × 1` input reaches `linear_bwd_weight`'s success branch. -/
theorem linear_bwd_weight_witness_ok :
    ∃ out, learn_port.dense.linear_bwd_weight denseOne denseOne 1#usize 1#usize 1#usize =
      ok (core.result.Result.Ok out) := by
  obtain ⟨out, hout, _⟩ := linear_bwd_weight_ok denseOne denseOne 1#usize 1#usize 1#usize
    dense_one_ne_zero dense_one_ne_zero dense_one_ne_zero
    (by simp [denseOne, Slice.from_val]) (by simp [denseOne, Slice.from_val])
    dense_one_mul_le_max dense_one_mul_le_max dense_one_mul_le_max
    (by
      intro c' hc' k' hk' t ht
      have h1v : (1#usize : Usize).val = 1 := rfl
      rw [h1v] at hc' hk' ht
      obtain rfl : c' = 0 := by omega
      obtain rfl : k' = 0 := by omega
      have hz : bwdWAcc denseOne denseOne 1#usize 1#usize 0 0 t = 0 :=
        dense_dsum_zero _ _ _ _ 0 0 dense_zero_op dense_zero_op t
      rw [hz]
      exact dense_i128_zero_range)
    (by
      intro t ht
      have h1v : (1#usize : Usize).val = 1 := rfl
      rw [h1v] at ht ⊢
      obtain rfl : t = 0 := by omega
      have hz : bwdWAcc denseOne denseOne 1#usize 1#usize 0 0 1 = 0 :=
        dense_dsum_zero _ _ _ _ 0 0 dense_zero_op dense_zero_op 1
      have hg : (rowsOf 1 (bwdWElem denseOne denseOne 1#usize 1#usize 1#usize) 1).getD 0 0 = 0 := by
        simp [rowsOf, pushN, bwdWElem, hz]
      rw [hg]
      exact dense_i128_range_zero)
    (by
      intro t ht
      have h1v : (1#usize : Usize).val = 1 := rfl
      rw [h1v] at ht ⊢
      obtain rfl : t = 0 := by omega
      have hz : bwdWAcc denseOne denseOne 1#usize 1#usize 0 0 1 = 0 :=
        dense_dsum_zero _ _ _ _ 0 0 dense_zero_op dense_zero_op 1
      have hg : (rowsOf 1 (bwdWElem denseOne denseOne 1#usize 1#usize 1#usize) 1).getD 0 0 = 0 := by
        simp [rowsOf, pushN, bwdWElem, hz]
      rw [hg]
      exact dense_nar_zero)
  exact ⟨out, hout⟩

end apeiron_word
