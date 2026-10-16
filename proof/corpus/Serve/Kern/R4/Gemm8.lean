/-
Integer and index refinement of the extracted `chart::q8act_fast_bound_ok` walk and driver
(`chart.q8act_fast_bound_ok_loop`, `chart.q8act_fast_bound_ok`), the eighth group of the
`kern::gemm` refinement.

Serving path: `Engine::next_token_state` -> `step::cols` -> `kern::gemm::fast_columns_ok` ->
`chart.q8act_fast_bound_ok`; also reached from `chart.q8act_plan` for the `Q8_0`/`Q5_0` family
bounds. `Serve/Kern/R4/Gemm6.lean` refined the `chart.QuantAct` accessors and `Serve/Kern/R4/Gemm7.lean`
the value of `chart.QuantAct.magnitude`. Both left `chart.q8act_fast_bound_ok` open as a separate
refinement (its loop runs over `core.num.I128.abs` and `I128.saturating_mul`). This module closes it.

The extracted `chart.q8act_fast_bound_ok_loop`, for `b < q.scales.len`, reads `scales[b]`, widens it to
`i128`, takes `I128.abs` and folds the running maximum `s_max`; at `b >= q.scales.len` it returns
`s_max`. The driver then, when `narrow`, computes the checked `i128` products `32 * top` and
`(32 * top) * 127` (either may refuse `integerOverflow`), saturating-multiplies `s_max` by their
product, and returns `false` when that exceeds `i128::from(i64::MAX)`; otherwise it saturating-chains
`n_in`, `F16_MAX_FIXED = 65504·2^32`, `top`, `127` and `s_max` and tests the product against
`i128::MAX`.

The reference is plain Lean over `Int`/`Nat`: `gemm8_smaxFrom` is the running maximum of the widened
lane magnitudes, `gemm8_satMul` is `i128::saturating_mul` on `Int`, and `gemm8_fastBoundRef` is the
two-step decision — refuse when the `narrow` scale-side saturating product exceeds `i64::MAX`, else
report whether the `f16`-scaled bound fits `i128`. `word_q8act_fast_bound_loop_ok` states the walk
from `(0, 0)` and `word_q8act_fast_bound_ok` the driver value; `word_q8act_fast_bound_ok_refuses` and
`word_q8act_fast_bound_ok_refuses_scale` state the two checked-product refusals.

NOT refined here, stated rather than skirted:

* `kern.gemm.fast_columns_ok` (the brief's Goal 2): its body calls `chart.QuantAct.len`,
  `chart.QuantAct.impl.scales`, `chart.QuantAct.magnitude` and `chart.q8act_fast_bound_ok`. This
  module supplies the last; `Gemm7` supplies the `magnitude` value; `Gemm6` the accessors. The
  column-loop induction over `kern.gemm.fast_columns_ok_loop` is not closed here.
* `chart.q8act_plan` / `chart.q8act` and their `ActMagnitude` consumers.

Trust boundary: rustc/LLVM codegen, the `F32`/AVX2 layers, the `axiom` pool driver and the
`R[x]/(x^2 - sigma)` word kernels are not covered here. No `sorry`/`admit`/`axiom`/`native_decide`/
`bv_decide`/`maxHeartbeats 0`; `decide` appears only on literal values and literal integer bounds.
-/
import Serve.Kern.R4.Gemm7

set_option autoImplicit false
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## References (plain Lean)

`gemm8_smaxFrom` is the running maximum of the widened lane magnitudes `|scales[j]|`, the empty fold
at `0`; `gemm8_fastBoundRef` is the driver decision over that maximum. -/

/-- `gemm8_smaxFrom l n` is `max_{j<n} |l[j]|` as an `Int`, with the empty fold at `0`. -/
def gemm8_smaxFrom (l : List Std.I64) : Nat → Int
  | 0 => 0
  | n + 1 => Max.max (gemm8_smaxFrom l n) ((l.getD n 0#i64).val.natAbs : Int)

/-- One unfolding of `gemm8_smaxFrom` at the successor index. -/
theorem gemm8_smaxFrom_succ (l : List Std.I64) (n : Nat) :
    gemm8_smaxFrom l (n + 1) =
      Max.max (gemm8_smaxFrom l n) ((l.getD n 0#i64).val.natAbs : Int) := rfl

/-- The running maximum is nonnegative. -/
theorem gemm8_smaxFrom_nonneg (l : List Std.I64) (n : Nat) : 0 ≤ gemm8_smaxFrom l n := by
  induction n with
  | zero => simp [gemm8_smaxFrom]
  | succ k ih => rw [gemm8_smaxFrom_succ]; exact le_trans ih (le_max_left _ _)

/-- The running magnitude maximum over the whole scale slice `q.scales`. -/
def gemm8_smaxRef (q : chart.QuantAct) : Int :=
  gemm8_smaxFrom q.scales.val q.scales.val.length

/-- Loop invariant of `chart.q8act_fast_bound_ok_loop`: the counter stays inside the scale slice and
the accumulator is the running magnitude maximum. -/
def gemm8_smaxInv (q : chart.QuantAct) (st : Std.I128 × Std.Usize) : Prop :=
  st.2.val ≤ q.scales.val.length ∧ st.1.val = gemm8_smaxFrom q.scales.val st.2.val

/-- `65504 · 2^32`, the value of `F16_MAX_FIXED`. -/
def gemm8_f16MaxFixed : Int := 65504 * 2 ^ 32

/-- `i128::saturating_mul` on `Int`: the product clamped to `[i128::MIN, i128::MAX]`. -/
def gemm8_satMul (x y : Int) : Int := max I128.min (min I128.max (x * y))

/-- The pure reference of `chart.q8act_fast_bound_ok`: over the activation scale `top`, the running
magnitude maximum `smax` and the code count `n`. With `narrow`, refuse (`false`) when the scale-side
saturating product `smax · 32 · top · 127` exceeds `i64::MAX`; otherwise report whether the
`f16::MAX`-scaled bound fits `i128`. The `narrow` guard is the Rust short-circuit `narrow && …`, so
the two checked `i128` products are not evaluated when `narrow` is false. -/
def gemm8_fastBoundRef (narrow : Bool) (n : Nat) (top smax : Int) : Bool :=
  if narrow then
    if gemm8_satMul smax (32 * top * 127) > 9223372036854775807 then
      false
    else
      decide
        (gemm8_satMul
            (gemm8_satMul (gemm8_satMul (gemm8_satMul (n : Int) gemm8_f16MaxFixed) top) 127)
            smax <
          170141183460469231731687303715884105727)
  else
    decide
      (gemm8_satMul
          (gemm8_satMul (gemm8_satMul (gemm8_satMul (n : Int) gemm8_f16MaxFixed) top) 127)
          smax <
        170141183460469231731687303715884105727)

/-! ## Value lemmas for the constants and the model operations -/

/-- The extracted `core.num.I64.MAX` is `i64::MAX`. -/
theorem gemm8_I64_MAX_val : core.num.I64.MAX.val = I64.max := by
  simp [core.num.I64.MAX, IScalar.ofInt, I64.rMax, I64.max, I64.numBits]

/-- The extracted `core.num.I128.MAX` is `i128::MAX`. -/
theorem gemm8_I128_MAX_val : core.num.I128.MAX.val = I128.max := by
  simp [core.num.I128.MAX, IScalar.ofInt, I128.rMax, I128.max, I128.numBits]

/-- `i64::MAX` as a literal. -/
theorem gemm8_I64_max_lit : I64.max = 9223372036854775807 := I64.max_eq

/-- `i128::MAX` as a literal. -/
theorem gemm8_I128_max_lit : I128.max = 170141183460469231731687303715884105727 := I128.max_eq

/-- `65504 · 2^32`, the value of `F16_MAX_FIXED`. -/
theorem gemm8_f16Max_lit : (65504 : Int) * 2 ^ 32 = 281337537757184 := by norm_num

/-- The `i128` value of `FromI128I64.from I64.MAX` is `i64::MAX`. -/
theorem gemm8_from_I64MAX_val :
    (core.convert.num.FromI128I64.from core.num.I64.MAX).val = 9223372036854775807 := by
  rw [core.convert.num.FromI128I64.from_val_eq, gemm8_I64_MAX_val, gemm8_I64_max_lit]

/-- The extracted `i128::saturating_mul` returns the `Int`-clamped product. -/
theorem gemm8_saturating_mul_spec (x y : Std.I128) :
    core.num.I128.saturating_mul x y ⦃ r => r.val = gemm8_satMul x.val y.val ⦄ := by
  unfold core.num.I128.saturating_mul gemm8_satMul
  simp only [WP.spec_ok]
  have hb : -2 ^ (IScalarTy.I128.numBits - 1) ≤ max I128.min (min I128.max (x.val * y.val)) ∧
      max I128.min (min I128.max (x.val * y.val)) < 2 ^ (IScalarTy.I128.numBits - 1) := by
    rw [IScalarTy.I128_numBits_eq, I128.max_eq, I128.min_eq]
    omega
  rw [show (⟨BitVec.ofInt 128 (max I128.min (min I128.max (x.val * y.val)))⟩ : Std.I128) =
      Std.I128.ofIntCore (max I128.min (min I128.max (x.val * y.val))) hb from rfl]
  exact Std.I128.ofInt_val_eq hb

/-- The extracted `chart.F16_MAX_FIXED` returns `65504 · 2^32`. -/
theorem gemm8_F16_MAX_FIXED_spec :
    chart.F16_MAX_FIXED ⦃ r => r.val = gemm8_f16MaxFixed ⦄ := by
  have h : chart.F16_MAX_FIXED = ok (Std.I128.ofInt (65504 * 2 ^ 32)) := by
    unfold chart.F16_MAX_FIXED
    rfl
  rw [h]
  simp only [WP.spec_ok, gemm8_f16MaxFixed]
  apply Std.I128.ofInt_val_eq

/-- The `UScalar.hcast .I128` of a `Usize` below `2^127` preserves the value. -/
theorem gemm8_hcast_val (n_in : Std.Usize) (h : n_in.val < 2 ^ 127) :
    (UScalar.hcast IScalarTy.I128 n_in).val = (n_in.val : Int) := by
  have hs := UScalar.hcast_inBounds_spec (src_ty := UScalarTy.Usize) IScalarTy.I128 n_in ?_
  · obtain ⟨y, hy, hyv⟩ := WP.spec_imp_exists hs
    have hy' : UScalar.hcast IScalarTy.I128 n_in = y := by simpa [lift] using hy
    rw [hy']
    exact hyv
  · rw [IScalar.max_IScalarTy_I128_eq, I128.max_eq]
    omega

/-! ## Reads and the `I128` magnitude -/

/-- In-bounds `List.getD` of an `i64` list is its `getElem`, at the value level. -/
theorem gemm8_i64_getD_getElem (l : List Std.I64) (i : Nat) (h : i < l.length) :
    (l[i]'h).val = (l.getD i 0#i64).val := by
  rw [List.getD, List.getElem?_eq_getElem h]
  rfl

/-- The loop guard `b < Vec.len q.scales` is the stored length bound. -/
theorem gemm8_len_lt (q : chart.QuantAct) (b : Std.Usize) :
    (b < alloc.vec.Vec.len q.scales) ↔ b.val < q.scales.val.length := by
  rw [UScalar.lt_equiv, alloc.vec.Vec.len_val]

/-- The `I64 → I128` widening followed by `I128.abs` returns the `Nat`-absolute value of the lane;
the `i64`-ranged argument is `Gemm7`'s `gemm7_abs_spec`. -/
theorem gemm8_I128_abs_from_spec (x : Std.I64) :
    core.num.I128.abs (core.convert.num.FromI128I64.from x)
      ⦃ y => y.val = ((x.val.natAbs : Nat) : Int) ⦄ := by
  have hb := x.hBounds
  rw [IScalarTy.I64_numBits_eq] at hb
  norm_num at hb
  have h := gemm7_abs_spec (core.convert.num.FromI128I64.from x)
    (by rw [core.convert.num.FromI128I64.from_val_eq, I64.min_eq]; exact hb.1)
    (by rw [core.convert.num.FromI128I64.from_val_eq, I64.max_eq]; omega)
  rw [core.convert.num.FromI128I64.from_val_eq] at h
  exact h

/-! ## The `q8act_fast_bound_ok_loop` body and loop -/

/-- One iteration: on `b < |scales|` it reads `scales[b]`, folds `|scales[b]|` into the running
maximum and advances the counter; otherwise it returns the accumulator. -/
theorem gemm8_loop_body_spec (q : chart.QuantAct) (hsz : q.scales.val.length < Std.Usize.max)
    (s_max : Std.I128) (b : Std.Usize) (hinv : gemm8_smaxInv q (s_max, b)) :
    chart.q8act_fast_bound_ok_loop.body q s_max b ⦃ r =>
      match r with
      | ControlFlow.done y => y.val = gemm8_smaxRef q
      | ControlFlow.cont st' => gemm8_smaxInv q st' ∧
          q.scales.val.length - st'.2.val < q.scales.val.length - b.val ⦄ := by
  obtain ⟨hb, hsmax⟩ := hinv
  by_cases hlt : b.val < q.scales.val.length
  · have hlt' : b < alloc.vec.Vec.len q.scales := (gemm8_len_lt q b).mpr hlt
    unfold chart.q8act_fast_bound_ok_loop.body
    simp only [bind_tc_ok]
    rw [if_pos hlt']
    rw [alloc.vec.Vec.index_slice_index]
    obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec q.scales b hlt)
    rw [hi1]
    simp only [bind_tc_ok]
    have hi1val : i1.val = (q.scales.val.getD b.val 0#i64).val := by
      rw [hi1v]
      exact gemm8_i64_getD_getElem q.scales.val b.val hlt
    simp only [lift, bind_tc_ok]
    obtain ⟨a, ha, hav0⟩ := WP.spec_imp_exists (gemm8_I128_abs_from_spec i1)
    rw [ha]
    simp only [bind_tc_ok]
    have hav : a.val = ((q.scales.val.getD b.val 0#i64).val.natAbs : Int) := by
      rw [hav0, hi1val]
    by_cases hs : a > s_max
    · rw [if_pos hs]
      simp only [bind_tc_ok]
      obtain ⟨b1, hb1, hb1v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := b) (y := 1#usize) (by
          have hone : ((1#usize : Std.Usize) : Nat) = 1 := rfl
          rw [hone]; omega))
      rw [hb1]
      simp only [bind_tc_ok]
      simp only [WP.spec_ok]
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · rw [hb1v]
        have hone : ((1#usize : Std.Usize) : Nat) = 1 := rfl
        rw [hone]; omega
      · have hb1e : b1.val = b.val + 1 := by
          have hone : ((1#usize : Std.Usize) : Nat) = 1 := rfl
          rw [hb1v, hone]
        rw [hb1e, gemm8_smaxFrom_succ, ← hsmax, ← hav]
        exact (max_eq_right (le_of_lt ((IScalar.lt_equiv s_max a).mp hs))).symm
      · rw [hb1v]
        have hone : ((1#usize : Std.Usize) : Nat) = 1 := rfl
        rw [hone]; omega
    · rw [if_neg hs]
      simp only [bind_tc_ok]
      obtain ⟨b1, hb1, hb1v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := b) (y := 1#usize) (by
          have hone : ((1#usize : Std.Usize) : Nat) = 1 := rfl
          rw [hone]; omega))
      rw [hb1]
      simp only [bind_tc_ok]
      simp only [WP.spec_ok]
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · rw [hb1v]
        have hone : ((1#usize : Std.Usize) : Nat) = 1 := rfl
        rw [hone]; omega
      · have hb1e : b1.val = b.val + 1 := by
          have hone : ((1#usize : Std.Usize) : Nat) = 1 := rfl
          rw [hb1v, hone]
        rw [hb1e, gemm8_smaxFrom_succ, ← hsmax, ← hav]
        exact (max_eq_left (by
          have h := (IScalar.lt_equiv s_max a).not.mp hs
          have h2 : a.val ≤ s_max.val := by omega
          exact h2)).symm
      · rw [hb1v]
        have hone : ((1#usize : Std.Usize) : Nat) = 1 := rfl
        rw [hone]; omega
  · unfold chart.q8act_fast_bound_ok_loop.body
    simp only [bind_tc_ok]
    rw [if_neg (fun h => hlt ((gemm8_len_lt q b).mp h))]
    simp only [WP.spec_ok]
    have hbl : b.val = q.scales.val.length := le_antisymm hb (Nat.le_of_not_lt hlt)
    rw [gemm8_smaxRef, ← hbl, ← hsmax]

/-- From `(0, 0)` the walk returns the running maximum over the whole scale slice. -/
theorem word_q8act_fast_bound_loop_ok (q : chart.QuantAct)
    (hsz : q.scales.val.length < Std.Usize.max) :
    chart.q8act_fast_bound_ok_loop q 0#i128 0#usize ⦃ r =>
      r.val = gemm8_smaxRef q ⦄ := by
  unfold chart.q8act_fast_bound_ok_loop
  apply loop.spec_decr_nat
    (measure := fun s : Std.I128 × Std.Usize => q.scales.val.length - s.2.val)
    (inv := fun s => gemm8_smaxInv q s)
    (post := fun y : Std.I128 => y.val = gemm8_smaxRef q)
  · rintro ⟨s_max, b⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists
      (gemm8_loop_body_spec q hsz s_max b hinv)
    rw [hr]
    rcases r with y | s'
    · simpa only [WP.spec_ok] using hpost
    · simpa only [WP.spec_ok] using hpost
  · refine ⟨by simp, by simp [gemm8_smaxFrom]⟩

/-! ## The `q8act_fast_bound_ok` driver -/

/-- The driver returns `gemm8_fastBoundRef` of the running magnitude maximum when the two checked
`i128` products (`32 · top` and `(32 · top) · 127`) are in range on `narrow` and `n_in` fits `i128`. -/
theorem word_q8act_fast_bound_ok (n_in : Std.Usize) (top : Std.I128) (q : chart.QuantAct)
    (narrow : Bool)
    (hsz : q.scales.val.length < Std.Usize.max)
    (hn : n_in.val < 2 ^ 127)
    (h1 : narrow = true → I128.min ≤ 32 * top.val ∧ 32 * top.val ≤ I128.max)
    (h2 : narrow = true → I128.min ≤ 32 * top.val * 127 ∧ 32 * top.val * 127 ≤ I128.max) :
    chart.q8act_fast_bound_ok n_in top q narrow ⦃ r =>
      r = gemm8_fastBoundRef narrow n_in.val top.val (gemm8_smaxRef q) ⦄ := by
  unfold chart.q8act_fast_bound_ok
  obtain ⟨s0, hs0, hs0v⟩ := WP.spec_imp_exists (word_q8act_fast_bound_loop_ok q hsz)
  rw [hs0]
  simp only [bind_tc_ok]
  by_cases hnar : narrow = true
  · rw [if_pos hnar]
    have hm1 := h1 hnar
    have hm2 := h2 hnar
    obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (I128.mul_spec (x := 32#i128) (y := top)
      (by rw [show (32#i128).val = 32 from by norm_num]; exact hm1.1)
      (by rw [show (32#i128).val = 32 from by norm_num]; exact hm1.2))
    rw [hi]
    simp only [bind_tc_ok]
    obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (I128.mul_spec (x := i) (y := 127#i128)
      (by rw [hiv, show (127#i128).val = 127 from by norm_num]; exact hm2.1)
      (by rw [hiv, show (127#i128).val = 127 from by norm_num]; exact hm2.2))
    rw [hi1]
    simp only [bind_tc_ok]
    obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (gemm8_saturating_mul_spec s0 i1)
    rw [hi2]
    simp only [bind_tc_ok, lift]
    have hi1v' : i1.val = 32 * top.val * 127 := by
      rw [hi1v, hiv, show (127#i128).val = 127 from by norm_num,
        show (32#i128).val = 32 from by norm_num]
    by_cases hc : i2 > core.convert.num.FromI128I64.from core.num.I64.MAX
    · rw [if_pos hc]
      rw [WP.spec_ok]
      unfold gemm8_fastBoundRef
      rw [if_pos hnar]
      have hcv : (9223372036854775807 : Int) <
          gemm8_satMul (gemm8_smaxRef q) (32 * top.val * 127) := by
        have h := (IScalar.lt_equiv _ _).mp hc
        rw [gemm8_from_I64MAX_val, hi2v, hs0v, hi1v'] at h
        exact h
      rw [if_pos hcv]
    · rw [if_neg hc]
      unfold gemm8_fastBoundRef
      rw [if_pos hnar]
      have hcv : ¬ ((9223372036854775807 : Int) <
          gemm8_satMul (gemm8_smaxRef q) (32 * top.val * 127)) := by
        intro h
        apply hc
        exact (IScalar.lt_equiv _ _).mpr (by
          rw [gemm8_from_I64MAX_val, hi2v, hs0v, hi1v']
          exact h)
      rw [if_neg hcv]
      obtain ⟨i5, hi5, hi5v⟩ := WP.spec_imp_exists gemm8_F16_MAX_FIXED_spec
      rw [hi5]
      simp only [bind_tc_ok]
      obtain ⟨i6, hi6, hi6v⟩ := WP.spec_imp_exists
        (gemm8_saturating_mul_spec (UScalar.hcast IScalarTy.I128 n_in) i5)
      rw [hi6]
      simp only [bind_tc_ok]
      obtain ⟨i7, hi7, hi7v⟩ := WP.spec_imp_exists (gemm8_saturating_mul_spec i6 top)
      rw [hi7]
      simp only [bind_tc_ok]
      obtain ⟨i8, hi8, hi8v⟩ := WP.spec_imp_exists (gemm8_saturating_mul_spec i7 127#i128)
      rw [hi8]
      simp only [bind_tc_ok]
      obtain ⟨bd, hbd, hbdv⟩ := WP.spec_imp_exists (gemm8_saturating_mul_spec i8 s0)
      rw [hbd]
      simp only [bind_tc_ok]
      rw [WP.spec_ok]
      simp only [IScalar.lt_equiv]
      rw [hbdv, hi8v, hi7v, hi6v, hi5v, gemm8_hcast_val n_in hn, hs0v, gemm8_I128_MAX_val,
        gemm8_I128_max_lit, show (127#i128).val = 127 from by norm_num]
  · rw [if_neg hnar]
    simp only [lift, bind_tc_ok]
    obtain ⟨i5, hi5, hi5v⟩ := WP.spec_imp_exists gemm8_F16_MAX_FIXED_spec
    rw [hi5]
    simp only [bind_tc_ok]
    obtain ⟨i6, hi6, hi6v⟩ := WP.spec_imp_exists
      (gemm8_saturating_mul_spec (UScalar.hcast IScalarTy.I128 n_in) i5)
    rw [hi6]
    simp only [bind_tc_ok]
    obtain ⟨i7, hi7, hi7v⟩ := WP.spec_imp_exists (gemm8_saturating_mul_spec i6 top)
    rw [hi7]
    simp only [bind_tc_ok]
    obtain ⟨i8, hi8, hi8v⟩ := WP.spec_imp_exists (gemm8_saturating_mul_spec i7 127#i128)
    rw [hi8]
    simp only [bind_tc_ok]
    obtain ⟨bd, hbd, hbdv⟩ := WP.spec_imp_exists (gemm8_saturating_mul_spec i8 s0)
    rw [hbd]
    simp only [bind_tc_ok]
    rw [WP.spec_ok]
    unfold gemm8_fastBoundRef
    rw [if_neg hnar]
    simp only [IScalar.lt_equiv]
    rw [hbdv, hi8v, hi7v, hi6v, hi5v, gemm8_hcast_val n_in hn, hs0v, gemm8_I128_MAX_val,
      gemm8_I128_max_lit, show (127#i128).val = 127 from by norm_num]

/-! ## Refusals: the checked `narrow` products overflow -/

/-- A checked `i128` product outside `[i128::MIN, i128::MAX]` refuses with `integerOverflow`. -/
theorem gemm8_I128_mul_overflow (x y : Std.I128)
    (h : ¬ (I128.min ≤ x.val * y.val ∧ x.val * y.val ≤ I128.max)) :
    x * y = fail .integerOverflow := by
  change IScalar.mul x y = fail .integerOverflow
  unfold IScalar.mul IScalar.tryMk IScalar.tryMkOpt
  rw [dif_neg (by
    rw [IScalar.check_bounds_eq_inBounds]; intro hb
    rw [IScalar.inBounds] at hb
    rw [show (2 : Int) ^ (IScalarTy.I128.numBits - 1) = 170141183460469231731687303715884105728
        from by rw [IScalarTy.I128_numBits_eq]; norm_num] at hb
    apply h
    rw [I128.min_eq, I128.max_eq]
    refine ⟨hb.1, ?_⟩
    omega)]
  rfl

/-- On `narrow`, the driver refuses when the first checked product `32 · top` overflows `i128`. -/
theorem word_q8act_fast_bound_ok_refuses (n_in : Std.Usize) (top : Std.I128) (q : chart.QuantAct)
    (narrow : Bool)
    (hsz : q.scales.val.length < Std.Usize.max)
    (hnar : narrow = true)
    (h : ¬ (I128.min ≤ 32 * top.val ∧ 32 * top.val ≤ I128.max)) :
    chart.q8act_fast_bound_ok n_in top q narrow = fail .integerOverflow := by
  unfold chart.q8act_fast_bound_ok
  obtain ⟨s0, hs0, _⟩ := WP.spec_imp_exists (word_q8act_fast_bound_loop_ok q hsz)
  rw [hs0]
  simp only [bind_tc_ok]
  rw [if_pos hnar]
  rw [gemm8_I128_mul_overflow 32#i128 top (by
    rw [show (32#i128).val = 32 from by norm_num]
    exact h)]
  simp only [bind_tc_fail]

/-- On `narrow`, the driver refuses when the second checked product `(32 · top) · 127` overflows
`i128`. -/
theorem word_q8act_fast_bound_ok_refuses_scale (n_in : Std.Usize) (top : Std.I128)
    (q : chart.QuantAct)
    (narrow : Bool)
    (hsz : q.scales.val.length < Std.Usize.max)
    (hnar : narrow = true)
    (h1 : I128.min ≤ 32 * top.val ∧ 32 * top.val ≤ I128.max)
    (h2 : ¬ (I128.min ≤ 32 * top.val * 127 ∧ 32 * top.val * 127 ≤ I128.max)) :
    chart.q8act_fast_bound_ok n_in top q narrow = fail .integerOverflow := by
  unfold chart.q8act_fast_bound_ok
  obtain ⟨s0, hs0, _⟩ := WP.spec_imp_exists (word_q8act_fast_bound_loop_ok q hsz)
  rw [hs0]
  simp only [bind_tc_ok]
  rw [if_pos hnar]
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (I128.mul_spec (x := 32#i128) (y := top)
    (by rw [show (32#i128).val = 32 from by norm_num]; exact h1.1)
    (by rw [show (32#i128).val = 32 from by norm_num]; exact h1.2))
  rw [hi]
  simp only [bind_tc_ok]
  rw [gemm8_I128_mul_overflow i 127#i128 (by
    rw [hiv, show (32#i128).val = 32 from by norm_num,
      show (127#i128).val = 127 from by norm_num]
    exact h2)]
  simp only [bind_tc_fail]

/-! ## Non-vacuity -/

/-- A zero scale maximum makes the second test hold: `gemm8_fastBoundRef true 1 128 0 = true`. -/
example : gemm8_fastBoundRef true 1 128 0 = true := by
  unfold gemm8_fastBoundRef gemm8_satMul gemm8_f16MaxFixed
  norm_num [I128.min_eq, I128.max_eq]

/-- `narrow = false` skips the scale-side refusal: `gemm8_fastBoundRef false 1 128 0 = true`. -/
example : gemm8_fastBoundRef false 1 128 0 = true := by
  unfold gemm8_fastBoundRef gemm8_satMul gemm8_f16MaxFixed
  norm_num [I128.min_eq, I128.max_eq]

/-- Non-vacuity of the maximum reference: `max(|-3|, |5|) = 5` over a two-lane scale slice. -/
example : gemm8_smaxFrom ([(-3)#i64, 5#i64] : List Std.I64) 2 = 5 := by
  rw [gemm8_smaxFrom_succ, gemm8_smaxFrom_succ, gemm8_smaxFrom]
  norm_num

/-- `usize::MAX` is at least `1`. -/
theorem gemm8_usize_max_ge_one : (1 : Nat) ≤ Std.Usize.max := by
  rcases Usize.bounds_eq with h | h <;> rw [h] <;> norm_num [U32.max_eq, U64.max_eq]

/-- The empty activation: the smallest element of the `chart.QuantAct` domain. -/
def gemm8_emptyQuant : chart.QuantAct :=
  { scales := alloc.vec.Vec.from ([] : List Std.I64) (by simp)
    codes := alloc.vec.Vec.from ([] : List Std.I8) (by simp) }

/-- The empty activation's scale slice has length `0`, so `hsz` holds. -/
theorem gemm8_emptyQuant_hsz : gemm8_emptyQuant.scales.val.length < Std.Usize.max := by
  rw [gemm8_emptyQuant, alloc.vec.Vec.from_val]
  simp only [List.length_nil]
  exact Nat.lt_of_lt_of_le Nat.zero_lt_one gemm8_usize_max_ge_one

/-- The loop headline's hypothesis bundle is discharged at the empty activation and the walk returns
the reference: `0`. -/
example : chart.q8act_fast_bound_ok_loop gemm8_emptyQuant 0#i128 0#usize ⦃ r =>
    r.val = 0 ⦄ := by
  have h := word_q8act_fast_bound_loop_ok gemm8_emptyQuant gemm8_emptyQuant_hsz
  have h0 : gemm8_smaxRef gemm8_emptyQuant = 0 := by
    simp [gemm8_smaxRef, gemm8_smaxFrom, gemm8_emptyQuant, alloc.vec.Vec.from_val]
  rw [h0] at h
  exact h

/-- The driver headline's joint hypotheses hold at `n_in = 32`, `top = 128`, `narrow = true` on the
empty activation: `32 · top` and `(32 · top) · 127` are in `i128`, and `n_in` fits `i128`. -/
example : (32#usize).val < 2 ^ 127 ∧
    (-2 ^ 127 ≤ 32 * (128#i128).val ∧ 32 * (128#i128).val < 2 ^ 127) ∧
    (-2 ^ 127 ≤ 32 * (128#i128).val * 127 ∧ 32 * (128#i128).val * 127 < 2 ^ 127) := by
  norm_num

/-- The driver headline at that witness: the empty activation returns `true`. -/
example : chart.q8act_fast_bound_ok 32#usize 128#i128 gemm8_emptyQuant true ⦃ r =>
    r = true ⦄ := by
  have h := word_q8act_fast_bound_ok 32#usize 128#i128 gemm8_emptyQuant true
    gemm8_emptyQuant_hsz (by norm_num)
    (by intro _; norm_num [I128.min_eq, I128.max_eq])
    (by intro _; norm_num [I128.min_eq, I128.max_eq])
  have href : gemm8_fastBoundRef true (32#usize).val (128#i128).val
      (gemm8_smaxRef gemm8_emptyQuant) = true := by
    have h0 : gemm8_smaxRef gemm8_emptyQuant = 0 := by
      simp [gemm8_smaxRef, gemm8_smaxFrom, gemm8_emptyQuant, alloc.vec.Vec.from_val]
    rw [h0]
    simp only [show (32#usize).val = 32 from by norm_num,
      show (128#i128).val = 128 from by norm_num]
    exact (show gemm8_fastBoundRef true 32 128 0 = true from by
      unfold gemm8_fastBoundRef gemm8_satMul gemm8_f16MaxFixed
      norm_num [I128.min_eq, I128.max_eq])
  rw [href] at h
  exact h

/-- Non-vacuity of the first refusal: `32 · i128::MAX` does not fit `i128`. -/
example : ¬ (I128.min ≤ 32 * (core.num.I128.MAX).val ∧
    32 * (core.num.I128.MAX).val ≤ I128.max) := by
  rw [gemm8_I128_MAX_val, gemm8_I128_max_lit, I128.min_eq]
  norm_num

/-- Non-vacuity of the second refusal: `(2^60) · 127` does not fit `i128`… -/
example : ¬ (I128.min ≤ 32 * ((2 ^ 122)#i128).val * 127 ∧
    32 * ((2 ^ 122)#i128).val * 127 ≤ I128.max) := by
  rw [I128.min_eq, I128.max_eq]
  norm_num

end apeiron_word
