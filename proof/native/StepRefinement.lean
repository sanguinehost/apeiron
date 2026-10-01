import ExactDotSource
import Aeneas.Std.Core.Iter
import Aeneas.Std.Scalar.Notations

/-
SCOPE: private Step wiring, full real exact_dot extraction and exhausted-tail narrowing.
REGISTER: model wiring and real narrowing/tail refinement; whole-loop arithmetic and cost excluded.
CORPUS: Aeneas scalar Step definitions read; no canonical Rust Step theorem.
REFERENCE: Aeneas/Std/Core/Iter.lean, existing UScalarStep overflowing functions.
GATE: proof/native/step_gate.py --actual-exact-dot --controls; 540 s whole invocation.
GRADE: pending fresh full-source kernel check and controls; Rust Result::ok source link open.
-/

set_option autoImplicit false

namespace StepRefinement
open Aeneas Aeneas.Std Result

private def usizeMax : Usize :=
  Usize.ofNatCore Usize.max (by simp [Usize.max, Usize.numBits])

theorem usize_forward_wiring (start amount : Usize) :
    core.iter.range.Step.forward_overflowing
      (core.iter.range.StepUsize) start amount =
    @core.iter.range.UScalarStep.forward_overflowing .Usize start amount := rfl

theorem usize_backward_wiring (start amount : Usize) :
    core.iter.range.Step.backward_overflowing
      (core.iter.range.StepUsize) start amount =
    @core.iter.range.UScalarStep.backward_overflowing .Usize start amount := rfl

theorem usize_forward_overflow_boundary :
    core.iter.range.Step.forward_overflowing core.iter.range.StepUsize
      usizeMax 1#usize =
      ok (0#usize, true) := by
  cases System.Platform.numBits_eq <;>
    simp_all [core.iter.range.StepUsize, core.iter.range.UScalarStep,
      core.iter.range.UScalarStep.forward_overflowing, usizeMax,
      Usize.ofNatCore, UScalar.ofNatCore, UScalar.val, UScalar.max,
      Usize.max, Usize.numBits, UScalarTy.numBits]
  all_goals apply Fin.ext
  all_goals simp_all [Fin.val_add]

theorem usize_backward_overflow_boundary :
    core.iter.range.Step.backward_overflowing core.iter.range.StepUsize
      (0#usize) 1#usize =
      ok (usizeMax, true) := by
  cases System.Platform.numBits_eq <;>
    simp_all [core.iter.range.StepUsize, core.iter.range.UScalarStep,
      core.iter.range.UScalarStep.backward_overflowing, usizeMax,
      Usize.ofNatCore, UScalar.ofNatCore, UScalar.val,
      Usize.max, Usize.numBits, UScalarTy.numBits]
  all_goals first | rfl | (apply BitVec.eq_of_toNat_eq; simp_all [BitVec.toNat_sub])

/-- The local representation retains every constructor of the real error payload. -/
theorem error_payload_identity (kind : apeiron.core.num.error.IntErrorKind) :
    (kind : apeiron.core.num.error.TryFromIntError) = kind := rfl

/-- Every admitted signed word narrows with its exact integer value, including both endpoints. -/
theorem in_range (x : I128)
    (hmin : (-9223372036854775808 : Int) ≤ x.val)
    (hmax : x.val ≤ (9223372036854775807 : Int)) :
    ∃ y : I64, apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from x = ok (core.result.Result.Ok y) ∧ y.val = x.val := by
  have hlo : ¬ x < (-9223372036854775808)#i128 := by
    intro hc
    have := (IScalar.lt_equiv x ((-9223372036854775808)#i128)).1 hc
    norm_num at this
    omega
  have hhi : ¬ x > 9223372036854775807#i128 := by
    intro hc
    have := (IScalar.lt_equiv (9223372036854775807#i128) x).1 hc
    norm_num at this
    omega
  obtain ⟨y, hy, hyv⟩ := WP.spec_imp_exists (IScalar.cast_inBounds_spec .I64 x (by
    simpa only [IScalar.min_IScalarTy_I64_eq, IScalar.max_IScalarTy_I64_eq,
      I64.min_eq, I64.max_eq] using And.intro hmin hmax))
  refine ⟨y, ?_, hyv⟩
  unfold apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from
  rw [if_neg hlo, if_neg hhi, hy]
  simp only [bind_tc_ok]

/-- The actual lower refusal retains the signed error distinction. -/
theorem below_min (x : I128) (h : x.val < (-9223372036854775808 : Int)) :
    apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from x = ok (core.result.Result.Err
      apeiron.core.num.error.IntErrorKind.NegOverflow) := by
  have hlo : x < (-9223372036854775808)#i128 :=
    (IScalar.lt_equiv x ((-9223372036854775808)#i128)).2 (by simpa using h)
  unfold apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from
  simp only [if_pos hlo]

/-- The actual upper refusal retains the signed error distinction. -/
theorem above_max (x : I128) (h : (9223372036854775807 : Int) < x.val) :
    apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from x = ok (core.result.Result.Err
      apeiron.core.num.error.IntErrorKind.PosOverflow) := by
  have hlo : ¬ x < (-9223372036854775808)#i128 := by
    intro hc
    have := (IScalar.lt_equiv x ((-9223372036854775808)#i128)).1 hc
    norm_num at this
    omega
  have hhi : x > 9223372036854775807#i128 :=
    (IScalar.lt_equiv (9223372036854775807#i128) x).2 (by simpa using h)
  unfold apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from
  simp only [if_neg hlo, if_pos hhi]

/-- This joint input/output witness makes the inclusive lower boundary non-vacuous. -/
theorem lower_boundary :
    apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from (-9223372036854775808)#i128 =
      ok (core.result.Result.Ok (-9223372036854775808)#i64) := by
  norm_num [apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from,
    lift, bind_tc_ok]
  rfl

/-- This joint input/output witness makes the inclusive upper boundary non-vacuous. -/
theorem upper_boundary :
    apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from 9223372036854775807#i128 =
      ok (core.result.Result.Ok 9223372036854775807#i64) := by
  norm_num [apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from,
    lift, bind_tc_ok]
  rfl

theorem zero_witness : apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from 0#i128 = ok (core.result.Result.Ok 0#i64) := by
  norm_num [apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from,
    lift, bind_tc_ok]
  rfl

theorem negative_refusal_witness : apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from (-9223372036854775809)#i128 =
    ok (core.result.Result.Err apeiron.core.num.error.IntErrorKind.NegOverflow) :=
  below_min _ (by norm_num)

theorem positive_refusal_witness : apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from 9223372036854775808#i128 =
    ok (core.result.Result.Err apeiron.core.num.error.IntErrorKind.PosOverflow) :=
  above_max _ (by norm_num)


/-- The premise observes actual checked rounding; its panic domain is not erased. -/
theorem exhausted_tail (row : Slice U8) (q : apeiron.engine.chart.QuantAct)
    (iter : core.ops.range.Range Usize) (acc x : I128)
    (hn : core.iter.range.IteratorRange.next core.iter.range.StepUsize iter = ok (none, iter))
    (hr : apeiron.engine.value.round_shift_32 acc = ok x) :
    apeiron.engine.kern.gemm.exact_dot_loop0.body row q iter acc = (do
      let r ← apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from x
      let o ← core.result.Result.ok r
      ok (ControlFlow.done o)) := by
  unfold apeiron.engine.kern.gemm.exact_dot_loop0.body
  rw [hn]
  simp only [bind_tc_ok]
  rw [hr]
  simp only [bind_tc_ok]
  rfl

theorem empty_next : core.iter.range.IteratorRange.next core.iter.range.StepUsize
    (⟨0#usize, 0#usize⟩ : core.ops.range.Range Usize) = ok (none, ⟨0#usize, 0#usize⟩) := by
  simp [core.iter.range.IteratorRange.next, core.iter.range.StepUsize,
    core.iter.range.UScalarStep, core.cmp.PartialOrdUsize,
    core.cmp.impls.PartialOrdUsize.lt, liftFun2, bind_tc_ok]

theorem tail_in_range (row : Slice U8) (q : apeiron.engine.chart.QuantAct)
    (iter : core.ops.range.Range Usize) (acc x : I128)
    (hn : core.iter.range.IteratorRange.next core.iter.range.StepUsize iter = ok (none, iter))
    (hr : apeiron.engine.value.round_shift_32 acc = ok x)
    (hmin : (-9223372036854775808 : Int) ≤ x.val)
    (hmax : x.val ≤ (9223372036854775807 : Int)) :
    ∃ y : I64, apeiron.engine.kern.gemm.exact_dot_loop0.body row q
      iter acc = ok (ControlFlow.done (some y)) ∧ y.val = x.val := by
  obtain ⟨y, hy, hv⟩ := in_range x hmin hmax
  refine ⟨y, ?_, hv⟩
  rw [exhausted_tail row q iter acc x hn hr, hy]
  simp only [bind_tc_ok, core.result.Result.ok]

theorem tail_below_min (row : Slice U8) (q : apeiron.engine.chart.QuantAct)
    (iter : core.ops.range.Range Usize) (acc x : I128)
    (hn : core.iter.range.IteratorRange.next core.iter.range.StepUsize iter = ok (none, iter))
    (hr : apeiron.engine.value.round_shift_32 acc = ok x)
    (h : x.val < (-9223372036854775808 : Int)) :
    apeiron.engine.kern.gemm.exact_dot_loop0.body row q iter acc =
      ok (ControlFlow.done none) := by
  rw [exhausted_tail row q iter acc x hn hr, below_min x h]
  simp only [bind_tc_ok, core.result.Result.ok]

theorem tail_above_max (row : Slice U8) (q : apeiron.engine.chart.QuantAct)
    (iter : core.ops.range.Range Usize) (acc x : I128)
    (hn : core.iter.range.IteratorRange.next core.iter.range.StepUsize iter = ok (none, iter))
    (hr : apeiron.engine.value.round_shift_32 acc = ok x)
    (h : (9223372036854775807 : Int) < x.val) :
    apeiron.engine.kern.gemm.exact_dot_loop0.body row q iter acc =
      ok (ControlFlow.done none) := by
  rw [exhausted_tail row q iter acc x hn hr, above_max x h]
  simp only [bind_tc_ok, core.result.Result.ok]


/-- A reachable subset for the joint tail witnesses, proved through scalar specifications. -/
theorem round_scaled (acc x : I128) (ha : acc.val = x.val * 4294967296)
    (hlo : (-18446744073709551616 : Int) ≤ x.val)
    (hhi : x.val ≤ (18446744073709551616 : Int)) :
    apeiron.engine.value.round_shift_32 acc = ok x := by
  obtain ⟨s, hs, hsv⟩ := WP.spec_imp_exists
    (IScalar.ShiftLeft_IScalar_spec (1#i128) (31#i32) (by norm_num) (by norm_num))
  have hsv : s.val = 2147483648 := by
    have h := hsv.1
    norm_num [IScalar.toNat, Int.shiftLeft_eq, Int.bmod] at h
    exact h
  obtain ⟨z, hz, hzv⟩ := WP.spec_imp_exists (IScalar.add_spec (x := acc) (y := s)
    (by simp only [ha, hsv, IScalar.min_IScalarTy_I128_eq, I128.min_eq]; omega)
    (by simp only [ha, hsv, IScalar.max_IScalarTy_I128_eq, I128.max_eq]; omega))
  obtain ⟨v, hv, hvv⟩ := WP.spec_imp_exists
    (IScalar.ShiftRight_IScalar_spec z (32#i32) (by norm_num) (by norm_num))
  have heq : v = x := by
    apply IScalar.eq_of_val_eq
    have h := hvv.1
    simp only [hzv, ha, hsv] at h
    norm_num [IScalar.toNat, Int.shiftRight_eq_div_pow] at h
    change v.val = (x.val * 4294967296 + 2147483648) / 4294967296 at h
    omega
  unfold apeiron.engine.value.round_shift_32
  rw [hs]
  simp only [bind_tc_ok]
  rw [hz]
  simp only [bind_tc_ok]
  rw [hv, heq]

theorem tail_lower_boundary (row : Slice U8) (q : apeiron.engine.chart.QuantAct) :
    apeiron.engine.kern.gemm.exact_dot_loop0.body row q ⟨0#usize, 0#usize⟩
      (-39614081257132168796771975168)#i128 = ok (ControlFlow.done (some (-9223372036854775808)#i64)) := by
  have hr : apeiron.engine.value.round_shift_32 (-39614081257132168796771975168)#i128 = ok (-9223372036854775808)#i128 := by
    exact round_scaled _ _ (by norm_num) (by norm_num) (by norm_num)
  rw [exhausted_tail row q _ _ _ empty_next hr]
  norm_num [apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from,
    core.result.Result.ok, lift, bind_tc_ok]
  all_goals rfl

theorem tail_upper_boundary (row : Slice U8) (q : apeiron.engine.chart.QuantAct) :
    apeiron.engine.kern.gemm.exact_dot_loop0.body row q ⟨0#usize, 0#usize⟩
      (39614081257132168792477007872)#i128 = ok (ControlFlow.done (some 9223372036854775807#i64)) := by
  have hr : apeiron.engine.value.round_shift_32 (39614081257132168792477007872)#i128 = ok (9223372036854775807)#i128 := by
    exact round_scaled _ _ (by norm_num) (by norm_num) (by norm_num)
  rw [exhausted_tail row q _ _ _ empty_next hr]
  norm_num [apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from,
    core.result.Result.ok, lift, bind_tc_ok]
  all_goals rfl

theorem tail_zero_witness (row : Slice U8) (q : apeiron.engine.chart.QuantAct) :
    apeiron.engine.kern.gemm.exact_dot_loop0.body row q ⟨0#usize, 0#usize⟩
      (0)#i128 = ok (ControlFlow.done (some 0#i64)) := by
  have hr : apeiron.engine.value.round_shift_32 (0)#i128 = ok (0)#i128 := by
    exact round_scaled _ _ (by norm_num) (by norm_num) (by norm_num)
  rw [exhausted_tail row q _ _ _ empty_next hr]
  norm_num [apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from,
    core.result.Result.ok, lift, bind_tc_ok]
  all_goals rfl

theorem tail_negative_refusal (row : Slice U8) (q : apeiron.engine.chart.QuantAct) :
    apeiron.engine.kern.gemm.exact_dot_loop0.body row q ⟨0#usize, 0#usize⟩
      (-39614081257132168801066942464)#i128 = ok (ControlFlow.done (none)) := by
  have hr : apeiron.engine.value.round_shift_32 (-39614081257132168801066942464)#i128 = ok (-9223372036854775809)#i128 := by
    exact round_scaled _ _ (by norm_num) (by norm_num) (by norm_num)
  rw [exhausted_tail row q _ _ _ empty_next hr]
  norm_num [apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from,
    core.result.Result.ok, lift, bind_tc_ok]

theorem tail_positive_refusal (row : Slice U8) (q : apeiron.engine.chart.QuantAct) :
    apeiron.engine.kern.gemm.exact_dot_loop0.body row q ⟨0#usize, 0#usize⟩
      (39614081257132168796771975168)#i128 = ok (ControlFlow.done (none)) := by
  have hr : apeiron.engine.value.round_shift_32 (39614081257132168796771975168)#i128 = ok (9223372036854775808)#i128 := by
    exact round_scaled _ _ (by norm_num) (by norm_num) (by norm_num)
  rw [exhausted_tail row q _ _ _ empty_next hr]
  norm_num [apeiron.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from,
    core.result.Result.ok, lift, bind_tc_ok]


/-- Concrete model row/activation values inhabit the joint success/refusal population.
    This model witness does not refine the Rust quant_act constructor. -/
theorem joint_population :
    ∃ (row : Slice U8) (q : apeiron.engine.chart.QuantAct),
      apeiron.engine.kern.gemm.exact_dot_loop0.body row q ⟨0#usize, 0#usize⟩ 0#i128 =
        ok (ControlFlow.done (some 0#i64)) ∧
      apeiron.engine.kern.gemm.exact_dot_loop0.body row q ⟨0#usize, 0#usize⟩
        (-39614081257132168801066942464)#i128 = ok (ControlFlow.done none) ∧
      apeiron.engine.kern.gemm.exact_dot_loop0.body row q ⟨0#usize, 0#usize⟩
        39614081257132168796771975168#i128 = ok (ControlFlow.done none) := by
  let row : Slice U8 := (alloc.vec.Vec.new U8).slice
  let q : apeiron.engine.chart.QuantAct := ⟨alloc.vec.Vec.new I64, alloc.vec.Vec.new I8⟩
  exact ⟨row, q, tail_zero_witness row q, tail_negative_refusal row q, tail_positive_refusal row q⟩

end StepRefinement

open Lean Elab Command in
elab "#step_audit " id:ident : command => do
  let constants ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for constant in constants do
    let dependencies ← liftTermElabM <| Lean.collectAxioms constant
    let bad := dependencies.filter (fun value => !allowed.contains value)
    unless bad.isEmpty do
      throwError m!"{constant}: forbidden dependencies {bad.toList}"
    logInfo m!"{constant}: dependencies {dependencies.toList}"

#step_audit StepRefinement.usizeMax
#step_audit StepRefinement.usize_forward_wiring
#step_audit StepRefinement.usize_backward_wiring
#step_audit StepRefinement.usize_forward_overflow_boundary
#step_audit StepRefinement.usize_backward_overflow_boundary
#step_audit StepRefinement.error_payload_identity
#step_audit StepRefinement.in_range
#step_audit StepRefinement.below_min
#step_audit StepRefinement.above_max
#step_audit StepRefinement.lower_boundary
#step_audit StepRefinement.upper_boundary
#step_audit StepRefinement.zero_witness
#step_audit StepRefinement.negative_refusal_witness
#step_audit StepRefinement.positive_refusal_witness
#step_audit StepRefinement.exhausted_tail
#step_audit StepRefinement.empty_next
#step_audit StepRefinement.tail_in_range
#step_audit StepRefinement.tail_below_min
#step_audit StepRefinement.tail_above_max
#step_audit StepRefinement.round_scaled
#step_audit StepRefinement.joint_population
#step_audit StepRefinement.tail_lower_boundary
#step_audit StepRefinement.tail_upper_boundary
#step_audit StepRefinement.tail_zero_witness
#step_audit StepRefinement.tail_negative_refusal
#step_audit StepRefinement.tail_positive_refusal
#step_audit Aeneas.Std.core.iter.range.Step
#step_audit Aeneas.Std.core.iter.range.Step.mk
#step_audit Aeneas.Std.core.iter.range.Step.forward_overflowing
#step_audit Aeneas.Std.core.iter.range.Step.backward_overflowing
#step_audit Aeneas.Std.core.iter.range.UScalarStep
#step_audit Aeneas.Std.core.iter.range.IScalarStep
#step_audit Aeneas.Std.core.iter.range.StepUsize
#step_audit Aeneas.Std.core.iter.range.UScalarStep.forward_overflowing
#step_audit Aeneas.Std.core.iter.range.UScalarStep.backward_overflowing
#step_audit Aeneas.Std.core.iter.range.IScalarStep.forward_overflowing
#step_audit Aeneas.Std.core.iter.range.IScalarStep.backward_overflowing
#step_audit Aeneas.Std.core.result.Result
#step_audit Aeneas.Std.core.result.Result.Ok
#step_audit Aeneas.Std.core.result.Result.Err
#step_audit Aeneas.Std.core.result.Result.ok
run_cmd Lean.logInfo "STEP_MODEL_KERNEL_OK"
