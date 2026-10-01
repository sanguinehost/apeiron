import NarrowSource
import Aeneas

/-
SCOPE: real pinned signed narrowing, local error bridge and inclusive-boundary witnesses.
REGISTER: source refinement; excludes shared Unit-valued conversion helpers, GPU and cost.
CORPUS: actual std conversion/error source; Aeneas IScalar.cast_inBounds_spec.
REFERENCE: /tmp/opencode/native-narrow-bridge-review.md, corrected Luna verdict.
GATE: python3 proof/native/narrow_gate.py --controls; 540s, 12GiB/no swap/one thread/lane.
GRADE: proved only on fresh kernel, shape rejection and both guard mutation/restoration.
-/

namespace NarrowRefinement
open Aeneas Aeneas.Std Result

/-- The local representation retains every constructor of the real error payload. -/
theorem error_payload_identity (kind : native_narrow.core.num.error.IntErrorKind) :
    (kind : native_narrow.core.num.error.TryFromIntError) = kind := rfl

/-- Every admitted signed word narrows with its exact integer value, including both endpoints. -/
theorem in_range (x : I128)
    (hmin : (-9223372036854775808 : Int) ≤ x.val)
    (hmax : x.val ≤ (9223372036854775807 : Int)) :
    ∃ y : I64, native_narrow.narrow x = ok (core.result.Result.Ok y) ∧ y.val = x.val := by
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
  unfold native_narrow.narrow native_narrow.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from
  rw [if_neg hlo, if_neg hhi, hy]
  simp only [bind_tc_ok]

/-- The actual lower refusal retains the signed error distinction. -/
theorem below_min (x : I128) (h : x.val < (-9223372036854775808 : Int)) :
    native_narrow.narrow x = ok (core.result.Result.Err
      native_narrow.core.num.error.IntErrorKind.NegOverflow) := by
  have hlo : x < (-9223372036854775808)#i128 :=
    (IScalar.lt_equiv x ((-9223372036854775808)#i128)).2 (by simpa using h)
  unfold native_narrow.narrow native_narrow.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from
  simp only [if_pos hlo]

/-- The actual upper refusal retains the signed error distinction. -/
theorem above_max (x : I128) (h : (9223372036854775807 : Int) < x.val) :
    native_narrow.narrow x = ok (core.result.Result.Err
      native_narrow.core.num.error.IntErrorKind.PosOverflow) := by
  have hlo : ¬ x < (-9223372036854775808)#i128 := by
    intro hc
    have := (IScalar.lt_equiv x ((-9223372036854775808)#i128)).1 hc
    norm_num at this
    omega
  have hhi : x > 9223372036854775807#i128 :=
    (IScalar.lt_equiv (9223372036854775807#i128) x).2 (by simpa using h)
  unfold native_narrow.narrow native_narrow.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from
  simp only [if_neg hlo, if_pos hhi]

/-- This joint input/output witness makes the inclusive lower boundary non-vacuous. -/
theorem lower_boundary :
    native_narrow.narrow (-9223372036854775808)#i128 =
      ok (core.result.Result.Ok (-9223372036854775808)#i64) := by
  norm_num [native_narrow.narrow,
    native_narrow.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from,
    lift, bind_tc_ok]
  rfl

/-- This joint input/output witness makes the inclusive upper boundary non-vacuous. -/
theorem upper_boundary :
    native_narrow.narrow 9223372036854775807#i128 =
      ok (core.result.Result.Ok 9223372036854775807#i64) := by
  norm_num [native_narrow.narrow,
    native_narrow.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from,
    lift, bind_tc_ok]
  rfl

theorem zero_witness : native_narrow.narrow 0#i128 = ok (core.result.Result.Ok 0#i64) := by
  norm_num [native_narrow.narrow,
    native_narrow.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from,
    lift, bind_tc_ok]
  rfl

theorem negative_refusal_witness : native_narrow.narrow (-9223372036854775809)#i128 =
    ok (core.result.Result.Err native_narrow.core.num.error.IntErrorKind.NegOverflow) :=
  below_min _ (by norm_num)

theorem positive_refusal_witness : native_narrow.narrow 9223372036854775808#i128 =
    ok (core.result.Result.Err native_narrow.core.num.error.IntErrorKind.PosOverflow) :=
  above_max _ (by norm_num)

end NarrowRefinement

open Lean Elab Command in
elab "#narrow_audit " id:ident : command => do
  let constants ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for constant in constants do
    let dependencies ← liftTermElabM <| Lean.collectAxioms constant
    let bad := dependencies.filter (fun value => !allowed.contains value)
    unless bad.isEmpty do
      throwError m!"{constant}: forbidden dependencies {bad.toList}"
    logInfo m!"{constant}: dependencies {dependencies.toList}"

#narrow_audit native_narrow.core.num.error.TryFromIntError
#narrow_audit native_narrow.core.num.error.IntErrorKind
#narrow_audit native_narrow.I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from
#narrow_audit native_narrow.narrow
#narrow_audit NarrowRefinement.error_payload_identity
#narrow_audit NarrowRefinement.in_range
#narrow_audit NarrowRefinement.below_min
#narrow_audit NarrowRefinement.above_max
#narrow_audit NarrowRefinement.lower_boundary
#narrow_audit NarrowRefinement.upper_boundary
#narrow_audit NarrowRefinement.zero_witness
#narrow_audit NarrowRefinement.negative_refusal_witness
#narrow_audit NarrowRefinement.positive_refusal_witness

run_cmd Lean.logInfo "NARROW_KERNEL_OK"
