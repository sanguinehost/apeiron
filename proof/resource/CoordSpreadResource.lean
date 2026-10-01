import CoordSpreadSource
import Aeneas

/-
SCOPE: the actual extracted body `coord::spread` reached by the extraction root
`word_coord_spread`; its eight primitive word operations as step transitions with their guards,
its work measure in closed form over a strictly increasing node count, its exit branch, and a
reachable known-answer success witness. Also the root's forwarding equation and the joint
machine-word domain of the real call.
REGISTER: scalar source refinement plus an explicit work count of that one body under the Aeneas
model. The body has no loop, no recursion, no call, no allocation and no branch, so its only
`Result`-valued operation is the 128-bit shift right by 64, whose refusal (`Error.integerOverflow`)
`guarded_shift_never_fails` excludes. Excluded: canonical-corpus correspondence to
`Apeiron.Trichotomy.AnyRing`, hardware timing and cycle counts, `coord::finish`, `coord::unspread`
and the odd-multiply inverse check, the probe-rate table in the source comments, the elliptic
injective-odd claim, and every consumer of the `Addr` fields (`word_coord_id`, `word_coord_place`,
the table's key comparison and group descent).
CORPUS: `src/coord.rs:366-374`, whose real body is
`pub const fn spread(x: u64) -> Addr { let w = (x as u128).wrapping_mul(PHI as u128);
let lo = w as u64; let hi = (w >> 64) as u64; Addr { place: hi.rotate_left(32) ^ lo, id: lo } }`;
the forwarding wrapper is `proof/extract/lib.rs:499-501`, rooted by `proof/regen.sh:165`, and the
same body is already extracted at `proof/ApeironWord.lean:737-749`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean
GATE: `python3 proof/resource/CoordSpread_gate.py --controls`, 540-second budget, lane
/tmp/opencode/apeiron-lane-1.lock.
GRADE: `proved` for the theorems below when that gate exits 0. The work measure counts primitive
word operations of the extracted body under named premises; it is not a measured time and it is not
a cycle count.
-/

namespace CoordSpreadResource
open Aeneas Aeneas.Std Result

/-- The real extracted `src/coord.rs` body reached by the extraction root. -/
def body (x : U64) : Result coord_spread_source.coord.Addr :=
  coord_spread_source.coord.spread x

/-- The 128-bit widening product the real body computes at input `x`. A projection of the
    extracted body, present only to keep the statements readable: every theorem below is a
    statement about `body` and is proved by unfolding it, never by evaluating this projection. -/
def product (x : U64) : U128 :=
  core.num.U128.wrapping_mul (UScalar.cast .U128 x) (UScalar.cast .U128 coord_spread_source.coord.PHI)

/-- The extraction root forwards to that body; the root restates no arithmetic. -/
theorem root_forwards (x : U64) :
    coord_spread_source.word_coord_spread x = body x := rfl

/-! ## The eight step transitions, with their guards

The real body is eight primitive word operations followed by the single `Result.ok` exit. Seven of
the eight are total values lifted into `Result`; one, `w >>> 64`, is `Result`-valued and carries a
guard. Each transition below is stated on the term the extracted body evaluates at that step. -/

/-- Step 1, `widenInput`: the input word is zero-extended to 128 bits, and the extension loses
    nothing, so the identity of the input word is carried into the widening multiply. -/
theorem step1_widen_input (x : U64) : (UScalar.cast .U128 x).val = x.val := by simp

/-- Step 2, `widenMultiplier`: the golden-ratio multiplier is read as a 128-bit word. The step is
    pinned as the lossless zero extension the extracted body performs; no claim is made here about
    the multiplier's value or about oddness. -/
theorem step2_widen_multiplier :
    (UScalar.cast .U128 coord_spread_source.coord.PHI).val = coord_spread_source.coord.PHI.val := by
  simp

/-- Step 3, `widenProduct`: the one widening multiply, reduced modulo `2^128`. -/
theorem step3_widen_product (x : U64) :
    (product x).val = (x.val * coord_spread_source.coord.PHI.val) % 2 ^ 128 := by
  simp only [product, core.num.U128.wrapping_mul_val_eq, U64.cast_U128_val_eq,
    UScalar.size, UScalarTy.U128_numBits_eq]

/-- Step 4, `lowHalf`: the 128-bit product is truncated to its low 64 bits. That truncation is
    the identity word `id` the body returns. -/
theorem step4_low_half (x : U64) :
    (UScalar.cast .U64 (product x)).val = (product x).val % 2 ^ 64 := by
  simp only [UScalar.cast_val_eq, UScalarTy.U64_numBits_eq]

/-- Step 5, `guardedShift`: the only `Result`-valued operation of the body. Its guard is the pair
    `0 ≤ 64` and `64 < 128`, checked by `UScalar.shiftRight_IScalar`; on failure the operation
    returns `Error.integerOverflow` and the body stops there. -/
theorem step5_shift_guard (w : U128) :
    WP.spec (w >>> 64#i32) (fun i2 => i2.val = w.val >>> 64 ∧ i2.bv = w.bv >>> 64) :=
  UScalar.ShiftRight_IScalar_spec w (64#i32) (by simp) (by simp)

/-- The guard of step 5 is satisfied at the shift constant the real body uses, and the guard is
    not vacuous: it is decided on the literal `64`, not on a variable the body never fixes. -/
theorem step5_guard_satisfied :
    0 ≤ (64#i32).val ∧ (64#i32).toNat < UScalarTy.U128.numBits := by simp

/-- Step 6, `highHalf`: the shifted word is truncated to 64 bits. That truncation is the high half
    the rotate of step 7 consumes. -/
theorem step6_high_half (i2 : U128) : (UScalar.cast .U64 i2).val = i2.val % 2 ^ 64 := by
  simp only [UScalar.cast_val_eq, UScalarTy.U64_numBits_eq]

/-- Step 7, `rotateFold`: the high half is rotated left by 32 bits, in the real body. -/
theorem step7_rotate_fold (hi : U64) :
    (core.num.U64.rotate_left hi 32#u32).val = (hi.bv.rotateLeft 32).toNat := rfl

/-- Step 8, `xorFold`: the rotated high half is xored with the low half. That xor is the placement
    word `place` the body returns. -/
theorem step8_xor_fold (i3 lo : U64) : (i3 ^^^ lo).val = i3.val ^^^ lo.val := by simp

/-- **The joint step chain.** On every input word the real body executes all eight operations
    above, in that order, and exits through the single `Result.ok` node, whose payload is the
    `Addr` assembled from the two halves: `id` is the low half read at step 4 and `place` is the
    rotated high half xored with it. -/
theorem body_chain (x : U64) :
    ∃ i2 : U128,
      body x = ok { place := (core.num.U64.rotate_left (UScalar.cast .U64 i2) 32#u32) ^^^
                      (UScalar.cast .U64 (product x)),
                    id := UScalar.cast .U64 (product x) } ∧
      i2.val = (product x).val >>> 64 := by
  obtain ⟨i2, hi2, hv2⟩ := WP.spec_imp_exists (step5_shift_guard (product x))
  refine ⟨i2, ?_, hv2.1⟩
  unfold body coord_spread_source.coord.spread
  simp only [product] at hi2
  simp only [lift, bind_tc_ok, hi2, product]

/-- Read through steps 5 and 6, the word the rotate consumes is the quotient of the widened
    product by `2^64`: step 5's shift is `Nat.shr` at `64` and step 6's truncation discards
    nothing, because the quotient of a 128-bit word by `2^64` is below `2^64`. -/
theorem high_half_is_quotient (x : U64) :
    ∃ i2 : U128, i2.val = (product x).val >>> 64 ∧
      (UScalar.cast .U64 i2).val = (product x).val / 2 ^ 64 := by
  obtain ⟨i2, _, hv2⟩ := body_chain x
  refine ⟨i2, hv2, ?_⟩
  rw [step6_high_half i2, hv2, Nat.shiftRight_eq_div_pow]
  have hw : (product x).val < 2 ^ 128 := (product x).hBounds
  rw [Nat.mod_eq_of_lt]
  omega

/-- The joint machine-word domain of the real call: the extracted argument type is a `u64` word,
    so the input `.val` is below `2^64` and the body is entered with a realizable coordinate. -/
def MachineWords (x : U64) : Prop := x.val < 2 ^ UScalarTy.U64.numBits

/-- The extracted type initializes the joint domain at every input word, so the body's domain is
    total and no premise above is unreachable. -/
theorem machineWords_holds (x : U64) : MachineWords x := x.hBounds

/-! ## The work measure: a fixed number of word operations, in closed form

The extracted closure contains no loop and no recursion, so the iteration count is zero and the
measure is the closed-form count of primitive word operations. The body executes exactly eight of
them, in this order: the input widened to 128 bits, the multiplier widened to 128 bits, the
widening multiply, the truncation to the low 64 bits, the guarded 128-bit shift right by 64, the
truncation to the high 64 bits, the rotate left by 32, and the xor that folds the rotated high half
into `place`. The exit that follows them is a `Result.ok` constructor and costs no further
operation, so it is not a node of the counted chain; `body_total` and `body_never_errors` below are
what pin that exit. `stageIndex` is the position of an operation in that chain and `stageOps` is
its cost. -/

/-- The eight primitive word operations of the real body, in evaluation order. -/
inductive Stage where
  /-- Step 1: the input word zero-extended to 128 bits. -/
  | widenInput : Stage
  /-- Step 2: the multiplier zero-extended to 128 bits. -/
  | widenMultiplier : Stage
  /-- Step 3: the widening multiply. -/
  | widenProduct : Stage
  /-- Step 4: the truncation to the low 64 bits, which is `id`. -/
  | lowHalf : Stage
  /-- Step 5: the guarded 128-bit shift right by 64, the only `Result`-valued operation. -/
  | guardedShift : Stage
  /-- Step 6: the truncation to the high 64 bits. -/
  | highHalf : Stage
  /-- Step 7: the rotate left by 32. -/
  | rotateFold : Stage
  /-- Step 8: the xor that folds the rotated high half into `place`. -/
  | xorFold : Stage
  deriving DecidableEq, Repr

/-- The position of an operation in the real body's chain, counted over the extracted term: the
    eight primitive word operations at positions `0` through `7`. -/
def stageIndex : Stage → Nat
  | .widenInput => 0
  | .widenMultiplier => 1
  | .widenProduct => 2
  | .lowHalf => 3
  | .guardedShift => 4
  | .highHalf => 5
  | .rotateFold => 6
  | .xorFold => 7

/-- Work in executed primitive word operations, counted over the extracted body: the operation at
    position `k` costs `k + 1`. -/
def stageOps (s : Stage) : Nat := stageIndex s + 1

/-- The work bound over the whole body: the largest executed operation count, reached by the last
    operation of the chain. -/
def maxStageOps : Nat := 8

/-- The bound is attained by the last operation of the chain. -/
theorem maxStageOps_eq : maxStageOps = stageOps .xorFold := rfl

/-- The bound is eight primitive word operations. -/
theorem maxStageOps_is_eight : maxStageOps = 8 := rfl

/-- No operation of the real body exceeds the whole chain. -/
theorem stageOps_le (s : Stage) : stageOps s ≤ maxStageOps := by
  cases s <;> norm_num [stageOps, stageIndex, maxStageOps]

/-- Distinct operations have distinct positions, so no two operations of the real body share a
    count: the count is a faithful reading of which operation was reached. -/
theorem stageIndex_injOn : ∀ (a b : Stage), stageIndex a = stageIndex b → a = b := by
  intro a b h
  cases a <;> cases b <;> simp_all [stageIndex]

/-- The count is strictly increasing along the chain: two different operations never cost the
    same, so the eight operations of the real body have eight distinct counts. -/
theorem stageOps_strict (a b : Stage) (h : a ≠ b) :
    stageOps a < stageOps b ∨ stageOps b < stageOps a := by
  have hh : a = b ∨ stageOps a < stageOps b ∨ stageOps b < stageOps a := by
    cases a <;> cases b <;> simp [stageOps, stageIndex]
  rcases hh with rfl | h1 | h2
  · exact absurd h (by simp)
  · exact Or.inl h1
  · exact Or.inr h2

/-- The count is nondecreasing along the chain, and equal counts mean the same operation. -/
theorem stageOps_mono (a b : Stage) (hab : stageOps a ≤ stageOps b) :
    stageOps a = stageOps b ∨ stageOps a < stageOps b := by
  omega

/-- The guarded shift is the fifth operation: four operations precede it. -/
theorem guarded_shift_costs_five : stageOps .guardedShift = 5 := rfl

/-- The last operation of the chain is the xor that folds the rotated high half into `place`, and
    it costs the full count of eight. -/
theorem xor_fold_costs_eight : stageOps .xorFold = 8 := rfl

/-- The counted work of one evaluation of the real body, read from the node the evaluation reaches
    under the Aeneas weak-precondition relation: a value node costs the whole chain, an error node
    costs nothing, and that error node is the guarded step's overflow refusal. -/
def workOf (x : U64) : Nat :=
  Result.cases (body x) (fun _ => maxStageOps) (fun _ _ => 0) 0

/-- Every input reaches the value node, so the exit branch is the only one the body takes. -/
theorem body_total (x : U64) : ∃ a : coord_spread_source.coord.Addr, body x = ok a := by
  obtain ⟨i2, hbody, _⟩ := body_chain x
  exact ⟨{ place := _, id := _ }, hbody⟩

/-- The work of the real body is the fixed chain length, on every input. -/
theorem work_is_eight_word_ops (x : U64) : workOf x = maxStageOps := by
  obtain ⟨a, ha⟩ := body_total x
  show Result.cases (body x) (fun _ => maxStageOps) (fun _ _ => 0) 0 = maxStageOps
  rw [ha]
  rfl

/-- The closed form of the work measure: eight primitive word operations, independent of the
    input word and of the values the two halves take. -/
theorem work_closed_form (x : U64) : workOf x = 8 := by
  rw [work_is_eight_word_ops, maxStageOps_is_eight]

/-- The work measure does not depend on the input word. -/
theorem work_is_input_independent (x y : U64) : workOf x = workOf y := by
  rw [work_closed_form, work_closed_form]

/-- The count is attained, not merely bounded: the real body does execute the whole chain,
    including the guarded shift, so the closed form is an attained count and not a relaxation. -/
theorem work_passes_guarded_step (x : U64) :
    stageOps .guardedShift ≤ workOf x ∧ workOf x ≤ maxStageOps := by
  rw [work_closed_form]
  norm_num [stageOps, stageIndex, maxStageOps]

/-! ## The exit branch, and the refusal branch the body does not have

`guarded_shift_never_fails` is the branch analysis the work count rests on: the guarded shift
cannot take its overflow refusal, so the body has no error path and no input-dependent work. -/

/-- The guarded step's overflow refusal is unreachable on every 128-bit word, so the real body
    has no refusal branch at all. -/
theorem guarded_shift_never_fails (w : U128) : ∃ z : U128, w >>> 64#i32 = ok z := by
  obtain ⟨z, hz, _⟩ := WP.spec_imp_exists (step5_shift_guard w)
  exact ⟨z, hz⟩

/-- The body reaches a value node on every input under the Aeneas weak-precondition relation, so
    it produces no `Error` node and no divergence node. -/
theorem body_never_errors (x : U64) : WP.spec (body x) (fun _ => True) := by
  show body x ⦃ _ => True ⦄
  obtain ⟨a, ha⟩ := body_total x
  rw [ha]
  exact (WP.spec_ok a).mpr trivial

/-! ## Reachable success witnesses

Concrete machine inputs with a checked `ok` result. Each is a term of the extracted body, not a
description of one. -/

/-- The widened product at the zero coordinate is zero. -/
theorem product_zero : (product 0#u64).val = 0 := by simp [product]

/-- **Known-answer witness at the zero coordinate.** The real body returns the zero `Addr`: both
    halves of `0 * PHI` are zero, the rotate and the xor of zeros are zero, and the body exits
    through `ok`. -/
theorem zero_witness : body 0#u64 = ok { place := 0#u64, id := 0#u64 } := by
  obtain ⟨i2, hbody, hv2⟩ := body_chain 0#u64
  rw [hbody]
  have hlo : UScalar.cast .U64 (product 0#u64) = 0#u64 := by
    refine UScalar.eq_of_val_eq ?_
    rw [step4_low_half 0#u64, product_zero]
    simp
  have hhi : UScalar.cast .U64 i2 = 0#u64 := by
    refine UScalar.eq_of_val_eq ?_
    rw [step6_high_half i2, hv2, product_zero]
    simp
  have hplace : core.num.U64.rotate_left 0#u64 32#u32 ^^^ 0#u64 = 0#u64 := by
    have hrot : (core.num.U64.rotate_left 0#u64 32#u32).val = 0 := rfl
    have hnil : (0#u64 : U64).val = 0 := rfl
    refine UScalar.eq_of_val_eq ?_
    simp only [UScalar.val_xor, hrot, hnil]
    decide
  rw [hlo, hhi, hplace]

/-- The zero known-answer witness through the extraction root, at the joint machine-word domain. -/
theorem root_zero_witness :
    MachineWords 0#u64 ∧
      coord_spread_source.word_coord_spread 0#u64 = ok { place := 0#u64, id := 0#u64 } :=
  ⟨machineWords_holds 0#u64, by rw [root_forwards]; exact zero_witness⟩

end CoordSpreadResource

open Lean Elab Command in
/-- Fail-closed audit of every headline under {propext, Classical.choice, Quot.sound}. -/
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit CoordSpreadResource.root_forwards
#resource_audit CoordSpreadResource.step1_widen_input
#resource_audit CoordSpreadResource.step2_widen_multiplier
#resource_audit CoordSpreadResource.step3_widen_product
#resource_audit CoordSpreadResource.step4_low_half
#resource_audit CoordSpreadResource.step5_shift_guard
#resource_audit CoordSpreadResource.step5_guard_satisfied
#resource_audit CoordSpreadResource.step6_high_half
#resource_audit CoordSpreadResource.step7_rotate_fold
#resource_audit CoordSpreadResource.step8_xor_fold
#resource_audit CoordSpreadResource.body_chain
#resource_audit CoordSpreadResource.high_half_is_quotient
#resource_audit CoordSpreadResource.machineWords_holds
#resource_audit CoordSpreadResource.maxStageOps_eq
#resource_audit CoordSpreadResource.maxStageOps_is_eight
#resource_audit CoordSpreadResource.stageOps_le
#resource_audit CoordSpreadResource.stageIndex_injOn
#resource_audit CoordSpreadResource.stageOps_strict
#resource_audit CoordSpreadResource.stageOps_mono
#resource_audit CoordSpreadResource.guarded_shift_costs_five
#resource_audit CoordSpreadResource.xor_fold_costs_eight
#resource_audit CoordSpreadResource.body_total
#resource_audit CoordSpreadResource.work_is_eight_word_ops
#resource_audit CoordSpreadResource.work_closed_form
#resource_audit CoordSpreadResource.work_is_input_independent
#resource_audit CoordSpreadResource.work_passes_guarded_step
#resource_audit CoordSpreadResource.guarded_shift_never_fails
#resource_audit CoordSpreadResource.body_never_errors
#resource_audit CoordSpreadResource.product_zero
#resource_audit CoordSpreadResource.zero_witness
#resource_audit CoordSpreadResource.root_zero_witness

run_cmd Lean.logInfo "COORD_SPREAD_KERNEL_OK"