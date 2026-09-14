import Slang.Machine
import Lean

open Lean Elab Command in
private def machineClosedAxioms : Lean.NameSet :=
  ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound

open Lean Elab Command in
elab "#machine_audit " id:ident : command => do
  let constants ← liftCoreM <| realizeGlobalConstWithInfos id
  for constant in constants do
    let axioms ← liftTermElabM <| Lean.collectAxioms constant
    let outside := axioms.filter (fun ax => !machineClosedAxioms.contains ax)
    if outside.isEmpty then
      logInfo m!"'{constant}' depends on axioms: {axioms.toList}"
    else
      throwError m!"'{constant}' depends on axioms outside the closed cone: {outside.toList}"

#machine_audit Slang.Word.signed_allOnes
#machine_audit Slang.Word.wrapping_add_boundary
#machine_audit Slang.Word.wrapping_mul_boundary
#machine_audit Slang.Word.signed_comparison_fixture
#machine_audit Slang.Word.bitwise_fixture
#machine_audit Slang.Word.tableSpread_multiplies_before_mask
#machine_audit Slang.Word.tableSpread_is_word_bounded
#machine_audit Slang.Machine.Memory.store_ownership
#machine_audit Slang.Machine.Memory.store_at
#machine_audit Slang.Machine.Memory.store_other
#machine_audit Slang.Machine.EvalExpr
#machine_audit Slang.Machine.EvalCall
#machine_audit Slang.Machine.resources_allocatedNodes
#machine_audit Slang.Machine.resources_externalStreamConsumption
#machine_audit Slang.Machine.lazy_inline_conditional_does_not_evaluate_untaken_branch
#machine_audit Slang.Machine.eager_bitOr_evaluates_right_operand
#machine_audit Slang.Machine.return_stops_execution
#machine_audit Slang.Machine.stack_exhaustion_is_not_a_word_return
