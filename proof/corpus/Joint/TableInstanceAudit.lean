import Joint.TableInstance
import Lean

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Lean Elab Command in
private def jointClosedAxioms : Lean.NameSet :=
  ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound

open Lean Elab Command in
elab "#joint_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !jointClosedAxioms.contains a)
    if bad.isEmpty then
      logInfo m!"'{c}' depends on axioms: {axioms.toList}"
    else
      throwError m!"'{c}' depends on axioms outside the closed cone: {bad.toList}"

#joint_audit Apeiron.Joint.TableInstance.replacementReadRep_iff_readRep
#joint_audit Apeiron.Joint.TableInstance.extractedReplacementMerge_spec
#joint_audit Apeiron.Joint.TableInstance.replacementMerge_spec
#joint_audit Apeiron.Joint.TableInstance.replacementReadRep_iff_scopedRep
#joint_audit Apeiron.Joint.TableInstance.replacement_new_readRep
#joint_audit Apeiron.Joint.TableInstance.replacement_get
#joint_audit Apeiron.Joint.TableInstance.replacement_insert_with
#joint_audit Apeiron.Joint.TableInstance.replacement_shardFull_refusal
#joint_audit Apeiron.Joint.TableInstance.replacement_buildInsertResult
#joint_audit Apeiron.Joint.TableInstance.replacement_build
#joint_audit Apeiron.Joint.TableInstance.replacement_build_get_acc
