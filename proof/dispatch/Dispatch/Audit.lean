import Dispatch.Refinement
import Lean

open Lean Elab Command in
elab "#audit_dispatch " n:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos n
  for c in cs do
    let axSet ← liftTermElabM <| Lean.collectAxioms c
    let allowed : NameSet := ({} : NameSet)
      |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
    let bad := axSet.filter (fun a => !allowed.contains a)
    if !bad.isEmpty then throwError m!"DISPATCH_AXIOM_REJECT {c}: {bad.toList}"
    logInfo m!"DISPATCH_KERNEL_OK {c}: {axSet.toList}"

#audit_dispatch Dispatch.Refinement.digest_same
#audit_dispatch Dispatch.Refinement.start_refines
#audit_dispatch Dispatch.Refinement.advance_refines
#audit_dispatch Dispatch.Refinement.finish_refines
#audit_dispatch Dispatch.Refinement.complete_refines
#audit_dispatch Dispatch.Refinement.permit_iff
#audit_dispatch Dispatch.Refinement.witness_success
#audit_dispatch Dispatch.Refinement.witness_refusal
#audit_dispatch Dispatch.Refinement.joint_reachable
