import Lean

/- A typed implementation prerequisite. The registry fixes every parameter, including the
   complete outcome/refusal type and the reviewed source-connected resource model. This
   structure proves improvement only in that model, never a hardware speed prediction. -/
namespace ProofDispatch

structure Admission {Input Output : Type}
    (before after : Input → Output)
    (reachable admissible : Input → Prop)
    (beforeCost afterCost : Input → Nat) where
  witness : Input
  joint : reachable witness ∧ admissible witness
  preserves : ∀ x, reachable x → admissible x → after x = before x
  improves : ∀ x, reachable x → admissible x → afterCost x < beforeCost x

open Lean Elab Command in
elab "#audit_admission " n:ident : command => do
  let names ← liftCoreM <| realizeGlobalConstWithInfos n
  for name in names do
    let axioms ← liftTermElabM <| Lean.collectAxioms name
    let allowed : NameSet := ({} : NameSet)
      |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
    let bad := axioms.filter (fun a => !allowed.contains a)
    if !bad.isEmpty then
      throwError m!"ADMISSION_AXIOM_REJECT {name}: {bad.toList}"
    logInfo m!"ADMISSION_KERNEL_OK {name}: {axioms.toList}"

end ProofDispatch
