import Lean

/-!
# Audit: the fail-closed axiom cone

@name Audit Closed Cone
@grade proved

The one `#audit` command the audit modules run. It collects each named declaration's transitive
axiom dependencies with `Lean.collectAxioms`, the same mechanism `#print axioms` uses, and calls
`throwError` on anything outside `{propext, Classical.choice, Quot.sound}`, so a `sorryAx`
anywhere in the cone is a build failure rather than a line in a log nobody reads. `#print axioms`
alone cannot do this: it only prints, so a `sorryAx` in the list would compile clean.

`#audit` speaks about declarations. It cannot see a runtime check such as a `#guard`, which adds
no declaration; those fail the build on their own and are not axiom cones.
-/

open Lean Elab Command in
/-- The three axioms every proof in this corpus may depend on. -/
def Apeiron.closedAxioms : Lean.NameSet :=
  ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound

open Lean Elab Command in
/-- **The gate.** `#audit foo` collects `foo`'s transitive axiom dependencies with
    `Lean.collectAxioms`, the same mechanism `#print axioms` uses, and calls `throwError` if any
    of them falls outside `Apeiron.closedAxioms`. A `sorry` in `foo`'s proof, or in anything
    `foo` depends on, surfaces as `Lean.sorryAx`, which is not in the closed set, so the build
    fails at this line rather than printing a warning nobody watches. Prints the cone on
    success, so the log carries the same information `#print axioms` did. -/
elab "#audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !Apeiron.closedAxioms.contains a)
    if bad.isEmpty then
      logInfo m!"'{c}' depends on axioms: {axioms.toList}"
    else
      throwError m!"'{c}' depends on axioms outside the closed cone " ++
        m!"(propext, Classical.choice, Quot.sound): {bad.toList}"
