/-
# Iota zero-axiom enforcement

`#audit_empty <name> : <type>` is the rejecting Lean command for the Iota
source-refinement chain. For the named declaration it

* resolves the declaration in the elaborated environment and rejects a missing
  name,
* rejects an `axiom` target,
* elaborates the registered type and rejects a declaration whose type is not
  definitionally equal to it, and
* collects the transitive axiom dependencies with `Lean.collectAxioms` and
  rejects a nonempty set.

`Lean.collectAxioms` walks the declaration's value and type, so the audited set
is transitive: referenced instances, equation lemmas and generated auxiliary
declarations (for example the `_native` axiom produced by `native_decide`) are
included. The required cone here is the *empty* set, stricter than the
`{propext, Classical.choice, Quot.sound}` cone of the Rust bridge; this command
does not change `proof/corpus/AxiomAudit.lean` or the canonical corpus.

On success the command logs the machine-readable line
`EMPTY_AXIOMS_OK <fully.qualified.Name>`. `tools/iota/zero_axioms.py` parses
those lines as the coverage report and compares them with
`tools/iota/zero_axioms.json`.
-/
import Iota.Source.Bytes
import Lean

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Audit.Harness

/-- A positive fixture: `rfl` uses no axiom. The runner's mutation control
replaces this proof with a `Classical.choice` term, so this declaration is the
red/restored-green target. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

/-- A positive fixture whose proof term is a core lemma. -/
theorem nat_mul_one (n : Nat) : n * 1 = n := Nat.mul_one n

/-- A positive fixture proved by case analysis. -/
theorem bool_and_comm (a b : Bool) : (a && b) = (b && a) := by
  cases a <;> cases b <;> rfl

end Iota.Audit.Harness

open Lean Elab Command Term Meta

namespace Iota.Audit.Empty

/-- **The rejecting gate.** See the module comment. The command throws, so a
nonempty cone, a missing name, an axiom target or a wrong type is a build
failure rather than a line in a log. -/
elab "#audit_empty " id:ident " : " ty:term : command => do
  let env ← getEnv
  let declName := id.getId
  let some info := env.find? declName
    | throwError "zero-axiom audit: declaration not found: {declName}"
  if let .axiomInfo _ := info then
    throwError "zero-axiom audit: {declName} is an axiom, not a checked declaration"
  let expected ← liftTermElabM do
    let typeExpr ← elabType ty
    synthesizeSyntheticMVarsNoPostponing
    instantiateMVars typeExpr
  unless ← liftTermElabM (Meta.isDefEq expected info.type) do
    throwError "zero-axiom audit: {declName} does not have the registered type {ty}"
  let axioms ← liftTermElabM <| Lean.collectAxioms declName
  if !axioms.isEmpty then
    throwError "zero-axiom audit: {declName} depends on a nonempty axiom set {axioms.toList}"
  logInfo m!"EMPTY_AXIOMS_OK {declName}"

end Iota.Audit.Empty

-- The registered harness population, mirrored in tools/iota/zero_axioms.json.
-- Each line binds the elaborated declaration to its registered type; the
-- command's log lines are the coverage report the runner consumes.
#audit_empty Iota.Audit.Harness.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Audit.Harness.nat_mul_one : ∀ (n : Nat), n * 1 = n
#audit_empty Iota.Audit.Harness.bool_and_comm : ∀ (a b : Bool), (a && b) = (b && a)
#audit_empty Iota.Source.encodeBytes : Iota.Source.SourceBytes → ByteArray
#audit_empty Iota.Source.ByteRep.eq : ∀ {left right : Iota.Source.SourceBytes} {packed : ByteArray}, Iota.Source.ByteRep left packed → Iota.Source.ByteRep right packed → left = right
