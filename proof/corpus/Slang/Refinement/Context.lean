import Slang.Machine

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Refinement

open Slang Slang.Machine

private def denoteBinary (operator : BinOp) (leftValue rightValue : Option Word) : Option Word :=
  match leftValue, rightValue with
  | some leftWord, some rightWord =>
      match operator with
      | .add => some (Word.add leftWord rightWord)
      | .sub => some (Word.sub leftWord rightWord)
      | .mul => some (Word.mul leftWord rightWord)
      | _ => none
  | _, _ => none

private def denoteExprImpl : Expr → Option Word
  | .literal value => some (Word.ofNat value)
  | .binary operator left right =>
      denoteBinary operator (denoteExprImpl left) (denoteExprImpl right)
  | .name _ | .call _ _ | .memory _ | .conditional _ _ _ => none

/-- Evaluates the pure arithmetic fragment admitted for source constants. -/
@[implemented_by denoteExprImpl]
def denoteExpr (expression : Expr) : Option Word :=
  Expr.rec
    (motive_1 := fun _ => Option Word)
    (motive_2 := fun _ => Unit)
    (fun value => some (Word.ofNat value))
    (fun _ => none)
    (fun _ _ _ => none)
    (fun _ _ => none)
    (fun operator _ _ leftValue rightValue =>
      Option.rec none (fun leftWord =>
        Option.rec none (fun rightWord =>
          BinOp.rec
            (some (Word.add leftWord rightWord))
            (some (Word.sub leftWord rightWord))
            (some (Word.mul leftWord rightWord))
            none none none none none none none none none
            operator)
          rightValue)
        leftValue)
    (fun _ _ _ _ _ _ => none)
    Unit.unit
    (fun _ _ _ _ => Unit.unit)
    expression

/-- The first matching decoded declaration determines a constant value. -/
def constantEnvironment (program : Program) : String → Option Word := fun name =>
  (program.constants.find? fun declaration => declaration.name == name).bind fun declaration =>
    denoteExpr declaration.value

/-- A machine context whose constants and functions come from one decoded program. -/
def checkedContext (program : Program) : Context :=
  Context.ofProgram program (constantEnvironment program)

def tablePhiDecl (span : Span) : ConstantDecl :=
  { name := "TABLE_PHI", value := .literal 2654435769, span }

theorem constantEnvironment_tablePhi (program : Program) (span : Span)
    (hfind : program.constants.find? (fun declaration => declaration.name == "TABLE_PHI") =
      some (tablePhiDecl span)) :
    constantEnvironment program "TABLE_PHI" = some Word.tablePhi := by
  unfold constantEnvironment
  rw [hfind]
  rfl

theorem checkedContext_constants (program : Program) :
    (checkedContext program).constants = constantEnvironment program := rfl

theorem disagreeing_tablePhi_not_checkedContext (program : Program) (context : Context)
    (hderived : constantEnvironment program "TABLE_PHI" = some Word.tablePhi)
    (hdisagree : context.constants "TABLE_PHI" ≠ some Word.tablePhi) :
    context ≠ checkedContext program := by
  intro heq
  apply hdisagree
  calc
    context.constants "TABLE_PHI" = (checkedContext program).constants "TABLE_PHI" :=
      congrArg (fun candidate : Context => candidate.constants "TABLE_PHI") heq
    _ = constantEnvironment program "TABLE_PHI" := rfl
    _ = some Word.tablePhi := hderived

/-- A concrete independently supplied environment that disagrees at `TABLE_PHI`. -/
def zeroTablePhiContext (program : Program) : Context :=
  Context.ofProgram program fun name =>
    if name == "TABLE_PHI" then some (Word.ofNat 0) else constantEnvironment program name

theorem zeroTablePhiContext_not_checkedContext (program : Program)
    (hderived : constantEnvironment program "TABLE_PHI" = some Word.tablePhi) :
    zeroTablePhiContext program ≠ checkedContext program := by
  apply disagreeing_tablePhi_not_checkedContext program (zeroTablePhiContext program) hderived
  intro heq
  have hzero : Word.ofNat 0 ≠ Word.tablePhi := by
    intro hword
    exact Nat.noConfusion (congrArg Fin.val hword)
  apply hzero
  change some (Word.ofNat 0) = some Word.tablePhi at heq
  cases heq

end Slang.Refinement
