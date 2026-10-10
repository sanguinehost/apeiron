import Iota.Refinement.Context
import Iota.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Refinement.ContextAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Refinement.ContextAudit

#audit_empty Iota.Refinement.ContextAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Refinement.denoteExpr : Iota.Expr → Option Iota.Word
#audit_empty Iota.Refinement.constantEnvironment : Iota.Program → String → Option Iota.Word
#audit_empty Iota.Refinement.checkedContext : Iota.Program → Iota.Machine.Context
#audit_empty Iota.Refinement.tablePhiDecl : Iota.Span → Iota.ConstantDecl
#audit_empty Iota.Refinement.constantEnvironment_tablePhi : ∀ (program : Iota.Program) (span : Iota.Span), program.constants.find? (fun declaration => declaration.name == "TABLE_PHI") = some (Iota.Refinement.tablePhiDecl span) → Iota.Refinement.constantEnvironment program "TABLE_PHI" = some Iota.Word.tablePhi
#audit_empty Iota.Refinement.checkedContext_constants : ∀ (program : Iota.Program), (Iota.Refinement.checkedContext program).constants = Iota.Refinement.constantEnvironment program
#audit_empty Iota.Refinement.disagreeing_tablePhi_not_checkedContext : ∀ (program : Iota.Program) (context : Iota.Machine.Context), Iota.Refinement.constantEnvironment program "TABLE_PHI" = some Iota.Word.tablePhi → context.constants "TABLE_PHI" ≠ some Iota.Word.tablePhi → context ≠ Iota.Refinement.checkedContext program
#audit_empty Iota.Refinement.zeroTablePhiContext : Iota.Program → Iota.Machine.Context
#audit_empty Iota.Refinement.zeroTablePhiContext_not_checkedContext : ∀ (program : Iota.Program), Iota.Refinement.constantEnvironment program "TABLE_PHI" = some Iota.Word.tablePhi → Iota.Refinement.zeroTablePhiContext program ≠ Iota.Refinement.checkedContext program
