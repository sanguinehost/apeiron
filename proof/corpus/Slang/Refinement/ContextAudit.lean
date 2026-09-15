import Slang.Refinement.Context
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Refinement.ContextAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Refinement.ContextAudit

#audit_empty Slang.Refinement.ContextAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Refinement.denoteExpr : Slang.Expr → Option Slang.Word
#audit_empty Slang.Refinement.constantEnvironment : Slang.Program → String → Option Slang.Word
#audit_empty Slang.Refinement.checkedContext : Slang.Program → Slang.Machine.Context
#audit_empty Slang.Refinement.tablePhiDecl : Slang.Span → Slang.ConstantDecl
#audit_empty Slang.Refinement.constantEnvironment_tablePhi : ∀ (program : Slang.Program) (span : Slang.Span), program.constants.find? (fun declaration => declaration.name == "TABLE_PHI") = some (Slang.Refinement.tablePhiDecl span) → Slang.Refinement.constantEnvironment program "TABLE_PHI" = some Slang.Word.tablePhi
#audit_empty Slang.Refinement.checkedContext_constants : ∀ (program : Slang.Program), (Slang.Refinement.checkedContext program).constants = Slang.Refinement.constantEnvironment program
#audit_empty Slang.Refinement.disagreeing_tablePhi_not_checkedContext : ∀ (program : Slang.Program) (context : Slang.Machine.Context), Slang.Refinement.constantEnvironment program "TABLE_PHI" = some Slang.Word.tablePhi → context.constants "TABLE_PHI" ≠ some Slang.Word.tablePhi → context ≠ Slang.Refinement.checkedContext program
#audit_empty Slang.Refinement.zeroTablePhiContext : Slang.Program → Slang.Machine.Context
#audit_empty Slang.Refinement.zeroTablePhiContext_not_checkedContext : ∀ (program : Slang.Program), Slang.Refinement.constantEnvironment program "TABLE_PHI" = some Slang.Word.tablePhi → Slang.Refinement.zeroTablePhiContext program ≠ Slang.Refinement.checkedContext program
