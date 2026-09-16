import Slang.Audit.Empty
import Slang.Memo.Fold

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Memo.FoldAudit

/-- The runner replaces this axiom-free proof in its rejecting control. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Memo.FoldAudit

set_option linter.missingDocs false

#audit_empty Slang.Memo.FoldAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Memo.fold : type_of% @Slang.Memo.fold
#audit_empty Slang.Memo.canonical_fold_preserves_point : type_of% @Slang.Memo.canonical_fold_preserves_point
#audit_empty Slang.Memo.entry_fold_stream_pointwise : type_of% @Slang.Memo.entry_fold_stream_pointwise
#audit_empty Slang.Memo.canonical_fold_denotes_build_at : type_of% @Slang.Memo.canonical_fold_denotes_build_at
#audit_empty Slang.Memo.fold_success_rep_acc : type_of% @Slang.Memo.fold_success_rep_acc
#audit_empty Slang.Memo.empty_fold_denotes_build : type_of% @Slang.Memo.empty_fold_denotes_build
#audit_empty Slang.Memo.fold_success_denotes_build_at : type_of% @Slang.Memo.fold_success_denotes_build_at
#audit_empty Slang.Memo.fold_success_rep : type_of% @Slang.Memo.fold_success_rep
#audit_empty Slang.Memo.control_reversal_changes_build : type_of% @Slang.Memo.control_reversal_changes_build

#print axioms Slang.Memo.FoldAudit.nat_add_zero
#print axioms Slang.Memo.fold
#print axioms Slang.Memo.canonical_fold_preserves_point
#print axioms Slang.Memo.entry_fold_stream_pointwise
#print axioms Slang.Memo.canonical_fold_denotes_build_at
#print axioms Slang.Memo.fold_success_rep_acc
#print axioms Slang.Memo.empty_fold_denotes_build
#print axioms Slang.Memo.fold_success_denotes_build_at
#print axioms Slang.Memo.fold_success_rep
#print axioms Slang.Memo.control_reversal_changes_build

set_option warningAsError false
