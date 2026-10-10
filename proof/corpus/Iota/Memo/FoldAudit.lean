import Iota.Audit.Empty
import Iota.Memo.Fold

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Iota.Memo.FoldAudit

/-- The runner replaces this axiom-free proof in its rejecting control. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Memo.FoldAudit

set_option linter.missingDocs false

#audit_empty Iota.Memo.FoldAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Memo.fold : type_of% @Iota.Memo.fold
#audit_empty Iota.Memo.canonical_fold_preserves_point : type_of% @Iota.Memo.canonical_fold_preserves_point
#audit_empty Iota.Memo.entry_fold_stream_pointwise : type_of% @Iota.Memo.entry_fold_stream_pointwise
#audit_empty Iota.Memo.canonical_fold_denotes_build_at : type_of% @Iota.Memo.canonical_fold_denotes_build_at
#audit_empty Iota.Memo.fold_success_rep_acc : type_of% @Iota.Memo.fold_success_rep_acc
#audit_empty Iota.Memo.empty_fold_denotes_build : type_of% @Iota.Memo.empty_fold_denotes_build
#audit_empty Iota.Memo.fold_success_denotes_build_at : type_of% @Iota.Memo.fold_success_denotes_build_at
#audit_empty Iota.Memo.fold_success_rep : type_of% @Iota.Memo.fold_success_rep
#audit_empty Iota.Memo.control_reversal_changes_build : type_of% @Iota.Memo.control_reversal_changes_build

#print axioms Iota.Memo.FoldAudit.nat_add_zero
#print axioms Iota.Memo.fold
#print axioms Iota.Memo.canonical_fold_preserves_point
#print axioms Iota.Memo.entry_fold_stream_pointwise
#print axioms Iota.Memo.canonical_fold_denotes_build_at
#print axioms Iota.Memo.fold_success_rep_acc
#print axioms Iota.Memo.empty_fold_denotes_build
#print axioms Iota.Memo.fold_success_denotes_build_at
#print axioms Iota.Memo.fold_success_rep
#print axioms Iota.Memo.control_reversal_changes_build

set_option warningAsError false
