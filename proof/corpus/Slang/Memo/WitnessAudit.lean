import Slang.Audit.Empty
import Slang.Memo.Witness

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Memo.WitnessAudit

/-- The runner replaces this axiom-free proof in its rejecting control. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Memo.WitnessAudit

set_option linter.missingDocs false

#audit_empty Slang.Memo.WitnessAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Memo.Witness.sources : type_of% @Slang.Memo.Witness.sources
#audit_empty Slang.Memo.Witness.sourceRef : type_of% @Slang.Memo.Witness.sourceRef
#audit_empty Slang.Memo.Witness.states : type_of% @Slang.Memo.Witness.states
#audit_empty Slang.Memo.Witness.results : type_of% @Slang.Memo.Witness.results
#audit_empty Slang.Memo.Witness.stateHandle : type_of% @Slang.Memo.Witness.stateHandle
#audit_empty Slang.Memo.Witness.resultHandle : type_of% @Slang.Memo.Witness.resultHandle
#audit_empty Slang.Memo.Witness.key : type_of% @Slang.Memo.Witness.key
#audit_empty Slang.Memo.Witness.defaultValue : type_of% @Slang.Memo.Witness.defaultValue
#audit_empty Slang.Memo.Witness.zeroStore : type_of% @Slang.Memo.Witness.zeroStore
#audit_empty Slang.Memo.Witness.zero_constructor_read_refusal : type_of% @Slang.Memo.Witness.zero_constructor_read_refusal
#audit_empty Slang.Memo.Witness.success_of_transition : type_of% @Slang.Memo.Witness.success_of_transition

#print axioms Slang.Memo.WitnessAudit.nat_add_zero
#print axioms Slang.Memo.Witness.sources
#print axioms Slang.Memo.Witness.sourceRef
#print axioms Slang.Memo.Witness.states
#print axioms Slang.Memo.Witness.results
#print axioms Slang.Memo.Witness.stateHandle
#print axioms Slang.Memo.Witness.resultHandle
#print axioms Slang.Memo.Witness.key
#print axioms Slang.Memo.Witness.defaultValue
#print axioms Slang.Memo.Witness.zeroStore
#print axioms Slang.Memo.Witness.zero_constructor_read_refusal
#print axioms Slang.Memo.Witness.success_of_transition

set_option warningAsError false
