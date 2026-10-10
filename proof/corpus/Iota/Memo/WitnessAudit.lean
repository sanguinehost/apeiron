import Iota.Audit.Empty
import Iota.Memo.Witness

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Iota.Memo.WitnessAudit

/-- The runner replaces this axiom-free proof in its rejecting control. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Memo.WitnessAudit

set_option linter.missingDocs false

#audit_empty Iota.Memo.WitnessAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Memo.Witness.sources : type_of% @Iota.Memo.Witness.sources
#audit_empty Iota.Memo.Witness.sourceRef : type_of% @Iota.Memo.Witness.sourceRef
#audit_empty Iota.Memo.Witness.states : type_of% @Iota.Memo.Witness.states
#audit_empty Iota.Memo.Witness.results : type_of% @Iota.Memo.Witness.results
#audit_empty Iota.Memo.Witness.stateHandle : type_of% @Iota.Memo.Witness.stateHandle
#audit_empty Iota.Memo.Witness.resultHandle : type_of% @Iota.Memo.Witness.resultHandle
#audit_empty Iota.Memo.Witness.key : type_of% @Iota.Memo.Witness.key
#audit_empty Iota.Memo.Witness.defaultValue : type_of% @Iota.Memo.Witness.defaultValue
#audit_empty Iota.Memo.Witness.zeroStore : type_of% @Iota.Memo.Witness.zeroStore
#audit_empty Iota.Memo.Witness.zero_constructor_read_refusal : type_of% @Iota.Memo.Witness.zero_constructor_read_refusal
#audit_empty Iota.Memo.Witness.success_of_transition : type_of% @Iota.Memo.Witness.success_of_transition

#print axioms Iota.Memo.WitnessAudit.nat_add_zero
#print axioms Iota.Memo.Witness.sources
#print axioms Iota.Memo.Witness.sourceRef
#print axioms Iota.Memo.Witness.states
#print axioms Iota.Memo.Witness.results
#print axioms Iota.Memo.Witness.stateHandle
#print axioms Iota.Memo.Witness.resultHandle
#print axioms Iota.Memo.Witness.key
#print axioms Iota.Memo.Witness.defaultValue
#print axioms Iota.Memo.Witness.zeroStore
#print axioms Iota.Memo.Witness.zero_constructor_read_refusal
#print axioms Iota.Memo.Witness.success_of_transition

set_option warningAsError false
