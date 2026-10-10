-- The zero-axiom coverage report for the ported table decoder.
import Iota.Generated.SourceWitness.Decoder.Assembly
import Iota.Audit.Empty

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Iota.Generated.SourceWitness.Decoder.Audit

/-- The runner's mutation anchor: `rfl` uses no axiom; the negative control replaces this proof
with a `Classical.choice` term. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Generated.SourceWitness.Decoder.Audit

#audit_empty Iota.Generated.SourceWitness.Decoder.Audit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Generated.SourceWitness.Decoder.tableDerivation : type_of% @Iota.Generated.SourceWitness.Decoder.tableDerivation
#audit_empty Iota.Generated.SourceWitness.Decoder.table_decodeTokens : type_of% @Iota.Generated.SourceWitness.Decoder.table_decodeTokens

#print axioms Iota.Generated.SourceWitness.Decoder.tableDerivation
#print axioms Iota.Generated.SourceWitness.Decoder.table_decodeTokens
