import Iota.Audit.Empty
import Iota.Source.Parsing.Tokens

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Iota.Source.Parsing.TokensAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Source.Parsing.TokensAudit

#audit_empty Iota.Source.Parsing.DecodeTokensDerivation : ByteArray → List Iota.Decode.Token → Nat → Iota.Decode.CertifiedDecoded → Prop
#audit_empty Iota.Source.Parsing.DecodeTokensDerivation.mk : type_of% @Iota.Source.Parsing.DecodeTokensDerivation.mk
#audit_empty Iota.Source.Parsing.DecodeTokensDerivation.exactConsumption : type_of% @Iota.Source.Parsing.DecodeTokensDerivation.exactConsumption
#audit_empty Iota.Source.Parsing.DecodeTokensDerivation.delimiters : type_of% @Iota.Source.Parsing.DecodeTokensDerivation.delimiters
#audit_empty Iota.Source.Parsing.DecodeTokensDerivation.selected : type_of% @Iota.Source.Parsing.DecodeTokensDerivation.selected
#audit_empty Iota.Source.Parsing.DecodeTokensDerivation.constantCount : type_of% @Iota.Source.Parsing.DecodeTokensDerivation.constantCount
#audit_empty Iota.Source.Parsing.DecodeTokensDerivation.functionCount : type_of% @Iota.Source.Parsing.DecodeTokensDerivation.functionCount
#audit_empty Iota.Source.Parsing.DecodeTokensDerivation.constantNames : type_of% @Iota.Source.Parsing.DecodeTokensDerivation.constantNames
#audit_empty Iota.Source.Parsing.DecodeTokensDerivation.functionNames : type_of% @Iota.Source.Parsing.DecodeTokensDerivation.functionNames
#audit_empty Iota.Source.Parsing.DecodeTokensDerivation.spans : type_of% @Iota.Source.Parsing.DecodeTokensDerivation.spans
#audit_empty Iota.Source.Parsing.DecodeTokensDerivation.calls : type_of% @Iota.Source.Parsing.DecodeTokensDerivation.calls
#audit_empty Iota.Source.Parsing.DecodeTokensDerivation.names : type_of% @Iota.Source.Parsing.DecodeTokensDerivation.names
#audit_empty Iota.Source.Parsing.DecodeTokensDerivation.result : type_of% @Iota.Source.Parsing.DecodeTokensDerivation.result
#audit_empty Iota.Source.Parsing.DecodeTokensDerivation.sound : type_of% @Iota.Source.Parsing.DecodeTokensDerivation.sound
#audit_empty Iota.Source.Parsing.TokensAudit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Iota.Source.Parsing.DecodeTokensDerivation
#print axioms Iota.Source.Parsing.DecodeTokensDerivation.mk
#print axioms Iota.Source.Parsing.DecodeTokensDerivation.exactConsumption
#print axioms Iota.Source.Parsing.DecodeTokensDerivation.delimiters
#print axioms Iota.Source.Parsing.DecodeTokensDerivation.selected
#print axioms Iota.Source.Parsing.DecodeTokensDerivation.constantCount
#print axioms Iota.Source.Parsing.DecodeTokensDerivation.functionCount
#print axioms Iota.Source.Parsing.DecodeTokensDerivation.constantNames
#print axioms Iota.Source.Parsing.DecodeTokensDerivation.functionNames
#print axioms Iota.Source.Parsing.DecodeTokensDerivation.spans
#print axioms Iota.Source.Parsing.DecodeTokensDerivation.calls
#print axioms Iota.Source.Parsing.DecodeTokensDerivation.names
#print axioms Iota.Source.Parsing.DecodeTokensDerivation.result
#print axioms Iota.Source.Parsing.DecodeTokensDerivation.sound
#print axioms Iota.Source.Parsing.TokensAudit.nat_add_zero
#audit_empty Iota.Source.Parsing.DecodeTokensDerivation.assemble : type_of% @Iota.Source.Parsing.DecodeTokensDerivation.assemble
#print axioms Iota.Source.Parsing.DecodeTokensDerivation.assemble
