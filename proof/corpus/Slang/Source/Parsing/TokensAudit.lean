import Slang.Audit.Empty
import Slang.Source.Parsing.Tokens

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Slang.Source.Parsing.TokensAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Source.Parsing.TokensAudit

#audit_empty Slang.Source.Parsing.DecodeTokensDerivation : ByteArray → List Slang.Decode.Token → Nat → Slang.Decode.CertifiedDecoded → Prop
#audit_empty Slang.Source.Parsing.DecodeTokensDerivation.mk : type_of% @Slang.Source.Parsing.DecodeTokensDerivation.mk
#audit_empty Slang.Source.Parsing.DecodeTokensDerivation.exactConsumption : type_of% @Slang.Source.Parsing.DecodeTokensDerivation.exactConsumption
#audit_empty Slang.Source.Parsing.DecodeTokensDerivation.delimiters : type_of% @Slang.Source.Parsing.DecodeTokensDerivation.delimiters
#audit_empty Slang.Source.Parsing.DecodeTokensDerivation.selected : type_of% @Slang.Source.Parsing.DecodeTokensDerivation.selected
#audit_empty Slang.Source.Parsing.DecodeTokensDerivation.constantCount : type_of% @Slang.Source.Parsing.DecodeTokensDerivation.constantCount
#audit_empty Slang.Source.Parsing.DecodeTokensDerivation.functionCount : type_of% @Slang.Source.Parsing.DecodeTokensDerivation.functionCount
#audit_empty Slang.Source.Parsing.DecodeTokensDerivation.constantNames : type_of% @Slang.Source.Parsing.DecodeTokensDerivation.constantNames
#audit_empty Slang.Source.Parsing.DecodeTokensDerivation.functionNames : type_of% @Slang.Source.Parsing.DecodeTokensDerivation.functionNames
#audit_empty Slang.Source.Parsing.DecodeTokensDerivation.spans : type_of% @Slang.Source.Parsing.DecodeTokensDerivation.spans
#audit_empty Slang.Source.Parsing.DecodeTokensDerivation.calls : type_of% @Slang.Source.Parsing.DecodeTokensDerivation.calls
#audit_empty Slang.Source.Parsing.DecodeTokensDerivation.names : type_of% @Slang.Source.Parsing.DecodeTokensDerivation.names
#audit_empty Slang.Source.Parsing.DecodeTokensDerivation.result : type_of% @Slang.Source.Parsing.DecodeTokensDerivation.result
#audit_empty Slang.Source.Parsing.DecodeTokensDerivation.sound : type_of% @Slang.Source.Parsing.DecodeTokensDerivation.sound
#audit_empty Slang.Source.Parsing.TokensAudit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Slang.Source.Parsing.DecodeTokensDerivation
#print axioms Slang.Source.Parsing.DecodeTokensDerivation.mk
#print axioms Slang.Source.Parsing.DecodeTokensDerivation.exactConsumption
#print axioms Slang.Source.Parsing.DecodeTokensDerivation.delimiters
#print axioms Slang.Source.Parsing.DecodeTokensDerivation.selected
#print axioms Slang.Source.Parsing.DecodeTokensDerivation.constantCount
#print axioms Slang.Source.Parsing.DecodeTokensDerivation.functionCount
#print axioms Slang.Source.Parsing.DecodeTokensDerivation.constantNames
#print axioms Slang.Source.Parsing.DecodeTokensDerivation.functionNames
#print axioms Slang.Source.Parsing.DecodeTokensDerivation.spans
#print axioms Slang.Source.Parsing.DecodeTokensDerivation.calls
#print axioms Slang.Source.Parsing.DecodeTokensDerivation.names
#print axioms Slang.Source.Parsing.DecodeTokensDerivation.result
#print axioms Slang.Source.Parsing.DecodeTokensDerivation.sound
#print axioms Slang.Source.Parsing.TokensAudit.nat_add_zero
