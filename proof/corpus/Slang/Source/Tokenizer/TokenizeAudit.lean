import Slang.Audit.Empty
import Slang.Source.Tokenizer.Tokenize

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Slang.Source.Tokenizer.TokenizeAudit

/-- A zero-axiom audit fixture. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Source.Tokenizer.TokenizeAudit

#audit_empty Slang.Source.Tokenizer.sourceBytes_map_length : ∀ (source : Slang.Source.SourceBytes), (source.map UInt8.ofFin).length = source.length
#audit_empty Slang.Source.Tokenizer.encodeBytes_data_length : ∀ (source : Slang.Source.SourceBytes), (Slang.Source.encodeBytes source).data.toList.length = source.length
#audit_empty Slang.Source.Tokenizer.TokenizeDerivation : Slang.Source.SourceBytes → List Slang.Decode.Token → Nat → Prop
#audit_empty Slang.Source.Tokenizer.TokenizeDerivation.intro : type_of% @Slang.Source.Tokenizer.TokenizeDerivation.intro
#audit_empty Slang.Source.Tokenizer.TokenizeDerivation.sound : type_of% @Slang.Source.Tokenizer.TokenizeDerivation.sound
#audit_empty Slang.Source.Tokenizer.TokenizeAudit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Slang.Source.Tokenizer.sourceBytes_map_length
#print axioms Slang.Source.Tokenizer.encodeBytes_data_length
#print axioms Slang.Source.Tokenizer.TokenizeDerivation
#print axioms Slang.Source.Tokenizer.TokenizeDerivation.intro
#print axioms Slang.Source.Tokenizer.TokenizeDerivation.sound
