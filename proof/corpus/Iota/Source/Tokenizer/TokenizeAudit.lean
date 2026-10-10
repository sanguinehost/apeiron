import Iota.Audit.Empty
import Iota.Source.Tokenizer.Tokenize

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Iota.Source.Tokenizer.TokenizeAudit

/-- A zero-axiom audit fixture. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Source.Tokenizer.TokenizeAudit

#audit_empty Iota.Source.Tokenizer.sourceBytes_map_length : ∀ (source : Iota.Source.SourceBytes), (source.map UInt8.ofFin).length = source.length
#audit_empty Iota.Source.Tokenizer.encodeBytes_data_length : ∀ (source : Iota.Source.SourceBytes), (Iota.Source.encodeBytes source).data.toList.length = source.length
#audit_empty Iota.Source.Tokenizer.TokenizeDerivation : Iota.Source.SourceBytes → List Iota.Decode.Token → Nat → Prop
#audit_empty Iota.Source.Tokenizer.TokenizeDerivation.intro : type_of% @Iota.Source.Tokenizer.TokenizeDerivation.intro
#audit_empty Iota.Source.Tokenizer.TokenizeDerivation.sound : type_of% @Iota.Source.Tokenizer.TokenizeDerivation.sound
#audit_empty Iota.Source.Tokenizer.TokenizeAudit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Iota.Source.Tokenizer.sourceBytes_map_length
#print axioms Iota.Source.Tokenizer.encodeBytes_data_length
#print axioms Iota.Source.Tokenizer.TokenizeDerivation
#print axioms Iota.Source.Tokenizer.TokenizeDerivation.intro
#print axioms Iota.Source.Tokenizer.TokenizeDerivation.sound
