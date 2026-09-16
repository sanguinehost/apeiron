import Slang.Audit.Empty
import Slang.Source.Tokenizer.Utf8

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Slang.Source.Tokenizer.Utf8Audit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Source.Tokenizer.Utf8Audit

#audit_empty Slang.Source.Tokenizer.Utf8Head : List UInt8 → Char → List UInt8 → Prop
#audit_empty Slang.Source.Tokenizer.Utf8Head.one : type_of% @Slang.Source.Tokenizer.Utf8Head.one
#audit_empty Slang.Source.Tokenizer.Utf8Head.two : type_of% @Slang.Source.Tokenizer.Utf8Head.two
#audit_empty Slang.Source.Tokenizer.Utf8Head.three : type_of% @Slang.Source.Tokenizer.Utf8Head.three
#audit_empty Slang.Source.Tokenizer.Utf8Head.four : type_of% @Slang.Source.Tokenizer.Utf8Head.four
#audit_empty Slang.Source.Tokenizer.Utf8Head.sound : type_of% @Slang.Source.Tokenizer.Utf8Head.sound
#audit_empty Slang.Source.Tokenizer.Utf8Scan : Nat → List UInt8 → List Char → Prop
#audit_empty Slang.Source.Tokenizer.Utf8Scan.done : type_of% @Slang.Source.Tokenizer.Utf8Scan.done
#audit_empty Slang.Source.Tokenizer.Utf8Scan.next : type_of% @Slang.Source.Tokenizer.Utf8Scan.next
#audit_empty Slang.Source.Tokenizer.Utf8Scan.sound : type_of% @Slang.Source.Tokenizer.Utf8Scan.sound
#audit_empty Slang.Source.Tokenizer.Utf8Audit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Slang.Source.Tokenizer.Utf8Head
#print axioms Slang.Source.Tokenizer.Utf8Head.one
#print axioms Slang.Source.Tokenizer.Utf8Head.two
#print axioms Slang.Source.Tokenizer.Utf8Head.three
#print axioms Slang.Source.Tokenizer.Utf8Head.four
#print axioms Slang.Source.Tokenizer.Utf8Head.sound
#print axioms Slang.Source.Tokenizer.Utf8Scan
#print axioms Slang.Source.Tokenizer.Utf8Scan.done
#print axioms Slang.Source.Tokenizer.Utf8Scan.next
#print axioms Slang.Source.Tokenizer.Utf8Scan.sound
#print axioms Slang.Source.Tokenizer.Utf8Audit.nat_add_zero
#audit_empty Slang.Source.Tokenizer.Utf8Scan.continue : type_of% @Slang.Source.Tokenizer.Utf8Scan.continue
#print axioms Slang.Source.Tokenizer.Utf8Scan.continue
