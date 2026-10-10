import Iota.Audit.Empty
import Iota.Source.Tokenizer.Utf8

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Iota.Source.Tokenizer.Utf8Audit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Source.Tokenizer.Utf8Audit

#audit_empty Iota.Source.Tokenizer.Utf8Head : List UInt8 → Char → List UInt8 → Prop
#audit_empty Iota.Source.Tokenizer.Utf8Head.one : type_of% @Iota.Source.Tokenizer.Utf8Head.one
#audit_empty Iota.Source.Tokenizer.Utf8Head.two : type_of% @Iota.Source.Tokenizer.Utf8Head.two
#audit_empty Iota.Source.Tokenizer.Utf8Head.three : type_of% @Iota.Source.Tokenizer.Utf8Head.three
#audit_empty Iota.Source.Tokenizer.Utf8Head.four : type_of% @Iota.Source.Tokenizer.Utf8Head.four
#audit_empty Iota.Source.Tokenizer.Utf8Head.sound : type_of% @Iota.Source.Tokenizer.Utf8Head.sound
#audit_empty Iota.Source.Tokenizer.Utf8Scan : Nat → List UInt8 → List Char → Prop
#audit_empty Iota.Source.Tokenizer.Utf8Scan.done : type_of% @Iota.Source.Tokenizer.Utf8Scan.done
#audit_empty Iota.Source.Tokenizer.Utf8Scan.next : type_of% @Iota.Source.Tokenizer.Utf8Scan.next
#audit_empty Iota.Source.Tokenizer.Utf8Scan.sound : type_of% @Iota.Source.Tokenizer.Utf8Scan.sound
#audit_empty Iota.Source.Tokenizer.Utf8Audit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Iota.Source.Tokenizer.Utf8Head
#print axioms Iota.Source.Tokenizer.Utf8Head.one
#print axioms Iota.Source.Tokenizer.Utf8Head.two
#print axioms Iota.Source.Tokenizer.Utf8Head.three
#print axioms Iota.Source.Tokenizer.Utf8Head.four
#print axioms Iota.Source.Tokenizer.Utf8Head.sound
#print axioms Iota.Source.Tokenizer.Utf8Scan
#print axioms Iota.Source.Tokenizer.Utf8Scan.done
#print axioms Iota.Source.Tokenizer.Utf8Scan.next
#print axioms Iota.Source.Tokenizer.Utf8Scan.sound
#print axioms Iota.Source.Tokenizer.Utf8Audit.nat_add_zero
#audit_empty Iota.Source.Tokenizer.Utf8Scan.continue : type_of% @Iota.Source.Tokenizer.Utf8Scan.continue
#print axioms Iota.Source.Tokenizer.Utf8Scan.continue
