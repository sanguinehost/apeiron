import Iota.Audit.Empty
import Iota.Source.Tokenizer.Scan

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Iota.Source.Tokenizer.ScanAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Source.Tokenizer.ScanAudit

#audit_empty Iota.Source.Tokenizer.TakeWhileScan : (Char → Bool) → List Char → List Char → List Char → Prop
#audit_empty Iota.Source.Tokenizer.TakeWhileScan.nil : type_of% @Iota.Source.Tokenizer.TakeWhileScan.nil
#audit_empty Iota.Source.Tokenizer.TakeWhileScan.stop : type_of% @Iota.Source.Tokenizer.TakeWhileScan.stop
#audit_empty Iota.Source.Tokenizer.TakeWhileScan.next : type_of% @Iota.Source.Tokenizer.TakeWhileScan.next
#audit_empty Iota.Source.Tokenizer.TakeWhileScan.sound : type_of% @Iota.Source.Tokenizer.TakeWhileScan.sound
#audit_empty Iota.Source.Tokenizer.SkipCommentScan : List Char → Nat → List Char → Nat → Prop
#audit_empty Iota.Source.Tokenizer.SkipCommentScan.done : type_of% @Iota.Source.Tokenizer.SkipCommentScan.done
#audit_empty Iota.Source.Tokenizer.SkipCommentScan.newline : type_of% @Iota.Source.Tokenizer.SkipCommentScan.newline
#audit_empty Iota.Source.Tokenizer.SkipCommentScan.next : type_of% @Iota.Source.Tokenizer.SkipCommentScan.next
#audit_empty Iota.Source.Tokenizer.SkipCommentScan.sound : type_of% @Iota.Source.Tokenizer.SkipCommentScan.sound
#audit_empty Iota.Source.Tokenizer.DigitsFold : Nat → List Char → Nat → Prop
#audit_empty Iota.Source.Tokenizer.DigitsFold.done : type_of% @Iota.Source.Tokenizer.DigitsFold.done
#audit_empty Iota.Source.Tokenizer.DigitsFold.next : type_of% @Iota.Source.Tokenizer.DigitsFold.next
#audit_empty Iota.Source.Tokenizer.DigitsFold.sound : type_of% @Iota.Source.Tokenizer.DigitsFold.sound
#audit_empty Iota.Source.Tokenizer.Utf8LengthFold : Nat → List Char → Nat → Prop
#audit_empty Iota.Source.Tokenizer.Utf8LengthFold.done : type_of% @Iota.Source.Tokenizer.Utf8LengthFold.done
#audit_empty Iota.Source.Tokenizer.Utf8LengthFold.next : type_of% @Iota.Source.Tokenizer.Utf8LengthFold.next
#audit_empty Iota.Source.Tokenizer.Utf8LengthFold.sound : type_of% @Iota.Source.Tokenizer.Utf8LengthFold.sound
#audit_empty Iota.Source.Tokenizer.LexHead : List Char → Nat → List Char → Nat → Option Iota.Decode.Token → Prop
#audit_empty Iota.Source.Tokenizer.LexHead.space : type_of% @Iota.Source.Tokenizer.LexHead.space
#audit_empty Iota.Source.Tokenizer.LexHead.newline : type_of% @Iota.Source.Tokenizer.LexHead.newline
#audit_empty Iota.Source.Tokenizer.LexHead.slashEnd : type_of% @Iota.Source.Tokenizer.LexHead.slashEnd
#audit_empty Iota.Source.Tokenizer.LexHead.comment : type_of% @Iota.Source.Tokenizer.LexHead.comment
#audit_empty Iota.Source.Tokenizer.LexHead.slash : type_of% @Iota.Source.Tokenizer.LexHead.slash
#audit_empty Iota.Source.Tokenizer.LexHead.ident : type_of% @Iota.Source.Tokenizer.LexHead.ident
#audit_empty Iota.Source.Tokenizer.LexHead.numeral : type_of% @Iota.Source.Tokenizer.LexHead.numeral
#audit_empty Iota.Source.Tokenizer.LexHead.ltEnd : type_of% @Iota.Source.Tokenizer.LexHead.ltEnd
#audit_empty Iota.Source.Tokenizer.LexHead.le : type_of% @Iota.Source.Tokenizer.LexHead.le
#audit_empty Iota.Source.Tokenizer.LexHead.lt : type_of% @Iota.Source.Tokenizer.LexHead.lt
#audit_empty Iota.Source.Tokenizer.LexHead.gtEnd : type_of% @Iota.Source.Tokenizer.LexHead.gtEnd
#audit_empty Iota.Source.Tokenizer.LexHead.ge : type_of% @Iota.Source.Tokenizer.LexHead.ge
#audit_empty Iota.Source.Tokenizer.LexHead.gt : type_of% @Iota.Source.Tokenizer.LexHead.gt
#audit_empty Iota.Source.Tokenizer.LexHead.lparen : type_of% @Iota.Source.Tokenizer.LexHead.lparen
#audit_empty Iota.Source.Tokenizer.LexHead.rparen : type_of% @Iota.Source.Tokenizer.LexHead.rparen
#audit_empty Iota.Source.Tokenizer.LexHead.lbrace : type_of% @Iota.Source.Tokenizer.LexHead.lbrace
#audit_empty Iota.Source.Tokenizer.LexHead.rbrace : type_of% @Iota.Source.Tokenizer.LexHead.rbrace
#audit_empty Iota.Source.Tokenizer.LexHead.lbracket : type_of% @Iota.Source.Tokenizer.LexHead.lbracket
#audit_empty Iota.Source.Tokenizer.LexHead.rbracket : type_of% @Iota.Source.Tokenizer.LexHead.rbracket
#audit_empty Iota.Source.Tokenizer.LexHead.comma : type_of% @Iota.Source.Tokenizer.LexHead.comma
#audit_empty Iota.Source.Tokenizer.LexHead.semi : type_of% @Iota.Source.Tokenizer.LexHead.semi
#audit_empty Iota.Source.Tokenizer.LexHead.assign : type_of% @Iota.Source.Tokenizer.LexHead.assign
#audit_empty Iota.Source.Tokenizer.LexHead.other : type_of% @Iota.Source.Tokenizer.LexHead.other
#audit_empty Iota.Source.Tokenizer.LexHead.sound : type_of% @Iota.Source.Tokenizer.LexHead.sound
#audit_empty Iota.Source.Tokenizer.ReverseTokens : List Iota.Decode.Token → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Tokenizer.ReverseTokens.result : type_of% @Iota.Source.Tokenizer.ReverseTokens.result
#audit_empty Iota.Source.Tokenizer.ReverseTokens.sound : type_of% @Iota.Source.Tokenizer.ReverseTokens.sound
#audit_empty Iota.Source.Tokenizer.CharsScan : Nat → List Char → Nat → List Iota.Decode.Token → List Iota.Decode.Token → Nat → Prop
#audit_empty Iota.Source.Tokenizer.CharsScan.done : type_of% @Iota.Source.Tokenizer.CharsScan.done
#audit_empty Iota.Source.Tokenizer.CharsScan.nextNone : type_of% @Iota.Source.Tokenizer.CharsScan.nextNone
#audit_empty Iota.Source.Tokenizer.CharsScan.nextToken : type_of% @Iota.Source.Tokenizer.CharsScan.nextToken
#audit_empty Iota.Source.Tokenizer.CharsScan.sound : type_of% @Iota.Source.Tokenizer.CharsScan.sound
#audit_empty Iota.Source.Tokenizer.ScanAudit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Iota.Source.Tokenizer.TakeWhileScan
#print axioms Iota.Source.Tokenizer.TakeWhileScan.nil
#print axioms Iota.Source.Tokenizer.TakeWhileScan.stop
#print axioms Iota.Source.Tokenizer.TakeWhileScan.next
#print axioms Iota.Source.Tokenizer.TakeWhileScan.sound
#print axioms Iota.Source.Tokenizer.SkipCommentScan
#print axioms Iota.Source.Tokenizer.SkipCommentScan.done
#print axioms Iota.Source.Tokenizer.SkipCommentScan.newline
#print axioms Iota.Source.Tokenizer.SkipCommentScan.next
#print axioms Iota.Source.Tokenizer.SkipCommentScan.sound
#print axioms Iota.Source.Tokenizer.DigitsFold
#print axioms Iota.Source.Tokenizer.DigitsFold.done
#print axioms Iota.Source.Tokenizer.DigitsFold.next
#print axioms Iota.Source.Tokenizer.DigitsFold.sound
#print axioms Iota.Source.Tokenizer.Utf8LengthFold
#print axioms Iota.Source.Tokenizer.Utf8LengthFold.done
#print axioms Iota.Source.Tokenizer.Utf8LengthFold.next
#print axioms Iota.Source.Tokenizer.Utf8LengthFold.sound
#print axioms Iota.Source.Tokenizer.LexHead
#print axioms Iota.Source.Tokenizer.LexHead.space
#print axioms Iota.Source.Tokenizer.LexHead.newline
#print axioms Iota.Source.Tokenizer.LexHead.slashEnd
#print axioms Iota.Source.Tokenizer.LexHead.comment
#print axioms Iota.Source.Tokenizer.LexHead.slash
#print axioms Iota.Source.Tokenizer.LexHead.ident
#print axioms Iota.Source.Tokenizer.LexHead.numeral
#print axioms Iota.Source.Tokenizer.LexHead.ltEnd
#print axioms Iota.Source.Tokenizer.LexHead.le
#print axioms Iota.Source.Tokenizer.LexHead.lt
#print axioms Iota.Source.Tokenizer.LexHead.gtEnd
#print axioms Iota.Source.Tokenizer.LexHead.ge
#print axioms Iota.Source.Tokenizer.LexHead.gt
#print axioms Iota.Source.Tokenizer.LexHead.lparen
#print axioms Iota.Source.Tokenizer.LexHead.rparen
#print axioms Iota.Source.Tokenizer.LexHead.lbrace
#print axioms Iota.Source.Tokenizer.LexHead.rbrace
#print axioms Iota.Source.Tokenizer.LexHead.lbracket
#print axioms Iota.Source.Tokenizer.LexHead.rbracket
#print axioms Iota.Source.Tokenizer.LexHead.comma
#print axioms Iota.Source.Tokenizer.LexHead.semi
#print axioms Iota.Source.Tokenizer.LexHead.assign
#print axioms Iota.Source.Tokenizer.LexHead.other
#print axioms Iota.Source.Tokenizer.LexHead.sound
#print axioms Iota.Source.Tokenizer.ReverseTokens
#print axioms Iota.Source.Tokenizer.ReverseTokens.result
#print axioms Iota.Source.Tokenizer.ReverseTokens.sound
#print axioms Iota.Source.Tokenizer.CharsScan
#print axioms Iota.Source.Tokenizer.CharsScan.done
#print axioms Iota.Source.Tokenizer.CharsScan.nextNone
#print axioms Iota.Source.Tokenizer.CharsScan.nextToken
#print axioms Iota.Source.Tokenizer.CharsScan.sound
#print axioms Iota.Source.Tokenizer.ScanAudit.nat_add_zero
#audit_empty Iota.Source.Tokenizer.ReverseTokens.of_eq : type_of% @Iota.Source.Tokenizer.ReverseTokens.of_eq
#audit_empty Iota.Source.Tokenizer.CharsScan.continue : type_of% @Iota.Source.Tokenizer.CharsScan.continue
#audit_empty Iota.Source.Tokenizer.DigitsFold.append : type_of% @Iota.Source.Tokenizer.DigitsFold.append
#audit_empty Iota.Source.Tokenizer.Utf8LengthFold.append : type_of% @Iota.Source.Tokenizer.Utf8LengthFold.append
#print axioms Iota.Source.Tokenizer.ReverseTokens.of_eq
#print axioms Iota.Source.Tokenizer.CharsScan.continue
#print axioms Iota.Source.Tokenizer.DigitsFold.append
#print axioms Iota.Source.Tokenizer.Utf8LengthFold.append
