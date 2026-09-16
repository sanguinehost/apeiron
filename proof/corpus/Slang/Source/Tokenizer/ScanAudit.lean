import Slang.Audit.Empty
import Slang.Source.Tokenizer.Scan

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Slang.Source.Tokenizer.ScanAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Source.Tokenizer.ScanAudit

#audit_empty Slang.Source.Tokenizer.TakeWhileScan : (Char → Bool) → List Char → List Char → List Char → Prop
#audit_empty Slang.Source.Tokenizer.TakeWhileScan.nil : type_of% @Slang.Source.Tokenizer.TakeWhileScan.nil
#audit_empty Slang.Source.Tokenizer.TakeWhileScan.stop : type_of% @Slang.Source.Tokenizer.TakeWhileScan.stop
#audit_empty Slang.Source.Tokenizer.TakeWhileScan.next : type_of% @Slang.Source.Tokenizer.TakeWhileScan.next
#audit_empty Slang.Source.Tokenizer.TakeWhileScan.sound : type_of% @Slang.Source.Tokenizer.TakeWhileScan.sound
#audit_empty Slang.Source.Tokenizer.SkipCommentScan : List Char → Nat → List Char → Nat → Prop
#audit_empty Slang.Source.Tokenizer.SkipCommentScan.done : type_of% @Slang.Source.Tokenizer.SkipCommentScan.done
#audit_empty Slang.Source.Tokenizer.SkipCommentScan.newline : type_of% @Slang.Source.Tokenizer.SkipCommentScan.newline
#audit_empty Slang.Source.Tokenizer.SkipCommentScan.next : type_of% @Slang.Source.Tokenizer.SkipCommentScan.next
#audit_empty Slang.Source.Tokenizer.SkipCommentScan.sound : type_of% @Slang.Source.Tokenizer.SkipCommentScan.sound
#audit_empty Slang.Source.Tokenizer.DigitsFold : Nat → List Char → Nat → Prop
#audit_empty Slang.Source.Tokenizer.DigitsFold.done : type_of% @Slang.Source.Tokenizer.DigitsFold.done
#audit_empty Slang.Source.Tokenizer.DigitsFold.next : type_of% @Slang.Source.Tokenizer.DigitsFold.next
#audit_empty Slang.Source.Tokenizer.DigitsFold.sound : type_of% @Slang.Source.Tokenizer.DigitsFold.sound
#audit_empty Slang.Source.Tokenizer.Utf8LengthFold : Nat → List Char → Nat → Prop
#audit_empty Slang.Source.Tokenizer.Utf8LengthFold.done : type_of% @Slang.Source.Tokenizer.Utf8LengthFold.done
#audit_empty Slang.Source.Tokenizer.Utf8LengthFold.next : type_of% @Slang.Source.Tokenizer.Utf8LengthFold.next
#audit_empty Slang.Source.Tokenizer.Utf8LengthFold.sound : type_of% @Slang.Source.Tokenizer.Utf8LengthFold.sound
#audit_empty Slang.Source.Tokenizer.LexHead : List Char → Nat → List Char → Nat → Option Slang.Decode.Token → Prop
#audit_empty Slang.Source.Tokenizer.LexHead.space : type_of% @Slang.Source.Tokenizer.LexHead.space
#audit_empty Slang.Source.Tokenizer.LexHead.newline : type_of% @Slang.Source.Tokenizer.LexHead.newline
#audit_empty Slang.Source.Tokenizer.LexHead.slashEnd : type_of% @Slang.Source.Tokenizer.LexHead.slashEnd
#audit_empty Slang.Source.Tokenizer.LexHead.comment : type_of% @Slang.Source.Tokenizer.LexHead.comment
#audit_empty Slang.Source.Tokenizer.LexHead.slash : type_of% @Slang.Source.Tokenizer.LexHead.slash
#audit_empty Slang.Source.Tokenizer.LexHead.ident : type_of% @Slang.Source.Tokenizer.LexHead.ident
#audit_empty Slang.Source.Tokenizer.LexHead.numeral : type_of% @Slang.Source.Tokenizer.LexHead.numeral
#audit_empty Slang.Source.Tokenizer.LexHead.ltEnd : type_of% @Slang.Source.Tokenizer.LexHead.ltEnd
#audit_empty Slang.Source.Tokenizer.LexHead.le : type_of% @Slang.Source.Tokenizer.LexHead.le
#audit_empty Slang.Source.Tokenizer.LexHead.lt : type_of% @Slang.Source.Tokenizer.LexHead.lt
#audit_empty Slang.Source.Tokenizer.LexHead.gtEnd : type_of% @Slang.Source.Tokenizer.LexHead.gtEnd
#audit_empty Slang.Source.Tokenizer.LexHead.ge : type_of% @Slang.Source.Tokenizer.LexHead.ge
#audit_empty Slang.Source.Tokenizer.LexHead.gt : type_of% @Slang.Source.Tokenizer.LexHead.gt
#audit_empty Slang.Source.Tokenizer.LexHead.lparen : type_of% @Slang.Source.Tokenizer.LexHead.lparen
#audit_empty Slang.Source.Tokenizer.LexHead.rparen : type_of% @Slang.Source.Tokenizer.LexHead.rparen
#audit_empty Slang.Source.Tokenizer.LexHead.lbrace : type_of% @Slang.Source.Tokenizer.LexHead.lbrace
#audit_empty Slang.Source.Tokenizer.LexHead.rbrace : type_of% @Slang.Source.Tokenizer.LexHead.rbrace
#audit_empty Slang.Source.Tokenizer.LexHead.lbracket : type_of% @Slang.Source.Tokenizer.LexHead.lbracket
#audit_empty Slang.Source.Tokenizer.LexHead.rbracket : type_of% @Slang.Source.Tokenizer.LexHead.rbracket
#audit_empty Slang.Source.Tokenizer.LexHead.comma : type_of% @Slang.Source.Tokenizer.LexHead.comma
#audit_empty Slang.Source.Tokenizer.LexHead.semi : type_of% @Slang.Source.Tokenizer.LexHead.semi
#audit_empty Slang.Source.Tokenizer.LexHead.assign : type_of% @Slang.Source.Tokenizer.LexHead.assign
#audit_empty Slang.Source.Tokenizer.LexHead.other : type_of% @Slang.Source.Tokenizer.LexHead.other
#audit_empty Slang.Source.Tokenizer.LexHead.sound : type_of% @Slang.Source.Tokenizer.LexHead.sound
#audit_empty Slang.Source.Tokenizer.ReverseTokens : List Slang.Decode.Token → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Tokenizer.ReverseTokens.result : type_of% @Slang.Source.Tokenizer.ReverseTokens.result
#audit_empty Slang.Source.Tokenizer.ReverseTokens.sound : type_of% @Slang.Source.Tokenizer.ReverseTokens.sound
#audit_empty Slang.Source.Tokenizer.CharsScan : Nat → List Char → Nat → List Slang.Decode.Token → List Slang.Decode.Token → Nat → Prop
#audit_empty Slang.Source.Tokenizer.CharsScan.done : type_of% @Slang.Source.Tokenizer.CharsScan.done
#audit_empty Slang.Source.Tokenizer.CharsScan.nextNone : type_of% @Slang.Source.Tokenizer.CharsScan.nextNone
#audit_empty Slang.Source.Tokenizer.CharsScan.nextToken : type_of% @Slang.Source.Tokenizer.CharsScan.nextToken
#audit_empty Slang.Source.Tokenizer.CharsScan.sound : type_of% @Slang.Source.Tokenizer.CharsScan.sound
#audit_empty Slang.Source.Tokenizer.ScanAudit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Slang.Source.Tokenizer.TakeWhileScan
#print axioms Slang.Source.Tokenizer.TakeWhileScan.nil
#print axioms Slang.Source.Tokenizer.TakeWhileScan.stop
#print axioms Slang.Source.Tokenizer.TakeWhileScan.next
#print axioms Slang.Source.Tokenizer.TakeWhileScan.sound
#print axioms Slang.Source.Tokenizer.SkipCommentScan
#print axioms Slang.Source.Tokenizer.SkipCommentScan.done
#print axioms Slang.Source.Tokenizer.SkipCommentScan.newline
#print axioms Slang.Source.Tokenizer.SkipCommentScan.next
#print axioms Slang.Source.Tokenizer.SkipCommentScan.sound
#print axioms Slang.Source.Tokenizer.DigitsFold
#print axioms Slang.Source.Tokenizer.DigitsFold.done
#print axioms Slang.Source.Tokenizer.DigitsFold.next
#print axioms Slang.Source.Tokenizer.DigitsFold.sound
#print axioms Slang.Source.Tokenizer.Utf8LengthFold
#print axioms Slang.Source.Tokenizer.Utf8LengthFold.done
#print axioms Slang.Source.Tokenizer.Utf8LengthFold.next
#print axioms Slang.Source.Tokenizer.Utf8LengthFold.sound
#print axioms Slang.Source.Tokenizer.LexHead
#print axioms Slang.Source.Tokenizer.LexHead.space
#print axioms Slang.Source.Tokenizer.LexHead.newline
#print axioms Slang.Source.Tokenizer.LexHead.slashEnd
#print axioms Slang.Source.Tokenizer.LexHead.comment
#print axioms Slang.Source.Tokenizer.LexHead.slash
#print axioms Slang.Source.Tokenizer.LexHead.ident
#print axioms Slang.Source.Tokenizer.LexHead.numeral
#print axioms Slang.Source.Tokenizer.LexHead.ltEnd
#print axioms Slang.Source.Tokenizer.LexHead.le
#print axioms Slang.Source.Tokenizer.LexHead.lt
#print axioms Slang.Source.Tokenizer.LexHead.gtEnd
#print axioms Slang.Source.Tokenizer.LexHead.ge
#print axioms Slang.Source.Tokenizer.LexHead.gt
#print axioms Slang.Source.Tokenizer.LexHead.lparen
#print axioms Slang.Source.Tokenizer.LexHead.rparen
#print axioms Slang.Source.Tokenizer.LexHead.lbrace
#print axioms Slang.Source.Tokenizer.LexHead.rbrace
#print axioms Slang.Source.Tokenizer.LexHead.lbracket
#print axioms Slang.Source.Tokenizer.LexHead.rbracket
#print axioms Slang.Source.Tokenizer.LexHead.comma
#print axioms Slang.Source.Tokenizer.LexHead.semi
#print axioms Slang.Source.Tokenizer.LexHead.assign
#print axioms Slang.Source.Tokenizer.LexHead.other
#print axioms Slang.Source.Tokenizer.LexHead.sound
#print axioms Slang.Source.Tokenizer.ReverseTokens
#print axioms Slang.Source.Tokenizer.ReverseTokens.result
#print axioms Slang.Source.Tokenizer.ReverseTokens.sound
#print axioms Slang.Source.Tokenizer.CharsScan
#print axioms Slang.Source.Tokenizer.CharsScan.done
#print axioms Slang.Source.Tokenizer.CharsScan.nextNone
#print axioms Slang.Source.Tokenizer.CharsScan.nextToken
#print axioms Slang.Source.Tokenizer.CharsScan.sound
#print axioms Slang.Source.Tokenizer.ScanAudit.nat_add_zero
