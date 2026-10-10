import Iota.Audit.Empty
import Iota.Source.Parsing.Selection

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Iota.Source.Parsing.SelectionAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Source.Parsing.SelectionAudit

#audit_empty Iota.Source.Parsing.DropNewlinesScan : List Iota.Decode.Token → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.DropNewlinesScan.done : type_of% @Iota.Source.Parsing.DropNewlinesScan.done
#audit_empty Iota.Source.Parsing.DropNewlinesScan.newline : type_of% @Iota.Source.Parsing.DropNewlinesScan.newline
#audit_empty Iota.Source.Parsing.DropNewlinesScan.stop : type_of% @Iota.Source.Parsing.DropNewlinesScan.stop
#audit_empty Iota.Source.Parsing.DropNewlinesScan.sound : type_of% @Iota.Source.Parsing.DropNewlinesScan.sound
#audit_empty Iota.Source.Parsing.TakeLineScan : List Iota.Decode.Token → List Iota.Decode.Token → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.TakeLineScan.done : type_of% @Iota.Source.Parsing.TakeLineScan.done
#audit_empty Iota.Source.Parsing.TakeLineScan.newline : type_of% @Iota.Source.Parsing.TakeLineScan.newline
#audit_empty Iota.Source.Parsing.TakeLineScan.next : type_of% @Iota.Source.Parsing.TakeLineScan.next
#audit_empty Iota.Source.Parsing.TakeLineScan.sound : type_of% @Iota.Source.Parsing.TakeLineScan.sound
#audit_empty Iota.Source.Parsing.ConstantLineParse : List Iota.Decode.Token → Iota.Expr → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.ConstantLineParse.intro : type_of% @Iota.Source.Parsing.ConstantLineParse.intro
#audit_empty Iota.Source.Parsing.ConstantLineParse.sound : type_of% @Iota.Source.Parsing.ConstantLineParse.sound
#audit_empty Iota.Source.Parsing.SelectedFunctionParse : String → Nat → List Iota.Decode.Token → Iota.FunctionDecl → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.SelectedFunctionParse.intro : type_of% @Iota.Source.Parsing.SelectedFunctionParse.intro
#audit_empty Iota.Source.Parsing.SelectedFunctionParse.sound : type_of% @Iota.Source.Parsing.SelectedFunctionParse.sound
#audit_empty Iota.Source.Parsing.NonIdentifierToken : Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.NonIdentifierToken.numeral : type_of% @Iota.Source.Parsing.NonIdentifierToken.numeral
#audit_empty Iota.Source.Parsing.NonIdentifierToken.lparen : type_of% @Iota.Source.Parsing.NonIdentifierToken.lparen
#audit_empty Iota.Source.Parsing.NonIdentifierToken.rparen : type_of% @Iota.Source.Parsing.NonIdentifierToken.rparen
#audit_empty Iota.Source.Parsing.NonIdentifierToken.lbrace : type_of% @Iota.Source.Parsing.NonIdentifierToken.lbrace
#audit_empty Iota.Source.Parsing.NonIdentifierToken.rbrace : type_of% @Iota.Source.Parsing.NonIdentifierToken.rbrace
#audit_empty Iota.Source.Parsing.NonIdentifierToken.lbracket : type_of% @Iota.Source.Parsing.NonIdentifierToken.lbracket
#audit_empty Iota.Source.Parsing.NonIdentifierToken.rbracket : type_of% @Iota.Source.Parsing.NonIdentifierToken.rbracket
#audit_empty Iota.Source.Parsing.NonIdentifierToken.comma : type_of% @Iota.Source.Parsing.NonIdentifierToken.comma
#audit_empty Iota.Source.Parsing.NonIdentifierToken.semi : type_of% @Iota.Source.Parsing.NonIdentifierToken.semi
#audit_empty Iota.Source.Parsing.NonIdentifierToken.newline : type_of% @Iota.Source.Parsing.NonIdentifierToken.newline
#audit_empty Iota.Source.Parsing.NonIdentifierToken.assign : type_of% @Iota.Source.Parsing.NonIdentifierToken.assign
#audit_empty Iota.Source.Parsing.NonIdentifierToken.lt : type_of% @Iota.Source.Parsing.NonIdentifierToken.lt
#audit_empty Iota.Source.Parsing.NonIdentifierToken.le : type_of% @Iota.Source.Parsing.NonIdentifierToken.le
#audit_empty Iota.Source.Parsing.NonIdentifierToken.gt : type_of% @Iota.Source.Parsing.NonIdentifierToken.gt
#audit_empty Iota.Source.Parsing.NonIdentifierToken.ge : type_of% @Iota.Source.Parsing.NonIdentifierToken.ge
#audit_empty Iota.Source.Parsing.NonIdentifierToken.other : type_of% @Iota.Source.Parsing.NonIdentifierToken.other
#audit_empty Iota.Source.Parsing.NonIdentifierToken.sound : type_of% @Iota.Source.Parsing.NonIdentifierToken.sound
#audit_empty Iota.Source.Parsing.ScanSelected : Nat → List Iota.Decode.Token → List String → List String → List Iota.ConstantDecl → List Iota.FunctionDecl → Prop
#audit_empty Iota.Source.Parsing.ScanSelected.done : type_of% @Iota.Source.Parsing.ScanSelected.done
#audit_empty Iota.Source.Parsing.ScanSelected.skipNonIdentifier : type_of% @Iota.Source.Parsing.ScanSelected.skipNonIdentifier
#audit_empty Iota.Source.Parsing.ScanSelected.identEnd : type_of% @Iota.Source.Parsing.ScanSelected.identEnd
#audit_empty Iota.Source.Parsing.ScanSelected.skipOther : type_of% @Iota.Source.Parsing.ScanSelected.skipOther
#audit_empty Iota.Source.Parsing.ScanSelected.skipConstant : type_of% @Iota.Source.Parsing.ScanSelected.skipConstant
#audit_empty Iota.Source.Parsing.ScanSelected.constant : type_of% @Iota.Source.Parsing.ScanSelected.constant
#audit_empty Iota.Source.Parsing.ScanSelected.skipFunction : type_of% @Iota.Source.Parsing.ScanSelected.skipFunction
#audit_empty Iota.Source.Parsing.ScanSelected.function : type_of% @Iota.Source.Parsing.ScanSelected.function
#audit_empty Iota.Source.Parsing.ScanSelected.sound : type_of% @Iota.Source.Parsing.ScanSelected.sound
#audit_empty Iota.Source.Parsing.SelectionAudit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Iota.Source.Parsing.DropNewlinesScan
#print axioms Iota.Source.Parsing.DropNewlinesScan.done
#print axioms Iota.Source.Parsing.DropNewlinesScan.newline
#print axioms Iota.Source.Parsing.DropNewlinesScan.stop
#print axioms Iota.Source.Parsing.DropNewlinesScan.sound
#print axioms Iota.Source.Parsing.TakeLineScan
#print axioms Iota.Source.Parsing.TakeLineScan.done
#print axioms Iota.Source.Parsing.TakeLineScan.newline
#print axioms Iota.Source.Parsing.TakeLineScan.next
#print axioms Iota.Source.Parsing.TakeLineScan.sound
#print axioms Iota.Source.Parsing.ConstantLineParse
#print axioms Iota.Source.Parsing.ConstantLineParse.intro
#print axioms Iota.Source.Parsing.ConstantLineParse.sound
#print axioms Iota.Source.Parsing.SelectedFunctionParse
#print axioms Iota.Source.Parsing.SelectedFunctionParse.intro
#print axioms Iota.Source.Parsing.SelectedFunctionParse.sound
#print axioms Iota.Source.Parsing.NonIdentifierToken
#print axioms Iota.Source.Parsing.NonIdentifierToken.numeral
#print axioms Iota.Source.Parsing.NonIdentifierToken.lparen
#print axioms Iota.Source.Parsing.NonIdentifierToken.rparen
#print axioms Iota.Source.Parsing.NonIdentifierToken.lbrace
#print axioms Iota.Source.Parsing.NonIdentifierToken.rbrace
#print axioms Iota.Source.Parsing.NonIdentifierToken.lbracket
#print axioms Iota.Source.Parsing.NonIdentifierToken.rbracket
#print axioms Iota.Source.Parsing.NonIdentifierToken.comma
#print axioms Iota.Source.Parsing.NonIdentifierToken.semi
#print axioms Iota.Source.Parsing.NonIdentifierToken.newline
#print axioms Iota.Source.Parsing.NonIdentifierToken.assign
#print axioms Iota.Source.Parsing.NonIdentifierToken.lt
#print axioms Iota.Source.Parsing.NonIdentifierToken.le
#print axioms Iota.Source.Parsing.NonIdentifierToken.gt
#print axioms Iota.Source.Parsing.NonIdentifierToken.ge
#print axioms Iota.Source.Parsing.NonIdentifierToken.other
#print axioms Iota.Source.Parsing.NonIdentifierToken.sound
#print axioms Iota.Source.Parsing.ScanSelected
#print axioms Iota.Source.Parsing.ScanSelected.done
#print axioms Iota.Source.Parsing.ScanSelected.skipNonIdentifier
#print axioms Iota.Source.Parsing.ScanSelected.identEnd
#print axioms Iota.Source.Parsing.ScanSelected.skipOther
#print axioms Iota.Source.Parsing.ScanSelected.skipConstant
#print axioms Iota.Source.Parsing.ScanSelected.constant
#print axioms Iota.Source.Parsing.ScanSelected.skipFunction
#print axioms Iota.Source.Parsing.ScanSelected.function
#print axioms Iota.Source.Parsing.ScanSelected.sound
#print axioms Iota.Source.Parsing.SelectionAudit.nat_add_zero
#audit_empty Iota.Source.Parsing.DropNewlinesScan.append : type_of% @Iota.Source.Parsing.DropNewlinesScan.append
#audit_empty Iota.Source.Parsing.TakeLineScan.append : type_of% @Iota.Source.Parsing.TakeLineScan.append
#audit_empty Iota.Source.Parsing.ScanSelected.continue : type_of% @Iota.Source.Parsing.ScanSelected.continue
#print axioms Iota.Source.Parsing.DropNewlinesScan.append
#print axioms Iota.Source.Parsing.TakeLineScan.append
#print axioms Iota.Source.Parsing.ScanSelected.continue
