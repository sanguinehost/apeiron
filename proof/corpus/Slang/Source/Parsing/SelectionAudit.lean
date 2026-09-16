import Slang.Audit.Empty
import Slang.Source.Parsing.Selection

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Slang.Source.Parsing.SelectionAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Source.Parsing.SelectionAudit

#audit_empty Slang.Source.Parsing.DropNewlinesScan : List Slang.Decode.Token → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.DropNewlinesScan.done : type_of% @Slang.Source.Parsing.DropNewlinesScan.done
#audit_empty Slang.Source.Parsing.DropNewlinesScan.newline : type_of% @Slang.Source.Parsing.DropNewlinesScan.newline
#audit_empty Slang.Source.Parsing.DropNewlinesScan.stop : type_of% @Slang.Source.Parsing.DropNewlinesScan.stop
#audit_empty Slang.Source.Parsing.DropNewlinesScan.sound : type_of% @Slang.Source.Parsing.DropNewlinesScan.sound
#audit_empty Slang.Source.Parsing.TakeLineScan : List Slang.Decode.Token → List Slang.Decode.Token → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.TakeLineScan.done : type_of% @Slang.Source.Parsing.TakeLineScan.done
#audit_empty Slang.Source.Parsing.TakeLineScan.newline : type_of% @Slang.Source.Parsing.TakeLineScan.newline
#audit_empty Slang.Source.Parsing.TakeLineScan.next : type_of% @Slang.Source.Parsing.TakeLineScan.next
#audit_empty Slang.Source.Parsing.TakeLineScan.sound : type_of% @Slang.Source.Parsing.TakeLineScan.sound
#audit_empty Slang.Source.Parsing.ConstantLineParse : List Slang.Decode.Token → Slang.Expr → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.ConstantLineParse.intro : type_of% @Slang.Source.Parsing.ConstantLineParse.intro
#audit_empty Slang.Source.Parsing.ConstantLineParse.sound : type_of% @Slang.Source.Parsing.ConstantLineParse.sound
#audit_empty Slang.Source.Parsing.SelectedFunctionParse : String → Nat → List Slang.Decode.Token → Slang.FunctionDecl → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.SelectedFunctionParse.intro : type_of% @Slang.Source.Parsing.SelectedFunctionParse.intro
#audit_empty Slang.Source.Parsing.SelectedFunctionParse.sound : type_of% @Slang.Source.Parsing.SelectedFunctionParse.sound
#audit_empty Slang.Source.Parsing.NonIdentifierToken : Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.NonIdentifierToken.numeral : type_of% @Slang.Source.Parsing.NonIdentifierToken.numeral
#audit_empty Slang.Source.Parsing.NonIdentifierToken.lparen : type_of% @Slang.Source.Parsing.NonIdentifierToken.lparen
#audit_empty Slang.Source.Parsing.NonIdentifierToken.rparen : type_of% @Slang.Source.Parsing.NonIdentifierToken.rparen
#audit_empty Slang.Source.Parsing.NonIdentifierToken.lbrace : type_of% @Slang.Source.Parsing.NonIdentifierToken.lbrace
#audit_empty Slang.Source.Parsing.NonIdentifierToken.rbrace : type_of% @Slang.Source.Parsing.NonIdentifierToken.rbrace
#audit_empty Slang.Source.Parsing.NonIdentifierToken.lbracket : type_of% @Slang.Source.Parsing.NonIdentifierToken.lbracket
#audit_empty Slang.Source.Parsing.NonIdentifierToken.rbracket : type_of% @Slang.Source.Parsing.NonIdentifierToken.rbracket
#audit_empty Slang.Source.Parsing.NonIdentifierToken.comma : type_of% @Slang.Source.Parsing.NonIdentifierToken.comma
#audit_empty Slang.Source.Parsing.NonIdentifierToken.semi : type_of% @Slang.Source.Parsing.NonIdentifierToken.semi
#audit_empty Slang.Source.Parsing.NonIdentifierToken.newline : type_of% @Slang.Source.Parsing.NonIdentifierToken.newline
#audit_empty Slang.Source.Parsing.NonIdentifierToken.assign : type_of% @Slang.Source.Parsing.NonIdentifierToken.assign
#audit_empty Slang.Source.Parsing.NonIdentifierToken.lt : type_of% @Slang.Source.Parsing.NonIdentifierToken.lt
#audit_empty Slang.Source.Parsing.NonIdentifierToken.le : type_of% @Slang.Source.Parsing.NonIdentifierToken.le
#audit_empty Slang.Source.Parsing.NonIdentifierToken.gt : type_of% @Slang.Source.Parsing.NonIdentifierToken.gt
#audit_empty Slang.Source.Parsing.NonIdentifierToken.ge : type_of% @Slang.Source.Parsing.NonIdentifierToken.ge
#audit_empty Slang.Source.Parsing.NonIdentifierToken.other : type_of% @Slang.Source.Parsing.NonIdentifierToken.other
#audit_empty Slang.Source.Parsing.NonIdentifierToken.sound : type_of% @Slang.Source.Parsing.NonIdentifierToken.sound
#audit_empty Slang.Source.Parsing.ScanSelected : Nat → List Slang.Decode.Token → List String → List String → List Slang.ConstantDecl → List Slang.FunctionDecl → Prop
#audit_empty Slang.Source.Parsing.ScanSelected.done : type_of% @Slang.Source.Parsing.ScanSelected.done
#audit_empty Slang.Source.Parsing.ScanSelected.skipNonIdentifier : type_of% @Slang.Source.Parsing.ScanSelected.skipNonIdentifier
#audit_empty Slang.Source.Parsing.ScanSelected.identEnd : type_of% @Slang.Source.Parsing.ScanSelected.identEnd
#audit_empty Slang.Source.Parsing.ScanSelected.skipOther : type_of% @Slang.Source.Parsing.ScanSelected.skipOther
#audit_empty Slang.Source.Parsing.ScanSelected.skipConstant : type_of% @Slang.Source.Parsing.ScanSelected.skipConstant
#audit_empty Slang.Source.Parsing.ScanSelected.constant : type_of% @Slang.Source.Parsing.ScanSelected.constant
#audit_empty Slang.Source.Parsing.ScanSelected.skipFunction : type_of% @Slang.Source.Parsing.ScanSelected.skipFunction
#audit_empty Slang.Source.Parsing.ScanSelected.function : type_of% @Slang.Source.Parsing.ScanSelected.function
#audit_empty Slang.Source.Parsing.ScanSelected.sound : type_of% @Slang.Source.Parsing.ScanSelected.sound
#audit_empty Slang.Source.Parsing.SelectionAudit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Slang.Source.Parsing.DropNewlinesScan
#print axioms Slang.Source.Parsing.DropNewlinesScan.done
#print axioms Slang.Source.Parsing.DropNewlinesScan.newline
#print axioms Slang.Source.Parsing.DropNewlinesScan.stop
#print axioms Slang.Source.Parsing.DropNewlinesScan.sound
#print axioms Slang.Source.Parsing.TakeLineScan
#print axioms Slang.Source.Parsing.TakeLineScan.done
#print axioms Slang.Source.Parsing.TakeLineScan.newline
#print axioms Slang.Source.Parsing.TakeLineScan.next
#print axioms Slang.Source.Parsing.TakeLineScan.sound
#print axioms Slang.Source.Parsing.ConstantLineParse
#print axioms Slang.Source.Parsing.ConstantLineParse.intro
#print axioms Slang.Source.Parsing.ConstantLineParse.sound
#print axioms Slang.Source.Parsing.SelectedFunctionParse
#print axioms Slang.Source.Parsing.SelectedFunctionParse.intro
#print axioms Slang.Source.Parsing.SelectedFunctionParse.sound
#print axioms Slang.Source.Parsing.NonIdentifierToken
#print axioms Slang.Source.Parsing.NonIdentifierToken.numeral
#print axioms Slang.Source.Parsing.NonIdentifierToken.lparen
#print axioms Slang.Source.Parsing.NonIdentifierToken.rparen
#print axioms Slang.Source.Parsing.NonIdentifierToken.lbrace
#print axioms Slang.Source.Parsing.NonIdentifierToken.rbrace
#print axioms Slang.Source.Parsing.NonIdentifierToken.lbracket
#print axioms Slang.Source.Parsing.NonIdentifierToken.rbracket
#print axioms Slang.Source.Parsing.NonIdentifierToken.comma
#print axioms Slang.Source.Parsing.NonIdentifierToken.semi
#print axioms Slang.Source.Parsing.NonIdentifierToken.newline
#print axioms Slang.Source.Parsing.NonIdentifierToken.assign
#print axioms Slang.Source.Parsing.NonIdentifierToken.lt
#print axioms Slang.Source.Parsing.NonIdentifierToken.le
#print axioms Slang.Source.Parsing.NonIdentifierToken.gt
#print axioms Slang.Source.Parsing.NonIdentifierToken.ge
#print axioms Slang.Source.Parsing.NonIdentifierToken.other
#print axioms Slang.Source.Parsing.NonIdentifierToken.sound
#print axioms Slang.Source.Parsing.ScanSelected
#print axioms Slang.Source.Parsing.ScanSelected.done
#print axioms Slang.Source.Parsing.ScanSelected.skipNonIdentifier
#print axioms Slang.Source.Parsing.ScanSelected.identEnd
#print axioms Slang.Source.Parsing.ScanSelected.skipOther
#print axioms Slang.Source.Parsing.ScanSelected.skipConstant
#print axioms Slang.Source.Parsing.ScanSelected.constant
#print axioms Slang.Source.Parsing.ScanSelected.skipFunction
#print axioms Slang.Source.Parsing.ScanSelected.function
#print axioms Slang.Source.Parsing.ScanSelected.sound
#print axioms Slang.Source.Parsing.SelectionAudit.nat_add_zero
