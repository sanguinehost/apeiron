import Iota.Audit.Empty
import Iota.Source.Parsing.Statement

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Iota.Source.Parsing.StatementAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Source.Parsing.StatementAudit

#audit_empty Iota.Source.Parsing.ParameterScan : Nat → List Iota.Decode.Token → List String → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.ParameterScan.done : type_of% @Iota.Source.Parsing.ParameterScan.done
#audit_empty Iota.Source.Parsing.ParameterScan.comma : type_of% @Iota.Source.Parsing.ParameterScan.comma
#audit_empty Iota.Source.Parsing.ParameterScan.last : type_of% @Iota.Source.Parsing.ParameterScan.last
#audit_empty Iota.Source.Parsing.ParameterScan.sound : type_of% @Iota.Source.Parsing.ParameterScan.sound
#audit_empty Iota.Source.Parsing.ExpressionStatementToken : Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.numeral : type_of% @Iota.Source.Parsing.ExpressionStatementToken.numeral
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.lparen : type_of% @Iota.Source.Parsing.ExpressionStatementToken.lparen
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.rparen : type_of% @Iota.Source.Parsing.ExpressionStatementToken.rparen
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.lbrace : type_of% @Iota.Source.Parsing.ExpressionStatementToken.lbrace
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.rbrace : type_of% @Iota.Source.Parsing.ExpressionStatementToken.rbrace
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.lbracket : type_of% @Iota.Source.Parsing.ExpressionStatementToken.lbracket
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.rbracket : type_of% @Iota.Source.Parsing.ExpressionStatementToken.rbracket
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.comma : type_of% @Iota.Source.Parsing.ExpressionStatementToken.comma
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.semi : type_of% @Iota.Source.Parsing.ExpressionStatementToken.semi
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.newline : type_of% @Iota.Source.Parsing.ExpressionStatementToken.newline
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.assign : type_of% @Iota.Source.Parsing.ExpressionStatementToken.assign
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.lt : type_of% @Iota.Source.Parsing.ExpressionStatementToken.lt
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.le : type_of% @Iota.Source.Parsing.ExpressionStatementToken.le
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.gt : type_of% @Iota.Source.Parsing.ExpressionStatementToken.gt
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.ge : type_of% @Iota.Source.Parsing.ExpressionStatementToken.ge
#audit_empty Iota.Source.Parsing.ExpressionStatementToken.other : type_of% @Iota.Source.Parsing.ExpressionStatementToken.other
#audit_empty Iota.Source.Parsing.StmtScan : Nat → List Iota.Decode.Token → Iota.Stmt → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.StmtScan.return_ : type_of% @Iota.Source.Parsing.StmtScan.return_
#audit_empty Iota.Source.Parsing.StmtScan.ifNoElseEnd : type_of% @Iota.Source.Parsing.StmtScan.ifNoElseEnd
#audit_empty Iota.Source.Parsing.StmtScan.ifNoElseNext : type_of% @Iota.Source.Parsing.StmtScan.ifNoElseNext
#audit_empty Iota.Source.Parsing.StmtScan.ifElse : type_of% @Iota.Source.Parsing.StmtScan.ifElse
#audit_empty Iota.Source.Parsing.StmtScan.store : type_of% @Iota.Source.Parsing.StmtScan.store
#audit_empty Iota.Source.Parsing.StmtScan.memoryExpression : type_of% @Iota.Source.Parsing.StmtScan.memoryExpression
#audit_empty Iota.Source.Parsing.StmtScan.bind : type_of% @Iota.Source.Parsing.StmtScan.bind
#audit_empty Iota.Source.Parsing.StmtScan.nameExpression : type_of% @Iota.Source.Parsing.StmtScan.nameExpression
#audit_empty Iota.Source.Parsing.StmtScan.otherExpression : type_of% @Iota.Source.Parsing.StmtScan.otherExpression
#audit_empty Iota.Source.Parsing.StmtScan.sound : type_of% @Iota.Source.Parsing.StmtScan.sound
#audit_empty Iota.Source.Parsing.BlockScan : Nat → List Iota.Decode.Token → List Iota.Stmt → Iota.Decode.Token → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.BlockScan.done : type_of% @Iota.Source.Parsing.BlockScan.done
#audit_empty Iota.Source.Parsing.BlockScan.next : type_of% @Iota.Source.Parsing.BlockScan.next
#audit_empty Iota.Source.Parsing.BlockScan.sound : type_of% @Iota.Source.Parsing.BlockScan.sound
#audit_empty Iota.Source.Parsing.FunctionScan : String → Nat → List Iota.Decode.Token → Iota.FunctionDecl → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.FunctionScan.intro : type_of% @Iota.Source.Parsing.FunctionScan.intro
#audit_empty Iota.Source.Parsing.FunctionScan.sound : type_of% @Iota.Source.Parsing.FunctionScan.sound
#audit_empty Iota.Source.Parsing.StatementAudit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Iota.Source.Parsing.ParameterScan
#print axioms Iota.Source.Parsing.ParameterScan.done
#print axioms Iota.Source.Parsing.ParameterScan.comma
#print axioms Iota.Source.Parsing.ParameterScan.last
#print axioms Iota.Source.Parsing.ParameterScan.sound
#print axioms Iota.Source.Parsing.ExpressionStatementToken
#print axioms Iota.Source.Parsing.ExpressionStatementToken.numeral
#print axioms Iota.Source.Parsing.ExpressionStatementToken.lparen
#print axioms Iota.Source.Parsing.ExpressionStatementToken.rparen
#print axioms Iota.Source.Parsing.ExpressionStatementToken.lbrace
#print axioms Iota.Source.Parsing.ExpressionStatementToken.rbrace
#print axioms Iota.Source.Parsing.ExpressionStatementToken.lbracket
#print axioms Iota.Source.Parsing.ExpressionStatementToken.rbracket
#print axioms Iota.Source.Parsing.ExpressionStatementToken.comma
#print axioms Iota.Source.Parsing.ExpressionStatementToken.semi
#print axioms Iota.Source.Parsing.ExpressionStatementToken.newline
#print axioms Iota.Source.Parsing.ExpressionStatementToken.assign
#print axioms Iota.Source.Parsing.ExpressionStatementToken.lt
#print axioms Iota.Source.Parsing.ExpressionStatementToken.le
#print axioms Iota.Source.Parsing.ExpressionStatementToken.gt
#print axioms Iota.Source.Parsing.ExpressionStatementToken.ge
#print axioms Iota.Source.Parsing.ExpressionStatementToken.other
#print axioms Iota.Source.Parsing.StmtScan
#print axioms Iota.Source.Parsing.StmtScan.return_
#print axioms Iota.Source.Parsing.StmtScan.ifNoElseEnd
#print axioms Iota.Source.Parsing.StmtScan.ifNoElseNext
#print axioms Iota.Source.Parsing.StmtScan.ifElse
#print axioms Iota.Source.Parsing.StmtScan.store
#print axioms Iota.Source.Parsing.StmtScan.memoryExpression
#print axioms Iota.Source.Parsing.StmtScan.bind
#print axioms Iota.Source.Parsing.StmtScan.nameExpression
#print axioms Iota.Source.Parsing.StmtScan.otherExpression
#print axioms Iota.Source.Parsing.StmtScan.sound
#print axioms Iota.Source.Parsing.BlockScan
#print axioms Iota.Source.Parsing.BlockScan.done
#print axioms Iota.Source.Parsing.BlockScan.next
#print axioms Iota.Source.Parsing.BlockScan.sound
#print axioms Iota.Source.Parsing.FunctionScan
#print axioms Iota.Source.Parsing.FunctionScan.intro
#print axioms Iota.Source.Parsing.FunctionScan.sound
#print axioms Iota.Source.Parsing.StatementAudit.nat_add_zero
