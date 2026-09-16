import Slang.Audit.Empty
import Slang.Source.Parsing.Statement

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Slang.Source.Parsing.StatementAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Source.Parsing.StatementAudit

#audit_empty Slang.Source.Parsing.ParameterScan : Nat → List Slang.Decode.Token → List String → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.ParameterScan.done : type_of% @Slang.Source.Parsing.ParameterScan.done
#audit_empty Slang.Source.Parsing.ParameterScan.comma : type_of% @Slang.Source.Parsing.ParameterScan.comma
#audit_empty Slang.Source.Parsing.ParameterScan.last : type_of% @Slang.Source.Parsing.ParameterScan.last
#audit_empty Slang.Source.Parsing.ParameterScan.sound : type_of% @Slang.Source.Parsing.ParameterScan.sound
#audit_empty Slang.Source.Parsing.ExpressionStatementToken : Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.numeral : type_of% @Slang.Source.Parsing.ExpressionStatementToken.numeral
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.lparen : type_of% @Slang.Source.Parsing.ExpressionStatementToken.lparen
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.rparen : type_of% @Slang.Source.Parsing.ExpressionStatementToken.rparen
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.lbrace : type_of% @Slang.Source.Parsing.ExpressionStatementToken.lbrace
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.rbrace : type_of% @Slang.Source.Parsing.ExpressionStatementToken.rbrace
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.lbracket : type_of% @Slang.Source.Parsing.ExpressionStatementToken.lbracket
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.rbracket : type_of% @Slang.Source.Parsing.ExpressionStatementToken.rbracket
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.comma : type_of% @Slang.Source.Parsing.ExpressionStatementToken.comma
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.semi : type_of% @Slang.Source.Parsing.ExpressionStatementToken.semi
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.newline : type_of% @Slang.Source.Parsing.ExpressionStatementToken.newline
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.assign : type_of% @Slang.Source.Parsing.ExpressionStatementToken.assign
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.lt : type_of% @Slang.Source.Parsing.ExpressionStatementToken.lt
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.le : type_of% @Slang.Source.Parsing.ExpressionStatementToken.le
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.gt : type_of% @Slang.Source.Parsing.ExpressionStatementToken.gt
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.ge : type_of% @Slang.Source.Parsing.ExpressionStatementToken.ge
#audit_empty Slang.Source.Parsing.ExpressionStatementToken.other : type_of% @Slang.Source.Parsing.ExpressionStatementToken.other
#audit_empty Slang.Source.Parsing.StmtScan : Nat → List Slang.Decode.Token → Slang.Stmt → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.StmtScan.return_ : type_of% @Slang.Source.Parsing.StmtScan.return_
#audit_empty Slang.Source.Parsing.StmtScan.ifNoElseEnd : type_of% @Slang.Source.Parsing.StmtScan.ifNoElseEnd
#audit_empty Slang.Source.Parsing.StmtScan.ifNoElseNext : type_of% @Slang.Source.Parsing.StmtScan.ifNoElseNext
#audit_empty Slang.Source.Parsing.StmtScan.ifElse : type_of% @Slang.Source.Parsing.StmtScan.ifElse
#audit_empty Slang.Source.Parsing.StmtScan.store : type_of% @Slang.Source.Parsing.StmtScan.store
#audit_empty Slang.Source.Parsing.StmtScan.memoryExpression : type_of% @Slang.Source.Parsing.StmtScan.memoryExpression
#audit_empty Slang.Source.Parsing.StmtScan.bind : type_of% @Slang.Source.Parsing.StmtScan.bind
#audit_empty Slang.Source.Parsing.StmtScan.nameExpression : type_of% @Slang.Source.Parsing.StmtScan.nameExpression
#audit_empty Slang.Source.Parsing.StmtScan.otherExpression : type_of% @Slang.Source.Parsing.StmtScan.otherExpression
#audit_empty Slang.Source.Parsing.StmtScan.sound : type_of% @Slang.Source.Parsing.StmtScan.sound
#audit_empty Slang.Source.Parsing.BlockScan : Nat → List Slang.Decode.Token → List Slang.Stmt → Slang.Decode.Token → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.BlockScan.done : type_of% @Slang.Source.Parsing.BlockScan.done
#audit_empty Slang.Source.Parsing.BlockScan.next : type_of% @Slang.Source.Parsing.BlockScan.next
#audit_empty Slang.Source.Parsing.BlockScan.sound : type_of% @Slang.Source.Parsing.BlockScan.sound
#audit_empty Slang.Source.Parsing.FunctionScan : String → Nat → List Slang.Decode.Token → Slang.FunctionDecl → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.FunctionScan.intro : type_of% @Slang.Source.Parsing.FunctionScan.intro
#audit_empty Slang.Source.Parsing.FunctionScan.sound : type_of% @Slang.Source.Parsing.FunctionScan.sound
#audit_empty Slang.Source.Parsing.StatementAudit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Slang.Source.Parsing.ParameterScan
#print axioms Slang.Source.Parsing.ParameterScan.done
#print axioms Slang.Source.Parsing.ParameterScan.comma
#print axioms Slang.Source.Parsing.ParameterScan.last
#print axioms Slang.Source.Parsing.ParameterScan.sound
#print axioms Slang.Source.Parsing.ExpressionStatementToken
#print axioms Slang.Source.Parsing.ExpressionStatementToken.numeral
#print axioms Slang.Source.Parsing.ExpressionStatementToken.lparen
#print axioms Slang.Source.Parsing.ExpressionStatementToken.rparen
#print axioms Slang.Source.Parsing.ExpressionStatementToken.lbrace
#print axioms Slang.Source.Parsing.ExpressionStatementToken.rbrace
#print axioms Slang.Source.Parsing.ExpressionStatementToken.lbracket
#print axioms Slang.Source.Parsing.ExpressionStatementToken.rbracket
#print axioms Slang.Source.Parsing.ExpressionStatementToken.comma
#print axioms Slang.Source.Parsing.ExpressionStatementToken.semi
#print axioms Slang.Source.Parsing.ExpressionStatementToken.newline
#print axioms Slang.Source.Parsing.ExpressionStatementToken.assign
#print axioms Slang.Source.Parsing.ExpressionStatementToken.lt
#print axioms Slang.Source.Parsing.ExpressionStatementToken.le
#print axioms Slang.Source.Parsing.ExpressionStatementToken.gt
#print axioms Slang.Source.Parsing.ExpressionStatementToken.ge
#print axioms Slang.Source.Parsing.ExpressionStatementToken.other
#print axioms Slang.Source.Parsing.StmtScan
#print axioms Slang.Source.Parsing.StmtScan.return_
#print axioms Slang.Source.Parsing.StmtScan.ifNoElseEnd
#print axioms Slang.Source.Parsing.StmtScan.ifNoElseNext
#print axioms Slang.Source.Parsing.StmtScan.ifElse
#print axioms Slang.Source.Parsing.StmtScan.store
#print axioms Slang.Source.Parsing.StmtScan.memoryExpression
#print axioms Slang.Source.Parsing.StmtScan.bind
#print axioms Slang.Source.Parsing.StmtScan.nameExpression
#print axioms Slang.Source.Parsing.StmtScan.otherExpression
#print axioms Slang.Source.Parsing.StmtScan.sound
#print axioms Slang.Source.Parsing.BlockScan
#print axioms Slang.Source.Parsing.BlockScan.done
#print axioms Slang.Source.Parsing.BlockScan.next
#print axioms Slang.Source.Parsing.BlockScan.sound
#print axioms Slang.Source.Parsing.FunctionScan
#print axioms Slang.Source.Parsing.FunctionScan.intro
#print axioms Slang.Source.Parsing.FunctionScan.sound
#print axioms Slang.Source.Parsing.StatementAudit.nat_add_zero
