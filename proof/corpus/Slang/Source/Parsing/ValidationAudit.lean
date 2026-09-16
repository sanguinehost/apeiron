import Slang.Audit.Empty
import Slang.Source.Parsing.Validation

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Slang.Source.Parsing.ValidationAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Source.Parsing.ValidationAudit

#audit_empty Slang.Source.Parsing.NonDelimiterToken : Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.NonDelimiterToken.ident : type_of% @Slang.Source.Parsing.NonDelimiterToken.ident
#audit_empty Slang.Source.Parsing.NonDelimiterToken.numeral : type_of% @Slang.Source.Parsing.NonDelimiterToken.numeral
#audit_empty Slang.Source.Parsing.NonDelimiterToken.comma : type_of% @Slang.Source.Parsing.NonDelimiterToken.comma
#audit_empty Slang.Source.Parsing.NonDelimiterToken.semi : type_of% @Slang.Source.Parsing.NonDelimiterToken.semi
#audit_empty Slang.Source.Parsing.NonDelimiterToken.newline : type_of% @Slang.Source.Parsing.NonDelimiterToken.newline
#audit_empty Slang.Source.Parsing.NonDelimiterToken.assign : type_of% @Slang.Source.Parsing.NonDelimiterToken.assign
#audit_empty Slang.Source.Parsing.NonDelimiterToken.lt : type_of% @Slang.Source.Parsing.NonDelimiterToken.lt
#audit_empty Slang.Source.Parsing.NonDelimiterToken.le : type_of% @Slang.Source.Parsing.NonDelimiterToken.le
#audit_empty Slang.Source.Parsing.NonDelimiterToken.gt : type_of% @Slang.Source.Parsing.NonDelimiterToken.gt
#audit_empty Slang.Source.Parsing.NonDelimiterToken.ge : type_of% @Slang.Source.Parsing.NonDelimiterToken.ge
#audit_empty Slang.Source.Parsing.NonDelimiterToken.other : type_of% @Slang.Source.Parsing.NonDelimiterToken.other
#audit_empty Slang.Source.Parsing.DelimitersBalancedDerivation : List Slang.Decode.Token → List Slang.Decode.TokenKind → Prop
#audit_empty Slang.Source.Parsing.DelimitersBalancedDerivation.done : type_of% @Slang.Source.Parsing.DelimitersBalancedDerivation.done
#audit_empty Slang.Source.Parsing.DelimitersBalancedDerivation.lparen : type_of% @Slang.Source.Parsing.DelimitersBalancedDerivation.lparen
#audit_empty Slang.Source.Parsing.DelimitersBalancedDerivation.lbrace : type_of% @Slang.Source.Parsing.DelimitersBalancedDerivation.lbrace
#audit_empty Slang.Source.Parsing.DelimitersBalancedDerivation.lbracket : type_of% @Slang.Source.Parsing.DelimitersBalancedDerivation.lbracket
#audit_empty Slang.Source.Parsing.DelimitersBalancedDerivation.rparen : type_of% @Slang.Source.Parsing.DelimitersBalancedDerivation.rparen
#audit_empty Slang.Source.Parsing.DelimitersBalancedDerivation.rbrace : type_of% @Slang.Source.Parsing.DelimitersBalancedDerivation.rbrace
#audit_empty Slang.Source.Parsing.DelimitersBalancedDerivation.rbracket : type_of% @Slang.Source.Parsing.DelimitersBalancedDerivation.rbracket
#audit_empty Slang.Source.Parsing.DelimitersBalancedDerivation.other : type_of% @Slang.Source.Parsing.DelimitersBalancedDerivation.other
#audit_empty Slang.Source.Parsing.DelimitersBalancedDerivation.sound : type_of% @Slang.Source.Parsing.DelimitersBalancedDerivation.sound
#audit_empty Slang.Source.Parsing.SpansValidDerivation : Nat → List Slang.Span → Prop
#audit_empty Slang.Source.Parsing.SpansValidDerivation.intro : type_of% @Slang.Source.Parsing.SpansValidDerivation.intro
#audit_empty Slang.Source.Parsing.SpansValidDerivation.sound : type_of% @Slang.Source.Parsing.SpansValidDerivation.sound
#audit_empty Slang.Source.Parsing.DistinctDerivation : List String → Prop
#audit_empty Slang.Source.Parsing.DistinctDerivation.intro : type_of% @Slang.Source.Parsing.DistinctDerivation.intro
#audit_empty Slang.Source.Parsing.DistinctDerivation.sound : type_of% @Slang.Source.Parsing.DistinctDerivation.sound
#audit_empty Slang.Source.Parsing.ExpressionCallsResolveDerivation : List String → Slang.Expr → Prop
#audit_empty Slang.Source.Parsing.ExpressionCallsResolveDerivation.literal : type_of% @Slang.Source.Parsing.ExpressionCallsResolveDerivation.literal
#audit_empty Slang.Source.Parsing.ExpressionCallsResolveDerivation.name : type_of% @Slang.Source.Parsing.ExpressionCallsResolveDerivation.name
#audit_empty Slang.Source.Parsing.ExpressionCallsResolveDerivation.call : type_of% @Slang.Source.Parsing.ExpressionCallsResolveDerivation.call
#audit_empty Slang.Source.Parsing.ExpressionCallsResolveDerivation.memory : type_of% @Slang.Source.Parsing.ExpressionCallsResolveDerivation.memory
#audit_empty Slang.Source.Parsing.ExpressionCallsResolveDerivation.binary : type_of% @Slang.Source.Parsing.ExpressionCallsResolveDerivation.binary
#audit_empty Slang.Source.Parsing.ExpressionCallsResolveDerivation.conditional : type_of% @Slang.Source.Parsing.ExpressionCallsResolveDerivation.conditional
#audit_empty Slang.Source.Parsing.ExpressionCallsResolveDerivation.sound : type_of% @Slang.Source.Parsing.ExpressionCallsResolveDerivation.sound
#audit_empty Slang.Source.Parsing.ExpressionsCallsResolveDerivation : List String → List Slang.Expr → Prop
#audit_empty Slang.Source.Parsing.ExpressionsCallsResolveDerivation.done : type_of% @Slang.Source.Parsing.ExpressionsCallsResolveDerivation.done
#audit_empty Slang.Source.Parsing.ExpressionsCallsResolveDerivation.next : type_of% @Slang.Source.Parsing.ExpressionsCallsResolveDerivation.next
#audit_empty Slang.Source.Parsing.ExpressionsCallsResolveDerivation.sound : type_of% @Slang.Source.Parsing.ExpressionsCallsResolveDerivation.sound
#audit_empty Slang.Source.Parsing.StatementCallsResolveDerivation : List String → Slang.Stmt → Prop
#audit_empty Slang.Source.Parsing.StatementCallsResolveDerivation.bind : type_of% @Slang.Source.Parsing.StatementCallsResolveDerivation.bind
#audit_empty Slang.Source.Parsing.StatementCallsResolveDerivation.store : type_of% @Slang.Source.Parsing.StatementCallsResolveDerivation.store
#audit_empty Slang.Source.Parsing.StatementCallsResolveDerivation.return_ : type_of% @Slang.Source.Parsing.StatementCallsResolveDerivation.return_
#audit_empty Slang.Source.Parsing.StatementCallsResolveDerivation.expression : type_of% @Slang.Source.Parsing.StatementCallsResolveDerivation.expression
#audit_empty Slang.Source.Parsing.StatementCallsResolveDerivation.ifNone : type_of% @Slang.Source.Parsing.StatementCallsResolveDerivation.ifNone
#audit_empty Slang.Source.Parsing.StatementCallsResolveDerivation.ifSome : type_of% @Slang.Source.Parsing.StatementCallsResolveDerivation.ifSome
#audit_empty Slang.Source.Parsing.StatementCallsResolveDerivation.sound : type_of% @Slang.Source.Parsing.StatementCallsResolveDerivation.sound
#audit_empty Slang.Source.Parsing.StatementsCallsResolveDerivation : List String → List Slang.Stmt → Prop
#audit_empty Slang.Source.Parsing.StatementsCallsResolveDerivation.done : type_of% @Slang.Source.Parsing.StatementsCallsResolveDerivation.done
#audit_empty Slang.Source.Parsing.StatementsCallsResolveDerivation.next : type_of% @Slang.Source.Parsing.StatementsCallsResolveDerivation.next
#audit_empty Slang.Source.Parsing.StatementsCallsResolveDerivation.sound : type_of% @Slang.Source.Parsing.StatementsCallsResolveDerivation.sound
#audit_empty Slang.Source.Parsing.FunctionCallsResolveDerivation : List String → List Slang.FunctionDecl → Prop
#audit_empty Slang.Source.Parsing.FunctionCallsResolveDerivation.done : type_of% @Slang.Source.Parsing.FunctionCallsResolveDerivation.done
#audit_empty Slang.Source.Parsing.FunctionCallsResolveDerivation.next : type_of% @Slang.Source.Parsing.FunctionCallsResolveDerivation.next
#audit_empty Slang.Source.Parsing.FunctionCallsResolveDerivation.sound : type_of% @Slang.Source.Parsing.FunctionCallsResolveDerivation.sound
#audit_empty Slang.Source.Parsing.CallsResolveDerivation : List Slang.FunctionDecl → Prop
#audit_empty Slang.Source.Parsing.CallsResolveDerivation.intro : type_of% @Slang.Source.Parsing.CallsResolveDerivation.intro
#audit_empty Slang.Source.Parsing.CallsResolveDerivation.sound : type_of% @Slang.Source.Parsing.CallsResolveDerivation.sound
#audit_empty Slang.Source.Parsing.ExpressionNamesResolveDerivation : List String → Slang.Expr → Prop
#audit_empty Slang.Source.Parsing.ExpressionNamesResolveDerivation.literal : type_of% @Slang.Source.Parsing.ExpressionNamesResolveDerivation.literal
#audit_empty Slang.Source.Parsing.ExpressionNamesResolveDerivation.name : type_of% @Slang.Source.Parsing.ExpressionNamesResolveDerivation.name
#audit_empty Slang.Source.Parsing.ExpressionNamesResolveDerivation.call : type_of% @Slang.Source.Parsing.ExpressionNamesResolveDerivation.call
#audit_empty Slang.Source.Parsing.ExpressionNamesResolveDerivation.memory : type_of% @Slang.Source.Parsing.ExpressionNamesResolveDerivation.memory
#audit_empty Slang.Source.Parsing.ExpressionNamesResolveDerivation.binary : type_of% @Slang.Source.Parsing.ExpressionNamesResolveDerivation.binary
#audit_empty Slang.Source.Parsing.ExpressionNamesResolveDerivation.conditional : type_of% @Slang.Source.Parsing.ExpressionNamesResolveDerivation.conditional
#audit_empty Slang.Source.Parsing.ExpressionNamesResolveDerivation.sound : type_of% @Slang.Source.Parsing.ExpressionNamesResolveDerivation.sound
#audit_empty Slang.Source.Parsing.ExpressionsNamesResolveDerivation : List String → List Slang.Expr → Prop
#audit_empty Slang.Source.Parsing.ExpressionsNamesResolveDerivation.done : type_of% @Slang.Source.Parsing.ExpressionsNamesResolveDerivation.done
#audit_empty Slang.Source.Parsing.ExpressionsNamesResolveDerivation.next : type_of% @Slang.Source.Parsing.ExpressionsNamesResolveDerivation.next
#audit_empty Slang.Source.Parsing.ExpressionsNamesResolveDerivation.sound : type_of% @Slang.Source.Parsing.ExpressionsNamesResolveDerivation.sound
#audit_empty Slang.Source.Parsing.StatementsNamesResolveDerivation : Nat → List String → List Slang.Stmt → Prop
#audit_empty Slang.Source.Parsing.StatementsNamesResolveDerivation.done : type_of% @Slang.Source.Parsing.StatementsNamesResolveDerivation.done
#audit_empty Slang.Source.Parsing.StatementsNamesResolveDerivation.bind : type_of% @Slang.Source.Parsing.StatementsNamesResolveDerivation.bind
#audit_empty Slang.Source.Parsing.StatementsNamesResolveDerivation.store : type_of% @Slang.Source.Parsing.StatementsNamesResolveDerivation.store
#audit_empty Slang.Source.Parsing.StatementsNamesResolveDerivation.return_ : type_of% @Slang.Source.Parsing.StatementsNamesResolveDerivation.return_
#audit_empty Slang.Source.Parsing.StatementsNamesResolveDerivation.expression : type_of% @Slang.Source.Parsing.StatementsNamesResolveDerivation.expression
#audit_empty Slang.Source.Parsing.StatementsNamesResolveDerivation.ifNone : type_of% @Slang.Source.Parsing.StatementsNamesResolveDerivation.ifNone
#audit_empty Slang.Source.Parsing.StatementsNamesResolveDerivation.ifSome : type_of% @Slang.Source.Parsing.StatementsNamesResolveDerivation.ifSome
#audit_empty Slang.Source.Parsing.StatementsNamesResolveDerivation.sound : type_of% @Slang.Source.Parsing.StatementsNamesResolveDerivation.sound
#audit_empty Slang.Source.Parsing.FunctionNamesResolveDerivation : List String → List Slang.FunctionDecl → Prop
#audit_empty Slang.Source.Parsing.FunctionNamesResolveDerivation.done : type_of% @Slang.Source.Parsing.FunctionNamesResolveDerivation.done
#audit_empty Slang.Source.Parsing.FunctionNamesResolveDerivation.next : type_of% @Slang.Source.Parsing.FunctionNamesResolveDerivation.next
#audit_empty Slang.Source.Parsing.FunctionNamesResolveDerivation.sound : type_of% @Slang.Source.Parsing.FunctionNamesResolveDerivation.sound
#audit_empty Slang.Source.Parsing.NamesResolveDerivation : List Slang.ConstantDecl → List Slang.FunctionDecl → Prop
#audit_empty Slang.Source.Parsing.NamesResolveDerivation.intro : type_of% @Slang.Source.Parsing.NamesResolveDerivation.intro
#audit_empty Slang.Source.Parsing.NamesResolveDerivation.sound : type_of% @Slang.Source.Parsing.NamesResolveDerivation.sound
#audit_empty Slang.Source.Parsing.ValidationAudit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Slang.Source.Parsing.NonDelimiterToken
#print axioms Slang.Source.Parsing.NonDelimiterToken.ident
#print axioms Slang.Source.Parsing.NonDelimiterToken.numeral
#print axioms Slang.Source.Parsing.NonDelimiterToken.comma
#print axioms Slang.Source.Parsing.NonDelimiterToken.semi
#print axioms Slang.Source.Parsing.NonDelimiterToken.newline
#print axioms Slang.Source.Parsing.NonDelimiterToken.assign
#print axioms Slang.Source.Parsing.NonDelimiterToken.lt
#print axioms Slang.Source.Parsing.NonDelimiterToken.le
#print axioms Slang.Source.Parsing.NonDelimiterToken.gt
#print axioms Slang.Source.Parsing.NonDelimiterToken.ge
#print axioms Slang.Source.Parsing.NonDelimiterToken.other
#print axioms Slang.Source.Parsing.DelimitersBalancedDerivation
#print axioms Slang.Source.Parsing.DelimitersBalancedDerivation.done
#print axioms Slang.Source.Parsing.DelimitersBalancedDerivation.lparen
#print axioms Slang.Source.Parsing.DelimitersBalancedDerivation.lbrace
#print axioms Slang.Source.Parsing.DelimitersBalancedDerivation.lbracket
#print axioms Slang.Source.Parsing.DelimitersBalancedDerivation.rparen
#print axioms Slang.Source.Parsing.DelimitersBalancedDerivation.rbrace
#print axioms Slang.Source.Parsing.DelimitersBalancedDerivation.rbracket
#print axioms Slang.Source.Parsing.DelimitersBalancedDerivation.other
#print axioms Slang.Source.Parsing.DelimitersBalancedDerivation.sound
#print axioms Slang.Source.Parsing.SpansValidDerivation
#print axioms Slang.Source.Parsing.SpansValidDerivation.intro
#print axioms Slang.Source.Parsing.SpansValidDerivation.sound
#print axioms Slang.Source.Parsing.DistinctDerivation
#print axioms Slang.Source.Parsing.DistinctDerivation.intro
#print axioms Slang.Source.Parsing.DistinctDerivation.sound
#print axioms Slang.Source.Parsing.ExpressionCallsResolveDerivation
#print axioms Slang.Source.Parsing.ExpressionCallsResolveDerivation.literal
#print axioms Slang.Source.Parsing.ExpressionCallsResolveDerivation.name
#print axioms Slang.Source.Parsing.ExpressionCallsResolveDerivation.call
#print axioms Slang.Source.Parsing.ExpressionCallsResolveDerivation.memory
#print axioms Slang.Source.Parsing.ExpressionCallsResolveDerivation.binary
#print axioms Slang.Source.Parsing.ExpressionCallsResolveDerivation.conditional
#print axioms Slang.Source.Parsing.ExpressionCallsResolveDerivation.sound
#print axioms Slang.Source.Parsing.ExpressionsCallsResolveDerivation
#print axioms Slang.Source.Parsing.ExpressionsCallsResolveDerivation.done
#print axioms Slang.Source.Parsing.ExpressionsCallsResolveDerivation.next
#print axioms Slang.Source.Parsing.ExpressionsCallsResolveDerivation.sound
#print axioms Slang.Source.Parsing.StatementCallsResolveDerivation
#print axioms Slang.Source.Parsing.StatementCallsResolveDerivation.bind
#print axioms Slang.Source.Parsing.StatementCallsResolveDerivation.store
#print axioms Slang.Source.Parsing.StatementCallsResolveDerivation.return_
#print axioms Slang.Source.Parsing.StatementCallsResolveDerivation.expression
#print axioms Slang.Source.Parsing.StatementCallsResolveDerivation.ifNone
#print axioms Slang.Source.Parsing.StatementCallsResolveDerivation.ifSome
#print axioms Slang.Source.Parsing.StatementCallsResolveDerivation.sound
#print axioms Slang.Source.Parsing.StatementsCallsResolveDerivation
#print axioms Slang.Source.Parsing.StatementsCallsResolveDerivation.done
#print axioms Slang.Source.Parsing.StatementsCallsResolveDerivation.next
#print axioms Slang.Source.Parsing.StatementsCallsResolveDerivation.sound
#print axioms Slang.Source.Parsing.FunctionCallsResolveDerivation
#print axioms Slang.Source.Parsing.FunctionCallsResolveDerivation.done
#print axioms Slang.Source.Parsing.FunctionCallsResolveDerivation.next
#print axioms Slang.Source.Parsing.FunctionCallsResolveDerivation.sound
#print axioms Slang.Source.Parsing.CallsResolveDerivation
#print axioms Slang.Source.Parsing.CallsResolveDerivation.intro
#print axioms Slang.Source.Parsing.CallsResolveDerivation.sound
#print axioms Slang.Source.Parsing.ExpressionNamesResolveDerivation
#print axioms Slang.Source.Parsing.ExpressionNamesResolveDerivation.literal
#print axioms Slang.Source.Parsing.ExpressionNamesResolveDerivation.name
#print axioms Slang.Source.Parsing.ExpressionNamesResolveDerivation.call
#print axioms Slang.Source.Parsing.ExpressionNamesResolveDerivation.memory
#print axioms Slang.Source.Parsing.ExpressionNamesResolveDerivation.binary
#print axioms Slang.Source.Parsing.ExpressionNamesResolveDerivation.conditional
#print axioms Slang.Source.Parsing.ExpressionNamesResolveDerivation.sound
#print axioms Slang.Source.Parsing.ExpressionsNamesResolveDerivation
#print axioms Slang.Source.Parsing.ExpressionsNamesResolveDerivation.done
#print axioms Slang.Source.Parsing.ExpressionsNamesResolveDerivation.next
#print axioms Slang.Source.Parsing.ExpressionsNamesResolveDerivation.sound
#print axioms Slang.Source.Parsing.StatementsNamesResolveDerivation
#print axioms Slang.Source.Parsing.StatementsNamesResolveDerivation.done
#print axioms Slang.Source.Parsing.StatementsNamesResolveDerivation.bind
#print axioms Slang.Source.Parsing.StatementsNamesResolveDerivation.store
#print axioms Slang.Source.Parsing.StatementsNamesResolveDerivation.return_
#print axioms Slang.Source.Parsing.StatementsNamesResolveDerivation.expression
#print axioms Slang.Source.Parsing.StatementsNamesResolveDerivation.ifNone
#print axioms Slang.Source.Parsing.StatementsNamesResolveDerivation.ifSome
#print axioms Slang.Source.Parsing.StatementsNamesResolveDerivation.sound
#print axioms Slang.Source.Parsing.FunctionNamesResolveDerivation
#print axioms Slang.Source.Parsing.FunctionNamesResolveDerivation.done
#print axioms Slang.Source.Parsing.FunctionNamesResolveDerivation.next
#print axioms Slang.Source.Parsing.FunctionNamesResolveDerivation.sound
#print axioms Slang.Source.Parsing.NamesResolveDerivation
#print axioms Slang.Source.Parsing.NamesResolveDerivation.intro
#print axioms Slang.Source.Parsing.NamesResolveDerivation.sound
#print axioms Slang.Source.Parsing.ValidationAudit.nat_add_zero
#audit_empty Slang.Source.Parsing.DelimitersBalancedDerivation.continue : type_of% @Slang.Source.Parsing.DelimitersBalancedDerivation.continue
#audit_empty Slang.Source.Parsing.SpansValidDerivation.all_append : type_of% @Slang.Source.Parsing.SpansValidDerivation.all_append
#audit_empty Slang.Source.Parsing.SpansValidDerivation.all_head : type_of% @Slang.Source.Parsing.SpansValidDerivation.all_head
#audit_empty Slang.Source.Parsing.SpansValidDerivation.all_tail : type_of% @Slang.Source.Parsing.SpansValidDerivation.all_tail
#audit_empty Slang.Source.Parsing.SpansValidDerivation.all_inner_append : type_of% @Slang.Source.Parsing.SpansValidDerivation.all_inner_append
#audit_empty Slang.Source.Parsing.SpansValidDerivation.append : type_of% @Slang.Source.Parsing.SpansValidDerivation.append
#audit_empty Slang.Source.Parsing.DistinctDerivation.reverseAux_length : type_of% @Slang.Source.Parsing.DistinctDerivation.reverseAux_length
#audit_empty Slang.Source.Parsing.DistinctDerivation.any_beq_false_of_not_mem : type_of% @Slang.Source.Parsing.DistinctDerivation.any_beq_false_of_not_mem
#audit_empty Slang.Source.Parsing.DistinctDerivation.eraseDupsLoop_length_le : type_of% @Slang.Source.Parsing.DistinctDerivation.eraseDupsLoop_length_le
#audit_empty Slang.Source.Parsing.DistinctDerivation.eraseDupsLoop_length_of_nodup : type_of% @Slang.Source.Parsing.DistinctDerivation.eraseDupsLoop_length_of_nodup
#audit_empty Slang.Source.Parsing.DistinctDerivation.of_nodup : type_of% @Slang.Source.Parsing.DistinctDerivation.of_nodup
#audit_empty Slang.Source.Parsing.DistinctDerivation.mem_append_cases : type_of% @Slang.Source.Parsing.DistinctDerivation.mem_append_cases
#audit_empty Slang.Source.Parsing.DistinctDerivation.nodup_append_of_cross : type_of% @Slang.Source.Parsing.DistinctDerivation.nodup_append_of_cross
#audit_empty Slang.Source.Parsing.DistinctDerivation.append : type_of% @Slang.Source.Parsing.DistinctDerivation.append
#audit_empty Slang.Source.Parsing.ExpressionsCallsResolveDerivation.append : type_of% @Slang.Source.Parsing.ExpressionsCallsResolveDerivation.append
#audit_empty Slang.Source.Parsing.StatementsCallsResolveDerivation.append : type_of% @Slang.Source.Parsing.StatementsCallsResolveDerivation.append
#audit_empty Slang.Source.Parsing.FunctionCallsResolveDerivation.append : type_of% @Slang.Source.Parsing.FunctionCallsResolveDerivation.append
#audit_empty Slang.Source.Parsing.CallsResolveDerivation.of_parts : type_of% @Slang.Source.Parsing.CallsResolveDerivation.of_parts
#audit_empty Slang.Source.Parsing.ExpressionsNamesResolveDerivation.append : type_of% @Slang.Source.Parsing.ExpressionsNamesResolveDerivation.append
#audit_empty Slang.Source.Parsing.StatementsNamesResolveDerivation.continue : type_of% @Slang.Source.Parsing.StatementsNamesResolveDerivation.continue
#audit_empty Slang.Source.Parsing.FunctionNamesResolveDerivation.append : type_of% @Slang.Source.Parsing.FunctionNamesResolveDerivation.append
#audit_empty Slang.Source.Parsing.NamesResolveDerivation.append : type_of% @Slang.Source.Parsing.NamesResolveDerivation.append
#print axioms Slang.Source.Parsing.DelimitersBalancedDerivation.continue
#print axioms Slang.Source.Parsing.SpansValidDerivation.all_append
#print axioms Slang.Source.Parsing.SpansValidDerivation.all_head
#print axioms Slang.Source.Parsing.SpansValidDerivation.all_tail
#print axioms Slang.Source.Parsing.SpansValidDerivation.all_inner_append
#print axioms Slang.Source.Parsing.SpansValidDerivation.append
#print axioms Slang.Source.Parsing.DistinctDerivation.reverseAux_length
#print axioms Slang.Source.Parsing.DistinctDerivation.any_beq_false_of_not_mem
#print axioms Slang.Source.Parsing.DistinctDerivation.eraseDupsLoop_length_le
#print axioms Slang.Source.Parsing.DistinctDerivation.eraseDupsLoop_length_of_nodup
#print axioms Slang.Source.Parsing.DistinctDerivation.of_nodup
#print axioms Slang.Source.Parsing.DistinctDerivation.mem_append_cases
#print axioms Slang.Source.Parsing.DistinctDerivation.nodup_append_of_cross
#print axioms Slang.Source.Parsing.DistinctDerivation.append
#print axioms Slang.Source.Parsing.ExpressionsCallsResolveDerivation.append
#print axioms Slang.Source.Parsing.StatementsCallsResolveDerivation.append
#print axioms Slang.Source.Parsing.FunctionCallsResolveDerivation.append
#print axioms Slang.Source.Parsing.CallsResolveDerivation.of_parts
#print axioms Slang.Source.Parsing.ExpressionsNamesResolveDerivation.append
#print axioms Slang.Source.Parsing.StatementsNamesResolveDerivation.continue
#print axioms Slang.Source.Parsing.FunctionNamesResolveDerivation.append
#print axioms Slang.Source.Parsing.NamesResolveDerivation.append

#audit_empty Slang.Source.Parsing.DelimitersBalancedDerivation.append : type_of% @Slang.Source.Parsing.DelimitersBalancedDerivation.append
#audit_empty Slang.Source.Parsing.DistinctDerivation.of_bool : type_of% @Slang.Source.Parsing.DistinctDerivation.of_bool
#audit_empty Slang.Source.Parsing.ExpressionCallsResolveDerivation.of_bool : type_of% @Slang.Source.Parsing.ExpressionCallsResolveDerivation.of_bool
#audit_empty Slang.Source.Parsing.ExpressionsCallsResolveDerivation.of_bool : type_of% @Slang.Source.Parsing.ExpressionsCallsResolveDerivation.of_bool
#audit_empty Slang.Source.Parsing.StatementCallsResolveDerivation.of_bool : type_of% @Slang.Source.Parsing.StatementCallsResolveDerivation.of_bool
#audit_empty Slang.Source.Parsing.StatementsCallsResolveDerivation.of_bool : type_of% @Slang.Source.Parsing.StatementsCallsResolveDerivation.of_bool
#audit_empty Slang.Source.Parsing.FunctionCallsResolveDerivation.of_bool : type_of% @Slang.Source.Parsing.FunctionCallsResolveDerivation.of_bool
#audit_empty Slang.Source.Parsing.CallsResolveDerivation.of_bool : type_of% @Slang.Source.Parsing.CallsResolveDerivation.of_bool
#audit_empty Slang.Source.Parsing.ExpressionNamesResolveDerivation.of_bool : type_of% @Slang.Source.Parsing.ExpressionNamesResolveDerivation.of_bool
#audit_empty Slang.Source.Parsing.ExpressionsNamesResolveDerivation.of_bool : type_of% @Slang.Source.Parsing.ExpressionsNamesResolveDerivation.of_bool
#audit_empty Slang.Source.Parsing.StatementsNamesResolveDerivation.of_bool : type_of% @Slang.Source.Parsing.StatementsNamesResolveDerivation.of_bool
#audit_empty Slang.Source.Parsing.FunctionNamesResolveDerivation.of_bool : type_of% @Slang.Source.Parsing.FunctionNamesResolveDerivation.of_bool
#audit_empty Slang.Source.Parsing.NamesResolveDerivation.of_bool : type_of% @Slang.Source.Parsing.NamesResolveDerivation.of_bool
