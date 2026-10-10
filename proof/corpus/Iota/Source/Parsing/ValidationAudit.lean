import Iota.Audit.Empty
import Iota.Source.Parsing.Validation

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Iota.Source.Parsing.ValidationAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Source.Parsing.ValidationAudit

#audit_empty Iota.Source.Parsing.NonDelimiterToken : Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.NonDelimiterToken.ident : type_of% @Iota.Source.Parsing.NonDelimiterToken.ident
#audit_empty Iota.Source.Parsing.NonDelimiterToken.numeral : type_of% @Iota.Source.Parsing.NonDelimiterToken.numeral
#audit_empty Iota.Source.Parsing.NonDelimiterToken.comma : type_of% @Iota.Source.Parsing.NonDelimiterToken.comma
#audit_empty Iota.Source.Parsing.NonDelimiterToken.semi : type_of% @Iota.Source.Parsing.NonDelimiterToken.semi
#audit_empty Iota.Source.Parsing.NonDelimiterToken.newline : type_of% @Iota.Source.Parsing.NonDelimiterToken.newline
#audit_empty Iota.Source.Parsing.NonDelimiterToken.assign : type_of% @Iota.Source.Parsing.NonDelimiterToken.assign
#audit_empty Iota.Source.Parsing.NonDelimiterToken.lt : type_of% @Iota.Source.Parsing.NonDelimiterToken.lt
#audit_empty Iota.Source.Parsing.NonDelimiterToken.le : type_of% @Iota.Source.Parsing.NonDelimiterToken.le
#audit_empty Iota.Source.Parsing.NonDelimiterToken.gt : type_of% @Iota.Source.Parsing.NonDelimiterToken.gt
#audit_empty Iota.Source.Parsing.NonDelimiterToken.ge : type_of% @Iota.Source.Parsing.NonDelimiterToken.ge
#audit_empty Iota.Source.Parsing.NonDelimiterToken.other : type_of% @Iota.Source.Parsing.NonDelimiterToken.other
#audit_empty Iota.Source.Parsing.DelimitersBalancedDerivation : List Iota.Decode.Token → List Iota.Decode.TokenKind → Prop
#audit_empty Iota.Source.Parsing.DelimitersBalancedDerivation.done : type_of% @Iota.Source.Parsing.DelimitersBalancedDerivation.done
#audit_empty Iota.Source.Parsing.DelimitersBalancedDerivation.lparen : type_of% @Iota.Source.Parsing.DelimitersBalancedDerivation.lparen
#audit_empty Iota.Source.Parsing.DelimitersBalancedDerivation.lbrace : type_of% @Iota.Source.Parsing.DelimitersBalancedDerivation.lbrace
#audit_empty Iota.Source.Parsing.DelimitersBalancedDerivation.lbracket : type_of% @Iota.Source.Parsing.DelimitersBalancedDerivation.lbracket
#audit_empty Iota.Source.Parsing.DelimitersBalancedDerivation.rparen : type_of% @Iota.Source.Parsing.DelimitersBalancedDerivation.rparen
#audit_empty Iota.Source.Parsing.DelimitersBalancedDerivation.rbrace : type_of% @Iota.Source.Parsing.DelimitersBalancedDerivation.rbrace
#audit_empty Iota.Source.Parsing.DelimitersBalancedDerivation.rbracket : type_of% @Iota.Source.Parsing.DelimitersBalancedDerivation.rbracket
#audit_empty Iota.Source.Parsing.DelimitersBalancedDerivation.other : type_of% @Iota.Source.Parsing.DelimitersBalancedDerivation.other
#audit_empty Iota.Source.Parsing.DelimitersBalancedDerivation.sound : type_of% @Iota.Source.Parsing.DelimitersBalancedDerivation.sound
#audit_empty Iota.Source.Parsing.SpansValidDerivation : Nat → List Iota.Span → Prop
#audit_empty Iota.Source.Parsing.SpansValidDerivation.intro : type_of% @Iota.Source.Parsing.SpansValidDerivation.intro
#audit_empty Iota.Source.Parsing.SpansValidDerivation.sound : type_of% @Iota.Source.Parsing.SpansValidDerivation.sound
#audit_empty Iota.Source.Parsing.DistinctDerivation : List String → Prop
#audit_empty Iota.Source.Parsing.DistinctDerivation.intro : type_of% @Iota.Source.Parsing.DistinctDerivation.intro
#audit_empty Iota.Source.Parsing.DistinctDerivation.sound : type_of% @Iota.Source.Parsing.DistinctDerivation.sound
#audit_empty Iota.Source.Parsing.ExpressionCallsResolveDerivation : List String → Iota.Expr → Prop
#audit_empty Iota.Source.Parsing.ExpressionCallsResolveDerivation.literal : type_of% @Iota.Source.Parsing.ExpressionCallsResolveDerivation.literal
#audit_empty Iota.Source.Parsing.ExpressionCallsResolveDerivation.name : type_of% @Iota.Source.Parsing.ExpressionCallsResolveDerivation.name
#audit_empty Iota.Source.Parsing.ExpressionCallsResolveDerivation.call : type_of% @Iota.Source.Parsing.ExpressionCallsResolveDerivation.call
#audit_empty Iota.Source.Parsing.ExpressionCallsResolveDerivation.memory : type_of% @Iota.Source.Parsing.ExpressionCallsResolveDerivation.memory
#audit_empty Iota.Source.Parsing.ExpressionCallsResolveDerivation.binary : type_of% @Iota.Source.Parsing.ExpressionCallsResolveDerivation.binary
#audit_empty Iota.Source.Parsing.ExpressionCallsResolveDerivation.conditional : type_of% @Iota.Source.Parsing.ExpressionCallsResolveDerivation.conditional
#audit_empty Iota.Source.Parsing.ExpressionCallsResolveDerivation.sound : type_of% @Iota.Source.Parsing.ExpressionCallsResolveDerivation.sound
#audit_empty Iota.Source.Parsing.ExpressionsCallsResolveDerivation : List String → List Iota.Expr → Prop
#audit_empty Iota.Source.Parsing.ExpressionsCallsResolveDerivation.done : type_of% @Iota.Source.Parsing.ExpressionsCallsResolveDerivation.done
#audit_empty Iota.Source.Parsing.ExpressionsCallsResolveDerivation.next : type_of% @Iota.Source.Parsing.ExpressionsCallsResolveDerivation.next
#audit_empty Iota.Source.Parsing.ExpressionsCallsResolveDerivation.sound : type_of% @Iota.Source.Parsing.ExpressionsCallsResolveDerivation.sound
#audit_empty Iota.Source.Parsing.StatementCallsResolveDerivation : List String → Iota.Stmt → Prop
#audit_empty Iota.Source.Parsing.StatementCallsResolveDerivation.bind : type_of% @Iota.Source.Parsing.StatementCallsResolveDerivation.bind
#audit_empty Iota.Source.Parsing.StatementCallsResolveDerivation.store : type_of% @Iota.Source.Parsing.StatementCallsResolveDerivation.store
#audit_empty Iota.Source.Parsing.StatementCallsResolveDerivation.return_ : type_of% @Iota.Source.Parsing.StatementCallsResolveDerivation.return_
#audit_empty Iota.Source.Parsing.StatementCallsResolveDerivation.expression : type_of% @Iota.Source.Parsing.StatementCallsResolveDerivation.expression
#audit_empty Iota.Source.Parsing.StatementCallsResolveDerivation.ifNone : type_of% @Iota.Source.Parsing.StatementCallsResolveDerivation.ifNone
#audit_empty Iota.Source.Parsing.StatementCallsResolveDerivation.ifSome : type_of% @Iota.Source.Parsing.StatementCallsResolveDerivation.ifSome
#audit_empty Iota.Source.Parsing.StatementCallsResolveDerivation.sound : type_of% @Iota.Source.Parsing.StatementCallsResolveDerivation.sound
#audit_empty Iota.Source.Parsing.StatementsCallsResolveDerivation : List String → List Iota.Stmt → Prop
#audit_empty Iota.Source.Parsing.StatementsCallsResolveDerivation.done : type_of% @Iota.Source.Parsing.StatementsCallsResolveDerivation.done
#audit_empty Iota.Source.Parsing.StatementsCallsResolveDerivation.next : type_of% @Iota.Source.Parsing.StatementsCallsResolveDerivation.next
#audit_empty Iota.Source.Parsing.StatementsCallsResolveDerivation.sound : type_of% @Iota.Source.Parsing.StatementsCallsResolveDerivation.sound
#audit_empty Iota.Source.Parsing.FunctionCallsResolveDerivation : List String → List Iota.FunctionDecl → Prop
#audit_empty Iota.Source.Parsing.FunctionCallsResolveDerivation.done : type_of% @Iota.Source.Parsing.FunctionCallsResolveDerivation.done
#audit_empty Iota.Source.Parsing.FunctionCallsResolveDerivation.next : type_of% @Iota.Source.Parsing.FunctionCallsResolveDerivation.next
#audit_empty Iota.Source.Parsing.FunctionCallsResolveDerivation.sound : type_of% @Iota.Source.Parsing.FunctionCallsResolveDerivation.sound
#audit_empty Iota.Source.Parsing.CallsResolveDerivation : List Iota.FunctionDecl → Prop
#audit_empty Iota.Source.Parsing.CallsResolveDerivation.intro : type_of% @Iota.Source.Parsing.CallsResolveDerivation.intro
#audit_empty Iota.Source.Parsing.CallsResolveDerivation.sound : type_of% @Iota.Source.Parsing.CallsResolveDerivation.sound
#audit_empty Iota.Source.Parsing.ExpressionNamesResolveDerivation : List String → Iota.Expr → Prop
#audit_empty Iota.Source.Parsing.ExpressionNamesResolveDerivation.literal : type_of% @Iota.Source.Parsing.ExpressionNamesResolveDerivation.literal
#audit_empty Iota.Source.Parsing.ExpressionNamesResolveDerivation.name : type_of% @Iota.Source.Parsing.ExpressionNamesResolveDerivation.name
#audit_empty Iota.Source.Parsing.ExpressionNamesResolveDerivation.call : type_of% @Iota.Source.Parsing.ExpressionNamesResolveDerivation.call
#audit_empty Iota.Source.Parsing.ExpressionNamesResolveDerivation.memory : type_of% @Iota.Source.Parsing.ExpressionNamesResolveDerivation.memory
#audit_empty Iota.Source.Parsing.ExpressionNamesResolveDerivation.binary : type_of% @Iota.Source.Parsing.ExpressionNamesResolveDerivation.binary
#audit_empty Iota.Source.Parsing.ExpressionNamesResolveDerivation.conditional : type_of% @Iota.Source.Parsing.ExpressionNamesResolveDerivation.conditional
#audit_empty Iota.Source.Parsing.ExpressionNamesResolveDerivation.sound : type_of% @Iota.Source.Parsing.ExpressionNamesResolveDerivation.sound
#audit_empty Iota.Source.Parsing.ExpressionsNamesResolveDerivation : List String → List Iota.Expr → Prop
#audit_empty Iota.Source.Parsing.ExpressionsNamesResolveDerivation.done : type_of% @Iota.Source.Parsing.ExpressionsNamesResolveDerivation.done
#audit_empty Iota.Source.Parsing.ExpressionsNamesResolveDerivation.next : type_of% @Iota.Source.Parsing.ExpressionsNamesResolveDerivation.next
#audit_empty Iota.Source.Parsing.ExpressionsNamesResolveDerivation.sound : type_of% @Iota.Source.Parsing.ExpressionsNamesResolveDerivation.sound
#audit_empty Iota.Source.Parsing.StatementsNamesResolveDerivation : Nat → List String → List Iota.Stmt → Prop
#audit_empty Iota.Source.Parsing.StatementsNamesResolveDerivation.done : type_of% @Iota.Source.Parsing.StatementsNamesResolveDerivation.done
#audit_empty Iota.Source.Parsing.StatementsNamesResolveDerivation.bind : type_of% @Iota.Source.Parsing.StatementsNamesResolveDerivation.bind
#audit_empty Iota.Source.Parsing.StatementsNamesResolveDerivation.store : type_of% @Iota.Source.Parsing.StatementsNamesResolveDerivation.store
#audit_empty Iota.Source.Parsing.StatementsNamesResolveDerivation.return_ : type_of% @Iota.Source.Parsing.StatementsNamesResolveDerivation.return_
#audit_empty Iota.Source.Parsing.StatementsNamesResolveDerivation.expression : type_of% @Iota.Source.Parsing.StatementsNamesResolveDerivation.expression
#audit_empty Iota.Source.Parsing.StatementsNamesResolveDerivation.ifNone : type_of% @Iota.Source.Parsing.StatementsNamesResolveDerivation.ifNone
#audit_empty Iota.Source.Parsing.StatementsNamesResolveDerivation.ifSome : type_of% @Iota.Source.Parsing.StatementsNamesResolveDerivation.ifSome
#audit_empty Iota.Source.Parsing.StatementsNamesResolveDerivation.sound : type_of% @Iota.Source.Parsing.StatementsNamesResolveDerivation.sound
#audit_empty Iota.Source.Parsing.FunctionNamesResolveDerivation : List String → List Iota.FunctionDecl → Prop
#audit_empty Iota.Source.Parsing.FunctionNamesResolveDerivation.done : type_of% @Iota.Source.Parsing.FunctionNamesResolveDerivation.done
#audit_empty Iota.Source.Parsing.FunctionNamesResolveDerivation.next : type_of% @Iota.Source.Parsing.FunctionNamesResolveDerivation.next
#audit_empty Iota.Source.Parsing.FunctionNamesResolveDerivation.sound : type_of% @Iota.Source.Parsing.FunctionNamesResolveDerivation.sound
#audit_empty Iota.Source.Parsing.NamesResolveDerivation : List Iota.ConstantDecl → List Iota.FunctionDecl → Prop
#audit_empty Iota.Source.Parsing.NamesResolveDerivation.intro : type_of% @Iota.Source.Parsing.NamesResolveDerivation.intro
#audit_empty Iota.Source.Parsing.NamesResolveDerivation.sound : type_of% @Iota.Source.Parsing.NamesResolveDerivation.sound
#audit_empty Iota.Source.Parsing.ValidationAudit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Iota.Source.Parsing.NonDelimiterToken
#print axioms Iota.Source.Parsing.NonDelimiterToken.ident
#print axioms Iota.Source.Parsing.NonDelimiterToken.numeral
#print axioms Iota.Source.Parsing.NonDelimiterToken.comma
#print axioms Iota.Source.Parsing.NonDelimiterToken.semi
#print axioms Iota.Source.Parsing.NonDelimiterToken.newline
#print axioms Iota.Source.Parsing.NonDelimiterToken.assign
#print axioms Iota.Source.Parsing.NonDelimiterToken.lt
#print axioms Iota.Source.Parsing.NonDelimiterToken.le
#print axioms Iota.Source.Parsing.NonDelimiterToken.gt
#print axioms Iota.Source.Parsing.NonDelimiterToken.ge
#print axioms Iota.Source.Parsing.NonDelimiterToken.other
#print axioms Iota.Source.Parsing.DelimitersBalancedDerivation
#print axioms Iota.Source.Parsing.DelimitersBalancedDerivation.done
#print axioms Iota.Source.Parsing.DelimitersBalancedDerivation.lparen
#print axioms Iota.Source.Parsing.DelimitersBalancedDerivation.lbrace
#print axioms Iota.Source.Parsing.DelimitersBalancedDerivation.lbracket
#print axioms Iota.Source.Parsing.DelimitersBalancedDerivation.rparen
#print axioms Iota.Source.Parsing.DelimitersBalancedDerivation.rbrace
#print axioms Iota.Source.Parsing.DelimitersBalancedDerivation.rbracket
#print axioms Iota.Source.Parsing.DelimitersBalancedDerivation.other
#print axioms Iota.Source.Parsing.DelimitersBalancedDerivation.sound
#print axioms Iota.Source.Parsing.SpansValidDerivation
#print axioms Iota.Source.Parsing.SpansValidDerivation.intro
#print axioms Iota.Source.Parsing.SpansValidDerivation.sound
#print axioms Iota.Source.Parsing.DistinctDerivation
#print axioms Iota.Source.Parsing.DistinctDerivation.intro
#print axioms Iota.Source.Parsing.DistinctDerivation.sound
#print axioms Iota.Source.Parsing.ExpressionCallsResolveDerivation
#print axioms Iota.Source.Parsing.ExpressionCallsResolveDerivation.literal
#print axioms Iota.Source.Parsing.ExpressionCallsResolveDerivation.name
#print axioms Iota.Source.Parsing.ExpressionCallsResolveDerivation.call
#print axioms Iota.Source.Parsing.ExpressionCallsResolveDerivation.memory
#print axioms Iota.Source.Parsing.ExpressionCallsResolveDerivation.binary
#print axioms Iota.Source.Parsing.ExpressionCallsResolveDerivation.conditional
#print axioms Iota.Source.Parsing.ExpressionCallsResolveDerivation.sound
#print axioms Iota.Source.Parsing.ExpressionsCallsResolveDerivation
#print axioms Iota.Source.Parsing.ExpressionsCallsResolveDerivation.done
#print axioms Iota.Source.Parsing.ExpressionsCallsResolveDerivation.next
#print axioms Iota.Source.Parsing.ExpressionsCallsResolveDerivation.sound
#print axioms Iota.Source.Parsing.StatementCallsResolveDerivation
#print axioms Iota.Source.Parsing.StatementCallsResolveDerivation.bind
#print axioms Iota.Source.Parsing.StatementCallsResolveDerivation.store
#print axioms Iota.Source.Parsing.StatementCallsResolveDerivation.return_
#print axioms Iota.Source.Parsing.StatementCallsResolveDerivation.expression
#print axioms Iota.Source.Parsing.StatementCallsResolveDerivation.ifNone
#print axioms Iota.Source.Parsing.StatementCallsResolveDerivation.ifSome
#print axioms Iota.Source.Parsing.StatementCallsResolveDerivation.sound
#print axioms Iota.Source.Parsing.StatementsCallsResolveDerivation
#print axioms Iota.Source.Parsing.StatementsCallsResolveDerivation.done
#print axioms Iota.Source.Parsing.StatementsCallsResolveDerivation.next
#print axioms Iota.Source.Parsing.StatementsCallsResolveDerivation.sound
#print axioms Iota.Source.Parsing.FunctionCallsResolveDerivation
#print axioms Iota.Source.Parsing.FunctionCallsResolveDerivation.done
#print axioms Iota.Source.Parsing.FunctionCallsResolveDerivation.next
#print axioms Iota.Source.Parsing.FunctionCallsResolveDerivation.sound
#print axioms Iota.Source.Parsing.CallsResolveDerivation
#print axioms Iota.Source.Parsing.CallsResolveDerivation.intro
#print axioms Iota.Source.Parsing.CallsResolveDerivation.sound
#print axioms Iota.Source.Parsing.ExpressionNamesResolveDerivation
#print axioms Iota.Source.Parsing.ExpressionNamesResolveDerivation.literal
#print axioms Iota.Source.Parsing.ExpressionNamesResolveDerivation.name
#print axioms Iota.Source.Parsing.ExpressionNamesResolveDerivation.call
#print axioms Iota.Source.Parsing.ExpressionNamesResolveDerivation.memory
#print axioms Iota.Source.Parsing.ExpressionNamesResolveDerivation.binary
#print axioms Iota.Source.Parsing.ExpressionNamesResolveDerivation.conditional
#print axioms Iota.Source.Parsing.ExpressionNamesResolveDerivation.sound
#print axioms Iota.Source.Parsing.ExpressionsNamesResolveDerivation
#print axioms Iota.Source.Parsing.ExpressionsNamesResolveDerivation.done
#print axioms Iota.Source.Parsing.ExpressionsNamesResolveDerivation.next
#print axioms Iota.Source.Parsing.ExpressionsNamesResolveDerivation.sound
#print axioms Iota.Source.Parsing.StatementsNamesResolveDerivation
#print axioms Iota.Source.Parsing.StatementsNamesResolveDerivation.done
#print axioms Iota.Source.Parsing.StatementsNamesResolveDerivation.bind
#print axioms Iota.Source.Parsing.StatementsNamesResolveDerivation.store
#print axioms Iota.Source.Parsing.StatementsNamesResolveDerivation.return_
#print axioms Iota.Source.Parsing.StatementsNamesResolveDerivation.expression
#print axioms Iota.Source.Parsing.StatementsNamesResolveDerivation.ifNone
#print axioms Iota.Source.Parsing.StatementsNamesResolveDerivation.ifSome
#print axioms Iota.Source.Parsing.StatementsNamesResolveDerivation.sound
#print axioms Iota.Source.Parsing.FunctionNamesResolveDerivation
#print axioms Iota.Source.Parsing.FunctionNamesResolveDerivation.done
#print axioms Iota.Source.Parsing.FunctionNamesResolveDerivation.next
#print axioms Iota.Source.Parsing.FunctionNamesResolveDerivation.sound
#print axioms Iota.Source.Parsing.NamesResolveDerivation
#print axioms Iota.Source.Parsing.NamesResolveDerivation.intro
#print axioms Iota.Source.Parsing.NamesResolveDerivation.sound
#print axioms Iota.Source.Parsing.ValidationAudit.nat_add_zero
#audit_empty Iota.Source.Parsing.DelimitersBalancedDerivation.continue : type_of% @Iota.Source.Parsing.DelimitersBalancedDerivation.continue
#audit_empty Iota.Source.Parsing.SpansValidDerivation.all_append : type_of% @Iota.Source.Parsing.SpansValidDerivation.all_append
#audit_empty Iota.Source.Parsing.SpansValidDerivation.all_head : type_of% @Iota.Source.Parsing.SpansValidDerivation.all_head
#audit_empty Iota.Source.Parsing.SpansValidDerivation.all_tail : type_of% @Iota.Source.Parsing.SpansValidDerivation.all_tail
#audit_empty Iota.Source.Parsing.SpansValidDerivation.all_inner_append : type_of% @Iota.Source.Parsing.SpansValidDerivation.all_inner_append
#audit_empty Iota.Source.Parsing.SpansValidDerivation.append : type_of% @Iota.Source.Parsing.SpansValidDerivation.append
#audit_empty Iota.Source.Parsing.DistinctDerivation.reverseAux_length : type_of% @Iota.Source.Parsing.DistinctDerivation.reverseAux_length
#audit_empty Iota.Source.Parsing.DistinctDerivation.any_beq_false_of_not_mem : type_of% @Iota.Source.Parsing.DistinctDerivation.any_beq_false_of_not_mem
#audit_empty Iota.Source.Parsing.DistinctDerivation.eraseDupsLoop_length_le : type_of% @Iota.Source.Parsing.DistinctDerivation.eraseDupsLoop_length_le
#audit_empty Iota.Source.Parsing.DistinctDerivation.eraseDupsLoop_length_of_nodup : type_of% @Iota.Source.Parsing.DistinctDerivation.eraseDupsLoop_length_of_nodup
#audit_empty Iota.Source.Parsing.DistinctDerivation.of_nodup : type_of% @Iota.Source.Parsing.DistinctDerivation.of_nodup
#audit_empty Iota.Source.Parsing.DistinctDerivation.mem_append_cases : type_of% @Iota.Source.Parsing.DistinctDerivation.mem_append_cases
#audit_empty Iota.Source.Parsing.DistinctDerivation.nodup_append_of_cross : type_of% @Iota.Source.Parsing.DistinctDerivation.nodup_append_of_cross
#audit_empty Iota.Source.Parsing.DistinctDerivation.append : type_of% @Iota.Source.Parsing.DistinctDerivation.append
#audit_empty Iota.Source.Parsing.ExpressionsCallsResolveDerivation.append : type_of% @Iota.Source.Parsing.ExpressionsCallsResolveDerivation.append
#audit_empty Iota.Source.Parsing.StatementsCallsResolveDerivation.append : type_of% @Iota.Source.Parsing.StatementsCallsResolveDerivation.append
#audit_empty Iota.Source.Parsing.FunctionCallsResolveDerivation.append : type_of% @Iota.Source.Parsing.FunctionCallsResolveDerivation.append
#audit_empty Iota.Source.Parsing.CallsResolveDerivation.of_parts : type_of% @Iota.Source.Parsing.CallsResolveDerivation.of_parts
#audit_empty Iota.Source.Parsing.ExpressionsNamesResolveDerivation.append : type_of% @Iota.Source.Parsing.ExpressionsNamesResolveDerivation.append
#audit_empty Iota.Source.Parsing.StatementsNamesResolveDerivation.continue : type_of% @Iota.Source.Parsing.StatementsNamesResolveDerivation.continue
#audit_empty Iota.Source.Parsing.FunctionNamesResolveDerivation.append : type_of% @Iota.Source.Parsing.FunctionNamesResolveDerivation.append
#audit_empty Iota.Source.Parsing.NamesResolveDerivation.append : type_of% @Iota.Source.Parsing.NamesResolveDerivation.append
#print axioms Iota.Source.Parsing.DelimitersBalancedDerivation.continue
#print axioms Iota.Source.Parsing.SpansValidDerivation.all_append
#print axioms Iota.Source.Parsing.SpansValidDerivation.all_head
#print axioms Iota.Source.Parsing.SpansValidDerivation.all_tail
#print axioms Iota.Source.Parsing.SpansValidDerivation.all_inner_append
#print axioms Iota.Source.Parsing.SpansValidDerivation.append
#print axioms Iota.Source.Parsing.DistinctDerivation.reverseAux_length
#print axioms Iota.Source.Parsing.DistinctDerivation.any_beq_false_of_not_mem
#print axioms Iota.Source.Parsing.DistinctDerivation.eraseDupsLoop_length_le
#print axioms Iota.Source.Parsing.DistinctDerivation.eraseDupsLoop_length_of_nodup
#print axioms Iota.Source.Parsing.DistinctDerivation.of_nodup
#print axioms Iota.Source.Parsing.DistinctDerivation.mem_append_cases
#print axioms Iota.Source.Parsing.DistinctDerivation.nodup_append_of_cross
#print axioms Iota.Source.Parsing.DistinctDerivation.append
#print axioms Iota.Source.Parsing.ExpressionsCallsResolveDerivation.append
#print axioms Iota.Source.Parsing.StatementsCallsResolveDerivation.append
#print axioms Iota.Source.Parsing.FunctionCallsResolveDerivation.append
#print axioms Iota.Source.Parsing.CallsResolveDerivation.of_parts
#print axioms Iota.Source.Parsing.ExpressionsNamesResolveDerivation.append
#print axioms Iota.Source.Parsing.StatementsNamesResolveDerivation.continue
#print axioms Iota.Source.Parsing.FunctionNamesResolveDerivation.append
#print axioms Iota.Source.Parsing.NamesResolveDerivation.append

#audit_empty Iota.Source.Parsing.DelimitersBalancedDerivation.append : type_of% @Iota.Source.Parsing.DelimitersBalancedDerivation.append
#audit_empty Iota.Source.Parsing.DistinctDerivation.of_bool : type_of% @Iota.Source.Parsing.DistinctDerivation.of_bool
#audit_empty Iota.Source.Parsing.ExpressionCallsResolveDerivation.of_bool : type_of% @Iota.Source.Parsing.ExpressionCallsResolveDerivation.of_bool
#audit_empty Iota.Source.Parsing.ExpressionsCallsResolveDerivation.of_bool : type_of% @Iota.Source.Parsing.ExpressionsCallsResolveDerivation.of_bool
#audit_empty Iota.Source.Parsing.StatementCallsResolveDerivation.of_bool : type_of% @Iota.Source.Parsing.StatementCallsResolveDerivation.of_bool
#audit_empty Iota.Source.Parsing.StatementsCallsResolveDerivation.of_bool : type_of% @Iota.Source.Parsing.StatementsCallsResolveDerivation.of_bool
#audit_empty Iota.Source.Parsing.FunctionCallsResolveDerivation.of_bool : type_of% @Iota.Source.Parsing.FunctionCallsResolveDerivation.of_bool
#audit_empty Iota.Source.Parsing.CallsResolveDerivation.of_bool : type_of% @Iota.Source.Parsing.CallsResolveDerivation.of_bool
#audit_empty Iota.Source.Parsing.ExpressionNamesResolveDerivation.of_bool : type_of% @Iota.Source.Parsing.ExpressionNamesResolveDerivation.of_bool
#audit_empty Iota.Source.Parsing.ExpressionsNamesResolveDerivation.of_bool : type_of% @Iota.Source.Parsing.ExpressionsNamesResolveDerivation.of_bool
#audit_empty Iota.Source.Parsing.StatementsNamesResolveDerivation.of_bool : type_of% @Iota.Source.Parsing.StatementsNamesResolveDerivation.of_bool
#audit_empty Iota.Source.Parsing.FunctionNamesResolveDerivation.of_bool : type_of% @Iota.Source.Parsing.FunctionNamesResolveDerivation.of_bool
#audit_empty Iota.Source.Parsing.NamesResolveDerivation.of_bool : type_of% @Iota.Source.Parsing.NamesResolveDerivation.of_bool
#audit_empty Iota.Source.Parsing.append_assoc_empty : type_of% @Iota.Source.Parsing.append_assoc_empty
#audit_empty Iota.Source.Parsing.DelimitersBalancedDerivation.of_bool : type_of% @Iota.Source.Parsing.DelimitersBalancedDerivation.of_bool

#print axioms Iota.Source.Parsing.append_assoc_empty
#print axioms Iota.Source.Parsing.DelimitersBalancedDerivation.of_bool
