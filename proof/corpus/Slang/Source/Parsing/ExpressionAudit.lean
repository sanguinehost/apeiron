import Slang.Audit.Empty
import Slang.Source.Parsing.Expression

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Slang.Source.Parsing.ExpressionAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Source.Parsing.ExpressionAudit

#audit_empty Slang.Source.Parsing.OrdinaryBinaryGuard : Slang.Decode.Token → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.OrdinaryBinaryGuard.notIs : type_of% @Slang.Source.Parsing.OrdinaryBinaryGuard.notIs
#audit_empty Slang.Source.Parsing.OrdinaryBinaryGuard.isEnd : type_of% @Slang.Source.Parsing.OrdinaryBinaryGuard.isEnd
#audit_empty Slang.Source.Parsing.OrdinaryBinaryGuard.isNext : type_of% @Slang.Source.Parsing.OrdinaryBinaryGuard.isNext
#audit_empty Slang.Source.Parsing.ExprScan : Nat → Nat → List Slang.Decode.Token → Slang.Expr → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.ExprScan.step : type_of% @Slang.Source.Parsing.ExprScan.step
#audit_empty Slang.Source.Parsing.ExprScan.sound : type_of% @Slang.Source.Parsing.ExprScan.sound
#audit_empty Slang.Source.Parsing.PrimaryScan : Nat → List Slang.Decode.Token → Slang.Expr → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.PrimaryScan.numeral : type_of% @Slang.Source.Parsing.PrimaryScan.numeral
#audit_empty Slang.Source.Parsing.PrimaryScan.nameEnd : type_of% @Slang.Source.Parsing.PrimaryScan.nameEnd
#audit_empty Slang.Source.Parsing.PrimaryScan.nameNext : type_of% @Slang.Source.Parsing.PrimaryScan.nameNext
#audit_empty Slang.Source.Parsing.PrimaryScan.call : type_of% @Slang.Source.Parsing.PrimaryScan.call
#audit_empty Slang.Source.Parsing.PrimaryScan.memory : type_of% @Slang.Source.Parsing.PrimaryScan.memory
#audit_empty Slang.Source.Parsing.PrimaryScan.parenthesized : type_of% @Slang.Source.Parsing.PrimaryScan.parenthesized
#audit_empty Slang.Source.Parsing.PrimaryScan.sound : type_of% @Slang.Source.Parsing.PrimaryScan.sound
#audit_empty Slang.Source.Parsing.ArgsScan : Nat → List Slang.Decode.Token → List Slang.Expr → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.ArgsScan.done : type_of% @Slang.Source.Parsing.ArgsScan.done
#audit_empty Slang.Source.Parsing.ArgsScan.comma : type_of% @Slang.Source.Parsing.ArgsScan.comma
#audit_empty Slang.Source.Parsing.ArgsScan.last : type_of% @Slang.Source.Parsing.ArgsScan.last
#audit_empty Slang.Source.Parsing.ArgsScan.sound : type_of% @Slang.Source.Parsing.ArgsScan.sound
#audit_empty Slang.Source.Parsing.BinaryScan : Nat → Nat → Slang.Expr → List Slang.Decode.Token → Slang.Expr → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.BinaryScan.done : type_of% @Slang.Source.Parsing.BinaryScan.done
#audit_empty Slang.Source.Parsing.BinaryScan.belowPrec : type_of% @Slang.Source.Parsing.BinaryScan.belowPrec
#audit_empty Slang.Source.Parsing.BinaryScan.operatorStep : type_of% @Slang.Source.Parsing.BinaryScan.operatorStep
#audit_empty Slang.Source.Parsing.BinaryScan.conditionalStop : type_of% @Slang.Source.Parsing.BinaryScan.conditionalStop
#audit_empty Slang.Source.Parsing.BinaryScan.conditional : type_of% @Slang.Source.Parsing.BinaryScan.conditional
#audit_empty Slang.Source.Parsing.BinaryScan.stop : type_of% @Slang.Source.Parsing.BinaryScan.stop
#audit_empty Slang.Source.Parsing.BinaryScan.isNotStop : type_of% @Slang.Source.Parsing.BinaryScan.isNotStop
#audit_empty Slang.Source.Parsing.BinaryScan.isNot : type_of% @Slang.Source.Parsing.BinaryScan.isNot
#audit_empty Slang.Source.Parsing.BinaryScan.sound : type_of% @Slang.Source.Parsing.BinaryScan.sound
#audit_empty Slang.Source.Parsing.ExprScanTop : List Slang.Decode.Token → Slang.Expr → List Slang.Decode.Token → Prop
#audit_empty Slang.Source.Parsing.ExprScanTop.intro : type_of% @Slang.Source.Parsing.ExprScanTop.intro
#audit_empty Slang.Source.Parsing.ExprScanTop.sound : type_of% @Slang.Source.Parsing.ExprScanTop.sound
#audit_empty Slang.Source.Parsing.ExpressionAudit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Slang.Source.Parsing.OrdinaryBinaryGuard
#print axioms Slang.Source.Parsing.OrdinaryBinaryGuard.notIs
#print axioms Slang.Source.Parsing.OrdinaryBinaryGuard.isEnd
#print axioms Slang.Source.Parsing.OrdinaryBinaryGuard.isNext
#print axioms Slang.Source.Parsing.ExprScan
#print axioms Slang.Source.Parsing.ExprScan.step
#print axioms Slang.Source.Parsing.ExprScan.sound
#print axioms Slang.Source.Parsing.PrimaryScan
#print axioms Slang.Source.Parsing.PrimaryScan.numeral
#print axioms Slang.Source.Parsing.PrimaryScan.nameEnd
#print axioms Slang.Source.Parsing.PrimaryScan.nameNext
#print axioms Slang.Source.Parsing.PrimaryScan.call
#print axioms Slang.Source.Parsing.PrimaryScan.memory
#print axioms Slang.Source.Parsing.PrimaryScan.parenthesized
#print axioms Slang.Source.Parsing.PrimaryScan.sound
#print axioms Slang.Source.Parsing.ArgsScan
#print axioms Slang.Source.Parsing.ArgsScan.done
#print axioms Slang.Source.Parsing.ArgsScan.comma
#print axioms Slang.Source.Parsing.ArgsScan.last
#print axioms Slang.Source.Parsing.ArgsScan.sound
#print axioms Slang.Source.Parsing.BinaryScan
#print axioms Slang.Source.Parsing.BinaryScan.done
#print axioms Slang.Source.Parsing.BinaryScan.belowPrec
#print axioms Slang.Source.Parsing.BinaryScan.operatorStep
#print axioms Slang.Source.Parsing.BinaryScan.conditionalStop
#print axioms Slang.Source.Parsing.BinaryScan.conditional
#print axioms Slang.Source.Parsing.BinaryScan.stop
#print axioms Slang.Source.Parsing.BinaryScan.isNotStop
#print axioms Slang.Source.Parsing.BinaryScan.isNot
#print axioms Slang.Source.Parsing.BinaryScan.sound
#print axioms Slang.Source.Parsing.ExprScanTop
#print axioms Slang.Source.Parsing.ExprScanTop.intro
#print axioms Slang.Source.Parsing.ExprScanTop.sound
#print axioms Slang.Source.Parsing.ExpressionAudit.nat_add_zero
