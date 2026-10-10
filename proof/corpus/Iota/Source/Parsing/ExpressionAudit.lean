import Iota.Audit.Empty
import Iota.Source.Parsing.Expression

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Iota.Source.Parsing.ExpressionAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Source.Parsing.ExpressionAudit

#audit_empty Iota.Source.Parsing.OrdinaryBinaryGuard : Iota.Decode.Token → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.OrdinaryBinaryGuard.notIs : type_of% @Iota.Source.Parsing.OrdinaryBinaryGuard.notIs
#audit_empty Iota.Source.Parsing.OrdinaryBinaryGuard.isEnd : type_of% @Iota.Source.Parsing.OrdinaryBinaryGuard.isEnd
#audit_empty Iota.Source.Parsing.OrdinaryBinaryGuard.isNext : type_of% @Iota.Source.Parsing.OrdinaryBinaryGuard.isNext
#audit_empty Iota.Source.Parsing.ExprScan : Nat → Nat → List Iota.Decode.Token → Iota.Expr → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.ExprScan.step : type_of% @Iota.Source.Parsing.ExprScan.step
#audit_empty Iota.Source.Parsing.ExprScan.sound : type_of% @Iota.Source.Parsing.ExprScan.sound
#audit_empty Iota.Source.Parsing.PrimaryScan : Nat → List Iota.Decode.Token → Iota.Expr → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.PrimaryScan.numeral : type_of% @Iota.Source.Parsing.PrimaryScan.numeral
#audit_empty Iota.Source.Parsing.PrimaryScan.nameEnd : type_of% @Iota.Source.Parsing.PrimaryScan.nameEnd
#audit_empty Iota.Source.Parsing.PrimaryScan.nameNext : type_of% @Iota.Source.Parsing.PrimaryScan.nameNext
#audit_empty Iota.Source.Parsing.PrimaryScan.call : type_of% @Iota.Source.Parsing.PrimaryScan.call
#audit_empty Iota.Source.Parsing.PrimaryScan.memory : type_of% @Iota.Source.Parsing.PrimaryScan.memory
#audit_empty Iota.Source.Parsing.PrimaryScan.parenthesized : type_of% @Iota.Source.Parsing.PrimaryScan.parenthesized
#audit_empty Iota.Source.Parsing.PrimaryScan.sound : type_of% @Iota.Source.Parsing.PrimaryScan.sound
#audit_empty Iota.Source.Parsing.ArgsScan : Nat → List Iota.Decode.Token → List Iota.Expr → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.ArgsScan.done : type_of% @Iota.Source.Parsing.ArgsScan.done
#audit_empty Iota.Source.Parsing.ArgsScan.comma : type_of% @Iota.Source.Parsing.ArgsScan.comma
#audit_empty Iota.Source.Parsing.ArgsScan.last : type_of% @Iota.Source.Parsing.ArgsScan.last
#audit_empty Iota.Source.Parsing.ArgsScan.sound : type_of% @Iota.Source.Parsing.ArgsScan.sound
#audit_empty Iota.Source.Parsing.BinaryScan : Nat → Nat → Iota.Expr → List Iota.Decode.Token → Iota.Expr → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.BinaryScan.done : type_of% @Iota.Source.Parsing.BinaryScan.done
#audit_empty Iota.Source.Parsing.BinaryScan.belowPrec : type_of% @Iota.Source.Parsing.BinaryScan.belowPrec
#audit_empty Iota.Source.Parsing.BinaryScan.operatorStep : type_of% @Iota.Source.Parsing.BinaryScan.operatorStep
#audit_empty Iota.Source.Parsing.BinaryScan.conditionalStop : type_of% @Iota.Source.Parsing.BinaryScan.conditionalStop
#audit_empty Iota.Source.Parsing.BinaryScan.conditional : type_of% @Iota.Source.Parsing.BinaryScan.conditional
#audit_empty Iota.Source.Parsing.BinaryScan.stop : type_of% @Iota.Source.Parsing.BinaryScan.stop
#audit_empty Iota.Source.Parsing.BinaryScan.isNotStop : type_of% @Iota.Source.Parsing.BinaryScan.isNotStop
#audit_empty Iota.Source.Parsing.BinaryScan.isNot : type_of% @Iota.Source.Parsing.BinaryScan.isNot
#audit_empty Iota.Source.Parsing.BinaryScan.sound : type_of% @Iota.Source.Parsing.BinaryScan.sound
#audit_empty Iota.Source.Parsing.ExprScanTop : List Iota.Decode.Token → Iota.Expr → List Iota.Decode.Token → Prop
#audit_empty Iota.Source.Parsing.ExprScanTop.intro : type_of% @Iota.Source.Parsing.ExprScanTop.intro
#audit_empty Iota.Source.Parsing.ExprScanTop.sound : type_of% @Iota.Source.Parsing.ExprScanTop.sound
#audit_empty Iota.Source.Parsing.ExpressionAudit.nat_add_zero : (0 : Nat) + 0 = 0

#print axioms Iota.Source.Parsing.OrdinaryBinaryGuard
#print axioms Iota.Source.Parsing.OrdinaryBinaryGuard.notIs
#print axioms Iota.Source.Parsing.OrdinaryBinaryGuard.isEnd
#print axioms Iota.Source.Parsing.OrdinaryBinaryGuard.isNext
#print axioms Iota.Source.Parsing.ExprScan
#print axioms Iota.Source.Parsing.ExprScan.step
#print axioms Iota.Source.Parsing.ExprScan.sound
#print axioms Iota.Source.Parsing.PrimaryScan
#print axioms Iota.Source.Parsing.PrimaryScan.numeral
#print axioms Iota.Source.Parsing.PrimaryScan.nameEnd
#print axioms Iota.Source.Parsing.PrimaryScan.nameNext
#print axioms Iota.Source.Parsing.PrimaryScan.call
#print axioms Iota.Source.Parsing.PrimaryScan.memory
#print axioms Iota.Source.Parsing.PrimaryScan.parenthesized
#print axioms Iota.Source.Parsing.PrimaryScan.sound
#print axioms Iota.Source.Parsing.ArgsScan
#print axioms Iota.Source.Parsing.ArgsScan.done
#print axioms Iota.Source.Parsing.ArgsScan.comma
#print axioms Iota.Source.Parsing.ArgsScan.last
#print axioms Iota.Source.Parsing.ArgsScan.sound
#print axioms Iota.Source.Parsing.BinaryScan
#print axioms Iota.Source.Parsing.BinaryScan.done
#print axioms Iota.Source.Parsing.BinaryScan.belowPrec
#print axioms Iota.Source.Parsing.BinaryScan.operatorStep
#print axioms Iota.Source.Parsing.BinaryScan.conditionalStop
#print axioms Iota.Source.Parsing.BinaryScan.conditional
#print axioms Iota.Source.Parsing.BinaryScan.stop
#print axioms Iota.Source.Parsing.BinaryScan.isNotStop
#print axioms Iota.Source.Parsing.BinaryScan.isNot
#print axioms Iota.Source.Parsing.BinaryScan.sound
#print axioms Iota.Source.Parsing.ExprScanTop
#print axioms Iota.Source.Parsing.ExprScanTop.intro
#print axioms Iota.Source.Parsing.ExprScanTop.sound
#print axioms Iota.Source.Parsing.ExpressionAudit.nat_add_zero
