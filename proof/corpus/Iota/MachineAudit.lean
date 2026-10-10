import Iota.Audit.Empty
import Iota.Machine

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Machine.Audit
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl
end Iota.Machine.Audit

#audit_empty Iota.Machine.Audit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Machine.Environment : Type
#audit_empty Iota.Machine.Environment.lookup : Iota.Machine.Environment → String → Option Iota.Word
#audit_empty Iota.Machine.Environment.bind : Iota.Machine.Environment → String → Iota.Word → Iota.Machine.Environment
#audit_empty Iota.Machine.Environment.ofParameters : List String → List Iota.Word → Option Iota.Machine.Environment
#audit_empty Iota.Machine.Environment.lookup_bind_same : ∀ (environment : Iota.Machine.Environment) (name : String) (value : Iota.Word), Iota.Machine.Environment.lookup (Iota.Machine.Environment.bind environment name value) name = some value
#audit_empty Iota.Machine.Environment.lookup_bind_other : ∀ (environment : Iota.Machine.Environment) (name other : String) (value : Iota.Word), other ≠ name → Iota.Machine.Environment.lookup (Iota.Machine.Environment.bind environment name value) other = Iota.Machine.Environment.lookup environment other

#audit_empty Iota.Machine.Memory : Type
#audit_empty Iota.Machine.SameOwnership : Iota.Machine.Memory → Iota.Machine.Memory → Prop
#audit_empty Iota.Machine.MemoryEquiv : Iota.Machine.Memory → Iota.Machine.Memory → Prop
#audit_empty Iota.Machine.Memory.coordinate : Iota.Word → Option Nat
#audit_empty Iota.Machine.Memory.load : Iota.Machine.Memory → Iota.Word → Option Iota.Word
#audit_empty Iota.Machine.Memory.store : Iota.Machine.Memory → Iota.Word → Iota.Word → Option Iota.Machine.Memory
#audit_empty Iota.Machine.Memory.empty : Iota.Machine.Memory
#audit_empty Iota.Machine.Memory.store_ownership : ∀ (memory next : Iota.Machine.Memory) (address value : Iota.Word), Iota.Machine.Memory.store memory address value = some next → next.owned = memory.owned
#audit_empty Iota.Machine.Memory.store_at : ∀ (memory next : Iota.Machine.Memory) (address value : Iota.Word) (coordinate : Nat), Iota.Machine.Memory.coordinate address = some coordinate → Iota.Machine.Memory.store memory address value = some next → next.initialized coordinate = some value
#audit_empty Iota.Machine.Memory.store_other : ∀ (memory next : Iota.Machine.Memory) (address value : Iota.Word) (coordinate other : Nat), Iota.Machine.Memory.coordinate address = some coordinate → Iota.Machine.Memory.store memory address value = some next → other ≠ coordinate → next.initialized other = memory.initialized other

#audit_empty Iota.Machine.StackPolicy : Type
#audit_empty Iota.Machine.instReprStackPolicy : Repr Iota.Machine.StackPolicy
#audit_empty Iota.Machine.stackPolicyBeq : Iota.Machine.StackPolicy → Iota.Machine.StackPolicy → Bool
#audit_empty Iota.Machine.instBEqStackPolicy : BEq Iota.Machine.StackPolicy
#audit_empty Iota.Machine.StackPolicy.allows : Iota.Machine.StackPolicy → Nat → Prop
#audit_empty Iota.Machine.Context : Type
#audit_empty Iota.Machine.Context.ofProgram : Iota.Program → (String → Option Iota.Word) → Iota.Machine.Context
#audit_empty Iota.Machine.Event : Type
#audit_empty Iota.Machine.binOpTag : Iota.BinOp → Nat
#audit_empty Iota.Machine.stringBeq : String → String → Bool
#audit_empty Iota.Machine.instReprEvent : Repr Iota.Machine.Event
#audit_empty Iota.Machine.Trace : Type
#audit_empty Iota.Machine.Resources : Type
#audit_empty Iota.Machine.instReprResources : Repr Iota.Machine.Resources
#audit_empty Iota.Machine.instBEqResources : BEq Iota.Machine.Resources
#audit_empty Iota.Machine.isProbe : String → Bool
#audit_empty Iota.Machine.emptyResources : Iota.Machine.Resources
#audit_empty Iota.Machine.account : Iota.Machine.Resources → Iota.Machine.Event → Iota.Machine.Resources
#audit_empty Iota.Machine.resources : Iota.Machine.Trace → Iota.Machine.Resources
#audit_empty Iota.Machine.arithmeticCount : Iota.Machine.Trace → Nat
#audit_empty Iota.Machine.comparisonCount : Iota.Machine.Trace → Nat
#audit_empty Iota.Machine.bitwiseCount : Iota.Machine.Trace → Nat
#audit_empty Iota.Machine.loadCount : Iota.Machine.Trace → Nat
#audit_empty Iota.Machine.storeCount : Iota.Machine.Trace → Nat
#audit_empty Iota.Machine.callCount : Iota.Machine.Trace → Nat
#audit_empty Iota.Machine.probeIterationCount : Iota.Machine.Trace → Nat
#audit_empty Iota.Machine.callStackMaximum : Iota.Machine.Trace → Nat

#audit_empty Iota.Machine.resources_arithmeticOps : type_of% @Iota.Machine.resources_arithmeticOps
#audit_empty Iota.Machine.resources_comparisons : type_of% @Iota.Machine.resources_comparisons
#audit_empty Iota.Machine.resources_bitwiseOps : type_of% @Iota.Machine.resources_bitwiseOps
#audit_empty Iota.Machine.resources_loads : type_of% @Iota.Machine.resources_loads
#audit_empty Iota.Machine.resources_stores : type_of% @Iota.Machine.resources_stores
#audit_empty Iota.Machine.resources_calls : type_of% @Iota.Machine.resources_calls
#audit_empty Iota.Machine.resources_probeIterations : type_of% @Iota.Machine.resources_probeIterations
#audit_empty Iota.Machine.resources_maxCallStack : type_of% @Iota.Machine.resources_maxCallStack
#audit_empty Iota.Machine.resources_allocatedNodes : ∀ (trace : Iota.Machine.Trace), (Iota.Machine.resources trace).allocatedNodes = 0
#audit_empty Iota.Machine.resources_externalStreamConsumption : ∀ (trace : Iota.Machine.Trace), (Iota.Machine.resources trace).externalStreamConsumption = 0

#audit_empty Iota.Machine.EvalResult : Type → Type
#audit_empty Iota.Machine.Control : Type
#audit_empty Iota.Machine.applyBinOp : Iota.BinOp → Iota.Word → Iota.Word → Iota.Word
#audit_empty Iota.Machine.operatorEvent : Iota.BinOp → Iota.Machine.Event
#audit_empty Iota.Machine.EvalExpr : Iota.Machine.Context → Iota.Machine.StackPolicy → Nat → Iota.Machine.Environment → Iota.Machine.Memory → Iota.Expr → Iota.Machine.EvalResult Iota.Word → Iota.Machine.Trace → Prop
#audit_empty Iota.Machine.EvalArgs : Iota.Machine.Context → Iota.Machine.StackPolicy → Nat → Iota.Machine.Environment → Iota.Machine.Memory → List Iota.Expr → Iota.Machine.EvalResult (List Iota.Word) → Iota.Machine.Trace → Prop
#audit_empty Iota.Machine.ExecStmt : Iota.Machine.Context → Iota.Machine.StackPolicy → Nat → Iota.Machine.Environment → Iota.Machine.Memory → Iota.Stmt → Iota.Machine.Control → Iota.Machine.Trace → Prop
#audit_empty Iota.Machine.ExecList : Iota.Machine.Context → Iota.Machine.StackPolicy → Nat → Iota.Machine.Environment → Iota.Machine.Memory → List Iota.Stmt → Iota.Machine.Control → Iota.Machine.Trace → Prop
#audit_empty Iota.Machine.EvalCall : Iota.Machine.Context → Iota.Machine.StackPolicy → Nat → String → List Iota.Word → Iota.Machine.Memory → Iota.Machine.EvalResult Iota.Word → Iota.Machine.Trace → Prop

-- The generated mutual recursors and constructors are part of the registered
-- execution relation, not inferred from auditing only its five type constants.
#audit_empty Iota.Machine.EvalExpr.rec : type_of% @Iota.Machine.EvalExpr.rec
#audit_empty Iota.Machine.EvalArgs.rec : type_of% @Iota.Machine.EvalArgs.rec
#audit_empty Iota.Machine.ExecStmt.rec : type_of% @Iota.Machine.ExecStmt.rec
#audit_empty Iota.Machine.ExecList.rec : type_of% @Iota.Machine.ExecList.rec
#audit_empty Iota.Machine.EvalCall.rec : type_of% @Iota.Machine.EvalCall.rec

#audit_empty Iota.Machine.EvalExpr.literal : type_of% @Iota.Machine.EvalExpr.literal
#audit_empty Iota.Machine.EvalExpr.local : type_of% @Iota.Machine.EvalExpr.local
#audit_empty Iota.Machine.EvalExpr.constant : type_of% @Iota.Machine.EvalExpr.constant
#audit_empty Iota.Machine.EvalExpr.load : type_of% @Iota.Machine.EvalExpr.load
#audit_empty Iota.Machine.EvalExpr.loadFault : type_of% @Iota.Machine.EvalExpr.loadFault
#audit_empty Iota.Machine.EvalExpr.memoryAddressFault : type_of% @Iota.Machine.EvalExpr.memoryAddressFault
#audit_empty Iota.Machine.EvalExpr.memoryAddressExhausted : type_of% @Iota.Machine.EvalExpr.memoryAddressExhausted
#audit_empty Iota.Machine.EvalExpr.binary : type_of% @Iota.Machine.EvalExpr.binary
#audit_empty Iota.Machine.EvalExpr.binaryLeftFault : type_of% @Iota.Machine.EvalExpr.binaryLeftFault
#audit_empty Iota.Machine.EvalExpr.binaryLeftExhausted : type_of% @Iota.Machine.EvalExpr.binaryLeftExhausted
#audit_empty Iota.Machine.EvalExpr.binaryRightFault : type_of% @Iota.Machine.EvalExpr.binaryRightFault
#audit_empty Iota.Machine.EvalExpr.binaryRightExhausted : type_of% @Iota.Machine.EvalExpr.binaryRightExhausted
#audit_empty Iota.Machine.EvalExpr.conditionalTrue : type_of% @Iota.Machine.EvalExpr.conditionalTrue
#audit_empty Iota.Machine.EvalExpr.conditionalFalse : type_of% @Iota.Machine.EvalExpr.conditionalFalse
#audit_empty Iota.Machine.EvalExpr.conditionalFault : type_of% @Iota.Machine.EvalExpr.conditionalFault
#audit_empty Iota.Machine.EvalExpr.conditionalExhausted : type_of% @Iota.Machine.EvalExpr.conditionalExhausted
#audit_empty Iota.Machine.EvalExpr.call : type_of% @Iota.Machine.EvalExpr.call
#audit_empty Iota.Machine.EvalExpr.callArgsFault : type_of% @Iota.Machine.EvalExpr.callArgsFault
#audit_empty Iota.Machine.EvalExpr.callArgsExhausted : type_of% @Iota.Machine.EvalExpr.callArgsExhausted

#audit_empty Iota.Machine.EvalArgs.nil : type_of% @Iota.Machine.EvalArgs.nil
#audit_empty Iota.Machine.EvalArgs.cons : type_of% @Iota.Machine.EvalArgs.cons
#audit_empty Iota.Machine.EvalArgs.headFault : type_of% @Iota.Machine.EvalArgs.headFault
#audit_empty Iota.Machine.EvalArgs.headExhausted : type_of% @Iota.Machine.EvalArgs.headExhausted
#audit_empty Iota.Machine.EvalArgs.tailFault : type_of% @Iota.Machine.EvalArgs.tailFault
#audit_empty Iota.Machine.EvalArgs.tailExhausted : type_of% @Iota.Machine.EvalArgs.tailExhausted

#audit_empty Iota.Machine.ExecStmt.bind : type_of% @Iota.Machine.ExecStmt.bind
#audit_empty Iota.Machine.ExecStmt.bindFault : type_of% @Iota.Machine.ExecStmt.bindFault
#audit_empty Iota.Machine.ExecStmt.bindExhausted : type_of% @Iota.Machine.ExecStmt.bindExhausted
#audit_empty Iota.Machine.ExecStmt.store : type_of% @Iota.Machine.ExecStmt.store
#audit_empty Iota.Machine.ExecStmt.storeFault : type_of% @Iota.Machine.ExecStmt.storeFault
#audit_empty Iota.Machine.ExecStmt.storeAddressFault : type_of% @Iota.Machine.ExecStmt.storeAddressFault
#audit_empty Iota.Machine.ExecStmt.storeAddressExhausted : type_of% @Iota.Machine.ExecStmt.storeAddressExhausted
#audit_empty Iota.Machine.ExecStmt.storeValueFault : type_of% @Iota.Machine.ExecStmt.storeValueFault
#audit_empty Iota.Machine.ExecStmt.storeValueExhausted : type_of% @Iota.Machine.ExecStmt.storeValueExhausted
#audit_empty Iota.Machine.ExecStmt.return_ : type_of% @Iota.Machine.ExecStmt.return_
#audit_empty Iota.Machine.ExecStmt.returnFault : type_of% @Iota.Machine.ExecStmt.returnFault
#audit_empty Iota.Machine.ExecStmt.returnExhausted : type_of% @Iota.Machine.ExecStmt.returnExhausted
#audit_empty Iota.Machine.ExecStmt.expression : type_of% @Iota.Machine.ExecStmt.expression
#audit_empty Iota.Machine.ExecStmt.expressionFault : type_of% @Iota.Machine.ExecStmt.expressionFault
#audit_empty Iota.Machine.ExecStmt.expressionExhausted : type_of% @Iota.Machine.ExecStmt.expressionExhausted
#audit_empty Iota.Machine.ExecStmt.ifTrue : type_of% @Iota.Machine.ExecStmt.ifTrue
#audit_empty Iota.Machine.ExecStmt.ifFalse : type_of% @Iota.Machine.ExecStmt.ifFalse
#audit_empty Iota.Machine.ExecStmt.ifFault : type_of% @Iota.Machine.ExecStmt.ifFault
#audit_empty Iota.Machine.ExecStmt.ifExhausted : type_of% @Iota.Machine.ExecStmt.ifExhausted

#audit_empty Iota.Machine.ExecList.nil : type_of% @Iota.Machine.ExecList.nil
#audit_empty Iota.Machine.ExecList.next : type_of% @Iota.Machine.ExecList.next
#audit_empty Iota.Machine.ExecList.returned : type_of% @Iota.Machine.ExecList.returned
#audit_empty Iota.Machine.ExecList.memoryFault : type_of% @Iota.Machine.ExecList.memoryFault
#audit_empty Iota.Machine.ExecList.stackExhausted : type_of% @Iota.Machine.ExecList.stackExhausted
#audit_empty Iota.Machine.EvalCall.run : type_of% @Iota.Machine.EvalCall.run
#audit_empty Iota.Machine.EvalCall.memoryFault : type_of% @Iota.Machine.EvalCall.memoryFault
#audit_empty Iota.Machine.EvalCall.nestedExhausted : type_of% @Iota.Machine.EvalCall.nestedExhausted
#audit_empty Iota.Machine.EvalCall.exhausted : type_of% @Iota.Machine.EvalCall.exhausted

#audit_empty Iota.Machine.emptyContext : Iota.Machine.Context
#audit_empty Iota.Machine.lazy_inline_conditional_does_not_evaluate_untaken_branch : type_of% @Iota.Machine.lazy_inline_conditional_does_not_evaluate_untaken_branch
#audit_empty Iota.Machine.eager_bitOr_evaluates_right_operand : type_of% @Iota.Machine.eager_bitOr_evaluates_right_operand
#audit_empty Iota.Machine.return_stops_execution : type_of% @Iota.Machine.return_stops_execution
#audit_empty Iota.Machine.stack_exhaustion_is_not_a_word_return : type_of% @Iota.Machine.stack_exhaustion_is_not_a_word_return
