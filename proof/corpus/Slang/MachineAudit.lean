import Slang.Audit.Empty
import Slang.Machine

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Machine.Audit
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl
end Slang.Machine.Audit

#audit_empty Slang.Machine.Audit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Machine.Environment : Type
#audit_empty Slang.Machine.Environment.lookup : Slang.Machine.Environment → String → Option Slang.Word
#audit_empty Slang.Machine.Environment.bind : Slang.Machine.Environment → String → Slang.Word → Slang.Machine.Environment
#audit_empty Slang.Machine.Environment.ofParameters : List String → List Slang.Word → Option Slang.Machine.Environment
#audit_empty Slang.Machine.Environment.lookup_bind_same : ∀ (environment : Slang.Machine.Environment) (name : String) (value : Slang.Word), Slang.Machine.Environment.lookup (Slang.Machine.Environment.bind environment name value) name = some value
#audit_empty Slang.Machine.Environment.lookup_bind_other : ∀ (environment : Slang.Machine.Environment) (name other : String) (value : Slang.Word), other ≠ name → Slang.Machine.Environment.lookup (Slang.Machine.Environment.bind environment name value) other = Slang.Machine.Environment.lookup environment other

#audit_empty Slang.Machine.Memory : Type
#audit_empty Slang.Machine.SameOwnership : Slang.Machine.Memory → Slang.Machine.Memory → Prop
#audit_empty Slang.Machine.MemoryEquiv : Slang.Machine.Memory → Slang.Machine.Memory → Prop
#audit_empty Slang.Machine.Memory.coordinate : Slang.Word → Option Nat
#audit_empty Slang.Machine.Memory.load : Slang.Machine.Memory → Slang.Word → Option Slang.Word
#audit_empty Slang.Machine.Memory.store : Slang.Machine.Memory → Slang.Word → Slang.Word → Option Slang.Machine.Memory
#audit_empty Slang.Machine.Memory.empty : Slang.Machine.Memory
#audit_empty Slang.Machine.Memory.store_ownership : ∀ (memory next : Slang.Machine.Memory) (address value : Slang.Word), Slang.Machine.Memory.store memory address value = some next → next.owned = memory.owned
#audit_empty Slang.Machine.Memory.store_at : ∀ (memory next : Slang.Machine.Memory) (address value : Slang.Word) (coordinate : Nat), Slang.Machine.Memory.coordinate address = some coordinate → Slang.Machine.Memory.store memory address value = some next → next.initialized coordinate = some value
#audit_empty Slang.Machine.Memory.store_other : ∀ (memory next : Slang.Machine.Memory) (address value : Slang.Word) (coordinate other : Nat), Slang.Machine.Memory.coordinate address = some coordinate → Slang.Machine.Memory.store memory address value = some next → other ≠ coordinate → next.initialized other = memory.initialized other

#audit_empty Slang.Machine.StackPolicy : Type
#audit_empty Slang.Machine.instReprStackPolicy : Repr Slang.Machine.StackPolicy
#audit_empty Slang.Machine.stackPolicyBeq : Slang.Machine.StackPolicy → Slang.Machine.StackPolicy → Bool
#audit_empty Slang.Machine.instBEqStackPolicy : BEq Slang.Machine.StackPolicy
#audit_empty Slang.Machine.StackPolicy.allows : Slang.Machine.StackPolicy → Nat → Prop
#audit_empty Slang.Machine.Context : Type
#audit_empty Slang.Machine.Context.ofProgram : Slang.Program → (String → Option Slang.Word) → Slang.Machine.Context
#audit_empty Slang.Machine.Event : Type
#audit_empty Slang.Machine.binOpTag : Slang.BinOp → Nat
#audit_empty Slang.Machine.stringBeq : String → String → Bool
#audit_empty Slang.Machine.instReprEvent : Repr Slang.Machine.Event
#audit_empty Slang.Machine.Trace : Type
#audit_empty Slang.Machine.Resources : Type
#audit_empty Slang.Machine.instReprResources : Repr Slang.Machine.Resources
#audit_empty Slang.Machine.instBEqResources : BEq Slang.Machine.Resources
#audit_empty Slang.Machine.isProbe : String → Bool
#audit_empty Slang.Machine.emptyResources : Slang.Machine.Resources
#audit_empty Slang.Machine.account : Slang.Machine.Resources → Slang.Machine.Event → Slang.Machine.Resources
#audit_empty Slang.Machine.resources : Slang.Machine.Trace → Slang.Machine.Resources
#audit_empty Slang.Machine.arithmeticCount : Slang.Machine.Trace → Nat
#audit_empty Slang.Machine.comparisonCount : Slang.Machine.Trace → Nat
#audit_empty Slang.Machine.bitwiseCount : Slang.Machine.Trace → Nat
#audit_empty Slang.Machine.loadCount : Slang.Machine.Trace → Nat
#audit_empty Slang.Machine.storeCount : Slang.Machine.Trace → Nat
#audit_empty Slang.Machine.callCount : Slang.Machine.Trace → Nat
#audit_empty Slang.Machine.probeIterationCount : Slang.Machine.Trace → Nat
#audit_empty Slang.Machine.callStackMaximum : Slang.Machine.Trace → Nat

#audit_empty Slang.Machine.resources_arithmeticOps : type_of% @Slang.Machine.resources_arithmeticOps
#audit_empty Slang.Machine.resources_comparisons : type_of% @Slang.Machine.resources_comparisons
#audit_empty Slang.Machine.resources_bitwiseOps : type_of% @Slang.Machine.resources_bitwiseOps
#audit_empty Slang.Machine.resources_loads : type_of% @Slang.Machine.resources_loads
#audit_empty Slang.Machine.resources_stores : type_of% @Slang.Machine.resources_stores
#audit_empty Slang.Machine.resources_calls : type_of% @Slang.Machine.resources_calls
#audit_empty Slang.Machine.resources_probeIterations : type_of% @Slang.Machine.resources_probeIterations
#audit_empty Slang.Machine.resources_maxCallStack : type_of% @Slang.Machine.resources_maxCallStack
#audit_empty Slang.Machine.resources_allocatedNodes : ∀ (trace : Slang.Machine.Trace), (Slang.Machine.resources trace).allocatedNodes = 0
#audit_empty Slang.Machine.resources_externalStreamConsumption : ∀ (trace : Slang.Machine.Trace), (Slang.Machine.resources trace).externalStreamConsumption = 0

#audit_empty Slang.Machine.EvalResult : Type → Type
#audit_empty Slang.Machine.Control : Type
#audit_empty Slang.Machine.applyBinOp : Slang.BinOp → Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Machine.operatorEvent : Slang.BinOp → Slang.Machine.Event
#audit_empty Slang.Machine.EvalExpr : Slang.Machine.Context → Slang.Machine.StackPolicy → Nat → Slang.Machine.Environment → Slang.Machine.Memory → Slang.Expr → Slang.Machine.EvalResult Slang.Word → Slang.Machine.Trace → Prop
#audit_empty Slang.Machine.EvalArgs : Slang.Machine.Context → Slang.Machine.StackPolicy → Nat → Slang.Machine.Environment → Slang.Machine.Memory → List Slang.Expr → Slang.Machine.EvalResult (List Slang.Word) → Slang.Machine.Trace → Prop
#audit_empty Slang.Machine.ExecStmt : Slang.Machine.Context → Slang.Machine.StackPolicy → Nat → Slang.Machine.Environment → Slang.Machine.Memory → Slang.Stmt → Slang.Machine.Control → Slang.Machine.Trace → Prop
#audit_empty Slang.Machine.ExecList : Slang.Machine.Context → Slang.Machine.StackPolicy → Nat → Slang.Machine.Environment → Slang.Machine.Memory → List Slang.Stmt → Slang.Machine.Control → Slang.Machine.Trace → Prop
#audit_empty Slang.Machine.EvalCall : Slang.Machine.Context → Slang.Machine.StackPolicy → Nat → String → List Slang.Word → Slang.Machine.Memory → Slang.Machine.EvalResult Slang.Word → Slang.Machine.Trace → Prop

-- The generated mutual recursors and constructors are part of the registered
-- execution relation, not inferred from auditing only its five type constants.
#audit_empty Slang.Machine.EvalExpr.rec : type_of% @Slang.Machine.EvalExpr.rec
#audit_empty Slang.Machine.EvalArgs.rec : type_of% @Slang.Machine.EvalArgs.rec
#audit_empty Slang.Machine.ExecStmt.rec : type_of% @Slang.Machine.ExecStmt.rec
#audit_empty Slang.Machine.ExecList.rec : type_of% @Slang.Machine.ExecList.rec
#audit_empty Slang.Machine.EvalCall.rec : type_of% @Slang.Machine.EvalCall.rec

#audit_empty Slang.Machine.EvalExpr.literal : type_of% @Slang.Machine.EvalExpr.literal
#audit_empty Slang.Machine.EvalExpr.local : type_of% @Slang.Machine.EvalExpr.local
#audit_empty Slang.Machine.EvalExpr.constant : type_of% @Slang.Machine.EvalExpr.constant
#audit_empty Slang.Machine.EvalExpr.load : type_of% @Slang.Machine.EvalExpr.load
#audit_empty Slang.Machine.EvalExpr.loadFault : type_of% @Slang.Machine.EvalExpr.loadFault
#audit_empty Slang.Machine.EvalExpr.memoryAddressFault : type_of% @Slang.Machine.EvalExpr.memoryAddressFault
#audit_empty Slang.Machine.EvalExpr.memoryAddressExhausted : type_of% @Slang.Machine.EvalExpr.memoryAddressExhausted
#audit_empty Slang.Machine.EvalExpr.binary : type_of% @Slang.Machine.EvalExpr.binary
#audit_empty Slang.Machine.EvalExpr.binaryLeftFault : type_of% @Slang.Machine.EvalExpr.binaryLeftFault
#audit_empty Slang.Machine.EvalExpr.binaryLeftExhausted : type_of% @Slang.Machine.EvalExpr.binaryLeftExhausted
#audit_empty Slang.Machine.EvalExpr.binaryRightFault : type_of% @Slang.Machine.EvalExpr.binaryRightFault
#audit_empty Slang.Machine.EvalExpr.binaryRightExhausted : type_of% @Slang.Machine.EvalExpr.binaryRightExhausted
#audit_empty Slang.Machine.EvalExpr.conditionalTrue : type_of% @Slang.Machine.EvalExpr.conditionalTrue
#audit_empty Slang.Machine.EvalExpr.conditionalFalse : type_of% @Slang.Machine.EvalExpr.conditionalFalse
#audit_empty Slang.Machine.EvalExpr.conditionalFault : type_of% @Slang.Machine.EvalExpr.conditionalFault
#audit_empty Slang.Machine.EvalExpr.conditionalExhausted : type_of% @Slang.Machine.EvalExpr.conditionalExhausted
#audit_empty Slang.Machine.EvalExpr.call : type_of% @Slang.Machine.EvalExpr.call
#audit_empty Slang.Machine.EvalExpr.callArgsFault : type_of% @Slang.Machine.EvalExpr.callArgsFault
#audit_empty Slang.Machine.EvalExpr.callArgsExhausted : type_of% @Slang.Machine.EvalExpr.callArgsExhausted

#audit_empty Slang.Machine.EvalArgs.nil : type_of% @Slang.Machine.EvalArgs.nil
#audit_empty Slang.Machine.EvalArgs.cons : type_of% @Slang.Machine.EvalArgs.cons
#audit_empty Slang.Machine.EvalArgs.headFault : type_of% @Slang.Machine.EvalArgs.headFault
#audit_empty Slang.Machine.EvalArgs.headExhausted : type_of% @Slang.Machine.EvalArgs.headExhausted
#audit_empty Slang.Machine.EvalArgs.tailFault : type_of% @Slang.Machine.EvalArgs.tailFault
#audit_empty Slang.Machine.EvalArgs.tailExhausted : type_of% @Slang.Machine.EvalArgs.tailExhausted

#audit_empty Slang.Machine.ExecStmt.bind : type_of% @Slang.Machine.ExecStmt.bind
#audit_empty Slang.Machine.ExecStmt.bindFault : type_of% @Slang.Machine.ExecStmt.bindFault
#audit_empty Slang.Machine.ExecStmt.bindExhausted : type_of% @Slang.Machine.ExecStmt.bindExhausted
#audit_empty Slang.Machine.ExecStmt.store : type_of% @Slang.Machine.ExecStmt.store
#audit_empty Slang.Machine.ExecStmt.storeFault : type_of% @Slang.Machine.ExecStmt.storeFault
#audit_empty Slang.Machine.ExecStmt.storeAddressFault : type_of% @Slang.Machine.ExecStmt.storeAddressFault
#audit_empty Slang.Machine.ExecStmt.storeAddressExhausted : type_of% @Slang.Machine.ExecStmt.storeAddressExhausted
#audit_empty Slang.Machine.ExecStmt.storeValueFault : type_of% @Slang.Machine.ExecStmt.storeValueFault
#audit_empty Slang.Machine.ExecStmt.storeValueExhausted : type_of% @Slang.Machine.ExecStmt.storeValueExhausted
#audit_empty Slang.Machine.ExecStmt.return_ : type_of% @Slang.Machine.ExecStmt.return_
#audit_empty Slang.Machine.ExecStmt.returnFault : type_of% @Slang.Machine.ExecStmt.returnFault
#audit_empty Slang.Machine.ExecStmt.returnExhausted : type_of% @Slang.Machine.ExecStmt.returnExhausted
#audit_empty Slang.Machine.ExecStmt.expression : type_of% @Slang.Machine.ExecStmt.expression
#audit_empty Slang.Machine.ExecStmt.expressionFault : type_of% @Slang.Machine.ExecStmt.expressionFault
#audit_empty Slang.Machine.ExecStmt.expressionExhausted : type_of% @Slang.Machine.ExecStmt.expressionExhausted
#audit_empty Slang.Machine.ExecStmt.ifTrue : type_of% @Slang.Machine.ExecStmt.ifTrue
#audit_empty Slang.Machine.ExecStmt.ifFalse : type_of% @Slang.Machine.ExecStmt.ifFalse
#audit_empty Slang.Machine.ExecStmt.ifFault : type_of% @Slang.Machine.ExecStmt.ifFault
#audit_empty Slang.Machine.ExecStmt.ifExhausted : type_of% @Slang.Machine.ExecStmt.ifExhausted

#audit_empty Slang.Machine.ExecList.nil : type_of% @Slang.Machine.ExecList.nil
#audit_empty Slang.Machine.ExecList.next : type_of% @Slang.Machine.ExecList.next
#audit_empty Slang.Machine.ExecList.returned : type_of% @Slang.Machine.ExecList.returned
#audit_empty Slang.Machine.ExecList.memoryFault : type_of% @Slang.Machine.ExecList.memoryFault
#audit_empty Slang.Machine.ExecList.stackExhausted : type_of% @Slang.Machine.ExecList.stackExhausted
#audit_empty Slang.Machine.EvalCall.run : type_of% @Slang.Machine.EvalCall.run
#audit_empty Slang.Machine.EvalCall.memoryFault : type_of% @Slang.Machine.EvalCall.memoryFault
#audit_empty Slang.Machine.EvalCall.nestedExhausted : type_of% @Slang.Machine.EvalCall.nestedExhausted
#audit_empty Slang.Machine.EvalCall.exhausted : type_of% @Slang.Machine.EvalCall.exhausted

#audit_empty Slang.Machine.emptyContext : Slang.Machine.Context
#audit_empty Slang.Machine.lazy_inline_conditional_does_not_evaluate_untaken_branch : type_of% @Slang.Machine.lazy_inline_conditional_does_not_evaluate_untaken_branch
#audit_empty Slang.Machine.eager_bitOr_evaluates_right_operand : type_of% @Slang.Machine.eager_bitOr_evaluates_right_operand
#audit_empty Slang.Machine.return_stops_execution : type_of% @Slang.Machine.return_stops_execution
#audit_empty Slang.Machine.stack_exhaustion_is_not_a_word_return : type_of% @Slang.Machine.stack_exhaustion_is_not_a_word_return
