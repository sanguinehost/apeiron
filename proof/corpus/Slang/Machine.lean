import Slang.Syntax
import Slang.Word

set_option autoImplicit true

namespace Slang.Machine

open Slang

abbrev Environment := List (String × Word)

namespace Environment

def lookup (environment : Environment) (name : String) : Option Word :=
  (environment.find? (fun entry => entry.1 == name)).map (·.2)

def bind (environment : Environment) (name : String) (value : Word) : Environment :=
  (name, value) :: environment

def ofParameters : List String → List Word → Option Environment
  | [], [] => some []
  | name :: names, value :: values =>
      ((name, value) :: ·) <$> ofParameters names values
  | _, _ => none

@[simp] theorem lookup_bind_same (environment : Environment) (name : String) (value : Word) :
    lookup (bind environment name value) name = some value := by
  simp [lookup, bind]

theorem lookup_bind_other (environment : Environment) (name other : String) (value : Word)
    (hne : other ≠ name) : lookup (bind environment name value) other = lookup environment other := by
  have hne' : name ≠ other := Ne.symm hne
  simp [lookup, bind, hne']

end Environment

structure Memory where
  owned : Nat → Bool
  initialized : Nat → Option Word

namespace Memory

def coordinate (address : Word) : Option Nat :=
  if Word.signed address < 0 then none else some address.val

def load (memory : Memory) (address : Word) : Option Word := do
  let coordinate ← coordinate address
  if memory.owned coordinate then memory.initialized coordinate else none

def store (memory : Memory) (address value : Word) : Option Memory := do
  let coordinate ← coordinate address
  if memory.owned coordinate then
    some { memory with initialized := Function.update memory.initialized coordinate (some value) }
  else none

def empty : Memory := ⟨fun _ => false, fun _ => none⟩

theorem store_ownership (memory next : Memory) (address value : Word)
    (hstore : store memory address value = some next) : next.owned = memory.owned := by
  rcases hcoordinate : coordinate address with _ | coordinate
  · simp [store, hcoordinate] at hstore
  · by_cases howned : memory.owned coordinate = true
    · simp [store, hcoordinate, howned] at hstore
      subst next
      rfl
    · simp [store, hcoordinate, howned] at hstore

theorem store_at (memory next : Memory) (address value : Word) (coordinate : Nat)
    (hcoordinate : Memory.coordinate address = some coordinate)
    (hstore : store memory address value = some next) : next.initialized coordinate = some value := by
  simp [store, hcoordinate] at hstore
  rcases hstore with ⟨_, rfl⟩
  simp

theorem store_other (memory next : Memory) (address value : Word) (coordinate other : Nat)
    (hcoordinate : Memory.coordinate address = some coordinate)
    (hstore : store memory address value = some next) (hne : other ≠ coordinate) :
    next.initialized other = memory.initialized other := by
  simp [store, hcoordinate] at hstore
  rcases hstore with ⟨_, rfl⟩
  simp [Function.update, hne]

end Memory

inductive StackPolicy where
  | unbounded
  | limited (entries : Nat)
  deriving Repr, BEq

def StackPolicy.allows : StackPolicy → Nat → Prop
  | .unbounded, _ => True
  | .limited entries, depth => depth < entries

structure Context where
  constants : String → Option Word
  functions : String → Option FunctionDecl

def Context.ofProgram (program : Program) (constants : String → Option Word) : Context where
  constants := constants
  functions := fun name => program.functions.find? (fun decl => decl.name == name)

inductive Event where
  | arithmetic (operator : BinOp)
  | comparison (operator : BinOp)
  | bitwise (operator : BinOp)
  | load (address : Word)
  | store (address : Word)
  | call (callee : String) (depth : Nat)
  deriving Repr, BEq

abbrev Trace := List Event

structure Resources where
  arithmeticOps : Nat
  comparisons : Nat
  bitwiseOps : Nat
  loads : Nat
  stores : Nat
  calls : Nat
  probeIterations : Nat
  maxCallStack : Nat
  allocatedNodes : Nat
  externalStreamConsumption : Nat
  deriving Repr, BEq

private def isProbe : String → Bool
  | "tableFindSlotGo" | "tableLookupSlotGo" => true
  | _ => false

def emptyResources : Resources := ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩

private def account (result : Resources) (event : Event) : Resources :=
  match event with
  | .arithmetic _ => { result with arithmeticOps := result.arithmeticOps + 1 }
  | .comparison _ => { result with comparisons := result.comparisons + 1 }
  | .bitwise _ => { result with bitwiseOps := result.bitwiseOps + 1 }
  | .load _ => { result with loads := result.loads + 1 }
  | .store _ => { result with stores := result.stores + 1 }
  | .call callee depth =>
      { result with
        calls := result.calls + 1
        probeIterations := result.probeIterations + if isProbe callee then 1 else 0
        maxCallStack := max result.maxCallStack depth }

def resources (trace : Trace) : Resources := trace.foldl account emptyResources

private theorem fold_allocatedNodes (trace : Trace) (initial : Resources) :
    (trace.foldl account initial).allocatedNodes = initial.allocatedNodes := by
  induction trace generalizing initial with
  | nil => rfl
  | cons event rest ih =>
      simp only [List.foldl_cons]
      rw [ih]
      cases event <;> rfl

private theorem fold_externalStreamConsumption (trace : Trace) (initial : Resources) :
    (trace.foldl account initial).externalStreamConsumption = initial.externalStreamConsumption := by
  induction trace generalizing initial with
  | nil => rfl
  | cons event rest ih =>
      simp only [List.foldl_cons]
      rw [ih]
      cases event <;> rfl

@[simp] theorem resources_allocatedNodes (trace : Trace) : (resources trace).allocatedNodes = 0 := by
  exact fold_allocatedNodes trace emptyResources

@[simp] theorem resources_externalStreamConsumption (trace : Trace) :
    (resources trace).externalStreamConsumption = 0 := by
  exact fold_externalStreamConsumption trace emptyResources

inductive EvalResult (α : Type) where
  | value (value : α) (memory : Memory)
  | memoryFault (memory : Memory)
  | stackExhausted (memory : Memory)

inductive Control where
  | next (environment : Environment) (memory : Memory)
  | returned (raw : Word) (memory : Memory)
  | memoryFault (memory : Memory)
  | stackExhausted (memory : Memory)

def applyBinOp : BinOp → Word → Word → Word
  | .add => Word.add
  | .sub => Word.sub
  | .mul => Word.mul
  | .bitAnd => Word.bitAnd
  | .bitOr => Word.bitOr
  | .bitXor => Word.bitXor
  | .eq => Word.eq
  | .ne => Word.ne
  | .lt => Word.lt
  | .le => Word.le
  | .gt => Word.gt
  | .ge => Word.ge

def operatorEvent : BinOp → Event
  | .add => .arithmetic .add
  | .sub => .arithmetic .sub
  | .mul => .arithmetic .mul
  | .bitAnd => .bitwise .bitAnd
  | .bitOr => .bitwise .bitOr
  | .bitXor => .bitwise .bitXor
  | .eq => .comparison .eq
  | .ne => .comparison .ne
  | .lt => .comparison .lt
  | .le => .comparison .le
  | .gt => .comparison .gt
  | .ge => .comparison .ge

mutual
  inductive EvalExpr (context : Context) (policy : StackPolicy) :
      Nat → Environment → Memory → Expr → EvalResult Word → Trace → Prop where
    | literal : EvalExpr context policy depth env memory (.literal value)
        (.value (Word.ofNat value) memory) []
    | local (hlookup : Environment.lookup env name = some value) :
        EvalExpr context policy depth env memory (.name name) (.value value memory) []
    | constant (hlocals : Environment.lookup env name = none)
        (hlookup : context.constants name = some value) :
        EvalExpr context policy depth env memory (.name name) (.value value memory) []
    | load (haddress : EvalExpr context policy depth env memory address (.value addressValue next) ta)
        (hload : next.load addressValue = some value) :
        EvalExpr context policy depth env memory (.memory address) (.value value next)
          (ta ++ [.load addressValue])
    | loadFault (haddress : EvalExpr context policy depth env memory address (.value addressValue next) ta)
        (hload : next.load addressValue = none) :
        EvalExpr context policy depth env memory (.memory address) (.memoryFault next)
          (ta ++ [.load addressValue])
    | memoryAddressFault
        (haddress : EvalExpr context policy depth env memory address (.memoryFault next) trace) :
        EvalExpr context policy depth env memory (.memory address) (.memoryFault next) trace
    | memoryAddressExhausted
        (haddress : EvalExpr context policy depth env memory address (.stackExhausted next) trace) :
        EvalExpr context policy depth env memory (.memory address) (.stackExhausted next) trace
    | binary (hleft : EvalExpr context policy depth env memory left (.value leftValue leftMemory) tl)
        (hright : EvalExpr context policy depth env leftMemory right (.value rightValue rightMemory) tr) :
        EvalExpr context policy depth env memory (.binary operator left right)
          (.value (applyBinOp operator leftValue rightValue) rightMemory)
          (tl ++ tr ++ [operatorEvent operator])
    | binaryLeftFault
        (hleft : EvalExpr context policy depth env memory left (.memoryFault next) trace) :
        EvalExpr context policy depth env memory (.binary operator left right) (.memoryFault next) trace
    | binaryLeftExhausted
        (hleft : EvalExpr context policy depth env memory left (.stackExhausted next) trace) :
        EvalExpr context policy depth env memory (.binary operator left right) (.stackExhausted next) trace
    | binaryRightFault
        (hleft : EvalExpr context policy depth env memory left (.value leftValue leftMemory) tl)
        (hright : EvalExpr context policy depth env leftMemory right (.memoryFault next) tr) :
        EvalExpr context policy depth env memory (.binary operator left right) (.memoryFault next) (tl ++ tr)
    | binaryRightExhausted
        (hleft : EvalExpr context policy depth env memory left (.value leftValue leftMemory) tl)
        (hright : EvalExpr context policy depth env leftMemory right (.stackExhausted next) tr) :
        EvalExpr context policy depth env memory (.binary operator left right) (.stackExhausted next) (tl ++ tr)
    | conditionalTrue
        (hcondition : EvalExpr context policy depth env memory condition (.value conditionValue next) tc)
        (htrue : Word.nonzero conditionValue)
        (hbranch : EvalExpr context policy depth env next thenValue result tb) :
        EvalExpr context policy depth env memory (.conditional thenValue condition elseValue) result (tc ++ tb)
    | conditionalFalse
        (hcondition : EvalExpr context policy depth env memory condition (.value conditionValue next) tc)
        (hfalse : ¬ Word.nonzero conditionValue)
        (hbranch : EvalExpr context policy depth env next elseValue result tb) :
        EvalExpr context policy depth env memory (.conditional thenValue condition elseValue) result (tc ++ tb)
    | conditionalFault
        (hcondition : EvalExpr context policy depth env memory condition (.memoryFault next) trace) :
        EvalExpr context policy depth env memory (.conditional thenValue condition elseValue)
          (.memoryFault next) trace
    | conditionalExhausted
        (hcondition : EvalExpr context policy depth env memory condition (.stackExhausted next) trace) :
        EvalExpr context policy depth env memory (.conditional thenValue condition elseValue)
          (.stackExhausted next) trace
    | call (hargs : EvalArgs context policy depth env memory arguments (.value values next) ta)
        (hcall : EvalCall context policy depth callee values next result tc) :
        EvalExpr context policy depth env memory (.call callee arguments) result (ta ++ tc)
    | callArgsFault
        (hargs : EvalArgs context policy depth env memory arguments (.memoryFault next) trace) :
        EvalExpr context policy depth env memory (.call callee arguments) (.memoryFault next) trace
    | callArgsExhausted
        (hargs : EvalArgs context policy depth env memory arguments (.stackExhausted next) trace) :
        EvalExpr context policy depth env memory (.call callee arguments) (.stackExhausted next) trace

  inductive EvalArgs (context : Context) (policy : StackPolicy) :
      Nat → Environment → Memory → List Expr → EvalResult (List Word) → Trace → Prop where
    | nil : EvalArgs context policy depth env memory [] (.value [] memory) []
    | cons (hhead : EvalExpr context policy depth env memory head (.value value next) th)
        (htail : EvalArgs context policy depth env next tail (.value values final) tt) :
        EvalArgs context policy depth env memory (head :: tail) (.value (value :: values) final) (th ++ tt)
    | headFault (hhead : EvalExpr context policy depth env memory head (.memoryFault next) trace) :
        EvalArgs context policy depth env memory (head :: tail) (.memoryFault next) trace
    | headExhausted (hhead : EvalExpr context policy depth env memory head (.stackExhausted next) trace) :
        EvalArgs context policy depth env memory (head :: tail) (.stackExhausted next) trace
    | tailFault (hhead : EvalExpr context policy depth env memory head (.value value next) th)
        (htail : EvalArgs context policy depth env next tail (.memoryFault final) tt) :
        EvalArgs context policy depth env memory (head :: tail) (.memoryFault final) (th ++ tt)
    | tailExhausted (hhead : EvalExpr context policy depth env memory head (.value value next) th)
        (htail : EvalArgs context policy depth env next tail (.stackExhausted final) tt) :
        EvalArgs context policy depth env memory (head :: tail) (.stackExhausted final) (th ++ tt)

  inductive ExecStmt (context : Context) (policy : StackPolicy) :
      Nat → Environment → Memory → Stmt → Control → Trace → Prop where
    | bind (heval : EvalExpr context policy depth env memory expression (.value value next) trace) :
        ExecStmt context policy depth env memory (.bind name expression)
          (.next (env.bind name value) next) trace
    | bindFault (heval : EvalExpr context policy depth env memory expression (.memoryFault next) trace) :
        ExecStmt context policy depth env memory (.bind name expression) (.memoryFault next) trace
    | bindExhausted (heval : EvalExpr context policy depth env memory expression (.stackExhausted next) trace) :
        ExecStmt context policy depth env memory (.bind name expression) (.stackExhausted next) trace
    | store (haddress : EvalExpr context policy depth env memory address (.value addressValue afterAddress) ta)
        (hvalue : EvalExpr context policy depth env afterAddress value (.value stored afterValue) tv)
        (hstore : afterValue.store addressValue stored = some next) :
        ExecStmt context policy depth env memory (.store address value) (.next env next)
          (ta ++ tv ++ [.store addressValue])
    | storeFault (haddress : EvalExpr context policy depth env memory address (.value addressValue afterAddress) ta)
        (hvalue : EvalExpr context policy depth env afterAddress value (.value stored afterValue) tv)
        (hstore : afterValue.store addressValue stored = none) :
        ExecStmt context policy depth env memory (.store address value) (.memoryFault afterValue)
          (ta ++ tv ++ [.store addressValue])
    | storeAddressFault
        (haddress : EvalExpr context policy depth env memory address (.memoryFault next) trace) :
        ExecStmt context policy depth env memory (.store address value) (.memoryFault next) trace
    | storeAddressExhausted
        (haddress : EvalExpr context policy depth env memory address (.stackExhausted next) trace) :
        ExecStmt context policy depth env memory (.store address value) (.stackExhausted next) trace
    | storeValueFault
        (haddress : EvalExpr context policy depth env memory address (.value addressValue afterAddress) ta)
        (hvalue : EvalExpr context policy depth env afterAddress value (.memoryFault next) tv) :
        ExecStmt context policy depth env memory (.store address value) (.memoryFault next) (ta ++ tv)
    | storeValueExhausted
        (haddress : EvalExpr context policy depth env memory address (.value addressValue afterAddress) ta)
        (hvalue : EvalExpr context policy depth env afterAddress value (.stackExhausted next) tv) :
        ExecStmt context policy depth env memory (.store address value) (.stackExhausted next) (ta ++ tv)
    | return_ (heval : EvalExpr context policy depth env memory expression (.value value next) trace) :
        ExecStmt context policy depth env memory (.return_ expression) (.returned value next) trace
    | returnFault (heval : EvalExpr context policy depth env memory expression (.memoryFault next) trace) :
        ExecStmt context policy depth env memory (.return_ expression) (.memoryFault next) trace
    | returnExhausted (heval : EvalExpr context policy depth env memory expression (.stackExhausted next) trace) :
        ExecStmt context policy depth env memory (.return_ expression) (.stackExhausted next) trace
    | expression (heval : EvalExpr context policy depth env memory value (.value resultWord next) trace) :
        ExecStmt context policy depth env memory (.expression value) (.next env next) trace
    | expressionFault (heval : EvalExpr context policy depth env memory value (.memoryFault next) trace) :
        ExecStmt context policy depth env memory (.expression value) (.memoryFault next) trace
    | expressionExhausted (heval : EvalExpr context policy depth env memory value (.stackExhausted next) trace) :
        ExecStmt context policy depth env memory (.expression value) (.stackExhausted next) trace
    | ifTrue (hcondition : EvalExpr context policy depth env memory condition (.value conditionValue next) tc)
        (htrue : Word.nonzero conditionValue)
        (hbody : ExecList context policy depth env next thenBody result tb) :
        ExecStmt context policy depth env memory (.if_ condition thenBody elseBody) result (tc ++ tb)
    | ifFalse (hcondition : EvalExpr context policy depth env memory condition (.value conditionValue next) tc)
        (hfalse : ¬ Word.nonzero conditionValue)
        (hbody : ExecList context policy depth env next (elseBody.getD []) result tb) :
        ExecStmt context policy depth env memory (.if_ condition thenBody elseBody) result (tc ++ tb)
    | ifFault (hcondition : EvalExpr context policy depth env memory condition (.memoryFault next) trace) :
        ExecStmt context policy depth env memory (.if_ condition thenBody elseBody) (.memoryFault next) trace
    | ifExhausted (hcondition : EvalExpr context policy depth env memory condition (.stackExhausted next) trace) :
        ExecStmt context policy depth env memory (.if_ condition thenBody elseBody) (.stackExhausted next) trace

  inductive ExecList (context : Context) (policy : StackPolicy) :
      Nat → Environment → Memory → List Stmt → Control → Trace → Prop where
    | nil : ExecList context policy depth env memory [] (.next env memory) []
    | next (hstmt : ExecStmt context policy depth env memory statement (.next nextEnv nextMemory) ts)
        (hrest : ExecList context policy depth nextEnv nextMemory rest result tr) :
        ExecList context policy depth env memory (statement :: rest) result (ts ++ tr)
    | returned (hstmt : ExecStmt context policy depth env memory statement (.returned value next) trace) :
        ExecList context policy depth env memory (statement :: rest) (.returned value next) trace
    | memoryFault (hstmt : ExecStmt context policy depth env memory statement (.memoryFault next) trace) :
        ExecList context policy depth env memory (statement :: rest) (.memoryFault next) trace
    | stackExhausted (hstmt : ExecStmt context policy depth env memory statement (.stackExhausted next) trace) :
        ExecList context policy depth env memory (statement :: rest) (.stackExhausted next) trace

  inductive EvalCall (context : Context) (policy : StackPolicy) :
      Nat → String → List Word → Memory → EvalResult Word → Trace → Prop where
    | run (hfunction : context.functions callee = some function)
        (hframe : Environment.ofParameters function.parameters arguments = some frame)
        (hcapacity : policy.allows (depth + 1))
        (hbody : ExecList context policy (depth + 1) frame memory function.body
          (.returned value next) trace) :
        EvalCall context policy depth callee arguments memory (.value value next)
          (.call callee (depth + 1) :: trace)
    | memoryFault (hfunction : context.functions callee = some function)
        (hframe : Environment.ofParameters function.parameters arguments = some frame)
        (hcapacity : policy.allows (depth + 1))
        (hbody : ExecList context policy (depth + 1) frame memory function.body
          (.memoryFault next) trace) :
        EvalCall context policy depth callee arguments memory (.memoryFault next)
          (.call callee (depth + 1) :: trace)
    | nestedExhausted (hfunction : context.functions callee = some function)
        (hframe : Environment.ofParameters function.parameters arguments = some frame)
        (hcapacity : policy.allows (depth + 1))
        (hbody : ExecList context policy (depth + 1) frame memory function.body
          (.stackExhausted next) trace) :
        EvalCall context policy depth callee arguments memory (.stackExhausted next)
          (.call callee (depth + 1) :: trace)
    | exhausted (hfunction : context.functions callee = some function)
        (hcapacity : ¬ policy.allows (depth + 1)) :
        EvalCall context policy depth callee arguments memory (.stackExhausted memory)
          [.call callee (depth + 1)]
end

def emptyContext : Context := ⟨fun _ => none, fun _ => none⟩

theorem lazy_inline_conditional_does_not_evaluate_untaken_branch :
    EvalExpr emptyContext .unbounded 0 [] Memory.empty
      (.conditional (.literal 7) (.literal 1) (.memory (.literal 0)))
      (.value (Word.ofNat 7) Memory.empty) [] := by
  apply EvalExpr.conditionalTrue (hcondition := EvalExpr.literal)
  · norm_num [Word.nonzero, Word.ofNat, wordModulus, wordBits]
  · exact EvalExpr.literal

theorem eager_bitOr_evaluates_right_operand :
    EvalExpr emptyContext .unbounded 0 [] Memory.empty
      (.binary .bitOr (.literal 1) (.memory (.literal 0)))
      (.memoryFault Memory.empty) [.load (Word.ofNat 0)] := by
  apply EvalExpr.binaryRightFault (hleft := EvalExpr.literal)
  apply EvalExpr.loadFault (haddress := EvalExpr.literal)
  simp [Memory.load, Memory.coordinate, Memory.empty, Word.signed, Word.ofNat,
    wordSignBit, wordModulus, wordBits]

theorem return_stops_execution (context : Context) (policy : StackPolicy) (depth : Nat)
    (environment : Environment) (memory next : Memory) (expression : Expr) (value : Word)
    (trace : Trace) (rest : List Stmt)
    (heval : EvalExpr context policy depth environment memory expression (.value value next) trace) :
    ExecList context policy depth environment memory (.return_ expression :: rest)
      (.returned value next) trace :=
  ExecList.returned (ExecStmt.return_ heval)

theorem stack_exhaustion_is_not_a_word_return (context : Context) (function : FunctionDecl)
    (callee : String) (arguments : List Word) (memory : Memory)
    (hfunction : context.functions callee = some function) :
    EvalCall context (.limited 0) 0 callee arguments memory (.stackExhausted memory)
      [.call callee 1] := by
  apply EvalCall.exhausted hfunction
  simp [StackPolicy.allows]

end Slang.Machine
