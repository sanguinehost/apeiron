import Slang.Syntax
import Slang.Word

set_option autoImplicit false

namespace Slang.Machine

open Slang

abbrev Environment := List (String × Word)

namespace Environment

def lookup : Environment → String → Option Word
  | [], _ => none
  | (entryName, value) :: rest, name =>
      match String.decEq entryName name with
      | isTrue _ => some value
      | isFalse _ => lookup rest name

def bind (environment : Environment) (name : String) (value : Word) : Environment :=
  (name, value) :: environment

def ofParameters : List String → List Word → Option Environment
  | [], values => match values with | [] => some [] | _ :: _ => none
  | name :: names, values =>
      match values with
      | [] => none
      | value :: rest =>
          match ofParameters names rest with
          | some environment => some ((name, value) :: environment)
          | none => none

@[simp] theorem lookup_bind_same (environment : Environment) (name : String) (value : Word) :
    lookup (bind environment name value) name = some value := by
  unfold bind
  rw [lookup]
  cases heq : String.decEq name name with
  | isTrue => rfl
  | isFalse hne => exact False.elim (hne rfl)

theorem lookup_bind_other (environment : Environment) (name other : String) (value : Word)
    (hne : other ≠ name) : lookup (bind environment name value) other = lookup environment other := by
  unfold bind
  rw [lookup]
  cases heq : String.decEq name other with
  | isTrue hequal => exact False.elim (hne (Eq.symm hequal))
  | isFalse => rfl

end Environment

structure Memory where
  owned : Nat → Bool
  initialized : Nat → Option Word

def SameOwnership (before after : Memory) : Prop :=
  ∀ q, after.owned q = before.owned q

def MemoryEquiv (before after : Memory) : Prop :=
  SameOwnership before after ∧ ∀ q, after.initialized q = before.initialized q

namespace Memory

def coordinate (address : Word) : Option Nat :=
  if Word.signed address < 0 then none else some address.val

def load (memory : Memory) (address : Word) : Option Word := do
  let coordinate ← coordinate address
  if memory.owned coordinate then memory.initialized coordinate else none

def store (memory : Memory) (address value : Word) : Option Memory := do
  let writtenCoordinate ← coordinate address
  if memory.owned writtenCoordinate then
    some {
      owned := memory.owned
      initialized := fun q =>
        if q = writtenCoordinate then some value else memory.initialized q
    }
  else none

def empty : Memory := ⟨fun _ => false, fun _ => none⟩

theorem store_ownership (memory next : Memory) (address value : Word)
    (hstore : store memory address value = some next) : next.owned = memory.owned := by
  unfold store at hstore
  cases hcoordinate : coordinate address with
  | none =>
      rw [hcoordinate] at hstore
      contradiction
  | some coordinate =>
      rw [hcoordinate] at hstore
      change (if memory.owned coordinate then some {
        owned := memory.owned
        initialized := fun q =>
          if q = coordinate then some value else memory.initialized q
      } else none) = some next at hstore
      cases howned : memory.owned coordinate with
      | false =>
          rw [howned] at hstore
          contradiction
      | true =>
          rw [howned] at hstore
          cases hstore
          rfl

theorem store_at (memory next : Memory) (address value : Word) (coordinate : Nat)
    (hcoordinate : Memory.coordinate address = some coordinate)
    (hstore : store memory address value = some next) : next.initialized coordinate = some value := by
  unfold store at hstore
  rw [hcoordinate] at hstore
  change (if memory.owned coordinate then some {
    owned := memory.owned
    initialized := fun q =>
      if q = coordinate then some value else memory.initialized q
  } else none) = some next at hstore
  cases howned : memory.owned coordinate with
  | false =>
      rw [howned] at hstore
      contradiction
  | true =>
      rw [howned] at hstore
      cases hstore
      change (if coordinate = coordinate then some value else memory.initialized coordinate) = some value
      exact if_pos rfl

theorem store_other (memory next : Memory) (address value : Word) (coordinate other : Nat)
    (hcoordinate : Memory.coordinate address = some coordinate)
    (hstore : store memory address value = some next) (hne : other ≠ coordinate) :
    next.initialized other = memory.initialized other := by
  unfold store at hstore
  rw [hcoordinate] at hstore
  change (if memory.owned coordinate then some {
    owned := memory.owned
    initialized := fun q =>
      if q = coordinate then some value else memory.initialized q
  } else none) = some next at hstore
  cases howned : memory.owned coordinate with
  | false =>
      rw [howned] at hstore
      contradiction
  | true =>
      rw [howned] at hstore
      cases hstore
      change (if other = coordinate then some value else memory.initialized other) =
        memory.initialized other
      exact if_neg hne

end Memory

inductive StackPolicy where
  | unbounded
  | limited (entries : Nat)
  deriving Repr

def stackPolicyBeq (left right : StackPolicy) : Bool :=
  match left with
  | .unbounded => match right with | .unbounded => true | .limited _ => false
  | .limited leftEntries =>
      match right with
      | .unbounded => false
      | .limited rightEntries => Nat.beq leftEntries rightEntries

instance instBEqStackPolicy : BEq StackPolicy := ⟨stackPolicyBeq⟩

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

def binOpTag : BinOp → Nat
  | .add => 0 | .sub => 1 | .mul => 2
  | .bitAnd => 3 | .bitOr => 4 | .bitXor => 5
  | .eq => 6 | .ne => 7 | .lt => 8 | .le => 9 | .gt => 10 | .ge => 11

def stringBeq (left right : String) : Bool :=
  match String.decEq left right with
  | isTrue _ => true
  | isFalse _ => false

instance instReprEvent : Repr Event where
  reprPrec event _ :=
    match event with
    | .arithmetic _ => "Event.arithmetic"
    | .comparison _ => "Event.comparison"
    | .bitwise _ => "Event.bitwise"
    | .load _ => "Event.load"
    | .store _ => "Event.store"
    | .call _ _ => "Event.call"

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

def isProbe : String → Bool
  | "tableFindSlotGo" | "tableLookupSlotGo" => true
  | _ => false

def emptyResources : Resources := ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩

def account (result : Resources) (event : Event) : Resources :=
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

def resources : Trace → Resources
  | [] => emptyResources
  | event :: rest => account (resources rest) event

def arithmeticCount : Trace → Nat
  | [] => 0
  | event :: rest => match event with
      | .arithmetic _ => arithmeticCount rest + 1
      | .comparison _ | .bitwise _ | .load _ | .store _ | .call _ _ => arithmeticCount rest

def comparisonCount : Trace → Nat
  | [] => 0
  | event :: rest => match event with
      | .comparison _ => comparisonCount rest + 1
      | .arithmetic _ | .bitwise _ | .load _ | .store _ | .call _ _ => comparisonCount rest

def bitwiseCount : Trace → Nat
  | [] => 0
  | event :: rest => match event with
      | .bitwise _ => bitwiseCount rest + 1
      | .arithmetic _ | .comparison _ | .load _ | .store _ | .call _ _ => bitwiseCount rest

def loadCount : Trace → Nat
  | [] => 0
  | event :: rest => match event with
      | .load _ => loadCount rest + 1
      | .arithmetic _ | .comparison _ | .bitwise _ | .store _ | .call _ _ => loadCount rest

def storeCount : Trace → Nat
  | [] => 0
  | event :: rest => match event with
      | .store _ => storeCount rest + 1
      | .arithmetic _ | .comparison _ | .bitwise _ | .load _ | .call _ _ => storeCount rest

def callCount : Trace → Nat
  | [] => 0
  | event :: rest => match event with
      | .call _ _ => callCount rest + 1
      | .arithmetic _ | .comparison _ | .bitwise _ | .load _ | .store _ => callCount rest

def probeIterationCount : Trace → Nat
  | [] => 0
  | event :: rest => match event with
      | .call callee _ => probeIterationCount rest + if isProbe callee then 1 else 0
      | .arithmetic _ | .comparison _ | .bitwise _ | .load _ | .store _ => probeIterationCount rest

def callStackMaximum : Trace → Nat
  | [] => 0
  | event :: rest => match event with
      | .call _ depth => max (callStackMaximum rest) depth
      | .arithmetic _ | .comparison _ | .bitwise _ | .load _ | .store _ => callStackMaximum rest

@[simp] theorem resources_arithmeticOps (trace : Trace) :
    (resources trace).arithmeticOps = arithmeticCount trace := by
  induction trace with
  | nil => rfl
  | cons event rest ih =>
      cases event with
      | arithmetic _ => exact congrArg (fun count => count + 1) ih
      | comparison _ | bitwise _ | load _ | store _ | call _ _ => exact ih

@[simp] theorem resources_comparisons (trace : Trace) :
    (resources trace).comparisons = comparisonCount trace := by
  induction trace with
  | nil => rfl
  | cons event rest ih =>
      cases event with
      | comparison _ => exact congrArg (fun count => count + 1) ih
      | arithmetic _ | bitwise _ | load _ | store _ | call _ _ => exact ih

@[simp] theorem resources_bitwiseOps (trace : Trace) :
    (resources trace).bitwiseOps = bitwiseCount trace := by
  induction trace with
  | nil => rfl
  | cons event rest ih =>
      cases event with
      | bitwise _ => exact congrArg (fun count => count + 1) ih
      | arithmetic _ | comparison _ | load _ | store _ | call _ _ => exact ih

@[simp] theorem resources_loads (trace : Trace) :
    (resources trace).loads = loadCount trace := by
  induction trace with
  | nil => rfl
  | cons event rest ih =>
      cases event with
      | load _ => exact congrArg (fun count => count + 1) ih
      | arithmetic _ | comparison _ | bitwise _ | store _ | call _ _ => exact ih

@[simp] theorem resources_stores (trace : Trace) :
    (resources trace).stores = storeCount trace := by
  induction trace with
  | nil => rfl
  | cons event rest ih =>
      cases event with
      | store _ => exact congrArg (fun count => count + 1) ih
      | arithmetic _ | comparison _ | bitwise _ | load _ | call _ _ => exact ih

@[simp] theorem resources_calls (trace : Trace) :
    (resources trace).calls = callCount trace := by
  induction trace with
  | nil => rfl
  | cons event rest ih =>
      cases event with
      | call _ _ => exact congrArg (fun count => count + 1) ih
      | arithmetic _ | comparison _ | bitwise _ | load _ | store _ => exact ih

@[simp] theorem resources_probeIterations (trace : Trace) :
    (resources trace).probeIterations = probeIterationCount trace := by
  induction trace with
  | nil => rfl
  | cons event rest ih =>
      cases event with
      | call callee _ => exact congrArg (fun count => count + if isProbe callee then 1 else 0) ih
      | arithmetic _ | comparison _ | bitwise _ | load _ | store _ => exact ih

@[simp] theorem resources_maxCallStack (trace : Trace) :
    (resources trace).maxCallStack = callStackMaximum trace := by
  induction trace with
  | nil => rfl
  | cons event rest ih =>
      cases event with
      | call _ depth => exact congrArg (fun largest => max largest depth) ih
      | arithmetic _ | comparison _ | bitwise _ | load _ | store _ => exact ih

@[simp] theorem resources_allocatedNodes (trace : Trace) : (resources trace).allocatedNodes = 0 := by
  induction trace with
  | nil => rfl
  | cons event rest ih => cases event <;> exact ih

@[simp] theorem resources_externalStreamConsumption (trace : Trace) :
    (resources trace).externalStreamConsumption = 0 := by
  induction trace with
  | nil => rfl
  | cons event rest ih => cases event <;> exact ih

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

set_option autoImplicit true in
mutual
  inductive EvalExpr (context : Context) (policy : StackPolicy) :
      Nat → Environment → Memory → Expr → EvalResult Word → Trace → Prop where
    | literal : EvalExpr context policy depth env memory (.literal literalValue)
        (.value (Word.ofNat literalValue) memory) []
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
        (hbranch : EvalExpr context policy depth env next thenValue evalResult tb) :
        EvalExpr context policy depth env memory (.conditional thenValue condition elseValue) evalResult (tc ++ tb)
    | conditionalFalse
        (hcondition : EvalExpr context policy depth env memory condition (.value conditionValue next) tc)
        (hfalse : ¬ Word.nonzero conditionValue)
        (hbranch : EvalExpr context policy depth env next elseValue evalResult tb) :
        EvalExpr context policy depth env memory (.conditional thenValue condition elseValue) evalResult (tc ++ tb)
    | conditionalFault
        (hcondition : EvalExpr context policy depth env memory condition (.memoryFault next) trace) :
        EvalExpr context policy depth env memory (.conditional thenValue condition elseValue)
          (.memoryFault next) trace
    | conditionalExhausted
        (hcondition : EvalExpr context policy depth env memory condition (.stackExhausted next) trace) :
        EvalExpr context policy depth env memory (.conditional thenValue condition elseValue)
          (.stackExhausted next) trace
    | call (hargs : EvalArgs context policy depth env memory arguments (.value values next) ta)
        (hcall : EvalCall context policy depth callee values next evalResult tc) :
        EvalExpr context policy depth env memory (.call callee arguments) evalResult (ta ++ tc)
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
        (hvalue : EvalExpr context policy depth env afterAddress valueExpr (.value stored afterValue) tv)
        (hstore : afterValue.store addressValue stored = some next) :
        ExecStmt context policy depth env memory (.store address valueExpr) (.next env next)
          (ta ++ tv ++ [.store addressValue])
    | storeFault (haddress : EvalExpr context policy depth env memory address (.value addressValue afterAddress) ta)
        (hvalue : EvalExpr context policy depth env afterAddress valueExpr (.value stored afterValue) tv)
        (hstore : afterValue.store addressValue stored = none) :
        ExecStmt context policy depth env memory (.store address valueExpr) (.memoryFault afterValue)
          (ta ++ tv ++ [.store addressValue])
    | storeAddressFault
        (haddress : EvalExpr context policy depth env memory address (.memoryFault next) trace) :
        ExecStmt context policy depth env memory (.store address valueExpr) (.memoryFault next) trace
    | storeAddressExhausted
        (haddress : EvalExpr context policy depth env memory address (.stackExhausted next) trace) :
        ExecStmt context policy depth env memory (.store address valueExpr) (.stackExhausted next) trace
    | storeValueFault
        (haddress : EvalExpr context policy depth env memory address (.value addressValue afterAddress) ta)
        (hvalue : EvalExpr context policy depth env afterAddress valueExpr (.memoryFault next) tv) :
        ExecStmt context policy depth env memory (.store address valueExpr) (.memoryFault next) (ta ++ tv)
    | storeValueExhausted
        (haddress : EvalExpr context policy depth env memory address (.value addressValue afterAddress) ta)
        (hvalue : EvalExpr context policy depth env afterAddress valueExpr (.stackExhausted next) tv) :
        ExecStmt context policy depth env memory (.store address valueExpr) (.stackExhausted next) (ta ++ tv)
    | return_ (heval : EvalExpr context policy depth env memory expression (.value value next) trace) :
        ExecStmt context policy depth env memory (.return_ expression) (.returned value next) trace
    | returnFault (heval : EvalExpr context policy depth env memory expression (.memoryFault next) trace) :
        ExecStmt context policy depth env memory (.return_ expression) (.memoryFault next) trace
    | returnExhausted (heval : EvalExpr context policy depth env memory expression (.stackExhausted next) trace) :
        ExecStmt context policy depth env memory (.return_ expression) (.stackExhausted next) trace
    | expression (heval : EvalExpr context policy depth env memory expressionValue (.value resultWord next) trace) :
        ExecStmt context policy depth env memory (.expression expressionValue) (.next env next) trace
    | expressionFault (heval : EvalExpr context policy depth env memory expressionValue (.memoryFault next) trace) :
        ExecStmt context policy depth env memory (.expression expressionValue) (.memoryFault next) trace
    | expressionExhausted (heval : EvalExpr context policy depth env memory expressionValue (.stackExhausted next) trace) :
        ExecStmt context policy depth env memory (.expression expressionValue) (.stackExhausted next) trace
    | ifTrue (hcondition : EvalExpr context policy depth env memory condition (.value conditionValue next) tc)
        (htrue : Word.nonzero conditionValue)
        (hbody : ExecList context policy depth env next thenBody controlResult tb) :
        ExecStmt context policy depth env memory (.if_ condition thenBody elseBody) controlResult (tc ++ tb)
    | ifFalse (hcondition : EvalExpr context policy depth env memory condition (.value conditionValue next) tc)
        (hfalse : ¬ Word.nonzero conditionValue)
        (hbody : ExecList context policy depth env next (elseBody.getD []) controlResult tb) :
        ExecStmt context policy depth env memory (.if_ condition thenBody elseBody) controlResult (tc ++ tb)
    | ifFault (hcondition : EvalExpr context policy depth env memory condition (.memoryFault next) trace) :
        ExecStmt context policy depth env memory (.if_ condition thenBody elseBody) (.memoryFault next) trace
    | ifExhausted (hcondition : EvalExpr context policy depth env memory condition (.stackExhausted next) trace) :
        ExecStmt context policy depth env memory (.if_ condition thenBody elseBody) (.stackExhausted next) trace

  inductive ExecList (context : Context) (policy : StackPolicy) :
      Nat → Environment → Memory → List Stmt → Control → Trace → Prop where
    | nil : ExecList context policy depth env memory [] (.next env memory) []
    | next (hstmt : ExecStmt context policy depth env memory statement (.next nextEnv nextMemory) ts)
        (hrest : ExecList context policy depth nextEnv nextMemory rest controlResult tr) :
        ExecList context policy depth env memory (statement :: rest) controlResult (ts ++ tr)
    | returned (hstmt : ExecStmt context policy depth env memory statement (.returned value next) trace) :
        ExecList context policy depth env memory (statement :: rest) (.returned value next) trace
    | memoryFault (hstmt : ExecStmt context policy depth env memory statement (.memoryFault next) trace) :
        ExecList context policy depth env memory (statement :: rest) (.memoryFault next) trace
    | stackExhausted (hstmt : ExecStmt context policy depth env memory statement (.stackExhausted next) trace) :
        ExecList context policy depth env memory (statement :: rest) (.stackExhausted next) trace

  inductive EvalCall (context : Context) (policy : StackPolicy) :
      Nat → String → List Word → Memory → EvalResult Word → Trace → Prop where
    | run (hfunction : context.functions callee = some function)
        (hframe : Environment.ofParameters function.parameters argumentValues = some frame)
        (hcapacity : policy.allows (depth + 1))
        (hbody : ExecList context policy (depth + 1) frame memory function.body
          (.returned value next) trace) :
        EvalCall context policy depth callee argumentValues memory (.value value next)
          (.call callee (depth + 1) :: trace)
    | memoryFault (hfunction : context.functions callee = some function)
        (hframe : Environment.ofParameters function.parameters argumentValues = some frame)
        (hcapacity : policy.allows (depth + 1))
        (hbody : ExecList context policy (depth + 1) frame memory function.body
          (.memoryFault next) trace) :
        EvalCall context policy depth callee argumentValues memory (.memoryFault next)
          (.call callee (depth + 1) :: trace)
    | nestedExhausted (hfunction : context.functions callee = some function)
        (hframe : Environment.ofParameters function.parameters argumentValues = some frame)
        (hcapacity : policy.allows (depth + 1))
        (hbody : ExecList context policy (depth + 1) frame memory function.body
          (.stackExhausted next) trace) :
        EvalCall context policy depth callee argumentValues memory (.stackExhausted next)
          (.call callee (depth + 1) :: trace)
    | exhausted (hfunction : context.functions callee = some function)
        (hcapacity : ¬ policy.allows (depth + 1)) :
        EvalCall context policy depth callee argumentValues memory (.stackExhausted memory)
          [.call callee (depth + 1)]
end

def emptyContext : Context := ⟨fun _ => none, fun _ => none⟩

theorem lazy_inline_conditional_does_not_evaluate_untaken_branch :
    EvalExpr emptyContext .unbounded 0 [] Memory.empty
      (.conditional (.literal 7) (.literal 1) (.memory (.literal 0)))
      (.value (Word.ofNat 7) Memory.empty) [] := by
  apply EvalExpr.conditionalTrue (hcondition := EvalExpr.literal)
  · intro heq
    exact Nat.noConfusion (congrArg Fin.val heq)
  · exact EvalExpr.literal

theorem eager_bitOr_evaluates_right_operand :
    EvalExpr emptyContext .unbounded 0 [] Memory.empty
      (.binary .bitOr (.literal 1) (.memory (.literal 0)))
      (.memoryFault Memory.empty) [.load (Word.ofNat 0)] := by
  apply EvalExpr.binaryRightFault (hleft := EvalExpr.literal)
  apply EvalExpr.loadFault (haddress := EvalExpr.literal)
  rfl

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
  exact Nat.not_lt_zero 1

end Slang.Machine
