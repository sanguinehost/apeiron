import Aeneas.Std.Primitives

/-
SCOPE: finite observations of the actual loop body and exact result/count projection.
REGISTER: generic model/count infrastructure; no Rust/source cost or GPU theorem.
CORPUS: Std.Primitives.loop; WP.loop.spec_decr_nat does not carry a count.
REFERENCE: actual pinned Aeneas/Std/Primitives.lean.
GATE: python3 proof/native/loop_trace_gate.py --controls, whole invocation540s.
GRADE: generic model proof on fresh kernel/audits and actual done-branch control.
-/
set_option autoImplicit false

namespace LoopTrace
universe u v
open Aeneas.Std Result

/-- A finite certificate observes each actual body invocation and charges its state.
    Failure preserves its exact error and charge; divergence has no constructor. -/
inductive Trace {State : Type u} {Value : Type v}
    (body : State → Result (ControlFlow State Value)) (charge : State → Nat) :
    State → Result Value → Nat → Prop where
  | done {state value} (observed : body state = ok (ControlFlow.done value)) :
      Trace body charge state (ok value) (charge state)
  | failure {state error} (observed : body state = fail error) :
      Trace body charge state (fail error) (charge state)
  | cont {state next result count}
      (observed : body state = ok (ControlFlow.cont next))
      (tail : Trace body charge next result count) :
      Trace body charge state result (charge state + count)

/-- The finite certificate projects to the actual partial-fixpoint loop, not an evaluator copy. -/
theorem projection {State : Type u} {Value : Type v}
    {body : State → Result (ControlFlow State Value)} {charge : State → Nat}
    {state : State} {result : Result Value} {count : Nat}
    (trace : Trace body charge state result count) : loop body state = result := by
  induction trace with
  | done observed =>
      unfold loop
      rw [observed]
      simp only [bind_ok]
  | failure observed =>
      unfold loop
      rw [observed]
      simp only [bind_fail]
  | cont observed tail ih =>
      unfold loop
      rw [observed]
      simp only [bind_ok]
      exact ih

/-- Fixed body and charge observations determine both the exact outcome and total charge. -/
theorem deterministic {State : Type u} {Value : Type v}
    {body : State → Result (ControlFlow State Value)} {charge : State → Nat}
    {state : State} {result₁ result₂ : Result Value} {count₁ count₂ : Nat}
    (first : Trace body charge state result₁ count₁)
    (second : Trace body charge state result₂ count₂) :
    result₁ = result₂ ∧ count₁ = count₂ := by
  induction first generalizing result₂ count₂ with
  | done observed =>
      cases second <;> simp_all
  | failure observed =>
      cases second <;> simp_all
  | @cont state next result count observed tail ih =>
      cases second with
      | done observed₂ => simp_all
      | failure observed₂ => simp_all
      | @cont state₂ next₂ result₂ count₂ observed₂ tail₂ =>
          have eq : next = next₂ := by
            simpa using observed.symm.trans observed₂
          cases eq
          obtain ⟨hr, hc⟩ := ih tail₂
          exact ⟨hr, by omega⟩

/-- Finite observations cannot project to divergence. -/
theorem excludes_divergence {State : Type u} {Value : Type v}
    {body : State → Result (ControlFlow State Value)} {charge : State → Nat}
    {state : State} {result : Result Value} {count : Nat}
    (trace : Trace body charge state result count) : result ≠ div := by
  induction trace with
  | done _ => exact ok_not_div
  | failure _ => exact fail_not_div
  | cont _ _ ih => exact ih

def successBody : Bool → Result (ControlFlow Bool Nat)
  | false => ok (.done 7)
  | true => ok (.cont false)

def failureBody : Bool → Result (ControlFlow Bool Nat)
  | false => fail .divisionByZero
  | true => ok (.cont false)

def unitCharge (_ : Bool) : Nat := 1

theorem success0 : Trace successBody unitCharge false (ok 7) 1 :=
  Trace.done rfl

theorem success_cont : Trace successBody unitCharge true (ok 7) 2 :=
  Trace.cont rfl success0

theorem failure0 : Trace failureBody unitCharge false (fail .divisionByZero) 1 :=
  Trace.failure rfl

theorem later_failure : Trace failureBody unitCharge true (fail .divisionByZero) 2 :=
  Trace.cont rfl failure0

/-- All joint typed success/refusal witnesses inhabit the same finite model API. -/
theorem joint_population :
    Trace successBody unitCharge false (ok 7) 1 ∧
    Trace successBody unitCharge true (ok 7) 2 ∧
    Trace failureBody unitCharge false (fail .divisionByZero) 1 ∧
    Trace failureBody unitCharge true (fail .divisionByZero) 2 :=
  ⟨success0, success_cont, failure0, later_failure⟩

theorem actual_success0 : loop successBody false = ok 7 := projection success0

theorem actual_success_cont : loop successBody true = ok 7 := projection success_cont

theorem actual_failure0 : loop failureBody false = fail .divisionByZero := projection failure0

theorem actual_later_failure : loop failureBody true = fail .divisionByZero := projection later_failure

def countdownBody : Nat → Result (ControlFlow Nat Unit)
  | 0 => ok (.done ())
  | n + 1 => ok (.cont n)

def countdownCharge : Nat → Nat
  | 0 => 0
  | _ + 1 => 1

/-- Induction supplies the count; the terminal body's zero charge is still included. -/
theorem countdown_trace (n : Nat) : Trace countdownBody countdownCharge n (ok ()) n := by
  induction n with
  | zero => exact Trace.done rfl
  | succ n ih =>
      have trace := Trace.cont (body := countdownBody) (charge := countdownCharge)
        (state := n + 1) rfl ih
      simpa only [countdownCharge, Nat.add_comm] using trace

theorem countdown_count_unique (n count : Nat)
    (trace : Trace countdownBody countdownCharge n (ok ()) count) : count = n :=
  (deterministic trace (countdown_trace n)).2

theorem countdown32 : Trace countdownBody countdownCharge 32 (ok ()) 32 := countdown_trace 32

theorem actual_countdown (n : Nat) : loop countdownBody n = ok () := projection (countdown_trace n)

end LoopTrace

/- Audit every owned declaration, including generated constructors and all witnesses. -/
open Lean Elab Command in
elab "#loop_trace_audit" : command => do
  let environment ← getEnv
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  let constants := environment.constants.fold (fun acc constant _ =>
    if (`LoopTrace).isPrefixOf constant then constant :: acc else acc) ([] : List Lean.Name)
  if constants.length < 20 then throwError "LOOP_TRACE_AUDIT_POPULATION"
  let mut failed := false
  for constant in constants do
    let dependencies ← liftTermElabM <| Lean.collectAxioms constant
    let bad := dependencies.filter (fun value => !allowed.contains value)
    unless bad.isEmpty do
      logError m!"{constant}: forbidden dependencies {bad.toList}"
      failed := true
    logInfo m!"{constant}: dependencies {dependencies.toList}"
  if failed then throwError "LOOP_TRACE_CLOSED_CONE_REJECT"
  logInfo m!"LOOP_TRACE_KERNEL_OK declarations={constants.length}"

#loop_trace_audit
