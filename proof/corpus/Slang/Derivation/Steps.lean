import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Derivation

/-- A layer exposes recursive calls instead of evaluating them. -/
inductive Layer (Request : Type) (Answer : Request → Type) (α : Type) where
  | done (value : α)
  | ask (request : Request) (next : Answer request → Layer Request Answer α)

def fold {Request : Type} {Answer : Request → Type} {α : Type}
    (evaluate : (request : Request) → Answer request) : Layer Request Answer α → α
  | .done value => value
  | .ask request next => fold evaluate (next (evaluate request))

@[simp] theorem fold_done {Request : Type} {Answer : Request → Type} {α : Type}
    (evaluate : (request : Request) → Answer request) (value : α) :
    fold evaluate (.done value) = value := rfl

@[simp] theorem fold_ask {Request : Type} {Answer : Request → Type} {α : Type}
    (evaluate : (request : Request) → Answer request) (request : Request)
    (next : Answer request → Layer Request Answer α) :
    fold evaluate (.ask request next) = fold evaluate (next (evaluate request)) := rfl

@[simp] theorem fold_ite {Request : Type} {Answer : Request → Type} {α : Type}
    (evaluate : (request : Request) → Answer request) {condition : Prop} [Decidable condition]
    (then_ else_ : Layer Request Answer α) :
    fold evaluate (if condition then then_ else else_) =
      if condition then fold evaluate then_ else fold evaluate else_ := by
  by_cases holds : condition
  · rw [if_pos holds, if_pos holds]
  · rw [if_neg holds, if_neg holds]

@[simp] theorem fold_pair {Request : Type} {Answer : Request → Type} {α : Type}
    (evaluate : (request : Request) → Answer request) {β γ : Type}
    (pair : β × γ) (next : β → γ → Layer Request Answer α) :
    fold evaluate (match pair with | (left, right) => next left right) =
      (match pair with | (left, right) => fold evaluate (next left right)) := by
  cases pair
  rfl

@[simp] theorem fold_except {Request : Type} {Answer : Request → Type} {α : Type}
    (evaluate : (request : Request) → Answer request) {β : Type}
    (result : Except String β) (ok : β → Layer Request Answer α)
    (error : String → Layer Request Answer α) :
    fold evaluate (match result with | .ok value => ok value | .error message => error message) =
      (match result with | .ok value => fold evaluate (ok value) | .error message => fold evaluate (error message)) := by
  cases result <;> rfl

@[simp] theorem fold_option {Request : Type} {Answer : Request → Type} {α : Type}
    (evaluate : (request : Request) → Answer request) {β : Type}
    (option : Option β) (some_ : β → Layer Request Answer α)
    (none_ : Layer Request Answer α) :
    fold evaluate (match option with | some value => some_ value | none => none_) =
      (match option with | some value => fold evaluate (some_ value) | none => fold evaluate none_) := by
  cases option <;> rfl

@[simp] theorem fold_list {Request : Type} {Answer : Request → Type} {α : Type}
    (evaluate : (request : Request) → Answer request) {β : Type}
    (list : List β) (nil_ : Layer Request Answer α)
    (cons_ : β → List β → Layer Request Answer α) :
    fold evaluate (match list with | [] => nil_ | head :: tail => cons_ head tail) =
      (match list with | [] => fold evaluate nil_ | head :: tail => fold evaluate (cons_ head tail)) := by
  cases list <;> rfl

/-- Both the requested subcomputation and its continuation need checked evidence. -/
inductive Derives {Request : Type} {Answer : Request → Type}
    (body : (request : Request) → Layer Request Answer (Answer request)) :
    {α : Type} → Layer Request Answer α → α → Prop where
  | done {α : Type} (value : α) : Derives body (.done value) value
  | ask {α : Type} {request : Request} {next : Answer request → Layer Request Answer α}
      {answer : Answer request} {value : α}
      (called : Derives body (body request) answer)
      (continued : Derives body (next answer) value) :
      Derives body (.ask request next) value

theorem sound {Request : Type} {Answer : Request → Type}
    {body : (request : Request) → Layer Request Answer (Answer request)}
    (evaluate : (request : Request) → Answer request)
    (unfolding : ∀ request, evaluate request = fold evaluate (body request))
    {α : Type} {layer : Layer Request Answer α} {value : α}
    (derivation : Derives body layer value) : fold evaluate layer = value := by
  induction derivation with
  | done => rfl
  | @ask α request next answer value _ _ calledIH continuedIH =>
      change fold evaluate (next (evaluate request)) = value
      exact (congrArg (fun result => fold evaluate (next result))
        ((unfolding request).trans calledIH)).trans continuedIH

theorem call_sound {Request : Type} {Answer : Request → Type}
    {body : (request : Request) → Layer Request Answer (Answer request)}
    (evaluate : (request : Request) → Answer request)
    (unfolding : ∀ request, evaluate request = fold evaluate (body request))
    {request : Request} {answer : Answer request}
    (derivation : Derives body (body request) answer) : evaluate request = answer :=
  (unfolding request).trans (sound evaluate unfolding derivation)

/-- A subcomputation whose continuation returns its answer derives that answer. -/
theorem Derives.ask_done {Request : Type} {Answer : Request → Type}
    {body : (request : Request) → Layer Request Answer (Answer request)}
    {request : Request} {answer : Answer request}
    (called : Derives body (body request) answer) :
    Derives body (.ask request (fun result => .done result)) answer :=
  .ask called (.done answer)

namespace Witness

def body : (request : Nat) → Layer Nat (fun _ => Nat) Nat
  | 0 => .done 0
  | n + 1 => .ask n (fun answer => .done (answer + 1))

theorem unfolding (request : Nat) : request = fold (fun n => n) (body request) := by
  cases request <;> rfl

theorem zero : Derives body (body 0) 0 := .done 0
theorem one : Derives body (body 1) 1 := .ask zero (.done 1)
theorem two : Derives body (body 2) 2 := .ask one (.done 2)

theorem result : (2 : Nat) = 2 := call_sound (fun n => n) unfolding two

theorem wrong_answer : ¬ Derives body (body 2) 3 := by
  intro derivation
  have equal : (2 : Nat) = 3 := call_sound (fun n => n) unfolding derivation
  exact Nat.noConfusion (Nat.succ.inj (Nat.succ.inj equal))

end Witness

#audit_empty Slang.Derivation.fold : {Request : Type} → {Answer : Request → Type} → {α : Type} → ((request : Request) → Answer request) → Layer Request Answer α → α
#audit_empty Slang.Derivation.fold_ite : ∀ {Request : Type} {Answer : Request → Type} {α : Type} (evaluate : (request : Request) → Answer request) {condition : Prop} [Decidable condition] (then_ else_ : Layer Request Answer α), fold evaluate (if condition then then_ else else_) = if condition then fold evaluate then_ else fold evaluate else_
#audit_empty Slang.Derivation.fold_pair : ∀ {Request : Type} {Answer : Request → Type} {α : Type} (evaluate : (request : Request) → Answer request) {β γ : Type} (pair : β × γ) (next : β → γ → Layer Request Answer α), fold evaluate (match pair with | (left, right) => next left right) = match pair with | (left, right) => fold evaluate (next left right)
#audit_empty Slang.Derivation.sound : ∀ {Request : Type} {Answer : Request → Type} {body : (request : Request) → Layer Request Answer (Answer request)} (evaluate : (request : Request) → Answer request), (∀ request, evaluate request = fold evaluate (body request)) → ∀ {α : Type} {layer : Layer Request Answer α} {value : α}, Derives body layer value → fold evaluate layer = value
#audit_empty Slang.Derivation.call_sound : ∀ {Request : Type} {Answer : Request → Type} {body : (request : Request) → Layer Request Answer (Answer request)} (evaluate : (request : Request) → Answer request), (∀ request, evaluate request = fold evaluate (body request)) → ∀ {request : Request} {answer : Answer request}, Derives body (body request) answer → evaluate request = answer
#audit_empty Slang.Derivation.Derives.ask_done : ∀ {Request : Type} {Answer : Request → Type} {body : (request : Request) → Layer Request Answer (Answer request)} {request : Request} {answer : Answer request}, Derives body (body request) answer → Derives body (.ask request fun result => .done result) answer
#audit_empty Slang.Derivation.Witness.unfolding : ∀ request : Nat, request = fold (fun n => n) (Witness.body request)
#audit_empty Slang.Derivation.Witness.zero : Derives Witness.body (Witness.body 0) 0
#audit_empty Slang.Derivation.Witness.one : Derives Witness.body (Witness.body 1) 1
#audit_empty Slang.Derivation.Witness.two : Derives Witness.body (Witness.body 2) 2
#audit_empty Slang.Derivation.Witness.result : (2 : Nat) = 2
#audit_empty Slang.Derivation.Witness.wrong_answer : ¬ Derives Witness.body (Witness.body 2) 3

end Slang.Derivation
