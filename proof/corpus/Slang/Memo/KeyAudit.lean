/-
# Zero-axiom audit for the memo key

The audit population of `Slang.Memo.Key`: the face and limits types, the input
and key structures, the admission domain, the key constructor, the constructive
decidable-equality instances, the Bool comparison and its correctness theorem.
Every headline is registered in `tools/slang/zero_axioms.json` under the
`memo-key` stage and must report an empty `Lean.collectAxioms` set.

The rejecting fixtures change exactly one data field of a key and prove that
`MemoKey.equal` is then `false`. They are quantified over an arbitrary base key,
so no concrete `SourceRef` has to be exhibited.
-/
import Slang.Memo.Key
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true
set_option maxRecDepth 10000
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Memo.KeyAudit

/-- A positive fixture: `rfl` uses no axiom. The runner's mutation control replaces this proof with
a `Classical.choice` term, so this declaration is the red/restored-green target. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

/-- Changing only `sigma` changes the key, so `MemoKey.equal` is `false`. -/
theorem equal_sigma_reject {sources : Slang.Source.SourceArena} {stateCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} (base : MemoKey sources states)
    (other : Slang.Word) (hne : other ≠ base.sigma) :
    MemoKey.equal base { base with sigma := other } = false := by
  cases h : MemoKey.equal base { base with sigma := other } with
  | true =>
      have heq : base = { base with sigma := other } := (MemoKey.equal_iff base _).mp h
      exact absurd (congrArg MemoKey.sigma heq).symm hne
  | false => rfl

/-- Changing only the `limits` field changes the key, so `MemoKey.equal` is `false`. -/
theorem equal_limits_reject {sources : Slang.Source.SourceArena} {stateCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} (base : MemoKey sources states)
    (limits : Limits) (hne : limits ≠ base.limits) :
    MemoKey.equal base { base with limits := limits } = false := by
  cases h : MemoKey.equal base { base with limits := limits } with
  | true =>
      have heq : base = { base with limits := limits } := (MemoKey.equal_iff base _).mp h
      exact absurd (congrArg MemoKey.limits heq).symm hne
  | false => rfl

/-- Changing only the input state handles changes the key, so `MemoKey.equal` is `false`. -/
theorem equal_states_reject {sources : Slang.Source.SourceArena} {stateCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} (base : MemoKey sources states)
    (handles : List (Slang.Arena.StateHandle states)) (hne : handles ≠ base.inputStates) :
    MemoKey.equal base { base with inputStates := handles } = false := by
  cases h : MemoKey.equal base { base with inputStates := handles } with
  | true =>
      have heq : base = { base with inputStates := handles } :=
        (MemoKey.equal_iff base _).mp h
      exact absurd (congrArg MemoKey.inputStates heq).symm hne
  | false => rfl

end Slang.Memo.KeyAudit

-- The registered population, mirrored in tools/slang/zero_axioms.json. Each line binds the
-- elaborated declaration to its registered type; the command's log lines are the coverage report.
#audit_empty Slang.Memo.Face : type_of% @Slang.Memo.Face
#audit_empty Slang.Memo.Limits : type_of% @Slang.Memo.Limits
#audit_empty Slang.Memo.FaceDomain : type_of% @Slang.Memo.FaceDomain
#audit_empty Slang.Memo.MemoInput : type_of% @Slang.Memo.MemoInput
#audit_empty Slang.Memo.AdmittedInput : type_of% @Slang.Memo.AdmittedInput
#audit_empty Slang.Memo.MemoKey : type_of% @Slang.Memo.MemoKey
#audit_empty Slang.Memo.mkMemoKey : type_of% @Slang.Memo.mkMemoKey
#audit_empty Slang.Memo.instDecidableEqCheckedInterval : type_of% @Slang.Memo.instDecidableEqCheckedInterval
#audit_empty Slang.Memo.instDecidableEqSourceRef : type_of% @Slang.Memo.instDecidableEqSourceRef
#audit_empty Slang.Memo.MemoKey.instDecidableEq : type_of% @Slang.Memo.MemoKey.instDecidableEq
#audit_empty Slang.Memo.MemoKey.equal : type_of% @Slang.Memo.MemoKey.equal
#audit_empty Slang.Memo.MemoKey.equal_iff : type_of% @Slang.Memo.MemoKey.equal_iff
#audit_empty Slang.Memo.KeyAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Memo.KeyAudit.equal_sigma_reject : type_of% @Slang.Memo.KeyAudit.equal_sigma_reject
#audit_empty Slang.Memo.KeyAudit.equal_limits_reject : type_of% @Slang.Memo.KeyAudit.equal_limits_reject
#audit_empty Slang.Memo.KeyAudit.equal_states_reject : type_of% @Slang.Memo.KeyAudit.equal_states_reject
