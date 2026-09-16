/-
# Zero-axiom audit for the memo value

The audit population of `Slang.Memo.Value`: the `MemoValue` structure and its
projections, the `TransitionDenotes` relation and its constructor, the input and
output projections, and the `MemoHit` predicate. Every headline is registered in
`tools/slang/zero_axioms.json` under the `memo-value` stage and must report an
empty `Lean.collectAxioms` set.

The rejecting fixtures use the constructor's membership fields. `TransitionDenotes`
requires `before ∈ key.inputStates` and `after ∈ value.resultingStates`. A key or a
value whose list is empty therefore supplies no such handle, the membership field
is uninhabited, and no `TransitionDenotes` exists; a hit, which is exactly a
`TransitionDenotes` witness, is rejected with it. The mechanism is the missing
handle, not a side condition: an absent handle cannot be the constructor's
`before` or `after` witness.

The fixtures are quantified over an arbitrary checked `ParsedRef`, key and value,
so no concrete `SourceRef`, state arena or result arena has to be exhibited.
-/
import Slang.Memo.Value
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true
set_option maxRecDepth 10000
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Memo.ValueAudit

/-- A positive fixture: `rfl` uses no axiom. The runner's mutation control replaces this proof with
a `Classical.choice` term, so this declaration is the red/restored-green target. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

/-- A key that names no input state has no transition: the constructor's
`before ∈ key.inputStates` field is uninhabited. -/
theorem no_transition_of_empty_inputStates {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity}
    {results : Slang.Arena.ResultArena resultCapacity}
    (parsed : Slang.Source.ParsedRef sources) (key : Slang.Memo.MemoKey sources states)
    (value : Slang.Memo.MemoValue sources states results) (empty : key.inputStates = []) :
    ¬ Slang.Memo.TransitionDenotes parsed key value := by
  intro transition
  cases transition with
  | returned before after trace beforeInput afterOutput execution =>
      rw [empty] at beforeInput
      exact List.not_mem_nil beforeInput

/-- A value that names no resulting state has no transition: the constructor's
`after ∈ value.resultingStates` field is uninhabited. A handle absent from the
value cannot be the output witness. -/
theorem no_transition_of_empty_resultingStates {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity}
    {results : Slang.Arena.ResultArena resultCapacity}
    (parsed : Slang.Source.ParsedRef sources) (key : Slang.Memo.MemoKey sources states)
    (value : Slang.Memo.MemoValue sources states results)
    (empty : value.resultingStates = []) :
    ¬ Slang.Memo.TransitionDenotes parsed key value := by
  intro transition
  cases transition with
  | returned before after trace beforeInput afterOutput execution =>
      rw [empty] at afterOutput
      exact List.not_mem_nil afterOutput

/-- A value with no resulting state cannot be a memo hit: `MemoHit` is exactly a
`TransitionDenotes` witness, and the value supplies no output handle. -/
theorem no_hit_of_empty_resultingStates {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity}
    {results : Slang.Arena.ResultArena resultCapacity}
    (parsed : Slang.Source.ParsedRef sources) (key : Slang.Memo.MemoKey sources states)
    (value : Slang.Memo.MemoValue sources states results)
    (empty : value.resultingStates = []) :
    ¬ Slang.Memo.MemoHit parsed key value :=
  no_transition_of_empty_resultingStates parsed key value empty

end Slang.Memo.ValueAudit

-- The registered population, mirrored in tools/slang/zero_axioms.json. Each line binds the
-- elaborated declaration to its registered type; the command's log lines are the coverage report.
#audit_empty Slang.Memo.MemoValue : type_of% @Slang.Memo.MemoValue
#audit_empty Slang.Memo.MemoValue.result : type_of% @Slang.Memo.MemoValue.result
#audit_empty Slang.Memo.MemoValue.resultingStates : type_of% @Slang.Memo.MemoValue.resultingStates
#audit_empty Slang.Memo.MemoValue.consumed : type_of% @Slang.Memo.MemoValue.consumed
#audit_empty Slang.Memo.TransitionDenotes : type_of% @Slang.Memo.TransitionDenotes
#audit_empty Slang.Memo.TransitionDenotes.returned : type_of% @Slang.Memo.TransitionDenotes.returned
#audit_empty Slang.Memo.TransitionDenotes.before_in_key : type_of% @Slang.Memo.TransitionDenotes.before_in_key
#audit_empty Slang.Memo.TransitionDenotes.after_in_value : type_of% @Slang.Memo.TransitionDenotes.after_in_value
#audit_empty Slang.Memo.MemoHit : type_of% @Slang.Memo.MemoHit
#audit_empty Slang.Memo.ValueAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Memo.ValueAudit.no_transition_of_empty_inputStates : type_of% @Slang.Memo.ValueAudit.no_transition_of_empty_inputStates
#audit_empty Slang.Memo.ValueAudit.no_transition_of_empty_resultingStates : type_of% @Slang.Memo.ValueAudit.no_transition_of_empty_resultingStates
#audit_empty Slang.Memo.ValueAudit.no_hit_of_empty_resultingStates : type_of% @Slang.Memo.ValueAudit.no_hit_of_empty_resultingStates
