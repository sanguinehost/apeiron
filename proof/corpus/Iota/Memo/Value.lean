/-
# The checked memo value and its transition denotation

A memo value is the tuple a memo table stores on a hit: the result handle into the
result arena, the state handles the computation produced, and the source reference
the computation consumed. Every field is rooted in the canonical re-root:
`Iota.Arena.ResultHandle` into `Iota.Arena.ResultArena`,
`Iota.Arena.StateHandle` into `Iota.Arena.StateArena`, and
`Iota.Source.SourceRef` for the consumed source.

`TransitionDenotes` is the checked transition denotation: a memo key and a memo
value denote one `Iota.Machine.EvalCall`. The machine context is
`Iota.Source.ParsedRef.context` of a checked `ParsedRef`, i.e.
`Iota.Refinement.checkedContext` of the program the parsed reference's bytes
decode to. The program is therefore the checked decode of the referenced bytes;
no `default` and no `Option.getD` supply it. The `before` handle is one of the
key's `inputStates` and the `after` handle is one of the value's
`resultingStates`, so a transition witness pins both endpoints to the key and the
value.

`MemoHit` is exactly the existence of a transition witness: there is no hit
predicate that does not go through `TransitionDenotes`. A value whose
`resultingStates` is empty, or a key whose `inputStates` is empty, has no
transition and is therefore not a hit; the rejecting fixtures live in
`Iota.Memo.ValueAudit`.

The cone is empty: the structure, the inductive and the projections use only
constructor fields and `cases`, so nothing here reaches `propext`, `Quot.sound`
or `Classical.choice`.
-/
import Iota.Memo.Key
import Iota.Source.Arena
import Iota.Arena.StateResult
import Iota.Machine

set_option autoImplicit false
set_option warningAsError true
set_option maxRecDepth 10000
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Iota.Memo

open Iota

/-- The checked value stored in a memo table slot: the result handle, the state
handles the computation produced, and the source reference the computation
consumed. -/
structure MemoValue (sources : Iota.Source.SourceArena) {stateCapacity resultCapacity : Nat}
    (states : Iota.Arena.StateArena stateCapacity)
    (results : Iota.Arena.ResultArena resultCapacity) where
  result : Iota.Arena.ResultHandle results
  resultingStates : List (Iota.Arena.StateHandle states)
  consumed : Iota.Source.SourceRef sources

/-- The checked transition denotation: a memo key and a memo value denote one
`EvalCall` executed in the machine context of a checked `ParsedRef`. The
`before` handle is drawn from the key's input states and the `after` handle from
the value's resulting states. -/
inductive TransitionDenotes {sources : Iota.Source.SourceArena}
    {stateCapacity resultCapacity : Nat}
    {states : Iota.Arena.StateArena stateCapacity}
    {results : Iota.Arena.ResultArena resultCapacity}
    (parsed : Iota.Source.ParsedRef sources)
    (key : MemoKey sources states) (value : MemoValue sources states results) : Prop
  | returned (before after : Iota.Arena.StateHandle states) (trace : Iota.Machine.Trace)
      (beforeInput : before ∈ key.inputStates)
      (afterOutput : after ∈ value.resultingStates)
      (execution : Iota.Machine.EvalCall (Iota.Source.ParsedRef.context parsed)
        (.limited key.limits.stack) 0 key.operation (key.context.map (fun entry => entry.2))
        (Iota.Arena.resolve states before).memory
        (.value (Iota.Arena.resolve results value.result).raw
          (Iota.Arena.resolve states after).memory) trace) :
      TransitionDenotes parsed key value

/-- A transition witness supplies an input handle drawn from the key's input
states. -/
theorem TransitionDenotes.before_in_key {sources : Iota.Source.SourceArena}
    {stateCapacity resultCapacity : Nat}
    {states : Iota.Arena.StateArena stateCapacity}
    {results : Iota.Arena.ResultArena resultCapacity}
    {parsed : Iota.Source.ParsedRef sources}
    {key : MemoKey sources states} {value : MemoValue sources states results}
    (transition : TransitionDenotes parsed key value) :
    ∃ before : Iota.Arena.StateHandle states, before ∈ key.inputStates := by
  cases transition with
  | returned before after trace beforeInput afterOutput execution =>
      exact ⟨before, beforeInput⟩

/-- A transition witness supplies an output handle drawn from the value's
resulting states. -/
theorem TransitionDenotes.after_in_value {sources : Iota.Source.SourceArena}
    {stateCapacity resultCapacity : Nat}
    {states : Iota.Arena.StateArena stateCapacity}
    {results : Iota.Arena.ResultArena resultCapacity}
    {parsed : Iota.Source.ParsedRef sources}
    {key : MemoKey sources states} {value : MemoValue sources states results}
    (transition : TransitionDenotes parsed key value) :
    ∃ after : Iota.Arena.StateHandle states, after ∈ value.resultingStates := by
  cases transition with
  | returned before after trace beforeInput afterOutput execution =>
      exact ⟨after, afterOutput⟩

/-- A memo hit is exactly a checked transition: the table answers with `value`
only when the `TransitionDenotes` witness exists. There is no hit predicate that
does not go through the witness. -/
def MemoHit {sources : Iota.Source.SourceArena} {stateCapacity resultCapacity : Nat}
    {states : Iota.Arena.StateArena stateCapacity}
    {results : Iota.Arena.ResultArena resultCapacity}
    (parsed : Iota.Source.ParsedRef sources) (key : MemoKey sources states)
    (value : MemoValue sources states results) : Prop :=
  TransitionDenotes parsed key value

end Iota.Memo
