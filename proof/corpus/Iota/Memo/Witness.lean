/- Joint non-vacuity witnesses for the fixed-shard memo store. -/
import Iota.Memo.Store

set_option autoImplicit false
set_option warningAsError true
set_option maxRecDepth 10000
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Iota.Memo.Witness

open Iota

/-- A concrete source arena containing one empty source file. -/
def sources : Iota.Source.SourceArena where
  files := [{ path := [], bytes := [], tokens := [] }]
  pathUnique := by
    intro left right _
    cases left with
    | mk leftValue leftBound =>
        cases right with
        | mk rightValue rightBound =>
            apply Fin.ext
            cases leftValue with
            | zero =>
                cases rightValue with
                | zero => rfl
                | succ value =>
                    exact False.elim
                      (Nat.not_lt_zero value (Nat.lt_of_succ_lt_succ rightBound))
            | succ value =>
                exact False.elim
                  (Nat.not_lt_zero value (Nat.lt_of_succ_lt_succ leftBound))

/-- The concrete checked reference covers the empty file exactly. -/
def sourceRef : Iota.Source.SourceRef sources where
  file := ⟨0, by decide⟩
  byteInterval := ⟨0, 0, Nat.le_refl 0, Nat.le_refl 0⟩
  tokenInterval := ⟨0, 0, Nat.le_refl 0, Nat.le_refl 0⟩
  correspondence := rfl

/-- One concrete bounded machine-state arena. -/
def states : Iota.Arena.StateArena 1 where
  entries := [{ environment := [], memory := Iota.Machine.Memory.empty, callDepth := 0 }]
  bounded := by decide

/-- One concrete bounded result arena. -/
def results : Iota.Arena.ResultArena 1 where
  entries := [{ raw := Iota.Word.ofNat 0 }]
  bounded := by decide

/-- The checked handle of the concrete state record. -/
def stateHandle : Iota.Arena.StateHandle states := ⟨0, by decide⟩

/-- The checked handle of the concrete result record. -/
def resultHandle : Iota.Arena.ResultHandle results := ⟨0, by decide⟩

/-- A concrete admitted memo key, including an input-state endpoint and `sigma`. -/
def key : MemoKey sources states where
  source := sourceRef
  operation := "constantReturn"
  context := []
  precedence := 0
  limits := ⟨0, 0, 0, 1, 1⟩
  inputStates := [stateHandle]
  sigma := Iota.Word.ofNat 0
  face := .parabolic
  addressCell := (Iota.Word.ofNat 0, Iota.Word.ofNat 0)
  addressAdmitted := True.intro

/-- The concrete total default memo value. -/
def defaultValue : MemoValue sources states results where
  result := resultHandle
  resultingStates := [stateHandle]
  consumed := sourceRef

/-- The concrete initialized zero-capacity store. -/
def zeroStore : MemoStore sources states results 0 := empty defaultValue

/-- Constructor establishment, total default reads, an absent concrete key, vacuous full layout,
and every candidate insertion's concrete zero-capacity refusal hold jointly for `zeroStore`. -/
theorem zero_constructor_read_refusal :
    RepMemo zeroStore (fun _ => defaultValue) ∧
      (∀ query, readValue zeroStore query = defaultValue) ∧
      findKeySlot (zeroStore.slots (Iota.Address.homeShard key)) key = none ∧
      HomeFull zeroStore key ∧
      ∀ entry : MemoEntry sources states results,
        insert zeroStore entry = (.shardFull : InsertResult zeroStore) := by
  have rep : RepMemo zeroStore (fun _ => defaultValue) := empty_rep defaultValue
  constructor
  · exact rep
  · constructor
    · intro query
      exact read_denotes_get rep query
    · constructor
      · rfl
      · constructor
        · intro slot
          cases slot with
          | mk value bound => exact False.elim (Nat.not_lt_zero value bound)
        · intro entry
          exact insert_zero entry

/-- An explicitly transition-checked entry jointly inhabits the successful write and present-read
premises. The hypotheses expose the checked decode/machine dependency rather than postulating a
concrete `ParsedRef`. -/
theorem success_of_transition
    {someSources : Iota.Source.SourceArena} {stateCapacity resultCapacity : Nat}
    {someStates : Iota.Arena.StateArena stateCapacity}
    {someResults : Iota.Arena.ResultArena resultCapacity}
    (initialDefault : MemoValue someSources someStates someResults)
    (parsed : Iota.Source.ParsedRef someSources)
    (someKey : MemoKey someSources someStates)
    (someValue : MemoValue someSources someStates someResults)
    (parsedSource : parsed.ref = someKey.source)
    (consumedSource : someValue.consumed = someKey.source)
    (transition : TransitionDenotes parsed someKey someValue) :
    ∃ (before after : MemoStore someSources someStates someResults 1)
      (entry : MemoEntry someSources someStates someResults),
      before = empty initialDefault ∧
      entry.key = someKey ∧
      RepMemo before (fun _ => initialDefault) ∧
      readValue before someKey = initialDefault ∧
      insert before entry = .inserted after ∧
      RepMemo after
        (Apeiron.Table.insertWith (fun new _ => new) someKey someValue
          (fun _ => initialDefault)) ∧
      readEntry? after someKey = some entry ∧
      readValue after someKey = someValue := by
  let entry : MemoEntry someSources someStates someResults :=
    { key := someKey
      parsed := parsed
      parsedSource := parsedSource
      value := someValue
      consumedSource := consumedSource
      transition := transition }
  let before : MemoStore someSources someStates someResults 1 := empty initialDefault
  let after : MemoStore someSources someStates someResults 1 :=
    writeSlot before (Iota.Address.homeShard someKey) ⟨0, by decide⟩ entry
  have rep : RepMemo before (fun _ => initialDefault) := empty_rep initialDefault
  have inserted : insert before entry = .inserted after := by
    rfl
  have afterRep := insert_ok_rep rep inserted
  have present := inserted_reads_self rep.1 inserted
  refine ⟨before, after, entry, rfl, rfl, rep, read_denotes_get rep someKey,
    inserted, afterRep, present, ?_⟩
  unfold readValue
  rw [present]

end Iota.Memo.Witness
