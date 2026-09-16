/- Ordered composition of admitted memo updates and its canonical table denotation. -/
import Slang.Memo.Store
import Apeiron.Table.Basic

set_option autoImplicit false
set_option warningAsError true
set_option maxRecDepth 10000
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Memo

/-- Apply transition-checked entries in list order, stopping at the first full shard. -/
def fold {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity} :
    MemoStore sources states results N → List (MemoEntry sources states results) →
      Option (MemoStore sources states results N)
  | store, [] => some store
  | store, entry :: rest =>
      match insert store entry with
      | .inserted after => fold after rest
      | .shardFull => none

/-- Applying identical canonical replacement streams preserves equality at an introduced key. -/
theorem canonical_fold_preserves_point {K V : Type} [DecidableEq K]
    (updates : List (K × V)) (left right : Apeiron.Table.HMap K V) (key : K)
    (same : Apeiron.Table.get left key = Apeiron.Table.get right key) :
    Apeiron.Table.get
        (updates.foldl (fun h update =>
          Apeiron.Table.insertWith (fun new _ => new) update.1 update.2 h) left) key =
      Apeiron.Table.get
        (updates.foldl (fun h update =>
          Apeiron.Table.insertWith (fun new _ => new) update.1 update.2 h) right) key := by
  induction updates generalizing left right with
  | nil => exact same
  | cons update rest ih =>
      apply ih
      unfold Apeiron.Table.get Apeiron.Table.insertWith
      cases equal : decEq key update.1 with
      | isTrue keyEqual =>
          change (if key = update.1 then update.2 else left key) =
            (if key = update.1 then update.2 else right key)
          rw [if_pos keyEqual]
          rw [if_pos keyEqual]
      | isFalse keyDifferent =>
          change (if key = update.1 then update.2 else left key) =
            (if key = update.1 then update.2 else right key)
          rw [if_neg keyDifferent]
          rw [if_neg keyDifferent]
          exact same

/-- Mapping checked entries to key-value pairs changes no canonical fold observation. -/
theorem entry_fold_stream_pointwise {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity}
    {results : Slang.Arena.ResultArena resultCapacity}
    (updates : List (MemoEntry sources states results))
    (h : Apeiron.Table.HMap (MemoKey sources states) (MemoValue sources states results))
    (key : MemoKey sources states) :
    Apeiron.Table.get
        (updates.foldl (fun map entry =>
          Apeiron.Table.insertWith (fun new _ => new) entry.key entry.value map) h) key =
      Apeiron.Table.get
        ((updates.map (fun entry => (entry.key, entry.value))).foldl (fun map update =>
          Apeiron.Table.insertWith (fun new _ => new) update.1 update.2 map) h) key := by
  induction updates generalizing h with
  | nil => rfl
  | cons entry rest ih =>
      exact ih (Apeiron.Table.insertWith (fun new _ => new) entry.key entry.value h)

/-- A canonical replacement fold from an arbitrary map is, at each key, the canonical build
started from that map's value at the same key. -/
theorem canonical_fold_denotes_build_at {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity}
    {results : Slang.Arena.ResultArena resultCapacity}
    (updates : List (MemoEntry sources states results))
    (h : Apeiron.Table.HMap (MemoKey sources states) (MemoValue sources states results))
    (key : MemoKey sources states) :
    Apeiron.Table.get
        (updates.foldl (fun map entry =>
          Apeiron.Table.insertWith (fun new _ => new) entry.key entry.value map) h) key =
      Apeiron.Table.get
        (Apeiron.Table.build (fun new _ => new) (Apeiron.Table.get h key)
          (updates.map (fun entry => (entry.key, entry.value)))) key := by
  rw [entry_fold_stream_pointwise updates h key]
  unfold Apeiron.Table.build
  exact canonical_fold_preserves_point
    (updates.map (fun entry => (entry.key, entry.value))) h
    (fun _ => Apeiron.Table.get h key) key rfl

/-- Successful physical folding preserves representation of the canonical insertion accumulator. -/
theorem fold_success_rep_acc {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity}
    {results : Slang.Arena.ResultArena resultCapacity}
    {updates : List (MemoEntry sources states results)}
    {store after : MemoStore sources states results N}
    {h : Apeiron.Table.HMap (MemoKey sources states) (MemoValue sources states results)}
    (rep : RepMemo store h) (success : fold store updates = some after) :
    RepMemo after
      (updates.foldl (fun map entry =>
        Apeiron.Table.insertWith (fun new _ => new) entry.key entry.value map) h) := by
  induction updates generalizing store after h with
  | nil =>
      unfold fold at success
      cases success
      exact rep
  | cons entry rest ih =>
      unfold fold at success
      cases inserted : insert store entry with
      | inserted next =>
          rw [inserted] at success
          exact ih (insert_ok_rep rep inserted) success
      | shardFull =>
          rw [inserted] at success
          cases success

/-- The empty physical fold denotes the empty canonical ordered build. -/
theorem empty_fold_denotes_build {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity}
    {results : Slang.Arena.ResultArena resultCapacity}
    (defaultValue : MemoValue sources states results) :
    fold (empty (N := N) defaultValue) [] = some (empty (N := N) defaultValue) ∧
      RepMemo (empty (N := N) defaultValue)
        (Apeiron.Table.build (fun new _ => new) defaultValue []) := by
  constructor
  · rfl
  · exact empty_rep defaultValue

/-- A successful physical fold has the canonical ordered-build observation at every key. -/
theorem fold_success_denotes_build_at {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity}
    {results : Slang.Arena.ResultArena resultCapacity}
    {updates : List (MemoEntry sources states results)}
    {store after : MemoStore sources states results N}
    {h : Apeiron.Table.HMap (MemoKey sources states) (MemoValue sources states results)}
    (rep : RepMemo store h) (success : fold store updates = some after)
    (key : MemoKey sources states) :
    denote after key =
      Apeiron.Table.get
        (Apeiron.Table.build (fun new _ => new) (Apeiron.Table.get h key)
          (updates.map (fun entry => (entry.key, entry.value)))) key := by
  have accumulated := fold_success_rep_acc rep success
  rw [accumulated.2 key]
  exact canonical_fold_denotes_build_at updates h key

/-- With a constant initial denotation, successful folding preserves `RepMemo` against canonical
ordered `build`. The explicit premise is necessary because `build` takes one scalar default. -/
theorem fold_success_rep {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity}
    {results : Slang.Arena.ResultArena resultCapacity}
    {updates : List (MemoEntry sources states results)}
    {store after : MemoStore sources states results N}
    {h : Apeiron.Table.HMap (MemoKey sources states) (MemoValue sources states results)}
    (defaultValue : MemoValue sources states results) (rep : RepMemo store h)
    (initial : ∀ key, Apeiron.Table.get h key = defaultValue)
    (success : fold store updates = some after) :
    RepMemo after
      (Apeiron.Table.build (fun new _ => new) defaultValue
        (updates.map (fun entry => (entry.key, entry.value)))) := by
  have accumulated := fold_success_rep_acc rep success
  constructor
  · exact accumulated.1
  · intro key
    rw [accumulated.2 key]
    rw [canonical_fold_denotes_build_at updates h key]
    rw [initial key]

/-- Reversing two same-key updates changes an ordered build for decimal append. -/
theorem control_reversal_changes_build :
    Apeiron.Table.get
        (Apeiron.Table.build (fun new old : Nat => old * 10 + new) 0
          [(true, 1), (true, 2)]) true = 12 ∧
      Apeiron.Table.get
        (Apeiron.Table.build (fun new old : Nat => old * 10 + new) 0
          [(true, 2), (true, 1)]) true = 21 ∧
      Apeiron.Table.get
          (Apeiron.Table.build (fun new old : Nat => old * 10 + new) 0
            [(true, 1), (true, 2)]) true ≠
        Apeiron.Table.get
          (Apeiron.Table.build (fun new old : Nat => old * 10 + new) 0
            [(true, 2), (true, 1)]) true := by
  decide

end Slang.Memo
