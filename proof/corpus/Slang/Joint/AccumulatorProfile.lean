import Slang.Table.TableWrite
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-!
# W30 accumulator/default profile

The decoded `tableInsertWith` body has two accumulator paths. This module records the profile rows
that a later accumulator correspondence must respect, and the rejecting check that refuses to equate
the fresh-key path with canonical addition at a nonzero default. It is a specification check: it
states no correspondence and proves none.

The canonical faces are `Apeiron.Table.HMap`, `Apeiron.Table.get` and `Apeiron.Table.insertWith` in
`Apeiron/Table/Basic.lean` at the pinned corpus; the read-back law is
`Apeiron.Table.get_insertWith_self`. The local computation lemmas are `insertWith_apply_self` and
`insertWith_apply_ne` in `Slang/Table/TableWrite.lean`.
-/

namespace Slang.Joint.AccumulatorProfile

open Slang

/-- The runner's mutation anchor: `rfl` uses no axiom, and the runner replaces this proof with a
`Classical.choice` term to require the audit to reject it. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

/-- The decoded fresh-key `MERGE_ACC` path stores the incoming `value` unchanged. -/
def sourceFresh (value : Word) : Word := value

/-- The decoded present-key `MERGE_ACC` path stores `old + value`. -/
def sourcePresent (old value : Word) : Word := Word.add old value

/-- The canonical `Apeiron.Table.insertWith` read-back at the written key is `merge value current`.
At a fresh key `current` is the table default, so the canonical result is `value + default`. -/
def canonicalAtKey (merge : Word → Word → Word) (value current : Word) : Word := merge value current

/-- The default the rejecting row names: nonzero, so canonical addition changes the stored value. -/
def witnessDefault : Word := Word.ofNat 1

/-- A present-key witness old value. -/
def witnessOld : Word := Word.ofNat 5

/-- A shared witness value. -/
def witnessValue : Word := Word.ofNat 7

/-- The profile's classification of a row. -/
inductive Verdict where
  | admitted
  | rejected

/-- One profile row: the decoded source result and the canonical result at a named witness, plus the
specification's verdict. The verdict is the profile's claim; `Row.holds` is the witness fact. -/
structure Row where
  id : String
  sourceResult : Word
  canonicalResult : Word
  verdict : Verdict

/-- The row's witness equality, decided by `Slang.Word`'s Boolean equality. -/
def Row.holds (row : Row) : Bool := row.sourceResult == row.canonicalResult

/-- A row is consistent when its stated verdict matches the witness: an admitted row must hold and a
rejected row must fail. -/
def Row.consistent (row : Row) : Bool :=
  match row.verdict with
  | .admitted => row.holds
  | .rejected => !row.holds

/-- Present-key accumulator row: `old + value` is canonical addition read back at the written key. -/
def presentRow : Row :=
  { id := "present-key-acc-matches-canonical-addition"
    sourceResult := sourcePresent witnessOld witnessValue
    canonicalResult := canonicalAtKey Word.add witnessValue witnessOld
    verdict := .admitted }

/-- Fresh-key accumulator row: the decoded path stores `value`, matching the canonical fresh store. -/
def freshRow : Row :=
  { id := "fresh-key-acc-stores-value"
    sourceResult := sourceFresh witnessValue
    canonicalResult := witnessValue
    verdict := .admitted }

/-- The rejected row: fresh-key ACC insertion equated with canonical addition at a nonzero default.
The witness stores `value`, while canonical addition stores `value + default`, so the row fails. -/
def rejectedRow : Row :=
  { id := "fresh-key-acc-equals-canonical-addition-at-nonzero-default"
    sourceResult := sourceFresh witnessValue
    canonicalResult := canonicalAtKey Word.add witnessValue witnessDefault
    verdict := .rejected }

/-- The W30 profile population. -/
def profile : List Row := [presentRow, freshRow, rejectedRow]

/-- The profile check: every row's stated verdict agrees with its witness. -/
def profileCheck (rows : List Row) : Bool := rows.all (fun row => Row.consistent row)

-- The compiled profile admits the two positive rows and rejects the equating row.
#guard profileCheck profile
#guard presentRow.holds
#guard freshRow.holds
#guard !rejectedRow.holds

/-- Word addition is commutative, so the present-key row is a genuine match. -/
theorem word_add_comm (left right : Word) : Word.add left right = Word.add right left := by
  apply Fin.eq_of_val_eq
  unfold Word.add Word.ofNat
  rw [Nat.add_comm]

/-- The decoded fresh-key path returns `value` unchanged. -/
theorem sourceFresh_eq_value (value : Word) : sourceFresh value = value := rfl

/-- Positive row: the present-key accumulator result is canonical addition read back. -/
theorem present_key_acc_matches_canonical (old value : Word) :
    sourcePresent old value = canonicalAtKey Word.add value old := by
  unfold sourcePresent canonicalAtKey
  exact word_add_comm old value

/-- Positive row: the fresh-key accumulator path stores `value`. -/
theorem fresh_key_acc_stores_value (value : Word) : sourceFresh value = value := rfl

/-- The rejected row is false: no nonzero default makes fresh-key ACC insertion equal canonical
addition at every value. -/
theorem fresh_key_acc_default_addition_refuted :
    ¬ (∀ (default value : Word),
        default ≠ Word.ofNat 0 → sourceFresh value = canonicalAtKey Word.add value default) := by
  intro claimed
  have instance_ := claimed (Word.ofNat 1) (Word.ofNat 0) (by decide)
  unfold sourceFresh canonicalAtKey at instance_
  exact (by decide :
    Word.add (Word.ofNat 0) (Word.ofNat 1) ≠ Word.ofNat 0) instance_.symm

/-- **Rejecting control.** The profile check admits exactly the rows whose verdict matches the
witness, so the equating row is rejected and a profile that admitted it would fail the check. -/
theorem profile_rejects_fresh_key_default_addition :
    profileCheck profile = true ∧ rejectedRow.verdict = Verdict.rejected := ⟨rfl, rfl⟩

end Slang.Joint.AccumulatorProfile

#audit_empty Slang.Joint.AccumulatorProfile.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Joint.AccumulatorProfile.sourceFresh : Slang.Word → Slang.Word
#audit_empty Slang.Joint.AccumulatorProfile.sourcePresent : Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Joint.AccumulatorProfile.canonicalAtKey : (Slang.Word → Slang.Word → Slang.Word) → Slang.Word → Slang.Word → Slang.Word
#audit_empty Slang.Joint.AccumulatorProfile.witnessDefault : Slang.Word
#audit_empty Slang.Joint.AccumulatorProfile.witnessOld : Slang.Word
#audit_empty Slang.Joint.AccumulatorProfile.witnessValue : Slang.Word
#audit_empty Slang.Joint.AccumulatorProfile.Verdict : Type
#audit_empty Slang.Joint.AccumulatorProfile.Row : Type
#audit_empty Slang.Joint.AccumulatorProfile.Row.holds : Slang.Joint.AccumulatorProfile.Row → Bool
#audit_empty Slang.Joint.AccumulatorProfile.Row.consistent : Slang.Joint.AccumulatorProfile.Row → Bool
#audit_empty Slang.Joint.AccumulatorProfile.presentRow : Slang.Joint.AccumulatorProfile.Row
#audit_empty Slang.Joint.AccumulatorProfile.freshRow : Slang.Joint.AccumulatorProfile.Row
#audit_empty Slang.Joint.AccumulatorProfile.rejectedRow : Slang.Joint.AccumulatorProfile.Row
#audit_empty Slang.Joint.AccumulatorProfile.profile : List Slang.Joint.AccumulatorProfile.Row
#audit_empty Slang.Joint.AccumulatorProfile.profileCheck : List Slang.Joint.AccumulatorProfile.Row → Bool
#audit_empty Slang.Joint.AccumulatorProfile.word_add_comm : type_of% @Slang.Joint.AccumulatorProfile.word_add_comm
#audit_empty Slang.Joint.AccumulatorProfile.sourceFresh_eq_value : type_of% @Slang.Joint.AccumulatorProfile.sourceFresh_eq_value
#audit_empty Slang.Joint.AccumulatorProfile.present_key_acc_matches_canonical : type_of% @Slang.Joint.AccumulatorProfile.present_key_acc_matches_canonical
#audit_empty Slang.Joint.AccumulatorProfile.fresh_key_acc_stores_value : type_of% @Slang.Joint.AccumulatorProfile.fresh_key_acc_stores_value
#audit_empty Slang.Joint.AccumulatorProfile.fresh_key_acc_default_addition_refuted : type_of% @Slang.Joint.AccumulatorProfile.fresh_key_acc_default_addition_refuted
#audit_empty Slang.Joint.AccumulatorProfile.profile_rejects_fresh_key_default_addition : type_of% @Slang.Joint.AccumulatorProfile.profile_rejects_fresh_key_default_addition
