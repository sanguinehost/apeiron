import Iota.Joint.SongTable
import Iota.Audit.Empty

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-!
# W29 rejecting control

The W29 control changes the decoded `MERGE_MAP` face from replacement to addition. On the replacement
face the decoded fresh-key path binds `newVal` to `value`, so the checked write stores `value` and
the canonical map is `Apeiron.Table.insertWith (fun new _old => new)`. On the addition face the fresh
path binds `newVal` to `oldVal + value`, so the map is `Apeiron.Table.insertWith (fun new old =>
Word.add old new)`. The two maps differ at the written key whenever the old value is nonzero, so the
correspondence headline's replacement conclusion cannot be reached by the addition face.

`insertWithBodyAddition` is the concrete throwaway variant of the decoded body; the `#guard` below
records that it is not the decoded replacement body, and `control_addition_not_replacement` is the
kernel-checked rejection. The file also audits the `SongTable` headlines with an empty
`Lean.collectAxioms` set.
-/

namespace Iota.Joint

open Iota Iota.Decode Iota.Machine Iota.Source Iota.Refinement Iota.Table

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

/-- The throwaway addition variant of the decoded `tableInsertWith` body: the fresh-key path binds
`newVal` to `oldVal + value` instead of `value`. Kept only as the rejected alternative the control
names; no theorem is stated about it. -/
def insertWithBodyAddition : List Stmt :=
  [.if_ (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0))
    [.return_ (.name "TABLE_EMPTY")] none,
   .if_ (.binary .eq (.name "key") (.name "TABLE_EMPTY"))
    [.return_ (.name "TABLE_EMPTY")] none,
   .if_ (.binary .bitOr
      (.binary .lt (.name "merge") (.name "MERGE_SET"))
      (.binary .gt (.name "merge") (.name "MERGE_ACC")))
    [.return_ (.name "TABLE_EMPTY")] none,
   .bind "mask" (.binary .sub (.name "cap") (.literal 1)),
   .bind "initSlot" (.binary .bitAnd (.call "tableSpread" [.name "key"])
     (.name "mask")),
   .bind "slot" (.call "tableFindSlotGo"
     [.name "base", .name "cap", .name "mask", .name "key", .name "initSlot",
      .literal 0]),
   .if_ (.binary .lt (.name "slot") (.literal 0))
     [.return_ (.binary .sub (.literal 0) (.literal 1))] none,
   .bind "oldKey" (.call "tableSlotKey" [.name "base", .name "slot"]),
   .bind "oldVal" (.call "tableSlotVal" [.name "base", .name "slot"]),
   .if_ (.binary .eq (.name "oldKey") (.name "TABLE_EMPTY"))
     [.bind "newVal" (.binary .add (.name "oldVal") (.name "value")),
      .expression (.call "tableSlotSet"
        [.name "base", .name "slot", .name "key", .name "newVal"]),
      .return_ (.name "newVal")] none,
   .bind "newVal" (.binary .add (.name "oldVal") (.name "value")),
   .if_ (.binary .eq (.name "merge") (.name "MERGE_ACC"))
     [.bind "newVal" (.binary .add (.name "oldVal") (.name "value"))] none,
   .return_ (.call "tableSlotValSet"
     [.name "base", .name "slot", .name "newVal"])]

/-- The throwaway addition variant is not the decoded replacement body.

Temporal O(1): both operands are fixed `List Stmt` literals, so the derived `List Stmt` `BEq` runs a
constant number of structural comparisons (list spines, `Stmt` tags and fields, each nested `Expr`
tree) before the first mismatch. Spatial O(1): the `BEq` stack is bounded by the literals' fixed
nesting depth. Operations counted: structural equality comparisons. -/
def additionVariantDiffers : Bool := insertWithBodyAddition != insertWithBody

#guard additionVariantDiffers

def controlKey : Word := Word.ofNat 1
def controlValue : Word := Word.ofNat 7
def controlDefault : Word := Word.ofNat 5
def controlDefaultMap : Word → Word := fun _ => controlDefault

/-- The canonical map the replacement face produces at the written key. -/
def controlReplacementMap : Word → Word :=
  Apeiron.Table.insertWith (fun new _old => new) controlKey controlValue controlDefaultMap

/-- The canonical map an addition face would produce at the written key. -/
def controlAdditionMap : Word → Word :=
  Apeiron.Table.insertWith (fun new old => Word.add old new) controlKey controlValue
    controlDefaultMap

theorem controlReplacementMap_key : controlReplacementMap controlKey = controlValue := by
  unfold controlReplacementMap
  exact insertWith_replacement_self controlKey controlValue controlDefaultMap

theorem controlAdditionMap_key :
    controlAdditionMap controlKey = Word.add controlDefault controlValue := by
  unfold controlAdditionMap
  exact insertWith_apply_self (fun new old => Word.add old new) controlKey controlValue
    controlDefaultMap

theorem control_add_default_value_ne_value :
    Word.add controlDefault controlValue ≠ controlValue := by
  intro equal
  have hval := congrArg Fin.val equal
  rw [Word.wrapping_add_agreement] at hval
  have hdefault : controlDefault.val = 5 := ofNat_val_emptyCone 5 (by decide)
  have hvalue : controlValue.val = 7 := ofNat_val_emptyCone 7 (by decide)
  rw [hdefault, hvalue] at hval
  have hmod : (5 + 7) % wordModulus = 5 + 7 := modEqOfLt (by decide)
  rw [hmod] at hval
  exact absurd hval (by decide)

/-- **Rejecting control.** The addition-face map is not the replacement-face map, so a
correspondence conclusion naming `Apeiron.Table.insertWith (fun new _old => new)` is refuted once
the decoded `MERGE_MAP` face is changed to addition. -/
theorem control_addition_not_replacement : controlAdditionMap ≠ controlReplacementMap := by
  intro equal
  have h := congrFun equal controlKey
  rw [controlAdditionMap_key, controlReplacementMap_key] at h
  exact control_add_default_value_ne_value h

end Iota.Joint

#audit_empty Iota.Joint.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Joint.insertOccupied_of_member_true : type_of% @Iota.Joint.insertOccupied_of_member_true
#audit_empty Iota.Joint.insertOccupied_of_member_false : type_of% @Iota.Joint.insertOccupied_of_member_false
#audit_empty Iota.Joint.observeRead_get : type_of% @Iota.Joint.observeRead_get
#audit_empty Iota.Joint.songTable_replacement_link : type_of% @Iota.Joint.songTable_replacement_link
#audit_empty Iota.Joint.control_addition_not_replacement : type_of% @Iota.Joint.control_addition_not_replacement
