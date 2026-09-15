import Slang.Table.TableAdmission
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Table

open Slang Slang.Decode Slang.Machine Slang.Source Slang.Refinement

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

theorem nat_lt_two_cases (value : Nat) (bound : value < 2) : value = 0 ∨ value = 1 := by
  cases value with
  | zero => exact Or.inl rfl
  | succ value =>
      cases value with
      | zero => exact Or.inr rfl
      | succ value =>
          exact False.elim (Nat.not_lt_zero value
            (Nat.lt_of_succ_lt_succ (Nat.lt_of_succ_lt_succ bound)))

/-- A one-slot layout with its single slot occupied: the joint witness for both a successful
replacement and an absent-key full-table refusal. -/
def writeWitnessBase : Word := Word.ofNat 0
def writeWitnessCap : Word := Word.ofNat 1
def writeWitnessKey : Word := Word.ofNat 1
def writeWitnessOther : Word := Word.ofNat 2
def writeWitnessValue : Word := Word.ofNat 7
def writeWitnessNewValue : Word := Word.ofNat 9
def writeWitnessDefault : Word := Word.ofNat 0
def writeWitnessOccupied : List Word := [writeWitnessKey]

def writeWitnessMap (key : Word) : Word :=
  if key = writeWitnessKey then writeWitnessValue else writeWitnessDefault

def writeWitnessMemory : Memory where
  owned := fun coordinate => coordinate < 2
  initialized := fun coordinate =>
    if coordinate = 0 then some writeWitnessKey
    else if coordinate = 1 then some writeWitnessValue
    else none

@[simp] theorem writeWitnessBase_val : writeWitnessBase.val = 0 := by
  unfold writeWitnessBase
  exact ofNat_val_emptyCone 0 (by decide)

@[simp] theorem writeWitnessCap_val : writeWitnessCap.val = 1 := by
  unfold writeWitnessCap
  exact ofNat_val_emptyCone 1 (by decide)

@[simp] theorem writeWitness_keyAt_zero :
    keyAt writeWitnessMemory writeWitnessBase 0 = some writeWitnessKey := by
  unfold keyAt keyCoordinate
  rw [writeWitnessBase_val, Nat.mul_zero, Nat.add_zero]
  rfl

@[simp] theorem writeWitness_valueAt_zero :
    valueAt writeWitnessMemory writeWitnessBase 0 = some writeWitnessValue := by
  unfold valueAt valueCoordinate
  rw [writeWitnessBase_val, Nat.mul_zero, Nat.zero_add]
  rfl

@[simp] theorem writeWitnessKey_ne_empty : writeWitnessKey ≠ tableEmpty := by decide

@[simp] theorem writeWitnessOther_ne_key : writeWitnessOther ≠ writeWitnessKey := by decide

@[simp] theorem writeWitnessOther_ne_empty : writeWitnessOther ≠ tableEmpty := by decide

@[simp] theorem writeWitnessKey_ne_other : writeWitnessKey ≠ writeWitnessOther := by decide

theorem writeWitness_probe_zero (key : Word) :
    probeSlot writeWitnessCap key 0 = 0 := by
  unfold probeSlot
  rw [writeWitnessCap_val, Nat.add_zero, Nat.mod_one]

theorem writeWitness_member_iff (key : Word) :
    member writeWitnessOccupied key = true ↔ key = writeWitnessKey := by
  unfold writeWitnessOccupied
  constructor
  · intro present
    rcases (member_cons_true writeWitnessKey key []).1 present with equal | absent
    · exact equal
    · exact Bool.noConfusion (absent : member ([] : List Word) key = true)
  · intro equal
    exact (member_cons_true writeWitnessKey key []).2 (Or.inl equal)

theorem writeWitness_member_true :
    member writeWitnessOccupied writeWitnessKey = true :=
  (writeWitness_member_iff writeWitnessKey).2 rfl

theorem writeWitness_repSong :
    RepSong writeWitnessMemory writeWitnessBase writeWitnessCap writeWitnessDefault
      writeWitnessOccupied writeWitnessMap := by
  refine ⟨⟨0, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨?_, ?_⟩
    · show writeWitnessBase.val + 2 * writeWitnessCap.val ≤ coordCeiling ∧
        writeWitnessBase.val + 2 * writeWitnessCap.val ≤ wordSignBit ∧
        ∀ offset < 2 * writeWitnessCap.val,
          writeWitnessMemory.owned (writeWitnessBase.val + offset) = true
      refine ⟨?_, ?_, ?_⟩
      · rw [writeWitnessBase_val, writeWitnessCap_val]; decide
      · rw [writeWitnessBase_val, writeWitnessCap_val]; decide
      · intro offset hoffset
        rw [writeWitnessCap_val] at hoffset
        rw [writeWitnessBase_val]
        rcases nat_lt_two_cases offset hoffset with rfl | rfl <;> rfl
    · intro offset hoffset
      rw [writeWitnessCap_val] at hoffset
      rw [writeWitnessBase_val]
      rcases nat_lt_two_cases offset hoffset with rfl | rfl <;> rfl
  · refine ⟨?_, ?_⟩
    · show member ([] : List Word) writeWitnessKey = false ∧ DuplicateFree []
      exact ⟨rfl, trivial⟩
    · intro key present
      rw [(writeWitness_member_iff key).1 present]
      exact writeWitnessKey_ne_empty
  · intro slot slotBound
    have hzero : slot = 0 := by
      rw [writeWitnessCap_val] at slotBound
      exact Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ slotBound)
    subst hzero
    exact Or.inr ⟨writeWitnessKey, writeWitness_member_true, writeWitness_keyAt_zero⟩
  · constructor
    · intro i hi j hj key keyNe keyI keyJ
      rw [writeWitnessCap_val] at hi hj
      have hiZero : i = 0 := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ hi)
      have hjZero : j = 0 := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ hj)
      rw [hiZero, hjZero]
    · intro key
      constructor
      · intro present
        refine ⟨0, ?_, ?_, ?_⟩
        · rw [writeWitnessCap_val]; decide
        · rw [(writeWitness_member_iff key).1 present]
          exact writeWitness_keyAt_zero
        · rw [(writeWitness_member_iff key).1 present]
          exact writeWitnessKey_ne_empty
      · rintro ⟨slot, slotBound, physical, notEmpty⟩
        rw [writeWitnessCap_val] at slotBound
        have slotZero : slot = 0 := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ slotBound)
        subst slotZero
        rw [writeWitness_keyAt_zero] at physical
        exact (writeWitness_member_iff key).2 (Option.some.inj physical).symm
  · intro key present
    rw [(writeWitness_member_iff key).1 present]
    refine ⟨0, by rw [writeWitnessCap_val]; decide, ?_, ?_⟩
    · rw [writeWitness_probe_zero]
      exact writeWitness_keyAt_zero
    · intro earlier earlierBound
      exact False.elim (Nat.not_lt_zero earlier earlierBound)
  · constructor
    · intro slot slotBound key present keyAtSlot
      rw [writeWitnessCap_val] at slotBound
      have slotZero : slot = 0 := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ slotBound)
      subst slotZero
      rw [writeWitness_keyAt_zero] at keyAtSlot
      have keyEq : key = writeWitnessKey := (Option.some.inj keyAtSlot).symm
      subst keyEq
      rw [writeWitness_valueAt_zero]
      unfold writeWitnessMap
      rw [if_pos rfl]
    · intro key absent
      unfold writeWitnessMap
      rw [if_neg]
      intro equal
      rw [(writeWitness_member_iff key).2 equal] at absent
      exact Bool.noConfusion absent

theorem writeWitnessKey_not_exhausted :
    ¬ ProbeExhausted writeWitnessMemory writeWitnessBase writeWitnessCap writeWitnessKey := by
  intro exhausted
  exact (exhausted 0 (by rw [writeWitnessCap_val]; decide)).2
    (by rw [writeWitness_probe_zero]; exact writeWitness_keyAt_zero)

theorem writeWitness_exhausted :
    ProbeExhausted writeWitnessMemory writeWitnessBase writeWitnessCap writeWitnessOther := by
  intro stop hstop
  rw [writeWitnessCap_val] at hstop
  have stopZero : stop = 0 := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ hstop)
  subst stopZero
  rw [writeWitness_probe_zero]
  exact ⟨fun equal => writeWitnessKey_ne_empty
      (Option.some.inj (writeWitness_keyAt_zero.symm.trans equal)),
    fun equal => writeWitnessKey_ne_other
      (Option.some.inj (writeWitness_keyAt_zero.symm.trans equal))⟩

/-- The joint witness: on the same one-slot full layout, replacing the present key succeeds and
the absent nonreserved key returns raw `tableEmpty` (`-1`) with the memory unchanged. Both are
checked executions against the constant environment of the decoded program. -/
theorem writeWitness_joint (program : Program)
    (capacityOk spread findProbe slotKey slotVal slotSet slotValSet : FunctionDecl)
    (hcapacityOk : (checkedContext program).functions "tableCapacityOk" = some capacityOk)
    (hspread : (checkedContext program).functions "tableSpread" = some spread)
    (hfindProbe : (checkedContext program).functions "tableFindSlotGo" = some findProbe)
    (hslotKey : (checkedContext program).functions "tableSlotKey" = some slotKey)
    (hslotVal : (checkedContext program).functions "tableSlotVal" = some slotVal)
    (hslotSet : (checkedContext program).functions "tableSlotSet" = some slotSet)
    (hslotValSet : (checkedContext program).functions "tableSlotValSet" = some slotValSet)
    (hinsert : (checkedContext program).functions "tableInsertWith" = some insertWithDecl)
    (hcapacityShape : TableCapacityOkShape capacityOk)
    (hspreadShape : TableSpreadShape spread)
    (hfindShape : TableFindSlotGoShape findProbe)
    (hslotKeyShape : TableSlotKeyShape slotKey)
    (hslotValShape : TableSlotValShape slotVal)
    (hslotSetShape : TableSlotSetShape slotSet)
    (hslotValSetShape : TableSlotValSetShape slotValSet)
    (hstride : (checkedContext program).constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : (checkedContext program).constants "TABLE_EMPTY" = some tableEmpty)
    (hmergeSet : (checkedContext program).constants "MERGE_SET" = some (Word.ofNat 1))
    (hmergeAcc : (checkedContext program).constants "MERGE_ACC" = some (Word.ofNat 3))
    (hphi : (checkedContext program).constants "TABLE_PHI" = some Word.tablePhi) :
    (∃ raw memory' trace,
        EvalCall (checkedContext program) .unbounded 0 "tableInsertWith"
          [writeWitnessBase, writeWitnessCap, writeWitnessKey, writeWitnessNewValue, Word.ofNat 2]
          writeWitnessMemory (.value raw memory') trace ∧
        raw = writeWitnessNewValue ∧
        RepSong memory' writeWitnessBase writeWitnessCap writeWitnessDefault writeWitnessOccupied
          (Apeiron.Table.insertWith (fun new _old => new) writeWitnessKey writeWitnessNewValue
            writeWitnessMap)) ∧
    (∃ trace,
        EvalCall (checkedContext program) .unbounded 0 "tableInsertWith"
          [writeWitnessBase, writeWitnessCap, writeWitnessOther, writeWitnessNewValue, Word.ofNat 2]
          writeWitnessMemory (.value tableEmpty writeWitnessMemory) trace) := by
  have success := tableInsertWith_admitted_success (checkedContext program) capacityOk spread
    findProbe slotKey slotVal slotSet slotValSet hcapacityOk hspread hfindProbe hslotKey hslotVal
    hslotSet hslotValSet hinsert hcapacityShape hspreadShape hfindShape hslotKeyShape
    hslotValShape hslotSetShape hslotValSetShape hstride hempty hmergeSet hmergeAcc hphi
    writeWitnessMemory writeWitnessBase writeWitnessCap writeWitnessDefault writeWitnessKey
    writeWitnessNewValue writeWitnessOccupied writeWitnessMap writeWitness_repSong
    writeWitnessKey_ne_empty ⟨writeWitnessKey_ne_empty, writeWitnessKey_not_exhausted⟩
  refine ⟨?_, ?_⟩
  · rcases success with ⟨raw, memory', trace, hexec, _, outcome⟩
    rcases outcome with ⟨slot, slotBound, present, keyAtSlot, hraw, hmem⟩ |
      ⟨slot, step, stepBound, slotEq, absent, emptyAt, hraw⟩
    · refine ⟨raw, memory', trace, hexec, hraw, ?_⟩
      rw [hmem]
      exact repSong_storeValueAt writeWitness_repSong slot slotBound writeWitnessKey
        writeWitnessNewValue present keyAtSlot
    · exact False.elim (Bool.noConfusion
        (writeWitness_member_true.symm.trans absent))
  · exact tableInsertWith_refusal_full (checkedContext program) capacityOk spread findProbe slotKey
      slotVal slotSet slotValSet hcapacityOk hspread hfindProbe hslotKey hslotVal hslotSet
      hslotValSet hinsert hcapacityShape hspreadShape hfindShape hslotKeyShape hslotValShape
      hslotSetShape hslotValSetShape hstride hempty hmergeSet hmergeAcc hphi writeWitnessMemory
      writeWitnessBase writeWitnessCap writeWitnessDefault writeWitnessOther writeWitnessNewValue
      writeWitnessOccupied writeWitnessMap writeWitness_repSong writeWitnessOther_ne_empty
      writeWitness_exhausted

#audit_empty Slang.Table.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Table.WriteAdmitted : type_of% @Slang.Table.WriteAdmitted
#audit_empty Slang.Table.WriteSuccess : type_of% @Slang.Table.WriteSuccess
#audit_empty Slang.Table.WriteSuccess.not_exhausted : type_of% @Slang.Table.WriteSuccess.not_exhausted
#audit_empty Slang.Table.tableInsertWith_admitted_success : type_of% @Slang.Table.tableInsertWith_admitted_success
#audit_empty Slang.Table.tableInsertWith_refusal_full : type_of% @Slang.Table.tableInsertWith_refusal_full
#audit_empty Slang.Table.tableInsertWith_admission_exact : type_of% @Slang.Table.tableInsertWith_admission_exact
#audit_empty Slang.Table.writeWitnessKey_not_exhausted : type_of% @Slang.Table.writeWitnessKey_not_exhausted
#audit_empty Slang.Table.writeWitness_exhausted : type_of% @Slang.Table.writeWitness_exhausted
#audit_empty Slang.Table.writeWitness_repSong : type_of% @Slang.Table.writeWitness_repSong
#audit_empty Slang.Table.writeWitness_joint : type_of% @Slang.Table.writeWitness_joint
#audit_empty Slang.Table.tableInsertWith_map_execution : type_of% @Slang.Table.tableInsertWith_map_execution
#audit_empty Slang.Table.WriteMapOutcome : type_of% @Slang.Table.WriteMapOutcome
#audit_empty Slang.Table.WriteMapOutcome.preserves : type_of% @Slang.Table.WriteMapOutcome.preserves
#audit_empty Slang.Table.RepSong.congr : type_of% @Slang.Table.RepSong.congr

end Slang.Table
