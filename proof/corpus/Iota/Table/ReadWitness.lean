import Iota.Table.TableRead
import Iota.Audit.Empty

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Iota.Table

open Iota Iota.Decode Iota.Machine Iota.Source Iota.Refinement

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

def readWitnessBase : Word := Word.ofNat 0
def readWitnessCap : Word := Word.ofNat 4
def readWitnessKey : Word := Word.ofNat 0
def readWitnessCollision : Word := Word.ofNat 4
def readWitnessMiss : Word := Word.ofNat 8
def readWitnessDefault : Word := Word.ofNat 0
def readWitnessOccupied : List Word := [readWitnessKey, readWitnessCollision]

def readWitnessMap (key : Word) : Word :=
  if key = readWitnessKey then tableEmpty
  else if key = readWitnessCollision then Word.ofNat 7
  else readWitnessDefault

def readWitnessMemory : Memory where
  owned := fun coordinate => coordinate < 8
  initialized := fun coordinate =>
    if coordinate = 0 then some readWitnessCollision
    else if coordinate = 1 then some (Word.ofNat 7)
    else if coordinate = 2 then some readWitnessKey
    else if coordinate = 3 then some tableEmpty
    else if coordinate = 4 then some tableEmpty
    else if coordinate = 5 then some (Word.ofNat 0)
    else if coordinate = 6 then some tableEmpty
    else if coordinate = 7 then some (Word.ofNat 0)
    else none

@[simp] theorem readWitness_keyAt_zero :
    keyAt readWitnessMemory readWitnessBase 0 = some readWitnessCollision := by
  rfl

@[simp] theorem readWitness_keyAt_one :
    keyAt readWitnessMemory readWitnessBase 1 = some readWitnessKey := by
  rfl

@[simp] theorem readWitness_keyAt_two :
    keyAt readWitnessMemory readWitnessBase 2 = some tableEmpty := by
  rfl

@[simp] theorem readWitness_keyAt_three :
    keyAt readWitnessMemory readWitnessBase 3 = some tableEmpty := by
  rfl

@[simp] theorem readWitness_valueAt_zero :
    valueAt readWitnessMemory readWitnessBase 0 = some (Word.ofNat 7) := by
  rfl

@[simp] theorem readWitness_valueAt_one :
    valueAt readWitnessMemory readWitnessBase 1 = some tableEmpty := by
  rfl

@[simp] theorem readWitness_valueAt_two :
    valueAt readWitnessMemory readWitnessBase 2 = some (Word.ofNat 0) := by
  rfl

@[simp] theorem readWitness_valueAt_three :
    valueAt readWitnessMemory readWitnessBase 3 = some (Word.ofNat 0) := by
  rfl

@[simp] theorem readWitness_probe_key :
    probeSlot readWitnessCap readWitnessKey 1 = 1 := by decide

@[simp] theorem readWitness_probe_collision :
    probeSlot readWitnessCap readWitnessCollision 0 = 0 := by decide

@[simp] theorem readWitness_probe_key_zero :
    probeSlot readWitnessCap readWitnessKey 0 = 0 := by decide

@[simp] theorem readWitnessCollision_ne_key : readWitnessCollision ≠ readWitnessKey := by decide

@[simp] theorem readWitnessCollision_ne_empty : readWitnessCollision ≠ tableEmpty := by decide

@[simp] theorem readWitnessKey_ne_empty : readWitnessKey ≠ tableEmpty := by decide

theorem nat_lt_four_cases (value : Nat) (bound : value < 4) :
    value = 0 ∨ value = 1 ∨ value = 2 ∨ value = 3 := by
  cases value with
  | zero => exact Or.inl rfl
  | succ value =>
      cases value with
      | zero => exact Or.inr (Or.inl rfl)
      | succ value =>
          cases value with
          | zero => exact Or.inr (Or.inr (Or.inl rfl))
          | succ value =>
              cases value with
              | zero => exact Or.inr (Or.inr (Or.inr rfl))
              | succ value =>
                  exact False.elim (Nat.not_lt_zero value
                    (Nat.lt_of_succ_lt_succ (Nat.lt_of_succ_lt_succ
                      (Nat.lt_of_succ_lt_succ (Nat.lt_of_succ_lt_succ bound)))))

theorem nat_lt_eight_cases (value : Nat) (bound : value < 8) :
    value = 0 ∨ value = 1 ∨ value = 2 ∨ value = 3 ∨
      value = 4 ∨ value = 5 ∨ value = 6 ∨ value = 7 := by
  cases value with
  | zero => exact Or.inl rfl
  | succ value =>
      cases value with
      | zero => exact Or.inr (Or.inl rfl)
      | succ value =>
          cases value with
          | zero => exact Or.inr (Or.inr (Or.inl rfl))
          | succ value =>
              cases value with
              | zero => exact Or.inr (Or.inr (Or.inr (Or.inl rfl)))
              | succ value =>
                  cases value with
                  | zero => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
                  | succ value =>
                      cases value with
                      | zero =>
                          exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
                      | succ value =>
                          cases value with
                          | zero =>
                              exact Or.inr (Or.inr (Or.inr
                                (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))
                          | succ value =>
                              cases value with
                              | zero =>
                                  exact Or.inr (Or.inr (Or.inr
                                    (Or.inr (Or.inr (Or.inr (Or.inr rfl))))))
                              | succ value =>
                                  exact False.elim (Nat.not_lt_zero value
                                    (Nat.lt_of_succ_lt_succ (Nat.lt_of_succ_lt_succ
                                      (Nat.lt_of_succ_lt_succ (Nat.lt_of_succ_lt_succ
                                        (Nat.lt_of_succ_lt_succ (Nat.lt_of_succ_lt_succ
                                          (Nat.lt_of_succ_lt_succ
                                            (Nat.lt_of_succ_lt_succ bound)))))))))

theorem readWitness_member_iff (key : Word) :
    member readWitnessOccupied key = true ↔
      key = readWitnessKey ∨ key = readWitnessCollision := by
  constructor
  · intro present
    unfold readWitnessOccupied member at present
    cases first : Word.decidableEq readWitnessKey key with
    | isTrue equal => exact Or.inl equal.symm
      | isFalse notFirst =>
        simp only [first] at present
        change member [readWitnessCollision] key = true at present
        unfold member at present
        cases second : Word.decidableEq readWitnessCollision key with
        | isTrue equal => exact Or.inr equal.symm
        | isFalse notSecond =>
            rw [second] at present
            contradiction
  · intro equal
    rcases equal with rfl | rfl
    · exact member_cons_self readWitnessKey [readWitnessCollision]
    · unfold readWitnessOccupied member
      decide

theorem readWitness_not_member_iff (key : Word) :
    member readWitnessOccupied key = false ↔
      key ≠ readWitnessKey ∧ key ≠ readWitnessCollision := by
  constructor
  · intro absent
    exact ⟨fun equal => Bool.noConfusion
      ((readWitness_member_iff key).2 (Or.inl equal) |>.symm.trans absent),
      fun equal => Bool.noConfusion
        ((readWitness_member_iff key).2 (Or.inr equal) |>.symm.trans absent)⟩
  · intro absent
    cases present : member readWitnessOccupied key with
    | false => rfl
    | true =>
        rcases (readWitness_member_iff key).1 present with equal | equal
        · exact False.elim (absent.1 equal)
        · exact False.elim (absent.2 equal)

theorem readWitness_repSong :
    RepSong readWitnessMemory readWitnessBase readWitnessCap readWitnessDefault
      readWitnessOccupied readWitnessMap := by
  refine ⟨⟨2, rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · constructor
    · exact ⟨by decide, by decide, fun offset hoffset => by
        change offset < 8 at hoffset
        change decide (0 + offset < 8) = true
        rw [Nat.zero_add]
        exact decide_eq_true hoffset⟩
    · intro offset hoffset
      rcases nat_lt_eight_cases offset hoffset with
        rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> rfl
  · constructor
    · exact ⟨by decide, ⟨rfl, True.intro⟩⟩
    · intro key present
      rcases (readWitness_member_iff key).1 present with rfl | rfl <;> decide
  · intro slot hslot
    rcases nat_lt_four_cases slot hslot with rfl | rfl | rfl | rfl
    · exact Or.inr ⟨readWitnessCollision,
        (readWitness_member_iff _).2 (Or.inr rfl), readWitness_keyAt_zero⟩
    · exact Or.inr ⟨readWitnessKey,
        (readWitness_member_iff _).2 (Or.inl rfl), readWitness_keyAt_one⟩
    · exact Or.inl readWitness_keyAt_two
    · exact Or.inl readWitness_keyAt_three
  · constructor
    · intro i hi j hj key hkey hiKey hjKey
      rcases nat_lt_four_cases i hi with rfl | rfl | rfl | rfl
      · rcases nat_lt_four_cases j hj with rfl | rfl | rfl | rfl
        · rfl
        · rw [readWitness_keyAt_zero] at hiKey
          rw [readWitness_keyAt_one] at hjKey
          exact False.elim (readWitnessCollision_ne_key
            (Option.some.inj (hiKey.trans hjKey.symm)))
        · exact False.elim (hkey (Option.some.inj (hjKey.symm.trans readWitness_keyAt_two)))
        · exact False.elim (hkey (Option.some.inj (hjKey.symm.trans readWitness_keyAt_three)))
      · rcases nat_lt_four_cases j hj with rfl | rfl | rfl | rfl
        · rw [readWitness_keyAt_zero] at hjKey
          rw [readWitness_keyAt_one] at hiKey
          exact False.elim (readWitnessCollision_ne_key
            (Option.some.inj (hjKey.trans hiKey.symm)))
        · rfl
        · exact False.elim (hkey (Option.some.inj (hjKey.symm.trans readWitness_keyAt_two)))
        · exact False.elim (hkey (Option.some.inj (hjKey.symm.trans readWitness_keyAt_three)))
      · exact False.elim (hkey (Option.some.inj (hiKey.symm.trans readWitness_keyAt_two)))
      · exact False.elim (hkey (Option.some.inj (hiKey.symm.trans readWitness_keyAt_three)))
    · intro key
      constructor
      · intro present
        rcases (readWitness_member_iff key).1 present with rfl | rfl
        · exact ⟨1, by decide, readWitness_keyAt_one, by decide⟩
        · exact ⟨0, by decide, readWitness_keyAt_zero, by decide⟩
      · intro occurrence
        rcases occurrence with ⟨slot, slotBound, physical, notEmpty⟩
        rcases nat_lt_four_cases slot slotBound with rfl | rfl | rfl | rfl
        · exact (readWitness_member_iff key).2
            (Or.inr (Option.some.inj (physical.symm.trans readWitness_keyAt_zero)))
        · exact (readWitness_member_iff key).2
            (Or.inl (Option.some.inj (physical.symm.trans readWitness_keyAt_one)))
        · exact False.elim (notEmpty
            (Option.some.inj (physical.symm.trans readWitness_keyAt_two)))
        · exact False.elim (notEmpty
            (Option.some.inj (physical.symm.trans readWitness_keyAt_three)))
  · intro key present
    rcases (readWitness_member_iff key).1 present with rfl | rfl
    · refine ⟨1, by decide, readWitness_keyAt_one, ?_⟩
      intro earlier earlierBound
      have earlierZero := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ earlierBound)
      subst earlier
      rw [readWitness_probe_key_zero, readWitness_keyAt_zero]
      intro equal
      exact readWitnessCollision_ne_empty (Option.some.inj equal)
    · exact ⟨0, by decide, readWitness_keyAt_zero,
        fun earlier earlierBound => False.elim (Nat.not_lt_zero earlier earlierBound)⟩
  · constructor
    · intro slot hslot key hmember hkeyAt
      rcases (readWitness_member_iff key).1 hmember with rfl | rfl
      · rcases nat_lt_four_cases slot hslot with rfl | rfl | rfl | rfl
        · rw [readWitness_keyAt_zero] at hkeyAt
          exact False.elim (readWitnessCollision_ne_key (Option.some.inj hkeyAt))
        · exact readWitness_valueAt_one
        · exact False.elim (readWitnessKey_ne_empty
            (Option.some.inj (hkeyAt.symm.trans readWitness_keyAt_two)))
        · exact False.elim (readWitnessKey_ne_empty
            (Option.some.inj (hkeyAt.symm.trans readWitness_keyAt_three)))
      · rcases nat_lt_four_cases slot hslot with rfl | rfl | rfl | rfl
        · exact readWitness_valueAt_zero
        · rw [readWitness_keyAt_one] at hkeyAt
          exact False.elim (readWitnessCollision_ne_key (Option.some.inj hkeyAt).symm)
        · exact False.elim (readWitnessCollision_ne_empty
            (Option.some.inj (hkeyAt.symm.trans readWitness_keyAt_two)))
        · exact False.elim (readWitnessCollision_ne_empty
            (Option.some.inj (hkeyAt.symm.trans readWitness_keyAt_three)))
    · intro key hnot
      have absent := (readWitness_not_member_iff key).1 hnot
      unfold readWitnessMap
      rw [if_neg absent.1, if_neg absent.2]

theorem readWitness_present :
    observeRead readWitnessMemory readWitnessBase readWitnessCap readWitnessDefault
      readWitnessOccupied readWitnessMap readWitnessKey tableEmpty := by
  refine ⟨readWitness_repSong, ?_⟩
  have present := (readWitness_member_iff readWitnessKey).2 (Or.inl rfl)
  rw [present]
  unfold readWitnessMap
  rw [if_pos rfl]
  exact .present readWitnessDefault tableEmpty tableEmpty rfl

theorem readWitness_miss :
    observeRead readWitnessMemory readWitnessBase readWitnessCap readWitnessDefault
      readWitnessOccupied readWitnessMap readWitnessMiss tableEmpty := by
  refine ⟨readWitness_repSong, ?_⟩
  have absent : member readWitnessOccupied readWitnessMiss = false :=
    (readWitness_not_member_iff readWitnessMiss).2 (by decide)
  rw [absent]
  exact .absent readWitnessDefault readWitnessDefault tableEmpty rfl

theorem readWitness_collision_reachability :
    probeSlot readWitnessCap readWitnessKey 0 = 0 ∧
    keyAt readWitnessMemory readWitnessBase 0 = some readWitnessCollision ∧
    probeSlot readWitnessCap readWitnessKey 1 = 1 ∧
    keyAt readWitnessMemory readWitnessBase 1 = some readWitnessKey := by
  exact ⟨readWitness_probe_key_zero, readWitness_keyAt_zero,
    readWitness_probe_key, readWitness_keyAt_one⟩

theorem readWitness_present_probe :
    ProbeOutcome false readWitnessMemory readWitnessBase readWitnessCap
      readWitnessKey 0 (Word.ofNat 1) := by
  apply ProbeOutcome.next 0 (Word.ofNat 1) (by decide)
  · rw [readWitness_probe_key_zero, readWitness_keyAt_zero]
    intro equal
    exact readWitnessCollision_ne_empty (Option.some.inj equal)
  · rw [readWitness_probe_key_zero, readWitness_keyAt_zero]
    intro equal
    exact readWitnessCollision_ne_key (Option.some.inj equal)
  · exact ProbeOutcome.match 1 (by decide) (by
      rw [readWitness_probe_key]
      exact readWitness_keyAt_one)

theorem readWitness_miss_probe :
    ProbeOutcome false readWitnessMemory readWitnessBase readWitnessCap
      readWitnessMiss 0 tableEmpty := by
  apply ProbeOutcome.next 0 tableEmpty (by decide)
  · decide
  · decide
  · apply ProbeOutcome.next 1 tableEmpty (by decide)
    · decide
    · decide
    · exact ProbeOutcome.empty 2 (by decide) (by decide)

def naiveAbsenceClassifier (raw : Word) : Bool := raw == tableEmpty

def ReadClassifierSound (classifier : Word → Bool) : Prop :=
  ∀ key raw, observeRead readWitnessMemory readWitnessBase readWitnessCap readWitnessDefault
    readWitnessOccupied readWitnessMap key raw →
      classifier raw = !member readWitnessOccupied key

theorem naive_absence_classifier_unsound : ¬ ReadClassifierSound naiveAbsenceClassifier := by
  intro sound
  have classified := sound readWitnessKey tableEmpty readWitness_present
  have left : naiveAbsenceClassifier tableEmpty = true := by
    unfold naiveAbsenceClassifier
    rfl
  have right : (!member readWitnessOccupied readWitnessKey) = false := by
    rw [(readWitness_member_iff readWitnessKey).2 (Or.inl rfl)]
    rfl
  exact Bool.noConfusion (left.symm.trans (classified.trans right))

-- Temporal O(|parameters| + |body|) element comparisons: `List.beq` scans the declaration's
-- parameters against `parameters` and its body against `body`, one element per step, stopping at
-- the first mismatch or the shorter list's end. Spatial O(1) auxiliary: the result is a `Bool` and
-- nothing is allocated; the scans' recursion depth is at most the compared length. Operations
-- counted: `String` equality comparisons, `Stmt` equality comparisons.
def declarationShapeCheck (declaration : FunctionDecl) (parameters : List String)
    (body : List Stmt) : Bool := declaration.parameters == parameters && declaration.body == body

-- Temporal O(|tableSource| + |stdSource| + F) for F = the number of decoded functions: one linear
-- `decodeClosure` over the two source byte arrays, then eight `List.find?` scans of
-- `value.val.program.functions` (each O(F)), then one `declarationShapeCheck` per found function.
-- Spatial O(|tableSource| + |stdSource| + F) for the decoded closure and its program; the `Bool`
-- result is O(1). Operations counted: source bytes decoded, function declarations scanned, and the
-- `String`/`Stmt` comparisons inside `declarationShapeCheck`. Each `find?` closes at the
-- function-list end and the decode closes at the input end.
def realReadShapes : Bool :=
  match Decode.decodeClosure
      (encodeBytes Iota.Generated.TableDerivation.tableSource)
      (encodeBytes Iota.Generated.TableDerivation.stdSource) with
  | .error _ => false
  | .ok value =>
      match value.val.program.functions.find? (fun declaration => declaration.name == "tableSlotKey"),
        value.val.program.functions.find? (fun declaration => declaration.name == "tableSlotVal"),
        value.val.program.functions.find? (fun declaration => declaration.name == "tableCapacityOk"),
        value.val.program.functions.find? (fun declaration => declaration.name == "tableSpread"),
        value.val.program.functions.find? (fun declaration => declaration.name == "tableFindSlotGo"),
        value.val.program.functions.find? (fun declaration => declaration.name == "tableLookupSlotGo"),
        value.val.program.functions.find? (fun declaration => declaration.name == "tableFind"),
        value.val.program.functions.find? (fun declaration => declaration.name == "tableGet") with
      | some slotKey, some slotVal, some capacityOk, some spread, some findProbe,
          some lookupProbe, some find, some get =>
          declarationShapeCheck slotKey ["base", "slot"] [.return_ (.memory slotAddressExpr)] &&
          declarationShapeCheck slotVal ["base", "slot"] [.return_ (.memory slotValueAddressExpr)] &&
          declarationShapeCheck capacityOk ["cap"]
            [.if_ (.binary .le (.name "cap") (.literal 0)) [.return_ (.literal 0)] none,
             .return_ (.conditional (.literal 1)
               (.binary .eq (.binary .bitAnd (.name "cap")
                 (.binary .sub (.name "cap") (.literal 1))) (.literal 0)) (.literal 0))] &&
          declarationShapeCheck spread ["k"]
            [.bind "pos" (.conditional (.binary .sub (.literal 0) (.name "k"))
              (.binary .lt (.name "k") (.literal 0)) (.name "k")),
             .return_ (.binary .bitAnd (.binary .mul (.name "pos") (.name "TABLE_PHI"))
               (.literal 4611686018427387903))] &&
          declarationShapeCheck findProbe ["base", "cap", "mask", "key", "slot", "step"]
            (probeBody "tableFindSlotGo" (.name "currSlot")) &&
          declarationShapeCheck lookupProbe ["base", "cap", "mask", "key", "slot", "step"]
            (probeBody "tableLookupSlotGo" (.binary .sub (.literal 0) (.literal 1))) &&
          declarationShapeCheck find ["base", "cap", "key"]
            [.if_ (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0))
              [.return_ (.name "TABLE_EMPTY")] none,
             .if_ (.binary .eq (.name "key") (.name "TABLE_EMPTY"))
              [.return_ (.name "TABLE_EMPTY")] none,
             .bind "mask" (.binary .sub (.name "cap") (.literal 1)),
             .bind "initSlot" (.binary .bitAnd (.call "tableSpread" [.name "key"])
               (.name "mask")),
             .return_ (.call "tableLookupSlotGo"
               [.name "base", .name "cap", .name "mask", .name "key", .name "initSlot",
                .literal 0])] &&
          declarationShapeCheck get ["base", "cap", "key"]
            [.bind "slot" (.call "tableFind" [.name "base", .name "cap", .name "key"]),
             .if_ (.binary .lt (.name "slot") (.literal 0))
               [.return_ (.name "TABLE_EMPTY")] none,
             .return_ (.call "tableSlotVal" [.name "base", .name "slot"])] &&
          constantEnvironment value.val.program "TABLE_STRIDE" == some (Word.ofNat 2) &&
          constantEnvironment value.val.program "TABLE_EMPTY" == some tableEmpty &&
          constantEnvironment value.val.program "TABLE_PHI" == some Word.tablePhi
      | _, _, _, _, _, _, _, _ => false

#guard realReadShapes

#audit_empty Iota.Table.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Table.TableSlotKeyShape : type_of% @Iota.Table.TableSlotKeyShape
#audit_empty Iota.Table.TableSlotValShape : type_of% @Iota.Table.TableSlotValShape
#audit_empty Iota.Table.TableFindSlotGoShape : type_of% @Iota.Table.TableFindSlotGoShape
#audit_empty Iota.Table.TableLookupSlotGoShape : type_of% @Iota.Table.TableLookupSlotGoShape
#audit_empty Iota.Table.TableFindShape : type_of% @Iota.Table.TableFindShape
#audit_empty Iota.Table.TableSpreadShape : type_of% @Iota.Table.TableSpreadShape
#audit_empty Iota.Table.source_probe_slot_eq : type_of% @Iota.Table.source_probe_slot_eq
#audit_empty Iota.Table.tableSlotKey_execution : type_of% @Iota.Table.tableSlotKey_execution
#audit_empty Iota.Table.ProbeOutcome : type_of% @Iota.Table.ProbeOutcome
#audit_empty Iota.Table.probe_execution : type_of% @Iota.Table.probe_execution
#audit_empty Iota.Table.tableLookupSlotGo_execution : type_of% @Iota.Table.tableLookupSlotGo_execution
#audit_empty Iota.Table.tableFindSlotGo_execution : type_of% @Iota.Table.tableFindSlotGo_execution
#audit_empty Iota.Table.ProbeOutcome.lookup_present : type_of% @Iota.Table.ProbeOutcome.lookup_present
#audit_empty Iota.Table.ProbeOutcome.lookup_miss : type_of% @Iota.Table.ProbeOutcome.lookup_miss
#audit_empty Iota.Table.tableSpread_execution : type_of% @Iota.Table.tableSpread_execution
#audit_empty Iota.Table.tableFind_execution : type_of% @Iota.Table.tableFind_execution
#audit_empty Iota.Table.TableGetShape : type_of% @Iota.Table.TableGetShape
#audit_empty Iota.Table.ReadObservation : type_of% @Iota.Table.ReadObservation
#audit_empty Iota.Table.observeRead : type_of% @Iota.Table.observeRead
#audit_empty Iota.Table.tableSlotVal_execution : type_of% @Iota.Table.tableSlotVal_execution
#audit_empty Iota.Table.tableGet_execution : type_of% @Iota.Table.tableGet_execution
#audit_empty Iota.Table.tableGet_read_correct : type_of% @Iota.Table.tableGet_read_correct
#audit_empty Iota.Table.tableGet_execution_of_certificate : type_of% @Iota.Table.tableGet_execution_of_certificate
#audit_empty Iota.Table.readWitness_repSong : type_of% @Iota.Table.readWitness_repSong
#audit_empty Iota.Table.readWitness_present : type_of% @Iota.Table.readWitness_present
#audit_empty Iota.Table.readWitness_miss : type_of% @Iota.Table.readWitness_miss
#audit_empty Iota.Table.readWitness_collision_reachability : type_of% @Iota.Table.readWitness_collision_reachability
#audit_empty Iota.Table.readWitness_present_probe : type_of% @Iota.Table.readWitness_present_probe
#audit_empty Iota.Table.readWitness_miss_probe : type_of% @Iota.Table.readWitness_miss_probe
#audit_empty Iota.Table.naive_absence_classifier_unsound : type_of% @Iota.Table.naive_absence_classifier_unsound

end Iota.Table
