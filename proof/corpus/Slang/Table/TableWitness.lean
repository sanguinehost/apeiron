import Slang.Table.TableInit
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Table

open Slang Slang.Decode Slang.Machine Slang.Source Slang.Refinement

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

-- Temporal O(1): `declaration.parameters == [...]` is a `List String` `BEq` bounded by the fixed
-- 4-string literal (≤ 4 string comparisons), and `declaration.body == [...]` a `List Stmt` `BEq`
-- bounded by the fixed 3-statement literal (≤ 3 structural `Stmt` comparisons, each bounded by the
-- literal's fixed expression tree). Spatial O(1): both `BEq` stacks are bounded by the literal
-- sizes. Operations counted: structural equality comparisons.
def tableSlotSetShapeCheck (declaration : FunctionDecl) : Bool :=
  declaration.parameters == ["base", "slot", "key", "value"] &&
  declaration.body ==
    [.store slotAddressExpr (.name "key"),
     .store slotValueAddressExpr (.name "value"),
     .return_ (.name "value")]

-- Temporal O(1): fixed 3-string and 3-statement literals bound the `List String` and `List Stmt`
-- `BEq` comparisons (≤ 3 each, each `Stmt` comparison bounded by the literal's fixed expression
-- tree). Spatial O(1): both `BEq` stacks are bounded by the literal sizes. Operations counted:
-- structural equality comparisons.
def tableInitGoShapeCheck (declaration : FunctionDecl) : Bool :=
  declaration.parameters == ["base", "cap", "i"] &&
  declaration.body ==
    [.if_ (.binary .ge (.name "i") (.name "cap")) [.return_ (.literal 0)] none,
     .expression (.call "tableSlotSet"
       [.name "base", .name "i", .name "TABLE_EMPTY", .literal 0]),
     .return_ (.call "tableInitGo"
       [.name "base", .name "cap", .binary .add (.name "i") (.literal 1)])]

-- Temporal O(1): fixed 2-string and 2-statement literals bound the `List String` and `List Stmt`
-- `BEq` comparisons (≤ 2 each, each `Stmt` comparison bounded by the literal's fixed expression
-- tree). Spatial O(1): both `BEq` stacks are bounded by the literal sizes. Operations counted:
-- structural equality comparisons.
def tableInitShapeCheck (declaration : FunctionDecl) : Bool :=
  declaration.parameters == ["base", "cap"] &&
  declaration.body ==
    [.if_ (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0))
      [.return_ (.name "TABLE_EMPTY")] none,
     .return_ (.call "tableInitGo" [.name "base", .name "cap", .literal 0])]

-- Temporal O(1): a fixed 1-string literal and a fixed 2-statement literal bound the `List String`
-- and `List Stmt` `BEq` comparisons (≤ 1 and ≤ 2, each `Stmt` comparison bounded by the literal's
-- fixed expression tree). Spatial O(1): both `BEq` stacks are bounded by the literal sizes.
-- Operations counted: structural equality comparisons.
def tableCapacityOkShapeCheck (declaration : FunctionDecl) : Bool :=
  declaration.parameters == ["cap"] &&
  declaration.body ==
    [.if_ (.binary .le (.name "cap") (.literal 0)) [.return_ (.literal 0)] none,
     .return_ (.conditional (.literal 1)
       (.binary .eq
         (.binary .bitAnd (.name "cap")
           (.binary .sub (.name "cap") (.literal 1)))
         (.literal 0))
       (.literal 0))]

-- Temporal O(n + m + d + c) for n, m the table/std source byte lengths, d the decoded declaration
-- count and c the decoded constant count: `encodeBytes` and `Decode.decodeClosure` make two linear
-- passes, the four `functions.find?` scans are linear in d with one `String` name equality per
-- declaration, and the two `constantEnvironment` lookups scan `constants` linearly in c before one
-- `denoteExpr` on the matched literal. Spatial O(n + m) for the decoded character, token and
-- declaration lists. Operations counted: source bytes decoded, declarations and constants scanned,
-- `String`/`Word` comparisons.
def realConstructorShapes : Bool :=
  match Decode.decodeClosure
      (encodeBytes Slang.Generated.TableDerivation.tableSource)
      (encodeBytes Slang.Generated.TableDerivation.stdSource) with
  | .error _ => false
  | .ok value =>
      match value.val.program.functions.find? (fun declaration =>
          declaration.name == "tableSlotSet"),
        value.val.program.functions.find? (fun declaration =>
          declaration.name == "tableInitGo"),
        value.val.program.functions.find? (fun declaration =>
          declaration.name == "tableInit"),
        value.val.program.functions.find? (fun declaration =>
          declaration.name == "tableCapacityOk") with
      | some slotSet, some initGo, some init, some capacityOk =>
          tableSlotSetShapeCheck slotSet && tableInitGoShapeCheck initGo &&
            tableInitShapeCheck init && tableCapacityOkShapeCheck capacityOk &&
            constantEnvironment value.val.program "TABLE_STRIDE" == some (Word.ofNat 2) &&
            constantEnvironment value.val.program "TABLE_EMPTY" == some tableEmpty
      | _, _, _, _ => false

-- The compiled evaluator binds the checked shapes to the complete exported source. It adds no theorem.
#guard realConstructorShapes

def witnessBase : Word := Word.ofNat 0
def witnessCap : Word := Word.ofNat 1

def witnessBefore : Memory where
  owned := fun coordinate => coordinate < 2
  initialized := fun _ => none

def witnessAfter : Memory where
  owned := witnessBefore.owned
  initialized := fun coordinate =>
    if coordinate = 0 then some tableEmpty
    else if coordinate = 1 then some (Word.ofNat 0)
    else none

theorem witness_valid_backing : ValidBacking witnessBefore witnessBase witnessCap := by
  refine ⟨by decide, by decide, ?_⟩
  intro offset hoffset
  unfold witnessBefore
  change decide (witnessBase.val + offset < 2) = true
  rw [show witnessBase.val = 0 by rfl, Nat.zero_add]
  exact decide_eq_true hoffset

theorem witness_empty_layout : InitializedEmptyLayout witnessAfter witnessBase witnessCap := by
  constructor <;> intro slot hslot
  · have equal : slot = 0 := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ hslot)
    subst slot
    rfl
  · have equal : slot = 0 := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ hslot)
    subst slot
    rfl

theorem witness_repSong :
    RepSong witnessAfter witnessBase witnessCap (Word.ofNat 0) [] (fun _ => Word.ofNat 0) := by
  apply repSong_empty_of_layout
  · exact ⟨0, rfl⟩
  · constructor
    · exact witness_valid_backing
    · intro offset hoffset
      have hcases : offset = 0 ∨ offset = 1 := by
        cases offset with
        | zero => exact Or.inl rfl
        | succ offset =>
            cases offset with
            | zero => exact Or.inr rfl
            | succ offset =>
                exact False.elim (Nat.not_lt_zero offset
                  (Nat.lt_of_succ_lt_succ (Nat.lt_of_succ_lt_succ hoffset)))
      rcases hcases with rfl | rfl <;>
        rfl
  · exact witness_empty_layout.1

theorem joint_constructor_execution_witness (program : Program)
    (slotSet initGo init capacityOk : FunctionDecl)
    (hslotSet : (checkedContext program).functions "tableSlotSet" = some slotSet)
    (hinitGo : (checkedContext program).functions "tableInitGo" = some initGo)
    (hinit : (checkedContext program).functions "tableInit" = some init)
    (hcapacityOk : (checkedContext program).functions "tableCapacityOk" = some capacityOk)
    (hslotSetShape : TableSlotSetShape slotSet) (hinitGoShape : TableInitGoShape initGo)
    (hinitShape : TableInitShape init) (hcapacityOkShape : TableCapacityOkShape capacityOk)
    (hstride : (checkedContext program).constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : (checkedContext program).constants "TABLE_EMPTY" = some tableEmpty) :
    ∃ after trace,
      EvalCall (checkedContext program) .unbounded 0 "tableInit" [witnessBase, witnessCap]
        witnessBefore (.value (Word.ofNat 0) after) trace ∧
      RepSong after witnessBase witnessCap (Word.ofNat 0) [] (fun _ => Word.ofNat 0) ∧
      storeAddresses trace = expectedInitStores witnessBase witnessCap ∧
      (resources trace).stores = 2 * witnessCap.val ∧
      PreservesOutside witnessBefore after witnessBase witnessCap := by
  have hcap : PositivePowerOfTwo witnessCap :=
    ⟨0, rfl⟩
  rcases constructor_execution_of_shapes (checkedContext program) slotSet initGo init capacityOk hslotSet
    hinitGo hinit hcapacityOk hslotSetShape hinitGoShape hinitShape hcapacityOkShape hstride
    hempty witnessBefore witnessBase witnessCap hcap witness_valid_backing with
    ⟨after, trace, execution, rep⟩
  exact ⟨after, trace, execution.call, rep, execution.exactStores, execution.storeCount,
    execution.frame⟩

def KeyOnlyTableSlotSetShape (declaration : FunctionDecl) : Prop :=
  declaration.parameters = ["base", "slot", "key", "value"] ∧
    declaration.body =
      [.store slotAddressExpr (.name "key"), .return_ (.name "value")]

theorem key_only_shape_cannot_establish_constructor (declaration : FunctionDecl)
    (hkeyOnly : KeyOnlyTableSlotSetShape declaration) : ¬ TableSlotSetShape declaration := by
  intro hfull
  have hbody := hkeyOnly.2.symm.trans hfull.2
  have tails := (List.cons.inj hbody).2
  have statements := (List.cons.inj tails).1
  exact Stmt.noConfusion statements

theorem key_only_cannot_initialize_value (declaration : FunctionDecl)
    (hkeyOnly : KeyOnlyTableSlotSetShape declaration) : ¬ TableSlotSetShape declaration :=
  key_only_shape_cannot_establish_constructor declaration hkeyOnly

theorem single_store_cannot_initialize_other (memory after : Memory) (address key : Word)
    (keyCoordinate valueCoordinate : Nat)
    (hcoordinate : Memory.coordinate address = some keyCoordinate)
    (hstore : memory.store address key = some after)
    (hdifferent : valueCoordinate ≠ keyCoordinate)
    (huninitialized : memory.initialized valueCoordinate = none) :
    after.initialized valueCoordinate ≠ some (Word.ofNat 0) := by
  intro initialized
  have unchanged := Memory.store_other memory after address key keyCoordinate valueCoordinate
    hcoordinate hstore hdifferent
  have impossible : none = some (Word.ofNat 0) := huninitialized ▸ unchanged.symm.trans initialized
  cases impossible

theorem full_slot_stores_initialize (memory afterKey afterValue : Memory)
    (keyAddress valueAddress key value : Word) (keyCoordinate valueCoordinate : Nat)
    (hkeyCoordinate : Memory.coordinate keyAddress = some keyCoordinate)
    (hvalueCoordinate : Memory.coordinate valueAddress = some valueCoordinate)
    (hkeyStore : memory.store keyAddress key = some afterKey)
    (hvalueStore : afterKey.store valueAddress value = some afterValue) :
    afterValue.initialized valueCoordinate = some value ∧
    (keyCoordinate ≠ valueCoordinate → afterValue.initialized keyCoordinate = some key) := by
  constructor
  · exact Memory.store_at afterKey afterValue valueAddress value valueCoordinate
      hvalueCoordinate hvalueStore
  · intro hdifferent
    rw [Memory.store_other afterKey afterValue valueAddress value valueCoordinate keyCoordinate
      hvalueCoordinate hvalueStore hdifferent]
    exact Memory.store_at memory afterKey keyAddress key keyCoordinate hkeyCoordinate hkeyStore

#audit_empty Slang.Table.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Table.TableSlotSetShape : type_of% @Slang.Table.TableSlotSetShape
#audit_empty Slang.Table.TableInitGoShape : type_of% @Slang.Table.TableInitGoShape
#audit_empty Slang.Table.TableInitShape : type_of% @Slang.Table.TableInitShape
#audit_empty Slang.Table.TableCapacityOkShape : type_of% @Slang.Table.TableCapacityOkShape
#audit_empty Slang.Table.ConstructorExecution : type_of% @Slang.Table.ConstructorExecution
#audit_empty Slang.Table.tableSlotSet_execution : type_of% @Slang.Table.tableSlotSet_execution
#audit_empty Slang.Table.tableInitGo_execution : type_of% @Slang.Table.tableInitGo_execution
#audit_empty Slang.Table.tableCapacityOk_execution : type_of% @Slang.Table.tableCapacityOk_execution
#audit_empty Slang.Table.tableInit_execution : type_of% @Slang.Table.tableInit_execution
#audit_empty Slang.Table.constructor_execution_of_shapes : type_of% @Slang.Table.constructor_execution_of_shapes
#audit_empty Slang.Table.constructor_execution_of_certificate : type_of% @Slang.Table.constructor_execution_of_certificate
#audit_empty Slang.Table.witness_valid_backing : type_of% @Slang.Table.witness_valid_backing
#audit_empty Slang.Table.witness_empty_layout : type_of% @Slang.Table.witness_empty_layout
#audit_empty Slang.Table.witness_repSong : type_of% @Slang.Table.witness_repSong
#audit_empty Slang.Table.joint_constructor_execution_witness : type_of% @Slang.Table.joint_constructor_execution_witness
#audit_empty Slang.Table.key_only_shape_cannot_establish_constructor : type_of% @Slang.Table.key_only_shape_cannot_establish_constructor
#audit_empty Slang.Table.key_only_cannot_initialize_value : type_of% @Slang.Table.key_only_cannot_initialize_value
#audit_empty Slang.Table.single_store_cannot_initialize_other : type_of% @Slang.Table.single_store_cannot_initialize_other
#audit_empty Slang.Table.full_slot_stores_initialize : type_of% @Slang.Table.full_slot_stores_initialize

end Slang.Table
