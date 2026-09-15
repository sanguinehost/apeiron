import Slang.Table.TableResources
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Table

open Slang Slang.Decode Slang.Machine Slang.Source Slang.Refinement

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

/-- A one-slot table whose only key is `TABLE_EMPTY` and whose value cell is `0`: the concrete
initialized extent shared by the termination and resource witnesses. -/
def resourceWitnessBase : Word := Word.ofNat 0
def resourceWitnessCap : Word := Word.ofNat 1
def resourceWitnessKey : Word := Word.ofNat 7
def resourceWitnessMemory : Memory where
  owned := fun coordinate => coordinate < 2
  initialized := fun coordinate =>
    if coordinate = 0 then some tableEmpty
    else if coordinate = 1 then some (Word.ofNat 0)
    else none

theorem resourceWitness_valid_backing :
    ValidBacking resourceWitnessMemory resourceWitnessBase resourceWitnessCap := by
  refine ⟨by decide, by decide, ?_⟩
  intro offset hoffset
  change decide (0 + offset < 2) = true
  rw [Nat.zero_add]
  exact decide_eq_true hoffset

theorem resourceWitness_empty_layout :
    InitializedEmptyLayout resourceWitnessMemory resourceWitnessBase resourceWitnessCap := by
  constructor <;> intro slot hslot
  · have equal : slot = 0 := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ hslot)
    subst equal
    rfl
  · have equal : slot = 0 := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ hslot)
    subst equal
    rfl

theorem resourceWitness_repSong :
    RepSong resourceWitnessMemory resourceWitnessBase resourceWitnessCap (Word.ofNat 0) []
      (fun _ => Word.ofNat 0) := by
  apply repSong_empty_of_layout
  · exact ⟨0, rfl⟩
  · constructor
    · exact resourceWitness_valid_backing
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
      rcases hcases with rfl | rfl <;> rfl
  · exact resourceWitness_empty_layout.1

theorem resourceWitnessKey_ne_empty : resourceWitnessKey ≠ tableEmpty := by decide

/-- The concrete `tableInitGo` termination witness: at `base = 0`, `cap = 1`, `i = 0` the decoded
recursion has a finite derivation and the decoded `cap - i` resource bounds. -/
theorem resourceWitness_initGo (program : Program) (slotSet initGo : FunctionDecl)
    (hslotSet : (checkedContext program).functions "tableSlotSet" = some slotSet)
    (hinitGo : (checkedContext program).functions "tableInitGo" = some initGo)
    (hslotSetShape : TableSlotSetShape slotSet) (hinitGoShape : TableInitGoShape initGo)
    (hstride : (checkedContext program).constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : (checkedContext program).constants "TABLE_EMPTY" = some tableEmpty) :
    ∃ after trace,
      EvalCall (checkedContext program) .unbounded 0 "tableInitGo"
        [resourceWitnessBase, resourceWitnessCap, Word.ofNat 0] resourceWitnessMemory
        (.value (Word.ofNat 0) after) trace ∧
      InitGoResourceBounds resourceWitnessCap 0 0 trace ∧
      (resources trace).stores = 2 * resourceWitnessCap.val := by
  rcases tableInitGo_execution_resources (checkedContext program) slotSet initGo hslotSet hinitGo
    hslotSetShape hinitGoShape hstride hempty resourceWitnessMemory resourceWitnessBase
    resourceWitnessCap 0 0 (Nat.zero_le _) resourceWitness_valid_backing with
    ⟨after, trace, hcall, _, hstores, _, hbounds⟩
  exact ⟨after, trace, hcall, hbounds,
    initGo_stores_exact trace resourceWitnessBase resourceWitnessCap 0 hstores⟩

/-- The concrete probe termination and resource witness: at `base = 0`, `cap = 1`, `step = 0`
both decoded probes have a finite derivation with the decoded `cap - step` bounds. -/
theorem resourceWitness_probe (program : Program)
    (slotKey findProbe lookupProbe : FunctionDecl)
    (hslotKey : (checkedContext program).functions "tableSlotKey" = some slotKey)
    (hfind : (checkedContext program).functions "tableFindSlotGo" = some findProbe)
    (hlookup : (checkedContext program).functions "tableLookupSlotGo" = some lookupProbe)
    (hslotKeyShape : TableSlotKeyShape slotKey)
    (hfindShape : TableFindSlotGoShape findProbe)
    (hlookupShape : TableLookupSlotGoShape lookupProbe)
    (hstride : (checkedContext program).constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : (checkedContext program).constants "TABLE_EMPTY" = some tableEmpty) :
    (∃ raw trace,
        EvalCall (checkedContext program) .unbounded 0 "tableFindSlotGo"
          [resourceWitnessBase, resourceWitnessCap, Word.sub resourceWitnessCap (Word.ofNat 1),
            resourceWitnessKey,
            Word.bitAnd (Word.tableSpread resourceWitnessKey)
              (Word.sub resourceWitnessCap (Word.ofNat 1)), Word.ofNat 0]
          resourceWitnessMemory (.value raw resourceWitnessMemory) trace ∧
        ProbeResourceBounds resourceWitnessCap 0 0 trace) ∧
    (∃ raw trace,
        EvalCall (checkedContext program) .unbounded 0 "tableLookupSlotGo"
          [resourceWitnessBase, resourceWitnessCap, Word.sub resourceWitnessCap (Word.ofNat 1),
            resourceWitnessKey,
            Word.bitAnd (Word.tableSpread resourceWitnessKey)
              (Word.sub resourceWitnessCap (Word.ofNat 1)), Word.ofNat 0]
          resourceWitnessMemory (.value raw resourceWitnessMemory) trace ∧
        ProbeResourceBounds resourceWitnessCap 0 0 trace) := by
  have hfindBound := probe_execution_resources true (checkedContext program) slotKey findProbe
    hslotKey hfind hslotKeyShape hfindShape.1
    (by simpa only [probeName, probeEmptyExpr] using hfindShape.2) hstride hempty
    resourceWitnessMemory resourceWitnessBase resourceWitnessCap (Word.ofNat 0) resourceWitnessKey
    [] (fun _ => Word.ofNat 0) 0 0 resourceWitness_repSong (Nat.zero_le _)
  have hlookupBound := probe_execution_resources false (checkedContext program) slotKey lookupProbe
    hslotKey hlookup hslotKeyShape hlookupShape.1
    (by simpa only [probeName, probeEmptyExpr] using hlookupShape.2) hstride hempty
    resourceWitnessMemory resourceWitnessBase resourceWitnessCap (Word.ofNat 0) resourceWitnessKey
    [] (fun _ => Word.ofNat 0) 0 0 resourceWitness_repSong (Nat.zero_le _)
  rcases hfindBound with ⟨raw, trace, hcall, _, hbounds⟩
  rcases hlookupBound with ⟨raw', trace', hcall', _, hbounds'⟩
  exact ⟨⟨raw, trace, hcall, hbounds⟩, ⟨raw', trace', hcall', hbounds'⟩⟩

/-- The concrete initialization store count: `cap = 1` writes both cells, so two raw stores. -/
theorem resourceWitness_initGo_stores (program : Program) (slotSet initGo : FunctionDecl)
    (hslotSet : (checkedContext program).functions "tableSlotSet" = some slotSet)
    (hinitGo : (checkedContext program).functions "tableInitGo" = some initGo)
    (hslotSetShape : TableSlotSetShape slotSet) (hinitGoShape : TableInitGoShape initGo)
    (hstride : (checkedContext program).constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : (checkedContext program).constants "TABLE_EMPTY" = some tableEmpty) :
    ∃ trace, (resources trace).stores = 2 ∧ (resources trace).allocatedNodes = 0 ∧
      (resources trace).externalStreamConsumption = 0 := by
  rcases resourceWitness_initGo program slotSet initGo hslotSet hinitGo hslotSetShape
    hinitGoShape hstride hempty with ⟨after, trace, _, _, hstores⟩
  exact ⟨trace, hstores, resources_allocatedNodes_zero trace,
    resources_externalStreamConsumption_zero trace⟩

/-- The concrete probe iteration count: `cap = 1` gives at most one recursive edge, so the
`probeIterationCount` call-event projection is at most `cap + 1 = 2`. -/
theorem resourceWitness_probe_iterations (program : Program)
    (slotKey findProbe lookupProbe : FunctionDecl)
    (hslotKey : (checkedContext program).functions "tableSlotKey" = some slotKey)
    (hfind : (checkedContext program).functions "tableFindSlotGo" = some findProbe)
    (hlookup : (checkedContext program).functions "tableLookupSlotGo" = some lookupProbe)
    (hslotKeyShape : TableSlotKeyShape slotKey)
    (hfindShape : TableFindSlotGoShape findProbe)
    (hlookupShape : TableLookupSlotGoShape lookupProbe)
    (hstride : (checkedContext program).constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : (checkedContext program).constants "TABLE_EMPTY" = some tableEmpty) :
    (∃ trace, (resources trace).probeIterations ≤ 2 ∧ (resources trace).stores = 0) ∧
    (∃ trace, (resources trace).probeIterations ≤ 2 ∧ (resources trace).stores = 0) := by
  rcases resourceWitness_probe program slotKey findProbe lookupProbe hslotKey hfind hlookup
    hslotKeyShape hfindShape hlookupShape hstride hempty with ⟨⟨raw, trace, _, hbounds⟩,
      ⟨raw', trace', _, hbounds'⟩⟩
  exact ⟨⟨trace, probe_resources_iterations trace hbounds,
      probe_resources_stores trace hbounds⟩,
    ⟨trace', probe_resources_iterations trace' hbounds',
      probe_resources_stores trace' hbounds'⟩⟩

-- The registered zero-axiom population for the `resources` stage.
#audit_empty Slang.Table.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Table.initGoMeasure : type_of% @Slang.Table.initGoMeasure
#audit_empty Slang.Table.probeMeasure : type_of% @Slang.Table.probeMeasure
#audit_empty Slang.Table.initGoMeasure_decrease : type_of% @Slang.Table.initGoMeasure_decrease
#audit_empty Slang.Table.probeMeasure_decrease : type_of% @Slang.Table.probeMeasure_decrease
#audit_empty Slang.Table.initGo_recursive_argument : type_of% @Slang.Table.initGo_recursive_argument
#audit_empty Slang.Table.probe_recursive_argument : type_of% @Slang.Table.probe_recursive_argument
#audit_empty Slang.Table.initGo_guard_at_bound : type_of% @Slang.Table.initGo_guard_at_bound
#audit_empty Slang.Table.probe_guard_at_bound : type_of% @Slang.Table.probe_guard_at_bound
#audit_empty Slang.Table.tableInitGo_termination : type_of% @Slang.Table.tableInitGo_termination
#audit_empty Slang.Table.tableFindSlotGo_termination : type_of% @Slang.Table.tableFindSlotGo_termination
#audit_empty Slang.Table.tableLookupSlotGo_termination : type_of% @Slang.Table.tableLookupSlotGo_termination
#audit_empty Slang.Table.InitGoResourceBounds : type_of% @Slang.Table.InitGoResourceBounds
#audit_empty Slang.Table.ProbeResourceBounds : type_of% @Slang.Table.ProbeResourceBounds
#audit_empty Slang.Table.probe_resources_arithmetic : type_of% @Slang.Table.probe_resources_arithmetic
#audit_empty Slang.Table.probe_resources_comparisons : type_of% @Slang.Table.probe_resources_comparisons
#audit_empty Slang.Table.probe_resources_bitwise : type_of% @Slang.Table.probe_resources_bitwise
#audit_empty Slang.Table.probe_resources_loads : type_of% @Slang.Table.probe_resources_loads
#audit_empty Slang.Table.probe_resources_stores : type_of% @Slang.Table.probe_resources_stores
#audit_empty Slang.Table.probe_resources_calls : type_of% @Slang.Table.probe_resources_calls
#audit_empty Slang.Table.probe_resources_iterations : type_of% @Slang.Table.probe_resources_iterations
#audit_empty Slang.Table.probe_resources_maxCallStack : type_of% @Slang.Table.probe_resources_maxCallStack
#audit_empty Slang.Table.initGo_resources_arithmetic : type_of% @Slang.Table.initGo_resources_arithmetic
#audit_empty Slang.Table.initGo_resources_comparisons : type_of% @Slang.Table.initGo_resources_comparisons
#audit_empty Slang.Table.initGo_resources_bitwise : type_of% @Slang.Table.initGo_resources_bitwise
#audit_empty Slang.Table.initGo_resources_loads : type_of% @Slang.Table.initGo_resources_loads
#audit_empty Slang.Table.initGo_resources_stores : type_of% @Slang.Table.initGo_resources_stores
#audit_empty Slang.Table.initGo_resources_calls : type_of% @Slang.Table.initGo_resources_calls
#audit_empty Slang.Table.initGo_resources_iterations : type_of% @Slang.Table.initGo_resources_iterations
#audit_empty Slang.Table.initGo_resources_maxCallStack : type_of% @Slang.Table.initGo_resources_maxCallStack
#audit_empty Slang.Table.constructor_execution_stores : type_of% @Slang.Table.constructor_execution_stores
#audit_empty Slang.Table.initGo_stores_exact : type_of% @Slang.Table.initGo_stores_exact
#audit_empty Slang.Table.resources_allocatedNodes_zero : type_of% @Slang.Table.resources_allocatedNodes_zero
#audit_empty Slang.Table.resources_externalStreamConsumption_zero : type_of% @Slang.Table.resources_externalStreamConsumption_zero
#audit_empty Slang.Table.resourceWitness_repSong : type_of% @Slang.Table.resourceWitness_repSong
#audit_empty Slang.Table.tableInitGo_execution_resources : type_of% @Slang.Table.tableInitGo_execution_resources
#audit_empty Slang.Table.probe_execution_resources : type_of% @Slang.Table.probe_execution_resources
#audit_empty Slang.Table.tableSpread_execution_resources : type_of% @Slang.Table.tableSpread_execution_resources
#audit_empty Slang.Table.resourceWitness_initGo : type_of% @Slang.Table.resourceWitness_initGo
#audit_empty Slang.Table.resourceWitness_probe : type_of% @Slang.Table.resourceWitness_probe
#audit_empty Slang.Table.resourceWitness_initGo_stores : type_of% @Slang.Table.resourceWitness_initGo_stores
#audit_empty Slang.Table.resourceWitness_probe_iterations : type_of% @Slang.Table.resourceWitness_probe_iterations

end Slang.Table