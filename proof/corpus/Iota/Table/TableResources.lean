import Iota.Table.TableTermination

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Iota.Table

open Iota Iota.Decode Iota.Machine Iota.Source Iota.Refinement

-- ---------------------------------------------------------------------------
-- Trace shapes and their exact counts
-- ---------------------------------------------------------------------------

/-- The probe prefix down to the slot-empty test, including the entry call. -/
def probeBaseTrace (b : Bool) (depth : Nat) (addr : Word) : Trace :=
  [.call (probeName b) (depth + 1), .comparison .ge, .arithmetic .add, .bitwise .bitAnd,
   .call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add, .load addr,
   .comparison .eq]

/-- The exhausted probe: the guard returns `0 - 1`. -/
def probeExhaustedTrace (b : Bool) (depth : Nat) : Trace :=
  [.call (probeName b) (depth + 1), .comparison .ge, .arithmetic .sub]

/-- The lookup probe that meets an empty slot: the guard and empty test are taken. -/
def probeLookupEmptyTrace (depth : Nat) (addr : Word) : Trace :=
  probeBaseTrace false depth addr ++ [.arithmetic .sub]

/-- The matching probe: the empty test fails and the key test returns the slot. -/
def probeMatchTrace (b : Bool) (depth : Nat) (addr : Word) : Trace :=
  probeBaseTrace b depth addr ++ [.comparison .eq]

/-- The recursing probe: both tests fail and the cursor advances.

Temporal O(1): the appended head lists are fixed — `probeBaseTrace b depth addr` has 9 events and
the two singleton lists have 1 each — so the three `List.append` recursions traverse 11 cons cells
independent of `rest`; `rest` is the shared tail and is not traversed. Spatial O(|rest| + 11) for
the returned trace: 11 new cons cells plus the shared `rest`, i.e. O(1) allocation beyond `rest`.
Operations counted: list links traversed, cons cells allocated. -/
def probeStepTrace (b : Bool) (depth : Nat) (addr : Word) (rest : Trace) : Trace :=
  probeBaseTrace b depth addr ++ ([.comparison .eq] ++ ([.arithmetic .add] ++ rest))

theorem probeBaseTrace_arithmetic (b : Bool) (depth : Nat) (addr : Word) :
    arithmeticCount (probeBaseTrace b depth addr) = 3 := rfl

theorem probeBaseTrace_comparisons (b : Bool) (depth : Nat) (addr : Word) :
    comparisonCount (probeBaseTrace b depth addr) = 2 := rfl

theorem probeBaseTrace_bitwise (b : Bool) (depth : Nat) (addr : Word) :
    bitwiseCount (probeBaseTrace b depth addr) = 1 := rfl

theorem probeBaseTrace_loads (b : Bool) (depth : Nat) (addr : Word) :
    loadCount (probeBaseTrace b depth addr) = 1 := rfl

theorem probeBaseTrace_stores (b : Bool) (depth : Nat) (addr : Word) :
    storeCount (probeBaseTrace b depth addr) = 0 := rfl

theorem probeBaseTrace_calls (b : Bool) (depth : Nat) (addr : Word) :
    callCount (probeBaseTrace b depth addr) = 2 := rfl

theorem probeBaseTrace_iterations (b : Bool) (depth : Nat) (addr : Word) :
    probeIterationCount (probeBaseTrace b depth addr) = 1 := by
  cases b <;> rfl

theorem probeBaseTrace_stack (b : Bool) (depth : Nat) (addr : Word) :
    callStackMaximum (probeBaseTrace b depth addr) ≤ depth + 2 := by
  unfold probeBaseTrace
  change max (max 0 (depth + 2)) (depth + 1) ≤ depth + 2
  exact natMax_le (natMax_le (Nat.zero_le _) (Nat.le_refl _)) (Nat.le_succ _)

theorem probeExhaustedTrace_arithmetic (b : Bool) (depth : Nat) :
    arithmeticCount (probeExhaustedTrace b depth) = 1 := rfl

theorem probeExhaustedTrace_comparisons (b : Bool) (depth : Nat) :
    comparisonCount (probeExhaustedTrace b depth) = 1 := rfl

theorem probeExhaustedTrace_bitwise (b : Bool) (depth : Nat) :
    bitwiseCount (probeExhaustedTrace b depth) = 0 := rfl

theorem probeExhaustedTrace_loads (b : Bool) (depth : Nat) :
    loadCount (probeExhaustedTrace b depth) = 0 := rfl

theorem probeExhaustedTrace_stores (b : Bool) (depth : Nat) :
    storeCount (probeExhaustedTrace b depth) = 0 := rfl

theorem probeExhaustedTrace_calls (b : Bool) (depth : Nat) :
    callCount (probeExhaustedTrace b depth) = 1 := rfl

theorem probeExhaustedTrace_iterations (b : Bool) (depth : Nat) :
    probeIterationCount (probeExhaustedTrace b depth) = 1 := by
  cases b <;> rfl

theorem probeExhaustedTrace_stack (b : Bool) (depth : Nat) :
    callStackMaximum (probeExhaustedTrace b depth) ≤ depth + 1 := by
  unfold probeExhaustedTrace
  change max 0 (depth + 1) ≤ depth + 1
  exact natMax_le (Nat.zero_le _) (Nat.le_refl _)

theorem probeLookupEmptyTrace_arithmetic (depth : Nat) (addr : Word) :
    arithmeticCount (probeLookupEmptyTrace depth addr) = 4 := by
  rw [probeLookupEmptyTrace, arithmeticCount_append, probeBaseTrace_arithmetic]
  rfl

theorem probeLookupEmptyTrace_comparisons (depth : Nat) (addr : Word) :
    comparisonCount (probeLookupEmptyTrace depth addr) = 2 := by
  rw [probeLookupEmptyTrace, comparisonCount_append, probeBaseTrace_comparisons]
  rfl

theorem probeLookupEmptyTrace_bitwise (depth : Nat) (addr : Word) :
    bitwiseCount (probeLookupEmptyTrace depth addr) = 1 := by
  rw [probeLookupEmptyTrace, bitwiseCount_append, probeBaseTrace_bitwise]
  rfl

theorem probeLookupEmptyTrace_loads (depth : Nat) (addr : Word) :
    loadCount (probeLookupEmptyTrace depth addr) = 1 := by
  rw [probeLookupEmptyTrace, loadCount_append, probeBaseTrace_loads]
  rfl

theorem probeLookupEmptyTrace_stores (depth : Nat) (addr : Word) :
    storeCount (probeLookupEmptyTrace depth addr) = 0 := by
  rw [probeLookupEmptyTrace, storeCount_append, probeBaseTrace_stores]
  rfl

theorem probeLookupEmptyTrace_calls (depth : Nat) (addr : Word) :
    callCount (probeLookupEmptyTrace depth addr) = 2 := by
  rw [probeLookupEmptyTrace, callCount_append, probeBaseTrace_calls]
  rfl

theorem probeLookupEmptyTrace_iterations (depth : Nat) (addr : Word) :
    probeIterationCount (probeLookupEmptyTrace depth addr) = 1 := by
  rw [probeLookupEmptyTrace, probeIterationCount_append, probeBaseTrace_iterations]
  rfl

theorem probeLookupEmptyTrace_stack (depth : Nat) (addr : Word) :
    callStackMaximum (probeLookupEmptyTrace depth addr) ≤ depth + 2 := by
  rw [probeLookupEmptyTrace, callStackMaximum_append]
  exact natMax_le (probeBaseTrace_stack false depth addr) (Nat.zero_le _)

theorem probeMatchTrace_arithmetic (b : Bool) (depth : Nat) (addr : Word) :
    arithmeticCount (probeMatchTrace b depth addr) = 3 := by
  rw [probeMatchTrace, arithmeticCount_append, probeBaseTrace_arithmetic]
  rfl

theorem probeMatchTrace_comparisons (b : Bool) (depth : Nat) (addr : Word) :
    comparisonCount (probeMatchTrace b depth addr) = 3 := by
  rw [probeMatchTrace, comparisonCount_append, probeBaseTrace_comparisons]
  rfl

theorem probeMatchTrace_bitwise (b : Bool) (depth : Nat) (addr : Word) :
    bitwiseCount (probeMatchTrace b depth addr) = 1 := by
  rw [probeMatchTrace, bitwiseCount_append, probeBaseTrace_bitwise]
  rfl

theorem probeMatchTrace_loads (b : Bool) (depth : Nat) (addr : Word) :
    loadCount (probeMatchTrace b depth addr) = 1 := by
  rw [probeMatchTrace, loadCount_append, probeBaseTrace_loads]
  rfl

theorem probeMatchTrace_stores (b : Bool) (depth : Nat) (addr : Word) :
    storeCount (probeMatchTrace b depth addr) = 0 := by
  rw [probeMatchTrace, storeCount_append, probeBaseTrace_stores]
  rfl

theorem probeMatchTrace_calls (b : Bool) (depth : Nat) (addr : Word) :
    callCount (probeMatchTrace b depth addr) = 2 := by
  rw [probeMatchTrace, callCount_append, probeBaseTrace_calls]
  rfl

theorem probeMatchTrace_iterations (b : Bool) (depth : Nat) (addr : Word) :
    probeIterationCount (probeMatchTrace b depth addr) = 1 := by
  rw [probeMatchTrace, probeIterationCount_append, probeBaseTrace_iterations]
  rfl

theorem probeMatchTrace_stack (b : Bool) (depth : Nat) (addr : Word) :
    callStackMaximum (probeMatchTrace b depth addr) ≤ depth + 2 := by
  rw [probeMatchTrace, callStackMaximum_append]
  exact natMax_le (probeBaseTrace_stack b depth addr) (Nat.zero_le _)

theorem probeStepTrace_arithmetic (b : Bool) (depth : Nat) (addr : Word) (rest : Trace) :
    arithmeticCount (probeStepTrace b depth addr rest) = arithmeticCount rest + 4 := by
  rw [probeStepTrace, arithmeticCount_append, arithmeticCount_append, arithmeticCount_append,
    probeBaseTrace_arithmetic]
  change 3 + (0 + (1 + arithmeticCount rest)) = arithmeticCount rest + 4
  rw [Nat.zero_add, ← Nat.add_assoc 3 1, Nat.add_comm 4]

theorem probeStepTrace_comparisons (b : Bool) (depth : Nat) (addr : Word) (rest : Trace) :
    comparisonCount (probeStepTrace b depth addr rest) = comparisonCount rest + 3 := by
  rw [probeStepTrace, comparisonCount_append, comparisonCount_append, comparisonCount_append,
    probeBaseTrace_comparisons]
  change 2 + (1 + (0 + comparisonCount rest)) = comparisonCount rest + 3
  rw [Nat.zero_add, ← Nat.add_assoc 2 1, Nat.add_comm 3]

theorem probeStepTrace_bitwise (b : Bool) (depth : Nat) (addr : Word) (rest : Trace) :
    bitwiseCount (probeStepTrace b depth addr rest) = bitwiseCount rest + 1 := by
  rw [probeStepTrace, bitwiseCount_append, bitwiseCount_append, bitwiseCount_append,
    probeBaseTrace_bitwise]
  change 1 + (0 + (0 + bitwiseCount rest)) = bitwiseCount rest + 1
  rw [Nat.zero_add, Nat.zero_add, Nat.add_comm 1]

theorem probeStepTrace_loads (b : Bool) (depth : Nat) (addr : Word) (rest : Trace) :
    loadCount (probeStepTrace b depth addr rest) = loadCount rest + 1 := by
  rw [probeStepTrace, loadCount_append, loadCount_append, loadCount_append,
    probeBaseTrace_loads]
  change 1 + (0 + (0 + loadCount rest)) = loadCount rest + 1
  rw [Nat.zero_add, Nat.zero_add, Nat.add_comm 1]

theorem probeStepTrace_stores (b : Bool) (depth : Nat) (addr : Word) (rest : Trace) :
    storeCount (probeStepTrace b depth addr rest) = storeCount rest := by
  rw [probeStepTrace, storeCount_append, storeCount_append, storeCount_append,
    probeBaseTrace_stores]
  change 0 + (0 + (0 + storeCount rest)) = storeCount rest
  rw [Nat.zero_add, Nat.zero_add, Nat.zero_add]

theorem probeStepTrace_calls (b : Bool) (depth : Nat) (addr : Word) (rest : Trace) :
    callCount (probeStepTrace b depth addr rest) = callCount rest + 2 := by
  rw [probeStepTrace, callCount_append, callCount_append, callCount_append,
    probeBaseTrace_calls]
  change 2 + (0 + (0 + callCount rest)) = callCount rest + 2
  rw [Nat.zero_add, Nat.zero_add, Nat.add_comm 2]

theorem probeStepTrace_iterations (b : Bool) (depth : Nat) (addr : Word) (rest : Trace) :
    probeIterationCount (probeStepTrace b depth addr rest) = probeIterationCount rest + 1 := by
  rw [probeStepTrace, probeIterationCount_append, probeIterationCount_append,
    probeIterationCount_append, probeBaseTrace_iterations]
  change 1 + (0 + (0 + probeIterationCount rest)) = probeIterationCount rest + 1
  rw [Nat.zero_add, Nat.zero_add, Nat.add_comm 1]

theorem probeStepTrace_stack (b : Bool) (depth : Nat) (addr : Word) (rest : Trace) :
    callStackMaximum (probeStepTrace b depth addr rest)
      ≤ max (callStackMaximum rest) (depth + 2) := by
  rw [probeStepTrace, callStackMaximum_append, callStackMaximum_append, callStackMaximum_append]
  apply natMax_le
  · exact Nat.le_trans (probeBaseTrace_stack b depth addr) (natLe_max_right _ _)
  · apply natMax_le
    · exact Nat.zero_le _
    · apply natMax_le
      · exact Nat.zero_le _
      · exact natLe_max_left _ _

-- ---------------------------------------------------------------------------
-- Resource bounds
-- ---------------------------------------------------------------------------

def probeArithmeticBound (cap : Word) (step : Nat) : Nat := 4 * (cap.val - step + 1)
def probeComparisonBound (cap : Word) (step : Nat) : Nat := 3 * (cap.val - step + 1)
def probeBitwiseBound (cap : Word) (step : Nat) : Nat := cap.val - step + 1
def probeLoadBound (cap : Word) (step : Nat) : Nat := cap.val - step + 1
def probeCallBound (cap : Word) (step : Nat) : Nat := 2 * (cap.val - step + 1)
def probeIterationBound (cap : Word) (step : Nat) : Nat := cap.val - step + 1
def probeStackBound (depth : Nat) (cap : Word) (step : Nat) : Nat := depth + (cap.val - step) + 1

def ProbeResourceBounds (cap : Word) (step depth : Nat) (trace : Trace) : Prop :=
  arithmeticCount trace ≤ probeArithmeticBound cap step ∧
  comparisonCount trace ≤ probeComparisonBound cap step ∧
  bitwiseCount trace ≤ probeBitwiseBound cap step ∧
  loadCount trace ≤ probeLoadBound cap step ∧
  storeCount trace = 0 ∧
  callCount trace ≤ probeCallBound cap step ∧
  probeIterationCount trace ≤ probeIterationBound cap step ∧
  callStackMaximum trace ≤ probeStackBound depth cap step

theorem probeArithmeticBound_step (cap : Word) (step : Nat) (bound : step < cap.val) :
    probeArithmeticBound cap (step + 1) + 4 ≤ probeArithmeticBound cap step := by
  unfold probeArithmeticBound
  rw [← sub_succ_difference bound, Nat.mul_succ]

theorem probeComparisonBound_step (cap : Word) (step : Nat) (bound : step < cap.val) :
    probeComparisonBound cap (step + 1) + 3 ≤ probeComparisonBound cap step := by
  unfold probeComparisonBound
  rw [← sub_succ_difference bound, Nat.mul_succ]

theorem probeBitwiseBound_step (cap : Word) (step : Nat) (bound : step < cap.val) :
    probeBitwiseBound cap (step + 1) + 1 ≤ probeBitwiseBound cap step := by
  unfold probeBitwiseBound
  rw [← sub_succ_difference bound]

theorem probeLoadBound_step (cap : Word) (step : Nat) (bound : step < cap.val) :
    probeLoadBound cap (step + 1) + 1 ≤ probeLoadBound cap step := by
  unfold probeLoadBound
  rw [← sub_succ_difference bound]

theorem probeCallBound_step (cap : Word) (step : Nat) (bound : step < cap.val) :
    probeCallBound cap (step + 1) + 2 ≤ probeCallBound cap step := by
  unfold probeCallBound
  rw [← sub_succ_difference bound, Nat.mul_succ]

theorem probeIterationBound_step (cap : Word) (step : Nat) (bound : step < cap.val) :
    probeIterationBound cap (step + 1) + 1 ≤ probeIterationBound cap step := by
  unfold probeIterationBound
  rw [← sub_succ_difference bound]

theorem probeStackBound_step (cap : Word) (step depth : Nat) (bound : step < cap.val) :
    probeStackBound (depth + 1) cap (step + 1) = probeStackBound depth cap step := by
  unfold probeStackBound
  rw [sub_succ_difference bound]
  rw [Nat.add_assoc (depth + 1) (cap.val - (step + 1)) 1,
      Nat.add_assoc depth 1 (cap.val - (step + 1) + 1),
      Nat.add_comm 1 (cap.val - (step + 1) + 1),
      ← Nat.add_assoc depth (cap.val - (step + 1) + 1) 1]

theorem probeStackBound_self (cap : Word) (step depth : Nat) (bound : step < cap.val) :
    depth + 2 ≤ probeStackBound depth cap step := by
  unfold probeStackBound
  have hpos : 1 ≤ cap.val - step := Nat.succ_le_of_lt (natSub_pos_of_lt bound)
  exact Nat.add_le_add_right (Nat.add_le_add_left hpos depth) 1

theorem numeralArithmeticBound (cap : Word) (step : Nat) {value : Nat} (h : value ≤ 4) :
    value ≤ probeArithmeticBound cap step := by
  unfold probeArithmeticBound
  exact Nat.le_trans h
    (Nat.le_trans (Nat.le_of_eq (Nat.mul_one 4).symm)
      (Nat.mul_le_mul_left 4 (Nat.le_add_left 1 (cap.val - step))))

theorem numeralComparisonBound (cap : Word) (step : Nat) {value : Nat} (h : value ≤ 3) :
    value ≤ probeComparisonBound cap step := by
  unfold probeComparisonBound
  exact Nat.le_trans h
    (Nat.le_trans (Nat.le_of_eq (Nat.mul_one 3).symm)
      (Nat.mul_le_mul_left 3 (Nat.le_add_left 1 (cap.val - step))))

theorem numeralBitwiseBound (cap : Word) (step : Nat) {value : Nat} (h : value ≤ 1) :
    value ≤ probeBitwiseBound cap step := by
  unfold probeBitwiseBound
  exact Nat.le_trans h (Nat.le_add_left 1 (cap.val - step))

theorem numeralLoadBound (cap : Word) (step : Nat) {value : Nat} (h : value ≤ 1) :
    value ≤ probeLoadBound cap step := by
  unfold probeLoadBound
  exact Nat.le_trans h (Nat.le_add_left 1 (cap.val - step))

theorem numeralCallBound (cap : Word) (step : Nat) {value : Nat} (h : value ≤ 2) :
    value ≤ probeCallBound cap step := by
  unfold probeCallBound
  exact Nat.le_trans h
    (Nat.le_trans (Nat.le_of_eq (Nat.mul_one 2).symm)
      (Nat.mul_le_mul_left 2 (Nat.le_add_left 1 (cap.val - step))))

theorem numeralIterationBound (cap : Word) (step : Nat) {value : Nat} (h : value ≤ 1) :
    value ≤ probeIterationBound cap step := by
  unfold probeIterationBound
  exact Nat.le_trans h (Nat.le_add_left 1 (cap.val - step))

theorem probeBaseTrace_bounds (b : Bool) (depth : Nat) (addr : Word)
    (cap : Word) (step : Nat) (bound : step < cap.val) :
    ProbeResourceBounds cap step depth (probeBaseTrace b depth addr) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [probeBaseTrace_arithmetic]
    exact numeralArithmeticBound cap step (by decide : 3 ≤ 4)
  · rw [probeBaseTrace_comparisons]
    exact numeralComparisonBound cap step (by decide : 2 ≤ 3)
  · rw [probeBaseTrace_bitwise]
    exact numeralBitwiseBound cap step (by decide : 1 ≤ 1)
  · rw [probeBaseTrace_loads]
    exact numeralLoadBound cap step (by decide : 1 ≤ 1)
  · rw [probeBaseTrace_stores]
  · rw [probeBaseTrace_calls]
    exact numeralCallBound cap step (by decide : 2 ≤ 2)
  · rw [probeBaseTrace_iterations]
    exact numeralIterationBound cap step (by decide : 1 ≤ 1)
  · exact Nat.le_trans (probeBaseTrace_stack b depth addr)
      (probeStackBound_self cap step depth bound)

theorem probeMatchTrace_bounds (b : Bool) (depth : Nat) (addr : Word)
    (cap : Word) (step : Nat) (bound : step < cap.val) :
    ProbeResourceBounds cap step depth (probeMatchTrace b depth addr) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [probeMatchTrace_arithmetic]
    exact numeralArithmeticBound cap step (by decide : 3 ≤ 4)
  · rw [probeMatchTrace_comparisons]
    exact numeralComparisonBound cap step (by decide : 3 ≤ 3)
  · rw [probeMatchTrace_bitwise]
    exact numeralBitwiseBound cap step (by decide : 1 ≤ 1)
  · rw [probeMatchTrace_loads]
    exact numeralLoadBound cap step (by decide : 1 ≤ 1)
  · rw [probeMatchTrace_stores]
  · rw [probeMatchTrace_calls]
    exact numeralCallBound cap step (by decide : 2 ≤ 2)
  · rw [probeMatchTrace_iterations]
    exact numeralIterationBound cap step (by decide : 1 ≤ 1)
  · exact Nat.le_trans (probeMatchTrace_stack b depth addr)
      (probeStackBound_self cap step depth bound)

theorem probeLookupEmptyTrace_bounds (depth : Nat) (addr : Word)
    (cap : Word) (step : Nat) (bound : step < cap.val) :
    ProbeResourceBounds cap step depth (probeLookupEmptyTrace depth addr) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [probeLookupEmptyTrace_arithmetic]
    exact numeralArithmeticBound cap step (by decide : 4 ≤ 4)
  · rw [probeLookupEmptyTrace_comparisons]
    exact numeralComparisonBound cap step (by decide : 2 ≤ 3)
  · rw [probeLookupEmptyTrace_bitwise]
    exact numeralBitwiseBound cap step (by decide : 1 ≤ 1)
  · rw [probeLookupEmptyTrace_loads]
    exact numeralLoadBound cap step (by decide : 1 ≤ 1)
  · rw [probeLookupEmptyTrace_stores]
  · rw [probeLookupEmptyTrace_calls]
    exact numeralCallBound cap step (by decide : 2 ≤ 2)
  · rw [probeLookupEmptyTrace_iterations]
    exact numeralIterationBound cap step (by decide : 1 ≤ 1)
  · exact Nat.le_trans (probeLookupEmptyTrace_stack depth addr)
      (probeStackBound_self cap step depth bound)

theorem probeExhaustedTrace_bounds (b : Bool) (depth : Nat)
    (cap : Word) (step : Nat) (done : step = cap.val) :
    ProbeResourceBounds cap step depth (probeExhaustedTrace b depth) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [probeExhaustedTrace_arithmetic]
    exact numeralArithmeticBound cap step (by decide : 1 ≤ 4)
  · rw [probeExhaustedTrace_comparisons]
    exact numeralComparisonBound cap step (by decide : 1 ≤ 3)
  · rw [probeExhaustedTrace_bitwise]
    exact numeralBitwiseBound cap step (by decide : 0 ≤ 1)
  · rw [probeExhaustedTrace_loads]
    exact numeralLoadBound cap step (by decide : 0 ≤ 1)
  · rw [probeExhaustedTrace_stores]
  · rw [probeExhaustedTrace_calls]
    exact numeralCallBound cap step (by decide : 1 ≤ 2)
  · rw [probeExhaustedTrace_iterations]
    exact numeralIterationBound cap step (by decide : 1 ≤ 1)
  · rw [done]
    unfold probeStackBound
    rw [Nat.sub_self, Nat.add_zero]
    exact Nat.le_refl _

theorem probeStepBounds (b : Bool) (base cap curr : Word) (step depth : Nat)
    (bound : step < cap.val) (rest : Trace)
    (hrest : ProbeResourceBounds cap (step + 1) (depth + 1) rest) :
    ProbeResourceBounds cap step depth
      (probeStepTrace b depth (Word.ofNat (keyCoordinate base curr.val)) rest) := by
  obtain ⟨harith, hcomparison, hbitwise, hload, hstore, hcall, hiteration, hstack⟩ := hrest
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [probeStepTrace_arithmetic]
    exact Nat.le_trans (Nat.add_le_add_right harith 4) (probeArithmeticBound_step cap step bound)
  · rw [probeStepTrace_comparisons]
    exact Nat.le_trans (Nat.add_le_add_right hcomparison 3)
      (probeComparisonBound_step cap step bound)
  · rw [probeStepTrace_bitwise]
    exact Nat.le_trans (Nat.add_le_add_right hbitwise 1) (probeBitwiseBound_step cap step bound)
  · rw [probeStepTrace_loads]
    exact Nat.le_trans (Nat.add_le_add_right hload 1) (probeLoadBound_step cap step bound)
  · rw [probeStepTrace_stores]
    exact hstore
  · rw [probeStepTrace_calls]
    exact Nat.le_trans (Nat.add_le_add_right hcall 2) (probeCallBound_step cap step bound)
  · rw [probeStepTrace_iterations]
    exact Nat.le_trans (Nat.add_le_add_right hiteration 1)
      (probeIterationBound_step cap step bound)
  · have hprefix : callStackMaximum rest ≤ probeStackBound depth cap step :=
      Nat.le_trans hstack (Nat.le_of_eq (probeStackBound_step cap step depth bound))
    have hself : depth + 2 ≤ probeStackBound depth cap step :=
      probeStackBound_self cap step depth bound
    exact Nat.le_trans
      (probeStepTrace_stack b depth (Word.ofNat (keyCoordinate base curr.val)) rest)
      (natMax_le hprefix hself)

theorem probe_execution_resources (emptyReturnsSlot : Bool) (context : Context)
    (slotKey probe : FunctionDecl)
    (hslotKey : context.functions "tableSlotKey" = some slotKey)
    (hprobe : context.functions (probeName emptyReturnsSlot) = some probe)
    (hslotKeyShape : TableSlotKeyShape slotKey)
    (hprobeParameters : probe.parameters =
      ["base", "cap", "mask", "key", "slot", "step"])
    (hprobeBody : probe.body =
      probeBody (probeName emptyReturnsSlot) (probeEmptyExpr emptyReturnsSlot))
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hemptyConstant : context.constants "TABLE_EMPTY" = some tableEmpty)
    (memory : Memory) (base cap default key : Word) (occupied : List Word)
    (h : Word → Word) (step depth : Nat)
    (rep : RepSong memory base cap default occupied h) (hstep : step ≤ cap.val) :
    ∃ raw trace,
      EvalCall context .unbounded depth (probeName emptyReturnsSlot)
        [base, cap, Word.sub cap (Word.ofNat 1), key,
          Word.bitAnd (Word.tableSpread key) (Word.sub cap (Word.ofNat 1)), Word.ofNat step]
        memory (.value raw memory) trace ∧
      ProbeOutcome emptyReturnsSlot memory base cap key step raw ∧
      ProbeResourceBounds cap step depth trace := by
  generalize measureEq : cap.val - step = measure
  revert step depth
  refine Nat.strong_induction_on measure ?_
  intro measure inductionHypothesis step depth hstep measureEq
  have hcap := rep.positivePowerOfTwo
  have hbacking := rep.validExtent.1
  have hcapSign : cap.val < wordSignBit := by
    exact cap_below_signBit_of_backing memory base cap hbacking
  have hstepSign : step < wordSignBit := lt_of_le_of_lt hstep hcapSign
  let mask := Word.sub cap (Word.ofNat 1)
  let home := Word.bitAnd (Word.tableSpread key) mask
  let stepWord := Word.ofNat step
  let environment : Environment :=
    [("base", base), ("cap", cap), ("mask", mask), ("key", key), ("slot", home),
      ("step", stepWord)]
  have hframe : Environment.ofParameters probe.parameters
      [base, cap, mask, key, home, stepWord] = some environment := by
    rw [hprobeParameters]
    rfl
  have hbase : Environment.lookup environment "base" = some base := by
    rfl
  have hcapLocal : Environment.lookup environment "cap" = some cap := by
    rfl
  have hmaskLocal : Environment.lookup environment "mask" = some mask := by
    rfl
  have hkeyLocal : Environment.lookup environment "key" = some key := by
    rfl
  have hhomeLocal : Environment.lookup environment "slot" = some home := by
    rfl
  have hstepLocal : Environment.lookup environment "step" = some stepWord := by
    rfl
  have hgeEval : EvalExpr context .unbounded (depth + 1) environment memory
      (.binary .ge (.name "step") (.name "cap"))
      (.value (Word.ge stepWord cap) memory) [.comparison .ge] := by
    exact EvalExpr.binary (operator := .ge) (EvalExpr.local hstepLocal)
      (EvalExpr.local hcapLocal)
  by_cases hdone : step = cap.val
  · have hcapWord : Word.ofNat step = cap := by
      apply Fin.ext
      rw [hdone]
      exact ofNat_val_emptyCone cap.val hcapSign
    have hgeOne : Word.ge stepWord cap = Word.ofNat 1 := by
      exact Eq.mp (congrArg (fun right =>
        Word.ge (Word.ofNat step) right = Word.ofNat 1) hcapWord)
        (wordGe_ofNat_self step hstepSign)
    have hnonzero : Word.nonzero (Word.ge stepWord cap) := by
      rw [hgeOne]
      exact wordOne_nonzero
    have hguard : ExecStmt context .unbounded (depth + 1) environment memory
        (.if_ (.binary .ge (.name "step") (.name "cap"))
          [.return_ (.binary .sub (.literal 0) (.literal 1))] none)
        (.returned tableEmpty memory) [.comparison .ge, .arithmetic .sub] := by
      apply ExecStmt.ifTrue hgeEval hnonzero
      apply ExecList.returned
      apply ExecStmt.return_
      rw [← tableEmpty_sub]
      exact EvalExpr.binary (operator := .sub) (EvalExpr.literal (literalValue := 0))
        (EvalExpr.literal (literalValue := 1))
    have hbody : ExecList context .unbounded (depth + 1) environment memory probe.body
        (.returned tableEmpty memory) [.comparison .ge, .arithmetic .sub] := by
      rw [hprobeBody, probeBody]
      exact ExecList.returned hguard
    refine ⟨tableEmpty, probeExhaustedTrace emptyReturnsSlot depth, ?_,
      .exhausted step hdone, ?_⟩
    · simpa only [probeExhaustedTrace, List.cons_append, listAppendAssoc, List.nil_append]
        using EvalCall.run (policy := .unbounded) (depth := depth)
          hprobe hframe True.intro hbody
    · exact probeExhaustedTrace_bounds emptyReturnsSlot depth cap step hdone
  · have hlt : step < cap.val := Nat.lt_of_le_of_ne hstep hdone
    have hgeZero : Word.ge stepWord cap = Word.ofNat 0 := by
      have capWord : Word.ofNat cap.val = cap := Fin.ext (ofNat_val_emptyCone _ hcapSign)
      rw [← capWord]
      exact wordGe_ofNat_lt step cap.val hstepSign hcapSign hlt
    have hguardFalse : ¬ Word.nonzero (Word.ge stepWord cap) := by
      rw [hgeZero]
      exact fun hzero => hzero rfl
    have hguard : ExecStmt context .unbounded (depth + 1) environment memory
        (.if_ (.binary .ge (.name "step") (.name "cap"))
          [.return_ (.binary .sub (.literal 0) (.literal 1))] none)
        (.next environment memory) [.comparison .ge] := by
      exact ExecStmt.ifFalse hgeEval hguardFalse ExecList.nil
    let currNat := probeSlot cap key step
    let curr := Word.ofNat currNat
    have hcurrLt : currNat < cap.val := Nat.mod_lt _ hcap.pos
    have hcurrSign : currNat < wordSignBit := lt_trans hcurrLt hcapSign
    have hcurrEq : Word.bitAnd (Word.add home stepWord) mask = curr := by
      apply Fin.ext
      rw [show curr.val = currNat by exact ofNat_val_emptyCone currNat hcurrSign]
      exact source_probe_slot_eq cap key step hcap hcapSign hlt
    have hcurrEval : EvalExpr context .unbounded (depth + 1) environment memory
        (.binary .bitAnd (.binary .add (.name "slot") (.name "step")) (.name "mask"))
        (.value curr memory) [.arithmetic .add, .bitwise .bitAnd] := by
      rw [← hcurrEq]
      exact EvalExpr.binary (operator := .bitAnd)
        (EvalExpr.binary (operator := .add) (EvalExpr.local hhomeLocal)
          (EvalExpr.local hstepLocal)) (EvalExpr.local hmaskLocal)
    let currEnvironment := Environment.bind environment "currSlot" curr
    have hcurrBind : ExecStmt context .unbounded (depth + 1) environment memory
        (.bind "currSlot"
          (.binary .bitAnd (.binary .add (.name "slot") (.name "step")) (.name "mask")))
        (.next currEnvironment memory) [.arithmetic .add, .bitwise .bitAnd] := by
      exact ExecStmt.bind hcurrEval
    have hcurrLocal : Environment.lookup currEnvironment "currSlot" = some curr := by
      exact Environment.lookup_bind_same _ _ _
    have hbaseCurr : Environment.lookup currEnvironment "base" = some base := by
      rw [Environment.lookup_bind_other _ "currSlot" "base" curr (by decide)]
      exact hbase
    rcases rep.2.2.2.1 currNat hcurrLt with hslotEmpty | ⟨slotValue, hmember, hslotValue⟩
    · let slotWord := tableEmpty
      have hslotCall := tableSlotKey_execution context slotKey hslotKey hslotKeyShape hstride
        memory base curr (depth + 1) slotWord
        (by
          rw [show curr.val = currNat by exact ofNat_val_emptyCone currNat hcurrSign]
          exact keyCoordinate_below_signBit_emptyCone base cap currNat hbacking.2.1 hcurrLt)
        (by
          rw [show curr.val = currNat by exact ofNat_val_emptyCone currNat hcurrSign]
          exact rep.owned_extent (2 * currNat)
            (double_lt_double hcurrLt))
        (by
          rw [show curr.val = currNat by exact ofNat_val_emptyCone currNat hcurrSign]
          exact hslotEmpty)
      let slotEnvironment := Environment.bind currEnvironment "slotKey" slotWord
      have hargs : EvalArgs context .unbounded (depth + 1) currEnvironment memory
          [.name "base", .name "currSlot"] (.value [base, curr] memory) [] := by
        exact EvalArgs.cons (EvalExpr.local hbaseCurr)
          (EvalArgs.cons (EvalExpr.local hcurrLocal) EvalArgs.nil)
      have hslotBind : ExecStmt context .unbounded (depth + 1) currEnvironment memory
          (.bind "slotKey" (.call "tableSlotKey" [.name "base", .name "currSlot"]))
          (.next slotEnvironment memory)
          [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
            .load (Word.ofNat (keyCoordinate base curr.val))] := by
        apply ExecStmt.bind
        exact EvalExpr.call hargs hslotCall
      have hslotLocal : Environment.lookup slotEnvironment "slotKey" = some tableEmpty := by
        exact Environment.lookup_bind_same _ _ _
      have hemptyLocal : Environment.lookup slotEnvironment "TABLE_EMPTY" = none := by
        rfl
      have heqOne : Word.eq tableEmpty tableEmpty = Word.ofNat 1 := by
        exact wordEq_self tableEmpty
      have heqEval : EvalExpr context .unbounded (depth + 1) slotEnvironment memory
          (.binary .eq (.name "slotKey") (.name "TABLE_EMPTY"))
          (.value (Word.ofNat 1) memory) [.comparison .eq] := by
        rw [← heqOne]
        exact EvalExpr.binary (operator := .eq) (EvalExpr.local hslotLocal)
          (EvalExpr.constant hemptyLocal hemptyConstant)
      have hone : Word.nonzero (Word.ofNat 1) := by
        exact wordOne_nonzero
      cases emptyReturnsSlot with
      | false =>
          have hemptyIf : ExecStmt context .unbounded (depth + 1) slotEnvironment memory
              (.if_ (.binary .eq (.name "slotKey") (.name "TABLE_EMPTY"))
                [.return_ (.binary .sub (.literal 0) (.literal 1))] none)
              (.returned tableEmpty memory) [.comparison .eq, .arithmetic .sub] := by
            apply ExecStmt.ifTrue heqEval hone
            apply ExecList.returned
            apply ExecStmt.return_
            rw [← tableEmpty_sub]
            exact EvalExpr.binary (operator := .sub)
              (EvalExpr.literal (literalValue := 0))
              (EvalExpr.literal (literalValue := 1))
          have hbody : ExecList context .unbounded (depth + 1) environment memory probe.body
              (.returned tableEmpty memory)
              ([.comparison .ge] ++ [.arithmetic .add, .bitwise .bitAnd] ++
                [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
                  .load (Word.ofNat (keyCoordinate base curr.val))] ++
                [.comparison .eq, .arithmetic .sub]) := by
            rw [hprobeBody]
            simp only [probeName, probeEmptyExpr, probeBody]
            simpa only [listAppendAssoc] using ExecList.next hguard
              (ExecList.next hcurrBind (ExecList.next hslotBind (ExecList.returned hemptyIf)))
          refine ⟨tableEmpty,
            probeLookupEmptyTrace depth (Word.ofNat (keyCoordinate base curr.val)), ?_,
            ProbeOutcome.empty (emptyReturnsSlot := false) step hlt hslotEmpty, ?_⟩
          · simpa only [probeLookupEmptyTrace, probeBaseTrace, List.cons_append,
              listAppendAssoc, List.nil_append, mask, home, stepWord, probeName]
              using EvalCall.run (policy := .unbounded) (depth := depth)
                hprobe hframe True.intro hbody
          · exact probeLookupEmptyTrace_bounds depth
              (Word.ofNat (keyCoordinate base curr.val)) cap step hlt
      | true =>
          have hcurrSlotIn : Environment.lookup slotEnvironment "currSlot" = some curr := by
            rw [Environment.lookup_bind_other _ "slotKey" "currSlot" slotWord (by decide)]
            exact hcurrLocal
          have hemptyIf : ExecStmt context .unbounded (depth + 1) slotEnvironment memory
              (.if_ (.binary .eq (.name "slotKey") (.name "TABLE_EMPTY"))
                [.return_ (.name "currSlot")] none)
              (.returned curr memory) [.comparison .eq] := by
            apply ExecStmt.ifTrue heqEval hone
            exact ExecList.returned (ExecStmt.return_ (EvalExpr.local hcurrSlotIn))
          have hbody : ExecList context .unbounded (depth + 1) environment memory probe.body
              (.returned curr memory)
              ([.comparison .ge] ++ [.arithmetic .add, .bitwise .bitAnd] ++
                [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
                  .load (Word.ofNat (keyCoordinate base curr.val))] ++ [.comparison .eq]) := by
            rw [hprobeBody]
            simp only [probeName, probeEmptyExpr, probeBody]
            simpa only [listAppendAssoc] using ExecList.next hguard
              (ExecList.next hcurrBind (ExecList.next hslotBind (ExecList.returned hemptyIf)))
          refine ⟨curr, probeBaseTrace true depth (Word.ofNat (keyCoordinate base curr.val)),
            ?_, ProbeOutcome.empty (emptyReturnsSlot := true) step hlt hslotEmpty, ?_⟩
          · simpa only [probeBaseTrace, List.cons_append, listAppendAssoc, List.nil_append,
              mask, home, stepWord, probeName]
              using EvalCall.run (policy := .unbounded) (depth := depth)
                hprobe hframe True.intro hbody
          · exact probeBaseTrace_bounds true depth
              (Word.ofNat (keyCoordinate base curr.val)) cap step hlt
    · have hslotCall := tableSlotKey_execution context slotKey hslotKey hslotKeyShape hstride
        memory base curr (depth + 1) slotValue
        (by
          rw [show curr.val = currNat by exact ofNat_val_emptyCone currNat hcurrSign]
          exact keyCoordinate_below_signBit_emptyCone base cap currNat hbacking.2.1 hcurrLt)
        (by
          rw [show curr.val = currNat by exact ofNat_val_emptyCone currNat hcurrSign]
          exact rep.owned_extent (2 * currNat)
            (double_lt_double hcurrLt))
        (by
          rw [show curr.val = currNat by exact ofNat_val_emptyCone currNat hcurrSign]
          exact hslotValue)
      let slotEnvironment := Environment.bind currEnvironment "slotKey" slotValue
      have hargs : EvalArgs context .unbounded (depth + 1) currEnvironment memory
          [.name "base", .name "currSlot"] (.value [base, curr] memory) [] := by
        exact EvalArgs.cons (EvalExpr.local hbaseCurr)
          (EvalArgs.cons (EvalExpr.local hcurrLocal) EvalArgs.nil)
      have hslotBind : ExecStmt context .unbounded (depth + 1) currEnvironment memory
          (.bind "slotKey" (.call "tableSlotKey" [.name "base", .name "currSlot"]))
          (.next slotEnvironment memory)
          [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
            .load (Word.ofNat (keyCoordinate base curr.val))] := by
        apply ExecStmt.bind
        exact EvalExpr.call hargs hslotCall
      have hslotLocal : Environment.lookup slotEnvironment "slotKey" = some slotValue := by
        exact Environment.lookup_bind_same _ _ _
      have hkeyIn : Environment.lookup slotEnvironment "key" = some key := by
        rw [Environment.lookup_bind_other _ "slotKey" "key" slotValue (by decide)]
        rw [Environment.lookup_bind_other _ "currSlot" "key" curr (by decide)]
        exact hkeyLocal
      have hbaseIn : Environment.lookup slotEnvironment "base" = some base := by
        rw [Environment.lookup_bind_other _ "slotKey" "base" slotValue (by decide)]
        exact hbaseCurr
      have hcapIn : Environment.lookup slotEnvironment "cap" = some cap := by
        rw [Environment.lookup_bind_other _ "slotKey" "cap" slotValue (by decide)]
        rw [Environment.lookup_bind_other _ "currSlot" "cap" curr (by decide)]
        exact hcapLocal
      have hmaskIn : Environment.lookup slotEnvironment "mask" = some mask := by
        rw [Environment.lookup_bind_other _ "slotKey" "mask" slotValue (by decide)]
        rw [Environment.lookup_bind_other _ "currSlot" "mask" curr (by decide)]
        exact hmaskLocal
      have hslotIn : Environment.lookup slotEnvironment "slot" = some home := by
        rw [Environment.lookup_bind_other _ "slotKey" "slot" slotValue (by decide)]
        rw [Environment.lookup_bind_other _ "currSlot" "slot" curr (by decide)]
        exact hhomeLocal
      have hstepIn : Environment.lookup slotEnvironment "step" = some stepWord := by
        rw [Environment.lookup_bind_other _ "slotKey" "step" slotValue (by decide)]
        rw [Environment.lookup_bind_other _ "currSlot" "step" curr (by decide)]
        exact hstepLocal
      have hcurrIn : Environment.lookup slotEnvironment "currSlot" = some curr := by
        rw [Environment.lookup_bind_other _ "slotKey" "currSlot" slotValue (by decide)]
        exact hcurrLocal
      have hemptyLocal : Environment.lookup slotEnvironment "TABLE_EMPTY" = none := by
        rfl
      have hneEmpty : slotValue ≠ tableEmpty := rep.2.2.1.2 slotValue hmember
      have heqEmptyZero : Word.eq slotValue tableEmpty = Word.ofNat 0 := by
        exact wordEq_ne slotValue tableEmpty hneEmpty
      have heqEmptyEval : EvalExpr context .unbounded (depth + 1) slotEnvironment memory
          (.binary .eq (.name "slotKey") (.name "TABLE_EMPTY"))
          (.value (Word.ofNat 0) memory) [.comparison .eq] := by
        rw [← heqEmptyZero]
        exact EvalExpr.binary (operator := .eq) (EvalExpr.local hslotLocal)
          (EvalExpr.constant hemptyLocal hemptyConstant)
      have hzero : ¬ Word.nonzero (Word.ofNat 0) := fun hzero => hzero rfl
      have hemptyIf : ExecStmt context .unbounded (depth + 1) slotEnvironment memory
          (.if_ (.binary .eq (.name "slotKey") (.name "TABLE_EMPTY"))
            [.return_ (probeEmptyExpr emptyReturnsSlot)] none)
          (.next slotEnvironment memory) [.comparison .eq] := by
        exact ExecStmt.ifFalse heqEmptyEval hzero ExecList.nil
      by_cases hmatch : slotValue = key
      · subst slotValue
        have heqKeyOne : Word.eq key key = Word.ofNat 1 := wordEq_self key
        have heqKeyEval : EvalExpr context .unbounded (depth + 1) slotEnvironment memory
            (.binary .eq (.name "slotKey") (.name "key"))
            (.value (Word.ofNat 1) memory) [.comparison .eq] := by
          rw [← heqKeyOne]
          exact EvalExpr.binary (operator := .eq) (EvalExpr.local hslotLocal)
            (EvalExpr.local hkeyIn)
        have hone : Word.nonzero (Word.ofNat 1) := by
          exact wordOne_nonzero
        have hmatchIf : ExecStmt context .unbounded (depth + 1) slotEnvironment memory
            (.if_ (.binary .eq (.name "slotKey") (.name "key"))
              [.return_ (.name "currSlot")] none)
            (.returned curr memory) [.comparison .eq] := by
          apply ExecStmt.ifTrue heqKeyEval hone
          exact ExecList.returned (ExecStmt.return_ (EvalExpr.local hcurrIn))
        have hbody : ExecList context .unbounded (depth + 1) environment memory probe.body
            (.returned curr memory)
            ([.comparison .ge] ++ [.arithmetic .add, .bitwise .bitAnd] ++
              [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
                .load (Word.ofNat (keyCoordinate base curr.val))] ++
              [.comparison .eq] ++ [.comparison .eq]) := by
          rw [hprobeBody]
          unfold probeBody
          simpa only [listAppendAssoc] using ExecList.next hguard
            (ExecList.next hcurrBind (ExecList.next hslotBind
              (ExecList.next hemptyIf (ExecList.returned hmatchIf))))
        refine ⟨curr, probeMatchTrace emptyReturnsSlot depth
          (Word.ofNat (keyCoordinate base curr.val)), ?_, .match step hlt hslotValue, ?_⟩
        · simpa only [probeMatchTrace, probeBaseTrace, List.cons_append, listAppendAssoc,
            List.nil_append, mask, home, stepWord]
            using EvalCall.run (policy := .unbounded) (depth := depth)
              hprobe hframe True.intro hbody
        · exact probeMatchTrace_bounds emptyReturnsSlot depth
            (Word.ofNat (keyCoordinate base curr.val)) cap step hlt
      · have heqKeyZero : Word.eq slotValue key = Word.ofNat 0 := by
          exact wordEq_ne slotValue key hmatch
        have heqKeyEval : EvalExpr context .unbounded (depth + 1) slotEnvironment memory
            (.binary .eq (.name "slotKey") (.name "key"))
            (.value (Word.ofNat 0) memory) [.comparison .eq] := by
          rw [← heqKeyZero]
          exact EvalExpr.binary (operator := .eq) (EvalExpr.local hslotLocal)
            (EvalExpr.local hkeyIn)
        have hmatchIf : ExecStmt context .unbounded (depth + 1) slotEnvironment memory
            (.if_ (.binary .eq (.name "slotKey") (.name "key"))
              [.return_ (.name "currSlot")] none)
            (.next slotEnvironment memory) [.comparison .eq] := by
          exact ExecStmt.ifFalse heqKeyEval hzero ExecList.nil
        have measureDecrease : cap.val - (step + 1) < measure := by
          rw [← measureEq, sub_succ_difference hlt]
          exact Nat.lt_succ_self _
        rcases inductionHypothesis (cap.val - (step + 1)) measureDecrease
          (step + 1) (depth + 1) (Nat.succ_le_of_lt hlt) rfl with
          ⟨raw, recursiveTrace, hrecursive, houtcome, hresources⟩
        have hsucc : Word.add stepWord (Word.ofNat 1) = Word.ofNat (step + 1) := by
          apply Fin.ext
          unfold stepWord Word.add Word.ofNat
          change ((step % wordModulus + 1 % wordModulus) % wordModulus) =
            (step + 1) % wordModulus
          rw [modEqOfLt (by decide : 1 < wordModulus), modAddAbsorbLeft]
        have hrecursiveArgs : EvalArgs context .unbounded (depth + 1) slotEnvironment memory
            [.name "base", .name "cap", .name "mask", .name "key", .name "slot",
              .binary .add (.name "step") (.literal 1)]
            (.value [base, cap, mask, key, home, Word.ofNat (step + 1)] memory)
            [.arithmetic .add] := by
          apply EvalArgs.cons (EvalExpr.local hbaseIn)
          apply EvalArgs.cons (EvalExpr.local hcapIn)
          apply EvalArgs.cons (EvalExpr.local hmaskIn)
          apply EvalArgs.cons (EvalExpr.local hkeyIn)
          apply EvalArgs.cons (EvalExpr.local hslotIn)
          have hthird : EvalExpr context .unbounded (depth + 1) slotEnvironment memory
              (.binary .add (.name "step") (.literal 1))
              (.value (Word.ofNat (step + 1)) memory) [.arithmetic .add] := by
            rw [← hsucc]
            exact EvalExpr.binary (operator := .add) (EvalExpr.local hstepIn)
              (EvalExpr.literal (literalValue := 1))
          exact EvalArgs.cons hthird EvalArgs.nil
        have hreturn : ExecStmt context .unbounded (depth + 1) slotEnvironment memory
            (.return_ (.call (probeName emptyReturnsSlot)
              [.name "base", .name "cap", .name "mask", .name "key", .name "slot",
                .binary .add (.name "step") (.literal 1)]))
            (.returned raw memory) ([.arithmetic .add] ++ recursiveTrace) := by
          apply ExecStmt.return_
          simpa only [mask, home] using EvalExpr.call hrecursiveArgs hrecursive
        have hbody : ExecList context .unbounded (depth + 1) environment memory probe.body
            (.returned raw memory)
            ([.comparison .ge] ++ [.arithmetic .add, .bitwise .bitAnd] ++
              [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
                .load (Word.ofNat (keyCoordinate base curr.val))] ++
              [.comparison .eq] ++ [.comparison .eq] ++
              ([.arithmetic .add] ++ recursiveTrace)) := by
          rw [hprobeBody]
          unfold probeBody
          simpa only [listAppendAssoc] using ExecList.next hguard
            (ExecList.next hcurrBind (ExecList.next hslotBind
              (ExecList.next hemptyIf (ExecList.next hmatchIf (ExecList.returned hreturn)))))
        refine ⟨raw, probeStepTrace emptyReturnsSlot depth
          (Word.ofNat (keyCoordinate base curr.val)) recursiveTrace, ?_, ?_, ?_⟩
        · simpa only [probeStepTrace, probeBaseTrace, List.cons_append, listAppendAssoc,
            List.nil_append, mask, home, stepWord, probeName]
            using EvalCall.run (policy := .unbounded) (depth := depth)
              hprobe hframe True.intro hbody
        · exact ProbeOutcome.next step raw hlt
            (by
              intro heq
              rw [hslotValue] at heq
              exact hneEmpty (Option.some.inj heq))
            (by
              intro heq
              rw [hslotValue] at heq
              exact hmatch (Option.some.inj heq)) houtcome
        · exact probeStepBounds emptyReturnsSlot base cap curr step depth hlt recursiveTrace hresources

-- ---------------------------------------------------------------------------
-- `tableSpread`: the wrapping multiply is counted before the mask
-- ---------------------------------------------------------------------------

def spreadNegativeTrace (depth : Nat) : Trace :=
  [.call "tableSpread" (depth + 1), .comparison .lt, .arithmetic .sub,
   .arithmetic .mul, .bitwise .bitAnd]

def spreadNonnegativeTrace (depth : Nat) : Trace :=
  [.call "tableSpread" (depth + 1), .comparison .lt, .arithmetic .mul, .bitwise .bitAnd]

theorem spreadNegativeTrace_arithmetic (depth : Nat) :
    arithmeticCount (spreadNegativeTrace depth) = 2 := rfl

theorem spreadNegativeTrace_comparisons (depth : Nat) :
    comparisonCount (spreadNegativeTrace depth) = 1 := rfl

theorem spreadNegativeTrace_bitwise (depth : Nat) :
    bitwiseCount (spreadNegativeTrace depth) = 1 := rfl

theorem spreadNegativeTrace_loads (depth : Nat) :
    loadCount (spreadNegativeTrace depth) = 0 := rfl

theorem spreadNegativeTrace_stores (depth : Nat) :
    storeCount (spreadNegativeTrace depth) = 0 := rfl

theorem spreadNegativeTrace_calls (depth : Nat) :
    callCount (spreadNegativeTrace depth) = 1 := rfl

theorem spreadNegativeTrace_iterations (depth : Nat) :
    probeIterationCount (spreadNegativeTrace depth) = 0 := by
  unfold spreadNegativeTrace
  rfl

theorem spreadNonnegativeTrace_arithmetic (depth : Nat) :
    arithmeticCount (spreadNonnegativeTrace depth) = 1 := rfl

theorem spreadNonnegativeTrace_comparisons (depth : Nat) :
    comparisonCount (spreadNonnegativeTrace depth) = 1 := rfl

theorem spreadNonnegativeTrace_bitwise (depth : Nat) :
    bitwiseCount (spreadNonnegativeTrace depth) = 1 := rfl

theorem spreadNonnegativeTrace_loads (depth : Nat) :
    loadCount (spreadNonnegativeTrace depth) = 0 := rfl

theorem spreadNonnegativeTrace_stores (depth : Nat) :
    storeCount (spreadNonnegativeTrace depth) = 0 := rfl

theorem spreadNonnegativeTrace_calls (depth : Nat) :
    callCount (spreadNonnegativeTrace depth) = 1 := rfl

theorem spreadNonnegativeTrace_iterations (depth : Nat) :
    probeIterationCount (spreadNonnegativeTrace depth) = 0 := by
  unfold spreadNonnegativeTrace
  rfl

/-- `tableSpread` always evaluates its wrapping multiply and its mask: the arithmetic projection
counts at least one and at most two operations, and the multiply is the one that precedes the
mask. Dropping the multiply from the accounting makes this bound unprovable. -/
theorem tableSpread_execution_resources (context : Context) (spread : FunctionDecl)
    (hspread : context.functions "tableSpread" = some spread)
    (hshape : TableSpreadShape spread)
    (hphi : context.constants "TABLE_PHI" = some Word.tablePhi)
    (memory : Memory) (key : Word) (depth : Nat) :
    ∃ trace, EvalCall context .unbounded depth "tableSpread" [key] memory
        (.value (Word.tableSpread key) memory) trace ∧
      1 ≤ arithmeticCount trace ∧ arithmeticCount trace ≤ 2 ∧
      comparisonCount trace = 1 ∧ bitwiseCount trace = 1 ∧
      loadCount trace = 0 ∧ storeCount trace = 0 ∧
      callCount trace = 1 ∧ probeIterationCount trace = 0 := by
  let environment : Environment := [("k", key)]
  have hframe : Environment.ofParameters spread.parameters [key] = some environment := by
    rw [hshape.1]
    rfl
  have hkey : Environment.lookup environment "k" = some key := by
    rfl
  have hphiLocal : Environment.lookup environment "TABLE_PHI" = none := by
    rfl
  have hltEval : EvalExpr context .unbounded (depth + 1) environment memory
      (.binary .lt (.name "k") (.literal 0))
      (.value (Word.lt key (Word.ofNat 0)) memory) [.comparison .lt] := by
    exact EvalExpr.binary (operator := .lt) (EvalExpr.local hkey) EvalExpr.literal
  by_cases hnegative : Word.signed key < 0
  · have hltOne : Word.lt key (Word.ofNat 0) = Word.ofNat 1 := by
      apply wordLt_true
      rw [signed_ofNat_emptyCone 0 wordSignBit_pos]
      exact hnegative
    have hone : Word.nonzero (Word.ofNat 1) := by
      exact wordOne_nonzero
    have hposEval : EvalExpr context .unbounded (depth + 1) environment memory
        (.conditional (.binary .sub (.literal 0) (.name "k"))
          (.binary .lt (.name "k") (.literal 0)) (.name "k"))
        (.value (Word.spreadPosition key) memory) [.comparison .lt, .arithmetic .sub] := by
      rw [Word.spreadPosition, if_pos hnegative]
      have hltEvalOne := hltEval
      rw [hltOne] at hltEvalOne
      apply EvalExpr.conditionalTrue hltEvalOne hone
      exact EvalExpr.binary (operator := .sub) (EvalExpr.literal (literalValue := 0))
        (EvalExpr.local hkey)
    let nextEnvironment := Environment.bind environment "pos" (Word.spreadPosition key)
    have hbind : ExecStmt context .unbounded (depth + 1) environment memory
        (.bind "pos" (.conditional (.binary .sub (.literal 0) (.name "k"))
          (.binary .lt (.name "k") (.literal 0)) (.name "k")))
        (.next nextEnvironment memory) [.comparison .lt, .arithmetic .sub] :=
      ExecStmt.bind hposEval
    have hpos : Environment.lookup nextEnvironment "pos" = some (Word.spreadPosition key) :=
      Environment.lookup_bind_same _ _ _
    have hphiNext : Environment.lookup nextEnvironment "TABLE_PHI" = none := by
      rw [Environment.lookup_bind_other _ "pos" "TABLE_PHI" _ (by decide)]
      exact hphiLocal
    have hreturn : ExecStmt context .unbounded (depth + 1) nextEnvironment memory
        (.return_ (.binary .bitAnd
          (.binary .mul (.name "pos") (.name "TABLE_PHI")) (.literal 4611686018427387903)))
        (.returned (Word.tableSpread key) memory) [.arithmetic .mul, .bitwise .bitAnd] := by
      apply ExecStmt.return_
      exact EvalExpr.binary (operator := .bitAnd)
        (EvalExpr.binary (operator := .mul) (EvalExpr.local hpos)
          (EvalExpr.constant hphiNext hphi))
        (EvalExpr.literal (literalValue := 4611686018427387903))
    have hbody : ExecList context .unbounded (depth + 1) environment memory spread.body
        (.returned (Word.tableSpread key) memory)
        ([.comparison .lt, .arithmetic .sub] ++ [.arithmetic .mul, .bitwise .bitAnd]) := by
      rw [hshape.2]
      exact ExecList.next hbind (ExecList.returned hreturn)
    refine ⟨spreadNegativeTrace depth, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simpa only [spreadNegativeTrace, List.cons_append, listAppendAssoc, List.nil_append]
        using EvalCall.run (policy := .unbounded) (depth := depth)
          hspread hframe True.intro hbody
    · rw [spreadNegativeTrace_arithmetic]; exact Nat.le_succ 1
    · rw [spreadNegativeTrace_arithmetic]
    · exact spreadNegativeTrace_comparisons depth
    · exact spreadNegativeTrace_bitwise depth
    · exact spreadNegativeTrace_loads depth
    · exact spreadNegativeTrace_stores depth
    · exact spreadNegativeTrace_calls depth
    · exact spreadNegativeTrace_iterations depth
  · have hltZero : Word.lt key (Word.ofNat 0) = Word.ofNat 0 := by
      apply wordLt_false
      rw [signed_ofNat_emptyCone 0 wordSignBit_pos]
      exact hnegative
    have hzero : ¬ Word.nonzero (Word.ofNat 0) := fun hz => hz rfl
    have hposEval : EvalExpr context .unbounded (depth + 1) environment memory
        (.conditional (.binary .sub (.literal 0) (.name "k"))
          (.binary .lt (.name "k") (.literal 0)) (.name "k"))
        (.value (Word.spreadPosition key) memory) [.comparison .lt] := by
      rw [Word.spreadPosition, if_neg hnegative]
      have hltEvalZero := hltEval
      rw [hltZero] at hltEvalZero
      exact EvalExpr.conditionalFalse hltEvalZero hzero (EvalExpr.local hkey)
    let nextEnvironment := Environment.bind environment "pos" (Word.spreadPosition key)
    have hbind : ExecStmt context .unbounded (depth + 1) environment memory
        (.bind "pos" (.conditional (.binary .sub (.literal 0) (.name "k"))
          (.binary .lt (.name "k") (.literal 0)) (.name "k")))
        (.next nextEnvironment memory) [.comparison .lt] := ExecStmt.bind hposEval
    have hpos : Environment.lookup nextEnvironment "pos" = some (Word.spreadPosition key) :=
      Environment.lookup_bind_same _ _ _
    have hphiNext : Environment.lookup nextEnvironment "TABLE_PHI" = none := by
      rw [Environment.lookup_bind_other _ "pos" "TABLE_PHI" _ (by decide)]
      exact hphiLocal
    have hreturn : ExecStmt context .unbounded (depth + 1) nextEnvironment memory
        (.return_ (.binary .bitAnd
          (.binary .mul (.name "pos") (.name "TABLE_PHI")) (.literal 4611686018427387903)))
        (.returned (Word.tableSpread key) memory) [.arithmetic .mul, .bitwise .bitAnd] := by
      apply ExecStmt.return_
      exact EvalExpr.binary (operator := .bitAnd)
        (EvalExpr.binary (operator := .mul) (EvalExpr.local hpos)
          (EvalExpr.constant hphiNext hphi))
        (EvalExpr.literal (literalValue := 4611686018427387903))
    have hbody : ExecList context .unbounded (depth + 1) environment memory spread.body
        (.returned (Word.tableSpread key) memory)
        ([.comparison .lt] ++ [.arithmetic .mul, .bitwise .bitAnd]) := by
      rw [hshape.2]
      exact ExecList.next hbind (ExecList.returned hreturn)
    refine ⟨spreadNonnegativeTrace depth, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simpa only [spreadNonnegativeTrace, List.cons_append, listAppendAssoc, List.nil_append]
        using EvalCall.run (policy := .unbounded) (depth := depth)
          hspread hframe True.intro hbody
    · rw [spreadNonnegativeTrace_arithmetic]
    · rw [spreadNonnegativeTrace_arithmetic]; exact Nat.le_succ 1
    · exact spreadNonnegativeTrace_comparisons depth
    · exact spreadNonnegativeTrace_bitwise depth
    · exact spreadNonnegativeTrace_loads depth
    · exact spreadNonnegativeTrace_stores depth
    · exact spreadNonnegativeTrace_calls depth
    · exact spreadNonnegativeTrace_iterations depth

-- ---------------------------------------------------------------------------
-- Separate source-level projections
-- ---------------------------------------------------------------------------

theorem probe_resources_arithmetic (trace : Trace) {cap : Word} {step depth : Nat}
    (bounds : ProbeResourceBounds cap step depth trace) :
    (resources trace).arithmeticOps ≤ probeArithmeticBound cap step := by
  rw [resources_arithmeticOps]
  exact bounds.1

theorem probe_resources_comparisons (trace : Trace) {cap : Word} {step depth : Nat}
    (bounds : ProbeResourceBounds cap step depth trace) :
    (resources trace).comparisons ≤ probeComparisonBound cap step := by
  rw [resources_comparisons]
  exact bounds.2.1

theorem probe_resources_bitwise (trace : Trace) {cap : Word} {step depth : Nat}
    (bounds : ProbeResourceBounds cap step depth trace) :
    (resources trace).bitwiseOps ≤ probeBitwiseBound cap step := by
  rw [resources_bitwiseOps]
  exact bounds.2.2.1

theorem probe_resources_loads (trace : Trace) {cap : Word} {step depth : Nat}
    (bounds : ProbeResourceBounds cap step depth trace) :
    (resources trace).loads ≤ probeLoadBound cap step := by
  rw [resources_loads]
  exact bounds.2.2.2.1

theorem probe_resources_stores (trace : Trace) {cap : Word} {step depth : Nat}
    (bounds : ProbeResourceBounds cap step depth trace) :
    (resources trace).stores = 0 := by
  rw [resources_stores]
  exact bounds.2.2.2.2.1

theorem probe_resources_calls (trace : Trace) {cap : Word} {step depth : Nat}
    (bounds : ProbeResourceBounds cap step depth trace) :
    (resources trace).calls ≤ probeCallBound cap step := by
  rw [resources_calls]
  exact bounds.2.2.2.2.2.1

theorem probe_resources_iterations (trace : Trace) {cap : Word} {step depth : Nat}
    (bounds : ProbeResourceBounds cap step depth trace) :
    (resources trace).probeIterations ≤ probeIterationBound cap step := by
  rw [resources_probeIterations]
  exact bounds.2.2.2.2.2.2.1

theorem probe_resources_maxCallStack (trace : Trace) {cap : Word} {step depth : Nat}
    (bounds : ProbeResourceBounds cap step depth trace) :
    (resources trace).maxCallStack ≤ probeStackBound depth cap step := by
  rw [resources_maxCallStack]
  exact bounds.2.2.2.2.2.2.2

theorem initGo_resources_arithmetic (trace : Trace) {cap : Word} {i depth : Nat}
    (bounds : InitGoResourceBounds cap i depth trace) :
    (resources trace).arithmeticOps ≤ initGoArithmeticBound cap i := by
  rw [resources_arithmeticOps]
  exact bounds.1

theorem initGo_resources_comparisons (trace : Trace) {cap : Word} {i depth : Nat}
    (bounds : InitGoResourceBounds cap i depth trace) :
    (resources trace).comparisons ≤ initGoComparisonBound cap i := by
  rw [resources_comparisons]
  exact bounds.2.1

theorem initGo_resources_bitwise (trace : Trace) {cap : Word} {i depth : Nat}
    (bounds : InitGoResourceBounds cap i depth trace) :
    (resources trace).bitwiseOps ≤ initGoBitwiseBound cap i := by
  rw [resources_bitwiseOps]
  exact bounds.2.2.1

theorem initGo_resources_loads (trace : Trace) {cap : Word} {i depth : Nat}
    (bounds : InitGoResourceBounds cap i depth trace) :
    (resources trace).loads ≤ initGoLoadBound cap i := by
  rw [resources_loads]
  exact bounds.2.2.2.1

theorem initGo_resources_stores (trace : Trace) {cap : Word} {i depth : Nat}
    (bounds : InitGoResourceBounds cap i depth trace) :
    (resources trace).stores ≤ initGoStoreBound cap i := by
  rw [resources_stores]
  exact bounds.2.2.2.2.1

theorem initGo_resources_calls (trace : Trace) {cap : Word} {i depth : Nat}
    (bounds : InitGoResourceBounds cap i depth trace) :
    (resources trace).calls ≤ initGoCallBound cap i := by
  rw [resources_calls]
  exact bounds.2.2.2.2.2.1

theorem initGo_resources_iterations (trace : Trace) {cap : Word} {i depth : Nat}
    (bounds : InitGoResourceBounds cap i depth trace) :
    (resources trace).probeIterations ≤ initGoIterationBound cap i := by
  rw [resources_probeIterations]
  exact bounds.2.2.2.2.2.2.1

theorem initGo_resources_maxCallStack (trace : Trace) {cap : Word} {i depth : Nat}
    (bounds : InitGoResourceBounds cap i depth trace) :
    (resources trace).maxCallStack ≤ initGoStackBound depth cap i := by
  rw [resources_maxCallStack]
  exact bounds.2.2.2.2.2.2.2

/-- Initialization's exact raw store count, re-exported from the constructor execution. -/
theorem constructor_execution_stores (context : Context) (memory after : Memory)
    (base cap : Word) (trace : Trace) (execution : ConstructorExecution context memory after base cap trace) :
    (resources trace).stores = 2 * cap.val := execution.storeCount

theorem resources_allocatedNodes_zero (trace : Trace) :
    (resources trace).allocatedNodes = 0 := resources_allocatedNodes trace

theorem resources_externalStreamConsumption_zero (trace : Trace) :
    (resources trace).externalStreamConsumption = 0 :=
  resources_externalStreamConsumption trace

theorem expectedInitStoresFuel_length (base : Word) (slot fuel : Nat) :
    (expectedInitStoresFuel base slot fuel).length = 2 * fuel := by
  induction fuel generalizing slot with
  | zero => rfl
  | succ fuel ih =>
      unfold expectedInitStoresFuel
      change (expectedInitStoresFuel base (slot + 1) fuel).length + 2 = 2 * (fuel + 1)
      rw [ih, Nat.mul_succ]

theorem expectedInitStoresFrom_length (base cap : Word) (i : Nat) :
    (expectedInitStoresFrom base cap i).length = 2 * (cap.val - i) := by
  unfold expectedInitStoresFrom
  exact expectedInitStoresFuel_length base i (cap.val - i)

/-- `tableInitGo` performs exactly `2 * (cap - i)` raw stores, not merely at most. -/
theorem initGo_stores_exact (trace : Trace) (base cap : Word) (i : Nat)
    (stores : storeAddresses trace = expectedInitStoresFrom base cap i) :
    (resources trace).stores = 2 * (cap.val - i) := by
  rw [resources_stores, storeCount_eq_storeAddresses_length, stores, expectedInitStoresFrom_length]

end Iota.Table
